# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Lightweight canonical chunk contracts, independent of model and layout.

The exporter supplies little-endian, C-order immutable bytes and a trusted
schema ID. These contracts do not normalize tensors, define aliases, acquire
snapshot leases, or compute the complete model schema/root.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from math import prod

from .limits import CodecLimits, DeltaCodecError, require_uint


_ELEMENT_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "uint64": 8,
    "int64": 8,
    "float64": 8,
}


def require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DeltaCodecError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name in (".", ".."):
            raise DeltaCodecError("tensor name must be nonempty")
        if any(char in self.name for char in ("/", "\\", "\x00")):
            raise DeltaCodecError("tensor name must not contain path separators or NUL")
        try:
            self.name.encode("utf-8")
        except UnicodeError as exc:
            raise DeltaCodecError("tensor name must be valid UTF-8") from exc
        if not isinstance(self.dtype, str) or self.dtype not in _ELEMENT_BYTES:
            raise DeltaCodecError("unsupported canonical dtype")
        if not isinstance(self.shape, tuple):
            raise DeltaCodecError("shape must be an immutable tuple")
        for dimension in self.shape:
            require_uint(dimension, "shape dimension")
        require_uint(self.nbytes, "tensor byte length")

    @property
    def element_size(self) -> int:
        return _ELEMENT_BYTES[self.dtype]

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * self.element_size

    def validate(self, limits: CodecLimits) -> None:
        if len(self.name.encode("utf-8")) > limits.max_tensor_name_bytes:
            raise DeltaCodecError("tensor name exceeds limit")
        if len(self.shape) > limits.max_tensor_rank:
            raise DeltaCodecError("tensor rank exceeds limit")


@dataclass(frozen=True)
class ChunkSpec:
    schema_id: str
    tensor: TensorSpec
    byte_offset: int
    byte_length: int

    def __post_init__(self) -> None:
        require_digest(self.schema_id, "schema_id")
        if not isinstance(self.tensor, TensorSpec):
            raise DeltaCodecError("chunk requires a TensorSpec")
        require_uint(self.byte_offset, "byte_offset")
        require_uint(self.byte_length, "byte_length")
        if self.byte_length == 0:
            raise DeltaCodecError("empty tensors have directory entries, not chunks")
        if self.byte_offset % self.tensor.element_size or self.byte_length % self.tensor.element_size:
            raise DeltaCodecError("chunk must be element-aligned")
        if self.byte_offset + self.byte_length > self.tensor.nbytes:
            raise DeltaCodecError("chunk exceeds tensor bounds")
        if self.element_count >= 1 << 32:
            raise DeltaCodecError("chunk element count must fit uint32 indices")

    @property
    def element_count(self) -> int:
        return self.byte_length // self.tensor.element_size

    def validate(self, limits: CodecLimits) -> None:
        self.tensor.validate(limits)
        if self.byte_length > limits.max_chunk_bytes:
            raise DeltaCodecError("decoded chunk exceeds limit")


@dataclass(frozen=True)
class CanonicalChunk:
    spec: ChunkSpec
    version: int
    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ChunkSpec):
            raise DeltaCodecError("canonical chunk requires a ChunkSpec")
        require_uint(self.version, "version")
        if type(self.data) is not bytes or len(self.data) != self.spec.byte_length:
            raise DeltaCodecError("chunk data must be immutable bytes of the declared length")


def iter_chunks(
    tensor: TensorSpec, schema_id: str, chunk_bytes: int, *, limits: CodecLimits = CodecLimits()
) -> Iterator[ChunkSpec]:
    """Yield fixed aligned intervals; zero-length tensors yield no chunks."""
    tensor.validate(limits)
    require_digest(schema_id, "schema_id")
    require_uint(chunk_bytes, "chunk_bytes", limits.max_chunk_bytes)
    if chunk_bytes < tensor.element_size or chunk_bytes % tensor.element_size:
        raise DeltaCodecError("chunk_bytes must be positive and element-aligned")
    for offset in range(0, tensor.nbytes, chunk_bytes):
        yield ChunkSpec(schema_id, tensor, offset, min(chunk_bytes, tensor.nbytes - offset))
