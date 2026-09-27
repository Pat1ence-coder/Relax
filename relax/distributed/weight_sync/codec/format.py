# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Versioned chunk envelope; not a model manifest or transport frame."""

import hashlib
import json
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..limits import CodecLimits, DeltaCodecError, require_uint
from ..schema import ChunkSpec, TensorSpec, require_digest


_HEADER = struct.Struct("<4sIQ")
_MAGIC = b"DWC1"


class Codec(str, Enum):
    COPY_BASE = "COPY_BASE"
    SPARSE_REPLACE_V1 = "SPARSE_REPLACE_V1"
    BITMAP_REPLACE_V1 = "BITMAP_REPLACE_V1"
    RAW_V1 = "RAW_V1"


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ChunkDescriptor:
    spec: ChunkSpec
    target_version: int
    codec: Codec
    replacement_count: int
    encoded_length: int
    payload_hash: str
    target_hash: str
    base_version: int | None = None
    base_hash: str | None = None
    format_version: int = 1

    def validate(self, limits: CodecLimits) -> None:
        if type(self.format_version) is not int or self.format_version != 1:
            raise DeltaCodecError("unsupported chunk format version")
        if not isinstance(self.spec, ChunkSpec) or not isinstance(self.codec, Codec):
            raise DeltaCodecError("invalid chunk spec or unknown codec")
        self.spec.validate(limits)
        require_uint(self.target_version, "target_version")
        require_uint(self.replacement_count, "replacement_count", self.spec.element_count)
        require_uint(self.encoded_length, "encoded_length", limits.max_chunk_bytes)
        require_digest(self.payload_hash, "payload_hash")
        require_digest(self.target_hash, "target_hash")
        count, size = self.spec.element_count, self.spec.tensor.element_size
        if self.codec == Codec.RAW_V1:
            if self.base_hash is not None or self.base_version is not None:
                raise DeltaCodecError("RAW must not depend on a base")
            if self.replacement_count != count:
                raise DeltaCodecError("RAW replacement count must equal element count")
            length = self.spec.byte_length
        else:
            require_uint(self.base_version, "base_version")
            require_digest(self.base_hash, "base_hash")
            if self.base_version >= self.target_version:
                raise DeltaCodecError("base version must precede target version")
            if self.codec == Codec.COPY_BASE:
                if self.replacement_count != 0 or self.base_hash != self.target_hash:
                    raise DeltaCodecError("COPY requires equal hashes and no replacements")
                length = 0
            elif self.codec == Codec.SPARSE_REPLACE_V1:
                length = self.replacement_count * (4 + size)
            else:
                length = (count + 7) // 8 + self.replacement_count * size
        if self.encoded_length != length:
            raise DeltaCodecError("encoded length does not match codec structure")

    def metadata_bytes(self, limits: CodecLimits = CodecLimits()) -> bytes:
        self.validate(limits)
        tensor = self.spec.tensor
        value = {
            "format_version": self.format_version,
            "chunk": {
                "schema_id": self.spec.schema_id,
                "tensor": {"name": tensor.name, "dtype": tensor.dtype, "shape": list(tensor.shape)},
                "byte_offset": self.spec.byte_offset,
                "byte_length": self.spec.byte_length,
            },
            "target_version": self.target_version,
            "base_version": self.base_version,
            "codec": self.codec.value,
            "replacement_count": self.replacement_count,
            "encoded_length": self.encoded_length,
            "payload_hash": self.payload_hash,
            "target_hash": self.target_hash,
            "base_hash": self.base_hash,
        }
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        if len(data) > limits.max_metadata_bytes:
            raise DeltaCodecError("chunk metadata exceeds limit")
        return data


def _exact_keys(value: Any, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise DeltaCodecError("unknown, missing, or invalid metadata fields")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DeltaCodecError("duplicate JSON key")
        result[key] = value
    return result


def _reject_number(value: str) -> None:
    raise DeltaCodecError("floating-point JSON numbers are not supported")


def _parse_int(value: str) -> int:
    if len(value) > 20:
        raise DeltaCodecError("JSON integer exceeds uint64 representation")
    return int(value)


def _parse_metadata(data: bytes, limits: CodecLimits) -> ChunkDescriptor:
    try:
        value = json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_float=_reject_number,
            parse_constant=_reject_number,
            parse_int=_parse_int,
        )
        _exact_keys(
            value,
            {
                "format_version",
                "chunk",
                "target_version",
                "base_version",
                "codec",
                "replacement_count",
                "encoded_length",
                "payload_hash",
                "target_hash",
                "base_hash",
            },
        )
        chunk = _exact_keys(value.pop("chunk"), {"schema_id", "tensor", "byte_offset", "byte_length"})
        tensor = _exact_keys(chunk.pop("tensor"), {"name", "dtype", "shape"})
        shape = tensor.pop("shape")
        if not isinstance(shape, list) or len(shape) > limits.max_tensor_rank:
            raise DeltaCodecError("invalid tensor shape or rank exceeds limit")
        spec = ChunkSpec(tensor=TensorSpec(shape=tuple(shape), **tensor), **chunk)
        value["codec"] = Codec(value["codec"])
        descriptor = ChunkDescriptor(spec=spec, **value)
        if descriptor.metadata_bytes(limits) != data:
            raise DeltaCodecError("chunk metadata must use canonical JSON serialization")
        return descriptor
    except DeltaCodecError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise DeltaCodecError("invalid chunk metadata") from exc


@dataclass(frozen=True)
class EncodedChunk:
    descriptor: ChunkDescriptor
    payload: bytes

    def validate(self, limits: CodecLimits = CodecLimits()) -> None:
        if not isinstance(self.descriptor, ChunkDescriptor):
            raise DeltaCodecError("encoded chunk requires a descriptor")
        self.descriptor.metadata_bytes(limits)
        if type(self.payload) is not bytes or len(self.payload) != self.descriptor.encoded_length:
            raise DeltaCodecError("payload length differs from descriptor")
        if content_hash(self.payload) != self.descriptor.payload_hash:
            raise DeltaCodecError("payload hash mismatch")

    def serialized_size(self, limits: CodecLimits = CodecLimits()) -> int:
        return _HEADER.size + len(self.descriptor.metadata_bytes(limits)) + self.descriptor.encoded_length

    def to_bytes(self, limits: CodecLimits = CodecLimits()) -> bytes:
        self.validate(limits)
        metadata = self.descriptor.metadata_bytes(limits)
        return _HEADER.pack(_MAGIC, len(metadata), len(self.payload)) + metadata + self.payload

    @classmethod
    def from_bytes(cls, data: bytes, limits: CodecLimits = CodecLimits()) -> "EncodedChunk":
        """Parse one bounded envelope, rejecting truncation and trailing data.

        Transport implementations must enforce their own limits before reading
        data. This parser bounds its own copies, not the caller's input buffer.
        """
        if type(data) is not bytes or len(data) < _HEADER.size:
            raise DeltaCodecError("truncated chunk header or non-bytes input")
        magic, metadata_length, payload_length = _HEADER.unpack_from(data)
        if magic != _MAGIC:
            raise DeltaCodecError("unknown chunk envelope magic/version")
        if metadata_length > limits.max_metadata_bytes or payload_length > limits.max_chunk_bytes:
            raise DeltaCodecError("chunk envelope exceeds limits")
        end = _HEADER.size + metadata_length
        if len(data) != end + payload_length:
            raise DeltaCodecError("truncated chunk envelope or trailing data")
        descriptor = _parse_metadata(data[_HEADER.size : end], limits)
        if descriptor.encoded_length != payload_length:
            raise DeltaCodecError("header payload length differs from descriptor")
        result = cls(descriptor, data[end:])
        result.validate(limits)
        return result
