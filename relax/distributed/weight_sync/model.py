# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Complete canonical tensor directory, including buffers and whole aliases."""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .limits import DeltaCodecError, SnapshotLimits, require_uint
from .schema import ChunkSpec, TensorSpec, iter_chunks, require_digest
from .serialization import canonical_json, exact_fields, parse_json


@dataclass(frozen=True)
class TensorEntry:
    tensor: TensorSpec
    kind: str = "parameter"
    alias_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.tensor.name,
            "dtype": self.tensor.dtype,
            "shape": list(self.tensor.shape),
            "kind": self.kind,
            "alias_of": self.alias_of,
        }


@dataclass(frozen=True)
class ModelSchema:
    logical_config_hash: str
    converter_semantics_id: str
    tensors: tuple[TensorEntry, ...]
    chunk_bytes: int = 8 * 1024 * 1024
    limits: SnapshotLimits = field(default=SnapshotLimits(), repr=False, compare=False)
    schema_id: str = field(init=False)
    directory_hash: str = field(init=False)
    logical_nbytes: int = field(init=False)
    canonical_nbytes: int = field(init=False)
    chunk_count: int = field(init=False)

    def __post_init__(self) -> None:
        require_digest(self.logical_config_hash, "logical_config_hash")
        require_digest(self.converter_semantics_id, "converter_semantics_id")
        require_uint(self.chunk_bytes, "chunk_bytes", self.limits.codec.max_chunk_bytes)
        if self.chunk_bytes == 0:
            raise DeltaCodecError("chunk_bytes must be positive")
        if not isinstance(self.tensors, tuple) or len(self.tensors) > self.limits.max_tensors:
            raise DeltaCodecError("tensor directory must be an immutable tuple within count limit")
        names = {}
        logical = canonical = chunks = 0
        for entry in self.tensors:
            if not isinstance(entry, TensorEntry) or not isinstance(entry.tensor, TensorSpec):
                raise DeltaCodecError("invalid tensor directory entry")
            entry.tensor.validate(self.limits.codec)
            if entry.kind not in ("parameter", "buffer"):
                raise DeltaCodecError("unknown tensor kind")
            if self.chunk_bytes % entry.tensor.element_size:
                raise DeltaCodecError("schema chunk rule must align every dtype")
            if entry.tensor.name in names:
                raise DeltaCodecError("duplicate tensor name")
            names[entry.tensor.name] = entry
            logical += entry.tensor.nbytes
            if entry.alias_of is None:
                canonical += entry.tensor.nbytes
                chunks += (entry.tensor.nbytes + self.chunk_bytes - 1) // self.chunk_bytes
        require_uint(logical, "logical_nbytes", self.limits.max_model_bytes)
        require_uint(canonical, "canonical_nbytes", self.limits.max_model_bytes)
        require_uint(chunks, "chunk_count", self.limits.max_chunks)
        for entry in self.tensors:
            if entry.alias_of is None:
                continue
            if not isinstance(entry.alias_of, str) or entry.alias_of not in names:
                raise DeltaCodecError("alias owner is missing")
            owner = names[entry.alias_of]
            if owner is entry or owner.alias_of is not None:
                raise DeltaCodecError("alias must directly reference a non-alias owner")
            if (owner.tensor.dtype, owner.tensor.shape) != (entry.tensor.dtype, entry.tensor.shape):
                raise DeltaCodecError("only identical dtype/shape whole-tensor aliases are supported")
        ordered = tuple(sorted(self.tensors, key=lambda entry: entry.tensor.name.encode("utf-8")))
        object.__setattr__(self, "tensors", ordered)
        object.__setattr__(self, "logical_nbytes", logical)
        object.__setattr__(self, "canonical_nbytes", canonical)
        object.__setattr__(self, "chunk_count", chunks)
        directory = canonical_json([entry.to_dict() for entry in ordered], self.limits.max_directory_bytes)
        object.__setattr__(self, "directory_hash", hashlib.sha256(directory).hexdigest())
        object.__setattr__(self, "schema_id", hashlib.sha256(self.to_bytes()).hexdigest())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "logical_config_hash": self.logical_config_hash,
            "converter_semantics_id": self.converter_semantics_id,
            "byte_order": "little",
            "order": "C",
            "chunk_bytes": self.chunk_bytes,
            "tensors": [entry.to_dict() for entry in self.tensors],
        }

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_dict(), self.limits.max_directory_bytes)

    @classmethod
    def from_bytes(cls, data: bytes, limits: SnapshotLimits = SnapshotLimits()) -> "ModelSchema":
        value = exact_fields(
            parse_json(data, limits.max_directory_bytes),
            {
                "format_version",
                "logical_config_hash",
                "converter_semantics_id",
                "byte_order",
                "order",
                "chunk_bytes",
                "tensors",
            },
        )
        if type(value["format_version"]) is not int or value["format_version"] != 1:
            raise DeltaCodecError("unsupported schema format version")
        if value["byte_order"] != "little" or value["order"] != "C":
            raise DeltaCodecError("unsupported canonical byte order")
        records = value["tensors"]
        if not isinstance(records, list) or len(records) > limits.max_tensors:
            raise DeltaCodecError("tensor directory count exceeds limit")
        tensors = []
        for record in records:
            exact_fields(record, {"name", "dtype", "shape", "kind", "alias_of"})
            shape = record["shape"]
            if not isinstance(shape, list) or len(shape) > limits.codec.max_tensor_rank:
                raise DeltaCodecError("invalid shape or tensor rank exceeds limit")
            tensors.append(
                TensorEntry(
                    TensorSpec(record["name"], record["dtype"], tuple(shape)), record["kind"], record["alias_of"]
                )
            )
        result = cls(
            value["logical_config_hash"], value["converter_semantics_id"], tuple(tensors), value["chunk_bytes"], limits
        )
        if result.to_bytes() != data:
            raise DeltaCodecError("tensor directory must use canonical name ordering")
        return result

    def iter_chunks(self) -> Iterator[ChunkSpec]:
        for entry in self.tensors:
            if entry.alias_of is None:
                yield from iter_chunks(entry.tensor, self.schema_id, self.chunk_bytes, limits=self.limits.codec)

    def validate(self, limits: SnapshotLimits) -> None:
        # Revalidate against the receiver's budgets, not the producer's limits.
        ModelSchema.from_bytes(self.to_bytes(), limits)
