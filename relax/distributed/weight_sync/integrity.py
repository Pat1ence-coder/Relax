# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Domain-separated streaming Merkle root for canonical model contents."""

import hashlib
import struct

from .limits import DeltaCodecError, require_uint
from .schema import ChunkSpec, require_digest


def domain_hash(tag: bytes, *fields: bytes) -> bytes:
    digest = hashlib.sha256(b"DWS1")
    for value in (tag, *fields):
        digest.update(struct.pack("<Q", len(value)))
        digest.update(value)
    return digest.digest()


class ModelRoot:
    """One hash per occupied tree level; no payload or leaf list is
    retained."""

    def __init__(self, schema_id: str, directory_hash: str) -> None:
        require_digest(schema_id, "schema_id")
        require_digest(directory_hash, "directory_hash")
        self.schema_id = schema_id
        self.directory_hash = directory_hash
        self.count = 0
        self._levels: list[bytes | None] = []

    def add(self, spec: ChunkSpec, content_hash: str) -> None:
        if spec.schema_id != self.schema_id:
            raise DeltaCodecError("chunk schema differs from root schema")
        require_digest(content_hash, "content_hash")
        require_uint(self.count + 1, "root chunk count")
        value = domain_hash(
            b"chunk-leaf",
            bytes.fromhex(spec.schema_id),
            spec.tensor.name.encode("utf-8"),
            struct.pack("<Q", spec.byte_offset),
            struct.pack("<Q", spec.byte_length),
            bytes.fromhex(content_hash),
        )
        level = 0
        while level < len(self._levels) and self._levels[level] is not None:
            value = domain_hash(b"merkle-node", self._levels[level], value)
            self._levels[level] = None
            level += 1
        if level == len(self._levels):
            self._levels.append(value)
        else:
            self._levels[level] = value
        self.count += 1

    def hexdigest(self) -> str:
        right = None
        height = 0
        for level, left in enumerate(self._levels):
            if left is None:
                continue
            if right is None:
                right, height = left, level
                continue
            while height < level:
                right = domain_hash(b"merkle-odd", right)
                height += 1
            right = domain_hash(b"merkle-node", left, right)
            height = level + 1
        tree = domain_hash(b"merkle-empty") if right is None else right
        return domain_hash(
            b"target-root",
            bytes.fromhex(self.schema_id),
            bytes.fromhex(self.directory_hash),
            struct.pack("<Q", self.count),
            tree,
        ).hex()
