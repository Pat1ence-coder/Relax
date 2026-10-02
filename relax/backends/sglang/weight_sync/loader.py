# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded copies and full actual-storage verification while the engine is
shut."""

import hashlib
from dataclasses import dataclass
from typing import Protocol

from relax.distributed.weight_sync import ChunkSpec, DeltaCodecError, VerifiedSnapshot
from relax.distributed.weight_sync.consumer import Installation
from relax.distributed.weight_sync.load import LoadTile, iter_load_tiles
from relax.distributed.weight_sync.serialization import canonical_json

from .inventory import TargetInventory


class LoadLease(Protocol):
    """Keeps canonical data immutable and fences execution for this install."""

    def validate(self) -> None: ...


@dataclass(frozen=True)
class LoadedStorageReceipt:
    installation_id: str
    installation_digest: str
    rank: int
    incarnation: str
    binding_id: str
    nbytes: int
    tensors: int
    loaded_fragment_root: str


class PreparedLoad:
    """Preparation validates identities and allocations without modifying
    weights.

    The lease must also fence live execution before load/verify. It is held by
    the transaction, not released by this object. Copies are synchronous, so a
    returned failure never leaves this loader's asynchronous DMA outstanding.
    """

    def __init__(
        self, snapshot: VerifiedSnapshot, inventory: TargetInventory, installation: Installation, lease: LoadLease
    ):
        if not isinstance(snapshot, VerifiedSnapshot) or not isinstance(installation, Installation):
            raise DeltaCodecError("prepare requires a verified snapshot and installation")
        if (
            snapshot.identity != installation.snapshot
            or inventory.plan.plan_id != installation.plan_id
            or snapshot.schema.schema_id != inventory.plan.schema.schema_id
            or tuple(m.rank for m in installation.members) != inventory.plan.participants
        ):
            raise DeltaCodecError("prepared load identity/layout/membership mismatch")
        lease.validate()
        inventory.validate_bindings()
        self.snapshot, self.inventory, self.installation, self.lease = snapshot, inventory, installation, lease
        self._cache_key = None
        self._cache = b""
        self._sources = {entry.tensor.name: entry.tensor for entry in snapshot.schema.tensors}

    def _expected(self, tile: LoadTile) -> bytes:
        if tile.source is None:
            return bytes(tile.nbytes)
        tensor = self._sources[tile.source]
        chunk_bytes = self.snapshot.schema.chunk_bytes
        offset = tile.source_offset // chunk_bytes * chunk_bytes
        key = tile.source, offset
        if self._cache_key != key:
            self._cache = b""
            spec = ChunkSpec(self.snapshot.schema.schema_id, tensor, offset, min(chunk_bytes, tensor.nbytes - offset))
            data = self.snapshot.read_chunk(spec)
            if type(data) is not bytes or len(data) != spec.byte_length:
                raise DeltaCodecError("canonical reader returned an invalid chunk")
            self._cache, self._cache_key = data, key
        relative = tile.source_offset - offset
        data = self._cache[relative : relative + tile.nbytes]
        if len(data) != tile.nbytes:
            raise DeltaCodecError("load tile exceeds its canonical chunk")
        return data

    def _validate(self) -> None:
        self.lease.validate()
        self.inventory.validate_bindings()

    def _synchronize(self) -> None:
        import torch

        device = next(iter(self.inventory.tensors.values())).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def load(self) -> None:
        import torch

        self._validate()
        try:
            with torch.no_grad():
                for tile in iter_load_tiles(self.inventory.plan, self.inventory.rank):
                    self.lease.validate()
                    data = bytearray(self._expected(tile))
                    source = torch.frombuffer(data, dtype=torch.uint8)
                    target = self.inventory.tensors[tile.target].detach().view(-1).view(torch.uint8)
                    target.narrow(0, tile.target_offset, tile.nbytes).copy_(source, non_blocking=False)
            self._synchronize()
            self._validate()
        finally:
            self._cache_key, self._cache = None, b""

    def verify_loaded(self) -> LoadedStorageReceipt:
        import torch

        self._validate()
        self._synchronize()
        digest = hashlib.sha256()
        digest.update(
            canonical_json({"installation": self.installation.digest, "binding": self.inventory.binding_id}, 4096)
        )
        total = 0
        try:
            for tile in iter_load_tiles(self.inventory.plan, self.inventory.rank):
                self.lease.validate()
                tensor = self.inventory.tensors[tile.target].detach().view(-1).view(torch.uint8)
                actual = tensor.narrow(0, tile.target_offset, tile.nbytes).to(device="cpu").numpy().tobytes()
                if actual != self._expected(tile):
                    raise DeltaCodecError(f"actual target bytes differ: {tile.target}")
                digest.update(
                    canonical_json({"name": tile.target, "offset": tile.target_offset, "length": tile.nbytes}, 4096)
                )
                digest.update(actual)
                total += tile.nbytes
            self._validate()
        finally:
            self._cache_key, self._cache = None, b""
        rank = self.inventory.rank
        return LoadedStorageReceipt(
            self.installation.installation_id,
            self.installation.digest,
            rank,
            self.installation.members[rank].incarnation,
            self.inventory.binding_id,
            total,
            len(self.inventory.tensors),
            digest.hexdigest(),
        )
