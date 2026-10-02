# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Backend-independent source plans and immutable capture contracts.

A source lease must fence model writes and finish device copies. These CPU
contracts validate identities and coverage; they do not acquire that fence or
prove device readiness. Publication and base selection remain with the caller.
"""

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .codec.format import content_hash
from .limits import DeltaCodecError, SnapshotLimits, require_uint
from .manifest import SnapshotIdentity
from .model import ModelSchema
from .schema import ChunkSpec, require_digest
from .serialization import canonical_json, require_identifier


@dataclass(frozen=True)
class ExportRequest:
    stream_id: str
    run_epoch: str
    version: int
    source_step: int
    exporter_revision: str

    def __post_init__(self) -> None:
        for name in ("stream_id", "run_epoch", "exporter_revision"):
            require_identifier(getattr(self, name), name)
        require_uint(self.version, "export version")
        require_uint(self.source_step, "source step")

    @property
    def request_id(self) -> str:
        return content_hash(canonical_json(asdict(self), 16 * 1024))

    def identity(self, schema: ModelSchema, target_root: str) -> SnapshotIdentity:
        return SnapshotIdentity(self.stream_id, self.run_epoch, self.version, schema.schema_id, target_root)


@dataclass(frozen=True)
class ExportBudget:
    """Payload working-set limits; metadata uses SnapshotLimits separately.

    CPU capture verification conservatively reserves four canonical chunks plus
    one live source tile, including the previous verification chunk and reader
    copies. Backend adapters must also bound their simultaneous GPU, pinned and
    CPU buffers before allocation. This is not a process RSS cap.
    """

    tile_bytes: int = 1024 * 1024
    max_cpu_bytes: int = 64 * 1024 * 1024
    max_pinned_bytes: int = 16 * 1024 * 1024
    max_gpu_bytes: int = 64 * 1024 * 1024
    max_spans_per_tile: int = 16384

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            require_uint(getattr(self, name), name)
        if not self.tile_bytes or not self.max_cpu_bytes or not self.max_spans_per_tile:
            raise DeltaCodecError("tile, CPU and span budgets must be positive")

    def validate_schema(self, schema: ModelSchema) -> None:
        largest = max(
            (min(schema.chunk_bytes, entry.tensor.nbytes) for entry in schema.tensors if entry.alias_of is None),
            default=0,
        )
        for entry in schema.tensors:
            if entry.tensor.nbytes and self.tile_bytes < entry.tensor.element_size:
                raise DeltaCodecError("tile budget cannot hold one element")
        require_uint(4 * largest + min(self.tile_bytes, largest), "capture CPU working set", self.max_cpu_bytes)


@dataclass(frozen=True)
class TensorOwner:
    name: str
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise DeltaCodecError("owner requires a tensor name")
        require_uint(self.rank, "owner rank", (1 << 32) - 1)


@dataclass(frozen=True)
class SourceExportPlan:
    """Physical ownership, separate from the portable canonical schema.

    Rank IDs are relative to the explicitly supplied export process group.
    Every participant, including those with no owned payload, supplies a
    receipt. Aliases have no independent owner; empty owners still appear.
    """

    schema: ModelSchema
    source_layout_id: str
    participants: tuple[int, ...]
    owners: tuple[TensorOwner, ...]
    limits: SnapshotLimits = field(default=SnapshotLimits(), repr=False, compare=False)
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.schema, ModelSchema):
            raise DeltaCodecError("export plan requires a model schema")
        self.schema.validate(self.limits)
        require_digest(self.source_layout_id, "source layout ID")
        if not isinstance(self.participants, tuple) or not 0 < len(self.participants) <= 4096:
            raise DeltaCodecError("export participants must be a nonempty bounded tuple")
        for rank in self.participants:
            require_uint(rank, "participant rank", (1 << 32) - 1)
        if self.participants != tuple(sorted(set(self.participants))):
            raise DeltaCodecError("export participants must be sorted and unique")
        if not isinstance(self.owners, tuple) or len(self.owners) > self.limits.max_tensors:
            raise DeltaCodecError("export owners must be a bounded tuple")
        expected = {entry.tensor.name for entry in self.schema.tensors if entry.alias_of is None}
        seen = set()
        for owner in self.owners:
            if not isinstance(owner, TensorOwner) or owner.rank not in self.participants:
                raise DeltaCodecError("invalid export owner or nonparticipating rank")
            if owner.name not in expected or owner.name in seen:
                raise DeltaCodecError("export owner directory contains unknown or duplicate names")
            seen.add(owner.name)
        if seen != expected:
            raise DeltaCodecError("export owner directory is incomplete")
        ordered = tuple(sorted(self.owners, key=lambda owner: owner.name.encode("utf-8")))
        object.__setattr__(self, "owners", ordered)
        data = canonical_json(
            {
                "format_version": 1,
                "schema_id": self.schema.schema_id,
                "source_layout_id": self.source_layout_id,
                "participants": self.participants,
                "owners": [asdict(owner) for owner in ordered],
            },
            self.limits.max_directory_bytes,
        )
        object.__setattr__(self, "plan_id", content_hash(data))

    def expected_receipts(self, request: ExportRequest) -> tuple["SourceReceipt", ...]:
        counts = {rank: [0, 0] for rank in self.participants}
        entries = {entry.tensor.name: entry for entry in self.schema.tensors}
        for owner in self.owners:
            size = entries[owner.name].tensor.nbytes
            counts[owner.rank][0] += (size + self.schema.chunk_bytes - 1) // self.schema.chunk_bytes
            counts[owner.rank][1] += size
        return tuple(
            SourceReceipt(request.request_id, self.plan_id, rank, *counts[rank]) for rank in self.participants
        )


@dataclass(frozen=True)
class SourceReceipt:
    request_id: str
    plan_id: str
    rank: int
    chunks: int
    nbytes: int

    def __post_init__(self) -> None:
        require_digest(self.request_id, "receipt request ID")
        require_digest(self.plan_id, "receipt plan ID")
        for name in ("rank", "chunks", "nbytes"):
            require_uint(getattr(self, name), f"receipt {name}")


def validate_receipts(plan: SourceExportPlan, request: ExportRequest, receipts: Iterable[SourceReceipt]) -> None:
    expected = {receipt.rank: receipt for receipt in plan.expected_receipts(request)}
    for receipt in receipts:
        if not isinstance(receipt, SourceReceipt) or expected.pop(receipt.rank, None) != receipt:
            raise DeltaCodecError("export receipt identity, rank or coverage mismatch")
    if expected:
        raise DeltaCodecError("missing export participant receipt")


@dataclass(frozen=True)
class CanonicalTile:
    request_id: str
    plan_id: str
    rank: int
    spec: ChunkSpec
    byte_offset: int
    data: bytes

    def __post_init__(self) -> None:
        require_digest(self.request_id, "tile request ID")
        require_digest(self.plan_id, "tile plan ID")
        require_uint(self.rank, "tile owner rank")
        if not isinstance(self.spec, ChunkSpec):
            raise DeltaCodecError("canonical tile requires a chunk spec")
        require_uint(self.byte_offset, "tile offset within chunk")
        if type(self.data) is not bytes or not self.data:
            raise DeltaCodecError("tile data must be nonempty immutable bytes")
        if self.byte_offset % self.spec.tensor.element_size or len(self.data) % self.spec.tensor.element_size:
            raise DeltaCodecError("canonical tile must be element-aligned")
        if self.byte_offset + len(self.data) > self.spec.byte_length:
            raise DeltaCodecError("canonical tile exceeds its chunk")


class SnapshotLease(Protocol):
    """Caller-held source fence, valid until all reads and copies complete.

    An adapter must release on every path. Releasing a source lease must not
    invalidate a successfully captured, separately owned frozen snapshot.
    """

    request: ExportRequest

    def validate(self) -> None: ...

    def release(self) -> None: ...
