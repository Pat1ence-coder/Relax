# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Portable, bounded canonical-to-target copy plans.

Plans describe physical target bytes, including every replica and zero padding.
They do not grant permission to mutate a running model or certify device state.
Offsets are bytes; strided source rows always produce a contiguous target
range.
"""

import heapq
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

from .codec.format import content_hash
from .limits import DeltaCodecError, require_uint
from .model import ModelSchema
from .schema import TensorSpec, require_digest
from .serialization import canonical_json


@dataclass(frozen=True)
class LoadBudget:
    tile_bytes: int = 1024 * 1024
    max_cpu_bytes: int = 64 * 1024 * 1024
    max_pinned_bytes: int = 2 * 1024 * 1024
    max_gpu_bytes: int = 2 * 1024 * 1024
    max_regions: int = 65536

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            require_uint(getattr(self, name), name)
            if not getattr(self, name):
                raise DeltaCodecError("load budgets must be positive")

    def validate_schema(self, schema: ModelSchema) -> None:
        largest = max((min(t.tensor.nbytes, schema.chunk_bytes) for t in schema.tensors), default=0)
        if any(self.tile_bytes < t.tensor.element_size for t in schema.tensors):
            raise DeltaCodecError("load tile cannot hold one element")
        # Reader bytes, verification copy, tile materialization and readback.
        require_uint(2 * largest + 4 * self.tile_bytes, "load CPU working set", self.max_cpu_bytes)
        require_uint(self.tile_bytes, "load pinned tile", self.max_pinned_bytes)
        require_uint(self.tile_bytes, "load GPU tile", self.max_gpu_bytes)


@dataclass(frozen=True)
class TargetTensor:
    rank: int
    tensor: TensorSpec
    alias_of: str | None = None

    def __post_init__(self) -> None:
        require_uint(self.rank, "target rank", 4095)
        if not isinstance(self.tensor, TensorSpec):
            raise DeltaCodecError("target requires a TensorSpec")
        if self.alias_of is not None and (not isinstance(self.alias_of, str) or not self.alias_of):
            raise DeltaCodecError("invalid target alias")


@dataclass(frozen=True)
class LoadRegion:
    """A contiguous target extent sourced from equally spaced canonical rows.

    A None source denotes deterministic zero padding. Striding describes row
    parallel weights without storing an index or one descriptor per row.
    """

    rank: int
    target: str
    target_offset: int
    source: str | None
    source_offset: int
    row_bytes: int
    rows: int = 1
    source_stride: int = 0

    def __post_init__(self) -> None:
        for name in ("rank", "target_offset", "source_offset", "row_bytes", "rows", "source_stride"):
            require_uint(getattr(self, name), name)
        if not isinstance(self.target, str) or not self.target or not self.row_bytes or not self.rows:
            raise DeltaCodecError("load region requires a target and nonempty rows")
        if self.source is not None and (not isinstance(self.source, str) or not self.source):
            raise DeltaCodecError("invalid canonical source name")
        if self.rows > 1 and self.source_stride < self.row_bytes:
            raise DeltaCodecError("source rows must not overlap")
        if self.source is None and (self.source_offset or self.source_stride or self.rows != 1):
            raise DeltaCodecError("zero padding must be one plain contiguous region")

    @property
    def nbytes(self) -> int:
        return self.row_bytes * self.rows

    def source_intervals(self) -> Iterator[tuple[int, int]]:
        if self.rows == 1 or self.source_stride == self.row_bytes:
            yield self.source_offset, self.source_offset + self.nbytes
        else:
            for row in range(self.rows):
                start = self.source_offset + row * self.source_stride
                yield start, start + self.row_bytes


@dataclass(frozen=True)
class LoadPlan:
    schema: ModelSchema
    execution_profile_id: str
    participants: tuple[int, ...]
    targets: tuple[TargetTensor, ...]
    regions: tuple[LoadRegion, ...]
    budget: LoadBudget = LoadBudget()
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.schema, ModelSchema) or not isinstance(self.budget, LoadBudget):
            raise DeltaCodecError("load plan requires a schema and budget")
        require_digest(self.execution_profile_id, "execution profile")
        self.budget.validate_schema(self.schema)
        if (
            not isinstance(self.participants, tuple)
            or not self.participants
            or len(self.participants) > 4096
            or self.participants != tuple(range(len(self.participants)))
            or any(type(rank) is not int for rank in self.participants)
        ):
            raise DeltaCodecError("load participants must be contiguous group-relative ranks")
        if not isinstance(self.targets, tuple) or not isinstance(self.regions, tuple):
            raise DeltaCodecError("load directories must be immutable tuples")
        if len(self.targets) > self.budget.max_regions or len(self.regions) > self.budget.max_regions:
            raise DeltaCodecError("load directory exceeds metadata budget")
        target_map = {}
        for target in self.targets:
            if not isinstance(target, TargetTensor) or target.rank not in self.participants:
                raise DeltaCodecError("invalid target or nonparticipating rank")
            target.tensor.validate(self.schema.limits.codec)
            key = target.rank, target.tensor.name
            if key in target_map:
                raise DeltaCodecError("duplicate target name")
            target_map[key] = target
        for target in self.targets:
            if target.alias_of is not None:
                owner = target_map.get((target.rank, target.alias_of))
                if owner is None or owner.alias_of is not None or owner is target:
                    raise DeltaCodecError("target alias must reference an independent owner")
                if (owner.tensor.shape, owner.tensor.dtype) != (target.tensor.shape, target.tensor.dtype):
                    raise DeltaCodecError("target whole alias shape/dtype mismatch")
        sources = {entry.tensor.name: entry for entry in self.schema.tensors if entry.alias_of is None}
        source_regions: dict[str, list[LoadRegion]] = {name: [] for name in sources}
        target_regions: dict[tuple[int, str], list[LoadRegion]] = {key: [] for key in target_map}
        for region in self.regions:
            if not isinstance(region, LoadRegion):
                raise DeltaCodecError("invalid load region")
            target = target_map.get((region.rank, region.target))
            if target is None or target.alias_of is not None:
                raise DeltaCodecError("load region has unknown or alias destination")
            width = target.tensor.element_size
            if any(
                value % width
                for value in (region.target_offset, region.source_offset, region.row_bytes, region.source_stride)
            ):
                raise DeltaCodecError("load region must be element aligned")
            if region.target_offset + region.nbytes > target.tensor.nbytes:
                raise DeltaCodecError("load region exceeds target")
            if region.source is not None:
                source = sources.get(region.source)
                if source is None or source.tensor.dtype != target.tensor.dtype:
                    raise DeltaCodecError("source must be a canonical owner with the same dtype")
                end = region.source_offset + (region.rows - 1) * region.source_stride + region.row_bytes
                if end > source.tensor.nbytes:
                    raise DeltaCodecError("load region exceeds canonical source")
                source_regions[region.source].append(region)
            target_regions[region.rank, region.target].append(region)
        for key, target in target_map.items():
            end = 0
            for region in sorted(target_regions[key], key=lambda r: r.target_offset):
                if region.target_offset != end:
                    raise DeltaCodecError("target coverage contains a gap or overlap")
                end += region.nbytes
            if target.alias_of is None and end != target.tensor.nbytes:
                raise DeltaCodecError("target coverage is incomplete")
        # Streaming union permits replicas, including replicated GQA heads, but
        # proves that no canonical byte is omitted from the complete group.
        for name, source in sources.items():
            end = 0
            streams = [region.source_intervals() for region in source_regions[name]]
            for start, stop in heapq.merge(*streams):
                if start > end:
                    raise DeltaCodecError("canonical coverage contains a gap")
                end = max(end, stop)
            if end != source.tensor.nbytes:
                raise DeltaCodecError("canonical coverage is incomplete")
        targets = tuple(sorted(self.targets, key=lambda t: (t.rank, t.tensor.name)))
        regions = tuple(sorted(self.regions, key=lambda r: (r.rank, r.target, r.target_offset)))
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "regions", regions)
        data = {
            "format_version": 1,
            "schema_id": self.schema.schema_id,
            "execution_profile_id": self.execution_profile_id,
            "participants": self.participants,
            "targets": [asdict(target) for target in targets],
            "regions": [asdict(region) for region in regions],
        }
        object.__setattr__(self, "plan_id", content_hash(canonical_json(data, self.schema.limits.max_directory_bytes)))


@dataclass(frozen=True)
class LoadTile:
    rank: int
    target: str
    target_offset: int
    source: str | None
    source_offset: int
    nbytes: int


def iter_load_tiles(plan: LoadPlan, rank: int) -> Iterator[LoadTile]:
    """Yield one copy at a time; source tiles never straddle canonical
    chunks."""
    if type(rank) is not int or rank not in plan.participants:
        raise DeltaCodecError("nonparticipating load rank")
    targets = {(target.rank, target.tensor.name): target.tensor for target in plan.targets}
    for region in plan.regions:
        if region.rank != rank:
            continue
        alignment = targets[rank, region.target].element_size
        tile_bytes = plan.budget.tile_bytes // alignment * alignment
        for row in range(region.rows):
            offset = 0
            while offset < region.row_bytes:
                source_offset = region.source_offset + row * region.source_stride + offset
                size = min(tile_bytes, region.row_bytes - offset)
                if region.source is not None:
                    size = min(size, plan.schema.chunk_bytes - source_offset % plan.schema.chunk_bytes)
                yield LoadTile(
                    rank,
                    region.target,
                    region.target_offset + row * region.row_bytes + offset,
                    region.source,
                    source_offset,
                    size,
                )
                offset += size
