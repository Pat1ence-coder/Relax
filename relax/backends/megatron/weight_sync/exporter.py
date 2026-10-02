# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Synchronous TP1/TP2 export with bounded host tiles and explicit groups.

The data group uses Gloo to transport raw CPU tiles after synchronous D2H
copies. Every rank executes the same broadcasts and votes; only export rank 0
writes a canonical disk generation. This first adapter is for DP/PP/CP/EP=1.
"""

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any

from relax.distributed.weight_sync import (
    CanonicalTile,
    DeltaCodecError,
    ExportBudget,
    ExportRequest,
    ModelRoot,
    SnapshotIdentity,
    SourceReceipt,
)
from relax.distributed.weight_sync.codec.format import content_hash
from relax.distributed.weight_sync.limits import require_uint
from relax.distributed.weight_sync.serialization import canonical_json
from relax.distributed.weight_sync.storage import DiskSnapshotStore, FrozenExport

from .boundary import TrainingBoundary
from .inventory import inspect_inventory, unwrap_model
from .profiles import Qwen3VLProfile
from .tile_plan import iter_source_spans


@dataclass(frozen=True)
class ExportResult:
    identity: SnapshotIdentity
    frozen: FrozenExport | None
    capture_seconds: float
    canonical_bytes: int
    payload_broadcast_bytes: int
    max_tile_bytes: int
    source_step: int


class MegatronSnapshotExporter:
    def __init__(
        self,
        model: Any,
        profile: Qwen3VLProfile,
        boundary: TrainingBoundary,
        *,
        control_group: Any = None,
        data_group: Any = None,
        allow_cpu: bool = False,
    ) -> None:
        import torch.distributed as dist

        self.model, self.profile, self.boundary = model, profile, boundary
        self.control_group, self.data_group, self.allow_cpu = control_group, data_group, allow_cpu
        self.rank, self.world = 0, profile.tp_size
        if control_group is None or data_group is None:
            if self.world != 1 or control_group is not None or data_group is not None:
                raise DeltaCodecError("TP2 export requires explicit control and data groups")
        else:
            if dist.get_backend(control_group) != "gloo" or dist.get_backend(data_group) != "gloo":
                raise DeltaCodecError("bounded CPU tile export requires Gloo control and data groups")
            if dist.get_world_size(control_group) != self.world or dist.get_world_size(data_group) != self.world:
                raise DeltaCodecError("export group sizes differ from native TP")
            self.rank = dist.get_rank(control_group)
            if dist.get_rank(data_group) != self.rank or any(
                dist.get_global_rank(control_group, rank) != dist.get_global_rank(data_group, rank)
                for rank in range(self.world)
            ):
                raise DeltaCodecError("export group membership/order differs")

    def _gather(self, value: Any) -> list[Any]:
        if self.control_group is None:
            return [value]
        import torch.distributed as dist

        result = [None] * self.world
        try:
            dist.all_gather_object(result, value, group=self.control_group)
        except BaseException:
            self.boundary.invalidate()
            raise
        return result

    def _validate_source(self) -> None:
        import torch.distributed as dist

        if unwrap_model(self.model) is not self.boundary.model:
            raise DeltaCodecError("source boundary belongs to a different model")
        if dist.is_initialized() and dist.get_world_size(dist.group.WORLD) != self.world:
            raise DeltaCodecError("export requires a DP1/PP1/CP1/EP1 training world")
        if self.allow_cpu:
            native_rank = dist.get_rank(dist.group.WORLD) if dist.is_initialized() else 0
        else:
            from megatron.core import parallel_state as mpu

            if not mpu.model_parallel_is_initialized():
                raise DeltaCodecError("native model parallel groups are not initialized")
            if mpu.get_tensor_model_parallel_world_size() != self.world:
                raise DeltaCodecError("native TP world differs from export profile")
            native_rank = mpu.get_tensor_model_parallel_rank()
        if native_rank != self.rank:
            raise DeltaCodecError("transport rank differs from native TP shard rank")

    def _vote(self, phase: str, error: BaseException | None = None, signature: str | None = None) -> None:
        # Only bounded digests and short diagnostics use object collectives.
        local = (phase, None if error is None else f"{type(error).__name__}: {str(error)[:512]}", signature)
        statuses = self._gather(local)
        if any(status[0] != phase for status in statuses):
            raise DeltaCodecError("export collective phase disagreement")
        failures = [(rank, status[1]) for rank, status in enumerate(statuses) if status[1] is not None]
        if failures:
            raise DeltaCodecError(f"export {phase} failed: {failures}") from error
        if any(status[2] != signature for status in statuses):
            raise DeltaCodecError(f"export {phase} signature disagreement")

    def capture(
        self, request: ExportRequest, store: DiskSnapshotStore | None, *, budget: ExportBudget = ExportBudget()
    ) -> ExportResult:
        import torch
        import torch.distributed as dist

        started = time.monotonic()
        lease = candidate = frozen = None
        error = None
        signature = None
        try:
            try:
                self._validate_source()
                lease = self.boundary.acquire(request)
                inventory = inspect_inventory(self.model, self.profile, allow_cpu=self.allow_cpu)
                plan = inventory.export_plan()
                budget.validate_schema(plan.schema)
                require_uint(8 * budget.tile_bytes, "export tile transport CPU working set", budget.max_cpu_bytes)
                signature = content_hash(
                    canonical_json(
                        {"request": request.request_id, "plan": plan.plan_id, "budget": asdict(budget)}, 16384
                    )
                )
                if self.rank == 0:
                    if store is None:
                        raise DeltaCodecError("export owner requires a private snapshot store")
                    candidate = store.capture(plan, request, budget)
                elif store is not None:
                    raise DeltaCodecError("non-owner must not create a canonical snapshot store")
            except BaseException as exc:
                error = exc
            self._vote("preflight", error, signature)
            mappings = {mapping.target.name: mapping for mapping in self.profile.mappings}
            root = ModelRoot(plan.schema.schema_id, plan.schema.directory_hash)
            written_chunks = written_bytes = broadcast_bytes = largest_tile = 0
            for spec in plan.schema.iter_chunks():
                mapping = mappings[spec.tensor.name]
                element_size = spec.tensor.element_size
                tile_size = budget.tile_bytes // element_size * element_size
                chunk_hash = hashlib.sha256()
                for offset in range(0, spec.byte_length, tile_size):
                    size = min(tile_size, spec.byte_length - offset)
                    largest_tile = max(largest_tile, size)
                    error = None
                    replica_digest = None
                    try:
                        spans = []
                        for span in iter_source_spans(
                            mapping, self.world, (spec.byte_offset + offset) // element_size, size // element_size
                        ):
                            if len(spans) == budget.max_spans_per_tile:
                                raise DeltaCodecError("tile source span count exceeds budget")
                            spans.append(span)
                        sizes = [0] * self.world
                        for span in spans:
                            sizes[span.rank] += span.elements * element_size
                        buffers = [bytearray(nbytes) for nbytes in sizes]
                        output = bytearray(size)
                        replicated = mapping.source.partition_dim == -1
                        local = buffers[self.rank] if not replicated or self.rank == 0 else bytearray(size)
                        cursor = 0
                        for span in spans:
                            if replicated or span.rank == self.rank:
                                data = inventory.read_span(mapping.source.name, span.source_offset, span.elements)
                                local[cursor : cursor + len(data)] = data
                                cursor += len(data)
                                del data
                        if replicated:
                            replica_digest = content_hash(local)
                        wire = [torch.frombuffer(buffer, dtype=torch.uint8) if buffer else None for buffer in buffers]
                    except BaseException as exc:
                        error = exc
                    self._vote("pack", error, replica_digest)
                    for rank, tensor in enumerate(wire):
                        if tensor is not None and self.data_group is not None:
                            try:
                                dist.broadcast(
                                    tensor, src=dist.get_global_rank(self.data_group, rank), group=self.data_group
                                )
                            except BaseException:
                                self.boundary.invalidate()
                                raise
                            broadcast_bytes += sizes[rank] * (self.world - 1)
                    error = None
                    try:
                        cursors = [0] * self.world
                        for span in spans:
                            length = span.elements * element_size
                            start = cursors[span.rank]
                            output[span.target_offset * element_size : span.target_offset * element_size + length] = (
                                memoryview(buffers[span.rank])[start : start + length]
                            )
                            cursors[span.rank] += length
                        chunk_hash.update(output)
                        if candidate is not None:
                            candidate.write_tile(
                                CanonicalTile(request.request_id, plan.plan_id, self.rank, spec, offset, bytes(output))
                            )
                            written_bytes += size
                    except BaseException as exc:
                        error = exc
                    self._vote("write", error)
                    del buffers, output, wire, local, spans
                root.add(spec, chunk_hash.hexdigest())
                if candidate is not None:
                    written_chunks += 1
            error = None
            try:
                lease.validate()
            except BaseException as exc:
                error = exc
            self._vote("captured-root", error, root.hexdigest())
            receipts = self._gather(
                SourceReceipt(request.request_id, plan.plan_id, self.rank, written_chunks, written_bytes)
            )
            identity = request.identity(plan.schema, root.hexdigest())
            error = None
            if candidate is not None:
                try:
                    frozen = candidate.finalize(receipts)
                    if frozen.snapshot.identity != identity:
                        raise DeltaCodecError("stored capture root differs from all-rank source root")
                except BaseException as exc:
                    error = exc
            self._vote("finalize", error)
            return ExportResult(
                identity,
                frozen,
                time.monotonic() - started,
                plan.schema.canonical_nbytes,
                broadcast_bytes,
                largest_tile,
                request.source_step,
            )
        except BaseException:
            if frozen is not None:
                frozen.close()
            elif candidate is not None:
                candidate.abort()
            raise
        finally:
            if lease is not None:
                lease.release()
