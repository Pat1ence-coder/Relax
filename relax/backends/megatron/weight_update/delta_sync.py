# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Trainer side of sparse delta weight sync (fully-async rollout updates).

Each trainer rank compares the parameters it is responsible for against a CPU
snapshot of the last version the rollout engines committed, bitwise. The
changed elements are translated to HF coordinates by running Relax's own
export chain (``all_gather_param`` -> ``BridgeConverter.convert``) on a buffer
holding the new values at changed positions and NaN elsewhere, with the TP
all-gather replaced by a local stub (own shard + NaN for the other ranks) and
every other collective forbidden. Non-NaN outputs are exactly this rank's
changed HF elements, provided the conversion only rearranges elements (checked
by the probe differential tests). The per-rank entries are merged on global
rank 0 and packed into buckets in the :mod:`relax.utils.delta_wire` format.

Contributors (each element is reported by exactly one rank):
experts by expert-data-parallel rank 0, TP-sharded params by data-parallel
rank 0, replicated params by (tp 0, dp 0).

Snapshots only advance after every engine acknowledged a version.
Supported: bridge conversion, PP=1, expert TP=1, no LoRA, no quantization.
"""

from contextlib import contextmanager

import torch
import torch.distributed as dist
from megatron.core import mpu

from relax.backends.megatron.weight_update.common import all_gather_param, named_params_and_buffers
from relax.utils.delta_wire import (
    IDX_NAME,
    META_NAME,
    VAL_NAME,
    align,
    dtype_name,
    encode_meta,
    int_view,
    payload_sha256,
)
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_PARAM_ATTRS = ("tensor_model_parallel", "partition_dim", "partition_stride", "parallel_mode")
_FORBIDDEN_COLLECTIVES = (
    "broadcast",
    "all_reduce",
    "reduce",
    "all_gather_object",
    "all_gather_into_tensor",
    "_all_gather_base",
    "gather",
    "gather_object",
    "scatter",
    "scatter_object_list",
    "broadcast_object_list",
    "barrier",
    "all_to_all",
    "all_to_all_single",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "send",
    "recv",
    "isend",
    "irecv",
    "batch_isend_irecv",
)


class DeltaUnavailable(RuntimeError):
    """This version cannot be expressed as a sparse delta; send it in full."""


def unsupported_reason(args, quantization_config) -> str | None:
    """Return why sparse delta sync cannot be used with this configuration."""
    if getattr(args, "megatron_to_hf_mode", None) != "bridge":
        return "requires --megatron-to-hf-mode bridge"
    if mpu.get_pipeline_model_parallel_world_size() != 1:
        return "pipeline parallelism is not supported"
    if getattr(args, "num_experts", 0) and mpu.get_expert_tensor_parallel_world_size() != 1:
        return "expert tensor parallelism > 1 is not supported"
    if getattr(args, "lora_rank", 0):
        return "LoRA is not supported"
    if quantization_config:
        return "quantized rollout weights are not supported"
    transport = getattr(args, "delta_transport", "nccl")
    if transport in ("shared_fs", "tcp") and not getattr(args, "delta_store_dir", None):
        return f"--delta-transport {transport} requires --delta-store-dir"
    return None


def _is_tp_sharded(name: str, param: torch.Tensor) -> bool:
    """Mirror of ``all_gather_param``'s pass-through rules."""
    if "expert_bias" in name or not hasattr(param, "tensor_model_parallel"):
        return False
    return bool(param.tensor_model_parallel) and getattr(param, "parallel_mode", None) != "duplicated"


@contextmanager
def _local_tp_gather(groups: list):
    """``dist.all_gather`` on the given groups writes the local shard and NaN
    elsewhere; any other collective raises."""
    import torch.distributed.distributed_c10d as c10d

    own_rank = {id(g): dist.get_rank(group=g) for g in groups}

    def all_gather(tensor_list, tensor, group=None, async_op=False):
        if id(group) not in own_rank or async_op:
            raise RuntimeError("delta probe: unexpected all_gather")
        for rank, out in enumerate(tensor_list):
            if rank == own_rank[id(group)]:
                out.copy_(tensor)
            else:
                out.fill_(float("nan"))

    def forbidden(op: str):
        def fail(*args, **kwargs):
            raise RuntimeError(f"delta probe: real collective {op} attempted")

        return fail

    saved = []
    for module in (dist, c10d):
        saved.append((module, "all_gather", module.all_gather))
        module.all_gather = all_gather
        for op in _FORBIDDEN_COLLECTIVES:
            if hasattr(module, op):
                saved.append((module, op, getattr(module, op)))
                setattr(module, op, forbidden(op))
    try:
        yield
    finally:
        for module, op, fn in reversed(saved):
            setattr(module, op, fn)


class SparseDeltaSync:
    """Snapshot, diff, probe and bucket packing for one trainer rank."""

    def __init__(self, args, model, bridge_converter) -> None:
        self.args = args
        self.model = model
        self.converter = bridge_converter
        self.device = next(model[0].parameters()).device
        self.committed_version: int | None = None
        self.deltas_since_verify = 0
        self.verify_next = False
        self.total_bytes = 0
        self._snapshot: dict[str, torch.Tensor] = {}
        self._pending: list[tuple[str, torch.Tensor, torch.Tensor]] = []

        tp_rank = mpu.get_tensor_model_parallel_rank()
        dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
        edp_rank = mpu.get_expert_data_parallel_rank() if getattr(args, "num_experts", 0) else 0
        self._rules = (tp_rank, dp_rank, edp_rank)
        self._tp_groups = [mpu.get_tensor_model_parallel_group()]
        if getattr(args, "num_experts", 0):
            self._tp_groups.append(mpu.get_expert_tensor_parallel_group())

    # ------------------------------------------------------------ snapshot

    def _contributes(self, name: str, param: torch.Tensor) -> bool:
        tp_rank, dp_rank, edp_rank = self._rules
        if ".experts." in name:
            return edp_rank == 0
        if _is_tp_sharded(name, param):
            return dp_rank == 0
        return tp_rank == 0 and dp_rank == 0

    def _contributed_params(self):
        for name, param in named_params_and_buffers(self.args, self.model):
            if self._contributes(name, param):
                yield name, param

    def reseed(self, version: int) -> None:
        """Snapshot the live weights after the engines committed a full sync to
        ``version`` (collective)."""
        local_bytes = 0
        for name, param in self._contributed_params():
            data = param.data.detach()
            snap = self._snapshot.get(name)
            if snap is None or snap.shape != data.shape or snap.dtype != data.dtype:
                self._snapshot[name] = data.to("cpu", copy=True)
            else:
                snap.copy_(data)
            local_bytes += data.numel() * data.element_size()
        total = torch.tensor([local_bytes], dtype=torch.int64)
        dist.all_reduce(total, group=get_gloo_group())
        self.total_bytes = int(total.item())
        self.committed_version = version
        self._pending = []
        # self-check: the first delta after every seed is re-sent in full and compared
        self.verify_next = True
        self.deltas_since_verify = 0

    def drop(self) -> None:
        self._snapshot.clear()
        self._pending = []
        self.committed_version = None

    def commit(self, version: int) -> None:
        """Advance the snapshot by the delta all engines acknowledged."""
        for name, pos, values in self._pending:
            self._snapshot[name].view(-1)[pos] = values
        self._pending = []
        self.committed_version = version
        self.deltas_since_verify += 1

    def discard(self) -> None:
        """Forget a delta the engines did not commit; the snapshot stays at the
        committed version."""
        self._pending = []

    def verified(self) -> None:
        self.verify_next = False
        self.deltas_since_verify = 0

    # ------------------------------------------------------------ diff + probe

    def compute_local(self) -> list[tuple[str, list[int], torch.dtype, torch.Tensor, torch.Tensor]]:
        """Return this rank's changed HF elements as ``(hf_name, shape, dtype,
        idx, values)``.

        Purely local. Raises :class:`DeltaUnavailable` when the version must be
        sent in full.
        """
        self.converter.init_tasks()
        entries = []
        pending = []
        for name, param in self._contributed_params():
            live = param.data
            snap = self._snapshot.get(name)
            if snap is None or snap.shape != live.shape or snap.dtype != live.dtype:
                raise DeltaUnavailable(f"no snapshot for {name}")
            old = snap.to(live.device, non_blocking=True)
            pos = (int_view(live) != int_view(old)).nonzero().view(-1)
            del old
            if pos.numel() == 0:
                continue
            if not live.is_floating_point():
                raise DeltaUnavailable(f"non-floating parameter {name} changed")
            values = live.reshape(-1)[pos]
            if bool(torch.isnan(values).any()):
                raise DeltaUnavailable(f"NaN among the changed values of {name}")
            probe = torch.full_like(live, float("nan"))
            probe.view(-1)[pos] = values
            probe = torch.nn.Parameter(probe, requires_grad=False)
            for attr in _PARAM_ATTRS:
                if hasattr(param, attr):
                    setattr(probe, attr, getattr(param, attr))
            with _local_tp_gather(self._tp_groups):
                outputs = self.converter.convert(name, all_gather_param(self.args, name, probe))
            # changes can legitimately vanish here (e.g. vocab padding rows removed by the conversion)
            for hf_name, hf_tensor in outputs:
                flat = hf_tensor.reshape(-1)
                if flat.numel() >= 2**31:
                    raise DeltaUnavailable(f"{hf_name} has too many elements for int32 positions")
                idx = (~torch.isnan(flat)).nonzero().view(-1)
                if idx.numel():
                    entries.append((hf_name, list(hf_tensor.shape), flat.dtype, idx.to(torch.int32), flat[idx]))
            pending.append((name, pos.cpu(), values.cpu()))
        self._pending = pending
        return entries

    # ------------------------------------------------------------ gather + pack

    def gather_to_rank0(self, entries: list) -> list | None:
        """Collect all ranks' entries on global rank 0 and merge them by HF
        name (collective).

        Returns the merged list on rank 0 and ``None`` elsewhere.
        """
        rank, world = dist.get_rank(), dist.get_world_size()
        meta = [(n, s, dtype_name(d), int(i.numel())) for n, s, d, i, _ in entries]
        metas = [None] * world if rank == 0 else None
        dist.gather_object(meta, metas, dst=0, group=get_gloo_group())

        if rank != 0:
            if entries:
                idx = torch.cat([e[3] for e in entries])
                dist.send(idx, dst=0, group=dist.group.WORLD)
                dist.send(_concat_aligned([e[4] for e in entries]), dst=0, group=dist.group.WORLD)
            return None

        merged: dict[str, list] = {}
        self.received_bytes = 0  # idx + val bytes received from the other ranks (byte accounting)
        for src in range(world):
            if src == 0:
                received = entries
            else:
                received = []
                if metas[src]:
                    dtypes = [_dtype(d) for _, _, d, _ in metas[src]]
                    n_idx = sum(n for *_, n in metas[src])
                    n_val = sum(align(n * d.itemsize) for (*_, n), d in zip(metas[src], dtypes))
                    idx = torch.empty(n_idx, dtype=torch.int32, device=self.device)
                    val = torch.empty(n_val, dtype=torch.uint8, device=self.device)
                    dist.recv(idx, src=src, group=dist.group.WORLD)
                    dist.recv(val, src=src, group=dist.group.WORLD)
                    self.received_bytes += idx.numel() * idx.element_size() + val.numel()
                    i0 = v0 = 0
                    for (name, shape, _, n), dtype in zip(metas[src], dtypes):
                        nbytes = n * dtype.itemsize
                        received.append((name, shape, dtype, idx[i0 : i0 + n], val[v0 : v0 + nbytes].view(dtype)))
                        i0 += n
                        v0 += align(nbytes)
            for name, shape, dtype, idx, values in received:
                slot = merged.setdefault(name, [shape, dtype, [], []])
                if slot[0] != shape or slot[1] != dtype:
                    raise DeltaUnavailable(f"inconsistent shape/dtype for {name} across ranks")
                slot[2].append(idx)
                slot[3].append(values.reshape(-1))
        return [(n, s, d, torch.cat(i), torch.cat(v)) for n, (s, d, i, v) in merged.items()]

    def payload_bytes(self, merged: list) -> int:
        return sum(i.numel() * 4 + align(v.numel() * v.element_size()) for _, _, _, i, v in merged)

    def pack(self, merged: list, base_version: int, version: int, bucket_bytes: int) -> list[list]:
        """Pack merged entries into wire-format buckets (rank 0)."""
        groups, current, size = [], [], 0
        for entry in merged:
            nbytes = entry[3].numel() * 4 + align(entry[4].numel() * entry[4].element_size())
            if current and size + nbytes > bucket_bytes:
                groups.append(current)
                current, size = [], 0
            current.append(entry)
            size += nbytes
        if current or not groups:
            groups.append(current)

        buckets = []
        for b, group in enumerate(groups):
            idx_parts, val_parts, meta_entries = [], [], []
            i0 = v0 = 0
            for name, shape, dtype, idx, values in group:
                raw = values.reshape(-1).view(torch.uint8)
                pad = align(v0) - v0
                if pad:
                    val_parts.append(torch.zeros(pad, dtype=torch.uint8, device=raw.device))
                    v0 += pad
                meta_entries.append(
                    {"name": name, "shape": shape, "dtype": dtype_name(dtype), "n": idx.numel(), "i0": i0, "v0": v0}
                )
                idx_parts.append(idx)
                val_parts.append(raw)
                i0 += idx.numel()
                v0 += raw.numel()
            idx = torch.cat(idx_parts) if idx_parts else torch.empty(0, dtype=torch.int32, device=self.device)
            val = torch.cat(val_parts) if val_parts else torch.empty(0, dtype=torch.uint8, device=self.device)
            meta = {
                "action": "delta",
                "base_version": base_version,
                "version": version,
                "bucket": b,
                "n_buckets": len(groups),
                "idx_len": i0,
                "val_len": v0,
                "sha256": payload_sha256(idx, val),
                "entries": meta_entries,
            }
            # NCCL broadcast needs non-empty tensors; the loader trims to idx_len/val_len
            if idx.numel() == 0:
                idx = torch.zeros(1, dtype=torch.int32, device=self.device)
            if val.numel() == 0:
                val = torch.zeros(1, dtype=torch.uint8, device=self.device)
            buckets.append([(META_NAME, encode_meta(meta, self.device)), (IDX_NAME, idx), (VAL_NAME, val)])
        return buckets


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _concat_aligned(values: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate raw bytes, padding each piece to ``VAL_ALIGN``."""
    parts = []
    for v in values:
        raw = v.reshape(-1).view(torch.uint8)
        parts.append(raw)
        pad = align(raw.numel()) - raw.numel()
        if pad:
            parts.append(torch.zeros(pad, dtype=torch.uint8, device=raw.device))
    return torch.cat(parts)
