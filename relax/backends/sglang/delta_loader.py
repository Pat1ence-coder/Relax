# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SGLang ``custom_weight_loader`` that installs sparse weight deltas in place.

Registered on the engine as ``relax.utils.delta_wire.LOADER_PATH`` and reached
through ``/update_weights_from_distributed(load_format=...)`` (see the Relax
SGLang patch). Message layout is described in :mod:`relax.utils.delta_wire`.

The rollout keeps no weight snapshot. For each sparse entry a full-shape tensor
holding the new values at changed positions and NaN elsewhere is handed to the
model's own ``load_weights``, while ``Tensor.copy_`` into model storage is
rewritten as ``where(isnan(src), dst, src)``. The masked copy is fail-closed: a
NaN-carrying source that cannot be masked (shape/dtype mismatch) raises instead
of being written. The trainer never sends NaN as a changed value (it falls back
to a full sync), so NaN only ever means "unchanged".

State is per TP rank and kept consistent across the TP group: after every local
step the ranks exchange their errors and all fail together. Any failure, and the
initial state, set ``must_full``: deltas are rejected until a full sync followed
by a ``reset`` message re-seeds the version. Errors are raised so SGLang reports
the call as failed (HTTP 400) without leaving the scheduler; the trainer then
falls back to a full sync in the same pause window.
"""

from contextlib import contextmanager

import torch
import torch.distributed as dist

from relax.utils.delta_wire import (
    DTYPES,
    ERROR_TAG,
    IDX_NAME,
    META_NAME,
    PONG,
    TENSOR_PATH_SAFE,
    UNSUPPORTED,
    VAL_NAME,
    VERIFY_MISMATCH,
    align,
    decode_meta,
    int_view,
    payload_sha256,
)


CHUNK_BYTES = 512 << 20

_STATE: dict = {
    "version": None,  # last committed weight version
    "epoch": None,  # store epoch of the last full package installed (shared-storage transport)
    "pending": None,  # (version, next_bucket, n_buckets) while a delta version is being installed
    "must_full": True,
    "error": "not seeded",
    "storage_ptrs": None,  # data_ptrs of every parameter/buffer storage
    "known_nan": set(),  # params that legitimately hold NaN after the last full sync
    "verify": {},  # name -> mismatched elements found by the in-progress verify stream
}


class MaskedCopyError(RuntimeError):
    pass


def load_weights(model: torch.nn.Module, named_tensors) -> None:
    """Entry point called by SGLang with the received named tensors."""
    named_tensors = list(named_tensors)
    if not named_tensors or named_tensors[0][0] != META_NAME:
        model.load_weights(named_tensors)
        return
    meta = decode_meta(named_tensors[0][1])
    rest = named_tensors[1:]
    action = meta.get("action")
    if action == "delta":
        _delta(model, meta, dict(rest))
    elif action == "full":
        _full(model, meta, rest)
    elif action == "reset":
        _reset(model, meta)
    elif action == "install":
        _install(model, meta)
    elif action == "ping":
        # the trainer expects this exact failure to prove the message reached this loader
        _agree(f"{PONG} {TENSOR_PATH_SAFE}" if _tensor_path_reports_errors() else PONG)
    else:
        _agree(f"unknown action {action!r}")


# ---------------------------------------------------------------- install (shared storage)


def _install(model: torch.nn.Module, meta: dict) -> None:
    """Install a published version package (:mod:`relax.utils.delta_store`) by
    replaying its buckets; every TP rank reads the package itself.

    A full package is followed by a ``reset`` message, as over NCCL.
    """
    from relax.utils.delta_store import FULL, read_buckets

    error = None
    if meta["kind"] != FULL and _STATE["epoch"] != meta["epoch"]:
        error = f"epoch mismatch: committed epoch {_STATE['epoch']}, package epoch {meta['epoch']}"
    _agree(error, on_error=_fail)
    buckets = read_buckets(meta["path"], next(model.parameters()).device)
    while True:
        bucket = None
        try:
            bucket = next(buckets, None)
        except Exception as exc:  # noqa: BLE001 - unreadable package; reported below
            error = f"reading {meta['path']}: {type(exc).__name__}: {exc}"
        _agree(error, on_error=_fail)
        if bucket is None:
            break
        bucket_meta, rest = decode_meta(bucket[0][1]), bucket[1:]
        if bucket_meta["action"] == "delta":
            _delta(model, bucket_meta, dict(rest))
        else:
            _full(model, bucket_meta, rest)


def _tensor_path_reports_errors() -> bool:
    """``/update_weights_from_tensor`` turns loader exceptions into a failed
    call only with the Relax SGLang patch; unpatched, they stop the
    scheduler."""
    import inspect

    try:
        from sglang.srt.model_executor.model_runner_components.weight_updater import WeightUpdater

        return "Custom weight loader failed" in inspect.getsource(WeightUpdater.update_weights_from_tensor)
    except Exception:  # noqa: BLE001 - unknown SGLang layout: treat as unpatched
        return False


# ---------------------------------------------------------------- delta


def _delta(model: torch.nn.Module, meta: dict, tensors: dict) -> None:
    error = None
    try:
        idx, val = tensors[IDX_NAME], tensors[VAL_NAME]
        error = _validate(meta, idx, val)
        if error is None:
            _apply(model, meta, idx, val)
    except Exception as exc:  # noqa: BLE001 - weights may be partially written; reported below
        error = f"{type(exc).__name__}: {exc}"
    _agree(error, on_error=_fail)

    if meta["bucket"] < meta["n_buckets"] - 1:
        _STATE["pending"] = (meta["version"], meta["bucket"] + 1, meta["n_buckets"])
        return
    error = None
    try:
        # every parameter: a loader that bypasses copy_ would not show up in any write log
        new_nan = sorted(set(_nan_params(model)) - _STATE["known_nan"])
        if new_nan:
            error = f"NaN in {len(new_nan)} params after install, e.g. {new_nan[:3]}"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    _agree(error, on_error=_fail)
    _STATE.update(version=meta["version"], pending=None)
    _clear_mm_cache()


def _validate(meta: dict, idx: torch.Tensor, val: torch.Tensor) -> str | None:
    if _STATE["must_full"]:
        return f"must_full ({_STATE['error']})"
    version, base, bucket, n_buckets = meta["version"], meta["base_version"], meta["bucket"], meta["n_buckets"]
    if base != _STATE["version"] or version <= base:
        return f"version mismatch: committed={_STATE['version']} bucket={base}->{version}"
    expected = (version, 0, n_buckets) if _STATE["pending"] is None else _STATE["pending"]
    if (version, bucket, n_buckets) != expected:
        return f"bucket order: got {(version, bucket, n_buckets)}, expected {expected}"
    if idx.dtype != torch.int32 or val.dtype != torch.uint8:
        return f"payload dtypes {idx.dtype}/{val.dtype}"
    n_idx, n_val = meta["idx_len"], meta["val_len"]
    if n_idx > idx.numel() or n_val > val.numel():
        return "payload shorter than metadata"
    idx, val = idx[:n_idx], val[:n_val]
    if payload_sha256(idx, val) != meta["sha256"]:
        return "payload sha256 mismatch"

    i_end = v_end = 0
    limits, counts = [], []
    for e in meta["entries"]:
        if e["dtype"] not in DTYPES:
            return f"dtype {e['dtype']} of {e['name']}"
        itemsize = DTYPES[e["dtype"]].itemsize
        if e["i0"] != i_end or e["v0"] != align(v_end) or e["n"] <= 0:
            return f"non-contiguous layout at {e['name']}"
        i_end, v_end = e["i0"] + e["n"], e["v0"] + e["n"] * itemsize
        limits.append(_numel(e["shape"]))
        counts.append(e["n"])
    if i_end != n_idx or v_end > n_val:
        return "payload size does not match entries"
    if counts:
        if max(limits) >= 2**31:
            return "tensor too large for int32 positions"
        limit = torch.repeat_interleave(
            torch.tensor(limits, dtype=torch.int32, device=idx.device),
            torch.tensor(counts, dtype=torch.int64, device=idx.device),
        )
        if bool(((idx < 0) | (idx >= limit)).any()):
            return "index out of range"
    # one NaN check per run of entries sharing a dtype (values are contiguous within a run)
    run_start = 0
    entries = meta["entries"]
    for i in range(1, len(entries) + 1):
        if i == len(entries) or entries[i]["dtype"] != entries[run_start]["dtype"]:
            if _values_have_nan(val, entries[run_start:i]):
                return f"NaN value in sparse entries ({entries[run_start]['dtype']})"
            run_start = i
    return None


def _values_have_nan(val: torch.Tensor, run: list[dict]) -> bool:
    pieces = [_entry_values(val, e) for e in run]
    return bool(torch.isnan(torch.cat(pieces)).any()) if pieces else False


def _entry_values(val: torch.Tensor, e: dict) -> torch.Tensor:
    dtype = DTYPES[e["dtype"]]
    return val[e["v0"] : e["v0"] + e["n"] * dtype.itemsize].view(dtype)


def _apply(model: torch.nn.Module, meta: dict, idx: torch.Tensor, val: torch.Tensor) -> None:
    if _STATE["storage_ptrs"] is None:
        _STATE["storage_ptrs"] = {t.untyped_storage().data_ptr() for t in [*model.parameters(), *model.buffers()]}
    chunk, nbytes = [], 0
    with _masked_copy(_STATE["storage_ptrs"]):
        for e in meta["entries"]:
            values = _entry_values(val, e)
            full = torch.full((_numel(e["shape"]),), float("nan"), dtype=values.dtype, device=values.device)
            full.index_copy_(0, idx[e["i0"] : e["i0"] + e["n"]].to(torch.int64), values)
            size = full.numel() * full.element_size()
            if chunk and nbytes + size > CHUNK_BYTES:
                model.load_weights(chunk)
                chunk, nbytes = [], 0
            chunk.append((e["name"], full.view(e["shape"])))
            nbytes += size
        if chunk:
            model.load_weights(chunk)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def _masked_copy(storage_ptrs: set[int]):
    """Rewrite ``copy_`` into model storage as a NaN-masked write (fail-
    closed)."""
    orig = torch.Tensor.copy_

    def copy_(self: torch.Tensor, src, *args, **kwargs):
        ptr = self.untyped_storage().data_ptr()
        if ptr not in storage_ptrs:
            return orig(self, src, *args, **kwargs)
        if (
            isinstance(src, torch.Tensor)
            and src.is_floating_point()
            and self.is_floating_point()
            and tuple(self.shape) == tuple(src.shape)
        ):
            cast = src.to(device=self.device, dtype=self.dtype)
            return orig(self, torch.where(torch.isnan(cast), self, cast))
        if isinstance(src, torch.Tensor) and src.is_floating_point() and bool(torch.isnan(src).any()):
            raise MaskedCopyError(
                f"NaN-carrying source reached an unmaskable write into model storage: "
                f"dst {tuple(self.shape)} {self.dtype}, src {tuple(src.shape)} {src.dtype}"
            )
        return orig(self, src, *args, **kwargs)

    torch.Tensor.copy_ = copy_
    try:
        yield
    finally:
        torch.Tensor.copy_ = orig


# ---------------------------------------------------------------- full / reset


def _full(model: torch.nn.Module, meta: dict, named_tensors: list) -> None:
    """One bucket of a full sync; with ``verify`` also checks that the live
    weights already equal it."""
    _STATE.update(must_full=True, error=f"full sync to {meta['version']} in progress", pending=None)
    error = None
    try:
        if meta.get("verify"):
            _load_and_compare(model, named_tensors, _STATE["verify"])
        else:
            model.load_weights(named_tensors)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    _agree(error)


def _load_and_compare(model: torch.nn.Module, named_tensors: list, mismatched: dict) -> None:
    """Load each tensor while snapshotting every model-storage ``copy_``
    destination first; any element that changes means the in-place state
    differed from the full version (verl ``verify_every``)."""
    if _STATE["storage_ptrs"] is None:
        _STATE["storage_ptrs"] = {t.untyped_storage().data_ptr() for t in [*model.parameters(), *model.buffers()]}
    ptrs = _STATE["storage_ptrs"]
    orig = torch.Tensor.copy_
    for name, tensor in named_tensors:
        before: dict = {}

        def copy_(self, src, *args, _before=before, **kwargs):
            if self.untyped_storage().data_ptr() in ptrs:
                key = (self.data_ptr(), tuple(self.shape), tuple(self.stride()), self.dtype)
                if key not in _before:
                    _before[key] = (self, self.detach().clone())
            return orig(self, src, *args, **kwargs)

        torch.Tensor.copy_ = copy_
        try:
            model.load_weights([(name, tensor)])
        finally:
            torch.Tensor.copy_ = orig
        bad = sum(int((int_view(dst) != int_view(old)).sum()) for dst, old in before.values())
        if bad:
            mismatched[name] = mismatched.get(name, 0) + bad


def _reset(model: torch.nn.Module, meta: dict) -> None:
    """After a full sync to ``meta['version']``: re-seed the delta state."""
    error = None
    try:
        reasons = _unsupported(model)
        mismatched, _STATE["verify"] = _STATE["verify"], {}
        _STATE.update(
            version=meta["version"],
            epoch=meta.get("epoch"),
            pending=None,
            known_nan=set(_nan_params(model)),
            must_full=bool(reasons),
            error=f"{UNSUPPORTED}: {reasons}" if reasons else None,
        )
        _clear_mm_cache()
        if reasons:
            error = f"{UNSUPPORTED}: {'; '.join(reasons)}"
        elif mismatched:
            top = sorted(mismatched.items(), key=lambda kv: -kv[1])[:5]
            error = (
                f"{VERIFY_MISMATCH}: {len(mismatched)} tensors / {sum(mismatched.values())} elements differ "
                f"from the full sync, e.g. {top}"
            )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    _agree(error)


def _unsupported(model: torch.nn.Module) -> list[str]:
    """Engine configurations whose live weights are not in load-time layout."""
    reasons = []
    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if getattr(quant_method, "use_flashinfer_trtllm_moe", False):
            reasons.append(f"MoE experts kept in flashinfer TRT-LLM kernel layout ({name})")
            break
    dtypes = {p.dtype for p in model.parameters()} - set(DTYPES.values())
    if dtypes:
        reasons.append(f"non-BF16/FP16/FP32 parameters {sorted(map(str, dtypes))}")
    return reasons


# ---------------------------------------------------------------- helpers


def _fail(error: str) -> None:
    _STATE.update(must_full=True, error=error, pending=None)


def _tp_group():
    from sglang.srt.distributed import get_tp_group

    return get_tp_group()


def _agree(error: str | None, on_error=None) -> None:
    """Exchange per-rank errors over the TP CPU group; raise on every rank if
    any failed."""
    group = _tp_group()
    errors = [error]
    if group.world_size > 1:
        errors = [None] * group.world_size
        dist.all_gather_object(errors, error, group=group.cpu_group)
    failed = [(rank, e) for rank, e in enumerate(errors) if e]
    if failed:
        message = "; ".join(f"tp{rank}: {e}" for rank, e in failed[:4])
        if on_error is not None:
            on_error(message)
        raise RuntimeError(f"{ERROR_TAG} {message}")


def _nan_params(model: torch.nn.Module) -> list[str]:
    names = []
    for name, p in model.named_parameters():
        if p.is_floating_point() and bool(torch.isnan(p).any()):
            names.append(name)
    return names


def _clear_mm_cache() -> None:
    from sglang.srt.managers import mm_utils

    cache = getattr(mm_utils, "embedding_cache", None)
    if cache is not None:
        cache.clear()


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n
