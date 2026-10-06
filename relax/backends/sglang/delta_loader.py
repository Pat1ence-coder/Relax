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

import os
import threading
import time
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
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

CHUNK_BYTES = 512 << 20
PREFETCH_MAX_BYTES = 2 << 30  # larger delta packages are read inside the pause
PREFETCH_POLL_S = 0.1
PREFETCH_GIVE_UP_S = 1800.0

_STATE: dict = {
    "version": None,  # last committed weight version
    "epoch": None,  # store epoch of the last full package installed (shared-storage transport)
    "pending": None,  # (version, next_bucket, n_buckets) while a delta version is being installed
    "must_full": True,
    "error": "not seeded",
    "storage_ptrs": None,  # data_ptrs of every parameter/buffer storage
    "known_nan": set(),  # params that legitimately hold NaN after the last full sync
    "verify": {},  # name -> mismatched elements found by the in-progress verify stream
    "epoch_dir": None,  # store epoch directory of the last installed package (shared-storage transport)
}

# next delta package read ahead to CPU while the rollout still generates (shared-storage transport):
# {"path": sealed package path, "buckets": [(bucket, payload sha256 or None), ...]}; guarded by _PREFETCH_LOCK
_PREFETCH: dict = {"generation": 0, "path": None, "buckets": None}
_PREFETCH_LOCK = threading.Lock()

# tcp transport: pid of this engine's receiver process (TP rank 0 only) and the trainer address it receives from
_RECEIVER: dict = {"address": None, "pid": None}


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
        if meta.get("tcp"):
            _connect(meta["tcp"])
        # the trainer expects this exact failure to prove the message reached this loader
        _agree(f"{PONG} {TENSOR_PATH_SAFE}" if _tensor_path_reports_errors() else PONG)
    else:
        _agree(f"unknown action {action!r}")


# ---------------------------------------------------------------- receive (tcp)


def _connect(address: str) -> None:
    """Receive the trainer's packages from ``address`` into a local spool
    shared by this engine's TP ranks (one connection per engine, from a
    receiver process started by TP rank 0); installs then find them there.

    The spool is created in the temporary directory (``TMPDIR``).
    """
    if _RECEIVER["address"] == address:
        return
    import shutil
    import signal
    import subprocess
    import sys
    import tempfile

    group = _tp_group()
    spool = [None]
    if getattr(group, "rank_in_group", 0) == 0:
        if _RECEIVER["pid"] is not None:
            os.kill(_RECEIVER["pid"], signal.SIGTERM)
            shutil.rmtree(_STATE["epoch_dir"], ignore_errors=True)
        spool = [tempfile.mkdtemp(prefix="dws-spool-")]
        # a package is acknowledged once every TP rank prefetched it.
        # Started through a short-lived intermediate process in its own session, so the receiver is neither a
        # child of the scheduler (SGLang's kill_process_tree) nor in its process group (Ray): it outlives a
        # killed engine just long enough to remove the spool.
        receiver = [sys.executable, "-m", "relax.utils.delta_tcp", address, spool[0], str(group.world_size)]
        # the receiver must not hold the pipe the pid is read from (its output goes to the engine's stderr)
        launcher = (
            "import subprocess, sys; "
            "print(subprocess.Popen(sys.argv[1:], stdout=sys.stderr, start_new_session=True).pid)"
        )
        out = subprocess.run(
            [sys.executable, "-c", launcher, *receiver, str(os.getpid())],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        _RECEIVER["pid"] = int(out.stdout)
    if group.world_size > 1:
        dist.broadcast_object_list(spool, src=group.first_rank, group=group.cpu_group)
    _RECEIVER["address"] = address
    _STATE["epoch_dir"] = spool[0]


# ---------------------------------------------------------------- install (shared storage)


def _install(model: torch.nn.Module, meta: dict) -> None:
    """Install a published version package (:mod:`relax.utils.delta_store`) by
    replaying its buckets; every TP rank reads the package itself.

    Without ``path`` (tcp transport) the package is the one received into this
    engine's spool. A full package is followed by a ``reset`` message, as over
    NCCL.
    """
    from relax.utils.delta_store import FULL, read_buckets, sealed_package

    error, path = None, meta.get("path")
    if meta["kind"] != FULL and _STATE["epoch"] != meta["epoch"]:
        error = f"epoch mismatch: committed epoch {_STATE['epoch']}, package epoch {meta['epoch']}"
    elif path is None:
        path = sealed_package(_STATE["epoch_dir"], meta["version"], meta["kind"]) if _STATE["epoch_dir"] else None
        if path is None:
            error = f"{meta['kind']} package of v{meta['version']} was not received"
    _agree(error, on_error=_fail)
    _STATE["epoch_dir"] = os.path.dirname(path.rstrip("/"))
    device = next(model.parameters()).device
    prefetched = _take_prefetch(path)
    if prefetched is not None:
        buckets = (([(n, t.to(device)) for n, t in bucket], sha) for bucket, sha in prefetched)
    else:
        buckets = ((bucket, None) for bucket in read_buckets(path, device))
    while True:
        item = None
        try:
            item = next(buckets, None)
        except Exception as exc:  # noqa: BLE001 - unreadable package; reported below
            error = f"reading {path}: {type(exc).__name__}: {exc}"
        _agree(error, on_error=_fail)
        if item is None:
            break
        bucket, sha = item
        bucket_meta, rest = decode_meta(bucket[0][1]), bucket[1:]
        if bucket_meta["action"] == "delta":
            _delta(model, bucket_meta, dict(rest), sha)
        else:
            _full(model, bucket_meta, rest)
    if meta["kind"] != FULL:
        _start_prefetch(meta["version"] + 1)  # a full package is followed by reset, which starts it


def _start_prefetch(version: int) -> None:
    """Read the delta package of ``version`` to CPU in the background as soon
    as the trainer seals it, and verify its payloads, so that the install in
    the pause only copies it to the GPU.

    Leaves a ready marker in the package either way (the trainer waits for
    them, bounded, before pausing). Best effort: any problem means the install
    reads the package itself.
    """
    epoch_dir = _STATE["epoch_dir"]
    if epoch_dir is None:
        return
    with _PREFETCH_LOCK:
        _PREFETCH.update(generation=_PREFETCH["generation"] + 1, path=None, buckets=None)
        generation = _PREFETCH["generation"]
    threading.Thread(target=_prefetch, args=(epoch_dir, version, generation), name="dws-prefetch", daemon=True).start()


def _prefetch(epoch_dir: str, version: int, generation: int) -> None:
    from relax.utils.delta_store import DELTA, mark_ready, read_buckets, read_manifest, sealed_package

    deadline = time.monotonic() + PREFETCH_GIVE_UP_S
    path = None
    while path is None:
        if _PREFETCH["generation"] != generation or time.monotonic() > deadline:
            return
        try:
            path = sealed_package(epoch_dir, version, DELTA)
        except OSError:
            path = None
        if path is None:
            time.sleep(PREFETCH_POLL_S)
    try:
        if sum(f["bytes"] for f in read_manifest(path)["files"]) <= PREFETCH_MAX_BYTES:
            buckets = []
            for bucket in read_buckets(path, "cpu"):
                bucket_meta, rest = decode_meta(bucket[0][1]), dict(bucket[1:])
                sha = None
                if bucket_meta["action"] == "delta":
                    sha = payload_sha256(
                        rest[IDX_NAME][: bucket_meta["idx_len"]], rest[VAL_NAME][: bucket_meta["val_len"]]
                    )
                buckets.append((bucket, sha))
            with _PREFETCH_LOCK:
                if _PREFETCH["generation"] == generation:
                    _PREFETCH.update(path=path, buckets=buckets)
    except Exception as exc:  # noqa: BLE001 - the install reads the package itself
        logger.warning(f"[delta] prefetch of {path} failed: {type(exc).__name__}: {exc}")
    try:
        mark_ready(path)
    except OSError as exc:
        logger.warning(f"[delta] cannot mark {path} ready: {exc}")


def _take_prefetch(path: str) -> list | None:
    """The prefetched buckets of ``path`` (consumed), or None."""
    with _PREFETCH_LOCK:
        hit = _PREFETCH["path"] is not None and os.path.realpath(_PREFETCH["path"]) == os.path.realpath(path)
        buckets = _PREFETCH["buckets"] if hit else None
        _PREFETCH.update(generation=_PREFETCH["generation"] + 1, path=None, buckets=None)
    return buckets


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


def _delta(model: torch.nn.Module, meta: dict, tensors: dict, sha: str | None = None) -> None:
    """``sha``: payload SHA-256 already computed on the same bytes (prefetch)."""
    error = None
    try:
        idx, val = tensors[IDX_NAME], tensors[VAL_NAME]
        error = _validate(meta, idx, val, sha)
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


def _validate(meta: dict, idx: torch.Tensor, val: torch.Tensor, sha: str | None = None) -> str | None:
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
    if (sha or payload_sha256(idx, val)) != meta["sha256"]:
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
    if error is None and meta.get("epoch") is not None:
        _start_prefetch(meta["version"] + 1)
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
