# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the batched device-to-host copy of the trainer-side delta
(``delta_sync._to_host``) and the snapshot commit that consumes it."""

import pytest
import torch


pytest.importorskip("megatron.core")

from relax.backends.megatron.weight_update import delta_sync  # noqa: E402
from relax.backends.megatron.weight_update.delta_sync import SparseDeltaSync  # noqa: E402
from relax.utils.delta_wire import int_view  # noqa: E402


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _pending(device: str) -> list:
    """Changed positions/values of params with mixed dtypes and odd byte
    sizes."""
    g = torch.Generator().manual_seed(0)
    out = []
    for name, numel, dtype, n in [
        ("a", 37, torch.bfloat16, 5),
        ("b", 11, torch.float32, 3),
        ("c", 64, torch.float16, 7),
    ]:
        pos = torch.randperm(numel, generator=g)[:n].sort().values
        values = torch.randn(n, generator=g).to(dtype)
        out.append((name, pos.to(device), values.to(device)))
    return out


def _sync(snapshot: dict) -> SparseDeltaSync:
    """A SparseDeltaSync with only the snapshot state (no Megatron groups)."""
    sync = SparseDeltaSync.__new__(SparseDeltaSync)
    sync._snapshot = snapshot
    sync._pending, sync._pending_ready = [], None
    sync.committed_version = 1
    sync.deltas_since_verify = 0
    return sync


@requires_cuda
def test_delta_sync_to_host_matches_per_param_copy():
    pending = _pending("cuda")
    host, ready = delta_sync._to_host(pending)
    assert ready is not None
    ready.synchronize()
    assert [n for n, _, _ in host] == ["a", "b", "c"]
    for (_, pos, values), (_, hpos, hvalues) in zip(pending, host):
        assert not hpos.is_cuda and hpos.is_pinned()
        assert hvalues.dtype == values.dtype
        assert torch.equal(hpos, pos.cpu())
        assert torch.equal(int_view(hvalues), int_view(values.cpu()))


def test_delta_sync_to_host_cpu_passthrough():
    pending = _pending("cpu")
    host, ready = delta_sync._to_host(pending)
    assert ready is None
    for (_, pos, values), (_, hpos, hvalues) in zip(pending, host):
        assert torch.equal(hpos, pos) and torch.equal(int_view(hvalues), int_view(values))


def test_delta_sync_to_host_empty():
    assert delta_sync._to_host([]) == ([], None)


@requires_cuda
def test_delta_sync_commit_applies_batched_copy():
    pending = _pending("cuda")
    snapshot = {name: torch.zeros(numel, dtype=v.dtype) for (name, _, v), numel in zip(pending, (37, 11, 64))}
    expected = {name: t.clone() for name, t in snapshot.items()}
    for name, pos, values in pending:
        expected[name][pos.cpu()] = values.cpu()

    sync = _sync(snapshot)
    sync._pending, sync._pending_ready = delta_sync._to_host(pending)
    sync.commit(2)

    assert sync.committed_version == 2 and sync._pending == [] and sync._pending_ready is None
    for name in snapshot:
        assert torch.equal(int_view(snapshot[name]), int_view(expected[name]))


@requires_cuda
def test_delta_sync_discard_drops_pending_copy():
    snapshot = {"a": torch.zeros(37, dtype=torch.bfloat16)}
    sync = _sync(snapshot)
    sync._pending, sync._pending_ready = delta_sync._to_host(_pending("cuda")[:1])
    sync.discard()
    assert sync._pending == [] and sync._pending_ready is None
    assert not bool(snapshot["a"].any())


class _IdentityConverter:
    """Bridge stand-in: every Megatron param maps 1:1 to an HF tensor."""

    def init_tasks(self) -> None:
        pass

    def convert(self, name: str, tensor: torch.Tensor) -> list:
        return [(f"hf.{name}", tensor)]


def _live_params() -> dict:
    g = torch.Generator().manual_seed(1)
    params = {}
    for name, shape, dtype in [
        ("w", (8, 6), torch.bfloat16),
        ("bias", (5,), torch.float32),
        ("e", (16,), torch.float16),
    ]:
        p = torch.nn.Parameter(torch.randn(*shape, generator=g).to(dtype).cuda(), requires_grad=False)
        p.tensor_model_parallel = False  # replicated: all_gather_param passes it through
        params[name] = p
    return params


def _local_sync(monkeypatch, params: dict) -> SparseDeltaSync:
    monkeypatch.setattr(delta_sync, "named_params_and_buffers", lambda args, model: iter(params.items()))
    sync = _sync({name: p.data.cpu().clone() for name, p in params.items()})
    sync.args, sync.model, sync.converter = None, None, _IdentityConverter()
    sync._rules, sync._tp_groups = (0, 0, 0), []
    return sync


@requires_cuda
def test_delta_sync_compute_local_then_commit_tracks_live_weights(monkeypatch):
    params = _live_params()
    sync = _local_sync(monkeypatch, params)
    with torch.no_grad():
        params["w"].view(-1)[[3, 17, 40]] += 1
        params["e"].view(-1)[[0, 15]] -= 2  # "bias" unchanged

    entries = sync.compute_local()
    assert sorted(e[0] for e in entries) == ["hf.e", "hf.w"]
    by_name = {e[0]: e for e in entries}
    assert by_name["hf.w"][3].tolist() == [3, 17, 40] and by_name["hf.e"][3].tolist() == [0, 15]
    assert sync._pending_ready is not None

    sync.commit(2)
    for name, p in params.items():
        assert torch.equal(int_view(sync._snapshot[name]), int_view(p.data.cpu())), name
    assert sync.compute_local() == []  # nothing changed since the commit


@requires_cuda
def test_delta_sync_compute_local_rejects_nan_change(monkeypatch):
    params = _live_params()
    sync = _local_sync(monkeypatch, params)
    with torch.no_grad():
        params["w"].view(-1)[2] += 1
        params["bias"][1] = float("nan")
    with pytest.raises(delta_sync.DeltaUnavailable, match="bias"):
        sync.compute_local()
