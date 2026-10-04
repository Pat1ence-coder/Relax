# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU tests for the sparse delta weight-sync wire format and SGLang delta
loader.

Messages are built with the trainer-side packer (``SparseDeltaSync.pack``) and
installed into a small fake model whose ``load_weights`` writes through
``copy_`` like SGLang's loaders (whole-param and fused narrow slices).
"""

from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("megatron.core")

from relax.backends.megatron.weight_update.delta_sync import SparseDeltaSync  # noqa: E402
from relax.backends.sglang import delta_loader  # noqa: E402
from relax.utils.delta_wire import META_NAME, VERIFY_MISMATCH, decode_meta, encode_meta, int_view  # noqa: E402


class FakeModel(torch.nn.Module):
    """``qkv`` is fused from ``q``/``k`` halves; ``w``/``b`` load whole."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.w = torch.nn.Parameter(torch.randn(8, 6, generator=g).bfloat16(), requires_grad=False)
        self.b = torch.nn.Parameter(torch.randn(6, generator=g).bfloat16(), requires_grad=False)
        self.qkv = torch.nn.Parameter(torch.randn(8, 4, generator=g).bfloat16(), requires_grad=False)
        self.loads = 0

    def load_weights(self, weights):
        for name, tensor in weights:
            self.loads += 1
            if name in ("q", "k"):
                half = self.qkv.shape[0] // 2
                self.qkv.data.narrow(0, 0 if name == "q" else half, half).copy_(tensor)
            else:
                getattr(self, name).data.copy_(tensor)

    def hf_state(self):
        half = self.qkv.shape[0] // 2
        return {
            "w": self.w.data.clone(),
            "b": self.b.data.clone(),
            "q": self.qkv.data[:half].clone(),
            "k": self.qkv.data[half:].clone(),
        }


@pytest.fixture(autouse=True)
def fresh_loader(monkeypatch):
    initial = {
        k: (set() if isinstance(v, set) else {} if isinstance(v, dict) else v) for k, v in delta_loader._STATE.items()
    }
    monkeypatch.setattr(delta_loader, "_tp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(delta_loader, "_clear_mm_cache", lambda: None)
    yield
    delta_loader._STATE.clear()
    delta_loader._STATE.update(initial)


def packer():
    p = object.__new__(SparseDeltaSync)
    p.device = torch.device("cpu")
    return p


def diff_entries(old: dict, new: dict):
    out = []
    for name in sorted(new):
        pos = (int_view(old[name]) != int_view(new[name])).nonzero().view(-1)
        if pos.numel():
            out.append((name, list(new[name].shape), new[name].dtype, pos.to(torch.int32), new[name].reshape(-1)[pos]))
    return out


def perturb(state: dict, seed: int, frac: float = 0.2) -> dict:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, t in state.items():
        x = t.clone()
        xi = int_view(x)
        mask = torch.rand(xi.shape, generator=g) < frac
        xi[mask] ^= torch.randint(1, 64, xi.shape, generator=g, dtype=torch.int16)[mask]
        out[name] = x
    return out


def full_message(version: int, state: dict, verify: bool = False):
    meta = {"action": "full", "version": version, "verify": verify}
    return [(META_NAME, encode_meta(meta, "cpu")), *sorted(state.items())]


def reset_message(version: int):
    return [(META_NAME, encode_meta({"action": "reset", "version": version}, "cpu"))]


def seed(model, version: int = 1) -> dict:
    state = model.hf_state()
    delta_loader.load_weights(model, full_message(version, state))
    delta_loader.load_weights(model, reset_message(version))
    return state


def install(model, buckets):
    for bucket in buckets:
        delta_loader.load_weights(model, bucket)


def assert_bitwise(model, state: dict):
    live = model.hf_state()
    for name in state:
        assert torch.equal(int_view(live[name]), int_view(state[name])), name


def test_delta_loader_installs_delta_bitwise_over_several_versions():
    model = FakeModel()
    state = seed(model)
    for v in range(2, 5):
        new = perturb(state, v)
        buckets = packer().pack(diff_entries(state, new), v - 1, v, bucket_bytes=64)
        assert len(buckets) > 1
        install(model, buckets)
        assert_bitwise(model, new)
        assert delta_loader._STATE["version"] == v and not delta_loader._STATE["must_full"]
        state = new


def test_delta_loader_rejects_before_seed_and_requires_full_after_failure():
    model = FakeModel()
    state = model.hf_state()
    new = perturb(state, 1)
    buckets = packer().pack(diff_entries(state, new), 1, 2, bucket_bytes=1 << 20)
    with pytest.raises(RuntimeError, match="must_full"):
        install(model, buckets)
    assert_bitwise(model, state)

    seed(model)
    tampered = [list(b) for b in buckets]
    val = tampered[0][2][1].clone()
    val[0] ^= 1
    tampered[0][2] = (tampered[0][2][0], val)
    with pytest.raises(RuntimeError, match="sha256"):
        install(model, tampered)
    assert_bitwise(model, state)
    # rejected bucket leaves the loader requiring a full sync, even for a valid delta
    with pytest.raises(RuntimeError, match="must_full"):
        install(model, buckets)
    seed(model)
    install(model, buckets)
    assert_bitwise(model, new)


def test_delta_loader_rejects_version_gaps_and_bucket_reordering():
    model = FakeModel()
    state = seed(model, version=5)
    new = perturb(state, 2)
    with pytest.raises(RuntimeError, match="version mismatch"):
        install(model, packer().pack(diff_entries(state, new), 4, 6, bucket_bytes=1 << 20))
    seed(model, version=5)
    buckets = packer().pack(diff_entries(state, new), 5, 6, bucket_bytes=64)
    with pytest.raises(RuntimeError, match="bucket order"):
        install(model, [buckets[1]])
    assert_bitwise(model, state)


def test_delta_loader_skipped_versions_are_accepted():
    model = FakeModel()
    state = seed(model, version=3)
    new = perturb(state, 3)
    install(model, packer().pack(diff_entries(state, new), 3, 7, bucket_bytes=1 << 20))
    assert_bitwise(model, new)
    assert delta_loader._STATE["version"] == 7


def test_delta_loader_rejects_nan_values():
    model = FakeModel()
    state = seed(model)
    new = perturb(state, 4)
    new["w"].view(-1)[0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN value"):
        install(model, packer().pack(diff_entries(state, new), 1, 2, bucket_bytes=1 << 20))
    assert_bitwise(model, state)


def test_delta_loader_fails_closed_on_unmaskable_copy():
    model = FakeModel()
    state = seed(model)

    def broadcast_loader(weights):
        for name, tensor in weights:
            getattr(model, name).data.unsqueeze(0).copy_(tensor)

    model.load_weights = broadcast_loader
    new = perturb({"b": state["b"]}, 5)
    with pytest.raises(RuntimeError, match="MaskedCopyError"):
        install(model, packer().pack(diff_entries({"b": state["b"]}, new), 1, 2, bucket_bytes=1 << 20))
    assert torch.equal(int_view(model.b.data), int_view(state["b"]))
    assert delta_loader._STATE["must_full"]


def test_delta_loader_detects_nan_leak_from_non_copy_writes():
    model = FakeModel()
    state = seed(model)

    def setitem_loader(weights):
        for name, tensor in weights:
            getattr(model, name).data[...] = tensor

    model.load_weights = setitem_loader
    new = perturb({"w": state["w"]}, 6)
    with pytest.raises(RuntimeError, match="NaN in 1 params"):
        install(model, packer().pack(diff_entries({"w": state["w"]}, new), 1, 2, bucket_bytes=1 << 20))
    assert delta_loader._STATE["must_full"]


def test_delta_loader_verify_reports_mismatch():
    model = FakeModel()
    state = seed(model)
    new = perturb(state, 7)
    install(model, packer().pack(diff_entries(state, new), 1, 2, bucket_bytes=1 << 20))
    # verification against the version just installed: no mismatch
    delta_loader.load_weights(model, full_message(2, new, verify=True))
    delta_loader.load_weights(model, reset_message(2))

    model.w.data.view(-1)[:1].view(torch.int16).bitwise_xor_(1)
    delta_loader.load_weights(model, full_message(2, new, verify=True))
    with pytest.raises(RuntimeError, match=VERIFY_MISMATCH):
        delta_loader.load_weights(model, reset_message(2))
    assert_bitwise(model, new)  # the full sync itself repaired the weights


def test_delta_loader_passes_plain_tensors_through():
    model = FakeModel()
    state = perturb(model.hf_state(), 8)
    delta_loader.load_weights(model, sorted(state.items()))
    assert_bitwise(model, state)


def test_delta_wire_meta_roundtrip():
    meta = {"action": "reset", "version": 3, "nested": {"a": [1, 2]}}
    assert decode_meta(encode_meta(meta, "cpu")) == meta


def test_delta_pack_empty_delta_still_produces_one_bucket():
    model = FakeModel()
    state = seed(model)
    buckets = packer().pack([], 1, 2, bucket_bytes=1 << 20)
    assert len(buckets) == 1 and decode_meta(buckets[0][0][1])["entries"] == []
    install(model, buckets)
    assert_bitwise(model, state)
    assert delta_loader._STATE["version"] == 2


def test_delta_loader_answers_ping_with_pong():
    from relax.utils.delta_wire import PONG

    with pytest.raises(RuntimeError, match=PONG):
        delta_loader.load_weights(FakeModel(), [(META_NAME, encode_meta({"action": "ping"}, "cpu"))])
