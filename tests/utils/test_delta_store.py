# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU tests for shared-storage version packages (``relax.utils.delta_store``),
their installation through the SGLang delta loader and the offline consumer."""

import json
import os
import time

import pytest
import torch


pytest.importorskip("megatron.core")

from relax.backends.sglang import delta_loader  # noqa: E402
from relax.tools import delta_consume  # noqa: E402
from relax.utils import delta_store  # noqa: E402
from relax.utils.delta_wire import META_NAME, encode_meta, int_view  # noqa: E402
from tests.backends.sglang.test_delta_loader import (  # noqa: E402,F401
    FakeModel,
    assert_bitwise,
    diff_entries,
    fresh_loader,
    packer,
    perturb,
)


@pytest.fixture(autouse=True)
def stop_prefetch():
    yield
    with delta_loader._PREFETCH_LOCK:  # stops background prefetch threads left polling
        delta_loader._PREFETCH.update(generation=delta_loader._PREFETCH["generation"] + 1, path=None, buckets=None)


def write(store, epoch, version, kind, buckets, base=None):
    w = delta_store.PackageWriter(str(store), epoch, version, kind, base)
    for b in buckets:
        w.add_bucket(b)
    return w.publish()


def full_buckets(version, state):
    meta = [(META_NAME, encode_meta({"action": "full", "version": version, "verify": False}, "cpu"))]
    names = sorted(state)
    return [meta + [(n, state[n]) for n in names[:2]], meta + [(n, state[n]) for n in names[2:]]]


def install(model, package):
    meta = {"action": "install", "kind": package.kind, "epoch": package.epoch, "version": package.version}
    delta_loader.load_weights(model, [(META_NAME, encode_meta(dict(meta, path=package.path), "cpu"))])
    if package.kind == delta_store.FULL:
        reset = {"action": "reset", "version": package.version, "epoch": package.epoch}
        delta_loader.load_weights(model, [(META_NAME, encode_meta(reset, "cpu"))])


def history(store, n_versions, seed_state):
    """Epoch 1: full v1, then deltas v2..v{n}; returns the state of every
    version."""
    epoch = delta_store.claim_epoch(str(store))
    states = {1: seed_state}
    write(store, epoch, 1, delta_store.FULL, full_buckets(1, seed_state))
    for v in range(2, n_versions + 1):
        states[v] = perturb(states[v - 1], v)
        buckets = packer().pack(diff_entries(states[v - 1], states[v]), v - 1, v, bucket_bytes=64)
        write(store, epoch, v, delta_store.DELTA, buckets, base=v - 1)
    return epoch, states


def test_delta_store_publish_is_atomic_and_never_overwrites(tmp_path):
    epoch = delta_store.claim_epoch(str(tmp_path))
    assert epoch == 1 and delta_store.claim_epoch(str(tmp_path)) == 2
    w = delta_store.PackageWriter(str(tmp_path), 1, 1, delta_store.FULL)
    w.add_bucket([("x", torch.ones(3))])
    assert delta_store.list_packages(str(tmp_path)) == []  # not visible before publish
    package = w.publish()
    assert [p.key for p in delta_store.list_packages(str(tmp_path))] == [(1, 1)]
    head = json.load(open(tmp_path / delta_store.HEAD))
    assert head == {"epoch": 1, "version": 1, "kind": "full", "path": os.path.relpath(package.path, tmp_path)}
    again = delta_store.PackageWriter(str(tmp_path), 1, 1, delta_store.FULL)
    again.add_bucket([("x", torch.zeros(3))])
    with pytest.raises(OSError):
        again.publish()
    again.abort()
    (bucket,) = delta_store.read_buckets(package.path)
    assert torch.equal(dict(bucket)["x"], torch.ones(3))


def test_delta_store_read_rejects_truncated_bucket(tmp_path):
    epoch = delta_store.claim_epoch(str(tmp_path))
    package = write(tmp_path, epoch, 1, delta_store.FULL, [[("x", torch.ones(1000))]])
    file = os.path.join(package.path, "b00000.safetensors")
    os.truncate(file, os.path.getsize(file) - 8)
    with pytest.raises(ValueError, match="manifest says"):
        list(delta_store.read_buckets(package.path))


def test_delta_store_plan_follows_chain_or_restarts_from_full(tmp_path):
    epoch, _ = history(tmp_path, 5, FakeModel().hf_state())
    keys = lambda ps: [(p.version, p.kind) for p in ps]  # noqa: E731
    assert keys(delta_store.plan(str(tmp_path))) == [
        (1, "full"),
        (2, "delta"),
        (3, "delta"),
        (4, "delta"),
        (5, "delta"),
    ]
    assert keys(delta_store.plan(str(tmp_path), have=(epoch, 3))) == [(4, "delta"), (5, "delta")]
    assert keys(delta_store.plan(str(tmp_path), target=(epoch, 2))) == [(1, "full"), (2, "delta")]
    # an anchor at v4 is preferred over replaying from v1; a consumer from another epoch restarts
    write(tmp_path, epoch, 4, delta_store.FULL, [[("w", torch.zeros(1))]])
    assert keys(delta_store.plan(str(tmp_path), have=(epoch - 1, 9))) == [(4, "full"), (5, "delta")]
    # a missing delta breaks the chain: fall back to the latest full before the gap
    for p in delta_store.list_packages(str(tmp_path)):
        if (p.version, p.kind) == (5, "delta"):
            import shutil

            shutil.rmtree(p.path)
    with pytest.raises(FileNotFoundError):
        delta_store.plan(str(tmp_path), have=(epoch, 3), target=(epoch, 5))


def test_delta_store_prune_keeps_two_full_packages_and_their_chain(tmp_path):
    epoch, _ = history(tmp_path, 3, FakeModel().hf_state())
    write(tmp_path, epoch, 3, delta_store.FULL, [[("w", torch.zeros(1))]])
    assert delta_store.prune(str(tmp_path), epoch) == []
    new_epoch = delta_store.claim_epoch(str(tmp_path))
    stale_tmp = delta_store.PackageWriter(str(tmp_path), epoch, 9, delta_store.DELTA, 8)  # left by a dead trainer
    write(tmp_path, new_epoch, 1, delta_store.FULL, [[("w", torch.zeros(1))]])
    removed = delta_store.prune(str(tmp_path), new_epoch)
    assert stale_tmp.tmp in removed
    assert [(p.epoch, p.version, p.kind) for p in delta_store.list_packages(str(tmp_path))] == [
        (epoch, 3, "full"),
        (new_epoch, 1, "full"),
    ]


def test_delta_loader_installs_sealed_package_before_publish(tmp_path):
    """The trainer installs a package at its sealed temporary path and
    publishes it only after every engine committed it."""
    model = FakeModel()
    state = model.hf_state()
    epoch = delta_store.claim_epoch(str(tmp_path))
    w = delta_store.PackageWriter(str(tmp_path), epoch, 1, delta_store.FULL)
    for b in full_buckets(1, state):
        w.add_bucket(b)
    sealed = delta_store.Package(epoch, 1, delta_store.FULL, None, w.seal())
    assert delta_store.list_packages(str(tmp_path)) == []
    install(model, sealed)
    assert_bitwise(model, state)
    w.publish()
    assert [p.key for p in delta_store.list_packages(str(tmp_path))] == [(epoch, 1)]


def test_delta_loader_installs_packages_bitwise_and_tracks_epoch(tmp_path):
    model = FakeModel()
    epoch, states = history(tmp_path, 4, model.hf_state())
    packages = delta_store.plan(str(tmp_path))
    for package in packages:
        install(model, package)
        assert_bitwise(model, states[package.version])
    assert delta_loader._STATE["version"] == 4 and delta_loader._STATE["epoch"] == epoch

    # a delta of another epoch (e.g. a restarted trainer) is rejected without writing
    other = delta_store.claim_epoch(str(tmp_path))
    nxt = perturb(states[4], 99)
    bad = write(tmp_path, other, 5, delta_store.DELTA, packer().pack(diff_entries(states[4], nxt), 4, 5, 1 << 20), 4)
    with pytest.raises(RuntimeError, match="epoch mismatch"):
        install(model, bad)
    assert_bitwise(model, states[4])
    assert delta_loader._STATE["must_full"]


def test_delta_loader_rejects_corrupt_package_and_recovers_from_full(tmp_path):
    model = FakeModel()
    epoch, states = history(tmp_path, 3, model.hf_state())
    full, d2, d3 = delta_store.plan(str(tmp_path))
    install(model, full)
    install(model, d2)
    file = os.path.join(d3.path, "b00000.safetensors")
    data = bytearray(open(file, "rb").read())
    data[-1] ^= 1  # flip a payload byte: sha256 of the bucket no longer matches
    open(file, "wb").write(bytes(data))
    with pytest.raises(RuntimeError, match="sha256|reading"):
        install(model, d3)
    assert_bitwise(model, states[2])
    with pytest.raises(RuntimeError, match="must_full"):
        install(model, d2)
    install(model, full)
    assert_bitwise(model, states[1])


def test_delta_consume_rebuilds_latest_version_offline(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    _, states = history(store, 6, FakeModel().hf_state())
    assert delta_consume.main(["--store-dir", str(store), "--output", str(out)]) == 0
    index = json.load(open(out / "model.safetensors.index.json"))["weight_map"]
    from safetensors.torch import load_file

    rebuilt = {}
    for file in set(index.values()):
        rebuilt.update(load_file(str(out / file)))
    assert sorted(rebuilt) == sorted(states[6])
    for name, t in states[6].items():
        assert torch.equal(int_view(rebuilt[name]), int_view(t)), name
    # an earlier version on request
    weights = delta_consume.rebuild(delta_store.plan(str(store), target=(1, 3)))
    for name, t in states[3].items():
        assert torch.equal(int_view(weights[name]), int_view(t)), name


def sealed_delta(store, epoch, version, prev, nxt, corrupt=False):
    w = delta_store.PackageWriter(str(store), epoch, version, delta_store.DELTA, version - 1)
    for b in packer().pack(diff_entries(prev, nxt), version - 1, version, bucket_bytes=64):
        w.add_bucket(b)
    if corrupt:
        file = os.path.join(w.tmp, "b00000.safetensors")
        data = bytearray(open(file, "rb").read())
        data[-1] ^= 1
        open(file, "wb").write(bytes(data))
    return w, delta_store.Package(epoch, version, delta_store.DELTA, version - 1, w.seal())


def wait_ready(path, n=1, timeout=10.0):
    deadline = time.monotonic() + timeout
    while delta_store.count_ready(path) < n:
        assert time.monotonic() < deadline, "prefetch did not finish"
        time.sleep(0.02)


def test_delta_loader_prefetches_sealed_package_before_install(tmp_path, monkeypatch):
    """After each install the loader reads the next sealed delta package in the
    background; the install of it then does not read the store."""
    model = FakeModel()
    s1 = model.hf_state()
    epoch = delta_store.claim_epoch(str(tmp_path))
    install(model, write(tmp_path, epoch, 1, delta_store.FULL, full_buckets(1, s1)))
    s2 = perturb(s1, 2)
    w2, p2 = sealed_delta(tmp_path, epoch, 2, s1, s2)
    wait_ready(p2.path)
    real_read = delta_store.read_buckets
    monkeypatch.setattr(delta_store, "read_buckets", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read")))
    install(model, p2)
    assert_bitwise(model, s2)
    w2.publish()
    # the next one is prefetched too (chain), and a package that is not the prefetched one is read normally
    monkeypatch.setattr(delta_store, "read_buckets", real_read)
    s3 = perturb(s2, 3)
    _, p3 = sealed_delta(tmp_path, epoch, 3, s2, s3)
    wait_ready(p3.path)
    install(model, delta_store.Package(epoch, 3, delta_store.DELTA, 2, p3.path + "/"))
    assert_bitwise(model, s3)
    assert delta_loader._PREFETCH["buckets"] is None


def test_delta_loader_rejects_corrupt_prefetched_package(tmp_path):
    model = FakeModel()
    s1 = model.hf_state()
    epoch = delta_store.claim_epoch(str(tmp_path))
    install(model, write(tmp_path, epoch, 1, delta_store.FULL, full_buckets(1, s1)))
    _, p2 = sealed_delta(tmp_path, epoch, 2, s1, perturb(s1, 2), corrupt=True)
    wait_ready(p2.path)
    with pytest.raises(RuntimeError, match="sha256"):
        install(model, p2)
    assert_bitwise(model, s1)
    assert delta_loader._STATE["must_full"]
