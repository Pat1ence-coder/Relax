# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU tests for the training/inference split of sparse delta weight sync:

``WeightPublisher`` (no engine handle) and ``WeightInstaller`` (no trainer),
which only meet through version packages.
"""

import os
import subprocess
import sys

import pytest
import torch


pytest.importorskip("megatron.core")

from relax.backends.sglang import delta_loader  # noqa: E402
from relax.distributed.weight_sync import store as delta_store  # noqa: E402
from relax.distributed.weight_sync.installer import (  # noqa: E402
    SharedStoreSource,
    TcpSource,
    WeightInstaller,
)
from relax.distributed.weight_sync.publisher import WeightPublisher  # noqa: E402
from relax.distributed.weight_sync.wire import META_NAME, encode_meta  # noqa: E402
from tests.backends.sglang.test_delta_loader import (  # noqa: E402,F401
    FakeModel,
    assert_bitwise,
    diff_entries,
    fresh_loader,
    packer,
    perturb,
)
from tests.distributed.weight_sync.test_store import full_buckets, history, stop_prefetch, write  # noqa: E402,F401


class FakeEngine:
    """An engine whose HTTP calls go straight to the delta loader of a fake
    model (one TP rank)."""

    def __init__(self, model):
        self.model, self.paused, self.calls, self.labels = model, False, [], []
        self.fail_on = None  # (action, version) to fail once with a corrupt-looking error

    def loader(self, meta, weight_version=None, flush_cache=False):
        self.calls.append((meta["action"], meta.get("version")))
        if meta["action"] == "install":
            assert self.paused, "installed while generating"
        if self.fail_on == (meta["action"], meta.get("version")):
            self.fail_on = None
            delta_loader._STATE.update(must_full=True, error="injected", pending=None)
            return False, "[dws] tp0: injected failure"
        try:
            delta_loader.load_weights(self.model, [(META_NAME, encode_meta(meta, "cpu"))])
        except RuntimeError as e:
            return False, str(e)
        if weight_version is not None:
            self.labels.append(weight_version)
        return True, "Succeeded"

    def pause(self):
        assert not self.paused
        self.paused = True

    def resume(self):
        assert self.paused
        self.paused = False


def publish_history(publisher, n_versions, seed_state):
    """What the trainer does through its publisher: full v1, deltas after."""
    states = {1: seed_state}
    w = publisher.begin(delta_store.FULL, 1)
    for b in full_buckets(1, seed_state):
        w.add_bucket(b)
    publisher.finish(publish=True)
    for v in range(2, n_versions + 1):
        states[v] = perturb(states[v - 1], v)
        w = publisher.begin(delta_store.DELTA, v, v - 1)
        for b in packer().pack(diff_entries(states[v - 1], states[v]), v - 1, v, bucket_bytes=64):
            w.add_bucket(b)
        publisher.finish(publish=True)
    return states


def test_weight_sync_roles_do_not_import_each_other():
    """The publisher holds no engine handle and the installer needs no trainer:

    neither pulls in Ray, Megatron, SGLang or torch.distributed state.
    """
    code = (
        "import sys; import relax.distributed.weight_sync.publisher, relax.distributed.weight_sync.installer, "
        "relax.distributed.weight_sync.consume, relax.distributed.weight_sync.tcp; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('ray', 'megatron', 'sglang') "
        "or m.startswith(('relax.backends', 'relax.distributed.ray', 'relax.distributed.checkpoint_service'))); "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.getcwd())
    assert out.returncode == 0, out.stdout + out.stderr


def test_weight_installer_brings_a_new_engine_to_head_without_the_trainer(tmp_path):
    publisher = WeightPublisher(str(tmp_path))
    states = publish_history(publisher, 5, FakeModel().hf_state())
    engine = FakeEngine(FakeModel())
    installer = WeightInstaller(engine, SharedStoreSource(str(tmp_path)))
    before = installer.current()
    assert before.key is None and before.must_full
    result = installer.install()
    assert result.ok and result.state.key == (publisher.epoch, 5) and not engine.paused
    assert [os.path.basename(p) for p in result.installed] == [
        "v000001.full",
        "v000002.delta",
        "v000003.delta",
        "v000004.delta",
        "v000005.delta",
    ]
    assert_bitwise(engine.model, states[5])
    assert engine.labels[-1] == "5"
    # already there: nothing to do, generation is not paused
    calls = len(engine.calls)
    assert installer.install().ok and engine.calls[calls:] == [("state", None)]


def test_weight_installer_replays_only_the_tail_and_prefers_the_smaller_path(tmp_path):
    publisher = WeightPublisher(str(tmp_path))
    states = publish_history(publisher, 3, FakeModel().hf_state())
    engine = FakeEngine(FakeModel())
    installer = WeightInstaller(engine, SharedStoreSource(str(tmp_path)))
    assert installer.install().ok
    for v in (4, 5):
        states[v] = perturb(states[v - 1], v)
        w = publisher.begin(delta_store.DELTA, v, v - 1)
        for b in packer().pack(diff_entries(states[v - 1], states[v]), v - 1, v, bucket_bytes=64):
            w.add_bucket(b)
        publisher.finish(publish=True)
    result = installer.install()
    assert result.ok and [os.path.basename(p) for p in result.installed] == ["v000004.delta", "v000005.delta"]
    assert_bitwise(engine.model, states[5])
    # a full package of the target is smaller than a long tail: taken instead
    found = delta_store.listing(str(tmp_path))
    full = delta_store.Package(publisher.epoch, 5, delta_store.FULL, None, "x", nbytes=1)
    tail = installer.plan(
        delta_store.Listing(found.head, [*found.packages, full], found.epochs),
        (publisher.epoch, 3),
        (publisher.epoch, 5),
    )
    assert tail == [full]


def test_weight_installer_retries_from_a_full_package_after_a_rejected_delta(tmp_path):
    publisher = WeightPublisher(str(tmp_path))
    states = publish_history(publisher, 4, FakeModel().hf_state())
    engine = FakeEngine(FakeModel())
    installer = WeightInstaller(engine, SharedStoreSource(str(tmp_path)))
    assert installer.install(target=(publisher.epoch, 2)).ok
    engine.fail_on = ("install", 3)
    result = installer.install()
    assert result.ok and not engine.paused
    assert [os.path.basename(p) for p in result.installed][-4:] == [
        "v000001.full",
        "v000002.delta",
        "v000003.delta",
        "v000004.delta",
    ]
    assert_bitwise(engine.model, states[4])


def test_weight_installer_leaves_an_engine_with_half_written_weights_paused(tmp_path):
    publisher = WeightPublisher(str(tmp_path))
    publish_history(publisher, 3, FakeModel().hf_state())
    engine = FakeEngine(FakeModel())
    installer = WeightInstaller(engine, SharedStoreSource(str(tmp_path)), max_attempts=2)
    assert installer.install(target=(publisher.epoch, 2)).ok
    # every package of v3 is now unreadable
    for p in delta_store.list_packages(str(tmp_path)):
        if p.version == 3:
            file = os.path.join(p.path, "b00000.safetensors")
            data = bytearray(open(file, "rb").read())
            data[-1] ^= 1
            open(file, "wb").write(bytes(data))
    engine.fail_on = ("install", 1)  # the retry from the full package fails as well
    result = installer.install()
    assert not result.ok and result.paused and engine.paused
    assert result.state.must_full and "injected" in result.error


def test_weight_installer_reinstalls_from_full_after_the_trainer_resumed_an_older_checkpoint(tmp_path):
    """e1 ran to v5; the trainer restarted from the checkpoint of v3 as e2.

    An engine still at e1:v5 serves an abandoned version and must take e2's
    full package; cross-epoch deltas are never applied.
    """
    p1 = WeightPublisher(str(tmp_path))
    states = publish_history(p1, 5, FakeModel().hf_state())
    engine = FakeEngine(FakeModel())
    installer = WeightInstaller(engine, SharedStoreSource(str(tmp_path)))
    assert installer.install().ok
    p2 = WeightPublisher(str(tmp_path), resumed_from=(p1.epoch, 3))
    s4 = perturb(states[3], 40)
    w = p2.begin(delta_store.FULL, 4)
    for b in full_buckets(4, s4):
        w.add_bucket(b)
    p2.finish(publish=True)
    result = installer.install()
    assert result.ok and result.abandoned
    assert [os.path.basename(p) for p in result.installed] == ["v000004.full"]
    assert result.state.key == (p2.epoch, 4)
    assert_bitwise(engine.model, s4)
    # the old trainer is still alive: its next publish is refused and HEAD stays on e2
    stale = p1.begin(delta_store.DELTA, 6, 5)
    stale.add_bucket(full_buckets(6, states[5])[0])
    with pytest.raises(delta_store.StaleEpochError):
        p1.finish(publish=True)
    assert delta_store.read_head(str(tmp_path))["epoch"] == p2.epoch


def test_weight_installer_verifies_deltas_against_a_full_package(tmp_path):
    publisher = WeightPublisher(str(tmp_path))
    states = publish_history(publisher, 3, FakeModel().hf_state())
    w = publisher.begin(delta_store.FULL, 3)  # e.g. a store anchor of v3
    for b in full_buckets(3, states[3]):
        w.add_bucket(b)
    publisher.finish(publish=True)
    engine = FakeEngine(FakeModel())
    installer = WeightInstaller(engine, SharedStoreSource(str(tmp_path)))
    assert installer.install(target=(publisher.epoch, 2)).ok
    result = installer.install(verify=True)
    assert result.ok and result.verified
    assert [os.path.basename(p) for p in result.installed] == ["v000003.delta", "v000003.full"]
    # a corrupted engine at the target: verify against v3's full package corrects it and reports the
    # mismatch instead of retrying
    with torch.no_grad():
        engine.model.w.data.view(-1)[0] += 1
    result = installer.install(verify=True)
    assert not result.ok and "verify_mismatch" in result.error and not result.paused and not engine.paused
    assert_bitwise(engine.model, states[3])
    assert installer.install(verify=True).ok


def test_weight_installer_fetches_packages_over_tcp_from_a_publisher(tmp_path):
    store = tmp_path / "store"
    publisher = WeightPublisher(str(store), tcp_host="127.0.0.1")
    states = publish_history(publisher, 4, FakeModel().hf_state())
    engine = FakeEngine(FakeModel())
    source = TcpSource(publisher.address, str(tmp_path / "spool"))
    try:
        result = WeightInstaller(engine, source).install()
    finally:
        source.close()
    assert result.ok and all(p.startswith(str(tmp_path / "spool")) for p in result.installed)
    assert_bitwise(engine.model, states[4])


def test_weight_publisher_writes_resume_point_and_install_meta(tmp_path):
    publisher = WeightPublisher(str(tmp_path), resumed_from=(7, 30))
    assert delta_store.read_epoch_info(str(tmp_path), publisher.epoch)["resumed_from"] == [7, 30]
    w = publisher.begin(delta_store.FULL, 31)
    w.add_bucket([("x", torch.ones(2))])
    meta = publisher.install_meta()
    assert meta == {"action": "install", "kind": "full", "epoch": publisher.epoch, "version": 31, "path": w.tmp}
    assert publisher.head() is None
    published = publisher.finish(publish=True)
    assert publisher.writing is None and publisher.head()["version"] == 31
    assert published.nbytes == w.nbytes
