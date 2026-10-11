# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the package side of ``DeviceDirectBackend``'s delta sync: it
writes and publishes through a ``WeightPublisher`` and stops publishing once a
newer trainer took over the store.

Run on a shell instance (``object.__new__``) with only the attributes these
methods touch.
"""

import pytest
import torch


# DeviceDirectBackend's module imports megatron.core at module level.
pytest.importorskip("megatron.core")

from relax.distributed.checkpoint_service.backends.device_direct import DeviceDirectBackend  # noqa: E402
from relax.distributed.weight_sync import store as delta_store  # noqa: E402
from relax.distributed.weight_sync.publisher import WeightPublisher  # noqa: E402


def _make_backend(store_dir) -> DeviceDirectBackend:
    backend = object.__new__(DeviceDirectBackend)
    backend._publisher = WeightPublisher(str(store_dir))
    backend._store_dir = str(store_dir)
    backend._store_epoch = backend._publisher.epoch
    backend._tcp_server = None
    backend._sent = {"export": 0, "distribution": 0, "store_write": 0}
    backend.rollout_engines = {0: object(), 1: object()}
    backend.args = type("Args", (), {"rollout_num_gpus_per_engine": 2})()
    sent = []
    backend._send_control = lambda meta, weight_version: sent.append((meta, weight_version))  # type: ignore
    backend.sent_controls = sent
    return backend


def test_device_direct_installs_and_publishes_through_the_publisher(tmp_path):
    backend = _make_backend(tmp_path)
    package = backend._new_package("full", 1)
    assert backend._package is package
    package.add_bucket([("x", torch.ones(4))])
    assert backend._install_package("1") is None
    ((meta, label),) = backend.sent_controls
    assert meta == {
        "action": "install",
        "kind": "full",
        "epoch": backend._store_epoch,
        "version": 1,
        "path": package.tmp,
    }
    assert label == "1"
    assert backend._sent["distribution"] == package.nbytes * 4  # every TP rank of every engine reads it
    assert backend._superseded_error() is None
    backend._finish_package(publish=True)
    assert backend._package is None and backend._sent["store_write"] == package.nbytes
    assert delta_store.read_head(str(tmp_path))["version"] == 1


def test_device_direct_stops_publishing_once_a_newer_trainer_claimed_the_store(tmp_path):
    backend = _make_backend(tmp_path)
    backend._new_package("full", 1).add_bucket([("x", torch.ones(4))])
    backend._finish_package(publish=True)
    newer = delta_store.claim_epoch(str(tmp_path))
    # the sync fails on every rank before the verdict (rank 0 reports it as an install error) ...
    assert f"epoch {newer}" in backend._superseded_error()
    # ... and a publish racing the takeover is dropped without raising past rank 0's last collective
    backend._new_package("delta", 2, 1).add_bucket([("x", torch.ones(4))])
    backend._finish_package(publish=True)
    assert backend._package is None
    assert delta_store.read_head(str(tmp_path))["epoch"] == backend._store_epoch
    assert [p.version for p in delta_store.list_packages(str(tmp_path))] == [1]
