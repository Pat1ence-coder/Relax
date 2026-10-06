# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU tests for the TCP package transport (``relax.utils.delta_tcp``) and its
use by the SGLang delta loader."""

import json
import os
import signal
import socket
import tempfile
import threading

import pytest
import torch


pytest.importorskip("megatron.core")

from relax.backends.sglang import delta_loader  # noqa: E402
from relax.utils import delta_store, delta_tcp  # noqa: E402
from relax.utils.delta_wire import META_NAME, PONG, encode_meta  # noqa: E402
from tests.backends.sglang.test_delta_loader import (  # noqa: E402,F401
    FakeModel,
    assert_bitwise,
    fresh_loader,
)
from tests.utils.test_delta_store import full_buckets, perturb, sealed_delta, stop_prefetch  # noqa: E402,F401


@pytest.fixture(autouse=True)
def close_receiver():
    yield
    pid = delta_loader._RECEIVER["pid"]
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    delta_loader._RECEIVER.update(address=None, pid=None)


def sealed(tmp_path, version, sizes):
    """A sealed package of random files with the given byte sizes."""
    epoch = delta_store.claim_epoch(str(tmp_path / "store")) if version == 1 else 1
    w = delta_store.PackageWriter(str(tmp_path / "store"), epoch, version, delta_store.FULL)
    for n in sizes:
        w.add_bucket([("x", torch.randint(0, 255, (n,), dtype=torch.uint8))])
    return w.seal()


def same_files(a, b):
    files = [f["name"] for f in delta_store.read_manifest(a)["files"]]
    return all(open(os.path.join(a, f), "rb").read() == open(os.path.join(b, f), "rb").read() for f in files)


def client(server, **hello):
    sock = socket.create_connection(server.address.rsplit(":", 1))
    delta_tcp.send_json(sock, delta_tcp.HELLO, {"id": "raw", **hello})
    return sock


def sending(server, path, receivers=1, timeout=10.0):
    result = {}
    t = threading.Thread(target=lambda: result.update(acked=server.send(path, receivers, timeout)))
    t.start()
    return t, result


def test_delta_tcp_sends_packages_in_chunks_and_keeps_only_the_latest(tmp_path):
    server = delta_tcp.PackageServer("127.0.0.1", chunk_bytes=1000)
    spool = tmp_path / "spool"
    spool.mkdir()
    receiver = delta_tcp.PackageReceiver(server.address, str(spool), window=3000)
    try:
        p1 = sealed(tmp_path, 1, [10_000, 1, 2500])
        assert server.send(p1, 1, 10.0) == 1
        files = delta_store.read_manifest(p1)["files"]
        frames = sum(-(-f["bytes"] // 1000) for f in files)  # 1000 B chunks
        header = delta_tcp.HEADER.size + delta_tcp.DATA_HEADER.size
        assert server.sent_bytes == sum(f["bytes"] for f in files) + frames * header
        got = spool / os.path.basename(p1)
        assert same_files(p1, str(got)) and (got / delta_store.MANIFEST).exists()
        p2 = sealed(tmp_path, 2, [4000])
        assert server.send(p2, 1, 10.0) == 1
        assert os.listdir(spool) == [os.path.basename(p2)]
        assert same_files(p2, str(spool / os.path.basename(p2)))
    finally:
        receiver.close()


def test_delta_tcp_stops_sending_when_the_window_is_used_up(tmp_path):
    server = delta_tcp.PackageServer("127.0.0.1", chunk_bytes=1000)
    path = sealed(tmp_path, 1, [10_000])
    t, result = sending(server, path, timeout=3.0)
    sock = client(server, window=2000)
    sock.settimeout(1.0)
    assert delta_tcp.recv_frame(sock)[0] == delta_tcp.PACKAGE
    data = [delta_tcp.recv_frame(sock) for _ in range(2)]
    assert [k for k, _ in data] == [delta_tcp.DATA] * 2
    with pytest.raises(socket.timeout):  # no credit returned: nothing more is sent
        delta_tcp.recv_frame(sock)
    delta_tcp.send_json(sock, delta_tcp.CREDIT, {"bytes": 1000})
    kind, payload = delta_tcp.recv_frame(sock)
    assert kind == delta_tcp.DATA and delta_tcp.DATA_HEADER.unpack_from(payload) == (0, 2000)
    sock.close()
    t.join()
    assert result["acked"] == 0


def test_delta_tcp_resumes_from_the_reported_offset_and_counts_a_lost_ack(tmp_path):
    server = delta_tcp.PackageServer("127.0.0.1", chunk_bytes=1000)
    path = sealed(tmp_path, 1, [3000, 3000])
    name = os.path.basename(path)
    size0, size1 = (f["bytes"] for f in delta_store.read_manifest(path)["files"])
    t, result = sending(server, path)
    with client(server, window=1 << 20) as sock:
        delta_tcp.recv_frame(sock)
        delta_tcp.recv_frame(sock)
    # reconnect holding all of file 0 and 1000 bytes of file 1
    with client(server, window=1 << 20, name=name, sizes=[size0, 1000], done=False) as sock:
        kind, payload = delta_tcp.recv_frame(sock)
        assert kind == delta_tcp.PACKAGE and json.loads(payload)["name"] == name
        offsets = [delta_tcp.DATA_HEADER.unpack_from(delta_tcp.recv_frame(sock)[1]) for _ in range(2)]
        assert offsets == [(1, 1000), (1, 2000)]
    # the receiver had it all but its ACK was lost: reconnecting counts it
    with client(server, window=1 << 20, name=name, sizes=[size0, size1], done=True):
        t.join()
    assert result["acked"] == 1


def test_delta_tcp_receiver_resumes_after_a_dropped_connection(tmp_path):
    server = delta_tcp.PackageServer("127.0.0.1", chunk_bytes=1000)
    spool = tmp_path / "spool"
    spool.mkdir()
    receiver = delta_tcp.PackageReceiver(server.address, str(spool), window=2000)
    real_send = delta_tcp.send_frame
    sent = {"data": 0}

    def drop_once(sock, kind, *parts):
        if kind == delta_tcp.DATA:
            sent["data"] += 1
            if sent["data"] == 4:
                sock.shutdown(socket.SHUT_RDWR)  # connection lost mid-file
        return real_send(sock, kind, *parts)

    try:
        delta_tcp.send_frame = drop_once
        path = sealed(tmp_path, 1, [9000])
        assert server.send(path, 1, 20.0) == 1
        assert same_files(path, str(spool / os.path.basename(path)))
        assert sent["data"] < 12  # resumed, not restarted from 0 (9 chunks + at most one lost in flight)
    finally:
        delta_tcp.send_frame = real_send
        receiver.close()


def test_delta_loader_installs_packages_received_over_tcp(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # the receiver spool goes here
    model = FakeModel()
    s1 = model.hf_state()
    store = tmp_path / "store"
    server = delta_tcp.PackageServer("127.0.0.1")
    ping = {"action": "ping", "tcp": server.address}
    with pytest.raises(RuntimeError, match=PONG):
        delta_loader.load_weights(model, [(META_NAME, encode_meta(ping, "cpu"))])
    epoch = delta_store.claim_epoch(str(store))

    def install(kind, version):
        meta = {"action": "install", "kind": kind, "epoch": epoch, "version": version}
        delta_loader.load_weights(model, [(META_NAME, encode_meta(meta, "cpu"))])

    w1 = delta_store.PackageWriter(str(store), epoch, 1, delta_store.FULL)
    for b in full_buckets(1, s1):
        w1.add_bucket(b)
    assert server.send(w1.seal(), 1, 60.0) == 1  # includes the receiver process start-up
    install(delta_store.FULL, 1)
    reset = {"action": "reset", "version": 1, "epoch": epoch}
    delta_loader.load_weights(model, [(META_NAME, encode_meta(reset, "cpu"))])
    assert_bitwise(model, s1)
    w1.publish()
    s2 = perturb(s1, 2)
    w2, p2 = sealed_delta(store, epoch, 2, s1, s2)
    assert server.send(p2.path, 1, 10.0) == 1  # acknowledged after the loader prefetched it
    spool = delta_loader._STATE["epoch_dir"]
    assert delta_store.count_ready(os.path.join(spool, os.path.basename(p2.path))) == 1
    install(delta_store.DELTA, 2)
    assert_bitwise(model, s2)
    # a version that was never received is rejected
    with pytest.raises(RuntimeError, match="was not received"):
        install(delta_store.DELTA, 3)
    assert delta_loader._STATE["must_full"]
