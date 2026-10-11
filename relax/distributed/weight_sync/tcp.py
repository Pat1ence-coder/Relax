# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""TCP transport for version packages
(:mod:`relax.distributed.weight_sync.store`).

Two ways to use a :class:`PackageServer`:

- push (the in-job sync): the trainer (global rank 0) offers each sealed
  package with :meth:`PackageServer.send`; every rollout engine runs one
  :class:`PackageReceiver`, which keeps one connection to it and writes each
  package into a local spool directory laid out like the store, so the SGLang
  delta loader installs (and prefetches) it like a shared-storage package.
- pull: a server given ``store_dir`` also answers :class:`PackageFetcher`
  clients, which list the store (``HEAD``, published packages, epoch records)
  and fetch any published package by its path. The server then only reads
  the store and knows nothing about its consumers, so it can also run as a
  process of its own (``python -m relax.distributed.weight_sync.tcp serve``)
  that keeps serving while the trainer restarts.

Frames are a ``!4sBI`` header (magic, type, payload length) and a payload:
JSON for control frames, ``!IQ`` (file index, offset) plus raw bytes for DATA.

- chunked transfer: package files are sent in order, in DATA frames of at most
  ``chunk_bytes``;
- backpressure: the receiver grants a byte window in HELLO and returns credit
  (CREDIT) only after a chunk is written; the server stops reading and sending
  while the window is used up;
- reconnect: the receiver reconnects with backoff and reports in HELLO the
  package it was receiving and the bytes it holds per file; the server resumes
  from there, or counts the package as acknowledged when it was complete. A
  package with another name supersedes a partial one. A fetcher re-requests
  the package with the bytes it holds per file.

Plain TCP without authentication: for trusted internal networks only.
"""

import json
import os
import shutil
import socket
import struct
import threading
import time
import uuid

from relax.distributed.weight_sync.store import DELTA, MANIFEST, Listing, Package, count_ready, listing, read_manifest
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

MAGIC = b"DWS1"
HEADER = struct.Struct("!4sBI")
DATA_HEADER = struct.Struct("!IQ")
HELLO, PACKAGE, DATA, CREDIT, ACK, REQUEST, LISTING = range(1, 8)
CHUNK_BYTES = 4 << 20
WINDOW_BYTES = 64 << 20
READY_WAIT_S = 10.0  # receiver: how long to wait for the engine's prefetch before acknowledging a delta
FETCH_RETRIES = 8  # fetcher: reconnects per request before giving up


def send_frame(sock: socket.socket, kind: int, *parts: bytes) -> None:
    sock.sendall(HEADER.pack(MAGIC, kind, sum(len(p) for p in parts)))
    for part in parts:
        sock.sendall(part)


def send_json(sock: socket.socket, kind: int, message: dict) -> None:
    send_frame(sock, kind, json.dumps(message).encode())


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:])
        if k == 0:
            raise ConnectionError("connection closed")
        got += k
    return buf


def recv_frame(sock: socket.socket) -> tuple[int, bytearray]:
    magic, kind, n = HEADER.unpack(_recv_exact(sock, HEADER.size))
    if magic != MAGIC:
        raise ConnectionError(f"bad frame magic {bytes(magic)!r}")
    return kind, _recv_exact(sock, n)


def _within(root: str, rel: str) -> str:
    """``root/rel``, refusing paths that leave ``root``."""
    root = os.path.realpath(root)
    path = os.path.realpath(os.path.join(root, rel))
    if os.path.commonpath([root, path]) != root or path == root:
        raise ValueError(f"path {rel!r} is outside {root}")
    return path


class PackageServer:
    """Sends the package of the running :meth:`send` to every connected
    receiver, and (with ``store_dir``) any published package of ``store_dir``
    to the fetchers that request it."""

    def __init__(self, host: str, port: int = 0, chunk_bytes: int = CHUNK_BYTES, store_dir: str | None = None):
        self.chunk_bytes = chunk_bytes
        self.store_dir = store_dir
        self._sock = socket.create_server((host, port))  # held while the server lives: no port race
        self.address = f"{host}:{self._sock.getsockname()[1]}"
        self._cond = threading.Condition()
        self._path: str | None = None
        self._gen = 0  # bumped by every send start and end; transfers of an older generation stop
        self._acks: set[str] = set()
        self.sent_bytes = 0  # DATA frame bytes sent to all receivers (headers included), for byte accounting
        threading.Thread(target=self._accept, name="dws-tcp-accept", daemon=True).start()

    def send(self, path: str, receivers: int, timeout: float) -> int:
        """Offer the sealed package at ``path`` until ``receivers`` receivers
        acknowledged it or ``timeout`` passed; returns how many did."""
        with self._cond:
            self._path, self._gen, self._acks = path, self._gen + 1, set()
            self._cond.notify_all()
            self._cond.wait_for(lambda: len(self._acks) >= receivers, timeout)
            acked = len(self._acks)
            self._path, self._gen = None, self._gen + 1
        return acked

    def _ack(self, receiver: str, gen: int) -> None:
        with self._cond:
            if gen == self._gen:
                self._acks.add(receiver)
                self._cond.notify_all()

    def _accept(self) -> None:
        while True:
            conn, _ = self._sock.accept()
            threading.Thread(target=self._serve, args=(conn,), name="dws-tcp-conn", daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        try:
            with conn:
                self._serve_receiver(conn)
        except OSError as e:
            logger.info(f"[delta] tcp receiver disconnected: {e}")

    def _serve_receiver(self, conn: socket.socket) -> None:
        kind, payload = recv_frame(conn)
        if kind != HELLO:
            raise ConnectionError(f"expected HELLO, got frame type {kind}")
        held = json.loads(payload)  # receiver id, window, and the package it holds (name, sizes, done)
        receiver, credit = held["id"], held["window"]
        if held.get("pull"):
            self._serve_fetcher(conn, credit)
            return
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._path is not None and receiver not in self._acks)
                path, gen = self._path, self._gen
            name, manifest = os.path.basename(path), read_manifest(path)
            if held.get("name") == name and held.get("done"):
                self._ack(receiver, gen)  # complete before the connection dropped: its ACK was lost
                held = {}
                continue
            offsets = held["sizes"] if held.get("name") == name else [0] * len(manifest["files"])
            held = {}
            send_json(conn, PACKAGE, {"name": name, "manifest": manifest})
            credit, sent = self._send_files(conn, path, manifest["files"], offsets, credit, gen)
            if not sent:
                continue  # superseded or withdrawn
            while True:
                kind, payload = recv_frame(conn)
                message = json.loads(payload)
                if kind == CREDIT:
                    credit += message["bytes"]
                elif kind == ACK and message["name"] == name:
                    self._ack(receiver, gen)
                    break

    def _serve_fetcher(self, conn: socket.socket, credit: int) -> None:
        """Answer REQUEST frames: ``{"op": "list"}`` with a LISTING, ``{"op":

        "get", "path", "sizes"}`` with the package at ``path`` (relative to the
        store) from ``sizes`` bytes per file, until its ACK.
        """
        while True:
            kind, payload = recv_frame(conn)
            if kind == CREDIT:  # left over from the previous package
                credit += json.loads(payload)["bytes"]
                continue
            if kind != REQUEST:
                raise ConnectionError(f"expected REQUEST, got frame type {kind}")
            request = json.loads(payload)
            if self.store_dir is None:
                send_json(conn, LISTING, {"error": "this server has no store"})
                continue
            if request.get("op") == "list":
                found = listing(self.store_dir)
                send_json(
                    conn,
                    LISTING,
                    {
                        "head": found.head,
                        "packages": [p.to_json(self.store_dir) for p in found.packages],
                        "epochs": {str(e): info for e, info in found.epochs.items()},
                    },
                )
                continue
            try:
                path = _within(self.store_dir, request["path"])
                manifest = read_manifest(path)
            except (OSError, ValueError, KeyError) as e:
                send_json(conn, PACKAGE, {"error": f"{type(e).__name__}: {e}"})
                continue
            offsets = request.get("sizes") or [0] * len(manifest["files"])
            send_json(conn, PACKAGE, {"name": os.path.basename(path), "manifest": manifest})
            credit, _ = self._send_files(conn, path, manifest["files"], offsets, credit, None)
            while True:
                kind, payload = recv_frame(conn)
                if kind == CREDIT:
                    credit += json.loads(payload)["bytes"]
                elif kind == ACK:
                    break

    def _send_files(self, conn, path, files, offsets, credit, gen) -> tuple[int, bool]:
        """DATA frames of ``files`` from ``offsets``; returns the remaining
        credit and whether everything was sent (False: the package stopped
        being offered; ``gen`` None: a fetched package, never withdrawn)."""
        for i, (entry, offset) in enumerate(zip(files, offsets)):
            with open(os.path.join(path, entry["name"]), "rb") as f:
                f.seek(offset)
                while offset < entry["bytes"]:
                    if gen is not None and self._gen != gen:
                        return credit, False
                    chunk = f.read(min(self.chunk_bytes, entry["bytes"] - offset))
                    while credit < len(chunk):
                        kind, payload = recv_frame(conn)
                        if kind != CREDIT:
                            raise ConnectionError(f"expected CREDIT, got frame type {kind}")
                        credit += json.loads(payload)["bytes"]
                    send_frame(conn, DATA, DATA_HEADER.pack(i, offset), chunk)
                    with self._cond:
                        self.sent_bytes += HEADER.size + DATA_HEADER.size + len(chunk)
                    credit -= len(chunk)
                    offset += len(chunk)
        return credit, True


class PackageReceiver:
    """Receives the packages of the :class:`PackageServer` at ``address`` into
    ``spool`` (kept to the latest package).

    A delta package is acknowledged once complete and ``ready_count`` consumers
    marked it read (:func:`relax.distributed.weight_sync.store.mark_ready`), bounded by
    ``READY_WAIT_S``.
    """

    def __init__(self, address: str, spool: str, ready_count: int = 0, window: int = WINDOW_BYTES):
        host, port = address.rsplit(":", 1)
        self.address, self.spool, self.ready_count, self.window = address, spool, ready_count, window
        self._target = (host, int(port))
        self._id = uuid.uuid4().hex
        self._name: str | None = None  # package being (or last) received
        self._manifest: dict | None = None
        self._done = False
        self._closed = False
        self._sock: socket.socket | None = None
        threading.Thread(target=self._run, name="dws-tcp-receiver", daemon=True).start()

    def close(self) -> None:
        """Stop receiving and remove the spool."""
        self._closed = True
        sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        shutil.rmtree(self.spool, ignore_errors=True)

    def _run(self) -> None:
        delay = 0.1
        while not self._closed:
            try:
                with socket.create_connection(self._target) as sock:
                    self._sock = sock
                    self._hello(sock)
                    delay = 0.1
                    self._receive(sock)
            except OSError as e:
                if self._closed:
                    return
                logger.info(f"[delta] tcp connection to {self.address} lost ({e}); retrying in {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 5.0)

    def _dir(self) -> str:
        return os.path.join(self.spool, self._name)

    def _file(self, i: int) -> str:
        return os.path.join(self._dir(), self._manifest["files"][i]["name"])

    def _size(self, i: int) -> int:
        return os.path.getsize(self._file(i)) if os.path.exists(self._file(i)) else 0

    def _hello(self, sock: socket.socket) -> None:
        held = {}
        if self._name is not None:
            sizes = [self._size(i) for i in range(len(self._manifest["files"]))]
            held = {"name": self._name, "sizes": sizes, "done": self._done}
        send_json(sock, HELLO, {"id": self._id, "window": self.window, **held})

    def _receive(self, sock: socket.socket) -> None:
        out = None  # (file index, file being written)
        try:
            while True:
                kind, payload = recv_frame(sock)
                if kind == PACKAGE:
                    message = json.loads(payload)
                    if out is not None:
                        out[1].close()
                        out = None
                    if message["name"] != self._name:
                        self._start(message["name"], message["manifest"])
                    self._finish_if_complete(sock)  # a resumed package may already be complete
                elif kind == DATA:
                    i, offset = DATA_HEADER.unpack_from(payload)
                    if out is None or out[0] != i:
                        if out is not None:
                            out[1].close()
                        out = (i, open(self._file(i), "ab"))
                    f = out[1]
                    if f.tell() != offset:
                        raise ConnectionError(f"file {i}: data at offset {offset}, have {f.tell()} bytes")
                    f.write(memoryview(payload)[DATA_HEADER.size :])
                    f.flush()  # sizes reported on reconnect are what is on disk
                    send_json(sock, CREDIT, {"bytes": len(payload) - DATA_HEADER.size})
                    if f.tell() == self._manifest["files"][i]["bytes"]:
                        f.close()
                        out = None
                        self._finish_if_complete(sock)
        finally:
            if out is not None:
                out[1].close()

    def _start(self, name: str, manifest: dict) -> None:
        for entry in os.listdir(self.spool):
            shutil.rmtree(os.path.join(self.spool, entry))
        os.mkdir(os.path.join(self.spool, name))
        self._name, self._manifest, self._done = name, manifest, False

    def _finish_if_complete(self, sock: socket.socket) -> None:
        files = self._manifest["files"]
        if self._done or any(self._size(i) != f["bytes"] for i, f in enumerate(files)):
            return
        tmp = os.path.join(self._dir(), f".{MANIFEST}")
        with open(tmp, "w") as f:
            json.dump(self._manifest, f)
        os.rename(tmp, os.path.join(self._dir(), MANIFEST))  # the package is now sealed in the spool
        if self._manifest["kind"] == DELTA:
            deadline = time.monotonic() + READY_WAIT_S
            while count_ready(self._dir()) < self.ready_count and time.monotonic() < deadline:
                time.sleep(0.02)
        self._done = True
        send_json(sock, ACK, {"name": self._name})


class PackageFetcher:
    """Client of a :class:`PackageServer` with a store: lists it and fetches
    published packages into a local directory laid out like the store.

    Each request reconnects (with backoff, at most ``retries`` times) when the
    connection drops; a fetch resumes from the bytes already on disk.
    """

    def __init__(self, address: str, window: int = WINDOW_BYTES, retries: int = FETCH_RETRIES):
        host, port = address.rsplit(":", 1)
        self.address, self.window, self.retries = address, window, retries
        self._target = (host, int(port))
        self._sock: socket.socket | None = None

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def listing(self) -> Listing:
        reply = self._request(lambda sock: self._list(sock))
        return Listing(
            reply["head"],
            [Package.from_json(p) for p in reply["packages"]],
            {int(e): info for e, info in reply["epochs"].items()},
        )

    def fetch(self, rel: str, dest: str) -> str:
        """Bring the package at ``rel`` (relative to the store) to
        ``dest/rel``; returns that path.

        A package already complete there is not fetched again.
        """
        path = _within(dest, rel)
        if os.path.exists(os.path.join(path, MANIFEST)):
            return path
        os.makedirs(path, exist_ok=True)
        self._request(lambda sock: self._get(sock, rel, path))
        return path

    def _connection(self) -> socket.socket:
        if self._sock is None:
            sock = socket.create_connection(self._target)
            send_json(sock, HELLO, {"id": uuid.uuid4().hex, "window": self.window, "pull": True})
            self._sock = sock
        return self._sock

    def _request(self, call):
        delay = 0.1
        for attempt in range(self.retries + 1):
            try:
                return call(self._connection())
            except FileNotFoundError:  # answered by the server: not a transport error
                raise
            except OSError as e:
                self.close()
                if attempt == self.retries:
                    raise
                logger.info(f"[delta] tcp connection to {self.address} lost ({e}); retrying in {delay:.1f}s")
                time.sleep(delay)
                delay = min(delay * 2, 5.0)

    def _list(self, sock: socket.socket) -> dict:
        send_json(sock, REQUEST, {"op": "list"})
        kind, payload = recv_frame(sock)
        reply = json.loads(payload)
        if kind != LISTING or "error" in reply:
            raise FileNotFoundError(f"{self.address}: {reply.get('error', f'frame type {kind}')}")
        return reply

    def _get(self, sock: socket.socket, rel: str, path: str) -> None:
        files = sorted(n for n in os.listdir(path) if not n.startswith("."))
        sizes = None
        manifest_cache = os.path.join(path, f".{MANIFEST}.partial")
        if os.path.exists(manifest_cache):  # resuming: report what is on disk
            with open(manifest_cache) as f:
                cached = json.load(f)["files"]
            sizes = [os.path.getsize(os.path.join(path, e["name"])) if e["name"] in files else 0 for e in cached]
        send_json(sock, REQUEST, {"op": "get", "path": rel, "sizes": sizes})
        kind, payload = recv_frame(sock)
        message = json.loads(payload)
        if kind != PACKAGE or "error" in message:
            raise FileNotFoundError(f"{self.address}: {rel}: {message.get('error', f'frame type {kind}')}")
        manifest = message["manifest"]
        with open(manifest_cache, "w") as f:
            json.dump(manifest, f)
        entries = manifest["files"]
        remaining = sum(e["bytes"] for e in entries) - sum(sizes or [0])
        out = None
        try:
            while remaining > 0:
                kind, payload = recv_frame(sock)
                if kind != DATA:
                    raise ConnectionError(f"expected DATA, got frame type {kind}")
                i, offset = DATA_HEADER.unpack_from(payload)
                if out is None or out[0] != i:
                    if out is not None:
                        out[1].close()
                    out = (i, open(os.path.join(path, entries[i]["name"]), "ab"))
                f = out[1]
                if f.tell() != offset:
                    raise ConnectionError(f"file {i}: data at offset {offset}, have {f.tell()} bytes")
                f.write(memoryview(payload)[DATA_HEADER.size :])
                f.flush()
                n = len(payload) - DATA_HEADER.size
                remaining -= n
                send_json(sock, CREDIT, {"bytes": n})
        finally:
            if out is not None:
                out[1].close()
        os.rename(manifest_cache, os.path.join(path, MANIFEST))  # complete: the package is now readable
        send_json(sock, ACK, {"name": os.path.basename(path)})


def serve(argv=None) -> None:
    """``python -m relax.distributed.weight_sync.tcp serve --store-dir D.

    [--host H] [--port P]``: serve the published packages of a store to
    fetchers, independently of the trainer that writes it.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="relax.distributed.weight_sync.tcp serve")
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    server = PackageServer(args.host, args.port, store_dir=args.store_dir)
    logger.info(f"[delta] serving {args.store_dir} at {server.address} (plain TCP: trusted networks only)")
    while True:
        time.sleep(3600)


def main(argv=None) -> None:
    """``python -m relax.distributed.weight_sync.tcp ADDRESS SPOOL READY_COUNT
    PARENT_PID``: run a receiver in its own process (the SGLang scheduler loop
    would starve a receiver thread of the GIL); it exits, removing the spool,
    when process ``PARENT_PID`` is gone.

    ``... serve ...``: see :func:`serve`.
    """
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "serve":
        serve(argv[1:])
        return
    address, spool, ready_count, parent = argv
    PackageReceiver(address, spool, int(ready_count))
    while True:
        try:
            os.kill(int(parent), 0)
        except ProcessLookupError:
            break
        time.sleep(1.0)
    shutil.rmtree(spool, ignore_errors=True)


if __name__ == "__main__":
    main()
