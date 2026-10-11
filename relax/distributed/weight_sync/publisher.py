# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Training side of sparse delta weight sync: write and publish version
packages.

A :class:`WeightPublisher` lives on the trainer's global rank 0. It owns the
store epoch, the package being written and (tcp transport) the package server;
it holds no rollout engine handle and never talks to an engine. Whoever
installs the packages (the in-job sync, or an engine-side
:class:`~relax.distributed.weight_sync.installer.WeightInstaller`) only needs
the package and :meth:`install_meta`.
"""

import os
import time

from relax.distributed.weight_sync.store import (
    Package,
    PackageWriter,
    claim_epoch,
    count_ready,
    prune,
    read_head,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class WeightPublisher:
    """Writes version packages into ``store_dir`` under a newly claimed epoch.

    ``tcp_host``: also serve the packages over TCP (push to receivers with
    :meth:`offer`, pull by fetchers). ``resumed_from``: the published
    (epoch, version) this run's weights start from, recorded in the epoch.
    ``prune_grace_s``: keep packages that fell out of retention readable for
    that long (consumers that do not install inside the trainer's pause).
    """

    PREFETCH_WAIT_S = 10.0  # shared storage: bound on waiting for engines to read a package
    TCP_SEND_TIMEOUT_S = 300.0  # tcp: bound on sending a package to every engine

    def __init__(
        self,
        store_dir: str,
        tcp_host: str | None = None,
        tcp_port: int = 0,
        resumed_from: tuple[int, int] | None = None,
        prune_grace_s: float = 0.0,
    ):
        self.store_dir = store_dir
        self.prune_grace_s = prune_grace_s
        self.epoch = claim_epoch(store_dir, resumed_from)
        self.tcp_server = None
        if tcp_host is not None:
            from relax.distributed.weight_sync.tcp import PackageServer

            self.tcp_server = PackageServer(tcp_host, tcp_port, store_dir=store_dir)
        self.writing: PackageWriter | None = None  # package of the version being sent

    @property
    def address(self) -> str | None:
        return self.tcp_server.address if self.tcp_server is not None else None

    def head(self) -> dict | None:
        return read_head(self.store_dir)

    def begin(self, kind: str, version: int, base_version: int | None = None) -> PackageWriter:
        """Start the package of ``version``; buckets are added to
        :attr:`writing`."""
        if self.writing is not None:
            self.writing.abort()
        self.writing = PackageWriter(self.store_dir, self.epoch, version, kind, base_version)
        return self.writing

    def install_meta(self) -> dict:
        """Loader message installing the sealed package being written: from its
        temporary path on shared storage, from the engine's spool over tcp."""
        package = self.writing
        meta = {"action": "install", "kind": package.kind, "epoch": package.epoch, "version": package.version}
        path = package.seal()
        if self.tcp_server is None:
            meta["path"] = path
        return meta

    def offer(self, engines: int, readers: int) -> str:
        """Before the pause: over tcp, send the sealed package to ``engines``
        receivers (each acknowledges it once received and prefetched); on
        shared storage, give the ``readers`` TP ranks up to
        :attr:`PREFETCH_WAIT_S` to read it. Returns a log note."""
        path = self.writing.seal()
        start = time.monotonic()
        if self.tcp_server is not None:
            acked = self.tcp_server.send(path, engines, self.TCP_SEND_TIMEOUT_S)
            return f", sent to {acked}/{engines} engines in {time.monotonic() - start:.2f}s"
        ready = count_ready(path)
        while ready < readers and time.monotonic() - start < self.PREFETCH_WAIT_S:
            time.sleep(0.02)
            ready = count_ready(path)
        return f", prefetched {ready}/{readers} in {time.monotonic() - start:.2f}s"

    def finish(self, publish: bool) -> Package | None:
        """Publish (or drop) the package being written, then prune the store.

        Raises :class:`~relax.distributed.weight_sync.store.StaleEpochError`
        when a newer trainer took over the store.
        """
        package, self.writing = self.writing, None
        if package is None:
            return None
        if not publish:
            package.abort()
            return None
        published = package.publish()
        removed = prune(self.store_dir, self.epoch, self.prune_grace_s)
        logger.info(
            f"[delta] published {os.path.relpath(published.path, self.store_dir)} ({package.nbytes} B)"
            + (f", pruned {len(removed)}" if removed else "")
        )
        return published
