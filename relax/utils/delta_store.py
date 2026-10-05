# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared-storage version packages for sparse delta weight sync.

A package holds the buckets of one weight version exactly as they would be
sent over NCCL (see :mod:`relax.utils.delta_wire`), one safetensors file per
bucket::

    <store>/e000001/v000002.delta/{manifest.json, b00000.safetensors, ...}
    <store>/e000001/v000001.full/...
    <store>/HEAD                       # latest published package

``epoch`` counts trainer runs on the same store (the trainer's weight version
restarts at 1 after a restart). A package is written under a ``.tmp-*`` name,
fsynced and renamed into place, so readers never see a partial package; once
published it is never modified. One writer (trainer rank 0) per store.

Kept free of Ray/GPU dependencies: used by the trainer, the SGLang delta loader
and the offline consumer.
"""

import json
import os
import shutil
import uuid
from dataclasses import dataclass

import torch
from safetensors.torch import load, save_file


MANIFEST = "manifest.json"
HEAD = "HEAD"
READY = ".ready-"  # marker a consumer leaves in a sealed package once it has read it (prefetch); removed on publish
FULL, DELTA = "full", "delta"
KEEP_FULL = 2  # retention: the latest KEEP_FULL full packages and everything after the oldest of them


@dataclass(frozen=True)
class Package:
    epoch: int
    version: int
    kind: str
    base_version: int | None
    path: str

    @property
    def key(self) -> tuple[int, int]:
        return self.epoch, self.version


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _epoch_dir(store: str, epoch: int) -> str:
    return os.path.join(store, f"e{epoch:06d}")


def claim_epoch(store: str) -> int:
    """Create the directory of a new epoch (one more than any existing)."""
    os.makedirs(store, exist_ok=True)
    while True:
        epochs = [int(n[1:]) for n in os.listdir(store) if n.startswith("e") and n[1:].isdigit()]
        epoch = max(epochs, default=0) + 1
        try:
            os.mkdir(_epoch_dir(store, epoch))
        except FileExistsError:
            continue
        _fsync_dir(store)
        return epoch


class PackageWriter:
    """Write one package bucket by bucket; :meth:`publish` makes it visible."""

    def __init__(self, store: str, epoch: int, version: int, kind: str, base_version: int | None = None):
        assert kind in (FULL, DELTA)
        self.store, self.epoch, self.version, self.kind, self.base_version = store, epoch, version, kind, base_version
        self.name = f"v{version:06d}.{kind}"
        self.tmp = os.path.join(_epoch_dir(store, epoch), f".tmp-{self.name}-{uuid.uuid4().hex[:8]}")
        os.mkdir(self.tmp)
        self.files: list[dict] = []
        self.sealed = False

    def add_bucket(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        assert not self.sealed
        name = f"b{len(self.files):05d}.safetensors"
        path = os.path.join(self.tmp, name)
        save_file({n: t.detach().contiguous().cpu() for n, t in named_tensors}, path)
        with open(path, "rb+") as f:
            os.fsync(f.fileno())
        self.files.append({"name": name, "bytes": os.path.getsize(path)})

    @property
    def nbytes(self) -> int:
        return sum(f["bytes"] for f in self.files)

    def seal(self) -> str:
        """Write the manifest; the package can then be read at :attr:`tmp`
        (consumers install it there before it is published)."""
        if self.sealed:
            return self.tmp
        manifest = {
            "epoch": self.epoch,
            "version": self.version,
            "kind": self.kind,
            "base_version": self.base_version,
            "files": self.files,
        }
        tmp_manifest = os.path.join(self.tmp, f".{MANIFEST}")
        with open(tmp_manifest, "w") as f:
            json.dump(manifest, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_manifest, os.path.join(self.tmp, MANIFEST))  # a visible manifest is complete
        _fsync_dir(self.tmp)
        self.sealed = True
        return self.tmp

    def publish(self) -> Package:
        self.seal()
        for name in os.listdir(self.tmp):
            if name.startswith(READY):
                os.remove(os.path.join(self.tmp, name))
        final = os.path.join(_epoch_dir(self.store, self.epoch), self.name)
        os.rename(self.tmp, final)  # fails if a package of that name already exists
        _fsync_dir(os.path.dirname(final))
        rel = os.path.relpath(final, self.store)
        head_tmp = os.path.join(self.store, f".{HEAD}.{uuid.uuid4().hex[:8]}")
        with open(head_tmp, "w") as f:
            json.dump({"epoch": self.epoch, "version": self.version, "kind": self.kind, "path": rel}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(head_tmp, os.path.join(self.store, HEAD))
        _fsync_dir(self.store)
        return Package(self.epoch, self.version, self.kind, self.base_version, final)

    def abort(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


def read_manifest(path: str) -> dict:
    with open(os.path.join(path, MANIFEST)) as f:
        return json.load(f)


def read_buckets(path: str, device: str | torch.device = "cpu"):
    """Yield each bucket of a package as a list of named tensors
    (``__dws_meta__`` first)."""
    manifest = read_manifest(path)
    for entry in manifest["files"]:
        file = os.path.join(path, entry["name"])
        # one sequential read: mmap (safetensors.load_file) faults pages in one by one, far slower on shared storage
        with open(file, "rb") as f:
            data = f.read()
        if len(data) != entry["bytes"]:
            raise ValueError(f"{file}: {len(data)} bytes, manifest says {entry['bytes']}")
        tensors = {name: t.to(device) for name, t in load(data).items()}
        del data
        yield sorted(tensors.items(), key=lambda kv: (not kv[0].startswith("__dws_meta__"), kv[0]))


def sealed_package(epoch_dir: str, version: int, kind: str) -> str | None:
    """Path of the sealed, not yet published package of ``version`` in
    ``epoch_dir`` (written by the trainer), if any."""
    prefix = f".tmp-v{version:06d}.{kind}-"
    for name in os.listdir(epoch_dir):
        if name.startswith(prefix) and os.path.exists(os.path.join(epoch_dir, name, MANIFEST)):
            return os.path.join(epoch_dir, name)
    return None


def mark_ready(path: str) -> None:
    # random name: unique per consumer without recording which host or process it is
    open(os.path.join(path, f"{READY}{uuid.uuid4().hex}"), "w").close()


def count_ready(path: str) -> int:
    return sum(name.startswith(READY) for name in os.listdir(path))


def list_packages(store: str) -> list[Package]:
    """All published packages, ordered by (epoch, version)."""
    packages = []
    if not os.path.isdir(store):
        return packages
    for e in sorted(os.listdir(store)):
        if not (e.startswith("e") and e[1:].isdigit()):
            continue
        edir = os.path.join(store, e)
        for n in os.listdir(edir):
            path = os.path.join(edir, n)
            if n.startswith(".") or not os.path.exists(os.path.join(path, MANIFEST)):
                continue
            m = read_manifest(path)
            packages.append(Package(m["epoch"], m["version"], m["kind"], m["base_version"], path))
    return sorted(packages, key=lambda p: (p.key, p.kind != FULL))


def plan(store: str, have: tuple[int, int] | None = None, target: tuple[int, int] | None = None) -> list[Package]:
    """Packages that bring a consumer at ``have`` (or nothing) to ``target``
    (default: the latest version): the delta chain from ``have`` if it is
    complete, else the latest full package at or before ``target`` followed
    by its delta chain."""
    packages = list_packages(store)
    if not packages:
        return []
    if target is None:
        target = max(p.key for p in packages)
    epoch = target[0]
    deltas = {p.version: p for p in packages if p.epoch == epoch and p.kind == DELTA}

    def chain(start: int) -> list[Package] | None:
        out, v = [], start
        while v < target[1]:
            nxt = deltas.get(v + 1)
            if nxt is None or nxt.base_version != v:
                return None
            out.append(nxt)
            v += 1
        return out

    if have is not None and have[0] == epoch and have[1] <= target[1]:
        found = chain(have[1])
        if found is not None:
            return found
    fulls = [p for p in packages if p.epoch == epoch and p.kind == FULL and p.version <= target[1]]
    for full in reversed(fulls):
        found = chain(full.version)
        if found is not None:
            return [full, *found]
    raise FileNotFoundError(f"no full package with a complete delta chain to e{target[0]}:v{target[1]} in {store}")


def prune(store: str, current_epoch: int) -> list[str]:
    """Delete packages older than the KEEP_FULL-th latest full package, and
    leftover temporary directories of other epochs.

    Returns removed paths.
    """
    packages = list_packages(store)
    fulls = [p for p in packages if p.kind == FULL]
    removed = []
    if len(fulls) >= KEEP_FULL:
        keep_from = fulls[-KEEP_FULL].key
        for p in packages:
            # a delta to the oldest kept full version is redundant with it
            if p.key < keep_from or (p.key == keep_from and p.kind == DELTA):
                shutil.rmtree(p.path)
                removed.append(p.path)
    for e in os.listdir(store):
        if not (e.startswith("e") and e[1:].isdigit()):
            continue
        edir = os.path.join(store, e)
        for n in os.listdir(edir):
            if n.startswith(".tmp-") and int(e[1:]) != current_epoch:
                shutil.rmtree(os.path.join(edir, n))
                removed.append(os.path.join(edir, n))
        if int(e[1:]) < current_epoch and not os.listdir(edir):
            os.rmdir(edir)
            removed.append(edir)
    return removed
