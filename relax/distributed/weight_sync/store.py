# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared-storage version packages for sparse delta weight sync.

A package holds the buckets of one weight version exactly as they would be
sent over NCCL (see :mod:`relax.distributed.weight_sync.wire`), one safetensors file per
bucket::

    <store>/e000001/v000002.delta/{manifest.json, b00000.safetensors, ...}
    <store>/e000001/v000001.full/...
    <store>/HEAD                       # latest published package

``epoch`` counts trainer runs on the same store (the trainer's weight version
restarts at 1 after a restart); ``<store>/eNNNNNN/EPOCH.json`` records which
published version, if any, the run resumed from, so a consumer can tell whether
the version it holds was abandoned (:meth:`Listing.on_head_lineage`). A package
is written under a ``.tmp-*`` name, fsynced and renamed into place, so readers
never see a partial package; once published it is never modified. One writer
(trainer rank 0) per store: a writer whose epoch is no longer the newest one
claimed is refused at publish (:class:`StaleEpochError`).

Consumers are not registered anywhere: they read ``HEAD`` and the package list
and :func:`plan` their own way there; :func:`prune` can leave packages that
fell out of retention readable for a grace period.

Kept free of Ray/GPU dependencies: used by the trainer, the SGLang delta loader,
the engine-side installer and the offline consumer.
"""

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field

import torch
from safetensors.torch import load, save_file


MANIFEST = "manifest.json"
HEAD = "HEAD"
READY = ".ready-"  # marker a consumer leaves in a sealed package once it has read it (prefetch); removed on publish
FULL, DELTA = "full", "delta"
KEEP_FULL = 2  # retention: the latest KEEP_FULL full packages and everything after the oldest of them
EPOCH_INFO = "EPOCH.json"  # in an epoch directory: {epoch, created_at, resumed_from: [epoch, version] | null}
PUBLISHED = ".published"  # in a package: {"time": publish time}, for the prune grace period


class StaleEpochError(RuntimeError):
    """A newer epoch was claimed on the store; this writer must stop
    publishing."""


@dataclass(frozen=True)
class Package:
    epoch: int
    version: int
    kind: str
    base_version: int | None
    path: str  # absolute, or relative to the store root when listed by a remote store (tcp)
    nbytes: int = 0

    @property
    def key(self) -> tuple[int, int]:
        return self.epoch, self.version

    def to_json(self, store: str | None = None) -> dict:
        path = os.path.relpath(self.path, store) if store is not None else self.path
        return {
            "epoch": self.epoch,
            "version": self.version,
            "kind": self.kind,
            "base_version": self.base_version,
            "path": path,
            "nbytes": self.nbytes,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Package":
        return cls(d["epoch"], d["version"], d["kind"], d["base_version"], d["path"], d.get("nbytes", 0))


@dataclass
class Listing:
    """What a consumer sees of a store: ``HEAD``, the published packages and
    the epoch records."""

    head: dict | None
    packages: list[Package]
    epochs: dict[int, dict] = field(default_factory=dict)

    @property
    def head_key(self) -> tuple[int, int] | None:
        return (self.head["epoch"], self.head["version"]) if self.head else None

    def on_head_lineage(self, key: tuple[int, int]) -> bool:
        """Whether version ``key`` is an ancestor of (or is) the ``HEAD``
        version: False for versions of an abandoned run, e.g. those after the
        checkpoint a newer epoch resumed from."""
        head = self.head_key
        return head is not None and self.on_lineage(key, head)

    def on_lineage(self, key: tuple[int, int], tip: tuple[int, int]) -> bool:
        """Whether version ``key`` is an ancestor of (or is) version ``tip``,
        following ``resumed_from`` across epochs."""
        epoch, limit = tip
        while True:
            if key[0] == epoch:
                return key[1] <= limit
            resumed = (self.epochs.get(epoch) or {}).get("resumed_from")
            if not resumed or resumed[0] >= epoch:
                return False
            epoch, limit = resumed


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _epoch_dir(store: str, epoch: int) -> str:
    return os.path.join(store, f"e{epoch:06d}")


def _epochs(store: str) -> list[int]:
    if not os.path.isdir(store):
        return []
    return sorted(int(n[1:]) for n in os.listdir(store) if n.startswith("e") and n[1:].isdigit())


def _write_json(path: str, obj: dict) -> None:
    """Atomically replace ``path`` with ``obj``."""
    tmp = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.{uuid.uuid4().hex[:8]}")
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(os.path.dirname(path))


def claim_epoch(store: str, resumed_from: tuple[int, int] | None = None) -> int:
    """Create the directory of a new epoch (one more than any existing).

    ``resumed_from``: the published (epoch, version) whose weights this run
    starts from (e.g. the version exported at the checkpoint it loaded);
    None for a run unrelated to earlier epochs.
    """
    os.makedirs(store, exist_ok=True)
    while True:
        epoch = max(_epochs(store), default=0) + 1
        try:
            os.mkdir(_epoch_dir(store, epoch))
        except FileExistsError:
            continue
        _fsync_dir(store)
        info = {
            "epoch": epoch,
            "created_at": time.time(),
            "resumed_from": list(resumed_from) if resumed_from else None,
        }
        _write_json(os.path.join(_epoch_dir(store, epoch), EPOCH_INFO), info)
        return epoch


def latest_epoch(store: str) -> int | None:
    return max(_epochs(store), default=None)


def read_epoch_info(store: str, epoch: int) -> dict:
    """The record written by :func:`claim_epoch` (empty if missing)."""
    try:
        with open(os.path.join(_epoch_dir(store, epoch), EPOCH_INFO)) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def read_head(store: str) -> dict | None:
    """``{epoch, version, kind, path}`` of the latest published package."""
    try:
        with open(os.path.join(store, HEAD)) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def listing(store: str) -> Listing:
    return Listing(read_head(store), list_packages(store), {e: read_epoch_info(store, e) for e in _epochs(store)})


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
        """Rename the package into place and point ``HEAD`` at it.

        Raises :class:`StaleEpochError` (and drops the package) when a newer
        epoch has been claimed on the store meanwhile: a trainer that outlived
        its replacement must not move ``HEAD`` back.
        """
        self.seal()
        newest = latest_epoch(self.store)
        if newest != self.epoch:
            self.abort()
            raise StaleEpochError(f"epoch {self.epoch} of {self.store} was superseded by epoch {newest}")
        for name in os.listdir(self.tmp):
            if name.startswith(READY):
                os.remove(os.path.join(self.tmp, name))
        with open(os.path.join(self.tmp, PUBLISHED), "w") as f:
            json.dump({"time": time.time()}, f)
        final = os.path.join(_epoch_dir(self.store, self.epoch), self.name)
        os.rename(self.tmp, final)  # fails if a package of that name already exists
        _fsync_dir(os.path.dirname(final))
        rel = os.path.relpath(final, self.store)
        _write_json(
            os.path.join(self.store, HEAD),
            {"epoch": self.epoch, "version": self.version, "kind": self.kind, "path": rel},
        )
        return Package(self.epoch, self.version, self.kind, self.base_version, final, self.nbytes)

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
            nbytes = sum(f["bytes"] for f in m["files"])
            packages.append(Package(m["epoch"], m["version"], m["kind"], m["base_version"], path, nbytes))
    return sorted(packages, key=lambda p: (p.key, p.kind != FULL))


def plan(store: str, have: tuple[int, int] | None = None, target: tuple[int, int] | None = None) -> list[Package]:
    """Packages that bring a consumer at ``have`` (or nothing) to ``target``
    (default: the latest version): the delta chain from ``have`` if it is
    complete, else the latest full package at or before ``target`` followed
    by its delta chain."""
    return plan_packages(list_packages(store), have, target, store)


def plan_packages(
    packages: list[Package],
    have: tuple[int, int] | None = None,
    target: tuple[int, int] | None = None,
    where: str = "the store",
) -> list[Package]:
    """:func:`plan` over an already listed set of packages (e.g. one listed
    by a remote store)."""
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
    raise FileNotFoundError(f"no full package with a complete delta chain to e{target[0]}:v{target[1]} in {where}")


def _published_at(path: str) -> float:
    try:
        with open(os.path.join(path, PUBLISHED)) as f:
            return json.load(f)["time"]
    except (OSError, ValueError, KeyError):
        return os.path.getmtime(os.path.join(path, MANIFEST))


def prune(store: str, current_epoch: int, grace_s: float = 0.0) -> list[str]:
    """Delete packages older than the KEEP_FULL-th latest full package, and
    leftover temporary directories of other epochs.

    With ``grace_s`` a package is only deleted once it has been out of
    retention for that long (counted from the publication of the full package
    that pushed it out), so a consumer that planned a path through it just
    before can still read it. Returns removed paths.
    """
    packages = list_packages(store)
    fulls = [p for p in packages if p.kind == FULL]
    removed = []
    now = time.time()

    def redundant(p: Package, f: Package) -> bool:
        # a delta to a kept full version is redundant with it
        return p.key < f.key or (p.key == f.key and p.kind == DELTA)

    if len(fulls) >= KEEP_FULL:
        for p in packages:
            if not redundant(p, fulls[-KEEP_FULL]):
                continue
            if grace_s > 0:
                # p went out of retention when the (KEEP_FULL-1)-th full after the first full that makes it
                # redundant was published
                first = next(i for i, f in enumerate(fulls) if redundant(p, f))
                if now - _published_at(fulls[first + KEEP_FULL - 1].path) < grace_s:
                    continue
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
        if int(e[1:]) < current_epoch and set(os.listdir(edir)) <= {EPOCH_INFO}:
            shutil.rmtree(edir)
            removed.append(edir)
    return removed
