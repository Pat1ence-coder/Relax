# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Inference side of sparse delta weight sync: bring one rollout engine to a
published version on its own.

A :class:`WeightInstaller` talks to one SGLang engine (started with the delta
loader, see :data:`~relax.distributed.weight_sync.wire.LOADER_PATH`) and to a
package source; it needs neither the trainer nor its job. It asks the loader
which version it holds, plans the packages from there to the target
(:func:`~relax.distributed.weight_sync.store.plan_packages`), brings them next
to the engine before pausing it, installs them in order and resumes. A
rejected package (the loader then requires a full one) is retried from the
latest full package; an engine whose weights may be half written is left
paused rather than serving them. A bitwise mismatch found by ``verify`` is
reported, not retried: the full package it came from has already corrected
the weights.

Requires the Relax SGLang patch (loader errors on ``/update_weights_from_tensor``
are reported instead of stopping the scheduler).

The in-job sync (``DeviceDirectBackend``) still installs on its engines itself,
inside the trainer's pause window; an installer serves the engines that the
trainer does not drive (offline consumers, new or restarted engines, or an
external orchestrator).
"""

import json
import os
from dataclasses import dataclass, field
from typing import Protocol

from relax.distributed.weight_sync.store import FULL, Listing, Package, listing, plan_packages
from relax.distributed.weight_sync.wire import (
    LOADER_PATH,
    META_NAME,
    STATE_REPLY,
    VERIFY_MISMATCH,
    encode_meta,
    encode_named_cpu_tensors,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------- engine


class EngineEndpoint(Protocol):
    """The HTTP API of one SGLang engine that an installer uses."""

    def loader(self, meta: dict, weight_version: str | None = None, flush_cache: bool = False) -> tuple[bool, str]:
        """Send a metadata-only delta-loader message; returns (success,
        message)."""
        ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...


class HttpEngine:
    """:class:`EngineEndpoint` of the SGLang server at ``url`` with ``tp`` TP
    ranks."""

    def __init__(self, url: str, tp: int = 1, timeout: float = 600.0):
        self.url, self.tp, self.timeout = url.rstrip("/"), tp, timeout

    def loader(self, meta: dict, weight_version: str | None = None, flush_cache: bool = False) -> tuple[bool, str]:
        import requests

        payload = {
            # one copy per TP rank of the engine; a CPU tensor pickles by value
            "serialized_named_tensors": [encode_named_cpu_tensors([(META_NAME, encode_meta(meta, "cpu"))])] * self.tp,
            "load_format": LOADER_PATH,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        response = requests.post(f"{self.url}/update_weights_from_tensor", json=payload, timeout=self.timeout)
        try:
            message = str(response.json().get("message", ""))
        except ValueError:
            message = response.text
        return response.status_code == 200, message

    def _post(self, path: str) -> None:
        import requests

        requests.post(f"{self.url}{path}", json={}, timeout=self.timeout).raise_for_status()

    def pause(self) -> None:
        self._post("/pause_generation")

    def resume(self) -> None:
        self._post("/continue_generation")


# ---------------------------------------------------------------- package sources


class PackageSource(Protocol):
    """Where an installer finds packages."""

    def listing(self) -> Listing: ...

    def local_path(self, package: Package) -> str:
        """A path at which every TP rank of the engine can read ``package``."""
        ...


class SharedStoreSource:
    """Packages read in place from a store every engine node mounts."""

    def __init__(self, store_dir: str):
        self.store_dir = store_dir

    def listing(self) -> Listing:
        return listing(self.store_dir)

    def local_path(self, package: Package) -> str:
        return package.path if os.path.isabs(package.path) else os.path.join(self.store_dir, package.path)


class TcpSource:
    """Packages fetched from a
    :class:`~relax.distributed.weight_sync.tcp.PackageServer` into
    ``spool_dir`` on the engine node (laid out like the store)."""

    def __init__(self, address: str, spool_dir: str):
        from relax.distributed.weight_sync.tcp import PackageFetcher

        self.spool_dir = spool_dir
        self._fetcher = PackageFetcher(address)

    def listing(self) -> Listing:
        return self._fetcher.listing()

    def local_path(self, package: Package) -> str:
        return self._fetcher.fetch(package.path, self.spool_dir)

    def close(self) -> None:
        self._fetcher.close()


# ---------------------------------------------------------------- installer


@dataclass(frozen=True)
class LoaderState:
    """What the engine's delta loader holds."""

    epoch: int | None
    version: int | None
    must_full: bool
    error: str | None = None

    @property
    def key(self) -> tuple[int, int] | None:
        if self.must_full or self.epoch is None or self.version is None:
            return None
        return self.epoch, self.version


@dataclass
class InstallResult:
    ok: bool
    state: LoaderState  # after the install
    installed: list[str] = field(default_factory=list)  # paths of the installed packages, in order
    verified: bool = False  # the result was compared bitwise with a full package of the target
    abandoned: bool = False  # before the install the engine served a version off the target's lineage
    paused: bool = False  # must stay paused: the weights may be half written (resumed otherwise, if it paused)
    error: str | None = None


class InstallError(RuntimeError):
    pass


def _parse_state(message: str) -> LoaderState:
    at = message.find(STATE_REPLY)
    if at < 0:
        raise InstallError(f"engine did not answer the state query through the delta loader: {message[:500]}")
    state, _ = json.JSONDecoder().raw_decode(message[at + len(STATE_REPLY) :])
    return LoaderState(state["epoch"], state["version"], bool(state["must_full"]), state.get("error"))


def _nbytes(packages: list[Package]) -> int:
    return sum(p.nbytes for p in packages)


class WeightInstaller:
    """Installs published versions from ``source`` on ``engine``.

    ``max_attempts``: installs of the target before giving up (each retry
    starts from a full package). ``weight_version``: label reported by the
    engine for an installed (epoch, version); by default the version, as the
    in-job sync reports it.
    """

    def __init__(self, engine: EngineEndpoint, source: PackageSource, max_attempts: int = 2, weight_version=None):
        self.engine, self.source, self.max_attempts = engine, source, max_attempts
        self.weight_version = weight_version or (lambda epoch, version: str(version))

    def current(self) -> LoaderState:
        ok, message = self.engine.loader({"action": "state"})
        if ok:
            raise InstallError(
                "state query succeeded: the engine passed it to model.load_weights, not the delta loader"
            )
        return _parse_state(message)

    def plan(
        self, found: Listing, have: tuple[int, int] | None, target: tuple[int, int], prefer_deltas: bool = False
    ) -> list[Package]:
        """Packages from ``have`` to ``target``: the delta chain from ``have``,
        or the latest full package and its chain, whichever is smaller (the
        delta chain whenever there is one with ``prefer_deltas``)."""
        from_full = plan_packages(found.packages, None, target)
        if have is None or have == target:
            return [] if have == target else from_full
        try:
            from_have = plan_packages(found.packages, have, target)
        except FileNotFoundError:
            return from_full
        return from_have if prefer_deltas or _nbytes(from_have) <= _nbytes(from_full) else from_full

    def install(
        self, target: tuple[int, int] | None = None, paused: bool = False, verify: bool = False
    ) -> InstallResult:
        """Bring the engine to ``target`` (default: ``HEAD``).

        ``paused``: the caller already paused the engine (and resumes it).
        ``verify``: when the target also has a full package and was reached
        by deltas, load that full package comparing it bitwise with the
        result.
        """
        found = self.source.listing()
        target = target or found.head_key
        before = self.current()
        if target is None:
            return InstallResult(False, before, error="nothing published yet")
        abandoned = before.key is not None and not found.on_lineage(before.key, target)
        have = before.key
        error = None
        installed: list[str] = []
        did_pause = False
        try:
            for attempt in range(self.max_attempts):
                # verify checks the deltas: reach the target by them when the engine holds a base
                packages = self.plan(found, None if attempt else have, target, prefer_deltas=verify)
                check = None
                if verify and (not packages or packages[-1].kind != FULL):
                    check = next((p for p in found.packages if p.key == target and p.kind == FULL), None)
                if not packages and check is None:
                    return InstallResult(True, before, abandoned=abandoned)
                # everything is local before the engine stops generating
                paths = [self.source.local_path(p) for p in packages]
                check_path = self.source.local_path(check) if check is not None else None
                if not paused and not did_pause:
                    self.engine.pause()
                    did_pause = True
                error = self._install_all(packages, paths, installed)
                if error is None and check is not None:
                    error = self._install_one(check, check_path, verify=True)
                    if error is None:
                        installed.append(check_path)
                if error is None:
                    state = self.current()
                    if state.key != target:
                        error = f"engine holds {state.key} after installing {target}"
                if error is None:
                    if did_pause:
                        self.engine.resume()
                    return InstallResult(True, state, installed, check is not None, abandoned)
                if VERIFY_MISMATCH in error:
                    break  # the full package already corrected the weights: report, do not retry
                logger.warning(f"[delta] installing e{target[0]}:v{target[1]} failed (attempt {attempt + 1}): {error}")
                found = self.source.listing()  # packages may have been pruned or superseded meanwhile
        except Exception as e:  # noqa: BLE001 - reported in the result
            error = f"{type(e).__name__}: {e}"
        state = self._state_or_unknown()
        if did_pause and not state.must_full:
            self.engine.resume()
        return InstallResult(False, state, installed, False, abandoned, state.must_full, error)

    def _install_all(self, packages: list[Package], paths: list[str], installed: list[str]) -> str | None:
        for package, path in zip(packages, paths):
            error = self._install_one(package, path)
            if error is not None:
                return error
            installed.append(path)
        return None

    def _install_one(self, package: Package, path: str, verify: bool = False) -> str | None:
        meta = {"action": "install", "kind": package.kind, "epoch": package.epoch, "version": package.version}
        meta["path"] = path
        if verify:
            meta["verify"] = True
        label = self.weight_version(package.epoch, package.version)
        ok, message = self.engine.loader(meta, label, flush_cache=package.kind != FULL)
        if not ok:
            return f"{os.path.basename(path)}: {message[:2000]}"
        if package.kind == FULL:
            reset = {"action": "reset", "version": package.version, "epoch": package.epoch}
            ok, message = self.engine.loader(reset, None, flush_cache=True)
            if not ok:
                return f"reset after {os.path.basename(path)}: {message[:2000]}"
        return None

    def _state_or_unknown(self) -> LoaderState:
        try:
            return self.current()
        except Exception as e:  # noqa: BLE001 - engine unreachable
            return LoaderState(None, None, True, f"{type(e).__name__}: {e}")
