# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Immutable layout publication, retained crash evidence and bounded
recovery."""

import errno
import json
import multiprocessing
import os
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.distributed.weight_sync import DeltaCodecError
from relax.distributed.weight_sync.codec.format import content_hash
from relax.distributed.weight_sync.storage import (
    PosixAccessPolicy,
    PosixArtifactStore,
    PosixDeployment,
    StorageDeploymentError,
    initialize_namespace,
    open_deployed_store,
)
from relax.distributed.weight_sync.storage import files as file_io
from relax.distributed.weight_sync.storage import layout as layout_io
from tests.distributed.weight_sync.test_storage_deployment import LIMITS, STORAGE, initialize, physical_root
from tests.distributed.weight_sync.test_storage_deployment import configured as configured


def _reader(config: PosixDeployment) -> PosixDeployment:
    return replace(config, access="reader", control_root=None, control_volume=None)


def _state(root: Path) -> dict[str, tuple[object, ...]]:
    return {
        str(path.relative_to(root)): (
            path.lstat().st_ino,
            path.lstat().st_mode,
            path.lstat().st_uid,
            path.lstat().st_gid,
            path.read_bytes() if path.is_file() and not path.is_symlink() else None,
        )
        for path in root.rglob("*")
    }


def _crash_initialization(config: PosixDeployment, phase: str) -> None:
    mkdir, chown, chmod, sync, link = os.mkdir, os.fchown, os.fchmod, os.fsync, os.link

    def crash_mkdir(path: str, *args: object, **kwargs: object) -> None:
        mkdir(path, *args, **kwargs)
        if (phase == "layout_mkdir" and str(path).startswith("layout-")) or (
            phase == "child_mkdir" and path == "chunks"
        ):
            os._exit(93)

    def crash_chown(*args: object, **kwargs: object) -> None:
        chown(*args, **kwargs)
        if phase == "chown":
            os._exit(93)

    def crash_chmod(*args: object, **kwargs: object) -> None:
        chmod(*args, **kwargs)
        if phase == "chmod":
            os._exit(93)

    def crash_sync(descriptor: int) -> None:
        sync(descriptor)
        path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        if (
            (phase == "child_sync" and path.name == "chunks")
            or (phase == "layout_sync" and path.name.startswith("layout-"))
            or (phase == "file_sync" and stat.S_ISREG(os.fstat(descriptor).st_mode))
            or (
                phase == "prepare_parent_sync"
                and path == config.artifact_root
                and any(config.artifact_root.glob("layout-*"))
                and not (config.artifact_root / "namespace.json").exists()
            )
            or (
                phase == "publish_parent_sync"
                and path == config.artifact_root
                and (config.artifact_root / "namespace.json").exists()
            )
        ):
            os._exit(93)

    def crash_link(*args: object, **kwargs: object) -> None:
        if phase == "before_link":
            os._exit(93)
        link(*args, **kwargs)
        if phase == "after_link":
            os._exit(93)

    os.mkdir, os.fchown, os.fchmod, os.fsync, os.link = crash_mkdir, crash_chown, crash_chmod, crash_sync, crash_link
    initialize(config)
    os._exit(94)


@pytest.mark.parametrize(
    "phase",
    [
        "layout_mkdir",
        "child_mkdir",
        "chown",
        "chmod",
        "child_sync",
        "layout_sync",
        "prepare_parent_sync",
        "file_sync",
        "before_link",
        "after_link",
        "publish_parent_sync",
    ],
)
def test_interrupted_initialization_preserves_leftovers_and_recovers(configured: PosixDeployment, phase: str) -> None:
    configured.artifact_root.chmod(0o750)
    config = replace(configured, access_policy=PosixAccessPolicy(shared_group_id=os.getegid()))
    process = multiprocessing.get_context("fork").Process(target=_crash_initialization, args=(config, phase))
    process.start()
    process.join(15)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("layout initialization crash fixture hung")
    assert process.exitcode == 93
    root = config.artifact_root
    published = (root / "namespace.json").exists()
    assert published == (phase in ("after_link", "publish_parent_sync"))
    before = _state(root)
    if not published:
        with pytest.raises(StorageDeploymentError, match="namespace_missing"):
            open_deployed_store(_reader(config), limits=LIMITS, storage_limits=STORAGE)
        assert _state(root) == before
    initialize(config)
    after = _state(root)
    assert all(after[name] == value for name, value in before.items())
    assert len(list(root.glob("layout-*"))) == (1 if published else 2)
    for directory in (physical_root(config), *(physical_root(config) / name for name in file_io.ARTIFACT_DIRECTORIES)):
        assert directory.stat().st_mode & 0o777 == 0o750
        assert directory.stat().st_gid == os.getegid()
    initialize(config)
    assert _state(root) == after
    with open_deployed_store(_reader(config), limits=LIMITS, storage_limits=STORAGE) as store:
        assert "atomic_directory_noreplace" not in store.capabilities().requirements


@pytest.mark.parametrize("failure", ["permissions", "file_sync", "link", "parent_sync"])
def test_initialization_failure_can_retry_without_overwrite(
    configured: PosixDeployment, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    configured.artifact_root.chmod(0o750)
    config = replace(configured, access_policy=PosixAccessPolicy(shared_group_id=os.getegid()))
    sync = os.fsync

    def denied(*args: object, **kwargs: object) -> None:
        raise PermissionError("fixture denied")

    def failed_sync(descriptor: int) -> None:
        regular = stat.S_ISREG(os.fstat(descriptor).st_mode)
        if (failure == "file_sync" and regular) or (
            failure == "parent_sync" and not regular and (config.artifact_root / "namespace.json").exists()
        ):
            raise OSError("fixture sync failure")
        sync(descriptor)

    with monkeypatch.context() as patch:
        if failure == "permissions":
            patch.setattr(os, "fchown", denied)
        elif failure == "link":
            patch.setattr(os, "link", denied)
        else:
            patch.setattr(os, "fsync", failed_sync)
        with pytest.raises(OSError):
            initialize(config)
    before = _state(config.artifact_root)
    initialize(config)
    after = _state(config.artifact_root)
    assert all(after[name] == value for name, value in before.items())


def test_layout_name_collision_never_adopts_existing_directory(
    configured: PosixDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = configured.artifact_root / ("layout-" + "a" * 32)
    existing.mkdir(mode=0o700)
    before = existing.stat()
    original = layout_io.uuid.uuid4
    choices = iter([SimpleNamespace(hex="a" * 32)])
    monkeypatch.setattr(layout_io.uuid, "uuid4", lambda: next(choices, None) or original())
    initialize(configured)
    assert physical_root(configured) != existing
    assert (existing.stat().st_ino, existing.stat().st_mode) == (before.st_ino, before.st_mode)
    assert not list(existing.iterdir())


def test_descriptor_publication_conflict_never_replaces_existing_file(
    configured: PosixDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = configured.artifact_root / "namespace.json"
    other = replace(configured.namespace, namespace_id="other").to_bytes()
    original = os.link
    installed = []

    def competing_link(*args: object, **kwargs: object) -> None:
        with marker.open("xb") as stream:
            stream.write(other)
        installed.append(marker.stat().st_ino)
        original(*args, **kwargs)

    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(DeltaCodecError, match="immutable object conflict|object length mismatch"):
        initialize(configured)
    assert marker.read_bytes() == other and marker.stat().st_ino == installed[0]
    with pytest.raises(StorageDeploymentError, match="namespace_binding"):
        initialize(configured)


@pytest.mark.parametrize("kind", ["file", "symlink", "nonempty_directory", "legacy_directory"])
def test_uninitialized_namespace_refuses_unknown_contents_without_cleanup(
    configured: PosixDeployment, kind: str
) -> None:
    root = configured.artifact_root
    outside = root.parent / "untouched"
    outside.mkdir(mode=0o700)
    (outside / "sentinel").write_bytes(b"keep")
    prepared = root / ("chunks" if kind == "legacy_directory" else ".tmp-dir-chunks")
    if kind == "file":
        prepared.write_bytes(b"keep")
    elif kind == "symlink":
        prepared.symlink_to(outside, target_is_directory=True)
    else:
        prepared.mkdir(mode=0o700)
        (prepared / "sentinel").write_bytes(b"keep")
    before = _state(root)
    with pytest.raises(DeltaCodecError):
        initialize(configured)
    after = _state(root)
    assert all(after[name] == value for name, value in before.items())
    assert (outside / "sentinel").read_bytes() == b"keep"
    assert not (root / "namespace.json").exists()
    assert not list(root.glob("layout-*"))


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_and_new_layouts_support_immutable_writes_and_readers(
    configured: PosixDeployment, version: int
) -> None:
    if version == 1:
        with PosixArtifactStore(configured.artifact_root, writable=True) as store:
            store.put_object("namespace.json", configured.namespace.to_bytes())
    initialize(configured)
    marker = configured.artifact_root / "namespace.json"
    before = marker.read_bytes(), marker.stat().st_ino
    payload = b"new immutable payload"
    key = "chunks/" + content_hash(payload)
    with open_deployed_store(configured, limits=LIMITS, storage_limits=STORAGE) as store:
        store.finish_recovery()
        store.put_immutable(key, payload, expected_hash=content_hash(payload))
        usage = store._objects, store._bytes
    initialize(configured)
    with open_deployed_store(configured, limits=LIMITS, storage_limits=STORAGE) as store:
        assert (store._objects, store._bytes) == usage
    with open_deployed_store(_reader(configured), limits=LIMITS, storage_limits=STORAGE) as store:
        assert store.read_object(key, 64) == payload
    assert (marker.read_bytes(), marker.stat().st_ino) == before
    assert json.loads(before[0])["format_version"] == version
    assert (physical_root(configured) / key).read_bytes() == payload


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("damage", ["missing", "symlink", "permissions"])
def test_published_layout_damage_is_rejected_without_repair(
    configured: PosixDeployment, version: int, damage: str
) -> None:
    if version == 1:
        with PosixArtifactStore(configured.artifact_root, writable=True) as store:
            store.put_object("namespace.json", configured.namespace.to_bytes())
    initialize(configured)
    child = physical_root(configured) / "chunks"
    if damage == "permissions":
        child.chmod(0o777)
    else:
        child.rmdir()
        if damage == "symlink":
            child.symlink_to(physical_root(configured) / "indexes", target_is_directory=True)
    before = _state(configured.artifact_root)
    for config in (configured, _reader(configured)):
        with pytest.raises(StorageDeploymentError):
            open_deployed_store(config, limits=LIMITS, storage_limits=STORAGE)
    with pytest.raises(StorageDeploymentError):
        initialize(configured)
    assert _state(configured.artifact_root) == before


@pytest.mark.parametrize("layout", [None, "", "../outside", "/absolute", "layout-../x", "layout-" + "g" * 32])
def test_invalid_layout_descriptor_is_rejected_before_writes(configured: PosixDeployment, layout: object) -> None:
    value = json.loads(configured.namespace.to_bytes())
    value.update(format_version=2, layout=layout)
    marker = configured.artifact_root / "namespace.json"
    marker.write_text(json.dumps(value))
    before = _state(configured.artifact_root)
    with pytest.raises(StorageDeploymentError):
        initialize(configured)
    assert _state(configured.artifact_root) == before


def test_reader_does_not_scan_or_consume_unpublished_layouts(
    configured: PosixDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    orphan = configured.artifact_root / ("layout-" + "b" * 32)
    orphan.mkdir(mode=0o700)
    initialize(configured)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("reader must not scan unpublished layouts")

    monkeypatch.setattr(os, "scandir", forbidden)
    with open_deployed_store(_reader(configured), limits=LIMITS, storage_limits=STORAGE) as store:
        assert store.capabilities().access == "reader"


@pytest.mark.parametrize("budget", ["layouts", "objects", "bytes"])
def test_leftovers_exhaust_persistent_budget_before_creating_another_layout(
    configured: PosixDeployment, budget: str
) -> None:
    orphan = configured.artifact_root / ("layout-" + "c" * 32)
    orphan.mkdir(mode=0o700)
    limits = {
        "layouts": replace(STORAGE, max_layouts=1),
        "objects": replace(STORAGE, max_objects=7),
        "bytes": replace(STORAGE, max_bytes=6 * 4096),
    }[budget]
    for _ in range(2):
        with pytest.raises(DeltaCodecError, match="layout"):
            initialize_namespace(configured, limits=LIMITS, storage_limits=limits)
        assert list(configured.artifact_root.glob("layout-*")) == [orphan]
        assert not list(orphan.iterdir())
        assert not (configured.artifact_root / "namespace.json").exists()


@pytest.mark.parametrize("kind", ["file", "symlink", "unknown_child", "nonempty_child"])
def test_unpublished_layout_inventory_refuses_unknown_objects(configured: PosixDeployment, kind: str) -> None:
    orphan = configured.artifact_root / ("layout-" + "d" * 32)
    if kind == "file":
        orphan.write_bytes(b"keep")
    elif kind == "symlink":
        orphan.symlink_to(configured.staging_root, target_is_directory=True)
    else:
        orphan.mkdir(mode=0o700)
        child = orphan / ("unknown" if kind == "unknown_child" else "chunks")
        child.mkdir(mode=0o700)
        (child / "sentinel").write_bytes(b"keep")
    before = _state(configured.artifact_root)
    with pytest.raises((DeltaCodecError, OSError)):
        initialize(configured)
    after = _state(configured.artifact_root)
    assert all(after[name] == value for name, value in before.items())
    assert not (configured.artifact_root / "namespace.json").exists()


def test_directory_allocation_growth_does_not_change_logical_quota_after_restart(
    configured: PosixDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize(configured)
    with open_deployed_store(configured, limits=LIMITS, storage_limits=STORAGE) as store:
        store.finish_recovery()
        payload = b"payload"
        store.put_object("chunks/" + content_hash(payload), payload)
        usage = store._objects, store._bytes
    original = os.fstat

    def grown_directory(descriptor: int) -> os.stat_result:
        info = original(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            return info
        values = list(info)
        values[6] = 1024 * 1024
        return os.stat_result(values)

    monkeypatch.setattr(os, "fstat", grown_directory)
    bounded = replace(STORAGE, max_objects=usage[0], max_bytes=usage[1])
    with open_deployed_store(configured, limits=LIMITS, storage_limits=bounded) as store:
        assert (store._objects, store._bytes) == usage


def test_legacy_marker_uses_its_own_record_budget(configured: PosixDeployment) -> None:
    data = configured.namespace.to_bytes()
    bounded = replace(STORAGE, max_record_bytes=len(data))
    with PosixArtifactStore(configured.artifact_root, writable=True) as store:
        store.put_object("namespace.json", data)
    initialize_namespace(configured, limits=LIMITS, storage_limits=bounded)
    with open_deployed_store(_reader(configured), limits=LIMITS, storage_limits=bounded) as store:
        assert store.read_object("namespace.json", len(data)) == data
    marker = configured.artifact_root / "namespace.json"
    marker.write_bytes(data + b" ")
    with pytest.raises(StorageDeploymentError, match="deployment_inspection"):
        open_deployed_store(configured, limits=LIMITS, storage_limits=bounded)


def test_published_descriptor_cannot_exceed_record_budget(configured: PosixDeployment) -> None:
    initialize(configured)
    size = (configured.artifact_root / "namespace.json").stat().st_size
    bounded = replace(STORAGE, max_record_bytes=size - 1)
    with pytest.raises(StorageDeploymentError, match="namespace_metadata_budget"):
        open_deployed_store(configured, limits=LIMITS, storage_limits=bounded)


def test_new_marker_budget_is_checked_before_initialization_writes(configured: PosixDeployment) -> None:
    bounded = replace(STORAGE, max_record_bytes=len(configured.namespace.to_bytes()))
    with pytest.raises(StorageDeploymentError, match="namespace_metadata_budget"):
        initialize_namespace(configured, limits=LIMITS, storage_limits=bounded)
    assert not list(configured.artifact_root.iterdir())


@pytest.mark.parametrize("damage", ["missing", "symlink"])
def test_descriptor_cannot_select_missing_or_symlinked_layout(configured: PosixDeployment, damage: str) -> None:
    initialize(configured)
    marker = configured.artifact_root / "namespace.json"
    descriptor = layout_io.NamespaceDescriptor.from_bytes(marker.read_bytes())
    target = configured.artifact_root / ("layout-" + "e" * 32)
    if damage == "symlink":
        target.symlink_to(physical_root(configured), target_is_directory=True)
    marker.write_bytes(replace(descriptor, layout=target.name).to_bytes())
    before = _state(configured.artifact_root)
    for candidate in (configured, _reader(configured)):
        with pytest.raises(StorageDeploymentError):
            open_deployed_store(candidate, limits=LIMITS, storage_limits=STORAGE)
    assert _state(configured.artifact_root) == before


@pytest.mark.parametrize("change", ["boolean_version", "future_version", "unknown_field", "missing_layout"])
def test_descriptor_schema_is_strict(configured: PosixDeployment, change: str) -> None:
    value = json.loads(configured.namespace.to_bytes())
    value.update(format_version=2, layout="layout-" + "f" * 32)
    if change == "boolean_version":
        value["format_version"] = True
    elif change == "future_version":
        value["format_version"] = 3
    elif change == "unknown_field":
        value["unexpected"] = "value"
    else:
        del value["layout"]
    marker = configured.artifact_root / "namespace.json"
    marker.write_text(json.dumps(value))
    before = _state(configured.artifact_root)
    with pytest.raises(StorageDeploymentError):
        initialize(configured)
    assert _state(configured.artifact_root) == before


@pytest.mark.parametrize("failure", ["command", "abi", "width", "filesystem"])
def test_unsupported_ofd_locks_fail_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    parent = file_io.open_directory(tmp_path)

    def unsupported(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "fixture unsupported")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not fall back to flock")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(file_io.fcntl, "flock", forbidden)
            if failure == "command":
                patch.delattr(file_io.fcntl, "F_OFD_SETLK")
            elif failure == "abi":
                patch.setattr(file_io.platform, "machine", lambda: "unsupported")
            elif failure == "width":
                original = file_io.struct.calcsize
                patch.setattr(file_io.struct, "calcsize", lambda kind: 4 if kind == "P" else original(kind))
            else:
                patch.setattr(file_io.fcntl, "fcntl", unsupported)
            with pytest.raises(DeltaCodecError, match="OFD locks"):
                file_io.exclusive_lock(parent, ".writer.lock")
        descriptor = file_io.exclusive_lock(parent, ".writer.lock")
        os.close(descriptor)
    finally:
        os.close(parent)


def _check_inherited_lock(root: str, inherited: int) -> None:
    # Closing the child's inherited reference must not unlock the parent.
    os.close(inherited)
    parent = file_io.open_directory(root)
    try:
        try:
            descriptor = file_io.exclusive_lock(parent, ".writer.lock")
        except BlockingIOError:
            os._exit(93)
        else:
            os.close(descriptor)
            os._exit(94)
    finally:
        os.close(parent)


def test_failed_second_open_and_fork_close_do_not_release_writer_lock(tmp_path: Path) -> None:
    parent = file_io.open_directory(tmp_path)
    descriptor = file_io.exclusive_lock(parent, ".writer.lock")
    try:
        with pytest.raises(BlockingIOError):
            file_io.exclusive_lock(parent, ".writer.lock")
        extra = os.open(".writer.lock", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        os.close(extra)
        process = multiprocessing.get_context("fork").Process(
            target=_check_inherited_lock, args=(str(tmp_path), descriptor)
        )
        process.start()
        process.join(15)
        if process.is_alive():
            process.kill()
            process.join()
            pytest.fail("lock conflict child hung")
        assert process.exitcode == 93
    finally:
        os.close(descriptor)
        os.close(parent)
    parent = file_io.open_directory(tmp_path)
    try:
        descriptor = file_io.exclusive_lock(parent, ".writer.lock")
        os.close(descriptor)
    finally:
        os.close(parent)


def _check_last_inherited_reference(root: str, inherited: int, ready: int, unused: int) -> None:
    os.close(unused)
    assert os.read(ready, 1) == b"x"
    os.close(ready)
    parent = file_io.open_directory(root)
    try:
        try:
            descriptor = file_io.exclusive_lock(parent, ".writer.lock")
        except BlockingIOError:
            pass
        else:
            os.close(descriptor)
            os._exit(94)
        os.close(inherited)
        descriptor = file_io.exclusive_lock(parent, ".writer.lock")
        os.close(descriptor)
    finally:
        os.close(parent)
    os._exit(93)


def test_lock_survives_parent_close_until_last_inherited_reference_closes(tmp_path: Path) -> None:
    parent = file_io.open_directory(tmp_path)
    descriptor = file_io.exclusive_lock(parent, ".writer.lock")
    ready, signal = os.pipe()
    try:
        process = multiprocessing.get_context("fork").Process(
            target=_check_last_inherited_reference, args=(str(tmp_path), descriptor, ready, signal)
        )
        process.start()
        os.close(descriptor)
        descriptor = -1
        os.close(ready)
        ready = -1
        os.write(signal, b"x")
        process.join(15)
        if process.is_alive():
            process.kill()
            process.join()
            pytest.fail("inherited lock child hung")
        assert process.exitcode == 93
    finally:
        for fd in (descriptor, ready, signal, parent):
            if fd >= 0:
                os.close(fd)
