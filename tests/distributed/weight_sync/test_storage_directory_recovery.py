# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Private directory preparation, non-replacement and interrupted recovery."""

import multiprocessing
import os
from pathlib import Path

import pytest

from relax.distributed.weight_sync import DeltaCodecError
from relax.distributed.weight_sync.storage import files as file_io


def _crash_directory_creation(root: str, phase: str) -> None:
    parent = file_io.open_directory(root)
    lock = file_io.exclusive_lock(parent, ".writer.lock")
    mkdir, chown, chmod, sync = os.mkdir, os.fchown, os.fchmod, os.fsync
    rename = file_io._rename_directory_no_replace

    def crash_mkdir(*args: object, **kwargs: object) -> None:
        mkdir(*args, **kwargs)
        if phase == "mkdir":
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
        if (phase == "directory_sync" and descriptor != parent) or (phase == "parent_sync" and descriptor == parent):
            os._exit(93)

    def crash_install(*args: object, **kwargs: object) -> bool:
        created = rename(*args, **kwargs)
        if phase == "install":
            os._exit(93)
        return created

    os.mkdir, os.fchown, os.fchmod, os.fsync = crash_mkdir, crash_chown, crash_chmod, crash_sync
    file_io._rename_directory_no_replace = crash_install
    try:
        descriptor = file_io.child_directory(parent, "chunks", create=True, mode=0o750, group_id=os.getegid())
        os.close(descriptor)
    finally:
        os.close(lock)
        os.close(parent)
    os._exit(94)


@pytest.mark.parametrize("phase", ["mkdir", "chown", "chmod", "directory_sync", "install", "parent_sync"])
def test_interrupted_group_directory_creation_recovers_without_partial_final_modes(tmp_path: Path, phase: str) -> None:
    root = tmp_path / "artifacts"
    root.mkdir(mode=0o700)
    process = multiprocessing.get_context("fork").Process(target=_crash_directory_creation, args=(str(root), phase))
    process.start()
    process.join(15)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("directory initialization crash fixture hung")
    assert process.exitcode == 93
    target = root / "chunks"
    if target.exists():
        assert target.stat().st_mode & 0o777 == 0o750
        assert target.stat().st_gid == os.getegid()
    else:
        assert (root / ".tmp-dir-chunks").is_dir()
    parent = file_io.open_directory(root)
    lock = file_io.exclusive_lock(parent, ".writer.lock")
    try:
        descriptor = file_io.child_directory(parent, "chunks", create=True, mode=0o750, group_id=os.getegid())
        try:
            assert os.fstat(descriptor).st_mode & 0o777 == 0o750
            assert os.fstat(descriptor).st_gid == os.getegid()
        finally:
            os.close(descriptor)
        assert not (root / ".tmp-dir-chunks").exists()
    finally:
        os.close(lock)
        os.close(parent)


def test_directory_install_never_replaces_a_concurrently_created_empty_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = file_io.open_directory(tmp_path)
    lock = file_io.exclusive_lock(parent, ".writer.lock")
    rename = file_io._rename_directory_no_replace
    existing = []

    def create_competitor(directory: int, temporary: str, name: str) -> bool:
        os.mkdir(name, mode=0o700, dir_fd=directory)
        existing.append(os.stat(name, dir_fd=directory, follow_symlinks=False))
        return rename(directory, temporary, name)

    monkeypatch.setattr(file_io, "_rename_directory_no_replace", create_competitor)
    try:
        descriptor = file_io.child_directory(parent, "chunks", create=True, mode=0o750, group_id=os.getegid())
        try:
            actual = os.fstat(descriptor)
            assert (actual.st_dev, actual.st_ino, actual.st_mode) == (
                existing[0].st_dev,
                existing[0].st_ino,
                existing[0].st_mode,
            )
        finally:
            os.close(descriptor)
        assert not (tmp_path / ".tmp-dir-chunks").exists()
    finally:
        os.close(lock)
        os.close(parent)


@pytest.mark.parametrize("failure", ["permissions", "unsupported_install"])
def test_failed_directory_preparation_is_cleaned_without_changing_final_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    parent = file_io.open_directory(tmp_path)
    lock = file_io.exclusive_lock(parent, ".writer.lock")

    def fail(*args: object, **kwargs: object) -> None:
        if failure == "permissions":
            raise PermissionError("fixture denied")
        raise DeltaCodecError("atomic directory installation is unsupported")

    try:
        with monkeypatch.context() as patch:
            if failure == "permissions":
                patch.setattr(os, "fchown", fail)
            else:
                patch.setattr(file_io, "_rename_directory_no_replace", fail)
            with pytest.raises((PermissionError, DeltaCodecError)):
                file_io.child_directory(parent, "chunks", create=True, mode=0o750, group_id=os.getegid())
        assert not (tmp_path / "chunks").exists()
        assert not (tmp_path / ".tmp-dir-chunks").exists()
        descriptor = file_io.child_directory(parent, "chunks", create=True, mode=0o750, group_id=os.getegid())
        os.close(descriptor)
    finally:
        os.close(lock)
        os.close(parent)


@pytest.mark.parametrize("kind", ["file", "symlink", "nonempty_directory"])
def test_preparation_recovery_refuses_unknown_contents(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "artifacts"
    root.mkdir(mode=0o700)
    outside = tmp_path / "untouched"
    outside.mkdir(mode=0o700)
    (outside / "sentinel").write_bytes(b"keep")
    prepared = root / ".tmp-dir-chunks"
    if kind == "file":
        prepared.write_bytes(b"keep")
    elif kind == "symlink":
        prepared.symlink_to(outside, target_is_directory=True)
    else:
        prepared.mkdir(mode=0o700)
        (prepared / "sentinel").write_bytes(b"keep")
    parent = file_io.open_directory(root)
    lock = file_io.exclusive_lock(parent, ".writer.lock")
    try:
        with pytest.raises(OSError):
            file_io.child_directory(parent, "chunks", create=True, mode=0o750, group_id=os.getegid())
        assert prepared.exists()
        assert (outside / "sentinel").read_bytes() == b"keep"
        assert not (root / "chunks").exists()
        if kind == "file":
            assert prepared.read_bytes() == b"keep"
        elif kind == "nonempty_directory":
            assert (prepared / "sentinel").read_bytes() == b"keep"
    finally:
        os.close(lock)
        os.close(parent)
