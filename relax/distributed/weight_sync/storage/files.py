# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Small POSIX primitives; trusted roots, no-follow children, bounded reads."""

import errno
import fcntl
import os
import stat
import uuid
from pathlib import Path

from ..limits import DeltaCodecError


ARTIFACT_DIRECTORIES = ("chunks", "indexes", "manifests", "catalog", "archives")


def open_directory(path: str | Path, *, create: bool = False) -> int:
    path = Path(path)
    if create:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        else:
            parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def child_directory(
    parent: int, name: str, *, create: bool = False, mode: int = 0o700, group_id: int | None = None
) -> int:
    if create and group_id is not None:
        _install_group_directory(parent, name, mode, group_id)
        return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    created = False
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent)
        except FileExistsError:
            pass
        else:
            created = True
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    try:
        if created:
            if group_id is not None:
                os.fchown(fd, -1, group_id)
            os.fchmod(fd, mode)
            os.fsync(fd)
            os.fsync(parent)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _rename_directory_no_replace(parent: int, temporary: str, name: str) -> bool:
    """Linux atomic directory installation; never fall back to replacement."""
    import ctypes

    rename = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if rename is None:
        raise DeltaCodecError("atomic directory installation is unsupported")
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    # RENAME_NOREPLACE: unlike os.rename, reject even an existing empty directory.
    if rename(parent, os.fsencode(temporary), parent, os.fsencode(name), 1) == 0:
        return True
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        return False
    if error in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
        raise DeltaCodecError("atomic directory installation is unsupported")
    raise OSError(error, os.strerror(error))


def _remove_prepared_directory(parent: int, name: str) -> None:
    # rmdir removes only an empty directory, never symlinks, files, mounted
    # directories or unknown contents. The caller holds the namespace writer lock.
    try:
        os.rmdir(name, dir_fd=parent)
    except FileNotFoundError:
        return
    os.fsync(parent)


def _install_group_directory(parent: int, name: str, mode: int, group_id: int) -> None:
    """Prepare modes privately under the exclusive namespace writer lock."""
    if name not in ARTIFACT_DIRECTORIES:
        raise DeltaCodecError("group directory requires a reserved artifact name")
    temporary = ".tmp-dir-" + name
    _remove_prepared_directory(parent, temporary)
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        # An earlier installation may have stopped before syncing its parent.
        # Existing directories are never chmodded or replaced.
        os.fsync(parent)
        return
    descriptor = -1
    try:
        os.mkdir(temporary, mode=0o700, dir_fd=parent)
        descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        os.fchown(descriptor, -1, group_id)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        _rename_directory_no_replace(parent, temporary, name)
        os.fsync(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _remove_prepared_directory(parent, temporary)


def exclusive_lock(directory: int, name: str) -> int:
    fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise DeltaCodecError("lock must be a regular file")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.fsync(directory)
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_file(directory: int, name: str, maximum: int, *, length: int | None = None) -> bytes:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise DeltaCodecError("object type or size exceeds limit")
        if length is not None and info.st_size != length:
            raise DeltaCodecError("object length mismatch")
        result = bytearray()
        while len(result) < info.st_size:
            part = os.read(fd, min(info.st_size - len(result), 1024 * 1024))
            if not part:
                raise DeltaCodecError("truncated object")
            result.extend(part)
        if os.read(fd, 1):
            raise DeltaCodecError("object grew while reading")
        return bytes(result)
    finally:
        os.close(fd)


def write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write made no progress")
        view = view[written:]


def install_file(directory: int, name: str, data: bytes, *, mode: int = 0o444, group_id: int | None = None) -> bool:
    """Fsync then link without overwrite; finally remove our temporary name.

    An exception after link can leave a complete final file. Retrying verifies
    it and fsyncs the directory. The caller must resolve publication
    separately.
    """
    temporary = ".tmp-" + uuid.uuid4().hex
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    try:
        write_all(fd, data)
        if group_id is not None:
            os.fchown(fd, -1, group_id)
        os.fchmod(fd, mode)
        os.fsync(fd)
        try:
            os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
            created = True
        except FileExistsError:
            if read_file(directory, name, len(data), length=len(data)) != data:
                raise DeltaCodecError("immutable object conflict")
            created = False
        os.fsync(directory)
        return created
    finally:
        os.close(fd)
        os.unlink(temporary, dir_fd=directory)
        os.fsync(directory)
