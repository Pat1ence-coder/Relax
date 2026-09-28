# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Small POSIX primitives; trusted roots, no-follow children, bounded reads."""

import errno
import fcntl
import os
import platform
import stat
import struct
import sys
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


def exclusive_lock(directory: int, name: str) -> int:
    # Linux LP64 struct flock: short type/whence, off_t start/length, pid_t pid.
    # Never silently fall back to flock: some shared mounts enforce it locally.
    if (
        sys.platform != "linux"
        or platform.machine() not in {"x86_64", "aarch64"}
        or tuple(struct.calcsize(kind) for kind in ("P", "l", "h", "i")) != (8, 8, 2, 4)
        or struct.calcsize("@hhqqi4x") != 32
        or not hasattr(fcntl, "F_OFD_SETLK")
    ):
        raise DeltaCodecError("Linux OFD locks require a supported 64-bit ABI and runtime")
    fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise DeltaCodecError("lock must be a regular file")
        record = struct.pack("@hhqqi4x", fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0)
        try:
            fcntl.fcntl(fd, fcntl.F_OFD_SETLK, record)
        except OSError as error:
            if error.errno in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                raise DeltaCodecError("Linux OFD locks are unsupported on this deployment") from None
            raise
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
