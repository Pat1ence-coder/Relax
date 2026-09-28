# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Small POSIX primitives; trusted roots, no-follow children, bounded reads."""

import fcntl
import os
import stat
import uuid
from pathlib import Path

from ..limits import DeltaCodecError


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


def child_directory(parent: int, name: str, *, create: bool = False) -> int:
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent)
        except FileExistsError:
            pass
        else:
            os.fsync(parent)
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)


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


def install_file(directory: int, name: str, data: bytes) -> bool:
    """Fsync then link without overwrite; finally remove our temporary name.

    An exception after link can leave a complete final file. Retrying verifies
    it and fsyncs the directory. The caller must resolve publication
    separately.
    """
    temporary = ".tmp-" + uuid.uuid4().hex
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    try:
        write_all(fd, data)
        os.fchmod(fd, 0o444)
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
