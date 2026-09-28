# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Immutable namespace descriptors and bounded, unpublished layout
preparation."""

import os
import re
import uuid
from dataclasses import dataclass

from ..limits import DeltaCodecError, require_uint
from ..serialization import canonical_json, exact_fields, parse_json, require_identifier
from .contracts import StorageLimits
from .files import ARTIFACT_DIRECTORIES, child_directory, read_file
from .placement import DirectoryGuard, StorageDeploymentError


NAMESPACE_LIMIT = 16 * 1024
LAYOUT_NAME = re.compile(r"layout-[0-9a-f]{32}")
_BINDING_FIELDS = {"namespace_id", "stream_id", "run_epoch"}


@dataclass(frozen=True)
class NamespaceDescriptor:
    namespace_id: str
    stream_id: str
    run_epoch: str
    layout: str | None = None

    def __post_init__(self) -> None:
        for name in _BINDING_FIELDS:
            require_identifier(getattr(self, name), name)
        if self.layout is not None and (
            not isinstance(self.layout, str) or LAYOUT_NAME.fullmatch(self.layout) is None
        ):
            raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "namespace_layout_name")

    def to_bytes(self) -> bytes:
        value = {name: getattr(self, name) for name in _BINDING_FIELDS}
        value["format_version"] = 1 if self.layout is None else 2
        if self.layout is not None:
            value["layout"] = self.layout
        return canonical_json(value, NAMESPACE_LIMIT)

    @classmethod
    def from_bytes(cls, data: bytes) -> "NamespaceDescriptor":
        value = parse_json(data, NAMESPACE_LIMIT)
        if not isinstance(value, dict) or type(value.get("format_version")) is not int:
            raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "namespace_format")
        version = value["format_version"]
        if version not in (1, 2):
            raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "namespace_format")
        value = exact_fields(value, _BINDING_FIELDS | {"format_version"} | ({"layout"} if version == 2 else set()))
        value.pop("format_version")
        if version == 2 and not isinstance(value["layout"], str):
            raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "namespace_layout_name")
        return cls(**value)


def read_descriptor(parent: int) -> NamespaceDescriptor | None:
    try:
        data = read_file(parent, "namespace.json", NAMESPACE_LIMIT)
    except FileNotFoundError:
        return None
    return NamespaceDescriptor.from_bytes(data)


def validate_directory(descriptor: int, guard: DirectoryGuard | None, group_id: int | None) -> None:
    if guard is not None:
        guard.validate_child_mount(descriptor)
    info = os.fstat(descriptor)
    if info.st_mode & 0o022:
        raise StorageDeploymentError("STORAGE_PERMISSION_DENIED", "artifact_directory_write_policy")
    if group_id is not None and (info.st_gid != group_id or info.st_mode & 0o050 != 0o050):
        raise StorageDeploymentError("STORAGE_PERMISSION_DENIED", "artifact_directory_group_policy")


def layout_inventory(
    parent: int, active: str | None, limits: StorageLimits, guard: DirectoryGuard | None
) -> tuple[int, int, int]:
    """Count layout directories, retaining all unselected generations.

    Unpublished layouts may contain only the five empty artifact directories.
    Unknown contents are refused, never adopted or cleaned. Directory metadata
    consumes one object and a fixed 4 KiB logical byte charge per directory.
    Actual filesystem allocation needs separate deployment headroom.
    """
    generations = objects = size = scanned = 0
    with os.scandir(parent) as entries:
        for entry in entries:
            scanned += 1
            require_uint(scanned, "root scan entries", limits.max_objects + limits.max_layouts + 8)
            if LAYOUT_NAME.fullmatch(entry.name) is None:
                continue
            generations += 1
            require_uint(generations, "layout count", limits.max_layouts)
            descriptor = child_directory(parent, entry.name)
            try:
                if guard is not None:
                    guard.validate_child_mount(descriptor)
                objects += 1
                size += 4096
                with os.scandir(descriptor) as children:
                    for index, child in enumerate(children):
                        if index >= len(ARTIFACT_DIRECTORIES) or child.name not in ARTIFACT_DIRECTORIES:
                            raise DeltaCodecError("unknown layout contents")
                        fd = child_directory(descriptor, child.name)
                        try:
                            if guard is not None:
                                guard.validate_child_mount(fd)
                            objects += 1
                            size += 4096
                            if entry.name != active:
                                with os.scandir(fd) as contents:
                                    if next(contents, None) is not None:
                                        raise DeltaCodecError("unpublished layout is not empty")
                        finally:
                            os.close(fd)
            finally:
                os.close(descriptor)
            require_uint(objects, "layout objects", limits.max_objects)
            require_uint(size, "layout bytes", limits.max_bytes)
    return generations, objects, size


def prepare_layout(
    parent: int, *, mode: int, group_id: int | None, guard: DirectoryGuard, limits: StorageLimits
) -> str:
    """Prepare a fresh directory under the writer lock; never repair a
    leftover."""
    generations, objects, size = layout_inventory(parent, None, limits, guard)
    require_uint(generations + 1, "layout count", limits.max_layouts)
    require_uint(objects + 6 + 1, "layout objects", limits.max_objects)
    require_uint(size + 6 * 4096, "layout bytes", limits.max_bytes)
    for _ in range(16):
        name = "layout-" + uuid.uuid4().hex
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent)
        except FileExistsError:
            continue
        break
    else:
        raise DeltaCodecError("layout name collision limit")
    descriptor = child_directory(parent, name)
    try:
        guard.validate_child_mount(descriptor)
        for child in ARTIFACT_DIRECTORIES:
            os.mkdir(child, mode=0o700, dir_fd=descriptor)
            fd = child_directory(descriptor, child)
            try:
                guard.validate_child_mount(fd)
                if group_id is not None:
                    os.fchown(fd, -1, group_id)
                os.fchmod(fd, mode)
                validate_directory(fd, guard, group_id)
                os.fsync(fd)
            finally:
                os.close(fd)
        if group_id is not None:
            os.fchown(descriptor, -1, group_id)
        os.fchmod(descriptor, mode)
        validate_directory(descriptor, guard, group_id)
        os.fsync(descriptor)
        os.fsync(parent)
    finally:
        os.close(descriptor)
    layout_inventory(parent, name, limits, guard)
    return name
