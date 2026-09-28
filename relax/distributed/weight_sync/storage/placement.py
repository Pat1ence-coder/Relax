# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded Linux mount observations and guards for opened directories."""

import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path

from ..codec.format import content_hash
from ..limits import DeltaCodecError


class StorageDeploymentError(DeltaCodecError):
    """Stable, path-free deployment diagnosis safe for shared reports."""

    def __init__(self, code: str, check: str) -> None:
        self.code, self.check = code, check
        super().__init__(f"{code}: {check}")


@dataclass(frozen=True)
class MountRecord:
    mount_id: int
    device: str
    mount_point: Path = field(repr=False)
    filesystem_type: str
    source_digest: str
    root_digest: str
    readonly: bool
    _root_components: tuple[str, ...] | None = field(default=None, repr=False)


def _component_hashes(path: Path) -> tuple[str, ...]:
    return tuple(content_hash(os.fsencode(part)) for part in path.parts if part != path.anchor)


def _unescape(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def parse_mountinfo(data: bytes, *, maximum: int = 1024 * 1024) -> tuple[MountRecord, ...]:
    """Discard raw sources and options, which may contain credentials."""
    if type(data) is not bytes or len(data) > maximum:
        raise StorageDeploymentError("STORAGE_CAPABILITY_UNSUPPORTED", "mount_table_budget")
    records = []
    try:
        for line in data.decode("utf-8", errors="surrogateescape").splitlines():
            fields = line.split()
            separator = fields.index("-")
            if separator < 6 or len(fields) != separator + 4:
                raise ValueError
            mount_id = int(fields[0])
            if mount_id <= 0 or re.fullmatch(r"[0-9]+:[0-9]+", fields[2]) is None:
                raise ValueError
            fs_type = fields[separator + 1]
            if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", fs_type) is None:
                raise ValueError
            point = Path(_unescape(fields[4]))
            root = Path(_unescape(fields[3]))
            if not point.is_absolute() or not root.is_absolute() or ".." in root.parts:
                raise ValueError
            records.append(
                MountRecord(
                    mount_id,
                    fields[2],
                    point,
                    fs_type,
                    content_hash(_unescape(fields[separator + 2]).encode("utf-8", errors="surrogateescape")),
                    content_hash(_unescape(fields[3]).encode("utf-8", errors="surrogateescape")),
                    "ro" in fields[5].split(",") or "ro" in fields[separator + 3].split(","),
                    _component_hashes(root),
                )
            )
        if not records or len({record.mount_id for record in records}) != len(records):
            raise ValueError
    except (ValueError, IndexError):
        raise StorageDeploymentError("STORAGE_CAPABILITY_UNSUPPORTED", "mount_table_format") from None
    return tuple(records)


def read_mounts() -> tuple[MountRecord, ...]:
    with open("/proc/self/mountinfo", "rb") as source:
        data = source.read(1024 * 1024 + 1)
    return parse_mountinfo(data)


def mount_for(path: Path, records: tuple[MountRecord, ...]) -> MountRecord:
    resolved = path.resolve(strict=True)
    matches = [record for record in records if resolved.is_relative_to(record.mount_point)]
    if not matches:
        raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "mount_missing")
    longest = max(len(record.mount_point.parts) for record in matches)
    selected = [record for record in matches if len(record.mount_point.parts) == longest]
    if len(selected) != 1:
        raise StorageDeploymentError("STORAGE_CAPABILITY_UNSUPPORTED", "ambiguous_mount")
    return selected[0]


def descriptor_mount_id(descriptor: int) -> int:
    with open(f"/proc/self/fdinfo/{descriptor}", "rb") as source:
        data = source.read(4097)
    if len(data) > 4096:
        raise StorageDeploymentError("STORAGE_CAPABILITY_UNSUPPORTED", "descriptor_mount_budget")
    values = re.findall(rb"^mnt_id:\s*([0-9]+)$", data, flags=re.MULTILINE)
    if len(values) != 1:
        raise StorageDeploymentError("STORAGE_CAPABILITY_UNSUPPORTED", "descriptor_mount_missing")
    return int(values[0])


def require_separate(left: Path, right: Path) -> None:
    left, right = left.resolve(), right.resolve()
    if left.is_relative_to(right) or right.is_relative_to(left):
        raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "control_and_artifacts_must_be_separate")

    # Resolve symlinks above; inode ancestry also detects directory aliases.
    # A missing legacy control directory still has existing ancestors to check.
    def identities(path: Path) -> tuple[tuple[int, int] | None, set[tuple[int, int]]]:
        own = None
        parents = set()
        for index, candidate in enumerate((path, *path.parents)):
            try:
                info = candidate.stat()
            except FileNotFoundError:
                continue
            identity = (info.st_dev, info.st_ino)
            parents.add(identity)
            if index == 0:
                own = identity
        return own, parents

    left_id, left_parents = identities(left)
    right_id, right_parents = identities(right)
    if (left_id is not None and left_id in right_parents) or (right_id is not None and right_id in left_parents):
        raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "physical_directories_overlap")


def require_mount_separation(left: Path, left_mount: MountRecord, right: Path, right_mount: MountRecord) -> None:
    """Reject overlapping filesystem coordinates exposed through bind mounts.

    Only hashed components are retained from mount roots. Distinct source names
    can still alias externally; volume evidence must describe those
    deployments.
    """
    same_source = (left_mount.filesystem_type, left_mount.source_digest) == (
        right_mount.filesystem_type,
        right_mount.source_digest,
    )
    if left_mount.device != right_mount.device and not same_source:
        return
    left_root, right_root = left_mount._root_components, right_mount._root_components
    if left_root is None and left_mount.root_digest == content_hash(b"/"):
        left_root = ()
    if right_root is None and right_mount.root_digest == content_hash(b"/"):
        right_root = ()
    if left_root is None or right_root is None:
        if left_mount.root_digest != right_mount.root_digest:
            raise StorageDeploymentError("STORAGE_CAPABILITY_UNSUPPORTED", "directory_separation_unproven")
        # Equal root digests permit comparison relative to that same root.
        left_root = right_root = ()
    left_parts = left_root + _component_hashes(left.resolve(strict=True).relative_to(left_mount.mount_point))
    right_parts = right_root + _component_hashes(right.resolve(strict=True).relative_to(right_mount.mount_point))
    if left_parts[: len(right_parts)] == right_parts or right_parts[: len(left_parts)] == left_parts:
        raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "physical_directories_overlap")


@dataclass(frozen=True)
class DirectoryGuard:
    path: Path = field(repr=False)
    identity: tuple[int, int] | None = None
    mount_id: int | None = None

    @classmethod
    def capture(cls, path: Path, *, mount_id: int | None = None, must_exist: bool = True) -> "DirectoryGuard":
        try:
            info = path.lstat()
        except FileNotFoundError:
            if must_exist:
                raise
            return cls(path)
        if not stat.S_ISDIR(info.st_mode):
            raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "directory_required")
        return cls(path, (info.st_dev, info.st_ino), mount_id)

    def validate_opened(self, descriptor: int) -> None:
        current, opened = self.path.lstat(), os.fstat(descriptor)
        actual = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISDIR(current.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or actual != (current.st_dev, current.st_ino)
            or (self.identity is not None and actual != self.identity)
            or (self.mount_id is not None and descriptor_mount_id(descriptor) != self.mount_id)
        ):
            raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "directory_changed_after_inspection")

    def validate_child_mount(self, descriptor: int) -> None:
        if self.mount_id is None or descriptor_mount_id(descriptor) != self.mount_id:
            raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "artifact_child_mount_mismatch")
