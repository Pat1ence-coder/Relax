# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""POSIX shared artifact store with exclusive writes and persistent quotas."""

import os
import re
import stat
import threading
from dataclasses import asdict
from pathlib import Path

from ..codec import EncodedChunk
from ..codec.format import content_hash
from ..limits import DeltaCodecError, SnapshotLimits, require_uint
from ..manifest import ChunkRecord, IndexPage, Manifest, PageRef
from ..schema import require_digest
from .contracts import ObjectReceipt, StorageLimits, StoreCapabilities
from .deployment import PROFILE, DeploymentAdmission, DeploymentReport
from .files import ARTIFACT_DIRECTORIES, child_directory, exclusive_lock, install_file, open_directory, read_file
from .layout import (
    LAYOUT_NAME,
    NamespaceDescriptor,
    layout_inventory,
    prepare_layout,
    read_descriptor,
    validate_directory,
)
from .placement import DirectoryGuard, StorageDeploymentError, require_separate
from .repository import read_manifest, validate_artifacts, write_manifest


_KEY = re.compile(
    r"(?:chunks/[0-9a-f]{64}|(?:indexes|manifests|archives)/[0-9a-f]{64}\.json|catalog/[0-9]{20}-[0-9a-f]{64}\.json|(?:authority|namespace)\.json)"
)
_DIRECTORIES = ARTIFACT_DIRECTORIES
_TEMP = re.compile(r"\.tmp-[0-9a-f]{32}")


class PosixArtifactStore:
    """Single writer, many readers on a trusted POSIX shared directory.

    Requires working advisory locks, hard links, and file/directory fsync on
    the deployed filesystem. This lock is not cross-host authority takeover.
    Final objects are never deleted; quota exhaustion stops writes. One
    temporary object per synchronous writer can add at most the current object
    size.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        writable: bool = False,
        limits: SnapshotLimits = SnapshotLimits(),
        storage_limits: StorageLimits = StorageLimits(),
        _admission: DeploymentAdmission | None = None,
    ) -> None:
        self.limits, self.storage_limits = limits, storage_limits
        self.root = Path(root).absolute()
        self._admission = _admission
        self._recovering = writable and _admission is not None
        self._fd = open_directory(self.root, create=writable and _admission is None)
        self._lock: int | None = None
        self._owner = (os.getpid(), threading.get_ident())
        self._poisoned = False
        self._reservations: dict[str, tuple[int, str]] = {}
        self._directories: dict[str, int] = {}
        self._layout_fd: int | None = None
        self._namespace: NamespaceDescriptor | None = None
        self._layout_pending = False
        self._bytes = self._objects = 0
        try:
            if _admission is not None:
                if writable != (_admission.config.access == "publisher"):
                    raise StorageDeploymentError("STORAGE_PERMISSION_DENIED", "store_role_mismatch")
                _admission.report.require_admitted()
                _admission.validate_artifact(self._fd)
            if writable:
                self._lock = exclusive_lock(self._fd, ".writer.lock")
                if _admission is not None:
                    _admission.validate_artifact(self._fd)
            self._namespace = read_descriptor(self._fd)
            policy = None if _admission is None else _admission.config.access_policy
            guard = None if _admission is None else _admission.artifact_guard
            self._layout_pending = _admission is not None and self._namespace is None
            if self._layout_pending:
                if not writable or not _admission.initializing:
                    raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "namespace_missing")
                self._inventory()
                binding = asdict(_admission.config.namespace)
                metadata = NamespaceDescriptor(**binding, layout="layout-" + "0" * 32).to_bytes()
                require_uint(self._objects + 7, "layout objects", storage_limits.max_objects)
                require_uint(self._bytes + 6 * 4096 + len(metadata), "layout bytes", storage_limits.max_bytes)
                name = prepare_layout(
                    self._fd,
                    mode=policy.directory_mode,
                    group_id=policy.shared_group_id,
                    guard=guard,
                    limits=storage_limits,
                )
                self._namespace = NamespaceDescriptor(**binding, layout=name)
            parent = self._fd
            if self._namespace is not None and self._namespace.layout is not None:
                parent = self._layout_fd = child_directory(self._fd, self._namespace.layout)
                validate_directory(parent, guard, None if policy is None else policy.shared_group_id)
            for name in _DIRECTORIES:
                descriptor = child_directory(
                    parent,
                    name,
                    create=writable and _admission is None and self._namespace is None,
                    mode=0o700 if policy is None else policy.directory_mode,
                    group_id=None if policy is None else policy.shared_group_id,
                )
                self._directories[name] = descriptor
                if policy is not None:
                    validate_directory(descriptor, guard, policy.shared_group_id)
            if writable:
                self._inventory(clean_temporaries=True)
                if self._layout_pending:
                    self.put_object("namespace.json", self._namespace.to_bytes())
                    self._layout_pending = False
                elif _admission is not None and _admission.initializing:
                    # A previous publisher may have linked the descriptor and
                    # stopped before completing its final durability barrier.
                    for fd in self._directories.values():
                        os.fsync(fd)
                    os.fsync(parent)
                    os.fsync(self._fd)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "PosixArtifactStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._lock is not None and self._owner != (os.getpid(), threading.get_ident()):
            raise DeltaCodecError("artifact writer close requires the owning process and thread")
        for fd in self._directories.values():
            os.close(fd)
        self._directories.clear()
        if self._layout_fd is not None:
            os.close(self._layout_fd)
            self._layout_fd = None
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    @property
    def deployment_report(self) -> DeploymentReport | None:
        return None if self._admission is None else self._admission.report

    def capabilities(self) -> StoreCapabilities:
        if self._fd < 0:
            raise DeltaCodecError("artifact store is closed")
        requirements = (
            "atomic_hard_link",
            "advisory_writer_lock",
            "linux_ofd_write_lock",
            "file_fsync",
            "directory_fsync",
        )
        return StoreCapabilities(
            backend="posix",
            profile=PROFILE,
            access="reader" if self._lock is None else "publisher",
            namespace_id=None if self._admission is None else self._admission.config.namespace.namespace_id,
            requirements=requirements,
            deployment_checked=self._admission is not None,
            stream_id=None if self._admission is None else self._admission.config.namespace.stream_id,
            run_epoch=None if self._admission is None else self._admission.config.namespace.run_epoch,
        )

    def prepare_control(self, location: Path) -> DirectoryGuard:
        self._require_writer()
        require_separate(self.root, location)
        if self._admission is not None:
            return self._admission.prepare_control(location)
        return DirectoryGuard.capture(location, must_exist=False)

    def begin_recovery(self) -> None:
        self._require_writer()
        self._recovering = True

    def finish_recovery(self) -> None:
        self._require_writer()
        if self._reservations:
            raise DeltaCodecError("pending publication reservations prevent new uploads")
        self._recovering = False

    def _require_writer(self) -> None:
        if self._fd < 0:
            raise DeltaCodecError("artifact store is closed")
        if self._lock is None:
            raise DeltaCodecError("artifact store is read-only")
        if self._owner != (os.getpid(), threading.get_ident()):
            raise DeltaCodecError("artifact writes require the owning process and thread")
        if self._poisoned:
            raise DeltaCodecError("artifact inventory is uncertain; close and reopen the writer")

    def put_immutable(self, key: str, data: bytes, *, expected_hash: str) -> ObjectReceipt:
        require_digest(expected_hash, "object hash")
        if type(data) is not bytes or content_hash(data) != expected_hash:
            raise DeltaCodecError("immutable write differs from expected hash")
        self.put_object(key, data)
        capabilities = self.capabilities()
        return ObjectReceipt(
            key,
            expected_hash,
            len(data),
            capabilities.namespace_id,
            capabilities.profile,
            None if self._admission is None else self._admission.config.required_fault_domain,
        )

    def _location(self, key: str) -> tuple[int, str]:
        if self._fd < 0:
            raise DeltaCodecError("artifact store is closed")
        if not isinstance(key, str) or _KEY.fullmatch(key) is None:
            raise DeltaCodecError("invalid artifact key")
        if "/" not in key:
            return self._fd, key
        directory, name = key.split("/")
        return self._directories[directory], name

    def _inventory(self, *, clean_temporaries: bool = False) -> None:
        active = None if self._namespace is None else self._namespace.layout
        _, count, total = layout_inventory(
            self._fd,
            active,
            self.storage_limits,
            None if self._admission is None else self._admission.artifact_guard,
        )
        scanned = 0
        for prefix, fd in (("", self._fd), *self._directories.items()):
            with os.scandir(fd) as entries:
                for entry in entries:
                    scanned += 1
                    require_uint(scanned, "inventory entries", self.storage_limits.max_objects + 8)
                    if not prefix and (entry.name == ".writer.lock" or LAYOUT_NAME.fullmatch(entry.name)):
                        continue
                    if not prefix and entry.name in _DIRECTORIES and active is None and not self._layout_pending:
                        continue
                    retained_temporary = not prefix and (active is not None or self._layout_pending)
                    temporary = _TEMP.fullmatch(entry.name) is not None
                    if temporary and clean_temporaries and not retained_temporary:
                        os.unlink(entry.name, dir_fd=fd)
                        os.fsync(fd)
                        continue
                    key = f"{prefix}/{entry.name}" if prefix else entry.name
                    if not (temporary and retained_temporary):
                        self._location(key)
                    info = entry.stat(follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode):
                        raise DeltaCodecError("artifact must be a regular file")
                    count += 1
                    total += info.st_size
                    require_uint(count, "object count", self.storage_limits.max_objects)
                    require_uint(total, "artifact bytes", self.storage_limits.max_bytes)
        self._bytes, self._objects = total, count

    def read_object(self, key: str, maximum: int, *, length: int | None = None) -> bytes:
        directory, name = self._location(key)
        require_uint(maximum, "read budget")
        return read_file(directory, name, min(maximum, self._object_limit(key)), length=length)

    def _object_limit(self, key: str) -> int:
        return {
            "chunks": self.limits.codec.max_chunk_bytes,
            "indexes": self.limits.max_index_page_bytes,
            "manifests": self.limits.max_manifest_bytes,
            "archives": self.storage_limits.max_archive_bytes,
            "catalog": self.storage_limits.max_record_bytes,
            "authority.json": self.storage_limits.max_record_bytes,
            "namespace.json": self.storage_limits.max_record_bytes,
        }[key.split("/")[0]]

    def ensure_capacity(self, key: str, data: bytes) -> bool:
        """Preflight under the single synchronous writer; return whether
        new."""
        self._require_writer()
        if self._recovering and not (key.startswith("catalog/") or key in ("authority.json", "namespace.json")):
            raise DeltaCodecError("publication recovery must finish before new artifact uploads")
        if key == "namespace.json" and self._admission is not None:
            if self._namespace is None or data != self._namespace.to_bytes():
                raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "namespace_write_conflict")
        if type(data) is not bytes:
            raise DeltaCodecError("artifact must be immutable bytes")
        directory, name = self._location(key)
        require_uint(len(data), "object bytes", self._object_limit(key))
        if key.split("/")[0] in ("chunks", "indexes", "manifests", "archives"):
            if content_hash(data) != name.removesuffix(".json"):
                raise DeltaCodecError("object key differs from content hash")
        try:
            existing = read_file(directory, name, len(data), length=len(data))
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != data:
                raise DeltaCodecError("immutable object conflict")
            return False
        reservation = self._reservations.get(key)
        if reservation is not None and reservation != (len(data), content_hash(data)):
            raise DeltaCodecError("reserved object differs from publication")
        reserved_bytes = sum(size for name, (size, _) in self._reservations.items() if name != key)
        reserved_count = len(self._reservations) - (reservation is not None)
        require_uint(self._bytes + reserved_bytes + len(data), "artifact bytes", self.storage_limits.max_bytes)
        require_uint(self._objects + reserved_count + 1, "object count", self.storage_limits.max_objects)
        return True

    def reserve_record(self, key: str, data: bytes) -> None:
        """Keep capacity for an undecided or committed-but-unexported
        record."""
        if not key.startswith("catalog/"):
            raise DeltaCodecError("only publication records may reserve capacity")
        if self.ensure_capacity(key, data):
            self._reservations[key] = (len(data), content_hash(data))

    def release_record(self, key: str) -> None:
        self._require_writer()
        self._reservations.pop(key, None)

    def put_object(self, key: str, data: bytes) -> None:
        created = self.ensure_capacity(key, data)
        directory, name = self._location(key)
        if not created:
            os.fsync(directory)
            self.release_record(key)
            return
        try:
            if self._admission is None:
                install_file(directory, name, data)
            else:
                policy = self._admission.config.access_policy
                install_file(directory, name, data, mode=policy.object_mode, group_id=policy.shared_group_id)
        except BaseException as error:
            try:
                self._inventory(clean_temporaries=True)
            except BaseException as inventory_error:
                self._poisoned = True
                raise error from inventory_error
            raise
        self._bytes += len(data)
        self._objects += 1
        self.release_record(key)

    def write_payload(self, record: ChunkRecord, payload: bytes) -> None:
        record.validate(self.limits)
        EncodedChunk(record.descriptor, payload).validate(self.limits.codec)
        if record.object_key is not None:
            self.put_object(record.object_key, payload)

    def read_payload(self, record: ChunkRecord) -> bytes:
        record.validate(self.limits)
        payload = (
            b""
            if record.object_key is None
            else self.read_object(
                record.object_key, self.limits.codec.max_chunk_bytes, length=record.descriptor.encoded_length
            )
        )
        EncodedChunk(record.descriptor, payload).validate(self.limits.codec)
        return payload

    def write_index(self, ref: PageRef, data: bytes) -> None:
        IndexPage.from_bytes(data, ref, self.limits)
        self.put_object(ref.object_key, data)

    def read_index(self, ref: PageRef) -> bytes:
        ref.validate(self.limits)
        data = self.read_object(ref.object_key, self.limits.max_index_page_bytes, length=ref.byte_length)
        IndexPage.from_bytes(data, ref, self.limits)
        return data

    def read_manifest(self, manifest_id: str) -> Manifest:
        return read_manifest(self, manifest_id)

    def validate_artifacts(self, manifest: Manifest) -> None:
        """Check coverage and stored hashes; consumers still verify decoded
        roots."""
        validate_artifacts(self, manifest)

    def write_manifest(self, manifest: Manifest) -> str:
        return write_manifest(self, manifest)

    def catalog_keys(self) -> list[str]:
        result = []
        with os.scandir(self._directories["catalog"]) as entries:
            for entry in entries:
                if _TEMP.fullmatch(entry.name):
                    continue
                key = "catalog/" + entry.name
                self._location(key)
                result.append(key)
                require_uint(len(result), "catalog records", self.storage_limits.max_records)
        return sorted(result)
