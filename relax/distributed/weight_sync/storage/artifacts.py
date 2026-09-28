# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""POSIX shared artifact store with exclusive writes and persistent quotas."""

import os
import re
import stat
import threading
from dataclasses import dataclass
from pathlib import Path

from ..codec import EncodedChunk
from ..codec.format import content_hash
from ..integrity import ModelRoot
from ..limits import DeltaCodecError, SnapshotLimits, require_uint
from ..manifest import ChunkRecord, IndexPage, Manifest, PageRef
from ..schema import require_digest
from .files import child_directory, exclusive_lock, install_file, open_directory, read_file


_KEY = re.compile(
    r"(?:chunks/[0-9a-f]{64}|(?:indexes|manifests|archives)/[0-9a-f]{64}\.json|catalog/[0-9]{20}-[0-9a-f]{64}\.json|authority\.json)"
)
_DIRECTORIES = ("chunks", "indexes", "manifests", "catalog", "archives")
_TEMP = re.compile(r"\.tmp-[0-9a-f]{32}")


@dataclass(frozen=True)
class StorageLimits:
    max_bytes: int = 1 << 40
    max_objects: int = 1 << 20
    max_records: int = 4096
    max_record_bytes: int = 16 * 1024
    max_archive_bytes: int = 1024 * 1024
    max_chain_depth: int = 128
    max_database_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            require_uint(getattr(self, name), name)
            if getattr(self, name) == 0:
                raise DeltaCodecError("storage limits must be positive")


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
    ) -> None:
        self.limits, self.storage_limits = limits, storage_limits
        self.root = Path(root).absolute()
        self._fd = open_directory(self.root, create=writable)
        self._lock: int | None = None
        self._owner = (os.getpid(), threading.get_ident())
        self._poisoned = False
        self._reservations: dict[str, tuple[int, str]] = {}
        self._directories: dict[str, int] = {}
        self._bytes = self._objects = 0
        try:
            if writable:
                self._lock = exclusive_lock(self._fd, ".writer.lock")
            for name in _DIRECTORIES:
                self._directories[name] = child_directory(self._fd, name, create=writable)
            if writable:
                self._inventory(clean_temporaries=True)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "PosixArtifactStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        for fd in self._directories.values():
            os.close(fd)
        self._directories.clear()
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

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
        total = count = 0
        for prefix, fd in (("", self._fd), *self._directories.items()):
            with os.scandir(fd) as entries:
                for entry in entries:
                    if not prefix and (entry.name in _DIRECTORIES or entry.name == ".writer.lock"):
                        continue
                    if _TEMP.fullmatch(entry.name) and clean_temporaries:
                        os.unlink(entry.name, dir_fd=fd)
                        os.fsync(fd)
                        continue
                    key = f"{prefix}/{entry.name}" if prefix else entry.name
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
        }[key.split("/")[0]]

    def ensure_capacity(self, key: str, data: bytes) -> bool:
        """Preflight under the single synchronous writer; return whether
        new."""
        if self._lock is None:
            raise DeltaCodecError("artifact store is read-only")
        if self._owner != (os.getpid(), threading.get_ident()):
            raise DeltaCodecError("artifact writes require the owning process and thread")
        if self._poisoned:
            raise DeltaCodecError("artifact inventory is uncertain; close and reopen the writer")
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
        self._reservations.pop(key, None)

    def put_object(self, key: str, data: bytes) -> None:
        created = self.ensure_capacity(key, data)
        directory, name = self._location(key)
        if not created:
            os.fsync(directory)
            self.release_record(key)
            return
        try:
            install_file(directory, name, data)
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
        require_digest(manifest_id, "manifest_id")
        data = self.read_object(f"manifests/{manifest_id}.json", self.limits.max_manifest_bytes)
        return Manifest.from_bytes(data, expected_manifest_id=manifest_id, limits=self.limits)

    def validate_artifacts(self, manifest: Manifest) -> None:
        """Check coverage and stored hashes; consumers still verify decoded
        roots."""
        manifest.validate(self.limits)
        specs = iter(manifest.schema.iter_chunks())
        root = ModelRoot(manifest.schema.schema_id, manifest.schema.directory_hash)
        for ref in manifest.pages:
            page = IndexPage.from_bytes(self.read_index(ref), ref, self.limits)
            for record in page.records:
                descriptor = record.descriptor
                if descriptor.spec != next(specs, None) or descriptor.target_version != manifest.target.version:
                    raise DeltaCodecError("artifact chunk coverage or version mismatch")
                if descriptor.base_version is not None and (
                    manifest.base is None or descriptor.base_version != manifest.base.version
                ):
                    raise DeltaCodecError("artifact base version mismatch")
                payload = self.read_payload(record)
                if descriptor.base_version is None and content_hash(payload) != descriptor.target_hash:
                    raise DeltaCodecError("RAW target hash mismatch")
                root.add(descriptor.spec, descriptor.target_hash)
        if next(specs, None) is not None or root.hexdigest() != manifest.target.target_root:
            raise DeltaCodecError("artifact directory root mismatch")

    def write_manifest(self, manifest: Manifest) -> str:
        self.validate_artifacts(manifest)
        data = manifest.to_bytes(self.limits)
        manifest_id = content_hash(data)
        self.put_object(f"manifests/{manifest_id}.json", data)
        return manifest_id

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
