# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Quota-bounded disk generations; sealed snapshots are immutable and
retained."""

import os
import re
import stat
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ..codec.format import content_hash
from ..limits import DeltaCodecError, SnapshotLimits, require_uint
from ..manifest import SnapshotIdentity
from ..model import ModelSchema
from ..schema import CanonicalChunk, ChunkSpec
from ..serialization import canonical_json, exact_fields, parse_json
from ..snapshot import VerifiedSnapshot, verify_snapshot
from .files import child_directory, exclusive_lock, install_file, open_directory, read_file, write_all


if TYPE_CHECKING:
    from ..export import ExportBudget, ExportRequest, SourceExportPlan
    from .capture import DiskCapture


class _UnfinishedGeneration(Protocol):
    def abort(self) -> None: ...


_GENERATION = re.compile(r"generation-[0-9a-f]{32}")
_TEMP = re.compile(r"\.tmp-[0-9a-f]{32}")


class DiskSnapshotReader:
    """Explicitly close after all VerifiedSnapshot users release their
    leases."""

    def __init__(self, directory: int, schema: ModelSchema, limits: SnapshotLimits) -> None:
        self.schema, self.limits = schema, limits
        self._fd = os.open("weights.bin", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        info = os.fstat(self._fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != schema.canonical_nbytes:
            self.close()
            raise DeltaCodecError("snapshot storage length/type mismatch")
        self._entries = {entry.tensor.name: entry for entry in schema.tensors}
        self._offsets = {}
        offset = 0
        for entry in schema.tensors:
            if entry.alias_of is None:
                self._offsets[entry.tensor.name] = offset
                offset += entry.tensor.nbytes
        for entry in schema.tensors:
            if entry.alias_of is not None:
                self._offsets[entry.tensor.name] = self._offsets[entry.alias_of]

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __del__(self) -> None:
        if getattr(self, "_fd", -1) >= 0:
            os.close(self._fd)

    def read_chunk(self, spec: ChunkSpec) -> bytes:
        spec.validate(self.limits.codec)
        entry = self._entries.get(spec.tensor.name)
        if self._fd < 0 or spec.schema_id != self.schema.schema_id or entry is None or entry.tensor != spec.tensor:
            raise DeltaCodecError("snapshot reader is closed or chunk identity is incompatible")
        offset = self._offsets[spec.tensor.name] + spec.byte_offset
        data = bytearray()
        while len(data) < spec.byte_length:
            part = os.pread(self._fd, spec.byte_length - len(data), offset + len(data))
            if not part:
                raise DeltaCodecError("truncated snapshot storage")
            data.extend(part)
        return bytes(data)


def open_snapshot(
    path: str | Path, *, expected_identity: SnapshotIdentity, limits: SnapshotLimits = SnapshotLimits()
) -> tuple[VerifiedSnapshot, DiskSnapshotReader]:
    """Reopen a sealed generation and hash its actual bytes before accepting
    it."""
    directory = open_directory(path)
    reader = None
    try:
        seal = exact_fields(
            parse_json(read_file(directory, "sealed.json", 256), 256), {"format_version", "metadata_hash"}
        )
        if type(seal["format_version"]) is not int or seal["format_version"] != 1:
            raise DeltaCodecError("unsupported snapshot seal")
        data = read_file(directory, "metadata.json", limits.max_manifest_bytes)
        if content_hash(data) != seal["metadata_hash"]:
            raise DeltaCodecError("snapshot metadata hash mismatch")
        value = exact_fields(parse_json(data, limits.max_manifest_bytes), {"format_version", "schema", "identity"})
        if type(value["format_version"]) is not int or value["format_version"] != 1:
            raise DeltaCodecError("unsupported snapshot metadata")
        identity = SnapshotIdentity.from_dict(value["identity"])
        if identity != expected_identity:
            raise DeltaCodecError("snapshot differs from trusted identity")
        schema = ModelSchema.from_bytes(canonical_json(value["schema"], limits.max_directory_bytes), limits)
        reader = DiskSnapshotReader(directory, schema, limits)
        return verify_snapshot(schema, identity, reader, limits=limits), reader
    except BaseException:
        if reader is not None:
            reader.close()
        raise
    finally:
        os.close(directory)


class DiskSnapshotStore:
    """One staging writer, bounded generations/bytes, no sealed-generation GC.

    close aborts this instance's unfinished generations. Crash leftovers can be
    explicitly cleaned under the exclusive writer lock; sealed files and other
    live writers are never reclaimed. Disk readers do not need the writer lock.
    """

    def __init__(
        self, root: str | Path, *, max_bytes: int, max_generations: int, limits: SnapshotLimits = SnapshotLimits()
    ) -> None:
        require_uint(max_bytes, "snapshot disk budget")
        require_uint(max_generations, "snapshot generation budget")
        if not max_bytes or not max_generations:
            raise DeltaCodecError("snapshot disk budgets must be positive")
        self.root, self.limits = Path(root).absolute(), limits
        self._owner = (os.getpid(), threading.get_ident())
        self.max_bytes, self.max_generations = max_bytes, max_generations
        self._fd = open_directory(self.root, create=True)
        self._lock: int | None = None
        self._active: dict[str, _UnfinishedGeneration] = {}
        try:
            self._lock = exclusive_lock(self._fd, ".writer.lock")
            self._usage(clean_temporaries=True)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "DiskSnapshotStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._assert_owner()
        try:
            for stage in tuple(self._active.values()):
                stage.abort()
        finally:
            if self._lock is not None:
                os.close(self._lock)
                self._lock = None
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1

    def _assert_owner(self) -> None:
        if self._owner != (os.getpid(), threading.get_ident()):
            raise DeltaCodecError("snapshot writes require the owning process and thread")

    def _generations(self) -> list[str]:
        self._assert_owner()
        if self._fd < 0:
            raise DeltaCodecError("snapshot store is closed")
        result = []
        with os.scandir(self._fd) as entries:
            for entry in entries:
                if entry.name == ".writer.lock":
                    continue
                if _GENERATION.fullmatch(entry.name) is None:
                    raise DeltaCodecError("unknown snapshot store entry")
                result.append(entry.name)
                require_uint(len(result), "snapshot generation count", self.max_generations)
        return result

    def _usage(self, *, clean_temporaries: bool = False) -> tuple[int, int]:
        generations = self._generations()
        total = 0
        for name in generations:
            directory = child_directory(self._fd, name)
            try:
                with os.scandir(directory) as entries:
                    count = 0
                    for entry in entries:
                        count += 1
                        require_uint(count, "generation file count", 4)
                        if clean_temporaries and _TEMP.fullmatch(entry.name):
                            os.unlink(entry.name, dir_fd=directory)
                            os.fsync(directory)
                            continue
                        if entry.name not in ("weights.bin", "metadata.json", "sealed.json") and not _TEMP.fullmatch(
                            entry.name
                        ):
                            raise DeltaCodecError("unknown snapshot generation entry")
                        info = entry.stat(follow_symlinks=False)
                        if not stat.S_ISREG(info.st_mode):
                            raise DeltaCodecError("snapshot generation contains a non-regular file")
                        total += info.st_size
                        require_uint(total, "snapshot disk bytes", self.max_bytes)
            finally:
                os.close(directory)
        return total, len(generations)

    def staging(self) -> "DiskStaging":
        self._assert_owner()
        if self._lock is None:
            raise DeltaCodecError("snapshot store is closed")
        return DiskStaging(self)

    def capture(self, plan: "SourceExportPlan", request: "ExportRequest", budget: "ExportBudget") -> "DiskCapture":
        """Capture a new source snapshot whose content root is not yet
        known."""
        from .capture import DiskCapture

        return DiskCapture(self, plan, request, budget)

    def _discard(self, name: str) -> None:
        directory = child_directory(self._fd, name)
        try:
            names = []
            with os.scandir(directory) as entries:
                for entry in entries:
                    names.append(entry.name)
                    require_uint(len(names), "generation file count", 4)
            if len(names) > 4 or any(
                name not in ("weights.bin", "metadata.json", "sealed.json") and not _TEMP.fullmatch(name)
                for name in names
            ):
                raise DeltaCodecError("refuse to remove unknown generation contents")
            for entry in names:
                os.unlink(entry, dir_fd=directory)
            os.fsync(directory)
        finally:
            os.close(directory)
        os.rmdir(name, dir_fd=self._fd)
        os.fsync(self._fd)

    def cleanup_incomplete(self) -> int:
        removed = 0
        for name in self._generations():
            if name in self._active:
                continue
            directory = child_directory(self._fd, name)
            try:
                try:
                    os.stat("sealed.json", dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    incomplete = True
                else:
                    incomplete = False
            finally:
                os.close(directory)
            if incomplete:
                self._discard(name)
                removed += 1
        return removed


class DiskStaging:
    def __init__(self, store: DiskSnapshotStore) -> None:
        self.store = store
        self.name = "generation-" + uuid.uuid4().hex
        self.path = store.root / self.name
        self._directory = self._writer = -1
        self._reader: DiskSnapshotReader | None = None
        self._created = self._sealed = self._started = False

    def begin(self, schema: ModelSchema, identity: SnapshotIdentity) -> None:
        self.store._assert_owner()
        if self._started:
            raise DeltaCodecError("staging generations are single-use")
        self._started = True
        if self.store._active:
            raise DeltaCodecError("only one unfinished generation is allowed per snapshot store")
        schema.validate(self.store.limits)
        if identity.schema_id != schema.schema_id:
            raise DeltaCodecError("staging schema mismatch")
        self._schema, self._identity = schema, identity
        self._specs = iter(schema.iter_chunks())
        self._metadata = canonical_json(
            {"format_version": 1, "schema": schema.to_dict(), "identity": asdict(identity)},
            self.store.limits.max_manifest_bytes,
        )
        total, count = self.store._usage()
        require_uint(count + 1, "snapshot generation count", self.store.max_generations)
        require_uint(
            total + schema.canonical_nbytes + len(self._metadata) + 256, "snapshot disk bytes", self.store.max_bytes
        )
        os.mkdir(self.name, mode=0o700, dir_fd=self.store._fd)
        self._created = True
        self.store._active[self.name] = self
        self._directory = child_directory(self.store._fd, self.name)
        os.fsync(self.store._fd)
        self._writer = os.open(
            "weights.bin", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self._directory
        )
        os.ftruncate(self._writer, schema.canonical_nbytes)
        install_file(self._directory, "metadata.json", self._metadata)
        self._reader = DiskSnapshotReader(self._directory, schema, self.store.limits)

    def write_chunk(self, chunk: CanonicalChunk) -> None:
        self.store._assert_owner()
        if (
            self._writer < 0
            or self._sealed
            or chunk.version != self._identity.version
            or chunk.spec != next(self._specs, None)
        ):
            raise DeltaCodecError("staging chunk identity/order mismatch")
        write_all(self._writer, chunk.data)

    def reader(self) -> DiskSnapshotReader:
        if self._reader is None:
            raise DeltaCodecError("staging reader is unavailable")
        return self._reader

    def seal(self) -> None:
        self.store._assert_owner()
        if self._writer < 0 or self._sealed or next(self._specs, None) is not None:
            raise DeltaCodecError("incomplete or closed staging generation")
        os.fchmod(self._writer, 0o444)
        os.fsync(self._writer)
        seal = canonical_json({"format_version": 1, "metadata_hash": content_hash(self._metadata)}, 256)
        install_file(self._directory, "sealed.json", seal)
        os.close(self._writer)
        self._writer = -1
        os.close(self._directory)
        self._directory = -1
        self._sealed = True
        self.store._active.pop(self.name)

    def abort(self) -> None:
        self.store._assert_owner()
        if self._sealed:
            raise DeltaCodecError("sealed generations require a separate retention protocol")
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        for name in ("_writer", "_directory"):
            fd = getattr(self, name)
            if fd >= 0:
                os.close(fd)
                setattr(self, name, -1)
        if self._created:
            self.store._discard(self.name)
            self._created = False
            self.store._active.pop(self.name, None)
