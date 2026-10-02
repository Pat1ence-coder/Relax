# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Single-use export capture with an initially unknown target root.

Uses the snapshot store's lock, quotas and generation format. Capturing and
reconstruction have separate state machines: no provisional identity is ever
written or exposed, and a failed candidate cannot be resumed or published.
"""

import hashlib
import os
import uuid
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from ..codec.format import content_hash
from ..export import CanonicalTile, ExportBudget, ExportRequest, SourceExportPlan, SourceReceipt, validate_receipts
from ..integrity import ModelRoot
from ..limits import DeltaCodecError, require_uint
from ..schema import ChunkSpec
from ..serialization import canonical_json
from ..snapshot import VerifiedSnapshot, verify_snapshot
from .files import child_directory, install_file, write_all
from .staging import DiskSnapshotReader, DiskSnapshotStore


class FrozenExport:
    """Owns a verified disk reader, independently of the source and store.

    Closing releases the reader only. Sealed generations remain charged to disk
    quotas; retention/GC and publication belong to a separate owner.
    """

    def __init__(
        self,
        snapshot: VerifiedSnapshot,
        reader: DiskSnapshotReader,
        request: ExportRequest,
        plan_id: str,
        path: Path,
    ) -> None:
        self._snapshot, self._reader = snapshot, reader
        self.request, self.plan_id, self.path = request, plan_id, path
        self._closed = False

    @property
    def snapshot(self) -> VerifiedSnapshot:
        if self._closed:
            raise DeltaCodecError("frozen export lease is closed")
        return self._snapshot

    def read_chunk(self, spec: ChunkSpec) -> bytes:
        return self.snapshot.read_chunk(spec)

    def close(self) -> None:
        self._reader.close()
        self._closed = True

    def __enter__(self) -> "FrozenExport":
        self.snapshot
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class DiskCapture:
    def __init__(
        self, store: DiskSnapshotStore, plan: SourceExportPlan, request: ExportRequest, budget: ExportBudget
    ) -> None:
        self.store, self.plan, self.request, self.budget = store, plan, request, budget
        self.name = "generation-" + uuid.uuid4().hex
        self.path = store.root / self.name
        self._directory = self._writer = -1
        self._reader: DiskSnapshotReader | None = None
        self._state = "new"
        self._created = False
        store._assert_owner()
        if store._lock is None:
            raise DeltaCodecError("snapshot store is closed")
        if store._active:
            raise DeltaCodecError("only one unfinished generation is allowed per snapshot store")
        plan.schema.validate(store.limits)
        budget.validate_schema(plan.schema)
        # This fixed-width digest is only for quota sizing, never an identity
        # stored in a generation. metadata.json is written after capture.
        metadata_size = len(self._metadata("f" * 64))
        used, count = store._usage()
        require_uint(count + 1, "snapshot generation count", store.max_generations)
        require_uint(
            used + plan.schema.canonical_nbytes + 2 * metadata_size + 512,
            "capture disk reservation",
            store.max_bytes,
        )
        self._specs = iter(plan.schema.iter_chunks())
        self._expected = next(self._specs, None)
        self._offset = 0
        self._chunk_hash = hashlib.sha256()
        self._root = ModelRoot(plan.schema.schema_id, plan.schema.directory_hash)
        self._owners = {owner.name: owner.rank for owner in plan.owners}
        self._request_id = request.request_id
        try:
            os.mkdir(self.name, mode=0o700, dir_fd=store._fd)
            self._created = True
            store._active[self.name] = self
            self._directory = child_directory(store._fd, self.name)
            os.fsync(store._fd)
            self._writer = os.open(
                "weights.bin", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self._directory
            )
            os.ftruncate(self._writer, plan.schema.canonical_nbytes)
            self._state = "capturing"
        except BaseException:
            self._state = "failed"
            self.abort()
            raise

    def _metadata(self, root: str) -> bytes:
        return canonical_json(
            {
                "format_version": 1,
                "schema": self.plan.schema.to_dict(),
                "identity": asdict(self.request.identity(self.plan.schema, root)),
            },
            self.store.limits.max_manifest_bytes,
        )

    def __enter__(self) -> "DiskCapture":
        self._require_capturing()
        return self

    def __exit__(self, *args: object) -> None:
        if self._state != "sealed":
            self.abort()

    def _require_capturing(self) -> None:
        self.store._assert_owner()
        if self._state != "capturing":
            raise DeltaCodecError("capture is not active; failed captures cannot be resumed")

    def write_tile(self, tile: CanonicalTile) -> None:
        self._require_capturing()
        try:
            if (
                not isinstance(tile, CanonicalTile)
                or tile.request_id != self._request_id
                or tile.plan_id != self.plan.plan_id
                or tile.spec != self._expected
                or tile.byte_offset != self._offset
                or tile.rank != self._owners.get(tile.spec.tensor.name)
            ):
                raise DeltaCodecError("capture tile identity, owner, order or coverage mismatch")
            require_uint(len(tile.data), "capture tile bytes", self.budget.tile_bytes)
            write_all(self._writer, tile.data)
            self._chunk_hash.update(tile.data)
            self._offset += len(tile.data)
            if self._offset == tile.spec.byte_length:
                self._root.add(tile.spec, self._chunk_hash.hexdigest())
                self._expected = next(self._specs, None)
                self._offset = 0
                self._chunk_hash = hashlib.sha256()
        except BaseException:
            self._state = "failed"
            raise

    def finalize(self, receipts: Iterable[SourceReceipt]) -> FrozenExport:
        self._require_capturing()
        self._state = "finalizing"
        try:
            if self._expected is not None:
                raise DeltaCodecError("capture directory coverage is incomplete")
            validate_receipts(self.plan, self.request, receipts)
            identity = self.request.identity(self.plan.schema, self._root.hexdigest())
            os.fchmod(self._writer, 0o444)
            os.fsync(self._writer)
            os.close(self._writer)
            self._writer = -1
            self._reader = DiskSnapshotReader(self._directory, self.plan.schema, self.store.limits)
            verified = verify_snapshot(self.plan.schema, identity, self._reader, limits=self.store.limits)
            metadata = self._metadata(identity.target_root)
            install_file(self._directory, "metadata.json", metadata)
            seal = canonical_json({"format_version": 1, "metadata_hash": content_hash(metadata)}, 256)
            install_file(self._directory, "sealed.json", seal)
            result = FrozenExport(verified, self._reader, self.request, self.plan.plan_id, self.path)
            os.close(self._directory)
            self._directory = -1
            self.store._active.pop(self.name)
            self._reader = None
            self._state = "sealed"
            return result
        except BaseException:
            if self._reader is not None:
                self._reader.close()
                self._reader = None
            # Even a linked seal still belongs to this unpublished candidate
            # if fsync/installation failed. abort may reclaim that generation.
            self._state = "failed"
            raise

    def abort(self) -> None:
        self.store._assert_owner()
        if self._state == "sealed":
            raise DeltaCodecError("sealed captures require a separate retention protocol")
        self._state = "failed"
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        for name in ("_writer", "_directory"):
            fd = getattr(self, name)
            if fd >= 0:
                os.close(fd)
                setattr(self, name, -1)
        if self._created:
            try:
                self.store._discard(self.name)
            except FileNotFoundError:
                # A previous attempt may have removed the directory and then
                # failed its final parent fsync. Retry that durability step;
                # missing children of a still-present generation remain errors.
                try:
                    os.stat(self.name, dir_fd=self.store._fd, follow_symlinks=False)
                except FileNotFoundError:
                    os.fsync(self.store._fd)
                else:
                    raise
            self._created = False
            self.store._active.pop(self.name, None)
