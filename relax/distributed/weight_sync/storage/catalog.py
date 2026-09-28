# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Local WAL publication authority; shared files are exported decisions
only."""

import os
import sqlite3
import stat
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from ..codec.format import content_hash
from ..limits import DeltaCodecError, require_uint
from ..manifest import Manifest, SnapshotIdentity
from ..schema import require_digest
from ..serialization import canonical_json, exact_fields, parse_json, require_identifier
from .contracts import StorageLimits, StorageReader, StorageWriter
from .files import exclusive_lock, open_directory
from .repository import read_manifest, write_manifest


class PublicationUncertain(DeltaCodecError):
    """Query the same operation ID; never replace an uncertain candidate."""

    def __init__(self, operation_id: str) -> None:
        self.operation_id = operation_id
        super().__init__("publication outcome/export uncertain; resolve the same operation_id")


@dataclass(frozen=True)
class Publication:
    authority_id: str
    operation_id: str
    manifest_id: str
    target: SnapshotIdentity
    kind: str
    writer_fence: int
    expected_head: str | None
    base_manifest_id: str | None
    advances_head: bool

    @property
    def key(self) -> str:
        return f"catalog/{self.target.version:020d}-{self.manifest_id}.json"

    def to_bytes(self, limits: StorageLimits) -> bytes:
        require_identifier(self.authority_id, "authority_id")
        require_identifier(self.operation_id, "operation_id")
        require_digest(self.manifest_id, "manifest_id")
        require_uint(self.writer_fence, "writer_fence")
        if self.expected_head is not None:
            require_digest(self.expected_head, "expected_head")
        if self.base_manifest_id is not None:
            require_digest(self.base_manifest_id, "base_manifest_id")
        if self.kind not in ("FULL", "DELTA") or (self.kind == "FULL") != (self.base_manifest_id is None):
            raise DeltaCodecError("invalid publication dependency")
        if type(self.advances_head) is not bool or (not self.advances_head and self.kind != "FULL"):
            raise DeltaCodecError("only FULL may attach to an existing version")
        return canonical_json({"format_version": 1, **asdict(self)}, limits.max_record_bytes)

    @classmethod
    def from_bytes(cls, data: bytes, limits: StorageLimits) -> "Publication":
        value = exact_fields(
            parse_json(data, limits.max_record_bytes),
            {
                "format_version",
                "authority_id",
                "operation_id",
                "manifest_id",
                "target",
                "kind",
                "writer_fence",
                "expected_head",
                "base_manifest_id",
                "advances_head",
            },
        )
        version = value.pop("format_version")
        if type(version) is not int or version != 1:
            raise DeltaCodecError("unsupported publication version")
        value["target"] = SnapshotIdentity.from_dict(value["target"])
        result = cls(**value)
        if result.to_bytes(limits) != data:
            raise DeltaCodecError("noncanonical publication")
        return result

    def load_manifest(self, store: StorageReader) -> Manifest:
        manifest = read_manifest(store, self.manifest_id)
        if (manifest.target, manifest.kind, manifest.writer_fence) != (self.target, self.kind, self.writer_fence):
            raise DeltaCodecError("publication differs from manifest")
        return manifest


class ProducerCatalog:
    """One local control process, one authorized stream/epoch per archive root.

    local_control_dir MUST be on a reliable local persistent filesystem, never
    NFS/shared storage. There is no automatic cross-host takeover or epoch
    switch. Every record/dependency is retained; quotas stop progress instead
    of unsafe GC.
    """

    def __init__(self, local_control_dir: str | Path, store: StorageWriter, *, stream_id: str, run_epoch: str) -> None:
        require_identifier(stream_id, "stream_id")
        require_identifier(run_epoch, "run_epoch")
        self.store, self.stream_id, self.run_epoch = store, stream_id, run_epoch
        self.limits = store.storage_limits
        self._fd = -1
        self._lock: int | None = None
        self._db: sqlite3.Connection | None = None
        control = Path(local_control_dir).absolute()
        capabilities = store.capabilities()
        if capabilities.access != "publisher":
            raise DeltaCodecError("producer catalog requires a publisher store")
        if capabilities.stream_id is not None and (capabilities.stream_id, capabilities.run_epoch) != (
            stream_id,
            run_epoch,
        ):
            raise DeltaCodecError("publication differs from deployment namespace")
        control_guard = store.prepare_control(control)
        try:
            self._fd = open_directory(control, create=True)
            control_guard.validate_opened(self._fd)
            self._lock = exclusive_lock(self._fd, ".authority.lock")
            store.begin_recovery()
            for name in ("catalog.sqlite3", "catalog.sqlite3-wal", "catalog.sqlite3-shm"):
                try:
                    info = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise DeltaCodecError("control database must use regular files")
            self._db = sqlite3.connect(control / "catalog.sqlite3", isolation_level=None, timeout=0)
            control_guard.validate_opened(self._fd)
            if self._db.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() != "wal":
                raise DeltaCodecError("SQLite WAL is required")
            self._db.execute("PRAGMA synchronous=FULL")
            if self._db.execute("PRAGMA synchronous").fetchone()[0] != 2:
                raise DeltaCodecError("SQLite FULL synchronous mode is required")
            self._db.execute("PRAGMA journal_size_limit=0")
            self._db.execute("PRAGMA wal_autocheckpoint=1")
            page_size = self._db.execute("PRAGMA page_size").fetchone()[0]
            pages = self.limits.max_database_bytes // page_size
            if pages < 16 or self._db.execute("PRAGMA page_count").fetchone()[0] > pages:
                raise DeltaCodecError("control database exceeds budget")
            self._db.execute(f"PRAGMA max_page_count={pages}")
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), authority TEXT NOT NULL,
                    stream TEXT NOT NULL, epoch TEXT NOT NULL, fence INTEGER NOT NULL, head TEXT);
                CREATE TABLE IF NOT EXISTS publications (operation TEXT PRIMARY KEY, manifest TEXT UNIQUE NOT NULL,
                    version TEXT NOT NULL, depth INTEGER NOT NULL, body BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS versions (version TEXT PRIMARY KEY, primary_manifest TEXT NOT NULL,
                    full_manifest TEXT);
            """)
            self._db.execute(
                "INSERT OR IGNORE INTO state VALUES(1, ?, ?, ?, 0, NULL)", (uuid.uuid4().hex, stream_id, run_epoch)
            )
            authority, stream, epoch = self._db.execute("SELECT authority, stream, epoch FROM state").fetchone()
            if (stream, epoch) != (stream_id, run_epoch):
                raise DeltaCodecError("control database stream/epoch mismatch")
            self.authority_id = authority
            binding = canonical_json(
                {"format_version": 1, "authority_id": authority, "stream_id": stream, "run_epoch": epoch},
                self.limits.max_record_bytes,
            )
            self._checkpoint()
            os.fsync(self._fd)
            store.put_immutable("authority.json", binding, expected_hash=content_hash(binding))
            self.export_pending()
            store.finish_recovery()
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "ProducerCatalog":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def _connection(self) -> sqlite3.Connection:
        if self._db is None:
            raise DeltaCodecError("producer catalog is closed")
        return self._db

    def _checkpoint(self) -> None:
        if self._connection().execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] != 0:
            raise DeltaCodecError("WAL checkpoint blocked; stop writes to preserve budget")

    def _commit(self) -> None:
        self._connection().execute("COMMIT")

    def acquire_writer(self) -> int:
        db = self._connection()
        self._checkpoint()
        db.execute("BEGIN IMMEDIATE")
        try:
            fence = db.execute("SELECT fence FROM state").fetchone()[0] + 1
            require_uint(fence, "writer fence", (1 << 63) - 1)
            db.execute("UPDATE state SET fence=?", (fence,))
            self._commit()
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        self._checkpoint()
        return fence

    def head(self) -> str | None:
        return self._connection().execute("SELECT head FROM state").fetchone()[0]

    def resolve(self, operation_id: str) -> Publication | None:
        require_identifier(operation_id, "operation_id")
        row = self._connection().execute("SELECT body FROM publications WHERE operation=?", (operation_id,)).fetchone()
        return None if row is None else Publication.from_bytes(row[0], self.limits)

    def _by_manifest(self, manifest_id: str) -> tuple[Publication, int]:
        row = (
            self._connection()
            .execute("SELECT body, depth FROM publications WHERE manifest=?", (manifest_id,))
            .fetchone()
        )
        if row is None:
            raise DeltaCodecError("base or head is not committed")
        return Publication.from_bytes(row[0], self.limits), row[1]

    def _export(self, record: Publication) -> None:
        data = record.to_bytes(self.limits)
        self.store.put_immutable(record.key, data, expected_hash=content_hash(data))

    def export_pending(self) -> None:
        # No mutable exported flag: idempotent re-export also repairs a missing
        # record after a crash. Final committed objects are never reclaimed.
        cursor = self._connection().execute("SELECT body FROM publications ORDER BY version, manifest")
        for count, row in enumerate(cursor, 1):
            require_uint(count, "catalog records", self.limits.max_records)
            self._export(Publication.from_bytes(row[0], self.limits))

    def publish(
        self, manifest: Manifest, *, operation_id: str, expected_head: str | None, attach_full: bool = False
    ) -> Publication:
        require_identifier(operation_id, "operation_id")
        if expected_head is not None:
            require_digest(expected_head, "expected_head")
        if type(attach_full) is not bool:
            raise DeltaCodecError("attach_full must be boolean")
        manifest_id = manifest.manifest_id(self.store.limits)
        existing = self.resolve(operation_id)
        if existing is not None:
            if (existing.manifest_id, existing.expected_head, existing.advances_head) != (
                manifest_id,
                expected_head,
                not attach_full,
            ):
                raise DeltaCodecError("operation_id reused for a different request")
            try:
                self._export(existing)
            except Exception as error:
                raise PublicationUncertain(operation_id) from error
            return existing
        if (manifest.target.stream_id, manifest.target.run_epoch) != (self.stream_id, self.run_epoch):
            raise DeltaCodecError("publication stream/epoch mismatch")
        db = self._connection()
        self._checkpoint()
        # Data durability precedes the authority decision; failures can leave
        # immutable orphans, retained within the store quota for explicit GC.
        write_manifest(self.store, manifest)
        db.execute("BEGIN IMMEDIATE")
        reserved_key = None
        try:
            fence, head = db.execute("SELECT fence, head FROM state").fetchone()
            if manifest.writer_fence != fence or fence == 0 or head != expected_head:
                raise DeltaCodecError("stale writer fence or expected head")
            if db.execute("SELECT count(*) FROM publications").fetchone()[0] >= self.limits.max_records:
                raise DeltaCodecError("catalog record budget exhausted")
            current = None if head is None else self._by_manifest(head)[0]
            if current is not None and current.target.schema_id != manifest.target.schema_id:
                raise DeltaCodecError("schema change requires a separately authorized epoch")
            version = f"{manifest.target.version:020d}"
            base_id = None
            depth = 0
            if attach_full:
                row = db.execute(
                    "SELECT primary_manifest, full_manifest FROM versions WHERE version=?", (version,)
                ).fetchone()
                if row is None or manifest.kind != "FULL" or row[1] is not None:
                    raise DeltaCodecError("FULL attachment requires an existing version without FULL")
                if self._by_manifest(row[0])[0].target != manifest.target:
                    raise DeltaCodecError("FULL attachment changes version identity")
            else:
                if current is None and manifest.kind != "FULL":
                    raise DeltaCodecError("first published version must be FULL")
                if current is not None and manifest.target.version <= current.target.version:
                    raise DeltaCodecError("publication must advance the version")
                if manifest.base is not None:
                    row = db.execute(
                        "SELECT COALESCE(full_manifest, primary_manifest) FROM versions WHERE version=?",
                        (f"{manifest.base.version:020d}",),
                    ).fetchone()
                    if row is None:
                        raise DeltaCodecError("delta base is not committed")
                    base_record, depth = self._by_manifest(row[0])
                    if base_record.target != manifest.base:
                        raise DeltaCodecError("delta base identity mismatch")
                    base_id, depth = row[0], depth + 1
                    require_uint(depth, "delta chain depth", self.limits.max_chain_depth)
            record = Publication(
                self.authority_id,
                operation_id,
                manifest_id,
                manifest.target,
                manifest.kind,
                manifest.writer_fence,
                expected_head,
                base_id,
                not attach_full,
            )
            # The exclusive synchronous store writer cannot consume this space
            # between the preflight and export. Known quota exhaustion must not
            # create an already-committed record that can never be exported.
            self.store.reserve_record(record.key, record.to_bytes(self.limits))
            reserved_key = record.key
            if db.execute("SELECT 1 FROM publications WHERE manifest=?", (manifest_id,)).fetchone():
                raise DeltaCodecError("manifest already committed under a different operation")
            db.execute(
                "INSERT INTO publications VALUES(?,?,?,?,?)",
                (operation_id, manifest_id, version, depth, record.to_bytes(self.limits)),
            )
            if attach_full:
                db.execute("UPDATE versions SET full_manifest=? WHERE version=?", (manifest_id, version))
            else:
                db.execute(
                    "INSERT INTO versions VALUES(?,?,?)",
                    (version, manifest_id, manifest_id if manifest.kind == "FULL" else None),
                )
                db.execute("UPDATE state SET head=?", (manifest_id,))
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            if reserved_key is not None:
                self.store.release_record(reserved_key)
            raise
        try:
            self._commit()
            self._checkpoint()
            self._export(record)
        except Exception as error:
            if db.in_transaction:
                db.execute("ROLLBACK")
                self.store.release_record(record.key)
            raise PublicationUncertain(operation_id) from error
        return record
