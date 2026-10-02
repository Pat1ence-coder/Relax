# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Independent local consumer WAL authority; never a producer catalog writer.

The journal decides the logical head. It cannot stop GPU writes, open a request
gate, or establish physical isolation from a previous process. Recovery must
first close and drain/fence execution, even when ACTIVE was never recorded.
"""

import os
import sqlite3
import stat
import threading
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from ..codec.format import content_hash
from ..consumer import Installation, InstallCommand, Member, RankReceipt, certify_receipts
from ..limits import DeltaCodecError, require_uint
from ..manifest import SnapshotIdentity
from ..serialization import canonical_json, parse_json, require_identifier
from .contracts import StorageLimits
from .files import exclusive_lock, open_directory
from .placement import DirectoryGuard, mount_for, read_mounts, require_mount_separation, require_separate


class ConsumerUncertain(DeltaCodecError):
    """Query the identical installation/operation before making another
    choice."""


def installation_from_dict(value: dict) -> Installation:
    value = dict(value)
    value["snapshot"] = SnapshotIdentity.from_dict(value["snapshot"])
    value["members"] = tuple(Member(**member) for member in value["members"])
    return Installation(**value)


class ConsumerJournal:
    """One process/thread and one local persistent control directory.

    All installations, command intents, outcomes and snapshot references remain
    retained. Quotas stop admission rather than garbage-collecting recovery
    data. SQLite commit errors are UNKNOWN until queried, including lost
    responses.
    """

    def __init__(
        self,
        local_control_dir: str | Path,
        *,
        consumer_id: str,
        snapshot_root: str | Path,
        artifact_root: str | Path | None = None,
        limits: StorageLimits = StorageLimits(),
    ):
        require_identifier(consumer_id, "consumer ID")
        self.limits, self.consumer_id = limits, consumer_id
        self._owner = os.getpid(), threading.get_ident()
        self._fd, self._lock, self._db = -1, None, None
        control = Path(local_control_dir).absolute()
        snapshots = Path(snapshot_root).absolute()
        require_separate(control, snapshots)
        roots = [snapshots] + ([] if artifact_root is None else [Path(artifact_root).absolute()])
        mounts = read_mounts()
        parent_mount = mount_for(control if control.exists() else control.parent, mounts)
        if parent_mount.readonly or parent_mount.filesystem_type not in {
            "ext2",
            "ext3",
            "ext4",
            "xfs",
            "btrfs",
            "zfs",
            "f2fs",
            "overlay",
        }:
            raise DeltaCodecError("consumer authority requires a writable local persistent control volume")
        for root in roots:
            require_separate(control, root)
            root_mount = mount_for(root, mounts)
            if control.exists():
                require_mount_separation(control, parent_mount, root, root_mount)
        guard = DirectoryGuard.capture(control, mount_id=parent_mount.mount_id, must_exist=False)
        try:
            self._fd = open_directory(control, create=True)
            guard.validate_opened(self._fd)
            info = os.fstat(self._fd)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise DeltaCodecError("consumer control directory must be private and owned")
            for root in roots:
                require_mount_separation(control, parent_mount, root, mount_for(root, mounts))
            self._lock = exclusive_lock(self._fd, ".consumer.lock")
            for name in ("consumer.sqlite3", "consumer.sqlite3-wal", "consumer.sqlite3-shm"):
                try:
                    info = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise DeltaCodecError("consumer database must be a private regular file")
            self._db = sqlite3.connect(control / "consumer.sqlite3", isolation_level=None, timeout=0)
            guard.validate_opened(self._fd)
            db = self._connection()
            if db.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() != "wal":
                raise DeltaCodecError("consumer requires SQLite WAL")
            db.execute("PRAGMA synchronous=FULL")
            if db.execute("PRAGMA synchronous").fetchone()[0] != 2:
                raise DeltaCodecError("consumer requires FULL synchronous mode")
            db.execute("PRAGMA journal_size_limit=0")
            db.execute("PRAGMA wal_autocheckpoint=1")
            pages = limits.max_database_bytes // db.execute("PRAGMA page_size").fetchone()[0]
            if pages < 16 or db.execute("PRAGMA page_count").fetchone()[0] > pages:
                raise DeltaCodecError("consumer database exceeds budget")
            db.execute(f"PRAGMA max_page_count={pages}")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1),
                    consumer TEXT NOT NULL, fence INTEGER NOT NULL, head TEXT);
                CREATE TABLE IF NOT EXISTS installs (id TEXT PRIMARY KEY, body BLOB NOT NULL,
                    expected_head TEXT, snapshot_ref TEXT NOT NULL, phase TEXT NOT NULL,
                    decision TEXT, decision_id TEXT, certificate TEXT, inherited_commit TEXT);
                CREATE TABLE IF NOT EXISTS executions (id TEXT PRIMARY KEY, members BLOB NOT NULL,
                    binding BLOB NOT NULL, isolation_evidence TEXT);
                CREATE TABLE IF NOT EXISTS commands (install TEXT NOT NULL, sequence TEXT NOT NULL,
                    digest TEXT UNIQUE NOT NULL, body BLOB NOT NULL, result BLOB,
                    PRIMARY KEY(install, sequence));
            """)
            db.execute("INSERT OR IGNORE INTO state VALUES(1, ?, 0, NULL)", (consumer_id,))
            if db.execute("SELECT consumer FROM state").fetchone()[0] != consumer_id:
                raise DeltaCodecError("consumer journal identity mismatch")
            self._checkpoint()
            os.fsync(self._fd)
        except BaseException:
            self.close()
            raise

    def _connection(self) -> sqlite3.Connection:
        if self._owner != (os.getpid(), threading.get_ident()):
            raise DeltaCodecError("consumer authority requires its owning process/thread")
        if self._db is None:
            raise DeltaCodecError("consumer journal is closed")
        return self._db

    def _checkpoint(self) -> None:
        if self._connection().execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] != 0:
            raise DeltaCodecError("consumer WAL checkpoint blocked")

    def _commit(self) -> None:
        self._connection().execute("COMMIT")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        db = self._connection()
        self._checkpoint()
        db.execute("BEGIN IMMEDIATE")
        committing = False
        try:
            yield db
            committing = True
            self._commit()
        except BaseException as exc:
            if db.in_transaction:
                db.execute("ROLLBACK")
            if committing:
                raise ConsumerUncertain("consumer COMMIT outcome is uncertain; query the same operation") from exc
            raise

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

    def __enter__(self) -> "ConsumerJournal":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def acquire_owner(self) -> int:
        with self._transaction() as db:
            fence = db.execute("SELECT fence FROM state").fetchone()[0] + 1
            require_uint(fence, "consumer fence", (1 << 63) - 1)
            db.execute("UPDATE state SET fence=?", (fence,))
        return fence

    def _fence(self, db: sqlite3.Connection, installation: Installation) -> None:
        if (
            installation.consumer_id != self.consumer_id
            or installation.owner_fence != db.execute("SELECT fence FROM state").fetchone()[0]
        ):
            raise DeltaCodecError("stale consumer owner fence")

    def head(self) -> str | None:
        return self._connection().execute("SELECT head FROM state").fetchone()[0]

    def get(self, installation_id: str) -> dict | None:
        require_identifier(installation_id, "installation ID")
        row = (
            self._connection()
            .execute(
                "SELECT body, expected_head, snapshot_ref, phase, decision, decision_id, certificate, inherited_commit "
                "FROM installs WHERE id=?",
                (installation_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return dict(
            zip(
                (
                    "installation",
                    "expected_head",
                    "snapshot_ref",
                    "phase",
                    "decision",
                    "decision_id",
                    "certificate",
                    "inherited_commit",
                ),
                (installation_from_dict(parse_json(row[0], self.limits.max_record_bytes)), *row[1:]),
            )
        )

    def _require(self, installation: Installation) -> dict:
        row = self.get(installation.installation_id)
        if row is None or row["installation"] != installation:
            raise DeltaCodecError("unknown or mismatched installation")
        return row

    def begin(self, installation: Installation, *, expected_head: str | None, snapshot_ref: str) -> None:
        require_identifier(snapshot_ref, "snapshot generation reference")
        body = canonical_json(asdict(installation), self.limits.max_record_bytes)
        with self._transaction() as db:
            self._fence(db, installation)
            existing = self.get(installation.installation_id)
            if existing is not None:
                if (existing["installation"], existing["expected_head"], existing["snapshot_ref"]) != (
                    installation,
                    expected_head,
                    snapshot_ref,
                ):
                    raise DeltaCodecError("installation ID reused with different arguments")
                return
            if self.head() != expected_head:
                raise DeltaCodecError("consumer expected head changed")
            if db.execute("SELECT count(*) FROM installs").fetchone()[0] >= self.limits.max_records:
                raise DeltaCodecError("consumer installation quota exhausted")
            pending = db.execute(
                "SELECT count(*) FROM installs WHERE phase NOT IN ('ACTIVE', 'ABORTED', 'RETIRED')"
            ).fetchone()[0]
            if pending:
                raise DeltaCodecError("resolve the previous installation before starting another")
            for record in self.recovery_records():
                if record["phase"] == "RETIRED":
                    continue
                prior = record["installation"]
                if (
                    record["phase"] == "ABORTED"
                    or prior.members != installation.members
                    or prior.owner_fence != installation.owner_fence
                ):
                    raise DeltaCodecError("physically retire the previous execution group before replacement")
            inherited = None
            if expected_head is not None:
                head_record = self.get(expected_head)
                old = head_record["installation"].snapshot
                new = installation.snapshot
                if (
                    (new.stream_id, new.run_epoch, new.schema_id) != (old.stream_id, old.run_epoch, old.schema_id)
                    or new.version < old.version
                    or (new.version == old.version and new != old)
                ):
                    raise DeltaCodecError("target contradicts the committed stream/head")
                if head_record["phase"] != "ACTIVE" and new != old:
                    raise DeltaCodecError("restore the committed target before installing a newer version")
                if new == old:
                    inherited = expected_head
            db.execute(
                "INSERT INTO installs VALUES(?, ?, ?, ?, 'PREPARED', ?, ?, NULL, ?)",
                (
                    installation.installation_id,
                    body,
                    expected_head,
                    snapshot_ref,
                    None if inherited is None else "COMMIT",
                    None if inherited is None else "inherited-" + installation.installation_id[:110],
                    inherited,
                ),
            )

    def issue(self, command: InstallCommand) -> None:
        body = canonical_json(asdict(command), self.limits.max_record_bytes)
        with self._transaction() as db:
            self._fence(db, command.installation)
            record = self._require(command.installation)
            existing = db.execute(
                "SELECT digest FROM commands WHERE install=? AND sequence=?",
                (command.installation.installation_id, str(command.sequence)),
            ).fetchone()
            if existing is not None:
                if existing[0] != command.digest:
                    raise DeltaCodecError("operation sequence reused with different request")
                return
            allowed = {
                "PREPARE": ("PREPARED",),
                "QUIESCE": ("PREPARED",),
                "LOAD": ("QUIESCED",),
                "VERIFY": ("RUNTIME_READY",),
                "ACK_COMMIT": ("COMMIT_DECIDED",),
                "ACTIVATE": ("ADMISSION_READY",),
            }
            if record["phase"] not in allowed.get(command.action, ()):
                raise DeltaCodecError("installation command is invalid in current phase")
            expected = {"PREPARE": ("CLOSED", "ACTIVE", "ABORTED"), "ACK_COMMIT": ("VERIFIED",)}.get(
                command.action, (record["phase"],)
            )
            if command.expected_phase not in expected:
                raise DeltaCodecError("worker expected phase differs from journal")
            if command.action == "ACK_COMMIT" and command.argument_digest != self.decision_certificate(
                command.installation.installation_id
            ):
                raise DeltaCodecError("COMMIT acknowledgement requires the durable decision certificate")
            if command.action == "ACTIVATE" and command.argument_digest != record["certificate"]:
                raise DeltaCodecError("activation requires the all-member admission certificate")
            count, pending = db.execute(
                "SELECT count(*), sum(result IS NULL) FROM commands WHERE install=?",
                (command.installation.installation_id,),
            ).fetchone()
            if count != command.sequence or pending:
                raise DeltaCodecError("command sequence gap or unresolved operation")
            if db.execute("SELECT count(*) FROM commands").fetchone()[0] >= self.limits.max_records * 8:
                raise DeltaCodecError("consumer command quota exhausted")
            db.execute(
                "INSERT INTO commands VALUES(?, ?, ?, ?, NULL)",
                (command.installation.installation_id, str(command.sequence), command.digest, body),
            )
            if command.action == "LOAD":
                db.execute("UPDATE installs SET phase='LOADING' WHERE id=?", (command.installation.installation_id,))

    def record_receipts(self, command: InstallCommand, receipts: tuple[RankReceipt, ...]) -> str:
        phase = {
            "PREPARE": "PREPARED",
            "QUIESCE": "QUIESCED",
            "LOAD": "RUNTIME_READY",
            "VERIFY": "VERIFIED",
            "ACK_COMMIT": "ADMISSION_READY",
            "ACTIVATE": "ACTIVE",
        }.get(command.action)
        certificate = certify_receipts(command, receipts, phase)
        result = canonical_json(
            {
                "phase": phase,
                "certificate": certificate,
                "receipts": [asdict(r) for r in sorted(receipts, key=lambda r: r.rank)],
            },
            self.limits.max_record_bytes,
        )
        with self._transaction() as db:
            self._fence(db, command.installation)
            record = self._require(command.installation)
            row = db.execute("SELECT result FROM commands WHERE digest=?", (command.digest,)).fetchone()
            if row is None:
                raise DeltaCodecError("receipt has no durable command intent")
            if row[0] is not None:
                if row[0] != result:
                    raise DeltaCodecError("operation receipt changed")
                return certificate
            if record["decision"] == "ABORT":
                raise DeltaCodecError("late receipt cannot revive an aborted installation")
            expected_phase = {
                "PREPARE": "PREPARED",
                "QUIESCE": "PREPARED",
                "LOAD": "LOADING",
                "VERIFY": "RUNTIME_READY",
                "ACK_COMMIT": "COMMIT_DECIDED",
                "ACTIVATE": "ADMISSION_READY",
            }[command.action]
            if record["phase"] != expected_phase:
                raise DeltaCodecError("late receipt is invalid in the current installation phase")
            if command.action in ("ACK_COMMIT", "ACTIVATE") and record["decision"] != "COMMIT":
                raise DeltaCodecError("activation requires a committed target")
            db.execute("UPDATE commands SET result=? WHERE digest=?", (result, command.digest))
            db.execute(
                "UPDATE installs SET phase=?, certificate=? WHERE id=?",
                (phase, certificate, command.installation.installation_id),
            )
        return certificate

    def decide(self, installation: Installation, decision: str, *, operation_id: str) -> str:
        require_identifier(operation_id, "decision operation ID")
        if decision not in ("COMMIT", "ABORT"):
            raise DeltaCodecError("invalid consumer decision")
        with self._transaction() as db:
            self._fence(db, installation)
            record = self._require(installation)
            if record["decision"] is not None:
                if (record["decision"], record["decision_id"]) != (decision, operation_id):
                    raise DeltaCodecError("installation already has a different immutable decision")
                if record["inherited_commit"] is not None and record["phase"] == "VERIFIED":
                    if self.head() != record["expected_head"]:
                        raise DeltaCodecError("head changed during committed target reinstall")
                    db.execute("UPDATE state SET head=?", (installation.installation_id,))
                    db.execute(
                        "UPDATE installs SET phase='COMMIT_DECIDED' WHERE id=?", (installation.installation_id,)
                    )
                return self.decision_certificate(installation.installation_id)
            if decision == "COMMIT":
                if record["phase"] != "VERIFIED" or self.head() != record["expected_head"]:
                    raise DeltaCodecError("COMMIT requires full verification and the expected head")
                db.execute("UPDATE state SET head=?", (installation.installation_id,))
            db.execute(
                "UPDATE installs SET decision=?, decision_id=?, phase=? WHERE id=?",
                (
                    decision,
                    operation_id,
                    "COMMIT_DECIDED" if decision == "COMMIT" else "ABORTED",
                    installation.installation_id,
                ),
            )
        return self.decision_certificate(installation.installation_id)

    def decision_certificate(self, installation_id: str) -> str:
        record = self.get(installation_id)
        if record is None or record["decision"] is None:
            raise DeltaCodecError("consumer decision is not known")
        return content_hash(
            canonical_json(
                {
                    "installation": record["installation"].digest,
                    "decision": record["decision"],
                    "operation": record["decision_id"],
                },
                4096,
            )
        )

    def retained_snapshots(self) -> tuple[str, ...]:
        return tuple(
            row[0]
            for row in self._connection().execute("SELECT DISTINCT snapshot_ref FROM installs ORDER BY snapshot_ref")
        )

    def recovery_records(self) -> tuple[dict, ...]:
        """Discover interrupted attempts without remembering their UUIDs."""
        rows = self._connection().execute("SELECT id FROM installs ORDER BY rowid").fetchall()
        require_uint(len(rows), "consumer recovery records", self.limits.max_records)
        return tuple(self.get(row[0]) for row in rows)

    def command_status(self, installation_id: str, sequence: int) -> dict | None:
        require_identifier(installation_id, "installation ID")
        require_uint(sequence, "operation sequence")
        row = (
            self._connection()
            .execute(
                "SELECT body, result FROM commands WHERE install=? AND sequence=?", (installation_id, str(sequence))
            )
            .fetchone()
        )
        if row is None:
            return None
        return {
            "command": parse_json(row[0], self.limits.max_record_bytes),
            "result": None if row[1] is None else parse_json(row[1], self.limits.max_record_bytes),
        }

    def bind_execution(
        self, execution_id: str, *, owner_fence: int, members: tuple[Member, ...], binding: dict
    ) -> None:
        """Persist backend process identities before allowing installation
        work."""
        require_identifier(execution_id, "execution ID")
        if not members or any(not isinstance(member, Member) for member in members):
            raise DeltaCodecError("execution binding requires current members")
        member_bytes = canonical_json([asdict(member) for member in members], self.limits.max_record_bytes)
        body = canonical_json(binding, self.limits.max_record_bytes)
        with self._transaction() as db:
            if owner_fence != db.execute("SELECT fence FROM state").fetchone()[0]:
                raise DeltaCodecError("stale execution binding owner")
            prior = db.execute(
                "SELECT members, binding, isolation_evidence FROM executions WHERE id=?", (execution_id,)
            ).fetchone()
            if prior is not None:
                if prior != (member_bytes, body, None):
                    raise DeltaCodecError("execution identity changed or was retired")
                return
            if db.execute("SELECT count(*) FROM executions").fetchone()[0] >= self.limits.max_records:
                raise DeltaCodecError("consumer execution quota exhausted")
            db.execute("INSERT INTO executions VALUES(?, ?, ?, NULL)", (execution_id, member_bytes, body))

    def execution_records(self) -> tuple[dict, ...]:
        return tuple(
            {
                "execution_id": row[0],
                "members": tuple(Member(**item) for item in parse_json(row[1], self.limits.max_record_bytes)),
                "binding": parse_json(row[2], self.limits.max_record_bytes),
                "isolation_evidence": row[3],
            }
            for row in self._connection().execute("SELECT * FROM executions ORDER BY rowid")
        )

    def record_execution_isolation(self, execution_id: str, *, owner_fence: int, evidence: str) -> None:
        from ..schema import require_digest

        require_digest(evidence, "execution isolation evidence")
        with self._transaction() as db:
            if owner_fence != db.execute("SELECT fence FROM state").fetchone()[0]:
                raise DeltaCodecError("stale execution isolation owner")
            row = db.execute("SELECT isolation_evidence FROM executions WHERE id=?", (execution_id,)).fetchone()
            if row is None or row[0] not in (None, evidence):
                raise DeltaCodecError("unknown execution or changed isolation evidence")
            db.execute("UPDATE executions SET isolation_evidence=? WHERE id=?", (evidence, execution_id))

    def retire_fenced(
        self, installation_id: str, *, owner_fence: int, fenced_members: tuple[Member, ...], isolation_evidence: str
    ) -> None:
        """Record externally established isolation and permit a fresh
        reinstall.

        The backend must physically stop all listed incarnations before this
        call. A database fence or a timeout is not isolation evidence. COMMIT
        and head survive; an undecided installation is durably aborted.
        """
        from ..schema import require_digest

        require_digest(isolation_evidence, "physical isolation evidence")
        with self._transaction() as db:
            if owner_fence != db.execute("SELECT fence FROM state").fetchone()[0]:
                raise DeltaCodecError("stale recovery owner")
            record = self.get(installation_id)
            if record is None or fenced_members != record["installation"].members:
                raise DeltaCodecError("recovery requires isolation of every old member")
            if record["phase"] == "RETIRED":
                if record["certificate"] != isolation_evidence:
                    raise DeltaCodecError("physical isolation evidence changed")
                return
            if record["decision"] is None:
                db.execute(
                    "UPDATE installs SET decision='ABORT', decision_id=? WHERE id=?",
                    ("recovery-" + installation_id[:110], installation_id),
                )
            db.execute(
                "UPDATE installs SET phase='RETIRED', certificate=? WHERE id=?", (isolation_evidence, installation_id)
            )
