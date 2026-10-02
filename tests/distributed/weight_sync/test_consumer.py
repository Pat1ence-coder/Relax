# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""All-member evidence, durable decisions, lost responses and local
recovery."""

from dataclasses import replace

import pytest

from relax.distributed.weight_sync import DeltaCodecError, SnapshotIdentity
from relax.distributed.weight_sync.consumer import Installation, InstallCommand, Member, RankReceipt, certify_receipts
from relax.distributed.weight_sync.storage.consumer import ConsumerJournal, ConsumerUncertain


def install(fence=1, version=0, name="install-0", members=None):
    return Installation(
        "consumer",
        fence,
        name,
        1,
        SnapshotIdentity("test", "epoch", version, "a" * 64, "b" * 64),
        "c" * 64,
        members or (Member(0, "rank0"), Member(1, "rank1")),
    )


def receipts(command, phase):
    return tuple(
        RankReceipt(command.digest, m.rank, m.incarnation, phase, "d" * 64) for m in command.installation.members
    )


def advance(journal, value, stop="VERIFIED"):
    for seq, action, before, after in (
        (0, "PREPARE", "CLOSED", "PREPARED"),
        (1, "QUIESCE", "PREPARED", "QUIESCED"),
        (2, "LOAD", "QUIESCED", "RUNTIME_READY"),
        (3, "VERIFY", "RUNTIME_READY", "VERIFIED"),
    ):
        command = InstallCommand(value, seq, before, action, "e" * 64)
        journal.issue(command)
        journal.record_receipts(command, receipts(command, after))
        if after == stop:
            return command


@pytest.fixture
def journal_path(tmp_path):
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    return tmp_path / "control", snapshots


def open_journal(paths):
    return ConsumerJournal(paths[0], consumer_id="consumer", snapshot_root=paths[1])


@pytest.mark.parametrize("bad", ["missing", "duplicate", "incarnation", "command", "phase"])
def test_all_member_receipts_reject_stale_or_incomplete_evidence(bad):
    command = InstallCommand(install(), 1, "RUNTIME_READY", "VERIFY", "e" * 64)
    values = receipts(command, "VERIFIED")
    if bad == "missing":
        values = values[:-1]
    elif bad == "duplicate":
        values = (values[0], values[0])
    else:
        field = {"incarnation": "incarnation", "command": "command_id", "phase": "phase"}[bad]
        new = {"incarnation": "old-worker", "command": "f" * 64, "phase": "ACTIVE"}[bad]
        values = (replace(values[0], **{field: new}), values[1])
    with pytest.raises(DeltaCodecError):
        certify_receipts(command, values, "VERIFIED")


def test_durable_commit_head_and_activation_are_distinct(journal_path):
    with open_journal(journal_path) as journal:
        value = install(journal.acquire_owner())
        journal.begin(value, expected_head=None, snapshot_ref="generation-0")
        with pytest.raises(DeltaCodecError, match="verification"):
            journal.decide(value, "COMMIT", operation_id="decision")
        advance(journal, value)
        certificate = journal.decide(value, "COMMIT", operation_id="decision")
        assert journal.head() == value.installation_id
        assert journal.get(value.installation_id)["phase"] == "COMMIT_DECIDED"
        assert journal.decide(value, "COMMIT", operation_id="decision") == certificate
        with pytest.raises(DeltaCodecError, match="immutable"):
            journal.decide(value, "ABORT", operation_id="abort")
        for seq, action, before, after in (
            (4, "ACK_COMMIT", "VERIFIED", "ADMISSION_READY"),
            (5, "ACTIVATE", "ADMISSION_READY", "ACTIVE"),
        ):
            command = InstallCommand(value, seq, before, action, certificate)
            journal.issue(command)
            certificate = journal.record_receipts(command, receipts(command, after))
        assert journal.get(value.installation_id)["phase"] == "ACTIVE"
    with open_journal(journal_path) as reopened:
        assert reopened.head() == value.installation_id
        assert reopened.retained_snapshots() == ("generation-0",)


def test_lost_commit_reply_must_query_same_decision(journal_path, monkeypatch):
    with open_journal(journal_path) as journal:
        value = install(journal.acquire_owner())
        journal.begin(value, expected_head=None, snapshot_ref="generation-0")
        advance(journal, value)
        original = journal._commit

        def lost_reply():
            original()
            raise OSError("reply lost after durable COMMIT")

        monkeypatch.setattr(journal, "_commit", lost_reply)
        with pytest.raises(ConsumerUncertain):
            journal.decide(value, "COMMIT", operation_id="decision")
        assert journal.get(value.installation_id)["decision"] == "COMMIT"
        assert journal.head() == value.installation_id


def test_mid_load_abort_keeps_head_and_late_receipt_cannot_revive(journal_path):
    with open_journal(journal_path) as journal:
        value = install(journal.acquire_owner())
        journal.begin(value, expected_head=None, snapshot_ref="generation-0")
        advance(journal, value, "QUIESCED")
        load = InstallCommand(value, 2, "QUIESCED", "LOAD", "e" * 64)
        journal.issue(load)
        assert journal.get(value.installation_id)["phase"] == "LOADING"
        journal.decide(value, "ABORT", operation_id="abort")
        assert journal.head() is None
        with pytest.raises(DeltaCodecError, match="late"):
            journal.record_receipts(load, receipts(load, "RUNTIME_READY"))


def test_recovery_requires_all_old_members_and_preserves_committed_head(journal_path):
    with open_journal(journal_path) as journal:
        value = install(journal.acquire_owner())
        journal.begin(value, expected_head=None, snapshot_ref="generation-0")
        advance(journal, value)
        journal.decide(value, "COMMIT", operation_id="decision")
    with open_journal(journal_path) as journal:
        fence = journal.acquire_owner()
        fresh = install(fence, name="reinstall", members=(Member(0, "new0"), Member(1, "new1")))
        with pytest.raises(DeltaCodecError, match="previous"):
            journal.begin(fresh, expected_head=value.installation_id, snapshot_ref="generation-0")
        with pytest.raises(DeltaCodecError, match="every"):
            journal.retire_fenced(
                value.installation_id, owner_fence=fence, fenced_members=value.members[:1], isolation_evidence="f" * 64
            )
        journal.retire_fenced(
            value.installation_id, owner_fence=fence, fenced_members=value.members, isolation_evidence="f" * 64
        )
        assert journal.get(value.installation_id)["decision"] == "COMMIT"
        assert journal.head() == value.installation_id
        journal.begin(fresh, expected_head=value.installation_id, snapshot_ref="generation-0")
        with pytest.raises(DeltaCodecError, match="stale"):
            journal.decide(value, "COMMIT", operation_id="decision")


def test_command_identity_conflict_order_and_single_writer(journal_path):
    with open_journal(journal_path) as journal:
        with pytest.raises((BlockingIOError, OSError)):
            open_journal(journal_path)
        value = install(journal.acquire_owner())
        journal.begin(value, expected_head=None, snapshot_ref="generation-0")
        command = InstallCommand(value, 0, "CLOSED", "PREPARE", "e" * 64)
        journal.issue(command)
        journal.issue(command)
        with pytest.raises(DeltaCodecError, match="reused"):
            journal.issue(replace(command, argument_digest="f" * 64))
        with pytest.raises(DeltaCodecError, match="unresolved"):
            journal.issue(InstallCommand(value, 1, "PREPARED", "QUIESCE", "e" * 64))
        journal.record_receipts(command, receipts(command, "PREPARED"))


def test_retirement_cannot_be_undone_by_late_same_owner_ack(journal_path):
    with open_journal(journal_path) as journal:
        value = install(journal.acquire_owner())
        journal.begin(value, expected_head=None, snapshot_ref="generation-0")
        advance(journal, value)
        certificate = journal.decide(value, "COMMIT", operation_id="decision")
        command = InstallCommand(value, 4, "VERIFIED", "ACK_COMMIT", certificate)
        journal.issue(command)
        journal.retire_fenced(
            value.installation_id,
            owner_fence=value.owner_fence,
            fenced_members=value.members,
            isolation_evidence="f" * 64,
        )
        fresh = install(name="new-attempt", members=(Member(0, "new0"), Member(1, "new1")))
        journal.begin(fresh, expected_head=value.installation_id, snapshot_ref="generation-0")
        with pytest.raises(DeltaCodecError, match="late"):
            journal.record_receipts(command, receipts(command, "ADMISSION_READY"))
        assert journal.get(value.installation_id)["phase"] == "RETIRED"


def test_same_version_restart_inherits_commit_and_must_restore_before_advancing(journal_path):
    with open_journal(journal_path) as journal:
        value = install(journal.acquire_owner())
        journal.begin(value, expected_head=None, snapshot_ref="generation-0")
        advance(journal, value)
        journal.decide(value, "COMMIT", operation_id="decision")
        journal.retire_fenced(
            value.installation_id, owner_fence=1, fenced_members=value.members, isolation_evidence="f" * 64
        )
        with pytest.raises(DeltaCodecError, match="restore"):
            journal.begin(
                install(version=1, name="next-version"),
                expected_head=value.installation_id,
                snapshot_ref="generation-1",
            )
        fresh = install(name="reinstall")
        journal.begin(fresh, expected_head=value.installation_id, snapshot_ref="generation-0")
        row = journal.get(fresh.installation_id)
        assert row["inherited_commit"] == value.installation_id and row["decision"] == "COMMIT"
        with pytest.raises(DeltaCodecError, match="immutable"):
            journal.decide(fresh, "ABORT", operation_id="failed-attempt")
        assert journal.head() == value.installation_id
        advance(journal, fresh)
        journal.decide(fresh, "COMMIT", operation_id=row["decision_id"])
        assert journal.head() == fresh.installation_id
        assert journal.get(fresh.installation_id)["phase"] == "COMMIT_DECIDED"


def _crash_at(control, snapshots, phase):
    import os

    with ConsumerJournal(control, consumer_id="consumer", snapshot_root=snapshots) as journal:
        value = install(journal.acquire_owner())
        journal.begin(value, expected_head=None, snapshot_ref="generation-0")
        if phase != "PREPARED":
            advance(journal, value, "QUIESCED" if phase == "LOADING" else "VERIFIED")
        if phase == "LOADING":
            journal.issue(InstallCommand(value, 2, "QUIESCED", "LOAD", "e" * 64))
        elif phase == "COMMIT_DECIDED":
            journal.decide(value, "COMMIT", operation_id="decision")
        os._exit(17)


@pytest.mark.parametrize("phase", ["PREPARED", "LOADING", "VERIFIED", "COMMIT_DECIDED"])
def test_crash_recovery_discovers_unknown_installation_from_wal(journal_path, phase):
    import multiprocessing

    process = multiprocessing.get_context("spawn").Process(target=_crash_at, args=(*journal_path, phase))
    process.start()
    process.join(15)
    assert process.exitcode == 17
    with open_journal(journal_path) as journal:
        records = journal.recovery_records()
        assert len(records) == 1 and records[0]["phase"] == phase
        value = records[0]["installation"]
        assert journal.head() == (value.installation_id if phase == "COMMIT_DECIDED" else None)
        if phase == "LOADING":
            command = journal.command_status(value.installation_id, 2)
            assert command["command"]["action"] == "LOAD" and command["result"] is None


def test_aborted_execution_must_be_physically_retired_before_replacement(journal_path):
    with open_journal(journal_path) as journal:
        value = install(journal.acquire_owner())
        journal.begin(value, expected_head=None, snapshot_ref="generation-0")
        journal.decide(value, "ABORT", operation_id="abort")
        with pytest.raises(DeltaCodecError, match="physically"):
            journal.begin(install(name="unsafe-replacement"), expected_head=None, snapshot_ref="generation-0")
        journal.retire_fenced(
            value.installation_id, owner_fence=1, fenced_members=value.members, isolation_evidence="f" * 64
        )
        journal.begin(install(name="replacement"), expected_head=None, snapshot_ref="generation-0")


def test_execution_bindings_survive_reopen_and_cannot_change(journal_path):
    with open_journal(journal_path) as journal:
        fence = journal.acquire_owner()
        journal.bind_execution("engine", owner_fence=fence, members=install().members, binding={"processes": []})
        journal.bind_execution("engine", owner_fence=fence, members=install().members, binding={"processes": []})
        with pytest.raises(DeltaCodecError, match="changed"):
            journal.bind_execution("engine", owner_fence=fence, members=install().members, binding={"processes": [1]})
    with open_journal(journal_path) as journal:
        (row,) = journal.execution_records()
        assert row["members"] == install().members and row["binding"] == {"processes": []}
        fence = journal.acquire_owner()
        journal.record_execution_isolation("engine", owner_fence=fence, evidence="e" * 64)
        with pytest.raises(DeltaCodecError, match="changed"):
            journal.record_execution_isolation("engine", owner_fence=fence, evidence="f" * 64)
