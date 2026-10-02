# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Independent single-engine consumer installation and recovery
coordination."""

import asyncio
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from relax.distributed.weight_sync import DeltaCodecError, SnapshotIdentity
from relax.distributed.weight_sync.codec.format import content_hash
from relax.distributed.weight_sync.consumer import Installation, InstallCommand
from relax.distributed.weight_sync.serialization import canonical_json
from relax.distributed.weight_sync.storage import open_snapshot
from relax.distributed.weight_sync.storage.consumer import ConsumerJournal, ConsumerUncertain

from .isolation import isolate_processes, isolation_digest


class ConsumerInstaller:
    """The caller owns one local WAL, engine and retained canonical generation
    set.

    Errors close admission. To recover, physically stop all old engine
    children, record isolation with retire_fenced, then create a new engine and
    reinstall the committed head (or the previous head after an authoritative
    ABORT). No timeout, task cancellation or SQLite fence is treated as GPU
    isolation.
    """

    def __init__(self, journal: ConsumerJournal, engine: Any, *, owner_fence: int):
        self.journal, self.engine, self.owner_fence = journal, engine, owner_fence
        self._lock = asyncio.Lock()
        self.last_installation = None
        self.evidence = []
        journal.bind_execution(
            engine.delta_execution_id,
            owner_fence=owner_fence,
            members=engine.tokenizer_manager.delta_members,
            binding=engine.delta_process_binding,
        )

    async def stop_and_fence(self) -> str:
        """Stop only this engine's children, then persist their physical
        isolation.

        Cancellation/timeout never calls this automatically. The caller can use
        a fresh engine and reinstall journal.head() after this method succeeds.
        """
        async with self._lock:
            manager = self.engine.tokenizer_manager
            manager.delta_ticket = None
            manager.delta_poisoned = True
            self.engine.shutdown()
            isolation = getattr(self.engine, "delta_isolation", ())
            if not isolation or any(item.get("exitcode") is None for item in isolation):
                raise DeltaCodecError("engine did not prove termination of its owned processes")
            evidence = content_hash(
                canonical_json(
                    {"members": [asdict(m) for m in manager.delta_members], "processes": isolation}, 64 * 1024
                )
            )
            self.journal.record_execution_isolation(
                self.engine.delta_execution_id, owner_fence=self.owner_fence, evidence=evidence
            )
            for record in self.journal.recovery_records():
                if record["installation"].members == manager.delta_members and record["phase"] != "RETIRED":
                    self.journal.retire_fenced(
                        record["installation"].installation_id,
                        owner_fence=self.owner_fence,
                        fenced_members=manager.delta_members,
                        isolation_evidence=evidence,
                    )
            return evidence

    async def install(self, identity: SnapshotIdentity, snapshot_ref: str, *, timeout: float = 300) -> Installation:
        async with self._lock:
            config, manager = self.engine.delta_runtime_config, self.engine.tokenizer_manager
            if manager.delta_poisoned or manager.delta_pending is not None:
                raise DeltaCodecError("physical engine recovery is required before another installation")
            # Verify before closing the old admission gate. Retain this reader
            # and the generation reference throughout the transaction.
            snapshot, reader = open_snapshot(Path(config.snapshot_root) / snapshot_ref, expected_identity=identity)
            previous_phase = "ACTIVE" if manager.delta_ticket is not None else "CLOSED"
            installation = Installation(
                self.journal.consumer_id,
                self.owner_fence,
                uuid.uuid4().hex,
                self.owner_fence,
                snapshot.identity,
                self.engine.delta_plan_id,
                manager.delta_members,
            )
            self.last_installation = installation
            try:
                self.journal.begin(installation, expected_head=self.journal.head(), snapshot_ref=snapshot_ref)
                sequence = 0

                async def operation(action, phase, argument, argument_digest=None):
                    nonlocal sequence
                    command = InstallCommand(
                        installation,
                        sequence,
                        phase,
                        action,
                        argument_digest or content_hash(canonical_json(argument, 64 * 1024)),
                    )
                    self.journal.issue(command)
                    started = time.monotonic()
                    receipts, records = await manager.delta_command(command, argument, timeout)
                    certificate = self.journal.record_receipts(command, receipts)
                    self.evidence.append(
                        {
                            "command": asdict(command),
                            "elapsed_seconds": time.monotonic() - started,
                            "records": records,
                            "certificate": certificate,
                        }
                    )
                    sequence += 1
                    return command, receipts, certificate

                await operation("PREPARE", previous_phase, {"snapshot_ref": snapshot_ref})
                await operation("QUIESCE", "PREPARED", {})
                # All previously dispatched requests have been aborted at the
                # scheduler. Wait for their tokenizer/stream bookkeeping too.
                await asyncio.wait_for(manager.delta_drain_requests(), timeout)
                await operation("LOAD", "QUIESCED", {})
                await operation("VERIFY", "RUNTIME_READY", {})
                record = self.journal.get(installation.installation_id)
                decision_op = record["decision_id"] or "commit-" + installation.installation_id
                try:
                    decision = self.journal.decide(installation, "COMMIT", operation_id=decision_op)
                except ConsumerUncertain:
                    record = self.journal.get(installation.installation_id)
                    if record is None or record["decision"] != "COMMIT" or record["decision_id"] != decision_op:
                        raise
                    decision = self.journal.decision_certificate(installation.installation_id)
                ack, receipts, certificate = await operation(
                    "ACK_COMMIT", "VERIFIED", {"decision_operation": decision_op}, decision
                )
                activation, _, _ = await operation(
                    "ACTIVATE",
                    "ADMISSION_READY",
                    {"ack_command": asdict(ack), "receipts": [asdict(r) for r in receipts]},
                    certificate,
                )
                # Worker activation alone does not admit users. The tokenizer
                # opens only after this current-incarnation all-rank result.
                manager.delta_ticket = activation.digest
                return installation
            except BaseException:
                manager.delta_ticket = None
                record = self.journal.get(installation.installation_id)
                if record is not None and record["decision"] is None:
                    self.journal.decide(installation, "ABORT", operation_id="abort-" + installation.installation_id)
                raise
            finally:
                reader.close()


async def recover_orphaned_engines(journal: ConsumerJournal, *, owner_fence: int) -> None:
    """Recover persisted local process bindings after a coordinator crash.

    Admission remains closed. After isolation, create a fresh engine and fully
    reinstall the committed head. Boot/namespace mismatch fails closed.
    """
    for execution in journal.execution_records():
        evidence = execution["isolation_evidence"]
        if evidence is None:
            processes = execution["binding"].get("processes")
            if not isinstance(processes, list):
                raise DeltaCodecError("recovery has no complete engine process binding")
            proof = await asyncio.to_thread(isolate_processes, processes)
            evidence = isolation_digest(execution["execution_id"], proof)
            journal.record_execution_isolation(execution["execution_id"], owner_fence=owner_fence, evidence=evidence)
        for record in journal.recovery_records():
            if record["installation"].members == execution["members"] and record["phase"] != "RETIRED":
                journal.retire_fenced(
                    record["installation"].installation_id,
                    owner_fence=owner_fence,
                    fenced_members=execution["members"],
                    isolation_evidence=evidence,
                )
