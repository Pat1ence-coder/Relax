# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Installation identity and all-member evidence, without execution
backends."""

from dataclasses import asdict, dataclass

from .codec.format import content_hash
from .limits import DeltaCodecError, require_uint
from .manifest import SnapshotIdentity
from .schema import require_digest
from .serialization import canonical_json, require_identifier


PHASES = (
    "PREPARED",
    "QUIESCED",
    "LOADING",
    "RUNTIME_READY",
    "VERIFIED",
    "COMMIT_DECIDED",
    "ADMISSION_READY",
    "ACTIVE",
    "ABORTED",
)


@dataclass(frozen=True)
class Member:
    rank: int
    incarnation: str

    def __post_init__(self) -> None:
        require_uint(self.rank, "member rank", 4095)
        require_identifier(self.incarnation, "worker incarnation")


@dataclass(frozen=True)
class Installation:
    consumer_id: str
    owner_fence: int
    installation_id: str
    membership_epoch: int
    snapshot: SnapshotIdentity
    plan_id: str
    members: tuple[Member, ...]

    def __post_init__(self) -> None:
        require_identifier(self.consumer_id, "consumer ID")
        require_identifier(self.installation_id, "installation ID")
        require_uint(self.owner_fence, "consumer owner fence")
        require_uint(self.membership_epoch, "membership epoch")
        if self.owner_fence == 0 or not isinstance(self.snapshot, SnapshotIdentity):
            raise DeltaCodecError("installation requires a positive owner fence and snapshot")
        require_digest(self.plan_id, "load plan ID")
        if (
            not isinstance(self.members, tuple)
            or not 0 < len(self.members) <= 4096
            or any(not isinstance(m, Member) for m in self.members)
            or tuple(m.rank for m in self.members) != tuple(range(len(self.members)))
            or len({m.incarnation for m in self.members}) != len(self.members)
        ):
            raise DeltaCodecError("installation requires a complete unique membership")

    @property
    def digest(self) -> str:
        return content_hash(canonical_json(asdict(self), 1024 * 1024))


@dataclass(frozen=True)
class InstallCommand:
    installation: Installation
    sequence: int
    expected_phase: str
    action: str
    argument_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.installation, Installation):
            raise DeltaCodecError("command requires an installation")
        require_uint(self.sequence, "operation sequence")
        if self.expected_phase not in (*PHASES, "CLOSED"):
            raise DeltaCodecError("unknown expected installation phase")
        require_identifier(self.action, "installation action")
        require_digest(self.argument_digest, "command argument digest")

    @property
    def digest(self) -> str:
        return content_hash(canonical_json(asdict(self), 1024 * 1024))


@dataclass(frozen=True)
class RankReceipt:
    command_id: str
    rank: int
    incarnation: str
    phase: str
    evidence_id: str

    def __post_init__(self) -> None:
        require_digest(self.command_id, "receipt command ID")
        require_digest(self.evidence_id, "receipt evidence ID")
        require_uint(self.rank, "receipt rank", 4095)
        require_identifier(self.incarnation, "receipt incarnation")
        if self.phase not in PHASES:
            raise DeltaCodecError("unknown receipt phase")


def certify_receipts(command: InstallCommand, receipts: tuple[RankReceipt, ...], phase: str) -> str:
    """Require exactly one matching current-incarnation receipt from each
    rank."""
    if not isinstance(receipts, tuple) or len(receipts) != len(command.installation.members):
        raise DeltaCodecError("incomplete all-member receipts")
    expected = {member.rank: member.incarnation for member in command.installation.members}
    for receipt in receipts:
        if (
            not isinstance(receipt, RankReceipt)
            or receipt.command_id != command.digest
            or receipt.phase != phase
            or expected.pop(receipt.rank, None) != receipt.incarnation
        ):
            raise DeltaCodecError("duplicate, stale or mismatched rank receipt")
    if expected:
        raise DeltaCodecError("missing member receipt")
    return content_hash(
        canonical_json(
            {
                "command": command.digest,
                "phase": phase,
                "receipts": [asdict(r) for r in sorted(receipts, key=lambda r: r.rank)],
            },
            1024 * 1024,
        )
    )
