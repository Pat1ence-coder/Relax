# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Explicit POSIX deployment admission and path-free diagnostic reports.

Inspection is read-only. Volume evidence is trusted operator configuration, not
a substitute for cross-node or infrastructure failure tests.
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..codec.format import content_hash
from ..limits import DeltaCodecError, SnapshotLimits, require_uint
from ..schema import require_digest
from ..serialization import canonical_json, exact_fields, parse_json, require_identifier
from .contracts import StorageLimits
from .files import ARTIFACT_DIRECTORIES, open_directory, read_file
from .placement import (
    DirectoryGuard,
    MountRecord,
    StorageDeploymentError,
    mount_for,
    read_mounts,
    require_mount_separation,
    require_separate,
)


if TYPE_CHECKING:
    from .artifacts import PosixArtifactStore


PROFILE = "posix_immutable_v1"
_REPORT_LIMIT = 128 * 1024
_NAMESPACE_LIMIT = 16 * 1024
_LOCAL_CONTROL = {"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "f2fs", "overlay"}


@dataclass(frozen=True)
class NamespaceBinding:
    namespace_id: str
    stream_id: str
    run_epoch: str

    def __post_init__(self) -> None:
        for name in ("namespace_id", "stream_id", "run_epoch"):
            require_identifier(getattr(self, name), name)

    def to_bytes(self) -> bytes:
        return canonical_json({"format_version": 1, **asdict(self)}, _NAMESPACE_LIMIT)

    @classmethod
    def from_bytes(cls, data: bytes) -> "NamespaceBinding":
        value = exact_fields(
            parse_json(data, _NAMESPACE_LIMIT), {"format_version", "namespace_id", "stream_id", "run_epoch"}
        )
        version = value.pop("format_version")
        if type(version) is not int or version != 1:
            raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "namespace_format")
        return cls(**value)


@dataclass(frozen=True)
class VolumeSpec:
    """Expected mount identity and operator-attested volume allocation.

    The same physical capacity must use the same volume_id across roots,
    including aliases not detectable from one process's mount table.
    """

    volume_id: str
    mount_point: Path = field(repr=False)
    filesystem_type: str
    source_digest: str
    root_digest: str
    local: bool
    persistent: bool
    fault_domains: tuple[str, ...]
    evidence_digest: str
    capacity_bytes: int
    reserved_bytes: int

    def __post_init__(self) -> None:
        require_identifier(self.volume_id, "volume_id")
        require_identifier(self.filesystem_type, "filesystem_type")
        if not isinstance(self.mount_point, Path) or not self.mount_point.is_absolute():
            raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "absolute_mount_required")
        for name in ("source_digest", "root_digest", "evidence_digest"):
            require_digest(getattr(self, name), name)
        if type(self.local) is not bool or type(self.persistent) is not bool:
            raise DeltaCodecError("volume locality and persistence must be boolean")
        if not isinstance(self.fault_domains, tuple) or not self.fault_domains or len(self.fault_domains) > 16:
            raise DeltaCodecError("bounded fault-domain evidence is required")
        for domain in self.fault_domains:
            require_identifier(domain, "fault_domain")
        for name in ("capacity_bytes", "reserved_bytes"):
            require_uint(getattr(self, name), name)
        if not 0 < self.reserved_bytes <= self.capacity_bytes:
            raise DeltaCodecError("volume reservation must fit its declared capacity")


@dataclass(frozen=True)
class PosixAccessPolicy:
    """Private owner access or explicit read-only access for a shared group."""

    shared_group_id: int | None = None

    def __post_init__(self) -> None:
        if self.shared_group_id is not None:
            require_uint(self.shared_group_id, "shared_group_id", (1 << 32) - 2)

    @property
    def directory_mode(self) -> int:
        return 0o700 if self.shared_group_id is None else 0o750

    @property
    def object_mode(self) -> int:
        return 0o400 if self.shared_group_id is None else 0o440


@dataclass(frozen=True)
class PosixDeployment:
    artifact_root: Path = field(repr=False)
    namespace: NamespaceBinding
    artifact_volume: VolumeSpec
    access: Literal["reader", "publisher"]
    required_fault_domain: str
    expected_uid: int
    expected_gid: int
    control_root: Path | None = field(default=None, repr=False)
    control_volume: VolumeSpec | None = None
    staging_root: Path | None = field(default=None, repr=False)
    staging_volume: VolumeSpec | None = None
    access_policy: PosixAccessPolicy = PosixAccessPolicy()

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, NamespaceBinding) or not isinstance(self.artifact_volume, VolumeSpec):
            raise DeltaCodecError("typed namespace and artifact volume configuration are required")
        if not isinstance(self.access_policy, PosixAccessPolicy):
            raise DeltaCodecError("typed POSIX access policy is required")
        if self.access not in ("reader", "publisher"):
            raise DeltaCodecError("unknown storage role")
        require_identifier(self.required_fault_domain, "required_fault_domain")
        for name in ("expected_uid", "expected_gid"):
            require_uint(getattr(self, name), name, (1 << 32) - 2)
        for name in ("artifact_root", "control_root", "staging_root"):
            path = getattr(self, name)
            if path is not None and (not isinstance(path, Path) or not path.is_absolute()):
                raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "absolute_root_required")
        if self.artifact_root is None:
            raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "artifact_root_required")
        for role in ("control", "staging"):
            volume = getattr(self, role + "_volume")
            if volume is not None and not isinstance(volume, VolumeSpec):
                raise DeltaCodecError("typed volume configuration is required")
            if (getattr(self, role + "_root") is None) != (getattr(self, role + "_volume") is None):
                raise DeltaCodecError("each configured root requires volume evidence")
        if self.access == "publisher" and self.control_root is None:
            raise StorageDeploymentError("STORAGE_CAPABILITY_UNSUPPORTED", "local_control_required")

    def digest(self) -> str:
        value = asdict(self)
        for name in ("artifact_root", "control_root", "staging_root"):
            if value[name] is not None:
                value[name] = str(value[name])
        for name in ("artifact_volume", "control_volume", "staging_volume"):
            if value[name] is not None:
                value[name]["mount_point"] = str(value[name]["mount_point"])
        return content_hash(canonical_json(value, _REPORT_LIMIT))


@dataclass(frozen=True)
class DeploymentCheck:
    name: str
    status: Literal["PASSED", "FAILED", "NOT_RUN"]
    code: str


@dataclass(frozen=True)
class DeploymentReport:
    config_digest: str
    namespace_id: str
    access: str
    checks: tuple[DeploymentCheck, ...]
    evidence_digests: tuple[str, ...]

    @property
    def admitted(self) -> bool:
        return bool(self.checks) and all(check.status == "PASSED" for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "profile": PROFILE,
            **asdict(self),
            "configuration_admitted": self.admitted,
            "evidence_source": "trusted_operator_attestation",
            "cross_node_tests": "NOT_RUN",
            "infrastructure_fault_tests": "NOT_RUN",
            "compatibility_verified": False,
        }

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_dict(), _REPORT_LIMIT)

    def require_admitted(self) -> None:
        for check in self.checks:
            if check.status != "PASSED":
                raise StorageDeploymentError(check.code, check.name)
        if not self.checks:
            raise StorageDeploymentError("STORAGE_CAPABILITY_UNSUPPORTED", "empty_deployment_report")


@dataclass(frozen=True)
class DeploymentAdmission:
    config: PosixDeployment = field(repr=False)
    report: DeploymentReport
    artifact_guard: DirectoryGuard = field(repr=False)
    control_guard: DirectoryGuard | None = field(repr=False)
    staging_guard: DirectoryGuard | None = field(repr=False)
    initializing: bool = False

    def validate_artifact(self, descriptor: int) -> None:
        self.artifact_guard.validate_opened(descriptor)
        _check_namespace(descriptor, self.config.namespace, initializing=self.initializing)
        _check_artifact_directories(descriptor, self.artifact_guard, allow_missing=self.config.access == "publisher")

    def prepare_control(self, location: Path) -> DirectoryGuard:
        if self.control_guard is None or location.resolve() != self.control_guard.path.resolve():
            raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "unapproved_control_location")
        require_separate(self.config.artifact_root, location)
        return self.control_guard


def _check_artifact_directories(descriptor: int, guard: DirectoryGuard, *, allow_missing: bool) -> None:
    for name in ARTIFACT_DIRECTORIES:
        try:
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
        except FileNotFoundError:
            if allow_missing:
                continue
            raise
        try:
            guard.validate_child_mount(child)
        finally:
            os.close(child)


def _check_namespace(descriptor: int, expected: NamespaceBinding, *, initializing: bool = False) -> None:
    try:
        actual = NamespaceBinding.from_bytes(read_file(descriptor, "namespace.json", _NAMESPACE_LIMIT))
    except FileNotFoundError:
        if not initializing:
            raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "namespace_missing") from None
    else:
        if actual != expected:
            raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "namespace_binding")
    try:
        authority = parse_json(read_file(descriptor, "authority.json", _NAMESPACE_LIMIT), _NAMESPACE_LIMIT)
    except FileNotFoundError:
        return
    authority = exact_fields(authority, {"format_version", "authority_id", "stream_id", "run_epoch"})
    if type(authority["format_version"]) is not int or authority["format_version"] != 1:
        raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "authority_format")
    require_identifier(authority["authority_id"], "authority_id")
    if (authority["stream_id"], authority["run_epoch"]) != (expected.stream_id, expected.run_epoch):
        raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "authority_binding")


def _volume_guard(
    role: str, path: Path, volume: VolumeSpec, config: PosixDeployment, mounts: tuple[MountRecord, ...]
) -> tuple[DirectoryGuard, MountRecord]:
    actual = mount_for(path, mounts)
    if (actual.mount_point, actual.filesystem_type, actual.source_digest, actual.root_digest) != (
        volume.mount_point,
        volume.filesystem_type,
        volume.source_digest,
        volume.root_digest,
    ):
        raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", role + "_mount_identity")
    if not volume.persistent or config.required_fault_domain not in volume.fault_domains:
        raise StorageDeploymentError("STORAGE_CAPABILITY_UNSUPPORTED", role + "_fault_domain_evidence")
    if role == "control" and (not volume.local or actual.filesystem_type not in _LOCAL_CONTROL):
        raise StorageDeploymentError("STORAGE_CAPABILITY_UNSUPPORTED", "control_requires_local_persistent_volume")
    writable = role != "artifact" or config.access == "publisher"
    if writable and actual.readonly:
        raise StorageDeploymentError("STORAGE_PERMISSION_DENIED", role + "_readonly_mount")
    required = os.R_OK | os.X_OK | (os.W_OK if writable else 0)
    if not os.access(path, required, effective_ids=True):
        raise StorageDeploymentError("STORAGE_PERMISSION_DENIED", role + "_access")
    guard = DirectoryGuard.capture(path, mount_id=actual.mount_id)
    descriptor = open_directory(path)
    try:
        guard.validate_opened(descriptor)
        info = os.fstat(descriptor)
        if role in ("control", "staging") and (info.st_uid != os.geteuid() or info.st_mode & 0o077):
            raise StorageDeploymentError("STORAGE_PERMISSION_DENIED", role + "_private_directory_required")
        if role == "artifact":
            group = config.access_policy.shared_group_id
            if info.st_mode & 0o022:
                raise StorageDeploymentError("STORAGE_PERMISSION_DENIED", "artifact_untrusted_write_access")
            if config.access == "publisher" and info.st_uid != os.geteuid():
                raise StorageDeploymentError("STORAGE_PERMISSION_DENIED", "artifact_owner_required")
            if group is not None and (info.st_gid != group or info.st_mode & 0o050 != 0o050):
                raise StorageDeploymentError("STORAGE_PERMISSION_DENIED", "artifact_group_read_policy")
    finally:
        os.close(descriptor)
    return guard, actual


def _inspect(
    config: PosixDeployment, limits: SnapshotLimits, storage_limits: StorageLimits, *, initializing: bool = False
) -> tuple[DeploymentReport, dict[str, DirectoryGuard]]:
    checks: list[DeploymentCheck] = []
    guards: dict[str, DirectoryGuard] = {}
    roles = [("artifact", config.artifact_root, config.artifact_volume)]
    for role in ("control", "staging"):
        path, volume = getattr(config, role + "_root"), getattr(config, role + "_volume")
        if path is not None and volume is not None:
            roles.append((role, path, volume))
    try:
        if (os.geteuid(), os.getegid()) != (config.expected_uid, config.expected_gid):
            raise StorageDeploymentError("STORAGE_PERMISSION_DENIED", "process_identity")
        for index, (_, left, _) in enumerate(roles):
            for _, right, _ in roles[index + 1 :]:
                require_separate(left, right)
        if initializing and config.access != "publisher":
            raise StorageDeploymentError("STORAGE_PERMISSION_DENIED", "namespace_initialization_requires_publisher")
        if len(config.namespace.to_bytes()) > storage_limits.max_record_bytes:
            raise StorageDeploymentError("RESOURCE_EXHAUSTED", "namespace_metadata_budget")
        mounts = read_mounts()
        volumes: dict[str, tuple[int, int, str]] = {}
        observed: dict[tuple[str, str, str], str] = {}
        inspected: list[tuple[Path, MountRecord, DirectoryGuard]] = []
        for role, path, volume in roles:
            guard, actual = _volume_guard(role, path, volume, config, mounts)
            for previous_path, previous_mount, previous_guard in inspected:
                if guard.identity == previous_guard.identity:
                    raise StorageDeploymentError("STORAGE_NAMESPACE_MISMATCH", "physical_directories_overlap")
                require_mount_separation(path, actual, previous_path, previous_mount)
            inspected.append((path, actual, guard))
            guards[role] = guard
            physical = (actual.device, actual.filesystem_type, actual.source_digest)
            previous_id = observed.setdefault(physical, volume.volume_id)
            if previous_id != volume.volume_id:
                raise StorageDeploymentError("RESOURCE_EXHAUSTED", "shared_volume_requires_shared_budget")
            total, capacity, evidence = volumes.get(
                volume.volume_id, (0, volume.capacity_bytes, volume.evidence_digest)
            )
            if (capacity, evidence) != (volume.capacity_bytes, volume.evidence_digest):
                raise StorageDeploymentError("RESOURCE_EXHAUSTED", "inconsistent_volume_budget")
            total += volume.reserved_bytes
            if total > capacity:
                raise StorageDeploymentError("RESOURCE_EXHAUSTED", "shared_volume_budget")
            volumes[volume.volume_id] = (total, capacity, evidence)
            checks.append(DeploymentCheck(role + "_placement", "PASSED", "OK"))
        temporary = max(
            limits.codec.max_chunk_bytes,
            limits.max_index_page_bytes,
            limits.max_manifest_bytes,
            storage_limits.max_archive_bytes,
            storage_limits.max_record_bytes,
        )
        if config.artifact_volume.reserved_bytes < storage_limits.max_bytes + temporary:
            raise StorageDeploymentError("RESOURCE_EXHAUSTED", "artifact_and_temporary_reservation")
        if (
            config.control_volume is not None
            and config.control_volume.reserved_bytes < 3 * storage_limits.max_database_bytes
        ):
            raise StorageDeploymentError("RESOURCE_EXHAUSTED", "control_database_and_wal_reservation")
        descriptor = open_directory(config.artifact_root)
        try:
            guards["artifact"].validate_opened(descriptor)
            _check_namespace(descriptor, config.namespace, initializing=initializing)
            _check_artifact_directories(descriptor, guards["artifact"], allow_missing=config.access == "publisher")
        finally:
            os.close(descriptor)
        checks.append(DeploymentCheck("namespace", "PASSED", "INITIALIZATION_REQUESTED" if initializing else "OK"))
        checks.append(DeploymentCheck("volume_allocations", "PASSED", "OPERATOR_ATTESTED"))
    except StorageDeploymentError as error:
        checks.append(DeploymentCheck(error.check, "FAILED", error.code))
    except PermissionError:
        checks.append(DeploymentCheck("filesystem_access", "FAILED", "STORAGE_PERMISSION_DENIED"))
    except FileNotFoundError:
        checks.append(DeploymentCheck("configured_directory_missing", "FAILED", "STORAGE_NAMESPACE_MISMATCH"))
    except OSError:
        checks.append(DeploymentCheck("filesystem_unavailable", "FAILED", "STORAGE_UNAVAILABLE"))
    except (ValueError, DeltaCodecError):
        checks.append(DeploymentCheck("deployment_inspection", "FAILED", "STORAGE_CAPABILITY_UNSUPPORTED"))
    report = DeploymentReport(
        config.digest(),
        config.namespace.namespace_id,
        config.access,
        tuple(checks),
        tuple(sorted({volume.evidence_digest for _, _, volume in roles})),
    )
    return report, guards


def inspect_deployment(
    config: PosixDeployment,
    *,
    limits: SnapshotLimits = SnapshotLimits(),
    storage_limits: StorageLimits = StorageLimits(),
) -> DeploymentReport:
    """Read-only admission checks; do not write probes or contact a
    producer."""
    return _inspect(config, limits, storage_limits)[0]


def _admit(
    config: PosixDeployment, limits: SnapshotLimits, storage_limits: StorageLimits, *, initializing: bool = False
) -> DeploymentAdmission:
    report, guards = _inspect(config, limits, storage_limits, initializing=initializing)
    report.require_admitted()
    return DeploymentAdmission(
        config, report, guards["artifact"], guards.get("control"), guards.get("staging"), initializing
    )


def open_deployed_store(
    config: PosixDeployment,
    *,
    limits: SnapshotLimits = SnapshotLimits(),
    storage_limits: StorageLimits = StorageLimits(),
) -> "PosixArtifactStore":
    """Open an existing namespace only after deployment admission."""
    from .artifacts import PosixArtifactStore

    admission = _admit(config, limits, storage_limits)
    return PosixArtifactStore(
        config.artifact_root,
        writable=config.access == "publisher",
        limits=limits,
        storage_limits=storage_limits,
        _admission=admission,
    )


def initialize_namespace(
    config: PosixDeployment,
    *,
    limits: SnapshotLimits = SnapshotLimits(),
    storage_limits: StorageLimits = StorageLimits(),
) -> DeploymentReport:
    """Explicit publisher-only enrollment; never mount or chmod parent
    roots."""
    from .artifacts import PosixArtifactStore

    admission = _admit(config, limits, storage_limits, initializing=True)
    with PosixArtifactStore(
        config.artifact_root, writable=True, limits=limits, storage_limits=storage_limits, _admission=admission
    ):
        pass
    return admission.report
