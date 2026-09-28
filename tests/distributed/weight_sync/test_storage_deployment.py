# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Deployment fixtures; these do not certify any real shared mount."""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from relax.distributed.weight_sync import CodecLimits, DeltaCodecError, SnapshotLimits
from relax.distributed.weight_sync.codec.format import content_hash
from relax.distributed.weight_sync.storage import (
    DeploymentReport,
    MountRecord,
    NamespaceBinding,
    PosixAccessPolicy,
    PosixDeployment,
    ProducerCatalog,
    StorageDeploymentError,
    StorageLimits,
    VolumeSpec,
    initialize_namespace,
    inspect_deployment,
    open_deployed_store,
    parse_mountinfo,
)
from relax.distributed.weight_sync.storage import artifacts as artifact_io
from relax.distributed.weight_sync.storage import deployment as deployment_io
from relax.distributed.weight_sync.storage import files as file_io
from relax.distributed.weight_sync.storage import layout as layout_io
from relax.distributed.weight_sync.storage import placement as placement_io


MIB = 1024 * 1024
LIMITS = SnapshotLimits(codec=CodecLimits(max_chunk_bytes=64))
STORAGE = StorageLimits(max_bytes=MIB, max_objects=256, max_records=16, max_database_bytes=MIB)


@pytest.fixture
def configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PosixDeployment:
    artifact, control, staging = (tmp_path / name for name in ("artifacts", "control", "staging"))
    for directory in (artifact, control, staging):
        directory.mkdir(mode=0o700)
    mount = MountRecord(71, "0:71", tmp_path, "ext4", content_hash(b"fixture-device"), content_hash(b"/"), False)
    monkeypatch.setattr(deployment_io, "read_mounts", lambda: (mount,))
    monkeypatch.setattr(placement_io, "descriptor_mount_id", lambda descriptor: 71)
    volume = VolumeSpec(
        "fixture-volume",
        tmp_path,
        "ext4",
        mount.source_digest,
        mount.root_digest,
        True,
        True,
        ("process_crash",),
        content_hash(b"fixture-attestation"),
        128 * MIB,
        20 * MIB,
    )
    return PosixDeployment(
        artifact,
        NamespaceBinding("fixture", "test", "epoch"),
        volume,
        "publisher",
        "process_crash",
        os.geteuid(),
        os.getegid(),
        control,
        replace(volume, reserved_bytes=4 * MIB),
        staging,
        replace(volume, reserved_bytes=8 * MIB),
    )


def inspect(config: PosixDeployment) -> DeploymentReport:
    return inspect_deployment(config, limits=LIMITS, storage_limits=STORAGE)


def initialize(config: PosixDeployment) -> None:
    initialize_namespace(config, limits=LIMITS, storage_limits=STORAGE)


def physical_root(config: PosixDeployment) -> Path:
    descriptor = layout_io.NamespaceDescriptor.from_bytes((config.artifact_root / "namespace.json").read_bytes())
    return config.artifact_root if descriptor.layout is None else config.artifact_root / descriptor.layout


def test_mount_table_parser_bounds_input_unescapes_paths_and_discards_secrets() -> None:
    data = b"71 1 0:71 / /mounted\\040volume rw shared:1 - fuse.fixture private-source rw,private-option\n"
    records = parse_mountinfo(data)
    assert records[0].mount_point == Path("/mounted volume")
    assert records[0].source_digest == content_hash(b"private-source")
    assert "private-source" not in repr(records) and "private-option" not in repr(records)
    assert "mounted volume" not in repr(records)
    with pytest.raises(StorageDeploymentError, match="budget"):
        parse_mountinfo(data, maximum=8)
    with pytest.raises(StorageDeploymentError, match="format"):
        parse_mountinfo(b"invalid private-source\n")


def test_namespace_enrollment_is_explicit_and_configuration_is_not_certification(configured: PosixDeployment) -> None:
    before = inspect(configured)
    assert not before.admitted
    assert not (configured.artifact_root / "namespace.json").exists()
    initialize(configured)
    initialize(configured)
    report = inspect(configured)
    assert report.admitted
    public = report.to_dict()
    assert public["compatibility_verified"] is False
    assert public["cross_node_tests"] == public["infrastructure_fault_tests"] == "NOT_RUN"
    assert str(configured.artifact_root).encode() not in report.to_bytes()
    assert str(configured.control_root).encode() not in report.to_bytes()
    assert b"fixture-device" not in report.to_bytes()


def test_configured_writer_blocks_upload_until_authority_recovery(configured: PosixDeployment) -> None:
    initialize(configured)
    with open_deployed_store(configured, limits=LIMITS, storage_limits=STORAGE) as store:
        payload = b"raw-data"
        key = "chunks/" + content_hash(payload)
        with pytest.raises(DeltaCodecError, match="recovery"):
            store.put_immutable(key, payload, expected_hash=content_hash(payload))
        with ProducerCatalog(configured.control_root, store, stream_id="test", run_epoch="epoch"):
            receipt = store.put_immutable(key, payload, expected_hash=content_hash(payload))
            assert receipt.namespace_id == "fixture" and receipt.fault_domain == "process_crash"
            assert store.capabilities().deployment_checked


def test_reader_open_performs_no_shared_writes(configured: PosixDeployment, monkeypatch: pytest.MonkeyPatch) -> None:
    initialize(configured)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("read-only deployment attempted a write")

    monkeypatch.setattr(artifact_io, "exclusive_lock", forbidden)
    monkeypatch.setattr(artifact_io, "install_file", forbidden)
    reader = replace(configured, access="reader", control_root=None, control_volume=None)
    with open_deployed_store(reader, limits=LIMITS, storage_limits=STORAGE) as store:
        assert store.capabilities().access == "reader"
        assert NamespaceBinding.from_bytes(store.read_object("namespace.json", 16384)) == configured.namespace
        with pytest.raises(DeltaCodecError, match="read-only"):
            store.put_object("chunks/" + content_hash(b"x"), b"x")
    with pytest.raises(StorageDeploymentError, match="initialization_requires_publisher"):
        initialize(reader)


@pytest.mark.parametrize("change", ["source", "mount", "namespace", "missing_binding", "fault_domain", "identity"])
def test_wrong_deployment_is_rejected_without_rewriting_data(configured: PosixDeployment, change: str) -> None:
    initialize(configured)
    marker = configured.artifact_root / "namespace.json"
    original = marker.read_bytes()
    if change == "source":
        candidate = replace(configured, artifact_volume=replace(configured.artifact_volume, source_digest="a" * 64))
    elif change == "mount":
        candidate = replace(
            configured, artifact_volume=replace(configured.artifact_volume, mount_point=configured.artifact_root)
        )
    elif change == "namespace":
        candidate = replace(configured, namespace=replace(configured.namespace, namespace_id="other"))
    elif change == "missing_binding":
        empty = configured.artifact_root.parent / "empty-local-directory"
        empty.mkdir(mode=0o700)
        candidate = replace(configured, artifact_root=empty)
    elif change == "fault_domain":
        candidate = replace(configured, required_fault_domain="storage_restart")
    else:
        candidate = replace(configured, expected_uid=(configured.expected_uid + 1) % 65536)
    report = inspect(candidate)
    assert not report.admitted
    with pytest.raises(StorageDeploymentError):
        open_deployed_store(candidate, limits=LIMITS, storage_limits=STORAGE)
    assert marker.read_bytes() == original
    assert not (configured.control_root / "catalog.sqlite3").exists()


def test_control_volume_cannot_be_declared_local_over_a_network_mount(
    configured: PosixDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize(configured)
    original = deployment_io.read_mounts()[0]
    remote = replace(original, mount_id=72, mount_point=configured.control_root, filesystem_type="nfs4")
    monkeypatch.setattr(deployment_io, "read_mounts", lambda: (original, remote))
    candidate = replace(
        configured,
        control_volume=replace(configured.control_volume, mount_point=configured.control_root, filesystem_type="nfs4"),
    )
    report = inspect(candidate)
    assert not report.admitted
    assert any(check.name == "control_requires_local_persistent_volume" for check in report.checks)


@pytest.mark.parametrize("change", ["overlap", "alias_budget", "total_budget", "temporary_budget", "wal_budget"])
def test_overlapping_roots_and_shared_capacity_are_rejected(configured: PosixDeployment, change: str) -> None:
    initialize(configured)
    if change == "overlap":
        candidate = replace(configured, control_root=configured.artifact_root)
    elif change == "alias_budget":
        candidate = replace(configured, control_volume=replace(configured.control_volume, volume_id="another-budget"))
    elif change == "total_budget":
        candidate = replace(
            configured,
            artifact_volume=replace(configured.artifact_volume, capacity_bytes=24 * MIB),
            control_volume=replace(configured.control_volume, capacity_bytes=24 * MIB),
            staging_volume=replace(configured.staging_volume, capacity_bytes=24 * MIB),
        )
    elif change == "temporary_budget":
        candidate = replace(configured, artifact_volume=replace(configured.artifact_volume, reserved_bytes=MIB))
    else:
        candidate = replace(configured, control_volume=replace(configured.control_volume, reserved_bytes=MIB))
    assert not inspect(candidate).admitted


def test_group_read_policy_only_sets_modes_on_new_objects(configured: PosixDeployment) -> None:
    configured.artifact_root.chmod(0o750)
    candidate = replace(configured, access_policy=PosixAccessPolicy(shared_group_id=os.getegid()))
    initialize(candidate)
    root_mode = candidate.artifact_root.stat().st_mode
    assert (physical_root(candidate) / "chunks").stat().st_mode & 0o777 == 0o750
    assert physical_root(candidate).stat().st_mode & 0o777 == 0o750
    assert (candidate.artifact_root / "namespace.json").stat().st_mode & 0o777 == 0o440
    assert candidate.artifact_root.stat().st_mode == root_mode


def test_opened_directory_must_match_the_admitted_inode(configured: PosixDeployment) -> None:
    initialize(configured)
    guard = placement_io.DirectoryGuard.capture(configured.artifact_root)
    other = configured.artifact_root.parent / "other"
    other.mkdir(mode=0o700)
    descriptor = os.open(other, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(StorageDeploymentError, match="changed_after_inspection"):
            guard.validate_opened(descriptor)
    finally:
        os.close(descriptor)


def fixture_mount(point: Path, root: str, mount_id: int) -> MountRecord:
    def escape(value: str) -> str:
        return value.replace("\\", "\\134").replace(" ", "\\040").replace("\t", "\\011").replace("\n", "\\012")

    data = f"{mount_id} 1 0:71 {escape(root)} {escape(str(point))} rw - ext4 fixture-device rw\n"
    return parse_mountinfo(os.fsencode(data))[0]


@pytest.mark.parametrize("relation", ["equal", "ancestor", "descendant", "sibling", "unknown"])
def test_mount_coordinates_reject_alias_overlap_but_allow_siblings(
    configured: PosixDeployment, monkeypatch: pytest.MonkeyPatch, relation: str
) -> None:
    initialize(configured)
    control_root = {
        "equal": "/fixture/pool/run",
        "ancestor": "/fixture/pool",
        "descendant": "/fixture/pool/run/control",
        "sibling": "/fixture/pool/other",
        "unknown": "/fixture/unknown",
    }[relation]
    artifact_mount = fixture_mount(configured.artifact_root, "/fixture/pool/run", 81)
    control_mount = fixture_mount(configured.control_root, control_root, 82)
    if relation == "unknown":
        control_mount = replace(control_mount, _root_components=None)
    original = deployment_io.read_mounts()[0]
    monkeypatch.setattr(deployment_io, "read_mounts", lambda: (original, artifact_mount, control_mount))

    def identity(path: Path) -> tuple[int, int]:
        info = path.stat()
        return info.st_dev, info.st_ino

    artifact_ids = {identity(configured.artifact_root), identity(physical_root(configured))} | {
        identity(physical_root(configured) / name) for name in file_io.ARTIFACT_DIRECTORIES
    }
    control_id = identity(configured.control_root)

    def mount_id(descriptor: int) -> int:
        info = os.fstat(descriptor)
        current = info.st_dev, info.st_ino
        return 81 if current in artifact_ids else (82 if current == control_id else 71)

    monkeypatch.setattr(placement_io, "descriptor_mount_id", mount_id)
    candidate = replace(
        configured,
        artifact_volume=replace(
            configured.artifact_volume, mount_point=artifact_mount.mount_point, root_digest=artifact_mount.root_digest
        ),
        control_volume=replace(
            configured.control_volume, mount_point=control_mount.mount_point, root_digest=control_mount.root_digest
        ),
    )
    report = inspect(candidate)
    assert report.admitted == (relation == "sibling")
    if relation != "sibling":
        expected = "directory_separation_unproven" if relation == "unknown" else "physical_directories_overlap"
        assert any(check.name == expected for check in report.checks)
        with pytest.raises(StorageDeploymentError, match=expected):
            open_deployed_store(candidate, limits=LIMITS, storage_limits=STORAGE)
        assert not (configured.control_root / "catalog.sqlite3").exists()
    assert control_root.encode() not in report.to_bytes()
    assert b"fixture-device" not in report.to_bytes()
    assert control_root not in repr(control_mount)


def test_directory_inode_alias_is_rejected_without_changing_either_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    real_stat = Path.stat
    left_info = left.stat()

    def aliased_stat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
        return left_info if path == right else real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", aliased_stat)
    with pytest.raises(StorageDeploymentError, match="physical_directories_overlap"):
        placement_io.require_separate(left, right)


def test_different_mount_devices_do_not_prove_separation(tmp_path: Path) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    first = fixture_mount(left, "/fixture/pool", 81)
    second = replace(fixture_mount(right, "/fixture/pool/nested", 82), device="0:82")
    with pytest.raises(StorageDeploymentError, match="physical_directories_overlap"):
        placement_io.require_mount_separation(left, first, right, second)


@pytest.mark.parametrize("name", (*file_io.ARTIFACT_DIRECTORIES, "."))
@pytest.mark.parametrize("access", ["publisher", "reader"])
def test_existing_submount_is_rejected_before_any_writer_operation(
    configured: PosixDeployment, monkeypatch: pytest.MonkeyPatch, name: str, access: str
) -> None:
    initialize(configured)
    candidate = (
        configured
        if access == "publisher"
        else replace(configured, access="reader", control_root=None, control_volume=None)
    )
    child = (physical_root(configured) / name).stat()

    def mount_id(descriptor: int) -> int:
        info = os.fstat(descriptor)
        return 72 if (info.st_dev, info.st_ino) == (child.st_dev, child.st_ino) else 71

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("submount admission must fail before mutation")

    monkeypatch.setattr(placement_io, "descriptor_mount_id", mount_id)
    monkeypatch.setattr(artifact_io, "exclusive_lock", forbidden)
    monkeypatch.setattr(artifact_io, "child_directory", forbidden)
    monkeypatch.setattr(artifact_io, "install_file", forbidden)
    with pytest.raises(StorageDeploymentError, match="artifact_child_mount_mismatch"):
        open_deployed_store(candidate, limits=LIMITS, storage_limits=STORAGE)


def test_child_mount_is_rechecked_after_inspection(
    configured: PosixDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize(configured)
    admit = deployment_io._admit
    child = (physical_root(configured) / "chunks").stat()

    def changed_after_admission(*args: object, **kwargs: object) -> object:
        admission = admit(*args, **kwargs)

        def mount_id(descriptor: int) -> int:
            info = os.fstat(descriptor)
            return 72 if (info.st_dev, info.st_ino) == (child.st_dev, child.st_ino) else 71

        monkeypatch.setattr(placement_io, "descriptor_mount_id", mount_id)
        return admission

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("changed child mount must be rejected before the writer lock")

    monkeypatch.setattr(deployment_io, "_admit", changed_after_admission)
    monkeypatch.setattr(artifact_io, "exclusive_lock", forbidden)
    with pytest.raises(StorageDeploymentError, match="artifact_child_mount_mismatch"):
        open_deployed_store(configured, limits=LIMITS, storage_limits=STORAGE)
