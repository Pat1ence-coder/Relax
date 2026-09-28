# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real POSIX files, local WAL publication, and offline snapshot replay."""

import hashlib
import multiprocessing
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from relax.distributed.weight_sync import (
    CanonicalChunk,
    ChunkSpec,
    CodecLimits,
    DeltaCodecError,
    Manifest,
    ModelSchema,
    SnapshotLimits,
    TensorEntry,
    TensorSpec,
    VerifiedSnapshot,
    build_manifest,
    reconstruct,
    verify_snapshot,
)
from relax.distributed.weight_sync.storage import (
    DiskSnapshotStore,
    OfflineArchive,
    OfflineCatalog,
    PosixArtifactStore,
    ProducerCatalog,
    PublicationUncertain,
    StorageLimits,
    open_snapshot,
)
from relax.distributed.weight_sync.storage import files as file_io
from relax.distributed.weight_sync.storage import staging as staging_io


LIMITS = SnapshotLimits(codec=CodecLimits(max_chunk_bytes=1024), max_chunks_per_page=2)
DISK = StorageLimits(max_bytes=32 * 1024 * 1024, max_objects=10000, max_records=256, max_database_bytes=1024 * 1024)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def schema(kind: str = "dense") -> ModelSchema:
    names = (
        ("model.weight", "model.scale") if kind == "dense" else ("model.experts.0.weight", "model.experts.1.weight")
    )
    return ModelSchema(
        "a" * 64,
        "b" * 64,
        (
            TensorEntry(TensorSpec(names[0], "bfloat16", (1027,))),
            TensorEntry(TensorSpec(names[1], "float32", (259,))),
            TensorEntry(TensorSpec("empty", "uint8", (0,)), "buffer"),
            TensorEntry(TensorSpec("tied", "bfloat16", (1027,)), alias_of=names[0]),
        ),
        1024,
        LIMITS,
    )


class Reader:
    def __init__(self, model: ModelSchema, version: int = 0) -> None:
        self.data = {}
        for entry in model.tensors:
            if entry.alias_of is None:
                value = bytearray(entry.tensor.nbytes)
                if value:
                    value[version % len(value)] = version % 256
                self.data[entry.tensor.name] = bytes(value)

    def read_chunk(self, spec: ChunkSpec) -> bytes:
        return self.data[spec.tensor.name][spec.byte_offset : spec.byte_offset + spec.byte_length]


def emit(
    store: PosixArtifactStore,
    fence: int,
    *,
    version: int = 0,
    base: VerifiedSnapshot | None = None,
    model: ModelSchema | None = None,
) -> tuple[Manifest, Reader]:
    model = schema() if model is None else model
    reader = Reader(model, version)
    manifest = build_manifest(
        model,
        reader,
        store,
        stream_id="test",
        run_epoch="epoch-1",
        version=version,
        writer_fence=fence,
        source_step=version,
        exporter_revision="test-v1",
        base=base,
        limits=LIMITS,
    )
    return manifest, reader


def store_at(path: Path, *, writable: bool = True, storage_limits: StorageLimits = DISK) -> PosixArtifactStore:
    return PosixArtifactStore(path, writable=writable, limits=LIMITS, storage_limits=storage_limits)


def catalog_at(path: Path, store: PosixArtifactStore) -> ProducerCatalog:
    return ProducerCatalog(path, store, stream_id="test", run_epoch="epoch-1")


def test_shared_store_idempotence_readonly_conflicts_and_reopen(tmp_path: Path) -> None:
    data = b"payload"
    key = "chunks/" + digest(data)
    with store_at(tmp_path / "shared") as store:
        store.put_object(key, data)
        store.put_object(key, data)
        assert store._objects == 1 and store._bytes == len(data)
        with pytest.raises(BlockingIOError):
            store_at(tmp_path / "shared")
        with store_at(tmp_path / "shared", writable=False) as reader:
            assert reader.read_object(key, 1024) == data
            with pytest.raises(DeltaCodecError, match="read-only"):
                reader.put_object(key, data)
        with pytest.raises(DeltaCodecError):
            store.put_object(key, b"different")
    with store_at(tmp_path / "shared") as reopened:
        assert reopened._objects == 1 and reopened._bytes == len(data)


@pytest.mark.parametrize("key", ["../escape", "/absolute", "chunks/../escape", "chunks/abc", "catalog/1.json"])
def test_shared_store_rejects_path_escape(tmp_path: Path, key: str) -> None:
    with store_at(tmp_path / "shared") as store:
        with pytest.raises(DeltaCodecError):
            store.put_object(key, b"x")
        with pytest.raises(DeltaCodecError):
            store.read_object(key, 1024)


def test_shared_store_rejects_symlink_file_directory_and_fifo(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.write_bytes(b"safe")
    with store_at(tmp_path / "shared") as store:
        key = "chunks/" + digest(b"safe")
        (store.root / key).symlink_to(outside)
        with pytest.raises(OSError):
            store.read_object(key, 1024)
        with pytest.raises(OSError):
            store.put_object(key, b"safe")
        assert outside.read_bytes() == b"safe"
        (store.root / key).unlink()
        os.mkfifo(store.root / key)
        with pytest.raises(DeltaCodecError, match="type"):
            store.read_object(key, 1024)
        (store.root / key).unlink()
    (tmp_path / "shared/chunks").rmdir()
    (tmp_path / "shared/chunks").symlink_to(tmp_path)
    with pytest.raises(OSError):
        store_at(tmp_path / "shared", writable=False)


@pytest.mark.parametrize("quota", ["bytes", "objects"])
def test_artifact_quota_survives_restart(tmp_path: Path, quota: str) -> None:
    limits = replace(DISK, max_bytes=3) if quota == "bytes" else replace(DISK, max_objects=1)
    with store_at(tmp_path / "shared", storage_limits=limits) as store:
        store.put_object("chunks/" + digest(b"abc"), b"abc")
    with store_at(tmp_path / "shared", storage_limits=limits) as store:
        store.put_object("chunks/" + digest(b"abc"), b"abc")
        with pytest.raises(DeltaCodecError):
            store.put_object("chunks/" + digest(b"d"), b"d")
        assert not list((store.root / "chunks").glob(".tmp-*"))


def test_bounded_read_checks_stat_before_allocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with store_at(tmp_path / "shared") as store:
        path = store.root / ("chunks/" + "a" * 64)
        with path.open("wb") as file:
            file.truncate(1 << 30)

        def forbidden_read(*args: object) -> bytes:
            pytest.fail("read must not occur for an oversized object")

        monkeypatch.setattr(os, "read", forbidden_read)
        with pytest.raises(DeltaCodecError, match="size"):
            store.read_object("chunks/" + "a" * 64, 1024)


def test_publication_idempotence_fence_head_and_authority_binding(tmp_path: Path) -> None:
    with store_at(tmp_path / "shared") as store, catalog_at(tmp_path / "control", store) as catalog:
        fence = catalog.acquire_writer()
        manifest, reader = emit(store, fence)
        assert OfflineCatalog(store, stream_id="test", run_epoch="epoch-1").records() == ()
        first = catalog.publish(manifest, operation_id="initial", expected_head=None)
        assert catalog.head() == first.manifest_id and catalog.resolve("initial") == first
        assert catalog.publish(manifest, operation_id="initial", expected_head=None) == first
        catalog.acquire_writer()
        # A committed retry remains valid after fence takeover.
        assert catalog.publish(manifest, operation_id="initial", expected_head=None) == first
        base = verify_snapshot(manifest.schema, manifest.target, reader, limits=LIMITS)
        stale, _ = emit(store, fence, version=1, base=base)
        with pytest.raises(DeltaCodecError, match="fence"):
            catalog.publish(stale, operation_id="stale", expected_head=first.manifest_id)
        with pytest.raises(DeltaCodecError, match="different request"):
            catalog.publish(stale, operation_id="initial", expected_head=first.manifest_id)
        with pytest.raises(BlockingIOError):
            catalog_at(tmp_path / "control", store)
        with pytest.raises(DeltaCodecError, match="conflict"):
            catalog_at(tmp_path / "other-control", store)
        assert catalog.head() == first.manifest_id and catalog.resolve("stale") is None
    with store_at(tmp_path / "shared") as store, catalog_at(tmp_path / "control", store) as catalog:
        assert catalog.resolve("initial") == first
        assert catalog.acquire_writer() > fence


@pytest.mark.parametrize("case", ["before_commit", "after_commit", "export"])
def test_unknown_publication_can_be_resolved_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    with store_at(tmp_path / "shared") as store, catalog_at(tmp_path / "control", store) as catalog:
        manifest, _ = emit(store, catalog.acquire_writer())
        commit, export = catalog._commit, catalog._export

        def failed_commit() -> None:
            if case == "after_commit":
                commit()
            raise OSError("lost commit response")

        def failed_export(record: object) -> None:
            raise OSError("export unavailable")

        monkeypatch.setattr(
            catalog, "_export" if case == "export" else "_commit", failed_export if case == "export" else failed_commit
        )
        with pytest.raises(PublicationUncertain) as error:
            catalog.publish(manifest, operation_id="publish-0", expected_head=None)
        assert error.value.operation_id == "publish-0"
        assert (catalog.resolve("publish-0") is None) == (case == "before_commit")
        monkeypatch.setattr(catalog, "_commit", commit)
        monkeypatch.setattr(catalog, "_export", export)
        committed = catalog.publish(manifest, operation_id="publish-0", expected_head=None)
        assert catalog.head() == committed.manifest_id
        assert len(OfflineCatalog(store, stream_id="test", run_epoch="epoch-1").records()) == 1


def _crashing_publisher(root: str, phase: str) -> None:
    path = Path(root)
    with store_at(path / "shared") as store, catalog_at(path / "control", store) as catalog:
        manifest, _ = emit(store, catalog.acquire_writer())
        commit, export = catalog._commit, catalog._export

        def crash_commit() -> None:
            if phase == "before_commit":
                os._exit(91)
            commit()
            os._exit(91)

        def crash_export(record: object) -> None:
            if phase == "after_export":
                export(record)
            os._exit(91)

        if phase in ("before_commit", "after_commit"):
            catalog._commit = crash_commit
        else:
            catalog._export = crash_export
        catalog.publish(manifest, operation_id="crash-publish", expected_head=None)


@pytest.mark.parametrize("phase", ["before_commit", "after_commit", "before_export", "after_export"])
def test_process_crash_recovers_the_authoritative_decision(tmp_path: Path, phase: str) -> None:
    process = multiprocessing.get_context("fork").Process(target=_crashing_publisher, args=(str(tmp_path), phase))
    process.start()
    process.join(15)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("crash fixture hung")
    assert process.exitcode == 91
    with store_at(tmp_path / "shared") as store, catalog_at(tmp_path / "control", store) as catalog:
        result = catalog.resolve("crash-publish")
        assert (result is None) == (phase == "before_commit")
        records = OfflineCatalog(store, stream_id="test", run_epoch="epoch-1").records()
        assert len(records) == (0 if result is None else 1)
        if result is not None:
            assert records == (result,) and catalog.head() == result.manifest_id
        assert catalog.acquire_writer() == 2


def test_record_quota_is_checked_before_committing(tmp_path: Path) -> None:
    with store_at(tmp_path / "shared") as store, catalog_at(tmp_path / "control", store) as catalog:
        manifest, _ = emit(store, catalog.acquire_writer())
        store.write_manifest(manifest)
        store.storage_limits = replace(DISK, max_objects=store._objects)
        with pytest.raises(DeltaCodecError, match="object count"):
            catalog.publish(manifest, operation_id="no-space", expected_head=None)
        assert catalog.head() is None and catalog.resolve("no-space") is None


def test_fallback_full_keeps_identity_and_resets_future_dependency_depth(tmp_path: Path) -> None:
    with (
        store_at(tmp_path / "shared", storage_limits=replace(DISK, max_chain_depth=1)) as store,
        catalog_at(tmp_path / "control", store) as catalog,
    ):
        fence = catalog.acquire_writer()
        initial, reader = emit(store, fence)
        first = catalog.publish(initial, operation_id="v0", expected_head=None)
        base = verify_snapshot(initial.schema, initial.target, reader, limits=LIMITS)
        delta, reader = emit(store, fence, version=1, base=base)
        second = catalog.publish(delta, operation_id="v1", expected_head=first.manifest_id)
        base = verify_snapshot(delta.schema, delta.target, reader, limits=LIMITS)
        next_delta, _ = emit(store, fence, version=2, base=base)
        with pytest.raises(DeltaCodecError, match="depth"):
            catalog.publish(next_delta, operation_id="v2", expected_head=second.manifest_id)
        full, _ = emit(store, fence, version=1)
        attached = catalog.publish(full, operation_id="full-v1", expected_head=second.manifest_id, attach_full=True)
        assert attached.target == second.target and catalog.head() == second.manifest_id
        third = catalog.publish(next_delta, operation_id="v2", expected_head=second.manifest_id)
        assert third.base_manifest_id == attached.manifest_id
        archive_id = OfflineCatalog(store, stream_id="test", run_epoch="epoch-1").seal_archive((2,))
        archive = OfflineArchive.open(store, expected_archive_id=archive_id)
        assert [record.manifest_id for record in archive.records] == [attached.manifest_id, third.manifest_id]


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_one_hundred_synthetic_versions_replay_offline_after_producer_closes(tmp_path: Path, kind: str) -> None:
    model = schema(kind)
    originals = {}
    with store_at(tmp_path / "shared") as store, catalog_at(tmp_path / "control", store) as catalog:
        fence, base, head = catalog.acquire_writer(), None, None
        for version in range(101):
            manifest, reader = emit(
                store, fence, version=version, base=None if version % 10 == 0 else base, model=model
            )
            publication = catalog.publish(manifest, operation_id=f"version-{version}", expected_head=head)
            base = verify_snapshot(model, manifest.target, reader, limits=LIMITS)
            originals[version] = reader.data
            head = publication.manifest_id
        archive_id = OfflineCatalog(store, stream_id="test", run_epoch="epoch-1").seal_archive(tuple(range(1, 101)))
    # Neither the producer nor its database is opened during consumption.
    with (
        store_at(tmp_path / "shared", writable=False) as store,
        DiskSnapshotStore(
            tmp_path / "snapshots", max_bytes=2 * 1024 * 1024, max_generations=101, limits=LIMITS
        ) as snapshots,
    ):
        archive = OfflineArchive.open(store, expected_archive_id=archive_id)
        assert len(archive.requested) == 100 and len(archive.records) == 101  # Includes V0 dependency.
        base, previous_reader = None, None
        paths = {}
        for record, manifest in archive.manifests(store):
            stage = snapshots.staging()
            rebuilt = reconstruct(
                manifest.to_bytes(LIMITS),
                store,
                stage,
                expected_manifest_id=record.manifest_id,
                base=None if manifest.kind == "FULL" else base,
                limits=LIMITS,
            )
            for spec in model.iter_chunks():
                expected = originals[record.target.version][spec.tensor.name][
                    spec.byte_offset : spec.byte_offset + spec.byte_length
                ]
                assert rebuilt.read_chunk(spec) == expected
            if previous_reader is not None:
                previous_reader.close()
            base, previous_reader = rebuilt, stage.reader()
            paths[record.target.version] = stage.path
        previous_reader.close()
    reopened, reader = open_snapshot(paths[100], expected_identity=base.identity, limits=LIMITS)
    assert reopened.identity == base.identity
    reader.close()


@pytest.mark.parametrize("damage", ["base_record", "payload", "index", "manifest", "seal"])
def test_offline_archive_rejects_missing_dependency_or_corruption(tmp_path: Path, damage: str) -> None:
    with store_at(tmp_path / "shared") as store, catalog_at(tmp_path / "control", store) as catalog:
        fence = catalog.acquire_writer()
        manifest, reader = emit(store, fence)
        first = catalog.publish(manifest, operation_id="v0", expected_head=None)
        base = verify_snapshot(manifest.schema, manifest.target, reader, limits=LIMITS)
        delta, _ = emit(store, fence, version=1, base=base)
        catalog.publish(delta, operation_id="v1", expected_head=first.manifest_id)
        offline = OfflineCatalog(store, stream_id="test", run_epoch="epoch-1")
        with pytest.raises(DeltaCodecError, match="missing"):
            offline.seal_archive((2,))
        archive_id = offline.seal_archive((1,))
    paths = {
        "base_record": first.key,
        "payload": next((tmp_path / "shared/chunks").iterdir()).relative_to(tmp_path / "shared"),
        "index": manifest.pages[0].object_key,
        "manifest": f"manifests/{first.manifest_id}.json",
        "seal": f"archives/{archive_id}.json",
    }
    (tmp_path / "shared" / paths[damage]).unlink()
    with store_at(tmp_path / "shared", writable=False) as store:
        with pytest.raises((DeltaCodecError, FileNotFoundError)):
            OfflineArchive.open(store, expected_archive_id=archive_id)


def test_disk_reopen_uses_trusted_identity_and_actual_bytes(tmp_path: Path) -> None:
    with (
        store_at(tmp_path / "shared") as store,
        DiskSnapshotStore(tmp_path / "snapshots", max_bytes=20000, max_generations=2, limits=LIMITS) as snapshots,
    ):
        manifest, reader = emit(store, 1)
        stage = snapshots.staging()
        rebuilt = reconstruct(
            manifest.to_bytes(LIMITS), store, stage, expected_manifest_id=manifest.manifest_id(LIMITS), limits=LIMITS
        )
        assert stage.path.exists() and rebuilt.read_chunk(next(manifest.schema.iter_chunks())) == reader.read_chunk(
            next(manifest.schema.iter_chunks())
        )
        with pytest.raises(DeltaCodecError, match="sealed"):
            stage.abort()
        with pytest.raises(DeltaCodecError, match="trusted identity"):
            open_snapshot(stage.path, expected_identity=replace(manifest.target, run_epoch="other"), limits=LIMITS)
        stage.reader().close()
        data = stage.path / "weights.bin"
        data.chmod(0o600)
        with data.open("r+b") as file:
            file.write(b"\x01")
        with pytest.raises(DeltaCodecError, match="root"):
            open_snapshot(stage.path, expected_identity=manifest.target, limits=LIMITS)


@pytest.mark.parametrize("quota", ["bytes", "generations"])
def test_disk_snapshot_quota_persists_and_failed_begin_is_safe(tmp_path: Path, quota: str) -> None:
    max_bytes = 7000 if quota == "bytes" else 20000
    max_generations = 2 if quota == "bytes" else 1
    with store_at(tmp_path / "shared") as store:
        manifest, _ = emit(store, 1)
        with DiskSnapshotStore(
            tmp_path / "snapshots", max_bytes=max_bytes, max_generations=max_generations, limits=LIMITS
        ) as snapshots:
            stage = snapshots.staging()
            reconstruct(
                manifest.to_bytes(LIMITS),
                store,
                stage,
                expected_manifest_id=manifest.manifest_id(LIMITS),
                limits=LIMITS,
            )
            stage.reader().close()
        with DiskSnapshotStore(
            tmp_path / "snapshots", max_bytes=max_bytes, max_generations=max_generations, limits=LIMITS
        ) as snapshots:
            candidate = snapshots.staging()
            with pytest.raises(DeltaCodecError):
                reconstruct(
                    manifest.to_bytes(LIMITS),
                    store,
                    candidate,
                    expected_manifest_id=manifest.manifest_id(LIMITS),
                    limits=LIMITS,
                )
            assert not candidate.path.exists() and stage.path.exists()
            assert snapshots.cleanup_incomplete() == 0


def test_unfinished_generation_reservations_cannot_overlap(tmp_path: Path) -> None:
    with (
        store_at(tmp_path / "shared") as store,
        DiskSnapshotStore(tmp_path / "snapshots", max_bytes=20000, max_generations=3, limits=LIMITS) as snapshots,
    ):
        manifest, _ = emit(store, 1)
        first, second = snapshots.staging(), snapshots.staging()
        first.begin(manifest.schema, manifest.target)
        with pytest.raises(DeltaCodecError, match="one unfinished"):
            second.begin(manifest.schema, manifest.target)
        second.abort()
        assert snapshots.cleanup_incomplete() == 0 and first.path.exists()
        first.abort()
        assert snapshots._usage() == (0, 0)


def test_short_writes_and_reads_finish_the_exact_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with store_at(tmp_path / "shared") as store:
        write, read = os.write, os.read
        monkeypatch.setattr(os, "write", lambda fd, data: write(fd, data[:3]))
        monkeypatch.setattr(os, "read", lambda fd, size: read(fd, min(size, 2)))
        data = b"arbitrary-payload"
        key = "chunks/" + digest(data)
        store.put_object(key, data)
        assert store.read_object(key, 1024) == data


@pytest.mark.parametrize("failure", ["write", "fsync", "after_link"])
def test_atomic_file_failure_never_exposes_partial_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    with store_at(tmp_path / "shared") as store:
        original_write, original_sync, original_link = os.write, os.fsync, os.link
        fired = False

        def failed_write(fd: int, data: bytes) -> int:
            original_write(fd, data[:1])
            raise OSError("disk full")

        def failed_sync(fd: int) -> None:
            nonlocal fired
            if not fired:
                fired = True
                raise OSError("fsync failed")
            original_sync(fd)

        def failed_link(*args: object, **kwargs: object) -> None:
            original_link(*args, **kwargs)
            raise OSError("lost link response")

        selected = {
            "write": ("write", failed_write),
            "fsync": ("fsync", failed_sync),
            "after_link": ("link", failed_link),
        }
        monkeypatch.setattr(file_io.os, *selected[failure])
        data = b"complete"
        key = "chunks/" + digest(data)
        with pytest.raises(OSError):
            store.put_object(key, data)
        if failure == "after_link":
            assert store.read_object(key, 1024) == data and store._objects == 1
        else:
            assert not (store.root / key).exists() and store._objects == 0
        assert not list((store.root / "chunks").glob(".tmp-*"))
        monkeypatch.setattr(os, "write", original_write)
        monkeypatch.setattr(os, "fsync", original_sync)
        monkeypatch.setattr(os, "link", original_link)
        store.put_object(key, data)
        assert store.read_object(key, 1024) == data


def test_failed_inventory_poisoning_prevents_quota_bypass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    limits = replace(DISK, max_objects=1)
    with store_at(tmp_path / "shared", storage_limits=limits) as store:
        unlink = os.unlink

        def failed_unlink(path: str, *args: object, **kwargs: object) -> None:
            if str(path).startswith(".tmp-"):
                raise OSError("temporary cleanup failed")
            unlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "unlink", failed_unlink)
        with pytest.raises(OSError, match="temporary cleanup failed"):
            store.put_object("chunks/" + digest(b"a"), b"a")
        monkeypatch.setattr(os, "unlink", unlink)
        with pytest.raises(DeltaCodecError, match="inventory is uncertain"):
            store.put_object("chunks/" + digest(b"b"), b"b")
    with store_at(tmp_path / "shared", storage_limits=limits) as store:
        assert store._objects == 1 and store._bytes == 1
        assert not list((store.root / "chunks").glob(".tmp-*"))
        with pytest.raises(DeltaCodecError, match="object count"):
            store.put_object("chunks/" + digest(b"b"), b"b")


def test_store_rejects_shared_writer_across_threads(tmp_path: Path) -> None:
    with store_at(tmp_path / "shared") as store, ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(store.put_object, "chunks/" + digest(b"x"), b"x")
        with pytest.raises(DeltaCodecError, match="owning process and thread"):
            future.result()
        assert store._objects == 0


def _crashing_stage(root: str, phase: str = "write") -> None:
    path = Path(root)
    with (
        store_at(path / "shared") as store,
        DiskSnapshotStore(path / "snapshots", max_bytes=20000, max_generations=2, limits=LIMITS) as snapshots,
    ):
        manifest, reader = emit(store, 1)
        stage = snapshots.staging()
        if phase == "metadata_link":

            def crash_cleanup(*args: object, **kwargs: object) -> None:
                os._exit(92)

            os.unlink = crash_cleanup
        stage.begin(manifest.schema, manifest.target)
        spec = next(manifest.schema.iter_chunks())
        stage.write_chunk(CanonicalChunk(spec, 0, reader.read_chunk(spec)))
        os._exit(92)


@pytest.mark.parametrize("phase", ["write", "metadata_link"])
def test_crashed_staging_is_cleaned_and_can_be_rebuilt(tmp_path: Path, phase: str) -> None:
    process = multiprocessing.get_context("fork").Process(target=_crashing_stage, args=(str(tmp_path), phase))
    process.start()
    process.join(15)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("staging crash fixture hung")
    assert process.exitcode == 92
    leftover = next((tmp_path / "snapshots").glob("generation-*"))
    budget = (leftover / "weights.bin").stat().st_size + (leftover / "metadata.json").stat().st_size + 256
    with (
        store_at(tmp_path / "shared") as store,
        DiskSnapshotStore(tmp_path / "snapshots", max_bytes=budget, max_generations=2, limits=LIMITS) as snapshots,
    ):
        assert snapshots.cleanup_incomplete() == 1
        manifest, _ = emit(store, 1)
        stage = snapshots.staging()
        reconstructed = reconstruct(
            manifest.to_bytes(LIMITS), store, stage, expected_manifest_id=manifest.manifest_id(LIMITS), limits=LIMITS
        )
        assert reconstructed.identity == manifest.target
        assert snapshots.cleanup_incomplete() == 0
        stage.reader().close()


@pytest.mark.parametrize("failure", ["write", "seal"])
def test_disk_staging_failure_aborts_only_private_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    with (
        store_at(tmp_path / "shared") as store,
        DiskSnapshotStore(tmp_path / "snapshots", max_bytes=20000, max_generations=2, limits=LIMITS) as snapshots,
    ):
        manifest, _ = emit(store, 1)
        first = snapshots.staging()
        reconstruct(
            manifest.to_bytes(LIMITS), store, first, expected_manifest_id=manifest.manifest_id(LIMITS), limits=LIMITS
        )
        first.reader().close()
        candidate = snapshots.staging()
        write, sync = staging_io.write_all, os.fsync

        def failed_write(fd: int, data: bytes) -> None:
            write(fd, data[:1])
            raise OSError("partial staging write")

        def failed_seal(fd: int) -> None:
            if fd == candidate._writer:
                raise OSError("seal fsync failed")
            sync(fd)

        if failure == "write":
            monkeypatch.setattr(staging_io, "write_all", failed_write)
        else:
            monkeypatch.setattr(os, "fsync", failed_seal)
        with pytest.raises(OSError):
            reconstruct(
                manifest.to_bytes(LIMITS),
                store,
                candidate,
                expected_manifest_id=manifest.manifest_id(LIMITS),
                limits=LIMITS,
            )
        assert not candidate.path.exists() and first.path.exists()
        assert len(snapshots._generations()) == 1


def test_sqlite_full_retains_original_error_and_last_committed_head(tmp_path: Path) -> None:
    limits = replace(DISK, max_database_bytes=65536, max_records=512)
    with (
        store_at(tmp_path / "shared", storage_limits=limits) as store,
        catalog_at(tmp_path / "control", store) as catalog,
    ):
        fence, head = catalog.acquire_writer(), None
        for version in range(256):
            manifest, _ = emit(store, fence, version=version)
            try:
                record = catalog.publish(manifest, operation_id=f"v{version}", expected_head=head)
            except sqlite3.OperationalError as error:
                assert "full" in str(error).lower() and "rollback" not in str(error).lower()
                assert catalog.resolve(f"v{version}") is None and catalog.head() == head
                assert not catalog._connection().in_transaction
                break
            head = record.manifest_id
        else:
            pytest.fail("small database quota was not enforced")


def test_blocked_wal_checkpoint_stops_further_publication(tmp_path: Path) -> None:
    with store_at(tmp_path / "shared") as store, catalog_at(tmp_path / "control", store) as catalog:
        fence = catalog.acquire_writer()
        initial, _ = emit(store, fence)
        first = catalog.publish(initial, operation_id="v0", expected_head=None)
        observer = sqlite3.connect(tmp_path / "control/catalog.sqlite3", isolation_level=None)
        try:
            observer.execute("BEGIN")
            observer.execute("SELECT * FROM state").fetchone()
            second, _ = emit(store, fence, version=1)
            with pytest.raises(PublicationUncertain):
                catalog.publish(second, operation_id="v1", expected_head=first.manifest_id)
            committed = catalog.resolve("v1")
            assert committed is not None and catalog.head() == committed.manifest_id
            third, _ = emit(store, fence, version=2)
            with pytest.raises(DeltaCodecError, match="checkpoint blocked"):
                catalog.publish(third, operation_id="v2", expected_head=committed.manifest_id)
            assert catalog.resolve("v2") is None
        finally:
            observer.close()
        assert catalog.publish(second, operation_id="v1", expected_head=first.manifest_id) == committed


@pytest.mark.parametrize("quota", ["bytes", "objects"])
def test_uncertain_export_keeps_its_reserved_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quota: str
) -> None:
    with store_at(tmp_path / "shared") as store, catalog_at(tmp_path / "control", store) as catalog:
        manifest, _ = emit(store, catalog.acquire_writer())
        export = catalog._export

        def unavailable(record: object) -> None:
            raise OSError("export unavailable")

        monkeypatch.setattr(catalog, "_export", unavailable)
        with pytest.raises(PublicationUncertain):
            catalog.publish(manifest, operation_id="v0", expected_head=None)
        assert catalog.resolve("v0") is not None and len(store._reservations) == 1
        if quota == "objects":
            store.storage_limits = replace(DISK, max_objects=store._objects + 2)
        else:
            reserved = sum(size for size, _ in store._reservations.values())
            store.storage_limits = replace(DISK, max_bytes=store._bytes + reserved + 1)
        store.put_object("chunks/" + digest(b"x"), b"x")
        with pytest.raises(DeltaCodecError):
            store.put_object("chunks/" + digest(b"y"), b"y")
        monkeypatch.setattr(catalog, "_export", export)
        committed = catalog.publish(manifest, operation_id="v0", expected_head=None)
        assert not store._reservations and (store.root / committed.key).exists()


def test_snapshot_store_rejects_a_second_thread_without_releasing_its_lock(tmp_path: Path) -> None:
    with (
        DiskSnapshotStore(tmp_path / "snapshots", max_bytes=20000, max_generations=2, limits=LIMITS) as snapshots,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        for method in (snapshots.staging, snapshots.close):
            with pytest.raises(DeltaCodecError, match="owning process and thread"):
                executor.submit(method).result()
        assert snapshots._usage() == (0, 0)
        with pytest.raises(BlockingIOError):
            DiskSnapshotStore(tmp_path / "snapshots", max_bytes=20000, max_generations=2, limits=LIMITS)
