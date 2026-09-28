# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Contract fixtures deliberately have no filesystem root or private fd."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from relax.distributed.weight_sync import (
    ChunkSpec,
    Codec,
    CodecLimits,
    DeltaCodecError,
    ModelSchema,
    SnapshotLimits,
    TensorEntry,
    TensorSpec,
    build_manifest,
    reconstruct,
    verify_snapshot,
)
from relax.distributed.weight_sync.codec import EncodedChunk
from relax.distributed.weight_sync.codec.format import content_hash
from relax.distributed.weight_sync.manifest import ChunkRecord, IndexPage, PageRef
from relax.distributed.weight_sync.storage import (
    DiskSnapshotStore,
    ObjectReceipt,
    OfflineArchive,
    OfflineCatalog,
    PosixArtifactStore,
    ProducerCatalog,
    StorageLimits,
    StoreCapabilities,
)
from relax.distributed.weight_sync.storage.placement import DirectoryGuard


LIMITS = SnapshotLimits(codec=CodecLimits(max_chunk_bytes=64), max_chunks_per_page=2)
STORAGE = StorageLimits(max_bytes=1024 * 1024, max_objects=128, max_records=16, max_database_bytes=1024 * 1024)


class MemoryStore:
    """Test double for public storage behavior, not a production backend."""

    def __init__(self) -> None:
        self.limits, self.storage_limits = LIMITS, STORAGE
        self.objects: dict[str, bytes] = {}
        self.reservations: dict[str, bytes] = {}
        self.recovering = True

    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities("memory-fixture", "fixture-v1", "publisher", "fixture", ())

    def prepare_control(self, location: Path) -> DirectoryGuard:
        return DirectoryGuard.capture(location, must_exist=False)

    def begin_recovery(self) -> None:
        self.recovering = True

    def finish_recovery(self) -> None:
        assert not self.reservations
        self.recovering = False

    def reserve_record(self, key: str, data: bytes) -> None:
        previous = self.reservations.setdefault(key, data)
        assert previous == data

    def release_record(self, key: str) -> None:
        self.reservations.pop(key, None)

    def put_immutable(self, key: str, data: bytes, *, expected_hash: str) -> ObjectReceipt:
        assert content_hash(data) == expected_hash
        assert not self.recovering or key == "authority.json" or key.startswith("catalog/")
        assert self.objects.setdefault(key, data) == data
        self.release_record(key)
        return ObjectReceipt(key, expected_hash, len(data), "fixture", "fixture-v1", None)

    def read_object(self, key: str, maximum: int, *, length: int | None = None) -> bytes:
        data = self.objects[key]
        assert len(data) <= maximum and (length is None or len(data) == length)
        return data

    def catalog_keys(self) -> list[str]:
        return sorted(key for key in self.objects if key.startswith("catalog/"))

    def write_payload(self, record: ChunkRecord, payload: bytes) -> None:
        EncodedChunk(record.descriptor, payload).validate(self.limits.codec)
        if record.object_key is not None:
            self.put_immutable(record.object_key, payload, expected_hash=content_hash(payload))

    def read_payload(self, record: ChunkRecord) -> bytes:
        data = (
            b""
            if record.object_key is None
            else self.read_object(
                record.object_key, self.limits.codec.max_chunk_bytes, length=record.descriptor.encoded_length
            )
        )
        EncodedChunk(record.descriptor, data).validate(self.limits.codec)
        return data

    def write_index(self, ref: PageRef, data: bytes) -> None:
        IndexPage.from_bytes(data, ref, self.limits)
        self.put_immutable(ref.object_key, data, expected_hash=content_hash(data))

    def read_index(self, ref: PageRef) -> bytes:
        data = self.read_object(ref.object_key, self.limits.max_index_page_bytes, length=ref.byte_length)
        IndexPage.from_bytes(data, ref, self.limits)
        return data


class ReadOnlyView:
    """No put, reservation, writer recovery or control-directory API."""

    def __init__(self, source: MemoryStore) -> None:
        self.source = source
        self.limits, self.storage_limits = source.limits, source.storage_limits

    def capabilities(self) -> StoreCapabilities:
        return replace(self.source.capabilities(), access="reader")

    def read_object(self, key: str, maximum: int, *, length: int | None = None) -> bytes:
        return self.source.read_object(key, maximum, length=length)

    def read_payload(self, record: ChunkRecord) -> bytes:
        return self.source.read_payload(record)

    def read_index(self, ref: PageRef) -> bytes:
        return self.source.read_index(ref)

    def catalog_keys(self) -> list[str]:
        return self.source.catalog_keys()


class Source:
    def read_chunk(self, spec: ChunkSpec) -> bytes:
        data = b"\x00\x80\xff\x7f\x01\x02\x03\x04"
        return data[spec.byte_offset : spec.byte_offset + spec.byte_length]


def model() -> ModelSchema:
    return ModelSchema("a" * 64, "b" * 64, (TensorEntry(TensorSpec("weight", "uint8", (8,))),), 4, LIMITS)


def test_catalog_publication_and_offline_rebuild_need_no_posix_private_state(tmp_path: Path) -> None:
    store = MemoryStore()
    reader = ReadOnlyView(store)
    with ProducerCatalog(tmp_path / "control", store, stream_id="test", run_epoch="epoch") as catalog:
        fence = catalog.acquire_writer()
        manifest = build_manifest(
            model(),
            Source(),
            store,
            stream_id="test",
            run_epoch="epoch",
            version=0,
            writer_fence=fence,
            source_step=0,
            exporter_revision="fixture",
            limits=LIMITS,
        )
        record = catalog.publish(manifest, operation_id="first", expected_head=None)
        assert catalog.publish(manifest, operation_id="first", expected_head=None) == record
        offline = OfflineCatalog(reader, stream_id="test", run_epoch="epoch")
        with pytest.raises(DeltaCodecError, match="explicit publisher"):
            offline.seal_archive((0,))
        archive_id = offline.seal_archive((0,), writer=store)
    archive = OfflineArchive.open(reader, expected_archive_id=archive_id)
    assert archive.requested == (record.manifest_id,)
    with DiskSnapshotStore(tmp_path / "snapshots", max_bytes=1024 * 1024, max_generations=2, limits=LIMITS) as disk:
        stage = disk.staging()
        rebuilt = reconstruct(
            manifest.to_bytes(LIMITS), reader, stage, expected_manifest_id=record.manifest_id, limits=LIMITS
        )
        for spec in model().iter_chunks():
            assert rebuilt.read_chunk(spec) == Source().read_chunk(spec)
        stage.reader().close()
    assert not hasattr(store, "root") and not hasattr(reader, "put_immutable")


def test_readonly_protocol_cannot_initialize_a_producer_authority(tmp_path: Path) -> None:
    control = tmp_path / "control"
    with pytest.raises(DeltaCodecError, match="publisher"):
        ProducerCatalog(control, ReadOnlyView(MemoryStore()), stream_id="test", run_epoch="epoch")
    assert not control.exists()


def test_artifact_identity_does_not_depend_on_backend_or_mount_root(tmp_path: Path) -> None:
    memory = MemoryStore()
    memory.finish_recovery()
    expected = build_manifest(
        model(),
        Source(),
        memory,
        stream_id="test",
        run_epoch="epoch",
        version=0,
        writer_fence=1,
        source_step=0,
        exporter_revision="fixture",
        limits=LIMITS,
    )
    for name in ("producer-view", "consumer-view"):
        with PosixArtifactStore(tmp_path / name, writable=True, limits=LIMITS, storage_limits=STORAGE) as store:
            actual = build_manifest(
                model(),
                Source(),
                store,
                stream_id="test",
                run_epoch="epoch",
                version=0,
                writer_fence=1,
                source_step=0,
                exporter_revision="fixture",
                limits=LIMITS,
            )
            assert actual.to_bytes(LIMITS) == expected.to_bytes(LIMITS)
            for key, data in memory.objects.items():
                assert store.read_object(key, len(data), length=len(data)) == data


def test_recovery_barrier_retains_record_reservations_until_export(tmp_path: Path) -> None:
    with PosixArtifactStore(tmp_path / "objects", limits=LIMITS, storage_limits=STORAGE, writable=True) as store:
        store.begin_recovery()
        data = b"pending-record"
        key = "catalog/" + "0" * 20 + "-" + "a" * 64 + ".json"
        store.reserve_record(key, data)
        with pytest.raises(DeltaCodecError, match="pending publication"):
            store.finish_recovery()
        payload = b"new-upload"
        with pytest.raises(DeltaCodecError, match="recovery"):
            store.put_immutable("chunks/" + content_hash(payload), payload, expected_hash=content_hash(payload))
        receipt = store.put_immutable(key, data, expected_hash=content_hash(data))
        assert receipt.content_hash == content_hash(data)
        store.finish_recovery()
        store.put_immutable("chunks/" + content_hash(payload), payload, expected_hash=content_hash(payload))


def test_control_guard_still_rejects_overlapping_artifacts_before_initialization(tmp_path: Path) -> None:
    with PosixArtifactStore(tmp_path / "objects", writable=True) as store:
        control = store.root / "control"
        with pytest.raises(DeltaCodecError, match="separate"):
            ProducerCatalog(control, store, stream_id="test", run_epoch="epoch")
        assert not control.exists()


def test_another_thread_cannot_close_and_release_a_writer_lock(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    with PosixArtifactStore(root, writable=True) as store:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with pytest.raises(DeltaCodecError, match="owning process and thread"):
                executor.submit(store.close).result()
        with pytest.raises(BlockingIOError):
            PosixArtifactStore(root, writable=True)


@pytest.mark.parametrize("phase", ["publication", "archive"])
@pytest.mark.parametrize(
    ("codec", "damage"),
    [
        (codec, damage)
        for codec in (Codec.RAW_V1, Codec.SPARSE_REPLACE_V1, Codec.BITMAP_REPLACE_V1)
        for damage in ("length", "hash", "type")
    ]
    + [(Codec.COPY_BASE, "length"), (Codec.COPY_BASE, "type")],
)
def test_unchecked_backend_payloads_are_rejected_before_publication_or_archive(
    tmp_path: Path, codec: Codec, damage: str, phase: str
) -> None:
    class UncheckedStore(MemoryStore):
        damage: str | None = None

        def read_payload(self, record: ChunkRecord) -> bytes:
            data = b"" if record.object_key is None else self.objects[record.object_key]
            if self.damage == "length":
                return data[:-1] if data else b"unexpected"
            if self.damage == "hash":
                return bytes([data[0] ^ 1]) + data[1:]
            if self.damage == "type":
                return bytearray(data)  # Deliberately violates the reader contract.
            return data

    class BytesSource:
        def __init__(self, data: bytes) -> None:
            self.data = data

        def read_chunk(self, spec: ChunkSpec) -> bytes:
            return self.data[spec.byte_offset : spec.byte_offset + spec.byte_length]

    store = UncheckedStore()
    store.limits = SnapshotLimits(codec=CodecLimits(max_chunk_bytes=4096))
    schema = ModelSchema(
        "a" * 64, "b" * 64, (TensorEntry(TensorSpec("weight", "uint8", (4096,))),), 4096, store.limits
    )
    before = BytesSource(bytes(4096))
    target = {
        Codec.COPY_BASE: bytes(4096),
        Codec.RAW_V1: b"\x01" * 4096,
        Codec.SPARSE_REPLACE_V1: b"\x01" + bytes(4095),
        Codec.BITMAP_REPLACE_V1: b"\x01\x00" * 2048,
    }[codec]
    with ProducerCatalog(tmp_path / "control", store, stream_id="test", run_epoch="epoch") as catalog:
        fence = catalog.acquire_writer()
        full = build_manifest(
            schema,
            before,
            store,
            stream_id="test",
            run_epoch="epoch",
            version=0,
            writer_fence=fence,
            source_step=0,
            exporter_revision="fixture",
            limits=store.limits,
        )
        first = catalog.publish(full, operation_id="base", expected_head=None)
        base = verify_snapshot(schema, full.target, before, limits=store.limits)
        delta = build_manifest(
            schema,
            BytesSource(target),
            store,
            stream_id="test",
            run_epoch="epoch",
            version=1,
            writer_fence=fence,
            source_step=1,
            exporter_revision="fixture",
            base=base,
            limits=store.limits,
        )
        page = IndexPage.from_bytes(store.read_index(delta.pages[0]), delta.pages[0], store.limits)
        assert page.records[0].descriptor.codec == codec
        if phase == "archive":
            catalog.publish(delta, operation_id="target", expected_head=first.manifest_id)
            # FULL data must stay readable so corruption reaches the delta record.
            original_read = store.read_payload

            def damaged_delta(record: ChunkRecord) -> bytes:
                store.damage = damage if record.descriptor.target_version == 1 else None
                return original_read(record)

            store.read_payload = damaged_delta
            with pytest.raises(DeltaCodecError, match="payload"):
                OfflineCatalog(store, stream_id="test", run_epoch="epoch").seal_archive((1,), prefer_full=False)
            assert not any(key.startswith("archives/") for key in store.objects)
        else:
            store.damage = damage
            with pytest.raises(DeltaCodecError, match="payload"):
                catalog.publish(delta, operation_id="target", expected_head=first.manifest_id)
            assert catalog.resolve("target") is None
            assert catalog.head() == first.manifest_id
            assert f"manifests/{delta.manifest_id(store.limits)}.json" not in store.objects
