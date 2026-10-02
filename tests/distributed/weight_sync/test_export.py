# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Capture conformance with owned bytes, failure atomicity and disk replay."""

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from relax.distributed.weight_sync import (
    CanonicalTile,
    ChunkSpec,
    CodecLimits,
    DeltaCodecError,
    ExportBudget,
    ExportRequest,
    ModelSchema,
    SnapshotLimits,
    SourceExportPlan,
    TensorEntry,
    TensorOwner,
    TensorSpec,
    build_manifest,
    reconstruct,
    validate_receipts,
)
from relax.distributed.weight_sync.storage import DiskSnapshotStore, PosixArtifactStore, open_snapshot
from relax.distributed.weight_sync.storage import capture as capture_io


LIMITS = SnapshotLimits(codec=CodecLimits(max_chunk_bytes=32))
BUDGET = ExportBudget(tile_bytes=8, max_cpu_bytes=136, max_pinned_bytes=0, max_gpu_bytes=0)
REQUEST = ExportRequest("test", "epoch", 0, 0, "capture-test")


def model() -> ModelSchema:
    return ModelSchema(
        "a" * 64,
        "b" * 64,
        (
            TensorEntry(TensorSpec("weight", "bfloat16", (19,))),
            TensorEntry(TensorSpec("tied", "bfloat16", (19,)), alias_of="weight"),
            TensorEntry(TensorSpec("counter", "int64", (1,)), "buffer"),
            TensorEntry(TensorSpec("empty", "float32", (0, 3)), "buffer"),
        ),
        chunk_bytes=32,
        limits=LIMITS,
    )


def plan(schema: ModelSchema | None = None) -> SourceExportPlan:
    schema = model() if schema is None else schema
    return SourceExportPlan(
        schema,
        "c" * 64,
        (0, 1),
        tuple(TensorOwner(entry.tensor.name, 0) for entry in schema.tensors if entry.alias_of is None),
        LIMITS,
    )


def data_for(schema: ModelSchema) -> dict[str, bytearray]:
    # Includes BF16 +0/-0/Inf/distinct NaN payloads, preserved as bytes.
    pattern = bytes.fromhex("00000080807f817fc17f")
    return {
        entry.tensor.name: bytearray((pattern * (entry.tensor.nbytes // len(pattern) + 1))[: entry.tensor.nbytes])
        for entry in schema.tensors
        if entry.alias_of is None
    }


def tiles(source_plan: SourceExportPlan, request: ExportRequest, data: dict[str, bytearray]):
    owners = {owner.name: owner.rank for owner in source_plan.owners}
    for spec in source_plan.schema.iter_chunks():
        size = BUDGET.tile_bytes // spec.tensor.element_size * spec.tensor.element_size
        for offset in range(0, spec.byte_length, size):
            start = spec.byte_offset + offset
            payload = bytes(data[spec.tensor.name][start : start + min(size, spec.byte_length - offset)])
            yield CanonicalTile(
                request.request_id, source_plan.plan_id, owners[spec.tensor.name], spec, offset, payload
            )


def store_at(path: Path, **kwargs: int) -> DiskSnapshotStore:
    return DiskSnapshotStore(
        path, max_bytes=kwargs.get("max_bytes", 20000), max_generations=kwargs.get("max_generations", 4), limits=LIMITS
    )


def capture(store: DiskSnapshotStore, request: ExportRequest = REQUEST):
    source_plan = plan()
    data = data_for(source_plan.schema)
    if request.version:
        data["weight"][0] ^= 1
    with store.capture(source_plan, request, BUDGET) as candidate:
        for tile in tiles(source_plan, request, data):
            candidate.write_tile(tile)
        return candidate.finalize(source_plan.expected_receipts(request)), data


def test_capture_owns_bytes_and_frozen_lease_outlives_store(tmp_path: Path) -> None:
    source_plan = plan()
    data = data_for(source_plan.schema)
    original = {name: bytes(value) for name, value in data.items()}
    with store_at(tmp_path / "snapshots") as store:
        with store.capture(source_plan, REQUEST, BUDGET) as candidate:
            assert not (candidate.path / "metadata.json").exists()
            assert not (candidate.path / "sealed.json").exists()
            for tile in tiles(source_plan, REQUEST, data):
                candidate.write_tile(tile)
            frozen = candidate.finalize(source_plan.expected_receipts(REQUEST))
        for value in data.values():
            value[:] = bytes(len(value))
        identity = frozen.snapshot.identity
        assert json.loads((candidate.path / "metadata.json").read_text())["identity"] == asdict(identity)
    for spec in source_plan.schema.iter_chunks():
        assert (
            frozen.read_chunk(spec)
            == original[spec.tensor.name][spec.byte_offset : spec.byte_offset + spec.byte_length]
        )
    alias = next(entry.tensor for entry in source_plan.schema.tensors if entry.alias_of)
    assert frozen.read_chunk(ChunkSpec(source_plan.schema.schema_id, alias, 0, 2)) == original["weight"][:2]
    reopened, reader = open_snapshot(frozen.path, expected_identity=identity, limits=LIMITS)
    assert reopened.identity == identity
    reader.close()
    stale_handle = frozen.snapshot
    frozen.close()
    frozen.close()
    with pytest.raises(DeltaCodecError, match="closed"):
        frozen.snapshot
    with pytest.raises(DeltaCodecError, match="closed"):
        stale_handle.read_chunk(next(source_plan.schema.iter_chunks()))
    assert (frozen.path / "sealed.json").exists()


@pytest.mark.parametrize(
    "case", ["duplicate", "skip", "offset", "request", "plan", "owner", "schema", "oversize", "alias"]
)
def test_invalid_tile_poisons_capture_without_damaging_base(tmp_path: Path, case: str) -> None:
    source_plan = plan()
    good = list(tiles(source_plan, REQUEST, data_for(source_plan.schema)))
    with store_at(tmp_path / "snapshots") as store:
        base, _ = capture(store)
        first = next(source_plan.schema.iter_chunks())
        previous = base.read_chunk(first)
        with store.capture(source_plan, REQUEST, BUDGET) as candidate:
            bad = good[0]
            if case == "duplicate":
                candidate.write_tile(good[0])
            elif case == "skip":
                bad = good[1]
            elif case == "offset":
                bad = replace(good[1], byte_offset=2)
            elif case in ("request", "plan"):
                bad = replace(bad, **{case + "_id": "e" * 64})
            elif case == "owner":
                bad = replace(bad, rank=1)
            elif case == "schema":
                bad = replace(bad, spec=replace(bad.spec, schema_id="e" * 64))
            elif case == "oversize":
                candidate.write_tile(good[0])
                bad = replace(good[1], data=bytes(16))
            else:
                alias = next(entry.tensor for entry in source_plan.schema.tensors if entry.alias_of)
                bad = replace(bad, spec=ChunkSpec(source_plan.schema.schema_id, alias, 0, 8))
            with pytest.raises(DeltaCodecError):
                candidate.write_tile(bad)
            with pytest.raises(DeltaCodecError, match="not active"):
                candidate.write_tile(good[0])
            with pytest.raises(DeltaCodecError, match="not active"):
                candidate.finalize(source_plan.expected_receipts(REQUEST))
        assert not candidate.path.exists()
        assert base.read_chunk(first) == previous
        base.close()


def test_incomplete_finalize_cannot_be_resumed(tmp_path: Path) -> None:
    source_plan = plan()
    with store_at(tmp_path / "snapshots") as store:
        candidate = store.capture(source_plan, REQUEST, BUDGET)
        with pytest.raises(DeltaCodecError, match="incomplete"):
            candidate.finalize(source_plan.expected_receipts(REQUEST))
        with pytest.raises(DeltaCodecError, match="not active"):
            candidate.write_tile(next(tiles(source_plan, REQUEST, data_for(source_plan.schema))))
        candidate.abort()
        candidate.abort()
        assert store._usage() == (0, 0)


@pytest.mark.parametrize("case", ["missing", "duplicate", "extra", "step", "plan", "bytes", "chunks"])
def test_receipts_require_every_participant_and_exact_identity(case: str) -> None:
    source_plan = plan()
    receipts = list(source_plan.expected_receipts(REQUEST))
    assert receipts[1].chunks == receipts[1].nbytes == 0
    if case == "missing":
        receipts.pop()
    elif case == "duplicate":
        receipts.append(receipts[0])
    elif case == "extra":
        receipts.append(replace(receipts[1], rank=2))
    elif case == "step":
        receipts[0] = replace(receipts[0], request_id=replace(REQUEST, source_step=1).request_id)
    elif case == "plan":
        receipts[0] = replace(receipts[0], plan_id="d" * 64)
    elif case == "bytes":
        receipts[0] = replace(receipts[0], nbytes=receipts[0].nbytes - 1)
    else:
        receipts[0] = replace(receipts[0], chunks=receipts[0].chunks - 1)
    with pytest.raises(DeltaCodecError):
        validate_receipts(source_plan, REQUEST, receipts)


def test_bad_receipts_prevent_seal_and_poison_capture(tmp_path: Path) -> None:
    source_plan = plan()
    with store_at(tmp_path / "snapshots") as store:
        with store.capture(source_plan, REQUEST, BUDGET) as candidate:
            for tile in tiles(source_plan, REQUEST, data_for(source_plan.schema)):
                candidate.write_tile(tile)
            with pytest.raises(DeltaCodecError, match="missing"):
                candidate.finalize(source_plan.expected_receipts(REQUEST)[:1])
            assert not (candidate.path / "sealed.json").exists()
            with pytest.raises(DeltaCodecError, match="not active"):
                candidate.finalize(source_plan.expected_receipts(REQUEST))


@pytest.mark.parametrize("empty", [True, False])
def test_empty_directories_and_zero_owners_finalize(tmp_path: Path, empty: bool) -> None:
    schema = ModelSchema(
        "a" * 64,
        "b" * 64,
        ()
        if empty
        else (
            TensorEntry(TensorSpec("empty", "float32", (0,)), "buffer"),
            TensorEntry(TensorSpec("alias", "float32", (0,)), "buffer", "empty"),
        ),
        32,
        LIMITS,
    )
    source_plan = plan(schema)
    with store_at(tmp_path / "snapshots") as store:
        with store.capture(source_plan, REQUEST, BUDGET) as candidate:
            frozen = candidate.finalize(source_plan.expected_receipts(REQUEST))
        assert frozen.snapshot.schema == schema
        assert (frozen.path / "weights.bin").stat().st_size == 0
        frozen.close()


def test_cpu_budget_includes_chunk_verification_before_creating_files(tmp_path: Path) -> None:
    with store_at(tmp_path / "snapshots") as store:
        with pytest.raises(DeltaCodecError, match="CPU working set"):
            store.capture(plan(), REQUEST, replace(BUDGET, max_cpu_bytes=BUDGET.max_cpu_bytes - 1))
        assert store._usage() == (0, 0)
        with store.capture(plan(), REQUEST, BUDGET):
            pass
        assert store._usage() == (0, 0)


@pytest.mark.parametrize("kind", ["bytes", "generations"])
def test_capture_quota_accounts_for_retained_base_and_restart(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "snapshots"
    with store_at(path) as store:
        frozen, _ = capture(store)
        retained, count = store._usage()
        frozen.close()
    with store_at(
        path, max_bytes=retained if kind == "bytes" else 20000, max_generations=1 if kind == "generations" else 4
    ) as store:
        with pytest.raises(DeltaCodecError):
            store.capture(plan(), replace(REQUEST, version=1), BUDGET)
        assert store._usage() == (retained, count)


@pytest.mark.parametrize("case", ["write", "sync", "metadata", "seal_after_link", "corrupt", "truncate"])
def test_capture_io_failure_aborts_only_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    source_plan = plan()
    with store_at(tmp_path / "snapshots") as store:
        base, _ = capture(store)
        saved = base.snapshot.identity
        candidate = store.capture(source_plan, replace(REQUEST, version=1), BUDGET)
        install, write, sync = capture_io.install_file, capture_io.write_all, os.fsync
        if case == "write":

            def failed_write(fd, data):
                os.write(fd, data[:1])
                raise OSError("partial write")

            monkeypatch.setattr(capture_io, "write_all", failed_write)
        elif case in ("metadata", "seal_after_link"):

            def failed_install(directory, name, data):
                if name == "metadata.json" and case == "metadata":
                    raise OSError("metadata failure")
                result = install(directory, name, data)
                if name == "sealed.json":
                    raise OSError("failure after seal link")
                return result

            monkeypatch.setattr(capture_io, "install_file", failed_install)
        with pytest.raises((OSError, DeltaCodecError)):
            for tile in tiles(source_plan, candidate.request, data_for(source_plan.schema)):
                candidate.write_tile(tile)
            if case == "corrupt":
                os.pwrite(candidate._writer, b"\xff", 0)
            elif case == "truncate":
                os.ftruncate(candidate._writer, 1)
            elif case == "sync":

                def failed_sync(fd):
                    raise OSError("fsync failure")

                monkeypatch.setattr(os, "fsync", failed_sync)
            candidate.finalize(source_plan.expected_receipts(candidate.request))
        monkeypatch.setattr(capture_io, "write_all", write)
        monkeypatch.setattr(capture_io, "install_file", install)
        monkeypatch.setattr(os, "fsync", sync)
        if case == "seal_after_link":
            assert (candidate.path / "sealed.json").exists()
        with pytest.raises(DeltaCodecError, match="not active"):
            candidate.finalize(source_plan.expected_receipts(candidate.request))
        candidate.abort()
        assert not candidate.path.exists()
        assert store._usage()[1] == 1
        reopened, reader = open_snapshot(base.path, expected_identity=saved, limits=LIMITS)
        assert reopened.identity == saved
        reader.close()
        base.close()


def test_store_close_aborts_unfinished_capture(tmp_path: Path) -> None:
    store = store_at(tmp_path / "snapshots")
    candidate = store.capture(plan(), REQUEST, BUDGET)
    with pytest.raises(DeltaCodecError, match="unfinished"):
        store.capture(plan(), REQUEST, BUDGET)
    store.close()
    assert not candidate.path.exists()
    candidate.abort()


def test_abort_retries_parent_sync_after_directory_was_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with store_at(tmp_path / "snapshots") as store:
        base, _ = capture(store)
        identity = base.snapshot.identity
        candidate = store.capture(plan(), REQUEST, BUDGET)
        sync = os.fsync
        failed = False

        def fail_after_removal(fd: int) -> None:
            nonlocal failed
            if fd == store._fd and not candidate.path.exists() and not failed:
                failed = True
                raise OSError("removed directory parent sync")
            sync(fd)

        monkeypatch.setattr(os, "fsync", fail_after_removal)
        with pytest.raises(OSError, match="parent sync"):
            candidate.abort()
        assert not candidate.path.exists()
        candidate.abort()
        with store.capture(plan(), REQUEST, BUDGET):
            pass
        assert store._usage()[1] == 1
        reopened, reader = open_snapshot(base.path, expected_identity=identity, limits=LIMITS)
        assert reopened.identity == identity
        reader.close()
        base.close()


def test_captured_full_and_delta_reconstruct_bitwise(tmp_path: Path) -> None:
    with (
        store_at(tmp_path / "source") as source,
        store_at(tmp_path / "restored") as restored,
        PosixArtifactStore(tmp_path / "artifacts", writable=True, limits=LIMITS) as artifacts,
    ):
        v0, data0 = capture(source)
        v1, data1 = capture(source, replace(REQUEST, version=1, source_step=3))
        assert data0 != data1
        base = None
        readers = []
        for frozen, expected in ((v0, data0), (v1, data1)):
            request = frozen.request
            manifest = build_manifest(
                frozen.snapshot.schema,
                frozen,
                artifacts,
                stream_id=request.stream_id,
                run_epoch=request.run_epoch,
                version=request.version,
                writer_fence=1,
                source_step=request.source_step,
                exporter_revision=request.exporter_revision,
                base=None if request.version == 0 else v0.snapshot,
                limits=LIMITS,
            )
            assert manifest.target == frozen.snapshot.identity
            assert manifest.kind == ("FULL" if request.version == 0 else "DELTA")
            encoded = manifest.to_bytes(LIMITS)
            stage = restored.staging()
            base = reconstruct(
                encoded,
                artifacts,
                stage,
                expected_manifest_id=hashlib.sha256(encoded).hexdigest(),
                base=base,
                limits=LIMITS,
            )
            readers.append(stage.reader())
            for spec in manifest.schema.iter_chunks():
                assert base.read_chunk(spec) == bytes(
                    expected[spec.tensor.name][spec.byte_offset : spec.byte_offset + spec.byte_length]
                )
        for reader in readers:
            reader.close()
        v0.close()
        v1.close()


@pytest.mark.parametrize("case", ["missing", "duplicate", "alias", "unknown_rank"])
def test_plan_requires_exact_owner_directory(case: str) -> None:
    source_plan = plan()
    owners = source_plan.owners
    if case == "missing":
        owners = owners[:-1]
    elif case == "duplicate":
        owners += owners[:1]
    elif case == "alias":
        owners += (TensorOwner("tied", 0),)
    else:
        owners = (replace(owners[0], rank=2), *owners[1:])
    with pytest.raises(DeltaCodecError):
        replace(source_plan, owners=owners)


def test_source_layout_and_request_do_not_change_portable_schema() -> None:
    source_plan = plan()
    alternate = replace(source_plan, source_layout_id="d" * 64)
    assert alternate.schema.schema_id == source_plan.schema.schema_id
    assert alternate.plan_id != source_plan.plan_id
    assert replace(REQUEST, source_step=1).request_id != REQUEST.request_id
