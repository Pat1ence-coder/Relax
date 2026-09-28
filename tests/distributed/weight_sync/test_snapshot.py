# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Model roots, strict paged metadata, and transactional staging
conformance."""

import hashlib
import json
import random
import struct
import tempfile
import tracemalloc
from dataclasses import asdict, replace
from typing import Any

import pytest

from relax.distributed.weight_sync import (
    CanonicalChunk,
    ChunkRecord,
    ChunkSpec,
    Codec,
    CodecLimits,
    DeltaCodecError,
    IndexPage,
    Manifest,
    ModelRoot,
    ModelSchema,
    PageRef,
    SnapshotIdentity,
    SnapshotLimits,
    TensorEntry,
    TensorSpec,
    VerifiedSnapshot,
    build_manifest,
    reconstruct,
    verify_snapshot,
)


LIMITS = SnapshotLimits(codec=CodecLimits(max_chunk_bytes=1024), max_chunks_per_page=2)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wire(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def model() -> ModelSchema:
    return ModelSchema(
        "a" * 64,
        "b" * 64,
        (
            TensorEntry(TensorSpec("z.tied", "bfloat16", (1031,)), alias_of="a.weight"),
            TensorEntry(TensorSpec("empty", "float32", (0, 7))),
            TensorEntry(TensorSpec("c.scale", "float32", (257,))),
            TensorEntry(TensorSpec("a.weight", "bfloat16", (1031,))),
            TensorEntry(TensorSpec("b.counter", "uint8", (1031,)), "buffer"),
        ),
        1024,
        LIMITS,
    )


class BytesReader:
    def __init__(self, schema: ModelSchema, *, fill: int = 0) -> None:
        self.data = {
            entry.tensor.name: bytes([fill]) * entry.tensor.nbytes
            for entry in schema.tensors
            if entry.alias_of is None
        }
        self.reads = 0

    def read_chunk(self, spec: ChunkSpec) -> bytes:
        self.reads += 1
        return self.data[spec.tensor.name][spec.byte_offset : spec.byte_offset + spec.byte_length]


class Artifacts:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.payload_reads = 0
        self.index_reads = 0

    def write_payload(self, record: ChunkRecord, payload: bytes) -> None:
        assert record.object_key is not None
        self.objects[record.object_key] = payload

    def write_index(self, ref: PageRef, data: bytes) -> None:
        self.objects[ref.object_key] = data

    def read_payload(self, record: ChunkRecord) -> bytes:
        self.payload_reads += 1
        return self.objects[record.object_key]

    def read_index(self, ref: PageRef) -> bytes:
        self.index_reads += 1
        return self.objects[ref.object_key]


class Staging:
    def __init__(self, *, fail: str = "") -> None:
        self.chunks: dict[tuple[str, int], bytes] = {}
        self.events: list[str] = []
        self.fail = fail
        self.sealed = False
        self.writes = 0

    def begin(self, schema: ModelSchema, identity: SnapshotIdentity) -> None:
        self.events.append("begin")
        self.schema, self.identity = schema, identity
        if self.fail == "begin":
            raise OSError("begin failure")

    def write_chunk(self, chunk: CanonicalChunk) -> None:
        self.writes += 1
        self.events.append("write")
        if self.fail == "write" and self.writes == 2:
            raise OSError("write failure")
        data = chunk.data
        if self.fail == "corrupt" and self.writes == 2:
            data = bytes([data[0] ^ 1]) + data[1:]
        self.chunks[chunk.spec.tensor.name, chunk.spec.byte_offset] = data

    def reader(self) -> "Staging":
        self.events.append("reader")
        return self

    def read_chunk(self, spec: ChunkSpec) -> bytes:
        self.events.append("readback")
        if self.fail == "read":
            raise OSError("readback failure")
        return self.chunks[spec.tensor.name, spec.byte_offset]

    def seal(self) -> None:
        self.events.append("seal")
        if self.fail == "seal":
            raise OSError("seal failure")
        self.sealed = True

    def abort(self) -> None:
        self.events.append("abort")
        self.chunks.clear()
        self.sealed = False


def build(
    schema: ModelSchema,
    reader: BytesReader | Staging,
    store: Artifacts,
    *,
    version: int = 0,
    base: VerifiedSnapshot | None = None,
    **kwargs: Any,
) -> Manifest:
    return build_manifest(
        schema,
        reader,
        store,
        stream_id="test",
        run_epoch="epoch-1",
        version=version,
        writer_fence=1,
        source_step=version,
        exporter_revision="test-v1",
        base=base,
        limits=kwargs.pop("limits", LIMITS),
        **kwargs,
    )


def restore(
    manifest: Manifest, store: Artifacts, staging: Staging, *, base: VerifiedSnapshot | None = None
) -> VerifiedSnapshot:
    data = manifest.to_bytes(LIMITS)
    return reconstruct(data, store, staging, expected_manifest_id=digest(data), base=base, limits=LIMITS)


def independent_root(schema_id: str, directory_hash: str, chunks: list[tuple[ChunkSpec, bytes]]) -> str:
    """List-based oracle, deliberately independent of the streaming
    accumulator."""

    def hashed(tag: bytes, *fields: bytes) -> bytes:
        message = b"DWS1"
        for value in (tag, *fields):
            message += struct.pack("<Q", len(value)) + value
        return hashlib.sha256(message).digest()

    leaves = [
        hashed(
            b"chunk-leaf",
            bytes.fromhex(schema_id),
            spec.tensor.name.encode(),
            struct.pack("<Q", spec.byte_offset),
            struct.pack("<Q", spec.byte_length),
            hashlib.sha256(data).digest(),
        )
        for spec, data in chunks
    ]
    while len(leaves) > 1:
        leaves = [
            hashed(b"merkle-node", leaves[i], leaves[i + 1])
            if i + 1 < len(leaves)
            else hashed(b"merkle-odd", leaves[i])
            for i in range(0, len(leaves), 2)
        ]
    tree = leaves[0] if leaves else hashed(b"merkle-empty")
    return hashed(
        b"target-root", bytes.fromhex(schema_id), bytes.fromhex(directory_hash), struct.pack("<Q", len(chunks)), tree
    ).hex()


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 15, 16, 17, 31, 32, 33, 100, 257])
def test_root_matches_independent_tree_at_every_boundary(count: int) -> None:
    root = ModelRoot("a" * 64, "b" * 64)
    chunks = []
    for index in range(count):
        spec = ChunkSpec("a" * 64, TensorSpec("weight", "uint8", (max(count, 1),)), index, 1)
        data = bytes([index % 256])
        chunks.append((spec, data))
        root.add(spec, digest(data))
    assert root.hexdigest() == independent_root("a" * 64, "b" * 64, chunks)
    assert root.hexdigest() == root.hexdigest()  # Finalization does not consume state.
    assert len(root._levels) <= max(1, count.bit_length())


@pytest.mark.parametrize(
    "count,expected",
    [
        (0, "06036f287829ac142953fa89e8a7ff6cad070e355756bdac574a785d7fbe81f8"),
        (1, "6c5d04fbc9d88ec63c498d511879c94a868226ae05db8d264afdf13e398b3627"),
        (3, "bf4e5612463be58304fe1a0770ef75271e40e7624820a7d4879496d0ad6c4550"),
    ],
)
def test_root_fixed_format_vectors(count: int, expected: str) -> None:
    root = ModelRoot("a" * 64, "b" * 64)
    for index in range(count):
        spec = ChunkSpec("a" * 64, TensorSpec("weight", "uint8", (max(count, 1),)), index, 1)
        root.add(spec, digest(bytes([index])))
    assert root.hexdigest() == expected


def test_schema_complete_directory_and_deterministic_identity() -> None:
    schema = model()
    assert [entry.tensor.name for entry in schema.tensors] == ["a.weight", "b.counter", "c.scale", "empty", "z.tied"]
    assert schema.canonical_nbytes == 2062 + 1031 + 1028
    assert schema.logical_nbytes == schema.canonical_nbytes + 2062
    assert schema.chunk_count == 7
    assert len(list(schema.iter_chunks())) == 7
    assert ModelSchema.from_bytes(schema.to_bytes(), LIMITS) == schema
    assert replace(schema, tensors=tuple(reversed(schema.tensors))).schema_id == schema.schema_id
    assert digest(wire(schema.to_dict())) == schema.schema_id
    assert digest(wire([entry.to_dict() for entry in schema.tensors])) == schema.directory_hash
    for name in ("empty", "z.tied"):
        changed = replace(schema, tensors=tuple(entry for entry in schema.tensors if entry.tensor.name != name))
        assert changed.schema_id != schema.schema_id
        before, after = Artifacts(), Artifacts()
        assert (
            build(changed, BytesReader(changed), after).target.target_root
            != build(schema, BytesReader(schema), before).target.target_root
        )
    assert (
        replace(schema, tensors=tuple(replace(entry, kind="parameter") for entry in schema.tensors)).schema_id
        != schema.schema_id
    )
    assert replace(schema, converter_semantics_id="c" * 64).schema_id != schema.schema_id
    assert replace(schema, logical_config_hash="c" * 64).schema_id != schema.schema_id
    assert replace(schema, chunk_bytes=512).schema_id != schema.schema_id


@pytest.mark.parametrize(
    "case", ["missing", "self", "cycle", "chain", "shape", "dtype", "duplicate", "kind", "alignment"]
)
def test_schema_rejects_ambiguous_directory(case: str) -> None:
    owner = TensorEntry(TensorSpec("a", "float32", (2,)))
    alias = TensorEntry(TensorSpec("b", "float32", (2,)), alias_of="a")
    entries, chunk_bytes = (owner, alias), 1024
    if case == "missing":
        entries = (replace(alias, alias_of="gone"),)
    if case == "self":
        entries = (replace(owner, alias_of="a"),)
    if case == "cycle":
        entries = (replace(owner, alias_of="b"), alias)
    if case == "chain":
        entries = (owner, alias, TensorEntry(TensorSpec("c", "float32", (2,)), alias_of="b"))
    if case == "shape":
        entries = (owner, replace(alias, tensor=TensorSpec("b", "float32", (1, 2))))
    if case == "dtype":
        entries = (owner, replace(alias, tensor=TensorSpec("b", "int32", (2,))))
    if case == "duplicate":
        entries = (owner, owner)
    if case == "kind":
        entries = (replace(owner, kind="unknown"),)
    if case == "alignment":
        chunk_bytes = 7
    with pytest.raises(DeltaCodecError):
        ModelSchema("a" * 64, "b" * 64, entries, chunk_bytes, LIMITS)


def test_full_delta_anchor_and_repeated_restore_are_bitwise_equal() -> None:
    schema, store = model(), Artifacts()
    initial = BytesReader(schema)
    full = build(schema, initial, store)
    base_stage = Staging()
    base = restore(full, store, base_stage)
    target = BytesReader(schema)
    target.data["a.weight"] = b"\x01\x80" + initial.data["a.weight"][2:]
    target.data["c.scale"] = b"\x80\x7f" * 514
    delta = build(schema, target, store, version=1, base=base)
    anchor = build(schema, target, store, version=1)
    assert delta.target == anchor.target
    assert delta.manifest_id(LIMITS) != anchor.manifest_id(LIMITS)
    assert delta.target.target_root == independent_root(
        schema.schema_id, schema.directory_hash, [(spec, target.read_chunk(spec)) for spec in schema.iter_chunks()]
    )
    codecs = {
        record.descriptor.codec
        for page in delta.pages
        for record in IndexPage.from_bytes(store.read_index(page), page, LIMITS).records
    }
    assert {Codec.COPY_BASE, Codec.SPARSE_REPLACE_V1, Codec.RAW_V1} <= codecs
    for _ in range(2):
        stage = Staging()
        rebuilt = restore(delta, store, stage, base=base)
        assert stage.events[-1] == "seal"
        assert stage.sealed
        for spec in schema.iter_chunks():
            assert rebuilt.read_chunk(spec) == target.read_chunk(spec)
            assert base.read_chunk(spec) == initial.read_chunk(spec)
    restored_full = restore(anchor, store, Staging())
    assert restored_full.identity == delta.target


@pytest.mark.parametrize("empty_directory", [False, True])
def test_empty_models_retain_catalog_without_pages(empty_directory: bool) -> None:
    entries = () if empty_directory else (TensorEntry(TensorSpec("empty", "bfloat16", (0,))),)
    schema = ModelSchema("a" * 64, "b" * 64, entries, 1024, LIMITS)
    store = Artifacts()
    manifest = build(schema, BytesReader(schema), store)
    assert manifest.pages == () and store.objects == {}
    stage = Staging()
    assert restore(manifest, store, stage).schema == schema
    assert stage.events == ["begin", "reader", "seal"]


def test_wrong_whole_base_root_is_rejected_even_if_referenced_chunk_matches() -> None:
    schema, store = model(), Artifacts()
    reader = BytesReader(schema)
    initial = build(schema, reader, store)
    wrong = BytesReader(schema)
    wrong.data["c.scale"] = bytes([1]) * len(wrong.data["c.scale"])
    assert wrong.read_chunk(next(schema.iter_chunks())) == reader.read_chunk(next(schema.iter_chunks()))
    with pytest.raises(DeltaCodecError, match="complete snapshot root"):
        verify_snapshot(schema, initial.target, wrong, limits=LIMITS)
    # A real but different verified snapshot is also incompatible with the exact base identity.
    wrong_manifest = build(schema, wrong, store)
    wrong_base = verify_snapshot(schema, wrong_manifest.target, wrong, limits=LIMITS)
    base = verify_snapshot(schema, initial.target, reader, limits=LIMITS)
    delta = build(schema, reader, store, version=1, base=base)
    with pytest.raises(DeltaCodecError, match="complete manifest base identity"):
        restore(delta, store, Staging(), base=wrong_base)
    other_epoch = verify_snapshot(schema, replace(initial.target, run_epoch="epoch-2"), reader, limits=LIMITS)
    with pytest.raises(DeltaCodecError, match="complete manifest base identity"):
        restore(delta, store, Staging(), base=other_epoch)
    with pytest.raises(DeltaCodecError):
        VerifiedSnapshot(schema, initial.target, wrong, _token=object())


def test_missing_base_fails_but_all_raw_delta_needs_no_base() -> None:
    schema, store = model(), Artifacts()
    reader = BytesReader(schema)
    initial = build(schema, reader, store)
    base = verify_snapshot(schema, initial.target, reader, limits=LIMITS)
    delta = build(schema, reader, store, version=1, base=base)
    stage = Staging()
    with pytest.raises(DeltaCodecError, match="requires a verified delta base"):
        restore(delta, store, stage)
    assert stage.events[-1] == "abort" and not stage.chunks
    raw = build(schema, BytesReader(schema, fill=255), store, version=1, base=base)
    assert restore(raw, store, Staging()).identity == raw.target


@pytest.mark.parametrize("fail", ["begin", "write", "corrupt", "read", "seal"])
def test_staging_failures_abort_without_exposing_partial_target(fail: str) -> None:
    schema, store = model(), Artifacts()
    manifest = build(schema, BytesReader(schema), store)
    stage = Staging(fail=fail)
    with pytest.raises((OSError, DeltaCodecError)):
        restore(manifest, store, stage)
    assert stage.events[-1] == "abort"
    assert not stage.sealed and not stage.chunks


def test_manifest_hash_must_come_from_trusted_caller() -> None:
    schema, store = model(), Artifacts()
    manifest = build(schema, BytesReader(schema), store)
    stage = Staging()
    with pytest.raises(DeltaCodecError, match="trusted identity"):
        reconstruct(manifest.to_bytes(LIMITS), store, stage, expected_manifest_id="0" * 64, limits=LIMITS)
    assert stage.events == [] and store.index_reads == 0


def repage(manifest: Manifest, store: Artifacts, page_number: int, records: tuple[ChunkRecord, ...]) -> Manifest:
    ref = manifest.pages[page_number]
    data = IndexPage(ref.first_chunk, records).to_bytes(LIMITS)
    changed = replace(ref, page_hash=digest(data), byte_length=len(data), object_key=f"indexes/{digest(data)}.json")
    store.write_index(changed, data)
    pages = list(manifest.pages)
    pages[page_number] = changed
    return replace(manifest, pages=tuple(pages))


@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "reorder",
        "wrong_offset",
        "wrong_schema",
        "version",
        "base_version",
        "full_copy",
        "root",
        "payload",
        "page",
        "last_hash",
    ],
)
def test_corrupt_model_stream_aborts(case: str) -> None:
    schema, store = model(), Artifacts()
    reader = BytesReader(schema)
    full = build(schema, reader, store)
    base = verify_snapshot(schema, full.target, reader, limits=LIMITS)
    manifest = build(schema, reader, store, version=2, base=base) if case in ("base_version", "full_copy") else full
    records = list(IndexPage.from_bytes(store.read_index(manifest.pages[0]), manifest.pages[0], LIMITS).records)
    first = records[0]
    if case == "duplicate":
        records[1] = first
    if case == "reorder":
        records.reverse()
    if case == "wrong_offset":
        records[0] = replace(
            first, descriptor=replace(first.descriptor, spec=replace(first.descriptor.spec, byte_offset=2))
        )
    if case == "wrong_schema":
        records[0] = replace(
            first, descriptor=replace(first.descriptor, spec=replace(first.descriptor.spec, schema_id="f" * 64))
        )
    if case == "version":
        records[0] = replace(first, descriptor=replace(first.descriptor, target_version=1))
    if case == "base_version":
        records[0] = replace(first, descriptor=replace(first.descriptor, base_version=1))
    if case in ("duplicate", "reorder", "wrong_offset", "wrong_schema", "version", "base_version"):
        manifest = repage(manifest, store, 0, tuple(records))
    if case == "full_copy":
        manifest = replace(manifest, kind="FULL", base=None)
    if case == "root":
        manifest = replace(manifest, target=replace(manifest.target, target_root="0" * 64))
    if case == "payload":
        store.objects[first.object_key] = b"corrupt"
    if case == "page":
        store.objects[manifest.pages[0].object_key] += b" "
    if case == "last_hash":
        last_ref = manifest.pages[-1]
        last = list(IndexPage.from_bytes(store.read_index(last_ref), last_ref, LIMITS).records)
        last[-1] = replace(last[-1], descriptor=replace(last[-1].descriptor, target_hash="0" * 64))
        manifest = repage(manifest, store, len(manifest.pages) - 1, tuple(last))
    stage = Staging()
    with pytest.raises(DeltaCodecError):
        restore(manifest, store, stage, base=base if case == "base_version" else None)
    assert stage.events[-1] == "abort" and not stage.chunks and not stage.sealed
    if case == "last_hash":
        assert stage.writes == schema.chunk_count - 1


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "missing",
        "feature",
        "bool",
        "float",
        "duplicate_key",
        "format",
        "page_gap",
        "page_missing",
        "page_duplicate",
        "page_order",
        "base",
        "path",
        "counts",
        "shape",
        "name_order",
    ],
)
def test_strict_manifest_metadata_rejects_invalid_forms(case: str) -> None:
    schema, store = model(), Artifacts()
    manifest = build(schema, BytesReader(schema), store)
    value = json.loads(manifest.to_bytes(LIMITS))
    if case == "unknown":
        value["unknown"] = 1
    if case == "missing":
        del value["target"]
    if case == "feature":
        value["required_features"].append("future-v2")
    if case == "bool":
        value["target"]["version"] = True
    if case == "float":
        value["source_step"] = 1.5
    if case == "format":
        value["format_version"] = 2
    if case == "page_gap":
        value["pages"][1]["first_chunk"] += 1
    if case == "page_missing":
        value["pages"].pop()
    if case == "page_duplicate":
        value["pages"].append(value["pages"][-1])
    if case == "page_order":
        value["pages"].reverse()
    if case == "base":
        value["base"] = asdict(manifest.target)
    if case == "path":
        value["pages"][0]["object_key"] = "../escape"
    if case == "counts":
        value["pages"][0]["chunk_count"] = True
    if case == "shape":
        value["schema"]["tensors"][0]["shape"] = [-1]
    if case == "name_order":
        value["schema"]["tensors"].reverse()
    data = wire(value)
    if case == "duplicate_key":
        data = data[:-1] + b',"kind":"FULL"}'
    with pytest.raises(DeltaCodecError):
        Manifest.from_bytes(data, expected_manifest_id=digest(data), limits=LIMITS)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tensors", 1),
        ("max_chunks", 1),
        ("max_pages", 1),
        ("max_model_bytes", 1),
        ("max_directory_bytes", 64),
        ("max_manifest_bytes", 64),
        ("max_index_bytes", 64),
        ("max_index_page_bytes", 64),
        ("max_chunks_per_page", 1),
    ],
)
def test_receiver_enforces_its_own_limits_before_io(field: str, value: int) -> None:
    schema, store = model(), Artifacts()
    manifest = build(schema, BytesReader(schema), store)
    data = manifest.to_bytes(LIMITS)
    stage = Staging()
    with pytest.raises(DeltaCodecError):
        reconstruct(data, store, stage, expected_manifest_id=digest(data), limits=replace(LIMITS, **{field: value}))
    assert store.index_reads == store.payload_reads == 0 and stage.events == []


def test_page_byte_budget_rolls_before_count_budget_and_fails_for_single_record() -> None:
    schema, store = model(), Artifacts()
    limits = replace(LIMITS, max_index_page_bytes=1100, max_chunks_per_page=100)
    manifest = build(schema, BytesReader(schema), store, limits=limits)
    assert len(manifest.pages) == schema.chunk_count
    assert all(ref.byte_length <= 1100 and ref.chunk_count == 1 for ref in manifest.pages)
    with pytest.raises(DeltaCodecError):
        build(schema, BytesReader(schema), Artifacts(), limits=replace(limits, max_index_page_bytes=64))


def test_one_hundred_synthetic_model_versions_with_periodic_anchors() -> None:
    schema, reader, rng = model(), BytesReader(model()), random.Random(374)
    store = Artifacts()
    manifest = build(schema, reader, store)
    base = restore(manifest, store, Staging())
    for version in range(1, 101):
        name = rng.choice(["a.weight", "b.counter", "c.scale"])
        value = bytearray(reader.data[name])
        for _ in range(version % 19):
            value[rng.randrange(len(value))] ^= rng.randrange(1, 256)
        reader.data[name] = bytes(value)
        store = Artifacts()
        previous = None if version % 10 == 0 else base
        manifest = build(schema, reader, store, version=version, base=previous)
        base = restore(manifest, store, Staging(), base=previous)
        for spec in schema.iter_chunks():
            assert base.read_chunk(spec) == reader.read_chunk(spec)


@pytest.mark.parametrize("case", ["version", "count", "first", "unknown", "descriptor", "key", "float", "duplicate"])
def test_index_page_parser_rejects_malformed_fields(case: str) -> None:
    schema, store = model(), Artifacts()
    manifest = build(schema, BytesReader(schema), store)
    ref = manifest.pages[0]
    value = json.loads(store.objects[ref.object_key])
    if case == "version":
        value["format_version"] = True
    if case == "count":
        value["records"].pop()
    if case == "first":
        value["first_chunk"] = 1
    if case == "unknown":
        value["records"][0]["extra"] = None
    if case == "descriptor":
        del value["records"][0]["descriptor"]["target_hash"]
    if case == "key":
        value["records"][0]["object_key"] = "../payload"
    if case == "float":
        value["first_chunk"] = 0.0
    data = wire(value)
    if case == "duplicate":
        data = data[:-1] + b',"first_chunk":0}'
    ref = replace(ref, page_hash=digest(data), byte_length=len(data), object_key=f"indexes/{digest(data)}.json")
    with pytest.raises(DeltaCodecError):
        IndexPage.from_bytes(data, ref, LIMITS)


def test_artifact_failures_propagate_and_abort_private_staging() -> None:
    schema, store = model(), Artifacts()
    manifest = build(schema, BytesReader(schema), store)
    del store.objects[manifest.pages[-1].object_key]
    stage = Staging()
    with pytest.raises(KeyError):
        restore(manifest, store, stage)
    assert stage.writes > 0 and not stage.chunks and stage.events[-1] == "abort"

    class FailedWriter(Artifacts):
        def write_index(self, ref: PageRef, data: bytes) -> None:
            raise OSError("index write failed")

    with pytest.raises(OSError, match="index write failed"):
        build(schema, BytesReader(schema), FailedWriter())


def test_abort_error_preserves_the_original_failure() -> None:
    class FailedAbort(Staging):
        def abort(self) -> None:
            raise OSError("abort failed")

    schema, store = model(), Artifacts()
    manifest = build(schema, BytesReader(schema), store)
    with pytest.raises(OSError, match="write failure") as caught:
        restore(manifest, store, FailedAbort(fail="write"))
    assert str(caught.value.__cause__) == "abort failed"


def test_snapshot_streams_large_payload_without_retaining_a_model_in_memory() -> None:
    """64 MiB virtual input, discarded transport payloads, real file staging.

    Retained input/output storage and IO buffering are adapter
    responsibilities. tracemalloc checks Python allocations, not process RSS or
    backend resources.
    """
    limits = replace(LIMITS, codec=CodecLimits(max_chunk_bytes=65536), max_chunks_per_page=32)
    schema = ModelSchema(
        "a" * 64, "b" * 64, (TensorEntry(TensorSpec("weight", "uint8", (64 * 1024 * 1024,))),), 65536, limits
    )

    class VirtualReader:
        def __init__(self) -> None:
            self.reads = 0

        def read_chunk(self, spec: ChunkSpec) -> bytes:
            self.reads += 1
            return b"\x80" * spec.byte_length

    reader = VirtualReader()

    class VirtualArtifacts(Artifacts):
        def write_payload(self, record: ChunkRecord, payload: bytes) -> None:
            assert len(payload) == record.descriptor.spec.byte_length

        def write_index(self, ref: PageRef, data: bytes) -> None:
            # Pages are written before the rest of the input is read.
            assert reader.reads <= ref.first_chunk + ref.chunk_count + 1
            super().write_index(ref, data)

        def read_payload(self, record: ChunkRecord) -> bytes:
            return b"\x80" * record.descriptor.encoded_length

    with tempfile.TemporaryFile() as file:

        class FileStaging(Staging):
            def write_chunk(self, chunk: CanonicalChunk) -> None:
                assert file.tell() == chunk.spec.byte_offset
                file.write(chunk.data)

            def read_chunk(self, spec: ChunkSpec) -> bytes:
                file.seek(spec.byte_offset)
                return file.read(spec.byte_length)

            def abort(self) -> None:
                file.truncate(0)
                super().abort()

        store, stage = VirtualArtifacts(), FileStaging()
        tracemalloc.start()
        try:
            manifest = build_manifest(
                schema,
                reader,
                store,
                stream_id="test",
                run_epoch="epoch-1",
                version=0,
                writer_fence=1,
                source_step=0,
                exporter_revision="test-v1",
                limits=limits,
            )
            data = manifest.to_bytes(limits)
            rebuilt = reconstruct(data, store, stage, expected_manifest_id=digest(data), limits=limits)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert rebuilt.identity == manifest.target and stage.sealed
        assert peak < 8 * 1024 * 1024
        assert len(manifest.pages) == 32 and reader.reads == 1024
