# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Streaming model encode, verify, and isolated reconstruction contracts.

Callbacks own IO, leases, and storage. This module neither publishes versions
nor mutates live inference weights.
"""

from dataclasses import dataclass, field
from typing import Protocol

from .codec import Codec, DeltaEncoder, EncodedChunk
from .codec.format import content_hash
from .integrity import ModelRoot
from .limits import DeltaCodecError, SnapshotLimits, require_uint
from .manifest import ChunkRecord, IndexPage, Manifest, PageRef, SnapshotIdentity
from .model import ModelSchema
from .schema import CanonicalChunk, ChunkSpec
from .serialization import canonical_json, require_identifier


class ChunkReader(Protocol):
    """Read one canonical chunk from a caller-owned immutable snapshot
    lease."""

    def read_chunk(self, spec: ChunkSpec) -> bytes: ...


class ArtifactWriter(Protocol):
    """Synchronous bounded writes into unpublished storage.

    On failure the caller must discard/reclaim orphaned objects. Successful
    build_manifest only returns metadata; publication is a separate operation.
    """

    def write_payload(self, record: ChunkRecord, payload: bytes) -> None: ...

    def write_index(self, ref: PageRef, data: bytes) -> None: ...


class ArtifactReader(Protocol):
    """Enforce reference lengths *before* IO allocation, including queue
    limits.

    The core rechecks returned bytes, but cannot bound allocations in adapters.
    """

    def read_payload(self, record: ChunkRecord) -> bytes: ...

    def read_index(self, ref: PageRef) -> bytes: ...


class StagingSink(Protocol):
    """Private canonical storage, never live weights or a published snapshot.

    begin reserves an isolated generation including the complete tensor
    catalog. reader reads actual stored bytes for verification. seal freezes
    them without publishing; abort discards this generation even after a failed
    begin/seal. Aliases reference their catalog owner and zero-sized tensors
    need no writes. The sealed reader must remain immutable for its entire
    consumer lease.
    """

    def begin(self, schema: ModelSchema, identity: SnapshotIdentity) -> None: ...

    def write_chunk(self, chunk: CanonicalChunk) -> None: ...

    def reader(self) -> ChunkReader: ...

    def seal(self) -> None: ...

    def abort(self) -> None: ...


_VERIFIED = object()


@dataclass(frozen=True, init=False)
class VerifiedSnapshot:
    """Root-verified immutable view; obtain via verify_snapshot/reconstruct.

    This is an API guard against accidentally labeling unchecked data as a
    base, not a security boundary against malicious in-process Python code. The
    caller owns the immutable lease and must keep it alive while this handle is
    used.
    """

    schema: ModelSchema
    identity: SnapshotIdentity
    _reader: ChunkReader = field(repr=False, compare=False)

    def __init__(
        self, schema: ModelSchema, identity: SnapshotIdentity, reader: ChunkReader, *, _token: object
    ) -> None:
        if _token is not _VERIFIED:
            raise DeltaCodecError("snapshot must pass complete root verification")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "_reader", reader)

    def read_chunk(self, spec: ChunkSpec) -> bytes:
        if spec.schema_id != self.schema.schema_id:
            raise DeltaCodecError("snapshot chunk schema mismatch")
        return self._reader.read_chunk(spec)


def verify_snapshot(
    schema: ModelSchema, identity: SnapshotIdentity, reader: ChunkReader, *, limits: SnapshotLimits = SnapshotLimits()
) -> VerifiedSnapshot:
    """Hash every owner chunk, not just chunks that a later delta
    references."""
    schema.validate(limits)
    if identity.schema_id != schema.schema_id:
        raise DeltaCodecError("snapshot identity schema mismatch")
    root = ModelRoot(schema.schema_id, schema.directory_hash)
    for spec in schema.iter_chunks():
        chunk = CanonicalChunk(spec, identity.version, reader.read_chunk(spec))
        root.add(spec, content_hash(chunk.data))
    if root.hexdigest() != identity.target_root:
        raise DeltaCodecError("complete snapshot root mismatch")
    return VerifiedSnapshot(schema, identity, reader, _token=_VERIFIED)


def build_manifest(
    schema: ModelSchema,
    target: ChunkReader,
    writer: ArtifactWriter,
    *,
    stream_id: str,
    run_epoch: str,
    version: int,
    writer_fence: int,
    source_step: int,
    exporter_revision: str,
    base: VerifiedSnapshot | None = None,
    limits: SnapshotLimits = SnapshotLimits(),
) -> Manifest:
    """Encode one immutable exported version with bounded chunks/index pages.

    The exporter is responsible for a consistent lease across all reads. A full
    anchor is built by omitting base, regardless of the target version number.
    """
    schema.validate(limits)
    identity = SnapshotIdentity(stream_id, run_epoch, version, schema.schema_id, "0" * 64)
    require_uint(writer_fence, "writer_fence")
    require_uint(source_step, "source_step")
    require_identifier(exporter_revision, "exporter_revision")
    if base is not None:
        if not isinstance(base, VerifiedSnapshot):
            raise DeltaCodecError("encoding requires a verified base snapshot")
        if (
            (base.identity.stream_id, base.identity.run_epoch, base.identity.schema_id)
            != (stream_id, run_epoch, schema.schema_id)
        ) or base.identity.version >= version:
            raise DeltaCodecError("incompatible base identity")
    encoder = DeltaEncoder(limits=limits.codec)
    root = ModelRoot(schema.schema_id, schema.directory_hash)
    pages: list[PageRef] = []
    records: list[ChunkRecord] = []
    first = index_bytes = record_bytes = 0

    def flush() -> None:
        nonlocal first, index_bytes, record_bytes
        if not records:
            return
        if len(pages) >= limits.max_pages:
            raise DeltaCodecError("manifest page count exceeds limit")
        data = IndexPage(first, tuple(records)).to_bytes(limits)
        index_bytes += len(data)
        require_uint(index_bytes, "total index bytes", limits.max_index_bytes)
        digest = content_hash(data)
        ref = PageRef(first, len(records), len(data), digest, f"indexes/{digest}.json")
        writer.write_index(ref, data)
        pages.append(ref)
        first += len(records)
        records.clear()
        record_bytes = 0

    for spec in schema.iter_chunks():
        chunk = CanonicalChunk(spec, version, target.read_chunk(spec))
        previous = None if base is None else CanonicalChunk(spec, base.identity.version, base.read_chunk(spec))
        encoded = encoder.encode(chunk, base=previous)
        descriptor = encoded.descriptor
        record = ChunkRecord(
            descriptor, None if descriptor.codec == Codec.COPY_BASE else f"chunks/{descriptor.payload_hash}"
        )
        size = len(canonical_json(record.to_dict(limits), limits.max_index_page_bytes))
        overhead = len(
            canonical_json({"format_version": 1, "first_chunk": first, "records": []}, limits.max_index_page_bytes)
        )
        if records and (
            len(records) >= limits.max_chunks_per_page
            or overhead + record_bytes + len(records) + size > limits.max_index_page_bytes
        ):
            flush()
        overhead = len(
            canonical_json({"format_version": 1, "first_chunk": first, "records": []}, limits.max_index_page_bytes)
        )
        if overhead + size > limits.max_index_page_bytes:
            raise DeltaCodecError("single chunk record exceeds index page limit")
        if record.object_key is not None:
            writer.write_payload(record, encoded.payload)
        records.append(record)
        record_bytes += size
        root.add(spec, descriptor.target_hash)
    flush()
    identity = SnapshotIdentity(
        identity.stream_id, identity.run_epoch, identity.version, schema.schema_id, root.hexdigest()
    )
    manifest = Manifest(
        schema,
        identity,
        "FULL" if base is None else "DELTA",
        tuple(pages),
        writer_fence,
        source_step,
        exporter_revision,
        None if base is None else base.identity,
    )
    manifest.to_bytes(limits)
    return manifest


def reconstruct(
    manifest_bytes: bytes,
    artifacts: ArtifactReader,
    staging: StagingSink,
    *,
    expected_manifest_id: str,
    base: VerifiedSnapshot | None = None,
    limits: SnapshotLimits = SnapshotLimits(),
) -> VerifiedSnapshot:
    """Verify manifest, coverage, base, payloads, and stored root before
    sealing.

    Failure aborts the isolated generation; the current live version is outside
    this API. A DELTA consisting solely of RAW chunks can be decoded without
    base.
    """
    manifest = Manifest.from_bytes(manifest_bytes, expected_manifest_id=expected_manifest_id, limits=limits)
    if base is not None and (not isinstance(base, VerifiedSnapshot) or base.identity != manifest.base):
        raise DeltaCodecError("provided base differs from complete manifest base identity")
    expected_specs = iter(manifest.schema.iter_chunks())
    root = ModelRoot(manifest.schema.schema_id, manifest.schema.directory_hash)
    encoder = DeltaEncoder(limits=limits.codec)
    try:
        staging.begin(manifest.schema, manifest.target)
        for ref in manifest.pages:
            page = IndexPage.from_bytes(artifacts.read_index(ref), ref, limits)
            for record in page.records:
                descriptor = record.descriptor
                spec = next(expected_specs, None)
                if descriptor.spec != spec or descriptor.target_version != manifest.target.version:
                    raise DeltaCodecError("chunk directory coverage, order, or target version mismatch")
                previous = None
                if descriptor.codec != Codec.RAW_V1:
                    if manifest.kind != "DELTA" or base is None:
                        raise DeltaCodecError("dependent chunk requires a verified delta base")
                    if descriptor.base_version != base.identity.version:
                        raise DeltaCodecError("chunk base version differs from manifest")
                    previous = CanonicalChunk(spec, base.identity.version, base.read_chunk(spec))
                payload = b"" if record.object_key is None else artifacts.read_payload(record)
                chunk = encoder.decode(EncodedChunk(descriptor, payload), expected=descriptor, base=previous)
                root.add(spec, descriptor.target_hash)
                staging.write_chunk(chunk)
            del page
        if next(expected_specs, None) is not None or root.hexdigest() != manifest.target.target_root:
            raise DeltaCodecError("complete reconstructed root mismatch")
        verified = verify_snapshot(manifest.schema, manifest.target, staging.reader(), limits=limits)
        staging.seal()
        return verified
    except BaseException as error:
        try:
            staging.abort()
        except Exception as abort_error:
            raise error from abort_error
        raise
