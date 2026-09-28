# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Content-addressed model manifest and bounded chunk-index pages."""

from dataclasses import asdict, dataclass
from typing import Any

from .codec import ChunkDescriptor, Codec
from .codec.format import content_hash
from .limits import DeltaCodecError, SnapshotLimits, require_uint
from .model import ModelSchema
from .schema import require_digest
from .serialization import canonical_json, exact_fields, parse_json, require_identifier


REQUIRED_FEATURES = ("canonical-le-v1", "chunk-codecs-v1", "model-root-v1", "paged-index-v1")


@dataclass(frozen=True)
class SnapshotIdentity:
    stream_id: str
    run_epoch: str
    version: int
    schema_id: str
    target_root: str

    def __post_init__(self) -> None:
        require_identifier(self.stream_id, "stream_id")
        require_identifier(self.run_epoch, "run_epoch")
        require_uint(self.version, "version")
        require_digest(self.schema_id, "schema_id")
        require_digest(self.target_root, "target_root")

    @classmethod
    def from_dict(cls, value: Any) -> "SnapshotIdentity":
        return cls(**exact_fields(value, {"stream_id", "run_epoch", "version", "schema_id", "target_root"}))


@dataclass(frozen=True)
class ChunkRecord:
    descriptor: ChunkDescriptor
    object_key: str | None

    def validate(self, limits: SnapshotLimits) -> None:
        if not isinstance(self.descriptor, ChunkDescriptor):
            raise DeltaCodecError("chunk record requires a descriptor")
        self.descriptor.validate(limits.codec)
        expected = None if self.descriptor.codec == Codec.COPY_BASE else f"chunks/{self.descriptor.payload_hash}"
        if self.object_key != expected:
            raise DeltaCodecError("invalid content-addressed payload key")

    def to_dict(self, limits: SnapshotLimits) -> dict[str, Any]:
        self.validate(limits)
        return {
            "descriptor": parse_json(self.descriptor.metadata_bytes(limits.codec), limits.codec.max_metadata_bytes),
            "object_key": self.object_key,
        }

    @classmethod
    def from_dict(cls, value: Any, limits: SnapshotLimits) -> "ChunkRecord":
        exact_fields(value, {"descriptor", "object_key"})
        descriptor = ChunkDescriptor.from_metadata_bytes(
            canonical_json(value["descriptor"], limits.codec.max_metadata_bytes), limits.codec
        )
        result = cls(descriptor, value["object_key"])
        result.validate(limits)
        return result


@dataclass(frozen=True)
class PageRef:
    first_chunk: int
    chunk_count: int
    byte_length: int
    page_hash: str
    object_key: str

    def validate(self, limits: SnapshotLimits) -> None:
        require_uint(self.first_chunk, "first_chunk", limits.max_chunks)
        require_uint(self.chunk_count, "page chunk_count", limits.max_chunks_per_page)
        require_uint(self.byte_length, "page byte_length", limits.max_index_page_bytes)
        if self.chunk_count == 0 or self.byte_length == 0:
            raise DeltaCodecError("empty index page")
        require_digest(self.page_hash, "page_hash")
        if self.object_key != f"indexes/{self.page_hash}.json":
            raise DeltaCodecError("invalid content-addressed index key")

    @classmethod
    def from_dict(cls, value: Any, limits: SnapshotLimits) -> "PageRef":
        result = cls(**exact_fields(value, {"first_chunk", "chunk_count", "byte_length", "page_hash", "object_key"}))
        result.validate(limits)
        return result


@dataclass(frozen=True)
class IndexPage:
    first_chunk: int
    records: tuple[ChunkRecord, ...]

    def to_bytes(self, limits: SnapshotLimits = SnapshotLimits()) -> bytes:
        require_uint(self.first_chunk, "first_chunk", limits.max_chunks)
        if not isinstance(self.records, tuple) or not 0 < len(self.records) <= limits.max_chunks_per_page:
            raise DeltaCodecError("invalid index page record count")
        if any(not isinstance(record, ChunkRecord) for record in self.records):
            raise DeltaCodecError("invalid index record")
        return canonical_json(
            {
                "format_version": 1,
                "first_chunk": self.first_chunk,
                "records": [record.to_dict(limits) for record in self.records],
            },
            limits.max_index_page_bytes,
        )

    @classmethod
    def from_bytes(cls, data: bytes, ref: PageRef, limits: SnapshotLimits = SnapshotLimits()) -> "IndexPage":
        ref.validate(limits)
        if type(data) is not bytes or len(data) != ref.byte_length or content_hash(data) != ref.page_hash:
            raise DeltaCodecError("index page length or hash mismatch")
        value = exact_fields(
            parse_json(data, limits.max_index_page_bytes), {"format_version", "first_chunk", "records"}
        )
        if type(value["format_version"]) is not int or value["format_version"] != 1:
            raise DeltaCodecError("unsupported index page version")
        if not isinstance(value["records"], list) or len(value["records"]) != ref.chunk_count:
            raise DeltaCodecError("index page record count differs from reference")
        require_uint(value["first_chunk"], "first_chunk", limits.max_chunks)
        if value["first_chunk"] != ref.first_chunk:
            raise DeltaCodecError("index page position differs from reference")
        return cls(value["first_chunk"], tuple(ChunkRecord.from_dict(record, limits) for record in value["records"]))


@dataclass(frozen=True)
class Manifest:
    schema: ModelSchema
    target: SnapshotIdentity
    kind: str
    pages: tuple[PageRef, ...]
    writer_fence: int
    source_step: int
    exporter_revision: str
    base: SnapshotIdentity | None = None

    def validate(self, limits: SnapshotLimits = SnapshotLimits()) -> None:
        if not isinstance(self.schema, ModelSchema) or not isinstance(self.target, SnapshotIdentity):
            raise DeltaCodecError("manifest requires schema and target identity")
        self.schema.validate(limits)
        if self.target.schema_id != self.schema.schema_id:
            raise DeltaCodecError("target schema differs from catalog")
        if self.kind not in ("FULL", "DELTA"):
            raise DeltaCodecError("unknown manifest kind")
        if self.kind == "FULL" and self.base is not None:
            raise DeltaCodecError("full snapshot must not depend on a base")
        if self.kind == "DELTA":
            if not isinstance(self.base, SnapshotIdentity):
                raise DeltaCodecError("delta snapshot requires a base identity")
            if (self.base.stream_id, self.base.run_epoch, self.base.schema_id) != (
                self.target.stream_id,
                self.target.run_epoch,
                self.target.schema_id,
            ):
                raise DeltaCodecError("delta base stream, epoch, or schema mismatch")
            if self.base.version >= self.target.version:
                raise DeltaCodecError("base version must precede target version")
        require_uint(self.writer_fence, "writer_fence")
        require_uint(self.source_step, "source_step")
        require_identifier(self.exporter_revision, "exporter_revision")
        if not isinstance(self.pages, tuple) or len(self.pages) > limits.max_pages:
            raise DeltaCodecError("manifest page count exceeds limit")
        count = total = 0
        for page in self.pages:
            if not isinstance(page, PageRef):
                raise DeltaCodecError("invalid page reference")
            page.validate(limits)
            if page.first_chunk != count:
                raise DeltaCodecError("index page gap, overlap, or reordering")
            count += page.chunk_count
            total += page.byte_length
            require_uint(total, "total index bytes", limits.max_index_bytes)
        if count != self.schema.chunk_count:
            raise DeltaCodecError("index does not cover the complete tensor directory")

    def to_bytes(self, limits: SnapshotLimits = SnapshotLimits()) -> bytes:
        self.validate(limits)
        return canonical_json(
            {
                "format_version": 1,
                "required_features": list(REQUIRED_FEATURES),
                "schema": self.schema.to_dict(),
                "target": asdict(self.target),
                "kind": self.kind,
                "base": None if self.base is None else asdict(self.base),
                "pages": [asdict(page) for page in self.pages],
                "writer_fence": self.writer_fence,
                "source_step": self.source_step,
                "exporter_revision": self.exporter_revision,
            },
            limits.max_manifest_bytes,
        )

    def manifest_id(self, limits: SnapshotLimits = SnapshotLimits()) -> str:
        return content_hash(self.to_bytes(limits))

    @classmethod
    def from_bytes(
        cls, data: bytes, *, expected_manifest_id: str, limits: SnapshotLimits = SnapshotLimits()
    ) -> "Manifest":
        require_digest(expected_manifest_id, "expected_manifest_id")
        if type(data) is not bytes or len(data) > limits.max_manifest_bytes:
            raise DeltaCodecError("manifest exceeds limit or is not bytes")
        if content_hash(data) != expected_manifest_id:
            raise DeltaCodecError("manifest hash differs from trusted identity")
        value = exact_fields(
            parse_json(data, limits.max_manifest_bytes),
            {
                "format_version",
                "required_features",
                "schema",
                "target",
                "kind",
                "base",
                "pages",
                "writer_fence",
                "source_step",
                "exporter_revision",
            },
        )
        if type(value["format_version"]) is not int or value["format_version"] != 1:
            raise DeltaCodecError("unsupported manifest version")
        if value["required_features"] != list(REQUIRED_FEATURES):
            raise DeltaCodecError("unsupported required manifest features")
        if not isinstance(value["pages"], list) or len(value["pages"]) > limits.max_pages:
            raise DeltaCodecError("manifest page count exceeds limit")
        schema = ModelSchema.from_bytes(canonical_json(value["schema"], limits.max_directory_bytes), limits)
        result = cls(
            schema,
            SnapshotIdentity.from_dict(value["target"]),
            value["kind"],
            tuple(PageRef.from_dict(page, limits) for page in value["pages"]),
            value["writer_fence"],
            value["source_step"],
            value["exporter_revision"],
            None if value["base"] is None else SnapshotIdentity.from_dict(value["base"]),
        )
        result.validate(limits)
        return result
