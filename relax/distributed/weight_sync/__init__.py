# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Portable weight-sync primitives; no backend or transport initialization."""

from .codec import ChunkDescriptor, Codec, DeltaEncoder, EncodedChunk
from .export import (
    CanonicalTile,
    ExportBudget,
    ExportRequest,
    SnapshotLease,
    SourceExportPlan,
    SourceReceipt,
    TensorOwner,
    validate_receipts,
)
from .integrity import ModelRoot
from .limits import CodecLimits, DeltaCodecError, SnapshotLimits
from .manifest import ChunkRecord, IndexPage, Manifest, PageRef, SnapshotIdentity
from .model import ModelSchema, TensorEntry
from .schema import CanonicalChunk, ChunkSpec, TensorSpec, iter_chunks
from .snapshot import (
    ArtifactReader,
    ArtifactWriter,
    ChunkReader,
    StagingSink,
    VerifiedSnapshot,
    build_manifest,
    reconstruct,
    verify_snapshot,
)


__all__ = [
    "ArtifactReader",
    "ArtifactWriter",
    "CanonicalChunk",
    "CanonicalTile",
    "ChunkDescriptor",
    "ChunkReader",
    "ChunkRecord",
    "ChunkSpec",
    "Codec",
    "CodecLimits",
    "DeltaCodecError",
    "DeltaEncoder",
    "EncodedChunk",
    "ExportBudget",
    "ExportRequest",
    "IndexPage",
    "Manifest",
    "ModelRoot",
    "ModelSchema",
    "PageRef",
    "SnapshotIdentity",
    "SnapshotLease",
    "SnapshotLimits",
    "SourceExportPlan",
    "SourceReceipt",
    "StagingSink",
    "TensorEntry",
    "TensorOwner",
    "TensorSpec",
    "VerifiedSnapshot",
    "build_manifest",
    "iter_chunks",
    "reconstruct",
    "validate_receipts",
    "verify_snapshot",
]
