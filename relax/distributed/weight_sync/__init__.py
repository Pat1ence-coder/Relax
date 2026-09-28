# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Portable weight-sync primitives; no backend or transport initialization."""

from .codec import ChunkDescriptor, Codec, DeltaEncoder, EncodedChunk
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
    "ChunkDescriptor",
    "ChunkReader",
    "ChunkRecord",
    "ChunkSpec",
    "Codec",
    "CodecLimits",
    "DeltaCodecError",
    "DeltaEncoder",
    "EncodedChunk",
    "IndexPage",
    "Manifest",
    "ModelRoot",
    "ModelSchema",
    "PageRef",
    "SnapshotIdentity",
    "SnapshotLimits",
    "StagingSink",
    "TensorEntry",
    "TensorSpec",
    "VerifiedSnapshot",
    "build_manifest",
    "iter_chunks",
    "reconstruct",
    "verify_snapshot",
]
