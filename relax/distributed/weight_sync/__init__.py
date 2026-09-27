# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Portable weight-sync primitives; no backend or transport initialization."""

from .codec import ChunkDescriptor, Codec, DeltaEncoder, EncodedChunk
from .limits import CodecLimits, DeltaCodecError
from .schema import CanonicalChunk, ChunkSpec, TensorSpec, iter_chunks


__all__ = [
    "CanonicalChunk",
    "ChunkDescriptor",
    "ChunkSpec",
    "Codec",
    "CodecLimits",
    "DeltaCodecError",
    "DeltaEncoder",
    "EncodedChunk",
    "TensorSpec",
    "iter_chunks",
]
