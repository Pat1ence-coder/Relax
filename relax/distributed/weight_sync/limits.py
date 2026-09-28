# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Per-call bounds for the CPU reference codec."""

from dataclasses import dataclass


class DeltaCodecError(ValueError):
    """Invalid input, incompatible base, corrupt data, or exceeded codec
    limit."""


def require_uint(value: int, name: str, maximum: int = (1 << 64) - 1) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise DeltaCodecError(f"{name} must be an unsigned integer <= {maximum}")


@dataclass(frozen=True)
class CodecLimits:
    """Bounds apply before allocating payloads or reconstructed chunks.

    Input/output bytes belong to the caller. Temporary data buffers scale with
    one chunk, never with the number of tensors or versions. This is not a
    process RSS limit or a bound on transport queues.
    """

    max_chunk_bytes: int = 8 * 1024 * 1024
    max_metadata_bytes: int = 16 * 1024
    max_tensor_name_bytes: int = 1024
    max_tensor_rank: int = 32

    def __post_init__(self) -> None:
        for name in ("max_chunk_bytes", "max_metadata_bytes", "max_tensor_name_bytes", "max_tensor_rank"):
            value = getattr(self, name)
            require_uint(value, name, (1 << 32) - 1)
            if value == 0:
                raise DeltaCodecError(f"{name} must be positive")


@dataclass(frozen=True)
class SnapshotLimits:
    """Separate catalog, manifest, index, model, and chunk budgets.

    Catalog and manifest metadata are bounded in memory. Index pages and data
    chunks are streamed. These limits do not bound storage or callback queues.
    """

    codec: CodecLimits = CodecLimits()
    max_directory_bytes: int = 8 * 1024 * 1024
    max_manifest_bytes: int = 16 * 1024 * 1024
    max_index_page_bytes: int = 256 * 1024
    max_index_bytes: int = 256 * 1024 * 1024
    max_tensors: int = 65536
    max_chunks: int = 1 << 20
    max_pages: int = 8192
    max_chunks_per_page: int = 128
    max_model_bytes: int = 1 << 40

    def __post_init__(self) -> None:
        if not isinstance(self.codec, CodecLimits):
            raise DeltaCodecError("snapshot limits require CodecLimits")
        for name in self.__dataclass_fields__:
            if name == "codec":
                continue
            value = getattr(self, name)
            require_uint(value, name)
            if value == 0:
                raise DeltaCodecError(f"{name} must be positive")
