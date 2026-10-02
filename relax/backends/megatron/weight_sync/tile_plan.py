# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Compile a canonical interval into bounded contiguous native shard spans."""

from collections.abc import Iterator
from dataclasses import dataclass
from math import prod

from relax.distributed.weight_sync import DeltaCodecError
from relax.distributed.weight_sync.limits import require_uint

from .profiles import TensorMapping


@dataclass(frozen=True)
class SourceSpan:
    rank: int
    source_offset: int
    target_offset: int
    elements: int


def _source_row(mapping: TensorMapping, row: int, tp_size: int) -> int:
    kind = mapping.transform
    if kind == "identity":
        return row
    if kind in ("gate", "up"):
        per_rank = mapping.target.shape[0] // tp_size
        rank, local = divmod(row, per_rank)
        return rank * 2 * per_rank + local + (per_rank if kind == "up" else 0)
    if kind == "vision_qkv":
        hidden = mapping.query_heads * mapping.head_dim
        projection, within = divmod(row, hidden)
        head, channel = divmod(within, mapping.head_dim)
        return (head * 3 + projection) * mapping.head_dim + channel
    if kind in ("q", "k", "v"):
        queries_per_group = mapping.query_heads // mapping.query_groups
        head, channel = divmod(row, mapping.head_dim)
        if kind == "q":
            group, within = divmod(head, queries_per_group)
        else:
            group, within = head, queries_per_group + (kind == "v")
        return (group * (queries_per_group + 2) + within) * mapping.head_dim + channel
    raise DeltaCodecError("unsupported canonical-to-native transform")


def iter_source_spans(mapping: TensorMapping, tp_size: int, offset: int, elements: int) -> Iterator[SourceSpan]:
    """Offsets and lengths are in elements, with no per-element index array.

    Only the current and previous row spans are retained. Large replicated
    tensors and ordinary column shards use a single span per shard boundary.
    """
    if type(tp_size) is not int or tp_size not in (1, 2):
        raise DeltaCodecError("tile planner supports TP1/TP2")
    require_uint(offset, "canonical element offset")
    require_uint(elements, "canonical element count")
    if elements == 0 or offset + elements > prod(mapping.target.shape):
        raise DeltaCodecError("canonical tile range exceeds tensor")
    local_shape = mapping.source.local_shape(tp_size)
    dim = mapping.source.partition_dim
    if dim == -1:
        if mapping.transform != "identity" or mapping.target.shape != mapping.source.shape:
            raise DeltaCodecError("replicated mapping must be an identical tensor")
        yield SourceSpan(0, offset, 0, elements)
        return
    if dim == 0 and mapping.transform == "identity":
        per_rank = prod(local_shape)
        current = 0
        while current < elements:
            rank, local = divmod(offset + current, per_rank)
            count = min(elements - current, per_rank - local)
            yield SourceSpan(rank, local, current, count)
            current += count
        return
    if dim not in (0, 1) or len(mapping.target.shape) > 2:
        raise DeltaCodecError("unsupported partitioned tensor dimensions")
    columns = prod(mapping.target.shape[1:])
    source_columns = prod(local_shape[1:])
    current = 0
    previous = None
    while current < elements:
        row, column = divmod(offset + current, columns)
        source_row = _source_row(mapping, row, tp_size)
        if dim == 0:
            rank, local_row = divmod(source_row, local_shape[0])
            local_column = column
            count = min(elements - current, columns - column)
        else:
            if mapping.transform != "identity":
                raise DeltaCodecError("row-parallel transforms must use identity rows")
            rank, local_column = divmod(column, source_columns)
            local_row = source_row
            count = min(elements - current, source_columns - local_column)
        span = SourceSpan(rank, local_row * source_columns + local_column, current, count)
        if previous is not None and (
            previous.rank == span.rank
            and previous.source_offset + previous.elements == span.source_offset
            and previous.target_offset + previous.elements == span.target_offset
        ):
            previous = SourceSpan(
                previous.rank, previous.source_offset, previous.target_offset, previous.elements + count
            )
        else:
            if previous is not None:
                yield previous
            previous = span
        current += count
    if previous is not None:
        yield previous
