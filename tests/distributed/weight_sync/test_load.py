# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Coverage, identity and bounded iteration contracts for target loading."""

from dataclasses import replace

import pytest

from relax.distributed.weight_sync import DeltaCodecError, ModelSchema, TensorEntry, TensorSpec
from relax.distributed.weight_sync.load import LoadBudget, LoadPlan, LoadRegion, TargetTensor, iter_load_tiles


def plan():
    schema = ModelSchema("a" * 64, "b" * 64, (TensorEntry(TensorSpec("x", "bfloat16", (3, 4))),), 8)
    targets = tuple(TargetTensor(rank, TensorSpec("w", "bfloat16", (3, 2))) for rank in range(2))
    regions = tuple(LoadRegion(rank, "w", 0, "x", rank * 4, 4, 3, 8) for rank in range(2))
    return LoadPlan(schema, "c" * 64, (0, 1), targets, regions, LoadBudget(tile_bytes=6))


def test_row_parallel_coverage_and_chunk_aligned_tiles():
    value = plan()
    source = bytes(range(24))
    for rank in value.participants:
        target = bytearray(12)
        for tile in iter_load_tiles(value, rank):
            assert tile.nbytes <= 6
            assert tile.source_offset // 8 == (tile.source_offset + tile.nbytes - 1) // 8
            target[tile.target_offset : tile.target_offset + tile.nbytes] = source[
                tile.source_offset : tile.source_offset + tile.nbytes
            ]
        assert target == b"".join(source[row * 8 + rank * 4 : row * 8 + rank * 4 + 4] for row in range(3))
    assert replace(value, regions=tuple(reversed(value.regions))).plan_id == value.plan_id
    assert replace(value, execution_profile_id="d" * 64).plan_id != value.plan_id


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: replace(p, regions=p.regions[:-1]),
        lambda p: replace(p, regions=p.regions + (p.regions[0],)),
        lambda p: replace(p, regions=(replace(p.regions[0], source_offset=4), p.regions[1])),
        lambda p: replace(p, regions=(replace(p.regions[0], target_offset=2), p.regions[1])),
        lambda p: replace(p, regions=(replace(p.regions[0], row_bytes=2), p.regions[1])),
        lambda p: replace(p, regions=(replace(p.regions[0], source="missing"), p.regions[1])),
        lambda p: replace(p, targets=p.targets + (p.targets[0],)),
        lambda p: replace(p, participants=(False, 1)),
        lambda p: replace(p, budget=LoadBudget(max_regions=1)),
        lambda p: replace(p, budget=LoadBudget(max_cpu_bytes=1)),
    ],
)
def test_load_plan_rejects_incomplete_ambiguous_or_unbounded_layout(mutate):
    with pytest.raises(DeltaCodecError):
        mutate(plan())


def test_alias_and_padding_are_separate_from_canonical_data():
    value = plan()
    targets = tuple(TargetTensor(r, TensorSpec("w", "bfloat16", (8,))) for r in range(2))
    targets += tuple(TargetTensor(r, TensorSpec("head", "bfloat16", (8,)), "w") for r in range(2))
    padded = replace(
        value, targets=targets, regions=value.regions + tuple(LoadRegion(r, "w", 12, None, 0, 4) for r in range(2))
    )
    assert sum(t.nbytes for t in iter_load_tiles(padded, 1)) == 16
    with pytest.raises(DeltaCodecError, match="alias"):
        replace(padded, regions=padded.regions + (LoadRegion(1, "head", 0, None, 0, 16),))
    with pytest.raises(DeltaCodecError, match="alias"):
        replace(padded, targets=targets[:-1] + (replace(targets[-1], alias_of="head"),))


def test_lightweight_load_import_has_no_backend_dependencies():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import relax.distributed.weight_sync.load; "
            "assert not any(x in sys.modules for x in ('torch', 'sglang', 'ray', 'megatron'))",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
