# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Independent numpy slicing oracle for every target parameter and TP rank."""

import re
from dataclasses import replace

import numpy as np
import pytest

from relax.backends.sglang.weight_sync.profiles import qwen3_vl_load_plan
from relax.distributed.weight_sync import DeltaCodecError
from relax.distributed.weight_sync.load import LoadBudget, iter_load_tiles


def canonical_arrays(schema):
    arrays = {}
    for ordinal, entry in enumerate(schema.tensors):
        if entry.alias_of is None:
            data = (np.arange(entry.tensor.nbytes // 2, dtype=np.uint32) * 73 + ordinal * 11).astype("<u2")
            data[:5] = [0, 0x8000, 0x7F80, 0x7F81, 0x7FC1][: min(5, len(data))]
            arrays[entry.tensor.name] = data.reshape(entry.tensor.shape)
    return arrays


def independent_target(name, arrays, rank, tp, config):
    text = config["text_config"]
    if name in ("model.embed_tokens.weight", "lm_head.weight", "visual.pos_embed.weight"):
        key = {
            "model.embed_tokens.weight": "model.language_model.embed_tokens.weight",
            "lm_head.weight": "model.language_model.embed_tokens.weight"
            if config["tie_word_embeddings"]
            else "lm_head.weight",
            "visual.pos_embed.weight": "model.visual.pos_embed.weight",
        }[name]
        original = arrays[key]
        padded = np.pad(original, ((0, (-original.shape[0]) % 64), (0, 0)))
        return np.array_split(padded, tp, axis=0)[rank]
    key = (
        name.replace("visual.", "model.visual.", 1)
        if name.startswith("visual.")
        else name.replace("model.", "model.language_model.", 1)
    )
    if ".self_attn.qkv_proj." in name:
        values = []
        for part in ("q", "k", "v"):
            original = arrays[key.replace("qkv_proj", part + "_proj")]
            heads = text["num_attention_heads"] if part == "q" else text["num_key_value_heads"]
            split_count = min(tp, heads)
            part_rank = rank if heads >= tp else rank // (tp // heads)
            values.append(np.array_split(original, split_count, axis=0)[part_rank])
        return np.concatenate(values)
    if ".gate_up_proj." in name:
        return np.concatenate(
            [
                np.array_split(arrays[key.replace("gate_up_proj", p + "_proj")], tp, axis=0)[rank]
                for p in ("gate", "up")
            ]
        )
    if ".attn.qkv_proj." in name:
        original = arrays[key.replace("qkv_proj", "qkv")]
        return np.concatenate([np.array_split(part, tp, axis=0)[rank] for part in np.split(original, 3)])
    original = arrays[key]
    if re.search(r"\.(o_proj|down_proj|proj|linear_fc2)\.weight$", name) and "patch_embed" not in name:
        return np.array_split(original, tp, axis=1)[rank]
    if ".linear_fc1." in name:
        return np.array_split(original, tp, axis=0)[rank]
    return original


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_all_target_bytes_against_independent_numpy_oracle(model_config, exporter_schema, tp):
    plan = qwen3_vl_load_plan(
        model_config, exporter_schema, tp_size=tp, execution_profile_id="a" * 64, budget=LoadBudget(tile_bytes=30)
    )
    arrays = canonical_arrays(exporter_schema)
    source = {name: array.tobytes() for name, array in arrays.items()}
    actual = {(t.rank, t.tensor.name): bytearray(t.tensor.nbytes) for t in plan.targets if t.alias_of is None}
    for rank in range(tp):
        for tile in iter_load_tiles(plan, rank):
            value = (
                b"\0" * tile.nbytes
                if tile.source is None
                else source[tile.source][tile.source_offset : tile.source_offset + tile.nbytes]
            )
            assert len(value) == tile.nbytes <= 30
            actual[rank, tile.target][tile.target_offset : tile.target_offset + tile.nbytes] = value
    for target in plan.targets:
        key = target.rank, target.alias_of or target.tensor.name
        expected = independent_target(target.tensor.name, arrays, target.rank, tp, model_config)
        assert tuple(expected.shape) == target.tensor.shape
        assert bytes(actual[key]) == expected.tobytes(), (tp, target.rank, target.tensor.name)


@pytest.mark.parametrize("change", ["missing", "dtype", "alias", "extra"])
def test_profile_requires_exact_canonical_directory(model_config, exporter_schema, change):
    entries = list(exporter_schema.tensors)
    if change == "missing":
        entries.pop()
    elif change == "dtype":
        entries[-1] = replace(entries[-1], tensor=replace(entries[-1].tensor, dtype="float16"))
    elif change == "alias":
        entries[0] = replace(entries[0], alias_of=None)
    else:
        entries.append(replace(entries[-1], tensor=replace(entries[-1].tensor, name="unexpected.weight")))
    schema = replace(exporter_schema, tensors=tuple(entries))
    with pytest.raises(DeltaCodecError):
        qwen3_vl_load_plan(model_config, schema, tp_size=2, execution_profile_id="a" * 64)
