# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Small independent canonical fixtures shared by target installation tests."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def model_config():
    return {
        "model_type": "qwen3_vl",
        "tie_word_embeddings": True,
        "text_config": {
            "hidden_size": 6,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "vocab_size": 67,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "attention_bias": False,
            "hidden_act": "silu",
            "rope_scaling": {"mrope_section": [0, 0, 1]},
        },
        "vision_config": {
            "hidden_size": 8,
            "intermediate_size": 12,
            "depth": 2,
            "num_heads": 4,
            "spatial_merge_size": 2,
            "out_hidden_size": 6,
            "deepstack_visual_indexes": [1],
            "patch_size": 2,
            "in_channels": 3,
            "temporal_patch_size": 2,
            "num_position_embeddings": 81,
            "hidden_act": "gelu_pytorch_tanh",
        },
    }


@pytest.fixture
def exporter_schema(model_config):
    # The independently implemented source exporter defines the portable input
    # schema. Loading this leaf avoids booting Megatron just to build metadata.
    path = Path(__file__).resolve().parents[4] / "relax/backends/megatron/weight_sync/profiles.py"
    name = "_source_schema_fixture"
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    try:
        yield module.qwen3_vl_profile(model_config, tp_size=2, padded_vocab_size=68, chunk_bytes=64).schema
    finally:
        del sys.modules[name]
