# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Resolved settings gate supported execution modes."""

import json
from pathlib import Path

import pytest

from relax.backends.sglang.weight_sync import capabilities
from relax.distributed.weight_sync import DeltaCodecError


def profile(settings, **options):
    return capabilities.execution_profile(
        settings,
        **dict(vit_graph=False, pickle_ipc=False, derived_weight_cache=False, **options),
    )


@pytest.fixture
def settings():
    return json.loads(Path(__file__).with_name("certified_settings.json").read_text())


@pytest.mark.parametrize(
    "key,value",
    [
        ("disable_overlap_schedule", False),
        ("enable_lora", None),
        ("pp_size", 2),
        ("dp_size", 2),
        ("weight_cache_mode", "client"),
        ("dtype", "float16"),
        ("enable_torch_compile", True),
        ("cpu_offload_gb", 1),
        ("enable_dp_attention", True),
        ("tokenizer_worker_num", 2),
        ("mm_enable_dp_encoder", True),
        ("enable_session_radix_cache", True),
        ("enable_pdmux", True),
        ("speculative_algorithm", "EAGLE"),
        ("tp_size", 3),
        ("ep_size", True),
    ],
)
def test_resolved_unsupported_execution_is_refused(settings, key, value):
    assert len(profile(settings)) == 64
    settings[key] = value
    with pytest.raises(DeltaCodecError):
        profile(settings)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_effective_graph_phase_must_be_disabled(settings, phase):
    settings["cuda_graph_config"][phase]["backend"] = "cuda_graph"
    with pytest.raises(DeltaCodecError, match="graph"):
        profile(settings)


@pytest.mark.parametrize("option", ["vit_graph", "pickle_ipc", "derived_weight_cache"])
def test_environment_derived_features_cannot_bypass_resolved_args(settings, option):
    options = dict(vit_graph=False, pickle_ipc=False, derived_weight_cache=False)
    options[option] = True
    with pytest.raises(DeltaCodecError):
        capabilities.execution_profile(settings, **options)
