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
        ("disable_overlap_schedule", None),
        ("enable_lora", True),
        ("warmups", "custom"),
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
def test_text_graph_configuration_is_preserved_in_execution_identity(settings, phase):
    original = profile(settings)
    settings["cuda_graph_config"][phase]["backend"] = "full"
    assert profile(settings) != original


@pytest.mark.parametrize(
    "key,value",
    [
        ("dtype", "auto"),
        ("model_impl", "auto"),
        ("disable_overlap_schedule", False),
        ("skip_server_warmup", False),
        ("bf16_gemm_backend", "auto"),
        ("attention_backend", "fa3"),
        ("mm_attention_backend", None),
        ("decode_attention_backend", "flashinfer"),
    ],
)
def test_normal_backend_and_scheduling_choices_are_not_overridden(settings, key, value):
    original = profile(settings)
    settings[key] = value
    assert profile(settings) != original
    assert settings[key] == value


def test_pickle_transport_is_part_of_execution_identity(settings):
    assert capabilities.execution_profile(
        settings, vit_graph=False, pickle_ipc=True, derived_weight_cache=False
    ) != profile(settings)


def test_worker_capture_capacity_is_not_tokenizer_execution_semantics(settings):
    settings["cuda_graph_config"]["prefill"].update(backend="full", bs=[16, 32, 64])
    before = profile(settings)
    settings["cuda_graph_config"]["prefill"].update(bs=[16, 32], full_prefill_max_req=4)
    assert profile(settings) == before
    settings["cuda_graph_config"]["prefill"]["tc_compiler"] = "inductor"
    assert profile(settings) != before


def test_unset_lora_is_disabled_only_without_auto_enable_paths(settings):
    disabled = profile(settings)
    settings["enable_lora"] = None
    assert profile(settings) == disabled
    settings["lora_paths"] = ["adapter"]
    with pytest.raises(DeltaCodecError, match="enable_lora"):
        profile(settings)


def test_unknown_graph_and_non_boolean_pickle_modes_are_rejected(settings):
    with pytest.raises(DeltaCodecError, match="pickle"):
        capabilities.execution_profile(settings, vit_graph=False, pickle_ipc=1, derived_weight_cache=False)
    settings["cuda_graph_config"]["decode"]["backend"] = "unknown"
    with pytest.raises(DeltaCodecError, match="graph"):
        profile(settings)


@pytest.mark.parametrize("option", ["vit_graph", "derived_weight_cache"])
def test_environment_derived_features_cannot_bypass_resolved_args(settings, option):
    options = dict(vit_graph=False, pickle_ipc=False, derived_weight_cache=False)
    options[option] = True
    with pytest.raises(DeltaCodecError):
        capabilities.execution_profile(settings, **options)
