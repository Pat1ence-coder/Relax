# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Capability checks for supported SGLang execution settings."""

from typing import Any, Mapping

from relax.distributed.weight_sync import DeltaCodecError
from relax.distributed.weight_sync.codec.format import content_hash
from relax.distributed.weight_sync.serialization import canonical_json


def execution_profile(
    resolved: Mapping[str, Any], *, vit_graph: bool, pickle_ipc: bool, derived_weight_cache: bool
) -> str:
    """Validate effective values, never just the caller's launch arguments."""
    required = {
        "pp_size": 1,
        "dp_size": 1,
        "ep_size": 1,
        "nnodes": 1,
        "tokenizer_worker_num": 1,
        "detokenizer_worker_num": 1,
        "device": "cuda",
        "quantization": None,
        "warmups": None,
        "enable_lora": False,
        "enable_dp_attention": False,
        "enable_dp_lm_head": False,
        "mm_enable_dp_encoder": False,
        "enable_prefill_cp": False,
        "enable_torch_compile": False,
        "speculative_algorithm": None,
        "disaggregation_mode": "null",
        "weight_cache_mode": "off",
        "rl_on_policy_target": None,
        "elastic_ep_backend": None,
        "enable_memory_saver": False,
        "enable_prefix_mm_cache": False,
        "enable_mm_global_cache": False,
        "enable_hierarchical_cache": False,
        "cpu_offload_gb": 0,
        "attn_cp_size": 1,
        "dcp_size": 1,
        "moe_dp_size": 1,
        "dwdp_size": 1,
        "enable_pdmux": False,
        "pdmux_config_path": None,
        "dllm_algorithm": None,
        "dllm_algorithm_config": None,
        "enable_hisparse": False,
        "enable_session_radix_cache": False,
        "enable_streaming_session": False,
        "encoder_only": False,
        "language_only": False,
        "encoder_urls": [],
        "encoder_register_urls": [],
        "enable_adaptive_dispatch_to_encoder": False,
        "custom_weight_loader": [],
        "grpc_mode": False,
        "smg_grpc_mode": False,
        "grpc_port": None,
        "sidecar": None,
    }
    for name, expected in required.items():
        actual = resolved.get(name, object())
        # SGLang leaves the ordinary unset LoRA flag as None. Its later
        # path-based auto-enable must still be rejected before startup.
        if name == "enable_lora" and actual is None and not resolved.get("lora_paths"):
            actual = False
        if type(actual) is not type(expected) or actual != expected:
            raise DeltaCodecError(f"unsupported resolved SGLang setting: {name}")
    tp = resolved.get("tp_size")
    if type(tp) is not int or tp not in (1, 2, 4):
        raise DeltaCodecError("unsupported resolved SGLang TP size")
    # Automatic selection is validated against the actual model/storage by the
    # worker. Do not override the ordinary backend or scheduler configuration.
    for name, choices in (("dtype", ("auto", "bfloat16")), ("model_impl", ("auto", "sglang"))):
        if resolved.get(name) not in choices:
            raise DeltaCodecError(f"unsupported resolved SGLang setting: {name}")
    for name in ("disable_overlap_schedule", "skip_server_warmup"):
        if type(resolved.get(name)) is not bool:
            raise DeltaCodecError(f"invalid resolved SGLang setting: {name}")
    graph = resolved.get("cuda_graph_config")
    if not isinstance(graph, dict) or any(
        not isinstance(graph.get(phase), dict) or graph[phase].get("backend") not in ("disabled", "full")
        for phase in ("prefill", "decode")
    ):
        raise DeltaCodecError("unsupported resolved text CUDA graph configuration")
    if vit_graph is not False or derived_weight_cache is not False:
        raise DeltaCodecError("vision graph and derived weight caches are not supported")
    if type(pickle_ipc) is not bool:
        raise DeltaCodecError("invalid effective pickle IPC mode")
    # Capture bucket sizes are resolved again inside each worker from its
    # request pool. They are evidence about local capacity, not shared model
    # semantics, and need not equal the tokenizer's pre-capture values.
    graph_profile = {
        phase: {"backend": graph[phase]["backend"], "tc_compiler": graph[phase].get("tc_compiler")}
        for phase in ("prefill", "decode")
    }
    effective = {
        name: resolved.get(name)
        for name in (
            "dtype",
            "model_impl",
            "disable_overlap_schedule",
            "skip_server_warmup",
            "warmups",
            "bf16_gemm_backend",
            "attention_backend",
            "decode_attention_backend",
            "prefill_attention_backend",
            "mm_attention_backend",
        )
    }
    return content_hash(
        canonical_json(
            {"settings": {**required, **effective}, "tp_size": tp, "graph": graph_profile, "pickle_ipc": pickle_ipc},
            32 * 1024,
        )
    )
