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
        "dtype": "bfloat16",
        "quantization": None,
        "enable_lora": False,
        "enable_dp_attention": False,
        "enable_dp_lm_head": False,
        "mm_enable_dp_encoder": False,
        "enable_prefill_cp": False,
        "enable_torch_compile": False,
        "disable_overlap_schedule": True,
        "speculative_algorithm": None,
        "disaggregation_mode": "null",
        "weight_cache_mode": "off",
        "skip_server_warmup": True,
        "warmups": None,
        "bf16_gemm_backend": "torch",
        "attention_backend": "triton",
        "mm_attention_backend": "sdpa",
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
        "model_impl": "sglang",
        "grpc_mode": False,
        "smg_grpc_mode": False,
        "grpc_port": None,
        "sidecar": None,
    }
    for name, expected in required.items():
        actual = resolved.get(name, object())
        if type(actual) is not type(expected) or actual != expected:
            raise DeltaCodecError(f"unsupported resolved SGLang setting: {name}")
    tp = resolved.get("tp_size")
    if type(tp) is not int or tp not in (1, 2, 4):
        raise DeltaCodecError("unsupported resolved SGLang TP size")
    graph = resolved.get("cuda_graph_config")
    if not isinstance(graph, dict) or any(
        not isinstance(graph.get(phase), dict) or graph[phase].get("backend") != "disabled"
        for phase in ("prefill", "decode")
    ):
        raise DeltaCodecError("both resolved CUDA graph phases must be disabled")
    if any(value is not False for value in (vit_graph, pickle_ipc, derived_weight_cache)):
        raise DeltaCodecError("vision graph, pickle IPC and derived weight caches must be disabled")
    return content_hash(canonical_json({"settings": required, "tp_size": tp, "graph": "disabled"}, 32 * 1024))
