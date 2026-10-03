# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Execution caches refreshed in place while every rank is idle."""

import hashlib
from typing import Any

from relax.distributed.weight_sync import DeltaCodecError
from relax.distributed.weight_sync.serialization import canonical_json


def clear_multimodal_execution_cache() -> dict:
    """Discard image embeddings derived from the previous visual weights."""
    from sglang.srt.managers import mm_utils

    cache = mm_utils.embedding_cache
    if cache is None:
        return {"present": False}
    evidence = {"present": True, "entries_before": len(cache), "bytes_before": cache.current_size}
    cache.clear()
    evidence.update(entries_after=len(cache), bytes_after=cache.current_size)
    if evidence["entries_after"] or evidence["bytes_after"]:
        raise DeltaCodecError("multimodal execution cache did not clear")
    return evidence


def validate_model_config(model: Any, config: dict) -> None:
    """Compare supplied logical configuration with the actual loaded model."""
    from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration

    if type(model) is not Qwen3VLForConditionalGeneration or model.quant_config is not None:
        raise DeltaCodecError("uncertified actual SGLang model implementation")
    actual = model.config.to_dict()

    def compare(expected: Any, observed: Any, prefix: str) -> None:
        if isinstance(expected, dict):
            if not isinstance(observed, dict):
                raise DeltaCodecError(f"actual model config differs: {prefix}")
            for key, value in expected.items():
                if key in ("_name_or_path", "transformers_version", "torch_dtype", "dtype"):
                    continue
                # Transformers normalizes the legacy rope_scaling key.
                other = observed.get(key, observed.get("rope_parameters") if key == "rope_scaling" else None)
                if key == "rope_theta" and other is None:
                    other = observed.get("rope_parameters", {}).get("rope_theta")
                if prefix == "model.vision_config" and key == "model_type" and value == "qwen3_vl":
                    value = "qwen3_vl_vision"
                compare(value, other, f"{prefix}.{key}")
        elif expected != observed:
            raise DeltaCodecError(f"actual model config differs: {prefix}")

    compare(config, actual, "model")


def _refresh_cache_in_place(cache: Any, replacement: Any) -> None:
    """Preserve storage retained by already captured text CUDA graphs."""
    if cache.shape != replacement.shape or cache.device != replacement.device or not cache.is_contiguous():
        raise DeltaCodecError("rotary refresh would change captured storage")
    cache.copy_(replacement)


def _compute_rotary_cache(module: Any, rows: int) -> Any:
    """Recompute the full runtime capacity, including graph warmup padding."""
    import torch

    inv_freq = module._compute_inv_freq(module.base)
    positions = torch.arange(rows, dtype=torch.float, device=module.cos_sin_cache.device)
    freqs = torch.einsum("i,j -> ij", positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _execution_state(model: Any, config: dict, *, max_bytes: int, tile_bytes: int, refresh: bool) -> str:
    """Rebuild shared RoPE caches once per module and hash actual device bytes.

    Derived caches have their own explicit allocation budget. This budget is
    separate from canonical tile copies and includes construction temporaries.
    """
    import torch
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

    validate_model_config(model, config)
    text, vision = config["text_config"], config["vision_config"]
    visited, descriptors = {}, []
    resident_bytes = 0
    digest = hashlib.sha256()
    device = next(model.parameters()).device
    rope_config = text.get("rope_parameters") or text.get("rope_scaling", {})
    with torch.no_grad(), torch.device(device):
        for name, module in model.named_modules(remove_duplicate=False):
            if id(module) in visited:
                digest.update(canonical_json({"module_alias": name, "owner": visited[id(module)]}, 4096))
                continue
            language = name.startswith("model.layers.") and name.endswith(".self_attn.rotary_emb")
            visual = name == "visual.rotary_pos_emb"
            if not (language or visual):
                continue
            if type(module) not in (RotaryEmbedding, MRotaryEmbedding):
                raise DeltaCodecError("uncertified derived rotary cache type")
            visited[id(module)] = name
            head = text["head_dim"] if language else vision["hidden_size"] // vision["num_heads"]
            expected = {
                "head_size": head,
                "rotary_dim": head if language else head // 2,
                "max_position_embeddings": text.get("max_position_embeddings", 32768) if language else 8192,
                "base": rope_config.get("rope_theta", text.get("rope_theta", 1000000)) if language else 10000.0,
                "is_neox_style": True,
            }
            for key, value in expected.items():
                if getattr(module, key, None) != value:
                    raise DeltaCodecError(f"rotary execution semantics differ: {key}")
            if language and (
                type(module) is not MRotaryEmbedding
                or module.mrope_section != rope_config.get("mrope_section")
                or module.mrope_interleaved != rope_config.get("mrope_interleaved", False)
                or module.mrope_interleaved_glm != rope_config.get("mrope_interleaved_glm", False)
            ):
                raise DeltaCodecError("MRoPE section/interleave semantics differ")
            cache = module.cos_sin_cache
            # SGLang extends both text and vision caches before graph capture
            # to the runtime context capacity plus padding. Logical model
            # limits remain unchanged, and these extra rows must be retained.
            if cache.ndim != 2 or cache.shape[0] < expected["max_position_embeddings"]:
                raise DeltaCodecError("rotary cache has insufficient position capacity")
            cache_bytes = cache.shape[0] * expected["rotary_dim"] * 4
            old_bytes = module.cos_sin_cache.numel() * module.cos_sin_cache.element_size()
            if resident_bytes + old_bytes + 4 * cache_bytes > max_bytes:
                raise DeltaCodecError("derived execution cache refresh exceeds its allocation budget")
            resident_bytes += cache_bytes
            if (
                cache.device != device
                or cache.dtype not in (torch.float32, torch.bfloat16)
                or not cache.is_contiguous()
                or cache.shape[1] != expected["rotary_dim"]
            ):
                raise DeltaCodecError("rotary cache has unexpected storage")
            if getattr(module, "position_cos", None) is not None or getattr(module, "position_sin", None) is not None:
                raise DeltaCodecError("unexpected cached rotary positions after quiescent refresh")
            if refresh:
                replacement = _compute_rotary_cache(module, cache.shape[0])
                _refresh_cache_in_place(cache, replacement)
                del replacement
            cache_bytes = cache.numel() * cache.element_size()
            descriptor = {
                "module": name,
                "semantics": expected,
                "nbytes": cache_bytes,
                "binding": [str(cache.device), cache.data_ptr(), list(cache.shape), str(cache.dtype)],
            }
            descriptors.append(descriptor)
            digest.update(canonical_json(descriptor, 4096))
            raw = cache.view(-1).view(torch.uint8)
            for start in range(0, cache_bytes, tile_bytes):
                digest.update(raw.narrow(0, start, min(tile_bytes, cache_bytes - start)).cpu().numpy().tobytes())
    if not descriptors:
        raise DeltaCodecError("expected derived execution state is missing")
    torch.cuda.synchronize(device)
    # Vision graph support is separate from the text graph path. Do not
    # silently disable or clear a vision graph to make an installation pass.
    if model.visual.graph_runners.block_graphs:
        raise DeltaCodecError("unexpected captured vision graph")
    return digest.hexdigest()


def refresh_execution_state(model: Any, config: dict, *, max_bytes: int, tile_bytes: int) -> str:
    return _execution_state(model, config, max_bytes=max_bytes, tile_bytes=tile_bytes, refresh=True)


def verify_execution_state(model: Any, config: dict, expected: str, *, max_bytes: int, tile_bytes: int) -> None:
    actual = _execution_state(model, config, max_bytes=max_bytes, tile_bytes=tile_bytes, refresh=False)
    if actual != expected:
        raise DeltaCodecError("actual derived execution state changed after refresh")
