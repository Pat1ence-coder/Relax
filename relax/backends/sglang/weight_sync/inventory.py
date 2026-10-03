# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Strict target storage inventory and bindings for a single resident rank."""

import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from relax.distributed.weight_sync import DeltaCodecError
from relax.distributed.weight_sync.codec.format import content_hash
from relax.distributed.weight_sync.load import LoadPlan
from relax.distributed.weight_sync.serialization import canonical_json


def storage_binding(tensor: Any) -> tuple[str, int, int, int, tuple[int, ...], tuple[int, ...]]:
    return (
        str(tensor.device),
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset() * tensor.element_size(),
        tensor.numel() * tensor.element_size(),
        tuple(tensor.stride()),
        tuple(tensor.shape),
    )


@dataclass(frozen=True)
class TargetInventory:
    model: Any
    plan: LoadPlan
    rank: int
    tensors: Mapping[str, Any]
    bindings: Mapping[str, tuple]
    binding_id: str
    derived_buffers: tuple[str, ...]
    derived_bindings: Mapping[str, tuple]

    def validate_bindings(self, *, captured_buffers: bool = True) -> None:
        current = dict(self.model.named_parameters(remove_duplicate=False))
        if current.keys() != self.tensors.keys():
            raise DeltaCodecError("target parameter directory changed after prepare")
        for name, tensor in self.tensors.items():
            if current[name] is not tensor or storage_binding(tensor) != self.bindings[name]:
                raise DeltaCodecError("target storage changed after prepare")
            if str(tensor.dtype) != "torch.bfloat16" or not tensor.is_contiguous():
                raise DeltaCodecError("target dtype or contiguity changed after prepare")
        buffers = dict(self.model.named_buffers(remove_duplicate=False))
        names = tuple(sorted(buffers))
        if names != self.derived_buffers:
            raise DeltaCodecError("target buffer directory changed after prepare")
        for name, tensor in buffers.items():
            if captured_buffers and (storage_binding(tensor), str(tensor.dtype)) != self.derived_bindings[name]:
                raise DeltaCodecError("target execution buffer storage changed after capture")


def inspect_inventory(model: Any, plan: LoadPlan, rank: int, *, allow_cpu: bool = False) -> TargetInventory:
    import torch

    if sys.byteorder != "little":
        raise DeltaCodecError("target loader requires a little-endian host")
    if type(rank) is not int or rank not in plan.participants:
        raise DeltaCodecError("target rank is not a load participant")
    if not allow_cpu and type(model).__name__ != "Qwen3VLForConditionalGeneration":
        raise DeltaCodecError("unsupported target model class")
    pairs = list(model.named_parameters(remove_duplicate=False))
    tensors = dict(pairs)
    expected = {target.tensor.name: target for target in plan.targets if target.rank == rank}
    if len(pairs) != len(tensors) or set(tensors) != set(expected):
        raise DeltaCodecError("target parameter directory differs from the complete load plan")
    bindings, allocations, devices = {}, {}, set()
    modules = dict(model.named_modules(remove_duplicate=False))
    for name, tensor in tensors.items():
        target = expected[name]
        if type(tensor) not in (torch.nn.Parameter, torch.Tensor) or tensor.is_meta:
            raise DeltaCodecError("unsupported target parameter storage")
        if tensor.device.type != "cuda" and not (allow_cpu and tensor.device.type == "cpu"):
            raise DeltaCodecError("load requires resident CUDA weights")
        if (
            str(tensor.dtype) != "torch." + target.tensor.dtype
            or tuple(tensor.shape) != target.tensor.shape
            or not tensor.is_contiguous()
        ):
            raise DeltaCodecError(f"target shape/dtype/stride differs: {name}")
        binding = storage_binding(tensor)
        bindings[name] = binding
        device, pointer, start, length, _, _ = binding
        devices.add(device)
        if target.alias_of is not None:
            if binding != storage_binding(tensors[target.alias_of]):
                raise DeltaCodecError("target whole alias storage differs")
        elif length:
            allocations.setdefault(device, []).append((pointer + start, pointer + start + length, name))
        owner = modules[name.rpartition(".")[0]]
        if not allow_cpu:
            if hasattr(owner, "tp_size"):
                if owner.tp_size != len(plan.participants):
                    raise DeltaCodecError("target module TP size differs from group")
                if type(owner).__name__ in ("VocabParallelEmbedding", "ParallelLMHead"):
                    indices = owner.shard_indices
                    rows = target.tensor.shape[0]
                    source_name = next(
                        region.source
                        for region in plan.regions
                        if region.target == (target.alias_of or name) and region.source is not None
                    )
                    source_spec = next(
                        entry.tensor for entry in plan.schema.tensors if entry.tensor.name == source_name
                    )
                    if (
                        indices.padded_org_vocab_start_index != rank * rows
                        or indices.padded_org_vocab_end_index != (rank + 1) * rows
                        or owner.org_vocab_size != source_spec.shape[0]
                        or owner.num_embeddings != source_spec.shape[0]
                        or owner.embedding_dim != source_spec.shape[1]
                    ):
                        raise DeltaCodecError("target embedding shard metadata differs from load plan")
                elif getattr(owner, "tp_rank", None) != rank:
                    raise DeltaCodecError("target module TP rank differs from group rank")
            method = getattr(owner, "quant_method", None)
            if method is not None and type(method).__name__ not in (
                "UnquantizedLinearMethod",
                "UnquantizedEmbeddingMethod",
            ):
                raise DeltaCodecError("uncertified target weight execution method")
    if len(devices) != 1:
        raise DeltaCodecError("all target weights must be on one resident device")
    for ranges in allocations.values():
        ranges.sort()
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            raise DeltaCodecError("undeclared or partial target storage alias")
    derived = {}
    for name, buffer in model.named_buffers(remove_duplicate=False):
        parent, _, leaf = name.rpartition(".")
        owner = modules[parent]
        text_rope = parent.startswith("model.layers.") and parent.endswith(".self_attn.rotary_emb")
        vision_rope = parent == "visual.rotary_pos_emb"
        if (
            leaf != "cos_sin_cache"
            or not (text_rope or vision_rope)
            or type(owner).__name__ not in ("RotaryEmbedding", "MRotaryEmbedding")
            or leaf not in owner._non_persistent_buffers_set
            or buffer.dtype not in (torch.float32, torch.bfloat16)
            or buffer.ndim != 2
            or buffer.shape[1] != owner.rotary_dim
            or not buffer.is_contiguous()
        ):
            raise DeltaCodecError(f"unclassified target buffer: {name}")
        derived[name] = (storage_binding(buffer), str(buffer.dtype))
    identity = content_hash(
        canonical_json(
            {"plan": plan.plan_id, "rank": rank, "bindings": bindings}, plan.schema.limits.max_directory_bytes
        )
    )
    return TargetInventory(
        model,
        plan,
        rank,
        MappingProxyType(tensors),
        MappingProxyType(bindings),
        identity,
        tuple(sorted(derived)),
        MappingProxyType(derived),
    )
