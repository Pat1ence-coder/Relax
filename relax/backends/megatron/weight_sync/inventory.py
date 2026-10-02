# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Strict runtime inventory for the supported native Qwen3-VL profile."""

import sys
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping

from relax.distributed.weight_sync import DeltaCodecError, SourceExportPlan, TensorOwner
from relax.distributed.weight_sync.codec.format import content_hash
from relax.distributed.weight_sync.serialization import canonical_json

from .profiles import Qwen3VLProfile


def unwrap_model(model: Any) -> Any:
    if isinstance(model, (tuple, list)):
        if len(model) != 1:
            raise DeltaCodecError("export supports one model chunk, without VPP")
        model = model[0]
    for _ in range(8):
        if type(model).__name__ not in ("DistributedDataParallel", "Float16Module"):
            return model
        model = model.module
    raise DeltaCodecError("too many native model wrappers")


def _storage_range(tensor: Any) -> tuple[str, int, int, int]:
    return (
        str(tensor.device),
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset() * tensor.element_size(),
        tensor.numel() * tensor.element_size(),
    )


@dataclass(frozen=True)
class SourceInventory:
    model: Any
    profile: Qwen3VLProfile
    tensors: Mapping[str, Any]
    source_layout_id: str
    excluded_buffers: tuple[str, ...]

    def export_plan(self) -> SourceExportPlan:
        return SourceExportPlan(
            self.profile.schema,
            self.source_layout_id,
            tuple(range(self.profile.tp_size)),
            tuple(
                TensorOwner(entry.tensor.name, 0) for entry in self.profile.schema.tensors if entry.alias_of is None
            ),
            self.profile.schema.limits,
        )

    def read_span(self, name: str, offset: int, elements: int) -> bytes:
        """Copy only the requested contiguous interval, preserving its bits."""
        import torch

        tensor = self.tensors[name]
        if offset < 0 or elements <= 0 or offset + elements > tensor.numel():
            raise DeltaCodecError("source span exceeds native tensor")
        raw = tensor.detach().view(-1).narrow(0, offset, elements).view(torch.uint8)
        # This synchronous tile copy completes before exposing owned bytes.
        # No full parameter .cpu(), .contiguous(), float cast or gather occurs.
        return raw.to(device="cpu").numpy().tobytes()


def inspect_inventory(model: Any, profile: Qwen3VLProfile, *, allow_cpu: bool = False) -> SourceInventory:
    import torch

    if sys.byteorder != "little":
        raise DeltaCodecError("native exporter requires a little-endian host")
    model = unwrap_model(model)
    language = getattr(model, "language_model", None)
    if language is None or getattr(language, "share_embeddings_and_output_weights", None) != profile.tied_embeddings:
        raise DeltaCodecError("native and logical tied embedding semantics differ")
    native_config = getattr(language, "config", None)
    for name, expected in (
        ("tensor_model_parallel_size", profile.tp_size),
        ("pipeline_model_parallel_size", 1),
        ("context_parallel_size", 1),
        ("expert_model_parallel_size", 1),
        ("expert_tensor_parallel_size", 1),
        ("num_layers", profile.text_layers),
        ("hidden_size", profile.text_hidden),
        ("num_attention_heads", profile.text_heads),
        ("num_query_groups", profile.text_groups),
        ("kv_channels", profile.head_dim),
        ("ffn_hidden_size", profile.text_ffn),
    ):
        if getattr(native_config, name, None) != expected:
            raise DeltaCodecError(f"unsupported or incompatible native config: {name}")
    if any(
        getattr(native_config, name, None)
        for name in (
            "num_moe_experts",
            "mtp_num_layers",
            "fp8",
            "fp4",
            "moe_grouped_gemm",
            "layernorm_zero_centered_gamma",
            "attention_output_gate",
        )
    ):
        raise DeltaCodecError("unsupported native export variant")
    if getattr(native_config, "virtual_pipeline_model_parallel_size", None) not in (None, 1):
        raise DeltaCodecError("virtual pipeline export is unsupported")
    if getattr(native_config, "cuda_graph_impl", "none") != "none":
        raise DeltaCodecError("native CUDA graph model aliases are not yet supported")
    vision_config = getattr(getattr(model, "vision_model", None), "config", None)
    for actual_config, contract in (
        (native_config, profile.text_semantics),
        (vision_config, profile.vision_semantics),
    ):
        for name, expected_value in contract:
            actual = getattr(actual_config, name, None)
            if isinstance(expected_value, tuple) and isinstance(actual, (tuple, list)):
                actual = tuple(actual)
            if actual != expected_value:
                raise DeltaCodecError(f"native model semantics differ from profile: {name}")
    if getattr(vision_config, "cuda_graph_impl", "none") != "none" or getattr(vision_config, "num_moe_experts", None):
        raise DeltaCodecError("unsupported visual graph or MoE variant")
    vocab = native_config.vocab_size
    if getattr(native_config, "should_pad_vocab", None) is True:
        multiple = getattr(native_config, "make_vocab_size_divisible_by", None)
        if type(multiple) is not int or multiple <= 0:
            raise DeltaCodecError("native vocabulary padding rule is unknown")
        multiple *= profile.tp_size
        vocab = (vocab + multiple - 1) // multiple * multiple
    elif getattr(native_config, "should_pad_vocab", None) is not False:
        raise DeltaCodecError("native vocabulary padding mode is unknown")
    if vocab != profile.sources()["language_model.embedding.word_embeddings.weight"].shape[0]:
        raise DeltaCodecError("native vocabulary padding disagrees with profile")

    pairs = list(model.named_parameters(remove_duplicate=False))
    tensors = dict(pairs)
    if len(tensors) != len(pairs):
        raise DeltaCodecError("duplicate native parameter name")
    expected = profile.sources()
    embedding_name = "language_model.embedding.word_embeddings.weight"
    output_name = "language_model.output_layer.weight"
    if profile.tied_embeddings:
        embedding = tensors.get(embedding_name)
        shared = language.shared_embedding_or_output_weight()
        if embedding is None or shared is None or _storage_range(embedding) != _storage_range(shared):
            raise DeltaCodecError("native shared embedding is not the declared whole alias")
        if (
            tuple(embedding.shape) != tuple(shared.shape)
            or embedding.dtype != shared.dtype
            or type(shared) not in (torch.Tensor, torch.nn.Parameter)
            or not shared.is_contiguous()
            or tuple(embedding.stride()) != tuple(shared.stride())
        ):
            raise DeltaCodecError("native shared embedding shape/dtype differs")
        if output_name in tensors:
            output = tensors.pop(output_name)
            if (
                _storage_range(output) != _storage_range(embedding)
                or output.dtype != embedding.dtype
                or tuple(output.shape) != tuple(embedding.shape)
                or not output.is_contiguous()
                or type(output) not in (torch.Tensor, torch.nn.Parameter)
                or tuple(output.stride()) != tuple(embedding.stride())
            ):
                raise DeltaCodecError("tied output parameter is not a whole storage alias")
    if set(tensors) != set(expected):
        missing, extra = sorted(set(expected) - set(tensors)), sorted(set(tensors) - set(expected))
        raise DeltaCodecError(f"native parameter inventory mismatch: missing={missing[:8]}, extra={extra[:8]}")

    modules = dict(model.named_modules(remove_duplicate=False))
    excluded = []
    for name, buffer in model.named_buffers(remove_duplicate=False):
        parent, _, leaf = name.rpartition(".")
        owner = modules[parent]
        derived_vision_rope = (
            name == "vision_model.rotary_pos_emb.inv_freq"
            and type(owner).__name__ == "Qwen3VLVisionRotaryEmbedding"
            and getattr(owner, "dim", None) == vision_config.kv_channels // 2
            and getattr(owner, "theta", None) == 10000.0
            and leaf in owner._non_persistent_buffers_set
            and buffer.dtype == torch.float32
            and buffer.ndim == 1
            and buffer.numel() == (vision_config.kv_channels // 2 + 1) // 2
        )
        if not derived_vision_rope:
            raise DeltaCodecError(f"unclassified native buffer: {name}")
        excluded.append(name)

    descriptors = []
    allocations: dict[tuple[str, int], list[tuple[int, int, str]]] = {}
    devices = set()
    for name in sorted(tensors):
        tensor, spec = tensors[name], expected[name]
        if type(tensor) not in (torch.nn.Parameter, torch.Tensor) or tensor.is_meta:
            raise DeltaCodecError(f"unsupported native tensor storage: {name}")
        if tensor.device.type != "cuda" and not (allow_cpu and tensor.device.type == "cpu"):
            raise DeltaCodecError("capture requires resident CUDA weights")
        devices.add(str(tensor.device))
        if tuple(tensor.shape) != spec.local_shape(profile.tp_size) or str(tensor.dtype) != "torch." + spec.dtype:
            raise DeltaCodecError(f"native shape or dtype mismatch: {name}")
        if not tensor.is_contiguous():
            raise DeltaCodecError(f"noncontiguous native source storage is unsupported: {name}")
        attrs = (
            getattr(tensor, "tensor_model_parallel", False),
            getattr(tensor, "partition_dim", -1),
            getattr(tensor, "partition_stride", 1),
        )
        if attrs != (spec.partition_dim != -1, spec.partition_dim, spec.partition_stride):
            raise DeltaCodecError(f"native TP attributes differ from profile: {name}")
        device, pointer, start, size = _storage_range(tensor)
        if size:
            allocations.setdefault((device, pointer), []).append((start, start + size, name))
        descriptors.append({"spec": asdict(spec), "local_shape": list(tensor.shape), "strides": list(tensor.stride())})
    if len(devices) != 1:
        raise DeltaCodecError("capture requires one resident device per rank")
    for intervals in allocations.values():
        intervals.sort()
        for left, right in zip(intervals, intervals[1:]):
            if left[1] > right[0]:
                raise DeltaCodecError(f"undeclared or partial native storage alias: {left[2]}, {right[2]}")
    layout = canonical_json({"tp": profile.tp_size, "tensors": descriptors}, profile.schema.limits.max_directory_bytes)
    return SourceInventory(model, profile, MappingProxyType(tensors), content_hash(layout), tuple(sorted(excluded)))
