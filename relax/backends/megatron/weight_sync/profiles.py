# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Versioned Qwen3-VL native-to-canonical weight semantics.

This directory is generated from the logical model configuration, independently
of the runtime parameter iterator. Unknown source entries are rejected by the
inventory adapter rather than silently omitted from a valid-looking root.
"""

from dataclasses import dataclass
from typing import Any

from relax.distributed.weight_sync import DeltaCodecError, ModelSchema, SnapshotLimits, TensorEntry, TensorSpec
from relax.distributed.weight_sync.codec.format import content_hash
from relax.distributed.weight_sync.serialization import canonical_json


SEMANTICS_ID = content_hash(b"qwen3-vl-native-to-hf-copy-permute-v1")


@dataclass(frozen=True)
class NativeTensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    partition_dim: int = -1
    partition_stride: int = 1

    def local_shape(self, tp_size: int) -> tuple[int, ...]:
        shape = list(self.shape)
        if self.partition_dim >= 0:
            if shape[self.partition_dim] % (tp_size * self.partition_stride):
                raise DeltaCodecError("native partition is not divisible by TP and stride")
            shape[self.partition_dim] //= tp_size
        return tuple(shape)


@dataclass(frozen=True)
class TensorMapping:
    source: NativeTensorSpec
    target: TensorSpec
    transform: str = "identity"
    head_dim: int = 0
    query_heads: int = 0
    query_groups: int = 0


@dataclass(frozen=True)
class Qwen3VLProfile:
    schema: ModelSchema
    mappings: tuple[TensorMapping, ...]
    tp_size: int
    tied_embeddings: bool
    text_layers: int
    text_hidden: int
    text_heads: int
    text_groups: int
    head_dim: int
    text_ffn: int
    vision_layers: int
    text_semantics: tuple[tuple[str, Any], ...]
    vision_semantics: tuple[tuple[str, Any], ...]

    def sources(self) -> dict[str, NativeTensorSpec]:
        sources = {}
        for mapping in self.mappings:
            previous = sources.setdefault(mapping.source.name, mapping.source)
            if previous != mapping.source:
                raise DeltaCodecError("inconsistent source descriptions in model profile")
        return sources


def _positive(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if type(value) is not int or value <= 0:
        raise DeltaCodecError(f"Qwen3-VL config requires a positive {name}")
    return value


def qwen3_vl_profile(
    config: dict[str, Any],
    *,
    tp_size: int,
    padded_vocab_size: int,
    dtype: str = "bfloat16",
    chunk_bytes: int = 8 * 1024 * 1024,
    limits: SnapshotLimits = SnapshotLimits(),
) -> Qwen3VLProfile:
    if type(tp_size) is not int or tp_size not in (1, 2):
        raise DeltaCodecError("Qwen3-VL export currently supports TP1 or TP2")
    if config.get("model_type") != "qwen3_vl" or config.get("quantization_config") is not None:
        raise DeltaCodecError("export requires the unquantized Qwen3-VL Dense profile")
    if dtype not in ("bfloat16", "float32"):
        raise DeltaCodecError("Qwen3-VL export requires BF16 or FP32 source weights without casting")
    text, vision = config.get("text_config"), config.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise DeltaCodecError("Qwen3-VL requires text and vision configurations")
    if (
        text.get("hidden_act", "silu") != "silu"
        or vision.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh"
    ):
        raise DeltaCodecError("unsupported Qwen3-VL activation semantics")
    if any(text.get(name) for name in ("num_experts", "num_experts_per_tok", "num_nextn_predict_layers")):
        raise DeltaCodecError("MoE and MTP are outside the Dense export profile")
    tied = config.get("tie_word_embeddings")
    attention_bias = text.get("attention_bias", False)
    if type(tied) is not bool or type(attention_bias) is not bool:
        raise DeltaCodecError("tie and attention bias semantics must be explicit booleans")
    h, f = _positive(text, "hidden_size"), _positive(text, "intermediate_size")
    nq, nk, d = (_positive(text, key) for key in ("num_attention_heads", "num_key_value_heads", "head_dim"))
    layers, vocab = _positive(text, "num_hidden_layers"), _positive(text, "vocab_size")
    hv, iv = _positive(vision, "hidden_size"), _positive(vision, "intermediate_size")
    nv, depth = _positive(vision, "num_heads"), _positive(vision, "depth")
    merge, out = _positive(vision, "spatial_merge_size"), _positive(vision, "out_hidden_size")
    deepstack = vision.get("deepstack_visual_indexes")
    if not isinstance(deepstack, list) or any(type(i) is not int or not 0 <= i < depth for i in deepstack):
        raise DeltaCodecError("invalid visual deepstack indexes")
    if deepstack != sorted(set(deepstack)) or out != h:
        raise DeltaCodecError("unsupported visual deepstack order or output size")
    if nq % nk or nk % tp_size or hv % nv or nv % tp_size:
        raise DeltaCodecError("attention groups/heads are incompatible with TP")
    if type(padded_vocab_size) is not int or padded_vocab_size < vocab or padded_vocab_size % tp_size:
        raise DeltaCodecError("invalid native padded vocabulary size")
    if layers > limits.max_tensors or depth > limits.max_tensors:
        raise DeltaCodecError("profile layer count exceeds tensor budget")
    mappings: list[TensorMapping] = []

    def add(
        source: str,
        target: str,
        shape: tuple[int, ...],
        partition: int = -1,
        *,
        native_shape: tuple[int, ...] | None = None,
        transform: str = "identity",
        stride: int = 1,
        head_dim: int = 0,
        heads: int = 0,
        groups: int = 0,
    ) -> None:
        native = NativeTensorSpec(source, dtype, shape if native_shape is None else native_shape, partition, stride)
        native.local_shape(tp_size)
        mappings.append(TensorMapping(native, TensorSpec(target, dtype, shape), transform, head_dim, heads, groups))
        if len(mappings) > limits.max_tensors:
            raise DeltaCodecError("profile tensor count exceeds directory budget")

    embed = "model.language_model.embed_tokens.weight"
    add("language_model.embedding.word_embeddings.weight", embed, (vocab, h), 0, native_shape=(padded_vocab_size, h))
    if not tied:
        add("language_model.output_layer.weight", "lm_head.weight", (vocab, h), 0, native_shape=(padded_vocab_size, h))
    add("language_model.decoder.final_layernorm.weight", "model.language_model.norm.weight", (h,))
    for i in range(layers):
        source, target = f"language_model.decoder.layers.{i}", f"model.language_model.layers.{i}"
        for suffix in ("weight", "bias") if attention_bias else ("weight",):
            tail = (h,) if suffix == "weight" else ()
            for part, heads in (("q", nq), ("k", nk), ("v", nk)):
                add(
                    f"{source}.self_attention.linear_qkv.{suffix}",
                    f"{target}.self_attn.{part}_proj.{suffix}",
                    (heads * d, *tail),
                    0,
                    native_shape=((nq + 2 * nk) * d, *tail),
                    transform=part,
                    head_dim=d,
                    heads=nq,
                    groups=nk,
                )
        add(f"{source}.self_attention.linear_proj.weight", f"{target}.self_attn.o_proj.weight", (h, nq * d), 1)
        for part in ("q", "k"):
            add(f"{source}.self_attention.{part}_layernorm.weight", f"{target}.self_attn.{part}_norm.weight", (d,))
        add(f"{source}.self_attention.linear_qkv.layer_norm_weight", f"{target}.input_layernorm.weight", (h,))
        add(f"{source}.mlp.linear_fc1.layer_norm_weight", f"{target}.post_attention_layernorm.weight", (h,))
        for part in ("gate", "up"):
            add(
                f"{source}.mlp.linear_fc1.weight",
                f"{target}.mlp.{part}_proj.weight",
                (f, h),
                0,
                native_shape=(2 * f, h),
                transform=part,
                stride=2,
            )
        add(f"{source}.mlp.linear_fc2.weight", f"{target}.mlp.down_proj.weight", (h, f), 1)

    for i in range(depth):
        source, target = f"vision_model.decoder.layers.{i}", f"model.visual.blocks.{i}"
        for suffix in ("weight", "bias"):
            tail = (hv,) if suffix == "weight" else ()
            add(
                f"{source}.self_attention.linear_qkv.{suffix}",
                f"{target}.attn.qkv.{suffix}",
                (3 * hv, *tail),
                0,
                transform="vision_qkv",
                head_dim=hv // nv,
                heads=nv,
            )
            add(
                f"{source}.self_attention.linear_proj.{suffix}",
                f"{target}.attn.proj.{suffix}",
                (hv, *tail),
                1 if suffix == "weight" else -1,
            )
            for src, dst in (("self_attention.linear_qkv", "norm1"), ("mlp.linear_fc1", "norm2")):
                add(f"{source}.{src}.layer_norm_{suffix}", f"{target}.{dst}.{suffix}", (hv,))
            add(f"{source}.mlp.linear_fc1.{suffix}", f"{target}.mlp.linear_fc1.{suffix}", (iv, *tail), 0)
            add(
                f"{source}.mlp.linear_fc2.{suffix}",
                f"{target}.mlp.linear_fc2.{suffix}",
                (hv, iv) if suffix == "weight" else (hv,),
                1 if suffix == "weight" else -1,
            )

    patch = _positive(vision, "patch_size")
    add(
        "vision_model.patch_embed.proj.weight",
        "model.visual.patch_embed.proj.weight",
        (hv, _positive(vision, "in_channels"), _positive(vision, "temporal_patch_size"), patch, patch),
    )
    add("vision_model.patch_embed.proj.bias", "model.visual.patch_embed.proj.bias", (hv,))
    add(
        "vision_model.pos_embed.weight",
        "model.visual.pos_embed.weight",
        (_positive(vision, "num_position_embeddings"), hv),
    )
    merged = hv * merge * merge
    for ordinal in range(-1, len(deepstack)):
        source = "vision_model.merger" if ordinal == -1 else f"vision_model.decoder.deepstack_merger_list.{ordinal}"
        target = "model.visual.merger" if ordinal == -1 else f"model.visual.deepstack_merger_list.{ordinal}"
        for suffix in ("weight", "bias"):
            add(f"{source}.patch_norm.{suffix}", f"{target}.norm.{suffix}", (hv if ordinal == -1 else merged,))
            add(
                f"{source}.linear_fc1.{suffix}",
                f"{target}.linear_fc1.{suffix}",
                (merged, merged) if suffix == "weight" else (merged,),
                0,
            )
            add(
                f"{source}.linear_fc2.{suffix}",
                f"{target}.linear_fc2.{suffix}",
                (out, merged) if suffix == "weight" else (out,),
                1 if suffix == "weight" else -1,
            )

    entries = [TensorEntry(mapping.target) for mapping in mappings]
    if tied:
        entries.append(TensorEntry(TensorSpec("lm_head.weight", dtype, (vocab, h)), alias_of=embed))
    logical = {key: value for key, value in config.items() if key not in ("_name_or_path", "transformers_version")}
    schema = ModelSchema(
        content_hash(canonical_json(logical, limits.max_directory_bytes)),
        SEMANTICS_ID,
        tuple(entries),
        chunk_bytes,
        limits,
    )
    text_semantics = (
        ("normalization", "RMSNorm"),
        ("gated_linear_unit", True),
        ("qk_layernorm", True),
        ("add_bias_linear", False),
        ("add_qkv_bias", attention_bias),
        ("vocab_size", vocab),
        ("layernorm_epsilon", text.get("rms_norm_eps", 1e-6)),
        ("position_embedding_type", "mrope"),
        ("rotary_base", text.get("rope_theta", 5000000.0)),
        ("rotary_percent", 1.0),
        ("mrope_section", tuple(text.get("rope_scaling", {}).get("mrope_section", (24, 20, 20)))),
    )
    vision_semantics = (
        ("tensor_model_parallel_size", tp_size),
        ("pipeline_model_parallel_size", 1),
        ("context_parallel_size", 1),
        ("expert_model_parallel_size", 1),
        ("num_layers", depth),
        ("hidden_size", hv),
        ("num_attention_heads", nv),
        ("kv_channels", hv // nv),
        ("num_query_groups", nv),
        ("ffn_hidden_size", iv),
        ("patch_size", patch),
        ("temporal_patch_size", vision["temporal_patch_size"]),
        ("in_channels", vision["in_channels"]),
        ("spatial_merge_size", merge),
        ("num_position_embeddings", vision["num_position_embeddings"]),
        ("out_hidden_size", out),
        ("deepstack_visual_indexes", tuple(deepstack)),
        ("gated_linear_unit", False),
        ("normalization", "LayerNorm"),
        ("layernorm_zero_centered_gamma", False),
        ("qk_layernorm", False),
        ("add_bias_linear", True),
        ("add_qkv_bias", True),
    )
    return Qwen3VLProfile(
        schema, tuple(mappings), tp_size, tied, layers, h, nq, nk, d, f, depth, text_semantics, vision_semantics
    )
