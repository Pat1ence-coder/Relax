# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Independent canonical-to-SGLang Qwen3-VL target layout.

No Megatron conversion or SGLang weight loader is imported here. The complete
canonical directory is checked against configuration before planning writes.
"""

from math import prod
from typing import Any

from relax.distributed.weight_sync import DeltaCodecError, ModelSchema, TensorEntry, TensorSpec
from relax.distributed.weight_sync.codec.format import content_hash
from relax.distributed.weight_sync.load import LoadBudget, LoadPlan, LoadRegion, TargetTensor
from relax.distributed.weight_sync.serialization import canonical_json


def qwen3_vl_load_plan(
    config: dict[str, Any],
    schema: ModelSchema,
    *,
    tp_size: int,
    execution_profile_id: str,
    budget: LoadBudget = LoadBudget(),
) -> LoadPlan:
    if type(tp_size) is not int or tp_size not in (1, 2, 4):
        raise DeltaCodecError("SGLang load profile supports TP1/TP2/TP4")
    if config.get("model_type") != "qwen3_vl" or config.get("quantization_config") is not None:
        raise DeltaCodecError("load requires unquantized Dense Qwen3-VL")
    if any(config.get(key) for key in ("encoder_only", "language_only")):
        raise DeltaCodecError("load requires the complete vision-language model")
    text, vision = config.get("text_config"), config.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise DeltaCodecError("load profile requires text and vision config")
    if text.get("attention_bias", False) is not False:
        raise DeltaCodecError("attention bias is not certified in this load profile")
    if (
        text.get("hidden_act", "silu") != "silu"
        or vision.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh"
    ):
        raise DeltaCodecError("unsupported activation semantics")
    if any(text.get(key) for key in ("num_experts", "num_experts_per_tok", "num_nextn_predict_layers")):
        raise DeltaCodecError("MoE and MTP are outside the load profile")
    tied = config.get("tie_word_embeddings")
    if type(tied) is not bool:
        raise DeltaCodecError("tie_word_embeddings must be explicit")

    def positive(section: dict[str, Any], key: str) -> int:
        value = section.get(key)
        if type(value) is not int or value <= 0 or value > schema.limits.max_model_bytes:
            raise DeltaCodecError(f"invalid model dimension: {key}")
        return value

    h, f, layers, vocab, nq, nk, d = (
        positive(text, key)
        for key in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        )
    )
    hv, iv, depth, nv, merge, out = (
        positive(vision, key)
        for key in ("hidden_size", "intermediate_size", "depth", "num_heads", "spatial_merge_size", "out_hidden_size")
    )
    deepstack = vision.get("deepstack_visual_indexes")
    if (
        not isinstance(deepstack, list)
        or any(type(i) is not int or not 0 <= i < depth for i in deepstack)
        or deepstack != sorted(set(deepstack))
        or out != h
    ):
        raise DeltaCodecError("unsupported deepstack configuration")
    if layers + depth > schema.limits.max_tensors:
        raise DeltaCodecError("layer count exceeds directory budget")
    if (
        nq % nk
        or nq % tp_size
        or hv % nv
        or nv % tp_size
        or f % tp_size
        or iv % tp_size
        or (nk >= tp_size and nk % tp_size)
        or (nk < tp_size and tp_size % nk)
    ):
        raise DeltaCodecError("model dimensions cannot use this tensor parallel layout")
    canonical: dict[str, TensorEntry] = {}
    targets: list[TargetTensor] = []
    regions: list[LoadRegion] = []

    def spec(name: str, shape: tuple[int, ...]) -> TensorSpec:
        result = TensorSpec(name, "bfloat16", shape)
        if result.nbytes > schema.limits.max_model_bytes:
            raise DeltaCodecError("profile tensor exceeds model budget")
        return result

    def source(name: str, shape: tuple[int, ...]) -> None:
        entry = TensorEntry(spec(name, shape))
        if canonical.setdefault(name, entry) != entry:
            raise DeltaCodecError("inconsistent canonical description")

    def target(rank: int, name: str, shape: tuple[int, ...], alias: str | None = None) -> None:
        targets.append(TargetTensor(rank, spec(name, shape), alias))

    def copy(
        rank: int,
        dst: str,
        src: str,
        count: int,
        src_offset: int = 0,
        dst_offset: int = 0,
        rows: int = 1,
        stride: int = 0,
    ) -> None:
        if count:
            regions.append(LoadRegion(rank, dst, dst_offset * 2, src, src_offset * 2, count * 2, rows, stride * 2))

    def regular(src: str, dst: str, shape: tuple[int, ...], partition: int = -1) -> None:
        source(src, shape)
        for rank in range(tp_size):
            local = list(shape)
            if partition >= 0:
                if local[partition] % tp_size:
                    raise DeltaCodecError("target partition dimension is not divisible")
                local[partition] //= tp_size
            target(rank, dst, tuple(local))
            size = prod(local)
            if partition == -1:
                copy(rank, dst, src, size)
            elif partition == 0:
                copy(rank, dst, src, size, rank * size)
            elif partition == 1 and len(shape) == 2:
                copy(rank, dst, src, local[1], rank * local[1], rows=local[0], stride=shape[1])
            else:
                raise DeltaCodecError("unsupported target partition")

    def embedding(src: str, dst: str, size: int, hidden: int) -> None:
        source(src, (size, hidden))
        padded = (size + 63) // 64 * 64
        per_rank = padded // tp_size
        for rank in range(tp_size):
            target(rank, dst, (per_rank, hidden))
            start = min(rank * per_rank, size)
            length = min((rank + 1) * per_rank, size) - start
            copy(rank, dst, src, length * hidden, start * hidden)
            if length < per_rank:
                regions.append(LoadRegion(rank, dst, length * hidden * 2, None, 0, (per_rank - length) * hidden * 2))

    embed = "model.language_model.embed_tokens.weight"
    embedding(embed, "model.embed_tokens.weight", vocab, h)
    if tied:
        canonical["lm_head.weight"] = TensorEntry(spec("lm_head.weight", (vocab, h)), alias_of=embed)
        for rank in range(tp_size):
            target(rank, "lm_head.weight", ((vocab + 63) // 64 * 64 // tp_size, h), "model.embed_tokens.weight")
    else:
        embedding("lm_head.weight", "lm_head.weight", vocab, h)
    regular("model.language_model.norm.weight", "model.norm.weight", (h,))
    for layer in range(layers):
        src, dst = f"model.language_model.layers.{layer}", f"model.layers.{layer}"
        local_q, local_kv = nq // tp_size * d, max(1, nk // tp_size) * d
        for part, heads in (("q", nq), ("k", nk), ("v", nk)):
            source(f"{src}.self_attn.{part}_proj.weight", (heads * d, h))
        for rank in range(tp_size):
            name = f"{dst}.self_attn.qkv_proj.weight"
            target(rank, name, (local_q + 2 * local_kv, h))
            for part, length, offset in (
                ("q", local_q, 0),
                ("k", local_kv, local_q),
                ("v", local_kv, local_q + local_kv),
            ):
                shard = rank if part == "q" or nk >= tp_size else rank // (tp_size // nk)
                copy(rank, name, f"{src}.self_attn.{part}_proj.weight", length * h, shard * length * h, offset * h)
        regular(f"{src}.self_attn.o_proj.weight", f"{dst}.self_attn.o_proj.weight", (h, nq * d), 1)
        for part in ("q", "k"):
            regular(f"{src}.self_attn.{part}_norm.weight", f"{dst}.self_attn.{part}_norm.weight", (d,))
        for norm in ("input_layernorm", "post_attention_layernorm"):
            regular(f"{src}.{norm}.weight", f"{dst}.{norm}.weight", (h,))
        for part in ("gate", "up"):
            source(f"{src}.mlp.{part}_proj.weight", (f, h))
        for rank in range(tp_size):
            name, length = f"{dst}.mlp.gate_up_proj.weight", f // tp_size * h
            target(rank, name, (2 * f // tp_size, h))
            for part, offset in (("gate", 0), ("up", length)):
                copy(rank, name, f"{src}.mlp.{part}_proj.weight", length, rank * length, offset)
        regular(f"{src}.mlp.down_proj.weight", f"{dst}.mlp.down_proj.weight", (h, f), 1)

    for layer in range(depth):
        src, dst = f"model.visual.blocks.{layer}", f"visual.blocks.{layer}"
        for suffix in ("weight", "bias"):
            tail = (hv,) if suffix == "weight" else ()
            source(f"{src}.attn.qkv.{suffix}", (3 * hv, *tail))
            for rank in range(tp_size):
                name, length = f"{dst}.attn.qkv_proj.{suffix}", hv // tp_size * prod(tail)
                target(rank, name, (3 * hv // tp_size, *tail))
                for part in range(3):
                    copy(
                        rank,
                        name,
                        f"{src}.attn.qkv.{suffix}",
                        length,
                        part * hv * prod(tail) + rank * length,
                        part * length,
                    )
            regular(
                f"{src}.attn.proj.{suffix}", f"{dst}.attn.proj.{suffix}", (hv, *tail), 1 if suffix == "weight" else -1
            )
            for norm in ("norm1", "norm2"):
                regular(f"{src}.{norm}.{suffix}", f"{dst}.{norm}.{suffix}", (hv,))
            regular(f"{src}.mlp.linear_fc1.{suffix}", f"{dst}.mlp.linear_fc1.{suffix}", (iv, *tail), 0)
            regular(
                f"{src}.mlp.linear_fc2.{suffix}",
                f"{dst}.mlp.linear_fc2.{suffix}",
                (hv, iv) if suffix == "weight" else (hv,),
                1 if suffix == "weight" else -1,
            )
    patch = positive(vision, "patch_size")
    regular(
        "model.visual.patch_embed.proj.weight",
        "visual.patch_embed.proj.weight",
        (hv, positive(vision, "in_channels"), positive(vision, "temporal_patch_size"), patch, patch),
    )
    regular("model.visual.patch_embed.proj.bias", "visual.patch_embed.proj.bias", (hv,))
    embedding(
        "model.visual.pos_embed.weight", "visual.pos_embed.weight", positive(vision, "num_position_embeddings"), hv
    )
    merged = hv * merge * merge
    for ordinal in range(-1, len(deepstack)):
        name = "merger" if ordinal == -1 else f"deepstack_merger_list.{ordinal}"
        for suffix in ("weight", "bias"):
            regular(
                f"model.visual.{name}.norm.{suffix}",
                f"visual.{name}.norm.{suffix}",
                (hv if ordinal == -1 else merged,),
            )
            regular(
                f"model.visual.{name}.linear_fc1.{suffix}",
                f"visual.{name}.linear_fc1.{suffix}",
                (merged, merged) if suffix == "weight" else (merged,),
                0,
            )
            regular(
                f"model.visual.{name}.linear_fc2.{suffix}",
                f"visual.{name}.linear_fc2.{suffix}",
                (out, merged) if suffix == "weight" else (out,),
                1 if suffix == "weight" else -1,
            )
    logical = {key: value for key, value in config.items() if key not in ("_name_or_path", "transformers_version")}
    expected = ModelSchema(
        content_hash(canonical_json(logical, schema.limits.max_directory_bytes)),
        content_hash(b"qwen3-vl-native-to-hf-copy-permute-v1"),
        tuple(canonical.values()),
        schema.chunk_bytes,
        schema.limits,
    )
    if expected.schema_id != schema.schema_id:
        raise DeltaCodecError("canonical schema/config differs from the complete certified load profile")
    return LoadPlan(schema, execution_profile_id, tuple(range(tp_size)), tuple(targets), tuple(regions), budget)
