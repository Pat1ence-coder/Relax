# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Independent tiny native/HF fixtures, including non-square GQA and vision."""

import hashlib
import struct
from types import SimpleNamespace

import pytest
import torch

from relax.distributed.weight_sync import CanonicalTile, DeltaCodecError, ExportBudget, ExportRequest
from relax.distributed.weight_sync.storage import DiskSnapshotStore


def config():
    return {
        "model_type": "qwen3_vl",
        "tie_word_embeddings": True,
        "text_config": {
            "hidden_size": 6,
            "intermediate_size": 6,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "num_hidden_layers": 1,
            "vocab_size": 5,
            "hidden_act": "silu",
            "attention_bias": False,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "rope_scaling": {"mrope_section": [0, 0, 1]},
        },
        "vision_config": {
            "hidden_size": 4,
            "intermediate_size": 6,
            "num_heads": 2,
            "depth": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 6,
            "deepstack_visual_indexes": [1],
            "patch_size": 2,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "num_position_embeddings": 9,
            "hidden_act": "gelu_pytorch_tanh",
        },
    }


def bits(name, shape, dtype=torch.bfloat16):
    length = 1
    for dimension in shape:
        length *= dimension
    salt = int.from_bytes(hashlib.sha256(name.encode()).digest()[:2], "little")
    values = [(salt + i * 73) % 65536 for i in range(length)]
    special_values = (
        (0, 0x8000, 0x7F80, 0x7F81, 0x7FC1)
        if dtype == torch.bfloat16
        else (0, 0x80000000, 0x7F800000, 0x7F800001, 0x7FC00001)
    )
    for i, special in enumerate(special_values):
        if i < length:
            values[i] = special
    code = "H" if dtype == torch.bfloat16 else "I"
    return torch.frombuffer(bytearray(struct.pack("<" + code * length, *values)), dtype=dtype).reshape(shape)


def raw(tensor):
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


class TinyLanguage(torch.nn.Module):
    def __init__(self, tp):
        super().__init__()
        self.share_embeddings_and_output_weights = True
        self.config = SimpleNamespace(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            num_layers=1,
            hidden_size=6,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=2,
            ffn_hidden_size=6,
            normalization="RMSNorm",
            gated_linear_unit=True,
            qk_layernorm=True,
            add_bias_linear=False,
            add_qkv_bias=False,
            vocab_size=5,
            should_pad_vocab=True,
            make_vocab_size_divisible_by=4,
            layernorm_epsilon=1e-6,
            position_embedding_type="mrope",
            rotary_base=10000.0,
            rotary_percent=1.0,
            mrope_section=[0, 0, 1],
        )

    def shared_embedding_or_output_weight(self):
        return self.embedding.word_embeddings.weight


def add_parameter(model, name, tensor, dim, stride=1):
    parent = model
    parts = name.split(".")
    for part in parts[:-1]:
        if part not in parent._modules:
            parent.add_module(part, torch.nn.Module())
        parent = parent._modules[part]
    value = torch.nn.Parameter(tensor.contiguous().clone())
    value.tensor_model_parallel = dim >= 0
    value.partition_dim, value.partition_stride = dim, stride
    parent.register_parameter(parts[-1], value)


def native_fixture(tp, dtype=torch.bfloat16):
    """Build both sides from explicit logical rows, never the production
    planner."""
    models = [torch.nn.Module() for _ in range(tp)]
    for model in models:
        model.add_module("language_model", TinyLanguage(tp))
        model.add_module("vision_model", torch.nn.Module())
        model.vision_model.config = SimpleNamespace(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            num_layers=2,
            hidden_size=4,
            num_attention_heads=2,
            kv_channels=2,
            num_query_groups=2,
            ffn_hidden_size=6,
            patch_size=2,
            temporal_patch_size=2,
            in_channels=3,
            spatial_merge_size=2,
            num_position_embeddings=9,
            out_hidden_size=6,
            deepstack_visual_indexes=[1],
            gated_linear_unit=False,
            normalization="LayerNorm",
            layernorm_zero_centered_gamma=False,
            qk_layernorm=False,
            add_bias_linear=True,
            add_qkv_bias=True,
        )
    expected = {}

    def logical(name, shape):
        value = bits(name, shape, dtype)
        expected[name] = raw(value)
        return value

    def native(name, value, dim=-1, stride=1):
        chunks = value.chunk(tp, dim=dim) if dim >= 0 else [value] * tp
        for model, chunk in zip(models, chunks):
            add_parameter(model, name, chunk, dim, stride)

    def direct(source, target, shape, dim=-1):
        native(source, logical(target, shape), dim)

    embedding = logical("model.language_model.embed_tokens.weight", (5, 6))
    native(
        "language_model.embedding.word_embeddings.weight", torch.cat((embedding, bits("padding", (3, 6), dtype))), 0
    )
    expected["lm_head.weight"] = raw(embedding)
    direct("language_model.decoder.final_layernorm.weight", "model.language_model.norm.weight", (6,))
    source, target = "language_model.decoder.layers.0", "model.language_model.layers.0"
    q = logical(f"{target}.self_attn.q_proj.weight", (8, 6))
    k = logical(f"{target}.self_attn.k_proj.weight", (4, 6))
    v = logical(f"{target}.self_attn.v_proj.weight", (4, 6))
    native(f"{source}.self_attention.linear_qkv.weight", torch.cat((q[:4], k[:2], v[:2], q[4:], k[2:], v[2:])), 0)
    direct(f"{source}.self_attention.linear_proj.weight", f"{target}.self_attn.o_proj.weight", (6, 8), 1)
    for part in ("q", "k"):
        direct(f"{source}.self_attention.{part}_layernorm.weight", f"{target}.self_attn.{part}_norm.weight", (2,))
    direct(f"{source}.self_attention.linear_qkv.layer_norm_weight", f"{target}.input_layernorm.weight", (6,))
    direct(f"{source}.mlp.linear_fc1.layer_norm_weight", f"{target}.post_attention_layernorm.weight", (6,))
    gate, up = (logical(f"{target}.mlp.{part}_proj.weight", (6, 6)) for part in ("gate", "up"))
    for rank, model in enumerate(models):
        local = torch.cat((gate.chunk(tp)[rank], up.chunk(tp)[rank]))
        add_parameter(model, f"{source}.mlp.linear_fc1.weight", local, 0, 2)
    direct(f"{source}.mlp.linear_fc2.weight", f"{target}.mlp.down_proj.weight", (6, 6), 1)

    for i in range(2):
        source, target = f"vision_model.decoder.layers.{i}", f"model.visual.blocks.{i}"
        for suffix in ("weight", "bias"):
            shape = (12, 4) if suffix == "weight" else (12,)
            qkv = logical(f"{target}.attn.qkv.{suffix}", shape)
            # Canonical stores all Q then all K then all V. Native interleaves heads.
            native(
                f"{source}.self_attention.linear_qkv.{suffix}",
                torch.cat((qkv[0:2], qkv[4:6], qkv[8:10], qkv[2:4], qkv[6:8], qkv[10:12])),
                0,
            )
            direct(
                f"{source}.self_attention.linear_proj.{suffix}",
                f"{target}.attn.proj.{suffix}",
                (4, 4) if suffix == "weight" else (4,),
                1 if suffix == "weight" else -1,
            )
            direct(f"{source}.self_attention.linear_qkv.layer_norm_{suffix}", f"{target}.norm1.{suffix}", (4,))
            direct(f"{source}.mlp.linear_fc1.layer_norm_{suffix}", f"{target}.norm2.{suffix}", (4,))
            direct(
                f"{source}.mlp.linear_fc1.{suffix}",
                f"{target}.mlp.linear_fc1.{suffix}",
                (6, 4) if suffix == "weight" else (6,),
                0,
            )
            direct(
                f"{source}.mlp.linear_fc2.{suffix}",
                f"{target}.mlp.linear_fc2.{suffix}",
                (4, 6) if suffix == "weight" else (4,),
                1 if suffix == "weight" else -1,
            )
    direct("vision_model.patch_embed.proj.weight", "model.visual.patch_embed.proj.weight", (4, 3, 2, 2, 2))
    direct("vision_model.patch_embed.proj.bias", "model.visual.patch_embed.proj.bias", (4,))
    direct("vision_model.pos_embed.weight", "model.visual.pos_embed.weight", (9, 4))
    for source, target, norm in (
        ("vision_model.merger", "model.visual.merger", 4),
        ("vision_model.decoder.deepstack_merger_list.0", "model.visual.deepstack_merger_list.0", 16),
    ):
        for suffix in ("weight", "bias"):
            direct(f"{source}.patch_norm.{suffix}", f"{target}.norm.{suffix}", (norm,))
            direct(
                f"{source}.linear_fc1.{suffix}",
                f"{target}.linear_fc1.{suffix}",
                (16, 16) if suffix == "weight" else (16,),
                0,
            )
            direct(
                f"{source}.linear_fc2.{suffix}",
                f"{target}.linear_fc2.{suffix}",
                (6, 16) if suffix == "weight" else (6,),
                1 if suffix == "weight" else -1,
            )
    return models, expected


def capture_fixture(backend, path, tp, tile_bytes, dtype=torch.bfloat16):
    profile = backend.profiles.qwen3_vl_profile(
        config(), tp_size=tp, padded_vocab_size=8, chunk_bytes=32, dtype=str(dtype).removeprefix("torch.")
    )
    models, expected = native_fixture(tp, dtype)
    inventories = [backend.inventory.inspect_inventory(model, profile, allow_cpu=True) for model in models]
    plans = [inventory.export_plan() for inventory in inventories]
    assert all(plan.plan_id == plans[0].plan_id for plan in plans)
    source_plan = plans[0]
    request = ExportRequest("test", "epoch", 0, 0, "fixture")
    budget = ExportBudget(tile_bytes=tile_bytes)
    mappings = {mapping.target.name: mapping for mapping in profile.mappings}
    assert set(expected) == {entry.tensor.name for entry in profile.schema.tensors}
    with DiskSnapshotStore(path, max_bytes=100000, max_generations=1) as store:
        with store.capture(source_plan, request, budget) as candidate:
            for spec in profile.schema.iter_chunks():
                for offset in range(0, spec.byte_length, tile_bytes):
                    size = min(tile_bytes, spec.byte_length - offset)
                    output = bytearray(size)
                    mapping = mappings[spec.tensor.name]
                    width = spec.tensor.element_size
                    for span in backend.tile_plan.iter_source_spans(
                        mapping, tp, (spec.byte_offset + offset) // width, size // width
                    ):
                        output[span.target_offset * width : (span.target_offset + span.elements) * width] = (
                            inventories[span.rank].read_span(mapping.source.name, span.source_offset, span.elements)
                        )
                    assert (
                        bytes(output)
                        == expected[spec.tensor.name][spec.byte_offset + offset : spec.byte_offset + offset + size]
                    )
                    candidate.write_tile(
                        CanonicalTile(request.request_id, source_plan.plan_id, 0, spec, offset, bytes(output))
                    )
            frozen = candidate.finalize(source_plan.expected_receipts(request))
        identity = frozen.snapshot.identity
        for spec in profile.schema.iter_chunks():
            assert (
                frozen.read_chunk(spec)
                == expected[spec.tensor.name][spec.byte_offset : spec.byte_offset + spec.byte_length]
            )
        frozen.close()
    return identity


@pytest.mark.parametrize("tile_bytes", [2, 10, 24, 32])
def test_tp_layout_and_tile_size_preserve_entire_model(backend, tmp_path, tile_bytes):
    first = capture_fixture(backend, tmp_path / "tp1", 1, tile_bytes)
    second = capture_fixture(backend, tmp_path / "tp2", 2, tile_bytes)
    assert first == second


def test_explicit_qkv_row_vectors_and_non_square_output(backend):
    profile = backend.profiles.qwen3_vl_profile(config(), tp_size=2, padded_vocab_size=8)
    mappings = {mapping.target.name: mapping for mapping in profile.mappings}
    for part, rows in (("q", [0, 1, 2, 3, 8, 9, 10, 11]), ("k", [4, 5, 12, 13]), ("v", [6, 7, 14, 15])):
        mapping = mappings[f"model.language_model.layers.0.self_attn.{part}_proj.weight"]
        assert [backend.tile_plan._source_row(mapping, row, 2) for row in range(len(rows))] == rows
    vision = mappings["model.visual.blocks.0.attn.qkv.weight"]
    assert [backend.tile_plan._source_row(vision, row, 2) for row in range(12)] == [
        0,
        1,
        6,
        7,
        2,
        3,
        8,
        9,
        4,
        5,
        10,
        11,
    ]
    assert mappings["model.language_model.layers.0.self_attn.o_proj.weight"].target.shape == (6, 8)


def test_fp32_tiles_preserve_signed_zero_and_nan_payloads(backend, tmp_path):
    assert capture_fixture(backend, tmp_path / "tp1", 1, 12, torch.float32) == capture_fixture(
        backend, tmp_path / "tp2", 2, 12, torch.float32
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "extra",
        "buffer",
        "dtype",
        "shape",
        "stride",
        "dim",
        "partial_alias",
        "fake_tie",
        "noncontiguous",
        "cpu",
    ],
)
def test_inventory_rejects_unproven_source_layout(backend, case):
    profile = backend.profiles.qwen3_vl_profile(config(), tp_size=2, padded_vocab_size=8)
    models, _ = native_fixture(2)
    model = models[0]
    mlp = model.language_model.decoder.layers._modules["0"].mlp
    weight = mlp.linear_fc1.weight
    if case == "missing":
        del model.vision_model.pos_embed.weight
    elif case == "extra":
        model.register_parameter("unknown", torch.nn.Parameter(torch.ones(1)))
    elif case == "buffer":
        model.register_buffer("persistent_unknown", torch.ones(1))
    elif case == "dtype":
        weight.data = weight.data.float()
    elif case == "shape":
        weight.data = weight.data[:1]
    elif case == "stride":
        weight.partition_stride = 1
    elif case == "dim":
        weight.partition_dim = 1
    elif case == "partial_alias":
        # Preserve expected shape with a larger backing storage shared by two valid parameters.
        shared = torch.empty(40, dtype=torch.bfloat16)
        weight.data = shared[:36].view(6, 6)
        model.vision_model.pos_embed.weight.data = shared[2:38].view(9, 4)
    elif case == "fake_tie":
        model.language_model.add_module("output_layer", torch.nn.Module())
        model.language_model.output_layer.register_parameter(
            "weight", torch.nn.Parameter(model.language_model.embedding.word_embeddings.weight.clone())
        )
    elif case == "noncontiguous":
        weight.data = weight.data.T
    with pytest.raises(DeltaCodecError):
        backend.inventory.inspect_inventory(model, profile, allow_cpu=case != "cpu")


def test_frozen_parameters_and_whole_tie_are_not_lost(backend):
    profile = backend.profiles.qwen3_vl_profile(config(), tp_size=1, padded_vocab_size=8)
    models, _ = native_fixture(1)
    model = models[0]
    model.vision_model.patch_embed.proj.weight.requires_grad_(False)
    model.language_model.add_module("output_layer", torch.nn.Module())
    model.language_model.output_layer.register_parameter(
        "weight", model.language_model.embedding.word_embeddings.weight
    )
    inventory = backend.inventory.inspect_inventory(model, profile, allow_cpu=True)
    assert "vision_model.patch_embed.proj.weight" in inventory.tensors
    assert "language_model.output_layer.weight" not in inventory.tensors
    assert (
        next(entry for entry in profile.schema.tensors if entry.tensor.name == "lm_head.weight").alias_of
        == "model.language_model.embed_tokens.weight"
    )


def test_only_known_nonpersistent_rotary_cache_is_excluded(backend):
    class Qwen3VLVisionRotaryEmbedding(torch.nn.Module):
        pass

    profile = backend.profiles.qwen3_vl_profile(config(), tp_size=1, padded_vocab_size=8)
    models, _ = native_fixture(1)
    model = models[0]
    before = backend.inventory.inspect_inventory(model, profile, allow_cpu=True)
    rotary = Qwen3VLVisionRotaryEmbedding()
    rotary.dim, rotary.theta = 1, 10000.0
    model.vision_model.add_module("rotary_pos_emb", rotary)
    rotary.register_buffer("inv_freq", torch.ones(1), persistent=False)
    after = backend.inventory.inspect_inventory(model, profile, allow_cpu=True)
    assert before.source_layout_id == after.source_layout_id
    assert after.excluded_buffers == ("vision_model.rotary_pos_emb.inv_freq",)
    rotary._non_persistent_buffers_set.clear()
    with pytest.raises(DeltaCodecError, match="buffer"):
        backend.inventory.inspect_inventory(model, profile, allow_cpu=True)


@pytest.mark.parametrize("case", ["tp", "vocab", "tie", "moe", "quantized", "deepstack"])
def test_unsupported_profile_fails_before_tensor_access(backend, case):
    cfg, tp, padded = config(), 2, 8
    if case == "tp":
        tp = 4
    elif case == "vocab":
        padded = 4
    elif case == "tie":
        del cfg["tie_word_embeddings"]
    elif case == "moe":
        cfg["text_config"]["num_experts"] = 4
    elif case == "quantized":
        cfg["quantization_config"] = {}
    else:
        cfg["vision_config"]["deepstack_visual_indexes"] = [2]
    with pytest.raises(DeltaCodecError):
        backend.profiles.qwen3_vl_profile(cfg, tp_size=tp, padded_vocab_size=padded)


def test_padding_is_a_source_layout_detail(backend):
    a = backend.profiles.qwen3_vl_profile(config(), tp_size=1, padded_vocab_size=5)
    b = backend.profiles.qwen3_vl_profile(config(), tp_size=2, padded_vocab_size=8)
    assert a.schema == b.schema


@pytest.mark.parametrize("case", ["vision_heads", "deepstack", "vision_glu", "vocab", "rope", "strided_tie"])
def test_shape_preserving_semantic_drift_is_rejected(backend, case):
    profile = backend.profiles.qwen3_vl_profile(config(), tp_size=1, padded_vocab_size=8)
    models, _ = native_fixture(1)
    model = models[0]
    if case == "vision_heads":
        model.vision_model.config.num_attention_heads = 1
    elif case == "deepstack":
        model.vision_model.config.deepstack_visual_indexes = [0]
    elif case == "vision_glu":
        model.vision_model.config.gated_linear_unit = True
    elif case == "vocab":
        model.language_model.config.vocab_size = 6
    elif case == "rope":
        model.language_model.config.rotary_base = 20000.0
    else:
        embedding = model.language_model.embedding.word_embeddings.weight
        model.language_model.add_module("output_layer", torch.nn.Module())
        model.language_model.output_layer.register_parameter(
            "weight", torch.nn.Parameter(embedding.as_strided((8, 6), (1, 8)))
        )
    with pytest.raises(DeltaCodecError):
        backend.inventory.inspect_inventory(model, profile, allow_cpu=True)


def test_tp1_adapter_uses_boundary_and_owned_capture(backend, tmp_path):
    profile = backend.profiles.qwen3_vl_profile(config(), tp_size=1, padded_vocab_size=8, chunk_bytes=32)
    models, expected = native_fixture(1)
    boundary = backend.boundary.TrainingBoundary(models[0], run_epoch="epoch")
    boundary.mark_synchronized()
    exporter = backend.exporter.MegatronSnapshotExporter(models[0], profile, boundary, allow_cpu=True)
    with DiskSnapshotStore(tmp_path / "store", max_bytes=100000, max_generations=2) as store:
        result = exporter.capture(
            ExportRequest("test", "epoch", 4, 0, "fixture"), store, budget=ExportBudget(tile_bytes=10)
        )
        assert result.max_tile_bytes == 10 and result.payload_broadcast_bytes == 0
        for spec in profile.schema.iter_chunks():
            assert (
                result.frozen.read_chunk(spec)
                == expected[spec.tensor.name][spec.byte_offset : spec.byte_offset + spec.byte_length]
            )
        assert boundary._lease is None
        boundary.begin_update()
        with torch.no_grad():
            models[0].language_model.embedding.word_embeddings.weight.zero_()
        boundary.end_update(True)
        boundary.mark_synchronized()
        second = exporter.capture(
            ExportRequest("test", "epoch", 5, 1, "fixture"), store, budget=ExportBudget(tile_bytes=10)
        )
        assert second.identity.target_root != result.identity.target_root
        for spec in profile.schema.iter_chunks():
            assert (
                result.frozen.read_chunk(spec)
                == expected[spec.tensor.name][spec.byte_offset : spec.byte_offset + spec.byte_length]
            )
        result.frozen.close()
        second.frozen.close()


def test_successful_step_counter_and_source_fence(backend):
    model = torch.nn.Linear(2, 2)
    boundary = backend.boundary.TrainingBoundary(model, run_epoch="epoch")
    request = ExportRequest("test", "epoch", 17, 0, "fixture")
    with pytest.raises(DeltaCodecError, match="synchronized"):
        boundary.acquire(request)
    for success in (True, False, True):
        boundary.begin_update()
        boundary.end_update(success)
    assert boundary.source_step == 2
    boundary.mark_synchronized()
    with pytest.raises(DeltaCodecError, match="step"):
        boundary.acquire(request)
    lease = boundary.acquire(ExportRequest("test", "epoch", 17, 2, "fixture"))
    with pytest.raises(DeltaCodecError, match="capture"):
        boundary.begin_update()
    lease.validate()
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(DeltaCodecError, match="changed"):
        lease.validate()
    lease.release()
    lease.release()
    with pytest.raises(DeltaCodecError, match="invalid"):
        boundary.mark_synchronized()


def test_capture_rejects_boundary_of_another_model(backend, tmp_path):
    models, _ = native_fixture(1)
    other, _ = native_fixture(1)
    profile = backend.profiles.qwen3_vl_profile(config(), tp_size=1, padded_vocab_size=8)
    boundary = backend.boundary.TrainingBoundary(other[0], run_epoch="epoch")
    boundary.mark_synchronized()
    exporter = backend.exporter.MegatronSnapshotExporter(models[0], profile, boundary, allow_cpu=True)
    with DiskSnapshotStore(tmp_path / "store", max_bytes=100000, max_generations=1) as store:
        with pytest.raises(DeltaCodecError, match="different model"):
            exporter.capture(ExportRequest("test", "epoch", 0, 0, "fixture"), store)
    assert boundary._lease is None


def test_optimizer_step_counts_success_and_poison_on_uncertain_failure(backend):
    boundary = backend.boundary.TrainingBoundary(torch.nn.Linear(2, 2), run_epoch="epoch")
    for success in (True, False, True):
        optimizer = SimpleNamespace(step=lambda: (success, 1.0, 0))
        assert boundary.optimizer_step(optimizer) == (success, 1.0, 0)
    assert boundary.source_step == 2

    def failing_step():
        raise RuntimeError("uncertain optimizer failure")

    with pytest.raises(RuntimeError, match="uncertain optimizer"):
        boundary.optimizer_step(SimpleNamespace(step=failing_step))
    with pytest.raises(DeltaCodecError, match="invalid"):
        boundary.begin_update()
