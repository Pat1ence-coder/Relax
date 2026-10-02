# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real CPU storage mutations exercise the same bounded byte-copy loader."""

import hashlib
from dataclasses import replace

import pytest
import torch
from test_profiles import canonical_arrays, independent_target

from relax.backends.sglang.weight_sync.inventory import inspect_inventory
from relax.backends.sglang.weight_sync.loader import PreparedLoad
from relax.backends.sglang.weight_sync.profiles import qwen3_vl_load_plan
from relax.distributed.weight_sync import DeltaCodecError, SnapshotIdentity, verify_snapshot
from relax.distributed.weight_sync.consumer import Installation, Member
from relax.distributed.weight_sync.integrity import ModelRoot
from relax.distributed.weight_sync.load import LoadBudget


class Lease:
    valid = True

    def validate(self):
        if not self.valid:
            raise DeltaCodecError("load lease is no longer valid")


def fixture(model_config, schema, tp=2, rank=1):
    plan = qwen3_vl_load_plan(
        model_config, schema, tp_size=tp, execution_profile_id="a" * 64, budget=LoadBudget(tile_bytes=30)
    )
    arrays = canonical_arrays(schema)

    class Reader:
        def read_chunk(self, spec):
            return arrays[spec.tensor.name].tobytes()[spec.byte_offset : spec.byte_offset + spec.byte_length]

    reader = Reader()
    root = ModelRoot(schema.schema_id, schema.directory_hash)
    for chunk in schema.iter_chunks():
        root.add(chunk, hashlib.sha256(reader.read_chunk(chunk)).hexdigest())
    identity = SnapshotIdentity("test", "epoch", 1, schema.schema_id, root.hexdigest())
    snapshot = verify_snapshot(schema, identity, reader)
    model = torch.nn.Module()
    for target in plan.targets:
        if target.rank != rank:
            continue
        parent = model
        parts = target.tensor.name.split(".")
        for part in parts[:-1]:
            if part not in parent._modules:
                parent.add_module(part, torch.nn.Module())
            parent = parent._modules[part]
        parent.register_parameter(
            parts[-1], torch.nn.Parameter(torch.full(target.tensor.shape, 1.0, dtype=torch.bfloat16))
        )
    if model_config["tie_word_embeddings"]:
        model.lm_head.weight = model.model.embed_tokens.weight
    inventory = inspect_inventory(model, plan, rank, allow_cpu=True)
    installation = Installation(
        "consumer", 1, "install-1", 0, identity, plan.plan_id, tuple(Member(r, f"worker-{r}") for r in range(tp))
    )
    lease = Lease()
    return PreparedLoad(snapshot, inventory, installation, lease), arrays, model, lease


def test_prepare_does_not_mutate_and_loaded_scan_checks_real_storage(model_config, exporter_schema):
    prepared, arrays, model, lease = fixture(model_config, exporter_schema)
    assert torch.all(model.model.embed_tokens.weight == 1)
    with pytest.raises(DeltaCodecError, match="actual target bytes"):
        prepared.verify_loaded()
    prepared.load()
    receipt = prepared.verify_loaded()
    assert receipt.rank == 1 and receipt.nbytes > 0
    for name, tensor in model.named_parameters(remove_duplicate=False):
        expected = independent_target(name, arrays, 1, 2, model_config).tobytes()
        assert tensor.detach().view(torch.uint8).numpy().tobytes() == expected, name
    with torch.no_grad():
        model.visual.pos_embed.weight.view(torch.uint8).view(-1)[-1] ^= 1
    with pytest.raises(DeltaCodecError, match="actual target bytes"):
        prepared.verify_loaded()
    prepared.load()
    assert prepared.verify_loaded() == receipt
    lease.valid = False
    with pytest.raises(DeltaCodecError, match="lease"):
        prepared.load()


@pytest.mark.parametrize("change", ["unknown", "alias", "replace", "buffer", "shape"])
def test_storage_rebinding_and_directory_mutation_are_rejected(model_config, exporter_schema, change):
    prepared, _, model, _ = fixture(model_config, exporter_schema)
    if change == "unknown":
        model.register_parameter("new_weight", torch.nn.Parameter(torch.ones(1)))
    elif change == "alias":
        model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight.clone())
    elif change == "replace":
        model.visual.pos_embed.weight.data = model.visual.pos_embed.weight.data.clone()
    elif change == "buffer":
        model.register_buffer("unknown", torch.ones(1))
    else:
        model.visual.pos_embed.weight.data = model.visual.pos_embed.weight.data.view(-1)
    with pytest.raises(DeltaCodecError):
        prepared.load()


def test_installation_identity_is_required_before_writing(model_config, exporter_schema):
    prepared, _, _, _ = fixture(model_config, exporter_schema)
    with pytest.raises(DeltaCodecError, match="identity"):
        PreparedLoad(
            prepared.snapshot, prepared.inventory, replace(prepared.installation, plan_id="f" * 64), prepared.lease
        )
