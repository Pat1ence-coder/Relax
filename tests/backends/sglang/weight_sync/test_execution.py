# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""A captured reference must observe refreshed bytes without storage
rebinding."""

from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.weight_sync.execution import _compute_rotary_cache, _refresh_cache_in_place
from relax.backends.sglang.weight_sync.inventory import TargetInventory, storage_binding
from relax.distributed.weight_sync import DeltaCodecError


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_refresh_preserves_existing_storage_references(dtype):
    cache = torch.zeros((8, 4), dtype=dtype)
    retained = cache.view(-1)
    pointer = cache.data_ptr()
    replacement = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 7
    _refresh_cache_in_place(cache, replacement)
    assert cache.data_ptr() == pointer == retained.data_ptr()
    assert torch.equal(retained, replacement.to(dtype).view(-1))


def test_rotary_refresh_refuses_shape_change_before_mutating():
    cache = torch.ones((8, 4))
    with pytest.raises(DeltaCodecError, match="captured storage"):
        _refresh_cache_in_place(cache, torch.zeros((16, 4)))
    assert torch.all(cache == 1)


def test_rotary_refresh_retains_extended_runtime_positions():
    cache = torch.zeros((12, 4), dtype=torch.bfloat16)
    module = SimpleNamespace(
        max_position_embeddings=8,
        base=1,
        cos_sin_cache=cache,
        _compute_inv_freq=lambda base: torch.ones(2),
    )
    pointer = cache.data_ptr()
    _refresh_cache_in_place(cache, _compute_rotary_cache(module, cache.shape[0]))
    assert cache.data_ptr() == pointer and module.max_position_embeddings == 8
    assert torch.equal(cache[:, 0], torch.arange(12).float().cos().bfloat16())
    assert torch.equal(cache[:, 3], torch.arange(12).float().sin().bfloat16())


def test_execution_buffer_rebinding_is_rejected_even_when_bytes_match():
    model = torch.nn.Module()
    model.register_buffer("cache", torch.ones((8, 4)), persistent=False)
    inventory = TargetInventory(
        model,
        None,
        0,
        {},
        {},
        "binding",
        ("cache",),
        {"cache": (storage_binding(model.cache), str(model.cache.dtype))},
    )
    inventory.validate_bindings()
    model.cache = model.cache.clone()
    with pytest.raises(DeltaCodecError, match="execution buffer storage"):
        inventory.validate_bindings()
    # With no captured graph, the caller may re-inventory the ordinary
    # first-forward dtype conversion before preparing a new transaction.
    inventory.validate_bindings(captured_buffers=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a CUDA device")
def test_rotary_refresh_updates_an_already_captured_graph():
    cache = torch.zeros((8, 4), device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = cache * 3
    replacement = torch.arange(32, device="cuda", dtype=torch.float32).reshape(8, 4)
    _refresh_cache_in_place(cache, replacement)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, replacement * 3)
