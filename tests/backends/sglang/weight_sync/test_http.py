# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""HTTP controls cannot bypass the tokenizer by mutating global settings."""

import asyncio
from types import SimpleNamespace

import pytest

from relax.backends.sglang.weight_sync.http import ConsumerHTTP


@pytest.mark.parametrize(
    "path",
    [
        "/continue_generation",
        "/update_weight_version",
        "/resume_memory_occupation",
        "/update_weights_from_disk",
        "/set_internal_state",
    ],
)
def test_legacy_http_handler_never_runs_even_when_consumer_is_active(path):
    calls, sent = [], []

    async def application(scope, receive, send):
        calls.append(scope)

    async def send(value):
        sent.append(value)

    adapter = ConsumerHTTP(application, SimpleNamespace(delta_ticket="current", delta_poisoned=False))
    asyncio.run(adapter({"type": "http", "path": path}, None, send))
    assert not calls and sent[0]["status"] == 409


@pytest.mark.parametrize("active", [False, True])
def test_native_http_route_uses_current_admission(active):
    calls, sent = [], []

    async def application(scope, receive, send):
        calls.append(scope)

    async def send(value):
        sent.append(value)

    adapter = ConsumerHTTP(
        application, SimpleNamespace(delta_ticket="current" if active else None, delta_poisoned=False)
    )
    asyncio.run(adapter({"type": "http", "path": "/generate"}, None, send))
    assert bool(calls) is active
    assert bool(sent) is not active


def test_lifespan_uses_consumer_ownership_and_closes_admission():
    events = iter(({"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}))
    sent, stopped = [], []
    manager = SimpleNamespace(delta_ticket="active", delta_poisoned=False)

    async def application(*args):
        raise AssertionError("upstream server warmup must not run")

    async def receive():
        return next(events)

    async def send(value):
        sent.append(value)

    adapter = ConsumerHTTP(application, manager, lambda: stopped.append(True))
    asyncio.run(adapter({"type": "lifespan"}, receive, send))
    assert stopped == [True] and manager.delta_ticket is None and manager.delta_poisoned
    assert [item["type"] for item in sent] == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
