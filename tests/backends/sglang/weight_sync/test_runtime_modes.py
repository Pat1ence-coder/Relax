# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Normal IPC and overlap draining without changing SGLang execution flags."""

import importlib.util
import multiprocessing
from array import array
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.weight_sync.runtime import _Worker


def _echo_ipc(address: str, count: int) -> None:
    import zmq
    from sglang.srt.managers.io_struct import _USE_PICKLE_IPC, sock_recv, sock_send

    from relax.backends.sglang.weight_sync.wire import register_wire_types

    register_wire_types()
    with zmq.Context() as context:
        with context.socket(zmq.PAIR) as socket:
            socket.setsockopt(zmq.RCVTIMEO, 120000)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(address)
            sock_send(socket, _USE_PICKLE_IPC)
            for _ in range(count):
                sock_send(socket, sock_recv(socket))


@pytest.mark.skipif(importlib.util.find_spec("sglang") is None, reason="SGLang is required for native IPC")
def test_default_pickle_socket_preserves_work_commands_and_receipts():
    import zmq
    from sglang.srt.managers.io_struct import (
        _USE_PICKLE_IPC,
        BatchTokenizedEmbeddingReqInput,
        BatchTokenizedGenerateReqInput,
        TokenizedEmbeddingReqInput,
        TokenizedGenerateReqInput,
        sock_recv,
        sock_send,
    )
    from sglang.srt.sampling.sampling_params import SamplingParams

    from relax.backends.sglang.weight_sync.wire import DeltaCommand, DeltaReply, DeltaWork, register_wire_types

    # No environment override: both processes must use the normal default.
    assert _USE_PICKLE_IPC is True
    register_wire_types()
    common = dict(
        rid="sample",
        input_text="hello",
        input_ids=array("i", [1, 2, 3]),
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=2),
    )
    generation = TokenizedGenerateReqInput(
        **common,
        input_embeds=None,
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=True,
    )
    embedding = TokenizedEmbeddingReqInput(**common)
    payloads = (
        generation,
        embedding,
        BatchTokenizedGenerateReqInput(batch=[generation]),
        BatchTokenizedEmbeddingReqInput(batch=[embedding]),
    )
    messages = [DeltaWork(ticket="generation-1", payload=p) for p in payloads]
    messages += [
        DeltaCommand(command=b"command", argument=b"argument"),
        DeltaReply(command_id="command-1", rank=1, incarnation="worker-1", success=True, body=b"receipt"),
    ]
    with zmq.Context() as context:
        with context.socket(zmq.PAIR) as socket:
            socket.setsockopt(zmq.RCVTIMEO, 120000)
            socket.setsockopt(zmq.LINGER, 0)
            port = socket.bind_to_random_port("tcp://127.0.0.1")
            process = multiprocessing.get_context("spawn").Process(
                target=_echo_ipc, args=(f"tcp://127.0.0.1:{port}", len(messages))
            )
            process.start()
            try:
                assert sock_recv(socket) is True
                for original in messages:
                    sock_send(socket, original)
                    received = sock_recv(socket)
                    assert type(received) is type(original)
                    if isinstance(original, DeltaWork):
                        assert received.ticket == original.ticket
                        assert type(received.payload) is type(original.payload)
                        item = received.payload.batch[0] if hasattr(received.payload, "batch") else received.payload
                        assert item.rid == "sample" and item.input_ids == array("i", [1, 2, 3])
                        assert item.sampling_params.max_new_tokens == 2
                    else:
                        assert received == original
                process.join(20)
                assert process.exitcode == 0
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(20)


def test_quiesce_keeps_overlap_running_until_results_are_drained(monkeypatch):
    events = []
    queue = deque([object()])
    scheduler = SimpleNamespace(
        _engine_paused=False,
        is_fully_idle=lambda: not queue,
        flush_cache=lambda: events.append("flush") or True,
    )
    worker = _Worker.__new__(_Worker)
    worker.scheduler = scheduler
    worker.pending = SimpleNamespace(action="QUIESCE")
    worker.ticket = "old-generation"
    worker.model = torch.nn.Linear(1, 1)
    worker._reply = lambda *args: events.append("reply")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("synchronize"))
    monkeypatch.setattr(
        "relax.backends.sglang.weight_sync.runtime.clear_multimodal_execution_cache",
        lambda: events.append("multimodal_cache") or {},
    )

    worker.advance_quiesce()
    assert not scheduler._engine_paused and worker.ticket == "old-generation" and not events
    queue.popleft()
    worker.advance_quiesce()
    assert scheduler._engine_paused and worker.ticket is None
    assert events == ["synchronize", "flush", "multimodal_cache", "reply"]
