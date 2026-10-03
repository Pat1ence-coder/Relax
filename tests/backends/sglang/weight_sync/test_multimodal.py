# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Multimodal admission must precede allocations owned by the receiver."""

import asyncio
import contextvars
import importlib.util
from multiprocessing import shared_memory
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.weight_sync import runtime
from relax.distributed.weight_sync import DeltaCodecError


pytestmark = pytest.mark.skipif(importlib.util.find_spec("sglang") is None, reason="SGLang is required")


@pytest.fixture
def tokenizer(monkeypatch):
    from sglang.srt.entrypoints import engine

    # Capture the real adapter class without constructing an engine or model.
    monkeypatch.setattr(
        engine,
        "init_tokenizer_manager",
        lambda *args, TokenizerManagerClass: TokenizerManagerClass.__new__(TokenizerManagerClass),
    )
    manager = runtime.init_delta_tokenizer(None, None, runtime_config=None)
    manager.delta_ticket = "current"
    manager.delta_poisoned = False
    return manager


@pytest.mark.parametrize(
    "ticket,active,poisoned",
    [("old", "current", False), ("old", None, False), (None, "current", False), ("current", "current", True)],
)
def test_closed_generation_allocates_no_multimodal_shared_memory(tokenizer, monkeypatch, ticket, active, poisoned):
    from sglang.srt.managers.mm_utils import ShmPointerMMData
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    allocated = []

    def send_one(self, request):
        # Model the native ordering: allocate a real segment, then dispatch.
        pointer = ShmPointerMMData(torch.arange(8, dtype=torch.float32))
        allocated.append(pointer.shm_name)
        self._dispatch_to_scheduler(request)

    monkeypatch.setattr(TokenizerManager, "_send_one_request", send_one)
    # A text-shaped typed request suffices to exercise the final version gate.
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.sampling.sampling_params import SamplingParams

    request = TokenizedGenerateReqInput(
        rid="image-request",
        input_text="image",
        input_ids=[1],
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1),
        input_embeds=None,
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )
    tokenizer.delta_ticket, tokenizer.delta_poisoned = active, poisoned
    token = runtime._TOKENIZER_TICKET.set(ticket)
    try:
        with pytest.raises(DeltaCodecError, match="generation changed"):
            tokenizer._send_one_request(request)
        assert not allocated, "rejected work allocated SHM with no receiving owner"
    finally:
        runtime._TOKENIZER_TICKET.reset(token)
        for name in allocated:
            segment = shared_memory.SharedMemory(name=name)
            segment.unlink()
            segment.close()


@pytest.mark.parametrize("batch", [False, True])
def test_current_generation_reaches_native_dispatch(tokenizer, monkeypatch, batch):
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    calls = []
    method = "_send_batch_request" if batch else "_send_one_request"
    monkeypatch.setattr(TokenizerManager, method, lambda self, obj: calls.append(obj))
    request = [object()] if batch else object()
    token = runtime._TOKENIZER_TICKET.set("current")
    try:
        getattr(tokenizer, method)(request)
    finally:
        runtime._TOKENIZER_TICKET.reset(token)
    assert calls == [request]


def test_closed_generation_rejects_batch_before_serialization(tokenizer, monkeypatch):
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    calls = []
    monkeypatch.setattr(TokenizerManager, "_send_batch_request", lambda self, obj: calls.append(obj))
    token = runtime._TOKENIZER_TICKET.set("old")
    try:
        with pytest.raises(DeltaCodecError, match="generation changed"):
            tokenizer._send_batch_request([object()])
    finally:
        runtime._TOKENIZER_TICKET.reset(token)
    assert not calls


def test_quiesce_invalidates_weight_dependent_vision_embeddings(monkeypatch):
    from sglang.srt.managers import mm_utils
    from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache

    cache = MultiModalStaticCache(1024)
    cache.set(123, EmbeddingResult(embedding=torch.ones(4, 4)))
    monkeypatch.setattr(mm_utils, "embedding_cache", cache)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    idle, replies = [], []
    worker = runtime._Worker.__new__(runtime._Worker)
    worker.scheduler = SimpleNamespace(
        _engine_paused=False, is_fully_idle=lambda: bool(idle), flush_cache=lambda: True
    )
    worker.pending = SimpleNamespace(action="QUIESCE")
    worker.model = torch.nn.Linear(1, 1)
    worker.ticket = "old-generation"
    worker._reply = lambda *args: replies.append(args)

    worker.advance_quiesce()
    assert cache.has(123) and not replies
    idle.append(True)
    worker.advance_quiesce()
    assert not cache.has(123), "new weights would reuse old image embeddings"
    assert cache.current_size == 0 and cache.max_size == 1024
    assert mm_utils.embedding_cache is cache
    assert replies[0][1] == "QUIESCED"


@pytest.fixture
def proxy_factory(monkeypatch):
    from sglang.srt.utils.cuda_ipc_transport_utils import CudaIpcTensorTransportProxy

    events = []

    class Event:
        def record(self, stream):
            events.append(("record", stream))

        def synchronize(self):
            events.append(("synchronize",))

    monkeypatch.setattr(torch.cuda, "Event", Event)

    def make(label):
        proxy = CudaIpcTensorTransportProxy.__new__(CudaIpcTensorTransportProxy)
        # Deliberately reuse a pool handle across distinct allocations.
        proxy.sync_data_meta = {"handle": "reused-slot"}
        proxy.acknowledge_consumption = lambda consumer_count: events.append(("ack", label, consumer_count))
        return proxy

    return make, events


def test_only_untransferred_allocations_retire_even_when_pool_names_are_reused(proxy_factory):
    from relax.backends.sglang.weight_sync.multimodal import UnsentMultimodalResources

    make, events = proxy_factory
    shared, reused, foreign = make("shared"), make("reused"), make("foreign")
    resources = UnsentMultimodalResources(4)
    resources.own(shared, "producer-stream")
    item = SimpleNamespace(feature=shared, precomputed_embeddings=None, model_specific_data={})
    # Parallel samples share one allocation; a send attempt is enough to
    # prevent producer-side retirement, even when transport success is unknown.
    resources.transfer(SimpleNamespace(mm_items=[item, item]))
    resources.own(reused, "producer-stream")
    resources.transfer(
        SimpleNamespace(
            mm_items=[SimpleNamespace(feature=foreign, precomputed_embeddings=None, model_specific_data={})]
        )
    )
    asyncio.run(resources.close())
    assert events == [("record", "producer-stream"), ("synchronize",), ("ack", "reused", 4)]


def test_cancelled_preprocessing_finishes_cleanup_despite_repeated_cancellation(proxy_factory):
    from relax.backends.sglang.weight_sync.multimodal import UnsentMultimodalResources

    make, events = proxy_factory

    async def run():
        resources = UnsentMultimodalResources(2)
        started, release = asyncio.Event(), asyncio.Event()

        async def process():
            started.set()
            await release.wait()
            resources.own(make("cancelled"), "worker-stream")
            return None

        async def request():
            try:
                await resources.process(process())
            finally:
                await resources.close()

        task = asyncio.create_task(request())
        await started.wait()
        task.cancel()
        while resources.cleanup is None:
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert resources.cleanup.done() and not resources.owned

    asyncio.run(run())
    assert events[-1] == ("ack", "cancelled", 2)


def test_processor_thread_allocations_retire_if_processing_raises(proxy_factory, monkeypatch):
    from relax.backends.sglang.weight_sync.multimodal import UnsentMultimodalResources, track_multimodal_processor

    make, events = proxy_factory
    current = contextvars.ContextVar("test_multimodal_resources", default=None)
    resources = UnsentMultimodalResources(2)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "actual-worker-stream")

    class Executor:
        async def run(self, function):
            return await asyncio.get_running_loop().run_in_executor(None, function)

    processor = SimpleNamespace(
        mm_processor_executor=Executor(), _wrap_tensor_for_cuda_ipc=lambda tensor: make("error")
    )

    def worker():
        processor._wrap_tensor_for_cuda_ipc(SimpleNamespace(device="cuda"))
        raise ValueError("processor failed after allocating")

    async def process():
        return await processor.mm_processor_executor.run(worker)

    processor.process_mm_data_async = process
    track_multimodal_processor(processor, current.get)

    async def run():
        token = current.set(resources)
        try:
            with pytest.raises(ValueError, match="after allocating"):
                await processor.process_mm_data_async()
        finally:
            current.reset(token)
            await resources.close()

    asyncio.run(run())
    assert events == [("record", "actual-worker-stream"), ("synchronize",), ("ack", "error", 2)]


def test_late_batch_sibling_cannot_allocate_after_cleanup_starts():
    from relax.backends.sglang.weight_sync.multimodal import UnsentMultimodalResources

    async def run():
        resources = UnsentMultimodalResources(2)
        started, release = asyncio.Event(), asyncio.Event()

        async def processing():
            started.set()
            await release.wait()

        task = asyncio.create_task(resources.process(processing()))
        await started.wait()
        close = asyncio.create_task(resources.close())
        while resources.cleanup is None:
            await asyncio.sleep(0)

        async def late():
            raise AssertionError("closed request started another preprocessing task")

        with pytest.raises(DeltaCodecError, match="already closing"):
            await resources.process(late())
        release.set()
        await task
        await close
        with pytest.raises(DeltaCodecError, match="already closing"):
            await resources.process(late())

    asyncio.run(run())


def test_parallel_samples_get_fresh_allocations_instead_of_reusing_consumed_slices(proxy_factory):
    from relax.backends.sglang.weight_sync.multimodal import UnsentMultimodalResources

    make, events = proxy_factory
    resources = UnsentMultimodalResources(2, repeat_inputs=True)
    original = make("prefix")
    created = []

    def resend():
        proxy = make("sample")
        created.append(proxy)
        resources.own(proxy, "stream")
        return proxy

    resources.own(original, "stream", resend)
    for index in range(3):
        # Native parallel sampling shallow-copies the original item each time.
        item = SimpleNamespace(feature=original, precomputed_embeddings=None, model_specific_data={})
        inputs = SimpleNamespace(mm_items=[item])
        resources.prepare(inputs)
        resources.transfer(inputs)
        assert (item.feature is original) is (index == 0)
    assert len(created) == 2 and created[0] is not created[1]
    asyncio.run(resources.close())
    assert not events, "a potentially sent slice must only be acknowledged by its receivers"


def test_resend_retains_source_on_its_actual_copy_stream(proxy_factory, monkeypatch):
    from relax.backends.sglang.weight_sync.multimodal import UnsentMultimodalResources, track_multimodal_processor

    make, _ = proxy_factory
    resources = UnsentMultimodalResources(2, repeat_inputs=True)
    lifetime = []
    source_stream = object()
    destination = SimpleNamespace(wait_stream=lambda stream: lifetime.append(("wait", stream)))
    active_stream = [source_stream]
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: active_stream[0])
    source = SimpleNamespace(device="cuda", record_stream=lambda stream: lifetime.append(("record", stream)))
    processor = SimpleNamespace(
        process_mm_data_async=None, mm_processor_executor=None, _wrap_tensor_for_cuda_ipc=lambda tensor: make("slice")
    )
    track_multimodal_processor(processor, lambda: resources)
    original = processor._wrap_tensor_for_cuda_ipc(source)
    item = SimpleNamespace(feature=original, precomputed_embeddings=None, model_specific_data={})
    inputs = SimpleNamespace(mm_items=[item])
    resources.transfer(inputs)
    active_stream[0] = destination
    resources.prepare(inputs)
    resources.transfer(inputs)
    assert item.feature is not original
    assert lifetime == [("wait", source_stream), ("record", destination)]
    asyncio.run(resources.close())
