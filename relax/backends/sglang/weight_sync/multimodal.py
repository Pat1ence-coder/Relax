# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Ownership of processor allocations that have not reached scheduler IPC."""

import asyncio
import contextvars
from functools import partial
from typing import Any, Callable

from relax.distributed.weight_sync import DeltaCodecError


def track_multimodal_processor(processor: Any, get_resources: Callable) -> None:
    """Track allocations in this opt-in tokenizer, including worker threads."""
    from sglang.srt.utils.cuda_ipc_transport_utils import CudaIpcTensorTransportProxy

    process = processor.process_mm_data_async
    wrap = processor._wrap_tensor_for_cuda_ipc

    async def tracked_process(*args, **kwargs):
        resources = get_resources()
        coroutine = process(*args, **kwargs)
        return await resources.process(coroutine) if resources is not None else await coroutine

    def tracked_wrap(tensor):
        proxy = wrap(tensor)
        resources = get_resources()
        if resources is not None and isinstance(proxy, CudaIpcTensorTransportProxy):
            import torch

            stream = torch.cuda.current_stream(tensor.device)

            def resend():
                destination = torch.cuda.current_stream(tensor.device)
                destination.wait_stream(stream)
                replacement = tracked_wrap(tensor)
                tensor.record_stream(destination)
                return replacement

            resources.own(proxy, stream, resend if resources.repeat_inputs else None)
        return proxy

    processor.process_mm_data_async = tracked_process
    processor._wrap_tensor_for_cuda_ipc = tracked_wrap
    executor = processor.mm_processor_executor
    if executor is not None:
        run = executor.run

        async def tracked_run(function, *args, **kwargs):
            context = contextvars.copy_context()
            return await run(partial(context.run, function), *args, **kwargs)

        executor.run = tracked_run


def _proxies(inputs: Any):
    from sglang.srt.utils.cuda_ipc_transport_utils import CudaIpcTensorTransportProxy

    def visit(value):
        if isinstance(value, CudaIpcTensorTransportProxy):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from visit(item)

    for item in getattr(inputs, "mm_items", ()):
        yield from visit(item.feature)
        yield from visit(item.precomputed_embeddings)
        yield from visit(item.model_specific_data)


class UnsentMultimodalResources:
    """Keep preprocessing alive until its untransferred pool slices can retire.

    Ownership is per producer proxy object, not SHM name: pool recycling reuses
    names. Parallel samples get fresh slices after the first send; each slice
    transfers once. A send attempt transfers even if its outcome is uncertain.
    """

    def __init__(self, tp_size: int, *, repeat_inputs: bool = False):
        self.tp_size = tp_size
        self.repeat_inputs = repeat_inputs
        self.tasks = []
        self.owned = {}
        self.transferred = set()
        self.cleanup = None

    async def process(self, coroutine: Any) -> Any:
        # A sibling tokenizer task may finish after another batch item failed.
        # Closing request ownership must also prevent late allocations.
        if self.cleanup is not None:
            coroutine.close()
            raise DeltaCodecError("multimodal request ownership is already closing")
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)

        # Native thread-pool work can finish allocating after cancellation of
        # its awaiter. Retain its result so close() can retire those allocations.
        return await asyncio.shield(task)

    def own(self, proxy: Any, stream: Any, resend: Any = None) -> None:
        # Worker registration finishes before its tracked task completes.
        self.owned[id(proxy)] = (proxy, stream, resend)

    def prepare(self, inputs: Any) -> None:
        """Give repeated parallel samples independent pool-consumption
        counts."""
        replacements = {}

        def visit(value):
            key = id(value)
            if key in self.transferred and key in self.owned:
                if key not in replacements:
                    resend = self.owned[key][2]
                    if resend is None:
                        raise DeltaCodecError("multimodal IPC allocation cannot be sent twice")
                    replacements[key] = resend()
                return replacements[key]
            if isinstance(value, list):
                return [visit(item) for item in value]
            if isinstance(value, tuple):
                return tuple(visit(item) for item in value)
            if isinstance(value, dict):
                return {name: visit(item) for name, item in value.items()}
            return value

        for item in getattr(inputs, "mm_items", ()):
            item.feature = visit(item.feature)
            item.precomputed_embeddings = visit(item.precomputed_embeddings)
            item.model_specific_data = visit(item.model_specific_data)

    def transfer(self, inputs: Any) -> None:
        self.transferred.update(id(proxy) for proxy in _proxies(inputs) if id(proxy) in self.owned)

    async def close(self) -> None:
        if self.cleanup is None:
            self.cleanup = asyncio.create_task(self._close())
        cancelled = False
        while not self.cleanup.done():
            try:
                await asyncio.shield(self.cleanup)
            except asyncio.CancelledError:
                cancelled = True
        self.cleanup.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self) -> None:
        await asyncio.gather(*self.tasks, return_exceptions=True)
        abandoned = [value for key, value in self.owned.items() if key not in self.transferred]
        if abandoned:
            import torch

            # Exceptional cleanup only: preprocessing enqueues nonblocking
            # copies on worker threads. Complete producer work before making
            # its unsent slices recyclable. Normal sends introduce no waits.
            streams = {stream for _, stream, _ in abandoned}
            for stream in streams:
                event = torch.cuda.Event()
                event.record(stream)
                await asyncio.to_thread(event.synchronize)
            for proxy, _, _ in abandoned:
                proxy.acknowledge_consumption(consumer_count=self.tp_size)
        self.owned.clear()
        self.tasks.clear()
