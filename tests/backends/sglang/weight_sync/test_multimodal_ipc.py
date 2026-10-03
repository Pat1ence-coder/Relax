# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real pickle socket, TP broadcast and shared-memory materialization."""

import importlib.util
import multiprocessing
from array import array
from datetime import timedelta
from multiprocessing import shared_memory
from types import SimpleNamespace

import pytest
import torch


def _receive_multimodal(rank, rendezvous, address, result):
    import torch.distributed as dist
    import zmq
    from sglang.srt.managers import mm_utils
    from sglang.srt.managers.io_struct import _USE_PICKLE_IPC, sock_recv
    from sglang.srt.managers.scheduler_components import request_receiver
    from sglang.srt.utils import broadcast_pyobj

    from relax.backends.sglang.weight_sync.runtime import install_scheduler_hooks

    assert _USE_PICKLE_IPC is True
    install_scheduler_hooks(None)
    # Only context fixtures: exercise native CPU SHM without loading a model.
    # Full-model tests separately retain their unmodified resolved settings.
    mm_utils._get_is_default_transport = lambda: False
    mm_utils.get_serving = lambda: SimpleNamespace(skip_tokenizer_init=False)
    request_receiver.get_parallel = lambda: SimpleNamespace(enable_dp_attention=False)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        requests = None
        if rank == 0:
            with zmq.Context() as context:
                with context.socket(zmq.PULL) as socket:
                    socket.setsockopt(zmq.RCVTIMEO, 120000)
                    socket.connect(address)
                    requests = [sock_recv(socket)]
        group = dist.group.WORLD
        requests = broadcast_pyobj(requests, rank, dist_group=group, src=0)
        receiver = SimpleNamespace(
            ps=SimpleNamespace(tp_size=2), model_config=SimpleNamespace(is_multimodal=True), tp_cpu_group=group
        )
        request_receiver.SchedulerRequestReceiver.unwrap_pickle_wrapper(receiver, requests)
        request_receiver.SchedulerRequestReceiver._finalize_shm_features(receiver, requests)
        envelope = requests[0]
        assert envelope.ticket == "image-generation"
        observed = []
        for item in envelope.payload.batch[0].mm_inputs.mm_items:
            observed.append((item.feature[0].tolist(), item.feature[1].tolist(), item.precomputed_embeddings.tolist()))
        result.send(observed)
    finally:
        result.close()
        dist.destroy_process_group()


@pytest.mark.skipif(importlib.util.find_spec("sglang") is None, reason="SGLang is required for native IPC")
def test_pickle_multimodal_envelope_survives_tp_broadcast_and_releases_shm(tmp_path):
    import zmq
    from sglang.srt.managers.io_struct import (
        _USE_PICKLE_IPC,
        BatchTokenizedGenerateReqInput,
        TokenizedGenerateReqInput,
        sock_send,
    )
    from sglang.srt.managers.mm_utils import ShmPointerMMData
    from sglang.srt.sampling.sampling_params import SamplingParams

    from relax.backends.sglang.weight_sync.wire import DeltaWork

    assert _USE_PICKLE_IPC is True
    pointers = [ShmPointerMMData(torch.arange(6).reshape(2, 3)), ShmPointerMMData(torch.ones(2, 3))]
    item = SimpleNamespace(
        feature=[pointers[0], torch.tensor([9])], precomputed_embeddings=pointers[1], model_specific_data={}
    )
    request = TokenizedGenerateReqInput(
        rid="image",
        input_text="describe",
        input_ids=array("i", [1, 2]),
        mm_inputs=SimpleNamespace(mm_items=[item]),
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1),
        input_embeds=None,
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )
    request.wrap_pickle_fields()
    envelope = DeltaWork(ticket="image-generation", payload=BatchTokenizedGenerateReqInput(batch=[request]))
    processes, readers = [], []
    context = multiprocessing.get_context("spawn")
    try:
        with zmq.Context() as transport:
            with transport.socket(zmq.PUSH) as socket:
                socket.setsockopt(zmq.SNDTIMEO, 120000)
                socket.setsockopt(zmq.LINGER, 0)
                port = socket.bind_to_random_port("tcp://127.0.0.1")
                for rank in range(2):
                    reader, writer = context.Pipe(duplex=False)
                    process = context.Process(
                        target=_receive_multimodal,
                        args=(rank, (tmp_path / "rendezvous").as_uri(), f"tcp://127.0.0.1:{port}", writer),
                    )
                    process.start()
                    writer.close()
                    readers.append(reader)
                    processes.append(process)
                sock_send(socket, envelope)
                for reader in readers:
                    assert reader.poll(120), "TP rank did not return materialized image data"
                    assert reader.recv() == [([[0, 1, 2], [3, 4, 5]], [9], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])]
        for process in processes:
            process.join(20)
            assert process.exitcode == 0
        for pointer in pointers:
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=pointer.shm_name)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(20)
        for reader in readers:
            reader.close()
        for pointer in pointers:
            try:
                segment = shared_memory.SharedMemory(name=pointer.shm_name)
            except FileNotFoundError:
                continue
            segment.unlink()
            segment.close()
