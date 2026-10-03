# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Explicit SGLang startup hooks with default-closed request admission.

No existing Relax launcher or default full-sync path is modified. Every spawned
scheduler installs its own hooks, checks execution settings, and creates a
fresh nonce. The private install channel correlates complete rank receipts by
command digest.
"""

import asyncio
import contextvars
import json
import os
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from functools import partial
from pathlib import Path
from typing import Any

from relax.distributed.weight_sync import DeltaCodecError, ModelSchema
from relax.distributed.weight_sync.codec.format import content_hash
from relax.distributed.weight_sync.consumer import Installation, InstallCommand, RankReceipt, certify_receipts
from relax.distributed.weight_sync.load import LoadBudget
from relax.distributed.weight_sync.serialization import canonical_json, parse_json, require_identifier
from relax.distributed.weight_sync.storage import open_snapshot
from relax.distributed.weight_sync.storage.consumer import installation_from_dict

from .capabilities import execution_profile
from .execution import (
    clear_multimodal_execution_cache,
    refresh_execution_state,
    validate_model_config,
    verify_execution_state,
)
from .inventory import inspect_inventory
from .isolation import arm_parent_death, process_identity
from .loader import PreparedLoad
from .multimodal import UnsentMultimodalResources, track_multimodal_processor
from .profiles import qwen3_vl_load_plan


_TOKENIZER_TICKET = contextvars.ContextVar("relax_delta_tokenizer_ticket", default=None)
_SCHEDULER_TICKET = contextvars.ContextVar("relax_delta_scheduler_ticket", default=None)
_TOKENIZER_RESOURCES = contextvars.ContextVar("relax_delta_tokenizer_resources", default=None)


@dataclass(frozen=True)
class RuntimeConfig:
    consumer_id: str
    snapshot_root: str
    schema_bytes: bytes
    model_config_json: str
    budget: LoadBudget = LoadBudget()
    max_runtime_refresh_bytes: int = 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        require_identifier(self.consumer_id, "consumer ID")
        if not Path(self.snapshot_root).is_absolute():
            raise DeltaCodecError("snapshot root must be an explicit absolute runtime path")
        ModelSchema.from_bytes(self.schema_bytes)
        if not isinstance(json.loads(self.model_config_json), dict):
            raise DeltaCodecError("runtime requires a logical model configuration")


def _profile(server_args: Any, *, startup: bool = False) -> str:
    from sglang.srt.environ import envs
    from sglang.srt.managers import io_struct
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        _unsupported_derived_weight_cache_error,
    )
    from sglang.srt.plugins.hook_registry import HookRegistry
    from sglang.srt.runtime_context import get_context

    if (
        envs.SGLANG_RUST_SERVER.get()
        or envs.SGLANG_TEST_SCRIPTED_RUNTIME.get()
        or envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get()
        or HookRegistry._hooks
    ):
        raise DeltaCodecError("external model, plugin hooks and alternative schedulers are unsupported")
    resolved = asdict(server_args) if startup else get_context().resolved_server_args_dict()
    if is_dataclass(resolved.get("cuda_graph_config")):
        resolved["cuda_graph_config"] = asdict(resolved["cuda_graph_config"])
    return execution_profile(
        resolved,
        vit_graph=envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get(),
        # Socket serialization caches this value when io_struct is imported.
        # A later environment change must not misreport the active transport.
        pickle_ipc=io_struct._USE_PICKLE_IPC,
        derived_weight_cache=_unsupported_derived_weight_cache_error() is not None,
    )


def _command_from_bytes(data: bytes) -> InstallCommand:
    record = parse_json(data, 1024 * 1024)
    record["installation"] = installation_from_dict(record["installation"])
    return InstallCommand(**record)


class _WorkerLease:
    def __init__(self, worker: "_Worker", installation: Installation):
        self.worker, self.installation = worker, installation

    def validate(self) -> None:
        worker = self.worker
        if (
            worker.installation != self.installation
            or worker.reader is None
            or not worker.scheduler._engine_paused
            or not worker.scheduler.is_fully_idle()
            or worker.pending is None
        ):
            raise DeltaCodecError("install execution/snapshot lease is not held")


class _Worker:
    def __init__(self, scheduler: Any, config: RuntimeConfig):
        import zmq
        from sglang.srt.managers.io_struct import sock_send
        from sglang.srt.utils.network import get_zmq_socket

        self.scheduler, self.config = scheduler, config
        self.rank = scheduler.ps.tp_rank
        self.incarnation = uuid.uuid4().hex
        self.profile_id = _profile(scheduler.server_args)
        self.model = scheduler.tp_worker.model_runner.model
        self.model_config = json.loads(config.model_config_json)
        validate_model_config(self.model, self.model_config)
        self.plan = qwen3_vl_load_plan(
            self.model_config,
            ModelSchema.from_bytes(config.schema_bytes),
            tp_size=scheduler.server_args.tp_size,
            execution_profile_id=self.profile_id,
            budget=config.budget,
        )
        self.inventory = inspect_inventory(self.model, self.plan, self.rank)
        self.context = zmq.Context(1)
        self.socket = get_zmq_socket(self.context, zmq.PUSH, scheduler._delta_tokenizer_ipc, False)
        self.send = partial(sock_send, self.socket)
        self.phase = "CLOSED"
        self.installation = None
        self.reader = None
        self.snapshot = None
        self.prepared = None
        self.pending = None
        self.last_command = None
        self.last_reply = None
        self.sequence = -1
        self.fence = 0
        self.ticket = None
        self.execution_id = None
        scheduler._engine_paused = True

    def info(self) -> dict:
        return {
            "rank": self.rank,
            "incarnation": self.incarnation,
            "profile_id": self.profile_id,
            "plan_id": self.plan.plan_id,
            "closed": self.scheduler._engine_paused,
            "consumer_id": self.config.consumer_id,
            "process": process_identity(os.getpid()),
        }

    def _reply(self, command: InstallCommand, phase: str, evidence: dict) -> None:
        from .wire import DeltaReply

        receipt = RankReceipt(
            command.digest, self.rank, self.incarnation, phase, content_hash(canonical_json(evidence, 64 * 1024))
        )
        reply = DeltaReply(
            command_id=command.digest,
            rank=self.rank,
            incarnation=self.incarnation,
            success=True,
            body=canonical_json({"receipt": asdict(receipt), "evidence": evidence}, 64 * 1024),
        )
        self.phase, self.sequence = phase, command.sequence
        self.last_command, self.last_reply = command, reply
        self.pending = None
        self.send(reply)

    def handle(self, message: Any) -> None:
        from sglang.srt.managers.io_struct import AbortReq

        from .wire import DeltaReply

        command = _command_from_bytes(message.command)
        try:
            value = command.installation
            if (
                value.consumer_id != self.config.consumer_id
                or value.plan_id != self.plan.plan_id
                or value.members[self.rank].incarnation != self.incarnation
                or tuple(m.rank for m in value.members) != self.plan.participants
            ):
                raise DeltaCodecError("install command is not bound to this worker")
            if self.last_command == command:
                self.send(self.last_reply)
                return
            if self.pending is not None:
                if self.pending == command:
                    return
                raise DeltaCodecError("another installation operation is still running")
            if value.owner_fence < self.fence:
                raise DeltaCodecError("stale execution owner fence")
            argument = parse_json(message.argument, 64 * 1024)
            if command.action == "PREPARE":
                if (
                    command.sequence != 0
                    or command.expected_phase != self.phase
                    or self.phase not in ("CLOSED", "ACTIVE")
                    or value.owner_fence != self.fence
                    and self.fence != 0
                    or content_hash(message.argument) != command.argument_digest
                ):
                    raise DeltaCodecError(
                        "new installation requires the current owner and a stable execution boundary"
                    )
                require_identifier(argument["snapshot_ref"], "snapshot generation")
                self.pending = command
                snapshot, reader = open_snapshot(
                    Path(self.config.snapshot_root) / argument["snapshot_ref"], expected_identity=value.snapshot
                )
                if self.reader is not None:
                    self.reader.close()
                self.snapshot, self.reader = snapshot, reader
                self.installation, self.fence = value, value.owner_fence
                self.sequence = -1
                self._reply(command, "PREPARED", {"snapshot": asdict(snapshot.identity), "plan_id": self.plan.plan_id})
                return
            if (
                value != self.installation
                or command.sequence != self.sequence + 1
                or command.expected_phase != self.phase
            ):
                raise DeltaCodecError("out-of-order or stale installation command")
            self.pending = command
            if command.action == "QUIESCE":
                self.scheduler.abort_request(AbortReq(abort_all=True))
                self.advance_quiesce()
            elif command.action == "LOAD":
                if self.phase != "QUIESCED":
                    raise DeltaCodecError("LOAD requires a quiescent group member")
                self.phase = "LOADING"
                validate_model_config(self.model, self.model_config)
                # Existing text graphs retain the startup storage addresses.
                # Re-enumeration alone would accept newly rebound parameters
                # or caches while a graph continued to read the old storage.
                # Eager MRoPE may convert FP32 to BF16 on its first request.
                # Re-inventory that cache only when no graph can retain it;
                # weight bindings are always strict, including eager mode.
                runner = self.scheduler.tp_worker.model_runner
                captured = any(
                    getattr(getattr(getattr(runner, phase + "_cuda_graph_runner"), "backend", None), "_graphs", None)
                    for phase in ("prefill", "decode")
                )
                self.inventory.validate_bindings(captured_buffers=captured)
                self.inventory = inspect_inventory(self.model, self.plan, self.rank)
                self.prepared = PreparedLoad(self.snapshot, self.inventory, value, _WorkerLease(self, value))
                self.prepared.load()
                self.execution_id = refresh_execution_state(
                    self.model,
                    self.model_config,
                    max_bytes=self.config.max_runtime_refresh_bytes,
                    tile_bytes=self.config.budget.tile_bytes,
                )
                self._reply(command, "RUNTIME_READY", {"execution_id": self.execution_id})
            elif command.action == "VERIFY":
                if self.phase != "RUNTIME_READY" or self.prepared is None:
                    raise DeltaCodecError("VERIFY requires freshly loaded runtime state")
                validate_model_config(self.model, self.model_config)
                receipt = self.prepared.verify_loaded()
                verify_execution_state(
                    self.model,
                    self.model_config,
                    self.execution_id,
                    max_bytes=self.config.max_runtime_refresh_bytes,
                    tile_bytes=self.config.budget.tile_bytes,
                )
                self._reply(command, "VERIFIED", {"loaded": asdict(receipt), "execution_id": self.execution_id})
            elif command.action == "ACK_COMMIT":
                expected = content_hash(
                    canonical_json(
                        {
                            "installation": value.digest,
                            "decision": "COMMIT",
                            "operation": argument["decision_operation"],
                        },
                        4096,
                    )
                )
                if self.phase != "VERIFIED" or command.argument_digest != expected:
                    raise DeltaCodecError("invalid COMMIT decision acknowledgement")
                self._reply(command, "ADMISSION_READY", {"decision_certificate": expected})
            elif command.action == "ACTIVATE":
                ack = _command_from_bytes(canonical_json(argument["ack_command"], 1024 * 1024))
                receipts = tuple(RankReceipt(**r) for r in argument["receipts"])
                certificate = certify_receipts(ack, receipts, "ADMISSION_READY")
                if (
                    self.phase != "ADMISSION_READY"
                    or ack.installation != value
                    or ack != self.last_command
                    or certificate != command.argument_digest
                ):
                    raise DeltaCodecError("activation requires this group's current all-member certificate")
                self.ticket = command.digest
                self.scheduler._engine_paused = False
                self._reply(command, "ACTIVE", {"ticket": self.ticket})
            else:
                raise DeltaCodecError("unsupported installation command")
        except Exception as error:
            # A failed/unknown operation poisons this incarnation. The caller
            # must physically stop it before starting a replacement engine.
            self.scheduler._engine_paused = True
            self.phase = "FAILED"
            self.pending = None
            self.send(
                DeltaReply(
                    command_id=command.digest,
                    rank=self.rank,
                    incarnation=self.incarnation,
                    success=False,
                    body=canonical_json({"error": type(error).__name__, "message": str(error)}, 64 * 1024),
                )
            )

    def advance_quiesce(self) -> None:
        import torch

        if self.pending is None or self.pending.action != "QUIESCE" or not self.scheduler.is_fully_idle():
            return
        command = self.pending
        self.scheduler._engine_paused = True
        torch.cuda.synchronize(next(self.model.parameters()).device)
        if not self.scheduler.flush_cache():
            raise DeltaCodecError("scheduler did not reach a flushable quiescent state")
        # Native flush_cache resets KV/grammar state, but leaves the image
        # embedding cache (including deepstack features) tied to old weights.
        multimodal_cache = clear_multimodal_execution_cache()
        self.ticket = None
        self._reply(command, "QUIESCED", {"idle": True, "cache_flushed": True, "multimodal_cache": multimodal_cache})


def install_scheduler_hooks(config: RuntimeConfig) -> None:
    """Called inside each spawned scheduler, before construction/dispatch
    binding."""
    from sglang.srt.managers import scheduler as module
    from sglang.srt.managers.io_struct import AbortReq, ShutdownReq
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.request_receiver import SchedulerRequestReceiver

    from .wire import WORK_TYPES, DeltaCommand, DeltaWork, register_wire_types

    register_wire_types()
    if getattr(module.Scheduler, "_relax_delta_hooks", False):
        raise DeltaCodecError("scheduler hooks were already installed")
    original = module.Scheduler
    original_req_init = Req.__init__

    def req_init(self, *args, **kwargs):
        original_req_init(self, *args, **kwargs)
        self._relax_delta_ticket = _SCHEDULER_TICKET.get()

    Req.__init__ = req_init
    for name in ("unwrap_pickle_wrapper", "_finalize_shm_features"):
        old = getattr(SchedulerRequestReceiver, name)

        def unwrap(self, requests, _original=old):
            return _original(
                self, None if requests is None else [r.payload if isinstance(r, DeltaWork) else r for r in requests]
            )

        setattr(SchedulerRequestReceiver, name, unwrap)

    class DeltaScheduler(original):
        _relax_delta_hooks = True

        def __init__(self, server_args, port_args, *args, **kwargs):
            self._delta_tokenizer_ipc = port_args.tokenizer_ipc_name
            super().__init__(server_args, port_args, *args, **kwargs)
            self._delta = _Worker(self, config)

        def init_running_status(self):
            super().init_running_status()
            self._engine_paused = True

        def get_init_info(self):
            result = super().get_init_info()
            result["relax_delta"] = self._delta.info()
            return result

        def process_input_requests(self, requests):
            from sglang.srt.managers.io_struct import RpcReqInput

            for request in requests:
                if isinstance(request, DeltaCommand):
                    self._delta.handle(request)
                elif isinstance(request, DeltaWork):
                    if self._delta.phase != "ACTIVE" or request.ticket != self._delta.ticket:
                        raise DeltaCodecError("stale work envelope reached scheduler admission")
                    token = _SCHEDULER_TICKET.set(request.ticket)
                    try:
                        super().process_input_requests([request.payload])
                    finally:
                        _SCHEDULER_TICKET.reset(token)
                elif isinstance(request, (AbortReq, ShutdownReq)):
                    super().process_input_requests([request])
                elif isinstance(request, WORK_TYPES) or isinstance(request, RpcReqInput):
                    raise DeltaCodecError("unenveloped work or generic RPC cannot bypass delta admission")
                else:
                    raise DeltaCodecError("legacy scheduler control is disabled in delta mode")
            super().process_input_requests([])
            self._delta.advance_quiesce()

        def run_batch(self, batch, *args, **kwargs):
            delta = self._delta
            draining = delta.phase == "PREPARED" and delta.ticket is not None
            if (
                self._engine_paused
                or not (delta.phase == "ACTIVE" or draining)
                or any(getattr(req, "_relax_delta_ticket", None) != delta.ticket for req in batch.reqs)
            ):
                raise DeltaCodecError("batch has no current execution generation")
            return super().run_batch(batch, *args, **kwargs)

    module.Scheduler = DeltaScheduler


def run_delta_scheduler(*args: Any, runtime_config: RuntimeConfig, owner_process: dict, **kwargs: Any) -> None:
    arm_parent_death(owner_process)
    from sglang.srt.managers import scheduler

    scheduler.load_plugins()
    install_scheduler_hooks(runtime_config)
    scheduler.run_scheduler_process(*args, **kwargs)


def run_delta_detokenizer(*args: Any, owner_process: dict, **kwargs: Any) -> None:
    arm_parent_death(owner_process)
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process

    run_detokenizer_process(*args, **kwargs)


def init_delta_tokenizer(server_args: Any, port_args: Any, *, runtime_config: RuntimeConfig):
    from sglang.srt.entrypoints.engine import init_tokenizer_manager
    from sglang.srt.managers.io_struct import AbortReq, ShutdownReq
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.utils import TypeBasedDispatcher

    from .wire import WORK_TYPES, DeltaCommand, DeltaReply, DeltaWork, register_wire_types

    register_wire_types()

    class DeltaTokenizer(TokenizerManager):
        def __init__(self, *args, **kwargs):
            self.delta_ticket = None
            self.delta_pending = None
            self.delta_poisoned = False
            self.delta_members = ()
            self.delta_requests = {}
            super().__init__(*args, **kwargs)
            if self.mm_processor is not None:
                track_multimodal_processor(self.mm_processor, _TOKENIZER_RESOURCES.get)
            self.delta_profile_id = _profile(self.server_args)
            self._result_dispatcher += TypeBasedDispatcher([(DeltaReply, self._delta_reply)])

        async def generate_request(self, obj, request=None):
            ticket = self.delta_ticket
            if ticket is None or self.delta_poisoned:
                raise DeltaCodecError("consumer is not active")
            generator = super().generate_request(obj, request)
            key = uuid.uuid4().hex
            sampling = obj.sampling_params
            options = sampling if isinstance(sampling, list) else [sampling]
            resources = UnsentMultimodalResources(
                self.server_args.tp_size, repeat_inputs=any(option and option.get("n", 1) > 1 for option in options)
            )
            record = {"generator": generator, "busy": False, "resources": resources}
            self.delta_requests[key] = record
            try:
                while True:
                    if ticket != self.delta_ticket or self.delta_poisoned:
                        raise DeltaCodecError("request generation was closed by installation")
                    token = _TOKENIZER_TICKET.set(ticket)
                    resource_token = _TOKENIZER_RESOURCES.set(resources)
                    record["busy"] = True
                    try:
                        result = await generator.__anext__()
                    except StopAsyncIteration:
                        return
                    finally:
                        record["busy"] = False
                        _TOKENIZER_TICKET.reset(token)
                        _TOKENIZER_RESOURCES.reset(resource_token)
                    if ticket != self.delta_ticket or self.delta_poisoned:
                        raise DeltaCodecError("request generation was closed by installation")

                    def provenance(item):
                        item = dict(item)
                        item["meta_info"] = dict(item.get("meta_info", {}), consumer_generation=ticket)
                        return item

                    yield [provenance(item) for item in result] if isinstance(result, list) else provenance(result)
            finally:
                try:
                    try:
                        await generator.aclose()
                    finally:
                        await resources.close()
                finally:
                    self.delta_requests.pop(key, None)

        async def delta_drain_requests(self):
            # A client may be suspended after a streaming yield. Closing the
            # underlying generator releases its state without waiting for that
            # client to resume; the wrapper checks the generation before reuse.
            while self.delta_requests or self.rid_to_state:
                for key, record in tuple(self.delta_requests.items()):
                    if not record["busy"]:
                        await record["generator"].aclose()
                        await record["resources"].close()
                        self.delta_requests.pop(key, None)
                if self.delta_requests or self.rid_to_state:
                    await asyncio.sleep(0.01)

        def _delta_admission_ticket(self):
            ticket = _TOKENIZER_TICKET.get()
            if ticket is None or ticket != self.delta_ticket or self.delta_poisoned:
                raise DeltaCodecError("request admission generation changed during tokenization")
            return ticket

        def _send_one_request(self, tokenized_obj):
            # Preprocessing can await across PREPARE. Reject its old result
            # before native dispatch allocates receiver-owned shared memory.
            self._delta_admission_ticket()
            resources = _TOKENIZER_RESOURCES.get()
            if resources is not None:
                resources.prepare(tokenized_obj.mm_inputs)
                resources.transfer(tokenized_obj.mm_inputs)
            return super()._send_one_request(tokenized_obj)

        def _send_batch_request(self, tokenized_objs):
            self._delta_admission_ticket()
            resources = _TOKENIZER_RESOURCES.get()
            if resources is not None:
                for obj in tokenized_objs:
                    resources.prepare(obj.mm_inputs)
                    resources.transfer(obj.mm_inputs)
            return super()._send_batch_request(tokenized_objs)

        def _dispatch_to_scheduler(self, obj):
            if isinstance(obj, WORK_TYPES):
                obj = DeltaWork(ticket=self._delta_admission_ticket(), payload=obj)
            elif not isinstance(obj, (DeltaCommand, AbortReq, ShutdownReq)):
                raise DeltaCodecError("legacy control cannot bypass consumer installation")
            super()._dispatch_to_scheduler(obj)

        async def _async_dispatch_to_scheduler(self, obj):
            # These are legacy pause/update controls; delta commands use their
            # own identity-correlated synchronous enqueue on the event loop.
            raise DeltaCodecError("legacy asynchronous control is disabled in delta mode")

        def _delta_reply(self, reply):
            pending = self.delta_pending
            if pending is None or reply.command_id != pending["command"].digest:
                self.delta_poisoned = True
                self.delta_ticket = None
                return
            expected = {m.rank: m.incarnation for m in pending["command"].installation.members}
            if expected.get(reply.rank) != reply.incarnation:
                if not pending["future"].done():
                    pending["future"].set_exception(DeltaCodecError("reply from stale worker incarnation"))
                self.delta_poisoned = True
                self.delta_ticket = None
                return
            prior = pending["replies"].get(reply.rank)
            if prior is not None and prior != reply:
                self.delta_poisoned = True
                self.delta_ticket = None
                if not pending["future"].done():
                    pending["future"].set_exception(DeltaCodecError("conflicting duplicate rank reply"))
                return
            pending["replies"][reply.rank] = reply
            if len(pending["replies"]) == len(expected) and not pending["future"].done():
                pending["future"].set_result(tuple(pending["replies"][r] for r in sorted(expected)))

        async def delta_command(self, command, argument, timeout):
            if self.delta_poisoned or self.delta_pending is not None:
                raise DeltaCodecError("consumer channel has an unresolved or failed operation")
            if command.installation.members != self.delta_members:
                raise DeltaCodecError("command membership differs from startup handshake")
            if command.action == "PREPARE":
                self.delta_ticket = None
            self.auto_create_handle_loop()
            future = asyncio.get_running_loop().create_future()
            self.delta_pending = {"command": command, "replies": {}, "future": future}
            self._dispatch_to_scheduler(
                DeltaCommand(
                    command=canonical_json(asdict(command), 1024 * 1024), argument=canonical_json(argument, 64 * 1024)
                )
            )
            try:
                replies = await asyncio.wait_for(asyncio.shield(future), timeout)
                if any(not reply.success for reply in replies):
                    errors = [parse_json(r.body, 64 * 1024) for r in replies if not r.success]
                    raise DeltaCodecError(f"worker installation failed: {errors}")
                records = [parse_json(r.body, 64 * 1024) for r in replies]
                receipts = tuple(RankReceipt(**record["receipt"]) for record in records)
                phase = {
                    "PREPARE": "PREPARED",
                    "QUIESCE": "QUIESCED",
                    "LOAD": "RUNTIME_READY",
                    "VERIFY": "VERIFIED",
                    "ACK_COMMIT": "ADMISSION_READY",
                    "ACTIVATE": "ACTIVE",
                }[command.action]
                certify_receipts(command, receipts, phase)
            except BaseException:
                self.delta_poisoned = True
                self.delta_ticket = None
                # Keep pending identity. A caller timeout cannot cancel GPU work.
                raise
            self.delta_pending = None
            return receipts, records

    async def reject_control(self, *args, **kwargs):
        raise DeltaCodecError("legacy control is disabled while consumer owns the engine")

    for name in (
        "pause_generation",
        "continue_generation",
        "update_weights_from_disk",
        "update_weights_from_tensor",
        "update_weights_from_ipc",
        "update_weights_from_distributed",
        "init_weights_update_group",
        "destroy_weights_update_group",
        "release_memory_occupation",
        "resume_memory_occupation",
        "post_process_weights",
        "load_lora_adapter",
        "load_lora_adapter_from_tensors",
        "update_lora_from_distributed",
        "unload_lora_adapter",
        "set_internal_state",
        "flush_cache",
        "open_session",
        "close_session",
        "attach_hicache_storage",
        "detach_hicache_storage",
    ):
        setattr(DeltaTokenizer, name, reject_control)
    return init_tokenizer_manager(server_args, port_args, TokenizerManagerClass=DeltaTokenizer)


def create_engine(runtime_config: RuntimeConfig, **kwargs: Any) -> Any:
    """Create one explicitly opted-in Python engine with scoped child cleanup.

    Call from a dedicated consumer process. Generic RPC is disabled; use the
    consumer installer. Existing default Relax engines are unaffected.
    """
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.server_args import ServerArgs

    from relax.distributed.weight_sync.consumer import Member

    server_args = kwargs.get("server_args") or ServerArgs(**kwargs)
    _profile(server_args, startup=True)
    owner_process = process_identity(os.getpid())

    class DeltaEngine(Engine):
        init_tokenizer_manager_func = staticmethod(partial(init_delta_tokenizer, runtime_config=runtime_config))
        run_scheduler_process_func = staticmethod(
            partial(run_delta_scheduler, runtime_config=runtime_config, owner_process=owner_process)
        )
        run_detokenizer_process_func = staticmethod(partial(run_delta_detokenizer, owner_process=owner_process))
        _owned_processes = []

        def __init__(self, **options):
            try:
                super().__init__(**options)
            except BaseException:
                self.shutdown()
                raise

        @classmethod
        def _launch_scheduler_processes(cls, server_args, port_args, run_scheduler_process_func):
            from sglang.srt.entrypoints import engine as module

            # This opt-in factory supports one local TP group only. Retain
            # each handle before start so even a partially failed spawn is
            # owned and cleaned by the constructor's exception path.
            processes, readers, infos = [], [], []
            for rank in range(server_args.tp_size):
                reader, writer = module.mp.Pipe(duplex=False)
                gpu_id = server_args.base_gpu_id + rank * server_args.gpu_id_step
                with module.maybe_reindex_device_id(gpu_id) as gpu_id:
                    process = module.mp.Process(
                        target=run_scheduler_process_func,
                        args=(server_args, port_args, gpu_id, rank, 0, 0, 0, 0, None, writer),
                    )
                    cls._owned_processes.append(process)
                    with module.numa_utils.configure_subprocess(server_args, gpu_id):
                        process.start()
                writer.close()
                processes.append(process)
                readers.append(reader)

            def wait_for_ready():
                infos.extend(module._wait_for_scheduler_ready(readers, processes))

            def block_until_exits():
                for process in processes:
                    process.join()

            return module.SchedulerInitResult(
                scheduler_infos=infos,
                all_child_pids=[p.pid for p in processes],
                wait_for_ready=wait_for_ready,
                block_until_scheduler_exits=block_until_exits,
            ), processes

        @classmethod
        def _launch_detokenizer_subprocesses(cls, server_args, port_args, run_detokenizer_process_func):
            from sglang.srt.entrypoints import engine as module

            process = module.mp.Process(target=run_detokenizer_process_func, args=(server_args, port_args))
            cls._owned_processes.append(process)
            process.start()
            return [process], ["detokenizer"]

        def shutdown(self):
            # The dependency's default shutdown kills every child of the host
            # process. Only this engine's retained Process handles are ours.
            manager = getattr(self, "tokenizer_manager", None)
            if manager is not None:
                manager.delta_ticket = None
                manager.delta_poisoned = True
                for task in tuple(getattr(manager, "asyncio_tasks", ())):
                    task.cancel()
                watchdog = getattr(manager, "_subprocess_watchdog", None)
                if watchdog is not None:
                    watchdog.stop()
            for process in self._owned_processes:
                if process.is_alive():
                    process.terminate()
            isolation = []
            for process in self._owned_processes:
                if process.pid is None:
                    continue
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                if process.is_alive():
                    raise DeltaCodecError("engine child isolation could not be established")
                if process.pid is None or process.exitcode is None:
                    raise DeltaCodecError("engine child termination evidence is incomplete")
                isolation.append({"pid": process.pid, "exitcode": process.exitcode})
            if isolation:
                self.delta_isolation = tuple(isolation)
            self._owned_processes.clear()
            socket = getattr(self, "send_to_rpc", None)
            if socket is not None:
                socket.close(linger=0)
                self.send_to_rpc = None

        def collective_rpc(self, *args, **options):
            raise DeltaCodecError("generic RPC is disabled while consumer owns the engine")

    engine = DeltaEngine(server_args=server_args)
    try:
        infos = [entry.get("relax_delta") for entry in engine._scheduler_init_result.scheduler_infos]
        tp = engine.server_args.tp_size
        if len(infos) != tp or any(
            not isinstance(info, dict)
            or info.get("closed") is not True
            or info.get("consumer_id") != runtime_config.consumer_id
            for info in infos
        ):
            raise DeltaCodecError("incomplete default-closed startup capability handshake")
        infos.sort(key=lambda info: info["rank"])
        if tuple(info["rank"] for info in infos) != tuple(range(tp)):
            raise DeltaCodecError("duplicate or missing startup rank")
        if len({info["profile_id"] for info in infos}) != 1 or len({info["plan_id"] for info in infos}) != 1:
            raise DeltaCodecError("worker runtime profiles/layouts differ")
        if infos[0]["profile_id"] != engine.tokenizer_manager.delta_profile_id:
            raise DeltaCodecError("tokenizer and workers resolved different execution profiles")
        engine.tokenizer_manager.delta_members = tuple(Member(info["rank"], info["incarnation"]) for info in infos)
        identities = [process_identity(process.pid) for process in engine._owned_processes]
        if (
            len(identities) != tp + 1
            or any(info["process"] not in identities for info in infos)
            or len({info["process"]["pid"] for info in infos}) != tp
        ):
            raise DeltaCodecError("startup workers differ from retained engine process identities")
        engine.delta_execution_id = uuid.uuid4().hex
        engine.delta_process_binding = {
            "processes": identities,
            "workers": [
                {"rank": info["rank"], "incarnation": info["incarnation"], "process": info["process"]}
                for info in infos
            ],
        }
        engine.delta_plan_id = infos[0]["plan_id"]
        engine.delta_runtime_config = runtime_config
        return engine
    except BaseException:
        engine.shutdown()
        raise


def create_http_app(engine: Any) -> Any:
    """Expose consumer-owned native /generate and /encode ASGI endpoints.

    Run in the same dedicated process as create_engine. Legacy control routes
    are rejected at the HTTP boundary as well as the tokenizer/schedulers.
    """
    from sglang.srt.entrypoints import http_server

    from .http import ConsumerHTTP

    if not hasattr(engine, "delta_runtime_config"):
        raise DeltaCodecError("HTTP admission requires a consumer-owned engine")
    http_server.set_global_state(
        http_server._GlobalState(
            engine.tokenizer_manager, engine.template_manager, engine._scheduler_init_result.scheduler_infos[0]
        )
    )
    return ConsumerHTTP(http_server.app, engine.tokenizer_manager, engine.shutdown)
