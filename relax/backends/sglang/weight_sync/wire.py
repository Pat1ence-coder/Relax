# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in typed SGLang IPC messages; imported only by the runtime adapter."""

from typing import Union

from sglang.srt.managers.io_struct import (
    BaseReq,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    hook_custom_types,
)


WORK_TYPES = (
    TokenizedGenerateReqInput,
    BatchTokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    BatchTokenizedEmbeddingReqInput,
)


class DeltaWork(BaseReq, tag="RelaxDeltaWorkV1"):
    ticket: str
    payload: Union[
        TokenizedGenerateReqInput,
        BatchTokenizedGenerateReqInput,
        TokenizedEmbeddingReqInput,
        BatchTokenizedEmbeddingReqInput,
    ]


class DeltaCommand(BaseReq, tag="RelaxDeltaCommandV1"):
    command: bytes
    argument: bytes


class DeltaReply(BaseReq, tag="RelaxDeltaReplyV1"):
    command_id: str
    rank: int
    incarnation: str
    success: bool
    body: bytes


def register_wire_types() -> None:
    hook_custom_types(DeltaWork, DeltaCommand, DeltaReply)
