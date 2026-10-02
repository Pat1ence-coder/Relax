# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Track successful optimizer updates and serialize source capture boundaries.

The training worker must use begin/end_update around its optimizer and call
mark_synchronized after parameter gather/writeback. Its synchronous execution
is the writer fence; this object does not lock arbitrary external tensor
writes.
"""

import os
import threading
from typing import Any

from relax.distributed.weight_sync import DeltaCodecError, ExportRequest
from relax.distributed.weight_sync.limits import require_uint
from relax.distributed.weight_sync.serialization import require_identifier

from .inventory import unwrap_model


def _marker(tensor: Any) -> tuple:
    return (
        id(tensor),
        tensor._version,
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


class TrainingBoundary:
    def __init__(self, model: Any, *, run_epoch: str, source_step: int = 0) -> None:
        require_identifier(run_epoch, "run epoch")
        require_uint(source_step, "successful source steps")
        self.model, self.run_epoch, self.source_step = unwrap_model(model), run_epoch, source_step
        self._owner = (os.getpid(), threading.get_ident())
        self._updating = self._ready = self._poisoned = False
        self._lease: SourceLease | None = None

    def _check(self) -> None:
        if self._owner != (os.getpid(), threading.get_ident()):
            raise DeltaCodecError("training boundary requires its owning process and thread")
        if self._poisoned:
            raise DeltaCodecError("training boundary is invalid after an uncertain update or concurrent write")

    def begin_update(self) -> None:
        self._check()
        if self._lease is not None or self._updating:
            raise DeltaCodecError("cannot update while capture or an optimizer update is active")
        self._updating, self._ready = True, False

    def end_update(self, successful: bool) -> None:
        self._check()
        if not self._updating or type(successful) is not bool:
            raise DeltaCodecError("optimizer completion requires an active update and a boolean success flag")
        next_step = self.source_step + int(successful)
        require_uint(next_step, "successful source steps")
        self.source_step = next_step
        self._updating = False

    def fail_update(self) -> None:
        self._check()
        self._updating = False
        self._poisoned = True

    def invalidate(self) -> None:
        """Prevent reuse after an uncertain transport or source failure."""
        self._ready = False
        self._poisoned = True

    def optimizer_step(self, optimizer: Any) -> tuple:
        """Count actual successful steps while preserving Megatron's result."""
        self.begin_update()
        try:
            result = optimizer.step()
            self.end_update(result[0])
            return result
        except BaseException:
            self.fail_update()
            raise

    def mark_synchronized(self) -> None:
        self._check()
        if self._updating or self._lease is not None:
            raise DeltaCodecError("parameter synchronization cannot complete during update or capture")
        self._ready = True

    def acquire(self, request: ExportRequest) -> "SourceLease":
        self._check()
        if self._updating or not self._ready or self._lease is not None:
            raise DeltaCodecError("capture requires a synchronized, idle training boundary")
        if (request.run_epoch, request.source_step) != (self.run_epoch, self.source_step):
            raise DeltaCodecError("export request differs from the completed training step or epoch")
        import torch

        # Complete copies/kernels on every stream before any raw host reads.
        # Callers must already have enqueued and completed host-side param sync.
        devices = {tensor.device for tensor in self.model.parameters() if tensor.device.type == "cuda"}
        for device in devices:
            torch.cuda.synchronize(device)
        self._lease = SourceLease(self, request)
        return self._lease


class SourceLease:
    def __init__(self, boundary: TrainingBoundary, request: ExportRequest) -> None:
        self.boundary, self.request = boundary, request
        self._markers = tuple(
            (name, _marker(tensor)) for name, tensor in boundary.model.named_parameters(remove_duplicate=False)
        )
        self._released = False

    def validate(self) -> None:
        self.boundary._check()
        if self._released or self.boundary._lease is not self:
            raise DeltaCodecError("source lease is not active")
        actual = tuple(
            (name, _marker(tensor)) for name, tensor in self.boundary.model.named_parameters(remove_duplicate=False)
        )
        if actual != self._markers or self.boundary.source_step != self.request.source_step:
            self.boundary._poisoned = True
            raise DeltaCodecError("native parameter storage changed during capture")

    def release(self) -> None:
        if self._released:
            return
        if self.boundary._owner != (os.getpid(), threading.get_ident()):
            raise DeltaCodecError("source lease release requires its owning thread")
        self.boundary._lease = None
        self._released = True
