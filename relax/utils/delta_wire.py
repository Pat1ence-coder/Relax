# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Wire format shared by the trainer (sender) and the SGLang delta loader
(receiver).

Every message travels over the existing rollout weight-update NCCL group
through ``/update_weights_from_distributed`` with ``load_format=LOADER_PATH``
and starts with a JSON metadata tensor named ``META_NAME``:

- sparse delta bucket: ``[META_NAME, IDX_NAME, VAL_NAME]``. ``IDX_NAME`` holds
  int32 flat positions in the HF tensor's row-major order, ``VAL_NAME`` the
  replacement values as raw bytes. Each entry is
  ``{name, shape, dtype, n, i0, v0}``: positions ``idx[i0:i0+n]`` and values
  ``val[v0:v0+n*itemsize]`` (``v0`` aligned to ``VAL_ALIGN``).
- full bucket sent while delta sync is enabled: ``[META_NAME, *hf_tensors]``.
- control message: ``[META_NAME]`` alone.

Failures are reported by the loader raising (SGLang turns that into an HTTP
400 whose message contains ``ERROR_TAG``).
"""

import hashlib
import json

import torch


META_NAME = "__dws_meta__"
IDX_NAME = "__dws_idx__"
VAL_NAME = "__dws_val__"
LOADER_PATH = "relax.backends.sglang.delta_loader.load_weights"
ERROR_TAG = "[dws]"
# reasons in loader error messages that make the trainer disable delta sync for the run
UNSUPPORTED = "unsupported"
VERIFY_MISMATCH = "verify_mismatch"
# reply of the loader to a ping message (sent as an error so it cannot come from model.load_weights)
PONG = "delta loader pong"
# appended to PONG when /update_weights_from_tensor reports loader errors instead of crashing
TENSOR_PATH_SAFE = "tensor-path-safe"
VAL_ALIGN = 8
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def align(nbytes: int) -> int:
    return (nbytes + VAL_ALIGN - 1) // VAL_ALIGN * VAL_ALIGN


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def int_view(t: torch.Tensor) -> torch.Tensor:
    """Flat integer view of ``t`` with the same element size (bitwise
    compare)."""
    return t.reshape(-1).view(_INT_VIEW[t.element_size()])


def encode_meta(meta: dict, device: torch.device | str) -> torch.Tensor:
    data = json.dumps(meta, separators=(",", ":")).encode()
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)


def decode_meta(t: torch.Tensor) -> dict:
    return json.loads(t.cpu().numpy().tobytes().decode())


def encode_named_cpu_tensors(named_tensors: list[tuple[str, torch.Tensor]]) -> str:
    """Serialize 1-D uint8 tensors (loader metadata) for
    ``/update_weights_from_tensor``: base64 of a plain pickle that rebuilds
    each tensor with ``torch.frombuffer``. The bytes travel by value (no CUDA
    IPC handle, so the engine may run on another node), and the pickle does
    not reference ``torch.storage._load_from_bytes``, which Megatron replaces
    with a function SGLang's restricted unpickler rejects."""
    import base64
    import functools
    import io
    import pickle

    class _Pickler(pickle.Pickler):
        def reducer_override(self, obj):
            if isinstance(obj, torch.Tensor):
                assert obj.dim() == 1 and obj.dtype == torch.uint8, (obj.shape, obj.dtype)
                return functools.partial(torch.frombuffer, dtype=torch.uint8), (
                    bytearray(obj.cpu().numpy().tobytes()),
                )
            return NotImplemented

    buf = io.BytesIO()
    _Pickler(buf).dump(list(named_tensors))
    return base64.b64encode(buf.getvalue()).decode()


def payload_sha256(idx: torch.Tensor, val: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(idx.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    h.update(val.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
