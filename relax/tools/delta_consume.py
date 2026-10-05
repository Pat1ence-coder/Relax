# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Consume a shared-storage delta store without the trainer.

Brings a weight version (default: the latest) out of a ``--delta-store-dir``
written by ``--delta-transport shared_fs``, starting from a full package and
replaying the deltas after it (:func:`relax.utils.delta_store.plan`)::

    # rebuild the HF weights on CPU
    python -m relax.tools.delta_consume --store-dir D --output OUT_DIR
    # install into a running SGLang engine started with
    #   --custom-weight-loader relax.backends.sglang.delta_loader.load_weights
    python -m relax.tools.delta_consume --store-dir D --engine-url http://host:port --tp 2

``--to EPOCH:VERSION`` selects another version. Only reads the store.
"""

import argparse
import json
import os
import sys

import requests
import torch
from safetensors.torch import save_file

from relax.utils.delta_store import FULL, plan, read_buckets
from relax.utils.delta_wire import (
    DTYPES,
    IDX_NAME,
    LOADER_PATH,
    META_NAME,
    VAL_NAME,
    decode_meta,
    encode_meta,
    encode_named_cpu_tensors,
    payload_sha256,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

SHARD_BYTES = 4 << 30


def rebuild(packages) -> dict[str, torch.Tensor]:
    """Replay ``packages`` (a full package, then deltas) into HF tensors on
    CPU."""
    weights: dict[str, torch.Tensor] = {}
    for package in packages:
        for bucket in read_buckets(package.path):
            meta, rest = decode_meta(bucket[0][1]), dict(bucket[1:])
            if meta["action"] == "full":
                weights.update(rest)
                continue
            idx, val = rest[IDX_NAME][: meta["idx_len"]], rest[VAL_NAME][: meta["val_len"]]
            if payload_sha256(idx, val) != meta["sha256"]:
                raise ValueError(f"{package.path}: payload sha256 mismatch in bucket {meta['bucket']}")
            for e in meta["entries"]:
                dtype = DTYPES[e["dtype"]]
                target = weights[e["name"]]
                if list(target.shape) != e["shape"] or target.dtype != dtype:
                    raise ValueError(
                        f"{e['name']}: delta {e['shape']} {dtype} vs base {list(target.shape)} {target.dtype}"
                    )
                values = val[e["v0"] : e["v0"] + e["n"] * dtype.itemsize].view(dtype)
                target.view(-1)[idx[e["i0"] : e["i0"] + e["n"]].to(torch.int64)] = values
    return weights


def write_safetensors(weights: dict[str, torch.Tensor], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=False)
    shards, current, size = [], {}, 0
    for name in sorted(weights):
        t = weights[name]
        if current and size + t.numel() * t.element_size() > SHARD_BYTES:
            shards.append(current)
            current, size = {}, 0
        current[name] = t.contiguous()
        size += t.numel() * t.element_size()
    if current:
        shards.append(current)
    index = {}
    for i, shard in enumerate(shards):
        file = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(out_dir, file))
        index.update(dict.fromkeys(shard, file))
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": index}, f, indent=1)


def install(packages, engine_url: str, tp: int) -> None:
    """Install ``packages`` into a running engine through the delta loader."""

    def send(meta: dict, weight_version: str | None) -> None:
        payload = {
            "serialized_named_tensors": [encode_named_cpu_tensors([(META_NAME, encode_meta(meta, "cpu"))])] * tp,
            "load_format": LOADER_PATH,
            "flush_cache": True,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        response = requests.post(f"{engine_url.rstrip('/')}/update_weights_from_tensor", json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"{meta['action']} {meta.get('path', '')}: {response.text[:2000]}")

    for package in packages:
        meta = {"action": "install", "kind": package.kind, "epoch": package.epoch, "version": package.version}
        send(dict(meta, path=package.path), str(package.version))
        if package.kind == FULL:
            send({"action": "reset", "version": package.version, "epoch": package.epoch}, None)
        logger.info(f"installed {os.path.basename(package.path)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--to", help="EPOCH:VERSION (default: the latest version in the store)")
    out = parser.add_mutually_exclusive_group(required=True)
    out.add_argument("--output", help="write the rebuilt HF weights as safetensors into this new directory")
    out.add_argument("--engine-url", help="install into this SGLang engine")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size of --engine-url")
    args = parser.parse_args(argv)

    target = tuple(int(x) for x in args.to.split(":")) if args.to else None
    packages = plan(args.store_dir, target=target)
    if not packages:
        print(f"no packages in {args.store_dir}", file=sys.stderr)
        return 1
    logger.info("plan: " + ", ".join(os.path.relpath(p.path, args.store_dir) for p in packages))
    if args.output:
        write_safetensors(rebuild(packages), args.output)
    else:
        install(packages, args.engine_url, args.tp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
