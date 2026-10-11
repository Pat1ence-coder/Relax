# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Consume a delta store without the trainer.

Brings a weight version (default: the latest) out of a ``--delta-store-dir``
written by ``--delta-transport shared_fs`` (or served over TCP), starting from a
full package and replaying the deltas after it
(:func:`relax.distributed.weight_sync.store.plan`)::

    # rebuild the HF weights on CPU
    python -m relax.distributed.weight_sync.consume --store-dir D --output OUT_DIR
    # bring a running SGLang engine started with
    #   --custom-weight-loader relax.backends.sglang.delta_loader.load_weights
    # from the version it holds to the latest one (WeightInstaller)
    python -m relax.distributed.weight_sync.consume --store-dir D --engine-url http://host:port --tp 2
    python -m relax.distributed.weight_sync.consume --tcp HOST:PORT --spool DIR --engine-url http://host:port

``--to EPOCH:VERSION`` selects another version. Only reads the store.
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

from relax.distributed.weight_sync.installer import HttpEngine, SharedStoreSource, TcpSource, WeightInstaller
from relax.distributed.weight_sync.store import plan, read_buckets
from relax.distributed.weight_sync.wire import DTYPES, IDX_NAME, VAL_NAME, decode_meta, payload_sha256
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


def install(engine_url: str, tp: int, source, target=None, verify: bool = False) -> int:
    """Bring a running engine from whatever version it holds to ``target``
    through the delta loader (:class:`WeightInstaller`)."""
    installer = WeightInstaller(HttpEngine(engine_url, tp, timeout=None), source)
    result = installer.install(target, verify=verify)
    for path in result.installed:
        logger.info(f"installed {os.path.basename(path)}")
    if result.abandoned:
        logger.warning("the engine held a version of an abandoned run; reinstalled from a full package")
    if not result.ok:
        print(f"install failed: {result.error}" + (" (engine left paused)" if result.paused else ""), file=sys.stderr)
        return 1
    logger.info(f"engine at e{result.state.epoch}:v{result.state.version}" + (", verified" if result.verified else ""))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--store-dir", help="read packages from this store")
    src.add_argument("--tcp", help="fetch packages from this package server (HOST:PORT; --engine-url only)")
    parser.add_argument("--spool", help="with --tcp: local directory for fetched packages")
    parser.add_argument("--to", help="EPOCH:VERSION (default: the latest version in the store)")
    out = parser.add_mutually_exclusive_group(required=True)
    out.add_argument("--output", help="write the rebuilt HF weights as safetensors into this new directory")
    out.add_argument("--engine-url", help="install into this SGLang engine")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size of --engine-url")
    parser.add_argument(
        "--verify", action="store_true", help="with --engine-url: check bitwise against a full package"
    )
    args = parser.parse_args(argv)

    target = tuple(int(x) for x in args.to.split(":")) if args.to else None
    if args.engine_url:
        if args.tcp:
            if not args.spool:
                parser.error("--tcp needs --spool")
            source = TcpSource(args.tcp, args.spool)
        else:
            source = SharedStoreSource(args.store_dir)
        return install(args.engine_url, args.tp, source, target, args.verify)
    if args.tcp:
        parser.error("--output reads a --store-dir")
    packages = plan(args.store_dir, target=target)
    if not packages:
        print(f"no packages in {args.store_dir}", file=sys.stderr)
        return 1
    logger.info("plan: " + ", ".join(os.path.relpath(p.path, args.store_dir) for p in packages))
    write_safetensors(rebuild(packages), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
