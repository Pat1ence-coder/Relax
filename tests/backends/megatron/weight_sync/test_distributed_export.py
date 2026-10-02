# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real two-process Gloo export and bounded collective failure checks."""

import importlib
import importlib.util
import json
import multiprocessing
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from test_qwen3_vl_export import config, native_fixture


def _worker(rank: int, root: str, case: str) -> None:
    import torch
    import torch.distributed as dist

    from relax.distributed.weight_sync import ExportBudget, ExportRequest
    from relax.distributed.weight_sync.storage import DiskSnapshotStore
    from relax.distributed.weight_sync.storage.capture import DiskCapture

    directory = Path(__file__).resolve().parents[4] / "relax/backends/megatron/weight_sync"
    spec = importlib.util.spec_from_file_location(
        "_dws_worker_backend", directory / "__init__.py", submodule_search_locations=[str(directory)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    profiles = importlib.import_module(spec.name + ".profiles")
    boundary_module = importlib.import_module(spec.name + ".boundary")
    exporter_module = importlib.import_module(spec.name + ".exporter")
    path = Path(root)
    result = {}
    store = frozen = None
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo", init_method=(path / "rendezvous").as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=10)
        )
        if case == "rank_exit" and rank == 1:
            os._exit(73)
        profile = profiles.qwen3_vl_profile(config(), tp_size=2, padded_vocab_size=8, chunk_bytes=256)
        models, expected = native_fixture(2)
        model = models[rank]
        if case == "replica" and rank == 1:
            with torch.no_grad():
                model.vision_model.patch_embed.proj.bias[0] = 1
        if case == "missing" and rank == 1:
            del model.vision_model.pos_embed.weight
        boundary = boundary_module.TrainingBoundary(model, run_epoch="epoch")
        boundary.mark_synchronized()
        exporter = exporter_module.MegatronSnapshotExporter(
            model, profile, boundary, control_group=dist.group.WORLD, data_group=dist.group.WORLD, allow_cpu=True
        )
        if case == "wrong_rank":
            exporter.rank = 1 - rank
        if case == "dp":
            # An initialized two-rank world is not admissible for TP1.
            tp1 = profiles.qwen3_vl_profile(config(), tp_size=1, padded_vocab_size=8)
            exporter = exporter_module.MegatronSnapshotExporter(model, tp1, boundary, allow_cpu=True)
        if rank == 0:
            store = DiskSnapshotStore(path / "store", max_bytes=100000, max_generations=2)
        if case == "write" and rank == 0:

            def broken_write(self, tile):
                raise OSError("injected owner disk error")

            DiskCapture.write_tile = broken_write
        original_broadcast = dist.broadcast
        maximum = 0

        def bounded_broadcast(tensor, *args, **kwargs):
            nonlocal maximum
            assert tensor.dtype == torch.uint8 and tensor.numel() <= 64
            maximum = max(maximum, tensor.numel())
            return original_broadcast(tensor, *args, **kwargs)

        dist.broadcast = bounded_broadcast
        request = ExportRequest("test", "epoch", 0, 1 if case == "step" and rank == 1 else 0, "fixture")
        captured = exporter.capture(request, store, budget=ExportBudget(tile_bytes=64))
        frozen = captured.frozen
        if rank == 0:
            for chunk in profile.schema.iter_chunks():
                assert (
                    frozen.read_chunk(chunk)
                    == expected[chunk.tensor.name][chunk.byte_offset : chunk.byte_offset + chunk.byte_length]
                )
        result = {
            "ok": True,
            "root": captured.identity.target_root,
            "max_broadcast": maximum,
            "lease_released": boundary._lease is None,
        }
    except BaseException as error:
        result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        if "boundary" in locals():
            result["lease_released"] = boundary._lease is None
            if case == "rank_exit":
                try:
                    boundary.begin_update()
                except Exception:
                    result["transport_invalidated"] = True
    finally:
        if frozen is not None:
            frozen.close()
        if store is not None:
            store.close()
        (path / f"rank-{rank}.json").write_text(json.dumps(result))
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("case", ["ok", "replica", "missing", "step", "write", "wrong_rank", "dp", "rank_exit"])
def test_tp2_collectives_capture_or_abort_together(tmp_path: Path, case: str) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_worker, args=(rank, str(tmp_path), case)) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
            assert not process.is_alive(), "bounded exporter failure test hung"
        assert processes[0].exitcode == 0
        assert processes[1].exitcode == (73 if case == "rank_exit" else 0)
        reports = [
            json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(1 if case == "rank_exit" else 2)
        ]
        if case == "ok":
            assert all(report["ok"] and report["lease_released"] for report in reports), reports
            assert reports[0]["root"] == reports[1]["root"]
            assert all(0 < report["max_broadcast"] <= 64 for report in reports)
        else:
            assert all(not report["ok"] and report.get("lease_released", True) for report in reports), reports
            assert not list((tmp_path / "store").glob("generation-*/sealed.json"))
            if case == "rank_exit":
                assert reports[0]["transport_invalidated"]
    finally:
        # Only children created by this test are eligible for termination.
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
