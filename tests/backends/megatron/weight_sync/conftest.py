# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Load the adapter package without Megatron's unrelated backend boot hooks."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="session")
def backend():
    directory = Path(__file__).resolve().parents[4] / "relax/backends/megatron/weight_sync"
    name = "_dws_backend_under_test"
    spec = importlib.util.spec_from_file_location(
        name, directory / "__init__.py", submodule_search_locations=[str(directory)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    yield SimpleNamespace(
        **{
            module: importlib.import_module(f"{name}.{module}")
            for module in ("profiles", "tile_plan", "inventory", "boundary", "exporter")
        }
    )
    for key in tuple(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]
