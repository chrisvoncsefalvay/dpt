"""Both material graph entry points retain private accumulation bindings."""

import importlib
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.material_reconstruction import MaterialReconstruction
from helpers.material_inputs import _inputs

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("entry", ["evaluation", "iteration"])
@pytest.mark.parametrize("mutation", ["pointer", "precision"])
def test_material_graph_rejects_changed_private_data_gradient_binding(
    entry: str, mutation: str
) -> None:
    inputs = _inputs("float64")
    inputs["device"] = "cuda:0"
    inputs["settings"] = replace(
        inputs["settings"],
        execution="graph",
        iterations=1,
        gradient_mapping_tolerance=1e9,
        relative_gradient_mapping_tolerance=0.0,
    )
    solver = MaterialReconstruction(**inputs)
    if entry == "evaluation":
        solver.evaluate()
    else:
        solver.solve()
    before = solver.gradient.numpy().copy()
    if mutation == "pointer":
        with solver.context.scope():
            solver._data_gradient = wp.empty(solver.size, dtype=wp.float64, device=solver.device)
    else:
        solver.views[0].accepted.volume_gradient_precision = "float32"
    with pytest.raises(ContractError, match="bindings must remain fixed"):
        if entry == "evaluation":
            solver.evaluate()
        else:
            solver.solve()
    np.testing.assert_array_equal(solver.gradient.numpy(), before)
