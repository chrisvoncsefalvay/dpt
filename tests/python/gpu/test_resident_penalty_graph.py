"""Actual custom-penalty CUDA capture/replay, fixed bindings and convex recovery."""

import gc
import importlib
import weakref
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.material_preconditioning import prepare_fisher_scaling
from dpt.material_reconstruction import MaterialReconstruction
from helpers.material_fisher_inputs import _inputs
from helpers.material_solver_inputs import _independent_triangle_qp
from helpers.resident_penalty_inputs import MATRIX, PreparedQuadraticPenalty, ValueOnlyPenalty

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("target", [(0.2, 0.25), (0.35, 0.15), (-0.1, 0.3)])
def test_custom_penalty_graph_replays_and_recovers_independent_convex_qp(
    precision, target, monkeypatch
):
    inputs = _inputs(coefficients=False, precision=precision)
    inputs["settings"] = replace(
        inputs["settings"],
        iterations=100,
        gradient_mapping_tolerance=2e-6,
        relative_gradient_mapping_tolerance=0.0,
    )
    reports, fields = [], []
    replays = []
    launch = wp.capture_launch

    def record_replay(graph, *args, **kwargs):
        replays.append(graph)
        return launch(graph, *args, **kwargs)

    monkeypatch.setattr(wp, "capture_launch", record_replay)
    for execution in ("eager", "graph"):
        penalty = PreparedQuadraticPenalty(
            np.ones(inputs["grid"].voxels), device="cuda:0", target=target
        )
        solver = MaterialReconstruction(
            **{**inputs, "settings": replace(inputs["settings"], execution=execution)},
            penalty_operator=penalty,
        )
        scaling = prepare_fisher_scaling(solver)
        scaling.install(solver)
        bindings = [int(array.ptr) for array in (solver.fractions, solver.gradient, penalty.factor)]
        reports.append(solver.solve())
        if execution == "eager":
            assert not replays
        fields.append(solver.fractions_numpy().reshape(2, -1))
        assert bindings == [
            int(array.ptr) for array in (solver.fractions, solver.gradient, penalty.factor)
        ]
        assert reports[-1]["termination"] == "projected_gradient_tolerance"
    expected = _independent_triangle_qp(MATRIX, np.zeros(2), -MATRIX @ np.asarray(target), 1.0)
    for result in fields:
        np.testing.assert_allclose(
            result, np.repeat(expected[:, None], inputs["grid"].voxels, axis=1), atol=8e-6, rtol=0
        )
    np.testing.assert_allclose(fields[0], fields[1], atol=3e-7, rtol=0)
    # solve() captures complete iterations, independently of evaluate()'s
    # graph cache. Check reuse of the graphs that CUDA actually launched.
    captured = {id(graph) for graph in replays}
    assert len(replays) > len(captured) >= 2


@pytest.mark.parametrize("declaration", [None, False, "yes", 1])
def test_custom_graph_penalty_requires_explicit_boolean_declaration(declaration):
    inputs = _inputs()
    inputs["settings"] = replace(inputs["settings"], execution="graph")
    prepared = PreparedQuadraticPenalty(np.ones(inputs["grid"].voxels), device="cuda:0")
    legacy: Any = ValueOnlyPenalty(prepared)
    if declaration is not None:
        legacy.graph_safe = declaration
    with pytest.raises(ContractError, match="graph_safe=True"):
        MaterialReconstruction(**inputs, penalty_operator=legacy)


def test_captured_penalty_binding_and_declaration_cannot_be_replaced():
    inputs = _inputs()
    inputs["settings"] = replace(inputs["settings"], execution="graph")
    penalty = PreparedQuadraticPenalty(np.ones(inputs["grid"].voxels), device="cuda:0")
    solver = MaterialReconstruction(**inputs, penalty_operator=penalty)
    solver.evaluate()
    original_fields = solver.fractions.numpy().copy()
    reference = weakref.ref(penalty)
    factor_pointer = int(penalty.factor.ptr)
    del penalty
    gc.collect()
    penalty = reference()
    assert penalty is not None
    assert penalty is solver.penalty_operator
    assert int(penalty.factor.ptr) == factor_pointer
    assert np.isfinite(solver.evaluate())
    penalty.graph_safe = False
    with pytest.raises(ContractError, match="retain graph_safe=True"):
        solver.evaluate()
    penalty.graph_safe = True
    solver.penalty_operator = ValueOnlyPenalty(penalty)
    with pytest.raises(ContractError, match="binding must remain fixed"):
        solver.evaluate()
    solver.penalty_operator = penalty
    assert np.isfinite(solver.evaluate())
    np.testing.assert_array_equal(solver.fractions.numpy(), original_fields)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_custom_penalty_graph_certifies_independent_active_face_stationary_start(
    precision, monkeypatch
):
    target = (0.8, 0.6)
    inputs = _inputs(coefficients=False, precision=precision)
    expected = _independent_triangle_qp(MATRIX, np.zeros(2), -MATRIX @ np.asarray(target), 1.0)
    assert abs(expected.sum() - 1.0) < 2e-15
    inputs["initial_fractions"] = np.ascontiguousarray(
        np.broadcast_to(expected[:, None], (2, inputs["grid"].voxels)), dtype=np.float32
    ).reshape(inputs["initial_fractions"].shape)
    inputs["settings"] = replace(
        inputs["settings"],
        execution="graph",
        gradient_mapping_tolerance=2e-6,
        relative_gradient_mapping_tolerance=0.0,
    )
    penalty = PreparedQuadraticPenalty(
        np.ones(inputs["grid"].voxels), device="cuda:0", target=target
    )
    solver = MaterialReconstruction(**inputs, penalty_operator=penalty)
    prepare_fisher_scaling(solver).install(solver)
    replays = []
    launch = wp.capture_launch

    def record_replay(graph, *args, **kwargs):
        replays.append(graph)
        return launch(graph, *args, **kwargs)

    monkeypatch.setattr(wp, "capture_launch", record_replay)
    first_loss = solver.evaluate()
    assert solver.evaluate() == first_loss
    report = solver.solve()
    assert replays
    assert report["termination"] == "projected_gradient_tolerance"
    assert report["accepted_steps"] == 0
    assert report["final_gradient_mapping"] <= 2e-6
    np.testing.assert_allclose(
        solver.fractions_numpy().reshape(2, -1),
        np.repeat(expected[:, None], inputs["grid"].voxels, axis=1),
        atol=8e-8,
        rtol=0,
    )
