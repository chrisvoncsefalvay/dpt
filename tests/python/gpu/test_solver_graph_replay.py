"""Real CUDA capture/replay of the immutable prepared material composition."""

import importlib
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dpt.contracts import NumericalError
from dpt.material_reconstruction import MaterialReconstruction
from helpers import material_inputs as cases
from helpers import pose_inputs as fixture

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


def test_cuda_graph_replays_prepared_accepted_trial_and_cached_gradient_chains():
    inputs = cases._inputs("float64")
    inputs["device"] = "cuda:0"
    eager = MaterialReconstruction(**inputs)
    eager_report = eager.solve()
    graph = MaterialReconstruction(
        **{**inputs, "settings": replace(inputs["settings"], execution="graph")}
    )
    report = graph.solve()
    assert report["execution"] == "graph"
    assert report["accepted_steps"] == eager_report["accepted_steps"] == 3
    graph.evaluate(gradient=True)
    graph.projected_trial(0.01)
    graph.evaluate(gradient=False, trial=True)
    assert {
        (True, False, False),
        (False, True, False),
        (True, False, True),
    } <= graph._evaluation_graphs.keys()
    np.testing.assert_allclose(
        graph.fractions_numpy(), eager.fractions_numpy(), atol=5e-7, rtol=5e-6
    )
    np.testing.assert_allclose(
        report["final_objective"], eager_report["final_objective"], rtol=2e-6
    )
    pointers = tuple(id(value) for value in graph._evaluation_graphs.values())
    reference = graph.evaluate()
    graph.views[-1].accepted.projection._status.fill_(8)
    try:
        graph.evaluate()
    except NumericalError:
        pass
    else:
        raise AssertionError("capture entry erased an unchecked numerical failure")
    assert graph.evaluate() == reference
    assert tuple(id(value) for value in graph._evaluation_graphs.values()) == pointers


def test_cuda_primary_graph_replays_pose_staging_and_prepared_control_readback():
    from dpt.objectives import ObjectiveSpec
    from dpt.recovery import PoseChart, PrimaryPoseEvaluator, PrimaryPoseProblem

    grid, values = fixture.phantom()
    geometry = fixture.geometries()[0]
    problem = PrimaryPoseProblem(
        grid,
        geometry,
        wp.array(values, dtype=wp.float32, device="cuda:0"),
        wp.full(geometry.pixels, 90.0, dtype=wp.float32, device="cuda:0"),
        objective=ObjectiveSpec(domain="counts"),
        open_beam=100.0,
        samples_per_ray=96,
    )
    chart = PoseChart(scales=(0.1, 0.1, 0.1, 0.01, 0.01, 0.01))
    for precision in ("float32", "float64"):
        selected = replace(problem, precision=precision)
        eager = PrimaryPoseEvaluator(selected, chart, device="cuda:0")
        graph = PrimaryPoseEvaluator(selected, chart, device="cuda:0", execution="graph")
        parameters = (
            (0.0,) * 6,
            (0.25, -0.18, 0.13, 0.3, -0.4, 0.2),
            (-0.17, 0.26, -0.22, -0.2, 0.15, -0.19),
        )
        for point in parameters:
            actual, expected = graph(point), eager(point)
            np.testing.assert_allclose(actual.loss, expected.loss, rtol=1e-7)
            np.testing.assert_allclose(actual.gradient, expected.gradient, rtol=1e-7, atol=1e-10)
        captured = graph._evaluation_graph
        assert captured is not None
        graph.projection._status.fill_(8)
        try:
            graph(parameters[-1])
        except NumericalError:
            pass
        else:
            raise AssertionError("graph entry erased an unchecked projection failure")
        assert graph(parameters[-1]) == actual
        assert graph._evaluation_graph is captured
