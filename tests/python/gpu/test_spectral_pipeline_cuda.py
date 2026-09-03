"""Actual CUDA graph replay of prepared spectral chains and multi-view recovery."""

import importlib
from typing import Any

import numpy as np
import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.recovery import PoseChart
from dpt.spectral_recovery import SpectralPoseEvaluator
from helpers.spectral_pipeline_inputs import make_pipeline, pose_problem

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("channels", ["serial", "fused"])
def test_graph_pipeline_replays_changed_fields_and_cotangents_with_all_gradients(
    precision: str,
    channels: str,
) -> None:
    graph, fields, pose, _ = make_pipeline(
        "cuda:0", precision=precision, execution="graph", channels=channels
    )
    eager, _, _, _ = make_pipeline("cuda:0", precision=precision, channels=channels)
    for value, cotangent in ((0.125, 1.0), (0.25, -0.5), (0.125, 0.0)):
        fields.fill_(value)
        expected_image = eager.forward(fields, pose).numpy().copy()
        actual_image = graph.forward(fields, pose).numpy().copy()
        np.testing.assert_array_equal(actual_image, expected_image)
        seed = wp.full(1, cotangent, dtype=getattr(wp, precision), device="cuda:0")
        expected, actual = eager.vjp(seed), graph.vjp(seed)
        for name in ("fields", "pose", "coefficients", "weights", "response"):
            left, right = getattr(actual, name), getattr(expected, name)
            assert left is not None and right is not None
            np.testing.assert_allclose(left.numpy(), right.numpy(), rtol=2e-6, atol=1e-8)
    assert graph._forward_graph is not None and graph._vjp_graph is not None


def test_graph_pipeline_value_checks_run_before_replay_and_checked_retry_recovers() -> None:
    graph, fields, pose, _ = make_pipeline("cuda:0", execution="graph")
    graph.forward(fields, pose)
    original = graph._forward_graph
    fields.fill_(2.0)
    with pytest.raises(NumericalError):
        graph.forward(fields, pose)
    with pytest.raises(ContractError, match="successful forward"):
        graph.vjp(wp.ones(1, dtype=wp.float64, device="cuda:0"))
    fields.fill_(0.125)
    graph.forward(fields, pose)
    assert graph._forward_graph is original
    gradient = graph.vjp(wp.ones(1, dtype=wp.float64, device="cuda:0")).fields
    assert gradient is not None
    assert np.all(np.isfinite(gradient.numpy()))


@pytest.mark.parametrize("blurred", [False, True])
def test_multiview_spectral_graph_matches_eager_and_chart_finite_differences(
    blurred: bool,
) -> None:
    problem = pose_problem("cuda:0", blurred=blurred)
    eager = SpectralPoseEvaluator(problem, PoseChart())
    graph = SpectralPoseEvaluator(problem, PoseChart(), execution="graph")
    point = (0.03, -0.02, 0.01, 0.1, -0.2, 0.3, 0.05, -0.4)
    actual, expected = graph(point), eager(point)
    assert actual.loss == pytest.approx(expected.loss, rel=1e-12)
    assert actual.gradient == pytest.approx(expected.gradient, rel=2e-6, abs=1e-8)
    captured = graph._evaluation_graph
    # Rotation chart steps of 0.1 are 0.001 rad; they resolve FP32 image changes.
    steps = (0.04, 0.04, 0.04, 0.1, 0.1, 0.1, 0.002, 0.04)
    for index in range(len(point)):
        step = steps[index]
        plus, minus = list(point), list(point)
        plus[index] += step
        minus[index] -= step
        finite_difference = (graph(tuple(plus)).loss - graph(tuple(minus)).loss) / (2 * step)
        assert actual.gradient[index] == pytest.approx(finite_difference, rel=0.02, abs=0.005)
    assert graph._evaluation_graph is captured
    assert graph(point) == actual
