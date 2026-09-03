"""Actual CUDA eager/graph joint recovery of a known six-coordinate pose.

The scene is an original asymmetric 9-cubed phantom with two 12x12 panels and
two spectral energies. Both modes share immutable inputs, chart, initial point
and bounded policy. Host solve decisions/readbacks remain outside capture.
"""

import importlib
import json
from typing import Any, cast

import numpy as np
import pytest

from dpt.contracts import NumericalError
from dpt.geometry import RigidTransform
from dpt.objectives import ObjectiveSpec
from dpt.recovery import PoseChart
from dpt.registration import RecoveryPolicy
from dpt.spectral import SpectralSpec
from dpt.spectral_recovery import (
    CalibrationBlock,
    SpectralPoseEvaluator,
    SpectralPoseProblem,
    SpectralView,
    recover_spectral_pose,
)
from dpt.validation.recovery import pose_error
from helpers.pose_inputs import CHART_SCALES, LANDMARKS, PROVENANCE, geometries, phantom

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


def _problem(device: str, stream: Any = None) -> tuple[SpectralPoseProblem, PoseChart]:
    grid, values = phantom()
    fields = wp.array([10 * value for value in values], dtype=wp.float32, device=device)
    coefficients = wp.array([0.2, 0.5], dtype=wp.float32, device=device)
    weights = wp.array([100.0, 80.0], dtype=wp.float32, device=device)
    response = wp.array([1.0, 2.0], dtype=wp.float32, device=device)
    spec = SpectralSpec(
        1, 2, (PROVENANCE,), PROVENANCE, PROVENANCE, "signal", "original analytic pose fixture"
    )
    views = tuple(
        SpectralView(
            f"view-{index}",
            geometry,
            spec,
            coefficients,
            weights,
            response,
            wp.zeros(geometry.pixels, dtype=wp.float32, device=device),
            "shared",
            PROVENANCE,
            objective=ObjectiveSpec(reduction="mean"),
        )
        for index, geometry in enumerate(geometries())
    )
    problem = SpectralPoseProblem(
        grid, 1, fields, PROVENANCE, views, (CalibrationBlock("shared"),), 96
    )
    chart = PoseChart(scales=CHART_SCALES)
    synthesis = SpectralPoseEvaluator(problem, chart, device=device, stream=stream)
    synthesis((0.0,) * 6)
    for view, image in zip(views, synthesis.predicted_images, strict=True):
        wp.copy(view.observation, image, stream=stream)
    if synthesis.context.device.is_cuda:
        wp.synchronize_stream(synthesis.context.stream)
    return problem, chart


def _owned_pointers(evaluator: SpectralPoseEvaluator) -> tuple[int, ...]:
    return tuple(
        int(cast(int, array.ptr))
        for array in (
            evaluator._pose_device,
            evaluator._parameters_device,
            evaluator._control_device,
            evaluator._nuisance_device,
            evaluator._shared_paths,
            evaluator._shared_adj_paths,
            *evaluator._shared_images,
            *evaluator.predicted_images,
        )
    )


def test_two_shared_views_recover_known_pose_in_cuda_eager_and_replayed_graph(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    wp.init()
    device = wp.get_device("cuda:0")
    assert device.is_cuda, "this acceptance case requires real CUDA"
    stream = wp.Stream(device)
    with wp.ScopedStream(stream):
        problem, chart = _problem("cuda:0", stream)
        eager = SpectralPoseEvaluator(problem, chart, device="cuda:0", stream=stream)
        graph = SpectralPoseEvaluator(
            problem, chart, device="cuda:0", stream=stream, execution="graph"
        )
    assert len(problem.views) == 2 and eager.dimension == graph.dimension == 6
    assert eager.context.stream == graph.context.stream == stream
    initial = (0.25, -0.18, 0.13, 0.3, -0.4, 0.2)
    policy = RecoveryPolicy(
        max_iterations=200,
        max_evaluations=1200,
        gradient_tolerance=1e-3,
        relative_loss_tolerance=1e-12,
    )
    initial_error = pose_error(chart.pose(initial), RigidTransform(), LANDMARKS)
    pointers = _owned_pointers(graph)
    real_launch = wp.capture_launch
    replay_graphs = []

    def observe_replay(captured: Any, *args: Any, **kwargs: Any) -> Any:
        assert kwargs.get("stream") == stream
        replay_graphs.append(captured)
        return real_launch(captured, *args, **kwargs)

    monkeypatch.setattr(wp, "capture_launch", observe_replay)
    eager_initial, graph_initial = eager(initial), graph(initial)
    assert graph_initial.loss == pytest.approx(eager_initial.loss, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(graph_initial.gradient, eager_initial.gradient, rtol=2e-6, atol=1e-8)
    captured = graph._evaluation_graph
    assert captured is not None and replay_graphs == [captured]
    eager_result = recover_spectral_pose(eager, initial=initial, policy=policy)
    replays_before_recovery = len(replay_graphs)
    graph_result = recover_spectral_pose(graph, initial=initial, policy=policy)
    recovery_replays = len(replay_graphs) - replays_before_recovery
    assert recovery_replays == graph_result.optimisation.evaluations
    assert recovery_replays > 1 and all(item is captured for item in replay_graphs)
    assert graph._evaluation_graph is captured and _owned_pointers(graph) == pointers
    errors = []
    for evaluator, result in ((eager, eager_result), (graph, graph_result)):
        optimisation = result.optimisation
        error = pose_error(result.pose, RigidTransform(), LANDMARKS)
        errors.append(error)
        assert optimisation.reason == "gradient_tolerance" and optimisation.stationary
        assert (
            max(abs(value) for value in optimisation.evaluation.gradient)
            <= policy.gradient_tolerance
        )
        assert optimisation.evaluation.loss < 1e-5
        assert error.translation_mm < 1e-3 and error.rotation_radians < 1e-3
        assert error.landmark_rms_mm < min(1e-3, initial_error.landmark_rms_mm / 100)
        assert len(optimisation.history) > 1
        assert all(
            following.loss <= previous.loss
            for previous, following in zip(
                optimisation.history, optimisation.history[1:], strict=False
            )
        )
        evaluator._status_checkpoint.check()
    np.testing.assert_allclose(
        graph_result.optimisation.parameters,
        eager_result.optimisation.parameters,
        rtol=2e-5,
        atol=2e-5,
    )
    assert graph_result.optimisation.reason == eager_result.optimisation.reason
    accepted = graph_result.optimisation.parameters
    graph_final, eager_final = graph(accepted), eager(accepted)
    assert graph_final.loss == pytest.approx(eager_final.loss, rel=2e-6, abs=1e-12)
    np.testing.assert_allclose(graph_final.gradient, eager_final.gradient, rtol=2e-6, atol=1e-8)
    # Pending failure delivery must precede staging/replay; consume then permits retry.
    with wp.ScopedStream(stream):
        graph._prepared[0].material.projection._status.fill_(8)
    before_failure = len(replay_graphs)
    with pytest.raises(NumericalError):
        graph(accepted)
    assert len(replay_graphs) == before_failure
    assert graph(accepted) == graph_final
    graph._status_checkpoint.check()
    assert graph._evaluation_graph is captured and _owned_pointers(graph) == pointers
    request.node.user_properties.append(
        (
            "s02_cuda_recovery",
            json.dumps(
                {
                    "device": str(device),
                    "views": 2,
                    "shared_pose_coordinates": 6,
                    "grid_voxels": problem.grid.voxels,
                    "pixels_per_view": [view.geometry.pixels for view in problem.views],
                    "samples_per_ray": problem.samples_per_ray,
                    "max_iterations": policy.max_iterations,
                    "max_evaluations": policy.max_evaluations,
                    "eager_evaluations": eager_result.optimisation.evaluations,
                    "graph_evaluations": graph_result.optimisation.evaluations,
                    "observed_recovery_replays": recovery_replays,
                    "capture_reused": True,
                    "owned_pointers_stable": True,
                    "pending_status_blocked_replay_and_retry_passed": True,
                    "eager_translation_error_mm": errors[0].translation_mm,
                    "graph_translation_error_mm": errors[1].translation_mm,
                    "graph_landmark_rms_error_mm": errors[1].landmark_rms_mm,
                    "graph_loss": graph_final.loss,
                },
                sort_keys=True,
            ),
        )
    )
