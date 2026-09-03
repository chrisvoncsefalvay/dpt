"""Bounded CUDA graph replay of tiled objectives and multi-level reductions."""

import importlib
from typing import Any

import pytest

from dpt.objectives import (
    ObjectiveSpec,
    evaluate_objective,
    evaluate_primary_objective,
    prepare_objective,
    reduce_objective_components,
)
from dpt.validation.objectives import scalar_objective

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu
approx: Any = pytest.approx  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


@pytest.mark.parametrize("size", [257, 65537])
def test_tiled_objective_graph_replay_recomputes_loss_and_seed(size: int) -> None:
    workspace = prepare_objective(ObjectiveSpec(precision="float64"), max_pixels=size)
    ctx = workspace.context
    prediction = wp.full(size, 1.125, dtype=wp.float64, device=ctx.device)
    observation = wp.zeros(size, dtype=wp.float32, device=ctx.device)
    loss = wp.empty(1, dtype=wp.float64, device=ctx.device)
    seed = wp.empty(size, dtype=wp.float64, device=ctx.device)

    def launch() -> None:
        evaluate_objective(
            prediction,
            observation,
            out_loss=loss,
            out_cotangent=seed,
            workspace=workspace,
            validate=False,
        )

    with ctx.scope():
        launch()
        workspace.check_status()
        with wp.ScopedCapture(device=ctx.device, stream=ctx.stream) as capture:
            launch()
        for value in (1.125, -2.5):
            prediction.fill_(value)
            wp.capture_launch(capture.graph, stream=ctx.stream)
            workspace.check_status()
            assert float(loss.numpy()[0]) == 0.5 * size * value * value
            assert seed.numpy().tolist() == [value] * size


@pytest.mark.parametrize("size", [257, 65537])
def test_primary_objective_graph_replay_recomputes_depth_cotangent(size: int) -> None:
    workspace = prepare_objective(ObjectiveSpec(kind="poisson", domain="counts"), max_pixels=size)
    ctx = workspace.context
    depth = wp.full(size, 0.125, dtype=wp.float64, device=ctx.device)
    observation = wp.ones(size, dtype=wp.float32, device=ctx.device)
    prediction = wp.empty(size, dtype=wp.float64, device=ctx.device)
    seed = wp.empty(size, dtype=wp.float64, device=ctx.device)
    loss = wp.empty(1, dtype=wp.float64, device=ctx.device)

    def launch() -> None:
        evaluate_primary_objective(
            depth,
            observation,
            open_beam=2.0,
            out_prediction=prediction,
            out_loss=loss,
            out_depth_cotangent=seed,
            workspace=workspace,
            validate=False,
        )

    with ctx.scope():
        launch()
        workspace.check_status()
        with wp.ScopedCapture(device=ctx.device, stream=ctx.stream) as capture:
            launch()
        for value in (0.125, 0.5):
            depth.fill_(value)
            wp.capture_launch(capture.graph, stream=ctx.stream)
            workspace.check_status()
            predicted = prediction.numpy().tolist()
            assert predicted == [predicted[0]] * size
            reference, _ = scalar_objective(float(predicted[0]), 1.0, poisson=True)
            assert float(loss.numpy()[0]) == approx(size * float(reference), rel=2e-14)
            assert seed.numpy().tolist() == [1.0 - float(predicted[0])] * size


@pytest.mark.parametrize("size", [257, 65537])
def test_component_reduction_graph_replay_recomputes_all_levels(size: int) -> None:
    workspace = prepare_objective(ObjectiveSpec(), max_pixels=size)
    ctx = workspace.context
    components = wp.full(size, 1.125, dtype=wp.float64, device=ctx.device)
    loss = wp.empty(1, dtype=wp.float64, device=ctx.device)

    def launch() -> None:
        reduce_objective_components(components, out_loss=loss, workspace=workspace, validate=False)

    with ctx.scope():
        launch()
        workspace.check_status()
        with wp.ScopedCapture(device=ctx.device, stream=ctx.stream) as capture:
            launch()
        for value in (1.125, -2.5):
            components.fill_(value)
            wp.capture_launch(capture.graph, stream=ctx.stream)
            workspace.check_status()
            assert float(loss.numpy()[0]) == size * value
