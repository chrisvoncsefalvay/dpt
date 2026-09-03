"""Each delivered evaluator checkpoint clears its diagnostic words once."""

import importlib
import json
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError, NumericalError
from dpt.objectives import ObjectiveSpec
from dpt.pose_optimization import PrimaryPoseLeastSquaresEvaluator
from dpt.recovery import PoseChart, PrimaryPoseEvaluator, PrimaryPoseProblem
from helpers.pose_inputs import geometries, phantom

wp: Any = importlib.import_module("warp")


def _fixture(
    kind: str,
    device: str,
    domain: Literal["counts", "log_transmission"] = "counts",
    precision: Literal["float32", "float64"] = "float64",
):
    grid, values = phantom()
    geometry = replace(
        geometries()[0], shape=(2, 3), origin_mm=(30.0, -0.2, -0.1), spacing_mm=(0.17, 0.19)
    )
    field = wp.array(values, dtype=wp.float32, device=device)
    observation = wp.full(
        geometry.pixels, 90.0 if domain == "counts" else -0.2, dtype=wp.float32, device=device
    )
    problem = PrimaryPoseProblem(
        grid,
        geometry,
        field,
        observation,
        objective=ObjectiveSpec(domain=domain),
        open_beam=100.0,
        samples_per_ray=64,
        precision=precision,
    )
    chart = PoseChart(scales=(0.1, 0.1, 0.1, 0.01, 0.01, 0.01))
    point = (0.1, -0.1, 0.1, 0.01, -0.01, 0.01)
    if kind == "primary":
        owner = PrimaryPoseEvaluator(problem, chart, device=device)
        checkpoint, poses = owner._status_checkpoint, (owner.pose_device,)
    else:
        second = replace(problem, geometry=replace(geometry, origin_mm=(30.0, -0.3, -0.2)))
        owner = PrimaryPoseLeastSquaresEvaluator((problem, second), chart, device=device)
        checkpoint, poses = owner._checkpoint, tuple(view.pose for view in owner._views)
    return owner, checkpoint, poses, point


def _same_result(actual, expected):
    np.testing.assert_allclose(actual.loss, expected.loss, rtol=2e-12, atol=1e-12)
    np.testing.assert_allclose(actual.gradient, expected.gradient, rtol=2e-10, atol=1e-11)
    if hasattr(expected, "gauss_newton"):
        np.testing.assert_allclose(
            actual.gauss_newton, expected.gauss_newton, rtol=2e-10, atol=1e-11
        )


@pytest.mark.parametrize(
    "kind,precision",
    [
        ("primary", "float32"),
        ("primary", "float64"),
        ("least_squares", "float64"),
    ],
)
@pytest.mark.parametrize("domain", ["counts", "log_transmission"])
def test_evaluation_clears_each_status_once_at_entry_and_completion(
    kind, precision, domain, warp_device, monkeypatch, request
):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        owner, checkpoint, _, point = _fixture(kind, warp_device, domain, precision)
        expected = owner(point)
        statuses = {int(status.ptr) for _, status in checkpoint._entries}
        zero = wp.array.zero_
        clears = Counter()

        def count(array):
            result = zero(array)
            if int(array.ptr) in statuses:
                clears[int(array.ptr)] += 1
            return result

        with monkeypatch.context() as patch:
            patch.setattr(wp.array, "zero_", count)
            actual = owner(point)
        _same_result(actual, expected)
        request.node.user_properties.append(
            (
                "status_clear_counts",
                json.dumps(
                    [
                        {
                            "workspace": type(workspace).__name__,
                            "pointer": int(status.ptr),
                            "clears": clears[int(status.ptr)],
                        }
                        for workspace, status in checkpoint._entries
                    ]
                ),
            )
        )
        assert clears == Counter(dict.fromkeys(statuses, 2))


@pytest.mark.parametrize(
    "kind,precision",
    [
        ("primary", "float32"),
        ("primary", "float64"),
        ("least_squares", "float64"),
    ],
)
def test_entry_delivers_every_pending_failure_before_pose_update_and_allows_retry(
    kind, precision, warp_device, monkeypatch
):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        owner, checkpoint, poses, point = _fixture(kind, warp_device, precision=precision)
        expected = owner(point)
        previous = [pose.numpy().copy() for pose in poses]
        statuses = {int(status.ptr) for _, status in checkpoint._entries}
        with owner.context.scope():
            checkpoint._entries[0][1].fill_(8)
            checkpoint._entries[-1][1].fill_(1)
        clears = Counter()
        zero = wp.array.zero_

        def count(array):
            result = zero(array)
            if int(array.ptr) in statuses:
                clears[int(array.ptr)] += 1
            return result

        with monkeypatch.context() as patch:
            patch.setattr(wp.array, "zero_", count)
            with pytest.raises((ContractError, NumericalError)) as delivered:
                owner(tuple(-value for value in point))
        assert tuple(row[1] for row in delivered.value.__dict__["status_diagnostics"]) == (8, 1)
        assert clears == Counter(dict.fromkeys(statuses, 1))
        for pose, before in zip(poses, previous, strict=True):
            np.testing.assert_array_equal(pose.numpy(), before)
        assert all(status.numpy().tolist() == [0] for _, status in checkpoint._entries)
        _same_result(owner(point), expected)
