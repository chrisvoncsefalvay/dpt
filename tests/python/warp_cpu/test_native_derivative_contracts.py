"""Native derivative shapes, analytic values, buffer ownership and failure recovery."""

from __future__ import annotations

# The pending-status regression deliberately inspects the workspace diagnostic.
# pyright: reportPrivateUsage=false
import importlib
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.autodiff import FirstOrderPass
from dpt.contracts import ContractError, NumericalError
from dpt.detector import (
    BlurSpec,
    CalibrationSpec,
    blur_transpose,
    calibration_vjp,
    prepare_blur,
    prepare_detector,
)
from dpt.geometry import RayBundleGeometry, RigidTransform
from dpt.materials import Provenance
from dpt.objectives import ObjectiveSpec, evaluate_objective, prepare_objective
from dpt.projection import ProjectionSpec, prepare_projection, projection_vjp
from dpt.transmission import (
    TransmissionSpec,
    optical_depth_from_removed_vjp,
    prepare_transmission,
    transmission_vjp,
    transmit,
)
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")
PROVENANCE = Provenance(
    "analytic:derivative-fixture", "a" * 64, "CC0-1.0", "Analytic operator inputs"
)


@pytest.fixture(autouse=True)
def _execution_context(warp_device: str) -> Iterator[None]:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def _array(values: Any, device: str, precision: str = "float64") -> Any:
    return wp.array(values, dtype=getattr(wp, precision), device=device)


def _projection(device: str) -> Any:
    grid = GridSpec((3, 3, 3), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0))
    geometry = RayBundleGeometry(((-0.6, -0.3, -0.2),), ((0.7, 0.4, 0.3),))
    workspace = prepare_projection(
        grid, geometry, ProjectionSpec(7, active_volume=True, precision="float64"), device=device
    )
    field = _array([0.125 + i / 1024 for i in range(grid.voxels)], device, "float32")
    return workspace, field, _array(RigidTransform().packed(), device), _array([-1.25], device)


@pytest.mark.parametrize("name,size", [("out_pose_tangent", 12), ("out_pose_matrix", 6)])
def test_pose_destinations_enforce_six_vs_twelve_shapes(
    name: str, size: int, warp_device: str
) -> None:
    workspace, field, pose, covector = _projection(warp_device)
    output = wp.full(size, 91.0, dtype=wp.float64, device=warp_device)
    with pytest.raises(ContractError, match="shape"):
        projection_vjp(field, pose, cotangent=covector, workspace=workspace, **{name: output})
    np.testing.assert_array_equal(output.numpy(), [91.0] * size)


def test_detector_calibration_and_blur_transpose_match_analytic_results(warp_device: str) -> None:
    workspace = prepare_detector(max_pixels=3, device=warp_device)
    spec = CalibrationSpec("ADU", active_exposure=True, active_offset=True)
    arrays = [_array(values, warp_device, "float32") for values in ([0, 2, 10], [3], [4], [-2])]
    covector = _array([1, -2, 0.5], warp_device, "float32")
    mean_gradient = wp.empty(3, dtype=wp.float32, device=warp_device)
    exposure_gradient = wp.empty(1, dtype=wp.float64, device=warp_device)
    offset_gradient = wp.empty(1, dtype=wp.float64, device=warp_device)
    calibration_vjp(
        *arrays,
        cotangent=covector,
        out_grad_mean=mean_gradient,
        out_grad_exposure=exposure_gradient,
        out_grad_offset=offset_gradient,
        workspace=workspace,
        spec=spec,
    )
    np.testing.assert_array_equal(mean_gradient.numpy(), [12, -24, 6])
    np.testing.assert_array_equal(exposure_gradient.numpy(), [3])
    np.testing.assert_array_equal(offset_gradient.numpy(), [-0.5])
    blur_workspace = prepare_blur(
        BlurSpec(1, 3, 1, 3, (0.125, 0.5, 0.375), PROVENANCE), device=warp_device
    )
    blur_transpose(cotangent=covector, out_grad_signal=mean_gradient, workspace=blur_workspace)
    matrix = np.array([[0.5, 0.375, 0], [0.125, 0.5, 0.375], [0, 0.125, 0.5]])
    np.testing.assert_array_equal(mean_gradient.numpy(), matrix.T @ np.array([1, -2, 0.5]))
    np.testing.assert_array_equal(covector.numpy(), [1, -2, 0.5])


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_objective_loss_only_leaves_previous_cotangent_unchanged(
    precision: str, warp_device: str
) -> None:
    workspace = prepare_objective(
        ObjectiveSpec("squared_error", "counts", precision=precision),
        max_pixels=2,
        device=warp_device,
    )
    prediction, observation = (
        _array([2, 5], warp_device, precision),
        _array([1, 3], warp_device, "float32"),
    )
    loss = wp.empty(1, dtype=wp.float64, device=warp_device)
    covector = wp.empty(2, dtype=getattr(wp, precision), device=warp_device)
    evaluate_objective(
        prediction, observation, out_loss=loss, out_cotangent=covector, workspace=workspace
    )
    np.testing.assert_array_equal(loss.numpy(), [2.5])
    np.testing.assert_array_equal(covector.numpy(), [1, 2])
    for kwargs in ({}, {"out_cotangent": None}):
        evaluate_objective(prediction, observation, out_loss=loss, workspace=workspace, **kwargs)
        np.testing.assert_array_equal(loss.numpy(), [2.5])
        np.testing.assert_array_equal(covector.numpy(), [1, 2])


def test_first_order_pass_propagates_the_signed_log_transmission_covector(warp_device: str) -> None:
    workspace = prepare_transmission(max_pixels=2, device=warp_device)
    depth = wp.ones(2, dtype=wp.float32, device=warp_device, requires_grad=True)
    output = wp.empty(2, dtype=wp.float32, device=warp_device, requires_grad=True)
    covector = _array([2, -3], warp_device, "float32")
    with FirstOrderPass([workspace]) as reverse:
        transmit(depth, out_log_T=output, workspace=workspace, tape=reverse.tape)
    reverse.backward(output, cotangent=covector)
    np.testing.assert_array_equal(depth.grad.numpy(), [-2, 3])
    np.testing.assert_array_equal(covector.numpy(), [2, -3])
    reverse.close()


@pytest.mark.parametrize("beam", ["none", "scalar", "device-scalar", "per-pixel"])
@pytest.mark.parametrize("zero", [False, True])
def test_transmission_combines_output_cotangents_and_accepts_none(
    beam: str, zero: bool, warp_device: str
) -> None:
    workspace = prepare_transmission(
        TransmissionSpec(beam=beam, active_beam=beam in ("device-scalar", "per-pixel")),
        max_pixels=2,
        device=warp_device,
    )
    depth = _array([0.25, 0.5], warp_device, "float32")
    n0 = (
        None
        if beam == "none"
        else 5.0
        if beam == "scalar"
        else _array([5.0] if beam == "device-scalar" else [5.0, 7.0], warp_device, "float32")
    )
    values = {
        "cotangent_T": [0.5, -0.75],
        "cotangent_counts": [-1.25, 0.25],
        "cotangent_log_T": [0.125, -0.25],
        "cotangent_removed": [-0.5, 1.5],
    }
    covectors = {
        key: None
        if zero or (beam == "none" and key == "cotangent_counts")
        else _array(value, warp_device, "float32")
        for key, value in values.items()
    }
    output = wp.empty(2, dtype=wp.float32, device=warp_device)
    beam_gradient = (
        None
        if beam in ("none", "scalar")
        else wp.empty(1 if beam == "device-scalar" else 2, dtype=wp.float32, device=warp_device)
    )
    transmission_vjp(
        depth, n0, **covectors, workspace=workspace, out_grad_L=output, out_grad_n0=beam_gradient
    )
    transmission = np.exp(-np.array([0.25, 0.5]))
    counts = np.zeros(2) if beam == "none" else np.array(values["cotangent_counts"])
    intensity = np.array([5.0, 7.0]) if beam == "per-pixel" else np.full(2, 5.0)
    expected = (
        np.zeros(2)
        if zero
        else transmission
        * (
            -np.array(values["cotangent_T"])
            - counts * intensity
            + np.array(values["cotangent_removed"])
        )
        - np.array(values["cotangent_log_T"])
    )
    np.testing.assert_allclose(output.numpy(), expected, rtol=2e-7, atol=1e-7)
    if beam_gradient is not None:
        expected_beam = np.zeros(2) if zero else counts * transmission
        if beam == "device-scalar":
            expected_beam = np.array([expected_beam.sum()])
        np.testing.assert_allclose(beam_gradient.numpy(), expected_beam, rtol=2e-7, atol=1e-7)


def test_inverse_depth_cotangent_matches_analytic_values(warp_device: str) -> None:
    workspace = prepare_transmission(max_pixels=2, device=warp_device)
    removed = _array([0.25, 0.5], warp_device, "float32")
    covector = _array([1.5, -0.75], warp_device, "float32")
    output = wp.empty(2, dtype=wp.float32, device=warp_device)
    optical_depth_from_removed_vjp(
        removed, cotangent=covector, workspace=workspace, out_grad_delta=output
    )
    np.testing.assert_array_equal(output.numpy(), [2, -1.5])


def test_pending_projection_failure_preserves_destination_and_allows_checked_retry(
    warp_device: str,
) -> None:
    workspace, field, pose, covector = _projection(warp_device)
    output = wp.full(6, 91.0, dtype=wp.float64, device=warp_device)
    workspace._status.fill_(2)
    with pytest.raises(NumericalError):
        projection_vjp(
            field, pose, workspace=workspace, cotangent=covector, out_pose_tangent=output
        )
    np.testing.assert_array_equal(output.numpy(), [91.0] * 6)
    with pytest.raises(NumericalError):
        workspace.consume_status()
    projection_vjp(field, pose, workspace=workspace, cotangent=covector, out_pose_tangent=output)
    fresh, fresh_field, fresh_pose, fresh_covector = _projection(warp_device)
    reference = wp.empty_like(output)
    projection_vjp(
        fresh_field,
        fresh_pose,
        workspace=fresh,
        cotangent=fresh_covector,
        out_pose_tangent=reference,
    )
    np.testing.assert_allclose(output.numpy(), reference.numpy(), rtol=3e-7, atol=1e-12)
