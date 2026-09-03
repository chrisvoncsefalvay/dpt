"""Representable clipped-ray and weighted-loss outputs survive intermediate overflow."""

import importlib
from decimal import Decimal, localcontext
from typing import Any

import pytest

from dpt.contracts import NumericalError
from dpt.geometry import RayBundleGeometry, RigidTransform
from dpt.objectives import ObjectiveSpec, evaluate_primary_objective, prepare_objective
from dpt.projection import ProjectionSpec, prepare_projection, projection_vjp
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("seed_value", [1e308, -1e308])
@pytest.mark.parametrize("pose_coordinates", ["tangent", "matrix"])
def test_clipped_unit_chord_has_representable_volume_and_pose_cotangents(
    integration: Any, seed_value: float, pose_coordinates: str, warp_device: str
) -> None:
    # The full world segment is 4 mm, but the one-voxel chord is exactly 1 mm.
    # Its volume derivative is 1; R00 storage scales its chord as 1/R00.
    workspace = prepare_projection(
        GridSpec((1, 1, 1), (1.0, 1.0, 1.0)),
        RayBundleGeometry(((-2.0, 0.0, 0.0),), ((2.0, 0.0, 0.0),)),
        ProjectionSpec(17, active_volume=True, precision="float64", integration=integration),
        device=warp_device,
    )
    field = wp.ones(1, dtype=wp.float32, device=warp_device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    seed = wp.array([seed_value], dtype=wp.float64, device=warp_device)
    volume_gradient = wp.empty(1, dtype=wp.float64, device=warp_device)
    components = 12 if pose_coordinates == "matrix" else 6
    pose_gradient = wp.empty(components, dtype=wp.float64, device=warp_device)
    projection_vjp(
        field,
        pose,
        cotangent=seed,
        workspace=workspace,
        out_mu=volume_gradient,
        out_pose_matrix=pose_gradient if pose_coordinates == "matrix" else None,
        out_pose_tangent=pose_gradient if pose_coordinates == "tangent" else None,
        volume_gradient_precision="float64",
    )
    assert float(volume_gradient.numpy()[0]) == pytest.approx(seed_value, rel=8e-15)
    expected_pose = [-seed_value, *([0.0] * 11)] if components == 12 else [0.0] * 6
    assert pose_gradient.numpy().tolist() == pytest.approx(expected_pose, rel=8e-15, abs=0.0)


@pytest.mark.parametrize("domain", ["counts", "log_transmission"])
@pytest.mark.parametrize("amplitude,weight", [(1e160, 1e-30), (1e200, 0.0), (1e100, 1e-30)])
def test_fused_squared_error_retains_representable_weighted_loss_and_depth_seed(
    domain: Any, amplitude: float, weight: float, warp_device: str
) -> None:
    workspace = prepare_objective(
        ObjectiveSpec(domain=domain, weighted=True, precision="float64"),
        max_pixels=1,
        device=warp_device,
    )
    depth = wp.array(
        [0.0 if domain == "counts" else amplitude], dtype=wp.float64, device=warp_device
    )
    observation = wp.zeros(1, dtype=wp.float32, device=warp_device)
    weights = wp.array([weight], dtype=wp.float32, device=warp_device)
    prediction = wp.empty(1, dtype=wp.float64, device=warp_device)
    loss = wp.empty(1, dtype=wp.float64, device=warp_device)
    depth_seed = wp.empty(1, dtype=wp.float64, device=warp_device)
    evaluate_primary_objective(
        depth,
        observation,
        open_beam=amplitude,
        weights=weights,
        workspace=workspace,
        out_prediction=prediction,
        out_loss=loss,
        out_depth_cotangent=depth_seed,
    )
    with localcontext() as context:
        context.prec = 100
        exact_amplitude = Decimal.from_float(amplitude)
        exact_weight = Decimal.from_float(float(weights.numpy()[0]))
        expected_loss = float(Decimal("0.5") * exact_weight * exact_amplitude**2)
        expected_seed = float(
            -exact_weight * exact_amplitude**2
            if domain == "counts"
            else exact_weight * exact_amplitude
        )
    assert float(prediction.numpy()[0]) == (amplitude if domain == "counts" else -amplitude)
    assert float(loss.numpy()[0]) == pytest.approx(expected_loss, rel=4e-15, abs=0.0)
    assert float(depth_seed.numpy()[0]) == pytest.approx(expected_seed, rel=4e-15, abs=0.0)


def test_fused_squared_error_still_rejects_unrepresentable_final_outputs(warp_device: str) -> None:
    workspace = prepare_objective(
        ObjectiveSpec(domain="counts", weighted=True), max_pixels=1, device=warp_device
    )
    with pytest.raises(NumericalError):
        evaluate_primary_objective(
            wp.zeros(1, dtype=wp.float64, device=warp_device),
            wp.zeros(1, dtype=wp.float32, device=warp_device),
            open_beam=1e200,
            weights=wp.array([1e-30], dtype=wp.float32, device=warp_device),
            workspace=workspace,
            out_prediction=wp.empty(1, dtype=wp.float64, device=warp_device),
            out_loss=wp.empty(1, dtype=wp.float64, device=warp_device),
            out_depth_cotangent=wp.empty(1, dtype=wp.float64, device=warp_device),
        )
