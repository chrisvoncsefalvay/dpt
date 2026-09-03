"""Real CUDA acceptance for the bounded per-ray pose diagnostic."""

# Pytest's approx annotations are incomplete; the production boundary stays strict.
# pyright: reportUnknownMemberType=false

import importlib
import math
from typing import Any

import pytest

from dpt.autodiff import FirstOrderPass
from dpt.contracts import ContractError, NumericalError
from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose
from dpt.projection import (
    ProjectionSpec,
    ProjectionWorkspace,
    prepare_projection,
    project_optical_depth,
    projection_pose_sensitivities,
    projection_vjp,
)
from dpt.validation.projection import integrate_sampled_field, quadratic_sample_values
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


def _case(
    shape: tuple[int, int],
    *,
    active_pose: bool = True,
    precision: Any = "float32",
    integration: Any = "midpoint",
) -> tuple[ProjectionWorkspace, Any, Any, RigidTransform, list[float]]:
    grid = GridSpec((7, 9, 11), (0.6, 0.8, 1.0), (-3.0, -3.0, -3.0))
    geometry = DetectorGeometry(
        (-16.0, 0.21, 0.37),
        (16.0, -4.0, -3.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.137, 0.19),
        shape,
    )
    transform = compose_pose(RigidTransform(), (0.13, -0.08, 0.07, 0.02, -0.03, 0.01))
    workspace = prepare_projection(
        grid,
        geometry,
        ProjectionSpec(47, active_pose=active_pose, precision=precision, integration=integration),
    )
    mu = wp.array(
        quadratic_sample_values(grid, intercept=0.03, curvature=(0.001, 0.002, 0.003)),
        dtype=wp.float32,
        device="cuda:0",
    )
    pose = wp.array(transform.packed(), dtype=wp.float64, device="cuda:0")
    return workspace, mu, pose, transform, mu.numpy().tolist()


@pytest.mark.parametrize(
    "precision,integration", [("float32", "midpoint"), ("float64", "cell_gauss")]
)
@pytest.mark.parametrize("shape", [(1, 1), (1, 127), (1, 129), (9, 17)])
def test_rows_equal_basis_vjps_and_contract_to_ordinary_vjp(
    shape: tuple[int, int], precision: Any, integration: Any
) -> None:
    workspace, mu, pose, _, _ = _case(shape, precision=precision, integration=integration)
    pixels = workspace.geometry.pixels
    seed_values = [0.0 if p % 11 == 3 else (-1.0) ** p * (0.5 + p % 7 / 8) for p in range(pixels)]
    seeds = wp.array(seed_values, dtype=workspace.dtype, device="cuda:0")
    output = wp.full(6 * pixels, float("nan"), dtype=wp.float64, device="cuda:0")
    projection_pose_sensitivities(
        mu, pose, cotangent=seeds, out_jacobian=output, workspace=workspace
    )
    actual = output.numpy().reshape(pixels, 6)
    assert all(math.isfinite(float(x)) for x in actual.flat)

    basis = wp.zeros(pixels, dtype=workspace.dtype, device="cuda:0")
    gradient = wp.empty(6, dtype=wp.float64, device="cuda:0")
    staging = wp.zeros(pixels, dtype=workspace.dtype, device="cpu", pinned=True)
    host = staging.numpy()
    for p in range(pixels):
        host[:] = 0
        host[p] = seed_values[p]
        wp.copy(basis, staging, stream=workspace.stream)
        projection_vjp(mu, pose, cotangent=basis, out_pose_tangent=gradient, workspace=workspace)
        assert actual[p].tolist() == pytest.approx(gradient.numpy().tolist(), rel=2e-13, abs=2e-13)
        if seed_values[p] == 0:
            assert actual[p].tolist() == [0.0] * 6
    projection_vjp(mu, pose, cotangent=seeds, out_pose_tangent=gradient, workspace=workspace)
    contracted = [math.fsum(float(actual[p, axis]) for p in range(pixels)) for axis in range(6)]
    assert contracted == pytest.approx(gradient.numpy().tolist(), rel=2e-12, abs=2e-12)
    depth = wp.empty(pixels, dtype=workspace.dtype, device="cuda:0")
    project_optical_depth(mu, pose, out_L=depth, workspace=workspace)
    for p, value in enumerate(depth.numpy()):
        if value == 0:
            assert actual[p].tolist() == [0.0] * 6
    # Unchecked calls retain identical outputs, with an explicit status boundary.
    projection_pose_sensitivities(
        mu, pose, cotangent=seeds, out_jacobian=output, workspace=workspace, validate=False
    )
    workspace.check_status()
    assert output.numpy().tolist() == actual.ravel().tolist()


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_each_axis_agrees_with_independent_discrete_reference(integration: Any) -> None:
    workspace, mu, pose, transform, values = _case((2, 7), integration=integration)
    geometry = workspace.geometry
    assert isinstance(geometry, DetectorGeometry)
    pixels = workspace.geometry.pixels
    seeds = wp.ones(pixels, dtype=wp.float32, device="cuda:0")
    output = wp.empty(6 * pixels, dtype=wp.float64, device="cuda:0")
    projection_pose_sensitivities(
        mu, pose, cotangent=seeds, out_jacobian=output, workspace=workspace
    )
    measured = output.numpy().reshape(pixels, 6)
    for pixel in (0, 6, 7, 13):
        row, column = divmod(pixel, geometry.shape[1])
        for axis in range(6):
            errors: list[float] = []
            for step in (1e-4, 3e-5, 1e-5, 3e-6):
                delta = tuple(step if j == axis else 0.0 for j in range(6))
                plus = integrate_sampled_field(
                    values,
                    workspace.grid,
                    geometry,
                    compose_pose(transform, delta),
                    row,
                    column,
                    midpoint_samples=47 if integration == "midpoint" else None,
                )
                minus = integrate_sampled_field(
                    values,
                    workspace.grid,
                    geometry,
                    compose_pose(transform, tuple(-x for x in delta)),
                    row,
                    column,
                    midpoint_samples=47 if integration == "midpoint" else None,
                )
                errors.append(abs((plus - minus) / (2 * step) - float(measured[pixel, axis])))
            assert min(errors) < 2e-6 * max(1.0, abs(float(measured[pixel, axis]))), (
                pixel,
                axis,
                errors,
            )


def test_shape_dtype_device_alias_stream_and_lifetime_contracts() -> None:
    workspace, mu, pose, _, _ = _case((1, 2))
    seeds = wp.ones(2, dtype=wp.float32, device="cuda:0")
    output = wp.empty(12, dtype=wp.float64, device="cuda:0")
    for invalid in (
        wp.empty(11, dtype=wp.float64, device="cuda:0"),
        wp.empty(12, dtype=wp.float32, device="cuda:0"),
        wp.empty((2, 6), dtype=wp.float64, device="cuda:0"),
        wp.empty(12, dtype=wp.float64, device="cpu"),
    ):
        with pytest.raises(ContractError):
            projection_pose_sensitivities(
                mu, pose, cotangent=seeds, out_jacobian=invalid, workspace=workspace
            )
    with pytest.raises(ContractError, match="overlaps"):
        projection_pose_sensitivities(
            mu, pose, cotangent=seeds, out_jacobian=pose, workspace=workspace
        )
    with pytest.raises(ContractError, match="stream"):
        projection_pose_sensitivities(
            mu,
            pose,
            cotangent=seeds,
            out_jacobian=output,
            workspace=workspace,
            stream=wp.Stream("cuda:0"),
        )
    with wp.Tape(), pytest.raises(ContractError, match="ambient"):
        projection_pose_sensitivities(
            mu, pose, cotangent=seeds, out_jacobian=output, workspace=workspace
        )

    differentiable_pose = wp.array(
        pose.numpy(), dtype=wp.float64, device="cuda:0", requires_grad=True
    )
    depth = wp.empty(2, dtype=wp.float32, device="cuda:0", requires_grad=True)
    with FirstOrderPass([workspace]) as reverse:
        project_optical_depth(
            mu, differentiable_pose, out_L=depth, workspace=workspace, tape=reverse.tape
        )
    with pytest.raises(ContractError, match="outstanding"):
        projection_pose_sensitivities(
            mu, differentiable_pose, cotangent=seeds, out_jacobian=output, workspace=workspace
        )
    reverse.backward(depth, seeds)
    fixed, fixed_mu, fixed_pose, _, _ = _case((1, 2), active_pose=False)
    with pytest.raises(ContractError, match="pose is fixed"):
        projection_pose_sensitivities(
            fixed_mu, fixed_pose, cotangent=seeds, out_jacobian=output, workspace=fixed
        )


def test_nonfinite_seed_and_geometry_status_are_rejected() -> None:
    workspace, mu, pose, _, _ = _case((1, 2))
    seeds = wp.array([1.0, float("nan")], dtype=wp.float32, device="cuda:0")
    output = wp.full(12, 123.0, dtype=wp.float64, device="cuda:0")
    with pytest.raises(NumericalError, match="non-finite"):
        projection_pose_sensitivities(
            mu, pose, cotangent=seeds, out_jacobian=output, workspace=workspace
        )
    assert output.numpy().tolist() == [123.0] * 12
    seeds.fill_(1.0)
    extreme = wp.array(
        (*RigidTransform().rotation, 1.7e308, -1.7e308, -1.7e308),
        dtype=wp.float64,
        device="cuda:0",
    )
    with pytest.raises(NumericalError, match="non-finite"):
        projection_pose_sensitivities(
            mu, extreme, cotangent=seeds, out_jacobian=output, workspace=workspace
        )
