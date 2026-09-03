"""Signed linear fields and explicit volume-cotangent storage use matched pairs."""

import importlib

# pyright: reportUnknownMemberType=false
import math
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any, cast

import pytest

from dpt._runtime import cpu_test_context
from dpt.autodiff import FirstOrderPass
from dpt.contracts import ContractError, NumericalError, TrialDomainError
from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


@pytest.fixture(autouse=True)
def _kernel_context(warp_device: str) -> Iterator[None]:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def _metadata() -> tuple[GridSpec, DetectorGeometry, RigidTransform]:
    grid = GridSpec((2, 3, 4), (1.2, 0.8, 1.1), (-1.8, -0.8, -0.55))
    detector = DetectorGeometry(
        (-8.0, 0.17, 0.23),
        (8.0, -0.43, -0.37),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.19, 0.23),
        (3, 4),
    )
    pose = compose_pose(RigidTransform(), (0.07, -0.03, 0.02, 0.01, -0.02, 0.005))
    return grid, detector, pose


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("gradient_precision", ["float32", "float64"])
def test_signed_forward_and_volume_transpose_against_independent_matrix(
    integration: Any, gradient_precision: Any, warp_device: str
) -> None:
    grid, detector, transform = _metadata()
    columns = [
        [
            integrate_sampled_field(
                [float(i == voxel) for i in range(grid.voxels)],
                grid,
                detector,
                transform,
                row,
                col,
                midpoint_samples=47 if integration == "midpoint" else None,
            )
            for row in range(detector.shape[0])
            for col in range(detector.shape[1])
        ]
        for voxel in range(grid.voxels)
    ]
    workspace = prepare_projection(
        grid,
        detector,
        ProjectionSpec(
            47,
            active_pose=False,
            active_volume=True,
            integration=integration,
            precision="float64",
            field_domain="signed_linear",
        ),
        device=warp_device,
    )
    values = [(-1.0) ** i * (i + 1) / 32 for i in range(grid.voxels)]
    seeds = [(-1.0) ** p * (p + 1) / 16 for p in range(detector.pixels)]
    field = wp.array(values, dtype=wp.float32, device=warp_device)
    pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device)
    output = wp.empty(detector.pixels, dtype=wp.float64, device=warp_device)
    cotangent = wp.array(seeds, dtype=wp.float64, device=warp_device)
    gradient = wp.empty(grid.voxels, dtype=getattr(wp, gradient_precision), device=warp_device)
    project_optical_depth(field, pose, workspace=workspace, out_L=output)
    expected_forward = [
        math.fsum(column[p] * value for column, value in zip(columns, values, strict=True))
        for p in range(detector.pixels)
    ]
    assert output.numpy().tolist() == pytest.approx(expected_forward, rel=3e-13, abs=3e-14)
    projection_vjp(
        field,
        pose,
        cotangent=cotangent,
        workspace=workspace,
        out_mu=gradient,
        volume_gradient_precision=gradient_precision,
    )
    expected_gradient = [
        math.fsum(a * b for a, b in zip(col, seeds, strict=True)) for col in columns
    ]
    tolerance = 2e-6 if gradient_precision == "float32" else 3e-13
    assert gradient.numpy().tolist() == pytest.approx(
        expected_gradient, rel=tolerance, abs=tolerance
    )
    # The finite-difference objective is built from independently integrated
    # signed fields, avoiding an adjoint/forward self-consistency-only check.
    direction = [(-1.0) ** (i // 3) for i in range(grid.voxels)]
    # The objective is linear, so a larger exactly representable step reduces
    # subtraction roundoff without introducing finite-difference truncation.
    step = 2.0**-6

    def objective(sign: float) -> float:
        perturbed = [v + sign * step * d for v, d in zip(values, direction, strict=True)]
        return math.fsum(
            seeds[row * detector.shape[1] + col]
            * integrate_sampled_field(
                perturbed,
                grid,
                detector,
                transform,
                row,
                col,
                midpoint_samples=47 if integration == "midpoint" else None,
                field_domain="signed_linear",
            )
            for row in range(detector.shape[0])
            for col in range(detector.shape[1])
        )

    finite_difference = (objective(1.0) - objective(-1.0)) / (2 * step)
    measured = math.fsum(float(g) * d for g, d in zip(gradient.numpy(), direction, strict=True))
    assert measured == pytest.approx(finite_difference, rel=tolerance, abs=tolerance)


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_double_volume_gradient_preserves_small_accumulated_increments(
    integration: Any, warp_device: str
) -> None:
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    detector = DetectorGeometry(
        (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0, 1, 0), (0, 0, 1), (1e-12, 1.0), (1, 257)
    )
    workspace = prepare_projection(
        grid,
        detector,
        ProjectionSpec(1, active_pose=False, active_volume=True, integration=integration),
        device=warp_device,
    )
    field = wp.ones(1, dtype=wp.float32, device=warp_device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    seeds = wp.ones(detector.pixels, dtype=wp.float32, device=warp_device)
    for precision in ("float32", "float64"):
        gradient = wp.full(1, 2.0**40, dtype=getattr(wp, precision), device=warp_device)
        projection_vjp(
            field,
            pose,
            cotangent=seeds,
            workspace=workspace,
            out_mu=gradient,
            accumulate=True,
            volume_gradient_precision=precision,
        )
        increment = float(gradient.numpy()[0]) - 2.0**40
        assert increment == (257.0 if precision == "float64" else 0.0)


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_double_destination_accepts_representable_cotangents_beyond_float32(
    integration: Any, warp_device: str
) -> None:
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    detector = DetectorGeometry(
        (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0, 1, 0), (0, 0, 1), (1.0, 1.0), (1, 1)
    )
    workspace = prepare_projection(
        grid,
        detector,
        ProjectionSpec(
            1, active_pose=False, active_volume=True, precision="float64", integration=integration
        ),
        device=warp_device,
    )
    field = wp.ones(1, dtype=wp.float32, device=warp_device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    seeds = wp.array([2.0**200], dtype=wp.float64, device=warp_device)
    gradient = wp.empty(1, dtype=wp.float64, device=warp_device)
    projection_vjp(
        field,
        pose,
        cotangent=seeds,
        workspace=workspace,
        out_mu=gradient,
        volume_gradient_precision="float64",
    )
    assert float(gradient.numpy()[0]) == 2.0**200
    single_gradient = wp.empty(1, dtype=wp.float32, device=warp_device)
    with pytest.raises(NumericalError, match="non-finite"):
        projection_vjp(field, pose, cotangent=seeds, workspace=workspace, out_mu=single_gradient)


@pytest.mark.parametrize("domain", ["physical", "signed_linear"])
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_all_field_domains_validate_nonfinite_samples_before_writing(
    domain: Any, invalid: float, warp_device: str
) -> None:
    grid, detector, transform = _metadata()
    workspace = prepare_projection(
        grid, detector, ProjectionSpec(3, field_domain=domain), device=warp_device
    )
    values = [0.1] * grid.voxels
    values[-1] = invalid
    field = wp.array(values, dtype=wp.float32, device=warp_device)
    pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device)
    output = wp.full(detector.pixels, 123.0, dtype=wp.float32, device=warp_device)
    with pytest.raises(TrialDomainError if domain == "physical" else NumericalError):
        project_optical_depth(field, pose, workspace=workspace, out_L=output)
    assert output.numpy().tolist() == [123.0] * detector.pixels


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_signed_field_tape_volume_gradient_retains_original_float32_storage(
    integration: Any, warp_device: str
) -> None:
    grid, detector, transform = _metadata()
    workspace = prepare_projection(
        grid,
        detector,
        ProjectionSpec(
            17,
            active_pose=False,
            active_volume=True,
            precision="float64",
            integration=integration,
            field_domain="signed_linear",
        ),
        device=warp_device,
    )
    field = wp.array(
        [(-1.0) ** i / 16 for i in range(grid.voxels)],
        dtype=wp.float32,
        device=warp_device,
        requires_grad=True,
    )
    pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device)
    output = wp.empty(detector.pixels, dtype=wp.float64, device=warp_device, requires_grad=True)
    seeds = wp.ones(detector.pixels, dtype=wp.float64, device=warp_device)
    with FirstOrderPass([workspace]) as reverse:
        project_optical_depth(field, pose, workspace=workspace, out_L=output, tape=reverse.tape)
    reverse.backward(output, seeds)
    assert field.grad.dtype == wp.float32
    expected = wp.empty(grid.voxels, dtype=wp.float64, device=warp_device)
    projection_vjp(
        field,
        pose,
        cotangent=seeds,
        workspace=workspace,
        out_mu=expected,
        volume_gradient_precision="float64",
    )
    assert field.grad.numpy().tolist() == pytest.approx(
        expected.numpy().tolist(), rel=2e-6, abs=2e-6
    )
    reverse.close()


def test_explicit_precision_and_domain_options_reject_unsupported_values(warp_device: str) -> None:
    with pytest.raises(ContractError, match="field_domain"):
        ProjectionSpec(1, field_domain="signed")  # type: ignore[arg-type]
    grid, detector, transform = _metadata()
    workspace = prepare_projection(
        grid, detector, ProjectionSpec(1, active_volume=True), device=warp_device
    )
    field = wp.ones(grid.voxels, dtype=wp.float32, device=warp_device)
    pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device)
    seeds = wp.ones(detector.pixels, dtype=wp.float32, device=warp_device)
    gradient = wp.empty(grid.voxels, dtype=wp.float64, device=warp_device)
    with pytest.raises(ContractError, match="dtype"):
        projection_vjp(field, pose, cotangent=seeds, workspace=workspace, out_mu=gradient)
    with pytest.raises(ContractError, match="volume_gradient_precision"):
        projection_vjp(
            field,
            pose,
            cotangent=seeds,
            workspace=workspace,
            out_mu=gradient,
            volume_gradient_precision=cast(Any, "mixed"),
        )
