"""Deterministic voxel gathering matches independent projection matrices."""

import importlib
import math
import random
import struct
from typing import Any, Literal

import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.geometry import DetectorGeometry, RayBundleGeometry, RigidTransform, compose_pose
from dpt.projection import (
    ProjectionSpec,
    prepare_projection,
    project_optical_depth,
    projection_adjoint_gather,
    projection_vjp,
)
from dpt.validation.projection import integrate_ray_bundle, integrate_sampled_field
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


def _metadata(case: str) -> tuple[GridSpec, DetectorGeometry | RayBundleGeometry, RigidTransform]:
    if case == "clamped":
        return (
            GridSpec((1, 2, 3), (1.0, 0.7, 1.2), (-1.0, 0.0, 0.0)),
            RayBundleGeometry(
                ((-4.0, -0.35, 0.0), (-4.0, 0.9, 0.0), (0.1, 0.1, 0.0), (-4.0, 9.0, 0.0)),
                ((4.0, -0.35, 0.0), (4.0, 0.9, 0.0), (0.7, 0.3, 0.0), (4.0, 9.0, 0.0)),
            ),
            RigidTransform(),
        )
    grid = GridSpec((2, 3, 4), (1.2, 0.8, 1.1), (-1.8, -0.8, -0.55))
    transform = compose_pose(RigidTransform(), (0.07, -0.03, 0.02, 0.01, -0.02, 0.005))
    if case == "bundle":
        return (
            grid,
            RayBundleGeometry(
                (
                    (-8.0, 0.1, 0.2),
                    (8.0, 0.1, 0.2),
                    (-8.0, 9.0, 9.0),
                    (0.1, 0.2, 0.3),
                    (-1.0, -5.0, 0.2),
                    (-0.7, 0.4, -8.0),
                ),
                (
                    (8.0, 0.8, 0.7),
                    (-8.0, 0.8, 0.7),
                    (8.0, 9.0, 9.0),
                    (0.3, 0.6, 0.7),
                    (-1.0, 5.0, 0.3),
                    (0.2, 0.4, 8.0),
                ),
            ),
            transform,
        )
    return (
        grid,
        DetectorGeometry(
            (-8.0, 0.17, 0.23),
            (8.0, -0.43, -0.37),
            (math.sin(0.1), math.cos(0.1), 0.0),
            (0.0, 0.0, 1.0),
            (0.19, 0.23),
            (2, 5),
        ),
        transform,
    )


def _matrix(
    grid: GridSpec,
    geometry: DetectorGeometry | RayBundleGeometry,
    pose: RigidTransform,
    integration: str,
) -> list[list[float]]:
    rows: list[list[float]] = [[] for _ in range(geometry.pixels)]
    for voxel in range(grid.voxels):
        basis = [float(index == voxel) for index in range(grid.voxels)]
        for pixel in range(geometry.pixels):
            samples = 31 if integration == "midpoint" else None
            if isinstance(geometry, RayBundleGeometry):
                value = integrate_ray_bundle(
                    basis, grid, geometry, pose, pixel, midpoint_samples=samples
                )
            else:
                row, column = divmod(pixel, geometry.shape[1])
                value = integrate_sampled_field(
                    basis, grid, geometry, pose, row, column, midpoint_samples=samples
                )
            rows[pixel].append(value)
    return rows


@pytest.mark.parametrize("case", ["nonsquare", "bundle", "clamped"])
@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_gather_matches_matrix_random_adjoint_pairs_and_is_repeatable(
    case: str, integration: Literal["midpoint", "cell_gauss"], warp_device: str
) -> None:
    grid, geometry, transform = _metadata(case)
    matrix = _matrix(grid, geometry, transform, integration)
    workspace = prepare_projection(
        grid,
        geometry,
        ProjectionSpec(
            31,
            active_pose=False,
            active_volume=True,
            precision="float64",
            integration=integration,
            field_domain="signed_linear",
        ),
        device=warp_device,
    )
    pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device)
    image = wp.empty(geometry.pixels, dtype=wp.float64, device=warp_device)
    gradient = wp.empty(grid.voxels, dtype=wp.float64, device=warp_device)
    scatter = wp.empty(grid.voxels, dtype=wp.float64, device=warp_device)
    generator = random.Random(584291)
    for _ in range(10):
        values = [
            struct.unpack("f", struct.pack("f", generator.uniform(-1.0, 1.0)))[0]
            for _ in range(grid.voxels)
        ]
        seeds = [generator.uniform(-1.0, 1.0) for _ in range(geometry.pixels)]
        field = wp.array(values, dtype=wp.float32, device=warp_device)
        cotangent = wp.array(seeds, dtype=wp.float64, device=warp_device)
        project_optical_depth(field, pose, workspace=workspace, out_L=image)
        projection_adjoint_gather(pose, cotangent=cotangent, workspace=workspace, out_mu=gradient)
        actual = gradient.numpy().copy()
        expected = [
            math.fsum(row[voxel] * seed for row, seed in zip(matrix, seeds, strict=True))
            for voxel in range(grid.voxels)
        ]
        assert actual.tolist() == pytest.approx(expected, rel=2e-12, abs=2e-12)
        left = math.fsum(
            float(value) * seed for value, seed in zip(image.numpy(), seeds, strict=True)
        )
        right = math.fsum(value * float(slope) for value, slope in zip(values, actual, strict=True))
        assert abs(left - right) < 2e-12 * max(1.0, abs(left), abs(right))
        projection_adjoint_gather(pose, cotangent=cotangent, workspace=workspace, out_mu=gradient)
        assert actual.tobytes() == gradient.numpy().tobytes()
        projection_vjp(
            field,
            pose,
            cotangent=cotangent,
            workspace=workspace,
            out_mu=scatter,
            volume_gradient_precision="float64",
        )
        assert actual.tolist() == pytest.approx(scatter.numpy().tolist(), rel=2e-12, abs=2e-12)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_gather_owned_pose_and_accumulation(
    precision: Literal["float32", "float64"], warp_device: str
) -> None:
    grid, geometry, transform = _metadata("clamped")
    workspace = prepare_projection(
        grid,
        geometry,
        ProjectionSpec(
            31,
            active_pose=False,
            precision="float64",
        ),
        device=warp_device,
    )
    field = wp.zeros(grid.voxels, dtype=wp.float32, device=warp_device)
    pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device)
    prepared = workspace.prepare_inputs(field, pose)
    cotangent = wp.ones(geometry.pixels, dtype=wp.float64, device=warp_device)
    output = wp.empty(grid.voxels, dtype=getattr(wp, precision), device=warp_device)
    projection_adjoint_gather(
        prepared,
        cotangent=cotangent,
        workspace=workspace,
        out_mu=output,
        volume_gradient_precision=precision,
    )
    first = output.numpy().copy()
    projection_adjoint_gather(
        prepared,
        cotangent=cotangent,
        workspace=workspace,
        out_mu=output,
        volume_gradient_precision=precision,
        accumulate=True,
    )
    assert output.numpy().tolist() == pytest.approx((2 * first).tolist(), abs=2e-6)


def test_gather_rejects_unbounded_work_and_invalid_current_cotangent(warp_device: str) -> None:
    grid, geometry, transform = _metadata("clamped")
    workspace = prepare_projection(
        grid, geometry, ProjectionSpec(31, precision="float64"), device=warp_device
    )
    pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device)
    cotangent = wp.ones(geometry.pixels, dtype=wp.float64, device=warp_device)
    output = wp.full(grid.voxels, 19.0, dtype=wp.float64, device=warp_device)
    with pytest.raises(ContractError, match="max_work"):
        projection_adjoint_gather(
            pose, cotangent=cotangent, workspace=workspace, out_mu=output, max_work=1
        )
    assert output.numpy().tolist() == [19.0] * grid.voxels
    cotangent.fill_(float("nan"))
    with pytest.raises(NumericalError):
        projection_adjoint_gather(pose, cotangent=cotangent, workspace=workspace, out_mu=output)
    assert output.numpy().tolist() == [19.0] * grid.voxels


def test_gather_checks_accumulation_overflow_and_rejects_fast_policy(warp_device: str) -> None:
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((0.0, 0.0, -2.0),), ((0.0, 0.0, 2.0),))
    workspace = prepare_projection(
        grid, rays, ProjectionSpec(3, precision="float64"), device=warp_device
    )
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    cotangent = wp.array([1e308], dtype=wp.float64, device=warp_device)
    output = wp.full(1, 1e308, dtype=wp.float64, device=warp_device)
    with pytest.raises(NumericalError):
        projection_adjoint_gather(
            pose, cotangent=cotangent, workspace=workspace, out_mu=output, accumulate=True
        )
    workspace.spec = ProjectionSpec(3, active_pose=False, sampling_precision="fast_float32")
    with pytest.raises(ContractError, match="strict float64"):
        projection_adjoint_gather(pose, cotangent=cotangent, workspace=workspace, out_mu=output)
