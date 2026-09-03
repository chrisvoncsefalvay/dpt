"""Local pose covectors follow independent finite rotations of affine ray segments."""

import importlib
import math
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any, Literal

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError
from dpt.geometry import RayBundleGeometry, RigidTransform
from dpt.material_projection import (
    material_projection_vjp,
    prepare_material_projection,
    project_material_paths,
)
from dpt.projection import (
    ProjectionSpec,
    prepare_projection,
    project_optical_depth,
    projection_pose_sensitivities,
    projection_vjp,
)
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")

_SOURCES = ((-1.75, -0.625, 0.5), (-1.0, 0.5, -0.25), (-1.25, -1.25, 0.75))
_ENDS = ((1.125, 0.75, -0.25), (0.75, 1.5, 0.5), (1.5, -0.25, -0.5))
_SLOPES = ((1 / 64, -1 / 128, 3 / 256), (-1 / 128, 3 / 256, 1 / 256))
_INTERCEPTS = (0.125, 0.1875)
_SEEDS = ((1.25, 0.0, -0.75), (-0.5, 0.75, 0.0))


@pytest.fixture(autouse=True)
def _native_context(warp_device: str) -> Iterator[None]:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def _scene() -> tuple[GridSpec, RayBundleGeometry, RigidTransform, np.ndarray]:
    grid = GridSpec((7, 7, 7), (1.0, 1.0, 1.0), (-3.0, -3.0, -3.0))
    transform = RigidTransform((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.25, -0.5, 0.75))
    geometry = RayBundleGeometry(
        tuple(transform.point(point) for point in _SOURCES),
        tuple(transform.point(point) for point in _ENDS),
    )
    fields = np.asarray(
        [
            [
                intercept + slope[0] * (x - 3) + slope[1] * (y - 3) + slope[2] * (z - 3)
                for z in range(7)
                for y in range(7)
                for x in range(7)
            ]
            for intercept, slope in zip(_INTERCEPTS, _SLOPES, strict=True)
        ],
        dtype=np.float32,
    )
    return grid, geometry, transform, fields


def _rotated_point(point: tuple[float, ...], axis: int, angle: float) -> tuple[float, ...]:
    """Apply an inverse coordinate rotation without pose or cross-product helpers."""
    x, y, z = point
    cosine, sine = math.cos(angle), math.sin(angle)
    if axis == 0:
        return x, cosine * y + sine * z, -sine * y + cosine * z
    if axis == 1:
        return cosine * x - sine * z, y, sine * x + cosine * z
    return cosine * x + sine * y, -sine * x + cosine * y, z


def _scalar_objective(seeds: tuple[tuple[float, ...], ...], axis: int, step: float) -> float:
    terms = []
    for material, channel_seeds in enumerate(seeds):
        slope, intercept = _SLOPES[material], _INTERCEPTS[material]
        for source, end, seed in zip(_SOURCES, _ENDS, channel_seeds, strict=True):
            midpoint = tuple((first + last) / 2 for first, last in zip(source, end, strict=True))
            if axis < 3:
                midpoint = tuple(
                    value - (step if coordinate == axis else 0.0)
                    for coordinate, value in enumerate(midpoint)
                )
            else:
                midpoint = _rotated_point(midpoint, axis - 3, step)
            length = math.sqrt(
                sum((last - first) ** 2 for first, last in zip(source, end, strict=True))
            )
            value = intercept + sum(
                coefficient * coordinate
                for coefficient, coordinate in zip(slope, midpoint, strict=True)
            )
            terms.append(seed * length * value)
    return math.fsum(terms)


def _expected_tangent(seeds: tuple[tuple[float, ...], ...]) -> np.ndarray:
    step = 2e-5
    return np.asarray(
        [
            (_scalar_objective(seeds, axis, step) - _scalar_objective(seeds, axis, -step))
            / (2 * step)
            for axis in range(6)
        ]
    )


def _expected_paths(materials: int) -> np.ndarray:
    paths = []
    for slope, intercept in zip(_SLOPES[:materials], _INTERCEPTS[:materials], strict=True):
        for source, end in zip(_SOURCES, _ENDS, strict=True):
            midpoint = tuple((first + last) / 2 for first, last in zip(source, end, strict=True))
            length = math.sqrt(
                sum((last - first) ** 2 for first, last in zip(source, end, strict=True))
            )
            paths.append(
                length
                * (
                    intercept
                    + sum(
                        coefficient * coordinate
                        for coefficient, coordinate in zip(slope, midpoint, strict=True)
                    )
                )
            )
    return np.asarray(paths)


def _storage_chain(packed: np.ndarray, transform: RigidTransform) -> np.ndarray:
    rotation = np.asarray(transform.rotation).reshape(3, 3)
    matrix = packed[:9].reshape(3, 3)
    # Differentiate R exp(delta) with the three coordinate rotation generators.
    generators = (
        np.asarray([[0, 0, 0], [0, 0, -1], [0, 1, 0]]),
        np.asarray([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]),
        np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 0]]),
    )
    return np.r_[
        rotation.T @ packed[9:],
        [np.sum(matrix * (rotation @ generator)) for generator in generators],
    ]


def _projection_outputs(
    device: str,
    precision: Literal["float32", "float64"],
    integration: Literal["midpoint", "cell_gauss"],
    active_fields: bool,
) -> dict[str, np.ndarray]:
    grid, geometry, transform, fields = _scene()
    workspace = prepare_projection(
        grid,
        geometry,
        ProjectionSpec(
            13, precision=precision, integration=integration, active_volume=active_fields
        ),
        device=device,
    )
    field = wp.array(fields[0], dtype=wp.float32, device=device)
    pose = wp.array(transform.packed(), dtype=wp.float64, device=device)
    seeds = wp.array(_SEEDS[0], dtype=getattr(wp, precision), device=device)
    output = wp.empty(3, dtype=getattr(wp, precision), device=device)
    local = wp.empty(6, dtype=wp.float64, device=device)
    storage = wp.empty(12, dtype=wp.float64, device=device)
    rows = wp.empty(18, dtype=wp.float64, device=device)
    field_gradient = (
        wp.empty(grid.voxels, dtype=wp.float32, device=device) if active_fields else None
    )
    project_optical_depth(field, pose, workspace=workspace, out_L=output)
    projection_vjp(
        field,
        pose,
        cotangent=seeds,
        workspace=workspace,
        out_pose_tangent=local,
        out_mu=field_gradient,
    )
    first_fields = field_gradient.numpy().copy() if field_gradient is not None else np.asarray([])
    projection_vjp(
        field,
        pose,
        cotangent=seeds,
        workspace=workspace,
        out_pose_matrix=storage,
        out_mu=field_gradient,
    )
    projection_pose_sensitivities(
        field, pose, cotangent=seeds, workspace=workspace, out_jacobian=rows
    )
    return {
        "paths": output.numpy().copy(),
        "local": local.numpy().copy(),
        "storage_chain": _storage_chain(storage.numpy(), transform),
        "storage": storage.numpy().copy(),
        "rows": rows.numpy().reshape(3, 6).copy(),
        "fields": first_fields,
        "storage_fields": field_gradient.numpy().copy()
        if field_gradient is not None
        else np.asarray([]),
        "status": workspace._status.numpy().copy(),
    }


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("active_fields", [False, True])
def test_projection_pose_matches_independent_finite_rotations_and_storage_chain(
    precision: Literal["float32", "float64"],
    integration: Literal["midpoint", "cell_gauss"],
    active_fields: bool,
    warp_device: str,
) -> None:
    actual = _projection_outputs(warp_device, precision, integration, active_fields)
    np.testing.assert_allclose(
        actual["paths"], _expected_paths(1), rtol=2e-6 if precision == "float32" else 2e-13, atol=0
    )
    np.testing.assert_allclose(
        actual["local"], _expected_tangent((_SEEDS[0],)), rtol=2e-8, atol=2e-10
    )
    np.testing.assert_allclose(actual["local"], actual["storage_chain"], rtol=3e-12, atol=3e-13)
    np.testing.assert_allclose(
        actual["local"], np.sum(actual["rows"], axis=0), rtol=3e-12, atol=3e-13
    )
    np.testing.assert_array_equal(actual["rows"][1], 0.0)
    np.testing.assert_array_equal(actual["status"], 0)
    if active_fields:
        np.testing.assert_allclose(actual["fields"], actual["storage_fields"], rtol=2e-6, atol=2e-7)
        fields = _scene()[3][0].astype(np.float64)
        assert float(actual["fields"] @ fields) == pytest.approx(
            float(actual["paths"] @ np.asarray(_SEEDS[0])), rel=2e-6, abs=2e-7
        )


def _material_outputs(
    device: str,
    precision: Literal["float32", "float64"],
    integration: Literal["midpoint", "cell_gauss"],
    execution: Literal["serial", "fused"],
    active_fields: bool,
) -> dict[str, np.ndarray]:
    grid, geometry, transform, host = _scene()
    dtype = getattr(wp, precision)
    fields = wp.array(host, dtype=wp.float32, device=device)
    seeds = wp.array(_SEEDS, dtype=dtype, device=device)
    workspace = prepare_material_projection(
        grid,
        geometry,
        ProjectionSpec(
            13, precision=precision, integration=integration, active_volume=active_fields
        ),
        materials=2,
        fields=fields,
        cotangent_paths=seeds,
        out_paths=wp.empty((2, 3), dtype=dtype, device=device),
        out_fields=wp.empty((2, grid.voxels), dtype=wp.float32, device=device)
        if active_fields
        else None,
        device=device,
        channel_execution=execution,
    )
    pose = wp.array(transform.packed(), dtype=wp.float64, device=device)
    local = wp.empty(6, dtype=wp.float64, device=device)
    storage = wp.empty(12, dtype=wp.float64, device=device)
    project_material_paths(pose, workspace=workspace)
    material_projection_vjp(pose, workspace=workspace, out_pose_tangent=local)
    first_fields = (
        workspace.grad_fields.numpy().copy()
        if workspace.grad_fields is not None
        else np.asarray([])
    )
    material_projection_vjp(pose, workspace=workspace, out_pose_matrix=storage)
    return {
        "paths": workspace.paths.numpy().copy(),
        "local": local.numpy().copy(),
        "storage": storage.numpy().copy(),
        "storage_chain": _storage_chain(storage.numpy(), transform),
        "fields": first_fields,
        "storage_fields": workspace.grad_fields.numpy().copy()
        if workspace.grad_fields is not None
        else np.asarray([]),
        "status": workspace.status.numpy().copy(),
    }


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("execution", ["serial", "fused"])
@pytest.mark.parametrize("active_fields", [False, True])
def test_material_pose_matches_independent_finite_rotations_and_storage_chain(
    precision: Literal["float32", "float64"],
    integration: Literal["midpoint", "cell_gauss"],
    execution: Literal["serial", "fused"],
    active_fields: bool,
    warp_device: str,
) -> None:
    actual = _material_outputs(warp_device, precision, integration, execution, active_fields)
    np.testing.assert_allclose(
        actual["paths"], _expected_paths(2), rtol=2e-6 if precision == "float32" else 2e-13, atol=0
    )
    np.testing.assert_allclose(actual["local"], _expected_tangent(_SEEDS), rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(actual["local"], actual["storage_chain"], rtol=3e-12, atol=3e-13)
    np.testing.assert_array_equal(actual["status"], 0)
    if active_fields:
        np.testing.assert_allclose(actual["fields"], actual["storage_fields"], rtol=2e-6, atol=2e-7)
        fields = _scene()[3].reshape(-1).astype(np.float64)
        assert float(actual["fields"] @ fields) == pytest.approx(
            float(actual["paths"] @ np.asarray(_SEEDS).reshape(-1)), rel=2e-6, abs=2e-7
        )


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_inactive_pose_refuses_before_volume_or_pose_destination_writes(
    integration: Literal["midpoint", "cell_gauss"],
    warp_device: str,
) -> None:
    grid, geometry, transform, fields = _scene()
    workspace = prepare_projection(
        grid,
        geometry,
        ProjectionSpec(13, active_pose=False, active_volume=True, integration=integration),
        device=warp_device,
    )
    field = wp.array(fields[0], dtype=wp.float32, device=warp_device)
    pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device)
    seeds = wp.array(_SEEDS[0], dtype=wp.float32, device=warp_device)
    local = wp.full(6, 7.0, dtype=wp.float64, device=warp_device)
    gradient = wp.full(grid.voxels, 11.0, dtype=wp.float32, device=warp_device)
    with pytest.raises(ContractError, match="pose is fixed"):
        projection_vjp(
            field,
            pose,
            cotangent=seeds,
            workspace=workspace,
            out_pose_tangent=local,
            out_mu=gradient,
        )
    np.testing.assert_array_equal(local.numpy(), 7.0)
    np.testing.assert_array_equal(gradient.numpy(), 11.0)
