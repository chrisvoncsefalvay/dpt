"""Fixed-pose mixed sampling is a qualified linear approximation with a matched transpose."""

import importlib

# pyright: reportUnknownMemberType=false
import math
import struct
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any, cast

import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError
from dpt.geometry import RayBundleGeometry, RigidTransform, compose_pose
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.validation.projection import _clip_segment
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


@pytest.fixture(autouse=True)
def _kernel_context(warp_device: str) -> Iterator[None]:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _matrix(
    grid: GridSpec, bundle: RayBundleGeometry, pose: RigidTransform, samples: int
) -> list[list[float]]:
    """Independent binary32 rounding/FMA and corner coefficients, no Warp code."""
    columns: list[list[float]] = [[] for _ in range(grid.voxels)]
    xyz = tuple(reversed(grid.shape))
    for ray in range(bundle.rays):
        start, direction, lower, upper, length = _clip_segment(grid, pose, *bundle.segment(ray))
        contributions: list[list[float]] = [[] for _ in range(grid.voxels)]
        if upper > lower:
            step = (upper - lower) / samples
            entry = [_f32(start[i] + lower * direction[i]) for i in range(3)]
            delta = [_f32(step * direction[i]) for i in range(3)]
            for sample in range(samples):
                offset = _f32(_f32(float(sample)) + 0.5)
                # Binary64 exactly holds the binary32 products in this bounded
                # fixture; rounding the sum once models the explicit FP32 FMA.
                point = [_f32(entry[i] + offset * delta[i]) for i in range(3)]
                if any(point[i] < -0.5 or point[i] > xyz[i] - 0.5 for i in range(3)):
                    continue
                q = [_f32(min(max(point[i], 0.0), xyz[i] - 1)) for i in range(3)]
                lo = [min(math.floor(q[i]), max(xyz[i] - 2, 0)) for i in range(3)]
                fraction = [_f32(q[i] - _f32(float(lo[i]))) for i in range(3)]
                for corner in range(8):
                    index = [min(lo[i] + ((corner >> i) & 1), xyz[i] - 1) for i in range(3)]
                    weight = 1.0
                    for axis in range(3):
                        factor = (
                            fraction[axis] if ((corner >> axis) & 1) else _f32(1.0 - fraction[axis])
                        )
                        weight = _f32(weight * factor)
                    contributions[(index[2] * xyz[1] + index[1]) * xyz[0] + index[0]].append(
                        length * step * weight
                    )
        for voxel, terms in enumerate(contributions):
            columns[voxel].append(math.fsum(terms))
    return columns


def _metadata() -> tuple[GridSpec, RayBundleGeometry, RigidTransform]:
    grid = GridSpec((3, 4, 5), (1.2, 0.8, 1.1), (-2.0, -1.0, -1.0))
    bundle = RayBundleGeometry(
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
    )
    transform = compose_pose(RigidTransform(), (0.07, -0.03, 0.02, 0.01, -0.02, 0.005))
    return grid, bundle, transform


@pytest.mark.parametrize("samples", [17, 97])
@pytest.mark.parametrize("gradient_precision", ["float32", "float64"])
def test_mixed_signed_forward_and_transpose_match_independent_rounded_matrix(
    samples: int, gradient_precision: Any, warp_device: str
) -> None:
    grid, bundle, transform = _metadata()
    matrix = _matrix(grid, bundle, transform, samples)
    workspace = prepare_projection(
        grid,
        bundle,
        ProjectionSpec(
            samples,
            active_pose=False,
            active_volume=True,
            precision="float64",
            field_domain="signed_linear",
            sampling_precision="mixed_float32",
        ),
        device=warp_device,
    )
    values = [(-1.0) ** i * (i + 1) / 64 for i in range(grid.voxels)]
    seed_values = [(-1.0) ** i * (i + 1) / 8 for i in range(bundle.rays)]
    field = wp.array(values, dtype=wp.float32, device=warp_device)
    pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device)
    seeds = wp.array(seed_values, dtype=wp.float64, device=warp_device)
    output = wp.empty(bundle.rays, dtype=wp.float64, device=warp_device)
    gradient = wp.empty(grid.voxels, dtype=getattr(wp, gradient_precision), device=warp_device)
    project_optical_depth(field, pose, workspace=workspace, out_L=output)
    projection_vjp(
        field,
        pose,
        cotangent=seeds,
        workspace=workspace,
        out_mu=gradient,
        volume_gradient_precision=gradient_precision,
    )
    forward = [
        math.fsum(column[ray] * value for column, value in zip(matrix, values, strict=True))
        for ray in range(bundle.rays)
    ]
    transpose = [
        math.fsum(a * b for a, b in zip(column, seed_values, strict=True)) for column in matrix
    ]
    assert output.numpy().tolist() == pytest.approx(forward, rel=3e-13, abs=3e-14)
    tolerance = 3e-6 if gradient_precision == "float32" else 3e-13
    assert gradient.numpy().tolist() == pytest.approx(transpose, rel=tolerance, abs=tolerance)
    lhs = math.fsum(float(a) * b for a, b in zip(output.numpy(), seed_values, strict=True))
    rhs = math.fsum(float(a) * b for a, b in zip(gradient.numpy(), values, strict=True))
    assert lhs == pytest.approx(rhs, rel=tolerance, abs=tolerance)


@pytest.mark.parametrize("samples", [17, 97])
@pytest.mark.parametrize("scale", [2.0**-100, 1.0, 2.0**100])
def test_nonuniform_mixed_error_against_strict_float64_is_qualified(
    samples: int, scale: float, warp_device: str
) -> None:
    grid, bundle, transform = _metadata()
    values = [
        scale * (0.07 + 0.012 * x + 0.019 * y + 0.011 * z + 0.003 * x * y)
        for z in range(grid.shape[0])
        for y in range(grid.shape[1])
        for x in range(grid.shape[2])
    ]
    field = wp.array(values, dtype=wp.float32, device=warp_device)
    pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device)
    actual = []
    for policy in ("float64", "mixed_float32"):
        workspace = prepare_projection(
            grid,
            bundle,
            ProjectionSpec(
                samples,
                active_pose=False,
                precision="float64",
                sampling_precision=cast(Any, policy),
            ),
            device=warp_device,
        )
        output = wp.empty(bundle.rays, dtype=wp.float64, device=warp_device)
        project_optical_depth(field, pose, workspace=workspace, out_L=output)
        actual.append(output.numpy().tolist())
    maximum = max(abs(a) for a in actual[0])
    error = max(abs(a - b) for a, b in zip(actual[0], actual[1], strict=True))
    assert error / maximum < 2e-6
    assert any(a != b for a, b in zip(actual[0], actual[1], strict=True))


def test_mixed_policy_rejects_pose_derivatives_exact_cell_claims_and_unsafe_axis_counts(
    warp_device: str,
) -> None:
    with pytest.raises(ContractError, match="fixed pose"):
        ProjectionSpec(1, sampling_precision="mixed_float32")
    with pytest.raises(ContractError, match="midpoint"):
        ProjectionSpec(
            1, active_pose=False, integration="cell_gauss", sampling_precision="mixed_float32"
        )
    with pytest.raises(ContractError, match="sampling_precision"):
        ProjectionSpec(1, sampling_precision=cast(Any, "half"))
    _, bundle, _ = _metadata()
    with pytest.raises(ContractError, match="axis counts"):
        prepare_projection(
            GridSpec((1, 1, 2**24 + 1), (1.0, 1.0, 1.0)),
            bundle,
            ProjectionSpec(1, active_pose=False, sampling_precision="mixed_float32"),
            device=warp_device,
        )
