"""Independent weights, clipping derivatives and origin stability for FP32."""

import math
from contextlib import nullcontext

import numpy as np
import pytest
import warp as wp

from dpt._runtime import cpu_test_context
from dpt.geometry import RayBundleGeometry, RigidTransform, compose_pose
from dpt.kernels import projection_fp32
from dpt.projection import ProjectionSpec, prepare_projection, projection_vjp
from dpt.validation.projection import _clip_segment
from dpt.volumes import GridSpec


@pytest.fixture(autouse=True)
def _context(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def _scene():
    angle = 0.19
    grid = GridSpec(
        (5, 6, 7),
        (1.2, 0.8, 1.1),
        (-3.2, -2.1, -1.7),
        (
            math.cos(angle),
            -math.sin(angle),
            0.0,
            math.sin(angle),
            math.cos(angle),
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )
    rays = RayBundleGeometry(
        (
            (-12.0, 0.17, 0.37),
            (12.0, 0.23, 0.48),
            (-0.7, -12.0, 0.29),
            (0.31, 0.41, -12.0),
            (-12.0, 30.0, 30.0),
        ),
        (
            (12.0, 0.81, 0.63),
            (-12.0, 0.89, 0.91),
            (0.2, 12.0, 0.86),
            (0.72, 0.83, 12.0),
            (12.0, 30.0, 30.0),
        ),
    )
    transform = compose_pose(RigidTransform(), (0.07, -0.03, 0.02, 0.013, -0.021, 0.009))
    values = np.asarray(
        [
            0.06 + 0.013 * x + 0.019 * y + 0.011 * z + 0.003 * x * y + 0.002 * y * z
            for z in range(5)
            for y in range(6)
            for x in range(7)
        ],
        dtype=np.float32,
    )
    return grid, rays, transform, values


def _fma32(a, b, c):
    # The product of two bounded FP32 inputs is exactly represented in FP64.
    return np.float32(float(a) * float(b) + float(c))


def _matrix(grid, rays, pose, samples):
    """Explicit rounded-coordinate interpolation matrix, independent of Warp."""
    matrix = np.zeros((rays.rays, grid.voxels), dtype=np.float64)
    shape = tuple(reversed(grid.shape))
    for pixel in range(rays.rays):
        origin, direction, _, _, length = _clip_segment(grid, pose, *rays.segment(pixel))
        origin, direction = (
            np.asarray(origin, dtype=np.float32),
            np.asarray(direction, dtype=np.float32),
        )
        lower, upper = np.float32(0), np.float32(1)
        for axis in range(3):
            low, high = np.float32(-0.5), np.float32(shape[axis] - 0.5)
            velocity = direction[axis]
            if velocity == 0:
                if origin[axis] < low or origin[axis] > high:
                    upper = np.float32(-1)
            else:
                first = np.float32(np.float32(low - origin[axis]) / velocity)
                last = np.float32(np.float32(high - origin[axis]) / velocity)
                first, last = min(first, last), max(first, last)
                lower, upper = max(lower, first), min(upper, last)
        if upper <= lower:
            continue
        step = np.float32(np.float32(upper - lower) / np.float32(samples))
        physical = np.float32(np.float32(length) * step)
        for index in range(samples):
            position = _fma32(np.float32(index) + np.float32(0.5), step, lower)
            point = [_fma32(position, direction[a], origin[a]) for a in range(3)]
            if any(point[a] < -0.5 or point[a] > shape[a] - 0.5 for a in range(3)):
                continue
            values = [np.float32(min(max(point[a], 0), shape[a] - 1)) for a in range(3)]
            lower_cell = [min(math.floor(values[a]), max(shape[a] - 2, 0)) for a in range(3)]
            fractions = [np.float32(values[a] - np.float32(lower_cell[a])) for a in range(3)]
            for corner in range(8):
                indices = [min(lower_cell[a] + ((corner >> a) & 1), shape[a] - 1) for a in range(3)]
                factors = [
                    fractions[a] if (corner >> a) & 1 else np.float32(1) - fractions[a]
                    for a in range(3)
                ]
                weight = np.float32(np.float32(factors[0] * factors[1]) * factors[2])
                voxel = (indices[2] * shape[1] + indices[1]) * shape[0] + indices[0]
                matrix[pixel, voxel] += float(np.float32(physical * weight))
    return matrix


def _prepare(grid, rays, pose, values, samples, device):
    workspace = prepare_projection(
        grid,
        rays,
        ProjectionSpec(samples, active_pose=True, active_volume=True, precision="float64"),
        device=device,
    )
    field = wp.array(values, dtype=wp.float32, device=device)
    packed = wp.array(pose.packed(), dtype=wp.float64, device=device)
    output = wp.empty(rays.rays, dtype=wp.float32, device=device)
    return workspace, field, packed, output


def _forward(workspace, field, packed, output):
    wp.launch(
        projection_fp32.get_forward(),
        dim=workspace.geometry.pixels,
        inputs=[field, packed, workspace._configuration],
        outputs=[output, workspace._status],
        device=workspace.device,
        stream=workspace.stream,
        block_dim=128,
    )
    workspace.check_status()
    return output.numpy().astype(np.float64)


def _reverse(
    workspace,
    field,
    packed,
    seeds,
    *,
    double_volume=False,
    matrix=False,
    per_ray=False,
    endpoints=False,
):
    dtype = wp.float64 if double_volume else wp.float32
    volume = wp.zeros(workspace.grid.voxels, dtype=dtype, device=workspace.device)
    seed = wp.array(seeds, dtype=wp.float32, device=workspace.device)
    components = 12 if matrix else 6
    count = (
        (workspace.geometry.pixels + 127) // 128
        if workspace.device.is_cuda
        else workspace.geometry.pixels
    )
    partials = (
        wp.empty(workspace.geometry.pixels * 6, dtype=wp.float64, device=workspace.device)
        if per_ray
        else workspace._reduction.partials[0]
    )
    wp.launch_tiled(
        projection_fp32.get_vjp(
            not per_ray,
            matrix,
            True,
            double_volume_gradient=double_volume,
            per_ray=per_ray,
            ray_coordinates=endpoints,
        ),
        dim=count,
        inputs=[field, packed, workspace._configuration, seed],
        outputs=[volume, partials, workspace._status],
        device=workspace.device,
        stream=workspace.stream,
        block_dim=128,
    )
    pose = wp.empty(components, dtype=wp.float64, device=workspace.device)
    if not per_ray:
        workspace._reduction.finish(
            count, pose, workspace._status, components=components, accumulate=False
        )
    workspace.check_status()
    return volume.numpy().astype(np.float64), partials.numpy().reshape(
        -1, 6
    ) if per_ray else pose.numpy()


@pytest.mark.parametrize("samples", [17, 97, 1024])
@pytest.mark.parametrize("double_volume", [False, True])
def test_fp32_forward_and_volume_vjp_match_independent_weights(samples, double_volume, warp_device):
    grid, rays, pose, values = _scene()
    matrix = _matrix(grid, rays, pose, samples)
    workspace, field, packed, output = _prepare(grid, rays, pose, values, samples, warp_device)
    seeds = np.asarray([0.7, -0.3, 0.2, 0.6, 1.0], dtype=np.float32)
    actual = _forward(workspace, field, packed, output)
    volume, _ = _reverse(workspace, field, packed, seeds, double_volume=double_volume)
    np.testing.assert_allclose(actual, matrix @ values, rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(volume, matrix.T @ seeds, rtol=2e-5, atol=3e-7)
    left, right = float(actual @ seeds), float(values.astype(np.float64) @ volume)
    assert abs(left - right) / max(abs(left), abs(right)) < 1e-5


def test_fp32_pose_matches_strict_and_directional_finite_differences(warp_device):
    grid, rays, pose, values = _scene()
    workspace, field, packed, output = _prepare(grid, rays, pose, values, 97, warp_device)
    seeds = np.asarray([0.7, -0.3, 0.2, 0.6, 0.0], dtype=np.float32)
    _, gradient = _reverse(workspace, field, packed, seeds)
    strict_gradient = wp.empty(6, dtype=wp.float64, device=warp_device)
    projection_vjp(
        field,
        packed,
        workspace=workspace,
        cotangent=wp.array(seeds.astype(np.float64), dtype=wp.float64, device=warp_device),
        out_pose_tangent=strict_gradient,
        stream=workspace.stream,
    )
    np.testing.assert_allclose(gradient, strict_gradient.numpy(), rtol=2e-3, atol=1e-5)
    for direction in (
        np.asarray([0.4, -0.2, 0.3, 0.01, -0.02, 0.015]),
        np.asarray([0.02, 0.03, -0.04, 0.15, -0.11, 0.07]),
    ):
        expected = float(gradient @ direction)
        errors = []
        for epsilon in (3e-2, 1e-2, 3e-3, 1e-3):
            losses = []
            for sign in (-1, 1):
                shifted = compose_pose(pose, tuple(sign * epsilon * direction))
                packed.assign(np.asarray(shifted.packed(), dtype=np.float64))
                losses.append(float(_forward(workspace, field, packed, output) @ seeds))
            difference = (losses[1] - losses[0]) / (2 * epsilon)
            errors.append(abs(difference - expected) / max(abs(expected), 1e-12))
        assert min(errors) < 1e-2, (expected, errors)


def test_fp32_clipped_endpoint_pose_derivative_and_singleton_axes(warp_device):
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((-2.0, 0.0, 0.0),), ((0.2, 0.0, 0.0),))
    workspace, field, packed, output = _prepare(
        grid, rays, RigidTransform(), [0.025], 1024, warp_device
    )
    np.testing.assert_allclose(_forward(workspace, field, packed, output), [0.025 * 0.7], rtol=2e-5)
    volume, pose = _reverse(workspace, field, packed, [1.0])
    np.testing.assert_allclose(volume, [0.7], rtol=2e-5)
    np.testing.assert_allclose(pose, [-0.025, 0, 0, 0, 0, 0], rtol=2e-5, atol=1e-8)


def test_fp32_local_coordinates_preserve_large_physical_origin(warp_device):
    grid, rays, _, values = _scene()
    pose = RigidTransform()
    workspace, field, packed, output = _prepare(grid, rays, pose, values, 97, warp_device)
    reference = _forward(workspace, field, packed, output)
    displacement = (1e8, -2e8, 3e8)
    shifted_grid = GridSpec(
        grid.shape,
        grid.spacing_mm,
        tuple(x + d for x, d in zip(grid.origin_mm, displacement, strict=True)),
        grid.orientation,
    )
    shifted_rays = RayBundleGeometry(
        tuple(
            tuple(x + d for x, d in zip(point, displacement, strict=True))
            for point in rays.sources_mm
        ),
        tuple(
            tuple(x + d for x, d in zip(point, displacement, strict=True))
            for point in rays.endpoints_mm
        ),
    )
    shifted_workspace, field, packed, output = _prepare(
        shifted_grid, shifted_rays, pose, values, 97, warp_device
    )
    np.testing.assert_allclose(
        _forward(shifted_workspace, field, packed, output), reference, rtol=2e-6, atol=2e-7
    )


def test_fp32_pose_reductions_and_endpoint_norm_chain_agree(warp_device):
    grid, rays, pose, values = _scene()
    workspace, field, packed, _ = _prepare(grid, rays, pose, values, 97, warp_device)
    seeds = np.asarray([0.7, -0.3, 0.2, 0.6, 0.0], dtype=np.float32)
    _, tangent = _reverse(workspace, field, packed, seeds)
    _, per_ray = _reverse(workspace, field, packed, seeds, per_ray=True)
    np.testing.assert_allclose(per_ray.sum(axis=0), tangent, rtol=2e-12, atol=2e-12)
    _, packed_gradient = _reverse(workspace, field, packed, seeds, matrix=True)
    rotation = np.asarray(pose.rotation).reshape(3, 3)
    translation = rotation.T @ packed_gradient[9:]
    np.testing.assert_allclose(translation, tangent[:3], rtol=2e-12, atol=2e-12)
    _, endpoint_gradient = _reverse(workspace, field, packed, seeds, per_ray=True, endpoints=True)
    # Moving both ends equally does not change ray norm. Compare the summed
    # endpoint cotangent to the opposite rigid-object translation cotangent.
    source_plus_end = endpoint_gradient[:, :3] + endpoint_gradient[:, 3:]
    np.testing.assert_allclose(
        -(rotation.T @ source_plus_end.sum(axis=0)), tangent[:3], rtol=2e-12, atol=2e-12
    )
