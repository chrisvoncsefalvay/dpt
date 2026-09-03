"""Exact scalar values and independent FP32 slope and route checks."""

from contextlib import nullcontext

import numpy as np
import pytest
import warp as wp
from test_projection_fp32 import _prepare, _scene

from dpt._runtime import cpu_test_context
from dpt.geometry import RayBundleGeometry, RigidTransform
from dpt.kernels import projection_fp32 as fp32
from dpt.volumes import GridSpec


@pytest.fixture(autouse=True)
def _context(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


@wp.kernel
def _samples(
    field: wp.array(dtype=wp.float32),
    points: wp.array(dtype=wp.vec3),
    shape: wp.vec3i,
    old_values: wp.array(dtype=wp.float32),
    new_values: wp.array(dtype=wp.float32),
    slopes: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    original = fp32.sample_field(field, points[index], shape)
    actual = fp32.sample_pose_field(field, points[index], shape)
    old_values[index] = original.value
    new_values[index] = actual.value
    slopes[index] = actual.gradient


def _sample(values, points, shape, device):
    old = wp.empty(len(points), dtype=wp.float32, device=device)
    new = wp.empty_like(old)
    slopes = wp.empty(len(points), dtype=wp.vec3, device=device)
    wp.launch(
        _samples,
        dim=len(points),
        inputs=[
            wp.array(values, dtype=wp.float32, device=device),
            wp.array(points, dtype=wp.vec3, device=device),
            wp.vec3i(*shape),
        ],
        outputs=[old, new, slopes],
        device=device,
    )
    np.testing.assert_array_equal(old.numpy().view(np.uint32), new.numpy().view(np.uint32))
    return new.numpy(), slopes.numpy()


@pytest.mark.parametrize("value", [0.0, 2.0**-149, 2.0**-126, 1.0, 1e20, np.finfo(np.float32).max])
def test_constant_field_has_exact_zero_slope_and_original_value(value, warp_device):
    points = [
        (0.125, 0.375, 0.625),
        (0.1, 0.3, 0.7),
        (0, 0, 0),
        (1, 1, 1),
        (-0.5, 0.25, 0.75),
        (1.5, 0.25, 0.75),
    ]
    _, slopes = _sample(np.full(8, value, dtype=np.float32), points, (2, 2, 2), warp_device)
    np.testing.assert_array_equal(slopes, np.zeros_like(slopes))


@pytest.mark.parametrize("shape", [(2, 2, 2), (1, 2, 3), (3, 1, 2), (2, 3, 1), (1, 1, 1)])
@pytest.mark.parametrize("scale", [2.0**-140, 0.125, 2.0**77])
def test_affine_slopes_follow_clamp_masks_including_singletons(shape, scale, warp_device):
    base = scale * 8
    coefficients = np.asarray([scale, scale * 2, scale * 4], dtype=np.float64)
    values = np.asarray(
        [
            base + coefficients @ [x, y, z]
            for z in range(shape[2])
            for y in range(shape[1])
            for x in range(shape[0])
        ],
        dtype=np.float32,
    )
    points = [
        (0.25, 0.5 if shape[1] > 1 else 0.25, 0.75 if shape[2] > 1 else 0.25),
        (0, 0, 0),
        tuple(float(size - 1) for size in shape),
        (-0.25, 0.25, 0.25),
    ]
    _, actual = _sample(values, points, shape, warp_device)
    expected = np.asarray(
        [
            [coefficients[axis] if 0 <= point[axis] < shape[axis] - 1 else 0 for axis in range(3)]
            for point in points
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "point",
    [
        (-0.5001, 0, 0),
        (1.5001, 0, 0),
        (float("nan"), 0, 0),
        (0, float("inf"), 0),
        (0, 0, -float("inf")),
    ],
)
def test_unsupported_points_return_zero_without_indexing(point, warp_device):
    values, slopes = _sample(np.arange(8, dtype=np.float32), [point], (2, 2, 2), warp_device)
    np.testing.assert_array_equal(values, [0])
    np.testing.assert_array_equal(slopes, [[0, 0, 0]])


@pytest.mark.parametrize("corner", range(8))
def test_maximum_finite_corner_has_finite_analytic_slopes(corner, warp_device):
    values = np.zeros(8, dtype=np.float32)
    values[corner] = np.finfo(np.float32).max
    point = (0.25, 0.5, 0.75)
    _, actual = _sample(values, [point], (2, 2, 2), warp_device)
    factors = [point[axis] if (corner >> axis) & 1 else 1 - point[axis] for axis in range(3)]
    expected = []
    for axis in range(3):
        value = float(values[corner]) * (1 if (corner >> axis) & 1 else -1)
        for other in range(3):
            if other != axis:
                value *= factors[other]
        expected.append(value)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[0], expected, rtol=2e-7, atol=0)


def test_small_slope_survives_a_large_orthogonal_background(warp_device):
    small, large = np.float32(2.0**-120), np.float32(2.0**100)
    values = np.asarray([0, small, 0, small, large, large, large, large], dtype=np.float32)
    _, slopes = _sample(values, [(0.25, 0.5, 0.5)], (2, 2, 2), warp_device)
    np.testing.assert_array_equal(slopes[0, :2], [small * np.float32(0.5), 0])
    np.testing.assert_allclose(slopes[0, 2], large, rtol=2e-7, atol=0)


def _reverse(workspace, field, packed, seeds, volume_active, pose_active, matrix, double):
    dtype = wp.float64 if double else wp.float32
    volume = wp.zeros(workspace.grid.voxels, dtype=dtype, device=workspace.device)
    seed = wp.array(seeds, dtype=wp.float32, device=workspace.device)
    components = 12 if matrix else 6
    count = (
        (workspace.geometry.pixels + 127) // 128
        if workspace.device.is_cuda
        else workspace.geometry.pixels
    )
    wp.launch_tiled(
        fp32.get_vjp(volume_active, matrix, pose_active, double_volume_gradient=double),
        dim=count,
        inputs=[field, packed, workspace._configuration, seed],
        outputs=[volume, workspace._reduction.partials[0], workspace._status],
        device=workspace.device,
        stream=workspace.stream,
        block_dim=128,
    )
    pose = wp.empty(components, dtype=wp.float64, device=workspace.device)
    if pose_active:
        workspace._reduction.finish(
            count, pose, workspace._status, components=components, accumulate=False
        )
    workspace.check_status()
    return volume.numpy(), pose.numpy() if pose_active else None


@pytest.mark.parametrize("scene", ["oblique", "singleton"])
@pytest.mark.parametrize("samples", [17, 1024])
@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize("seed_mode", ["signed", "zero"])
def test_joint_and_separate_routes_use_the_same_slopes(
    scene, samples, matrix, double, seed_mode, warp_device
):
    if scene == "oblique":
        grid, rays, pose, values = _scene()
    else:
        grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
        rays = RayBundleGeometry(
            ((-2.0, 0, 0), (2.0, 0.1, 0), (-2.0, 2.0, 0)),
            ((0.2, 0, 0), (-0.1, 0.1, 0), (2.0, 2.0, 0)),
        )
        pose, values = RigidTransform(), [0.025]
    workspace, field, packed, _ = _prepare(grid, rays, pose, values, samples, warp_device)
    seeds = np.linspace(-0.7, 0.9, rays.rays, dtype=np.float32)
    if seed_mode == "zero":
        seeds[:] = 0
    joint_volume, joint_pose = _reverse(workspace, field, packed, seeds, True, True, matrix, double)
    volume, _ = _reverse(workspace, field, packed, seeds, True, False, matrix, double)
    _, pose_gradient = _reverse(workspace, field, packed, seeds, False, True, matrix, double)
    np.testing.assert_allclose(joint_volume, volume, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(joint_pose, pose_gradient, rtol=2e-5, atol=1e-8)
