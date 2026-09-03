"""Material FP32 paths and both VJPs against independent numerical references."""

from contextlib import nullcontext

import numpy as np
import pytest
import warp as wp

from dpt._runtime import cpu_test_context
from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose
from dpt.kernels import material_projection_fp32
from dpt.material_projection import material_projection_vjp, prepare_material_projection
from dpt.projection import ProjectionSpec
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec


@pytest.fixture(autouse=True)
def _context(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def _prepare(device, materials=3, double_volume=False, double_output=False, samples=47):
    grid = GridSpec((3, 4, 5), (1.2, 0.8, 1.1), (-2.0, -1.0, -1.0))
    panel = DetectorGeometry(
        (-12.0, 0.31, 0.43),
        (12.0, -0.6, -0.6),
        (0, 1, 0),
        (0, 0, 1),
        (0.071, 0.11),
        (3, 5),
    )
    transform = compose_pose(RigidTransform(), (0.13, -0.08, 0.07, 0.02, -0.03, 0.01))
    values = np.asarray(
        [
            [
                (m + 1) * (0.06 + 0.013 * x + 0.019 * y + 0.011 * z) + 0.003 * m * x * y
                for z in range(3)
                for y in range(4)
                for x in range(5)
            ]
            for m in range(materials)
        ],
        dtype=np.float32,
    )
    seeds = np.asarray(
        [
            [((-1) ** m) * (0.2 + 0.03 * p + 0.04 * m) for p in range(panel.pixels)]
            for m in range(materials)
        ],
        dtype=np.float64 if double_output else np.float32,
    )
    path_dtype = wp.float64 if double_output else wp.float32
    gradient_dtype = wp.float64 if double_volume else wp.float32
    fields = wp.array(values, dtype=wp.float32, device=device)
    paths = wp.empty((materials, panel.pixels), dtype=path_dtype, device=device)
    seed = wp.array(seeds, dtype=path_dtype, device=device)
    gradients = wp.empty((materials, grid.voxels), dtype=gradient_dtype, device=device)
    workspace = prepare_material_projection(
        grid,
        panel,
        ProjectionSpec(
            samples,
            active_pose=True,
            active_volume=True,
            precision="float64" if double_output else "float32",
        ),
        materials=materials,
        fields=fields,
        out_paths=paths,
        cotangent_paths=seed,
        out_fields=gradients,
        device=device,
        field_domain="nonnegative",
        channel_execution="fused",
        volume_gradient_precision="float64" if double_volume else "float32",
    )
    packed = wp.array(transform.packed(), dtype=wp.float64, device=device)
    return workspace, packed, transform, values, seeds


def _forward(workspace, packed):
    projection = workspace.projection
    wp.launch(
        material_projection_fp32.get_material_forward(
            workspace.materials, projection.spec.precision == "float64", False
        ),
        dim=projection.geometry.pixels,
        inputs=[workspace.fields, packed, projection._configuration, projection.grid.voxels],
        outputs=[workspace.paths, projection._status],
        device=projection.device,
        stream=projection.stream,
        block_dim=128,
    )
    workspace.check_status()
    return workspace.paths.numpy().reshape(workspace.materials, -1).astype(np.float64)


def _reverse(workspace, packed, *, active_fields=True, active_pose=True):
    projection = workspace.projection
    with projection.context.scope():
        workspace.grad_fields.zero_()
    count = (
        (projection.geometry.pixels + 127) // 128
        if projection.device.is_cuda
        else projection.geometry.pixels
    )
    wp.launch_tiled(
        material_projection_fp32.get_material_vjp(
            workspace.materials,
            projection.spec.precision == "float64",
            False,
            active_fields,
            active_pose,
            workspace.volume_gradient_precision == "float64",
        ),
        dim=count,
        inputs=[
            workspace.fields,
            packed,
            projection._configuration,
            projection.grid.voxels,
            workspace.cotangent_paths,
        ],
        outputs=[workspace.grad_fields, projection._reduction.partials[0], projection._status],
        device=projection.device,
        stream=projection.stream,
        block_dim=128,
    )
    pose = wp.zeros(6, dtype=wp.float64, device=projection.device)
    if active_pose:
        projection._reduction.finish(
            count, pose, projection._status, components=6, accumulate=False
        )
    workspace.check_status()
    return workspace.grad_fields.numpy().reshape(workspace.materials, -1).astype(
        np.float64
    ), pose.numpy()


def _matrix(workspace, transform):
    projection = workspace.projection
    grid, panel = projection.grid, projection.geometry
    result = np.empty((panel.pixels, grid.voxels), dtype=np.float64)
    for voxel in range(grid.voxels):
        basis = [float(i == voxel) for i in range(grid.voxels)]
        for pixel in range(panel.pixels):
            row, column = divmod(pixel, panel.shape[1])
            result[pixel, voxel] = integrate_sampled_field(
                basis,
                grid,
                panel,
                transform,
                row,
                column,
                midpoint_samples=projection.spec.samples_per_ray,
            )
    return result


@pytest.mark.parametrize("materials", [1, 3])
@pytest.mark.parametrize("double_volume", [False, True])
def test_fp32_material_forward_and_adjoint_match_independent_matrix(
    materials, double_volume, warp_device
):
    workspace, packed, transform, values, seeds = _prepare(warp_device, materials, double_volume)
    matrix = _matrix(workspace, transform)
    paths = _forward(workspace, packed)
    gradient, _ = _reverse(workspace, packed)
    np.testing.assert_allclose(paths, values.astype(np.float64) @ matrix.T, rtol=5e-6, atol=3e-7)
    np.testing.assert_allclose(gradient, seeds.astype(np.float64) @ matrix, rtol=3e-5, atol=3e-6)
    left, right = float(np.sum(paths * seeds)), float(np.sum(gradient * values))
    assert abs(left - right) / max(abs(left), abs(right)) < 1e-5


def test_fp32_material_pose_and_represented_volume_directional_derivatives(warp_device):
    workspace, packed, transform, values, seeds = _prepare(warp_device)
    _forward(workspace, packed)
    gradient, pose_gradient = _reverse(workspace, packed)
    strict_pose = wp.empty(6, dtype=wp.float64, device=warp_device)
    material_projection_vjp(
        packed,
        workspace=workspace,
        out_pose_tangent=strict_pose,
        stream=workspace.projection.stream,
    )
    np.testing.assert_allclose(pose_gradient, strict_pose.numpy(), rtol=2e-3, atol=2e-5)
    direction = np.random.default_rng(41072).normal(size=values.shape).astype(np.float32) * 0.03
    epsilon = 0.03
    plus, minus = (
        (values + epsilon * direction).astype(np.float32),
        (values - epsilon * direction).astype(np.float32),
    )
    losses = []
    for moved in (plus, minus):
        workspace.fields.assign(moved.reshape(-1))
        losses.append(float(np.sum(_forward(workspace, packed) * seeds)))
    represented = (plus.astype(np.float64) - minus.astype(np.float64)) / (2 * epsilon)
    expected = float(np.sum(gradient * represented))
    difference = (losses[0] - losses[1]) / (2 * epsilon)
    assert abs(difference - expected) / max(abs(expected), 1e-12) < 1e-3
    workspace.fields.assign(values.reshape(-1))
    pose_direction = np.asarray([0.4, -0.2, 0.3, 0.01, -0.02, 0.015])
    expected = float(pose_gradient @ pose_direction)
    errors = []
    for epsilon in (3e-2, 1e-2, 3e-3, 1e-3):
        losses = []
        for sign in (-1, 1):
            moved = compose_pose(transform, tuple(sign * epsilon * pose_direction))
            packed.assign(np.asarray(moved.packed(), dtype=np.float64))
            losses.append(float(np.sum(_forward(workspace, packed) * seeds)))
        errors.append(
            abs((losses[1] - losses[0]) / (2 * epsilon) - expected) / max(abs(expected), 1e-12)
        )
    assert min(errors) < 1e-2, (expected, errors)


def test_fp32_material_specialisations_zero_seeds_and_wider_storage(warp_device):
    workspace, packed, _, values, seeds = _prepare(
        warp_device, double_volume=True, double_output=True, samples=1024
    )
    paths = _forward(workspace, packed)
    both_volume, both_pose = _reverse(workspace, packed)
    only_volume, no_pose = _reverse(workspace, packed, active_pose=False)
    no_volume, only_pose = _reverse(workspace, packed, active_fields=False)
    np.testing.assert_allclose(only_volume, both_volume, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(only_pose, both_pose, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(no_pose, np.zeros(6))
    np.testing.assert_array_equal(no_volume, np.zeros_like(no_volume))
    left, right = float(np.sum(paths * seeds)), float(np.sum(both_volume * values))
    assert abs(left - right) / max(abs(left), abs(right)) < 1e-5
    with workspace.projection.context.scope():
        workspace.cotangent_paths.zero_()
    volume, pose = _reverse(workspace, packed)
    np.testing.assert_array_equal(volume, np.zeros_like(volume))
    np.testing.assert_array_equal(pose, np.zeros(6))
