"""Cell aggregation preserves rounded increments and finite accumulation."""

from contextlib import nullcontext
from functools import cache

import numpy as np
import pytest
import warp as wp

from dpt._runtime import cpu_test_context
from dpt.contracts import NumericalError
from dpt.geometry import RayBundleGeometry, RigidTransform
from dpt.kernels import projection_fp32 as fp32
from dpt.kernels.projection import Configuration, ray_for_pixel
from dpt.projection import ProjectionSpec, prepare_projection
from dpt.volumes import GridSpec

wp.set_module_options(fp32.OPTIONS)


@cache
def reference_scatter(double_destination):
    dtype = wp.float64 if double_destination else wp.float32
    scatter = fp32.get_scatter_field(double_destination)

    @wp.kernel(module="unique", module_options=fp32.OPTIONS)
    def reference(
        pose: wp.array(dtype=wp.float64),
        config: Configuration,
        seeds: wp.array(dtype=wp.float32),
        destination: wp.array(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        precise = ray_for_pixel(config, pose, pixel)
        origin, direction = wp.vec3(precise.origin), wp.vec3(precise.direction)
        interval = fp32.finite_interval(origin, direction, config.shape)
        span = interval.upper - interval.lower
        seed = seeds[pixel]
        if span > wp.float32(0.0) and seed != wp.float32(0.0):
            step = span / wp.float32(config.samples)
            physical_step = wp.float32(precise.length) * step
            factor = seed * physical_step
            for index in range(config.samples):
                position = fp32.sample_position(interval.lower, step, index)
                point = fp32.sample_point(origin, direction, position)
                scatter(destination, point, config.shape, factor, status)

    return reference


@pytest.fixture(autouse=True)
def context(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def evaluate(
    grid, rays, seeds, samples, device, *, joint=False, initial=None, double_reference=True
):
    workspace = prepare_projection(
        grid,
        rays,
        ProjectionSpec(samples, active_pose=joint, active_volume=True),
        device=device,
    )
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=device)
    field = wp.array(
        np.linspace(0.01, 0.1, grid.voxels, dtype=np.float32),
        dtype=wp.float32,
        device=device,
    )
    seed = wp.array(np.asarray(seeds, dtype=np.float32), dtype=wp.float32, device=device)
    initial = np.zeros(grid.voxels, dtype=np.float32) if initial is None else initial
    reference = wp.array(
        initial, dtype=wp.float64 if double_reference else wp.float32, device=device
    )
    actual = wp.array(initial, dtype=wp.float32, device=device)
    wp.launch(
        reference_scatter(double_reference),
        dim=rays.pixels,
        inputs=[pose, workspace._configuration, seed],
        outputs=[reference, workspace._status],
        device=device,
        stream=workspace.stream,
    )
    with workspace.context.scope():
        workspace._status.zero_()
    count = (rays.pixels + 127) // 128 if workspace.device.is_cuda else rays.pixels
    partials = wp.empty(count * 6, dtype=wp.float64, device=device)
    wp.launch_tiled(
        fp32.get_vjp(True, False, joint),
        dim=count,
        inputs=[field, pose, workspace._configuration, seed],
        outputs=[actual, partials, workspace._status],
        device=device,
        stream=workspace.stream,
        block_dim=128,
    )
    workspace.check_status()
    return reference.numpy(), actual.numpy()


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 2, 5), (3, 1, 6), (4, 5, 1), (3, 4, 5)])
@pytest.mark.parametrize("samples", [17, 1024])
@pytest.mark.parametrize("joint", [False, True])
def test_cell_runs_match_per_sample_scatter(shape, samples, joint, warp_device):
    nz, ny, nx = shape
    centre = ((ny - 1) / 2, (nz - 1) / 2)
    sources = ((-3.0, *centre), (nx + 2.0, *centre), (-3.0, ny + 4.0, nz + 4.0))
    targets = ((nx + 2.0, *centre), (-3.0, *centre), (nx + 2.0, ny + 4.0, nz + 4.0))
    rays = RayBundleGeometry(
        tuple(sources[i % 3] for i in range(65)),
        tuple(targets[i % 3] for i in range(65)),
    )
    seeds = np.asarray([0.7, -0.3, 0.0, -0.0, 0.2] * 13, dtype=np.float32)
    reference, actual = evaluate(
        GridSpec(shape, (1.0, 1.0, 1.0)),
        rays,
        seeds,
        samples,
        warp_device,
        joint=joint,
    )
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=3e-7)


def test_local_overflow_flushes_into_opposing_existing_destination(warp_device):
    grid = GridSpec((1, 1, 1), (2.0, 1.0, 1.0))
    rays = RayBundleGeometry(((-2.0, 0.0, 0.0),), ((2.0, 0.0, 0.0),))
    reference, actual = evaluate(
        grid,
        rays,
        [2e38],
        1024,
        warp_device,
        initial=np.asarray([-3e38], dtype=np.float32),
    )
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=0)
    np.testing.assert_allclose(actual, [1e38], rtol=2e-5, atol=0)


def test_subnormal_increments_retain_atomic_device_semantics(warp_device):
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((-2.0, 0.0, 0.0),), ((2.0, 0.0, 0.0),))
    reference, actual = evaluate(grid, rays, [1e-35], 1024, warp_device, double_reference=False)
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=0)
    if warp_device != "cpu":
        np.testing.assert_array_equal(actual, [0.0])


@pytest.mark.parametrize("seed", [float("nan"), float("inf"), 3e38])
def test_invalid_increment_keeps_numerical_status(seed, warp_device):
    grid = GridSpec((1, 1, 1), (2.0, 1.0, 1.0))
    rays = RayBundleGeometry(((-2.0, 0.0, 0.0),), ((2.0, 0.0, 0.0),))
    with pytest.raises(NumericalError):
        evaluate(grid, rays, [seed], 1, warp_device)
