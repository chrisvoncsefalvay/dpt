"""Bounded strict increments retain their exact rounding and status."""

from functools import cache

import numpy as np
import pytest
import warp as wp

from dpt.kernels.projection import get_scatter_field


@cache
def scatter_kernel(certified):
    scatter = get_scatter_field(certified_support=certified, certified_range=certified)

    @wp.kernel(enable_backward=False)
    def kernel(
        points: wp.array(dtype=wp.vec3d),
        factors: wp.array(dtype=wp.float64),
        shape: wp.vec3i,
        result: wp.array(dtype=wp.float32),
        status: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        offset = i * shape[0] * shape[1] * shape[2]
        scatter(result, points[i], shape, factors[i], status, offset)

    return kernel


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 4), (9, 11, 7)])
def test_bounded_increment_rounding_is_unchanged(shape, warp_device):
    rng = np.random.default_rng(1738)
    maximum = float(np.finfo(np.float32).max)
    factors = np.array([0.0, 1e-300, 1e-45, 1e-30, 0.3, 1.0, maximum, -maximum], dtype=np.float64)
    factors = np.tile(factors, 20)
    points = rng.uniform(size=(len(factors), 3)) * np.asarray(shape) - 0.5
    points[0] = -0.5
    points[1] = np.asarray(shape) - 0.5
    points[2] = 0
    inputs = [
        wp.array(points, dtype=wp.vec3d, device=warp_device),
        wp.array(factors, dtype=wp.float64, device=warp_device),
        wp.vec3i(*shape),
    ]
    results = []
    for certified in (False, True):
        result = wp.zeros(len(points) * int(np.prod(shape)), device=warp_device)
        status = wp.zeros(1, dtype=wp.int32, device=warp_device)
        wp.launch(
            scatter_kernel(certified),
            dim=len(points),
            inputs=inputs,
            outputs=[result, status],
            device=warp_device,
        )
        assert status.numpy()[0] == 0
        results.append(result.numpy())
    np.testing.assert_array_equal(results[0], results[1])
