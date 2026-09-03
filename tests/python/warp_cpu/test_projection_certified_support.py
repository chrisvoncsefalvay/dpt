"""Endpoint certification covers every realised midpoint, including rounding tails."""

import numpy as np
import pytest
import warp as wp

from dpt.kernels import projection as strict


@wp.kernel(enable_backward=False)
def check_sequence(
    origins: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    shape: wp.vec3i,
    samples: int,
    flags: wp.array(dtype=wp.vec3i),
):
    index = wp.tid()
    ray = strict.Ray()
    ray.origin = origins[index]
    ray.direction = directions[index]
    interval = strict.finite_interval(ray.origin, ray.direction, shape)
    span = interval.upper - interval.lower
    if span > wp.float64(0.0):
        step = span / wp.float64(samples)
        certified = strict.midpoints_inside(ray, interval, step, samples, shape)
        outside = wp.int32(0)
        for sample in range(samples):
            position = interval.lower + (wp.float64(sample) + wp.float64(0.5)) * step
            point = ray.origin + position * ray.direction
            if not strict.inside_support(point, shape):
                outside += 1
        flags[index] = wp.vec3i(1, int(certified), outside)


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 7, 9), (256, 256, 256)])
@pytest.mark.parametrize("samples", [1, 3, 17, 1024])
def test_certification_never_admits_an_outside_midpoint(shape, samples, warp_device):
    rng = np.random.default_rng(43819)
    centre = (np.asarray(shape, dtype=np.float64) - 1) / 2
    origins = []
    directions = []
    for scale in (1.0, 1e3, 1e8, 1e15, 1e100, 1e300):
        displacement = rng.normal(size=(32, 3)) * scale
        origins.extend(centre + displacement)
        directions.extend(-2 * displacement)
    for coordinate in (-0.5, np.nextafter(-0.5, -np.inf), np.nextafter(-0.5, np.inf)):
        origins.extend(((coordinate, -1, 0), (coordinate, 0, -1)))
        directions.extend(((0, shape[1] + 1, 0), (0, 0, shape[2] + 1)))
    flags = wp.zeros(len(origins), dtype=wp.vec3i, device=warp_device)
    wp.launch(
        check_sequence,
        dim=len(origins),
        inputs=[
            wp.array(origins, dtype=wp.vec3d, device=warp_device),
            wp.array(directions, dtype=wp.vec3d, device=warp_device),
            wp.vec3i(*shape),
            samples,
        ],
        outputs=[flags],
        device=warp_device,
    )
    result = flags.numpy()
    admitted = result[:, 1] != 0
    assert admitted.any()
    assert np.all(result[admitted, 2] == 0)
    assert np.any(result[:, 0] == 0)
