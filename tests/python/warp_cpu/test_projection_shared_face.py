"""Shared-face carry preserves rounded terms and all full-flush fallbacks."""

from contextlib import nullcontext

import numpy as np
import pytest
import warp as wp
from test_projection_cell_aggregation import evaluate

from dpt._runtime import cpu_test_context
from dpt.geometry import RayBundleGeometry
from dpt.kernels import projection_fp32 as fp32
from dpt.volumes import GridSpec


@pytest.fixture(autouse=True)
def _context(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


@wp.kernel(enable_backward=False)
def _transition_probe(
    lower: wp.vec3i,
    following: wp.vec3i,
    shape: wp.vec3i,
    values: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
    carried: wp.array(dtype=wp.float32),
):
    sums = fp32.Vec8f()
    for corner in range(8):
        sums[corner] = values[corner]
    result = fp32.transition_cell(gradient, lower, following, shape, sums)
    for corner in range(8):
        carried[corner] = result[corner]


_TRANSITIONS = [
    ((7, 7, 7), (2, 2, 2), delta)
    for delta in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
] + [
    ((7, 7, 7), (2, 2, 2), (1, 1, 0)),
    ((7, 7, 7), (2, 2, 2), (-1, 0, -1)),
    ((7, 7, 7), (2, 2, 2), (2, 0, 0)),
    ((7, 7, 7), (2, 2, 2), (0, -2, 0)),
    ((1, 7, 7), (0, 2, 2), (0, 1, 0)),
    ((7, 1, 7), (2, 0, 2), (0, 0, -1)),
    ((7, 7, 1), (2, 2, 0), (1, 0, 0)),
]


@pytest.mark.parametrize("shape,lower,delta", _TRANSITIONS)
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_transition_flushes_departing_corners_and_maps_shared_face(
    shape, lower, delta, sign, warp_device
):
    values = sign * np.exp2(np.arange(8, dtype=np.float32))
    values[0] = np.copysign(np.float32(0), np.float32(sign))
    following = tuple(a + b for a, b in zip(lower, delta, strict=True))
    gradient = wp.zeros(int(np.prod(shape)), dtype=wp.float32, device=warp_device)
    carried = wp.empty(8, dtype=wp.float32, device=warp_device)
    wp.launch(
        _transition_probe,
        dim=1,
        inputs=[
            wp.vec3i(*lower),
            wp.vec3i(*following),
            wp.vec3i(*shape),
            wp.array(values, dtype=wp.float32, device=warp_device),
        ],
        outputs=[gradient, carried],
        device=warp_device,
    )
    expected_gradient = np.zeros(int(np.prod(shape)), dtype=np.float32)
    expected_carried = np.zeros(8, dtype=np.float32)
    changing = [axis for axis, value in enumerate(delta) if value]
    eligible = min(shape) > 1 and len(changing) == 1 and abs(delta[changing[0]]) == 1
    axis = changing[0]
    for corner in range(8):
        if eligible and ((corner >> axis) & 1) == int(delta[axis] > 0):
            expected_carried[corner ^ (1 << axis)] = values[corner]
        else:
            x, y, z = [min(lower[a] + ((corner >> a) & 1), shape[a] - 1) for a in range(3)]
            expected_gradient[(z * shape[1] + y) * shape[0] + x] += values[corner]
    np.testing.assert_array_equal(gradient.numpy(), expected_gradient)
    np.testing.assert_array_equal(carried.numpy().view(np.uint32), expected_carried.view(np.uint32))


@pytest.mark.parametrize(
    "direction",
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 1, 0), (-1, -1, -1)],
)
@pytest.mark.parametrize("samples", [2, 7, 1024])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("seed_scale", [1.0, 1000.0])
def test_shared_faces_match_independent_per_sample_scatter(
    direction, samples, joint, seed_scale, warp_device
):
    centre = np.asarray([2.0, 2.0, 2.0])
    displacement = np.asarray(direction) * 10.0
    source, target = centre - displacement, centre + displacement
    rays = RayBundleGeometry(
        (tuple(source), tuple(target), tuple(source + 100.0)),
        (tuple(target), tuple(source), tuple(target + 100.0)),
    )
    reference, actual = evaluate(
        GridSpec((5, 5, 5), (1.0, 1.0, 1.0)),
        rays,
        seed_scale * np.asarray([0.7, -0.3, 0.0], dtype=np.float32),
        samples,
        warp_device,
        joint=joint,
    )
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=3e-7)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_shared_sum_overflow_flushes_into_opposing_destination(sign, reverse, warp_device):
    source, target = (-3.0, 0.0, 0.0), (7.0, 0.0, 0.0)
    if reverse:
        source, target = target, source
    grid = GridSpec((2, 2, 3), (2.0, 1.0, 1.0))
    reference, actual = evaluate(
        grid,
        RayBundleGeometry((source,), (target,)),
        [sign * 2e38],
        1024,
        warp_device,
        initial=np.full(grid.voxels, -sign * 3e38, dtype=np.float32),
    )
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=0)
    np.testing.assert_allclose(actual[1], sign * 1e38, rtol=2e-5, atol=0)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("joint", [False, True])
def test_shared_faces_do_not_restore_flushed_subnormal_products(sign, joint, warp_device):
    reference, actual = evaluate(
        GridSpec((2, 2, 3), (1.0, 1.0, 1.0)),
        RayBundleGeometry(((-2.0, 0.5, 0.5),), ((5.0, 0.5, 0.5),)),
        [sign * 1e-35],
        1024,
        warp_device,
        joint=joint,
        double_reference=False,
    )
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=0)
    if warp_device != "cpu":
        np.testing.assert_array_equal(actual, np.zeros_like(actual))
