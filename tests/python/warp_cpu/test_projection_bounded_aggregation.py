"""Bounded products preserve C01 rounding, signed zeros and fallback behaviour."""

from contextlib import nullcontext

import numpy as np
import pytest
import warp as wp
from test_projection_cell_aggregation import evaluate

from dpt._runtime import cpu_test_context
from dpt.geometry import RayBundleGeometry
from dpt.kernels import projection_fp32 as fp32
from dpt.volumes import GridSpec

wp.set_module_options(fp32.OPTIONS)


@wp.kernel(module="unique", module_options=fp32.OPTIONS)
def compare_products(
    factors: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    original: wp.array(dtype=wp.float32),
    selected: wp.array(dtype=wp.float32),
    bounded: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    factor, weight = factors[index], weights[index]
    original[index] = fp32.scatter_increment(factor, weight)
    if wp.isfinite(factor) and wp.abs(factor) <= wp.float32(1.0):
        selected[index] = fp32.bounded_scatter_increment(factor, weight)
        bounded[index] = 1
    else:
        selected[index] = fp32.scatter_increment(factor, weight)
        bounded[index] = 0


@pytest.fixture(autouse=True)
def context(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def test_selected_product_matches_original_bits_and_device_underflow(warp_device):
    tiny = np.finfo(np.float32).tiny
    subnormal = np.nextafter(np.float32(0.0), np.float32(1.0))
    below_one = np.nextafter(np.float32(1.0), np.float32(0.0))
    above_one = np.nextafter(np.float32(1.0), np.float32(2.0))
    positive = np.asarray(
        [0.0, subnormal, np.nextafter(tiny, np.float32(0.0)), tiny, 0.5, below_one, 1.0],
        dtype=np.float32,
    )
    factors = np.concatenate(
        [
            positive,
            -positive,
            np.asarray(
                [
                    above_one,
                    -above_one,
                    16.0,
                    -16.0,
                    1e20,
                    -1e20,
                    3e38,
                    -3e38,
                    np.inf,
                    -np.inf,
                    np.nan,
                ],
                dtype=np.float32,
            ),
        ]
    )
    weights = np.asarray(
        [
            0.0,
            -0.0,
            subnormal,
            np.nextafter(tiny, np.float32(0.0)),
            tiny,
            np.nextafter(tiny, np.float32(1.0)),
            0.125,
            0.5,
            below_one,
            1.0,
        ],
        dtype=np.float32,
    )
    factors, weights = np.meshgrid(factors, weights, indexing="ij")
    factors, weights = factors.ravel(), weights.ravel()
    original = wp.empty(factors.size, dtype=wp.float32, device=warp_device)
    selected = wp.empty_like(original)
    bounded = wp.empty(factors.size, dtype=wp.int32, device=warp_device)
    wp.launch(
        compare_products,
        dim=factors.size,
        inputs=[wp.array(factors, device=warp_device), wp.array(weights, device=warp_device)],
        outputs=[original, selected, bounded],
        device=warp_device,
    )
    original_values, selected_values = original.numpy(), selected.numpy()
    np.testing.assert_array_equal(np.isnan(selected_values), np.isnan(original_values))
    valid = ~np.isnan(original_values)
    np.testing.assert_array_equal(
        selected_values[valid].view(np.uint32), original_values[valid].view(np.uint32)
    )
    np.testing.assert_array_equal(
        bounded.numpy(), (np.isfinite(factors) & (np.abs(factors) <= 1)).astype(np.int32)
    )
    with np.errstate(invalid="ignore", under="ignore"):
        expected = np.multiply(factors, weights, dtype=np.float32)
    if warp_device != "cpu":
        underflow = np.abs(expected) < tiny
        expected[underflow] = np.copysign(np.float32(0.0), expected[underflow])
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(
        selected_values[valid].view(np.uint32), expected[valid].view(np.uint32)
    )


@pytest.mark.parametrize(
    "factor",
    [
        np.nextafter(np.float32(-1.0), np.float32(-2.0)),
        -1.0,
        np.nextafter(np.float32(-1.0), np.float32(0.0)),
        -0.0,
        0.0,
        np.nextafter(np.float32(1.0), np.float32(0.0)),
        1.0,
        np.nextafter(np.float32(1.0), np.float32(2.0)),
        -16.0,
        16.0,
        -1e20,
        1e20,
    ],
)
@pytest.mark.parametrize("joint", [False, True])
def test_threshold_and_general_factors_keep_rounded_cell_sum(factor, joint, warp_device):
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((-2.0, 0.0, 0.0),), ((2.0, 0.0, 0.0),))
    seed = np.float32(factor) * np.float32(1024.0)
    reference, actual = evaluate(grid, rays, [seed], 1024, warp_device, joint=joint)
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=0)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_subnormal_ray_factor_does_not_reappear_as_normal_sum(sign, warp_device):
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((-2.0, 0.0, 0.0),), ((2.0, 0.0, 0.0),))
    factor = np.nextafter(np.finfo(np.float32).tiny, np.float32(0.0)) * np.float32(sign)
    seed = factor * np.float32(1024.0)
    reference, actual = evaluate(grid, rays, [seed], 1024, warp_device, double_reference=False)
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=0)
    if warp_device != "cpu":
        np.testing.assert_array_equal(actual, [0.0])
