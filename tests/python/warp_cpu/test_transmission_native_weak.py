"""Weak transmission preserves Decimal rounding and every binary32 scale."""

import importlib
from typing import Any

import pytest

from dpt.transmission import optical_depth_from_removed, prepare_transmission, transmit
from dpt.validation.transmission import forward_reference, inverse_reference, round_binary32

np: Any = importlib.import_module("numpy")
wp: Any = importlib.import_module("warp")


def _points() -> Any:
    positive = np.array([2.0**exponent for exponent in range(-149, 0)], dtype=np.float32)
    seams = np.array([2.0**-25, 0.125, 0.5, 1.0], dtype=np.float32)
    neighbours = np.concatenate(
        (
            np.nextafter(positive, np.float32(0)),
            positive,
            np.nextafter(positive, np.float32(1)),
            np.nextafter(seams, np.float32(0)),
            seams[:-1],
            np.nextafter(seams[:-1], np.float32(1)),
            np.array([0.0, -0.0], dtype=np.float32),
        )
    )
    rng = np.random.default_rng(130879)
    random = np.concatenate(
        (
            rng.integers(0, 0x3F800000, 512, dtype=np.uint32).view(np.float32),
            rng.uniform(0, 1, 2048).astype(np.float32),
        )
    )
    return np.concatenate((neighbours, random))


@pytest.mark.parametrize("inverse", [False, True])
def test_native_weak_functions_against_exact_decimal_rounding(
    inverse: bool, warp_device: str
) -> None:
    values = _points()
    if not inverse:
        values = np.concatenate(
            (
                values,
                np.array(
                    [1.0, 64.0, 87.0, 103.0, 110.0, 200.0, 1024.0, np.finfo(np.float32).max],
                    dtype=np.float32,
                ),
            )
        )
    workspace = prepare_transmission(max_pixels=len(values), device=warp_device)
    source = wp.array(values, dtype=wp.float32, device=warp_device)
    output = wp.empty(len(values), dtype=wp.float32, device=warp_device)
    if inverse:
        optical_depth_from_removed(source, out_L=output, workspace=workspace)
        expected = [round_binary32(inverse_reference(float(x), precision=200)) for x in values]
    else:
        transmit(source, out_removed=output, workspace=workspace)
        expected = [
            round_binary32(forward_reference(float(x), precision=200).removed) for x in values
        ]
    expected_array = np.array(expected, dtype=np.float32)
    np.testing.assert_array_equal(output.numpy().view(np.uint32), expected_array.view(np.uint32))


def test_exact_small_range_identity_and_positive_zero(warp_device: str) -> None:
    values = _points()
    values = values[values <= np.float32(2.0**-25)]
    workspace = prepare_transmission(max_pixels=len(values), device=warp_device)
    source = wp.array(values, dtype=wp.float32, device=warp_device)
    decrement = wp.empty(len(values), dtype=wp.float32, device=warp_device)
    depth = wp.empty(len(values), dtype=wp.float32, device=warp_device)
    transmit(source, out_removed=decrement, workspace=workspace)
    optical_depth_from_removed(source, out_L=depth, workspace=workspace)
    expected = values.copy()
    expected[expected == 0] = 0.0
    np.testing.assert_array_equal(decrement.numpy().view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(depth.numpy().view(np.uint32), expected.view(np.uint32))


def test_native_weak_change_preserves_deep_tail_count_products(warp_device: str) -> None:
    from dpt.transmission import TransmissionSpec

    depths = np.array([0.0, -0.0, 2.0**-149, 2.0**-25, 0.5, 110.0, 200.0], dtype=np.float32)
    beam = np.float32(1.0e30)
    workspace = prepare_transmission(
        TransmissionSpec(beam="scalar"), max_pixels=len(depths), device=warp_device
    )
    field = wp.array(depths, dtype=wp.float32, device=warp_device)
    transmission = wp.empty(len(depths), dtype=wp.float32, device=warp_device)
    counts = wp.empty(len(depths), dtype=wp.float32, device=warp_device)
    removed = wp.empty(len(depths), dtype=wp.float32, device=warp_device)
    transmit(
        field,
        float(beam),
        out_T=transmission,
        out_counts=counts,
        out_removed=removed,
        workspace=workspace,
    )
    expected = [
        forward_reference(float(value), float(beam), precision=200).rounded() for value in depths
    ]
    np.testing.assert_allclose(
        counts.numpy(), [value["counts"] for value in expected], rtol=1.0e-7, atol=0.0
    )
    assert float(transmission.numpy()[5]) == 0.0 and float(counts.numpy()[5]) > 0.0
    np.testing.assert_array_equal(removed.numpy(), [value["removed"] for value in expected])


def test_count_free_specializations_share_compiled_modules() -> None:
    from dpt.kernels.transmission import get_forward_kernel, get_vjp_kernel

    for mask in (1, 4, 5, 8, 9, 12, 13):
        forward = [get_forward_kernel(mask, mode) for mode in range(4)]
        assert all(kernel is forward[0] for kernel in forward)
    for mask in (0, 1, 4, 5, 8, 9, 12, 13):
        for accumulate in (False, True):
            reverse = [get_vjp_kernel(mask, mode, True, False, accumulate) for mode in range(4)]
            assert all(kernel is reverse[0] for kernel in reverse)
    # Required physical beam layouts retain distinct count kernels and all
    # original preparation errors, including a pointwise beam gradient.
    counts = [get_forward_kernel(2, mode) for mode in (1, 2, 3)]
    assert len({id(kernel.module) for kernel in counts}) == 3
    with pytest.raises(ValueError, match="open beam"):
        get_forward_kernel(2, 0)
    with pytest.raises(ValueError, match="per-pixel"):
        get_vjp_kernel(1, 1, False, True, False)


def test_count_free_per_pixel_beam_gradient_remains_exact_zero(warp_device: str) -> None:
    from dpt.transmission import TransmissionSpec, transmission_vjp

    workspace = prepare_transmission(
        TransmissionSpec(beam="per-pixel", active_beam=True), max_pixels=3, device=warp_device
    )
    depths = wp.array([0.0, 0.25, 110.0], dtype=wp.float32, device=warp_device)
    beam = wp.array([1.0, 100.0, 1.0e30], dtype=wp.float32, device=warp_device)
    seed = wp.array([2.0, -3.0, 4.0], dtype=wp.float32, device=warp_device)
    beam_gradient = wp.full(3, 19.0, dtype=wp.float32, device=warp_device)
    depth_gradient = wp.empty(3, dtype=wp.float32, device=warp_device)
    transmission_vjp(
        depths,
        beam,
        cotangent_T=seed,
        out_grad_L=depth_gradient,
        out_grad_n0=beam_gradient,
        workspace=workspace,
    )
    np.testing.assert_array_equal(beam_gradient.numpy(), [0.0, 0.0, 0.0])
    expected = [
        -float(s) * forward_reference(float(x), precision=200).rounded()["T"]
        for x, s in zip(depths.numpy(), seed.numpy(), strict=True)
    ]
    np.testing.assert_allclose(depth_gradient.numpy(), expected, rtol=1.0e-7, atol=0.0)
