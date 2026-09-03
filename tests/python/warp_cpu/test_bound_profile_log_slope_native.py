"""Native profile and complete terminal scores against independent derivatives."""

# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
import importlib
import math
from typing import Any

import pytest

from bound_profile_slope_reference import (
    MIDPOINT_CASES,
    SEAM_CASES,
    SLOPE_CASES,
    TERMINAL_CASES,
    original_joint_cosine_score,
    original_profile_log_slope,
    terminal_state,
)
from helpers.bound_occupancy_native import parameters

wp: Any = importlib.import_module("warp")
bound: Any = importlib.import_module("dpt.transport.bound_compton_kernels")
physical: Any = importlib.import_module("dpt.transport.physical_kernels")


@wp.kernel(enable_backward=False)
def _slopes(momentum: wp.float64, height: wp.float64, result: wp.array(dtype=wp.float64)):
    result[0] = bound.impulse_profile_log_slope(momentum, height)


@wp.kernel(enable_backward=False)
def _joint_score(
    energy: wp.float64,
    cosine: wp.float64,
    outgoing: wp.float64,
    physical: physical.PhysicalParameters,
    result: wp.array(dtype=wp.vec3d),
):
    result[0] = bound.bound_joint_scores(energy, cosine, outgoing, 0, physical)


@pytest.mark.parametrize("momentum,height", SLOPE_CASES)
def test_native_profile_slope_matches_original_product_derivative(warp_device, momentum, height):
    output = wp.empty(1, dtype=wp.float64, device=warp_device)
    wp.launch(_slopes, 1, inputs=[momentum, height, output], device=warp_device)
    actual = float(output.numpy()[0])
    expected = original_profile_log_slope(momentum, height)
    assert math.isfinite(actual)
    if abs(expected) < 1e-300:
        assert actual == expected
    else:
        assert actual == pytest.approx(expected, rel=8e-16, abs=0)


@pytest.mark.parametrize("name,energy,entries", TERMINAL_CASES)
def test_native_terminal_score_keeps_original_decimal90_criterion(
    warp_device, name, energy, entries
):
    cosine, outgoing, table = terminal_state(name, energy, entries)
    if name == "narrow-profile":
        assert cosine == 0.0 and outgoing == 83.63335978121236
    shells = tuple(
        (shell.occupancy, shell.binding_kev, shell.momentum_height) for shell in table.shells
    )
    output = wp.empty(1, dtype=wp.vec3d, device=warp_device)
    wp.launch(
        _joint_score,
        1,
        inputs=[energy, cosine, outgoing, parameters(warp_device, shells), output],
        device=warp_device,
    )
    actual = float(output.numpy()[0][0])
    expected = original_joint_cosine_score(energy, cosine, outgoing, table)
    assert actual == pytest.approx(expected, rel=2e-11, abs=2e-12)


@pytest.mark.parametrize("momentum,height", SEAM_CASES)
def test_native_profile_slope_arithmetic_seam_matches_original_product(
    warp_device, momentum, height
):
    output = wp.empty(1, dtype=wp.float64, device=warp_device)
    wp.launch(_slopes, 1, inputs=[momentum, height, output], device=warp_device)
    actual = float(output.numpy()[0])
    expected = original_profile_log_slope(momentum, height)
    assert math.isfinite(actual)
    assert abs(actual - expected) <= 2 * math.ulp(expected)


@pytest.mark.parametrize("momentum,height", MIDPOINT_CASES)
def test_native_profile_slope_subnormal_midpoint_is_faithfully_rounded(
    warp_device, momentum, height
):
    output = wp.empty(1, dtype=wp.float64, device=warp_device)
    wp.launch(_slopes, 1, inputs=[momentum, height, output], device=warp_device)
    actual = float(output.numpy()[0])
    expected = original_profile_log_slope(momentum, height)
    assert abs(expected) == math.ulp(0.0)
    assert abs(actual - expected) <= math.ulp(0.0)
