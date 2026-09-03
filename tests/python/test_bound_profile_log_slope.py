"""Profile precision controls retain the original terminal score criterion."""

import math

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
from dpt.transport.bound_compton import _profile_log_slope, bound_compton_joint_scores


@pytest.mark.parametrize("momentum,height", SLOPE_CASES)
def test_host_profile_slope_matches_original_product_derivative(momentum, height):
    expected = original_profile_log_slope(momentum, height)
    actual = _profile_log_slope(momentum, height)
    assert math.isfinite(actual)
    if abs(expected) < 1e-300:
        assert actual == expected
    else:
        assert actual == pytest.approx(expected, rel=8e-16, abs=0)


@pytest.mark.parametrize("name,energy,entries", TERMINAL_CASES)
def test_host_terminal_score_keeps_original_decimal90_criterion(name, energy, entries):
    cosine, outgoing, table = terminal_state(name, energy, entries)
    if name == "narrow-profile":
        assert cosine == 0.0 and outgoing == 83.63335978121236
    actual = bound_compton_joint_scores(energy, cosine, outgoing, table)[0]
    expected = original_joint_cosine_score(energy, cosine, outgoing, table)
    assert actual == pytest.approx(expected, rel=2e-11, abs=2e-12)


@pytest.mark.parametrize("momentum,height", SEAM_CASES)
def test_host_profile_slope_arithmetic_seam_matches_original_product(momentum, height):
    expected = original_profile_log_slope(momentum, height)
    actual = _profile_log_slope(momentum, height)
    assert math.isfinite(actual)
    assert abs(actual - expected) <= 2 * math.ulp(expected)


@pytest.mark.parametrize("momentum,height", MIDPOINT_CASES)
def test_host_profile_slope_subnormal_midpoint_is_faithfully_rounded(momentum, height):
    expected = original_profile_log_slope(momentum, height)
    actual = _profile_log_slope(momentum, height)
    assert abs(expected) == math.ulp(0.0)
    assert abs(actual - expected) <= math.ulp(0.0)
