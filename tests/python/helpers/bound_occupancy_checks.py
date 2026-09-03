"""Original-CDF discrete energy inverse checks for balanced shell controls."""

from __future__ import annotations

import math

import pytest

from bound_occupancy_balanced_reference import (
    original_balanced_energy_cdf,
    original_balanced_energy_inverse,
)
from bound_occupancy_reference import Shells

ENERGY = 1.01


def assert_discrete_energy_inverse(
    outgoing: float,
    momentum: float | None,
    index: int,
    cosine: float,
    shell_mark: float,
    energy_mark: float,
    shells: Shells,
) -> None:
    expected, expected_index, expected_momentum = original_balanced_energy_inverse(
        ENERGY, cosine, shell_mark, energy_mark, shells
    )
    assert index == expected_index
    assert outgoing == pytest.approx(expected, rel=0, abs=3 * math.ulp(expected))
    if momentum is not None:
        assert momentum == pytest.approx(
            expected_momentum, rel=0, abs=2 * math.ulp(expected_momentum)
        )
    lower = max(math.nextafter(0.0, math.inf), expected - 3 * math.ulp(expected))
    upper = min(ENERGY - shells[index][1], expected + 3 * math.ulp(expected))
    actual_cdf = original_balanced_energy_cdf(ENERGY, cosine, outgoing, index, shells)
    lower_cdf = original_balanced_energy_cdf(ENERGY, cosine, lower, index, shells)
    upper_cdf = original_balanced_energy_cdf(ENERGY, cosine, upper, index, shells)
    # The same original CDF determines the allowed threeULP output grid. The
    # discrete interval must bracket both the input mark and returned state.
    assert lower_cdf <= energy_mark <= upper_cdf
    assert lower_cdf <= actual_cdf <= upper_cdf
    binding = shells[index][1]
    recoil = ENERGY - outgoing - binding
    assert 0 < outgoing <= ENERGY - binding and recoil >= 0
    assert outgoing + binding + recoil == pytest.approx(ENERGY, rel=0, abs=math.ulp(ENERGY))
