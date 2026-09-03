"""Custom-profile regressions against independent high-precision physical laws."""

import hashlib
import math

import pytest

from custom_bound_reference import (
    decimal_interval_inverse,
    decimal_interval_mass,
    original_angle_narrow_normalisation,
    tiny_height_thomson_pair,
)
from dpt.materials import Provenance
from dpt.transport.bound_compton import (
    FINE_STRUCTURE_CONSTANT,
    BoundComptonTable,
    ComptonShell,
    profile_interval_inverse,
    profile_interval_mass,
    shell_accessibility,
)
from dpt.transport.bound_compton_quadrature import bound_angular_normalisation_pair


def custom_table(height: float, binding: float) -> BoundComptonTable:
    provenance = Provenance(
        "Explicit analytical custom shell",
        hashlib.sha256(b"custom-bound-regression").hexdigest(),
        "CC0-1.0",
        "Independent numerical conditioning regression",
    )
    return BoundComptonTable(
        1, (ComptonShell("custom", 1.0, binding, height * FINE_STRUCTURE_CONSTANT),), provenance
    )


@pytest.mark.parametrize(
    "height,lower,upper",
    [
        (1e-30, -1.0, 0.0058708535427502425),
        (1e-30, -1.0, -0.2),
        (1e-30, 0.1, 0.4),
        (0.8, -1.0, 0.3),
        (0.8, -1.0, -0.2),
        (0.8, 0.1, 0.4),
        (0.8, 0.125, 0.125 + 2e-15),
        (1e6, -1e-6, 1e-6),
        (1e6, -1e-6, -2e-7),
        (1e6, 1e-7, 4e-7),
    ],
)
def test_interval_mass_and_conditional_quantiles_against_decimal(
    height: float,
    lower: float,
    upper: float,
) -> None:
    expected = decimal_interval_mass(lower, upper, height)
    assert expected > 0
    assert profile_interval_mass(lower, upper, height) == pytest.approx(expected, rel=5e-15, abs=0)
    for probability in (1e-12, 0.1, 0.5, 0.9, 1 - 1e-12):
        reference = decimal_interval_inverse(probability, lower, upper, height)
        actual = profile_interval_inverse(probability, lower, upper, height)
        assert lower <= actual <= upper
        assert actual == pytest.approx(reference, rel=7e-15, abs=2e-15 * (upper - lower))


def test_tiny_profile_accessibility_and_normaliser_keep_representable_mass() -> None:
    energy, height = 3.0, 1e-30
    table = custom_table(height, 0.0)
    assert shell_accessibility(energy, -1.0, table.shells[0]) == pytest.approx(
        1.0058708535427502425399016893104561095550324382873107918767477823161884e-30,
        rel=3e-15,
        abs=0,
    )
    value, derivative = bound_angular_normalisation_pair(energy, table)
    reference, reference_derivative = tiny_height_thomson_pair(energy)
    assert value / height == pytest.approx(reference, rel=5e-13)
    assert derivative / height == pytest.approx(reference_derivative, rel=5e-13)
    assert math.isfinite(derivative / value)


def test_narrow_profile_normaliser_and_energy_derivative_resolve_angular_transition() -> None:
    energy = 100.0
    value, derivative = bound_angular_normalisation_pair(energy, custom_table(1e6, 1.0))
    reference = original_angle_narrow_normalisation(energy)
    refined = original_angle_narrow_normalisation(energy, 8192)
    assert reference == pytest.approx(refined, rel=5e-14)
    assert value == pytest.approx(reference, rel=5e-13)
    step = 0.005
    references = [
        original_angle_narrow_normalisation(energy + offset * step) for offset in (-2, -1, 1, 2)
    ]
    reference_derivative = (
        references[0] - 8 * references[1] + 8 * references[2] - references[3]
    ) / (12 * step)
    assert derivative == pytest.approx(reference_derivative, rel=2e-9, abs=2e-12)
