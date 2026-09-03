"""Domain and conditioning contracts for the three-material worked example."""

import math
from dataclasses import replace

import pytest

from dpt.contracts import ContractError, TrialDomainError
from dpt.examples.three_material import fraction_complement, synthetic_model


@pytest.mark.parametrize(
    "coordinates, expected",
    [
        ((0.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 1.0), (0.0, 1.0, 0.0)),
        ((0.125, 0.25), (0.125, 0.25, 0.625)),
        ((0.1, 0.2), (0.1, 0.2, 0.7)),
    ],
)
def test_two_coordinates_fill_cell(
    coordinates: tuple[float, float], expected: tuple[float, float, float]
) -> None:
    fractions = fraction_complement(coordinates)
    assert fractions == expected
    assert math.fsum(fractions) == 1.0


@pytest.mark.parametrize(
    "coordinates",
    [(-0.01, 0.2), (0.4, -0.01), (0.7, 0.4), (math.nan, 0.2), (0.2, math.inf), (0.2,)],
)
def test_outside_fraction_chart_is_rejected(coordinates: tuple[float, ...]) -> None:
    with pytest.raises(TrialDomainError):
        fraction_complement(coordinates)


def test_identifiable_declared_synthetic_model() -> None:
    model = synthetic_model()
    assert model.contrast_matrix == ((0.5, -0.25), (-0.25, 0.75))
    assert model.contrast_condition == pytest.approx((3 + math.sqrt(5)) / 2, rel=2e-15)
    assert model.global_condition_bound == pytest.approx((3 + math.sqrt(5)) / 2 * math.e, rel=2e-15)
    assert model.global_condition_bound < model.maximum_condition
    assert model.energies_kev == (50.0, 100.0)
    assert model.path_mm == 8.0
    assert all("Synthetic" in record.description for record in model.coefficients_provenance)


@pytest.mark.parametrize(
    "coefficients",
    [
        ((0.125, 0.03125), (0.125, 0.03125), (0.0625, 0.0625)),
        ((0.0, 0.03125), (0.0625, 0.0625), (0.125, 0.09375)),
    ],
)
def test_rank_one_material_contrasts_are_rejected(
    coefficients: tuple[tuple[float, float], ...],
) -> None:
    with pytest.raises(ContractError, match="rank less than two"):
        replace(synthetic_model(), coefficients_mm_inverse=coefficients)


@pytest.mark.parametrize(
    "bands",
    [((4096.0, 0.0), (4096.0, 0.0)), ((2048.0, 2048.0), (4096.0, 4096.0))],
)
def test_duplicate_energy_bands_are_rejected(bands: tuple[tuple[float, float], ...]) -> None:
    with pytest.raises(ContractError, match="linearly dependent"):
        replace(synthetic_model(), band_weights=bands)


def test_nearly_degenerate_bands_fail_conditioning_admission() -> None:
    with pytest.raises(ContractError, match="conditioning exceeds"):
        replace(synthetic_model(), band_weights=((4096.0, 4096.0), (4096.0, 4096.5)))


@pytest.mark.parametrize(
    "changes",
    [
        {"path_mm": 0.0},
        {"path_mm": math.inf},
        {"response": (1.0, 0.0)},
        {"band_weights": ((0.0, 0.0), (0.0, 4096.0))},
        {"energies_kev": (100.0, 50.0)},
        {"maximum_condition": 2.0},
        {"coefficients_provenance": ()},
        {"coefficients_mm_inverse": ((-0.1, 0.1), (0.2, 0.1), (0.3, 0.2))},
    ],
)
def test_invalid_known_acquisition_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ContractError):
        replace(synthetic_model(), **changes)


def test_two_mixed_bands_remain_identifiable_when_independent() -> None:
    model = replace(
        synthetic_model(), band_weights=((3072.0, 1024.0), (1024.0, 3072.0)), maximum_condition=20.0
    )
    assert model.response_condition((0.25, 0.375)) < model.global_condition_bound
    assert model.global_condition_bound < 20.0
