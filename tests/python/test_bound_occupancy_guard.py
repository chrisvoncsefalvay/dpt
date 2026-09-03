"""Reject NaNs from negligible shells whose exponents overflow binary64."""

from __future__ import annotations

import math

import pytest

from bound_occupancy_guard_reference import (
    broad_reference_shells,
    guard_fixtures,
    omitted_physical_log_bound,
)
from bound_occupancy_reference import (
    Shells,
    occupancy_states,
    original_occupancy_angle_proposal,
    original_occupancy_energy_cdf,
    original_occupancy_joint,
    original_occupancy_normalisation,
    original_occupancy_phase_pdf,
    original_physical_momentum,
)
from dpt.transport.bound_compton import bound_compton_joint_pdf, bound_compton_joint_scores
from dpt.transport.bound_compton_mixed import (
    mixed_angle_proposal,
    mixed_conditional_energy,
    mixed_joint_scores,
    mixed_normalisation,
    mixed_phase_pdf,
)
from dpt.transport.bound_compton_quadrature import bound_angular_normalisation_pair
from helpers.bound_occupancy_models import occupancy_table, physical_shells

CASES = guard_fixtures()
ENERGY = 1.01


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_overflowing_irrelevant_shell_cannot_poison_normalisation(
    label: str, shells: Shells
) -> None:
    table = occupancy_table(shells)
    actual_shells = physical_shells(table)
    index, broad = broad_reference_shells(actual_shells)
    assert omitted_physical_log_bound(ENERGY, actual_shells) < -300
    expected = original_occupancy_normalisation(ENERGY, broad)
    actual = mixed_normalisation(ENERGY, table)
    assert actual.reference_shell == index
    assert math.isfinite(actual.scaled_value) and actual.scaled_value > 0
    assert actual.scaled_value == pytest.approx(float(expected.scaled_value), rel=3e-10, abs=0)
    assert actual.centred_energy_derivative == pytest.approx(
        float(expected.centred_scaled_derivative), rel=3e-9, abs=1e-12
    )
    assert actual.reference_energy_score == pytest.approx(
        float(expected.centre_score), rel=3e-12, abs=0
    )
    assert bound_angular_normalisation_pair(ENERGY, table) == (0.0, 0.0)


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_overflowing_irrelevant_shell_preserves_finite_production_pdfs_and_scores(
    label: str, shells: Shells
) -> None:
    table = occupancy_table(shells)
    _, broad = broad_reference_shells(physical_shells(table))
    energy_score = mixed_normalisation(ENERGY, table).energy_score
    for cosine, outgoing in occupancy_states(ENERGY, broad):
        pdf, scores = original_occupancy_joint(ENERGY, cosine, outgoing, broad)
        assert mixed_phase_pdf(ENERGY, cosine, table) == pytest.approx(
            original_occupancy_phase_pdf(ENERGY, cosine, broad), rel=3e-9, abs=0
        )
        assert bound_compton_joint_pdf(ENERGY, cosine, outgoing, table) == pytest.approx(
            pdf, rel=3e-9, abs=0
        )
        assert mixed_joint_scores(ENERGY, cosine, outgoing, table) == pytest.approx(
            scores, rel=3e-10, abs=2e-9
        )
        assert bound_compton_joint_scores(
            ENERGY, cosine, outgoing, table, normalisation_energy_score=energy_score
        ) == pytest.approx(scores, rel=3e-10, abs=2e-9)


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_overflowing_irrelevant_shell_preserves_proposal_acceptance(
    label: str, shells: Shells
) -> None:
    table = occupancy_table(shells)
    _, broad = broad_reference_shells(physical_shells(table))
    for mark in (1e-12, 0.125, 0.5, 0.875, 1 - 1e-12):
        expected_cosine, expected_acceptance = original_occupancy_angle_proposal(
            ENERGY, mark, broad
        )
        cosine, acceptance = mixed_angle_proposal(ENERGY, table, mark)
        assert cosine == pytest.approx(
            expected_cosine, rel=0, abs=max(math.ulp(expected_cosine), 4e-15)
        )
        assert 0 < acceptance <= 1
        assert acceptance == pytest.approx(expected_acceptance, rel=3e-9, abs=0)


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_overflowing_irrelevant_shell_preserves_conditional_cdf_and_energy_ledger(
    label: str, shells: Shells
) -> None:
    table = occupancy_table(shells)
    expected_index, broad = broad_reference_shells(physical_shells(table))
    cosine = original_occupancy_angle_proposal(ENERGY, 0.5, broad)[0]
    for shell_mark in (0.125, 0.5, 0.875):
        for energy_mark in (1e-12, 0.125, 0.5, 0.875, 1 - 1e-12):
            outgoing, index, momentum = mixed_conditional_energy(
                ENERGY, cosine, table, shell_mark, energy_mark
            )
            assert index == expected_index and 0 < outgoing <= ENERGY
            assert original_occupancy_energy_cdf(
                ENERGY, cosine, outgoing, 0, broad
            ) == pytest.approx(energy_mark, rel=0, abs=3e-9)
            assert momentum == pytest.approx(
                original_physical_momentum(ENERGY, cosine, outgoing),
                rel=0,
                abs=4 * math.ulp(momentum),
            )
            recoil = ENERGY - outgoing
            assert recoil >= 0 and outgoing + recoil == pytest.approx(
                ENERGY, rel=0, abs=math.ulp(ENERGY)
            )
