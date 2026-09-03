"""Near-balanced shell occupancies in deep momentum-profile tails."""

from __future__ import annotations

import math

import pytest

from bound_occupancy_balanced_reference import (
    balanced_fixtures,
    original_balanced_angle_proposal,
    original_balanced_joint,
    original_balanced_normalisation,
    original_balanced_phase_pdf,
    original_balanced_shell_fractions,
    original_balanced_states,
)
from bound_occupancy_reference import Shells
from dpt.transport.bound_compton import bound_compton_joint_pdf, bound_compton_joint_scores
from dpt.transport.bound_compton_mixed import (
    mixed_angle_proposal,
    mixed_conditional_energy,
    mixed_joint_pdf,
    mixed_joint_scores,
    mixed_normalisation,
    mixed_phase_pdf,
)
from dpt.transport.bound_compton_quadrature import bound_angular_normalisation_pair
from helpers.bound_occupancy_checks import ENERGY, assert_discrete_energy_inverse
from helpers.bound_occupancy_models import occupancy_table, physical_shells

CASES = balanced_fixtures()


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_balanced_occupancy_reference_weights_normalisation_and_centred_derivative(
    label: str, shells: Shells
) -> None:
    table = occupancy_table(shells)
    actual_shells = physical_shells(table)
    fractions = original_balanced_shell_fractions(ENERGY, -1.0, actual_shells)
    assert all(0.1 < fraction < 0.9 for fraction in fractions)
    reference = original_balanced_normalisation(ENERGY, actual_shells)
    actual = mixed_normalisation(ENERGY, table)
    assert actual.reference_shell == reference.reference_shell
    assert actual.scaled_value == pytest.approx(float(reference.scaled_value), rel=3e-10, abs=0)
    assert actual.centred_energy_derivative == pytest.approx(
        float(reference.centred_scaled_derivative), rel=3e-9, abs=0
    )
    assert actual.reference_energy_score == pytest.approx(
        float(reference.centre_score), rel=3e-12, abs=0
    )
    assert bound_angular_normalisation_pair(ENERGY, table) == (0.0, 0.0)


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_balanced_occupancy_pdfs_and_all_scores_retain_both_shell_laws(
    label: str, shells: Shells
) -> None:
    table = occupancy_table(shells)
    actual_shells = physical_shells(table)
    normalisation = mixed_normalisation(ENERGY, table)
    for cosine, outgoing in original_balanced_states(ENERGY, actual_shells):
        pdf, scores = original_balanced_joint(ENERGY, cosine, outgoing, actual_shells)
        phase = original_balanced_phase_pdf(ENERGY, cosine, actual_shells)
        assert mixed_phase_pdf(ENERGY, cosine, table) == pytest.approx(phase, rel=3e-9, abs=0)
        assert mixed_joint_pdf(ENERGY, cosine, outgoing, table) == pytest.approx(
            pdf, rel=3e-9, abs=0
        )
        assert mixed_joint_scores(ENERGY, cosine, outgoing, table) == pytest.approx(
            scores, rel=3e-10, abs=2e-9
        )
        assert bound_compton_joint_pdf(ENERGY, cosine, outgoing, table) == pytest.approx(
            pdf, rel=3e-9, abs=0
        )
        assert bound_compton_joint_scores(
            ENERGY,
            cosine,
            outgoing,
            table,
            normalisation_energy_score=normalisation.energy_score,
        ) == pytest.approx(scores, rel=3e-10, abs=2e-9)


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_balanced_occupancy_proposal_preserves_original_uniform_marks(
    label: str, shells: Shells
) -> None:
    table = occupancy_table(shells)
    actual_shells = physical_shells(table)
    for mark in (1e-12, 0.125, 0.5, 0.875, 1 - 1e-12):
        expected_cosine, expected_acceptance = original_balanced_angle_proposal(
            ENERGY, mark, actual_shells
        )
        cosine, acceptance = mixed_angle_proposal(ENERGY, table, mark)
        assert cosine == pytest.approx(expected_cosine, rel=0, abs=math.ulp(expected_cosine))
        assert 0 < acceptance <= 1
        assert acceptance == pytest.approx(expected_acceptance, rel=3e-9, abs=0)


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_balanced_occupancy_conditional_marks_select_both_shells_and_keep_energy_ledger(
    label: str, shells: Shells
) -> None:
    table = occupancy_table(shells)
    actual_shells = physical_shells(table)
    cosine = original_balanced_angle_proposal(ENERGY, 0.5, actual_shells)[0]
    selected = set()
    for shell_mark in (0.125, 0.5, 0.875):
        for energy_mark in (1e-12, 0.125, 0.5, 0.875, 1 - 1e-12):
            outgoing, index, momentum = mixed_conditional_energy(
                ENERGY, cosine, table, shell_mark, energy_mark
            )
            selected.add(index)
            assert_discrete_energy_inverse(
                outgoing, momentum, index, cosine, shell_mark, energy_mark, actual_shells
            )
    assert selected == {0, 1}
