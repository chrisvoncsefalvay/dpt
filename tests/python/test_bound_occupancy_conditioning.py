"""Original-law regressions for subnormal shell occupancies and mixed supports."""

from __future__ import annotations

import math

import pytest

from bound_occupancy_reference import (
    Shells,
    occupancy_fixtures,
    occupancy_states,
    original_occupancy_angle_proposal,
    original_occupancy_energy_cdf,
    original_occupancy_joint,
    original_occupancy_normalisation,
    original_occupancy_phase_pdf,
    original_occupancy_shell,
    original_physical_momentum,
)
from dpt.transport.bound_compton import (
    bound_compton_joint_pdf,
    bound_compton_joint_scores,
)
from dpt.transport.bound_compton_mixed import (
    mixed_angle_proposal,
    mixed_conditional_energy,
    mixed_joint_pdf,
    mixed_joint_scores,
    mixed_normalisation,
    mixed_phase_pdf,
)
from dpt.transport.bound_compton_quadrature import (
    bound_angular_normalisation_pair,
    bound_normalisation_energy_score,
)
from helpers.bound_occupancy_models import occupancy_table, physical_shells

CASES = occupancy_fixtures()
ENERGY = 1.01


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_occupancy_scaled_and_raw_normalisations_preserve_original_physical_law(
    label: str, shells: Shells
) -> None:
    table = occupancy_table(shells)
    expected = original_occupancy_normalisation(ENERGY, physical_shells(table))
    actual = mixed_normalisation(ENERGY, table)
    assert actual.reference_shell == expected.reference_shell
    assert math.isfinite(actual.scaled_value) and actual.scaled_value > 0
    assert actual.scaled_value == pytest.approx(float(expected.scaled_value), rel=3e-10, abs=0)
    assert actual.centred_energy_derivative == pytest.approx(
        float(expected.centred_scaled_derivative), rel=3e-9, abs=1e-12
    )
    assert actual.reference_energy_score == pytest.approx(
        float(expected.centre_score), rel=3e-12, abs=0
    )
    assert actual.energy_score == pytest.approx(float(expected.energy_score), rel=3e-10, abs=0)
    raw_value, raw_derivative = bound_angular_normalisation_pair(ENERGY, table)
    assert raw_value == float(expected.raw_value)
    assert raw_derivative == float(expected.raw_derivative)
    assert bound_normalisation_energy_score(ENERGY, table) == pytest.approx(
        float(expected.energy_score), rel=3e-10, abs=0
    )
    if label.startswith("steep") or label.startswith("negative"):
        assert len(expected.omitted_component_log_bounds) == 1
        assert expected.omitted_component_log_bounds[0][1] < -300


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_occupancy_production_pdfs_and_all_three_scores_at_actual_binary64_states(
    label: str, shells: Shells
) -> None:
    table = occupancy_table(shells)
    original_shells = physical_shells(table)
    energy_score = bound_normalisation_energy_score(ENERGY, table)
    for cosine, outgoing in occupancy_states(ENERGY, original_shells):
        expected_pdf, expected_scores = original_occupancy_joint(
            ENERGY, cosine, outgoing, original_shells
        )
        assert mixed_phase_pdf(ENERGY, cosine, table) == pytest.approx(
            original_occupancy_phase_pdf(ENERGY, cosine, original_shells), rel=3e-9, abs=0
        )
        assert mixed_joint_pdf(ENERGY, cosine, outgoing, table) == pytest.approx(
            expected_pdf, rel=3e-9, abs=0
        )
        assert mixed_joint_scores(ENERGY, cosine, outgoing, table) == pytest.approx(
            expected_scores, rel=3e-10, abs=2e-9
        )
        assert bound_compton_joint_pdf(ENERGY, cosine, outgoing, table) == pytest.approx(
            expected_pdf, rel=3e-9, abs=0
        )
        assert bound_compton_joint_pdf(
            ENERGY, cosine, outgoing, table, solid_angle_normalisation=0.0
        ) == pytest.approx(expected_pdf, rel=3e-9, abs=0)
        assert bound_compton_joint_scores(
            ENERGY, cosine, outgoing, table, normalisation_energy_score=energy_score
        ) == pytest.approx(expected_scores, rel=3e-10, abs=2e-9)
        assert bound_compton_joint_scores(ENERGY, cosine, outgoing, table) == pytest.approx(
            expected_scores[:2], rel=3e-10, abs=2e-9
        )


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_mixed_angle_proposal_keeps_original_uniform_and_acceptance_marks(
    label: str, shells: Shells
) -> None:
    table = occupancy_table(shells)
    original_shells = physical_shells(table)
    for uniform in (1e-12, 0.125, 0.5, 0.875, 1 - 1e-12):
        expected_cosine, expected_acceptance = original_occupancy_angle_proposal(
            ENERGY, uniform, original_shells
        )
        cosine, acceptance = mixed_angle_proposal(ENERGY, table, uniform)
        assert cosine == pytest.approx(
            expected_cosine, rel=0, abs=max(math.ulp(expected_cosine), 4e-15)
        )
        assert 0 < acceptance <= 1
        assert acceptance == pytest.approx(expected_acceptance, rel=3e-9, abs=0)


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_mixed_conditional_energy_preserves_shell_and_profile_cdfs_and_ledger(
    label: str, shells: Shells
) -> None:
    table = occupancy_table(shells)
    original_shells = physical_shells(table)
    cosine = original_occupancy_angle_proposal(ENERGY, 0.5, original_shells)[0]
    energies = []
    for shell_mark in (0.125, 0.5, 0.875):
        for energy_mark in (1e-12, 0.125, 0.5, 0.875, 1 - 1e-12):
            outgoing, index, momentum = mixed_conditional_energy(
                ENERGY, cosine, table, shell_mark, energy_mark
            )
            assert index == original_occupancy_shell(ENERGY, cosine, shell_mark, original_shells)
            binding = original_shells[index][1]
            assert 0 < outgoing <= ENERGY - binding
            # The CDF is evaluated at the actual returned binary64 energy. This
            # checks the original physical inverse without requiring an
            # unrepresentable absolute-momentum inverse near outgoing=0.
            assert original_occupancy_energy_cdf(
                ENERGY, cosine, outgoing, index, original_shells
            ) == pytest.approx(energy_mark, rel=0, abs=3e-9)
            assert momentum == pytest.approx(
                original_physical_momentum(ENERGY, cosine, outgoing),
                rel=0,
                abs=4 * math.ulp(momentum),
            )
            recoil = ENERGY - outgoing - binding
            assert recoil >= 0
            assert outgoing + binding + recoil == pytest.approx(ENERGY, rel=0, abs=math.ulp(ENERGY))
            energies.append(outgoing)
    assert len(set(energies)) >= 5
