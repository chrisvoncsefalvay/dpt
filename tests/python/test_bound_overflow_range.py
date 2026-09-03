"""Original-law regressions for ordinary and finite-exponent overflow mixtures."""

from __future__ import annotations

import hashlib
import math
from decimal import Decimal

import pytest

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
from bound_overflow_reference import (
    finite_reference_shells,
    omitted_physical_log_bound,
    original_cutoff_range,
    overflow_fixtures,
)
from dpt.materials import Provenance
from dpt.transport.bound_compton import (
    FINE_STRUCTURE_CONSTANT,
    BoundComptonTable,
    ComptonShell,
    bound_compton_joint_pdf,
    bound_compton_joint_scores,
)
from dpt.transport.bound_compton_mixed import (
    mixed_angle_proposal,
    mixed_conditional_energy,
    mixed_joint_scores,
    mixed_normalisation,
    mixed_phase_pdf,
)
from dpt.transport.bound_compton_quadrature import bound_angular_normalisation_pair
from helpers.bound_occupancy_models import physical_shells

CASES = overflow_fixtures()


def overflow_table(shells: Shells, z: int) -> BoundComptonTable:
    definition = b"Original-law saturated-truncation arithmetic regression"
    return BoundComptonTable(
        z,
        tuple(
            ComptonShell(str(i), occupancy, binding, height * FINE_STRUCTURE_CONSTANT)
            for i, (occupancy, binding, height) in enumerate(shells)
        ),
        Provenance(
            definition.decode(),
            hashlib.sha256(definition).hexdigest(),
            "Apache-2.0",
            "Synthetic explicit shells",
        ),
    )


@pytest.mark.parametrize("label,energy,z,shells", CASES, ids=[case[0] for case in CASES])
def test_saturated_truncation_cannot_poison_normalisation(
    label: str, energy: float, z: int, shells: Shells
) -> None:
    table = overflow_table(shells, z)
    actual_shells = physical_shells(table)
    index, broad = finite_reference_shells(actual_shells)
    assert omitted_physical_log_bound(energy, actual_shells) < -300
    expected = original_occupancy_normalisation(energy, broad)
    actual = mixed_normalisation(energy, table)
    assert actual.reference_shell == index
    assert math.isfinite(actual.scaled_value) and actual.scaled_value > 0
    assert actual.scaled_value == pytest.approx(float(expected.scaled_value), rel=3e-10, abs=0)
    assert actual.centred_energy_derivative == pytest.approx(
        float(expected.centred_scaled_derivative), rel=3e-9, abs=1e-12
    )
    assert actual.reference_energy_score == pytest.approx(
        float(expected.centre_score), rel=3e-12, abs=0
    )
    raw_value, raw_derivative = bound_angular_normalisation_pair(energy, table)
    assert raw_value == pytest.approx(float(expected.raw_value), rel=3e-10, abs=0)
    assert raw_derivative == pytest.approx(float(expected.raw_derivative), rel=3e-9, abs=0)
    assert (raw_value > 0) == (z == 2)


@pytest.mark.parametrize("label,energy,z,shells", CASES, ids=[case[0] for case in CASES])
def test_saturated_truncation_preserves_finite_production_pdfs_and_scores(
    label: str, energy: float, z: int, shells: Shells
) -> None:
    table = overflow_table(shells, z)
    _, broad = finite_reference_shells(physical_shells(table))
    energy_score = mixed_normalisation(energy, table).energy_score
    for cosine, outgoing in occupancy_states(energy, broad):
        pdf, scores = original_occupancy_joint(energy, cosine, outgoing, broad)
        assert mixed_phase_pdf(energy, cosine, table) == pytest.approx(
            original_occupancy_phase_pdf(energy, cosine, broad), rel=3e-9, abs=0
        )
        assert bound_compton_joint_pdf(energy, cosine, outgoing, table) == pytest.approx(
            pdf, rel=3e-9, abs=0
        )
        assert mixed_joint_scores(energy, cosine, outgoing, table) == pytest.approx(
            scores, rel=3e-10, abs=2e-9
        )
        assert bound_compton_joint_scores(
            energy, cosine, outgoing, table, normalisation_energy_score=energy_score
        ) == pytest.approx(scores, rel=3e-10, abs=2e-9)


@pytest.mark.parametrize("label,energy,z,shells", CASES, ids=[case[0] for case in CASES])
def test_saturated_truncation_preserves_proposal_acceptance(
    label: str, energy: float, z: int, shells: Shells
) -> None:
    table = overflow_table(shells, z)
    _, broad = finite_reference_shells(physical_shells(table))
    for mark in (1e-12, 0.125, 0.5, 0.875, 1 - 1e-12):
        expected_cosine, expected_acceptance = original_occupancy_angle_proposal(
            energy, mark, broad
        )
        cosine, acceptance = mixed_angle_proposal(energy, table, mark)
        assert cosine == pytest.approx(
            expected_cosine, rel=0, abs=max(math.ulp(expected_cosine), 4e-15)
        )
        assert 0 < acceptance <= 1
        assert acceptance == pytest.approx(expected_acceptance, rel=3e-9, abs=0)


@pytest.mark.parametrize("label,energy,z,shells", CASES, ids=[case[0] for case in CASES])
def test_saturated_truncation_preserves_conditional_cdf_and_energy_ledger(
    label: str, energy: float, z: int, shells: Shells
) -> None:
    table = overflow_table(shells, z)
    expected_index, broad = finite_reference_shells(physical_shells(table))
    cosine = original_occupancy_angle_proposal(energy, 0.5, broad)[0]
    for shell_mark in (0.125, 0.5, 0.875):
        for energy_mark in (1e-12, 0.125, 0.5, 0.875, 1 - 1e-12):
            outgoing, index, momentum = mixed_conditional_energy(
                energy, cosine, table, shell_mark, energy_mark
            )
            assert index == expected_index and 0 < outgoing <= energy
            assert original_occupancy_energy_cdf(
                energy, cosine, outgoing, 0, broad
            ) == pytest.approx(energy_mark, rel=0, abs=3e-9)
            assert momentum == pytest.approx(
                original_physical_momentum(energy, cosine, outgoing),
                rel=0,
                abs=4 * math.ulp(momentum),
            )
            recoil = energy - outgoing
            assert recoil >= 0 and outgoing + recoil == pytest.approx(
                energy, rel=0, abs=math.ulp(energy)
            )


def test_corrected_cutoff_candidate_has_finite_profile_and_unrepresentable_truncation():
    cutoff, profile = original_cutoff_range(3.0, 1.0, 1e154)
    largest = Decimal.from_float(float.fromhex("0x1.fffffffffffffp+1023"))
    assert cutoff > largest and -largest < profile < 0


def test_withdrawn_cutoff_candidate_is_representable():
    cutoff, profile = original_cutoff_range(1.01, 1.0, 8e153)
    largest = Decimal.from_float(float.fromhex("0x1.fffffffffffffp+1023"))
    assert 0 < cutoff < largest and -largest < profile < 0
