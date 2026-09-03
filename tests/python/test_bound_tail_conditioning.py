"""Independent regressions for underflow-conditioned bound-electron laws."""

from __future__ import annotations

import hashlib
import math

import pytest

from bound_tail_reference import (
    decimal_joint_finite_scores,
    extreme_tail_states,
    original_mixture_conditional_energy,
    original_mixture_joint,
    original_mixture_normalisation,
    original_tail_angle_proposal,
    original_tail_conditional_energy,
    original_tail_joint,
    original_tail_normalisation,
    original_tail_phase_pdf,
)
from custom_bound_reference import tiny_height_thomson_pair
from dpt.materials import Provenance
from dpt.transport.bound_compton import (
    FINE_STRUCTURE_CONSTANT,
    BoundComptonTable,
    ComptonShell,
    bound_compton_joint_pdf,
    bound_compton_joint_scores,
    shell_accessibility,
)
from dpt.transport.bound_compton_quadrature import bound_angular_normalisation
from dpt.transport.bound_compton_tail import (
    scaled_bound_normalisation,
    tail_angle_proposal,
    tail_conditional_energy,
    tail_joint_pdf,
    tail_joint_scores,
    tail_phase_pdf,
)


def tail_table(height: float = 1e6, binding: float = 1.0) -> BoundComptonTable:
    definition = b"Independent negative-tail regression: original analytical shell law"
    return BoundComptonTable(
        1,
        (ComptonShell("K", 1.0, binding, height * FINE_STRUCTURE_CONSTANT),),
        Provenance(
            definition.decode(),
            hashlib.sha256(definition).hexdigest(),
            "Apache-2.0",
            "Synthetic fixture with explicit occupancy, binding and profile height",
        ),
    )


@pytest.mark.parametrize("energy", [1.01, 1.010001, math.nextafter(1.0, math.inf)])
def test_scaled_normalisation_retains_original_underflow_conditioned_law(energy: float) -> None:
    table = tail_table()
    shell = table.shells[0]
    assert shell_accessibility(energy, -1.0, shell) == 0
    expected = original_tail_normalisation(energy, shell.binding_kev, shell.momentum_height)
    actual = scaled_bound_normalisation(energy, table)
    assert actual.reference_shell == 0
    assert math.isfinite(actual.scaled_value) and actual.scaled_value > 0
    assert actual.scaled_value == pytest.approx(float(expected.scaled_value), rel=3e-11, abs=0)
    assert actual.centred_energy_derivative == pytest.approx(
        float(expected.centred_scaled_derivative), rel=3e-10, abs=0
    )
    assert actual.reference_energy_score == pytest.approx(
        float(expected.centre_score), rel=3e-14, abs=0
    )
    assert actual.centred_energy_derivative / actual.scaled_value == pytest.approx(
        float(expected.centred_score), rel=3e-10, abs=0
    )


def test_scaled_pdfs_and_three_scores_match_actual_binary64_physical_states() -> None:
    energy, table = 1.01, tail_table()
    shell = table.shells[0]
    height, binding = shell.momentum_height, shell.binding_kev
    normalisation = scaled_bound_normalisation(energy, table)
    for cosine, outgoing in extreme_tail_states(energy, binding, height):
        expected_pdf, expected_scores = original_tail_joint(
            energy, cosine, outgoing, binding, height
        )
        finite_scores = decimal_joint_finite_scores(energy, cosine, outgoing, binding, height)
        assert expected_scores == pytest.approx(finite_scores, rel=3e-15, abs=0)
        assert tail_phase_pdf(energy, cosine, table, normalisation) == pytest.approx(
            original_tail_phase_pdf(energy, cosine, binding, height), rel=3e-10, abs=0
        )
        assert tail_joint_pdf(energy, cosine, outgoing, table, normalisation) == pytest.approx(
            expected_pdf, rel=3e-9, abs=0
        )
        assert tail_joint_scores(energy, cosine, outgoing, table, normalisation) == pytest.approx(
            expected_scores, rel=3e-12, abs=0
        )
        assert bound_compton_joint_pdf(energy, cosine, outgoing, table) == pytest.approx(
            expected_pdf, rel=3e-9, abs=0
        )
        assert bound_compton_joint_pdf(
            energy, cosine, outgoing, table, solid_angle_normalisation=0.0
        ) == pytest.approx(expected_pdf, rel=3e-9, abs=0)
        assert bound_compton_joint_scores(
            energy,
            cosine,
            outgoing,
            table,
            normalisation_energy_score=normalisation.energy_score,
        ) == pytest.approx(expected_scores, rel=3e-12, abs=0)


@pytest.mark.parametrize("uniform", [1e-12, 0.125, 0.5, 0.9, 1 - 1e-12])
@pytest.mark.parametrize("energy", [1.01, math.nextafter(1.0, math.inf)])
def test_tail_angle_proposal_preserves_exact_conditional_angle_law(
    uniform: float, energy: float
) -> None:
    table = tail_table()
    shell = table.shells[0]
    expected_cosine, expected_acceptance = original_tail_angle_proposal(
        energy, uniform, shell.binding_kev, shell.momentum_height
    )
    cosine, acceptance = tail_angle_proposal(energy, table, uniform)
    assert cosine == pytest.approx(expected_cosine, rel=0, abs=math.ulp(expected_cosine))
    assert 0 < acceptance <= 1
    assert acceptance == pytest.approx(expected_acceptance, rel=3e-11, abs=0)


@pytest.mark.parametrize("energy", [1.01, math.nextafter(1.0, math.inf)])
def test_tail_conditional_energy_retains_broadening_and_binding_energy_ledger(
    energy: float,
) -> None:
    table = tail_table()
    shell = table.shells[0]
    reference = original_tail_normalisation(energy, shell.binding_kev, shell.momentum_height)
    rate = float(reference.proposal_rate)
    cosine = -1 + 0.75 / rate
    energies = []
    for uniform in (1e-12, 0.125, 0.5, 0.9, 1 - 1e-12, 0.5281075817765668):
        expected_energy, expected_momentum = original_tail_conditional_energy(
            energy, cosine, uniform, shell.binding_kev, shell.momentum_height
        )
        outgoing, index, momentum = tail_conditional_energy(energy, cosine, table, 0.375, uniform)
        assert index == 0
        assert 0 < outgoing <= energy - shell.binding_kev
        assert outgoing == pytest.approx(expected_energy, rel=0, abs=3 * math.ulp(expected_energy))
        assert momentum == pytest.approx(
            expected_momentum, rel=0, abs=2 * math.ulp(expected_momentum)
        )
        recoil = energy - outgoing - shell.binding_kev
        assert recoil >= 0
        assert outgoing + shell.binding_kev + recoil == pytest.approx(
            energy, rel=0, abs=math.ulp(energy)
        )
        energies.append(outgoing)
    assert len(set(energies)) > 3


def test_near_equal_extreme_shell_weights_keep_the_original_mixture_law() -> None:
    energy = 1.01
    base = tail_table()
    table = BoundComptonTable(
        2,
        (
            base.shells[0],
            ComptonShell(
                "L",
                1.0,
                math.nextafter(1.0, math.inf),
                math.nextafter(1e6, -math.inf) * FINE_STRUCTURE_CONSTANT,
            ),
        ),
        base.provenance,
    )
    shells = tuple((s.occupancy, s.binding_kev, s.momentum_height) for s in table.shells)
    actual = scaled_bound_normalisation(energy, table)
    expected = original_mixture_normalisation(energy, shells, actual.reference_shell)
    assert actual.scaled_value == pytest.approx(float(expected.scaled_value), rel=3e-11, abs=0)
    assert actual.centred_energy_derivative == pytest.approx(
        float(expected.centred_scaled_derivative), rel=3e-10, abs=0
    )
    for cosine, outgoing in extreme_tail_states()[:3]:
        density, scores = original_mixture_joint(
            energy, cosine, outgoing, shells, actual.reference_shell
        )
        assert tail_joint_pdf(energy, cosine, outgoing, table, actual) == pytest.approx(
            density, rel=3e-9, abs=0
        )
        assert tail_joint_scores(energy, cosine, outgoing, table, actual) == pytest.approx(
            scores, rel=3e-12, abs=0
        )
        for shell_uniform in (0.125, 0.5, 0.875):
            expected_out, expected_index, expected_momentum = original_mixture_conditional_energy(
                energy, cosine, shell_uniform, 0.5, shells
            )
            out, index, momentum = tail_conditional_energy(
                energy, cosine, table, shell_uniform, 0.5
            )
            assert index == expected_index
            assert out == pytest.approx(expected_out, rel=0, abs=3 * math.ulp(expected_out))
            assert momentum == pytest.approx(
                expected_momentum, rel=0, abs=2 * math.ulp(expected_momentum)
            )


def test_inaccessible_first_shell_keeps_accessible_conditioned_tail_law() -> None:
    energy = 1.01
    base = tail_table()
    table = BoundComptonTable(
        2,
        (ComptonShell("inactive", 1.0, 2.0, base.shells[0].profile_zero), base.shells[0]),
        base.provenance,
    )
    shell = table.shells[1]
    normalisation = scaled_bound_normalisation(energy, table)
    reference = original_tail_normalisation(energy, shell.binding_kev, shell.momentum_height)
    assert normalisation.reference_shell == 1
    assert normalisation.scaled_value == pytest.approx(
        float(reference.scaled_value), rel=3e-11, abs=0
    )
    for cosine, outgoing in extreme_tail_states()[:3]:
        expected_pdf, expected_scores = original_tail_joint(
            energy, cosine, outgoing, shell.binding_kev, shell.momentum_height
        )
        assert tail_phase_pdf(energy, cosine, table, normalisation) == pytest.approx(
            original_tail_phase_pdf(energy, cosine, shell.binding_kev, shell.momentum_height),
            rel=3e-10,
            abs=0,
        )
        assert bound_compton_joint_pdf(energy, cosine, outgoing, table) == pytest.approx(
            expected_pdf, rel=3e-9, abs=0
        )
        assert bound_compton_joint_scores(
            energy, cosine, outgoing, table, normalisation_energy_score=normalisation.energy_score
        ) == pytest.approx(expected_scores, rel=3e-12, abs=0)
        for uniform in (0.125, 0.5, 0.875):
            out, index, momentum = tail_conditional_energy(energy, cosine, table, uniform, 0.5)
            expected_out, expected_momentum = original_tail_conditional_energy(
                energy, cosine, 0.5, shell.binding_kev, shell.momentum_height
            )
            assert index == 1
            assert out == pytest.approx(expected_out, rel=0, abs=3 * math.ulp(expected_out))
            assert momentum == pytest.approx(
                expected_momentum, rel=0, abs=2 * math.ulp(expected_momentum)
            )
    for uniform in (0.125, 0.5, 0.875):
        cosine, acceptance = tail_angle_proposal(energy, table, uniform)
        expected_cosine, expected_acceptance = original_tail_angle_proposal(
            energy, uniform, shell.binding_kev, shell.momentum_height
        )
        assert cosine == pytest.approx(expected_cosine, rel=0, abs=math.ulp(expected_cosine))
        assert acceptance == pytest.approx(expected_acceptance, rel=3e-11, abs=0)


@pytest.mark.parametrize("scaled_momentum", [18.69, 17.44, 17.46])
def test_near_underflow_dispatch_keeps_normalised_law_across_mass_switch(
    scaled_momentum: float,
) -> None:
    energy, height, mass = 100.0, 1e6, 510.99895069
    a = scaled_momentum / height
    binding = (2 * energy**2 / mass + 2 * energy * a) / (1 + 2 * energy / mass + a)
    table = tail_table(height, binding)
    shell = table.shells[0]
    back_mass = shell_accessibility(energy, -1.0, shell)
    assert back_mass > 0
    if scaled_momentum == 18.69:
        assert bound_angular_normalisation(energy, table) == 0
    else:
        assert bound_angular_normalisation(energy, table) > 0
        assert (back_mass < 1e-280) == (scaled_momentum > 17.45)
    reference = original_tail_normalisation(energy, binding, shell.momentum_height)
    normalisation = scaled_bound_normalisation(energy, table)
    assert normalisation.scaled_value == pytest.approx(
        float(reference.scaled_value), rel=3e-11, abs=0
    )
    assert normalisation.centred_energy_derivative == pytest.approx(
        float(reference.centred_scaled_derivative), rel=3e-10, abs=0
    )
    for cosine, outgoing in extreme_tail_states(energy, binding, shell.momentum_height)[:3]:
        expected_pdf, expected_scores = original_tail_joint(
            energy, cosine, outgoing, binding, shell.momentum_height
        )
        assert bound_compton_joint_pdf(energy, cosine, outgoing, table) == pytest.approx(
            expected_pdf, rel=3e-9, abs=0
        )
        assert bound_compton_joint_scores(
            energy, cosine, outgoing, table, normalisation_energy_score=normalisation.energy_score
        ) == pytest.approx(expected_scores, rel=3e-12, abs=0)
        expected_out, expected_momentum = original_tail_conditional_energy(
            energy, cosine, 0.5, binding, shell.momentum_height
        )
        out, index, momentum = tail_conditional_energy(energy, cosine, table, 0.25, 0.5)
        assert index == 0
        assert out == pytest.approx(expected_out, rel=0, abs=3 * math.ulp(expected_out))
        assert momentum == pytest.approx(
            expected_momentum, rel=0, abs=2 * math.ulp(expected_momentum)
        )
    for uniform in (0.125, 0.5, 0.875):
        cosine, acceptance = tail_angle_proposal(energy, table, uniform)
        expected_cosine, expected_acceptance = original_tail_angle_proposal(
            energy, uniform, binding, shell.momentum_height
        )
        assert cosine == pytest.approx(expected_cosine, rel=0, abs=math.ulp(expected_cosine))
        assert acceptance == pytest.approx(expected_acceptance, rel=3e-11, abs=0)


def test_positive_endpoint_tiny_profile_retains_ordinary_normalisation_and_density() -> None:
    energy, height, cosine, outgoing = 3.0, 1e-300, 0.0, 1.2
    table = tail_table(height, 0.0)
    value, _ = tiny_height_thomson_pair(energy)
    assert bound_angular_normalisation(energy, table) / height == pytest.approx(value, rel=3e-11)
    ratio = outgoing / energy
    factor = 1 + energy / 510.99895069 * (1 - cosine)
    q2 = (1 - ratio) ** 2 + 2 * ratio * (1 - cosine)
    jacobian = (factor - cosine + ratio * (1 - factor * cosine)) / (energy * q2**1.5)
    kn = factor**-3 + factor**-1 - (1 - cosine**2) * factor**-2
    assert bound_compton_joint_pdf(energy, cosine, outgoing, table) == pytest.approx(
        kn * jacobian / value, rel=3e-9, abs=0
    )
