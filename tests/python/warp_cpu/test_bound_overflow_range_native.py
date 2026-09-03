"""Native production no-NaN regressions for an irrelevant overflowing shell."""

from __future__ import annotations

import math

import numpy as np
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
    overflow_fixtures,
)
from helpers.bound_occupancy_native import (
    _energies,
    _normalisation,
    _pdf_scores,
    _proposals,
    _scatter,
    _scatter_marks,
    parameters,
    wp,
)

CASES = overflow_fixtures()
MAX_TRIALS = 16


@pytest.mark.parametrize("label,energy,z,shells", CASES, ids=[case[0] for case in CASES])
def test_native_saturated_truncation_keeps_finite_scaled_normalisation(
    warp_device: str, label: str, energy: float, z: int, shells: Shells
) -> None:
    index, broad = finite_reference_shells(shells)
    assert omitted_physical_log_bound(energy, shells) < -300
    reference = original_occupancy_normalisation(energy, broad)
    result = wp.empty(1, dtype=wp.vec4d, device=warp_device)
    raw = wp.empty(1, dtype=wp.vec2d, device=warp_device)
    wp.launch(
        _normalisation,
        dim=1,
        inputs=[energy, parameters(warp_device, shells), result, raw],
        device=warp_device,
    )
    actual_index, value, derivative, centre = result.numpy()[0]
    assert actual_index == index and math.isfinite(value) and value > 0
    assert value == pytest.approx(float(reference.scaled_value), rel=3e-10, abs=0)
    assert derivative == pytest.approx(
        float(reference.centred_scaled_derivative), rel=3e-9, abs=1e-12
    )
    assert centre == pytest.approx(float(reference.centre_score), rel=3e-12, abs=0)
    raw_value, raw_derivative = raw.numpy()[0]
    assert raw_value == pytest.approx(float(reference.raw_value), rel=3e-10, abs=0)
    assert raw_derivative == pytest.approx(float(reference.raw_derivative), rel=3e-9, abs=0)
    assert (raw_value > 0) == (z == 2)


@pytest.mark.parametrize("label,energy,z,shells", CASES, ids=[case[0] for case in CASES])
def test_native_saturated_truncation_keeps_finite_production_pdfs_and_three_scores(
    warp_device: str, label: str, energy: float, z: int, shells: Shells
) -> None:
    _, broad = finite_reference_shells(shells)
    states = occupancy_states(energy, broad)
    densities = wp.empty(len(states), dtype=wp.vec2d, device=warp_device)
    scores = wp.empty(len(states), dtype=wp.vec3d, device=warp_device)
    wp.launch(
        _pdf_scores,
        dim=len(states),
        inputs=[
            energy,
            parameters(warp_device, shells),
            wp.array(states, dtype=wp.vec2d, device=warp_device),
            densities,
            scores,
        ],
        device=warp_device,
    )
    actual_density, actual_scores = densities.numpy(), scores.numpy()
    assert np.isfinite(actual_density).all() and (actual_density > 0).all()
    assert np.isfinite(actual_scores).all()
    for index, (cosine, outgoing) in enumerate(states):
        pdf, expected_scores = original_occupancy_joint(energy, cosine, outgoing, broad)
        assert actual_density[index, 0] == pytest.approx(
            original_occupancy_phase_pdf(energy, cosine, broad), rel=3e-9, abs=0
        )
        assert actual_density[index, 1] == pytest.approx(pdf, rel=3e-9, abs=0)
        assert tuple(actual_scores[index]) == pytest.approx(expected_scores, rel=3e-10, abs=2e-9)


@pytest.mark.parametrize("label,energy,z,shells", CASES, ids=[case[0] for case in CASES])
def test_native_saturated_truncation_keeps_proposal_acceptance(
    warp_device: str, label: str, energy: float, z: int, shells: Shells
) -> None:
    _, broad = finite_reference_shells(shells)
    uniforms = (1e-12, 0.125, 0.5, 0.875, 1 - 1e-12)
    result = wp.empty(len(uniforms), dtype=wp.vec2d, device=warp_device)
    wp.launch(
        _proposals,
        dim=len(uniforms),
        inputs=[
            energy,
            parameters(warp_device, shells),
            wp.array(uniforms, dtype=wp.float64, device=warp_device),
            result,
        ],
        device=warp_device,
    )
    for (cosine, acceptance), mark in zip(result.numpy(), uniforms, strict=True):
        expected_cosine, expected_acceptance = original_occupancy_angle_proposal(
            energy, mark, broad
        )
        assert cosine == pytest.approx(
            expected_cosine, rel=0, abs=max(math.ulp(expected_cosine), 4e-15)
        )
        assert 0 < acceptance <= 1
        assert acceptance == pytest.approx(expected_acceptance, rel=3e-9, abs=0)


@pytest.mark.parametrize("next_event", [0, 1])
@pytest.mark.parametrize("label,energy,z,shells", CASES, ids=[case[0] for case in CASES])
def test_native_saturated_truncation_keeps_conditional_marks_and_ledger(
    warp_device: str, label: str, energy: float, z: int, shells: Shells, next_event: int
) -> None:
    _, broad = finite_reference_shells(shells)
    cosine = original_occupancy_angle_proposal(energy, 0.5, broad)[0]
    count = 32
    result = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    marks = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _energies,
        dim=count,
        inputs=[energy, cosine, parameters(warp_device, shells), next_event, result, marks],
        device=warp_device,
    )
    actual, uniforms = result.numpy(), marks.numpy()
    assert np.all(actual[:, 3] == 0) and np.all(actual[:, 1] == broad[0][1])
    assert len(set(actual[:, 0])) > 16
    for row, draws in zip(actual, uniforms, strict=True):
        outgoing, _, momentum, _ = row
        assert 0 < outgoing <= energy
        assert original_occupancy_energy_cdf(
            energy, cosine, float(outgoing), 0, broad
        ) == pytest.approx(float(draws[1]), rel=0, abs=3e-9)
        assert momentum == pytest.approx(
            original_physical_momentum(energy, cosine, float(outgoing)),
            rel=0,
            abs=4 * math.ulp(float(momentum)),
        )
        recoil = energy - outgoing
        assert recoil >= 0 and outgoing + recoil == pytest.approx(
            energy, rel=0, abs=math.ulp(energy)
        )


@pytest.mark.parametrize("label,energy,z,shells", CASES, ids=[case[0] for case in CASES])
def test_native_saturated_truncation_completes_bounded_scatter_with_original_marks(
    warp_device: str, label: str, energy: float, z: int, shells: Shells
) -> None:
    _, broad = finite_reference_shells(shells)
    count = 32
    fixture = parameters(warp_device, shells)
    result = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    angles = wp.empty(count * MAX_TRIALS, dtype=wp.vec4d, device=warp_device)
    energies = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(_scatter, dim=count, inputs=[energy, fixture, result], device=warp_device)
    wp.launch(
        _scatter_marks,
        dim=count * MAX_TRIALS,
        inputs=[fixture, angles, energies],
        device=warp_device,
    )
    actual = result.numpy()
    angle_marks = angles.numpy().reshape(count, MAX_TRIALS, 4)
    energy_marks = energies.numpy()
    assert np.all(actual[:, 3] == 0)
    assert len(set(actual[:, 0])) > 16 and len(set(actual[:, 2])) > 16
    accepted_trials = []
    for history in range(count):
        completed = False
        for trial in range(MAX_TRIALS):
            marks = angle_marks[history, trial]
            cosine, acceptance = original_occupancy_angle_proposal(energy, float(marks[0]), broad)
            if marks[1] < acceptance:
                accepted_trials.append(trial)
                assert actual[history, 0] == pytest.approx(
                    cosine, rel=0, abs=max(math.ulp(cosine), 4e-15)
                )
                assert actual[history, 1] == pytest.approx(2 * math.pi * marks[2], rel=2e-15, abs=0)
                outgoing = float(actual[history, 2])
                assert 0 < outgoing <= energy
                assert original_occupancy_energy_cdf(
                    energy, float(actual[history, 0]), outgoing, 0, broad
                ) == pytest.approx(float(energy_marks[history, 1]), rel=0, abs=3e-9)
                recoil = energy - outgoing
                assert recoil >= 0 and outgoing + recoil == pytest.approx(
                    energy, rel=0, abs=math.ulp(energy)
                )
                completed = True
                break
        assert completed
    assert len(accepted_trials) == count and max(accepted_trials) < MAX_TRIALS
