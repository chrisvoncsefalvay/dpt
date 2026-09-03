"""Native checks of near-balanced shell occupancies in deep profile tails."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bound_occupancy_balanced_reference import (
    balanced_fixtures,
    original_balanced_angle_proposal,
    original_balanced_energy_inverse,
    original_balanced_joint,
    original_balanced_normalisation,
    original_balanced_phase_pdf,
    original_balanced_states,
)
from bound_occupancy_reference import Shells
from helpers.bound_occupancy_checks import assert_discrete_energy_inverse
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

CASES = balanced_fixtures()
ENERGY = 1.01
MAX_TRIALS = 16


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_native_balanced_shells_preserve_normalisation_and_centred_derivative(
    warp_device: str, label: str, shells: Shells
) -> None:
    reference = original_balanced_normalisation(ENERGY, shells)
    result = wp.empty(1, dtype=wp.vec4d, device=warp_device)
    raw = wp.empty(1, dtype=wp.vec2d, device=warp_device)
    wp.launch(
        _normalisation,
        dim=1,
        inputs=[ENERGY, parameters(warp_device, shells), result, raw],
        device=warp_device,
    )
    index, value, derivative, centre = result.numpy()[0]
    assert index == reference.reference_shell
    assert value == pytest.approx(float(reference.scaled_value), rel=3e-10, abs=0)
    assert derivative == pytest.approx(float(reference.centred_scaled_derivative), rel=3e-9, abs=0)
    assert centre == pytest.approx(float(reference.centre_score), rel=3e-12, abs=0)
    assert tuple(raw.numpy()[0]) == (0.0, 0.0)


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_native_balanced_production_pdfs_and_all_scores_retain_both_shells(
    warp_device: str, label: str, shells: Shells
) -> None:
    states = original_balanced_states(ENERGY, shells)
    densities = wp.empty(len(states), dtype=wp.vec2d, device=warp_device)
    scores = wp.empty(len(states), dtype=wp.vec3d, device=warp_device)
    wp.launch(
        _pdf_scores,
        dim=len(states),
        inputs=[
            ENERGY,
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
        pdf, expected_scores = original_balanced_joint(ENERGY, cosine, outgoing, shells)
        assert actual_density[index, 0] == pytest.approx(
            original_balanced_phase_pdf(ENERGY, cosine, shells), rel=3e-9, abs=0
        )
        assert actual_density[index, 1] == pytest.approx(pdf, rel=3e-9, abs=0)
        assert tuple(actual_scores[index]) == pytest.approx(expected_scores, rel=3e-10, abs=2e-9)


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_native_balanced_proposal_preserves_original_marks(
    warp_device: str, label: str, shells: Shells
) -> None:
    uniforms = (1e-12, 0.125, 0.5, 0.875, 1 - 1e-12)
    result = wp.empty(len(uniforms), dtype=wp.vec2d, device=warp_device)
    wp.launch(
        _proposals,
        dim=len(uniforms),
        inputs=[
            ENERGY,
            parameters(warp_device, shells),
            wp.array(uniforms, dtype=wp.float64, device=warp_device),
            result,
        ],
        device=warp_device,
    )
    for (cosine, acceptance), uniform in zip(result.numpy(), uniforms, strict=True):
        expected_cosine, expected_acceptance = original_balanced_angle_proposal(
            ENERGY, uniform, shells
        )
        assert cosine == pytest.approx(expected_cosine, rel=0, abs=math.ulp(expected_cosine))
        assert 0 < acceptance <= 1
        assert acceptance == pytest.approx(expected_acceptance, rel=3e-9, abs=0)


@pytest.mark.parametrize("next_event", [0, 1])
@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_native_balanced_conditional_energy_preserves_original_counter_marks(
    warp_device: str, label: str, shells: Shells, next_event: int
) -> None:
    cosine = original_balanced_angle_proposal(ENERGY, 0.5, shells)[0]
    count = 32
    result = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    marks = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _energies,
        dim=count,
        inputs=[ENERGY, cosine, parameters(warp_device, shells), next_event, result, marks],
        device=warp_device,
    )
    actual, uniforms = result.numpy(), marks.numpy()
    assert np.all(actual[:, 3] == 0)
    selected = set()
    for row, draws in zip(actual, uniforms, strict=True):
        outgoing, binding, momentum, _ = row
        index = original_balanced_energy_inverse(
            ENERGY, cosine, float(draws[0]), float(draws[1]), shells
        )[1]
        selected.add(index)
        assert binding == shells[index][1]
        assert_discrete_energy_inverse(
            float(outgoing),
            float(momentum),
            index,
            cosine,
            float(draws[0]),
            float(draws[1]),
            shells,
        )
    assert selected == {0, 1}


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_native_balanced_scattering_completes_bounded_trials_with_original_marks(
    warp_device: str, label: str, shells: Shells
) -> None:
    count = 32
    fixture = parameters(warp_device, shells)
    result = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    angles = wp.empty(count * MAX_TRIALS, dtype=wp.vec4d, device=warp_device)
    energies = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(_scatter, dim=count, inputs=[ENERGY, fixture, result], device=warp_device)
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
            cosine, acceptance = original_balanced_angle_proposal(ENERGY, float(marks[0]), shells)
            if marks[1] < acceptance:
                accepted_trials.append(trial)
                assert actual[history, 0] == pytest.approx(cosine, rel=0, abs=math.ulp(cosine))
                assert actual[history, 1] == pytest.approx(2 * math.pi * marks[2], rel=2e-15, abs=0)
                draws = energy_marks[history]
                actual_cosine = float(actual[history, 0])
                outgoing = float(actual[history, 2])
                selected = original_balanced_energy_inverse(
                    ENERGY, actual_cosine, float(draws[0]), float(draws[1]), shells
                )[1]
                assert_discrete_energy_inverse(
                    outgoing,
                    None,
                    selected,
                    actual_cosine,
                    float(draws[0]),
                    float(draws[1]),
                    shells,
                )
                completed = True
                break
        assert completed
    assert len(accepted_trials) == count and max(accepted_trials) < MAX_TRIALS
