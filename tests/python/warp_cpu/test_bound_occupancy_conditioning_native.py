"""Bounded native checks of occupancy-conditioned production laws and marks."""

# Warp annotations are runtime DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
from __future__ import annotations

import math

import numpy as np
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

CASES = occupancy_fixtures()
ENERGY = 1.01
MAX_TRIALS = 16


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_native_occupancy_normalisation_keeps_physical_raw_and_scaled_law(
    warp_device: str, label: str, shells: Shells
) -> None:
    expected = original_occupancy_normalisation(ENERGY, shells)
    result = wp.empty(1, dtype=wp.vec4d, device=warp_device)
    raw = wp.empty(1, dtype=wp.vec2d, device=warp_device)
    wp.launch(
        _normalisation,
        dim=1,
        inputs=[ENERGY, parameters(warp_device, shells), result, raw],
        device=warp_device,
    )
    index, value, derivative, centre = result.numpy()[0]
    assert index == expected.reference_shell
    assert math.isfinite(value) and value > 0
    assert value == pytest.approx(float(expected.scaled_value), rel=3e-10, abs=0)
    assert derivative == pytest.approx(
        float(expected.centred_scaled_derivative), rel=3e-9, abs=1e-12
    )
    assert centre == pytest.approx(float(expected.centre_score), rel=3e-12, abs=0)
    assert centre + derivative / value == pytest.approx(
        float(expected.energy_score), rel=3e-10, abs=0
    )
    assert tuple(raw.numpy()[0]) == (float(expected.raw_value), float(expected.raw_derivative))


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_native_production_occupancy_pdfs_and_three_scores(
    warp_device: str, label: str, shells: Shells
) -> None:
    states = occupancy_states(ENERGY, shells)
    density = wp.empty(len(states), dtype=wp.vec2d, device=warp_device)
    scores = wp.empty(len(states), dtype=wp.vec3d, device=warp_device)
    wp.launch(
        _pdf_scores,
        dim=len(states),
        inputs=[
            ENERGY,
            parameters(warp_device, shells),
            wp.array(states, dtype=wp.vec2d, device=warp_device),
            density,
            scores,
        ],
        device=warp_device,
    )
    actual_density, actual_scores = density.numpy(), scores.numpy()
    assert np.isfinite(actual_density).all() and (actual_density > 0).all()
    assert np.isfinite(actual_scores).all()
    for index, (cosine, outgoing) in enumerate(states):
        expected_pdf, expected_scores = original_occupancy_joint(ENERGY, cosine, outgoing, shells)
        assert actual_density[index, 0] == pytest.approx(
            original_occupancy_phase_pdf(ENERGY, cosine, shells), rel=3e-9, abs=0
        )
        assert actual_density[index, 1] == pytest.approx(expected_pdf, rel=3e-9, abs=0)
        assert tuple(actual_scores[index]) == pytest.approx(expected_scores, rel=3e-10, abs=2e-9)


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_native_mixed_proposal_preserves_original_uniform_marks(
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
        expected_cosine, expected_acceptance = original_occupancy_angle_proposal(
            ENERGY, uniform, shells
        )
        assert cosine == pytest.approx(
            expected_cosine, rel=0, abs=max(math.ulp(expected_cosine), 4e-15)
        )
        assert 0 < acceptance <= 1
        assert acceptance == pytest.approx(expected_acceptance, rel=3e-9, abs=0)


@pytest.mark.parametrize("next_event", [0, 1])
@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_native_production_conditional_energy_keeps_counter_marks_shell_and_ledger(
    warp_device: str, label: str, shells: Shells, next_event: int
) -> None:
    cosine = original_occupancy_angle_proposal(ENERGY, 0.5, shells)[0]
    count = 32
    values = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    marks = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _energies,
        dim=count,
        inputs=[ENERGY, cosine, parameters(warp_device, shells), next_event, values, marks],
        device=warp_device,
    )
    actual, uniforms = values.numpy(), marks.numpy()
    assert np.all(actual[:, 3] == 0)
    assert len(set(actual[:, 0])) > 16
    for row, draws in zip(actual, uniforms, strict=True):
        outgoing, binding, momentum, _ = row
        index = original_occupancy_shell(ENERGY, cosine, float(draws[0]), shells)
        assert binding == shells[index][1]
        assert 0 < outgoing <= ENERGY - binding
        assert original_occupancy_energy_cdf(
            ENERGY, cosine, float(outgoing), index, shells
        ) == pytest.approx(float(draws[1]), rel=0, abs=3e-9)
        assert momentum == pytest.approx(
            original_physical_momentum(ENERGY, cosine, float(outgoing)),
            rel=0,
            abs=4 * math.ulp(float(momentum)),
        )
        recoil = ENERGY - outgoing - binding
        assert recoil >= 0
        assert outgoing + binding + recoil == pytest.approx(ENERGY, rel=0, abs=math.ulp(ENERGY))


@pytest.mark.parametrize("label,shells", CASES, ids=[case[0] for case in CASES])
def test_native_production_scatter_completes_bounded_trials_with_original_marks(
    warp_device: str, label: str, shells: Shells
) -> None:
    count = 32
    fixture = parameters(warp_device, shells)
    result = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    angle_marks = wp.empty(count * MAX_TRIALS, dtype=wp.vec4d, device=warp_device)
    energy_marks = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(_scatter, dim=count, inputs=[ENERGY, fixture, result], device=warp_device)
    wp.launch(
        _scatter_marks,
        dim=count * MAX_TRIALS,
        inputs=[fixture, angle_marks, energy_marks],
        device=warp_device,
    )
    actual = result.numpy()
    assert np.all(actual[:, 3] == 0)
    assert np.all((-1 <= actual[:, 0]) & (actual[:, 0] < 1))
    assert len(set(actual[:, 0])) > 16 and len(set(actual[:, 2])) > 16
    angles = angle_marks.numpy().reshape(count, MAX_TRIALS, 4)
    energies = energy_marks.numpy()
    accepted_trials = []
    for history in range(count):
        completed = False
        for trial in range(MAX_TRIALS):
            draws = angles[history, trial]
            cosine, acceptance = original_occupancy_angle_proposal(ENERGY, float(draws[0]), shells)
            if draws[1] < acceptance:
                accepted_trials.append(trial)
                assert actual[history, 0] == pytest.approx(
                    cosine, rel=0, abs=max(math.ulp(cosine), 4e-15)
                )
                assert actual[history, 1] == pytest.approx(2 * math.pi * draws[2], rel=2e-15, abs=0)
                # Use the actual production angle in the original conditional
                # energy law. Rounded cosines can differ by one binary64 ULP.
                selected = original_occupancy_shell(
                    ENERGY, float(actual[history, 0]), float(energies[history, 0]), shells
                )
                outgoing = float(actual[history, 2])
                binding = shells[selected][1]
                assert 0 < outgoing <= ENERGY - binding
                assert original_occupancy_energy_cdf(
                    ENERGY, float(actual[history, 0]), outgoing, selected, shells
                ) == pytest.approx(float(energies[history, 1]), rel=0, abs=3e-9)
                recoil = ENERGY - outgoing - binding
                assert recoil >= 0
                assert outgoing + binding + recoil == pytest.approx(
                    ENERGY, rel=0, abs=math.ulp(ENERGY)
                )
                completed = True
                break
        assert completed
    assert len(accepted_trials) == count and max(accepted_trials) < MAX_TRIALS
