"""Production native regressions for extreme conditioned impulse tails."""

# Warp annotations and decorators are runtime DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np
import pytest

from bound_tail_reference import (
    extreme_tail_states,
    original_mixture_conditional_energy,
    original_mixture_joint,
    original_mixture_normalisation,
    original_tail_angle_cdf,
    original_tail_angle_proposal,
    original_tail_conditional_energy,
    original_tail_joint,
    original_tail_normalisation,
    original_tail_phase_pdf,
)
from custom_bound_reference import tiny_height_thomson_pair

wp: Any = importlib.import_module("warp")
bound: Any = importlib.import_module("dpt.transport.bound_compton_kernels")
physical: Any = importlib.import_module("dpt.transport.physical_kernels")
sampling: Any = importlib.import_module("dpt.transport.sampling_kernels")

SEED = 48217
EVENT = 3
MAX_TRIALS = 16


@wp.kernel(enable_backward=False)
def _normalisation(
    energy: wp.float64,
    parameters: physical.PhysicalParameters,
    result: wp.array(dtype=wp.vec4d),
):
    result[0] = bound.impulse_tail_normalisation(energy, 0, parameters)


@wp.kernel(enable_backward=False)
def _ordinary_normalisation(
    energy: wp.float64,
    parameters: physical.PhysicalParameters,
    result: wp.array(dtype=wp.vec2d),
):
    result[0] = bound.impulse_angular_normalisation_pair(energy, 0, parameters)


@wp.kernel(enable_backward=False)
def _pdf_scores(
    energy: wp.float64,
    parameters: physical.PhysicalParameters,
    states: wp.array(dtype=wp.vec2d),
    densities: wp.array(dtype=wp.vec2d),
    scores: wp.array(dtype=wp.vec3d),
):
    index = wp.tid()
    cosine, outgoing = states[index][0], states[index][1]
    densities[index] = wp.vec2d(
        bound.bound_phase_pdf(energy, cosine, 0, parameters),
        bound.bound_joint_pdf(energy, cosine, outgoing, 0, parameters),
    )
    scores[index] = bound.bound_joint_scores(energy, cosine, outgoing, 0, parameters)


@wp.kernel(enable_backward=False)
def _angle_proposals(
    energy: wp.float64,
    parameters: physical.PhysicalParameters,
    uniforms: wp.array(dtype=wp.float64),
    result: wp.array(dtype=wp.vec2d),
):
    index = wp.tid()
    result[index] = bound.impulse_tail_angle_proposal(energy, 0, uniforms[index], parameters)


@wp.kernel(enable_backward=False)
def _conditional_events(
    energy: wp.float64,
    cosine: wp.float64,
    parameters: physical.PhysicalParameters,
    next_event: int,
    result: wp.array(dtype=wp.vec4d),
    marks: wp.array(dtype=wp.vec4d),
):
    index = wp.tid()
    result[index] = bound.bound_energy_event(
        energy, cosine, 0, wp.uint64(48217), wp.uint64(index), wp.uint32(3), next_event, parameters
    )
    domain = wp.uint32(0x10000000) + wp.uint32(parameters.random_branch) * wp.uint32(65536)
    if next_event != 0:
        domain += wp.uint32(0x10000000)
    marks[index] = sampling.transport_random4(
        wp.uint64(48217), wp.uint64(index), wp.uint32(3), domain, parameters.sampling
    )


@wp.kernel(enable_backward=False)
def _scattering_events(
    energy: wp.float64,
    parameters: physical.PhysicalParameters,
    result: wp.array(dtype=wp.vec4d),
):
    index = wp.tid()
    result[index] = bound.bound_scatter_event(
        energy, 0, wp.uint64(48217), wp.uint64(index), wp.uint32(3), 16, parameters
    )


@wp.kernel(enable_backward=False)
def _scattering_marks(
    parameters: physical.PhysicalParameters,
    angle_marks: wp.array(dtype=wp.vec4d),
    energy_marks: wp.array(dtype=wp.vec4d),
):
    index = wp.tid()
    history = index // 16
    trial = index % 16
    domain = (
        wp.uint32(0x80000000)
        + wp.uint32(parameters.random_branch) * wp.uint32(65536)
        + wp.uint32(trial)
    )
    angle_marks[index] = sampling.transport_random4(
        wp.uint64(48217), wp.uint64(history), wp.uint32(3), domain, parameters.sampling
    )
    if trial == 0:
        energy_domain = wp.uint32(0x10000000) + wp.uint32(parameters.random_branch) * wp.uint32(
            65536
        )
        energy_marks[history] = sampling.transport_random4(
            wp.uint64(48217), wp.uint64(history), wp.uint32(3), energy_domain, parameters.sampling
        )


def _parameters(device: str, height: float = 1e6, binding: float = 1.0) -> Any:
    parameters = physical.PhysicalParameters()
    parameters.compton_shell_offsets = wp.array([0, 1], dtype=wp.int32, device=device)
    parameters.compton_occupancy = wp.array([1.0], dtype=wp.float64, device=device)
    parameters.compton_binding = wp.array([binding], dtype=wp.float64, device=device)
    parameters.compton_height = wp.array([height], dtype=wp.float64, device=device)
    return parameters


def _mixture_parameters(device: str) -> tuple[Any, tuple[tuple[float, float, float], ...]]:
    shells = (
        (1.0, 1.0, 1e6),
        (1.0, math.nextafter(1.0, math.inf), math.nextafter(1e6, -math.inf)),
    )
    parameters = physical.PhysicalParameters()
    parameters.compton_shell_offsets = wp.array([0, 2], dtype=wp.int32, device=device)
    parameters.compton_occupancy = wp.array([s[0] for s in shells], dtype=wp.float64, device=device)
    parameters.compton_binding = wp.array([s[1] for s in shells], dtype=wp.float64, device=device)
    parameters.compton_height = wp.array([s[2] for s in shells], dtype=wp.float64, device=device)
    return parameters, shells


@pytest.mark.parametrize("energy", [1.01, 1.010001, math.nextafter(1.0, math.inf)])
def test_native_scaled_normalisation_matches_independent_original_angle_law(
    warp_device: str, energy: float
) -> None:
    expected = original_tail_normalisation(energy)
    result = wp.empty(1, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _normalisation,
        dim=1,
        inputs=[energy, _parameters(warp_device), result],
        device=warp_device,
    )
    reference_shell, value, derivative, centre = result.numpy()[0]
    assert reference_shell == 0
    assert math.isfinite(value) and value > 0
    assert value == pytest.approx(float(expected.scaled_value), rel=3e-11, abs=0)
    assert derivative == pytest.approx(float(expected.centred_scaled_derivative), rel=3e-10, abs=0)
    assert centre == pytest.approx(float(expected.centre_score), rel=3e-14, abs=0)
    assert derivative / value == pytest.approx(float(expected.centred_score), rel=3e-10, abs=0)


def test_production_native_joint_pdfs_and_scores_match_binary64_physical_states(
    warp_device: str,
) -> None:
    energy, binding, height = 1.01, 1.0, 1e6
    states = extreme_tail_states(energy, binding, height)
    densities = wp.empty(len(states), dtype=wp.vec2d, device=warp_device)
    scores = wp.empty(len(states), dtype=wp.vec3d, device=warp_device)
    wp.launch(
        _pdf_scores,
        dim=len(states),
        inputs=[
            energy,
            _parameters(warp_device),
            wp.array(states, dtype=wp.vec2d, device=warp_device),
            densities,
            scores,
        ],
        device=warp_device,
    )
    actual_densities, actual_scores = densities.numpy(), scores.numpy()
    assert np.isfinite(actual_densities).all() and (actual_densities > 0).all()
    assert np.isfinite(actual_scores).all()
    for index, (cosine, outgoing) in enumerate(states):
        expected_pdf, expected_scores = original_tail_joint(
            energy, cosine, outgoing, binding, height
        )
        assert actual_densities[index, 0] == pytest.approx(
            original_tail_phase_pdf(energy, cosine, binding, height), rel=3e-10, abs=0
        )
        assert actual_densities[index, 1] == pytest.approx(expected_pdf, rel=3e-9, abs=0)
        assert tuple(actual_scores[index]) == pytest.approx(expected_scores, rel=3e-12, abs=0)


@pytest.mark.parametrize("energy", [1.01, math.nextafter(1.0, math.inf)])
def test_native_tail_angle_proposal_matches_original_uniform_marks(
    warp_device: str, energy: float
) -> None:
    uniforms = (1e-12, 0.125, 0.5, 0.9, 1 - 1e-12)
    result = wp.empty(len(uniforms), dtype=wp.vec2d, device=warp_device)
    wp.launch(
        _angle_proposals,
        dim=len(uniforms),
        inputs=[
            energy,
            _parameters(warp_device),
            wp.array(uniforms, dtype=wp.float64, device=warp_device),
            result,
        ],
        device=warp_device,
    )
    for (cosine, acceptance), uniform in zip(result.numpy(), uniforms, strict=True):
        expected_cosine, expected_acceptance = original_tail_angle_proposal(
            energy, uniform, 1.0, 1e6
        )
        assert cosine == pytest.approx(expected_cosine, rel=0, abs=math.ulp(expected_cosine))
        assert 0 < acceptance <= 1
        assert acceptance == pytest.approx(expected_acceptance, rel=3e-11, abs=0)


@pytest.mark.parametrize("next_event", [0, 1])
@pytest.mark.parametrize("energy", [1.01, math.nextafter(1.0, math.inf)])
def test_production_native_conditional_energy_keeps_marks_binding_and_ledger(
    warp_device: str, next_event: int, energy: float
) -> None:
    count, binding, height = 128, 1.0, 1e6
    reference = original_tail_normalisation(energy, binding, height)
    cosine = -1 + 0.75 / float(reference.proposal_rate)
    result = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    marks = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _conditional_events,
        dim=count,
        inputs=[energy, cosine, _parameters(warp_device), next_event, result, marks],
        device=warp_device,
    )
    values, uniforms = result.numpy(), marks.numpy()
    assert np.all(values[:, 3] == 0)
    assert np.all(values[:, 1] == binding)
    assert np.all((values[:, 0] > 0) & (values[:, 0] <= energy - binding))
    assert len(set(values[:, 0])) > 16
    recoil = energy - values[:, 0] - binding
    assert np.all(recoil >= 0)
    np.testing.assert_allclose(
        values[:, 0] + binding + recoil, energy, rtol=0, atol=math.ulp(energy)
    )
    for index in range(16):
        expected_outgoing, expected_momentum = original_tail_conditional_energy(
            energy, cosine, float(uniforms[index, 1]), binding, height
        )
        assert values[index, 0] == pytest.approx(
            expected_outgoing, rel=0, abs=3 * math.ulp(expected_outgoing)
        )
        assert values[index, 2] == pytest.approx(
            expected_momentum, rel=0, abs=2 * math.ulp(expected_momentum)
        )


def test_production_native_tail_scattering_completes_with_original_counter_addresses(
    warp_device: str,
) -> None:
    count, energy, binding, height = 512, 1.01, 1.0, 1e6
    parameters = _parameters(warp_device)
    result = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _scattering_events,
        dim=count,
        inputs=[energy, parameters, result],
        device=warp_device,
    )
    checked = 32
    angle_marks = wp.empty(checked * MAX_TRIALS, dtype=wp.vec4d, device=warp_device)
    energy_marks = wp.empty(checked, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _scattering_marks,
        dim=checked * MAX_TRIALS,
        inputs=[parameters, angle_marks, energy_marks],
        device=warp_device,
    )
    values = result.numpy()
    assert np.all(values[:, 3] == 0)
    assert np.all((-1 <= values[:, 0]) & (values[:, 0] < 1))
    assert np.all((values[:, 2] > 0) & (values[:, 2] <= energy - binding))
    assert len(set(values[:, 0])) > 32 and len(set(values[:, 2])) > 32
    angle_uniforms = angle_marks.numpy().reshape(checked, MAX_TRIALS, 4)
    energy_uniforms = energy_marks.numpy()
    for index in range(checked):
        completed = False
        for trial in range(MAX_TRIALS):
            marks = angle_uniforms[index, trial]
            cosine, acceptance = original_tail_angle_proposal(
                energy, float(marks[0]), binding, height
            )
            if marks[1] < acceptance:
                expected_energy, _ = original_tail_conditional_energy(
                    energy, cosine, float(energy_uniforms[index, 1]), binding, height
                )
                assert values[index, 0] == pytest.approx(cosine, rel=0, abs=math.ulp(cosine))
                assert values[index, 1] == pytest.approx(2 * math.pi * marks[2], rel=2e-15, abs=0)
                assert values[index, 2] == pytest.approx(
                    expected_energy, rel=0, abs=3 * math.ulp(expected_energy)
                )
                completed = True
                break
        assert completed
    reference = original_tail_normalisation(energy, binding, height)
    scaled_angles = (values[:, 0] + 1) * float(reference.proposal_rate)
    for limit in (0.5, 1.0, 2.0):
        probability = original_tail_angle_cdf(energy, limit, binding, height)
        expected_count = count * probability
        sigma = math.sqrt(count * probability * (1 - probability))
        assert abs(np.count_nonzero(scaled_angles <= limit) - expected_count) < 6 * sigma + 1


def test_native_near_equal_extreme_shell_weights_keep_the_physical_mixture(
    warp_device: str,
) -> None:
    energy = 1.01
    parameters, shells = _mixture_parameters(warp_device)
    normalisation = wp.empty(1, dtype=wp.vec4d, device=warp_device)
    wp.launch(_normalisation, dim=1, inputs=[energy, parameters, normalisation], device=warp_device)
    reference_index, value, derivative, _ = normalisation.numpy()[0]
    expected = original_mixture_normalisation(energy, shells, int(reference_index))
    assert value == pytest.approx(float(expected.scaled_value), rel=3e-11, abs=0)
    assert derivative == pytest.approx(float(expected.centred_scaled_derivative), rel=3e-10, abs=0)
    states = extreme_tail_states()[:3]
    densities = wp.empty(len(states), dtype=wp.vec2d, device=warp_device)
    scores = wp.empty(len(states), dtype=wp.vec3d, device=warp_device)
    wp.launch(
        _pdf_scores,
        dim=len(states),
        inputs=[
            energy,
            parameters,
            wp.array(states, dtype=wp.vec2d, device=warp_device),
            densities,
            scores,
        ],
        device=warp_device,
    )
    actual_densities, actual_scores = densities.numpy(), scores.numpy()
    for index, (cosine, outgoing) in enumerate(states):
        expected_pdf, expected_scores = original_mixture_joint(
            energy, cosine, outgoing, shells, int(reference_index)
        )
        assert actual_densities[index, 1] == pytest.approx(expected_pdf, rel=3e-9, abs=0)
        assert tuple(actual_scores[index]) == pytest.approx(expected_scores, rel=3e-12, abs=0)
    count, cosine = 32, states[0][0]
    result = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    marks = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _conditional_events,
        dim=count,
        inputs=[energy, cosine, parameters, 0, result, marks],
        device=warp_device,
    )
    values, uniforms = result.numpy(), marks.numpy()
    assert np.all(values[:, 3] == 0)
    for index in range(count):
        expected_outgoing, expected_index, expected_momentum = original_mixture_conditional_energy(
            energy, cosine, float(uniforms[index, 0]), float(uniforms[index, 1]), shells
        )
        assert values[index, 1] == shells[expected_index][1]
        assert values[index, 0] == pytest.approx(
            expected_outgoing, rel=0, abs=3 * math.ulp(expected_outgoing)
        )
        assert values[index, 2] == pytest.approx(
            expected_momentum, rel=0, abs=2 * math.ulp(expected_momentum)
        )


def test_production_native_inaccessible_first_shell_keeps_accessible_tail_events(
    warp_device: str,
) -> None:
    energy, binding, height = 1.01, 1.0, 1e6
    parameters = physical.PhysicalParameters()
    parameters.compton_shell_offsets = wp.array([0, 2], dtype=wp.int32, device=warp_device)
    parameters.compton_occupancy = wp.array([1.0, 1.0], dtype=wp.float64, device=warp_device)
    parameters.compton_binding = wp.array([2.0, binding], dtype=wp.float64, device=warp_device)
    parameters.compton_height = wp.array([height, height], dtype=wp.float64, device=warp_device)
    normalisation = wp.empty(1, dtype=wp.vec4d, device=warp_device)
    wp.launch(_normalisation, dim=1, inputs=[energy, parameters, normalisation], device=warp_device)
    reference_index, value, _, _ = normalisation.numpy()[0]
    reference = original_tail_normalisation(energy, binding, height)
    assert reference_index == 1
    assert value == pytest.approx(float(reference.scaled_value), rel=3e-11, abs=0)
    states = extreme_tail_states()[:3]
    densities = wp.empty(len(states), dtype=wp.vec2d, device=warp_device)
    scores = wp.empty(len(states), dtype=wp.vec3d, device=warp_device)
    wp.launch(
        _pdf_scores,
        dim=len(states),
        inputs=[
            energy,
            parameters,
            wp.array(states, dtype=wp.vec2d, device=warp_device),
            densities,
            scores,
        ],
        device=warp_device,
    )
    actual_densities, actual_scores = densities.numpy(), scores.numpy()
    for index, (cosine, outgoing) in enumerate(states):
        expected_pdf, expected_scores = original_tail_joint(
            energy, cosine, outgoing, binding, height
        )
        assert actual_densities[index, 0] == pytest.approx(
            original_tail_phase_pdf(energy, cosine, binding, height), rel=3e-10, abs=0
        )
        assert actual_densities[index, 1] == pytest.approx(expected_pdf, rel=3e-9, abs=0)
        assert tuple(actual_scores[index]) == pytest.approx(expected_scores, rel=3e-12, abs=0)
    count, cosine = 16, states[0][0]
    conditional = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    marks = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _conditional_events,
        dim=count,
        inputs=[energy, cosine, parameters, 0, conditional, marks],
        device=warp_device,
    )
    values, uniforms = conditional.numpy(), marks.numpy()
    assert np.all(values[:, 3] == 0) and np.all(values[:, 1] == binding)
    for index in range(count):
        expected_outgoing, expected_momentum = original_tail_conditional_energy(
            energy, cosine, float(uniforms[index, 1]), binding, height
        )
        assert values[index, 0] == pytest.approx(
            expected_outgoing, rel=0, abs=3 * math.ulp(expected_outgoing)
        )
        assert values[index, 2] == pytest.approx(
            expected_momentum, rel=0, abs=2 * math.ulp(expected_momentum)
        )
    scattering = wp.empty(32, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _scattering_events,
        dim=32,
        inputs=[energy, parameters, scattering],
        device=warp_device,
    )
    events = scattering.numpy()
    assert np.all(events[:, 3] == 0)
    assert np.all((-1 <= events[:, 0]) & (events[:, 0] < 1))
    assert np.all((events[:, 2] > 0) & (events[:, 2] <= energy - binding))


@pytest.mark.parametrize("scaled_momentum", [18.69, 17.44, 17.46])
def test_native_near_underflow_dispatch_keeps_law_across_mass_switch(
    warp_device: str, scaled_momentum: float
) -> None:
    energy, height, mass = 100.0, 1e6, 510.99895069
    a = scaled_momentum / height
    binding = (2 * energy**2 / mass + 2 * energy * a) / (1 + 2 * energy / mass + a)
    parameters = _parameters(warp_device, height, binding)
    raw_normalisation = wp.empty(1, dtype=wp.vec2d, device=warp_device)
    wp.launch(
        _ordinary_normalisation,
        dim=1,
        inputs=[energy, parameters, raw_normalisation],
        device=warp_device,
    )
    if scaled_momentum == 18.69:
        assert raw_normalisation.numpy()[0, 0] == 0
    else:
        assert raw_normalisation.numpy()[0, 0] > 0
    normalisation = wp.empty(1, dtype=wp.vec4d, device=warp_device)
    wp.launch(_normalisation, dim=1, inputs=[energy, parameters, normalisation], device=warp_device)
    reference = original_tail_normalisation(energy, binding, height)
    _, value, derivative, _ = normalisation.numpy()[0]
    assert value == pytest.approx(float(reference.scaled_value), rel=3e-11, abs=0)
    assert derivative == pytest.approx(float(reference.centred_scaled_derivative), rel=3e-10, abs=0)
    states = extreme_tail_states(energy, binding, height)[:3]
    densities = wp.empty(len(states), dtype=wp.vec2d, device=warp_device)
    scores = wp.empty(len(states), dtype=wp.vec3d, device=warp_device)
    wp.launch(
        _pdf_scores,
        dim=len(states),
        inputs=[
            energy,
            parameters,
            wp.array(states, dtype=wp.vec2d, device=warp_device),
            densities,
            scores,
        ],
        device=warp_device,
    )
    actual_pdf, actual_scores = densities.numpy(), scores.numpy()
    for index, (cosine, outgoing) in enumerate(states):
        pdf, score = original_tail_joint(energy, cosine, outgoing, binding, height)
        assert actual_pdf[index, 0] == pytest.approx(
            original_tail_phase_pdf(energy, cosine, binding, height), rel=3e-10, abs=0
        )
        assert actual_pdf[index, 1] == pytest.approx(pdf, rel=3e-9, abs=0)
        assert tuple(actual_scores[index]) == pytest.approx(score, rel=3e-12, abs=0)
    proposals = wp.empty(3, dtype=wp.vec2d, device=warp_device)
    uniforms = (0.125, 0.5, 0.875)
    wp.launch(
        _angle_proposals,
        dim=3,
        inputs=[
            energy,
            parameters,
            wp.array(uniforms, dtype=wp.float64, device=warp_device),
            proposals,
        ],
        device=warp_device,
    )
    for actual, uniform in zip(proposals.numpy(), uniforms, strict=True):
        cosine, acceptance = original_tail_angle_proposal(energy, uniform, binding, height)
        assert actual[0] == pytest.approx(cosine, rel=0, abs=math.ulp(cosine))
        assert actual[1] == pytest.approx(acceptance, rel=3e-11, abs=0)
    count, cosine = 16, states[0][0]
    events = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    marks = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _conditional_events,
        dim=count,
        inputs=[energy, cosine, parameters, 0, events, marks],
        device=warp_device,
    )
    values, random = events.numpy(), marks.numpy()
    assert np.all(values[:, 3] == 0)
    for index in range(count):
        out, momentum = original_tail_conditional_energy(
            energy, cosine, float(random[index, 1]), binding, height
        )
        assert values[index, 0] == pytest.approx(out, rel=0, abs=3 * math.ulp(out))
        assert values[index, 2] == pytest.approx(momentum, rel=0, abs=2 * math.ulp(momentum))
    if scaled_momentum > 17.45:
        scattered = wp.empty(32, dtype=wp.vec4d, device=warp_device)
        wp.launch(
            _scattering_events,
            dim=32,
            inputs=[energy, parameters, scattered],
            device=warp_device,
        )
        assert np.all(scattered.numpy()[:, 3] == 0)


def test_native_positive_endpoint_tiny_height_keeps_ordinary_law(warp_device: str) -> None:
    energy, height, cosine, outgoing = 3.0, 1e-300, 0.0, 1.2
    parameters = _parameters(warp_device, height, 0.0)
    normalisation = wp.empty(1, dtype=wp.vec2d, device=warp_device)
    wp.launch(
        _ordinary_normalisation,
        dim=1,
        inputs=[energy, parameters, normalisation],
        device=warp_device,
    )
    value, derivative = tiny_height_thomson_pair(energy)
    actual = normalisation.numpy()[0]
    assert actual[0] / height == pytest.approx(value, rel=3e-11, abs=0)
    assert actual[1] / height == pytest.approx(derivative, rel=3e-10, abs=0)
    densities = wp.empty(1, dtype=wp.vec2d, device=warp_device)
    scores = wp.empty(1, dtype=wp.vec3d, device=warp_device)
    wp.launch(
        _pdf_scores,
        dim=1,
        inputs=[
            energy,
            parameters,
            wp.array([(cosine, outgoing)], dtype=wp.vec2d, device=warp_device),
            densities,
            scores,
        ],
        device=warp_device,
    )
    ratio = outgoing / energy
    factor = 1 + energy / 510.99895069 * (1 - cosine)
    q2 = (1 - ratio) ** 2 + 2 * ratio * (1 - cosine)
    jacobian = (factor - cosine + ratio * (1 - factor * cosine)) / (energy * q2**1.5)
    kn = factor**-3 + factor**-1 - (1 - cosine**2) * factor**-2
    assert densities.numpy()[0, 1] == pytest.approx(kn * jacobian / value, rel=3e-9, abs=0)
    assert np.isfinite(scores.numpy()).all()
