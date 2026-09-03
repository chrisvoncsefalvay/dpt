"""Actual native custom-profile mass, sampler, normaliser and score regressions."""

# Warp annotations are runtime DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
import importlib
import math
from typing import Any

import numpy as np
import pytest

from custom_bound_reference import (
    decimal_interval_inverse,
    decimal_interval_mass,
    original_angle_narrow_normalisation,
    tiny_height_thomson_pair,
)

wp: Any = importlib.import_module("warp")
bound: Any = importlib.import_module("dpt.transport.bound_compton_kernels")
physical: Any = importlib.import_module("dpt.transport.physical_kernels")
sampling: Any = importlib.import_module("dpt.transport.sampling_kernels")


@wp.kernel(enable_backward=False)
def _intervals(states: wp.array(dtype=wp.vec4d), out: wp.array(dtype=wp.vec2d)):
    i = wp.tid()
    state = states[i]
    out[i] = wp.vec2d(
        bound.impulse_profile_interval_mass(state[0], state[1], state[2]),
        bound.impulse_profile_interval_inverse(state[3], state[0], state[1], state[2]),
    )


@wp.kernel(enable_backward=False)
def _normaliser(
    energy: wp.float64, parameters: physical.PhysicalParameters, out: wp.array(dtype=wp.vec3d)
):
    values = bound.impulse_angular_normalisation_pair(energy, 0, parameters)
    out[0] = wp.vec3d(
        values[0], values[1], bound.impulse_normalisation_energy_score(energy, 0, parameters)
    )


@wp.kernel(enable_backward=False)
def _events(
    energy: wp.float64,
    cosine: wp.float64,
    parameters: physical.PhysicalParameters,
    out: wp.array(dtype=wp.vec4d),
    draws: wp.array(dtype=wp.vec4d),
):
    i = wp.tid()
    out[i] = bound.bound_energy_event(
        energy, cosine, 0, wp.uint64(48217), wp.uint64(i), wp.uint32(0), 0, parameters
    )
    draws[i] = sampling.transport_random4(
        wp.uint64(48217), wp.uint64(i), wp.uint32(0), wp.uint32(0x10000000), parameters.sampling
    )


def _parameters(device: str, height: float, binding: float) -> Any:
    parameters = physical.PhysicalParameters()
    parameters.compton_shell_offsets = wp.array([0, 1], dtype=wp.int32, device=device)
    parameters.compton_occupancy = wp.array([1.0], dtype=wp.float64, device=device)
    parameters.compton_binding = wp.array([binding], dtype=wp.float64, device=device)
    parameters.compton_height = wp.array([height], dtype=wp.float64, device=device)
    return parameters


def test_native_interval_mass_and_quantiles_keep_tiny_and_narrow_profiles(warp_device: str) -> None:
    cases = [
        (-1.0, 0.0058708535427502425, 1e-30),
        (-1.0, -0.2, 1e-30),
        (0.1, 0.4, 1e-30),
        (-1.0, 0.3, 0.8),
        (-1.0, -0.2, 0.8),
        (0.1, 0.4, 0.8),
        (0.125, 0.125 + 2e-15, 0.8),
        (-1e-6, 1e-6, 1e6),
        (-1e-6, -2e-7, 1e6),
        (1e-7, 4e-7, 1e6),
    ]
    states = [(lo, hi, h, u) for lo, hi, h in cases for u in (1e-12, 0.1, 0.5, 0.9, 1 - 1e-12)]
    out = wp.empty(len(states), dtype=wp.vec2d, device=warp_device)
    wp.launch(
        _intervals,
        dim=len(states),
        inputs=[wp.array(states, dtype=wp.vec4d, device=warp_device), out],
        device=warp_device,
    )
    actual = out.numpy()
    for index, (lo, hi, height, probability) in enumerate(states):
        mass = decimal_interval_mass(lo, hi, height)
        momentum = decimal_interval_inverse(probability, lo, hi, height)
        assert actual[index, 0] == pytest.approx(mass, rel=7e-15, abs=0)
        assert lo <= actual[index, 1] <= hi
        assert actual[index, 1] == pytest.approx(momentum, rel=8e-15, abs=3e-15 * (hi - lo))


@pytest.mark.parametrize("energy,height,binding", [(3.0, 1e-30, 0.0), (100.0, 1e6, 1.0)])
def test_native_normaliser_and_incident_score_match_independent_original_angle_law(
    warp_device: str,
    energy: float,
    height: float,
    binding: float,
) -> None:
    out = wp.empty(1, dtype=wp.vec3d, device=warp_device)
    wp.launch(
        _normaliser,
        dim=1,
        inputs=[energy, _parameters(warp_device, height, binding), out],
        device=warp_device,
    )
    value, derivative, score = out.numpy()[0]
    if binding == 0:
        n, n_e = tiny_height_thomson_pair(energy)
        expected_value, expected_derivative = height * n, height * n_e
    else:
        expected_value = original_angle_narrow_normalisation(energy)
        step = 0.005
        references = [
            original_angle_narrow_normalisation(energy + offset * step) for offset in (-2, -1, 1, 2)
        ]
        expected_derivative = (
            references[0] - 8 * references[1] + 8 * references[2] - references[3]
        ) / (12 * step)
    assert value == pytest.approx(expected_value, rel=5e-13, abs=0)
    assert derivative == pytest.approx(expected_derivative, rel=2e-9, abs=0)
    assert score == pytest.approx(expected_derivative / expected_value, rel=2e-9, abs=0)


@pytest.mark.parametrize(
    "energy,cosine,height,binding",
    [(3.0, -1.0, 1e-30, 0.0), (100.0, 0.8, 1e6, 1.0), (80.0, 0.3, 0.8, 0.0)],
)
def test_production_conditional_energy_uses_original_marks_and_broadened_profile(
    warp_device: str,
    energy: float,
    cosine: float,
    height: float,
    binding: float,
) -> None:
    count = 128
    parameters = _parameters(warp_device, height, binding)
    out = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    draws = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _events, dim=count, inputs=[energy, cosine, parameters, out, draws], device=warp_device
    )
    actual, marks = out.numpy(), draws.numpy()
    upper = (energy * (energy - binding) * (1 - cosine) / 510.99895069 - binding) / math.sqrt(
        2 * energy * (energy - binding) * (1 - cosine) + binding**2
    )
    assert np.all(actual[:, 3] == 0)
    assert np.all((actual[:, 0] > 0) & (actual[:, 0] <= energy - binding))
    assert np.all(actual[:, 1] == binding)
    assert actual[:, 2].std() > 0
    for index in range(16):
        reference = decimal_interval_inverse(float(marks[index, 1]), -1.0, upper, height)
        assert actual[index, 2] == pytest.approx(
            reference, rel=8e-15, abs=3e-15 * max(1e-6, 1 + upper)
        )
    transfer = np.sqrt(energy**2 + actual[:, 0] ** 2 - 2 * energy * actual[:, 0] * cosine)
    residual = actual[:, 0] - energy + energy * actual[:, 0] * (1 - cosine) / 510.99895069
    np.testing.assert_allclose(residual, actual[:, 2] * transfer, rtol=3e-10, atol=3e-13)
