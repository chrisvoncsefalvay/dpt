"""Native bound-occupancy kernels and physical parameter fixtures."""

# Warp annotations are runtime DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
from __future__ import annotations

import importlib
from typing import Any

from bound_occupancy_reference import Shells

wp: Any = importlib.import_module("warp")
bound: Any = importlib.import_module("dpt.transport.bound_compton_kernels")
mixed: Any = importlib.import_module("dpt.transport.bound_compton_mixed_kernels")
physical: Any = importlib.import_module("dpt.transport.physical_kernels")
sampling: Any = importlib.import_module("dpt.transport.sampling_kernels")


@wp.kernel(enable_backward=False)
def _normalisation(
    energy: wp.float64,
    parameters: physical.PhysicalParameters,
    result: wp.array(dtype=wp.vec4d),
    raw: wp.array(dtype=wp.vec2d),
):
    result[0] = mixed.impulse_mixed_normalisation(energy, 0, parameters)
    raw[0] = bound.impulse_angular_normalisation_pair(energy, 0, parameters)


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
def _proposals(
    energy: wp.float64,
    parameters: physical.PhysicalParameters,
    uniforms: wp.array(dtype=wp.float64),
    result: wp.array(dtype=wp.vec2d),
):
    index = wp.tid()
    result[index] = mixed.impulse_mixed_angle_proposal(energy, 0, uniforms[index], parameters)


@wp.kernel(enable_backward=False)
def _energies(
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
def _scatter(
    energy: wp.float64, parameters: physical.PhysicalParameters, result: wp.array(dtype=wp.vec4d)
):
    index = wp.tid()
    result[index] = bound.bound_scatter_event(
        energy, 0, wp.uint64(48217), wp.uint64(index), wp.uint32(3), 16, parameters
    )


@wp.kernel(enable_backward=False)
def _scatter_marks(
    parameters: physical.PhysicalParameters,
    angle: wp.array(dtype=wp.vec4d),
    energy: wp.array(dtype=wp.vec4d),
):
    index = wp.tid()
    history, trial = index // 16, index % 16
    domain = (
        wp.uint32(0x80000000)
        + wp.uint32(parameters.random_branch) * wp.uint32(65536)
        + wp.uint32(trial)
    )
    angle[index] = sampling.transport_random4(
        wp.uint64(48217), wp.uint64(history), wp.uint32(3), domain, parameters.sampling
    )
    if trial == 0:
        energy_domain = wp.uint32(0x10000000) + wp.uint32(parameters.random_branch) * wp.uint32(
            65536
        )
        energy[history] = sampling.transport_random4(
            wp.uint64(48217), wp.uint64(history), wp.uint32(3), energy_domain, parameters.sampling
        )


def parameters(device: str, shells: Shells) -> Any:
    result = physical.PhysicalParameters()
    result.compton_shell_offsets = wp.array([0, len(shells)], dtype=wp.int32, device=device)
    result.compton_occupancy = wp.array([s[0] for s in shells], dtype=wp.float64, device=device)
    result.compton_binding = wp.array([s[1] for s in shells], dtype=wp.float64, device=device)
    result.compton_height = wp.array([s[2] for s in shells], dtype=wp.float64, device=device)
    return result
