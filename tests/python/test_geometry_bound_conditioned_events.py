"""Addressed geometry collisions use the full conditioned bound law."""

import math

import pytest

from dpt.transport.bound_compton import (
    bound_compton_angle_proposal,
    bound_compton_conditional_energy,
    bound_compton_joint_pdf,
    bound_compton_joint_scores,
)
from dpt.transport.bound_compton_quadrature import bound_angular_normalisation
from dpt.transport.geometry_tangents import RayTangent
from dpt.transport.physical_geometry_events import (
    AddressedUniform,
    PhysicalEventDraws,
    conditional_physical_event,
    terminal_physical_kernels,
)
from helpers.geometry_conditioned_cases import CASES, TERMINAL_CASES, physical_case


@pytest.mark.parametrize("name,energy,entries", CASES)
def test_geometry_conditional_bound_event_keeps_shared_law_marks_and_rigid_jet(
    name, energy, entries
):
    spec, physical, table = physical_case(name, energy, entries)
    event, branch = 2, 123
    marks = [(0, 1, 0.5), (3, 0, 0.5), (0x10000000, 0, 0.5), (0x10000000, 1, 0.5)]
    marks += [
        (0x80000000 + trial, lane, value)
        for trial in range(3)
        for lane, value in enumerate((0.5, 1e-8, 0.25))
    ]
    draws = PhysicalEventDraws(
        event,
        branch,
        "bound-compton-v3",
        tuple(
            AddressedUniform(event, domain + branch * 65536, lane, value)
            for domain, lane, value in marks
        ),
        "fixed original domain/branch/event coordinates; no redraw",
    )
    incoming = RayTangent((0.2, 0.3, 0.4), (0, 0, 1), (0.1, 0.2, 0.3), (0.2, -0.3, 0))
    result = conditional_physical_event(incoming, energy, 0, spec, (0, 0), (1, 1), physical, draws)
    cosine, acceptance = bound_compton_angle_proposal(energy, table, 0.5)
    assert acceptance > 1e-8 and result.status == "continuing" and result.outgoing is not None
    expected_energy, shell, momentum = bound_compton_conditional_energy(
        energy, cosine, table, 0.5, 0.5
    )
    assert (
        result.energy_kev == expected_energy
        and result.shell_index == shell
        and result.momentum == momentum
    )
    assert result.outgoing.direction[2] == pytest.approx(cosine, abs=3e-15)
    omega = (0.3, 0.2, 0)
    direction = result.outgoing.direction
    exact_dot = (
        omega[1] * direction[2],
        -omega[0] * direction[2],
        omega[0] * direction[1] - omega[1] * direction[0],
    )
    assert result.outgoing.direction_dot == pytest.approx(exact_dot, abs=2e-15)
    assert (
        result.outgoing.origin_mm == incoming.origin_mm
        and result.outgoing.origin_dot == incoming.origin_dot
    )
    assert (
        math.isfinite(result.energy_kev)
        and 0 < result.energy_kev <= energy - table.shells[shell].binding_kev
    )
    used = {(item.domain - branch * 65536, item.lane) for item in result.used_uniforms}
    assert used == {
        (0, 1),
        (3, 0),
        (0x80000000, 0),
        (0x80000000, 1),
        (0x80000000, 2),
        (0x10000000, 0),
        (0x10000000, 1),
    }


@pytest.mark.parametrize("name,energy,entries", TERMINAL_CASES)
def test_host_bound_terminal_uses_matched_conditioned_density_and_fixed_energy_score(
    name, energy, entries
):
    spec, physical, table = physical_case(name, energy, entries, trials=64)
    cosine, _ = bound_compton_angle_proposal(energy, table, 0.5)
    outgoing, _, _ = bound_compton_conditional_energy(energy, cosine, table, 0.5, 0.5)
    assert outgoing > physical.cutoff_kev
    incident = RayTangent((0, 0, 0), (0, 0, 1), direction_dot=(0.2, -0.3, 0))
    endpoint = (math.sqrt((1 - cosine) * (1 + cosine)), 0.0, cosine)
    (kernel,) = terminal_physical_kernels(
        incident,
        endpoint,
        energy,
        0,
        spec,
        (0, 0),
        (1, 1),
        physical,
        bound_outgoing_kev=outgoing,
    )
    assert kernel.density > 0 and math.isfinite(kernel.density_dot)
    density = bound_compton_joint_pdf(energy, kernel.cosine, outgoing, table)
    score = bound_compton_joint_scores(energy, kernel.cosine, outgoing, table)[0]
    assert kernel.density == density
    assert kernel.density_dot == pytest.approx(density * score * kernel.cosine_dot, rel=2e-14)
    if name == "representable-negative-tail":
        assert 0 < bound_angular_normalisation(energy, table) < 1e-100
