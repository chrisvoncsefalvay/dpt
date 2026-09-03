"""Physical shell eligibility precedes only the selected incoherent bound law."""

from dataclasses import replace

import pytest

from dpt.transport.bound_compton import ComptonShell
from dpt.transport.geometry_tangents import RayTangent
from dpt.transport.model import TransportError
from dpt.transport.physical_geometry_events import (
    AddressedUniform,
    PhysicalEventDraws,
    conditional_physical_event,
)
from helpers.geometry_bound_cases import model


def supplied_draws(*entries):
    return PhysicalEventDraws(
        0,
        0,
        "bound-compton-v3",
        tuple(AddressedUniform(0, domain, lane, value) for domain, lane, value in entries),
        "Explicit eligibility controls; unused angular and momentum coordinates absent",
    )


@pytest.mark.parametrize("energy", (0.5, 1.0))
def test_selected_bound_law_without_accessible_shell_has_physical_category(energy):
    spec, medium = model()
    with pytest.raises(TransportError, match="no accessible shell envelope at the reached energy"):
        conditional_physical_event(
            RayTangent((0.5, 0.5, 0.5), (0, 0, 1)),
            energy,
            0,
            spec,
            medium.absorption_mm_inverse,
            medium.scattering_mm_inverse,
            medium.physical,
            supplied_draws((0, 1, 0.9), (3, 0, 0.5)),
        )


@pytest.mark.parametrize("energy", (0.5, 1.0))
def test_absorption_bypasses_inaccessible_bound_shells(energy):
    spec, medium = model()
    result = conditional_physical_event(
        RayTangent((0.5, 0.5, 0.5), (0, 0, 1)),
        energy,
        0,
        spec,
        medium.absorption_mm_inverse,
        medium.scattering_mm_inverse,
        medium.physical,
        supplied_draws((0, 1, 0.1), (2, 0, 0.9)),
    )
    assert result.channel == "absorption" and result.status == "absorbed"
    assert result.deposited_kev == energy
    assert {(draw.domain, draw.lane) for draw in result.used_uniforms} == {(0, 1), (2, 0)}


@pytest.mark.parametrize("energy", (0.5, 1.0))
def test_rayleigh_bypasses_inaccessible_bound_shells(energy):
    spec, medium = model()
    assert medium.physical is not None
    physical = replace(medium.physical, rayleigh_mm_inverse=medium.scattering_mm_inverse)
    result = conditional_physical_event(
        RayTangent((0.5, 0.5, 0.5), (0, 0, 1)),
        energy,
        0,
        spec,
        medium.absorption_mm_inverse,
        medium.scattering_mm_inverse,
        physical,
        supplied_draws(
            (0, 1, 0.9),
            (3, 0, 0.5),
            (0xC0000000, 0, 0.5),
            (0xC0000000, 1, 0.001),
            (0xC0000000, 2, 0.25),
        ),
    )
    assert result.channel == "rayleigh" and result.status == "continuing"
    assert result.energy_kev == energy and result.outgoing is not None
    assert {(draw.domain, draw.lane) for draw in result.used_uniforms} == {
        (0, 1),
        (3, 0),
        (0xC0000000, 0),
        (0xC0000000, 1),
        (0xC0000000, 2),
    }


def test_zero_occupancy_shell_is_rejected_by_the_public_record():
    with pytest.raises(TransportError, match=r"shell occupancy.*positive"):
        ComptonShell("unoccupied shell", 0.0, 0.0, 0.1)
