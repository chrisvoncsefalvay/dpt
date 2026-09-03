"""Complete physical continuations, exact initial sides and failure lineage."""

import hashlib
import math
from dataclasses import replace
from typing import Literal

import pytest

from dpt.materials import Provenance
from dpt.transport.geometry_histories import GeometryPhysicalMedium, complete_geometry_history
from dpt.transport.geometry_tangents import RayTangent
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.physical import AtomicScatteringTable, FluorescenceLine, PhysicalTransport


class DeclaredCoordinates:
    identity = "Explicit fixed coordinate fixture; no generator or QMC convergence claim"
    layout: Literal["legacy-v1", "packed-compton-v2", "bound-compton-v3"] = "legacy-v1"

    def uniform(self, lineage, event, domain, lane):
        if domain == 0:
            return (0.8, 0.1, 0.95, 0.3)[lane]
        if domain == 2:
            return (0.1, 0.95, 0.3)[lane]
        return 0.1


def _scene():
    spec = TransportSpec(
        MaterialGrid((0.0, -1.0, 0.0), (1.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-10.0, -10.0), (20.0, 20.0), (1, 1), 3.0),
        80.0,
        "whole actual fluorescence future",
        (10.0, 80.0),
        max_events=8,
        max_angle_trials=2,
    )
    provenance = Provenance(
        "declared analytical fixture",
        hashlib.sha256(b"mu(E)=0.7*(E-10)/70; line10keV; branch=0.5; F(q)=1").hexdigest(),
        "CC0-1.0",
        "Complete fluorescence continuation reference",
    )
    atom = AtomicScatteringTable(1, (0.0, 10.0), (1.0, 1.0), (0.0, 1.0), provenance)
    line = FluorescenceLine(10.0, 20.0, (0.5, 0.5), provenance)
    physical = PhysicalTransport((atom,), (0.0, 0.0), fluorescence=((line,),), cutoff_kev=0.1)
    medium = GeometryPhysicalMedium((0,), (0.0, 0.7), (0.0, 0.0), (1.0,), (1.0,), physical)
    return spec, medium


def test_extinction_exact_table_node_preserves_small_following_coefficient():
    spec, medium = _scene()
    medium = replace(medium, absorption_mm_inverse=(1e12, 0.1), scattering_mm_inverse=(1e12, 0.03))
    assert medium.extinction(80.0, spec) == (0.13,)


def test_unsupported_energy_missed_material_box_escapes_before_coefficient_lookup():
    spec, medium = _scene()
    result = complete_geometry_history(
        RayTangent((2.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        201.0,
        spec,
        medium,
        DeclaredCoordinates(),
        lineage=(12,),
        cotangent=(-2.0,),
    )
    assert result.response == -2.0 and result.terminal == "escaped" and not result.prefixes


@pytest.mark.parametrize("policy", ("analogue", "continuous-absorption"))
def test_exact_source_side_continuation_includes_actual_fluorescent_photon(policy):
    spec, medium = _scene()
    spec = replace(spec, estimator=policy, scoring="energy-kev")
    ray = RayTangent((0.0, 0.0, -1.0), (0.0, 0.0, 1.0))
    outside = complete_geometry_history(
        ray,
        80.0,
        spec,
        medium,
        DeclaredCoordinates(),
        lineage=(7, 4, 0),
        cotangent=(-2.0,),
        initial_material_side=(0, 0, -1),
    )
    inside = complete_geometry_history(
        ray,
        80.0,
        spec,
        medium,
        DeclaredCoordinates(),
        lineage=(7, 4, 1),
        cotangent=(-2.0,),
        initial_material_side=(0, 0, 1),
    )
    assert outside.response == -160.0 and outside.terminal == "escaped" and not outside.prefixes
    assert inside.response == -20.0 and inside.terminal == "escaped" and len(inside.prefixes) == 1
    assert inside.energy_kev == 10.0
    assert inside.prefixes[0].incoming.origin_mm[0] == 0.0
    assert inside.lineage == (7, 4, 1)
    assert any(u.domain == 2 for u in inside.used_uniforms)


def test_forced_material_collision_continues_complete_future_without_resampling_a_flight():
    spec, medium = _scene()
    result = complete_geometry_history(
        RayTangent((0.5, 0.0, 0.5), (0.0, 0.0, 1.0)),
        80.0,
        spec,
        medium,
        DeclaredCoordinates(),
        lineage=(8, 2, 3),
        cotangent=(1.0,),
        forced_collision_cell=0,
    )
    assert result.response == 1.0 and result.energy_kev == 10.0
    assert len(result.prefixes) == 1 and result.prefixes[0].incoming.origin_mm == (0.5, 0.0, 0.5)


def test_event_guard_is_not_a_zero_or_absorbed_future():
    spec, medium = _scene()
    spec = replace(spec, max_events=1)
    with pytest.raises(TransportError, match="event guard"):
        complete_geometry_history(
            RayTangent((0.5, 0.0, -1.0), (0.0, 0.0, 1.0)),
            80.0,
            spec,
            medium,
            DeclaredCoordinates(),
            lineage=(0,),
            cotangent=(1.0,),
        )


def test_energy_table_jet_is_independent_linear_rate_slope_and_rejects_node_kink():
    spec, medium = _scene()
    rates, dots = medium.extinction_jet(45.0, 2.0, spec)
    assert rates == pytest.approx((0.35,)) and dots == pytest.approx((0.02,))
    spec = replace(spec, coefficient_energies_kev=(10.0, 45.0, 80.0))
    medium = replace(
        medium, absorption_mm_inverse=(0.0, 0.2, 0.7), scattering_mm_inverse=(0.0, 0.0, 0.0)
    )
    with pytest.raises(TransportError, match="interpolation kink"):
        medium.extinction_jet(45.0, 1.0, spec)


def test_invalid_coordinates_and_coefficient_support_abort_whole_continuation():
    spec, medium = _scene()

    class BadCoordinates(DeclaredCoordinates):
        def uniform(self, lineage, event, domain, lane):
            return math.nan

    with pytest.raises(TransportError, match="coefficient support"):
        complete_geometry_history(
            RayTangent((0.5, 0.0, -1.0), (0.0, 0.0, 1.0)),
            5.0,
            spec,
            medium,
            DeclaredCoordinates(),
            lineage=(0,),
            cotangent=(1.0,),
        )

    with pytest.raises(TransportError, match="open-interval"):
        complete_geometry_history(
            RayTangent((0.5, 0.0, -1.0), (0.0, 0.0, 1.0)),
            80.0,
            spec,
            medium,
            BadCoordinates(),
            lineage=(0,),
            cotangent=(1.0,),
        )


@pytest.mark.parametrize("energy", (40.0, 50.0))
def test_cutoff_matches_canonical_material_entry_and_includes_equality(energy):
    spec, medium = _scene()
    assert medium.physical is not None
    medium = replace(medium, physical=replace(medium.physical, cutoff_kev=50.0))
    spec = replace(spec, scoring="energy-kev")
    inside = complete_geometry_history(
        RayTangent((0.5, 0.0, -1.0), (0.0, 0.0, 1.0)),
        energy,
        spec,
        medium,
        DeclaredCoordinates(),
        lineage=(4, 0, 0),
        cotangent=(-2.0,),
    )
    outside = complete_geometry_history(
        RayTangent((-0.5, 0.0, -1.0), (0.0, 0.0, 1.0)),
        energy,
        spec,
        medium,
        DeclaredCoordinates(),
        lineage=(4, 0, 0),
        cotangent=(-2.0,),
    )
    assert inside.terminal == "local-deposition" and inside.response == 0.0
    assert outside.terminal == "escaped" and outside.response == -2.0 * energy
    assert not inside.used_uniforms and not outside.used_uniforms


@pytest.mark.parametrize("side, expected", ((None, -80.0), ((0, 1, -1), 0.0), ((0, 1, 1), -80.0)))
def test_initial_cutoff_uses_half_open_upper_face_and_explicit_sides(side, expected):
    spec, medium = _scene()
    assert medium.physical is not None
    medium = replace(medium, physical=replace(medium.physical, cutoff_kev=50.0))
    result = complete_geometry_history(
        RayTangent((1.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        40.0,
        replace(spec, scoring="energy-kev"),
        medium,
        DeclaredCoordinates(),
        lineage=(6, 0, 0),
        cotangent=(-2.0,),
        initial_material_side=side,
    )
    assert result.response == expected


def test_finite_seed_product_overflow_invalidates_complete_escaped_history():
    spec, medium = _scene()
    with pytest.raises(TransportError, match="escaped geometry response"):
        complete_geometry_history(
            RayTangent((-0.5, 0.0, -1.0), (0.0, 0.0, 1.0)),
            80.0,
            replace(spec, scoring="energy-kev"),
            medium,
            DeclaredCoordinates(),
            lineage=(8, 0, 0),
            cotangent=(1e308,),
        )
