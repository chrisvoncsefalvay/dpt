"""Independent terminal geometric/physical formulas and gauge composition."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import pytest

from dpt.materials import Provenance
from dpt.transport.bound_compton import BoundComptonTable, ComptonShell
from dpt.transport.geometry_histories import GeometryPhysicalMedium, PhysicalGeometryPrefix
from dpt.transport.geometry_tangents import RayTangent, RigidMotion
from dpt.transport.geometry_terminal import relative_rigid_motion, terminal_geometry_observation
from dpt.transport.model import (
    ELECTRON_REST_ENERGY_KEV,
    MaterialGrid,
    PlanarDetector,
    TransportSpec,
)
from dpt.transport.physical import AtomicScatteringTable, FluorescenceLine, PhysicalTransport
from dpt.transport.uncollided_geometry import AcquisitionGeometryTangent


def _scene():
    spec = TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((0.0, 0.0), (1.0, 1.0), (1, 1), 2.0),
        80.0,
        "terminal reference mu=1, isotropic",
        max_angle_trials=3,
    )
    medium = GeometryPhysicalMedium((0,), (0.0,), (1.0,), (1.0,), (1.0,))
    prefix = PhysicalGeometryPrefix(RayTangent((0.4, 0.6, 0.5), (0.0, 0.0, 1.0)), 80.0, 0, 0, 0)
    return spec, medium, prefix


def _independent_q(u, v):
    # Complete five-component unit-cube grid-plane mixture, derived by
    # integrating 1/(delta+|coordinate-face|) in the chosen coordinate.
    result = 0.5
    planes = ((0.4, u, 0.0), (0.6, u, 1.0), (0.6, v, 0.0), (0.4, v, 1.0))
    for delta, amount, face in planes:

        def primitive(t, delta=delta):
            return math.copysign(math.log1p(abs(t) / delta), t) if t else 0.0

        normal = primitive(1.0 - face) - primitive(-face)
        result += (0.5 / 4.0) / (normal * (delta + abs(amount - face)))
    return result


@pytest.mark.parametrize("score", ("photon-count", "energy-kev"))
def test_isotropic_normal_translation_matches_independent_closed_terminal_formula(score):
    spec, medium, prefix = _scene()
    spec = replace(spec, scoring=score)
    tangent = AcquisitionGeometryTangent(detector=RigidMotion(translation=(0.0, 0.0, 1.0)))
    value = terminal_geometry_observation(
        prefix,
        spec,
        medium,
        tangent,
        RigidMotion(),
        (-2.0,),
        detector_component=0.25,
        detector_coordinate=(0.3, 0.7),
    )
    h = 1.5
    transverse2 = 0.02
    r = math.sqrt(h * h + transverse2)
    density = 1.0 / (4.0 * math.pi)
    q = _independent_q(0.3, 0.7)
    expected = -2.0 * density * h / r**3 * math.exp(-0.5 * r / h) / q
    explicit = 80.0 if score == "energy-kev" else 1.0
    derivative = expected * (1.0 / h - 3.0 * h / r**2 - 0.5 * (1.0 / r - r / h**2))
    assert value.value == pytest.approx(expected * explicit, rel=3e-14)
    assert value.derivative == pytest.approx(derivative * explicit, rel=3e-14)
    assert value.cutoff_boundary == 0.0 and value.outgoing_channels == 1


def test_every_terminal_component_obeys_common_rigid_gauge_invariance():
    spec, medium, prefix = _scene()
    gauge = RigidMotion((0.3, -0.4, 0.5), (0.2, 0.1, -0.3), (4.0, 5.0, 6.0))
    result = terminal_geometry_observation(
        prefix,
        spec,
        medium,
        AcquisitionGeometryTangent(detector=gauge),
        gauge,
        (1.0,),
        detector_component=0.25,
        detector_coordinate=(0.3, 0.7),
    )
    assert result.derivative == pytest.approx(0.0, abs=1e-15)
    assert result.value > 0.0
    other = RigidMotion((-0.2, 0.8, 0.4), (-0.1, 0.2, 0.3), (-1.0, 2.0, 3.0))
    relative = relative_rigid_motion(other, gauge)
    point = (0.7, -0.3, 0.8)
    assert relative.point_velocity(point) == pytest.approx(
        tuple(
            a - b
            for a, b in zip(other.point_velocity(point), gauge.point_velocity(point), strict=True)
        )
    )


def test_fixed_energy_cutoff_endpoint_has_empty_measure_and_zero_geometry_derivative():
    spec, medium, prefix = _scene()
    spec = replace(
        spec, scattering_law="free-electron-compton", coefficient_energies_kev=(0.1, 200.0)
    )
    provenance = Provenance(
        "cutoff equal to fixed incident energy",
        hashlib.sha256(b"F1-r0-fixed-cutoff80").hexdigest(),
        "CC0-1.0",
        "Empty angular cap",
    )
    atomic = AtomicScatteringTable(1, (0.0, 20.0), (1.0, 1.0), (0.0, 1.0), provenance)
    medium = replace(
        medium,
        absorption_mm_inverse=(0.0, 0.0),
        scattering_mm_inverse=(1.0, 1.0),
        physical=PhysicalTransport((atomic,), (0.0, 0.0), cutoff_kev=80.0),
    )
    result = terminal_geometry_observation(
        prefix,
        spec,
        medium,
        AcquisitionGeometryTangent(detector=RigidMotion((1, 0, 0))),
        RigidMotion(),
        (1.0,),
        detector_component=0.25,
        detector_coordinate=(0.3, 0.7),
        cutoff_azimuth=0.37,
    )
    assert result.value == result.derivative == result.cutoff_boundary == 0.0


def test_mixed_rayleigh_and_fluorescence_use_all_actual_fixed_energy_channels():
    spec, medium, prefix = _scene()
    spec = replace(spec, coefficient_energies_kev=(10.0, 80.0))
    provenance = Provenance(
        "F(q)=1; r=.2,a=.6,s=.4; line10p=.5",
        hashlib.sha256(b"F1-r.2-a.6-s.4-line10p.5").hexdigest(),
        "CC0-1.0",
        "Declared analytical mixed law",
    )
    atomic = AtomicScatteringTable(1, (0.0, 10.0), (1.0, 1.0), (0.0, 1.0), provenance)
    line = FluorescenceLine(10.0, 20.0, (0.5, 0.5), provenance)
    physical = PhysicalTransport((atomic,), (0.2, 0.2), fluorescence=((line,),), cutoff_kev=0.1)
    medium = replace(
        medium,
        absorption_mm_inverse=(0.6, 0.6),
        scattering_mm_inverse=(0.4, 0.4),
        physical=physical,
    )
    result = terminal_geometry_observation(
        prefix,
        spec,
        medium,
        AcquisitionGeometryTangent(),
        RigidMotion(),
        (1.0,),
        detector_component=0.25,
        detector_coordinate=(0.3, 0.7),
    )
    h = 1.5
    r = math.sqrt(h * h + 0.02)
    c = h / r
    density = 0.5 / (4.0 * math.pi) + 0.2 * 3.0 / (16.0 * math.pi) * (1.0 + c * c)
    expected = density * h / r**3 * math.exp(-0.5 * r / h) / _independent_q(0.3, 0.7)
    assert result.value == pytest.approx(expected, rel=3e-14)
    assert result.derivative == 0.0 and result.outgoing_channels == 3


def test_free_compton_energy_and_phase_jet_match_independent_kn_formula():
    spec, medium, prefix = _scene()
    spec = replace(
        spec,
        coefficient_energies_kev=(0.1, 200.0),
        scattering_law="free-electron-compton",
        scoring="energy-kev",
    )
    medium = replace(medium, absorption_mm_inverse=(0.0, 0.0), scattering_mm_inverse=(1.0, 1.0))
    result = terminal_geometry_observation(
        prefix,
        spec,
        medium,
        AcquisitionGeometryTangent(detector=RigidMotion(translation=(0.0, 0.0, 1.0))),
        RigidMotion(),
        (1.0,),
        detector_component=0.25,
        detector_coordinate=(0.3, 0.7),
    )
    h = 1.5
    r = math.sqrt(h * h + 0.02)
    c = h / r
    cdot = 0.02 / r**3
    alpha = 80.0 / ELECTRON_REST_ENERGY_KEV
    ratio = 1.0 / (1.0 + alpha * (1.0 - c))
    rdot = alpha * ratio * ratio
    shape = ratio**3 + ratio - ratio * ratio + ratio * ratio * c * c
    slope = (
        rdot * (3.0 * ratio * ratio + 1.0 - 2.0 * ratio + 2.0 * ratio * c * c)
        + 2.0 * c * ratio * ratio
    )
    logarithm = math.log1p(2.0 * alpha)
    integral = 2.0 * (
        (1.0 + alpha) / alpha**2 * (2.0 * (1.0 + alpha) / (1.0 + 2.0 * alpha) - logarithm / alpha)
        + logarithm / (2.0 * alpha)
        - (1.0 + 3.0 * alpha) / (1.0 + 2.0 * alpha) ** 2
    )
    density = shape / (2.0 * math.pi * integral)
    energy = 80.0 * ratio
    expected = density * energy * h / r**3 * math.exp(-0.5 * r / h) / _independent_q(0.3, 0.7)
    expected_derivative = expected * (
        slope / shape * cdot
        + alpha * ratio * cdot
        + 1.0 / h
        - 3.0 * h / r**2
        - 0.5 * (1.0 / r - r / h**2)
    )
    assert result.value == pytest.approx(expected, rel=5e-12)
    assert result.derivative == pytest.approx(expected_derivative, rel=5e-12)


def _zero_binding_scene():
    spec, medium, prefix = _scene()
    spec = replace(
        spec, coefficient_energies_kev=(0.1, 80.0), scattering_law="bound-electron-compton"
    )
    provenance = Provenance(
        "Explicit zero-binding Hartree-Fock impulse fixture",
        hashlib.sha256(b"Z1-U0-profile.01-F1-mu1").hexdigest(),
        "CC0-1.0",
        "Analytical forward joint-law reference",
    )
    atom = AtomicScatteringTable(1, (0.0, 10.0), (1.0, 1.0), (0.0, 1.0), provenance)
    bound = BoundComptonTable(1, (ComptonShell("declared shell", 1.0, 0.0, 0.01),), provenance)
    physical = PhysicalTransport((atom,), (0.0, 0.0), cutoff_kev=0.1, bound_compton=(bound,))
    medium = replace(
        medium,
        absorption_mm_inverse=(0.0, 0.0),
        scattering_mm_inverse=(1.0, 1.0),
        physical=physical,
    )
    return spec, medium, prefix


def test_actual_zero_binding_terminal_importance_has_the_repaired_radial_power():
    spec, medium, prefix = _zero_binding_scene()
    scaled = []
    for theta in (1e-3, 1e-4, 1e-5):
        energy = 80.0 * (1.0 - 0.5 * theta)
        coordinate = (0.4 + 1.5 * math.tan(theta), 0.6)
        result = terminal_geometry_observation(
            prefix,
            spec,
            medium,
            AcquisitionGeometryTangent(detector=RigidMotion(translation=(1.0, 0.0, 0.0))),
            RigidMotion(),
            (1.0,),
            detector_component=0.25,
            detector_coordinate=coordinate,
            forward_component=0.25,
            bound_energy_uniform=(energy - 0.1) / (80.0 - 0.1),
        )
        assert math.isfinite(result.derivative) and result.derivative != 0.0
        scaled.append(result.derivative * math.sqrt(theta))
    # Joint point derivative grows theta^-2; divide by q_area~theta^-3/2
    # to get a per-draw gradient ~theta^-1/2. Its associated draw measure
    # across an energy band~theta makes the full second moment integrable.
    assert scaled[-1] / scaled[-2] == pytest.approx(1.0, rel=0.015)
    assert scaled[-2] / scaled[-3] == pytest.approx(1.0, rel=0.05)


def test_bound_forward_disc_miss_retains_zero_observation_without_rejection():
    spec, medium, prefix = _zero_binding_scene()
    result = terminal_geometry_observation(
        prefix,
        spec,
        medium,
        AcquisitionGeometryTangent(detector=RigidMotion(translation=(1.0, 0.0, 0.0))),
        RigidMotion(),
        (1.0,),
        detector_component=0.25,
        detector_coordinate=(0.9, 0.25),
        forward_component=0.75,
        bound_energy_uniform=0.5,
    )
    assert result.value == 0.0 and result.derivative == 0.0 and result.outgoing_channels == 0
    assert len(result.reference_density_identity) == 64
