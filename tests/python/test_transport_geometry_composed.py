"""Independent complete-generator, primary-boundary and lineage controls."""

import math
from dataclasses import replace
from typing import Literal

import pytest

from dpt.transport.geometry_composed import (
    composed_geometry_observation,
    material_patch_at,
    material_patch_count,
)
from dpt.transport.geometry_histories import GeometryPhysicalMedium
from dpt.transport.geometry_interfaces import material_interfaces
from dpt.transport.geometry_tangents import GridGeometryTangent, RigidMotion
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.source import ParallelBeam
from dpt.transport.uncollided_geometry import AcquisitionGeometryTangent


class ExplicitOriginalHistoryCoordinates:
    identity = "independent declared full-history integration fixture; no CI claim"
    layout: Literal["legacy-v1", "packed-compton-v2", "bound-compton-v3"] = "legacy-v1"

    def __init__(self, *, plane=0, source_plane=0, edge=0, physical=0.01):
        self.plane, self.source_plane, self.edge, self.physical = (
            plane,
            source_plane,
            edge,
            physical,
        )
        self.addresses = []

    def uniform(self, lineage, event, domain, lane):
        self.addresses.append((lineage, event, domain, lane))
        if domain == 0x30000003:
            return ((self.plane + 0.5) / 6, (self.source_plane + 0.5) / 6, 0.37)[lane]
        if domain == 0x30000004 and lane == 0:
            return (self.edge + 0.5) / 5
        if domain == 0 and lane == 0:
            return self.physical
        return 0.37


def _scene(*, absorption=0.7, detector=None):
    spec = TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
        detector or PlanarDetector((-5.0, -5.0), (10.0, 10.0), (1, 1), 3.0),
        80.0,
        "independent complete geometry controls",
        (80.0,),
        max_events=4,
        max_angle_trials=2,
    )
    medium = GeometryPhysicalMedium((0,), (absorption,), (0.0,), (1.0,), (1.0,))
    return spec, medium


def test_patch_index_decoder_matches_independently_enumerated_grid_without_allocation():
    spec, _ = _scene()
    spec = replace(spec, grid=MaterialGrid((-2, 3, -7), (0.4, 0.7, 2), (3, 2, 4), 1))
    expected = material_interfaces(spec)
    assert material_patch_count(spec) == len(expected)
    assert tuple(material_patch_at(i, spec) for i in range(len(expected))) == expected
    with pytest.raises(TransportError, match="complete measure"):
        material_patch_at(len(expected), spec)


@pytest.mark.parametrize("policy", ("analogue", "continuous-absorption"))
def test_full_generator_matches_absorbing_slab_thickness_without_pathwise_double_count(policy):
    spec, medium = _scene()
    spec = replace(spec, estimator=policy)
    source = ParallelBeam((-1.0, -1.0, -1.0), (3.0, 3.0))
    tangent = AcquisitionGeometryTangent(object=GridGeometryTangent(spacing_dot=(0, 0, 1)))
    observations = [
        composed_geometry_observation(
            source,
            (0.4, 0.45),
            80.0,
            spec,
            medium,
            ExplicitOriginalHistoryCoordinates(plane=plane),
            original_history=17,
            cotangent=(1.0,),
            tangent=tangent,
        )
        for plane in range(6)
    ]
    expected = -0.7 * math.exp(-0.7)
    assert math.fsum(o.derivative for o in observations) / 6 == pytest.approx(expected, abs=3e-15)
    assert all(o.mean == pytest.approx(math.exp(-0.7)) for o in observations)
    assert all(
        o.collided_generator == o.grazing_source == o.primary_detector == 0 for o in observations
    )
    assert sum(o.continuation_histories for o in observations) == 2


def test_whole_grazing_source_ray_jump_matches_absorbing_silhouette_expectation():
    spec, medium = _scene()
    source = ParallelBeam((-0.5, 0.1, -1.0), (1.1, 0.8))
    tangent = AcquisitionGeometryTangent(object=GridGeometryTangent(motion=RigidMotion((1, 0, 0))))
    count = 256
    values = [
        composed_geometry_observation(
            source,
            (0.1, 0.4),
            80.0,
            spec,
            medium,
            ExplicitOriginalHistoryCoordinates(source_plane=0, physical=(i + 0.5) / count),
            original_history=i,
            cotangent=(1.0,),
            tangent=tangent,
        )
        for i in range(count)
    ]
    # A single selected plane carries the inverse 1/6 selection probability.
    estimate = math.fsum(o.derivative for o in values) / (6 * count)
    exact = (1 - math.exp(-0.7)) / 1.1
    assert estimate == pytest.approx(exact, abs=0.5 / (count * 1.1))
    assert all(o.continuation_histories == 2 and o.physical_prefixes == 0 for o in values)
    assert all(o.ballistic_generator == o.collided_generator == 0 for o in values)


def test_signed_primary_pixel_jumps_match_independent_overlap_derivative():
    detector = PlanarDetector((-0.5, -1.0), (0.75, 3.0), (2, 1), 3.0)
    spec, medium = _scene(absorption=0.0, detector=detector)
    source = ParallelBeam((0.0, 0.2, -1.0), (2.0, 0.6))
    tangent = AcquisitionGeometryTangent(detector=RigidMotion((1, 0, 0)))
    observations = [
        composed_geometry_observation(
            source,
            (0.18, 0.37),
            80.0,
            spec,
            medium,
            ExplicitOriginalHistoryCoordinates(edge=edge),
            original_history=edge,
            cotangent=(-2.0, 3.0),
            tangent=tangent,
        )
        for edge in range(5)
    ]
    # The right detector edge adds seed3; the moving interior edge replaces3 by-2.
    assert math.fsum(o.derivative for o in observations) / 5 == pytest.approx(-1.0, abs=2e-15)
    assert all(o.derivative == o.primary_detector for o in observations)


def test_auxiliary_terms_preserve_original_history_and_common_continuation_lineage():
    spec, medium = _scene()
    source = ParallelBeam((-1.0, -1.0, -1.0), (3.0, 3.0))
    coordinates = ExplicitOriginalHistoryCoordinates(plane=5)
    result = composed_geometry_observation(
        source,
        (0.4, 0.45),
        80.0,
        spec,
        medium,
        coordinates,
        original_history=31,
        cotangent=(-2.0,),
        tangent=AcquisitionGeometryTangent(object=GridGeometryTangent(spacing_dot=(0, 0, 1))),
    )
    assert result.original_history == 31 and result.random_identity == coordinates.identity
    assert {address[0][0] for address in coordinates.addresses} == {31}
    assert {address[0] for address in coordinates.addresses if address[0][1] == 1} == {(31, 1, 0)}
    assert result.derivative == pytest.approx(12 * 0.7 * math.exp(-0.7))


def test_original_spectrum_mark_sets_primary_energy_score_and_pixel_boundary_jump():
    detector = PlanarDetector((-0.5, -1.0), (0.75, 3.0), (2, 1), 3.0)
    spec, medium = _scene(absorption=0.0, detector=detector)
    spec = replace(spec, coefficient_energies_kev=(40.0, 80.0), scoring="energy-kev")
    medium = replace(medium, absorption_mm_inverse=(0.0, 0.0), scattering_mm_inverse=(0.0, 0.0))
    source = ParallelBeam((0.0, 0.2, -1.0), (2.0, 0.6))
    observations = [
        composed_geometry_observation(
            source,
            (0.18, 0.37),
            40.0,
            spec,
            medium,
            ExplicitOriginalHistoryCoordinates(edge=edge),
            original_history=edge,
            cotangent=(-2.0, 3.0),
            tangent=AcquisitionGeometryTangent(detector=RigidMotion((1, 0, 0))),
        )
        for edge in range(5)
    ]
    assert spec.energy_kev == 80.0
    assert all(o.mean == pytest.approx(120.0, rel=3e-15) for o in observations)
    assert math.fsum(o.derivative for o in observations) / 5 == pytest.approx(-40.0, abs=5e-14)


def test_joint_active_material_silhouette_and_detector_edge_rejects_two_sided_kink():
    detector = PlanarDetector((0.0, -2.0), (2.0, 5.0), (1, 1), 3.0)
    spec, medium = _scene(detector=detector)
    source = ParallelBeam((-0.5, 0.1, -1.0), (1.1, 0.8))
    tangent = AcquisitionGeometryTangent(object=GridGeometryTangent(motion=RigidMotion((1, 0, 0))))
    # Independently, F(b)=[.6*T+(1-T)*max(b,0)]/1.1 locally. Its
    # left/right derivatives are0 and(1-T)/1.1; no two-sided JVP exists.
    coordinates = ExplicitOriginalHistoryCoordinates()
    with pytest.raises(TransportError, match="coincident active material/detector"):
        composed_geometry_observation(
            source,
            (0.1, 0.4),
            80.0,
            spec,
            medium,
            coordinates,
            original_history=19,
            cotangent=(1.0,),
            tangent=tangent,
        )
    assert not coordinates.addresses
