"""Independent box-shadow area derivatives at canonical below/equal cutoff."""

import math
from dataclasses import replace
from typing import Literal

import pytest

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.materials import Provenance
from dpt.transport.geometry_composed import composed_geometry_observation
from dpt.transport.geometry_histories import GeometryPhysicalMedium
from dpt.transport.geometry_opaque_source import (
    opaque_boundary_sides,
    opaque_material_hit,
    opaque_source_boundaries,
)
from dpt.transport.geometry_tangents import GridGeometryTangent, RayTangent, RigidMotion
from dpt.transport.model import (
    MaterialGrid,
    OrientedDetector,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from dpt.transport.physical import AtomicScatteringTable, PhysicalTransport
from dpt.transport.source import ParallelBeam, PointConeSource
from dpt.transport.uncollided_geometry import AcquisitionGeometryTangent, SourceGeometryTangent


class OpaqueCoordinates:
    identity = "exact independent source-square box-shadow quadrature"
    layout: Literal["legacy-v1", "packed-compton-v2", "bound-compton-v3"] = "legacy-v1"

    def __init__(self, boundary=0, edge=0, line_coordinate=0.5):
        self.boundary, self.edge, self.line_coordinate = boundary, edge, line_coordinate
        self.draws = []

    def uniform(self, lineage, event, domain, lane):
        self.draws.append((lineage, event, domain, lane))
        if domain == 0x30000003:
            return (0.5, (self.boundary + 0.5) / 18, self.line_coordinate)[lane]
        if domain == 0x30000004 and lane == 0:
            return (self.edge + 0.5) / 4
        return 0.5


def _scene():
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 1), 1),
        PlanarDetector((-10, -10), (20, 20), (1, 1), 3),
        80,
        "independent canonical opaque-box geometry",
        (10, 80),
        max_events=4,
        max_angle_trials=2,
    )
    provenance = Provenance("analytic test", "0" * 64, "CC0-1.0", "constant atomic angular fixture")
    atom = AtomicScatteringTable(1, (0, 10), (1, 1), (0, 1), provenance)
    physical = PhysicalTransport((atom,), (0, 0), cutoff_kev=50)
    medium = GeometryPhysicalMedium((0,), (0, 0), (0, 0), (1,), (1,), physical)
    return spec, medium


@pytest.mark.parametrize("energy", (40.0, 50.0))
def test_subcutoff_mean_deposits_on_entry_and_escapes_on_miss_even_zero_coefficients(energy):
    spec, medium = _scene()
    source = ParallelBeam((-1, 0.1, -1), (3, 0.8))
    hit = composed_geometry_observation(
        source,
        (0.5, 0.5),
        energy,
        spec,
        medium,
        OpaqueCoordinates(),
        original_history=4,
        cotangent=(-2,),
        tangent=AcquisitionGeometryTangent(),
    )
    miss = composed_geometry_observation(
        source,
        (0.1, 0.5),
        energy,
        spec,
        medium,
        OpaqueCoordinates(),
        original_history=4,
        cotangent=(-2,),
        tangent=AcquisitionGeometryTangent(),
    )
    assert hit.mean == 0 and miss.mean == -2
    assert hit.physical_prefixes == miss.physical_prefixes == 0


@pytest.mark.parametrize("energy", (40.0, 50.0))
@pytest.mark.parametrize("active", ("object", "source"))
def test_parallel_partial_shadow_exact_object_and_source_translation(energy, active):
    spec, medium = _scene()
    source = ParallelBeam((-0.5, 0.1, -1), (1.1, 0.8), weight=2)
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=RigidMotion((1, 0, 0)))
        if active == "object"
        else GridGeometryTangent(),
        source=SourceGeometryTangent(motion=RigidMotion((1, 0, 0)))
        if active == "source"
        else SourceGeometryTangent(),
    )
    values = [
        composed_geometry_observation(
            source,
            (0.1, 0.4),
            energy,
            spec,
            medium,
            OpaqueCoordinates(boundary=i),
            original_history=i,
            cotangent=(-3,),
            tangent=tangent,
        )
        for i in range(18)
    ]
    # Intersection[0,.6] shrinks/grows by exactly1mm; source is1.1mm wide.
    expected = (-6 / 1.1) * (1 if active == "object" else -1)
    assert math.fsum(v.derivative for v in values) / 18 == pytest.approx(expected, abs=2e-14)
    assert all(v.continuation_histories == v.physical_prefixes == 0 for v in values)


@pytest.mark.parametrize("active, expected", (("object", 2 / 3), ("source", -0.5)))
def test_cone_partial_shadow_matches_independent_projected_front_face_area(active, expected):
    spec, medium = _scene()
    source = PointConeSource(
        DetectorGeometry(
            source_mm=(0.5, 0.5, -1),
            origin_mm=(-1, 0, 3),
            u=(1, 0, 0),
            v=(0, 1, 0),
            spacing_mm=(4, 6),
            shape=(1, 1),
        )
    )
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=RigidMotion((1, 0, 0)))
        if active == "object"
        else GridGeometryTangent(),
        source=SourceGeometryTangent(motion=RigidMotion((1, 0, 0)))
        if active == "source"
        else SourceGeometryTangent(),
    )
    # Front-face projection hasheight4 andx[-1.5,2.5], targetx[-3,1].
    # Moving object+dx movesleftedge4dx; movingpoint+dx movesit-3dx.
    values = [
        composed_geometry_observation(
            source,
            (0.1, 0.4),
            40,
            spec,
            medium,
            OpaqueCoordinates(boundary=i, line_coordinate=(j + 0.5) / 3),
            original_history=i,
            cotangent=(1,),
            tangent=tangent,
        )
        for i in range(18)
        for j in range(3)
    ]
    # The full source-line measure retains null draws outside the projected
    # edge segment: exactly two of these three equal source intervals hit it.
    assert math.fsum(v.derivative for v in values) / 54 == pytest.approx(expected, abs=3e-14)


def test_half_open_upper_parallel_entry_and_exact_coordinate_side_limits():
    spec, _ = _scene()
    assert not opaque_material_hit(RayTangent((1, 0.5, -1), (0, 0, 1)), spec)
    source = ParallelBeam((-0.5, 0.1, -1), (1.1, 0.8))
    records = opaque_source_boundaries(source, spec, AcquisitionGeometryTangent())
    row = next(
        row
        for row in records
        if row.boundary.coefficients[2] != 0 and row.boundary.coefficients[0] != 0
    )
    sample = row.boundary.sample(0.5)
    assert sample is not None
    sides = opaque_boundary_sides(source, sample[0], row, spec)
    assert sides[0] != sides[1]


@pytest.mark.parametrize("dot", ((1.7e308, 1.7e308, 1.7e308), (1e200, 0, 0)))
def test_finite_huge_nonorthogonal_ray_jet_cannot_pass_via_overflowed_norm(dot):
    with pytest.raises(TransportError, match="orthogonal"):
        RayTangent((0, 0, 0), (1, 0, 0), direction_dot=dot)


def test_finite_huge_orthogonal_ray_jet_retains_valid_domain():
    assert RayTangent((0, 0, 0), (1, 0, 0), direction_dot=(0, 1.7e308, 1.7e308))


def test_active_entire_launch_face_is_rejected_before_any_draw():
    spec, medium = _scene()
    source = ParallelBeam((0.1, 0.1, 0), (0.8, 0.8))
    coordinates = OpaqueCoordinates()
    with pytest.raises(TransportError, match="launch plane coincides"):
        composed_geometry_observation(
            source,
            (0.5, 0.5),
            40,
            spec,
            medium,
            coordinates,
            original_history=4,
            cotangent=(1,),
            tangent=AcquisitionGeometryTangent(
                source=SourceGeometryTangent(motion=RigidMotion((0, 0, 1)))
            ),
        )
    assert not coordinates.draws


def test_rotated_diamond_shadow_attached_faces_match_independent_chord_derivative():
    spec, _ = _scene()
    c = math.sqrt(0.5)
    spec = replace(spec, object_pose=RigidTransform(rotation=(c, -c, 0, c, c, 0, 0, 0, 1)))
    source = ParallelBeam((-0.5, -0.2, -1), (1.1, 1.8))
    tangent = AcquisitionGeometryTangent(object=GridGeometryTangent(motion=RigidMotion((1, 0, 0))))
    rows = opaque_source_boundaries(source, spec, tangent)
    assert len(rows) == 4
    count = 512
    values = []
    bounds = []
    for row in rows:
        terms = []
        for j in range(count):
            sample = row.boundary.sample((j + 0.5) / count)
            assert sample is not None
            coordinate, coarea = sample
            negative, positive = opaque_boundary_sides(source, coordinate, row, spec)
            terms.append(coarea * (int(negative) - int(positive)))
        values.append(math.fsum(terms) / count)
        # Each diamond face has one clipped segment, hence at most two partial
        # midpoint intervals. This is an independent deterministic error bound.
        bounds.append(2 * max(abs(value) for value in terms) / count)
    # Rotated-unit-square chords atx=-.5/.6 have lengths sqrt2-1 and
    # sqrt2-1.2. Translation increases shadowarea by.2 permm, so escaping
    # source-area derivative is-.2/(1.1*1.8), regardless binary64 rotation.
    expected = -0.2 / (1.1 * 1.8)
    assert abs(math.fsum(values) - expected) <= math.fsum(bounds)
    assert math.fsum(values) < 0


@pytest.mark.parametrize("mark", (0.125, 0.25, 0.5, 0.75, 0.875))
def test_rotated_single_face_has_exact_same_complete_jump_at_every_mark(mark):
    spec, _ = _scene()
    c = math.sqrt(0.5)
    spec = replace(spec, object_pose=RigidTransform(rotation=(c, -c, 0, c, c, 0, 0, 0, 1)))
    source = ParallelBeam((0.2, 0.1, -1), (0.2, 0.4))
    rows = opaque_source_boundaries(
        source,
        spec,
        AcquisitionGeometryTangent(
            object=GridGeometryTangent(motion=RigidMotion((1, 0, 0))),
        ),
    )
    assert len(rows) == 1
    sample = rows[0].boundary.sample(mark)
    assert sample is not None
    coordinate, coarea = sample
    negative, positive = opaque_boundary_sides(source, coordinate, rows[0], spec)
    assert coarea * (int(negative) - int(positive)) == pytest.approx(-2.5, abs=1e-14)


def test_stationary_detector_and_opaque_joint_line_does_not_invent_a_derivative():
    spec, medium = _scene()
    spec = replace(spec, detector=PlanarDetector((0, -2), (2, 5), (1, 1), 3))
    result = composed_geometry_observation(
        ParallelBeam((-0.5, 0.1, -1), (1.1, 0.8)),
        (0.1, 0.4),
        40,
        spec,
        medium,
        OpaqueCoordinates(),
        original_history=4,
        cotangent=(1,),
        tangent=AcquisitionGeometryTangent(),
    )
    assert result.derivative == 0


def test_active_detector_and_opaque_joint_line_rejects_a_classical_kink():
    spec, medium = _scene()
    spec = replace(spec, detector=PlanarDetector((0, -2), (2, 5), (1, 1), 3))
    with pytest.raises(TransportError, match="coincident active opaque/detector"):
        composed_geometry_observation(
            ParallelBeam((-0.5, 0.1, -1), (1.1, 0.8)),
            (0.1, 0.4),
            40,
            spec,
            medium,
            OpaqueCoordinates(),
            original_history=4,
            cotangent=(1,),
            tangent=AcquisitionGeometryTangent(
                object=GridGeometryTangent(motion=RigidMotion((1, 0, 0)))
            ),
        )


def test_active_relative_grazing_angle_rejects_box_shadow_absolute_value_kink():
    spec, medium = _scene()
    source = ParallelBeam((-1, -1, -1), (3, 3))
    coordinates = OpaqueCoordinates()
    with pytest.raises(TransportError, match="angular motion leaves an opaque grazing"):
        composed_geometry_observation(
            source,
            (0.5, 0.5),
            40,
            spec,
            medium,
            coordinates,
            original_history=0,
            cotangent=(1,),
            tangent=AcquisitionGeometryTangent(
                source=SourceGeometryTangent(direction_dot=(1, 0, 0))
            ),
        )
    assert not coordinates.draws


def test_grazing_angle_without_visible_shadow_boundary_remains_admitted():
    spec, medium = _scene()
    source = ParallelBeam((0.2, 0.2, -1), (0.2, 0.2))
    result = composed_geometry_observation(
        source,
        (0.5, 0.5),
        40,
        spec,
        medium,
        OpaqueCoordinates(),
        original_history=0,
        cotangent=(1,),
        tangent=AcquisitionGeometryTangent(source=SourceGeometryTangent(direction_dot=(1, 0, 0))),
    )
    assert result.mean == result.derivative == 0


def test_common_rigid_rotation_cancels_relative_grazing_departure():
    spec, medium = _scene()
    source = ParallelBeam((-1, -1, -1), (3, 3))
    motion = RigidMotion(angular=(0, 1, 0))
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=motion),
        source=SourceGeometryTangent(motion=motion),
        detector=motion,
    )
    result = composed_geometry_observation(
        source,
        (0.1, 0.4),
        40,
        spec,
        medium,
        OpaqueCoordinates(),
        original_history=0,
        cotangent=(1,),
        tangent=tangent,
    )
    assert result.derivative == 0


@pytest.mark.parametrize("visible", (True, False))
def test_entire_source_plane_grazing_angle_checks_original_area_before_null_row(visible):
    spec, medium = _scene()
    detector = OrientedDetector(
        DetectorGeometry(
            source_mm=(-2, 0.5, 0),
            origin_mm=(3, 0.5, 0),
            u=(0, 1, 0),
            v=(0, 0, 1),
            spacing_mm=(2, 2),
            shape=(1, 1),
        )
    )
    spec = replace(spec, detector=detector)
    source = ParallelBeam((-2, 0.1 if visible else 2, 0), (0.5, 0.8), direction=(1, 0, 0))
    tangent = AcquisitionGeometryTangent(source=SourceGeometryTangent(direction_dot=(0, 0, 1)))
    coordinates = OpaqueCoordinates()
    if visible:
        # For any small negativeangle the entirebeam missesandescapes; atzero
        # andpositiveangle itentersanddeposits. There is no classical JVP.
        with pytest.raises(TransportError, match="whole-source opaque grazing"):
            composed_geometry_observation(
                source,
                (0.5, 0.5),
                40,
                spec,
                medium,
                coordinates,
                original_history=0,
                cotangent=(1,),
                tangent=tangent,
            )
        assert not coordinates.draws
    else:
        result = composed_geometry_observation(
            source,
            (0.5, 0.5),
            40,
            spec,
            medium,
            coordinates,
            original_history=0,
            cotangent=(1,),
            tangent=tangent,
        )
        assert result.derivative == 0
