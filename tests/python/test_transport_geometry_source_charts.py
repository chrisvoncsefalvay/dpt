"""Independent angular support and stationary source-plane chart controls."""

# The tests inspect prepared source constraints as a numerical reference.
# pyright: reportPrivateUsage=false
import math
from dataclasses import replace
from typing import Literal

import pytest

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.transport.geometry_composed import composed_geometry_observation
from dpt.transport.geometry_flights import optical_flight_tangent
from dpt.transport.geometry_histories import GeometryPhysicalMedium
from dpt.transport.geometry_source_jumps import (
    material_source_ray_boundaries,
    sample_material_source_ray_jump,
)
from dpt.transport.geometry_tangents import (
    AffineBoundary,
    GridGeometryTangent,
    RayTangent,
    RigidMotion,
    material_side_local_ray,
)
from dpt.transport.model import (
    MaterialGrid,
    OrientedDetector,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from dpt.transport.source import ParallelBeam, PointConeSource
from dpt.transport.uncollided_geometry import AcquisitionGeometryTangent, SourceGeometryTangent


def _scene():
    geometry = DetectorGeometry((0, 0.5, -1), (0, 0.5, 3), (1, 0, 0), (0, 1, 0), (2, 0.8), (1, 1))
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 1), 1),
        PlanarDetector((-2, -2), (4, 4), (1, 1), 3),
        80,
        "Explicit coplanar source angular-support control",
        (80,),
        max_events=4,
        max_angle_trials=2,
    )
    return PointConeSource(geometry), spec


def test_stationary_coplanar_cone_and_common_rigid_chart_remain_admitted():
    source, spec = _scene()
    assert material_source_ray_boundaries(source, spec, AcquisitionGeometryTangent()) == ()
    motion = RigidMotion((0, 1, 0), (0, 0, 0.25), (0, 0.5, -1))
    common = AcquisitionGeometryTangent(
        source=SourceGeometryTangent(motion=motion, target_motion=motion),
        object=GridGeometryTangent(motion=motion),
        detector=motion,
    )
    assert material_source_ray_boundaries(source, spec, common) == ()


def test_active_cone_target_uses_original_source_angular_area_and_complete_sides():
    source, spec = _scene()
    tangent = AcquisitionGeometryTangent(
        source=SourceGeometryTangent(target_motion=RigidMotion((1, 0, 0)))
    )
    (record,) = material_source_ray_boundaries(source, spec, tangent)
    assert (record.axis, record.face) == (0, 0)
    assert record.boundary.coefficients == (2, 0, -1)
    assert record.boundary.coefficients_dot == (0, 0, 1)
    sample = sample_material_source_ray_jump(source, record, 0.5, spec=spec)
    assert sample.source_coordinate == (0.5, 0.5)
    assert sample.ray.direction == (0, 0, 1)
    assert sample.coarea_weight == 0.5
    assert sample.negative_side == (0, 0, -1) and sample.positive_side == (0, 0, 1)


def test_coplanar_cone_relative_normal_departure_keeps_its_kink_guard():
    source, spec = _scene()
    with pytest.raises(TransportError, match="relative normal motion"):
        material_source_ray_boundaries(
            source,
            spec,
            AcquisitionGeometryTangent(source=SourceGeometryTangent(motion=RigidMotion((1, 0, 0)))),
        )


def test_parallel_grazing_direction_without_visible_source_boundary_is_regular():
    _, spec = _scene()
    source = ParallelBeam((0.4, 0.45, -1), (0, 0))
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=RigidMotion(angular=(0, 1, 0)))
    )
    assert material_source_ray_boundaries(source, spec, tangent) == ()


def test_stationary_parallel_pencil_on_half_open_face_remains_admitted():
    _, spec = _scene()
    source = ParallelBeam((0, 0.45, -1), (0, 0))
    assert material_source_ray_boundaries(source, spec, AcquisitionGeometryTangent()) == ()


def test_invisible_cone_plane_does_not_reject_relative_normal_motion():
    source, spec = _scene()
    source = replace(
        source, geometry=replace(source.geometry, source_mm=(0, 10, -1), origin_mm=(0, 10, 3))
    )
    tangent = AcquisitionGeometryTangent(
        source=SourceGeometryTangent(motion=RigidMotion((1, 0, 0)))
    )
    assert material_source_ray_boundaries(source, spec, tangent) == ()


def test_invisible_cone_line_on_original_source_edge_remains_admitted():
    source, spec = _scene()
    source = replace(
        source,
        geometry=replace(source.geometry, source_mm=(0, 10, -1), origin_mm=(1, 10, 3)),
    )
    tangent = AcquisitionGeometryTangent(
        source=SourceGeometryTangent(target_motion=RigidMotion((1, 0, 0)))
    )
    assert material_source_ray_boundaries(source, spec, tangent) == ()


def test_source_edge_constraint_rescaling_has_zero_boundary_velocity():
    assert AffineBoundary("fixed-left-edge", (2, 0, 0), (1, 0, 0)).endpoints() is None
    assert AffineBoundary("fixed-top-edge", (0, 2, -2), (0, 1, -1)).endpoints() is None
    with pytest.raises(TransportError, match="moving boundary coincides"):
        AffineBoundary("moving-left-edge", (2, 0, 0), (0, 0, 1)).endpoints()


class AngularCoordinates:
    identity = "declared full-history midpoint quadrature; no confidence claim"
    layout: Literal["legacy-v1", "packed-compton-v2", "bound-compton-v3"] = "legacy-v1"

    def __init__(self, uniform):
        self.uniform_value = uniform

    def uniform(self, lineage, event, domain, lane):
        if domain == 0x30000003:
            return (0.05, 0.05, 0.5)[lane]
        if domain == 0 and lane == 0:
            return self.uniform_value
        return 0.37


def test_complete_cone_angular_future_jump_matches_independent_absorbing_face():
    source, spec = _scene()
    medium = GeometryPhysicalMedium((0,), (0.7,), (0,), (1,), (1,))
    tangent = AcquisitionGeometryTangent(
        source=SourceGeometryTangent(target_motion=RigidMotion((1, 0, 0)))
    )
    count = 512
    observations = [
        composed_geometry_observation(
            source,
            (0.25, 0.5),
            80,
            spec,
            medium,
            AngularCoordinates((i + 0.5) / count),
            original_history=i,
            cotangent=(1,),
            tangent=tangent,
        )
        for i in range(count)
    ]
    # At the selected angular line v=.5, the ray is exactly +z. Its negative
    # side misses; its positive side traverses one mm of an absorbing slab.
    # The inverse plane-selection probability is6 and angular coarea1/2.
    expected = 3 * (math.exp(-0.7) - 1)
    assert math.fsum(o.derivative for o in observations) / count == pytest.approx(
        expected, abs=1.5 / count
    )
    assert all(o.continuation_histories == 2 and o.physical_prefixes == 0 for o in observations)
    assert all(
        o.ballistic_generator == o.collided_generator == o.terminal_geometry == 0
        for o in observations
    )


@pytest.mark.parametrize("angle", [0.37, 0.73, 1.2])
@pytest.mark.parametrize("coordinate", [0.125, 0.37, 0.618, 0.875])
def test_rotated_cone_attached_local_side_preserves_exact_absorbing_chord(angle, coordinate):
    c, s = math.cos(angle), math.sin(angle)
    pose = RigidTransform((c, 0, s, 0, 1, 0, -s, 0, c))
    geometry = DetectorGeometry(
        pose.point((0.5, -1, 0)),
        pose.point((0.5, 3, 0)),
        pose.direction((1, 0, 0)),
        pose.direction((0, 0, 1)),
        (0.8, 2),
        (1, 1),
    )
    source = PointConeSource(geometry)
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 1), 1),
        OrientedDetector(geometry),
        80,
        "Rotated attached cone direction",
        (80,),
        object_pose=pose,
    )
    tangent = AcquisitionGeometryTangent(
        source=SourceGeometryTangent(target_motion=RigidMotion(pose.direction((0, 0, 1))))
    )
    record = next(
        row
        for row in material_source_ray_boundaries(source, spec, tangent)
        if row.axis == 2 and row.face == 0
    )
    sample = sample_material_source_ray_jump(source, record, coordinate, spec=spec)
    # The physical world proxy is retained. The explicit face label owns its
    # local zero normal, including roundoff introduced by a proper rigid pose.
    local = material_side_local_ray(sample.ray, spec.grid, pose, GridGeometryTangent(), (2, 0, 1))
    assert local.direction[2] == 0
    expected_length = math.hypot(4, 0.8 * (sample.source_coordinate[0] - 0.5)) / 4
    positive = optical_flight_tangent(
        sample.ray, spec, extinction_mm_inverse=(0.7,), optical_depth=100, material_side=(2, 0, 1)
    )
    negative = optical_flight_tangent(
        sample.ray, spec, extinction_mm_inverse=(0.7,), optical_depth=100, material_side=(2, 0, -1)
    )
    assert positive.total_optical_depth == pytest.approx(0.7 * expected_length, rel=2e-14)
    assert negative.total_optical_depth == 0


@pytest.mark.parametrize(
    "origin,direction", [((0.5, -1, 0.01), (0, 1, 0)), ((0.5, -1, 0), (0, math.sqrt(0.99), 0.1))]
)
def test_attached_material_side_rejects_a_wrong_plane_or_transverse_ray(origin, direction):
    _, spec = _scene()
    with pytest.raises(TransportError, match=r"attached material face|parallel to its face"):
        material_side_local_ray(
            RayTangent(origin, direction),
            spec.grid,
            spec.object_pose,
            GridGeometryTangent(),
            (2, 0, 1),
        )
