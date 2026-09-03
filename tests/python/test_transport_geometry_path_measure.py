"""Independent area, volume, source support and attached phase-space measures."""

from __future__ import annotations

from dataclasses import replace

import pytest

from dpt.geometry import DetectorGeometry
from dpt.transport.geometry_parameters import GeometryParameter
from dpt.transport.geometry_path_measure import (
    VolumeBoundary,
    collision_vertex_tangent,
    detector_vertex_tangent,
    source_flux_tangent,
    source_volume_boundaries,
)
from dpt.transport.geometry_tangents import GridGeometryTangent, RigidMotion
from dpt.transport.model import (
    MaterialGrid,
    OrientedDetector,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from dpt.transport.source import ParallelBeam, PointConeSource
from dpt.transport.uncollided_geometry import SourceGeometryTangent


def _spec() -> TransportSpec:
    return TransportSpec(
        MaterialGrid((-2.0, -2.0, 0.0), (1.0, 1.0, 1.0), (2, 4, 4), 1),
        PlanarDetector((-4.0, -4.0), (1.0, 1.0), (8, 8), 4.0),
        80.0,
        "synthetic collision-volume measure",
    )


def test_collision_measure_carries_grid_spacing_and_rigid_motion() -> None:
    motion = RigidMotion(translation=(1.0, 2.0, 3.0), angular=(0.0, 0.0, 1.0))
    tangent = GridGeometryTangent(motion=motion, spacing_dot=(0.1, 0.2, 0.3))
    vertex = collision_vertex_tangent((0.625, 0.375, 0.25), _spec(), tangent)
    assert vertex.point_mm == (0.5, -0.5, 0.5)
    assert vertex.cell == 6
    assert vertex.point_dot == pytest.approx((1.75, 2.8, 3.15))
    assert vertex.volume_jacobian == 32.0
    assert vertex.volume_jacobian_dot == pytest.approx(19.2)


def test_detector_measure_keeps_pixel_and_moves_area_and_support() -> None:
    detector = _spec().detector
    tangent = detector_vertex_tangent(
        (0.3, 0.7), detector, RigidMotion(translation=(1.0, 0.0, 0.0)), spacing_dot=(0.1, 0.2)
    )
    assert tangent.pixel == 42
    assert tangent.point_mm == pytest.approx((-1.6, 1.6, 4.0))
    assert tangent.point_dot == pytest.approx((1.24, 1.12, 0.0))
    assert tangent.area_jacobian == 64.0
    assert tangent.area_jacobian_dot == pytest.approx(19.2)


def test_parallel_incident_fluence_has_area_and_projection_normalisation() -> None:
    direction = (0.6, 0.0, 0.8)
    source = ParallelBeam((-2.0, -2.0, -1.0), (4.0, 4.0), direction, weight=2.0)
    flux = source_flux_tangent(
        source, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), SourceGeometryTangent(extent_dot=(1.0, 0.0))
    )
    assert flux.distance_mm == 1.25
    assert flux.incoming.origin_mm == pytest.approx((-0.75, 0.0, -1.0))
    assert flux.density == pytest.approx(2.0 / (16.0 * 0.8))
    assert flux.density_dot == pytest.approx(-flux.density / 4.0)
    assert flux.support_dot[0] == pytest.approx((1.0 - 0.3125) / 4.0)


def test_cone_fluence_matches_original_uniform_target_area_law() -> None:
    geometry = DetectorGeometry(
        (0.0, 0.0, -1.0), (0.0, 0.0, 4.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (8.0, 8.0), (1, 1)
    )
    source = PointConeSource(geometry, weight=2.0)
    # At x=0, z=1 the target-to-collision distance ratio is 5/2. Fluence is
    # weight/area times its square. Moving the collision along z differentiates
    # 1/(z+1)^2 at z=1, hence relative slope -1/mm.
    flux = source_flux_tangent(source, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    assert flux.density == pytest.approx(2.0 / 64.0 * (5.0 / 2.0) ** 2)
    assert flux.density_dot == pytest.approx(-flux.density)
    assert flux.visible


def test_full_geometry_rotation_cancels_incident_fluence_jet() -> None:
    source = ParallelBeam((-2.0, -2.0, -1.0), (4.0, 4.0), weight=2.0)
    motion = RigidMotion(translation=(0.3, 0.4, -0.2), angular=(0.2, -0.1, 0.3))
    point = (0.2, 0.3, 1.0)
    flux = source_flux_tangent(
        source, point, motion.point_velocity(point), SourceGeometryTangent(motion=motion)
    )
    assert flux.distance_dot == pytest.approx(0.0, abs=1e-14)
    assert flux.density_dot == pytest.approx(0.0, abs=1e-14)
    assert flux.support_dot == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1e-14)


@pytest.mark.parametrize("coordinate", ((0.1, 0.2), (0.4, 0.7), (0.9, 0.8)))
def test_volume_plane_sampler_carries_exact_coarea_measure(coordinate: tuple[float, float]) -> None:
    boundary = VolumeBoundary("plane", (1.0, 1.0, 0.0), -1.0, (0.0, 0.0, 0.0), 2.0)
    # The rectangle x+y=1 has area sqrt(2), normal magnitude sqrt(2),
    # giving signed weight exactly 2 for constant g_dot=2.
    point, weight = boundary.sample(coordinate) or pytest.fail("plane is present")
    assert point[0] + point[1] == pytest.approx(1.0, abs=1e-14)
    assert weight == pytest.approx(2.0, abs=1e-14)


def test_source_footprint_boundary_moves_in_collision_volume_coordinates() -> None:
    source = ParallelBeam((-0.5, -3.0, -1.0), (1.0, 6.0))
    source_tangent = SourceGeometryTangent(motion=RigidMotion(translation=(1.0, 0.0, 0.0)))
    boundaries = source_volume_boundaries(source, _spec(), source_tangent=source_tangent)
    assert {boundary.key for boundary in boundaries} == {"source:0", "source:1"}
    by_key = {boundary.key: boundary for boundary in boundaries}
    _, left_weight = by_key["source:0"].sample((0.3, 0.7)) or pytest.fail("left edge")
    _, right_weight = by_key["source:1"].sample((0.3, 0.7)) or pytest.fail("right edge")
    # Unit reference x spans 4 mm: each moving plane's g_dot/|grad g| = +/-1/4.
    assert left_weight == pytest.approx(-0.25)
    assert right_weight == pytest.approx(0.25)


def test_volume_boundary_does_not_use_coordinate_epsilon_or_discard_coincidence() -> None:
    with pytest.raises(TransportError, match="coincides"):
        VolumeBoundary("coincident", (1.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0), 1.0).polygon()


def test_oriented_detector_spacing_uses_first_pixel_centre_as_anchor() -> None:
    geometry = DetectorGeometry(
        (0.0, 0.0, -1.0), (0.0, 0.0, 4.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0), (2, 4)
    )
    endpoint = detector_vertex_tangent(
        (0.25, 0.75), OrientedDetector(geometry), spacing_dot=(0.2, 0.3)
    )
    assert endpoint.point_dot == pytest.approx((0.1, 0.3, 0.0))
    assert endpoint.area_jacobian_dot == pytest.approx(4.0)


def test_cone_launch_spacing_changes_density_at_its_existing_centre_anchor() -> None:
    geometry = DetectorGeometry(
        (0.0, 0.0, -1.0), (0.0, 0.0, 4.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (8.0, 8.0), (1, 1)
    )
    source = PointConeSource(geometry)
    flux = source_flux_tangent(
        source,
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        SourceGeometryTangent(target_spacing_dot=(1.0, 0.0)),
    )
    assert flux.density_dot == pytest.approx(-flux.density / 8.0)
    assert flux.support_dot == (0.0, 0.0, 0.0, 0.0)


def test_independent_beam_direction_changes_projection_without_rotating_source_area() -> None:
    direction = (0.6, 0.0, 0.8)
    source = ParallelBeam((-2.0, -2.0, -1.0), (4.0, 4.0), direction)
    tangent = SourceGeometryTangent(direction_dot=(0.8, 0.0, -0.6))
    flux = source_flux_tangent(source, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), tangent)
    assert flux.incoming.direction_dot == tangent.direction_dot
    assert flux.distance_dot == pytest.approx(0.6 / 0.8**2)
    assert flux.density_dot == pytest.approx(flux.density * 0.6 / 0.8)
    with pytest.raises(TransportError, match="orthogonal"):
        source_flux_tangent(
            source,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            SourceGeometryTangent(direction_dot=(0.6, 0.0, 0.8)),
        )


def test_collision_local_label_is_fixed_under_object_motion() -> None:
    spec = _spec()
    moved = replace(spec, object_pose=replace(spec.object_pose, translation_mm=(1.0, 0.0, 0.0)))
    first = collision_vertex_tangent((0.2, 0.3, 0.4), spec)
    second = collision_vertex_tangent((0.2, 0.3, 0.4), moved)
    assert first.cell == second.cell
    assert second.point_mm[0] == first.point_mm[0] + 1.0


def test_named_point_source_rotation_about_external_world_pivot_moves_the_emitter() -> None:
    geometry = DetectorGeometry(
        (1.0, 0.0, -1.0), (0.0, 0.0, 4.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (8.0, 8.0), (1, 1)
    )
    source = PointConeSource(geometry)
    tangent = GeometryParameter("source", "rotation", 2, (0.0, 0.0, 0.0)).tangent(source)
    assert tangent.source.motion.point_velocity(geometry.source_mm) == (0.0, 1.0, 0.0)
