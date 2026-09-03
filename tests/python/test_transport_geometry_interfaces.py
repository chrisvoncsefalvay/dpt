"""Independent shape-generator signs and normalised near-surface bridge measures."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from dpt.geometry import DetectorGeometry
from dpt.transport.geometry_interfaces import (
    direct_interface_incident,
    interface_generator_observation,
    interface_point_tangent,
    material_interfaces,
    sample_interface_bridge,
    source_path_gauge,
)
from dpt.transport.geometry_tangents import GridGeometryTangent, RigidMotion
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.source import ParallelBeam, PointConeSource
from dpt.transport.uncollided_geometry import SourceGeometryTangent


def _scene() -> TransportSpec:
    return TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (3.0, 3.0), (1, 1), 2.0),
        80.0,
        "exact slab interface-generator and Cauchy-product proposal",
    )


def test_interface_rectangles_bind_both_cells_and_complete_surface_area() -> None:
    spec = replace(_scene(), grid=MaterialGrid((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1, 1, 2), 2))
    interfaces = material_interfaces(spec)
    assert len(interfaces) == 11
    middle = next(face for face in interfaces if face.axis == 0 and face.face == 1)
    assert (middle.negative_cell, middle.positive_cell) == (0, 1)
    point = interface_point_tangent(
        middle, (0.25, 0.75), spec, GridGeometryTangent(spacing_dot=(0.1, 0.2, 0.3))
    )
    assert point.point_mm == (0.5, 0.25, 0.75)
    assert point.point_dot == pytest.approx((0.1, 0.05, 0.225))
    assert point.normal == (1.0, 0.0, 0.0) and point.area_mm2 == 1.0
    with pytest.raises(TransportError, match="cell sides"):
        interface_point_tangent(replace(middle, negative_cell=1), (0.3, 0.7), spec)


def test_absorption_shape_generator_matches_exact_slab_thickness_derivative() -> None:
    spec, mu = _scene(), 0.7
    face = next(item for item in material_interfaces(spec) if item.axis == 2 and item.face == 1)
    point = interface_point_tangent(
        face, (0.4, 0.6), spec, GridGeometryTangent(spacing_dot=(0.0, 0.0, 1.0))
    )
    result = interface_generator_observation(
        point,
        negative_extinction_mm_inverse=mu,
        positive_extinction_mm_inverse=0.0,
        negative_collision_response=0.0,
        positive_collision_response=None,
        unchanged_direction_response=1.0,
        incident_importance=math.exp(-mu),
    )
    assert result == pytest.approx(-mu * math.exp(-mu), rel=1e-15)


def test_two_translating_slab_faces_cancel_with_independent_physical_future_scores() -> None:
    spec, mu = _scene(), 0.7
    values = []
    for face in material_interfaces(spec):
        if face.axis != 2:
            continue
        point = interface_point_tangent(
            face,
            (0.4, 0.6),
            spec,
            GridGeometryTangent(motion=RigidMotion(translation=(0.0, 0.0, 1.0))),
        )
        before, after = (0.0, mu) if face.face == 0 else (mu, 0.0)
        values.append(
            interface_generator_observation(
                point,
                negative_extinction_mm_inverse=before,
                positive_extinction_mm_inverse=after,
                negative_collision_response=None if before == 0.0 else 0.0,
                positive_collision_response=None if after == 0.0 else 0.0,
                unchanged_direction_response=math.exp(-mu) if face.face == 0 else 1.0,
                incident_importance=1.0 if face.face == 0 else math.exp(-mu),
            )
        )
    assert math.fsum(values) == 0.0


def test_grazing_ballistic_source_requires_whole_ray_jump_not_regular_surface_product() -> None:
    spec = _scene()
    face = next(item for item in material_interfaces(spec) if item.axis == 0 and item.face == 0)
    point = interface_point_tangent(face, (0.4, 0.6), spec)
    source = ParallelBeam((-1.0, -1.0, -1.0), (3.0, 3.0))
    with pytest.raises(TransportError, match="whole-source-ray jump"):
        direct_interface_incident(point, source)


@pytest.mark.parametrize("height", (0.3, 1e-4, 1e-10))
def test_bridge_importance_cancels_inverse_square_without_an_epsilon(height: float) -> None:
    spec = _scene()
    face = next(item for item in material_interfaces(spec) if item.axis == 2 and item.face == 0)
    previous = (0.4, 0.6, height)
    for coordinate in ((0.1, 0.2), (0.4, 0.7), (0.6, 0.8)):
        sample = sample_interface_bridge(face, previous, coordinate, spec)
        direct_distance = math.dist(previous, sample.point.point_mm)
        inverse_pdf = math.exp(-sample.log_area_density)
        assert math.exp(sample.log_geometric_importance) == pytest.approx(
            inverse_pdf / direct_distance**2, rel=1e-12
        )
        assert math.exp(sample.log_geometric_importance) <= math.prod(sample.axis_normalisers) * (
            1.0 + 1e-12
        )
        assert all(
            math.isfinite(value)
            for value in (sample.log_area_density, sample.log_geometric_importance)
        )


def test_surface_proposal_rounding_failure_is_fatal_instead_of_an_epsilon_displacement() -> None:
    spec = _scene()
    face = next(item for item in material_interfaces(spec) if item.axis == 2 and item.face == 0)
    with pytest.raises(TransportError, match="distinguishable"):
        sample_interface_bridge(face, (0.4, 0.6, 1e-100), (0.4, 0.7), spec)


def test_proposal_normalisation_matches_independent_simpson_integration() -> None:
    height, bounds = 0.3, ((-0.4, 0.6), (-0.6, 0.4))
    n = 1024
    integrals = []
    for lower, upper in bounds:
        normal = math.asinh(upper / height) - math.asinh(lower / height)
        step = (upper - lower) / n
        integral = (
            math.fsum(
                (1.0 if i in (0, n) else 4.0 if i % 2 else 2.0)
                / math.hypot(height, lower + i * step)
                / normal
                for i in range(n + 1)
            )
            * step
            / 3.0
        )
        integrals.append(integral)
    assert math.prod(integrals) == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize("cone", (False, True))
def test_source_path_rigid_gauge_preserves_actual_launch_position_and_direction_jets(
    cone: bool,
) -> None:
    source = (
        PointConeSource(
            DetectorGeometry(
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 2.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 2.0),
                (2, 2),
            )
        )
        if cone
        else (ParallelBeam((-1.0, -1.0, -1.0), (2.0, 2.0)))
    )
    motion = RigidMotion(translation=(0.1, 0.2, 0.3), angular=(0.2, -0.1, 0.1))
    tangent = SourceGeometryTangent(motion=motion, target_motion=motion)
    gauge = source_path_gauge(source, (0.3, 0.7), tangent)
    from dpt.transport.uncollided_geometry import _source_ray

    ray = _source_ray(source, (0.3, 0.7), tangent)[0]
    assert gauge.point_velocity(ray.origin_mm) == pytest.approx(ray.origin_dot, abs=1e-15)
    assert gauge.direction_velocity(ray.direction) == pytest.approx(ray.direction_dot, abs=1e-15)


def test_missing_nonvacuum_response_is_not_replaced_by_zero() -> None:
    spec = _scene()
    face = material_interfaces(spec)[0]
    point = interface_point_tangent(face, (0.3, 0.7), spec)
    with pytest.raises(TransportError, match="actual collision continuation"):
        interface_generator_observation(
            point,
            negative_extinction_mm_inverse=0.0,
            positive_extinction_mm_inverse=0.7,
            negative_collision_response=None,
            positive_collision_response=None,
            unchanged_direction_response=1.0,
            incident_importance=1.0,
        )
