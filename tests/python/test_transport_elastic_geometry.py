"""Collision-order integrands checked against independent solid-angle formulae."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from dpt.transport.elastic_geometry import (
    CollisionOrderLaw,
    CollisionPathSample,
    ElasticGeometryMedium,
    SourceBoundaryPathSample,
    elastic_geometry_jvp,
)
from dpt.transport.geometry_path_measure import source_volume_boundaries
from dpt.transport.geometry_tangents import GridGeometryTangent, RigidMotion
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.source import ParallelBeam
from dpt.transport.uncollided_geometry import AcquisitionGeometryTangent, SourceGeometryTangent


def _scene() -> tuple[ParallelBeam, TransportSpec, ElasticGeometryMedium]:
    source = ParallelBeam((-2.0, -2.0, -1.0), (4.0, 4.0))
    spec = TransportSpec(
        MaterialGrid((-0.5, -0.5, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-0.5, -0.5), (1.0, 1.0), (1, 1), 3.0),
        80.0,
        "synthetic elastic collision-volume reference",
    )
    return source, spec, ElasticGeometryMedium((0.3,), (0.2,))


def _one(tangent: AcquisitionGeometryTangent | None = None):
    tangent = AcquisitionGeometryTangent() if tangent is None else tangent
    source, spec, medium = _scene()
    return elastic_geometry_jvp(
        source,
        spec,
        medium,
        tangent=tangent,
        cotangent=(1.0,),
        order_law=CollisionOrderLaw(0.5),
        histories=(CollisionPathSample(((0.5, 0.5, 0.5),), (0.5, 0.5)),),
        source_boundaries={},
    )


def test_single_collision_integrand_matches_independent_volume_solid_angle() -> None:
    result = _one()
    # One 1mm^3 volume vertex at z=.5, source fluence 1/16, isotropic
    # collision density .2/(4*pi), projected detector solid angle 1/2.5^2,
    # total unit-slab attenuation exp(-.3), and order importance 1/P(N=1)=2.
    expected = (1.0 / 16.0) * (0.2 / (4.0 * math.pi)) / 2.5**2 * math.exp(-0.3) * 2.0
    assert result.mean == pytest.approx(expected, rel=1e-14)
    assert result.derivative == 0.0


def test_object_and_detector_translation_include_inverse_square_geometry() -> None:
    motion = RigidMotion(translation=(0.0, 0.0, 1.0))
    object_result = _one(AcquisitionGeometryTangent(object=GridGeometryTangent(motion=motion)))
    detector_result = _one(AcquisitionGeometryTangent(detector=motion))
    assert object_result.derivative == pytest.approx(2.0 / 2.5 * object_result.mean, rel=1e-14)
    assert detector_result.derivative == pytest.approx(-2.0 / 2.5 * detector_result.mean, rel=1e-14)


def test_full_common_rotation_and_translation_cancel_every_path_geometry_term() -> None:
    source, spec, medium = _scene()
    motion = RigidMotion(translation=(0.3, -0.2, 0.1), angular=(0.1, 0.2, -0.3))
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=motion),
        source=SourceGeometryTangent(motion=motion),
        detector=motion,
    )
    result = elastic_geometry_jvp(
        source,
        spec,
        medium,
        tangent=tangent,
        cotangent=(1.0,),
        order_law=CollisionOrderLaw(),
        histories=(CollisionPathSample(((0.4, 0.5, 0.25), (0.6, 0.4, 0.75)), (0.4, 0.6)),),
        source_boundaries={},
    )
    assert result.derivative == pytest.approx(0.0, abs=result.mean * 5e-14)


def test_detector_pixel_area_velocity_contributes_in_addition_to_ray_motion() -> None:
    result = _one(AcquisitionGeometryTangent(detector_spacing_dot=(1.0, 0.0)))
    # The detector point starts on axis: its lateral velocity has no first-order
    # inverse-square/projection change. The full detector-area Jacobian doubles
    # per mm at the 1mm pixel width, so relative derivative is exactly one.
    assert result.derivative == pytest.approx(result.mean, rel=1e-14)


def test_two_collision_integrand_matches_independent_flight_and_volume_measures() -> None:
    source, spec, medium = _scene()
    result = elastic_geometry_jvp(
        source,
        spec,
        medium,
        tangent=AcquisitionGeometryTangent(),
        cotangent=(1.0,),
        order_law=CollisionOrderLaw(),
        histories=(CollisionPathSample(((0.4, 0.5, 0.25), (0.6, 0.5, 0.75)), (0.5, 0.5)),),
        source_boundaries={},
    )
    between = math.hypot(0.2, 0.5)
    final = math.hypot(0.1, 2.25)
    projection = 2.25 / final
    last_material_length = 0.25 / projection
    depth = 0.3 * (0.25 + between + last_material_length)
    expected = (
        (1.0 / 16.0)
        * (0.2 / (4.0 * math.pi)) ** 2
        * projection
        / (between**2 * final**2)
        * math.exp(-depth)
        * 4.0
    )
    assert result.mean == pytest.approx(expected, rel=1e-14)


def test_source_footprint_jumps_are_separate_from_scatter_interior_derivatives() -> None:
    source, spec, medium = _scene()
    source = replace(source, lower_mm=(-0.1, -2.0, -1.0), extent_xy_mm=(0.5, 4.0))
    tangent = AcquisitionGeometryTangent(
        source=SourceGeometryTangent(motion=RigidMotion(translation=(1.0, 0.0, 0.0)))
    )
    boundaries = source_volume_boundaries(source, spec, tangent.object, tangent.source)
    records = {
        boundary.key: (SourceBoundaryPathSample((0.4, 0.6), (), (0.5, 0.5)),)
        for boundary in boundaries
    }
    result = elastic_geometry_jvp(
        source,
        spec,
        medium,
        tangent=tangent,
        cotangent=(1.0,),
        order_law=CollisionOrderLaw(),
        histories=(CollisionPathSample(((0.5, 0.5, 0.5),), (0.5, 0.5)),),
        source_boundaries=records,
    )
    assert result.interior == 0.0
    assert result.source_boundary != 0.0
    assert len(result.boundary_histories) == 2


def test_signed_cotangent_is_contracted_before_geometry_reduction() -> None:
    source, spec, medium = _scene()
    tangent = AcquisitionGeometryTangent(detector=RigidMotion(translation=(0.0, 0.0, 1.0)))
    records = (CollisionPathSample(((0.5, 0.5, 0.5),), (0.5, 0.5)),)
    positive = elastic_geometry_jvp(
        source,
        spec,
        medium,
        tangent=tangent,
        cotangent=(1.0,),
        order_law=CollisionOrderLaw(),
        histories=records,
        source_boundaries={},
    )
    signed = elastic_geometry_jvp(
        source,
        spec,
        medium,
        tangent=tangent,
        cotangent=(-3.0,),
        order_law=CollisionOrderLaw(),
        histories=records,
        source_boundaries={},
    )
    assert signed.mean == pytest.approx(-3.0 * positive.mean, rel=1e-14)
    assert signed.derivative == pytest.approx(-3.0 * positive.derivative, rel=1e-14)


def test_order_sampler_has_full_support_and_never_redraws_guard_failure() -> None:
    law = CollisionOrderLaw(0.5)
    assert law.sample_order(0.25) == 1
    assert law.sample_order(0.625) == 2
    assert law.sample_order(0.875) == 4
    with pytest.raises(TransportError, match="stopping"):
        CollisionOrderLaw(1.0)
    source, spec, medium = _scene()
    with pytest.raises(TransportError, match="guard"):
        elastic_geometry_jvp(
            source,
            replace(spec, max_events=1),
            medium,
            tangent=AcquisitionGeometryTangent(),
            cotangent=(1.0,),
            order_law=law,
            histories=(CollisionPathSample(((0.4, 0.4, 0.4), (0.6, 0.6, 0.6)), (0.5, 0.5)),),
            source_boundaries={},
        )


def test_partial_extinction_cannot_be_a_negative_absorption_medium() -> None:
    with pytest.raises(TransportError, match="extinction"):
        ElasticGeometryMedium((0.1,), (0.2,))
