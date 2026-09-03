"""Independent moving-face and visibility integrals, including zero pathwise scores."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.transport import TransportSpec
from dpt.transport.geometry_tangents import (
    AffineBoundary,
    GridGeometryTangent,
    RayTangent,
    RigidMotion,
    cell_segments_tangent,
    detector_hit_tangent,
    normalised_direction_jet,
)
from dpt.transport.model import MaterialGrid, OrientedDetector, PlanarDetector, TransportError
from dpt.transport.source import ParallelBeam, PointConeSource
from dpt.transport.uncollided_geometry import (
    AcquisitionGeometryTangent,
    SourceGeometryTangent,
    uncollided_geometry_boundaries,
    uncollided_geometry_jvp,
)


def _samples() -> tuple[tuple[float, float], ...]:
    return tuple(((i + 0.5) / 16, (j + 0.5) / 8) for i in range(16) for j in range(8))


def _estimate(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    tangent: AcquisitionGeometryTangent,
    rates: tuple[float, ...],
    cotangent: tuple[float, ...] = (1.0,),
):
    boundaries = uncollided_geometry_boundaries(source, spec, tangent)
    return uncollided_geometry_jvp(
        source,
        spec,
        extinction_mm_inverse=rates,
        cotangent=cotangent,
        tangent=tangent,
        source_coordinates=_samples(),
        boundary_coordinates={
            boundary.key: ((0.5 / 8), (2.5 / 8), (4.5 / 8), (6.5 / 8)) for boundary in boundaries
        },
    )


def _slab() -> TransportSpec:
    return TransportSpec(
        MaterialGrid((0.0, -1.0, 0.0), (1.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-10.0, -10.0), (20.0, 20.0), (1, 1), 3.0),
        80.0,
        "synthetic independent moving-boundary reference",
    )


@pytest.mark.parametrize("rate", (0.0, 0.2, 1.0, 30.0))
def test_parallel_side_face_translation_has_nonzero_boundary_derivative(rate: float) -> None:
    source = ParallelBeam((-0.25, -0.5, -1.0), (1.0, 1.0))
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=RigidMotion(translation=(1.0, 0.0, 0.0)))
    )
    estimate = _estimate(source, _slab(), tangent, (rate,))
    # An exact rectangular source overlap has width 3/4-theta. The probability
    # of transmission is 1-(3/4-theta)*(1-exp(-rate)), for theta near zero.
    assert estimate.mean == pytest.approx(1.0 - 0.75 * -math.expm1(-rate), abs=1e-14)
    assert estimate.interior == 0.0
    assert estimate.detector_boundary == 0.0
    assert estimate.material_boundary == pytest.approx(-math.expm1(-rate), abs=1e-14)


def test_source_translation_and_object_translation_are_opposite() -> None:
    source = ParallelBeam((-0.25, -0.5, -1.0), (1.0, 1.0))
    direction = RigidMotion(translation=(1.0, 0.0, 0.0))
    object_result = _estimate(
        source,
        _slab(),
        AcquisitionGeometryTangent(object=GridGeometryTangent(motion=direction)),
        (0.7,),
    )
    source_result = _estimate(
        source,
        _slab(),
        AcquisitionGeometryTangent(source=SourceGeometryTangent(motion=direction)),
        (0.7,),
    )
    assert source_result.derivative == -object_result.derivative


def test_internal_material_face_uses_exact_jump_of_entire_transmission() -> None:
    spec = replace(_slab(), grid=MaterialGrid((-1.0, -1.0, 0.0), (1.0, 2.0, 1.0), (1, 1, 2), 2))
    source = ParallelBeam((-0.5, -0.5, -1.0), (1.0, 1.0))
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=RigidMotion(translation=(1.0, 0.0, 0.0)))
    )
    result = _estimate(source, spec, tangent, (0.2, 1.1))
    assert result.interior == 0.0
    assert result.derivative == pytest.approx(math.exp(-0.2) - math.exp(-1.1), abs=1e-14)


def test_detector_outer_edge_visibility_term_in_vacuum() -> None:
    spec = replace(
        _slab(),
        detector=PlanarDetector((0.0, -1.0), (2.0, 2.0), (1, 1), 3.0),
        grid=MaterialGrid((10.0, -1.0, 0.0), (1.0, 2.0, 1.0), (1, 1, 1), 1),
    )
    source = ParallelBeam((-0.5, -0.4, -1.0), (2.0, 1.0))
    result = _estimate(
        source,
        spec,
        AcquisitionGeometryTangent(detector=RigidMotion(translation=(1.0, 0.0, 0.0))),
        (0.0,),
    )
    assert result.mean == 0.75
    assert result.interior == 0.0
    assert result.material_boundary == 0.0
    assert result.detector_boundary == -0.5


def test_interior_pixel_edges_contract_signed_cotangents() -> None:
    spec = replace(
        _slab(),
        detector=PlanarDetector((-1.0, -2.0), (1.0, 4.0), (2, 1), 3.0),
        grid=MaterialGrid((10.0, -1.0, 0.0), (1.0, 2.0, 1.0), (1, 1, 1), 1),
    )
    source = ParallelBeam((-0.5, -0.4, -1.0), (1.0, 1.0))
    tangent = AcquisitionGeometryTangent(detector=RigidMotion(translation=(1.0, 0.0, 0.0)))
    result = _estimate(source, spec, tangent, (0.0,), (-2.0, 3.0))
    # Shifting the internal edge right increases the -2 pixel at expense of +3.
    assert result.derivative == -5.0


def test_local_grid_spacing_derivative_changes_full_path_length() -> None:
    source = ParallelBeam((0.2, -0.4, -1.0), (0.4, 0.8))
    tangent = AcquisitionGeometryTangent(object=GridGeometryTangent(spacing_dot=(0.0, 0.0, 1.0)))
    result = _estimate(source, _slab(), tangent, (0.7,))
    assert result.material_boundary == 0.0
    assert result.interior == pytest.approx(-0.7 * math.exp(-0.7), rel=1e-14)


def test_source_extent_parameter_preserves_probability_normalisation() -> None:
    source = ParallelBeam((-0.25, -0.5, -1.0), (1.0, 1.0))
    tangent = AcquisitionGeometryTangent(source=SourceGeometryTangent(extent_dot=(1.0, 0.0)))
    result = _estimate(source, _slab(), tangent, (0.7,))
    # The covered source fraction is (extent-1/4)/extent at extent=1.
    assert result.derivative == pytest.approx(-0.25 * -math.expm1(-0.7), abs=1e-14)


def test_jvp_is_linear_in_all_independent_geometry_directions() -> None:
    source = ParallelBeam((-0.25, -0.5, -1.0), (1.0, 1.0))
    grid = GridGeometryTangent(
        motion=RigidMotion(translation=(2.0, 0.0, 0.0)), spacing_dot=(0.0, 0.0, 3.0)
    )
    source_tangent = SourceGeometryTangent(motion=RigidMotion(translation=(4.0, 0.0, 0.0)))
    both = _estimate(
        source, _slab(), AcquisitionGeometryTangent(object=grid, source=source_tangent), (0.7,)
    )
    object_only = _estimate(source, _slab(), AcquisitionGeometryTangent(object=grid), (0.7,))
    source_only = _estimate(
        source, _slab(), AcquisitionGeometryTangent(source=source_tangent), (0.7,)
    )
    assert both.derivative == pytest.approx(
        object_only.derivative + source_only.derivative, abs=1e-14
    )


def test_plane_hit_has_attached_detector_tilt_and_source_direction() -> None:
    detector = PlanarDetector((-2.0, -2.0), (1.0, 1.0), (4, 4), 3.0)
    ray = RayTangent(
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
        origin_dot=(0.2, 0.3, 0.0),
        direction_dot=(0.4, -0.2, 0.0),
    )
    result = detector_hit_tangent(ray, detector)
    assert result.column_dot == pytest.approx(1.8)
    assert result.row_dot == pytest.approx(-0.5)
    # A tilt about world y moves the hit along the detector x basis by -3 mm/rad.
    tilt = detector_hit_tangent(
        RayTangent((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        detector,
        RigidMotion(angular=(0.0, 1.0, 0.0)),
    )
    assert tilt.column_dot == pytest.approx(-3.0)


def test_object_frame_tangent_cancels_common_rigid_motion() -> None:
    pose = RigidTransform(translation_mm=(1.0, 2.0, 3.0))
    motion = RigidMotion(translation=(0.3, 0.2, -0.1), angular=(0.1, 0.2, 0.3))
    position, direction = (2.0, 3.0, -1.0), (0.0, 0.0, 1.0)
    ray = RayTangent(
        position, direction, motion.point_velocity(position), motion.direction_velocity(direction)
    )
    local = ray.in_object_frame(pose, motion)
    assert local.origin_dot == (0.0, 0.0, 0.0)
    assert local.direction_dot == (0.0, 0.0, 0.0)


def test_segment_tangents_keep_transverse_object_rotation() -> None:
    direction = (0.6, 0.0, 0.8)
    ray = RayTangent((-1.0, 0.0, -1.0), direction)
    grid = MaterialGrid((-2.0, -2.0, 0.0), (4.0, 4.0, 1.0), (1, 1, 1), 1)
    segments = cell_segments_tangent(
        ray,
        grid,
        tangent=GridGeometryTangent(motion=RigidMotion(angular=(0.0, 1.0, 0.0))),
        maximum_distance_mm=10.0,
    )
    # For a rotated infinite slab the path length is 1/(cos(theta)*0.8 +
    # sin(theta)*0.6); at theta=0 its derivative is -0.6/0.8**2.
    assert len(segments) == 1
    assert segments[0].length_mm == pytest.approx(1.25)
    assert segments[0].length_dot == pytest.approx(-0.6 / 0.8**2)


def test_normalisation_jet_is_not_a_detached_direction() -> None:
    direction, tangent = normalised_direction_jet((3.0, 0.0, 4.0), (1.0, 0.0, 0.0))
    assert direction == pytest.approx((0.6, 0.0, 0.8))
    assert tangent == pytest.approx((0.128, 0.0, -0.096))


def test_affine_diagonal_boundary_includes_coarea_jacobian() -> None:
    boundary = AffineBoundary("independent", (1.0, 1.0, -1.0), (0.0, 0.0, 1.0))
    point, weight = boundary.sample(0.25) or pytest.fail("boundary is present")
    assert sum(point) == 1.0
    assert weight == pytest.approx(1.0)


def test_cone_detector_edge_is_sampled_in_original_launch_area() -> None:
    geometry = DetectorGeometry(
        (0.0, 0.0, -1.0), (0.0, 0.0, 3.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 2.0), (1, 1)
    )
    source = PointConeSource(geometry)
    # Measurement support differs from the explicit source launch area. Its
    # left edge cuts the launch area halfway; translating it right loses 1/2/mm.
    detector = OrientedDetector(replace(geometry, origin_mm=(1.0, 0.0, 3.0)))
    spec = replace(
        _slab(),
        detector=detector,
        grid=MaterialGrid((10.0, 10.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
    )
    result = _estimate(
        source,
        spec,
        AcquisitionGeometryTangent(detector=RigidMotion(translation=(1.0, 0.0, 0.0))),
        (0.0,),
    )
    assert result.derivative == pytest.approx(-0.5, abs=1e-14)


def test_missing_boundary_or_coincident_support_cannot_return_zero() -> None:
    source = ParallelBeam((-0.25, -0.5, -1.0), (1.0, 1.0))
    with pytest.raises(TransportError, match="keys"):
        uncollided_geometry_jvp(
            source,
            _slab(),
            extinction_mm_inverse=(0.7,),
            cotangent=(1.0,),
            tangent=AcquisitionGeometryTangent(),
            source_coordinates=_samples(),
            boundary_coordinates={},
        )
    with pytest.raises(TransportError, match="source support"):
        AffineBoundary("coincident", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)).endpoints()
    with pytest.raises(TransportError, match="directional boundary"):
        uncollided_geometry_boundaries(
            source,
            _slab(),
            AcquisitionGeometryTangent(
                object=GridGeometryTangent(motion=RigidMotion(angular=(0.0, 1.0, 0.0)))
            ),
        )
