"""Detached analogue-prefix densities and original-history grouping."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from dpt.transport.elastic_geometry import ElasticGeometryMedium
from dpt.transport.elastic_geometry_prefix import (
    GeometryBoundaryPrefixHistory,
    GeometryEventDraw,
    GeometryPrefixDraws,
    elastic_prefix_geometry_jvp,
    sample_elastic_geometry_prefix,
)
from dpt.transport.geometry_parameters import GeometryParameter, elastic_prefix_geometry_vjp
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
        "synthetic extinction-adapted prefix reference",
    )
    return source, spec, ElasticGeometryMedium((0.3,), (0.2,))


def _event(*, survival: float = 0.9, flight: float = 0.5) -> GeometryEventDraw:
    return GeometryEventDraw(survival, 0.5, flight, ((0.75, 0.5, 0.25),))


def test_absorbed_reference_path_still_scores_its_conditional_next_event() -> None:
    source, spec, medium = _scene()
    history = sample_elastic_geometry_prefix(
        source, spec, medium, GeometryPrefixDraws((0.5, 0.5, 0.5), (0.5, 0.5), (_event(),))
    )
    assert history.terminal == "absorbed"
    assert len(history.vertices) == 1
    result = elastic_prefix_geometry_jvp(
        source,
        spec,
        medium,
        tangent=AcquisitionGeometryTangent(),
        cotangent=(1.0,),
        histories=(history,),
        source_boundaries={},
    )
    expected = 1.0 / 16.0 * 0.2 / (4.0 * math.pi) * math.exp(-0.3) / 2.5**2
    assert result.mean == pytest.approx(expected, rel=1e-14)
    assert result.original_histories == 1
    assert result.prefixes == 1


def test_extinction_adapted_transition_has_complete_survival_and_flight_density() -> None:
    source, spec, medium = _scene()
    history = sample_elastic_geometry_prefix(
        source,
        spec,
        medium,
        GeometryPrefixDraws(
            (0.5, 0.5, 0.5),
            (0.5, 0.5),
            (
                _event(survival=0.1, flight=math.exp(-0.03)),
                _event(),
            ),
        ),
    )
    assert len(history.vertices) == 2
    assert history.transition_log_densities == pytest.approx(
        (math.log((0.2 / 0.3) * (1.0 / (4.0 * math.pi)) * 0.3 * math.exp(-0.03) / 0.1**2),),
        abs=1e-14,
    )
    result = elastic_prefix_geometry_jvp(
        source,
        spec,
        medium,
        tangent=AcquisitionGeometryTangent(),
        cotangent=(1.0,),
        histories=(history,),
        source_boundaries={},
    )
    # Independently telescope the proposal: the intermediate phase, extinction,
    # attenuation and inverse-square factors all cancel. Both NEE prefixes share
    # the same original observation and denominator.
    x = -math.sqrt(0.75) * 0.1
    z = 0.55
    final_distance = math.hypot(x, 3.0 - z)
    projection = (3.0 - z) / final_distance
    last_depth = 0.3 * (1.0 - z) / projection
    first = 1.0 / 16.0 * 0.2 / (4.0 * math.pi) * math.exp(-0.3) / 2.5**2
    second = (
        1.0
        / 16.0
        * 0.2
        / (4.0 * math.pi)
        * projection
        / final_distance**2
        * (math.exp(-0.15 - last_depth))
    )
    assert result.mean == pytest.approx(first + second, rel=2e-14)
    assert result.original_histories == 1
    assert result.prefixes == 2
    assert len(result.original_observations) == 1


def test_every_prefix_attaches_full_common_rigid_motion() -> None:
    source, spec, medium = _scene()
    history = sample_elastic_geometry_prefix(
        source,
        spec,
        medium,
        GeometryPrefixDraws(
            (0.5, 0.5, 0.5),
            (0.5, 0.5),
            (
                _event(survival=0.1, flight=math.exp(-0.03)),
                _event(),
            ),
        ),
    )
    motion = RigidMotion(translation=(0.1, 0.2, 0.3), angular=(0.2, -0.1, 0.1))
    result = elastic_prefix_geometry_jvp(
        source,
        spec,
        medium,
        tangent=AcquisitionGeometryTangent(
            object=GridGeometryTangent(motion=motion),
            source=SourceGeometryTangent(motion=motion),
            detector=motion,
        ),
        cotangent=(1.0,),
        histories=(history,),
        source_boundaries={},
    )
    assert result.derivative == pytest.approx(0.0, abs=result.mean * 1e-13)


def test_prefix_draw_budget_and_event_guard_fail_before_returning_a_partial_estimate() -> None:
    source, spec, medium = _scene()
    draws = GeometryPrefixDraws(
        (0.5, 0.5, 0.5), (0.5, 0.5), (_event(survival=0.1, flight=math.exp(-0.03)),)
    )
    with pytest.raises(TransportError, match="draw budget"):
        sample_elastic_geometry_prefix(source, spec, medium, draws)
    with pytest.raises(TransportError, match="event guard"):
        sample_elastic_geometry_prefix(source, replace(spec, max_events=1), medium, draws)


def test_source_boundary_continuations_are_bound_and_grouped_by_original_history() -> None:
    source, spec, medium = _scene()
    source = replace(source, lower_mm=(-0.1, -2.0, -1.0), extent_xy_mm=(0.5, 4.0))
    tangent = AcquisitionGeometryTangent(
        source=SourceGeometryTangent(motion=RigidMotion(translation=(1.0, 0.0, 0.0)))
    )
    draws = GeometryPrefixDraws((0.5, 0.5, 0.5), (0.5, 0.5), (_event(),))
    interior = sample_elastic_geometry_prefix(source, spec, medium, draws)
    boundaries = source_volume_boundaries(source, spec, tangent.object, tangent.source)
    records = {}
    for boundary in boundaries:
        point, _ = boundary.sample((0.4, 0.6)) or pytest.fail("source edge is present")
        history = sample_elastic_geometry_prefix(source, spec, medium, draws, first_vertex=point)
        records[boundary.key] = (GeometryBoundaryPrefixHistory((0.4, 0.6), history),)
    result = elastic_prefix_geometry_jvp(
        source,
        spec,
        medium,
        tangent=tangent,
        cotangent=(1.0,),
        histories=(interior,),
        source_boundaries=records,
    )
    assert result.interior == 0.0
    assert result.source_boundary != 0.0
    assert result.boundary_original_histories == (("source:0", 1), ("source:1", 1))
    wrong = {
        **records,
        "source:0": (GeometryBoundaryPrefixHistory((0.3, 0.6), records["source:0"][0].history),),
    }
    with pytest.raises(TransportError, match="sampled point"):
        elastic_prefix_geometry_jvp(
            source,
            spec,
            medium,
            tangent=tangent,
            cotangent=(1.0,),
            histories=(interior,),
            source_boundaries=wrong,
        )


def test_source_support_at_discontinuous_interface_has_no_two_sided_derivative() -> None:
    source = ParallelBeam((0.5, -1.0, -1.0), (1.0, 3.0))
    spec = TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1, 1, 2), 2),
        PlanarDetector((-1.0, -1.0), (3.0, 3.0), (1, 1), 2.0),
        80.0,
        "equal extinction, unequal scattering across interface",
    )
    medium = ElasticGeometryMedium((2.0, 2.0), (1.0, 2.0))
    history = sample_elastic_geometry_prefix(
        source,
        spec,
        medium,
        GeometryPrefixDraws((0.75, 0.5, 0.5), (0.5, 0.5), (_event(flight=0.0001),)),
    )
    with pytest.raises(TransportError, match="discontinuous material interface"):
        elastic_prefix_geometry_jvp(
            source,
            spec,
            medium,
            tangent=AcquisitionGeometryTangent(
                source=SourceGeometryTangent(motion=RigidMotion(translation=(1.0, 0.0, 0.0)))
            ),
            cotangent=(1.0,),
            histories=(history,),
            source_boundaries={},
        )


def test_grazing_object_rotation_is_not_reported_as_a_pathwise_zero() -> None:
    source, spec, medium = _scene()
    history = sample_elastic_geometry_prefix(
        source, spec, medium, GeometryPrefixDraws((0.5, 0.5, 0.5), (0.5, 0.5), (_event(),))
    )
    with pytest.raises(TransportError, match="grazing material-face"):
        elastic_prefix_geometry_jvp(
            source,
            spec,
            medium,
            tangent=AcquisitionGeometryTangent(
                object=GridGeometryTangent(motion=RigidMotion(angular=(0.0, 1.0, 0.0)))
            ),
            cotangent=(1.0,),
            histories=(history,),
            source_boundaries={},
        )


def test_cube_absorption_rotation_has_opposite_one_sided_shape_derivatives() -> None:
    mu, epsilon = 0.7, 1e-6
    coefficient = 1.0 + math.exp(-mu) - 2.0 * -math.expm1(-mu) / mu

    def cross_section(theta: float) -> float:
        cosine, sine = math.cos(theta), abs(math.sin(theta))
        depth = mu / cosine
        return (cosine - sine) * -math.expm1(-depth) + 2.0 * sine * (
            1.0 - -math.expm1(-depth) / depth
        )

    at_zero = cross_section(0.0)
    assert (cross_section(epsilon) - at_zero) / epsilon == pytest.approx(coefficient, rel=1e-5)
    assert (cross_section(-epsilon) - at_zero) / -epsilon == pytest.approx(-coefficient, rel=1e-5)


def test_retained_transition_density_cannot_be_changed_without_changing_the_reference_law() -> None:
    source, spec, medium = _scene()
    history = sample_elastic_geometry_prefix(
        source,
        spec,
        medium,
        GeometryPrefixDraws(
            (0.5, 0.5, 0.5),
            (0.5, 0.5),
            (
                _event(survival=0.1, flight=math.exp(-0.03)),
                _event(),
            ),
        ),
    )
    forged = replace(history, transition_log_densities=(history.transition_log_densities[0] + 0.1,))
    with pytest.raises(TransportError, match="physical transition"):
        elastic_prefix_geometry_jvp(
            source,
            spec,
            medium,
            tangent=AcquisitionGeometryTangent(),
            cotangent=(1.0,),
            histories=(forged,),
            source_boundaries={},
        )


def test_named_prefix_vjp_is_the_transpose_of_joint_object_source_detector_jvp() -> None:
    source, spec, medium = _scene()
    spec = replace(spec, detector=PlanarDetector((-0.5, -0.5), (0.5, 1.0), (2, 1), 3.0))
    histories = tuple(
        sample_elastic_geometry_prefix(
            source,
            spec,
            medium,
            GeometryPrefixDraws(
                (0.4, 0.6, 0.5),
                (u, 0.4),
                (
                    _event(survival=0.1, flight=math.exp(-0.03)),
                    _event(),
                ),
            ),
        )
        for u in (0.25, 0.75)
    )
    parameters = (
        GeometryParameter("object", "translation", 2),
        GeometryParameter("source", "extent", 0),
        GeometryParameter("detector", "translation", 0),
    )
    seed, coefficients = (1.25, -0.75), (0.3, 0.2, 0.7)
    vjp = elastic_prefix_geometry_vjp(
        source,
        spec,
        medium,
        parameters=parameters,
        cotangent=seed,
        histories=histories,
        source_boundaries={},
    )
    jvp = elastic_prefix_geometry_jvp(
        source,
        spec,
        medium,
        tangent=AcquisitionGeometryTangent(
            object=GridGeometryTangent(motion=RigidMotion(translation=(0.0, 0.0, 0.3))),
            source=SourceGeometryTangent(extent_dot=(0.2, 0.0)),
            detector=RigidMotion(translation=(0.7, 0.0, 0.0)),
        ),
        cotangent=seed,
        histories=histories,
        source_boundaries={},
    )
    assert sum(a * b for a, b in zip(vjp.gradient, coefficients, strict=True)) == pytest.approx(
        jvp.derivative, rel=1e-13, abs=1e-20
    )
    assert all(column.original_histories == 2 and column.prefixes == 4 for column in vjp.columns)
