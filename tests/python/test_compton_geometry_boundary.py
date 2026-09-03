"""Analytic cone areas and independent planar coarea checks for energy support."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from itertools import pairwise
from typing import cast

import pytest

from dpt.geometry import DetectorGeometry, Vector3, cross, dot
from dpt.transport.compton_geometry_boundary import (
    ComptonEnergyBoundarySample,
    FreeComptonEnergyBoundary,
)
from dpt.transport.geometry_tangents import RayTangent, RigidMotion
from dpt.transport.model import (
    ELECTRON_REST_ENERGY_KEV,
    OrientedDetector,
    PlanarDetector,
    TransportError,
)

E = 100.0
C = 0.8
H = 2.0
L = 1.0 / (1.0 / E + (1.0 - C) / ELECTRON_REST_ENERGY_KEV)


def _detector() -> PlanarDetector:
    return PlanarDetector((-3.0, -3.0), (3.0, 3.0), (2, 2), H)


def _boundary(
    *, incident_energy_dot: float = 0.0, threshold_energy_dot: float = 0.0
) -> FreeComptonEnergyBoundary:
    return FreeComptonEnergyBoundary(
        RayTangent((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        E,
        L,
        incident_energy_dot=incident_energy_dot,
        threshold_energy_dot=threshold_energy_dot,
    )


def _integrate(
    boundary: FreeComptonEnergyBoundary,
    detector: PlanarDetector | OrientedDetector,
    motion: RigidMotion | None = None,
    *,
    spacing_dot: tuple[float, float] = (0.0, 0.0),
    jump: Callable[[ComptonEnergyBoundarySample], float] = lambda _: 1.0,
    count: int = 128,
) -> float:
    contributions = []
    for index in range(count):
        sample = boundary.sample((index + 0.5) / count, detector, motion, spacing_dot=spacing_dot)
        contributions.append(0.0 if sample is None else sample.world_area_weight * jump(sample))
    return math.fsum(contributions) / count


def test_detector_normal_translation_has_exact_disk_area_boundary_derivative() -> None:
    # The accepted cone is a disk of area pi*h²*(1/c²-1), so dA/dh is exact.
    derivative = _integrate(_boundary(), _detector(), RigidMotion(translation=(0.0, 0.0, 1.0)))
    expected = 2.0 * math.pi * H * (1.0 / C**2 - 1.0)
    assert derivative == pytest.approx(expected, rel=2e-14, abs=2e-14)


def test_object_normal_translation_has_opposite_disk_boundary_derivative() -> None:
    boundary = replace(
        _boundary(),
        incoming=RayTangent(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            origin_dot=(0.0, 0.0, 1.0),
        ),
    )
    derivative = _integrate(boundary, _detector())
    assert derivative == pytest.approx(-2.0 * math.pi * H * (1.0 / C**2 - 1.0), rel=2e-14)


@pytest.mark.parametrize("energy_dot,threshold_dot", ((1.0, 0.0), (0.0, 1.0), (1.2, -0.7)))
def test_incident_and_threshold_energy_jets_match_analytic_disk_area(
    energy_dot: float,
    threshold_dot: float,
) -> None:
    boundary = _boundary(incident_energy_dot=energy_dot, threshold_energy_dot=threshold_dot)
    cosine_dot = ELECTRON_REST_ENERGY_KEV * (threshold_dot / L**2 - energy_dot / E**2)
    expected = -2.0 * math.pi * H**2 * cosine_dot / C**3
    assert _integrate(boundary, _detector()) == pytest.approx(expected, rel=2e-14, abs=2e-14)


def test_incoming_source_angle_jet_has_exact_first_moment_boundary_derivative() -> None:
    boundary = replace(
        _boundary(),
        incoming=RayTangent(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            direction_dot=(1.0, 0.0, 0.0),
        ),
    )
    # The tilted cone's ellipse centre is h*sin(theta)*cos(theta)/(c²-sin²(theta)).
    # Its first x moment has derivative (disk area)*h/c² at theta=0.
    expected = math.pi * H**3 * (1.0 - C**2) / C**4
    assert _integrate(
        boundary, _detector(), jump=lambda sample: sample.endpoint.point_mm[0]
    ) == pytest.approx(expected, rel=2e-14)
    assert _integrate(boundary, _detector()) == pytest.approx(0.0, abs=2e-14)


def test_object_and_detector_lateral_jets_have_opposite_first_moment_terms() -> None:
    area = math.pi * H**2 * (1.0 / C**2 - 1.0)
    boundary = replace(
        _boundary(),
        incoming=RayTangent(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            origin_dot=(1.0, 0.0, 0.0),
        ),
    )

    def moment(sample: ComptonEnergyBoundarySample) -> float:
        return sample.endpoint.point_mm[0]

    assert _integrate(boundary, _detector(), jump=moment) == pytest.approx(area, rel=2e-14)
    assert _integrate(
        _boundary(), _detector(), RigidMotion(translation=(1.0, 0.0, 0.0)), jump=moment
    ) == pytest.approx(-area, rel=2e-14)


def test_common_rigid_motion_cancels_full_boundary_jet() -> None:
    motion = RigidMotion(
        translation=(0.4, -0.2, 0.3), angular=(0.2, -0.1, 0.3), pivot_mm=(1.0, -1.0, 0.5)
    )
    boundary = replace(
        _boundary(),
        incoming=RayTangent(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            origin_dot=motion.point_velocity((0.0, 0.0, 0.0)),
            direction_dot=motion.direction_velocity((0.0, 0.0, 1.0)),
        ),
    )
    for index in range(16):
        sample = boundary.sample((index + 0.5) / 16, _detector(), motion)
        assert sample is not None
        assert sample.world_area_weight == pytest.approx(0.0, abs=2e-14)
        assert sample.levelset_dot_kev == pytest.approx(0.0, abs=2e-14)


def test_detector_rotation_has_opposite_incoming_angle_moment_term() -> None:
    expected = -math.pi * H**3 * (1.0 - C**2) / C**4
    assert _integrate(
        _boundary(),
        _detector(),
        RigidMotion(angular=(0.0, 1.0, 0.0)),
        jump=lambda sample: sample.endpoint.point_mm[0],
    ) == pytest.approx(expected, rel=2e-14)


def test_reference_area_weight_includes_detector_jacobian_exactly_once() -> None:
    detector = _detector()
    sample = _boundary().sample(0.19, detector, spacing_dot=(0.3, 0.6))
    assert sample is not None
    assert sample.reference_area_weight * sample.endpoint.area_jacobian == pytest.approx(
        sample.world_area_weight, rel=2e-16
    )
    area = math.pi * H**2 * (1.0 / C**2 - 1.0)
    # Expanding the detector chart moves the energy boundary inward in chart
    # coordinates, cancelling the smooth area-Jacobian derivative exactly.
    derivative = _integrate(_boundary(), detector, spacing_dot=(0.3, 0.6))
    assert derivative == pytest.approx(-area * (0.3 / 3.0 + 0.6 / 3.0), rel=2e-14)


def test_half_open_pixels_and_signed_continuation_jump_use_original_draw_count() -> None:
    boundary = _boundary(incident_energy_dot=1.0)
    # Only one quadrant is retained, so the full-azimuth estimate is one quarter
    # of the disk derivative rather than the mean conditioned on accepted arcs.
    detector = PlanarDetector((0.0, 0.0), (3.0, 3.0), (1, 1), H)
    samples = [boundary.sample((index + 0.5) / 128, detector) for index in range(128)]
    assert sum(sample is not None for sample in samples) == 32
    expected = 2.0 * math.pi * H**2 * ELECTRON_REST_ENERGY_KEV / (C**3 * E**2)
    assert _integrate(boundary, detector) == pytest.approx(expected / 4.0, rel=2e-14)
    assert _integrate(boundary, detector, jump=lambda _: -2.7) == pytest.approx(
        -2.7 * expected / 4.0, rel=2e-14
    )


def test_two_sided_continuation_jump_can_change_law_across_threshold() -> None:
    boundary = _boundary(incident_energy_dot=1.0)
    higher, lower = 2.3, -0.4
    expected = _integrate(boundary, _detector()) * (higher - lower)
    assert _integrate(boundary, _detector(), jump=lambda _: higher - lower) == pytest.approx(
        expected, rel=2e-15
    )


def test_fixed_detector_endpoint_energy_jet_obeys_chain_rule() -> None:
    boundary = _boundary(incident_energy_dot=0.7, threshold_energy_dot=0.2)
    sample = boundary.sample(0.17, _detector(), RigidMotion(translation=(0.0, 0.0, 0.9)))
    assert sample is not None
    energy = sample.outgoing_energy_kev
    cosine_dot = dot(boundary.incoming.direction, sample.outgoing.direction_dot)
    expected = energy**2 * (0.7 / E**2 + cosine_dot / ELECTRON_REST_ENERGY_KEV)
    assert energy == pytest.approx(L, rel=2e-16)
    assert sample.outgoing_energy_dot == pytest.approx(expected, rel=2e-15)
    assert sample.levelset_dot_kev == pytest.approx(expected - 0.2, rel=2e-15)


def test_oriented_spacing_uses_existing_first_pixel_centre_anchor() -> None:
    geometry = DetectorGeometry(
        (0.0, 0.0, -1.0), (-1.5, -1.5, H), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (3.0, 3.0), (2, 2)
    )
    detector = OrientedDetector(geometry)
    sample = _boundary().sample(0.19, detector, spacing_dot=(0.3, 0.6))
    assert sample is not None
    x, y, _ = sample.endpoint.point_mm
    assert sample.endpoint.point_dot == pytest.approx(((x + 1.5) * 0.1, (y + 1.5) * 0.2, 0.0))


def test_oblique_plane_coarea_matches_independent_arclength_and_gradient() -> None:
    # This reference differentiates the cone-plane curve in azimuth and uses
    # |dy/dphi|/|grad_plane cosine|, independently of the production solid angle.
    angle = 0.31
    u: Vector3 = (math.cos(angle), 0.0, -math.sin(angle))
    v: Vector3 = (0.0, 1.0, 0.0)
    geometry = DetectorGeometry((0.0, 0.0, -1.0), (-2.0, -2.0, 3.0), u, v, (1.0, 1.0), (6, 6))
    detector = OrientedDetector(geometry)
    boundary = _boundary(incident_energy_dot=1.0)
    accepted = 0
    for coordinate in (0.07, 0.19, 0.39, 0.63, 0.87):
        sample = boundary.sample(coordinate, detector)
        if sample is None:
            continue
        accepted += 1
        d = sample.outgoing.direction
        r = math.dist(boundary.incoming.origin_mm, sample.endpoint.point_mm)
        k = boundary.incoming.direction
        # d_phi = k cross d for the right-handed cone basis.
        d_phi = cross(k, d)
        projection = dot(detector.normal, d)
        distance_phi = -r * dot(detector.normal, d_phi) / projection
        curve_tangent = tuple(distance_phi * d[i] + r * d_phi[i] for i in range(3))
        gradient = cast(Vector3, tuple((k[i] - C * d[i]) / r for i in range(3)))
        normal_component = dot(gradient, detector.normal)
        plane_gradient = tuple(
            gradient[i] - normal_component * detector.normal[i] for i in range(3)
        )
        ratio = math.hypot(*curve_tangent) / math.hypot(*plane_gradient)
        assert ratio == pytest.approx(r**2 / projection, rel=8e-15)
        assert sample.world_area_weight == pytest.approx(
            -2.0 * math.pi * ratio * boundary.cosine_dot, rel=8e-15
        )
    assert accepted >= 3


def test_tilted_cone_ellipse_area_and_first_moment_derivatives() -> None:
    theta = 0.3
    sine, cosine = math.sin(theta), math.cos(theta)
    boundary = replace(
        _boundary(),
        incoming=RayTangent(
            (0.0, 0.0, 0.0),
            (sine, 0.0, cosine),
            direction_dot=(cosine, 0.0, -sine),
        ),
    )
    detector = PlanarDetector((-20.0, -20.0), (20.0, 20.0), (2, 2), H)
    a = C**2 - sine**2
    area = math.pi * H**2 * C * (1.0 - C**2) / a**1.5
    area_dot = area * 3.0 * sine * cosine / a
    centre = H * sine * cosine / a
    centre_dot = H * (cosine**2 - sine**2) / a + 2.0 * H * sine**2 * cosine**2 / a**2
    assert _integrate(boundary, detector, count=256) == pytest.approx(area_dot, rel=3e-14)
    assert _integrate(
        boundary, detector, count=256, jump=lambda sample: sample.endpoint.point_mm[0]
    ) == pytest.approx(area_dot * centre + area * centre_dot, rel=3e-14)


def _gauss16(function: Callable[[float], float], left: float, right: float) -> float:
    # Positive abscissas and weights of the degree-31 exact Gauss-Legendre rule.
    nodes = (
        0.09501250983763744,
        0.2816035507792589,
        0.4580167776572274,
        0.6178762444026438,
        0.755404408355003,
        0.8656312023878318,
        0.9445750230732326,
        0.9894009349916499,
    )
    weights = (
        0.1894506104550685,
        0.1826034150449236,
        0.1691565193950025,
        0.1495959888165767,
        0.1246289712555339,
        0.09515851168249278,
        0.06225352393864789,
        0.027152459411754095,
    )
    middle, half = (left + right) / 2.0, (right - left) / 2.0
    return half * math.fsum(
        weight * (function(middle - half * node) + function(middle + half * node))
        for node, weight in zip(nodes, weights, strict=True)
    )


def test_signed_pixel_cotangents_match_independent_clipped_ellipse_quadrature() -> None:
    theta = 0.3
    sine, cosine = math.sin(theta), math.cos(theta)
    boundary = replace(
        _boundary(),
        incoming=RayTangent(
            (0.0, 0.0, 0.0),
            (sine, 0.0, cosine),
            direction_dot=(cosine, 0.0, -sine),
        ),
    )
    detector = PlanarDetector((-20.0, -20.0), (20.0, 20.0), (2, 2), H)
    cotangents = (-0.4, 1.2, -2.3, 0.7)
    a = C**2 - sine**2
    area = math.pi * H**2 * C * (1.0 - C**2) / a**1.5
    area_dot = area * 3.0 * sine * cosine / a
    alpha = sine * cosine / (C * math.sqrt(1.0 - C**2))
    alpha_dot = (cosine**2 - sine**2) / (C * math.sqrt(1.0 - C**2))
    right_fraction = 0.5 + (math.asin(alpha) + alpha * math.sqrt(1.0 - alpha**2)) / math.pi
    fraction_dot = 2.0 * math.sqrt(1.0 - alpha**2) * alpha_dot / math.pi
    right_dot = 0.5 * (area_dot * right_fraction + area * fraction_dot)
    left_dot = 0.5 * (area_dot * (1.0 - right_fraction) - area * fraction_dot)
    expected = left_dot * (cotangents[0] + cotangents[2]) + right_dot * (
        cotangents[1] + cotangents[3]
    )
    crossing = math.asin(C * sine / (math.sqrt(1.0 - C**2) * cosine))
    breaks = (0.0, crossing, math.pi / 2.0, math.pi - crossing, 3.0 * math.pi / 2.0, 2.0 * math.pi)

    def integrand(phi: float) -> float:
        sample = boundary.sample(phi / (2.0 * math.pi), detector)
        assert sample is not None
        return sample.world_area_weight * cotangents[sample.endpoint.pixel] / (2.0 * math.pi)

    coarse = math.fsum(_gauss16(integrand, left, right) for left, right in pairwise(breaks))
    fine = math.fsum(
        _gauss16(integrand, left, (left + right) / 2.0)
        + _gauss16(integrand, (left + right) / 2.0, right)
        for left, right in pairwise(breaks)
    )
    assert fine == pytest.approx(expected, rel=3e-14, abs=3e-14)
    assert coarse == pytest.approx(fine, rel=3e-13, abs=3e-13)


@pytest.mark.parametrize("threshold", (L / 10.0, E * 2.0))
def test_strictly_inactive_thresholds_have_no_boundary(threshold: float) -> None:
    assert replace(_boundary(), threshold_energy_kev=threshold).sample(0.17, _detector()) is None


def test_active_forward_endpoint_requires_one_sided_derivative() -> None:
    boundary = replace(_boundary(), threshold_energy_kev=E, threshold_energy_dot=1.0)
    with pytest.raises(TransportError, match="one-sided"):
        boundary.sample(0.17, _detector())
    assert replace(boundary, threshold_energy_dot=0.0).sample(0.17, _detector()) is None


@pytest.mark.parametrize(
    "source_motion,detector_motion",
    (
        ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ),
)
def test_moving_right_angle_cone_coincident_with_pixel_edge_requires_one_side(
    source_motion: Vector3,
    detector_motion: Vector3,
) -> None:
    mass = ELECTRON_REST_ENERGY_KEV
    boundary = FreeComptonEnergyBoundary(
        RayTangent(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            origin_dot=source_motion,
        ),
        mass,
        mass / 2.0,
    )
    assert boundary.cosine == 0.0
    with pytest.raises(TransportError, match="coincides"):
        boundary.sample(0.17, _detector(), RigidMotion(translation=detector_motion))


def test_stationary_right_angle_pixel_edge_has_zero_boundary_motion() -> None:
    mass = ELECTRON_REST_ENERGY_KEV
    boundary = FreeComptonEnergyBoundary(
        RayTangent(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            origin_dot=(1.0, 0.0, 0.0),
        ),
        mass,
        mass / 2.0,
    )
    sample = boundary.sample(0.72, _detector(), RigidMotion(translation=(1.0, 0.0, 0.0)))
    assert sample is not None
    assert sample.world_area_weight == 0.0


def test_coincident_collision_plane_is_rejected_without_epsilon() -> None:
    with pytest.raises(TransportError, match="lies in"):
        _boundary().sample(0.17, replace(_detector(), z_mm=0.0))


def test_backward_detector_and_empty_support_are_null_draws() -> None:
    assert _boundary().sample(0.17, replace(_detector(), z_mm=-1.0)) is None
    assert _boundary().sample(0.17, replace(_detector(), lower_xy_mm=(100.0, 100.0))) is None


@pytest.mark.parametrize("draw", (0.0, 1.0, -0.1, math.nan, math.inf))
def test_draw_domain_is_explicit(draw: float) -> None:
    with pytest.raises(TransportError, match="strictly inside"):
        _boundary().sample(draw, _detector())


@pytest.mark.parametrize(
    "field,value",
    (
        ("incident_energy_kev", 0.0),
        ("threshold_energy_kev", -1.0),
        ("incident_energy_dot", math.inf),
        ("threshold_energy_dot", math.nan),
    ),
)
def test_energy_domain_rejects_invalid_fields(field: str, value: float) -> None:
    with pytest.raises(TransportError):
        replace(_boundary(), **{field: value})
