"""Independent optical-depth expectation derivatives and exact vacuum-gap jumps."""

from __future__ import annotations

import cmath
import math

import pytest

from dpt.transport.geometry_flights import (
    material_box_interval,
    optical_boundary_observation,
    optical_flight_tangent,
)
from dpt.transport.geometry_tangents import GridGeometryTangent, RayTangent, RigidMotion
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec


def _scene(cells: int = 2) -> TransportSpec:
    return TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (cells, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 4.0),
        80.0,
        "independent piecewise exponential flight expectation",
    )


def _exact_expectation(
    height: complex, rates: tuple[float, ...], values: tuple[float, ...], escape: float
) -> complex:
    depth, total = 0j, 0j
    for index, (rate, response) in enumerate(zip(rates, values, strict=True)):
        if rate > 0.0:
            increment = rate * height
            mass = 1.0 - cmath.exp(-increment)
            moment = 1.0 - (1.0 + increment) * cmath.exp(-increment)
            total += response * cmath.exp(-depth) * (index * height * mass + moment / rate)
            depth += increment
    return total + escape * cmath.exp(-depth)


@pytest.mark.parametrize(
    "rates,values", (((0.3, 0.8), (1.2, -0.7)), ((0.3, 0.0, 0.8), (1.2, 0.0, -0.7)))
)
def test_interface_jumps_plus_interior_jet_equal_independent_exact_flight_expectation(
    rates: tuple[float, ...],
    values: tuple[float, ...],
) -> None:
    spec = _scene(len(rates))
    ray = RayTangent((0.0, 0.0, -1.0), (0.0, 0.0, 1.0))
    tangent = GridGeometryTangent(spacing_dot=(0.0, 0.0, 1.0))
    record = optical_flight_tangent(
        ray, spec, extinction_mm_inverse=rates, optical_depth=0.15, tangent=tangent
    )
    escape, interior, depth = 0.4, 0.0, 0.0
    for index, (rate, response) in enumerate(zip(rates, values, strict=True)):
        if rate > 0.0:
            probability = math.exp(-depth) * -math.expm1(-rate)
            interior += probability * response * (index - depth / rate)
            depth += rate
    support = 0.0
    for boundary in record.boundaries:
        left = boundary.before.cell
        assert left is not None
        before = values[left] * boundary.before.ray.origin_mm[2]
        right = boundary.after.cell
        after = escape if right is None else values[right] * boundary.after.ray.origin_mm[2]
        support += optical_boundary_observation(boundary, before_value=before, after_value=after)
    independent = _exact_expectation(1.0 + 1e-20j, rates, values, escape).imag / 1e-20
    assert interior + support == pytest.approx(independent, rel=2e-14, abs=2e-14)
    if 0.0 in rates:
        assert record.boundaries[0].before.ray.origin_mm[2] == 1.0
        assert record.boundaries[0].after.ray.origin_mm[2] == 2.0


def test_collision_position_jet_and_full_common_translation_match_analytic_flight() -> None:
    ray = RayTangent((0.0, 0.0, -1.0), (0.0, 0.0, 1.0))
    record = optical_flight_tangent(
        ray,
        _scene(),
        extinction_mm_inverse=(0.3, 0.8),
        optical_depth=0.5,
        tangent=GridGeometryTangent(spacing_dot=(0.0, 0.0, 1.0)),
    )
    assert record.cell == 1
    assert record.ray.origin_mm == pytest.approx((0.0, 0.0, 1.25))
    assert record.ray.origin_dot == pytest.approx((0.0, 0.0, 0.625))
    motion = RigidMotion(translation=(0.3, -0.2, 0.7))
    common = optical_flight_tangent(
        RayTangent(ray.origin_mm, ray.direction, motion.translation),
        _scene(),
        extinction_mm_inverse=(0.3, 0.8),
        optical_depth=0.5,
        tangent=GridGeometryTangent(motion=motion),
    )
    assert common.ray.origin_dot == motion.translation
    assert common.total_optical_depth_dot == 0.0
    assert all(boundary.weight == 0.0 for boundary in common.boundaries)


def test_escape_still_retains_conditional_interface_contributions() -> None:
    record = optical_flight_tangent(
        RayTangent((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        _scene(),
        extinction_mm_inverse=(0.3, 0.8),
        optical_depth=9.0,
        tangent=GridGeometryTangent(spacing_dot=(0.0, 0.0, 1.0)),
    )
    assert record.terminal == "escape"
    assert len(record.boundaries) == 2
    assert record.boundaries[-1].after.kind == "escape"
    with pytest.raises(TransportError, match="partition boundary"):
        optical_flight_tangent(
            RayTangent((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
            _scene(),
            extinction_mm_inverse=(0.3, 0.8),
            optical_depth=0.3,
        )


def test_crossing_and_grazing_guards_do_not_return_partial_flights() -> None:
    from dataclasses import replace

    spec = replace(_scene(), max_crossings=1)
    with pytest.raises(TransportError, match="crossing guard"):
        optical_flight_tangent(
            RayTangent((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
            spec,
            extinction_mm_inverse=(0.3, 0.8),
            optical_depth=0.2,
        )
    with pytest.raises(TransportError, match="tangent"):
        material_box_interval(
            RayTangent((1.0, 0.0, -1.0), (0.0, 0.0, 1.0), origin_dot=(1.0, 0.0, 0.0)), _scene()
        )


def test_absent_or_vacuum_material_flight_is_physically_empty() -> None:
    record = optical_flight_tangent(
        RayTangent((3.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        _scene(),
        extinction_mm_inverse=(0.3, 0.8),
        optical_depth=0.2,
    )
    assert record.terminal == "escape" and record.boundaries == ()
    vacuum = optical_flight_tangent(
        RayTangent((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        _scene(),
        extinction_mm_inverse=(0.0, 0.0),
        optical_depth=0.2,
    )
    assert vacuum.total_optical_depth == 0.0 and vacuum.boundaries == ()
