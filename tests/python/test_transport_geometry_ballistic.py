"""Independent transverse ray-delta pullback and signed slab thickness."""

from __future__ import annotations

import math

import pytest

from dpt.transport.geometry_ballistic import ballistic_interface_crossings
from dpt.transport.geometry_interfaces import interface_generator_observation
from dpt.transport.geometry_tangents import GridGeometryTangent, RayTangent
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec


def _spec():
    return TransportSpec(
        MaterialGrid((-2.0, -2.0, 0.0), (4.0, 4.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-5.0, -5.0), (10.0, 10.0), (1, 1), 3.0),
        80.0,
        "transverse original ray delta",
    )


def test_original_ray_transverse_delta_has_exact_inverse_normal_projection():
    spec = _spec()
    root2 = math.sqrt(2.0)
    ray = RayTangent((-1.0, 0.0, -1.0), (1.0 / root2, 0.0, 1.0 / root2))
    records = ballistic_interface_crossings(ray, spec)
    assert len(records) == 2
    assert [r.point.interface.key for r in records] == ["interface:2:0:0:0", "interface:2:1:0:0"]
    assert [r.distance_mm for r in records] == pytest.approx([root2, 2.0 * root2])
    assert all(r.inverse_normal_projection == pytest.approx(root2) for r in records)
    assert records[0].point.point_mm == pytest.approx((0.0, 0.0, 0.0))
    assert records[1].point.point_mm == pytest.approx((1.0, 0.0, 1.0))


def test_absorbing_slab_spacing_derivative_matches_closed_physical_ray_integral():
    spec = _spec()
    ray = RayTangent((0.4, 0.6, -1.0), (0.0, 0.0, 1.0))
    records = ballistic_interface_crossings(
        ray, spec, GridGeometryTangent(spacing_dot=(0.0, 0.0, 1.0))
    )
    mu = 0.7
    terms = []
    for record in records:
        negative = mu if record.point.interface.negative_cell is not None else 0.0
        positive = mu if record.point.interface.positive_cell is not None else 0.0
        initial_depth = mu * max(0.0, record.point.point_mm[2])
        unchanged = math.exp(-mu * max(0.0, 1.0 - record.point.point_mm[2]))
        terms.append(
            interface_generator_observation(
                record.point,
                negative_extinction_mm_inverse=negative,
                positive_extinction_mm_inverse=positive,
                negative_collision_response=0.0 if negative else None,
                positive_collision_response=0.0 if positive else None,
                unchanged_direction_response=unchanged,
                incident_importance=math.exp(-initial_depth) * record.inverse_normal_projection,
            )
        )
    assert math.fsum(terms) == pytest.approx(-mu * math.exp(-mu), abs=2e-15)


def test_ballistic_corner_is_not_silently_assigned_to_one_material_patch():
    spec = _spec()
    ray = RayTangent((-2.0, 0.0, -1.0), (0.0, 0.0, 1.0))
    with pytest.raises(TransportError, match="patch edge"):
        ballistic_interface_crossings(ray, spec)
