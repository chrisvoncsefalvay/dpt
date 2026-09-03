"""Whole-ray contracts and a declared ideal half-space continuation reference."""

from __future__ import annotations

import math

import pytest

from dpt.transport.geometry_flights import optical_flight_tangent
from dpt.transport.geometry_source_jumps import (
    admit_joint_source_boundary_chart,
    material_source_ray_boundaries,
    material_source_ray_jump_observation,
    sample_material_source_ray_jump,
)
from dpt.transport.geometry_tangents import AffineBoundary, GridGeometryTangent, RigidMotion
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.source import ParallelBeam
from dpt.transport.uncollided_geometry import AcquisitionGeometryTangent


def _scene() -> tuple[ParallelBeam, TransportSpec]:
    return ParallelBeam((-0.25, -0.5, -1.0), (1.0, 1.0)), TransportSpec(
        MaterialGrid((0.0, -1.0, 0.0), (1.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-10.0, -10.0), (20.0, 20.0), (1, 1), 3.0),
        80.0,
        "whole conditional source-ray material jump",
    )


def test_exact_proportional_joint_lines_reject_despite_different_rounded_norms() -> None:
    first = AffineBoundary("material", (1, 1, -0.5), (0, 0, -1))
    second = AffineBoundary("detector", (3, 3, -1.5), (0, 0, 0))
    assert tuple(v / math.hypot(1, 1) for v in first.coefficients) != tuple(
        v / math.hypot(3, 3) for v in second.coefficients
    )
    with pytest.raises(TransportError, match="coincident active material/detector"):
        admit_joint_source_boundary_chart((first, second), family="material/detector")


def test_distinct_represented_lines_and_stationary_duplicates_remain_regular() -> None:
    first = AffineBoundary("material", (1, 1, -0.5), (0, 0, -1))
    second = AffineBoundary("detector", (3, 3, math.nextafter(-1.5, -math.inf)), (0, 0, 0))
    admit_joint_source_boundary_chart((first, second))
    admit_joint_source_boundary_chart(
        (
            AffineBoundary("first", (1, 1, -0.5), (0, 0, 0)),
            AffineBoundary("same", (-3, -3, 1.5), (0, 0, 0)),
        )
    )


def test_source_material_jump_contains_all_fluorescent_future_not_only_primary_transmission() -> (
    None
):
    source, spec = _scene()
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=RigidMotion(translation=(1.0, 0.0, 0.0)))
    )
    boundaries = material_source_ray_boundaries(source, spec, tangent)
    assert len(boundaries) == 1
    sample = sample_material_source_ray_jump(source, boundaries[0], 0.4)
    # Declared ideal half-space continuation (not the finite detector in
    # _scene): a primary absorption emits one fluorescent photon with
    # probability f, the lower line sees zero extinction, and a half-space
    # detector accepts half its isotropic directions. These are COMPLETE
    # unit-photon expectations, including one positive-order interaction.
    mu, fluorescence = 0.7, 0.4
    primary = math.exp(-mu)
    full_inside = primary + 0.5 * fluorescence * -math.expm1(-mu)
    result = material_source_ray_jump_observation(
        sample, negative_complete_response=1.0, positive_complete_response=full_inside
    )
    assert result == pytest.approx(1.0 - full_inside, abs=1e-15)
    assert result != pytest.approx(1.0 - primary, abs=1e-5)


def test_exact_first_flight_sides_need_no_spatial_displacement() -> None:
    source, spec = _scene()
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=RigidMotion(translation=(1.0, 0.0, 0.0)))
    )
    sample = sample_material_source_ray_jump(
        source, material_source_ray_boundaries(source, spec, tangent)[0], 0.4
    )
    assert sample.ray.origin_mm[0] == 0.0
    outside = optical_flight_tangent(
        sample.ray,
        spec,
        extinction_mm_inverse=(0.7,),
        optical_depth=0.2,
        material_side=sample.negative_side,
    )
    inside = optical_flight_tangent(
        sample.ray,
        spec,
        extinction_mm_inverse=(0.7,),
        optical_depth=0.2,
        material_side=sample.positive_side,
    )
    assert outside.terminal == "escape"
    assert inside.terminal == "collision" and inside.cell == 0
    assert inside.ray.origin_mm[0] == 0.0
