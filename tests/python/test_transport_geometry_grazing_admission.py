"""Whole-source initial-generator equality and correlated corner references."""

import math
from dataclasses import replace

import pytest

from dpt.geometry import RigidTransform
from dpt.transport.geometry_grazing_admission import admit_source_grazing_chart
from dpt.transport.geometry_histories import GeometryPhysicalMedium
from dpt.transport.geometry_source_jumps import material_source_ray_boundaries
from dpt.transport.geometry_tangents import GridGeometryTangent, RigidMotion
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.source import ParallelBeam
from dpt.transport.uncollided_geometry import AcquisitionGeometryTangent, SourceGeometryTangent


def _scene(shape=(2, 1, 1), rates=(0.7, 0.7), energies=(80,), source=None):
    cells = math.prod(shape)
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), shape[::-1], cells),
        PlanarDetector((-2, -2), (6, 6), (1, 1), 3),
        80,
        "Exact whole-source measured generator",
        energies,
    )
    medium = GeometryPhysicalMedium(
        tuple(range(cells)), rates, (0,) * len(rates), (1,) * cells, (1,) * cells
    )
    return source or ParallelBeam((1, 0.5, -1), (0, 0)), spec, medium


@pytest.mark.parametrize("extent", [(0, 0), (0, 0.8)])
@pytest.mark.parametrize("mode", ["object", "source", "angular"])
def test_identical_initial_grazing_laws_admit_active_pencil_and_line(extent, mode):
    source, spec, medium = _scene(source=ParallelBeam((1, 0.1, -1), extent))
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=RigidMotion((1, 0, 0)))
        if mode == "object"
        else GridGeometryTangent(),
        source=SourceGeometryTangent(motion=RigidMotion((1, 0, 0)))
        if mode == "source"
        else SourceGeometryTangent(direction_dot=(1, 0, 0))
        if mode == "angular"
        else SourceGeometryTangent(),
    )
    admitted = admit_source_grazing_chart(source, spec, medium, tangent)
    assert admitted.equivalent_faces == frozenset({(0, 1)})
    assert (
        material_source_ray_boundaries(
            source, spec, tangent, equivalent_whole_faces=admitted.equivalent_faces
        )
        == ()
    )


def test_unused_coefficient_energies_do_not_change_initial_generator_equality():
    source, spec, medium = _scene(rates=(0.2, 0.7, 1.4, 0.7), energies=(40, 80))
    tangent = AcquisitionGeometryTangent(object=GridGeometryTangent(motion=RigidMotion((1, 0, 0))))
    assert admit_source_grazing_chart(source, spec, medium, tangent).equivalent_faces == frozenset(
        {(0, 1)}
    )
    assert not admit_source_grazing_chart(
        source, spec, medium, tangent, emitted_energies_kev=(40, 80)
    ).equivalent_faces


def test_different_initial_grazing_generators_retain_the_discontinuity_guard():
    source, spec, medium = _scene(rates=(0.7, 0.9))
    tangent = AcquisitionGeometryTangent(object=GridGeometryTangent(motion=RigidMotion((1, 0, 0))))
    admitted = admit_source_grazing_chart(source, spec, medium, tangent)
    assert not admitted.equivalent_faces
    with pytest.raises(TransportError, match="whole source lies"):
        material_source_ray_boundaries(
            source, spec, tangent, equivalent_whole_faces=admitted.equivalent_faces
        )


@pytest.mark.parametrize(
    "velocity,rates",
    [
        ((1, 1, 0), (0.7, 1.3, 1.7, 0.7)),
        ((1, -1, 0), (2.1, 0.7, 0.7, 0.7)),
    ],
)
def test_correlated_corner_motion_ignores_unreached_neighbour_quadrants(velocity, rates):
    source, spec, medium = _scene((2, 2, 1), rates, source=ParallelBeam((1, 1, -1), (0, 0)))
    tangent = AcquisitionGeometryTangent(object=GridGeometryTangent(motion=RigidMotion(velocity)))
    admitted = admit_source_grazing_chart(source, spec, medium, tangent)
    assert admitted.equivalent_faces == frozenset({(0, 1), (1, 1)})
    assert (
        material_source_ray_boundaries(
            source, spec, tangent, equivalent_whole_faces=admitted.equivalent_faces
        )
        == ()
    )


def test_reached_diagonal_generator_jump_is_rejected_at_a_corner():
    source, spec, medium = _scene(
        (2, 2, 1), (1.3, 0.7, 0.7, 0.7), source=ParallelBeam((1, 1, -1), (0, 0))
    )
    tangent = AcquisitionGeometryTangent(object=GridGeometryTangent(motion=RigidMotion((1, 1, 0))))
    assert not admit_source_grazing_chart(source, spec, medium, tangent).equivalent_faces


@pytest.mark.parametrize("rates,expected", [((0, 0), True), ((0.7, 0.7), False)])
def test_outer_grazing_face_compares_actual_vacuum_generator(rates, expected):
    source, spec, medium = _scene(rates=rates, source=ParallelBeam((0, 0.5, -1), (0, 0)))
    tangent = AcquisitionGeometryTangent(object=GridGeometryTangent(motion=RigidMotion((1, 0, 0))))
    assert (
        bool(admit_source_grazing_chart(source, spec, medium, tangent).equivalent_faces) == expected
    )


def test_invisible_whole_face_and_common_motion_are_regular():
    source, spec, medium = _scene(rates=(0.7, 1.3), source=ParallelBeam((1, 10, -1), (0, 0)))
    move = RigidMotion((1, 0, 0))
    tangent = AcquisitionGeometryTangent(object=GridGeometryTangent(motion=move))
    assert admit_source_grazing_chart(source, spec, medium, tangent).equivalent_faces == frozenset(
        {(0, 1)}
    )
    common = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=move), source=SourceGeometryTangent(motion=move)
    )
    assert material_source_ray_boundaries(source, spec, common) == ()


@pytest.mark.parametrize(
    "pivot,rates", [((0, 1, 0), (1.3, 0.7, 1.7, 0.7)), ((1, 0, 0), (1.3, 1.7, 0.7, 0.7))]
)
def test_inactive_corner_face_rotational_curvature_rejects_forward_discontinuity(pivot, rates):
    source, spec, medium = _scene((2, 2, 1), rates, source=ParallelBeam((1, 1, -1), (0, 0)))
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=RigidMotion(angular=(0, 0, 1), pivot_mm=pivot))
    )
    with pytest.raises(TransportError, match="rotational curvature"):
        admit_source_grazing_chart(source, spec, medium, tangent)


def test_curvature_rejects_a_single_face_without_any_first_order_active_face():
    source, spec, medium = _scene(rates=(1.3, 0.7))
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=RigidMotion(angular=(0, 0, 1), pivot_mm=(0, 0.5, 0)))
    )
    with pytest.raises(TransportError, match="rotational curvature"):
        admit_source_grazing_chart(source, spec, medium, tangent)


def test_curvature_compares_only_actual_neighbour_quadrants():
    source, spec, medium = _scene(
        (2, 2, 1), (0.7, 1.3, 0.7, 0.7), source=ParallelBeam((1, 1, -1), (0, 0))
    )
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=RigidMotion(angular=(0, 0, 1), pivot_mm=(0, 1, 0)))
    )
    assert admit_source_grazing_chart(source, spec, medium, tangent).equivalent_faces == frozenset(
        {(0, 1), (1, 1)}
    )


def test_source_rotation_curvature_and_exact_common_motion():
    source, spec, medium = _scene(rates=(1.3, 0.7))
    motion = RigidMotion(angular=(0, 0, 1), pivot_mm=(0, 0.5, 0))
    tangent = AcquisitionGeometryTangent(source=SourceGeometryTangent(motion=motion))
    with pytest.raises(TransportError, match="rotational curvature"):
        admit_source_grazing_chart(source, spec, medium, tangent)
    common = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=motion), source=SourceGeometryTangent(motion=motion)
    )
    assert not admit_source_grazing_chart(source, spec, medium, common).equivalent_faces


def _crossover_scene(rates):
    spec = TransportSpec(
        MaterialGrid((0, 0, -1), (1, 1, 1), (2, 2, 2), 8),
        PlanarDetector((-5, -5), (10, 10), (1, 1), 3),
        80,
        "Independent compound-flow crossover",
        (80,),
    )
    source = ParallelBeam((1, 1, -2), (0, 0))
    medium = GeometryPhysicalMedium(tuple(range(8)), rates * 2, (0,) * 8, (1,) * 8, (1,) * 8)
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=RigidMotion(angular=(0, 1, 0))),
        source=SourceGeometryTangent(motion=RigidMotion(angular=(1, 0, 0))),
    )
    return source, spec, medium, tangent


def test_compound_crossover_rejects_discontinuous_vanishing_wedge():
    source, spec, medium, tangent = _crossover_scene((0.7, 1.3, 1.7, 0.7))
    with pytest.raises(TransportError, match="crossover chart is not certified"):
        admit_source_grazing_chart(source, spec, medium, tangent)
    # Exact finite attached ray Ry(-t) Rx(t)(1,1,z), independent of the jets.
    baseline = math.exp(-1.4)
    for sign in (-1, 1):
        theta = sign * 1e-6
        sine, cosine = math.sin(theta), math.cos(theta)
        wedge = 4 * math.sin(theta / 2) ** 2 / abs(sine * cosine)
        value = math.exp(-1.4 / cosine**2 - wedge)
        assert (value - baseline) / theta == pytest.approx(-sign * baseline, abs=1e-6)


def test_compound_crossover_equal_reached_laws_is_regular():
    source, spec, medium, tangent = _crossover_scene((0.7,) * 4)
    record = admit_source_grazing_chart(source, spec, medium, tangent)
    assert record.equivalent_faces == frozenset({(0, 1), (1, 1)})
    assert record.auxiliary_cells == tuple(range(8))


def test_same_axis_relative_rotation_keeps_constant_correlated_sides():
    source, spec, medium = _scene(
        (2, 2, 1), (1.3, 0.7, 0.7, 0.7), source=ParallelBeam((1, 1, -1), (0, 0))
    )
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=RigidMotion(angular=(0, 0, 2))),
        source=SourceGeometryTangent(motion=RigidMotion(angular=(0, 0, 1))),
    )
    assert admit_source_grazing_chart(source, spec, medium, tangent).equivalent_faces == frozenset(
        {(0, 1), (1, 1)}
    )


def test_compound_launch_crossover_rejects_one_sided_vanishing_wedge():
    source, spec, medium, tangent = _crossover_scene((0.7, 1.3, 1.7, 0.7))
    source = ParallelBeam((1, 1, 0), (0, 0))
    with pytest.raises(TransportError, match="crossover chart is not certified"):
        admit_source_grazing_chart(source, spec, medium, tangent)
    baseline = math.exp(-0.7)
    for sign, slope in ((1, 1.4), (-1, 2.4)):
        theta = sign * 1e-6
        sine, cosine = math.sin(theta), math.cos(theta)
        distance = (1 - sine * (1 + cosine)) / cosine**2
        wedge = 4 * math.sin(theta / 2) ** 2 / abs(sine * cosine) if sign < 0 else 0.0
        value = math.exp(-0.7 * distance - wedge)
        assert (value - baseline) / theta == pytest.approx(slope * baseline, abs=3e-6)


@pytest.mark.parametrize("equal", [False, True])
def test_source_coordinate_endpoint_remains_in_conservative_crossover_closure(equal):
    source, spec, medium, _ = _crossover_scene((0.7,) * 4 if equal else (0.7, 1.3, 1.7, 0.7))
    # Exact permutation rotation maps local intersection z to world source x.
    # The leading zero z=U+s=0 occurs only at U=s=0. Its higher-order wedge
    # can be benign after integrating U; this chart does not certify it.
    spec = replace(spec, object_pose=RigidTransform((0, 0, 1, 0, 1, 0, -1, 0, 0), (0, 0, 0)))
    source = ParallelBeam((0, 1, -1), (1, 0), (1, 0, 0))
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=RigidMotion(angular=(0, 1, 0))),
        source=SourceGeometryTangent(motion=RigidMotion(angular=(0, 0, -1))),
    )
    if equal:
        assert admit_source_grazing_chart(source, spec, medium, tangent).equivalent_faces
    else:
        with pytest.raises(TransportError, match="crossover chart is not certified"):
            admit_source_grazing_chart(source, spec, medium, tangent)
