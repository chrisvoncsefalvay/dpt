"""Native cutoff shadows against analytic box ownership and projected areas."""

# Warp annotations are runtime DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
import importlib
import math
from dataclasses import fields, replace
from typing import Any

import numpy as np
import pytest

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.transport.geometry_interface_arguments import interface_arguments
from dpt.transport.geometry_interfaces import material_interfaces
from dpt.transport.geometry_opaque_source import (
    opaque_boundary_sides,
    opaque_source_boundaries,
    source_coordinate_ray,
)
from dpt.transport.geometry_source_boundary_arguments import (
    affine_source_boundary_arguments,
    source_geometry_arguments,
)
from dpt.transport.geometry_tangents import GridGeometryTangent, RigidMotion
from dpt.transport.model import HistoryStatus, MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.source import ParallelBeam, PointConeSource
from dpt.transport.uncollided_geometry import AcquisitionGeometryTangent, SourceGeometryTangent

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.geometry_opaque_kernels")
source_native: Any = importlib.import_module("dpt.transport.geometry_source_boundary_kernels")
interface_native: Any = importlib.import_module("dpt.transport.geometry_interface_kernels")
ray_native: Any = importlib.import_module("dpt.transport.compton_geometry_boundary_kernels")


@wp.kernel
def _hits(
    rays: wp.array(dtype=ray_native.NativeBoundaryRay),
    grid: interface_native.InterfaceGrid,
    labels: wp.array2d(dtype=wp.int32),
    result: wp.array2d(dtype=wp.int32),
):
    index = wp.tid()
    hit = native.opaque_material_hit(
        rays[index],
        grid,
        labels[index, 0],
        labels[index, 1],
        labels[index, 2],
        labels[index, 3],
        labels[index, 4],
        labels[index, 5],
    )
    result[index, 0] = hit.status
    result[index, 1] = hit.hit


@wp.kernel
def _sides(
    source: source_native.NativeSourceGeometry,
    grid: interface_native.InterfaceGrid,
    records: wp.array(dtype=native.NativeOpaqueBoundary),
    coordinates: wp.array(dtype=wp.vec2d),
    result: wp.array2d(dtype=wp.int32),
):
    index = wp.tid()
    sides = native.opaque_boundary_sides(source, coordinates[index], records[index], grid)
    result[index, 0] = sides.status
    result[index, 1] = sides.negative_hit
    result[index, 2] = sides.positive_hit


@wp.kernel
def _coordinate_rays(
    source: source_native.NativeSourceGeometry,
    coordinates: wp.array(dtype=wp.vec2d),
    dots: wp.array(dtype=wp.vec2d),
    output: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    sample = native.source_coordinate_ray(source, coordinates[index], dots[index])
    status[index] = sample.status
    if sample.status == 0:
        for axis in range(3):
            output[index, axis] = sample.ray.origin_mm[axis]
            output[index, axis + 3] = sample.ray.direction[axis]
            output[index, axis + 6] = sample.ray.origin_dot[axis]
            output[index, axis + 9] = sample.ray.direction_dot[axis]


def _scene() -> TransportSpec:
    return TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 1), 1),
        PlanarDetector((-10, -10), (20, 20), (1, 1), 3),
        80,
        "analytic cutoff shadow primitive",
    )


def _grid(spec: TransportSpec) -> Any:
    _, arguments = interface_arguments(material_interfaces(spec)[0], spec)
    result = interface_native.InterfaceGrid()
    for field in fields(arguments):
        setattr(result, field.name, getattr(arguments, field.name))
    return result


def _source(source: ParallelBeam | PointConeSource) -> Any:
    arguments = source_geometry_arguments(source, SourceGeometryTangent())
    result = source_native.NativeSourceGeometry()
    for field in fields(arguments):
        setattr(result, field.name, getattr(arguments, field.name))
    return result


def _record(record: Any) -> Any:
    result = native.NativeOpaqueBoundary()
    arguments = affine_source_boundary_arguments(
        record.boundary, axis=0, face=record.index, family="material"
    )
    for field in fields(arguments):
        setattr(result.boundary, field.name, getattr(arguments, field.name))
    result.face_count = len(record.attached_faces)
    result.axis0, result.high0 = record.attached_faces[0]
    if result.face_count == 2:
        result.axis1, result.high1 = record.attached_faces[1]
    return result


def _ray(
    origin: tuple[float, float, float],
    direction=(0.0, 0.0, 1.0),
    origin_dot=(0.0, 0.0, 0.0),
    direction_dot=(0.0, 0.0, 0.0),
) -> Any:
    result = ray_native.NativeBoundaryRay()
    result.origin_mm, result.direction = wp.vec3d(*origin), wp.vec3d(*direction)
    result.origin_dot, result.direction_dot = wp.vec3d(*origin_dot), wp.vec3d(*direction_dot)
    return result


@pytest.mark.parametrize(
    "origin,side,od,expected",
    [
        ((0.5, 0.5, -1.0), 0, (0, 0, 0), 1),
        ((0.5, 0.5, 2.0), 0, (0, 0, 0), 0),
        ((0.5, 0.5, 0.5), 0, (0, 0, 0), 1),
        ((0.0, 0.5, -1.0), 0, (0, 0, 0), 1),
        ((1.0, 0.5, -1.0), 0, (0, 0, 0), 0),
        ((0.0, 0.5, -1.0), -1, (1, 0, 0), 0),
        ((0.0, 0.5, -1.0), 1, (1, 0, 0), 1),
        ((1.0, 0.5, -1.0), -1, (1, 0, 0), 1),
        ((1.0, 0.5, -1.0), 1, (1, 0, 0), 0),
    ],
)
def test_native_half_open_box_and_exact_coordinate_sides(
    warp_device: str, origin, side, od, expected
) -> None:
    output = wp.full((1, 2), -9, dtype=wp.int32, device=warp_device)
    wp.launch(
        _hits,
        1,
        inputs=[
            wp.array(
                [_ray(origin, origin_dot=od)],
                dtype=ray_native.NativeBoundaryRay,
                device=warp_device,
            ),
            _grid(_scene()),
            wp.array([[side, 0, 0, 0, 0, 0]], dtype=wp.int32, device=warp_device),
        ],
        outputs=[output],
        device=warp_device,
        record_tape=False,
    )
    assert output.numpy().tolist() == [[0, expected]]


@pytest.mark.parametrize("kind", ("parallel", "cone"))
def test_native_coordinate_jet_ignores_active_physical_geometry(
    warp_device: str, kind: str
) -> None:
    source = (
        ParallelBeam((-0.5, 0.1, -1), (1.1, 0.8))
        if kind == "parallel"
        else PointConeSource(
            DetectorGeometry((0.5, 0.5, -1), (-1, 0, 3), (1, 0, 0), (0, 1, 0), (4, 6), (1, 1))
        )
    )
    packed = _source(source)
    packed.motion_translation = wp.vec3d(17, -31, 43)
    packed.motion_angular = wp.vec3d(2, 3, 5)
    packed.extent_dot = wp.vec2d(11, 13)
    packed.target_motion_translation = wp.vec3d(71, 73, 79)
    coordinates = [(0.125, 0.25), (0.5, 0.5), (0.875, 0.75)]
    dots = [(1, 0), (0, 1), (-0.6, 0.8)]
    output = wp.array(np.full((3, 12), np.nan), dtype=wp.float64, device=warp_device)
    status = wp.full(3, -9, dtype=wp.int32, device=warp_device)
    wp.launch(
        _coordinate_rays,
        3,
        inputs=[
            packed,
            wp.array(coordinates, dtype=wp.vec2d, device=warp_device),
            wp.array(dots, dtype=wp.vec2d, device=warp_device),
        ],
        outputs=[output, status],
        device=warp_device,
        record_tape=False,
    )
    assert status.numpy().tolist() == [0, 0, 0]
    for actual, uv, uv_dot in zip(output.numpy(), coordinates, dots, strict=True):
        reference = source_coordinate_ray(source, uv, uv_dot)
        assert actual.tolist() == pytest.approx(
            (
                *reference.origin_mm,
                *reference.direction,
                *reference.origin_dot,
                *reference.direction_dot,
            ),
            abs=2e-14,
        )


@pytest.mark.parametrize("kind", ("parallel", "rotated", "cone"))
@pytest.mark.parametrize("active,expected", (("object", 2 / 3), ("source", -0.5)))
def test_native_attached_sides_and_cone_projected_area(
    warp_device: str, kind: str, active: str, expected: float
) -> None:
    spec = _scene()
    source: ParallelBeam | PointConeSource
    if kind == "cone":
        source = PointConeSource(
            DetectorGeometry((0.5, 0.5, -1), (-1, 0, 3), (1, 0, 0), (0, 1, 0), (4, 6), (1, 1))
        )
    else:
        source = ParallelBeam((-0.5, 0.1, -1), (1.1, 0.8))
        if kind == "rotated":
            c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
            spec = replace(
                spec, object_pose=RigidTransform((c, -s, 0, s, c, 0, 0, 0, 1), (0, 0, 0))
            )
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=RigidMotion((1, 0, 0)))
        if active == "object"
        else GridGeometryTangent(),
        source=SourceGeometryTangent(motion=RigidMotion((1, 0, 0)))
        if active == "source"
        else SourceGeometryTangent(),
    )
    pairs = [
        (record, record.boundary.sample((j + 0.5) / 3))
        for record in opaque_source_boundaries(source, spec, tangent)
        for j in range(3)
    ]
    pairs = [(record, sample) for record, sample in pairs if sample is not None]
    assert pairs
    output = wp.full((len(pairs), 3), -9, dtype=wp.int32, device=warp_device)
    wp.launch(
        _sides,
        len(pairs),
        inputs=[
            _source(source),
            _grid(spec),
            wp.array(
                [_record(record) for record, _ in pairs],
                dtype=native.NativeOpaqueBoundary,
                device=warp_device,
            ),
            wp.array([sample[0] for _, sample in pairs], dtype=wp.vec2d, device=warp_device),
        ],
        outputs=[output],
        device=warp_device,
        record_tape=False,
    )
    total = 0.0
    for actual, (record, sample) in zip(output.numpy(), pairs, strict=True):
        reference = opaque_boundary_sides(source, sample[0], record, spec)
        assert actual.tolist() == [0, int(reference[0]), int(reference[1])]
        total += sample[1] * (int(actual[1]) - int(actual[2])) / 3
    if kind == "cone":
        assert total == pytest.approx(expected, abs=3e-14)
    elif kind == "parallel":
        # Moving the lower box edge right shrinks the shadow and grows escape.
        assert total == pytest.approx((1 if active == "object" else -1) / 1.1, abs=3e-14)


@pytest.mark.parametrize(
    "malformed",
    (
        "zero_direction",
        "nonorthogonal_jet",
        "nan_origin",
        "overflow_upper",
        "collapsed_upper",
        "bad_pose",
        "bad_label",
        "unresolved_edge",
    ),
)
def test_native_malformed_chart_fails_before_a_miss(warp_device: str, malformed: str) -> None:
    grid = _grid(_scene())
    ray = _ray((10, 10, -1))
    labels = [0, 0, 0, 0, 0, 0]
    if malformed == "zero_direction":
        ray.direction = wp.vec3d(0, 0, 0)
    elif malformed == "nonorthogonal_jet":
        ray.direction_dot = wp.vec3d(0, 0, 1)
    elif malformed == "nan_origin":
        ray.origin_mm = wp.vec3d(math.nan, 10, -1)
    elif malformed == "overflow_upper":
        grid.origin, grid.spacing = wp.vec3d(1e308, 0, 0), wp.vec3d(1e308, 1, 1)
    elif malformed == "collapsed_upper":
        grid.origin = wp.vec3d(1e308, 0, 0)
    elif malformed == "bad_pose":
        grid.rotation = wp.mat33d(2, 0, 0, 0, 1, 0, 0, 0, 1)
    elif malformed == "bad_label":
        labels = [1, 2, 0, 0, 0, 1]
    else:
        ray = _ray((0.5, 0.5, -1), direction=(0.6, 0, 0.8))
        labels = [1, 2, 0, 0, 2, 0]
    output = wp.full((1, 2), -9, dtype=wp.int32, device=warp_device)
    wp.launch(
        _hits,
        1,
        inputs=[
            wp.array([ray], dtype=ray_native.NativeBoundaryRay, device=warp_device),
            grid,
            wp.array([labels], dtype=wp.int32, device=warp_device),
        ],
        outputs=[output],
        device=warp_device,
        record_tape=False,
    )
    assert output.numpy()[0, 0] == int(HistoryStatus.NUMERICAL_FAILURE)


@pytest.mark.parametrize("present,face_count", ((-1, 1), (2, 1), (1, 0)))
def test_native_boundary_presence_and_attached_face_contract(
    warp_device: str, present: int, face_count: int
) -> None:
    spec = _scene()
    source = ParallelBeam((-0.5, 0.1, -1), (1.1, 0.8))
    tangent = AcquisitionGeometryTangent(object=GridGeometryTangent(motion=RigidMotion((1, 0, 0))))
    record = opaque_source_boundaries(source, spec, tangent)[0]
    sample = record.boundary.sample(0.5)
    assert sample is not None
    packed = _record(record)
    packed.boundary.present = present
    packed.face_count = face_count
    output = wp.full((1, 3), -9, dtype=wp.int32, device=warp_device)
    wp.launch(
        _sides,
        1,
        inputs=[
            _source(source),
            _grid(spec),
            wp.array([packed], dtype=native.NativeOpaqueBoundary, device=warp_device),
            wp.array([sample[0]], dtype=wp.vec2d, device=warp_device),
        ],
        outputs=[output],
        device=warp_device,
        record_tape=False,
    )
    assert output.numpy()[0, 0] == int(HistoryStatus.NUMERICAL_FAILURE)
