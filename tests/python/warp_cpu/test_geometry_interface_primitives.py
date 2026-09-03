"""Independent surface proposal, generator sign and source-gauge checks."""

# Warp kernel annotations are runtime DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
# Host reference helpers expose their checked ray arithmetic internally.
# pyright: reportPrivateUsage=false
import importlib
import math
from dataclasses import fields, replace
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pytest

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.transport.geometry_interface_arguments import interface_arguments
from dpt.transport.geometry_interfaces import (
    interface_generator_observation,
    interface_point_tangent,
    material_interfaces,
    sample_interface_bridge,
    source_path_gauge,
)
from dpt.transport.geometry_tangents import GridGeometryTangent, RigidMotion
from dpt.transport.model import (
    HistoryStatus,
    MaterialGrid,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from dpt.transport.source import ParallelBeam, PointConeSource
from dpt.transport.uncollided_geometry import SourceGeometryTangent, _source_ray

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.geometry_interface_kernels")


@wp.func
def _write_point(
    point: native.NativeInterfacePoint, index: int, output: wp.array2d(dtype=wp.float64)
):
    for component in range(2):
        output[index, component] = point.coordinate[component]
    for component in range(3):
        output[index, 2 + component] = point.point_mm[component]
        output[index, 5 + component] = point.point_dot[component]
        output[index, 8 + component] = point.normal[component]
        output[index, 11 + component] = point.normal_dot[component]
    output[index, 14] = point.area_mm2


@wp.kernel
def _points(
    patches: wp.array(dtype=native.InterfacePatch),
    coordinates: wp.array(dtype=wp.vec2d),
    grid: native.InterfaceGrid,
    output: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    point = native.interface_point_tangent(patches[index], coordinates[index], grid)
    status[index] = point.status
    if point.status == 0:
        _write_point(point, index, output)


@wp.kernel
def _bridges(
    patch: native.InterfacePatch,
    previous: wp.array(dtype=wp.vec3d),
    coordinates: wp.array(dtype=wp.vec2d),
    grid: native.InterfaceGrid,
    output: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    sample = native.sample_interface_bridge(patch, previous[index], coordinates[index], grid)
    status[index] = sample.status
    if sample.status == 0:
        _write_point(sample.point, index, output)
        output[index, 15] = sample.log_area_density
        output[index, 16] = sample.log_geometric_importance
        output[index, 17] = sample.axis_normalisers[0]
        output[index, 18] = sample.axis_normalisers[1]


@wp.kernel
def _gauges(
    origin: wp.array(dtype=wp.vec3d),
    direction: wp.array(dtype=wp.vec3d),
    origin_dot: wp.array(dtype=wp.vec3d),
    direction_dot: wp.array(dtype=wp.vec3d),
    output: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    gauge = native.source_path_gauge(
        origin[index], direction[index], origin_dot[index], direction_dot[index]
    )
    status[index] = gauge.status
    if gauge.status == 0:
        for component in range(3):
            output[index, component] = gauge.translation[component]
            output[index, component + 3] = gauge.angular[component]
            output[index, component + 6] = gauge.pivot[component]


@wp.kernel
def _observations(
    patch: native.InterfacePatch,
    grid: native.InterfaceGrid,
    gauge: native.InterfaceGauge,
    responses: wp.array2d(dtype=wp.float64),
    present: wp.array(dtype=wp.vec2i),
    output: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    point = native.interface_point_tangent(patch, wp.vec2d(wp.float64(0.4), wp.float64(0.6)), grid)
    observation = native.interface_generator_observation(
        point,
        gauge,
        responses[index, 0],
        responses[index, 1],
        responses[index, 2],
        responses[index, 3],
        present[index][0],
        present[index][1],
        responses[index, 4],
        responses[index, 5],
    )
    status[index] = observation.status
    if observation.status == 0:
        output[index] = observation.value


@wp.kernel
def _direct_status(
    patch: native.InterfacePatch,
    grid: native.InterfaceGrid,
    directions: wp.array(dtype=wp.vec3d),
    flags: wp.array(dtype=wp.vec2i),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    point = native.interface_point_tangent(patch, wp.vec2d(wp.float64(0.4), wp.float64(0.6)), grid)
    status[index] = native.direct_interface_status(
        point, directions[index], flags[index][0], flags[index][1]
    )


def _scene() -> TransportSpec:
    return TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (3.0, 3.0), (1, 1), 2.0),
        80.0,
        "material-interface primitive acceptance",
    )


def _packed(
    face: Any, spec: TransportSpec, tangent: GridGeometryTangent | None = None
) -> tuple[Any, Any]:
    patch_args, grid_args = interface_arguments(face, spec, tangent)
    patch, grid = native.InterfacePatch(), native.InterfaceGrid()
    for field in fields(patch_args):
        setattr(patch, field.name, getattr(patch_args, field.name))
    for field in fields(grid_args):
        setattr(grid, field.name, getattr(grid_args, field.name))
    return patch, grid


def _buffers(count: int, columns: int, device: str) -> tuple[Any, Any]:
    # Failed lanes retain NaN storage, which must never become a zero score.
    return (
        wp.array(np.full((count, columns), np.nan), dtype=wp.float64, device=device),
        wp.full(count, -1, dtype=wp.int32, device=device),
    )


def _point_values(point: Any) -> tuple[float, ...]:
    return (
        *point.coordinate,
        *point.point_mm,
        *point.point_dot,
        *point.normal,
        *point.normal_dot,
        point.area_mm2,
    )


def _decimal_asinh(value: Decimal) -> Decimal:
    absolute = abs(value)
    result = (absolute + (absolute * absolute + 1).sqrt()).ln()
    return result if value >= 0 else -result


def _decimal_proposal(
    height: float, previous: tuple[float, float], draw: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    with localcontext() as context:
        context.prec = 90
        h = Decimal.from_float(height)
        offsets, normalisers = [], []
        for prior, uniform in zip(previous, draw, strict=True):
            lower = -Decimal.from_float(prior)
            upper = 1 + lower
            first, last = _decimal_asinh(lower / h), _decimal_asinh(upper / h)
            u = Decimal.from_float(uniform)
            argument = (1 - u) * first + u * last
            offset = h * (argument.exp() - (-argument).exp()) / 2
            offsets.append(float(offset + Decimal.from_float(prior)))
            normalisers.append(float(last - first))
        return (offsets[0], offsets[1]), (normalisers[0], normalisers[1])


@pytest.mark.parametrize("rotated", (False, True))
def test_native_face_points_preserve_all_axes_area_rotation_and_spacing_jets(
    warp_device: str, rotated: bool
) -> None:
    angle = 0.37
    pose = (
        RigidTransform(
            (
                math.cos(angle),
                -math.sin(angle),
                0.0,
                math.sin(angle),
                math.cos(angle),
                0.0,
                0.0,
                0.0,
                1.0,
            ),
            (1.0, -2.0, 0.5),
        )
        if rotated
        else RigidTransform()
    )
    spec = replace(
        _scene(),
        grid=MaterialGrid((-0.5, -0.2, 0.3), (0.7, 1.1, 0.9), (2, 2, 2), 2),
        detector=PlanarDetector((-1.0, -1.0), (3.0, 3.0), (1, 1), 5.0),
        object_pose=pose,
    )
    tangent = GridGeometryTangent(
        RigidMotion((0.2, -0.3, 0.4), (0.1, -0.2, 0.3), (0.1, 0.2, -0.3)),
        (0.1, -0.1, 0.2),
        (0.03, 0.04, -0.02),
    )
    faces = material_interfaces(spec)
    patches = [_packed(face, spec, tangent)[0] for face in faces]
    grid = _packed(faces[0], spec, tangent)[1]
    coordinates = [(0.25, 0.75)] * len(faces)
    output, status = _buffers(len(faces), 15, warp_device)
    wp.launch(
        _points,
        len(faces),
        inputs=[
            wp.array(patches, dtype=native.InterfacePatch, device=warp_device),
            wp.array(coordinates, dtype=wp.vec2d, device=warp_device),
            grid,
        ],
        outputs=[output, status],
        device=warp_device,
        record_tape=False,
    )
    assert np.array_equal(status.numpy(), np.zeros(len(faces), dtype=np.int32))
    expected = np.array(
        [
            _point_values(interface_point_tangent(face, coordinates[i], spec, tangent))
            for i, face in enumerate(faces)
        ]
    )
    np.testing.assert_allclose(output.numpy(), expected, rtol=2e-14, atol=2e-14)
    # Independent area: each axis owns two other physical spacings, with no
    # rigid-pose area factor or derivative hidden in its sampling density.
    for i, face in enumerate(faces):
        axes = [axis for axis in range(3) if axis != face.axis]
        with localcontext() as context:
            context.prec = 90
            area = Decimal.from_float(spec.grid.spacing_mm[axes[0]]) * Decimal.from_float(
                spec.grid.spacing_mm[axes[1]]
            )
        assert output.numpy()[i, 14] == pytest.approx(float(area), rel=2e-16)


@pytest.mark.parametrize("height", (0.3, 1e-4, 1e-10))
def test_native_bridge_matches_decimal_inverse_cdf_and_actual_represented_density(
    warp_device: str, height: float
) -> None:
    spec = _scene()
    face = next(face for face in material_interfaces(spec) if face.axis == 2 and face.face == 0)
    patch, grid = _packed(face, spec)
    coordinates = [(0.1, 0.2), (0.4, 0.7), (0.6, 0.8)]
    previous = [(0.4, 0.6, height)] * len(coordinates)
    output, status = _buffers(len(coordinates), 19, warp_device)
    wp.launch(
        _bridges,
        len(coordinates),
        inputs=[
            patch,
            wp.array(previous, dtype=wp.vec3d, device=warp_device),
            wp.array(coordinates, dtype=wp.vec2d, device=warp_device),
            grid,
        ],
        outputs=[output, status],
        device=warp_device,
        record_tape=False,
    )
    assert np.array_equal(status.numpy(), np.zeros(len(coordinates), dtype=np.int32))
    actual = output.numpy()
    for i, draw in enumerate(coordinates):
        decimal_point, decimal_z = _decimal_proposal(height, (0.4, 0.6), draw)
        assert actual[i, 2:4] == pytest.approx(decimal_point, abs=2e-15, rel=0)
        assert actual[i, 17:19] == pytest.approx(decimal_z, rel=3e-14)
        with localcontext() as context:
            context.prec = 90
            h = Decimal.from_float(height)
            x = Decimal.from_float(float(actual[i, 2])) - Decimal.from_float(0.4)
            y = Decimal.from_float(float(actual[i, 3])) - Decimal.from_float(0.6)
            zx, zy = (Decimal.from_float(value) for value in decimal_z)
            q = 1 / ((h * h + x * x).sqrt() * (h * h + y * y).sqrt() * zx * zy)
            importance = 1 / ((h * h + x * x + y * y) * q)
            expected_logq, expected_logw = float(q.ln()), float(importance.ln())
        assert actual[i, 15] == pytest.approx(expected_logq, abs=3e-13)
        assert actual[i, 16] == pytest.approx(expected_logw, abs=3e-13)
        assert math.exp(actual[i, 16]) <= math.prod(decimal_z) * (1 + 2e-13)
        host = sample_interface_bridge(face, previous[i], draw, spec)
        assert actual[i, 2:5] == pytest.approx(host.point.point_mm, abs=2e-15, rel=0)


def test_native_rotated_bridge_preserves_represented_cdf_and_physical_area_pdf(
    warp_device: str,
) -> None:
    angle = 0.37
    pose = RigidTransform(
        (
            math.cos(angle),
            -math.sin(angle),
            0.0,
            math.sin(angle),
            math.cos(angle),
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        (1.0, -2.0, 0.5),
    )
    spec = replace(
        _scene(), object_pose=pose, detector=PlanarDetector((-1.0, -1.0), (3.0, 3.0), (1, 1), 5.0)
    )
    face = next(face for face in material_interfaces(spec) if face.axis == 2 and face.face == 0)
    patch, grid = _packed(face, spec)
    previous = pose.point((0.4, 0.6, 0.3))
    draws = [(0.1, 0.2), (0.4, 0.7), (0.6, 0.8)]
    output, status = _buffers(len(draws), 19, warp_device)
    wp.launch(
        _bridges,
        len(draws),
        inputs=[
            patch,
            wp.array([previous] * len(draws), dtype=wp.vec3d, device=warp_device),
            wp.array(draws, dtype=wp.vec2d, device=warp_device),
            grid,
        ],
        outputs=[output, status],
        device=warp_device,
        record_tape=False,
    )
    assert np.all(status.numpy() == 0)
    local_previous = pose.inverse().point(previous)
    actual = output.numpy()
    for i, draw in enumerate(draws):
        represented = pose.inverse().point(
            (float(actual[i, 2]), float(actual[i, 3]), float(actual[i, 4]))
        )
        expected = sample_interface_bridge(face, previous, draw, spec)
        assert actual[i, 2:5] == pytest.approx(expected.point.point_mm, abs=2e-15, rel=0)
        with localcontext() as context:
            context.prec = 90
            h = abs(Decimal.from_float(local_previous[2]))
            factors = []
            for axis in range(2):
                prior = Decimal.from_float(local_previous[axis])
                lower, upper = -prior, 1 - prior
                normal = _decimal_asinh(upper / h) - _decimal_asinh(lower / h)
                offset = Decimal.from_float(represented[axis]) - prior
                cdf = (_decimal_asinh(offset / h) - _decimal_asinh(lower / h)) / normal
                assert float(cdf) == pytest.approx(draw[axis], abs=3e-14)
                factors.append((h * h + offset * offset).sqrt() * normal)
            expected_logq = float(-(factors[0] * factors[1]).ln())
        assert actual[i, 15] == pytest.approx(expected_logq, abs=3e-13)


@pytest.mark.parametrize("scale", (1e-150, 1e150))
def test_native_bridge_physical_scale_changes_area_pdf_but_not_geometric_importance(
    warp_device: str, scale: float
) -> None:
    spec = TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (scale, scale, scale), (1, 1, 1), 1),
        PlanarDetector((-scale, -scale), (3 * scale, 3 * scale), (1, 1), 2 * scale),
        80.0,
        "scale-invariant product-asinh bridge",
    )
    face = next(face for face in material_interfaces(spec) if face.axis == 2 and face.face == 0)
    patch, grid = _packed(face, spec)
    previous = (0.4 * scale, 0.6 * scale, 0.3 * scale)
    draw = (0.4, 0.7)
    output, status = _buffers(1, 19, warp_device)
    wp.launch(
        _bridges,
        1,
        inputs=[
            patch,
            wp.array([previous], dtype=wp.vec3d, device=warp_device),
            wp.array([draw], dtype=wp.vec2d, device=warp_device),
            grid,
        ],
        outputs=[output, status],
        device=warp_device,
        record_tape=False,
    )
    assert status.numpy()[0] == 0
    actual = output.numpy()[0]
    with localcontext() as context:
        context.prec = 90
        s = Decimal.from_float(scale)
        h = Decimal.from_float(previous[2])
        factors, offsets = [], []
        for axis in range(2):
            prior = Decimal.from_float(previous[axis])
            first, last = _decimal_asinh(-prior / h), _decimal_asinh((s - prior) / h)
            offset = Decimal.from_float(float(actual[2 + axis])) - prior
            factors.append((h * h + offset * offset).sqrt() * (last - first))
            offsets.append(offset)
        q = 1 / (factors[0] * factors[1])
        weight = 1 / ((h * h + offsets[0] ** 2 + offsets[1] ** 2) * q)
        expected_logq, expected_logw = float(q.ln()), float(weight.ln())
    assert actual[15] == pytest.approx(expected_logq, abs=3e-13)
    assert actual[16] == pytest.approx(expected_logw, abs=3e-13)
    base = _scene()
    base_face = next(
        face for face in material_interfaces(base) if face.axis == 2 and face.face == 0
    )
    base_sample = sample_interface_bridge(base_face, (0.4, 0.6, 0.3), draw, base)
    assert actual[15] + 2 * math.log(scale) == pytest.approx(
        base_sample.log_area_density, abs=3e-13
    )
    assert actual[16] == pytest.approx(base_sample.log_geometric_importance, abs=3e-13)


def test_native_face_measure_overflow_and_underflow_are_explicit_fatal_results(
    warp_device: str,
) -> None:
    base = _scene()
    face = material_interfaces(base)[0]
    for scale in (1e-200, 1e200):
        patch, grid = _packed(face, base)
        grid.spacing = (scale, scale, scale)
        output, status = _buffers(1, 15, warp_device)
        wp.launch(
            _points,
            1,
            inputs=[
                wp.array([patch], dtype=native.InterfacePatch, device=warp_device),
                wp.array([(0.4, 0.6)], dtype=wp.vec2d, device=warp_device),
                grid,
            ],
            outputs=[output, status],
            device=warp_device,
            record_tape=False,
        )
        assert status.numpy()[0] == int(HistoryStatus.NUMERICAL_FAILURE)
        assert np.isnan(output.numpy()).all()


def test_native_bridge_rejects_exact_plane_and_unresolved_coordinates(warp_device: str) -> None:
    spec = _scene()
    face = next(face for face in material_interfaces(spec) if face.axis == 2 and face.face == 0)
    patch, grid = _packed(face, spec)
    previous = [
        (0.4, 0.6, 0.0),
        (0.4, 0.6, 1e-100),
        (0.4, 0.6, 0.3),
        (0.4, 0.6, 0.3),
        (math.nan, 0.6, 0.3),
    ]
    coordinates = [(0.4, 0.7), (0.4, 0.7), (0.0, 0.7), (1.0, 0.7), (0.4, 0.7)]
    output, status = _buffers(len(previous), 19, warp_device)
    wp.launch(
        _bridges,
        len(previous),
        inputs=[
            patch,
            wp.array(previous, dtype=wp.vec3d, device=warp_device),
            wp.array(coordinates, dtype=wp.vec2d, device=warp_device),
            grid,
        ],
        outputs=[output, status],
        device=warp_device,
        record_tape=False,
    )
    assert np.all(status.numpy() == int(HistoryStatus.NUMERICAL_FAILURE))
    assert np.isnan(output.numpy()).all()
    with pytest.raises(TransportError):
        sample_interface_bridge(face, previous[0], coordinates[0], spec)
    with pytest.raises(TransportError):
        sample_interface_bridge(face, previous[1], coordinates[1], spec)


def test_native_face_rejects_wrong_cell_sides_and_coordinate_support(warp_device: str) -> None:
    spec = _scene()
    face = material_interfaces(spec)[0]
    patch, grid = _packed(face, spec)
    wrong = native.InterfacePatch()
    wrong.axis, wrong.face, wrong.tangential_cells = patch.axis, patch.face, patch.tangential_cells
    wrong.negative_cell, wrong.positive_cell = 0, patch.positive_cell
    output, status = _buffers(3, 15, warp_device)
    wp.launch(
        _points,
        3,
        inputs=[
            wp.array([wrong, patch, patch], dtype=native.InterfacePatch, device=warp_device),
            wp.array([(0.4, 0.6), (0.0, 0.6), (0.4, math.nan)], dtype=wp.vec2d, device=warp_device),
            grid,
        ],
        outputs=[output, status],
        device=warp_device,
        record_tape=False,
    )
    assert np.all(status.numpy() == int(HistoryStatus.NUMERICAL_FAILURE))
    assert np.isnan(output.numpy()).all()


@pytest.mark.parametrize("cone", (False, True))
def test_native_source_gauge_matches_original_source_address_jets(
    warp_device: str, cone: bool
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
        else ParallelBeam((-1.0, -1.0, -1.0), (2.0, 2.0))
    )
    tangent = SourceGeometryTangent(
        motion=RigidMotion((0.1, 0.2, 0.3), (0.2, -0.1, 0.1)),
        target_motion=RigidMotion((-0.2, 0.1, 0.2), (0.1, 0.3, -0.2)),
        extent_dot=(0.0, 0.0) if cone else (0.1, -0.2),
        target_spacing_dot=(0.2, -0.3),
    )
    coordinates = [(0.3, 0.7), (0.7, 0.4)]
    rays = [_source_ray(source, coordinate, tangent)[0] for coordinate in coordinates]
    output, status = _buffers(len(rays), 9, warp_device)
    inputs = [
        wp.array([getattr(ray, name) for ray in rays], dtype=wp.vec3d, device=warp_device)
        for name in ("origin_mm", "direction", "origin_dot", "direction_dot")
    ]
    wp.launch(
        _gauges,
        len(rays),
        inputs=inputs,
        outputs=[output, status],
        device=warp_device,
        record_tape=False,
    )
    assert np.all(status.numpy() == 0)
    for i, coordinate in enumerate(coordinates):
        gauge = source_path_gauge(source, coordinate, tangent)
        assert output.numpy()[i] == pytest.approx(
            (*gauge.translation, *gauge.angular, *gauge.pivot_mm), abs=3e-15, rel=3e-15
        )
        assert gauge.point_velocity(rays[i].origin_mm) == pytest.approx(
            rays[i].origin_dot, abs=3e-15
        )
        assert gauge.direction_velocity(rays[i].direction) == pytest.approx(
            rays[i].direction_dot, abs=3e-15
        )


def _observation_run(
    device: str,
    spec: TransportSpec,
    face: Any,
    tangent: GridGeometryTangent,
    responses: list[tuple[float, ...]],
    present: list[tuple[int, int]],
    motion: RigidMotion | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    motion = RigidMotion() if motion is None else motion
    patch, grid = _packed(face, spec, tangent)
    gauge = native.InterfaceGauge()
    gauge.translation, gauge.angular, gauge.pivot, gauge.status = (
        motion.translation,
        motion.angular,
        motion.pivot_mm,
        0,
    )
    output = wp.array(np.full(len(responses), np.nan), dtype=wp.float64, device=device)
    status = wp.full(len(responses), -1, dtype=wp.int32, device=device)
    wp.launch(
        _observations,
        len(responses),
        inputs=[
            patch,
            grid,
            gauge,
            wp.array(responses, dtype=wp.float64, device=device),
            wp.array(present, dtype=wp.vec2i, device=device),
        ],
        outputs=[output, status],
        device=device,
        record_tape=False,
    )
    return output.numpy(), status.numpy()


def test_native_generator_signed_responses_match_independent_decimal_contraction(
    warp_device: str,
) -> None:
    spec = _scene()
    face = next(face for face in material_interfaces(spec) if face.axis == 2 and face.face == 1)
    tangent = GridGeometryTangent(spacing_dot=(0.0, 0.0, 1.0))
    responses = [
        (0.7, 0.0, 0.0, 0.0, 1.0, math.exp(-0.7)),
        (0.7, 0.2, 3.1, -2.2, 0.3, 1.4),
        (0.3, 0.3, -4.0, 2.0, -0.2, -1.0),
        (0.7, 0.2, 0.3, 0.3, 0.3, 1.4),
    ]
    present = [(1, 0), (1, 1), (1, 1), (1, 1)]
    output, status = _observation_run(warp_device, spec, face, tangent, responses, present)
    assert np.all(status == 0)
    for actual, row in zip(output, responses, strict=True):
        with localcontext() as context:
            context.prec = 90
            left, right, cl, cr, future, incident = (Decimal.from_float(value) for value in row)
            expected = incident * (left * (cl - future) - right * (cr - future))
        assert actual == pytest.approx(float(expected), abs=5e-16, rel=3e-15)
    point = interface_point_tangent(face, (0.4, 0.6), spec, tangent)
    assert output[0] == pytest.approx(
        interface_generator_observation(
            point,
            negative_extinction_mm_inverse=0.7,
            positive_extinction_mm_inverse=0.0,
            negative_collision_response=0.0,
            positive_collision_response=None,
            unchanged_direction_response=1.0,
            incident_importance=math.exp(-0.7),
        ),
        rel=3e-15,
    )


def test_native_common_rigid_motion_cancels_relative_interface_velocity(warp_device: str) -> None:
    spec = _scene()
    motion = RigidMotion((0.2, -0.3, 0.4), (0.1, -0.2, 0.3), (0.1, 0.2, -0.3))
    for face in material_interfaces(spec):
        output, status = _observation_run(
            warp_device,
            spec,
            face,
            GridGeometryTangent(motion),
            [(0.7, 0.2, 3.1, -2.2, 0.3, 1.4)],
            [(1, 1)],
            motion,
        )
        assert status[0] == 0
        assert output[0] == 0.0


def test_native_two_slab_faces_cancel_without_losing_the_thickness_sign(warp_device: str) -> None:
    spec, mu = _scene(), 0.7
    tangent = GridGeometryTangent(RigidMotion((0.0, 0.0, 1.0)))
    scores = []
    for face in material_interfaces(spec):
        if face.axis != 2:
            continue
        row = (
            (0.0, mu, 0.0, 0.0, math.exp(-mu), 1.0)
            if face.face == 0
            else (mu, 0.0, 0.0, 0.0, 1.0, math.exp(-mu))
        )
        output, status = _observation_run(
            warp_device, spec, face, tangent, [row], [(0, 1) if face.face == 0 else (1, 0)]
        )
        assert status[0] == 0
        scores.append(output[0])
    assert scores[0] > 0 and scores[1] < 0 and math.fsum(scores) == 0.0


def test_native_generator_fails_missing_nonvacuum_nonfinite_or_overflowed_continuations(
    warp_device: str,
) -> None:
    spec = _scene()
    face = next(face for face in material_interfaces(spec) if face.axis == 2 and face.face == 1)
    tangent = GridGeometryTangent(spacing_dot=(0.0, 0.0, 1.0))
    rows = [
        (0.7, 0.2, 1.0, 1.0, 0.3, 1.4),
        (-0.7, 0.2, 1.0, 1.0, 0.3, 1.4),
        (0.7, 0.2, math.nan, 1.0, 0.3, 1.4),
        (1e308, 0.2, 1e308, 1.0, 0.3, 1.4),
        (0.7, 0.2, 1.0, 1.0, 0.3, math.inf),
    ]
    output, status = _observation_run(
        warp_device, spec, face, tangent, rows, [(1, 0), (1, 1), (1, 1), (1, 1), (1, 1)]
    )
    assert np.all(status == int(HistoryStatus.NUMERICAL_FAILURE))
    assert np.isnan(output).all()


def test_native_direct_grazing_chart_is_rejected_only_for_visible_collimated_fluence(
    warp_device: str,
) -> None:
    spec = _scene()
    face = next(face for face in material_interfaces(spec) if face.axis == 0 and face.face == 0)
    patch, grid = _packed(face, spec)
    status = wp.full(4, -1, dtype=wp.int32, device=warp_device)
    wp.launch(
        _direct_status,
        4,
        inputs=[
            patch,
            grid,
            wp.array(
                [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
                dtype=wp.vec3d,
                device=warp_device,
            ),
            wp.array([(1, 1), (0, 1), (1, 0), (1, 1)], dtype=wp.vec2i, device=warp_device),
        ],
        outputs=[status],
        device=warp_device,
        record_tape=False,
    )
    assert status.numpy().tolist() == [int(HistoryStatus.NUMERICAL_FAILURE), 0, 0, 0]


def test_normalised_product_proposal_has_a_finite_normal_height_log_moment_bound() -> None:
    # For |x|,|y| <= L and 0<h<=H, Zx*Zy <= 4[B+log(H/h)]²,
    # B=log(1+2L/H). Thus its squared bridge weight integrates to at most
    # 16H sum_k binomial(4,k) B^(4-k) k!, without replacing h by epsilon.
    length, height_limit = 1.0, 0.3
    b = math.log1p(2 * length / height_limit)
    bound = (
        16
        * height_limit
        * math.fsum(math.comb(4, k) * b ** (4 - k) * math.factorial(k) for k in range(5))
    )
    assert math.isfinite(bound) and bound > 0.0
    for height in (0.3, 1e-4, 1e-10, 1e-100, 1e-300):
        zx = _decimal_proposal(height, (0.4, 0.6), (0.2, 0.7))[1]
        assert math.prod(zx) <= 4 * (b + math.log(height_limit / height)) ** 2 * (1 + 1e-13)
    # Independent Simpson integration of each actual marginal at regular h.
    n, height = 2048, 0.3
    for prior in (0.4, 0.6):
        lower, upper = -prior, 1 - prior
        normal = math.asinh(upper / height) - math.asinh(lower / height)
        step = (upper - lower) / n
        integral = (
            math.fsum(
                (1 if i in (0, n) else 4 if i % 2 else 2)
                / (math.hypot(height, lower + i * step) * normal)
                for i in range(n + 1)
            )
            * step
            / 3
        )
        assert integral == pytest.approx(1.0, abs=2e-14)
