"""Native source charts checked against exact coarea and attached launch jets."""

# Warp annotations are runtime DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
import importlib
import math
from dataclasses import fields, replace
from decimal import Decimal, localcontext
from typing import Any, cast

import numpy as np
import pytest

from dpt.geometry import DetectorGeometry
from dpt.transport.geometry_source_boundary_arguments import (
    affine_source_boundary_arguments,
    source_geometry_arguments,
)
from dpt.transport.geometry_tangents import AffineBoundary, RigidMotion
from dpt.transport.model import HistoryStatus, TransportError
from dpt.transport.source import ParallelBeam, PointConeSource
from dpt.transport.uncollided_geometry import SourceGeometryTangent, _source_ray

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.geometry_source_boundary_kernels")


@wp.kernel
def _rays(
    source: native.NativeSourceGeometry,
    coordinates: wp.array(dtype=wp.vec2d),
    output: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    sample = native.source_ray_tangent(source, coordinates[index])
    status[index] = sample.status
    if sample.status == 0:
        for component in range(3):
            output[index, component] = sample.ray.origin_mm[component]
            output[index, 3 + component] = sample.ray.direction[component]
            output[index, 6 + component] = sample.ray.origin_dot[component]
            output[index, 9 + component] = sample.ray.direction_dot[component]
        output[index, 12] = sample.source_coordinate[0]
        output[index, 13] = sample.source_coordinate[1]
        output[index, 14] = sample.source_weight


@wp.kernel
def _boundaries(
    boundary: native.NativeAffineSourceBoundary,
    coordinates: wp.array(dtype=wp.float64),
    output: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    sample = native.sample_source_boundary(boundary, coordinates[index])
    status[index] = sample.status
    if sample.status == 0:
        output[index, 0] = sample.coordinate[0]
        output[index, 1] = sample.coordinate[1]
        output[index, 2] = sample.coarea_weight
        output[index, 3] = sample.levelset_dot
        output[index, 4] = sample.line_length
        output[index, 5] = wp.float64(sample.axis)
        output[index, 6] = wp.float64(sample.face)
        output[index, 7] = wp.float64(sample.family)


def _source(source: ParallelBeam | PointConeSource, tangent: SourceGeometryTangent) -> Any:
    arguments = source_geometry_arguments(source, tangent)
    result = native.NativeSourceGeometry()
    for field in fields(arguments):
        setattr(result, field.name, getattr(arguments, field.name))
    return result


def _boundary(boundary: AffineBoundary | None, *, detector: bool = False) -> Any:
    arguments = affine_source_boundary_arguments(
        boundary, axis=1, face=2, family="detector" if detector else "material"
    )
    result = native.NativeAffineSourceBoundary()
    for field in fields(arguments):
        setattr(result, field.name, getattr(arguments, field.name))
    return result


def _run_rays(
    device: str, source: Any, coordinates: list[tuple[float, float]]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    output = wp.array(np.full((len(coordinates), 15), np.nan), dtype=wp.float64, device=device)
    status = wp.full(len(coordinates), -9, dtype=wp.int32, device=device)
    wp.launch(
        _rays,
        len(coordinates),
        inputs=[source, wp.array(coordinates, dtype=wp.vec2d, device=device)],
        outputs=[output, status],
        device=device,
        record_tape=False,
    )
    return output.numpy(), status.numpy()


def _run_boundaries(
    device: str, boundary: Any, coordinates: list[float] | None = None
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[float]]:
    coordinates = [0.125, 0.5, 0.875] if coordinates is None else coordinates
    output = wp.array(np.full((len(coordinates), 8), np.nan), dtype=wp.float64, device=device)
    status = wp.full(len(coordinates), -9, dtype=wp.int32, device=device)
    wp.launch(
        _boundaries,
        len(coordinates),
        inputs=[boundary, wp.array(coordinates, dtype=wp.float64, device=device)],
        outputs=[output, status],
        device=device,
        record_tape=False,
    )
    return output.numpy(), status.numpy(), coordinates


def _decimal_line(
    boundary: AffineBoundary, coordinate: float
) -> tuple[float, float, float, float, float]:
    """Independent high-precision line intersections and Euclidean coarea."""
    with localcontext() as context:
        context.prec = 90
        a, b, c = map(Decimal.from_float, boundary.coefficients)
        ad, bd, cd = map(Decimal.from_float, boundary.coefficients_dot)
        points: set[tuple[Decimal, Decimal]] = set()
        if b:
            points.update(
                (u, -(a * u + c) / b)
                for u in (Decimal(0), Decimal(1))
                if 0 <= -(a * u + c) / b <= 1
            )
        if a:
            points.update(
                (-(b * v + c) / a, v)
                for v in (Decimal(0), Decimal(1))
                if 0 <= -(b * v + c) / a <= 1
            )
        assert len(points) == 2
        # The host edge order first visits increasing u, then increasing v.
        ordered = sorted(points, key=lambda p: (p[0] not in (0, 1), p[0], p[1]))
        first, last = ordered
        r = Decimal.from_float(coordinate)
        u, v = (first[i] + r * (last[i] - first[i]) for i in (0, 1))
        length = sum(((last[i] - first[i]) ** 2 for i in (0, 1)), Decimal(0)).sqrt()
        gdot = ad * u + bd * v + cd
        weight = length * gdot / (a * a + b * b).sqrt()
        return float(u), float(v), float(weight), float(gdot), float(length)


@pytest.mark.parametrize(
    "coefficients",
    (
        (2.0, 0.0, -0.6),
        (0.0, -3.0, 1.2),
        (1.1, 0.7, -0.61),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, -1.0e-10),
    ),
)
def test_native_affine_point_and_weight_match_decimal_and_host(
    warp_device: str, coefficients: tuple[float, float, float]
) -> None:
    boundary = AffineBoundary("independent", coefficients, (0.2, -0.3, 0.7))
    output, status, draws = _run_boundaries(warp_device, _boundary(boundary))
    assert np.all(status == 0)
    expected = np.asarray([_decimal_line(boundary, value) for value in draws])
    np.testing.assert_allclose(output[:, :5], expected, rtol=4e-14, atol=2e-15)
    for index, draw in enumerate(draws):
        sample = boundary.sample(draw)
        assert sample is not None
        np.testing.assert_allclose(output[index, :2], sample[0], rtol=4e-14, atol=2e-15)
        assert output[index, 2] == pytest.approx(sample[1], rel=4e-14, abs=2e-15)
    np.testing.assert_array_equal(output[:, 5:], np.tile((1, 2, 0), (len(draws), 1)))


@pytest.mark.parametrize("scale", (1e-150, 1e150))
def test_native_affine_coarea_is_invariant_to_levelset_scaling(
    warp_device: str, scale: float
) -> None:
    coefficients = (scale, scale, -scale)
    derivatives = (scale * 0.2, scale * -0.3, scale * 0.7)
    boundary = AffineBoundary("scaled", coefficients, derivatives)
    output, status, draws = _run_boundaries(warp_device, _boundary(boundary))
    assert np.all(status == 0)
    expected = np.asarray([_decimal_line(boundary, value) for value in draws])
    np.testing.assert_allclose(output[:, :3], expected[:, :3], rtol=4e-14, atol=2e-15)


@pytest.mark.parametrize("mode", ("fixed", "rigid", "extent-direction", "pencil"))
def test_native_parallel_source_preserves_original_coordinate_and_weight(
    warp_device: str, mode: str
) -> None:
    source = ParallelBeam((-2.0, 0.3, -4.0), (3.0, 2.0), weight=7.5)
    tangent = SourceGeometryTangent()
    if mode == "rigid":
        tangent = replace(
            tangent,
            motion=RigidMotion((0.2, -0.4, 0.7), (0.1, -0.2, 0.3), (1.0, 2.0, 3.0)),
        )
    elif mode == "extent-direction":
        tangent = replace(tangent, extent_dot=(0.5, -0.3), direction_dot=(0.2, -0.7, 0.0))
    elif mode == "pencil":
        source = replace(source, extent_xy_mm=(0.0, 0.0))
    draws = [(0.0, 0.0), (0.2, 0.8), (1.0, 1.0)]
    output, status = _run_rays(warp_device, _source(source, tangent), draws)
    assert np.all(status == 0)
    for index, coordinate in enumerate(draws):
        ray, _, _ = _source_ray(source, coordinate, tangent)
        expected = (*ray.origin_mm, *ray.direction, *ray.origin_dot, *ray.direction_dot)
        np.testing.assert_allclose(output[index, :12], expected, rtol=3e-14, atol=3e-14)
    np.testing.assert_array_equal(output[:, 12:14], draws)
    np.testing.assert_array_equal(output[:, 14], np.full(len(draws), source.weight))


def _cone() -> PointConeSource:
    return PointConeSource(
        DetectorGeometry(
            (-1.0, 0.4, -3.0),
            (-0.4, 0.6, 4.0),
            (0.8, 0.0, -0.6),
            (0.0, 1.0, 0.0),
            (0.6, 0.8),
            (3, 4),
        ),
        weight=2.3,
    )


@pytest.mark.parametrize("mode", ("fixed", "independent-motions-spacing", "common-motion"))
def test_native_cone_jet_preserves_first_centre_spacing_anchor(warp_device: str, mode: str) -> None:
    source, tangent = _cone(), SourceGeometryTangent()
    motion = RigidMotion((0.2, -0.3, 0.4), (0.1, -0.2, 0.05), (0.5, -0.2, 0.1))
    if mode == "independent-motions-spacing":
        tangent = replace(
            tangent,
            motion=motion,
            target_motion=RigidMotion((-0.1, 0.3, 0.2), (-0.2, 0.1, 0.2), (1.0, 0.0, 2.0)),
            target_spacing_dot=(0.13, -0.17),
        )
    elif mode == "common-motion":
        tangent = replace(tangent, motion=motion, target_motion=motion)
    draws = [(0.0, 0.0), (0.2, 0.8), (1.0, 1.0)]
    output, status = _run_rays(warp_device, _source(source, tangent), draws)
    assert np.all(status == 0)
    for index, coordinate in enumerate(draws):
        ray, displacement, displacement_dot = _source_ray(source, coordinate, tangent)
        expected = (*ray.origin_mm, *ray.direction, *ray.origin_dot, *ray.direction_dot)
        np.testing.assert_allclose(output[index, :12], expected, rtol=4e-14, atol=3e-14)
        with localcontext() as context:
            context.prec = 90
            r, rd = [
                list(map(Decimal.from_float, values)) for values in (displacement, displacement_dot)
            ]
            length = sum((value * value for value in r), Decimal(0)).sqrt()
            direction = [value / length for value in r]
            projection = sum(a * b for a, b in zip(direction, rd, strict=True))
            derivative = [(rd[i] - direction[i] * projection) / length for i in range(3)]
        np.testing.assert_allclose(
            output[index, 3:6], list(map(float, direction)), rtol=4e-14, atol=3e-14
        )
        np.testing.assert_allclose(
            output[index, 9:12], list(map(float, derivative)), rtol=4e-14, atol=3e-14
        )
        if mode == "common-motion":
            np.testing.assert_allclose(
                output[index, 9:12],
                np.cross(motion.angular, output[index, 3:6]),
                rtol=4e-14,
                atol=3e-14,
            )
    np.testing.assert_array_equal(output[:, 12:14], draws)
    np.testing.assert_array_equal(output[:, 14], np.full(len(draws), source.weight))


def test_native_absent_outside_and_corner_only_charts_are_null(warp_device: str) -> None:
    for chart in (
        None,
        AffineBoundary("outside", (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
        AffineBoundary("corner", (1.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ):
        arguments = _boundary(chart)
        if chart is not None:
            arguments.present = 1
        output, status, _ = _run_boundaries(warp_device, arguments)
        np.testing.assert_array_equal(status, np.full(len(status), -1))
        assert np.all(np.isnan(output))


@pytest.mark.parametrize("active", (False, True))
def test_native_whole_source_edge_is_null_or_fatal_without_half_weight(
    warp_device: str, active: bool
) -> None:
    chart = AffineBoundary("edge", (0.0, 1.0, -1.0), (0.0, 0.0, 1.0 if active else 0.0))
    if active:
        with pytest.raises(TransportError, match="moving boundary"):
            affine_source_boundary_arguments(chart, axis=0, face=0, family="material")
        arguments = native.NativeAffineSourceBoundary()
        arguments.b, arguments.c, arguments.c_dot = 1.0, -1.0, 1.0
        arguments.present = 1
    else:
        arguments = _boundary(chart)
        arguments.present = 1
    _, status, _ = _run_boundaries(warp_device, arguments)
    expected = int(HistoryStatus.NUMERICAL_FAILURE) if active else -1
    np.testing.assert_array_equal(status, np.full(len(status), expected))


def test_native_invalid_chart_and_source_inputs_are_fatal(warp_device: str) -> None:
    chart = _boundary(AffineBoundary("valid", (1.0, 1.0, -1.0), (0.0, 0.0, 1.0)))
    _, status, _ = _run_boundaries(warp_device, chart, [0.0, 1.0, math.nan, math.inf])
    assert np.all(status == int(HistoryStatus.NUMERICAL_FAILURE))
    for name, value in (("a", math.nan), ("family", 2), ("axis", 3), ("face", -1), ("present", 2)):
        malformed = _boundary(AffineBoundary("valid", (1.0, 1.0, -1.0), (0.0, 0.0, 1.0)))
        setattr(malformed, name, value)
        _, status, _ = _run_boundaries(warp_device, malformed)
        assert np.all(status == int(HistoryStatus.NUMERICAL_FAILURE))
    source = _source(ParallelBeam((0.0, 0.0, -1.0), (1.0, 1.0)), SourceGeometryTangent())
    _, status = _run_rays(warp_device, source, [(0.5, math.nan), (-1.0, 0.5), (0.5, 1.1)])
    assert np.all(status == int(HistoryStatus.NUMERICAL_FAILURE))
    source.direction_dot = (0.0, 0.0, 1.0)
    _, status = _run_rays(warp_device, source, [(0.5, 0.5)])
    assert np.all(status == int(HistoryStatus.NUMERICAL_FAILURE))
    with pytest.raises(TransportError, match="orthogonal"):
        source_geometry_arguments(
            ParallelBeam((0.0, 0.0, -1.0), (1.0, 1.0)),
            SourceGeometryTangent(direction_dot=(0.0, 0.0, 1.0)),
        )
    with pytest.raises(TransportError, match="cone launch extents"):
        source_geometry_arguments(_cone(), SourceGeometryTangent(extent_dot=(1.0, 0.0)))


def test_native_unrepresentable_affine_operations_fail_whole_history(warp_device: str) -> None:
    for coefficients, derivatives in (
        ((1e308, 1e308, 1e308), (0.0, 0.0, 1.0)),
        ((1e308, 0.0, -5e307), (0.0, 0.0, 1e-300)),
    ):
        chart = _boundary(AffineBoundary("fatal", coefficients, derivatives))
        chart.present = 1
        _, status, _ = _run_boundaries(warp_device, chart)
        assert np.all(status == int(HistoryStatus.NUMERICAL_FAILURE))


def test_native_open_line_mark_that_rounds_to_endpoint_is_fatal(warp_device: str) -> None:
    chart = AffineBoundary("unresolved", (1.0, 1.0, -1.5), (0.0, 0.0, 1.0))
    mark = 1e-300
    first, last = (1.0, 0.5), (0.5, 1.0)
    with localcontext() as context:
        context.prec = 350
        expected = tuple(
            Decimal.from_float(first[i])
            + Decimal.from_float(mark)
            * (Decimal.from_float(last[i]) - Decimal.from_float(first[i]))
            for i in (0, 1)
        )
        assert all(
            Decimal.from_float(min(first[i], last[i]))
            < expected[i]
            < Decimal.from_float(max(first[i], last[i]))
            for i in (0, 1)
        )
    _, status, _ = _run_boundaries(warp_device, _boundary(chart), [mark])
    assert np.all(status == int(HistoryStatus.NUMERICAL_FAILURE))


def test_native_material_shape_coarea_matches_physical_slab_count_derivative(
    warp_device: str,
) -> None:
    # The moving material side x=xi expands the absorbing slab towards +x.
    # Complete original-ray scores are exp(-mu*length) inside and 1 outside.
    # g=x-xi, so positive-minus-negative is 1-exp(-mu*length).
    extent, source_weight, mu, length, velocity = 4.0, 3.7, 0.31, 2.2, -0.6
    chart = AffineBoundary("material:0:1", (extent, 0.0, -1.3), (0.0, 0.0, -velocity))
    output, status, _ = _run_boundaries(warp_device, _boundary(chart))
    assert np.all(status == 0)
    jump = 1.0 - math.exp(-mu * length)
    measured = math.fsum(float(value) * jump * source_weight for value in output[:, 2]) / len(
        status
    )
    expected = -velocity * source_weight / extent * jump
    assert measured == pytest.approx(expected, rel=3e-14, abs=2e-15)
    reversed_chart = AffineBoundary(
        "reversed",
        cast(tuple[float, float, float], tuple(-v for v in chart.coefficients)),
        cast(tuple[float, float, float], tuple(-v for v in chart.coefficients_dot)),
    )
    reversed_output, reversed_status, _ = _run_boundaries(warp_device, _boundary(reversed_chart))
    assert np.all(reversed_status == 0)
    reversed_value = math.fsum(
        float(value) * -jump * source_weight for value in reversed_output[:, 2]
    ) / len(reversed_status)
    assert reversed_value == pytest.approx(measured, rel=3e-14, abs=2e-15)


def test_native_detector_pixel_jump_keeps_signed_cotangent_and_fixed_source_measure(
    warp_device: str,
) -> None:
    extent, panel_velocity, left, right, source_weight = 2.3, 0.4, -1.7, 0.6, 4.2
    chart = AffineBoundary("detector:0:1", (extent, 0.0, -0.9), (0.0, 0.0, -panel_velocity))
    output, status, _ = _run_boundaries(warp_device, _boundary(chart, detector=True))
    assert np.all(status == 0)
    measured = math.fsum(
        float(value) * (right - left) * source_weight for value in output[:, 2]
    ) / len(status)
    expected = panel_velocity / extent * (left - right) * source_weight
    assert measured == pytest.approx(expected, rel=3e-14, abs=2e-15)
    np.testing.assert_array_equal(output[:, 7], np.ones(len(status)))
