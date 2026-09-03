"""Huge finite direction jets retain the unit-vector tangent condition."""

# Warp annotations are runtime DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
import importlib
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pytest

wp: Any = importlib.import_module("warp")
interface: Any = importlib.import_module("dpt.transport.geometry_interface_kernels")
source: Any = importlib.import_module("dpt.transport.geometry_source_boundary_kernels")


@wp.kernel
def _gauge(
    direction: wp.vec3d,
    derivative: wp.vec3d,
    output: wp.array(dtype=wp.vec3d),
    status: wp.array(dtype=wp.int32),
):
    result = interface.source_path_gauge(wp.vec3d(), direction, wp.vec3d(), derivative)
    status[0] = result.status
    if result.status == 0:
        output[0] = result.angular


@wp.kernel
def _launch(
    direction: wp.vec3d,
    derivative: wp.vec3d,
    output: wp.array(dtype=wp.vec3d),
    status: wp.array(dtype=wp.int32),
):
    arguments = source.NativeSourceGeometry()
    arguments.kind = 0
    arguments.direction = direction
    arguments.direction_dot = derivative
    arguments.weight = wp.float64(1.0)
    result = source.source_ray_tangent(arguments, wp.vec2d(wp.float64(0.25), wp.float64(0.75)))
    status[0] = result.status
    if result.status == 0:
        output[0] = result.ray.direction_dot


CASES = [
    ((1.0, 0.0, 0.0), (1.7e308, 1.7e308, 1.7e308), False),
    ((1.0, 0.0, 0.0), (0.0, 1.7e308, 1.7e308), True),
    ((1.0, 0.0, 0.0), (1.0e296, 1.7e308, 1.7e308), True),
    ((1.0, 0.0, 0.0), (5.0e296, 1.7e308, 1.7e308), False),
    ((0.0, 0.0, 1.0), (1.7e308, 1.7e308, 0.0), True),
    ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0), True),
    ((1.0, 0.0, 0.0), (1.0, 1.0, 1.0), False),
]


@pytest.mark.parametrize("direction,derivative,admitted", CASES)
@pytest.mark.parametrize("primitive", ["gauge", "launch"])
def test_direction_jet_relative_orthogonality(
    warp_device: str,
    direction: tuple[float, float, float],
    derivative: tuple[float, float, float],
    admitted: bool,
    primitive: str,
) -> None:
    # Independent exact binary64 inputs, with a 100-digit Euclidean reference.
    with localcontext() as context:
        context.prec = 100
        exact_direction = tuple(Decimal.from_float(v) for v in direction)
        exact_derivative = tuple(Decimal.from_float(v) for v in derivative)
        projection = abs(
            sum((a * b for a, b in zip(exact_direction, exact_derivative, strict=True)), Decimal(0))
        )
        norm = sum((v * v for v in exact_derivative), Decimal(0)).sqrt()
        assert (projection <= Decimal.from_float(1.0e-12) * max(Decimal(1), norm)) == admitted
        cross = tuple(
            exact_direction[(i + 1) % 3] * exact_derivative[(i + 2) % 3]
            - exact_direction[(i + 2) % 3] * exact_derivative[(i + 1) % 3]
            for i in range(3)
        )
    output = wp.array(np.full((1, 3), np.nan), dtype=wp.vec3d, device=warp_device)
    status = wp.full(1, -9, dtype=wp.int32, device=warp_device)
    kernel = _gauge if primitive == "gauge" else _launch
    wp.launch(
        kernel,
        1,
        inputs=[wp.vec3d(*direction), wp.vec3d(*derivative)],
        outputs=[output, status],
        device=warp_device,
        record_tape=False,
    )
    assert int(status.numpy()[0]) == (0 if admitted else 4)
    actual = output.numpy()[0]
    if admitted:
        expected = np.asarray([float(v) for v in cross] if primitive == "gauge" else derivative)
        np.testing.assert_array_equal(actual, expected)
        assert np.isfinite(actual).all()
    else:
        assert np.isnan(actual).all()
