"""Native selected-face crossings against exact plane and patch references."""

# Warp annotations name runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
import importlib
import math
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from helpers.geometry_ballistic_native import (
    _decode,
    _grid,
    _ray,
    _run,
)

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.geometry_ballistic_kernels")
interfaces: Any = importlib.import_module("dpt.transport.geometry_interface_kernels")
coarea: Any = importlib.import_module("dpt.transport.compton_geometry_boundary_kernels")


def test_all_original_ray_planes_match_exact_rational_crossings(warp_device):
    output, status = _run(_ray(), _grid(), list(range(12)), warp_device)
    np.testing.assert_array_equal(status[:7], -1)
    np.testing.assert_array_equal(status[7:], 0)
    for face in range(5):
        row = output[7 + face]
        assert row[0] == float(Fraction(1) + Fraction(face, 4))
        assert row[1] == 1.0
        np.testing.assert_array_equal(
            row[2:6],
            (2, face, -1 if face == 0 else (face - 1) * 6 + 2, -1 if face == 4 else face * 6 + 2),
        )
        np.testing.assert_allclose(row[6:9], (0.35, 0.6, face / 4), rtol=0, atol=2e-16)
        np.testing.assert_array_equal(row[9:12], (0.1, -0.2, 0.3))
        np.testing.assert_array_equal(row[12:], 0.0)


def test_every_patch_matches_independent_face_enumeration(warp_device):
    shape = (3, 2, 2)
    expected = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for face in range(shape[axis] + 1):
            for first in range(shape[other[0]]):
                for second in range(shape[other[1]]):
                    cells = [0, 0, 0]
                    cells[other[0]], cells[other[1]] = first, second
                    cells[axis] = face - 1
                    negative = (
                        -1 if face == 0 else cells[0] + shape[0] * (cells[1] + shape[1] * cells[2])
                    )
                    cells[axis] = face
                    positive = (
                        -1
                        if face == shape[axis]
                        else cells[0] + shape[0] * (cells[1] + shape[1] * cells[2])
                    )
                    expected.append((0, axis, face, first, second, negative, positive))
    output = _decode(_grid(shape), list(range(len(expected))), warp_device)
    np.testing.assert_array_equal(output[:, :7], expected)
    np.testing.assert_array_equal(output[:, 8], sum(shape) + 3)
    np.testing.assert_array_equal(output[:, 9], len(expected))
    np.testing.assert_array_equal(output[: sum(shape) + 3, 7], 0)
    np.testing.assert_array_equal(output[sum(shape) + 3 :, 7], 4)


@pytest.mark.parametrize("index", [-1, 12, 2147483647])
def test_invalid_selected_plane_is_fatal(warp_device, index):
    output, status = _run(_ray(), _grid(), [index], warp_device)
    assert status[0] == 4
    np.testing.assert_array_equal(output[0], 0.0)


@pytest.mark.parametrize("origin", [(0.0, 0.6, -1.0), (0.35, 0.5, -1.0), (1.4, 0.6, -1.0)])
def test_exact_patch_edges_are_fatal(warp_device, origin):
    output, status = _run(_ray(origin), _grid(), [7], warp_device)
    assert status[0] == 4
    np.testing.assert_array_equal(output[0], 0.0)


@pytest.mark.parametrize(
    "origin,direction",
    [
        ((2.0, 0.6, -1.0), (0.0, 0.0, 1.0)),
        ((0.35, 0.6, 1.5), (0.0, 0.0, 1.0)),
        ((0.35, 0.6, -1.0), (1.0, 0.0, 0.0)),
    ],
)
def test_genuine_misses_and_grazing_keep_null_status(warp_device, origin, direction):
    output, status = _run(_ray(origin, direction), _grid(), [7], warp_device)
    assert status[0] == -1
    np.testing.assert_array_equal(output[0], 0.0)


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_rotated_world_crossing_preserves_coarea_and_source_gauge(scale, warp_device):
    grid = _grid(scale=scale, rotated=True)
    ray = _ray((-0.5 * scale, 0.1 * scale, 0.725 * scale), (1.0, 0.0, 0.0))
    output, status = _run(ray, grid, [9], warp_device)
    assert status[0] == 0
    np.testing.assert_allclose(
        output[0, 6:9], (scale, 0.1 * scale, 0.725 * scale), rtol=3e-15, atol=0
    )
    assert output[0, 0] == pytest.approx(1.5 * scale, rel=3e-15)
    assert output[0, 1] == 1
    np.testing.assert_array_equal(output[0, 12:], 0)


def test_oblique_inverse_normal_projection(warp_device):
    direction = (0.0, 0.6, 0.8)
    output, status = _run(_ray((0.35, 0.2, -0.2), direction), _grid(), [7], warp_device)
    assert status[0] == 0
    assert output[0, 0] == 0.25
    assert output[0, 1] == 1.25


@pytest.mark.parametrize("shape", [(0, 1, 1), (65536, 65536, 1), (2147483647, 1, 1)])
def test_integer_budget_is_checked_before_signed_products(warp_device, shape):
    output = _decode(_grid(shape), [0], warp_device)
    assert output[0, 0] == 4
    assert output[0, 7] == 4
    assert output[0, 8] == output[0, 9] == -1


@pytest.mark.parametrize("direction", [(0.0, 0.0, 2.0), (0.0, math.nan, 1.0)])
def test_invalid_physical_ray_is_fatal(warp_device, direction):
    _, status = _run(_ray(direction=direction), _grid(), [7], warp_device)
    assert status[0] == 4
