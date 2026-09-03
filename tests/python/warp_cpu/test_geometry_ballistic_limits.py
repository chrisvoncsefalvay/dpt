"""Constant-space decoders retain the last admissible signed-index faces."""

import math

import numpy as np
import pytest

from helpers.geometry_ballistic_native import _decode, _grid, _ray, _run, wp


def test_last_admissible_patch_count_preserves_axis_splits(warp_device):
    size = 429496729
    indices = [0, 429496729, 429496730, 1288490187, 2147483645]
    output = _decode(_grid((size, 1, 1), (1.0, 1.0, 1.0)), indices, warp_device)
    expected = [
        (0, 0, 0, 0, 0, -1, 0),
        (0, 0, size, 0, 0, size - 1, -1),
        (0, 1, 0, 0, 0, -1, 0),
        (0, 1, 1, size - 1, 0, size - 1, -1),
        (0, 2, 1, size - 1, 0, size - 1, -1),
    ]
    np.testing.assert_array_equal(output[:, :7], expected)
    np.testing.assert_array_equal(output[:, 9], 2147483646)


def test_last_admissible_plane_crosses_when_patch_population_exceeds_int32(warp_device):
    grid = _grid((2147483642, 1, 1), (1.0, 1.0, 1.0))
    last = 2147483646
    decoded = _decode(grid, [last], warp_device)
    np.testing.assert_array_equal(decoded[0, 7:], (0, 2147483647, -1))
    output, status = _run(_ray((0.25, 0.25, -1.0)), grid, [last], warp_device)
    assert status[0] == 0
    np.testing.assert_array_equal(output[0, :4], (2.0, 1.0, 2.0, 1.0))


@pytest.mark.parametrize("diagonal", [2.0, math.nan, math.inf])
def test_malformed_rotation_is_fatal_before_parallel_null(warp_device, diagonal):
    grid = _grid((1, 1, 1), (1.0, 1.0, 1.0))
    grid.rotation = wp.mat33d(diagonal, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    _, status = _run(_ray(direction=(0.0, 0.0, 1.0)), grid, [0], warp_device)
    assert status[0] == 4
