"""Unrepresentable geometry is fatal even when a selected plane would miss."""

import pytest

from helpers.geometry_ballistic_native import _grid, _ray, _run, wp


@pytest.mark.parametrize(
    "kind",
    [
        "distance",
        "location",
        "collapsed",
        "invalid-grazing",
        "rotation-zero",
        "rotation-reflection",
    ],
)
def test_overflow_and_collapsed_extent_cannot_be_original_null(warp_device, kind):
    grid = _grid((1, 1, 1), (1.0, 1.0, 1.0))
    ray = _ray((-1e308, 0.25, 0.25), (1e-308, 0.0, 1.0))
    if kind == "location":
        grid.origin = wp.vec3d(1e308, 0.0, 0.0)
        grid.spacing = wp.vec3d(1e308, 1.0, 1.0)
        ray = _ray((0.0, 0.25, 0.25), (1.0, 0.0, 0.0))
    elif kind == "collapsed":
        grid.origin = wp.vec3d(1e308, 0.0, 0.0)
        ray = _ray((-1e308, 0.25, 0.25), (1.0, 0.0, 0.0))
    elif kind == "invalid-grazing":
        grid.origin = wp.vec3d(1e308, 0.0, 0.0)
        grid.spacing = wp.vec3d(1e308, 1.0, 1.0)
        ray = _ray((0.0, 0.25, 0.25), (0.0, 0.0, 1.0))
    elif kind == "rotation-zero":
        grid.rotation = wp.mat33d(0.0)
    elif kind == "rotation-reflection":
        grid.rotation = wp.mat33d(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        ray = _ray((-1.0, 0.25, 0.25), (1.0, 0.0, 0.0))
    _, status = _run(ray, grid, [1 if kind == "location" else 0], warp_device)
    assert status[0] == 4
