"""CPU checks for full-volume resampling and shared physical acquisition."""

import importlib.util
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from dpt.imaging import ct_to_attenuation
from dpt.imaging_io import ImagingMetadata, ImagingVolume
from dpt.volumes import GridSpec


@pytest.fixture(scope="module")
def inputs():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "fullscan_inputs.py"
    spec = importlib.util.spec_from_file_location("fullscan_inputs", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def oblique_grid(shape):
    c, s = math.cos(0.31), math.sin(0.31)
    return GridSpec(shape, (1.25, 2.75, 3.5), (19.0, -37.0, 11.0), (c, 0, s, 0, 1, 0, -s, 0, c))


def test_resampling_preserves_oblique_faces_and_centre(inputs):
    original = oblique_grid((4, 5, 6))
    result = inputs.resampled_grid(original, (9, 3, 8))
    np.testing.assert_allclose(
        inputs.face_corners(result), inputs.face_corners(original), atol=1e-13
    )
    np.testing.assert_allclose(inputs.centre(result), inputs.centre(original), atol=1e-13)
    assert result.orientation == original.orientation
    np.testing.assert_allclose(result.spacing_mm, (1.25 * 6 / 8, 2.75 * 5 / 3, 3.5 * 4 / 9))
    assert result.origin_mm != original.origin_mm


def test_linear_interpolation_matches_independent_eight_corner_rule_and_clamps(inputs):
    grid = oblique_grid((3, 4, 5))
    z, y, x = np.indices(grid.shape)
    values = (17 * x * y + 11 * z**2 - 1073).astype(np.float64)
    volume = ImagingVolume(values, grid, ImagingMetadata("synthetic", "RAS", "HU"))
    shape = (7, 3, 9)
    actual = inputs.resample_hu(volume, shape)
    expected = np.zeros(shape)
    for index in np.ndindex(shape):
        point = (np.asarray(index) + 0.5) * np.asarray(grid.shape) / shape - 0.5
        point = np.clip(point, 0, np.asarray(grid.shape) - 1)
        lower = np.floor(point).astype(int)
        upper = np.minimum(lower + 1, np.asarray(grid.shape) - 1)
        fraction = point - lower
        for corner in itertools.product((0, 1), repeat=3):
            source = tuple(upper[i] if corner[i] else lower[i] for i in range(3))
            weight = math.prod(fraction[i] if corner[i] else 1 - fraction[i] for i in range(3))
            expected[index] += weight * values[source]
    np.testing.assert_allclose(actual.values, expected, rtol=0, atol=2e-11)
    assert actual.values.dtype == np.float64
    assert not actual.values.flags.writeable
    np.testing.assert_array_equal(volume.values, values)
    expected_mu = (np.maximum(expected / 1000.0 + 1.0, 0.0) * 0.02059).astype(np.float32)
    np.testing.assert_array_equal(ct_to_attenuation(actual, energy_kev=60.0).values, expected_mu)


def test_circular_vectors_represent_same_physical_pixels_in_both_frames(inputs):
    grid = GridSpec(
        (256, 256, 256),
        (1.5234379768371582, 1.5234379768371582, 1.357421875),
        (194.6191405057907, 171.3191374540329, -348.3212890625),
        (-1.0, 0, 0, 0, -1.0, 0, 0, 0, 1.0),
    )
    geometry, angles = inputs.circular_geometry(grid)
    assert geometry.shape == (360, 512, 512)
    assert angles[0] == 0 and angles[-1] < 2 * np.pi
    # Subtracting independently rounded angles incurs an error at their scale.
    np.testing.assert_allclose(
        np.diff(angles), 2 * np.pi / 360, rtol=0, atol=4 * np.spacing(2 * np.pi)
    )
    np.testing.assert_allclose(geometry.views[0].source_mm, inputs.centre(grid) - (0, 600, 0))
    world, local = inputs.local_vectors(geometry, grid)
    rotation = np.asarray(grid.orientation).reshape(3, 3)
    for view in (0, 17, 90, 180, 359):
        for row, column in ((0, 0), (213, 307), (511, 511)):
            endpoint = geometry.views[view].pixel_centre(row, column)
            world_endpoint = (
                world[view, 3:6]
                + (column - 255.5) * world[view, 6:9]
                + (row - 255.5) * world[view, 9:12]
            )
            local_endpoint = (
                local[view, 3:6]
                + (column - 255.5) * local[view, 6:9]
                + (row - 255.5) * local[view, 9:12]
            )
            np.testing.assert_allclose(world_endpoint, endpoint, rtol=0, atol=3e-13)
            np.testing.assert_allclose(
                local_endpoint @ rotation.T + inputs.centre(grid), endpoint, rtol=0, atol=5e-13
            )
    coverage = inputs.detector_coverage(grid, geometry)
    assert coverage["all_eight_corners_inside_every_detector"]
    assert coverage["minimum_square_pitch_for_coverage_mm"] * 1.05 == pytest.approx(
        inputs.PIXEL_PITCH_MM, abs=2e-15
    )


def test_resampling_rejects_unknown_units(inputs):
    volume = ImagingVolume(
        np.zeros((2, 3, 4)), oblique_grid((2, 3, 4)), ImagingMetadata("synthetic", "RAS", "unknown")
    )
    with pytest.raises(ValueError, match="declared HU"):
        inputs.resample_hu(volume)
