"""CPU checks for the physical frame and bounded-memory comparison helpers."""

from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks.competitors import tigre_circular_parameters
from benchmarks.fullscan_competitors import (
    box_comparison,
    box_view,
    compact_rays,
    compare_arrays,
    inner_product,
)


def circular_vectors(views=360, pitch=2.6375731636032262):
    # RAS world zero source is -Y, while this CT's local frame has -X/-Y axes.
    theta = 2 * np.pi * np.arange(views) / views
    angle = theta + np.pi / 2
    source = 600 * np.column_stack((np.cos(angle), np.sin(angle), np.zeros(views)))
    columns = pitch * np.column_stack((-np.sin(angle), np.cos(angle), np.zeros(views)))
    rows = np.tile([0, 0, pitch], (views, 1))
    return np.column_stack((source, -source, columns, rows))


def test_full_circle_tigre_mapping_retains_each_physical_ray():
    vectors = circular_vectors()
    angles, sod, sdd, residuals = tigre_circular_parameters(
        {
            "vectors": vectors,
            "sources_mm": vectors[:, :3],
        }
    )
    np.testing.assert_allclose(sod, 600, atol=1e-12)
    np.testing.assert_allclose(sdd, 1200, atol=1e-12)
    np.testing.assert_allclose(np.cos(angles), vectors[:, 0] / 600, atol=1e-15)
    np.testing.assert_allclose(np.sin(angles), vectors[:, 1] / 600, atol=1e-15)
    assert max(residuals.values()) < 1e-10
    np.testing.assert_allclose(vectors[0, 6:9], [-2.6375731636032262, 0, 0], atol=1e-15)


def test_tigre_mapping_rejects_unrecentered_world_geometry():
    vectors = circular_vectors(4)
    vectors[:, :3] += [0.38, -22.9, -175.25]
    vectors[:, 3:6] += [0.38, -22.9, -175.25]
    with pytest.raises(ValueError, match="not represented"):
        tigre_circular_parameters({"vectors": vectors, "sources_mm": vectors[:, :3]})


@pytest.mark.parametrize("shape", [(5, 7), (4, 6)])
def test_physical_box_uses_detector_pixel_centres(shape):
    vector = np.array([0, -20, 0, 0, 20, 0, 2, 0, 0, 0, 0, 3], dtype=np.float64)
    actual = box_view(vector, shape, [-100, -4, -100], [100, 4, 100], value=0.02)
    rows = 3 * (np.arange(shape[0]) - (shape[0] - 1) / 2)
    cols = 2 * (np.arange(shape[1]) - (shape[1] - 1) / 2)
    expected = 0.02 * 8 * np.sqrt(1 + (rows[:, None] ** 2 + cols[None, :] ** 2) / 40**2)
    np.testing.assert_allclose(actual, expected, rtol=5e-15)
    np.testing.assert_allclose(actual, actual[::-1, ::-1], rtol=0, atol=0)


def test_box_intersection_handles_parallel_and_finite_ray_support():
    vector = np.array([0, -20, 0, 0, 20, 0, 1, 0, 0, 0, 0, 1], dtype=np.float64)
    assert box_view(vector, (1, 1), [-2, -4, -3], [2, 4, 3])[0, 0] == pytest.approx(0.08)
    assert box_view(vector, (1, 1), [1, -4, -3], [2, 4, 3])[0, 0] == 0
    assert box_view(vector, (1, 1), [-2, 21, -3], [2, 24, 3])[0, 0] == 0


def test_marker_probe_detects_mirrored_detector_columns():
    rays = {"vectors": circular_vectors(4, 1), "views": 4, "image_shape": (41, 41)}
    low, high = np.array([2, 3, -4]), np.array([5, 7, 0])
    actual = np.array(
        [box_view(vector, rays["image_shape"], low, high) for vector in rays["vectors"]]
    )
    exact = box_comparison(actual, rays, low, high, centroids=True)
    mirrored = box_comparison(actual[..., ::-1], rays, low, high, centroids=True)
    assert exact["relative_l2"] == 0
    assert max(exact["centroid_distance_pixels"]) == 0
    assert mirrored["relative_l2"] > 0.5
    assert max(mirrored["centroid_distance_pixels"]) > 2


def test_chunked_metrics_and_inner_product_match_complete_fp64_calculation():
    expected = np.arange(120, dtype=np.float64).reshape(5, 4, 6) / 17
    actual = (expected + np.sin(expected) * 0.03).astype(np.float32)
    delta = actual.astype(np.float64) - expected
    metrics = compare_arrays(actual, expected)
    assert metrics["relative_l2"] == pytest.approx(np.linalg.norm(delta) / np.linalg.norm(expected))
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean(delta**2)))
    assert metrics["max_abs"] == np.max(np.abs(delta))
    assert inner_product(actual, expected) == pytest.approx(
        np.sum(actual.astype(np.float64) * expected)
    )


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_metrics_reject_nonfinite_values(invalid):
    with pytest.raises(ArithmeticError, match="nonfinite"):
        compare_arrays(np.array([[invalid]]), np.ones((1, 1)))


def test_compact_geometry_retains_only_twelve_values_per_view(tmp_path):
    vectors = circular_vectors()
    np.save(tmp_path / "vectors.npy", vectors)
    prepared = SimpleNamespace(
        volume=SimpleNamespace(
            values=np.empty((2, 3, 4)), grid=SimpleNamespace(spacing_mm=(1, 2, 3))
        ),
        metadata={
            "arrays": {"vectors_local": {"file": "vectors.npy"}},
            "acquisition": {"views": 360, "detector_shape": [512, 512]},
        },
    )
    rays = compact_rays(prepared, tmp_path)
    np.testing.assert_array_equal(rays["vectors"], vectors)
    assert rays["vectors"].nbytes == 360 * 12 * 8
    assert rays["shape_zyx"] == (2, 3, 4)
    assert "targets_mm" not in rays
