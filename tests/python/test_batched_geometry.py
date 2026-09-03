"""Independent projective, circle, intrinsic and ASTRA coordinate oracles."""

import dataclasses
import math
from typing import cast

import pytest

from dpt.contracts import ContractError
from dpt.geometry import (
    AstraVectorGeometry,
    BatchedGeometry,
    ConeBeamIntrinsics,
    CurvedDetectorGeometry,
    DetectorGeometry,
    ParallelGeometry,
    Vector3,
)


def test_projection_matrix_intersects_oblique_panel() -> None:
    panel = DetectorGeometry(
        (1.0, -12.0, 3.0), (-4.0, 8.0, -2.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.6, 1.2), (9, 7)
    )
    endpoint = panel.pixel_centre(5, 3)
    # All points on the same line have the same projective pixel coordinate.
    for fraction in (0.1, 0.5, 1.0, 2.0):
        point = tuple(
            panel.source_mm[i] + fraction * (endpoint[i] - panel.source_mm[i]) for i in range(3)
        )
        assert panel.project_point(cast(Vector3, point)) == pytest.approx((5.0, 3.0), abs=1e-13)
    with pytest.raises(ContractError, match="parallel"):
        panel.project_point(panel.source_mm)


def test_parallel_segments_are_translated_equal_length_lines() -> None:
    panel = ParallelGeometry(
        (-2.0, 10.0, -3.0),
        (0.0, 1.0, 0.0),
        40.0,
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.7),
        (3, 4),
    )
    for row, column in ((0, 0), (2, 3)):
        source, endpoint = panel.segment(row, column)
        assert tuple(endpoint[i] - source[i] for i in range(3)) == pytest.approx((0.0, 40.0, 0.0))
        assert endpoint == pytest.approx((-2.0 + column * 0.5, 10.0, -3.0 + row * 0.7))
    with pytest.raises(ContractError, match="unit"):
        dataclasses.replace(panel, direction=(0.0, 2.0, 0.0))


def test_curved_columns_lie_on_cylinder_with_arc_spacing() -> None:
    panel = CurvedDetectorGeometry(
        (0.0, 0.0, 0.0), (0.0, 100.0, -2.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (2.0, 0.5), (5, 21)
    )
    for column in (0, 1, 20):
        x, y, z = panel.pixel_centre(3, column)
        assert math.hypot(x, y) == pytest.approx(100.0, abs=2e-14)
        assert math.atan2(x, y) == pytest.approx(0.02 * column, abs=1e-15)
        assert z == -0.5
    with pytest.raises(ContractError, match="tangent"):
        dataclasses.replace(panel, origin_mm=(1.0, 100.0, -2.0))


@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (1, 1), (1, 8)])
def test_astra_cone_vectors_have_independent_centre_and_pixel_steps(shape: tuple[int, int]) -> None:
    rows, columns = shape
    vector = (10.0, -100.0, 8.0, 2.0, 200.0, -3.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.9)
    imported = BatchedGeometry.from_astra_vectors(AstraVectorGeometry("cone_vec", shape, (vector,)))
    for row, column in ((0, 0), (rows - 1, columns - 1)):
        source, endpoint = imported.segment(0, row, column)
        assert source == (10.0, -100.0, 8.0)
        assert endpoint == pytest.approx(
            (2.0 + (column - (columns - 1) / 2) * 0.4, 200.0, -3.0 + (row - (rows - 1) / 2) * 0.9)
        )
    assert imported.to_astra_vectors().vectors[0] == pytest.approx(vector)


def test_astra_parallel_units_direction_and_finite_length() -> None:
    geometry = AstraVectorGeometry(
        "parallel3d_vec", (2, 3), ((0.0, 2.0, 0.0, 1.0, 8.0, -2.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2),)
    )
    with pytest.raises(ContractError, match="parallel_length"):
        BatchedGeometry.from_astra_vectors(geometry)
    imported = BatchedGeometry.from_astra_vectors(
        geometry, length_scale_mm=10.0, parallel_length_mm=100.0
    )
    source, endpoint = imported.segment(0, 0, 0)
    assert endpoint == (9.0, 80.0, -21.0)
    assert source == (9.0, -20.0, -21.0)
    assert imported.to_astra_vectors().vectors[0] == pytest.approx(
        (0.0, 1.0, 0.0, 10.0, 80.0, -20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0)
    )


def test_intrinsic_ray_jacobian_matches_independent_central_differences() -> None:
    intrinsics = ConeBeamIntrinsics(450.0, 900.0, (4, 7), (0.3, 0.4), (1.2, -0.8), (2.0, 3.0, 4.0))
    step = 1e-4
    names = ("source_object_distance_mm", "source_detector_distance_mm")
    angle = 0.7
    expected = intrinsics.ray_derivatives(angle)
    for parameter in range(4):
        if parameter < 2:
            name = names[parameter]
            plus = dataclasses.replace(intrinsics, **{name: getattr(intrinsics, name) + step})
            minus = dataclasses.replace(intrinsics, **{name: getattr(intrinsics, name) - step})
        else:
            axis = parameter - 2
            plus = dataclasses.replace(
                intrinsics,
                detector_offset_mm=tuple(
                    x + (step if i == axis else 0.0)
                    for i, x in enumerate(intrinsics.detector_offset_mm)
                ),
            )
            minus = dataclasses.replace(
                intrinsics,
                detector_offset_mm=tuple(
                    x - (step if i == axis else 0.0)
                    for i, x in enumerate(intrinsics.detector_offset_mm)
                ),
            )
        plus_view, minus_view = plus.view(angle), minus.view(angle)
        actual_source = tuple(
            (plus_view.source_mm[i] - minus_view.source_mm[i]) / (2 * step) for i in range(3)
        )
        for row, column in ((0, 0), (3, 6)):
            actual_endpoint = tuple(
                (plus_view.pixel_centre(row, column)[i] - minus_view.pixel_centre(row, column)[i])
                / (2 * step)
                for i in range(3)
            )
            assert actual_source == pytest.approx(expected[parameter][0], abs=5e-10)
            assert actual_endpoint == pytest.approx(expected[parameter][1], abs=5e-10)


def test_batch_requires_common_shape_and_retains_only_view_metadata() -> None:
    intrinsics = ConeBeamIntrinsics(100.0, 200.0, (512, 512), (0.5, 0.5))
    batch = intrinsics.batch((0.0, 1.0, 2.0))
    assert batch.shape == (3, 512, 512)
    assert batch.view_offsets == (0, 262144, 524288, 786432)
    assert len(batch.views) == 3
    assert not hasattr(batch, "sources_mm")
    with pytest.raises(ContractError, match="share"):
        BatchedGeometry((batch.views[0], dataclasses.replace(batch.views[0], shape=(3, 4))))


def test_fan_geometry_has_one_row_and_expected_angle() -> None:
    batch = ConeBeamIntrinsics(100.0, 250.0, (1, 5), (2.0, 1.0)).batch((math.pi / 2,))
    source, endpoint = batch.segment(0, 0, 2)
    assert source == pytest.approx((100.0, 0.0, 0.0), abs=1e-13)
    assert endpoint == pytest.approx((-150.0, 0.0, 0.0), abs=1e-13)
