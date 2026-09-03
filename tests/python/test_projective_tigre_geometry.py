"""Metric camera recovery and official TIGRE physical-coordinate conventions."""

import math
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from dpt.contracts import ContractError
from dpt.geometry import BatchedGeometry, DetectorGeometry, RigidTransform, compose_pose, cross, dot
from dpt.projective import ProjectionMatrix, ProjectiveRows
from dpt.tigre_geometry import TigreConeGeometry


def _acquisition() -> TigreConeGeometry:
    return TigreConeGeometry(
        (7, 11),
        (0.73, 0.91),
        (95.0, 102.0, 99.0),
        (211.0, 218.0, 214.0),
        ((0.31, 0.11, 0.08), (1.29, -0.17, -0.05), (-2.1, 0.23, 0.13)),
        ((0.37, -0.41), (-0.23, 0.53), (0.61, -0.73)),
        ((0.07, -0.09, 0.11), (-0.12, 0.15, -0.18), (0.21, -0.24, 0.27)),
        ((0.2, -0.3, 0.4), (-0.5, 0.6, -0.7), (0.8, -0.9, 1.0)),
        (0.13, -0.19, 0.23),
    )


def _rotate(
    point: tuple[float, float, float], axis: int, angle: float
) -> tuple[float, float, float]:
    # Independent axis-wise Rodrigues application, without production matrices.
    result = list(point)
    first, second = ((1, 2), (2, 0), (0, 1))[axis]
    result[first] = math.cos(angle) * point[first] - math.sin(angle) * point[second]
    result[second] = math.sin(angle) * point[first] + math.cos(angle) * point[second]
    return result[0], result[1], result[2]


def _official_segment(
    record: TigreConeGeometry, view: int, row: int, column: int
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    # Geometry-only interpretation of the pinned official computeDeltas +
    # Python _types.pxd mappings. Kernel pixelV reverses the internal V axis.
    rows, columns = record.shape
    dv, du = record.spacing_rows_columns_mm
    pixel = (0.0, (column - (columns - 1) / 2) * du, (row - (rows - 1) / 2) * dv)
    rx, ry, rz = record.detector_rotation_xyz_rad[view]
    for axis, angle in ((0, rx), (1, ry), (2, rz)):
        pixel = _rotate(pixel, axis, angle)
    sod, sdd = record.source_object_distance_mm[view], record.source_detector_distance_mm[view]
    off_v, off_u = record.detector_offset_rows_columns_mm[view]
    pixel = (pixel[0] + sod - sdd, pixel[1] + off_u, pixel[2] + off_v)
    source = (sod, 0.0, 0.0)
    alpha, theta, psi = record.angles_rad[view]
    for axis, angle in ((2, psi), (1, theta), (2, alpha)):
        pixel, source = _rotate(pixel, axis, angle), _rotate(source, axis, angle)
    off_z, off_y, off_x = record.origin_offset_zyx_mm[view]
    cor = record.centre_rotation_mm[view]
    shift = (-off_x - cor * math.sin(alpha), -off_y + cor * math.cos(alpha), -off_z)
    return (
        tuple(source[axis] + shift[axis] for axis in range(3)),
        tuple(pixel[axis] + shift[axis] for axis in range(3)),
    )  # type: ignore[return-value]


def test_tigre_full_tilt_zyz_origin_cor_and_non_square_pixels_match_official_coordinates() -> None:
    record = _acquisition()
    world = compose_pose(RigidTransform(), (1.1, -2.3, 3.7, 0.12, -0.21, 0.31))
    batch = record.to_dpt(world_from_tigre=world)
    for view in range(3):
        for row in range(7):
            for column in range(11):
                source, endpoint = _official_segment(record, view, row, column)
                actual_source, actual_endpoint = batch.segment(view, row, column)
                assert actual_source == pytest.approx(world.point(source), abs=8e-14)
                assert actual_endpoint == pytest.approx(world.point(endpoint), abs=8e-14)
    exported = TigreConeGeometry.from_dpt(
        batch,
        angles_rad=record.angles_rad,
        world_from_tigre=world,
        origin_offset_zyx_mm=record.origin_offset_zyx_mm,
        centre_rotation_mm=record.centre_rotation_mm,
    )
    assert exported.source_object_distance_mm == pytest.approx(
        record.source_object_distance_mm, abs=5e-14
    )
    assert exported.source_detector_distance_mm == pytest.approx(
        record.source_detector_distance_mm, abs=2e-13
    )
    for view in range(3):
        assert exported.detector_rotation_xyz_rad[view] == pytest.approx(
            record.detector_rotation_xyz_rad[view], abs=5e-15
        )
        assert exported.detector_offset_rows_columns_mm[view] == pytest.approx(
            record.detector_offset_rows_columns_mm[view], abs=6e-14
        )
        for actual, expected in zip(
            exported.to_dpt(world_from_tigre=world).segment(view, 2, 7),
            batch.segment(view, 2, 7),
            strict=True,
        ):
            assert actual == pytest.approx(expected, abs=2e-13)


@pytest.mark.parametrize("scale", [1.0, -1.0, 1e200, -1e200, 1e-200, -1e-200])
def test_projection_matrix_metric_roundtrip_preserves_homogeneous_sign_and_physical_hemisphere(
    scale: float,
) -> None:
    batch = _acquisition().to_dpt()
    for panel in batch.views:
        assert isinstance(panel, DetectorGeometry)
        original = ProjectionMatrix.from_detector(panel)
        matrix = ProjectionMatrix(
            cast(
                ProjectiveRows,
                tuple(tuple(scale * entry for entry in row) for row in original.rows),
            )
        )
        offset = tuple(panel.origin_mm[axis] - panel.source_mm[axis] for axis in range(3))
        distance = abs(dot(cast(tuple[float, float, float], offset), cross(panel.u, panel.v)))
        recovered = matrix.to_detector(
            shape=panel.shape,
            detector_distance_mm=distance,
            positive_depth_point_mm=panel.pixel_centre(3, 5),
        )
        assert recovered.source_mm == pytest.approx(panel.source_mm, abs=2e-12)
        assert recovered.origin_mm == pytest.approx(panel.origin_mm, abs=2e-12)
        assert recovered.spacing_mm == pytest.approx(panel.spacing_mm, abs=2e-14)
        assert recovered.u == pytest.approx(panel.u, abs=2e-14)
        assert recovered.v == pytest.approx(panel.v, abs=2e-14)
        for row, column in ((0, 0), (2, 7), (6, 10)):
            endpoint = panel.pixel_centre(row, column)
            assert original.project(endpoint) == pytest.approx((row, column), abs=2e-13)
            point = tuple(
                panel.source_mm[axis] + 0.43 * (endpoint[axis] - panel.source_mm[axis])
                for axis in range(3)
            )
            assert ProjectionMatrix.from_detector(recovered).project(
                cast(tuple[float, float, float], point)
            ) == pytest.approx((row, column), abs=5e-13)


@pytest.mark.parametrize("pitch", [-math.pi / 2, math.pi / 2])
def test_tigre_euler_gimbal_export_preserves_rays_without_inventing_unique_angles(
    pitch: float,
) -> None:
    record = TigreConeGeometry(
        (3, 5),
        (0.7, 0.9),
        (100.0,),
        (220.0,),
        ((0.31, 0.11, -0.19),),
        ((0.73, 0.41),),
        ((0.13, pitch, -0.21),),
    )
    batch = record.to_dpt()
    exported = TigreConeGeometry.from_dpt(batch, angles_rad=record.angles_rad)
    for row, col in ((0, 0), (1, 2), (2, 4)):
        for actual, expected in zip(
            exported.to_dpt().segment(0, row, col), batch.segment(0, row, col), strict=True
        ):
            assert actual == pytest.approx(expected, abs=7e-13)


def test_native_tigre_roundtrip_owns_arrays_and_retains_python_axis_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _acquisition()
    fake = ModuleType("tigre")
    fake.geometry = SimpleNamespace  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tigre", fake)
    native, angles = record.to_tigre()
    copied = TigreConeGeometry.from_tigre(native, angles)
    assert copied == record
    native.offDetector[0, 0] += 7
    native.offOrigin[1, 2] += 9
    angles[2, 1] += 11
    assert copied == record
    assert native.nDetector.tolist() == [7, 11]
    assert native.dDetector.tolist() == [0.73, 0.91]
    scalar = SimpleNamespace(
        mode="cone", nDetector=(3, 5), dDetector=(0.7, 0.9), DSO=100.0, DSD=200.0
    )
    broadcast = TigreConeGeometry.from_tigre(scalar, (0.2, 0.7))
    assert broadcast.angles_rad == ((0.2, 0.0, 0.0), (0.7, 0.0, 0.0))
    assert broadcast.source_object_distance_mm == (100.0, 100.0)


def test_projective_and_tigre_reject_skew_degeneracy_missing_metric_and_wrong_orbit() -> None:
    panel = _acquisition().to_dpt().views[0]
    assert isinstance(panel, DetectorGeometry)
    matrix = ProjectionMatrix.from_detector(panel)
    arguments: dict[str, Any] = dict(
        shape=panel.shape, detector_distance_mm=200.0, positive_depth_point_mm=panel.origin_mm
    )
    with pytest.raises(ContractError, match="positive"):
        matrix.to_detector(**{**arguments, "detector_distance_mm": 0.0})
    with pytest.raises(ContractError, match="source plane"):
        ProjectionMatrix(
            ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))
        ).to_detector(**{**arguments, "positive_depth_point_mm": (1.0, 2.0, 0.0)})
    skew = tuple(matrix.rows[0][axis] + 0.3 * matrix.rows[1][axis] for axis in range(4))
    with pytest.raises(ContractError, match="orthogonal"):
        ProjectionMatrix(cast(ProjectiveRows, (skew, matrix.rows[1], matrix.rows[2]))).to_detector(
            **arguments
        )
    with pytest.raises(ContractError, match="perspective"):
        ProjectionMatrix(
            ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        ).to_detector(**arguments)
    with pytest.raises(ContractError, match="finite positive"):
        matrix.project(
            cast(
                tuple[float, float, float],
                tuple(2 * panel.source_mm[axis] - panel.origin_mm[axis] for axis in range(3)),
            )
        )
    with pytest.raises(ContractError, match="orbit"):
        TigreConeGeometry.from_dpt(BatchedGeometry((panel,)), angles_rad=((0.0, 0.0, 0.0),))
