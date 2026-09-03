"""Explicit axis names preserve legacy layouts and anisotropic physical maps."""

# pyright: reportUnknownMemberType=false

import pytest

from dpt.contracts import ContractError
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.volumes import GridSpec


def test_named_grid_preserves_legacy_tuple_order_and_defaults() -> None:
    grid = GridSpec.from_axes(
        nx=7, ny=5, nz=3, spacing_x_mm=0.25, spacing_y_mm=1.5, spacing_z_mm=4.0
    )
    assert grid == GridSpec((3, 5, 7), (0.25, 1.5, 4.0))
    assert grid.shape_zyx == grid.shape == (3, 5, 7)
    assert grid.spacing_xyz_mm == grid.spacing_mm == (0.25, 1.5, 4.0)
    assert grid.voxels == 105
    assert grid.support == ((-0.5, -0.5, -0.5), (6.5, 4.5, 2.5))


def test_named_anisotropic_grid_preserves_oriented_coordinate_maps() -> None:
    grid = GridSpec.from_axes(
        nx=7,
        ny=5,
        nz=3,
        spacing_x_mm=0.25,
        spacing_y_mm=1.5,
        spacing_z_mm=4.0,
        origin_mm=(10.0, 20.0, 30.0),
        orientation=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    index_xyz = (2.0, 3.0, 1.25)
    point_mm = (5.5, 20.5, 35.0)
    assert grid.grid_to_object(index_xyz) == pytest.approx(point_mm)
    assert grid.object_to_grid(point_mm) == pytest.approx(index_xyz)


def test_named_detector_preserves_legacy_tuple_order_and_pixel_centres() -> None:
    detector = DetectorGeometry.from_axes(
        source_mm=(-50.0, 15.0, 10.0),
        origin_mm=(10.0, 20.0, 30.0),
        u=(0.0, 0.0, 1.0),
        v=(0.0, 1.0, 0.0),
        rows=3,
        columns=7,
        spacing_column_mm=0.25,
        spacing_row_mm=1.5,
    )
    assert detector == DetectorGeometry(
        (-50.0, 15.0, 10.0),
        (10.0, 20.0, 30.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (0.25, 1.5),
        (3, 7),
    )
    assert detector.shape_rows_columns == detector.shape == (3, 7)
    assert detector.spacing_columns_rows_mm == detector.spacing_mm == (0.25, 1.5)
    assert detector.pixels == 21
    assert detector.pixel_centre(2, 6) == pytest.approx((10.0, 23.0, 31.5))
    with pytest.raises(ContractError, match="pixel index outside detector"):
        detector.pixel_centre(3, 0)
    with pytest.raises(ContractError, match="pixel index outside detector"):
        detector.pixel_centre(0, 7)


def test_named_detector_axes_follow_rigid_transform() -> None:
    transform = RigidTransform((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), (10.0, 20.0, 30.0))
    detector = DetectorGeometry.from_axes(
        source_mm=transform.point((0.0, 0.0, -10.0)),
        origin_mm=transform.point((1.0, 2.0, 10.0)),
        u=transform.direction((1.0, 0.0, 0.0)),
        v=transform.direction((0.0, 1.0, 0.0)),
        rows=3,
        columns=7,
        spacing_column_mm=0.25,
        spacing_row_mm=1.5,
    )
    assert detector.pixel_centre(2, 6) == pytest.approx((5.0, 22.5, 40.0))


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("dimension", [0, -1, True, 1.5, 2**31])
def test_named_grid_rejects_invalid_dimensions(axis: str, dimension: int) -> None:
    with pytest.raises(ContractError, match=r"grid shape|signed 32-bit indexing"):
        GridSpec.from_axes(
            nx=dimension if axis == "x" else 7,
            ny=dimension if axis == "y" else 5,
            nz=dimension if axis == "z" else 3,
            spacing_x_mm=0.25,
            spacing_y_mm=1.5,
            spacing_z_mm=4.0,
        )


@pytest.mark.parametrize("axis", ["rows", "columns"])
@pytest.mark.parametrize("dimension", [0, -1, True, 1.5, 2**31])
def test_named_detector_rejects_invalid_dimensions(axis: str, dimension: int) -> None:
    with pytest.raises(ContractError, match=r"detector shape|signed 32-bit indexing"):
        DetectorGeometry.from_axes(
            source_mm=(0.0, 0.0, -10.0),
            origin_mm=(0.0, 0.0, 10.0),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            rows=dimension if axis == "rows" else 3,
            columns=dimension if axis == "columns" else 7,
            spacing_column_mm=0.25,
            spacing_row_mm=1.5,
        )
