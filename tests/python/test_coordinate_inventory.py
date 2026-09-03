"""Independent asymmetric coordinate examples across host geometry adapters."""

from __future__ import annotations

import pytest

from dpt.geometry import AstraVectorGeometry, BatchedGeometry, DetectorGeometry, RigidTransform
from dpt.projective import ProjectionMatrix
from dpt.tigre_geometry import TigreConeGeometry
from dpt.transport import MaterialGrid, OrientedDetector, PlanarDetector
from dpt.volumes import GridSpec


def test_oriented_grid_centres_faces_and_storage_axes_have_independent_values() -> None:
    # A quarter-turn sends local x to object y and local y to negative object x.
    grid = GridSpec(
        (2, 3, 4),
        (2.0, 3.0, 5.0),
        (11.0, -7.0, 5.0),
        (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    assert grid.shape_zyx == (2, 3, 4)
    assert grid.spacing_xyz_mm == (2.0, 3.0, 5.0)
    assert grid.grid_to_object((3.0, 2.0, 1.0)) == (5.0, -1.0, 10.0)
    assert grid.object_to_grid((5.0, -1.0, 10.0)) == (3.0, 2.0, 1.0)
    cells, pose = MaterialGrid.from_oriented_grid_spec(grid, materials=2)
    assert cells.origin_mm == (-1.0, -1.5, -2.5)
    assert cells.upper_mm == (7.0, 7.5, 7.5)
    assert pose.point((6.0, 6.0, 5.0)) == (5.0, -1.0, 10.0)
    assert pose.point(cells.origin_mm) == (12.5, -8.0, 2.5)
    assert cells.to_grid_spec().origin_mm == (0.0, 0.0, 0.0)


def test_axis_aligned_grid_face_conversion_preserves_support_not_interpolation() -> None:
    grid = GridSpec((2, 3, 4), (2.0, 3.0, 5.0), (11.0, -7.0, 5.0))
    cells = MaterialGrid.from_grid_spec(grid, materials=1)
    assert cells.origin_mm == (10.0, -8.5, 2.5)
    assert cells.upper_mm == (18.0, 0.5, 12.5)
    assert cells.shape_xyz == (4, 3, 2)
    assert cells.to_grid_spec() == grid


def panel() -> DetectorGeometry:
    return DetectorGeometry(
        (0.0, 0.0, -10.0), (-6.0, -8.0, 20.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 3.0), (3, 5)
    )


def test_planar_transport_width_height_and_lower_faces_match_pixel_centres() -> None:
    geometry = panel()
    detector = PlanarDetector.from_detector_geometry(geometry)
    assert detector.shape == (5, 3)
    assert detector.image_shape == (3, 5)
    assert detector.lower_xy_mm == (-7.0, -9.5)
    assert geometry.pixel_centre(2, 4) == (2.0, -2.0, 20.0)
    assert detector.to_detector_geometry(source_mm=geometry.source_mm) == geometry


def test_tilted_transport_footprints_preserve_column_row_handedness() -> None:
    geometry = DetectorGeometry(
        (0.0, 0.0, 0.0), (30.0, -8.0, -6.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (2.0, 3.0), (3, 5)
    )
    detector = OrientedDetector(geometry)
    assert geometry.pixel_centre(2, 4) == (30.0, -2.0, -14.0)
    assert detector.lower_corner_mm == (30.0, -9.5, -5.0)
    assert detector.normal == (1.0, -0.0, 0.0)
    assert detector.source_distance((0.0, 0.0, 0.0)) == -30.0
    assert detector.shape == (5, 3)


def test_astra_centred_pixels_and_projective_column_row_order_use_scalar_values() -> None:
    geometry = panel()
    batch = BatchedGeometry((geometry, geometry))
    assert batch.shape == (2, 3, 5)
    assert batch.view_offsets == (0, 15, 30)
    assert batch.to_astra_vectors().vectors[0] == (
        0.0,
        0.0,
        -10.0,
        -2.0,
        -5.0,
        20.0,
        2.0,
        0.0,
        0.0,
        0.0,
        3.0,
        0.0,
    )
    # Input centimetres express the same fixed source, panel centre and pixel vectors.
    external = AstraVectorGeometry(
        "cone_vec", (3, 5), ((0.0, 0.0, -1.0, -0.2, -0.5, 2.0, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0),)
    )
    imported = BatchedGeometry.from_astra_vectors(external, length_scale_mm=10.0)
    assert imported.segment(0, 2, 4) == ((0.0, 0.0, -10.0), (2.0, -2.0, 20.0))
    matrix = ProjectionMatrix.from_detector(geometry)
    # Halfway from source to the last pixel lies at (1,-1,5), hence (row,column)=(2,4).
    assert matrix.project((1.0, -1.0, 5.0)) == pytest.approx((2.0, 4.0))


def test_tigre_native_zyx_offsets_and_row_column_spacing_have_independent_values() -> None:
    native = TigreConeGeometry(
        (3, 5),
        (3.0, 2.0),
        (50.0,),
        (90.0,),
        ((0.0, 0.0, 0.0),),
        ((7.0, 11.0),),
        origin_offset_zyx_mm=((2.0, 3.0, 5.0),),
        centre_rotation_mm=(4.0,),
    )
    geometry = native.to_dpt().views[0]
    assert isinstance(geometry, DetectorGeometry)
    # Native source starts at (+50,0,0), shifted by (-5,+1,-2).
    assert geometry.source_mm == (45.0, 1.0, -2.0)
    assert geometry.spacing_mm == (2.0, 3.0)
    assert geometry.origin_mm == (-45.0, 8.0, 2.0)
    assert geometry.pixel_centre(2, 4) == (-45.0, 16.0, 8.0)
    world = RigidTransform(translation_mm=(100.0, -10.0, 3.0))
    mapped = native.to_dpt(world_from_tigre=world).views[0]
    assert isinstance(mapped, DetectorGeometry)
    assert mapped.source_mm == (145.0, -9.0, 1.0)
    assert mapped.pixel_centre(2, 4) == (55.0, 6.0, 11.0)
