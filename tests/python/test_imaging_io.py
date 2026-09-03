"""Synthetic file fixtures check world coordinates, scaling and source integrity."""

import dataclasses
import hashlib
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.imaging_io import (
    ImagingMetadata,
    ImagingVolume,
    read_dicom_series,
    read_nifti,
    sphere_phantom,
)
from dpt.volumes import GridSpec

nib: Any = importlib.import_module("nibabel")
pydicom: Any = importlib.import_module("pydicom")


def _nifti(
    path: Path,
    values: Any,
    affine: Any,
    *,
    units: str = "mm",
    slope: float = 1.0,
    intercept: float = 0.0,
) -> Any:
    image = nib.Nifti1Image(np.asarray(values, dtype=np.int16), affine)
    image.header.set_xyzt_units(units)
    image.header.set_slope_inter(slope, intercept)
    nib.save(image, path)
    return image


def test_nifti_handedness_reindex_preserves_world_voxels_and_scaling(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.nii.gz"
    stored = np.arange(24, dtype=np.int16).reshape(3, 4, 2)
    affine = np.diag([-2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = (50.0, 60.0, 70.0)
    _nifti(path, stored, affine, slope=2.0, intercept=-1000.0)
    volume = read_nifti(path, value_units="HU")
    assert volume.shape == (2, 4, 3)
    assert volume.metadata.coordinate_frame == "RAS"
    assert volume.metadata.reindexed_axes == (0,)
    assert volume.metadata.rescale == ((2.0, -1000.0),)
    assert volume.metadata.source_sha256 == (hashlib.sha256(path.read_bytes()).hexdigest(),)
    assert volume.values.dtype == np.float64
    assert volume.values.flags.c_contiguous and volume.values.flags.owndata
    assert not volume.values.flags.writeable
    for x, y, z in ((0, 0, 0), (2, 3, 1), (1, 2, 1)):
        assert volume.values[z, y, x] == stored[2 - x, y, z] * 2.0 - 1000.0
        expected = affine @ np.array((2 - x, y, z, 1.0))
        assert volume.grid.grid_to_object((float(x), float(y), float(z))) == pytest.approx(
            expected[:3]
        )


@pytest.mark.parametrize("unit,factor", [("mm", 1.0), ("meter", 1000.0), ("micron", 0.001)])
def test_nifti_declared_spatial_units_convert_voxel_centres_to_mm(
    unit: str, factor: float, tmp_path: Path
) -> None:
    affine = np.array(
        [[0.0, -0.3, 0.0, 1.1], [0.2, 0.0, 0.0, -2.2], [0.0, 0.0, 0.4, 3.3], [0.0, 0.0, 0.0, 1.0]]
    )
    path = tmp_path / "synthetic.nii"
    _nifti(path, np.arange(24).reshape(2, 3, 4), affine, units=unit)
    volume = read_nifti(path)
    encoded_affine = nib.load(path).affine
    for x, y, z in ((0, 0, 0), (1, 2, 3)):
        expected = (encoded_affine @ np.array((x, y, z, 1)))[:3] * factor
        assert volume.grid.grid_to_object((float(x), float(y), float(z))) == pytest.approx(
            expected, rel=1e-14, abs=1e-11
        )
    assert volume.grid.spacing_mm == pytest.approx(
        (0.2 * factor, 0.3 * factor, 0.4 * factor), rel=1e-7
    )
    assert volume.metadata.value_units == "unknown"


def test_nifti_sform_precedence_and_qform_fallback(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.nii"
    image = nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.int16), np.eye(4))
    image.header.set_xyzt_units("mm")
    qform = np.eye(4)
    qform[:3, 3] = (10.0, 20.0, 30.0)
    sform = np.eye(4)
    sform[:3, 3] = (50.0, 60.0, 70.0)
    image.set_qform(qform, code=1)
    image.set_sform(sform, code=2)
    nib.save(image, path)
    volume = read_nifti(path)
    assert volume.grid.origin_mm == (50.0, 60.0, 70.0)
    assert volume.metadata.affine_source == "sform"
    image.set_sform(None, code=0)
    nib.save(image, path)
    volume = read_nifti(path)
    assert volume.grid.origin_mm == (10.0, 20.0, 30.0)
    assert volume.metadata.affine_source == "qform"


def test_nifti_paired_file_provenance_covers_image_and_header(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.hdr"
    image = nib.Nifti1Pair(
        np.arange(24, dtype=np.int16).reshape(2, 3, 4), np.diag([2.0, 3.0, 4.0, 1.0])
    )
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    volume = read_nifti(path)
    assert set(volume.metadata.source_sha256) == {
        hashlib.sha256(path.read_bytes()).hexdigest(),
        hashlib.sha256(path.with_suffix(".img").read_bytes()).hexdigest(),
    }
    assert volume.values[3, 2, 1] == 23.0


def test_nifti2_retains_fractional_scalar_values(tmp_path: Path) -> None:
    path = tmp_path / "synthetic2.nii"
    image = nib.Nifti2Image(
        np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 10.0, np.diag([2.0, 3.0, 4.0, 1.0])
    )
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    volume = read_nifti(path, value_units="mm^-1")
    assert volume.values[3, 2, 1] == 2.3
    assert volume.grid.spacing_mm == (2.0, 3.0, 4.0)


@pytest.mark.parametrize("spacing", [1e-200, 1e200])
def test_nifti2_finite_sform_spacing_preserves_extreme_length_scales(
    spacing: float, tmp_path: Path
) -> None:
    path = tmp_path / "synthetic-spacing.nii"
    image = nib.Nifti2Image(np.ones((2, 2, 2), dtype=np.float64), np.eye(4))
    image.header.set_xyzt_units("mm")
    image.set_sform(np.diag([spacing, spacing, spacing, 1.0]), code=2)
    nib.save(image, path)
    volume = read_nifti(path)
    assert tuple(value / spacing for value in volume.grid.spacing_mm) == (1.0, 1.0, 1.0)
    assert volume.grid.orientation == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def test_nifti_unknown_units_need_explicit_input_and_conflicts_fail(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.nii"
    _nifti(path, np.ones((2, 3, 4)), np.diag([2.0, 3.0, 4.0, 1.0]), units="unknown")
    with pytest.raises(ContractError, match="provide spatial_unit"):
        read_nifti(path)
    assert read_nifti(path, spatial_unit="mm").grid.spacing_mm == (2.0, 3.0, 4.0)
    _nifti(path, np.ones((2, 3, 4)), np.eye(4), units="mm")
    with pytest.raises(ContractError, match="conflicts"):
        read_nifti(path, spatial_unit="meter")


def test_nifti_shear_missing_affine_and_extra_dimensions_fail(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.nii"
    affine = np.eye(4)
    affine[0, 1] = 0.2
    _nifti(path, np.ones((2, 3, 4)), affine)
    with pytest.raises(ContractError, match="sheared"):
        read_nifti(path)
    image = nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.int16), np.eye(4))
    image.header.set_xyzt_units("mm")
    image.set_sform(None, code=0)
    image.set_qform(None, code=0)
    nib.save(image, path)
    with pytest.raises(ContractError, match="coded"):
        read_nifti(path)
    _nifti(path, np.ones((2, 3, 4, 2)), np.eye(4))
    with pytest.raises(ContractError, match="three-dimensional"):
        read_nifti(path)


def _dicom_slice(
    path: Path,
    *,
    slice_index: int,
    position_mm: tuple[float, float, float],
    series_uid: str,
    frame_uid: str,
) -> Any:
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dataset = pydicom.dataset.FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.SeriesInstanceUID = series_uid
    dataset.FrameOfReferenceUID = frame_uid
    dataset.PatientName = "Synthetic^Fixture"
    dataset.PatientID = "DPT_SYNTHETIC_ONLY"
    dataset.Modality = "CT"
    dataset.Rows, dataset.Columns = 2, 3
    dataset.ImageOrientationPatient = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    dataset.ImagePositionPatient = list(position_mm)
    dataset.PixelSpacing = [0.7, 2.5]
    dataset.SliceThickness = 20.0
    dataset.InstanceNumber = 20 - slice_index
    dataset.RescaleSlope = float(2 + slice_index)
    dataset.RescaleIntercept = float(-1000 + 10 * slice_index)
    dataset.RescaleType = "HU"
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    pixels = np.arange(6, dtype=np.int16).reshape(2, 3) + 100 * slice_index
    dataset.PixelData = pixels.astype("<i2").tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return dataset


def _series(
    tmp_path: Path,
    positions: tuple[tuple[float, float, float], ...] = (
        (10.0, 20.0, 30.0),
        (13.0, 20.0, 30.0),
        (16.0, 20.0, 30.0),
    ),
) -> tuple[Path, ...]:
    series, frame = pydicom.uid.generate_uid(), pydicom.uid.generate_uid()
    paths = []
    for index, position in enumerate(positions):
        path = tmp_path / f"synthetic-{index}.dcm"
        _dicom_slice(
            path, slice_index=index, position_mm=position, series_uid=series, frame_uid=frame
        )
        paths.append(path)
    return tuple(paths)


def test_dicom_oblique_stack_orders_positions_rescales_and_retains_lps(tmp_path: Path) -> None:
    paths = _series(tmp_path)
    volume = read_dicom_series((paths[2], paths[0], paths[1]))
    assert volume.shape == (3, 2, 3)
    assert volume.grid.spacing_mm == pytest.approx((2.5, 0.7, 3.0))
    assert volume.metadata.coordinate_frame == "LPS"
    assert volume.metadata.value_units == "HU"
    assert volume.metadata.rescale == ((2.0, -1000.0), (3.0, -990.0), (4.0, -980.0))
    assert volume.metadata.source_sha256 == tuple(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    )
    for x, y, z in ((0, 0, 0), (2, 1, 2), (1, 1, 1)):
        assert volume.values[z, y, x] == (100 * z + 3 * y + x) * (2 + z) - 1000 + 10 * z
        assert volume.grid.grid_to_object((float(x), float(y), float(z))) == pytest.approx(
            (10.0 + 3.0 * z, 20.0 + 2.5 * x, 30.0 + 0.7 * y)
        )
    assert "Synthetic" not in repr(volume.metadata)
    assert "Patient" not in repr(dataclasses.asdict(volume.metadata))


def test_dicom_single_slice_uses_pitch_and_never_slice_thickness(tmp_path: Path) -> None:
    paths = _series(tmp_path)
    with pytest.raises(ContractError, match="single_slice_spacing"):
        read_dicom_series(paths[:1])
    volume = read_dicom_series(paths[:1], single_slice_spacing_mm=3.0)
    assert volume.grid.spacing_mm[2] == 3.0
    dataset = pydicom.dcmread(paths[0])
    dataset.SpacingBetweenSlices = 4.0
    dataset.save_as(paths[0], enforce_file_format=True)
    assert read_dicom_series(paths[:1]).grid.spacing_mm[2] == 4.0


def test_dicom_rounding_bound_covers_accepted_slice_position_error(tmp_path: Path) -> None:
    positions = ((10.0, 20.0, 30.0), (13.0 + 2e-5, 20.0, 30.0), (16.0, 20.0, 30.0))
    volume = read_dicom_series(_series(tmp_path, positions))
    for index, position in enumerate(positions):
        represented = np.array(volume.grid.grid_to_object((0.0, 0.0, float(index))))
        error = float(np.linalg.norm(represented - position))
        assert error <= volume.metadata.affine_error_mm + 1e-13
    assert volume.metadata.affine_error_mm == pytest.approx(2e-5, abs=1e-12)


@pytest.mark.parametrize(
    "positions,match",
    [
        (((10.0, 20.0, 30.0), (13.0, 20.0, 30.0), (17.0, 20.0, 30.0)), "uniform"),
        (((10.0, 20.0, 30.0), (13.0, 20.0, 30.0), (13.0, 20.0, 30.0)), "uniform"),
        (((10.0, 20.0, 30.0), (13.0, 20.2, 30.0), (16.0, 20.4, 30.0)), "sheared"),
    ],
)
def test_dicom_irregular_or_sheared_stack_requires_resampling(
    positions: Any, match: str, tmp_path: Path
) -> None:
    with pytest.raises(ContractError, match=match):
        read_dicom_series(_series(tmp_path, positions))


@pytest.mark.parametrize(
    "tag,value,match",
    [
        ("FrameOfReferenceUID", "1.2.3.4", "frame of reference"),
        ("NumberOfFrames", 2, "multiframe"),
        ("SamplesPerPixel", 3, "scalar"),
        ("AnatomicalOrientationType", "QUADRUPED", "BIPED"),
        ("ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], "share orientation"),
        ("PixelSpacing", [0.8, 2.5], "share pixel spacing"),
    ],
)
def test_dicom_incompatible_records_fail_before_pixel_decoding(
    tag: str, value: Any, match: str, tmp_path: Path
) -> None:
    paths = _series(tmp_path)
    dataset = pydicom.dcmread(paths[1])
    setattr(dataset, tag, value)
    dataset.save_as(paths[1], enforce_file_format=True)
    with pytest.raises(ContractError, match=match):
        read_dicom_series(paths)


def test_imaging_volume_owns_storage_and_preserves_signed_values() -> None:
    original = np.arange(24, dtype=np.float64).reshape(2, 3, 4) - 1000.0
    volume = ImagingVolume(
        original,
        GridSpec((2, 3, 4), (1.0, 1.0, 1.0)),
        ImagingMetadata("synthetic", "synthetic", "HU"),
    )
    original.fill(0.0)
    assert volume.values[0, 0, 0] == -1000.0
    assert volume.metadata.value_units == "HU"
    with pytest.raises(ValueError, match="read-only"):
        volume.values[0, 0, 0] = 1.0
    with pytest.raises(ContractError, match="finite"):
        ImagingVolume(np.full((2, 3, 4), np.nan), volume.grid, volume.metadata)


def test_complex_scalar_values_are_rejected_before_conversion(tmp_path: Path) -> None:
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    metadata = ImagingMetadata("synthetic", "synthetic", "HU")
    with pytest.raises(ContractError, match="real"):
        ImagingVolume(np.array([[[1.0 + 2.0j]]]), grid, metadata)
    path = tmp_path / "synthetic-complex.nii"
    image = nib.Nifti1Image(np.ones((1, 1, 1), dtype=np.complex128) * (1.0 + 2.0j), np.eye(4))
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    with pytest.raises(ContractError, match="real"):
        read_nifti(path)


@pytest.mark.parametrize(
    "spacing,radius,expected",
    [(1e-200, 1e-210, [1.0, 0.0, 0.0]), (1e200, 1e200, [1.0, 1.0, 0.0])],
)
def test_sphere_phantom_retains_boundaries_at_extreme_length_scales(
    spacing: float, radius: float, expected: list[float]
) -> None:
    grid = GridSpec((1, 1, 3), (spacing, 1.0, 1.0))
    volume = sphere_phantom(grid, radius_mm=radius)
    assert volume.values.ravel().tolist() == expected


def test_sphere_phantom_matches_world_distance_on_rotated_anisotropic_grid() -> None:
    grid = GridSpec(
        (3, 4, 5),
        (2.0, 1.0, 3.0),
        (10.0, 20.0, 30.0),
        (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    centre = (8.0, 24.0, 33.0)
    volume = sphere_phantom(
        grid, radius_mm=3.0, centre_mm=centre, inside=0.02, outside=0.001, value_units="mm^-1"
    )
    for z in range(3):
        for y in range(4):
            for x in range(5):
                world = np.array((10.0 - y, 20.0 + 2 * x, 30.0 + 3 * z))
                expected = 0.02 if np.sum((world - centre) ** 2) <= 9.0 else 0.001
                assert volume.values[z, y, x] == expected
    assert volume.metadata.source_format == "synthetic"
    assert volume.metadata.source_sha256 == ()
