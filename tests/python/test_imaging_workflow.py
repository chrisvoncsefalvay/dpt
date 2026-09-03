"""Real file decoding and physical geometry for the high-level CT route."""

import itertools
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest

from dpt.contracts import ContractError
from dpt.imaging import ct_to_attenuation, read_volume, standard_views
from dpt.imaging_io import ImagingMetadata, ImagingVolume
from dpt.volumes import GridSpec


def _nifti(path, values, affine, *, slope=None, intercept=None):
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units("mm")
    if slope is not None:
        image.header.set_slope_inter(slope, intercept)
    nib.save(image, path)


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_nifti_preserves_unscaled_dtype_and_owns_immutable_values(tmp_path, dtype):
    source = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    path = tmp_path / "ct.nii.gz"
    _nifti(path, source, np.diag([1.2, 2.3, 3.4, 1.0]))
    volume = read_volume(path, units="HU")
    assert volume.values.dtype == dtype
    assert volume.values.flags.owndata and volume.values.flags.c_contiguous
    assert not volume.values.flags.writeable
    np.testing.assert_array_equal(volume.values, source.transpose(2, 1, 0))
    assert volume.metadata.value_units == "HU"


def test_nifti_scaling_once_preserves_oblique_world_coordinates(tmp_path):
    angle = 0.31
    rotation = np.array(
        [
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation * (1.25, 2.0, 3.5)
    affine[:3, 3] = (19.0, -37.0, 13.0)
    source = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    path = tmp_path / "oblique.nii.gz"
    _nifti(path, source, affine, slope=0.5, intercept=-1024.0)
    volume = read_volume(path, units="HU")
    np.testing.assert_array_equal(volume.values, (source * 0.5 - 1024).transpose(2, 1, 0))
    encoded = nib.load(path).affine
    for index in itertools.product((0, 1), (0, 2), (0, 3)):
        np.testing.assert_allclose(
            volume.grid.grid_to_object(index),
            (encoded @ (*index, 1))[:3],
            rtol=0,
            atol=max(volume.metadata.affine_error_mm, 1e-12),
        )


def _dicom_series(directory: Path, uid: str, *, offset: int = 0):
    directory.mkdir(parents=True, exist_ok=True)
    frame_uid = pydicom.uid.generate_uid()
    angle = 0.4
    u = np.array((math.cos(angle), math.sin(angle), 0.0))
    v = np.array((-math.sin(angle), math.cos(angle), 0.0))
    first = np.array((23.0, -17.0, 9.0))
    planes = []
    for index in range(2):
        path = directory / f"slice-{1 - index}.dcm"
        meta = pydicom.dataset.FileMetaDataset()
        meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        image = pydicom.dataset.FileDataset(str(path), {}, file_meta=meta, preamble=bytes(128))
        image.SOPClassUID = meta.MediaStorageSOPClassUID
        image.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        image.SeriesInstanceUID = uid
        image.FrameOfReferenceUID = frame_uid
        image.SeriesDescription = f"test series {offset}"
        image.Modality = "CT"
        image.ImageOrientationPatient = [*u, *v]
        image.ImagePositionPatient = (first + np.array((0.0, 0.0, 3.2 * index))).tolist()
        image.PixelSpacing = [2.0, 1.5]
        image.Rows, image.Columns = 2, 3
        image.SamplesPerPixel, image.PhotometricInterpretation = 1, "MONOCHROME2"
        image.BitsAllocated, image.BitsStored, image.HighBit = 16, 16, 15
        image.PixelRepresentation = 1
        image.RescaleSlope, image.RescaleIntercept, image.RescaleType = 2.0, -1000.0, "HU"
        values = np.arange(6, dtype=np.int16).reshape(2, 3) + 6 * index + offset
        image.PixelData = values.tobytes()
        image.save_as(path, enforce_file_format=True)
        planes.append(values * 2.0 - 1000.0)
    return np.stack(planes), first, u, v


def test_single_dicom_series_rescales_once_and_converts_full_lps_affine_to_ras(tmp_path):
    values, origin, u, v = _dicom_series(tmp_path / "nested", pydicom.uid.generate_uid())
    (tmp_path / "readme.txt").write_text("not an image", encoding="utf-8")
    volume = read_volume(tmp_path)
    np.testing.assert_array_equal(volume.values, values)
    assert volume.metadata.value_units == "HU"
    assert volume.metadata.coordinate_frame == "RAS"
    assert volume.grid.spacing_mm == pytest.approx((1.5, 2.0, 3.2))
    for x, y, z in itertools.product((0, 2), (0, 1), (0, 1)):
        expected_lps = origin + u * x * 1.5 + v * y * 2.0 + (0.0, 0.0, z * 3.2)
        np.testing.assert_allclose(
            volume.grid.grid_to_object((x, y, z)), expected_lps * (-1.0, -1.0, 1.0), atol=1e-12
        )


def test_multiple_dicom_series_report_actual_uids_and_require_selection(tmp_path):
    uid_a, uid_b = pydicom.uid.generate_uid(), pydicom.uid.generate_uid()
    _dicom_series(tmp_path / "a", uid_a)
    expected, *_ = _dicom_series(tmp_path / "b", uid_b, offset=100)
    with pytest.raises(ContractError, match="multiple image series") as error:
        read_volume(tmp_path)
    assert uid_a in str(error.value) and uid_b in str(error.value)
    np.testing.assert_array_equal(read_volume(tmp_path, series_uid=uid_b).values, expected)
    with pytest.raises(ContractError, match="available series") as error:
        read_volume(tmp_path, series_uid=pydicom.uid.generate_uid())
    assert uid_a in str(error.value) and uid_b in str(error.value)


def test_dicom_known_value_units_cannot_be_relabelled(tmp_path):
    _dicom_series(tmp_path, pydicom.uid.generate_uid())
    with pytest.raises(ContractError, match="conflicts"):
        read_volume(tmp_path, units="mm^-1")
    path = tmp_path / "slice-1.dcm"
    image = pydicom.dcmread(path)
    image.RescaleType = "BQML"
    image.save_as(path, enforce_file_format=True)
    with pytest.raises(ContractError, match="conflicts"):
        read_volume(tmp_path, units="HU")


def test_declared_units_can_supply_an_unknown_dicom_slice_label(tmp_path):
    expected, *_ = _dicom_series(tmp_path, pydicom.uid.generate_uid())
    path = tmp_path / "slice-0.dcm"
    image = pydicom.dcmread(path)
    del image.RescaleType
    image.save_as(path, enforce_file_format=True)
    np.testing.assert_array_equal(read_volume(tmp_path, units="HU").values, expected)


def test_attenuation_input_retains_values_and_declared_units(tmp_path):
    values = np.array([[[0.01, 0.02, 0.03]]], dtype=np.float32)
    path = tmp_path / "attenuation.nii.gz"
    _nifti(path, values, np.eye(4))
    volume = read_volume(path, units="mm^-1")
    np.testing.assert_array_equal(volume.values, values.transpose(2, 1, 0))
    assert volume.metadata.value_units == "mm^-1"
    with pytest.raises(ContractError, match="HU"):
        ct_to_attenuation(volume, energy_kev=60.0)


def test_ct_water_conversion_preserves_hu_and_grid():
    values = np.array([[[-1200, -1000, -500, 0, 1000]]], dtype=np.int16)
    grid = GridSpec(values.shape, (0.7, 0.8, 2.5), (37.0, -25.0, 12.0))
    volume = ImagingVolume(values, grid, ImagingMetadata("nifti", "RAS", "HU"))
    attenuation = ct_to_attenuation(volume, energy_kev=60.0)
    np.testing.assert_allclose(
        attenuation.values, [[[0.0, 0.0, 0.010295, 0.02059, 0.04118]]], rtol=1e-7, atol=0
    )
    np.testing.assert_array_equal(volume.values, values)
    assert attenuation.grid is grid
    assert attenuation.metadata.value_units == "mm^-1"
    assert attenuation.values.dtype == np.float32
    assert not attenuation.values.flags.writeable
    assert not np.shares_memory(attenuation.values, volume.values)
    for energy in (0.0, 59.0, 80.0, math.nan):
        with pytest.raises(ContractError, match="energy_kev"):
            ct_to_attenuation(volume, energy_kev=energy)


@pytest.mark.parametrize("shape", [(512, 512), (511, 769)])
def test_standard_views_use_physical_centre_and_cover_all_oblique_support_corners(shape):
    angle = 0.53
    grid = GridSpec(
        (76, 507, 512),
        (0.7, 0.8, 2.5),
        (91.0, -137.0, 44.0),
        (
            math.cos(angle),
            0.0,
            math.sin(angle),
            0.0,
            1.0,
            0.0,
            -math.sin(angle),
            0.0,
            math.cos(angle),
        ),
    )
    centre = np.array(grid.grid_to_object((255.5, 253.0, 37.5)))
    geometry = standard_views(grid, shape=shape)
    diagonal = math.hypot(512 * 0.7, 507 * 0.8, 76 * 2.5)
    sod, sdd = max(1000, 2 * diagonal), max(1500, 3 * diagonal)
    width = 1.05 * diagonal * sdd / (sod - diagonal / 2)
    expected_directions = ((0, 1, 0), (1, 0, 0), (2**-0.5, 2**-0.5, 0))
    assert geometry.shape == (3, *shape)
    corners = itertools.product(*zip(*grid.support, strict=True))
    world_corners = [grid.grid_to_object(corner) for corner in corners]
    for panel, direction in zip(geometry.views, expected_directions, strict=True):
        np.testing.assert_allclose(
            np.array(panel.source_mm) - centre, sod * np.array(direction), atol=1e-12
        )
        detector_centre = np.array(panel.origin_mm) + (
            np.array(panel.u) * panel.spacing_mm[0] * (shape[1] - 1) / 2
            + np.array(panel.v) * panel.spacing_mm[1] * (shape[0] - 1) / 2
        )
        assert np.linalg.norm(detector_centre - panel.source_mm) == pytest.approx(sdd)
        assert panel.spacing_mm == pytest.approx((width / shape[1], width / shape[0]))
        np.testing.assert_allclose(
            panel.project_point(tuple(centre)), ((shape[0] - 1) / 2, (shape[1] - 1) / 2)
        )
        for corner in world_corners:
            row, column = panel.project_point(corner)
            assert -0.5 < row < shape[0] - 0.5
            assert -0.5 < column < shape[1] - 0.5


def test_invalid_volume_sources_and_view_names_are_explicit(tmp_path):
    with pytest.raises(ContractError, match="no DICOM image series"):
        read_volume(tmp_path)
    unsupported = tmp_path / "ct.txt"
    unsupported.write_text("not an image", encoding="utf-8")
    with pytest.raises(ContractError, match="NIfTI"):
        read_volume(unsupported)
    grid = GridSpec((2, 2, 2), (1.0, 1.0, 1.0))
    for names in ((), "ap", ("unknown",)):
        with pytest.raises(ContractError, match="names"):
            standard_views(grid, names)
