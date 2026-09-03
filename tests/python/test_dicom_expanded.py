"""Independent nonclinical geometry, transfer-syntax and mapping references."""

import hashlib
import importlib
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
pydicom = pytest.importorskip("pydicom")

from dicom_fixtures import (  # noqa: E402 - optional modules must be gated before fixture imports
    item,
    pixels,
    record,
    remove_rescale,
    rle_encode,
    rw_linear,
    save,
)
from dpt.contracts import ContractError  # noqa: E402
from dpt.imaging_io import ImagingMetadata, read_dicom_series  # noqa: E402


def test_enhanced_reorders_frames_and_applies_each_rescale_once(tmp_path: Path) -> None:
    path = tmp_path / "enhanced.dcm"
    save(record(path), path)
    volume = read_dicom_series((path,))
    assert volume.shape == (3, 2, 3)
    assert volume.grid.spacing_mm == pytest.approx((2.0, 1.25, 4.0))
    for z in range(3):
        for y in range(2):
            for x in range(3):
                assert volume.values[z, y, x] == pixels()[z, y, x] * (z + 2) - 100 + 3 * z
                assert volume.grid.grid_to_object((float(x), float(y), float(z))) == pytest.approx(
                    (31 + 1.2 * x - y, -9 + 1.6 * x + 0.75 * y, 15 + 4 * z), abs=1e-12
                )
    assert volume.metadata.rescale == ((2.0, -100.0), (3.0, -97.0), (4.0, -94.0))
    assert volume.metadata.value_mappings == ("rescale",) * 3
    assert volume.metadata.source_sha256 == (hashlib.sha256(path.read_bytes()).hexdigest(),)
    assert volume.metadata.value_units == "HU"
    assert volume.values.dtype == np.float64
    assert not volume.values.flags.writeable


def test_per_frame_geometry_and_shared_transform(tmp_path: Path) -> None:
    path = tmp_path / "per-frame.dcm"
    ds = record(path)
    shared = ds.SharedFunctionalGroupsSequence[0]
    shared.PixelValueTransformationSequence = [
        item(RescaleSlope=0.5, RescaleIntercept=-7.0, RescaleType="HU")
    ]
    for frame in ds.PerFrameFunctionalGroupsSequence:
        frame.PlaneOrientationSequence = shared.PlaneOrientationSequence
        frame.PixelMeasuresSequence = shared.PixelMeasuresSequence
        del frame.PixelValueTransformationSequence
    del shared.PlaneOrientationSequence
    del shared.PixelMeasuresSequence
    save(ds, path)
    volume = read_dicom_series((path,))
    np.testing.assert_array_equal(volume.values, pixels() * 0.5 - 7.0)


def test_enhanced_single_frame_uses_shared_pixel_measures_pitch(tmp_path: Path) -> None:
    path = tmp_path / "single.dcm"
    ds = record(path, count=1)
    ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SpacingBetweenSlices = 6.0
    save(ds, path)
    volume = read_dicom_series((path,))
    assert volume.grid.origin_mm == (31.0, -9.0, 23.0)
    assert volume.grid.spacing_mm[2] == 6.0


def test_enhanced_negative_normal_preserves_world_coordinates_and_scaling(tmp_path: Path) -> None:
    path = tmp_path / "negative-normal.dcm"
    ds = record(path)
    ds.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient = [
        0.6,
        0.8,
        0,
        0.8,
        -0.6,
        0,
    ]
    save(ds, path)
    volume = read_dicom_series((path,))
    for z in range(3):
        native = 2 - z
        np.testing.assert_array_equal(
            volume.values[z], pixels()[native] * (native + 2) - 100 + 3 * native
        )
        assert volume.grid.grid_to_object((2.0, 1.0, float(z))) == pytest.approx(
            (31 + 2.4 + 1, -9 + 3.2 - 0.75, 23 - 4 * z), abs=1e-12
        )


def test_enhanced_geometry_rounding_bound_covers_native_frame_position(tmp_path: Path) -> None:
    path = tmp_path / "rounding.dcm"
    ds = record(path)
    ds.PerFrameFunctionalGroupsSequence[2].PlanePositionSequence[0].ImagePositionPatient[2] += 2e-5
    save(ds, path)
    volume = read_dicom_series((path,))
    assert volume.metadata.affine_error_mm == pytest.approx(2e-5, abs=1e-12)
    represented = volume.grid.grid_to_object((2.0, 1.0, 1.0))
    assert np.linalg.norm(np.asarray(represented) - (32.4, -5.05, 19.0 + 2e-5)) <= (
        volume.metadata.affine_error_mm + 1e-12
    )


def test_enhanced_identity_accepts_explicit_units_when_rescale_type_absent(tmp_path: Path) -> None:
    path = tmp_path / "identity-units.dcm"
    ds = record(path)
    remove_rescale(ds)
    save(ds, path)
    volume = read_dicom_series((path,), value_units="HU")
    np.testing.assert_array_equal(volume.values, pixels())
    assert volume.metadata.value_units == "HU"
    assert volume.metadata.value_mappings == ("identity",) * 3


@pytest.mark.parametrize("scope_mutation", ["organization", "functional_group", "private_creator"])
def test_enhanced_dimension_indices_are_scoped_and_cannot_merge_independent_stacks(
    tmp_path: Path, scope_mutation: str
) -> None:
    paths = (tmp_path / "echo-a.dcm", tmp_path / "echo-b.dcm")
    first, second = (record(path) for path in paths)
    second.SeriesInstanceUID = first.SeriesInstanceUID
    second.FrameOfReferenceUID = first.FrameOfReferenceUID
    for ds, echo in ((first, 10.0), (second, 20.0)):
        ds.DimensionIndexSequence[1].DimensionIndexPointer = 0x00189082
        ds.DimensionIndexSequence[1].FunctionalGroupPointer = 0x00189114
        ds.SharedFunctionalGroupsSequence[0].MREchoSequence = [item(EffectiveEchoTime=echo)]
    for frame in second.PerFrameFunctionalGroupsSequence:
        frame.PlanePositionSequence[0].ImagePositionPatient[2] += 12
    if scope_mutation == "organization":
        second.DimensionOrganizationSequence[0].DimensionOrganizationUID = "1.2.826.0.1.2"
        for dimension in second.DimensionIndexSequence:
            dimension.DimensionOrganizationUID = "1.2.826.0.1.2"
    elif scope_mutation == "functional_group":
        second.DimensionIndexSequence[1].FunctionalGroupPointer = 0x00189117
    elif scope_mutation == "private_creator":
        second.DimensionIndexSequence[1].DimensionIndexPrivateCreator = "AUTHORED_SCOPE"
    for ds, path in zip((first, second), paths, strict=True):
        save(ds, path)
    with pytest.raises(ContractError, match="one scalar spatial volume"):
        read_dicom_series(paths)


@pytest.mark.parametrize("enhanced", [False, True])
def test_compressed_rle_has_independent_packbits_pixels(tmp_path: Path, enhanced: bool) -> None:
    path = tmp_path / "rle.dcm"
    ds = record(path, enhanced=enhanced)
    rle_encode(ds)
    save(ds, path)
    volume = read_dicom_series((path,))
    expected = pixels().astype(np.float64)
    if enhanced:
        for z in range(3):
            expected[z] = expected[z] * (z + 2) - 100 + 3 * z
    else:
        expected = expected[:1]
    np.testing.assert_array_equal(volume.values, expected)


@pytest.mark.parametrize("syntax", ["JPEGBaseline8Bit", "JPEGLSLossless", "JPEG2000Lossless"])
def test_absent_compressed_decoder_has_actionable_gate(tmp_path: Path, syntax: str) -> None:
    uid = getattr(pydicom.uid, syntax)
    if importlib.import_module("pydicom.pixels").get_decoder(uid).is_available:
        pytest.skip("absent-codec gate requires missing decoder; actual RLE is covered separately")
    path = tmp_path / "missing-codec.dcm"
    ds = record(path, enhanced=False)
    ds.file_meta.TransferSyntaxUID = uid
    ds.PixelData = pydicom.encaps.encapsulate([b"authored-codec-gate"])
    ds["PixelData"].is_undefined_length = True
    save(ds, path)
    with pytest.raises(RuntimeError, match=r"pydicom decoder.*transfer syntax") as caught:
        read_dicom_series((path,))
    assert str(uid) in str(caught.value)
    assert "requires" in str(caught.value)


def test_modality_lut_signed_input_uses_endpoint_clipping(tmp_path: Path) -> None:
    path = tmp_path / "lut.dcm"
    ds = record(path, enhanced=False)
    lut = item(LUTDescriptor=[4, -1, 16], ModalityLUTType="HU")
    lut.add_new(0x00283006, "US", [10, 20, 45, 100])
    ds.ModalityLUTSequence = [lut]
    save(ds, path)
    volume = read_dicom_series((path,))
    # Stored [12,-3,0;7,2,4], first input -1: clip at both LUT endpoints.
    np.testing.assert_array_equal(volume.values, [[[100, 10, 20], [100, 100, 100]]])
    assert volume.metadata.rescale == ()
    assert volume.metadata.value_mappings == ("modality_lut",)
    assert volume.metadata.value_units == "HU"


def test_modality_lut_word_data_respects_big_endian(tmp_path: Path) -> None:
    import struct

    path = tmp_path / "big-endian-lut.dcm"
    ds = record(path, enhanced=False)
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRBigEndian
    ds.PixelData = pixels()[0].astype(">i2").tobytes()
    lut = item(LUTDescriptor=[4, -1, 16], ModalityLUTType="HU")
    lut.add_new(0x00283006, "OW", struct.pack(">4H", 10, 20, 45, 100))
    ds.ModalityLUTSequence = [lut]
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    np.testing.assert_array_equal(
        read_dicom_series((path,)).values, [[[100, 10, 20], [100, 100, 100]]]
    )


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("word_padded", [False, True])
def test_modality_lut_8bit_word_data_and_byte_data(
    tmp_path: Path, endian: str, word_padded: bool
) -> None:
    path = tmp_path / "8bit-lut.dcm"
    ds = record(path, enhanced=False)
    ds.file_meta.TransferSyntaxUID = (
        pydicom.uid.ExplicitVRLittleEndian if endian == "<" else pydicom.uid.ExplicitVRBigEndian
    )
    ds.PixelData = pixels()[0].astype(f"{endian}i2").tobytes()
    lut = item(LUTDescriptor=[5, -1, 8], ModalityLUTType="HU")
    values = [10, 20, 45, 100, 200]
    if word_padded:
        encoded = np.asarray(values, dtype=f"{endian}u2").tobytes()
    else:
        # Pack entries in low-byte-first OW words, including even-length padding.
        encoded = np.frombuffer(bytes([*values, 0]), dtype="<u2").astype(f"{endian}u2").tobytes()
    lut.add_new(0x00283006, "OW", encoded)
    ds.ModalityLUTSequence = [lut]
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    np.testing.assert_array_equal(
        read_dicom_series((path,)).values, [[[200, 10, 20], [200, 100, 200]]]
    )


def test_modality_lut_zero_count_has_65536_entries_without_signed_overflow(tmp_path: Path) -> None:
    path = tmp_path / "full-lut.dcm"
    ds = record(path, enhanced=False)
    lut = item(LUTDescriptor=[0, -32768, 16], ModalityLUTType="HU")
    lut.add_new(0x00283006, "OW", np.arange(65536, dtype="<u2").tobytes())
    ds.ModalityLUTSequence = [lut]
    save(ds, path)
    np.testing.assert_array_equal(
        read_dicom_series((path,)).values, (pixels()[0].astype(np.int64) + 32768)[None]
    )


@pytest.mark.parametrize("enhanced", [False, True])
def test_real_world_linear_maps_stored_values_with_coded_units(
    tmp_path: Path, enhanced: bool
) -> None:
    path = tmp_path / "rw.dcm"
    ds = record(path, enhanced=enhanced)
    if enhanced:
        remove_rescale(ds)
        ds.SharedFunctionalGroupsSequence[0].RealWorldValueMappingSequence = [rw_linear()]
    else:
        ds.RealWorldValueMappingSequence = [rw_linear()]
    save(ds, path)
    volume = read_dicom_series((path,))
    np.testing.assert_array_equal(volume.values, pixels()[: 3 if enhanced else 1] * 0.25 - 2)
    assert volume.metadata.value_units == "1/mm"
    assert volume.metadata.measurement_unit_code == ("UCUM", "1/mm", "")
    assert volume.metadata.value_mappings == ("real_world_linear",) * (3 if enhanced else 1)


def test_real_world_lut_is_nonlinear_and_per_frame(tmp_path: Path) -> None:
    path = tmp_path / "rw-lut.dcm"
    ds = record(path)
    remove_rescale(ds)
    for frame in ds.PerFrameFunctionalGroupsSequence:
        z = int(frame.FrameContentSequence[0].InStackPositionNumber) - 1
        mapping = rw_linear()
        del mapping.RealWorldValueSlope
        del mapping.RealWorldValueIntercept
        mapping.RealWorldValueLUTData = [float(v * v + 10 * z) for v in range(-3, 13)]
        frame.RealWorldValueMappingSequence = [mapping]
    save(ds, path)
    volume = read_dicom_series((path,))
    for z in range(3):
        np.testing.assert_array_equal(
            volume.values[z], pixels()[z].astype(np.float64) ** 2 + 10 * z
        )
    assert volume.metadata.rescale == ()
    assert volume.metadata.value_mappings == ("real_world_lut",) * 3


def test_real_world_urn_units_preserve_absent_coding_scheme(tmp_path: Path) -> None:
    path = tmp_path / "urn-units.dcm"
    ds = record(path, enhanced=False)
    mapping = rw_linear()
    uri = "urn:dpt:authored-nonclinical:unit:inverse-mm"
    mapping.MeasurementUnitsCodeSequence = [item(URNCodeValue=uri, CodeMeaning="authored unit")]
    ds.RealWorldValueMappingSequence = [mapping]
    save(ds, path)
    volume = read_dicom_series((path,))
    assert volume.metadata.measurement_unit_code == ("", uri, "")
    assert volume.metadata.value_units == uri
    np.testing.assert_array_equal(volume.values, pixels()[:1] * 0.25 - 2)


@pytest.mark.parametrize(
    "unit,match",
    [
        (dict(URNCodeValue="relative-unit", CodeMeaning="invalid URI"), "URNCodeValue"),
        (
            dict(
                URNCodeValue="urn:dpt:unit", CodingSchemeVersion="1", CodeMeaning="invalid version"
            ),
            "CodingSchemeVersion",
        ),
    ],
)
def test_real_world_urn_units_require_self_identifying_codes_and_valid_version_scope(
    tmp_path: Path, unit: dict[str, str], match: str
) -> None:
    path = tmp_path / "invalid-urn.dcm"
    ds = record(path, enhanced=False)
    mapping = rw_linear()
    mapping.MeasurementUnitsCodeSequence = [item(**unit)]
    ds.RealWorldValueMappingSequence = [mapping]
    save(ds, path)
    with pytest.raises(ContractError, match=match):
        read_dicom_series((path,))


def test_real_world_piecewise_disjoint_same_units_covers_pixels(tmp_path: Path) -> None:
    path = tmp_path / "piecewise.dcm"
    ds = record(path, enhanced=False)
    ds.RealWorldValueMappingSequence = [
        rw_linear(low=-3, high=0, slope=2.0, intercept=4.0),
        rw_linear(low=1, high=12, slope=0.5, intercept=-1.0),
    ]
    save(ds, path)
    np.testing.assert_array_equal(read_dicom_series((path,)).values, [[[5, -2, 4], [2.5, 0, 1]]])


def test_competing_mapping_paths_require_choice_and_never_compose(tmp_path: Path) -> None:
    path = tmp_path / "choice.dcm"
    ds = record(path)
    ds.SharedFunctionalGroupsSequence[0].RealWorldValueMappingSequence = [rw_linear()]
    save(ds, path)
    with pytest.raises(ContractError, match="value_mapping"):
        read_dicom_series((path,))
    rw = read_dicom_series((path,), value_mapping="real_world")
    np.testing.assert_array_equal(rw.values, pixels() * 0.25 - 2)
    modality = read_dicom_series((path,), value_mapping="modality")
    assert modality.values[0, 0, 0] == -76.0
    assert modality.metadata.value_units == "HU"


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("group_count", "number of frames"),
        ("duplicate_group", "shared and per-frame"),
        ("missing_position", "PlanePositionSequence"),
        ("missing_orientation", "PlaneOrientationSequence"),
        ("orientation", "share orientation"),
        ("spacing", "share pixel spacing"),
        ("duplicate_position", "uniform"),
        ("shear", "sheared"),
        ("temporal", "one scalar spatial volume"),
        ("temporal_offset", "one scalar spatial volume"),
        ("energy", "one scalar spatial volume"),
        ("stack", "one scalar spatial volume"),
        ("distorted", "VOLUME"),
        ("color", "scalar"),
        ("mapping_units", "units"),
    ],
)
def test_enhanced_ambiguous_or_incompatible_records_fail(
    tmp_path: Path, mutation: str, match: str
) -> None:
    path = tmp_path / "invalid.dcm"
    ds = record(path)
    shared = ds.SharedFunctionalGroupsSequence[0]
    frame = ds.PerFrameFunctionalGroupsSequence[1]
    if mutation == "group_count":
        ds.PerFrameFunctionalGroupsSequence.pop()
    elif mutation == "duplicate_group":
        frame.PixelMeasuresSequence = shared.PixelMeasuresSequence
    elif mutation == "missing_position":
        del frame.PlanePositionSequence
    elif mutation == "missing_orientation":
        del shared.PlaneOrientationSequence
    elif mutation == "orientation":
        orientation = shared.PlaneOrientationSequence
        for group in ds.PerFrameFunctionalGroupsSequence:
            group.PlaneOrientationSequence = [
                item(ImageOrientationPatient=list(orientation[0].ImageOrientationPatient))
            ]
        del shared.PlaneOrientationSequence
        frame.PlaneOrientationSequence[0].ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    elif mutation == "spacing":
        for group in ds.PerFrameFunctionalGroupsSequence:
            group.PixelMeasuresSequence = [item(PixelSpacing=[1, 1])]
        del shared.PixelMeasuresSequence
        frame.PixelMeasuresSequence[0].PixelSpacing = [1, 2]
    elif mutation == "duplicate_position":
        frame.PlanePositionSequence[0].ImagePositionPatient = [31, -9, 23]
    elif mutation == "shear":
        frame.PlanePositionSequence[0].ImagePositionPatient = [31.2, -9, 15]
    elif mutation == "temporal":
        frame.FrameContentSequence[0].TemporalPositionIndex = 2
        frame.FrameContentSequence[0].DimensionIndexValues = [1, 2]
    elif mutation == "temporal_offset":
        for group in ds.PerFrameFunctionalGroupsSequence:
            group.TemporalPositionSequence = [item(TemporalPositionTimeOffset=0.0)]
        frame.TemporalPositionSequence[0].TemporalPositionTimeOffset = 1.0
    elif mutation == "energy":
        ds.DimensionIndexSequence[1].DimensionIndexPointer = 0x00189362
        frame.FrameContentSequence[0].DimensionIndexValues = [1, 2]
    elif mutation == "stack":
        frame.FrameContentSequence[0].StackID = "two"
    elif mutation == "distorted":
        ds.VolumetricProperties = "DISTORTED"
    elif mutation == "color":
        ds.PhotometricInterpretation = "PALETTE COLOR"
    elif mutation == "mapping_units":
        frame.PixelValueTransformationSequence[0].RescaleType = "OD"
    save(ds, path)
    with pytest.raises(ContractError, match=match):
        read_dicom_series((path,))


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("overlap", "overlap"),
        ("gap", "unmapped"),
        ("unit", "units"),
        ("missing_code", "MeasurementUnitsCodeSequence"),
        ("lut_length", "LUT"),
        ("both", "linear or LUT"),
        ("range", "range"),
        ("unit_override", "conflicts"),
    ],
)
def test_invalid_real_world_maps_fail_explicitly(tmp_path: Path, mutation: str, match: str) -> None:
    path = tmp_path / "invalid-rw.dcm"
    ds = record(path, enhanced=False)
    mapping = rw_linear()
    ds.RealWorldValueMappingSequence = [mapping]
    if mutation == "overlap":
        ds.RealWorldValueMappingSequence.append(rw_linear())
    elif mutation == "gap":
        mapping.RealWorldValueFirstValueMapped = 0
    elif mutation == "unit":
        mapping.RealWorldValueLastValueMapped = 0
        ds.RealWorldValueMappingSequence.append(rw_linear(low=1, code="mm"))
    elif mutation == "missing_code":
        del mapping.MeasurementUnitsCodeSequence
    elif mutation == "lut_length":
        del mapping.RealWorldValueSlope
        del mapping.RealWorldValueIntercept
        mapping.RealWorldValueLUTData = [1.0, 2.0]
    elif mutation == "both":
        mapping.RealWorldValueLUTData = list(range(16))
    elif mutation == "range":
        mapping.RealWorldValueLastValueMapped = -4
    save(ds, path)
    keywords: dict[str, Any] = {"value_units": "mm"} if mutation == "unit_override" else {}
    with pytest.raises(ContractError, match=match):
        read_dicom_series((path,), **keywords)


def test_optional_pydicom_missing_is_lazy_and_actionable(monkeypatch: Any) -> None:
    actual_import = importlib.import_module

    def missing(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pydicom":
            raise ModuleNotFoundError("authored missing optional pydicom", name=name)
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match=r"pydicom is required.*imaging"):
        read_dicom_series((Path("unused.dcm"),))


@pytest.mark.parametrize("bits", [32, 64])
def test_real_world_floating_pixels_use_double_range_without_modality_rescale(
    tmp_path: Path, bits: int
) -> None:
    path = tmp_path / "floating-rw.dcm"
    ds = record(path, enhanced=False)
    del ds.PixelData
    del ds.PixelRepresentation
    del ds.BitsStored
    del ds.HighBit
    ds.BitsAllocated = bits
    stored = np.array([[0.125, -0.25, 1.5], [2.75, 0.0, -1.0]], dtype=f"<f{bits // 8}")
    setattr(ds, "FloatPixelData" if bits == 32 else "DoubleFloatPixelData", stored.tobytes())
    mapping = rw_linear(slope=2.5, intercept=0.75)
    del mapping.RealWorldValueFirstValueMapped
    del mapping.RealWorldValueLastValueMapped
    mapping.DoubleFloatRealWorldValueFirstValueMapped = -1.0
    mapping.DoubleFloatRealWorldValueLastValueMapped = 2.75
    ds.RealWorldValueMappingSequence = [mapping]
    save(ds, path)
    volume = read_dicom_series((path,))
    np.testing.assert_array_equal(volume.values, (stored.astype(np.float64) * 2.5 + 0.75)[None])
    assert volume.metadata.rescale == ((2.5, 0.75),)


def test_multiple_enhanced_files_form_one_position_ordered_volume(tmp_path: Path) -> None:
    paths = (tmp_path / "first.dcm", tmp_path / "second.dcm")
    first, second = (record(path) for path in paths)
    second.SeriesInstanceUID = first.SeriesInstanceUID
    second.FrameOfReferenceUID = first.FrameOfReferenceUID
    for frame in second.PerFrameFunctionalGroupsSequence:
        frame.PlanePositionSequence[0].ImagePositionPatient[2] += 12
        frame.FrameContentSequence[0].InStackPositionNumber += 3
        indices = frame.FrameContentSequence[0].DimensionIndexValues
        indices[0] += 3
    for ds, path in zip((first, second), paths, strict=True):
        save(ds, path)
    volume = read_dicom_series(paths[::-1])
    assert volume.shape == (6, 2, 3)
    np.testing.assert_array_equal(volume.values[:3], volume.values[3:])
    assert volume.grid.grid_to_object((0.0, 0.0, 5.0)) == (31.0, -9.0, 35.0)
    assert volume.metadata.source_sha256 == tuple(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("lut_and_rescale", "either rescale or Modality LUT"),
        ("lut_count", "count and bit depth"),
        ("lut_depth", "8 or 16"),
        ("shared_frame_content", "only in per-frame"),
        ("dimension_count", "dimension schema"),
        ("dimension_zero", "positive ordinals"),
        ("transform_incomplete", "slope, intercept and units"),
        ("top_and_frame_rescale", "top-level and functional"),
        ("mapping_choice", "value_mapping must"),
        ("real_world_missing", "requires a real-world"),
    ],
)
def test_malformed_transforms_and_frame_metadata_fail_before_decode(
    tmp_path: Path, mutation: str, match: str
) -> None:
    path = tmp_path / "malformed.dcm"
    ds = record(path, enhanced=not mutation.startswith("lut"))
    keywords: dict[str, Any] = {}
    if mutation.startswith("lut"):
        lut = item(LUTDescriptor=[4, -1, 16], ModalityLUTType="HU")
        lut.add_new(0x00283006, "US", [10, 20, 45, 100])
        ds.ModalityLUTSequence = [lut]
        if mutation == "lut_and_rescale":
            ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        elif mutation == "lut_count":
            lut.LUTDescriptor = [5, -1, 16]
        elif mutation == "lut_depth":
            lut.LUTDescriptor = [4, -1, 12]
    elif mutation == "shared_frame_content":
        ds.SharedFunctionalGroupsSequence[0].FrameContentSequence = [item(StackID="one")]
    elif mutation == "dimension_count":
        ds.PerFrameFunctionalGroupsSequence[0].FrameContentSequence[0].DimensionIndexValues = [1]
    elif mutation == "dimension_zero":
        ds.PerFrameFunctionalGroupsSequence[0].FrameContentSequence[0].DimensionIndexValues = [0, 1]
    elif mutation == "transform_incomplete":
        del ds.PerFrameFunctionalGroupsSequence[0].PixelValueTransformationSequence[0].RescaleType
    elif mutation == "top_and_frame_rescale":
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
    elif mutation == "mapping_choice":
        keywords["value_mapping"] = "invalid"
    elif mutation == "real_world_missing":
        keywords["value_mapping"] = "real_world"
    # Empty pixels demonstrate that metadata rejection precedes decoding.
    ds.PixelData = b""
    save(ds, path)
    with pytest.raises(ContractError, match=match):
        read_dicom_series((path,), **keywords)


def test_metadata_new_defaults_preserve_legacy_constructor() -> None:
    metadata = ImagingMetadata("synthetic", "synthetic", "dimensionless")
    assert metadata.value_mappings == ()
    assert metadata.measurement_unit_code is None


def test_imaging_module_import_does_not_load_optional_dependencies() -> None:
    import subprocess
    import sys

    from dpt import imaging_io

    python_root = Path(imaging_io.__file__).parents[1]
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import dpt.imaging_io; "
        "assert 'numpy' not in sys.modules; assert 'pydicom' not in sys.modules; "
        "assert 'nibabel' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(python_root)],
        check=True,
        capture_output=True,
        text=True,
    )
