"""Decode synthetic scalar volumes through optional real JPEG-family codecs.

Lossless cases compare with integer patterns fixed before compression. Lossy
cases bound stored-pixel error before applying each frame's physical mapping.
Missing optional encoders or decoders skip the corresponding codec cases.
"""

import importlib
import io
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
pydicom = pytest.importorskip("pydicom")

from dicom_fixtures import item, record, save  # noqa: E402
from dpt.imaging_io import read_dicom_series  # noqa: E402

_CODECS = {
    "jpeg": ("JPEGBaseline8Bit", 8, 3),
    "jpeg_ls": ("JPEGLSLossless", 16, 0),
    "jpeg_ls_near": ("JPEGLSNearLossless", 16, 2),
    "jpeg2000": ("JPEG2000Lossless", 16, 0),
    "jpeg2000_lossy": ("JPEG2000", 8, 3),
}


def _codec(codec: str) -> tuple[Any, int, int]:
    name, bits, tolerance = _CODECS[codec]
    syntax = getattr(pydicom.uid, name)
    decoder = pydicom.pixels.get_decoder(syntax)
    if not decoder.is_available:
        pytest.skip(f"{name} requires an optional DICOM decoder")
    if codec.startswith("jpeg_ls"):
        pytest.importorskip("jpeg_ls")
    else:
        pytest.importorskip("PIL.Image")
        features = pytest.importorskip("PIL.features")
        feature = "jpg_2000" if codec.startswith("jpeg2000") else "jpg"
        if not features.check(feature):
            pytest.skip(f"Pillow was built without {feature}")
    return syntax, bits, tolerance


def _stored(bits: int) -> Any:
    z, y, x = np.indices((3, 32, 40), dtype=np.int64)
    if bits == 8:
        return ((2 * x + y + 19 * z) % 256).astype(np.uint8)
    # Includes both byte planes, unsigned extrema and spatial edges.
    values = ((131 * x + 47 * y + 701 * z) ^ ((x % 7) << 8)).astype(np.uint16)
    values[0, 0, 0], values[2, -1, -1] = 0, 65535
    return values


def _encode(plane: Any, codec: str) -> bytes:
    if codec.startswith("jpeg_ls"):
        jpeg_ls = importlib.import_module("jpeg_ls")
        return bytes(jpeg_ls.encode(plane, lossy_error=2 if codec.endswith("near") else 0))
    image_module = importlib.import_module("PIL.Image")
    output = io.BytesIO()
    image = image_module.fromarray(plane)
    if codec == "jpeg":
        image.save(output, format="JPEG", quality=100, subsampling=0, optimize=False)
    else:
        image.save(
            output,
            format="JPEG2000",
            no_jp2=True,
            irreversible=codec.endswith("lossy"),
            num_resolutions=3,
        )
    return output.getvalue()


def _volume_files(
    directory: Path, codec: str, enhanced: bool, mapping: str
) -> tuple[tuple[Path, ...], Any, int]:
    syntax, bits, tolerance = _codec(codec)
    original = _stored(bits)
    if mapping != "linear":
        # FD has a 16-bit explicit-VR length; keep this separate LUT fixture
        # below that encoding limit while retaining a nonlinear oracle.
        original %= 4096
    order = (2, 0, 1)
    directory.mkdir()
    paths = []
    for ordinal, z in enumerate((0,) if enhanced else order):
        path = directory / f"slice-{ordinal}.dcm"
        ds = record(path, enhanced=enhanced)
        ds.Rows, ds.Columns = original.shape[1:]
        ds.BitsAllocated = bits
        ds.BitsStored = bits
        ds.HighBit = bits - 1
        ds.PixelRepresentation = 0
        ds.SeriesInstanceUID = "1.2.826.0.1.3680043.10.999.100"
        ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.10.999.101"
        ds.SOPInstanceUID = f"1.2.826.0.1.3680043.10.999.102.{ordinal + 1}"
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        frame_indices = order if enhanced else (z,)
        ds.PixelData = pydicom.encaps.encapsulate(
            [_encode(original[frame], codec) for frame in frame_indices]
        )
        ds["PixelData"].is_undefined_length = True
        ds.file_meta.TransferSyntaxUID = syntax
        if tolerance:
            ds.LossyImageCompression = "01"
            ds.LossyImageCompressionMethod = (
                "ISO_14495_1"
                if codec.startswith("jpeg_ls")
                else "ISO_15444_1"
                if codec.startswith("jpeg2000")
                else "ISO_10918_1"
            )
        if not enhanced:
            ds.ImagePositionPatient = [31.0, -9.0, 15.0 + 4 * z]
            ds.PixelSpacing = [1.25, 2.0]
        if mapping == "linear":
            if not enhanced:
                ds.RescaleSlope, ds.RescaleIntercept, ds.RescaleType = z + 2, -100 + 3 * z, "HU"
        else:
            if enhanced:
                for frame in ds.PerFrameFunctionalGroupsSequence:
                    del frame.PixelValueTransformationSequence
            maximum = int(original.max())
            if mapping == "modality_lut":
                ds.ModalityLUTSequence = [
                    item(
                        LUTDescriptor=[maximum + 1, 0, 16],
                        LUTData=np.array(
                            [
                                (value * value + 7 * value + 13) % 65536
                                for value in range(maximum + 1)
                            ],
                            dtype="<u2",
                        ).tobytes(),
                        ModalityLUTType="codec-unit",
                    )
                ]
            else:
                ds.RealWorldValueMappingSequence = [
                    item(
                        RealWorldValueFirstValueMapped=0,
                        RealWorldValueLastValueMapped=maximum,
                        RealWorldValueLUTData=[
                            value * value / 32 - 3 for value in range(maximum + 1)
                        ],
                        LUTLabel="authored-codec",
                        MeasurementUnitsCodeSequence=[
                            item(
                                CodeValue="1/mm",
                                CodingSchemeDesignator="UCUM",
                                CodeMeaning="inverse millimetre",
                            )
                        ],
                    )
                ]
        save(ds, path)
        paths.append(path)
    return tuple(paths), original, tolerance


@pytest.mark.parametrize("enhanced", [False, True], ids=["classic", "enhanced"])
@pytest.mark.parametrize("codec", tuple(_CODECS))
def test_real_compressed_pixels_and_per_frame_rescale(
    tmp_path: Path, codec: str, enhanced: bool
) -> None:
    paths, original, tolerance = _volume_files(tmp_path / "encoded", codec, enhanced, "linear")
    volume = read_dicom_series(paths)
    assert volume.shape == original.shape
    assert volume.grid.spacing_mm == pytest.approx((2.0, 1.25, 4.0))
    assert volume.metadata.coordinate_frame == "LPS"
    assert volume.metadata.value_units == "HU"
    assert volume.metadata.rescale == ((2.0, -100.0), (3.0, -97.0), (4.0, -94.0))
    for z in range(3):
        expected = original[z].astype(np.float64) * (z + 2) - 100 + 3 * z
        if tolerance:
            np.testing.assert_allclose(volume.values[z], expected, rtol=0, atol=tolerance * (z + 2))
        else:
            np.testing.assert_array_equal(volume.values[z], expected)
        assert volume.grid.grid_to_object((3.0, 4.0, float(z))) == pytest.approx(
            (31 + 1.2 * 3 - 4, -9 + 1.6 * 3 + 0.75 * 4, 15 + 4 * z), abs=1e-12
        )
    assert volume.values.dtype == np.float64
    assert not volume.values.flags.writeable


@pytest.mark.parametrize("enhanced", [False, True], ids=["classic", "enhanced"])
@pytest.mark.parametrize("codec", ["jpeg_ls", "jpeg2000"])
@pytest.mark.parametrize("mapping", ["modality_lut", "real_world_lut"])
def test_lossless_compression_preserves_nonlinear_mapping(
    tmp_path: Path, codec: str, enhanced: bool, mapping: str
) -> None:
    paths, original, tolerance = _volume_files(tmp_path / "encoded", codec, enhanced, mapping)
    assert tolerance == 0
    volume = read_dicom_series(paths)
    values = original.astype(np.float64)
    expected = (
        (values * values + 7 * values + 13) % 65536
        if mapping == "modality_lut"
        else values * values / 32 - 3
    )
    np.testing.assert_array_equal(volume.values, expected)
    assert volume.metadata.value_mappings == (mapping,) * 3
    if mapping == "real_world_lut":
        assert volume.metadata.measurement_unit_code == ("UCUM", "1/mm", "")
        assert volume.metadata.value_units == "1/mm"
