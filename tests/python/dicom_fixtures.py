"""Synthetic nonclinical DICOM records with independent integer pixel patterns."""

import importlib
from pathlib import Path
from typing import Any

import numpy as np

pydicom: Any = importlib.import_module("pydicom")


def item(**attributes: Any) -> Any:
    record = pydicom.dataset.Dataset()
    for name, value in attributes.items():
        setattr(record, name, value)
    return record


def pixels() -> Any:
    return np.array(
        [[[12, -3, 0], [7, 2, 4]], [[1, 8, -2], [6, 3, 9]], [[10, 5, 11], [-1, 4, 2]]],
        dtype=np.int16,
    )


def record(path: Path, *, enhanced: bool = True, count: int = 3) -> Any:
    meta = pydicom.dataset.FileMetaDataset()
    meta.MediaStorageSOPClassUID = (
        pydicom.uid.EnhancedCTImageStorage if enhanced else pydicom.uid.CTImageStorage
    )
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds = pydicom.dataset.FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
    ds.PatientName = "Authored^Nonclinical"
    ds.PatientID = "DPT_INDEPENDENT_SYNTHETIC"
    ds.Modality = "CT"
    ds.Rows, ds.Columns = 2, 3
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    # Enhanced world formula: (31 + 1.2*x - y, -9 + 1.6*x + .75*y, 15 + 4*z).
    orientation = [0.6, 0.8, 0.0, -0.8, 0.6, 0.0]
    if enhanced:
        ds.NumberOfFrames = count
        ds.VolumetricProperties = "VOLUME"
        ds.DimensionOrganizationSequence = [item(DimensionOrganizationUID="1.2.826.0.1.1")]
        ds.DimensionIndexSequence = [
            item(
                DimensionOrganizationUID="1.2.826.0.1.1",
                DimensionIndexPointer=0x00209057,
                FunctionalGroupPointer=0x00209111,
            ),
            item(
                DimensionOrganizationUID="1.2.826.0.1.1",
                DimensionIndexPointer=0x00209128,
                FunctionalGroupPointer=0x00209111,
            ),
        ]
        ds.SharedFunctionalGroupsSequence = [
            item(
                PlaneOrientationSequence=[item(ImageOrientationPatient=orientation)],
                PixelMeasuresSequence=[item(PixelSpacing=[1.25, 2.0], SliceThickness=19.0)],
            )
        ]
        ds.PerFrameFunctionalGroupsSequence = []
        # Encoded order is z=2,0,1: InstanceNumber and frame ordinal cannot establish geometry.
        for z in (2, 0, 1)[:count]:
            ds.PerFrameFunctionalGroupsSequence.append(
                item(
                    PlanePositionSequence=[item(ImagePositionPatient=[31.0, -9.0, 15.0 + 4 * z])],
                    PixelValueTransformationSequence=[
                        item(
                            RescaleSlope=float(z + 2),
                            RescaleIntercept=float(-100 + 3 * z),
                            RescaleType="HU",
                        )
                    ],
                    FrameContentSequence=[
                        item(
                            StackID="one",
                            InStackPositionNumber=z + 1,
                            TemporalPositionIndex=1,
                            DimensionIndexValues=[z + 1, 1],
                        )
                    ],
                )
            )
        ds.PixelData = pixels()[[2, 0, 1][:count]].astype("<i2").tobytes()
    else:
        ds.ImageOrientationPatient = orientation
        ds.ImagePositionPatient = [31.0, -9.0, 15.0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SpacingBetweenSlices = 4.0
        ds.PixelData = pixels()[0].astype("<i2").tobytes()
    return ds


def save(ds: Any, path: Path) -> Path:
    ds.save_as(path, enforce_file_format=True)
    return path


def rw_linear(
    *,
    low: int = -3,
    high: int = 12,
    slope: float = 0.25,
    intercept: float = -2.0,
    code: str = "1/mm",
) -> Any:
    return item(
        RealWorldValueFirstValueMapped=low,
        RealWorldValueLastValueMapped=high,
        RealWorldValueSlope=slope,
        RealWorldValueIntercept=intercept,
        LUTLabel="authored",
        LUTExplanation="Independent synthetic mapping",
        MeasurementUnitsCodeSequence=[
            item(CodeValue=code, CodingSchemeDesignator="UCUM", CodeMeaning="synthetic unit")
        ],
    )


def remove_rescale(ds: Any) -> None:
    for frame in ds.PerFrameFunctionalGroupsSequence:
        if hasattr(frame, "PixelValueTransformationSequence"):
            del frame.PixelValueTransformationSequence


def rle_encode(ds: Any) -> None:
    """Hand-encode literal PackBits segments, independent of pydicom's encoder."""
    import struct

    raw = np.frombuffer(ds.PixelData, dtype="<i2").reshape(-1, 2, 3)
    frames = []
    for plane in raw:
        words = plane.astype("<i2").tobytes()
        # DICOM RLE: most significant byte segment first, rows encoded separately.
        segments = []
        for byte_index in (1, 0):
            body = bytearray()
            for row in range(2):
                body.append(2)  # 3 literal bytes follow.
                body.extend(words[2 * 3 * row + byte_index : 2 * 3 * (row + 1) : 2])
            segments.append(bytes(body))
        header = struct.pack("<16I", 2, 64, 64 + len(segments[0]), *([0] * 13))
        frames.append(header + b"".join(segments))
    ds.PixelData = pydicom.encaps.encapsulate(frames)
    ds["PixelData"].is_undefined_length = True
    ds.file_meta.TransferSyntaxUID = pydicom.uid.RLELossless
