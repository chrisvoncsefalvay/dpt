"""Validated DICOM frame and stored-value transforms; optional imports stay lazy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, cast

from dpt.contracts import ContractError, finite_scalar

type ValueMapping = Literal["auto", "modality", "real_world"]
type MappingKind = Literal[
    "identity",
    "rescale",
    "modality_lut",
    "real_world_linear",
    "real_world_lut",
    "real_world_piecewise",
]
type UnitCode = tuple[str, str, str]


def _one(sequence: Any, name: str) -> Any:
    if sequence is None or len(sequence) != 1:
        raise ContractError(f"DICOM requires {name} with one item")
    return sequence[0]


def _macro(shared: Any, frame: Any, name: str, *, required: bool = False) -> Any:
    common, individual = getattr(shared, name, None), getattr(frame, name, None)
    if common is not None and individual is not None:
        raise ContractError(f"DICOM {name} cannot occur in shared and per-frame groups")
    sequence = individual if individual is not None else common
    if required or sequence is not None:
        return _one(sequence, name)
    return None


def _unit_code(mapping: Any) -> UnitCode:
    unit = _one(
        getattr(mapping, "MeasurementUnitsCodeSequence", None), "MeasurementUnitsCodeSequence"
    )
    scheme = str(getattr(unit, "CodingSchemeDesignator", "")).strip()
    codes = [
        (name, str(getattr(unit, name)).strip())
        for name in ("CodeValue", "LongCodeValue", "URNCodeValue")
        if hasattr(unit, name)
    ]
    if len(codes) != 1 or not codes[0][1] or (not scheme and codes[0][0] != "URNCodeValue"):
        raise ContractError("DICOM measurement units require one code and coding scheme")
    if codes[0][0] == "URNCodeValue" and not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9+.-]*:\S+", codes[0][1]
    ):
        raise ContractError("DICOM URNCodeValue requires an absolute URN or URL")
    if not scheme and hasattr(unit, "CodingSchemeVersion"):
        raise ContractError("DICOM CodingSchemeVersion requires CodingSchemeDesignator")
    return scheme, codes[0][1], str(getattr(unit, "CodingSchemeVersion", ""))


@dataclass(frozen=True, slots=True)
class _RealWorldItem:
    low: float
    high: float
    rescale: tuple[float, float] | None
    lut: Any
    unit_code: UnitCode


@dataclass(frozen=True, slots=True)
class DicomTransform:
    kind: MappingKind
    units: str
    unit_code: UnitCode | None = None
    rescale: tuple[float, float] | None = None
    modality: Any = None
    real_world: tuple[_RealWorldItem, ...] = ()

    def apply(self, pixels: Any, np: Any) -> Any:
        if self.kind == "modality_lut":
            if not np.issubdtype(pixels.dtype, np.integer):
                raise ContractError("DICOM Modality LUT requires integer stored pixels")
            first, lut = self.modality
            # PS3.3 C.11.1 clips stored values outside the LUT at each endpoint.
            # Widen signed inputs before subtraction to prevent integer overflow.
            indices = np.clip(pixels.astype(np.int64) - first, 0, len(lut) - 1)
            return lut[indices]
        if not self.real_world:
            slope, intercept = self.rescale if self.rescale is not None else (1.0, 0.0)
            return slope * pixels.astype(np.float64) + intercept
        output = np.empty(pixels.shape, dtype=np.float64)
        covered = np.zeros(pixels.shape, dtype=bool)
        for mapping in self.real_world:
            mask = (pixels >= mapping.low) & (pixels <= mapping.high)
            if mapping.rescale is not None:
                slope, intercept = mapping.rescale
                output[mask] = slope * pixels[mask].astype(np.float64) + intercept
            else:
                if not np.issubdtype(pixels.dtype, np.integer):
                    raise ContractError("DICOM real-world LUT requires integer stored pixels")
                indices = pixels[mask].astype(np.int64) - int(mapping.low)
                output[mask] = mapping.lut[indices]
            covered |= mask
        if not covered.all():
            raise ContractError("DICOM real-world mapping leaves unmapped stored pixels")
        return output


def _real_world(sequence: Any, np: Any) -> DicomTransform:
    if not sequence:
        raise ContractError("DICOM RealWorldValueMappingSequence requires mapping items")
    mappings: list[_RealWorldItem] = []
    for mapping in sequence:
        code = _unit_code(mapping)
        integer_names = ("RealWorldValueFirstValueMapped", "RealWorldValueLastValueMapped")
        double_names = (
            "DoubleFloatRealWorldValueFirstValueMapped",
            "DoubleFloatRealWorldValueLastValueMapped",
        )
        integer = tuple(getattr(mapping, name, None) for name in integer_names)
        double = tuple(getattr(mapping, name, None) for name in double_names)
        if all(value is not None for value in integer) and all(value is None for value in double):
            low, high = (
                finite_scalar(float(cast(Any, value)), "real-world range") for value in integer
            )
        elif all(value is not None for value in double) and all(value is None for value in integer):
            low, high = (
                finite_scalar(float(cast(Any, value)), "real-world range") for value in double
            )
        else:
            raise ContractError("DICOM real-world range requires exactly one complete range pair")
        if high < low:
            raise ContractError("DICOM real-world mapping range must be ordered")
        lut_data = getattr(mapping, "RealWorldValueLUTData", None)
        slope = getattr(mapping, "RealWorldValueSlope", None)
        intercept = getattr(mapping, "RealWorldValueIntercept", None)
        rescale, lut = None, None
        if lut_data is not None:
            if slope is not None or intercept is not None:
                raise ContractError(
                    "DICOM real-world mapping requires either linear or LUT mapping"
                )
            if integer[0] is None or low != int(low) or high != int(high):
                raise ContractError("DICOM real-world LUT requires integer range endpoints")
            lut = np.atleast_1d(np.asarray(lut_data, dtype=np.float64))
            if lut.ndim != 1 or len(lut) != int(high - low + 1) or not np.isfinite(lut).all():
                raise ContractError(
                    "DICOM real-world LUT must have one finite entry per stored value"
                )
        elif slope is not None and intercept is not None:
            rescale = (
                finite_scalar(float(slope), "RealWorldValueSlope"),
                finite_scalar(float(intercept), "RealWorldValueIntercept"),
            )
        else:
            raise ContractError("DICOM real-world mapping requires complete linear or LUT mapping")
        mappings.append(_RealWorldItem(low, high, rescale, lut, code))
    mappings.sort(key=lambda mapping: mapping.low)
    if len({mapping.unit_code for mapping in mappings}) != 1:
        raise ContractError("DICOM real-world mapping items must share measurement units")
    if any(left.high >= right.low for left, right in pairwise(mappings)):
        raise ContractError(
            "DICOM real-world mapping ranges overlap; select an unambiguous mapping"
        )
    code = mappings[0].unit_code
    units = code[1] if code[0] in ("UCUM", "") else f"{code[0]}:{code[1]}"
    kind: MappingKind = "real_world_piecewise"
    if len(mappings) == 1:
        kind = "real_world_linear" if mappings[0].rescale is not None else "real_world_lut"
    return DicomTransform(
        kind,
        units,
        code,
        mappings[0].rescale if len(mappings) == 1 else None,
        real_world=tuple(mappings),
    )


def _modality(dataset: Any, transform: Any, np: Any) -> DicomTransform:
    sequence = getattr(dataset, "ModalityLUTSequence", None)
    if sequence is not None:
        if (
            transform is not None
            or hasattr(dataset, "RescaleSlope")
            or hasattr(dataset, "RescaleIntercept")
        ):
            raise ContractError("DICOM modality transform must use either rescale or Modality LUT")
        lut = _one(sequence, "ModalityLUTSequence")
        descriptor = getattr(lut, "LUTDescriptor", None)
        if descriptor is None or len(descriptor) != 3:
            raise ContractError("DICOM Modality LUT requires a three-value LUTDescriptor")
        count, first, depth = (int(value) for value in descriptor)
        if count < 0 or count > 65535 or depth not in (8, 16):
            raise ContractError("DICOM Modality LUT descriptor requires 8 or 16 bit entries")
        count = count or 65536
        data = getattr(lut, "LUTData", None)
        if data is None:
            raise ContractError("DICOM Modality LUT requires LUTData")
        if lut["LUTData"].VR == "OW":
            if not isinstance(data, bytes):
                raise ContractError("DICOM Modality LUT word data must match LUTDescriptor")
            endian = "<" if dataset.file_meta.TransferSyntaxUID.is_little_endian else ">"
            if len(data) == 2 * count:
                # Also accepts the standard's documented padded 8-bit word encoding.
                values = np.frombuffer(data, dtype=f"{endian}u2")
            elif depth == 8 and len(data) == count + count % 2:
                # OW endian ordering applies to words even when two entries share one word.
                words = np.frombuffer(data, dtype=f"{endian}u2")
                values = np.frombuffer(words.astype("<u2").tobytes(), dtype=np.uint8)[:count]
            else:
                raise ContractError("DICOM Modality LUT word data must match LUTDescriptor")
        else:
            values = np.atleast_1d(np.asarray(data))
        if (
            values.ndim != 1
            or len(values) != count
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0)
            or np.any(values >= 2**depth)
        ):
            raise ContractError(
                "DICOM Modality LUT entries must match descriptor count and bit depth"
            )
        if first < -32768 or first > 65535:
            raise ContractError("DICOM Modality LUT first mapped value is invalid")
        units = str(getattr(lut, "ModalityLUTType", "")).strip()
        if not units:
            raise ContractError("DICOM Modality LUT requires ModalityLUTType units")
        return DicomTransform("modality_lut", units, modality=(first, values))
    source = dataset if transform is None else transform
    if transform is not None and not all(
        hasattr(source, name) for name in ("RescaleSlope", "RescaleIntercept", "RescaleType")
    ):
        raise ContractError(
            "DICOM PixelValueTransformationSequence requires slope, intercept and units"
        )
    slope = finite_scalar(float(getattr(source, "RescaleSlope", 1.0)), "RescaleSlope")
    intercept = finite_scalar(float(getattr(source, "RescaleIntercept", 0.0)), "RescaleIntercept")
    kind: MappingKind = (
        "rescale"
        if any(hasattr(source, name) for name in ("RescaleSlope", "RescaleIntercept"))
        else "identity"
    )
    return DicomTransform(
        kind, str(getattr(source, "RescaleType", "unknown")), rescale=(slope, intercept)
    )


@dataclass(frozen=True, slots=True)
class DicomFrame:
    header: Any
    source_index: int
    frame_index: int
    frame_count: int
    enhanced: bool
    dimensions: tuple[tuple[str, str], ...]
    transform: DicomTransform


def expand_frames(
    datasets: list[Any], *, value_mapping: ValueMapping, np: Any, pydicom: Any
) -> list[DicomFrame]:
    if value_mapping not in ("auto", "modality", "real_world"):
        raise ContractError("value_mapping must be auto, modality or real_world")
    frames: list[DicomFrame] = []
    for source_index, dataset in enumerate(datasets):
        count = int(getattr(dataset, "NumberOfFrames", 1))
        if count < 1:
            raise ContractError("DICOM number of frames must be positive")
        enhanced = any(
            hasattr(dataset, name)
            for name in ("SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence")
        )
        if count > 1 and not enhanced:
            raise ContractError("multiframe DICOM requires explicit per-frame patient geometry")
        if int(getattr(dataset, "SamplesPerPixel", 1)) != 1 or str(
            getattr(dataset, "PhotometricInterpretation", "")
        ) not in ("MONOCHROME1", "MONOCHROME2"):
            raise ContractError("DICOM ingestion requires scalar monochrome pixels")
        if str(getattr(dataset, "VolumetricProperties", "VOLUME")) != "VOLUME":
            raise ContractError("DICOM scalar geometry requires VOLUME volumetric properties")
        shared = (
            _one(
                getattr(dataset, "SharedFunctionalGroupsSequence", None),
                "SharedFunctionalGroupsSequence",
            )
            if enhanced
            else pydicom.dataset.Dataset()
        )
        if hasattr(shared, "FrameContentSequence"):
            raise ContractError("DICOM FrameContentSequence belongs only in per-frame groups")
        groups = getattr(dataset, "PerFrameFunctionalGroupsSequence", None)
        if groups is not None and len(groups) != count:
            raise ContractError("DICOM per-frame group count must equal number of frames")
        organizations = tuple(
            sorted(
                str(getattr(organization, "DimensionOrganizationUID", ""))
                for organization in getattr(dataset, "DimensionOrganizationSequence", ())
            )
        )
        schema: list[tuple[int, str, int, str, str]] = []
        for dimension in getattr(dataset, "DimensionIndexSequence", ()):
            uid = str(getattr(dimension, "DimensionOrganizationUID", ""))
            pointer = getattr(dimension, "DimensionIndexPointer", None)
            if not uid or uid not in organizations or pointer is None:
                raise ContractError(
                    "DICOM dimension schema requires declared organization UID and pointer"
                )
            schema.append(
                (
                    int(pointer),
                    uid,
                    int(getattr(dimension, "FunctionalGroupPointer", 0)),
                    str(getattr(dimension, "DimensionIndexPrivateCreator", "")),
                    str(getattr(dimension, "FunctionalGroupPrivateCreator", "")),
                )
            )
        for frame_index in range(count):
            group = groups[frame_index] if groups is not None else pydicom.dataset.Dataset()
            header = pydicom.dataset.Dataset()
            for name in (
                "Rows",
                "Columns",
                "SeriesInstanceUID",
                "FrameOfReferenceUID",
                "AnatomicalOrientationType",
            ):
                if hasattr(dataset, name):
                    setattr(header, name, getattr(dataset, name))
            if enhanced:
                for macro, names in (
                    ("PlaneOrientationSequence", ("ImageOrientationPatient",)),
                    ("PlanePositionSequence", ("ImagePositionPatient",)),
                    ("PixelMeasuresSequence", ("PixelSpacing", "SpacingBetweenSlices")),
                ):
                    contents = _macro(shared, group, macro, required=True)
                    for name in names:
                        if hasattr(contents, name):
                            setattr(header, name, getattr(contents, name))
            else:
                for name in (
                    "ImageOrientationPatient",
                    "ImagePositionPatient",
                    "PixelSpacing",
                    "SpacingBetweenSlices",
                ):
                    if hasattr(dataset, name):
                        setattr(header, name, getattr(dataset, name))
            content = _macro(shared, group, "FrameContentSequence")
            dimensions: list[tuple[str, str]] = []
            temporal = _macro(shared, group, "TemporalPositionSequence")
            if temporal is not None:
                offset = getattr(temporal, "TemporalPositionTimeOffset", None)
                if offset is None:
                    raise ContractError("DICOM TemporalPositionSequence requires a time offset")
                dimensions.append(
                    (
                        "TemporalPositionTimeOffset",
                        str(finite_scalar(float(offset), "TemporalPositionTimeOffset")),
                    )
                )
            for name in ("TemporalPositionIdentifier", "EchoNumbers"):
                if hasattr(dataset, name):
                    dimensions.append((name, str(getattr(dataset, name))))
            if content is not None:
                for name in ("StackID", "TemporalPositionIndex"):
                    if hasattr(content, name):
                        # Ordinals and StackID are scoped to a Dimension Organization UID.
                        # Cross-organization stacks require explicit selection/comparison.
                        scope = ":".join(organizations)
                        if name == "StackID" and not scope:
                            raise ContractError(
                                "DICOM StackID requires a dimension organization UID"
                            )
                        dimensions.append((f"{name}:{scope}", str(getattr(content, name))))
            if schema:
                indices = getattr(content, "DimensionIndexValues", None)
                indices = [indices] if isinstance(indices, int) else indices
                if indices is None or len(indices) != len(schema):
                    raise ContractError("DICOM DimensionIndexValues must match dimension schema")
                # Spatial order comes from physical coordinates; all other dimensions must be fixed.
                for definition, value in zip(schema, indices, strict=True):
                    if int(value) < 1:
                        raise ContractError("DICOM dimension indices must be positive ordinals")
                    pointer, uid, group_pointer, private, group_private = definition
                    if pointer not in (0x00209057, 0x00200032):
                        identity = (
                            f"dimension:{uid}:{pointer:08x}:{group_pointer:08x}:"
                            f"{private}:{group_private}"
                        )
                        dimensions.append((identity, str(value)))
            transform = _macro(shared, group, "PixelValueTransformationSequence")
            if (
                enhanced
                and transform is not None
                and any(hasattr(dataset, name) for name in ("RescaleSlope", "RescaleIntercept"))
            ):
                raise ContractError(
                    "DICOM rescale cannot occur at both top-level and functional groups"
                )
            common_rw = getattr(shared, "RealWorldValueMappingSequence", None)
            frame_rw = getattr(group, "RealWorldValueMappingSequence", None)
            top_rw = getattr(dataset, "RealWorldValueMappingSequence", None)
            candidates = [
                candidate for candidate in (common_rw, frame_rw, top_rw) if candidate is not None
            ]
            if len(candidates) > 1:
                raise ContractError(
                    "DICOM real-world mapping cannot occur at multiple group locations"
                )
            rw = candidates[0] if candidates else None
            has_modality = transform is not None or any(
                hasattr(dataset, name)
                for name in ("ModalityLUTSequence", "RescaleSlope", "RescaleIntercept")
            )
            if rw is not None and has_modality and value_mapping == "auto":
                raise ContractError(
                    "DICOM has independent modality and real-world transforms; choose value_mapping"
                )
            if value_mapping == "real_world" and rw is None:
                raise ContractError(
                    "value_mapping real_world requires a real-world mapping for every frame"
                )
            use_rw = rw is not None and value_mapping != "modality"
            conversion = _real_world(rw, np) if use_rw else _modality(dataset, transform, np)
            frames.append(
                DicomFrame(
                    header,
                    source_index,
                    frame_index,
                    count,
                    enhanced,
                    tuple(dimensions),
                    conversion,
                )
            )
    if len({frame.dimensions for frame in frames}) > 1:
        raise ContractError(
            "DICOM ingestion requires one scalar spatial volume "
            "with fixed temporal/energy/stack dimensions"
        )
    return frames
