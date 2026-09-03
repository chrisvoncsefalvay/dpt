"""Read calibrated voxel values with explicit geometry, units and provenance.

NIfTI affines describe RAS coordinates; DICOM patient coordinates use LPS.
Both become right-handed ``GridSpec`` records in millimetres. Pixel rescaling
is applied once, and the resulting values retain their declared units.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._dicom_io import MappingKind, UnitCode, ValueMapping, expand_frames
from dpt.contracts import ContractError, finite_scalar
from dpt.geometry import Matrix3, Vector3
from dpt.volumes import GridSpec

if TYPE_CHECKING:
    import numpy as np
    from numpy import float64
    from numpy.typing import NDArray

type PathInput = str | os.PathLike[str]
type SpatialUnit = Literal["mm", "meter", "micron"]
_LENGTH_MM = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
_DIRECTION_TOLERANCE = 1e-6


def _optional(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"{module} is required for imaging IO; install "
            "`pip install 'differentiable-photon-transport[imaging]'`"
        ) from error


@dataclass(frozen=True, slots=True)
class ImagingMetadata:
    """Describe coordinates and scaling for ``values[z,y,x]``.

    ``coordinate_frame`` names the world axes; spatial lengths are always mm.
    ``rescale`` contains applied (slope,intercept) pairs in slice order when
    every frame has a linear mapping; otherwise it is empty. ``value_mappings``
    identifies each DICOM frame's transform. ``measurement_unit_code`` retains
    the (coding scheme, code, scheme version) of real-world mapping units.
    An empty coding scheme records a self-identifying URN/URI unit code.
    Source hashes identify the input bytes without retaining patient tags.
    ``affine_error_mm`` bounds the header-rounding correction over the grid.
    """

    source_format: Literal["nifti", "dicom", "synthetic"]
    coordinate_frame: Literal["RAS", "LPS", "synthetic"]
    value_units: str
    source_sha256: tuple[str, ...] = ()
    source_shape: tuple[int, ...] = ()
    affine_source: Literal["sform", "qform", "dicom", "synthetic"] = "synthetic"
    reindexed_axes: tuple[int, ...] = ()
    rescale: tuple[tuple[float, float], ...] = ()
    affine_error_mm: float = 0.0
    spatial_units: Literal["mm"] = "mm"
    value_mappings: tuple[MappingKind, ...] = ()
    measurement_unit_code: UnitCode | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.value_units), str) or not self.value_units.strip():
            raise ContractError("value_units must be a non-empty unit label")
        if self.spatial_units != "mm":
            raise ContractError("ImagingMetadata spatial units must be mm")
        if self.source_format not in ("nifti", "dicom", "synthetic"):
            raise ContractError("source_format must be nifti, dicom or synthetic")
        if self.coordinate_frame not in ("RAS", "LPS", "synthetic"):
            raise ContractError("coordinate_frame must be RAS, LPS or synthetic")
        if not isinstance(cast(object, self.source_sha256), tuple):
            raise ContractError("source_sha256 must be an immutable tuple")
        for digest in self.source_sha256:
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ContractError("source_sha256 must contain lowercase SHA256 digests")
        if any(axis not in (0, 1, 2) for axis in self.reindexed_axes):
            raise ContractError("reindexed_axes uses input (x,y,z) axis indices")
        if self.affine_source not in ("sform", "qform", "dicom", "synthetic"):
            raise ContractError("affine_source must name the coordinate mapping")
        if not isinstance(cast(object, self.source_shape), tuple) or any(
            type(count) is not int or count <= 0 for count in self.source_shape
        ):
            raise ContractError("source_shape must be an immutable tuple of positive counts")
        if not isinstance(cast(object, self.reindexed_axes), tuple) or len(
            set(self.reindexed_axes)
        ) != len(self.reindexed_axes):
            raise ContractError("reindexed_axes must be an immutable tuple of distinct axes")
        if not isinstance(cast(object, self.rescale), tuple):
            raise ContractError("rescale must be an immutable tuple")
        for pair in self.rescale:
            if not isinstance(cast(object, pair), tuple) or len(pair) != 2:
                raise ContractError("each rescale entry is (slope,intercept)")
            finite_scalar(pair[0], "rescale slope")
            finite_scalar(pair[1], "rescale intercept")
        finite_scalar(self.affine_error_mm, "affine_error_mm", minimum=0.0)
        if not isinstance(cast(object, self.value_mappings), tuple) or any(
            kind
            not in (
                "identity",
                "rescale",
                "modality_lut",
                "real_world_linear",
                "real_world_lut",
                "real_world_piecewise",
            )
            for kind in self.value_mappings
        ):
            raise ContractError("value_mappings must be an immutable tuple of mapping kinds")
        if self.measurement_unit_code is not None:
            code = self.measurement_unit_code
            if (
                not isinstance(cast(object, code), tuple)
                or len(code) != 3
                or any(not isinstance(cast(object, part), str) for part in code)
                or not code[1]
            ):
                raise ContractError("measurement_unit_code requires (scheme,code,version)")


@dataclass(frozen=True, slots=True)
class ImagingVolume:
    """Own finite real values of shape (nz,ny,nx), with x-fast contiguous storage.

    ``grid.grid_to_object((x,y,z))`` gives each voxel centre in the coordinate
    frame named by metadata. Values retain units such as HU, inverse mm or
    dimensionless labels. For example, ``volume.values.ravel()`` provides the
    x-fast array for an explicit FP32 conversion before device projection.
    """

    values: NDArray[Any]
    grid: GridSpec
    metadata: ImagingMetadata

    def __post_init__(self) -> None:
        np = _optional("numpy")
        raw = np.asarray(self.values)
        if raw.dtype.kind not in "biuf":
            raise ContractError("imaging values must be real")
        values = np.array(raw, order="C", copy=True)
        if tuple(values.shape) != self.grid.shape:
            raise ContractError("imaging values must have grid shape (nz,ny,nx)")
        if not np.isfinite(values).all():
            raise ContractError("imaging values must be finite")
        values.flags.writeable = False
        object.__setattr__(self, "values", values)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.grid.shape


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _grid_from_affine(
    affine_mm: NDArray[np.float64], shape_xyz: tuple[int, int, int]
) -> tuple[GridSpec, bool, float]:
    np = _optional("numpy")
    affine = np.array(affine_mm, dtype=np.float64, copy=True)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ContractError("image affine must be a finite 4 by 4 matrix")
    if not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-12):
        raise ContractError("image affine must have homogeneous row (0,0,0,1)")
    axes = affine[:3, :3]
    spacing = np.hypot.reduce(axes, axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
        raise ContractError("image affine must have finite positive voxel spacings")
    orientation = axes / spacing
    if np.max(np.abs(orientation.T @ orientation - np.eye(3))) > _DIRECTION_TOLERANCE:
        raise ContractError(
            "sheared image geometry needs explicit resampling before GridSpec ingestion"
        )
    flipped = bool(np.linalg.det(orientation) < 0.0)
    if flipped:
        affine[:3, 3] += affine[:3, 0] * (shape_xyz[0] - 1)
        axes[:, 0] *= -1.0
        orientation[:, 0] *= -1.0
    # Header direction cosines are often FP32. Remove only their rounding error.
    left, _, right = np.linalg.svd(orientation)
    rotation = left @ right
    residual = np.abs(rotation * spacing - axes)
    bound = float(np.sum(residual @ (np.array(shape_xyz) - 1)))
    grid = GridSpec(
        (shape_xyz[2], shape_xyz[1], shape_xyz[0]),
        cast(Vector3, tuple(float(value) for value in spacing)),
        cast(Vector3, tuple(float(value) for value in affine[:3, 3])),
        cast(Matrix3, tuple(float(value) for value in rotation.ravel())),
    )
    return grid, flipped, bound


def read_nifti(
    path: PathInput,
    *,
    value_units: str = "unknown",
    spatial_unit: SpatialUnit | None = None,
) -> ImagingVolume:
    """value=slope*stored+intercept and x_RAS=A_mm*[i,j,k,1] at voxel centres.

    Read one real scalar NIfTI-1/NIfTI-2 volume with input shape (nx,ny,nz).
    ``ImagingVolume.values`` owns read-only contiguous storage with shape
    (nz,ny,nx), where x varies fastest. The source dtype is preserved when
    scaling is identity; nontrivial scaling uses NiBabel's calculated dtype.
    Values keep the ``value_units``
    label, such as HU or mm^-1, without conversion to attenuation. The grid
    maps (x,y,z) indices to right-handed RAS coordinates in mm and metadata
    records original source hashes, scaling and any x reindexing.

    A coded sform takes precedence over qform. Header length units convert to
    mm, while ``spatial_unit`` supplies unknown units or confirms declared
    units. Left-handed affines reverse x storage while preserving locations.
    Sheared grids require resampling before ingestion. This loader applies
    scaling once and performs no differentiation or device upload.

        >>> import tempfile
        >>> from pathlib import Path
        >>> import nibabel as nib
        >>> import numpy as np
        >>> from dpt.imaging_io import read_nifti
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     path = Path(directory) / "field.nii"
        ...     image = nib.Nifti1Image(np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        ...                             np.diag((2., 3., 4., 1.)))
        ...     image.header.set_xyzt_units("mm")
        ...     image.set_sform(image.affine, code=1)
        ...     nib.save(image, path)
        ...     volume = read_nifti(path, value_units="HU")
        >>> (volume.shape, volume.grid.spacing_mm, volume.metadata.coordinate_frame)
        ((2, 2, 2), (2.0, 3.0, 4.0), 'RAS')
        >>> (volume.values[0, 0, 1], volume.values.dtype.name, volume.metadata.value_units)
        (np.float32(4.0), 'float32', 'HU')

    """
    nib = _optional("nibabel")
    np = _optional("numpy")
    source = Path(path)
    image = nib.load(source)
    if not isinstance(image.header, (nib.Nifti1Header, nib.Nifti2Header)):
        raise ContractError("read_nifti requires a NIfTI-1 or NIfTI-2 image")
    if len(image.shape) != 3 or any(count <= 0 for count in image.shape):
        raise ContractError("NIfTI ingestion requires one three-dimensional scalar volume")
    if np.issubdtype(image.get_data_dtype(), np.complexfloating):
        raise ContractError("NIfTI imaging values must be real")
    header_unit = image.header.get_xyzt_units()[0]
    if spatial_unit is not None and spatial_unit not in _LENGTH_MM:
        raise ContractError("spatial_unit must be mm, meter or micron")
    if header_unit in _LENGTH_MM and spatial_unit is not None and spatial_unit != header_unit:
        raise ContractError("spatial_unit conflicts with declared NIfTI header units")
    unit = header_unit if spatial_unit is None else spatial_unit
    if unit not in _LENGTH_MM:
        raise ContractError("NIfTI length units are unknown; provide spatial_unit")
    sform, sform_code = image.get_sform(coded=True)
    qform, qform_code = image.get_qform(coded=True)
    if sform_code:
        affine, affine_source = sform, "sform"
    elif qform_code:
        affine, affine_source = qform, "qform"
    else:
        raise ContractError("NIfTI geometry requires a coded sform or qform")
    affine = np.array(affine, dtype=np.float64, copy=True)
    affine[:3] *= _LENGTH_MM[unit]
    shape_xyz = cast(tuple[int, int, int], tuple(int(count) for count in image.shape))
    grid, flipped, bound = _grid_from_affine(affine, shape_xyz)
    # The proxy applies slope/intercept exactly once without forcing a second
    # full-volume FP64 representation for integer or FP32 source data.
    values = np.asanyarray(image.dataobj)
    if flipped:
        values = values[::-1, :, :]
    # Paired NIfTI images have both header and image bytes in their provenance.
    source_files = tuple(
        dict.fromkeys(
            Path(holder.filename)
            for holder in image.file_map.values()
            if holder.filename is not None
        )
    )
    metadata = ImagingMetadata(
        "nifti",
        "RAS",
        value_units,
        source_sha256=tuple(_sha256(path) for path in source_files),
        source_shape=shape_xyz,
        affine_source=affine_source,
        reindexed_axes=(0,) if flipped else (),
        rescale=((float(image.dataobj.slope), float(image.dataobj.inter)),),
        affine_error_mm=bound,
    )
    return ImagingVolume(values.transpose(2, 1, 0), grid, metadata)


def _dicom_vector(dataset: Any, name: str, length: int) -> tuple[float, ...]:
    value = getattr(dataset, name, None)
    if value is None or len(value) != length:
        raise ContractError(f"DICOM requires {name} with {length} values")
    return tuple(finite_scalar(float(item), f"DICOM {name}") for item in value)


def read_dicom_series(
    paths: Sequence[PathInput],
    *,
    value_units: str | None = None,
    single_slice_spacing_mm: float | None = None,
    value_mapping: ValueMapping = "auto",
) -> ImagingVolume:
    """x_LPS=o+c*s_c*u+r*s_r*v+k*s_z*(u cross v), with value=M(stored).

    ``paths`` is a non-empty sequence of DICOM files containing one regular
    scalar volume. Patient positions order classic slices or enhanced frames
    along their orientation normal. ``PixelSpacing`` uses (row,column) mm,
    while the returned grid spacing uses (column,row,slice) mm. The owned
    read-only binary64 values have shape (slices,rows,columns), with columns
    varying fastest, and retain their declared HU or real-world unit label.
    The grid is right-handed in LPS coordinates and metadata retains the
    applied mappings, coded units, source hashes and header correction bound.

    ``value_mapping`` selects the modality or real-world transform when both
    exist, and ``auto`` rejects that ambiguity. Linear rescaling is applied
    once, modality LUTs clamp at their endpoints and real-world maps require
    complete pixel coverage and consistent units. ``value_units`` supplies a
    missing label but does not convert values. A single slice requires declared
    SpacingBetweenSlices or ``single_slice_spacing_mm``. Compressed images use
    installed pydicom decoders, and tilted or multiple temporal/energy stacks
    require explicit resampling or volume selection. No parameters are active.

        >>> import tempfile
        >>> from pathlib import Path
        >>> import numpy as np
        >>> from pydicom.dataset import FileDataset, FileMetaDataset
        >>> from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
        >>> from dpt.imaging_io import read_dicom_series
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     path = Path(directory) / "slice.dcm"
        ...     meta = FileMetaDataset()
        ...     meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ...     meta.MediaStorageSOPClassUID = CTImageStorage
        ...     meta.MediaStorageSOPInstanceUID = generate_uid()
        ...     image = FileDataset(str(path), {}, file_meta=meta, preamble=bytes(128))
        ...     image.SOPClassUID = meta.MediaStorageSOPClassUID
        ...     image.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ...     image.SeriesInstanceUID = generate_uid()
        ...     image.FrameOfReferenceUID = generate_uid()
        ...     image.ImageOrientationPatient = [1., 0., 0., 0., 1., 0.]
        ...     image.ImagePositionPatient = [0., 0., 0.]
        ...     image.PixelSpacing = [2., 3.]
        ...     image.Rows, image.Columns = 2, 2
        ...     image.SamplesPerPixel, image.PhotometricInterpretation = 1, "MONOCHROME2"
        ...     image.BitsAllocated, image.BitsStored, image.HighBit = 16, 16, 15
        ...     image.PixelRepresentation = 1
        ...     image.RescaleSlope, image.RescaleIntercept, image.RescaleType = 2., -1000., "HU"
        ...     image.PixelData = np.arange(4, dtype=np.int16).reshape(2, 2).tobytes()
        ...     image.save_as(path, enforce_file_format=True)
        ...     volume = read_dicom_series((path,), single_slice_spacing_mm=4.)
        >>> (volume.shape, volume.grid.spacing_mm, volume.metadata.coordinate_frame)
        ((1, 2, 2), (3.0, 2.0, 4.0), 'LPS')
        >>> volume.values.tolist()
        [[[-1000.0, -998.0], [-996.0, -994.0]]]

    """
    pydicom = _optional("pydicom")
    np = _optional("numpy")
    if not paths or isinstance(paths, (str, os.PathLike)):
        raise ContractError("paths must be a non-empty sequence of DICOM files")
    sources = tuple(Path(path) for path in paths)
    datasets = [pydicom.dcmread(path) for path in sources]
    frames = expand_frames(datasets, value_mapping=value_mapping, np=np, pydicom=pydicom)
    first = frames[0].header
    rows, columns = int(getattr(first, "Rows", 0)), int(getattr(first, "Columns", 0))
    if min(rows, columns) <= 0:
        raise ContractError("DICOM Rows and Columns must be positive")
    orientation = _dicom_vector(first, "ImageOrientationPatient", 6)
    column_axis, row_axis = np.array(orientation[:3]), np.array(orientation[3:])
    normal = np.cross(column_axis, row_axis)
    basis = np.column_stack((column_axis, row_axis, normal))
    if np.max(np.abs(basis.T @ basis - np.eye(3))) > _DIRECTION_TOLERANCE:
        raise ContractError("DICOM orientation must have orthonormal direction cosines")
    normal /= np.linalg.norm(normal)
    row_spacing, column_spacing = _dicom_vector(first, "PixelSpacing", 2)
    if min(row_spacing, column_spacing) <= 0:
        raise ContractError("DICOM PixelSpacing must be positive")
    series = getattr(first, "SeriesInstanceUID", None)
    reference = getattr(first, "FrameOfReferenceUID", None)
    if not series or not reference:
        raise ContractError("DICOM stack requires SeriesInstanceUID and FrameOfReferenceUID")
    positions: list[NDArray[float64]] = []
    units: list[str] = []
    for frame in frames:
        dataset = frame.header
        if str(getattr(dataset, "AnatomicalOrientationType", "BIPED")) != "BIPED":
            raise ContractError("DICOM LPS ingestion requires BIPED patient coordinates")
        if (
            getattr(dataset, "SeriesInstanceUID", None) != series
            or getattr(dataset, "FrameOfReferenceUID", None) != reference
        ):
            raise ContractError("DICOM files must share one series and frame of reference")
        if (int(getattr(dataset, "Rows", 0)), int(getattr(dataset, "Columns", 0))) != (
            rows,
            columns,
        ):
            raise ContractError("DICOM slices must share Rows and Columns")
        if not np.allclose(
            _dicom_vector(dataset, "ImageOrientationPatient", 6),
            orientation,
            rtol=0.0,
            atol=_DIRECTION_TOLERANCE,
        ):
            raise ContractError("DICOM slices must share orientation")
        if not np.allclose(
            _dicom_vector(dataset, "PixelSpacing", 2),
            (row_spacing, column_spacing),
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ContractError("DICOM slices must share pixel spacing")
        positions.append(np.array(_dicom_vector(dataset, "ImagePositionPatient", 3)))
        units.append(frame.transform.units)
    expanded_mapping = any(
        frame.enhanced
        or frame.transform.kind
        in ("modality_lut", "real_world_linear", "real_world_lut", "real_world_piecewise")
        for frame in frames
    )
    if len(set(units)) > 1 and (value_units is None or expanded_mapping):
        raise ContractError("DICOM frames must share mapping units or explicit value_units")
    codes = {frame.transform.unit_code for frame in frames}
    if len(codes) > 1:
        raise ContractError("DICOM frames must share coded measurement units")
    if value_units is not None and any(unit not in ("unknown", value_units) for unit in units):
        raise ContractError("value_units conflicts with declared DICOM mapping units")
    label = units[0] if value_units is None else value_units
    distances = np.array([np.dot(position, normal) for position in positions])
    order = np.argsort(distances)
    origin = positions[int(order[0])]
    sorted_distances = distances[order]
    if len(frames) == 1:
        spacing_z = single_slice_spacing_mm
        if spacing_z is None:
            spacing_z = getattr(first, "SpacingBetweenSlices", None)
        if spacing_z is None:
            raise ContractError(
                "one DICOM slice requires single_slice_spacing_mm or SpacingBetweenSlices"
            )
        spacing_z = finite_scalar(float(spacing_z), "slice spacing", minimum=0.0)
        if spacing_z <= 0:
            raise ContractError("slice spacing must be positive")
    else:
        deltas = np.diff(sorted_distances)
        spacing_z = float(np.mean(deltas))
        if spacing_z <= 0 or not np.allclose(deltas, spacing_z, rtol=1e-5, atol=1e-4):
            raise ContractError("DICOM slice centres must have positive uniform spacing")
        for index in order:
            expected = origin + normal * (distances[index] - sorted_distances[0])
            if np.max(np.abs(positions[int(index)] - expected)) > 1e-4:
                raise ContractError(
                    "DICOM gantry tilt or sheared stacks require explicit resampling"
                )
    affine = np.eye(4)
    affine[:3, :3] = np.column_stack(
        (column_axis * column_spacing, row_axis * row_spacing, normal * spacing_z)
    )
    affine[:3, 3] = origin
    grid, flipped, bound = _grid_from_affine(affine, (columns, rows, len(frames)))
    if flipped:
        raise ContractError("DICOM orientation could not form a right-handed stack")
    grid_axes = np.array(grid.orientation).reshape(3, 3) * np.array(grid.spacing_mm)
    # Include accepted header/slice-position rounding in the world-coordinate bound.
    for slice_index, source_index in enumerate(order):
        dataset = frames[int(source_index)].header
        native_orientation = np.array(_dicom_vector(dataset, "ImageOrientationPatient", 6))
        native_spacing = _dicom_vector(dataset, "PixelSpacing", 2)
        column_error = np.abs(native_orientation[:3] * native_spacing[1] - grid_axes[:, 0])
        row_error = np.abs(native_orientation[3:] * native_spacing[0] - grid_axes[:, 1])
        centre_error = np.abs(
            positions[int(source_index)]
            - (np.array(grid.origin_mm) + slice_index * grid_axes[:, 2])
        )
        bound = max(
            bound,
            float(np.sum(centre_error + (columns - 1) * column_error + (rows - 1) * row_error)),
        )
    planes: list[NDArray[float64]] = []
    rescale: list[tuple[float, float]] = []
    hashes: list[str] = []
    value_mappings: list[MappingKind] = []
    decoded: dict[int, Any] = {}
    hashed_sources: set[int] = set()
    for index in order:
        frame = frames[int(index)]
        if frame.source_index not in decoded:
            try:
                pixels = np.asarray(datasets[frame.source_index].pixel_array)
            except (RuntimeError, NotImplementedError) as error:
                syntax = getattr(
                    datasets[frame.source_index].file_meta, "TransferSyntaxUID", "unknown"
                )
                raise RuntimeError(
                    "DICOM pixel decoding requires a pydicom decoder for this transfer syntax "
                    f"({syntax}): {error}"
                ) from error
            expected_shape = (
                (rows, columns) if frame.frame_count == 1 else (frame.frame_count, rows, columns)
            )
            if pixels.shape != expected_shape:
                raise ContractError(
                    "DICOM decoded scalar pixels must match declared frame/rows/columns shape"
                )
            if np.iscomplexobj(pixels) or not np.isfinite(pixels).all():
                raise ContractError("DICOM stored pixels must be finite and real")
            decoded[frame.source_index] = pixels
        pixels = decoded[frame.source_index]
        plane = pixels if frame.frame_count == 1 else pixels[frame.frame_index]
        planes.append(frame.transform.apply(plane, np))
        if frame.transform.rescale is not None:
            rescale.append(frame.transform.rescale)
        value_mappings.append(frame.transform.kind)
        if frame.source_index not in hashed_sources:
            hashes.append(_sha256(sources[frame.source_index]))
            hashed_sources.add(frame.source_index)
    return ImagingVolume(
        np.stack(planes),
        grid,
        ImagingMetadata(
            "dicom",
            "LPS",
            label,
            source_sha256=tuple(hashes),
            source_shape=(len(frames), rows, columns),
            affine_source="dicom",
            rescale=tuple(rescale) if len(rescale) == len(frames) else (),
            affine_error_mm=bound,
            value_mappings=tuple(value_mappings),
            measurement_unit_code=frames[0].transform.unit_code,
        ),
    )


def sphere_phantom(
    grid: GridSpec,
    *,
    radius_mm: float,
    centre_mm: Vector3 = (0.0, 0.0, 0.0),
    inside: float = 1.0,
    outside: float = 0.0,
    value_units: str = "dimensionless",
) -> ImagingVolume:
    """f(x)=inside if ||x-centre||<=radius, and outside otherwise.

    ``grid`` supplies oriented voxel centres in object mm, with shape (nz,ny,nx),
    spacing (x,y,z) and a row-major rotation. ``radius_mm`` is positive and
    ``centre_mm`` is a length-3 object coordinate. ``inside`` and ``outside``
    are finite scalar values in the declared ``value_units``. Return an
    ``ImagingVolume`` with read-only contiguous binary64 values in (z,y,x)
    order, the original grid and synthetic metadata. Boundary centres belong
    to the sphere and no fractional voxel occupancy is computed.

    This forward sampler has no active derivatives. It supplies an explicit
    field whose values can subsequently be converted to device storage.

        >>> from dpt.volumes import GridSpec
        >>> from dpt.imaging_io import sphere_phantom
        >>> grid = GridSpec((1, 1, 3), (1., 1., 1.), origin_mm=(-1., 0., 0.))
        >>> volume = sphere_phantom(grid, radius_mm=0.5, inside=0.02, value_units="mm^-1")
        >>> volume.values.tolist()
        [[[0.0, 0.02, 0.0]]]
        >>> (volume.values.dtype.name, volume.values.flags.writeable)
        ('float64', False)

    """
    np = _optional("numpy")
    radius = finite_scalar(radius_mm, "radius_mm", minimum=0.0)
    if radius <= 0:
        raise ContractError("radius_mm must be positive")
    centre = np.array([finite_scalar(value, "centre_mm") for value in centre_mm])
    if centre.shape != (3,):
        raise ContractError("centre_mm must have three coordinates")
    inside_value = finite_scalar(inside, "inside")
    outside_value = finite_scalar(outside, "outside")
    nz, ny, nx = grid.shape
    rotation = np.array(grid.orientation).reshape(3, 3)
    local_centre = rotation.T @ (centre - np.array(grid.origin_mm))
    # Orthogonal grids allow separable axes; hypot avoids squaring the length scale.
    x = np.arange(nx) * grid.spacing_mm[0] - local_centre[0]
    y = np.arange(ny) * grid.spacing_mm[1] - local_centre[1]
    z = np.arange(nz) * grid.spacing_mm[2] - local_centre[2]
    values = np.where(
        np.hypot(z[:, None, None], np.hypot(y[None, :, None], x[None, None, :])) <= radius,
        inside_value,
        outside_value,
    )
    return ImagingVolume(
        values,
        grid,
        ImagingMetadata("synthetic", "synthetic", value_units, source_shape=grid.shape),
    )
