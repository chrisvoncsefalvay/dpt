"""Real CT input, declared attenuation conversion and useful RAS view geometry."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from dpt.contracts import ContractError, finite_scalar
from dpt.geometry import BatchedGeometry, ConeBeamIntrinsics, Matrix3, Vector3
from dpt.imaging_io import ImagingVolume, PathInput, _optional, read_dicom_series, read_nifti
from dpt.volumes import GridSpec

__all__ = ["ct_to_attenuation", "read_volume", "standard_views"]


def _dicom_paths(directory: Path, series_uid: str | None) -> tuple[Path, ...]:
    """Select an actual image series without decoding its pixel payloads."""
    pydicom = _optional("pydicom")
    series: dict[str, list[Path]] = {}
    descriptions: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        try:
            header = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=[
                    "SeriesInstanceUID",
                    "SeriesDescription",
                    "Modality",
                    "Rows",
                    "Columns",
                ],
            )
        except pydicom.errors.InvalidDicomError:
            continue
        if not getattr(header, "Rows", 0) or not getattr(header, "Columns", 0):
            continue
        uid = str(getattr(header, "SeriesInstanceUID", ""))
        if not uid:
            raise ContractError(f"DICOM image has no SeriesInstanceUID: {path}")
        series.setdefault(uid, []).append(path)
        descriptions[uid] = ", ".join(
            str(value)
            for value in (getattr(header, "Modality", ""), getattr(header, "SeriesDescription", ""))
            if value
        )
    if not series:
        raise ContractError(f"no DICOM image series found in {directory}")
    choices = "\n".join(
        f"  {uid} ({len(paths)} files; {descriptions[uid] or 'no description'})"
        for uid, paths in sorted(series.items())
    )
    if series_uid is None:
        if len(series) != 1:
            raise ContractError(
                "DICOM folder contains multiple image series; provide series_uid from:\n" + choices
            )
        series_uid = next(iter(series))
    if series_uid not in series:
        raise ContractError(
            f"DICOM series_uid {series_uid!r} was not found; available series:\n{choices}"
        )
    return tuple(series[series_uid])


def read_volume(
    path: PathInput, *, units: str | None = None, series_uid: str | None = None
) -> ImagingVolume:
    """Load a NIfTI file or DICOM series directory in RAS millimetres.

    Values are immutable, contiguous ``(z,y,x)`` samples. NIfTI scaling and
    DICOM value mappings are applied once; ``units`` declares unknown value
    units and never converts values. DICOM's known mapping units must agree.
    Multiple DICOM image series require an actual ``series_uid`` reported by
    this function. DICOM LPS geometry is converted to RAS by changing the
    world axes, preserving the complete oblique transform and voxel values.
    """
    if units is not None and (not isinstance(units, str) or not units.strip()):
        raise ContractError("units must be a non-empty unit label")
    if series_uid is not None and (not isinstance(series_uid, str) or not series_uid.strip()):
        raise ContractError("series_uid must be a non-empty DICOM SeriesInstanceUID")
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        volume = read_dicom_series(_dicom_paths(source, series_uid), value_units=units)
        # LPS -> RAS acts on every world coordinate, including origin and all
        # three direction columns; storage indices and spacings do not change.
        signs = (-1.0, -1.0, 1.0)
        grid = GridSpec(
            volume.grid.shape,
            volume.grid.spacing_mm,
            cast(Vector3, tuple(signs[i] * volume.grid.origin_mm[i] for i in range(3))),
            cast(
                Matrix3,
                tuple(signs[i // 3] * value for i, value in enumerate(volume.grid.orientation)),
            ),
        )
        return ImagingVolume(volume.values, grid, replace(volume.metadata, coordinate_frame="RAS"))
    if series_uid is not None:
        raise ContractError("series_uid applies to a DICOM series directory")
    if not source.name.lower().endswith((".nii", ".nii.gz", ".hdr", ".hdr.gz", ".img", ".img.gz")):
        raise ContractError("read_volume requires a NIfTI file or a DICOM series directory")
    return read_nifti(source, value_units=units if units is not None else "unknown")


def ct_to_attenuation(volume: ImagingVolume, *, energy_kev: float) -> ImagingVolume:
    """Convert declared HU to water-scaled monoenergetic attenuation in mm^-1.

    Return FP32 ``max(1 + HU/1000, 0) * 0.02059`` on the same grid, leaving
    the HU volume unchanged. Evaluate the conversion in FP64 one slice at a
    time and round once to FP32 storage. The supported model is 60 keV water at
    1 g/cm³. NIST gives 0.2059 cm²/g at 0.06 MeV, hence 0.02059 mm^-1:
    https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html

    This is an explicitly chosen water-density model, not a material
    calibration inferred from CT values or scanner kVp. Other energies need
    an appropriate supplied calibration and are rejected here.
    """
    if not isinstance(volume, ImagingVolume):
        raise ContractError("volume must be an ImagingVolume")
    if volume.metadata.value_units != "HU":
        raise ContractError("ct_to_attenuation requires volume units='HU'")
    energy = finite_scalar(energy_kev, "energy_kev", minimum=0.0)
    if energy != 60.0:
        raise ContractError("the water-scaled HU model supports energy_kev=60.0 only")
    np = _optional("numpy")
    values = np.empty(volume.values.shape, dtype=np.float32, order="C")
    for index in range(values.shape[0]):
        converted = np.array(volume.values[index], dtype=np.float64, copy=True)
        converted /= 1000.0
        converted += 1.0
        np.maximum(converted, 0.0, out=converted)
        converted *= 0.02059
        values[index] = converted
    return ImagingVolume(
        values,
        volume.grid,
        replace(volume.metadata, value_units="mm^-1", measurement_unit_code=None),
    )


def standard_views(
    grid: GridSpec,
    names: Sequence[str] = ("ap", "lateral", "oblique"),
    *,
    shape: tuple[int, int] = (512, 512),
) -> BatchedGeometry:
    """Construct centred AP, lateral and oblique cone views for a RAS grid.

    Views share ``shape=(rows,columns)`` and cover the complete physical
    support, including oblique corners. AP places the source anterior (+Y);
    lateral uses +X and oblique lies between them. Detector rows increase
    towards superior (+Z). The returned panels expose source/detector
    positions and pixel pitch in millimetres for recording or calibration.

    Let D be the physical support diagonal. SOD is max(1000,2D), SDD is
    max(1500,3D), and panel width is 1.05*D*SDD/(SOD-D/2), all in mm.
    These are declared example acquisition settings, not scanner calibration.
    """
    if not isinstance(grid, GridSpec):
        raise ContractError("grid must be a GridSpec in RAS coordinates")
    if isinstance(names, str) or not names:
        raise ContractError("names must be a non-empty sequence of view names")
    angles = {"ap": math.pi, "lateral": math.pi / 2, "oblique": 3 * math.pi / 4}
    if any(name not in angles for name in names):
        raise ContractError("standard view names are ap, lateral and oblique")
    if len(shape) != 2 or any(type(count) is not int or count < 1 for count in shape):
        raise ContractError("shape must contain positive integer (rows,columns)")
    nz, ny, nx = grid.shape
    centre = grid.grid_to_object(((nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2))
    diagonal = math.hypot(
        *(count * pitch for count, pitch in zip((nx, ny, nz), grid.spacing_mm, strict=True))
    )
    sod, sdd = max(1000.0, 2.0 * diagonal), max(1500.0, 3.0 * diagonal)
    width = 1.05 * diagonal * sdd / (sod - diagonal / 2.0)
    if not math.isfinite(width):
        raise ContractError("grid extent exceeds the finite standard-view geometry range")
    intrinsics = ConeBeamIntrinsics(
        sod, sdd, shape, (width / shape[1], width / shape[0]), centre_mm=centre
    )
    return intrinsics.batch(tuple(angles[name] for name in names))
