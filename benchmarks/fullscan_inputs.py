#!/usr/bin/env python3
"""Prepare the complete CTChest at 256³ and one shared 360-view acquisition.

Run with the imaging extra plus SimpleITK installed. Array caches belong in a
user-selected data directory outside Git. This is a derived, linearly resampled
full chest, not a native 256³ scan or a cropped region. No CUDA work is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dpt.geometry import BatchedGeometry, ConeBeamIntrinsics, DetectorGeometry
from dpt.imaging import ct_to_attenuation, read_volume
from dpt.imaging_io import ImagingMetadata, ImagingVolume
from dpt.volumes import GridSpec

CHEST_SHA256 = "4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e"
CHEST_URL = "https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/" + CHEST_SHA256
DERIVED_SHAPE = (256, 256, 256)
DETECTOR_SHAPE = (512, 512)
VIEWS = 360
PIXEL_PITCH_MM = 2.6375731636032262
SOD_MM = 600.0
SDD_MM = 1200.0
SAMPLES_PER_RAY = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def affine(grid: GridSpec) -> np.ndarray:
    """Return voxel-centre xyz -> RAS mm, retaining all orientation columns."""
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(grid.orientation).reshape(3, 3) @ np.diag(grid.spacing_mm)
    result[:3, 3] = grid.origin_mm
    return result


def face_corners(grid: GridSpec) -> np.ndarray:
    xyz = grid.shape[::-1]
    return np.asarray(
        [grid.grid_to_object(p) for p in itertools.product(*[(-0.5, n - 0.5) for n in xyz])]
    )


def centre(grid: GridSpec) -> np.ndarray:
    return np.asarray(grid.grid_to_object(tuple((n - 1) / 2 for n in grid.shape[::-1])))


def resampled_grid(grid: GridSpec, shape: tuple[int, int, int]) -> GridSpec:
    """Preserve voxel faces and orientation; reposition the new first centre."""
    if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("shape must contain three positive integer z,y,x counts")
    old_spacing = np.asarray(grid.spacing_mm)
    spacing = old_spacing * np.asarray(grid.shape[::-1]) / np.asarray(shape[::-1])
    orientation = np.asarray(grid.orientation).reshape(3, 3)
    origin = np.asarray(grid.origin_mm) + orientation @ ((spacing - old_spacing) / 2)
    return GridSpec(shape, tuple(spacing), tuple(origin), grid.orientation)


def resample_hu(volume: ImagingVolume, shape=DERIVED_SHAPE) -> ImagingVolume:
    """Linearly interpolate HU in FP64, clamping samples at the source edges.

    New sample i maps to old continuous index (i+0.5)*n_old/n_new-0.5.
    Both grids share every voxel-face corner; centres need not coincide.
    No smoothing, cropping, padding, HU clipping or attenuation conversion is
    applied here. The edge rule matches DPT's constant extension to voxel faces.
    """
    import SimpleITK as sitk  # noqa: N813

    if volume.metadata.coordinate_frame != "RAS" or volume.metadata.value_units != "HU":
        raise ValueError("resampling requires declared HU values in RAS millimetres")
    grid = resampled_grid(volume.grid, shape)
    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    image = sitk.GetImageFromArray(np.asarray(volume.values, dtype=np.float64))
    image.SetSpacing(volume.grid.spacing_mm)
    image.SetOrigin(tuple(ras_to_lps @ volume.grid.origin_mm))
    image.SetDirection(tuple((ras_to_lps @ np.asarray(grid.orientation).reshape(3, 3)).ravel()))
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(shape[::-1])
    resampler.SetOutputSpacing(grid.spacing_mm)
    resampler.SetOutputOrigin(tuple(ras_to_lps @ grid.origin_mm))
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputPixelType(sitk.sitkFloat64)
    resampler.SetUseNearestNeighborExtrapolator(True)
    resampler.SetNumberOfThreads(1)
    resampler.SetNumberOfWorkUnits(1)
    values = sitk.GetArrayFromImage(resampler.Execute(image))
    return ImagingVolume(values, grid, volume.metadata)


def circular_geometry(grid: GridSpec) -> tuple[BatchedGeometry, np.ndarray]:
    """Full world-Z orbit; zero view source is centre-(0,600,0), rows are +Z."""
    angles = 2.0 * np.pi * np.arange(VIEWS, dtype=np.float64) / VIEWS
    intrinsics = ConeBeamIntrinsics(
        SOD_MM,
        SDD_MM,
        DETECTOR_SHAPE,
        (PIXEL_PITCH_MM, PIXEL_PITCH_MM),
        centre_mm=tuple(centre(grid)),
    )
    return intrinsics.batch(tuple(angles)), angles


def local_vectors(geometry: BatchedGeometry, grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """ASTRA [source, panel centre, pixel-column, pixel-row] vectors in mm.

    The local frame is centred on the volume with axes given by its orientation.
    It uses physical millimetres, not voxel indices. Do not rotate/reindex values.
    """
    world = np.asarray(geometry.to_astra_vectors().vectors, dtype=np.float64)
    rotation = np.asarray(grid.orientation).reshape(3, 3)
    local = world.copy()
    for start in (0, 3):
        local[:, start : start + 3] = (world[:, start : start + 3] - centre(grid)) @ rotation
    for start in (6, 9):
        local[:, start : start + 3] = world[:, start : start + 3] @ rotation
    return world, local


def detector_coverage(grid: GridSpec, geometry: BatchedGeometry) -> dict:
    """Project all eight support corners; check the full perspective panel."""
    corners = face_corners(grid)
    maximum = np.zeros(2)
    minimum_depth = np.inf
    for panel in geometry.views:
        source = np.asarray(panel.source_mm)
        u, v = np.asarray(panel.u), np.asarray(panel.v)
        panel_centre = (
            np.asarray(panel.origin_mm)
            + (
                (panel.shape[1] - 1) * panel.spacing_mm[0] * u
                + (panel.shape[0] - 1) * panel.spacing_mm[1] * v
            )
            / 2
        )
        normal = (panel_centre - source) / np.linalg.norm(panel_centre - source)
        depth = (corners - source) @ normal
        if np.any(depth <= 0):
            raise ValueError("volume support reaches or crosses a source plane")
        projected = (
            source + (corners - source) * (np.linalg.norm(panel_centre - source) / depth)[:, None]
        )
        offsets = np.column_stack(((projected - panel_centre) @ u, (projected - panel_centre) @ v))
        maximum = np.maximum(maximum, np.max(np.abs(offsets), axis=0))
        minimum_depth = min(minimum_depth, float(depth.min()))
        panel_half = np.asarray(panel.shape[::-1]) * np.asarray(panel.spacing_mm) / 2
        if np.any(np.abs(offsets) > panel_half + 1e-10):
            raise ValueError("detector does not cover the full physical volume")
    return {
        "all_eight_corners_inside_every_detector": True,
        "max_abs_projected_column_row_mm": maximum.tolist(),
        "minimum_source_plane_depth_mm": minimum_depth,
        "minimum_square_pitch_for_coverage_mm": float(2 * maximum.max() / DETECTOR_SHAPE[0]),
        "frozen_pitch_mm": PIXEL_PITCH_MM,
        "coverage_margin_factor": 1.05,
    }


def array_record(path: Path, values: np.ndarray, units: str) -> dict:
    return {
        "file": path.name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "units": units,
        "sha256": sha256(path),
        "raw_sha256": hashlib.sha256(
            memoryview(np.ascontiguousarray(values)).cast("B")
        ).hexdigest(),
        "bytes": path.stat().st_size,
    }


def verify_source(nifti: Path, nrrd: Path, volume: ImagingVolume) -> dict:
    """Bind the converted NIfTI to the checksum-pinned original CTChest."""
    import SimpleITK as sitk  # noqa: N813

    if sha256(nrrd) != CHEST_SHA256:
        raise ValueError("source NRRD does not match the pinned CTChest SHA256")
    original = sitk.ReadImage(str(nrrd))
    original_values = sitk.GetArrayFromImage(original)
    if volume.grid.shape != (139, 512, 512) or not np.array_equal(original_values, volume.values):
        raise ValueError("NIfTI values/dimensions do not match the complete source CTChest")
    source_affine = np.eye(4)
    source_affine[:3, :3] = np.asarray(original.GetDirection()).reshape(3, 3) @ np.diag(
        original.GetSpacing()
    )
    source_affine[:3, 3] = original.GetOrigin()
    source_affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ source_affine
    indices = np.asarray(list(itertools.product(*[(-0.5, n - 0.5) for n in original.GetSize()])))
    source_corners = np.column_stack((indices, np.ones(8))) @ source_affine.T
    corner_error = float(np.max(np.abs(source_corners[:, :3] - face_corners(volume.grid))))
    if corner_error > 1e-4:
        raise ValueError("NIfTI physical transform differs from the pinned source")
    return {
        "image": "Slicer CTChest",
        "nrrd_url": CHEST_URL,
        "nrrd_sha256": CHEST_SHA256,
        "nifti_file": nifti.name,
        "nifti_sha256": sha256(nifti),
        "grid": asdict(volume.grid),
        "affine_ras_mm": affine(volume.grid).tolist(),
        "shape_xyz": list(volume.grid.shape[::-1]),
        "shape_zyx": list(volume.grid.shape),
        "storage_dtype": str(volume.values.dtype),
        "value_range_hu": [float(volume.values.min()), float(volume.values.max())],
        "nrrd_nifti_values_equal": True,
        "nrrd_nifti_corner_max_abs_error_mm": corner_error,
    }


def prepare(ct: Path, source_nrrd: Path, output: Path) -> dict:
    """Prepare immutable-input files in a new directory; never overwrite a run."""
    output.mkdir(parents=True, exist_ok=False)
    volume = read_volume(ct, units="HU")
    source = verify_source(ct, source_nrrd, volume)
    hu = resample_hu(volume)
    attenuation = ct_to_attenuation(hu, energy_kev=60.0)
    geometry, angles = circular_geometry(hu.grid)
    world, local = local_vectors(geometry, hu.grid)
    corner_error = float(np.max(np.abs(face_corners(volume.grid) - face_corners(hu.grid))))
    if corner_error > 1e-10:
        raise ArithmeticError("resampling changed voxel-face extent")
    arrays = {}
    for name, values, units in (
        ("hu", hu.values, "HU"),
        ("attenuation", attenuation.values, "mm^-1"),
        ("angles_rad", angles, "rad"),
        ("vectors_ras", world, "mm"),
        ("vectors_local", local, "mm"),
    ):
        path = output / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        arrays[name] = array_record(path, values, units)
    result = {
        "schema_version": 1,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "source": source,
        "grid": asdict(hu.grid),
        "affine_ras_mm": affine(hu.grid).tolist(),
        "centre_ras_mm": centre(hu.grid).tolist(),
        "face_corners_ras_mm": face_corners(hu.grid).tolist(),
        "derived_volume": {
            "description": "complete chest linearly resampled to 256^3; no crop or padding",
            "shape_zyx": list(DERIVED_SHAPE),
            "hu_dtype": "float64",
            "attenuation_dtype": "float32",
            "value_range_hu": [float(hu.values.min()), float(hu.values.max())],
            "resampling": "FP64 SimpleITK linear, single work unit, source-edge clamping",
            "index_map": "old_xyz=(new_xyz+0.5)*old_shape_xyz/new_shape_xyz-0.5",
            "affine_rule": (
                "spacing_new=spacing_old*n_old/n_new; origin_new=origin_old+R*(s_new-s_old)/2"
            ),
            "face_corner_max_abs_error_mm": corner_error,
            "attenuation_model": "max(1+resampled_HU/1000,0)*0.02059 mm^-1",
            "attenuation_arithmetic": (
                "FP64 division/addition/clipping/multiplication; one FP32 cast"
            ),
            "energy_kev": 60.0,
            "water_density_g_cm3": 1.0,
            "mu_water_mm_inverse": 0.02059,
            "model_reference": "https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html",
        },
        "geometry": [asdict(panel) for panel in geometry.views],
        "acquisition": {
            "views": VIEWS,
            "detector_shape": list(DETECTOR_SHAPE),
            "source_object_distance_mm": SOD_MM,
            "source_detector_distance_mm": SDD_MM,
            "object_detector_distance_mm": SDD_MM - SOD_MM,
            "pitch_column_row_mm": [PIXEL_PITCH_MM, PIXEL_PITCH_MM],
            "angles": "theta_k=2*pi*k/360, k=0..359, no duplicate endpoint",
            "world_frame": "RAS mm; rotation around world +Z through physical volume centre",
            "zero_view": "source centre-(0,600,0); panel centre+(0,600,0); columns +X; rows +Z",
            "vector_layout": "source_xyz,detector_centre_xyz,pixel_column_xyz,pixel_row_xyz",
            "local_frame": "R.T@(world_position-volume_centre), physical mm; axes R.T@world_axes",
            "pose": "identity; fixed acquisition shared by all comparison backends",
            "coverage": detector_coverage(hu.grid, geometry),
        },
        "sampling": {
            "samples_per_ray": SAMPLES_PER_RAY,
            "rule": "1024 uniform midpoint samples on each ray clipped to voxel-face support",
            "interpolation": "trilinear with boundary sample clamping inside voxel-face support",
            "output": "dimensionless optical depth",
            "precision_modes": {
                "fp64": "FP32 stored field; FP64 geometry/interpolation/accumulation/output",
                "fp32": "FP32 stored field and local sample/interpolation/accumulation/output",
            },
        },
        "arrays": arrays,
        "environment": {
            "python": platform.python_version(),
            "versions": {
                name: importlib.metadata.version(name) for name in ("numpy", "SimpleITK", "nibabel")
            },
        },
        "preparation_source_sha256": sha256(Path(__file__)),
    }
    (output / "metadata.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


@dataclass(frozen=True)
class PreparedScan:
    volume: ImagingVolume
    geometry: BatchedGeometry
    metadata: dict


def load_prepared(directory: Path, *, verify_hashes: bool = True) -> PreparedScan:
    """Load saved input without regenerating geometry or resampling the CT."""
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    arrays = {}
    for name, record in metadata["arrays"].items():
        path = directory / record["file"]
        if verify_hashes and sha256(path) != record["sha256"]:
            raise ValueError(f"prepared {name} array SHA256 mismatch")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != record["shape"] or str(values.dtype) != record["dtype"]:
            raise ValueError(f"prepared {name} array shape/dtype mismatch")
        arrays[name] = values
    g = metadata["grid"]
    grid = GridSpec(
        tuple(g["shape"]), tuple(g["spacing_mm"]), tuple(g["origin_mm"]), tuple(g["orientation"])
    )
    volume = ImagingVolume(
        arrays["attenuation"],
        grid,
        ImagingMetadata("nifti", "RAS", "mm^-1", (metadata["source"]["nifti_sha256"],)),
    )
    geometry = BatchedGeometry(
        tuple(
            DetectorGeometry(**{key: tuple(value) for key, value in panel.items()})
            for panel in metadata["geometry"]
        )
    )
    world, local = local_vectors(geometry, grid)
    if not np.array_equal(world, arrays["vectors_ras"]) or not np.array_equal(
        local, arrays["vectors_local"]
    ):
        raise ValueError("saved vector arrays do not match the saved physical geometry")
    return PreparedScan(volume, geometry, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True, help="Converted pinned CTChest NIfTI")
    parser.add_argument(
        "--source-nrrd", type=Path, required=True, help="Pinned original CTChest NRRD"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New external data/cache directory"
    )
    args = parser.parse_args()
    result = prepare(args.ct, args.source_nrrd, args.output)
    print(
        json.dumps(
            {"output": str(args.output), "grid": result["grid"], "arrays": result["arrays"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
