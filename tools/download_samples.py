#!/usr/bin/env python3
"""Fetch checksum-pinned Slicer CT samples and preserve their physical geometry.

Dependencies: SimpleITK, nibabel, numpy. Data are stored only in the requested
directory, never alongside this script by default. Samples retain their source
terms; DPT's software licence does not relicense downloaded image data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

SAMPLES = {
    "chest": {
        "name": "CTChest",
        "sha256": "4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e",
        "expected_bytes": 42189103,
        "source": "https://github.com/Slicer/SlicerTestingData",
        "acknowledgements": "https://www.slicer.org/wiki/Documentation/4.10/Modules/SampleData",
    },
    "brain": {
        "name": "CTBrain",
        "sha256": "6a5b6caccb76576a863beb095e3bfb910c50ca78f4c9bf043aa42f976cfa53d1",
        "expected_bytes": 16366792,
        "source": "https://github.com/Slicer/SlicerTestingData",
        "acknowledgements": "https://www.slicer.org/wiki/Documentation/4.10/Modules/SampleData",
    },
}
DOWNLOAD_BASE = "https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/"
# NIfTI-1 stores affine coefficients as binary32; compare physical positions in mm.
GEOMETRY_TOLERANCE_MM = 1e-4


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_sample(name: str, directory: Path) -> tuple[Path, bool]:
    spec = SAMPLES[name]
    destination = directory / f"{spec['name']}.nrrd"
    if destination.exists():
        if sha256(destination) != spec["sha256"]:
            raise ValueError(f"Existing file has the wrong SHA256: {destination}")
        return destination, False
    url = DOWNLOAD_BASE + str(spec["sha256"])
    request = Request(url, headers={"User-Agent": "DPT-sample-downloader/1"})
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as out:
            temporary = Path(out.name)
            digest = hashlib.sha256()
            count = 0
            with urlopen(request, timeout=120) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(chunk)
                    out.write(chunk)
                    count += len(chunk)
        if digest.hexdigest() != spec["sha256"]:
            raise ValueError(f"SHA256 mismatch for {name}: {digest.hexdigest()}")
        if count != spec["expected_bytes"]:
            raise ValueError(f"Unexpected byte count for {name}: {count}")
        os.replace(temporary, destination)
        return destination, True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def convert_and_verify(source: Path, destination: Path) -> dict:
    import nibabel as nib
    import numpy as np
    import SimpleITK as sitk  # noqa: N813

    original = sitk.ReadImage(str(source))
    if original.GetDimension() != 3 or original.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("The sample must contain one three-dimensional scalar image")
    source_values = sitk.GetArrayFromImage(original)
    if not np.isfinite(source_values).all():
        raise ValueError("The source contains nonfinite values")
    converted_at = None
    if not destination.exists():
        # SimpleITK's NIfTI writer converts its LPS physical frame to NIfTI RAS.
        temporary = destination.with_name(destination.name.removesuffix(".nii.gz") + ".tmp.nii.gz")
        try:
            sitk.WriteImage(original, str(temporary), useCompression=True)
            os.replace(temporary, destination)
            converted_at = timestamp()
        finally:
            temporary.unlink(missing_ok=True)

    reloaded = sitk.ReadImage(str(destination))
    roundtrip_values = sitk.GetArrayFromImage(reloaded)
    nifti = nib.load(destination)
    nibabel_values = np.asanyarray(nifti.dataobj).transpose(2, 1, 0)
    values_equal = bool(np.array_equal(source_values, roundtrip_values))
    nifti_values_equal = bool(np.array_equal(source_values, nibabel_values))
    if not values_equal or not nifti_values_equal:
        raise ValueError("Voxel values changed during NRRD to NIfTI conversion")
    if source_values.dtype != roundtrip_values.dtype or source_values.dtype != nibabel_values.dtype:
        raise ValueError("Voxel storage dtype changed during NRRD to NIfTI conversion")

    spacing = np.asarray(original.GetSpacing(), dtype=np.float64)
    direction = np.asarray(original.GetDirection(), dtype=np.float64).reshape(3, 3)
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = original.GetOrigin()
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    expected_ras = lps_to_ras @ affine_lps
    size = tuple(int(x) for x in original.GetSize())
    centre_corners = np.array(
        list(itertools.product(*[(0, x - 1) for x in size])), dtype=np.float64
    )
    support_corners = np.array(
        list(itertools.product(*[(-0.5, x - 0.5) for x in size])), dtype=np.float64
    )

    def corner_report(indices):
        homogeneous = np.column_stack([indices, np.ones(len(indices))])
        source_ras = (expected_ras @ homogeneous.T).T[:, :3]
        stored_ras = (nifti.affine @ homogeneous.T).T[:, :3]
        difference = np.linalg.norm(source_ras - stored_ras, axis=1)
        return {
            "index_xyz": indices.tolist(),
            "source_ras_mm": source_ras.tolist(),
            "nifti_ras_mm": stored_ras.tolist(),
            "maximum_position_error_mm": float(difference.max()),
        }

    centres = corner_report(centre_corners)
    support = corner_report(support_corners)
    error = max(centres["maximum_position_error_mm"], support["maximum_position_error_mm"])
    if error > GEOMETRY_TOLERANCE_MM:
        raise ValueError(f"Physical corner error {error} exceeds {GEOMETRY_TOLERANCE_MM} mm")
    return {
        "source_file": str(source.resolve()),
        "nifti_file": str(destination.resolve()),
        "nifti_sha256": sha256(destination),
        "nifti_bytes": destination.stat().st_size,
        "converted_at_utc": converted_at,
        "verified_at_utc": timestamp(),
        "units": "HU",
        "source_dimensions_xyz": list(size),
        "array_dimensions_zyx": list(source_values.shape),
        "storage_dtype": str(source_values.dtype),
        "value_min": float(source_values.min()),
        "value_max": float(source_values.max()),
        "spacing_xyz_mm": spacing.tolist(),
        "origin_lps_mm": list(original.GetOrigin()),
        "direction_lps": direction.tolist(),
        "source_affine_lps_mm": affine_lps.tolist(),
        "expected_affine_ras_mm": expected_ras.tolist(),
        "nifti_affine_ras_mm": nifti.affine.tolist(),
        "nifti_spatial_units": nifti.header.get_xyzt_units()[0],
        "nifti_sform_code": int(nifti.header["sform_code"]),
        "nifti_qform_code": int(nifti.header["qform_code"]),
        "nifti_scaling_slope": float(nifti.dataobj.slope),
        "nifti_scaling_intercept": float(nifti.dataobj.inter),
        "checks": {
            "all_voxels_equal_after_simpleitk_roundtrip": values_equal,
            "all_voxels_equal_after_nibabel_scaling": nifti_values_equal,
            "storage_dtype_preserved": True,
            "geometry_tolerance_mm": GEOMETRY_TOLERANCE_MM,
            "voxel_centre_corners": centres,
            "voxel_support_corners": support,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "samples", choices=[*SAMPLES, "all"], nargs="+", help="Named real CT sample(s)"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="User-selected download/conversion cache"
    )
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    samples = list(SAMPLES) if "all" in args.samples else list(dict.fromkeys(args.samples))
    metadata_path = args.data_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        metadata = {"schema_version": 1, "samples": {}}
    metadata["verified_at_utc"] = timestamp()
    metadata["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("SimpleITK", "nibabel", "numpy")
        },
    }
    metadata["data_notice"] = (
        "Downloaded image data retain source terms; "
        "DPT's software licence does not apply to the data."
    )
    for name in samples:
        source, downloaded = download_sample(name, args.data_dir)
        cache_verified_at = timestamp()
        spec = SAMPLES[name]
        destination = args.data_dir / f"{spec['name']}.nii.gz"
        previous = metadata["samples"].get(name, {})
        entry = convert_and_verify(source, destination)
        entry.update(spec)
        entry["source_url"] = DOWNLOAD_BASE + str(spec["sha256"])
        entry["source_bytes"] = source.stat().st_size
        entry["source_sha256_verified"] = True
        entry["source_cache_verified_at_utc"] = cache_verified_at
        entry["downloaded_during_this_run"] = downloaded
        entry["converted_at_utc"] = entry["converted_at_utc"] or previous.get("converted_at_utc")
        metadata["samples"][name] = entry
        # Commit each sample's metadata so a later download failure loses no evidence.
        with tempfile.NamedTemporaryFile(
            mode="w", dir=args.data_dir, suffix=".json", delete=False
        ) as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
            temporary = Path(stream.name)
        os.replace(temporary, metadata_path)
        error = entry["checks"]["voxel_support_corners"]["maximum_position_error_mm"]
        print(
            f"{name}: {destination}; all voxel values preserved; "
            f"maximum support-corner error {error:.8g} mm",
            flush=True,
        )
    print(f"Metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
