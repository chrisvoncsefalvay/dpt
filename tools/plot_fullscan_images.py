"""Draw real full-scan outputs without running a projector or changing their values.

Inputs are prepared full-chest data and two completed benchmarks/fullscan.py runs.
The 512 x 512 x 139 source was resampled to 256 cubed over its full physical extent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from plot_baseline import (
    add_ct_slice,
    configure_style,
    label_detector_axis,
    physical_detector_extent,
    save_figure,
    sha256,
)

VIEWS = (0, 45, 90)
NAMES = ("PA", "Oblique", "Lateral")


def read_json(path):
    return json.loads(path.read_text())


def checked_array(path, record, shape, dtype):
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Array checksum differs from recorded result: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        values.shape != tuple(shape)
        or values.dtype != np.dtype(dtype)
        or list(values.shape) != record["shape"]
        or str(values.dtype) != record["dtype"]
    ):
        raise ValueError(f"Array shape/dtype differs from the full-scan inputs: {path}")
    raw = hashlib.sha256(memoryview(values).cast("B")).hexdigest()
    if raw != record["raw_sha256"]:
        raise ValueError(f"Array values differ from recorded result: {path}")
    for plane in values:
        if not np.isfinite(plane).all():
            raise ValueError(f"Nonfinite values in {path}")
    return values


def load_run(directory, mode, metadata, metadata_hash):
    results_path, outputs_path = directory / "results.json", directory / "outputs.json"
    results, outputs = read_json(results_path), read_json(outputs_path)
    if not (
        results.get("status") == "complete"
        and results.get("source_unchanged") is True
        and results.get("completed_at")
        and not results.get("error")
        and not results.get("failure")
    ):
        raise ValueError(f"Only completed full-scan runs may supply figures: {directory}")
    if not (
        results["precision"] == outputs["precision"] == mode
        and results["source"] == results["source_after"]
        and results["source"]["sha256"] == outputs["source_sha256"]
        and results["input"]["metadata_sha256"] == outputs["input_metadata_sha256"] == metadata_hash
        and results["input"]["metadata"] == metadata
        and results["input"]["selected_view_indices"] == list(range(360))
        and results["input"]["selected_geometry"] == metadata["geometry"]
        and outputs["views"] == 360
        and outputs["cotangent_sha256"] == results["input"]["cotangent"]["sha256"]
    ):
        raise ValueError(f"Run source/input/output bindings differ: {directory}")
    image = outputs["arrays"]["images"]
    if image["units"] != "dimensionless optical depth":
        raise ValueError("Image quantity is not optical depth")
    values = checked_array(
        directory / image["file"], image, (360, 512, 512), mode.replace("fp", "float")
    )
    binding = {
        "results_sha256": sha256(results_path),
        "outputs_sha256": sha256(outputs_path),
        "source_sha256": results["source"]["sha256"],
        "started_at_utc": results["started_at"],
        "completed_at_utc": results["completed_at"],
        "precision": mode,
        "images": image,
        "cotangent_sha256": outputs["cotangent_sha256"],
        "contract_sha256": results["contract"]["sha256"],
    }
    return values, binding, outputs


def difference_metrics(reference, actual, indices):
    """Accumulate error over actual pixels in FP64 without a full-scan temporary."""
    count, error_sum, reference_sum, largest = 0, 0.0, 0.0, 0.0
    for index in indices:
        ref = np.asarray(reference[index], dtype=np.float64)
        difference = np.asarray(actual[index], dtype=np.float64) - ref
        largest = max(largest, float(np.max(np.abs(difference))))
        error_sum += float(np.sum(difference * difference, dtype=np.float64))
        reference_sum += float(np.sum(ref * ref, dtype=np.float64))
        count += ref.size
    if not count or not reference_sum:
        raise ValueError("Error comparison requires pixels and a nonzero reference")
    return {
        "pixels": count,
        "max_abs": largest,
        "rmse": float(np.sqrt(error_sum / count)),
        "relative_l2": float(np.sqrt(error_sum / reference_sum)),
        "units": "dimensionless optical depth",
    }


def caption(figure, text):
    figure.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=9, color="black")


def detector(axis, values, view, title, limits, cmap="gray"):
    artist = axis.imshow(
        values,
        origin="lower",
        extent=physical_detector_extent(view),
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
        interpolation="nearest",
        rasterized=True,
    )
    axis.set_title(title, loc="left")
    label_detector_axis(axis, view)
    return artist


def overview(ct, reference, metadata, output, slice_index, window, depth_limit):
    figure, axes = plt.subplots(1, 4, figsize=(16.8, 5.6), layout="constrained")
    figure.get_layout_engine().set(rect=(0, 0.10, 1, 0.88))
    ct_artist, _ = add_ct_slice(axes[0], ct, slice_index, window)
    axes[0].set_title("Full chest resampled to 256 cubed", loc="left")
    figure.colorbar(ct_artist, ax=axes[0], location="bottom", label="CT value (HU)")
    for axis, index, name in zip(axes[1:], VIEWS, NAMES, strict=True):
        artist = detector(
            axis,
            reference[index],
            metadata["geometry"][index],
            f"{name} · {index}° · FP64",
            (0, depth_limit),
        )
    figure.colorbar(
        artist, ax=list(axes[1:]), location="bottom", label="Optical depth (dimensionless)"
    )
    z = float(ct.affine[2, 3] + slice_index * ct.affine[2, 2])
    caption(
        figure,
        f"Computed DRRs from the full 512 × 512 × 139 CT, linearly resampled to 256³. "  # noqa: RUF001
        f"Axial slice z = {z:.2f} mm.\n"
        "Full physical extent retained. 360 views, 512² detector and 1024 clipped-ray midpoints. "
        "The 0° source is posterior in RAS.",
    )
    save_figure(figure, output, "fullscan_ct_projections")


def error_figure(reference, actual, metadata, output, limit):
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 5.7), layout="constrained")
    figure.get_layout_engine().set(rect=(0, 0.09, 1, 0.90))
    # A zero-only result remains white at zero. Record the display fallback separately.
    display_limit = limit if limit else np.finfo(np.float64).eps
    for axis, index, name in zip(axes, VIEWS, NAMES, strict=True):
        artist = detector(
            axis,
            actual[index].astype(np.float64) - reference[index],
            metadata["geometry"][index],
            f"{name} · {index}° · FP32 minus FP64",
            (-display_limit, display_limit),
            "RdBu_r",
        )
    bar = figure.colorbar(
        artist,
        ax=list(axes),
        location="bottom",
        label="Signed optical-depth difference (dimensionless)",
    )
    bar.formatter.set_powerlimits((-3, 3))
    bar.update_ticks()
    caption(
        figure,
        "Identical physical rays, FP32 stored attenuation and 1024 midpoint samples. "
        "One symmetric scale spans the three displayed views.",
    )
    save_figure(figure, output, "fullscan_fp32_fp64_error")
    return display_limit


def sensitivity_figure(ct, gradient, mode, output, slice_index, window):
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 5.7), layout="constrained")
    figure.get_layout_engine().set(rect=(0, 0.10, 1, 0.89))
    ct_artist, _ = add_ct_slice(axes[0], ct, slice_index, window)
    figure.colorbar(ct_artist, ax=axes[0], location="bottom", label="CT value (HU)")
    plane = gradient[slice_index]
    limit = float(np.max(np.abs(plane)))
    display_limit = limit if limit else np.finfo(np.float32).eps
    # Reuse the same affine-aware axial extent and radiological display orientation.
    image = nib.Nifti1Image(np.asarray(gradient).transpose(2, 1, 0), ct.affine)
    artist, _ = add_ct_slice(axes[1], image, slice_index, (-display_limit, display_limit))
    artist.set_cmap("RdBu_r")
    axes[1].set_title(f"Whole-scan sensitivity · {mode.upper()}", loc="left")
    bar = figure.colorbar(artist, ax=axes[1], location="bottom", label="∂ objective / ∂ μ (mm)")
    bar.formatter.set_powerlimits((-3, 3))
    bar.update_ticks()
    caption(
        figure,
        "Saved volume VJP for the shared dense, positive, asymmetric detector cotangent "
        "over all 360 views.\nFP32 gradient storage. Values are displayed without normalisation.",
    )
    save_figure(figure, output, "fullscan_volume_sensitivity")
    return {"slice_max_abs": limit, "display_half_range": display_limit, "units": "mm"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared chest-256 directory")
    parser.add_argument("--fp64-run", type=Path, required=True)
    parser.add_argument("--fp32-run", type=Path, required=True)
    parser.add_argument("--volume-vjp", type=Path, help="Saved VJP from either supplied run")
    parser.add_argument("--output", type=Path, required=True, help="New figure directory")
    parser.add_argument("--slice-index", type=int, default=128)
    parser.add_argument("--ct-window", type=float, nargs=2, default=(-1000.0, 1000.0))
    parser.add_argument(
        "--font-directory", type=Path, default=Path.home() / ".cache/dpt/figure-fonts"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists. Select a new path")
    if not 0 <= args.slice_index < 256 or not args.ct_window[0] < args.ct_window[1]:
        parser.error("Need a slice in [0,255] and an increasing CT window")
    metadata_path = args.data / "metadata.json"
    metadata, metadata_hash = read_json(metadata_path), sha256(metadata_path)
    if metadata["grid"]["shape"] != [256, 256, 256] or len(metadata["geometry"]) != 360:
        raise ValueError("This figure requires the full chest resampled to 256 cubed / 360 views")
    hu_record = metadata["arrays"]["hu"]
    hu = checked_array(args.data / hu_record["file"], hu_record, (256, 256, 256), "float64")
    ct = nib.Nifti1Image(hu.transpose(2, 1, 0), np.asarray(metadata["affine_ras_mm"]))
    arrays, bindings, outputs = {}, {}, {}
    for mode in ("fp64", "fp32"):
        arrays[mode], bindings[mode], outputs[mode] = load_run(
            getattr(args, mode + "_run"), mode, metadata, metadata_hash
        )
    if bindings["fp64"]["cotangent_sha256"] != bindings["fp32"]["cotangent_sha256"]:
        raise ValueError("Precision runs used different detector objectives")
    if bindings["fp64"]["contract_sha256"] != bindings["fp32"]["contract_sha256"]:
        raise ValueError("Precision runs used different frozen contracts")
    metrics = {
        "full_360_views": difference_metrics(arrays["fp64"], arrays["fp32"], range(360)),
        "displayed_views": difference_metrics(arrays["fp64"], arrays["fp32"], VIEWS),
    }
    gradient, gradient_binding = None, None
    if args.volume_vjp:
        actual_hash = sha256(args.volume_vjp)
        matches = [
            mode
            for mode in ("fp64", "fp32")
            if outputs[mode]["arrays"].get("volume_vjp", {}).get("sha256") == actual_hash
        ]
        if not matches:
            raise ValueError("Optional VJP is not an output of either supplied full-scan run")
        gradient_mode = matches[0]
        record = outputs[gradient_mode]["arrays"]["volume_vjp"]
        gradient = checked_array(args.volume_vjp, record, (256, 256, 256), "float32")
        gradient_binding = {"precision": gradient_mode, "array": record}
    family = configure_style(args.font_directory)
    args.output.mkdir(parents=True, exist_ok=False)
    depth_limit = max(float(arrays[mode][list(VIEWS)].max()) for mode in ("fp64", "fp32"))
    overview(
        ct, arrays["fp64"], metadata, args.output, args.slice_index, args.ct_window, depth_limit
    )
    error_limit = error_figure(
        arrays["fp64"], arrays["fp32"], metadata, args.output, metrics["displayed_views"]["max_abs"]
    )
    if gradient is not None:
        gradient_binding["display"] = sensitivity_figure(
            ct, gradient, gradient_mode, args.output, args.slice_index, args.ct_window
        )
    figure_files = {
        path.name: sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()
    }
    sidecar = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "generator_sha256": sha256(Path(__file__)),
        "plot_helpers_sha256": sha256(Path(__file__).with_name("plot_baseline.py")),
        "input_metadata_sha256": metadata_hash,
        "source_ct_sha256": metadata["source"]["nifti_sha256"],
        "native_shape_zyx": metadata["source"]["shape_zyx"],
        "derived_shape_zyx": metadata["grid"]["shape"],
        "hu": hu_record,
        "runs": bindings,
        "view_indices": list(VIEWS),
        "view_names": list(NAMES),
        "geometry": [metadata["geometry"][index] for index in VIEWS],
        "grid": metadata["grid"],
        "affine_ras_mm": metadata["affine_ras_mm"],
        "slice_index": args.slice_index,
        "ct_display_window_hu": args.ct_window,
        "shared_optical_depth_scale": [0, depth_limit],
        "shared_signed_error_scale": [-error_limit, error_limit],
        "error_metrics": metrics,
        "sensitivity": gradient_binding,
        "font": family,
        "font_sha256": sha256(args.font_directory / "Geist-Regular.ttf"),
        "figures": figure_files,
    }
    (args.output / "figure-data.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"Saved {len(figure_files)} figure files and figure-data.json to {args.output}")


if __name__ == "__main__":
    main()
