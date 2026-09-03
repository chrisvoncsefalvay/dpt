# ruff: noqa: E501, RUF001
# Preserve scientific minus/multiplication glyphs and complete generated Markdown rows.
#!/usr/bin/env python3
"""Plot real CT/projector outputs and saved measurements from baseline_ct.py.

Example:
  python plot_baseline.py --results-dir baseline --ct CTChest.nii.gz --output figures

Requires numpy, nibabel and matplotlib. The script checks the CT checksum,
requires completed measurements, and reads numerical projections unchanged.
Geist Sans is obtained from Vercel's pinned official release if absent locally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter

MODES = ("float64", "mixed_float32", "fast_float32")
MODE_LABELS = {"float64": "Strict", "mixed_float32": "Mixed FP32", "fast_float32": "Fast FP32"}
FONT_URL = "https://github.com/vercel/geist-font/releases/download/1.4.2/geist-font-1.4.2.zip"
FONT_ARCHIVE_SHA256 = "e94caf733a6b019a0d4d97da9548c2ca8cbe4b2703ef1d07113a82d9e774cfe5"
FONT_ARCHIVE_PREFIX = "geist-font-1.4.2/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style(font_directory: Path) -> str:
    font_directory.mkdir(parents=True, exist_ok=True)
    archive_path = font_directory / "geist-font-1.4.2.zip"
    if not archive_path.exists():
        with urlopen(FONT_URL, timeout=120) as response:
            archive_bytes = response.read()
        if hashlib.sha256(archive_bytes).hexdigest() != FONT_ARCHIVE_SHA256:
            raise ValueError("The official Geist archive does not match the pinned SHA256")
        archive_path.write_bytes(archive_bytes)
    if sha256(archive_path) != FONT_ARCHIVE_SHA256:
        raise ValueError(f"Geist archive checksum mismatch: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        for weight in ("Regular", "Bold"):
            filename = f"Geist-{weight}.ttf"
            payload = archive.read(FONT_ARCHIVE_PREFIX + "fonts/Geist/ttf/" + filename)
            path = font_directory / filename
            if not path.exists() or path.read_bytes() != payload:
                path.write_bytes(payload)
            font_manager.fontManager.addfont(path)
        (font_directory / "OFL.txt").write_bytes(archive.read(FONT_ARCHIVE_PREFIX + "OFL.txt"))
    family = font_manager.FontProperties(fname=font_directory / "Geist-Regular.ttf").get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "axes.grid": False,
        }
    )
    return family


def load_inputs(
    directory: Path, ct_path: Path
) -> tuple[dict, dict[str, np.ndarray], nib.Nifti1Image]:
    results = json.loads((directory / "results.json").read_text())
    if not results.get("measurement_finished_at"):
        raise ValueError("Baseline measurements are incomplete. No final figure was generated")
    if sha256(ct_path) != results["input"]["sha256"]:
        raise ValueError("The supplied CT does not match the measured input SHA256")
    native_shape = tuple(results["input"]["native_shape_zyx"])
    ct = nib.load(ct_path)
    if tuple(reversed(ct.shape)) != native_shape:
        raise ValueError("The supplied CT dimensions do not match the measured native volume")
    views = results["input"]["geometry"]
    shapes = [tuple(view["shape"]) for view in views]
    if len(set(shapes)) != 1:
        raise ValueError("This figure requires equal detector dimensions across views")
    expected_shape = (len(views), *shapes[0])
    arrays = {}
    for mode in MODES:
        if mode not in results["modes"]:
            raise ValueError(f"Missing measured mode: {mode}")
        array = np.load(directory / f"{mode}_depth.npy", allow_pickle=False)
        if array.shape != expected_shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid projection data for {mode}: {array.shape}")
        arrays[mode] = array
    return results, arrays, ct


def measured_row(results: dict, mode: str, operation: str) -> dict:
    label = mode + "/" + operation
    matches = [entry for entry in results["measurements"] if entry["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one measurement for {label}, found {len(matches)}")
    row = matches[0]
    for boundary in ("device", "wall"):
        raw = [value for rd in row["rounds"] for value in rd[boundary + "_ms"]]
        if not raw or not np.isfinite(raw).all() or np.any(np.asarray(raw) < 0):
            raise ValueError(f"Missing/invalid raw timing samples for {label}")
        for metric, computed in (("median", np.median(raw)), ("p95", np.percentile(raw, 95))):
            recorded = row[f"{boundary}_{metric}_ms"]
            if not np.isclose(recorded, computed, rtol=1e-10, atol=1e-10):
                raise ValueError(f"Summary differs from raw samples: {label}/{boundary}/{metric}")
    return row


def error_metrics(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    reference = arrays["float64"].astype(np.float64, copy=False)
    norm = np.linalg.norm(reference)
    metrics = {}
    for mode, image in arrays.items():
        difference = image.astype(np.float64, copy=False) - reference
        metrics[mode] = {
            "max_abs": float(np.abs(difference).max()),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "relative_l2": float(np.linalg.norm(difference) / norm) if norm else 0.0,
        }
    return metrics


def physical_detector_extent(view: dict) -> tuple[float, float, float, float]:
    rows, columns = view["shape"]
    du, dv = view["spacing_mm"]
    return (-columns * du / 2, columns * du / 2, -rows * dv / 2, rows * dv / 2)


def label_detector_axis(axis, view: dict) -> None:
    axis.set_xlabel("Detector u (mm)")
    axis.set_ylabel("Detector v (mm)")
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.set_aspect("equal")


def add_ct_slice(axis, image, slice_index: int | None, window: tuple[float, float]):
    # CTChest is axis aligned. An oblique image would require an explicit reslice,
    # which this baseline figure intentionally refuses to infer.
    affine = image.affine
    if not np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3])), atol=1e-6, rtol=0):
        raise ValueError(
            "The baseline CT panel expects axis-aligned CTChest. Use an explicit oblique reslice"
        )
    index = image.shape[2] // 2 if slice_index is None else slice_index
    if not 0 <= index < image.shape[2]:
        raise ValueError("The requested CT slice is out of range")
    values = np.asarray(image.dataobj[:, :, index]).T
    first = affine @ np.array([-0.5, -0.5, index, 1.0])
    last = affine @ np.array([image.shape[0] - 0.5, image.shape[1] - 0.5, index, 1.0])
    artist = axis.imshow(
        values,
        origin="lower",
        extent=(first[0], last[0], first[1], last[1]),
        cmap="gray",
        vmin=window[0],
        vmax=window[1],
        interpolation="nearest",
        rasterized=True,
    )
    axis.set_title(f"Chest CT · z = {first[2]:.1f} mm", loc="left")
    axis.set_xlabel("RAS x (mm)")
    axis.set_ylabel("RAS y (mm)")
    # Radiological axial display: patient's right is left. Anterior is up.
    axis.set_xlim(max(first[0], last[0]), min(first[0], last[0]))
    axis.set_ylim(min(first[1], last[1]), max(first[1], last[1]))
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    return artist, index


def timing_rows(results: dict, errors: dict) -> list[dict]:
    rows = []
    for mode in MODES:
        resident = measured_row(results, mode, "resident_forward_3views")
        checked = measured_row(results, mode, "checked_public_forward_3views")
        completed = measured_row(results, mode, "completed_numpy_user_call_3views")
        rows.append(
            {
                "mode": mode,
                "resident_device_median_ms": resident["device_median_ms"],
                "resident_device_p95_ms": resident["device_p95_ms"],
                "checked_public_wall_median_ms": checked["wall_median_ms"],
                "checked_public_wall_p95_ms": checked["wall_p95_ms"],
                "completed_numpy_wall_median_ms": completed["wall_median_ms"],
                "completed_numpy_wall_p95_ms": completed["wall_p95_ms"],
                "image_rmse_vs_strict": errors[mode]["rmse"],
                "image_max_abs_error_vs_strict": errors[mode]["max_abs"],
                "image_relative_l2_error_vs_strict": errors[mode]["relative_l2"],
                "measurement_rounds": len(resident["rounds"]),
                "resident_samples_per_round": resident["repeats_per_round"],
            }
        )
    return rows


def export_table(rows: list[dict], results: dict, output: Path) -> None:
    with (output / "baseline_timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "| Existing mode | Resident forward median / p95 (ms) | Checked public call median / p95 (ms) | Completed NumPy call median / p95 (ms) | Image RMSE vs strict |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        text.append(
            f"| {MODE_LABELS[row['mode']]} | {row['resident_device_median_ms']:.3f} / {row['resident_device_p95_ms']:.3f} "
            f"| {row['checked_public_wall_median_ms']:.3f} / {row['checked_public_wall_p95_ms']:.3f} "
            f"| {row['completed_numpy_wall_median_ms']:.3f} / {row['completed_numpy_wall_p95_ms']:.3f} "
            f"| {row['image_rmse_vs_strict']:.6g} |"
        )
    inp = results["input"]
    shape = inp["geometry"][0]["shape"]
    native = inp["native_shape_zyx"]
    text.extend(
        [
            "",
            f"GPU: {results['environment']['gpu']}. Native CT: {native[2]} × {native[1]} × {native[0]} voxels. "
            f"Three views, each {shape[0]} × {shape[1]} pixels. {inp['samples_per_ray']} samples per ray, {inp['integration']} integration.",
            "",
            "Resident timings use device events. Checked public and completed NumPy timings use synchronised wall time. "
            "All entries include the three views. These existing modes use FP32 field storage and FP64 ray accumulation/output.",
            "",
            f"Actual measurement interval: {results['measurement_started_at']} to {results['measurement_finished_at']}.",
            "",
        ]
    )
    (output / "baseline_timings.md").write_text("\n".join(text))


def save_figure(figure, output: Path, stem: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def plot_overview(results, arrays, ct, rows, output, slice_index, window):
    figure = plt.figure(figsize=(17.0, 7.5), layout="constrained")
    grid = figure.add_gridspec(3, 4, height_ratios=(4.0, 1.3, 0.48), hspace=0.08)
    axes = [figure.add_subplot(grid[0, column]) for column in range(4)]
    ct_artist, selected_slice = add_ct_slice(axes[0], ct, slice_index, window)
    figure.colorbar(
        ct_artist, ax=axes[0], label="CT value (HU)", location="bottom", fraction=0.06, pad=0.1
    )
    image_max = float(np.max(arrays["float64"]))
    norm = Normalize(vmin=0, vmax=image_max)
    projection_artist = None
    for index, axis in enumerate(axes[1:]):
        view = results["input"]["geometry"][index]
        projection_artist = axis.imshow(
            arrays["float64"][index],
            extent=physical_detector_extent(view),
            origin="lower",
            cmap="gray",
            norm=norm,
            interpolation="nearest",
            rasterized=True,
        )
        axis.set_title(results["input"]["view_names"][index].upper(), loc="left")
        label_detector_axis(axis, view)
    figure.colorbar(
        projection_artist,
        ax=axes[1:],
        label="Optical depth (dimensionless)",
        location="bottom",
        fraction=0.06,
        pad=0.1,
    )
    table_axis = figure.add_subplot(grid[1, :])
    table_axis.axis("off")
    table_data = []
    for row in rows:
        table_data.append(
            [
                MODE_LABELS[row["mode"]],
                f"{row['resident_device_median_ms']:.2f} / {row['resident_device_p95_ms']:.2f}",
                f"{row['checked_public_wall_median_ms']:.2f} / {row['checked_public_wall_p95_ms']:.2f}",
                f"{row['completed_numpy_wall_median_ms']:.2f} / {row['completed_numpy_wall_p95_ms']:.2f}",
                f"{row['image_rmse_vs_strict']:.3g}",
            ]
        )
    table = table_axis.table(
        cellText=table_data,
        colLabels=[
            "Existing mode",
            "Resident forward\nmedian / p95 (ms)",
            "Checked public call\nmedian / p95 (ms)",
            "Completed NumPy call\nmedian / p95 (ms)",
            "Image RMSE\nvs strict",
        ],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.16, 0.23, 0.23, 0.23, 0.15],
        bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, _), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        cell.PAD = 0.03
        if r == 0:
            cell.get_text().set_weight("bold")
    caption_axis = figure.add_subplot(grid[2, :])
    caption_axis.axis("off")
    inp = results["input"]
    detector = inp["geometry"][0]["shape"]
    shape = inp["native_shape_zyx"]
    date = results["measurement_started_at"].split("T")[0]
    caption_axis.text(
        0,
        1,
        f"{results['environment']['gpu']} · {date} · native CT {shape[2]} × {shape[1]} × {shape[0]} · three {detector[0]} × {detector[1]} views · {inp['samples_per_ray']} samples/ray\n"
        "Three-view timings: resident = device events. Public/completed = wall time. Existing modes retain FP64 accumulation/output.",
        ha="left",
        va="top",
        transform=caption_axis.transAxes,
        fontsize=10,
    )
    save_figure(figure, output, "baseline_chest")
    return selected_slice


def plot_errors(results, arrays, output):
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained", squeeze=False)
    comparisons = ("mixed_float32", "fast_float32")
    differences = {
        mode: arrays[mode].astype(np.float64) - arrays["float64"] for mode in comparisons
    }
    limit = max(float(np.abs(difference).max()) for difference in differences.values())
    if limit == 0:
        raise ValueError("All arrays are identical: an error colour scale is unnecessary")
    norm = Normalize(vmin=-limit, vmax=limit)
    artist = None
    for row, mode in enumerate(comparisons):
        for column, view in enumerate(results["input"]["geometry"]):
            axis = axes[row, column]
            artist = axis.imshow(
                differences[mode][column],
                origin="lower",
                extent=physical_detector_extent(view),
                cmap="RdBu_r",
                norm=norm,
                interpolation="nearest",
                rasterized=True,
            )
            name = results["input"]["view_names"][column].upper()
            axis.set_title(f"{MODE_LABELS[mode]} − strict · {name}", loc="left")
            label_detector_axis(axis, view)
    formatter = ScalarFormatter(useMathText=False)
    formatter.set_powerlimits((0, 0))
    bar = figure.colorbar(
        artist, ax=list(axes.flat), location="right", shrink=0.93, format=formatter
    )
    bar.set_label("Optical-depth difference (dimensionless)")
    bar.set_ticks(np.linspace(-limit, limit, 5))
    save_figure(figure, output, "baseline_error")
    return limit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path(__file__).resolve().parent / "fonts")
    parser.add_argument("--slice", type=int, default=None, dest="slice_index")
    parser.add_argument(
        "--ct-window", nargs=2, type=float, default=(-1000.0, 1000.0), metavar=("LOW_HU", "HIGH_HU")
    )
    args = parser.parse_args()
    results, arrays, ct = load_inputs(args.results_dir, args.ct)
    errors = error_metrics(arrays)
    for mode, computed in errors.items():
        recorded = results["modes"][mode].get("image_error_vs_strict")
        if recorded is None:
            raise ValueError(f"Missing stored image error for {mode}")
        for name, value in computed.items():
            if not np.isclose(recorded[name], value, rtol=1e-8, atol=1e-14):
                raise ValueError(f"Stored image error disagrees with arrays for {mode}/{name}")
    rows = timing_rows(results, errors)
    if args.ct_window[1] <= args.ct_window[0]:
        raise ValueError("CT display window upper bound must exceed its lower bound")
    family = configure_style(args.font_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    selected_slice = plot_overview(
        results, arrays, ct, rows, args.output, args.slice_index, args.ct_window
    )
    limit = plot_errors(results, arrays, args.output)
    export_table(rows, results, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "font": family,
                "ct_slice_index": selected_slice,
                "ct_display_window_hu": args.ct_window,
                "error_shared_scale": [-limit, limit],
                "modes": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
