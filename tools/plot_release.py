"""Draw release figures exclusively from completed measured real-CT outputs."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter
from plot_baseline import (
    add_ct_slice,
    configure_style,
    label_detector_axis,
    physical_detector_extent,
    save_figure,
    sha256,
)


def read_result(directory):
    result = json.loads((directory / "results.json").read_text())
    if not result.get("measurement_finished_at") or result.get("failure"):
        raise ValueError(f"Only completed measurements can generate final figures: {directory}")
    return result


def image_arrays(directory, result):
    shape = (len(result["input"]["geometry"]), *result["input"]["geometry"][0]["shape"])
    arrays = {
        mode: np.load(directory / f"{mode}_depth.npy", allow_pickle=False)
        for mode in ("fp32", "fp64")
    }
    for mode, array in arrays.items():
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{mode} image does not match measured geometry")
    if arrays["fp32"].dtype != np.float32 or arrays["fp64"].dtype != np.float64:
        raise ValueError("Measured figures require the precision-labelled outputs")
    return arrays


def ct_slice(axis, ct, index):
    if np.allclose(ct.affine[:3, :3], np.diag(np.diag(ct.affine[:3, :3])), rtol=0, atol=1e-6):
        return add_ct_slice(axis, ct, index, (-1000, 1000))
    index = ct.shape[2] // 2 if index is None else index
    if not 0 <= index < ct.shape[2]:
        raise ValueError("CT slice outside volume")
    sx, sy = np.linalg.norm(ct.affine[:3, :3], axis=0)[:2]
    extent = (-sx / 2, (ct.shape[0] - 0.5) * sx, -sy / 2, (ct.shape[1] - 0.5) * sy)
    artist = axis.imshow(
        np.asarray(ct.dataobj[:, :, index]).T,
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=-1000,
        vmax=1000,
        interpolation="nearest",
    )
    axis.set(
        title=f"Oblique CT plane {index}",
        xlabel="Local grid x (mm)",
        ylabel="Local grid y (mm)",
    )
    return artist, index


def overview(result, arrays, ct, output, index):
    names = result["input"]["view_names"]
    views = result["input"]["geometry"]
    fig, axes = plt.subplots(
        1, len(names) + 1, figsize=(4 * (len(names) + 1), 4.7), layout="constrained"
    )
    artist, _ = ct_slice(axes[0], ct, index)
    fig.colorbar(artist, ax=axes[0], location="bottom", label="CT value (HU)")
    high = float(max(array.max() for array in arrays.values()))
    for axis, name, view, array in zip(axes[1:], names, views, arrays["fp32"], strict=True):
        artist = axis.imshow(
            array,
            origin="lower",
            extent=physical_detector_extent(view),
            cmap="gray",
            vmin=0,
            vmax=high,
            interpolation="nearest",
            rasterized=True,
        )
        axis.set_title(name.upper(), loc="left")
        label_detector_axis(axis, view)
    fig.colorbar(
        artist,
        ax=list(axes[1:]),
        location="bottom",
        label="Optical depth (dimensionless)",
    )
    save_figure(fig, output, "ct_projections")
    plt.close(fig)


def errors(result, arrays, output):
    delta = arrays["fp32"].astype(np.float64) - arrays["fp64"]
    limit = float(np.abs(delta).max())
    if not limit:
        limit = np.finfo(np.float64).eps
    names = result["input"]["view_names"]
    views = result["input"]["geometry"]
    fig, axes = plt.subplots(
        2,
        len(names),
        figsize=(4.1 * len(names), 8.3),
        layout="constrained",
        squeeze=False,
    )
    high = float(arrays["fp64"].max())
    for i, (name, view) in enumerate(zip(names, views, strict=True)):
        top = axes[0, i].imshow(
            arrays["fp64"][i],
            origin="lower",
            extent=physical_detector_extent(view),
            cmap="gray",
            vmin=0,
            vmax=high,
            interpolation="nearest",
            rasterized=True,
        )
        bottom = axes[1, i].imshow(
            delta[i],
            origin="lower",
            extent=physical_detector_extent(view),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            rasterized=True,
        )
        axes[0, i].set_title(f"{name.upper()} · FP64", loc="left")
        axes[1, i].set_title(f"{name.upper()} · FP32 minus FP64", loc="left")
        label_detector_axis(axes[0, i], view)
        label_detector_axis(axes[1, i], view)
    fig.colorbar(top, ax=list(axes[0]), location="bottom", label="Optical depth (dimensionless)")
    bar = fig.colorbar(
        bottom,
        ax=list(axes[1]),
        location="bottom",
        label="Signed optical-depth error (dimensionless)",
    )
    bar.formatter.set_powerlimits((-3, 3))
    bar.update_ticks()
    save_figure(fig, output, "fp32_fp64_error")
    plt.close(fig)
    return {
        "max_abs": float(np.abs(delta).max()),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(arrays["fp64"])),
    }


def corners(grid):
    shape = list(reversed(grid["shape"]))
    indices = np.array(list(itertools.product(*[(-0.5, n - 0.5) for n in shape])))
    transform = np.array(grid["orientation"]).reshape(3, 3) @ np.diag(grid["spacing_mm"])
    return np.array(grid["origin_mm"]) + indices @ transform.T


def detector_segment(panel):
    rows, columns = panel["shape"]
    du, dv = panel["spacing_mm"]
    origin, u, v = (np.array(panel[key]) for key in ("origin_mm", "u", "v"))
    centre = origin + (columns - 1) * du * u / 2 + (rows - 1) * dv * v / 2
    return centre, centre - columns * du * u / 2, centre + columns * du * u / 2


def draw_geometry(axis, grid, panels, names=None, active=None):
    volume = corners(grid)
    for i in range(8):
        for j in range(i + 1, 8):
            if (i ^ j).bit_count() == 1:
                axis.plot(volume[[i, j], 0], volume[[i, j], 1], color="black", linewidth=0.8)
    palette = ("#0072B2", "#D55E00", "#009E73")
    for index, panel in enumerate(panels):
        source = np.array(panel["source_mm"])
        centre, left, right = detector_segment(panel)
        colour = (
            palette[index % len(palette)]
            if active is None
            else ("#0072B2" if index == active else "#D0D0D0")
        )
        width = 1.8 if active is None or index == active else 0.6
        axis.plot([left[0], right[0]], [left[1], right[1]], color=colour, linewidth=width)
        axis.scatter(source[0], source[1], marker="o", s=18, color=colour, zorder=3)
        if active is None or index == active:
            axis.plot(
                [source[0], centre[0]],
                [source[1], centre[1]],
                color=colour,
                linewidth=0.8,
                linestyle="--",
            )
            if names:
                axis.annotate(
                    names[index].upper(),
                    source[:2],
                    xytext=(-5 if source[0] > volume[:, 0].max() else 5, 5),
                    textcoords="offset points",
                    color="black",
                    horizontalalignment="right" if source[0] > volume[:, 0].max() else "left",
                )
    axis.set(xlabel="RAS x (mm)", ylabel="RAS y (mm)")
    axis.set_aspect("equal")
    axis.set_title("Source and detector geometry · axial view", loc="left")


def geometry(result, output):
    fig, axis = plt.subplots(figsize=(7, 6), layout="constrained")
    draw_geometry(
        axis,
        result["input"]["grid"],
        result["input"]["geometry"],
        result["input"]["view_names"],
    )
    save_figure(fig, output, "acquisition_geometry")
    plt.close(fig)


def sensitivity(directory, result, arrays, ct, output, index, mode):
    measured = read_result(directory)
    if measured["input"]["sha256"] != result["input"]["sha256"]:
        raise ValueError("Sensitivity CT differs from projected CT")
    match = [
        row for row in measured["measurements"] if row["label"] == mode + "/resident_volume_vjp_ap"
    ]
    if len(match) != 1:
        raise ValueError("Need actual AP native volume VJP measurement")
    if match[0]["configuration"]["geometry"][0] != result["input"]["geometry"][0]:
        raise ValueError("Sensitivity and displayed projection geometry differ")
    gradient = np.load(directory / f"{mode}_ap_sensitivity.npy", allow_pickle=False)
    if (
        gradient.shape != tuple(result["input"]["native_shape_zyx"])
        or not np.isfinite(gradient).all()
    ):
        raise ValueError("Sensitivity does not match native CT grid")
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.8), layout="constrained")
    ct_artist, index = ct_slice(axes[0], ct, index)
    fig.colorbar(ct_artist, ax=axes[0], location="bottom", label="CT value (HU)")
    panel = result["input"]["geometry"][0]
    extent = physical_detector_extent(panel)
    depth = axes[1].imshow(
        arrays[mode][0],
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=0,
        vmax=float(arrays[mode].max()),
        interpolation="nearest",
    )
    rows, columns = panel["shape"]
    du, dv = panel["spacing_mm"]
    r0, r1 = rows * 5 // 16, rows * 11 // 16
    c0, c1 = columns * 5 // 16, columns * 11 // 16
    axes[1].add_patch(
        Rectangle(
            (extent[0] + c0 * du, extent[2] + r0 * dv),
            (c1 - c0) * du,
            (r1 - r0) * dv,
            fill=False,
            edgecolor="#E69F00",
            linewidth=1.5,
        )
    )
    axes[1].set_title("AP detector-region mean", loc="left")
    label_detector_axis(axes[1], panel)
    fig.colorbar(depth, ax=axes[1], location="bottom", label="Optical depth (dimensionless)")
    ct_slice(axes[2], ct, index)
    section = gradient[index]
    # Show the actual gradient at the displayed CT plane on its physical axes.
    high = float(np.max(np.abs(section)))
    if high == 0:
        raise ValueError("Requested CT plane has zero sensitivity. Choose another actual slice")
    heat = axes[2].imshow(
        section,
        origin="lower",
        extent=ct_artist.get_extent(),
        cmap="magma",
        vmin=0,
        vmax=high,
        alpha=0.85,
        interpolation="nearest",
        rasterized=True,
    )
    axes[2].set_xlim(axes[0].get_xlim())
    axes[2].set_ylim(axes[0].get_ylim())
    axes[2].set_title("Volume sensitivity on the CT plane", loc="left")
    bar = fig.colorbar(heat, ax=axes[2], location="bottom", label="Sensitivity to attenuation (mm)")
    bar.formatter.set_powerlimits((-3, 3))
    bar.update_ticks()
    save_figure(fig, output, "detector_region_sensitivity")
    plt.close(fig)


def sweep(directory, result, output, mode):
    measured = read_result(directory)
    if measured["input"]["sha256"] != result["input"]["sha256"]:
        raise ValueError("Sweep CT differs from projected CT")
    match = [
        row
        for row in measured["measurements"]
        if row["label"] == f"{mode}/batched_16views_resident"
    ]
    if len(match) != 1:
        raise ValueError("Need actual 16-view batched measurement")
    configuration = match[0]["configuration"]
    panels = configuration["geometry"]
    angles = configuration["angles_radians"]
    images = np.load(directory / f"{mode}_batched_16_depth.npy", allow_pickle=False)
    if images.shape != (16, *panels[0]["shape"]) or not np.isfinite(images).all():
        raise ValueError("Actual sweep output does not match recorded geometry")
    maximum = float(images.max())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    artist = axes[0].imshow(
        images[0],
        origin="lower",
        extent=physical_detector_extent(panels[0]),
        cmap="gray",
        vmin=0,
        vmax=maximum,
        interpolation="nearest",
    )
    label_detector_axis(axes[0], panels[0])
    fig.colorbar(artist, ax=axes[0], location="bottom", label="Optical depth (dimensionless)")

    def frame(index):
        artist.set_data(images[index])
        axes[0].set_title(f"View {index + 1}/16 · {np.degrees(angles[index]):.1f}°", loc="left")
        axes[1].clear()
        draw_geometry(axes[1], measured["input"]["grid"], panels, active=index)
        return (artist,)

    frame(0)
    save_figure(fig, output, "view_sweep_geometry")
    animation = FuncAnimation(fig, frame, frames=16, interval=250, blit=False)
    animation.save(output / "view_sweep.gif", writer=PillowWriter(fps=4), dpi=100)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), layout="constrained")
    for axis, index in zip(axes, (0, 4, 8, 12), strict=True):
        artist = axis.imshow(
            images[index],
            origin="lower",
            extent=physical_detector_extent(panels[index]),
            cmap="gray",
            vmin=0,
            vmax=maximum,
            interpolation="nearest",
            rasterized=True,
        )
        label_detector_axis(axis, panels[index])
        axis.set_title(f"{np.degrees(angles[index]):.1f}°", loc="left")
    fig.colorbar(artist, ax=list(axes), location="bottom", label="Optical depth (dimensionless)")
    save_figure(fig, output, "view_sweep_samples")
    plt.close(fig)


def performance(release_paths, competitor_paths, output):
    """Plot completed forward calls against their recorded image discrepancy."""
    from update_performance_docs import verify_samples

    documents = []
    for path in [*release_paths, *competitor_paths]:
        data = json.loads(path.read_text())
        if not data.get("measurement_finished_at") or data.get("failure"):
            raise ValueError(f"Incomplete performance result: {path}")
        verify_samples(data)
        documents.append((path, data))
    points = []
    for path, data in documents:
        inputs = data["input"]
        if inputs["native_shape_zyx"] != [139, 512, 512]:
            continue
        if len(inputs["geometry"]) != 3 or inputs["geometry"][0]["shape"] != [512, 512]:
            continue
        for row in data.get("measurements", []):
            label = row["label"]
            if label.endswith("/public_projector_3views"):
                mode = label.split("/")[0]
                checks = [
                    d["checks"][mode + "_real_volume"]
                    for _, d in documents
                    if mode + "_real_volume" in d.get("checks", {})
                    and d.get("source", {}).get("sha256") == data["source"]["sha256"]
                    and all(
                        d.get("input", {}).get(k) == inputs.get(k)
                        for k in ("sha256", "grid", "geometry", "samples_per_ray", "pose")
                    )
                ]
                if not checks:
                    raise ValueError(f"No source-matched image error for {path}:{mode}")
                error = checks[0]["image_error_vs_fp64"]
                title = "DPT " + mode.upper()
                group = title
            elif label.endswith("/forward_3views") and "backends" in data:
                backend = label.split("/")[0]
                info = data["backends"][backend]
                error = info.get(
                    "image_error_vs_dpt_fp64", info.get("image_error_vs_frozen_dpt_fp64")
                )
                if error is None:
                    raise ValueError(f"No measured image error: {label}")
                title = backend
                group = (
                    "DiffDRR trilinear"
                    if backend.startswith("diffdrr-trilinear-")
                    else {
                        "diffdrr-siddon": "DiffDRR Siddon",
                        "astra": "ASTRA DLPack",
                        "tigre": "TIGRE, including transfers",
                    }[backend]
                )
            else:
                continue
            points.append(
                {
                    "label": title,
                    "group": group,
                    "wall_median_ms": row["wall_median_ms"],
                    "wall_p95_ms": row["wall_p95_ms"],
                    "image_rmse": error["rmse"],
                    "result_file": str(path),
                    "result_sha256": sha256(path),
                    "operation": label,
                    "configuration": row.get("configuration", {}),
                }
            )
    if not points:
        raise ValueError("No completed native three-view forward measurements")
    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    fig.subplots_adjust(left=0.14, bottom=0.19, right=0.98, top=0.88)
    colours = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    markers = ("o", "s", "^", "D", "v", "P")
    groups = list(dict.fromkeys(p["group"] for p in points))
    for group, colour, marker in zip(groups, colours, markers, strict=False):
        subset = sorted(
            (p for p in points if p["group"] == group), key=lambda p: p["wall_median_ms"]
        )
        axis.plot(
            [p["wall_median_ms"] for p in subset],
            [p["image_rmse"] for p in subset],
            marker=marker,
            color=colour,
            linestyle="-" if len(subset) > 1 else "none",
            label=group,
            markersize=6,
            linewidth=1,
        )
        if group == "DiffDRR trilinear":
            for index, point in enumerate(subset):
                axis.annotate(
                    point["label"].rsplit("-", 1)[-1],
                    (point["wall_median_ms"], point["image_rmse"]),
                    xytext=(0, 9 + index * 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
    axis.set_xscale("log")
    positive = [p["image_rmse"] for p in points if p["image_rmse"] > 0]
    axis.set_yscale("symlog", linthresh=min(positive) / 10 if positive else 1e-7)
    if positive:
        low, high = min(positive), max(positive)
        ticks = [
            scale * 10.0**exponent
            for exponent in range(int(np.floor(np.log10(low))), int(np.ceil(np.log10(high))) + 1)
            for scale in (1, 2, 5)
            if low * 0.65 <= scale * 10.0**exponent <= high * 1.6
        ]
        if any(p["image_rmse"] == 0 for p in points):
            ticks.insert(0, 0)
            axis.set_ylim(-low / 30, high * 1.4)
        else:
            axis.set_ylim(low * 0.75, high * 1.4)
        axis.yaxis.set_major_locator(FixedLocator(ticks))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1g}"))
    axis.set(
        xlabel="Completed three-view forward, median (ms)",
        ylabel="Optical-depth RMSE against DPT FP64",
    )
    axis.legend(frameon=False, loc="best", fontsize=9)
    axis.set_title("Native chest CT · three 512² views", loc="left")
    fig.text(
        0.14,
        0.07,
        "Physical rays and field units are shared. Discretisations differ.",
        ha="left",
        fontsize=9,
    )
    save_figure(fig, output, "latency_error")
    plt.close(fig)
    return points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path)
    p.add_argument("--ct", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sensitivity-results", type=Path)
    p.add_argument("--sweep-results", type=Path)
    p.add_argument("--mode", choices=("fp32", "fp64"), default="fp32")
    p.add_argument("--slice", type=int)
    p.add_argument("--font-cache", type=Path, default=Path.home() / ".cache/dpt/fonts")
    p.add_argument("--performance-results", type=Path, nargs="*", default=[])
    p.add_argument("--competitor-results", type=Path, nargs="*", default=[])
    p.add_argument("--performance-only", action="store_true")
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error("output directory must be absent or empty")
    a.output.mkdir(parents=True, exist_ok=True)
    family = configure_style(a.font_cache)
    if a.performance_only:
        points = performance(a.performance_results, a.competitor_results, a.output)
        (a.output / "figure_results.json").write_text(
            json.dumps(
                {
                    "generated_at_utc": datetime.now(UTC).isoformat(),
                    "font_family": family,
                    "points": points,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        return
    if a.results is None or a.ct is None:
        p.error("--results and --ct are required for CT/projection figures")
    result = read_result(a.results)
    if sha256(a.ct) != result["input"]["sha256"]:
        raise ValueError("CT file differs from measured input")
    ct = nib.load(a.ct)
    if tuple(reversed(ct.shape)) != tuple(result["input"]["native_shape_zyx"]):
        raise ValueError("Native CT shape mismatch")
    arrays = image_arrays(a.results, result)
    overview(result, arrays, ct, a.output, a.slice)
    metrics = errors(result, arrays, a.output)
    geometry(result, a.output)
    if a.sensitivity_results:
        sensitivity(a.sensitivity_results, result, arrays, ct, a.output, a.slice, a.mode)
    if a.sweep_results:
        sweep(a.sweep_results, result, a.output, a.mode)
    points = []
    if a.performance_results or a.competitor_results:
        points = performance(a.performance_results, a.competitor_results, a.output)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_results_sha256": sha256(a.results / "results.json"),
        "ct_sha256": sha256(a.ct),
        "font_family": family,
        "image_error": metrics,
        "latency_error_points": points,
        "sensitivity_source": str(a.sensitivity_results) if a.sensitivity_results else None,
        "sweep_source": str(a.sweep_results) if a.sweep_results else None,
        "files": [p.name for p in sorted(a.output.iterdir())],
    }
    (a.output / "figure_results.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(a.output),
                "image_error": metrics,
                "latency_error_points": len(points),
                "files": summary["files"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
