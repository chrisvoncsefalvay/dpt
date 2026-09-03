"""Plot saved two-view pose-recovery outputs without rerunning the solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from plot_baseline import configure_style, physical_detector_extent, sha256

from dpt.examples._common import load_case


def plot(case_path: Path, run: Path, output: Path, fonts: Path) -> dict:
    case = load_case(case_path)
    summary = json.loads((run / "summary.json").read_text())
    optimisation = json.loads((run / "optimisation.json").read_text())
    if summary["case_sha256"] != sha256(case_path):
        raise ValueError("Run and case checksums differ")
    views = case.config["registration"]["views"]
    names = case.config.get("simulated_acquisition", {}).get(
        "view_names", [view["id"] for view in views]
    )
    if len(views) != 2:
        raise ValueError("This figure requires two views")
    arrays, input_hashes = [], {}
    for view in views:
        shape = tuple(view["geometry"]["shape"])
        counts = case.array(view["observation"], shape=shape, units="counts")
        mask = case.array(view["mask"], shape=shape, units="dimensionless") == 1
        row = {"depth": -np.log(np.maximum(counts, 0.5) / view["open_beam_counts"])}
        for stage in ("initial", "final"):
            name = f"{'initial-' if stage == 'initial' else ''}residual-{view['id']}.npy"
            path = run / name
            values = np.load(path, allow_pickle=False)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid saved residual: {name}")
            row[stage] = np.where(mask, values, np.nan)
            input_hashes[name] = sha256(path)
        row["depth"] = np.where(mask, row["depth"], np.nan)
        arrays.append(row)
    limits = {
        key: max(float(np.nanmax(np.abs(row[key]))) for row in arrays)
        for key in ("depth", "initial", "final")
    }
    if any(not np.isfinite(value) or value <= 0 for value in limits.values()):
        raise ValueError("Plot requires finite, nonzero image ranges")
    configure_style(fonts)
    fig = plt.figure(figsize=(11.5, 10), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, 0.58))
    titles = ("Simulated radiograph", "Initial residual", "Fitted residual")
    axes_by_column, artists = [[], [], []], {}
    for index, (view, row) in enumerate(zip(views, arrays, strict=True)):
        for column, key in enumerate(("depth", "initial", "final")):
            axis = fig.add_subplot(grid[index, column])
            axes_by_column[column].append(axis)
            artists[key] = axis.imshow(
                row[key],
                origin="lower",
                extent=physical_detector_extent(view["geometry"]),
                interpolation="nearest",
                cmap="gray" if key == "depth" else "RdBu_r",
                vmin=0 if key == "depth" else -limits[key],
                vmax=limits[key],
                rasterized=True,
            )
            title = f"{names[index].upper()} · {titles[column]}"
            if key != "depth":
                rms = float(np.sqrt(np.nanmean(row[key] ** 2)))
                title += f"\nRMS {rms:,.3f} counts"
            axis.set_title(title, loc="left", fontsize=10)
            axis.set_xlabel("Detector u (mm)")
            if column == 0:
                axis.set_ylabel("Detector v (mm)")
            axis.spines[["top", "right"]].set_visible(False)
    for column, key in enumerate(("depth", "initial", "final")):
        label = "Optical depth" if key == "depth" else "Predicted minus observed (counts)"
        bar = fig.colorbar(
            artists[key], ax=axes_by_column[column], location="bottom", label=label, shrink=0.9
        )
        bar.locator = MaxNLocator(5)
        if key == "initial":
            bar.formatter = ScalarFormatter(useMathText=True)
            bar.formatter.set_powerlimits((0, 0))
        bar.update_ticks()
        bar.ax.tick_params(labelsize=8)
    axis = fig.add_subplot(grid[2, :])
    history = optimisation["history"]
    axis.semilogy(
        [row["evaluations"] for row in history],
        [row["loss"] for row in history],
        color="#0072B2",
        marker="o",
        markersize=3,
        linewidth=1.7,
    )
    axis.set(
        xlabel="Cumulative solver evaluations (accepted steps shown)",
        ylabel="Poisson half-deviance",
        title=f"{summary['status']} · {summary['termination']}",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(axis="y", alpha=0.15)
    fig.suptitle("Two-view pose recovery from a real CT", x=0.02, ha="left", fontsize=16)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"ct-pose-recovery.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    report = {
        "case_sha256": sha256(case_path),
        "summary_sha256": sha256(run / "summary.json"),
        "optimisation_sha256": sha256(run / "optimisation.json"),
        "residual_sha256": input_hashes,
        "colour_limits": limits,
        "residual_scales": "separate symmetric ranges for initial and fitted residuals",
        "zero_count_display_floor": 0.5,
        "images": "simulated from the real CT; not acquired radiographs",
    }
    (output / "figure.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path.home() / ".cache/dpt/fonts")
    args = parser.parse_args()
    print(json.dumps(plot(args.case, args.run, args.output, args.font_dir), indent=2))


if __name__ == "__main__":
    main()
