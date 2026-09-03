"""Regenerate material benchmark tables and figures from the saved numeric results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
from plot_baseline import configure_style, plt, save_figure
from update_performance_docs import number, percentile, table

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/assets/results/material"
CASES = ("dev-material1-chest-ap-central-fp64", "dev-material3-chest-oblique-dense-fp64")
COLOURS = ("#0072B2", "#D55E00", "#009E73")
LEVELS = ("L1", "L2", "DRAM")
BOUNDARY_LABELS = {"wall": "Completed wall", "device": "CUDA event"}
OPERATION_LABELS = {
    "volume_vjp": "Volume VJP",
    "pose_vjp": "Pose VJP",
    "material_vjp": "Material VJP",
    "torch_complete_iteration": "PyTorch forward + volume/pose VJP",
    "spectral_complete_iteration": "Spectral forward + volume/pose VJP",
}


def case_label(case):
    """Describe the measured geometry, field and arithmetic."""
    geometry = case["geometry"].replace("-ap", " AP").replace("-oblique", " oblique")
    count = case["materials"]
    field = (
        "scalar" if case["kind"] == "scalar" else f"{count} {'basis' if count == 1 else 'bases'}"
    )
    return f"{geometry}, {field}, {case['cotangent']}, {case['precision'].upper()}"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counter(value, spec=",", suffix=""):
    return "unavailable" if value is None else format(value, spec) + suffix


def verify_samples(data):
    """Recompute DPT medians and p95 values from the recorded samples."""
    for row in data["measurements"]:
        for boundary in ("wall", "device"):
            value = row["candidate"][boundary]
            raw = value["raw_ms"]
            pooled = [x for r in row["candidate"]["rounds"] for x in r[boundary + "_ms"]]
            require(raw == pooled and len(raw) == value["count"], "Raw sample mismatch")
            require(all(math.isfinite(x) and x > 0 for x in raw), "Invalid timing sample")
            for key, actual in (
                ("median_ms", statistics.median(raw)),
                ("p95_ms", percentile(raw, 0.95)),
            ):
                require(math.isclose(actual, value[key], rel_tol=1e-12), key + " mismatch")


def profile_groups(profiles, data):
    groups = []
    for case in CASES:
        group = [
            capture
            for capture in profiles["captures"]
            if capture["case"]["id"] == case
            and capture["metadata"]["source"]["sha256"] == data["sources"]["candidate"]["sha256"]
        ]
        require(len(group) == 1, "Expected one capture of the timed implementation per case")
        require(group[0]["metadata"]["completed_checks_passed"], "Failed capture checks")
        groups.append(group)
    return groups


def numeric_tables(data, groups, diagnostic):
    tables = {}
    measured = {(r["case"]["id"], r["operation"]): r for r in data["measurements"]}
    rows = []
    for row in data["measurements"]:
        case = row["case"]
        if not (
            row["operation"] == "material_vjp"
            and case["precision"] == "fp64"
            and case["suite"] == "dev"
            and case["cotangent"] in ("central", "dense")
        ):
            continue
        count = case["materials"]
        view = "AP" if case["geometry"].endswith("-ap") else "oblique"
        pattern = "central region" if case["cotangent"] == "central" else "dense"
        value = row["candidate"][row["timing_boundary"]]
        complete = measured[case["id"], "spectral_complete_iteration"]
        iteration = complete["candidate"][complete["timing_boundary"]]
        rows.append(
            [
                f"{count} {'basis' if count == 1 else 'bases'}, {view}, {pattern}",
                f"{value['median_ms']:.3f} / {value['p95_ms']:.3f}",
                f"{iteration['median_ms']:.3f} / {iteration['p95_ms']:.3f}",
            ]
        )
    tables["FP64 CUDA-event timing summary"] = table(
        [
            "FP64 workload",
            "Material VJP median / p95 (ms)",
            "Spectral iteration median / p95 (ms)",
        ],
        rows,
    )
    tables["Environment"] = table(
        ["GPU", "Driver", "CUDA", "Torch", "Warp", "Python"],
        [
            [
                e["gpu"],
                e["driver"],
                e["torch_cuda"],
                e["versions"]["torch"],
                e["versions"]["warp-lang"],
                e["python"],
            ]
            for e in data["environments"]
        ],
    )
    rows = []
    for row in data["measurements"]:
        boundary = row["timing_boundary"]
        shape = data["configuration"]["geometry"][row["case"]["geometry"]]["panel"]["shape"]
        value = row["candidate"][boundary]
        rows.append(
            [
                case_label(row["case"]),
                OPERATION_LABELS[row["operation"]],
                "x".join(map(str, shape)),
                BOUNDARY_LABELS[boundary],
                f"{value['median_ms']:.6f} / {value['p95_ms']:.6f}",
                value["count"],
            ]
        )
    tables["Completed operation timings"] = table(
        [
            "Case",
            "Operation",
            "Detector",
            "Boundary",
            "Median / p95 (ms)",
            "Samples",
        ],
        rows,
    )
    rows = []
    for report in data["checks"]:
        if report["role"] != "candidate":
            continue
        check = report["checks"]
        pose = check["pose_fd"]
        require(
            all(check[name]["passed"] for name in ("adjoint", "volume_fd", "pose_fd")),
            "Failed numerical check",
        )
        selected = next(p for p in pose["samples"] if p["step"] == pose["selected_step"])
        rows.append(
            [
                case_label(report["case"]),
                "optical depth" if report["case"]["kind"] == "scalar" else "photons",
                number(check["adjoint"]["relative_error"]),
                number(check["volume_fd"]["relative_error"]),
                number(selected["relative_error"]),
                number(selected["step"]),
            ]
        )
    tables["Numerical checks"] = table(
        [
            "Case",
            "Objective units",
            "Adjoint relative",
            "Volume FD relative",
            "Pose FD relative",
            "Pose step",
        ],
        rows,
    )
    rows = []
    for group, name in zip(groups, ("M1 AP central", "M3 oblique dense"), strict=True):
        for capture in group:
            m = capture["main_metrics"]
            rows.append(
                [
                    name,
                    counter(m.get("fp64_add_mul_fma_flops")),
                    counter(m.get("conversion_thread_instructions")),
                    counter(m.get("global_atomic_and_reduction_thread_instructions")),
                    counter(m.get("registers_per_thread"), ".0f"),
                    counter(m.get("fp64_pipe_active_percent"), ".2f", "%"),
                    counter(m.get("register_spill_instructions"), ".0f"),
                    counter(m.get("local_load_sector_bytes"), ".0f")
                    + " / "
                    + counter(m.get("local_store_sector_bytes"), ".0f"),
                ]
            )
    tables["Material kernel counters"] = table(
        [
            "Native case",
            "FP64 ADD/MUL/FMA FLOPs",
            "Conversions",
            "Atomics",
            "Registers/thread",
            "FP64 pipe active",
            "Classified spills",
            "Local load / store B",
        ],
        rows,
    )
    if diagnostic:
        require(
            all(
                check["passed"] for check in diagnostic["captures"]["candidate"]["checks"].values()
            ),
            "Failed diagnostic checks",
        )
        require(
            diagnostic["captures"]["candidate"]["source"]["sha256"]
            == data["sources"]["candidate"]["sha256"],
            "Diagnostic source differs",
        )
        metrics = diagnostic["captures"]["candidate"]["kernel_metrics"][0]["metrics"]
        tables["32-basis analytic diagnostic"] = table(
            ["Main-kernel metric", "DPT"],
            [
                [key, counter(metrics.get(key), ",.0f")]
                for key in (
                    "registers_per_thread",
                    "register_spill_instructions",
                    "local_load_sector_bytes",
                    "local_store_sector_bytes",
                    "fp64_add_mul_fma_flops",
                    "global_atomic_and_reduction_thread_instructions",
                )
            ],
        )
    return tables


def latency(data, output):
    complete = [r for r in data["measurements"] if r["operation"].endswith("complete_iteration")]
    patterns = ("central", "dense", "signed", "zero")
    largest = max(sum(r["case"]["cotangent"] == p for r in complete) for p in patterns)
    figure, axes = plt.subplots(
        2, 2, figsize=(15.5, max(7.5, largest * 0.64 + 2.2)), layout="constrained"
    )
    for axis, pattern in zip(axes.flat, patterns, strict=True):
        rows = sorted(
            (r for r in complete if r["case"]["cotangent"] == pattern),
            key=lambda r: tuple(
                r["case"][k] for k in ("suite", "geometry", "materials", "precision")
            ),
        )
        labels = []
        for index, row in enumerate(rows):
            case, boundary = row["case"], row["timing_boundary"]
            shape = data["configuration"]["geometry"][case["geometry"]]["panel"]["shape"]
            labels.append(
                f"{case_label(case)}\n{'x'.join(map(str, shape))}, {BOUNDARY_LABELS[boundary]}"
            )
            value = row["candidate"][boundary]
            median, p95 = value["median_ms"], value["p95_ms"]
            axis.plot([median, p95], [index] * 2, color=COLOURS[0], lw=1)
            axis.plot(
                median,
                index,
                marker="o",
                color=COLOURS[0],
                markersize=4,
                label="Median" if index == 0 else None,
            )
            axis.plot(
                p95,
                index,
                marker="|",
                color=COLOURS[0],
                markersize=7,
                label="p95" if index == 0 else None,
            )
        axis.set_yticks(range(len(rows)), labels)
        axis.invert_yaxis()
        axis.set_xscale("log")
        axis.set_xlabel("Completed iteration latency (ms, log scale)")
        axis.set_title(pattern.title() + " cotangent", loc="left")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "DPT complete iterations",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    save_figure(figure, output, "latency")
    plt.close(figure)


def resources(group, output, stem):
    panels = (
        ("replay_kernel_ms", "Replay kernel duration (ms)", 1),
        ("registers_per_thread", "Registers / thread", 1),
        ("fp64_pipe_active_percent", "FP64 pipeline active (%)", 1),
        ("predicated_on_lanes_per_warp_instruction", "Predicated lanes / warp instruction", 1),
        ("conversion_thread_instructions", "Conversion thread instructions (billions)", 1e-9),
        ("register_spill_instructions", "Register spill instructions", 1),
        (
            "global_atomic_and_reduction_thread_instructions",
            "Global atomic/reduction thread instructions (billions)",
            1e-9,
        ),
        ("L2_request_counter_bytes", "L2 request counter traffic (GB)", 1e-9),
    )
    figure, axes = plt.subplots(2, 4, figsize=(13.6, 6.4))
    for axis, (metric, ylabel, scale) in zip(axes.flat, panels, strict=True):
        for i, capture in enumerate(group):
            value = capture["main_metrics"].get(metric)
            if value is None:
                axis.text(i, 0, "unavailable", ha="center")
            else:
                axis.bar(i, value * scale, color=COLOURS[i])
        axis.set_xticks(range(len(group)), ["DPT"] * len(group))
        axis.set_ylabel(ylabel, fontsize=8)
        axis.set_ylim(bottom=0)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(case_label(group[0]["case"]), x=0.06, ha="left", fontsize=12)
    figure.text(0.06, 0.025, "Kernel duration and counters measured during Nsight Compute replay.")
    figure.subplots_adjust(left=0.06, right=0.99, bottom=0.15, top=0.91, wspace=0.55, hspace=0.65)
    save_figure(figure, output, stem + "-resources")
    plt.close(figure)


def roofline(group, output, stem):
    kernels = [next(p for p in c["kernels"] if p["id"] == c["main_kernel_id"]) for c in group]
    reference = kernels[0]
    peak = reference["precision"]["fp64"]["model_peak_flops_per_second"]
    ceilings = reference["bandwidth_model_peak_bytes_per_second"]
    require(peak and all(ceilings.get(level) for level in LEVELS), "Missing roofline ceilings")
    points = []
    for capture, kernel in zip(group, kernels, strict=True):
        values = kernel["precision"]["fp64"]
        for level in LEVELS:
            x = values.get("arithmetic_intensity_flops_per_byte", {}).get(level)
            y = values.get("flops_per_second")
            if x is not None and y is not None and x > 0 and y > 0:
                points.append((capture["label"], level, x, y / 1e9))
    require(points, "No measured FP64 FLOP points")
    xmin = 10 ** math.floor(math.log10(min(p[2] for p in points)) - 0.4)
    xmax = max(1e3, 10 ** math.ceil(math.log10(max(p[2] for p in points)) + 0.3))
    grid = np.geomspace(xmin, xmax, 400)
    figure, axis = plt.subplots(figsize=(8.5, 5.7))
    for level, style in zip(LEVELS, (":", "--", "-"), strict=True):
        axis.plot(grid, np.minimum(peak, grid * ceilings[level]) / 1e9, color="black", ls=style)
    for index, capture in enumerate(group):
        chosen = [p for p in points if p[0] == capture["label"]]
        axis.plot([p[2] for p in chosen], [p[3] for p in chosen], color=COLOURS[index], lw=0.7)
        for _, level, x, y in chosen:
            axis.scatter(
                x, y, color=COLOURS[index], marker={"L1": "o", "L2": "^", "DRAM": "s"}[level]
            )
        axis.plot([], [], color=COLOURS[index], label="DPT")
    axis.set(
        xscale="log",
        yscale="log",
        xlim=(xmin, xmax),
        xlabel="FP64 arithmetic intensity (FLOP/byte)",
        ylabel="FP64 arithmetic throughput (GFLOP/s)",
    )
    axis.set_title(case_label(group[0]["case"]), loc="left", fontsize=10)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.text(0.12, 0.085, reference["device"] + " · main material VJP kernel", fontsize=8)
    figure.text(
        0.12,
        0.055,
        "Theoretical NCU ceilings at capture clocks.",
        fontsize=8,
    )
    figure.text(
        0.12,
        0.025,
        "L1 circle/dotted · L2 triangle/dashed · DRAM square/solid. L1/L2 writeback proxies.",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.91)
    save_figure(figure, output, stem + "-roofline-fp64")
    plt.close(figure)


def report_text(data, tables):
    sampling = data["configuration"]["sampling"]
    counts = sorted(
        {row["candidate"][row["timing_boundary"]]["count"] for row in data["measurements"]}
    )
    text = [
        "# Material projection performance",
        "",
        f"Measured {data['measurement_started_at']} to {data['measurement_finished_at']}.",
        "",
        "Native chest and brain CT volumes use "
        f"{sampling['samples_per_ray']} clipped midpoint samples per ray. "
        "Volume fields use FP32 storage. Each row specifies its FP32 or FP64 arithmetic.",
        "",
        "Material densities are deterministic algebraic fields derived from "
        "`rho = max(1 + HU/1000, 0)`, with each channel cast to FP32. "
        "Scalar objectives use optical depth; material objectives use photon counts.",
        "",
        f"Each timing row contains {' or '.join(map(str, counts))} samples across "
        f"{data['rounds']} rounds. CUDA-event timings cover resident native operations "
        "and destination resets. Completed-wall timings cover the full PyTorch call, "
        "including validation and allocation.",
        "",
    ]
    for name, value in tables.items():
        text += ["## " + name, "", value, ""]
    text += [
        "FLOPs count predicated-thread ADD + MUL + 2 × FMA. Conversion and atomic "  # noqa: RUF001
        "instructions are reported separately. Local-memory sector bytes count load "
        "and store traffic; classified spills count register-spill instructions.",
        "",
        "Roofline ceilings use the theoretical NCU model at the recorded capture clocks. "
        "The L1 and L2 intensities use writeback traffic proxies.",
        "",
    ]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DATA / "results.json")
    parser.add_argument("--profiles", type=Path, default=DATA / "profiles.json")
    parser.add_argument("--diagnostic", type=Path, default=DATA / "material32-diagnostic.json")
    parser.add_argument("--font-cache", type=Path, default=Path.home() / ".cache/dpt/fonts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-page", type=Path)
    args = parser.parse_args()
    data, profiles = [json.loads(p.read_text()) for p in (args.results, args.profiles)]
    diagnostic = json.loads(args.diagnostic.read_text()) if args.diagnostic else None
    verify_samples(data)
    groups = profile_groups(profiles, data)
    tables = numeric_tables(data, groups, diagnostic)
    if args.check_page:

        def normalise(text):
            return "\n".join(
                "|".join(c.strip() for c in line.split("|"))
                for line in text.splitlines()
                if line.startswith("|")
            )

        page = normalise(args.check_page.read_text())
        for name, value in tables.items():
            require(normalise(value) in page, "Documentation table differs: " + name)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "tables.md").write_text(report_text(data, tables))
    configure_style(args.font_cache)
    latency(data, args.output)
    for group in groups:
        stem = group[0]["case"]["id"].removeprefix("dev-") + "-" + group[0]["operation"]
        resources(group, args.output, stem)
    roofline(
        groups[1], args.output, groups[1][0]["case"]["id"].removeprefix("dev-") + "-material_vjp"
    )
    print(f"Regenerated {len(tables)} numeric tables and four figures in PNG/SVG/PDF.")


if __name__ == "__main__":
    main()
