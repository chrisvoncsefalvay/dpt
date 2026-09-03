"""Summarise every kernel in exported Nsight Compute operations, without GPU work.

Pass NCU raw-page CSVs and matching *-instances.csv exports. Exact traffic
counters remain separate from the official hierarchical-roofline writeback
proxies. Missing counters are null. Profile times are never headline timings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

LABELS = {
    "float64-forward": "Strict forward",
    "mixed_float32-forward": "Mixed forward",
    "fast_float32-forward": "Fast forward",
    "float64-volume": "Strict volume VJP",
    "float64-pose": "Strict pose VJP",
    "fp32-material-vjp": "FP32 material VJP",
    "fp64-material-vjp": "FP64 material VJP",
    **{
        f"{p}-{op}": f"{p.upper()} {label}"
        for p in ("fp32", "fp64")
        for op, label in (
            ("forward", "forward"),
            ("volume", "volume VJP"),
            ("pose", "pose VJP"),
        )
    },
}
LEVELS = ("L1", "L2", "DRAM")
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def read_raw(path):
    """Return all wide raw-page kernel rows and their shared units."""
    with path.open(newline="") as stream:
        rows = [row for row in csv.reader(stream) if row]
    header_index = next(
        (i for i, row in enumerate(rows) if row[0] == "ID" and "Kernel Name" in row),
        None,
    )
    if header_index is None or len(rows) <= header_index + 2:
        raise ValueError(f"Missing NCU raw header, units or kernel rows: {path}")
    header, units = rows[header_index : header_index + 2]
    kernels = []
    for row in rows[header_index + 2 :]:
        if len(row) != len(header):
            raise ValueError(f"Malformed NCU row in {path}: {len(row)} != {len(header)}")
        if row[0] == "ID":
            raise ValueError(f"Repeated NCU table header in {path}")
        kernels.append(dict(zip(header, row, strict=True)))
    return kernels, dict(zip(header, units, strict=True))


def number(values, units, key):
    text = values.get(key, "").strip()
    if text.lower() in ("", "n/a", "nan", "not available", "not supported"):
        return None
    try:
        value = float(text.replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    unit = units.get(key, "")
    if unit in ("ms", "us", "ns"):
        return value * {"ms": 1e-3, "us": 1e-6, "ns": 1e-9}[unit]
    for prefix, factor in (("K", 1e3), ("M", 1e6), ("G", 1e9), ("T", 1e12)):
        if unit.startswith(prefix) and unit[len(prefix) :].startswith(("byte", "hz", "Hz")):
            return value * factor
    return value


def product(*values):
    return math.prod(values) if all(v is not None for v in values) else None


def total(values):
    values = list(values)
    return sum(values) if values and all(v is not None for v in values) else None


def divide(a, b):
    return a / b if a is not None and b is not None and b > 0 else None


def instances(value):
    return {
        name.strip(): int(count.replace(",", ""))
        for name, count in re.findall(r"([^;()]+):\s*([\d,]+)", value.partition(" (")[2])
    }


def row_key(values):
    return tuple(values.get(key) for key in ("ID", "Kernel Name", "Context", "Stream"))


def analyse(path, instance_directory=None):
    rows, units = read_raw(path)
    instance_path = path.with_name(path.stem + "-instances.csv")
    if not instance_path.exists() and instance_directory is not None:
        instance_path = instance_directory / instance_path.name
    instance_rows = read_raw(instance_path)[0] if instance_path.exists() else []
    indexed = {row_key(row): row for row in instance_rows}
    if len(indexed) != len(instance_rows):
        raise ValueError(f"Duplicate NCU kernel identity in {instance_path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    profiles = []
    for ordinal, values in enumerate(rows):

        def n(key, values=values):
            return number(values, units, key)

        ivals = indexed.get(row_key(values), {})
        ops = instances(ivals.get("sass__thread_inst_executed_true_per_opcode", ""))
        categories = instances(ivals.get("sass__thread_inst_executed_per_opcode_category", ""))
        duration = n("gpu__time_duration.sum")
        sm_clock = n("sm__cycles_elapsed.avg.per_second")
        smsp_clock = n("smsp__cycles_elapsed.avg.per_second")
        rates = {
            "L1": n("derived__l1tex__lsu_writeback_bytes_mem_lgds.sum.per_second"),
            "L2": n("derived__lts__lts2xbar_bytes.sum.per_second"),
            "DRAM": n("dram__bytes.sum.per_second"),
        }
        ceilings = {
            "L1": product(
                n("l1tex__lsu_writeback_active_mem_lgds.sum.peak_sustained"),
                128,
                n("l1tex__cycles_elapsed.avg.per_second"),
            ),
            "L2": product(
                n("lts__lts2xbar_cycles_active.sum.peak_sustained"),
                32,
                n("lts__cycles_elapsed.avg.per_second"),
            ),
            "DRAM": product(
                n("dram__bytes.sum.peak_sustained"),
                n("dram__cycles_elapsed.avg.per_second"),
            ),
        }
        exact = {}
        exact_source = {}
        for level, metric in (
            ("L1", "l1tex__t_bytes.sum"),
            ("L2", "lts__t_bytes.sum"),
            ("DRAM", "dram__bytes.sum"),
        ):
            value = n(metric)
            definition = metric
            if value is None and level != "DRAM":
                metric = {"L1": "l1tex__t_sectors.sum", "L2": "lts__t_sectors.sum"}[level]
                value, definition = product(n(metric), 32), "32 bytes * " + metric
            if value is None and level == "DRAM":
                value = total([n("dram__bytes_read.sum"), n("dram__bytes_write.sum")])
                definition = "dram__bytes_read.sum + dram__bytes_write.sum"
            exact[level] = value
            exact_source[level] = definition if value is not None else None
        proxy_bytes = {}
        proxy_source = {}
        for level, metric, multiplier in (
            ("L1", "l1tex__lsu_writeback_active_mem_lgds.sum", 128),
            ("L2", "lts__lts2xbar_cycles_active.sum", 32),
            ("DRAM", "dram__bytes.sum", 1),
        ):
            value = product(n(metric), multiplier)
            definition = f"{multiplier} * {metric}"
            if value is None:
                value = product(rates[level], duration)
                definition = "NCU hierarchical section rate * profiled kernel duration"
            proxy_bytes[level] = value
            proxy_source[level] = definition if value is not None else None
        precision = {}
        for name, prefix in (("fp32", "f"), ("fp64", "d")):
            sum_metrics = [
                f"smsp__sass_thread_inst_executed_op_{prefix}{op}_pred_on.sum"
                for op in ("add", "mul", "fma")
            ]
            counter_rates = [n(metric + ".per_cycle_elapsed") for metric in sum_metrics]
            official_rate = product(
                total(product(v, m) for v, m in zip(counter_rates, (1, 1, 2), strict=True)),
                smsp_clock,
            )
            flops = (
                sum(
                    ops.get((prefix + op).upper(), 0) * m
                    for op, m in (("add", 1), ("mul", 1), ("fma", 2))
                )
                if ops
                else None
            )
            flops_source = "predicated-thread SASS ADD + MUL + 2*FMA" if ops else None
            if flops is None:
                flops = total(
                    product(n(metric), m) for metric, m in zip(sum_metrics, (1, 1, 2), strict=True)
                )
                flops_source = (
                    "SM subpartition predicated-thread ADD + MUL + 2*FMA integer sums"
                    if flops is not None
                    else None
                )
            rate = divide(flops, duration) if flops is not None else official_rate
            peak = product(
                2,
                n(f"sm__sass_thread_inst_executed_op_{prefix}fma_pred_on.sum.peak_sustained"),
                sm_clock,
            )
            precision[name] = {
                "add_mul_fma_flops": flops,
                "flops_source": flops_source,
                "flops_per_second": rate,
                "official_section_flops_per_second": official_rate,
                "model_peak_flops_per_second": peak,
                "arithmetic_intensity_flops_per_byte": {
                    level: divide(flops, proxy_bytes[level])
                    if flops is not None
                    else divide(rate, rates[level])
                    for level in LEVELS
                },
                "arithmetic_intensity_counter_flops_per_byte": {
                    level: divide(flops, exact[level]) for level in LEVELS
                },
            }
        profiles.append(
            {
                "id": path.stem if len(rows) == 1 else f"{path.stem}:k{values['ID']}",
                "operation_id": path.stem,
                "ncu_kernel_id": values["ID"],
                "kernel_ordinal": ordinal,
                "kernel": values["Kernel Name"],
                "source_csv": str(path),
                "source_csv_sha256": digest,
                "instances_csv_sha256": hashlib.sha256(instance_path.read_bytes()).hexdigest()
                if instance_path.exists()
                else None,
                "device": values.get("device__attribute_display_name"),
                "compute_capability": values.get("CC"),
                "sm_count": n("device__attribute_multiprocessor_count"),
                "block": values.get("Block Size"),
                "grid": values.get("Grid Size"),
                "profiled_kernel_seconds": duration,
                "registers_per_thread": n("launch__registers_per_thread"),
                "occupancy_theoretical_percent": n("sm__maximum_warps_per_active_cycle_pct"),
                "occupancy_achieved_percent": n(
                    "sm__warps_active.avg.pct_of_peak_sustained_active"
                ),
                "active_threads_per_warp_instruction": n(
                    "smsp__thread_inst_executed_per_inst_executed.ratio"
                ),
                "predicated_on_threads_per_warp_instruction": n(
                    "smsp__thread_inst_executed_pred_on_per_inst_executed.ratio"
                ),
                "issue_active_percent": n("smsp__issue_active.avg.pct_of_peak_sustained_active"),
                "fp64_pipe_active_percent": n(
                    "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active"
                ),
                "fp64_pipe_elapsed_percent": n(
                    "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_elapsed"
                ),
                "throughput_percent_of_peak": {
                    "SM_elapsed": n("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                    "memory_elapsed": n(
                        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
                    ),
                    "DRAM_elapsed": n("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
                    "L2_elapsed": n("lts__throughput.avg.pct_of_peak_sustained_elapsed"),
                    "L1_active": n("l1tex__throughput.avg.pct_of_peak_sustained_active"),
                },
                "register_spill_instructions": n("sass__inst_executed_register_spilling"),
                "sm_clock_hz": sm_clock,
                "memory_clock_hz": n("dram__cycles_elapsed.avg.per_second"),
                "traffic_rate_bytes_per_second": rates,
                "traffic_bytes_from_rate_times_duration": {
                    level: product(rate, duration) for level, rate in rates.items()
                },
                "roofline_traffic_bytes": proxy_bytes,
                "roofline_traffic_source": proxy_source,
                "memory_counter_bytes": exact,
                "memory_counter_source": exact_source,
                "memory_integer_counters": {
                    key: n(key)
                    for key in values
                    if key.endswith(".sum")
                    and key.startswith(("l1tex__t_", "lts__t_", "dram__bytes"))
                },
                "bandwidth_model_peak_bytes_per_second": ceilings,
                "l1_sector_hit_percent": n("l1tex__t_sector_hit_rate.pct"),
                "l2_sector_hit_percent": n("lts__t_sector_hit_rate.pct"),
                "l2_lookup_miss_sectors": n("lts__t_sectors_lookup_miss.sum"),
                "precision": precision,
                "predicated_thread_opcode_counts": ops,
                "predicated_thread_category_counts": categories,
                "atomic_thread_instructions": {
                    "global_returning_ATOM_ATOMG": sum(ops.get(op, 0) for op in ("ATOM", "ATOMG"))
                    if ops
                    else None,
                    "global_reduction_RED_REDG": sum(ops.get(op, 0) for op in ("RED", "REDG"))
                    if ops
                    else None,
                    "shared_ATOMS_REDS": sum(ops.get(op, 0) for op in ("ATOMS", "REDS"))
                    if ops
                    else None,
                },
                "stall_warp_cycles_per_issued_warp_instruction": {
                    key.removeprefix("smsp__average_warps_issue_stalled_").removesuffix(
                        "_per_issue_active.ratio"
                    ): n(key)
                    for key in values
                    if key.startswith("smsp__average_warps_issue_stalled_")
                    and key.endswith("_per_issue_active.ratio")
                },
            }
        )
    return profiles


def operation_summary(operation, profiles, input_directory):
    metadata_path = input_directory / (operation + ".metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    return {
        "id": operation,
        "kernel_count": len(profiles),
        "kernels": [p["id"] for p in profiles],
        "capture_metadata": metadata,
        "scope": "complete selected NVTX operation"
        if metadata
        else "legacy main-kernel capture; additional kernels excluded",
        "sum_profiled_kernel_seconds": total(p["profiled_kernel_seconds"] for p in profiles),
        "time_limit": (
            "Sum of replay-profiled kernels; excludes host gaps and is not operation latency."
        ),
        "flops": {
            precision: total(p["precision"][precision]["add_mul_fma_flops"] for p in profiles)
            for precision in ("fp32", "fp64")
        },
        "memory_counter_bytes": {
            level: total(p["memory_counter_bytes"][level] for p in profiles) for level in LEVELS
        },
        "roofline_traffic_bytes": {
            level: total(p["roofline_traffic_bytes"][level] for p in profiles) for level in LEVELS
        },
        "kernels_missing_exact_memory_counters": {
            level: sum(p["memory_counter_bytes"][level] is None for p in profiles)
            for level in LEVELS
        },
    }


def profile_label(profile):
    text = LABELS.get(profile["operation_id"], profile["operation_id"])
    return (
        text
        if profile["id"] == profile["operation_id"]
        else text + f" · kernel {profile['kernel_ordinal'] + 1}"
    )


def roofline(profiles, precision, out, title):
    points = [
        (
            p["precision"][precision]["arithmetic_intensity_flops_per_byte"][level],
            p["precision"][precision]["flops_per_second"],
            p,
            level,
        )
        for p in profiles
        for level in LEVELS
    ]
    points = [
        (x, y / 1e9, p, level)
        for x, y, p, level in points
        if x is not None and y is not None and x > 0 and y > 0
    ]
    reference = next(
        (
            p
            for p in profiles
            if p["precision"][precision]["model_peak_flops_per_second"]
            and all(p["bandwidth_model_peak_bytes_per_second"].values())
        ),
        None,
    )
    if not points or reference is None:
        return {
            "created": False,
            "reason": "No positive measured points or complete model ceilings",
        }
    fig, ax = plt.subplots(figsize=(9.2, 6.3))
    xmin = 10 ** np.floor(np.log10(min(x for x, *_ in points)) - 0.4)
    xmax = max(1e3, 10 ** np.ceil(np.log10(max(x for x, *_ in points)) + 0.3))
    x = np.geomspace(xmin, xmax, 500)
    peak = reference["precision"][precision]["model_peak_flops_per_second"] / 1e9
    for level, style in (("L1", ":"), ("L2", "--"), ("DRAM", "-")):
        bandwidth = reference["bandwidth_model_peak_bytes_per_second"][level]
        ax.plot(x, np.minimum(peak, x * bandwidth / 1e9), color="black", ls=style, lw=1)
    handles = []
    operation_order = list(dict.fromkeys(p["operation_id"] for p in profiles))
    for p in profiles:
        selected = [(x, y, level) for x, y, record, level in points if record["id"] == p["id"]]
        if not selected:
            continue
        color = COLORS[operation_order.index(p["operation_id"]) % len(COLORS)]
        alpha = max(0.45, 1 - p["kernel_ordinal"] * 0.12)
        ax.plot(
            [v[0] for v in selected],
            [v[1] for v in selected],
            color=color,
            lw=0.65,
            alpha=alpha,
        )
        for px, py, level in selected:
            ax.scatter(
                px,
                py,
                marker={"L1": "o", "L2": "^", "DRAM": "s"}[level],
                s=35,
                color=color,
                alpha=alpha,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
        handles.append(Line2D([], [], color=color, alpha=alpha, lw=2, label=profile_label(p)))
    handles += [
        Line2D(
            [],
            [],
            color="black",
            marker=marker,
            ls=style,
            markersize=4,
            label=level + " roof / traffic",
        )
        for level, marker, style in (
            ("L1", "o", ":"),
            ("L2", "^", "--"),
            ("DRAM", "s", "-"),
        )
    ]
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(xmin, xmax),
        ylim=(
            min(y for _, y, *_ in points) / 2,
            max(peak, max(y for _, y, *_ in points)) * 2.2,
        ),
        xlabel=f"{precision.upper()} arithmetic intensity (FLOP/byte)",
        ylabel=f"{precision.upper()} arithmetic throughput (GFLOP/s)",
    )
    ax.set_title(
        f"{'Single' if precision == 'fp32' else 'Double'} precision instructions · {title}",
        loc="left",
        fontsize=12,
        weight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", which="major", color="#e5e5e5", lw=0.5)
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False, ncol=2)
    fig.text(
        0.12,
        0.067,
        f"{reference['device']} · native chest · AP 512 × 512 · 1024 midpoints",  # noqa: RUF001
        fontsize=8,
        ha="left",
    )
    fig.text(
        0.12,
        0.044,
        (
            "Roofs: NVIDIA theoretical peak-sustained model at recorded "
            "clocks. No measured sustained ceiling."
        ),
        fontsize=8,
        ha="left",
    )
    fig.text(
        0.12,
        0.021,
        (
            "L1/L2 points use the official writeback proxies. Exact cache "
            "request counters are reported separately."
        ),
        fontsize=8,
        ha="left",
    )
    fig.subplots_adjust(bottom=0.20, top=0.91, left=0.12, right=0.98)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(out / ("roofline_" + precision + "." + suffix), dpi=300, facecolor="white")
    plt.close(fig)
    return {
        "created": True,
        "plotted_kernel_ids": list(dict.fromkeys(p["id"] for _, _, p, _ in points)),
        "ceiling_reference_kernel": reference["id"],
    }


def display(value, scale=1, decimals=3):
    return "unavailable" if value is None else f"{value * scale:.{decimals}f}"


def write_tables(profiles, operations, output, title):
    fields = [
        "profile",
        "operation",
        "kernel",
        "kernel_ms",
        "registers",
        "occupancy_pct",
        "fp64_pipe_active_pct",
        "predicated_lanes",
        "fp32_GFLOP",
        "fp64_GFLOP",
        "L1_roofline_GB",
        "L2_roofline_GB",
        "DRAM_roofline_GB",
        "L1_counter_GB",
        "L2_counter_GB",
        "DRAM_counter_GB",
        "conversion_thread_instructions",
        "global_atomic_thread_instructions",
        "global_reduction_thread_instructions",
    ]
    with (output / "profiles.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for p in profiles:
            values = [
                p["id"],
                p["operation_id"],
                p["kernel"],
                product(1e3, p["profiled_kernel_seconds"]),
                p["registers_per_thread"],
                p["occupancy_achieved_percent"],
                p["fp64_pipe_active_percent"],
                p["predicated_on_threads_per_warp_instruction"],
                *[product(1e-9, p["precision"][k]["add_mul_fma_flops"]) for k in ("fp32", "fp64")],
                *[product(1e-9, p["roofline_traffic_bytes"][k]) for k in LEVELS],
                *[product(1e-9, p["memory_counter_bytes"][k]) for k in LEVELS],
                p["predicated_thread_category_counts"].get("Conversion", 0)
                if p["predicated_thread_category_counts"]
                else None,
                p["atomic_thread_instructions"]["global_returning_ATOM_ATOMG"],
                p["atomic_thread_instructions"]["global_reduction_RED_REDG"],
            ]
            writer.writerow(dict(zip(fields, values, strict=True)))
    lines = [
        f"# {title} kernel profiles",
        "",
        "Replay-profiled kernel times are not complete-call benchmark latencies.",
        "",
        "| Operation | Scope | Captured kernels | Sum of replay kernel times (ms) |",
        "|---|---|---:|---:|",
    ]
    for op in operations:
        lines.append(
            f"| {LABELS.get(op['id'], op['id'])} | {op['scope']} | {op['kernel_count']} | "
            f"{display(op['sum_profiled_kernel_seconds'], 1e3)} |"
        )
    lines += [
        "",
        (
            "| Kernel | Profile time (ms) | Registers/thread | Occupancy (%) "
            "| FP64 pipe active (%) | Predicated lanes/32 |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for p in profiles:
        values = [
            display(p["profiled_kernel_seconds"], 1e3, 6),
            display(p["registers_per_thread"], decimals=0),
            display(p["occupancy_achieved_percent"]),
            display(p["fp64_pipe_active_percent"]),
            display(p["predicated_on_threads_per_warp_instruction"]),
        ]
        lines.append("| " + profile_label(p) + " | " + " | ".join(values) + " |")
    lines += [
        "",
        (
            "Exact cache/request bytes are distinct from the L1/L2 writeback "
            "proxies used by NVIDIA hierarchical roofline. Missing counters "
            "are unavailable; integer sector sums use 32 bytes per sector."
        ),
        "",
        (
            "| Kernel | FP32 GFLOP | FP64 GFLOP | L1 counter GB | L2 counter "
            "GB | DRAM counter GB | L1 roofline GB | L2 roofline GB | DRAM "
            "roofline GB |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in profiles:
        values = [p["precision"][k]["add_mul_fma_flops"] for k in ("fp32", "fp64")]
        values += [
            p[group][level]
            for group in ("memory_counter_bytes", "roofline_traffic_bytes")
            for level in LEVELS
        ]
        lines.append(
            "| " + profile_label(p) + " | " + " | ".join(display(v, 1e-9, 6) for v in values) + " |"
        )
    (output / "profiles.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--instances",
        type=Path,
        help="Optional separate directory for historical instance exports",
    )
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--title", default="DPT")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(args.font)
    bold = args.font.with_name("Geist-Bold.ttf")
    if bold.exists():
        font_manager.fontManager.addfont(bold)
    plt.rcParams.update(
        {
            "font.family": "Geist",
            "font.size": 10,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    profiles, operations = [], []
    for name in LABELS:
        path = args.input / (name + ".csv")
        if not path.exists():
            continue
        records = analyse(path, args.instances or args.output)
        profiles.extend(records)
        operations.append(operation_summary(name, records, args.input))
    if not profiles:
        raise ValueError("No supported NCU operation CSVs found")
    completed = args.input / "completed-at.txt"
    data = {
        "schema_version": 2,
        "analysis_created_at": datetime.now(UTC).isoformat(),
        "measurement_completed_at": completed.read_text().strip() if completed.exists() else None,
        "timing_boundary": (
            "Every kernel row from each supplied capture; replay-profiled "
            "duration is not clean latency. See per-operation capture scope."
        ),
        "flops_definition": (
            "Predicated-thread ADD + MUL + 2*FMA; excludes comparisons, "
            "conversions, divide/sqrt helpers, atomics and integer/address "
            "work."
        ),
        "traffic_definition": (
            "Exact cache request/sector byte counters are reported "
            "separately. Hierarchical roofs use official L1 "
            "global/local/dshared writeback and L2-to-XBAR writeback proxies; "
            "these are derived from counters where available, else rates "
            "times kernel duration. DRAM counts read+write."
        ),
        "ceiling_definition": (
            "NVIDIA theoretical peak_sustained model counts times recorded "
            "clocks. No independently measured sustained ceiling."
        ),
        "missing_counter_policy": (
            "null, never an estimated zero; operation sums are null if any constituent is absent"
        ),
        "operations": operations,
        "profiles": profiles,
        "figures": {
            precision: roofline(profiles, precision, args.output, args.title)
            for precision in ("fp32", "fp64")
        },
    }
    (args.output / "profiles.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    write_tables(profiles, operations, args.output, args.title)
    print(f"Analysed {len(profiles)} kernels in {len(operations)} captures: {args.output}")


if __name__ == "__main__":
    main()
