"""Insert the completed full-scan report into the README and manual."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from update_performance_docs import table


def replace_block(path, marker, text):
    opening, closing = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
    original = path.read_text()
    if opening not in original and closing not in original:
        original += f"\n\n{opening}\n\n{closing}\n"
    if original.count(opening) != 1 or original.count(closing) != 1:
        raise ValueError(f"Ambiguous generated block in {path}")
    prefix, remainder = original.split(opening)
    _, suffix = remainder.split(closing)
    path.write_text(prefix + opening + "\n\n" + text.strip() + "\n\n" + closing + suffix)


def profiling_findings(profiles):
    def value(number):
        return "unavailable" if number is None else f"{number:,.6g}"

    def label(workload):
        if workload.get("operation") == "forward":
            return "forward"
        gradients = workload.get("requested_gradients")
        if gradients == ["volume"]:
            return "volume-only VJP"
        if gradients == ["volume", "pose"]:
            return "joint volume/pose VJP"
        return "backward, gradients unrecorded"

    operations = {row["id"]: row for row in profiles["operations"]}
    if len(operations) != len(profiles["operations"]):
        raise ValueError("Duplicate profile operation identity")
    counters = []
    for row in profiles["profiles"]:
        operation = operations[row["operation_id"]]
        if operation["role"] != "final" or not row["plotted_hot_kernel"]:
            continue
        workload = operation["workload"]
        if row.get("requested_gradients") != workload.get("requested_gradients"):
            raise ValueError("Kernel and capture gradient metadata differ")
        counters.append(
            [
                row["operation_id"] + f" / k{row['ncu_kernel_id']}",
                workload["precision"],
                label(workload),
                workload["views"],
                value(row.get("registers_per_thread")),
                value(row.get("register_spill_instructions")),
                value(row.get("additional_counters", {}).get("local_memory_sector_bytes")),
                value(row.get("fp64_pipe_active_percent")),
                value(row.get("throughput_percent_of_peak", {}).get("L2_elapsed")),
                value(row.get("throughput_percent_of_peak", {}).get("DRAM_elapsed")),
                value(row.get("atomic_thread_instructions", {}).get("global_reduction_RED_REDG")),
            ]
        )
    if not counters:
        raise ValueError("No final main-kernel counters were supplied")
    for row in profiles.get("comparisons", []):
        before, after = operations[row["before"]], operations[row["after"]]
        if after["workload"].get("requested_gradients") != ["volume"]:
            continue
        if not (
            before["role"] == "before"
            and after["role"] == "final"
            and before["comparison_signature"] is not None
            and before["comparison_signature"] == after["comparison_signature"]
            and before["workload"].get("requested_gradients") == ["volume"]
        ):
            raise ValueError("Volume counter comparison does not match the measured workload")
    lines = [
        "### Kernel resources\n",
        "Each row records the requested gradients, view count and main-kernel counters.\n",
        table(
            [
                "Capture / kernel",
                "Precision",
                "Operation",
                "Views",
                "Registers/thread",
                "Spill instructions",
                "Local sector bytes",
                "FP64 pipe active %",
                "L2 model %",
                "DRAM model %",
                "Global RED/REDG instructions",
            ],
            counters,
        ),
    ]
    lines += [
        "FLOPs count ADD + MUL + 2 * FMA. Conversion and global reduction "
        "instructions have their own counters. Local-memory traffic and register "
        "spills are reported individually.\n",
    ]
    traces = []
    for trace in profiles.get("timelines", []):
        if trace["role"] != "final":
            continue
        for measured in trace["ranges"]:
            operation = "forward" if measured["range"] == "probe_forward" else "backward"
            traces.append(
                [
                    trace["id"],
                    trace["precision"],
                    trace["views"],
                    label(
                        {
                            "operation": operation,
                            "requested_gradients": trace.get("requested_gradients"),
                        }
                    ),
                    value(measured.get("inclusive_nvtx_ms")),
                    value(measured.get("kernel_count")),
                    value(measured.get("full_volume_validation_count")),
                    value(measured.get("four_byte_d2h_count")),
                ]
            )
    if traces:
        lines += [
            "Nsight Systems records complete application ranges. The D2H column counts "
            "four-byte device-to-host copy events.\n",
            table(
                [
                    "Capture",
                    "Precision",
                    "Views",
                    "Operation",
                    "Profiled NVTX ms",
                    "Kernels",
                    "Volume scans",
                    "4-byte D2H copies",
                ],
                traces,
            ),
            "NVTX intervals, kernel sums and API waits overlap. Compute captures use one "
            "view; Systems captures use all 360 views.\n",
        ]
    return "\n\n".join(block.strip() for block in lines) + "\n"


def volume_report_text(directory, data):
    if directory is None:
        return ""
    from fullscan_volume_report import markdown

    volume = json.loads((directory / "results.json").read_text())
    candidate = data["dpt"]["candidate"]["fp32"]
    for measured in volume["dpt"]["candidate"].values():
        if measured["source"] != candidate["source"]:
            raise ValueError("Volume iteration source differs from the main timed implementation")
    for key in ("metadata_sha256", "metadata", "selected_view_indices", "selected_geometry"):
        if volume["input"][key] != data["input"][key]:
            raise ValueError("Volume iteration physical input differs: " + key)
    for key in ("sha256", "raw_sha256"):
        if volume["input"]["cotangent"][key] != data["input"]["cotangent"][key]:
            raise ValueError("Volume iteration cotangent differs")
    generated = markdown(volume)
    if (directory / "summary.md").read_text() != generated:
        raise ValueError("Volume iteration summary differs from its numeric report")
    generated = re.sub(r"^(#{1,2}) ", r"\1# ", generated, flags=re.MULTILINE)
    return (
        generated + "\n\n[Volume-only raw samples, checks and output identities]"
        "(../assets/results/fullscan/volume_iteration.json).\n\n"
    )


def validate_profile_binding(profiles, data):
    """Require final profiles to describe the timed package, field and seed."""
    sources = {data["dpt"]["candidate"][mode]["source"]["sha256"] for mode in ("fp32", "fp64")}
    final = [
        row
        for group in ("operations", "timelines")
        for row in profiles[group]
        if row["role"] == "final"
    ]
    if len(sources) != 1 or not final or any(row["source_sha256"] not in sources for row in final):
        raise ValueError("Nsight sources must match the timed implementation before writing docs")
    inputs = data["input"]
    samples = inputs["metadata"]["sampling"]["samples_per_ray"]
    for row in profiles["operations"]:
        if row["role"] != "final":
            continue
        workload = row["workload"]
        views = workload["views"]
        if not (
            workload["input_metadata_sha256"] == inputs["metadata_sha256"]
            and workload["cotangent_sha256"] == inputs["cotangent"]["sha256"]
            and workload["samples_per_ray"] == samples
            and views in (1, 16, 360)
            and workload["view_indices"] == [index * 360 // views for index in range(views)]
        ):
            raise ValueError("Final NCU workload differs from the timed field, seed or sampling")
    for row in profiles["timelines"]:
        if row["role"] != "final":
            continue
        views = row["views"]
        if not (
            row["input_metadata"] == inputs["metadata"]
            and row["cotangent"]["sha256"] == inputs["cotangent"]["sha256"]
            and views in (1, 16, 360)
            and row["view_indices"] == [index * 360 // views for index in range(views)]
        ):
            raise ValueError("Final NSYS workload differs from the timed field, seed or sampling")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--volume-report", type=Path, help="Completed volume-only iteration report")
    parser.add_argument("--support-report", type=Path, help="FP64 volume timing and counter report")
    parser.add_argument("--keep-readme", action="store_true")
    parser.add_argument("--wandb-run", help="URL of the completed measurement run")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    data = json.loads((args.report / "measurements.json").read_text())
    profiles = json.loads((args.profiles / "profile_report.json").read_text())
    validate_profile_binding(profiles, data)
    findings = profiling_findings(profiles)
    volume_text = volume_report_text(args.volume_report, data)
    support_text = ""
    if args.support_report:
        from fullscan_support_report import markdown as support_markdown

        support = json.loads((args.support_report / "support_checks.json").read_text())
        if support["candidate"]["source"] != data["dpt"]["candidate"]["fp64"]["source"]:
            raise ValueError("FP64 volume source differs from the main timed implementation")
        support_text = support_markdown(support)
    rows, dates = [], []
    for precision in ("fp32", "fp64"):
        entry = data["dpt"]["candidate"][precision]
        measurements = {row["label"]: row for row in entry["measurements"]}
        dates.extend((entry["started_at"], entry["completed_at"]))
        rows.append(
            [precision]
            + [
                f"{measurements[name]['wall_median_ms']:.2f} / "
                f"{measurements[name]['wall_p95_ms']:.2f}"
                for name in ("forward", "volume_vjp", "pose_vjp", "iteration")
            ]
        )
    intro = (
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition. Complete real chest CT "
        "resampled to 256³ with its physical extent preserved. The acquisition uses "
        "360 circular views, "
        "512² detectors and 1024 clipped midpoint samples per ray. Completed resident "
        "PyTorch calls, median / p95 milliseconds from three rounds of ten samples. "
        "The complete iteration requests both volume and pose gradients."
    )
    if not args.keep_readme:
        replace_block(
            args.root / "README.md",
            "release",
            "## Measured performance\n\n"
            + intro
            + "\n\n"
            + table(
                [
                    "Precision",
                    "Forward (ms)",
                    "Volume VJP (ms)",
                    "Pose VJP (ms)",
                    "Full iteration (ms)",
                ],
                rows,
            )
            + "\n\nSee [ASTRA/TIGRE comparisons, errors and Nsight profiles]"
            "(docs/performance/fullscan.md).",
        )
    profile_text = (args.profiles / "profile_summary.md").read_text()
    profile_text = profile_text.removeprefix("# Fullscan profiling evidence\n")
    profile_text = re.sub(r"^## ", "### ", profile_text, flags=re.MULTILINE)
    tables_text = (args.report / "tables.md").read_text().replace(
        "(competitor_reference.json)", "(../assets/results/fullscan/competitor_reference.json)"
    )
    details = (
        "## Measured results\n\n"
        + intro
        + f"\n\nMeasured: {min(dates)} to {max(dates)}.\n\n"
        + "[Numeric results and raw samples](../assets/results/fullscan/measurements.json) "
        "record source digests, physical settings, versions and hardware telemetry. "
        "NVML reports sampled process memory, including Warp allocations.\n\n"
        + "![Real resampled chest and computed projections]"
        "(../assets/figures/fullscan/fullscan_ct_projections.png)\n\n"
        + tables_text
        + "\n\n"
        + volume_text
        + "\n\n"
        + support_text
        + "\n\n![DPT completed-call latency]"
        "(../assets/figures/fullscan/fullscan-latency.png)\n\n"
        + "![Measured image error and forward latency]"
        "(../assets/figures/fullscan/fullscan-latency-error.png)\n\n"
        + "![FP32 minus FP64 optical depth]"
        "(../assets/figures/fullscan/fullscan_fp32_fp64_error.png)\n\n"
        + "![Volume sensitivity to the saved dense detector objective]"
        "(../assets/figures/fullscan/fullscan_volume_sensitivity.png)\n\n"
        + "## Nsight captures\n\n"
        + findings
        + profile_text
        + "\n\n![Single-precision hierarchical roofline]"
        "(../assets/figures/fullscan/roofline_fp32.png)\n\n"
        + "![Double-precision hierarchical roofline]"
        "(../assets/figures/fullscan/roofline_fp64.png)\n\n"
        + "[Numeric Nsight counters and capture settings]"
        "(../assets/results/fullscan/profile_report.json) accompany the figures. "
        "Raw profiler reports are stored separately."
    )
    if args.wandb_run:
        details += f"\n\n[Completed measurement run]({args.wandb_run})."
    replace_block(args.root / "docs/performance/fullscan.md", "fullscan-results", details)


if __name__ == "__main__":
    main()
