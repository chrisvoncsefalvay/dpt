"""Generate the strict volume-backward comparison from completed measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from fullscan_report import compact_measurement


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_run(directory):
    path = directory / "results.json"
    raw = json.loads(path.read_text())
    require(
        raw["status"] == "complete"
        and raw["source_unchanged"]
        and raw["source"] == raw["source_after"]
        and raw["precision"] == "fp64",
        "Expected a completed unchanged FP64 run",
    )
    require(
        raw["input"]["selected_view_indices"] == list(range(360))
        and raw["samples_per_ray"] == 1024
        and len(raw["measurements"]) == 1,
        "Workload differs from the complete rotation",
    )
    row = raw["measurements"][0]
    require(
        row["label"] == "volume_only_iteration"
        and row["configuration"]["requested_gradients"] == ["volume"]
        and row["repeats_per_round"] == 10
        and len(row["rounds"]) == 3,
        "Expected thirty completed volume-only iterations",
    )
    samples = [value for repeat in row["rounds"] for value in repeat["wall_ms"]]
    require(
        len(samples) == 30
        and all(math.isfinite(value) and value > 0 for value in samples)
        and math.isclose(statistics.median(samples), row["wall_median_ms"], rel_tol=1e-12),
        "Stored median does not match the raw samples",
    )
    for phase in ("before_timing", "after_timing"):
        check = raw["checks"][phase]
        require(check["passed"], "Numerical gate failed")
        require(
            check["volume_adjoint"]["passed"]
            and check["volume_adjoint"]["relative_discrepancy"] <= 2e-5,
            "Volume-adjoint gate failed",
        )
    return raw, hashlib.sha256(path.read_bytes()).hexdigest()


def collect(baseline, candidate, profiles):
    before, before_hash = read_run(baseline)
    after, after_hash = read_run(candidate)
    require(before["input"] == after["input"], "Physical inputs or saved seed differ")
    require(before["contract"] == after["contract"], "Frozen numerical contract differs")
    require(
        before["environment"]["gpu"] == after["environment"]["gpu"],
        "GPU models differ",
    )
    for phase in ("before_timing", "after_timing"):
        checks = after["checks"][phase]
        require(checks["images_vs_reference"]["bitwise_equal"], "Forward image changed")
        require(
            checks["volume_vjp_vs_reference"]["relative_l2"] <= 2e-4,
            "Volume-gradient error exceeds the fixed bound",
        )
    profile_path = profiles / "profile_report.json"
    profile = json.loads(profile_path.read_text())
    operations = {row["id"]: row for row in profile["operations"]}
    counters = {}
    for row in profile["profiles"]:
        operation = operations[row["operation_id"]]
        workload = operation["workload"]
        role = operation["role"]
        if (
            role in ("before", "final")
            and row["plotted_hot_kernel"]
            and workload["precision"] == "fp64"
            and workload["requested_gradients"] == ["volume"]
        ):
            require(role not in counters, "Ambiguous strict volume profile")
            expected = before if role == "before" else after
            require(
                operation["source_sha256"] == expected["source"]["sha256"],
                "Profiled source differs from the timed source",
            )
            counters[role] = row
    require(set(counters) == {"before", "final"}, "Both matched strict profiles are required")
    first = operations[counters["before"]["operation_id"]]
    last = operations[counters["final"]["operation_id"]]
    require(
        first["comparison_signature"]
        and first["comparison_signature"] == last["comparison_signature"],
        "Profile rays, seed, sampling or capture settings differ",
    )
    return {
        "baseline": before,
        "candidate": after,
        "result_sha256": {"baseline": before_hash, "candidate": after_hash},
        "profile_report_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "counters": counters,
    }


def markdown(data):
    candidate = data["candidate"]
    measured = candidate["measurements"][0]
    error = candidate["checks"]["after_timing"]
    lines = [
        "## FP64 forward and volume backward",
        "",
        "Completed resident forward plus volume-only backward uses the full chest "
        "resampled to 256³, 360 views, 512² detectors and 1024 midpoint samples. "
        "The gradient destination is FP32. Three rounds of ten repetitions run outside Nsight.",
        "",
        "| Operation | Median (ms) | p95 (ms) |",
        "|---|---:|---:|",
        f"| Forward + volume backward | {measured['wall_median_ms']:.2f} | "
        f"{measured['wall_p95_ms']:.2f} |",
        "",
        f"Volume-gradient relative L2 error against the recorded same-mode reference is "
        f"{error['volume_vjp_vs_reference']['relative_l2']:.3g}. "
        f"The adjoint discrepancy is {error['volume_adjoint']['relative_discrepancy']:.3g}.",
        "",
        f"Measured {candidate['started_at']} to {candidate['completed_at']}.",
        "",
        "The one-view Nsight Compute capture records the following main-kernel counters.",
        "",
        "| Main volume kernel counter | Value |",
        "|---|---:|",
    ]
    counters = data["counters"]["final"]
    for opcode in ("DSETP", "FSETP", "REDG"):
        value = counters["predicated_thread_opcode_counts"].get(opcode)
        label = "unavailable" if value is None else f"{value:,}"
        lines.append(f"| {opcode} thread instructions | {label} |")
    for key, label in (
        ("registers_per_thread", "Registers per thread"),
        ("register_spill_instructions", "Spill instructions"),
    ):
        value = counters.get(key)
        displayed = "unavailable" if value is None else f"{value:g}"
        lines.append(f"| {label} | {displayed} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.baseline, args.candidate, args.profiles)
    for role in ("baseline", "candidate"):
        data[role]["measurements"] = [
            compact_measurement(row) for row in data[role]["measurements"]
        ]
    data["telemetry_storage"] = (
        "All timing samples remain here. Full NVML sample streams remain in the "
        "original hashed result files, which are stored separately."
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "support_checks.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.output / "support_checks.md").write_text(markdown(data))


if __name__ == "__main__":
    main()
