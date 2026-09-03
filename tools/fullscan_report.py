# ruff: noqa: RUF001
# Scientific multiplication signs are intentional in rendered tables.
"""Generate full-scan tables and figures from completed measured evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from plot_baseline import configure_style, plt, save_figure
from plot_fullscan_images import load_run
from update_performance_docs import percentile, table

OPERATIONS = ("forward", "volume_vjp", "pose_vjp", "iteration")
ROLE_LABELS = {"baseline": "Reference", "candidate": "DPT"}
LABELS = {
    "forward": "Forward",
    "volume_vjp": "Volume VJP",
    "pose_vjp": "Pose VJP",
    "iteration": "Forward + volume/pose VJP",
}
PRECISION_METHOD = "untimed FP64 reductions over the selected saved FP32 and FP64 images"


def read(path):
    value = json.loads(path.read_text())
    return value, hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_interval(start, end):
    first, last = (datetime.fromisoformat(value) for value in (start, end))
    require(
        first.utcoffset() == last.utcoffset() == timedelta(0) and last >= first,
        "measurement dates must be ordered UTC timestamps",
    )


def nvml_summary(value):
    samples = value["samples"]
    result = {key: v for key, v in value.items() if key != "samples"}
    result["sample_count"] = len(samples)
    result["counters"] = {}
    for name in (
        "gpu_utilisation_percent",
        "memory_utilisation_percent",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "power_watts",
        "power_limit_watts",
    ):
        observed = [sample[name] for sample in samples if sample.get(name) is not None]
        result["counters"][name] = (
            {"min": min(observed), "max": max(observed), "mean": statistics.fmean(observed)}
            if observed
            else None
        )
    result["observed_compute_pids"] = sorted(
        {p["pid"] for sample in samples for p in sample.get("compute_processes", [])}
    )
    result["counter_errors"] = list(
        {str(sample["errors"]) for sample in samples if sample.get("errors")}
    )
    return result


def compact_measurement(row):
    result = {key: value for key, value in row.items() if key != "rounds"}
    result["rounds"] = []
    for measured in row["rounds"]:
        utc_interval(measured["measured_at"], measured["finished_at"])
        record = {key: value for key, value in measured.items() if key != "nvml"}
        record["nvml"] = nvml_summary(measured["nvml"])
        result["rounds"].append(record)
    require(len(result["rounds"]) == 3, "headline measurement needs three actual rounds")
    require(
        [row["round"] for row in result["rounds"]] == [0, 1, 2],
        "measurement rounds are missing or duplicated",
    )
    repeats = row["repeats_per_round"]
    require(type(repeats) is int and repeats > 0, "invalid declared repetition count")
    require(
        all(
            isinstance(r["wall_ms"], list) and len(r["wall_ms"]) == repeats
            for r in result["rounds"]
        ),
        "each wall-time round must contain its full declared sample count",
    )
    if all(r["device_ms"] is None for r in result["rounds"]):
        require(
            row["device_median_ms"] is None and row["device_p95_ms"] is None,
            "unavailable device samples require unavailable summaries",
        )
    else:
        require(
            all(
                isinstance(r["device_ms"], list) and len(r["device_ms"]) == repeats
                for r in result["rounds"]
            ),
            "device samples must be complete in every round or null throughout",
        )
    for boundary in ("wall", "device"):
        key = boundary + "_ms"
        samples = [sample for r in result["rounds"] for sample in (r.get(key) or [])]
        if not samples:
            continue
        require(all(math.isfinite(x) and x > 0 for x in samples), "invalid timing sample")
        for suffix, expected in (
            ("median_ms", statistics.median(samples)),
            ("p95_ms", percentile(samples, 0.95)),
        ):
            name = boundary + "_" + suffix
            require(math.isclose(row[name], expected, rel_tol=1e-12), name + " differs from raw")
    return result


def collect_quality(path, data):
    raw, digest = read(path / "results.json")
    candidate = data["dpt"]["candidate"]["fp32"]
    require(
        raw["status"] == "complete"
        and raw["source_unchanged"]
        and raw["harness_unchanged"]
        and raw["source"] == raw["source_after"] == candidate["source"]
        and raw["precision"] == "fp32"
        and not raw.get("error"),
        "sampling sweep must be complete and use the selected FP32 source",
    )
    utc_interval(raw["started_at"], raw["completed_at"])
    for key in ("sha256", "settings"):
        require(raw["contract"][key] == candidate["contract"][key], "sweep contract differs")
    for key in ("metadata_sha256", "metadata", "selected_view_indices", "selected_geometry"):
        require(raw["input"][key] == data["input"][key], "sweep input/rays differ: " + key)
    contract = candidate["contract"]["settings"]
    require(
        raw["harness"]["fullscan_inputs.py"] == contract["input_helper_sha256"],
        "sweep input helper differs",
    )
    reference = raw["reference"]
    reference_roles = [
        role
        for role in ("baseline", "candidate")
        if reference["results_sha256"] == data["evidence"][role + "_fp64"]["sha256"]
    ]
    require(len(reference_roles) == 1, "sweep must identify one measured FP64 reference")
    reference_role = reference_roles[0]
    reference_run = data["dpt"][reference_role]["fp64"]
    reference_evidence = data["evidence"][reference_role + "_fp64"]
    require(
        reference["outputs_sha256"] == reference_evidence["outputs_sha256"]
        and reference["source_sha256"] == reference_run["source"]["sha256"]
        and reference["completed_at"] == reference_run["completed_at"]
        and reference["samples_per_ray"] == 1024
        and reference["images"] == reference_run["outputs"]["arrays"]["images"],
        "sweep reference differs from the measured FP64/1024 full scan",
    )
    require(
        sorted(raw["requested_samples_per_ray"]) == [128, 256, 512, 1024]
        and len(raw["measurements"]) == 4,
        "sampling sweep requires all four distinct sample counts",
    )
    counts = []
    rows = []
    for row in raw["measurements"]:
        config = row["configuration"]
        count = config["samples_per_ray"]
        counts.append(count)
        require(row["label"] == f"fp32/{count}-samples/forward", "sweep label differs")
        expected = {
            "operation": "forward",
            "precision": "fp32",
            "field_dtype": "float32",
            "output_dtype": "float32",
            "requested_gradients": [],
            "pose": "identity fixed geometry",
            "views": 360,
            "detector_shape": [512, 512],
            "volume_shape_zyx": [256, 256, 256],
            "input_residency": "CUDA",
            "interpolation": contract["sampling"]["interpolation"],
            "wall_boundary": contract["operations"]["wall_boundary"],
            "event_boundary": contract["operations"]["event_boundary"],
        }
        require(
            all(config.get(key) == value for key, value in expected.items()),
            "sweep operation or timing boundary differs",
        )
        check = raw["checks"][row["label"]]
        errors = check["error_vs_fp64_1024"]
        require(
            errors == row["error_vs_fp64_1024"] == config["error_vs_fp64_1024"]
            and all(
                math.isfinite(errors[key]) and errors[key] >= 0
                for key in ("max_abs", "rmse", "relative_l2")
            )
            and check["after_timing"]["bitwise_unchanged"]
            and check["raw_sha256"] == check["after_timing"]["raw_sha256"],
            "sampling image checks differ or failed",
        )
        gate = check["arithmetic_accuracy_gate"]
        if count == 1024:
            limits = contract["quality"]["image_fp32_vs_fp64"]
            require(
                gate["passed"]
                and gate["thresholds"] == limits
                and all(errors[key] <= limits[key] for key in ("max_abs", "rmse", "relative_l2"))
                and check["raw_sha256"] == candidate["outputs"]["arrays"]["images"]["raw_sha256"],
                "1024-sample accuracy or same-mode image agreement failed",
            )
        else:
            require(gate is None, "coarse sampling must not claim the arithmetic-only gate")
        require(
            row["repeats_per_round"] == 10
            and all(len(r[key]) == 10 for r in row["rounds"] for key in ("wall_ms", "device_ms")),
            "sweep requires 3 x 10 samples",
        )
        rows.append(compact_measurement(row))
    require(sorted(counts) == [128, 256, 512, 1024], "duplicate or missing sample count")
    return {
        **{key: value for key, value in raw.items() if key != "measurements"},
        "raw_result_sha256": digest,
        "measurements": rows,
    }


def collect_cold(path, data):
    raw, digest = read(path)
    candidate = data["dpt"]["candidate"]
    require(
        raw["status"] == "complete"
        and raw["rounds_per_mode"] == 3
        and raw["instrumented_launches"] is False,
        "cold measurements incomplete",
    )
    utc_interval(raw["measurement_started_at"], raw["measurement_finished_at"])
    require(
        raw["source"] == candidate["fp32"]["source"] == candidate["fp64"]["source"]
        and raw["contract"] == candidate["fp32"]["contract"]
        and raw["input"] == {key: data["input"][key] for key in ("metadata_sha256", "metadata")},
        "cold source, contract or input differs",
    )
    require(
        raw["harness"]["fullscan_inputs.py"] == raw["contract"]["settings"]["input_helper_sha256"],
        "cold input helper differs",
    )
    references = {}
    for mode, reference in raw["references"].items():
        require(mode in ("fp32", "fp64"), "unknown cold precision")
        matches = [
            role
            for role in ("baseline", "candidate")
            if reference["result_sha256"] == data["evidence"][f"{role}_{mode}"]["sha256"]
        ]
        require(len(matches) == 1, "cold reference is not a measured same-mode full scan")
        role = matches[0]
        run = data["dpt"][role][mode]
        require(
            reference["source_sha256"] == run["source"]["sha256"]
            and reference["manifest_sha256"] == data["evidence"][f"{role}_{mode}"]["outputs_sha256"]
            and reference["record"] == run["outputs"]["arrays"]["images"]
            and reference["measurement_completed_at"] == run["completed_at"]
            and reference["source_relation"]
            == ("same_source" if run["source"] == raw["source"] else "cross_source_same_mode"),
            "cold reference binding differs",
        )
        references[mode] = {key: value for key, value in reference.items() if key != "file"}
    seen, rows, caches = set(), [], {}
    for row in raw["runs"]:
        mode, sample, state = row["precision"], row["sample"], row["cache_state"]
        key = (mode, sample, state)
        require(
            key not in seen
            and mode in references
            and sample in range(3)
            and state in ("cold", "warm_process")
            and row["returncode"] == 0,
            "invalid or duplicate cold process",
        )
        seen.add(key)
        utc_interval(row["started_at"], row["completed_at"])
        child = row["measurement"]
        utc_interval(child["measurement_started_at"], child["measurement_finished_at"])
        require(
            child["status"] == "passed"
            and child["source"] == raw["source"]
            and child["source_expected"] == raw["source"]["sha256"]
            and child["input"] == raw["input"]
            and child["reference"] == raw["references"][mode]
            and (child["precision"], child["sample"], child["cache_state"]) == key,
            "cold child source/input/result differs",
        )
        for output in child["output"].values():
            require(
                output["passed"]
                and output["error_vs_reference"]["bitwise_equal"]
                and output["raw_sha256"] == references[mode]["record"]["raw_sha256"]
                and output["shape"] == [360, 512, 512]
                and output["dtype"] == references[mode]["record"]["dtype"],
                "cold first/warm image differs",
            )
        require(set(child["output"]) == {"first", "last_warm"}, "cold output check missing")
        require(
            all(math.isfinite(value) and value >= 0 for value in child["seconds"].values())
            and math.isfinite(child["first_device_image_seconds_since_child_entry"])
            and child["first_device_image_seconds_since_child_entry"] > 0,
            "invalid cold phase duration",
        )
        warm = child["warm"]
        require(warm["repeats"] == len(warm["samples"]) == 3, "three warm samples required")
        for index, timing in enumerate(warm["samples"]):
            require(timing["repeat"] == index, "warm sample order differs")
            utc_interval(timing["started_at"], timing["completed_at"])
        for boundary in ("wall", "device"):
            values = [timing[boundary + "_ms"] for timing in warm["samples"]]
            require(
                values == warm[boundary + "_ms"]
                and all(math.isfinite(value) and value > 0 for value in values),
                "warm raw samples differ",
            )
            require(
                math.isclose(
                    warm[boundary + "_median_ms"], statistics.median(values), rel_tol=1e-12
                )
                and math.isclose(
                    warm[boundary + "_p95_ms"], percentile(values, 0.95), rel_tol=1e-12
                ),
                "warm summary differs from raw",
            )
        cache = {name: child["cache"][name] for name in ("warp", "cuda", "inductor", "triton")}
        if state == "cold":
            require(not any(row["cache_files_before"].values()), "cold cache was populated")
            require(
                all(cache != previous for previous in caches.values()), "cold rounds reused a cache"
            )
            caches[(mode, sample)] = cache
        else:
            prior = next(
                r
                for r in raw["runs"]
                if (r["precision"], r["sample"], r["cache_state"]) == (mode, sample, "cold")
            )
            require(
                cache == caches[(mode, sample)]
                and row["cache_files_before"] == prior["cache_files_after"],
                "second process did not reuse its paired cache",
            )
        compact = {key: value for key, value in child.items() if key not in ("cache", "reference")}
        compact["reference"] = references[mode]
        rows.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in ("command", "measurement")
                },
                "measurement": compact,
            }
        )
    require(
        seen
        == {
            (mode, sample, state)
            for mode in references
            for sample in range(3)
            for state in ("cold", "warm_process")
        },
        "missing cold cache pair",
    )
    for mode in references:
        for state in ("cold", "warm_process"):
            first = [
                r["measurement"]["first_device_image_seconds_since_child_entry"]
                for r in rows
                if (r["precision"], r["cache_state"]) == (mode, state)
            ]
            summary = raw["summary"][mode][state]
            require(
                summary["samples"] == 3
                and summary["first_image_seconds_raw"] == first
                and math.isclose(
                    summary["first_image_seconds_median"], statistics.median(first), rel_tol=1e-12
                ),
                "cold summary differs from raw processes",
            )
    return {
        **{key: value for key, value in raw.items() if key not in ("runs", "references")},
        "raw_result_sha256": digest,
        "references": references,
        "runs": rows,
    }


def guard_role(value, data, seen):
    require(value["passed"], "numerical guard failed")
    precision = value["precision"]
    require(precision in ("fp32", "fp64"), "unknown guard precision")
    roles = [
        role
        for role in ("baseline", "candidate")
        if value["source"] == data["dpt"][role][precision]["source"]
    ]
    require(len(roles) == 1, "guard source does not identify exactly one measured role")
    role = roles[0]
    require((role, precision) not in seen, "duplicate role/precision guard")
    require(
        value["input"] == data["input"]["metadata"]
        and value["frozen_contract_sha256"] == data["dpt"][role][precision]["contract"]["sha256"]
        and value["cotangent_sha256"] == data["input"]["cotangent"]["sha256"],
        "guard input, frozen contract or cotangent differs from timed rows",
    )
    utc_interval(value["started_at_utc"], value["measured_at_utc"])
    seen.add((role, precision))
    return role


def collect(args):
    result = {
        "schema_version": 1,
        "description": "Measured complete chest resampled to 256³, with 360 views at 512²",
        "dpt": {},
        "competitors": [],
        "timelines": [],
        "guards": [],
        "evidence": {},
        "quality": None,
        "precision_comparison": None,
        "cold": [],
    }
    for role in ("baseline", "candidate"):
        result["dpt"][role] = {}
        for precision in ("fp64", "fp32"):
            path = getattr(args, role + "_" + precision) / "results.json"
            raw, digest = read(path)
            require(
                raw["status"] == "complete"
                and raw["source_unchanged"]
                and raw["source"] == raw["source_after"],
                str(path) + " incomplete",
            )
            require(raw["precision"] == precision, "wrong precision")
            require(raw["input"]["selected_view_indices"] == list(range(360)), "not a full360scan")
            require(
                {m["label"] for m in raw["measurements"]} == set(OPERATIONS), "missing operation"
            )
            require(
                all(raw["checks"][name]["passed"] for name in OPERATIONS), "failed numerical gate"
            )
            result["evidence"][role + "_" + precision] = {"sha256": digest, "file": path.name}
            outputs, output_digest = read(path.with_name("outputs.json"))
            require(
                outputs["source_sha256"] == raw["source"]["sha256"]
                and outputs["input_metadata_sha256"] == raw["input"]["metadata_sha256"]
                and outputs["cotangent_sha256"] == raw["input"]["cotangent"]["sha256"]
                and outputs["precision"] == precision
                and outputs["views"] == 360,
                "DPT output manifest differs",
            )
            result["evidence"][role + "_" + precision]["outputs_sha256"] = output_digest
            result["dpt"][role][precision] = {
                key: raw[key]
                for key in (
                    "started_at",
                    "completed_at",
                    "source",
                    "environment",
                    "checks",
                    "preparation_seconds",
                    "contract",
                )
            }
            result["dpt"][role][precision]["measurements"] = [
                compact_measurement(row) for row in raw["measurements"]
            ]
            result["dpt"][role][precision]["outputs"] = outputs
            if "input" not in result:
                result["input"] = raw["input"]
            else:
                require(raw["input"] == result["input"], "DPT inputs differ")
    for directory in args.competitor:
        raw, digest = read(directory / "results.json")
        require(
            "measurement_finished_at" in raw and not raw.get("failures"), "competitor incomplete"
        )
        require(
            raw["input_metadata"]["sha256"] == result["input"]["metadata_sha256"],
            "competitor volume differs",
        )
        require(
            raw["cotangent"]["sha256"] == result["input"]["cotangent"]["sha256"],
            "competitor seed differs",
        )
        entry = {key: value for key, value in raw.items() if key not in ("measurements", "input")}
        entry["measurements"] = [compact_measurement(row) for row in raw["measurements"]]
        entry["raw_result_sha256"] = digest
        result["competitors"].append(entry)
    for path in args.timeline:
        value, digest = read(path)
        result["timelines"].append({"raw_result_sha256": digest, "summary": value})
    seen_guards = set()
    for path in args.guard:
        value, digest = read(path)
        role = guard_role(value, result, seen_guards)
        if result["guards"]:
            previous = result["guards"][0]["summary"]
            require(
                value["guard"] == previous["guard"]
                and value["guard_contract_sha256"] == previous["guard_contract_sha256"],
                "guard configurations differ",
            )
        result["guards"].append(
            {
                "role": role,
                "raw_result_sha256": digest,
                "summary": {
                    key: item for key, item in value.items() if key not in ("source", "input")
                },
            }
        )
    require(
        seen_guards
        == {(role, mode) for role in ("baseline", "candidate") for mode in ("fp32", "fp64")},
        "both precision guards required exactly once for both revisions",
    )
    if args.quality:
        result["quality"] = collect_quality(args.quality, result)
    else:
        result["precision_comparison"] = collect_precision_comparison(
            {mode: getattr(args, "candidate_" + mode) for mode in ("fp32", "fp64")}, result
        )
    for path in args.cold:
        result["cold"].append(collect_cold(path, result))
    if args.cold:
        modes = [mode for entry in result["cold"] for mode in entry["references"]]
        require(sorted(modes) == ["fp32", "fp64"], "cold results need both modes exactly once")
    return result


def rows_by_operation(data, role, precision):
    return {row["label"]: row for row in data["dpt"][role][precision]["measurements"]}


def peak(row):
    values = [r["nvml"].get("sampled_peak_process_memory_used_bytes") for r in row["rounds"]]
    values = [v for v in values if v is not None]
    return max(values) / 2**30 if values else None


def sampling_table(data):
    sweep = data.get("quality")
    if sweep is None:
        return ""
    rows = []
    for row in sorted(sweep["measurements"], key=lambda r: r["configuration"]["samples_per_ray"]):
        error = row["error_vs_fp64_1024"]
        memory = peak(row)
        rows.append(
            [
                str(row["configuration"]["samples_per_ray"]),
                f"{row['wall_median_ms']:.2f} / {row['wall_p95_ms']:.2f}",
                f"{row['device_median_ms']:.2f} / {row['device_p95_ms']:.2f}",
                "unavailable" if memory is None else f"{memory:.2f}",
                f"{error['max_abs']:.6g}",
                f"{error['rmse']:.6g}",
                f"{error['relative_l2']:.6g}",
            ]
        )
    return (
        "## FP32 sampling quality versus time\n\n"
        "The sample-count sweep uses the same complete field and "
        "360 physical views. Errors use the measured FP64/1024-sample optical-depth image "
        "as reference. "
        "Every row contains three rounds of ten warm resident public forward calls, including "
        "output allocation and internal validation. VRAM is the maximum observed NVML "
        "process sample.\n\n"
        + table(
            [
                "Midpoints per clipped ray",
                "Completed wall median / p95 (ms)",
                "CUDA-event median / p95 (ms)",
                "Sampled process VRAM (GiB)",
                "Maximum absolute error",
                "RMSE",
                "Relative L2 error",
            ],
            rows,
        )
        + f"\n\nMeasured {sweep['started_at']} to {sweep['completed_at']}. "
        + f"Selected source `{sweep['source']['sha256']}`. "
        + "The 1024-sample row includes the same-discretisation arithmetic check.\n"
    )


def cold_table(data):
    rows = []
    for entry in data.get("cold", []):
        for mode in sorted(entry["references"], reverse=True):
            for state, label in (
                ("cold", "Empty compilation caches"),
                ("warm_process", "Reused compilation caches"),
            ):
                runs = [
                    row["measurement"]
                    for row in entry["runs"]
                    if (row["precision"], row["cache_state"]) == (mode, state)
                ]
                first = [row["first_device_image_seconds_since_child_entry"] for row in runs]
                forward = [row["seconds"]["first_forward"] for row in runs]
                warm = [value for row in runs for value in row["warm"]["wall_ms"]]
                rows.append(
                    [
                        mode.upper(),
                        label,
                        str(len(runs)),
                        f"{statistics.median(first):.3f} / {percentile(first, 0.95):.3f}",
                        f"{statistics.median(forward):.3f} / {percentile(forward, 0.95):.3f}",
                        f"{statistics.median(warm):.2f} / {percentile(warm, 0.95):.2f}",
                    ]
                )
    if not rows:
        return ""
    dates = [
        f"{entry['measurement_started_at']} to {entry['measurement_finished_at']}"
        for entry in data["cold"]
    ]
    return (
        "## Fresh-process startup\n\n"
        "Each precision has three independent empty-cache processes, each followed by a "
        "new process reusing that pair of compilation caches. The first-image clock starts "
        "at benchmark child entry, after interpreter startup and initial standard-library "
        "imports. It includes scientific-package imports, CUDA initialisation, loading "
        "the prepared attenuation field, "
        "upload, module construction and the first completed public projection. Lazy "
        "workspace preparation and JIT are inside the first public call. Original CT "
        "resampling/calibration, output download and subsequent correctness checks are "
        "outside the interval. The OS filesystem cache remains in its observed state.\n\n"
        + table(
            [
                "Arithmetic",
                "New-process cache state",
                "Processes",
                "First device image median / p95 (s)",
                "First public forward median / p95 (s)",
                "Warm completed wall median / p95 (ms)",
            ],
            rows,
        )
        + "\n\nWarm rows pool three calls per process. The reported p95 values are sample "
        + "quantiles. First-image p95 uses three processes. "
        + "Raw phase timings, module events and actual process dates remain in the compact "
        + "results. Measurement intervals: "
        + ". ".join(dates)
        + ".\n"
    )


def precision_run_binding(data, mode):
    """Identify a selected full-scan image without copying or changing timing rows."""
    run = data["dpt"]["candidate"][mode]
    outputs = run["outputs"]
    image = outputs["arrays"]["images"]
    source = run["source"]
    require(
        bool(source.get("sha256"))
        and bool(source.get("files"))
        and source == data["dpt"]["candidate"]["fp64"]["source"],
        "precision error source maps differ or are missing",
    )
    require(
        data["input"]["selected_view_indices"] == list(range(360))
        and data["input"]["selected_geometry"] == data["input"]["metadata"]["geometry"]
        and run["contract"]["settings"]["sampling"]["samples_per_ray"] == 1024
        and data["input"]["metadata"]["sampling"]["samples_per_ray"] == 1024,
        "precision error physical input or sampling differs",
    )
    rows = [row for row in run["measurements"] if row["label"] == "forward"]
    require(len(rows) == 1, "precision error needs one completed forward row")
    config = rows[0]["configuration"]
    require(
        config["precision"] == mode
        and config["views"] == 360
        and config["detector_shape"] == [512, 512]
        and config["samples_per_ray"] == 1024
        and outputs["precision"] == mode
        and outputs["views"] == 360
        and outputs["source_sha256"] == source["sha256"]
        and outputs["input_metadata_sha256"] == data["input"]["metadata_sha256"]
        and outputs["cotangent_sha256"] == data["input"]["cotangent"]["sha256"]
        and image["shape"] == [360, 512, 512]
        and image["dtype"] == mode.replace("fp", "float")
        and image["units"] == "dimensionless optical depth",
        "precision error image shape, dtype or run binding differs",
    )
    evidence = data["evidence"]["candidate_" + mode]
    require(
        all(
            isinstance(value, str) and len(value) == 64
            for value in (
                evidence["sha256"],
                evidence["outputs_sha256"],
                image["sha256"],
                image["raw_sha256"],
            )
        ),
        "precision error needs recorded result, manifest and array hashes",
    )
    return {
        "results_sha256": evidence["sha256"],
        "outputs_sha256": evidence["outputs_sha256"],
        "source_sha256": source["sha256"],
        "started_at_utc": run["started_at"],
        "completed_at_utc": run["completed_at"],
        "precision": mode,
        "images": image,
        "cotangent_sha256": outputs["cotangent_sha256"],
        "contract_sha256": run["contract"]["sha256"],
    }


def collect_precision_comparison(directories, data):
    """Read both checked NPY files and derive an untimed precision comparison."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    try:
        from benchmarks.fullscan import array_error
    finally:
        sys.path.pop(0)
    helper = root / "benchmarks/fullscan.py"
    require(
        Path(array_error.__code__.co_filename).resolve() == helper,
        "precision error helper is not the shipped benchmark",
    )
    runs = {mode: precision_run_binding(data, mode) for mode in ("fp32", "fp64")}
    arrays = {}
    for mode in runs:
        arrays[mode], binding, outputs = load_run(
            directories[mode], mode, data["input"]["metadata"], data["input"]["metadata_sha256"]
        )
        require(
            binding == runs[mode] and outputs == data["dpt"]["candidate"][mode]["outputs"],
            "precision error saved files differ from the compact run",
        )
    return {
        "schema_version": 1,
        "method": PRECISION_METHOD,
        "timed": False,
        "source": data["dpt"]["candidate"]["fp32"]["source"],
        "input": data["input"],
        "samples_per_ray": 1024,
        "runs": runs,
        "calculation": {
            "file": "benchmarks/fullscan.py",
            "sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
            "function": "array_error",
        },
        "errors": array_error(arrays["fp32"], arrays["fp64"], np),
    }


def saved_precision_error(data):
    comparison = data["precision_comparison"]
    require(
        comparison["schema_version"] == 1
        and comparison["method"] == PRECISION_METHOD
        and comparison["timed"] is False
        and data.get("quality") is None
        and comparison["samples_per_ray"] == 1024
        and comparison["source"] == data["dpt"]["candidate"]["fp32"]["source"]
        and comparison["input"] == data["input"]
        and comparison["runs"]
        == {mode: precision_run_binding(data, mode) for mode in ("fp32", "fp64")},
        "precision error saved-image provenance differs",
    )
    calculation = comparison["calculation"]
    require(
        calculation["file"] == "benchmarks/fullscan.py"
        and calculation["function"] == "array_error"
        and isinstance(calculation["sha256"], str)
        and len(calculation["sha256"]) == 64,
        "precision error calculation identity is missing",
    )
    errors = comparison["errors"]
    require(
        all(
            math.isfinite(errors[key]) and errors[key] >= 0
            for key in ("max_abs", "rmse", "relative_l2")
        )
        and errors["bitwise_equal"] is False,
        "precision error values are invalid",
    )
    return errors


def fp32_fp64_error(data):
    """Return the measured precision error only when its image identities are bound."""
    if data.get("precision_comparison") is not None:
        return saved_precision_error(data)
    quality = data.get("quality")
    if quality is None:
        return None
    candidate = data["dpt"]["candidate"]["fp32"]
    require(
        quality["status"] == "complete"
        and quality["source_unchanged"]
        and quality["source"] == quality["source_after"] == candidate["source"],
        "precision error source differs from the measured implementation",
    )
    for key in ("metadata_sha256", "metadata", "selected_view_indices", "selected_geometry"):
        require(quality["input"][key] == data["input"][key], "precision error input differs")
    reference = quality["reference"]
    roles = [
        role
        for role in ("baseline", "candidate")
        if reference["results_sha256"] == data["evidence"][role + "_fp64"]["sha256"]
    ]
    require(len(roles) == 1, "precision error needs one measured FP64 reference")
    role = roles[0]
    precise = data["dpt"][role]["fp64"]
    require(
        reference["outputs_sha256"] == data["evidence"][role + "_fp64"]["outputs_sha256"]
        and reference["source_sha256"] == precise["source"]["sha256"]
        and reference["completed_at"] == precise["completed_at"]
        and reference["samples_per_ray"] == 1024
        and reference["images"] == precise["outputs"]["arrays"]["images"],
        "precision error FP64 reference differs",
    )
    check = quality["checks"]["fp32/1024-samples/forward"]
    errors = check["error_vs_fp64_1024"]
    rows = [
        row for row in quality["measurements"] if row["configuration"]["samples_per_ray"] == 1024
    ]
    require(
        len(rows) == 1
        and rows[0]["error_vs_fp64_1024"] == errors
        and rows[0]["configuration"]["error_vs_fp64_1024"] == errors,
        "precision error check and plotted measurement differ",
    )
    limits = candidate["contract"]["settings"]["quality"]["image_fp32_vs_fp64"]
    require(
        check["raw_sha256"] == candidate["outputs"]["arrays"]["images"]["raw_sha256"]
        and check["after_timing"]["bitwise_unchanged"]
        and check["after_timing"]["raw_sha256"] == check["raw_sha256"]
        and check["arithmetic_accuracy_gate"]["passed"]
        and check["arithmetic_accuracy_gate"]["thresholds"] == limits
        and all(
            math.isfinite(errors[key]) and 0 <= errors[key] <= limits[key]
            for key in ("max_abs", "rmse", "relative_l2")
        ),
        "precision error image or arithmetic gate differs",
    )
    return errors


def tables(data):
    rows = []
    for precision in ("fp64", "fp32"):
        measured = rows_by_operation(data, "candidate", precision)
        for name in OPERATIONS:
            a = measured[name]
            memory = peak(a)
            rows.append(
                [
                    precision.upper(),
                    LABELS[name],
                    f"{a['wall_median_ms']:.2f} / {a['wall_p95_ms']:.2f}",
                    "unavailable" if memory is None else f"{memory:.2f}",
                ]
            )
    output = [
        "## DPT: completed resident PyTorch calls\n\n"
        + table(
            [
                "Arithmetic",
                "Operation",
                "Median / p95 (ms)",
                "Sampled process VRAM (GiB)",
            ],
            rows,
        )
    ]
    comparisons = []
    for entry in data["competitors"]:
        for row in entry["measurements"]:
            config = row["configuration"]
            backend = entry["backends"][config["backend"]]
            image = backend["image_error_vs_dpt_fp64"]["relative_l2"]
            defect = backend.get("adjoint", {}).get("relative_defect")
            memory = peak(row)
            comparisons.append(
                [
                    config["backend"],
                    row["label"].split("/")[1],
                    config["operation"],
                    f"{len(row['rounds'])} × {row['repeats_per_round']}",
                    f"{row['wall_median_ms']:.2f} / {row['wall_p95_ms']:.2f}",
                    "unavailable" if memory is None else f"{memory:.2f}",
                    f"{image:.3g}",
                    "—" if defect is None else f"{defect:.3g}",
                ]
            )
    output.append(
        "## ASTRA and TIGRE: their native algorithms\n\n"
        + table(
            [
                "Implementation",
                "Boundary",
                "Operation",
                "Rounds × repeats",
                "Median / p95 (ms)",
                "Sampled process VRAM (GiB)",
                "Image relative L2 vs DPT FP64",
                "Measured adjoint defect",
            ],
            comparisons,
        )
    )
    output.append(
        "Image and backprojection errors use the saved FP64 arrays identified in "
        "[the numerical reference manifest](competitor_reference.json). "
        "DPT's complete iteration requests both volume and pose gradients. Competitor "
        "iterations request volume gradients only. The volume-only table below gives "
        "the corresponding DPT measurement.\n"
    )
    output.append(
        "ASTRA uses cuda3d native slice traversal and voxel-driven backprojection, with "
        "detector and voxel supersampling both set to one. Its resident rows link CUDA "
        "arrays through DLPack. TIGRE Siddon uses piecewise-constant voxel intersections "
        "and explicit `Atb(backprojection_type='matched')`. Its interpolation rows vary "
        "the native `accuracy` setting. TIGRE Torch rows include the installed wrapper's "
        "host staging and required native transfers. "
        "All rows report completed wall time and sampled NVML process memory.\n"
    )
    quality = []
    for precision in ("fp64", "fp32"):
        checks = data["dpt"]["candidate"][precision]["checks"]
        image = checks["forward"]["images_vs_reference"]
        gradient = checks["volume_vjp"]["volume_vjp_vs_reference"]
        quality.append(
            [
                precision.upper(),
                str(image["bitwise_equal"]),
                f"{gradient['relative_l2']:.6g}",
                f"{checks['volume_vjp']['volume_adjoint']['relative_discrepancy']:.6g}",
            ]
        )
    fast = fp32_fp64_error(data)
    precision_text = (
        "\n\nFP32 against FP64 on the same stored field and physical rays: "
        + f"maximum optical-depth error {fast['max_abs']:.6g}, "
        + f"RMSE {fast['rmse']:.6g}, relative L2 {fast['relative_l2']:.6g}. "
        + "These measurements describe this resampled chest and acquisition.\n"
        if fast is not None
        else ""
    )
    if data.get("precision_comparison") is not None:
        precision_text += (
            "This untimed comparison uses the selected runs' saved images and their recorded "
            "NPY and raw-array checksums. Its plot point uses the measured FP32 forward "
            "median.\n"
        )
    output.append(
        "## Numerical agreement\n\n"
        + table(
            [
                "Arithmetic",
                "Image identical to same-mode reference",
                "Volume VJP relative L2 vs reference",
                "Volume adjoint discrepancy",
            ],
            quality,
        )
        + precision_text
    )
    derivatives = []
    for item in data["guards"]:
        if item["role"] != "candidate":
            continue
        guard = item["summary"]
        pose = next(check for check in guard["pose_checks"] if check["acceptance_step"])
        volume = guard["volume_directional_check"]
        derivatives.append(
            [
                guard["precision"].upper(),
                f"{volume['absolute_error']:.6g}",
                f"{volume['allowed_absolute_error']:.6g}",
                f"{pose['absolute_error']:.6g}",
                f"{pose['allowed_absolute_error']:.6g}",
                f"{max(ray['absolute_error'] for ray in guard['independent_rays']):.6g}",
            ]
        )
    output.append(
        "Nonidentity-pose guards use views 0, 37 and 113, the fixed asymmetric detector "
        "objective and the saved finite-difference steps. Volume differences account "
        "for the realised FP32 input perturbation. Independent CPU rays use an explicit "
        "slab intersection and eight interpolation weights.\n\n"
        + table(
            [
                "Arithmetic",
                "Volume FD absolute error",
                "Volume error bound",
                "Pose FD absolute error",
                "Pose error bound",
                "CPU ray max absolute error",
            ],
            derivatives,
        )
    )
    output.extend(value for value in (sampling_table(data), cold_table(data)) if value)
    return "\n\n".join(output)


def figures(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    positions = np.arange(4)
    for axis, precision in zip(axes, ("fp64", "fp32"), strict=True):
        rows = rows_by_operation(data, "candidate", precision)
        times = [rows[name]["wall_median_ms"] / 1000 for name in OPERATIONS]
        axis.bar(positions, times, width=0.65, color="#0072B2")
        axis.set_xticks(
            positions, ["Forward", "Volume VJP", "Pose VJP", "Full iteration"], rotation=20
        )
        axis.set_ylabel("Completed wall time (s)")
        axis.set_title(precision.upper())
        axis.spines[["right", "top"]].set_visible(False)
    save_figure(fig, output, "fullscan-latency")
    plt.close(fig)
    points = []
    for entry in data["competitors"]:
        for row in entry["measurements"]:
            config = row["configuration"]
            if config["operation"] != "forward":
                continue
            boundary = row["label"].split("/")[1]
            # Compare the best actual supported boundary for each backend.
            if boundary != ("resident" if config["backend"].startswith("astra") else "host"):
                continue
            error = entry["backends"][config["backend"]]["image_error_vs_dpt_fp64"]["relative_l2"]
            points.append(
                (row["wall_median_ms"], error, f"{config['backend']} · {boundary}", "competitor")
            )
    precision_error = fp32_fp64_error(data)
    if precision_error is not None and data.get("quality") is not None:
        for row in sorted(
            data["quality"]["measurements"], key=lambda r: r["configuration"]["samples_per_ray"]
        ):
            count = row["configuration"]["samples_per_ray"]
            points.append(
                (
                    row["wall_median_ms"],
                    row["error_vs_fp64_1024"]["relative_l2"],
                    f"DPT FP32 · {count} midpoints · resident",
                    "dpt",
                )
            )
    elif precision_error is not None:
        forward = rows_by_operation(data, "candidate", "fp32")["forward"]
        points.append(
            (
                forward["wall_median_ms"],
                precision_error["relative_l2"],
                "DPT FP32 · 1024 midpoints · resident",
                "dpt",
            )
        )
    fig, axis = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=0.10, right=0.58, bottom=0.23, top=0.95)
    markers = iter(("o", "s", "^", "D", "v", "P", "X", "<", ">"))
    for milliseconds, error, label, kind in points:
        if error <= 0:
            continue
        axis.scatter(
            milliseconds,
            error,
            s=42,
            marker=next(markers, "o"),
            label=label,
            color="#0072B2" if kind == "dpt" else "#D55E00",
        )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="Completed forward wall time (ms)",
        ylabel="Image relative L2 error against DPT FP64",
    )
    axis.margins(x=0.15, y=0.18)
    axis.legend(loc="upper left", bbox_to_anchor=(1.03, 1), frameon=False, fontsize=9)
    axis.spines[["right", "top"]].set_visible(False)
    fig.text(
        0.10,
        0.06,
        "Different discretisations. Reference: DPT FP64, 1024 midpoints.\n"
        "Measured native timing boundaries are listed in the legend. Exact-zero errors "
        "are omitted from log axes.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    save_figure(fig, output, "fullscan-latency-error")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Previously collected compact numeric results")
    for role in ("baseline", "candidate"):
        for precision in ("fp64", "fp32"):
            parser.add_argument("--" + role + "-" + precision, type=Path)
    parser.add_argument("--competitor", type=Path, action="append", default=[])
    parser.add_argument("--timeline", type=Path, action="append", default=[])
    parser.add_argument("--guard", type=Path, action="append", default=[])
    parser.add_argument("--quality", type=Path, help="Completed FP32 sampling-sweep directory")
    parser.add_argument(
        "--cold",
        type=Path,
        action="append",
        default=[],
        help="Completed fullscan_cold results.json with both modes",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fonts", type=Path, required=True)
    args = parser.parse_args()
    if args.results and (args.quality or args.cold):
        parser.error("--results reuses collected inputs. Add --quality/--cold during collection")
    if not args.results and any(
        getattr(args, role + "_" + precision) is None
        for role in ("baseline", "candidate")
        for precision in ("fp64", "fp32")
    ):
        parser.error("collection requires all four --baseline/--candidate precision directories")
    data = read(args.results)[0] if args.results else collect(args)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "measurements.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "tables.md").write_text(tables(data))
    configure_style(args.fonts)
    figures(data, args.output)


if __name__ == "__main__":
    main()
