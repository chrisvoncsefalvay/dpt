# ruff: noqa: E501, RUF001
# Preserve scientific minus/multiplication glyphs and complete generated Markdown rows.
"""Generate performance tables directly from completed, saved numeric measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
from pathlib import Path

MODES = ("float64", "mixed_float32", "fast_float32")
LABELS = ("Strict", "Mixed FP32", "Fast FP32")
RUN_LABELS = {
    "candidate-before-512": "unmodified-bounds-512",
    "candidate-bounds-512": "invariant-bounds-512",
}


def replace_block(path: Path, tag: str, replacement: str) -> None:
    start, end = f"<!-- {tag}:start -->", f"<!-- {tag}:end -->"
    content = path.read_text()
    if start not in content or end not in content:
        raise ValueError(f"Missing generated-table markers in {path}")
    content = re.sub(
        re.escape(start) + r".*?" + re.escape(end),
        lambda _: start + "\n\n" + replacement + "\n\n" + end,
        content,
        flags=re.DOTALL,
    )
    path.write_text(content)


def baseline_tables(root: Path, source: Path, *, readme: bool) -> None:
    result = json.loads(source.read_text())
    if not result.get("measurement_finished_at"):
        raise ValueError("Measurement run is incomplete")
    indexed = {row["label"]: row for row in result["measurements"]}
    short = [
        "| Precision | Three-view forward median / p95 (ms) | AP volume VJP median / p95 (ms) |",
        "|---|---:|---:|",
    ]
    full = [
        "| Precision | Forward median / p95 (ms) | Public call median / p95 (ms) | Completed NumPy call median / p95 (ms) | AP volume VJP median / p95 (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    errors = [
        "| Precision | Image RMSE | Maximum image error | Relative L2 image error | Adjoint discrepancy | First forward (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for mode, label in zip(MODES, LABELS, strict=True):
        resident = indexed[mode + "/resident_forward_3views"]
        checked = indexed[mode + "/checked_public_forward_3views"]
        completed = indexed[mode + "/completed_numpy_user_call_3views"]
        vjp = indexed[mode + "/resident_volume_vjp_ap"]

        def pair(record, boundary):
            return f"{record[boundary + '_median_ms']:.3f} / {record[boundary + '_p95_ms']:.3f}"

        short.append(f"| {label} | {pair(resident, 'device')} | {pair(vjp, 'device')} |")
        full.append(
            f"| {label} | {pair(resident, 'device')} | {pair(checked, 'wall')} | {pair(completed, 'wall')} | {pair(vjp, 'device')} |"
        )
        info = result["modes"][mode]
        error = info["image_error_vs_strict"]
        errors.append(
            f"| {label} | {error['rmse']:.6g} | {error['max_abs']:.6g} | {error['relative_l2']:.6g} | {info['volume_adjoint']['relative_discrepancy']:.6g} | {info['cold_forward_seconds']:.6g} |"
        )
        rows.append(
            {
                "mode": mode,
                "forward_median_ms": resident["device_median_ms"],
                "forward_p95_ms": resident["device_p95_ms"],
                "vjp_median_ms": vjp["device_median_ms"],
                "vjp_p95_ms": vjp["device_p95_ms"],
                **error,
                "adjoint_discrepancy": info["volume_adjoint"]["relative_discrepancy"],
            }
        )
    if readme and "<!-- baseline:start -->" in (root / "README.md").read_text():
        replace_block(root / "README.md", "baseline", "\n".join(short))
    inputs = result["input"]
    dimensions = " × ".join(map(str, reversed(inputs["native_shape_zyx"])))
    detector = " × ".join(map(str, inputs["geometry"][0]["shape"]))
    views = "/".join(inputs["view_names"])
    description = (
        f"Measured {result['measurement_started_at']} to {result['measurement_finished_at']} on "
        f"{result['environment']['gpu']}. Native volume {dimensions}, {views} views, "
        f"{detector} pixels each, {inputs['samples_per_ray']} {inputs['integration']} samples per ray. "
        "All modes use the same FP32 attenuation field and FP64 accumulation/output."
    )
    setup = "| Boundary | Seconds |\n|---|---:|\n" + "\n".join(
        f"| {label} | {result[key]:.6g} |"
        for label, key in (
            ("Import", "import_seconds"),
            ("File loading", "file_loading_seconds"),
            ("HU conversion", "ct_conversion_seconds"),
            ("Device upload", "device_upload_seconds"),
        )
    )
    block = description + "\n\n" + "\n".join(full) + "\n\n" + "\n".join(errors) + "\n\n" + setup
    replace_block(root / "docs/performance.md", "baseline", block)
    path = root / "docs/assets/results/baseline_summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated tables from {source}")


def table(headers, rows):
    """Escape arbitrary recorded text before embedding it in a Markdown table."""

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace(";", ",")

    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        + ["| " + " | ".join(map(cell, row)) + " |" for row in rows]
    )


def number(value, *, digits=6):
    return "unmeasured" if value is None else f"{value:.{digits}g}"


def pair(record, boundary="device"):
    a, b = record.get(boundary + "_median_ms"), record.get(boundary + "_p95_ms")
    return "unmeasured" if a is None or b is None else f"{a:.3f} / {b:.3f}"


def percentile(samples, probability):
    values = sorted(samples)
    position = (len(values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def verify_samples(document):
    """Reject stale summaries and incomplete rounds rather than inventing missing data."""
    for row in document.get("measurements", []):
        if row.get("wall_median_ms") is None or not row.get("rounds"):
            raise ValueError(f"Incomplete measured operation: {row['label']}")
        for boundary in ("device", "wall"):
            samples = [
                x for round_ in row.get("rounds", []) for x in (round_.get(boundary + "_ms") or [])
            ]
            median = row.get(boundary + "_median_ms")
            p95 = row.get(boundary + "_p95_ms")
            if median is None and p95 is None:
                continue
            if not samples or any(not math.isfinite(x) or x < 0 for x in samples):
                raise ValueError(f"Missing or invalid raw {boundary} samples: {row['label']}")
            if not math.isclose(median, statistics.median(samples), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Stale median: {row['label']}")
            if not math.isclose(p95, percentile(samples, 0.95), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Stale p95: {row['label']}")
        expected = row.get("repeats_per_round")
        if expected and any(len(r.get("wall_ms", [])) != expected for r in row["rounds"]):
            raise ValueError(f"Incomplete measurement round: {row['label']}")


def load_results(paths, category, assets):
    results = []
    for source in paths:
        data = json.loads(source.read_text())
        partial = category == "partial-baseline"
        if not partial and (not data.get("measurement_finished_at") or data.get("failure")):
            raise ValueError(f"Cannot publish incomplete or failed {category} run: {source}")
        if (
            data.get("source_unchanged_during_measurement") is False
            or data.get("source_unchanged") is False
        ):
            raise ValueError(f"Timed source changed: {source}")
        eligible, excluded = [], []
        if partial:
            for measured in data.get("measurements", []):
                try:
                    verify_samples({"measurements": [measured]})
                    expected = data.get("timing_policy", {}).get("rounds", 3)
                    if len(measured["rounds"]) != expected:
                        raise ValueError("Incomplete independent rounds")
                except ValueError as error:
                    excluded.append({"label": measured["label"], "reason": str(error)})
                else:
                    eligible.append(measured)
            if not eligible:
                raise ValueError(f"No complete baseline operations: {source}")
        else:
            verify_samples(data)
            eligible = data.get("measurements", [])
        name = source.parent.name if source.name == "results.json" else source.stem
        filename = f"{category}-{name}.json"
        target = assets / filename
        compact = (
            json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
        )
        if target.exists() and json.loads(target.read_text()) != data:
            raise FileExistsError(f"Refusing to replace a different measured result: {target}")
        target.write_text(compact)
        results.append(
            {
                "name": name,
                "file": filename,
                "data": data,
                "eligible_measurements": eligible,
                "excluded_measurements": excluded,
                "partial": partial,
            }
        )
    return results


def dataset(data):
    inputs = data.get("input", {})
    shape = inputs.get("native_shape_zyx", inputs.get("grid", {}).get("shape", []))
    anatomy = inputs.get("anatomy", "CT")
    return anatomy + " " + " × ".join(map(str, reversed(shape)))


def configuration(data, row):
    inputs = data.get("input", {})
    config = row.get("configuration", {})
    geometry = config.get("geometry", inputs.get("geometry", []))
    if isinstance(geometry, dict):
        geometry = inputs.get("geometry", [])
    if "view" in config:
        view_index = inputs["view_names"].index(config["view"])
        geometry = [inputs["geometry"][view_index]]
    elif "fp32_gather" in data and row.get("method") in ("scatter", "gather"):
        # The saved gather schema uses the first AP panel from its acquisition.
        geometry = [inputs["geometry"][0]]
    shape = config.get("detector_shape") or (geometry[0].get("shape", []) if geometry else [])
    views = config.get("views", len(geometry))
    if isinstance(views, list):
        views = len(views)
    if "package" in config:
        samples = config.get("n_points") or "native traversal"
    else:
        samples = config.get("samples_per_ray", inputs.get("samples_per_ray"))
    return config, " × ".join(map(str, shape)), views, samples


def memory_peak(row):
    sources = [r.get("nvml", {}) for r in row.get("rounds", [])]
    sources.append(row.get("memory", {}))
    values = [
        s["sampled_peak_process_memory_used_bytes"]
        for s in sources
        if "sampled_peak_process_memory_used_bytes" in s
    ]
    return max(values) if values else None


def latency_rows(records):
    rows, numeric = [], []
    for record in records:
        data = record["data"]
        for measured in record.get("eligible_measurements", data.get("measurements", [])):
            config, shape, views, samples = configuration(data, measured)
            peak = memory_peak(measured)
            count = sum(len(r.get("wall_ms", [])) for r in measured.get("rounds", []))
            rows.append(
                (
                    dataset(data),
                    measured["label"],
                    shape,
                    views,
                    samples,
                    pair(measured),
                    pair(measured, "wall"),
                    count,
                    number(peak / 2**30, digits=4) if peak is not None else "unmeasured",
                )
            )
            numeric.append(
                {
                    "result": record["file"],
                    "measured_at": data["measurement_started_at"],
                    "dataset": dataset(data),
                    "operation": measured["label"],
                    "detector_shape": shape,
                    "views": views,
                    "samples_per_ray": samples,
                    "device_median_ms": measured.get("device_median_ms"),
                    "device_p95_ms": measured.get("device_p95_ms"),
                    "wall_median_ms": measured.get("wall_median_ms"),
                    "wall_p95_ms": measured.get("wall_p95_ms"),
                    "raw_sample_count": count,
                    "sampled_peak_process_bytes": peak,
                    "boundary": config.get(
                        "boundary", measured.get("boundary", "see raw configuration")
                    ),
                }
            )
    return rows, numeric


def correctness_rows(records):
    rows = []
    for record in records:
        data, checks = record["data"], record["data"].get("checks", {})
        for mode in ("fp32", "fp64"):
            check = checks.get(mode + "_real_volume")
            adjoint = checks.get(mode + "_native_adjoint", {})
            if check or adjoint:
                check = check or {}
                error = check.get("image_error_vs_fp64", {})
                rows.append(
                    (
                        dataset(data),
                        " × ".join(map(str, data["input"]["geometry"][0]["shape"])),
                        len(data["input"]["geometry"]),
                        mode,
                        number(error.get("max_abs")),
                        number(error.get("rmse")),
                        number(error.get("relative_l2")),
                        number(adjoint.get("relative_discrepancy")),
                        number(check.get("volume_directional", {}).get("relative_error")),
                        number(check.get("pose_directional", {}).get("relative_error")),
                    )
                )
    return rows


def preparation_tables(records):
    rows, rates = [], []
    for record in records:
        data = record["data"]
        for name, value in data.get("preparation", {}).items():
            if isinstance(value, (float, int)) and "seconds" in name:
                rows.append((record["name"], name, number(value)))
        for measured in record.get("eligible_measurements", data.get("measurements", [])):
            label = measured["label"]
            if not (
                label.endswith("/resident_forward_3views")
                or ("/batched_" in label and label.endswith("_resident"))
            ):
                continue
            config, shape, views, _ = configuration(data, measured)
            duration = measured.get("device_median_ms")
            if duration is None:
                continue
            geometry = config.get("geometry", data["input"]["geometry"])
            pixels = math.prod(geometry[0]["shape"])
            rates.append(
                (
                    dataset(data),
                    label,
                    shape,
                    views,
                    number(views * pixels * 1000 / duration),
                    number(views * 1000 / duration),
                    number(
                        config.get(
                            "preparation_seconds",
                            data.get("preparation", {}).get(
                                label.split("/")[0] + "_native_forward_seconds"
                            )
                            if label.endswith("/resident_forward_3views")
                            else None,
                        )
                    ),
                )
            )
    sections = []
    if rows:
        sections.append(table(("Measured run", "Preparation boundary", "Seconds"), rows))
    if rates:
        sections.append(
            table(
                (
                    "Native volume",
                    "Resident operation",
                    "Detector",
                    "Views",
                    "Rays/s from measured median",
                    "Projections/s from measured median",
                    "One preparation (s)",
                ),
                rates,
            )
        )
    return "\n\n".join(sections)


def material_checks(records):
    rows = []
    for record in records:
        data = record["data"]
        for mode in ("fp32", "fp64"):
            check = data.get("checks", {}).get(mode + "_material_spectral")
            if check:
                error = check["spectral_error_vs_numpy"]
                rows.append(
                    (
                        record["name"],
                        mode,
                        number(error.get("max_abs")),
                        number(error.get("rmse")),
                        number(check.get("path_adjoint", {}).get("relative_discrepancy")),
                        number(check.get("volume_directional", {}).get("relative_error")),
                    )
                )
    if not rows:
        return ""
    return table(
        (
            "Native-volume run",
            "Precision",
            "Maximum spectral error (photons/pixel)",
            "Spectral RMSE (photons/pixel)",
            "Material path-adjoint relative discrepancy",
            "Complete photon-objective volume FD relative error",
        ),
        rows,
    )


def profile_tables(paths, root, source_digests):
    """Publish per-kernel counters and roofline definitions."""
    sections = []
    for path in paths:
        data = json.loads(path.read_text())
        if not data.get("profiles") or data.get("schema_version") != 2:
            raise ValueError(f"Expected complete schema-2 per-kernel analysis: {path}")
        for operation in data["operations"]:
            metadata = operation.get("capture_metadata")
            if metadata and (
                metadata.get("completed_checks_passed") is not True
                or metadata.get("source", {}).get("sha256") not in source_digests
            ):
                raise ValueError(f"Incomplete profile or source differs from release: {path}")
        name = path.parent.name
        title = {
            "profile-analysis-candidate-04": "Scalar projector profiles",
            "profile-analysis-material-02": "Material projector profiles",
        }.get(name, name)
        target = root / "docs/assets/results" / ("profiles-" + name + ".json")
        target.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
        figures = []
        for mode in ("fp32", "fp64"):
            if data["figures"].get(mode, {}).get("created"):
                for suffix in ("png", "svg", "pdf"):
                    original = path.parent / f"roofline_{mode}.{suffix}"
                    destination = root / "docs/assets/figures" / f"{name}-roofline-{mode}.{suffix}"
                    if not original.exists():
                        raise FileNotFoundError(original)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(original, destination)
                figures.append(
                    f"![{title} {mode} roofline](assets/figures/{name}-roofline-{mode}.png)"
                )
        counters, resources, activity = [], [], []
        for profile in data["profiles"]:
            counters.append(
                (
                    profile["id"],
                    *(
                        number(profile["precision"][mode]["add_mul_fma_flops"])
                        for mode in ("fp32", "fp64")
                    ),
                    *(
                        number(profile[group][level])
                        for group in ("memory_counter_bytes", "roofline_traffic_bytes")
                        for level in ("L1", "L2", "DRAM")
                    ),
                )
            )
            resources.append(
                (
                    profile["id"],
                    number(profile["registers_per_thread"]),
                    number(profile["occupancy_achieved_percent"]),
                    number(profile["predicated_on_threads_per_warp_instruction"]),
                    number(profile["fp64_pipe_active_percent"]),
                    number(profile["register_spill_instructions"]),
                )
            )
            activity.append(
                (
                    profile["id"],
                    *(
                        number(profile.get("throughput_percent_of_peak", {}).get(name))
                        for name in ("SM_elapsed", "DRAM_elapsed", "L2_elapsed", "L1_active")
                    ),
                    *(
                        number(profile.get("atomic_thread_instructions", {}).get(name))
                        for name in ("global_returning_ATOM_ATOMG", "global_reduction_RED_REDG")
                    ),
                )
            )
        scopes = [
            (operation["id"], operation["scope"], operation["kernel_count"])
            for operation in data["operations"]
        ]
        sections.extend(
            (
                f"### {title}",
                f"Measurement completion: {data.get('measurement_completed_at') or 'not recorded'}. "
                f"[Per-kernel counters and definitions](assets/results/{target.name}).",
                table(("Operation", "Capture scope", "Constituent kernels"), scopes),
                table(
                    (
                        "Kernel",
                        "FP32 ADD/MUL/FMA FLOPs",
                        "FP64 ADD/MUL/FMA FLOPs",
                        "L1 counter bytes",
                        "L2 counter bytes",
                        "DRAM counter bytes",
                        "L1 roofline proxy bytes",
                        "L2 roofline proxy bytes",
                        "DRAM roofline bytes",
                    ),
                    counters,
                ),
                table(
                    (
                        "Kernel",
                        "Registers/thread",
                        "Achieved occupancy (%)",
                        "Predicated lanes/32",
                        "FP64 pipe active (%)",
                        "Spill instructions",
                    ),
                    resources,
                ),
                table(
                    (
                        "Kernel",
                        "SM elapsed throughput (% peak)",
                        "DRAM elapsed throughput (% peak)",
                        "L2 elapsed throughput (% peak)",
                        "L1 active throughput (% peak)",
                        "Global ATOM/ATOMG thread instructions",
                        "Global RED/REDG thread instructions",
                    ),
                    activity,
                ),
                *figures,
            )
        )
    return "\n\n".join(sections)


def timeline_tables(directories, root, source_digests):
    sections = []
    for directory in directories:
        captures, rows, validation = {}, [], []
        for metadata_path in sorted(directory.glob("*-manifest.csv")):
            name = metadata_path.name.removesuffix("-manifest.csv")
            data = {}
            for kind in ("manifest", "ranges", "kernels", "transfers", "apis"):
                with (directory / f"{name}-{kind}.csv").open(newline="") as stream:
                    data["metadata" if kind == "manifest" else kind] = list(csv.DictReader(stream))
            metadata = data["metadata"][0]
            if name != "strict-public" and metadata["source_sha256"] not in source_digests:
                raise ValueError(f"Timeline source differs from the measured release: {name}")
            if (
                name != "strict-public"
                and metadata.get("kernel_count_and_sum_agree_with_builtin_nsys_stats") != "True"
            ):
                raise ValueError(f"Timeline kernel totals disagree with Nsight statistics: {name}")
            if int(metadata["gpu_events_overlapping_operation_without_matching_launch"]):
                raise ValueError(f"Unattributed GPU events in complete operation: {name}")
            captures[name] = data
            outer = [
                r
                for r in data["ranges"]
                if r["range"] == ("public" if name == "strict-public" else "operation")
            ]
            if len(outer) != 1:
                raise ValueError(f"Ambiguous complete timeline range: {name}")
            measured = outer[0]
            rows.append(
                (
                    name,
                    metadata["measurement_completed_at"] or "not recorded in legacy export",
                    metadata["views"],
                    number(float(measured["inclusive_nvtx_ms"])),
                    measured["launched_kernel_count"],
                    number(float(measured["launched_kernel_sum_ms"])),
                    number(float(measured["launched_gpu_work_union_ms"])),
                    measured["host_to_device_bytes"],
                    measured["device_to_host_bytes"],
                )
            )
            validation.append(
                (
                    name,
                    measured["full_volume_validation_count"],
                    number(float(measured["full_volume_validation_ms"])),
                    measured["four_byte_readback_count"],
                    number(float(measured["four_byte_readback_gpu_ms"])),
                    measured["synchronise_api_count"],
                    number(float(measured["synchronise_api_sum_ms"])),
                    measured["allocation_free_pool_api_count"],
                )
            )
        if not captures:
            raise ValueError(f"No completed timeline CSVs: {directory}")
        output = root / "docs/assets/results" / f"timeline-{directory.name}.json"
        output.write_text(
            json.dumps({"schema_version": 1, "captures": captures}, separators=(",", ":")) + "\n"
        )
        sections.extend(
            (
                f"[Complete ranges, kernels, transfers and API events](assets/results/{output.name}).",
                table(
                    (
                        "Capture",
                        "Completed UTC",
                        "Views",
                        "Complete NVTX wall (ms)",
                        "Kernels",
                        "Kernel sum (ms)",
                        "GPU-work union (ms)",
                        "H2D bytes",
                        "D2H bytes",
                    ),
                    rows,
                ),
                table(
                    (
                        "Capture",
                        "Full-volume scans",
                        "Scan GPU time (ms)",
                        "Four-byte reads",
                        "Read GPU time (ms)",
                        "Synchronise APIs",
                        "Overlapping synchronise CPU sum (ms)",
                        "Allocation/free API calls",
                    ),
                    validation,
                ),
            )
        )
    return "\n\n".join(sections)


def cold_tables(records):
    rows, phases = [], []
    for record in records:
        data = record["data"]
        if data.get("instrumented_launches"):
            raise ValueError("Instrumented cold-start diagnostics cannot become headline timings")
        groups = sorted({(r["precision"], r["cache_state"]) for r in data["runs"]})
        for mode, state in groups:
            runs = [r for r in data["runs"] if (r["precision"], r["cache_state"]) == (mode, state)]
            first = [r["measurement"]["first_device_image_seconds_since_child_entry"] for r in runs]
            warm = [x for r in runs for x in r["measurement"]["warm"]["wall_ms"]]
            rows.append(
                (
                    mode,
                    state,
                    len(runs),
                    f"{statistics.median(first):.3f} / {percentile(first, 0.95):.3f}",
                    f"{statistics.median(warm):.3f} / {percentile(warm, 0.95):.3f}",
                )
            )
            for phase in runs[0]["measurement"]["seconds"]:
                samples = [r["measurement"]["seconds"][phase] for r in runs]
                phases.append(
                    (
                        mode,
                        state,
                        phase,
                        f"{statistics.median(samples):.6g}",
                        f"{percentile(samples, 0.95):.6g}",
                    )
                )
    if not rows:
        return ""
    return (
        table(
            (
                "Precision",
                "Cache state",
                "Processes",
                "First device image from child entry, median / p95 (s)",
                "Warm completed call median / p95 (ms)",
            ),
            rows,
        )
        + "\n\n"
        + table(("Precision", "Cache state", "Phase", "Median (s)", "p95 (s)"), phases)
    )


def compile_diagnostics(path, root, source_digests):
    if path is None:
        return ""
    data = json.loads(path.read_text())
    if data["source_sha256"] not in source_digests:
        raise ValueError("Compile diagnostic source differs from the measured release")
    target = root / "docs/assets/results/compile-diagnostic.json"
    target.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    rows = []
    for mode, diagnostic in data["diagnostic"].items():
        rows.append(
            (
                mode,
                diagnostic["modules"],
                diagnostic["identities"],
                number(data["clean"][mode]["cold"]["module_load_ms_median"]),
                number(data["clean"][mode]["warm_process"]["module_load_ms_median"]),
                diagnostic["generated_source_bytes"],
                diagnostic["launch_count"],
                diagnostic["lto_artifacts"],
            )
        )
    return (
        table(
            (
                "Precision",
                "Modules",
                "Identities",
                "Clean cold module work, median (ms)",
                "Clean cached module work, median (ms)",
                "Instrumented generated source (bytes)",
                "Instrumented launches",
                "LTO artefacts",
            ),
            rows,
        )
        + "\n\nThe NVIDIA compile-time probe records module identities, cache reuse and "
        "100 resident forward calls. "
        "[Recorded compile/cache diagnostic](assets/results/compile-diagnostic.json)."
    )


def competitor_table(records):
    rows, derivatives = [], []
    for record in records:
        data = record["data"]
        for label, info in data.get("backends", {}).items():
            error = info.get(
                "image_error_vs_dpt_fp64", info.get("image_error_vs_frozen_dpt_fp64", {})
            )
            rows.append(
                (
                    label,
                    info.get("algorithm", "see raw settings"),
                    number(error.get("rmse")),
                    number(error.get("relative_l2")),
                    number(info.get("adjoint", {}).get("relative_defect")),
                    info.get("boundary", "see measured operation"),
                )
            )
            pose = info.get("pose_directional_check", {})
            for sample in pose.get("epsilon_sweep", []):
                derivatives.append(
                    (label, sample["epsilon"], number(sample["relative_error"]), pose["scope"])
                )
    result = (
        table(
            (
                "Backend",
                "Algorithm",
                "Image RMSE against DPT FP64",
                "Relative L2",
                "Measured adjoint discrepancy",
                "Boundary",
            ),
            rows,
        )
        if rows
        else ""
    )

    if derivatives:
        result += "\n\n" + table(
            ("Backend pose diagnostic", "Direction step", "Relative FD discrepancy", "Scope"),
            derivatives,
        )
        result += (
            "\n\nPose finite-difference discrepancies use the direction and step recorded "
            "for each backend. Adjoint discrepancies compare each forward/backprojection pair."
        )
    return result


def before_after(baseline, records, extra):
    old = {
        r["label"].split("/", 1)[1]: r
        for r in baseline["measurements"]
        if r["label"].startswith("float64/")
    }
    for record in extra:
        for measured in record.get("eligible_measurements", []):
            if measured["label"].startswith("strict/"):
                old.setdefault(measured["label"].split("/", 1)[1], measured)
    rows = []
    for record in records:
        data = record["data"]
        inputs, original = data.get("input", {}), baseline["input"]
        matched = all(
            inputs.get(k) == original.get(k)
            for k in ("sha256", "grid", "geometry", "samples_per_ray", "integration", "pose")
        )
        if not matched or not inputs.get("field_matches_baseline_formula_exactly"):
            continue
        for measured in data.get("measurements", []):
            if not measured["label"].startswith("fp64/"):
                continue
            previous = old.get(measured["label"].split("/", 1)[1])
            if previous is None:
                continue
            previous_geometry = previous.get("configuration", {}).get("geometry")
            current_geometry = measured.get("configuration", {}).get("geometry")
            if previous_geometry is not None and previous_geometry != current_geometry:
                continue
            current, prior = measured.get("device_median_ms"), previous.get("device_median_ms")
            if current and prior:
                rows.append(
                    (measured["label"], f"{prior:.3f}", f"{current:.3f}", f"{prior / current:.3f}×")
                )
    return (
        table(
            (
                "Same FP64 operation",
                "Reference median (ms)",
                "Measured median (ms)",
                "Measured speedup",
            ),
            rows,
        )
        if rows
        else (
            "No completed release timing with exactly matching baseline inputs has been supplied."
        )
    )


def gather_comparison(records):
    rows = []
    for record in records:
        data = record["data"]
        if "fp32_gather" not in data:
            continue
        indexed = {r["label"]: r for r in record["eligible_measurements"]}
        for pattern, check in data["checks"].items():
            scatter = indexed.get(pattern + "/scatter")
            gather = indexed.get(pattern + "/gather")
            if scatter is not None and gather is not None:
                rows.append(
                    (
                        pattern,
                        pair(scatter),
                        pair(gather),
                        number(gather["device_median_ms"] / scatter["device_median_ms"]),
                        number(check["max_abs"]),
                        number(check["relative_l2"]),
                    )
                )
    if not rows:
        return "No completed scatter/gather comparison supplied."
    return table(
        (
            "Real CT cotangent",
            "Scatter median / p95 (ms)",
            "Gather median / p95 (ms)",
            "Gather/scatter elapsed ratio",
            "Maximum gradient difference",
            "Relative L2 gradient difference",
        ),
        rows,
    ) + (
        "\n\nThe deterministic FP64 gather alternative is retained as a measured "
        "comparison. A ratio above one means it took longer than scatter on that "
        "cotangent. Both use FP64 gradient destinations and the same realised "
        "strict sampling weights. The gather preparation cost is recorded separately. "
        "The existing gather does not implement the FP32 sampling transpose."
    )


def experiment_tables(records):
    rows = []
    for record in records:
        for measured in record["eligible_measurements"]:
            if (
                measured["label"] == "float64/resident_forward_3views"
                or measured.get("sampling_precision") == "float64"
            ):
                rows.append(
                    (
                        RUN_LABELS.get(record["name"], record["name"]),
                        measured["label"],
                        measured.get("block_size", 128),
                        pair(measured),
                        sum(len(r["wall_ms"]) for r in measured["rounds"]),
                    )
                )
    if not rows:
        return "No separate arithmetic or launch comparison supplied."
    return table(
        (
            "Measured variant",
            "FP64 operation",
            "Threads/block",
            "CUDA event median / p95 (ms)",
            "Raw repeats",
        ),
        rows,
    )


def environment_table(records):
    grouped = {}
    for record in records:
        env = record["data"]["environment"]
        versions = env["versions"]
        smi = env.get("nvidia_smi_before", {}).get("stdout", "")
        driver = re.search(r"Driver Version\s*:\s*([^\n]+)", smi)
        key = (
            env["gpu"],
            driver.group(1).strip() if driver else "see raw environment",
            env.get("torch_cuda", "unrecorded"),
            versions.get("warp-lang", "unrecorded"),
            versions.get("torch", "unrecorded"),
            env.get("python", "unrecorded"),
        )
        grouped.setdefault(key, []).append(record["name"])
    return table(
        ("GPU", "Driver", "Torch CUDA", "Warp", "Torch", "Python", "Measured runs"),
        [(*key, ", ".join(names)) for key, names in grouped.items()],
    )


def optimisation_table(records):
    rows = []
    for record in records:
        data = record["data"]
        if "Generated FP64" not in data["target"]:
            raise ValueError("This diagnostic table requires an explicitly generated FP64 target")
        for mode, result in data["modes"].items():
            if not result["accepted"]:
                raise ValueError(f"Failed optimisation acceptance: {record['name']}/{mode}")
            rows.append(
                (
                    mode,
                    number(math.sqrt(result["initial_mse"])),
                    number(result["final_rmse"]),
                    number(result["final_to_initial_mse_ratio"]),
                    len(result["closure_trace"]),
                    number(result["completed_seconds_including_objective_trace_readbacks"]),
                )
            )
    if not rows:
        return ""
    return table(
        (
            "Precision",
            "Initial optical-depth RMSE",
            "Final optical-depth RMSE",
            "Final/initial MSE",
            "Closure evaluations",
            "Completed solve including trace readbacks (s)",
        ),
        rows,
    )


def result_index(records):
    rows = []
    for record in records:
        data = record["data"]
        environment = data.get("environment", {})
        if not environment and data.get("runs"):
            environment = data["runs"][0]["measurement"].get("environment", {})
        rows.append(
            (
                f"[{RUN_LABELS.get(record['name'], record['name'])}]"
                f"(assets/results/{record['file']})",
                data["measurement_started_at"],
                data.get("measurement_finished_at", "run incomplete, completed rows only"),
                environment.get("gpu", "see raw environment"),
                data.get("source", {}).get("sha256", "see raw source digest"),
            )
        )
    return table(
        ("Raw result", "Started (UTC)", "Finished (UTC)", "GPU", "Timed source SHA256"), rows
    )


def headline(records):
    """A small README table. Leave absent boundaries explicitly unmeasured."""
    indexed = {}
    for record in records:
        data = record["data"]
        if data.get("input", {}).get("native_shape_zyx") != [139, 512, 512]:
            continue
        geometry = data.get("input", {}).get("geometry", [])
        if not geometry or geometry[0]["shape"] != [512, 512]:
            continue
        for row in data.get("measurements", []):
            if row["label"] in indexed:
                raise ValueError(f"Ambiguous repeated headline operation: {row['label']}")
            indexed[row["label"]] = row
    rows = []
    for mode in ("fp32", "fp64"):
        rows.append(
            (
                mode,
                *(
                    pair(indexed.get(mode + "/" + operation, {}), boundary)
                    for operation, boundary in (
                        ("resident_forward_3views", "device"),
                        ("resident_volume_vjp_ap", "device"),
                        ("resident_pose_vjp_ap", "device"),
                        ("torch_forward_backward_both_ap", "wall"),
                    )
                ),
            )
        )
    return table(
        (
            "Precision",
            "Three-view forward (ms)",
            "AP volume VJP (ms)",
            "AP pose VJP (ms)",
            "AP Torch forward/backward, both gradients (ms)",
        ),
        rows,
    )


def publish_release(
    root, baseline, groups, profiles, timelines, compile_result, *, update_readme=True
):
    release, competitors, cold, extra = (
        groups[key] for key in ("release", "competitor", "cold", "extra")
    )
    partial = groups["partial-baseline"]
    extra = extra + partial
    optimisation = groups["optimisation"]
    all_records = release + competitors + cold + optimisation
    source_digests = {r["data"]["source"]["sha256"] for r in release}
    gpu_names = sorted({record["data"]["environment"]["gpu"] for record in release})
    rows, numeric = latency_rows(release)
    competitor_rows, _ = latency_rows(competitors)
    _, other_numeric = latency_rows(competitors + extra)
    headers = (
        "Native volume",
        "Operation / precision",
        "Detector",
        "Views",
        "Samples/ray",
        "CUDA event median / p95 (ms)",
        "Completed wall median / p95 (ms)",
        "Raw repeats",
        "Sampled process peak (GiB)",
    )
    pieces = ["## Measured performance"]

    def section(title, content, description=""):
        if content:
            pieces.extend(part for part in (title, description, content) if part)

    if release:
        pieces.append(environment_table(release))
    if rows:
        pieces.extend(
            (
                "CUDA-event intervals measure device execution; completed wall intervals "
                "measure each operation through its completion boundary. The operation "
                "configuration records its requested gradients. VRAM is the largest observed "
                "per-process NVML sample, including Warp allocations.",
                table(headers, rows),
            )
        )
    section("### Preparation and batching", preparation_tables(release))
    checks = correctness_rows(release)
    if checks:
        section(
            "### Numerical checks",
            table(
                (
                    "Native volume",
                    "Detector",
                    "Views",
                    "Precision",
                    "Maximum optical-depth error",
                    "RMSE",
                    "Relative L2",
                    "Adjoint relative discrepancy",
                    "Volume directional relative error",
                    "Pose directional relative error",
                ),
                checks,
            ),
            "Image errors compare FP32 and FP64 on the same physical rays and sampling rule. "
            "Directional checks use the recorded pose, finite-difference step and detector objective.",
        )
    section(
        "### Material and spectral calculation",
        material_checks(release),
        "The AP material workload uses one explicitly derived water-relative-density basis, "
        "`max(1 + HU/1000, 0)`, a 60 keV coefficient of 0.02059 mm⁻¹ stored in FP32, "
        "10,000 incident photons per detector pixel and unit detector efficiency. "
        "Basis preparation has a separate timing interval. The spectral reference evaluates "
        "`10000 * exp(-coefficient * measured_path)` in NumPy FP64. "
        "The complete derivative check uses the measured photon objective and realised FP32 "
        "density perturbation. Path-adjoint checks use its actual spectral cotangent.",
    )
    section(
        "### Pose-fitting check",
        optimisation_table(optimisation),
        "The pose fit uses a real CT, a simulated FP64 identity-pose target and Torch LBFGS. "
        "The recorded initial "
        "tangent is (1, −0.8, 0.6) mm and (0.003, −0.004, 0.002) radians. Eight optimiser steps "
        "use strong-Wolfe line search.",
    )
    section(
        "### Cold and warm process startup",
        cold_tables(cold),
        "Cold measurements use new processes and private empty Warp/CUDA caches. A second process "
        "reuses each populated cache pair. The OS filesystem cache remains populated. First-image "
        "time includes loading and preparing the native CT. Warm timing covers completed device images.",
    )
    section(
        "### Compilation and cache",
        compile_diagnostics(compile_result, root, source_digests),
    )
    if competitors:
        section(
            "### Projector measurements",
            "\n\n".join(
                part
                for part in (
                    competitor_table(competitors),
                    table(headers, competitor_rows) if competitor_rows else "",
                )
                if part
            ),
            "Each backend records its algorithm, image error and completion boundary. "
            "ASTRA's resident calls use DLPack; TIGRE records its transfer and backprojection interval.",
        )
    section(
        "## Application timeline",
        timeline_tables(timelines, root, source_digests),
        "Each Nsight Systems timeline captures one warmed operation. "
        "The public boundary is `dpt.project(ImagingVolume, geometry)`: upload, preparation, "
        "validation, three projections and completed CUDA tensor return. File loading and HU "
        "conversion precede capture. The resident iteration computes three views and differentiates "
        "the AP detector-region objective with respect to volume and shared pose. The two other "
        "view cotangents are zero. "
        "NVTX ranges are inclusive CPU intervals. GPU work is attributed by launch correlation "
        "and process, including Torch's autograd worker. GPU-work union measures elapsed device "
        "time across overlapping kernels and transfers; kernel sum adds their individual durations.",
    )
    section(
        "## Kernel counters and rooflines",
        profile_tables(profiles, root, source_digests),
        "FLOPs count measured predicated-thread "
        "ADD + MUL + 2×FMA. Division helpers, comparisons, conversions, atomics and integer/address "
        "work have separate counters. Request/sector bytes and NVIDIA hierarchical-roofline "
        "writeback proxies occupy distinct columns. Unavailable counters are labelled unmeasured. "
        "The roofs use NVIDIA's theoretical peak-sustained model at captured clocks.",
    )
    if all_records:
        section("### Measurement files", result_index(all_records))
        pieces.append("[CSV timing summary](assets/results/release_summary.csv).")
    replace_block(root / "docs/performance.md", "release", "\n\n".join(pieces))
    if update_readme:
        replace_block(
            root / "README.md",
            "release",
            "## Measured performance\n\n" + ", ".join(gpu_names) + ". "
            "Native 512 × 512 × 139 chest CT, 512² detectors, 1024 clipped midpoints/ray. "
            "Entries are median / p95. Forward covers AP/lateral/oblique. Derivatives and "
            "complete Torch iteration use the AP detector-region objective.\n\n"
            + headline(release),
        )
    output = root / "docs/assets/results/release_summary.csv"
    all_numeric = numeric + other_numeric
    if all_numeric:
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_numeric[0]))
            writer.writeheader()
            writer.writerows(all_numeric)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results", type=Path, help="Precision benchmark JSON")
    parser.add_argument("--release-results", type=Path, nargs="*", default=[])
    parser.add_argument("--competitor-results", type=Path, nargs="*", default=[])
    parser.add_argument("--cold-results", type=Path, nargs="*", default=[])
    parser.add_argument("--extra-results", type=Path, nargs="*", default=[])
    parser.add_argument("--optimisation-results", type=Path, nargs="*", default=[])
    parser.add_argument("--experiment-results", type=Path, nargs="*", default=[])
    parser.add_argument("--profile-results", type=Path, nargs="*", default=[])
    parser.add_argument("--timeline-results", type=Path, nargs="*", default=[])
    parser.add_argument("--compile-result", type=Path)
    parser.add_argument(
        "--keep-readme",
        action="store_true",
        help="Update the full manual and retain the README",
    )
    parser.add_argument(
        "--partial-baseline-results",
        type=Path,
        nargs="*",
        default=[],
        help="Explicitly retain complete baseline rows from a later failed run",
    )
    args = parser.parse_args()
    baseline = args.results or args.root / "docs/assets/results/baseline-chest-512.json"
    if "<!-- baseline:start -->" in (args.root / "docs/performance.md").read_text():
        baseline_tables(args.root, baseline, readme=not bool(args.release_results))
    if args.release_results:
        assets = args.root / "docs/assets/results"
        groups = {
            name: load_results(paths, name, assets)
            for name, paths in (
                ("release", args.release_results),
                ("competitor", args.competitor_results),
                ("cold", args.cold_results),
                ("extra", args.extra_results),
                ("optimisation", args.optimisation_results),
                ("experiment", args.experiment_results),
                ("partial-baseline", args.partial_baseline_results),
            )
        }
        publish_release(
            args.root,
            None,
            groups,
            args.profile_results,
            args.timeline_results,
            args.compile_result,
            update_readme=not args.keep_readme,
        )
        print(f"Generated tables from {sum(map(len, groups.values()))} result files.")


if __name__ == "__main__":
    main()
