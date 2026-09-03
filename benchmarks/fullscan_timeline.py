"""Summarise completed full-scan Nsight Systems probe ranges, read-only.

Correlate CUDA launches across the main and autograd worker threads. Inclusive
NVTX time, kernel sums, activity unions and host API waits are separate measures;
they cannot be added into a latency decomposition. These are profiled captures,
not clean benchmark medians. Gaps mean no *attributed* activity was traced and do
not prove that the physical GPU was idle or that the host was the sole cause.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from benchmarks.analyse_timeline import allocation, synchronisation, union_ns, write_csv


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merged_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def gaps_inside(intervals, start, end):
    gaps, cursor = [], start
    for first, last in merged_intervals(
        (max(a, start), min(b, end)) for a, b in intervals if a < end and b > start
    ):
        if first > cursor:
            gaps.append((cursor, first))
        cursor = last
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def kernel_category(row, name, voxels):
    lower = name.lower()
    if "validate" in lower:
        block = row["blockX"] * row["blockY"] * row["blockZ"]
        threads = block * row["gridX"] * row["gridY"] * row["gridZ"]
        if voxels <= threads < voxels + block:
            return "full_volume_validation"
        return "other_validation"
    if "get_forward__locals__forward" in name:
        return "projection_forward"
    if "get_vjp__locals__vjp" in name:
        return "projection_vjp"
    if "reduce_kernel__locals__reduce" in name or "finish_kernel__locals__finish" in name:
        return "pose_reduction"
    if "CUDAFunctor_add<float>" in name:
        return "torch_fp32_add"
    return "other_tensor_work"


def requested_gradients(metadata):
    """Older probes always requested both derivatives. New probes must be explicit."""
    keys = ("gradients", "requested_gradients", "pose_requires_grad")
    markers = (
        *keys,
        "gradient_shapes",
        "gradient_dtypes",
        "pose_representation",
        "target_source_files",
        "nvtx_ranges",
        "current_torch_stream",
    )
    if not any(key in metadata for key in markers):
        return {
            "gradients": "both",
            "requested_gradients": ["volume", "pose"],
            "pose_requires_grad": True,
            "gradient_metadata": "legacy_joint_default",
        }
    if not all(key in metadata for key in keys):
        raise ValueError("incomplete requested-gradient metadata")
    mode = metadata["gradients"]
    targets = ["volume", "pose"] if mode == "both" else ["volume"]
    if not (
        mode in ("volume", "both")
        and metadata["requested_gradients"] == targets
        and metadata["pose_requires_grad"] is (mode == "both")
    ):
        raise ValueError("inconsistent requested-gradient metadata")
    return {
        "gradients": mode,
        "requested_gradients": targets,
        "pose_requires_grad": mode == "both",
        "gradient_metadata": "explicit",
    }


def analyse(sqlite_path, metadata_path, output, profile_harness=None, seed_description=None):
    metadata = json.loads(metadata_path.read_text())
    if not metadata.get("completed_at") or not metadata.get("source", {}).get("sha256"):
        raise ValueError("a completed probe.json with a source digest is required")
    harness_digest = sha256(profile_harness) if profile_harness else None
    if metadata.get("harness_sha256") and harness_digest != metadata["harness_sha256"]:
        raise ValueError("the supplied captured harness does not match the probe metadata")
    gradients = requested_gradients(metadata)
    seed = {
        "description": metadata.get("cotangent") or seed_description or "not recorded",
        "sha256": metadata.get("cotangent_sha256"),
        "description_source": "capture metadata"
        if metadata.get("cotangent")
        else "analyst description of archived harness"
        if seed_description
        else "unknown",
    }
    output.mkdir(parents=True, exist_ok=False)
    connection = sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }

    def rows(table):
        if table not in tables:
            return []
        return [dict(row) for row in connection.execute('SELECT * FROM "' + table + '"')]

    strings = {row["id"]: row["value"] for row in rows("StringIds")}
    ranges = []
    occurrences = defaultdict(int)
    for row in sorted(rows("NVTX_EVENTS"), key=lambda row: row["start"]):
        name = row["text"] or strings.get(row["textId"], "")
        if row["end"] is not None and name in ("probe_forward", "probe_backward"):
            occurrences[name] += 1
            ranges.append({**row, "name": name, "occurrence": occurrences[name]})
    if not all(occurrences[name] for name in ("probe_forward", "probe_backward")):
        raise ValueError("capture must contain probe_forward and probe_backward NVTX ranges")
    for previous, current in pairwise(ranges):
        if current["start"] < previous["end"]:
            raise ValueError("overlapping probe ranges are outside this serial attribution model")
    if any(span.get("globalTid") is None for span in ranges):
        raise ValueError("each probe range must identify its NVTX thread and process")
    probe_processes = {span["globalTid"] & ~0xFFFFFF for span in ranges}
    if len(probe_processes) != 1:
        raise ValueError("serial probe ranges must belong to one captured process")
    probe_process = next(iter(probe_processes))
    apis = [
        {**row, "name": strings[row["nameId"]], "table": table}
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")
        for row in rows(table)
    ]
    launches = defaultdict(list)
    for api in apis:
        if api.get("globalTid") is not None:
            launches[(api["correlationId"], api["globalTid"] & ~0xFFFFFF)].append(api)
    events = [
        {**row, "kind": kind}
        for kind, table in (
            ("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"),
            ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
            ("memset", "CUPTI_ACTIVITY_KIND_MEMSET"),
        )
        for row in rows(table)
    ]
    foreign_overlaps = [
        event
        for event in events
        if event.get("globalPid") != probe_process
        and any(event["start"] < span["end"] and event["end"] > span["start"] for span in ranges)
    ]
    if foreign_overlaps:
        raise ValueError(
            "GPU activity from another or unidentified process overlaps a probe operation"
        )
    copy_labels = {row["id"]: row["label"] for row in rows("ENUM_CUDA_MEMCPY_OPER")}
    voxels = 1
    for count in metadata["input_metadata"]["grid"]["shape"]:
        voxels *= count

    def launched_within(event, span):
        pid = span["globalTid"] & ~0xFFFFFF
        if event.get("globalPid") != pid:
            return False
        return any(
            span["start"] <= api["start"] < span["end"]
            for api in launches[(event["correlationId"], pid)]
        )

    range_rows, kernel_rows, api_rows, transfer_rows, gap_rows = [], [], [], [], []
    attributed_ids = set()
    for span in ranges:
        prefix = {"range": span["name"], "occurrence": span["occurrence"]}
        selected = [event for event in events if launched_within(event, span)]
        attributed_ids.update(id(event) for event in selected)
        kernels = [event for event in selected if event["kind"] == "kernel"]
        copies = [event for event in selected if event["kind"] == "memcpy"]
        sets = [event for event in selected if event["kind"] == "memset"]
        span_apis = [
            api
            for api in apis
            if api.get("globalTid") is not None
            and api["globalTid"] & ~0xFFFFFF == span["globalTid"] & ~0xFFFFFF
            and span["start"] <= api["start"] < span["end"]
        ]
        sync = [api for api in span_apis if synchronisation(api["name"])]
        alloc = [api for api in span_apis if allocation(api["name"])]
        spans = [(event["start"], event["end"]) for event in selected]
        gaps = gaps_inside(spans, span["start"], span["end"])
        categories = defaultdict(list)
        for kernel in kernels:
            categories[kernel_category(kernel, strings[kernel["demangledName"]], voxels)].append(
                kernel
            )
        overlapping_unattributed = [
            event
            for event in events
            if event["start"] < span["end"]
            and event["end"] > span["start"]
            and not launched_within(event, span)
        ]
        worker_kernels = [
            kernel
            for kernel in kernels
            if any(
                api["globalTid"] != span["globalTid"]
                and span["start"] <= api["start"] < span["end"]
                for api in launches[(kernel["correlationId"], probe_process)]
            )
        ]
        row = {
            **prefix,
            "process_global_pid": probe_process,
            "nvtx_global_tid": span["globalTid"],
            "cuda_api_global_tids": sorted({api["globalTid"] for api in span_apis}),
            "kernels_launched_on_other_process_threads": len(worker_kernels),
            "foreign_process_gpu_events_overlapping_nvtx": 0,
            "nvtx_start_ns": span["start"],
            "nvtx_end_ns": span["end"],
            "inclusive_nvtx_ms": (span["end"] - span["start"]) / 1e6,
            "kernel_count": len(kernels),
            "kernel_sum_ms": sum(k["end"] - k["start"] for k in kernels) / 1e6,
            "kernel_union_ms": union_ns((k["start"], k["end"]) for k in kernels) / 1e6,
            "gpu_activity_union_ms": union_ns(spans) / 1e6,
            "gpu_activity_inside_nvtx_union_ms": union_ns(
                (max(a, span["start"]), min(b, span["end"]))
                for a, b in spans
                if a < span["end"] and b > span["start"]
            )
            / 1e6,
            "no_attributed_gpu_activity_inside_nvtx_ms": sum(b - a for a, b in gaps) / 1e6,
            "no_attributed_gpu_activity_gap_count": len(gaps),
            "longest_no_attributed_gpu_activity_gap_ms": max((b - a for a, b in gaps), default=0)
            / 1e6,
            "gpu_tail_after_nvtx_ms": max(
                0, max((b for _, b in spans), default=span["end"]) - span["end"]
            )
            / 1e6,
            "unattributed_gpu_events_overlapping_nvtx": len(overlapping_unattributed),
            "cuda_api_count": len(span_apis),
            "cuda_api_threads": len({api["globalTid"] for api in span_apis}),
            "cuda_api_union_ms": union_ns((a["start"], a["end"]) for a in span_apis) / 1e6,
            "synchronise_api_count": len(sync),
            "synchronise_api_sum_ms": sum(a["end"] - a["start"] for a in sync) / 1e6,
            "synchronise_api_union_ms": union_ns((a["start"], a["end"]) for a in sync) / 1e6,
            "allocation_free_pool_api_count": len(alloc),
            "allocation_free_pool_api_sum_ms": sum(a["end"] - a["start"] for a in alloc) / 1e6,
            "memset_count": len(sets),
            "memset_bytes": sum(a["bytes"] for a in sets),
            "memset_gpu_sum_ms": sum(a["end"] - a["start"] for a in sets) / 1e6,
            "memcpy_count": len(copies),
            "memcpy_gpu_sum_ms": sum(a["end"] - a["start"] for a in copies) / 1e6,
            "four_byte_d2h_count": sum(
                a["bytes"] == 4 and copy_labels.get(a["copyKind"]) == "Device-to-Host"
                for a in copies
            ),
        }
        for category in (
            "full_volume_validation",
            "other_validation",
            "projection_forward",
            "projection_vjp",
            "pose_reduction",
            "torch_fp32_add",
            "other_tensor_work",
        ):
            group = categories[category]
            row[category + "_count"] = len(group)
            row[category + "_sum_ms"] = sum(k["end"] - k["start"] for k in group) / 1e6
        range_rows.append(row)
        for index, (first, last) in enumerate(gaps):
            gap_rows.append(
                {
                    **prefix,
                    "gap": index,
                    "start_relative_to_nvtx_ms": (first - span["start"]) / 1e6,
                    "duration_ms": (last - first) / 1e6,
                }
            )
        kernel_groups = defaultdict(list)
        for kernel in kernels:
            name = strings[kernel["demangledName"]]
            kernel_groups[
                (
                    name,
                    kernel_category(kernel, name, voxels),
                    kernel["registersPerThread"],
                    tuple(kernel[key] for key in ("gridX", "gridY", "gridZ")),
                    tuple(kernel[key] for key in ("blockX", "blockY", "blockZ")),
                )
            ].append(kernel)
        for (name, category, registers, grid, block), group in kernel_groups.items():
            kernel_rows.append(
                {
                    **prefix,
                    "name": name,
                    "category": category,
                    "count": len(group),
                    "sum_ms": sum(k["end"] - k["start"] for k in group) / 1e6,
                    "registers_per_thread": registers,
                    "grid": "x".join(map(str, grid)),
                    "block": "x".join(map(str, block)),
                }
            )
        api_groups = defaultdict(list)
        for api in span_apis:
            api_groups[api["name"]].append(api)
        for name, group in api_groups.items():
            api_rows.append(
                {
                    **prefix,
                    "name": name,
                    "count": len(group),
                    "sum_ms": sum(a["end"] - a["start"] for a in group) / 1e6,
                    "synchronise": synchronisation(name),
                    "allocation_free_pool": allocation(name),
                }
            )
        transfer_groups = defaultdict(list)
        for event in copies:
            transfer_groups[(copy_labels.get(event["copyKind"], "unknown"), event["bytes"])].append(
                event
            )
        for (direction, size), group in transfer_groups.items():
            transfer_rows.append(
                {
                    **prefix,
                    "direction": direction,
                    "bytes_per_copy": size,
                    "count": len(group),
                    "total_bytes": size * len(group),
                    "gpu_sum_ms": sum(a["end"] - a["start"] for a in group) / 1e6,
                }
            )
    capture_kernels = [event for event in events if event["kind"] == "kernel"]
    checks = {
        "one_probe_process": True,
        "no_foreign_process_gpu_overlap": not foreign_overlaps,
        "foreign_process_gpu_overlap_count": len(foreign_overlaps),
        "kernels_launched_on_other_process_threads": sum(
            row["kernels_launched_on_other_process_threads"] for row in range_rows
        ),
        "all_gpu_events_attributed": len(attributed_ids) == len(events),
        "capture_gpu_event_count": len(events),
        "attributed_gpu_event_count": len(attributed_ids),
        "capture_kernel_count": len(capture_kernels),
        "summed_range_kernel_count": sum(row["kernel_count"] for row in range_rows),
        "capture_kernel_sum_ns": sum(k["end"] - k["start"] for k in capture_kernels),
        "summed_range_kernel_sum_ns": sum(
            k["end"] - k["start"] for k in capture_kernels if id(k) in attributed_ids
        ),
    }
    result = {
        "schema_version": 1,
        "attribution_scope": {
            "kind": "process",
            "process_global_pid": probe_process,
            "nvtx_global_tids": sorted({span["globalTid"] for span in ranges}),
            "join": "CUDA correlationId and globalPid to same-process API starts inside NVTX",
            "includes_autograd_worker_threads": True,
            "operation_boundary": "serial completed operation marked by the captured harness",
            "thread_identity_encoding": "globalPid = globalTid with its lower 24 bits cleared",
            "foreign_process_overlap_policy": "reject the analysis",
            "nvtx_note": (
                "Attribution spans all threads of this process; a push/pop marker's CPU thread "
                "does not restrict which correlated CUDA launches belong to the operation."
            ),
        },
        "analysed_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": metadata["source"]["sha256"],
        "measurement_started_at": metadata.get("measured_at"),
        "measurement_completed_at": metadata["completed_at"],
        "views": metadata["views"],
        "precision": metadata["precision"],
        "input_metadata": metadata["input_metadata"],
        "probe_status_counts": metadata.get("status_counts"),
        **gradients,
        "probe_workload": (
            "both field/pose active. Identity pose. Seed: "
            if gradients["gradients"] == "both"
            else "volume derivative only. Fixed identity geometry without a pose tensor. Seed: "
        )
        + seed["description"],
        "cotangent": seed,
        "view_indices": metadata.get("view_indices"),
        "timing_status": "single profiled interval per occurrence; not a clean median/p95",
        "sqlite": {"file": str(sqlite_path), "sha256": sha256(sqlite_path)},
        "metadata": {"file": str(metadata_path), "sha256": sha256(metadata_path)},
        "analysis_script_sha256": sha256(Path(__file__)),
        "profile_harness_sha256": harness_digest,
        "profile_harness_file": str(profile_harness) if profile_harness else None,
        "allocation_bytes_trace_available": any("MEMORY_USAGE" in table for table in tables),
        "method": (
            "CUDA activity joined by correlation ID and process to API starts within the NVTX "
            "range on any process thread. GPU tails remain attributed to the initiating range. "
            "Kernel sums, busy unions and inclusive API waits overlap and must not be added. "
            "No-attributed-activity gaps are trace gaps, "
            "not proof of physical GPU idle or host starvation."
        ),
        "ranges": range_rows,
        "checks": checks,
    }
    for suffix, data in (
        ("ranges", range_rows),
        ("kernels", kernel_rows),
        ("apis", api_rows),
        ("transfers", transfer_rows),
        ("gaps", gap_rows),
    ):
        write_csv(output / (suffix + ".csv"), data)
    (output / "timeline.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    connection.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument(
        "--metadata", type=Path, required=True, help="Matching completed probe.json"
    )
    parser.add_argument("--profile-harness", type=Path)
    parser.add_argument(
        "--seed-description",
        help="Description verified from an archived harness when old metadata omits the seed",
    )
    parser.add_argument("--output", type=Path, required=True, help="New compact-output directory")
    args = parser.parse_args()
    result = analyse(
        args.sqlite, args.metadata, args.output, args.profile_harness, args.seed_description
    )
    print(json.dumps({"ranges": result["ranges"], "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    main()
