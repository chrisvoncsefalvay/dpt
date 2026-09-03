"""Export compact read-only summaries of completed Nsight Systems captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


def union_ns(intervals):
    total = 0
    last = None
    for start, end in sorted(intervals):
        if last is None or start > last:
            total += end - start
        elif end > last:
            total += end - last
        last = max(last or end, end)
    return total


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def synchronisation(name):
    return "Synchronize" in name or "Synchronise" in name


def allocation(name):
    return any(token in name for token in ("MemAlloc", "Malloc", "MemFree", "Free", "MemPool"))


def analyse(report, directory, output, legacy):
    sqlite = directory / (report + ".sqlite")
    metadata_path = directory / (report + ".metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if not legacy and metadata.get("completed_checks_passed") is not True:
        raise ValueError("Capture has no completed checks: " + report)
    con = sqlite3.connect(sqlite.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def rows(table):
        return (
            [dict(row) for row in con.execute('SELECT * FROM "' + table + '"')]
            if table in tables
            else []
        )

    strings = {row["id"]: row["value"] for row in rows("StringIds")}
    ranges = [
        {**row, "name": row["text"] or strings.get(row["textId"], "")}
        for row in rows("NVTX_EVENTS")
        if row["end"] is not None
    ]
    outer_name = "public" if legacy else "operation"
    outers = [row for row in ranges if row["name"] == outer_name]
    if len(outers) != 1:
        raise ValueError(f"Expected one {outer_name!r} range: {report}, found {len(outers)}")
    outer = outers[0]
    ranges = [
        row for row in ranges if outer["start"] <= row["start"] and row["end"] <= outer["end"]
    ]
    apis = []
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        for row in rows(table):
            if outer["start"] <= row["start"] < outer["end"]:
                apis.append({**row, "name": strings[row["nameId"]], "table": table})
    launch_apis = defaultdict(list)
    for row in apis:
        launch_apis[row["correlationId"]].append(row)

    def launched_within(event, span):
        for api in launch_apis[event["correlationId"]]:
            if api["globalTid"] is None:
                continue
            if (
                event.get("globalPid") is not None
                and (api["globalTid"] & ~0xFFFFFF) != event["globalPid"]
            ):
                continue
            if (api["globalTid"] & ~0xFFFFFF) == (span["globalTid"] & ~0xFFFFFF) and span[
                "start"
            ] <= api["start"] < span["end"]:
                return True
        return False

    kernels = rows("CUPTI_ACTIVITY_KIND_KERNEL")
    memcpy = rows("CUPTI_ACTIVITY_KIND_MEMCPY")
    memset = rows("CUPTI_ACTIVITY_KIND_MEMSET")
    copy_labels = {row["id"]: row["label"] for row in rows("ENUM_CUDA_MEMCPY_OPER")}
    memory_labels = {row["id"]: row["label"] for row in rows("ENUM_CUDA_MEM_KIND")}
    voxels = 1
    for count in metadata.get("input", {}).get("grid", {}).get("shape", []):
        voxels *= count
    if voxels == 1 and legacy:
        voxels = 512 * 512 * 139

    def category(kernel):
        name = strings[kernel["demangledName"]]
        if "validate" in name.lower():
            threads = (
                kernel["gridX"]
                * kernel["gridY"]
                * kernel["gridZ"]
                * kernel["blockX"]
                * kernel["blockY"]
                * kernel["blockZ"]
            )
            block = kernel["blockX"] * kernel["blockY"] * kernel["blockZ"]
            if voxels <= threads < voxels + block:
                return "full_volume_validation"
            return "other_validation"
        return "calculation"

    range_rows, kernel_rows, api_rows, transfer_rows = [], [], [], []
    for span in sorted(ranges, key=lambda row: (row["start"], -row["end"])):
        span_apis = [
            row
            for row in apis
            if row["globalTid"] is not None
            and (row["globalTid"] & ~0xFFFFFF) == (span["globalTid"] & ~0xFFFFFF)
            and span["start"] <= row["start"] < span["end"]
        ]
        span_kernels = [row for row in kernels if launched_within(row, span)]
        span_copies = [row for row in memcpy if launched_within(row, span)]
        span_sets = [row for row in memset if launched_within(row, span)]
        gpu = span_kernels + span_copies + span_sets
        sync = [row for row in span_apis if synchronisation(row["name"])]
        alloc = [row for row in span_apis if allocation(row["name"])]
        full = [row for row in span_kernels if category(row) == "full_volume_validation"]
        other = [row for row in span_kernels if category(row) == "other_validation"]
        device_reads = [
            row for row in span_copies if copy_labels[row["copyKind"]] == "Device-to-Host"
        ]
        four_byte_reads = [row for row in device_reads if row["bytes"] == 4]
        common = {"report": report, "range": span["name"]}
        range_rows.append(
            {
                **common,
                "inclusive_nvtx_ms": (span["end"] - span["start"]) / 1e6,
                "start_relative_to_operation_us": (span["start"] - outer["start"]) / 1e3,
                "end_relative_to_operation_us": (span["end"] - outer["start"]) / 1e3,
                "launched_kernel_count": len(span_kernels),
                "launched_kernel_sum_ms": sum(row["end"] - row["start"] for row in span_kernels)
                / 1e6,
                "launched_kernel_union_ms": union_ns(
                    (row["start"], row["end"]) for row in span_kernels
                )
                / 1e6,
                "launched_gpu_work_union_ms": union_ns((row["start"], row["end"]) for row in gpu)
                / 1e6,
                "launched_gpu_work_span_ms": (
                    max(row["end"] for row in gpu) - min(row["start"] for row in gpu)
                )
                / 1e6
                if gpu
                else 0,
                "gpu_work_inside_nvtx_union_ms": union_ns(
                    (max(row["start"], span["start"]), min(row["end"], span["end"]))
                    for row in gpu
                    if row["start"] < span["end"] and row["end"] > span["start"]
                )
                / 1e6,
                "full_volume_validation_count": len(full),
                "full_volume_validation_ms": sum(row["end"] - row["start"] for row in full) / 1e6,
                "other_validation_count": len(other),
                "other_validation_ms": sum(row["end"] - row["start"] for row in other) / 1e6,
                "host_to_device_bytes": sum(
                    row["bytes"]
                    for row in span_copies
                    if copy_labels[row["copyKind"]] == "Host-to-Device"
                ),
                "device_to_host_bytes": sum(row["bytes"] for row in device_reads),
                "device_to_device_bytes": sum(
                    row["bytes"]
                    for row in span_copies
                    if copy_labels[row["copyKind"]] == "Device-to-Device"
                ),
                "memcpy_gpu_sum_ms": sum(row["end"] - row["start"] for row in span_copies) / 1e6,
                "four_byte_readback_count": len(four_byte_reads),
                "four_byte_readback_gpu_ms": sum(
                    row["end"] - row["start"] for row in four_byte_reads
                )
                / 1e6,
                "memset_count": len(span_sets),
                "memset_bytes": sum(row["bytes"] for row in span_sets),
                "memset_gpu_ms": sum(row["end"] - row["start"] for row in span_sets) / 1e6,
                "cuda_api_count": len(span_apis),
                "cuda_api_thread_count": len({row["globalTid"] for row in span_apis}),
                "cuda_api_union_ms": union_ns((row["start"], row["end"]) for row in span_apis)
                / 1e6,
                "synchronise_api_count": len(sync),
                "synchronise_api_sum_ms": sum(row["end"] - row["start"] for row in sync) / 1e6,
                "synchronise_api_union_ms": union_ns((row["start"], row["end"]) for row in sync)
                / 1e6,
                "allocation_free_pool_api_count": len(alloc),
                "allocation_free_pool_api_sum_ms": sum(row["end"] - row["start"] for row in alloc)
                / 1e6,
                "allocation_bytes_available": any("MEMORY_USAGE" in table for table in tables),
            }
        )
        groups = defaultdict(list)
        for row in span_kernels:
            groups[
                (
                    strings[row["demangledName"]],
                    category(row),
                    tuple(row[axis] for axis in ("gridX", "gridY", "gridZ")),
                    tuple(row[axis] for axis in ("blockX", "blockY", "blockZ")),
                    row["registersPerThread"],
                )
            ].append(row)
        for (name, kind, grid, block, registers), group in groups.items():
            total = sum(row["end"] - row["start"] for row in group)
            kernel_rows.append(
                {
                    **common,
                    "category": kind,
                    "name": name,
                    "grid": "x".join(map(str, grid)),
                    "block": "x".join(map(str, block)),
                    "registers_per_thread": registers,
                    "count": len(group),
                    "total_ms": total / 1e6,
                    "mean_us": total / len(group) / 1e3,
                }
            )
        groups = defaultdict(list)
        for row in span_apis:
            groups[row["name"]].append(row)
        for name, group in groups.items():
            total = sum(row["end"] - row["start"] for row in group)
            api_rows.append(
                {
                    **common,
                    "name": name,
                    "synchronise": synchronisation(name),
                    "allocation_free_pool": allocation(name),
                    "count": len(group),
                    "total_ms": total / 1e6,
                    "mean_us": total / len(group) / 1e3,
                }
            )
        groups = defaultdict(list)
        for row in span_copies:
            groups[
                (
                    copy_labels[row["copyKind"]],
                    row["bytes"],
                    memory_labels.get(row["srcKind"]),
                    memory_labels.get(row["dstKind"]),
                )
            ].append(row)
        for (direction, size, source, destination), group in groups.items():
            transfer_rows.append(
                {
                    **common,
                    "direction": direction,
                    "bytes_per_copy": size,
                    "count": len(group),
                    "total_bytes": size * len(group),
                    "source_memory": source,
                    "destination_memory": destination,
                    "gpu_sum_ms": sum(row["end"] - row["start"] for row in group) / 1e6,
                }
            )
    attributed = [row for row in kernels + memcpy + memset if launched_within(row, outer)]
    in_outer = [
        row
        for row in kernels + memcpy + memset
        if row["start"] < outer["end"] and row["end"] > outer["start"]
    ]
    unattributed = [row for row in in_outer if not launched_within(row, outer)]
    if unattributed:
        raise ValueError(f"{report}: {len(unattributed)} GPU events have no attributed launch")
    stats = directory / (report + "-stats_cuda_gpu_kern_sum.csv")
    independently_verified = False
    if stats.exists():
        with stats.open() as handle:
            built_in = list(csv.DictReader(handle))
        assert (
            int(sum(float(row["Instances"]) for row in built_in))
            == range_rows[0]["launched_kernel_count"]
        )
        assert (
            abs(
                sum(float(row["Total Time (ns)"]) for row in built_in) / 1e6
                - range_rows[0]["launched_kernel_sum_ms"]
            )
            < 1e-12
        )
        independently_verified = True
    manifest = {
        "report": report,
        "analysis_at": datetime.now(UTC).isoformat(),
        "measurement_completed_at": metadata.get("profile_completed_at"),
        "source_sha256": metadata.get("source", {}).get("sha256"),
        "profile_harness_sha256": metadata.get("harness_sha256"),
        "baseline_sha256": metadata.get("baseline_sha256"),
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sqlite_path": str(sqlite),
        "sqlite_sha256": hashlib.sha256(sqlite.read_bytes()).hexdigest(),
        "boundary": metadata.get("boundary", "Legacy resident one-view raw-buffer public call"),
        "views": ";".join(metadata.get("views", ["ap"])),
        "volume_voxels": voxels,
        "volume_storage_bytes": voxels * 4,
        "sampling_count": metadata.get("input", {}).get("samples_per_ray", 1024),
        "gpu_events_attributed_to_operation": len(attributed),
        "gpu_events_overlapping_operation_without_matching_launch": len(unattributed),
        "kernel_count_and_sum_agree_with_builtin_nsys_stats": independently_verified,
        "allocation_size_trace": "available"
        if any("MEMORY_USAGE" in table for table in tables)
        else "not collected; API count/time only; cached allocator activity is not fully visible",
        "timing_status": "profiled single capture; not a headline latency or median/p95",
        "attribution": (
            "GPU events matched by launch correlation ID and process; launch API starts inside "
            "the NVTX time window, including Torch autograd worker threads. Captured workload "
            "is a single serial operation with no unrelated process work."
        ),
    }
    for suffix, data in (
        ("ranges", range_rows),
        ("kernels", kernel_rows),
        ("apis", api_rows),
        ("transfers", transfer_rows),
        ("manifest", [manifest]),
    ):
        write_csv(output / f"{report}-{suffix}.csv", data)
    con.close()
    print(report, json.dumps(range_rows[0], sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reports",
        nargs="+",
        default=["fp32-public", "fp32-iteration", "fp64-public", "fp64-iteration"],
    )
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for report in args.reports:
        analyse(report, args.directory, args.output, args.legacy)


if __name__ == "__main__":
    main()
