#!/usr/bin/env python3
"""Fresh-process full-scan startup through the ordinary public Projector.

Each precision uses three independent empty cache pairs by default. A second
process reuses each populated pair. The full 256^3 prepared chest and all 360
512^2 views are fixed by the contract; resampling is not repeated here. Lazy
workspace preparation and Warp compilation belong to the first public call.
Run the parent under the shared GPU lock; it serialises its CUDA children.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from benchmarks.cold_start import parse_module_events
from benchmarks.common import command, digest_sources, utc
from benchmarks.fullscan import array_error, file_sha256, write_json

CHILD_START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
HELPERS = ("fullscan_cold.py", "cold_start.py", "fullscan.py", "fullscan_inputs.py", "common.py")


def harness_hashes():
    return {name: file_sha256(ROOT / "benchmarks" / name) for name in HELPERS}


def validate_input(data, contract_path):
    contract = json.loads(contract_path.read_text())
    metadata_path = data / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if file_sha256(metadata_path) != contract["input_metadata_sha256"]:
        raise ValueError("prepared input metadata differs from the frozen contract")
    if file_sha256(ROOT / "benchmarks/fullscan_inputs.py") != contract["input_helper_sha256"]:
        raise ValueError("prepared input helper differs from the frozen contract")
    if (
        metadata["grid"]["shape"] != [256, 256, 256]
        or metadata["acquisition"]["views"] != 360
        or metadata["acquisition"]["detector_shape"] != [512, 512]
        or len(metadata["geometry"]) != 360
        or metadata["sampling"]["samples_per_ray"] != 1024
        or contract["sampling"]["samples_per_ray"] != 1024
        or metadata["arrays"]["attenuation"]["sha256"] != contract["field"]["file_sha256"]
    ):
        raise ValueError("cold workload must retain the frozen full-scan dimensions and sampling")
    for name, record in metadata["arrays"].items():
        if file_sha256(data / record["file"]) != record["sha256"]:
            raise ValueError(f"prepared {name} file checksum mismatch")
    return contract, metadata


def reference_record(directory, precision, contract_path, metadata_sha):
    manifest_path = directory / "outputs.json"
    result_path = directory / "results.json"
    manifest = json.loads(manifest_path.read_text())
    result = json.loads(result_path.read_text())
    if (
        result["status"] != "complete"
        or not result["source_unchanged"]
        or result["contract"]["sha256"] != file_sha256(contract_path)
        or manifest["precision"] != precision
        or manifest["views"] != 360
        or manifest["input_metadata_sha256"] != metadata_sha
        or result["precision"] != precision
        or result["source"] != result["source_after"]
        or result["source"]["sha256"] != manifest["source_sha256"]
        or result["input"]["metadata_sha256"] != metadata_sha
        or result["input"]["selected_view_indices"] != list(range(360))
        or result["input"]["cotangent"]["sha256"] != manifest["cotangent_sha256"]
    ):
        raise ValueError("reference must be a completed same-mode frozen full-scan result")
    record = manifest["arrays"]["images"]
    path = directory / record["file"]
    if record["shape"] != [360, 512, 512] or record["dtype"] != (
        "float32" if precision == "fp32" else "float64"
    ):
        raise ValueError("reference image shape or precision differs")
    if file_sha256(path) != record["sha256"]:
        raise ValueError("reference image file checksum mismatch")
    return {
        "file": str(path.resolve()),
        "record": record,
        "source_sha256": manifest["source_sha256"],
        "manifest_sha256": file_sha256(manifest_path),
        "result_sha256": file_sha256(result_path),
        "measurement_completed_at": result["completed_at"],
    }


def measure_child(config, result):
    times = result["seconds"]
    started = time.perf_counter()
    import numpy as np
    import torch
    import warp as wp

    wp.config.lineinfo = True
    wp.config.verbose = True
    import dpt
    from benchmarks.fullscan_inputs import load_prepared
    from dpt.torch import Projector

    times["imports"] = time.perf_counter() - started
    if not Path(dpt.__file__).resolve().is_relative_to(Path(config["source"]) / "python/dpt"):
        raise RuntimeError("DPT imported outside the selected source")
    started = time.perf_counter()
    torch.cuda.init()
    wp.init()
    stream = torch.cuda.current_stream()
    stream.synchronize()
    times["cuda_initialisation"] = time.perf_counter() - started
    actual_cache = Path(wp.config.kernel_cache_dir).resolve()
    requested_cache = Path(os.environ["WARP_CACHE_PATH"]).resolve()
    if requested_cache != actual_cache and requested_cache not in actual_cache.parents:
        raise RuntimeError("Warp cache escaped its private directory")
    started = time.perf_counter()
    # Hash verification is separate from this load/preparation boundary. This
    # loads the saved attenuation field; it does not resample the CT again.
    scan = load_prepared(Path(config["data"]), verify_hashes=False)
    times["file_loading_and_preparation"] = time.perf_counter() - started
    started = time.perf_counter()
    field = torch.tensor(scan.volume.values, device="cuda:0", dtype=torch.float32)
    stream.synchronize()
    times["device_upload"] = time.perf_counter() - started
    started = time.perf_counter()
    module = Projector(
        grid=scan.volume.grid,
        geometry=scan.geometry,
        precision=config["precision"],
        samples_per_ray=1024,
    )
    times["module_construction"] = time.perf_counter() - started
    started = time.perf_counter()
    with torch.no_grad():
        first = module(field)
    stream.synchronize()
    times["first_forward"] = time.perf_counter() - started
    result["first_device_image_seconds_since_child_entry"] = time.perf_counter() - CHILD_START
    result["summed_prepare_to_first_image_seconds"] = sum(times.values())
    warm = []
    for repeat in range(config["warm_repeats"]):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        stream.synchronize()
        measured_at = utc()
        begun = time.perf_counter()
        start.record(stream)
        with torch.no_grad():
            image = module(field)
        end.record(stream)
        end.synchronize()
        wall_ms = (time.perf_counter() - begun) * 1000
        device_ms = start.elapsed_time(end)
        warm.append(
            {
                "repeat": repeat,
                "started_at": measured_at,
                "completed_at": utc(),
                "wall_ms": wall_ms,
                "device_ms": device_ms,
            }
        )
    result["warm"] = {
        "repeats": len(warm),
        "samples": warm,
        **{
            f"{boundary}_ms": [r[f"{boundary}_ms"] for r in warm] for boundary in ("wall", "device")
        },
        **{
            f"{boundary}_{stat}_ms": float(
                np.percentile([r[f"{boundary}_ms"] for r in warm], percentile)
            )
            for boundary in ("wall", "device")
            for stat, percentile in (("median", 50), ("p95", 95))
        },
    }
    started = time.perf_counter()
    first_host = first.detach().cpu().numpy()
    warm_host = image.detach().cpu().numpy()
    times["output_download"] = time.perf_counter() - started
    started = time.perf_counter()
    reference = np.load(config["reference"]["file"], mmap_mode="r", allow_pickle=False)
    outputs = {}
    for label, actual in (("first", first_host), ("last_warm", warm_host)):
        error = array_error(actual, reference, np)
        raw_sha = hashlib.sha256(memoryview(actual).cast("B")).hexdigest()
        outputs[label] = {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "raw_sha256": raw_sha,
            "error_vs_reference": error,
            "passed": error["bitwise_equal"]
            and raw_sha == config["reference"]["record"]["raw_sha256"],
        }
    result["output"] = outputs
    contract, metadata = validate_input(Path(config["data"]), Path(config["contract"]))
    if contract != config["contract_settings"] or metadata != scan.metadata:
        raise RuntimeError("frozen inputs changed during measurement")
    if (
        hashlib.sha256(memoryview(scan.volume.values).cast("B")).hexdigest()
        != metadata["arrays"]["attenuation"]["raw_sha256"]
    ):
        raise RuntimeError("loaded attenuation values differ from frozen input")
    if file_sha256(Path(config["reference"]["file"])) != config["reference"]["record"]["sha256"]:
        raise RuntimeError("reference changed during measurement")
    result["source"] = digest_sources(Path(config["source"]))
    if result["source"] != config["source_digest"] or harness_hashes() != config["harness"]:
        raise RuntimeError("source or benchmark helpers changed during measurement")
    times["reference_and_input_verification"] = time.perf_counter() - started
    result["cache"] = {
        **config["cache"],
        "warp_actual": str(actual_cache),
        "cache_kernels": wp.config.cache_kernels,
    }
    result["environment"] = {
        "recorded_at": utc(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "warp-lang", "nibabel")
        },
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "nvidia_smi": command("nvidia-smi", "-q"),
    }
    if not all(row["passed"] for row in outputs.values()):
        raise ArithmeticError("first/warm image differs bitwise from same-mode measured reference")


def child(config_path):
    config = json.loads(config_path.read_text())
    output = Path(config["result"])
    if output.exists():
        raise FileExistsError(output)
    result = {
        "schema_version": 1,
        "status": "running",
        "measurement_started_at": utc(),
        "precision": config["precision"],
        "sample": config["sample"],
        "cache_state": config["cache_state"],
        "seconds": {},
        "source_expected": config["source_digest"]["sha256"],
        "input": config["input"],
        "reference": config["reference"],
        "native_preparation_boundary": "included in first_forward; no private preparation call",
    }
    try:
        measure_child(config, result)
        result["status"] = "passed"
    except BaseException:
        result.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        result["measurement_finished_at"] = utc()
        write_json(output, result)


def run(args):
    source, data, output, contract_path = (
        p.resolve() for p in (args.source, args.data, args.output, args.contract)
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory must be absent or empty")
    contract, metadata = validate_input(data, contract_path)
    metadata_sha = file_sha256(data / "metadata.json")
    references = {
        mode: reference_record(
            getattr(args, f"reference_{mode}").resolve(), mode, contract_path, metadata_sha
        )
        for mode in args.modes
    }
    output.mkdir(parents=True, exist_ok=True)
    source_digest, harness = digest_sources(source), harness_hashes()
    for reference in references.values():
        reference["source_relation"] = (
            "same_source"
            if reference["source_sha256"] == source_digest["sha256"]
            else "cross_source_same_mode"
        )
    result = {
        "schema_version": 1,
        "status": "running",
        "measurement_started_at": utc(),
        "source": source_digest,
        "harness": harness,
        "contract": {"sha256": file_sha256(contract_path), "settings": contract},
        "input": {"metadata_sha256": metadata_sha, "metadata": metadata},
        "references": references,
        "rounds_per_mode": args.rounds,
        "runs": [],
        "instrumented_launches": False,
        "boundary": (
            "public completed forward, including lazy workspaces/JIT and output allocation; "
            "no private preparation"
        ),
        "cache_policy": (
            "independent persistent empty private directories per precision/round; "
            "second process reuses the pair"
        ),
        "filesystem_cache": (
            "OS filesystem cache is not purged; cold means process and GPU compilation caches"
        ),
        "prepared_input_boundary": (
            "saved derived full-chest attenuation is loaded; original CT resampling/calibration "
            "occurred at the recorded preparation time"
        ),
    }
    write_json(output / "results.json", result)
    for mode in args.modes:
        for sample in range(args.rounds):
            cache = {
                name: str(output / "cache" / f"{mode}-{sample:02d}" / name)
                for name in ("warp", "cuda", "inductor", "triton")
            }
            for path in cache.values():
                Path(path).mkdir(parents=True, exist_ok=False)
            env = dict(os.environ)
            env.update(
                {
                    "WARP_CACHE_PATH": cache["warp"],
                    "WARP_CACHE_ROOT": cache["warp"],
                    "CUDA_CACHE_PATH": cache["cuda"],
                    "TORCHINDUCTOR_CACHE_DIR": cache["inductor"],
                    "TRITON_CACHE_DIR": cache["triton"],
                    "PYTHONPATH": os.pathsep.join((str(source / "python"), str(ROOT))),
                }
            )
            env.pop("CUDA_CACHE_DISABLE", None)
            for state in ("cold", "warm_process"):
                name = f"{mode}-{sample:02d}-{state}"
                counts_before = {
                    key: sum(p.is_file() for p in Path(path).rglob("*"))
                    for key, path in cache.items()
                }
                if state == "cold" and any(counts_before.values()):
                    raise RuntimeError("fresh cache directory is not empty")
                config = {
                    "result": str(output / f"{name}.json"),
                    "source": str(source),
                    "source_digest": source_digest,
                    "harness": harness,
                    "data": str(data),
                    "contract": str(contract_path),
                    "contract_settings": contract,
                    "input": result["input"],
                    "reference": references[mode],
                    "precision": mode,
                    "sample": sample,
                    "cache_state": state,
                    "cache": cache,
                    "warm_repeats": args.warm_repeats,
                }
                config_path = output / f"{name}-config.json"
                write_json(config_path, config)
                cmd = [
                    sys.executable,
                    "-m",
                    "benchmarks.fullscan_cold",
                    "--child",
                    str(config_path),
                ]
                started_at, started = utc(), time.perf_counter()
                print("Starting", name, started_at, flush=True)
                log_path = output / f"{name}.log"
                with log_path.open("w") as log:
                    process = subprocess.run(
                        cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                    )
                row = {
                    "precision": mode,
                    "sample": sample,
                    "cache_state": state,
                    "started_at": started_at,
                    "completed_at": utc(),
                    "completed_process_seconds": time.perf_counter() - started,
                    "command": cmd,
                    "returncode": process.returncode,
                    "result_file": f"{name}.json",
                    "log_file": log_path.name,
                    "cache_files_before": counts_before,
                    "cache_files_after": {
                        key: sum(p.is_file() for p in Path(path).rglob("*"))
                        for key, path in cache.items()
                    },
                    "module_events": parse_module_events(log_path.read_text()),
                }
                if Path(config["result"]).exists():
                    row["measurement"] = json.loads(Path(config["result"]).read_text())
                result["runs"].append(row)
                if process.returncode or row.get("measurement", {}).get("status") != "passed":
                    result.update(status="failed", measurement_finished_at=utc())
                    write_json(output / "results.json", result)
                    raise RuntimeError(f"{name} failed; inspect {log_path}")
                write_json(output / "results.json", result)
                print("Completed", name, row["completed_process_seconds"], "seconds", flush=True)
    result["summary"] = {}
    for mode in args.modes:
        result["summary"][mode] = {}
        for state in ("cold", "warm_process"):
            rows = [
                r for r in result["runs"] if (r["precision"], r["cache_state"]) == (mode, state)
            ]
            first = [r["measurement"]["first_device_image_seconds_since_child_entry"] for r in rows]
            result["summary"][mode][state] = {
                "samples": len(first),
                "first_image_seconds_raw": first,
                "first_image_seconds_median": statistics.median(first),
                "module_event_counts": [
                    {
                        kind: sum(e["status"] == kind for e in r["module_events"])
                        for kind in ("compiled", "cached", "error")
                    }
                    for r in rows
                ],
            }
    result.update(status="complete", measurement_finished_at=utc())
    write_json(output / "results.json", result)
    print("RESULTS", output / "results.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument(
        "--reference-fp32", type=Path, help="Completed same-mode fullscan result directory"
    )
    parser.add_argument(
        "--reference-fp64", type=Path, help="Completed same-mode fullscan result directory"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--modes", choices=("fp32", "fp64"), nargs="+", default=["fp32", "fp64"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warm-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return
    if any(getattr(args, name) is None for name in ("source", "data", "contract", "output")):
        parser.error("--source, --data, --contract and --output are required")
    if any(getattr(args, f"reference_{mode}") is None for mode in args.modes):
        parser.error(
            "each selected precision needs its --reference-fp32/--reference-fp64 directory"
        )
    if args.rounds < 1 or args.warm_repeats < 1 or len(set(args.modes)) != len(args.modes):
        parser.error("rounds/repeats must be positive and modes must be unique")
    run(args)


if __name__ == "__main__":
    main()
