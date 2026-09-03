"""Fresh-process native CT startup with private Warp and CUDA caches.

Three independent cold-cache pairs per precision are followed by a second
process using that pair's populated caches. Each child also measures warm
resident Projector calls. Existing shared caches are neither cleared nor used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

CHILD_START = time.perf_counter()


def utc():
    return datetime.now(UTC).isoformat()


def write_json(path, result):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest_sources(source):
    entries = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((source / "python/dpt").rglob("*.py"))
    }
    return {
        "sha256": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
        "files": entries,
    }


def child(config_path):
    config = json.loads(config_path.read_text())
    output = Path(config["result"])
    if output.exists():
        raise FileExistsError(output)
    started_at = utc()
    times = {}
    started = time.perf_counter()
    import numpy as np
    import torch
    import warp as wp

    # Match the measured implementation's module identities. Configuration must
    # precede imports which create DPT kernel modules.
    wp.config.lineinfo = True
    wp.config.verbose = True
    from dpt.geometry import BatchedGeometry, DetectorGeometry
    from dpt.imaging import ct_to_attenuation, read_volume
    from dpt.torch import Projector

    times["imports"] = time.perf_counter() - started
    started = time.perf_counter()
    torch.cuda.init()
    wp.init()
    stream = torch.cuda.current_stream()
    stream.synchronize()
    times["cuda_initialisation"] = time.perf_counter() - started
    actual_cache = Path(wp.config.kernel_cache_dir).resolve()
    requested_cache = Path(os.environ["WARP_CACHE_PATH"]).resolve()
    if requested_cache != actual_cache and requested_cache not in actual_cache.parents:
        raise RuntimeError(f"Warp cache escaped isolation: {actual_cache} != {requested_cache}")
    started = time.perf_counter()
    ct = read_volume(config["ct"], units="HU")
    times["file_loading"] = time.perf_counter() - started
    started = time.perf_counter()
    attenuation = ct_to_attenuation(ct, energy_kev=60)
    times["ct_conversion"] = time.perf_counter() - started
    panels = tuple(
        DetectorGeometry(**{k: tuple(v) if isinstance(v, list) else v for k, v in p.items()})
        for p in config["geometry"]
    )
    started = time.perf_counter()
    field = torch.tensor(attenuation.values, dtype=torch.float32, device="cuda:0")
    stream.synchronize()
    times["device_upload"] = time.perf_counter() - started
    started = time.perf_counter()
    module = Projector(
        grid=ct.grid,
        geometry=BatchedGeometry(panels),
        precision=config["precision"],
        samples_per_ray=1024,
    )
    times["module_construction"] = time.perf_counter() - started
    started = time.perf_counter()
    # Split lazy native preparation from the subsequent ordinary public call.
    # Any compilation triggered during preparation remains in this boundary.
    poses = module._poses(field, None, None)
    workspaces = module._prepare(field, poses)
    stream.synchronize()
    times["workspace_preparation"] = time.perf_counter() - started
    started = time.perf_counter()
    image = module(field)
    stream.synchronize()
    times["first_forward"] = time.perf_counter() - started
    first_image_ready = time.perf_counter() - CHILD_START
    wall, device = [], []
    # Warm forward uses the same input/geometry/dtype and actual owning stream.
    for _ in range(config["warm_repeats"]):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        stream.synchronize()
        begun = time.perf_counter()
        start.record(stream)
        image = module(field)
        end.record(stream)
        end.synchronize()
        wall.append((time.perf_counter() - begun) * 1000)
        device.append(start.elapsed_time(end))
    started = time.perf_counter()
    host = image.detach().cpu().numpy()
    times["output_download"] = time.perf_counter() - started
    # Validation and hashes are outside all measured phase intervals.
    if not np.isfinite(host).all():
        raise ArithmeticError("Nonfinite cold-start image")
    from dataclasses import asdict

    if json.loads(json.dumps(asdict(ct.grid))) != config["grid"]:
        raise ValueError("Native CT physical grid changed")
    if hashlib.sha256(Path(config["ct"]).read_bytes()).hexdigest() != config["ct_sha256"]:
        raise ValueError("Native CT file differs from baseline")
    if config.get("reference"):
        reference = np.load(config["reference"], allow_pickle=False)
        delta = host.astype(np.float64) - reference.astype(np.float64)
        error = {
            "max_abs": float(np.abs(delta).max()),
            "rmse": float(np.sqrt(np.mean(delta**2))),
        }
    else:
        error = {"status": "not measured", "reason": "no candidate reference supplied"}
    import importlib.metadata
    import platform

    completed = {
        "schema_version": 1,
        "measurement_started_at": started_at,
        "measurement_finished_at": utc(),
        "precision": config["precision"],
        "cache_state": config["cache_state"],
        "sample": config["sample"],
        "source": digest_sources(Path(config["source"])),
        "source_expected": config["source_digest"],
        "seconds": times,
        "first_device_image_seconds_since_child_entry": first_image_ready,
        "summed_prepare_to_first_image_seconds": sum(
            times[k] for k in times if k != "output_download"
        ),
        "warm": {
            "repeats": config["warm_repeats"],
            "wall_ms": wall,
            "device_ms": device,
            "wall_median_ms": float(np.median(wall)),
            "wall_p95_ms": float(np.percentile(wall, 95)),
            "device_median_ms": float(np.median(device)),
            "device_p95_ms": float(np.percentile(device, 95)),
        },
        "input": {
            "ct_sha256": config["ct_sha256"],
            "grid": config["grid"],
            "geometry": config["geometry"],
            "native": True,
            "resampled": False,
            "sampling": "1024 clipped midpoints per ray",
            "attenuation": "60 keV water-scaled HU; FP64 conversion then FP32 storage",
            "field_sha256": hashlib.sha256(attenuation.values.tobytes()).hexdigest(),
        },
        "output": {
            "shape": list(host.shape),
            "dtype": str(host.dtype),
            "sha256": hashlib.sha256(host.tobytes()).hexdigest(),
            "error_vs_candidate_reference": error,
        },
        "workspace_bytes": sum(w.scratch_bytes for w in workspaces),
        "cache": {
            "warp_requested": str(requested_cache),
            "warp_actual": str(actual_cache),
            "cuda": os.environ["CUDA_CACHE_PATH"],
            "cache_kernels": wp.config.cache_kernels,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "warp-lang", "nibabel")
            },
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "nvidia_smi": subprocess.run(
                ["nvidia-smi", "-q"], capture_output=True, text=True, check=False
            ).stdout,
        },
    }
    if completed["source"]["sha256"] != config["source_digest"]:
        raise RuntimeError("Source changed during cold-start collection")
    write_json(output, completed)
    print("CHILD_RESULT", output, flush=True)


def parse_module_events(log):
    pattern = re.compile(
        "Module (?P<name>.+?) (?P<hash>[0-9a-f]+) load on device .+? took "
        "(?P<ms>[0-9.]+) ms\\s+\\((?P<status>compiled|cached|error)\\)"
    )
    return [
        {
            "name": m["name"],
            "hash": m["hash"],
            "milliseconds": float(m["ms"]),
            "status": m["status"],
        }
        for m in pattern.finditer(log)
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--ct", type=Path)
    p.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument(
        "--baseline-results",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/assets/results/baseline-chest-512.json",
    )
    p.add_argument("--output", type=Path)
    p.add_argument("--modes", nargs="+", choices=("fp32", "fp64"), default=["fp32", "fp64"])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warm-repeats", type=int, default=100)
    p.add_argument(
        "--reference-dir",
        type=Path,
        help="Candidate fp32_depth.npy/fp64_depth.npy images",
    )
    p.add_argument(
        "--probe-script",
        type=Path,
        help="Official NVIDIA warp_compile_probe.py; adds its launch recorder",
    )
    a = p.parse_args()
    if a.child:
        child(a.child)
        return
    if any(x is None for x in (a.ct, a.source, a.baseline_results, a.output)):
        p.error("--ct, --source, --baseline-results and --output are required")
    if a.rounds < 1 or a.warm_repeats < 1:
        p.error("rounds and warm-repeats must be positive")
    if a.output.exists() and any(a.output.iterdir()):
        p.error("output directory must be absent or empty")
    a.output.mkdir(parents=True, exist_ok=True)
    a.source = a.source.resolve()
    a.ct = a.ct.resolve()
    baseline = json.loads(a.baseline_results.read_text())["input"]
    if (
        baseline["samples_per_ray"] != 1024
        or baseline["resampled"]
        or baseline["geometry"][0]["shape"] != [512, 512]
    ):
        raise ValueError(
            "Cold workload needs frozen native chest at 512 squared and 1024 midpoints"
        )
    if hashlib.sha256(a.ct.read_bytes()).hexdigest() != baseline["sha256"]:
        raise ValueError("CT checksum differs from frozen baseline")
    source = digest_sources(a.source)
    result = {
        "schema_version": 1,
        "measurement_started_at": utc(),
        "source": source,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rounds_per_mode": a.rounds,
        "runs": [],
        "cache_policy": (
            "each cold sample has private empty Warp/CUDA directories; warm "
            "process reuses its populated pair; no shared cache clearing"
        ),
        "filesystem_cache": (
            "OS filesystem cache is not purged; cold refers to process and GPU compilation caches"
        ),
        "boundary": (
            "imports, CUDA initialisation, file load, HU conversion, upload, "
            "module/native preparation, first completed device image, warm "
            "completed Projector"
        ),
        "launch_probe": None,
        "instrumented_launches": bool(a.probe_script),
        "timing_use": "compile/launch diagnostic only"
        if a.probe_script
        else "clean startup measurement",
    }
    injection = None
    if a.probe_script:
        spec = importlib.util.spec_from_file_location("nvidia_warp_compile_probe", a.probe_script)
        probe = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = probe
        spec.loader.exec_module(probe)
        injection = probe._SITECUSTOMIZE
        result["launch_probe"] = {
            "script": str(a.probe_script),
            "sha256": hashlib.sha256(a.probe_script.read_bytes()).hexdigest(),
        }
    write_json(a.output / "results.json", result)
    for mode in a.modes:
        for sample in range(a.rounds):
            with tempfile.TemporaryDirectory(prefix=f"dpt-cold-{mode}-{sample}-") as scratch:
                scratch = Path(scratch)
                warp_cache = scratch / "warp"
                cuda_cache = scratch / "cuda"
                warp_cache.mkdir()
                cuda_cache.mkdir()
                env = dict(os.environ)
                env.update(
                    {
                        "WARP_CACHE_PATH": str(warp_cache),
                        "WARP_CACHE_ROOT": str(warp_cache),
                        "CUDA_CACHE_PATH": str(cuda_cache),
                    }
                )
                env.pop("CUDA_CACHE_DISABLE", None)
                pythonpath = [str(a.source / "python")]
                if injection:
                    inject = scratch / "inject"
                    inject.mkdir()
                    (inject / "sitecustomize.py").write_text(injection)
                    pythonpath.insert(0, str(inject))
                if env.get("PYTHONPATH"):
                    pythonpath.append(env["PYTHONPATH"])
                env["PYTHONPATH"] = os.pathsep.join(pythonpath)
                for cache_state in ("cold", "warm_process"):
                    name = f"{mode}-{sample:02d}-{cache_state}"
                    child_result = a.output / f"{name}.json"
                    launch_record = a.output / f"{name}-launches.json"
                    if injection:
                        env["_WARP_PROBE_RECORD"] = str(launch_record.resolve())
                    reference = a.reference_dir / f"{mode}_depth.npy" if a.reference_dir else None
                    config = {
                        "result": str(child_result.resolve()),
                        "source": str(a.source),
                        "source_digest": source["sha256"],
                        "ct": str(a.ct),
                        "ct_sha256": baseline["sha256"],
                        "grid": baseline["grid"],
                        "geometry": baseline["geometry"],
                        "precision": mode,
                        "sample": sample,
                        "cache_state": cache_state,
                        "warm_repeats": a.warm_repeats,
                        "reference": str(reference.resolve()) if reference else None,
                    }
                    config_path = a.output / f"{name}-config.json"
                    write_json(config_path, config)
                    cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--child",
                        str(config_path.resolve()),
                    ]
                    print("Starting", name, utc(), "output", child_result, flush=True)
                    started = time.perf_counter()
                    with (a.output / f"{name}.log").open("w") as log:
                        completed = subprocess.run(
                            cmd,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                    elapsed = time.perf_counter() - started
                    log = (a.output / f"{name}.log").read_text()
                    row = {
                        "precision": mode,
                        "sample": sample,
                        "cache_state": cache_state,
                        "completed_process_seconds": elapsed,
                        "returncode": completed.returncode,
                        "result_file": child_result.name,
                        "log_file": f"{name}.log",
                        "module_events": parse_module_events(log),
                        "cache_was_empty": cache_state == "cold",
                        "warp_cache_files_after": sum(
                            1 for x in warp_cache.rglob("*") if x.is_file()
                        ),
                        "cuda_cache_files_after": sum(
                            1 for x in cuda_cache.rglob("*") if x.is_file()
                        ),
                    }
                    result["runs"].append(row)
                    write_json(a.output / "results.json", result)
                    if completed.returncode:
                        raise RuntimeError(f"{name} failed; inspect {a.output / name}.log")
                    row["measurement"] = json.loads(child_result.read_text())
                    write_json(a.output / "results.json", result)
                    print("Completed", name, "wall seconds", elapsed, flush=True)
    result["summary"] = {}
    for mode in a.modes:
        result["summary"][mode] = {}
        for state in ("cold", "warm_process"):
            rows = [
                r for r in result["runs"] if r["precision"] == mode and r["cache_state"] == state
            ]
            values = [
                r["measurement"]["first_device_image_seconds_since_child_entry"] for r in rows
            ]
            result["summary"][mode][state] = {
                "samples": len(values),
                "first_image_seconds_raw": values,
                "first_image_seconds_median": statistics.median(values),
            }
    result["measurement_finished_at"] = utc()
    write_json(a.output / "results.json", result)
    print("RESULTS", a.output / "results.json", flush=True)


if __name__ == "__main__":
    main()
