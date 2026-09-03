# ruff: noqa: E402, B023
# Startup is timed before imports; each benchmark closure executes within its mode loop.
"""Measure the reviewed projector on native CT without changing its kernels."""

from __future__ import annotations

import sys
from pathlib import Path

# The chosen checkout supplies the timed implementation, including legacy modes.
selected_source = next(
    (value.split("=", 1)[1] for value in sys.argv[1:] if value.startswith("--source=")), None
)
if "--source" in sys.argv and sys.argv.index("--source") + 1 < len(sys.argv):
    selected_source = sys.argv[sys.argv.index("--source") + 1]
if selected_source:
    sys.path.insert(0, str(Path(selected_source).resolve() / "python"))

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

START = time.perf_counter()
import numpy as np
import torch
import warp as wp

# Set before DPT kernel modules are imported or created.
wp.config.lineinfo = True
from dpt.geometry import ConeBeamIntrinsics, RigidTransform
from dpt.imaging_io import read_nifti
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp


def utc():
    return datetime.now(UTC).isoformat()


def command(*args):
    r = subprocess.run(args, capture_output=True, text=True)
    return {
        "command": list(args),
        "returncode": r.returncode,
        "stdout": r.stdout,
        "stderr": r.stderr,
    }


def digest_sources(source):
    entries = {}
    for p in sorted((source / "python" / "dpt").rglob("*.py")):
        entries[str(p.relative_to(source))] = hashlib.sha256(p.read_bytes()).hexdigest()
    combined = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {
        "sha256": combined,
        "files": entries,
        "revision": command("git", "-C", str(source), "rev-parse", "HEAD"),
    }


def measure(fn, stream, rounds, repeats, label):
    fn()
    stream.synchronize()
    started = time.perf_counter()
    fn()
    stream.synchronize()
    pilot = time.perf_counter() - started
    count = repeats if pilot < 0.2 else min(repeats, 10)
    results = []
    for rd in range(rounds):
        wall, gpu = [], []
        timestamp = utc()
        for _ in range(count):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            stream.synchronize()
            t0 = time.perf_counter()
            begin.record(stream)
            fn()
            end.record(stream)
            end.synchronize()
            wall.append((time.perf_counter() - t0) * 1e3)
            gpu.append(begin.elapsed_time(end))
        results.append({"round": rd, "measured_at": timestamp, "wall_ms": wall, "device_ms": gpu})
        print(label, "round", rd + 1, "n", count, "median device ms", np.median(gpu), flush=True)
    device = [x for r in results for x in r["device_ms"]]
    wall = [x for r in results for x in r["wall_ms"]]
    return {
        "label": label,
        "rounds": results,
        "repeats_per_round": count,
        "device_median_ms": float(np.median(device)),
        "device_p95_ms": float(np.percentile(device, 95)),
        "wall_median_ms": float(np.median(wall)),
        "wall_p95_ms": float(np.percentile(wall, 95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--shape", type=int, nargs=2, default=[512, 512])
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--modes", nargs="+", default=["float64", "mixed_float32", "fast_float32"])
    ap.add_argument("--derivatives", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {
        "schema_version": 1,
        "measurement_started_at": utc(),
        "source": digest_sources(args.source),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": {
                n: importlib.metadata.version(n) for n in ["numpy", "torch", "warp-lang", "nibabel"]
            },
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "nvidia_smi_before": command("nvidia-smi", "-q"),
        },
        "import_seconds": time.perf_counter() - START,
        "measurements": [],
        "modes": {},
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n"
        )

    started = time.perf_counter()
    ct = read_nifti(args.ct, value_units="HU")
    results["file_loading_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    # One field, identically rounded to FP32 before every precision comparison.
    mu = (np.maximum(1.0 + ct.values / 1000.0, 0.0) * 0.02059).astype(np.float32)
    results["ct_conversion_seconds"] = time.perf_counter() - started
    grid = ct.grid
    nz, ny, nx = grid.shape
    centre = grid.grid_to_object(((nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2))
    # Circle around the physical RAS centre; detector covers a conservative
    # volume bounding sphere, including perspective enlargement at near faces.
    diagonal = math.sqrt(
        sum((n * s) ** 2 for n, s in zip((nx, ny, nz), grid.spacing_mm, strict=True))
    )
    sod, sdd = max(1000.0, 2 * diagonal), max(1500.0, 3 * diagonal)
    width = 1.05 * diagonal * sdd / (sod - diagonal / 2)
    rows, columns = args.shape
    acquisition = ConeBeamIntrinsics(
        sod, sdd, (rows, columns), (width / columns, width / rows), centre_mm=centre
    )
    names = ["ap", "lateral", "oblique"]
    panels = [acquisition.view(a) for a in [math.pi, math.pi / 2, 3 * math.pi / 4]]
    results["input"] = {
        "file": str(args.ct),
        "sha256": hashlib.sha256(args.ct.read_bytes()).hexdigest(),
        "grid": asdict(grid),
        "native_shape_zyx": list(ct.shape),
        "resampled": False,
        "units": "HU",
        "attenuation_model": "water-scaled monoenergetic",
        "energy_kev": 60,
        "mu_water_mm_inverse": 0.02059,
        "field_dtype": "float32",
        "geometry": [asdict(p) for p in panels],
        "view_names": names,
        "samples_per_ray": args.samples,
        "integration": "clipped midpoint",
        "source_object_distance_mm": sod,
        "source_detector_distance_mm": sdd,
        "detector_pitch_column_row_mm": list(acquisition.spacing_mm),
        "pose": list(RigidTransform().packed()),
    }
    wp.init()
    stream = torch.cuda.current_stream()
    ws_stream = wp.stream_from_torch(stream)
    started = time.perf_counter()
    field = wp.array(mu.ravel(), dtype=wp.float32, device="cuda:0")
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cuda:0")
    wp.synchronize_device("cuda:0")
    results["device_upload_seconds"] = time.perf_counter() - started
    reference = None
    save()
    for mode in args.modes:
        print("Preparing", mode, utc(), flush=True)
        started = time.perf_counter()
        spec = ProjectionSpec(
            args.samples,
            precision="float64",
            active_pose=False,
            active_volume=True,
            sampling_precision=mode,
        )
        workspaces = [
            prepare_projection(grid, p, spec, device="cuda:0", stream=ws_stream) for p in panels
        ]
        outputs = [wp.empty(p.pixels, dtype=wp.float64, device="cuda:0") for p in panels]
        wp.synchronize_device("cuda:0")
        info = {
            "spec": asdict(spec),
            "preparation_seconds": time.perf_counter() - started,
            "pose_vjp": "unsupported by existing faster sampling modes"
            if mode != "float64"
            else "measured separately",
            "output_dtype": "float64",
            "workspace_bytes": sum(w.scratch_bytes for w in workspaces),
        }

        def forward(validate=False):
            for w, out in zip(workspaces, outputs, strict=True):
                project_optical_depth(
                    field, pose, out_L=out, workspace=w, validate=validate, stream=ws_stream
                )

        started = time.perf_counter()
        forward(True)
        stream.synchronize()
        info["cold_forward_seconds"] = time.perf_counter() - started
        images = np.stack([out.numpy().reshape(rows, columns) for out in outputs])
        np.save(args.output / f"{mode}_depth.npy", images)
        if mode == "float64":
            reference = images
        if reference is not None:
            error = images - reference
            info["image_error_vs_strict"] = {
                "max_abs": float(np.max(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(reference)),
            }
        results["modes"][mode] = info
        results["measurements"].append(
            measure(forward, stream, args.rounds, args.repeats, mode + "/resident_forward_3views")
        )
        results["measurements"].append(
            measure(
                lambda: forward(True),
                stream,
                args.rounds,
                args.repeats,
                mode + "/checked_public_forward_3views",
            )
        )

        def completed():
            forward(True)
            return [out.numpy() for out in outputs]

        results["measurements"].append(
            measure(
                completed,
                stream,
                args.rounds,
                args.repeats,
                mode + "/completed_numpy_user_call_3views",
            )
        )
        if args.derivatives:
            w = workspaces[0]
            region = np.zeros((rows, columns), dtype=np.float64)
            region[rows * 5 // 16 : rows * 11 // 16, columns * 5 // 16 : columns * 11 // 16] = 1
            region /= region.sum()
            cotangent = wp.array(region.ravel(), dtype=wp.float64, device="cuda:0")
            grad = wp.empty(grid.voxels, dtype=wp.float64, device="cuda:0")

            def volume_vjp():
                with wp.ScopedStream(ws_stream):
                    grad.zero_()
                projection_vjp(
                    field,
                    pose,
                    cotangent=cotangent,
                    workspace=w,
                    out_mu=grad,
                    validate=False,
                    accumulate=True,
                    volume_gradient_precision="float64",
                    check_volume_gradient=False,
                    stream=ws_stream,
                )

            results["measurements"].append(
                measure(
                    volume_vjp, stream, args.rounds, args.repeats, mode + "/resident_volume_vjp_ap"
                )
            )
            volume_vjp()
            wp.synchronize_device("cuda:0")
            w.check_status()
            gradient = grad.numpy().reshape(grid.shape)
            if not np.isfinite(gradient).all():
                raise RuntimeError("Nonfinite completed volume gradient")
            lhs = float(np.sum(images[0] * region))
            rhs = float(np.sum(mu.astype(np.float64) * gradient))
            info["volume_adjoint"] = {
                "inner_product_forward": lhs,
                "inner_product_adjoint": rhs,
                "relative_discrepancy": abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-15),
            }
            np.save(args.output / f"{mode}_sensitivity.npy", gradient.astype(np.float32))
        save()
    results["measurement_finished_at"] = utc()
    results["environment"]["nvidia_smi_after"] = command("nvidia-smi", "-q")
    save()
    print("RESULTS", args.output / "results.json", flush=True)


if __name__ == "__main__":
    main()
