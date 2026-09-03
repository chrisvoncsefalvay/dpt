"""Profile one complete warmed real-CT operation; never emit headline timing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path


def utc():
    return datetime.now(UTC).isoformat()


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {
        "command": list(args),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def source_digest(source):
    files = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((source / "python/dpt").rglob("*.py"))
    }
    if not files:
        raise ValueError("No DPT Python sources under " + str(source))
    return {
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "files": files,
        "revision": command("git", "-C", str(source), "rev-parse", "HEAD"),
    }


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ct", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--mode", choices=("fp32", "fp64", "float32", "float64"), default="fp64")
    parser.add_argument(
        "--operation",
        choices=("forward", "volume", "pose", "public", "raw-public", "iteration", "material-vjp"),
        default="forward",
    )
    parser.add_argument("--views", nargs="+", choices=("ap", "lateral", "oblique"), default=["ap"])
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.warmup < 1:
        parser.error("warmup must be positive")
    if len(args.views) != len(set(args.views)):
        parser.error("views must be distinct")
    return args


def material_profile(args, record, save):
    """Capture the exact material closure used by the clean benchmark.

    Preparation and the spectral seed use ReleaseBenchmark.material directly.
    The override replaces only its timing loop; no basis or derivative formula
    is duplicated here. The sentinel stops before unrelated later operations.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.common import load_runtime
    from benchmarks.real_ct import ReleaseBenchmark

    base = load_runtime()

    class CapturedError(Exception):
        pass

    class MaterialProfile(ReleaseBenchmark):
        def measure(self, name, call, metadata, *, check=None):
            if not name.endswith("/material_fused_material_vjp"):
                call()
                self.stream.synchronize()
                if check is not None:
                    check()
                return
            for _ in range(args.warmup):
                call()
                self.stream.synchronize()
                if check is not None:
                    check()
            record.update(
                {
                    "input": self.input,
                    "material_configuration": metadata,
                    "resolved_geometry": metadata["geometry"],
                    "workspace_bytes": metadata["workspace_bytes"],
                    "boundary": metadata["boundary"],
                    "requested_gradients": metadata["requested_gradients"],
                    "dtypes": {
                        "field": "float32",
                        "paths": metadata["gradient_storage"],
                        "path_cotangent": metadata["gradient_storage"],
                        "volume_gradient": metadata["gradient_storage"],
                        "pose_gradient": "float64",
                    },
                    "warmup_completed_at": utc(),
                }
            )
            save()
            self.torch.cuda.synchronize()
            self.torch.cuda.cudart().cudaProfilerStart()
            try:
                self.torch.cuda.nvtx.range_push("operation")
                try:
                    # Includes field/pose gradient zeroing and all reductions.
                    call()
                    self.torch.cuda.synchronize()
                finally:
                    self.torch.cuda.nvtx.range_pop()
            finally:
                self.torch.cuda.cudart().cudaProfilerStop()
            if check is not None:
                check()
            self.stream.synchronize()
            record["profile_completed_at"] = utc()
            record["completed_checks_passed"] = True
            raise CapturedError

    config = record["input"]
    benchmark_args = argparse.Namespace(
        ct=args.ct or Path(config["file"]),
        anatomy="chest",
        source=args.source.resolve(),
        baseline_results=args.baseline,
        output=args.metadata.parent / (args.metadata.stem + ".preparation"),
        shape=(512, 512),
        modes=[record["precision"]],
        cases=["material"],
        view="ap",
        rounds=1,
        repeats=1,
        long_repeats=1,
        long_threshold_seconds=0.2,
        nvml_interval_seconds=0.05,
        nvml_device_index=None,
    )
    record["benchmark_sources"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(sys.modules[ReleaseBenchmark.__module__].__file__),
            Path(base.__file__),
        )
    }
    benchmark = MaterialProfile(benchmark_args, base)
    try:
        try:
            benchmark.material()
        except CapturedError:
            pass
        else:
            raise RuntimeError("Material benchmark did not reach the requested VJP")
        if source_digest(args.source)["sha256"] != record["source"]["sha256"]:
            raise RuntimeError("DPT sources changed during material profile")
        record["environment"]["nvidia_smi_after"] = command("nvidia-smi", "-q")
        save()
        print("PROFILE_METADATA", args.metadata, flush=True)
    finally:
        benchmark.monitor.close()


def main():
    args = arguments()
    if args.metadata.exists():
        raise FileExistsError("Refusing to replace " + str(args.metadata))
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.baseline.read_text())
    config = baseline["input"]
    if config["resampled"] or config["samples_per_ray"] != 1024:
        raise ValueError("Frozen native CT and 1024 clipped midpoints required")
    if config["mu_water_mm_inverse"] != 0.02059 or config["energy_kev"] != 60:
        raise ValueError("Frozen water-scaled 60 keV conversion required")
    mode = "fp32" if args.mode in ("fp32", "float32") else "fp64"
    sampling = "float32" if mode == "fp32" else "float64"
    record = {
        "schema_version": 1,
        "preparation_started_at": utc(),
        "operation": args.operation,
        "precision": mode,
        "views": args.views,
        "source": source_digest(args.source),
        "input": config,
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "warmup_calls": args.warmup,
        "capture": {
            "nvtx_range": "operation",
            "start_stop_api": "cudaProfilerStart/Stop",
            "selection": (
                "whole first matching operation range; no kernel-name or launch-count filter"
            ),
            "timing_limit": "Nsight replay/timeline only; not a clean latency measurement",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "nvidia_smi_before": command("nvidia-smi", "-q"),
            "warp_cache_path": os.environ.get("WARP_CACHE_PATH"),
            "cuda_cache_path": os.environ.get("CUDA_CACHE_PATH"),
        },
    }

    def save():
        temporary = args.metadata.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.metadata)

    save()
    sys.path.insert(0, str(args.source / "python"))
    import numpy as np
    import torch
    import warp as wp

    wp.config.lineinfo = True
    from dpt.geometry import DetectorGeometry
    from dpt.imaging_io import ImagingVolume, read_nifti
    from dpt.projection import (
        ProjectionSpec,
        prepare_projection,
        project_optical_depth,
        projection_vjp,
    )

    record["environment"].update(
        {
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "warp-lang", "nibabel")
            },
            "gpu": torch.cuda.get_device_name(0),
            "torch_cuda": torch.version.cuda,
            "warp_lineinfo_before_dpt_kernel_import": True,
        }
    )
    if args.operation == "material-vjp":
        if args.views != ["ap"]:
            raise ValueError("Material VJP profile uses the measured AP-only workload")
        material_profile(args, record, save)
        return
    path = args.ct or Path(config["file"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != config["sha256"]:
        raise ValueError("CT file checksum differs from baseline")
    ct = read_nifti(path, value_units="HU")
    if json.loads(json.dumps(asdict(ct.grid))) != config["grid"]:
        raise ValueError("Physical grid differs from baseline")
    mu = (np.maximum(1.0 + ct.values / 1000.0, 0.0) * 0.02059).astype(np.float32)
    indices = [config["view_names"].index(name) for name in args.views]
    panels = [
        DetectorGeometry(
            **{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in config["geometry"][index].items()
            }
        )
        for index in indices
    ]
    record["resolved_geometry"] = [asdict(panel) for panel in panels]
    wp.init()
    torch_stream = torch.cuda.current_stream()
    stream = wp.stream_from_torch(torch_stream)
    dtype = wp.float32 if mode == "fp32" else wp.float64
    torch_dtype = torch.float32 if mode == "fp32" else torch.float64
    region = np.zeros(panels[0].shape, dtype=np.float64)
    rows, columns = panels[0].shape
    region[rows * 5 // 16 : rows * 11 // 16, columns * 5 // 16 : columns * 11 // 16] = 1
    region /= region.sum()
    spec = ProjectionSpec(
        config["samples_per_ray"],
        precision=sampling,
        sampling_precision=sampling,
        active_volume=args.operation == "volume",
        active_pose=args.operation == "pose",
    )
    workspaces = []
    outputs = []
    if args.operation not in ("public", "iteration"):
        with wp.ScopedStream(stream):
            field = wp.array(mu.ravel(), dtype=wp.float32, device="cuda:0")
            pose = wp.array(config["pose"], dtype=wp.float64, device="cuda:0")
            seed = wp.array(region.ravel(), dtype=dtype, device="cuda:0")
            gradient = wp.empty(ct.grid.voxels, dtype=dtype, device="cuda:0")
            pose_gradient = wp.empty(6, dtype=wp.float64, device="cuda:0")
            for panel in panels:
                workspaces.append(
                    prepare_projection(ct.grid, panel, spec, device="cuda:0", stream=stream)
                )
                outputs.append(wp.empty(panel.pixels, dtype=dtype, device="cuda:0"))
        record["native_spec"] = asdict(spec)
        record["workspace_bytes"] = sum(workspace.scratch_bytes for workspace in workspaces)
        record["boundary"] = (
            "resident preallocated buffers; deferred final status check outside capture"
            if args.operation != "raw-public"
            else (
                "raw-buffer public call with validate='once'; current boundary "
                "resolves its content scans and status readbacks"
            )
        )
        for workspace, output in zip(workspaces, outputs, strict=True):
            project_optical_depth(
                field,
                pose,
                workspace=workspace,
                out_L=output,
                stream=stream,
                validate=True,
            )
    else:
        from dpt import project
        from dpt.torch import Projector

        if args.operation == "public":
            attenuation = ImagingVolume(mu, ct.grid, replace(ct.metadata, value_units="mm^-1"))
            record["boundary"] = (
                "dpt.project(ImagingVolume, geometry): device upload, "
                "module/workspace preparation, validation, forward and completed "
                "CUDA tensor return; file I/O and HU conversion prepared outside "
                "capture"
            )
        else:
            torch_field = torch.tensor(mu, device="cuda", dtype=torch.float32, requires_grad=True)
            torch_pose = torch.zeros(6, device="cuda", dtype=torch.float64, requires_grad=True)
            projector = Projector(
                grid=ct.grid,
                geometry=tuple(panels),
                precision=mode,
                samples_per_ray=config["samples_per_ray"],
            )
            torch_seed = torch.tensor(region, device="cuda", dtype=torch_dtype)
            record["boundary"] = (
                "resident ordinary Torch Projector forward for all views, "
                "first-view region objective, and backward for volume plus shared "
                "six-axis pose; gradient reset included"
            )
    record["requested_gradients"] = {
        "volume": ["volume"],
        "pose": ["pose"],
        "iteration": ["volume", "pose"],
    }.get(args.operation, [])
    record["dtypes"] = {
        "field": "float32",
        "depth": sampling,
        "volume_gradient": sampling
        if args.operation == "volume"
        else "float32"
        if args.operation == "iteration"
        else None,
        "pose": "float64",
        "pose_gradient": "float64"
        if record["requested_gradients"] and args.operation != "volume"
        else None,
    }

    def run():
        if args.operation == "public":
            return project(
                attenuation,
                tuple(panels),
                device="cuda",
                precision=mode,
                samples_per_ray=config["samples_per_ray"],
            )
        if args.operation == "iteration":
            torch_field.grad = None
            torch_pose.grad = None
            with torch.cuda.nvtx.range("forward"):
                prediction = projector(torch_field, torch_pose)
            with torch.cuda.nvtx.range("objective"):
                loss = (prediction[0] * torch_seed).sum()
            with torch.cuda.nvtx.range("backward"):
                loss.backward()
            return prediction
        if args.operation == "volume":
            with wp.ScopedStream(stream):
                gradient.zero_()
        for workspace, output in zip(workspaces, outputs, strict=True):
            if args.operation in ("forward", "raw-public"):
                project_optical_depth(
                    field,
                    pose,
                    out_L=output,
                    workspace=workspace,
                    stream=stream,
                    validate="once" if args.operation == "raw-public" else False,
                )
            elif args.operation == "volume":
                projection_vjp(
                    field,
                    pose,
                    cotangent=seed,
                    workspace=workspace,
                    out_mu=gradient,
                    stream=stream,
                    validate=False,
                    accumulate=True,
                    volume_gradient_precision=sampling,
                    check_volume_gradient=False,
                )
            else:
                projection_vjp(
                    field,
                    pose,
                    cotangent=seed,
                    workspace=workspace,
                    out_pose_tangent=pose_gradient,
                    stream=stream,
                    validate=False,
                )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    record["warmup_completed_at"] = utc()
    save()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("operation")
    try:
        with torch.cuda.nvtx.range(mode + "/" + args.operation):
            run()
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    record["profile_completed_at"] = utc()
    if args.operation == "volume":
        workspaces[0].enqueue_volume_gradient_check(
            gradient, volume_gradient_precision=sampling, stream=stream
        )
    for workspace in workspaces:
        workspace.check_status()
    if args.operation == "iteration":
        if not bool(torch.isfinite(torch_field.grad).all().item()) or not bool(
            torch.isfinite(torch_pose.grad).all().item()
        ):
            raise ArithmeticError("Nonfinite completed Torch gradient")
    if source_digest(args.source)["sha256"] != record["source"]["sha256"]:
        raise RuntimeError("DPT sources changed during profile; discard this capture")
    record["completed_checks_passed"] = True
    record["environment"]["nvidia_smi_after"] = command("nvidia-smi", "-q")
    save()
    print("PROFILE_METADATA", args.metadata, flush=True)


if __name__ == "__main__":
    main()
