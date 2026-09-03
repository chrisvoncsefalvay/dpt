"""Profile completed full-scan Projector calls after exact-workload warmup.

The saved cotangent is mandatory and is selected without renormalisation.
Status-readback instrumentation runs before capture. NVTX ranges include the
ordinary checked forward and requested backward calls. Their timings are
profiling diagnostics, not clean benchmark medians. Joint field/pose backward
is the default. Pass --gradients volume for a fixed-pose volume-only VJP.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Source checkout whose python/dpt package will run (default: this checkout)",
    )
    parser.add_argument(
        "--inputs-module",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing fullscan_inputs.py and common.py, or the input helper file",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New probe-output directory")
    parser.add_argument("--views", type=int, default=16, help="1..360 evenly spaced saved views")
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp64")
    parser.add_argument(
        "--gradients",
        choices=("volume", "both"),
        default="both",
        help="Volume-only backward or joint field and pose backward",
    )
    parser.add_argument("--profile", action="store_true", help="Use cudaProfilerStart/Stop")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cotangent", type=Path, required=True, help="Saved full 360-view NPY")
    args = parser.parse_args(argv)
    if not 1 <= args.views <= 360:
        parser.error("--views must be between 1 and 360")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    for name in ("source", "inputs_module", "data", "output", "cotangent"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def load_helper(path, name):
    """Import the exact requested helper, including when profiling an older checkout."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(args):
    helper = args.inputs_module
    if helper.is_dir():
        helper /= "fullscan_inputs.py"
    common_path = helper.with_name("common.py")
    if not (args.source / "python/dpt/__init__.py").is_file():
        raise FileNotFoundError(f"No source DPT package under {args.source}")
    if not helper.is_file() or not common_path.is_file():
        raise FileNotFoundError("--inputs-module must identify fullscan_inputs.py and common.py")
    # Direct script execution must not let benchmarks/profile.py shadow stdlib profile.
    script_directory = Path(__file__).resolve().parent
    sys.path[:] = [
        entry
        for entry in sys.path
        if not (
            Path(entry or ".").resolve() == script_directory
            and (script_directory / "profile.py").is_file()
        )
    ]
    sys.path.insert(0, str(args.source / "python"))

    import numpy as np
    import torch
    import warp as wp

    wp.config.lineinfo = True
    inputs = load_helper(helper, "_dpt_profile_inputs")
    common = load_helper(common_path, "_dpt_profile_common")
    import dpt
    from dpt._runtime import HostReadback
    from dpt.geometry import BatchedGeometry
    from dpt.torch import Projector

    if Path(dpt.__file__).resolve() != args.source / "python/dpt/__init__.py":
        raise RuntimeError("Imported DPT does not come from --source")
    args.output.mkdir(parents=True, exist_ok=False)
    scan = inputs.load_prepared(args.data)
    if (
        scan.volume.grid.shape != (256, 256, 256)
        or scan.volume.values.dtype != np.float32
        or scan.geometry.shape != (360, 512, 512)
        or scan.metadata.get("sampling", {}).get("samples_per_ray") != 1024
    ):
        raise ValueError(
            "Expected the prepared FP32 256^3 field, 360x512x512 views and 1024 samples"
        )
    indices = np.linspace(0, 360, args.views, endpoint=False, dtype=int)
    geometry = BatchedGeometry(tuple(scan.geometry.views[index] for index in indices))
    seed_saved = np.load(args.cotangent, mmap_mode="r", allow_pickle=False)
    if seed_saved.shape != scan.geometry.shape:
        raise ValueError("Saved cotangent shape differs from the full prepared acquisition")
    field = torch.tensor(np.asarray(scan.volume.values), device="cuda", requires_grad=True)
    pose = (
        torch.zeros(6, dtype=torch.float64, device="cuda", requires_grad=True)
        if args.gradients == "both"
        else None
    )
    gradient_inputs = (field, pose) if args.gradients == "both" else field
    projector = Projector(
        grid=scan.volume.grid,
        geometry=geometry,
        precision=args.precision,
        samples_per_ray=1024,
    )
    seed_dtype = torch.float64 if args.precision == "fp64" else torch.float32
    seed = torch.tensor(seed_saved[indices], device="cuda", dtype=seed_dtype)
    result = {
        "measured_at": common.utc(),
        "source": common.digest_sources(args.source),
        "input_metadata": scan.metadata,
        "views": args.views,
        "view_indices": indices.tolist(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": {
            "fullscan_inputs.py": inputs.sha256(helper),
            "common.py": inputs.sha256(common_path),
        },
        "precision": args.precision,
        "gradients": args.gradients,
        "requested_gradients": ["volume", "pose"] if args.gradients == "both" else ["volume"],
        "pose_requires_grad": pose is not None and pose.requires_grad,
        "pose_representation": (
            "shared six-component right tangent at identity"
            if args.gradients == "both"
            else "fixed identity geometry without a pose tensor"
        ),
        "sampling": {"rule": "clipped midpoint", "samples_per_ray": 1024},
        "nvtx_ranges": {"forward": "probe_forward", "backward": "probe_backward"},
        "current_torch_stream": int(torch.cuda.current_stream().cuda_stream),
        "probe_only": True,
        "cotangent": str(args.cotangent),
        "cotangent_sha256": inputs.sha256(args.cotangent),
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "warp-lang", "numpy")
        },
        "torch_cuda": torch.version.cuda,
        "gpu_before": common.command("nvidia-smi"),
        "status_counts": {},
        "samples": [],
    }
    result["target_source_files"] = {
        name: result["source"]["files"][name]
        for name in (
            "python/dpt/torch/modules.py",
            "python/dpt/torch/_projection_ops.py",
            "python/dpt/projection.py",
            "python/dpt/kernels/projection.py",
            "python/dpt/kernels/projection_fp32.py",
        )
    }
    start = time.perf_counter()
    image = projector(field, pose) if args.gradients == "both" else projector(field)
    torch.autograd.grad(image, gradient_inputs, seed)
    torch.cuda.synchronize()
    result["warmup_completed_seconds"] = time.perf_counter() - start
    original = HostReadback.read
    counts = [0]

    def counted(self):
        counts[0] += 1
        return original(self)

    HostReadback.read = counted
    try:
        image = projector(field, pose) if args.gradients == "both" else projector(field)
        result["status_counts"]["forward"] = counts[0]
        counts[0] = 0
        torch.autograd.grad(image, gradient_inputs, seed)
        result["status_counts"]["backward"] = counts[0]
    finally:
        HostReadback.read = original

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        for index in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.cuda.nvtx.range("probe_forward"):
                image = projector(field, pose) if args.gradients == "both" else projector(field)
            torch.cuda.synchronize()
            forward_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            with torch.cuda.nvtx.range("probe_backward"):
                gradients = torch.autograd.grad(image, gradient_inputs, seed)
            torch.cuda.synchronize()
            backward_ms = (time.perf_counter() - start) * 1000
            result["samples"].append({"forward_ms": forward_ms, "backward_ms": backward_ms})
            print(index, result["samples"][-1], flush=True)
    finally:
        if args.profile:
            torch.cuda.cudart().cudaProfilerStop()

    result["gpu_after"] = common.command("nvidia-smi")
    result["completed_at"] = common.utc()
    result["image_range"] = [image.min().item(), image.max().item()]
    result["gradient_norms"] = [gradient.double().norm().item() for gradient in gradients]
    result["gradient_shapes"] = [list(gradient.shape) for gradient in gradients]
    result["gradient_dtypes"] = [str(gradient.dtype) for gradient in gradients]
    result["finite_outputs"] = bool(
        torch.isfinite(image).all()
        and all(torch.isfinite(gradient).all() for gradient in gradients)
    )
    result["source_unchanged"] = common.digest_sources(args.source) == result["source"]
    if not result["finite_outputs"] or not result["source_unchanged"]:
        raise ArithmeticError("Profile output failed finite/source checks")
    (args.output / "probe.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("completed", args.output, "status", result["status_counts"], flush=True)
    return result


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
