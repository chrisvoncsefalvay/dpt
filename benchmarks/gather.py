"""Measure native CT FP64 scatter and prepared deterministic gather on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

from benchmarks.common import NvmlMonitor, load_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/assets/results/baseline-chest-512.json",
    )
    p.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--ct", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.source / "python"))
    base = load_runtime()
    np, torch, wp = base.np, base.torch, base.wp
    from benchmarks.baseline_ct import measure
    from dpt.geometry import DetectorGeometry, RigidTransform
    from dpt.imaging import ct_to_attenuation, read_volume
    from dpt.projection import (
        ProjectionSpec,
        prepare_projection,
        prepare_projection_adjoint,
        projection_adjoint_gather,
        projection_vjp,
    )

    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    b = json.loads(a.baseline.read_text())
    config = dict(b["input"])
    if hashlib.sha256(a.ct.read_bytes()).hexdigest() != config["sha256"]:
        raise ValueError("CT checksum differs from the frozen geometry baseline")
    config["file"] = str(a.ct.resolve())
    ct = read_volume(a.ct, units="HU")
    mu = ct_to_attenuation(ct, energy_kev=60)
    panel = DetectorGeometry(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in config["geometry"][0].items()}
    )
    wp.init()
    ts = torch.cuda.current_stream()
    stream = wp.stream_from_torch(ts)
    field = wp.array(mu.values.ravel(), dtype=wp.float32, device="cuda")
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cuda")
    workspace = prepare_projection(
        ct.grid,
        panel,
        ProjectionSpec(1024, precision="float64", active_volume=True, active_pose=False),
        device="cuda",
        stream=stream,
    )
    scatter = wp.empty(ct.grid.voxels, dtype=wp.float64, device="cuda")
    gather = wp.empty_like(scatter)
    monitor = NvmlMonitor(
        base, argparse.Namespace(nvml_interval_seconds=0.05, nvml_device_index=None)
    )
    result = {
        "schema_version": 1,
        "measurement_started_at": base.utc(),
        "source": base.digest_sources(a.source),
        "input": config,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "warp-lang", "nibabel")
            },
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "nvidia_smi_before": base.command("nvidia-smi", "-q"),
        },
        "nvml_before": monitor.sample(),
        "measurements": [],
        "checks": {},
        "precision": "FP64 strict midpoint; FP32 field storage; FP64 gradient destination",
        "fp32_gather": "unsupported: existing deterministic gather uses strict FP64 weights",
    }

    def save():
        (a.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    try:
        ts.synchronize()
        begun = time.perf_counter()
        plan = prepare_projection_adjoint(
            pose,
            workspace=workspace,
            max_work=1_000_000_000,
            max_prepare_voxels=50_000_000,
            max_rays_per_voxel=256,
            max_bytes=256 * 1024 * 1024,
            stream=stream,
        )
        ts.synchronize()
        result["preparation"] = {
            "completed_seconds_including_first_use_compilation": time.perf_counter() - begun,
            "candidate_rays": plan.candidate_rays,
            "quadrature_nodes": plan.quadrature_nodes,
            "fallback_voxels": plan.fallback_voxels,
            "work_bound": plan.work_bound,
            "preparation_work_bound": plan.preparation_work_bound,
            "device_bytes": plan.device_bytes,
        }
        save()
        for pattern in ("detector_region", "full_detector_mean"):
            weights = np.ones(panel.shape, dtype=np.float64)
            if pattern == "detector_region":
                weights.fill(0)
                rows, cols = panel.shape
                weights[rows * 5 // 16 : rows * 11 // 16, cols * 5 // 16 : cols * 11 // 16] = 1
            weights /= weights.sum()
            seed = wp.array(weights.ravel(), dtype=wp.float64, device="cuda")

            def run_scatter(seed=seed):
                with wp.ScopedStream(stream):
                    scatter.zero_()
                projection_vjp(
                    field,
                    pose,
                    cotangent=seed,
                    workspace=workspace,
                    out_mu=scatter,
                    validate=False,
                    stream=stream,
                    accumulate=True,
                    volume_gradient_precision="float64",
                    check_volume_gradient=False,
                )

            def run_gather(seed=seed):
                projection_adjoint_gather(
                    plan,
                    cotangent=seed,
                    workspace=workspace,
                    out_mu=gather,
                    validate=False,
                    stream=stream,
                    max_work=1_000_000_000,
                    volume_gradient_precision="float64",
                )

            for name, call in (("scatter", run_scatter), ("gather", run_gather)):
                begun = time.perf_counter()
                call()
                ts.synchronize()
                warm_seconds = time.perf_counter() - begun
                workspace.check_status()
                repeats = 100 if name == "scatter" else (3 if warm_seconds > 1 else 10)
                monitor.start()
                measured = measure(call, ts, 3, repeats, f"{pattern}/{name}")
                measured.update(
                    pattern=pattern,
                    method=name,
                    memory=monitor.stop(),
                    first_warm_call_seconds=warm_seconds,
                    boundary=(
                        "resident operation; scatter includes gradient zero; "
                        "gather overwrites each voxel"
                    ),
                )
                result["measurements"].append(measured)
                save()
            x, y = scatter.numpy(), gather.numpy()
            difference = x - y
            result["checks"][pattern] = {
                "max_abs": float(np.max(np.abs(difference))),
                "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(y), 1e-30)),
                "accepted_rtol": 2e-10,
                "accepted_atol": 1e-12,
            }
            np.testing.assert_allclose(x, y, rtol=2e-10, atol=1e-12)
            save()
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise
    finally:
        result["measurement_finished_at"] = base.utc()
        result["nvml_after"] = monitor.sample()
        monitor.close()
        save()


if __name__ == "__main__":
    main()
