# ruff: noqa: B023
# Each optimiser closure runs synchronously within its current precision loop.
"""Native-CT pose optimisation with an explicitly generated FP64 target.

This is a numerical self-consistency check, not measured-fluoroscopy registration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from benchmarks.common import load_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/assets/results/baseline-chest-512.json",
    )
    p.add_argument("--ct", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.source / "python"))
    base = load_runtime()
    from dpt.geometry import DetectorGeometry
    from dpt.imaging import ct_to_attenuation, read_volume
    from dpt.torch import Projector

    torch = base.torch
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    b = json.loads(a.baseline.read_text())["input"]
    ct_path = a.ct.resolve()
    if hashlib.sha256(ct_path.read_bytes()).hexdigest() != b["sha256"]:
        raise ValueError("CT checksum differs from the fixed real-volume acquisition")
    ct = read_volume(ct_path, units="HU")
    mu = ct_to_attenuation(ct, energy_kev=60)
    views = tuple(
        DetectorGeometry(**{k: tuple(v) if isinstance(v, list) else v for k, v in panel.items()})
        for panel in b["geometry"]
    )
    field = torch.tensor(mu.values, device="cuda", dtype=torch.float32)
    reference_projector = Projector(
        grid=ct.grid, geometry=views, precision="fp64", samples_per_ray=1024
    )
    with torch.no_grad():
        target = reference_projector(field).detach()
    torch.cuda.synchronize()
    initial = [1.0, -0.8, 0.6, 0.003, -0.004, 0.002]
    scales = torch.tensor([1, 1, 1, 0.01, 0.01, 0.01], device="cuda", dtype=torch.float64)
    result = {
        "schema_version": 1,
        "measurement_started_at": base.utc(),
        "source": base.digest_sources(a.source),
        "input": b,
        "loaded_ct_file": str(ct_path),
        "environment": {
            "versions": {
                name: importlib.metadata.version(name) for name in ("torch", "numpy", "warp-lang")
            },
            "gpu": torch.cuda.get_device_name(0),
            "torch_cuda": torch.version.cuda,
            "nvidia_smi_before": base.command("nvidia-smi", "-q"),
        },
        "target": (
            "Generated FP64 projection of this CT at identity pose; numerical "
            "self-consistency only, not measured-fluoroscopy registration"
        ),
        "target_sha256": hashlib.sha256(target.cpu().numpy().tobytes()).hexdigest(),
        "initial_tangent_translation_mm_rotation_rad": initial,
        "optimiser": {
            "name": "Torch LBFGS",
            "lr": 1.0,
            "max_iter_per_step": 20,
            "max_eval_per_step": 30,
            "steps": 8,
            "line_search_fn": "strong_wolfe",
            "tolerance_grad": 1e-12,
            "tolerance_change": 1e-14,
        },
        "acceptance_predeclared": {
            "maximum_final_rmse_optical_depth": 5e-4,
            "maximum_final_to_initial_mse_ratio": 0.01,
        },
        "modes": {},
    }

    def save():
        (a.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    try:
        for mode in ("fp32", "fp64"):
            projector = Projector(
                grid=ct.grid, geometry=views, precision=mode, samples_per_ray=1024
            )
            coordinates = torch.nn.Parameter(
                torch.tensor(initial, device="cuda", dtype=torch.float64) / scales
            )
            optimiser = torch.optim.LBFGS(
                [coordinates],
                lr=1.0,
                max_iter=20,
                max_eval=30,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-12,
                tolerance_change=1e-14,
            )
            trace = []

            def evaluate():
                prediction = projector(field, coordinates * scales)
                return (prediction.double() - target).square().mean()

            def closure():
                optimiser.zero_grad(set_to_none=True)
                loss = evaluate()
                loss.backward()
                trace.append(
                    {
                        "evaluation": len(trace),
                        "mse": float(loss.detach()),
                        "scaled_gradient": coordinates.grad.detach().cpu().tolist(),
                    }
                )
                return loss

            initial_loss = float(evaluate().detach())
            torch.cuda.synchronize()
            begun = time.perf_counter()
            iterations = []
            for index in range(8):
                optimiser.step(closure)
                with torch.no_grad():
                    mse = float(evaluate())
                iterations.append(
                    {
                        "step": index + 1,
                        "mse": mse,
                        "tangent": (coordinates.detach() * scales).cpu().tolist(),
                    }
                )
            torch.cuda.synchronize()
            final = iterations[-1]["mse"]
            accepted = final**0.5 <= 5e-4 and final <= initial_loss * 0.01
            result["modes"][mode] = {
                "initial_mse": initial_loss,
                "final_mse": final,
                "final_rmse": final**0.5,
                "final_to_initial_mse_ratio": final / initial_loss,
                "completed_seconds_including_objective_trace_readbacks": time.perf_counter()
                - begun,
                "iterations": iterations,
                "closure_trace": trace,
                "accepted": accepted,
            }
            save()
            if not accepted:
                raise AssertionError(
                    f"{mode} did not satisfy the predeclared numerical optimisation check"
                )
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["measurement_finished_at"] = base.utc()
        result["source_unchanged"] = (
            result["source"]["sha256"] == base.digest_sources(a.source)["sha256"]
        )
        save()


if __name__ == "__main__":
    main()
