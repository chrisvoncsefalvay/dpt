#!/usr/bin/env python3
"""Completed public PyTorch projection/VJP timings for the frozen full chest.

The selected source supplies DPT; this driver and its unchanged common timer
are shared by reference and comparison runs. Execute CUDA jobs under the shared
GPU lock. Outputs, checks and raw timings are written incrementally to a new
directory. Loading, upload, correctness checks and downloads are not timed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from benchmarks.common import Measurements, NvmlMonitor, command, digest_sources, utc


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def array_error(actual, reference, np):
    """FP64 reductions in bounded host chunks, including exact equality."""
    if actual.shape != reference.shape:
        raise ValueError("reference shape differs from the measured output")
    a, b = actual.reshape(-1), reference.reshape(-1)
    maximum, square_error, square_reference = 0.0, 0.0, 0.0
    equal = actual.dtype == reference.dtype
    for start in range(0, a.size, 1024 * 1024):
        a_chunk, b_chunk = a[start : start + 1024 * 1024], b[start : start + 1024 * 1024]
        x = np.asarray(a_chunk, dtype=np.float64)
        y = np.asarray(b_chunk, dtype=np.float64)
        delta = x - y
        if equal:
            equal = bool(np.array_equal(a_chunk.view(np.uint8), b_chunk.view(np.uint8)))
        maximum = max(maximum, float(np.max(np.abs(delta))))
        square_error += float(np.dot(delta, delta))
        square_reference += float(np.dot(y, y))
    return {
        "max_abs": maximum,
        "rmse": float(np.sqrt(square_error / a.size)),
        "relative_l2": float(np.sqrt(square_error / max(square_reference, 1e-300))),
        "bitwise_equal": equal,
    }


def inner_product(a, b, np):
    result = 0.0
    a, b = a.reshape(-1), b.reshape(-1)
    if a.shape != b.shape:
        raise ValueError("inner-product operands have different shapes")
    for start in range(0, a.size, 1024 * 1024):
        result += float(
            np.dot(
                np.asarray(a[start : start + 1024 * 1024], dtype=np.float64),
                np.asarray(b[start : start + 1024 * 1024], dtype=np.float64),
            )
        )
    return result


class FullScan(Measurements):
    def __init__(self, args):
        self.args = args
        self.results = {
            "schema_version": 1,
            "status": "preparing",
            "started_at": utc(),
            "measurements": [],
            "checks": {},
            "preparation_seconds": {},
        }
        self.monitor = None
        self.outputs = {}
        self.save()
        self.contract = json.loads(args.contract.read_text())
        self.results["contract"] = {
            "sha256": file_sha256(args.contract),
            "settings": self.contract,
        }
        self.source_before = digest_sources(args.source)
        self.results["source"] = self.source_before
        self.results["source_revision"] = command(
            "git", "-C", str(args.source), "rev-parse", "HEAD"
        )
        self.results["harness"] = {
            name: file_sha256(Path(__file__).parent / name)
            for name in ("fullscan.py", "common.py", "fullscan_inputs.py")
        }
        sys.path.insert(0, str(args.source / "python"))
        import numpy as np
        import torch
        import warp as wp

        wp.config.lineinfo = True
        import dpt
        from benchmarks.fullscan_inputs import load_prepared
        from dpt.geometry import BatchedGeometry
        from dpt.torch import Projector

        if not Path(dpt.__file__).resolve().is_relative_to(args.source.resolve() / "python"):
            raise RuntimeError("DPT was imported from a different source checkout")
        self.np, self.torch, self.wp, self.Projector = np, torch, wp, Projector
        self.base = SimpleNamespace(torch=torch, wp=wp, utc=utc)
        start = time.perf_counter()
        self.scan = load_prepared(args.data)
        input_hash = file_sha256(args.data / "metadata.json")
        if input_hash != self.contract["input_metadata_sha256"]:
            raise ValueError("input metadata differs from the frozen contract")
        if self.results["harness"]["fullscan_inputs.py"] != self.contract["input_helper_sha256"]:
            raise ValueError("frozen input helper changed")
        self.geometry = BatchedGeometry(self.scan.geometry.views[: args.views])
        self.seed_metadata = json.loads(args.cotangent.with_suffix(".json").read_text())
        if self.seed_metadata["contract_sha256"] != self.results["contract"]["sha256"]:
            raise ValueError("cotangent was prepared against another contract")
        if file_sha256(args.cotangent) != self.seed_metadata["sha256"]:
            raise ValueError("cotangent file SHA256 mismatch")
        self.seed_host = np.load(args.cotangent, mmap_mode="r", allow_pickle=False)[: args.views]
        if self.seed_host.dtype != np.float32 or self.seed_host.shape != self.geometry.shape:
            raise ValueError("cotangent shape/dtype differs from frozen settings")
        self.results["preparation_seconds"]["input_loading_and_hash_checks"] = (
            time.perf_counter() - start
        )
        self.results["input"] = {
            "metadata_sha256": input_hash,
            "metadata": self.scan.metadata,
            "selected_view_indices": list(range(args.views)),
            "selected_geometry": self.scan.metadata["geometry"][: args.views],
            "cotangent": self.seed_metadata,
            "cotangent_subset_renormalised": False,
        }
        self.results["precision"] = args.precision
        self.results["requested_operations"] = args.operations
        start = time.perf_counter()
        wp.init()
        torch.cuda.init()
        self.stream = torch.cuda.current_stream()
        self.field = torch.tensor(self.scan.volume.values, dtype=torch.float32, device="cuda:0")
        output_dtype = torch.float64 if args.precision == "fp64" else torch.float32
        self.seed = torch.tensor(self.seed_host, dtype=output_dtype, device="cuda:0")
        self.stream.synchronize()
        self.results["preparation_seconds"]["cuda_initialisation_and_upload"] = (
            time.perf_counter() - start
        )
        self.results["environment"] = {
            "recorded_at": utc(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "warp-lang", "nibabel", "nvidia-ml-py")
            },
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "nvidia_smi": command("nvidia-smi", "-q"),
            "nsys": command("nsys", "--version"),
            "ncu": command("ncu", "--version"),
        }
        self.monitor = NvmlMonitor(self.base, args)
        self.results["before_timing_gpu_sample"] = self.monitor.sample()
        self.results["checks"]["pose_finite_difference"] = {
            "status": "separate guard required",
            "thresholds": self.contract["quality"]["pose_fd"],
        }
        self.results["checks"]["independent_selected_ray_reference"] = {
            "status": "not run by this timer; separate CPU/adapter guard required",
        }
        self.results["status"] = "ready"
        self.save()

    def operation(self, name):
        torch = self.torch
        field = self.field.detach().requires_grad_(name in ("volume_vjp", "iteration"))
        pose = torch.zeros(
            6,
            dtype=torch.float64,
            device=field.device,
            requires_grad=name in ("pose_vjp", "iteration"),
        )
        module = self.Projector(
            grid=self.scan.volume.grid,
            geometry=self.geometry,
            precision=self.args.precision,
            samples_per_ray=1024,
        )
        state = {}
        if name in ("volume_vjp", "pose_vjp"):
            # Separate graphs: the unrequested input does not require grad,
            # so the custom autograd Function cannot compute both VJPs.
            state["image"] = module(field, pose if name == "pose_vjp" else None)

        def call():
            if name == "forward":
                with torch.no_grad():
                    state["image"] = module(field)
            elif name == "volume_vjp":
                state["volume_vjp"] = torch.autograd.grad(
                    state["image"], field, self.seed, retain_graph=True
                )[0]
            elif name == "pose_vjp":
                state["pose_vjp"] = torch.autograd.grad(
                    state["image"], pose, self.seed, retain_graph=True
                )[0]
            else:
                image = module(field, pose)
                state["volume_vjp"], state["pose_vjp"] = torch.autograd.grad(
                    image, (field, pose), self.seed
                )
                state["image"] = image.detach()

        return call, state

    def save_array(self, name, values):
        from benchmarks.fullscan_inputs import array_record

        path = self.args.output / (name + ".npy")
        if path.exists():
            previous = self.np.load(path, mmap_mode="r", allow_pickle=False)
            return array_error(values, previous, self.np)
        self.np.save(path, values, allow_pickle=False)
        units = {
            "images": "dimensionless optical depth",
            "volume_vjp": "objective per mm^-1",
            "pose_vjp": "objective per mm for translation / per radian for rotation",
        }
        self.outputs[name] = array_record(path, values, units[name])
        write_json(
            self.args.output / "outputs.json",
            {
                "source_sha256": self.source_before["sha256"],
                "input_metadata_sha256": self.results["input"]["metadata_sha256"],
                "cotangent_sha256": self.seed_metadata["sha256"],
                "precision": self.args.precision,
                "views": self.args.views,
                "arrays": self.outputs,
            },
        )
        return None

    def check_reference(self, name, values):
        if self.args.reference is None:
            return {"status": "reference run; no prior output supplied"}
        outputs = json.loads((self.args.reference / "outputs.json").read_text())
        if (
            outputs["input_metadata_sha256"] != self.results["input"]["metadata_sha256"]
            or outputs["cotangent_sha256"] != self.seed_metadata["sha256"]
            or outputs["views"] != self.args.views
        ):
            raise ValueError("reference inputs/views/cotangent differ")
        if name not in outputs["arrays"]:
            return {"status": "not available in reference"}
        record = outputs["arrays"][name]
        path = self.args.reference / record["file"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("reference array hash mismatch")
        reference = self.np.load(path, mmap_mode="r", allow_pickle=False)
        error = array_error(values, reference, self.np)
        if name == "images":
            if outputs["precision"] == self.args.precision:
                passed = error["bitwise_equal"]
            else:
                bounds = self.contract["quality"]["image_fp32_vs_fp64"]
                passed = all(
                    error[key] <= bounds[key] for key in ("max_abs", "rmse", "relative_l2")
                )
        elif name == "volume_vjp":
            passed = (
                error["relative_l2"]
                <= self.contract["quality"]["field_vjp_vs_baseline"]["relative_l2_max"]
            )
        else:
            # Finite-difference accuracy belongs to the separate frozen guard.
            return {"status": "diagnostic comparison; FD gate separate", **error}
        return {"passed": passed, **error}

    def check_outputs(self, name, state):
        checks, host = {}, {}
        for key, tensor in state.items():
            output_name = "images" if key == "image" else key
            value = tensor.detach().cpu().numpy()
            if not self.np.isfinite(value).all():
                raise ArithmeticError(f"nonfinite {output_name}")
            host[output_name] = value
            repeated = self.save_array(output_name, value)
            if repeated is not None:
                if output_name == "images":
                    repeated["passed"] = repeated["bitwise_equal"]
                elif output_name == "volume_vjp":
                    repeated["passed"] = (
                        repeated["relative_l2"]
                        <= self.contract["quality"]["field_vjp_vs_baseline"]["relative_l2_max"]
                    )
                else:
                    repeated["status"] = "diagnostic comparison; FD gate separate"
                checks[output_name + "_vs_prior_operation"] = repeated
            checks[output_name + "_vs_reference"] = self.check_reference(output_name, value)
        if "volume_vjp" in host:
            lhs = inner_product(host["images"], self.seed_host, self.np)
            rhs = inner_product(self.scan.volume.values, host["volume_vjp"], self.np)
            discrepancy = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
            checks["volume_adjoint"] = {
                "image_dot_seed_fp64": lhs,
                "field_dot_vjp_fp64": rhs,
                "relative_discrepancy": discrepancy,
                "passed": discrepancy
                <= self.contract["quality"]["volume_adjoint"]["relative_discrepancy_max"],
                "reduction_dtype": "float64",
                "gradient_storage_dtype": "float32",
            }
        checks["passed"] = all(value.get("passed", True) for value in checks.values())
        self.results["checks"][name] = checks
        self.save()
        if not checks["passed"]:
            raise ArithmeticError(f"{name} correctness gate failed; timing not collected")

    def run(self):
        for name in self.args.operations:
            self.results["stage"] = name + " preparation and correctness"
            self.save()
            print(self.results["stage"], utc(), flush=True)
            start = time.perf_counter()
            call, state = self.operation(name)
            call()
            self.stream.synchronize()
            self.results["preparation_seconds"][name + "_graph_and_first_call"] = (
                time.perf_counter() - start
            )
            self.check_outputs(name, state)
            self.results["stage"] = name + " timing"
            self.save()
            self.measure(
                name,
                call,
                {
                    "precision": self.args.precision,
                    "field_dtype": "float32",
                    "pose_dtype": "float64",
                    "pose": [0.0] * 6,
                    "views": self.args.views,
                    "detector_shape": [512, 512],
                    "samples_per_ray": 1024,
                    "input_residency": "CUDA",
                    "operation": self.contract["operations"][name],
                    "wall_boundary": self.contract["operations"]["wall_boundary"],
                    "event_boundary": self.contract["operations"]["event_boundary"],
                    "gradients_accumulate": False,
                    "initial_correctness_call_before_measurement_warmup": True,
                },
            )
            del call, state
            gc.collect()
        self.results["source_after"] = digest_sources(self.args.source)
        self.results["source_unchanged"] = self.results["source_after"] == self.source_before
        if not self.results["source_unchanged"]:
            raise RuntimeError("timed source changed during this run")
        self.results.update(status="complete", stage="complete", completed_at=utc())
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp64", "fp32"), required=True)
    parser.add_argument("--views", type=int, choices=(1, 16, 360), default=360)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=("forward", "volume_vjp", "pose_vjp", "iteration"),
        default=["forward", "volume_vjp", "pose_vjp", "iteration"],
    )
    parser.add_argument(
        "--contract",
        type=Path,
        required=True,
        help="Settings manifest written by benchmarks.prepare_fullscan",
    )
    parser.add_argument("--cotangent", type=Path, required=True)
    parser.add_argument(
        "--reference", type=Path, help="Completed baseline directory with outputs.json"
    )
    parser.add_argument("--nvml-interval-seconds", type=float, default=0.02)
    parser.add_argument("--nvml-device-index", type=int)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error("rounds and repeats must be positive")
    args.source, args.data = args.source.resolve(), args.data.resolve()
    args.long_threshold_seconds, args.long_repeats = 0.0, args.repeats
    args.output.mkdir(parents=True, exist_ok=False)
    benchmark = FullScan.__new__(FullScan)
    try:
        benchmark.__init__(args)
        benchmark.run()
    except Exception as error:
        if hasattr(benchmark, "results"):
            benchmark.results.update(
                status="failed",
                failed_at=utc(),
                error=repr(error),
                traceback=traceback.format_exc(),
            )
            benchmark.save()
        raise
    finally:
        if getattr(benchmark, "monitor", None) is not None:
            benchmark.monitor.close()


if __name__ == "__main__":
    main()
