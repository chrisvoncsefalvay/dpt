#!/usr/bin/env python3
"""Measure FP32 sampling quality versus time on the identical complete chest scan.

Run as ``python -m benchmarks.fullscan_quality`` under the shared GPU lock.
Changing 128/256/512/1024 midpoint samples is a sampling experiment, not an
arithmetic-only optimisation. Reference images use the frozen 1024-sample FP64
operator. Only resident public forward calls are timed; no derivatives are run.
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
from benchmarks.fullscan import array_error, file_sha256

SAMPLES = (128, 256, 512, 1024)
HELPERS = ("fullscan_quality.py", "fullscan.py", "fullscan_inputs.py", "common.py")


def harness_hashes():
    return {name: file_sha256(Path(__file__).with_name(name)) for name in HELPERS}


def reference_record(directory, contract, contract_hash, metadata):
    result_path, output_path = directory / "results.json", directory / "outputs.json"
    result, outputs = json.loads(result_path.read_text()), json.loads(output_path.read_text())
    if not (
        result.get("status") == "complete"
        and result.get("source_unchanged") is True
        and not result.get("error")
        and not result.get("failure")
        and result["precision"] == outputs["precision"] == "fp64"
        and result["source"] == result["source_after"]
        and result["source"]["sha256"] == outputs["source_sha256"]
        and result["contract"] == {"sha256": contract_hash, "settings": contract}
        and result["input"]["metadata"] == metadata
        and result["input"]["metadata_sha256"]
        == outputs["input_metadata_sha256"]
        == contract["input_metadata_sha256"]
        and result["input"]["selected_view_indices"] == list(range(360))
        and result["input"]["selected_geometry"] == metadata["geometry"]
        and outputs["views"] == 360
    ):
        raise ValueError("Reference must be a completed frozen 360-view FP64 full-scan result")
    image = outputs["arrays"]["images"]
    if (
        image["shape"] != [360, 512, 512]
        or image["dtype"] != "float64"
        or image["units"] != "dimensionless optical depth"
    ):
        raise ValueError("Reference image shape, dtype or quantity differs")
    path = directory / image["file"]
    if file_sha256(path) != image["sha256"]:
        raise ValueError("FP64 reference image checksum differs")
    return path, {
        "results_sha256": file_sha256(result_path),
        "outputs_sha256": file_sha256(output_path),
        "source_sha256": result["source"]["sha256"],
        "completed_at": result["completed_at"],
        "samples_per_ray": 1024,
        "images": image,
    }


class FullscanQuality(Measurements):
    def __init__(self, args):
        self.args, self.monitor = args, None
        self.results = {
            "schema_version": 1,
            "status": "preparing",
            "started_at": utc(),
            "purpose": "measured forward sampling-quality sweep; not arithmetic-only optimisation",
            "precision": "fp32",
            "requested_samples_per_ray": args.samples,
            "measurements": [],
            "checks": {},
            "preparation_seconds": {},
            "saved_images": {},
        }
        self.save()
        self.contract = json.loads(args.contract.read_text())
        self.results["contract"] = {
            "sha256": file_sha256(args.contract),
            "settings": self.contract,
            "scope": (
                "fixed input/acquisition and reference; candidate sample count varies explicitly"
            ),
        }
        self.results["source"] = digest_sources(args.source)
        self.results["harness"] = harness_hashes()
        sys.path.insert(0, str(args.source / "python"))
        import numpy as np
        import torch
        import warp as wp

        wp.config.lineinfo = True
        import dpt
        from benchmarks.fullscan_inputs import load_prepared
        from dpt.torch import Projector

        if not Path(dpt.__file__).resolve().is_relative_to(args.source / "python/dpt"):
            raise RuntimeError("DPT imported outside the selected source")
        self.np, self.torch, self.wp, self.Projector = np, torch, wp, Projector
        self.base = SimpleNamespace(torch=torch, wp=wp, utc=utc)
        started = time.perf_counter()
        metadata_hash = file_sha256(args.data / "metadata.json")
        if (
            metadata_hash != self.contract["input_metadata_sha256"]
            or self.results["harness"]["fullscan_inputs.py"] != self.contract["input_helper_sha256"]
        ):
            raise ValueError("Prepared data/helper differs from the frozen full scan")
        self.scan = load_prepared(args.data)
        if (
            self.scan.volume.values.shape != (256, 256, 256)
            or self.scan.volume.values.dtype != np.float32
            or self.scan.geometry.shape != (360, 512, 512)
            or self.scan.metadata["arrays"]["attenuation"]["sha256"]
            != self.contract["field"]["file_sha256"]
            or self.contract["sampling"]["samples_per_ray"] != 1024
        ):
            raise ValueError("The sampling sweep requires the complete frozen field/acquisition")
        self.results["input"] = {
            "metadata_sha256": metadata_hash,
            "metadata": self.scan.metadata,
            "selected_view_indices": list(range(360)),
            "selected_geometry": self.scan.metadata["geometry"],
            "field_dtype": "float32",
            "field_units": "mm^-1",
        }
        self.reference_path, self.results["reference"] = reference_record(
            args.reference, self.contract, self.results["contract"]["sha256"], self.scan.metadata
        )
        self.reference = np.load(self.reference_path, mmap_mode="r", allow_pickle=False)
        if (
            self.reference.shape != (360, 512, 512)
            or self.reference.dtype != np.float64
            or hashlib.sha256(memoryview(self.reference).cast("B")).hexdigest()
            != self.results["reference"]["images"]["raw_sha256"]
            or any(not np.isfinite(plane).all() for plane in self.reference)
        ):
            raise ValueError("Reference values differ or are nonfinite")
        self.results["preparation_seconds"]["load_and_verify_inputs_reference"] = (
            time.perf_counter() - started
        )
        started = time.perf_counter()
        wp.init()
        torch.cuda.init()
        self.stream = torch.cuda.current_stream()
        self.field = torch.tensor(self.scan.volume.values, dtype=torch.float32, device="cuda:0")
        self.stream.synchronize()
        self.results["preparation_seconds"]["cuda_initialisation_and_upload"] = (
            time.perf_counter() - started
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
        }
        self.monitor = NvmlMonitor(self.base, args)
        self.results["before_timing_gpu_sample"] = self.monitor.sample()
        self.results["status"] = "ready"
        self.save()

    def check_image(self, image, samples):
        torch, np = self.torch, self.np
        if image.shape != (360, 512, 512) or image.dtype != torch.float32:
            raise ArithmeticError("Forward output shape/precision differs")
        actual = image.detach().cpu().numpy()
        if not np.isfinite(actual).all():
            raise ArithmeticError("Forward output contains nonfinite values")
        errors = array_error(actual, self.reference, np)
        gate = None
        if samples == 1024:
            limits = self.contract["quality"]["image_fp32_vs_fp64"]
            gate = {
                "scope": "same 1024-sample discretisation only",
                "thresholds": limits,
                "passed": all(
                    errors[key] <= limits[key] for key in ("max_abs", "rmse", "relative_l2")
                ),
            }
        return actual, {
            "measured_at": utc(),
            "error_vs_fp64_1024": errors,
            "units": "dimensionless optical depth",
            "raw_sha256": hashlib.sha256(memoryview(actual).cast("B")).hexdigest(),
            "arithmetic_accuracy_gate": gate,
            "quality_policy": (
                "128/256/512-sample errors are measured, "
                "not assigned the 1024-sample arithmetic tolerance"
            ),
        }

    def run(self):
        for samples in self.args.samples:
            label = f"fp32/{samples}-samples/forward"
            self.results["stage"] = label + " preparation and image check"
            self.save()
            print(self.results["stage"], utc(), flush=True)
            started = time.perf_counter()
            module = self.Projector(
                grid=self.scan.volume.grid,
                geometry=self.scan.geometry,
                precision="fp32",
                samples_per_ray=samples,
            )
            state = {}

            def forward(state=state, module=module):
                with self.torch.no_grad():
                    state["image"] = module(self.field)

            forward()
            self.stream.synchronize()
            self.results["preparation_seconds"][str(samples) + "_module_and_first_call"] = (
                time.perf_counter() - started
            )
            actual, check = self.check_image(state["image"], samples)
            self.results["checks"][label] = check
            self.save()
            if (
                check["arithmetic_accuracy_gate"]
                and not check["arithmetic_accuracy_gate"]["passed"]
            ):
                raise ArithmeticError(
                    "The 1024-sample FP32 result fails its unchanged accuracy gate"
                )
            if self.args.save_last_three:
                from benchmarks.fullscan_inputs import array_record

                path = self.args.output / f"fp32-{samples}-last-three-views.npy"
                selected = self.np.ascontiguousarray(actual[-3:])
                self.np.save(path, selected, allow_pickle=False)
                self.results["saved_images"][label] = {
                    **array_record(path, selected, "dimensionless optical depth"),
                    "view_indices": [357, 358, 359],
                }
            del actual
            self.results["stage"] = label + " timing"
            row = self.measure(
                label,
                forward,
                {
                    "operation": "forward",
                    "precision": "fp32",
                    "field_dtype": "float32",
                    "output_dtype": "float32",
                    "requested_gradients": [],
                    "pose": "identity fixed geometry",
                    "views": 360,
                    "detector_shape": [512, 512],
                    "volume_shape_zyx": [256, 256, 256],
                    "samples_per_ray": samples,
                    "sampling": "uniform midpoints on each ray clipped to voxel-face support",
                    "interpolation": self.contract["sampling"]["interpolation"],
                    "output": "dimensionless optical depth",
                    "input_residency": "CUDA",
                    "wall_boundary": self.contract["operations"]["wall_boundary"],
                    "event_boundary": self.contract["operations"]["event_boundary"],
                    "allocation": "public output allocation included; exact prepared module reused",
                    "comparison": "changed sample count; not an arithmetic-only optimisation",
                    "error_vs_fp64_1024": check["error_vs_fp64_1024"],
                },
            )
            row["first_use_limit"] = (
                "already warmed by the preceding image check; not startup. "
                "The module construction and actual first call are recorded in "
                f"preparation_seconds['{samples}_module_and_first_call']."
            )
            final, final_check = self.check_image(state["image"], samples)
            unchanged = final_check["raw_sha256"] == check["raw_sha256"]
            check["after_timing"] = {
                "raw_sha256": final_check["raw_sha256"],
                "bitwise_unchanged": unchanged,
            }
            row["error_vs_fp64_1024"] = check["error_vs_fp64_1024"]
            self.save()
            if not unchanged:
                raise ArithmeticError("Repeated warmed forward changed its output")
            del final, state, forward, module
            gc.collect()
        self.results["source_after"] = digest_sources(self.args.source)
        self.results["source_unchanged"] = self.results["source_after"] == self.results["source"]
        self.results["harness_unchanged"] = harness_hashes() == self.results["harness"]
        if not self.results["source_unchanged"] or not self.results["harness_unchanged"]:
            raise RuntimeError("Package source or measurement helpers changed during the sweep")
        if file_sha256(self.reference_path) != self.results["reference"]["images"]["sha256"]:
            raise RuntimeError("FP64 reference changed during the sweep")
        self.results["after_timing_gpu_sample"] = self.monitor.sample()
        self.results.update(status="complete", stage="complete", completed_at=utc())
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "reference", "contract", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--samples", type=int, choices=SAMPLES, nargs="+", default=list(SAMPLES))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--save-last-three",
        action="store_true",
        help="Save only views 357,358,359 for each sample count",
    )
    parser.add_argument("--nvml-interval-seconds", type=float, default=0.02)
    parser.add_argument("--nvml-device-index", type=int)
    args = parser.parse_args()
    if min(args.rounds, args.repeats, args.nvml_interval_seconds) <= 0 or len(
        set(args.samples)
    ) != len(args.samples):
        parser.error("Positive repeat/round/monitor counts and distinct sample counts required")
    for name in ("source", "data", "reference", "contract", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.long_threshold_seconds, args.long_repeats = 0.0, args.repeats
    args.output.mkdir(parents=True, exist_ok=False)
    benchmark = FullscanQuality.__new__(FullscanQuality)
    try:
        benchmark.__init__(args)
        benchmark.run()
    except BaseException as error:
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
