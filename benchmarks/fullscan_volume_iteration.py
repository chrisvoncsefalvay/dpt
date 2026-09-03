#!/usr/bin/env python3
"""Measure completed resident forward and volume-only backward on the full scan.

Run under the shared GPU lock. The 256-sample operation changes discretisation.
Only a corresponding same-sample reference can gate its paired gradient error.
An optional 1024-sample comparison is reported separately without a pass flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path


def require_reference_metadata(result, outputs, current_input, precision, samples):
    """Reject incomplete, differently sampled or differently seeded references."""
    source = result.get("source")
    if not (
        isinstance(source, dict)
        and isinstance(source.get("sha256"), str)
        and source["sha256"]
        and isinstance(source.get("files"), dict)
        and source["files"]
    ):
        raise ValueError("Reference must contain a source hash and source file map")
    if not (
        result.get("status") == "complete"
        and result.get("source_unchanged") is True
        and result.get("source") == result.get("source_after")
        and result.get("source", {}).get("sha256") == outputs.get("source_sha256")
        and result.get("completed_at")
        and not result.get("error")
        and not result.get("failure")
        and result.get("precision") == outputs.get("precision") == precision
        and outputs.get("views") == 360
    ):
        raise ValueError("Reference must be a completed, unchanged same-mode full scan")
    reference_input = result["input"]
    for key in (
        "metadata_sha256",
        "metadata",
        "selected_view_indices",
        "selected_geometry",
        "cotangent_subset_renormalised",
    ):
        if reference_input.get(key) != current_input.get(key):
            raise ValueError(f"Reference input differs: {key}")
    if not (
        reference_input["metadata_sha256"] == outputs["input_metadata_sha256"]
        and reference_input["cotangent"]["sha256"]
        == current_input["cotangent"]["sha256"]
        == outputs["cotangent_sha256"]
        and reference_input["cotangent"]["raw_sha256"] == current_input["cotangent"]["raw_sha256"]
        and reference_input["selected_view_indices"] == list(range(360))
    ):
        raise ValueError("Reference output manifest, views or cotangent differs")
    measured_samples = {row["configuration"]["samples_per_ray"] for row in result["measurements"]}
    if measured_samples != {samples} or outputs.get("samples_per_ray", samples) != samples:
        raise ValueError("Reference must use the corresponding sample count")

    def inspect_checks(value):
        if not isinstance(value, dict):
            return False
        if value.get("passed") is False:
            raise ValueError("Reference contains a failed numerical check")
        found = value.get("volume_adjoint", {}).get("passed") is True
        return any([found, *(inspect_checks(child) for child in value.values())])

    if not inspect_checks(result.get("checks", {})):
        raise ValueError("Reference has no passed volume-adjoint check")


def make_probe_class(helper_source):
    sys.path.insert(0, str(helper_source))
    from benchmarks.common import digest_sources, utc
    from benchmarks.fullscan import FullScan, array_error, file_sha256, write_json

    class VolumeIteration(FullScan):
        def __init__(self, args):
            super().__init__(args)
            self.results.update(
                schema_version=1,
                experiment="resident_volume_only_iteration",
                samples_per_ray=args.samples,
                arguments={
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            )
            self.results["harness"]["fullscan_volume_iteration.py"] = file_sha256(Path(__file__))
            self.results["harness_paths"] = {
                name: str(
                    Path(__file__)
                    if name == "fullscan_volume_iteration.py"
                    else helper_source / "benchmarks" / name
                )
                for name in self.results["harness"]
            }
            self.results["checks"] = {}
            self.results["sampling"] = {
                **self.contract["sampling"],
                "samples_per_ray": args.samples,
                "scope": "resolved operation sampling, overriding the 1024-sample headline",
                "changed_from_frozen_headline": args.samples != 1024,
            }
            if not (
                self.scan.volume.values.shape == (256, 256, 256)
                and self.geometry.shape == (360, 512, 512)
                and self.scan.metadata["arrays"]["attenuation"]["sha256"]
                == self.contract["field"]["file_sha256"]
                and hashlib.sha256(memoryview(self.seed_host).cast("B")).hexdigest()
                == self.seed_metadata["raw_sha256"]
            ):
                raise ValueError("Prepared volume, complete acquisition or saved seed differs")
            self.references = {}
            for label, directory, samples in (
                ("same_sample", args.reference, args.samples),
                ("sampling_difference", args.difference_reference_1024, 1024),
            ):
                if directory is not None:
                    self.references[label] = self.load_reference(directory, samples)
            self.results["references"] = {
                key: value["record"] for key, value in self.references.items()
            }
            self.save()

        def load_reference(self, directory, samples):
            result_path, output_path = directory / "results.json", directory / "outputs.json"
            result = json.loads(result_path.read_text())
            outputs = json.loads(output_path.read_text())
            require_reference_metadata(
                result, outputs, self.results["input"], self.args.precision, samples
            )
            arrays = {}
            for name, shape, dtype in (
                ("images", (360, 512, 512), self.args.precision.replace("fp", "float")),
                ("volume_vjp", (256, 256, 256), "float32"),
            ):
                record = outputs["arrays"][name]
                path = (directory / record["file"]).resolve()
                if not path.is_relative_to(directory) or file_sha256(path) != record["sha256"]:
                    raise ValueError("Reference array path or checksum differs")
                array = self.np.load(path, mmap_mode="r", allow_pickle=False)
                if not (
                    array.shape == shape
                    and str(array.dtype) == dtype
                    and record["shape"] == list(shape)
                    and record["dtype"] == dtype
                    and hashlib.sha256(memoryview(array).cast("B")).hexdigest()
                    == record["raw_sha256"]
                    and all(self.np.isfinite(plane).all() for plane in array)
                ):
                    raise ValueError("Reference array values, shape or dtype differs")
                arrays[name] = array
            return {
                "arrays": arrays,
                "record": {
                    "directory": str(directory),
                    "results_sha256": file_sha256(result_path),
                    "outputs_sha256": file_sha256(output_path),
                    "source": result["source"],
                    "completed_at": result["completed_at"],
                    "precision": result["precision"],
                    "samples_per_ray": samples,
                    "arrays": outputs["arrays"],
                },
            }

        def check_reference(self, name, values):
            if "same_sample" not in self.references:
                return {"status": "baseline run without paired reference"}
            error = array_error(values, self.references["same_sample"]["arrays"][name], self.np)
            passed = (
                error["bitwise_equal"]
                if name == "images"
                else (
                    error["relative_l2"]
                    <= self.contract["quality"]["field_vjp_vs_baseline"]["relative_l2_max"]
                )
            )
            return {"passed": passed, **error}

        def save_array(self, name, values):
            repeated = super().save_array(name, values)
            manifest_path = self.args.output / "outputs.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update(
                samples_per_ray=self.args.samples,
                operation="fresh forward plus volume-only backward",
                requested_gradients=["volume"],
            )
            write_json(manifest_path, manifest)
            return repeated

        def run(self):
            torch = self.torch
            field = self.field.detach().requires_grad_(True)
            start = time.perf_counter()
            module = self.Projector(
                grid=self.scan.volume.grid,
                geometry=self.geometry,
                precision=self.args.precision,
                samples_per_ray=self.args.samples,
            )
            state = {}

            def call():
                state.clear()
                image = module(field)
                state["volume_vjp"] = torch.autograd.grad(
                    image, field, self.seed, retain_graph=False, create_graph=False
                )[0]
                state["image"] = image.detach()

            self.results["stage"] = "correctness before timing"
            self.save()
            call()
            self.stream.synchronize()
            self.results["preparation_seconds"]["module_and_first_complete_iteration"] = (
                time.perf_counter() - start
            )
            self.check_outputs("before_timing", state)
            adjoint = self.results["checks"]["before_timing"]["volume_adjoint"]
            self.results["objective"] = {
                "value": adjoint["image_dot_seed_fp64"],
                "formula": "FP64 sum of optical_depth times the exact saved FP32 cotangent",
                "units": "dimensionless optical depth",
            }
            if "sampling_difference" in self.references:
                self.results["sampling_difference_vs_1024"] = {
                    "scope": "discretisation difference only, not an accuracy gate",
                    "samples_per_ray": self.args.samples,
                    "reference_samples_per_ray": 1024,
                    "image": array_error(
                        self.np.load(self.args.output / "images.npy", mmap_mode="r"),
                        self.references["sampling_difference"]["arrays"]["images"],
                        self.np,
                    ),
                    "volume_vjp": array_error(
                        self.np.load(self.args.output / "volume_vjp.npy", mmap_mode="r"),
                        self.references["sampling_difference"]["arrays"]["volume_vjp"],
                        self.np,
                    ),
                }
            self.results["stage"] = "clean resident volume-only iteration timing"
            self.save()
            row = self.measure(
                "volume_only_iteration",
                call,
                {
                    "precision": self.args.precision,
                    "field_dtype": "float32",
                    "gradient_storage_dtype": "float32",
                    "requested_gradients": ["volume"],
                    "pose": "fixed identity, no pose tensor or pose gradient",
                    "samples_per_ray": self.args.samples,
                    "views": 360,
                    "detector_shape": [512, 512],
                    "input_residency": "CUDA",
                    "operation": "fresh Projector forward plus volume-only torch.autograd.grad",
                    "wall_boundary": (
                        "completed resident forward and volume backward including output/gradient "
                        "allocation, prior-output release and internal validation. Excludes "
                        "upload, correctness checks, reductions, downloads and file writes"
                    ),
                    "event_boundary": (
                        "CUDA events on the current Torch stream enclosing Projector stream bridges"
                    ),
                    "gradients_accumulate": False,
                    "graph_reused": False,
                },
            )
            row["first_use_limit"] = (
                "already warmed by the correctness iteration. Initial module and call time is "
                "preparation_seconds.module_and_first_complete_iteration"
            )
            self.check_outputs("after_timing", state)
            self.results["checks"]["no_leaf_gradient_accumulation"] = {
                "passed": field.grad is None,
            }
            if field.grad is not None:
                raise ArithmeticError("Unexpected leaf gradient accumulation")
            self.results["source_after"] = digest_sources(self.args.source)
            self.results["source_unchanged"] = self.results["source_after"] == self.source_before
            self.results["harness_after"] = {
                name: file_sha256(path) for name, path in self.results["harness_paths"].items()
            }
            if not (
                self.results["source_unchanged"]
                and self.results["harness_after"] == self.results["harness"]
            ):
                raise RuntimeError("Package or measurement helper changed during execution")
            self.results.update(status="complete", stage="complete", completed_at=utc())
            self.save()

    return VolumeIteration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--helper-source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--cotangent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp64"), required=True)
    parser.add_argument("--samples", choices=(256, 1024), type=int, required=True)
    parser.add_argument("--reference", type=Path, help="Completed same-mode, same-sample run")
    parser.add_argument(
        "--difference-reference-1024",
        type=Path,
        help="Optional same-mode 1024-sample run for an informational sampling comparison",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--nvml-interval-seconds", type=float, default=0.02)
    parser.add_argument("--nvml-device-index", type=int)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1 or args.nvml_interval_seconds <= 0:
        parser.error("rounds, repeats and NVML interval must be positive")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.views, args.operations = 360, ["volume_only_iteration"]
    args.long_threshold_seconds, args.long_repeats = 0.0, args.repeats
    probe_class = make_probe_class(args.helper_source)
    args.output.mkdir(parents=True, exist_ok=False)
    probe = probe_class.__new__(probe_class)
    try:
        probe.__init__(args)
        probe.run()
    except Exception as error:
        if hasattr(probe, "results"):
            from benchmarks.common import utc

            probe.results.update(
                status="failed",
                failed_at=utc(),
                error=repr(error),
                traceback=traceback.format_exc(),
            )
            probe.save()
        raise
    finally:
        if getattr(probe, "monitor", None) is not None:
            probe.monitor.close()


if __name__ == "__main__":
    main()
