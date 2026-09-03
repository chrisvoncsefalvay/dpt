#!/usr/bin/env python3
"""Validate and measure paired real-CT implementations under one serial GPU lock.

Only the worker imports CUDA libraries. The parent first completes both numerical
checks, then alternates reference/comparison measurement processes for three rounds.
Large shared reference arrays are written once per case, never once per round.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace


def utc():
    return datetime.now(UTC).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_digest(path):
    files = {str(p.relative_to(path)): sha(p) for p in sorted((path / "python/dpt").rglob("*.py"))}
    if not files:
        raise ValueError(f"No package source under {path}")
    return {
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "files": files,
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def same_json_input(actual, reference):
    """Compare saved inputs exactly, normalising only JSON sequence containers."""
    options = {"sort_keys": True, "separators": (",", ":"), "allow_nan": False}
    return json.dumps(actual, **options) == json.dumps(reference, **options)


def scalar_check(actual, expected, limits):
    error = abs(actual - expected)
    bound = limits["atol"] + limits["rtol"] * max(abs(actual), abs(expected))
    return {
        "actual": actual,
        "reference": expected,
        "absolute_error": error,
        "relative_error": error / max(abs(actual), abs(expected), 1e-300),
        "allowed_absolute_error": bound,
        "passed": error <= bound,
    }


def compare_arrays(np, actual, reference, limits):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        return {
            "passed": False,
            "reason": "shape or dtype differs",
            "actual_shape": list(actual.shape),
            "reference_shape": list(reference.shape),
            "actual_dtype": str(actual.dtype),
            "reference_dtype": str(reference.dtype),
        }
    a, b = actual.reshape(-1), reference.reshape(-1)
    maximum, difference_squared, reference_squared, bad = 0.0, 0.0, 0.0, 0
    for first in range(0, a.size, 1_000_000):
        x, y = (
            a[first : first + 1_000_000].astype(np.float64),
            b[first : first + 1_000_000].astype(np.float64),
        )
        difference = x - y
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return {"passed": False, "reason": "nonfinite actual or reference"}
        maximum = max(maximum, float(np.max(np.abs(difference), initial=0)))
        difference_squared += float(np.dot(difference, difference))
        reference_squared += float(np.dot(y, y))
        bad += int(
            np.count_nonzero(np.abs(difference) > limits["atol"] + limits["rtol"] * np.abs(y))
        )
    return {
        "passed": bad == 0,
        "failed_elements": bad,
        "elements": a.size,
        "max_abs": maximum,
        "rmse": (difference_squared / max(a.size, 1)) ** 0.5,
        "relative_l2": (difference_squared / max(reference_squared, 1e-300)) ** 0.5,
        "limits": limits,
    }


def inner(np, a, b):
    x, y, total = a.reshape(-1), b.reshape(-1), 0.0
    if x.size != y.size:
        raise ValueError("Inner-product shapes differ")
    for first in range(0, x.size, 1_000_000):
        total += float(
            np.dot(
                x[first : first + 1_000_000].astype(np.float64),
                y[first : first + 1_000_000].astype(np.float64),
            )
        )
    return total


def array_info(array):
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
    }


def detector_seed(np, shape, pattern, seed, dtype):
    rows, columns = shape
    result = np.zeros(shape, dtype=np.float64)
    if pattern == "central":
        result[rows * 5 // 16 : rows * 11 // 16, columns * 5 // 16 : columns * 11 // 16] = 1
    elif pattern == "dense":
        result[:] = 1
    elif pattern == "signed":
        phase = np.random.default_rng(seed).uniform(-np.pi, np.pi, 2)
        u, v = np.linspace(-1, 1, columns), np.linspace(-1, 1, rows)
        result = (
            np.sin(2 * np.pi * u + phase[0])[None, :] * np.cos(3 * np.pi * v + phase[1])[:, None]
        )
    elif pattern != "zero":
        raise ValueError(pattern)
    normalisation = float(np.sum(np.abs(result)))
    if normalisation:
        result /= normalisation
    return result.astype(dtype)


def density_basis(np, hu, materials):
    if materials not in (1, 2, 3):
        raise ValueError("Only frozen 1/2/3-channel bases are declared")
    nz, ny, nx = hu.shape
    x = np.linspace(-np.pi, np.pi, nx, dtype=np.float64)[None, :]
    y = np.linspace(-np.pi, np.pi, ny, dtype=np.float64)[:, None]
    if materials == 1:
        weights = [np.ones((ny, nx), dtype=np.float64)]
    elif materials == 2:
        w0 = 0.35 + 0.1 * np.sin(x) * np.cos(y)
        weights = [w0, 1 - w0]
    else:
        w0, w1 = 0.2 + 0.1 * np.sin(x), 0.3 + 0.1 * np.cos(y)
        weights = [w0, w1, 1 - w0 - w1]
    output = np.empty((materials, nz, ny, nx), dtype=np.float32)
    for z in range(nz):
        density = np.maximum(1 + hu[z].astype(np.float64) / 1000, 0)
        for channel, weight in enumerate(weights):
            output[channel, z] = density * weight
    return output


def build_case_class(release_benchmark):
    class CampaignCase(release_benchmark):
        def __init__(self, args, runtime, case, manifest, contract):
            super().__init__(args, runtime)
            from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose

            self.case, self.manifest, self.contract = case, manifest, contract
            self.mode = case["precision"]
            self.arithmetic = "float32" if self.mode == "fp32" else "float64"
            fixed = manifest["geometry"][case["geometry"]]
            if json.loads(json.dumps(asdict(self.grid))) != fixed["grid"]:
                raise ValueError("Physical grid differs from frozen campaign manifest")
            if self.input["sha256"] != fixed["image_sha256"]:
                raise ValueError("CT file differs from frozen campaign manifest")
            self.panel = DetectorGeometry(
                **{k: tuple(v) if isinstance(v, list) else v for k, v in fixed["panel"].items()}
            )
            self.panels = [self.panel]
            self.anchor = compose_pose(RigidTransform(), tuple(contract["anchor_tangent"]))
            with self.wp.ScopedStream(self.wp_stream):
                self.pose.assign(self.np.array(self.anchor.packed(), dtype=self.np.float64))
            self.seed_host = detector_seed(
                self.np,
                self.panel.shape,
                case["cotangent"],
                case["seed"],
                self.np.float32 if self.mode == "fp32" else self.np.float64,
            )
            with self.wp.ScopedStream(self.wp_stream):
                self.seed = self.wp.array(
                    self.seed_host.ravel(), dtype=self.dtype(self.mode), device="cuda:0"
                )
            self.results["campaign_case"] = case
            self.results["campaign_contract"] = contract
            self.results["input"]["geometry"] = [asdict(self.panel)]
            self.results["input"]["pose"] = list(self.anchor.packed())
            self.results["input"]["detector_cotangent"] = array_info(self.seed_host)
            self.operations = {}
            self.prepare_scalar() if case["kind"] == "scalar" else self.prepare_material()
            self.stream.synchronize()
            self.results["input"]["active_fields"] = array_info(self.fields_host)
            self.save()

        def prepare_scalar(self):
            from dpt.torch import Projector

            wp = self.wp
            self.fields_host = self.mu[None]
            self.fields_device = self.field
            self.workspace = self.base.prepare_projection(
                self.grid,
                self.panel,
                self.mode_spec(self.mode, pose=True, volume=True),
                device="cuda:0",
                stream=self.wp_stream,
            )
            with wp.ScopedStream(self.wp_stream):
                self.image = wp.empty(
                    self.panel.pixels, dtype=self.dtype(self.mode), device="cuda:0"
                )
                self.gradient = wp.empty(
                    self.grid.voxels, dtype=self.dtype(self.mode), device="cuda:0"
                )
                self.pose_gradient = wp.empty(6, dtype=wp.float64, device="cuda:0")

            def forward():
                self.base.project_optical_depth(
                    self.field,
                    self.pose,
                    workspace=self.workspace,
                    out_L=self.image,
                    validate=False,
                    stream=self.wp_stream,
                )

            def volume():
                with wp.ScopedStream(self.wp_stream):
                    self.gradient.zero_()
                self.base.projection_vjp(
                    self.field,
                    self.pose,
                    cotangent=self.seed,
                    workspace=self.workspace,
                    out_mu=self.gradient,
                    validate=False,
                    accumulate=True,
                    check_volume_gradient=False,
                    volume_gradient_precision=self.arithmetic,
                    stream=self.wp_stream,
                )

            def pose():
                self.base.projection_vjp(
                    self.field,
                    self.pose,
                    cotangent=self.seed,
                    workspace=self.workspace,
                    out_pose_tangent=self.pose_gradient,
                    validate=False,
                    accumulate=False,
                    stream=self.wp_stream,
                )

            self.projector = Projector(
                grid=self.grid,
                geometry=self.panel,
                precision=self.mode,
                samples_per_ray=1024,
                anchor=self.anchor,
            )
            self.torch_field = self.field_torch.detach().requires_grad_(True)
            self.torch_pose = self.torch.zeros(
                6, dtype=self.torch.float64, device="cuda:0", requires_grad=True
            )
            self.torch_seed = self.torch.tensor(self.seed_host, device="cuda:0")

            def torch_iteration():
                self.torch_field.grad = None
                self.torch_pose.grad = None
                self.torch_prediction = self.projector(self.torch_field, pose=self.torch_pose)
                (self.torch_prediction[0].double() * self.torch_seed.double()).sum().backward()

            self.forward = forward
            self.reverse = lambda: (volume(), pose())
            self.complete = lambda: (forward(), volume(), pose())
            self.operations = {
                "volume_vjp": volume,
                "pose_vjp": pose,
                "torch_complete_iteration": torch_iteration,
            }
            self.workspace.validate_inputs(self.field, self.pose, stream=self.wp_stream)

        def prepare_material(self):
            from dpt.material_projection import (
                material_projection_vjp,
                prepare_material_projection,
                project_material_paths,
            )
            from dpt.spectral import (
                SpectralSpec,
                prepare_spectral,
                spectral_signal,
                spectral_vjp,
            )

            np, wp, count = self.np, self.wp, self.case["materials"]
            self.fields_host = density_basis(np, self.ct.values, count)
            spectrum = self.manifest["spectra"][str(count)]
            coefficients = np.array(spectrum["coefficients_mm_inverse"], dtype=np.float32)
            weights = np.array(spectrum["incident_photons_per_pixel"], dtype=np.float32)
            response = np.array(spectrum["response"], dtype=np.float32)
            with wp.ScopedStream(self.wp_stream):
                self.fields_device = wp.array(
                    self.fields_host.reshape(count, -1),
                    dtype=wp.float32,
                    device="cuda:0",
                )
                self.paths = wp.empty(
                    (count, self.panel.pixels),
                    dtype=self.dtype(self.mode),
                    device="cuda:0",
                )
                self.path_seed = wp.empty_like(self.paths)
                self.gradient = wp.empty(
                    (count, self.grid.voxels),
                    dtype=self.dtype(self.mode),
                    device="cuda:0",
                )
                self.pose_gradient = wp.empty(6, dtype=wp.float64, device="cuda:0")
                self.image = wp.empty(
                    self.panel.pixels, dtype=self.dtype(self.mode), device="cuda:0"
                )
                coefficients_device = wp.array(coefficients, dtype=wp.float32, device="cuda:0")
                weights_device = wp.array(weights, dtype=wp.float32, device="cuda:0")
                response_device = wp.array(response, dtype=wp.float32, device="cuda:0")
            self.workspace = prepare_material_projection(
                self.grid,
                self.panel,
                self.mode_spec(self.mode, pose=True, volume=True),
                materials=count,
                fields=self.fields_device,
                out_paths=self.paths,
                cotangent_paths=self.path_seed,
                out_fields=self.gradient,
                device="cuda:0",
                stream=self.wp_stream,
                field_domain="nonnegative",
                channel_execution="fused",
                volume_gradient_precision=self.arithmetic,
            )
            self.spectral = prepare_spectral(
                SpectralSpec(
                    count,
                    len(weights),
                    precision=self.arithmetic,
                    output_unit="expected photons per pixel",
                    input_description="Declared deterministic CT basis and test spectrum",
                ),
                max_pixels=self.panel.pixels,
                device="cuda:0",
                stream=self.wp_stream,
            )

            def forward():
                project_material_paths(
                    self.pose,
                    workspace=self.workspace,
                    validate=False,
                    stream=self.wp_stream,
                )
                spectral_signal(
                    self.paths,
                    coefficients_device,
                    weights_device,
                    response_device,
                    out_mean=self.image,
                    workspace=self.spectral,
                    validate=False,
                    stream=self.wp_stream,
                )

            def seed_paths():
                spectral_vjp(
                    self.paths,
                    coefficients_device,
                    weights_device,
                    response_device,
                    cotangent=self.seed,
                    out_grad_paths=self.path_seed,
                    workspace=self.spectral,
                    validate=False,
                    stream=self.wp_stream,
                )

            def reverse():
                with wp.ScopedStream(self.wp_stream):
                    self.gradient.zero_()
                    self.pose_gradient.zero_()
                material_projection_vjp(
                    self.pose,
                    workspace=self.workspace,
                    out_pose_tangent=self.pose_gradient,
                    validate=False,
                    accumulate=True,
                    check_volume_gradient=False,
                    stream=self.wp_stream,
                )

            def complete():
                forward()
                seed_paths()
                reverse()

            self.forward, self.reverse, self.complete = forward, reverse, complete
            self.seed_paths = seed_paths
            self.operations = {
                "material_vjp": reverse,
                "spectral_complete_iteration": complete,
            }
            self.results["material_configuration"] = {
                "basis": self.manifest["material_basis"],
                "spectrum": spectrum,
                "coefficients": array_info(coefficients),
                "weights": array_info(weights),
                "response": array_info(response),
                "gradient_storage": self.arithmetic,
                "pose_storage": "float64",
            }
            project_material_paths(
                self.pose,
                workspace=self.workspace,
                validate=True,
                stream=self.wp_stream,
            )
            spectral_signal(
                self.paths,
                coefficients_device,
                weights_device,
                response_device,
                out_mean=self.image,
                workspace=self.spectral,
                validate=True,
                stream=self.wp_stream,
            )
            seed_paths()

        def check_status(self):
            native = self.workspace if self.case["kind"] == "scalar" else self.workspace.projection
            destinations = (
                (self.gradient,) if self.case["kind"] == "scalar" else self.workspace.gradient_views
            )
            for destination in destinations:
                native.enqueue_volume_gradient_check(
                    destination,
                    volume_gradient_precision=self.arithmetic,
                    stream=self.wp_stream,
                )
            self.workspace.check_status()
            if self.case["kind"] == "material":
                self.spectral.check_status()

        def snapshot(self):
            self.complete()
            self.stream.synchronize()
            self.check_status()
            arrays = {
                "image": self.image.numpy().reshape(self.panel.shape),
                "volume_gradient": self.gradient.numpy().reshape(self.fields_host.shape),
                "pose_gradient": self.pose_gradient.numpy(),
            }
            if self.case["kind"] == "material":
                arrays.update(paths=self.paths.numpy(), path_cotangent=self.path_seed.numpy())
            else:
                self.operations["torch_complete_iteration"]()
                self.stream.synchronize()
                native32 = arrays["volume_gradient"][0].astype(self.np.float32)
                paired = self.contract["modes"][self.mode]["paired"]
                self.results["checks"]["torch_matches_native"] = {
                    "image": compare_arrays(
                        self.np,
                        self.torch_prediction.detach().cpu().numpy()[0],
                        arrays["image"],
                        paired["image"],
                    ),
                    "volume": compare_arrays(
                        self.np,
                        self.torch_field.grad.detach().cpu().numpy(),
                        native32,
                        self.contract["modes"]["fp32"]["paired"]["volume_gradient"],
                    ),
                    "pose": compare_arrays(
                        self.np,
                        self.torch_pose.grad.detach().cpu().numpy(),
                        arrays["pose_gradient"],
                        paired["pose_gradient"],
                    ),
                }
            return arrays

        def assign_fields(self, value):
            shaped = (
                value.ravel()
                if self.case["kind"] == "scalar"
                else value.reshape(self.case["materials"], -1)
            )
            with self.wp.ScopedStream(self.wp_stream):
                self.fields_device.assign(shaped)

        def objective(self):
            self.forward()
            self.stream.synchronize()
            self.workspace.check_status()
            if self.case["kind"] == "material":
                self.spectral.check_status()
            return inner(self.np, self.image.numpy(), self.seed_host)

        def numerical_checks(self, arrays):
            from dpt.geometry import compose_pose

            np = self.np
            limits = self.contract["modes"][self.mode][self.case["kind"]]
            values = arrays["image"] if self.case["kind"] == "scalar" else arrays["paths"]
            cotangent = (
                self.seed_host if self.case["kind"] == "scalar" else arrays["path_cotangent"]
            )
            checks = {
                "adjoint": scalar_check(
                    inner(np, values, cotangent),
                    inner(np, self.fields_host, arrays["volume_gradient"]),
                    limits["adjoint"],
                )
            }
            if self.case["cotangent"] == "zero":
                zero_keys = ("volume_gradient", "pose_gradient") + (
                    ("path_cotangent",) if self.case["kind"] == "material" else ()
                )
                checks["zero"] = {
                    "passed": all(np.count_nonzero(arrays[k]) == 0 for k in zero_keys),
                    "nonzero_counts": {k: int(np.count_nonzero(arrays[k])) for k in zero_keys},
                }
            step = limits["volume_fd"]["step"]
            factor = 0.25 + 0.75 * np.sin(
                np.linspace(-np.pi, np.pi, self.grid.shape[2], dtype=np.float64)
            )
            positive, negative = (
                np.empty_like(self.fields_host),
                np.empty_like(self.fields_host),
            )
            realised_vjp = 0.0
            for channel in range(self.fields_host.shape[0]):
                for z in range(self.grid.shape[0]):
                    field = self.fields_host[channel, z].astype(np.float64)
                    positive[channel, z] = field + step * field * factor
                    negative[channel, z] = field - step * field * factor
                    direction = (
                        positive[channel, z].astype(np.float64)
                        - negative[channel, z].astype(np.float64)
                    ) / (2 * step)
                    realised_vjp += inner(np, arrays["volume_gradient"][channel, z], direction)
            objectives = []
            for field in (positive, negative):
                self.assign_fields(field)
                objectives.append(self.objective())
            self.assign_fields(self.fields_host)
            checks["volume_fd"] = {
                **scalar_check(
                    realised_vjp,
                    (objectives[0] - objectives[1]) / (2 * step),
                    limits["volume_fd"],
                ),
                "step": step,
                "objectives": objectives,
                "direction": self.manifest["volume_fd_direction"],
            }
            del positive, negative
            direction = np.array(self.contract["pose_direction"], dtype=np.float64)
            pose_vjp = float(np.dot(arrays["pose_gradient"], direction))
            pose_rows = []
            for step in limits["pose_fd"]["steps"]:
                values = []
                for sign in (1, -1):
                    shifted = compose_pose(self.anchor, tuple(sign * step * direction))
                    with self.wp.ScopedStream(self.wp_stream):
                        self.pose.assign(np.array(shifted.packed(), dtype=np.float64))
                    values.append(self.objective())
                pose_rows.append(
                    {
                        "step": step,
                        "objectives": values,
                        **scalar_check(
                            pose_vjp,
                            (values[0] - values[1]) / (2 * step),
                            limits["pose_fd"],
                        ),
                    }
                )
            selected = next(
                row for row in pose_rows if row["step"] == limits["pose_fd"]["selected_step"]
            )
            checks["pose_fd"] = {
                "passed": selected["passed"],
                "selected_step": selected["step"],
                "samples": pose_rows,
                "selection": "fixed before baseline measurements",
            }
            with self.wp.ScopedStream(self.wp_stream):
                self.pose.assign(np.array(self.anchor.packed(), dtype=np.float64))
            self.complete()
            self.stream.synchronize()
            self.check_status()
            self.results["checks"].update(checks)
            self.save()

    return CampaignCase


def all_passed(value):
    if isinstance(value, dict):
        # Pose diagnostic steps are retained but only the preselected step gates.
        if "passed" in value:
            return bool(value["passed"])
        return all(all_passed(item) for item in value.values())
    if isinstance(value, list):
        return all(all_passed(item) for item in value)
    return True


def worker(args):
    run = json.loads(args.run_manifest.read_text())
    source = Path(run["sources"][args.role]["path"])
    if source_digest(source) != run["sources"][args.role]["digest"]:
        raise RuntimeError("Source changed since the paired job was frozen")
    if sha(Path(__file__)) != run["harness_sha256"]:
        raise RuntimeError("Harness changed during the paired job")
    for path, digest in run["helper_hashes"].items():
        if sha(path) != digest:
            raise RuntimeError("Shared benchmark helper changed during the paired job")
    manifest, contract = run["input_manifest"], run["gradient_contract"]
    sys.path[:0] = [str(source / "python"), run["helper_source"]]
    from benchmarks.common import load_runtime
    from benchmarks.real_ct import ReleaseBenchmark

    runtime = load_runtime()
    case_class = build_case_class(ReleaseBenchmark)
    output = Path(run["output"])
    phase_name = "gate" if args.phase == "gate" else f"round-{args.round_index}"
    worker_output = output / phase_name / args.role
    worker_output.mkdir(parents=True, exist_ok=False)
    status = {
        "started_at": utc(),
        "role": args.role,
        "phase": args.phase,
        "source": run["sources"][args.role],
        "cases": [],
        "passed": False,
    }
    write_json(worker_output / "worker.json", status)
    try:
        for case in run["cases"]:
            fixed = manifest["geometry"][case["geometry"]]
            options = SimpleNamespace(
                ct=Path(run["ct"][fixed["anatomy"]]),
                anatomy=fixed["anatomy"],
                source=source,
                baseline_results=Path(run["baseline_results"]),
                output=worker_output / case["id"],
                shape=tuple(fixed["panel"]["shape"]),
                modes=[case["precision"]],
                cases=[case["kind"]],
                view="ap" if case["geometry"].endswith("-ap") else "oblique",
                rounds=1,
                repeats=run["repeats"],
                long_repeats=10,
                long_threshold_seconds=0.2,
                nvml_interval_seconds=0.05,
                nvml_device_index=None,
            )
            print(utc(), args.role, phase_name, case["id"], flush=True)
            benchmark = None
            try:
                benchmark = case_class(options, runtime, case, manifest, contract)
                benchmark.results["paired_run"] = {
                    "input_manifest_sha256": run["input_manifest_sha256"],
                    "gradient_contract_sha256": run["gradient_contract_sha256"],
                    "harness_sha256": run["harness_sha256"],
                    "role": args.role,
                    "phase": args.phase,
                    "round": args.round_index,
                }
                reference_directory = output / "references" / case["id"]
                if args.phase == "gate":
                    arrays = benchmark.snapshot()
                    benchmark.results["snapshot_arrays"] = {
                        name: array_info(array) for name, array in arrays.items()
                    }
                    benchmark.numerical_checks(arrays)
                    if args.role == "baseline":
                        reference_directory.mkdir(parents=True, exist_ok=False)
                        reference = {
                            "source": run["sources"]["baseline"]["digest"],
                            "input": benchmark.results["input"],
                            "arrays": {},
                        }
                        for name, array in arrays.items():
                            path = reference_directory / (name + ".npy")
                            runtime.np.save(path, array, allow_pickle=False)
                            reference["arrays"][name] = {
                                "file": path.name,
                                "file_sha256": sha(path),
                                **array_info(array),
                            }
                        write_json(reference_directory / "reference.json", reference)
                    else:
                        reference = json.loads((reference_directory / "reference.json").read_text())
                        for key in (
                            "sha256",
                            "grid",
                            "geometry",
                            "pose",
                            "detector_cotangent",
                            "active_fields",
                        ):
                            if not same_json_input(
                                benchmark.results["input"][key], reference["input"][key]
                            ):
                                raise AssertionError(
                                    f"Candidate input differs from frozen baseline: {key}"
                                )
                        paired = {}
                        for name, array in arrays.items():
                            info = reference["arrays"][name]
                            path = reference_directory / info["file"]
                            if sha(path) != info["file_sha256"]:
                                raise AssertionError("Frozen reference array changed: " + str(path))
                            frozen = runtime.np.load(path, mmap_mode="r", allow_pickle=False)
                            paired[name] = compare_arrays(
                                runtime.np,
                                array,
                                frozen,
                                contract["modes"][case["precision"]]["paired"][name],
                            )
                        benchmark.results["checks"]["paired_baseline"] = paired
                    benchmark.results["gate_passed"] = all_passed(benchmark.results["checks"])
                    benchmark.save()
                    if not benchmark.results["gate_passed"]:
                        raise AssertionError(
                            "Numerical gate failed; inspect stored checks; no timings authorised"
                        )
                    del arrays
                else:
                    for role in ("baseline", "candidate"):
                        gate = json.loads(
                            (output / "gate" / role / case["id"] / "results.json").read_text()
                        )
                        if (
                            not gate.get("gate_passed")
                            or gate["source"] != run["sources"][role]["digest"]
                        ):
                            raise AssertionError(
                                "Timing requires matching passed baseline and candidate gates"
                            )
                    benchmark.complete()
                    benchmark.stream.synchronize()
                    benchmark.check_status()
                    for name, call in benchmark.operations.items():
                        benchmark.measure(
                            case["id"] + "/" + name,
                            call,
                            {
                                "case": case,
                                "sampling": manifest["sampling"],
                                "geometry": fixed,
                                "pose": list(benchmark.anchor.packed()),
                                "boundary": (
                                    "resident completed operation; destination resets included; "
                                    "status/finite checks outside timing"
                                ),
                                "torch_boundary": (
                                    "Torch complete iteration also includes module "
                                    "validation, allocation and autograd"
                                ),
                            },
                            check=benchmark.check_status,
                        )
                ending = source_digest(source)
                if ending != run["sources"][args.role]["digest"]:
                    raise AssertionError(
                        "Source changed during operation; result cannot be accepted"
                    )
                benchmark.results["source_unchanged_during_measurement"] = True
                benchmark.results["measurement_finished_at"] = utc()
                benchmark.results["environment"]["nvidia_smi_after"] = runtime.command(
                    "nvidia-smi", "-q"
                )
                benchmark.save()
                status["cases"].append(
                    {
                        "id": case["id"],
                        "passed": True,
                        "result": str(options.output / "results.json"),
                    }
                )
                write_json(worker_output / "worker.json", status)
            except BaseException as error:
                if benchmark is not None:
                    benchmark.results["failure"] = {
                        "at": utc(),
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    benchmark.save()
                raise
            finally:
                if benchmark is not None:
                    benchmark.monitor.close()
                benchmark = None
                import gc

                gc.collect()
        status["passed"] = True
    finally:
        status["finished_at"] = utc()
        write_json(worker_output / "worker.json", status)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (
        position - low
    )


def summarise(run):
    root, rows = Path(run["output"]), []
    for case in run["cases"]:
        for operation in case["operations"]:
            row = {
                "case": case,
                "operation": operation,
                "timing_boundary": "wall" if operation.startswith("torch") else "device",
            }
            round_medians = {"baseline": [], "candidate": []}
            for role in ("baseline", "candidate"):
                values = {"wall": [], "device": []}
                paths = []
                for index in range(3):
                    path = root / f"round-{index}" / role / case["id"] / "results.json"
                    data = json.loads(path.read_text())
                    entry = next(
                        r
                        for r in data["measurements"]
                        if r["label"] == case["id"] + "/" + operation
                    )
                    if len(entry["rounds"]) != 1 or not data.get(
                        "source_unchanged_during_measurement"
                    ):
                        raise ValueError("Incomplete timed round")
                    paths.append(str(path))
                    for kind in values:
                        samples = entry["rounds"][0][kind + "_ms"]
                        if len(samples) != entry["repeats_per_round"]:
                            raise ValueError("Incomplete raw sample list")
                        values[kind].extend(samples)
                        if kind == row["timing_boundary"]:
                            round_medians[role].append(percentile(samples, 0.5))
                row[role] = {
                    kind: {
                        "median_ms": percentile(samples, 0.5),
                        "p95_ms": percentile(samples, 0.95),
                        "count": len(samples),
                        "raw_ms": samples,
                    }
                    for kind, samples in values.items()
                }
                row[role]["result_paths"] = paths
            kind = row["timing_boundary"]
            before, after = (
                row["baseline"][kind]["median_ms"],
                row["candidate"][kind]["median_ms"],
            )
            row["speedup"] = before / after
            row["latency_change_fraction"] = after / before - 1
            row["paired_round_speedups"] = [
                a / b
                for a, b in zip(round_medians["baseline"], round_medians["candidate"], strict=True)
            ]
            row["three_percent_gain_every_round"] = all(
                value >= 1.03 for value in row["paired_round_speedups"]
            )
            row["regression_over_three_percent"] = after > before * 1.03
            rows.append(row)
    primary = [
        row
        for row in rows
        if row["case"]["suite"] == "dev"
        and row["case"]["precision"] == "fp64"
        and row["case"]["cotangent"] != "zero"
        and row["operation"].endswith("complete_iteration")
    ]
    guards = [
        row
        for row in rows
        if row["case"]["precision"] == "fp32" or row["case"]["suite"] == "holdout"
    ]

    def geometric_mean(values):
        return math.exp(sum(math.log(v) for v in values) / len(values))

    primary_speedup = geometric_mean([row["speedup"] for row in primary]) if primary else None
    primary_rounds = (
        [geometric_mean([row["paired_round_speedups"][i] for row in primary]) for i in range(3)]
        if primary
        else []
    )
    regressions = [
        {
            "case": row["case"]["id"],
            "operation": row["operation"],
            "latency_change_fraction": row["latency_change_fraction"],
        }
        for row in guards
        if row["regression_over_three_percent"]
    ]
    selection = {
        "definition": (
            "Geometric mean of nonzero FP64 development complete-iteration "
            "speedups; exact-zero controls excluded"
        ),
        "primary_cases": [row["case"]["id"] for row in primary],
        "primary_geometric_mean_speedup": primary_speedup,
        "primary_geometric_mean_speedup_by_round": primary_rounds,
        "primary_three_percent_gain_every_round": bool(primary_rounds)
        and all(value >= 1.03 for value in primary_rounds),
        "fp32_and_holdout_regressions_over_three_percent": regressions,
        "regression_interpretation": (
            "Flags require explanation or rejection; this harness does not waive a regression"
        ),
        "full_frozen_manifest_measured": {c["id"] for c in run["cases"]}
        == {c["id"] for c in run["input_manifest"]["cases"]},
    }
    return {
        "completed_at": utc(),
        "correctness_passed": True,
        "rows": rows,
        "selection": selection,
        "source": run["sources"],
        "input_manifest_sha256": run["input_manifest_sha256"],
        "gradient_contract_sha256": run["gradient_contract_sha256"],
        "selection_limit": (
            "These per-case flags do not establish a universal optimisation win; "
            "inspect holdouts and resource evidence."
        ),
    }


def parent(args):
    manifest = json.loads(args.input_manifest.read_text())
    contract = json.loads(args.contract.read_text())
    if sha(args.contract) != manifest["gradient_contract_sha256"]:
        raise ValueError("Frozen numerical contract checksum differs")
    cases = [c for c in manifest["cases"] if args.suite == "both" or c["suite"] == args.suite]
    if args.case_ids:
        unknown = set(args.case_ids) - {c["id"] for c in cases}
        if unknown:
            raise ValueError(f"Cases absent from selected frozen suite: {unknown}")
        cases = [c for c in cases if c["id"] in args.case_ids]
    if args.list_cases:
        print(json.dumps(cases, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    run = {
        "schema_version": 1,
        "created_at": utc(),
        "output": str(args.output.resolve()),
        "input_manifest": manifest,
        "input_manifest_sha256": sha(args.input_manifest),
        "gradient_contract": contract,
        "gradient_contract_sha256": sha(args.contract),
        "harness_sha256": sha(Path(__file__)),
        "helper_source": str(args.baseline_source.resolve()),
        "ct": {"chest": str(args.chest.resolve()), "brain": str(args.brain.resolve())},
        "baseline_results": str(
            (args.baseline_source / "docs/assets/results/baseline-chest-512.json").resolve()
        ),
        "cases": cases,
        "repeats": 100 if args.preset == "final" else 20,
        "lock": str(args.lock.resolve()),
        "sources": {},
        "correctness_passed": False,
        "commands": [],
        "status": "waiting_for_gpu_lock",
    }
    for role in ("baseline", "candidate"):
        source = getattr(args, role + "_source").resolve()
        run["sources"][role] = {"path": str(source), "digest": source_digest(source)}
    run["helper_hashes"] = {
        str((args.baseline_source / "benchmarks" / name).resolve()): sha(
            args.baseline_source / "benchmarks" / name
        )
        for name in ("real_ct.py", "common.py")
    }
    run_file = args.output.resolve() / "campaign.json"
    write_json(run_file, run)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.lock.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            run["lock_acquired_at"], run["status"] = utc(), "correctness"
            write_json(run_file, run)

            def launch(role, phase, index=0):
                command = [
                    sys.executable,
                    "-P",
                    str(Path(__file__).resolve()),
                    "worker",
                    "--run-manifest",
                    str(run_file),
                    "--role",
                    role,
                    "--phase",
                    phase,
                    "--round-index",
                    str(index),
                ]
                cache = args.output.resolve() / "cache" / role
                env = {
                    **os.environ,
                    "WARP_CACHE_PATH": str(cache / "warp"),
                    "CUDA_CACHE_PATH": str(cache / "cuda"),
                    "OMP_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "2",
                }
                (cache / "warp").mkdir(parents=True, exist_ok=True)
                (cache / "cuda").mkdir(parents=True, exist_ok=True)
                row = {"command": command, "started_at": utc()}
                run["commands"].append(row)
                write_json(run_file, run)
                process = subprocess.run(command, env=env, check=False)
                row.update(returncode=process.returncode, finished_at=utc())
                write_json(run_file, run)
                if process.returncode:
                    raise RuntimeError(
                        f"{role} {phase} failed with exit {process.returncode}; no further timings"
                    )

            launch("baseline", "gate")
            launch("candidate", "gate")
            run["correctness_passed"] = True
            run["status"] = "gates_passed" if args.gate_only else "timing"
            write_json(run_file, run)
            if not args.gate_only:
                for index in range(3):
                    order = (
                        ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
                    )
                    for role in order:
                        launch(role, "time", index)
                write_json(args.output / "summary.json", summarise(run))
            run["status"], run["completed_at"] = "complete", utc()
    except BaseException as error:
        run["status"] = "failed"
        run["failure"] = {
            "at": utc(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        write_json(run_file, run)
        write_json(
            args.output / "candidate_manifest.json",
            {
                "schema_version": 1,
                "source": run["sources"]["candidate"],
                "baseline_source": run["sources"]["baseline"],
                "input_manifest_sha256": run["input_manifest_sha256"],
                "gradient_contract_sha256": run["gradient_contract_sha256"],
                "harness_sha256": run["harness_sha256"],
                "status": run["status"],
                "correctness_passed": run["correctness_passed"],
                "campaign_result": str(run_file),
                "summary": str(args.output.resolve() / "summary.json")
                if (args.output / "summary.json").exists()
                else None,
                "gates": {
                    role: str(args.output.resolve() / "gate" / role / "worker.json")
                    for role in ("baseline", "candidate")
                },
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Gate both revisions, then measure paired rounds under a lock")
    run.add_argument("--baseline-source", type=Path, required=True)
    run.add_argument("--candidate-source", type=Path, required=True)
    run.add_argument("--chest", type=Path, required=True)
    run.add_argument("--brain", type=Path, required=True)
    run.add_argument(
        "--input-manifest",
        type=Path,
        default=Path(__file__).with_name("input_manifest.json"),
    )
    run.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("gradient_contract.json"),
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--lock", type=Path, required=True)
    run.add_argument("--suite", choices=("dev", "holdout", "both"), default="dev")
    run.add_argument("--preset", choices=("dev", "final"), default="dev")
    run.add_argument("--case-ids", nargs="+")
    run.add_argument("--gate-only", action="store_true")
    run.add_argument("--list-cases", action="store_true")
    child = sub.add_parser("worker", help=argparse.SUPPRESS)
    child.add_argument("--run-manifest", type=Path, required=True)
    child.add_argument("--role", choices=("baseline", "candidate"), required=True)
    child.add_argument("--phase", choices=("gate", "time"), required=True)
    child.add_argument("--round-index", type=int, default=0)
    args = parser.parse_args()
    parent(args) if args.command == "run" else worker(args)


if __name__ == "__main__":
    main()
