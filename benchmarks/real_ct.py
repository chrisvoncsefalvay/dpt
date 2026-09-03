# ruff: noqa: B023
# Timed closures execute synchronously inside their current precision/view loop.
"""Measure native real-CT workloads on one GPU, one process at a time.

Every output directory is new. Select cases to split long jobs; all timings use
three rounds by default and retain raw event/wall samples and sampled NVML peaks.
The frozen chest acquisition and FP32 attenuation coefficients are unchanged.
Run correctness checks before selecting a precision/performance winner.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from benchmarks.common import Measurements, NvmlMonitor, load_runtime

# Declared before collection; these are acceptance bounds for these real cases,
# not universal patient-data guarantees. Never alter a failed bound during a run.
ACCEPTANCE = {
    "fp32": {
        "image_max_abs": 5e-4,
        "image_rmse": 2e-5,
        "image_relative_l2": 1e-5,
        "adjoint_relative": 2e-5,
        "volume_directional_rtol": 5e-3,
        "volume_directional_atol": 1e-6,
        "pose_directional_rtol": 2e-2,
        "pose_directional_atol": 2e-5,
    },
    "fp64": {
        "image_max_abs": 2e-10,
        "image_rmse": 2e-11,
        "image_relative_l2": 2e-11,
        "adjoint_relative": 2e-10,
        "volume_directional_rtol": 2e-5,
        "volume_directional_atol": 1e-8,
        "pose_directional_rtol": 5e-3,
        "pose_directional_atol": 1e-7,
    },
}

MATERIAL_ACCEPTANCE = {
    "fp32": {"adjoint_relative": 2e-5, "volume_rtol": 5e-3, "volume_atol": 1e-2},
    "fp64": {"adjoint_relative": 2e-10, "volume_rtol": 2e-5, "volume_atol": 1e-6},
}


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ct", type=Path, required=True)
    p.add_argument("--anatomy", choices=("chest", "brain"), default="chest")
    p.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument(
        "--baseline-results",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/assets/results/baseline-chest-512.json",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--shape",
        type=int,
        nargs=2,
        help="rows columns; defaults to 512 512 or brain 511 769",
    )
    p.add_argument("--modes", nargs="+", choices=("fp32", "fp64"), default=["fp32", "fp64"])
    p.add_argument(
        "--cases",
        nargs="+",
        required=True,
        choices=("checks", "forward", "derivatives", "torch", "batched", "material"),
    )
    p.add_argument("--view", choices=("ap", "lateral", "oblique"), default="ap")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--long-repeats", type=int, default=10)
    p.add_argument("--long-threshold-seconds", type=float, default=0.2)
    p.add_argument("--nvml-interval-seconds", type=float, default=0.05)
    p.add_argument("--nvml-device-index", type=int)
    p.add_argument("--batch-views", nargs="+", type=int, default=[1, 16, 128])
    p.add_argument("--max-batched-bytes", type=int, default=8 * 1024**3)
    p.add_argument(
        "--torch-gradients",
        nargs="+",
        choices=("volume", "pose", "both"),
        default=["volume", "pose", "both"],
    )
    p.add_argument(
        "--forward-boundaries",
        nargs="+",
        choices=("resident", "public", "completed"),
        default=["resident", "public", "completed"],
    )
    a = p.parse_args()
    a.shape = tuple(a.shape or ((512, 512) if a.anatomy == "chest" else (511, 769)))
    if min(a.rounds, a.repeats, a.long_repeats, *a.shape, *a.batch_views) < 1:
        p.error("rounds, repeats, shapes and view counts must be positive")
    if a.max_batched_bytes < 1:
        p.error("max-batched-bytes must be positive")
    if a.long_threshold_seconds <= 0 or a.nvml_interval_seconds <= 0:
        p.error("time intervals must be positive")
    if a.output.exists() and any(a.output.iterdir()):
        p.error("output directory must be absent or empty; existing measurements are preserved")
    if "batched" in a.cases and (a.anatomy != "chest" or a.shape != (512, 512)):
        p.error("batched 1/16/128 workload uses native chest with a 512 by 512 detector")
    return a


class ReleaseBenchmark(Measurements):
    """Reuse the shared timing and NVML methods; prepare measured inputs anew."""

    def __init__(self, args: argparse.Namespace, base: Any):
        from dpt.geometry import DetectorGeometry
        from dpt.imaging import ct_to_attenuation, read_volume, standard_views

        self.args, self.base = args, base
        self.np, self.torch, self.wp = base.np, base.torch, base.wp
        self.baseline = json.loads(args.baseline_results.read_text())
        frozen = self.baseline["input"]
        if frozen["samples_per_ray"] != 1024 or frozen["resampled"]:
            raise ValueError("Frozen baseline must use the native CT and 1024 midpoints")
        if frozen["energy_kev"] != 60 or frozen["mu_water_mm_inverse"] != 0.02059:
            raise ValueError("Frozen baseline must use the declared water-scaled 60 keV model")
        args.output.mkdir(parents=True, exist_ok=True)
        self.results = {
            "schema_version": 2,
            "measurement_started_at": base.utc(),
            "source": base.digest_sources(args.source),
            "source_revision": base.command("git", "-C", str(args.source), "rev-parse", "HEAD"),
            "timed_harness_sources": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__),
                    Path(base.__file__),
                    Path(__file__).with_name("common.py"),
                )
            },
            "baseline_results_sha256": hashlib.sha256(
                args.baseline_results.read_bytes()
            ).hexdigest(),
            "acceptance_thresholds": ACCEPTANCE,
            "material_acceptance_thresholds": MATERIAL_ACCEPTANCE,
            "acceptance_scope": "these pinned native-volume cases only; fixed before measurements",
            "acceptance_quantity_units": {
                "image": "dimensionless optical depth; absolute and relative image norms",
                "adjoint": "dimensionless relative discrepancy of <A mu,seed> and <mu,A^T seed>",
                "volume_directional": (
                    "optical-depth objective per unit dimensionless field-direction step"
                ),
                "pose_directional": (
                    "optical-depth objective per unit direction step; vector entries "
                    "in mm and radians"
                ),
                "material_adjoint": (
                    "relative discrepancy of <L density,path cotangent> and "
                    "<density,L^T path cotangent>; both inner products in expected photons/pixel"
                ),
                "material_volume_directional": (
                    "expected photons/pixel per unit dimensionless density-direction step; "
                    "FP32 absolute tolerance is 1e-6 of 10000 incident photons/pixel"
                ),
            },
            "selected_cases": args.cases,
            "selected_modes": args.modes,
            "environment": {
                "platform": base.platform.platform(),
                "python": base.platform.python_version(),
                "versions": {
                    n: base.importlib.metadata.version(n)
                    for n in ("numpy", "torch", "warp-lang", "nibabel", "nvidia-ml-py")
                },
                "torch_cuda": self.torch.version.cuda,
                "gpu": self.torch.cuda.get_device_name(0),
                "nvidia_smi_before": base.command("nvidia-smi", "-q"),
                "nsys_version": base.command("nsys", "--version"),
                "ncu_version": base.command("ncu", "--version"),
            },
            "timing_policy": {
                "rounds": args.rounds,
                "short_repeats": args.repeats,
                "long_repeats": args.long_repeats,
                "long_threshold_seconds": args.long_threshold_seconds,
                "warmup": "one first use and one completed pilot of the exact operation",
                "events": (
                    "CUDA events on the current Torch stream shared by the measured Warp workspaces"
                ),
                "event_limit": (
                    "host launch gaps inside a completed operation remain in event intervals"
                ),
                "wall": (
                    "completed operation with its declared allocation, validation and "
                    "output boundary"
                ),
                "logging": "NVML background samples; JSON/array writes outside timing",
                "cold_limit": (
                    "first use is reported separately; this is not an isolated-cache cold-start run"
                ),
                "quiet_interval": (
                    "inspect actual utilisation/process samples; no isolation inferred"
                ),
            },
            "preparation": {},
            "measurements": [],
            "checks": {},
        }
        self.save()
        started = time.perf_counter()
        self.ct = read_volume(args.ct, units="HU")
        self.results["preparation"]["file_loading_seconds"] = time.perf_counter() - started
        digest = hashlib.sha256(args.ct.read_bytes()).hexdigest()
        if args.anatomy == "chest":
            if (
                digest != frozen["sha256"]
                or json.loads(json.dumps(asdict(self.ct.grid))) != frozen["grid"]
            ):
                raise ValueError("Chest values/physical grid differ from the frozen baseline")
        else:
            expected = "da99b9b2175a5126f5d12e3c05633f677ce32ac85d7e068a53e62b1b420419e7"
            if digest != expected:
                raise ValueError("Brain file differs from the verified pinned Slicer conversion")
        started = time.perf_counter()
        self.attenuation = ct_to_attenuation(self.ct, energy_kev=60)
        self.mu = self.attenuation.values
        self.results["preparation"]["ct_conversion_seconds"] = time.perf_counter() - started
        # Check the shared public conversion against the exact baseline formula,
        # one slice at a time so verification does not require a second full FP64 CT.
        for index in range(self.mu.shape[0]):
            expected = (
                self.np.maximum(1.0 + self.ct.values[index].astype(self.np.float64) / 1000.0, 0.0)
                * 0.02059
            ).astype(self.np.float32)
            if not self.np.array_equal(self.mu[index], expected):
                raise ValueError(
                    "Public HU conversion does not preserve the baseline FP32 field values"
                )
        self.grid = self.ct.grid
        if args.anatomy == "chest":
            original = [
                DetectorGeometry(
                    **{k: tuple(v) if isinstance(v, list) else v for k, v in entry.items()}
                )
                for entry in frozen["geometry"]
            ]
            self.panels = [self.resize_panel(panel, args.shape) for panel in original]
        else:
            self.panels = list(
                standard_views(self.grid, ("ap", "lateral", "oblique"), shape=args.shape).views
            )
        self.input = {
            "file": str(args.ct.resolve()),
            "sha256": digest,
            "anatomy": args.anatomy,
            "grid": asdict(self.grid),
            "native_shape_zyx": list(self.grid.shape),
            "resampled": False,
            "units": "HU",
            "attenuation_model": "water-scaled monoenergetic",
            "energy_kev": 60,
            "mu_water_mm_inverse": 0.02059,
            "field_dtype": "float32",
            "field_sha256": hashlib.sha256(self.mu.tobytes()).hexdigest(),
            "field_matches_baseline_formula_exactly": True,
            "geometry": [asdict(p) for p in self.panels],
            "view_names": ["ap", "lateral", "oblique"],
            "samples_per_ray": 1024,
            "integration": "clipped midpoint",
            "pose": frozen["pose"],
            "physical_acquisition": "frozen chest source/panel extents"
            if args.anatomy == "chest"
            else "standard RAS views around native oblique brain support",
        }
        if args.anatomy == "chest":
            for key in ("source_object_distance_mm", "source_detector_distance_mm"):
                self.input[key] = frozen[key]
            self.input["detector_pitch_column_row_mm"] = list(self.panels[0].spacing_mm)
        self.results["input"] = self.input
        self.view_index = self.input["view_names"].index(args.view)
        self.panel = self.panels[self.view_index]
        self.wp.init()
        self.stream = self.torch.cuda.current_stream()
        self.wp_stream = self.wp.stream_from_torch(self.stream)
        self.monitor = NvmlMonitor(base, args)
        self.results["environment"]["nvml_before"] = self.monitor.sample()
        started = time.perf_counter()
        with self.wp.ScopedStream(self.wp_stream):
            self.field = self.wp.array(self.mu.ravel(), dtype=self.wp.float32, device="cuda:0")
            self.pose = self.wp.array(self.input["pose"], dtype=self.wp.float64, device="cuda:0")
        self.stream.synchronize()
        self.results["preparation"]["device_upload_seconds"] = time.perf_counter() - started
        self.field_torch = self.wp.to_torch(self.field).reshape(self.grid.shape)
        rows, columns = self.panel.shape
        self.region = self.np.zeros((rows, columns), dtype=self.np.float64)
        self.region[rows * 5 // 16 : rows * 11 // 16, columns * 5 // 16 : columns * 11 // 16] = 1.0
        self.region /= self.region.sum()
        self.save()

    @staticmethod
    def resize_panel(panel: Any, shape: tuple[int, int]) -> Any:
        if panel.shape == shape:
            return panel
        rows, columns = shape
        old_rows, old_columns = panel.shape
        old_du, old_dv = panel.spacing_mm
        centre = tuple(
            panel.origin_mm[i]
            + (old_columns - 1) * old_du * panel.u[i] / 2
            + (old_rows - 1) * old_dv * panel.v[i] / 2
            for i in range(3)
        )
        du, dv = old_columns * old_du / columns, old_rows * old_dv / rows
        origin = tuple(
            centre[i] - (columns - 1) * du * panel.u[i] / 2 - (rows - 1) * dv * panel.v[i] / 2
            for i in range(3)
        )
        return replace(panel, origin_mm=origin, spacing_mm=(du, dv), shape=shape)

    def mode_spec(self, mode: str, *, pose: bool = False, volume: bool = False) -> Any:
        arithmetic = "float32" if mode == "fp32" else "float64"
        return self.base.ProjectionSpec(
            1024,
            precision=arithmetic,
            sampling_precision=arithmetic,
            active_pose=pose,
            active_volume=volume,
        )

    def dtype(self, mode: str) -> Any:
        return self.wp.float32 if mode == "fp32" else self.wp.float64

    def metadata(self, mode: str, *, views: Any = None, gradients: Any = ()) -> dict[str, Any]:
        return {
            "precision": mode,
            "field_storage": "float32",
            "sample_arithmetic": mode,
            "geometry": [asdict(p) for p in (self.panels if views is None else views)],
            "samples_per_ray": 1024,
            "sampling_rule": "clipped midpoint",
            "requested_gradients": list(gradients),
            "objective": "central detector-region mean",
        }

    def native_forward(self, mode: str) -> tuple[Any, Any, Any]:
        workspaces = [
            self.base.prepare_projection(
                self.grid,
                panel,
                self.mode_spec(mode),
                device="cuda:0",
                stream=self.wp_stream,
            )
            for panel in self.panels
        ]
        with self.wp.ScopedStream(self.wp_stream):
            outputs = [
                self.wp.empty(p.pixels, dtype=self.dtype(mode), device="cuda:0")
                for p in self.panels
            ]

        def forward() -> None:
            with self.wp.ScopedStream(self.wp_stream):
                for workspace, output in zip(workspaces, outputs, strict=True):
                    self.base.project_optical_depth(
                        self.field,
                        self.pose,
                        workspace=workspace,
                        out_L=output,
                        validate=False,
                        stream=self.wp_stream,
                    )

        def check() -> None:
            for workspace in workspaces:
                workspace.check_status()

        for workspace in workspaces:
            workspace.validate_inputs(self.field, self.pose, stream=self.wp_stream)
        return forward, check, (workspaces, outputs)

    def forward(self) -> None:
        from dpt import project
        from dpt.geometry import BatchedGeometry
        from dpt.torch import Projector

        for mode in self.args.modes:
            started = time.perf_counter()
            operation, check, (workspaces, outputs) = self.native_forward(mode)
            self.stream.synchronize()
            self.results["preparation"][mode + "_native_forward_seconds"] = (
                time.perf_counter() - started
            )
            metadata = {
                **self.metadata(mode),
                "workspace_bytes": sum(w.scratch_bytes for w in workspaces),
            }
            if "resident" in self.args.forward_boundaries:
                self.measure(
                    mode + "/resident_forward_3views",
                    operation,
                    {
                        **metadata,
                        "boundary": (
                            "preallocated resident field/output; final status outside timing"
                        ),
                    },
                    check=check,
                )
            else:
                operation()
                self.stream.synchronize()
                check()
            image = self.np.stack([x.numpy().reshape(self.panel.shape) for x in outputs])
            self.np.save(self.args.output / f"{mode}_depth.npy", image, allow_pickle=False)
            module = Projector(
                grid=self.grid,
                geometry=BatchedGeometry(tuple(self.panels)),
                precision=mode,
                samples_per_ray=1024,
            )
            if "public" in self.args.forward_boundaries:
                self.measure(
                    mode + "/public_projector_3views",
                    lambda: module(self.field_torch),
                    {
                        **metadata,
                        "boundary": (
                            "resident input tensor; prepared module, allocation and "
                            "validation included"
                        ),
                    },
                )
            if "completed" in self.args.forward_boundaries:
                self.measure(
                    mode + "/completed_user_call_3views",
                    lambda: (
                        project(
                            self.attenuation,
                            BatchedGeometry(tuple(self.panels)),
                            device="cuda:0",
                            precision=mode,
                            samples_per_ray=1024,
                        )
                        .cpu()
                        .numpy()
                    ),
                    {
                        **metadata,
                        "boundary": (
                            "loaded host attenuation to completed NumPy image; upload, "
                            "preparation, validation, allocation and download included"
                        ),
                    },
                )
            module = operation = check = workspaces = outputs = None
            gc.collect()

    def derivatives(self) -> None:
        wp = self.wp
        for mode in self.args.modes:
            workspace = self.base.prepare_projection(
                self.grid,
                self.panel,
                self.mode_spec(mode, pose=True, volume=True),
                device="cuda:0",
                stream=self.wp_stream,
            )
            precision = "float32" if mode == "fp32" else "float64"
            with wp.ScopedStream(self.wp_stream):
                seed = wp.array(self.region.ravel(), dtype=self.dtype(mode), device="cuda:0")
                gradient = wp.empty(self.grid.voxels, dtype=self.dtype(mode), device="cuda:0")
                pose_gradient = wp.empty(6, dtype=wp.float64, device="cuda:0")
                depth = wp.empty(self.panel.pixels, dtype=self.dtype(mode), device="cuda:0")
            self.base.project_optical_depth(
                self.field,
                self.pose,
                workspace=workspace,
                out_L=depth,
                validate=True,
                stream=self.wp_stream,
            )

            def volume() -> None:
                with wp.ScopedStream(self.wp_stream):
                    gradient.zero_()
                self.base.projection_vjp(
                    self.field,
                    self.pose,
                    cotangent=seed,
                    workspace=workspace,
                    out_mu=gradient,
                    validate=False,
                    accumulate=True,
                    check_volume_gradient=False,
                    volume_gradient_precision=precision,
                    stream=self.wp_stream,
                )

            def check_volume() -> None:
                workspace.enqueue_volume_gradient_check(
                    gradient, volume_gradient_precision=precision, stream=self.wp_stream
                )
                workspace.check_status()

            def pose() -> None:
                self.base.projection_vjp(
                    self.field,
                    self.pose,
                    cotangent=seed,
                    workspace=workspace,
                    out_pose_tangent=pose_gradient,
                    validate=False,
                    accumulate=False,
                    stream=self.wp_stream,
                )

            meta = {
                **self.metadata(mode, views=(self.panel,)),
                "spec": asdict(workspace.spec),
                "workspace_bytes": workspace.scratch_bytes,
                "view": self.args.view,
            }
            self.measure(
                mode + "/resident_volume_vjp_" + self.args.view,
                volume,
                {
                    **meta,
                    "requested_gradients": ["volume"],
                    "gradient_storage": precision,
                    "boundary": (
                        "resident buffers; zero destination included; finite check outside timing"
                    ),
                },
                check=check_volume,
            )
            self.measure(
                mode + "/resident_pose_vjp_" + self.args.view,
                pose,
                {
                    **meta,
                    "requested_gradients": ["pose"],
                    "pose_gradient_storage": "float64",
                    "boundary": (
                        "resident buffers and six-coordinate reduction; status outside timing"
                    ),
                },
                check=workspace.check_status,
            )
            host_gradient = gradient.numpy().reshape(self.grid.shape)
            actual_seed = seed.numpy().reshape(self.panel.shape).astype(self.np.float64)
            left = float(
                self.np.sum(
                    depth.numpy().reshape(self.panel.shape).astype(self.np.float64) * actual_seed
                )
            )
            right = float(self.np.sum(self.mu.astype(self.np.float64) * host_gradient))
            discrepancy = abs(left - right) / max(abs(left), abs(right), 1e-300)
            self.results["checks"][mode + "_native_adjoint"] = {
                "forward": left,
                "reverse": right,
                "relative_discrepancy": discrepancy,
                "threshold": ACCEPTANCE[mode]["adjoint_relative"],
                "passed": discrepancy <= ACCEPTANCE[mode]["adjoint_relative"],
                "pose_gradient": pose_gradient.numpy().tolist(),
            }
            self.np.save(
                self.args.output / f"{mode}_{self.args.view}_sensitivity.npy",
                host_gradient,
                allow_pickle=False,
            )
            self.save()
            if discrepancy > ACCEPTANCE[mode]["adjoint_relative"]:
                raise AssertionError(f"{mode} native real-CT adjoint discrepancy {discrepancy}")
            workspace = seed = gradient = pose_gradient = depth = host_gradient = None
            gc.collect()

    def torch_iterations(self) -> None:
        from dpt.torch import Projector

        torch = self.torch
        rows, columns = self.panel.shape
        region = (
            0,
            slice(rows * 5 // 16, rows * 11 // 16),
            slice(columns * 5 // 16, columns * 11 // 16),
        )
        for mode in self.args.modes:
            for active in self.args.torch_gradients:
                active_volume, active_pose = active != "pose", active != "volume"
                field = self.field_torch.detach().requires_grad_(active_volume)
                tangent = torch.zeros(
                    6, dtype=torch.float64, device="cuda:0", requires_grad=active_pose
                )
                projector = Projector(
                    grid=self.grid,
                    geometry=self.panel,
                    precision=mode,
                    samples_per_ray=1024,
                )
                last = {}

                def iteration() -> None:
                    field.grad, tangent.grad = None, None
                    image = projector(field, pose=tangent if active_pose else None)
                    loss = image[region].mean()
                    loss.backward()
                    last["image"], last["loss"] = image.detach(), loss.detach()

                self.measure(
                    mode + "/torch_forward_backward_" + active + "_" + self.args.view,
                    iteration,
                    {
                        **self.metadata(mode, views=(self.panel,)),
                        "requested_gradients": [
                            name
                            for name, flag in (
                                ("volume", active_volume),
                                ("pose", active_pose),
                            )
                            if flag
                        ],
                        "boundary": (
                            "ordinary Projector, region mean, complete backward with "
                            "validation and allocations"
                        ),
                        "gradient_reset": "both leaf gradients set to None on every sample",
                        "volume_gradient_storage": "float32" if active_volume else None,
                        "pose_chart": "FP64 SE3 exponential included for active pose",
                    },
                )
                self.results["checks"][mode + "_torch_" + active] = {
                    "objective": float(last["loss"].cpu()),
                    "pose_gradient": tangent.grad.cpu().tolist() if active_pose else None,
                    "volume_gradient_norm": float(
                        torch.linalg.vector_norm(field.grad.double()).cpu()
                    )
                    if active_volume
                    else None,
                }
                self.save()
                field = tangent = projector = last = None
                gc.collect()

    def batched(self) -> None:
        from dpt.batched_projection import (
            prepare_batched_projection,
            project_batched_optical_depth,
        )
        from dpt.geometry import ConeBeamIntrinsics

        wp = self.wp
        centre = self.grid.grid_to_object(tuple((n - 1) / 2 for n in reversed(self.grid.shape)))
        acquisition = ConeBeamIntrinsics(
            self.input["source_object_distance_mm"],
            self.input["source_detector_distance_mm"],
            (512, 512),
            tuple(self.input["detector_pitch_column_row_mm"]),
            centre_mm=centre,
        )
        for mode in self.args.modes:
            for count in self.args.batch_views:
                angles = tuple(math.pi - 2 * math.pi * i / count for i in range(count))
                geometry = acquisition.batch(angles)
                started = time.perf_counter()
                workspace = prepare_batched_projection(
                    self.grid,
                    geometry,
                    self.mode_spec(mode),
                    device="cuda:0",
                    stream=self.wp_stream,
                    max_bytes=self.args.max_batched_bytes,
                )
                with wp.ScopedStream(self.wp_stream):
                    poses = wp.array(
                        [self.input["pose"]] * count, dtype=wp.float64, device="cuda:0"
                    )
                    output = wp.empty(geometry.shape, dtype=self.dtype(mode), device="cuda:0")
                self.stream.synchronize()
                preparation = time.perf_counter() - started
                self.results["preparation"][f"{mode}_batched_{count}_seconds"] = preparation

                def forward(validate: bool = False) -> None:
                    with wp.ScopedStream(self.wp_stream):
                        project_batched_optical_depth(
                            self.field,
                            poses,
                            workspace=workspace,
                            out_L=output,
                            validate=validate,
                            stream=self.wp_stream,
                        )

                forward(True)
                for boundary in ("resident", "checked"):
                    self.measure(
                        f"{mode}/batched_{count}views_{boundary}",
                        (lambda: forward(False))
                        if boundary == "resident"
                        else (lambda: forward(True)),
                        {
                            **self.metadata(mode, views=geometry.views),
                            "views": count,
                            "angles_radians": angles,
                            "workspace_bytes": workspace.scratch_bytes,
                            "preparation_seconds": preparation,
                            "max_bytes_limit": self.args.max_batched_bytes,
                            "boundary": "resident buffers with current per-call ray refresh; "
                            + boundary,
                        },
                        check=workspace.check_status,
                    )
                self.np.save(
                    self.args.output / f"{mode}_batched_{count}_first_ap.npy",
                    wp.to_torch(output)[0].cpu().numpy(),
                    allow_pickle=False,
                )
                if count == 16:
                    self.np.save(
                        self.args.output / f"{mode}_batched_16_depth.npy",
                        wp.to_torch(output).cpu().numpy(),
                        allow_pickle=False,
                    )
                workspace = poses = output = None
                gc.collect()

    def correctness(self) -> None:
        from dpt.geometry import RigidTransform, compose_pose
        from dpt.torch import Projector

        np, torch = self.np, self.torch
        (
            reference_forward,
            reference_check,
            (_reference_workspaces, reference_outputs),
        ) = self.native_forward("fp64")
        reference_forward()
        self.stream.synchronize()
        reference_check()
        reference = np.stack([x.numpy().reshape(self.panel.shape) for x in reference_outputs])
        np.save(self.args.output / "fp64_reference_depth.npy", reference, allow_pickle=False)
        baseline_file = self.args.baseline_results.parent / "float64_depth.npy"
        if (
            self.args.anatomy == "chest"
            and self.args.shape == (512, 512)
            and baseline_file.exists()
        ):
            baseline_error = self.error(reference, np.load(baseline_file, allow_pickle=False))
            baseline_passed = all(
                baseline_error[key] <= ACCEPTANCE["fp64"]["image_" + key]
                for key in ("max_abs", "rmse", "relative_l2")
            )
            self.results["checks"]["fp64_vs_frozen_baseline"] = {
                **baseline_error,
                "passed": baseline_passed,
            }
            self.save()
            if not baseline_passed:
                raise AssertionError("Candidate FP64 disagrees with the frozen chest baseline")
        else:
            self.results["checks"]["fp64_vs_frozen_baseline"] = {
                "status": "not measured",
                "reason": "different anatomy/shape or frozen baseline array is unavailable",
            }
        # The fixed check pose avoids placing the entire acquisition at an exact
        # coordinate-axis or voxel symmetry. Individual CT sample knots remain
        # piecewise differentiable; report all FD steps instead of selecting one.
        anchor = compose_pose(RigidTransform(), (0.137, -0.219, 0.083, 0.0017, -0.0023, 0.0009))
        direction = torch.tensor(
            [0.31, -0.27, 0.19, 0.0007, -0.0011, 0.0009],
            dtype=torch.float64,
            device="cuda:0",
        )
        rows, columns = self.panel.shape
        region = (
            0,
            slice(rows * 5 // 16, rows * 11 // 16),
            slice(columns * 5 // 16, columns * 11 // 16),
        )
        # Smooth, nonuniform field direction; zero attenuation remains zero.
        coordinate = torch.linspace(
            -math.pi, math.pi, self.grid.shape[2], device="cuda:0", dtype=torch.float32
        )
        volume_direction = self.field_torch * (0.25 + 0.75 * torch.sin(coordinate))[None, None, :]
        h_volume, h_pose = 0.01, 0.02
        for mode in self.args.modes:
            forward, check, (_, outputs) = self.native_forward(mode)
            forward()
            self.stream.synchronize()
            check()
            image = np.stack([x.numpy().reshape(self.panel.shape) for x in outputs])
            image_error = self.error(image, reference)
            np.save(self.args.output / f"{mode}_depth.npy", image, allow_pickle=False)
            threshold = ACCEPTANCE[mode]
            image_passed = all(
                image_error[key] <= threshold["image_" + key]
                for key in ("max_abs", "rmse", "relative_l2")
            )
            module = Projector(
                grid=self.grid,
                geometry=self.panel,
                precision=mode,
                samples_per_ray=1024,
                anchor=anchor,
            )
            field = self.field_torch.detach().requires_grad_(True)
            pose = torch.zeros(6, dtype=torch.float64, device="cuda:0", requires_grad=True)
            prediction = module(field, pose=pose)
            objective = prediction[region].double().mean()
            volume_gradient, pose_gradient = torch.autograd.grad(objective, (field, pose))
            plus, minus = (
                (field.detach() + h_volume * volume_direction).contiguous(),
                (field.detach() - h_volume * volume_direction).contiguous(),
            )
            # Derivative against the realised FP32 input perturbation, avoiding
            # a false discrepancy caused solely by its storage rounding.
            realised = (plus.double() - minus.double()) / (2 * h_volume)
            volume_ad = float((volume_gradient.double() * realised).sum().cpu())
            pose_ad = float((pose_gradient * direction).sum().cpu())
            with torch.no_grad():
                volume_fd = float(
                    (
                        (
                            module(plus, pose=pose)[region].double().mean()
                            - module(minus, pose=pose)[region].double().mean()
                        )
                        / (2 * h_volume)
                    ).cpu()
                )
                pose_fds = []
                for step in (0.04, 0.02, 0.01):
                    positive = module(field.detach(), pose=step * direction)[region].double().mean()
                    negative = (
                        module(field.detach(), pose=-step * direction)[region].double().mean()
                    )
                    pose_fds.append(
                        {
                            "step": step,
                            "directional_derivative": float(
                                ((positive - negative) / (2 * step)).cpu()
                            ),
                        }
                    )
            pose_fd = next(
                row["directional_derivative"] for row in pose_fds if row["step"] == h_pose
            )

            def agreement(actual: float, finite_difference: float, name: str) -> dict[str, Any]:
                absolute = abs(actual - finite_difference)
                scale = max(abs(actual), abs(finite_difference))
                bound = (
                    threshold[name + "_directional_atol"]
                    + threshold[name + "_directional_rtol"] * scale
                )
                return {
                    "autograd": actual,
                    "finite_difference": finite_difference,
                    "absolute_error": absolute,
                    "relative_error": absolute / max(scale, 1e-300),
                    "allowed_absolute_error": bound,
                    "passed": absolute <= bound,
                }

            volume_check = agreement(volume_ad, volume_fd, "volume")
            pose_check = agreement(pose_ad, pose_fd, "pose")
            entry = {
                "image_error_vs_fp64": image_error,
                "image_passed": image_passed,
                "volume_directional": volume_check,
                "volume_step": h_volume,
                "volume_direction": (
                    "mu * (0.25 + 0.75*sin(linspace(-pi,pi,nx))); realised FP32 perturbation"
                ),
                "pose_directional": pose_check,
                "pose_fd_samples": pose_fds,
                "pose_selected_step": h_pose,
                "pose_direction": direction.cpu().tolist(),
                "pose_check_anchor": list(anchor.packed()),
                "objective": "FP64 mean of central detector region from actual projected output",
                "passed": image_passed and volume_check["passed"] and pose_check["passed"],
            }
            self.results["checks"][mode + "_real_volume"] = entry
            self.save()
            if not entry["passed"]:
                raise AssertionError(
                    f"{mode} real-volume acceptance failed; "
                    "inspect saved checks, do not loosen bounds"
                )
            module = field = pose = prediction = objective = volume_gradient = pose_gradient = (
                plus
            ) = minus = realised = outputs = None
            gc.collect()
        _reference_workspaces = reference_outputs = None

    def material(self) -> None:
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

        wp, np = self.wp, self.np
        started = time.perf_counter()
        density = np.empty(self.grid.shape, dtype=np.float32)
        for index in range(density.shape[0]):
            density[index] = np.maximum(
                1.0 + self.ct.values[index].astype(np.float64) / 1000.0, 0.0
            )
        self.results["preparation"]["water_basis_conversion_seconds"] = (
            time.perf_counter() - started
        )
        for mode in self.args.modes:
            precision = "float32" if mode == "fp32" else "float64"
            with wp.ScopedStream(self.wp_stream):
                fields = wp.array(density.reshape(1, -1), dtype=wp.float32, device="cuda:0")
                paths = wp.empty((1, self.panel.pixels), dtype=self.dtype(mode), device="cuda:0")
                path_seed = wp.empty_like(paths)
                gradients = wp.empty((1, self.grid.voxels), dtype=self.dtype(mode), device="cuda:0")
                grad_pose = wp.empty(6, dtype=wp.float64, device="cuda:0")
                coefficients = wp.array([[0.02059]], dtype=wp.float32, device="cuda:0")
                weights = wp.array([10000.0], dtype=wp.float32, device="cuda:0")
                response = wp.array([1.0], dtype=wp.float32, device="cuda:0")
                mean = wp.empty(self.panel.pixels, dtype=self.dtype(mode), device="cuda:0")
                seed = wp.array(self.region.ravel(), dtype=self.dtype(mode), device="cuda:0")
            started = time.perf_counter()
            material = prepare_material_projection(
                self.grid,
                self.panel,
                self.mode_spec(mode, pose=True, volume=True),
                materials=1,
                fields=fields,
                out_paths=paths,
                cotangent_paths=path_seed,
                out_fields=gradients,
                device="cuda:0",
                stream=self.wp_stream,
                field_domain="nonnegative",
                channel_execution="fused",
                volume_gradient_precision=precision,
            )
            spectral = prepare_spectral(
                SpectralSpec(
                    1,
                    1,
                    precision=precision,
                    output_unit="expected photons per pixel",
                    input_description=(
                        "60 keV monoenergetic water-scaled CT; ideal unit-efficiency detector"
                    ),
                ),
                max_pixels=self.panel.pixels,
                device="cuda:0",
                stream=self.wp_stream,
            )
            self.stream.synchronize()
            self.results["preparation"][mode + "_material_spectral_seconds"] = (
                time.perf_counter() - started
            )
            meta = {
                **self.metadata(mode, views=(self.panel,)),
                "execution": "fused",
                "projection_spec": asdict(material.projection.spec),
                "spectral_spec": asdict(spectral.spec),
                "workspace_bytes": material.scratch_bytes + spectral.scratch_bytes,
                "basis": (
                    "one water-relative-density field; max(1+HU/1000,0), FP64 "
                    "conversion then FP32 storage"
                ),
                "basis_limits": (
                    "algebraic conversion; no inferred tissue segmentation or scanner calibration"
                ),
                "energies_kev": [60],
                "coefficients_mm_inverse": [float(np.float32(0.02059))],
                "incident_photons_per_detector_pixel": [10000],
                "response_per_photon": [1],
                "sampled_source_histories": None,
                "gradient_storage": precision,
            }

            def projection() -> None:
                project_material_paths(
                    self.pose, workspace=material, validate=False, stream=self.wp_stream
                )

            def spectrum() -> None:
                spectral_signal(
                    paths,
                    coefficients,
                    weights,
                    response,
                    out_mean=mean,
                    workspace=spectral,
                    validate=False,
                    stream=self.wp_stream,
                )

            def spectral_reverse() -> None:
                spectral_vjp(
                    paths,
                    coefficients,
                    weights,
                    response,
                    cotangent=seed,
                    out_grad_paths=path_seed,
                    workspace=spectral,
                    validate=False,
                    stream=self.wp_stream,
                )

            def material_reverse() -> None:
                with wp.ScopedStream(self.wp_stream):
                    gradients.zero_()
                    grad_pose.zero_()
                material_projection_vjp(
                    self.pose,
                    workspace=material,
                    out_pose_tangent=grad_pose,
                    validate=False,
                    accumulate=True,
                    check_volume_gradient=False,
                    stream=self.wp_stream,
                )

            def complete() -> None:
                projection()
                spectrum()
                spectral_reverse()
                material_reverse()

            def check() -> None:
                material.check_status()
                spectral.check_status()

            def check_gradient() -> None:
                material.projection.enqueue_volume_gradient_check(
                    gradients.flatten(),
                    volume_gradient_precision=precision,
                    stream=self.wp_stream,
                )
                check()

            project_material_paths(
                self.pose, workspace=material, validate=True, stream=self.wp_stream
            )
            spectral_signal(
                paths,
                coefficients,
                weights,
                response,
                out_mean=mean,
                workspace=spectral,
                validate=True,
                stream=self.wp_stream,
            )
            spectral_reverse()
            for name, call, active in (
                ("projection", projection, []),
                ("spectral", spectrum, []),
                ("spectral_vjp", spectral_reverse, ["paths"]),
                ("material_vjp", material_reverse, ["volume", "pose"]),
                ("complete_iteration", complete, ["volume", "pose"]),
            ):
                self.measure(
                    f"{mode}/material_fused_{name}",
                    call,
                    {
                        **meta,
                        "requested_gradients": active,
                        "boundary": (
                            "resident buffers; derivative reset included; finite/status "
                            "checks outside timing"
                        ),
                    },
                    check=check_gradient if active == ["volume", "pose"] else check,
                )
            host_paths, host_mean = paths.numpy(), mean.numpy()
            expected = 10000.0 * np.exp(
                -float(np.float32(0.02059)) * host_paths[0].astype(np.float64)
            )
            self.results["checks"][mode + "_material_spectral"] = {
                "spectral_error_vs_numpy": self.error(host_mean, expected),
                "pose_tangent_gradient": grad_pose.numpy().tolist(),
            }
            np.testing.assert_allclose(
                host_mean,
                expected,
                rtol=2e-6 if mode == "fp32" else 2e-12,
                atol=2e-3 if mode == "fp32" else 1e-12,
            )
            # Check the realised material transpose on the same native field,
            # region cotangent, physical AP rays and 1024 midpoint rule.
            host_gradient = gradients.numpy().reshape(self.grid.shape)
            host_cotangent = path_seed.numpy().astype(np.float64)
            lhs = float(np.sum(host_paths.astype(np.float64) * host_cotangent))
            rhs = 0.0
            for index in range(density.shape[0]):
                rhs += float(np.sum(density[index].astype(np.float64) * host_gradient[index]))
            discrepancy = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
            thresholds = MATERIAL_ACCEPTANCE[mode]
            adjoint_check = {
                "path_inner_product": lhs,
                "field_inner_product": rhs,
                "relative_discrepancy": discrepancy,
                "threshold": thresholds["adjoint_relative"],
                "passed": discrepancy <= thresholds["adjoint_relative"],
                "cotangent": "spectral VJP of the recorded AP detector-region photon mean",
                "pose": self.pose.numpy().tolist(),
            }

            # A small fixed step controls the nonlinearity of exp(-mu * path).
            # Construct and reduce in slabs to avoid full-volume FP64 copies.
            step = 0.001
            coordinate = np.linspace(-math.pi, math.pi, density.shape[2], dtype=np.float64)
            factor = (0.25 + 0.75 * np.sin(coordinate))[None, :]
            positive, negative = np.empty_like(density), np.empty_like(density)
            directional_vjp = 0.0
            for index in range(density.shape[0]):
                original = density[index].astype(np.float64)
                direction = original * factor
                positive[index] = original + step * direction
                negative[index] = original - step * direction
                realised = (
                    positive[index].astype(np.float64) - negative[index].astype(np.float64)
                ) / (2 * step)
                directional_vjp += float(np.sum(host_gradient[index] * realised))
            if np.min(positive) < 0 or np.min(negative) < 0:
                raise AssertionError(
                    "Material FD perturbed the nonnegative field outside its domain"
                )
            objectives = []
            for perturbation in (positive, negative):
                with wp.ScopedStream(self.wp_stream):
                    fields.assign(perturbation.reshape(1, -1))
                projection()
                spectrum()
                self.stream.synchronize()
                check()
                objectives.append(
                    float(np.sum(mean.numpy().astype(np.float64) * self.region.ravel()))
                )
            finite_difference = (objectives[0] - objectives[1]) / (2 * step)
            absolute = abs(directional_vjp - finite_difference)
            scale = max(abs(directional_vjp), abs(finite_difference))
            bound = thresholds["volume_atol"] + thresholds["volume_rtol"] * scale
            volume_check = {
                "vjp": directional_vjp,
                "finite_difference": finite_difference,
                "absolute_error": absolute,
                "relative_error": absolute / max(scale, 1e-300),
                "allowed_absolute_error": bound,
                "passed": absolute <= bound,
                "step": step,
                "direction": (
                    "density * (0.25 + 0.75*sin(linspace(-pi,pi,nx))); "
                    "realised FP32 input perturbation"
                ),
                "objective": "FP64 reduction of actual spectral output with the region cotangent",
                "objective_units": "expected photons per detector pixel",
                "positive_objective": objectives[0],
                "negative_objective": objectives[1],
            }
            entry = self.results["checks"][mode + "_material_spectral"]
            entry["path_adjoint"] = adjoint_check
            entry["volume_directional"] = volume_check
            entry["passed"] = adjoint_check["passed"] and volume_check["passed"]
            self.save()
            if not entry["passed"]:
                raise AssertionError(
                    f"{mode} real material derivative acceptance failed; inspect saved checks"
                )
            with wp.ScopedStream(self.wp_stream):
                fields.assign(density.reshape(1, -1))
            self.stream.synchronize()
            positive = negative = host_gradient = host_cotangent = None
            np.save(
                self.args.output / f"{mode}_water_paths_mm.npy",
                host_paths.reshape(self.panel.shape),
                allow_pickle=False,
            )
            np.save(
                self.args.output / f"{mode}_expected_photons.npy",
                host_mean.reshape(self.panel.shape),
                allow_pickle=False,
            )
            self.save()
            material = spectral = fields = paths = path_seed = gradients = grad_pose = mean = (
                seed
            ) = None
            gc.collect()


def main() -> None:
    args = arguments()
    args.source = args.source.resolve()
    sys.path.insert(0, str(args.source / "python"))
    base = load_runtime()

    benchmark = None
    try:
        benchmark = ReleaseBenchmark(args, base)
        for case in args.cases:
            print("Starting", case, base.utc(), "output", args.output, flush=True)
            {
                "checks": benchmark.correctness,
                "forward": benchmark.forward,
                "derivatives": benchmark.derivatives,
                "torch": benchmark.torch_iterations,
                "batched": benchmark.batched,
                "material": benchmark.material,
            }[case]()
            gc.collect()
        ending_digest = base.digest_sources(args.source)
        benchmark.results["source_unchanged_during_measurement"] = (
            ending_digest["sha256"] == benchmark.results["source"]["sha256"]
        )
        if not benchmark.results["source_unchanged_during_measurement"]:
            raise RuntimeError(
                "Timed source changed during this process; this run cannot be a release headline"
            )
        benchmark.results["measurement_finished_at"] = base.utc()
        benchmark.results["environment"]["nvidia_smi_after"] = base.command("nvidia-smi", "-q")
        benchmark.results["environment"]["nvml_after"] = benchmark.monitor.sample()
        benchmark.save()
        print("RESULTS", args.output / "results.json", flush=True)
    except BaseException as error:
        if benchmark is not None:
            benchmark.results["failure"] = {
                "measured_at": base.utc(),
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            benchmark.save()
        raise
    finally:
        if benchmark is not None:
            benchmark.monitor.close()


if __name__ == "__main__":
    main()
