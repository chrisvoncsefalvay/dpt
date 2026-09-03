# ruff: noqa: B023
# Each timed closure completes within its workload loop.
"""Additional native-CT reference timings; run selected cases serially on one GPU.

Example:
  python baseline_extra.py --ct chest.nii.gz --source /path/to/source \
    --baseline-results /path/to/baseline/results.json --output /path/to/extra \
    --cases derivatives torch

The existing reference adapter retains its host transfers and per-view status
readbacks. No timing from that boundary is labelled a resident-kernel timing.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmarks.common import Measurements, NvmlMonitor


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        choices=("derivatives", "torch", "batched", "reference", "material"),
    )
    parser.add_argument("--view", choices=("ap", "lateral", "oblique"), default="ap")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--long-repeats", type=int, default=10)
    parser.add_argument("--long-threshold-seconds", type=float, default=0.2)
    parser.add_argument(
        "--torch-gradients",
        nargs="+",
        default=["volume", "pose", "both"],
        choices=("volume", "pose", "both"),
    )
    parser.add_argument("--batch-views", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--max-batched-bytes", type=int, default=8 * 1024**3)
    parser.add_argument(
        "--batched-boundaries",
        nargs="+",
        default=["resident", "checked"],
        choices=("resident", "checked", "completed"),
    )
    parser.add_argument(
        "--reference-operations",
        nargs="+",
        default=["forward", "backward"],
        choices=("forward", "backward"),
    )
    parser.add_argument(
        "--material-execution", nargs="+", default=["serial", "fused"], choices=("serial", "fused")
    )
    parser.add_argument("--nvml-interval-seconds", type=float, default=0.05)
    parser.add_argument(
        "--nvml-device-index",
        type=int,
        help="Explicit physical NVML index if CUDA UUID/PCI mapping is unavailable.",
    )
    args = parser.parse_args()
    if min(args.rounds, args.repeats, args.long_repeats, *args.batch_views) < 1:
        parser.error("rounds, repeats and view counts must be positive")
    if args.long_threshold_seconds <= 0 or args.nvml_interval_seconds <= 0:
        parser.error("time intervals must be positive")
    if args.max_batched_bytes < 1:
        parser.error("max-batched-bytes must be positive")
    return args


class Benchmark(Measurements):
    def __init__(self, args: argparse.Namespace, base: Any):
        self.args, self.base = args, base
        self.np, self.torch, self.wp = base.np, base.torch, base.wp
        self.baseline = json.loads(args.baseline_results.read_text())
        self.input = self.baseline["input"]
        source = base.digest_sources(args.source)
        if source["sha256"] != self.baseline["source"]["sha256"]:
            raise ValueError(
                "DPT source digest differs from baseline; use the reviewed baseline checkout"
            )
        if self.input["samples_per_ray"] != 1024:
            raise ValueError(
                "This driver requires the requested baseline of 1024 clipped midpoints"
            )
        if self.input["resampled"]:
            raise ValueError("Native-volume baseline required")
        if self.input["energy_kev"] != 60 or self.input["mu_water_mm_inverse"] != 0.02059:
            raise ValueError("Baseline must use the declared 60 keV water-scaled model")
        self.results: dict[str, Any] = {
            "schema_version": 1,
            "measurement_started_at": base.utc(),
            "baseline_results": str(args.baseline_results.resolve()),
            "baseline_results_sha256": hashlib.sha256(
                args.baseline_results.read_bytes()
            ).hexdigest(),
            "source": source,
            "source_revision": base.command("git", "-C", str(args.source), "rev-parse", "HEAD"),
            "timed_harness_sources": {},
            "input": self.input,
            "loaded_ct_file": str(args.ct.resolve()),
            "environment": {
                "platform": base.platform.platform(),
                "python": base.platform.python_version(),
                "versions": {
                    name: base.importlib.metadata.version(name)
                    for name in ("numpy", "torch", "warp-lang", "nibabel", "nvidia-ml-py")
                },
                "torch_cuda": self.torch.version.cuda,
                "gpu": self.torch.cuda.get_device_name(0),
                "nvidia_smi_before": base.command("nvidia-smi", "-q"),
            },
            "timing_policy": {
                "rounds": args.rounds,
                "short_repeats": args.repeats,
                "long_repeats": min(args.repeats, args.long_repeats),
                "long_threshold_seconds": args.long_threshold_seconds,
                "warmup": "one first use and one completed pilot of each exact operation",
                "events": "CUDA events on each operation's actual owning Warp/Torch stream",
                "wall": "completed call including requested transfers and host work",
                "event_limit": "event intervals include idle gaps between host-issued kernels",
                "logging": "NVML samples in background; file writes outside timed regions",
                "release_headlines": (
                    "assess recorded utilisation/process activity; no quiet claim inferred"
                ),
            },
            "measurements": [],
            "preparation": {},
            "checks": {},
        }
        for path in (
            Path(__file__),
            Path(base.__file__),
            Path(__file__).with_name("common.py"),
            args.source / "benchmarks/reference.py",
        ):
            self.results["timed_harness_sources"][str(path)] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output / "results.json").exists():
            raise FileExistsError(f"Refusing to replace {args.output / 'results.json'}")
        self.save()
        started = time.perf_counter()
        self.ct = base.read_nifti(args.ct, value_units="HU")
        self.results["preparation"]["file_loading_seconds"] = time.perf_counter() - started
        digest = hashlib.sha256(args.ct.read_bytes()).hexdigest()
        if digest != self.input["sha256"]:
            raise ValueError("CT checksum differs from the original baseline")
        if json.loads(json.dumps(asdict(self.ct.grid))) != self.input["grid"]:
            raise ValueError("CT physical grid differs from the original baseline")
        if list(self.ct.shape) != self.input["native_shape_zyx"]:
            raise ValueError("CT native shape differs from the original baseline")
        started = time.perf_counter()
        self.mu = (self.np.maximum(1.0 + self.ct.values / 1000.0, 0.0) * 0.02059).astype(
            self.np.float32
        )
        self.results["preparation"]["ct_conversion_seconds"] = time.perf_counter() - started
        from dpt.geometry import DetectorGeometry

        self.panels = [
            DetectorGeometry(
                **{
                    key: tuple(value) if isinstance(value, list) else value
                    for key, value in entry.items()
                }
            )
            for entry in self.input["geometry"]
        ]
        self.view_index = self.input["view_names"].index(args.view)
        self.panel = self.panels[self.view_index]
        self.grid = self.ct.grid
        self.wp.init()
        self.stream = self.torch.cuda.current_stream()
        self.wp_stream = self.wp.stream_from_torch(self.stream)
        self.monitor = NvmlMonitor(base, args)
        self.results["environment"]["nvml_before"] = self.monitor.sample()
        with self.wp.ScopedStream(self.wp_stream):
            started = time.perf_counter()
            self.field = self.wp.array(self.mu.ravel(), dtype=self.wp.float32, device="cuda:0")
            self.pose = self.wp.array(self.input["pose"], dtype=self.wp.float64, device="cuda:0")
            self.stream.synchronize()
            self.results["preparation"]["device_upload_seconds"] = time.perf_counter() - started
        rows, columns = self.panel.shape
        self.region = self.np.zeros((rows, columns), dtype=self.np.float64)
        self.region[rows * 5 // 16 : rows * 11 // 16, columns * 5 // 16 : columns * 11 // 16] = 1.0
        self.region /= self.region.sum()
        with self.wp.ScopedStream(self.wp_stream):
            self.seed = self.wp.array(self.region.ravel(), dtype=self.wp.float64, device="cuda:0")
        self.reference_path = args.baseline_results.parent / "float64_depth.npy"
        self.reference = (
            self.np.load(self.reference_path, mmap_mode="r")
            if self.reference_path.exists()
            else None
        )
        self.save()

    def spec(self, *, pose: bool, volume: bool, precision: str = "float64") -> Any:
        return self.base.ProjectionSpec(
            1024,
            precision=precision,
            active_pose=pose,
            active_volume=volume,
            sampling_precision="float64",
        )

    def workspace(self, *, pose: bool, volume: bool) -> Any:
        return self.base.prepare_projection(
            self.grid,
            self.panel,
            self.spec(pose=pose, volume=volume),
            device="cuda:0",
            stream=self.wp_stream,
        )

    def derivatives(self) -> None:
        wp = self.wp
        workspace = self.workspace(pose=True, volume=True)
        with wp.ScopedStream(self.wp_stream):
            depth = wp.empty(self.panel.pixels, dtype=wp.float64, device="cuda:0")
            volume_gradient = wp.empty(self.grid.voxels, dtype=wp.float64, device="cuda:0")
            pose_gradient = wp.empty(6, dtype=wp.float64, device="cuda:0")
        self.base.project_optical_depth(
            self.field,
            self.pose,
            workspace=workspace,
            out_L=depth,
            validate=True,
            stream=self.wp_stream,
        )
        metadata = {
            "spec": asdict(workspace.spec),
            "view": self.args.view,
            "workspace_bytes": workspace.scratch_bytes,
            "cotangent": "central detector-region mean",
            "field_dtype": "float32",
            "depth_dtype": "float64",
            "volume_gradient_dtype": "float64",
            "pose_gradient": "six local-right tangent components, mm and radians",
        }

        def volume() -> None:
            with wp.ScopedStream(self.wp_stream):
                volume_gradient.zero_()
            self.base.projection_vjp(
                self.field,
                self.pose,
                cotangent=self.seed,
                workspace=workspace,
                out_mu=volume_gradient,
                validate=False,
                accumulate=True,
                volume_gradient_precision="float64",
                check_volume_gradient=False,
                stream=self.wp_stream,
            )

        def check_volume() -> None:
            workspace.enqueue_volume_gradient_check(
                volume_gradient, volume_gradient_precision="float64", stream=self.wp_stream
            )
            workspace.check_status()

        def pose() -> None:
            self.base.projection_vjp(
                self.field,
                self.pose,
                cotangent=self.seed,
                workspace=workspace,
                out_pose_tangent=pose_gradient,
                validate=False,
                accumulate=False,
                stream=self.wp_stream,
            )

        self.measure(
            "strict/resident_volume_vjp_" + self.args.view,
            volume,
            {
                **metadata,
                "requested_gradients": ["volume"],
                "boundary": (
                    "preallocated buffers; destination zero included; per-sample "
                    "final finite check outside timing"
                ),
            },
            check=check_volume,
        )
        self.measure(
            "strict/resident_pose_vjp_" + self.args.view,
            pose,
            {
                **metadata,
                "requested_gradients": ["pose"],
                "boundary": "preallocated buffers; VJP and reduction; status check outside timing",
            },
            check=workspace.check_status,
        )
        host_gradient = volume_gradient.numpy().reshape(self.grid.shape)
        host_depth = depth.numpy().reshape(self.panel.shape)
        left = float(self.np.sum(host_depth * self.region))
        right = float(self.np.sum(self.mu.astype(self.np.float64) * host_gradient))
        self.results["checks"]["strict_derivatives"] = {
            "adjoint_forward": left,
            "adjoint_reverse": right,
            "relative_adjoint_discrepancy": abs(left - right) / max(abs(left), abs(right), 1e-300),
            "pose_tangent_gradient": pose_gradient.numpy().tolist(),
        }
        self.np.save(
            self.args.output / f"strict_{self.args.view}_sensitivity.npy",
            host_gradient.astype(self.np.float32),
        )
        self.np.save(
            self.args.output / f"strict_{self.args.view}_pose_gradient.npy", pose_gradient.numpy()
        )
        self.save()

    def torch_iterations(self) -> None:
        from dpt.interop.torch import project_optical_depth
        from dpt.torch.geometry import compose_pose

        torch = self.torch
        anchor = torch.tensor(self.input["pose"], dtype=torch.float64, device="cuda:0")
        rows, columns = self.panel.shape
        region = (
            slice(rows * 5 // 16, rows * 11 // 16),
            slice(columns * 5 // 16, columns * 11 // 16),
        )
        for active in self.args.torch_gradients:
            active_volume, active_pose = active != "pose", active != "volume"
            workspace = self.workspace(pose=active_pose, volume=active_volume)
            field = torch.tensor(
                self.mu, dtype=torch.float32, device="cuda:0", requires_grad=active_volume
            )
            tangent = torch.zeros(
                6, dtype=torch.float64, device="cuda:0", requires_grad=active_pose
            )
            last: dict[str, Any] = {}

            def iteration() -> None:
                field.grad, tangent.grad = None, None
                posed = compose_pose(anchor, tangent) if active_pose else anchor
                image = project_optical_depth(field, posed, workspace=workspace, validate=True)
                loss = image[region].mean()
                loss.backward()
                last["image"], last["loss"] = image.detach(), loss.detach()

            self.measure(
                "strict/torch_forward_backward_" + active + "_" + self.args.view,
                iteration,
                {
                    "spec": asdict(workspace.spec),
                    "view": self.args.view,
                    "requested_gradients": [
                        name
                        for name, flag in (("volume", active_volume), ("pose", active_pose))
                        if flag
                    ],
                    "workspace_bytes": workspace.scratch_bytes,
                    "boundary": (
                        "current public Torch call, region mean, complete backward; "
                        "checked native calls and allocations included"
                    ),
                    "gradient_reset": "leaf gradients set to None at every iteration",
                    "pose_chart": "FP64 SE(3) matrix exponential; included for pose cases",
                    "volume_gradient_dtype": "float32" if active_volume else None,
                },
                check=workspace.check_status,
            )
            image = last["image"].cpu().numpy()
            check: dict[str, Any] = {"objective": float(last["loss"].cpu())}
            if self.reference is not None:
                check["image_error_vs_baseline_strict"] = self.error(
                    image, self.reference[self.view_index]
                )
            if active_pose:
                check["pose_tangent_gradient"] = tangent.grad.detach().cpu().numpy().tolist()
            if active_volume:
                gradient = field.grad.detach().cpu().numpy()
                check["volume_gradient_l2"] = float(
                    self.np.linalg.norm(gradient.astype(self.np.float64))
                )
                self.np.save(
                    self.args.output / f"torch_{active}_{self.args.view}_sensitivity.npy", gradient
                )
            self.results["checks"]["torch_" + active] = check
            self.save()
            field = tangent = workspace = last = None
            gc.collect()

    def batched(self) -> None:
        from dpt.batched_projection import prepare_batched_projection, project_batched_optical_depth
        from dpt.geometry import BatchedGeometry, ConeBeamIntrinsics

        wp = self.wp
        if tuple(self.panel.shape) != (512, 512):
            raise ValueError(
                "Requested batched workload requires the shared 512 by 512 baseline acquisition"
            )
        centre = self.grid.grid_to_object(tuple((n - 1) / 2 for n in reversed(self.grid.shape)))
        acquisition = ConeBeamIntrinsics(
            self.input["source_object_distance_mm"],
            self.input["source_detector_distance_mm"],
            (512, 512),
            tuple(self.input["detector_pitch_column_row_mm"]),
            centre_mm=centre,
        )
        for views in self.args.batch_views:
            angles = [math.pi - 2 * math.pi * i / views for i in range(views)]
            geometry = BatchedGeometry(tuple(acquisition.view(angle) for angle in angles))
            before = self.monitor.sample()
            started = time.perf_counter()
            workspace = prepare_batched_projection(
                self.grid,
                geometry,
                self.spec(pose=False, volume=False),
                device="cuda:0",
                stream=self.wp_stream,
                max_bytes=self.args.max_batched_bytes,
            )
            with wp.ScopedStream(self.wp_stream):
                poses = wp.array([self.input["pose"]] * views, dtype=wp.float64, device="cuda:0")
                output = wp.empty(geometry.shape, dtype=wp.float64, device="cuda:0")
            self.stream.synchronize()
            preparation = {
                "completed_seconds": time.perf_counter() - started,
                "max_bytes_limit": self.args.max_batched_bytes,
                "workspace_bytes": workspace.scratch_bytes,
                "nvml_before": before,
                "nvml_after": self.monitor.sample(),
            }
            self.results["preparation"][f"batched_{views}"] = preparation

            def forward(validate: bool = False) -> None:
                # The existing batched wrapper also invokes nested validation
                # without forwarding its stream argument. Preserve its caller
                # scope as well as passing the explicit owning stream.
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
            operations = {
                "resident": lambda: forward(False),
                "checked": lambda: forward(True),
                "completed": lambda: (forward(True), output.numpy())[1],
            }
            for boundary in self.args.batched_boundaries:
                self.measure(
                    f"strict/batched_{views}views_{boundary}",
                    operations[boundary],
                    {
                        "spec": asdict(workspace.spec),
                        "views": views,
                        "angles_radians": angles,
                        "geometry": [asdict(p) for p in geometry.views],
                        "requested_gradients": [],
                        "preparation": preparation,
                        "boundary": {
                            "resident": (
                                "device field/output; includes current per-call ray refresh"
                            ),
                            "checked": "device field/output; ray refresh and checked public call",
                            "completed": "checked public call and full NumPy output download",
                        }[boundary],
                    },
                    check=workspace.check_status,
                )
            first = wp.to_torch(output)[0].detach().cpu().numpy()
            self.np.save(self.args.output / f"batched_{views}_first_ap_depth.npy", first)
            if self.reference is not None:
                self.results["checks"][f"batched_{views}_ap"] = self.error(first, self.reference[0])
            self.save()
            workspace = poses = output = None
            gc.collect()

    def reference_adapter(self) -> None:
        path = self.args.source / "benchmarks/reference.py"
        spec = importlib.util.spec_from_file_location("dpt_baseline_reference", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        native_shape = self.grid.shape

        class NativeCTDptOperator(module.DptOperator):
            def backward(self, projection: Any) -> Any:
                # Exact original per-view body; only final native-volume reshape changes.
                with self.context.scope():
                    self.gradient.zero_()
                for index, workspace in enumerate(self.workspaces):
                    with self.context.scope():
                        self.output.assign(projection[index].reshape(-1))
                    self.module.projection_vjp(
                        self.volume,
                        self.pose,
                        adj_L=self.output,
                        out_mu=self.gradient,
                        workspace=workspace,
                        validate=False,
                        accumulate=True,
                        check_volume_gradient=False,
                    )
                    workspace.check_status()
                return self.gradient.numpy().reshape(native_shape).copy()

        if self.panel.shape[0] != self.panel.shape[1]:
            raise ValueError("Original reference adapter requires a square detector")
        # Its constructor uses 2*size solely to select N=1024 here; the actual
        # native grid and all panels are supplied directly, without resampling.
        case = SimpleNamespace(
            size=512,
            grid=self.grid,
            panels=tuple(self.panels),
            detector=self.panel.shape[0],
            views=len(self.panels),
            sampling_precision="float64",
        )
        started = time.perf_counter()
        operator = NativeCTDptOperator(case, "cuda:0")
        operator.synchronise()
        self.results["preparation"]["reference_adapter_seconds"] = time.perf_counter() - started
        adapter_stream = self.wp.stream_to_torch(operator.context.stream)
        seeds = self.np.zeros((len(self.panels), *self.panel.shape), dtype=self.np.float32)
        seeds[self.view_index] = self.region.astype(self.np.float32)
        metadata = {
            "algorithm": operator.description,
            "native_shape_zyx": list(self.grid.shape),
            "geometry": [asdict(p) for p in self.panels],
            "samples_per_ray": 1024,
            "resident_bytes": operator.resident_bytes,
            "input_dtype": "float32",
            "output_dtype": "float32",
            "gradient_dtype": "float32",
            "adaptation": (
                "real native grid/panels, final backward reshape only; original forward unchanged"
            ),
            "boundary": "host arrays; per-view launch, status readback and transfer/copy preserved",
        }
        try:
            image = operator.forward(self.mu)
            for name in self.args.reference_operations:
                call = (
                    (lambda: operator.forward(self.mu))
                    if name == "forward"
                    else (lambda: operator.backward(seeds))
                )
                self.measure(
                    "strict/reference_adapter_" + name,
                    call,
                    {**metadata, "requested_gradients": ["volume"] if name == "backward" else []},
                    stream=adapter_stream,
                    check=lambda: [workspace.check_status() for workspace in operator.workspaces],
                )
            if self.reference is not None:
                self.results["checks"]["reference_adapter_image"] = self.error(
                    image, self.reference
                )
            self.np.save(self.args.output / "reference_adapter_depth.npy", image)
            self.save()
        finally:
            operator.close()

    def material(self) -> None:
        from dpt.material_projection import (
            material_projection_vjp,
            prepare_material_projection,
            project_material_paths,
        )
        from dpt.spectral import SpectralSpec, prepare_spectral, spectral_signal, spectral_vjp

        wp = self.wp
        started = time.perf_counter()
        density = self.np.maximum(1.0 + self.ct.values / 1000.0, 0.0).astype(self.np.float32)
        self.results["preparation"]["water_basis_conversion_seconds"] = (
            time.perf_counter() - started
        )
        with wp.ScopedStream(self.wp_stream):
            fields = wp.array(density.reshape(1, -1), dtype=wp.float32, device="cuda:0")
            paths = wp.empty((1, self.panel.pixels), dtype=wp.float64, device="cuda:0")
            path_seed = wp.empty_like(paths)
            gradients = wp.empty((1, self.grid.voxels), dtype=wp.float64, device="cuda:0")
            grad_pose = wp.empty(6, dtype=wp.float64, device="cuda:0")
            coefficients = wp.array([[0.02059]], dtype=wp.float32, device="cuda:0")
            weights = wp.array([10000.0], dtype=wp.float32, device="cuda:0")
            response = wp.array([1.0], dtype=wp.float32, device="cuda:0")
            mean = wp.empty(self.panel.pixels, dtype=wp.float64, device="cuda:0")
        spectral = prepare_spectral(
            SpectralSpec(
                1,
                1,
                precision="float64",
                output_unit="expected photons per pixel",
                input_description=(
                    "60 keV monoenergetic water-scaled CT; ideal unit-efficiency detector"
                ),
            ),
            max_pixels=self.panel.pixels,
            device="cuda:0",
            stream=self.wp_stream,
        )
        for execution in self.args.material_execution:
            started = time.perf_counter()
            material = prepare_material_projection(
                self.grid,
                self.panel,
                self.spec(pose=True, volume=True),
                materials=1,
                fields=fields,
                out_paths=paths,
                cotangent_paths=path_seed,
                out_fields=gradients,
                device="cuda:0",
                stream=self.wp_stream,
                field_domain="nonnegative",
                channel_execution=execution,
                volume_gradient_precision="float64",
            )
            self.stream.synchronize()
            self.results["preparation"]["material_" + execution + "_seconds"] = (
                time.perf_counter() - started
            )
            metadata = {
                "view": self.args.view,
                "execution": execution,
                "projection_spec": asdict(material.projection.spec),
                "spectral_spec": asdict(spectral.spec),
                "workspace_bytes": material.scratch_bytes + spectral.scratch_bytes,
                "basis": (
                    "one nonnegative dimensionless water-relative-density field; max(1+HU/1000,0)"
                ),
                "basis_derivation": (
                    "algebraic HU conversion; no tissue labels or clinical material calibration"
                ),
                "energy_kev": [60.0],
                "coefficient_mm_inverse": [float(self.np.float32(0.02059))],
                "incident_photons_per_detector_pixel": [10000.0],
                "response_per_photon": [1.0],
                "source_histories": None,
                "boundary": "resident buffers; final finite/status checks outside timing",
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
                    cotangent=self.seed,
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
                    gradients.flatten(), volume_gradient_precision="float64", stream=self.wp_stream
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
                    f"strict/material_{execution}_{name}",
                    call,
                    {**metadata, "requested_gradients": active},
                    check=check_gradient
                    if name in ("material_vjp", "complete_iteration")
                    else check,
                )
            material.projection.enqueue_volume_gradient_check(
                gradients.flatten(), volume_gradient_precision="float64", stream=self.wp_stream
            )
            check()
            host_paths, host_mean = paths.numpy(), mean.numpy()
            expected = 10000.0 * self.np.exp(-float(self.np.float32(0.02059)) * host_paths[0])
            self.results["checks"]["material_" + execution] = {
                "spectral_error_vs_numpy": self.error(host_mean, expected),
                "pose_tangent_gradient": grad_pose.numpy().tolist(),
            }
            self.np.testing.assert_allclose(host_mean, expected, rtol=2e-12, atol=1e-12)
            self.np.save(
                self.args.output / f"material_{execution}_expected_photons.npy",
                host_mean.reshape(self.panel.shape),
            )
            self.np.save(
                self.args.output / f"material_{execution}_water_paths_mm.npy",
                host_paths.reshape(self.panel.shape),
            )
            self.save()


def main() -> None:
    args = arguments()
    args.source = args.source.resolve()
    sys.path.insert(0, str(args.source / "python"))
    from benchmarks import baseline_ct as base

    benchmark: Benchmark | None = None
    try:
        benchmark = Benchmark(args, base)
        for selected in args.cases:
            print("Starting", selected, base.utc(), "output", args.output, flush=True)
            {
                "derivatives": benchmark.derivatives,
                "torch": benchmark.torch_iterations,
                "batched": benchmark.batched,
                "reference": benchmark.reference_adapter,
                "material": benchmark.material,
            }[selected]()
            gc.collect()
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
