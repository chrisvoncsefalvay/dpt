"""Measure real native-CT competitors with explicit geometry and timing boundaries.

Run one backend at a time on one GPU. This script never downloads data.
Every backend must pass the same independent constant/cuboid geometry probes before
its CT measurements are eligible for reporting. Different discretisations are
reported as error-versus-time observations, not interchangeable projector ratios.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks.common import Measurements, NvmlMonitor, load_runtime


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/assets/results/baseline-chest-512.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-image", type=Path, help="Actual DPT FP64 three-view output array"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        required=True,
        choices=("diffdrr-trilinear", "diffdrr-siddon", "astra", "tigre"),
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        default=["forward", "volume_vjp", "pose_vjp", "iteration"],
        choices=("forward", "volume_vjp", "pose_vjp", "iteration"),
    )
    parser.add_argument("--diffdrr-points", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--patch-rays", type=int, default=32768)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--long-repeats", type=int, default=10)
    parser.add_argument("--long-threshold-seconds", type=float, default=0.2)
    parser.add_argument("--nvml-interval-seconds", type=float, default=0.05)
    parser.add_argument("--nvml-device-index", type=int)
    parser.add_argument("--tigre-source", type=Path)
    parser.add_argument("--probes-only", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.rounds,
            args.repeats,
            args.long_repeats,
            args.patch_rays,
            *args.diffdrr_points,
        )
        < 1
    ):
        parser.error("counts must be positive")
    if min(args.diffdrr_points) < 2:
        parser.error("DiffDRR Trilinear needs at least two inclusive points")
    if args.long_threshold_seconds <= 0 or args.nvml_interval_seconds <= 0:
        parser.error("time intervals must be positive")
    return args


def file_digest(path):
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def source_files(module):
    """Hash installed backend Python and extension modules, not unrelated Torch binaries."""
    directory = Path(module.__file__).resolve().parent
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and (path.suffix in {".py", ".pyd"} or ".so" in path.name)
    }


def error_metrics(actual, expected):
    actual, expected = (
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
    )
    if not np.isfinite(actual).all():
        raise ArithmeticError("Nonfinite competitor output")
    difference = actual - expected
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-300)),
    }


def acquisition(input_record):
    """Map unchanged world rays to grid-local millimetres and voxel-centre indices."""
    grid = input_record["grid"]
    size = np.asarray(grid["shape"][::-1], dtype=np.int64)
    spacing = np.asarray(grid["spacing_mm"], dtype=np.float64)
    orientation = np.asarray(grid["orientation"], dtype=np.float64).reshape(3, 3)
    origin = np.asarray(grid["origin_mm"], dtype=np.float64)
    centre_index = (size - 1) / 2
    centre_world = origin + orientation @ (centre_index * spacing)
    sources, targets, vectors = [], [], []
    world_sources, world_targets = [], []
    for panel in input_record["geometry"]:
        rows, columns = panel["shape"]
        row, column = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
        u = np.asarray(panel["u"]) * panel["spacing_mm"][0]
        v = np.asarray(panel["v"]) * panel["spacing_mm"][1]
        source = np.asarray(panel["source_mm"], dtype=np.float64)
        ends = np.asarray(panel["origin_mm"]) + column[..., None] * u + row[..., None] * v
        local_source = (source - centre_world) @ orientation
        local_targets = (ends.reshape(-1, 3) - centre_world) @ orientation
        detector_centre = (
            np.asarray(panel["origin_mm"])
            + (columns - 1) * u / 2
            + (rows - 1) * v / 2
            - centre_world
        ) @ orientation
        vectors.append(
            np.concatenate((local_source, detector_centre, u @ orientation, v @ orientation))
        )
        sources.append(local_source)
        targets.append(local_targets)
        world_sources.append(source)
        world_targets.append(ends.reshape(-1, 3))
    return {
        "shape_zyx": tuple(grid["shape"]),
        "size_xyz": size,
        "spacing_xyz": spacing,
        "orientation": orientation,
        "origin": origin,
        "centre_world": centre_world,
        "centre_index": centre_index,
        "sources_mm": np.asarray(sources),
        "targets_mm": np.asarray(targets),
        "vectors": np.asarray(vectors),
        "world_sources": np.asarray(world_sources),
        "world_targets": np.asarray(world_targets),
        "sources_voxel": np.asarray(sources) / spacing + centre_index,
        "targets_voxel": np.asarray(targets) / spacing + centre_index,
        "image_shape": tuple(input_record["geometry"][0]["shape"]),
    }


def box_projection(rays, low, high, value=0.01):
    """Independent finite-segment slab calculation in physical millimetres."""
    source = rays["sources_mm"][:, None, :]
    direction = rays["targets_mm"] - source
    lower = np.zeros(direction.shape[:-1])
    upper = np.ones_like(lower)
    for axis in range(3):
        velocity = direction[..., axis]
        origin = np.broadcast_to(source[..., axis], velocity.shape)
        parallel = velocity == 0
        first = np.divide(
            low[axis] - origin,
            velocity,
            out=np.full_like(velocity, -np.inf),
            where=~parallel,
        )
        last = np.divide(
            high[axis] - origin,
            velocity,
            out=np.full_like(velocity, np.inf),
            where=~parallel,
        )
        lower = np.maximum(lower, np.minimum(first, last))
        upper = np.minimum(upper, np.maximum(first, last))
        outside = parallel & ((origin < low[axis]) | (origin > high[axis]))
        upper = np.where(outside, -1.0, upper)
    result = value * np.linalg.norm(direction, axis=-1) * np.maximum(upper - lower, 0)
    return result.reshape(len(source), *rays["image_shape"])


def centroid(image):
    image = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    row, column = np.indices(image.shape)
    total = image.sum()
    if total <= 0:
        raise ArithmeticError("Marker projection is empty")
    return np.asarray([(image * row).sum() / total, (image * column).sum() / total])


def compose_torch_pose(torch, anchor, tangent):
    """Ordinary Torch G=anchor@exp(hat(xi)); translations in mm, rotations in radians."""
    x, y, z = tangent[3:].unbind()
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    top = torch.cat((skew, tangent[:3, None]), dim=1)
    bottom = torch.zeros((1, 4), dtype=tangent.dtype, device=tangent.device)
    increment = torch.linalg.matrix_exp(torch.cat((top, bottom), dim=0))
    rotation = anchor[:9].reshape(3, 3)
    shifted = rotation @ increment[:3, 3] + anchor[9:]
    return torch.cat(((rotation @ increment[:3, :3]).reshape(9), shifted))


class Driver:
    def __init__(self, args):
        sys.path.insert(0, str(args.source / "python"))
        base = load_runtime()

        self.args, self.base = args, base
        self.np, self.torch, self.wp = np, base.torch, base.wp
        self.wp.init()
        self.stream = self.torch.cuda.current_stream()
        self.baseline = json.loads(args.baseline_results.read_text())
        self.input = self.baseline["input"]
        if self.input["resampled"] or self.input["samples_per_ray"] != 1024:
            raise ValueError("Frozen native-volume, 1024-midpoint baseline required")
        if self.input["mu_water_mm_inverse"] != 0.02059 or self.input["energy_kev"] != 60:
            raise ValueError("Expected declared 60 keV water-scaled attenuation")
        if len(self.input["geometry"]) != 3 or any(
            p["shape"] != [512, 512] for p in self.input["geometry"]
        ):
            raise ValueError(
                "This comparison requires the frozen three-view 512-square acquisition"
            )
        identity = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        if self.input["pose"] != identity:
            raise ValueError("Frozen acquisition must have identity object pose")
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output / "results.json").exists():
            raise FileExistsError("Refusing to overwrite existing competitor results")
        self.results = {
            "schema_version": 1,
            "measurement_started_at": base.utc(),
            "baseline_results": file_digest(args.baseline_results),
            "baseline_source": self.baseline["source"],
            "input": self.input,
            "harness": file_digest(__file__),
            "helper_sources": [
                file_digest(base.__file__),
                file_digest(inspect.getfile(Measurements)),
            ],
            "auxiliary_dpt_source": base.digest_sources(args.source),
            "environment": {
                "platform": base.platform.platform(),
                "python": base.platform.python_version(),
                "gpu": self.torch.cuda.get_device_name(0),
                "torch_cuda": self.torch.version.cuda,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "numpy",
                        "warp-lang",
                        "nibabel",
                        "nvidia-ml-py",
                    )
                },
                "nvidia_smi_before": base.command("nvidia-smi", "-q"),
            },
            "timing_policy": {
                "rounds": args.rounds,
                "short_repeats": args.repeats,
                "long_repeats": args.long_repeats,
                "long_threshold_seconds": args.long_threshold_seconds,
                "known_stream": "Torch events on executing stream plus completed wall time",
                "unknown_stream": (
                    "completed wall time with CUDA device synchronization; device_ms is null"
                ),
                "first_use": "same process; not an isolated-cache cold start",
                "comparison": (
                    "matched physical rays and units; algorithms differ; no fair "
                    "speed ratio inferred"
                ),
            },
            "geometry_acceptance": {
                "constant_relative_l2_max": 0.05,
                "marker_relative_l2_max": 0.10,
                "marker_centroid_distance_pixels_max": 2.0,
                "scope": "predeclared mapping smoke checks, not clinical image-accuracy guarantees",
            },
            "backends": {},
            "measurements": [],
            "preparation": {},
            "checks": {},
        }
        self.save()
        start = time.perf_counter()
        ct = base.read_nifti(args.ct, value_units="HU")
        self.results["preparation"]["file_loading_seconds"] = time.perf_counter() - start
        if hashlib.sha256(args.ct.read_bytes()).hexdigest() != self.input["sha256"]:
            raise ValueError("CT file differs from frozen baseline")
        if json.loads(json.dumps(asdict(ct.grid))) != self.input["grid"]:
            raise ValueError("CT physical grid differs from frozen baseline")
        start = time.perf_counter()
        self.mu = (np.maximum(1.0 + ct.values / 1000.0, 0.0) * 0.02059).astype(np.float32)
        self.results["preparation"]["ct_conversion_seconds"] = time.perf_counter() - start
        self.rays = acquisition(self.input)
        self.reference_file = (
            args.reference_image or args.baseline_results.parent / "float64_depth.npy"
        )
        if not self.reference_file.is_file():
            raise FileNotFoundError(
                "Run the FP64 real-CT benchmark first and pass its array with --reference-image"
            )
        self.reference = np.load(self.reference_file)
        self.results["reference_image"] = file_digest(self.reference_file)
        self.monitor = NvmlMonitor(base, args)
        self.results["environment"]["nvml_before"] = self.monitor.sample()
        start = time.perf_counter()
        self.field = self.torch.as_tensor(self.mu, device="cuda", dtype=self.torch.float32)
        self.torch.cuda.synchronize()
        self.results["preparation"]["device_upload_seconds"] = time.perf_counter() - start
        region = np.zeros_like(self.reference, dtype=np.float32)
        rows, columns = self.rays["image_shape"]
        region[0, rows * 5 // 16 : rows * 11 // 16, columns * 5 // 16 : columns * 11 // 16] = 1.0
        region /= region.sum()
        self.seed = self.torch.as_tensor(region, device="cuda")
        self.region = region
        self.save()

    save = Measurements.save

    def measure(self, label, operation, metadata, *, known_stream):
        if known_stream:
            return Measurements.measure(self, label, operation, metadata)
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        operation()
        self.torch.cuda.synchronize()
        first = time.perf_counter() - started
        started = time.perf_counter()
        operation()
        self.torch.cuda.synchronize()
        pilot = time.perf_counter() - started
        count = (
            self.args.repeats
            if pilot < self.args.long_threshold_seconds
            else min(self.args.repeats, self.args.long_repeats)
        )
        result = {
            "label": label,
            "configuration": metadata,
            "first_use_in_this_process_seconds": first,
            "pilot_completed_seconds": pilot,
            "repeats_per_round": count,
            "rounds": [],
            "device_median_ms": None,
            "device_p95_ms": None,
        }
        self.results["measurements"].append(result)
        for index in range(self.args.rounds):
            row = {
                "round": index,
                "measured_at": self.base.utc(),
                "wall_ms": [],
                "device_ms": None,
            }
            self.torch.cuda.reset_peak_memory_stats()
            self.monitor.start()
            try:
                for _ in range(count):
                    self.torch.cuda.synchronize()
                    start = time.perf_counter()
                    with self.torch.cuda.nvtx.range(label):
                        operation()
                    self.torch.cuda.synchronize()
                    row["wall_ms"].append((time.perf_counter() - start) * 1000.0)
            finally:
                row["nvml"] = self.monitor.stop()
                row["torch_peak_allocated_bytes"] = self.torch.cuda.max_memory_allocated()
                row["torch_peak_reserved_bytes"] = self.torch.cuda.max_memory_reserved()
                row["finished_at"] = self.base.utc()
                result["rounds"].append(row)
                self.save()
            print(
                label,
                "round",
                index + 1,
                "samples",
                count,
                "median wall ms",
                np.median(row["wall_ms"]),
                flush=True,
            )
        samples = [x for row in result["rounds"] for x in row["wall_ms"]]
        result["wall_median_ms"] = float(np.median(samples))
        result["wall_p95_ms"] = float(np.percentile(samples, 95))
        self.save()
        return result

    def geometry_probes(self, backend):
        rays = self.rays
        extent = rays["size_xyz"] * rays["spacing_xyz"]
        low, high = -extent / 2, extent / 2
        constant = self.torch.full_like(self.field, 0.01)
        with self.torch.no_grad():
            actual = backend.forward(constant).detach().cpu().numpy().copy()
        expected = box_projection(rays, low, high)
        constant_error = error_metrics(actual, expected)
        del constant
        start_xyz = np.floor(rays["size_xyz"] * [0.15, 0.57, 0.32]).astype(int)
        end_xyz = np.floor(rays["size_xyz"] * [0.28, 0.70, 0.48]).astype(int)
        marker = self.torch.zeros_like(self.field)
        x0, y0, z0 = start_xyz
        x1, y1, z1 = end_xyz
        marker[z0:z1, y0:y1, x0:x1] = 0.01
        expected_marker = box_projection(
            rays,
            low + start_xyz * rays["spacing_xyz"],
            low + end_xyz * rays["spacing_xyz"],
        )
        with self.torch.no_grad():
            actual_marker = backend.forward(marker).detach().cpu().numpy().copy()
        marker_error = error_metrics(actual_marker, expected_marker)
        distances = [
            float(np.linalg.norm(centroid(a) - centroid(e)))
            for a, e in zip(actual_marker, expected_marker, strict=True)
        ]
        policy = self.results["geometry_acceptance"]
        passed = (
            constant_error["relative_l2"] <= policy["constant_relative_l2_max"]
            and marker_error["relative_l2"] <= policy["marker_relative_l2_max"]
            and max(distances) <= policy["marker_centroid_distance_pixels_max"]
        )
        result = {
            "measured_at": self.base.utc(),
            "passed": passed,
            "constant_error": constant_error,
            "marker_error": marker_error,
            "marker_centroid_distance_pixels": distances,
            "marker_start_xyz": start_xyz.tolist(),
            "marker_end_exclusive_xyz": end_xyz.tolist(),
            "reference": "independent physical-mm finite-segment box intersection",
            "native_shape_zyx": list(self.mu.shape),
        }
        self.results["checks"][backend.label + "/geometry"] = result
        self.save()
        if not passed:
            raise ArithmeticError(
                "Geometry probe failed; no headline timing is admitted for " + backend.label
            )

    def run_backend(self, backend):
        torch = self.torch
        self.results["backends"][backend.label] = backend.metadata
        self.save()
        self.geometry_probes(backend)
        if self.args.probes_only:
            return
        with torch.no_grad():
            prediction = backend.forward(self.field).detach().cpu().numpy().copy()
        info = self.results["backends"][backend.label]
        info["image_error_vs_dpt_fp64"] = error_metrics(prediction, self.reference)
        np.save(self.args.output / (backend.label + "_depth.npy"), prediction)
        common = {
            **backend.metadata,
            "views": 3,
            "detector_shape": [512, 512],
            "native_shape_zyx": list(self.mu.shape),
            "field_dtype": "float32",
            "objective": "mean of central AP detector region; other two view cotangents are zero",
        }
        if "forward" in self.args.operations:

            def forward():
                with torch.no_grad():
                    return backend.forward(self.field)

            self.measure(
                backend.label + "/forward_3views",
                forward,
                {**common, "requested_gradients": []},
                known_stream=backend.known_stream,
            )
        field = self.field.detach().requires_grad_(True)
        if "volume_vjp" in self.args.operations or "iteration" in self.args.operations:
            if backend.known_stream:
                graph = backend.forward(field)

                def volume_vjp():
                    return torch.autograd.grad(graph, field, self.seed, retain_graph=True)[0]

                boundary = (
                    "backward only; resident full three-view forward graph prepared outside timing"
                )
            else:

                def volume_vjp():
                    return backend.backward(self.seed)

                boundary = backend.metadata["backward_boundary"]
            gradient = volume_vjp()
            torch.cuda.synchronize()
            if not bool(torch.isfinite(gradient).all().item()):
                raise ArithmeticError("Nonfinite competitor volume backprojection")
            left = float(np.sum(prediction.astype(np.float64) * self.region))
            right = float(torch.sum(field.detach().double() * gradient.detach().double()).item())
            info["adjoint"] = {
                "forward_inner_product": left,
                "backprojection_inner_product": right,
                "relative_defect": abs(left - right) / max(abs(left), abs(right), 1e-300),
                "backprojection": backend.metadata["backprojection"],
            }
            if "volume_vjp" in self.args.operations:
                self.measure(
                    backend.label + "/volume_vjp_3views",
                    volume_vjp,
                    {**common, "requested_gradients": ["volume"], "boundary": boundary},
                    known_stream=backend.known_stream,
                )
            del gradient, volume_vjp
            if backend.known_stream:
                del graph
        if "pose_vjp" in self.args.operations:
            if backend.pose_supported:
                tangent = torch.zeros(6, dtype=torch.float64, device="cuda", requires_grad=True)
                graph = backend.forward(self.field, tangent=tangent)

                def pose_vjp():
                    return torch.autograd.grad(graph, tangent, self.seed, retain_graph=True)[0]

                pose_gradient = pose_vjp().detach()
                if not bool(torch.isfinite(pose_gradient).all().item()):
                    raise ArithmeticError("Nonfinite competitor pose VJP")
                direction = torch.tensor(
                    [0.6, -0.3, 0.2, 0.0007, -0.0005, 0.0003],
                    dtype=torch.float64,
                    device="cuda",
                )
                exact = float(torch.dot(pose_gradient, direction).item())
                differences = []
                with torch.no_grad():
                    for epsilon in (0.1, 0.03, 0.01):
                        plus = (
                            (backend.forward(self.field, tangent=epsilon * direction) * self.seed)
                            .double()
                            .sum()
                        )
                        minus = (
                            (backend.forward(self.field, tangent=-epsilon * direction) * self.seed)
                            .double()
                            .sum()
                        )
                        finite_difference = float(((plus - minus) / (2 * epsilon)).item())
                        differences.append(
                            {
                                "epsilon": epsilon,
                                "finite_difference": finite_difference,
                                "relative_error": abs(finite_difference - exact)
                                / max(abs(exact), abs(finite_difference), 1e-12),
                            }
                        )
                info["pose_directional_check"] = {
                    "gradient": pose_gradient.cpu().tolist(),
                    "direction_translation_mm_rotation_rad": direction.cpu().tolist(),
                    "analytic_directional_derivative": exact,
                    "epsilon_sweep": differences,
                    "scope": (
                        "local derivative of this backend's own discretisation; branch "
                        "changes may affect finite differences"
                    ),
                }
                self.measure(
                    backend.label + "/pose_vjp_3views",
                    pose_vjp,
                    {
                        **common,
                        "requested_gradients": ["pose"],
                        "boundary": (
                            "backward only; full resident forward graph prepared outside timing"
                        ),
                    },
                    known_stream=True,
                )
                del graph, pose_vjp, tangent, pose_gradient
            else:
                info["pose_vjp"] = {
                    "status": "unsupported",
                    "reason": "backend has fixed geometry without pose autograd",
                }
        if "iteration" in self.args.operations:
            tangent = torch.zeros(
                6,
                dtype=torch.float64,
                device="cuda",
                requires_grad=backend.pose_supported,
            )

            def iteration():
                field.grad = None
                tangent.grad = None
                if backend.pose_supported:
                    output = backend.forward(field, tangent=tangent)
                else:
                    output = backend.differentiable_forward(field)
                (output * self.seed).sum().backward()
                return output

            self.measure(
                backend.label + "/complete_iteration_3views",
                iteration,
                {
                    **common,
                    "requested_gradients": ["volume", "pose"]
                    if backend.pose_supported
                    else ["volume"],
                    "boundary": (
                        "forward, detector objective and backward; leaf gradients reset "
                        "every iteration"
                    ),
                    "backward_is_exact_transpose": backend.known_stream,
                    "gradient_caveat": (
                        "see measured adjoint defect; standard native BP is not assumed "
                        "to be an exact transpose"
                    ),
                },
                known_stream=backend.known_stream,
            )
            del iteration, tangent
        self.save()


class DiffDRR:
    known_stream = True
    pose_supported = True

    def __init__(self, driver, method, points):
        import diffdrr
        from diffdrr import renderers

        self.d, self.torch, self.method, self.points = (
            driver,
            driver.torch,
            method,
            points,
        )
        version = importlib.metadata.version("diffdrr")
        if version != "0.6.1":
            raise RuntimeError("Expected DiffDRR 0.6.1, got " + version)
        self.label = "diffdrr-" + method + ("-" + str(points) if method == "trilinear" else "")
        self.renderer = renderers.Trilinear() if method == "trilinear" else renderers.Siddon()
        self.renderers = renderers
        self.metadata = {
            "package": "diffdrr",
            "version": version,
            "sources": source_files(diffdrr),
            "algorithm": "trilinear inclusive rectangular rule"
            if method == "trilinear"
            else "Siddon exact cell intersections with nearest samples",
            "n_points": points if method == "trilinear" else None,
            "patch_rays": driver.args.patch_rays,
            "voxel_shift": 0.5,
            "align_corners": False,
            "padding": "grid_sample default zero",
            "sampling_interval": (
                "one library-computed global alpha interval per full view, reused by all patches"
            )
            if method == "trilinear"
            else "all axis-plane crossings; default unfiltered intersections",
            "geometry_dtype": (
                "local voxel coordinates FP32; pose/world-to-local composition FP64 before cast"
            ),
            "boundary": (
                "resident Torch input/output; geometry and patch loop included; "
                "no image host transfer"
            ),
            "backprojection": "native PyTorch autograd of the actual forward discretisation",
            "comparison_limit": (
                "DPT uses half-cell extension and clipped midpoints; no "
                "same-algorithm speedup claim"
            ),
        }
        torch = self.torch
        rays = driver.rays
        self.static_source = torch.as_tensor(
            rays["sources_voxel"], device="cuda", dtype=torch.float32
        )
        self.static_target = torch.as_tensor(
            rays["targets_voxel"], device="cuda", dtype=torch.float32
        )
        self.world_source = torch.as_tensor(
            rays["world_sources"], device="cuda", dtype=torch.float64
        )
        self.world_target = torch.as_tensor(
            rays["world_targets"], device="cuda", dtype=torch.float64
        )
        self.rotation = torch.as_tensor(rays["orientation"], device="cuda", dtype=torch.float64)
        self.origin = torch.as_tensor(rays["origin"], device="cuda", dtype=torch.float64)
        self.spacing = torch.as_tensor(rays["spacing_xyz"], device="cuda", dtype=torch.float64)
        self.anchor = torch.as_tensor(driver.input["pose"], device="cuda", dtype=torch.float64)
        lengths = np.linalg.norm(rays["targets_mm"] - rays["sources_mm"][:, None, :], axis=-1)
        self.lengths = torch.as_tensor(lengths, device="cuda", dtype=torch.float32)

    def forward(self, field, tangent=None):
        torch = self.torch
        if tangent is None:
            source, target = self.static_source, self.static_target
        else:
            pose = compose_torch_pose(torch, self.anchor, tangent)
            rotation, translation = pose[:9].reshape(3, 3), pose[9:]
            source = (
                ((self.world_source - translation) @ rotation - self.origin)
                @ self.rotation
                / self.spacing
            ).float()
            target = (
                ((self.world_target - translation) @ rotation - self.origin)
                @ self.rotation
                / self.spacing
            ).float()
        volume = field.permute(2, 1, 0)
        images = []
        for view in range(len(source)):
            starts = source[view].reshape(1, 1, 3)
            ends = target[view].unsqueeze(0)
            kwargs = {}
            if self.method == "trilinear":
                dims = self.renderer.dims(volume)
                alpha_min, alpha_max = self.renderers._get_alpha_minmax(
                    starts, ends, dims, self.renderer.voxel_shift, self.renderer.eps
                )
                kwargs = {
                    "n_points": self.points,
                    "alphamin": alpha_min.min(),
                    "alphamax": alpha_max.max(),
                }
            chunks = []
            for begin in range(0, ends.shape[1], self.d.args.patch_rays):
                end = min(begin + self.d.args.patch_rays, ends.shape[1])
                lengths = self.lengths[view, begin:end].reshape(1, 1, -1)
                chunks.append(
                    self.renderer(volume, starts, ends[:, begin:end], lengths, **kwargs).reshape(-1)
                )
            images.append(torch.cat(chunks).reshape(self.d.rays["image_shape"]))
        return torch.stack(images)

    def close(self):
        pass


class Astra:
    known_stream = False
    pose_supported = False

    def __init__(self, driver):
        import astra

        self.d, self.torch, self.astra, self.label = (
            driver,
            driver.torch,
            astra,
            "astra",
        )
        version = importlib.metadata.version("astra-toolbox")
        if version != "2.5.0":
            raise RuntimeError("Expected ASTRA 2.5.0, got " + version)
        rays = driver.rays
        nz, ny, nx = rays["shape_zyx"]
        sx, sy, sz = rays["size_xyz"] * rays["spacing_xyz"]
        self.volume_geometry = astra.create_vol_geom(
            ny, nx, nz, -sx / 2, sx / 2, -sy / 2, sy / 2, -sz / 2, sz / 2
        )
        rows, columns = rays["image_shape"]
        self.geometry = astra.create_proj_geom("cone_vec", rows, columns, rays["vectors"])
        self.projector = astra.create_projector("cuda3d", self.geometry, self.volume_geometry)
        if not astra.use_cuda():
            raise RuntimeError("ASTRA CUDA unavailable")
        self.output = driver.torch.empty(
            (rows, 3, columns), device="cuda", dtype=driver.torch.float32
        )
        self.gradient = driver.torch.empty_like(driver.field)
        self.seed_native = driver.seed.permute(1, 0, 2).contiguous()
        self.metadata = {
            "package": "astra-toolbox",
            "version": version,
            "sources": source_files(astra),
            "algorithm": (
                "cuda3d Joseph/slice-interpolated forward; standard voxel-driven backprojection"
            ),
            "sampling": "ASTRA native slice traversal; no DPT midpoint-count equivalence",
            "geometry": {
                "cone_vec": rays["vectors"].tolist(),
                "volume_extent_xyz_mm": [sx, sy, sz],
            },
            "layout": "native ZYX volume; detector-row, view, detector-column ASTRA output",
            "boundary": (
                "direct_FP with preallocated resident Torch DLPack buffers; zero "
                "and device synchronisation included"
            ),
            "backward_boundary": (
                "direct_BP with preallocated resident DLPack buffers; zero and "
                "device synchronisation included"
            ),
            "backprojection": (
                "ASTRA standard BP; actual adjoint defect measured, not assumed exact"
            ),
            "stream": (
                "internal stream not assumed to equal Torch current stream; "
                "device event times omitted"
            ),
            "api_reference": (
                "https://raw.githubusercontent.com/astra-toolbox/astra-toolbox/v2."
                "5.0/python/astra/projector3d.py"
            ),
        }
        self.autograd = native_autograd(self)

    def forward(self, field):
        self.output.zero_()
        self.torch.cuda.synchronize()
        self.astra.projector3d.direct_FP(self.projector, field.detach(), out=self.output)
        self.torch.cuda.synchronize()
        return self.output.permute(1, 0, 2)

    def backward(self, seed):
        self.gradient.zero_()
        native = (
            self.seed_native if seed is self.d.seed else seed.detach().permute(1, 0, 2).contiguous()
        )
        self.torch.cuda.synchronize()
        self.astra.projector3d.direct_BP(self.projector, native, out=self.gradient)
        self.torch.cuda.synchronize()
        return self.gradient

    def differentiable_forward(self, field):
        return self.autograd.apply(field)

    def close(self):
        self.astra.projector3d.delete(self.projector)


def native_autograd(backend):
    torch = backend.torch

    class Native(torch.autograd.Function):
        @staticmethod
        def forward(ctx, field):
            return backend.forward(field)

        @staticmethod
        def backward(ctx, seed):
            return backend.backward(seed)

    return Native


def tigre_circular_parameters(rays):
    """Derive and reconstruct TIGRE's native circular detector frame before use."""
    sources = rays["sources_mm"]
    centres, u, v = (
        rays["vectors"][:, 3:6],
        rays["vectors"][:, 6:9],
        rays["vectors"][:, 9:12],
    )
    angles = np.arctan2(sources[:, 1], sources[:, 0])
    sod = np.linalg.norm(sources, axis=1)
    sdd = np.linalg.norm(sources - centres, axis=1)
    source_check = np.column_stack((sod * np.cos(angles), sod * np.sin(angles), np.zeros_like(sod)))
    centre_check = source_check * (1 - sdd / sod)[:, None]
    column_check = np.column_stack((-np.sin(angles), np.cos(angles), np.zeros_like(sod)))
    row_check = np.tile([0.0, 0.0, 1.0], (len(sod), 1))
    discrepancies = {
        "source_max_abs_mm": float(np.max(np.abs(source_check - sources))),
        "detector_centre_max_abs_mm": float(np.max(np.abs(centre_check - centres))),
        "column_direction_max_abs": float(
            np.max(np.abs(column_check - u / np.linalg.norm(u, axis=1)[:, None]))
        ),
        "row_direction_max_abs": float(
            np.max(np.abs(row_check - v / np.linalg.norm(v, axis=1)[:, None]))
        ),
    }
    if max(discrepancies.values()) > 1e-8:
        raise ValueError(
            "Frozen rays are not represented by this TIGRE circular mapping: " + repr(discrepancies)
        )
    return angles, sod, sdd, discrepancies


class Tigre:
    known_stream = False
    pose_supported = False

    def __init__(self, driver):
        import tigre
        from tigre.utilities.gpu import GpuIds
        from tigre.utilities.pytorch_bindings import A

        self.d, self.torch, self.tigre, self.label = (
            driver,
            driver.torch,
            tigre,
            "tigre",
        )
        version = getattr(tigre, "__version__", None)
        for distribution in ("pytigre", "tigre"):
            try:
                version = importlib.metadata.version(distribution)
                break
            except importlib.metadata.PackageNotFoundError:
                pass
        if str(version) != "3.1.3":
            raise RuntimeError("Expected TIGRE 3.1.3, got " + str(version))
        rays = driver.rays
        angles, sod, sdd, mapping = tigre_circular_parameters(rays)
        geo = tigre.geometry(mode="cone", default=False)
        geo.nVoxel = np.asarray(rays["shape_zyx"], dtype=np.int32)
        geo.dVoxel = rays["spacing_xyz"][::-1].copy()
        geo.sVoxel = geo.nVoxel * geo.dVoxel
        geo.nDetector = np.asarray(rays["image_shape"], dtype=np.int32)
        pitch = driver.input["geometry"][0]["spacing_mm"]
        geo.dDetector = np.asarray(pitch[::-1])
        geo.sDetector = geo.nDetector * geo.dDetector
        geo.DSO, geo.DSD = sod, sdd
        geo.offOrigin, geo.offDetector, geo.rotDetector = (
            np.zeros(3),
            np.zeros(2),
            np.zeros(3),
        )
        geo.accuracy, geo.COR = 0.5, 0.0
        self.geo, self.angles = geo, angles
        self.gpuids = GpuIds()
        self.gpuids.devices = [0]
        self.ax = A(geo, angles, self.gpuids)
        self.metadata = {
            "package": "TIGRE",
            "version": str(version),
            "sources": source_files(tigre),
            "algorithm": "Siddon forward; explicit matched native Atb",
            "sampling": "native voxel intersections; no DPT midpoint-count equivalence",
            "geometry": {
                "angles_radians": angles.tolist(),
                "DSO_mm": sod.tolist(),
                "DSD_mm": sdd.tolist(),
                "mapping_residuals": mapping,
            },
            "boundary": (
                "shipped TIGRE PyTorch A: GPU tensor to CPU NumPy, native "
                "upload/project/download, GPU output"
            ),
            "backward_boundary": (
                "benchmark adapter: GPU cotangent to CPU, explicit Atb matched, GPU volume result"
            ),
            "backprojection": "Atb(..., backprojection_type='matched'); measured adjoint defect",
            "iteration_adapter": (
                "shipped Torch forward plus explicit matched BP; shipped default "
                "FDK backward is not used"
            ),
            "stream": "internal stream unknown; completed device-synchronised wall time only",
        }
        self.metadata["native_extensions"] = {
            name: file_digest(importlib.import_module(name).__file__)
            for name in ("_Ax", "_Atb", "_gpuUtils")
        }
        if driver.args.tigre_source:
            self.metadata["source_revision"] = driver.base.command(
                "git", "-C", str(driver.args.tigre_source), "rev-parse", "HEAD"
            )
            self.metadata["source_dirty"] = driver.base.command(
                "git", "-C", str(driver.args.tigre_source), "status", "--short"
            )
        self.autograd = native_autograd(self)

    def forward(self, field):
        return self.ax(field)

    def backward(self, seed):
        host = np.ascontiguousarray(seed.detach().cpu().numpy(), dtype=np.float32)
        result = self.tigre.Atb(
            host,
            self.geo,
            self.angles,
            backprojection_type="matched",
            gpuids=self.gpuids,
        )
        return self.torch.from_numpy(result).to(seed.device)

    def differentiable_forward(self, field):
        return self.autograd.apply(field)

    def close(self):
        pass


def main():
    args = arguments()
    driver = Driver(args)
    try:
        for name in args.backends:
            counts = args.diffdrr_points if name == "diffdrr-trilinear" else [None]
            for points in counts:
                backend = None
                label = name + ("-" + str(points) if points is not None else "")
                try:
                    started = time.perf_counter()
                    if name.startswith("diffdrr-"):
                        backend = DiffDRR(driver, name.split("-", 1)[1], points)
                    elif name == "astra":
                        backend = Astra(driver)
                    else:
                        backend = Tigre(driver)
                    driver.torch.cuda.synchronize()
                    driver.results["preparation"][label + "_seconds"] = (
                        time.perf_counter() - started
                    )
                    print("START", label, driver.base.utc(), flush=True)
                    driver.run_backend(backend)
                except Exception as error:
                    driver.results.setdefault("failures", []).append(
                        {
                            "backend": label,
                            "measured_at": driver.base.utc(),
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    driver.save()
                    raise
                finally:
                    if backend is not None:
                        backend.close()
                    del backend
                    gc.collect()
                    driver.torch.cuda.empty_cache()
        driver.results["measurement_finished_at"] = driver.base.utc()
        driver.results["environment"]["nvidia_smi_after"] = driver.base.command("nvidia-smi", "-q")
        driver.save()
    finally:
        driver.monitor.close()
    print("RESULTS", args.output / "results.json", flush=True)


if __name__ == "__main__":
    main()
