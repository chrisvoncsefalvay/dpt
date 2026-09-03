"""Compare ASTRA/TIGRE on the saved full-extent 256³ CTChest acquisition.

Run on the serial GPU queue after preparing inputs and the actual DPT FP64
reference. Native ASTRA uses resident DLPack buffers. TIGRE's native API uses
host arrays; its shipped Torch forward copies through the host. Each boundary
is measured separately, with completed wall time and no unrelated-stream CUDA
event. Native backprojections are checked, never presumed exact transposes.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import time
import traceback
from pathlib import Path

import numpy as np

from benchmarks.common import Measurements, NvmlMonitor, load_runtime
from benchmarks.competitors import (
    Astra,
    Driver,
    Tigre,
    centroid,
    native_autograd,
    source_files,
)
from benchmarks.fullscan_inputs import load_prepared, sha256


def digest(path: Path) -> dict:
    return {"file": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def compare_arrays(actual: np.ndarray, reference: np.ndarray) -> dict:
    """Compare complete arrays using bounded FP64 slice temporaries."""
    if actual.shape != reference.shape:
        raise ValueError(f"array shapes differ: {actual.shape} != {reference.shape}")
    squared_error = squared_reference = maximum = 0.0
    for a, b in zip(actual, reference, strict=True):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ArithmeticError("comparison contains nonfinite values")
        delta = a - b
        squared_error += float(np.sum(delta * delta))
        squared_reference += float(np.sum(b * b))
        maximum = max(maximum, float(np.max(np.abs(delta))))
    return {
        "max_abs": maximum,
        "rmse": float(np.sqrt(squared_error / actual.size)),
        "relative_l2": float(np.sqrt(squared_error / max(squared_reference, 1e-300))),
    }


def inner_product(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("inner-product shapes differ")
    return sum(
        float(np.sum(np.asarray(x, dtype=np.float64) * np.asarray(y, dtype=np.float64)))
        for x, y in zip(a, b, strict=True)
    )


def compact_rays(prepared, directory: Path) -> dict:
    """Retain twelve physical-mm values per view, not all detector endpoints."""
    metadata = prepared.metadata
    vectors = np.load(directory / metadata["arrays"]["vectors_local"]["file"], allow_pickle=False)
    views = metadata["acquisition"]["views"]
    if vectors.shape != (views, 12) or not np.isfinite(vectors).all():
        raise ValueError("invalid compact local acquisition vectors")
    return {
        "vectors": vectors,
        "sources_mm": vectors[:, :3],
        "shape_zyx": prepared.volume.values.shape,
        "size_xyz": np.asarray(prepared.volume.values.shape[::-1]),
        "spacing_xyz": np.asarray(prepared.volume.grid.spacing_mm),
        "image_shape": tuple(metadata["acquisition"]["detector_shape"]),
        "views": views,
    }


def box_view(vector, shape, low, high, value=0.01) -> np.ndarray:
    """Independent exact finite-ray intersection with a constant physical box."""
    rows, columns = shape
    column = np.arange(columns, dtype=np.float64) - (columns - 1) / 2
    row = np.arange(rows, dtype=np.float64) - (rows - 1) / 2
    source = vector[:3]
    target = vector[3:6] + column[None, :, None] * vector[6:9]
    target = target + row[:, None, None] * vector[9:12]
    direction = target - source
    enter, leave = np.zeros(shape), np.ones(shape)
    valid = np.ones(shape, dtype=bool)
    for axis in range(3):
        velocity = direction[..., axis]
        parallel = velocity == 0
        valid &= ~parallel | ((source[axis] >= low[axis]) & (source[axis] <= high[axis]))
        divisor = np.where(parallel, 1.0, velocity)
        first, second = (low[axis] - source[axis]) / divisor, (high[axis] - source[axis]) / divisor
        enter = np.maximum(enter, np.where(parallel, -np.inf, np.minimum(first, second)))
        leave = np.minimum(leave, np.where(parallel, np.inf, np.maximum(first, second)))
    return (
        value
        * np.linalg.norm(direction, axis=-1)
        * np.where(valid, np.maximum(leave - enter, 0), 0)
    )


def box_comparison(actual, rays, low, high, *, centroids=False) -> dict:
    squared_error = squared_reference = maximum = 0.0
    distances = []
    if actual.shape != (rays["views"], *rays["image_shape"]):
        raise ValueError("backend projection shape does not match the physical acquisition")
    for observed, vector in zip(actual, rays["vectors"], strict=True):
        expected = box_view(vector, rays["image_shape"], low, high)
        observed = np.asarray(observed, dtype=np.float64)
        if not np.isfinite(observed).all():
            raise ArithmeticError("geometry probe contains nonfinite projection values")
        delta = observed - expected
        squared_error += float(np.sum(delta * delta))
        squared_reference += float(np.sum(expected * expected))
        maximum = max(maximum, float(np.max(np.abs(delta))))
        if centroids:
            distances.append(float(np.linalg.norm(centroid(observed) - centroid(expected))))
    return {
        "max_abs": maximum,
        "rmse": float(np.sqrt(squared_error / actual.size)),
        "relative_l2": float(np.sqrt(squared_error / max(squared_reference, 1e-300))),
        "centroid_distance_pixels": distances if centroids else None,
    }


class AstraFullscan(Astra):
    """The existing resident ASTRA adapter with an arbitrary saved view count."""

    def __init__(self, driver, supersampling: int):
        import astra

        self.d, self.torch, self.astra = driver, driver.torch, astra
        self.label = f"astra-ss{supersampling}"
        version = importlib.metadata.version("astra-toolbox")
        if version != "2.5.0":
            raise RuntimeError(f"Expected measured ASTRA 2.5.0, found {version}")
        rays = driver.rays
        nz, ny, nx = rays["shape_zyx"]
        sx, sy, sz = rays["size_xyz"] * rays["spacing_xyz"]
        rows, columns = rays["image_shape"]
        volume = astra.create_vol_geom(
            ny, nx, nz, -sx / 2, sx / 2, -sy / 2, sy / 2, -sz / 2, sz / 2
        )
        projection = astra.create_proj_geom("cone_vec", rows, columns, rays["vectors"])
        self.projector = astra.create_projector(
            "cuda3d",
            projection,
            volume,
            options={"DetectorSuperSampling": supersampling, "VoxelSuperSampling": 1},
        )
        self.output = self.torch.empty(
            (rows, rays["views"], columns), device="cuda", dtype=self.torch.float32
        )
        self.gradient = self.torch.empty_like(driver.field)
        self.seed_native = driver.seed.permute(1, 0, 2).contiguous()
        self.host_output = np.empty(tuple(self.output.shape), dtype=np.float32)
        self.host_gradient = np.empty(rays["shape_zyx"], dtype=np.float32)
        self.autograd = native_autograd(self)
        self.metadata = {
            "package": "astra-toolbox",
            "version": version,
            "sources": source_files(astra),
            "precision": "FP32 input/output and native CUDA arithmetic; no FP64 API claimed",
            "algorithm": "ASTRA cuda3d native forward and voxel-driven native backprojection",
            "sampling": {
                "DetectorSuperSampling": supersampling,
                "VoxelSuperSampling": 1,
                "meaning": "native slice traversal; detector supersampling averages subpixel rays",
                "same_discretisation_as_dpt": False,
            },
            "quality_caveat": "supersampling changes pixel integration, not DPT's ray sample count",
            "resident_boundary": (
                "preallocated CUDA DLPack direct_FP/direct_BP; "
                "zeros and device synchronisation included"
            ),
            "host_boundary": (
                "native CPU DLPack direct_FP/direct_BP; ASTRA performs required transfers; "
                "completed host arrays returned"
            ),
            "torch_boundary": "benchmark autograd adapter around resident direct_FP/direct_BP",
            "backprojection": "native BP, without FDK filtering; exact transpose is not assumed",
            "layout": (
                "native ZYX volume; ASTRA row/view/column buffers exposed in view/row/column order"
            ),
            "geometry": {
                "volume_extent_xyz_mm": [float(sx), float(sy), float(sz)],
                "cone_vec": rays["vectors"].tolist(),
            },
            "source_references": [
                "https://raw.githubusercontent.com/astra-toolbox/astra-toolbox/v2.5.0/python/astra/projector3d.py",
                "https://astra-toolbox.com/docs/data3d.html",
                "https://astra-toolbox.com/docs/geom3d.html",
                "https://astra-toolbox.com/docs/algs/FP3D_CUDA.html",
                "https://astra-toolbox.com/docs/algs/BP3D_CUDA.html",
            ],
        }

    def host_forward(self, field):
        self.host_output.fill(0)
        self.torch.cuda.synchronize()
        self.astra.projector3d.direct_FP(self.projector, field, out=self.host_output)
        self.torch.cuda.synchronize()
        return self.host_output.transpose(1, 0, 2)

    def host_backward(self, seed):
        self.host_gradient.fill(0)
        native = np.ascontiguousarray(seed.transpose(1, 0, 2))
        self.torch.cuda.synchronize()
        self.astra.projector3d.direct_BP(self.projector, native, out=self.host_gradient)
        self.torch.cuda.synchronize()
        return self.host_gradient

    def release_resident_buffers(self):
        self.output = self.gradient = self.seed_native = None


class TigreFullscan(Tigre):
    """Preserve TIGRE's real native and shipped Torch transfer behaviour."""

    def __init__(self, driver, algorithm: str, accuracy: float):
        super().__init__(driver)
        self.algorithm, self.geo.accuracy = algorithm, accuracy
        self.label = "tigre-siddon" if algorithm == "Siddon" else f"tigre-interpolated-{accuracy:g}"
        self.metadata.update(
            {
                "precision": "FP32 arrays and native float CUDA arithmetic; no FP64 API claimed",
                "algorithm": f"TIGRE {algorithm} forward; explicit matched native Atb",
                "sampling": {
                    "projection_type": algorithm,
                    "accuracy": accuracy,
                    "accuracy_used_by_forward": algorithm == "interpolated",
                    "meaning": (
                        "Siddon piecewise-constant voxel intersections, "
                        "or native interpolated ray integration"
                    ),
                    "same_discretisation_as_dpt": False,
                },
                "resident_boundary": None,
                "resident_unsupported_reason": (
                    "installed native Ax/Atb and shipped Torch bindings copy through host arrays"
                ),
                "host_boundary": (
                    "native Ax/Atb NumPy inputs and outputs; "
                    "required GPU transfers/allocation included"
                ),
                "torch_boundary": (
                    "shipped TIGRE Torch A forward; adapter uses explicit matched Atb backward"
                    if algorithm == "Siddon"
                    else (
                        "benchmark Torch adapter for interpolated Ax and explicit matched Atb; "
                        "required host copies included"
                    )
                ),
                "backprojection": (
                    "Atb(backprojection_type='matched'); "
                    "measured adjoint defect, not presumed exact"
                ),
                "source_references": [
                    "https://github.com/CERN/TIGRE/blob/6b0951a8a88aa88a7db5b2a95181a33a95aa8b28/Python/tigre/utilities/pytorch_bindings.py",
                    "https://github.com/CERN/TIGRE/blob/6b0951a8a88aa88a7db5b2a95181a33a95aa8b28/Python/tigre/utilities/Ax.py",
                    "https://github.com/CERN/TIGRE/blob/6b0951a8a88aa88a7db5b2a95181a33a95aa8b28/Python/tigre/utilities/Atb.py",
                ],
            }
        )

    def host_forward(self, field):
        return self.tigre.Ax(
            field, self.geo, self.angles, projection_type=self.algorithm, gpuids=self.gpuids
        )

    def host_backward(self, seed):
        return self.tigre.Atb(
            seed, self.geo, self.angles, backprojection_type="matched", gpuids=self.gpuids
        )

    def forward(self, field):
        if self.algorithm == "Siddon":
            return self.ax(field)
        host = np.ascontiguousarray(field.detach().cpu().numpy(), dtype=np.float32)
        return self.torch.from_numpy(self.host_forward(host)).to(field.device)


class FullscanDriver:
    # Reuse the existing completed-wall timer: no CUDA events are attributed to
    # the backends' internal streams, and every round retains raw samples/NVML.
    measure = Driver.measure
    save = Measurements.save

    def __init__(self, args):
        self.args, self.base, self.np = args, load_runtime(), np
        self.torch = self.base.torch
        self.base.wp.init()
        if not self.torch.cuda.is_available():
            raise RuntimeError("CUDA is required for actual ASTRA/TIGRE measurements")
        args.output.mkdir(parents=True, exist_ok=False)
        start = time.perf_counter()
        prepared = load_prepared(args.input)
        self.input = prepared.metadata
        # NumPy cannot export a read-only ImagingVolume buffer through DLPack.
        # Make this caller-owned host input once, outside all timed operations.
        self.mu = np.array(prepared.volume.values, copy=True, order="C")
        if self.mu.dtype != np.float32 or not self.mu.flags.c_contiguous:
            raise ValueError("prepared attenuation must be C-contiguous FP32 ZYX")
        self.rays = compact_rays(prepared, args.input)
        shape = (self.rays["views"], *self.rays["image_shape"])
        self.region = np.load(args.cotangent, mmap_mode="r", allow_pickle=False)
        if (
            self.region.shape != shape
            or self.region.dtype != np.float32
            or not np.isfinite(self.region).all()
        ):
            raise ValueError(
                "cotangent must be finite FP32 with the saved complete acquisition shape"
            )
        # TIGRE registers its host input without cudaHostRegisterReadOnly.
        # Own a writable buffer once; timed native calls reuse these exact values.
        seed_copy_started = time.perf_counter()
        self.region = np.array(self.region, copy=True, order="C")
        seed_copy_seconds = time.perf_counter() - seed_copy_started
        self.reference = None
        self.reference_gradient = None
        if not args.probes_only:
            if args.reference_image is None:
                raise ValueError("--reference-image must name an actual DPT FP64 projection")
            self.reference = np.load(args.reference_image, mmap_mode="r", allow_pickle=False)
            if self.reference.shape != shape or self.reference.dtype != np.float64:
                raise ValueError("DPT FP64 reference projection has wrong shape/dtype")
            if args.reference_gradient:
                self.reference_gradient = np.load(
                    args.reference_gradient, mmap_mode="r", allow_pickle=False
                )
                if (
                    self.reference_gradient.shape != self.mu.shape
                    or self.reference_gradient.dtype not in (np.float32, np.float64)
                ):
                    raise ValueError(
                        "DPT FP64-arithmetic reference VJP has wrong shape/storage dtype"
                    )
        self.results = {
            "schema_version": 1,
            "measurement_started_at": self.base.utc(),
            "workload": "complete CTChest resampled to 256^3, 360 circular views; no cropping",
            "input": self.input,
            "input_metadata": digest(args.input / "metadata.json"),
            "cotangent": digest(args.cotangent),
            "reference_image": digest(args.reference_image) if args.reference_image else None,
            "reference_gradient": digest(args.reference_gradient)
            if args.reference_gradient
            else None,
            "reference_gradient_storage_dtype": str(self.reference_gradient.dtype)
            if self.reference_gradient is not None
            else None,
            "source": self.base.digest_sources(args.source),
            "harness_sources": {
                p.name: digest(p)
                for p in (
                    Path(__file__),
                    Path(__file__).with_name("competitors.py"),
                    Path(__file__).with_name("common.py"),
                    Path(__file__).with_name("fullscan_inputs.py"),
                )
            },
            "geometry_acceptance": {
                "constant_relative_l2_max": 0.05,
                "marker_relative_l2_max": 0.10,
                "marker_centroid_distance_pixels_max": 2.0,
                "scope": "predeclared gross geometry gate; not a matched-discretisation error gate",
            },
            "unsupported": {
                "fp64": "neither tested ASTRA nor TIGRE API provides the requested FP64 path",
                "pose_vjp": "fixed-geometry native operators do not provide pose autograd",
            },
            "timing": (
                "completed wall time; CUDA synchronisation before/after; "
                "internal backend streams unknown"
            ),
            "environment": {
                "python": self.base.platform.python_version(),
                "torch": self.torch.__version__,
                "torch_cuda": self.torch.version.cuda,
                "warp": self.base.wp.__version__,
                "gpu": self.torch.cuda.get_device_name(0),
                "nvidia_smi_before": self.base.command("nvidia-smi", "-q"),
            },
            "preparation": {
                "load_and_verify_inputs_seconds": time.perf_counter() - start,
                "cotangent_host_copy": {
                    "seconds": seed_copy_seconds,
                    "allocation_bytes": self.region.nbytes,
                    "dtype": str(self.region.dtype),
                    "owned": self.region.flags.owndata,
                    "writable": self.region.flags.writeable,
                    "c_contiguous": self.region.flags.c_contiguous,
                    "boundary": (
                        "once during preparation; included in load_and_verify_inputs_seconds; "
                        "excluded from repeated native and Torch operation timings"
                    ),
                },
            },
            "backends": {},
            "checks": {},
            "measurements": [],
        }
        self.monitor = NvmlMonitor(self.base, args)
        self.results["environment"]["nvml_before"] = self.monitor.sample()
        self.field = self.seed = None
        self.prepare_tensor_inputs()
        self.save()

    def prepare_tensor_inputs(self):
        if self.field is None:
            start = time.perf_counter()
            self.field = self.torch.tensor(self.mu, device="cuda", dtype=self.torch.float32)
            self.seed = self.torch.tensor(self.region, device="cuda", dtype=self.torch.float32)
            self.torch.cuda.synchronize()
            self.results["preparation"].setdefault("tensor_input_uploads", []).append(
                {
                    "seconds": time.perf_counter() - start,
                    "prepared_at": self.base.utc(),
                }
            )

    def geometry_probes(self, backend):
        rays = self.rays
        extent = rays["size_xyz"] * rays["spacing_xyz"]
        low, high = -extent / 2, extent / 2
        probe = np.full(self.mu.shape, 0.01, dtype=np.float32)
        observed = backend.host_forward(probe)
        constant = box_comparison(observed, rays, low, high)
        del observed
        first = np.floor(rays["size_xyz"] * [0.15, 0.57, 0.32]).astype(int)
        last = np.floor(rays["size_xyz"] * [0.28, 0.70, 0.48]).astype(int)
        probe.fill(0)
        x0, y0, z0 = first
        x1, y1, z1 = last
        probe[z0:z1, y0:y1, x0:x1] = 0.01
        observed = backend.host_forward(probe)
        marker = box_comparison(
            observed,
            rays,
            low + first * rays["spacing_xyz"],
            low + last * rays["spacing_xyz"],
            centroids=True,
        )
        policy = self.results["geometry_acceptance"]
        passed = (
            constant["relative_l2"] <= policy["constant_relative_l2_max"]
            and marker["relative_l2"] <= policy["marker_relative_l2_max"]
            and max(marker["centroid_distance_pixels"])
            <= policy["marker_centroid_distance_pixels_max"]
        )
        self.results["checks"][backend.label + "/geometry"] = {
            "measured_at": self.base.utc(),
            "passed": passed,
            "constant_box": constant,
            "asymmetric_marker": marker,
            "marker_first_xyz": first.tolist(),
            "marker_end_exclusive_xyz": last.tolist(),
            "reference": "independent exact physical-mm finite-segment box intersection",
            "views_checked": rays["views"],
            "canonical_output_order": "view,row,column",
        }
        self.save()
        if not passed:
            raise ArithmeticError(f"{backend.label} geometry probe failed; no timing admitted")

    def run_backend(self, backend):
        torch = self.torch
        self.results["backends"][backend.label] = backend.metadata
        self.save()
        self.geometry_probes(backend)
        if self.args.probes_only:
            return
        info = backend.metadata
        with torch.no_grad():
            prediction = backend.forward(self.field).detach().cpu().numpy().copy()
        info["image_error_vs_dpt_fp64"] = compare_arrays(prediction, self.reference)
        np.save(self.args.output / f"{backend.label}_depth.npy", prediction, allow_pickle=False)
        # Validate host and tensor boundaries calculate the same operator. No
        # automatic scaling, mirroring, ray reordering or error-driven fitting.
        native = backend.host_forward(self.mu)
        info["host_vs_tensor_image_error"] = compare_arrays(native, prediction)
        if info["host_vs_tensor_image_error"]["relative_l2"] > 1e-6:
            raise ArithmeticError("host/tensor boundary changed the backend's forward operator")
        del native
        common = {
            "backend": backend.label,
            "views": self.rays["views"],
            "detector_shape": list(self.rays["image_shape"]),
            "volume_shape_zyx": list(self.mu.shape),
            "field_dtype": "float32",
            "cotangent": self.results["cotangent"],
            "algorithm": info["algorithm"],
            "sampling": info["sampling"],
            "same_discretisation_as_dpt": False,
        }
        if any(op in self.args.operations for op in ("backprojection", "iteration")):
            gradient = backend.backward(self.seed).detach().cpu().numpy().copy()
            if not np.isfinite(gradient).all():
                raise ArithmeticError("native backprojection returned nonfinite values")
            left = inner_product(prediction, self.region)
            right = inner_product(self.mu, gradient)
            info["adjoint"] = {
                "forward_inner_product": left,
                "backprojection_inner_product": right,
                "relative_defect": abs(left - right) / max(abs(left), abs(right), 1e-300),
                "field": "shared FP32 real attenuation",
                "cotangent": self.results["cotangent"],
                "interpretation": "one measured input/seed pair; not proof of an exact transpose",
            }
            if self.reference_gradient is not None:
                info["backprojection_error_vs_dpt_fp64_vjp"] = compare_arrays(
                    gradient, self.reference_gradient
                )
            np.save(
                self.args.output / f"{backend.label}_backprojection.npy",
                gradient,
                allow_pickle=False,
            )
            native = backend.host_backward(self.region)
            info["host_vs_tensor_backprojection_error"] = compare_arrays(native, gradient)
            if info["host_vs_tensor_backprojection_error"]["relative_l2"] > 1e-6:
                raise ArithmeticError("host/tensor boundary changed the native backprojection")
            del native, gradient
        del prediction
        self.save()
        # Finish tensor boundaries before the native host boundary so its peak
        # memory does not include unused resident copies of all 360 images.
        boundaries = [b for b in ("resident", "torch", "host") if b in self.args.boundaries]
        for boundary in boundaries:
            if boundary == "resident" and info["resident_boundary"] is None:
                info["resident"] = {
                    "status": "unsupported",
                    "reason": info["resident_unsupported_reason"],
                }
                continue
            host = boundary == "host"
            if host:
                self.field = self.seed = None
                if isinstance(backend, AstraFullscan):
                    backend.release_resident_buffers()
                gc.collect()
                torch.cuda.empty_cache()
            info.setdefault("boundary_initial_memory", {})[boundary] = {
                "nvml": self.monitor.sample(),
                "torch_allocated_bytes": torch.cuda.memory_allocated(),
                "torch_reserved_bytes": torch.cuda.memory_reserved(),
                "unused_tensor_inputs_released": host,
            }
            for operation in self.args.operations:
                if operation == "iteration" and boundary != "torch":
                    continue
                if operation == "forward":

                    def perform(host=host):
                        with torch.no_grad():
                            return (
                                backend.host_forward(self.mu)
                                if host
                                else backend.forward(self.field)
                            )
                elif operation == "backprojection":

                    def perform(host=host):
                        return (
                            backend.host_backward(self.region)
                            if host
                            else backend.backward(self.seed)
                        )
                else:
                    field = self.field.detach().requires_grad_(True)

                    def perform(field=field):
                        field.grad = None
                        output = backend.differentiable_forward(field)
                        (output * self.seed).sum().backward()
                        return output

                measured = self.measure(
                    f"{backend.label}/{boundary}/{operation}",
                    perform,
                    {
                        **common,
                        "operation": operation,
                        "boundary": info[boundary + "_boundary"],
                        "requested_gradients": [] if operation == "forward" else ["volume"],
                        "backprojection_is_exact_transpose": None
                        if operation == "forward"
                        else "not assumed; see measured adjoint defect",
                        "state_reset": (
                            "output/gradient overwritten or zeroed; "
                            "Torch leaf gradients set to None per iteration"
                        ),
                    },
                    known_stream=False,
                )
                measured["first_warmup_seconds"] = measured.pop("first_use_in_this_process_seconds")
                measured["cold_start_measured"] = False
                if operation == "iteration":
                    field.grad = None
                    del field
                del perform
        self.save()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Immutable fullscan_inputs output directory"
    )
    parser.add_argument(
        "--cotangent", type=Path, required=True, help="Shared FP32 complete-detector cotangent NPY"
    )
    parser.add_argument(
        "--reference-image", type=Path, help="Actual complete DPT FP64 optical-depth NPY"
    )
    parser.add_argument(
        "--reference-gradient", type=Path, help="Actual DPT FP64 volume VJP for this cotangent"
    )
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output", type=Path, required=True, help="New result directory; never overwrite a run"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        required=True,
        choices=("astra", "tigre-siddon", "tigre-interpolated"),
    )
    parser.add_argument(
        "--boundaries",
        nargs="+",
        default=["resident", "host", "torch"],
        choices=("resident", "host", "torch"),
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        default=["forward", "backprojection", "iteration"],
        choices=("forward", "backprojection", "iteration"),
    )
    parser.add_argument("--astra-supersampling", type=int, nargs="+", default=[1])
    parser.add_argument("--tigre-accuracy", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    parser.add_argument("--tigre-source", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--long-repeats", type=int, default=10)
    parser.add_argument("--long-threshold-seconds", type=float, default=0.2)
    parser.add_argument("--nvml-interval-seconds", type=float, default=0.05)
    parser.add_argument("--nvml-device-index", type=int)
    parser.add_argument("--probes-only", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.repeats, args.long_repeats, *args.astra_supersampling) < 1:
        parser.error("counts must be positive")
    if any(not np.isfinite(x) or not 0 < x <= 1 for x in args.tigre_accuracy):
        parser.error("TIGRE accuracy must be finite and within (0,1]")
    if min(args.long_threshold_seconds, args.nvml_interval_seconds) <= 0:
        parser.error("time intervals must be positive")
    return args


def main():
    args = arguments()
    driver = FullscanDriver(args)
    try:
        for name in args.backends:
            values = (
                args.astra_supersampling
                if name == "astra"
                else (args.tigre_accuracy if name == "tigre-interpolated" else [0.5])
            )
            for value in values:
                backend = None
                try:
                    started = time.perf_counter()
                    driver.prepare_tensor_inputs()
                    backend = (
                        AstraFullscan(driver, value)
                        if name == "astra"
                        else TigreFullscan(
                            driver,
                            "Siddon" if name == "tigre-siddon" else "interpolated",
                            value,
                        )
                    )
                    driver.torch.cuda.synchronize()
                    driver.results["preparation"][backend.label + "_seconds"] = (
                        time.perf_counter() - started
                    )
                    print("START", backend.label, driver.base.utc(), flush=True)
                    driver.run_backend(backend)
                except Exception as error:
                    driver.results.setdefault("failures", []).append(
                        {
                            "backend": name,
                            "parameter": value,
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
