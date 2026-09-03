"""Compare CT projectors using shared millimetre geometry and FP32 storage.

Run one installed backend per process, for example ``python benchmarks/reference.py
--backend astra --size 32 --views 3 --detector 32 --output result.json``. Reference
libraries are installed separately under their own licences. The harness records
each projector's discretisation and measured adjoint defect; its results compare
geometry and workloads, rather than asserting equality of different operators.
"""

from __future__ import annotations

# Dispatch before optional GPU imports so --ct ... --help works on a CPU host.
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks._entry import real_ct_route

    if real_ct_route("reference"):
        raise SystemExit(0)

import argparse
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from dpt.contracts import integer
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.phantoms import circular_cone_beam
from dpt.volumes import GridSpec


@dataclass(frozen=True)
class Case:
    """An n³ centred grid and circular cone-beam panels, with lengths in mm."""

    size: int = 32
    views: int = 3
    detector: int = 32
    repeats: int = 3
    sampling_precision: str = "float64"

    def __post_init__(self) -> None:
        integer(self.size, "size", minimum=8, maximum=512)
        integer(self.views, "views", minimum=1, maximum=32)
        integer(self.detector, "detector", minimum=1, maximum=256)
        integer(self.repeats, "repeats", minimum=1, maximum=10)
        if self.sampling_precision not in ("float64", "mixed_float32"):
            raise ValueError("sampling_precision must be float64 or mixed_float32")

    @property
    def grid(self) -> GridSpec:
        origin = -(self.size - 1) / 2
        return GridSpec(
            (self.size, self.size, self.size), (1.0, 1.0, 1.0), (origin, origin, origin)
        )

    @property
    def panels(self) -> tuple[DetectorGeometry, ...]:
        pitch = 2 * self.size / self.detector
        return circular_cone_beam(
            [2 * math.pi * i / self.views for i in range(self.views)],
            source_radius_mm=3 * self.size,
            detector_radius_mm=3 * self.size,
            shape_rows_columns=(self.detector, self.detector),
            spacing_column_row_mm=(pitch, pitch),
        )


def panel_vectors(panel: DetectorGeometry) -> tuple[float, ...]:
    """ASTRA cone_vec row: source, panel centre, column step, row step."""
    centre = tuple(
        panel.origin_mm[i]
        + 0.5 * (panel.shape[1] - 1) * panel.spacing_mm[0] * panel.u[i]
        + 0.5 * (panel.shape[0] - 1) * panel.spacing_mm[1] * panel.v[i]
        for i in range(3)
    )
    return (
        *panel.source_mm,
        *centre,
        *(panel.spacing_mm[0] * x for x in panel.u),
        *(panel.spacing_mm[1] * x for x in panel.v),
    )


def cube_interval(
    source: tuple[float, ...], target: tuple[float, ...], side: int
) -> tuple[float, float]:
    """Intersect a finite segment parameterised on [0,1] with the centred cube."""
    direction = tuple(b - a for a, b in zip(source, target, strict=True))
    lo, hi = 0.0, 1.0
    for coordinate, step in zip(source, direction, strict=True):
        if step:
            entry, exit = sorted(((-side / 2 - coordinate) / step, (side / 2 - coordinate) / step))
            lo, hi = max(lo, entry), min(hi, exit)
        elif not -side / 2 <= coordinate <= side / 2:
            return 1.0, 0.0
    return lo, hi


def box_depth(source: tuple[float, ...], target: tuple[float, ...], side: int) -> float:
    """Exact unit-density finite-segment integral through the centred cube."""
    lo, hi = cube_interval(source, target, side)
    return math.dist(source, target) * max(0.0, hi - lo)


def gaussian_parameters(side: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Off-centre, anisotropic Gaussian parameters distinguish all coordinate axes."""
    return (0.12 * side, -0.18 * side, 0.08 * side), (0.07 * side, 0.11 * side, 0.09 * side)


def gaussian_depth(source: tuple[float, ...], target: tuple[float, ...], side: int) -> float:
    """Integrate 0.02 exp(-½ Σ((xᵢ-cᵢ)/σᵢ)²) over the segment inside the cube."""
    lo, hi = cube_interval(source, target, side)
    if hi <= lo or source == target:
        return 0.0
    centre, sigma = gaussian_parameters(side)
    origin = tuple((a - c) / s for a, c, s in zip(source, centre, sigma, strict=True))
    direction = tuple((b - a) / s for a, b, s in zip(source, target, sigma, strict=True))
    aa = sum(d * d for d in direction)
    bb = sum(a * d for a, d in zip(origin, direction, strict=True))
    cc = sum(a * a for a in origin)
    scale = math.sqrt(2 * aa)
    integral = (
        math.exp(-0.5 * (cc - bb * bb / aa))
        * math.sqrt(math.pi / (2 * aa))
        * (math.erf((aa * hi + bb) / scale) - math.erf((aa * lo + bb) / scale))
    )
    return 0.02 * math.dist(source, target) * integral


class Operator(Protocol):
    description: str
    version: str
    execution_device: str
    resident_bytes: int | None

    def forward(self, volume: Any) -> Any: ...
    def backward(self, projection: Any) -> Any: ...
    def synchronise(self) -> None: ...
    def close(self) -> None: ...


def reference_completion(device: str) -> Callable[[], None]:
    """Complete the CUDA-zero primary context used by the reference adapters."""
    if device == "cpu":
        return lambda: None
    wp: Any = importlib.import_module("warp")
    selected = wp.get_device(device)
    if not selected.is_cuda or not selected.is_primary:
        raise RuntimeError("Reference completion requires a CUDA primary context")
    return lambda: wp.synchronize_device(selected)


class AstraOperator:
    """ASTRA CUDA Joseph/slice-interpolated FP and its standard BP."""

    description = "ASTRA cuda3d Joseph/slice interpolation; standard voxel-driven BP"
    execution_device = "cuda:0"
    resident_bytes: int | None = None

    def __init__(self, case: Case) -> None:
        self.case = case
        self.np: Any = importlib.import_module("numpy")
        self.astra: Any = importlib.import_module("astra")
        self.version = importlib.metadata.version("astra-toolbox")
        self.synchronise = reference_completion(self.execution_device)
        extent = [-case.size / 2, case.size / 2] * 3
        volume_geometry = self.astra.create_vol_geom(case.size, case.size, case.size, *extent)
        vectors = self.np.asarray([panel_vectors(p) for p in case.panels], dtype=self.np.float64)
        geometry = self.astra.create_proj_geom("cone_vec", case.detector, case.detector, vectors)
        self.projector = self.astra.create_projector("cuda3d", geometry, volume_geometry)
        if not self.astra.use_cuda():
            self.close()
            raise RuntimeError("ASTRA CUDA unavailable")

    def forward(self, volume: Any) -> Any:
        result = self.np.zeros(
            (self.case.detector, self.case.views, self.case.detector), dtype=self.np.float32
        )
        self.astra.projector3d.direct_FP(self.projector, volume, out=result)
        return self.np.ascontiguousarray(result.transpose(1, 0, 2))

    def backward(self, projection: Any) -> Any:
        result = self.np.zeros((self.case.size,) * 3, dtype=self.np.float32)
        native = self.np.ascontiguousarray(projection.transpose(1, 0, 2))
        self.astra.projector3d.direct_BP(self.projector, native, out=result)
        return result

    def close(self) -> None:
        self.astra.projector3d.delete(self.projector)


class LeapOperator:
    """LEAP modular-beam native FP and SF-selected BP on one selected device."""

    description = "LEAP modular-beam native FP; SF-selected library backprojector"
    resident_bytes: int | None = None

    def __init__(self, case: Case, device: str) -> None:
        self.case = case
        self.np: Any = importlib.import_module("numpy")
        module: Any = importlib.import_module("leapctype")
        self.execution_device = device
        self.synchronise = reference_completion(self.execution_device)
        self.model = module.tomographicModels()
        if self.model.libprojectors is None:
            raise RuntimeError("LEAP shared library unavailable")
        self.version = self.model.version()
        self.model.set_gpu(0 if device == "cuda:0" else -1)
        if device == "cuda:0" and self.model.get_gpu() != 0:
            raise RuntimeError("LEAP did not select CUDA device 0")
        vectors = self.np.asarray([panel_vectors(p) for p in case.panels], dtype=self.np.float32)
        pitch = 2 * case.size / case.detector
        configured = self.model.set_modularbeam(
            case.views,
            case.detector,
            case.detector,
            pitch,
            pitch,
            self.np.ascontiguousarray(vectors[:, :3]),
            self.np.ascontiguousarray(vectors[:, 3:6]),
            self.np.ascontiguousarray(vectors[:, 9:12] / pitch),
            self.np.ascontiguousarray(vectors[:, 6:9] / pitch),
        )
        if not configured:
            raise RuntimeError("LEAP rejected the shared geometry")
        if not self.model.set_volumeDimensionOrder(1) or self.model.get_volumeDimensionOrder() != 1:
            raise RuntimeError("LEAP did not select the declared ZYX volume order")
        if not self.model.set_volume(case.size, case.size, case.size, 1.0, 1.0, 0, 0, 0):
            raise RuntimeError("LEAP rejected the shared volume")
        if not self.model.set_projector("SF") or self.model.get_projector() != "SF":
            raise RuntimeError("LEAP did not select the declared SF backprojector")
        # LEAP otherwise derives a circular reconstruction mask from detector
        # coverage. Include every cube corner in the shared support domain.
        if not self.model.set_diameterFOV(2 * case.size):
            raise RuntimeError("LEAP rejected the shared cube support")
        # The Python convenience methods discard the C boolean result. Retain
        # that result so an unsupported CUDA binary cannot produce a false pass.
        pointer = self.np.ctypeslib.ndpointer(self.np.float32, flags="C_CONTIGUOUS")
        for function in (self.model.libprojectors.project, self.model.libprojectors.backproject):
            function.argtypes = [pointer, pointer, ctypes.c_bool]
            function.restype = ctypes.c_bool

    def forward(self, volume: Any) -> Any:
        result = self.np.zeros(
            (self.case.views, self.case.detector, self.case.detector), dtype=self.np.float32
        )
        self.model.set_model()
        if not self.model.libprojectors.project(result, volume, True):
            raise RuntimeError("LEAP forward projection failed")
        return result

    def backward(self, projection: Any) -> Any:
        result = self.np.zeros((self.case.size,) * 3, dtype=self.np.float32)
        self.model.set_model()
        if not self.model.libprojectors.backproject(projection, result, True):
            raise RuntimeError("LEAP backprojection failed")
        return result

    def close(self) -> None:
        self.model.reset()


class TigreOperator:
    """TIGRE interpolated FP and matched BP in its Python [view,row,column] layout."""

    description = "TIGRE Ax(interpolated), Atb(matched), accuracy=0.5 voxel"
    execution_device = "cuda:0"
    resident_bytes: int | None = None

    def __init__(self, case: Case) -> None:
        self.case = case
        self.np: Any = importlib.import_module("numpy")
        self.module: Any = importlib.import_module("tigre")
        self.version = importlib.metadata.version("tigre")
        self.geometry = self.module.geometry()
        self.geometry.mode = "cone"
        self.geometry.nVoxel = self.np.asarray((case.size,) * 3, dtype=self.np.int32)
        self.geometry.dVoxel = self.np.ones(3, dtype=self.np.float32)
        self.geometry.sVoxel = self.np.asarray((case.size,) * 3, dtype=self.np.float32)
        self.geometry.nDetector = self.np.asarray((case.detector,) * 2, dtype=self.np.int32)
        self.geometry.dDetector = self.np.full(
            2, 2 * case.size / case.detector, dtype=self.np.float32
        )
        self.geometry.sDetector = self.geometry.nDetector * self.geometry.dDetector
        self.geometry.DSO = float(3 * case.size)
        self.geometry.DSD = float(6 * case.size)
        self.geometry.accuracy = 0.5
        self.angles = self.np.asarray(
            [2 * math.pi * i / case.views for i in range(case.views)], dtype=self.np.float32
        )
        gpu: Any = importlib.import_module("tigre.utilities.gpu")
        self.gpu_ids = gpu.GpuIds()
        self.gpu_ids.devices = [0]
        self.synchronise = reference_completion(self.execution_device)

    def forward(self, volume: Any) -> Any:
        # TIGRE's CUDA pixelV = nDetecV - v - 1 reverses its internal -z
        # detector basis, so the Python result already increases rows along +z.
        return self.module.Ax(
            volume, self.geometry, self.angles, projection_type="interpolated", gpuids=self.gpu_ids
        )

    def backward(self, projection: Any) -> Any:
        return self.module.Atb(
            projection,
            self.geometry,
            self.angles,
            backprojection_type="matched",
            gpuids=self.gpu_ids,
        )

    def close(self) -> None:
        pass


class DptOperator:
    """DPT per-view midpoint trilinear projection and its discrete VJP."""

    def __init__(self, case: Case, device: str) -> None:
        self.case = case
        self.np: Any = importlib.import_module("numpy")
        self.module: Any = importlib.import_module("dpt.projection")
        self.version = importlib.metadata.version("differentiable-photon-transport")
        self.description = (
            f"DPT trilinear midpoint, {2 * case.size} samples, {case.sampling_precision} sampling"
        )
        self.execution_device = device
        spec = self.module.ProjectionSpec(
            2 * case.size,
            active_pose=False,
            active_volume=True,
            precision="float32",
            field_domain="signed_linear",
            sampling_precision=case.sampling_precision,
        )
        first = self.module.prepare_projection(case.grid, case.panels[0], spec, device=device)
        self.workspaces = [first]
        self.workspaces.extend(
            self.module.prepare_projection(
                case.grid, panel, spec, device=device, stream=first.context.stream
            )
            for panel in case.panels[1:]
        )
        self.context = self.workspaces[0].context
        self.wp = self.context.wp
        with self.context.scope():
            self.pose = self.wp.array(
                RigidTransform().packed(), dtype=self.wp.float64, device=self.context.device
            )
            self.volume = self.wp.empty(
                case.grid.voxels, dtype=self.wp.float32, device=self.context.device
            )
            self.output = self.wp.empty(
                case.detector**2, dtype=self.wp.float32, device=self.context.device
            )
            self.gradient = self.wp.empty(
                case.grid.voxels, dtype=self.wp.float32, device=self.context.device
            )
        self.resident_bytes: int | None = (
            8 * case.grid.voxels
            + 4 * case.detector**2
            + 96
            + sum(workspace.scratch_bytes for workspace in self.workspaces)
        )

    def forward(self, volume: Any) -> Any:
        result = []
        with self.context.scope():
            self.volume.assign(volume.reshape(-1))
        for workspace in self.workspaces:
            self.module.project_optical_depth(
                self.volume, self.pose, out_L=self.output, workspace=workspace, validate=False
            )
            workspace.check_status()
            result.append(self.output.numpy().reshape((self.case.detector,) * 2).copy())
        return self.np.stack(result)

    def backward(self, projection: Any) -> Any:
        with self.context.scope():
            self.gradient.zero_()
        for index, workspace in enumerate(self.workspaces):
            with self.context.scope():
                self.output.assign(projection[index].reshape(-1))
            self.module.projection_vjp(
                self.volume,
                self.pose,
                cotangent=self.output,
                out_mu=self.gradient,
                workspace=workspace,
                validate=False,
                accumulate=True,
                check_volume_gradient=False,
            )
            workspace.check_status()
        result = self.gradient.numpy().reshape((self.case.size,) * 3).copy()
        return result

    def synchronise(self) -> None:
        if self.context.device.is_cuda:
            self.wp.synchronize_stream(self.context.stream)

    def close(self) -> None:
        self.synchronise()


def timed(
    call: Callable[[], Any],
    repeats: int,
    *,
    synchronise: Callable[[], None],
    validate: Callable[[Any], None] | None = None,
) -> tuple[Any, list[float]]:
    """Measure completed host-array calls, including transfers and synchronisation."""
    durations = []
    result = None
    for _ in range(repeats):
        synchronise()
        started = time.perf_counter()
        result = call()
        synchronise()
        elapsed = time.perf_counter() - started
        if validate is not None:
            validate(result)
        durations.append(elapsed)
    return result, durations


def run(backend: str, case: Case, device: str = "cuda:0") -> dict[str, object]:
    """Record independent box error, adjoint/FD checks and completed-call timings."""
    if backend in ("astra", "tigre") and device != "cuda:0":
        raise ValueError(f"{backend} 3D adapter requires cuda:0")
    started = time.perf_counter()
    np: Any = importlib.import_module("numpy")

    def validate_result(result: Any) -> None:
        if not np.isfinite(result).all():
            raise RuntimeError("projector returned non-finite arrays")

    operator: Operator = (
        AstraOperator(case)
        if backend == "astra"
        else TigreOperator(case)
        if backend == "tigre"
        else LeapOperator(case, device)
        if backend == "leap"
        else DptOperator(case, device)
        if backend == "dpt"
        else _unknown_backend(backend)
    )
    try:
        operator.synchronise()
        setup = time.perf_counter() - started
        shape = (case.size,) * 3
        ones = np.ones(shape, dtype=np.float32)
        uniform, cold_forward = timed(
            lambda: operator.forward(ones),
            1,
            synchronise=operator.synchronise,
            validate=validate_result,
        )
        expected = np.asarray(
            [
                [
                    [
                        box_depth(p.source_mm, p.pixel_centre(row, col), case.size)
                        for col in range(case.detector)
                    ]
                    for row in range(case.detector)
                ]
                for p in case.panels
            ],
            dtype=np.float64,
        )
        rng = np.random.default_rng(20261002)
        volume = rng.uniform(0.01, 0.03, size=shape).astype(np.float32)
        seed = rng.normal(size=(case.views, case.detector, case.detector)).astype(np.float32)
        backward, cold_backward = timed(
            lambda: operator.backward(seed),
            1,
            synchronise=operator.synchronise,
            validate=validate_result,
        )
        for _ in range(2):
            operator.synchronise()
            warm_forward = operator.forward(volume)
            operator.synchronise()
            warm_backward = operator.backward(seed)
            operator.synchronise()
            validate_result(warm_forward)
            validate_result(warm_backward)
        forward, forward_seconds = timed(
            lambda: operator.forward(volume),
            case.repeats,
            synchronise=operator.synchronise,
            validate=validate_result,
        )
        backward, backward_seconds = timed(
            lambda: operator.backward(seed),
            case.repeats,
            synchronise=operator.synchronise,
            validate=validate_result,
        )
        lhs = float(np.sum(forward.astype(np.float64) * seed, dtype=np.float64))
        rhs = float(np.sum(volume.astype(np.float64) * backward, dtype=np.float64))
        # This positive scaling direction has an independently known linear derivative.
        step = 0.01
        plus = operator.forward((volume * np.float32(1 + step)).astype(np.float32))
        minus = operator.forward((volume * np.float32(1 - step)).astype(np.float32))
        finite_difference = float(
            np.sum((plus.astype(np.float64) - minus) * seed, dtype=np.float64) / (2 * step)
        )
        if not all(np.isfinite(x).all() for x in (uniform, forward, backward, plus, minus)):
            raise RuntimeError("projector returned non-finite arrays")
        centre, sigma = gaussian_parameters(case.size)
        axis = np.arange(case.size, dtype=np.float32) - (case.size - 1) / 2
        factors = [
            np.exp(-0.5 * ((axis - c) / s) ** 2).astype(np.float32)
            for c, s in zip(centre, sigma, strict=True)
        ]
        gaussian = np.empty(shape, dtype=np.float32)
        plane = np.float32(0.02) * factors[1][:, None] * factors[0][None, :]
        for index in range(case.size):
            gaussian[index] = factors[2][index] * plane
        gaussian_projection = operator.forward(gaussian)
        gaussian_expected = np.asarray(
            [
                [
                    [
                        gaussian_depth(panel.source_mm, panel.pixel_centre(row, col), case.size)
                        for col in range(case.detector)
                    ]
                    for row in range(case.detector)
                ]
                for panel in case.panels
            ],
            dtype=np.float64,
        )
        if not np.isfinite(gaussian_projection).all():
            raise RuntimeError("Gaussian projection contains non-finite entries")
        geometry_hash = hashlib.sha256(
            json.dumps([panel_vectors(p) for p in case.panels]).encode()
        ).hexdigest()
        return {
            "backend": backend,
            "version": operator.version,
            "operator": operator.description,
            "case": asdict(case),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "execution_device": operator.execution_device,
            "geometry_vectors_sha256": geometry_hash,
            "units": {"coordinates": "mm", "volume": "mm^-1", "projection": "dimensionless"},
            "storage_precision": "float32",
            "array_layout": "volume[z,y,x]; projection[view,row,column]",
            "timing_scope": "completed host-array API; allocations, transfers and sync included",
            "completion_api": (
                "synchronous CPU"
                if device == "cpu"
                else "warp.synchronize_stream"
                if backend == "dpt"
                else "warp.synchronize_device (CUDA primary context)"
            ),
            "warmup_calls_per_operation": 2,
            "setup_import_geometry_seconds": setup,
            "cold_forward_seconds": cold_forward[0],
            "cold_backward_seconds": cold_backward[0],
            "warm_forward_seconds": forward_seconds,
            "warm_backward_seconds": backward_seconds,
            "median_forward_seconds": statistics.median(forward_seconds),
            "median_backward_seconds": statistics.median(backward_seconds),
            "forward_rays_per_second": case.views
            * case.detector**2
            / statistics.median(forward_seconds),
            "input_output_bytes": 4 * (case.size**3 + case.views * case.detector**2),
            "known_resident_bytes": operator.resident_bytes,
            "maximum_absolute_box_error": float(np.max(np.abs(uniform - expected))),
            "relative_l2_box_error": float(
                np.linalg.norm(uniform - expected) / np.linalg.norm(expected)
            ),
            "maximum_absolute_gaussian_error": float(
                np.max(np.abs(gaussian_projection - gaussian_expected))
            ),
            "relative_l2_gaussian_error": float(
                np.linalg.norm(gaussian_projection - gaussian_expected)
                / np.linalg.norm(gaussian_expected)
            ),
            "adjoint_lhs": lhs,
            "adjoint_rhs": rhs,
            "relative_adjoint_defect": abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30),
            "volume_scaling_fd": finite_difference,
            "relative_forward_fd_error": abs(finite_difference - lhs) / max(abs(lhs), 1e-30),
            "relative_backprojection_gradient_error": abs(finite_difference - rhs)
            / max(abs(finite_difference), abs(rhs), 1e-30),
            "discretisation_match": (
                "geometry and storage match; each named integration/BP rule is measured separately"
            ),
        }
    finally:
        operator.close()


def _unknown_backend(name: str) -> Operator:
    raise ValueError(f"unknown backend {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("dpt", "astra", "leap", "tigre"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--detector", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--sampling-precision", choices=("float64", "mixed_float32"), default="float64"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.backend,
        Case(args.size, args.views, args.detector, args.repeats, args.sampling_precision),
        args.device,
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
