"""Multi-view spectral pose recovery with constrained shared acquisition parameters.

This adapter composes the canonical material projector, spectral response,
spatial response, calibration and objective. The optimiser sees a fixed SE(3)
chart followed by named nuisance blocks. Detector-sized data stay on CUDA;
pinned control buffers move pose, nuisance values, losses and gradients.
One prepared completion payload transfers losses, pose/nuisance gradients and
workspace status words; a separate entry checkpoint preserves pending errors.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    import warp as wp

from dpt._runtime import CompletionReadback, StatusCheckpoint, prepare_context, require_no_tape
from dpt.contracts import (
    ContractError,
    NumericalError,
    TrialDomainError,
    binary32_scalar,
    finite_scalar,
    integer,
)
from dpt.detector import (
    BlurSpec,
    BlurWorkspace,
    CalibrationSpec,
    DetectorWorkspace,
    SeparableBlurSpec,
    blur,
    blur_transpose,
    calibrate,
    calibration_vjp,
    prepare_blur,
    prepare_detector,
)
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_projection import (
    MaterialProjectionWorkspace,
    material_projection_vjp,
    prepare_material_projection,
    project_material_paths,
)
from dpt.materials import Provenance
from dpt.objectives import ObjectiveSpec, ObjectiveWorkspace, evaluate_objective, prepare_objective
from dpt.projection import ProjectionSpec
from dpt.recovery import PoseChart
from dpt.registration import (
    Evaluation,
    RecoveryCheckpoint,
    RecoveryPolicy,
    RecoveryResult,
    Vector,
    recover_parameters,
)
from dpt.solver_lifecycle import LifecycleObserver
from dpt.spectral import (
    SpectralSpec,
    SpectralWorkspace,
    prepare_spectral,
    spectral_signal,
    spectral_vjp,
)
from dpt.volumes import GridSpec

# Public Python boundaries validate runtime inputs.
# pyright: reportUnnecessaryIsInstance=false


def _physical_parameter(value: float, name: str, *, positive: bool = False) -> float:
    """Translate shared scalar storage checks into rejectable trial-domain errors."""
    try:
        rounded = binary32_scalar(value, f"trial {name}")
    except ContractError as error:
        raise TrialDomainError(str(error)) from error
    if positive and rounded <= 0:
        raise TrialDomainError(f"trial {name} lies outside its representable physical domain")
    return rounded


def _scaled_product(*values: float) -> float:
    """Preserve exponent range until the final binary64 chart product is formed."""
    mantissa, exponent = 1.0, 0
    for value in values:
        if not math.isfinite(value):
            raise NumericalError("non-finite physical derivative or chart scale")
        part, power = math.frexp(value)
        mantissa *= part
        exponent += power
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError as error:
        raise NumericalError("chart derivative overflows binary64") from error


@dataclass(frozen=True, slots=True)
class CalibrationBlock:
    """Named acquisition group sharing one gain/exposure/offset across its views.

    Exactly one multiplicative scale may be fitted. Its positive chart is
    reference*exp(scale_step*z); the other scale is fixed to remove the gain /
    exposure gauge. An active offset is reference+offset_step*z in output units.
    Optional Gaussian priors act on these dimensionless chart coordinates.

    Device parameters are rounded to binary32. The VJP differentiates the smooth
    physical chart at those represented values, not the discontinuous rounding
    map. Shared physical derivatives stay binary64 until the chart product is
    formed, so a large positive scale can preserve a small logarithmic partial.
    """

    name: str
    gain: float = 1.0
    exposure: float = 1.0
    offset: float = 0.0
    fit_scale: Literal["none", "gain", "exposure"] = "none"
    fit_offset: bool = False
    scale_step: float = 1.0
    offset_step: float = 1.0
    scale_prior_precision: float = 0.0
    offset_prior_precision: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ContractError("calibration groups need non-empty names")
        if self.fit_scale not in ("none", "gain", "exposure") or type(self.fit_offset) is not bool:
            raise ContractError("invalid calibration activity specification")
        for name in ("gain", "exposure", "scale_step", "offset_step"):
            if finite_scalar(getattr(self, name), name, minimum=0) <= 0:
                raise ContractError(f"{name} must be strictly positive in the recovery chart")
        finite_scalar(self.offset, "offset")
        for name in ("scale_prior_precision", "offset_prior_precision"):
            finite_scalar(getattr(self, name), name, minimum=0)
        if self.fit_scale == "none" and self.scale_prior_precision:
            raise ContractError("a scale prior requires an active scale")
        if not self.fit_offset and self.offset_prior_precision:
            raise ContractError("an offset prior requires an active offset")
        for name in ("gain", "exposure", "offset"):
            try:
                _physical_parameter(getattr(self, name), name, positive=name != "offset")
            except NumericalError as error:
                raise ContractError(str(error)) from error

    @property
    def dimension(self) -> int:
        return int(self.fit_scale != "none") + int(self.fit_offset)

    def decode(self, coordinates: Vector) -> tuple[float, float, float]:
        if len(coordinates) != self.dimension:
            raise ContractError("nuisance coordinates differ from the declared group chart")
        if not all(math.isfinite(value) for value in coordinates):
            raise TrialDomainError("trial nuisance coordinates are non-finite")
        gain, exposure, offset = self.gain, self.exposure, self.offset
        cursor = 0
        if self.fit_scale != "none":
            try:
                scale = math.exp(self.scale_step * coordinates[cursor])
            except OverflowError as error:
                raise TrialDomainError("trial logarithmic calibration scale overflowed") from error
            if self.fit_scale == "gain":
                gain *= scale
            else:
                exposure *= scale
            cursor += 1
        if self.fit_offset:
            offset += self.offset_step * coordinates[cursor]
        return (
            _physical_parameter(gain, "gain", positive=True),
            _physical_parameter(exposure, "exposure", positive=True),
            _physical_parameter(offset, "offset"),
        )

    def gradient(self, coordinates: Vector, physical_gradient: Vector) -> Vector:
        if len(physical_gradient) != 3:
            raise ContractError("a calibration gradient needs gain, exposure and offset partials")
        gain, exposure, _ = self.decode(coordinates)
        gradient: list[float] = []
        if self.fit_scale != "none":
            index, value = (0, gain) if self.fit_scale == "gain" else (1, exposure)
            gradient.append(_scaled_product(self.scale_step, value, physical_gradient[index]))
        if self.fit_offset:
            gradient.append(_scaled_product(self.offset_step, physical_gradient[2]))
        return tuple(gradient)

    def prior(self, coordinates: Vector) -> Evaluation:
        if len(coordinates) != self.dimension:
            raise ContractError("prior coordinates differ from calibration chart")
        precisions = () if self.fit_scale == "none" else (self.scale_prior_precision,)
        precisions += (self.offset_prior_precision,) if self.fit_offset else ()
        return Evaluation(
            math.fsum(0.5 * p * z * z for p, z in zip(precisions, coordinates, strict=True)),
            tuple(p * z for p, z in zip(precisions, coordinates, strict=True)),
        )


@dataclass(frozen=True, slots=True)
class SpectralView:
    name: str
    geometry: DetectorGeometry
    spectral: SpectralSpec
    coefficients: Any
    weights: Any
    response: Any
    observation: Any
    calibration_group: str
    observation_provenance: Provenance | None = None
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    objective_weights: Any = None
    objective_weight: float = 1.0
    spatial_response: BlurSpec | SeparableBlurSpec | None = None

    def __post_init__(self) -> None:
        if self.spectral.output_domain != "linear":
            raise ContractError("spectral pose recovery requires linear spectral signal")
        if self.spectral.precision != "float32" or self.objective.precision != "float32":
            raise ContractError(
                "spectral pose recovery requires float32 spectral and objective precision; "
                "its calibration and optional spatial-response buffers are binary32"
            )
        if self.observation_provenance is not None and not isinstance(
            self.observation_provenance, Provenance
        ):
            raise ContractError("each observation requires acquisition provenance")
        if not self.name.strip() or not self.calibration_group.strip():
            raise ContractError("view and calibration-group names are required")
        if finite_scalar(self.objective_weight, "view objective weight", minimum=0) <= 0:
            raise ContractError("view objective weight must be strictly positive")
        if self.objective.domain == "log_transmission":
            raise ContractError(
                "spectral recovery compares calibrated signal or counts, not log data"
            )
        if (
            self.spatial_response is not None
            and (self.spatial_response.height, self.spatial_response.width) != self.geometry.shape
        ):
            raise ContractError("spatial response and detector shapes differ")
        if not self.spectral.active_paths:
            raise ContractError("spectral pose recovery needs active material paths")
        if (
            self.spectral.active_weights
            or self.spectral.active_response
            or self.spectral.active_coefficients
        ):
            raise ContractError(
                "this recovery adapter keeps spectrum, response and coefficients fixed"
            )
        if not self.objective.weighted and self.objective_weights is not None:
            raise ContractError("objective weights require an explicitly weighted objective")


@dataclass(frozen=True, slots=True)
class SpectralPoseProblem:
    grid: GridSpec
    materials: int
    fields: Any
    fields_provenance: Provenance | None
    views: tuple[SpectralView, ...]
    calibration: tuple[CalibrationBlock, ...]
    samples_per_ray: int = 256

    def __post_init__(self) -> None:
        integer(self.materials, "materials", minimum=1, maximum=32)
        integer(self.samples_per_ray, "samples_per_ray", minimum=1)
        if self.fields_provenance is not None and not isinstance(
            self.fields_provenance, Provenance
        ):
            raise ContractError("fixed material fields require provenance")
        if not isinstance(self.views, tuple) or not self.views:
            raise ContractError(
                "a spectral recovery problem needs an immutable non-empty view tuple"
            )
        if not isinstance(self.calibration, tuple) or not self.calibration:
            raise ContractError("calibration blocks must be an immutable non-empty tuple")
        if len({view.name for view in self.views}) != len(self.views):
            raise ContractError("view names must be unique")
        groups = {block.name: block for block in self.calibration}
        if len(groups) != len(self.calibration):
            raise ContractError("calibration group names must be unique")
        if set(groups) != {view.calibration_group for view in self.views}:
            raise ContractError("every named calibration group must be defined and used")
        for group in groups:
            units = {
                view.spectral.output_unit for view in self.views if view.calibration_group == group
            }
            if len(units) != 1:
                raise ContractError(
                    "shared calibration groups require identical detector output units"
                )
        for view in self.views:
            if view.objective.domain == "counts" and view.spectral.output_unit != "counts":
                raise ContractError("count-domain views must declare spectral output_unit='counts'")
            if view.spectral.materials != self.materials:
                raise ContractError("all views must use the same fixed material basis")
            block = groups[view.calibration_group]
            if view.objective.kind == "poisson" and (
                block.gain != 1.0
                or block.offset != 0.0
                or block.fit_scale == "gain"
                or block.fit_offset
                or view.spatial_response is not None
            ):
                raise ContractError(
                    "Poisson counts require unit gain, zero offset and no spatial blur"
                )


@dataclass(slots=True)
class _PreparedView:
    view: SpectralView
    group: int
    material: MaterialProjectionWorkspace
    spectral: SpectralWorkspace
    detector: DetectorWorkspace
    objective: ObjectiveWorkspace
    blur: BlurWorkspace | None
    calibration_spec: CalibrationSpec
    mean: Any
    spread: Any
    prediction: Any
    image_seed: Any
    spread_seed: Any
    mean_seed: Any
    loss: Any
    pose_gradient: Any
    nuisance_gradients: tuple[Any, Any, Any]
    coefficients: Any
    weights: Any
    response: Any
    observation: Any
    objective_weights: Any


@dataclass(frozen=True, slots=True)
class CalibrationEstimate:
    name: str
    gain: float
    exposure: float
    offset: float


@dataclass(frozen=True, slots=True)
class SpectralRecoveryResult:
    pose: RigidTransform
    calibration: tuple[CalibrationEstimate, ...]
    optimisation: RecoveryResult


class SpectralPoseEvaluator:
    """Prepared single-stream multi-view objective with manual first-order products.

    Fields, input spectra, responses, observations and weights are immutable for
    this object's lifetime. Contiguous declared axes or flat device inputs are
    bound to persistent no-copy flat views during preparation. This accepts
    public (M,E) basis uploads, (M,N)/(M,Z,Y,X) fields, (E,P)/(E,H,W)
    per-pixel spectral factors and (H,W) images. Preparation checks their values.
    Numerical flags are checked through each workspace after evaluation. Every path,
    image and adjoint buffer is allocated once. Concurrent/re-entrant evaluation
    and ambient tape recording are rejected before any staging buffer is changed.
    ``execution="graph"`` captures the prepared multi-view forward/VJP chain on
    CUDA; chart decoding, host staging and completion checks remain outside it.
    """

    def __init__(
        self,
        problem: SpectralPoseProblem,
        chart: PoseChart,
        *,
        device: str = "cuda:0",
        stream: Any = None,
        execution: Literal["eager", "graph"] = "eager",
    ) -> None:
        require_no_tape()
        if execution not in ("eager", "graph"):
            raise ContractError("execution must be eager or graph")
        self.execution = execution
        self._evaluation_graph: Any = None
        self.problem, self.chart = problem, chart
        self.context = prepare_context(device=device, stream=stream)
        ctx, wp = self.context, self.context.wp
        if execution == "graph" and not ctx.device.is_cuda:
            raise ContractError("graph execution requires CUDA")
        self._fields = self._flat_input(
            problem.fields,
            "fields",
            ((problem.materials, problem.grid.voxels), (problem.materials, *problem.grid.shape)),
        )
        self._group_slices: tuple[slice, ...] = self._parameter_slices()
        self.dimension = 6 + sum(block.dimension for block in problem.calibration)
        self._running = False
        self._evaluation_lock = Lock()
        self.last_view_losses: tuple[float, ...] | None = None
        views = len(problem.views)
        self._maximum_pixels = max(view.geometry.pixels for view in problem.views)
        self._spectral_cache: dict[SpectralSpec, SpectralWorkspace] = {}
        self._objective_cache: dict[ObjectiveSpec, ObjectiveWorkspace] = {}
        self._blur_cache: dict[BlurSpec | SeparableBlurSpec, BlurWorkspace] = {}
        with ctx.scope():
            self._shared_paths = wp.empty(
                problem.materials * self._maximum_pixels, dtype=wp.float32, device=ctx.device
            )
            self._shared_adj_paths = wp.empty_like(self._shared_paths)
            self._shared_images = tuple(
                wp.empty(self._maximum_pixels, dtype=wp.float32, device=ctx.device)
                for _ in range(5)
            )
            self._pose_device = wp.empty(12, dtype=wp.float64, device=ctx.device)
            self._parameters_device = wp.empty(
                3 * len(problem.calibration), dtype=wp.float32, device=ctx.device
            )
            self._control_device = wp.empty(7 * views, dtype=wp.float64, device=ctx.device)
            self._nuisance_device = wp.zeros(3 * views, dtype=wp.float64, device=ctx.device)
        self._parameter_views = tuple(
            tuple(self._parameters_device[3 * i + j : 3 * i + j + 1] for j in range(3))
            for i in range(len(problem.calibration))
        )
        self._pose_host = wp.empty(12, dtype=wp.float64, device="cpu", pinned=ctx.device.is_cuda)
        self._parameters_host = wp.empty(
            3 * len(problem.calibration),
            dtype=wp.float32,
            device="cpu",
            pinned=ctx.device.is_cuda,
        )
        self._pose_values = self._pose_host.numpy()
        self._parameter_values = self._parameters_host.numpy()
        self._control_values: Any = None
        self._nuisance_values: Any = None
        self._shared_detector = prepare_detector(
            max_pixels=self._maximum_pixels, device=ctx.device, stream=ctx.stream
        )
        self._prepared = tuple(
            self._prepare_view(index, view) for index, view in enumerate(problem.views)
        )
        self._status_checkpoint = StatusCheckpoint(ctx, self._workspaces())
        self._completion_readback = CompletionReadback(
            self._status_checkpoint, (self._control_device, self._nuisance_device)
        )
        self._validate_fixed_inputs()

    def _parameter_slices(self) -> tuple[slice, ...]:
        cursor = 6
        slices: list[slice] = []
        for block in self.problem.calibration:
            slices.append(slice(cursor, cursor + block.dimension))
            cursor += block.dimension
        return tuple(slices)

    def _flat_input(self, values: Any, name: str, shapes: tuple[tuple[int, ...], ...]) -> Any:
        """Retain a no-copy flat device binding for declared contiguous axes."""
        ctx = self.context
        for shape in shapes:
            if getattr(values, "shape", None) == shape:
                ctx.array(values, name, dtype=ctx.wp.float32, shape=shape, ndim=len(shape))
                return values.flatten() if len(shape) > 1 else values
        ctx.array(values, name, dtype=ctx.wp.float32, shape=(math.prod(shapes[0]),))
        return values

    def _prepare_view(self, index: int, view: SpectralView) -> _PreparedView:
        ctx, wp = self.context, self.context.wp
        group = next(
            i
            for i, block in enumerate(self.problem.calibration)
            if block.name == view.calibration_group
        )
        block = self.problem.calibration[group]
        pixels, spec = view.geometry.pixels, view.spectral
        coefficients = self._flat_input(
            view.coefficients, "coefficients", ((spec.materials, spec.energies),)
        )
        weights = self._flat_input(
            view.weights,
            "weights",
            ((spec.energies,),)
            if spec.shared_weights
            else ((spec.energies, pixels), (spec.energies, *view.geometry.shape)),
        )
        response = self._flat_input(
            view.response,
            "response",
            ((spec.energies,),)
            if spec.shared_response
            else ((spec.energies, pixels), (spec.energies, *view.geometry.shape)),
        )
        observation = self._flat_input(view.observation, "observation", (view.geometry.shape,))
        objective_weights = None
        if view.objective.weighted:
            objective_weights = self._flat_input(
                view.objective_weights, "objective_weights", (view.geometry.shape,)
            )
        with ctx.scope():
            paths = self._shared_paths[: spec.materials * pixels]
            adj_paths = self._shared_adj_paths[: spec.materials * pixels]
            mean, mean_seed, blur_signal, blur_seed, image_seed = tuple(
                values[:pixels] for values in self._shared_images
            )
            spread = blur_signal if view.spatial_response is not None else mean
            spread_seed = blur_seed if view.spatial_response is not None else mean_seed
            # Public predicted_images retains every current view. Other image
            # scratch is consumed completely before advancing to the next view.
            prediction = wp.empty(pixels, dtype=wp.float32, device=ctx.device)
        material = prepare_material_projection(
            self.problem.grid,
            view.geometry,
            ProjectionSpec(self.problem.samples_per_ray),
            materials=self.problem.materials,
            fields=self._fields,
            out_paths=paths,
            cotangent_paths=adj_paths,
            device=ctx.device,
            stream=ctx.stream,
        )
        if spec not in self._spectral_cache:
            self._spectral_cache[spec] = prepare_spectral(
                spec, max_pixels=self._maximum_pixels, device=ctx.device, stream=ctx.stream
            )
        spectral = self._spectral_cache[spec]
        detector = self._shared_detector
        if view.objective not in self._objective_cache:
            self._objective_cache[view.objective] = prepare_objective(
                view.objective,
                max_pixels=self._maximum_pixels,
                device=ctx.device,
                stream=ctx.stream,
            )
        objective = self._objective_cache[view.objective]
        spatial = None
        if view.spatial_response is not None:
            if view.spatial_response not in self._blur_cache:
                self._blur_cache[view.spatial_response] = prepare_blur(
                    view.spatial_response, device=ctx.device, stream=ctx.stream
                )
            spatial = self._blur_cache[view.spatial_response]
        calibration_spec = CalibrationSpec(
            spec.output_unit,
            active_gain=block.fit_scale == "gain",
            active_exposure=block.fit_scale == "exposure",
            active_offset=block.fit_offset,
        )
        return _PreparedView(
            view,
            group,
            material,
            spectral,
            detector,
            objective,
            spatial,
            calibration_spec,
            mean,
            spread,
            prediction,
            image_seed,
            spread_seed,
            mean_seed,
            self._control_device[7 * index : 7 * index + 1],
            self._control_device[7 * index + 1 : 7 * index + 7],
            tuple(self._nuisance_device[3 * index + j : 3 * index + j + 1] for j in range(3)),
            coefficients,
            weights,
            response,
            observation,
            objective_weights,
        )

    def _workspaces(
        self,
    ) -> tuple[
        MaterialProjectionWorkspace
        | SpectralWorkspace
        | DetectorWorkspace
        | ObjectiveWorkspace
        | BlurWorkspace,
        ...,
    ]:
        """Use public workspace diagnostics; status encodings stay with operators."""
        result: list[
            MaterialProjectionWorkspace
            | SpectralWorkspace
            | DetectorWorkspace
            | ObjectiveWorkspace
            | BlurWorkspace
        ] = []
        seen: set[int] = set()
        for item in self._prepared:
            for workspace in (
                item.material,
                item.spectral,
                item.detector,
                item.objective,
                item.blur,
            ):
                if workspace is not None and id(workspace) not in seen:
                    seen.add(id(workspace))
                    result.append(workspace)
        return tuple(result)

    def _clear_statuses(self) -> None:
        self._status_checkpoint.consume()

    def _validate_fixed_inputs(self) -> None:
        for item in self._prepared:
            try:
                item.material.validate_inputs(stream=self.context.stream)
            except TrialDomainError as error:
                # These fields are immutable fitting inputs, never trial parameters.
                raise ContractError(f"invalid fixed material fields: {error}") from error
            item.spectral.validate_inputs(
                item.coefficients,
                item.weights,
                item.response,
                pixels=item.view.geometry.pixels,
                probability_response=item.view.objective.kind == "poisson",
                stream=self.context.stream,
            )
            item.objective.validate_observation(
                item.observation,
                weights=item.objective_weights,
                stream=self.context.stream,
            )

    def __call__(self, parameters: Vector) -> Evaluation:
        """f(x,z)=sum_views
        objective(calibrate(blur(spectral(project(f,T(x))))),observation)+priors(z).

        Parameters is a dimensionless binary64 tuple of length 6 plus the declared
        nuisance-block dimensions. The first six coordinates use the fixed PoseChart
        with scaled mm/rad Lie coordinates and the remaining coordinates follow
        calibration-block order. Fixed fields are FP32 (M,voxels), coefficients
        (M,K) in mm^-1, weights/response (K,) or (K,P), and observations/weights
        (P,) per view. Spectral, calibration and optional spatial-response buffers
        retain FP32 and scalar loss/shared-gradient controls retain FP64.
        Evaluation returns loss and a matching fixed-chart gradient tuple. Active
        nuisance scales use the declared exponential chart and offsets use its
        linear chart, including supplied Gaussian priors. Fixed acquisition/material
        parameters are not differentiated. Device images stay resident and only
        small controls cross the host boundary. Calls require sequential ownership.

        """
        require_no_tape()
        if not self._evaluation_lock.acquire(blocking=False):
            raise ContractError("a spectral evaluator cannot be used concurrently or recursively")
        try:
            return self._evaluate(parameters)
        finally:
            self._evaluation_lock.release()

    def _evaluate(self, parameters: Vector) -> Evaluation:
        if self._running:
            raise ContractError("a spectral evaluator cannot be used concurrently or recursively")
        if len(parameters) != self.dimension:
            raise ContractError("parameter count differs from pose and calibration charts")
        pose = self.chart.pose(parameters[:6])
        decoded = tuple(
            block.decode(parameters[section])
            for block, section in zip(self.problem.calibration, self._group_slices, strict=True)
        )
        ctx, wp = self.context, self.context.wp
        self._running = True
        self.last_view_losses = None
        try:
            with ctx.scope():
                staging_complete = False
                try:
                    self._clear_statuses()
                    self._pose_values[:] = pose.packed()
                    self._parameter_values[:] = tuple(value for block in decoded for value in block)
                    wp.copy(self._pose_device, self._pose_host, stream=ctx.stream)
                    wp.copy(self._parameters_device, self._parameters_host, stream=ctx.stream)
                    if self.execution == "eager":
                        self._launch_evaluation()
                    else:
                        if self._evaluation_graph is None:
                            self._launch_evaluation()
                            self._status_checkpoint.consume()
                            with wp.ScopedCapture(device=ctx.device, stream=ctx.stream) as capture:
                                self._launch_evaluation()
                            self._evaluation_graph = capture.graph
                        wp.capture_launch(self._evaluation_graph, stream=ctx.stream)
                    self._control_values, self._nuisance_values = (
                        self._completion_readback.consume()
                    )
                    staging_complete = True
                finally:
                    # Completion already waited on success. Failed launches must
                    # finish pinned reads before the next trial changes staging.
                    if not staging_complete and ctx.device.is_cuda:
                        wp.synchronize_stream(ctx.stream)
            return self._assemble(parameters)
        finally:
            self._running = False

    def _launch_evaluation(self) -> None:
        """Run the fixed-pointer forward/VJP chain; staging and readback stay outside."""
        ctx = self.context
        for item in self._prepared:
            gain, exposure, offset = self._parameter_views[item.group]
            project_material_paths(
                self._pose_device,
                workspace=item.material,
                stream=ctx.stream,
                validate=False,
            )
            spectral_signal(
                item.material.paths,
                item.coefficients,
                item.weights,
                item.response,
                out_mean=item.mean,
                workspace=item.spectral,
                stream=ctx.stream,
                validate=False,
            )
            if item.blur is not None:
                blur(
                    item.mean,
                    out_signal=item.spread,
                    workspace=item.blur,
                    stream=ctx.stream,
                    validate=False,
                )
            calibrate(
                item.spread,
                gain,
                exposure,
                offset,
                out_signal=item.prediction,
                spec=item.calibration_spec,
                workspace=item.detector,
                stream=ctx.stream,
                validate=False,
            )
            evaluate_objective(
                item.prediction,
                item.observation,
                out_loss=item.loss,
                out_cotangent=item.image_seed,
                weights=item.objective_weights,
                workspace=item.objective,
                stream=ctx.stream,
                validate=False,
            )
            calibration_vjp(
                item.spread,
                gain,
                exposure,
                offset,
                cotangent=item.image_seed,
                out_grad_mean=item.spread_seed,
                out_grad_gain=item.nuisance_gradients[0]
                if item.calibration_spec.active_gain
                else None,
                out_grad_exposure=item.nuisance_gradients[1]
                if item.calibration_spec.active_exposure
                else None,
                out_grad_offset=item.nuisance_gradients[2]
                if item.calibration_spec.active_offset
                else None,
                spec=item.calibration_spec,
                workspace=item.detector,
                stream=ctx.stream,
                validate=False,
            )
            if item.blur is not None:
                blur_transpose(
                    item.spread_seed,
                    out_grad_signal=item.mean_seed,
                    workspace=item.blur,
                    stream=ctx.stream,
                    validate=False,
                )
            spectral_vjp(
                item.material.paths,
                item.coefficients,
                item.weights,
                item.response,
                cotangent=item.mean_seed,
                out_grad_paths=item.material.cotangent_paths,
                workspace=item.spectral,
                stream=ctx.stream,
                validate=False,
            )
            material_projection_vjp(
                self._pose_device,
                workspace=item.material,
                out_pose_tangent=item.pose_gradient,
                stream=ctx.stream,
                validate=False,
            )

    def _assemble(self, parameters: Vector) -> Evaluation:
        losses = tuple(float(self._control_values[7 * i]) for i in range(len(self._prepared)))
        pose_gradient = tuple(
            math.fsum(
                item.view.objective_weight * float(self._control_values[7 * i + 1 + j])
                for i, item in enumerate(self._prepared)
            )
            for j in range(6)
        )
        gradient = list(self.chart.gradient(parameters[:6], pose_gradient))
        prior_losses: list[float] = []
        for group, (block, section) in enumerate(
            zip(self.problem.calibration, self._group_slices, strict=True)
        ):
            physical = tuple(
                math.fsum(
                    item.view.objective_weight * float(self._nuisance_values[3 * i + j])
                    for i, item in enumerate(self._prepared)
                    if item.group == group
                )
                for j in range(3)
            )
            prior = block.prior(parameters[section])
            prior_losses.append(prior.loss)
            derivative = block.gradient(parameters[section], physical)
            gradient.extend(a + b for a, b in zip(derivative, prior.gradient, strict=True))
        loss = math.fsum(
            [
                *(
                    item.view.objective_weight * value
                    for item, value in zip(self._prepared, losses, strict=True)
                ),
                *prior_losses,
            ]
        )
        if not math.isfinite(loss) or not all(math.isfinite(value) for value in gradient):
            raise NumericalError("multi-view loss or chart gradient is non-finite")
        self.last_view_losses = losses
        return Evaluation(loss, tuple(gradient))

    def calibration_estimates(self, parameters: Vector) -> tuple[CalibrationEstimate, ...]:
        if len(parameters) != self.dimension:
            raise ContractError("parameter count differs from evaluator chart")
        return tuple(
            CalibrationEstimate(block.name, *block.decode(parameters[section]))
            for block, section in zip(self.problem.calibration, self._group_slices, strict=True)
        )

    @property
    def predicted_images(self) -> tuple[wp.array[wp.float32] | wp.array[wp.float64], ...]:
        """Borrow the current device predictions; they may belong to a rejected trial.

        Re-evaluate accepted parameters before explicit export. No host copy is
        made by this accessor, and callers must not mutate these borrowed buffers.
        """
        return cast(
            "tuple[wp.array[wp.float32] | wp.array[wp.float64], ...]",
            tuple(item.prediction for item in self._prepared),
        )


def recover_spectral_pose(
    evaluator: SpectralPoseEvaluator,
    *,
    initial: Vector | None = None,
    policy: RecoveryPolicy | None = None,
    objective_id: str | None = None,
    checkpoint: RecoveryCheckpoint | None = None,
    save_checkpoint: Callable[[RecoveryCheckpoint], None] | None = None,
    checkpoint_relative_tolerance: float = 0.0,
    observe_lifecycle: LifecycleObserver | None = None,
) -> SpectralRecoveryResult:
    """(T,z)_accepted minimises the spectral objective in the fixed pose/nuisance chart.

    Initial is a dimensionless binary64 tuple of length 6 plus the prepared
    nuisance dimensions, or is recovered from a verified checkpoint. The
    SpectralPoseEvaluator holds fixed fields, acquisitions, factors and
    observations and returns scalar loss with a matching chart gradient.
    RecoveryPolicy selects the same safeguarded deterministic driver as primary
    recovery. SpectralRecoveryResult returns the accepted RigidTransform in world
    mm, physical CalibrationEstimate records and the optimisation result.
    Multiplicative nuisance parameters use the declared exponential chart and
    offsets use its linear chart. No chart rebasing occurs. Reevaluate accepted
    parameters to refresh reusable device prediction buffers before exporting.

    """
    start = (0.0,) * evaluator.dimension if initial is None and checkpoint is None else initial
    optimisation = recover_parameters(
        evaluator,
        start,
        policy=policy,
        objective_id=objective_id,
        checkpoint=checkpoint,
        save_checkpoint=save_checkpoint,
        checkpoint_relative_tolerance=checkpoint_relative_tolerance,
        observe_lifecycle=observe_lifecycle,
    )
    return SpectralRecoveryResult(
        evaluator.chart.pose(optimisation.parameters[:6]),
        evaluator.calibration_estimates(optimisation.parameters),
        optimisation,
    )
