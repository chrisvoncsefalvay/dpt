"""Concrete CUDA oracle for squared-expected-signal recovery in a fixed chart.

The controller sees only small parameter/gradient vectors and scalar objective
checkpoints. Source histories, transport, detector sums, independent estimator
products and objective reductions remain on one CUDA stream. No Python history
loop, full image download or per-replicate device allocation is used.
"""

# Internal workspaces are composed by this package, never exposed as a backend API.
# pyright: reportPrivateUsage=false
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from dpt._runtime import HostReadback, load_kernels, require_no_tape
from dpt.contracts import NumericalError, TrialDomainError, finite_scalar, integer
from dpt.objectives import (
    ObjectiveSpec,
    ObjectiveWorkspace,
    prepare_objective,
    reduce_objective_components,
)
from dpt.registration import Vector

from .derivatives import (
    FUSED_PARAMETER_LIMIT,
    TransportParameter,
    derivative_columns,
    derivative_histories,
)
from .estimators import (
    EstimatorWorkspace,
    history_mean,
    independent_squared_gradient,
    independent_squared_loss,
    prepare_estimators,
)
from .forward import TransportWorkspace, trace_histories
from .model import TransportError
from .rng import HistoryBatch, require_independent
from .source import ParallelBeam as ParallelBeam
from .source import PointConeSource, sample_parallel_beam, sample_point_cone


@dataclass(slots=True)
class TransportSquaredOracle:
    """Prepared implementation of `IndependentSquaredOracle`.

    Chart coordinates are absolute log density for each selected material and
    log source amplitude for a selected `log-source-amplitude` parameter.
    Its chain factor is formed in the per-history score before reduction. Unselected
    densities use `base_density`; an unselected amplitude uses `fixed_amplitude`.
    This positive chart excludes exact zero amplitude from optimisation; the
    lower-level derivative operator still supports its one-sided zero boundary.

    Capacity must cover the controller's maximum batch. Instances are mutable
    single-stream workspaces and must not be shared concurrently. Arrays derived
    from a batch are overwritten by the next call. Observations and pixel weights
    remain immutable for this oracle's lifetime. `histories_traced` counts actual
    forward/replay work, unlike the controller's unique-random-history budget.
    """

    workspace: TransportWorkspace
    source: ParallelBeam | PointConeSource
    parameters: tuple[TransportParameter, ...]
    observation: Any
    pixel_weights: Any
    fixed_amplitude: float
    _moments: EstimatorWorkspace = field(repr=False)
    _objective: ObjectiveWorkspace = field(repr=False)
    _arrays: dict[str, Any] = field(repr=False)
    _parameter_host: Any = field(repr=False)
    _parameter_view: Any = field(repr=False)
    _scalar_host: Any = field(repr=False)
    _scalar_view: Any = field(repr=False)
    _gradient_readback: HostReadback = field(repr=False)
    _fused_derivatives: bool = field(default=False, repr=False)
    _symmetric_gradient: bool = field(default=False, repr=False)
    _model_kernels: Any = field(default=None, repr=False)
    _model_partials: tuple[Any, ...] = field(default=(), repr=False)
    _model_host: Any = field(default=None, repr=False)
    _model_view: Any = field(default=None, repr=False)
    _deterministic_sampling: bool = field(default=False, repr=False)
    _last_parameters: Vector | None = field(default=None, repr=False)
    histories_traced: int = 0
    parameter_upload_bytes: int = 0
    scalar_download_bytes: int = 0

    @property
    def deterministic_sampling(self) -> bool:
        """Preparation established zero scattering and a fixed ray source."""
        return self._deterministic_sampling

    @property
    def supports_local_model(self) -> bool:
        return bool(self._model_partials)

    @property
    def allocated_bytes(self) -> int:
        """Owned device bytes, excluding shared transport/static input buffers."""
        wp = self.workspace.context.wp
        return (
            sum(
                int(value.size) * wp.types.type_size_in_bytes(value.dtype)
                for value in self._arrays.values()
            )
            + sum(int(value.size) * 8 for value in self._model_partials)
            + self._moments.scratch_bytes
            + self._objective.scratch_bytes
        )

    def _chart(self, values: Vector) -> float:
        if len(values) != len(self.parameters):
            raise TransportError("parameter vector does not match the prepared transport chart")
        values = tuple(finite_scalar(value, "chart coordinate") for value in values)
        amplitude = self.fixed_amplitude
        for index, parameter in enumerate(self.parameters):
            try:
                physical = math.exp(values[index])
            except OverflowError as error:
                raise TrialDomainError(
                    "log parameter overflows its physical representation"
                ) from error
            if not math.isfinite(physical) or physical <= 0:
                raise TrialDomainError(
                    "log parameter has no positive finite binary64 representation"
                )
            if parameter.kind == "log-source-amplitude":
                amplitude = physical
        if values != self._last_parameters:
            self._last_parameters = None
            for index, value in enumerate(values):
                self._parameter_view[index] = value
            context = self.workspace.context
            context.wp.copy(self._arrays["chart"], self._parameter_host, stream=context.stream)
            self.workspace._launch(
                self.workspace._kernels.update_density_chart,
                self.workspace.spec.grid.materials,
                [
                    self._arrays["chart"],
                    self._arrays["material_parameter"],
                    self._arrays["base_density"],
                    self._arrays["density"],
                    self.workspace._status,
                ],
            )
            self.workspace.check_status()
            self.parameter_upload_bytes += 8 * len(values)
            self._last_parameters = values
        return amplitude

    def _source_arrays(self, batch: HistoryBatch) -> list[Any]:
        if batch.count > self.workspace.max_histories:
            raise TransportError("oracle batch exceeds prepared capacity")
        arrays = [self._arrays[name][: batch.count] for name in ("position", "direction", "weight")]
        if isinstance(self.source, ParallelBeam):
            sample_parallel_beam(
                self.source,
                batch=batch,
                workspace=self.workspace,
                out_position=arrays[0],
                out_direction=arrays[1],
                out_weight=arrays[2],
                stream=self.workspace.context.stream,
            )
        else:
            sample_point_cone(
                self.source,
                batch=batch,
                workspace=self.workspace,
                out_position=arrays[0],
                out_direction=arrays[1],
                out_weight=arrays[2],
                stream=self.workspace.context.stream,
            )
        return [*arrays, self._arrays["density"]]

    def _mean(self, batch: HistoryBatch, amplitude: float, destination: Any) -> None:
        inputs = self._source_arrays(batch)
        outputs = {
            f"out_{name}": self._arrays[name][: batch.count]
            for name in ("pixel", "score", "energy", "events", "status")
        }
        trace_histories(
            *inputs,
            batch=batch,
            workspace=self.workspace,
            source_amplitude=amplitude,
            stream=self.workspace.context.stream,
            validate=False,
            **outputs,
        )
        self.histories_traced += batch.count
        history_mean(
            outputs["out_pixel"],
            outputs["out_score"],
            outputs["out_status"],
            batch=batch,
            workspace=self._moments,
            out_mean=destination,
            stream=self.workspace.context.stream,
            validate=False,
        )

    def _scalar(self) -> float:
        context = self.workspace.context
        reduce_objective_components(
            self._arrays["components"],
            out_loss=self._arrays["scalar"],
            workspace=self._objective,
            stream=context.stream,
            validate=False,
        )
        context.wp.copy(self._scalar_host, self._arrays["scalar"], stream=context.stream)
        # Check both producers before accepting the scalar; a failed transport
        # kernel may still leave finite zeros that a loss kernel cannot diagnose.
        self.workspace.check_status()
        self._objective.check_status()
        self.scalar_download_bytes += 8
        return float(self._scalar_view[0])

    def gradient_replicate(
        self,
        parameters: Vector,
        mean_batch: HistoryBatch,
        derivative_batch: HistoryBatch,
    ) -> Vector:
        """G[d]=sum_p w[p]*(mean_A[p]-y[p])*dmean_B[p]/dx[d].

        Parameters is a length-D dimensionless binary64 tuple of absolute log-material
        density and selected log-source-amplitude coordinates. The oracle binds fixed
        FP64 observation/pixel_weights (P,), material tables and source geometry in
        world mm. Mean_batch and derivative_batch are complete independent designs,
        including their library source-draw identities. The returned binary64 tuple
        has length D and loss units per chart coordinate. Detector arrays and derivative
        columns remain resident and only the small controls are transferred. Optional
        symmetric_gradient averages the two independent-factor orientations. Calls
        reuse single-stream scratch and preserve observations/fixed parameters.

        """
        try:
            return self._gradient_replicate(parameters, mean_batch, derivative_batch)
        finally:
            # An exception after asynchronous H2D upload must not release pinned
            # staging for the next call while CUDA still reads the previous values.
            self.workspace.context.wp.synchronize_stream(self.workspace.context.stream)

    def change_replicate(
        self,
        before: Vector,
        after: Vector,
        first: HistoryBatch,
        second: HistoryBatch,
    ) -> float:
        """Delta=sum_p
        w[p]*(residual_A(after)*residual_B(after)-residual_A(before)*residual_B(before))/2.

        Before/after are matching length-D dimensionless binary64 tuples in the
        absolute log-density/log-amplitude chart. First/second are independent complete
        HistoryBatch designs and are reused across the two points for common random
        numbers within each product factor. Fixed observation/pixel_weights are FP64
        (P,), and all detector scores/products remain on the owning device. The
        binary64 scalar result has squared-score loss units and may have either sign.
        Independent repeated calls with fresh complete identities give change
        uncertainty, including pixel and common-random-number covariance. Calls do
        not change the caller's incumbent parameters and reuse prepared scratch.

        """
        try:
            return self._change_replicate(before, after, first, second)
        finally:
            self.workspace.context.wp.synchronize_stream(self.workspace.context.stream)

    def _gradient_replicate(
        self,
        parameters: Vector,
        mean_batch: HistoryBatch,
        derivative_batch: HistoryBatch,
        capture_model: bool = False,
    ) -> Vector:
        result = self._one_sided_gradient(parameters, mean_batch, derivative_batch, capture_model)
        if self._symmetric_gradient:
            reverse = self._one_sided_gradient(parameters, derivative_batch, mean_batch, False)
            return tuple(
                first / 2 + second / 2 for first, second in zip(result, reverse, strict=True)
            )
        return result

    def _one_sided_gradient(
        self,
        parameters: Vector,
        mean_batch: HistoryBatch,
        derivative_batch: HistoryBatch,
        capture_model: bool = False,
    ) -> Vector:
        """Use disjoint source/transport samples for the two nonlinear factors."""
        require_no_tape()
        sampler = (
            "parallel-rectangle-v1"
            if isinstance(self.source, ParallelBeam)
            else "point-cone-area-v1"
        )
        mean_batch = mean_batch.with_library_source(sampler)
        derivative_batch = derivative_batch.with_library_source(sampler)
        require_independent(mean_batch, derivative_batch)
        amplitude = self._chart(parameters)
        self._mean(mean_batch, amplitude, self._arrays["mean_a"])
        inputs = self._source_arrays(derivative_batch)
        pixel = self._arrays["pixel"][: derivative_batch.count]
        derivative = self._arrays["score"][: derivative_batch.count]
        status = self._arrays["status"][: derivative_batch.count]
        for column, parameter in enumerate(self.parameters):
            if self._fused_derivatives:
                group_column = column % FUSED_PARAMETER_LIMIT
                if group_column == 0:
                    selected = self.parameters[column : column + FUSED_PARAMETER_LIMIT]
                    derivative_columns(
                        *inputs,
                        parameters=selected,
                        batch=derivative_batch,
                        workspace=self.workspace,
                        out_pixel=pixel,
                        out_derivatives=self._arrays["derivative_columns"][
                            : len(selected) * derivative_batch.count
                        ],
                        out_status=status,
                        source_amplitude=amplitude,
                        stream=self.workspace.context.stream,
                        validate=False,
                    )
                    self.histories_traced += derivative_batch.count
                offset = group_column * derivative_batch.count
                derivative = self._arrays["derivative_columns"][
                    offset : offset + derivative_batch.count
                ]
            else:
                derivative_histories(
                    *inputs,
                    parameter=parameter,
                    batch=derivative_batch,
                    workspace=self.workspace,
                    out_pixel=pixel,
                    out_derivative=derivative,
                    out_status=status,
                    source_amplitude=amplitude,
                    stream=self.workspace.context.stream,
                    validate=False,
                )
                self.histories_traced += derivative_batch.count
            history_mean(
                pixel,
                derivative,
                status,
                batch=derivative_batch,
                workspace=self._moments,
                out_mean=self._arrays["mean_b"],
                stream=self.workspace.context.stream,
                validate=False,
            )
            if capture_model:
                self.workspace._launch(
                    self._model_kernels.store_column,
                    self.workspace.spec.detector.pixels,
                    [
                        self._arrays["mean_b"],
                        column,
                        self.workspace.spec.detector.pixels,
                        self._arrays["jacobian"],
                    ],
                )
            independent_squared_gradient(
                self._arrays["mean_a"],
                self._arrays["mean_b"],
                self.observation,
                self.pixel_weights,
                batches=(mean_batch, derivative_batch),
                workspace=self.workspace,
                out_components=self._arrays["components"],
                stream=self.workspace.context.stream,
                validate=False,
            )
            reduce_objective_components(
                self._arrays["components"],
                out_loss=self._arrays["gradient"][column : column + 1],
                workspace=self._objective,
                stream=self.workspace.context.stream,
                validate=False,
            )
        values = self._gradient_readback.read()
        self.workspace.check_status()
        self._objective.check_status()
        self.scalar_download_bytes += 8 * len(self.parameters)
        result = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in result):
            raise NumericalError("inverse chart derivative overflow")
        return result

    def model_replicate(
        self,
        parameters: Vector,
        mean_batch: HistoryBatch,
        derivative_batch: HistoryBatch,
    ) -> tuple[Vector, tuple[Vector, ...]]:
        """G=J_B^T*W*(mean_A-y) and B=J_B^T*W*J_B form a gradient and PSD proposal metric.

        Parameters is a length-D dimensionless binary64 log-density/log-amplitude
        tuple with D <= 16 and local_model enabled at preparation. The two complete
        HistoryBatch designs are independent including source draws. The returned pair
        contains a binary64 gradient tuple (D,) and symmetric PSD matrix tuple (D,D)
        in fixed chart units. Observation/pixel_weights are FP64 (P,) and Jacobian
        columns/pixel contractions remain resident. The estimated-Jacobian Gram
        metric retains sampling bias and serves proposal construction. Fresh held-out
        acceptance decides the actual objective change. Fixed model/source parameters
        remain unchanged and calls reuse their owning stream/scratch.

        """
        if not self._model_partials:
            raise TransportError("prepare the inverse oracle with local_model=True")
        context = self.workspace.context
        wp = context.wp
        try:
            gradient = self._gradient_replicate(parameters, mean_batch, derivative_batch, True)
            dimension = len(self.parameters)
            pixels = self.workspace.spec.detector.pixels
            count = (pixels + 255) // 256
            wp.launch_tiled(
                self._model_kernels.gram_tiles,
                dim=dimension * dimension * count,
                block_dim=256,
                inputs=[
                    self._arrays["jacobian"],
                    self.pixel_weights,
                    pixels,
                    dimension,
                    count,
                    self._model_partials[0],
                    self.workspace._status,
                ],
                device=context.device,
                stream=context.stream,
                record_tape=False,
            )
            previous = self._model_partials[0]
            for destination in self._model_partials[1:]:
                next_count = (count + 255) // 256
                wp.launch_tiled(
                    self._model_kernels.sum_gram_tiles,
                    dim=dimension * dimension * next_count,
                    block_dim=256,
                    inputs=[previous, count, next_count, destination],
                    device=context.device,
                    stream=context.stream,
                    record_tape=False,
                )
                previous, count = destination, next_count
            wp.copy(self._model_host, previous, stream=context.stream)
            self.workspace.check_status()
            self.scalar_download_bytes += 8 * dimension * dimension
            curvature = tuple(
                tuple(float(self._model_view[i * dimension + j]) for j in range(dimension))
                for i in range(dimension)
            )
            if not all(math.isfinite(x) for row in curvature for x in row):
                raise NumericalError("proposal curvature exceeds the finite chart range")
            return gradient, curvature
        finally:
            wp.synchronize_stream(context.stream)

    def _change_replicate(
        self,
        before: Vector,
        after: Vector,
        first: HistoryBatch,
        second: HistoryBatch,
    ) -> float:
        """Independent product factors; common random numbers across parameter points."""
        require_no_tape()
        sampler = (
            "parallel-rectangle-v1"
            if isinstance(self.source, ParallelBeam)
            else "point-cone-area-v1"
        )
        first, second = first.with_library_source(sampler), second.with_library_source(sampler)
        require_independent(first, second)
        losses: list[float] = []
        for parameters in (before, after):
            amplitude = self._chart(parameters)
            self._mean(first, amplitude, self._arrays["mean_a"])
            self._mean(second, amplitude, self._arrays["mean_b"])
            independent_squared_loss(
                self._arrays["mean_a"],
                self._arrays["mean_b"],
                self.observation,
                self.pixel_weights,
                batches=(first, second),
                workspace=self.workspace,
                out_components=self._arrays["components"],
                stream=self.workspace.context.stream,
                validate=False,
            )
            losses.append(self._scalar())
        difference = losses[1] - losses[0]
        if not math.isfinite(difference):
            raise NumericalError("objective-change estimate overflow")
        return difference


def prepare_transport_inverse(
    workspace: TransportWorkspace,
    *,
    source: ParallelBeam | PointConeSource,
    parameters: tuple[TransportParameter, ...],
    observation: Any,
    pixel_weights: Any,
    base_density: tuple[float, ...],
    fixed_amplitude: float = 1.0,
    local_model: bool = False,
    local_model_max_bytes: int = 256 * 1024 * 1024,
    fused_derivatives: bool = False,
    fused_derivative_max_bytes: int = 256 * 1024 * 1024,
    symmetric_gradient: bool = False,
) -> TransportSquaredOracle:
    """Allocate a complete reusable inverse oracle; no physics execution is implied.

    The preparation call binds immutable device observations/weights and uploads
    only fixed density values and the material-to-parameter map. Dynamic small
    chart uploads and scalar downloads are counted explicitly on the oracle.
    """
    require_no_tape()
    if type(local_model) is not bool:
        raise TransportError("local_model must be a boolean preparation choice")
    if type(fused_derivatives) is not bool:
        raise TransportError("fused_derivatives must be a boolean preparation choice")
    if type(symmetric_gradient) is not bool:
        raise TransportError("symmetric_gradient must be a boolean preparation choice")
    integer(fused_derivative_max_bytes, "fused_derivative_max_bytes", minimum=1)
    integer(local_model_max_bytes, "local_model_max_bytes", minimum=1)
    parameters = tuple(parameters)
    base_density = tuple(base_density)
    if any(parameter.kind == "source-amplitude" for parameter in parameters):
        raise TransportError(
            "the inverse chart requires log-source-amplitude, not direct amplitude"
        )
    if not parameters or len(set(parameters)) != len(parameters):
        raise TransportError("select at least one distinct supported transport parameter")
    if len(base_density) != workspace.spec.grid.materials:
        raise TransportError("base_density must contain one scale per material")
    if any(finite_scalar(value, "base density", minimum=0.0) == 0 for value in base_density):
        raise TransportError("base density must be positive")
    finite_scalar(fixed_amplitude, "fixed source amplitude", minimum=0.0)
    if isinstance(source, ParallelBeam) and any(
        workspace.spec.detector.source_distance(
            (source.lower_mm[0] + x, source.lower_mm[1] + y, source.lower_mm[2])
        )
        >= 0
        for x in (0.0, source.extent_xy_mm[0])
        for y in (0.0, source.extent_xy_mm[1])
    ):
        raise TransportError("source plane must be below the detector plane")
    if (
        isinstance(source, PointConeSource)
        and workspace.spec.detector.source_distance(source.geometry.source_mm) >= 0
    ):
        raise TransportError("point source must be on the incident side of the detector plane")
    context = workspace.context
    wp = context.wp
    pixels = workspace.spec.detector.pixels
    dimension = len(parameters)
    fused_columns = min(FUSED_PARAMETER_LIMIT, dimension)
    if fused_derivatives:
        if fused_columns * workspace.max_histories >= 2**31:
            raise TransportError("fused derivative storage exceeds signed 32-bit indexing")
        required = 8 * fused_columns * workspace.max_histories
        if required > fused_derivative_max_bytes:
            raise TransportError(
                f"fused derivative storage needs {required} bytes, exceeding preparation budget"
            )
    partial_sizes: list[int] = []
    if local_model:
        if dimension > 16:
            raise TransportError("dense local models support at most 16 active parameters")
        count = (pixels + 255) // 256
        while True:
            partial_sizes.append(dimension * dimension * count)
            if count == 1:
                break
            count = (count + 255) // 256
        if dimension * pixels >= 2**31 or any(size >= 2**31 for size in partial_sizes):
            raise TransportError("local model exceeds signed 32-bit indexing")
        required_bytes = 8 * (dimension * pixels + sum(partial_sizes) + dimension * dimension)
        if required_bytes > local_model_max_bytes:
            raise TransportError(
                f"local model needs {required_bytes} bytes, exceeding preparation budget"
            )
    for name, value in (("observation", observation), ("pixel_weights", pixel_weights)):
        context.array(value, name, dtype=wp.float64, shape=(pixels,))
    workspace._launch(
        workspace._kernels.validate_measurement,
        pixels,
        [observation, pixel_weights, workspace._status],
    )
    workspace.check_status()
    mapping = [-1] * workspace.spec.grid.materials
    for index, parameter in enumerate(parameters):
        if parameter.material is not None:
            if parameter.material >= len(mapping):
                raise TransportError("active material is outside this model")
            mapping[parameter.material] = index
    arrays: dict[str, Any] = {}
    with context.scope():
        for name, dtype in (
            ("position", wp.vec3d),
            ("direction", wp.vec3d),
            ("weight", wp.float64),
            ("pixel", wp.int32),
            ("score", wp.float64),
            ("energy", wp.float64),
            ("events", wp.int32),
            ("status", wp.int32),
        ):
            arrays[name] = wp.empty(workspace.max_histories, dtype=dtype, device=context.device)
        for name in ("mean_a", "mean_b", "components"):
            arrays[name] = wp.empty(pixels, dtype=wp.float64, device=context.device)
        arrays["scalar"] = wp.empty(1, dtype=wp.float64, device=context.device)
        arrays["gradient"] = wp.empty(dimension, dtype=wp.float64, device=context.device)
        if fused_derivatives:
            arrays["derivative_columns"] = wp.empty(
                fused_columns * workspace.max_histories, dtype=wp.float64, device=context.device
            )
        arrays["density"] = wp.empty(len(base_density), dtype=wp.float64, device=context.device)
        arrays["base_density"] = wp.array(
            list(base_density), dtype=wp.float64, device=context.device
        )
        arrays["material_parameter"] = wp.array(mapping, dtype=wp.int32, device=context.device)
        arrays["chart"] = wp.empty(len(parameters), dtype=wp.float64, device=context.device)
        parameter_host = wp.empty(
            len(parameters), dtype=wp.float64, device="cpu", pinned=context.device.is_cuda
        )
        scalar_host = wp.empty(1, dtype=wp.float64, device="cpu", pinned=context.device.is_cuda)
    model_kernels = None
    model_partials: tuple[Any, ...] = ()
    model_host = None
    model_view = None
    deterministic = False
    if local_model or workspace.spec.estimator == "continuous-absorption":
        model_kernels = load_kernels("dpt.transport.recovery_kernels")
    with context.scope():
        if local_model:
            arrays["jacobian"] = wp.empty(
                dimension * pixels, dtype=wp.float64, device=context.device
            )
            model_partials = tuple(
                wp.empty(n, dtype=wp.float64, device=context.device) for n in partial_sizes
            )
            model_host = wp.empty(
                dimension * dimension,
                dtype=wp.float64,
                device="cpu",
                pinned=context.device.is_cuda,
            )
            model_view = model_host.numpy()
        if (
            workspace.spec.estimator == "continuous-absorption"
            and isinstance(source, ParallelBeam)
            and source.extent_xy_mm
            == (
                0.0,
                0.0,
            )
        ):
            assert model_kernels is not None
            flag = wp.zeros(1, dtype=wp.int32, device=context.device)
            workspace._launch(
                model_kernels.flag_scattering,
                int(workspace.scattering.size),
                [workspace.scattering, flag],
            )
            wp.synchronize_stream(context.stream)
            deterministic = int(flag.numpy()[0]) == 0
    moments = prepare_estimators(workspace)
    objective = prepare_objective(
        ObjectiveSpec(), max_pixels=pixels, device=context.device, stream=context.stream
    )
    return TransportSquaredOracle(
        workspace,
        source,
        parameters,
        observation,
        pixel_weights,
        fixed_amplitude,
        moments,
        objective,
        arrays,
        parameter_host,
        parameter_host.numpy(),
        scalar_host,
        scalar_host.numpy(),
        HostReadback(context, arrays["gradient"]),
        _fused_derivatives=fused_derivatives,
        _symmetric_gradient=symmetric_gradient,
        _model_kernels=model_kernels,
        _model_partials=model_partials,
        _model_host=model_host,
        _model_view=model_view,
        _deterministic_sampling=deterministic,
    )
