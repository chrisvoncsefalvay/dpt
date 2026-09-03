"""Bounded replicate-axis transport oracles using the canonical counter replay.

Fixed-size replicate batches share one material chart and fixed source law. The
backend stores resident R by pixel means and downloads one R by P gradient checkpoint.
Unequal sizes or requests above the declared replicate capacity use scalar
fallback without allocating a larger workspace. Local metrics retain the scalar
oracle's existing PSD proposal construction and fuse up to eight parameters.
"""

from __future__ import annotations

# Workspaces compose reviewed package-private launch and staging contracts.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass, field
from typing import Any

from dpt._runtime import load_kernels, require_no_tape
from dpt.contracts import NumericalError, integer
from dpt.registration import Vector

from ._constants import LOG_SOURCE_AMPLITUDE, SOURCE_AMPLITUDE
from .derivatives import FUSED_PARAMETER_LIMIT, TransportParameter
from .inverse import TransportSquaredOracle
from .model import TransportError
from .rng import HistoryBatch, require_independent
from .source import ParallelBeam

type BatchPairs = tuple[tuple[HistoryBatch, HistoryBatch], ...]
BATCHED_MODEL_PARAMETER_LIMIT = 8


@dataclass(slots=True)
class BatchedTransportSquaredOracle:
    """Opt-in plural oracle with bounded resident original-history staging.

    This mutable workspace shares the scalar oracle's chart and owning stream;
    neither instance may be used concurrently. Original-history identities are
    preserved exactly. Atomic tally order can change binary64 rounding.
    """

    oracle: TransportSquaredOracle
    max_replicates: int
    _kernels: Any = field(repr=False)
    _source: Any = field(repr=False)
    _arrays: dict[str, Any] = field(repr=False)
    _partials: tuple[Any, ...] = field(repr=False)
    _row_kernels: Any = field(repr=False)
    _ids_host: Any = field(repr=False)
    _ids_view: Any = field(repr=False)
    _gradient_host: Any = field(repr=False)
    _gradient_view: Any = field(repr=False)
    _loss_host: Any = field(repr=False)
    _loss_view: Any = field(repr=False)
    _model_kernels: Any = field(default=None, repr=False)
    _model_host: Any = field(default=None, repr=False)
    _model_view: Any = field(default=None, repr=False)
    _batched_histories_traced: int = 0
    _batched_upload_bytes: int = 0
    _batched_download_bytes: int = 0
    _upload_pending: bool = field(default=False, repr=False)

    @property
    def histories_traced(self) -> int:
        return self.oracle.histories_traced + self._batched_histories_traced

    @property
    def parameter_upload_bytes(self) -> int:
        return self.oracle.parameter_upload_bytes + self._batched_upload_bytes

    @property
    def scalar_download_bytes(self) -> int:
        return self.oracle.scalar_download_bytes + self._batched_download_bytes

    @property
    def supports_local_model(self) -> bool:
        return self.oracle.supports_local_model

    @property
    def deterministic_sampling(self) -> bool:
        return self.oracle.deterministic_sampling

    @property
    def allocated_bytes(self) -> int:
        wp = self.oracle.workspace.context.wp
        return (
            self.oracle.allocated_bytes
            + sum(
                int(value.size) * wp.types.type_size_in_bytes(value.dtype)
                for value in self._arrays.values()
            )
            + sum(int(value.size) * 8 for value in self._partials)
        )

    @property
    def host_staging_bytes(self) -> int:
        dimension = len(self.oracle.parameters)
        model = 8 * dimension * (dimension + 1) if self._model_kernels is not None else 0
        return self.max_replicates * (32 + 8 * dimension + 8 + model)

    def gradient_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> Vector:
        """G[d]=sum_p w[p]*(mean_A[p]-y[p])*dmean_B[p]/dx[d] delegates one complete pair.

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
        return self.oracle.gradient_replicate(parameters, mean_batch, derivative_batch)

    def change_replicate(
        self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
    ) -> float:
        """Delta=(estimated_squared_loss(after)-estimated_squared_loss(before)) delegates one
        paired change.

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
        return self.oracle.change_replicate(before, after, first, second)

    def model_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> tuple[Vector, tuple[Vector, ...]]:
        """(G,B)=(J_B^T*W*(mean_A-y), J_B^T*W*J_B) delegates one prepared model pair.

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
        return self.oracle.model_replicate(parameters, mean_batch, derivative_batch)

    def _pairs(self, pairs: BatchPairs) -> BatchPairs:
        if any(batch.sampling is not None for pair in pairs for batch in pair):
            raise TransportError(
                "batched inverse kernels require IID batches; use separately prepared "
                "complete randomisations for full-path design replicates"
            )
        sampler = (
            "parallel-rectangle-v1"
            if isinstance(self.oracle.source, ParallelBeam)
            else "point-cone-area-v1"
        )
        result = tuple(
            (first.with_library_source(sampler), second.with_library_source(sampler))
            for first, second in pairs
        )
        require_independent(*(batch for pair in result for batch in pair))
        return result

    def _fixed_count(self, pairs: BatchPairs) -> int | None:
        if not pairs or len(pairs) > self.max_replicates:
            return None
        count = pairs[0][0].count
        if count > self.oracle.workspace.max_histories:
            raise TransportError("oracle batch exceeds prepared history capacity")
        if any(batch.count != count for pair in pairs for batch in pair):
            return None
        return count

    def _upload_pairs(self, pairs: BatchPairs) -> None:
        ctx = self.oracle.workspace.context
        replicas = len(pairs)
        for row, (first, second) in enumerate(pairs):
            self._ids_view[0, row] = first.seed
            self._ids_view[1, row] = first.first_history
            self._ids_view[2, row] = second.seed
            self._ids_view[3, row] = second.first_history
        self._upload_pending = True
        with ctx.scope():
            for column, name in enumerate(("seed_a", "first_a", "seed_b", "first_b")):
                start = column * self.max_replicates
                ctx.wp.copy(
                    self._arrays[name][:replicas],
                    self._ids_host[start : start + replicas],
                    stream=ctx.stream,
                )
        self._batched_upload_bytes += 32 * replicas

    def _barrier(self) -> None:
        ctx = self.oracle.workspace.context
        if self._upload_pending:
            if ctx.device.is_cuda:
                ctx.wp.synchronize_stream(ctx.stream)
            self._upload_pending = False

    def _sample(self, label: str, replicas: int, count: int) -> None:
        workspace = self.oracle.workspace
        total = replicas * count
        workspace._launch(
            self._kernels.sample_sources,
            total,
            [
                self._source,
                self._arrays[f"seed_{label}"][:replicas],
                self._arrays[f"first_{label}"][:replicas],
                count,
                self._arrays["position"][:total],
                self._arrays["direction"][:total],
                self._arrays["weight"][:total],
            ],
        )

    def _trace(
        self,
        label: str,
        replicas: int,
        count: int,
        amplitude: float,
        parameters: tuple[TransportParameter, ...] = (),
        *,
        primal: bool = True,
    ) -> None:
        workspace = self.oracle.workspace
        wp = workspace.context.wp
        total = replicas * count
        selected = [
            parameter.material
            if parameter.material is not None
            else LOG_SOURCE_AMPLITUDE
            if parameter.kind == "log-source-amplitude"
            else SOURCE_AMPLITUDE
            for parameter in parameters
        ]
        selected.extend([SOURCE_AMPLITUDE] * (FUSED_PARAMETER_LIMIT - len(selected)))
        workspace._launch(
            self._kernels.trace_batched,
            total,
            [
                self._arrays["position"][:total],
                self._arrays["direction"][:total],
                self._arrays["weight"][:total],
                self.oracle._arrays["density"],
                *workspace._model_inputs(),
                self._arrays[f"seed_{label}"][:replicas],
                self._arrays[f"first_{label}"][:replicas],
                count,
                amplitude,
                wp.vec4i(*selected),
                len(parameters),
                int(primal),
                self._arrays["pixel"][:total],
                self._arrays["score"][:total],
                self._arrays["derivatives"][: max(1, len(parameters)) * total],
                self._arrays["history_status"][:total],
                workspace._status,
            ],
        )
        self._batched_histories_traced += total

    def _mean(self, values: Any, destination: Any, replicas: int, count: int) -> None:
        workspace = self.oracle.workspace
        pixels = workspace.spec.detector.pixels
        total = replicas * count
        pixels_total = replicas * pixels
        scale = self._arrays["scale"][:pixels_total]
        sums = self._arrays["sum"][:pixels_total]
        pixel = self._arrays["pixel"][:total]
        status = self._arrays["history_status"][:total]
        with workspace.context.scope():
            scale.zero_()
            sums.zero_()
        workspace._launch(
            self._kernels.find_tally_scale,
            total,
            [pixel, values, status, pixels, count, scale, workspace._status],
        )
        workspace._launch(
            self._kernels.tally_sum,
            total,
            [pixel, values, status, pixels, count, scale, sums, workspace._status],
        )
        workspace._launch(
            self._kernels.finish_mean,
            pixels_total,
            [sums, scale, count, destination, workspace._status],
        )

    def _reduce(
        self,
        replicas: int,
        other: Any,
        destination: Any,
        stride: int,
        column: int,
        *,
        loss: bool = False,
    ) -> None:
        workspace = self.oracle.workspace
        ctx = workspace.context
        pixels = workspace.spec.detector.pixels
        tiles = (pixels + 255) // 256
        previous = self._partials[0][: replicas * tiles]
        ctx.wp.launch_tiled(
            self._kernels.objective_tiles,
            dim=replicas * tiles,
            block_dim=256,
            inputs=[
                self._arrays["mean_a"][: replicas * pixels],
                other,
                self.oracle.observation,
                self.oracle.pixel_weights,
                pixels,
                tiles,
                int(loss),
                previous,
                workspace._status,
            ],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
        count = tiles
        for capacity in self._partials[1:]:
            next_count = (count + 255) // 256
            following = capacity[: replicas * next_count]
            ctx.wp.launch_tiled(
                self._row_kernels.sum_gram_tiles,
                dim=replicas * next_count,
                block_dim=256,
                inputs=[previous, count, next_count, following],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
            previous, count = following, next_count
        workspace._launch(
            self._kernels.finish_values, replicas, [previous, stride, column, destination]
        )

    def _read(self, name: str, count: int) -> None:
        workspace = self.oracle.workspace
        ctx = workspace.context
        host = (
            self._gradient_host
            if name == "gradient"
            else self._model_host
            if name == "model"
            else self._loss_host
        )
        with ctx.scope():
            ctx.wp.copy(host[:count], self._arrays[name][:count], stream=ctx.stream)
        # The status copy/wait completes both transfers on this same owning stream.
        code = workspace._status_readback.read()
        self._upload_pending = False
        workspace._raise_status(code)
        self._batched_download_bytes += 8 * count

    def _gradient(
        self,
        parameters: Vector,
        pairs: BatchPairs,
        count: int,
        *,
        reverse: bool = False,
        upload: bool = True,
        capture_jacobian: bool = False,
    ) -> None:
        replicas = len(pairs)
        # The scalar chart also reuses pinned staging; preserve its lifetime if
        # a launch raises before the chart's own checkpoint completes.
        self._upload_pending = True
        amplitude = self.oracle._chart(parameters)
        if upload:
            self._upload_pairs(pairs)
        mean_label, derivative_label = ("b", "a") if reverse else ("a", "b")
        self._sample(mean_label, replicas, count)
        self._trace(mean_label, replicas, count, amplitude)
        total = replicas * count
        pixels_total = replicas * self.oracle.workspace.spec.detector.pixels
        self._mean(
            self._arrays["score"][:total], self._arrays["mean_a"][:pixels_total], replicas, count
        )
        self._sample(derivative_label, replicas, count)
        dimension = len(self.oracle.parameters)
        for first_column in range(0, dimension, FUSED_PARAMETER_LIMIT):
            selected = self.oracle.parameters[first_column : first_column + FUSED_PARAMETER_LIMIT]
            self._trace(derivative_label, replicas, count, amplitude, selected, primal=False)
            for group_column in range(len(selected)):
                derivative = self._arrays["derivatives"][
                    group_column * total : (group_column + 1) * total
                ]
                mean = self._arrays["mean_b"][:pixels_total]
                self._mean(derivative, mean, replicas, count)
                self._reduce(
                    replicas,
                    mean,
                    self._arrays["gradient"][: replicas * dimension],
                    dimension,
                    first_column + group_column,
                )
                if capture_jacobian:
                    self.oracle.workspace._launch(
                        self._model_kernels.store_column,
                        pixels_total,
                        [
                            mean,
                            first_column + group_column,
                            self.oracle.workspace.spec.detector.pixels,
                            dimension,
                            self._arrays["jacobian"][: dimension * pixels_total],
                        ],
                    )

    def _copy_previous(self, name: str, count: int) -> None:
        ctx = self.oracle.workspace.context
        with ctx.scope():
            ctx.wp.copy(
                self._arrays[f"previous_{name}"][:count],
                self._arrays[name][:count],
                stream=ctx.stream,
            )

    def _combine(self, name: str, count: int, *, average: bool) -> None:
        self.oracle.workspace._launch(
            self._kernels.combine_values,
            count,
            [
                self._arrays[f"previous_{name}"][:count],
                self._arrays[name][:count],
                int(average),
                self.oracle.workspace._status,
            ],
        )

    def _gradient_result(self, replicas: int) -> tuple[Vector, ...]:
        dimension = len(self.oracle.parameters)
        self._read("gradient", replicas * dimension)
        result = tuple(
            tuple(float(self._gradient_view[row, column]) for column in range(dimension))
            for row in range(replicas)
        )
        if not all(math.isfinite(value) for gradient in result for value in gradient):
            raise NumericalError("batched transport gradient exceeds finite range")
        return result

    def gradient_replicates(self, parameters: Vector, pairs: BatchPairs) -> tuple[Vector, ...]:
        """G[r,d]=sum_p w[p]*(mean_A[r,p]-y[p])*dmean_B[r,p]/dx[d].

        Parameters is a length-D dimensionless binary64 log-density/log-amplitude
        tuple and pairs is a length-R tuple of two independent complete HistoryBatch
        designs per replicate. All identities across replicates must be disjoint.
        The returned binary64 tuple has R rows of length D, retaining pair order.
        Fixed observations/weights are FP64 (P,), while the prepared device replica
        axis keeps images/derivatives resident and transfers only small controls.
        Supported equal-size designs use batched launches within max_replicates and
        the scalar oracle handles other supported shapes. Uncertainty is formed
        across complete independent rows, including whole scrambles for Sobol designs.

        """
        require_no_tape()
        self.oracle._objective.check_status()
        pairs = self._pairs(pairs)
        count = self._fixed_count(pairs)
        if count is None:
            return tuple(
                self.oracle.gradient_replicate(parameters, first, second) for first, second in pairs
            )
        try:
            self._gradient(parameters, pairs, count)
            if self.oracle._symmetric_gradient:
                size = len(pairs) * len(self.oracle.parameters)
                self._copy_previous("gradient", size)
                self._gradient(parameters, pairs, count, reverse=True, upload=False)
                self._combine("gradient", size, average=True)
            return self._gradient_result(len(pairs))
        finally:
            self._barrier()

    def model_replicates(
        self, parameters: Vector, pairs: BatchPairs
    ) -> tuple[tuple[Vector, tuple[Vector, ...]], ...]:
        """(G_r,B_r)=(J_B,r^T*W*(mean_A,r-y), J_B,r^T*W*J_B,r) retains one model per pair.

        Parameters is a length-D dimensionless binary64 fixed log chart and pairs
        contains R complete independent mean/derivative HistoryBatch pairs with
        disjoint identities across replicates. Local_model must be prepared. Each
        returned tuple row contains gradient (D,) and symmetric PSD metric (D,D)
        in binary64 chart units. Fixed observations/pixel_weights have FP64 (P,)
        layout. The estimated Gram metric retains sampling bias and provides a
        proposal metric. The prepared replica axis fuses supported equal-size designs
        with at most eight parameters and the scalar path handles remaining supported
        shapes. Fresh acceptance remains independent of these model samples.

        """
        require_no_tape()
        self.oracle._objective.check_status()
        pairs = self._pairs(pairs)
        count = self._fixed_count(pairs)
        if count is None or self._model_kernels is None:
            return tuple(
                self.oracle.model_replicate(parameters, first, second) for first, second in pairs
            )
        try:
            replicas, dimension = len(pairs), len(self.oracle.parameters)
            self._gradient(parameters, pairs, count, capture_jacobian=True)
            if self.oracle._symmetric_gradient:
                size = replicas * dimension
                self._copy_previous("gradient", size)
                self._gradient(parameters, pairs, count, reverse=True, upload=False)
                self._combine("gradient", size, average=True)
            self._metric(replicas)
            width = dimension * (dimension + 1)
            self._read("model", replicas * width)
            result = tuple(
                (
                    tuple(float(self._model_view[replica, column]) for column in range(dimension)),
                    tuple(
                        tuple(
                            float(self._model_view[replica, dimension + row * dimension + column])
                            for column in range(dimension)
                        )
                        for row in range(dimension)
                    ),
                )
                for replica in range(replicas)
            )
            if not all(
                math.isfinite(value)
                for gradient, metric in result
                for row in (gradient, *metric)
                for value in row
            ):
                raise NumericalError("batched transport proposal metric exceeds finite range")
            return result
        finally:
            self._barrier()

    def _metric(self, replicas: int) -> None:
        workspace = self.oracle.workspace
        ctx = workspace.context
        dimension = len(self.oracle.parameters)
        pixels = workspace.spec.detector.pixels
        tiles = (pixels + 255) // 256
        rows = replicas * dimension * dimension
        previous = self._partials[0][: rows * tiles]
        ctx.wp.launch_tiled(
            self._model_kernels.gram_tiles,
            dim=rows * tiles,
            block_dim=256,
            inputs=[
                self._arrays["jacobian"][: replicas * dimension * pixels],
                self.oracle.pixel_weights,
                pixels,
                dimension,
                tiles,
                previous,
                workspace._status,
            ],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
        count = tiles
        for capacity in self._partials[1:]:
            next_count = (count + 255) // 256
            following = capacity[: rows * next_count]
            ctx.wp.launch_tiled(
                self._row_kernels.sum_gram_tiles,
                dim=rows * next_count,
                block_dim=256,
                inputs=[previous, count, next_count, following],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
            previous, count = following, next_count
        workspace._launch(
            self._model_kernels.finish_model,
            replicas * dimension * (dimension + 1),
            [
                self._arrays["gradient"][: replicas * dimension],
                previous,
                dimension,
                self._arrays["model"][: replicas * dimension * (dimension + 1)],
            ],
        )

    def _losses(
        self, parameters: Vector, pairs: BatchPairs, count: int, *, upload: bool = True
    ) -> None:
        replicas = len(pairs)
        self._upload_pending = True
        amplitude = self.oracle._chart(parameters)
        if upload:
            self._upload_pairs(pairs)
        total = replicas * count
        pixels_total = replicas * self.oracle.workspace.spec.detector.pixels
        for label, name in (("a", "mean_a"), ("b", "mean_b")):
            self._sample(label, replicas, count)
            self._trace(label, replicas, count, amplitude)
            self._mean(
                self._arrays["score"][:total], self._arrays[name][:pixels_total], replicas, count
            )
        self._reduce(
            replicas,
            self._arrays["mean_b"][:pixels_total],
            self._arrays["loss"][:replicas],
            1,
            0,
            loss=True,
        )

    def change_replicates(
        self, before: Vector, after: Vector, pairs: BatchPairs
    ) -> tuple[float, ...]:
        """Delta[r]=estimated_F_r(after)-estimated_F_r(before) retains complete paired changes.

        Before/after are matching length-D dimensionless binary64 fixed log-chart
        tuples. Pairs has R independent complete HistoryBatch pairs with transport
        and source identities separated both within and across pairs. Common random
        numbers couple before/after within each factor. Returned binary64 scalar
        tuple (R,) preserves pair order and squared-score loss units. Fixed FP64
        observation/pixel_weights (P,) and all detector products remain resident.
        Prepared equal-size designs use the replica axis and other supported shapes
        use the scalar path. The caller summarises complete changes to preserve
        pixel/common-random-number covariance, without updating the incumbent here.

        """
        require_no_tape()
        self.oracle._objective.check_status()
        pairs = self._pairs(pairs)
        count = self._fixed_count(pairs)
        if count is None:
            return tuple(
                self.oracle.change_replicate(before, after, first, second)
                for first, second in pairs
            )
        try:
            self._losses(before, pairs, count)
            self._copy_previous("loss", len(pairs))
            self._losses(after, pairs, count, upload=False)
            self._combine("loss", len(pairs), average=False)
            self._read("loss", len(pairs))
            result = tuple(float(self._loss_view[row]) for row in range(len(pairs)))
            if not all(math.isfinite(value) for value in result):
                raise NumericalError("batched transport objective change overflow")
            return result
        finally:
            self._barrier()


def prepare_transport_batched(
    oracle: TransportSquaredOracle,
    *,
    max_replicates: int = 4,
    max_bytes: int = 256 * 1024 * 1024,
) -> BatchedTransportSquaredOracle:
    """Allocate a bounded replica axis, with scalar unequal-size fallback.

    Set ``max_replicates`` to the controller's replicate count to use the device
    axis. ``max_bytes`` caps this wrapper's additional owned device allocation;
    the shared scalar oracle and fixed transport model are accounted separately.
    Local metrics fuse at most eight parameters and share reduction scratch with
    gradients; larger metrics retain the prepared scalar implementation.
    """
    require_no_tape()
    max_replicates = integer(max_replicates, "max_replicates", minimum=1, maximum=1024)
    max_bytes = integer(max_bytes, "max_bytes", minimum=1)
    workspace = oracle.workspace
    workspace.check_status()
    oracle._objective.check_status()
    context = workspace.context
    wp = context.wp
    dimension = len(oracle.parameters)
    count = max_replicates * workspace.max_histories
    pixels = max_replicates * workspace.spec.detector.pixels
    columns = min(FUSED_PARAMETER_LIMIT, dimension)
    batched_models = oracle.supports_local_model and dimension <= BATCHED_MODEL_PARAMETER_LIMIT
    if max(count * columns, pixels, max_replicates * dimension) >= 2**31:
        raise TransportError("batched transport storage exceeds signed 32-bit indexing")
    tile_count = (workspace.spec.detector.pixels + 255) // 256
    partial_sizes: list[int] = []
    while True:
        partial_sizes.append(max_replicates * tile_count * (dimension**2 if batched_models else 1))
        if tile_count == 1:
            break
        tile_count = (tile_count + 255) // 256
    required = (
        count * (72 + 8 * columns)
        + pixels * 32
        + 32 * max_replicates
        + 16 * max_replicates * (dimension + 1)
        + 8 * sum(partial_sizes)
        + (
            8 * (pixels * dimension + max_replicates * dimension * (dimension + 1))
            if batched_models
            else 0
        )
    )
    if max(partial_sizes) >= 2**31 or (batched_models and pixels * dimension >= 2**31):
        raise TransportError("batched transport model storage exceeds signed 32-bit indexing")
    if required > max_bytes:
        raise TransportError(
            f"batched transport needs {required} owned device bytes, exceeding preparation budget"
        )
    kernels = load_kernels("dpt.transport.batched_kernels")
    source = kernels.SourceParameters()
    if isinstance(oracle.source, ParallelBeam):
        beam = oracle.source
        source.kind = 0
        source.lower = wp.vec3d(*beam.lower_mm)
        source.direction = wp.vec3d(*beam.direction)
        source.extent = wp.vec2d(*beam.extent_xy_mm)
        source.weight = beam.weight
    else:
        cone = oracle.source
        geometry = cone.geometry
        source.kind = 1
        source.position = wp.vec3d(*geometry.source_mm)
        source.lower = wp.vec3d(
            *(
                geometry.origin_mm[i]
                - 0.5
                * (geometry.spacing_mm[0] * geometry.u[i] + geometry.spacing_mm[1] * geometry.v[i])
                for i in range(3)
            )
        )
        source.u, source.v = wp.vec3d(*geometry.u), wp.vec3d(*geometry.v)
        source.extent = wp.vec2d(
            geometry.spacing_mm[0] * geometry.shape[1], geometry.spacing_mm[1] * geometry.shape[0]
        )
        source.weight = cone.weight
    arrays: dict[str, Any] = {}
    with context.scope():
        for name, dtype in (
            ("position", wp.vec3d),
            ("direction", wp.vec3d),
            ("weight", wp.float64),
            ("pixel", wp.int32),
            ("history_status", wp.int32),
            ("score", wp.float64),
        ):
            arrays[name] = wp.empty(count, dtype=dtype, device=context.device)
        arrays["derivatives"] = wp.empty(count * columns, dtype=wp.float64, device=context.device)
        for name in ("mean_a", "mean_b", "scale", "sum"):
            arrays[name] = wp.empty(pixels, dtype=wp.float64, device=context.device)
        for name in ("seed_a", "first_a", "seed_b", "first_b"):
            arrays[name] = wp.empty(max_replicates, dtype=wp.uint64, device=context.device)
        arrays["gradient"] = wp.empty(
            max_replicates * dimension, dtype=wp.float64, device=context.device
        )
        arrays["loss"] = wp.empty(max_replicates, dtype=wp.float64, device=context.device)
        arrays["previous_gradient"] = wp.empty(
            max_replicates * dimension, dtype=wp.float64, device=context.device
        )
        arrays["previous_loss"] = wp.empty(max_replicates, dtype=wp.float64, device=context.device)
        partials = tuple(
            wp.empty(size, dtype=wp.float64, device=context.device) for size in partial_sizes
        )
        ids_host = wp.empty(
            max_replicates * 4, dtype=wp.uint64, device="cpu", pinned=context.device.is_cuda
        )
        gradient_host = wp.empty(
            max_replicates * dimension,
            dtype=wp.float64,
            device="cpu",
            pinned=context.device.is_cuda,
        )
        loss_host = wp.empty(
            max_replicates, dtype=wp.float64, device="cpu", pinned=context.device.is_cuda
        )
        model_host = None
        if batched_models:
            arrays["jacobian"] = wp.empty(
                pixels * dimension, dtype=wp.float64, device=context.device
            )
            arrays["model"] = wp.empty(
                max_replicates * dimension * (dimension + 1),
                dtype=wp.float64,
                device=context.device,
            )
            model_host = wp.empty(
                max_replicates * dimension * (dimension + 1),
                dtype=wp.float64,
                device="cpu",
                pinned=context.device.is_cuda,
            )
    return BatchedTransportSquaredOracle(
        oracle,
        max_replicates,
        kernels,
        source,
        arrays,
        partials,
        load_kernels("dpt.transport.recovery_kernels"),
        ids_host,
        ids_host.numpy().reshape(4, max_replicates),
        gradient_host,
        gradient_host.numpy().reshape(max_replicates, dimension),
        loss_host,
        loss_host.numpy(),
        _model_kernels=load_kernels("dpt.transport.batched_model_kernels")
        if model_host is not None
        else None,
        _model_host=model_host,
        _model_view=model_host.numpy().reshape(max_replicates, dimension * (dimension + 1))
        if model_host is not None
        else None,
    )
