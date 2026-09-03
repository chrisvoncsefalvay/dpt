"""Prepared CUDA replay of one explicitly identified original-history batch.

The captured source, batch and launch dimensions stay fixed. Updating the bound
density buffer changes the physical coefficients; replaying unchanged inputs
reuses the same random sample. Completion checks happen outside capture.
"""

from __future__ import annotations

# Resident launch plumbing is shared with prepared transport and simulation.
# pyright: reportPrivateUsage=false
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._runtime import require_no_tape

from .estimators import history_moments
from .forward import _validate_call, trace_histories
from .model import TransportError, _require_instance
from .rng import HistoryBatch
from .simulation import (
    HistoryStatusHistogram,
    SimulationResult,
    SimulationWorkspace,
    simulate,
)
from .source import ParallelBeam, PointConeSource, sample_parallel_beam, sample_point_cone

if TYPE_CHECKING:
    import warp as wp


def _arrays(workspace: SimulationWorkspace, density: Any) -> tuple[Any, ...]:
    transport = workspace.transport
    return (
        density,
        transport.material_ids,
        transport.voxel_density,
        transport.absorption,
        transport.scattering,
        transport._energies,
        transport._status,
        workspace._position,
        workspace._direction,
        workspace._weight,
        *workspace._histories.values(),
        workspace.mean,
        workspace.estimated_variance_of_mean,
        workspace.estimators._sum,
        workspace.estimators._scale,
        workspace.estimators._hits,
        workspace._histogram,
        workspace._histogram_host,
    )


def _binding_identity(workspace: SimulationWorkspace, density: Any) -> tuple[Any, ...]:
    """Capture metadata only; content checks remain device operations."""
    transport = workspace.transport
    context = transport.context
    return (
        id(transport),
        id(context),
        id(transport.spec),
        id(transport._parameters),
        id(workspace.estimators),
        id(workspace.estimators.transport),
        workspace.max_histories,
        transport.max_histories,
        workspace.estimators.tally_strategy,
        *(
            (
                0 if array.ptr is None else int(array.ptr),
                tuple(array.shape),
                array.dtype,
                str(array.device),
            )
            for array in _arrays(workspace, density)
        ),
    )


@dataclass(frozen=True, slots=True)
class TransportGraph:
    """Replay mean and estimated variance for one fixed IID history identity.

    ``density`` is a bound resident FP64 array of shape ``(materials,)``.
    Update its values on the owning stream before ``replay``; all material,
    source, geometry, amplitude and batch metadata remain fixed. The returned
    images alias the simulation workspace, as they do for ``simulate``.

    Every replay samples the same source and transport counter addresses. It
    is useful for comparing coefficient changes, but repeated unchanged replay
    is not an independent replicate for estimating sampling uncertainty.

    Example::

        graph = prepare_transport_graph(
            density, source=source, batch=batch, workspace=simulation
        )
        result = graph.replay()
        density.fill_(1.2)  # order this update on the graph's owning stream
        changed = graph.replay()

    Use the workspace sequentially. Keep result images by copying them before
    replay. Source/model buffers retain the existing immutable-input contract.
    Supported likelihood-ratio derivatives can replay the same ``batch`` and
    source buffers through the existing derivative API after this forward
    replay; this helper captures forward moments only.
    """

    density: wp.array[wp.float64, Literal[1]]
    source: ParallelBeam | PointConeSource
    batch: HistoryBatch
    source_amplitude: float
    _workspace: SimulationWorkspace = field(repr=False)
    _graph: Any = field(repr=False)
    _positions: Any = field(repr=False)
    _directions: Any = field(repr=False)
    _weights: Any = field(repr=False)
    _histories: dict[str, Any] = field(repr=False)
    _identity: tuple[Any, ...] = field(repr=False)
    _retained_arrays: tuple[Any, ...] = field(repr=False)

    def replay(self, *, stream: Any = None) -> SimulationResult:
        """Mean[p]=sum_i X_i[p]/N replays the captured complete-batch detector estimator.

        The graph retains a fixed source, amplitude, HistoryBatch, FP64 density
        (materials,) pointer and prepared SimulationWorkspace on one CUDA stream.
        Current positive density contents are validated before replay and the batch
        identity does not advance. Returned SimulationResult images are FP64
        (pixels,) in score units, with row/column image_shape and complete terminal
        counts. IID variance uses M2/(N*(N-1)); non-IID design uncertainty requires
        independent complete designs. The graph owns no derivative product.
        Shape/pointer/alias checks and pending diagnostics precede launch, then
        completion status/histogram reads occur outside capture. No device allocation
        or image download occurs and subsequent replay overwrites result images.

        """
        require_no_tape()
        workspace = self._workspace
        if _binding_identity(workspace, self.density) != self._identity:
            raise TransportError("transport graph bindings changed; prepare a new graph")
        transport = workspace.transport
        context = transport.context
        context.assert_stream(stream)
        _validate_call(
            transport,
            self.batch,
            self._positions,
            self._directions,
            self._weights,
            self.density,
            [],
            self.source_amplitude,
            stream,
            True,
        )
        with context.scope():
            context.wp.capture_launch(self._graph, stream=context.stream)
        transport.check_status()
        with context.scope():
            context.wp.copy(workspace._histogram_host, workspace._histogram, stream=context.stream)
            context.wp.synchronize_stream(context.stream)
        counts = tuple(int(value) for value in workspace._histogram_host.numpy())
        histogram = HistoryStatusHistogram(
            (counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6])
        )
        if histogram.total != self.batch.count:
            raise TransportError("graph terminal histogram does not account for every history")
        return SimulationResult(
            workspace.mean,
            workspace.estimated_variance_of_mean,
            histogram,
            self.batch,
            transport.spec.detector.image_shape,
            str(context.device),
            transport.spec.estimator,
            transport.spec.scoring,
        )


def prepare_transport_graph(
    density: Any,
    *,
    source: ParallelBeam | PointConeSource,
    batch: HistoryBatch,
    workspace: SimulationWorkspace,
    source_amplitude: float = 1.0,
    stream: Any = None,
) -> TransportGraph:
    """Capture ``mean=sum(X)/N`` and ``variance=M2/(N*(N-1))`` on CUDA.

    Requires one library IID batch with N>=2 and a prepared simulation workspace.
    One checked warm simulation validates inputs/completion and compiles the
    selected source/transport/moment/histogram variants before capture. It
    overwrites the workspace's images. Captured operations perform no host
    readback, and each later replay validates mutable density before launch.

    The density pointer, fixed source, source amplitude, original-history batch,
    model and workspace bindings remain fixed for the graph lifetime. For a new
    batch identity, prepare another graph; this helper never advances RNG state.
    No CUDA graph is simulated on CPU: use ``simulate`` for the CPU reference.

    Example::

        captured = prepare_transport_graph(
            density,
            source=source,
            batch=HistoryBatch(17, 0, 4096),
            workspace=simulation,
        )
        estimate = captured.replay()
    """
    require_no_tape()
    _require_instance(workspace, SimulationWorkspace, "transport graph workspace")
    _require_instance(batch, HistoryBatch, "transport graph batch")
    if not isinstance(cast(Any, source), (ParallelBeam, PointConeSource)):
        raise TransportError("transport graph requires a library IID source")
    transport = workspace.transport
    context = transport.context
    context.assert_stream(stream)
    if not context.device.is_cuda:
        raise TransportError("transport graph requires a CUDA device; use simulate on CPU")
    # Checked warmup preserves any prior sticky failure and uses the existing
    # source, alias, physical-domain and original-history moment contracts.
    warmed = simulate(
        density,
        source=source,
        batch=batch,
        workspace=workspace,
        source_amplitude=source_amplitude,
        stream=stream,
    )
    batch = warmed.batch
    positions = workspace._position[: batch.count]
    directions = workspace._direction[: batch.count]
    weights = workspace._weight[: batch.count]
    histories = {name: value[: batch.count] for name, value in workspace._histories.items()}
    with (
        context.scope(),
        context.wp.ScopedCapture(device=context.device, stream=context.stream) as capture,
    ):
        if isinstance(source, ParallelBeam):
            sample_parallel_beam(
                source,
                batch=batch,
                workspace=transport,
                out_position=positions,
                out_direction=directions,
                out_weight=weights,
                stream=context.stream,
            )
        else:
            sample_point_cone(
                source,
                batch=batch,
                workspace=transport,
                out_position=positions,
                out_direction=directions,
                out_weight=weights,
                stream=context.stream,
            )
        trace_histories(
            positions,
            directions,
            weights,
            density,
            batch=batch,
            workspace=transport,
            source_amplitude=source_amplitude,
            stream=context.stream,
            validate=False,
            **histories,
        )
        history_moments(
            histories["out_pixel"],
            histories["out_score"],
            histories["out_status"],
            batch=batch,
            workspace=workspace.estimators,
            out_mean=workspace.mean,
            out_variance_of_mean=workspace.estimated_variance_of_mean,
            stream=context.stream,
            validate=False,
        )
        workspace._histogram.zero_()
        transport._launch(
            workspace._diagnostics.status_histogram,
            batch.count,
            [histories["out_status"], workspace._histogram, transport._status],
        )
    return TransportGraph(
        density,
        source,
        batch,
        source_amplitude,
        workspace,
        capture.graph,
        positions,
        directions,
        weights,
        histories,
        _binding_identity(workspace, density),
        _arrays(workspace, density),
    )
