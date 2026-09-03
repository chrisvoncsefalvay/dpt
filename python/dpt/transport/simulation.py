"""Checked transport simulation with resident estimates and typed diagnostics.

Library source samplers assign independent original-history draws. Detector
misses and absorbed histories contribute zero and remain in the denominator.
One prepared workspace can be reused sequentially; it owns the returned images.
"""

from __future__ import annotations

# Launch plumbing is shared with the existing transport workspaces.
# pyright: reportPrivateUsage=false
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._runtime import load_kernels, require_no_tape

from .estimators import EstimatorWorkspace, history_mean, history_moments, prepare_estimators
from .forward import TransportWorkspace, trace_histories
from .model import HistoryStatus, TransportError, _require_instance
from .rng import HistoryBatch
from .source import ParallelBeam, PointConeSource, sample_parallel_beam, sample_point_cone

if TYPE_CHECKING:
    import warp as wp

type StatusCounts = tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class HistoryStatusHistogram:
    """Seven exact counts ordered by ``HistoryStatus``, one per original history.

    An escaped history may miss the detector. Escape and detection are distinct
    events; the histogram accounts for termination, rather than detected photons.
    """

    counts: StatusCounts

    def __post_init__(self) -> None:
        if len(self.counts) != len(HistoryStatus) or any(
            type(value) is not int or value < 0 for value in self.counts
        ):
            raise TransportError("status histogram requires seven nonnegative integer counts")
        object.__setattr__(self, "counts", tuple(self.counts))

    @property
    def total(self) -> int:
        """Number of launched original histories accounted for."""
        return sum(self.counts)

    def count(self, status: HistoryStatus) -> int:
        """Return the count of a typed terminal status."""
        _require_instance(status, HistoryStatus, "histogram status")
        return self.counts[int(status)]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """A completed original-history estimate, with images on its device.

    ``mean`` and ``estimated_variance_of_mean`` are flat FP64 Warp arrays of
    shape ``(pixels,)``; ``image_shape`` gives their row/column layout. For pixel
    p, each launched original history contributes X_i(p), including zero for a
    miss or absorption. At each pixel, ``mean = sum(X_i) / N`` and
    ``estimated_variance_of_mean = sum((X_i - mean)**2) / (N * (N - 1))``.

    The second image estimates marginal variance of the mean, rather than
    per-history variance or the exact sampling variance. Detector pixels may
    be correlated. At least two histories and an IID original-history source
    law are required; stratified batches need independent complete replicates.

    Randomised full-path designs return ``estimated_variance_of_mean=None`` and
    report their actual sampling law. Use independent complete design replicates
    with ``design_replicate_moments`` for their uncertainty.

    The arrays alias the prepared workspace. Its next simulation overwrites
    them; copy images you want to retain. Histogram and batch metadata are
    immutable snapshots. No image is downloaded by ``simulate``.
    """

    mean: wp.array[wp.float64, Literal[1]]
    estimated_variance_of_mean: wp.array[wp.float64, Literal[1]] | None
    status_histogram: HistoryStatusHistogram
    batch: HistoryBatch
    image_shape: tuple[int, int]
    device: str
    estimator: Literal["analogue", "continuous-absorption"]
    scoring: Literal["photon-count", "energy-kev"]
    sampling_law: Literal["iid_original_histories", "stratified", "scrambled-sobol"] = (
        "iid_original_histories"
    )


@dataclass(slots=True)
class SimulationWorkspace:
    """Bounded source/history buffers, moments and seven-count host staging.

    Returned estimate images remain owned by this workspace. Use each instance
    sequentially on its transport workspace's owning device and stream.
    """

    transport: TransportWorkspace
    max_histories: int
    mean: wp.array[wp.float64, Literal[1]]
    estimated_variance_of_mean: wp.array[wp.float64, Literal[1]]
    estimators: EstimatorWorkspace
    _position: Any = field(repr=False)
    _direction: Any = field(repr=False)
    _weight: Any = field(repr=False)
    _histories: dict[str, Any] = field(repr=False)
    _histogram: Any = field(repr=False)
    _histogram_host: Any = field(repr=False)
    _diagnostics: Any = field(repr=False)

    @property
    def allocated_bytes(self) -> int:
        """Additional source/history, image and diagnostic bytes, including host staging."""
        return 84 * self.max_histories + 36 * self.transport.spec.detector.pixels + 56


def prepare_simulation(
    transport: TransportWorkspace,
    *,
    max_histories: int | None = None,
    max_bytes: int = 256 * 1024**2,
    tally_strategy: Literal["plain", "grouped"] = "plain",
) -> SimulationWorkspace:
    """Allocate reusable buffers for ``simulate`` on the transport's owning device.

    The additional-byte budget covers source arrays, one sparse record per
    history, FP64 mean/variance images, moment scratch and seven status counts
    on device and host. Material arrays belong to ``transport`` and are excluded.
    Plain atomics are the default; ``grouped`` is an explicit workload choice.
    """
    require_no_tape()
    _require_instance(transport, TransportWorkspace, "prepare_simulation transport")
    capacity = transport.max_histories if max_histories is None else max_histories
    if type(capacity) is not int or not 2 <= capacity <= transport.max_histories:
        raise TransportError("simulation capacity must be 2..transport.max_histories")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise TransportError("simulation max_bytes must be a positive integer")
    required = 84 * capacity + 36 * transport.spec.detector.pixels + 56
    if required > max_bytes:
        raise TransportError(f"simulation requires {required} bytes, exceeding max_bytes")
    if tally_strategy not in ("plain", "grouped"):
        raise TransportError("unsupported tally strategy")
    context = transport.context
    wp = context.wp
    diagnostics = load_kernels("dpt.kernels.transport_diagnostics")
    with context.scope():
        position = wp.empty(capacity, dtype=wp.vec3d, device=context.device)
        direction = wp.empty_like(position)
        weight = wp.empty(capacity, dtype=wp.float64, device=context.device)
        histories = {
            f"out_{name}": wp.empty(capacity, dtype=dtype, device=context.device)
            for name, dtype in (
                ("pixel", wp.int32),
                ("score", wp.float64),
                ("energy", wp.float64),
                ("events", wp.int32),
                ("status", wp.int32),
            )
        }
        mean = wp.empty(transport.spec.detector.pixels, dtype=wp.float64, device=context.device)
        variance = wp.empty_like(mean)
        histogram = wp.zeros(len(HistoryStatus), dtype=wp.int32, device=context.device)
        histogram_host = wp.empty(
            len(HistoryStatus), dtype=wp.int32, device="cpu", pinned=context.device.is_cuda
        )
    return SimulationWorkspace(
        transport,
        capacity,
        mean,
        variance,
        prepare_estimators(transport, tally_strategy=tally_strategy),
        position,
        direction,
        weight,
        histories,
        histogram,
        histogram_host,
        diagnostics,
    )


def simulate(
    density: Any,
    *,
    source: ParallelBeam | PointConeSource,
    batch: HistoryBatch,
    workspace: SimulationWorkspace,
    source_amplitude: float = 1.0,
    stream: Any = None,
) -> SimulationResult:
    """Mean[p]=sum_i X_i[p]/N estimates the detector expectation per original history.

    Density is a positive dimensionless FP64 (materials,) array and source is a
    fixed ParallelBeam or PointConeSource in world mm/rad. Batch contains the
    original-history identities and sampling design, and source_amplitude scales
    complete scores. SimulationResult references workspace-owned FP64 mean
    (pixels,) in photon-count/keV score units, image_shape (rows,columns), immutable
    batch/histogram metadata and the selected sampling law. For IID N >= 2 it
    also provides marginal variance-of-mean M2/(N*(N-1)). A non-IID complete
    design returns its mean without within-design variance and requires independent
    complete-design/scramble replicates for uncertainty. All histories, including
    misses/absorption, stay in N. Completion transfers status words and seven
    terminal counts while images remain resident. No device array is allocated
    per call and the next simulation overwrites the returned image storage.

    """
    require_no_tape()
    _require_instance(workspace, SimulationWorkspace, "simulate workspace")
    _require_instance(batch, HistoryBatch, "simulate batch")
    if not 2 <= batch.count <= workspace.max_histories:
        raise TransportError("simulation batch must contain 2..workspace.max_histories histories")
    if not isinstance(cast(Any, source), (ParallelBeam, PointConeSource)):
        raise TransportError("simulate requires a library ParallelBeam or PointConeSource")
    transport = workspace.transport
    context = transport.context
    wp = context.wp
    context.assert_stream(stream)
    context.array(density, "density", dtype=wp.float64, shape=(transport.spec.grid.materials,))
    context.disjoint(
        [*transport._reads(), ("density", density)],
        [
            ("simulation_position", workspace._position),
            ("simulation_direction", workspace._direction),
            ("simulation_weight", workspace._weight),
            *workspace._histories.items(),
            ("simulation_mean", workspace.mean),
            ("simulation_variance", workspace.estimated_variance_of_mean),
            ("simulation_histogram", workspace._histogram),
            ("simulation_moment_sum", workspace.estimators._sum),
            ("simulation_moment_scale", workspace.estimators._scale),
            ("simulation_moment_hits", workspace.estimators._hits),
        ],
    )
    positions = workspace._position[: batch.count]
    directions = workspace._direction[: batch.count]
    weights = workspace._weight[: batch.count]
    outputs = {name: value[: batch.count] for name, value in workspace._histories.items()}
    if isinstance(source, ParallelBeam):
        identity = sample_parallel_beam(
            source,
            batch=batch,
            workspace=transport,
            out_position=positions,
            out_direction=directions,
            out_weight=weights,
            stream=stream,
        )
    else:
        identity = sample_point_cone(
            source,
            batch=batch,
            workspace=transport,
            out_position=positions,
            out_direction=directions,
            out_weight=weights,
            stream=stream,
        )
    batch = batch.with_library_source(identity.sampler)
    trace_histories(
        positions,
        directions,
        weights,
        density,
        batch=batch,
        workspace=transport,
        source_amplitude=source_amplitude,
        stream=stream,
        **outputs,
    )
    moment_operation = cast(Any, history_moments if batch.sampling is None else history_mean)
    moment_operation(
        outputs["out_pixel"],
        outputs["out_score"],
        outputs["out_status"],
        batch=batch,
        workspace=workspace.estimators,
        out_mean=workspace.mean,
        **(
            {"out_variance_of_mean": workspace.estimated_variance_of_mean}
            if batch.sampling is None
            else {}
        ),
        stream=stream,
    )
    with context.scope():
        workspace._histogram.zero_()
        transport._launch(
            workspace._diagnostics.status_histogram,
            batch.count,
            [outputs["out_status"], workspace._histogram, transport._status],
        )
        wp.copy(workspace._histogram_host, workspace._histogram, stream=context.stream)
        if context.device.is_cuda:
            wp.synchronize_stream(context.stream)
    transport.check_status()
    counts = tuple(int(value) for value in workspace._histogram_host.numpy())
    histogram = HistoryStatusHistogram(
        (counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6])
    )
    if histogram.total != batch.count:
        raise TransportError("terminal status histogram does not account for every history")
    return SimulationResult(
        workspace.mean,
        workspace.estimated_variance_of_mean if batch.sampling is None else None,
        histogram,
        batch,
        transport.spec.detector.image_shape,
        str(context.device),
        transport.spec.estimator,
        transport.spec.scoring,
        "iid_original_histories" if batch.sampling is None else batch.sampling.design.law,
    )
