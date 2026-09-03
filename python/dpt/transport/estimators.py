"""Original-history uncertainty and independent products for stochastic inversion.

Detector misses count as zero histories. Photon branches are not independent
samples, and plug-in squared sample means are not unbiased squared expectations.
The product estimators below explicitly require disjoint random-stream identities.
"""

from __future__ import annotations

# Workspace internals are shared only within this package.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from dpt._runtime import require_no_tape
from dpt.statistics import mean_standard_error

from .forward import TransportWorkspace, _validate_call
from .model import TransportError
from .rng import HistoryBatch, require_independent

_NO_VARIANCE = object()


@dataclass(slots=True)
class EstimatorWorkspace:
    """Reusable detector moments; no per-call allocations or hidden downloads."""

    transport: TransportWorkspace
    _sum: Any = field(repr=False)
    _hits: Any = field(repr=False)
    _scale: Any = field(repr=False)
    tally_strategy: Literal["grouped", "plain"] = "plain"

    @property
    def scratch_bytes(self) -> int:
        return 20 * self.transport.spec.detector.pixels


def prepare_estimators(
    workspace: TransportWorkspace, *, tally_strategy: Literal["grouped", "plain"] = "plain"
) -> EstimatorWorkspace:
    """Allocate binary64 sum/scale images and int32 hit counts on the owning stream.

    Plain atomics are the default. Select ``grouped`` explicitly after measuring
    its benefit for the intended detector occupancy and device.
    """
    require_no_tape()
    if tally_strategy not in ("grouped", "plain"):
        raise TransportError("unsupported tally strategy")
    context = workspace.context
    with context.scope():
        sums = context.wp.empty(
            workspace.spec.detector.pixels, dtype=context.wp.float64, device=context.device
        )
        hits = context.wp.empty(
            workspace.spec.detector.pixels, dtype=context.wp.int32, device=context.device
        )
        scale = context.wp.empty_like(sums)
    return EstimatorWorkspace(workspace, sums, hits, scale, tally_strategy)


@dataclass(slots=True)
class FusedMeanWorkspace:
    """Optional mean-only transport readout with declared per-pixel score bounds.

    Scratch depends on detector pixels, with no per-history outputs. Bounds are
    immutable positive finite upper bounds on each original-history score. A
    violated bound or a lossy normalisation fails the entire estimate.
    """

    transport: TransportWorkspace
    score_bound: Any
    tally_strategy: Literal["grouped", "plain"]
    _sum: Any = field(repr=False)

    @property
    def scratch_bytes(self) -> int:
        return 8 * self.transport.spec.detector.pixels

    def check_status(self) -> None:
        self.transport.check_status()


def prepare_fused_mean(
    workspace: TransportWorkspace,
    *,
    score_bound: Any,
    tally_strategy: Literal["grouped", "plain"] = "plain",
) -> FusedMeanWorkspace:
    """Bind explicit bounded scoring for an allocation-free mean-only readout."""
    require_no_tape()
    if tally_strategy not in ("grouped", "plain"):
        raise TransportError("unsupported tally strategy")
    context = workspace.context
    context.array(
        score_bound,
        "score_bound",
        dtype=context.wp.float64,
        shape=(workspace.spec.detector.pixels,),
    )
    workspace.check_status()
    workspace._launch(
        workspace._kernels.validate_tally_bound,
        workspace.spec.detector.pixels,
        [score_bound, workspace._status],
    )
    workspace.check_status()
    with context.scope():
        sums = context.wp.empty_like(score_bound)
    return FusedMeanWorkspace(workspace, score_bound, tally_strategy, sums)


def trace_mean(
    positions: Any,
    directions: Any,
    weights: Any,
    density: Any,
    *,
    batch: HistoryBatch,
    workspace: FusedMeanWorkspace,
    out_mean: Any,
    source_amplitude: float = 1.0,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Out_mean[p] = sum_i X_i[p]/N tallies complete original-history scores directly.

    Positions/directions are vec3d (N,) in world mm/unit directions, weights is
    FP64 (N,) and density is FP64 (materials,) positive dimensionless coefficient
    scales. Batch gives the original history identity and N, including misses
    and absorption. Out_mean is FP64 (pixels,) in score units per original history,
    with photon-count or keV scoring and explicit source_amplitude. The fused
    workspace binds a finite positive FP64 score_bound (pixels,) covering every
    complete-history score. One canonical walk forms the mean without history
    records or per-call allocation. This exposes no within-batch variance or
    derivative. IID histories may use the separate moment path and stratified/Sobol
    designs require independent complete-design replicates for sampling uncertainty.
    Unchecked calls require a completed workspace.check_status().

    """
    transport = workspace.transport
    context = transport.context
    _validate_call(
        transport,
        batch,
        positions,
        directions,
        weights,
        density,
        [],
        source_amplitude,
        stream,
        validate,
    )
    context.array(
        out_mean, "out_mean", dtype=context.wp.float64, shape=(transport.spec.detector.pixels,)
    )
    context.disjoint(
        [
            *transport._reads(),
            ("positions", positions),
            ("directions", directions),
            ("weights", weights),
            ("density", density),
            ("score_bound", workspace.score_bound),
        ],
        [("out_mean", out_mean), ("fused_mean_sum", workspace._sum)],
    )
    with context.scope():
        workspace._sum.zero_()
    wp = context.wp
    transport._launch(
        transport._kernels.trace_tally,
        batch.count,
        [
            positions,
            directions,
            weights,
            density,
            *transport._model_inputs(),
            wp.uint64(batch.seed),
            wp.uint64(batch.first_history),
            source_amplitude,
            workspace.score_bound,
            int(workspace.tally_strategy == "grouped"),
            workspace._sum,
            transport._status,
        ],
    )
    transport._launch(
        transport._kernels.finish_bounded_mean,
        transport.spec.detector.pixels,
        [workspace._sum, workspace.score_bound, batch.count, out_mean, transport._status],
    )
    if validate:
        workspace.check_status()


def _accumulate_history_mean(
    pixel: Any,
    score: Any,
    history_status: Any,
    *,
    batch: HistoryBatch,
    workspace: EstimatorWorkspace,
    out_mean: Any,
    out_variance_of_mean: Any = _NO_VARIANCE,
    stream: Any,
    validate: bool,
) -> None:
    """Validate shared ownership, clear scratch and launch scaled original-history means."""
    transport = workspace.transport
    context = transport.context
    context.assert_stream(stream)
    require_no_tape()
    wp = context.wp
    pixels = transport.spec.detector.pixels
    for name, value, dtype in (
        ("pixel", pixel, wp.int32),
        ("score", score, wp.float64),
        ("history_status", history_status, wp.int32),
    ):
        context.array(value, name, dtype=dtype, shape=(batch.count,))
    outputs = [("out_mean", out_mean)]
    scratch = [("moment_sum", workspace._sum), ("moment_scale", workspace._scale)]
    if out_variance_of_mean is not _NO_VARIANCE:
        outputs.append(("out_variance_of_mean", out_variance_of_mean))
        scratch.append(("moment_hits", workspace._hits))
    for name, value in outputs:
        context.array(value, name, dtype=wp.float64, shape=(pixels,))
    context.disjoint(
        [
            *transport._reads(),
            ("pixel", pixel),
            ("score", score),
            ("history_status", history_status),
        ],
        outputs + scratch,
    )
    if validate:
        transport.check_status()
    with context.scope():
        workspace._sum.zero_()
        workspace._scale.zero_()
        if out_variance_of_mean is not _NO_VARIANCE:
            workspace._hits.zero_()
            out_variance_of_mean.zero_()
    transport._launch(
        transport._kernels.find_tally_scale
        if workspace.tally_strategy == "grouped"
        else transport._kernels.find_tally_scale_plain,
        batch.count,
        [pixel, score, history_status, pixels, workspace._scale, transport._status],
    )
    transport._launch(
        transport._kernels.tally_sum
        if workspace.tally_strategy == "grouped"
        else transport._kernels.tally_sum_plain,
        batch.count,
        [pixel, score, history_status, pixels, workspace._scale, workspace._sum, transport._status],
    )
    transport._launch(
        transport._kernels.finish_mean,
        pixels,
        [workspace._sum, workspace._scale, batch.count, out_mean, transport._status],
    )


def history_moments(
    pixel: Any,
    score: Any,
    history_status: Any,
    *,
    batch: HistoryBatch,
    workspace: EstimatorWorkspace,
    out_mean: Any,
    out_variance_of_mean: Any,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Mean[p]=sum_i X_i[p]/N and Var_mean[p]=sum_i (X_i[p]-Mean[p])^2/(N*(N-1)).

    Pixel/history_status are int32 (N,) and score is FP64 (N,) sparse original-history
    storage. Each history contributes its score to one pixel and zero elsewhere.
    Out_mean and out_variance_of_mean are disjoint FP64 (pixels,), with score
    units per original history and squared score units. N >= 2 and the declared
    sampling law must be IID original histories with finite second moments.
    FP64 scale/centre passes preserve the result range without raw squared sums.
    Non-IID stratified/Sobol histories cannot use this within-history variance:
    use independent complete-design replicates. The outputs are marginal
    variances and do not remove covariance between pixels. Splitting descendants
    must first be combined by original history. Failed statuses invalidate all
    outputs and no parameter derivative is formed.

    """
    if batch.sampling is not None:
        raise TransportError(
            "dependent path designs require independent complete-batch replicate variance"
        )
    if batch.count < 2:
        raise TransportError("variance of a mean requires at least two original histories")
    _accumulate_history_mean(
        pixel,
        score,
        history_status,
        batch=batch,
        workspace=workspace,
        out_mean=out_mean,
        out_variance_of_mean=out_variance_of_mean,
        stream=stream,
        validate=validate,
    )
    transport = workspace.transport
    pixels = transport.spec.detector.pixels
    transport._launch(
        transport._kernels.centred_moments
        if workspace.tally_strategy == "grouped"
        else transport._kernels.centred_moments_plain,
        batch.count,
        [
            pixel,
            score,
            out_mean,
            workspace._scale,
            out_variance_of_mean,
            workspace._hits,
            transport._status,
        ],
    )
    transport._launch(
        transport._kernels.finish_variance,
        pixels,
        [
            out_mean,
            workspace._scale,
            workspace._hits,
            batch.count,
            out_variance_of_mean,
            transport._status,
        ],
    )
    if validate:
        transport.check_status()


def _product(
    mean_a: Any,
    other_b: Any,
    observed: Any,
    weights: Any,
    *,
    batches: tuple[HistoryBatch, HistoryBatch],
    workspace: TransportWorkspace,
    out_components: Any,
    loss: bool,
    stream: Any,
    validate: bool,
) -> None:
    require_independent(*batches)
    require_no_tape()
    context = workspace.context
    context.assert_stream(stream)
    arrays = [
        ("mean_a", mean_a),
        ("other_b", other_b),
        ("observed", observed),
        ("weights", weights),
    ]
    shape = (workspace.spec.detector.pixels,)
    for name, value in [*arrays, ("out_components", out_components)]:
        context.array(value, name, dtype=context.wp.float64, shape=shape)
    context.disjoint(workspace._reads() + arrays, [("out_components", out_components)])
    if validate:
        workspace.check_status()
    workspace._launch(
        workspace._kernels.independent_product,
        shape[0],
        [mean_a, other_b, observed, weights, int(loss), out_components, workspace._status],
    )
    if validate:
        workspace.check_status()


def independent_squared_gradient(
    mean_a: Any,
    derivative_mean_b: Any,
    observed: Any,
    weights: Any,
    *,
    batches: tuple[HistoryBatch, HistoryBatch],
    workspace: TransportWorkspace,
    out_components: Any,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """G[p] = weights[p]*(mean_a[p]-observed[p])*derivative_mean_b[p].

    All five arrays, including out_components, are disjoint FP64 (pixels,).
    Mean_a is an unbiased expected-score estimate and derivative_mean_b is an
    independent unbiased derivative for one declared parameter coordinate.
    Observed is a fixed finite score and weights is fixed nonnegative loss
    weight, giving gradient components in loss per parameter-coordinate units.
    The two HistoryBatch identities must be independent in transport and source
    draws. A shared-history product generally adds covariance and is rejected.
    FP64 component arithmetic overwrites caller storage without reducing pixels.
    Sum components on device and estimate scalar-gradient uncertainty across
    independent product replicates to retain detector-pixel covariance.

    """
    _product(
        mean_a,
        derivative_mean_b,
        observed,
        weights,
        batches=batches,
        workspace=workspace,
        out_components=out_components,
        loss=False,
        stream=stream,
        validate=validate,
    )


def independent_squared_loss(
    mean_a: Any,
    mean_b: Any,
    observed: Any,
    weights: Any,
    *,
    batches: tuple[HistoryBatch, HistoryBatch],
    workspace: TransportWorkspace,
    out_components: Any,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """F[p] = weights[p]*(mean_a[p]-observed[p])*(mean_b[p]-observed[p])/2.

    Inputs and out_components are disjoint FP64 (pixels,) arrays. The two mean
    estimates must be unbiased and their supplied HistoryBatch identities must
    separate transport/source draws. Observed is a fixed finite score and weights
    is fixed nonnegative, giving component units of weighted squared score.
    FP64 local products overwrite caller storage and are summed separately.
    Individual estimates may be negative and clipping them would bias the loss.
    Candidate/incumbent evaluations may share random numbers within each factor,
    while the two factors remain independent. Sampling uncertainty comes from
    independent complete product replicates, including detector-pixel covariance.
    No parameter gradient is formed here.

    """
    _product(
        mean_a,
        mean_b,
        observed,
        weights,
        batches=batches,
        workspace=workspace,
        out_components=out_components,
        loss=True,
        stream=stream,
        validate=validate,
    )


@dataclass(frozen=True, slots=True)
class ReplicateEstimate:
    """A scalar checkpoint, after device reduction and explicit host transfer."""

    value: float
    batches: tuple[HistoryBatch, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or not self.batches:
            raise TransportError(
                "replicate estimates need a finite value and their batch identities"
            )
        require_independent(*self.batches)


@dataclass(frozen=True, slots=True)
class MeanUncertainty:
    mean: float
    standard_error: float
    replicates: int


def summarise_replicates(estimates: tuple[ReplicateEstimate, ...]) -> MeanUncertainty:
    """Mean=sum_r estimate[r]/R and SE=sqrt(sum_r (estimate[r]-Mean)^2/(R*(R-1))).

    Estimates is a tuple of R >= 2 ReplicateEstimate records containing scalar
    binary64 values and their distinct HistoryBatch identities. MeanUncertainty
    returns scalar mean/standard_error in estimate units and the integer replicate
    count. Different records must use independent complete sampling units.
    Paired candidate-minus-incumbent changes may use common random numbers within
    a record, with each distinct batch listed once. Stratified or scrambled-Sobol
    uncertainty likewise uses complete independent designs/scrambles rather than
    dependent points inside one design. This host checkpoint uses range-preserving
    binary64 summary arithmetic and performs no device transfers or derivatives.

    >>> from dpt.transport.estimators import ReplicateEstimate, summarise_replicates
    >>> from dpt.transport.rng import HistoryBatch
    >>> estimates = tuple(
    ...     ReplicateEstimate(value, (HistoryBatch(10 + i, 0, 8),))
    ...     for i, value in enumerate((1.0, 2.0, 3.0))
    ... )
    >>> result = summarise_replicates(estimates)
    >>> (result.mean, round(result.standard_error, 6), result.replicates)
    (2.0, 0.57735, 3)

    """
    if len(estimates) < 2:
        raise TransportError("uncertainty requires at least two independent replicates")
    for index, estimate in enumerate(estimates):
        for other in estimates[index + 1 :]:
            require_independent(*estimate.batches, *other.batches)
    count = len(estimates)
    mean, standard_error = mean_standard_error(tuple(estimate.value for estimate in estimates))
    return MeanUncertainty(mean, standard_error, count)


def history_mean(
    pixel: Any,
    score: Any,
    history_status: Any,
    *,
    batch: HistoryBatch,
    workspace: EstimatorWorkspace,
    out_mean: Any,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Out_mean[p] = sum_i score[i]*1(pixel[i]=p)/N includes zero-scoring histories.

    Pixel/history_status are int32 (N,) sparse original-history records and score
    is FP64 (N,) in the transport's declared photon-count/keV score units. Batch
    retains their original identities and denominator N, including misses and
    absorption. Out_mean is disjoint FP64 (pixels,) on the owning device, flattened
    in row,column order. FP64 tally arithmetic overwrites the mean without forming
    squared scores. One history is valid for mean-only use. IID or non-IID
    complete designs may supply a mean, but sampling uncertainty for stratified/
    Sobol designs comes from independent complete-design replicates. Failed
    resource/energy/numerical statuses invalidate the estimate. No parameter
    gradient is formed by this reduction.

    """
    _accumulate_history_mean(
        pixel,
        score,
        history_status,
        batch=batch,
        workspace=workspace,
        out_mean=out_mean,
        stream=stream,
        validate=validate,
    )
    if validate:
        workspace.transport.check_status()
