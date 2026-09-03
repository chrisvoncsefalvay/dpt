"""Optional finite-sample checkpoints for explicitly bounded IID replicates.

The caller certifies support bounds and independent identically distributed
sampling units. Checking observed samples cannot prove either assertion. This
module does not infer transport bounds or alter Gaussian controller defaults.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Protocol, cast

from dpt.contracts import ContractError, finite_scalar, finite_tuple, integer
from dpt.registration import Vector
from dpt.statistics import mean_standard_error
from dpt.transport.rng import HistoryBatch, require_independent

type HistoryPairs = tuple[tuple[HistoryBatch, HistoryBatch], ...]


@dataclass(frozen=True, slots=True)
class ReplicateBounds:
    """User-certified almost-sure support, declared before sampling.

    These bound the complete scalar estimator returned for one independent
    replicate, including importance weights and common-random-number changes.
    Observed minima/maxima, dependent jackknife pseudovalues and clipped samples
    are not a substitute for an almost-sure estimator bound.
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower = finite_scalar(self.lower, "replicate lower bound")
        upper = finite_scalar(self.upper, "replicate upper bound")
        if lower > upper or not math.isfinite(upper - lower):
            raise ContractError("replicate support must be ordered with a finite width")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class BoundedInterval:
    mean: float
    standard_error: float
    radius: float
    lower: float
    upper: float
    alpha: float
    replicates: int
    method: str = "maurer_pontil_empirical_bernstein_two_sided"


def empirical_bernstein_interval(
    values: Sequence[float], *, bounds: ReplicateBounds, alpha: float
) -> BoundedInterval:
    """r = sqrt(2*s^2*log(4/alpha)/n) + 7*width*log(4/alpha)/(3*(n-1)).

    The interval is [mean-r, mean+r] intersected with the declared support.
    Values is a length-n sequence of scalar IID replicates with n >= 2, bounds
    certifies their almost-sure lower and upper limits, and alpha is a scalar
    error allocation in (0, 1) fixed before sampling. Sample variance s^2 uses
    n-1. The returned BoundedInterval gives binary64 mean, standard error,
    radius and endpoints in sample units, plus alpha and the replicate count.
    Outward rounding preserves the bound after exact rational normalisation.
    Zero empirical variance retains the width term. The calculation implements
    the two-sided form of Maurer and Pontil's empirical Bernstein inequality.

    >>> from dpt.bounded_statistics import ReplicateBounds, empirical_bernstein_interval
    >>> interval = empirical_bernstein_interval(
    ...     (0.2, 0.4, 0.6), bounds=ReplicateBounds(0.0, 1.0), alpha=0.05
    ... )
    >>> (round(interval.mean, 6), interval.lower, interval.upper)
    (0.4, 0.0, 1.0)

    """
    count = integer(len(values), "replicate count", minimum=2)
    alpha = finite_scalar(alpha, "confidence alpha")
    if not 0 < alpha < 1:
        raise ContractError("confidence alpha must lie in (0,1)")
    if not isinstance(cast(object, bounds), ReplicateBounds):
        raise ContractError("provide certified ReplicateBounds before sampling")
    supplied = tuple(finite_scalar(value, "bounded replicate") for value in values)
    if any(value < bounds.lower or value > bounds.upper for value in supplied):
        raise ContractError("a replicate violated its certified support; discard the checkpoint")
    exact = tuple(Fraction.from_float(value) for value in supplied)
    mean = sum(exact, Fraction(0)) / count
    if bounds.lower == bounds.upper:
        return BoundedInterval(float(mean), 0.0, 0.0, bounds.lower, bounds.upper, alpha, count)
    lower = Fraction.from_float(bounds.lower)
    upper = Fraction.from_float(bounds.upper)
    width = upper - lower
    variance = sum(((value - mean) ** 2 for value in exact), Fraction(0)) / ((count - 1) * width**2)
    # Round the variance and confidence arithmetic outward before the endpoints.
    variance_upper = math.nextafter(float(variance), math.inf)
    log_term = math.nextafter(math.log(4.0) - math.log(alpha), math.inf)
    root = math.nextafter(math.sqrt(2 * variance_upper * log_term / count), math.inf)
    linear = math.nextafter(7 * log_term / (3 * (count - 1)), math.inf)
    scaled_radius = min(1.0, math.nextafter(root + linear, math.inf))
    exact_radius = width * Fraction.from_float(scaled_radius)
    interval_lower = max(
        bounds.lower, math.nextafter(float(max(lower, mean - exact_radius)), -math.inf)
    )
    interval_upper = min(
        bounds.upper, math.nextafter(float(min(upper, mean + exact_radius)), math.inf)
    )
    # The diagnostic SE normalises by sample scale, not a possibly much broader
    # certified support, to preserve tiny representable deviations.
    _, error = mean_standard_error(supplied)
    return BoundedInterval(
        mean=float(mean),
        standard_error=error,
        radius=float(exact_radius),
        lower=interval_lower,
        upper=interval_upper,
        alpha=alpha,
        replicates=count,
    )


class BoundedChangeOracle(Protocol):
    """Fixed-proposal IID changes with certified bounds for the actual estimator.

    CRN coupling within each first/second pair across before/after is allowed.
    Distinct pairs must be independent, with the same distribution and history
    count. The oracle must honor both transport and source random identities.
    Bounds are obtained for every planned look before any change is sampled.
    """

    def change_bounds(self, before: Vector, after: Vector, batch_size: int) -> ReplicateBounds: ...

    def change_replicate(
        self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
    ) -> float: ...


class _ChangeOracle(Protocol):
    def change_replicate(
        self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class BoundedChangeAdapter:
    """Attach an explicit caller-certified support to an existing scalar oracle.

    The caller certifies that these bounds hold for every point and history
    count passed through this adapter. It forwards the estimator unchanged.
    Use a custom change_bounds method when the certified support depends on the
    proposal or batch size; never estimate the support from observed samples.
    """

    oracle: _ChangeOracle
    bounds: ReplicateBounds

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.bounds), ReplicateBounds):
            raise ContractError("the adapter requires caller-certified ReplicateBounds")

    def change_bounds(self, before: Vector, after: Vector, batch_size: int) -> ReplicateBounds:
        return self.bounds

    def change_replicate(
        self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
    ) -> float:
        return self.oracle.change_replicate(before, after, first, second)


@dataclass(frozen=True, slots=True)
class BoundedChangePlan:
    """Immutable held-out look schedule and explicit bounded-run error allocation.

    Each look uses fresh pairs rather than pooling adapted samples. A proposal
    may depend on earlier optimisation work, but before/after and the schedule
    are fixed before its independent validation samples. The caller may execute
    at most maximum_proposals plans in the declared run and must keep validation
    identities fresh relative to that earlier work. Identity checks here cover
    the complete supplied schedule, not unseen calls outside the plan.
    """

    looks: tuple[HistoryPairs, ...]
    alpha: float = 0.05
    maximum_proposals: int = 1

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.looks), tuple) or not self.looks:
            raise ContractError("bounded acceptance needs a nonempty immutable look schedule")
        integer(self.maximum_proposals, "maximum proposals", minimum=1)
        alpha = finite_scalar(self.alpha, "run confidence alpha")
        if not 0 < alpha < 1:
            raise ContractError("run confidence alpha must lie in (0,1)")
        object.__setattr__(self, "alpha", alpha)
        previous_replicates = 1
        batch_size = None
        all_batches: list[HistoryBatch] = []
        for pairs in self.looks:
            if not isinstance(cast(object, pairs), tuple) or len(pairs) <= previous_replicates:
                raise ContractError("fresh look replicate counts must be strictly increasing >=2")
            previous_replicates = len(pairs)
            for pair in pairs:
                if (
                    not isinstance(cast(object, pair), tuple)
                    or len(pair) != 2
                    or not all(isinstance(cast(object, batch), HistoryBatch) for batch in pair)
                ):
                    raise ContractError("each replicate must declare two HistoryBatch identities")
                for batch in pair:
                    if batch_size is None:
                        batch_size = batch.count
                    if batch.count != batch_size:
                        raise ContractError(
                            "all planned replicate histories must have equal counts"
                        )
                    all_batches.append(batch)
        require_independent(*all_batches)
        if self.alpha_per_look == 0:
            raise ContractError("the declared confidence budget underflows binary64")

    @property
    def alpha_per_look(self) -> float:
        return self.alpha / self.maximum_proposals / len(self.looks)

    @property
    def reserved_histories(self) -> int:
        return sum(batch.count for pairs in self.looks for pair in pairs for batch in pair)


@dataclass(frozen=True, slots=True)
class BoundedChangeCheckpoint:
    interval: BoundedInterval
    changes: tuple[float, ...]
    pairs: HistoryPairs
    decision: Literal["accepted", "rejected", "ambiguous"]


@dataclass(frozen=True, slots=True)
class BoundedChangeResult:
    decision: Literal["accepted", "rejected", "ambiguous"]
    checkpoints: tuple[BoundedChangeCheckpoint, ...]
    threshold: float
    alpha_per_look: float
    reserved_histories: int
    sampled_histories: int


def validate_bounded_change(
    oracle: BoundedChangeOracle,
    before: Vector,
    after: Vector,
    *,
    plan: BoundedChangePlan,
    threshold: float = 0.0,
) -> BoundedChangeResult:
    """Accept when upper(Delta) < threshold and reject when lower(Delta) >= threshold.

    Before and after are length-D finite binary64 parameter tuples. The oracle
    returns scalar IID loss-change replicates with support certified before each
    look. A BoundedChangePlan declares disjoint HistoryBatch pairs and splits its
    run alpha across the maximum proposals and scheduled looks. Each replicate
    may use common random numbers for before and after, while replicate pairs
    remain independent. The result records scalar intervals, decisions and
    history counts. An exhausted schedule returns ambiguous and leaves parameter
    ownership with the caller. All host arithmetic is binary64.

    >>> from dpt.bounded_statistics import (
    ...     BoundedChangeAdapter,
    ...     BoundedChangePlan,
    ...     ReplicateBounds,
    ...     validate_bounded_change,
    ... )
    >>> from dpt.transport.rng import HistoryBatch
    >>> class ConstantChange:
    ...     def change_replicate(self, before, after, first, second):
    ...         return after[0] - before[0]
    >>> pairs = tuple(
    ...     (HistoryBatch(11, 2 * i, 1), HistoryBatch(11, 2 * i + 1, 1))
    ...     for i in range(2)
    ... )
    >>> oracle = BoundedChangeAdapter(ConstantChange(), ReplicateBounds(-1.0, -1.0))
    >>> result = validate_bounded_change(
    ...     oracle, (2.0,), (1.0,), plan=BoundedChangePlan((pairs,))
    ... )
    >>> result.decision
    'accepted'

    """
    before = finite_tuple(before, "before", minimum=None)
    after = finite_tuple(after, "after", minimum=None, length=len(before))
    threshold = finite_scalar(threshold, "acceptance threshold")
    if not isinstance(cast(object, plan), BoundedChangePlan):
        raise ContractError("provide a predeclared BoundedChangePlan")
    declared = tuple(oracle.change_bounds(before, after, pairs[0][0].count) for pairs in plan.looks)
    if not all(isinstance(cast(object, bounds), ReplicateBounds) for bounds in declared):
        raise ContractError("the oracle must certify ReplicateBounds before sampling")
    checkpoints: list[BoundedChangeCheckpoint] = []
    sampled = 0
    decision: Literal["accepted", "rejected", "ambiguous"] = "ambiguous"
    for pairs, bounds in zip(plan.looks, declared, strict=True):
        changes = tuple(oracle.change_replicate(before, after, a, b) for a, b in pairs)
        interval = empirical_bernstein_interval(changes, bounds=bounds, alpha=plan.alpha_per_look)
        sampled += sum(batch.count for pair in pairs for batch in pair)
        if interval.upper < threshold:
            decision = "accepted"
        elif interval.lower >= threshold:
            decision = "rejected"
        else:
            decision = "ambiguous"
        checkpoints.append(BoundedChangeCheckpoint(interval, changes, pairs, decision))
        if decision != "ambiguous":
            break
    return BoundedChangeResult(
        decision,
        tuple(checkpoints),
        threshold,
        plan.alpha_per_look,
        plan.reserved_histories,
        sampled,
    )
