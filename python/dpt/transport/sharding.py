"""Explicit host checkpoints for complete IID original-history partitions.

Transfer each shard's images deliberately before constructing these records.
The tuples are immutable snapshots, independent of reusable device workspaces.
Pooling combines marginal empirical moments, including zero contributions and
between-shard variation; it does not launch work or claim multi-GPU execution.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

from dpt.contracts import ContractError, NumericalError, finite_scalar

# Shared checked transport helpers do not import a device runtime.
# pyright: reportPrivateUsage=false
from .model import HistoryStatus, IncompleteHistoryError, TransportError, _require_instance
from .rng import HistoryBatch, HistoryShard


def _snapshot(
    values: Sequence[float], name: str, *, nonnegative: bool = False
) -> tuple[float, ...]:
    try:
        result = tuple(
            finite_scalar(value, name, minimum=0.0 if nonnegative else None) for value in values
        )
    except (ContractError, TypeError, OverflowError) as error:
        raise TransportError(f"{name} requires finite real checkpoint values") from error
    if not result:
        raise TransportError(f"{name} requires a nonempty checkpoint")
    return result


def _token(value: str, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise TransportError(f"{name} requires a nonempty explicit provenance token")


def _representable(value: Fraction, name: str) -> float:
    try:
        result = float(value)
    except OverflowError as error:
        raise NumericalError(f"pooled {name} exceeds binary64 range") from error
    if not math.isfinite(result):
        raise NumericalError(f"pooled {name} exceeds binary64 range")
    return result


@dataclass(frozen=True, slots=True)
class ShardMoments:
    """One completed shard's marginal moments after explicit host transfer.

    ``mean`` divides by ``shard.batch.count``, including misses and absorption.
    Supply exactly one of ``centred_sum_squares`` (raw M2) or
    ``estimated_variance_of_mean`` (M2 / (n * (n - 1))). A singleton requires
    raw M2 equal to zero; its own variance estimate remains unavailable.

    ``configuration_identity`` declares one canonical physical model, source,
    density snapshot, scoring convention and image/component layout. The IID
    sampling law applies to original histories, rather than descendants or
    strata. For generic stochastic sources without a structural source token,
    ``iid_source_provenance`` must declare the canonical source snapshot whose
    independent draws are indexed by those original histories. These are caller
    declarations; matching labels do not verify arbitrary buffers' contents or
    manufacture source independence.

    Seven ``status_counts`` must account for every original history and contain
    only escape/absorption completions. Failed estimates cannot become records.
    """

    shard: HistoryShard
    mean: tuple[float, ...]
    configuration_identity: str
    status_counts: tuple[int, ...] = field(kw_only=True)
    centred_sum_squares: tuple[float, ...] | None = field(default=None, kw_only=True)
    estimated_variance_of_mean: tuple[float, ...] | None = field(default=None, kw_only=True)
    sampling_law: Literal["iid_original_histories"] = field(
        default="iid_original_histories", kw_only=True
    )
    iid_source_provenance: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _require_instance(self.shard, HistoryShard, "shard moments identity")
        if self.shard.original_batch.sampling is not None:
            raise TransportError("dependent path designs require complete-batch replicate variance")
        _token(self.configuration_identity, "configuration_identity")
        if self.sampling_law != "iid_original_histories":
            raise TransportError("shard moments require the IID original-history sampling law")
        if self.iid_source_provenance is not None:
            _token(self.iid_source_provenance, "iid_source_provenance")
        original = self.shard.original_batch
        if (
            original.source_identity is None
            and original.source_namespace != "deterministic"
            and self.iid_source_provenance is None
        ):
            raise TransportError(
                "generic stochastic sources require explicit IID source provenance"
            )
        mean = _snapshot(self.mean, "shard mean")
        object.__setattr__(self, "mean", mean)
        if (self.centred_sum_squares is None) == (self.estimated_variance_of_mean is None):
            raise TransportError("supply exactly one shard moment representation")
        for name in ("centred_sum_squares", "estimated_variance_of_mean"):
            values = getattr(self, name)
            if values is not None:
                snapshot = _snapshot(values, name, nonnegative=True)
                if len(snapshot) != len(mean):
                    raise TransportError("shard means and moments must have the same layout")
                object.__setattr__(self, name, snapshot)
        if self.shard.batch.count == 1 and (
            self.centred_sum_squares is None or any(self.centred_sum_squares)
        ):
            raise TransportError("a singleton requires raw centred sum zero, without a variance")
        try:
            statuses = tuple(self.status_counts)
        except TypeError as error:
            raise TransportError("shard status counts require seven integer counts") from error
        if len(statuses) != len(HistoryStatus) or any(
            type(count) is not int or count < 0 for count in statuses
        ):
            raise TransportError("shard status counts require seven nonnegative integer counts")
        if sum(statuses) != self.shard.batch.count:
            raise TransportError("shard status counts must account for every original history")
        if any(statuses[2:]):
            raise IncompleteHistoryError("failed shard histories cannot be pooled")
        object.__setattr__(self, "status_counts", statuses)

    @property
    def variance_of_mean(self) -> tuple[float, ...] | None:
        """Marginal sample variance / n, unavailable for a singleton."""
        count = self.shard.batch.count
        if count == 1:
            return None
        if self.estimated_variance_of_mean is not None:
            return self.estimated_variance_of_mean
        assert self.centred_sum_squares is not None
        denominator = count * (count - 1)
        return tuple(
            _representable(Fraction.from_float(value) / denominator, "shard variance")
            for value in self.centred_sum_squares
        )


@dataclass(frozen=True, slots=True)
class PooledHistoryMoments:
    """Marginal mean and sample variance / N for one complete original batch.

    Pixels/components may be correlated; these marginal variances do not give
    the variance of a sum across pixels or a nonlinear objective.
    """

    batch: HistoryBatch
    mean: tuple[float, ...]
    estimated_variance_of_mean: tuple[float, ...]
    status_counts: tuple[int, ...]
    configuration_identity: str
    sampling_law: Literal["iid_original_histories"]
    iid_source_provenance: str | None


def pool_shard_moments(shards: Sequence[ShardMoments]) -> PooledHistoryMoments:
    """Mean=sum_s n_s*mean_s/N and Var_mean=sum_s (M2_s+n_s*(mean_s-Mean)^2)/(N*(N-1)).

    Shards is a sequence of complete ShardMoments snapshots whose disjoint leaves
    cover one original HistoryBatch exactly once. Each mean/M2 or variance tuple
    has length P in the same declared component layout, and seven terminal-count
    entries include every original history with no failed status. Model, source
    and IID sampling-law declarations must agree. PooledHistoryMoments returns
    binary64 tuples of length P, in score and squared-score units, plus immutable
    identity/count metadata. Nested/unequal and singleton shards are valid when
    N >= 2 and singleton M2 is zero. Exact binary rational pooling precedes final
    rounding, retaining within-shard and between-shard variation. These are
    marginal variances and no parameter derivative or transfer is performed.

    >>> from dpt.transport.rng import HistoryBatch
    >>> from dpt.transport.sharding import ShardMoments, pool_shard_moments
    >>> batch = HistoryBatch(41, 0, 4)
    >>> shards = tuple(
    ...     ShardMoments(
    ...         batch.shard(2, rank),
    ...         (float(rank + 1),),
    ...         "slab-v1",
    ...         status_counts=(2, 0, 0, 0, 0, 0, 0),
    ...         centred_sum_squares=(0.0,),
    ...     )
    ...     for rank in range(2)
    ... )
    >>> result = pool_shard_moments(shards)
    >>> (result.mean, round(result.estimated_variance_of_mean[0], 6))
    ((1.5,), 0.083333)

    """
    try:
        records = tuple(shards)
    except TypeError as error:
        raise TransportError("pooling requires completed shard records") from error
    if not records:
        raise TransportError("pooling requires at least one completed shard")
    for record in records:
        _require_instance(record, ShardMoments, "pooling shard")
    ordered = sorted(records, key=lambda record: record.shard.offset)
    first = ordered[0]
    original = first.shard.original_batch
    count = original.count
    if count < 2:
        raise TransportError("pooled variance requires at least two original histories")
    expected_offset = 0
    for record in ordered:
        if record.shard.original_batch != original:
            raise TransportError("pooled shards must retain the same original history identity")
        if (
            record.configuration_identity != first.configuration_identity
            or record.sampling_law != first.sampling_law
            or record.iid_source_provenance != first.iid_source_provenance
            or len(record.mean) != len(first.mean)
        ):
            raise TransportError("pooled shards must share model/layout and IID source provenance")
        if record.shard.offset != expected_offset:
            raise TransportError("pooled shards must cover the original range once without gaps")
        expected_offset += record.shard.batch.count
    if expected_offset != count:
        raise TransportError("pooled shards must cover every original history")
    means: list[float] = []
    variances: list[float] = []
    denominator = count * (count - 1)
    for component in range(len(first.mean)):
        exact_means = tuple(Fraction.from_float(record.mean[component]) for record in ordered)
        pooled_mean = (
            sum(
                (
                    record.shard.batch.count * mean
                    for record, mean in zip(ordered, exact_means, strict=True)
                ),
                Fraction(0),
            )
            / count
        )
        pooled_m2 = Fraction(0)
        for record, mean in zip(ordered, exact_means, strict=True):
            size = record.shard.batch.count
            if record.centred_sum_squares is not None:
                within = Fraction.from_float(record.centred_sum_squares[component])
            else:
                assert record.estimated_variance_of_mean is not None
                within = (
                    Fraction.from_float(record.estimated_variance_of_mean[component])
                    * size
                    * (size - 1)
                )
            pooled_m2 += within + size * (mean - pooled_mean) ** 2
        means.append(_representable(pooled_mean, "mean"))
        variances.append(_representable(pooled_m2 / denominator, "variance of mean"))
    statuses = tuple(sum(record.status_counts[index] for record in ordered) for index in range(7))
    if sum(statuses) != count:
        raise TransportError("pooled status counts must account for every original history")
    if any(statuses[2:]):
        raise IncompleteHistoryError("failed shard histories cannot be pooled")
    return PooledHistoryMoments(
        original,
        tuple(means),
        tuple(variances),
        statuses,
        first.configuration_identity,
        first.sampling_law,
        first.iid_source_provenance,
    )
