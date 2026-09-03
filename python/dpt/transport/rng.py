"""History identity and independence declarations; no GPU import or RNG execution.

Philox4x32-10 uses (history low, history high, event, 0) as its counter and the
64-bit run seed as its key. Draw lanes 0,1,2,3 are respectively free-flight optical
depth, interaction type, polar cosine and azimuth. Crossing a material face does
not advance the event or discard residual optical depth. Integer counter
identity is independent of launch ordering, chunking, and derivative replay.
"""

from __future__ import annotations

# Runtime dataclass validation shares the transport model's checked helper.
# pyright: reportPrivateUsage=false
import operator
from dataclasses import dataclass, replace
from typing import Any, SupportsIndex, cast

from .model import TransportError, _require_instance
from .sampling_identity import SamplingIdentity


def _identity_integer(
    value: SupportsIndex, name: str, maximum: int, *, positive: bool = False
) -> int:
    if isinstance(value, bool):
        raise TransportError(f"{name} must be an integer history identity")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TransportError(f"{name} must be an integer history identity") from error
    if not int(positive) <= result < maximum:
        raise TransportError(f"{name} is outside its declared history identity range")
    return result


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Structural token for a library sampler's reserved source coordinates.

    Different sampler names do not make overlapping counter words independent.
    The token describes the actual draw addresses and complete path design,
    rather than a caller's label.
    """

    seed: int
    first_history: int
    count: int
    sampler: str
    sampling: SamplingIdentity | None = None

    def __post_init__(self) -> None:
        for name, maximum, positive in (
            ("seed", 2**64, False),
            ("first_history", 2**64, False),
            ("count", 2**31, True),
        ):
            object.__setattr__(
                self, name, _identity_integer(getattr(self, name), name, maximum, positive=positive)
            )
        if self.first_history + self.count > 2**64:
            raise TransportError("the source history range would wrap its 64-bit identity")
        if type(self.sampler) is not str or not self.sampler.strip():
            raise TransportError("source identity requires a sampler kind")
        _check_sampling(self.sampling, self.seed, self.first_history, self.count)

    def independent_of(self, other: SourceIdentity) -> bool:
        if self.sampling is not None or other.sampling is not None:
            return self.seed != other.seed
        return self.seed != other.seed or (
            self.first_history + self.count <= other.first_history
            or other.first_history + other.count <= self.first_history
        )


@dataclass(frozen=True, slots=True, init=False)
class HistoryBatch:
    """A disjoint range of original histories, with an explicit source-draw namespace.

    `source_namespace` must identify the caller's source random stream as well as
    any random importance weights. Disjoint transport histories do not make a
    reused random source draw independent. Use distinct namespaces for independent
    source batches, or the literal ``deterministic`` for a deterministic source.
    An attached ``SamplingIdentity`` records one complete randomised path design.
    Its disjoint shards share the same randomisation and remain dependent.
    """

    seed: int
    first_history: int
    count: int
    source_namespace: str = "deterministic"
    source_identity: SourceIdentity | None = None
    sampling: SamplingIdentity | None = None

    def __init__(
        self,
        seed: SupportsIndex,
        first_history: SupportsIndex,
        count: SupportsIndex,
        source_namespace: str = "deterministic",
        source_identity: SourceIdentity | None = None,
        sampling: SamplingIdentity | None = None,
    ) -> None:
        object.__setattr__(self, "seed", _identity_integer(seed, "seed", 2**64))
        object.__setattr__(
            self, "first_history", _identity_integer(first_history, "first_history", 2**64)
        )
        object.__setattr__(self, "count", _identity_integer(count, "count", 2**31, positive=True))
        object.__setattr__(self, "source_namespace", source_namespace)
        object.__setattr__(self, "source_identity", source_identity)
        object.__setattr__(self, "sampling", sampling)
        self.__post_init__()

    def __post_init__(self) -> None:
        for name, value in (("seed", self.seed), ("first_history", self.first_history)):
            object.__setattr__(self, name, _identity_integer(value, name, 2**64))
        object.__setattr__(
            self, "count", _identity_integer(self.count, "count", 2**31, positive=True)
        )
        if self.first_history + self.count > 2**64:
            raise TransportError("the history range would wrap its 64-bit identity")
        _check_sampling(self.sampling, self.seed, self.first_history, self.count)
        if type(self.source_namespace) is not str or not self.source_namespace.strip():
            raise TransportError("declare the source random-stream namespace")
        token = self.source_identity
        if token is not None:
            _require_instance(token, SourceIdentity, "source_identity")
        if token is not None and (token.seed, token.first_history, token.count) != (
            self.seed,
            self.first_history,
            self.count,
        ):
            raise TransportError(
                "structural source identity must match the transport history range"
            )
        if token is not None and token.sampling != self.sampling:
            raise TransportError("source and transport sampling identities must match")

    def with_library_source(self, sampler: str) -> HistoryBatch:
        """Attach the known reserved source counters for an actual library sampler."""
        return replace(
            self,
            source_identity=SourceIdentity(
                self.seed, self.first_history, self.count, sampler, self.sampling
            ),
        )

    def with_sampling(self, identity: SamplingIdentity) -> HistoryBatch:
        """Retain one complete design identity through source sampling and sharding."""
        source = self.source_identity
        if source is not None:
            source = replace(source, sampling=identity)
        return replace(self, sampling=identity, source_identity=source)

    def shard(self, parts: SupportsIndex, rank: SupportsIndex) -> HistoryShard:
        """Partition original history addresses into balanced nonempty shards.

        Pass the returned ``.batch`` to transport calls. ``.original_count``
        remains the count of this complete batch, including undetected histories.
        Source namespaces and sampler counter addresses are preserved unchanged.
        """
        return _make_shard(self, ((parts, rank),))

    def independent_of(self, other: HistoryBatch) -> bool:
        """Check declared stream separation, not empirical statistical independence."""
        transport_separate = self.seed != other.seed or (
            self.first_history + self.count <= other.first_history
            or other.first_history + other.count <= self.first_history
        )
        if self.sampling is not None or other.sampling is not None:
            transport_separate = self.seed != other.seed
        source_separate = (
            self.source_namespace == other.source_namespace == "deterministic"
            or self.source_namespace != other.source_namespace
        )
        if self.source_identity is not None and other.source_identity is not None:
            source_separate = self.source_identity.independent_of(other.source_identity)
        return transport_separate and source_separate


def _check_sampling(
    sampling: SamplingIdentity | None, seed: int, first_history: int, count: int
) -> None:
    if sampling is not None:
        _require_instance(sampling, SamplingIdentity, "sampling")
        if (
            sampling.seed != seed
            or first_history < sampling.first_history
            or (first_history + count > sampling.first_history + sampling.count)
        ):
            raise TransportError("history range must belong to its complete sampling identity")


def _partition_batch(
    batch: HistoryBatch, parts: SupportsIndex, rank: SupportsIndex
) -> tuple[HistoryBatch, int, tuple[int, int]]:
    parts = _identity_integer(parts, "shard parts", batch.count + 1, positive=True)
    rank = _identity_integer(rank, "shard rank", parts)
    quotient, remainder = divmod(batch.count, parts)
    count = quotient + int(rank < remainder)
    offset = rank * quotient + min(rank, remainder)
    first = batch.first_history + offset
    source = batch.source_identity
    if source is not None:
        source = replace(source, first_history=first, count=count)
    child = replace(batch, first_history=first, count=count, source_identity=source)
    return child, offset, (parts, rank)


@dataclass(frozen=True, slots=True)
class HistoryShard:
    """One immutable subset with its complete original-history lineage.

    ``offset`` is relative to ``original_batch``; it never resets at a device or
    nested partition. ``lineage`` records each balanced ``(parts, rank)`` split,
    and ``batch`` retains the actual Philox addresses. A shard is not a new
    independent replicate. Generic source namespaces retain their existing
    conservative independence declarations.
    """

    original_batch: HistoryBatch
    batch: HistoryBatch
    offset: int
    lineage: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        _require_instance(self.original_batch, HistoryBatch, "shard original_batch")
        _require_instance(self.batch, HistoryBatch, "shard batch")
        offset = _identity_integer(self.offset, "shard offset", self.original_batch.count)
        try:
            partitions = cast(tuple[Any, ...], tuple(self.lineage))
        except TypeError as error:
            raise TransportError("shard lineage requires balanced partition pairs") from error
        if not partitions:
            raise TransportError("shard lineage requires at least one partition")
        expected = self.original_batch
        expected_offset = 0
        checked: list[tuple[int, int]] = []
        for partition in partitions:
            if not isinstance(partition, (list, tuple)):
                raise TransportError("shard lineage requires balanced partition pairs")
            supplied = cast(tuple[SupportsIndex, ...], tuple(cast(Any, partition)))
            if len(supplied) != 2:
                raise TransportError("shard lineage requires balanced partition pairs")
            expected, step, pair = _partition_batch(expected, supplied[0], supplied[1])
            expected_offset += step
            checked.append(pair)
        if expected != self.batch or expected_offset != offset:
            raise TransportError("shard batch and offset must match their original lineage")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "lineage", tuple(checked))

    @property
    def original_count(self) -> int:
        """Number of launched histories in the complete original batch."""
        return self.original_batch.count

    @property
    def parts(self) -> int:
        """Number of partitions at the final lineage step."""
        return self.lineage[-1][0]

    @property
    def rank(self) -> int:
        """Rank within the final lineage step."""
        return self.lineage[-1][1]

    @property
    def parent_batch(self) -> HistoryBatch:
        """History range immediately before the final partition."""
        parent = self.original_batch
        for parts, rank in self.lineage[:-1]:
            parent, _, _ = _partition_batch(parent, parts, rank)
        return parent

    def shard(self, parts: SupportsIndex, rank: SupportsIndex) -> HistoryShard:
        """Partition this subset while retaining the original global count."""
        _, _, pair = _partition_batch(self.batch, parts, rank)
        return _make_shard(self.original_batch, (*self.lineage, pair))

    def with_library_source(self, sampler: str) -> HistoryShard:
        """Attach one sampler domain consistently to the root and child range."""
        return _make_shard(self.original_batch.with_library_source(sampler), self.lineage)


def _make_shard(
    original: HistoryBatch, lineage: tuple[tuple[SupportsIndex, SupportsIndex], ...]
) -> HistoryShard:
    child = original
    offset = 0
    checked: list[tuple[int, int]] = []
    for parts, rank in lineage:
        child, step, pair = _partition_batch(child, parts, rank)
        offset += step
        checked.append(pair)
    return HistoryShard(original, child, offset, tuple(checked))


def require_independent(*batches: HistoryBatch) -> None:
    """Reject overlapping random histories before combining nonlinear estimators."""
    for index, left in enumerate(batches):
        for right in batches[index + 1 :]:
            if not left.independent_of(right):
                raise TransportError(
                    "nonlinear estimator requires disjoint transport/source streams"
                )
