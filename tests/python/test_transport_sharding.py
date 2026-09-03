"""Original-address lineage and pooled moments against independent host oracles."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, cast

import pytest

from dpt.contracts import NumericalError
from dpt.transport.model import IncompleteHistoryError, TransportError
from dpt.transport.rng import HistoryBatch, HistoryShard, SourceIdentity
from dpt.transport.sharding import ShardMoments, pool_shard_moments


def _oracle(values: tuple[float, ...]) -> tuple[float, float]:
    with localcontext() as context:
        context.prec = 1200
        exact = tuple(Decimal.from_float(value) for value in values)
        mean = sum(exact, Decimal(0)) / len(values)
        m2 = sum(((value - mean) ** 2 for value in exact), Decimal(0))
        return float(mean), float(m2 / (len(values) * (len(values) - 1)))


def _record(
    shard: HistoryShard,
    mean: tuple[float, ...] = (0.0,),
    m2: tuple[float, ...] = (0.0,),
    **kwargs: Any,
) -> ShardMoments:
    return ShardMoments(
        shard,
        mean,
        "canonical-model-density-source-layout-v1",
        status_counts=(shard.batch.count, 0, 0, 0, 0, 0, 0),
        centred_sum_squares=m2,
        **kwargs,
    )


def test_sharding_imports_without_optional_runtimes() -> None:
    source = Path(__file__).parents[2] / "python"
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(source)!r}); "
        "from dpt.transport.sharding import ShardMoments, pool_shard_moments; "
        "assert not {'warp', 'numpy', 'torch'}.intersection(sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True)


@pytest.mark.parametrize("count,parts", [(1, 1), (7, 3), (11, 7), (37, 2), (37, 37)])
def test_shards_balance_counts_and_preserve_every_global_address(count: int, parts: int) -> None:
    original = HistoryBatch(2**63 + 7, 2**48 + 91, count)
    shards = tuple(original.shard(parts, rank) for rank in range(parts))
    counts = tuple(shard.batch.count for shard in shards)
    assert max(counts) - min(counts) <= 1
    assert sum(counts) == count
    addresses = tuple(
        address
        for shard in shards
        for address in range(
            shard.batch.first_history, shard.batch.first_history + shard.batch.count
        )
    )
    assert addresses == tuple(range(original.first_history, original.first_history + count))
    for rank, shard in enumerate(shards):
        assert shard.original_batch == original
        assert shard.original_count == count
        assert shard.parent_batch == original
        assert shard.batch.seed == original.seed
        assert (shard.parts, shard.rank) == (parts, rank)
        assert shard.batch.first_history == original.first_history + shard.offset


def test_terminal_uint64_and_maximum_signed_count_partition_without_wrapping() -> None:
    original = HistoryBatch(2**64 - 1, 2**64 - 7, 7)
    assert tuple(original.shard(3, rank).batch.first_history for rank in range(3)) == (
        2**64 - 7,
        2**64 - 4,
        2**64 - 2,
    )
    assert original.shard(7, 6).batch == HistoryBatch(2**64 - 1, 2**64 - 1, 1)
    maximum = HistoryBatch(0, 0, 2**31 - 1)
    shards = tuple(maximum.shard(3, rank) for rank in range(3))
    assert sum(shard.batch.count for shard in shards) == 2**31 - 1


@pytest.mark.parametrize(
    "parts,rank",
    [(0, 0), (-1, 0), (8, 0), (2, -1), (2, 2), (True, 0), (2, False), (2.0, 0), (2, "1")],
)
def test_invalid_partition_arguments_are_rejected(parts: Any, rank: Any) -> None:
    with pytest.raises(TransportError, match="shard"):
        HistoryBatch(5, 0, 7).shard(parts, rank)


def test_partition_accepts_exact_integer_protocol() -> None:
    class Integral:
        def __index__(self) -> int:
            return 2

    shard = HistoryBatch(5, 0, 7).shard(Integral(), 1)
    assert (shard.batch.count, shard.offset) == (3, 4)


def test_nested_lineage_retains_original_count_and_immutable_metadata() -> None:
    original = HistoryBatch(71, 100, 11, "supplied-source")
    parent = original.shard(3, 1)
    nested = parent.shard(3, 2)
    assert nested.batch == HistoryBatch(71, 107, 1, "supplied-source")
    assert nested.original_batch == original
    assert nested.original_count == 11
    assert nested.parent_batch == parent.batch
    assert nested.offset == 7
    assert nested.lineage == ((3, 1), (3, 2))
    with pytest.raises(FrozenInstanceError):
        cast(Any, nested).offset = 0
    supplied: Any = [[3, 1], [3, 2]]
    snapshot = replace(nested, lineage=supplied)
    supplied[0][1] = 0
    assert snapshot.lineage == nested.lineage


@pytest.mark.parametrize(
    "changes",
    [
        {"offset": 0},
        {"lineage": ((3, 0),)},
        {"lineage": ()},
        {"lineage": ((3,),)},
        {"lineage": ((True, 0),)},
        {"batch": HistoryBatch(71, 0, 2)},
        {"original_batch": HistoryBatch(72, 100, 11)},
    ],
)
def test_forged_shard_lineage_is_rejected(changes: dict[str, Any]) -> None:
    shard = HistoryBatch(71, 100, 11).shard(3, 1)
    with pytest.raises(TransportError):
        replace(shard, **changes)


def test_sharding_slices_structural_source_tokens_without_renaming_namespace() -> None:
    original = HistoryBatch(51, 2**48, 7, "canonical-source").with_library_source(
        "parallel-rectangle-v1"
    )
    left, right = original.shard(2, 0), original.shard(2, 1)
    assert left.batch.source_namespace == right.batch.source_namespace == "canonical-source"
    assert right.batch.source_identity == SourceIdentity(51, 2**48 + 4, 3, "parallel-rectangle-v1")
    assert left.batch.independent_of(right.batch)
    overlapping = right.batch.with_library_source("different-sampler-name")
    assert not right.batch.independent_of(overlapping)


def test_library_sampler_attachment_updates_full_origin_and_nested_subset_together() -> None:
    original = HistoryBatch(7, 100, 11, "unchanged-label")
    nested = original.shard(3, 1).shard(3, 2).with_library_source("parallel-rectangle-v1")
    assert nested.original_batch == original.with_library_source("parallel-rectangle-v1")
    assert nested.batch.source_identity == SourceIdentity(7, 107, 1, "parallel-rectangle-v1")
    assert nested.parent_batch.source_identity == SourceIdentity(7, 104, 4, "parallel-rectangle-v1")
    assert nested.offset == 7
    with pytest.raises(TransportError, match="lineage"):
        replace(nested, batch=nested.batch.with_library_source("other-sampler"))


def test_generic_source_shards_retain_conservative_independence_and_require_explicit_law() -> None:
    original = HistoryBatch(17, 0, 4, "shared-random-source")
    left, right = original.shard(2, 0), original.shard(2, 1)
    assert not left.batch.independent_of(right.batch)
    with pytest.raises(TransportError, match="explicit IID source provenance"):
        _record(left)
    records = tuple(
        _record(shard, iid_source_provenance="canonical independent source draw per original ID")
        for shard in (left, right)
    )
    result = pool_shard_moments(records)
    assert result.batch == original
    assert not left.batch.independent_of(right.batch)
    assert result.iid_source_provenance == records[0].iid_source_provenance


def test_unequal_shard_pooling_matches_direct_original_history_decimal_oracle() -> None:
    values = (0.0, 4.0, 0.0, -2.0, 3.0, 0.0, 5.0)
    original = HistoryBatch(71, 100, len(values))
    records: list[ShardMoments] = []
    for rank in range(3):
        shard = original.shard(3, rank)
        subset = values[shard.offset : shard.offset + shard.batch.count]
        mean, variance = _oracle(subset)
        records.append(
            ShardMoments(
                shard,
                (mean,),
                "analytic original-history array",
                estimated_variance_of_mean=(variance,),
                status_counts=(len(subset), 0, 0, 0, 0, 0, 0),
            )
        )
    pooled = pool_shard_moments(tuple(reversed(records)))
    expected_mean, expected_variance = _oracle(values)
    assert pooled.mean == pytest.approx((expected_mean,), rel=1e-15)
    assert pooled.estimated_variance_of_mean == pytest.approx((expected_variance,), rel=1e-15)
    assert pooled.mean[0] != pytest.approx(math.fsum(record.mean[0] for record in records) / 3)
    without_between = math.fsum(
        record.shard.batch.count * (record.shard.batch.count - 1) * record.variance_of_mean[0]
        for record in records
        if record.variance_of_mean is not None
    ) / (len(values) * (len(values) - 1))
    assert pooled.estimated_variance_of_mean[0] > without_between
    assert pooled.status_counts == (7, 0, 0, 0, 0, 0, 0)


def test_nested_leaves_and_singletons_pool_within_and_between_moments() -> None:
    original = HistoryBatch(2, 100, 5)
    parent = original.shard(2, 0)
    leaves = (parent.shard(2, 0), parent.shard(2, 1), original.shard(2, 1))
    records = (
        _record(leaves[0], (1.0,), (2.0,)),
        _record(leaves[1], (5.0,), (0.0,)),
        _record(leaves[2], (2.0,), (18.0,)),
    )
    assert records[1].variance_of_mean is None
    pooled = pool_shard_moments(records)
    mean, variance = _oracle((0.0, 2.0, 5.0, -1.0, 5.0))
    assert pooled.mean == pytest.approx((mean,))
    assert pooled.estimated_variance_of_mean == pytest.approx((variance,))


def test_all_singleton_shards_have_no_own_variance_but_have_pooled_variance() -> None:
    original = HistoryBatch(2, 0, 2)
    records = tuple(
        _record(original.shard(2, rank), (value,)) for rank, value in enumerate((1.0, 3.0))
    )
    assert all(record.variance_of_mean is None for record in records)
    pooled = pool_shard_moments(records)
    assert pooled.mean == (2.0,)
    assert pooled.estimated_variance_of_mean == (1.0,)


@pytest.mark.parametrize("values", [(math.inf,), (math.nan,), (True,), ("0",), (10**400,)])
def test_checkpoint_values_are_finite_real_scalars(values: Any) -> None:
    with pytest.raises(TransportError):
        _record(HistoryBatch(2, 0, 4).shard(2, 0), values)


@pytest.mark.parametrize(
    "changes",
    [
        {"centred_sum_squares": None},
        {"estimated_variance_of_mean": (0.0,)},
        {"centred_sum_squares": (-1.0,)},
        {"centred_sum_squares": (0.0, 0.0)},
        {"configuration_identity": ""},
        {"sampling_law": "stratified"},
        {"status_counts": (1, 0, 0, 0, 0, 0, 0)},
        {"status_counts": (True, 1, 0, 0, 0, 0, 0)},
        {"status_counts": (2,)},
    ],
)
def test_invalid_moment_contract_is_rejected(changes: dict[str, Any]) -> None:
    record = _record(HistoryBatch(2, 0, 4).shard(2, 0))
    with pytest.raises(TransportError):
        replace(record, **changes)


def test_failed_shard_cannot_enter_pool_and_singleton_cannot_claim_zero_variance() -> None:
    record = _record(HistoryBatch(2, 0, 4).shard(2, 0))
    with pytest.raises(IncompleteHistoryError, match="failed shard"):
        replace(record, status_counts=(1, 0, 1, 0, 0, 0, 0))
    single = _record(HistoryBatch(2, 0, 2).shard(2, 0))
    with pytest.raises(TransportError, match="singleton"):
        replace(single, centred_sum_squares=None, estimated_variance_of_mean=(0.0,))
    with pytest.raises(TransportError, match="singleton"):
        replace(single, centred_sum_squares=(1.0,))
    with pytest.raises(TransportError, match="two original histories"):
        pool_shard_moments((_record(HistoryBatch(2, 0, 1).shard(1, 0)),))


def test_host_snapshots_copy_mutable_values_and_do_not_alias_workspace_arrays() -> None:
    mean: Any = [1.0]
    m2: Any = [2.0]
    statuses: Any = [2, 0, 0, 0, 0, 0, 0]
    record = ShardMoments(
        HistoryBatch(2, 0, 4).shard(2, 0),
        mean,
        "host snapshot",
        centred_sum_squares=m2,
        status_counts=statuses,
    )
    mean[0], m2[0], statuses[0] = 100.0, 200.0, 100
    assert (record.mean, record.centred_sum_squares, record.status_counts[0]) == ((1.0,), (2.0,), 2)
    with pytest.raises(FrozenInstanceError):
        cast(Any, record).mean = (10.0,)


def test_pool_rejects_duplicate_missing_overlapping_and_mixed_model_shards() -> None:
    original = HistoryBatch(2, 0, 7)
    records = tuple(_record(original.shard(3, rank)) for rank in range(3))
    for invalid in ((), records[:2], (*records, records[0]), (records[0], records[2])):
        with pytest.raises(TransportError):
            pool_shard_moments(invalid)
    overlap = (_record(original.shard(2, 0)), *records[1:])
    with pytest.raises(TransportError, match="without gaps"):
        pool_shard_moments(overlap)
    for change in (
        {"configuration_identity": "other-density-snapshot"},
        {"iid_source_provenance": "other source law"},
        {"mean": (0.0, 0.0), "centred_sum_squares": (0.0, 0.0)},
        {"shard": HistoryBatch(3, 0, 7).shard(3, 2)},
    ):
        with pytest.raises(TransportError):
            pool_shard_moments((*records[:2], replace(records[2], **change)))


def test_equal_extreme_means_and_normalised_variances_do_not_overflow_raw_m2() -> None:
    original = HistoryBatch(2, 0, 4)
    records = tuple(
        ShardMoments(
            original.shard(2, rank),
            (1e308, 0.0, 0.0),
            "extreme binary64 checkpoint",
            estimated_variance_of_mean=(0.0, 1e308, 1e-300),
            status_counts=(2, 0, 0, 0, 0, 0, 0),
        )
        for rank in range(2)
    )
    result = pool_shard_moments(records)
    assert result.mean == (1e308, 0.0, 0.0)
    assert result.estimated_variance_of_mean == pytest.approx(
        (0.0, 1e308 / 3, 1e-300 / 3), rel=2e-15, abs=0
    )


def test_between_shard_square_overflow_is_rescued_when_final_variance_fits() -> None:
    original = HistoryBatch(2, 0, 2)
    records = tuple(
        _record(original.shard(2, rank), (mean,)) for rank, mean in enumerate((-1e154, 1e154))
    )
    result = pool_shard_moments(records)
    expected_mean, expected_variance = _oracle((-1e154, 1e154))
    assert result.mean == (expected_mean,)
    assert result.estimated_variance_of_mean == (expected_variance,)


def test_unrepresentable_variance_raises_even_if_extreme_mean_cancels() -> None:
    original = HistoryBatch(2, 0, 2)
    records = tuple(
        _record(original.shard(2, rank), (mean,)) for rank, mean in enumerate((-1e308, 1e308))
    )
    with pytest.raises(NumericalError, match="variance of mean exceeds binary64 range"):
        pool_shard_moments(records)


def test_cancelling_large_means_preserve_representable_tiny_residual() -> None:
    values = (1e154, -1e154, 1e-300)
    original = HistoryBatch(2, 0, len(values))
    records = tuple(_record(original.shard(3, rank), (mean,)) for rank, mean in enumerate(values))
    result = pool_shard_moments(records)
    expected_mean, expected_variance = _oracle(values)
    assert result.mean == (expected_mean,)
    assert result.mean[0] > 0
    assert result.estimated_variance_of_mean == (expected_variance,)


def test_final_subnormal_mean_and_variance_rounding_match_binary64() -> None:
    smallest = math.ulp(0.0)
    original = HistoryBatch(2, 0, 3)
    records = tuple(
        _record(original.shard(3, rank), (value,))
        for rank, value in enumerate((smallest, 0.0, 0.0))
    )
    result = pool_shard_moments(records)
    # Both exact results are positive and smaller than half the minimum
    # binary64 subnormal. Final rounding, rather than intermediate scaling,
    # therefore gives zero.
    assert result.mean == (0.0,)
    assert result.estimated_variance_of_mean == (0.0,)


def test_pooled_marginals_preserve_zeros_and_do_not_sum_pixel_covariance_away() -> None:
    original = HistoryBatch(2, 0, 4)
    records = (
        _record(original.shard(2, 0), (1.0, 0.0), (0.0, 0.0)),
        _record(original.shard(2, 1), (0.0, 1.0), (0.0, 0.0)),
    )
    result = pool_shard_moments(records)
    assert result.mean == (0.5, 0.5)
    assert result.estimated_variance_of_mean == pytest.approx((1 / 12, 1 / 12))
    # Every whole history has summed score one. Its variance is exactly zero,
    # while the sum of these marginal estimates is positive.
    assert sum(result.estimated_variance_of_mean) > 0
