"""Independent full-path coordinate, digital-net and randomisation oracles."""

from __future__ import annotations

import itertools
from typing import Any, Literal, cast

import numpy as np
import pytest

from dpt.transport import HistoryBatch, SamplingDesign, SamplingIdentity, TransportError
from dpt.transport.estimators import history_moments
from dpt.transport.rng import require_independent
from dpt.transport.sampling import (
    _design_words,
    _philox_words,
    _sobol_directions,
    design_replicate_moments,
    pool_design_shard_means,
    sampling_uniforms,
)
from dpt.transport.sharding import ShardMoments


def test_joe_kuo_all_directions_match_independent_scipy_provider() -> None:
    from scipy.stats import qmc

    reference = qmc.Sobol(d=21201, scramble=False, bits=32)
    np.testing.assert_array_equal(_sobol_directions(21201), reference._sv)


def test_host_philox_random123_vector() -> None:
    assert tuple(int(v) for v in _philox_words(0, [0], 0, 0)[0]) == (
        0x6627E8D5,
        0xE169C58D,
        0xBC57AC4C,
        0x9B00DBD8,
    )


def test_every_bounded_counter_lane_has_one_unique_dimension() -> None:
    design = SamplingDesign("scrambled-sobol", 7, 9, 3)
    coordinates = [design.dimension(0x80000000, 0, lane) for lane in range(2)]
    coordinates.append(design.dimension(0x80000002, 0, 0))
    roles = [(0, 4), (1, 1), (2, 3), (3, 1), (0x40000000, 2), (0x40000001, 1)]
    roles.extend(
        (namespace + trial, 3) for namespace in (0x80000000, 0xC0000000) for trial in range(9)
    )
    for branch, event in itertools.product(range(3), range(7)):
        for domain, width in roles:
            coordinates.extend(
                design.dimension(event, domain + branch * 65536, lane) for lane in range(width)
            )
    assert sorted(coordinates) == list(range(design.dimensions))
    for event, domain, lane in [
        (7, 0, 0),
        (0, 3 * 65536, 0),
        (0, 0x80000009, 0),
        (0, 0xC0000009, 0),
        (0, 0x10000000, 0),
        (0, 1, 1),
    ]:
        with pytest.raises(TransportError):
            design.dimension(event, domain, lane)


def test_complete_design_identity_shards_and_independence() -> None:
    design = SamplingDesign("scrambled-sobol", 4, 8)
    identity = SamplingIdentity(design, 5, 2**63 + 19, 128)
    batch = HistoryBatch(identity.seed, identity.first_history, identity.count).with_sampling(
        identity
    )
    shards = [batch.shard(3, rank) for rank in range(3)]
    assert all(shard.batch.sampling == identity for shard in shards)
    assert not shards[0].batch.independent_of(shards[1].batch)
    with pytest.raises(TransportError, match="disjoint"):
        require_independent(shards[0].batch, shards[1].batch)
    whole = sampling_uniforms(identity, event=3, domain=0xC0000007)
    split = np.concatenate(
        [
            sampling_uniforms(
                identity,
                event=3,
                domain=0xC0000007,
                histories=np.arange(s.batch.count, dtype=np.uint64)
                + np.uint64(s.batch.first_history),
            )
            for s in shards
        ]
    )
    np.testing.assert_array_equal(whole, split)
    shifted = SamplingIdentity(design, 5, 2**63 + 4096, 128)
    # Sobol point indices are relative to the complete batch, not absolute history.
    np.testing.assert_array_equal(whole, sampling_uniforms(shifted, event=3, domain=0xC0000007))


@pytest.mark.parametrize("law", ["stratified", "scrambled-sobol"])
def test_all_role_marginals_are_open_uniform_and_dyadically_balanced(
    law: Literal["stratified", "scrambled-sobol"],
) -> None:
    identity = SamplingIdentity(SamplingDesign(law, 3, 5, 2), 721, 0, 256)
    for event, domain in [
        (0x80000000, 0),
        (0x80000002, 0),
        (0, 0),
        (2, 65538),
        (1, 0x40010000),
        (2, 0x80010004),
        (2, 0xC0010004),
    ]:
        sample = sampling_uniforms(identity, event=event, domain=domain)
        assert bool(np.all((sample > 0) & (sample < 1)))
        for lane in range(4):
            try:
                identity.design.dimension(event, domain, lane)
            except TransportError:
                np.testing.assert_array_equal(sample[:, lane], 0.5)
                continue
            np.testing.assert_array_equal(
                np.sort((sample[:, lane] * 256).astype(int)), np.arange(256)
            )
            assert abs(float(np.mean(sample[:, lane])) - 0.5) < 1 / 256
            assert abs(float(np.mean(sample[:, lane] ** 2)) - 1 / 3) < 2 / 256


def test_lms_scramble_is_invertible_binary_and_preserves_first_dimension_net() -> None:
    identity = SamplingIdentity(SamplingDesign("scrambled-sobol", 2, 3), 83, 0, 64)
    directions, shift = _design_words(identity)
    # First-coordinate base directions are the basis vectors in MSB-first order.
    # LMS must be triangular with unit diagonal, rather than an arbitrary hash.
    first = directions[0]
    for bit in range(32):
        assert int(first[bit]) & (1 << (31 - bit))
        assert int(first[bit]) >> (32 - bit) == 0
    point = sampling_uniforms(identity, event=0x80000000, domain=0)[0, 0]
    assert point == (int(shift[0]) + 0.5) / 2**32


def test_explicit_budget_and_invalid_addresses_fail() -> None:
    assert SamplingDesign("scrambled-sobol", 1177, 1).dimensions == 21189
    with pytest.raises(TransportError, match="dimension"):
        SamplingDesign("scrambled-sobol", 1178, 1)
    with pytest.raises(TransportError, match="dimension"):
        SamplingDesign("scrambled-sobol", 2, 4096)
    for count in (1, 3, 1023):
        with pytest.raises(TransportError):
            SamplingIdentity(SamplingDesign("stratified", 2, 4), 0, 0, count)
    identity = SamplingIdentity(SamplingDesign("stratified", 2, 4), 0, 0, 64)
    with pytest.raises(TransportError):
        sampling_uniforms(identity, event=2, domain=0)
    with pytest.raises(TransportError):
        sampling_uniforms(identity, event=0, domain=0, histories=np.array([64]))
    for event, domain in [(-1, 0), (0, -1), (True, 0), (0, 2**32)]:
        with pytest.raises(TransportError):
            sampling_uniforms(identity, event=event, domain=domain)


@pytest.mark.parametrize("law", ["stratified", "scrambled-sobol"])
def test_largest_dyadic_count_and_uint64_history_endpoints(
    law: Literal["stratified", "scrambled-sobol"],
) -> None:
    first, count = 2**64 - 2**30, 2**30
    identity = SamplingIdentity(SamplingDesign(law, 2, 4), 2**64 - 1, first, count)
    history = np.asarray([first, 2**64 - 1], dtype=np.uint64)
    sample = sampling_uniforms(identity, event=1, domain=0xC0000003, histories=history)
    assert bool(np.all((sample > 0) & (sample < 1)))
    assert sample.shape == (2, 4)


def test_iid_moment_and_shard_variance_reject_path_designs_before_array_access() -> None:
    identity = SamplingIdentity(SamplingDesign("stratified", 2, 4), 9, 0, 64)
    batch = HistoryBatch(9, 0, 64).with_sampling(identity)
    with pytest.raises(TransportError, match="replicate variance"):
        history_moments(
            None,
            None,
            None,
            batch=batch,
            workspace=cast(Any, None),
            out_mean=None,
            out_variance_of_mean=None,
        )
    with pytest.raises(TransportError, match="replicate variance"):
        ShardMoments(batch.shard(2, 0), (0.0,), "same-model", status_counts=(32, 0, 0, 0, 0, 0, 0))


def test_complete_replicate_moments_and_rejections() -> None:
    design = SamplingDesign("scrambled-sobol", 2, 4)
    batches = tuple(
        HistoryBatch(seed, 0, 64).with_sampling(SamplingIdentity(design, seed, 0, 64))
        for seed in (4, 6, 9)
    )
    values = np.array([[1.0, 2.0], [3.0, 5.0], [8.0, 7.0]])
    mean, variance = design_replicate_moments(values, batches=batches)
    np.testing.assert_allclose(mean, np.mean(values, axis=0), rtol=2e-16)
    np.testing.assert_allclose(variance, np.var(values, axis=0, ddof=1) / 3, rtol=5e-16)
    for supplied in [(batches[0], batches[0]), tuple(batch.shard(2, 0).batch for batch in batches)]:
        with pytest.raises(TransportError):
            design_replicate_moments(values[: len(supplied)], batches=supplied)


def test_design_shard_means_use_original_count_and_reject_missing_ranges() -> None:
    identity = SamplingIdentity(SamplingDesign("scrambled-sobol", 2, 4), 4, 0, 128)
    batch = HistoryBatch(4, 0, 128).with_sampling(identity)
    shards = tuple(batch.shard(3, rank) for rank in range(3))
    full = sampling_uniforms(identity, event=0, domain=0)[:, :2]
    means = np.asarray(
        [np.mean(full[shard.offset : shard.offset + shard.batch.count], axis=0) for shard in shards]
    )
    np.testing.assert_allclose(
        pool_design_shard_means(means, shards=shards), np.mean(full, axis=0), rtol=5e-16
    )
    for incomplete in ((shards[0], shards[0], shards[2]), shards[:2]):
        with pytest.raises(TransportError):
            pool_design_shard_means(means[: len(incomplete)], shards=incomplete)


@pytest.mark.parametrize("law", ["stratified", "scrambled-sobol"])
def test_collision_path_mean_and_variance_against_exact_absorption_probability(
    law: Literal["stratified", "scrambled-sobol"],
) -> None:
    # An absorbing slab transmits iff its *path* optical-depth coordinate exceeds
    # tau. Source coordinates are absent from this exact transport observable.
    count, tau = 256, 0.71
    design = SamplingDesign(law, 2, 4)
    replicate = []
    iid = []
    for seed in range(48):
        identity = SamplingIdentity(design, seed, 0, count)
        draw = sampling_uniforms(identity, event=0, domain=0)[:, 0]
        replicate.append(float(np.mean(-np.log(draw) > tau)))
        uniform = (
            _philox_words(seed, np.arange(count, dtype=np.uint64), 0, 0)[:, 0].astype(np.float64)
            + 0.5
        ) / 2**32
        iid.append(float(np.mean(-np.log(uniform) > tau)))
    exact = float(np.exp(-tau))
    assert abs(float(np.mean(replicate)) - exact) < 1 / count
    assert float(np.var(replicate, ddof=1)) < float(np.var(iid, ddof=1)) / 16
