"""Tail, per-count and variance checks with explicit statistical mutant controls."""

import importlib
import math
from typing import Any

import numpy as np
import pytest
from scipy import stats

from dpt._runtime import cpu_test_context
from dpt.detector import (
    ObservationIdentity,
    add_gaussian_read_noise,
    prepare_detector,
    sample_poisson_counts,
)

wp: Any = importlib.import_module("warp")


def per_count_pvalue(samples, rate):
    """Keep individual counts whose expectations exceed 20, pooling only tails."""
    n = len(samples)
    support = np.arange(
        int(stats.poisson.ppf(1e-8, rate)), int(stats.poisson.ppf(1 - 1e-8, rate)) + 1
    )
    masses = stats.poisson.pmf(support, rate)
    retained = support[masses * n >= 20]
    lower, upper = int(retained[0]), int(retained[-1])
    while lower > 0 and stats.poisson.cdf(lower - 1, rate) * n < 20:
        lower += 1
    while stats.poisson.sf(upper, rate) * n < 20:
        upper -= 1
    central = np.arange(lower, upper + 1)
    probabilities = np.r_[
        stats.poisson.cdf(lower - 1, rate),
        stats.poisson.pmf(central, rate),
        stats.poisson.sf(upper, rate),
    ]
    observed = np.bincount(
        np.clip(samples.astype(np.int64), lower - 1, upper + 1) - lower + 1,
        minlength=len(probabilities),
    )
    nonempty = probabilities > 0
    expected = probabilities[nonempty] * n
    # Independent PMF/CDF evaluations can differ by tiny normalisation roundoff.
    expected *= observed[nonempty].sum() / expected.sum()
    assert np.all(expected >= 20)
    return stats.chisquare(observed[nonempty], expected).pvalue


def assert_poisson_moments_and_tails(samples, rate):
    n = len(samples)
    centered = samples.astype(np.float64) - rate
    assert abs(centered.mean()) < 6 * math.sqrt(rate / n)
    # Poisson central fourth moment = rate + 3*rate**2.
    assert abs(centered.var(ddof=1) - rate) < 6 * math.sqrt((rate + 2 * rate**2) / n)
    for multiple in (3, 4):
        cutoff = math.floor(rate + multiple * math.sqrt(rate))
        expected = float(stats.poisson.sf(cutoff, rate))
        actual = np.count_nonzero(samples > cutoff) / n
        assert abs(actual - expected) < 6 * math.sqrt(expected * (1 - expected) / n) + 1 / n
    if rate <= 100:
        skew = np.mean((centered / math.sqrt(rate)) ** 3)
        # E[Z^6] bounds the variance of the sample third moment.
        uncertainty = math.sqrt((15 + 25 / rate + 1 / rate**2) / n)
        assert abs(skew - 1 / math.sqrt(rate)) < 6 * uncertainty


@pytest.mark.parametrize("rate", [0.5, 9.999, 10.0, 100.0, 1e6])
@pytest.mark.parametrize("seed", [719, 5417])
def test_poisson_per_count_tail_and_variance(rate, seed, warp_device):
    with cpu_test_context():
        n = 2**22
        workspace = prepare_detector(max_pixels=n, device=warp_device)
        mean = wp.full(n, rate, dtype=wp.float32, device=warp_device)
        output = wp.empty(n, dtype=wp.uint64, device=warp_device)
        sample_poisson_counts(
            mean, out_counts=output, identity=ObservationIdentity(seed, 23), workspace=workspace
        )
        samples = output.numpy()
        actual_rate = float(np.float32(rate))
        assert per_count_pvalue(samples, actual_rate) > 1e-6
        assert_poisson_moments_and_tails(samples, actual_rate)
        # A variance-matching middle of the distribution cannot conceal lost tails.
        clipped = np.clip(
            samples,
            max(0, math.ceil(actual_rate - 2 * math.sqrt(actual_rate))),
            math.floor(actual_rate + 2 * math.sqrt(actual_rate)),
        )
        assert per_count_pvalue(clipped, actual_rate) < 1e-6


def test_poisson_oracles_reject_rounded_normal_and_two_percent_overdispersion():
    n, rate = 2**22, 100.0
    rng = np.random.default_rng(712)
    rounded_normal = np.maximum(0, np.rint(rng.normal(rate, math.sqrt(rate), n)))
    assert per_count_pvalue(rounded_normal, rate) < 1e-6
    with pytest.raises(AssertionError):
        assert_poisson_moments_and_tails(rounded_normal, rate)
    # Gamma-Poisson mixture: E[rate]=100, Var[rate]=2, hence Var[count]=102.
    overdispersed = rng.poisson(rng.gamma(rate / 0.02, 0.02, n))
    with pytest.raises(AssertionError):
        assert_poisson_moments_and_tails(overdispersed, rate)


@pytest.mark.parametrize("seed", [5417, 719, 8131])
def test_read_noise_variance_detects_two_percent_sigma_error(seed, warp_device):
    with cpu_test_context():
        n, sigma = 2**20, 2.7
        workspace = prepare_detector(max_pixels=n, device=warp_device)
        source = wp.zeros(n, dtype=wp.float32, device=warp_device)
        output = wp.empty_like(source)
        add_gaussian_read_noise(
            source,
            standard_deviation=sigma,
            out_signal=output,
            identity=ObservationIdentity(seed, 31),
            workspace=workspace,
        )
        z = output.numpy().astype(np.float64) / sigma
        assert abs(z.mean()) < 6 / math.sqrt(n)
        variance_bound = 6 * math.sqrt(2 / (n - 1))
        assert abs(z.var(ddof=1) - 1) < variance_bound
        assert stats.kstest(z, "norm").pvalue > 1e-6
        assert abs((1.02 * z).var(ddof=1) - 1) > variance_bound
