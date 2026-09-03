"""Distribution-level checks expose samplers that merely match low moments."""

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
from dpt.validation.detector import poisson_log_mass

wp: Any = importlib.import_module("warp")


def poisson_statistic(samples, rate):
    # Quantile bins merge discrete ties at low rates. Expected probabilities
    # are independently summed from the cancellation-safe mathematical log PMF.
    boundaries = np.unique(stats.poisson.ppf(np.arange(1, 10) / 10, rate).astype(int))
    observed = np.histogram(samples, bins=np.r_[-np.inf, boundaries + 0.5, np.inf])[0]
    lower = max(0, int(rate - 10 * math.sqrt(rate + 1)))
    upper = int(rate + 10 * math.sqrt(rate + 1)) + 10
    k = np.arange(lower, upper + 1)
    masses = np.array([math.exp(poisson_log_mass(int(count), rate)) for count in k])
    expected = np.histogram(k, bins=np.r_[-np.inf, boundaries + 0.5, np.inf], weights=masses)[0]
    expected *= len(samples) / expected.sum()
    assert np.all(expected > 5)
    return stats.chisquare(observed, expected).pvalue


@pytest.mark.parametrize("rate", [0.1, 3.0, 9.999, 10.0, 10.001, 100.0, 1e6])
def test_poisson_sampler_log_mass_chi_square(rate, warp_device):
    with cpu_test_context():
        n = 65536
        w = prepare_detector(max_pixels=n, device=warp_device)
        x = wp.full(n, rate, dtype=wp.float32, device=warp_device)
        out = wp.empty(n, dtype=wp.uint64, device=warp_device)
        sample_poisson_counts(x, out_counts=out, identity=ObservationIdentity(719, 23), workspace=w)
        actual_rate = float(np.float32(rate))
        assert poisson_statistic(out.numpy(), actual_rate) > 1e-6
        # Drop/count-floor mutants with the same output type cannot pass.
        assert poisson_statistic(np.full(n, int(actual_rate)), actual_rate) < 1e-6


def test_read_noise_ks_with_nonzero_sigma_rejects_drop_noise_mutant(warp_device):
    with cpu_test_context():
        n = 32768
        sigma = 2.7
        w = prepare_detector(max_pixels=n, device=warp_device)
        x = wp.zeros(n, dtype=wp.float32, device=warp_device)
        out = wp.empty_like(x)
        add_gaussian_read_noise(
            x,
            standard_deviation=sigma,
            out_signal=out,
            identity=ObservationIdentity(5417, 31),
            workspace=w,
        )
        assert stats.kstest(out.numpy() / sigma, "norm").pvalue > 1e-6
        assert stats.kstest(np.zeros(n), "norm").pvalue < 1e-6
