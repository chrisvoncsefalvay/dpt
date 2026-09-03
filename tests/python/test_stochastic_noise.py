"""Seeded IID Gaussian and sparse CRN cases; no GPU or history simulation."""

import math
import random
from typing import Any, cast

import numpy as np
from scipy.stats import binomtest, t

from dpt.statistics import student_t_critical
from dpt.stochastic_recovery import StochasticPolicy, acceptance_decision, recover_expected_signal


def test_student_critical_matches_independent_library():
    for degrees in (3, 7, 15, 30):
        for alpha in (0.05, 0.005, 0.00005):
            assert math.isclose(
                student_t_critical(degrees, alpha), t.isf(alpha, degrees), rel_tol=2e-12
            )


def test_seeded_null_oracle_acceptance_probability_is_at_most_alpha():
    rng = np.random.default_rng(7319)
    trials = 12000
    looks = 4
    alpha = 0.05
    rows = rng.normal(size=(trials, looks, 4))
    accepted = sum(
        any(
            acceptance_decision(tuple(row), 0.0, alpha=alpha, maximum_looks=looks).outcome
            == "accepted"
            for row in trial
        )
        for trial in rows
    )
    assert 1 - (1 - alpha / looks) ** looks <= alpha
    # A finite seeded sample has binomial variation: reject excessive probability,
    # not a harmless fluctuation around the analytically bounded null rate.
    assert binomtest(accepted, trials, alpha, alternative="greater").pvalue > 0.001
    # The uncalibrated fixed-z two-replicate mutant over-accepts this same null.
    mutant = (
        rows[:, :, :2].mean(axis=2) + 2 * rows[:, :, :2].std(axis=2, ddof=1) / math.sqrt(2) < 0
    ).any(axis=1)
    assert mutant.mean() > alpha


def test_point_estimate_at_threshold_is_rejected_despite_large_uncertainty():
    result = acceptance_decision((-100.0, 100.0, -100.0, 100.0), 0.0, alpha=0.05, maximum_looks=4)
    assert result.outcome == "rejected"


class NoisyStationary:
    deterministic_sampling = False

    def gradient_replicate(self, parameters, a, b):
        rng = random.Random(a.first_history + 239 * a.seed)
        return (parameters[0] + rng.gauss(0, 0.02 / math.sqrt(a.count)),)

    def model_replicate(self, parameters, a, b):
        return self.gradient_replicate(parameters, a, b), ((1.0,),)

    def change_replicate(self, before, after, a, b):
        rng = random.Random(a.first_history + 31 * a.seed)
        residual = rng.gauss(0, 0.02 / math.sqrt(a.count))
        return 0.5 * (after[0] ** 2 - before[0] ** 2) + (after[0] - before[0]) * residual


def test_noisy_default_final_checkpoint_certifies_most_stationary_points():
    policy = StochasticPolicy(
        proposal="quadratic",
        initial_batch=4,
        maximum_batch=4096,
        replicates=8,
        gradient_tolerance=0.002,
        unique_history_budget=500000,
    )
    results = [
        recover_expected_signal(cast(Any, NoisyStationary()), (0.0,), seed=i, policy=policy)
        for i in range(80)
    ]
    assert sum(r.reason == "gradient_band" for r in results) >= 72
    assert all(r.final_validation_batch == 4096 for r in results)
    assert all(len(r.final_validation_attempts) <= 1 for r in results)


def test_sparse_zero_samples_do_not_certify_stationarity():
    class RareHit(NoisyStationary):
        def gradient_replicate(self, parameters, a, b):
            rng = random.Random(a.first_history + 13 * a.seed)
            return (100.0 / a.count if rng.random() < a.count / 1000 else 0.0,)

    result = recover_expected_signal(
        cast(Any, RareHit()),
        (0.0,),
        seed=9,
        policy=StochasticPolicy(initial_batch=4, maximum_batch=64, replicates=4, iterations=1),
    )
    assert result.reason != "gradient_band"
