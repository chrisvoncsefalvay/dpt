"""Independent finite-sample algebra, Gaussian calibration and controller checks."""

import json
import math
from dataclasses import asdict, replace
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from scipy.stats import binomtest, t

from dpt.contracts import ContractError, NumericalError
from dpt.registration import Vector
from dpt.statistics import all_pairs_product, all_pairs_squared_change, predicted_batch_size
from dpt.stochastic_recovery import StochasticPolicy, _gradient_attempt, recover_expected_signal
from dpt.transport.rng import HistoryBatch, require_independent


def _exhaustive_product(left, right):
    return sum(
        (
            Fraction.from_float(float(a)) * Fraction.from_float(float(b))
            for i, a in enumerate(left)
            for j, b in enumerate(right)
            if i != j
        ),
        Fraction(0),
    ) / (len(left) * (len(left) - 1))


def test_all_pairs_gradient_excludes_correlated_same_batch_products():
    left = (0.2, -0.7, 1.1, 0.4, -0.8, 0.6)
    right = (0.4, -0.5, 0.9, -0.1, -0.6, 0.5)
    result = all_pairs_product(left, right)
    exact = _exhaustive_product(left, right)
    pseudovalues = [
        float(
            len(left) * exact
            - (len(left) - 1)
            * _exhaustive_product(left[:i] + left[i + 1 :], right[:i] + right[i + 1 :])
        )
        for i in range(len(left))
    ]
    assert result.estimate == float(exact)
    np.testing.assert_allclose(result.pseudovalues, pseudovalues, rtol=0, atol=0)
    np.testing.assert_allclose(
        result.standard_error,
        np.std(pseudovalues, ddof=1) / math.sqrt(len(left)),
        rtol=2e-15,
    )
    assert result.estimate != np.mean(left) * np.mean(right)
    assert result.uncertainty_method == "delete_one_batch_jackknife_asymptotic"


def test_all_pairs_squared_change_keeps_crn_and_large_cancellation():
    before = (1e150,) * 6
    after = (math.nextafter(1e150, math.inf),) * 6
    exact = (_exhaustive_product(after, after) - _exhaustive_product(before, before)) / 2
    result = all_pairs_squared_change(before, after)
    assert result.estimate == float(exact)
    assert result.standard_error == 0
    assert all_pairs_squared_change(before, before).estimate == 0
    assert all_pairs_squared_change(before, before).standard_error == 0
    with pytest.raises(NumericalError):
        all_pairs_product((1e308,) * 4)


def test_near_zero_residual_all_pairs_variance_matches_independent_gaussian_law():
    rng = np.random.default_rng(712946)
    count = 16
    draws = rng.normal(size=(24000, count))
    # Independent explicit pair enumeration checks the public implementation;
    # the large seeded law check uses the independently vectorised pair algebra.
    for row in draws[:32]:
        np.testing.assert_allclose(
            all_pairs_product(tuple(row)).estimate,
            sum(row[i] * row[j] for i in range(count) for j in range(i + 1, count))
            / math.comb(count, 2),
            rtol=2e-14,
            atol=2e-16,
        )
    complete = (draws.sum(axis=1) ** 2 - (draws**2).sum(axis=1)) / (count * (count - 1))
    disjoint = (draws[:, ::2] * draws[:, 1::2]).mean(axis=1)
    np.testing.assert_allclose(np.var(complete, ddof=1), 2 / (count * (count - 1)), rtol=0.08)
    np.testing.assert_allclose(np.var(disjoint, ddof=1), 2 / count, rtol=0.06)
    assert np.var(complete) < np.var(disjoint) / 12


@pytest.mark.parametrize("replicates", [4, 8, 16])
def test_stationarity_student_vector_bound_matches_scipy_and_has_gaussian_coverage(replicates):
    rng = np.random.default_rng(19031 + replicates)
    dimension, alpha = 3, 0.05
    true_gradient = np.asarray((0.6, 0.8, 0.0))
    rows = rng.normal(size=(16000, replicates, dimension)) * 0.12 + true_gradient
    means = rows.mean(axis=1)
    errors = rows.std(axis=1, ddof=1) / math.sqrt(replicates)
    critical = t.isf(alpha / (2 * dimension), replicates - 1)
    bounds = np.linalg.norm(np.abs(means) + critical * errors, axis=1)
    failures = ((np.abs(means - true_gradient) > critical * errors).any(axis=1)).sum()
    assert binomtest(int(failures), len(rows), alpha, alternative="greater").pvalue > 0.001
    policy = StochasticPolicy(stationarity_calibration="student_t", stationarity_alpha=alpha)
    for index in range(24):
        attempt = _gradient_attempt(
            tuple(tuple(row) for row in rows[index]),
            tuple(true_gradient),
            policy,
            False,
            iteration=0,
            batch_size=4,
            first_history=0,
            histories_used=2 * replicates * 4,
            model_id=None,
            pool="final_validation",
        )
        assert attempt.stationarity_upper_bound is not None
        np.testing.assert_allclose(attempt.stationarity_upper_bound, bounds[index], rtol=2e-13)
        assert attempt.band_method == "student_t_bonferroni_vector"
        assert attempt.confidence_alpha == alpha
    if replicates == 4:
        # This fixture distinguishes the calibrated vector policy from fixed z=2.
        mutant_failures = (np.abs(means - true_gradient) > 2 * errors).any(axis=1).sum()
        assert (
            binomtest(int(mutant_failures), len(rows), alpha, alternative="greater").pvalue < 1e-12
        )


@pytest.mark.parametrize(
    "uncertainty,target,expected", [(4, 1, 160), (1, 2, 11), (0, 1, 20), (1e308, 1e-300, 1000)]
)
def test_predicted_batch_is_bounded_and_range_safe(uncertainty, target, expected):
    assert predicted_batch_size(10, 1000, uncertainty, target) == expected


class ScalarQuadratic:
    deterministic_sampling = True
    supports_local_model = True

    def __init__(self):
        self.calls = []
        self.histories_traced = 17

    def gradient_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> Vector:
        first, second = mean_batch, derivative_batch
        self.calls.append(("gradient", first, second))
        self.histories_traced += first.count + 2 * second.count
        return tuple(parameters)

    def model_replicate(self, parameters, first, second):
        return self.gradient_replicate(parameters, first, second), ((1.0,),)

    def change_replicate(self, before, after, first, second):
        self.calls.append(("change", first, second))
        self.histories_traced += 2 * (first.count + second.count)
        return 0.5 * (after[0] ** 2 - before[0] ** 2)


class BatchedQuadratic(ScalarQuadratic):
    def __init__(self):
        super().__init__()
        self.batched_calls = []

    def gradient_replicates(self, parameters, pairs):
        self.batched_calls.append("gradient")
        return tuple(
            super(BatchedQuadratic, self).gradient_replicate(parameters, a, b) for a, b in pairs
        )

    def model_replicates(self, parameters, pairs):
        self.batched_calls.append("model")
        return tuple(
            super(BatchedQuadratic, self).model_replicate(parameters, a, b) for a, b in pairs
        )

    def change_replicates(self, before, after, pairs):
        self.batched_calls.append("change")
        return tuple(
            super(BatchedQuadratic, self).change_replicate(before, after, a, b) for a, b in pairs
        )


def _small_policy(**updates: Any):
    return replace(
        StochasticPolicy(initial_batch=2, maximum_batch=16, replicates=4, iterations=6), **updates
    )


def test_plural_oracles_preserve_scalar_trajectory_and_identity_ledger():
    policy = _small_policy(proposal="auto")
    scalar, batched = ScalarQuadratic(), BatchedQuadratic()
    first = recover_expected_signal(scalar, (0.4,), seed=31, policy=policy)
    second = recover_expected_signal(batched, (0.4,), seed=31, policy=policy)
    assert first.parameters == second.parameters == (0.0,)
    assert first.reason == second.reason == "gradient_band"
    assert first.steps == second.steps
    assert first.unique_histories == second.unique_histories
    assert first.histories_traced == second.histories_traced == batched.histories_traced - 17
    assert first.histories_traced is not None
    assert first.histories_traced > first.unique_histories
    assert len(batched.batched_calls) == len(second.reservations)
    assert all(row.execution == "batched" for row in second.reservations)
    require_independent(
        *(batch for row in second.reservations for pair in row.pairs for batch in pair)
    )
    json.dumps(asdict(second), allow_nan=False)


def test_auto_respects_disabled_local_model_capability():
    oracle = ScalarQuadratic()
    oracle.supports_local_model = False
    result = recover_expected_signal(oracle, (0.4,), seed=1, policy=_small_policy(proposal="auto"))
    assert result.proposal_policy == "linear"
    assert result.local_models == ()


def test_malformed_plural_output_keeps_failed_reservation_snapshot():
    class MissingReplicate(BatchedQuadratic):
        def model_replicates(self, parameters, pairs):
            return super().model_replicates(parameters, pairs)[:-1]

    with pytest.raises(ContractError, match="exactly one result") as failure:
        recover_expected_signal(
            MissingReplicate(), (0.4,), seed=3, policy=_small_policy(proposal="auto")
        )
    snapshot = failure.value.__dict__["recovery_diagnostics"]
    assert snapshot.parameters == (0.4,)
    assert snapshot.reservations[0].execution == "batched"
    assert snapshot.reservations[0].status == "oracle_error"
    assert snapshot.reservations[0].completed_replicates == 0
    assert snapshot.reservations[0].attempted_replicates == 4


def test_predeclared_stationarity_looks_spend_alpha_and_keep_independent_pools():
    class ScheduledNoise(ScalarQuadratic):
        deterministic_sampling = False

        def gradient_replicate(
            self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
        ) -> Vector:
            first, second = mean_batch, derivative_batch
            super().gradient_replicate(parameters, first, second)
            sign = 1 if first.first_history // (2 * first.count) % 2 else -1
            return (sign * (0.1 if first.count == 4 else 0.001),)

    oracle = ScheduledNoise()
    result = recover_expected_signal(
        oracle,
        (0.0,),
        seed=7,
        policy=_small_policy(
            stationarity_calibration="student_t",
            gradient_tolerance=0.02,
            final_validation_batches=(4, 8),
        ),
    )
    assert result.reason == "gradient_band"
    assert result.final_validation_outcome == "certified"
    assert [row.batch_size for row in result.final_validation_attempts] == [4, 8]
    assert all(row.confidence_alpha == 0.025 for row in result.final_validation_attempts)
    assert len(result.gradient_attempts) == 1
    assert result.steps == ()
    require_independent(*(batch for _, first, second in oracle.calls for batch in (first, second)))


def test_predicted_gradient_growth_skips_blind_doubling_epochs():
    class RootNNoise(ScalarQuadratic):
        deterministic_sampling = False

        def gradient_replicate(
            self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
        ) -> Vector:
            first, second = mean_batch, derivative_batch
            super().gradient_replicate(parameters, first, second)
            sign = 1 if first.first_history // (2 * first.count) % 2 else -1
            return (1 + sign * 4 / math.sqrt(first.count),)

    result = recover_expected_signal(
        RootNNoise(),
        (1.0,),
        seed=1,
        policy=_small_policy(
            batch_growth="predicted", iterations=1, initial_batch=2, maximum_batch=64
        ),
    )
    sizes = [row.batch_size for row in result.gradient_attempts]
    assert sizes[0] == 2
    assert sizes[1] > 4
    assert len(sizes) <= 3


def test_predicted_acceptance_growth_uses_fresh_decisions_with_declared_alpha():
    class RootNChange(ScalarQuadratic):
        deterministic_sampling = False

        def gradient_replicate(
            self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
        ) -> Vector:
            super().gradient_replicate(parameters, mean_batch, derivative_batch)
            sign = 1 if mean_batch.first_history // (2 * mean_batch.count) % 2 else -1
            return (parameters[0] + sign * 1e-6,)

        def change_replicate(self, before, after, first, second):
            change = super().change_replicate(before, after, first, second)
            sign = 1 if first.first_history // (2 * first.count) % 2 else -1
            return change + sign * 0.25 / math.sqrt(first.count)

    oracle = RootNChange()
    result = recover_expected_signal(
        oracle,
        (1.0,),
        seed=12,
        policy=_small_policy(
            batch_growth="predicted",
            maximum_batch=128,
            acceptance_maximum_looks=3,
            acceptance_error_scope="run",
            iterations=1,
        ),
    )
    attempts = result.acceptance_attempts
    assert attempts[0].outcome == "ambiguous"
    assert attempts[-1].outcome == "accepted"
    assert attempts[1].batch_size > 4
    assert len(attempts) <= 3
    assert all(attempt.confidence_alpha == 0.05 / 3 for attempt in attempts)
    assert all(attempt.error_scope == "run" for attempt in attempts)
    require_independent(*(batch for _, first, second in oracle.calls for batch in (first, second)))


def test_run_wide_acceptance_alpha_and_linear_cache_are_explicit():
    class Rejections(ScalarQuadratic):
        def change_replicate(self, before, after, first, second):
            super().change_replicate(before, after, first, second)
            return 1.0

    oracle = Rejections()
    result = recover_expected_signal(
        oracle,
        (1.0,),
        seed=5,
        policy=_small_policy(
            cache_linear_gradient=True,
            linear_radius_growth="agreement",
            acceptance_error_scope="run",
            acceptance_maximum_looks=2,
            iterations=3,
        ),
    )
    assert len(result.gradient_attempts) == 1
    assert len(result.acceptance_attempts) == 3
    assert result.acceptance_error_scope == "run"
    assert result.reason == "iteration_budget"
    assert len([call for call in oracle.calls if call[0] == "gradient"]) == 4


def test_linear_radius_growth_requires_good_agreement_when_selected():
    class WeakAgreement(ScalarQuadratic):
        def change_replicate(self, before, after, first, second):
            super().change_replicate(before, after, first, second)
            return -0.2 * abs(after[0] - before[0]) * before[0]

    result = recover_expected_signal(
        WeakAgreement(),
        (1.0,),
        seed=4,
        policy=_small_policy(linear_radius_growth="agreement", iterations=3),
    )
    assert all(row.accepted for row in result.steps)
    assert [row.radius for row in result.steps] == [0.1] * 3


@pytest.mark.parametrize(
    "updates",
    [
        {"acceptance_maximum_looks": 0},
        {"stationarity_alpha": 0.5},
        {"final_validation_batches": (2, 4)},
        {"stationarity_calibration": "student_t", "final_validation_batches": (4, 2)},
    ],
)
def test_invalid_statistical_policy_is_rejected_before_sampling(updates):
    with pytest.raises(ContractError):
        _small_policy(**updates)
