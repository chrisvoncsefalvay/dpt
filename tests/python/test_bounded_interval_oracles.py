"""Independent bounded laws and explicit fixed-proposal validation contracts."""

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.stats import binom, binomtest

from dpt.bounded_statistics import (
    BoundedChangeAdapter,
    BoundedChangePlan,
    ReplicateBounds,
    empirical_bernstein_interval,
    validate_bounded_change,
)
from dpt.contracts import ContractError
from dpt.registration import Vector
from dpt.transport.rng import HistoryBatch, TransportError


def _independent_radius(values, width, alpha):
    return math.sqrt(2 * np.var(values, ddof=1) * math.log(4 / alpha) / len(values)) + (
        7 * width * math.log(4 / alpha) / (3 * (len(values) - 1))
    )


@pytest.mark.parametrize("alpha", [0.2, 0.05, 1e-250])
def test_interval_matches_independent_formula_and_support_endpoints(alpha):
    values = tuple(np.linspace(-2.0, 3.0, 129))
    bounds = ReplicateBounds(-4.0, 5.0)
    result = empirical_bernstein_interval(values, bounds=bounds, alpha=alpha)
    radius = min(bounds.upper - bounds.lower, _independent_radius(values, 9, alpha))
    assert result.mean == pytest.approx(np.mean(values), abs=1e-15)
    assert result.standard_error == pytest.approx(np.std(values, ddof=1) / math.sqrt(129))
    assert result.radius >= radius
    assert result.radius == pytest.approx(radius, rel=2e-15)
    assert result.lower == pytest.approx(max(-4, np.mean(values) - radius))
    assert result.upper == pytest.approx(min(5, np.mean(values) + radius))
    assert result.method == "maurer_pontil_empirical_bernstein_two_sided"
    constant = empirical_bernstein_interval((-4.0,) * 129, bounds=bounds, alpha=alpha)
    assert constant.standard_error == 0
    assert constant.radius > 0
    assert constant.lower == -4
    assert constant.upper > -4
    assert constant.upper <= 5


def test_bernoulli_exact_law_controls_rare_missed_events_and_detects_zero_variance_mutant():
    count, alpha = 128, 0.05
    successes = np.arange(count + 1)
    means = successes / count
    variances = count * means * (1 - means) / (count - 1)
    radii = np.sqrt(2 * variances * math.log(4 / alpha) / count) + (
        7 * math.log(4 / alpha) / (3 * (count - 1))
    )
    lower, upper = np.maximum(0, means - radii), np.minimum(1, means + radii)
    # All possible sufficient statistics checked against an independent formula.
    for k in successes:
        result = empirical_bernstein_interval(
            (1.0,) * int(k) + (0.0,) * int(count - k),
            bounds=ReplicateBounds(0.0, 1.0),
            alpha=alpha,
        )
        assert result.lower <= lower[k] + 1e-15
        assert result.upper >= upper[k] - 1e-15
        assert result.radius == pytest.approx(min(1, radii[k]), rel=2e-15)
    for probability in (0.01, 0.05, 0.5, 0.95, 0.99):
        probability_mass = binom.pmf(successes, count, probability)
        failure = probability_mass[(lower > probability) | (upper < probability)].sum()
        assert failure <= alpha
    # Omitting the finite-range term reports certainty after a rare event is
    # missed. The exact failure mass already exceeds alpha for the all-zero row.
    assert (1 - 0.01) ** count > 5 * alpha
    assert (
        empirical_bernstein_interval(
            (0.0,) * count, bounds=ReplicateBounds(0.0, 1.0), alpha=alpha
        ).upper
        > 0.01
    )


def test_capped_heavy_tail_calibration_matches_known_expectation_without_normality():
    rng = np.random.default_rng(199801)
    trials, count, shape, cap, alpha = 4000, 128, 0.7, 1000.0, 0.05
    # P(X>x)=x^-shape for x>=1, with a point mass at the cap. Its bounded
    # expectation follows independently by integrating that survival function.
    values = np.minimum(1 + rng.pareto(shape, size=(trials, count)), cap)
    true_mean = 1 + (cap ** (1 - shape) - 1) / (1 - shape)
    means, variances = values.mean(axis=1), values.var(axis=1, ddof=1)
    radius = np.sqrt(2 * variances * math.log(4 / alpha) / count) + (
        7 * (cap - 1) * math.log(4 / alpha) / (3 * (count - 1))
    )
    failures = np.count_nonzero((means - radius > true_mean) | (means + radius < true_mean))
    assert binomtest(int(failures), trials, alpha, alternative="greater").pvalue > 0.001
    assert np.mean((values == cap).any(axis=1)) > 0.5
    for row in values[:12]:
        result = empirical_bernstein_interval(
            tuple(row), bounds=ReplicateBounds(1.0, cap), alpha=alpha
        )
        independent = _independent_radius(row, cap - 1, alpha)
        assert result.radius == pytest.approx(min(cap - 1, independent), rel=2e-15)
        assert result.lower <= true_mean <= result.upper


@pytest.mark.parametrize(
    "bounds,values",
    [
        (ReplicateBounds(-1e308, 0.0), (-1e308, -1e308)),
        (ReplicateBounds(0.0, 1e308), (1e308, 1e308)),
        (ReplicateBounds(-1e-300, 1e-300), (-1e-300, 1e-300, 0.0)),
        (
            ReplicateBounds(1e308, math.nextafter(1e308, math.inf)),
            (1e308, math.nextafter(1e308, math.inf), 1e308),
        ),
    ],
)
def test_extreme_finite_supports_preserve_mean_and_finite_clipped_endpoints(bounds, values):
    result = empirical_bernstein_interval(values, bounds=bounds, alpha=0.05)
    exact_mean = sum((Fraction.from_float(value) for value in values), Fraction(0)) / len(values)
    assert result.mean == float(exact_mean)
    assert bounds.lower <= result.lower <= result.mean <= result.upper <= bounds.upper
    assert all(math.isfinite(value) for value in (result.standard_error, result.radius))
    # Short looks exhaust the support instead of overflowing an unclipped endpoint.
    assert (result.lower, result.upper) == (bounds.lower, bounds.upper)


def test_degenerate_certified_support_is_exact_and_does_not_infer_zero_variance():
    result = empirical_bernstein_interval((3.0,) * 2, bounds=ReplicateBounds(3.0, 3.0), alpha=0.05)
    assert (result.mean, result.lower, result.upper) == (3, 3, 3)
    assert result.radius == result.standard_error == 0
    with pytest.raises(ContractError, match="certified support"):
        empirical_bernstein_interval((3.0, 3.00001), bounds=ReplicateBounds(3, 3), alpha=0.05)


def test_tiny_sample_deviations_survive_a_broad_certified_support():
    result = empirical_bernstein_interval(
        (1e-300, 2e-300), bounds=ReplicateBounds(0.0, 1e308), alpha=0.05
    )
    assert result.standard_error == pytest.approx(5e-301, rel=2e-15, abs=0)
    assert result.lower == 0
    assert result.upper == 1e308


def _pairs(counts=(2, 8, 64), batch_size=7):
    first = 0
    looks = []
    for count in counts:
        pairs = []
        for _ in range(count):
            a = HistoryBatch(71, first, batch_size).with_library_source("test-source")
            b = HistoryBatch(71, first + batch_size, batch_size).with_library_source("test-source")
            pairs.append((a, b))
            first += 2 * batch_size
        looks.append(tuple(pairs))
    return tuple(looks)


class BoundedOracle:
    def __init__(self, value: float = -1.0, bounds: ReplicateBounds | None = None):
        self.value = value
        self.bounds = ReplicateBounds(-1, 1) if bounds is None else bounds
        self.calls: list[tuple[str, int]] = []

    def change_bounds(self, before: Vector, after: Vector, batch_size: int) -> ReplicateBounds:
        self.calls.append(("bounds", batch_size))
        return self.bounds

    def change_replicate(
        self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
    ) -> float:
        self.calls.append(("change", first.first_history))
        return self.value


def test_fixed_schedule_declares_bounds_before_samples_and_allocates_run_and_look_error():
    oracle = BoundedOracle()
    plan = BoundedChangePlan(_pairs(), alpha=0.05, maximum_proposals=10)
    result = validate_bounded_change(oracle, (0.0,), (1.0,), plan=plan)
    assert result.decision == "accepted"
    assert [row.decision for row in result.checkpoints] == ["ambiguous", "ambiguous", "accepted"]
    assert oracle.calls[:3] == [("bounds", 7)] * 3
    assert result.alpha_per_look == pytest.approx(0.05 / 10 / 3)
    assert [row.interval.alpha for row in result.checkpoints] == [result.alpha_per_look] * 3
    assert result.reserved_histories == result.sampled_histories == 2 * 7 * (2 + 8 + 64)
    assert [len(row.changes) for row in result.checkpoints] == [2, 8, 64]
    assert result.checkpoints[0].interval.radius == 2
    assert result.checkpoints[-1].interval.radius > 0


def test_explicit_adapter_forwards_an_existing_oracle_without_clipping_or_resampling():
    oracle = BoundedOracle(-1.0)
    adapter = BoundedChangeAdapter(oracle, ReplicateBounds(-2, -1))
    plan = BoundedChangePlan(_pairs((2, 8)))
    result = validate_bounded_change(adapter, (0.0,), (1.0,), plan=plan)
    assert result.decision == "accepted"
    assert result.checkpoints[0].changes == (-1.0, -1.0)
    assert oracle.calls == [("change", pair[0].first_history) for pair in plan.looks[0]]
    with pytest.raises(ContractError, match="certified support"):
        validate_bounded_change(
            BoundedChangeAdapter(oracle, ReplicateBounds(0, 1)), (0.0,), (1.0,), plan=plan
        )


@pytest.mark.parametrize(
    "value,bounds,threshold,expected",
    [
        (-1.0, ReplicateBounds(-2, -1), 0.0, "accepted"),
        (1.0, ReplicateBounds(1, 2), 0.0, "rejected"),
        (0.0, ReplicateBounds(-2, 0), 0.0, "ambiguous"),
    ],
)
def test_fixed_schedule_endpoints_and_early_decisions(value, bounds, threshold, expected):
    plan = BoundedChangePlan(_pairs((2, 8)))
    result = validate_bounded_change(
        BoundedOracle(value, bounds), (0.0,), (1.0,), plan=plan, threshold=threshold
    )
    assert result.decision == expected
    assert len(result.checkpoints) == (2 if expected == "ambiguous" else 1)
    assert result.sampled_histories <= result.reserved_histories


def test_invalid_identity_schedule_and_observed_support_abort_before_a_decision():
    pair = _pairs((2,))[0][0]
    with pytest.raises(TransportError, match="disjoint"):
        BoundedChangePlan(((pair, pair),))
    with pytest.raises(ContractError, match="equal counts"):
        BoundedChangePlan((((HistoryBatch(8, 0, 1), HistoryBatch(8, 1, 2)), _pairs((2,))[0][0]),))
    with pytest.raises(ContractError, match="strictly increasing"):
        BoundedChangePlan(_pairs((2, 2)))
    with pytest.raises(ContractError, match="certified support"):
        validate_bounded_change(
            BoundedOracle(2.0), (0.0,), (1.0,), plan=BoundedChangePlan(_pairs())
        )


@pytest.mark.parametrize(
    "values,bounds,alpha",
    [
        ((0.0,), ReplicateBounds(0, 1), 0.05),
        ((0.0, math.inf), ReplicateBounds(0, 1), 0.05),
        ((0.0, True), ReplicateBounds(0, 1), 0.05),
        ((0.0, 2.0), ReplicateBounds(0, 1), 0.05),
        ((0.0, 1.0), ReplicateBounds(0, 1), 0.0),
        ((0.0, 1.0), ReplicateBounds(0, 1), 1.0),
    ],
)
def test_invalid_confidence_checkpoint_contracts(values, bounds, alpha):
    with pytest.raises(ContractError):
        empirical_bernstein_interval(values, bounds=bounds, alpha=alpha)
    with pytest.raises(ContractError):
        ReplicateBounds(-1e308, 1e308)
