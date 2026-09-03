"""Independent Hermite-polynomial and sufficient-decrease/curvature oracles."""

import math

import pytest

from dpt.contracts import ContractError, NumericalError, TrialDomainError
from dpt.line_search import (
    CubicSample,
    CubicWolfePolicy,
    ScalarEvaluation,
    cubic_strong_wolfe,
    cubic_trial,
)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
@pytest.mark.parametrize("reverse", [False, True])
def test_hermite_cubic_recovers_known_interior_minimum(scale: float, reverse: bool) -> None:
    # A true cubic with a known stationary minimum, independent of the solver.
    def sample(x: float) -> CubicSample:
        displacement = x - 0.7
        return CubicSample(
            x,
            ScalarEvaluation(
                scale * (displacement**2 + displacement**3 / 5),
                scale * (2 * displacement + 3 * displacement**2 / 5),
            ),
        )

    left, right = sample(0.0), sample(1.0)
    if reverse:
        left, right = right, left
    assert cubic_trial(left, right) == pytest.approx(0.7, abs=2e-15)


@pytest.mark.parametrize(
    "left,right,minimum", [(-3.0, 5.0, 0.3), (0.1, 9.0, 4.0), (-11.0, -2.0, -7.0)]
)
def test_unequal_signed_bracket_quadratic(left: float, right: float, minimum: float) -> None:
    a = CubicSample(left, ScalarEvaluation((left - minimum) ** 2, 2 * (left - minimum)))
    b = CubicSample(right, ScalarEvaluation((right - minimum) ** 2, 2 * (right - minimum)))
    assert cubic_trial(a, b) == pytest.approx(minimum)


def test_guard_flat_and_monotone_polynomials_use_midpoint() -> None:
    for values in (
        (ScalarEvaluation(0.0, 0.0), ScalarEvaluation(0.0, 0.0)),
        (ScalarEvaluation(0.0, 1.0), ScalarEvaluation(1.0, 1.0)),
        (ScalarEvaluation(0.0001, -0.02), ScalarEvaluation(0.9801, 1.98)),
    ):
        assert cubic_trial(CubicSample(0.0, values[0]), CubicSample(1.0, values[1])) == 0.5
    with pytest.raises(ContractError):
        cubic_trial(
            CubicSample(0.0, ScalarEvaluation(0.0, 0.0)),
            CubicSample(0.0, ScalarEvaluation(0.0, 0.0)),
        )


@pytest.mark.parametrize("initial", [0.001, 1.0, 12.0])
def test_search_satisfies_actual_strong_wolfe(initial: float) -> None:
    def function(step: float) -> ScalarEvaluation:
        return ScalarEvaluation((step - 2.3) ** 2, 2 * (step - 2.3))

    base = function(0.0)
    policy = CubicWolfePolicy(curvature=0.1)
    result = cubic_strong_wolfe(function, base, initial_step=initial, policy=policy)
    assert result.reason == "wolfe"
    assert result.evaluation.loss <= base.loss + policy.armijo * result.step * base.derivative
    assert abs(result.evaluation.derivative) <= -policy.curvature * base.derivative
    assert result.evaluation == function(result.step)


def test_domain_rejection_only_bisects_when_endpoint_is_unavailable() -> None:
    calls: list[float] = []

    def function(step: float) -> ScalarEvaluation:
        calls.append(step)
        if step >= 0.8:
            raise TrialDomainError("declared chart boundary")
        return ScalarEvaluation((step - 0.3) ** 2, 2 * (step - 0.3))

    result = cubic_strong_wolfe(function, function(0.0), initial_step=2.0)
    assert result.accepted and result.reason == "wolfe"
    assert result.rejected[0][0] == 2.0
    assert all("chart boundary" in message for _, message in result.rejected)
    assert 0 < result.step < 0.8
    assert calls[1:4] == [2.0, 1.0, 0.5]


def test_real_constraint_boundary_can_accept_armijo_but_algorithm_cap_cannot() -> None:
    def decreasing(step: float) -> ScalarEvaluation:
        return ScalarEvaluation(10 - step, -1.0)

    bounded = cubic_strong_wolfe(decreasing, decreasing(0.0), feasible_maximum=0.2)
    assert bounded.reason == "armijo_bound" and bounded.step == 0.2
    capped = cubic_strong_wolfe(
        decreasing, decreasing(0.0), policy=CubicWolfePolicy(maximum_step=0.2)
    )
    assert not capped.accepted and capped.step == 0 and capped.evaluation == decreasing(0.0)


def test_failed_budget_and_nondescent_preserve_base() -> None:
    base = ScalarEvaluation(1.0, -1.0)
    result = cubic_strong_wolfe(
        lambda _: ScalarEvaluation(2.0, -1.0), base, policy=CubicWolfePolicy(max_evaluations=1)
    )
    assert result.reason == "evaluation_budget" and result.evaluation is base
    result = cubic_strong_wolfe(
        lambda _: pytest.fail("must not evaluate"), ScalarEvaluation(1.0, 0.0)
    )
    assert result.reason == "not_descent" and result.evaluations == 0


@pytest.mark.parametrize(
    "value", [ScalarEvaluation(math.inf, -1.0), ScalarEvaluation(0.0, math.nan)]
)
def test_numerical_failure_is_fatal(value: ScalarEvaluation) -> None:
    with pytest.raises(NumericalError):
        cubic_strong_wolfe(lambda _: value, ScalarEvaluation(1.0, -1.0))
