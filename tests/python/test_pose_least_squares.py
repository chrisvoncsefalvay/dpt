"""Independent least-squares, active-constraint and accepted-state oracles."""

import math
from typing import Literal

import pytest

from dpt.contracts import ContractError, NumericalError, TrialDomainError
from dpt.pose_optimization import (
    BoundedGradientPolicy,
    NormalEquations,
    ParameterBounds,
    PoseLeastSquaresPolicy,
    recover_bounded_parameters,
    recover_least_squares,
)
from dpt.recovery import PoseChart
from dpt.registration import Evaluation


def linear(point: tuple[float, ...]) -> NormalEquations:
    x, y = point
    first, second = x + 2 * y - 3, 2 * x - y - 1
    return NormalEquations(
        0.5 * (first**2 + second**2),
        (first + 2 * second, 2 * first - second),
        ((5.0, 0.0), (0.0, 5.0)),
    )


@pytest.mark.parametrize("method", ["gauss_newton", "levenberg_marquardt"])
def test_linear_model_recovers_independent_normal_equation_solution(
    method: Literal["gauss_newton", "levenberg_marquardt"],
) -> None:
    result = recover_least_squares(
        linear, (4.0, -5.0), policy=PoseLeastSquaresPolicy(method=method, gradient_tolerance=1e-10)
    )
    assert result.stationary and result.parameters == pytest.approx((1.0, 1.0), abs=1e-10)
    assert result.evaluation.loss < 1e-20
    assert all(a.loss >= b.loss for a, b in zip(result.history, result.history[1:], strict=False))
    if method == "gauss_newton":
        assert result.evaluations == 2  # Exact linear least squares takes one full Newton step.


@pytest.mark.parametrize("method", ["gauss_newton", "levenberg_marquardt"])
def test_coupled_bounded_model_converges_to_kkt_solution(
    method: Literal["gauss_newton", "levenberg_marquardt"],
) -> None:
    calls: list[tuple[float, ...]] = []

    def coupled(point: tuple[float, ...]) -> NormalEquations:
        calls.append(point)
        x, y = point
        first, second = x + 2 * y - 1, 2 * x + 3 * y
        return NormalEquations(
            0.5 * (first**2 + second**2),
            (first + 2 * second, 2 * first + 3 * second),
            ((5.0, 8.0), (8.0, 13.0)),
        )

    box = ParameterBounds((0.0, -math.inf), (0.8, math.inf))
    result = recover_least_squares(
        coupled,
        (0.3, 0.2),
        bounds=box,
        policy=PoseLeastSquaresPolicy(method=method, max_iterations=300, gradient_tolerance=1e-9),
    )
    assert result.reason == "projected_gradient_tolerance"
    assert result.parameters == pytest.approx((0.0, 2 / 13), abs=1e-9)
    assert result.evaluation.gradient[0] == pytest.approx(3 / 13)
    assert abs(result.evaluation.gradient[1]) < 1e-9
    assert all(box.project(point) == point for point in calls)


def test_lm_rejects_overshoot_and_keeps_accepted_history() -> None:
    def nonlinear(point: tuple[float, ...]) -> NormalEquations:
        x = point[0]
        residual = x * x - 1
        return NormalEquations(0.5 * residual**2, (2 * x * residual,), ((4 * x * x,),))

    result = recover_least_squares(
        nonlinear,
        (0.1,),
        policy=PoseLeastSquaresPolicy(max_iterations=100, gradient_tolerance=1e-10),
    )
    assert result.stationary and result.parameters[0] == pytest.approx(1.0, abs=1e-9)
    assert any(not row.accepted for row in result.history)
    for previous, current in zip(result.history, result.history[1:], strict=False):
        if not current.accepted:
            assert current.parameters == previous.parameters and current.loss == previous.loss
        else:
            assert current.loss <= previous.loss


def test_rank_deficient_lm_and_fixed_coordinate_bounds() -> None:
    def singular(point: tuple[float, ...]) -> NormalEquations:
        residual = sum(point) - 1
        return NormalEquations(0.5 * residual**2, (residual, residual), ((1.0, 1.0), (1.0, 1.0)))

    unconstrained = recover_least_squares(singular, (0.0, 0.0))
    assert unconstrained.stationary and sum(unconstrained.parameters) == pytest.approx(
        1.0, abs=1e-7
    )
    constrained = recover_least_squares(
        singular, (0.2, 0.0), bounds=ParameterBounds((0.2, -2.0), (0.2, 2.0))
    )
    assert constrained.stationary and constrained.parameters == pytest.approx((0.2, 0.8), abs=1e-7)


def test_budget_domain_cancel_and_small_step_do_not_claim_stationarity() -> None:
    def invalid(point: tuple[float, ...]) -> NormalEquations:
        if point != (0.0,):
            raise TrialDomainError("trial outside fixed chart")
        return NormalEquations(1.0, (-1.0,), ((1.0,),))

    result = recover_least_squares(
        invalid, (0.0,), policy=PoseLeastSquaresPolicy(max_evaluations=3)
    )
    assert result.reason == "evaluation_budget" and not result.stationary
    assert result.parameters == (0.0,) and len(result.rejected_trials) == 2
    cancelled = recover_least_squares(linear, (0.0, 0.0), cancelled=lambda: True)
    assert cancelled.reason == "cancelled" and cancelled.evaluations == 1
    small = recover_least_squares(
        linear, (0.0, 0.0), policy=PoseLeastSquaresPolicy(step_tolerance=10.0)
    )
    assert small.reason == "step_tolerance" and not small.stationary


@pytest.mark.parametrize(
    "normal",
    [NormalEquations(math.nan, (1.0,), ((1.0,),)), NormalEquations(1.0, (1.0,), ((math.inf,),))],
)
def test_nonfinite_normal_equations_fail_explicitly(normal: NormalEquations) -> None:
    with pytest.raises(NumericalError):
        recover_least_squares(lambda _: normal, (0.0,))


def test_indefinite_or_dimension_mismatched_normal_equations_are_contract_errors() -> None:
    with pytest.raises(ContractError, match="positive semidefinite"):
        recover_least_squares(
            lambda _: NormalEquations(1.0, (1.0, 1.0), ((1.0, 2.0), (2.0, 1.0))), (0.0, 0.0)
        )
    with pytest.raises(ContractError, match="dimension"):
        recover_least_squares(lambda _: NormalEquations(1.0, (1.0,), ((1.0,),)), (0.0, 0.0))
    with pytest.raises(ContractError, match="initial"):
        recover_least_squares(linear, (-1.0, 0.0), bounds=ParameterBounds((0.0, -2.0), (2.0, 2.0)))


def test_translation_bounds_use_declared_fixed_chart_mm() -> None:
    chart = PoseChart(scales=(2.0, 3.0, 4.0, 0.01, 0.02, 0.03))
    box = ParameterBounds.pose_translation(chart, (-2.0, -6.0, -12.0), (4.0, 9.0, 16.0))
    assert box.lower == (-1.0, -2.0, -3.0, -math.inf, -math.inf, -math.inf)
    assert box.upper == (2.0, 3.0, 4.0, math.inf, math.inf, math.inf)


def test_existing_gradient_evaluator_has_feasible_calibration_coordinate_steps() -> None:
    def evaluator(point: tuple[float, ...]) -> Evaluation:
        normal = linear(point)
        return Evaluation(normal.loss, normal.gradient)

    result = recover_bounded_parameters(
        evaluator,
        (0.0, 0.0),
        bounds=ParameterBounds((0.0, -2.0), (0.4, 2.0)),
        policy=BoundedGradientPolicy(max_iterations=200),
    )
    assert result.reason == "projected_gradient_tolerance"
    assert result.parameters == pytest.approx((0.4, 1.0), abs=1e-7)
