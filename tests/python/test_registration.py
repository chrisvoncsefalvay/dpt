"""Independent optimiser acceptance cases, including finite-range safeguards."""

import math
from typing import Any

import pytest

from dpt.contracts import ContractError, NumericalError, TrialDomainError
from dpt.registration import Evaluation, RecoveryPolicy, recover_parameters

# pytest.approx has incomplete upstream annotations; keep that boundary local.
approx: Any = pytest.approx  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def test_scaled_quadratic_recovers_known_stationary_point():
    def evaluate(x: tuple[float, ...]) -> Evaluation:
        residual = (x[0] - 2.0, x[1] + 3.0)
        return Evaluation(
            0.5 * (residual[0] ** 2 + 7 * residual[1] ** 2), (residual[0], 7 * residual[1])
        )

    result = recover_parameters(evaluate, (11.0, 4.0))
    assert result.stationary
    assert result.parameters == approx((2.0, -3.0), abs=2e-6)
    assert all(a.loss >= b.loss for a, b in zip(result.history, result.history[1:], strict=False))


def test_invalid_trials_shrink_without_replacing_the_accepted_state():
    rejected: list[float] = []

    def evaluate(x: tuple[float, ...]) -> Evaluation:
        if x[0] <= 0:
            rejected.append(x[0])
            raise TrialDomainError("outside positive parameter domain")
        return Evaluation(x[0] - math.log(x[0]), (1.0 - 1.0 / x[0],))

    result = recover_parameters(evaluate, (0.01,))
    assert result.parameters[0] > 0
    assert result.evaluation.loss <= 0.01 - math.log(0.01)


def test_budget_returns_last_accepted_parameters():
    def evaluate(x: tuple[float, ...]) -> Evaluation:
        return Evaluation(x[0] ** 2, (2 * x[0],))

    result = recover_parameters(evaluate, (3.0,), policy=RecoveryPolicy(max_evaluations=1))
    assert result.reason == "evaluation_budget"
    assert result.parameters == (3.0,)
    assert result.evaluations == 1


def test_programming_errors_are_not_misreported_as_failed_line_search():
    with pytest.raises(ContractError, match="dimension"):
        recover_parameters(lambda _: Evaluation(1.0, (1.0,)), (1.0, 2.0))


def test_stationarity_and_cancellation_are_distinct():
    stationary = recover_parameters(lambda _: Evaluation(4.0, (0.0,)), (1.0,))
    cancelled = recover_parameters(
        lambda _: Evaluation(4.0, (1.0,)), (1.0,), cancelled=lambda: True
    )
    assert stationary.stationary
    assert cancelled.reason == "cancelled"
    assert not cancelled.stationary


def test_bad_derivative_never_forces_acceptance():
    # Objective increases along the claimed descent direction at every trial.
    result = recover_parameters(lambda x: Evaluation(x[0] ** 2, (-2 * x[0],)), (1.0,))
    assert result.reason == "line_search_failed"
    assert result.parameters == (1.0,)


def test_rosenbrock_checks_curvature_memory_on_a_nonquadratic_valley():
    def evaluate(x: tuple[float, ...]) -> Evaluation:
        a = 1 - x[0]
        b = x[1] - x[0] ** 2
        return Evaluation(a * a + 100 * b * b, (-2 * a - 400 * x[0] * b, 200 * b))

    result = recover_parameters(evaluate, (-1.2, 1.0), policy=RecoveryPolicy(max_iterations=500))
    assert result.evaluation.loss < 1e-8
    assert result.parameters == approx((1.0, 1.0), abs=5e-4)


def test_unrepresentable_directional_product_preserves_the_incumbent() -> None:
    # Each gradient component is finite; its squared norm is not. Such a
    # direction cannot supply a representable strong-Wolfe certificate.
    result = recover_parameters(lambda _: Evaluation(1.0, (1e308, -1e308)), (0.0, 0.0))
    assert result.reason == "line_search_failed"
    assert result.parameters == (0.0, 0.0)
    assert result.evaluation == Evaluation(1.0, (1e308, -1e308))
    assert result.evaluations == 1


def test_nonfinite_trial_gradient_is_rejected_before_acceptance() -> None:
    attempts: list[tuple[float, ...]] = []

    def evaluate(x: tuple[float, ...]) -> Evaluation:
        attempts.append(x)
        # The proposed point has a lower loss but no valid derivative. Returning
        # it would incorrectly certify a failed reverse pass as convergence.
        return Evaluation(1.0, (1.0,)) if x == (0.0,) else Evaluation(0.0, (math.nan,))

    with pytest.raises(NumericalError):
        recover_parameters(evaluate, (0.0,))
    assert len(attempts) == 2


def test_large_objective_scale_does_not_exhaust_initial_backtracking() -> None:
    def evaluate(x: tuple[float, ...]) -> Evaluation:
        return Evaluation(0.5e20 * x[0] ** 2, (1e20 * x[0],))

    result = recover_parameters(evaluate, (1.0,))
    assert result.stationary
    assert result.parameters == (0.0,)
    assert result.evaluations == 2


def test_empty_curvature_memory_still_retries_a_failed_initial_search() -> None:
    # A narrow quadratic requires a representable displacement much smaller
    # than the scaled unit direction. The conservative retry gets its own
    # bounded search budget even before the first curvature pair exists.
    target = 2.0**-38

    def evaluate(x: tuple[float, ...]) -> Evaluation:
        residual = x[0] - target
        return Evaluation(0.5 * (residual / target) ** 2, (residual / target**2,))

    result = recover_parameters(evaluate, (0.0,))
    assert result.stationary
    assert result.parameters == (target,)


def test_step_bound_accepts_armijo_decrease_without_wolfe_curvature() -> None:
    result = recover_parameters(
        lambda x: Evaluation(-x[0], (-1.0,)),
        (0.0,),
        policy=RecoveryPolicy(max_iterations=1, maximum_step=0.25),
    )
    assert result.reason == "iteration_budget"
    assert result.parameters == (0.25,)
    assert result.evaluation.loss == -0.25
    assert result.history[-1].step == 0.25
