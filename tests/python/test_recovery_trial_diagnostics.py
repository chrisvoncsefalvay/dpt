"""Regressions for rejected chart trials and preserved recovery diagnostics."""

import math
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any

import pytest

from dpt.contracts import ContractError, NumericalError, TrialDomainError
from dpt.recovery import PoseChart, PrimaryPoseEvaluator
from dpt.registration import Evaluation, RecoveryPolicy, Vector, recover_parameters
from dpt.spectral_recovery import CalibrationBlock, SpectralPoseEvaluator
from dpt.stochastic_models import Matrix
from dpt.stochastic_recovery import StochasticPolicy, recover_expected_signal
from dpt.transport.rng import HistoryBatch


def test_pose_trial_leaving_chart_backtracks_and_recovers() -> None:
    chart = PoseChart(scales=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), rotation_radius_radians=0.5)

    def objective(parameters: Vector) -> Evaluation:
        chart.pose(parameters)
        residual = parameters[3] + 0.3
        return Evaluation(residual**2, (0.0, 0.0, 0.0, 2 * residual, 0.0, 0.0))

    result = recover_parameters(objective, (0.0, 0.0, 0.0, 0.45, 0.0, 0.0))
    assert result.reason == "gradient_tolerance"
    assert result.parameters[3] == pytest.approx(-0.3)
    assert result.rejected_trials


def test_calibration_trial_overflow_is_rejectable() -> None:
    block = CalibrationBlock("gain", fit_scale="gain")
    with pytest.raises(TrialDomainError):
        block.decode((1000.0,))


def test_fatal_numerical_failure_retains_last_accepted_result() -> None:
    def objective(parameters: Vector) -> Evaluation:
        if parameters[0] != 1.0:
            raise NumericalError("corrupt fixed data")
        return Evaluation(0.5, (1.0,))

    with pytest.raises(NumericalError, match="corrupt fixed data") as failure:
        recover_parameters(objective, (1.0,))
    snapshot = failure.value.__dict__["recovery_diagnostics"]
    assert snapshot.parameters == (1.0,)
    assert snapshot.evaluation == Evaluation(0.5, (1.0,))
    assert snapshot.evaluations == 2
    assert len(snapshot.history) == 1


class ExactScalarQuadratic:
    deterministic_sampling = True

    def gradient_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> Vector:
        return parameters

    def model_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> tuple[Vector, Matrix]:
        return parameters, ((1.0,),)

    def change_replicate(
        self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
    ) -> float:
        step = after[0] - before[0]
        return before[0] * step + 0.5 * step**2


def quadratic_policy(**updates: Any) -> StochasticPolicy:
    return StochasticPolicy(
        proposal="quadratic",
        initial_batch=2,
        maximum_batch=2,
        replicates=4,
        iterations=1,
        **updates,
    )


def test_small_radius_at_ordinary_chart_magnitude_survives_rounding() -> None:
    radius = 0.1 * 2**-10
    result = recover_expected_signal(
        ExactScalarQuadratic(),
        (-6.9,),
        seed=11,
        policy=quadratic_policy(initial_radius=radius),
    )
    assert result.acceptance_attempts
    attempt = result.acceptance_attempts[0]
    assert attempt.outcome == "accepted"
    assert math.hypot(*attempt.step) <= radius + math.ulp(6.9)
    assert attempt.boundary
    assert result.parameters != (-6.9,)


def test_failed_final_checkpoint_preserves_iteration_budget_reason() -> None:
    result = recover_expected_signal(
        ExactScalarQuadratic(),
        (2.0,),
        seed=11,
        policy=quadratic_policy(),
    )
    assert result.reason == "iteration_budget"
    assert result.termination_detail == "iteration_budget"
    assert result.final_validation_outcome == "clearly_nonstationary"
    assert result.final_validation_attempts[0].decision == "resolved"


@pytest.mark.parametrize("evaluator_class", [PrimaryPoseEvaluator, SpectralPoseEvaluator])
def test_evaluator_rejects_concurrent_use_before_staging_changes(evaluator_class: Any) -> None:
    # Exercise the actual call boundary with a blocked evaluation, without CUDA.
    evaluator = evaluator_class.__new__(evaluator_class)
    evaluator._evaluation_lock = Lock()
    entered, release = Event(), Event()
    calls: list[Vector] = []

    def evaluate(parameters: Vector) -> Evaluation:
        calls.append(parameters)
        entered.set()
        assert release.wait(timeout=2)
        return Evaluation(0.0, (0.0,))

    evaluator._evaluate = evaluate
    evaluator.context = SimpleNamespace(
        wp=SimpleNamespace(synchronize_stream=lambda stream: None), stream=None
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(evaluator, (1.0,))
        try:
            assert entered.wait(timeout=2)
            with pytest.raises(ContractError, match="concurrently or recursively"):
                evaluator((2.0,))
            assert calls == [(1.0,)]
        finally:
            release.set()
        assert first.result(timeout=2) == Evaluation(0.0, (0.0,))
    assert evaluator((3.0,)) == Evaluation(0.0, (0.0,))


def test_line_search_history_distinguishes_armijo_fallback() -> None:
    def kink(parameters: Vector) -> Evaluation:
        return Evaluation(abs(parameters[0] - 0.3), (1.0 if parameters[0] > 0.3 else -1.0,))

    result = recover_parameters(
        kink,
        (1.0,),
        policy=RecoveryPolicy(curvature=0.1, line_search_evaluations=5, max_iterations=1),
    )
    assert result.history[0].line_search == "initial"
    assert result.history[1].line_search == "armijo_best"
