"""Controller events preserve accepted states, numerical paths and legacy callbacks."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dpt.contracts import ContractError, NumericalError, TrialDomainError
from dpt.pose_optimization import (
    NormalEquations,
    ParameterBounds,
    PoseLeastSquaresPolicy,
    recover_bounded_parameters,
    recover_least_squares,
    recover_pose_sensitivities,
)
from dpt.recovery import recover_pose
from dpt.registration import Evaluation, RecoveryPolicy, recover_parameters
from dpt.solver_lifecycle import solver_termination, termination_reason
from dpt.spectral_recovery import recover_spectral_pose
from dpt.stochastic_recovery import StochasticPolicy, recover_expected_signal


class QuadraticOracle:
    deterministic_sampling = True

    def __init__(self, reject=False):
        self.calls = []
        self.reject = reject
        self.changes = 0

    def gradient_replicate(self, parameters, mean_batch, derivative_batch):
        self.calls.append(("gradient", parameters, mean_batch, derivative_batch))
        return parameters

    def change_replicate(self, before, after, first, second):
        self.calls.append(("change", before, after, first, second))
        self.changes += 1
        return 1.0 if self.reject and self.changes <= 4 else 0.5 * (after[0] ** 2 - before[0] ** 2)


def stochastic_policy():
    return StochasticPolicy(
        iterations=3,
        replicates=4,
        initial_batch=2,
        maximum_batch=2,
        initial_radius=0.2,
    )


def quadratic(point):
    a, b = point[0] - 1, point[1] + 0.3
    return Evaluation(0.5 * (20 * a * a + 2 * b * b + a * b), (20 * a + 0.5 * b, 2 * b + 0.5 * a))


def nonlinear_normal(point):
    x = point[0]
    residual = x * x - 1
    return NormalEquations(0.5 * residual * residual, (2 * x * residual,), ((4 * x * x,),))


def solve(kind, observer=None, *, reject=False) -> tuple[Any, list[Any]]:
    calls = []
    if kind == "registration":

        def registration_oracle(point):
            calls.append(point)
            if reject and point[0] <= 0:
                raise TrialDomainError("positive chart")
            if reject:
                return Evaluation(20 * (point[0] - 0.2) ** 2, (40 * (point[0] - 0.2),))
            return quadratic(point)

        result = recover_parameters(
            registration_oracle, (1.0,) if reject else (3.0, 2.0), observe_lifecycle=observer
        )
    elif kind == "least_squares":

        def least_squares_oracle(point):
            calls.append(point)
            return nonlinear_normal(point)

        result = recover_least_squares(
            least_squares_oracle,
            (0.1,),
            observe_lifecycle=observer,
            policy=PoseLeastSquaresPolicy(max_iterations=20),
        )
    else:
        oracle = QuadraticOracle(reject)
        result = recover_expected_signal(
            oracle, (2.0,), seed=17, policy=stochastic_policy(), observe_lifecycle=observer
        )
        calls = oracle.calls
    return result, calls


@pytest.mark.parametrize("kind", ["registration", "least_squares", "stochastic"])
@pytest.mark.parametrize("reject", [False, True])
def test_events_leave_oracle_calls_and_complete_results_unchanged(kind, reject):
    expected, calls = solve(kind, reject=reject)
    events = []
    actual, observed_calls = solve(kind, events.append, reject=reject)
    assert asdict(actual) == asdict(expected)
    assert observed_calls == calls
    assert events[0].status == "initial"
    assert events[0].iteration == events[0].accepted_steps == 0
    assert events[-1].status == "terminal"
    assert events[-1].termination == solver_termination(actual)
    assert sum(e.status == "terminal" for e in events) == 1
    assert events[-1].accepted_steps == sum(e.status == "accepted" for e in events)
    if kind == "stochastic":
        assert all(e.objective is None for e in events)
        assert events[-1].detail == actual.termination_detail
    else:
        assert events[-1].objective == actual.evaluation.loss
    accepted = events[0]
    for event in events[1:]:
        if event.status == "accepted":
            assert event.accepted_steps == accepted.accepted_steps + 1
            accepted = event
        elif event.status == "rejected":
            assert event.accepted_steps == accepted.accepted_steps
            assert event.objective == accepted.objective
    if reject or kind == "least_squares":
        assert any(e.status == "rejected" for e in events)
    with pytest.raises(FrozenInstanceError):
        events[0].objective = 99


@pytest.mark.parametrize(
    "error_type", [ValueError, NumericalError, TrialDomainError, KeyboardInterrupt, SystemExit]
)
@pytest.mark.parametrize("kind", ["registration", "least_squares", "stochastic"])
@pytest.mark.parametrize("phase", ["accepted", "rejected", "terminal"])
def test_lifecycle_observer_errors_propagate_without_retry(kind, phase, error_type):
    events = []
    sentinel = error_type("recorder failed")

    def observer(event):
        events.append(event)
        if event.status == phase:
            raise sentinel

    with pytest.raises(error_type) as raised:
        solve(kind, observer, reject=True)
    assert raised.value is sentinel
    assert events[-1].status == phase
    assert sum(e.status == phase for e in events) == 1
    snapshot = raised.value.__dict__["solver_lifecycle"]
    assert snapshot.status == "terminal"
    assert snapshot.reason == "exception"
    assert snapshot.accepted_steps == events[-1].accepted_steps
    assert snapshot.objective == events[-1].objective
    assert snapshot.termination.category == "failed"


def test_resume_events_follow_verification_and_preserve_trajectory():
    rules = RecoveryPolicy(
        max_iterations=30, gradient_tolerance=1e-10, relative_loss_tolerance=0, step_tolerance=0
    )
    full = recover_parameters(quadratic, (3.0, 2.0), policy=rules)
    first = recover_parameters(
        quadratic,
        (3.0, 2.0),
        policy=replace(rules, max_iterations=2),
        objective_id="fixed-two-coordinate-chart",
    )
    assert first.checkpoint is not None
    events = []
    resumed = recover_parameters(
        quadratic,
        policy=rules,
        checkpoint=first.checkpoint,
        objective_id=first.checkpoint.objective_id,
        observe_lifecycle=events.append,
    )
    assert events[0].status == "resumed"
    assert events[0].iteration == events[0].accepted_steps == 2
    assert events[0].objective == first.evaluation.loss
    assert resumed.parameters == full.parameters
    assert resumed.evaluation == full.evaluation
    assert [r.loss for r in resumed.history] == [r.loss for r in full.history]
    assert resumed.evaluations == full.evaluations + 1
    bad_events = []
    with pytest.raises(ContractError, match="oracle"):
        recover_parameters(
            lambda p: Evaluation(quadratic(p).loss + 1, quadratic(p).gradient),
            policy=rules,
            checkpoint=first.checkpoint,
            objective_id=first.checkpoint.objective_id,
            observe_lifecycle=bad_events.append,
        )
    assert [e.status for e in bad_events] == ["terminal"]
    assert bad_events[0].reason == "exception"


def test_legacy_observer_and_checkpoint_order_and_committed_failure():
    order = []
    rules = RecoveryPolicy(max_iterations=1, relative_loss_tolerance=0, step_tolerance=0)
    result = recover_parameters(
        quadratic,
        (3.0, 2.0),
        policy=rules,
        objective_id="fixed-v1",
        observe=lambda _: order.append("legacy"),
        save_checkpoint=lambda _: order.append("checkpoint"),
        observe_lifecycle=lambda e: order.append(e.status),
    )
    assert order == ["initial", "accepted", "legacy", "checkpoint", "terminal"]
    events = []
    sentinel = ValueError("legacy recorder")

    def fail(_):
        raise sentinel

    with pytest.raises(ValueError) as raised:
        recover_parameters(
            quadratic,
            (3.0, 2.0),
            policy=rules,
            objective_id="fixed-v1",
            observe=fail,
            observe_lifecycle=events.append,
        )
    assert raised.value is sentinel
    assert [e.status for e in events] == ["initial", "accepted", "terminal"]
    assert events[-1].objective == result.evaluation.loss
    assert raised.value.__dict__["recovery_diagnostics"].parameters == result.parameters


@pytest.mark.parametrize("secondary_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_terminal_observer_failure_does_not_mask_original_oracle_exception(secondary_type):
    original = NumericalError("fixed input failed")
    events = []

    def oracle(point):
        if point == (1.0,):
            return Evaluation(0.5, (1.0,))
        raise original

    def observer(event):
        events.append(event)
        if event.status == "terminal":
            raise secondary_type("secondary recorder error")

    with pytest.raises(NumericalError) as raised:
        recover_parameters(oracle, (1.0,), observe_lifecycle=observer)
    assert raised.value is original
    assert [e.status for e in events] == ["initial", "terminal"]
    assert events[-1].objective == 0.5
    assert events[-1].accepted_steps == 0
    assert secondary_type.__name__ in raised.value.__notes__[-1]


@pytest.mark.parametrize("stop", ["cancelled", "evaluation_budget", "line_search_failed"])
def test_real_registration_terminal_exits_retain_reason_and_incumbent(stop):
    events = []
    calls = []

    def oracle(point):
        calls.append(point)
        return Evaluation(1.0, (1.0,))

    result = recover_parameters(
        oracle,
        (1.0,),
        cancelled=(lambda: True) if stop == "cancelled" else None,
        policy=RecoveryPolicy(max_evaluations=1 if stop == "evaluation_budget" else 100),
        observe_lifecycle=events.append,
    )
    assert result.reason == events[-1].reason == stop
    assert result.parameters == (1.0,)
    assert events[-1].objective == 1.0
    assert events[-1].accepted_steps == 0
    assert sum(e.status == "terminal" for e in events) == 1
    if stop == "line_search_failed":
        assert events[-2].status == "rejected"
        assert events[-2].detail == "line_search_failed"
    else:
        assert len(calls) == 1


def test_least_squares_cancellation_keeps_initial_state():
    events = []
    result = recover_least_squares(
        nonlinear_normal, (0.1,), cancelled=lambda: True, observe_lifecycle=events.append
    )
    assert result.reason == "cancelled"
    assert result.parameters == (0.1,)
    assert [e.status for e in events] == ["initial", "terminal"]
    assert events[-1].accepted_steps == 0


def test_stochastic_budget_and_final_validation_terminal_details_are_retained():
    events = []
    oracle = QuadraticOracle()
    result = recover_expected_signal(
        oracle,
        (2.0,),
        seed=11,
        policy=replace(stochastic_policy(), unique_history_budget=16),
        observe_lifecycle=events.append,
    )
    assert result.reason == "history_budget"
    assert result.parameters == (2.0,)
    assert events[-1].detail == result.termination_detail
    assert events[-1].termination.category == "budget"
    assert len(oracle.calls) == 4


@pytest.mark.parametrize("mode", ["linear", "quadratic", "auto"])
@pytest.mark.parametrize("outcome", ["certified", "nonstationary_budget", "failed_trigger"])
def test_stochastic_terminal_follows_real_held_out_validation(mode, outcome):
    class HeldOutOracle(QuadraticOracle):
        def __init__(self):
            super().__init__()
            self.models = 0
            self.gradients = 0

        def model_replicate(self, parameters, mean_batch, derivative_batch):
            self.models += 1
            self.calls.append(("model", parameters, mean_batch, derivative_batch))
            return parameters, ((1.0,),)

        def gradient_replicate(self, parameters, mean_batch, derivative_batch):
            self.gradients += 1
            value = super().gradient_replicate(parameters, mean_batch, derivative_batch)
            if outcome == "failed_trigger" and (self.models or self.gradients > 4):
                return (1.0,)
            return value

    rules = replace(
        stochastic_policy(),
        iterations=1,
        proposal=mode,
        final_validation_batch=2 if mode == "linear" else None,
    )
    initial = (2.0,) if outcome == "nonstationary_budget" else (0.0,)
    baseline = HeldOutOracle()
    expected = recover_expected_signal(baseline, initial, seed=23, policy=rules)
    observed = HeldOutOracle()
    events, terminal_call_counts = [], []

    def observer(event):
        events.append(event)
        if event.status == "terminal":
            terminal_call_counts.append(len(observed.calls))
            assert len(observed.calls) >= 8
            assert all(call[0] == "gradient" for call in observed.calls[-4:])

    actual = recover_expected_signal(
        observed, initial, seed=23, policy=rules, observe_lifecycle=observer
    )
    assert asdict(actual) == asdict(expected)
    assert observed.calls == baseline.calls
    assert terminal_call_counts == [len(observed.calls)]
    assert len(actual.final_validation_attempts) == 1
    assert actual.final_validation_attempts[0].pool == "final_validation"
    assert actual.reservations[-1].operation == "final_validation"
    assert events[-1].reason == actual.reason
    assert events[-1].detail == actual.termination_detail
    if outcome == "certified":
        assert actual.reason == "gradient_band"
        assert actual.final_validation_outcome == "certified"
        assert actual.termination_detail == "held_out_gradient_band"
    elif outcome == "nonstationary_budget":
        assert actual.reason == "iteration_budget"
        assert actual.final_validation_outcome == "clearly_nonstationary"
        assert actual.final_validation_trigger == "iteration_budget"
    else:
        assert actual.reason == "sampling_unresolved"
        assert actual.final_validation_outcome == "clearly_nonstationary"
        assert actual.termination_detail == "final_validation_failed"


@pytest.mark.parametrize(
    "reason,category",
    [
        ("gradient_tolerance", "stationary"),
        ("projected_gradient_tolerance", "stationary"),
        ("step_tolerance", "stagnation"),
        ("loss_stagnation", "stagnation"),
        ("loss_resolution", "stagnation"),
        ("radius_limit", "stagnation"),
        ("iteration_budget", "budget"),
        ("evaluation_budget", "budget"),
        ("history_budget", "budget"),
        ("cancelled", "cancelled"),
        ("line_search_failed", "failed"),
        ("damping_limit", "failed"),
        ("evaluation_failed", "failed"),
        ("exception", "failed"),
        ("gradient_band", "uncertainty"),
        ("sampling_unresolved", "uncertainty"),
    ],
)
def test_common_reason_categories_preserve_original_meaning(reason, category):
    assert termination_reason(reason).reason == reason
    assert solver_termination(SimpleNamespace(reason=reason)).category == category
    assert solver_termination({"termination": reason}).category == category


def test_unknown_or_absent_reason_is_rejected():
    with pytest.raises(ContractError, match="unknown"):
        termination_reason("maybe_converged")
    with pytest.raises(ContractError, match="string"):
        solver_termination({})


def test_nested_and_concurrent_calls_keep_lifecycle_streams_separate():
    outer, inner = [], []

    def observer(event):
        outer.append(event)
        if event.status == "accepted":
            recover_least_squares(nonlinear_normal, (1.0,), observe_lifecycle=inner.append)

    recover_parameters(
        quadratic, (3.0, 2.0), policy=RecoveryPolicy(max_iterations=2), observe_lifecycle=observer
    )
    assert all(e.solver == "registration" for e in outer)
    assert all(e.solver == "least_squares" for e in inner)
    with ThreadPoolExecutor(max_workers=2) as executor:
        streams = list(executor.map(lambda kind: _record(kind), ["registration", "stochastic"]))
    assert {e.solver for e in streams[0]} == {"registration"}
    assert {e.solver for e in streams[1]} == {"stochastic"}


def _record(kind):
    events = []
    solve(kind, events.append)
    return events


def test_high_level_entry_points_forward_events_without_device_preparation():
    class PoseOracle:
        dimension = 6
        chart = SimpleNamespace(pose=lambda p: p)

        def __call__(self, point):
            return Evaluation(0.0, (0.0,) * len(point))

        def calibration_estimates(self, point):
            return ()

    for driver in (recover_pose, recover_spectral_pose):
        events = []
        result = driver(cast(Any, PoseOracle()), observe_lifecycle=events.append)
        assert [e.status for e in events] == ["initial", "terminal"]
        assert result.optimisation.reason == events[-1].reason == "gradient_tolerance"
    events = []
    result = recover_bounded_parameters(
        lambda p: Evaluation(0.5 * p[0] ** 2, p),
        (1.0,),
        bounds=ParameterBounds((-2.0,), (2.0,)),
        observe_lifecycle=events.append,
    )
    assert events[-1].termination == solver_termination(result)
    normal = SimpleNamespace(chart=SimpleNamespace(pose=lambda p: p))

    class NormalOracle:
        chart = normal.chart

        def __call__(self, point):
            return NormalEquations(
                0.0, (0.0,) * 6, tuple(tuple(float(i == j) for j in range(6)) for i in range(6))
            )

    events = []
    result = recover_pose_sensitivities(cast(Any, NormalOracle()), observe_lifecycle=events.append)
    assert [e.status for e in events] == ["initial", "terminal"]
    assert result.optimisation.reason == "gradient_tolerance"
