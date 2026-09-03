"""Public recovery checks for per-call reports and failed-work accounting."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from threading import Barrier

import pytest

from dpt.contracts import NumericalError
from dpt.registration import Vector
from dpt.solver_lifecycle import SolverEvent
from dpt.stochastic_recovery import StochasticPolicy, recover_expected_signal
from dpt.transport.rng import HistoryBatch


class ReportingOracle:
    deterministic_sampling = True

    def __init__(self, *, barrier: Barrier | None = None, trace_mode: str = "counter"):
        self.calls: list[tuple[str, Vector, Vector | None, HistoryBatch, HistoryBatch]] = []
        self.timeline: list[str] = []
        self.traced = 0
        self.changes = 0
        self.gradients = 0
        self.fail_change: int | None = None
        self.fail_gradient: int | None = None
        self.error: BaseException = NumericalError("later oracle failed")
        self.barrier = barrier
        self.trace_mode = trace_mode

    @property
    def histories_traced(self) -> object:
        self.timeline.append("trace")
        return {"counter": self.traced, "none": None, "bool": True, "text": "unknown"}[
            self.trace_mode
        ]

    def record(self, kind: str, before: Vector, after: Vector | None, a, b):
        first_call = not self.calls
        self.calls.append((kind, before, after, a, b))
        self.timeline.append(kind)
        self.traced += a.count + b.count
        if first_call and self.barrier is not None:
            self.barrier.wait(timeout=3)

    def gradient_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> Vector:
        self.record("gradient", parameters, None, mean_batch, derivative_batch)
        self.gradients += 1
        if self.gradients == self.fail_gradient:
            raise self.error
        return parameters

    def model_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ):
        self.record("model", parameters, None, mean_batch, derivative_batch)
        return (
            parameters,
            tuple(
                tuple(float(i == j) for j in range(len(parameters))) for i in range(len(parameters))
            ),
        )

    def change_replicate(
        self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
    ):
        self.record("change", before, after, first, second)
        self.changes += 1
        if self.changes == self.fail_change:
            raise self.error
        return 0.5 * sum((y * y - x * x for x, y in zip(before, after, strict=True)))


def policy(*, quadratic: bool = False):
    return StochasticPolicy(
        iterations=2,
        replicates=4,
        initial_batch=1,
        maximum_batch=1,
        initial_radius=0.1,
        maximum_radius=0.1,
        proposal="quadratic" if quadratic else "linear",
    )


def serialised(result):
    return json.dumps(asdict(result), sort_keys=True, allow_nan=False)


def test_nested_stochastic_reports_keep_current_incumbents_and_final_metadata_separate():
    expected_oracle = ReportingOracle()
    expected = recover_expected_signal(expected_oracle, (1.0,), seed=11, policy=policy())
    inner_expected_oracle = ReportingOracle()
    inner_expected = recover_expected_signal(
        inner_expected_oracle, (2.0,), seed=21, policy=policy(quadratic=True)
    )
    outer_oracle, inner_oracle = ReportingOracle(), ReportingOracle()
    inner_results = []

    def observer(event: SolverEvent):
        if event.status == "accepted" and not inner_results:
            inner_results.append(
                recover_expected_signal(
                    inner_oracle, (2.0,), seed=21, policy=policy(quadratic=True)
                )
            )

    outer = recover_expected_signal(
        outer_oracle, (1.0,), seed=11, policy=policy(), observe_lifecycle=observer
    )
    assert outer == expected and outer_oracle.calls == expected_oracle.calls
    assert inner_results == [inner_expected] and inner_oracle.calls == inner_expected_oracle.calls
    assert outer.parameters == (0.8,) and inner_results[0].parameters == (1.7999999999999998,)
    assert outer.final_validation_outcome is None
    assert inner_results[0].final_validation_outcome == "clearly_nonstationary"
    assert [m.model_id for m in inner_results[0].local_models] == [0, 1]


def test_concurrent_stochastic_reports_match_sequential_complete_reports_and_calls():
    def run(quadratic, barrier=None):
        oracle = ReportingOracle(barrier=barrier)
        result = recover_expected_signal(
            oracle,
            (2.0,) if quadratic else (1.0,),
            seed=21 if quadratic else 11,
            policy=policy(quadratic=quadratic),
        )
        return result, oracle.calls, oracle.timeline

    expected = [run(False), run(True)]
    barrier = Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(lambda quadratic: run(quadratic, barrier), [False, True]))
    assert actual == expected


def test_failure_after_acceptance_reports_incumbent_and_actual_partial_reservation():
    oracle = ReportingOracle()
    oracle.fail_change = 6
    original = oracle.error
    events = []
    with pytest.raises(NumericalError, match="later oracle failed") as caught:
        recover_expected_signal(
            oracle, (1.0,), seed=11, policy=policy(), observe_lifecycle=events.append
        )
    assert caught.value is original
    result = caught.value.__dict__["recovery_diagnostics"]
    assert result.parameters == (0.9,) and len(result.steps) == 1 and result.steps[0].accepted
    assert [a.outcome for a in result.acceptance_attempts] == ["accepted", "oracle_error"]
    failed = result.reservations[-1]
    assert (failed.first_history, failed.reserved_histories) == (24, 8)
    assert (failed.attempted_replicates, failed.completed_replicates) == (2, 1)
    assert failed.status == "oracle_error"
    assert len(result.acceptance_attempts[-1].replicate_changes) == 1
    assert result.unique_histories == 32 and result.histories_traced == 28
    assert caught.value.__notes__ == [
        "Recovery reservation/attempt snapshot is in recovery_diagnostics."
    ]
    assert [e.status for e in events] == ["initial", "accepted", "terminal"]
    assert events[-1].accepted_steps == 1 and events[-1].reason == "exception"
    assert oracle.timeline == [
        "trace",
        *["gradient"] * 4,
        *["change"] * 4,
        *["gradient"] * 4,
        *["change"] * 2,
        "trace",
    ]
    serialised(result)


def test_fatal_failure_preserves_existing_exception_only_report_boundary():
    oracle = ReportingOracle()
    oracle.fail_change = 6
    oracle.error = SystemExit("fatal later oracle")
    with pytest.raises(SystemExit) as caught:
        recover_expected_signal(oracle, (1.0,), seed=11, policy=policy())
    assert caught.value is oracle.error
    assert "recovery_diagnostics" not in caught.value.__dict__
    assert caught.value.__dict__["solver_lifecycle"].accepted_steps == 1
    assert oracle.timeline.count("trace") == 1


def test_reused_oracle_does_not_mutate_previous_report_or_leak_failed_run_state():
    oracle = ReportingOracle()
    first = recover_expected_signal(oracle, (2.0,), seed=21, policy=policy(quadratic=True))
    frozen = serialised(first)
    before_failure = oracle.traced
    oracle.fail_change = oracle.changes + 6
    with pytest.raises(NumericalError) as caught:
        recover_expected_signal(oracle, (1.0,), seed=11, policy=policy())
    failed = caught.value.__dict__["recovery_diagnostics"]
    assert failed.histories_traced == oracle.traced - before_failure
    oracle.fail_change = None
    before_last = oracle.traced
    last = recover_expected_signal(oracle, (2.0,), seed=21, policy=policy(quadratic=True))
    assert serialised(first) == frozen and first == last
    assert last.histories_traced == oracle.traced - before_last
    assert failed.parameters == (0.9,) and failed.final_validation_trigger is None
    assert [m.model_id for m in last.local_models] == [0, 1]
    assert last.reservations[0].first_history == 0


@pytest.mark.parametrize("trace_mode", ["counter", "none", "bool", "text"])
@pytest.mark.parametrize("fail", [False, True])
def test_trace_property_reads_stay_at_entry_and_result_time(trace_mode, fail):
    oracle = ReportingOracle(trace_mode=trace_mode)
    oracle.fail_change = 6 if fail else None
    if fail:
        with pytest.raises(NumericalError) as caught:
            recover_expected_signal(oracle, (1.0,), seed=11, policy=policy())
        result = caught.value.__dict__["recovery_diagnostics"]
    else:
        result = recover_expected_signal(oracle, (1.0,), seed=11, policy=policy())
    assert oracle.timeline[0] == "trace"
    reads = 2 if trace_mode == "counter" else 1
    assert oracle.timeline.count("trace") == reads
    assert oracle.timeline[-1] == ("trace" if reads == 2 else "change")
    assert result.histories_traced == (oracle.traced if trace_mode == "counter" else None)


class ScheduledOracle(ReportingOracle):
    deterministic_sampling = False

    def gradient_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> Vector:
        super().gradient_replicate(parameters, mean_batch, derivative_batch)
        sign = 1 if mean_batch.first_history // (2 * mean_batch.count) % 2 else -1
        return (sign * (0.1 if mean_batch.count == 4 else 0.001),)


@pytest.mark.parametrize("fail_second_look", [False, True])
def test_multiple_held_out_looks_retain_work_and_partial_failure_metadata(fail_second_look):
    oracle = ScheduledOracle()
    oracle.fail_gradient = 10 if fail_second_look else None
    selected = replace(
        policy(),
        initial_batch=2,
        maximum_batch=16,
        stationarity_calibration="student_t",
        gradient_tolerance=0.02,
        final_validation_batches=(4, 8),
    )
    events = []
    if fail_second_look:
        with pytest.raises(NumericalError) as caught:
            recover_expected_signal(
                oracle, (0.0,), seed=7, policy=selected, observe_lifecycle=events.append
            )
        result = caught.value.__dict__["recovery_diagnostics"]
        assert result.final_validation_outcome == "not_certified"
        assert [a.batch_size for a in result.final_validation_attempts] == [4]
        failed = result.reservations[-1]
        assert failed.operation == "final_validation" and failed.first_history == 48
        assert (failed.reserved_histories, failed.completed_replicates) == (64, 1)
        assert failed.attempted_replicates == 2 and failed.status == "oracle_error"
        assert result.histories_traced == 80
    else:
        result = recover_expected_signal(
            oracle, (0.0,), seed=7, policy=selected, observe_lifecycle=events.append
        )
        assert result.reason == "gradient_band" and result.final_validation_outcome == "certified"
        assert [a.batch_size for a in result.final_validation_attempts] == [4, 8]
        assert [a.confidence_alpha for a in result.final_validation_attempts] == [0.025, 0.025]
        assert result.histories_traced == 112
    assert result.parameters == (0.0,) and result.steps == ()
    assert result.final_validation_trigger == "proposal_gradient_band"
    assert result.unique_histories == 112
    assert [e.status for e in events] == ["initial", "terminal"]
    assert oracle.timeline[0] == oracle.timeline[-1] == "trace"
    serialised(result)
