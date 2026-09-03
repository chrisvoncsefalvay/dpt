"""Resident lifecycle observers leave fields, checkpoints and numerical work intact."""

from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError, NumericalError, TrialDomainError
from dpt.material_reconstruction import MaterialReconstruction
from dpt.solver_lifecycle import solver_termination
from helpers import material_inputs as cases


def inputs(device, **settings):
    result = cases._inputs("float64")
    result["device"] = device
    result["settings"] = replace(result["settings"], **settings)
    return result


def without_time(value):
    if isinstance(value, dict):
        return {k: without_time(v) for k, v in value.items() if k != "wall_seconds"}
    if isinstance(value, list):
        return [without_time(v) for v in value]
    return value


@pytest.mark.parametrize("exit_kind", ["accepted", "stationary", "search_failed"])
def test_material_events_preserve_report_fields_predictions_and_calls(warp_device, exit_kind):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        options = {}
        if exit_kind == "stationary":
            options = {"gradient_mapping_tolerance": 2}
        elif exit_kind == "search_failed":
            options = {"initial_step": 1e20, "maximum_step": 1e20, "maximum_backtracks": 1}
        supplied = inputs(warp_device, **options)
        baseline = MaterialReconstruction(**supplied)
        expected = baseline.solve()
        solver = MaterialReconstruction(**supplied)
        calls = []
        evaluate = solver.evaluate

        def count(*, gradient=True, trial=False):
            calls.append((gradient, trial))
            return evaluate(gradient=gradient, trial=trial)

        solver.evaluate = count
        events = []
        actual = solver.solve(observe_lifecycle=events.append)
        if warp_device == "cpu":
            assert without_time(actual) == without_time(expected)
        else:
            assert actual["termination"] == expected["termination"]
            assert actual["accepted_steps"] == expected["accepted_steps"]
            assert actual["evaluations"] == expected["evaluations"]
            np.testing.assert_allclose(
                actual["final_objective"], expected["final_objective"], rtol=2e-6, atol=1e-8
            )
        np.testing.assert_allclose(
            solver.fractions_numpy(), baseline.fractions_numpy(), rtol=2e-6, atol=5e-7
        )
        for observed, original in zip(
            solver.predictions_numpy(), baseline.predictions_numpy(), strict=True
        ):
            np.testing.assert_allclose(observed, original, rtol=2e-6, atol=1e-8)
        assert events[0].status == "initial"
        assert events[-1].status == "terminal"
        assert events[-1].termination == solver_termination(actual)
        assert events[-1].accepted_steps == actual["accepted_steps"]
        assert events[-1].objective == actual["final_objective"]
        assert sum(e.status == "terminal" for e in events) == 1
        if exit_kind == "search_failed":
            assert [e.status for e in events] == ["initial", "rejected", "terminal"]
            assert events[1].objective == events[0].objective
            assert events[1].accepted_steps == 0
            assert calls == [(True, False), (False, True), (False, False)]
        elif exit_kind == "stationary":
            assert [e.status for e in events] == ["initial", "terminal"]
            assert calls == [(True, False)]


def test_material_resume_events_follow_verification_and_preserve_trajectory(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        supplied = inputs(warp_device, iterations=5, step_selection="bb")
        baseline = MaterialReconstruction(**supplied)
        expected = baseline.solve()
        first = MaterialReconstruction(**inputs(warp_device, iterations=2, step_selection="bb"))
        first.solve(objective_id="fixed-v1")
        checkpoint = first.checkpoint()
        events = []
        resumed = MaterialReconstruction(**supplied)
        actual = resumed.solve(
            checkpoint=checkpoint, objective_id="fixed-v1", observe_lifecycle=events.append
        )
        assert events[0].status == "resumed"
        assert events[0].iteration == events[0].accepted_steps == 2
        assert events[0].objective == checkpoint.state["accepted_loss"]
        assert events[-1].accepted_steps == actual["accepted_steps"] == expected["accepted_steps"]
        np.testing.assert_allclose(
            resumed.fractions_numpy(), baseline.fractions_numpy(), rtol=2e-6, atol=5e-7
        )
        np.testing.assert_allclose(
            [r["objective"] for r in actual["history"]],
            [r["objective"] for r in expected["history"]],
            rtol=2e-6,
            atol=1e-8,
        )
        assert actual["evaluations"] == expected["evaluations"] + 1
        bad_events = []
        with pytest.raises(ContractError, match="identity"):
            MaterialReconstruction(**supplied).solve(
                checkpoint=checkpoint,
                objective_id="different-model",
                observe_lifecycle=bad_events.append,
            )
        assert [e.status for e in bad_events] == ["terminal"]


@pytest.mark.parametrize("phase", ["accepted", "rejected", "terminal"])
@pytest.mark.parametrize("error_type", [ValueError, NumericalError, TrialDomainError])
def test_material_observer_error_never_backtracks_or_retries(warp_device, phase, error_type):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        options: dict[str, Any] = {"iterations": 1}
        if phase == "rejected":
            options.update(initial_step=1e20, maximum_step=1e20, maximum_backtracks=1)
        solver = MaterialReconstruction(**inputs(warp_device, **options))
        events = []
        sentinel = error_type("recorder error")

        def observer(event):
            events.append(event)
            if event.status == phase:
                raise sentinel

        with pytest.raises(error_type) as raised:
            solver.solve(objective_id="fixed-v1", observe_lifecycle=observer)
        assert raised.value is sentinel
        assert events[-1].status == phase
        assert sum(e.status == phase for e in events) == 1
        snapshot = raised.value.__dict__["solver_lifecycle"]
        assert snapshot.objective == events[-1].objective
        assert snapshot.accepted_steps == events[-1].accepted_steps
        if phase != "rejected":
            checkpoint = solver.checkpoint()
            assert checkpoint.state["accepted"] == 1
            assert checkpoint.state["accepted_loss"] == snapshot.objective
            # Committed fields remain reusable after a recorder exception.
            assert np.isfinite(solver.evaluate()).all()


def test_material_legacy_callback_order_and_exception_checkpoint(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        supplied = inputs(warp_device, iterations=1)
        order = []
        solver = MaterialReconstruction(**supplied)
        expected = solver.solve(
            objective_id="fixed-v1",
            callback=lambda *_: order.append("legacy"),
            save_checkpoint=lambda _: order.append("checkpoint"),
            observe_lifecycle=lambda e: order.append(e.status),
        )
        assert [value for value in order if value != "rejected"] == [
            "initial",
            "accepted",
            "legacy",
            "checkpoint",
            "terminal",
        ]
        failed = MaterialReconstruction(**supplied)
        events = []
        sentinel = ValueError("legacy recorder")

        def callback(*_):
            raise sentinel

        with pytest.raises(ValueError) as raised:
            failed.solve(
                objective_id="fixed-v1", callback=callback, observe_lifecycle=events.append
            )
        assert raised.value is sentinel
        assert [e.status for e in events if e.status != "rejected"] == [
            "initial",
            "accepted",
            "terminal",
        ]
        checkpoint = failed.checkpoint()
        assert checkpoint.state["accepted"] == 1
        assert checkpoint.state["accepted_loss"] == expected["final_objective"]
        np.testing.assert_allclose(
            failed.fractions_numpy(), solver.fractions_numpy(), rtol=2e-6, atol=5e-7
        )


def test_fresh_checkpoint_id_free_callback_failure_invalidates_prior_run(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        solver = MaterialReconstruction(**inputs(warp_device, iterations=1))
        solver.solve(objective_id="fixed-v1")
        previous = solver.checkpoint()
        original_fields = previous.fields.copy()
        sentinel = ValueError("accepted recorder failed")

        def callback(*_):
            raise sentinel

        with pytest.raises(ValueError) as raised:
            solver.solve(callback=callback)
        assert raised.value is sentinel
        assert not np.array_equal(solver.fractions_numpy(), original_fields)
        # A prior exported snapshot remains owned and valid after the new run.
        np.testing.assert_array_equal(previous.fields, original_fields)
        with pytest.raises(ContractError, match="accepted state"):
            solver.checkpoint("fixed-v1")


def test_fresh_checkpoint_declared_identity_preserves_interrupted_accepted_resume(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        supplied = inputs(warp_device, iterations=1, step_selection="bb")
        baseline = MaterialReconstruction(**supplied)
        baseline.solve(objective_id="fixed-v1")
        baseline.settings = replace(baseline.settings, iterations=2)
        expected = baseline.solve()
        solver = MaterialReconstruction(**supplied)
        solver.solve(objective_id="fixed-v1")
        solver.settings = replace(solver.settings, iterations=2)
        sentinel = ValueError("accepted recorder failed")

        def callback(*_):
            raise sentinel

        with pytest.raises(ValueError) as raised:
            solver.solve(callback=callback, objective_id="fixed-v1")
        assert raised.value is sentinel
        checkpoint = solver.checkpoint()
        assert checkpoint.state["accepted"] == 1
        np.testing.assert_allclose(
            checkpoint.state["accepted_loss"], solver.evaluate(), rtol=2e-6, atol=1e-8
        )
        resumed = MaterialReconstruction(**inputs(warp_device, iterations=2, step_selection="bb"))
        actual = resumed.solve(checkpoint=checkpoint, objective_id="fixed-v1")
        assert actual["accepted_steps"] == expected["accepted_steps"] == 2
        # An interrupted update has not refreshed its accepted gradient yet;
        # resume performs the same next evaluation as the uninterrupted run.
        assert actual["evaluations"] == expected["evaluations"]
        np.testing.assert_allclose(
            resumed.fractions_numpy(), baseline.fractions_numpy(), rtol=2e-6, atol=5e-7
        )
        np.testing.assert_allclose(
            [row["objective"] for row in actual["history"]],
            [row["objective"] for row in expected["history"]],
            rtol=2e-6,
            atol=1e-8,
        )


def test_fresh_checkpoint_successful_id_free_run_supports_explicit_export(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        solver = MaterialReconstruction(**inputs(warp_device, iterations=1))
        solver.solve(objective_id="fixed-v1")
        previous = solver.checkpoint()
        report = solver.solve()
        current = solver.checkpoint("fixed-v1")
        assert current.state["accepted_loss"] == report["final_objective"]
        assert current.state["evaluations"] == report["evaluations"]
        assert current.state["completed_iterations"] == 1
        np.testing.assert_array_equal(current.fields, solver.fractions_numpy())
        assert current.state["accepted_loss"] != previous.state["accepted_loss"]


@pytest.mark.parametrize("refusal", ["tolerance", "identity", "save_identity", "resume_identity"])
def test_fresh_checkpoint_refused_arguments_preserve_prior_snapshot(warp_device, refusal):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        solver = MaterialReconstruction(**inputs(warp_device, iterations=1))
        solver.solve(objective_id="fixed-v1")
        previous = solver.checkpoint()
        arguments: dict[str, Any] = {
            "tolerance": {"checkpoint_relative_tolerance": -1},
            "identity": {"objective_id": ""},
            "save_identity": {"save_checkpoint": lambda _: None},
            "resume_identity": {"checkpoint": previous, "objective_id": "wrong-model"},
        }[refusal]
        with pytest.raises(ContractError):
            solver.solve(**arguments)
        current = solver.checkpoint()
        assert current.objective_id == previous.objective_id
        assert current.state == previous.state
        np.testing.assert_array_equal(current.fields, previous.fields)
