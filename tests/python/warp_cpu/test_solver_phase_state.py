"""Run state remains owned across real resident solves and observer failures."""

import math
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.material_reconstruction import MaterialReconstruction
from dpt.solver_lifecycle import SolverEvent
from dpt.stochastic_recovery import StochasticPolicy, recover_expected_signal
from helpers import material_inputs
from helpers.transport_inputs import absorbing_oracle


def _without_time(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_time(item) for key, item in value.items() if key != "wall_seconds"}
    if isinstance(value, list):
        return [_without_time(item) for item in value]
    return value


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("acceleration", ["none", "inertial"])
@pytest.mark.parametrize("step_selection", ["geometric", "bb"])
def test_interleaved_material_instances_keep_reports_fields_and_checkpoint_state(
    cache: bool, acceleration: str, step_selection: str, warp_device: str
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        supplied = material_inputs._inputs("float64")
        supplied["device"] = warp_device
        supplied["settings"] = replace(
            supplied["settings"],
            iterations=3,
            cache_accepted_forward=cache,
            acceleration=acceleration,
            step_selection=step_selection,
        )
        expected = MaterialReconstruction(**supplied)
        expected_report = expected.solve(objective_id="interleaved-v1")
        primary, secondary = MaterialReconstruction(**supplied), MaterialReconstruction(**supplied)
        secondary_reports: list[dict[str, Any]] = []

        def accepted(_iteration: int, _record: dict[str, Any]) -> None:
            if not secondary_reports:
                secondary_reports.append(secondary.solve(objective_id="interleaved-v1"))

        actual = primary.solve(callback=accepted, objective_id="interleaved-v1")
        assert len(secondary_reports) == 1
        for solver, report in ((primary, actual), (secondary, secondary_reports[0])):
            if warp_device == "cpu":
                assert _without_time(report) == _without_time(expected_report)
            else:
                assert report["termination"] == expected_report["termination"]
                assert report["accepted_steps"] == expected_report["accepted_steps"]
                assert report["evaluations"] == expected_report["evaluations"]
            np.testing.assert_allclose(
                solver.fractions_numpy(), expected.fractions_numpy(), rtol=2e-6, atol=5e-7
            )
            independent = material_inputs._independent_objective(supplied, solver.fractions_numpy())
            assert report["final_objective"] == pytest.approx(independent, rel=2e-5, abs=1e-6)
            checkpoint = solver.checkpoint()
            assert checkpoint.state["accepted"] == report["accepted_steps"]
            assert checkpoint.state["evaluations"] == report["evaluations"]
            assert checkpoint.state["accepted_loss"] == report["final_objective"]
            np.testing.assert_array_equal(checkpoint.fields, solver.fractions_numpy())


@pytest.mark.parametrize("proposal", ["linear", "quadratic"])
def test_real_transport_fresh_run_after_observer_failure_has_independent_controller_state(
    proposal: Literal["linear", "quadratic"], warp_device: str
) -> None:
    policy = StochasticPolicy(
        iterations=40,
        replicates=4,
        initial_batch=16,
        maximum_batch=16,
        unique_history_budget=100_000,
        initial_radius=0.2,
        gradient_tolerance=1e-5,
        numerical_gradient_allowance=1e-13,
        final_validation_batch=16,
        proposal=proposal,
    )
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        oracle = absorbing_oracle(
            "continuous-absorption", 16, warp_device, local_model=proposal == "quadratic"
        )
        reference = absorbing_oracle(
            "continuous-absorption", 16, warp_device, local_model=proposal == "quadratic"
        )
        events: list[SolverEvent] = []
        sentinel = ValueError("accepted observer failure")

        def observer(event: SolverEvent) -> None:
            events.append(event)
            if event.status == "accepted":
                raise sentinel

        with pytest.raises(ValueError) as raised:
            recover_expected_signal(
                oracle, (0.0,), seed=9051, policy=policy, observe_lifecycle=observer
            )
        assert raised.value is sentinel
        assert [event.status for event in events].count("accepted") == 1
        snapshot = raised.value.__dict__["solver_lifecycle"]
        assert snapshot.accepted_steps == 1
        assert snapshot.objective is None
        actual = recover_expected_signal(oracle, (0.0,), seed=9051, policy=policy)
        expected = recover_expected_signal(reference, (0.0,), seed=9051, policy=policy)
        assert actual == expected
        # The supplied observation is exp(-3/2) through an independently known unit chord.
        assert actual.parameters[0] == pytest.approx(math.log(1.5), abs=3e-4)
        assert actual.reservations[0].first_history == 0
        assert actual.final_validation_outcome == "certified"
        assert all(
            model.parameters in ((0.0,), *(step.candidate for step in actual.acceptance_attempts))
            for model in actual.local_models
        )
