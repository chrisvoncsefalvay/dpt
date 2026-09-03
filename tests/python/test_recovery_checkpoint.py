"""Immutable-chart checkpoint continuation preserves safeguarded L-BFGS state."""

from dataclasses import replace
from typing import Any

import pytest

from dpt.contracts import ContractError
from dpt.registration import Evaluation, RecoveryPolicy, recover_parameters


def oracle(point):
    a, b, c = (x - y for x, y in zip(point, (1.0, -0.3, 2.0), strict=True))
    return Evaluation(
        0.5 * (80 * a * a + 4 * b * b + c * c + 2 * a * b), (80 * a + b, 4 * b + a, c)
    )


def policy():
    return RecoveryPolicy(
        max_iterations=30,
        max_evaluations=500,
        gradient_tolerance=1e-10,
        relative_loss_tolerance=0,
        step_tolerance=0,
    )


def test_resume_verifies_oracle_and_preserves_curvature_trajectory():
    selected = policy()
    full = recover_parameters(oracle, (3.0, 2.0, -1.0), policy=selected)
    snapshots = []
    first = recover_parameters(
        oracle,
        (3.0, 2.0, -1.0),
        policy=replace(selected, max_iterations=3),
        objective_id="fixture-v1/fixed-three-coordinate-chart",
        save_checkpoint=snapshots.append,
    )
    assert first.reason == "iteration_budget"
    assert first.checkpoint is not None
    assert first.checkpoint.curvature_pairs
    assert len(snapshots) == 3
    resumed = recover_parameters(
        oracle,
        policy=selected,
        objective_id=first.checkpoint.objective_id,
        checkpoint=first.checkpoint,
    )
    assert resumed.parameters == full.parameters
    assert resumed.evaluation == full.evaluation
    assert resumed.reason == full.reason == "gradient_tolerance"
    assert [record.loss for record in resumed.history] == [record.loss for record in full.history]
    assert resumed.evaluations == full.evaluations + 1  # Fresh accepted-point verification.
    assert snapshots[0].history[-1].iteration == 1  # Prior snapshots remain immutable.


def test_resume_refuses_changed_identity_policy_initial_or_oracle():
    selected = policy()
    first = recover_parameters(
        oracle, (3.0, 2.0, -1.0), policy=replace(selected, max_iterations=2), objective_id="v1"
    )
    assert first.checkpoint is not None
    args: dict[str, Any] = dict(policy=selected, objective_id="v1", checkpoint=first.checkpoint)
    with pytest.raises(ContractError, match="identity"):
        recover_parameters(oracle, **{**args, "objective_id": "v2"})
    with pytest.raises(ContractError, match="policy"):
        recover_parameters(oracle, **{**args, "policy": replace(selected, memory=9)})
    with pytest.raises(ContractError, match="initial"):
        recover_parameters(oracle, (0.0, 0.0, 0.0), **args)
    with pytest.raises(ContractError, match="oracle"):
        recover_parameters(
            lambda point: Evaluation(oracle(point).loss + 1, oracle(point).gradient), **args
        )
    with pytest.raises(ContractError, match="max_evaluations"):
        recover_parameters(
            oracle, **{**args, "policy": replace(selected, max_evaluations=first.evaluations)}
        )


def test_checkpoint_recording_requires_a_declared_identity():
    with pytest.raises(ContractError, match="objective_id"):
        recover_parameters(oracle, (0.0, 0.0, 0.0), save_checkpoint=lambda _: None)
