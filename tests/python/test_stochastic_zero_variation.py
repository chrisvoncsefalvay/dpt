"""Conservative handling of stochastic replicate sets with no observed variation."""

import random

import pytest

from dpt.registration import Vector
from dpt.stochastic_recovery import (
    StochasticPolicy,
    acceptance_decision,
    recover_expected_signal,
)
from dpt.transport.rng import HistoryBatch


@pytest.mark.parametrize("change", [-0.2, 0.0, 0.2])
def test_identical_stochastic_changes_cannot_be_accepted(change: float) -> None:
    decision = acceptance_decision((change,) * 8, -0.01, alpha=0.05, maximum_looks=11)
    assert decision.standard_error == 0.0
    assert decision.outcome == ("ambiguous" if change < -0.01 else "rejected")


def test_explicit_deterministic_classification_allows_identical_decrease() -> None:
    decision = acceptance_decision(
        (-0.2,) * 8, -0.01, alpha=0.05, maximum_looks=11, deterministic=True
    )
    assert decision.outcome == "accepted"


def test_sparse_objective_increase_is_not_accepted_from_zero_hit_replicates() -> None:
    # E[change] = 0.1*2.3 + 0.9*(-0.2) = +0.05: the common
    # primary-only decrement is misleading when all rare contributions are absent.
    rng = random.Random(713)
    zero_hit_sets = 0
    for _ in range(400):
        changes = tuple(2.3 if rng.random() < 0.1 else -0.2 for _ in range(8))
        decision = acceptance_decision(changes, -0.01, alpha=0.05, maximum_looks=11)
        if len(set(changes)) == 1 and changes[0] < 0:
            zero_hit_sets += 1
            assert decision.outcome == "ambiguous"
        assert decision.outcome != "accepted"
    assert zero_hit_sets > 100


@pytest.mark.parametrize("gradient", [0.0, 1e-6, 1.0])
def test_identical_stochastic_gradients_grow_batches_without_proposal(gradient: float) -> None:
    class UnobservedRareEvents:
        deterministic_sampling = False

        def gradient_replicate(
            self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
        ) -> Vector:
            return (gradient,)

        def change_replicate(
            self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
        ) -> float:
            raise AssertionError("unresolved gradients must not produce an acceptance trial")

    result = recover_expected_signal(
        UnobservedRareEvents(),
        (1.0,),
        seed=9,
        policy=StochasticPolicy(initial_batch=2, maximum_batch=8, replicates=4),
    )
    assert result.reason == "sampling_unresolved"
    assert [attempt.batch_size for attempt in result.gradient_attempts] == [2, 4, 8]
    assert {attempt.decision for attempt in result.gradient_attempts} == {
        "zero_sample" if gradient == 0 else "zero_variance"
    }
    assert result.steps == ()
    assert result.acceptance_attempts == ()
