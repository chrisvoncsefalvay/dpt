"""Stochastic model refresh and distinct rejected candidates."""

from dpt.registration import Vector
from dpt.stochastic_models import Matrix
from dpt.stochastic_recovery import StochasticPolicy, recover_expected_signal
from dpt.transport.rng import HistoryBatch


class GrowingModel:
    deterministic_sampling = False

    def __init__(self) -> None:
        self.model_batches: list[int] = []

    def gradient_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> Vector:
        first = mean_batch
        variation = 1e-6 if first.first_history // (2 * first.count) % 2 else -1e-6
        return (parameters[0] + variation,)

    def model_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> tuple[Vector, Matrix]:
        first = mean_batch
        self.model_batches.append(first.count)
        scale = 2.0 if first.count == 2 else 1.0
        gradient = self.gradient_replicate(parameters, mean_batch, derivative_batch)
        # Both models predict the same Newton candidate, so looks must retain
        # their cumulative allocation even when the model identifier changes.
        return (scale * gradient[0],), ((scale,),)

    def change_replicate(
        self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
    ) -> float:
        odd = first.first_history // (2 * first.count) % 2
        if first.count == 2:
            return -0.2 + (1.0 if odd else -1.0)
        return 0.5 * (after[0] ** 2 - before[0] ** 2) + (1e-6 if odd else -1e-6)


def test_acceptance_growth_refreshes_stochastic_model_before_next_candidate_test() -> None:
    oracle = GrowingModel()
    result = recover_expected_signal(
        oracle,
        (1.0,),
        seed=17,
        policy=StochasticPolicy(
            proposal="quadratic",
            replicates=4,
            initial_batch=2,
            maximum_batch=4,
            initial_radius=2.0,
            maximum_radius=2.0,
            iterations=2,
        ),
    )
    assert oracle.model_batches == [2] * 4 + [4] * 4
    assert [model.batch_size for model in result.local_models] == [2, 4]
    assert [attempt.model_id for attempt in result.acceptance_attempts] == [0, 1]
    assert [attempt.look_index for attempt in result.acceptance_attempts] == [1, 2]
    assert result.reason == "gradient_band"
    assert abs(result.parameters[0]) < 1e-6


def test_rejected_interior_proposal_does_not_retest_same_candidate() -> None:
    class InteriorRejections:
        deterministic_sampling = True

        def gradient_replicate(
            self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
        ) -> Vector:
            return (1.0,)

        def model_replicate(
            self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
        ) -> tuple[Vector, Matrix]:
            return (1.0,), ((1000.0,),)

        def change_replicate(
            self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
        ) -> float:
            return 1.0

    result = recover_expected_signal(
        InteriorRejections(),
        (2.0,),
        seed=17,
        policy=StochasticPolicy(
            proposal="quadratic",
            replicates=4,
            initial_batch=2,
            maximum_batch=2,
            initial_radius=0.1,
            iterations=4,
        ),
    )
    candidates = [attempt.candidate for attempt in result.acceptance_attempts]
    assert len(candidates) == 4
    assert len(set(candidates)) == 4
    assert result.acceptance_attempts[1].radius < 0.001
