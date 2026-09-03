"""Exact policy oracles isolate acceptance algebra from transport execution.

These are optimiser control tests, not simulated evidence of photon transport.
"""

import json
import math
from dataclasses import asdict

import pytest

from dpt.contracts import ContractError, NumericalError, TrialDomainError
from dpt.registration import Vector
from dpt.stochastic_models import Matrix
from dpt.stochastic_recovery import StochasticPolicy, recover_expected_signal
from dpt.transport.rng import HistoryBatch, require_independent


def close(
    actual: float | tuple[float, ...],
    expected: float | tuple[float, ...],
    *,
    rel: float = 1e-6,
    abs: float = 1e-12,
) -> bool:
    if isinstance(actual, tuple):
        return (
            isinstance(expected, tuple)
            and len(actual) == len(expected)
            and all(
                math.isclose(a, b, rel_tol=rel, abs_tol=abs)
                for a, b in zip(actual, expected, strict=True)
            )
        )
    return not isinstance(expected, tuple) and math.isclose(
        actual, expected, rel_tol=rel, abs_tol=abs
    )


class ExactQuadratic:
    deterministic_sampling = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, HistoryBatch, HistoryBatch]] = []

    def gradient_replicate(
        self,
        parameters: tuple[float, ...],
        mean_batch: HistoryBatch,
        derivative_batch: HistoryBatch,
    ) -> tuple[float, ...]:
        self.calls.append(("gradient", mean_batch, derivative_batch))
        return (parameters[0],)

    def change_replicate(
        self,
        before: tuple[float, ...],
        after: tuple[float, ...],
        first: HistoryBatch,
        second: HistoryBatch,
    ) -> float:
        self.calls.append(("change", first, second))
        return 0.5 * (after[0] ** 2 - before[0] ** 2)


def test_accepted_decrease_uses_disjoint_gradient_and_acceptance_pools() -> None:
    oracle = ExactQuadratic()
    result = recover_expected_signal(
        oracle,
        (2.0,),
        seed=91,
        policy=StochasticPolicy(iterations=3, replicates=4, initial_batch=2, maximum_batch=2),
    )
    assert result.parameters[0] < 2.0
    assert all(step.accepted and step.mean_change < 0 for step in result.steps)
    batches = [batch for _, first, second in oracle.calls for batch in (first, second)]
    require_independent(*batches)
    assert result.unique_histories == sum(batch.count for batch in batches)


def test_zero_observed_gradient_grows_sampling_instead_of_claiming_recovery() -> None:
    oracle = ExactQuadratic()
    oracle.deterministic_sampling = False
    result = recover_expected_signal(
        oracle,
        (0.0,),
        seed=17,
        policy=StochasticPolicy(replicates=4, initial_batch=2, maximum_batch=8),
    )
    assert result.reason == "sampling_unresolved"
    assert {a.count for _, a, _ in oracle.calls} == {2, 4, 8}
    assert result.parameters == (0.0,)
    assert result.steps == ()


def test_budget_failure_keeps_incumbent_without_using_partial_replicate_set() -> None:
    oracle = ExactQuadratic()
    result = recover_expected_signal(
        oracle,
        (2.0,),
        seed=9,
        policy=StochasticPolicy(
            replicates=4, initial_batch=4, maximum_batch=4, unique_history_budget=32
        ),
    )
    assert result.reason == "history_budget"
    assert result.parameters == (2.0,)
    assert all(kind == "gradient" for kind, _, _ in oracle.calls)


class InconclusiveChange(ExactQuadratic):
    deterministic_sampling = False

    def gradient_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> Vector:
        self.calls.append(("gradient", mean_batch, derivative_batch))
        return (parameters[0] + (1e-6 if len(self.calls) % 2 else -1e-6),)

    def change_replicate(
        self,
        before: tuple[float, ...],
        after: tuple[float, ...],
        first: HistoryBatch,
        second: HistoryBatch,
    ) -> float:
        self.calls.append(("change", first, second))
        # Alternating finite replicate outcomes whose band straddles acceptance.
        return -2.0 if len(self.calls) % 2 else 1.0


def test_uncertain_change_cannot_become_an_accepted_decrease() -> None:
    result = recover_expected_signal(
        InconclusiveChange(),
        (2.0,),
        seed=3,
        policy=StochasticPolicy(replicates=4, initial_batch=2, maximum_batch=8),
    )
    assert result.reason == "sampling_unresolved"
    assert result.parameters == (2.0,)
    assert result.steps == ()
    assert math.isfinite(result.parameters[0])


class LimitedDomain(ExactQuadratic):
    def change_replicate(
        self,
        before: tuple[float, ...],
        after: tuple[float, ...],
        first: HistoryBatch,
        second: HistoryBatch,
    ) -> float:
        if after[0] <= 0:
            raise TrialDomainError("positive chart")
        return super().change_replicate(before, after, first, second)


def test_only_classified_parameter_domain_failure_shrinks_the_trial() -> None:
    result = recover_expected_signal(
        LimitedDomain(),
        (0.1,),
        seed=3,
        policy=StochasticPolicy(
            iterations=3, initial_radius=1, replicates=4, initial_batch=2, maximum_batch=2
        ),
    )
    assert result.parameters == (0.1,)
    assert len(result.domain_rejections) == 3
    assert [trial.radius for trial in result.domain_rejections] == [1, 0.5, 0.25]

    class FailedEstimate(ExactQuadratic):
        def change_replicate(
            self,
            before: tuple[float, ...],
            after: tuple[float, ...],
            first: HistoryBatch,
            second: HistoryBatch,
        ) -> float:
            raise NumericalError("failed estimator")

    with pytest.raises(NumericalError, match="failed estimator"):
        recover_expected_signal(
            FailedEstimate(),
            (1.0,),
            seed=2,
            policy=StochasticPolicy(replicates=4, initial_batch=2, maximum_batch=2),
        )


def test_finite_components_with_unrepresentable_norm_are_rejected() -> None:
    class HugeGradient(ExactQuadratic):
        def gradient_replicate(
            self,
            parameters: tuple[float, ...],
            mean_batch: HistoryBatch,
            derivative_batch: HistoryBatch,
        ) -> tuple[float, ...]:
            return (1.7e308, 1.7e308)

    with pytest.raises(NumericalError, match="norm"):
        recover_expected_signal(
            HugeGradient(),
            (1.0, 1.0),
            seed=1,
            policy=StochasticPolicy(replicates=4, initial_batch=2, maximum_batch=2),
        )


def test_gradient_attempts_record_complete_independent_pools_and_serialize() -> None:
    oracle = ExactQuadratic()
    oracle.deterministic_sampling = False
    result = recover_expected_signal(
        oracle,
        (0.0,),
        seed=17,
        policy=StochasticPolicy(replicates=4, initial_batch=2, maximum_batch=8),
    )
    assert result.termination_detail == "zero_gradient_and_variance_at_maximum_batch"
    assert [a.batch_size for a in result.gradient_attempts] == [2, 4, 8]
    assert [a.first_history for a in result.gradient_attempts] == [0, 16, 48]
    assert [a.histories_used for a in result.gradient_attempts] == [16, 32, 64]
    assert result.unique_histories == 112
    assert all(
        a.parameters == (0.0,) and a.decision == "zero_sample" for a in result.gradient_attempts
    )
    restored = json.loads(json.dumps(asdict(result), allow_nan=False))
    assert restored["gradient_attempts"][-1]["gradient"] == [0.0]
    assert restored["gradient_attempts"][-1]["standard_error"] == [0.0]


def test_nonzero_unresolved_gradient_records_actual_uncertainty() -> None:
    class UncertainGradient(ExactQuadratic):
        deterministic_sampling = False

        def gradient_replicate(
            self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
        ) -> Vector:
            self.calls.append(("gradient", mean_batch, derivative_batch))
            return (-1.0 if len(self.calls) % 2 else 3.0,)

    result = recover_expected_signal(
        UncertainGradient(),
        (1.0,),
        seed=3,
        policy=StochasticPolicy(replicates=4, initial_batch=2, maximum_batch=4),
    )
    assert result.reason == "sampling_unresolved"
    assert result.termination_detail == "gradient_uncertainty_at_maximum_batch"
    assert result.parameters == (1.0,)
    assert result.steps == ()
    assert len(result.gradient_attempts) == 2
    for attempt in result.gradient_attempts:
        assert attempt.gradient == (1.0,)
        assert close(attempt.standard_error, (2 / math.sqrt(3),))
        assert attempt.gradient_norm == 1.0
        assert close(attempt.standard_error_norm, 2 / math.sqrt(3))
        assert attempt.decision == "relative_uncertainty"


def test_small_nonzero_gradient_records_convergence_band() -> None:
    result = recover_expected_signal(
        ExactQuadratic(),
        (1e-6,),
        seed=3,
        policy=StochasticPolicy(replicates=4, initial_batch=2, maximum_batch=2),
    )
    assert result.reason == "gradient_band"
    assert result.termination_detail == "proposal_gradient_band"
    assert result.gradient_attempts[0].decision == "gradient_band"
    assert result.gradient_attempts[0].gradient == (1e-6,)
    assert result.gradient_attempts[0].standard_error == (0.0,)
    assert result.steps == ()


def test_acceptance_uncertainty_is_distinguished_from_gradient_uncertainty() -> None:
    result = recover_expected_signal(
        InconclusiveChange(),
        (2.0,),
        seed=3,
        policy=StochasticPolicy(replicates=4, initial_batch=2, maximum_batch=8),
    )
    assert result.termination_detail == "acceptance_uncertainty_at_maximum_batch"
    assert len(result.gradient_attempts) == 1
    assert result.gradient_attempts[0].decision == "resolved"


class DenseQuadratic:
    """An exact algebraic controller fixture, explicitly classified by its author."""

    deterministic_sampling = True

    def __init__(self, matrix: Matrix) -> None:
        self.matrix = matrix
        self.calls: list[tuple[str, HistoryBatch, HistoryBatch]] = []

    def gradient(self, parameters: Vector) -> Vector:
        return tuple(
            math.fsum(a * b for a, b in zip(row, parameters, strict=True)) for row in self.matrix
        )

    def gradient_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> Vector:
        self.calls.append(("gradient", mean_batch, derivative_batch))
        return self.gradient(parameters)

    def model_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> tuple[Vector, Matrix]:
        self.calls.append(("model", mean_batch, derivative_batch))
        return self.gradient(parameters), self.matrix

    def change_replicate(
        self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
    ) -> float:
        self.calls.append(("change", first, second))
        return 0.5 * math.fsum(
            b * gb - a * ga
            for a, b, ga, gb in zip(
                before, after, self.gradient(before), self.gradient(after), strict=True
            )
        )


def quadratic_policy(**overrides: int | float | str | None) -> StochasticPolicy:
    # Named construction below keeps the helper's public options intentionally
    # constrained to fields exercised by these controller tests.
    values: dict[str, int | float | str | None] = {
        "proposal": "quadratic",
        "replicates": 4,
        "initial_batch": 2,
        "maximum_batch": 8,
        "initial_radius": 1.0,
        "maximum_radius": 4.0,
        "numerical_gradient_allowance": 1e-12,
    }
    values.update(overrides)
    return StochasticPolicy(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("size", [1, 2, 8])
def test_quadratic_controller_reaches_held_out_stationarity(size: int) -> None:
    matrix = tuple(
        tuple((i + 1.0 if i == j else 0.0) + 0.1 for j in range(size)) for i in range(size)
    )
    oracle = DenseQuadratic(matrix)
    result = recover_expected_signal(oracle, (0.2,) * size, seed=11, policy=quadratic_policy())
    assert result.reason == "gradient_band"
    assert result.termination_detail == "held_out_gradient_band"
    assert close(result.parameters, (0.0,) * size, abs=1e-12)
    assert len(result.steps) == 1
    assert not result.acceptance_attempts[0].boundary
    assert len(result.final_validation_attempts) == 1
    assert result.final_validation_attempts[0].pool == "final_validation"
    assert result.final_validation_attempts[0].batch_size == 2
    require_independent(*(batch for _, first, second in oracle.calls for batch in (first, second)))
    assert sum(value.reserved_histories for value in result.reservations) == result.unique_histories
    json.dumps(asdict(result), allow_nan=False)


def test_deterministic_numerical_allowance_does_not_come_from_empirical_variance() -> None:
    oracle = DenseQuadratic(((1.0,),))
    result = recover_expected_signal(oracle, (0.0,), seed=1, policy=quadratic_policy())
    assert result.reason == "gradient_band"
    assert result.sampling_classification == "verified_deterministic_expectation"
    assert result.gradient_attempts[0].numerical_allowance == 1e-12
    assert result.final_validation_attempts[0].band_method == "deterministic_numerical_allowance"
    # The same exact zeros with stochastic classification retain the rare-event
    # guard; sample variance is never a deterministic classification mechanism.
    oracle.deterministic_sampling = False
    stochastic = recover_expected_signal(oracle, (0.0,), seed=1, policy=quadratic_policy())
    assert stochastic.reason == "sampling_unresolved"
    assert [value.batch_size for value in stochastic.gradient_attempts] == [2, 4, 8]
    assert all(value.decision == "zero_sample" for value in stochastic.gradient_attempts)


def test_rare_zero_epoch_can_grow_to_a_real_hit_without_early_termination() -> None:
    class RareQuadratic(DenseQuadratic):
        deterministic_sampling = False

        def model_replicate(
            self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
        ) -> tuple[Vector, Matrix]:
            gradient, matrix = super().model_replicate(parameters, mean_batch, derivative_batch)
            noise = 1e-6 if len(self.calls) % 2 else -1e-6
            return ((0.0,) if mean_batch.count == 2 else (gradient[0] + noise,)), matrix

        def change_replicate(
            self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
        ) -> float:
            value = super().change_replicate(before, after, first, second)
            return value + (1e-6 if len(self.calls) % 2 else -1e-6)

    result = recover_expected_signal(
        RareQuadratic(((1.0,),)), (0.2,), seed=4, policy=quadratic_policy(iterations=1)
    )
    assert [a.decision for a in result.gradient_attempts] == ["zero_sample", "resolved"]
    assert len(result.steps) == 1
    # Exact zero in this stochastic fixture is deliberately still unresolved.
    assert result.termination_detail == "iteration_budget"
    assert result.final_validation_outcome == "not_certified"


def test_rejected_radii_cache_the_model_and_buy_independent_acceptance_sets() -> None:
    class Rejecting(DenseQuadratic):
        def change_replicate(
            self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
        ) -> float:
            self.calls.append(("change", first, second))
            return 1.0

    oracle = Rejecting(((1.0,),))
    result = recover_expected_signal(oracle, (2.0,), seed=7, policy=quadratic_policy(iterations=3))
    assert [kind for kind, _, _ in oracle.calls].count("model") == 4
    assert len(result.local_models) == 1
    assert [a.model_id for a in result.acceptance_attempts] == [0, 0, 0]
    assert [a.radius for a in result.acceptance_attempts] == [1.0, 0.5, 0.25]
    assert [a.first_history for a in result.acceptance_attempts] == [16, 32, 48]
    assert result.unique_histories == 80  # one model, three acceptance sets, final pool
    require_independent(*(batch for _, a, b in oracle.calls for batch in (a, b)))


def test_acceptance_growth_keeps_candidate_fixed_and_logs_every_look() -> None:
    result = recover_expected_signal(
        InconclusiveChange(),
        (2.0,),
        seed=3,
        policy=StochasticPolicy(replicates=4, initial_batch=2, maximum_batch=8),
    )
    assert [a.outcome for a in result.acceptance_attempts] == ["ambiguous"] * 3
    assert [a.look_index for a in result.acceptance_attempts] == [1, 2, 3]
    assert len({a.candidate for a in result.acceptance_attempts}) == 1
    assert [a.batch_size for a in result.acceptance_attempts] == [2, 4, 8]
    assert all(len(a.replicate_changes) == 4 for a in result.acceptance_attempts)
    assert sum(a.reserved_histories for a in result.reservations) == result.unique_histories


def test_failed_acceptance_reservation_records_required_work_without_identities() -> None:
    result = recover_expected_signal(
        ExactQuadratic(),
        (2.0,),
        seed=9,
        policy=StochasticPolicy(
            replicates=4, initial_batch=4, maximum_batch=4, unique_history_budget=32
        ),
    )
    attempt = result.acceptance_attempts[0]
    assert attempt.outcome == "budget_denied"
    assert attempt.required_histories == 32 and attempt.histories_used == 0
    reservation = result.reservations[-1]
    assert reservation.operation == "acceptance"
    assert reservation.pairs == () and reservation.reserved_histories == 0


def test_checkpoint_work_is_reserved_before_new_models() -> None:
    oracle = DenseQuadratic(((1.0,),))
    result = recover_expected_signal(
        oracle, (0.2,), seed=5, policy=quadratic_policy(unique_history_budget=15)
    )
    assert result.reason == "history_budget" and result.unique_histories == 0
    assert oracle.calls == []
    assert result.reservations[0].required_histories == 16
    assert result.reservations[0].checkpoint_histories == 32
    # Exactly enough for model, acceptance and the fixed final checkpoint.
    completed = recover_expected_signal(
        oracle, (0.2,), seed=5, policy=quadratic_policy(unique_history_budget=48)
    )
    assert completed.reason == "gradient_band" and completed.unique_histories == 48
    assert completed.reservations[-2].status == "budget_denied"
    assert completed.reservations[-1].operation == "final_validation"


def test_final_pool_is_predeclared_and_failed_validation_never_selects_another_point() -> None:
    class FailedFinal(DenseQuadratic):
        def gradient_replicate(
            self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
        ) -> Vector:
            self.calls.append(("gradient", mean_batch, derivative_batch))
            return (0.5,)

    oracle = FailedFinal(((1.0,),))
    result = recover_expected_signal(
        oracle, (0.2,), seed=5, policy=quadratic_policy(final_validation_batch=4)
    )
    assert result.reason == "sampling_unresolved"
    assert result.termination_detail == "final_validation_failed"
    assert close(result.parameters, (0.0,), abs=1e-15)
    assert len(result.steps) == 1 and len(result.final_validation_attempts) == 1
    assert result.final_validation_attempts[0].batch_size == 4
    assert [kind for kind, _, _ in oracle.calls][-2:] == ["gradient", "gradient"]
    assert result.final_validation_attempts[0].gradient == (0.5,)


def test_partial_domain_failure_distinguishes_reserved_and_completed_calls() -> None:
    class PartialDomain(DenseQuadratic):
        def change_replicate(
            self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
        ) -> float:
            if first.first_history >= 20:
                raise TrialDomainError("fixture domain")
            return super().change_replicate(before, after, first, second)

    result = recover_expected_signal(
        PartialDomain(((1.0,),)), (2.0,), seed=7, policy=quadratic_policy(iterations=1)
    )
    attempt = result.acceptance_attempts[0]
    reservation = result.reservations[1]
    assert attempt.outcome == "domain_error" and len(attempt.replicate_changes) == 1
    assert reservation.reserved_histories == 16
    assert reservation.completed_replicates == 1 and reservation.attempted_replicates == 2
    assert reservation.status == "domain_error"


def test_numerical_oracle_failure_keeps_failure_snapshot_on_exception() -> None:
    class Failed(DenseQuadratic):
        def change_replicate(
            self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
        ) -> float:
            raise NumericalError("failure, not domain rejection")

    with pytest.raises(NumericalError, match="failure, not domain") as caught:
        recover_expected_signal(Failed(((1.0,),)), (2.0,), seed=7, policy=quadratic_policy())
    snapshot = caught.value.__dict__["recovery_diagnostics"]
    assert snapshot.reservations[-1].status == "oracle_error"
    assert snapshot.reservations[-1].completed_replicates == 0
    assert snapshot.reservations[-1].attempted_replicates == 1
    assert snapshot.acceptance_attempts[-1].outcome == "oracle_error"
    assert snapshot.domain_rejections == ()


def test_small_gradient_does_not_hide_indefinite_model() -> None:
    with pytest.raises(NumericalError, match="indefinite"):
        recover_expected_signal(
            DenseQuadratic(((-1.0,),)), (0.0,), seed=5, policy=quadratic_policy()
        )


def test_sampling_flag_is_an_explicit_boolean_contract() -> None:
    oracle = DenseQuadratic(((1.0,),))
    oracle.__dict__["deterministic_sampling"] = "yes"
    with pytest.raises(ContractError, match="boolean"):
        recover_expected_signal(oracle, (0.0,), seed=1, policy=quadratic_policy())


def test_radius_growth_requires_boundary_and_good_independent_agreement() -> None:
    boundary = recover_expected_signal(
        DenseQuadratic(((1.0,),)), (2.0,), seed=1, policy=quadratic_policy()
    )
    assert [a.radius for a in boundary.acceptance_attempts] == [1.0, 2.0]
    assert all(close(a.agreement or 0.0, 1.0) for a in boundary.acceptance_attempts)

    class ConservativeModel(DenseQuadratic):
        def model_replicate(
            self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
        ) -> tuple[Vector, Matrix]:
            gradient, _ = super().model_replicate(parameters, mean_batch, derivative_batch)
            return gradient, ((2.0,),)

    interior = recover_expected_signal(
        ConservativeModel(((1.0,),)),
        (0.5,),
        seed=1,
        policy=quadratic_policy(iterations=3),
    )
    assert [a.radius for a in interior.acceptance_attempts] == [1.0] * 3
    assert all(not a.boundary and a.outcome == "accepted" for a in interior.acceptance_attempts)

    class PoorAgreement(DenseQuadratic):
        def change_replicate(
            self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
        ) -> float:
            return 0.2 * super().change_replicate(before, after, first, second)

    poor = recover_expected_signal(
        PoorAgreement(((1.0,),)), (3.0,), seed=1, policy=quadratic_policy(iterations=2)
    )
    assert [a.radius for a in poor.acceptance_attempts] == [1.0, 1.0]
    assert all(a.boundary and a.outcome == "accepted" for a in poor.acceptance_attempts)


def test_quadratic_contract_requires_bounded_model_interface() -> None:
    with pytest.raises(ContractError, match="model_replicate"):
        recover_expected_signal(ExactQuadratic(), (1.0,), seed=1, policy=quadratic_policy())
    with pytest.raises(ContractError, match="16"):
        recover_expected_signal(
            DenseQuadratic(((1.0,),)), (1.0,) * 17, seed=1, policy=quadratic_policy()
        )


def test_extreme_chart_records_realised_step_and_its_quadratic_prediction() -> None:
    class RoundedChart(DenseQuadratic):
        def gradient(self, parameters: Vector) -> Vector:
            return (0.3 + (parameters[0] - 1e15),)

        def change_replicate(
            self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
        ) -> float:
            self.calls.append(("change", first, second))
            actual = after[0] - before[0]
            return self.gradient(before)[0] * actual + 0.5 * actual**2

    result = recover_expected_signal(
        RoundedChart(((1.0,),)),
        (1e15,),
        seed=3,
        policy=quadratic_policy(iterations=1, initial_radius=0.3),
    )
    attempt = result.acceptance_attempts[0]
    assert attempt.step == (-0.25,)
    assert attempt.step[0] == attempt.candidate[0] - attempt.incumbent[0]
    assert close(attempt.predicted_decrease, 0.04375, abs=1e-15)
    assert close(attempt.agreement or 0.0, 1.0, abs=1e-15)
    assert not attempt.boundary
    assert attempt.proposal_diagnostics is not None
    assert attempt.proposal_diagnostics.step == attempt.step


def test_chart_rounding_cannot_silently_exceed_the_trust_radius() -> None:
    class RoundedOutward(DenseQuadratic):
        def gradient(self, parameters: Vector) -> Vector:
            return (4.0,)

    result = recover_expected_signal(
        RoundedOutward(((1.0,),)),
        (1e16,),
        seed=3,
        policy=quadratic_policy(initial_radius=3.0),
    )
    assert result.termination_detail == "chart_rounding_exceeds_trust_radius"
    assert result.final_validation_outcome == "clearly_nonstationary"
    assert result.final_validation_trigger == "chart_rounding_exceeds_trust_radius"
    assert result.parameters == (1e16,)
    assert result.acceptance_attempts == ()


@pytest.mark.parametrize("failure", ["nonfinite_replicate", "overflowing_band"])
def test_failed_acceptance_arithmetic_retains_attempt_and_reservation_snapshot(
    failure: str,
) -> None:
    class FailedArithmetic(DenseQuadratic):
        deterministic_sampling = False

        def model_replicate(
            self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
        ) -> tuple[Vector, Matrix]:
            gradient, matrix = super().model_replicate(parameters, mean_batch, derivative_batch)
            return (gradient[0] + (1e-6 if len(self.calls) % 2 else -1e-6),), matrix

        def change_replicate(
            self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
        ) -> float:
            self.calls.append(("change", first, second))
            if failure == "nonfinite_replicate":
                return float("nan")
            return -1.7e308 if first.first_history < 24 else 1.7e308

    with pytest.raises(NumericalError) as caught:
        recover_expected_signal(
            FailedArithmetic(((1.0,),)), (2.0,), seed=4, policy=quadratic_policy()
        )
    snapshot = caught.value.__dict__["recovery_diagnostics"]
    assert snapshot.acceptance_attempts[-1].outcome == "oracle_error"
    assert snapshot.acceptance_attempts[-1].histories_used == 16
    reservation = snapshot.reservations[-1]
    assert reservation.reserved_histories == 16
    expected_complete = 0 if failure == "nonfinite_replicate" else 4
    assert reservation.completed_replicates == expected_complete
    assert len(snapshot.acceptance_attempts[-1].replicate_changes) == expected_complete
    json.dumps(asdict(snapshot), allow_nan=False)


@pytest.mark.parametrize(
    "incumbent, expected",
    [(1e-6, "gradient_band"), (0.5, "sampling_unresolved"), (0.0, "sampling_unresolved")],
)
def test_acceptance_ceiling_spends_reserved_final_pool_once(
    incumbent: float, expected: str
) -> None:
    class NoisyDevelopment(DenseQuadratic):
        deterministic_sampling = False

        def model_replicate(
            self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
        ) -> tuple[Vector, Matrix]:
            self.calls.append(("model", mean_batch, derivative_batch))
            return (1.0 + (1e-6 if len(self.calls) % 2 else -1e-6),), ((1.0,),)

        def gradient_replicate(
            self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
        ) -> Vector:
            self.calls.append(("gradient", mean_batch, derivative_batch))
            factor = 1.0 + (1e-6 if len(self.calls) % 2 else -1e-6)
            return (parameters[0] * factor,)

        def change_replicate(
            self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
        ) -> float:
            self.calls.append(("change", first, second))
            return -3.0 if first.first_history < 24 else 1.0

    oracle = NoisyDevelopment(((1.0,),))
    result = recover_expected_signal(
        oracle,
        (incumbent,),
        seed=19,
        policy=quadratic_policy(maximum_batch=2, unique_history_budget=48),
    )
    assert result.reason == expected
    assert result.final_validation_trigger == "acceptance_uncertainty_at_maximum_batch"
    assert result.parameters == (incumbent,) and result.steps == ()
    assert len(result.final_validation_attempts) == 1
    assert result.final_validation_attempts[0].batch_size == 2
    assert result.final_validation_attempts[0].first_history == 32
    assert result.unique_histories == 48
    assert [kind for kind, _, _ in oracle.calls] == [
        *["model"] * 4,
        *["change"] * 4,
        *["gradient"] * 4,
    ]
    if incumbent == 0:
        assert result.final_validation_attempts[0].decision == "zero_sample"
    require_independent(*(batch for _, left, right in oracle.calls for batch in (left, right)))


def test_zero_development_and_final_samples_stay_unresolved() -> None:
    oracle = DenseQuadratic(((1.0,),))
    oracle.deterministic_sampling = False
    result = recover_expected_signal(
        oracle,
        (0.0,),
        seed=19,
        policy=quadratic_policy(maximum_batch=2),
    )
    assert result.reason == "sampling_unresolved"
    assert result.termination_detail == "zero_gradient_and_variance_at_maximum_batch"
    assert result.final_validation_outcome == "not_certified"
    assert result.final_validation_trigger == "zero_gradient_and_variance_at_maximum_batch"
    assert result.gradient_attempts[0].decision == "zero_sample"
    assert result.final_validation_attempts[0].decision == "zero_sample"
    assert result.steps == ()


def test_radius_limit_uses_fixed_final_pool_when_configured() -> None:
    class RejectingAtLimit(DenseQuadratic):
        def model_replicate(
            self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
        ) -> tuple[Vector, Matrix]:
            self.calls.append(("model", mean_batch, derivative_batch))
            return (1.0,), ((1.0,),)

        def change_replicate(
            self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
        ) -> float:
            self.calls.append(("change", first, second))
            return 1.0

    result = recover_expected_signal(
        RejectingAtLimit(((1.0,),)),
        (0.0,),
        seed=4,
        policy=quadratic_policy(initial_radius=0.1, minimum_radius=0.1),
    )
    assert result.reason == "gradient_band" and result.final_validation_trigger == "radius_limit"
    assert len(result.steps) == 1 and not result.steps[0].accepted
    assert len(result.final_validation_attempts) == 1
