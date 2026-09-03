"""Independent-batch squared-expected-signal optimisation in a fixed chart.

The per-proposal acceptance band uses Student-t calibration for independent
Gaussian replicate means. Unresolved or identical stochastic replicates require
batch growth or exhaustion. General nonlinear losses need different estimators
and are outside this interface's contract.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Literal, Protocol, TypeVar, cast

from dpt.contracts import ContractError, NumericalError, TrialDomainError, finite_scalar, integer
from dpt.registration import Vector
from dpt.solver_lifecycle import LifecycleObserver, LifecycleRecorder
from dpt.statistics import mean_standard_error, predicted_batch_size, student_t_critical
from dpt.stochastic_models import (
    Matrix,
    QuadraticProposal,
    quadratic_proposal,
    quadratic_reduction,
    validate_curvature,
)
from dpt.transport.rng import HistoryBatch

StochasticStopReason = Literal[
    "gradient_band", "sampling_unresolved", "history_budget", "radius_limit", "iteration_budget"
]
FinalValidationOutcome = Literal["certified", "not_certified", "clearly_nonstationary"]
HistoryPairs = tuple[tuple[HistoryBatch, HistoryBatch], ...]


class IndependentSquaredOracle(Protocol):
    """Original histories are IID within each caller-specified batch.

    Source sampling must be keyed by the provided history identity too. The two
    independent batches supply distinct mean and derivative estimates. Replaying
    an original history to obtain its derivative does not create a new sample.
    """

    def gradient_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> Vector:
        """Return (estimated mean - observed) times an independent mean derivative."""
        ...

    def change_replicate(
        self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
    ) -> float:
        """Return an unbiased squared-expected-signal loss difference.

        Each pose uses the product of independent residual-mean estimates.
        Reuse the specified streams between poses for common random numbers;
        batches first and second must stay independent within either pose.
        """
        ...


class QuadraticSquaredOracle(IndependentSquaredOracle, Protocol):
    def model_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> tuple[Vector, Matrix]:
        """Return independent-product gradient and PSD estimated-Jacobian metric."""
        ...


class BatchedSquaredOracle(IndependentSquaredOracle, Protocol):
    """Optional batched checkpoints; exactly one result per independent pair.

    A backend may fuse launches and readbacks. The controller preserves identity
    reservations and uses scalar operations when a plural operation is absent.
    Every pair is an independent sampling unit; outputs must retain that order.
    """

    def gradient_replicates(
        self, parameters: Vector, pairs: tuple[tuple[HistoryBatch, HistoryBatch], ...]
    ) -> tuple[Vector, ...]: ...

    def change_replicates(
        self, before: Vector, after: Vector, pairs: tuple[tuple[HistoryBatch, HistoryBatch], ...]
    ) -> tuple[float, ...]: ...


class BatchedQuadraticSquaredOracle(QuadraticSquaredOracle, BatchedSquaredOracle, Protocol):
    def model_replicates(
        self, parameters: Vector, pairs: HistoryPairs
    ) -> tuple[tuple[Vector, Matrix], ...]: ...


class _TraceAccountingOracle(Protocol):
    histories_traced: int


@dataclass(frozen=True, slots=True)
class StochasticPolicy:
    """Explicit controller and statistical policies; legacy defaults are retained.

    ``auto`` chooses a supported quadratic oracle, otherwise linear. Optional
    linear caching and agreement growth avoid re-estimating rejected incumbents.
    Predicted batch growth assumes an inverse-square-root error law only for
    allocation; every fresh decision measures its own error. Acceptance controls
    either a proposal's predeclared looks or the bounded run using Bonferroni.
    Student stationarity uses simultaneous marginal Gaussian-replicate intervals
    and an independent final checkpoint. A predeclared final schedule splits its
    alpha across those looks; failed looks never feed another proposal. None of
    these Gaussian assumptions is established from empirical skew or variance.
    """

    iterations: int = 100
    replicates: int = 8
    initial_batch: int = 1024
    maximum_batch: int = 1_048_576
    unique_history_budget: int = 100_000_000
    initial_radius: float = 0.1
    minimum_radius: float = 1e-7
    maximum_radius: float = 1.0
    acceptance_alpha: float = 0.05
    acceptance_fraction: float = 0.1
    standard_error_multiplier: float = 2.0
    gradient_tolerance: float = 1e-5
    relative_gradient_uncertainty: float = 0.5
    proposal: Literal["linear", "quadratic", "auto"] = "linear"
    numerical_gradient_allowance: float = 0.0
    final_validation_batch: int | None = None
    final_validation_batches: tuple[int, ...] | None = None
    damping_relative: float = 1e-12
    rank_tolerance: float = 1e-10
    radius_growth_agreement: float = 0.75
    batch_growth: Literal["doubling", "predicted"] = "doubling"
    acceptance_maximum_looks: int | None = None
    acceptance_error_scope: Literal["proposal", "run"] = "proposal"
    stationarity_calibration: Literal["heuristic", "student_t"] = "heuristic"
    stationarity_alpha: float = 0.05
    cache_linear_gradient: bool = False
    linear_radius_growth: Literal["always", "agreement"] = "always"

    def __post_init__(self) -> None:
        for name in ("iterations", "initial_batch", "maximum_batch", "unique_history_budget"):
            integer(getattr(self, name), name, minimum=1, maximum=2**63 - 1)
        integer(self.replicates, "replicates", minimum=4)
        if self.initial_batch > self.maximum_batch or self.maximum_batch >= 2**31:
            raise ContractError("batch sizes must be ordered positive signed-32-bit counts")
        for name in (
            "initial_radius",
            "minimum_radius",
            "maximum_radius",
            "acceptance_fraction",
            "standard_error_multiplier",
            "gradient_tolerance",
            "relative_gradient_uncertainty",
        ):
            if finite_scalar(getattr(self, name), name, minimum=0.0) == 0:
                raise ContractError(f"{name} must be positive")
        if not self.minimum_radius <= self.initial_radius <= self.maximum_radius:
            raise ContractError("trust radii must satisfy minimum <= initial <= maximum")
        if not 0 < self.acceptance_alpha < 0.5:
            raise ContractError("acceptance_alpha must lie in (0, .5)")
        if not 0 < self.acceptance_fraction < 1:
            raise ContractError("acceptance fraction must lie in (0,1)")
        if self.proposal not in ("linear", "quadratic", "auto"):
            raise ContractError("proposal must be linear, quadratic or auto")
        if self.batch_growth not in ("doubling", "predicted"):
            raise ContractError("batch growth must be doubling or predicted")
        if self.acceptance_error_scope not in ("proposal", "run"):
            raise ContractError("acceptance error scope must be proposal or run")
        if self.stationarity_calibration not in ("heuristic", "student_t"):
            raise ContractError("stationarity calibration must be heuristic or student_t")
        if not 0 < finite_scalar(self.stationarity_alpha, "stationarity alpha") < 0.5:
            raise ContractError("stationarity alpha must lie in (0,.5)")
        if self.acceptance_maximum_looks is not None:
            integer(self.acceptance_maximum_looks, "acceptance maximum looks", minimum=1)
        if type(self.cache_linear_gradient) is not bool:
            raise ContractError("cache_linear_gradient must be a boolean")
        if self.linear_radius_growth not in ("always", "agreement"):
            raise ContractError("linear radius growth must be always or agreement")
        finite_scalar(self.numerical_gradient_allowance, "numerical_gradient_allowance", minimum=0)
        finite_scalar(self.damping_relative, "damping_relative", minimum=0)
        finite_scalar(self.rank_tolerance, "rank_tolerance", minimum=0)
        finite_scalar(self.radius_growth_agreement, "radius_growth_agreement", minimum=0)
        if not 0 < self.rank_tolerance < 1 or not 0 < self.radius_growth_agreement <= 1:
            raise ContractError("rank tolerance and radius agreement must lie in (0,1] (rank < 1)")
        if self.final_validation_batch is not None:
            integer(self.final_validation_batch, "final_validation_batch", minimum=1)
            if self.final_validation_batch > self.maximum_batch:
                raise ContractError("final validation batch exceeds prepared batch capacity")
        if self.final_validation_batches is not None:
            if self.final_validation_batch is not None:
                raise ContractError("choose a single final batch or a predeclared final schedule")
            if (
                not isinstance(cast(object, self.final_validation_batches), tuple)
                or not self.final_validation_batches
            ):
                raise ContractError("final validation schedule must be a nonempty immutable tuple")
            if self.stationarity_calibration != "student_t":
                raise ContractError(
                    "repeated final looks require Student-t stationarity calibration"
                )
            previous = 0
            for size in self.final_validation_batches:
                integer(size, "final validation batch", minimum=previous + 1)
                if size > self.maximum_batch:
                    raise ContractError("final validation schedule exceeds prepared batch capacity")
                previous = size


@dataclass(frozen=True, slots=True)
class StochasticStep:
    iteration: int
    accepted: bool
    mean_change: float
    change_standard_error: float
    predicted_decrease: float
    radius: float
    batch_size: int
    first_history: int
    histories_used: int


@dataclass(frozen=True, slots=True)
class GradientAttempt:
    """Host statistics for one complete set of independent gradient replicates."""

    iteration: int
    parameters: Vector
    batch_size: int
    first_history: int
    histories_used: int
    gradient: Vector
    standard_error: Vector
    gradient_norm: float
    standard_error_norm: float
    decision: Literal[
        "zero_sample", "zero_variance", "gradient_band", "resolved", "relative_uncertainty"
    ]
    model_id: int | None = None
    pool: str = "proposal"
    band_method: str = "heuristic_euclidean_marginal_se"
    numerical_allowance: float = 0.0
    replicate_gradients: tuple[Vector, ...] = ()
    stationarity_upper_bound: float | None = None
    confidence_alpha: float | None = None


@dataclass(frozen=True, slots=True)
class WorkReservation:
    """Random identities reserved, distinct from completed oracle calls/work.

    A failed call may have executed partial CUDA work. Only the operator's work
    counters can report that; completed_replicates does not infer replay costs.
    No identity from a failed reservation is reused.
    """

    operation: str
    iteration: int
    first_history: int
    required_histories: int
    reserved_histories: int
    checkpoint_histories: int
    pairs: tuple[tuple[HistoryBatch, HistoryBatch], ...]
    status: str
    attempted_replicates: int = 0
    completed_replicates: int = 0
    execution: str = "scalar"


@dataclass(frozen=True, slots=True)
class LocalModel:
    model_id: int
    parameters: Vector
    gradient: Vector
    curvature: Matrix
    first_history: int
    histories_used: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class AcceptanceAttempt:
    iteration: int
    incumbent: Vector
    candidate: Vector
    model_id: int | None
    step: Vector
    predicted_decrease: float
    radius: float
    batch_size: int
    first_history: int
    histories_used: int
    required_histories: int
    replicate_changes: tuple[float, ...]
    mean_change: float | None
    standard_error: float | None
    band_method: str
    look_index: int
    threshold: float
    outcome: str
    agreement: float | None
    boundary: bool
    proposal_diagnostics: QuadraticProposal | None = None
    confidence_alpha: float | None = None
    error_scope: str = "proposal"


@dataclass(frozen=True, slots=True)
class StochasticRecoveryResult:
    parameters: Vector
    reason: StochasticStopReason
    unique_histories: int
    steps: tuple[StochasticStep, ...]
    domain_rejections: tuple[DomainRejection, ...] = ()
    gradient_attempts: tuple[GradientAttempt, ...] = ()
    termination_detail: str = ""
    diagnostic_version: int = 4
    acceptance_attempts: tuple[AcceptanceAttempt, ...] = ()
    reservations: tuple[WorkReservation, ...] = ()
    local_models: tuple[LocalModel, ...] = ()
    final_validation_attempts: tuple[GradientAttempt, ...] = ()
    sampling_classification: str = "stochastic"
    proposal_policy: str = "linear"
    uncertainty_policy: str = "heuristic_euclidean_marginal_se"
    numerical_gradient_allowance: float = 0.0
    final_validation_batch: int | None = None
    final_validation_trigger: str | None = None
    final_validation_outcome: FinalValidationOutcome | None = None
    histories_traced: int | None = None
    acceptance_error_scope: str = "proposal"


@dataclass(frozen=True, slots=True)
class DomainRejection:
    iteration: int
    radius: float
    message: str


_Replicate = TypeVar("_Replicate")


class _SamplingLedger:
    """Reserve IID identities and execute complete scalar or batched checkpoints."""

    def __init__(self, seed: int, policy: StochasticPolicy) -> None:
        self.seed, self.policy = seed, policy
        self.first_history = 0
        self.reservations: list[WorkReservation] = []

    def reserve(self, size: int, operation: str, iteration: int, holdback: int = 0) -> int | None:
        required = 2 * size * self.policy.replicates
        start = self.first_history
        if start + required + holdback > self.policy.unique_history_budget:
            self.reservations.append(
                WorkReservation(
                    operation=operation,
                    iteration=iteration,
                    first_history=start,
                    required_histories=required,
                    reserved_histories=0,
                    checkpoint_histories=holdback,
                    pairs=(),
                    status="budget_denied",
                )
            )
            return None
        pairs: list[tuple[HistoryBatch, HistoryBatch]] = []
        for _ in range(self.policy.replicates):
            left = HistoryBatch(
                self.seed, self.first_history, size, f"inverse-source-{self.first_history}"
            )
            self.first_history += size
            right = HistoryBatch(
                self.seed, self.first_history, size, f"inverse-source-{self.first_history}"
            )
            self.first_history += size
            pairs.append((left, right))
        self.reservations.append(
            WorkReservation(
                operation=operation,
                iteration=iteration,
                first_history=start,
                required_histories=required,
                reserved_histories=required,
                checkpoint_histories=holdback,
                pairs=tuple(pairs),
                status="reserved",
            )
        )
        return len(self.reservations) - 1

    def execute(
        self,
        index: int,
        operation: Callable[[HistoryBatch, HistoryBatch], _Replicate],
        *,
        batched: Callable[[tuple[tuple[HistoryBatch, HistoryBatch], ...]], tuple[_Replicate, ...]]
        | None = None,
    ) -> tuple[_Replicate, ...]:
        outputs: list[_Replicate] = []
        if batched is not None:
            self.reservations[index] = replace(
                self.reservations[index],
                execution="batched",
                attempted_replicates=len(self.reservations[index].pairs),
            )
            try:
                values = tuple(batched(self.reservations[index].pairs))
                if len(values) != len(self.reservations[index].pairs):
                    raise ContractError(
                        "batched oracle must return exactly one result per reserved pair"
                    )
            except Exception as error:
                self._failed(index, error, 0)
                raise
            self.reservations[index] = replace(
                self.reservations[index], status="complete", completed_replicates=len(values)
            )
            return values
        for left, right in self.reservations[index].pairs:
            self.reservations[index] = replace(
                self.reservations[index], attempted_replicates=len(outputs) + 1
            )
            try:
                outputs.append(operation(left, right))
            except Exception as error:
                self._failed(index, error, len(outputs))
                raise
            self.reservations[index] = replace(
                self.reservations[index], completed_replicates=len(outputs)
            )
        self.reservations[index] = replace(self.reservations[index], status="complete")
        return tuple(outputs)

    def _failed(self, index: int, error: Exception, completed: int) -> None:
        self.reservations[index] = replace(
            self.reservations[index],
            status="domain_error" if isinstance(error, TrialDomainError) else "oracle_error",
            completed_replicates=completed,
        )


class _ProposalCache:
    """Typed incumbent model state, invalidated only when precision or pose changes."""

    def __init__(self, deterministic: bool) -> None:
        self.deterministic: bool = deterministic
        self.models: list[LocalModel] = []
        self.linear: GradientAttempt | None = None

    @property
    def gradient(self) -> Vector:
        if self.models:
            return self.models[0].gradient
        if self.linear is None:
            raise ContractError("proposal gradient cache is empty")
        return self.linear.gradient


@dataclass(slots=True)
class _StochasticReportState:
    """Per-call report histories; numerical controller aliases retain their lists."""

    ledger: _SamplingLedger
    reservations: list[WorkReservation]
    selected: StochasticPolicy
    oracle: IndependentSquaredOracle
    deterministic: bool
    quadratic: bool
    band_method: str
    final_size: int | None
    traced_start: object
    steps: list[StochasticStep]
    domain_rejections: list[DomainRejection]
    gradient_attempts: list[GradientAttempt]
    acceptance_attempts: list[AcceptanceAttempt]
    local_models: list[LocalModel]
    validation_attempts: list[GradientAttempt]
    validation_trigger: str | None
    validation_outcome: FinalValidationOutcome | None

    def result(
        self, parameters: Vector, reason: StochasticStopReason, detail: str = ""
    ) -> StochasticRecoveryResult:
        return StochasticRecoveryResult(
            parameters=parameters,
            reason=reason,
            unique_histories=self.ledger.first_history,
            steps=tuple(self.steps),
            domain_rejections=tuple(self.domain_rejections),
            gradient_attempts=tuple(self.gradient_attempts),
            termination_detail=detail or reason,
            acceptance_attempts=tuple(self.acceptance_attempts),
            reservations=tuple(self.reservations),
            local_models=tuple(self.local_models),
            final_validation_attempts=tuple(self.validation_attempts),
            sampling_classification="verified_deterministic_expectation"
            if self.deterministic
            else "stochastic",
            proposal_policy="quadratic" if self.quadratic else "linear",
            uncertainty_policy=self.band_method,
            numerical_gradient_allowance=self.selected.numerical_gradient_allowance,
            final_validation_batch=self.final_size,
            final_validation_trigger=self.validation_trigger,
            final_validation_outcome=self.validation_outcome,
            histories_traced=cast(_TraceAccountingOracle, self.oracle).histories_traced
            - self.traced_start
            if type(self.traced_start) is int
            else None,
            acceptance_error_scope=self.selected.acceptance_error_scope,
        )

    def log_acceptance(
        self,
        base: AcceptanceAttempt,
        changes: list[float],
        outcome: str,
        change: float | None = None,
        error: float | None = None,
    ) -> None:
        ratio = None if change is None else -change / base.predicted_decrease
        self.acceptance_attempts.append(
            replace(
                base,
                outcome=outcome,
                replicate_changes=tuple(changes),
                mean_change=change,
                standard_error=error,
                agreement=ratio if ratio is None or math.isfinite(ratio) else None,
            )
        )


def _gradient_attempt(
    replicates: tuple[Vector, ...],
    parameters: Vector,
    selected: StochasticPolicy,
    deterministic: bool,
    *,
    iteration: int,
    batch_size: int,
    first_history: int,
    histories_used: int,
    model_id: int | None,
    pool: str = "proposal",
    alpha: float | None = None,
) -> GradientAttempt:
    if any(len(value) != len(parameters) for value in replicates):
        raise ContractError("gradient replicate dimension differs from the active chart")
    statistics = tuple(
        mean_standard_error(tuple(row[j] for row in replicates)) for j in range(len(parameters))
    )
    gradient = tuple(mean for mean, _ in statistics)
    norm = math.hypot(*gradient)
    empirical_error = math.hypot(*(error for _, error in statistics))
    if not math.isfinite(norm) or not math.isfinite(empirical_error):
        raise NumericalError("gradient or uncertainty norm exceeds the finite chart range")
    if deterministic and empirical_error > selected.numerical_gradient_allowance:
        raise NumericalError("verified deterministic oracle exceeds its numerical allowance")
    error_norm = 0.0 if deterministic else empirical_error
    allowance = selected.numerical_gradient_allowance if deterministic else 0.0
    confidence_alpha = None
    upper_bound = norm + selected.standard_error_multiplier * error_norm + allowance
    calibrated = not deterministic and selected.stationarity_calibration == "student_t"
    if calibrated:
        confidence_alpha = selected.stationarity_alpha if alpha is None else alpha
        critical = student_t_critical(len(replicates) - 1, confidence_alpha / (2 * len(parameters)))
        upper_bound = math.hypot(*(abs(mean) + critical * error for mean, error in statistics))
        if not math.isfinite(upper_bound):
            raise NumericalError("stationarity confidence bound exceeds the finite chart range")
    decision: Literal[
        "zero_sample", "zero_variance", "gradient_band", "resolved", "relative_uncertainty"
    ]
    if not deterministic and error_norm == 0.0:
        decision = "zero_sample" if norm == 0 else "zero_variance"
    elif upper_bound <= selected.gradient_tolerance:
        decision = "gradient_band"
    elif norm > 0 and error_norm + allowance <= selected.relative_gradient_uncertainty * norm:
        decision = "resolved"
    else:
        decision = "relative_uncertainty"
    return GradientAttempt(
        iteration=iteration,
        parameters=parameters,
        batch_size=batch_size,
        first_history=first_history,
        histories_used=histories_used,
        gradient=gradient,
        standard_error=(0.0,) * len(parameters)
        if deterministic
        else tuple(error for _, error in statistics),
        gradient_norm=norm,
        standard_error_norm=error_norm,
        decision=decision,
        model_id=model_id,
        pool=pool,
        band_method="deterministic_numerical_allowance"
        if deterministic
        else "student_t_bonferroni_vector"
        if calibrated
        else "heuristic_euclidean_marginal_se",
        numerical_allowance=allowance,
        replicate_gradients=replicates,
        stationarity_upper_bound=upper_bound,
        confidence_alpha=confidence_alpha,
    )


@dataclass(frozen=True, slots=True)
class AcceptanceDecision:
    mean: float
    standard_error: float
    band: float
    outcome: Literal["accepted", "rejected", "ambiguous"]


def acceptance_decision(
    changes: tuple[float, ...],
    threshold: float,
    *,
    alpha: float,
    maximum_looks: int,
    deterministic: bool = False,
) -> AcceptanceDecision:
    """Band = t_(n-1,alpha/maximum_looks)*SE and accept when mean(Delta)+Band < threshold.

    Changes is a length-n tuple of independent finite binary64 scalar loss-change
    replicates with n >= 4. Threshold shares loss units and alpha is a scalar
    error allocation split across the positive integer maximum_looks declared
    before sampling. AcceptanceDecision contains binary64 mean, standard error,
    band and the accepted/rejected/ambiguous outcome. A point estimate at or
    above threshold is rejected. Student calibration assumes independent Gaussian
    replicate means. Deterministic=True is an explicit model classification and
    uses a zero band, rather than inferring determinism from zero sample variance.
    This host decision does not change the incumbent parameters.

    >>> from dpt.stochastic_recovery import acceptance_decision
    >>> result = acceptance_decision(
    ...     (-1.0, -1.0, -1.0, -1.0),
    ...     0.0,
    ...     alpha=0.05,
    ...     maximum_looks=1,
    ...     deterministic=True,
    ... )
    >>> result.outcome
    'accepted'

    """
    if len(changes) < 4 or maximum_looks < 1 or not 0 < alpha < 0.5:
        raise ContractError("acceptance needs >=4 replicates and a valid look/alpha budget")
    mean, error = mean_standard_error(changes)
    band = (
        0.0
        if deterministic
        else student_t_critical(len(changes) - 1, alpha / maximum_looks) * error
    )
    if not math.isfinite(band):
        raise NumericalError("objective-change uncertainty band exceeds finite range")
    outcome: Literal["accepted", "rejected", "ambiguous"] = "ambiguous"
    if mean >= threshold:
        outcome = "rejected"
    elif (deterministic or error > 0) and mean + band < threshold:
        outcome = "accepted"
    return AcceptanceDecision(mean, error, band, outcome)


def recover_expected_signal(
    oracle: IndependentSquaredOracle,
    initial: Vector,
    *,
    seed: int,
    policy: StochasticPolicy | None = None,
    observe_lifecycle: LifecycleObserver | None = None,
) -> StochasticRecoveryResult:
    """Minimise ||E[Y(x)]-observation||^2/2 using independent products and held-out changes.

    Initial is a finite length-D binary64 tuple in the oracle's fixed chart.
    The oracle returns length-D gradient replicates and scalar loss changes,
    using the supplied disjoint HistoryBatch identities for independent mean
    and derivative estimates. Quadratic proposals also return a D-by-D PSD
    metric with D <= 16. Seed is a 64-bit history key and policy bounds radii,
    replicates, histories and acceptance looks. StochasticRecoveryResult records
    accepted parameters, reason, uncertainty and identity/accounting diagnostics.
    Proposal/acceptance pools remain separate and a scheduled final checkpoint
    never feeds another proposal. Gaussian Student assumptions belong to the
    selected calibration and deterministic_sampling must come from immutable
    physical/source properties. Active variables are exactly the oracle's chart
    parameters. Host controls use binary64 and observation-sized arrays remain
    owned by the oracle.

    >>> from dpt.stochastic_recovery import StochasticPolicy, recover_expected_signal
    >>> class LinearSignal:
    ...     deterministic_sampling = True
    ...
    ...     def gradient_replicate(self, x, mean_batch, derivative_batch):
    ...         return (x[0] - 1.0,)
    ...
    ...     def change_replicate(self, before, after, first, second):
    ...         return ((after[0] - 1.0) ** 2 - (before[0] - 1.0) ** 2) / 2
    >>> result = recover_expected_signal(
    ...     LinearSignal(),
    ...     (0.0,),
    ...     seed=41,
    ...     policy=StochasticPolicy(
    ...         iterations=3, replicates=4, initial_batch=2, maximum_batch=2
    ...     ),
    ... )
    >>> assert result.parameters[0] > 0.0

    """
    with LifecycleRecorder("stochastic", observe_lifecycle) as lifecycle:
        result = _recover_expected_signal(
            oracle,
            initial,
            seed=seed,
            policy=policy,
            lifecycle=lifecycle,
        )
        lifecycle.finish(result, detail=result.termination_detail)
        return result


def _recover_expected_signal(
    oracle: IndependentSquaredOracle,
    initial: Vector,
    *,
    seed: int,
    policy: StochasticPolicy | None = None,
    lifecycle: LifecycleRecorder,
) -> StochasticRecoveryResult:
    """Fresh proposal/acceptance pools with an optional held-out final checkpoint.

    Quadratic proposals cache a PSD metric at the unchanged incumbent; rejected
    radii use fresh acceptance identities. Final sample size is fixed here before
    observing any samples. A failed final checkpoint preserves the triggering
    termination cause and never feeds a new proposal. ``linear`` uses the same
    acceptance test with a first-order proposal and a heuristic stopping rule.

    The explicit ``deterministic_sampling`` oracle flag must be established from
    immutable physical/source properties. Empirical variance never establishes
    that classification. Its numerical allowance requires independent validation
    by the caller. The default stochastic norm bands are heuristic. Opt-in
    Student stationarity bounds require independent Gaussian replicate errors
    and use a held-out checkpoint or predeclared Bonferroni schedule. Proposal
    screens are never certificates in that mode. Unique identities are not
    replay costs; histories_traced reports the oracle counter delta if available.
    """
    return _StochasticController(
        oracle, initial, seed=seed, policy=policy, lifecycle=lifecycle
    ).run()


class _StochasticController:
    """Own proposal state, sampling reservations and final validation for one run."""

    def __init__(
        self,
        oracle: IndependentSquaredOracle,
        initial: Vector,
        *,
        seed: int,
        policy: StochasticPolicy | None,
        lifecycle: LifecycleRecorder,
    ) -> None:
        self.oracle = oracle
        self.lifecycle = lifecycle
        self.selected: StochasticPolicy = policy or StochasticPolicy()
        self.gradient_batched = (
            cast(BatchedSquaredOracle, self.oracle).gradient_replicates
            if callable(getattr(self.oracle, "gradient_replicates", None))
            else None
        )
        self.model_batched = (
            cast(BatchedQuadraticSquaredOracle, self.oracle).model_replicates
            if callable(getattr(self.oracle, "model_replicates", None))
            else None
        )
        self.change_batched = (
            cast(BatchedSquaredOracle, self.oracle).change_replicates
            if callable(getattr(self.oracle, "change_replicates", None))
            else None
        )
        model_available = callable(getattr(self.oracle, "model_replicate", None))
        model_capability = getattr(self.oracle, "supports_local_model", model_available)
        if type(model_capability) is not bool:
            raise ContractError("supports_local_model must be an explicit boolean capability")
        self.quadratic = self.selected.proposal == "quadratic" or (
            self.selected.proposal == "auto" and model_available and model_capability
        )
        integer(seed, "seed", maximum=2**64 - 1)
        self.parameters: Vector = tuple(
            finite_scalar(value, "initial parameter") for value in initial
        )
        if not self.parameters:
            raise ContractError("a stochastic inverse problem needs active parameters")
        if self.quadratic and len(self.parameters) > 16:
            raise ContractError("dense stochastic models support at most 16 active parameters")
        if self.quadratic and (not model_available or not model_capability):
            raise ContractError("quadratic proposals require an oracle model_replicate operation")
        classification = getattr(self.oracle, "deterministic_sampling", False)
        if type(classification) is not bool:
            raise ContractError("deterministic_sampling must be an explicit boolean contract")
        self.deterministic: bool = bool(classification)
        self.final_size = self.selected.final_validation_batch
        if (
            self.quadratic or self.selected.stationarity_calibration == "student_t"
        ) and self.final_size is None:
            self.final_size = (
                self.selected.initial_batch if self.deterministic else self.selected.maximum_batch
            )
        self.final_sizes = self.selected.final_validation_batches or (
            () if self.final_size is None else (self.final_size,)
        )
        self.final_size = self.final_sizes[-1] if self.final_sizes else None
        self.final_cost = 2 * sum(self.final_sizes) * self.selected.replicates
        self.ledger = _SamplingLedger(seed, self.selected)
        self.radius = self.selected.initial_radius
        self.batch_size = self.selected.initial_batch
        self.damping_relative = self.selected.damping_relative
        self.rank_tolerance = self.selected.rank_tolerance
        self.steps: list[StochasticStep] = []
        self.domain_rejections: list[DomainRejection] = []
        self.gradient_attempts: list[GradientAttempt] = []
        self.acceptance_attempts: list[AcceptanceAttempt] = []
        self.reservations = self.ledger.reservations
        self.local_models: list[LocalModel] = []
        self.validation_attempts: list[GradientAttempt] = []
        self.validation_trigger: str | None = None
        self.validation_outcome: FinalValidationOutcome | None = None
        self.proposal_cache: _ProposalCache = _ProposalCache(self.deterministic)
        self.cached: list[LocalModel] = self.proposal_cache.models
        self.gradient: Vector = ()
        self.candidate_looks: dict[tuple[Vector, Vector], int] = {}
        self.maximum_looks = self.selected.acceptance_maximum_looks or (
            1 + math.ceil(math.log2(self.selected.maximum_batch / self.selected.initial_batch))
        )
        self.acceptance_alpha = self.selected.acceptance_alpha / (
            self.selected.iterations if self.selected.acceptance_error_scope == "run" else 1
        )
        self.traced_start = getattr(self.oracle, "histories_traced", None)
        self.band_method = (
            "deterministic_numerical_allowance"
            if self.deterministic
            else "student_t_bonferroni_vector"
            if self.selected.stationarity_calibration == "student_t"
            else "heuristic_euclidean_marginal_se"
        )

        self.report = _StochasticReportState(
            ledger=self.ledger,
            reservations=self.reservations,
            selected=self.selected,
            oracle=self.oracle,
            deterministic=self.deterministic,
            quadratic=self.quadratic,
            band_method=self.band_method,
            final_size=self.final_size,
            traced_start=self.traced_start,
            steps=self.steps,
            domain_rejections=self.domain_rejections,
            gradient_attempts=self.gradient_attempts,
            acceptance_attempts=self.acceptance_attempts,
            local_models=self.local_models,
            validation_attempts=self.validation_attempts,
            validation_trigger=self.validation_trigger,
            validation_outcome=self.validation_outcome,
        )

    def result(
        self,
        reason: StochasticStopReason,
        detail: str = "",
    ) -> StochasticRecoveryResult:
        return self.report.result(self.parameters, reason, detail)

    def record_exception(self, error: Exception) -> None:
        # Preserve the normal numerical exception contract, with a serialisable
        # snapshot for callers recording a failed run. Partial device work is
        # deliberately unknown here; the production operator records its cost.
        error.__dict__["recovery_diagnostics"] = self.result(
            "sampling_unresolved", "oracle_exception"
        )
        error.add_note("Recovery reservation/attempt snapshot is in recovery_diagnostics.")

    def terminate(
        self, reason: StochasticStopReason, detail: str, iteration: int
    ) -> StochasticRecoveryResult:
        if (
            self.final_size is None
            or self.ledger.first_history + self.final_cost > self.selected.unique_history_budget
        ):
            return self.result(reason, detail)
        self.report.validation_trigger = detail or reason
        for size in self.final_sizes:
            index = self.ledger.reserve(size, "final_validation", iteration)
            if index is None:
                return self.result("history_budget", "final_validation_reservation_failed")
            try:
                values = self.ledger.execute(
                    index,
                    partial(self.oracle.gradient_replicate, self.parameters),
                    batched=partial(self.gradient_batched, self.parameters)
                    if self.gradient_batched is not None
                    else None,
                )
                attempt = _gradient_attempt(
                    values,
                    self.parameters,
                    self.selected,
                    self.deterministic,
                    iteration=iteration,
                    batch_size=size,
                    first_history=self.reservations[index].first_history,
                    histories_used=self.reservations[index].reserved_histories,
                    model_id=None,
                    pool="final_validation",
                    alpha=self.selected.stationarity_alpha / len(self.final_sizes),
                )
            except Exception as error:
                self.record_exception(error)
                raise
            self.validation_attempts.append(attempt)
            if attempt.decision == "gradient_band":
                self.report.validation_outcome = "certified"
                return self.result("gradient_band", "held_out_gradient_band")
            self.report.validation_outcome = (
                "clearly_nonstationary" if attempt.decision == "resolved" else "not_certified"
            )
        if reason == "gradient_band":
            return self.result("sampling_unresolved", "final_validation_failed")
        return self.result(reason, detail)

    def run(self) -> StochasticRecoveryResult:
        self.lifecycle.start(objective=None)
        for iteration in range(self.selected.iterations):
            if not self.cached and self.proposal_cache.linear is None:
                # Fresh-batch growth is deliberately retained: there is no pooling
                # across adaptive epochs or different parameter values.
                while True:
                    checkpoint = self.final_cost
                    if self.final_size is not None:
                        checkpoint += 2 * self.batch_size * self.selected.replicates
                    index = self.ledger.reserve(
                        self.batch_size,
                        "model" if self.quadratic else "gradient",
                        iteration,
                        checkpoint,
                    )
                    if index is None:
                        return self.terminate(
                            "history_budget",
                            "proposal_checkpoint_reservation_failed"
                            if checkpoint
                            else "history_budget",
                            iteration,
                        )
                    model_id = len(self.local_models) if self.quadratic else None
                    try:
                        replicates, curvature = self._estimate_replicates(index)
                        attempt = _gradient_attempt(
                            replicates,
                            self.parameters,
                            self.selected,
                            self.proposal_cache.deterministic,
                            iteration=iteration,
                            batch_size=self.batch_size,
                            first_history=self.reservations[index].first_history,
                            histories_used=self.reservations[index].reserved_histories,
                            model_id=model_id,
                        )
                    except Exception as error:
                        self.record_exception(error)
                        raise
                    self.gradient_attempts.append(attempt)
                    if model_id is not None:
                        self.local_models.append(
                            LocalModel(
                                model_id,
                                self.parameters,
                                attempt.gradient,
                                curvature,
                                attempt.first_history,
                                attempt.histories_used,
                                self.batch_size,
                            )
                        )
                    if attempt.decision in ("zero_sample", "zero_variance"):
                        if self.batch_size == self.selected.maximum_batch:
                            return self.terminate(
                                "sampling_unresolved",
                                "zero_gradient_and_variance_at_maximum_batch"
                                if attempt.decision == "zero_sample"
                                else "zero_empirical_gradient_variance_at_maximum_batch",
                                iteration,
                            )
                        self.batch_size = min(2 * self.batch_size, self.selected.maximum_batch)
                        continue
                    if attempt.decision == "gradient_band":
                        return self.terminate("gradient_band", "proposal_gradient_band", iteration)
                    if attempt.decision == "resolved":
                        self.gradient, norm = attempt.gradient, attempt.gradient_norm
                        if model_id is not None:
                            self.cached[:] = [self.local_models[-1]]
                        elif self.selected.cache_linear_gradient:
                            self.proposal_cache.linear = attempt
                        break
                    if self.batch_size == self.selected.maximum_batch:
                        return self.terminate(
                            "sampling_unresolved",
                            "gradient_uncertainty_at_maximum_batch",
                            iteration,
                        )
                    self.batch_size = (
                        predicted_batch_size(
                            self.batch_size,
                            self.selected.maximum_batch,
                            attempt.standard_error_norm,
                            self.selected.relative_gradient_uncertainty * attempt.gradient_norm,
                        )
                        if self.selected.batch_growth == "predicted"
                        else min(2 * self.batch_size, self.selected.maximum_batch)
                    )
            else:
                self.gradient = self.proposal_cache.gradient
                norm = math.hypot(*self.gradient)
            step, predicted, boundary, active_curvature, proposal, candidate = self._propose(norm)
            if (
                not math.isfinite(predicted)
                or predicted <= 0
                or not all(map(math.isfinite, candidate))
            ):
                raise NumericalError("stochastic proposal exceeds the finite chart range")
            if candidate == self.parameters:
                return self.terminate(
                    "sampling_unresolved", "proposal_below_chart_resolution", iteration
                )
            # Addition in a large finite chart can round the requested displacement.
            # Log the realised move and assess its actual quadratic prediction. The
            # Linear proposals retain their first-order predicted decrease.
            step = tuple(b - a for a, b in zip(self.parameters, candidate, strict=True))
            if proposal is not None:
                actual_norm = math.hypot(*step)
                chart_rounding = math.hypot(
                    *(
                        math.ulp(max(abs(a), abs(b)))
                        for a, b in zip(self.parameters, candidate, strict=True)
                    )
                )
                # Permit representational noise at ordinary chart magnitudes, but
                # reject a materially enlarged move in a poorly resolved chart.
                radius_slack = max(1e-12 * self.radius, min(chart_rounding, 1e-8 * self.radius))
                if actual_norm > self.radius + radius_slack:
                    return self.terminate(
                        "sampling_unresolved", "chart_rounding_exceeds_trust_radius", iteration
                    )
                predicted = quadratic_reduction(self.gradient, active_curvature, step)
                boundary = proposal.boundary and actual_norm >= self.radius - radius_slack
                proposal = replace(
                    proposal, step=step, predicted_decrease=predicted, boundary=boundary
                )
            candidate_key = (self.parameters, candidate)
            look = self.candidate_looks.get(candidate_key, 0)
            while True:
                if look >= self.maximum_looks:
                    return self.terminate(
                        "sampling_unresolved", "candidate_acceptance_looks_exhausted", iteration
                    )
                look += 1
                self.candidate_looks[candidate_key] = look
                start = self.ledger.first_history
                index = self.ledger.reserve(
                    self.batch_size, "acceptance", iteration, self.final_cost
                )
                changes: list[float] = []
                threshold = -self.selected.acceptance_fraction * predicted

                base = AcceptanceAttempt(
                    iteration=iteration,
                    incumbent=self.parameters,
                    candidate=candidate,
                    model_id=None if not self.cached else self.cached[0].model_id,
                    step=step,
                    predicted_decrease=predicted,
                    radius=self.radius,
                    batch_size=self.batch_size,
                    first_history=start,
                    histories_used=self.ledger.first_history - start,
                    required_histories=2 * self.batch_size * self.selected.replicates,
                    replicate_changes=(),
                    mean_change=None,
                    standard_error=None,
                    band_method="deterministic_numerical_allowance"
                    if self.deterministic
                    else "student_t_bonferroni",
                    look_index=look,
                    threshold=threshold,
                    outcome="reserved",
                    agreement=None,
                    boundary=boundary,
                    proposal_diagnostics=proposal,
                    confidence_alpha=None
                    if self.deterministic
                    else self.acceptance_alpha / self.maximum_looks,
                    error_scope=self.selected.acceptance_error_scope,
                )

                if index is None:
                    self.report.log_acceptance(base, changes, "budget_denied")
                    return self.terminate(
                        "history_budget",
                        "acceptance_checkpoint_reservation_failed"
                        if self.final_cost
                        else "history_budget",
                        iteration,
                    )

                def change_operation(
                    a: HistoryBatch,
                    b: HistoryBatch,
                    *,
                    before: Vector = self.parameters,
                    after: Vector = candidate,
                    outputs: list[float] = changes,
                ) -> float:
                    value = self.oracle.change_replicate(before, after, a, b)
                    if not math.isfinite(value):
                        raise NumericalError("objective-change replicate is nonfinite")
                    outputs.append(value)
                    return value

                try:
                    decision = self._test_candidate(
                        index, candidate, changes, threshold, change_operation
                    )
                    change, standard_error = decision.mean, decision.standard_error
                except TrialDomainError as error:
                    self.report.log_acceptance(base, changes, "domain_error")
                    self.domain_rejections.append(
                        DomainRejection(iteration, self.radius, str(error))
                    )
                    self._shrink_radius(step)
                    self.lifecycle.rejected(iteration + 1, "domain")
                    if self.radius < self.selected.minimum_radius:
                        return self.terminate("radius_limit", "radius_limit", iteration)
                    break
                except Exception as error:
                    self.report.log_acceptance(base, changes, "oracle_error")
                    self.record_exception(error)
                    raise
                if decision.outcome == "accepted":
                    self.report.log_acceptance(base, changes, "accepted", change, standard_error)
                    self._record_step(iteration, True, change, standard_error, predicted, start)
                    self.parameters = candidate
                    self.cached.clear()
                    self.proposal_cache.linear = None
                    if (not self.quadratic and self.selected.linear_radius_growth == "always") or (
                        boundary and -change / predicted >= self.selected.radius_growth_agreement
                    ):
                        self.radius = min(2 * self.radius, self.selected.maximum_radius)
                    self.lifecycle.accepted(iteration + 1, None)
                    break
                if decision.outcome == "rejected":
                    self.report.log_acceptance(base, changes, "rejected", change, standard_error)
                    self._record_step(iteration, False, change, standard_error, predicted, start)
                    self._shrink_radius(step)
                    self.lifecycle.rejected(iteration + 1, "agreement")
                    if self.radius < self.selected.minimum_radius:
                        return self.terminate("radius_limit", "radius_limit", iteration)
                    break
                self.report.log_acceptance(base, changes, "ambiguous", change, standard_error)
                if self.batch_size == self.selected.maximum_batch:
                    return self.terminate(
                        "sampling_unresolved", "acceptance_uncertainty_at_maximum_batch", iteration
                    )
                self.batch_size = (
                    predicted_batch_size(
                        self.batch_size,
                        self.selected.maximum_batch,
                        decision.band,
                        threshold - change,
                    )
                    if self.selected.batch_growth == "predicted"
                    else min(2 * self.batch_size, self.selected.maximum_batch)
                )
                if (
                    self.quadratic
                    and not self.deterministic
                    and self.batch_size > self.cached[0].batch_size
                ):
                    # A more precise change test must not keep proposing from a
                    # coarser stochastic gradient/metric. Rebuild at the incumbent
                    # with the enlarged batch and fresh, disjoint model identities.
                    self.cached.clear()
                    break
                if (
                    self.proposal_cache.linear is not None
                    and self.batch_size > self.proposal_cache.linear.batch_size
                ):
                    self.proposal_cache.linear = None
                    break
        return self.terminate("iteration_budget", "iteration_budget", self.selected.iterations)

    def _estimate_replicates(self, index: int) -> tuple[tuple[Vector, ...], Matrix]:
        curvature: Matrix = ()
        if self.quadratic:
            model_oracle = cast(QuadraticSquaredOracle, self.oracle)
            models = self.ledger.execute(
                index,
                partial(model_oracle.model_replicate, self.parameters),
                batched=partial(self.model_batched, self.parameters)
                if self.model_batched is not None
                else None,
            )
            replicates = tuple(value[0] for value in models)
            size = len(self.parameters)
            if any(
                len(value[1]) != size or any(len(row) != size for row in value[1])
                for value in models
            ):
                raise ContractError("model curvature dimension differs from the active chart")
            curvature = tuple(
                tuple(
                    mean_standard_error(tuple(value[1][i][j] for value in models))[0]
                    for j in range(size)
                )
                for i in range(size)
            )
            validate_curvature(curvature, size)
        else:
            replicates = self.ledger.execute(
                index,
                partial(self.oracle.gradient_replicate, self.parameters),
                batched=partial(self.gradient_batched, self.parameters)
                if self.gradient_batched is not None
                else None,
            )
        return replicates, curvature

    def _propose(
        self,
        norm: float,
    ) -> tuple[Vector, float, bool, Matrix, QuadraticProposal | None, Vector]:
        proposal: QuadraticProposal | None = None
        active_curvature: Matrix = ()
        if self.quadratic:
            assert self.cached
            active_curvature = self.cached[0].curvature
            proposal = quadratic_proposal(
                self.gradient,
                active_curvature,
                self.radius,
                damping_relative=self.damping_relative,
                rank_tolerance=self.rank_tolerance,
            )
            step, predicted, boundary = (
                proposal.step,
                proposal.predicted_decrease,
                proposal.boundary,
            )
        else:
            step = tuple(-self.radius * (value / norm) for value in self.gradient)
            predicted, boundary = self.radius * norm, True
        candidate: Vector = tuple(a + b for a, b in zip(self.parameters, step, strict=True))
        return step, predicted, boundary, active_curvature, proposal, candidate

    def _test_candidate(
        self,
        index: int,
        candidate: Vector,
        changes: list[float],
        threshold: float,
        change_operation: Callable[[HistoryBatch, HistoryBatch], float],
    ) -> AcceptanceDecision:
        if self.change_batched is not None:
            changes.extend(
                self.ledger.execute(
                    index,
                    partial(self.oracle.change_replicate, self.parameters, candidate),
                    batched=partial(self.change_batched, self.parameters, candidate),
                )
            )
            if not all(math.isfinite(value) for value in changes):
                raise NumericalError("objective-change replicate is nonfinite")
        else:
            self.ledger.execute(index, change_operation)
        decision = acceptance_decision(
            tuple(changes),
            threshold,
            alpha=self.acceptance_alpha,
            maximum_looks=self.maximum_looks,
            deterministic=self.deterministic,
        )
        return decision

    def _record_step(
        self,
        iteration: int,
        accepted: bool,
        change: float,
        standard_error: float,
        predicted: float,
        start: int,
    ) -> None:
        self.steps.append(
            StochasticStep(
                iteration=iteration,
                accepted=accepted,
                mean_change=change,
                change_standard_error=standard_error,
                predicted_decrease=predicted,
                radius=self.radius,
                batch_size=self.batch_size,
                first_history=start,
                histories_used=self.ledger.first_history - start,
            )
        )

    def _shrink_radius(self, step: Vector) -> None:
        self.radius = 0.5 * min(self.radius, math.hypot(*step))
