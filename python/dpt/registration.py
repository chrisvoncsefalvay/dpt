"""Safeguarded limited-memory optimisation in a fixed, scaled parameter chart.

The evaluator owns the CUDA composition. It returns only a scalar objective and
the small parameter gradient, never a detector image. Parameterisations (pose
anchor, units, constrained nuisance variables) remain fixed throughout a solve.
Starting a new chart requires a new solve and therefore empty curvature history.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from dpt.contracts import ContractError, NumericalError, TrialDomainError, finite_scalar, integer
from dpt.solver_lifecycle import LifecycleObserver, LifecycleRecorder

Vector = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Objective and derivative with respect to the supplied dimensionless chart."""

    loss: float
    gradient: Vector


class Evaluator(Protocol):
    def __call__(self, parameters: Vector, /) -> Evaluation:
        """Evaluate parameters; raise TrialDomainError only for a rejectable chart trial."""
        ...


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    """Safeguarded deterministic chart search.

    absolute_loss_resolution is an opt-in caller-declared bound on meaningful
    loss changes. Unresolved trials can stop as loss_resolution; they are never
    accepted using an error allowance and never establish stationarity.
    """

    max_iterations: int = 200
    max_evaluations: int = 2000
    memory: int = 10
    line_search_evaluations: int = 32
    gradient_tolerance: float = 1e-6
    relative_loss_tolerance: float = 1e-12
    step_tolerance: float = 1e-10
    armijo: float = 1e-4
    curvature: float = 0.9
    maximum_step: float = 16.0
    minimum_step: float = 1e-14
    absolute_loss_resolution: float = 0.0

    def __post_init__(self) -> None:
        for name in ("max_iterations", "max_evaluations", "memory", "line_search_evaluations"):
            integer(getattr(self, name), name, minimum=1)
        for name in (
            "gradient_tolerance",
            "relative_loss_tolerance",
            "step_tolerance",
            "absolute_loss_resolution",
        ):
            finite_scalar(getattr(self, name), name, minimum=0.0)
        for name in ("armijo", "curvature", "minimum_step", "maximum_step"):
            finite_scalar(getattr(self, name), name, minimum=0.0)
        if not 0.0 < self.armijo < self.curvature < 1.0:
            raise ContractError("strong-Wolfe constants must satisfy 0 < c1 < c2 < 1")
        if not 0.0 < self.minimum_step < self.maximum_step:
            raise ContractError("line-search step interval must be positive and nonempty")


StopReason = Literal[
    "gradient_tolerance",
    "step_tolerance",
    "loss_stagnation",
    "line_search_failed",
    "evaluation_budget",
    "iteration_budget",
    "cancelled",
    "evaluation_failed",
    "loss_resolution",
]


@dataclass(frozen=True, slots=True)
class IterationRecord:
    iteration: int
    loss: float
    gradient_infinity_norm: float
    step: float
    evaluations: int
    curvature_pairs: int
    line_search: Literal["initial", "wolfe", "armijo_bound", "armijo_best"] = "initial"


@dataclass(frozen=True, slots=True)
class RejectedTrial:
    parameters: Vector
    message: str


@dataclass(frozen=True, slots=True)
class RecoveryCheckpoint:
    """Host state for the same immutable objective and parameter chart.

    objective_id is a caller-supplied identity for fitting inputs, chart, model
    and active parameters. Resume re-evaluates the accepted point and checks its
    value and gradient before reusing curvature. Budgets may be increased; all
    other numerical policy fields must agree. No device scratch is serialised.
    """

    objective_id: str
    policy: RecoveryPolicy
    parameters: Vector
    evaluation: Evaluation
    curvature_pairs: tuple[tuple[Vector, Vector, float], ...]
    history: tuple[IterationRecord, ...]
    rejected_trials: tuple[RejectedTrial, ...]
    evaluations: int
    resolution_limited_trials: int


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    parameters: Vector
    evaluation: Evaluation
    reason: StopReason
    evaluations: int
    history: tuple[IterationRecord, ...]
    rejected_trials: tuple[RejectedTrial, ...] = ()
    resolution_limited_trials: int = 0
    checkpoint: RecoveryCheckpoint | None = None

    @property
    def stationary(self) -> bool:
        """Only the gradient criterion establishes numerical stationarity here.

        Small loss/step changes indicate stagnation; they do not prove the pose
        was recovered or that the current point is an identifiable minimum.
        """
        return self.reason == "gradient_tolerance"


def _dot(a: Vector, b: Vector) -> float:
    try:
        return math.fsum(x * y for x, y in zip(a, b, strict=True))
    except (OverflowError, ValueError):
        # Curvature and line-search callers reject an unrepresentable product;
        # it cannot certify a direction or replace the accepted iterate.
        return math.nan


def _norm(a: Vector) -> float:
    return max(abs(x) for x in a)


def _checked(evaluation: Evaluation, dimension: int) -> Evaluation:
    if len(evaluation.gradient) != dimension:
        raise ContractError("evaluator gradient dimension differs from its parameter chart")
    if not math.isfinite(evaluation.loss) or not all(map(math.isfinite, evaluation.gradient)):
        raise NumericalError("non-finite objective or parameter gradient")
    return evaluation


def _direction(gradient: Vector, pairs: list[tuple[Vector, Vector, float]]) -> Vector:
    """Two-loop recursion; curvature vectors all belong to the same scaled chart."""
    if not pairs:
        return _steepest_direction(gradient)
    q = gradient
    coefficients: list[float] = []
    for displacement, change, reciprocal in reversed(pairs):
        coefficient = reciprocal * _dot(displacement, q)
        coefficients.append(coefficient)
        q = tuple(a - coefficient * b for a, b in zip(q, change, strict=True))
    scale = 1.0
    if pairs:
        displacement, change, _ = pairs[-1]
        denominator = _dot(change, change)
        if not math.isfinite(denominator) or denominator <= 0.0:
            return _steepest_direction(gradient)
        scale = _dot(displacement, change) / denominator
        if not math.isfinite(scale) or scale <= 0.0:
            return _steepest_direction(gradient)
    result = tuple(scale * value for value in q)
    for (displacement, change, reciprocal), coefficient in zip(
        pairs, reversed(coefficients), strict=True
    ):
        beta = reciprocal * _dot(change, result)
        result = tuple(
            a + (coefficient - beta) * b for a, b in zip(result, displacement, strict=True)
        )
    return tuple(-value for value in result)


def _steepest_direction(gradient: Vector) -> Vector:
    """Bound the initial chart displacement without squaring a large gradient."""
    scale = max(1.0, _norm(gradient))
    return tuple(-value / scale for value in gradient)


class _BudgetExhaustedError(Exception):
    pass


def _line_search(
    evaluate: Callable[[Vector], Evaluation],
    x: Vector,
    base: Evaluation,
    direction: Vector,
    policy: RecoveryPolicy,
    *,
    initial_step: float = 1.0,
    on_rejection: Callable[[RejectedTrial], None] | None = None,
    on_acceptance: Callable[[Literal["wolfe", "armijo_bound", "armijo_best"]], None] | None = None,
    on_resolution: Callable[[], None] | None = None,
) -> tuple[float, Evaluation] | None:
    """Seek strong Wolfe with a best strict-decrease Armijo fallback.

    Bisection sacrifices polynomial interpolation speed to keep invalid-domain
    endpoints unambiguous. Numerical failures remain fatal. No unchecked trial is
    returned as an accepted iterate. The last accepted state lives in the caller.
    """
    slope = _dot(base.gradient, direction)
    if not math.isfinite(slope) or slope >= 0:
        return None
    attempts = 0
    best: tuple[float, Evaluation] | None = None

    def accepted(
        chosen: tuple[float, Evaluation] | None,
        kind: Literal["wolfe", "armijo_bound", "armijo_best"],
    ) -> tuple[float, Evaluation] | None:
        if chosen is not None and on_acceptance is not None:
            on_acceptance(kind)
        return chosen

    def trial(alpha: float) -> Evaluation | None:
        nonlocal attempts, best
        attempts += 1
        point = tuple(value + alpha * step for value, step in zip(x, direction, strict=True))
        if not all(map(math.isfinite, point)):
            return None
        try:
            candidate = _checked(evaluate(point), len(x))
        except TrialDomainError as error:
            if on_rejection is not None:
                on_rejection(RejectedTrial(point, str(error)))
            return None
        except _BudgetExhaustedError:
            if best is None:
                raise
            attempts = policy.line_search_evaluations
            return None
        if (
            policy.absolute_loss_resolution > 0
            and abs(candidate.loss - base.loss) <= policy.absolute_loss_resolution
            and -policy.armijo * alpha * slope <= policy.absolute_loss_resolution
            and on_resolution is not None
        ):
            on_resolution()
        if candidate.loss < base.loss and armijo(alpha, candidate):
            if best is None or candidate.loss < best[1].loss:
                best = alpha, candidate
        return candidate

    def armijo(alpha: float, candidate: Evaluation) -> bool:
        return candidate.loss <= base.loss + policy.armijo * alpha * slope

    def zoom(low: float, high: float, low_value: Evaluation) -> tuple[float, Evaluation] | None:
        while attempts < policy.line_search_evaluations:
            alpha = 0.5 * (low + high)
            if abs(high - low) < policy.minimum_step or alpha == low or alpha == high:
                return accepted(best, "armijo_best")
            candidate = trial(alpha)
            if (
                candidate is None
                or not armijo(alpha, candidate)
                or candidate.loss >= low_value.loss
            ):
                high = alpha
                continue
            derivative = _dot(candidate.gradient, direction)
            if abs(derivative) <= -policy.curvature * slope:
                return accepted((alpha, candidate), "wolfe")
            if derivative * (high - low) >= 0:
                high = low
            low, low_value = alpha, candidate
        return accepted(best, "armijo_best")

    previous_alpha, previous_value = 0.0, base
    alpha = min(initial_step, policy.maximum_step)
    while attempts < policy.line_search_evaluations:
        candidate = trial(alpha)
        if (
            candidate is None
            or not armijo(alpha, candidate)
            or (previous_alpha > 0 and candidate.loss >= previous_value.loss)
        ):
            return zoom(previous_alpha, alpha, previous_value)
        derivative = _dot(candidate.gradient, direction)
        if abs(derivative) <= -policy.curvature * slope:
            return accepted((alpha, candidate), "wolfe")
        if derivative >= 0:
            return zoom(alpha, previous_alpha, candidate)
        if alpha == policy.maximum_step and candidate.loss < base.loss:
            return accepted((alpha, candidate), "armijo_bound")
        if alpha == policy.maximum_step:
            return accepted(best, "armijo_best")
        previous_alpha, previous_value = alpha, candidate
        alpha = min(2.0 * alpha, policy.maximum_step)
    return accepted(best, "armijo_best")


def recover_parameters(
    evaluator: Evaluator,
    initial: Vector | None = None,
    *,
    policy: RecoveryPolicy | None = None,
    observe: Callable[[IterationRecord], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    objective_id: str | None = None,
    checkpoint: RecoveryCheckpoint | None = None,
    save_checkpoint: Callable[[RecoveryCheckpoint], None] | None = None,
    checkpoint_relative_tolerance: float = 0.0,
    observe_lifecycle: LifecycleObserver | None = None,
) -> RecoveryResult:
    """Minimise f(x) using its gradient in one fixed dimensionless parameter chart.

    Initial is a finite length-D binary64 tuple, or is recovered from the supplied
    checkpoint. Evaluator returns Evaluation with scalar loss and length-D
    gradient in loss per chart-coordinate units, using the same observation,
    forward model and chart at every call. It must complete numerical checks
    before returning. RecoveryPolicy selects safeguarded quasi-Newton updates,
    line search and termination. RecoveryResult contains accepted parameters,
    evaluation, reason, evaluation count and scalar iteration records. Rejected
    trials never replace the incumbent. Checkpoint restart verifies objective_id
    and reevaluated state at the declared tolerance. Observers run after verified
    or committed states and their exceptions propagate. Only the small chart
    and scalar control records are managed by this host driver.

    >>> from dpt.registration import Evaluation, recover_parameters
    >>> def quadratic(x):
    ...     return Evaluation(0.5 * (x[0] - 2.0) ** 2, (x[0] - 2.0,))
    >>> result = recover_parameters(quadratic, (0.0,))
    >>> assert abs(result.parameters[0] - 2.0) < 1e-8

    """
    with LifecycleRecorder("registration", observe_lifecycle) as lifecycle:
        result = _recover_parameters(
            evaluator,
            initial,
            policy=policy,
            observe=observe,
            cancelled=cancelled,
            objective_id=objective_id,
            checkpoint=checkpoint,
            save_checkpoint=save_checkpoint,
            checkpoint_relative_tolerance=checkpoint_relative_tolerance,
            lifecycle=lifecycle,
        )
        lifecycle.finish(result)
        return result


def _recover_parameters(
    evaluator: Evaluator,
    initial: Vector | None = None,
    *,
    policy: RecoveryPolicy | None = None,
    observe: Callable[[IterationRecord], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    objective_id: str | None = None,
    checkpoint: RecoveryCheckpoint | None = None,
    save_checkpoint: Callable[[RecoveryCheckpoint], None] | None = None,
    checkpoint_relative_tolerance: float = 0.0,
    lifecycle: LifecycleRecorder,
) -> RecoveryResult:
    """Minimise a deterministic objective in an immutable, dimensionless chart.

    The evaluator must use the same observation, forward model and parameter
    coordinates for every call. It may reuse device scratch, but must complete
    numerical-status checks before returning. Rejected trials never replace the
    accepted parameters in the result. Observer exceptions are not swallowed.
    """
    selected = policy or RecoveryPolicy()
    checkpoint_relative_tolerance = finite_scalar(
        checkpoint_relative_tolerance, "checkpoint_relative_tolerance", minimum=0
    )
    if objective_id is not None and (not isinstance(objective_id, str) or not objective_id.strip()):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ContractError("objective_id must be a nonempty immutable model/chart identity")
    if save_checkpoint is not None and objective_id is None:
        raise ContractError("save_checkpoint requires objective_id")
    if checkpoint is not None:
        if not isinstance(checkpoint, RecoveryCheckpoint):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ContractError("checkpoint must be a RecoveryCheckpoint")
        if objective_id is None or objective_id != checkpoint.objective_id:
            raise ContractError("checkpoint objective/chart identity differs")
        if (
            replace(
                checkpoint.policy,
                max_iterations=selected.max_iterations,
                max_evaluations=selected.max_evaluations,
            )
            != selected
        ):
            raise ContractError("checkpoint numerical policy differs; only budgets may change")
        if initial is not None and initial != checkpoint.parameters:
            raise ContractError("initial parameters differ from the checkpoint")
        initial = checkpoint.parameters
    if initial is None:
        raise ContractError("supply initial parameters or a checkpoint")
    x: Vector = tuple(finite_scalar(value, "initial parameter") for value in initial)
    if not x:
        raise ContractError("recovery requires at least one active parameter")
    calls = checkpoint.evaluations if checkpoint is not None else 0
    if checkpoint is not None and calls >= selected.max_evaluations:
        raise ContractError("increase max_evaluations to verify and resume this checkpoint")
    rejected_trials = list(checkpoint.rejected_trials) if checkpoint is not None else []
    resolution_limited_trials = (
        checkpoint.resolution_limited_trials if checkpoint is not None else 0
    )

    def record_resolution() -> None:
        nonlocal resolution_limited_trials
        resolution_limited_trials += 1

    def evaluate(point: Vector) -> Evaluation:
        nonlocal calls
        if calls >= selected.max_evaluations:
            raise _BudgetExhaustedError
        calls += 1
        return _checked(evaluator(point), len(x))

    current = evaluate(x)
    history = list(checkpoint.history) if checkpoint is not None else []
    if checkpoint is not None:
        previous = _checked(checkpoint.evaluation, len(x))
        if not all(
            math.isclose(a, b, rel_tol=checkpoint_relative_tolerance, abs_tol=0)
            for a, b in zip(
                (current.loss, *current.gradient), (previous.loss, *previous.gradient), strict=True
            )
        ):
            raise ContractError(
                "checkpoint accepted value/gradient differs from the current oracle"
            )
    line_search_kind: Literal["wolfe", "armijo_bound", "armijo_best"] = "wolfe"

    def record_acceptance(kind: Literal["wolfe", "armijo_bound", "armijo_best"]) -> None:
        nonlocal line_search_kind
        line_search_kind = kind

    pairs = list(checkpoint.curvature_pairs) if checkpoint is not None else []
    for displacement, difference, reciprocal in pairs:
        if (
            len(displacement) != len(x)
            or len(difference) != len(x)
            or not all(map(math.isfinite, (*displacement, *difference, reciprocal)))
            or reciprocal <= 0
        ):
            raise ContractError("invalid checkpoint curvature pair")
    if len(pairs) > selected.memory:
        raise ContractError("checkpoint exceeds the curvature memory bound")
    reason: StopReason = "iteration_budget"
    if not history:
        history.append(IterationRecord(0, current.loss, _norm(current.gradient), 0.0, calls, 0))
    if history[-1].iteration > selected.max_iterations:
        raise ContractError("max_iterations precedes the checkpoint iteration")

    def snapshot() -> RecoveryCheckpoint | None:
        if objective_id is None:
            return None
        return RecoveryCheckpoint(
            objective_id,
            selected,
            x,
            current,
            tuple(pairs),
            tuple(history),
            tuple(rejected_trials),
            calls,
            resolution_limited_trials,
        )

    lifecycle.start(
        objective=current.loss,
        iteration=history[-1].iteration,
        accepted_steps=sum(record.iteration > 0 for record in history),
        resumed=checkpoint is not None,
    )

    def record_rejection(trial: RejectedTrial) -> None:
        rejected_trials.append(trial)
        lifecycle.rejected(iteration, "domain: " + trial.message)

    try:
        for iteration in range(history[-1].iteration + 1, selected.max_iterations + 1):
            if cancelled is not None and cancelled():
                reason = "cancelled"
                break
            if _norm(current.gradient) <= selected.gradient_tolerance:
                reason = "gradient_tolerance"
                break
            resolution_before = resolution_limited_trials
            direction = _direction(current.gradient, pairs)
            slope = _dot(direction, current.gradient)
            if not all(map(math.isfinite, direction)) or not math.isfinite(slope) or slope >= 0:
                pairs.clear()
                direction = _steepest_direction(current.gradient)
            try:
                accepted = _line_search(
                    evaluate,
                    x,
                    current,
                    direction,
                    selected,
                    on_rejection=record_rejection,
                    on_acceptance=record_acceptance,
                    on_resolution=record_resolution,
                )
                if accepted is None:
                    # A bad inverse-Hessian estimate need not terminate a sound
                    # forward model: try steepest descent once with empty memory.
                    pairs.clear()
                    direction = _steepest_direction(current.gradient)
                    accepted = _line_search(
                        evaluate,
                        x,
                        current,
                        direction,
                        selected,
                        on_rejection=record_rejection,
                        on_acceptance=record_acceptance,
                        on_resolution=record_resolution,
                        initial_step=max(
                            selected.minimum_step, 2.0 ** -min(16, selected.line_search_evaluations)
                        ),
                    )
            except _BudgetExhaustedError:
                reason = "evaluation_budget"
                break
            if accepted is None:
                reason = (
                    "loss_resolution"
                    if resolution_limited_trials > resolution_before
                    else "line_search_failed"
                )
                lifecycle.rejected(iteration, reason)
                break
            alpha, new_value = accepted
            step: Vector = tuple(alpha * value for value in direction)
            new_x: Vector = tuple(a + b for a, b in zip(x, step, strict=True))
            change = tuple(a - b for a, b in zip(new_value.gradient, current.gradient, strict=True))
            curvature = _dot(step, change)
            threshold = 1e-12 * math.hypot(*step) * math.hypot(*change)
            if (
                math.isfinite(curvature)
                and curvature > max(0.0, threshold)
                and math.isfinite(1.0 / curvature)
            ):
                pairs.append((step, change, 1.0 / curvature))
                if len(pairs) > selected.memory:
                    del pairs[0]
            old_loss = current.loss
            x, current = new_x, new_value
            record = IterationRecord(
                iteration,
                current.loss,
                _norm(current.gradient),
                alpha,
                calls,
                len(pairs),
                line_search_kind,
            )
            history.append(record)
            lifecycle.accepted(iteration, current.loss)
            if observe is not None:
                observe(record)
            if save_checkpoint is not None:
                saved = snapshot()
                assert saved is not None
                save_checkpoint(saved)
            if _norm(current.gradient) <= selected.gradient_tolerance:
                reason = "gradient_tolerance"
                break
            if _norm(step) <= selected.step_tolerance * max(1.0, _norm(x)):
                reason = "step_tolerance"
                break
            if abs(old_loss - current.loss) <= selected.relative_loss_tolerance * max(
                1.0, abs(old_loss), abs(current.loss)
            ):
                reason = "loss_stagnation"
                break
        return RecoveryResult(
            x,
            current,
            reason,
            calls,
            tuple(history),
            tuple(rejected_trials),
            resolution_limited_trials,
            snapshot(),
        )

    except Exception as error:
        error.__dict__["recovery_diagnostics"] = RecoveryResult(
            x,
            current,
            "evaluation_failed",
            calls,
            tuple(history),
            tuple(rejected_trials),
            resolution_limited_trials,
            snapshot(),
        )
        error.add_note("Last accepted recovery state is in recovery_diagnostics.")
        raise
