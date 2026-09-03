"""Scalar progress and termination contracts shared by the recovery drivers.

Lifecycle observers receive committed controller states, without exporting
resident fields. Iteration zero is the verified starting state; subsequent
iterations count controller steps, including rejected steps. A rejected event
reports the incumbent objective and accepted-step count, not a trial objective.
Observers record progress: their return value does not stop or alter a solve.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, cast

from dpt.contracts import ContractError

TerminationCategory = Literal[
    "stationary", "stagnation", "budget", "cancelled", "failed", "uncertainty"
]
LifecycleStatus = Literal["initial", "resumed", "accepted", "rejected", "terminal"]

_REASON_CATEGORIES: dict[str, TerminationCategory] = {
    "gradient_tolerance": "stationary",
    "projected_gradient_tolerance": "stationary",
    "step_tolerance": "stagnation",
    "loss_stagnation": "stagnation",
    "loss_resolution": "stagnation",
    "radius_limit": "stagnation",
    "iteration_budget": "budget",
    "evaluation_budget": "budget",
    "history_budget": "budget",
    "cancelled": "cancelled",
    "line_search_failed": "failed",
    "damping_limit": "failed",
    "evaluation_failed": "failed",
    "exception": "failed",
    "gradient_band": "uncertainty",
    "sampling_unresolved": "uncertainty",
}


@dataclass(frozen=True, slots=True)
class SolverTermination:
    """Original solver reason and its common category.

    Only deterministic gradient criteria map to ``stationary``. Stochastic
    gradient bands map to ``uncertainty``; their calibration and final-validation
    details remain in the original result. Small-step, small-loss and radius
    limits map to ``stagnation``. Exhausted iteration/evaluation/history budgets
    map to ``budget``; failed searches, damping limits and exceptions to
    ``failed``. Cancellation retains its own category.
    """

    reason: str
    category: TerminationCategory


def termination_reason(reason: str) -> SolverTermination:
    """Normalise an existing reason without changing its numerical meaning."""
    try:
        return SolverTermination(reason, _REASON_CATEGORIES[reason])
    except KeyError as error:
        raise ContractError(f"unknown solver termination reason: {reason!r}") from error


def solver_termination(result: object) -> SolverTermination:
    """Read a recovery result or a material solver report through one interface."""
    reason: object = (
        cast(Mapping[str, object], result).get("termination")
        if isinstance(result, Mapping)
        else getattr(result, "reason", None)
    )
    if not isinstance(reason, str):
        raise ContractError("solver result must provide a string termination reason")
    return termination_reason(reason)


@dataclass(frozen=True, slots=True)
class SolverEvent:
    """Immutable scalar progress snapshot.

    ``objective`` is the last accepted loss, or None for a stochastic driver
    which estimates objective changes instead. ``detail`` identifies a rejected
    domain/search proposal or retains a terminal diagnostic. Checkpoint exports
    remain explicit and use each solver's existing checkpoint type.
    """

    solver: str
    status: LifecycleStatus
    iteration: int
    accepted_steps: int
    objective: float | None
    reason: str | None = None
    detail: str | None = None
    termination: SolverTermination | None = None


LifecycleObserver = Callable[[SolverEvent], None]


class LifecycleRecorder:
    """Per-call recorder; no global state, device work or implicit exports."""

    def __init__(self, solver: str, observer: LifecycleObserver | None) -> None:
        self.solver = solver
        self.observer = observer
        self.iteration = 0
        self.accepted_steps = 0
        self.objective: float | None = None
        self._observer_failed = False

    def __enter__(self) -> LifecycleRecorder:
        return self

    def event(
        self,
        status: LifecycleStatus,
        *,
        reason: str | None = None,
        detail: str | None = None,
    ) -> SolverEvent:
        return SolverEvent(
            self.solver,
            status,
            self.iteration,
            self.accepted_steps,
            self.objective,
            reason,
            detail,
            None if reason is None else termination_reason(reason),
        )

    def emit(self, status: LifecycleStatus, *, detail: str | None = None) -> None:
        if self.observer is not None:
            self._notify(self.event(status, detail=detail))

    def _notify(self, event: SolverEvent) -> None:
        assert self.observer is not None
        try:
            self.observer(event)
        except BaseException:
            self._observer_failed = True
            raise

    def start(
        self,
        *,
        objective: float | None,
        iteration: int = 0,
        accepted_steps: int = 0,
        resumed: bool = False,
    ) -> None:
        self.objective = objective
        self.iteration = iteration
        self.accepted_steps = accepted_steps
        self.emit("resumed" if resumed else "initial")

    def accepted(self, iteration: int, objective: float | None) -> None:
        self.iteration = iteration
        self.accepted_steps += 1
        self.objective = objective
        self.emit("accepted")

    def rejected(self, iteration: int, detail: str) -> None:
        self.iteration = iteration
        self.emit("rejected", detail=detail)

    def finish(self, result: object, *, detail: str | None = None) -> None:
        if self.observer is not None:
            reason = solver_termination(result).reason
            self._notify(self.event("terminal", reason=reason, detail=detail))

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if error is None:
            return
        event = self.event("terminal", reason="exception", detail=type(error).__name__)
        error.__dict__["solver_lifecycle"] = event
        if self.observer is not None and not self._observer_failed:
            try:
                self._notify(event)
            except BaseException as observer_error:
                error.add_note(
                    f"Terminal lifecycle observer raised {type(observer_error).__name__}."
                )
