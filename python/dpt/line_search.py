"""Safeguarded cubic interpolation and bounded strong-Wolfe searches.

The callback returns the loss and the derivative along one feasible search ray.
Only TrialDomainError rejects a trial. NumericalError remains a fatal evaluation
failure. A boundary Armijo step is labelled separately from a Wolfe step.
"""


# Public Python boundaries validate runtime inputs.
# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from dpt.contracts import ContractError, NumericalError, TrialDomainError, finite_scalar, integer


@dataclass(frozen=True, slots=True)
class ScalarEvaluation:
    loss: float
    derivative: float


@dataclass(frozen=True, slots=True)
class CubicSample:
    step: float
    evaluation: ScalarEvaluation


def _checked(value: ScalarEvaluation) -> ScalarEvaluation:
    if not isinstance(value, ScalarEvaluation):
        raise ContractError("line-search callback must return ScalarEvaluation")
    if not math.isfinite(value.loss) or not math.isfinite(value.derivative):
        raise NumericalError("non-finite line-search loss or directional derivative")
    return value


def cubic_trial(left: CubicSample, right: CubicSample, *, safeguard: float = 0.1) -> float:
    """h(a)=f_a, h'(a)=d_a, h(b)=f_b and h'(b)=d_b define a Hermite cubic.

    Return its interior local minimum within the fraction safeguard of both
    bracket ends, or the midpoint when no representable safeguarded minimum
    exists. Each CubicSample holds a scalar binary64 step and ScalarEvaluation.
    Loss has objective units and derivative has objective units per step unit.
    The scalar result uses the same step units. Reversed and negative brackets
    are valid, while equal endpoints are rejected. This interpolation constructs
    a trial without evaluating the underlying active model parameters.

    >>> from dpt.line_search import CubicSample, ScalarEvaluation, cubic_trial
    >>> left = CubicSample(0.0, ScalarEvaluation(1.0, -2.0))
    >>> right = CubicSample(2.0, ScalarEvaluation(1.0, 2.0))
    >>> cubic_trial(left, right)
    1.0

    """
    finite_scalar(left.step, "left step")
    finite_scalar(right.step, "right step")
    _checked(left.evaluation)
    _checked(right.evaluation)
    guard = finite_scalar(safeguard, "safeguard", minimum=0.0)
    if not 0 <= guard < 0.5 or left.step == right.step:
        raise ContractError("cubic interpolation needs distinct endpoints and 0 <= safeguard < 0.5")
    low, high = sorted((left.step, right.step))
    midpoint = low * 0.5 + high * 0.5
    width = right.step - left.step
    first, second = left.evaluation, right.evaluation
    # Scale before taking differences, retaining small objectives and preventing
    # cancellation-free arithmetic from overflowing for large finite losses.
    scale = max(abs(first.loss), abs(second.loss), abs(first.derivative), abs(second.derivative))
    if scale == 0 or not math.isfinite(width):
        return midpoint
    f0, f1 = first.loss / scale, second.loss / scale
    d0, d1 = (first.derivative / scale) * width, (second.derivative / scale) * width
    if not math.isfinite(d0) or not math.isfinite(d1):
        return midpoint
    a = 2 * (f0 - f1) + d0 + d1
    b = 3 * (f1 - f0) - 2 * d0 - d1
    roots: list[float] = []
    if a == 0:
        if b > 0:
            roots.append(-d0 / (2 * b))
    else:
        discriminant = b * b - 3 * a * d0
        if discriminant >= 0 and math.isfinite(discriminant):
            root = math.sqrt(discriminant)
            # Avoid subtraction of almost equal quantities in the quadratic.
            q = -b - math.copysign(root, b)
            if q != 0:
                roots.extend((q / (3 * a), d0 / q))
            else:
                roots.append(-b / (3 * a))
    minima = [t for t in roots if guard <= t <= 1 - guard and 6 * a * t + 2 * b > 0]
    if not minima:
        return midpoint
    chosen = min(minima, key=lambda t: ((a * t + b) * t + d0) * t + f0)
    trial = left.step * (1 - chosen) + right.step * chosen
    return trial if low < trial < high else midpoint


@dataclass(frozen=True, slots=True)
class CubicWolfePolicy:
    armijo: float = 1e-4
    curvature: float = 0.9
    maximum_step: float = 16.0
    minimum_step: float = 1e-14
    max_evaluations: int = 32
    safeguard: float = 0.1

    def __post_init__(self) -> None:
        integer(self.max_evaluations, "max_evaluations", minimum=1)
        for name in ("armijo", "curvature", "maximum_step", "minimum_step", "safeguard"):
            finite_scalar(getattr(self, name), name, minimum=0.0)
        if not 0 < self.armijo < self.curvature < 1:
            raise ContractError("Wolfe constants must satisfy 0 < c1 < c2 < 1")
        if not 0 < self.minimum_step < self.maximum_step or not 0 <= self.safeguard < 0.5:
            raise ContractError("invalid cubic-Wolfe step interval or safeguard")


@dataclass(frozen=True, slots=True)
class CubicWolfeResult:
    step: float
    evaluation: ScalarEvaluation
    reason: Literal["wolfe", "armijo_bound", "failed", "evaluation_budget", "not_descent"]
    evaluations: int
    rejected: tuple[tuple[float, str], ...] = ()

    @property
    def accepted(self) -> bool:
        return self.reason in ("wolfe", "armijo_bound")


def cubic_strong_wolfe(
    evaluate: Callable[[float], ScalarEvaluation],
    base: ScalarEvaluation,
    *,
    initial_step: float = 1.0,
    feasible_maximum: float | None = None,
    policy: CubicWolfePolicy | None = None,
) -> CubicWolfeResult:
    """f(a) <= f(0)+c1*a*f'(0) and |f'(a)| <= c2*|f'(0)| define a Wolfe step.

    Evaluate receives one scalar binary64 step and returns ScalarEvaluation
    with scalar loss and derivative along the chosen feasible parameter ray.
    Base is its evaluation at zero. Initial_step, feasible_maximum and the
    returned step share the ray's units. Policy controls the Wolfe constants,
    bracket safeguards and evaluation budget. The result retains the supplied
    base on failure. An actual feasible boundary may accept strict Armijo
    decrease without curvature and records armijo_bound. TrialDomainError
    contracts the bracket, while other numerical failures propagate.

    >>> from dpt.line_search import ScalarEvaluation, cubic_strong_wolfe
    >>> def evaluate(step):
    ...     return ScalarEvaluation((step - 1.0) ** 2, 2.0 * (step - 1.0))
    >>> result = cubic_strong_wolfe(evaluate, evaluate(0.0))
    >>> (result.step, result.reason, result.evaluation.loss)
    (1.0, 'wolfe', 0.0)

    """
    rules = CubicWolfePolicy() if policy is None else policy
    base = _checked(base)
    initial_step = finite_scalar(initial_step, "initial_step", minimum=0.0)
    if initial_step == 0:
        raise ContractError("initial_step must be positive")
    limit = rules.maximum_step
    boundary = False
    if feasible_maximum is not None:
        feasible_maximum = finite_scalar(feasible_maximum, "feasible_maximum", minimum=0.0)
        boundary = feasible_maximum <= limit
        limit = min(limit, feasible_maximum)
    if base.derivative >= 0:
        return CubicWolfeResult(0.0, base, "not_descent", 0)
    attempts = 0
    rejected: list[tuple[float, str]] = []

    def finish(
        step: float,
        value: ScalarEvaluation,
        reason: Literal["wolfe", "armijo_bound", "failed", "evaluation_budget", "not_descent"],
    ) -> CubicWolfeResult:
        return CubicWolfeResult(step, value, reason, attempts, tuple(rejected))

    def trial(step: float) -> ScalarEvaluation | None:
        nonlocal attempts
        attempts += 1
        try:
            return _checked(evaluate(step))
        except TrialDomainError as error:
            rejected.append((step, str(error)))
            return None

    def armijo(step: float, value: ScalarEvaluation) -> bool:
        return (
            value.loss < base.loss
            and value.loss <= base.loss + rules.armijo * step * base.derivative
        )

    def zoom(low: CubicSample, high_step: float, high: ScalarEvaluation | None) -> CubicWolfeResult:
        while attempts < rules.max_evaluations:
            if abs(high_step - low.step) < rules.minimum_step:
                return finish(0.0, base, "failed")
            step = (
                cubic_trial(low, CubicSample(high_step, high), safeguard=rules.safeguard)
                if high is not None
                else low.step * 0.5 + high_step * 0.5
            )
            if step == low.step or step == high_step:
                return finish(0.0, base, "failed")
            value = trial(step)
            if value is None or not armijo(step, value) or value.loss >= low.evaluation.loss:
                high_step, high = step, value
                continue
            if abs(value.derivative) <= -rules.curvature * base.derivative:
                return finish(step, value, "wolfe")
            if value.derivative * (high_step - low.step) >= 0:
                high_step, high = low.step, low.evaluation
            low = CubicSample(step, value)
        return finish(0.0, base, "evaluation_budget")

    if limit < rules.minimum_step:
        return finish(0.0, base, "failed")
    previous = CubicSample(0.0, base)
    step = min(initial_step, limit)
    while attempts < rules.max_evaluations:
        value = trial(step)
        if (
            value is None
            or not armijo(step, value)
            or (previous.step > 0 and value.loss >= previous.evaluation.loss)
        ):
            return zoom(previous, step, value)
        if abs(value.derivative) <= -rules.curvature * base.derivative:
            return finish(step, value, "wolfe")
        if step == limit and boundary:
            return finish(step, value, "armijo_bound")
        if value.derivative >= 0:
            return zoom(CubicSample(step, value), previous.step, previous.evaluation)
        if step == limit:
            return finish(0.0, base, "failed")
        previous = CubicSample(step, value)
        step = min(step * 2, limit)
    return finish(0.0, base, "evaluation_budget")
