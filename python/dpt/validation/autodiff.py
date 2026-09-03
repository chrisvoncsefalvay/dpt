"""CPU-only, scale-aware directional derivative reports for independent checks."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DirectionalRecord:
    step: float
    numerical: float
    predicted: float
    absolute_error: float
    relative_error: float
    scheme: str


def directional_sweep(
    evaluate: Callable[[tuple[float, ...]], float],
    point: Sequence[float],
    gradient: Sequence[float],
    direction: Sequence[float],
    steps: Sequence[float],
    *,
    scales: Sequence[float] | None = None,
    admissible: Callable[[tuple[float, ...]], bool] | None = None,
) -> tuple[DirectionalRecord, ...]:
    """D_d f(x) = sum_i gradient[i] * direction[i] * scales[i].

    Compare this supplied directional derivative with independently evaluated
    finite differences. ``point``, ``gradient``, ``direction`` and optional
    ``scales`` are length-n sequences with n > 0, while ``steps`` is a sequence
    of positive dimensionless perturbation sizes. Points and scales use the
    coordinate's physical units and gradient entries use objective units per
    coordinate unit. Direction entries are dimensionless and scales default to
    one. Python binary64 arithmetic returns one ``DirectionalRecord`` per step,
    with numerical and predicted derivatives in objective units and a
    dimensionless relative error.

    Both admissible perturbations give a central difference. If only one is
    admissible, its forward or backward difference is labelled in the record.
    The callback defines the active coordinates and supplies every objective
    evaluation, so the check does not reuse an operator's derivative routine.

        >>> from dpt.validation.autodiff import directional_sweep
        >>> records = directional_sweep(
        ...     lambda x: x[0]**2 + 3*x[1], (2., 1.), (4., 3.),
        ...     (1., 0.), (0.125, 0.0625), scales=(0.5, 2.))
        >>> [(r.numerical, r.predicted, r.scheme) for r in records]
        [(2.0, 2.0, 'central'), (2.0, 2.0, 'central')]

    """
    count = len(point)
    if not count or len(gradient) != count or len(direction) != count:
        raise ValueError("point, gradient and direction must have equal nonzero length")
    units = tuple(scales) if scales is not None else (1.0,) * count
    if len(units) != count or any(not math.isfinite(x) or x <= 0 for x in units):
        raise ValueError("scales must be positive finite coordinate units")
    if any(not math.isfinite(x) for values in (point, gradient, direction) for x in values):
        raise ValueError("directional inputs must be finite")
    delta = tuple(direction[i] * units[i] for i in range(count))
    if not any(x != 0 for x in delta):
        raise ValueError("direction must be nonzero")
    predicted = math.fsum(gradient[i] * delta[i] for i in range(count))
    baseline = float(evaluate(tuple(point)))
    if not math.isfinite(baseline):
        raise ValueError("baseline evaluation is non-finite")
    records: list[DirectionalRecord] = []
    for step in steps:
        if not math.isfinite(step) or step <= 0:
            raise ValueError("steps must be positive finite values")
        plus = tuple(point[i] + step * delta[i] for i in range(count))
        minus = tuple(point[i] - step * delta[i] for i in range(count))
        valid_plus, valid_minus = (
            (True, True) if admissible is None else (admissible(plus), admissible(minus))
        )
        if valid_plus and valid_minus:
            numerical = (evaluate(plus) - evaluate(minus)) / (2 * step)
            scheme = "central"
        elif valid_plus:
            numerical = (evaluate(plus) - baseline) / step
            scheme = "forward"
        elif valid_minus:
            numerical = (baseline - evaluate(minus)) / step
            scheme = "backward"
        else:
            raise ValueError(f"neither perturbation is admissible at step {step}")
        if not math.isfinite(numerical):
            raise ValueError("perturbed evaluation produced a non-finite derivative")
        error = abs(numerical - predicted)
        denominator = max(abs(numerical), abs(predicted), math.ulp(0.0))
        records.append(
            DirectionalRecord(step, numerical, predicted, error, error / denominator, scheme)
        )
    return tuple(records)
