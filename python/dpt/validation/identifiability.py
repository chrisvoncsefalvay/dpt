"""Bounded offline sensitivity diagnostics with explicit parameter and noise units.

The caller supplies a small Jacobian from a declared diagnostic experiment. This
module does not download images or retain Jacobians inside a production solve.
NumPy, supplied by the optional scientific environment, is loaded only on use.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dpt.contracts import ContractError, NumericalError, finite_scalar


@dataclass(frozen=True, slots=True)
class SensitivitySpectrum:
    singular_values: tuple[float, ...]
    right_directions: tuple[tuple[float, ...], ...]
    rank: int
    threshold: float
    condition_number: float | None
    observations: int
    parameters: int


def _scaled_entry(value: float, unit: float, weight: float) -> float:
    value = finite_scalar(value, "Jacobian entry")
    root_weight = math.sqrt(finite_scalar(weight, "precision weight", minimum=0))
    if value == 0 or root_weight == 0:
        return 0.0
    mantissa, exponent = 1.0, 0
    for factor in (value, unit, root_weight):
        coefficient, power = math.frexp(factor)
        mantissa *= coefficient
        exponent += power
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError as error:
        raise NumericalError("scaled sensitivity exceeds binary64 range") from error


def sensitivity_spectrum(
    jacobian_rows: Sequence[Sequence[float]],
    parameter_scales: Sequence[float],
    *,
    precision_weights: Sequence[float] | None = None,
    relative_threshold: float = 1e-8,
    absolute_threshold: float = 0.0,
) -> SensitivitySpectrum:
    """Compute ``sqrt(W) J S = U diag(singular_values) V.T``.

    ``jacobian_rows`` has shape ``(observations, parameters)``, bounded to
    4,096 observations and 32 parameters. ``parameter_scales`` contains one
    positive physical scale per parameter. Optional non-negative diagonal
    ``precision_weights`` contains one weight per observation and defaults
    to ones. Right directions have ``(parameters, parameters)`` shape and
    use dimensionless chart coordinates, including zero directions when
    there are fewer observations than parameters.

    Singular values and an explicitly chosen threshold describe local sensitivity,
    not global identifiability or clinical recovery. Duplicate views add no new
    independent directions, although weighting can change threshold-defined rank.
    Noise correlations require an explicitly prewhitened Jacobian; diagonal
    precision weights must not stand in for an unknown covariance model.

    Example:
        >>> result = sensitivity_spectrum(((1.0, 0.0), (0.0, 2.0)), (1.0, 1.0))
        >>> result.singular_values, result.rank, result.condition_number
        ((2.0, 1.0), 2, 2.0)

    """
    rows, columns = len(jacobian_rows), len(parameter_scales)
    if not 1 <= rows <= 4096 or not 1 <= columns <= 32:
        raise ContractError("offline diagnostic is bounded to 4096 observations and 32 parameters")
    if any(len(row) != columns for row in jacobian_rows):
        raise ContractError("every Jacobian row must contain each active parameter")
    units = tuple(finite_scalar(value, "parameter scale", minimum=0) for value in parameter_scales)
    if min(units) == 0:
        raise ContractError("parameter scales must be positive")
    relative = finite_scalar(relative_threshold, "relative threshold", minimum=0)
    absolute = finite_scalar(absolute_threshold, "absolute threshold", minimum=0)
    if relative >= 1:
        raise ContractError("relative rank threshold must be less than one")
    weights = tuple(precision_weights) if precision_weights is not None else (1.0,) * rows
    if len(weights) != rows:
        raise ContractError("one fixed precision weight is required per observation")
    scaled = [
        [_scaled_entry(value, units[column], weights[index]) for column, value in enumerate(row)]
        for index, row in enumerate(jacobian_rows)
    ]
    if not all(math.isfinite(value) for row in scaled for value in row):
        raise NumericalError("scaled sensitivity exceeds binary64 range")
    # Zero rows preserve the missing directions when observations < parameters,
    # without requesting the large square left-singular-vector matrix.
    scaled.extend([[0.0] * columns for _ in range(max(0, columns - rows))])
    np: Any = importlib.import_module("numpy")
    _, values, right = np.linalg.svd(np.asarray(scaled, dtype=np.float64), full_matrices=False)
    singular = tuple(float(value) for value in values)
    if not all(map(math.isfinite, singular)):
        raise NumericalError("sensitivity decomposition produced nonfinite singular values")
    threshold = max(absolute, relative * singular[0])
    rank = sum(value > threshold for value in singular)
    condition = singular[0] / singular[-1] if rank == columns else None
    if condition is not None and not math.isfinite(condition):
        condition = None
    return SensitivitySpectrum(
        singular,
        tuple(tuple(float(value) for value in row) for row in right),
        rank,
        threshold,
        condition,
        rows,
        columns,
    )
