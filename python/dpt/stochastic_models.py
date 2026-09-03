"""Bounded host solves for a proposal metric in the fixed Euclidean log chart.

Curvature is a PSD proposal metric, not a certified Hessian. Common scalar
normalisation protects the solve without changing either parameter coordinates
or the trust-region norm. The at-most-16-dimensional Jacobi eigensolve and
secular solve allocate only small host objects; they never handle detector data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dpt.contracts import ContractError, NumericalError, finite_scalar
from dpt.registration import Vector

Matrix = tuple[Vector, ...]


@dataclass(frozen=True, slots=True)
class QuadraticProposal:
    step: Vector
    predicted_decrease: float
    boundary: bool
    rank: int
    condition: float | None
    damping: float
    curvature_scale: float
    solver: str
    eigenvalues: Vector


def _eigensystem(matrix: Matrix) -> tuple[Vector, Matrix]:
    """Symmetric Jacobi rotations on a matrix already scaled to unit magnitude."""
    size = len(matrix)
    work = [list(row) for row in matrix]
    vectors = [[float(i == j) for j in range(size)] for i in range(size)]
    for _ in range(max(1, 100 * size * size)):
        pairs = ((abs(work[i][j]), i, j) for i in range(size) for j in range(i + 1, size))
        off, p, q = max(pairs, default=(0.0, 0, 0))
        if off <= 8 * math.ulp(1.0):
            return tuple(work[i][i] for i in range(size)), tuple(tuple(row) for row in vectors)
        # atan2 avoids the unstable division in the textbook tangent formula.
        angle = 0.5 * math.atan2(2 * work[p][q], work[q][q] - work[p][p])
        c, s = math.cos(angle), math.sin(angle)
        pp, qq, pq = work[p][p], work[q][q], work[p][q]
        work[p][p] = c * c * pp - 2 * s * c * pq + s * s * qq
        work[q][q] = s * s * pp + 2 * s * c * pq + c * c * qq
        work[p][q] = work[q][p] = 0.0
        for i in range(size):
            if i != p and i != q:
                ip, iq = work[i][p], work[i][q]
                work[i][p] = work[p][i] = c * ip - s * iq
                work[i][q] = work[q][i] = s * ip + c * iq
            vp, vq = vectors[i][p], vectors[i][q]
            vectors[i][p] = c * vp - s * vq
            vectors[i][q] = s * vp + c * vq
    raise NumericalError("small symmetric eigensolve did not converge")


def _metric_eigensystem(curvature: Matrix, size: int) -> tuple[float, Vector, Matrix]:
    if not 1 <= size <= 16 or len(curvature) != size or any(len(row) != size for row in curvature):
        raise ContractError("dense stochastic models require a square chart of 1 to 16 parameters")
    if not all(math.isfinite(value) for row in curvature for value in row):
        raise NumericalError("proposal curvature is nonfinite")
    scale = max(abs(value) for row in curvature for value in row)
    if scale == 0:
        identity = tuple(tuple(float(i == j) for j in range(size)) for i in range(size))
        return scale, (0.0,) * size, identity
    if any(
        abs(curvature[i][j] / scale - curvature[j][i] / scale) > 1e-12
        for i in range(size)
        for j in range(i)
    ):
        raise NumericalError("proposal curvature is not symmetric")
    scaled = tuple(
        tuple(0.5 * (curvature[i][j] / scale + curvature[j][i] / scale) for j in range(size))
        for i in range(size)
    )
    eigen, vectors = _eigensystem(scaled)
    if min(eigen) < -64 * size * math.ulp(1.0):
        raise NumericalError("proposal curvature is indefinite")
    if not all(math.isfinite(value * scale) for value in eigen):
        raise NumericalError("curvature spectrum exceeds finite range")
    return scale, tuple(max(value, 0.0) for value in eigen), vectors


def validate_curvature(curvature: Matrix, size: int) -> None:
    """B = B^T and lambda_min(B) >= 0 define an admissible proposal metric.

    Curvature is a size-by-size tuple of finite binary64 rows with 1 <= size <= 16.
    Its entries have objective units per pair of fixed chart coordinates. This
    host check returns None and raises for an asymmetric, indefinite or nonfinite
    matrix, including when the estimated gradient is already small. The metric
    is used to construct a proposal and need not equal the objective Hessian.

    >>> from dpt.stochastic_models import validate_curvature
    >>> validate_curvature(((2.0, 0.0), (0.0, 1.0)), 2)

    """
    _metric_eigensystem(curvature, size)


def quadratic_reduction(gradient: Vector, curvature: Matrix, step: Vector) -> float:
    """Predicted decrease is -g^T*s - s^T*B*s/2 at the actual chart displacement s.

    Gradient and step are length-D binary64 tuples and curvature is a D-by-D
    tuple of rows. Step uses the same fixed parameter chart as the gradient and
    metric, giving a scalar result in objective units. The returned decrease
    must be finite and positive. The host calculation uses accurate scalar sums
    and raises NumericalError when the proposed displacement does not decrease
    the declared model.

    >>> from dpt.stochastic_models import quadratic_reduction
    >>> quadratic_reduction((2.0,), ((2.0,),), (-1.0,))
    1.0

    """
    linear = math.fsum(-(g * s) for g, s in zip(gradient, step, strict=True))
    quadratic = math.fsum(
        step[i] * math.fsum(curvature[i][j] * step[j] for j in range(len(step)))
        for i in range(len(step))
    )
    predicted = linear - 0.5 * quadratic
    if not math.isfinite(predicted) or predicted <= 0:
        raise NumericalError("quadratic predicted decrease is not positive finite")
    return predicted


def quadratic_proposal(
    gradient: Vector,
    curvature: Matrix,
    radius: float,
    *,
    damping_relative: float = 1e-12,
    rank_tolerance: float = 1e-10,
) -> QuadraticProposal:
    """s = argmin_{||s||<=radius} g^T*s + s^T*(B+damping*I)*s/2.

    Gradient is a length-D finite binary64 tuple and curvature is a symmetric
    positive-semidefinite D-by-D metric with 1 <= D <= 16. Radius is a positive
    scalar in fixed chart units. The returned QuadraticProposal contains the
    length-D step, predicted decrease under the original undamped metric,
    boundary flag and spectrum diagnostics. The damping floor is
    damping_relative*lambda_max and rank is measured before damping using
    rank_tolerance. The small host eigensolve and secular solve use binary64
    arithmetic. Coordinate scales must already be reflected in g, B and radius.

    >>> from dpt.stochastic_models import quadratic_proposal
    >>> proposal = quadratic_proposal((2.0,), ((2.0,),), radius=2.0)
    >>> (proposal.step, proposal.predicted_decrease, proposal.boundary)
    ((-1.0,), 1.0, False)

    """
    size = len(gradient)
    scale, positive, vectors = _metric_eigensystem(curvature, size)
    finite_scalar(radius, "radius", minimum=0.0)
    finite_scalar(damping_relative, "damping_relative", minimum=0.0)
    finite_scalar(rank_tolerance, "rank_tolerance", minimum=0.0)
    if radius == 0 or not 0 < rank_tolerance < 1:
        raise ContractError("radius must be positive and rank tolerance must lie in (0,1)")
    norm = math.hypot(*gradient)
    if norm == 0 or not math.isfinite(norm):
        raise NumericalError("quadratic proposal requires a positive finite gradient norm")
    if scale == 0:
        step = tuple(-radius * (value / norm) for value in gradient)
        predicted = radius * norm
        if not math.isfinite(predicted) or predicted <= 0:
            raise NumericalError("Cauchy prediction is not positive finite")
        return QuadraticProposal(
            step, predicted, True, 0, None, 0.0, 0.0, "cauchy_zero", (0.0,) * size
        )
    maximum = max(positive)
    rank = sum(value > rank_tolerance * maximum for value in positive)
    condition = maximum / min(positive) if rank == size else None
    damping_scaled = max(0.0, damping_relative * maximum - min(positive))
    # Scaling the gradient separately avoids g/B overflow in a tiny-curvature
    # model. The secular multiplier is measured in units max(B, ||g||/radius).
    solve_scale = max(scale, norm / radius)
    if not math.isfinite(solve_scale):
        raise NumericalError("quadratic solve scale exceeds finite range")
    relative_scale = scale / solve_scale
    projected = tuple(
        math.fsum(vectors[i][j] * (gradient[i] / solve_scale) for i in range(size))
        for j in range(size)
    )
    diagonal = tuple((value + damping_scaled) * relative_scale for value in positive)

    def coordinates(multiplier: float) -> Vector | None:
        result: list[float] = []
        for value, entry in zip(projected, diagonal, strict=True):
            denominator = entry + multiplier
            if denominator == 0:
                if value != 0:
                    return None
                result.append(0.0)
            else:
                result.append(-value / denominator)
        return tuple(result)

    solution = coordinates(0.0)
    boundary = solution is None or math.hypot(*solution) >= radius
    if boundary:
        low, high = 0.0, math.hypot(*projected) / radius
        # lambda=||projected gradient||/radius is feasible and avoids
        # losing extremely small multipliers to a unit-sized bisection interval.
        for _ in range(120):
            middle = 0.5 * (low + high)
            trial = coordinates(middle)
            if trial is None or math.hypot(*trial) > radius:
                low = middle
            else:
                high = middle
        solution = coordinates(high)
    if solution is None:
        raise NumericalError("quadratic secular solve failed")
    step = tuple(math.fsum(vectors[i][j] * solution[j] for j in range(size)) for i in range(size))
    # Roundoff in the orthogonal transform may exceed the declared ball by ulps.
    step_norm = math.hypot(*step)
    if step_norm > radius:
        step = tuple(value * (radius / step_norm) for value in step)
    predicted = quadratic_reduction(gradient, curvature, step)
    return QuadraticProposal(
        step,
        predicted,
        boundary,
        rank,
        condition,
        damping_scaled * scale,
        scale,
        "damped_eigen" if damping_scaled else "eigen",
        tuple(value * scale for value in positive),
    )
