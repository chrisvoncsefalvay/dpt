"""Dense finite-difference and convex certificate references for tiny fixtures."""

from decimal import Decimal, localcontext

import numpy as np
from scipy.optimize import LinearConstraint, minimize


def decimal_tv_value_and_gradient(
    values: np.ndarray,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    tv: float,
    isotropic: bool,
) -> tuple[float, np.ndarray]:
    """Independent physical edge model with 1200-digit reference arithmetic."""
    nz, ny, nx = shape
    with localcontext() as context:
        context.prec = 1200
        samples = [Decimal.from_float(float(value)) for value in values]
        step = [Decimal.from_float(value) for value in spacing]
        weight = Decimal.from_float(tv)
        gradient = [Decimal(0) for _ in values]
        total = Decimal(0)
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    origin = (z * ny + y) * nx + x
                    targets = (
                        origin + 1 if x + 1 < nx else None,
                        origin + nx if y + 1 < ny else None,
                        origin + nx * ny if z + 1 < nz else None,
                    )
                    difference = [
                        (samples[target] - samples[origin]) / step[axis]
                        if target is not None
                        else Decimal(0)
                        for axis, target in enumerate(targets)
                    ]
                    norm = (
                        sum((component * component for component in difference), Decimal(0)).sqrt()
                        if isotropic
                        else sum((abs(component) for component in difference), Decimal(0))
                    )
                    total += weight * norm
                    for axis, target in enumerate(targets):
                        if target is None or difference[axis] == 0:
                            continue
                        direction = (
                            difference[axis] / norm
                            if isotropic
                            else (Decimal(1) if difference[axis] > 0 else Decimal(-1))
                        )
                        edge = weight * direction / step[axis]
                        gradient[origin] -= edge
                        gradient[target] += edge
        return float(total), np.asarray([float(component) for component in gradient])


def difference_matrix(
    shape: tuple[int, int, int], spacing: tuple[float, float, float]
) -> np.ndarray:
    nz, ny, nx = shape
    count = nz * ny * nx
    matrix = np.zeros((count, 3, count))
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                index = (z * ny + y) * nx + x
                for axis, neighbour in enumerate(
                    (
                        (x + 1 < nx, index + 1),
                        (y + 1 < ny, index + nx),
                        (z + 1 < nz, index + nx * ny),
                    )
                ):
                    exists, target = neighbour
                    if exists:
                        matrix[index, axis, index] = -1 / spacing[axis]
                        matrix[index, axis, target] = 1 / spacing[axis]
    return matrix


def penalty(values: np.ndarray, matrix: np.ndarray, tv: float, l1: float, isotropic: bool) -> float:
    difference = np.einsum("vaj,j->va", matrix, values)
    norms = np.linalg.norm(difference, axis=1) if isotropic else np.abs(difference).sum(axis=1)
    return float(tv * norms.sum() + l1 * np.abs(values).sum())


def certificate(
    values: np.ndarray,
    y: np.ndarray,
    dual: np.ndarray,
    matrix: np.ndarray,
    support: np.ndarray,
    tv: float,
    l1: float,
    isotropic: bool,
) -> tuple[float, float]:
    primal = 0.5 * float(np.dot(values - y, values - y)) + penalty(
        values, matrix, tv, l1, isotropic
    )
    transpose = np.einsum("vaj,va->j", matrix, dual)
    argument = y - transpose
    maximiser = np.sign(argument) * np.maximum(np.abs(argument) - l1, 0)
    maximiser[~support] = 0
    # Conjugate is independently evaluated at its scalar optimum rather than
    # repeating the production expanded energy formula.
    conjugate = -transpose * maximiser - 0.5 * (maximiser - y) ** 2 - l1 * np.abs(maximiser)
    return primal, float(-conjugate.sum())


def anisotropic_epigraph_optimum(
    y: np.ndarray, matrix: np.ndarray, support: np.ndarray, tv: float, l1: float
) -> np.ndarray:
    """Independent smooth QP with linear absolute-value epigraph constraints."""
    count = len(y)
    difference = matrix.reshape(-1, count)
    edges = len(difference)
    size = 2 * count + edges
    # Variables x[n], t[3n], u[n]; minimise quadratic + tv*t + l1*u.
    constraint = np.zeros((2 * edges + 2 * count, size))
    constraint[:edges, :count] = -difference
    constraint[edges : 2 * edges, :count] = difference
    constraint[:edges, count : count + edges] = np.eye(edges)
    constraint[edges : 2 * edges, count : count + edges] = np.eye(edges)
    constraint[2 * edges : 2 * edges + count, :count] = -np.eye(count)
    constraint[2 * edges + count :, :count] = np.eye(count)
    constraint[2 * edges : 2 * edges + count, count + edges :] = np.eye(count)
    constraint[2 * edges + count :, count + edges :] = np.eye(count)
    initial = y.copy()
    initial[~support] = 0
    start = np.concatenate((initial, np.abs(difference @ initial) + 0.1, np.abs(initial) + 0.1))

    def objective(point: np.ndarray) -> tuple[float, np.ndarray]:
        residual = point[:count] - y
        value = (
            0.5 * np.dot(residual, residual)
            + tv * point[count : count + edges].sum()
            + l1 * point[count + edges :].sum()
        )
        gradient = np.concatenate((residual, np.full(edges, tv), np.full(count, l1)))
        return float(value), gradient

    bounds = [(None, None) if inside else (0.0, 0.0) for inside in support] + [(0.0, None)] * (
        edges + count
    )
    fit = minimize(
        objective,
        start,
        jac=True,
        method="SLSQP",
        bounds=bounds,
        constraints=LinearConstraint(constraint, 0, np.inf),
        options={"ftol": 1e-12, "maxiter": 500},
    )
    assert fit.success, fit.message
    return fit.x[:count]
