"""Independent finite-dimensional algebra checks, not transport evidence."""

import math

import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.stochastic_models import Matrix, quadratic_proposal


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


def product(matrix: Matrix, vector: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(math.fsum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix)


@pytest.mark.parametrize("size", [1, 2, 8])
def test_dense_positive_quadratic_has_exact_interior_step(size: int) -> None:
    curvature = tuple(
        tuple((i + 1.0 if i == j else 0.0) + 0.1 for j in range(size)) for i in range(size)
    )
    point = tuple((i + 1) / size for i in range(size))
    gradient = product(curvature, point)
    solved = quadratic_proposal(gradient, curvature, 10.0)
    assert close(solved.step, tuple(-value for value in point), abs=2e-13)
    assert not solved.boundary
    assert solved.rank == size
    assert solved.damping == 0.0
    assert close(
        solved.predicted_decrease,
        0.5 * math.fsum((a * b for a, b in zip(point, gradient, strict=True))),
    )


def test_clipped_scalar_quadratic_and_saved_absorption_correction() -> None:
    clipped = quadratic_proposal((2.0,), ((4.0,),), 0.1)
    assert close(clipped.step, (-0.1,))
    assert close(clipped.predicted_decrease, 0.18)
    assert clipped.boundary
    corrected = quadratic_proposal((0.000030462662138244425,), ((0.01641861973683762,),), 0.05)
    assert close(corrected.step, (-0.0018553729014075954,))
    assert not corrected.boundary


@pytest.mark.parametrize("size", [2, 8])
def test_dense_boundary_solution_satisfies_independent_kkt_conditions(size: int) -> None:
    matrix = tuple(
        tuple((i + 1.0 if i == j else 0.0) + 0.2 for j in range(size)) for i in range(size)
    )
    gradient = tuple(i + 0.5 for i in range(size))
    solved = quadratic_proposal(gradient, matrix, 0.25)
    assert close(math.hypot(*solved.step), 0.25, abs=1e-14)
    residual = tuple(g + bs for g, bs in zip(gradient, product(matrix, solved.step), strict=True))
    multipliers = tuple(-r / s for r, s in zip(residual, solved.step, strict=True))
    assert min(multipliers) > 0
    assert max(multipliers) - min(multipliers) < 1e-11


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_common_scalar_scaling_preserves_fixed_chart_solution(scale: float) -> None:
    solved = quadratic_proposal((scale, 2 * scale), ((3 * scale, scale), (scale, 2 * scale)), 0.2)
    reference = quadratic_proposal((1.0, 2.0), ((3.0, 1.0), (1.0, 2.0)), 0.2)
    assert close(solved.step, reference.step, rel=1e-12)
    assert close(solved.predicted_decrease / scale, reference.predicted_decrease)


def test_zero_tiny_and_rank_deficient_metrics_report_actual_rank() -> None:
    zero = quadratic_proposal((3.0, 4.0), ((0.0, 0.0), (0.0, 0.0)), 0.5)
    assert close(zero.step, (-0.3, -0.4))
    assert zero.rank == 0 and zero.condition is None and zero.solver == "cauchy_zero"
    tiny = quadratic_proposal((1.0,), ((1e-300,),), 0.1)
    assert close(tiny.step, (-0.1,))
    assert tiny.boundary and tiny.rank == 1
    rank = quadratic_proposal((1.0, 0.0), ((1.0, 0.0), (0.0, 0.0)), 2.0)
    assert close(rank.step, (-1.0, 0.0), rel=2e-12)
    assert rank.rank == 1 and rank.condition is None and rank.damping > 0
    assert rank.solver == "damped_eigen"


@pytest.mark.parametrize(
    "matrix, message",
    [
        (((-1.0,),), "indefinite"),
        (((1.0, 2.0), (2.0, 1.0)), "indefinite"),
        (((1.0, 0.0), (1.0, 1.0)), "symmetric"),
        (((float("nan"),),), "nonfinite"),
    ],
)
def test_invalid_metric_cannot_masquerade_as_a_successful_solve(
    matrix: Matrix, message: str
) -> None:
    with pytest.raises(NumericalError, match=message):
        quadratic_proposal((1.0,) * len(matrix), matrix, 1.0)


def test_dense_capacity_is_explicit() -> None:
    with pytest.raises(ContractError, match="16"):
        quadratic_proposal((1.0,) * 17, ((0.0,) * 17,) * 17, 1.0)
