"""Independent scale/order regressions for positive-semidefinite normal matrices."""

import math

import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.pose_optimization import NormalEquations, PoseLeastSquaresPolicy, recover_least_squares


@pytest.mark.parametrize("column_scale", [1e-6, 1e-7, 1e-8, 1e-100])
@pytest.mark.parametrize("reverse", [False, True])
def test_rank_one_jacobian_with_unequal_column_units_is_accepted(
    column_scale: float, reverse: bool
) -> None:
    # One observed residual, J=(column_scale, 1), supplies an exact rank-one
    # normal matrix. Permuting its columns cannot change PSD or solvability.
    jacobian = (1.0, column_scale) if reverse else (column_scale, 1.0)
    matrix = tuple(tuple(x * y for y in jacobian) for x in jacobian)

    def residual(point: tuple[float, ...]) -> NormalEquations:
        value = math.fsum(x * y for x, y in zip(jacobian, point, strict=True)) - 1
        return NormalEquations(0.5 * value**2, tuple(x * value for x in jacobian), matrix)

    result = recover_least_squares(
        residual, (0.0, 0.0), policy=PoseLeastSquaresPolicy(gradient_tolerance=1e-10)
    )
    assert result.stationary
    fitted = math.fsum(x * y for x, y in zip(jacobian, result.parameters, strict=True))
    assert abs(fitted - 1) < 1e-10


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200, 1e308])
@pytest.mark.parametrize("reverse", [False, True])
def test_uniform_scaled_psd_validation_never_squares_raw_matrix_entries(
    scale: float, reverse: bool
) -> None:
    # Outer product of (1,1) multiplied by a positive scalar. This remains PSD
    # despite raw-entry multiplication overflowing or underflowing.
    matrix = ((scale, scale), (scale, scale))
    if reverse:
        matrix = tuple(tuple(row[i] for i in (1, 0)) for row in reversed(matrix))
    result = recover_least_squares(lambda _: NormalEquations(0.0, (0.0, 0.0), matrix), (0.0, 0.0))
    assert result.stationary and result.evaluations == 1


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
@pytest.mark.parametrize("reverse", [False, True])
def test_uniform_scaled_indefinite_matrix_cannot_hide_below_absolute_tolerance(
    scale: float, reverse: bool
) -> None:
    # Eigenvectors (1,1),(1,-1) have eigenvalues 3*scale,-scale.
    matrix = ((scale, 2 * scale), (2 * scale, scale))
    if reverse:
        matrix = tuple(tuple(row[i] for i in (1, 0)) for row in reversed(matrix))
    with pytest.raises(ContractError, match="positive semidefinite"):
        recover_least_squares(lambda _: NormalEquations(0.0, (0.0, 0.0), matrix), (0.0, 0.0))


@pytest.mark.parametrize(
    "matrix",
    [
        ((0.0, 1e-200), (1e-200, 1.0)),
        ((-1e-200, 0.0), (0.0, 1.0)),
        ((1e200, 1e-100), (2e-100, 1.0)),
        ((1e-200, 1e200), (1e200, 1e-200)),
    ],
)
def test_zero_negative_asymmetric_and_unrepresentable_correlation_controls(
    matrix: tuple[tuple[float, ...], ...],
) -> None:
    with pytest.raises(ContractError):
        recover_least_squares(lambda _: NormalEquations(0.0, (0.0, 0.0), matrix), (0.0, 0.0))


def test_normalised_psd_pivoting_detects_higher_order_indefiniteness() -> None:
    # Every 2-by-2 principal block is PSD, but the (1,1,1) eigenvector has
    # negative eigenvalue -0.5. Pairwise Cauchy checks alone are insufficient.
    matrix = ((1.0, -0.75, -0.75), (-0.75, 1.0, -0.75), (-0.75, -0.75, 1.0))
    with pytest.raises(ContractError, match="positive semidefinite"):
        recover_least_squares(
            lambda _: NormalEquations(0.0, (0.0, 0.0, 0.0), matrix), (0.0, 0.0, 0.0)
        )


def test_unrepresentable_solver_directional_product_is_numerical_failure() -> None:
    with pytest.raises(NumericalError, match="inner product"):
        recover_least_squares(lambda _: NormalEquations(1.0, (1e200,), ((1.0,),)), (0.0,))
