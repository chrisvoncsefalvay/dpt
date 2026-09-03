"""Scalar moment range cases with independently known means and standard errors."""

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from dpt.contracts import NumericalError
from dpt.statistics import (
    all_pairs_product,
    all_pairs_squared_change,
    mean_standard_error,
    student_t_critical,
)


def test_raw_sum_overflow_does_not_destroy_a_finite_mean() -> None:
    assert mean_standard_error((1e308, 1e308)) == (1e308, 0)
    mean, error = mean_standard_error((-1.7e308, 1.7e308))
    assert mean == 0
    assert math.isclose(error, 1.7e308, rel_tol=1e-15)


def test_uniform_subnormal_mean_survives_replicate_normalisation() -> None:
    tiny = math.ulp(0.0)
    assert mean_standard_error((tiny,) * 16) == (tiny, 0)


def test_cancelling_extremes_do_not_erase_a_representable_small_mean() -> None:
    mean, error = mean_standard_error((1e308, -1e308, 1e-300))
    assert mean == 1e-300 / 3
    assert math.isfinite(error) and error > 0


def test_standard_error_centres_before_rounding_a_subnormal_mean() -> None:
    tiny = math.ulp(0.0)
    mean, error = mean_standard_error((0.0, 2 * tiny))
    assert mean == tiny
    assert error == tiny


def test_centred_deviation_retains_adjacent_large_values() -> None:
    left = 2.0**500
    right = left + 4 * math.ulp(left)
    mean, error = mean_standard_error((left, right))
    assert mean == left + 2 * math.ulp(left)
    assert math.isclose(error, 2 * math.ulp(left), rel_tol=1e-15)


@pytest.mark.parametrize(
    ("tail", "expected"),
    [(0.25, 1.0), (0.1, math.sqrt(5 + 2 * math.sqrt(5))), (0.05, 6.313751514675043)],
)
def test_cauchy_critical_matches_independent_ordinary_quantiles(tail, expected) -> None:
    assert math.isclose(student_t_critical(1, tail), expected, rel_tol=2e-15)


@pytest.mark.parametrize("tail", [1e-10, 1e-100, 1e-250, 1e-300])
def test_cauchy_critical_retains_small_representable_tails(tail) -> None:
    # Cauchy survival is asymptotic to 1/(pi*t); the first omitted relative
    # correction is O(tail**2), below binary64 rounding for these cases.
    assert math.isclose(student_t_critical(1, tail) * tail, 1 / math.pi, rel_tol=2e-15)


@pytest.mark.parametrize("tail", [0.5 - 1e-10, math.nextafter(0.5, 0.0)])
def test_cauchy_critical_retains_small_quantiles_near_half(tail) -> None:
    expected = math.pi * (0.5 - tail)
    assert math.isclose(student_t_critical(1, tail), expected, rel_tol=2e-15)


@pytest.mark.parametrize("tail", [1e-309, 1e-310, math.ulp(0.0)])
def test_unrepresentable_cauchy_critical_is_conservatively_infinite(tail) -> None:
    assert student_t_critical(1, tail) == math.inf


@pytest.mark.parametrize(
    ("degrees", "tail"),
    [(0, 0.05), (-1, 0.05), (1, 0.0), (1, 0.5), (1, -0.1), (1, math.nan), (1, math.inf)],
)
def test_student_critical_preserves_invalid_input_rejection(degrees, tail) -> None:
    with pytest.raises(ValueError, match="invalid Student"):
        student_t_critical(degrees, tail)


def _ordered_product(left, right):
    return sum(
        (
            Fraction.from_float(a) * Fraction.from_float(b)
            for i, a in enumerate(left)
            for j, b in enumerate(right)
            if i != j
        ),
        Fraction(0),
    ) / (len(left) * (len(left) - 1))


def _exact_jackknife_oracle(left, right, *, squared_change=False):
    def statistic(a, b):
        if squared_change:
            return (_ordered_product(b, b) - _ordered_product(a, a)) / 2
        return _ordered_product(a, b)

    count = len(left)
    estimate = statistic(left, right)
    pseudo = tuple(
        count * estimate
        - (count - 1) * statistic(left[:i] + left[i + 1 :], right[:i] + right[i + 1 :])
        for i in range(count)
    )
    centre = sum(pseudo, Fraction(0)) / count
    variance = sum(((value - centre) ** 2 for value in pseudo), Fraction(0)) / (count * (count - 1))
    # An independent exact variance plus Decimal square root avoids overflowing
    # binary64 squared errors or rounding away residuals around a large offset.
    with localcontext() as context:
        context.prec = 90
        error = float((Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt())
    return float(estimate), tuple(map(float, pseudo)), error


@pytest.mark.parametrize("scale", [1.0, 2.0**500, 2.0**511])
def test_all_pairs_product_centres_exact_pseudovalues_before_rounding(scale) -> None:
    delta = math.ulp(scale)
    left = (scale, scale + delta, scale - delta)
    right = (scale, scale - delta, scale + delta)
    expected, pseudo, error = _exact_jackknife_oracle(left, right)
    result = all_pairs_product(left, right)
    assert result.estimate == expected
    assert result.pseudovalues == pseudo == (expected,) * 3
    assert error > 0
    assert math.isclose(error, 2 * delta * delta / 3, rel_tol=2e-15)
    assert math.isclose(result.standard_error, error, rel_tol=2e-15)


@pytest.mark.parametrize("scale", [1.0, 2.0**500, 2.0**511])
def test_squared_change_centres_exact_paired_pseudovalues_before_rounding(scale) -> None:
    delta = math.ulp(scale)
    before = (scale, scale + delta, scale - delta)
    after = (scale / 2, scale / 2 + 2 * delta, scale / 2 - 2 * delta)
    expected, pseudo, error = _exact_jackknife_oracle(before, after, squared_change=True)
    result = all_pairs_squared_change(before, after)
    assert result.estimate == expected
    assert result.pseudovalues == pseudo == (expected,) * 3
    assert error == delta * delta
    assert math.isclose(result.standard_error, error, rel_tol=2e-15)


def test_all_pairs_product_retains_finite_error_with_unrepresentable_centred_deviation() -> None:
    left = (1.0,) * 3
    right = (-1.7e308, 1.7e308, 1.7e308)
    expected, pseudo, error = _exact_jackknife_oracle(left, right)
    result = all_pairs_product(left, right)
    assert result.estimate == expected
    assert result.pseudovalues == pseudo
    assert math.isfinite(error) and error > 1e308
    assert math.isclose(result.standard_error, error, rel_tol=2e-15)


def test_all_pairs_squared_product_matches_exact_delete_one_oracle() -> None:
    values = (0.25, -0.5, 1.125, -0.75)
    expected, pseudo, error = _exact_jackknife_oracle(values, values)
    result = all_pairs_product(values)
    assert result.estimate == expected
    assert result.pseudovalues == pseudo
    assert math.isclose(result.standard_error, error, rel_tol=2e-15)


def test_squared_change_retains_a_small_residual_between_unrepresentable_squares() -> None:
    before = (2.0**1000, -(2.0**1000), 0.0)
    after = (*before[:2], 2.0**-1000)
    expected, pseudo, error = _exact_jackknife_oracle(before, after, squared_change=True)
    result = all_pairs_squared_change(before, after)
    assert result.estimate == expected == 0
    assert result.pseudovalues == pseudo == (1.0, -1.0, 0.0)
    assert math.isclose(error, 1 / math.sqrt(3), rel_tol=2e-15)
    assert math.isclose(result.standard_error, error, rel_tol=2e-15)
    unchanged = all_pairs_squared_change((2.0**1000,) * 3, (2.0**1000,) * 3)
    assert unchanged.estimate == unchanged.standard_error == 0


def test_all_pairs_unrepresentable_output_pseudovalues_still_raise() -> None:
    scale = 2.0**512
    with pytest.raises(NumericalError, match="binary64 range"):
        all_pairs_product((scale, -scale, scale))


@pytest.mark.parametrize("squared_change", [False, True])
def test_all_pairs_genuine_checkpoint_overflow_still_raises(squared_change) -> None:
    with pytest.raises(NumericalError, match="binary64 range"):
        if squared_change:
            all_pairs_squared_change((0.0,) * 3, (1e308,) * 3)
        else:
            all_pairs_product((1e308,) * 3)
