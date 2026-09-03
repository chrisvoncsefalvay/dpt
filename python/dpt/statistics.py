"""Range-aware scalar summaries shared by independent-replicate controllers.

The caller establishes independence and the sampling unit. These arithmetic
helpers cannot infer independence, normality or confidence-interval coverage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from dpt.contracts import ContractError, NumericalError, integer


@dataclass(frozen=True, slots=True)
class JackknifeEstimate:
    """Unbiased degree-two estimate and an asymptotic delete-one standard error.

    Pseudovalues share batches and are dependent. They must not be supplied to
    an acceptance test that assumes independent Gaussian replicate means. The
    jackknife is not a finite-sample coverage guarantee at a degenerate optimum.
    """

    estimate: float
    standard_error: float
    pseudovalues: tuple[float, ...]
    batches: int
    uncertainty_method: str = "delete_one_batch_jackknife_asymptotic"


def _all_pairs_exact(
    left: Sequence[float], right: Sequence[float]
) -> tuple[Fraction, tuple[Fraction, ...]]:
    count = len(left)
    if count < 3 or len(right) != count:
        raise ContractError("all-pairs jackknife requires at least three matching batches")
    if not all(math.isfinite(value) for values in (left, right) for value in values):
        raise NumericalError("all-pairs batches must be finite")
    a = tuple(Fraction.from_float(float(value)) for value in left)
    b = tuple(Fraction.from_float(float(value)) for value in right)
    total_a, total_b = sum(a, Fraction(0)), sum(b, Fraction(0))
    diagonal = sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))
    exact = (total_a * total_b - diagonal) / (count * (count - 1))
    pseudo = tuple(
        count * exact - ((total_a - x) * (total_b - y) - diagonal + x * y) / (count - 2)
        for x, y in zip(a, b, strict=True)
    )
    return exact, pseudo


def _jackknife_result(exact: Fraction, pseudovalues: tuple[Fraction, ...]) -> JackknifeEstimate:
    try:
        estimate = float(exact)
        pseudo = tuple(float(value) for value in pseudovalues)
        count = len(pseudovalues)
        centre = sum(pseudovalues, Fraction(0)) / count
        deviations = tuple(value - centre for value in pseudovalues)
        scale = max(map(abs, deviations))
        if scale:
            normalised = tuple(float(value / scale) for value in deviations)
            factor = math.hypot(*normalised) / math.sqrt(count * (count - 1))
            # Centre before narrowing: rounded pseudovalues may coincide even
            # when their exact differences yield representable uncertainty.
            # Restore the rational scale only after the bounded norm so an
            # unrepresentable intermediate deviation need not overflow the SE.
            error = float(scale * Fraction.from_float(factor))
        else:
            error = 0.0
    except OverflowError as failure:
        raise NumericalError("all-pairs checkpoint exceeds binary64 range") from failure
    return JackknifeEstimate(estimate, error, pseudo, count)


def all_pairs_product(
    left: Sequence[float], right: Sequence[float] | None = None
) -> JackknifeEstimate:
    """U = sum_{i != j} left[i]*right[j]/(n*(n-1)) estimates E[left]*E[right].

    The two length-n sequences contain scalar IID batch estimates with n >= 3.
    Their entries may be correlated within a batch, while different batches must
    be independent. Omitting right estimates the square of the expected left
    signal. The returned JackknifeEstimate contains a scalar value, its standard
    error and n leave-one pseudovalues in the corresponding signal-product units.
    Exact binary rational sums precede the final binary64 conversion. This is a
    host checkpoint with no active differentiation parameters.

    >>> from dpt.statistics import all_pairs_product
    >>> result = all_pairs_product((1.0, 2.0, 3.0))
    >>> round(result.estimate, 6)
    3.666667
    >>> len(result.pseudovalues)
    3

    """
    return _jackknife_result(*_all_pairs_exact(left, left if right is None else right))


def all_pairs_squared_change(before: Sequence[float], after: Sequence[float]) -> JackknifeEstimate:
    """Delta = (U(after,after) - U(before,before))/2 estimates the squared-loss change.

    Before and after are equally sized length-n scalar residual-mean sequences,
    where n >= 3 independent batches share a common distribution. Common random
    numbers may couple the two values at each index. The returned scalar estimate
    and jackknife standard error have squared-residual units and retain this
    pairing through their n pseudovalues. Exact rational sums protect cancellation
    before the final binary64 result. The parameters are observations of batches,
    not differentiation inputs.

    >>> from dpt.statistics import all_pairs_squared_change
    >>> change = all_pairs_squared_change((2.0, 2.0, 2.0), (1.0, 1.0, 1.0))
    >>> (change.estimate, change.standard_error)
    (-1.5, 0.0)

    """
    if len(before) != len(after):
        raise ContractError("squared-change batches must have matching lengths")
    first, pseudo_before = _all_pairs_exact(before, before)
    second, pseudo_after = _all_pairs_exact(after, after)
    pseudo = tuple((b - a) / 2 for a, b in zip(pseudo_before, pseudo_after, strict=True))
    return _jackknife_result((second - first) / 2, pseudo)


def predicted_batch_size(current: int, maximum: int, uncertainty: float, target: float) -> int:
    """n_next = ceil(n*(SE/target)^2), capped by maximum and increased by at least one.

    Current and maximum are positive integer history counts. Uncertainty and
    target are nonnegative binary64 scalars in matching estimator units, so their
    ratio is dimensionless. The returned integer proposes a fresh batch under the
    inverse-square-root standard-error scaling law. A zero uncertainty or target
    doubles the current count up to maximum. Subsequent acceptance uses its own
    held-out estimates rather than treating this allocation as evidence.

    >>> from dpt.statistics import predicted_batch_size
    >>> predicted_batch_size(100, 1000, uncertainty=0.2, target=0.1)
    400

    """
    integer(current, "current batch", minimum=1)
    integer(maximum, "maximum batch", minimum=current)
    if not all(math.isfinite(value) and value >= 0 for value in (uncertainty, target)):
        raise ContractError("batch uncertainty and target must be finite and nonnegative")
    if current == maximum:
        return maximum
    if uncertainty == 0 or target == 0:
        return min(2 * current, maximum)
    log_ratio = 2 * (math.log(uncertainty) - math.log(target))
    if log_ratio >= math.log(maximum / current):
        return maximum
    return min(maximum, max(current + 1, math.ceil(current * (uncertainty / target) ** 2)))


def mean_standard_error(values: Sequence[float]) -> tuple[float, float]:
    """mean = sum(x)/n and SE = sqrt(sum((x-mean)^2)/(n*(n-1))).

    Values is a length-n sequence of finite scalar replicates with n >= 2. Both
    returned binary64 scalars have the units of the samples. Exact binary rational
    centres and scale-normalised deviations preserve finite results for extreme
    sample magnitudes. Independent replicates are required when interpreting the
    second result as uncertainty in an expected measurement. No gradient is
    formed by this host reduction.

    >>> from dpt.statistics import mean_standard_error
    >>> mean, error = mean_standard_error((1.0, 2.0, 3.0))
    >>> (mean, round(error, 6))
    (2.0, 0.57735)

    """
    count = len(values)
    if count < 2 or not all(map(math.isfinite, values)):
        raise NumericalError("replicate checkpoint requires at least two finite values")
    scale = max(map(abs, values))
    if scale == 0:
        return 0.0, 0.0
    exact = tuple(Fraction.from_float(float(value)) for value in values)
    exact_mean = sum(exact, Fraction(0)) / count
    exact_scale = Fraction.from_float(scale)
    mean = float(exact_mean)
    deviations = tuple(float((value - exact_mean) / exact_scale) for value in exact)
    error = scale * (math.hypot(*deviations) / math.sqrt(count * (count - 1)))
    if not math.isfinite(mean) or not math.isfinite(error):
        raise NumericalError("replicate mean or standard error exceeds binary64 range")
    return mean, error


@lru_cache(maxsize=256)
def student_t_critical(degrees: int, tail_probability: float) -> float:
    """P(T_degrees > t) = tail_probability defines the returned one-sided critical t.

    Degrees is a positive scalar integer and tail_probability is a dimensionless
    scalar in (0, 0.5). The dimensionless binary64 result uses the regularised
    incomplete beta function, with the analytic Cauchy quantile for one degree of
    freedom. Student intervals apply to independent Gaussian replicate errors.
    This host quantile calculation has no active parameters.

    >>> from dpt.statistics import student_t_critical
    >>> round(student_t_critical(1, 0.25), 12)
    1.0

    """
    if degrees < 1 or not 0 < tail_probability < 0.5:
        raise ValueError("invalid Student degrees or tail probability")
    if degrees == 1:
        # The Cauchy quantile avoids the incomplete-beta path's overflowing
        # t*t at small tails. Use the complementary angle near probability 1/2
        # so subtracting the angle from pi/2 cannot erase a small quantile.
        if tail_probability > 0.25:
            return math.tan(math.pi * (0.5 - tail_probability))
        return 1.0 / math.tan(math.pi * tail_probability)

    def beta_fraction(a: float, b: float, x: float) -> float:
        qab, qap, qam = a + b, a + 1, a - 1
        c = 1.0
        d = 1.0 - qab * x / qap
        d = 1.0 / max(abs(d), 1e-300) * (1 if d >= 0 else -1)
        h = d
        for m in range(1, 401):
            aa = m * (b - m) * x / ((qam + 2 * m) * (a + 2 * m))
            d = 1 + aa * d
            c = 1 + aa / c
            if abs(d) < 1e-300:
                d = 1e-300
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + 2 * m) * (qap + 2 * m))
            d = 1 + aa * d
            c = 1 + aa / c
            if abs(d) < 1e-300:
                d = 1e-300
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1 / d
            delta = d * c
            h *= delta
            if abs(delta - 1) < 3e-14:
                return h
        raise ArithmeticError("incomplete beta did not converge")

    def beta(x: float, a: float, b: float) -> float:
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        factor = math.exp(
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + a * math.log(x)
            + b * math.log1p(-x)
        )
        if x < (a + 1) / (a + b + 2):
            return factor * beta_fraction(a, b, x) / a
        return 1 - factor * beta_fraction(b, a, 1 - x) / b

    def tail(t: float) -> float:
        return 0.5 * beta(degrees / (degrees + t * t), degrees / 2, 0.5)

    low = 0.0
    high = 1.0
    while tail(high) > tail_probability:
        high *= 2
    for _ in range(70):
        mid = (low + high) / 2
        if tail(mid) > tail_probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2
