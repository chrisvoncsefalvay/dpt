"""Independent dense-matrix and moment oracles for detector validation.

Dense matrices deliberately belong only here: production blur uses its compact
finite stencil and transpose, never an N² detector matrix.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal, localcontext


def poisson_log_mass(count: int, rate: float) -> Decimal:
    """log P(K=k | lambda) = k log(lambda) - lambda - log(k!).

    ``count`` is a scalar non-negative integer photon count and ``rate`` is a
    positive finite binary64 expected count. The scalar ``Decimal`` result is
    dimensionless and is evaluated with 80 significant decimal digits at the
    exact supplied binary64 rate. This forward probability reference has no
    active parameters or gradient output.

    Factorials through k=1000 are exact integers. Larger counts use eight
    terms of the Stirling expansion, whose next-term error is below 2e-52.
    This direct expression checks the probability mass independently of the
    detector's cancellation-aware deviance calculation.

        >>> from dpt.validation.detector import poisson_log_mass
        >>> float(poisson_log_mass(0, 2.0))
        -2.0
        >>> round(float(poisson_log_mass(2, 2.0)), 6)
        -1.306853

    """
    if isinstance(count, bool) or count < 0 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Poisson reference requires integer count >= 0 and finite rate > 0")
    with localcontext() as context:
        context.prec = 80
        k, mean = Decimal(count), Decimal.from_float(rate)
        if count <= 1000:
            log_factorial = Decimal(math.factorial(count)).ln()
        else:
            pi = Decimal("3.1415926535897932384626433832795028841971693993751058209749445923078164")
            log_factorial = (k + Decimal("0.5")) * k.ln() - k + (2 * pi).ln() / 2
            bernoulli_coefficients = (
                (1, 12),
                (-1, 360),
                (1, 1260),
                (-1, 1680),
                (1, 1188),
                (-691, 360360),
                (7, 1092),
                (-3617, 122400),
            )
            for index, (numerator, denominator) in enumerate(bernoulli_coefficients):
                log_factorial += Decimal(numerator) / Decimal(denominator) / k ** (2 * index + 1)
        return +(k * mean.ln() - mean - log_factorial)


def spatial_response_matrix(
    height: int,
    width: int,
    kernel_height: int,
    kernel_width: int,
    weights: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """B[d,s] = weights[s_row-d_row+kh//2, s_col-d_col+kw//2].

    Entries outside the finite stencil are zero. ``height`` and ``width`` define
    an image with N=height*width pixels and ``kernel_height`` and
    ``kernel_width`` are positive odd stencil dimensions. ``weights`` is a
    length-kh*kw row-major sequence of dimensionless response weights. The
    returned tuple has shape (N,N), with destination pixels in rows and source
    pixels in columns, both flattened with columns varying fastest.

    Input weights become Python binary64 values without normalisation, and
    missing pixels beyond the image boundary contribute zero. This constructs
    the forward map for fixed weights and does not differentiate the stencil.

        >>> from dpt.validation.detector import spatial_response_matrix
        >>> spatial_response_matrix(1, 2, 1, 3, (0.25, 0.5, 0.25))
        ((0.5, 0.25), (0.25, 0.5))

    """
    if min(height, width, kernel_height, kernel_width) <= 0:
        raise ValueError("matrix and stencil dimensions must be positive")
    if kernel_height % 2 != 1 or kernel_width % 2 != 1:
        raise ValueError("stencil dimensions must be odd")
    if len(weights) != kernel_height * kernel_width:
        raise ValueError("weight count does not match stencil dimensions")
    pixels = height * width
    matrix = [[0.0 for _ in range(pixels)] for _ in range(pixels)]
    for destination in range(pixels):
        row, col = divmod(destination, width)
        for source in range(pixels):
            sr, sc = divmod(source, width)
            kr = sr - row + kernel_height // 2
            kc = sc - col + kernel_width // 2
            if 0 <= kr < kernel_height and 0 <= kc < kernel_width:
                matrix[destination][source] = float(weights[kr * kernel_width + kc])
    return tuple(tuple(row) for row in matrix)


def matrix_product(
    matrix: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    transpose: bool = False,
) -> tuple[float, ...]:
    """y_i = sum_j B[i,j] x_j, or y_j = sum_i B[i,j] x_i for transpose=True.

    ``matrix`` is a rectangular (rows,columns) sequence. ``values`` has length
    columns for the forward map or rows for the transpose, and the returned
    tuple has the complementary length. Detector matrices use destination rows
    and source columns in row-major pixel order. Dimensionless response weights
    preserve the signal units of ``values``.

    Python binary64 products are accumulated with ``math.fsum``. The transpose
    supplies the cotangent action of a fixed linear map, while matrix entries
    remain inactive. An empty matrix returns an empty tuple.

        >>> from dpt.validation.detector import matrix_product
        >>> matrix_product(((0.5, 0.25), (0.25, 0.5)), (2., 4.))
        (2.0, 2.5)
        >>> matrix_product(((1., 2.), (3., 4.)), (1., 0.), transpose=True)
        (1.0, 2.0)

    """
    if not matrix:
        return ()
    rows, columns = len(matrix), len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError("matrix must be rectangular")
    if len(values) != (rows if transpose else columns):
        raise ValueError("matrix/vector shape mismatch")
    if transpose:
        return tuple(
            math.fsum(matrix[r][c] * values[r] for r in range(rows)) for c in range(columns)
        )
    return tuple(
        math.fsum(value * weight for value, weight in zip(values, row, strict=True))
        for row in matrix
    )


def compound_poisson_moments(
    rates: Sequence[float],
    scores: Sequence[float],
) -> tuple[float, float, float]:
    """E[S]=sum_k lambda_k q_k, Var[S]=sum_k lambda_k q_k².

    For S=sum_k q_k K_k with independent Poisson counts, the fourth central
    moment is sum_k lambda_k q_k⁴ + 3 Var[S]². ``rates`` and ``scores`` are
    equal-length sequences with one non-negative expected photon count and
    deterministic per-photon score per energy channel. Scores use the detector's
    signal units per photon. The returned (mean,variance,fourth central moment)
    tuple uses signal units, their square and their fourth power respectively.

    Products and ``math.fsum`` reductions use Python binary64 arithmetic. These
    are forward moments at fixed rates and scores, with no derivative output.

        >>> from dpt.validation.detector import compound_poisson_moments
        >>> compound_poisson_moments((2., 3.), (1., 2.))
        (8.0, 14.0, 638.0)

    """
    if len(rates) != len(scores):
        raise ValueError("one deterministic photon score is required per rate")
    if any(not math.isfinite(v) or v < 0 for v in (*rates, *scores)):
        raise ValueError("rates and scores must be finite and non-negative")
    mean = math.fsum(rate * score for rate, score in zip(rates, scores, strict=True))
    variance = math.fsum(rate * score**2 for rate, score in zip(rates, scores, strict=True))
    fourth_cumulant = math.fsum(rate * score**4 for rate, score in zip(rates, scores, strict=True))
    return mean, variance, fourth_cumulant + 3 * variance**2


def spread_covariance(
    matrix: Sequence[Sequence[float]],
    independent_variances: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """Cov[Y] = B diag(v) Bᵀ for independent input events with variances v.

    ``matrix`` has shape (destinations,sources) and
    ``independent_variances`` has length sources. The returned tuple has shape
    (destinations,destinations), with both axes in the output pixel order of B.
    Dimensionless response weights leave covariance entries in squared signal
    units. Python binary64 products are summed with ``math.fsum``.

    This forward covariance at fixed B and v shows why spreading independent
    photon events creates correlated output pixels. It returns no gradients.

        >>> from dpt.validation.detector import spread_covariance
        >>> spread_covariance(((1., 0.5), (0., 1.)), (2., 4.))
        ((3.0, 2.0), (2.0, 4.0))

    """
    if any(len(row) != len(independent_variances) for row in matrix):
        raise ValueError("covariance dimensions do not match the response matrix")
    return tuple(
        tuple(
            math.fsum(
                a * b * variance
                for a, b, variance in zip(left, right, independent_variances, strict=True)
            )
            for right in matrix
        )
        for left in matrix
    )
