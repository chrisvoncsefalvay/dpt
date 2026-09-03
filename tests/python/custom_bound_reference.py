"""Independent Decimal profile and original-angle references for custom shells."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

MASS = 510.99895069


def decimal_interval_mass(lower: float, upper: float, height: float) -> float:
    with localcontext() as context:
        context.prec = 110
        h = Decimal.from_float(height)

        def cdf(point: Decimal) -> Decimal:
            x = abs(point)
            tail = (-2 * h * x * (1 + h * x)).exp() / 2
            return tail if point <= 0 else 1 - tail

        return float(cdf(Decimal.from_float(upper)) - cdf(Decimal.from_float(lower)))


def decimal_interval_inverse(
    probability: float, lower: float, upper: float, height: float
) -> float:
    with localcontext() as context:
        context.prec = 110
        h = Decimal.from_float(height)
        lo, hi = Decimal.from_float(lower), Decimal.from_float(upper)

        def cdf(point: Decimal) -> Decimal:
            x = abs(point)
            tail = (-2 * h * x * (1 + h * x)).exp() / 2
            return tail if point <= 0 else 1 - tail

        target = cdf(lo) + Decimal.from_float(probability) * (cdf(hi) - cdf(lo))
        for _ in range(240):
            midpoint = (lo + hi) / 2
            if cdf(midpoint) < target:
                lo = midpoint
            else:
                hi = midpoint
        return float((lo + hi) / 2)


def original_angle_narrow_normalisation(energy: float, intervals: int = 4096) -> float:
    """h=1e6,U=1: resolve original c-space transition, with 90-digit KN primitive.

    The omitted tails have |h*pmax|>38 and probability below exp(-2964)/2.
    This reference never uses profile quantiles or the production quadrature.
    """
    binding, height = 1.0, 1e6
    alpha = energy / MASS
    threshold = 1 - MASS * binding / (energy * (energy - binding))
    lower, upper = threshold - 64 / height, threshold + 64 / height
    with localcontext() as context:
        context.prec = 90
        a = Decimal.from_float(alpha)

        def primitive(cosine: float) -> Decimal:
            factor = 1 + a * (1 - Decimal.from_float(cosine))
            c1, c2 = 1 - 2 / a - 2 / a**2, 2 / a + 1 / a**2
            return -1 / (2 * factor**2) + c1 * factor.ln() - c2 / factor + factor / a**2

        below = float((primitive(-1.0) - primitive(lower)) / a)

    def integrand(cosine: float) -> float:
        momentum = (energy * (energy - binding) * (1 - cosine) / MASS - binding) / math.sqrt(
            2 * energy * (energy - binding) * (1 - cosine) + binding**2
        )
        tail = 0.5 * math.exp(-2 * height * abs(momentum) * (1 + height * abs(momentum)))
        probability = tail if momentum < 0 else 1 - tail
        factor = 1 + alpha * (1 - cosine)
        kn = factor**-3 + factor**-1 + (cosine**2 - 1) * factor**-2
        return probability * kn

    step = (upper - lower) / intervals
    terms = [integrand(lower), integrand(upper)]
    terms.extend((4 if i % 2 else 2) * integrand(lower + i * step) for i in range(1, intervals))
    return 2 * math.pi * (below + step / 3 * math.fsum(terms))


def tiny_height_thomson_pair(energy: float, intervals: int = 4096) -> tuple[float, float]:
    """Independent original-angle integral of N/h and N_E/h as h tends to zero.

    With U=0, pmax=(E/m)*sin(theta/2), and J/h tends to one uniformly
    on the physically accessible bounded interval. At h=1e-30, the omitted
    terms are far below binary64 precision. Integrate this limiting law in
    original angle, rather than reproducing the production Fubini rule.
    """

    def integrand(sine_half: float) -> tuple[float, float]:
        transfer = 2 * sine_half**2
        ratio = 1 / (1 + energy / MASS * transfer)
        kn = ratio**3 + ratio - transfer * (2 - transfer) * ratio**2
        kn_e = (
            -transfer / MASS * ratio**2 * (3 * ratio**2 + 1 - 2 * transfer * (2 - transfer) * ratio)
        )
        profile = 1 + energy / MASS * sine_half
        return 4 * sine_half * kn * profile, 4 * sine_half * (
            kn_e * profile + kn * sine_half / MASS
        )

    values: list[float] = []
    derivatives: list[float] = []
    for index in range(intervals + 1):
        value, derivative = integrand(index / intervals)
        coefficient = 1 if index in (0, intervals) else (4 if index % 2 else 2)
        values.append(coefficient * value)
        derivatives.append(coefficient * derivative)
    coefficient = 2 * math.pi / (3 * intervals)
    return coefficient * math.fsum(values), coefficient * math.fsum(derivatives)
