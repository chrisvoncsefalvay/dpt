"""Shared fixtures and independent oracles: metric oracles."""

from fractions import Fraction


def exact_qp(field, gradient, step, metric):
    a, b, c = map(Fraction, metric)
    x, y = (Fraction(float(v)) for v in field)
    q, r = [Fraction(step) * Fraction(float(v)) for v in gradient]
    determinant = a * c - b * b
    unconstrained = (x - (c * q - b * r) / determinant, y - (a * r - b * q) / determinant)

    def clip(v):
        return min(Fraction(1), max(Fraction(0), v))

    t = clip(((a - b) * x + (b - c) * y - q + r + c - b) / (a + c - 2 * b))
    candidates = [
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (clip(x + (b * y - q) / a), Fraction(0)),
        (Fraction(0), clip(y + (b * x - r) / c)),
        (t, 1 - t),
    ]
    if min(unconstrained) >= 0 and sum(unconstrained) <= 1:
        candidates.append(unconstrained)

    def score(v):
        u, w = v[0] - x, v[1] - y
        return (a * u * u + 2 * b * u * w + c * w * w) / 2 + q * u + r * w

    return min(candidates, key=score)
