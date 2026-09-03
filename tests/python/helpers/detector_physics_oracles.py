"""Independent surface quadrature and high-precision slab-response references."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from functools import cache
from typing import Any

import numpy as np


def _decimal_atan_small(value: Decimal) -> Decimal:
    power, term, result = value, value, value
    square = value * value
    index = 1
    while term:
        power *= -square
        term = power / (2 * index + 1)
        previous = result
        result += term
        if result == previous:
            break
        index += 1
    return result


@cache
def _decimal_pi() -> Decimal:
    with localcontext() as context:
        context.prec = 480
        return 16 * _decimal_atan_small(Decimal(1) / 5) - 4 * _decimal_atan_small(Decimal(1) / 239)


def _decimal_atan(value: Decimal) -> Decimal:
    if value < 0:
        return -_decimal_atan(-value)
    if value > 1:
        return _decimal_pi() / 2 - _decimal_atan(1 / value)
    multiplier = 1
    while value > Decimal("0.1"):
        value /= 1 + (1 + value * value).sqrt()
        multiplier *= 2
    return multiplier * _decimal_atan_small(value)


def decimal_rectangle_angle(x: float, y: float, d: float, du: float, dv: float) -> float:
    """Independent 480-digit rectangular antiderivative, with no triangles.

    Decimal retains declared boundaries even when binary64 corners coincide.
    atan(x*y / (d*sqrt(x²+y²+d²))) is the surface integral's corner primitive.
    """
    with localcontext() as context:
        context.prec = 480
        cx, cy, depth, width, height = map(Decimal.from_float, (x, y, d, du, dv))
        terms = []
        for a, sx in ((cx - width / 2, -1), (cx + width / 2, 1)):
            for b, sy in ((cy - height / 2, -1), (cy + height / 2, 1)):
                radius = (a * a + b * b + depth * depth).sqrt()
                terms.append(sx * sy * _decimal_atan(a * b / (depth * radius)))
        return float(sum(terms))


def surface_solid_angle(panel: Any, row: int, column: int, *, order: int = 40) -> float:
    """Integrate n·(p-s)/|p-s|³ over world-space pixel surface coordinates.

    The reference constructs and differentiates the parametrized surface, rather
    than evaluating triangles or the production cylindrical antiderivative.
    """
    from dpt.geometry import CurvedDetectorGeometry

    nodes, weights = np.polynomial.legendre.leggauss(order)
    source = np.asarray(panel.source_mm)
    origin = np.asarray(panel.origin_mm)
    u, v = np.asarray(panel.u), np.asarray(panel.v)
    du, dv = panel.spacing_mm
    terms: list[float] = []
    for a, wa in zip(nodes, weights, strict=True):
        coordinate_u = (column + float(a) / 2) * du
        for b, wb in zip(nodes, weights, strict=True):
            coordinate_v = (row + float(b) / 2) * dv
            if isinstance(panel, CurvedDetectorGeometry):
                radial = np.asarray(panel.radial_mm)
                radius = panel.radius_mm
                angle = coordinate_u / radius
                position = (
                    origin
                    + (math.cos(angle) - 1) * radial
                    + math.sin(angle) * radius * u
                    + coordinate_v * v
                )
                tangent = -math.sin(angle) * radial / radius + math.cos(angle) * u
            else:
                position = origin + coordinate_u * u + coordinate_v * v
                tangent = u
            vector = position - source
            normal = np.cross(tangent, v)
            integrand = abs(float(np.dot(normal, vector))) / float(np.linalg.norm(vector)) ** 3
            terms.append(float(wa * wb) * integrand * du * dv / 4)
    return math.fsum(terms)


def decimal_response(mu: float, thickness: float, cosine: float, score: float) -> float:
    with localcontext() as context:
        context.prec = 100
        a, t, c, s = (Decimal.from_float(v) for v in (mu, thickness, cosine, score))
        # A separately evaluated alternating series retains very small losses;
        # the ordinary Decimal exponential handles the moderate/deep regimes.
        x = a * t / c
        if x < Decimal("1e-20"):
            probability = x - x * x / 2 + x**3 / 6
        else:
            probability = 1 - (-x).exp()
        return float(s * probability)


def decimal_derivatives(
    mu: float, thickness: float, cosine: float, score: float, seed: float
) -> tuple[float, float]:
    with localcontext() as context:
        context.prec = 100
        a, t, c, s, adj = (Decimal.from_float(v) for v in (mu, thickness, cosine, score, seed))
        derivative = s * adj * a / c * (-a * t / c).exp()
        return float(derivative), float(-derivative * t / c)
