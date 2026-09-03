"""Independent affine-ray and represented-value Poisson acceptance references."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import TypedDict


class PoissonReference(TypedDict):
    means: tuple[Decimal, ...]
    gradient: tuple[Decimal, Decimal]
    optimum: tuple[Decimal, Decimal]
    loss: Decimal


class AffineReference(TypedDict):
    output: list[float]
    coefficients: list[list[float]]
    weights: list[float] | list[list[float]]
    response: list[float] | list[list[float]]
    pose: list[float]
    field_amplitude: float


SOURCES = ((-1.75, -0.625, 0.5), (-1.0, 0.5, -0.25), (-1.25, -1.25, 0.75))
ENDS = ((1.125, 0.75, -0.25), (0.75, 1.5, 0.5), (1.5, -0.25, -0.5))
SLOPES = ((1 / 64, -1 / 128, 3 / 256), (-1 / 128, 3 / 256, 1 / 256))
INTERCEPTS = (0.125, 0.1875)
COEFFICIENTS = ((0.25, 0.5, 0.75), (0.75, 0.5, 0.25))
WEIGHTS = (16.0, 32.0, 8.0)
RESPONSE = (0.5, 0.25, 0.75)
SEED = (0.25, -0.5, 0.75)
TAU = Fraction(1, 32)
HESSIAN_MINIMUM = Fraction(2304)
COUNTS = (6144, 5632)
INCIDENT = 8192


def gamma(count: int, bits: int) -> Fraction:
    unit = Fraction(1, 2**bits)
    return count * unit / (1 - count * unit)


def solver_gradient_budget(mode: str) -> Fraction:
    """Uniform gradient allowance on the original R4 final-endpoint box.

    Two material products/additions cost gamma_4. A 4-ULP exp admission costs
    at most 8u relative on [exp(-1),1]. Strict paths have a conservative 64-op
    FP64 geometry/interpolation allowance; retained FP32 paths add one cast.
    A seed is rounded once, each two-channel contraction is rounded once and
    each one-voxel adjoint has at most 64 FP32 operations. The division seed
    is bounded by one on the box (predictions and counts in [4096,8192]).
    Separate compound-mean admission is also required by the test.
    """
    assert mode in ("float32_strict", "float64_strict", "float32_arithmetic32")
    narrow = mode != "float64_strict"
    path = gamma(64, 53) + (Fraction(1, 2**24) if narrow else 0)
    depth = Fraction(5, 8) * (path + gamma(4, 24 if "arithmetic32" in mode else 53))
    exp_relative = Fraction(8, 2**24 if "arithmetic32" in mode else 2**53)
    # exp is 1-Lipschitz for nonnegative optical depths. Mean rounding adds u.
    mean_error = INCIDENT * (
        depth + exp_relative + (Fraction(1, 2**24) if narrow else gamma(8, 53))
    )
    # g=A.T*(y-mu). Sum |A| per coordinate is 5/4. The seed rounding
    # and rounded two-channel path seed are amplified by mu<=B and length2.
    seed_round = Fraction(1, 2**24 if narrow else 2**53)
    # The objective seed uses the retained mean but the VJP reuses unrounded
    # attenuation. Two mean-error allowances cover their ratio as well as the
    # ideal-forward discrepancy; retained means are >=B/2 on the final box.
    contraction = Fraction(5, 2) * mean_error + INCIDENT * Fraction(5, 2) * seed_round
    scatter = INCIDENT * Fraction(5, 4) * gamma(64, 24)
    return contraction + scatter + INCIDENT * gamma(64, 53)


def poisson_reference(fields: tuple[float, ...], precision: int = 100) -> PoissonReference:
    """Ideal Beer-Lambert half-deviance, Euclidean gradient and exact MLE."""
    assert len(fields) == 2
    with localcontext() as ctx:
        ctx.prec = precision
        d = Decimal.from_float
        a, b = map(d, fields)
        means = tuple(Decimal(INCIDENT) * (-depth).exp() for depth in (a + b / 4, a / 4 + b))
        residual = tuple(Decimal(y) - mu for y, mu in zip(COUNTS, means, strict=True))
        gradient = (residual[0] + residual[1] / 4, residual[0] / 4 + residual[1])
        depths = tuple(-(Decimal(y) / INCIDENT).ln() for y in COUNTS)
        optimum = ((16 * depths[0] - 4 * depths[1]) / 15, (-4 * depths[0] + 16 * depths[1]) / 15)
        loss = sum(
            (
                mu - y + Decimal(y) * (Decimal(y) / mu).ln()
                for y, mu in zip(COUNTS, means, strict=True)
            ),
            Decimal(0),
        )
        return {"means": means, "gradient": gradient, "optimum": optimum, "loss": loss}


def affine_reference(
    shared: bool, domain: str, amplitude: float, displacement: tuple[float, float, float]
) -> AffineReference:
    """Differentiate affine line integrals and the spectral chain algebraically.

    The rigid frame is a known quarter turn. A displaced anchor changes its
    local midpoint by the supplied local translation. Rotations use the three
    inverse-coordinate generator formulae, independently of pose helpers.
    """
    paths, tangents = [], []
    for slope, intercept in zip(SLOPES, INTERCEPTS, strict=True):
        channel, derivatives = [], []
        for source, end in zip(SOURCES, ENDS, strict=True):
            x, y, z = tuple(
                (s + e) / 2 - shift for s, e, shift in zip(source, end, displacement, strict=True)
            )
            length = math.sqrt(sum((e - s) ** 2 for s, e in zip(source, end, strict=True)))
            channel.append(
                amplitude * length * (intercept + slope[0] * x + slope[1] * y + slope[2] * z)
            )
            derivatives.append(
                tuple(
                    amplitude * length * q
                    for q in (
                        -slope[0],
                        -slope[1],
                        -slope[2],
                        slope[1] * z - slope[2] * y,
                        -slope[0] * z + slope[2] * x,
                        slope[0] * y - slope[1] * x,
                    )
                )
            )
        paths.append(channel)
        tangents.append(derivatives)
    pattern = (0.75, 1.0, 1.25)
    weights = [[w * (1 if shared else pattern[p]) for p in range(3)] for w in WEIGHTS]
    response = [[r * (1 if shared else pattern[2 - p]) for p in range(3)] for r in RESPONSE]
    attenuation = [
        [math.exp(-sum(COEFFICIENTS[m][k] * paths[m][p] for m in range(2))) for p in range(3)]
        for k in range(3)
    ]
    bins = [
        [weights[k][p] * response[k][p] * attenuation[k][p] for p in range(3)] for k in range(3)
    ]
    means = [math.fsum(bins[k][p] for k in range(3)) for p in range(3)]
    scale = [SEED[p] / means[p] if domain == "log" else SEED[p] for p in range(3)]
    bin_seed = [[bins[k][p] * scale[p] for p in range(3)] for k in range(3)]
    path_seed = [
        [-math.fsum(COEFFICIENTS[m][k] * bin_seed[k][p] for k in range(3)) for p in range(3)]
        for m in range(2)
    ]
    coeff = [
        [-math.fsum(paths[m][p] * bin_seed[k][p] for p in range(3)) for k in range(3)]
        for m in range(2)
    ]
    gw = [[response[k][p] * attenuation[k][p] * scale[p] for p in range(3)] for k in range(3)]
    gr = [[weights[k][p] * attenuation[k][p] * scale[p] for p in range(3)] for k in range(3)]
    return {
        "output": [math.log(mu) if domain == "log" else mu for mu in means],
        "coefficients": coeff,
        "weights": [math.fsum(row) for row in gw] if shared else gw,
        "response": [math.fsum(row) for row in gr] if shared else gr,
        "pose": [
            math.fsum(path_seed[m][p] * tangents[m][p][axis] for m in range(2) for p in range(3))
            for axis in range(6)
        ],
        "field_amplitude": math.fsum(
            path_seed[m][p] * paths[m][p] / amplitude for m in range(2) for p in range(3)
        ),
    }
