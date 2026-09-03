"""Independent Decimal reference for the specified discrete spectral primary model.

This module is an oracle, never imported by production GPU operators. It treats
binary32 input values as exact real numbers supplied by the test harness and
performs the defining energy sums at a caller-selected decimal precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext


@dataclass(frozen=True, slots=True)
class SpectralReference:
    mean: tuple[Decimal, ...]
    grad_paths: tuple[Decimal, ...]
    grad_coefficients: tuple[Decimal, ...]
    grad_weights: tuple[Decimal, ...]
    grad_response: tuple[Decimal, ...]


def spectral_reference(
    paths: tuple[float, ...],
    coefficients: tuple[float, ...],
    weights: tuple[float, ...],
    response: tuple[float, ...],
    *,
    materials: int,
    energies: int,
    seed: tuple[float, ...] | None = None,
    shared_weights: bool = True,
    shared_response: bool = True,
    precision: int = 100,
) -> SpectralReference:
    """Evaluate ``I[p] = sum_k w[k,p] r[k,p] exp(-sum_m mu[m,k] A[m,p])``.

    Inputs are flat real tuples. Paths ``A`` have ``materials * pixels``
    entries in material-major order and units mm, and coefficients ``mu``
    have ``materials * energies`` entries in material/energy order and units
    mm^-1. Shared weights/response have ``energies`` entries, whilst their
    pixel-specific layouts have ``energies * pixels`` in energy-major order.
    Weights are integrated bin populations and the response sets the output
    unit. Decimal arithmetic treats the supplied binary floats as exact inputs.

    ``seed`` is an output cotangent of length ``pixels``. ``None`` selects
    all ones, differentiating the sum of the image. The result's Decimal
    tuples contain ``pixels`` means and input-shaped derivatives of
    ``sum_p seed[p] * I[p]``. Shared-factor derivatives sum over pixels.

    Example:
        >>> result = spectral_reference(
        ...     (2.0,), (0.25,), (100.0,), (1.0,), materials=1, energies=1,
        ... )
        >>> round(float(result.mean[0]), 8)
        60.65306597
        >>> round(float(result.grad_paths[0]), 8)
        -15.16326649

    """
    if materials < 1 or energies < 1 or len(paths) % materials:
        raise ValueError("invalid material/energy/path dimensions")
    pixels = len(paths) // materials
    if len(coefficients) != materials * energies:
        raise ValueError("invalid coefficient dimensions")
    if len(weights) != energies * (1 if shared_weights else pixels):
        raise ValueError("invalid spectral weight dimensions")
    if len(response) != energies * (1 if shared_response else pixels):
        raise ValueError("invalid response dimensions")
    if seed is None:
        seed = (1.0,) * pixels
    if len(seed) != pixels:
        raise ValueError("invalid seed dimensions")
    with localcontext() as context:
        context.prec = precision
        a = tuple(Decimal.from_float(float(value)) for value in paths)
        mu = tuple(Decimal.from_float(float(value)) for value in coefficients)
        n = tuple(Decimal.from_float(float(value)) for value in weights)
        r = tuple(Decimal.from_float(float(value)) for value in response)
        s = tuple(Decimal.from_float(float(value)) for value in seed)
        if any(not value.is_finite() or value < 0 for value in (*a, *mu, *n, *r)):
            raise ValueError("reference physical inputs must be finite and non-negative")
        if any(not value.is_finite() for value in s):
            raise ValueError("reference seeds must be finite")
        mean = [Decimal(0) for _ in range(pixels)]
        ga = [Decimal(0) for _ in a]
        gmu = [Decimal(0) for _ in mu]
        gn = [Decimal(0) for _ in n]
        gr = [Decimal(0) for _ in r]
        for pixel in range(pixels):
            for energy in range(energies):
                ni = energy if shared_weights else energy * pixels + pixel
                ri = energy if shared_response else energy * pixels + pixel
                depth = sum(
                    (a[m * pixels + pixel] * mu[m * energies + energy] for m in range(materials)),
                    Decimal(0),
                )
                survival = (-depth).exp()
                contribution = n[ni] * r[ri] * survival
                mean[pixel] += contribution
                gn[ni] += s[pixel] * r[ri] * survival
                gr[ri] += s[pixel] * n[ni] * survival
                for material in range(materials):
                    ai, mi = material * pixels + pixel, material * energies + energy
                    ga[ai] -= s[pixel] * contribution * mu[mi]
                    gmu[mi] -= s[pixel] * contribution * a[ai]
        return SpectralReference(tuple(mean), tuple(ga), tuple(gmu), tuple(gn), tuple(gr))
