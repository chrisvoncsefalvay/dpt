"""Original-law Decimal reference for nearly balanced deep negative shells.

This reference enlarges the Decimal exponent range so direct physical
masses at log density around -1.9e12 remain representable. It does not use
production's anchored relative-profile arithmetic.
"""

from __future__ import annotations

from decimal import MAX_EMAX, MIN_EMIN, Decimal, localcontext

import bound_occupancy_reference as original
from bound_occupancy_reference import OccupancyNormalisationReference, Shells


def balanced_fixtures() -> tuple[tuple[str, Shells], ...]:
    minimum = float.fromhex("0x0.0000000000001p-1022")
    alpha = 7.2973525643e-3
    height = ((1e6 - 0.0001935) * alpha) / alpha
    base = (1.0, 1.0, (1e6 * alpha) / alpha)
    tiny = (minimum, 1.0, height)
    return (("balanced-tiny-first", (tiny, base)), ("balanced-tiny-last", (base, tiny)))


def original_balanced_normalisation(
    energy: float, shells: Shells
) -> OccupancyNormalisationReference:
    with localcontext() as context:
        context.prec, context.Emin, context.Emax = original.PRECISION, MIN_EMIN, MAX_EMAX
        return original.original_occupancy_normalisation(energy, shells)


def original_balanced_phase_pdf(energy: float, cosine: float, shells: Shells) -> float:
    with localcontext() as context:
        context.prec, context.Emin, context.Emax = original.PRECISION, MIN_EMIN, MAX_EMAX
        return original.original_occupancy_phase_pdf(energy, cosine, shells)


def original_balanced_joint(
    energy: float, cosine: float, outgoing: float, shells: Shells
) -> tuple[float, tuple[float, float, float]]:
    with localcontext() as context:
        context.prec, context.Emin, context.Emax = original.PRECISION, MIN_EMIN, MAX_EMAX
        return original.original_occupancy_joint(energy, cosine, outgoing, shells)


def original_balanced_finite_scores(
    energy: float, cosine: float, outgoing: float, shells: Shells
) -> tuple[float, float, float]:
    with localcontext() as context:
        context.prec, context.Emin, context.Emax = original.PRECISION, MIN_EMIN, MAX_EMAX
        return original.original_occupancy_finite_scores(energy, cosine, outgoing, shells)


def original_balanced_angle_proposal(
    energy: float, uniform: float, shells: Shells
) -> tuple[float, float]:
    with localcontext() as context:
        context.prec, context.Emin, context.Emax = original.PRECISION, MIN_EMIN, MAX_EMAX
        return original.original_occupancy_angle_proposal(energy, uniform, shells)


def original_balanced_shell(energy: float, cosine: float, uniform: float, shells: Shells) -> int:
    with localcontext() as context:
        context.prec, context.Emin, context.Emax = original.PRECISION, MIN_EMIN, MAX_EMAX
        return original.original_occupancy_shell(energy, cosine, uniform, shells)


def original_balanced_energy_cdf(
    energy: float, cosine: float, outgoing: float, shell_index: int, shells: Shells
) -> float:
    with localcontext() as context:
        context.prec, context.Emin, context.Emax = original.PRECISION, MIN_EMIN, MAX_EMAX
        return original.original_occupancy_energy_cdf(energy, cosine, outgoing, shell_index, shells)


def original_balanced_energy_inverse(
    energy: float, cosine: float, shell_uniform: float, energy_uniform: float, shells: Shells
) -> tuple[float, int, float]:
    with localcontext() as context:
        context.prec, context.Emin, context.Emax = original.PRECISION, MIN_EMIN, MAX_EMAX
        return original.original_occupancy_energy_inverse(
            energy, cosine, shell_uniform, energy_uniform, shells
        )


def original_balanced_states(energy: float, shells: Shells) -> tuple[tuple[float, float], ...]:
    with localcontext() as context:
        context.prec, context.Emin, context.Emax = original.PRECISION, MIN_EMIN, MAX_EMAX
        return original.occupancy_states(energy, shells)


def original_balanced_shell_fractions(
    energy: float, cosine: float, shells: Shells
) -> tuple[Decimal, ...]:
    with localcontext() as context:
        context.prec, context.Emin, context.Emax = original.PRECISION, MIN_EMIN, MAX_EMAX
        e, c = Decimal.from_float(energy), Decimal.from_float(cosine)
        weights = tuple(
            Decimal.from_float(f)
            * original._mass(e, c, Decimal.from_float(u), Decimal.from_float(h))
            for f, u, h in shells
        )
        total = sum(weights, Decimal(0))
        return tuple(weight / total for weight in weights)
