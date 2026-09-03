"""Independent derivatives of the original Hartree-Fock profile and joint law."""

from __future__ import annotations

import hashlib
import math
from decimal import Decimal, localcontext

from dpt.materials import Provenance
from dpt.transport.bound_compton import (
    BoundComptonTable,
    ComptonShell,
    bound_compton_angle_proposal,
    bound_compton_conditional_energy,
)

SLOPE_CASES = (
    (0.0, 1e308),
    (0.0, 1e6),
    (math.ulp(0.0), 1e200),
    (-math.ulp(0.0), 1e200),
    (1e-300, 1e200),
    (-1e-300, 1e200),
    (1e-20, 1e6),
    (-1e-20, 1e6),
    (1e-12, 1e6),
    (-1e-12, 1e6),
    (0.3, 1e-30),
    (-0.3, 1e-30),
    (1e300, 1e-280),
    (-1e300, 1e-280),
    (1e300, 1e-312),
    (0.5, 1.0),
    (-0.5, 1.0),
    (math.ulp(0.0), 0.49),
    (-math.ulp(0.0), 0.49),
    (math.ulp(0.0), 0.51),
    (-math.ulp(0.0), 0.51),
    (math.ulp(0.0), 0.75),
    (-math.ulp(0.0), 0.75),
    (math.ulp(0.0), 1e5),
    (-math.ulp(0.0), 1e5),
)

_NORMAL_BOUNDARY_MOMENTUM = 2.2250738585072014e-308 / 1e5
SEAM_CASES = tuple(
    (sign * momentum, 1e5)
    for sign in (1, -1)
    for momentum in (
        math.nextafter(_NORMAL_BOUNDARY_MOMENTUM, 0.0),
        _NORMAL_BOUNDARY_MOMENTUM,
        math.nextafter(_NORMAL_BOUNDARY_MOMENTUM, math.inf),
    )
)
MIDPOINT_CASES = ((3 * math.ulp(0.0), 0.25), (-3 * math.ulp(0.0), 0.25))

TERMINAL_CASES = (
    ("ordinary", 80.0, ((1.0, 1.0, 0.1),)),
    ("tiny-profile", 80.0, ((1.0, 0.0, 1e-30),)),
    ("narrow-profile", 100.0, ((1.0, 1.0, 1e6),)),
    ("surviving-negative-tail", 1.2, ((1.0, 1.0, 7.2973525693),)),
    ("representable-negative-tail", 100.0, ((1.0, 30.0, 7.2973525693),)),
    ("tiny-occupancy", 80.0, ((1e-320, 1.0, 0.1), (1.0, 1000.0, 0.1))),
    ("mixed-tail-occupancy", 1.2, ((1.0, 1.0, 1e6), (1e-320, 0.0, 0.1))),
)


def original_profile_log_slope(momentum: float, height: float) -> float:
    """Differentiate the original product, cancelling only its common exp factor.

    The polynomial derivative is evaluated directly, independently of the
    production factorisation.800digits resolve the extreme binary64 inputs.
    """
    with localcontext() as context:
        context.prec = 800
        p, h = map(Decimal.from_float, (momentum, height))
        x = abs(p)
        prefactor = 1 + 2 * h * x
        prefactor_derivative = 2 * h
        exponent_derivative = -2 * h - 4 * h * h * x
        derivative = (prefactor_derivative + prefactor * exponent_derivative) / prefactor
        return float(derivative if p >= 0 else -derivative)


def terminal_state(name, energy, entries):
    provenance = Provenance(
        "https://doi.org/10.1016/0168-9002(96)00652-3",
        hashlib.sha256(repr((name, entries)).encode()).hexdigest(),
        "CC0-1.0 analytic control",
        "Original conditioned impulse-law terminal score control",
    )
    table = BoundComptonTable(
        1,
        tuple(ComptonShell(str(i), *entry) for i, entry in enumerate(entries)),
        provenance,
    )
    cosine, _ = bound_compton_angle_proposal(energy, table, 0.5)
    outgoing, _, _ = bound_compton_conditional_energy(energy, cosine, table, 0.5, 0.5)
    return cosine, outgoing, table


def original_joint_cosine_score(energy, cosine, outgoing, table):
    """Decimal90 five-point derivative of original KN * profile * dp/dEout.

    The independent bound-Compton criterion holds the finite-difference step,
    exact binary64 physical states and normaliser fixed.
    """
    with localcontext() as context:
        context.prec = 90
        e, c, out = map(Decimal.from_float, (energy, cosine, outgoing))
        mass = Decimal.from_float(510.99895069)
        ratio = out / e

        def log_joint(angle):
            factor = 1 + e / mass * (1 - angle)
            q = ((1 - ratio) ** 2 + 2 * ratio * (1 - angle)).sqrt()
            momentum = (factor * ratio - 1) / q
            logs = []
            for shell in table.shells:
                if out < e - Decimal.from_float(shell.binding_kev):
                    height = Decimal.from_float(shell.momentum_height)
                    amount = height * abs(momentum)
                    logs.append(
                        Decimal.from_float(shell.occupancy).ln()
                        + height.ln()
                        + (1 + 2 * amount).ln()
                        - 2 * amount * (1 + amount)
                    )
            largest = max(logs)
            profile = largest + sum(((value - largest).exp() for value in logs), Decimal(0)).ln()
            jacobian_numerator = factor - angle + ratio * (1 - factor * angle)
            klein_nishina = -2 * factor.ln() + (1 / factor + factor - 1 + angle**2).ln()
            return klein_nishina + profile + jacobian_numerator.ln() - e.ln() - 3 * q.ln()

        height = max(Decimal.from_float(shell.momentum_height) for shell in table.shells)
        step = min(Decimal("1e-6"), (1 - abs(c)) / 1000, 1 / (height * 10**12))
        assert step > 0
        result = (
            log_joint(c - 2 * step)
            - 8 * log_joint(c - step)
            + 8 * log_joint(c + step)
            - log_joint(c + 2 * step)
        ) / (12 * step)
        return float(result)
