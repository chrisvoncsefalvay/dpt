"""Profile-quantile integration of the relativistic impulse angular law.

Fubini exchanges the momentum and angular integrals. Each momentum quantile
then carries a smooth Klein-Nishina angular interval, so narrow custom shell
profiles do not require an angular grid to resolve their transition width.
"""

from __future__ import annotations

import math

from .bound_compton import (
    BoundComptonTable,
    _profile_interval_inverse_pair,  # pyright: ignore[reportPrivateUsage]
    profile_interval_mass,
    shell_momentum_max,
)
from .model import ELECTRON_REST_ENERGY_KEV, TransportError

_NODES = (
    0.09501250983763744,
    0.2816035507792589,
    0.4580167776572274,
    0.6178762444026438,
    0.755404408355003,
    0.8656312023878318,
    0.9445750230732326,
    0.9894009349916499,
)
_WEIGHTS = (
    0.1894506104550685,
    0.1826034150449236,
    0.16915651939500254,
    0.14959598881657673,
    0.12462897125553387,
    0.09515851168249278,
    0.06225352393864789,
    0.027152459411754095,
)


def _kn(energy: float, transfer: float) -> tuple[float, float]:
    ratio = 1 / (1 + energy / ELECTRON_REST_ENERGY_KEV * transfer)
    sine_squared = transfer * (2 - transfer)
    value = ratio**3 + ratio - sine_squared * ratio**2
    derivative = (
        -transfer
        / ELECTRON_REST_ENERGY_KEV
        * ratio**2
        * (3 * ratio**2 + 1 - 2 * sine_squared * ratio)
    )
    return value, derivative


def _kn_interval(energy: float, lower: float) -> tuple[float, float]:
    """Integral over ``lower < 1-c < 2`` and its fixed-limit E derivative."""
    width = 2 - lower
    if width <= 0:
        return 0.0, 0.0
    alpha = energy / ELECTRON_REST_ENERGY_KEV
    if alpha < 0.125:
        # Smooth low-alpha rational integrand; this avoids cancellation between
        # the exact primitive's inverse-alpha terms near the Thomson limit.
        midpoint, radius = (lower + 2) / 2, width / 2
        values: list[float] = []
        derivatives: list[float] = []
        for node, weight in zip(_NODES, _WEIGHTS, strict=True):
            for sign in (-1, 1):
                value, derivative = _kn(energy, midpoint + sign * radius * node)
                values.append(radius * weight * value)
                derivatives.append(radius * weight * derivative)
        return math.fsum(values), math.fsum(derivatives)
    inverse = 1 / alpha
    left, right = 1 / (1 + alpha * lower), 1 / (1 + 2 * alpha)
    difference = alpha * width * left * right
    logarithm = math.log1p(alpha * width * left)
    c1, c2 = 1 - 2 * inverse - 2 * inverse**2, 2 * inverse + inverse**2
    primitive = math.fsum(
        (
            0.5 * difference * (left + right),
            c1 * logarithm,
            c2 * difference,
            width * inverse,
        )
    )
    value = primitive * inverse
    left_dot, right_dot = -lower * left**2, -2 * right**2
    difference_dot = width * left * right * (1 - alpha * (lower * left + 2 * right))
    c1_dot, c2_dot = 2 * inverse**2 + 4 * inverse**3, -2 * inverse**2 - 2 * inverse**3
    primitive_dot = math.fsum(
        (
            0.5 * difference_dot * (left + right),
            0.5 * difference * (left_dot + right_dot),
            c1_dot * logarithm,
            c1 * width * left * right,
            c2_dot * difference,
            c2 * difference_dot,
            -width * inverse**2,
        )
    )
    return value, (primitive_dot - value) * inverse / ELECTRON_REST_ENERGY_KEV


def _minimum_transfer(energy: float, binding: float, momentum: float) -> float:
    """Invert monotone pmax with its negative-root cancellation rationalised."""
    if binding == 0 and momentum <= 0:
        return 0.0
    mass = ELECTRON_REST_ENERGY_KEV
    root = math.hypot(mass * momentum, math.sqrt(binding) * math.sqrt(binding + 2 * mass))
    if momentum < 0:
        numerator = binding**2 * (1 - momentum) * (1 + momentum)
        y = numerator / (binding + mass * momentum**2 - momentum * root)
    else:
        y = binding + mass * momentum**2 + momentum * root
    return mass * y / energy / (energy - binding)


def bound_angular_normalisation_scaled_pair(
    energy: float,
    table: BoundComptonTable,
    occupancy_scale: float,
) -> tuple[float, float]:
    """Return the solid-angle integral and its incident-energy derivative.

    Shell endpoints have zero inner angular interval. Differentiation at fixed
    momentum therefore has no moving upper-endpoint term. Quantiles are used
    only to integrate these two physical integrands, not differentiated as
    fixed quadrature marks. Both results use the same shell support and rule.
    """
    if not math.isfinite(energy) or energy <= 0:
        raise TransportError("bound angular normalisation requires finite positive energy")
    values: list[float] = []
    derivatives: list[float] = []
    for shell in table.shells:
        binding, height = shell.binding_kev, shell.momentum_height
        if energy <= binding:
            continue
        upper = shell_momentum_max(energy, -1.0, binding)
        segments = ((-1.0, upper),) if upper <= 0 else ((-1.0, 0.0), (0.0, upper))
        for segment_lower, segment_upper in segments:
            accessibility = profile_interval_mass(segment_lower, segment_upper, height)
            if accessibility <= 0:
                continue
            radius = 1 / 64
            for interval in range(32):
                midpoint = (interval + 0.5) / 32
                for node, weight in zip(_NODES, _WEIGHTS, strict=True):
                    for sign in (-1, 1):
                        coordinate = midpoint + sign * radius * node
                        left, right = coordinate**4, (1 - coordinate) ** 4
                        total = left + right
                        probability, complement = left / total, right / total
                        jacobian = 4 * coordinate**3 * (1 - coordinate) ** 3 / total**2
                        momentum = _profile_interval_inverse_pair(
                            probability,
                            complement,
                            segment_lower,
                            segment_upper,
                            height,
                        )
                        lower = _minimum_transfer(energy, binding, momentum)
                        value, derivative = _kn_interval(energy, lower)
                        moving_limit = lower * (1 / energy + 1 / (energy - binding))
                        derivative += _kn(energy, lower)[0] * moving_limit
                        coefficient = (
                            (shell.occupancy / occupancy_scale)
                            * accessibility
                            * radius
                            * weight
                            * jacobian
                        )
                        values.append(coefficient * value)
                        derivatives.append(coefficient * derivative)
    return 2 * math.pi * math.fsum(values), 2 * math.pi * math.fsum(derivatives)


def bound_angular_normalisation_pair(
    energy: float, table: BoundComptonTable
) -> tuple[float, float]:
    """Physical raw normalisation and derivative; common scale restored last."""
    from .bound_compton import accessible_occupancy_scale
    from .bound_compton_mixed import mixed_raw_normalisation_pair, uses_mixed_conditioning

    if uses_mixed_conditioning(energy, table):
        return mixed_raw_normalisation_pair(energy, table)

    scale = accessible_occupancy_scale(energy, table)
    value, derivative = bound_angular_normalisation_scaled_pair(energy, table, scale)
    return value * scale, derivative * scale


def bound_angular_normalisation(energy: float, table: BoundComptonTable) -> float:
    """Solid-angle normaliser of the declared bound-electron angular law."""
    return bound_angular_normalisation_pair(energy, table)[0]


def bound_normalisation_energy_score(energy: float, table: BoundComptonTable) -> float:
    """Incident-energy derivative of log angular normalisation, in keV^-1."""
    from .bound_compton import accessible_occupancy_scale
    from .bound_compton_mixed import mixed_normalisation, uses_mixed_conditioning
    from .bound_compton_tail import scaled_bound_normalisation, uses_scaled_tail

    if uses_mixed_conditioning(energy, table):
        return mixed_normalisation(energy, table).energy_score

    value, derivative = bound_angular_normalisation_scaled_pair(
        energy, table, accessible_occupancy_scale(energy, table)
    )

    if value == 0.0 or uses_scaled_tail(energy, table):
        return scaled_bound_normalisation(energy, table).energy_score
    if value < 0 or not math.isfinite(value):
        raise TransportError("bound angular law has no finite positive normalisation")
    return derivative / value
