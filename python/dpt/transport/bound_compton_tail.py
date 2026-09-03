"""Conditioned negative-profile tails with anchored scales and bounded proposals.

The raw solid-angle integral can lie below binary64 while the conditional
law remains well resolved. These helpers retain relative shell weights and
profile offsets, and centre derivatives before combining normalised scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from .bound_compton import (
    BoundComptonTable,
    _profile_log_slope,  # pyright: ignore[reportPrivateUsage]
    accessible_occupancy_scale,
    shell_accessibility,
    shell_momentum_max,
)
from .bound_compton_quadrature import _NODES, _WEIGHTS, _kn  # pyright: ignore[reportPrivateUsage]
from .model import ELECTRON_REST_ENERGY_KEV, TransportError

MASS = ELECTRON_REST_ENERGY_KEV


# Select an equivalent scaled representation before raw probabilities lose
# precision in the subnormal range. This changes arithmetic, never support.
_TAIL_MASS_SWITCH = 1e-280


def uses_scaled_tail(energy: float, table: BoundComptonTable, cosine: float = -1.0) -> bool:
    accessible = tuple(shell for shell in table.shells if energy > shell.binding_kev)
    return (
        bool(accessible)
        and all(shell_momentum_max(energy, cosine, shell.binding_kev) < 0.0 for shell in accessible)
        and (
            math.fsum(
                (shell.occupancy / accessible_occupancy_scale(energy, table))
                * shell_accessibility(energy, cosine, shell)
                for shell in accessible
            )
            < _TAIL_MASS_SWITCH
            or any(
                _state_at_cosine(energy, cosine, shell.binding_kev, shell.momentum_height).rate
                * math.ulp(2.0)
                > 1e-12
                for shell in accessible
            )
        )
    )


@dataclass(frozen=True, slots=True)
class ScaledBoundNormalisation:
    """A shared shell scale, normalised integral and centred E derivative."""

    reference_shell: int
    scaled_value: float
    centred_energy_derivative: float
    reference_energy_score: float

    @property
    def energy_score(self) -> float:
        return self.reference_energy_score + self.centred_energy_derivative / self.scaled_value


@dataclass(frozen=True, slots=True)
class _TailState:
    binding: float
    height: float
    product: float
    product_e: float
    q: float
    numerator: float
    momentum: float
    span: float
    cutoff: float
    log_fraction: float
    rate: float
    score: float
    log_mass: float


def _cutoff(span: float, height: float) -> tuple[float, float]:
    # The factorised logarithm remains representable if the exponent product
    # itself underflows. No profile height or physical support is restricted.
    log_size = math.log(2.0) + math.log(height) + math.log(span) + math.log1p(height * (2.0 - span))
    if log_size > 709.782712893384:
        return -math.inf, 0.0
    if log_size > 6.0:
        return -math.exp(log_size), 0.0
    size = 2.0 * height * span * (1.0 + height * (2.0 - span))
    if size == 0.0:
        return -size, log_size
    return -size, math.log(-math.expm1(-size))


def _two_sum(first: float, second: float) -> tuple[float, float]:
    total = first + second
    temporary = total - first
    error = (first - (total - temporary)) + (second - temporary)
    return total, error


def _product_error(first: float, second: float) -> float:
    # Error-free Dekker product; kinematic inputs are bounded photon energies,
    # transfers and masses, independent of arbitrary shell profile heights.
    splitter = 134217729.0
    first_split, second_split = splitter * first, splitter * second
    first_high = first_split - (first_split - first)
    second_high = second_split - (second_split - second)
    first_low, second_low = first - first_high, second - second_high
    product = first * second
    return (
        (first_high * second_high - product) + first_high * second_low + first_low * second_high
    ) + first_low * second_low


def _kinematic_pair(
    energy: float, transfer: float, binding: float, transfer_error: float
) -> tuple[float, float]:
    difference, difference_error = _two_sum(energy, -binding)
    product = energy * difference
    product_error = _product_error(energy, difference) + energy * difference_error
    transferred = product * transfer
    transferred_error = (
        _product_error(product, transfer)
        + product_error * transfer
        + product * transfer_error
        + product_error * transfer_error
    )
    divided = transferred / MASS
    divided_product = divided * MASS
    division_error = (
        (transferred - divided_product) - _product_error(divided, MASS) + transferred_error
    ) / MASS
    numerator, numerator_error = _two_sum(divided, -binding)
    numerator += numerator_error + division_error
    square, square_error = _two_sum(2.0 * transferred, binding * binding)
    square_error += 2.0 * transferred_error + _product_error(binding, binding)
    q = math.sqrt(square)
    q_error = ((square - q * q) - _product_error(q, q) + square_error) / (2.0 * q)
    return numerator, q + q_error


def joint_kinematic_pair(energy: float, cosine: float, outgoing: float) -> tuple[float, float]:
    transfer, transfer_error = _two_sum(1.0, -cosine)
    difference, difference_error = _two_sum(energy, -outgoing)
    product = energy * outgoing
    product_error = _product_error(energy, outgoing)
    transferred = product * transfer
    transferred_error = (
        _product_error(product, transfer)
        + product_error * transfer
        + product * transfer_error
        + product_error * transfer_error
    )
    divided = transferred / MASS
    division_error = (
        (transferred - divided * MASS) - _product_error(divided, MASS) + transferred_error
    ) / MASS
    numerator, numerator_error = _two_sum(divided, -difference)
    numerator += numerator_error + division_error - difference_error
    square, square_error = _two_sum(2.0 * transferred, difference * difference)
    square_error += (
        2.0 * transferred_error
        + _product_error(difference, difference)
        + 2.0 * difference * difference_error
        + difference_error * difference_error
    )
    q = math.sqrt(square)
    q_error = ((square - q * q) - _product_error(q, q) + square_error) / (2.0 * q)
    return numerator, q + q_error


def _state_with_transfer_error(
    energy: float, transfer: float, binding: float, height: float, transfer_error: float
) -> _TailState:
    product = energy * (energy - binding)
    product_e = 2.0 * energy - binding
    numerator, q = _kinematic_pair(energy, transfer, binding, transfer_error)
    momentum = numerator / q
    span = product * transfer / q * (1.0 / MASS + 2.0 / (q + binding))
    cutoff, log_fraction = _cutoff(span, height)
    p_t_over_span = (1.0 / MASS - momentum / q) / (transfer * (1.0 / MASS + 2.0 / (q + binding)))
    cutoff_ratio = 1.0 if cutoff == 0.0 else -cutoff / -math.expm1(cutoff)
    rate = (
        p_t_over_span
        * (1.0 - 2.0 * height * momentum)
        / (1.0 + height * (2.0 - span))
        * cutoff_ratio
    )
    score = rate * transfer * product_e / product
    if not math.isfinite(cutoff):
        # The truncation correction exp(-D) has rounded to zero. Compute
        # derivatives from the endpoint exponent, without first forming D.
        p_t = product * (1.0 / (MASS * q) - numerator / (q * q * q))
        p_e = product_e * transfer * (1.0 / (MASS * q) - numerator / (q * q * q))
        rate = 2.0 * height * ((1.0 - 2.0 * height * momentum) * p_t)
        score = 2.0 * height * ((1.0 - 2.0 * height * momentum) * p_e)
    a = -height * momentum
    log_mass = -2.0 * a * (1.0 + a) + log_fraction - math.log(2.0)
    return _TailState(
        binding,
        height,
        product,
        product_e,
        q,
        numerator,
        momentum,
        span,
        cutoff,
        log_fraction,
        rate,
        score,
        log_mass,
    )


def _state(energy: float, transfer: float, binding: float, height: float) -> _TailState:
    return _state_with_transfer_error(energy, transfer, binding, height, 0.0)


def _state_at_cosine(energy: float, cosine: float, binding: float, height: float) -> _TailState:
    transfer, error = _two_sum(1.0, -cosine)
    return _state_with_transfer_error(energy, transfer, binding, height, error)


def _fraction_score(state: _TailState, transfer: float) -> float:
    if state.cutoff < -745.0:
        return 0.0
    q_e = state.product_e * transfer / state.q
    factor = 1.0 / MASS + 2.0 / (state.q + state.binding)
    log_span_e = (
        state.product_e / state.product
        - q_e / state.q
        - 2.0 * q_e / ((state.q + state.binding) ** 2 * factor)
    )
    phi = 1.0
    if state.cutoff != 0.0:
        phi = state.cutoff * math.exp(state.cutoff) / math.expm1(state.cutoff)
    correction = (1.0 + state.height * (2.0 - 2.0 * state.span)) / (
        1.0 + state.height * (2.0 - state.span)
    )
    return log_span_e * correction * phi


def _binding_delta(
    energy: float, transfer: float, first: _TailState, second: _TailState
) -> tuple[float, float]:
    difference = first.binding - second.binding
    qsum = first.q + second.q
    v = first.binding + second.binding - 2.0 * energy * transfer
    factor = -(energy * transfer / MASS + 1.0) * second.q - second.numerator * v / qsum
    first_q_e = first.product_e * transfer / first.q
    second_q_e = second.product_e * transfer / second.q
    second_n_e = second.product_e * transfer / MASS
    factor_e = (
        -transfer / MASS * second.q
        - (energy * transfer / MASS + 1.0) * second_q_e
        - second_n_e * v / qsum
        + 2.0 * transfer * second.numerator / qsum
        + second.numerator * v * (first_q_e + second_q_e) / (qsum * qsum)
    )
    coefficient = difference / (first.q * second.q)
    delta = coefficient * factor
    delta_e = coefficient * (factor_e - factor * (first_q_e / first.q + second_q_e / second.q))
    return delta, delta_e


def _relative_shell(
    energy: float, transfer: float, first: _TailState, reference: _TailState
) -> tuple[float, float]:
    delta, delta_e = _binding_delta(energy, transfer, first, reference)
    height_difference = first.height - reference.height
    a_difference = -height_difference * reference.momentum - first.height * delta
    a_sum = -first.height * first.momentum - reference.height * reference.momentum
    log_weight = -2.0 * a_difference * (1.0 + a_sum) + first.log_fraction - reference.log_fraction
    reference_p_e = (
        reference.product_e
        * transfer
        * (1.0 / (MASS * reference.q) - reference.momentum / (reference.q * reference.q))
    )
    first_product = height_difference * reference_p_e + first.height * delta_e
    second_product = height_difference * (
        first.height + reference.height
    ) * reference.momentum * reference_p_e + first.height * first.height * (
        delta * reference_p_e + reference.momentum * delta_e + delta * delta_e
    )
    centre = (
        2.0 * first_product
        - 4.0 * second_product
        + _fraction_score(first, transfer)
        - _fraction_score(reference, transfer)
    )
    return log_weight, centre


def _angle_relative_at_transfer(
    energy: float, state: _TailState, distance: float, transfer: float
) -> tuple[float, float]:
    q = math.sqrt(state.q * state.q - 2.0 * state.product * distance)
    factor = state.q / MASS - 2.0 * state.numerator / (state.q + q)
    coefficient = state.product * distance / (state.q * q)
    delta = coefficient * factor
    q0_e, q_e = 2.0 * state.product_e / state.q, state.product_e * (2.0 - distance) / q
    factor_e = (
        q0_e / MASS
        - 4.0 * state.product_e / (MASS * (state.q + q))
        + 2.0 * state.numerator * (q0_e + q_e) / (state.q + q) ** 2
    )
    delta_e = (
        delta * (state.product_e / state.product - q0_e / state.q - q_e / q)
        + coefficient * factor_e
    )
    p0_e = 2.0 * state.product_e * (1.0 / (MASS * state.q) - state.momentum / (state.q * state.q))
    log_difference = (
        -2.0
        * state.height
        * delta
        * (1.0 - 2.0 * state.height * state.momentum + state.height * delta)
    )
    log_difference_e = (
        -2.0
        * state.height
        * (
            delta_e * (1.0 - 2.0 * state.height * state.momentum + 2.0 * state.height * delta)
            - 2.0 * state.height * delta * p0_e
        )
    )
    if transfer <= 0.0:
        return -math.inf, 0.0
    angle = _state(energy, transfer, state.binding, state.height)
    log_relative = log_difference + angle.log_fraction - state.log_fraction
    centred_score = (
        log_difference_e + _fraction_score(angle, transfer) - _fraction_score(state, 2.0)
    )
    return log_relative, centred_score


def _angle_relative(energy: float, state: _TailState, distance: float) -> tuple[float, float]:
    return _angle_relative_at_transfer(energy, state, distance, 2.0 - distance)


def _reference(energy: float, table: BoundComptonTable, transfer: float = 2.0) -> int:
    if transfer <= 0.0:
        raise TransportError("bound conditional tail has no accessible angular support")
    if any(
        shell_momentum_max(energy, 1.0 - transfer, shell.binding_kev) >= 0.0
        for shell in table.shells
        if energy > shell.binding_kev
    ):
        raise TransportError("this underflow-conditioned bound law needs occupancy scaling")
    candidates = [
        (
            math.log(shell.occupancy / accessible_occupancy_scale(energy, table))
            + _state(energy, transfer, shell.binding_kev, shell.momentum_height).log_mass,
            i,
        )
        for i, shell in enumerate(table.shells)
        if energy > shell.binding_kev
    ]
    if not candidates or transfer <= 0.0:
        raise TransportError("bound conditional tail has no accessible shell")
    return max(candidates)[1]


def _envelope_integral(rate: float) -> float:
    return -math.expm1(-2.0 * rate) / rate


def _proposal_distance(fraction: float, complement: float, rate: float) -> float:
    if rate > 0.5:
        # Retaining both fractions also handles a residual quantile rounded to1.
        return -math.log(complement + fraction * math.exp(-2.0 * rate)) / rate
    return -math.log1p(-fraction * -math.expm1(-2.0 * rate)) / rate


def _proposal_cosine(fraction: float, complement: float, rate: float) -> float:
    if rate > 1.0:
        return -1.0 + _proposal_distance(fraction, complement, rate)
    if fraction < 0.125:
        return -1.0 - math.log1p(-fraction * -math.expm1(-2.0 * rate)) / rate
    if complement < 0.125:
        return 1.0 - math.log1p(complement * math.expm1(2.0 * rate)) / rate
    # Centre the inverse at c=0; forming -1+distance loses cosine precision.
    square = rate * rate
    cosh_term = square / 2.0
    sinh_term = rate
    cosh_minus_one = cosh_term
    sinh = sinh_term
    for order in range(1, 20):
        cosh_term *= square / ((2.0 * order + 1.0) * (2.0 * order + 2.0))
        sinh_term *= square / ((2.0 * order) * (2.0 * order + 1.0))
        cosh_minus_one += cosh_term
        sinh += sinh_term
    return -math.log1p(cosh_minus_one + (complement - fraction) * sinh) / rate


def _proposal_transfer(fraction: float, complement: float, rate: float, cosine: float) -> float:
    if rate <= 1.0 and complement < 0.125:
        return math.log1p(complement * math.expm1(2.0 * rate)) / rate
    return 1.0 - cosine


@lru_cache(maxsize=128)
def scaled_bound_normalisation(energy: float, table: BoundComptonTable) -> ScaledBoundNormalisation:
    reference_index = _reference(energy, table)
    shell0 = table.shells[reference_index]
    reference = _state(energy, 2.0, shell0.binding_kev, shell0.momentum_height)
    values: list[float] = []
    derivatives: list[float] = []
    for shell in table.shells:
        if energy <= shell.binding_kev:
            continue
        state = _state(energy, 2.0, shell.binding_kev, shell.momentum_height)
        log_back, back_centre = _relative_shell(energy, 2.0, state, reference)
        weight = (
            (shell.occupancy / accessible_occupancy_scale(energy, table))
            * math.exp(log_back)
            * _envelope_integral(state.rate)
        )
        for panel in range(32):
            midpoint, radius = (panel + 0.5) / 32.0, 1.0 / 64.0
            for node, gauss_weight in zip(_NODES, _WEIGHTS, strict=True):
                for sign in (-1.0, 1.0):
                    x = midpoint + sign * radius * node
                    left, right = x**4, (1.0 - x) ** 4
                    total = left + right
                    fraction, complement = left / total, right / total
                    distance = _proposal_distance(fraction, complement, state.rate)
                    log_relative, centre = _angle_relative(energy, state, distance)
                    proposal_ratio = math.exp(log_relative + state.rate * distance)
                    kn, kn_e = _kn(energy, 2.0 - distance)
                    jacobian = 4.0 * x**3 * (1.0 - x) ** 3 / total**2
                    coefficient = weight * radius * gauss_weight * jacobian * proposal_ratio
                    values.append(coefficient * kn)
                    derivatives.append(coefficient * (kn_e + kn * (back_centre + centre)))
    value = 2.0 * math.pi * math.fsum(values)
    derivative = 2.0 * math.pi * math.fsum(derivatives)
    if not math.isfinite(value) or value <= 0.0 or not math.isfinite(derivative):
        raise TransportError("bound conditional tail exceeds binary64 normalisation range")
    return ScaledBoundNormalisation(reference_index, value, derivative, reference.score)


def tail_angle_proposal(
    energy: float, table: BoundComptonTable, uniform: float
) -> tuple[float, float]:
    reference_index = _reference(energy, table)
    shell0 = table.shells[reference_index]
    reference = _state(energy, 2.0, shell0.binding_kev, shell0.momentum_height)
    states = [
        _state(energy, 2.0, shell.binding_kev, shell.momentum_height)
        if energy > shell.binding_kev
        else None
        for shell in table.shells
    ]
    weights = [
        (shell.occupancy / accessible_occupancy_scale(energy, table))
        * math.exp(_relative_shell(energy, 2.0, state, reference)[0])
        * _envelope_integral(state.rate)
        if state is not None
        else 0.0
        for shell, state in zip(table.shells, states, strict=True)
    ]
    total = math.fsum(weights)
    target = uniform * total
    cumulative = 0.0
    distance = math.nan
    cosine = math.nan
    transfer = math.nan
    for index, (state, weight) in enumerate(zip(states, weights, strict=True)):
        previous = cumulative
        cumulative += weight
        if target < cumulative and state is not None and weight > 0.0:
            trailing = math.fsum(weights[index + 1 :])
            fraction = (target - previous) / weight
            complement = ((1.0 - uniform) * total - trailing) / weight
            cosine = _proposal_cosine(fraction, complement, state.rate)
            distance = cosine + 1.0
            transfer = _proposal_transfer(fraction, complement, state.rate, cosine)
            break
    physical: list[float] = []
    envelope: list[float] = []
    for shell, state in zip(table.shells, states, strict=True):
        if state is None:
            continue
        log_back = _relative_shell(energy, 2.0, state, reference)[0]
        physical.append(
            (shell.occupancy / accessible_occupancy_scale(energy, table))
            * math.exp(log_back + _angle_relative_at_transfer(energy, state, distance, transfer)[0])
        )
        envelope.append(
            (shell.occupancy / accessible_occupancy_scale(energy, table))
            * math.exp(log_back - state.rate * distance)
        )
    acceptance = 0.5 * _kn(energy, transfer)[0] * math.fsum(physical) / math.fsum(envelope)
    return cosine, acceptance


def _momentum_deficit(energy: float, cosine: float, state: _TailState, uniform: float) -> float:
    complement = 1.0 - uniform
    relative_mass = -math.expm1(state.cutoff)
    product = complement * relative_mass
    if product <= 0.5:
        offset = -math.log1p(-product)
        ratio = 1.0 if product == 0.0 else offset / product
        fraction_ratio = 1.0 if state.cutoff == 0.0 else relative_mass / -state.cutoff
        offset_over_height = (
            ratio
            * complement
            * fraction_ratio
            * 2.0
            * state.span
            * (1.0 + state.height * (2.0 - state.span))
        )
    else:
        offset = -math.log(uniform + complement * math.exp(state.cutoff))
        offset_over_height = offset / state.height
    slope = 1.0 - 2.0 * state.height * state.momentum
    return offset_over_height / (math.hypot(slope, math.sqrt(2.0 * offset)) + slope)


def _energy_deficit(energy: float, cosine: float, binding: float, momentum_deficit: float) -> float:
    upper = (energy - binding) / energy
    lower, higher = 0.0, upper
    transfer = 1.0 - cosine
    alpha = energy / MASS
    # Integrate dp/dr over an endpoint deficit instead of subtracting momenta.
    # The derivative has an explicit factor1-c, retaining forward-angle support.
    for _ in range(80):
        deficit = (lower + higher) * 0.5
        midpoint, radius = upper - deficit * 0.5, deficit * 0.5
        terms: list[float] = []
        for node, weight in zip(_NODES, _WEIGHTS, strict=True):
            for sign in (-1.0, 1.0):
                r = midpoint + sign * radius * node
                q2 = (1.0 - r) ** 2 + 2.0 * r * transfer
                derivative = (
                    transfer * (1.0 + alpha + r * (1.0 - alpha * cosine)) / (q2 * math.sqrt(q2))
                )
                terms.append(radius * weight * derivative)
        if math.fsum(terms) < momentum_deficit:
            lower = deficit
        else:
            higher = deficit
    return energy * (lower + higher) * 0.5


def _momentum_span(state: _TailState, uniform: float) -> float:
    size = -state.cutoff
    if size <= 0.5:
        relative = math.expm1(size)
        product = uniform * relative
        offset = math.log1p(product)
        logarithm_ratio = 1.0 if product == 0.0 else offset / product
        fraction_ratio = 1.0 if size == 0.0 else relative / size
        offset_over_height = (
            logarithm_ratio
            * uniform
            * fraction_ratio
            * 2.0
            * state.span
            * (1.0 + state.height * (2.0 - state.span))
        )
    else:
        log_product = math.log(uniform) + size + math.log1p(-math.exp(-size))
        if log_product > 0.0:
            offset = log_product + math.log1p(math.exp(-log_product))
        else:
            offset = math.log1p(math.exp(log_product))
        offset_over_height = offset / state.height
    if offset >= 2.2250738585072014e-308:
        offset_over_height = offset / state.height
    slope = 1.0 + 2.0 * state.height
    return offset_over_height / (math.sqrt(slope * slope - 2.0 * offset) + slope)


def _energy_from_span(energy: float, cosine: float, binding: float, momentum_span: float) -> float:
    lower, upper = 0.0, (energy - binding) / energy
    transfer = 1.0 - cosine
    alpha = energy / MASS
    for _ in range(120):
        ratio = (lower + upper) * 0.5
        q = math.sqrt((1.0 - ratio) ** 2 + 2.0 * ratio * transfer)
        span = ratio * transfer / q * (alpha + 2.0 / (q + 1.0 - ratio))
        if span < momentum_span:
            lower = ratio
        else:
            upper = ratio
    return energy * (lower + upper) * 0.5


def _threshold_energy(energy: float, cosine: float, state: _TailState, uniform: float) -> float:
    # log(1-u+u*exp(D))/D is the Bernoulli cumulant series. For D<0.001
    # terms through D**5 leave relative error below binary64 resolution. Retain the endpoint
    # fraction instead of an individually rounded, sub-ULP momentum span.
    size = -state.cutoff
    u = uniform
    polynomial = (1.0 - 30.0 * u + 150.0 * u * u - 240.0 * u**3 + 120.0 * u**4) / 720.0
    polynomial = (1.0 - 2.0 * u) * (1.0 - 12.0 * u + 12.0 * u * u) / 120.0 + size * polynomial
    polynomial = (1.0 - 6.0 * u + 6.0 * u * u) / 24.0 + size * polynomial
    polynomial = (1.0 - 2.0 * u) / 6.0 + size * polynomial
    polynomial = 0.5 + size * polynomial
    fraction, fraction_error = _two_sum(u, u * (1.0 - u) * size * polynomial)
    upper = energy - state.binding
    ratio_upper = upper / energy
    curvature = cosine + (1.0 + cosine) / (2.0 * (1.0 + energy / MASS))
    correction = (
        fraction
        * (1.0 - fraction)
        * (curvature * ratio_upper - state.height * state.span / (1.0 + 2.0 * state.height))
    )
    fraction, fraction_error2 = _two_sum(fraction, fraction_error + correction)
    outgoing = upper * fraction
    return outgoing + (_product_error(upper, fraction) + upper * fraction_error2)


def tail_conditional_energy(
    energy: float,
    cosine: float,
    table: BoundComptonTable,
    shell_uniform: float,
    energy_uniform: float,
) -> tuple[float, int, float]:
    transfer = 1.0 - cosine
    reference_index = _reference(energy, table, transfer)
    shell0 = table.shells[reference_index]
    reference = _state_at_cosine(energy, cosine, shell0.binding_kev, shell0.momentum_height)
    states = [
        _state_at_cosine(energy, cosine, shell.binding_kev, shell.momentum_height)
        if energy > shell.binding_kev
        else None
        for shell in table.shells
    ]
    weights = [
        (shell.occupancy / accessible_occupancy_scale(energy, table))
        * math.exp(_relative_shell(energy, transfer, state, reference)[0])
        if state is not None
        else 0.0
        for shell, state in zip(table.shells, states, strict=True)
    ]
    target = shell_uniform * math.fsum(weights)
    cumulative = 0.0
    for index, (state, weight) in enumerate(zip(states, weights, strict=True)):
        cumulative += weight
        if target < cumulative and state is not None and weight > 0.0:
            size = -state.cutoff
            if size <= 64.0 or math.log(energy_uniform) < -0.5 * size:
                span = _momentum_span(state, energy_uniform)
                if (energy - state.binding) / energy < 1e-12 and size < 0.001:
                    outgoing = _threshold_energy(energy, cosine, state, energy_uniform)
                else:
                    outgoing = _energy_from_span(energy, cosine, state.binding, span)
                return outgoing, index, -1.0 + span
            deficit = _momentum_deficit(energy, cosine, state, energy_uniform)
            outgoing = (
                energy - state.binding - _energy_deficit(energy, cosine, state.binding, deficit)
            )
            return outgoing, index, state.momentum - deficit
    raise TransportError("bound conditional tail shell selection failed")


def tail_phase_pdf(
    energy: float,
    cosine: float,
    table: BoundComptonTable,
    normalisation: ScaledBoundNormalisation | None = None,
) -> float:
    if not -1.0 <= cosine < 1.0:
        return 0.0
    normalisation = normalisation or scaled_bound_normalisation(energy, table)
    shell0 = table.shells[normalisation.reference_shell]
    reference = _state(energy, 2.0, shell0.binding_kev, shell0.momentum_height)
    values: list[float] = []
    for shell in table.shells:
        if energy > shell.binding_kev:
            state = _state(energy, 2.0, shell.binding_kev, shell.momentum_height)
            log_back = _relative_shell(energy, 2.0, state, reference)[0]
            log_angle = _angle_relative(energy, state, cosine + 1.0)[0]
            values.append(
                (shell.occupancy / accessible_occupancy_scale(energy, table))
                * math.exp(log_back + log_angle)
            )
    return _kn(energy, 1.0 - cosine)[0] * math.fsum(values) / normalisation.scaled_value


def _local_delta(
    energy: float, cosine: float, outgoing: float, reference: _TailState
) -> tuple[float, float]:
    distance = cosine + 1.0
    deficit = energy - reference.binding - outgoing
    q0 = reference.q
    n0 = reference.numerator
    q = math.sqrt((energy - outgoing) ** 2 + 2.0 * energy * outgoing * (1.0 - cosine))
    n_delta = -deficit * (1.0 + 2.0 * energy / MASS) - energy * outgoing * distance / MASS
    q2_delta = (
        deficit * (2.0 * reference.binding + deficit - 4.0 * energy)
        - 2.0 * energy * outgoing * distance
    )
    factor = n_delta * q0 - n0 * q2_delta / (q + q0)
    delta = factor / (q * q0)
    q0_e = 2.0 * reference.product_e / q0
    q_e = (energy + outgoing * ((1.0 - cosine) - 1.0)) / q
    n_delta_e = -(1.0 + 2.0 * energy / MASS) - 2.0 * deficit / MASS - outgoing * distance / MASS
    q2_delta_e = 2.0 * reference.binding - 2.0 * deficit - 4.0 * energy - 2.0 * outgoing * distance
    factor_e = (
        n_delta_e * q0
        + n_delta * q0_e
        - 2.0 * reference.product_e / MASS * q2_delta / (q + q0)
        - n0 * q2_delta_e / (q + q0)
        + n0 * q2_delta * (q_e + q0_e) / (q + q0) ** 2
    )
    delta_e = (factor_e - factor * (q_e / q + q0_e / q0)) / (q * q0)
    return delta, delta_e


def _joint_profile(
    energy: float,
    cosine: float,
    outgoing: float,
    table: BoundComptonTable,
    normalisation: ScaledBoundNormalisation,
) -> tuple[float, float, float, float]:
    shell0 = table.shells[normalisation.reference_shell]
    reference = _state(energy, 2.0, shell0.binding_kev, shell0.momentum_height)
    delta, delta_e = _local_delta(energy, cosine, outgoing, reference)
    momentum = reference.momentum + delta
    reference_p_e = (
        2.0
        * reference.product_e
        * (1.0 / (MASS * reference.q) - reference.momentum / reference.q**2)
    )
    momentum_e = reference_p_e + delta_e
    values: list[float] = []
    slopes: list[float] = []
    centres: list[float] = []
    for shell in table.shells:
        if outgoing >= energy - shell.binding_kev:
            continue
        h, h0 = shell.momentum_height, reference.height
        difference = h - h0
        a_difference = -difference * reference.momentum - h * delta
        a_sum = -h * momentum - h0 * reference.momentum
        exponent = -2.0 * a_difference * (1.0 + a_sum)
        log_factor = (
            math.log(2.0) + math.log(h) + math.log1p(-2.0 * h * momentum) - reference.log_fraction
        )
        value = (shell.occupancy / accessible_occupancy_scale(energy, table)) * math.exp(
            exponent + log_factor
        )
        slope = _profile_log_slope(momentum, h)
        first_product = difference * reference_p_e + h * delta_e
        second_product = difference * (h + h0) * reference.momentum * reference_p_e + h * h * (
            delta * reference_p_e + reference.momentum * delta_e + delta * delta_e
        )
        centre = (
            2.0 * first_product
            - 4.0 * second_product
            - 2.0 * h / (1.0 - 2.0 * h * momentum) * momentum_e
            - _fraction_score(reference, 2.0)
        )
        values.append(value)
        slopes.append(value * slope)
        centres.append(value * centre)
    density = math.fsum(values)
    if density <= 0.0:
        return 0.0, 0.0, 0.0, momentum
    return density, math.fsum(slopes) / density, math.fsum(centres) / density, momentum


def tail_joint_pdf(
    energy: float,
    cosine: float,
    outgoing: float,
    table: BoundComptonTable,
    normalisation: ScaledBoundNormalisation | None = None,
) -> float:
    if not 0.0 < outgoing < energy or not -1.0 <= cosine < 1.0:
        return 0.0
    normalisation = normalisation or scaled_bound_normalisation(energy, table)
    density = _joint_profile(energy, cosine, outgoing, table, normalisation)[0]
    ratio = outgoing / energy
    factor = 1.0 + energy / MASS * (1.0 - cosine)
    q2 = (1.0 - ratio) ** 2 + 2.0 * ratio * (1.0 - cosine)
    jacobian = (factor - cosine + ratio * (1.0 - factor * cosine)) / (energy * q2 * math.sqrt(q2))
    return _kn(energy, 1.0 - cosine)[0] * density * jacobian / normalisation.scaled_value


def tail_joint_scores(
    energy: float,
    cosine: float,
    outgoing: float,
    table: BoundComptonTable,
    normalisation: ScaledBoundNormalisation | None = None,
) -> tuple[float, float, float]:
    normalisation = normalisation or scaled_bound_normalisation(energy, table)
    density, profile_score, centre, momentum = _joint_profile(
        energy, cosine, outgoing, table, normalisation
    )
    if density <= 0.0:
        raise TransportError("joint conditional tail score has zero physical density")
    ratio = outgoing / energy
    k = energy / MASS
    factor = 1.0 + k * (1.0 - cosine)
    q2 = (1.0 - ratio) ** 2 + 2.0 * ratio * (1.0 - cosine)
    q = math.sqrt(q2)
    n = factor - cosine + ratio * (1.0 - factor * cosine)
    tau = 1.0 / factor
    kn_term = tau + factor - 1.0 + cosine * cosine
    kn_c = 2.0 * k * tau + (k * tau * tau - k + 2.0 * cosine) / kn_term
    p_c = -k * ratio / q + momentum * ratio / q2
    n_c = -k - 1.0 + ratio * (k * cosine - factor)
    score_c = kn_c + profile_score * p_c + n_c / n + 3.0 * ratio / q2
    p_r = n / (q * q2)
    score_outgoing = (
        profile_score * p_r + (1.0 - factor * cosine) / n - 3.0 * (ratio - cosine) / q2
    ) / energy
    a_e = (1.0 - cosine) / MASS
    n_e = a_e * (1.0 - ratio * cosine) - ratio / energy * (1.0 - factor * cosine)
    kn_e = -2.0 * tau * a_e + (1.0 - tau * tau) * a_e / kn_term
    score_incident = (
        kn_e
        + centre
        + n_e / n
        - 1.0 / energy
        + 3.0 * ratio * (ratio - cosine) / (energy * q2)
        - normalisation.centred_energy_derivative / normalisation.scaled_value
    )
    return score_c, score_outgoing, score_incident
