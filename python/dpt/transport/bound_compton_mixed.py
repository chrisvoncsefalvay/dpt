"""Log-scaled mixtures of the declared bound-electron shell profiles."""

# The sibling module supplies reviewed internal arithmetic, not a second law.
# pyright: reportPrivateUsage=false
from __future__ import annotations

import math
from functools import lru_cache

from . import bound_compton_tail as tail
from .bound_compton import (
    BoundComptonTable,
    _profile_interval_inverse_pair,
    _profile_log_slope,
    impulse_energy,
    profile_interval_inverse,
    profile_interval_mass,
    shell_momentum_max,
)
from .bound_compton_quadrature import (
    _NODES,
    _WEIGHTS,
    _kn,
    _kn_interval,
    _minimum_transfer,
)
from .model import TransportError

MASS = tail.MASS


def _log_occupancy_ratio(first: float, second: float) -> float:
    ratio = first / second
    if ratio > 0.0 and math.isfinite(ratio):
        return math.log(ratio)
    return math.log(first) - math.log(second)


def _profile_log(momentum: float, height: float) -> float:
    amplitude = height * abs(momentum)
    return -2.0 * amplitude * (1.0 + amplitude)


def _profile_log_density(momentum: float, height: float) -> float:
    amplitude = height * abs(momentum)
    return math.log(height) + math.log1p(2.0 * amplitude) + _profile_log(momentum, height)


def _mass_pair(energy: float, cosine: float, binding: float, height: float) -> tuple[float, float]:
    if energy <= binding or cosine >= 1.0:
        return -math.inf, 0.0
    transfer, transfer_error = tail._two_sum(1.0, -cosine)
    numerator, q = tail._kinematic_pair(energy, transfer, binding, transfer_error)
    momentum = numerator / q
    if momentum < 0.0:
        log_profile = _profile_log(momentum, height)
        if not math.isfinite(log_profile):
            return -math.inf, 0.0
        # When 1-exp(L(-1)-L(pmax)) rounds to one, the lower truncation
        # contributes no representable relative mass. Avoid constructing its
        # potentially overflowing absolute cutoff before a component is weighted.
        span = 1.0 + momentum
        if span > 0.0:
            log_size = (
                math.log(2.0)
                + math.log(height)
                + math.log(span)
                + math.log1p(height * (2.0 - span))
            )
            if log_size > 6.0:
                numerator, q = tail._kinematic_pair(energy, 1.0 - cosine, binding, 0.0)
                p_e = (
                    (2.0 * energy - binding)
                    * (1.0 - cosine)
                    * (1.0 / (MASS * q) - numerator / (q * q * q))
                )
                return log_profile - math.log(2.0), 2.0 * height * (
                    (1.0 - 2.0 * height * momentum) * p_e
                )
        state = tail._state_at_cosine(energy, cosine, binding, height)
        return state.log_mass, state.score
    mass = profile_interval_mass(-1.0, momentum, height)
    numerator, q = tail._kinematic_pair(energy, 1.0 - cosine, binding, 0.0)
    p_e = (2.0 * energy - binding) * (1.0 - cosine) * (1.0 / (MASS * q) - numerator / (q * q * q))
    log_mass = math.log(mass)
    return log_mass, math.exp(_profile_log_density(momentum, height) - log_mass) * p_e


def _back_rate(energy: float, binding: float, height: float) -> float:
    if shell_momentum_max(energy, -1.0, binding) >= 0.0:
        return 0.0
    return tail._state(energy, 2.0, binding, height).rate


def _envelope_integral(rate: float) -> float:
    return 2.0 if rate == 0.0 else tail._envelope_integral(rate)


def mixed_reference(energy: float, table: BoundComptonTable, cosine: float = -1.0) -> int:
    candidates: list[tuple[float, int]] = []
    for index, shell in enumerate(table.shells):
        if energy <= shell.binding_kev:
            continue
        log_mass, _ = _mass_pair(energy, cosine, shell.binding_kev, shell.momentum_height)
        if log_mass == -math.inf:
            continue
        envelope = _envelope_integral(_back_rate(energy, shell.binding_kev, shell.momentum_height))
        candidates.append((math.log(shell.occupancy) + log_mass + math.log(envelope), index))
    if not candidates:
        raise TransportError("bound mixture has no accessible shell")
    return max(candidates, key=lambda item: item[0])[1]


def uses_mixed_conditioning(energy: float, table: BoundComptonTable) -> bool:
    accessible = [s for s in table.shells if energy > s.binding_kev]
    if not accessible:
        return False
    positive_endpoint = any(
        shell_momentum_max(energy, -1.0, s.binding_kev) >= 0.0 for s in accessible
    )
    logs = [
        math.log(s.occupancy) + _mass_pair(energy, -1.0, s.binding_kev, s.momentum_height)[0]
        for s in accessible
    ]
    for shell, log_weight in zip(accessible, logs, strict=True):
        if not math.isfinite(log_weight):
            if any(math.isfinite(value) for value in logs):
                return True
        elif shell_momentum_max(energy, -1.0, shell.binding_kev) < 0.0:
            _, q = tail._kinematic_pair(energy, 2.0, shell.binding_kev, 0.0)
            span = (
                energy
                * (energy - shell.binding_kev)
                * 2.0
                / q
                * (1.0 / MASS + 2.0 / (q + shell.binding_kev))
            )
            log_size = (
                math.log(2.0)
                + math.log(shell.momentum_height)
                + math.log(span)
                + math.log1p(shell.momentum_height * (2.0 - span))
            )
            if log_size > 709.782712893384:
                return True
    # This is an arithmetic dispatch, not a restriction on a physical profile.
    if max(logs) >= math.log(1e-280):
        return False
    if positive_endpoint:
        return True
    return any(shell.occupancy < 1e-280 for shell in accessible)


def _back_relative(
    energy: float, binding: float, height: float, reference_binding: float, reference_height: float
) -> tuple[float, float]:
    log_mass, score = _mass_pair(energy, -1.0, binding, height)
    if log_mass == -math.inf:
        return -math.inf, 0.0
    if (
        shell_momentum_max(energy, -1.0, binding) < 0.0
        and shell_momentum_max(energy, -1.0, reference_binding) < 0.0
    ):
        state = tail._state(energy, 2.0, binding, height)
        reference = tail._state(energy, 2.0, reference_binding, reference_height)
        return tail._relative_shell(energy, 2.0, state, reference)
    ref_log_mass, ref_score = _mass_pair(energy, -1.0, reference_binding, reference_height)
    return log_mass - ref_log_mass, score - ref_score


def _angle_relative(
    energy: float,
    cosine: float,
    binding: float,
    height: float,
    reference_binding: float,
    reference_height: float,
) -> tuple[float, float]:
    back_log, back_centre = _back_relative(
        energy, binding, height, reference_binding, reference_height
    )
    if back_log == -math.inf:
        return -math.inf, 0.0
    if shell_momentum_max(energy, -1.0, binding) < 0.0:
        state = tail._state(energy, 2.0, binding, height)
        angle_log, angle_centre = tail._angle_relative_at_transfer(
            energy, state, cosine + 1.0, 1.0 - cosine
        )
        return back_log + angle_log, back_centre + angle_centre
    local_log, local_score = _mass_pair(energy, cosine, binding, height)
    reference_log, reference_score = _mass_pair(energy, -1.0, reference_binding, reference_height)
    return local_log - reference_log, local_score - reference_score


def _local_mass_relative(
    energy: float,
    cosine: float,
    binding: float,
    height: float,
    reference_binding: float,
    reference_height: float,
) -> tuple[float, float]:
    local_log, local_score = _mass_pair(energy, cosine, binding, height)
    if local_log == -math.inf:
        return -math.inf, 0.0
    if (
        shell_momentum_max(energy, cosine, binding) < 0.0
        and shell_momentum_max(energy, cosine, reference_binding) < 0.0
    ):
        state = tail._state_at_cosine(energy, cosine, binding, height)
        reference = tail._state_at_cosine(energy, cosine, reference_binding, reference_height)
        return tail._relative_shell(energy, 1.0 - cosine, state, reference)
    reference_log, reference_score = _mass_pair(energy, cosine, reference_binding, reference_height)
    return local_log - reference_log, local_score - reference_score


@lru_cache(maxsize=128)
def mixed_normalisation(energy: float, table: BoundComptonTable) -> tail.ScaledBoundNormalisation:
    reference_index = mixed_reference(energy, table)
    reference_shell = table.shells[reference_index]
    ref_binding, ref_height = reference_shell.binding_kev, reference_shell.momentum_height
    ref_log_mass, ref_score = _mass_pair(energy, -1.0, ref_binding, ref_height)
    values: list[float] = []
    derivatives: list[float] = []
    for shell in table.shells:
        binding, height = shell.binding_kev, shell.momentum_height
        if energy <= binding:
            continue
        log_occupancy = _log_occupancy_ratio(shell.occupancy, reference_shell.occupancy)
        back_log, back_centre = _back_relative(energy, binding, height, ref_binding, ref_height)
        if math.exp(log_occupancy + back_log) == 0.0:
            continue
        upper = shell_momentum_max(energy, -1.0, binding)
        if upper < 0.0:
            state = tail._state(energy, 2.0, binding, height)
            weight = math.exp(log_occupancy + back_log) * _envelope_integral(state.rate)
            for panel in range(32):
                midpoint, radius = (panel + 0.5) / 32.0, 1.0 / 64.0
                for node, gauss_weight in zip(_NODES, _WEIGHTS, strict=True):
                    for sign in (-1.0, 1.0):
                        x = midpoint + sign * radius * node
                        left, right = x**4, (1.0 - x) ** 4
                        total = left + right
                        fraction, complement = left / total, right / total
                        distance = tail._proposal_distance(fraction, complement, state.rate)
                        angle_log, angle_centre = tail._angle_relative(energy, state, distance)
                        proposal_ratio = math.exp(angle_log + state.rate * distance)
                        kn, kn_e = tail._kn(energy, 2.0 - distance)
                        jacobian = 4.0 * x**3 * (1.0 - x) ** 3 / total**2
                        coefficient = weight * radius * gauss_weight * jacobian * proposal_ratio
                        values.append(coefficient * kn)
                        derivatives.append(coefficient * (kn_e + kn * (back_centre + angle_centre)))
        else:
            for lo, hi in ((-1.0, 0.0), (0.0, upper)):
                interval_mass = profile_interval_mass(lo, hi, height)
                if interval_mass <= 0.0:
                    continue
                relative_mass = math.exp(log_occupancy + math.log(interval_mass) - ref_log_mass)
                if relative_mass == 0.0:
                    continue
                for panel in range(32):
                    midpoint, radius = (panel + 0.5) / 32.0, 1.0 / 64.0
                    for node, gauss_weight in zip(_NODES, _WEIGHTS, strict=True):
                        for sign in (-1.0, 1.0):
                            x = midpoint + sign * radius * node
                            left, right = x**4, (1.0 - x) ** 4
                            total = left + right
                            momentum = _profile_interval_inverse_pair(
                                left / total, right / total, lo, hi, height
                            )
                            lower = _minimum_transfer(energy, binding, momentum)
                            value, derivative = _kn_interval(energy, lower)
                            moving_limit = lower * (1.0 / energy + 1.0 / (energy - binding))
                            derivative += _kn(energy, lower)[0] * moving_limit
                            jacobian = 4.0 * x**3 * (1.0 - x) ** 3 / total**2
                            coefficient = relative_mass * radius * gauss_weight * jacobian
                            values.append(coefficient * value)
                            derivatives.append(coefficient * (derivative - ref_score * value))
    value, derivative = 2.0 * math.pi * math.fsum(values), 2.0 * math.pi * math.fsum(derivatives)
    if not math.isfinite(value) or value <= 0.0 or not math.isfinite(derivative):
        raise TransportError("bound mixture exceeds binary64 normalisation range")
    return tail.ScaledBoundNormalisation(reference_index, value, derivative, ref_score)


def mixed_raw_normalisation_pair(energy: float, table: BoundComptonTable) -> tuple[float, float]:
    normalisation = mixed_normalisation(energy, table)
    reference = table.shells[normalisation.reference_shell]
    log_mass, _ = _mass_pair(energy, -1.0, reference.binding_kev, reference.momentum_height)
    log_scale = math.log(reference.occupancy) + log_mass
    # Restore the raw physical factor last, preserving subnormal results where possible.
    value = math.exp(math.log(normalisation.scaled_value) + log_scale)
    score = normalisation.energy_score
    derivative = (
        0.0
        if score == 0.0
        else math.copysign(
            math.exp(math.log(abs(score)) + math.log(normalisation.scaled_value) + log_scale), score
        )
    )
    return value, derivative


def mixed_log_raw_normalisation(energy: float, table: BoundComptonTable) -> float:
    normalisation = mixed_normalisation(energy, table)
    reference = table.shells[normalisation.reference_shell]
    log_mass = _mass_pair(energy, -1.0, reference.binding_kev, reference.momentum_height)[0]
    return math.log(normalisation.scaled_value) + math.log(reference.occupancy) + log_mass


def mixed_phase_pdf(energy: float, cosine: float, table: BoundComptonTable) -> float:
    if not -1.0 <= cosine < 1.0:
        return 0.0
    normalisation = mixed_normalisation(energy, table)
    reference = table.shells[normalisation.reference_shell]
    weights: list[float] = []
    for shell in table.shells:
        if energy > shell.binding_kev:
            log_mass = _angle_relative(
                energy,
                cosine,
                shell.binding_kev,
                shell.momentum_height,
                reference.binding_kev,
                reference.momentum_height,
            )[0]
            weights.append(
                math.exp(_log_occupancy_ratio(shell.occupancy, reference.occupancy) + log_mass)
            )
    return tail._kn(energy, 1.0 - cosine)[0] * math.fsum(weights) / normalisation.scaled_value


def mixed_angle_proposal(
    energy: float, table: BoundComptonTable, uniform: float
) -> tuple[float, float]:
    reference = table.shells[mixed_reference(energy, table)]
    rates: list[float] = []
    back_weights: list[float] = []
    weights: list[float] = []
    for shell in table.shells:
        if energy <= shell.binding_kev:
            rates.append(0.0)
            back_weights.append(0.0)
            weights.append(0.0)
            continue
        back_log = _back_relative(
            energy,
            shell.binding_kev,
            shell.momentum_height,
            reference.binding_kev,
            reference.momentum_height,
        )[0]
        back_weight = math.exp(
            _log_occupancy_ratio(shell.occupancy, reference.occupancy) + back_log
        )
        rate = (
            _back_rate(energy, shell.binding_kev, shell.momentum_height)
            if back_weight > 0.0
            else 0.0
        )
        rates.append(rate)
        back_weights.append(back_weight)
        weights.append(back_weight * _envelope_integral(rate))
    total = math.fsum(weights)
    target, cumulative = uniform * total, 0.0
    cosine, transfer = math.nan, math.nan
    for index, weight in enumerate(weights):
        previous = cumulative
        cumulative += weight
        if weight > 0.0 and target < cumulative:
            fraction = (target - previous) / weight
            complement = ((1.0 - uniform) * total - math.fsum(weights[index + 1 :])) / weight
            if rates[index] == 0.0:
                cosine = 2.0 * fraction - 1.0
                transfer = 2.0 * complement
            else:
                cosine = tail._proposal_cosine(fraction, complement, rates[index])
                transfer = tail._proposal_transfer(fraction, complement, rates[index], cosine)
            break
    numerator: list[float] = []
    denominator: list[float] = []
    for shell, rate, back_weight in zip(table.shells, rates, back_weights, strict=True):
        if back_weight <= 0.0:
            continue
        if rate > 0.0:
            state = tail._state(energy, 2.0, shell.binding_kev, shell.momentum_height)
            relative = tail._angle_relative_at_transfer(energy, state, cosine + 1.0, transfer)[0]
        else:
            local = _mass_pair(energy, cosine, shell.binding_kev, shell.momentum_height)[0]
            back = _mass_pair(energy, -1.0, shell.binding_kev, shell.momentum_height)[0]
            relative = local - back
        numerator.append(back_weight * math.exp(relative))
        denominator.append(back_weight * math.exp(-rate * (cosine + 1.0)))
    acceptance = 0.5 * tail._kn(energy, transfer)[0] * math.fsum(numerator) / math.fsum(denominator)
    return cosine, acceptance


def mixed_shell_weights(
    energy: float,
    cosine: float,
    table: BoundComptonTable,
) -> tuple[float, ...]:
    local_logs = [
        math.log(s.occupancy) + _mass_pair(energy, cosine, s.binding_kev, s.momentum_height)[0]
        if energy > s.binding_kev
        else -math.inf
        for s in table.shells
    ]
    reference_index = max(range(len(local_logs)), key=local_logs.__getitem__)
    reference = table.shells[reference_index]
    weights = [
        math.exp(
            _log_occupancy_ratio(shell.occupancy, reference.occupancy)
            + _local_mass_relative(
                energy,
                cosine,
                shell.binding_kev,
                shell.momentum_height,
                reference.binding_kev,
                reference.momentum_height,
            )[0]
        )
        if energy > shell.binding_kev
        else 0.0
        for shell in table.shells
    ]
    return tuple(weights)


def mixed_conditional_energy(
    energy: float,
    cosine: float,
    table: BoundComptonTable,
    shell_uniform: float,
    energy_uniform: float,
) -> tuple[float, int, float]:
    weights = mixed_shell_weights(energy, cosine, table)
    target, cumulative = shell_uniform * math.fsum(weights), 0.0
    for index, (shell, weight) in enumerate(zip(table.shells, weights, strict=True)):
        cumulative += weight
        if weight <= 0.0 or target >= cumulative:
            continue
        upper = shell_momentum_max(energy, cosine, shell.binding_kev)
        if upper >= 0.0:
            momentum = profile_interval_inverse(energy_uniform, -1.0, upper, shell.momentum_height)
            outgoing = impulse_energy(energy, cosine, momentum, shell.binding_kev)
            numerator, q = tail.joint_kinematic_pair(energy, cosine, outgoing)
            return outgoing, index, numerator / q
        state = tail._state_at_cosine(energy, cosine, shell.binding_kev, shell.momentum_height)
        size = -state.cutoff
        if size <= 64.0 or math.log(energy_uniform) < -0.5 * size:
            span = tail._momentum_span(state, energy_uniform)
            if (energy - state.binding) / energy < 1e-12 and size < 0.001:
                outgoing = tail._threshold_energy(energy, cosine, state, energy_uniform)
            else:
                outgoing = tail._energy_from_span(energy, cosine, state.binding, span)
            return outgoing, index, -1.0 + span
        deficit = tail._momentum_deficit(energy, cosine, state, energy_uniform)
        outgoing = (
            energy - state.binding - tail._energy_deficit(energy, cosine, state.binding, deficit)
        )
        return outgoing, index, state.momentum - deficit
    raise TransportError("bound mixture conditional shell selection failed")


def _joint_profile(
    energy: float,
    cosine: float,
    outgoing: float,
    table: BoundComptonTable,
    normalisation: tail.ScaledBoundNormalisation,
) -> tuple[float, float, float, float]:
    shell0 = table.shells[normalisation.reference_shell]
    ref_log_mass, ref_score = _mass_pair(energy, -1.0, shell0.binding_kev, shell0.momentum_height)
    numerator, physical_q = tail.joint_kinematic_pair(energy, cosine, outgoing)
    momentum = numerator / physical_q
    momentum_e = (-1.0 + outgoing * (1.0 - cosine) / MASS) / physical_q - momentum * (
        energy - outgoing * cosine
    ) / (physical_q * physical_q)
    use_anchor = momentum < 0.0 and shell_momentum_max(energy, -1.0, shell0.binding_kev) < 0.0
    values: list[float] = []
    slopes: list[float] = []
    centres: list[float] = []
    for shell in table.shells:
        if outgoing >= energy - shell.binding_kev:
            continue
        h = shell.momentum_height
        if _profile_log(momentum, h) == -math.inf:
            continue
        slope = _profile_log_slope(momentum, h)
        if use_anchor:
            reference = tail._state(energy, 2.0, shell0.binding_kev, shell0.momentum_height)
            delta, delta_e = tail._local_delta(energy, cosine, outgoing, reference)
            reference_p_e = (
                2.0
                * reference.product_e
                * (1.0 / (MASS * reference.q) - reference.momentum / reference.q**2)
            )
            h0, difference = reference.height, h - reference.height
            a_difference = -difference * reference.momentum - h * delta
            a_sum = -h * (reference.momentum + delta) - h0 * reference.momentum
            exponent = -2.0 * a_difference * (1.0 + a_sum)
            log_factor = (
                math.log(2.0)
                + math.log(h)
                + math.log1p(-2.0 * h * (reference.momentum + delta))
                - reference.log_fraction
            )
            log_value = (
                _log_occupancy_ratio(shell.occupancy, shell0.occupancy) + exponent + log_factor
            )
            value = math.exp(log_value)
            if value == 0.0:
                continue
            first_product = difference * reference_p_e + h * delta_e
            second_product = difference * (h + h0) * reference.momentum * reference_p_e + h * h * (
                delta * reference_p_e + reference.momentum * delta_e + delta * delta_e
            )
            centre = (
                2.0 * first_product
                - 4.0 * second_product
                - 2.0
                * h
                / (1.0 - 2.0 * h * (reference.momentum + delta))
                * (reference_p_e + delta_e)
                - tail._fraction_score(reference, 2.0)
            )
        else:
            value = math.exp(
                _log_occupancy_ratio(shell.occupancy, shell0.occupancy)
                + _profile_log_density(momentum, h)
                - ref_log_mass
            )
            if value == 0.0:
                continue
            centre = slope * momentum_e - ref_score
        values.append(value)
        slopes.append(value * slope)
        centres.append(value * centre)
    density = math.fsum(values)
    if density <= 0.0:
        return 0.0, 0.0, 0.0, momentum
    return density, math.fsum(slopes) / density, math.fsum(centres) / density, momentum


def mixed_joint_pdf(
    energy: float, cosine: float, outgoing: float, table: BoundComptonTable
) -> float:
    if not 0.0 < outgoing < energy or not -1.0 <= cosine < 1.0:
        return 0.0
    normalisation = mixed_normalisation(energy, table)
    density = _joint_profile(energy, cosine, outgoing, table, normalisation)[0]
    ratio = outgoing / energy
    factor = 1.0 + energy / MASS * (1.0 - cosine)
    q2 = (1.0 - ratio) ** 2 + 2.0 * ratio * (1.0 - cosine)
    jacobian = (factor - cosine + ratio * (1.0 - factor * cosine)) / (energy * q2 * math.sqrt(q2))
    return tail._kn(energy, 1.0 - cosine)[0] * density * jacobian / normalisation.scaled_value


def mixed_joint_scores(
    energy: float, cosine: float, outgoing: float, table: BoundComptonTable
) -> tuple[float, float, float]:
    normalisation = mixed_normalisation(energy, table)
    density, profile_score, centre, momentum = _joint_profile(
        energy, cosine, outgoing, table, normalisation
    )
    if density <= 0.0:
        raise TransportError("bound mixture score has zero physical density")
    ratio, k = outgoing / energy, energy / MASS
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
