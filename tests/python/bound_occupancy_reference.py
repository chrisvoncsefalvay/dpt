"""Independent Decimal references for the original occupancy-weighted impulse law.

The oracle integrates the physical angle, not a production proposal or momentum
quadrature. All input scalars are the exact values of their binary64 encodings.
Very steep components retain a logarithmic physical upper bound when their mass
is below the Decimal context's exponent range.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import lru_cache

PRECISION = 110
MASS = 510.99895069
PI = Decimal(
    "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865133"
)
Shells = tuple[tuple[float, float, float], ...]


def _d(value: float) -> Decimal:
    return Decimal.from_float(value)


def _profile_log(momentum: Decimal, height: Decimal) -> Decimal:
    amplitude = height * abs(momentum)
    return -2 * amplitude * (1 + amplitude)


def _density(momentum: Decimal, height: Decimal) -> Decimal:
    return height * (1 + 2 * height * abs(momentum)) * _profile_log(momentum, height).exp()


def _cdf(momentum: Decimal, height: Decimal) -> Decimal:
    tail = _profile_log(momentum, height).exp() / 2
    return tail if momentum < 0 else 1 - tail


def _endpoint(energy: Decimal, cosine: Decimal, binding: Decimal) -> tuple[Decimal, Decimal]:
    transfer = 1 - cosine
    if binding == 0:
        root = (transfer / 2).sqrt()
        return energy / _d(MASS) * root, root / _d(MASS)
    product = energy * (energy - binding)
    q = (2 * product * transfer + binding * binding).sqrt()
    momentum = (product * transfer / _d(MASS) - binding) / q
    derivative = (2 * energy - binding) * transfer * (1 / (_d(MASS) * q) - momentum / (q * q))
    return momentum, derivative


def _mass(energy: Decimal, cosine: Decimal, binding: Decimal, height: Decimal) -> Decimal:
    if energy <= binding:
        return Decimal(0)
    upper, _ = _endpoint(energy, cosine, binding)
    return _cdf(upper, height) - _cdf(Decimal(-1), height)


def _log_mass(energy: Decimal, cosine: Decimal, binding: Decimal, height: Decimal) -> Decimal:
    upper, _ = _endpoint(energy, cosine, binding)
    if upper < 0:
        high, low = _profile_log(upper, height), _profile_log(Decimal(-1), height)
        return high - Decimal(2).ln() + (1 - (low - high).exp()).ln()
    return _mass(energy, cosine, binding, height).ln()


def _kn_pair(energy: Decimal, cosine: Decimal) -> tuple[Decimal, Decimal]:
    transfer = 1 - cosine
    ratio = 1 / (1 + energy * transfer / _d(MASS))
    sine2 = 1 - cosine * cosine
    value = ratio**3 + ratio - sine2 * ratio**2
    derivative = -transfer / _d(MASS) * ratio**2 * (3 * ratio**2 + 1 - 2 * sine2 * ratio)
    return value, derivative


@dataclass(frozen=True)
class OccupancyNormalisationReference:
    reference_shell: int
    raw_value: Decimal
    raw_derivative: Decimal
    scale: Decimal
    scaled_value: Decimal
    centred_scaled_derivative: Decimal
    centre_score: Decimal
    refinement_relative_error: Decimal
    omitted_component_log_bounds: tuple[tuple[int, Decimal], ...]

    @property
    def energy_score(self) -> Decimal:
        return self.raw_derivative / self.raw_value


def _simpson_component(
    energy: Decimal, binding: Decimal, height: Decimal, panels: int
) -> tuple[Decimal, Decimal]:
    value, derivative = Decimal(0), Decimal(0)
    for index in range(1, panels + 1):
        # Original physical angle: s=sin(theta/2), c=1-2s^2, |dc|=4s ds.
        s = Decimal(index) / panels
        cosine = 1 - 2 * s * s
        upper, upper_e = _endpoint(energy, cosine, binding)
        mass = _cdf(upper, height) - _cdf(Decimal(-1), height)
        kn, kn_e = _kn_pair(energy, cosine)
        coefficient = Decimal(1 if index == panels else 4 if index % 2 else 2) * 4 * s
        value += coefficient * kn * mass
        derivative += coefficient * (kn_e * mass + kn * _density(upper, height) * upper_e)
    return 2 * PI * value / (3 * panels), 2 * PI * derivative / (3 * panels)


@lru_cache(maxsize=32)
def _component_integral(
    energy: float, binding: float, height: float
) -> tuple[Decimal, Decimal, Decimal]:
    with localcontext() as context:
        context.prec = PRECISION
        e, u, h = _d(energy), _d(binding), _d(height)
        if _endpoint(e, Decimal(-1), u)[0] < 0 and _back_rate(e, u, h) > 32:
            return _negative_component_integral(e, u, h)
        coarse = _simpson_component(e, u, h, 8192)
        fine = _simpson_component(e, u, h, 16384)
        value, derivative = (16 * fine[0] - coarse[0]) / 15, (16 * fine[1] - coarse[1]) / 15
        error = max(abs((fine[0] - coarse[0]) / value), abs((fine[1] - coarse[1]) / derivative))
        return value, derivative, error


def _negative_component_integral(
    energy: Decimal, binding: Decimal, height: Decimal
) -> tuple[Decimal, Decimal, Decimal]:
    """Original cosine integral in linear z=rate*(1+c), not a proposal CDF.

    Log-concavity bounds the omitted z>=64 angular mass by the tangent
    exponential. Its relative contribution is below2e-26 for these fixtures.
    """
    rate = _back_rate(energy, binding, height)
    back_mass = _mass(energy, Decimal(-1), binding, height)
    upper, upper_e = _endpoint(energy, Decimal(-1), binding)
    centre = _density(upper, height) * upper_e / back_mass
    stop = min(Decimal(64), 2 * rate)
    refinements = []
    for panels in (8192, 16384):
        value, centred = Decimal(0), Decimal(0)
        for index in range(panels + 1):
            cosine = -1 + stop * index / (panels * rate)
            endpoint, endpoint_e = _endpoint(energy, cosine, binding)
            mass = _mass(energy, cosine, binding, height)
            kn, kn_e = _kn_pair(energy, cosine)
            coefficient = Decimal(1 if index in (0, panels) else 4 if index % 2 else 2)
            value += coefficient * kn * mass / back_mass
            centred += (
                coefficient
                * (kn_e * mass + kn * (_density(endpoint, height) * endpoint_e - centre * mass))
                / back_mass
            )
        factor = 2 * PI * stop / (3 * panels * rate)
        refinements.append((factor * value, factor * centred))
    coarse, fine = refinements
    value = (16 * fine[0] - coarse[0]) / 15
    centred = (16 * fine[1] - coarse[1]) / 15
    error = max(abs((fine[0] - coarse[0]) / value), abs((fine[1] - coarse[1]) / centred))
    return back_mass * value, back_mass * (centred + centre * value), error


def _back_rate(energy: Decimal, binding: Decimal, height: Decimal) -> Decimal:
    upper, _ = _endpoint(energy, Decimal(-1), binding)
    if upper >= 0:
        return Decimal(0)
    q = (4 * energy * (energy - binding) + binding * binding).sqrt()
    p_transfer = energy * (energy - binding) * (1 / (_d(MASS) * q) - upper / (q * q))
    log_density = height.ln() + (1 + 2 * height * abs(upper)).ln() + _profile_log(upper, height)
    return (log_density - _log_mass(energy, Decimal(-1), binding, height)).exp() * p_transfer


def _back_components(
    energy: Decimal, shells: Shells
) -> tuple[tuple[int, Decimal, Decimal, Decimal], ...]:
    components = []
    for index, (occupancy, binding, height) in enumerate(shells):
        f, u, h = _d(occupancy), _d(binding), _d(height)
        if energy <= u:
            continue
        log_mass = _log_mass(energy, Decimal(-1), u, h)
        rate = _back_rate(energy, u, h)
        envelope = Decimal(2) if rate == 0 else (1 - (-2 * rate).exp()) / rate
        components.append((index, f.ln() + log_mass, rate, envelope))
    return tuple(components)


@lru_cache(maxsize=64)
def original_occupancy_normalisation(
    energy: float, shells: Shells
) -> OccupancyNormalisationReference:
    with localcontext() as context:
        context.prec = PRECISION
        e = _d(energy)
        components = _back_components(e, shells)
        reference_index, reference_log, _, _ = max(
            components, key=lambda item: item[1] + item[3].ln()
        )
        scale = reference_log.exp()
        _, ref_binding, ref_height = shells[reference_index]
        ref_upper, ref_upper_e = _endpoint(e, Decimal(-1), _d(ref_binding))
        centre = (
            _density(ref_upper, _d(ref_height))
            * ref_upper_e
            / _mass(e, Decimal(-1), _d(ref_binding), _d(ref_height))
        )
        value, derivative, error = Decimal(0), Decimal(0), Decimal(0)
        omitted = []
        for index, component_log, _, _ in components:
            occupancy, binding, height = shells[index]
            # KN<=2 and dOmega integrates to4pi. The original interval mass
            # increases toward backscatter. This is a physical bound, not a
            # proposal-conditioning approximation.
            if component_log < reference_log - 400:
                omitted.append((index, component_log + (8 * PI).ln()))
                continue
            component_value, component_derivative, component_error = _component_integral(
                energy, binding, height
            )
            value += _d(occupancy) * component_value
            derivative += _d(occupancy) * component_derivative
            error = max(error, component_error)
        bounds = tuple((index, log_upper - value.ln()) for index, log_upper in omitted)
        if any(bound >= Decimal(-300) for _, bound in bounds):
            raise ArithmeticError("omitted physical component is not below110-digit resolution")
        return OccupancyNormalisationReference(
            reference_index,
            value,
            derivative,
            scale,
            value / scale,
            (derivative - centre * value) / scale,
            centre,
            error,
            bounds,
        )


def _joint_raw(
    energy: Decimal, cosine: Decimal, outgoing: Decimal, shells: Shells
) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
    ratio = outgoing / energy
    factor = 1 + energy / _d(MASS) * (1 - cosine)
    q2 = (1 - ratio) ** 2 + 2 * ratio * (1 - cosine)
    q = q2.sqrt()
    momentum = (factor * ratio - 1) / q
    n = factor - cosine + ratio * (1 - factor * cosine)
    density, slope_numerator = Decimal(0), Decimal(0)
    for occupancy, binding, height in shells:
        if outgoing >= energy - _d(binding):
            continue
        h = _d(height)
        component = _d(occupancy) * _density(momentum, h)
        slope = 2 * h / (1 + 2 * h * abs(momentum)) - 2 * h - 4 * h * h * abs(momentum)
        if momentum < 0:
            slope = -slope
        density += component
        slope_numerator += component * slope
    kn = _kn_pair(energy, cosine)[0]
    return kn * density * n / (energy * q**3), slope_numerator / density, momentum, q2, n


def original_occupancy_phase_pdf(energy: float, cosine: float, shells: Shells) -> float:
    reference = original_occupancy_normalisation(energy, shells)
    with localcontext() as context:
        context.prec = PRECISION
        e, c = _d(energy), _d(cosine)
        mass = sum(
            (_d(f) * _mass(e, c, _d(u), _d(h)) for f, u, h in shells if e > _d(u)), Decimal(0)
        )
        return float(_kn_pair(e, c)[0] * mass / reference.raw_value)


def original_occupancy_joint(
    energy: float, cosine: float, outgoing: float, shells: Shells
) -> tuple[float, tuple[float, float, float]]:
    reference = original_occupancy_normalisation(energy, shells)
    with localcontext() as context:
        context.prec = PRECISION
        e, c, out, m = _d(energy), _d(cosine), _d(outgoing), _d(MASS)
        raw, profile_score, p, q2, n = _joint_raw(e, c, out, shells)
        q, ratio, k = q2.sqrt(), out / e, e / m
        factor = 1 + k * (1 - c)
        tau, a_e = 1 / factor, (1 - c) / m
        kn_term = tau + factor - 1 + c * c
        score_c = (
            2 * k * tau
            + (k * tau * tau - k + 2 * c) / kn_term
            + profile_score * (-k * ratio / q + p * ratio / q2)
            + (-k - 1 + ratio * (k * c - factor)) / n
            + 3 * ratio / q2
        )
        score_out = (profile_score * n / (q * q2) + (1 - factor * c) / n - 3 * (ratio - c) / q2) / e
        p_e = -ratio / (e * q) + p * ratio * (ratio - c) / (e * q2)
        n_e = a_e * (1 - ratio * c) - ratio / e * (1 - factor * c)
        score_e = (
            -2 * tau * a_e
            + (1 - tau * tau) * a_e / kn_term
            + profile_score * p_e
            + n_e / n
            - 1 / e
            + 3 * ratio * (ratio - c) / (e * q2)
            - reference.energy_score
        )
        return float(raw / reference.raw_value), (float(score_c), float(score_out), float(score_e))


def original_occupancy_finite_scores(
    energy: float, cosine: float, outgoing: float, shells: Shells
) -> tuple[float, float, float]:
    reference = original_occupancy_normalisation(energy, shells)
    with localcontext() as context:
        context.prec = PRECISION
        point = [_d(energy), _d(cosine), _d(outgoing)]
        step = Decimal("1e-25")
        derivatives = []
        for axis in (1, 2, 0):
            values = []
            for displacement in (-2, -1, 1, 2):
                state = point.copy()
                state[axis] += displacement * step
                # The copied coordinate list has exactly three Decimal entries.
                values.append(_joint_raw(*state, shells)[0].ln())  # pyright: ignore[reportCallIssue]
            derivatives.append(
                (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step)
            )
        derivatives[2] -= reference.energy_score
        return float(derivatives[0]), float(derivatives[1]), float(derivatives[2])


def original_occupancy_angle_proposal(
    energy: float, uniform: float, shells: Shells
) -> tuple[float, float]:
    with localcontext() as context:
        context.prec = PRECISION
        e, mark = _d(energy), _d(uniform)
        components = _back_components(e, shells)
        log_reference = max(item[1] + item[3].ln() for item in components)
        weights = [
            (index, (log_weight - log_reference).exp() * envelope, rate, log_weight)
            for index, log_weight, rate, envelope in components
        ]
        total = sum((item[1] for item in weights), Decimal(0))
        target, cumulative = mark * total, Decimal(0)
        cosine = Decimal("NaN")
        for _, weight, rate, _ in weights:
            previous, cumulative = cumulative, cumulative + weight
            if weight and target < cumulative:
                fraction = (target - previous) / weight
                distance = (
                    2 * fraction
                    if rate == 0
                    else -(1 - fraction * (1 - (-2 * rate).exp())).ln() / rate
                )
                cosine = -1 + distance
                break
        numerator, denominator = Decimal(0), Decimal(0)
        for index, _, rate, log_weight in weights:
            occupancy, binding, height = shells[index]
            numerator += (
                _d(occupancy).ln() + _log_mass(e, cosine, _d(binding), _d(height)) - log_reference
            ).exp()
            denominator += (log_weight - log_reference - rate * (cosine + 1)).exp()
        acceptance = _kn_pair(e, cosine)[0] * numerator / (2 * denominator)
        return float(cosine), float(acceptance)


def original_occupancy_shell(
    energy: float, cosine: float, shell_uniform: float, shells: Shells
) -> int:
    with localcontext() as context:
        context.prec = PRECISION
        e, c = _d(energy), _d(cosine)
        weights = [
            _d(f) * _mass(e, c, _d(u), _d(h)) if e > _d(u) else Decimal(0) for f, u, h in shells
        ]
        target = _d(shell_uniform) * sum(weights, Decimal(0))
        cumulative = Decimal(0)
        for index, weight in enumerate(weights):
            cumulative += weight
            if weight and target < cumulative:
                return index
        raise ArithmeticError("original shell mark outside support")


def original_occupancy_energy_cdf(
    energy: float, cosine: float, outgoing: float, shell_index: int, shells: Shells
) -> float:
    with localcontext() as context:
        context.prec = PRECISION
        e, c, out = _d(energy), _d(cosine), _d(outgoing)
        _, binding, height = shells[shell_index]
        ratio = out / e
        momentum = ((1 + e / _d(MASS) * (1 - c)) * ratio - 1) / (
            (1 - ratio) ** 2 + 2 * ratio * (1 - c)
        ).sqrt()
        return float(
            (_cdf(momentum, _d(height)) - _cdf(Decimal(-1), _d(height)))
            / _mass(e, c, _d(binding), _d(height))
        )


def original_physical_momentum(energy: float, cosine: float, outgoing: float) -> float:
    with localcontext() as context:
        context.prec = PRECISION
        e, c, out = _d(energy), _d(cosine), _d(outgoing)
        ratio = out / e
        return float(
            ((1 + e / _d(MASS) * (1 - c)) * ratio - 1)
            / ((1 - ratio) ** 2 + 2 * ratio * (1 - c)).sqrt()
        )


def original_occupancy_energy_inverse(
    energy: float, cosine: float, shell_uniform: float, energy_uniform: float, shells: Shells
) -> tuple[float, int, float]:
    index = original_occupancy_shell(energy, cosine, shell_uniform, shells)
    with localcontext() as context:
        context.prec = PRECISION
        e, c = _d(energy), _d(cosine)
        _, binding, height = shells[index]
        u, h = _d(binding), _d(height)
        upper = _endpoint(e, c, u)[0]
        target = _cdf(Decimal(-1), h) + _d(energy_uniform) * _mass(e, c, u, h)
        lo, hi = Decimal(-1), upper
        for _ in range(300):
            mid = (lo + hi) / 2
            if _cdf(mid, h) < target:
                lo = mid
            else:
                hi = mid
        momentum = (lo + hi) / 2
        lo, hi = Decimal(0), e - u
        for _ in range(300):
            out = (lo + hi) / 2
            ratio = out / e
            current = ((1 + e / _d(MASS) * (1 - c)) * ratio - 1) / (
                (1 - ratio) ** 2 + 2 * ratio * (1 - c)
            ).sqrt()
            if current < momentum:
                lo = out
            else:
                hi = out
        return float((lo + hi) / 2), index, float(momentum)


def occupancy_states(energy: float, shells: Shells) -> tuple[tuple[float, float], ...]:
    if any(energy > u and _endpoint(_d(energy), Decimal(-1), _d(u))[0] >= 0 for _, u, _ in shells):
        return ((-0.99, 0.005), (-0.9, 0.15), (-0.2, 0.45), (0.6, 0.85))
    states = []
    for angle_mark, energy_mark in ((0.125, 0.125), (0.5, 0.5), (0.875, 0.875)):
        cosine = original_occupancy_angle_proposal(energy, angle_mark, shells)[0]
        outgoing = original_occupancy_energy_inverse(energy, cosine, 0.5, energy_mark, shells)[0]
        states.append((cosine, outgoing))
    return tuple(states)


def occupancy_fixtures() -> tuple[tuple[str, Shells], ...]:
    minimum = float.fromhex("0x0.0000000000001p-1022")
    inactive = (1.0, 2.0, 0.01)
    broad = (minimum, 0.0, 0.01)
    other = (2 * minimum, 0.0, 0.03)
    steep = (1.0, 1.0, 1e6)
    narrow = (minimum, 1.0, 1e3)
    return (
        ("inactive-first", (inactive, broad)),
        ("inactive-last", (broad, inactive)),
        ("two-broad", (inactive, broad, other)),
        ("two-broad-reversed", (inactive, other, broad)),
        ("steep-first", (steep, broad)),
        ("steep-last", (broad, steep)),
        ("negative-tiny-first", (narrow, steep)),
        ("negative-tiny-last", (steep, narrow)),
    )
