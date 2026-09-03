"""Decimal110 references evaluated from the original bound-Compton law.

The references deliberately use the original momentum, profile CDF and
energy Jacobian. They do not import DPT or use its anchored differences,
profile-quantile quadrature or binary64 proposal arithmetic. Decimal inputs
retain the exact values of the binary64 states supplied to production code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import lru_cache

MASS = 510.99895069
PRECISION = 110
TWO_PI = Decimal.from_float(2 * math.pi)


@dataclass(frozen=True)
class TailNormalisationReference:
    """Normalisation relative to one shell's original backscatter mass."""

    scaled_value: Decimal
    centred_scaled_derivative: Decimal
    centre_score: Decimal
    proposal_rate: Decimal
    backscatter_log_profile: Decimal
    cutoff_ratio: Decimal
    refinements: tuple[tuple[int, Decimal, Decimal], ...]

    @property
    def centred_score(self) -> Decimal:
        with localcontext() as context:
            context.prec = PRECISION
            return self.centred_scaled_derivative / self.scaled_value


def _parameters(energy: float, binding: float, height: float) -> tuple[Decimal, ...]:
    return tuple(map(Decimal.from_float, (energy, binding, height, MASS)))


def _profile_log(momentum: Decimal, height: Decimal) -> Decimal:
    return -2 * height * abs(momentum) * (1 + height * abs(momentum))


def _endpoint(
    energy: Decimal, binding: Decimal, mass: Decimal, cosine: Decimal
) -> tuple[Decimal, Decimal, Decimal]:
    """Original shell endpoint and derivatives at fixed physical angle."""
    product = energy * (energy - binding)
    transfer = 1 - cosine
    q = (2 * product * transfer + binding * binding).sqrt()
    momentum = (product * transfer / mass - binding) / q
    common = 1 / (mass * q) - momentum / (q * q)
    p_e = (2 * energy - binding) * transfer * common
    p_t = product * common
    return momentum, p_e, p_t


def _kn_pair(energy: Decimal, mass: Decimal, cosine: Decimal) -> tuple[Decimal, Decimal]:
    transfer = 1 - cosine
    ratio = 1 / (1 + energy / mass * transfer)
    sine_squared = 1 - cosine * cosine
    value = ratio**3 + ratio - sine_squared * ratio**2
    derivative = -transfer / mass * ratio**2 * (3 * ratio**2 + 1 - 2 * sine_squared * ratio)
    return value, derivative


def _baseline(
    energy: Decimal, binding: Decimal, height: Decimal, mass: Decimal
) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
    momentum, p_e, p_t = _endpoint(energy, binding, mass, Decimal(-1))
    profile_log = _profile_log(momentum, height)
    cutoff_ratio = (_profile_log(Decimal(-1), height) - profile_log).exp()
    slope = 2 * height * (1 - 2 * height * momentum)
    return (
        profile_log,
        cutoff_ratio,
        slope * p_e / (1 - cutoff_ratio),
        slope * p_t / (1 - cutoff_ratio),
        momentum,
    )


def _angular_pair(
    energy: Decimal,
    binding: Decimal,
    height: Decimal,
    mass: Decimal,
    cosine: Decimal,
    back_log: Decimal,
    cutoff_ratio: Decimal,
    centre_score: Decimal,
) -> tuple[Decimal, Decimal]:
    momentum, p_e, _ = _endpoint(energy, binding, mass, cosine)
    exponent = _profile_log(momentum, height)
    cutoff = (_profile_log(Decimal(-1), height) - exponent).exp()
    if cutoff == 1:
        return Decimal(0), Decimal(0)
    relative_mass = (exponent - back_log).exp() * (1 - cutoff) / (1 - cutoff_ratio)
    mass_score = 2 * height * (1 - 2 * height * momentum) * p_e / (1 - cutoff)
    kn, kn_e = _kn_pair(energy, mass, cosine)
    return kn * relative_mass, (kn_e + kn * (mass_score - centre_score)) * relative_mass


def _simpson_pair(
    energy: Decimal,
    binding: Decimal,
    height: Decimal,
    mass: Decimal,
    intervals: int,
    upper_z: Decimal,
) -> tuple[Decimal, Decimal]:
    back_log, cutoff_ratio, centre_score, rate, _ = _baseline(energy, binding, height, mass)
    value, derivative = Decimal(0), Decimal(0)
    for index in range(intervals + 1):
        z = upper_z * index / intervals
        cosine = -1 + z / rate
        term, term_e = _angular_pair(
            energy, binding, height, mass, cosine, back_log, cutoff_ratio, centre_score
        )
        coefficient = 1 if index in (0, intervals) else (4 if index % 2 else 2)
        value += coefficient * term
        derivative += coefficient * term_e
    coefficient = TWO_PI * upper_z / (3 * intervals * rate)
    return coefficient * value, coefficient * derivative


@lru_cache(maxsize=16)
def original_tail_normalisation(
    energy: float = 1.01, binding: float = 1.0, height: float = 1e6
) -> TailNormalisationReference:
    """Original-angle Simpson8192/16384 and fourth-order extrapolation.

    The full original angle interval is used for broad negative-tail fixtures.
    For narrow fixtures the omitted z>=64 tail is bounded by the log-concave
    tangent and contributes less than 2e-26 relatively.
    Refined integrals are retained to expose convergence instead of treating
    a copied expected constant as a reference implementation.
    """
    with localcontext() as context:
        context.prec = PRECISION
        e, u, h, m = _parameters(energy, binding, height)
        back_log, cutoff, centre, rate, momentum = _baseline(e, u, h, m)
        if not -1 < momentum < 0 or rate <= 0:
            raise ValueError("reference requires a resolved negative-tail fixture")
        upper_z = min(Decimal(64), 2 * rate)
        refinements = tuple((n, *_simpson_pair(e, u, h, m, n, upper_z)) for n in (8192, 16384))
        coarse, fine = refinements
        return TailNormalisationReference(
            (16 * fine[1] - coarse[1]) / 15,
            (16 * fine[2] - coarse[2]) / 15,
            centre,
            rate,
            back_log,
            cutoff,
            refinements,
        )


def original_tail_phase_pdf(energy: float, cosine: float, binding: float, height: float) -> float:
    reference = original_tail_normalisation(energy, binding, height)
    with localcontext() as context:
        context.prec = PRECISION
        e, u, h, m = _parameters(energy, binding, height)
        value, _ = _angular_pair(
            e,
            u,
            h,
            m,
            Decimal.from_float(cosine),
            reference.backscatter_log_profile,
            reference.cutoff_ratio,
            reference.centre_score,
        )
        return float(value / reference.scaled_value)


def original_tail_joint(
    energy: float, cosine: float, outgoing: float, binding: float, height: float
) -> tuple[float, tuple[float, float, float]]:
    """Joint PDF and all three physical scores at the actual binary64 state.

    The large local incident score and normaliser centre are subtracted in
    Decimal. Rounding two complete scores before subtracting them would lose
    precision, as would subtracting independently rounded absolute log scales.
    """
    reference = original_tail_normalisation(energy, binding, height)
    with localcontext() as context:
        context.prec = PRECISION
        e, u, h, m = _parameters(energy, binding, height)
        c, out = map(Decimal.from_float, (cosine, outgoing))
        if not -1 < c < 1 or not 0 < out < e - u:
            raise ValueError("reference requires interior physical support")
        ratio = out / e
        k = e / m
        factor = 1 + k * (1 - c)
        q2 = (1 - ratio) ** 2 + 2 * ratio * (1 - c)
        q = q2.sqrt()
        momentum = (factor * ratio - 1) / q
        n = factor - c + ratio * (1 - factor * c)
        jacobian = n / (e * q**3)
        log_profile = _profile_log(momentum, h)
        relative_density = (
            2
            * h
            * (1 + 2 * h * abs(momentum))
            * (log_profile - reference.backscatter_log_profile).exp()
            / (1 - reference.cutoff_ratio)
        )
        profile_score = -(2 * h / (1 - 2 * h * momentum) - 2 * h + 4 * h**2 * momentum)
        tau = 1 / factor
        kn_term = tau + factor - 1 + c * c
        kn = tau * tau * kn_term
        pdf = kn * relative_density * jacobian / reference.scaled_value
        kn_c = 2 * k * tau + (k * tau * tau - k + 2 * c) / kn_term
        p_c = -k * ratio / q + momentum * ratio / q2
        n_c = -k - 1 + ratio * (k * c - factor)
        score_c = kn_c + profile_score * p_c + n_c / n + 3 * ratio / q2
        p_r = n / (q * q2)
        score_outgoing = (profile_score * p_r + (1 - factor * c) / n - 3 * (ratio - c) / q2) / e
        a_e = (1 - c) / m
        p_e = -ratio / (e * q) + momentum * ratio * (ratio - c) / (e * q2)
        n_e = a_e * (1 - ratio * c) - ratio / e * (1 - factor * c)
        kn_e = -2 * tau * a_e + (1 - tau * tau) * a_e / kn_term
        score_incident = (
            kn_e
            + profile_score * p_e
            + n_e / n
            - 1 / e
            + 3 * ratio * (ratio - c) / (e * q2)
            - reference.centre_score
            - reference.centred_score
        )
        return float(pdf), (float(score_c), float(score_outgoing), float(score_incident))


def original_tail_conditional_energy(
    energy: float, cosine: float, probability: float, binding: float, height: float
) -> tuple[float, float]:
    """Original Decimal profile inverse and physical energy-equation bisection."""
    with localcontext() as context:
        context.prec = PRECISION
        e, u, h, m = _parameters(energy, binding, height)
        c, fraction = map(Decimal.from_float, (cosine, probability))
        upper, _, _ = _endpoint(e, u, m, c)
        upper_log = _profile_log(upper, h)
        cutoff_ratio = (_profile_log(Decimal(-1), h) - upper_log).exp()
        target_log = upper_log + (fraction + (1 - fraction) * cutoff_ratio).ln()
        momentum = -((1 - 2 * target_log).sqrt() - 1) / (2 * h)
        factor = 1 + e / m * (1 - c)
        lo, hi = Decimal(0), (e - u) / e
        for _ in range(300):
            ratio = (lo + hi) / 2
            q = ((1 - ratio) ** 2 + 2 * ratio * (1 - c)).sqrt()
            if (factor * ratio - 1) / q < momentum:
                lo = ratio
            else:
                hi = ratio
        return float(e * (lo + hi) / 2), float(momentum)


def decimal_joint_finite_scores(
    energy: float, cosine: float, outgoing: float, binding: float, height: float
) -> tuple[float, float, float]:
    """Sub-ULP finite differences of the original unnormalised log density.

    Decimal perturbations of 1e-25 start from the exact binary64 test state.
    The independently integrated physical N_E/N is subtracted only for the
    incident derivative. Thus the reference checks all local analytic scores
    without host finite-difference rounding or the production score formulas.
    """
    reference = original_tail_normalisation(energy, binding, height)
    with localcontext() as context:
        context.prec = PRECISION
        e, _, h, m = _parameters(energy, binding, height)
        c, out = map(Decimal.from_float, (cosine, outgoing))

        def log_density(incident: Decimal, angle: Decimal, final: Decimal) -> Decimal:
            ratio = final / incident
            factor = 1 + incident / m * (1 - angle)
            q2 = (1 - ratio) ** 2 + 2 * ratio * (1 - angle)
            q = q2.sqrt()
            momentum = (factor * ratio - 1) / q
            numerator = factor - angle + ratio * (1 - factor * angle)
            jacobian = numerator / (incident * q**3)
            kn, _ = _kn_pair(incident, m, angle)
            return (
                kn.ln()
                + h.ln()
                + (1 + 2 * h * abs(momentum)).ln()
                + _profile_log(momentum, h)
                + jacobian.ln()
            )

        step = Decimal("1e-25")
        derivatives: list[Decimal] = []
        for axis in (1, 2, 0):
            values: list[Decimal] = []
            for multiple in (-2, -1, 1, 2):
                state = [e, c, out]
                state[axis] += multiple * step
                values.append(log_density(*state))
            derivatives.append(
                (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step)
            )
        derivatives[2] -= reference.centre_score + reference.centred_score
        return float(derivatives[0]), float(derivatives[1]), float(derivatives[2])


def original_tail_angle_cdf(energy: float, upper_z: float, binding: float, height: float) -> float:
    """Original angular probability below c=-1+z/lambda."""
    reference = original_tail_normalisation(energy, binding, height)
    with localcontext() as context:
        context.prec = PRECISION
        e, u, h, m = _parameters(energy, binding, height)
        value, _ = _simpson_pair(e, u, h, m, 1024, Decimal.from_float(upper_z))
        return float(value / reference.scaled_value)


def extreme_tail_states(
    energy: float = 1.01, binding: float = 1.0, height: float = 1e6
) -> tuple[tuple[float, float], ...]:
    """Representable interior states produced by the independent inverse law."""
    reference = original_tail_normalisation(energy, binding, height)
    rate = float(reference.proposal_rate)
    states: list[tuple[float, float]] = []
    for z in (0.125, 0.75, 2.0):
        cosine = -1 + z / rate
        for uniform in (0.1, 0.5, 0.9):
            outgoing, _ = original_tail_conditional_energy(energy, cosine, uniform, binding, height)
            states.append((cosine, outgoing))
    return tuple(states)


def original_tail_angle_proposal(
    energy: float, uniform: float, binding: float, height: float
) -> tuple[float, float]:
    """Exact Decimal exponential proposal and original-law accept probability.

    Acceptance is evaluated at the ideal proposal coordinate before rounding
    its cosine to binary64, as required when checking original random marks.
    PDF and score references separately use the returned binary64 cosine.
    """
    reference = original_tail_normalisation(energy, binding, height)
    with localcontext() as context:
        context.prec = PRECISION
        e, u, h, m = _parameters(energy, binding, height)
        fraction = Decimal.from_float(uniform)
        rate = reference.proposal_rate
        truncated_mass = 1 - (-2 * rate).exp()
        distance = -(1 - fraction * truncated_mass).ln() / rate
        cosine = -1 + distance
        value, _ = _angular_pair(
            e,
            u,
            h,
            m,
            cosine,
            reference.backscatter_log_profile,
            reference.cutoff_ratio,
            reference.centre_score,
        )
        acceptance = value / (2 * (-rate * distance).exp())
        return float(cosine), float(acceptance)


@lru_cache(maxsize=8)
def original_mixture_normalisation(
    energy: float,
    shells: tuple[tuple[float, float, float], ...],
    reference_shell: int,
) -> TailNormalisationReference:
    """Combine independently integrated original shell laws in a common scale.

    Each tuple is (occupancy, binding, height). Relative weights subtract
    original Decimal exponents directly, retaining near-equal enormous logs.
    """
    references = tuple(original_tail_normalisation(energy, u, h) for _, u, h in shells)
    reference = references[reference_shell]
    with localcontext() as context:
        context.prec = PRECISION
        value, derivative = Decimal(0), Decimal(0)
        for (occupancy, _, _), component in zip(shells, references, strict=True):
            relative = (
                Decimal.from_float(occupancy)
                * (component.backscatter_log_profile - reference.backscatter_log_profile).exp()
                * (1 - component.cutoff_ratio)
                / (1 - reference.cutoff_ratio)
            )
            value += relative * component.scaled_value
            derivative += relative * (
                component.centred_scaled_derivative
                + (component.centre_score - reference.centre_score) * component.scaled_value
            )
        return TailNormalisationReference(
            value,
            derivative,
            reference.centre_score,
            reference.proposal_rate,
            reference.backscatter_log_profile,
            reference.cutoff_ratio,
            (),
        )


def original_mixture_joint(
    energy: float,
    cosine: float,
    outgoing: float,
    shells: tuple[tuple[float, float, float], ...],
    reference_shell: int,
) -> tuple[float, tuple[float, float, float]]:
    """Original mixture density and Decimal finite physical log derivatives."""
    reference = original_mixture_normalisation(energy, shells, reference_shell)
    with localcontext() as context:
        context.prec = PRECISION
        e, c, out, m = map(Decimal.from_float, (energy, cosine, outgoing, MASS))
        components = tuple(tuple(map(Decimal.from_float, shell)) for shell in shells)

        def density(incident: Decimal, angle: Decimal, final: Decimal) -> Decimal:
            ratio = final / incident
            factor = 1 + incident / m * (1 - angle)
            q2 = (1 - ratio) ** 2 + 2 * ratio * (1 - angle)
            q = q2.sqrt()
            momentum = (factor * ratio - 1) / q
            jacobian = (factor - angle + ratio * (1 - factor * angle)) / (incident * q**3)
            profile = Decimal(0)
            for occupancy, binding, height in components:
                if final < incident - binding:
                    profile += (
                        occupancy
                        * 2
                        * height
                        * (1 + 2 * height * abs(momentum))
                        * (_profile_log(momentum, height) - reference.backscatter_log_profile).exp()
                        / (1 - reference.cutoff_ratio)
                    )
            kn, _ = _kn_pair(incident, m, angle)
            return kn * profile * jacobian

        value = density(e, c, out) / reference.scaled_value
        step = Decimal("1e-25")
        scores: list[Decimal] = []
        for axis in (1, 2, 0):
            values: list[Decimal] = []
            for multiple in (-2, -1, 1, 2):
                state = [e, c, out]
                state[axis] += multiple * step
                values.append(density(*state).ln())
            scores.append((values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step))
        scores[2] -= reference.centre_score + reference.centred_score
        return float(value), (float(scores[0]), float(scores[1]), float(scores[2]))


def original_mixture_conditional_energy(
    energy: float,
    cosine: float,
    shell_uniform: float,
    energy_uniform: float,
    shells: tuple[tuple[float, float, float], ...],
) -> tuple[float, int, float]:
    """Select shells from original Decimal truncated profile probabilities."""
    with localcontext() as context:
        context.prec = PRECISION
        e, c, m = map(Decimal.from_float, (energy, cosine, MASS))
        logs: list[Decimal] = []
        for occupancy, binding, height in shells:
            f, u, h = map(Decimal.from_float, (occupancy, binding, height))
            momentum, _, _ = _endpoint(e, u, m, c)
            exponent = _profile_log(momentum, h)
            cutoff = (_profile_log(Decimal(-1), h) - exponent).exp()
            logs.append(f.ln() + exponent + (1 - cutoff).ln())
        reference = max(logs)
        weights = [(weight - reference).exp() for weight in logs]
        target = Decimal.from_float(shell_uniform) * sum(weights)
        cumulative = Decimal(0)
        for index, weight in enumerate(weights):
            cumulative += weight
            if target < cumulative:
                _, binding, height = shells[index]
                outgoing, momentum = original_tail_conditional_energy(
                    energy, cosine, energy_uniform, binding, height
                )
                return outgoing, index, momentum
    raise ValueError("original mixture shell selection failed")
