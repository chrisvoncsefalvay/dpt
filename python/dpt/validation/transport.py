"""Independent analytic transport oracles; never called by the CUDA operator.

These use complete-path limits and Decimal arithmetic rather than replaying the
production voxel traversal. They are mathematical validation cases, not material
assets or independently measured physical data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext


@dataclass(frozen=True, slots=True)
class AbsorbingSlabReference:
    transmission: Decimal
    log_density_derivative: tuple[Decimal, ...]
    variance_per_history: Decimal


def absorbing_layers(
    layer_optical_depths: tuple[float, ...],
) -> AbsorbingSlabReference:
    """P=exp(-sum_i tau_i) and dP/dlog(rho_i)=-tau_i*P.

    ``layer_optical_depths`` is a non-empty length-N tuple of finite,
    non-negative dimensionless depths tau_i=rho_i*mu_i*length_i. The returned
    ``AbsorbingSlabReference`` contains scalar survival probability P, a length-N
    tuple of log-density derivatives in layer order and scalar per-history
    Bernoulli variance P*(1-P). All outputs are dimensionless ``Decimal`` values
    evaluated at exact binary64 inputs with 70 significant decimal digits.

    Every layer's log density is differentiated with its coefficient and
    physical path fixed. The detector gives a unit score to survivors and zero
    to absorbed histories, which defines the reported variance.

        >>> from dpt.validation.transport import absorbing_layers
        >>> result = absorbing_layers((0., 0.))
        >>> (float(result.transmission), tuple(map(float, result.log_density_derivative)),
        ...  float(result.variance_per_history))
        (1.0, (0.0, 0.0), 0.0)

    """
    if not layer_optical_depths or any(
        not math.isfinite(value) or value < 0 for value in layer_optical_depths
    ):
        raise ValueError("supply finite nonnegative layer optical depths")
    with localcontext() as context:
        context.prec = 70
        depths = tuple(Decimal.from_float(value) for value in layer_optical_depths)
        transmission = (-sum(depths, Decimal(0))).exp()
        derivatives = tuple(-depth * transmission for depth in depths)
        return AbsorbingSlabReference(transmission, derivatives, transmission * (1 - transmission))


def compton_energy(incident_kev: float, cosine: float) -> Decimal:
    """E'=E/[1+(E/510.99895069)*(1-cos(theta))] for a free stationary electron.

    ``incident_kev`` is one positive finite binary64 incident energy in keV and
    ``cosine`` is one scattering-angle cosine in [-1,1]. The scalar ``Decimal``
    scattered photon energy is returned in keV, using exact binary64 inputs
    and 70 significant decimal digits with electron rest energy 510.99895069
    keV. This forward kinematic reference has no gradient output.

        >>> from dpt.validation.transport import compton_energy
        >>> float(compton_energy(80., 1.))
        80.0
        >>> 0 < compton_energy(80., -1.) < 80
        True

    """
    if not math.isfinite(incident_kev) or incident_kev <= 0 or not -1 <= cosine <= 1:
        raise ValueError("positive incident energy and a cosine in [-1,1] are required")
    with localcontext() as context:
        context.prec = 70
        incident = Decimal.from_float(incident_kev)
        return incident / (
            1 + incident / Decimal("510.99895069") * (1 - Decimal.from_float(cosine))
        )


def klein_nishina_total_ratio(energy_kev: float) -> Decimal:
    """R(E)=sigma_KN(E)/sigma_Thomson=integral(d sigma_KN/d Omega) d Omega/sigma_Thomson.

    ``energy_kev`` is one positive finite binary64 photon energy in keV. The
    returned scalar dimensionless ``Decimal`` ratio uses the integrated
    free-electron Klein-Nishina formula with alpha=E/510.99895069 and 90
    significant decimal digits at the exact supplied input. High precision
    resolves cancellation as alpha approaches zero and R approaches one.

    This forward normalisation reference has no active derivatives. It checks
    the conditional scattering law and leaves sourced macroscopic interaction
    coefficients to the material model.

        >>> from dpt.validation.transport import klein_nishina_total_ratio
        >>> round(float(klein_nishina_total_ratio(80.)), 6)
        0.777585

    """
    if not math.isfinite(energy_kev) or energy_kev <= 0:
        raise ValueError("energy must be positive and finite")
    with localcontext() as context:
        context.prec = 90
        alpha = Decimal.from_float(energy_kev) / Decimal("510.99895069")
        one = Decimal(1)
        log = (one + 2 * alpha).ln()
        return (
            Decimal(3)
            / 4
            * (
                (one + alpha) / alpha**3 * (2 * alpha * (one + alpha) / (one + 2 * alpha) - log)
                + log / (2 * alpha)
                - (one + 3 * alpha) / (one + 2 * alpha) ** 2
            )
        )


@dataclass(frozen=True, slots=True)
class BernoulliInverseReference:
    expected_signal: float
    expected_loss: float
    true_amplitude_gradient: float
    independent_gradient_expectation: float
    reused_history_gradient_expectation: float


def bernoulli_inverse(
    probability: float,
    amplitude: float,
    observed: float,
) -> BernoulliInverseReference:
    """J(a)=0.5*(a*p-y)² and dJ/da=(a*p-y)*p for Z~Bernoulli(p).

    ``probability`` is a scalar in [0,1], ``amplitude`` scales the unit score Z
    into signal units and ``observed`` is a scalar in the same signal units.
    The five binary64 fields of ``BernoulliInverseReference`` are expected
    signal, loss, true amplitude derivative, independent-history gradient
    expectation and reused-history gradient expectation. Loss uses squared
    signal units and derivatives use signal units per unit score. Only amplitude
    is active, while probability and observation remain fixed.

    Exact enumeration of the four independent outcome pairs gives the true
    gradient expectation. Reusing one outcome instead gives p*(a-y), exposing
    the bias a*p*(1-p) of coupling a stochastic residual with its derivative.

        >>> from dpt.validation.transport import bernoulli_inverse
        >>> result = bernoulli_inverse(0.5, 2., 0.)
        >>> (result.true_amplitude_gradient, result.independent_gradient_expectation,
        ...  result.reused_history_gradient_expectation)
        (0.5, 0.5, 1.0)

    """
    if not 0 <= probability <= 1 or not all(map(math.isfinite, (amplitude, observed))):
        raise ValueError("invalid Bernoulli inverse parameters")
    outcomes = ((0.0, 1.0 - probability), (1.0, probability))
    independent = math.fsum(
        left_mass * right_mass * (amplitude * left - observed) * right
        for left, left_mass in outcomes
        for right, right_mass in outcomes
    )
    reused = math.fsum(mass * (amplitude * value - observed) * value for value, mass in outcomes)
    signal = amplitude * probability
    return BernoulliInverseReference(
        signal,
        0.5 * (signal - observed) ** 2,
        (signal - observed) * probability,
        independent,
        reused,
    )
