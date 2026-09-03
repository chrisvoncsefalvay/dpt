"""Independent primary/single-scatter cuboid integrals and omitted-order bounds.

This validation module never calls transport, its DDA, RNG or derivative replay.
It is not a runtime inverse oracle. The source is a fixed unit-direction pencil
entering the centre of the lower z face of one homogeneous square cuboid. The
detector is a centred square in a z plane strictly above the cuboid. Outside is
vacuum. Only upward outgoing single-scatter rays can reach this detector.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from typing import Literal

from .transport import klein_nishina_total_ratio


@dataclass(frozen=True, slots=True)
class ThinScatteringReference:
    """Truncated expected values, rigorous order tails and empirical quadrature error.

    Mean lies between ``mean`` and ``mean + value_tail_bound``, apart from
    quadrature error. The signed density derivative differs from the truncated
    value by at most ``derivative_tail_bound``, apart from quadrature error.
    Quadrature errors are differences between the final two deterministic
    refinements, not certified interval-arithmetic bounds. Callers must inspect
    refinement convergence before treating them as a numerical error allowance.
    """

    mean: float
    log_density_derivative: float
    primary: float
    single_scatter: float
    value_tail_bound: float
    derivative_tail_bound: float
    quadrature_value_error: float
    quadrature_derivative_error: float
    previous_value_error: float
    previous_derivative_error: float
    quadrature_orders: tuple[int, int, int]
    maximum_scattering_probability: float
    diameter_mm: float


@lru_cache(maxsize=16)
def _gauss_legendre(order: int) -> tuple[tuple[float, float], ...]:
    """Legendre roots and weights from the defining polynomial recurrence."""
    nodes: list[tuple[float, float]] = []
    for index in range(1, order + 1):
        root = math.cos(math.pi * (index - 0.25) / (order + 0.5))
        for _ in range(64):
            prior, value = 1.0, root
            for degree in range(2, order + 1):
                prior, value = (
                    value,
                    ((2 * degree - 1) * root * value - (degree - 1) * prior) / degree,
                )
            derivative = order * (root * value - prior) / (root * root - 1.0)
            correction = value / derivative
            root -= correction
            if abs(correction) <= 2e-16:
                break
        else:
            raise ArithmeticError("Legendre quadrature root did not converge")
        # Re-evaluate at the final root rather than reusing the previous iterate.
        prior, value = 1.0, root
        for degree in range(2, order + 1):
            prior, value = value, ((2 * degree - 1) * root * value - (degree - 1) * prior) / degree
        derivative = order * (root * value - prior) / (root * root - 1.0)
        nodes.append((root, 2 / ((1 - root * root) * derivative * derivative)))
    return tuple(nodes)


def _rule(lower: float, upper: float, order: int) -> tuple[tuple[float, float], ...]:
    half = (upper - lower) / 2
    centre = (upper + lower) / 2
    return tuple((centre + half * node, half * weight) for node, weight in _gauss_legendre(order))


def _coefficient(energy: float, nodes: tuple[float, ...], values: tuple[float, ...]) -> float:
    if len(nodes) == 1:
        return values[0]
    index = min(max(bisect_right(nodes, energy) - 1, 0), len(nodes) - 2)
    fraction = (energy - nodes[index]) / (nodes[index + 1] - nodes[index])
    return values[index] + fraction * (values[index + 1] - values[index])


def primary_single_scatter(
    *,
    absorption: tuple[float, ...],
    scattering: tuple[float, ...],
    coefficient_energies_kev: tuple[float, ...] = (80.0,),
    energy_kev: float = 80.0,
    scattering_law: Literal["isotropic-elastic", "free-electron-compton"] = "isotropic-elastic",
    scoring: Literal["photon-count", "energy-kev"] = "photon-count",
    density: float = 1.0,
    half_width_mm: float = 0.5,
    thickness_mm: float = 1.0,
    detector_half_width_mm: float = 1.0,
    detector_z_mm: float = 2.0,
    source_weight: float = 1.0,
    source_amplitude: float = 1.0,
    quadrature_order: int = 12,
) -> ThinScatteringReference:
    """E[S]_0+1 = S_primary + integral rho*mu_s(E0)*phase(u)*score(E')*exp(-T).

    Integrate the primary and exactly one scattering event in a homogeneous
    cuboid [-h,h]² x [0,L], with a pencil source entering (0,0,0) along +z and
    a centred detector [-R,R]² at z=Z>L. ``absorption``, ``scattering`` and
    ``coefficient_energies_kev`` are equal-length tuples in node order, with
    coefficients in mm^-1 at unit density and energies in keV. ``density`` is
    a positive dimensionless multiplier and all geometry lengths use mm.
    Source weight and amplitude are non-negative dimensionless factors.

    Python binary64 quadrature returns ``ThinScatteringReference`` scalar means,
    primary and single-scatter contributions, log-density derivative and error
    bounds. Scores use weighted photon counts for ``photon-count`` or keV for
    ``energy-kev``, and derivatives and their bounds use the same units.
    ``diameter_mm`` is in mm, ``maximum_scattering_probability`` is
    dimensionless and ``quadrature_orders`` records (q,2q,4q). Only log density
    is differentiated, with source, geometry, coefficients and scattering law
    fixed. The returned numerical errors are differences between successive
    refinements and the separate order-tail bounds cover omitted events.

    At depth z, let H=Z-z. Eight azimuthal symmetries reduce phi to [0,pi/4]
    and the detector accepts cosines u>=H*cos(phi)/sqrt(R²+H²*cos(phi)²).
    Escape distance is min((L-z)/u,h/(sqrt(1-u²)*cos(phi))). The attenuation
    depth T=rho*[mu_t(E0)*z+mu_t(E')*escape] gives derivative integrand
    (1-T) times the single-scatter integrand. The primary derivative is
    -rho*mu_t(E0)*L*S_primary. Quadrature splits at top/side ownership and
    coefficient interpolation knots independently of the voxel traversal.

    Let D be the cuboid diameter, q=1-exp(-rho*max(mu_s)*D),
    q0=1-exp(-rho*mu_s(E0)*L), K=rho*(max(mu_a)+max(mu_s))*D and r=q/(1-q).
    A score bound W gives value tail <=W*q0*q and derivative tail
    <=W*q0*q*[(2+r)+K*(3+r)]. Compton tables must cover [0,E0] so their node
    maxima bound every continuation energy. The isotropic law is elastic and
    the free-electron Compton law uses the corresponding energy change and
    normalised Klein-Nishina angular density.

        >>> import math
        >>> from dpt.validation.transport_weighted import primary_single_scatter
        >>> result = primary_single_scatter(absorption=(0.125,), scattering=(0.,),
        ...                                 quadrature_order=4)
        >>> result.mean == math.exp(-0.125)
        True
        >>> (result.single_scatter, result.value_tail_bound, result.quadrature_orders)
        (0.0, 0.0, (4, 8, 16))
        >>> result.log_density_derivative == -0.125 * result.mean
        True

    """
    positive = (density, half_width_mm, thickness_mm, detector_half_width_mm, energy_kev)
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("density, dimensions and source energy must be finite and positive")
    if not math.isfinite(detector_z_mm) or detector_z_mm <= thickness_mm:
        raise ValueError("detector must lie above the cuboid")
    if any(not math.isfinite(value) or value < 0 for value in (source_weight, source_amplitude)):
        raise ValueError("source factors must be finite and nonnegative")
    nodes = coefficient_energies_kev
    if (
        not nodes
        or len(absorption) != len(nodes)
        or len(scattering) != len(nodes)
        or any(
            not math.isfinite(value) or value < 0 for value in (*nodes, *absorption, *scattering)
        )
        or any(right <= left for left, right in pairwise(nodes))
        or not nodes[0] <= energy_kev <= nodes[-1]
    ):
        raise ValueError("provide nonnegative finite coefficient tables covering source energy")
    if scattering_law not in ("isotropic-elastic", "free-electron-compton"):
        raise ValueError("unsupported conditional scattering law")
    compton = scattering_law == "free-electron-compton"
    if compton and (len(nodes) < 2 or nodes[0] != 0.0):
        raise ValueError("Compton tail bounds require energy support covering [0, source energy]")
    if scoring not in ("photon-count", "energy-kev"):
        raise ValueError("unsupported detector scoring")
    if type(quadrature_order) is not int or not 4 <= quadrature_order <= 64:
        raise ValueError("quadrature_order must be an integer between 4 and 64")
    source = source_weight * source_amplitude
    score_bound = source * (energy_kev if scoring == "energy-kev" else 1.0)
    if not math.isfinite(score_bound):
        raise ValueError("this reference requires a finite unattenuated score bound")
    initial_absorption = _coefficient(energy_kev, nodes, absorption)
    initial_scattering = _coefficient(energy_kev, nodes, scattering)
    initial_total = initial_absorption + initial_scattering
    diameter = math.hypot(2 * half_width_mm, 2 * half_width_mm, thickness_mm)
    scattering_depth_bound = density * max(scattering) * diameter
    total_depth_bound = density * (max(absorption) + max(scattering)) * diameter
    probability = -math.expm1(-scattering_depth_bound)
    first_probability = -math.expm1(-density * initial_scattering * thickness_mm)
    if probability >= 1 or not math.isfinite(total_depth_bound):
        raise ValueError("the geometric order-tail bound is not numerically finite")
    second_probability = first_probability * probability
    remainder = probability / (1 - probability)
    value_tail = score_bound * second_probability
    derivative_tail = value_tail * ((2 + remainder) + total_depth_bound * (3 + remainder))
    optical_depth = density * initial_total * thickness_mm
    primary = score_bound * math.exp(-optical_depth)
    primary_derivative = -optical_depth * primary
    angular_normalisation = 2.0
    if compton:
        angular_normalisation = (8 / 3) * float(klein_nishina_total_ratio(energy_kev))
    energy_splits = (
        [1 - 510.99895069 * (1 / energy - 1 / energy_kev) for energy in nodes if energy > 0]
        if compton
        else []
    )

    def integrate(order: int) -> tuple[float, float]:
        if initial_scattering == 0 or source == 0:
            return 0.0, 0.0
        z_splits = [0.0, thickness_mm]
        if detector_half_width_mm != half_width_mm:
            transition = (detector_half_width_mm * thickness_mm - half_width_mm * detector_z_mm) / (
                detector_half_width_mm - half_width_mm
            )
            if 0 < transition < thickness_mm:
                z_splits.insert(1, transition)
        values: list[float] = []
        derivatives: list[float] = []
        for z_lower, z_upper in pairwise(z_splits):
            for z, z_weight in _rule(z_lower, z_upper, order):
                for phi, phi_weight in _rule(0.0, math.pi / 4, order):
                    cosine_phi = math.cos(phi)
                    detector_height = (detector_z_mm - z) * cosine_phi
                    lower_cosine = detector_height / math.hypot(
                        detector_half_width_mm, detector_height
                    )
                    box_height = (thickness_mm - z) * cosine_phi
                    face_cosine = box_height / math.hypot(half_width_mm, box_height)
                    splits = sorted(
                        {lower_cosine, 1.0}
                        | {
                            point
                            for point in (*energy_splits, face_cosine)
                            if lower_cosine < point < 1
                        }
                    )
                    angular_values: list[float] = []
                    angular_derivatives: list[float] = []
                    for lower, upper in pairwise(splits):
                        for cosine, cosine_weight in _rule(lower, upper, order):
                            outgoing = energy_kev
                            phase = 0.5
                            if compton:
                                ratio = 1 / (1 + energy_kev / 510.99895069 * (1 - cosine))
                                outgoing *= ratio
                                phase = (
                                    ratio**3 + ratio - ratio**2 * (1 - cosine**2)
                                ) / angular_normalisation
                            exit_distance = min(
                                (thickness_mm - z) / cosine,
                                half_width_mm / (math.sqrt(1 - cosine * cosine) * cosine_phi),
                            )
                            outgoing_total = _coefficient(
                                outgoing, nodes, absorption
                            ) + _coefficient(outgoing, nodes, scattering)
                            depth = density * (initial_total * z + outgoing_total * exit_distance)
                            score = outgoing if scoring == "energy-kev" else 1.0
                            integrand = cosine_weight * phase * score * math.exp(-depth)
                            angular_values.append(integrand)
                            angular_derivatives.append(integrand * (1 - depth))
                    scale = (
                        source
                        * density
                        * initial_scattering
                        * (4 / math.pi)
                        * z_weight
                        * phi_weight
                    )
                    values.append(scale * math.fsum(angular_values))
                    derivatives.append(scale * math.fsum(angular_derivatives))
        return math.fsum(values), math.fsum(derivatives)

    orders = (quadrature_order, 2 * quadrature_order, 4 * quadrature_order)
    estimates = [integrate(order) for order in orders]
    single, single_derivative = estimates[-1]
    return ThinScatteringReference(
        primary + single,
        primary_derivative + single_derivative,
        primary,
        single,
        value_tail,
        derivative_tail,
        abs(single - estimates[-2][0]),
        abs(single_derivative - estimates[-2][1]),
        abs(estimates[-2][0] - estimates[-3][0]),
        abs(estimates[-2][1] - estimates[-3][1]),
        orders,
        probability,
        diameter,
    )
