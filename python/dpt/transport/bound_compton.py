"""Independent-atom relativistic impulse scattering with shell momentum profiles.

The analytical Hartree-Fock profile is the Brusa/Salvat profile parameterised
by the actual zero-momentum value. Binding and the positive photon-energy
constraint are applied to the same profile in the angle and energy laws.
"""

from __future__ import annotations

# Shared physical-data contracts and runtime validation of untyped callers.
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false
import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, overload

from dpt.materials import Provenance
from dpt.physical_data import ProviderIdentity, _xraylib_identity

from .model import ELECTRON_REST_ENERGY_KEV, TransportError, _physical

FINE_STRUCTURE_CONSTANT = 7.2973525643e-3
MODEL_VERSION = "hartree-fock-impulse-v1"
_OCCUPANCY_SHA256 = "40d12564ffd30957d8953f582d8de1cf842561f5f85786741fd721d1dbc775dc"


def _profile_log_slope(momentum: float, height: float) -> float:
    """Signed derivative of log J with no subtraction of near-equal terms.

    Apply the bounded factor to the smaller multiplier first: h*h need not
    be representable, and a subnormal final slope must retain its scale.
    """
    if momentum == 0.0:
        return 0.0
    amplitude = height * abs(momentum)
    if amplitude < 2.2250738585072014e-308:
        # Here h is below2**53 for nonzero binary64 momentum. The reciprocal
        # correction rounds to1, and forming8*h*h first avoids quantising h*p.
        magnitude = ((8.0 * height) * height) * abs(momentum)
        return -magnitude if momentum >= 0.0 else magnitude
    factor = 4.0 * (1.0 + 1.0 / (1.0 + 2.0 * amplitude))
    magnitude = (factor * min(height, amplitude)) * max(height, amplitude)
    return -magnitude if momentum >= 0.0 else magnitude


@dataclass(frozen=True, slots=True)
class ComptonShell:
    """A neutral-atom shell and its analytical Hartree-Fock momentum profile.

    ``occupancy`` is the number of electrons, ``binding_kev`` is the shell
    ionisation energy in keV and ``profile_zero`` is the per-electron Compton
    profile at zero longitudinal momentum in inverse atomic units. For
    dimensionless momentum ``p=p_z/(m_e*c)`` and ``h=profile_zero/alpha``,
    ``J(p)=h*(1+2*h*abs(p))*exp(-2*h*abs(p)-2*h*h*p*p)``. The profile is
    normalised on the complete real line and has the tabulated zero value.
    This is the analytical impulse profile of Brusa et al. (1996), with no
    finite momentum cutoff or inferred power-law tail. The leading impulse
    approximation sets the additional profile correction ``F(p)=1``.

    The shell law is immutable during material, coefficient, spectrum and
    geometry differentiation. Electron and vacancy energies are deposited
    locally by photon transport, while the outgoing photon is transported.
    Its inputs and ``momentum_height`` are host scalars in binary64 arithmetic;
    the record owns no device arrays.

    A synthetic shell demonstrates the normalised profile and momentum units:

    Reference: https://doi.org/10.1016/0168-9002(96)00652-3
    """

    name: str
    occupancy: float
    binding_kev: float
    profile_zero: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or len(self.name) > 32:
            raise TransportError("Compton shell needs an explicit shell name")
        _physical(self.occupancy, "shell occupancy", positive=True)
        _physical(self.binding_kev, "shell binding energy", nonnegative=True)
        _physical(self.profile_zero, "zero-momentum shell profile", positive=True)
        if not math.isfinite(self.profile_zero / FINE_STRUCTURE_CONSTANT):
            raise TransportError("shell profile exceeds dimensionless momentum range")

    @property
    def momentum_height(self) -> float:
        """Profile density at zero for momentum in units of ``m_e*c``."""
        return self.profile_zero / FINE_STRUCTURE_CONSTANT


@dataclass(frozen=True, slots=True)
class BoundComptonTable:
    """An explicit independent-atom relativistic impulse model for one element.

    For incident photon energy ``E``, scattering cosine ``c`` and outgoing
    energy ratio ``r=E'/E``, define ``A=1+E*(1-c)/(m_e*c**2)`` and
    ``p=(A*r-1)/sqrt(1+r*r-2*c*r)``. A shell is accessible when ``E>U``.
    Its momentum range is ``-1<p<p(E-U)``. The angular density is
    ``KN(E,c)*sum_i f_i*integral(J_i, -1, p(E-U_i))``, normalised over solid
    angle, and the conditional energy density is the corresponding shell
    mixture transformed by ``dp/dE'`` with the leading impulse factor ``F(p)=1``.
    Angle, shell and energy sampling
    therefore use the same physical law, including next-event connections.

    ``shells`` contains every occupied shell with occupancies summing to ``Z``.
    The represented independent-atom model uses analytical Hartree-Fock shell
    profiles and shell binding rather than the stationary scattering function.
    Photon energies are in keV and the model domain is ``0.1<=E<=5000`` keV.
    ``provenance`` records the actual profile and binding data. The optional
    provider identity records the installed data implementation hashes.

    Example::

        from dpt.transport import xraylib_bound_compton
        table = xraylib_bound_compton(13)
        assert sum(s.occupancy for s in table.shells) == 13

    Impulse model: https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/penelope_compton.html
    """

    atomic_number: int
    shells: tuple[ComptonShell, ...]
    provenance: Provenance
    provider: ProviderIdentity | None = None
    version: str = MODEL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "shells", tuple(self.shells))
        if type(self.atomic_number) is not int or not 1 <= self.atomic_number <= 100:
            raise TransportError("bound Compton atomic number must be in 1..100")
        if not self.shells or any(not isinstance(s, ComptonShell) for s in self.shells):
            raise TransportError("bound Compton requires actual occupied shell records")
        if len({s.name for s in self.shells}) != len(self.shells):
            raise TransportError("bound Compton shell names must be unique")
        if math.fsum(s.occupancy for s in self.shells) != self.atomic_number:
            raise TransportError("bound Compton occupancies must sum to the atomic number")
        if self.version != MODEL_VERSION:
            raise TransportError("unsupported bound Compton physical-model version")
        if not isinstance(self.provenance, Provenance):
            raise TransportError("bound Compton requires physical data provenance")
        if self.provider is not None and not isinstance(self.provider, ProviderIdentity):
            raise TransportError("bound Compton provider identity is invalid")


def xraylib_bound_compton(atomic_number: int) -> BoundComptonTable:
    """Load matching Biggs occupancies, profile heights and actual binding data.

    This host operation queries installed xraylib 4.3.0 ``ComptonProfile_Partial``
    at zero momentum and ``EdgeEnergy``. Matching occupancies come from the
    hash-pinned Biggs profile dataset, because xraylib's public ``ElectronConfig``
    exposes a different atomic configuration. Every occupied shell must have
    both profile and binding data. An unsupported shell raises immediately.

    The return value contains immutable per-electron analytical shell profiles,
    with hashes of all queried values, the pinned occupancy source and the
    actual provider implementation. It needs no momentum extrapolation,
    network access or device allocation.

    ``atomic_number`` is a scalar integer. The returned shell tuple contains
    host binary64 binding energies in keV and profile heights in inverse
    atomic units. All occupied shells must have provider data; the helper
    raises ``TransportError`` identifying an unavailable shell.

    With the physical-data extra installed, load aluminium's complete record:

    Data: https://github.com/tschoonj/xraylib/blob/xraylib-4.3.0/data/comptonprofiles.dat
    """
    if type(atomic_number) is not int or not 1 <= atomic_number <= 100:
        raise TransportError("bound Compton atomic number must be in 1..100")
    payload = files("dpt.transport").joinpath("data/BIGGS_OCCUPANCIES.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != _OCCUPANCY_SHA256:
        raise TransportError("bound Compton occupancy resource hash mismatch")
    document: Any = json.loads(payload)
    occupancies = document["elements"][atomic_number - 1]["occupancy"]
    module: Any = importlib.import_module("xraylib")
    if module.__version__ != "4.3.0":
        raise TransportError("bound Compton provider must be pinned xraylib 4.3.0")
    identity = _xraylib_identity(module)
    names = {
        getattr(module, n): n.removesuffix("_SHELL")
        for n in dir(module)
        if n.endswith("_SHELL") and isinstance(getattr(module, n), int)
    }
    shells: list[ComptonShell] = []
    for index, occupancy in enumerate(occupancies):
        if occupancy > 0:
            try:
                binding = float(module.EdgeEnergy(atomic_number, index))
                height = float(module.ComptonProfile_Partial(atomic_number, index, 0.0))
            except (ValueError, TypeError, OverflowError) as error:
                raise TransportError(
                    f"xraylib bound Compton data are unavailable for Z={atomic_number}, "
                    f"occupied shell {names.get(index, str(index))}: {error}"
                ) from error
            shells.append(ComptonShell(names[index], float(occupancy), binding, height))
    canonical = json.dumps(
        {
            "Z": atomic_number,
            "shells": [(s.name, s.occupancy, s.binding_kev, s.profile_zero) for s in shells],
            "occupancy_source": document["source"],
            "occupancy_source_sha256": document["sha256"],
            "provider": identity.version,
            "artifacts": identity.artifacts,
            "model": MODEL_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    provenance = Provenance(
        "https://github.com/tschoonj/xraylib/tree/xraylib-4.3.0",
        hashlib.sha256(canonical).hexdigest(),
        "BSD-3-Clause: https://github.com/tschoonj/xraylib/blob/xraylib-4.3.0/license_all.txt",
        "Canonical shell names, actual Biggs occupancies, J(0), EdgeEnergy, provider hashes "
        "and analytical Hartree-Fock impulse-model version.",
    )
    return BoundComptonTable(atomic_number, tuple(shells), provenance, identity)


def profile_cdf(momentum: float, height: float) -> float:
    """CDF of the normalised analytical shell profile in ``m_e*c`` units."""
    exponent = -2 * height * abs(momentum) * (1 + height * abs(momentum))
    tail = 0.5 * math.exp(exponent)
    return tail if momentum <= 0 else 1 - tail


def profile_inverse(probability: float, height: float) -> float:
    """Inverse profile CDF, with rationalised cancellation near its median."""
    tail = min(probability, 1 - probability)
    if not 0 < tail <= 0.5:
        raise TransportError("profile inverse requires probability strictly between zero and one")
    logarithm = -math.log(2 * tail)
    magnitude = logarithm / (height * (math.sqrt(1 + 2 * logarithm) + 1))
    return -magnitude if probability < 0.5 else magnitude


def profile_interval_mass(lower: float, upper: float, height: float) -> float:
    """Profile probability on ``[lower, upper]``, retaining small interval mass.

    Momentum endpoints use ``m_e*c`` units. Tail differences use ``expm1``;
    intervals crossing zero are added as two positive median distances.
    """
    if lower >= upper:
        return 0.0
    if lower < 0 < upper:
        return 0.5 * math.fsum(
            (
                -math.expm1(-2 * height * abs(lower) * (1 + height * abs(lower))),
                -math.expm1(-2 * height * upper * (1 + height * upper)),
            )
        )
    near, far = (abs(upper), abs(lower)) if upper <= 0 else (lower, upper)
    exponent = -2 * height * near * (1 + height * near)
    difference = -2 * height * (upper - lower) * (1 + height * (near + far))
    return 0.5 * math.exp(exponent) * -math.expm1(difference)


def _profile_log_inverse(log_tail_twice: float, height: float) -> float:
    logarithm = -log_tail_twice
    return logarithm / (height * (math.sqrt(1 + 2 * logarithm) + 1))


def _profile_interval_inverse_pair(
    fraction: float,
    complement: float,
    lower: float,
    upper: float,
    height: float,
) -> float:
    """Invert a truncated law using separately retained left/right fractions."""
    if lower < 0 < upper:
        left = profile_interval_mass(lower, 0.0, height)
        right = profile_interval_mass(0.0, upper, height)
        displacement = fraction * right - complement * left
        if displacement < 0:
            return _profile_interval_inverse_pair(
                fraction * (left + right) / left,
                -displacement / left,
                lower,
                0.0,
                height,
            )
        if displacement > 0:
            return _profile_interval_inverse_pair(
                displacement / right,
                complement * (left + right) / right,
                0.0,
                upper,
                height,
            )
        return 0.0
    near, far = (abs(upper), abs(lower)) if upper <= 0 else (lower, upper)
    exponent = -2 * height * near * (1 + height * near)
    difference = -2 * height * (upper - lower) * (1 + height * (near + far))
    relative_mass = -math.expm1(difference)
    weight = complement if upper <= 0 else fraction
    other = fraction if upper <= 0 else complement
    if weight * relative_mass <= 0.5:
        logarithm = exponent + math.log1p(-weight * relative_mass)
    else:
        logarithm = exponent + math.log(other + weight * math.exp(difference))
    magnitude = _profile_log_inverse(logarithm, height)
    return -magnitude if upper <= 0 else magnitude


def profile_interval_inverse(
    probability: float,
    lower: float,
    upper: float,
    height: float,
) -> float:
    """Sample the profile conditioned on a finite nonempty momentum interval.

    ``probability`` is the original open-unit draw. Inversion retains distances
    from the median instead of adding tiny mass to a CDF rounded to one half.
    """
    if not 0 < probability < 1 or not lower < upper or not height > 0:
        raise TransportError("profile interval inverse requires an open draw and ordered support")
    return _profile_interval_inverse_pair(probability, 1 - probability, lower, upper, height)


def shell_momentum_max(energy: float, cosine: float, binding: float) -> float:
    """Largest initial momentum compatible with ``E'=E-U`` in the impulse law."""
    outgoing = energy - binding
    numerator = energy * outgoing * (1 - cosine) / ELECTRON_REST_ENERGY_KEV - binding
    denominator = math.sqrt(2 * energy * outgoing * (1 - cosine) + binding * binding)
    if denominator == 0:
        return 0.0
    if abs(numerator) < 1e-3 * (abs(numerator + binding) + binding):
        from .bound_compton_tail import _kinematic_pair, _two_sum

        transfer, error = _two_sum(1.0, -cosine)
        numerator, denominator = _kinematic_pair(energy, transfer, binding, error)
    return numerator / denominator


def shell_accessibility(energy: float, cosine: float, shell: ComptonShell) -> float:
    """Profile probability with positive outgoing photon and recoil energy."""
    if energy <= shell.binding_kev:
        return 0.0
    height = shell.momentum_height
    upper = shell_momentum_max(energy, cosine, shell.binding_kev)
    return profile_interval_mass(-1.0, upper, height)


def impulse_energy(energy: float, cosine: float, momentum: float, binding: float) -> float:
    """Invert the monotone relativistic impulse momentum relation in keV.

    Bisection avoids subtractive quadratic roots at ``p=+/-1`` and the
    removable quadratic singularity. Its bracket enforces shell binding,
    and 56 iterations resolve the represented binary64 outgoing ratio.
    """
    lower, upper = 0.0, (energy - binding) / energy
    factor = 1 + energy / ELECTRON_REST_ENERGY_KEV * (1 - cosine)
    for _ in range(56):
        ratio = (lower + upper) * 0.5
        q = math.sqrt((1 - ratio) ** 2 + 2 * ratio * (1 - cosine))
        value = (factor * ratio - 1) / q
        if value < momentum:
            lower = ratio
        else:
            upper = ratio
    return energy * (lower + upper) * 0.5


def impulse_angular_weight(energy: float, cosine: float, table: BoundComptonTable) -> float:
    """Unnormalised angular impulse law, with ``KN`` omitting its constant factor."""
    ratio = 1 / (1 + energy / ELECTRON_REST_ENERGY_KEV * (1 - cosine))
    kn = ratio * ratio * (ratio + 1 / ratio - 1 + cosine * cosine)
    return kn * math.fsum(
        s.occupancy * shell_accessibility(energy, cosine, s) for s in table.shells
    )


def bound_compton_joint_pdf(
    energy: float,
    cosine: float,
    outgoing_kev: float,
    table: BoundComptonTable,
    *,
    solid_angle_normalisation: float | None = None,
) -> float:
    """Evaluate ``P(c,E'|E) = KN(E,c)*sum_i f_i*J_i(p)*dp/dE' / N(E)``.

    Supply the solid-angle integral of ``impulse_angular_weight`` at ``energy``.
    Geometry scores must differentiate this complete density, including its
    energy Jacobian and shell support, when the direction or outgoing energy
    changes. The immutable shell law contributes no material-density score.
    All inputs and the return value are host scalars in binary64 arithmetic.
    ``energy`` and ``outgoing_kev`` are keV; ``cosine`` is dimensionless. The
    returned density has units sr^-1 keV^-1 and is zero outside shell support.

    """
    if not 0 < outgoing_kev < energy or not -1 <= cosine < 1:
        return 0.0
    from .bound_compton_mixed import (
        mixed_joint_pdf,
        mixed_log_raw_normalisation,
        uses_mixed_conditioning,
    )

    if uses_mixed_conditioning(energy, table):
        value = mixed_joint_pdf(energy, cosine, outgoing_kev, table)
        if solid_angle_normalisation is not None and solid_angle_normalisation != 0.0:
            _physical(solid_angle_normalisation, "angular normalisation", positive=True)
            value *= math.exp(
                mixed_log_raw_normalisation(energy, table) - math.log(solid_angle_normalisation)
            )
        return value
    occupancy_scale = accessible_occupancy_scale(energy, table)
    if solid_angle_normalisation is None or solid_angle_normalisation == 0.0:
        from .bound_compton_quadrature import bound_angular_normalisation_scaled_pair
        from .bound_compton_tail import tail_joint_pdf, uses_scaled_tail

        actual_normalisation = bound_angular_normalisation_scaled_pair(
            energy, table, occupancy_scale
        )[0]
        if actual_normalisation == 0.0 or uses_scaled_tail(energy, table):
            return tail_joint_pdf(energy, cosine, outgoing_kev, table)
        if solid_angle_normalisation is None or actual_normalisation * occupancy_scale == 0.0:
            solid_angle_normalisation = actual_normalisation
    else:
        solid_angle_normalisation /= occupancy_scale
    _physical(solid_angle_normalisation, "angular normalisation", positive=True)
    ratio = outgoing_kev / energy
    factor = 1 + energy / ELECTRON_REST_ENERGY_KEV * (1 - cosine)
    q = math.sqrt((1 - ratio) ** 2 + 2 * ratio * (1 - cosine))
    momentum = (factor * ratio - 1) / q
    if abs(factor * ratio - 1) < 1e-3 * (abs(factor * ratio) + 1):
        from .bound_compton_tail import joint_kinematic_pair

        numerator, physical_q = joint_kinematic_pair(energy, cosine, outgoing_kev)
        momentum = numerator / physical_q
    if momentum <= -1:
        return 0.0
    jacobian = (factor - cosine + ratio * (1 - factor * cosine)) / (energy * q**3)
    density = math.fsum(
        (shell.occupancy / occupancy_scale)
        * shell.momentum_height
        * (1 + 2 * shell.momentum_height * abs(momentum))
        * math.exp(
            -2 * shell.momentum_height * abs(momentum) * (1 + shell.momentum_height * abs(momentum))
        )
        for shell in table.shells
        if outgoing_kev < energy - shell.binding_kev
    )
    stationary = 1 / factor
    kn = stationary * stationary * (stationary + factor - 1 + cosine * cosine)
    return kn * density * jacobian / solid_angle_normalisation


@overload
def bound_compton_joint_scores(
    energy: float,
    cosine: float,
    outgoing_kev: float,
    table: BoundComptonTable,
    *,
    normalisation_energy_score: None = None,
) -> tuple[float, float]: ...


@overload
def bound_compton_joint_scores(
    energy: float,
    cosine: float,
    outgoing_kev: float,
    table: BoundComptonTable,
    *,
    normalisation_energy_score: float,
) -> tuple[float, float, float]: ...


def bound_compton_joint_scores(
    energy: float,
    cosine: float,
    outgoing_kev: float,
    table: BoundComptonTable,
    *,
    normalisation_energy_score: float | None = None,
) -> tuple[float, float] | tuple[float, float, float]:
    """Return ``(d log P/dc, d log P/dE')`` inside the joint shell support.

    The first derivative holds both photon energies fixed and the second holds
    incident energy and cosine fixed. The joint law includes the momentum
    profile and its energy Jacobian. These derivatives apply inside shell
    support. Moving support boundaries need their separate boundary estimator.
    The angular normalisation depends only on incident energy and data. Supply
    ``normalisation_energy_score=d(log N)/dE`` to obtain a third derivative in
    incident energy, with outgoing energy and cosine fixed, for chained
    scattering geometry scores. Energy derivative units are inverse keV.
    At an exact incident shell threshold ``E=U_i``, that third derivative is
    undefined and raises. The cosine and outgoing-energy scores remain interior
    derivatives of the accessible shell mixture.
    Inputs and arithmetic are host binary64 scalars. The default result is a
    two-scalar tuple; supplying the normalisation energy score adds the incident
    energy derivative as its third member. Cosine is dimensionless and both
    photon energies are keV, so the energy derivatives have units keV^-1.

    """
    if not 0 < outgoing_kev < energy or not -1 < cosine < 1:
        raise TransportError("joint Compton scores require interior energy and angular support")
    if normalisation_energy_score is not None and any(
        energy == shell.binding_kev for shell in table.shells
    ):
        raise TransportError("incident Compton score is on a shell threshold")
    from .bound_compton_mixed import (
        mixed_joint_scores,
        mixed_normalisation,
        uses_mixed_conditioning,
    )

    if uses_mixed_conditioning(energy, table):
        if any(outgoing_kev == energy - shell.binding_kev for shell in table.shells):
            raise TransportError("joint Compton score is on a shell support boundary")
        scores = mixed_joint_scores(energy, cosine, outgoing_kev, table)
        if normalisation_energy_score is None:
            return scores[:2]
        actual_score = mixed_normalisation(energy, table).energy_score
        return scores[0], scores[1], scores[2] + (actual_score - normalisation_energy_score)
    ratio = outgoing_kev / energy
    k = energy / ELECTRON_REST_ENERGY_KEV
    factor = 1 + k * (1 - cosine)
    q2 = (1 - ratio) ** 2 + 2 * ratio * (1 - cosine)
    q = math.sqrt(q2)
    p = (factor * ratio - 1) / q
    if abs(factor * ratio - 1) < 1e-3 * (abs(factor * ratio) + 1):
        from .bound_compton_tail import joint_kinematic_pair

        numerator, physical_q = joint_kinematic_pair(energy, cosine, outgoing_kev)
        p = numerator / physical_q
    n = factor - cosine + ratio * (1 - factor * cosine)
    occupancy_scale = accessible_occupancy_scale(energy, table)
    density = 0.0
    derivative = 0.0
    for shell in table.shells:
        if outgoing_kev == energy - shell.binding_kev:
            raise TransportError("joint Compton score is on a shell support boundary")
        if outgoing_kev < energy - shell.binding_kev:
            height = shell.momentum_height
            absolute = abs(p)
            value = (
                (shell.occupancy / occupancy_scale)
                * height
                * (1 + 2 * height * absolute)
                * math.exp(-2 * height * absolute * (1 + height * absolute))
            )
            slope = _profile_log_slope(p, height)
            derivative += value * slope
            density += value
    from .bound_compton_tail import uses_scaled_tail

    if uses_scaled_tail(energy, table):
        from .bound_compton_tail import scaled_bound_normalisation, tail_joint_scores

        normalisation = scaled_bound_normalisation(energy, table)
        result = tail_joint_scores(energy, cosine, outgoing_kev, table, normalisation)
        if normalisation_energy_score is None:
            return result[0], result[1]
        if normalisation_energy_score == normalisation.energy_score:
            return result
        return (
            result[0],
            result[1],
            result[2] + normalisation.energy_score - normalisation_energy_score,
        )
    if density <= 0 or p <= -1:
        raise TransportError("joint Compton score has zero physical density")
    profile_score = derivative / density
    tau = 1 / factor
    kn_term = tau + factor - 1 + cosine * cosine
    kn_score = 2 * k * tau + (k * tau * tau - k + 2 * cosine) / kn_term
    p_c = -k * ratio / q + p * ratio / q2
    n_c = -k - 1 + ratio * (k * cosine - factor)
    score_c = kn_score + profile_score * p_c + n_c / n + 3 * ratio / q2
    p_r = n / (q * q2)
    score_e = (profile_score * p_r + (1 - factor * cosine) / n - 3 * (ratio - cosine) / q2) / energy
    if normalisation_energy_score is not None:
        _physical(normalisation_energy_score, "angular normalisation energy score")
        a_e = (1 - cosine) / ELECTRON_REST_ENERGY_KEV
        p_e = -ratio / (energy * q) + p * ratio * (ratio - cosine) / (energy * q2)
        n_e = a_e * (1 - ratio * cosine) - ratio / energy * (1 - factor * cosine)
        kn_e = -2 * tau * a_e + (1 - tau * tau) * a_e / kn_term
        score_incident = (
            kn_e
            + profile_score * p_e
            + n_e / n
            - 1 / energy
            + 3 * ratio * (ratio - cosine) / (energy * q2)
            - normalisation_energy_score
        )
        return score_c, score_e, score_incident
    return score_c, score_e


def accessible_occupancy_scale(energy: float, table: BoundComptonTable) -> float:
    """A common scale from occupied shells accessible at the incident energy."""
    largest = max((s.occupancy for s in table.shells if energy > s.binding_kev), default=0.0)
    return largest if 0.0 < largest < 1e-280 else 1.0


def bound_compton_shell_weights(
    energy: float, cosine: float, table: BoundComptonTable
) -> tuple[float, ...]:
    from .bound_compton_mixed import mixed_shell_weights, uses_mixed_conditioning

    if uses_mixed_conditioning(energy, table):
        return mixed_shell_weights(energy, cosine, table)
    """Conditional shell masses in a common, internally representable scale."""
    scale = accessible_occupancy_scale(energy, table)
    return tuple(s.occupancy / scale * shell_accessibility(energy, cosine, s) for s in table.shells)


def bound_compton_phase_pdf(energy: float, cosine: float, table: BoundComptonTable) -> float:
    """Matched angular density per steradian, retaining conditional shell weights."""
    if not -1.0 <= cosine < 1.0:
        return 0.0
    from .bound_compton_mixed import mixed_phase_pdf, uses_mixed_conditioning
    from .bound_compton_quadrature import bound_angular_normalisation_scaled_pair

    if uses_mixed_conditioning(energy, table):
        return mixed_phase_pdf(energy, cosine, table)
    from .bound_compton_tail import tail_phase_pdf, uses_scaled_tail

    scale = accessible_occupancy_scale(energy, table)
    value = bound_angular_normalisation_scaled_pair(energy, table, scale)[0]
    if value == 0.0 or uses_scaled_tail(energy, table):
        return tail_phase_pdf(energy, cosine, table)
    ratio = 1.0 / (1.0 + energy / ELECTRON_REST_ENERGY_KEV * (1.0 - cosine))
    kn = ratio * ratio * (ratio + 1.0 / ratio - 1.0 + cosine * cosine)
    return kn * math.fsum(bound_compton_shell_weights(energy, cosine, table)) / value


def bound_compton_angle_proposal(
    energy: float, table: BoundComptonTable, uniform: float
) -> tuple[float, float]:
    """Angle proposal and acceptance using one original open-interval mark."""
    from .bound_compton_mixed import mixed_angle_proposal, uses_mixed_conditioning
    from .bound_compton_tail import tail_angle_proposal, uses_scaled_tail

    if uses_mixed_conditioning(energy, table):
        return mixed_angle_proposal(energy, table, uniform)

    if uses_scaled_tail(energy, table):
        return tail_angle_proposal(energy, table, uniform)
    cosine = 2.0 * uniform - 1.0
    back = math.fsum(bound_compton_shell_weights(energy, -1.0, table))
    if back <= 0.0:
        raise TransportError("bound law has no representable conditional shell mass")
    ratio = 1.0 / (1.0 + energy / ELECTRON_REST_ENERGY_KEV * (1.0 - cosine))
    kn = ratio * ratio * (ratio + 1.0 / ratio - 1.0 + cosine * cosine)
    return cosine, 0.5 * kn * math.fsum(bound_compton_shell_weights(energy, cosine, table)) / back


def bound_compton_conditional_energy(
    energy: float,
    cosine: float,
    table: BoundComptonTable,
    shell_uniform: float,
    energy_uniform: float,
) -> tuple[float, int, float]:
    """Matched shell and broadened energy with the original two conditional marks."""
    from .bound_compton_mixed import mixed_conditional_energy, uses_mixed_conditioning
    from .bound_compton_tail import tail_conditional_energy, uses_scaled_tail

    if uses_mixed_conditioning(energy, table):
        return mixed_conditional_energy(energy, cosine, table, shell_uniform, energy_uniform)

    if uses_scaled_tail(energy, table, cosine):
        return tail_conditional_energy(energy, cosine, table, shell_uniform, energy_uniform)
    weights = bound_compton_shell_weights(energy, cosine, table)
    target = shell_uniform * math.fsum(weights)
    cumulative = 0.0
    for index, (shell, weight) in enumerate(zip(table.shells, weights, strict=True)):
        cumulative += weight
        if target < cumulative and weight > 0.0:
            upper = shell_momentum_max(energy, cosine, shell.binding_kev)
            momentum = profile_interval_inverse(energy_uniform, -1.0, upper, shell.momentum_height)
            return impulse_energy(energy, cosine, momentum, shell.binding_kev), index, momentum
    raise TransportError("bound conditional shell selection failed")
