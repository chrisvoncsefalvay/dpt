"""Conditional physical collision measures and attached geometry jets.

At fixed incoming energy and material data, a rigidly transported collision
frame has angular velocity ``omega = incoming cross incoming_dot``. Rotating
the sampled outgoing direction with the same velocity preserves its scattering
cosine and outgoing energy. This is a coupling of the conditional law, rather
than a derivative of the implementation's auxiliary-axis choice.

Terminal kernels below are densities with respect to solid angle, or solid
angle and outgoing keV for bound Compton scattering. Detector-area Jacobians,
attenuation, pixel jumps and moving energy-support boundaries belong to their
own geometry measures and must be composed by the caller.

Physical references: https://doi.org/10.1063/1.555523 and
https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/penelope_compton.html.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from typing import Literal, cast

from dpt.geometry import Vector3, cross, dot

from .bound_compton import (
    BoundComptonTable,
    bound_compton_angle_proposal,
    bound_compton_conditional_energy,
    bound_compton_joint_pdf,
    bound_compton_joint_scores,
)
from .geometry_tangents import RayTangent, normalised_direction_jet
from .model import ELECTRON_REST_ENERGY_KEV, TransportError, TransportSpec
from .physical import HC_KEV_ANGSTROM, AtomicScatteringTable, PhysicalTransport

Layout = Literal["legacy-v1", "packed-compton-v2", "bound-compton-v3"]
Channel = Literal[
    "isotropic", "free-compton", "bound-compton", "rayleigh", "fluorescence", "absorption", "cutoff"
]


@dataclass(frozen=True, slots=True)
class AddressedUniform:
    """One supplied open-interval uniform at an actual transport counter lane.

    ``event`` and ``domain`` are uint32 counters and ``lane`` is in 0..3.
    No random generator, seed conversion or missing-coordinate substitute is
    used here. The record is immutable and all arithmetic remains host binary64.

    Example::

        draw = AddressedUniform(event=0, domain=3, lane=0, value=0.25)
        assert draw.domain == 3

    """

    event: int
    domain: int
    lane: int
    value: float

    def __post_init__(self) -> None:
        for name, maximum in (("event", 2**32 - 1), ("domain", 2**32 - 1), ("lane", 3)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= maximum:
                raise TransportError(f"physical event {name} is outside its counter range")
        if not math.isfinite(self.value) or not 0.0 < self.value < 1.0:
            raise TransportError("physical event uniforms must be in the open unit interval")


@dataclass(frozen=True, slots=True)
class PhysicalEventDraws:
    """Explicit addressed coordinates for one collision of one descendant.

    The branch domain offset is ``branch*65536``. Collision, fluorescence and
    coherent selection domains are 0, 2 and 3. Compton proposal domains start
    at 0x80000000 and Rayleigh at 0xC0000000. Bound shell and momentum lanes are
    0 and 1 of 0x10000000, with 0x20000000 reserved for a next-event energy.
    The latter two domains require ``bound-compton-v3``. Packed Compton uses
    four lanes per proposal block and collision lane two for its azimuth.

    ``provenance`` identifies the caller's random design and original-history
    address. Only the coordinates actually requested by the physical law are
    returned in its used-coordinate ledger. Missing coordinates raise.

    Example::

        draws = PhysicalEventDraws(0, 0, "legacy-v1",
            (AddressedUniform(0, 0, 1, 0.25),), "Supplied collision coordinate")
        assert len(draws.uniforms) == 1

    """

    event: int
    branch: int
    layout: Layout
    uniforms: tuple[AddressedUniform, ...]
    provenance: str

    def __post_init__(self) -> None:
        if type(self.event) is not int or not 0 <= self.event < 2**32:
            raise TransportError("physical event number must be a uint32 counter")
        if type(self.branch) is not int or not 0 <= self.branch < 4096:
            raise TransportError("physical event branch must be in 0..4095")
        if self.layout not in ("legacy-v1", "packed-compton-v2", "bound-compton-v3"):
            raise TransportError("physical event layout version is unsupported")
        object.__setattr__(self, "uniforms", tuple(self.uniforms))
        if not isinstance(cast(object, self.provenance), str) or not self.provenance.strip():
            raise TransportError("physical event draws require explicit provenance")
        addresses: set[tuple[int, int]] = set()
        for draw in self.uniforms:
            if not isinstance(cast(object, draw), AddressedUniform) or draw.event != self.event:
                raise TransportError("physical draw records must belong to this event")
            address = (draw.domain, draw.lane)
            if address in addresses:
                raise TransportError("physical event draw addresses must be unique")
            addresses.add(address)


@dataclass(frozen=True, slots=True)
class PhysicalCollisionJet:
    """A conditional event with outgoing keV and mm-based world-ray jet.

    Terminal absorption or cutoff has ``outgoing=None`` and deposits the full
    incoming energy. A continuing photon deposits the incoming-minus-outgoing
    energy. ``angle-budget`` is an invalid finite-budget sample and must never
    be tallied as absorption. Energies stay fixed under this geometry coupling.
    ``total_mm_inverse`` includes the caller's explicit density factor.

    """

    channel: Channel
    status: Literal["continuing", "absorbed", "local-deposition", "angle-budget"]
    outgoing: RayTangent | None
    energy_kev: float
    deposited_kev: float
    total_mm_inverse: float
    used_uniforms: tuple[AddressedUniform, ...]
    draw_provenance: str
    line_index: int | None = None
    shell_index: int | None = None
    momentum: float | None = None


@dataclass(frozen=True, slots=True)
class PhysicalTerminalKernel:
    """One conditional outgoing density and its endpoint geometry derivative.

    ``density`` is ``rate/total * phase`` in sr^-1 for a discrete outgoing
    energy and in sr^-1 keV^-1 for the fixed-energy bound Compton joint mark.
    ``density_dot`` differentiates endpoint and incoming direction while holding
    material data and incident energy fixed. Free Compton's ``energy_dot``
    differentiates its kinematic outgoing energy. Rayleigh and fluorescent line
    energies have zero tangent. Exact angular/table/support chart boundaries
    with nonzero velocity raise and require a separate boundary measure.

    """

    channel: Channel
    outgoing: RayTangent
    cosine: float
    cosine_dot: float
    energy_kev: float
    energy_dot: float
    density: float
    density_dot: float
    measure: Literal["solid-angle", "solid-angle-energy"]
    line_index: int | None = None


class _DrawReader:
    def __init__(self, draws: PhysicalEventDraws) -> None:
        self.draws = draws
        self.lookup = {(item.domain, item.lane): item for item in draws.uniforms}
        self.used: list[AddressedUniform] = []

    def get(self, domain: int, lane: int) -> float:
        actual = domain + self.draws.branch * 65536
        if domain in (0x10000000, 0x20000000) and self.draws.layout != "bound-compton-v3":
            raise TransportError("bound energy coordinates require bound-compton-v3")
        try:
            item = self.lookup[(actual, lane)]
        except KeyError as error:
            raise TransportError(
                f"missing physical coordinate domain={actual:#x}, lane={lane}"
            ) from error
        if item not in self.used:
            self.used.append(item)
        return item.value


def _scale(vector: Vector3, factor: float) -> Vector3:
    return cast(Vector3, tuple(factor * value for value in vector))


def _add(left: Vector3, right: Vector3) -> Vector3:
    return cast(Vector3, tuple(a + b for a, b in zip(left, right, strict=True)))


def _sub(left: Vector3, right: Vector3) -> Vector3:
    return cast(Vector3, tuple(a - b for a, b in zip(left, right, strict=True)))


def _rotate(incident: RayTangent, cosine: float, azimuth: float) -> RayTangent:
    auxiliary: Vector3 = (1.0, 0.0, 0.0) if abs(incident.direction[2]) > 0.9 else (0.0, 0.0, 1.0)
    tangent = cross(auxiliary, incident.direction)
    tangent = _scale(tangent, 1.0 / math.hypot(*tangent))
    bitangent = cross(incident.direction, tangent)
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    outgoing = _add(
        _scale(incident.direction, cosine),
        _scale(
            _add(_scale(tangent, math.cos(azimuth)), _scale(bitangent, math.sin(azimuth))), sine
        ),
    )
    outgoing = _scale(outgoing, 1.0 / math.hypot(*outgoing))
    angular_velocity = cross(incident.direction, incident.direction_dot)
    return RayTangent(
        incident.origin_mm, outgoing, incident.origin_dot, cross(angular_velocity, outgoing)
    )


def _coordinate(energy: float, nodes: tuple[float, ...]) -> tuple[int, float]:
    if not math.isfinite(energy) or energy <= 0 or energy < nodes[0] or energy > nodes[-1]:
        raise TransportError("physical geometry energy is outside coefficient support")
    if len(nodes) == 1:
        return 0, 0.0
    lower = 0
    while lower < len(nodes) - 2 and energy > nodes[lower + 1]:
        lower += 1
    return lower, (energy - nodes[lower]) / (nodes[lower + 1] - nodes[lower])


def _interpolate(values: tuple[float, ...], lower: int, fraction: float) -> float:
    return (
        values[lower]
        if len(values) == 1
        else values[lower] + fraction * (values[lower + 1] - values[lower])
    )


def _rates(
    energy: float,
    material: int,
    spec: TransportSpec,
    absorption: tuple[float, ...],
    scattering: tuple[float, ...],
    physical: PhysicalTransport | None,
) -> tuple[float, float, float, int, float]:
    nodes = spec.energy_nodes
    materials = len(absorption) // len(nodes)
    if (
        materials != spec.grid.materials
        or type(material) is not int
        or not 0 <= material < materials
        or len(absorption) != len(scattering)
        or len(absorption) % len(nodes)
    ):
        raise TransportError(
            "physical geometry coefficients require matching material-major energy tables"
        )
    if any(not math.isfinite(v) or v < 0 for v in (*absorption, *scattering)):
        raise TransportError("physical geometry coefficients must be finite and nonnegative")
    lower, fraction = _coordinate(energy, nodes)
    first = material * len(nodes)
    a = _interpolate(absorption[first : first + len(nodes)], lower, fraction)
    s = _interpolate(scattering[first : first + len(nodes)], lower, fraction)
    r = 0.0
    if physical is not None:
        if len(physical.scattering_tables) != materials or len(physical.rayleigh_mm_inverse) != len(
            scattering
        ):
            raise TransportError(
                "physical geometry atomic/coherent tables must match every material"
            )
        r = _interpolate(physical.rayleigh_mm_inverse[first : first + len(nodes)], lower, fraction)
        if r > s:
            raise TransportError("Rayleigh rate exceeds aggregate scattering")
        if physical.fluorescence and len(physical.fluorescence) != materials:
            raise TransportError("physical geometry fluorescence must match every material")
        if (
            spec.scattering_law == "bound-electron-compton"
            and len(physical.bound_compton) != materials
        ):
            raise TransportError("bound geometry requires complete shell data per material")
        if (
            physical.scattering_tables[material].momentum_transfer_inverse_angstrom[-1]
            < energy / HC_KEV_ANGSTROM
        ):
            raise TransportError("physical geometry atomic table does not cover incident energy")
        if (
            spec.scattering_law == "bound-electron-compton"
            and physical.bound_compton[material].atomic_number
            != physical.scattering_tables[material].atomic_number
        ):
            raise TransportError("bound shell and atomic scattering elements must match")
        if spec.scattering_law != "bound-electron-compton" and physical.bound_compton:
            raise TransportError("shell tables require the bound-electron-compton scattering law")
    elif spec.scattering_law == "bound-electron-compton":
        raise TransportError("bound geometry requires the declared physical shell model")
    if not math.isfinite(a + s) or a + s <= 0:
        raise TransportError("a conditional collision requires finite positive total rate")
    if spec.scattering_law == "bound-electron-compton":
        assert physical is not None
        if not 0.1 <= energy <= 5000.0 or physical.cutoff_kev < 0.1:
            raise TransportError(
                "bound geometry requires energy in0.1..5000keV and cutoff at least0.1keV"
            )
        if spec.compton_sampler != "legacy-rejection":
            raise TransportError("bound geometry requires its legacy rejection sampler")
    return a, s, r, lower, fraction


def continuing_radiative_collision(
    energy: float,
    material: int,
    spec: TransportSpec,
    absorption: tuple[float, ...],
    scattering: tuple[float, ...],
    physical: PhysicalTransport | None,
) -> bool:
    """Whether a reached collision has any nonzero detector continuation law.

    This predicate only removes identically absent terminal/bridge measures.
    The actual conditional collision and its addressed marks still execute.
    """
    a, s, r, lower, fraction = _rates(energy, material, spec, absorption, scattering, physical)
    cutoff = 0.0 if physical is None else physical.cutoff_kev
    if energy <= cutoff:
        return False
    if r > 0.0:
        return True
    if s > r:
        if spec.scattering_law != "bound-electron-compton":
            return True
        assert physical is not None
        if any(
            shell.occupancy > 0.0 and energy - shell.binding_kev > cutoff
            for shell in physical.bound_compton[material].shells
        ):
            return True
    if a > 0.0 and physical is not None and physical.fluorescence:
        return any(
            energy >= line.threshold_kev
            and line.energy_kev > cutoff
            and _interpolate(line.probabilities, lower, fraction) > 0.0
            for line in physical.fluorescence[material]
        )
    return False


def _atomic_value(table: AtomicScatteringTable, q: float) -> tuple[float, float]:
    nodes, values = table.momentum_transfer_inverse_angstrom, table.form_factor
    if q < 0 or q > nodes[-1]:
        raise TransportError("Rayleigh geometry momentum transfer is outside atomic table support")
    lower = 0
    while lower < len(nodes) - 2 and q > nodes[lower + 1]:
        lower += 1
    slope = (values[lower + 1] - values[lower]) / (nodes[lower + 1] - nodes[lower])
    return values[lower] + slope * (q - nodes[lower]), slope


def _kn(energy: float, cosine: float) -> tuple[float, float, float]:
    alpha = energy / ELECTRON_REST_ENERGY_KEV
    ratio = 1.0 / (1.0 + alpha * (1.0 - cosine))
    shape = ratio**3 + ratio - ratio * ratio * (1 - cosine * cosine)
    slope = (alpha * ratio) * ratio * (
        3 * ratio * ratio + 1 - 2 * ratio * (1 - cosine * cosine)
    ) + 2 * cosine * ratio * ratio
    return shape, slope, ratio


@lru_cache(maxsize=128)
def _kn_normalisation(energy: float) -> float:
    # Same analytic total and small-energy expansion as the canonical phase
    # law. The inverse-alpha form also avoids squaring a very large alpha.
    alpha = energy / ELECTRON_REST_ENERGY_KEV
    if alpha > 0.05:
        inverse = 1 / alpha
        logarithm = math.log1p(2 * alpha)
        normal = (
            2
            * inverse
            * (
                (1 + inverse) * (2 * (1 + inverse) / (2 + inverse) - logarithm * inverse)
                + logarithm / 2
                - (inverse + 3) / (inverse + 2) ** 2
            )
        )
    else:
        coefficients = (
            2.6666666666666665,
            -5.333333333333333,
            13.866666666666667,
            -35.46666666666667,
            87.16190476190476,
            -207.23809523809524,
            480.5079365079365,
            -1092.9777777777779,
            2449.3252525252524,
            -5424.09696969697,
            11896.540792540793,
            -25885.069597069596,
            55945.658608058606,
            -120227.35238095238,
            257100.29803921568,
            -547445.1241830066,
            1161297.018231854,
            -2455262.1660818714,
            5175602.943358396,
            -10880905.198268399,
        )
        normal = coefficients[-1]
        for coefficient in reversed(coefficients[:-1]):
            normal = coefficient + alpha * normal
    if not math.isfinite(normal) or normal <= 0:
        raise TransportError("Klein-Nishina angular normalisation is not finite and positive")
    return 2 * math.pi * normal


@lru_cache(maxsize=128)
def _rayleigh_normalisation(energy: float, table: AtomicScatteringTable) -> float:
    # Four-point Gauss integrates the degree-seven polynomial on every linear
    # form-factor interval exactly, including its physical q-to-c Jacobian.
    qmax = energy / HC_KEV_ANGSTROM
    if table.momentum_transfer_inverse_angstrom[-1] < qmax:
        raise TransportError("Rayleigh atomic table does not cover incident energy")
    terms: list[float] = []
    gauss_nodes = (-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526)
    gauss_weights = (
        0.34785484513745385,
        0.6521451548625461,
        0.6521451548625461,
        0.34785484513745385,
    )
    nodes = table.momentum_transfer_inverse_angstrom
    for a, b in pairwise(nodes):
        b = min(b, qmax)
        if a >= b:
            continue
        midpoint, radius = (a + b) / 2, (b - a) / 2
        for node, weight in zip(gauss_nodes, gauss_weights, strict=True):
            q = midpoint + radius * node
            f = _atomic_value(table, q)[0]
            c = 1 - 2 * (q / qmax) ** 2
            terms.append(radius * weight * q * f * f * (1 + c * c))
    return 2 * math.pi * 4 / qmax**2 * math.fsum(terms)


def _bound_energy(
    energy: float, cosine: float, table: BoundComptonTable, reader: _DrawReader
) -> tuple[float, int, float]:
    return bound_compton_conditional_energy(
        energy, cosine, table, reader.get(0x10000000, 0), reader.get(0x10000000, 1)
    )


def conditional_physical_event(
    incident: RayTangent,
    energy_kev: float,
    material: int,
    spec: TransportSpec,
    absorption: tuple[float, ...],
    scattering: tuple[float, ...],
    physical: PhysicalTransport | None,
    draws: PhysicalEventDraws,
    *,
    density_factor: float = 1.0,
) -> PhysicalCollisionJet:
    """Apply one addressed analogue collision law with a rigid-frame jet.

    Coefficients are host tuples in material-major, energy-fastest order, in
    mm^-1. ``density_factor`` is a separately supplied positive dimensionless
    material/voxel multiplier. It scales the total rate and cancels from channel
    probabilities. A forced-interface continuation can therefore evaluate the
    two actual material laws with the same supplied coordinates.

    Incoming energy, tabulated physical laws and all uniforms are fixed under
    the geometry coordinate. Direction and origin tangents are retained in the
    continuing ray. Fluorescence follows the declared single-vacancy radiative
    branches with the complement deposited locally. Budget exhaustion is an
    explicit invalid status. This helper requires the analogue estimator.

    Example of one supplied free-electron collision coordinate::

        from dpt.transport.model import MaterialGrid, PlanarDetector
        grid = MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 1), 1)
        detector = PlanarDetector((-2, -2), (1, 1), (4, 4), 3)
        spec = TransportSpec(grid, detector, 80, "Supplied mm^-1 coefficients",
            (0.1, 200), scattering_law="free-electron-compton", max_angle_trials=1)
        incident = RayTangent((0, 0, 0), (0, 0, 1), direction_dot=(0.3, 0, 0))
        draws = PhysicalEventDraws(0, 0, "legacy-v1", (
            AddressedUniform(0, 0, 1, 0.8),
            AddressedUniform(0, 0x80000000, 0, 0.65),
            AddressedUniform(0, 0x80000000, 1, 0.001),
            AddressedUniform(0, 0x80000000, 2, 0.43)), "Supplied conditional coordinates")
        event = conditional_physical_event(incident, 80, 0, spec,
            (0.2, 0.2), (0.8, 0.8), None, draws)
        assert event.status == "continuing"
        assert event.energy_kev + event.deposited_kev == 80

    The example uses ``MaterialGrid`` and ``PlanarDetector`` from
    ``dpt.transport.model`` and the records in this module. All inputs and
    outputs are host values and no Warp or Torch tape is constructed.

    """
    if spec.estimator != "analogue":
        raise TransportError("conditional physical geometry events require analogue transport")
    if not math.isfinite(density_factor) or density_factor <= 0:
        raise TransportError("conditional physical density factor must be finite and positive")
    if not math.isfinite(energy_kev) or energy_kev <= 0:
        raise TransportError("conditional physical energy must be finite and positive")
    if spec.max_angle_trials > 65536:
        raise TransportError("physical event angular budget exceeds its65536-lane branch domain")
    if spec.scattering_law == "bound-electron-compton" and draws.layout != "bound-compton-v3":
        raise TransportError("bound geometry requires bound-compton-v3 coordinates")
    reader = _DrawReader(draws)
    if draws.event >= spec.max_events:
        raise TransportError("physical event number exceeds the declared event budget")
    if spec.scattering_law == "free-electron-compton" and (
        (spec.compton_sampler == "packed-uniform-rejection")
        != (draws.layout == "packed-compton-v2")
    ):
        raise TransportError("physical Compton draw layout does not match the declared sampler")
    if physical is not None and energy_kev < physical.cutoff_kev:
        return PhysicalCollisionJet(
            "cutoff", "local-deposition", None, 0.0, energy_kev, 0.0, (), draws.provenance
        )
    a, s, r, lower, fraction = _rates(energy_kev, material, spec, absorption, scattering, physical)
    total = density_factor * (a + s)
    if not math.isfinite(total) or total <= 0:
        raise TransportError(
            "physical collision density-scaled total rate is not finite and positive"
        )
    if reader.get(0, 1) < a / (a + s):
        if physical is None:
            return PhysicalCollisionJet(
                "absorption",
                "absorbed",
                None,
                0.0,
                energy_kev,
                total,
                tuple(reader.used),
                draws.provenance,
            )
        branch = reader.get(2, 0)
        lines = () if not physical.fluorescence else physical.fluorescence[material]
        if any(len(line.probabilities) != len(spec.energy_nodes) for line in lines):
            raise TransportError("fluorescence geometry probabilities must match energy nodes")
        eligible_sum = math.fsum(
            _interpolate(line.probabilities, lower, fraction)
            for line in lines
            if energy_kev >= line.threshold_kev
        )
        if eligible_sum > 1.0:
            raise TransportError("eligible fluorescent branch probabilities exceed one")
        cumulative = 0.0
        for index, line in enumerate(lines):
            if energy_kev >= line.threshold_kev:
                cumulative += _interpolate(line.probabilities, lower, fraction)
                if branch < cumulative:
                    cosine, azimuth = 2 * reader.get(2, 1) - 1, 2 * math.pi * reader.get(2, 2)
                    return PhysicalCollisionJet(
                        "fluorescence",
                        "continuing",
                        _rotate(incident, cosine, azimuth),
                        line.energy_kev,
                        energy_kev - line.energy_kev,
                        total,
                        tuple(reader.used),
                        draws.provenance,
                        line_index=index,
                    )
        return PhysicalCollisionJet(
            "absorption",
            "absorbed",
            None,
            0.0,
            energy_kev,
            total,
            tuple(reader.used),
            draws.provenance,
        )
    coherent = physical is not None and reader.get(3, 0) < r / s
    channels: dict[str, Channel] = {
        "isotropic-elastic": "isotropic",
        "free-electron-compton": "free-compton",
        "bound-electron-compton": "bound-compton",
    }
    channel: Channel = "rayleigh" if coherent else channels[spec.scattering_law]
    outgoing, shell_index, momentum = energy_kev, None, None
    if channel == "isotropic":
        cosine, azimuth = 2 * reader.get(0, 2) - 1, 2 * math.pi * reader.get(0, 3)
    else:
        table = None if physical is None else physical.scattering_tables[material]
        shell_table = (
            None
            if channel != "bound-compton" or physical is None
            else physical.bound_compton[material]
        )
        if channel == "bound-compton" and draws.layout != "bound-compton-v3":
            raise TransportError("bound geometry requires its declared coordinates")
        if shell_table is not None and not any(
            shell.occupancy > 0.0 and energy_kev > shell.binding_kev for shell in shell_table.shells
        ):
            raise TransportError(
                "bound geometry has no accessible shell envelope at the reached energy"
            )
        accepted: tuple[float, float] | None = None
        for trial in range(spec.max_angle_trials):
            domain = (0xC0000000 if coherent else 0x80000000) + trial
            if channel == "free-compton" and draws.layout == "packed-compton-v2":
                domain = 0x80000000 + trial // 2
                polar, reject = (
                    reader.get(domain, 2 * (trial % 2)),
                    reader.get(domain, 2 * (trial % 2) + 1),
                )
                azimuth = 2 * math.pi * reader.get(0, 2)
            else:
                polar, reject = reader.get(domain, 0), reader.get(domain, 1)
                azimuth = 2 * math.pi * reader.get(domain, 2)
            cosine = 2 * polar - 1
            if coherent:
                assert table is not None
                q = energy_kev / HC_KEV_ANGSTROM * math.sqrt((1 - cosine) / 2)
                acceptance = (
                    0.5
                    * (1 + cosine * cosine)
                    * (_atomic_value(table, q)[0] / table.atomic_number) ** 2
                )
            elif shell_table is not None:
                cosine, acceptance = bound_compton_angle_proposal(energy_kev, shell_table, polar)
            else:
                acceptance = 0.5 * _kn(energy_kev, cosine)[0]
            if reject < acceptance:
                accepted = (cosine, azimuth)
                if shell_table is not None:
                    outgoing, shell_index, momentum = _bound_energy(
                        energy_kev, cosine, shell_table, reader
                    )
                elif not coherent:
                    outgoing = energy_kev * _kn(energy_kev, cosine)[2]
                break
        if accepted is None:
            return PhysicalCollisionJet(
                channel,
                "angle-budget",
                None,
                energy_kev,
                0.0,
                total,
                tuple(reader.used),
                draws.provenance,
            )
        cosine, azimuth = accepted
    return PhysicalCollisionJet(
        channel,
        "continuing",
        _rotate(incident, cosine, azimuth),
        outgoing,
        energy_kev - outgoing,
        total,
        tuple(reader.used),
        draws.provenance,
        shell_index=shell_index,
        momentum=momentum,
    )


def terminal_physical_kernels(
    incident: RayTangent,
    endpoint_mm: Vector3,
    energy_kev: float,
    material: int,
    spec: TransportSpec,
    absorption: tuple[float, ...],
    scattering: tuple[float, ...],
    physical: PhysicalTransport | None,
    *,
    endpoint_dot: Vector3 = (0.0, 0.0, 0.0),
    bound_outgoing_kev: float | None = None,
) -> tuple[PhysicalTerminalKernel, ...]:
    """Evaluate the complete mixed collision kernel at a physical endpoint.

    Endpoint and ray origins are world mm vectors. Their jets determine the
    connecting direction and cosine jet. This returns every eligible radiative
    branch and both coherent and noncoherent scattering contributions, weighted
    by their actual conditional collision probabilities. Absorbed and locally
    deposited energy contributes no outgoing-photon kernel.

    Bound Compton requires an explicit fixed outgoing-energy integration mark
    in keV and evaluates the selected model's joint sr^-1 keV^-1 kernel. Its
    mark-proposal density must be divided out by the caller. No stationary
    Compton energy or angular-only replacement is inferred.

    Example of a discrete-energy terminal kernel::

        from dpt.transport.model import MaterialGrid, PlanarDetector
        spec = TransportSpec(MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 1), 1),
            PlanarDetector((-2, -2), (1, 1), (4, 4), 3), 80,
            "Supplied mm^-1 coefficients", (0.1, 200),
            scattering_law="free-electron-compton")
        incident = RayTangent((0, 0, 0), (0, 0, 1))
        kernel, = terminal_physical_kernels(incident, (1, 0, 3), 80, 0,
            spec, (0.2, 0.2), (0.8, 0.8), None, endpoint_dot=(0.2, 0, 0))
        assert kernel.channel == "free-compton"
        assert kernel.energy_kev < 80

    """
    if spec.estimator != "analogue":
        raise TransportError("mixed terminal physical geometry requires analogue transport")
    if physical is not None and energy_kev < physical.cutoff_kev:
        return ()
    a, s, r, lower, fraction = _rates(energy_kev, material, spec, absorption, scattering, physical)
    direction, direction_dot = normalised_direction_jet(
        _sub(endpoint_mm, incident.origin_mm), _sub(endpoint_dot, incident.origin_dot)
    )
    outgoing = RayTangent(incident.origin_mm, direction, incident.origin_dot, direction_dot)
    cosine = max(-1.0, min(1.0, dot(incident.direction, direction)))
    cosine_dot = dot(incident.direction_dot, direction) + dot(incident.direction, direction_dot)
    kernels: list[PhysicalTerminalKernel] = []

    def append(
        channel: Channel,
        energy: float,
        energy_dot: float,
        density: float,
        density_dot: float,
        measure: Literal["solid-angle", "solid-angle-energy"] = "solid-angle",
        line_index: int | None = None,
    ) -> None:
        if physical is not None and energy == physical.cutoff_kev and energy_dot != 0.0:
            raise TransportError("terminal energy is on a moving cutoff boundary")
        if physical is None or energy >= physical.cutoff_kev:
            kernels.append(
                PhysicalTerminalKernel(
                    channel,
                    outgoing,
                    cosine,
                    cosine_dot,
                    energy,
                    energy_dot,
                    density,
                    density_dot,
                    measure,
                    line_index,
                )
            )

    total = a + s
    if r > 0:
        assert physical is not None
        table = physical.scattering_tables[material]
        q = energy_kev / HC_KEV_ANGSTROM * math.sqrt((1 - cosine) / 2)
        f, slope = _atomic_value(table, q)
        if (
            q == 0.0
            and slope != 0.0
            and math.hypot(*_sub(direction_dot, incident.direction_dot)) != 0.0
        ):
            raise TransportError("Rayleigh geometry derivative is on its forward angular cusp")
        if cosine_dot != 0 and (q == 0 or q in table.momentum_transfer_inverse_angstrom[1:-1]):
            raise TransportError(
                "Rayleigh geometry derivative is on an angular interpolation boundary"
            )
        q_dot = (
            0.0
            if cosine_dot == 0
            else -energy_kev / (4 * HC_KEV_ANGSTROM * math.sqrt((1 - cosine) / 2)) * cosine_dot
        )
        normalisation = _rayleigh_normalisation(energy_kev, table)
        density = r / total * (1 + cosine * cosine) * f * f / normalisation
        derivative = (
            r
            / total
            * (2 * cosine * cosine_dot * f * f + 2 * (1 + cosine * cosine) * f * slope * q_dot)
            / normalisation
        )
        append("rayleigh", energy_kev, 0.0, density, derivative)
    if s > r:
        probability = (s - r) / total
        if spec.scattering_law == "isotropic-elastic":
            append("isotropic", energy_kev, 0.0, probability / (4 * math.pi), 0.0)
        elif spec.scattering_law == "free-electron-compton":
            shape, slope, ratio = _kn(energy_kev, cosine)
            outgoing_energy = energy_kev * ratio
            energy_dot = (
                0.0
                if cosine_dot == 0
                else (outgoing_energy / ELECTRON_REST_ENERGY_KEV) * outgoing_energy * cosine_dot
            )
            normalisation = _kn_normalisation(energy_kev)
            append(
                "free-compton",
                outgoing_energy,
                energy_dot,
                probability * shape / normalisation,
                0.0 if cosine_dot == 0 else probability * slope / normalisation * cosine_dot,
            )
        else:
            if bound_outgoing_kev is None:
                raise TransportError(
                    "bound terminal geometry requires an explicit joint outgoing-energy mark"
                )
            assert physical is not None
            table_bound = physical.bound_compton[material]
            if not math.isfinite(bound_outgoing_kev) or bound_outgoing_kev <= 0:
                raise TransportError(
                    "bound terminal outgoing-energy mark must be finite and positive"
                )
            if cosine_dot != 0.0 and any(
                bound_outgoing_kev == energy_kev - shell.binding_kev for shell in table_bound.shells
            ):
                raise TransportError("bound terminal mark is on a shell support boundary")
            density = probability * bound_compton_joint_pdf(
                energy_kev,
                cosine,
                bound_outgoing_kev,
                table_bound,
            )
            derivative = 0.0
            if density > 0 and cosine_dot != 0:
                derivative = (
                    density
                    * bound_compton_joint_scores(
                        energy_kev, cosine, bound_outgoing_kev, table_bound
                    )[0]
                    * cosine_dot
                )
            append(
                "bound-compton", bound_outgoing_kev, 0.0, density, derivative, "solid-angle-energy"
            )
    if physical is not None and physical.fluorescence:
        eligible_sum = 0.0
        for index, line in enumerate(physical.fluorescence[material]):
            if len(line.probabilities) != len(spec.energy_nodes):
                raise TransportError("fluorescence geometry probabilities must match energy nodes")
            if energy_kev >= line.threshold_kev:
                probability = _interpolate(line.probabilities, lower, fraction)
                eligible_sum += probability
                if probability > 0:
                    append(
                        "fluorescence",
                        line.energy_kev,
                        0.0,
                        a / total * probability / (4 * math.pi),
                        0.0,
                        line_index=index,
                    )
        if eligible_sum > 1:
            raise TransportError("eligible fluorescent branch probabilities exceed one")
    return tuple(kernels)
