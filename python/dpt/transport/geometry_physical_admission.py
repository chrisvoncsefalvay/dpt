"""Reachable energy/cell support of complete analogue photon geometry laws.

Free Compton scattering has open outgoing support ``(F(E),E)``, where
``F(E)=m*E/(m+2*E)`` and ``m`` is the electron rest energy in keV. Inside a
positive noncoherent affine coefficient edge ``(a,b)``, finite sequences of
arbitrarily small deflections can descend towards ``a`` and rotate with an
arbitrarily small additional inverse-energy loss. If incident reachability
extends up to ``h``, a final collision gives the open outgoing interval
``(F(a),min(h,b))``, clipped by cutoff. This edge closure retains real rate-zero
barriers and avoids replacing the physical law by an event limit.

Elastic, fluorescent and bound-electron channels retain their declared energy
supports. The exact source chart and rational interpolation primitives are
shared with the sealed bound admission helper. This module certifies ideal
physical support, while numerical profile conditioning and actual geometry
estimator/confidence qualification remain separate.
"""

from __future__ import annotations

# Shared exact host admission primitives retain the existing geometry/table law.
# pyright: reportPrivateUsage=false
# Runtime admission also validates untyped callers.
# pyright: reportUnnecessaryIsInstance=false
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from .geometry_bound_admission import _f, _interpolate, _positive_segments, _source_cells
from .model import ELECTRON_REST_ENERGY_KEV, TransportError, TransportSpec
from .physical import HC_KEV_ANGSTROM
from .source import ParallelBeam, PointConeSource

if TYPE_CHECKING:
    from .geometry_histories import GeometryPhysicalMedium

EnergyPair = tuple[Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class GeometryReachableMark:
    """An attained discrete incident energy in keV and its cell support."""

    energy_kev: float
    cells: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GeometryReachableInterval:
    """An open incident-energy interval in keV and an explicit cell family.

    Both endpoints must be exact ``Fraction`` values and ``cells`` are unique
    flattened x-fastest indices. Neither endpoint is an attained mark. An
    auxiliary interval describes an unrestricted cell/energy family because it
    supplies no position/direction chart; its spatial closure may conservatively
    reject a valid particular outward boundary ray.
    """

    lower_kev: Fraction
    upper_kev: Fraction
    cells: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cells", tuple(self.cells))
        if not isinstance(self.lower_kev, Fraction) or not isinstance(self.upper_kev, Fraction):
            raise TransportError("open energy family endpoints must be exact Fractions")
        if not 0 <= self.lower_kev < self.upper_kev:
            raise TransportError("open energy family must have 0<=lower<upper")
        if any(type(cell) is not int or cell < 0 for cell in self.cells) or len(
            set(self.cells)
        ) != len(self.cells):
            raise TransportError("open energy family requires unique nonnegative integer cells")


@dataclass(frozen=True, slots=True)
class GeometryPhysicalAdmission:
    """Immutable complete-law support certificate for fixed host input records.

    ``source_cells`` retain exact half-open source-flight ownership. The derived
    ``marks`` and ``intervals`` describe continuing photons only, so energies at
    or below cutoff do not activate a channel. Caller-supplied auxiliary marks
    and intervals are retained separately from the derived closure. They are
    unrestricted/conservative cell-energy families, not an automatic proof of
    every active geometry future or a particular forced ray's angular reachability.
    ``bound_profile_conditioning_certified`` is false for the bound law and
    ``None`` for other laws, which do not use bound profile conditioning.

    Physical closure uses finite histories of unrestricted order. Execution
    guards still invalidate incomplete launched histories and are not widened
    by this host calculation. No arrays are uploaded and no native work occurs.
    """

    source_cells: tuple[int, ...]
    source_energies_kev: tuple[float, ...]
    auxiliary_marks: tuple[tuple[int, float], ...]
    auxiliary_intervals: tuple[GeometryReachableInterval, ...]
    marks: tuple[GeometryReachableMark, ...]
    intervals: tuple[GeometryReachableInterval, ...]
    scope: Literal["source-physical-law", "source-and-conservative-auxiliary-families"]
    bound_profile_conditioning_certified: Literal[False] | None


def _free_minimum(energy: Fraction) -> Fraction:
    mass = _f(ELECTRON_REST_ENERGY_KEV)
    return mass * energy / (mass + 2 * energy)


def _represented_open_maximum(low: Fraction, high: Fraction) -> float | None:
    """Largest binary64 energy strictly inside an exact finite open interval.

    Conversion rounds to nearest, so step down precisely when it reaches or
    exceeds the excluded upper endpoint. Positive binary64 division by ``hc``
    is monotone and its maximum is therefore attained at this represented mark.
    This calculation introduces neither an endpoint atom nor a tolerance.
    """
    candidate = float(high)
    if _f(candidate) >= high:
        candidate = math.nextafter(candidate, -math.inf)
    return candidate if _f(candidate) > low else None


def _validate(spec: TransportSpec, medium: GeometryPhysicalMedium) -> None:
    count, materials, width = (
        math.prod(spec.grid.shape),
        spec.grid.materials,
        len(spec.energy_nodes),
    )
    if len(medium.material_ids) != count or any(
        type(m) is not int or not 0 <= m < materials for m in medium.material_ids
    ):
        raise TransportError("physical admission requires one valid material ID per cell")
    if len(medium.density) != materials or len(medium.voxel_density) != count:
        raise TransportError("physical admission density ownership does not match the grid")
    if any(not math.isfinite(v) or v <= 0 for v in medium.density) or any(
        not math.isfinite(v) or v < 0 for v in medium.voxel_density
    ):
        raise TransportError(
            "physical admission requires positive material and nonnegative voxel densities"
        )
    arrays = (medium.absorption_mm_inverse, medium.scattering_mm_inverse)
    if any(len(values) != materials * width for values in arrays) or any(
        not math.isfinite(v) or v < 0 for values in arrays for v in values
    ):
        raise TransportError(
            "physical admission coefficients require finite nonnegative material-major tables"
        )
    for cell, material in enumerate(medium.material_ids):
        factor = medium.density[material] * medium.voxel_density[cell]
        if not math.isfinite(factor) or (medium.voxel_density[cell] > 0 and factor <= 0):
            raise TransportError("physical admission requires representable cell density factors")
        for index in range(width):
            rate = (
                medium.absorption_mm_inverse[material * width + index]
                + medium.scattering_mm_inverse[material * width + index]
            )
            scaled = factor * rate
            if (
                not math.isfinite(rate)
                or not math.isfinite(scaled)
                or (factor > 0 and rate > 0 and scaled == 0)
            ):
                raise TransportError("physical admission requires representable cell attenuation")
    physical = medium.physical
    bound = spec.scattering_law == "bound-electron-compton"
    if physical is None:
        if bound:
            raise TransportError("bound geometry requires explicit shell and angular tables")
        return
    if (
        len(physical.scattering_tables) != materials
        or len(physical.rayleigh_mm_inverse) != materials * width
    ):
        raise TransportError("physical admission requires complete atomic/Rayleigh material tables")
    if any(
        r > s
        for r, s in zip(physical.rayleigh_mm_inverse, medium.scattering_mm_inverse, strict=True)
    ):
        raise TransportError("Rayleigh partial exceeds aggregate scattering")
    if bound:
        if (
            len(physical.bound_compton) != materials
            or physical.cutoff_kev < 0.1
            or spec.compton_sampler != "legacy-rejection"
        ):
            raise TransportError(
                "bound admission requires complete shells, cutoff>=0.1 and the legacy sampler"
            )
        if any(
            a.atomic_number != b.atomic_number
            for a, b in zip(physical.scattering_tables, physical.bound_compton, strict=True)
        ):
            raise TransportError("bound and atomic table elements disagree")
    elif physical.bound_compton:
        raise TransportError("shell tables require the bound-electron-Compton law")
    if physical.fluorescence and len(physical.fluorescence) != materials:
        raise TransportError("fluorescence ownership does not match the materials")
    for lines in physical.fluorescence:
        for line in lines:
            if len(line.probabilities) != width:
                raise TransportError("fluorescence probabilities must match coefficient nodes")
            if any(
                e < line.threshold_kev and p != 0
                for e, p in zip(spec.energy_nodes, line.probabilities, strict=True)
            ):
                raise TransportError("fluorescence probability is nonzero below threshold")
        if any(
            math.fsum(line.probabilities[index] for line in lines) > 1 for index in range(width)
        ):
            raise TransportError("fluorescence branch probabilities exceed one")


def _merge(pairs: list[EnergyPair]) -> list[EnergyPair]:
    result: list[EnergyPair] = []
    for low, high in sorted(pairs):
        if result and low < result[-1][1]:
            previous_low, previous_high = result[-1]
            result[-1] = (previous_low, max(previous_high, high))
        else:
            result.append((low, high))
    return result


def admit_geometry_physical_law(
    spec: TransportSpec,
    medium: GeometryPhysicalMedium,
    source: ParallelBeam | PointConeSource,
    *,
    auxiliary_marks: tuple[tuple[int, float], ...] = (),
    auxiliary_intervals: tuple[GeometryReachableInterval, ...] = (),
) -> GeometryPhysicalAdmission:
    """Certify reachable coefficient and conditional-law support for every model.

    Material-major coefficient tuples have shape ``(materials*energy_nodes,)``
    in mm inverse and cell arrays have shape ``(nz*ny*nx,)``. Host source/pose
    records define exact source-cell support through the shared affine chart.
    Coefficients, atomic laws and source marks stay fixed during this calculation
    and the return is neither a tape gradient nor a moving-support estimator.

    Free scattering on a reached positive affine edge ``(a,b)`` emits an open
    interval down to ``F(a)=m*a/(m+2*a)``, clipped by the deposition cutoff.
    A source atom with zero noncoherent rate cannot seed that edge. A rate-zero
    barrier above the first coefficient node is valid when its final outgoing
    minimum stays within support, even if cutoff is below that node. Fluorescent
    line atoms remain distinct from open continuum endpoints. Missing material
    and purely absorbing paths do not create lower-energy obligations.

    A supplied physical record also binds the existing collision dispatch's
    atomic-table coverage policy at every reached positive-total collision
    state, including absorption-only states. Attained energies use the actual
    binary64 ``E/hc`` comparison. Open positive-total edges use the largest
    represented energy strictly within the exact interval and monotonicity of
    division. Vacuum and exact zero-total states make no collision-law demand.

    Optional auxiliary marks/families describe unrestricted cell-energy support
    and retain their declarations in the result. Their conservative closure may
    overreject a ray-constrained outward future, and the caller must establish
    completeness of actual active geometry auxiliary support. Bound profile numerical
    conditioning, native execution and confidence/gradient composition remain
    separate obligations. No finite event guard truncates this physical law.

    With existing ``spec``, ``medium`` and ``source`` records::

        certificate = admit_geometry_physical_law(spec, medium, source)
        assert certificate.scope == "source-physical-law"

    Compton reference: https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/compton.html
    """
    _validate(spec, medium)
    physical = medium.physical
    cutoff = _f(0.0 if physical is None else physical.cutoff_kev)
    nodes = tuple(_f(v) for v in spec.energy_nodes)
    width, materials = len(nodes), spec.grid.materials
    bound = spec.scattering_law == "bound-electron-compton"
    cells = tuple(c for c, voxel in enumerate(medium.voxel_density) if voxel > 0)
    source_cells = _source_cells(spec, medium, source)
    source_energies = (
        physical.spectrum.energy_kev
        if physical is not None and physical.spectrum is not None
        else (spec.energy_kev,)
    )
    if bound and any(not 0.1 <= energy <= 5000 for energy in source_energies):
        raise TransportError("bound source energies must remain in 0.1..5000 keV")
    marks: dict[Fraction, set[int]] = {}
    families: dict[tuple[int, ...], list[EnergyPair]] = {}

    def add_mark(energy: Fraction, reached: tuple[int, ...]) -> None:
        if energy > cutoff and reached:
            marks.setdefault(energy, set()).update(reached)

    def add_interval(low: Fraction, high: Fraction, reached: tuple[int, ...]) -> None:
        low = max(low, cutoff)
        if low < high and reached:
            key = tuple(sorted(set(reached)))
            families[key] = _merge([*families.get(key, ()), (low, high)])

    for energy in source_energies:
        add_mark(_f(energy), source_cells)
    for cell, energy in auxiliary_marks:
        if (
            type(cell) is not int
            or not 0 <= cell < len(medium.material_ids)
            or not math.isfinite(energy)
            or energy <= 0
        ):
            raise TransportError(
                "auxiliary physical marks require valid cells and positive finite energies"
            )
        if bound and not 0.1 <= energy <= 5000:
            raise TransportError("auxiliary bound marks must remain in 0.1..5000 keV")
        if medium.voxel_density[cell] > 0:
            add_mark(_f(energy), (cell,))
    for interval in auxiliary_intervals:
        if not isinstance(interval, GeometryReachableInterval) or any(
            cell >= len(medium.material_ids) for cell in interval.cells
        ):
            raise TransportError(
                "auxiliary physical intervals require explicit valid cell families"
            )
        reached = tuple(cell for cell in interval.cells if medium.voxel_density[cell] > 0)
        add_interval(interval.lower_kev, interval.upper_kev, reached)
    absorption = tuple(
        tuple(_f(v) for v in medium.absorption_mm_inverse[m * width : (m + 1) * width])
        for m in range(materials)
    )
    rayleigh_values = (
        (0.0,) * (materials * width) if physical is None else physical.rayleigh_mm_inverse
    )
    rayleigh = tuple(
        tuple(_f(v) for v in rayleigh_values[m * width : (m + 1) * width]) for m in range(materials)
    )
    noncoherent = tuple(
        tuple(
            _f(s) - _f(r)
            for s, r in zip(
                medium.scattering_mm_inverse[m * width : (m + 1) * width],
                rayleigh_values[m * width : (m + 1) * width],
                strict=True,
            )
        )
        for m in range(materials)
    )
    positive = tuple(_positive_segments(nodes, values) for values in noncoherent)
    binding = (
        tuple(
            min(_f(shell.binding_kev) for shell in table.shells) for table in physical.bound_compton
        )
        if physical is not None and bound
        else ()
    )

    def check_energy(energy: Fraction) -> None:
        if not nodes[0] <= energy <= nodes[-1]:
            raise TransportError("reachable photon energy is outside supplied coefficient support")
        if bound and not _f(0.1) <= energy <= 5000:
            raise TransportError("reachable bound incident energy is outside its declared domain")

    def check_angular(material: int, low: Fraction, high: Fraction) -> None:
        if physical is None:
            return
        qmax = physical.scattering_tables[material].momentum_transfer_inverse_angstrom[-1]
        total = tuple(
            a + b + r
            for a, b, r in zip(
                absorption[material], noncoherent[material], rayleigh[material], strict=True
            )
        )
        if low == high:
            candidates = (float(low),) if _interpolate(nodes, total, low) > 0 else ()
        else:
            candidates = tuple(
                energy
                for a, b in _positive_segments(nodes, total)
                if max(low, a) < min(high, b)
                and (energy := _represented_open_maximum(max(low, a), min(high, b))) is not None
            )
        if any(energy / HC_KEV_ANGSTROM > qmax for energy in candidates):
            raise TransportError("reachable physical collision exceeds its angular table support")

    previous = None
    while True:
        signature = (
            tuple((energy, tuple(sorted(reached))) for energy, reached in sorted(marks.items())),
            tuple((reached, tuple(pairs)) for reached, pairs in sorted(families.items())),
        )
        if signature == previous:
            break
        previous = signature
        for energy, reached in tuple(marks.items()):
            check_energy(energy)
            for material in {medium.material_ids[cell] for cell in tuple(reached)}:
                a = _interpolate(nodes, absorption[material], energy)
                b = _interpolate(nodes, noncoherent[material], energy)
                r = _interpolate(nodes, rayleigh[material], energy)
                check_angular(material, energy, energy)
                if r > 0:
                    reached.update(cells)
                if b > 0:
                    if bound:
                        if energy <= binding[material]:
                            raise TransportError(
                                "reachable positive bound rate has zero accessible-shell normaliser"
                            )
                        add_interval(cutoff, energy - binding[material], cells)
                    elif spec.scattering_law == "isotropic-elastic":
                        reached.update(cells)
                    else:
                        for low, high in positive[material]:
                            if low <= energy <= high:
                                add_interval(_free_minimum(max(low, cutoff)), energy, cells)
                if a > 0 and physical is not None and physical.fluorescence:
                    for line in physical.fluorescence[material]:
                        if (
                            energy >= _f(line.threshold_kev)
                            and _interpolate(
                                nodes, tuple(_f(p) for p in line.probabilities), energy
                            )
                            > 0
                        ):
                            add_mark(_f(line.energy_kev), cells)
        for reached, pairs in tuple(families.items()):
            for low, high in tuple(pairs):
                if low < nodes[0] or high > nodes[-1]:
                    raise TransportError(
                        "reachable open photon-energy family exceeds supplied coefficient support"
                    )
                if bound and (low < _f(0.1) or high > 5000):
                    raise TransportError(
                        "reachable open bound-energy family exceeds its declared domain"
                    )
                for material in {medium.material_ids[cell] for cell in reached}:
                    check_angular(material, low, high)
                    for left, right in _positive_segments(nodes, rayleigh[material]):
                        add_interval(max(low, left), min(high, right), cells)
                    for left, right in positive[material]:
                        incident_low, incident_high = max(low, left), min(high, right)
                        if incident_low >= incident_high:
                            continue
                        if bound:
                            if incident_low < min(incident_high, binding[material]):
                                raise TransportError(
                                    "reachable positive bound-rate interval has inaccessible shells"
                                )
                            add_interval(cutoff, incident_high - binding[material], cells)
                        elif spec.scattering_law == "isotropic-elastic":
                            add_interval(incident_low, incident_high, cells)
                        else:
                            add_interval(_free_minimum(max(left, cutoff)), incident_high, cells)
                    if physical is not None and physical.fluorescence:
                        for line in physical.fluorescence[material]:
                            eligible = _positive_segments(
                                nodes,
                                absorption[material],
                                tuple(_f(p) for p in line.probabilities),
                            )
                            if any(
                                max(low, left, _f(line.threshold_kev)) < min(high, right)
                                for left, right in eligible
                            ):
                                add_mark(_f(line.energy_kev), cells)
    return GeometryPhysicalAdmission(
        source_cells,
        tuple(source_energies),
        tuple((cell, energy) for cell, energy in auxiliary_marks),
        tuple(auxiliary_intervals),
        tuple(
            GeometryReachableMark(float(energy), tuple(sorted(reached)))
            for energy, reached in sorted(marks.items())
        ),
        tuple(
            GeometryReachableInterval(low, high, reached)
            for reached, pairs in sorted(families.items())
            for low, high in pairs
        ),
        "source-and-conservative-auxiliary-families"
        if auxiliary_marks or auxiliary_intervals
        else "source-physical-law",
        False if bound else None,
    )
