"""Exact sufficient signatures of reachable measured photon generators.

The photon-count/energy transport generator is ``-mu*I + R*K_R + B*K_B,>L
+ sum_l A*p_l*K_l,>L``. Local deposition is not a response of this generator.
Represented coefficients are exact rational inputs to their declared affine
interpolation, and products are retained as polynomials rather than sampled.
This module certifies equality of these symbolic laws. Different signatures
mean that equality is uncertified, not that no accidental equality exists.
"""

from __future__ import annotations

# Exact admission primitives and runtime checking of untyped callers are shared.
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false
import math
from dataclasses import dataclass, fields
from fractions import Fraction
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

from .geometry_bound_admission import _f, _interpolate
from .geometry_physical_admission import (
    GeometryReachableInterval,
    GeometryReachableMark,
    _represented_open_maximum,
    _validate,
)
from .model import TransportError, TransportSpec
from .physical import HC_KEV_ANGSTROM, AtomicScatteringTable

if TYPE_CHECKING:
    from .geometry_histories import GeometryPhysicalMedium

Polynomial = tuple[Fraction, ...]
ZERO: Polynomial = (Fraction(0),)


@dataclass(frozen=True, slots=True)
class GeometryLawSupport:
    """An attained mark or strictly open family in keV, without endpoint atoms."""

    lower_kev: Fraction
    upper_kev: Fraction
    attained: bool


@dataclass(frozen=True, slots=True)
class GeometryRayleighPiece:
    """An affine ``F(q)/F(0)`` piece on an inverse-angstrom interval."""

    lower_q: Fraction
    upper_q: Fraction
    intercept: Fraction
    slope: Fraction


@dataclass(frozen=True, slots=True)
class GeometryBoundProfile:
    """Grouped accessible shell binding, height and relative occupancy.

    Height is the actual ``profile_zero/alpha`` in inverse dimensionless
    momentum, while binding is in keV. All accessible shells are retained when
    any outgoing shell support exceeds cutoff, including shells contributing
    only to the common normaliser and local deposition.
    """

    binding_kev: Fraction
    momentum_height: Fraction
    relative_occupancy: Fraction


@dataclass(frozen=True, slots=True)
class GeometryMeasuredGenerator:
    """Sufficient exact signature of a measured generator on one support piece.

    Rate polynomials are in ascending powers of incident energy in keV and
    evaluate to mm inverse. ``total_mm_inverse`` is the extinction sink and
    ``noncoherent_mm_inverse`` is retained only for a continuing noncoherent
    kernel. A terminal-only noncoherent rate can exchange with absorption when
    the total and all continuing kernels agree. Fluorescence entries group
    equal outgoing line energies and contain the full ``A*p`` rate polynomial.
    Angular/profile shapes are conditional-law signatures, independent of rate.
    """

    support: GeometryLawSupport
    total_mm_inverse: Polynomial
    noncoherent_mm_inverse: Polynomial
    noncoherent_law: str | None
    rayleigh_mm_inverse: Polynomial
    rayleigh_shape: tuple[GeometryRayleighPiece, ...]
    bound_profiles: tuple[GeometryBoundProfile, ...]
    fluorescence: tuple[tuple[Fraction, Polynomial], ...]


@dataclass(frozen=True, slots=True)
class GeometryGeneratorWitness:
    """A differing symbolic component on an actual compared support piece."""

    support: GeometryLawSupport
    component: str
    left: str
    right: str


@dataclass(frozen=True, slots=True)
class GeometryGeneratorComparison:
    """An exact sufficient equality certificate or an uncertified result.

    ``status='uncertified'`` does not prove inequality or nondifferentiability.
    In particular an isolated bound-energy atom can have an accidental ``B/N``
    compensation despite different terminal-shell profiles. Such identities
    are outside this structural certificate. Reachable support completeness,
    geometry face visibility and numerical profile conditioning are supplied
    and qualified by their respective callers.
    """

    status: Literal["equivalent", "uncertified"]
    left_cell: int | None
    right_cell: int | None
    reachable_marks: tuple[GeometryReachableMark, ...]
    reachable_intervals: tuple[GeometryReachableInterval, ...]
    supports: tuple[GeometryLawSupport, ...]
    witnesses: tuple[GeometryGeneratorWitness, ...]

    @property
    def equivalent(self) -> bool:
        """Whether every retained symbolic measured generator agrees exactly."""
        return self.status == "equivalent"


def _trim(values: Polynomial) -> Polynomial:
    while len(values) > 1 and values[-1] == 0:
        values = values[:-1]
    return values


def _sum(left: Polynomial, right: Polynomial) -> Polynomial:
    return _trim(
        tuple(
            (left[i] if i < len(left) else Fraction(0))
            + (right[i] if i < len(right) else Fraction(0))
            for i in range(max(len(left), len(right)))
        )
    )


def _scale(values: Polynomial, factor: Fraction) -> Polynomial:
    return _trim(tuple(value * factor for value in values))


def _product(left: Polynomial, right: Polynomial) -> Polynomial:
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, first in enumerate(left):
        for j, last in enumerate(right):
            result[i + j] += first * last
    return _trim(tuple(result))


def _polynomial(
    nodes: tuple[Fraction, ...], values: tuple[Fraction, ...], support: GeometryLawSupport
) -> Polynomial:
    if support.attained:
        return (_interpolate(nodes, values, support.lower_kev),)
    for index, (lower, upper) in enumerate(pairwise(nodes)):
        if lower <= support.lower_kev < support.upper_kev <= upper:
            slope = (values[index + 1] - values[index]) / (upper - lower)
            return _trim((values[index] - slope * lower, slope))
    raise TransportError("generator support must fit a supplied coefficient edge")


def _rayleigh_shape(
    table: AtomicScatteringTable, energy: float
) -> tuple[GeometryRayleighPiece, ...]:
    qmax = energy / HC_KEV_ANGSTROM
    if qmax > table.momentum_transfer_inverse_angstrom[-1]:
        raise TransportError("generator physical collision exceeds angular table support")
    end, anchor = _f(qmax), _f(table.form_factor[0])
    if end == 0:
        return (GeometryRayleighPiece(Fraction(0), Fraction(0), Fraction(1), Fraction(0)),)
    pieces: list[GeometryRayleighPiece] = []
    for index in range(len(table.form_factor) - 1):
        lower, upper = map(_f, table.momentum_transfer_inverse_angstrom[index : index + 2])
        upper = min(upper, end)
        if lower >= upper:
            break
        full_upper = _f(table.momentum_transfer_inverse_angstrom[index + 1])
        first, last = map(_f, table.form_factor[index : index + 2])
        slope = (last - first) / (full_upper - lower) / anchor
        intercept = first / anchor - slope * lower
        if pieces and (pieces[-1].intercept, pieces[-1].slope) == (intercept, slope):
            previous = pieces.pop()
            lower = previous.lower_q
        pieces.append(GeometryRayleighPiece(lower, upper, intercept, slope))
    return tuple(pieces)


def _signature(
    spec: TransportSpec,
    medium: GeometryPhysicalMedium,
    cell: int | None,
    support: GeometryLawSupport,
) -> GeometryMeasuredGenerator:
    empty = GeometryMeasuredGenerator(support, ZERO, ZERO, None, ZERO, (), (), ())
    if cell is None:
        return empty
    if type(cell) is not int or not 0 <= cell < len(medium.material_ids):
        raise TransportError("generator cell must be valid or None for vacuum")
    if medium.voxel_density[cell] == 0:
        return empty
    physical = medium.physical
    cutoff = _f(0.0 if physical is None else physical.cutoff_kev)
    if support.upper_kev <= cutoff:
        return empty
    nodes = tuple(map(_f, spec.energy_nodes))
    if support.lower_kev < nodes[0] or support.upper_kev > nodes[-1]:
        raise TransportError("generator incident support exceeds supplied coefficient range")
    if spec.scattering_law == "bound-electron-compton" and (
        support.lower_kev < _f(0.1) or support.upper_kev > 5000
    ):
        raise TransportError("generator bound incident support exceeds 0.1..5000 keV")
    material, width = medium.material_ids[cell], len(nodes)
    start, stop = material * width, (material + 1) * width
    a = _polynomial(nodes, tuple(map(_f, medium.absorption_mm_inverse[start:stop])), support)
    s = _polynomial(nodes, tuple(map(_f, medium.scattering_mm_inverse[start:stop])), support)
    r = (
        ZERO
        if physical is None
        else _polynomial(
            nodes,
            tuple(map(_f, physical.rayleigh_mm_inverse[start:stop])),
            support,
        )
    )
    b = _sum(s, _scale(r, Fraction(-1)))
    total = _sum(a, s)
    factor = _f(medium.density[material] * medium.voxel_density[cell])
    represented = (
        float(support.lower_kev)
        if support.attained
        else _represented_open_maximum(support.lower_kev, support.upper_kev)
    )
    if (
        physical is not None
        and total != ZERO
        and represented is not None
        and (
            represented / HC_KEV_ANGSTROM
            > physical.scattering_tables[material].momentum_transfer_inverse_angstrom[-1]
        )
    ):
        raise TransportError("generator physical collision exceeds angular table support")
    rayleigh_shape = (
        ()
        if r == ZERO or represented is None
        else _rayleigh_shape(physical.scattering_tables[material], represented)
        if physical is not None
        else ()
    )
    bound_profiles: tuple[GeometryBoundProfile, ...] = ()
    law = None if b == ZERO else spec.scattering_law
    if b != ZERO and law == "bound-electron-compton":
        assert physical is not None
        if support.lower_kev < _f(0.1) or support.upper_kev > 5000:
            raise TransportError("generator bound incident support exceeds 0.1..5000 keV")
        shells = physical.bound_compton[material].shells
        accessible = tuple(
            shell
            for shell in shells
            if (
                _f(shell.binding_kev) < support.lower_kev
                if support.attained
                else _f(shell.binding_kev) <= support.lower_kev
            )
        )
        if not accessible:
            raise TransportError("generator positive bound rate has no accessible shells")
        continuing = any(
            _f(shell.binding_kev) + cutoff < support.lower_kev
            if support.attained
            else _f(shell.binding_kev) + cutoff <= support.lower_kev
            for shell in accessible
        )
        if continuing:
            groups: dict[tuple[Fraction, Fraction], Fraction] = {}
            for shell in accessible:
                key = (_f(shell.binding_kev), _f(shell.momentum_height))
                groups[key] = groups.get(key, Fraction(0)) + _f(shell.occupancy)
            occupancy = sum(groups.values(), Fraction(0))
            bound_profiles = tuple(
                GeometryBoundProfile(binding, height, occupied / occupancy)
                for (binding, height), occupied in sorted(groups.items())
            )
        else:
            b, law = ZERO, None
    fluorescence: dict[Fraction, Polynomial] = {}
    if a != ZERO and physical is not None and physical.fluorescence:
        for line in physical.fluorescence[material]:
            if _f(line.energy_kev) <= cutoff or _f(line.threshold_kev) > support.lower_kev:
                continue
            probability = _polynomial(nodes, tuple(map(_f, line.probabilities)), support)
            rate = _scale(_product(a, probability), factor)
            if rate != ZERO:
                energy = _f(line.energy_kev)
                fluorescence[energy] = _sum(fluorescence.get(energy, ZERO), rate)
    return GeometryMeasuredGenerator(
        support,
        _scale(total, factor),
        _scale(b, factor),
        law,
        _scale(r, factor),
        rayleigh_shape,
        bound_profiles,
        tuple(sorted(fluorescence.items())),
    )


def measured_geometry_generator(
    spec: TransportSpec, medium: GeometryPhysicalMedium, cell: int | None, energy_kev: float
) -> GeometryMeasuredGenerator:
    """Return the sufficient exact measured-law signature at one attained mark.

    ``energy_kev`` is a positive finite binary64 keV mark and ``cell`` is a
    flattened x-fastest index or ``None`` for vacuum. Material-major A/S/R
    tables have shape ``(materials*energy_nodes,)`` and density ownership is
    fixed. This host function owns no device buffers and differentiates no
    inputs. Positive-total physical states retain canonical ``E/hc`` coverage.
    Profiles, fluorescence thresholds and cutoff are the actual supplied laws.

    Terminal outcomes affect extinction only. This signature certifies a
    photon-count/energy continuation generator, not a deposition objective,
    complete geometry derivative or numerical equality of sampled executions.

    With existing admitted records::

        signature = measured_geometry_generator(spec, medium, 0, spec.energy_kev)
        assert signature.support.attained

    """
    _validate(spec, medium)
    if not math.isfinite(energy_kev) or energy_kev <= 0:
        raise TransportError("generator mark must have positive finite energy")
    energy = _f(energy_kev)
    return _signature(spec, medium, cell, GeometryLawSupport(energy, energy, True))


def compare_geometry_generators(
    spec: TransportSpec,
    medium: GeometryPhysicalMedium,
    left_cell: int | None,
    right_cell: int | None,
    reachable_marks: tuple[GeometryReachableMark, ...],
    reachable_intervals: tuple[GeometryReachableInterval, ...],
) -> GeometryGeneratorComparison:
    """Certify adjacent measured generators on actual reached energy support.

    Only marks/families whose cell sets meet either compared material cell are
    retained. ``None`` denotes a vacuum side. Families are split symbolically
    at coefficient nodes, shell bindings, binding-plus-cutoff transitions and
    fluorescence thresholds. Open endpoints remain open and no arbitrary test
    energies are used. Grouped lines retain exact polynomial emission rates,
    accessible identical profiles combine occupancy, and Rayleigh compares
    normalised affine form factors only over reached momentum transfer.

    ``equivalent`` is an exact sufficient certificate. ``uncertified`` retains
    each distinct symbolic component without proving physical inequality or a
    missing derivative. An isolated accidental bound ``B/N`` compensation can
    require a stronger analytical identity. Numerical profile conditioning,
    face activation and completeness of supplied physical/auxiliary support
    are separate obligations. This cold host calculation imports no provider,
    submits no native work and changes no active gradient parameter.

    With an existing admission certificate::

        result = compare_geometry_generators(
            spec, medium, 0, 1, certificate.marks, certificate.intervals,
        )
        assert result.status in ("equivalent", "uncertified")

    """
    _validate(spec, medium)
    if any(not isinstance(mark, GeometryReachableMark) for mark in reachable_marks):
        raise TransportError("generator reachable marks must be explicit mark records")
    reachable_marks = tuple(
        GeometryReachableMark(mark.energy_kev, tuple(mark.cells)) for mark in reachable_marks
    )
    reachable_intervals = tuple(reachable_intervals)
    compared = {cell for cell in (left_cell, right_cell) if cell is not None}
    if any(type(cell) is not int or not 0 <= cell < len(medium.material_ids) for cell in compared):
        raise TransportError("generator cells must be valid or None for vacuum")
    physical = medium.physical
    cutoff = _f(0.0 if physical is None else physical.cutoff_kev)
    splits = set(map(_f, spec.energy_nodes)) | {cutoff}
    if physical is not None:
        materials = {
            medium.material_ids[cell] for cell in compared if medium.voxel_density[cell] > 0
        }
        for material in materials:
            if physical.bound_compton:
                for shell in physical.bound_compton[material].shells:
                    splits.update((_f(shell.binding_kev), _f(shell.binding_kev) + cutoff))
            if physical.fluorescence:
                splits.update(_f(line.threshold_kev) for line in physical.fluorescence[material])
    supports: set[GeometryLawSupport] = set()
    for mark in reachable_marks:
        if (
            not math.isfinite(mark.energy_kev)
            or mark.energy_kev <= 0
            or any(
                type(cell) is not int or not 0 <= cell < len(medium.material_ids)
                for cell in mark.cells
            )
        ):
            raise TransportError("generator reachable marks require valid energy/cells")
        energy = _f(mark.energy_kev)
        if compared.intersection(mark.cells) and energy > cutoff:
            supports.add(GeometryLawSupport(energy, energy, True))
    for family in reachable_intervals:
        if not isinstance(family, GeometryReachableInterval) or any(
            cell >= len(medium.material_ids) for cell in family.cells
        ):
            raise TransportError("generator reachable intervals require valid families")
        if not compared.intersection(family.cells):
            continue
        low, high = max(family.lower_kev, cutoff), family.upper_kev
        if low >= high:
            continue
        bounds = sorted({low, high} | {value for value in splits if low < value < high})
        supports.update(GeometryLawSupport(a, b, False) for a, b in pairwise(bounds))
    ordered = tuple(sorted(supports, key=lambda s: (s.lower_kev, s.upper_kev, s.attained)))
    witnesses: list[GeometryGeneratorWitness] = []
    for support in ordered:
        left, right = (_signature(spec, medium, cell, support) for cell in (left_cell, right_cell))
        for field in fields(GeometryMeasuredGenerator):
            if field.name == "support":
                continue
            first, last = getattr(left, field.name), getattr(right, field.name)
            if first != last:
                witnesses.append(
                    GeometryGeneratorWitness(support, field.name, repr(first), repr(last))
                )
    return GeometryGeneratorComparison(
        "uncertified" if witnesses else "equivalent",
        left_cell,
        right_cell,
        reachable_marks,
        reachable_intervals,
        ordered,
        tuple(witnesses),
    )
