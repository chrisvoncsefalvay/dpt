"""Reachable-energy admission for the declared bound-electron geometry law.

For occupied shell binding energies ``U_i``, the ideal analytical impulse
profile has positive density throughout its momentum support. Consequently
the conditional normaliser is positive exactly when ``E > min(U_i)``, and a
selected bound collision has outgoing support ``0 < E' < E-min(U_i)``. Only
``E' > cutoff`` continues. This module checks those supports against the actual
noncoherent partial ``B(E)=scattering(E)-Rayleigh(E)`` rather than requiring a
cutoff above every shell binding. All coefficients use their declared linear
interpolants, and all endpoint comparisons use exact rational representations
of the supplied binary64 values.

This is a host-only physical-law certificate, independent of finite launch
budgets. It does not establish numerical conditioning of a profile CDF or a
confidence estimator. Auxiliary forced geometry continuations require explicit
caller-derived seeds; an unperturbed source certificate alone cannot establish
their support. No angular or energy support is truncated to obtain admission.

Physical reference: https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/penelope_compton.html
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, pairwise
from typing import TYPE_CHECKING, Literal, cast

from dpt.geometry import Vector3

from .model import TransportError, TransportSpec
from .physical import HC_KEV_ANGSTROM, PhysicalTransport
from .source import ParallelBeam, PointConeSource

if TYPE_CHECKING:
    from .geometry_histories import GeometryPhysicalMedium

Vector = tuple[Fraction, Fraction, Fraction]
Constraint = tuple[Vector, Fraction, bool]


@dataclass(frozen=True, slots=True)
class BoundReachableMark:
    """An attained discrete photon energy and its positive-density cells."""

    energy_kev: float
    cells: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BoundReachableInterval:
    """A globally reachable open interval in keV; neither endpoint is attained.

    Fractions retain the exact real-arithmetic endpoints implied by binary64
    source energies, cutoffs and bindings. ``float(upper_kev)`` is suitable for
    display, but must not turn the supremum into a fluorescence energy mark.
    """

    lower_kev: Fraction
    upper_kev: Fraction
    cells: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BoundGeometryAdmission:
    """Immutable support certificate for a full, untruncated analogue law.

    ``source_cells`` are cells entered for a positive-measure source set, or a
    positive-length flight of a deterministic pencil. Lower and internal
    parallel faces retain the canonical half-open owner; mere point contacts
    do not enter a cell. ``marks`` include continuing source and fluorescence
    energies. Bound scattering contributes ``interval``. Positive Rayleigh
    permits finite sequences of small rotations within an interior collision
    neighbourhood, so it can expand a mark's spatial support at unchanged
    energy. Fluorescence and bound scattering expand only outgoing energies.

    ``auxiliary_seeds`` bind extra unrestricted cell/energy support families.
    They omit a forced point/direction, so their spatial closure is conservative
    and can reject a valid ray-constrained outward future. They do not assert
    that all active geometry interface/source-side futures were enumerated. Numerical
    CDF/normalisation conditioning, finite event guards and the geometry
    confidence proof remain separate obligations.
    All fields are host data; this function performs no uploads or native work.
    """

    source_cells: tuple[int, ...]
    source_energies_kev: tuple[float, ...]
    auxiliary_seeds: tuple[tuple[int, float], ...]
    marks: tuple[BoundReachableMark, ...]
    interval: BoundReachableInterval | None
    scope: Literal["source-physical-law", "source-and-explicit-auxiliary-law"]
    numerical_conditioning_certified: Literal[False] = False


def _f(value: float) -> Fraction:
    return Fraction(value)


def _det(rows: tuple[Vector, Vector, Vector]) -> Fraction:
    a, b, c = rows
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _strict_feasible(constraints: tuple[Constraint, ...]) -> bool:
    """Decide a bounded affine polytope with strict faces by exact vertices.

    A closed polytope admits all strict inequalities simultaneously iff each
    strict inequality has slack at some feasible vertex: the average of those
    witness vertices has every required slack. This preserves positive-measure
    source support without an arbitrary tolerance or a displaced ray.
    """
    witnessed = [not strict for _, _, strict in constraints]
    found = False
    for selected in combinations(constraints, 3):
        rows = (selected[0][0], selected[1][0], selected[2][0])
        determinant = _det(rows)
        if determinant == 0:
            continue
        right = (selected[0][1], selected[1][1], selected[2][1])
        coordinates: list[Fraction] = []
        for column in range(3):
            replaced = tuple(
                tuple(right[r] if k == column else rows[r][k] for k in range(3)) for r in range(3)
            )
            coordinates.append(_det(cast(tuple[Vector, Vector, Vector], replaced)) / determinant)
        slacks = [
            rhs - sum((a * x for a, x in zip(row, coordinates, strict=True)), Fraction())
            for row, rhs, _ in constraints
        ]
        if any(slack < 0 for slack in slacks):
            continue
        found = True
        for index, slack in enumerate(slacks):
            witnessed[index] = witnessed[index] or slack > 0
        if all(witnessed):
            return True
    return found and all(witnessed)


def _source_chart(
    spec: TransportSpec, source: ParallelBeam | PointConeSource
) -> tuple[Vector, tuple[Vector, Vector, Vector], tuple[Constraint, ...]]:
    inverse = spec.object_pose.inverse()
    if isinstance(source, ParallelBeam):
        origin = inverse.point(source.lower_mm)
        columns = (
            inverse.direction((source.extent_xy_mm[0], 0.0, 0.0)),
            inverse.direction((0.0, source.extent_xy_mm[1], 0.0)),
            inverse.direction(source.direction),
        )
        constraints: tuple[Constraint, ...] = (
            ((Fraction(-1), Fraction(), Fraction()), Fraction(), source.extent_xy_mm[0] > 0),
            ((Fraction(1), Fraction(), Fraction()), Fraction(1), source.extent_xy_mm[0] > 0),
            ((Fraction(), Fraction(-1), Fraction()), Fraction(), source.extent_xy_mm[1] > 0),
            ((Fraction(), Fraction(1), Fraction()), Fraction(1), source.extent_xy_mm[1] > 0),
            ((Fraction(), Fraction(), Fraction(-1)), Fraction(), False),
        )
    else:
        detector = source.geometry
        origin = inverse.point(detector.source_mm)
        lower = tuple(
            detector.origin_mm[k]
            - 0.5
            * (detector.spacing_mm[0] * detector.u[k] + detector.spacing_mm[1] * detector.v[k])
            for k in range(3)
        )
        columns = (
            inverse.direction(
                cast(Vector3, tuple(lower[k] - detector.source_mm[k] for k in range(3)))
            ),
            inverse.direction(
                cast(
                    Vector3,
                    tuple(detector.spacing_mm[0] * detector.shape[1] * v for v in detector.u),
                )
            ),
            inverse.direction(
                cast(
                    Vector3,
                    tuple(detector.spacing_mm[1] * detector.shape[0] * v for v in detector.v),
                )
            ),
        )
        # x=source+a*(lower-source)+b*width*u+c*height*v,
        # a>0, 0<b<a, 0<c<a: rays continue beyond their detector targets.
        constraints = (
            ((Fraction(-1), Fraction(), Fraction()), Fraction(), True),
            ((Fraction(), Fraction(-1), Fraction()), Fraction(), True),
            ((Fraction(-1), Fraction(1), Fraction()), Fraction(), True),
            ((Fraction(), Fraction(), Fraction(-1)), Fraction(), True),
            ((Fraction(-1), Fraction(), Fraction(1)), Fraction(), True),
        )
    rows = tuple(tuple(_f(columns[j][k]) for j in range(3)) for k in range(3))
    return (
        cast(Vector, tuple(_f(v) for v in origin)),
        cast(tuple[Vector, Vector, Vector], rows),
        constraints,
    )


def _chart_enters(
    chart: tuple[Vector, tuple[Vector, Vector, Vector], tuple[Constraint, ...]],
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> bool:
    origin, rows, initial = chart
    constraints = list(initial)
    for value, row, low, high in zip(origin, rows, lower, upper, strict=True):
        if all(coefficient == 0 for coefficient in row):
            if not _f(low) <= value < _f(high):
                return False
        else:
            constraints.extend(
                (
                    ((-row[0], -row[1], -row[2]), value - _f(low), True),
                    (row, _f(high) - value, True),
                )
            )
    return _strict_feasible(tuple(constraints))


def _source_cells(
    spec: TransportSpec, medium: GeometryPhysicalMedium, source: ParallelBeam | PointConeSource
) -> tuple[int, ...]:
    chart = _source_chart(spec, source)
    if not _chart_enters(chart, spec.grid.origin_mm, spec.grid.upper_mm):
        return ()
    nx, ny, _ = spec.grid.shape_xyz
    result: list[int] = []
    for cell in range(len(medium.material_ids)):
        if medium.voxel_density[cell] == 0:
            continue
        coordinate = (cell % nx, (cell // nx) % ny, cell // (nx * ny))
        lower = tuple(
            spec.grid.origin_mm[k] + coordinate[k] * spec.grid.spacing_mm[k] for k in range(3)
        )
        upper = tuple(
            spec.grid.origin_mm[k] + (coordinate[k] + 1) * spec.grid.spacing_mm[k] for k in range(3)
        )
        if _chart_enters(chart, cast(Vector3, lower), cast(Vector3, upper)):
            result.append(cell)
    return tuple(result)


def _interpolate(
    nodes: tuple[Fraction, ...], values: tuple[Fraction, ...], energy: Fraction
) -> Fraction:
    if not nodes[0] <= energy <= nodes[-1]:
        raise TransportError(
            f"reachable bound geometry energy {float(energy):.17g} keV "
            "is outside coefficient support"
        )
    for index, (low, high) in enumerate(pairwise(nodes)):
        if low <= energy <= high:
            fraction = (energy - low) / (high - low)
            return (1 - fraction) * values[index] + fraction * values[index + 1]
    return values[0]


def _positive_segments(
    nodes: tuple[Fraction, ...], *tables: tuple[Fraction, ...]
) -> tuple[tuple[Fraction, Fraction], ...]:
    # Nonnegative affine functions are positive throughout an edge interior
    # unless both endpoint values vanish. Their product has the same property.
    return tuple(
        (low, high)
        for index, (low, high) in enumerate(pairwise(nodes))
        if all(values[index] > 0 or values[index + 1] > 0 for values in tables)
    )


def _overlaps(
    low: Fraction, high: Fraction, segments: tuple[tuple[Fraction, Fraction], ...]
) -> bool:
    return any(max(low, left) < min(high, right) for left, right in segments)


def _validate(
    spec: TransportSpec, medium: GeometryPhysicalMedium, physical: PhysicalTransport
) -> None:
    # Structural validation only: old/new callers may impose a broader support
    # rule in medium.validate, which this reachable-law admission must not reuse.
    count, materials, energies = (
        math.prod(spec.grid.shape),
        spec.grid.materials,
        len(spec.energy_nodes),
    )
    if len(medium.material_ids) != count or any(
        type(m) is not int or not 0 <= m < materials for m in medium.material_ids
    ):
        raise TransportError("bound admission requires one valid material ID per cell")
    if len(medium.density) != materials or len(medium.voxel_density) != count:
        raise TransportError("bound admission density ownership does not match the grid")
    if any(not math.isfinite(v) or v <= 0 for v in medium.density) or any(
        not math.isfinite(v) or v < 0 for v in medium.voxel_density
    ):
        raise TransportError(
            "bound admission requires positive material and nonnegative voxel densities"
        )
    arrays = (
        medium.absorption_mm_inverse,
        medium.scattering_mm_inverse,
        physical.rayleigh_mm_inverse,
    )
    if any(len(values) != materials * energies for values in arrays) or any(
        not math.isfinite(v) or v < 0 for values in arrays for v in values
    ):
        raise TransportError(
            "bound admission coefficients require finite nonnegative material-major tables"
        )
    if any(
        r > s
        for r, s in zip(physical.rayleigh_mm_inverse, medium.scattering_mm_inverse, strict=True)
    ):
        raise TransportError("Rayleigh partial exceeds aggregate scattering")
    if len(physical.bound_compton) != materials or len(physical.scattering_tables) != materials:
        raise TransportError("bound admission requires complete physical tables for each material")
    if physical.cutoff_kev < 0.1 or spec.compton_sampler != "legacy-rejection":
        raise TransportError(
            "bound admission preserves the cutoff>=0.1 keV and legacy sampler domain"
        )
    if physical.fluorescence and len(physical.fluorescence) != materials:
        raise TransportError("fluorescence ownership does not match material count")
    for material, table in enumerate(physical.bound_compton):
        angular = physical.scattering_tables[material]
        if table.atomic_number != angular.atomic_number:
            raise TransportError("bound and atomic table elements disagree")
        for line in physical.fluorescence[material] if physical.fluorescence else ():
            if len(line.probabilities) != energies:
                raise TransportError(
                    "fluorescence probabilities must match coefficient energy nodes"
                )
            if any(
                energy < line.threshold_kev and probability != 0
                for energy, probability in zip(spec.energy_nodes, line.probabilities, strict=True)
            ):
                raise TransportError(
                    "fluorescence probability is nonzero below its vacancy threshold"
                )
        for index, energy in enumerate(spec.energy_nodes):
            probability = (
                math.fsum(
                    line.probabilities[index]
                    for line in physical.fluorescence[material]
                    if energy >= line.threshold_kev
                )
                if physical.fluorescence
                else 0.0
            )
            if probability > 1:
                raise TransportError("eligible fluorescence branch probabilities exceed one")
    for cell, material in enumerate(medium.material_ids):
        factor = medium.density[material] * medium.voxel_density[cell]
        if not math.isfinite(factor) or (medium.voxel_density[cell] > 0 and factor <= 0):
            raise TransportError(
                "bound admission requires finite representable cell density factors"
            )
        for index in range(energies):
            rate = (
                medium.absorption_mm_inverse[material * energies + index]
                + medium.scattering_mm_inverse[material * energies + index]
            )
            scaled = factor * rate
            if (
                not math.isfinite(rate)
                or not math.isfinite(scaled)
                or (factor > 0 and rate > 0 and scaled == 0)
            ):
                raise TransportError(
                    "bound admission requires finite representable cell attenuation"
                )


def admit_bound_geometry_law(
    spec: TransportSpec,
    medium: GeometryPhysicalMedium,
    source: ParallelBeam | PointConeSource,
    *,
    auxiliary_seeds: tuple[tuple[int, float], ...] = (),
) -> BoundGeometryAdmission | None:
    """Reject an undefined bound law only at actually reachable continuing energies.

    The arguments are immutable host records and material-major coefficient
    tuples; energies are keV and coefficients mm inverse. ``auxiliary_seeds``
    are explicit ``(flat_cell, incident_energy_kev)`` marks for unrestricted
    cell/energy families, not automatically inferred moving-interface support.
    Without a point/direction chart their closure conservatively overapproximates
    a particular forced ray and may reject it. The return is ``None`` for a
    nonbound law. A ``TransportError`` identifies a
    reachable zero normaliser or coefficient-support gap before geometry output
    may be accepted. This validates physical support, not numerical CDF accuracy
    or a gradient/confidence theorem. Finite event budgets cannot remove a
    physical-law obligation.

    A threshold-tuned law with nodes ``(.1,1,3)``, shell binding ``1`` and
    noncoherent rates ``(0,0,1)`` is admissible at source energy ``3`` and cutoff
    ``.1``: the emitted interval ``(.1,2)`` reaches energies below binding, but
    their noncoherent rate is zero. Rates ``(1,1,1)`` instead require an undefined
    second bound collision there and are rejected. Rayleigh remains elastic;
    fluorescence eligibility requires both a positive photoelectric rate and
    positive line probability at a reachable incident energy.

    With existing ``spec``, ``medium`` and ``source`` host records::

        certificate = admit_bound_geometry_law(spec, medium, source)
        if certificate is not None:
            assert not certificate.numerical_conditioning_certified

    Inputs remain fixed during this admission calculation; the certificate is
    neither a tape gradient nor an estimator of moving-support derivatives.
    """
    if spec.scattering_law != "bound-electron-compton":
        return None
    physical = medium.physical
    if physical is None:
        raise TransportError("bound geometry requires explicit shell and angular tables")
    _validate(spec, medium, physical)
    nodes = tuple(_f(e) for e in spec.energy_nodes)
    cutoff = _f(physical.cutoff_kev)
    cells = tuple(c for c, factor in enumerate(medium.voxel_density) if factor > 0)
    source_cells = _source_cells(spec, medium, source)
    energies = physical.spectrum.energy_kev if physical.spectrum is not None else (spec.energy_kev,)
    if any(not 0.1 <= e <= 5000.0 for e in energies):
        raise TransportError("bound source energies must remain in 0.1..5000 keV")
    marks: dict[Fraction, set[int]] = {}
    for energy in energies:
        if _f(energy) > cutoff:
            marks.setdefault(_f(energy), set()).update(source_cells)
    for cell, energy in auxiliary_seeds:
        if (
            type(cell) is not int
            or not 0 <= cell < len(medium.material_ids)
            or not math.isfinite(energy)
            or not 0.1 <= energy <= 5000.0
        ):
            raise TransportError(
                "auxiliary bound seeds require valid cells and finite domain energies"
            )
        if _f(energy) > cutoff and medium.voxel_density[cell] > 0:
            marks.setdefault(_f(energy), set()).add(cell)
    width = len(nodes)
    absorption = tuple(
        tuple(_f(v) for v in medium.absorption_mm_inverse[m * width : (m + 1) * width])
        for m in range(spec.grid.materials)
    )
    rayleigh = tuple(
        tuple(_f(v) for v in physical.rayleigh_mm_inverse[m * width : (m + 1) * width])
        for m in range(spec.grid.materials)
    )
    noncoherent = tuple(
        tuple(
            _f(s) - _f(r)
            for s, r in zip(
                medium.scattering_mm_inverse[m * width : (m + 1) * width],
                physical.rayleigh_mm_inverse[m * width : (m + 1) * width],
                strict=True,
            )
        )
        for m in range(spec.grid.materials)
    )
    binding = tuple(
        min(_f(shell.binding_kev) for shell in table.shells) for table in physical.bound_compton
    )
    upper = cutoff
    changed = True
    while changed:
        previous = (
            upper,
            tuple((energy, frozenset(reached)) for energy, reached in sorted(marks.items())),
        )
        for energy, reached in tuple(marks.items()):
            if not reached:
                continue
            if not nodes[0] <= energy <= nodes[-1]:
                raise TransportError(
                    f"reachable bound geometry mark {float(energy):.17g} keV "
                    "is outside coefficient support"
                )
            for material in {medium.material_ids[cell] for cell in tuple(reached)}:
                a = _interpolate(nodes, absorption[material], energy)
                b = _interpolate(nodes, noncoherent[material], energy)
                r = _interpolate(nodes, rayleigh[material], energy)
                if r > 0:
                    if (
                        physical.scattering_tables[material].momentum_transfer_inverse_angstrom[-1]
                        < float(energy) / HC_KEV_ANGSTROM
                    ):
                        raise TransportError(
                            "reachable Rayleigh energy exceeds its angular table support"
                        )
                    reached.update(cells)
                if b > 0:
                    if energy <= binding[material]:
                        raise TransportError(
                            f"undefined bound normaliser in material {material} at reachable "
                            f"{float(energy):.17g} keV "
                            f"(binding {float(binding[material]):.17g} keV)"
                        )
                    upper = max(upper, energy - binding[material])
                if a > 0 and physical.fluorescence:
                    for line in physical.fluorescence[material]:
                        if (
                            line.energy_kev > physical.cutoff_kev
                            and energy >= _f(line.threshold_kev)
                            and _interpolate(
                                nodes, tuple(_f(p) for p in line.probabilities), energy
                            )
                            > 0
                        ):
                            marks.setdefault(_f(line.energy_kev), set()).update(cells)
        if upper > cutoff and cells:
            if cutoff < nodes[0] or upper > nodes[-1]:
                raise TransportError(
                    "reachable open bound-energy interval "
                    f"({float(cutoff):.17g}, {float(upper):.17g}) keV "
                    "exceeds coefficient support"
                )
            for material in {medium.material_ids[c] for c in cells}:
                angular_upper = _f(
                    physical.scattering_tables[material].momentum_transfer_inverse_angstrom[-1]
                ) * _f(HC_KEV_ANGSTROM)
                if _overlaps(
                    max(cutoff, angular_upper), upper, _positive_segments(nodes, rayleigh[material])
                ):
                    raise TransportError(
                        "reachable Rayleigh interval exceeds its angular table support"
                    )
                positive = _positive_segments(nodes, noncoherent[material])
                if _overlaps(cutoff, min(upper, binding[material]), positive):
                    raise TransportError(
                        f"undefined bound normaliser in material {material} on reachable energies "
                        f"above {float(cutoff):.17g} and below/equal binding "
                        f"{float(binding[material]):.17g} keV"
                    )
                candidates = tuple(
                    min(upper, right)
                    for left, right in positive
                    if max(cutoff, left) < min(upper, right)
                )
                if candidates:
                    upper = max(upper, max(candidates) - binding[material])
                if physical.fluorescence:
                    for line in physical.fluorescence[material]:
                        eligible = _positive_segments(
                            nodes, absorption[material], tuple(_f(p) for p in line.probabilities)
                        )
                        if line.energy_kev > physical.cutoff_kev and _overlaps(
                            max(cutoff, _f(line.threshold_kev)), upper, eligible
                        ):
                            marks.setdefault(_f(line.energy_kev), set()).update(cells)
        changed = previous != (
            upper,
            tuple((energy, frozenset(reached)) for energy, reached in sorted(marks.items())),
        )
    return BoundGeometryAdmission(
        source_cells,
        tuple(energies),
        tuple(auxiliary_seeds),
        tuple(
            BoundReachableMark(float(energy), tuple(sorted(reached)))
            for energy, reached in sorted(marks.items())
            if reached
        ),
        BoundReachableInterval(cutoff, upper, cells) if upper > cutoff and cells else None,
        "source-and-explicit-auxiliary-law" if auxiliary_seeds else "source-physical-law",
    )
