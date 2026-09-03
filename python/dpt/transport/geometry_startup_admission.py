"""Pre-sampling admission of transverse finite-source startup interfaces."""

from __future__ import annotations

# The launch chart shares the checked source-law primitive.
# pyright: reportPrivateUsage=false
import math
from fractions import Fraction
from itertools import pairwise, product
from typing import TYPE_CHECKING, cast

from .model import TransportError, TransportSpec, _physical
from .source import ParallelBeam, PointConeSource
from .uncollided_geometry import AcquisitionGeometryTangent, _source_ray

if TYPE_CHECKING:
    from .geometry_histories import GeometryPhysicalMedium
    from .physical import AtomicScatteringTable

Point = tuple[Fraction, Fraction]
Affine = tuple[Fraction, Fraction, Fraction]
Law = tuple[object, ...]


def _affine(first: float, second: float, third: float) -> Affine:
    # Exact affine arithmetic of the represented binary64 corner chart.
    values = tuple(Fraction.from_float(value) for value in (first, second, third))
    return values[1] - values[0], values[2] - values[0], values[0]


def _value(coefficients: Affine, point: Point) -> Fraction:
    return coefficients[0] * point[0] + coefficients[1] * point[1] + coefficients[2]


def _clip(polygon: tuple[Point, ...], coefficients: Affine) -> tuple[Point, ...]:
    """Clip a convex source polygon by one exact affine nonnegative halfspace."""
    if not polygon:
        return ()
    output: list[Point] = []
    previous = polygon[-1]
    previous_value = _value(coefficients, previous)
    for current in polygon:
        current_value = _value(coefficients, current)
        if (previous_value < 0) != (current_value < 0):
            fraction = previous_value / (previous_value - current_value)
            output.append(
                (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                )
            )
        if current_value >= 0:
            output.append(current)
        previous, previous_value = current, current_value
    return tuple(output)


def _positive_area(polygon: tuple[Point, ...]) -> bool:
    if len(polygon) < 3:
        return False
    return (
        abs(
            sum(
                (
                    polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
                    - polygon[i][1] * polygon[(i + 1) % len(polygon)][0]
                    for i in range(len(polygon))
                ),
                Fraction(0),
            )
        )
        > 0
    )


def _canonical(points: tuple[Point, ...]) -> tuple[Point, ...]:
    output: list[Point] = []
    for point in points:
        output.append(point)
        while len(output) >= 3:
            a, b, c = output[-3:]
            if (b[1] - a[1]) * (c[0] - b[0]) != (c[1] - b[1]) * (b[0] - a[0]):
                break
            del output[-2]
    return tuple(output)


def _interpolate(points: tuple[Point, ...], argument: Fraction) -> Fraction:
    if len(points) == 1:
        return points[0][1]
    for first, second in pairwise(points):
        if first[0] <= argument <= second[0]:
            ratio = (argument - first[0]) / (second[0] - first[0])
            return first[1] + ratio * (second[1] - first[1])
    raise TransportError("startup signature argument is outside its declared support")


def _rayleigh_law(table: AtomicScatteringTable, maximum_q: Fraction) -> tuple[Point, ...]:
    """Canonical linear F/F(0) on q accessible at the initial photon energy."""
    scale = Fraction.from_float(table.form_factor[0])
    nodes = tuple(
        (Fraction.from_float(q), Fraction.from_float(f) / scale)
        for q, f in zip(table.momentum_transfer_inverse_angstrom, table.form_factor, strict=True)
    )
    if maximum_q > nodes[-1][0]:
        raise TransportError("startup Rayleigh signature requires complete active angular support")
    points = tuple(point for point in nodes if point[0] < maximum_q)
    return _canonical(points + ((maximum_q, _interpolate(nodes, maximum_q)),))  # noqa: RUF005


def _interpolate_initial(values: tuple[float, ...], energy: float, spec: TransportSpec) -> float:
    """Keep the canonical two weighted binary64 coefficient terms."""
    nodes = spec.energy_nodes
    if len(nodes) == 1:
        return values[0]
    lower = 0
    while lower < len(nodes) - 2 and energy > nodes[lower + 1]:
        lower += 1
    fraction = (energy - nodes[lower]) / (nodes[lower + 1] - nodes[lower])
    return (1.0 - fraction) * values[lower] + fraction * values[lower + 1]


def _material_law(
    cell: int | None, energy: float, spec: TransportSpec, medium: GeometryPhysicalMedium
) -> Law:
    empty: Law = (0.0, 0.0, 0.0, (), (), ())
    physical = medium.physical
    if cell is None or (physical is not None and energy <= physical.cutoff_kev):
        return empty
    material = medium.material_ids[cell]
    factor = medium.factor(cell)
    count = len(spec.energy_nodes)
    first = material * count
    absorption = factor * _interpolate_initial(
        medium.absorption_mm_inverse[first : first + count], energy, spec
    )
    scattering = factor * _interpolate_initial(
        medium.scattering_mm_inverse[first : first + count], energy, spec
    )
    coherent = (
        0.0
        if physical is None
        else factor
        * _interpolate_initial(physical.rayleigh_mm_inverse[first : first + count], energy, spec)
    )
    if any(not math.isfinite(v) for v in (absorption, scattering, coherent)):
        raise TransportError("startup physical rates are not representable in binary64")
    if absorption == scattering == coherent == 0:
        return empty
    angular: tuple[Point, ...] = ()
    shells: tuple[tuple[Fraction, float, float], ...] = ()
    fluorescence: tuple[tuple[float, Fraction], ...] = ()
    if physical is not None:
        if coherent > 0:
            from .physical import HC_KEV_ANGSTROM

            angular = _rayleigh_law(
                physical.scattering_tables[material],
                Fraction.from_float(energy / HC_KEV_ANGSTROM),
            )
        if spec.scattering_law == "bound-electron-compton" and scattering > coherent:
            table = physical.bound_compton[material]
            grouped: dict[tuple[float, float], Fraction] = {}
            for shell in table.shells:
                if shell.binding_kev >= energy:
                    continue
                key = shell.binding_kev, shell.profile_zero
                grouped[key] = grouped.get(key, Fraction(0)) + Fraction.from_float(shell.occupancy)
            total = sum(grouped.values(), Fraction(0))
            if total == 0:
                raise TransportError("startup bound interaction has no accessible occupied shell")
            shells = tuple(
                sorted(
                    (weight / total, binding, profile)
                    for (binding, profile), weight in grouped.items()
                )
            )
        if absorption > 0 and physical.fluorescence:
            grouped_lines: dict[float, Fraction] = {}
            for line in physical.fluorescence[material]:
                # Its complete detector photon-count/energy continuation is zero
                # at or below cutoff, as for nonradiative local absorption.
                if energy < line.threshold_kev or line.energy_kev <= physical.cutoff_kev:
                    continue
                probability = _interpolate_initial(line.probabilities, energy, spec)
                if probability > 0:
                    grouped_lines[line.energy_kev] = grouped_lines.get(
                        line.energy_kev, Fraction(0)
                    ) + Fraction.from_float(probability)
            fluorescence = tuple(sorted(grouped_lines.items()))
    return absorption, scattering, coherent, angular, shells, fluorescence


def _initial_energies(
    spec: TransportSpec,
    medium: GeometryPhysicalMedium,
    emitted_energies_kev: tuple[float, ...] | None,
) -> tuple[float, ...]:
    physical = medium.physical
    if emitted_energies_kev is None:
        energies = (
            (spec.energy_kev,)
            if physical is None or physical.spectrum is None
            else physical.spectrum.energy_kev
        )
    else:
        energies = emitted_energies_kev
    if not isinstance(cast(object, energies), tuple) or not energies:
        raise TransportError("startup admission requires nonempty immutable emitted energies")
    cutoff = 0.0 if physical is None else physical.cutoff_kev
    checked = tuple(
        _physical(energy, "startup emitted energy", positive=True) for energy in energies
    )
    for energy in checked:
        if energy > cutoff and not spec.energy_nodes[0] <= energy <= spec.energy_nodes[-1]:
            raise TransportError("startup emitted energy is outside surviving coefficient support")
    return checked


def admit_source_startup_chart(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    medium: GeometryPhysicalMedium,
    tangent: AcquisitionGeometryTangent,
    *,
    emitted_energies_kev: tuple[float, ...] | None = None,
) -> None:
    """Reject a moving full launch plane on a genuinely different transverse law.

    The caller validates the medium and calls this before drawing coordinates.
    This is admission for the detector photon-count/energy-keV generator.
    Subcutoff fluorescent futures join local absorption without changing any
    physical event law. Compare actual initial spectrum atoms or original energy
    marks supplied by a composed observation. Equal local collision generators
    act on the same complete future; unused coefficient energies do not create
    a startup kink. Generic source-inside launches remain supported. Point
    emitters and the subcutoff opaque-box law keep their separate admission.
    Pencils and one-dimensional beams retain the same source unit-square
    measure, with half-open cell ownership for constant tangential coordinates.
    """
    if not isinstance(source, ParallelBeam):
        return
    energies = _initial_energies(spec, medium, emitted_energies_kev)
    samples = tuple(
        _source_ray(source, uv, tangent.source)[0].in_object_frame(
            spec.object_pose, tangent.object.motion
        )
        for uv in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    )
    coordinates = tuple(
        _affine(*(sample.origin_mm[axis] for sample in samples)) for axis in range(3)
    )
    velocities = tuple(
        _affine(*(sample.origin_dot[axis] for sample in samples)) for axis in range(3)
    )
    counts = spec.grid.shape_xyz
    signatures: dict[tuple[int | None, float], Law] = {}
    for axis, count in enumerate(counts):
        if samples[0].direction[axis] == 0.0:
            continue
        coordinate = coordinates[axis]
        if coordinate[:2] != (Fraction(0), Fraction(0)):
            continue
        for face in range(count + 1):
            location = spec.grid.origin_mm[axis] + face * spec.grid.spacing_mm[axis]
            if not math.isfinite(location):
                raise TransportError("startup material face is not representable in binary64")
            if coordinate[2] != Fraction.from_float(location):
                continue
            face_dot = tangent.object.origin_dot[axis] + face * tangent.object.spacing_dot[axis]
            if not math.isfinite(face_dot):
                raise TransportError(
                    "startup material face velocity is not representable in binary64"
                )
            velocity = velocities[axis][:2] + (  # noqa: RUF005
                velocities[axis][2] - Fraction.from_float(face_dot),
            )
            if velocity == (Fraction(0), Fraction(0), Fraction(0)):
                continue
            other = tuple(i for i in range(3) if i != axis)
            for first, second in product(range(counts[other[0]]), range(counts[other[1]])):
                polygon: tuple[Point, ...] = (
                    (Fraction(0), Fraction(0)),
                    (Fraction(1), Fraction(0)),
                    (Fraction(1), Fraction(1)),
                    (Fraction(0), Fraction(1)),
                )
                indices = [0, 0, 0]
                indices[other[0]], indices[other[1]] = first, second
                for tangential in other:
                    lower = (
                        spec.grid.origin_mm[tangential]
                        + indices[tangential] * spec.grid.spacing_mm[tangential]
                    )
                    upper = (
                        spec.grid.origin_mm[tangential]
                        + (indices[tangential] + 1) * spec.grid.spacing_mm[tangential]
                    )
                    a, b, c = coordinates[tangential]
                    if a == b == 0:
                        if not Fraction.from_float(lower) <= c < Fraction.from_float(upper):
                            polygon = ()
                            break
                        continue
                    polygon = _clip(polygon, (a, b, c - Fraction.from_float(lower)))
                    polygon = _clip(polygon, (-a, -b, Fraction.from_float(upper) - c))
                if not _positive_area(polygon):
                    continue
                indices[axis] = face - 1
                negative = (
                    ((indices[2] * counts[1] + indices[1]) * counts[0] + indices[0])
                    if face
                    else None
                )
                indices[axis] = face
                positive = (
                    ((indices[2] * counts[1] + indices[1]) * counts[0] + indices[0])
                    if face < count
                    else None
                )
                for energy in energies:
                    for cell in (negative, positive):
                        key = cell, energy
                        if key not in signatures:
                            signatures[key] = _material_law(cell, energy, spec, medium)
                    if signatures[negative, energy] != signatures[positive, energy]:
                        raise TransportError(
                            "active source launch plane coincides with a discontinuous "
                            "transverse material face"
                        )


__all__ = ["admit_source_startup_chart"]
