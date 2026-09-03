"""Measured-generator continuity for whole-source grazing material faces.

An original pencil or one-dimensional source can lie in an internal material
face for its complete flight. The initial generator must agree across every
admitted side and crossover closure. Radiative continuation laws require their separate whole-plane
certificate before detector proposals or geometry derivatives are evaluated.
The checks below use exact rational source/flight support and actual initial
energies; they do not compare unused coefficient energies or displace rays.
"""

from __future__ import annotations

# These new admission modules share their dependency-free support primitives.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import comb
from typing import TYPE_CHECKING, cast

from .geometry_bound_admission import Vector, _source_chart, _strict_feasible
from .geometry_startup_admission import _initial_energies, _material_law
from .geometry_tangents import RayTangent
from .model import TransportError, TransportSpec
from .source import ParallelBeam, PointConeSource
from .uncollided_geometry import AcquisitionGeometryTangent, _source_ray

if TYPE_CHECKING:
    from .geometry_histories import GeometryPhysicalMedium


@dataclass(frozen=True, slots=True)
class GrazingSourceAdmission:
    """Initial generator identities and enclosed auxiliary cell candidates.

    ``auxiliary_cells`` retain the conservative crossover closure, including
    source/launch endpoints; they need not all occur on one exact finite flow.
    The complete physical-law certificate keeps these declarations separate.
    """

    equivalent_faces: frozenset[tuple[int, int]]
    initial_energies_kev: tuple[float, ...]
    auxiliary_cells: tuple[int, ...] = ()


Rational3 = tuple[Fraction, Fraction, Fraction]


def _rational(vector: tuple[float, float, float]) -> Rational3:
    return cast(Rational3, tuple(Fraction(x) for x in vector))


def _cross(left: Rational3, right: Rational3) -> Rational3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _flow_derivatives(
    ray: RayTangent,
    spec: TransportSpec,
    tangent: AcquisitionGeometryTangent,
    *,
    direction: bool,
) -> tuple[Rational3, ...]:
    """Exact first 22 derivatives of the declared attached rigid-flow curve.

    A rotation contributes eigenfrequencies 0,+w,-w. The product of source and
    inverse object rotations has at most nine frequencies; linear intrinsic
    shape/translation coordinates add multiplicity at most two. Thus this
    component is an exponential polynomial of order at most 18. Its first 22
    coefficients determine exact stationarity, without an angle tolerance.
    Unit-direction normalisation has a positive scalar factor and cannot
    change the first nonzero normal coefficient on a whole grazing face.
    """
    # The caller supplies the actual validated RayTangent. Using its stored
    # first jet also preserves the canonical source-spacing/basis convention.
    source_motion, object_motion = tangent.source.motion, tangent.object.motion
    value = ray.direction if direction else ray.origin_mm
    velocity = ray.direction_dot if direction else ray.origin_dot
    rigid_velocity = (
        source_motion.direction_velocity(value)
        if direction
        else source_motion.point_velocity(value)
    )
    intrinsic = _rational(
        cast(
            tuple[float, float, float],
            tuple(a - b for a, b in zip(velocity, rigid_velocity, strict=True)),
        )
    )
    ws, wo = _rational(source_motion.angular), _rational(object_motion.angular)
    minus_wo = cast(Rational3, tuple(-x for x in wo))
    point = _rational(value)
    ps = (Fraction(),) * 3 if direction else _rational(source_motion.pivot_mm)
    po = (Fraction(),) * 3 if direction else _rational(object_motion.pivot_mm)
    vs = (Fraction(),) * 3 if direction else _rational(source_motion.translation)
    vo = (Fraction(),) * 3 if direction else _rational(object_motion.translation)
    initial = cast(Rational3, tuple(a - b for a, b in zip(point, ps, strict=True)))
    powers = [initial]
    intrinsic_powers = [intrinsic]
    for _ in range(21):
        powers.append(_cross(ws, powers[-1]))
        intrinsic_powers.append(_cross(ws, intrinsic_powers[-1]))
    world: list[Rational3] = []
    for order in range(22):
        if order == 0:
            item = point
        else:
            item = cast(
                Rational3,
                tuple(
                    powers[order][axis]
                    + order * intrinsic_powers[order - 1][axis]
                    + (vs[axis] if order == 1 else 0)
                    for axis in range(3)
                ),
            )
        world.append(
            cast(
                Rational3,
                tuple(
                    item[axis] - (po[axis] if order == 0 else 0) - (vo[axis] if order == 1 else 0)
                    for axis in range(3)
                ),
            )
        )
    rotation = tuple(Fraction(x) for x in spec.object_pose.inverse().rotation)
    result: list[Rational3] = []
    for order in range(22):
        vector = [Fraction()] * 3
        for k in range(order + 1):
            item = world[k]
            for _ in range(order - k):
                item = _cross(minus_wo, item)
            for axis in range(3):
                vector[axis] += comb(order, k) * item[axis]
        result.append(
            cast(
                Rational3,
                tuple(
                    sum(rotation[3 * axis + j] * vector[j] for j in range(3)) for axis in range(3)
                ),
            )
        )
    return tuple(result)


def admit_source_grazing_chart(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    medium: GeometryPhysicalMedium,
    tangent: AcquisitionGeometryTangent,
    *,
    emitted_energies_kev: tuple[float, ...] | None = None,
) -> GrazingSourceAdmission:
    """Identify regular whole-source faces before material-chart preparation.

    A finite point cone retains its separate angular chart. For a parallel
    source, every initial ray is represented by the original unit-square
    coordinates and positive flight distance. On an active coplanar face the
    first-order normal displacement is affine in those three variables. Both
    signs of the active coordinate are tested. Simultaneous faces retain their
    correlated velocity signs, including diagonal corner motions.

    The local measured generator is compared at every actual initial energy.
    Subcutoff fluorescence has zero detector continuation, while continuing
    channels retain their complete conditional laws. A failed equivalence is
    left to the ordinary material chart's discontinuity admission; it is never
    replaced by an arbitrary one-sided derivative.
    """
    energies = _initial_energies(spec, medium, emitted_energies_kev)
    if not isinstance(source, ParallelBeam):
        return GrazingSourceAdmission(frozenset(), energies)
    chart = _source_chart(spec, source)
    origin, rows, initial = chart
    samples = tuple(
        _source_ray(source, uv, tangent.source)[0].in_object_frame(
            spec.object_pose, tangent.object.motion
        )
        for uv in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    )
    whole: dict[int, int] = {}
    active: dict[int, Vector] = {}
    offsets: dict[int, Fraction] = {}
    orders: dict[int, int] = {}
    launched = tuple(
        _source_ray(source, uv, tangent.source)[0] for uv in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    )
    common_rigid = (
        tangent.source.motion == tangent.object.motion
        and tangent.source.extent_dot == (0.0, 0.0)
        and tangent.source.direction_dot == (0.0, 0.0, 0.0)
    )
    curves = None
    for axis, count in enumerate(spec.grid.shape_xyz):
        if rows[axis] != (Fraction(), Fraction(), Fraction()):
            continue
        for face in range(count + 1):
            position = spec.grid.origin_mm[axis] + face * spec.grid.spacing_mm[axis]
            if not math.isfinite(position):
                raise TransportError("grazing material face is not representable")
            if origin[axis] != Fraction(position):
                continue
            whole[axis] = face
            velocity = (
                samples[1].origin_dot[axis] - samples[0].origin_dot[axis],
                samples[2].origin_dot[axis] - samples[0].origin_dot[axis],
                samples[0].direction_dot[axis],
            )
            offset = samples[0].origin_dot[axis] - (
                tangent.object.origin_dot[axis] + face * tangent.object.spacing_dot[axis]
            )
            if not all(math.isfinite(x) for x in (*velocity, offset)):
                raise TransportError("grazing relative velocity is not representable")
            if (*velocity, offset) != (0, 0, 0, 0):
                active[axis] = cast(Vector, tuple(Fraction(x) for x in velocity))
                offsets[axis], orders[axis] = Fraction(offset), 1
            elif not common_rigid and (
                tangent.source.motion.angular != (0.0, 0.0, 0.0)
                or tangent.object.motion.angular != (0.0, 0.0, 0.0)
            ):
                if curves is None:
                    curves = tuple(
                        (
                            _flow_derivatives(ray, spec, tangent, direction=False),
                            _flow_derivatives(ray, spec, tangent, direction=True),
                        )
                        for ray in launched
                    )
                for order in range(2, 22):
                    coefficient = (
                        curves[1][0][order][axis] - curves[0][0][order][axis],
                        curves[2][0][order][axis] - curves[0][0][order][axis],
                        curves[0][1][order][axis],
                    )
                    constant = curves[0][0][order][axis]
                    if (*coefficient, constant) != (0, 0, 0, 0):
                        active[axis] = coefficient
                        offsets[axis], orders[axis] = constant, order
                        break
            break
    if not active:
        return GrazingSourceAdmission(frozenset(), energies)
    active_axes = tuple(active)
    ordinary_axes = tuple(axis for axis in range(3) if axis not in whole)
    signatures: dict[tuple[int | None, float], tuple[object, ...]] = {}
    auxiliary_cells: set[int] = set()
    counts = spec.grid.shape_xyz

    def cell_at(indices: list[int]) -> int | None:
        if any(not 0 <= index < count for index, count in zip(indices, counts, strict=True)):
            return None
        return (indices[2] * counts[1] + indices[1]) * counts[0] + indices[0]

    for indices_part in product(*(range(counts[axis]) for axis in ordinary_axes)):
        constraints = list(initial)
        # Crossover wedges can enter through source-U endpoints or the launch
        # endpoint as well as a cell-patch boundary. This is a conservative
        # closed chart; the ordinary source/flight measure stays strict.
        closure_constraints = [(row, rhs, False) for row, rhs, _ in initial]
        # A positive-length future, rather than a launch-point contact.
        constraints.append(((Fraction(), Fraction(), Fraction(-1)), Fraction(), True))
        closure_constraints.append(((Fraction(), Fraction(), Fraction(-1)), Fraction(), False))
        indices = [whole.get(axis, 0) for axis in range(3)]
        absent = False
        for axis, index in zip(ordinary_axes, indices_part, strict=True):
            indices[axis] = index
            low = Fraction(spec.grid.origin_mm[axis] + index * spec.grid.spacing_mm[axis])
            high = Fraction(spec.grid.origin_mm[axis] + (index + 1) * spec.grid.spacing_mm[axis])
            row = rows[axis]
            if row == (Fraction(), Fraction(), Fraction()):
                if not low <= origin[axis] < high:
                    absent = True
                    break
            else:
                constraints.extend(
                    (
                        (cast(Vector, tuple(-x for x in row)), origin[axis] - low, True),
                        (row, high - origin[axis], True),
                    )
                )
                # A vanishing wedge may meet another cell face at the common
                # leading root. Keep source marks/positive flight strict, but
                # include closures of physical cell patches in this chart gate.
                closure_constraints.extend(
                    (
                        (cast(Vector, tuple(-x for x in row)), origin[axis] - low, False),
                        (row, high - origin[axis], False),
                    )
                )
        if absent:
            continue
        baseline = cell_at(indices)
        for coordinate_sign, choices in product((-1, 1), product((-1, 1), repeat=len(active_axes))):
            branch = list(indices)
            selected = list(constraints)
            closure = list(closure_constraints)
            for axis, normal_sign in zip(active_axes, choices, strict=True):
                branch[axis] = whole[axis] - int(normal_sign < 0)
                factor = -(coordinate_sign ** orders[axis]) * normal_sign
                row = cast(Vector, tuple(factor * x for x in active[axis]))
                selected.append((row, -factor * offsets[axis], True))
                closure.append((row, -factor * offsets[axis], False))
            strict_support = _strict_feasible(tuple(selected))
            if not strict_support and not _strict_feasible(tuple(closure)):
                continue
            neighbour = cell_at(branch)
            auxiliary_cells.update(cell for cell in (baseline, neighbour) if cell is not None)
            for energy in energies:
                for cell in (baseline, neighbour):
                    if (cell, energy) not in signatures:
                        signatures[cell, energy] = _material_law(cell, energy, spec, medium)
                if signatures[baseline, energy] != signatures[neighbour, energy]:
                    if not strict_support:
                        raise TransportError(
                            "whole-source crossover chart is not certified: "
                            "a leading zero locus meets different measured generators"
                        )
                    if any(order > 1 for order in orders.values()):
                        raise TransportError(
                            "rotational curvature leaves a whole-source material face "
                            "with a different measured generator"
                        )
                    return GrazingSourceAdmission(frozenset(), energies)
    return GrazingSourceAdmission(
        frozenset((axis, whole[axis]) for axis in active_axes),
        energies,
        tuple(sorted(auxiliary_cells)),
    )


__all__ = ["GrazingSourceAdmission", "admit_source_grazing_chart"]
