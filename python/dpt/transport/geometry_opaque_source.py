"""Exact source-chart visibility of the canonical subcutoff material box.

An emitted photon at or below the cutoff deposits on material-box entry and
otherwise escapes. Its moving box shadow is a source-area support boundary,
including cone silhouettes and launch points crossing an outer face. Samples
use exact one-sided ray jets; no displaced ray or finite-difference step is used.
"""

from __future__ import annotations

# Checked geometry arithmetic is shared within the geometry implementation.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import cast

from dpt.geometry import Vector3, cross, dot

from .geometry_tangents import (
    AffineBoundary,
    RayTangent,
    _add,
    _scale,
    _sub,
    normalised_direction_jet,
)
from .model import TransportError, TransportSpec
from .source import ParallelBeam, PointConeSource
from .uncollided_geometry import (
    AcquisitionGeometryTangent,
    SourceGeometryTangent,
    _affine_values,
    _source_ray,
)


@dataclass(frozen=True, slots=True)
class OpaqueSourceBoundary:
    """One nonduplicated line in the complete eighteen-row box chart."""

    index: int
    boundary: AffineBoundary
    attached_faces: tuple[tuple[int, int], ...]


def _corner(
    bits: tuple[int, int, int],
    spec: TransportSpec,
    tangent: AcquisitionGeometryTangent,
) -> tuple[Vector3, Vector3]:
    local = cast(
        Vector3,
        tuple(
            spec.grid.origin_mm[i] + bits[i] * spec.grid.shape_xyz[i] * spec.grid.spacing_mm[i]
            for i in range(3)
        ),
    )
    local_dot = cast(
        Vector3,
        tuple(
            tangent.object.origin_dot[i]
            + bits[i] * spec.grid.shape_xyz[i] * tangent.object.spacing_dot[i]
            for i in range(3)
        ),
    )
    point = spec.object_pose.point(local)
    velocity = _add(
        tangent.object.motion.point_velocity(point), spec.object_pose.direction(local_dot)
    )
    if any(not math.isfinite(x) for x in (*point, *velocity)):
        raise TransportError("opaque box corner is not representable in binary64")
    return point, velocity


def source_coordinate_ray(
    source: ParallelBeam | PointConeSource,
    coordinate: tuple[float, float],
    coordinate_dot: tuple[float, float],
) -> RayTangent:
    """Ray jet when varying only the two original emission coordinates."""
    base, displacement, _ = _source_ray(source, coordinate, SourceGeometryTangent())
    if isinstance(source, ParallelBeam):
        return RayTangent(
            base.origin_mm,
            base.direction,
            (
                coordinate_dot[0] * source.extent_xy_mm[0],
                coordinate_dot[1] * source.extent_xy_mm[1],
                0.0,
            ),
        )
    geometry = source.geometry
    displacement_dot = _add(
        _scale(geometry.u, coordinate_dot[0] * geometry.shape[1] * geometry.spacing_mm[0]),
        _scale(geometry.v, coordinate_dot[1] * geometry.shape[0] * geometry.spacing_mm[1]),
    )
    direction, direction_dot = normalised_direction_jet(displacement, displacement_dot)
    return RayTangent(base.origin_mm, direction, direction_dot=direction_dot)


def opaque_material_hit(
    ray: RayTangent,
    spec: TransportSpec,
    *,
    side: int = 0,
    attached_faces: tuple[tuple[int, int], ...] = (),
) -> bool:
    """Half-open forward box ownership, or its exact first-order side limit.

    ``side`` is +1/-1 for the ray's supplied coordinate jet. Lexicographic slab
    roots resolve edge tangencies exactly; an unresolved moving chart is fatal.
    """
    if side not in (-1, 0, 1):
        raise TransportError("opaque source side must be -1, 0 or 1")
    local = ray.in_object_frame(spec.object_pose, AcquisitionGeometryTangent().object.motion)
    constrained = dict(attached_faces)
    if (
        len(constrained) != len(attached_faces)
        or len(constrained) > 2
        or any(axis not in (0, 1, 2) or high not in (0, 1) for axis, high in attached_faces)
    ):
        raise TransportError("opaque attached side chart has invalid face labels")
    source_scale = max(
        abs(value)
        for value in (
            *ray.origin_mm,
            *local.origin_mm,
            *spec.grid.origin_mm,
            *spec.grid.spacing_mm,
            *spec.grid.upper_mm,
            *spec.object_pose.translation_mm,
        )
    )
    resolution = 512.0 * math.ulp(source_scale)
    anchor = 0.0
    if len(attached_faces) == 2:
        edge_roots: list[float] = []
        root_resolution = 0.0
        for axis, high in attached_faces:
            direction = local.direction[axis]
            if direction == 0.0:
                raise TransportError(
                    "opaque edge side chart requires two transverse face crossings"
                )
            location = (spec.grid.origin_mm, spec.grid.upper_mm)[high][axis]
            distance = (location - local.origin_mm[axis]) / direction
            if not math.isfinite(distance):
                raise TransportError("opaque attached edge distance is not representable")
            edge_roots.append(distance)
            root_resolution = max(
                root_resolution, resolution / abs(direction), 512.0 * math.ulp(distance)
            )
        if abs(edge_roots[0] - edge_roots[1]) > root_resolution:
            raise TransportError(
                "opaque source coordinate does not represent its attached box edge"
            )
        anchor = edge_roots[0]
    near = (0.0, 0.0)
    far = (math.inf, 0.0)
    for axis in range(3):
        lo, hi = spec.grid.origin_mm[axis], spec.grid.upper_mm[axis]
        origin, direction = local.origin_mm[axis], local.direction[axis]
        od, dd = local.origin_dot[axis], local.direction_dot[axis]
        if len(attached_faces) == 1 and axis in constrained:
            location = (lo, hi)[constrained[axis]]
            if abs(origin - location) > resolution:
                raise TransportError(
                    "opaque source coordinate does not represent its attached box face"
                )
            origin = location
        if direction == 0.0:
            if origin < lo or origin > hi:
                return False
            if origin == lo and side * od < 0.0:
                return False
            if origin == hi and (side == 0 or side * od >= 0.0):
                return False
            if side != 0 and dd != 0.0 and origin in (lo, hi):
                raise TransportError("opaque parallel-face side needs a complete joint chart")
            continue
        roots: list[tuple[float, float]] = []
        for face in (lo, hi):
            distance = (face - origin) / direction
            if (
                len(attached_faces) == 2
                and axis in constrained
                and face == (lo, hi)[constrained[axis]]
            ):
                distance = anchor
            derivative = (-od - distance * dd) / direction
            if not math.isfinite(distance) or not math.isfinite(derivative):
                raise TransportError("opaque box crossing is not representable in binary64")
            roots.append((distance, side * derivative))
        roots.sort()
        near = max(near, roots[0])
        far = min(far, roots[1])
    if far[0] > near[0]:
        return True
    if far[0] < near[0] or side == 0:
        return False
    return far[1] > near[1]


def opaque_source_boundaries(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    tangent: AcquisitionGeometryTangent,
) -> tuple[OpaqueSourceBoundary, ...]:
    """Prepare box-edge silhouettes and startup faces before drawing samples.

    The full selection measure has twelve edges and six outer-face startup
    rows. Identical projected lines with identical normal velocity are counted
    once; coincident active lines with differing velocity have no regular chart.
    """
    records: list[OpaqueSourceBoundary] = []
    identities: dict[tuple[float, float, float], tuple[float, float, float]] = {}

    def append(
        index: int,
        values: tuple[float, float, float],
        derivatives: tuple[float, float, float],
        attached_faces: tuple[tuple[int, int], ...],
    ) -> None:
        norm = math.hypot(*values[:2])
        if norm == 0.0:
            if values[2] == 0.0 and derivatives != (0.0, 0.0, 0.0):
                raise TransportError("active source launch plane coincides with an opaque box face")
            return
        scale = math.copysign(norm, values[0] if values[0] != 0.0 else values[1])
        identity = cast(tuple[float, float, float], tuple(value / scale for value in values))
        # At g=0, the normal speed is invariant to a changing scalar multiple
        # of g. Compare the two affine speed restrictions on their shared line.
        speed = cast(tuple[float, float, float], tuple(value / scale for value in derivatives))
        previous = identities.get(identity)
        if previous is not None:
            da, db, dc = (speed[i] - previous[i] for i in range(3))
            a, b, c = identity
            if a * dc != da * c or b * dc != db * c or a * db != da * b:
                raise TransportError("coincident moving opaque silhouettes need a joint chart")
            return
        boundary = AffineBoundary(f"opaque:{index}", values, derivatives)
        if boundary.endpoints() is not None:
            identities[identity] = speed
            records.append(OpaqueSourceBoundary(index, boundary, attached_faces))

    index = 0
    local_direction = (
        spec.object_pose.inverse().direction(source.direction)
        if isinstance(source, ParallelBeam)
        else None
    )
    for axis in range(3):
        other = tuple(i for i in range(3) if i != axis)
        for first, second in product((0, 1), repeat=2):
            if local_direction is not None and any(local_direction[i] == 0.0 for i in other):
                index += 1
                continue
            bits = [0, 0, 0]
            bits[other[0]], bits[other[1]] = first, second
            start, start_dot = _corner(cast(tuple[int, int, int], tuple(bits)), spec, tangent)
            bits[axis] = 1
            end, end_dot = _corner(cast(tuple[int, int, int], tuple(bits)), spec, tangent)
            # Only edges between opposing view-facing faces bound a convex
            # box silhouette. Same-facing projected interior edges are not
            # support boundaries and may coincide without an active kink.
            view = (
                _scale(source.direction, -1.0)
                if isinstance(source, ParallelBeam)
                else _sub(source.geometry.source_mm, start)
            )
            signs: list[float] = []
            for normal_axis, high in zip(other, (first, second), strict=True):
                local_normal = cast(
                    Vector3, tuple((2 * high - 1) if i == normal_axis else 0 for i in range(3))
                )
                signs.append(dot(spec.object_pose.direction(local_normal), view))
            if signs[0] * signs[1] > 0.0:
                index += 1
                continue

            def constraint(
                uv: tuple[float, float],
                start: Vector3 = start,
                end: Vector3 = end,
                start_dot: Vector3 = start_dot,
                end_dot: Vector3 = end_dot,
            ) -> tuple[float, float]:
                ray, displacement, displacement_dot = _source_ray(source, uv, tangent.source)
                edge, edge_dot = _sub(end, start), _sub(end_dot, start_dot)
                if isinstance(source, ParallelBeam):
                    normal = cross(edge, ray.direction)
                    normal_dot = _add(
                        cross(edge_dot, ray.direction), cross(edge, ray.direction_dot)
                    )
                    offset, offset_dot = _sub(ray.origin_mm, start), _sub(ray.origin_dot, start_dot)
                else:
                    first_offset, first_dot = (
                        _sub(start, ray.origin_mm),
                        _sub(start_dot, ray.origin_dot),
                    )
                    second_offset, second_dot = (
                        _sub(end, ray.origin_mm),
                        _sub(end_dot, ray.origin_dot),
                    )
                    normal = cross(first_offset, second_offset)
                    normal_dot = _add(
                        cross(first_dot, second_offset), cross(first_offset, second_dot)
                    )
                    offset, offset_dot = displacement, displacement_dot
                return dot(normal, offset), dot(normal_dot, offset) + dot(normal, offset_dot)

            append(
                index,
                _affine_values(lambda uv: constraint(uv)[0]),
                _affine_values(lambda uv: constraint(uv)[1]),
                ((other[0], first), (other[1], second)),
            )
            index += 1
    for axis in range(3):
        for high in (0, 1):
            location = (
                spec.grid.origin_mm[axis]
                + high * spec.grid.shape_xyz[axis] * spec.grid.spacing_mm[axis]
            )
            location_dot = (
                tangent.object.origin_dot[axis]
                + high * spec.grid.shape_xyz[axis] * tangent.object.spacing_dot[axis]
            )

            def startup(
                uv: tuple[float, float],
                axis: int = axis,
                location: float = location,
                location_dot: float = location_dot,
            ) -> tuple[float, float]:
                ray = _source_ray(source, uv, tangent.source)[0].in_object_frame(
                    spec.object_pose, tangent.object.motion
                )
                return ray.origin_mm[axis] - location, ray.origin_dot[axis] - location_dot

            append(
                index,
                _affine_values(lambda uv: startup(uv)[0]),
                _affine_values(lambda uv: startup(uv)[1]),
                ((axis, high),),
            )
            index += 1
    if isinstance(source, ParallelBeam):
        local = _source_ray(source, (0.0, 0.0), tangent.source)[0].in_object_frame(
            spec.object_pose, tangent.object.motion
        )
        for axis in range(3):
            if local.direction[axis] != 0.0 or local.direction_dot[axis] == 0.0:
                continue
            origins = tuple(
                spec.object_pose.inverse().point(
                    _source_ray(source, uv, SourceGeometryTangent())[0].origin_mm
                )[axis]
                for uv in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
            )
            if any(
                all(origin == location for origin in origins)
                for location in (spec.grid.origin_mm[axis], spec.grid.upper_mm[axis])
            ) and _whole_grazing_visible(source, spec, axis):
                raise TransportError(
                    "active angular motion leaves a whole-source opaque grazing face"
                )
        for record in records:
            if len(record.attached_faces) != 1:
                continue
            axis, _ = record.attached_faces[0]
            if (
                local.direction[axis] == 0.0
                and local.direction_dot[axis] != 0.0
                and _grazing_visible(source, record, spec)
            ):
                raise TransportError(
                    "active angular motion leaves an opaque grazing silhouette chart"
                )
    return tuple(records)


def _whole_grazing_visible(source: ParallelBeam, spec: TransportSpec, selected_axis: int) -> bool:
    """Positive original-source-area overlap with a coplanar box face shadow."""
    inverse = spec.object_pose.inverse()
    positions = tuple(
        inverse.point(_source_ray(source, uv, SourceGeometryTangent())[0].origin_mm)
        for uv in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    )
    direction = inverse.direction(source.direction)
    near = [(Fraction(0), Fraction(0), Fraction(0))]
    far: list[tuple[Fraction, Fraction, Fraction]] = []
    constraints: list[tuple[Fraction, Fraction, Fraction]] = []
    for axis in range(3):
        if axis == selected_axis:
            continue
        origin = Fraction(positions[0][axis])
        a, b = Fraction(positions[1][axis]) - origin, Fraction(positions[2][axis]) - origin
        lo, hi = Fraction(spec.grid.origin_mm[axis]), Fraction(spec.grid.upper_mm[axis])
        if direction[axis] == 0:
            constraints.extend(((a, b, origin - lo), (-a, -b, hi - origin)))
        else:
            d = Fraction(direction[axis])
            roots = ((-a / d, -b / d, (lo - origin) / d), (-a / d, -b / d, (hi - origin) / d))
            lower, upper = roots if d > 0 else roots[::-1]
            near.append(lower)
            far.append(upper)
    for exit_ in far:
        for entry in near:
            constraints.append(
                cast(
                    tuple[Fraction, Fraction, Fraction],
                    tuple(exit_[i] - entry[i] for i in range(3)),
                )
            )
    polygon = [
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(1)),
    ]
    for a, b, c in constraints:
        if a == b == 0:
            if c <= 0:
                return False
            continue
        clipped: list[tuple[Fraction, Fraction]] = []
        for previous, current in zip(polygon[-1:] + polygon[:-1], polygon, strict=True):
            first = a * previous[0] + b * previous[1] + c
            last = a * current[0] + b * current[1] + c
            if (first >= 0) != (last >= 0):
                fraction = first / (first - last)
                clipped.append(
                    (
                        previous[0] + fraction * (current[0] - previous[0]),
                        previous[1] + fraction * (current[1] - previous[1]),
                    )
                )
            if last >= 0:
                clipped.append(current)
        polygon = clipped
        if len(polygon) < 3:
            return False
    area_twice = sum(
        first[0] * last[1] - first[1] * last[0]
        for first, last in zip(polygon, polygon[1:] + polygon[:1], strict=True)
    )
    return area_twice != 0


def _grazing_visible(
    source: ParallelBeam, record: OpaqueSourceBoundary, spec: TransportSpec
) -> bool:
    """Exact positive-length intersection of a source line with a face shadow."""
    ends = record.boundary.endpoints()
    if ends is None:
        return False
    inverse = spec.object_pose.inverse()
    first = inverse.point(_source_ray(source, ends[0], SourceGeometryTangent())[0].origin_mm)
    last = inverse.point(_source_ray(source, ends[1], SourceGeometryTangent())[0].origin_mm)
    direction = inverse.direction(source.direction)
    near = [(Fraction(0), Fraction(0))]
    far: list[tuple[Fraction, Fraction]] = []
    constraints: list[tuple[Fraction, Fraction]] = []
    selected_axis = record.attached_faces[0][0]
    for axis in range(3):
        if axis == selected_axis:
            continue
        origin, slope = Fraction(first[axis]), Fraction(last[axis]) - Fraction(first[axis])
        lo, hi = Fraction(spec.grid.origin_mm[axis]), Fraction(spec.grid.upper_mm[axis])
        if direction[axis] == 0.0:
            constraints.extend(((slope, origin - lo), (-slope, hi - origin)))
        else:
            d = Fraction(direction[axis])
            roots = ((-slope / d, (lo - origin) / d), (-slope / d, (hi - origin) / d))
            lower, upper = roots if d > 0 else roots[::-1]
            near.append(lower)
            far.append(upper)
    for exit_a, exit_b in far:
        for entry_a, entry_b in near:
            constraints.append((exit_a - entry_a, exit_b - entry_b))
    lower, upper = Fraction(0), Fraction(1)
    for a, b in constraints:
        if a == 0:
            if b <= 0:
                return False
        elif a > 0:
            lower = max(lower, -b / a)
        else:
            upper = min(upper, -b / a)
    return upper > lower


def opaque_boundary_sides(
    source: ParallelBeam | PointConeSource,
    coordinate: tuple[float, float],
    record: OpaqueSourceBoundary,
    spec: TransportSpec,
) -> tuple[bool, bool]:
    """Material-entry status on negative and positive source-line sides."""
    a, b, _ = record.boundary.coefficients
    norm = math.hypot(a, b)
    ray = source_coordinate_ray(source, coordinate, (a / norm, b / norm))
    return (
        opaque_material_hit(ray, spec, side=-1, attached_faces=record.attached_faces),
        opaque_material_hit(ray, spec, side=1, attached_faces=record.attached_faces),
    )


__all__ = [
    "OpaqueSourceBoundary",
    "opaque_boundary_sides",
    "opaque_material_hit",
    "opaque_source_boundaries",
    "source_coordinate_ray",
]
