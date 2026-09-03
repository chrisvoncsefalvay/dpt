"""Geometry jets and explicit boundary measures for transport integrals.

Distances are in mm. A dot denotes a directional derivative with respect to
the caller's coordinate, whose units therefore also determine tangent units.
These dependency-free routines are shared by transport references and boundary
samplers; they do not differentiate a realised analogue collision sequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal, cast

from dpt.geometry import RigidTransform, Vector3, cross, dot, matvec, transpose, vector3

from .model import MaterialGrid, OrientedDetector, PlanarDetector, TransportError

ZERO: Vector3 = (0.0, 0.0, 0.0)


def _add(left: Vector3, right: Vector3) -> Vector3:
    return cast(Vector3, tuple(a + b for a, b in zip(left, right, strict=True)))


def _sub(left: Vector3, right: Vector3) -> Vector3:
    return cast(Vector3, tuple(a - b for a, b in zip(left, right, strict=True)))


def _scale(value: Vector3, scale: float) -> Vector3:
    return cast(Vector3, tuple(scale * a for a in value))


@dataclass(frozen=True, slots=True)
class RigidMotion:
    """Spatial rigid velocity ``translation + angular cross (point-pivot)``.

    Translation is mm per active coordinate, angular velocity radians per
    active coordinate, and the pivot is a world point in mm. ``point_velocity``
    and ``direction_velocity`` describe the same infinitesimal rigid motion.
    Object, source and detector can each receive a separate motion.
    """

    translation: Vector3 = ZERO
    angular: Vector3 = ZERO
    pivot_mm: Vector3 = ZERO

    def __post_init__(self) -> None:
        for name in ("translation", "angular", "pivot_mm"):
            object.__setattr__(self, name, vector3(getattr(self, name), name))

    def point_velocity(self, point: Vector3) -> Vector3:
        return _add(self.translation, cross(self.angular, _sub(point, self.pivot_mm)))

    def direction_velocity(self, direction: Vector3) -> Vector3:
        return cross(self.angular, direction)


@dataclass(frozen=True, slots=True)
class RayTangent:
    """World ray ``origin + distance*direction`` and its attached geometry jet.

    ``direction`` is unit length and its tangent is orthogonal to it, so
    distance remains a physical path length. A moving source changes both
    ray origin and direction; neither tangent is detached during propagation.
    """

    origin_mm: Vector3
    direction: Vector3
    origin_dot: Vector3 = ZERO
    direction_dot: Vector3 = ZERO

    def __post_init__(self) -> None:
        for name in ("origin_mm", "direction", "origin_dot", "direction_dot"):
            object.__setattr__(self, name, vector3(getattr(self, name), name))
        if abs(dot(self.direction, self.direction) - 1.0) > 1e-12:
            raise TransportError("ray tangent requires a unit direction")
        tangent_scale = max(abs(value) for value in self.direction_dot)
        if tangent_scale > 1e150:
            scaled = cast(Vector3, tuple(value / tangent_scale for value in self.direction_dot))
            nonorthogonal = abs(dot(self.direction, scaled)) > 1e-12 * math.hypot(*scaled)
        else:
            nonorthogonal = abs(dot(self.direction, self.direction_dot)) > 1e-12 * max(
                1.0, math.hypot(*self.direction_dot)
            )
        if nonorthogonal:
            raise TransportError("the unit-direction tangent must be orthogonal to direction")

    def point(self, distance_mm: float) -> Vector3:
        return _add(self.origin_mm, _scale(self.direction, distance_mm))

    def point_velocity(self, distance_mm: float, distance_dot: float = 0.0) -> Vector3:
        return _add(
            _add(self.origin_dot, _scale(self.direction_dot, distance_mm)),
            _scale(self.direction, distance_dot),
        )

    def in_object_frame(self, pose: RigidTransform, motion: RigidMotion) -> RayTangent:
        """Transform the entire jet, including motion of the object frame."""
        inverse = transpose(pose.rotation)
        return RayTangent(
            matvec(inverse, _sub(self.origin_mm, pose.translation_mm)),
            matvec(inverse, self.direction),
            matvec(inverse, _sub(self.origin_dot, motion.point_velocity(self.origin_mm))),
            matvec(
                inverse,
                _sub(self.direction_dot, motion.direction_velocity(self.direction)),
            ),
        )


def normalised_direction_jet(
    displacement: Vector3, displacement_dot: Vector3
) -> tuple[Vector3, Vector3]:
    """Return ``r/|r|`` and ``(I-ddᵀ)*r_dot/|r|`` for a finite segment."""
    length = math.hypot(*displacement)
    if not math.isfinite(length) or length == 0.0:
        raise TransportError("a geometry segment must have finite positive length")
    direction = _scale(displacement, 1.0 / length)
    tangent = _scale(
        _sub(displacement_dot, _scale(direction, dot(direction, displacement_dot))),
        1.0 / length,
    )
    return direction, tangent


@dataclass(frozen=True, slots=True)
class PlaneHitTangent:
    """Outgoing detector hit in pixel coordinates, with a full attached jet."""

    distance_mm: float
    distance_dot: float
    column: float
    row: float
    column_dot: float
    row_dot: float
    pixel: int


def detector_corner_velocity(
    detector: PlanarDetector | OrientedDetector,
    motion: RigidMotion,
    spacing_dot: tuple[float, float],
) -> Vector3:
    """Respect lower-face PlanarDetector versus centre-based DetectorGeometry fields."""
    velocity = motion.point_velocity(detector.lower_corner_mm)
    if isinstance(detector, OrientedDetector):
        velocity = _sub(
            velocity,
            _add(
                _scale(detector.u, 0.5 * spacing_dot[0]), _scale(detector.v, 0.5 * spacing_dot[1])
            ),
        )
    return velocity


def detector_hit_tangent(
    ray: RayTangent,
    detector: PlanarDetector | OrientedDetector,
    motion: RigidMotion | None = None,
    *,
    spacing_dot: tuple[float, float] = (0.0, 0.0),
) -> PlaneHitTangent:
    """Differentiate plane intersection, basis motion and pixel spacing.

    Pixel classification is half-open and deliberately has no pathwise
    derivative. Its jumps belong to a separately sampled boundary integral.
    Rays parallel to the detector plane or travelling away from it are outside
    this hit chart. This routine raises rather than inventing a zero tangent.
    """
    motion = RigidMotion() if motion is None else motion
    if len(spacing_dot) != 2 or not all(math.isfinite(value) for value in spacing_dot):
        raise TransportError("detector spacing tangent must have two finite components")
    corner, normal = detector.lower_corner_mm, detector.normal
    corner_dot = detector_corner_velocity(detector, motion, spacing_dot)
    normal_dot = motion.direction_velocity(normal)
    numerator = dot(_sub(corner, ray.origin_mm), normal)
    denominator = dot(ray.direction, normal)
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise TransportError("detector hit chart requires an outgoing transverse ray")
    distance = numerator / denominator
    if not math.isfinite(distance) or distance <= 0.0:
        raise TransportError("detector hit chart requires a finite forward intersection")
    numerator_dot = dot(_sub(corner_dot, ray.origin_dot), normal) + dot(
        _sub(corner, ray.origin_mm), normal_dot
    )
    denominator_dot = dot(ray.direction_dot, normal) + dot(ray.direction, normal_dot)
    distance_dot = (numerator_dot - distance * denominator_dot) / denominator
    displacement = _sub(ray.point(distance), corner)
    displacement_dot = _sub(ray.point_velocity(distance, distance_dot), corner_dot)
    coordinates: list[float] = []
    tangents: list[float] = []
    for axis, spacing, step_dot in zip(
        (detector.u, detector.v), detector.spacing_xy_mm, spacing_dot, strict=True
    ):
        coordinate = dot(displacement, axis) / spacing
        tangent = (
            dot(displacement_dot, axis)
            + dot(displacement, motion.direction_velocity(axis))
            - coordinate * step_dot
        ) / spacing
        coordinates.append(coordinate)
        tangents.append(tangent)
    column, row = coordinates
    width, height = detector.shape
    pixel = -1
    if 0.0 <= column < width and 0.0 <= row < height:
        pixel = math.floor(row) * width + math.floor(column)
    values = (distance, distance_dot, column, row, *tangents)
    if not all(math.isfinite(value) for value in values):
        raise TransportError("detector geometry jet is not representable in binary64")
    return PlaneHitTangent(distance, distance_dot, column, row, tangents[0], tangents[1], pixel)


@dataclass(frozen=True, slots=True)
class CellSegmentTangent:
    """One fixed-topology cell interval and its moving entry/exit distances."""

    cell: int
    entry_mm: float
    exit_mm: float
    entry_dot: float
    exit_dot: float

    @property
    def length_mm(self) -> float:
        return self.exit_mm - self.entry_mm

    @property
    def length_dot(self) -> float:
        return self.exit_dot - self.entry_dot


@dataclass(frozen=True, slots=True)
class GridGeometryTangent:
    """Object rigid motion and local origin/spacing velocities.

    Local face ``origin[axis] + index*spacing[axis]`` moves at
    ``origin_dot[axis] + index*spacing_dot[axis]``. Labels and grid shape remain
    fixed. A positive spacing stays positive in a neighbourhood of the point.
    """

    motion: RigidMotion = field(default_factory=RigidMotion)
    origin_dot: Vector3 = ZERO
    spacing_dot: Vector3 = ZERO

    def __post_init__(self) -> None:
        for name in ("origin_dot", "spacing_dot"):
            object.__setattr__(self, name, vector3(getattr(self, name), name))


def material_side_local_ray(
    ray: RayTangent,
    grid: MaterialGrid,
    pose: RigidTransform,
    tangent: GridGeometryTangent,
    side: tuple[int, int, Literal[-1, 1]] | None,
) -> RayTangent:
    """Resolve a labelled grazing constraint in the local material chart.

    The face label fixes the normal component exactly. A rotated world proxy
    may acquire roundoff on the inverse transform; only that representational
    residual is removed. The original world ray and its random marks remain
    unchanged. An unlabelled ray uses the ordinary transform.
    """
    local = ray.in_object_frame(pose, tangent.motion)
    if side is None:
        return local
    axis, face, sign = side
    if axis not in (0, 1, 2) or sign not in (-1, 1) or not 0 <= face <= grid.shape_xyz[axis]:
        raise TransportError("one-sided initial ray chart has an invalid material face")
    location = grid.origin_mm[axis] + face * grid.spacing_mm[axis]
    origin_scale = max(1.0, *(abs(x) for x in (*ray.origin_mm, *local.origin_mm, location)))
    if abs(local.origin_mm[axis] - location) > 512 * math.ulp(origin_scale):
        raise TransportError("one-sided ray origin does not represent its attached material face")
    if abs(local.direction[axis]) > 512 * math.ulp(1.0):
        raise TransportError("one-sided initial ray chart must be parallel to its face")
    if local.direction[axis] == 0.0:
        return local
    direction = cast(Vector3, tuple(0.0 if i == axis else x for i, x in enumerate(local.direction)))
    return RayTangent(local.origin_mm, direction, local.origin_dot, local.direction_dot)


def cell_segments_tangent(
    ray: RayTangent,
    grid: MaterialGrid,
    pose: RigidTransform | None = None,
    tangent: GridGeometryTangent | None = None,
    *,
    maximum_distance_mm: float,
    maximum_distance_dot: float = 0.0,
    side: tuple[int, int, Literal[-1, 1]] | None = None,
) -> tuple[CellSegmentTangent, ...]:
    """Enumerate material intervals and analytic face-crossing derivatives.

    This is an interior chart, not the derivative of detector visibility or
    the missing material silhouette measure. ``side`` evaluates the exact
    one-sided limit at a ray lying in one grid face, without moving coordinates
    by an arbitrary epsilon. Tied transverse crossings with different crossing
    velocities require a separate boundary chart and raise here.
    """
    pose = RigidTransform() if pose is None else pose
    tangent = GridGeometryTangent() if tangent is None else tangent
    if not math.isfinite(maximum_distance_mm) or maximum_distance_mm <= 0.0:
        raise TransportError("maximum ray distance must be finite and positive")
    if not math.isfinite(maximum_distance_dot):
        raise TransportError("maximum ray-distance tangent must be finite")
    local = material_side_local_ray(ray, grid, pose, tangent, side)
    boundaries: list[tuple[float, float]] = [
        (0.0, 0.0),
        (maximum_distance_mm, maximum_distance_dot),
    ]
    for axis, count in enumerate(grid.shape_xyz):
        direction = local.direction[axis]
        if direction == 0.0:
            continue
        for index in range(count + 1):
            face = grid.origin_mm[axis] + index * grid.spacing_mm[axis]
            distance = (face - local.origin_mm[axis]) / direction
            if 0.0 < distance < maximum_distance_mm:
                face_dot = tangent.origin_dot[axis] + index * tangent.spacing_dot[axis]
                distance_dot = (
                    face_dot - local.origin_dot[axis] - distance * local.direction_dot[axis]
                ) / direction
                if not math.isfinite(distance_dot):
                    raise TransportError("grid crossing derivative is not representable")
                boundaries.append((distance, distance_dot))
    boundaries.sort()
    unique: list[tuple[float, float]] = []
    for distance, derivative in boundaries:
        if unique and distance == unique[-1][0]:
            if derivative != unique[-1][1]:
                raise TransportError("a tied crossing has distinct moving-face velocities")
        else:
            unique.append((distance, derivative))
    segments: list[CellSegmentTangent] = []
    nx, ny, nz = grid.shape_xyz
    for (entry, entry_dot), (exit_, exit_dot) in pairwise(unique):
        midpoint = local.point(entry + (exit_ - entry) / 2.0)
        indices = [
            math.floor((midpoint[axis] - grid.origin_mm[axis]) / grid.spacing_mm[axis])
            for axis in range(3)
        ]
        if side is not None:
            axis, face_index, sign = side
            if axis not in (0, 1, 2) or sign not in (-1, 1):
                raise TransportError("one-sided face chart has an invalid axis or side")
            if local.direction[axis] != 0.0:
                raise TransportError("one-sided face chart requires a ray parallel to its face")
            indices[axis] = face_index - int(sign < 0)
        x, y, z = indices
        if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
            segments.append(
                CellSegmentTangent((z * ny + y) * nx + x, entry, exit_, entry_dot, exit_dot)
            )
    return tuple(segments)


@dataclass(frozen=True, slots=True)
class AffineBoundary:
    """A moving line ``a*u+b*v+c=0`` in a unit-square source chart.

    The positive-side contribution uses ``g_dot * (f_positive-f_negative)``.
    With ``r`` uniform along the clipped line and length ``L``, the unbiased
    sample is ``L*g_dot(r)*(f_positive-f_negative)/hypot(a,b)``. This directly
    samples the distributional boundary term; it is not a finite difference.
    """

    key: str
    coefficients: tuple[float, float, float]
    coefficients_dot: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in ("coefficients", "coefficients_dot"):
            values = tuple(getattr(self, name))
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise TransportError("an affine boundary requires three finite coefficients")
            object.__setattr__(self, name, values)
        if math.hypot(*self.coefficients[:2]) == 0.0:
            raise TransportError("boundary chart has no transverse source coordinate")

    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        a, b, c = self.coefficients
        points: list[tuple[float, float]] = []
        if b != 0.0:
            for u in (0.0, 1.0):
                v = -(a * u + c) / b
                if 0.0 <= v <= 1.0:
                    points.append((u, v))
        if a != 0.0:
            for v in (0.0, 1.0):
                u = -(b * v + c) / a
                if 0.0 <= u <= 1.0:
                    points.append((u, v))
        unique = tuple(dict.fromkeys(points))
        if len(unique) < 2:
            return None
        # A boundary coinciding with source support requires one-sided source
        # calculus. A two-sided derivative cannot be manufactured from a half.
        if (a == 0.0 and -c / b in (0.0, 1.0)) or (b == 0.0 and -c / a in (0.0, 1.0)):
            a_dot, b_dot, c_dot = self.coefficients_dot
            stationary = (
                a_dot == 0.0 and math.fsum((b_dot * (-c / b), c_dot)) == 0.0
                if a == 0.0
                else b_dot == 0.0 and math.fsum((a_dot * (-c / a), c_dot)) == 0.0
            )
            if stationary:
                return None
            raise TransportError("a moving boundary coincides with source support")
        return unique[0], unique[-1]

    def sample(self, coordinate: float) -> tuple[tuple[float, float], float] | None:
        """Return a source point and signed coarea weight for one uniform draw."""
        if not math.isfinite(coordinate) or not 0.0 < coordinate < 1.0:
            raise TransportError("boundary sample coordinate must lie strictly inside (0,1)")
        ends = self.endpoints()
        if ends is None:
            return None
        first, last = ends
        point = tuple(first[i] + coordinate * (last[i] - first[i]) for i in (0, 1))
        length = math.hypot(last[0] - first[0], last[1] - first[1])
        g_dot = math.fsum(
            (
                self.coefficients_dot[0] * point[0],
                self.coefficients_dot[1] * point[1],
                self.coefficients_dot[2],
            )
        )
        return cast(tuple[float, float], point), length * g_dot / math.hypot(*self.coefficients[:2])


__all__ = [
    "AffineBoundary",
    "CellSegmentTangent",
    "GridGeometryTangent",
    "PlaneHitTangent",
    "RayTangent",
    "RigidMotion",
    "cell_segments_tangent",
    "detector_corner_velocity",
    "detector_hit_tangent",
    "normalised_direction_jet",
]
