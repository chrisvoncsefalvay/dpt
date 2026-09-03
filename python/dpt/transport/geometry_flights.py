"""Attached analogue flights and exact optical-depth interface continuations.

Scattering and energy coordinates stay fixed. Geometry changes the collision
position inside each material interval and the interval's optical endpoints.
Their signed continuation terms supply the moving interaction-law jumps.
"""

from __future__ import annotations

# These transport modules share their checked vector primitives internally.
# pyright: reportPrivateUsage=false
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .geometry_tangents import (
    GridGeometryTangent,
    RayTangent,
    cell_segments_tangent,
    material_side_local_ray,
)
from .model import TransportError, TransportSpec


@dataclass(frozen=True, slots=True)
class BoxRayInterval:
    entry_mm: float
    entry_dot: float
    exit_mm: float
    exit_dot: float


def material_box_interval(
    ray: RayTangent,
    spec: TransportSpec,
    tangent: GridGeometryTangent | None = None,
    *,
    side: tuple[int, int, Literal[-1, 1]] | None = None,
) -> BoxRayInterval | None:
    """Intersect a full attached ray with the convex grid box.

    A geometrically absent interval remains absent in a neighbourhood on this
    chart. Active tangency is a separate boundary, never a manufactured zero.
    This function does not displace a boundary point by a numerical epsilon.
    """
    tangent = GridGeometryTangent() if tangent is None else tangent
    local = material_side_local_ray(ray, spec.grid, spec.object_pose, tangent, side)
    if side is not None:
        axis, face, sign = side
        if (
            axis not in (0, 1, 2)
            or sign not in (-1, 1)
            or not 0 <= face <= spec.grid.shape_xyz[axis]
        ):
            raise TransportError("one-sided initial ray chart has an invalid material face")
        if local.direction[axis] != 0.0:
            raise TransportError("one-sided initial ray chart must be parallel to its face")
        selected = face - int(sign < 0)
        if not 0 <= selected < spec.grid.shape_xyz[axis]:
            return None
    near: list[tuple[float, float]] = [(0.0, 0.0)]
    far: list[tuple[float, float]] = []
    for axis, count in enumerate(spec.grid.shape_xyz):
        direction, derivative = local.direction[axis], local.direction_dot[axis]
        lower, upper = spec.grid.origin_mm[axis], spec.grid.upper_mm[axis]
        origin = local.origin_mm[axis]
        if direction == 0.0:
            if side is not None and side[0] == axis:
                continue
            if origin < lower or origin > upper:
                return None
            if origin in (lower, upper) and (
                derivative != 0.0
                or (
                    local.origin_dot[axis]
                    != tangent.origin_dot[axis]
                    + (count if origin == upper else 0) * tangent.spacing_dot[axis]
                )
            ):
                raise TransportError("active ray is tangent to material-box support")
            continue
        roots: list[tuple[float, float]] = []
        for index, face in ((0, lower), (count, upper)):
            distance = (face - origin) / direction
            distance_dot = (
                tangent.origin_dot[axis]
                + index * tangent.spacing_dot[axis]
                - local.origin_dot[axis]
                - distance * derivative
            ) / direction
            roots.append((distance, distance_dot))
        roots.sort()
        near.append(roots[0])
        far.append(roots[1])
    entry, entry_dot = max(near)
    exit_, exit_dot = min(far)
    if not all(math.isfinite(value) for value in (entry, entry_dot, exit_, exit_dot)):
        raise TransportError("material box interval is not representable in binary64")
    for candidates, distance, derivative in ((near, entry, entry_dot), (far, exit_, exit_dot)):
        if any(other == distance and other_dot != derivative for other, other_dot in candidates):
            raise TransportError("tied material-box crossings have distinct velocities")
    if exit_ <= entry:
        if exit_ == entry and exit_dot != entry_dot:
            raise TransportError("active material-box tangency needs its own boundary chart")
        return None
    return BoxRayInterval(entry, entry_dot, exit_, exit_dot)


@dataclass(frozen=True, slots=True)
class OpticalDepthSide:
    """One physical conditional continuation at an exact material interface.

    A collision side starts with the given incoming ray and cell and applies
    that cell's conditional physical collision law. An escape side follows the
    incoming ray without a collision. The ray origin is the exact interface;
    no coefficient lookup based on rounded coordinates selects its material.
    """

    kind: Literal["collision", "escape"]
    cell: int | None
    ray: RayTangent


@dataclass(frozen=True, slots=True)
class OpticalDepthBoundary:
    """Signed expectation jump at a moving optical-depth partition endpoint.

    The contribution is ``weight*(before_value-after_value)``. Both sides
    require actual physical continuations, with their own complete addressed
    draw records. They may share random coordinates for a valid coupling.
    """

    key: str
    optical_depth: float
    optical_depth_dot: float
    weight: float
    before: OpticalDepthSide
    after: OpticalDepthSide


@dataclass(frozen=True, slots=True)
class OpticalFlightTangent:
    """One fixed-exponential-depth analogue flight plus all its support terms."""

    terminal: Literal["collision", "escape"]
    cell: int | None
    ray: RayTangent
    distance_mm: float | None
    distance_dot: float
    total_optical_depth: float
    total_optical_depth_dot: float
    boundaries: tuple[OpticalDepthBoundary, ...]


def _endpoint_ray(ray: RayTangent, distance: float, derivative: float) -> RayTangent:
    return RayTangent(
        ray.point(distance),
        ray.direction,
        ray.point_velocity(distance, derivative),
        ray.direction_dot,
    )


def _boundary_weight(depth: float, derivative: float) -> float:
    if derivative == 0.0:
        return 0.0
    try:
        return math.copysign(math.exp(math.log(abs(derivative)) - depth), derivative)
    except OverflowError as error:
        raise TransportError("optical interface continuation weight overflows binary64") from error


def optical_flight_tangent(
    ray: RayTangent,
    spec: TransportSpec,
    *,
    extinction_mm_inverse: Sequence[float],
    optical_depth: float,
    tangent: GridGeometryTangent | None = None,
    material_side: tuple[int, int, Literal[-1, 1]] | None = None,
) -> OpticalFlightTangent:
    """Reparameterise a flight and expose every moving conditional-law boundary.

    Conditional on its fixed incoming energy and material labels, extinction
    rates stay fixed under geometry. In a positive-rate interval, the collision
    distance jet is ``entry_dot - preceding_depth_dot/rate``. At optical-depth
    interfaces the missing derivative is ``exp(-tau)*tau_dot*(F_left-F_right)``.
    Vacuum gaps keep both endpoint positions: collapsing them to one interface
    would lose the jump of the inverse optical-depth map.

    A guard failure invalidates the entire caller history. A draw exactly on a
    partition endpoint is rejected rather than assigned an arbitrary side.
    Boundary terms are retained even when the particular draw absorbs/escapes;
    they are terms of the conditional flight expectation, not realised events.
    """
    tangent = GridGeometryTangent() if tangent is None else tangent
    rates = tuple(extinction_mm_inverse)
    if len(rates) != math.prod(spec.grid.shape) or any(
        not math.isfinite(rate) or rate < 0.0 for rate in rates
    ):
        raise TransportError("flight extinction must be a finite nonnegative rate per cell")
    if not math.isfinite(optical_depth) or optical_depth <= 0.0:
        raise TransportError("flight optical depth must be finite and strictly positive")
    interval = material_box_interval(ray, spec, tangent, side=material_side)
    if interval is None:
        return OpticalFlightTangent("escape", None, ray, None, 0.0, 0.0, 0.0, ())
    segments = cell_segments_tangent(
        ray,
        spec.grid,
        spec.object_pose,
        tangent,
        maximum_distance_mm=interval.exit_mm,
        maximum_distance_dot=interval.exit_dot,
        side=material_side,
    )
    if len(segments) > spec.max_crossings:
        raise TransportError("crossing guard invalidates the complete geometry history")
    positive = tuple(segment for segment in segments if rates[segment.cell] > 0.0)
    if not positive:
        return OpticalFlightTangent("escape", None, ray, None, 0.0, 0.0, 0.0, ())
    depth, depth_dot = 0.0, 0.0
    boundaries: list[OpticalDepthBoundary] = []
    selected: tuple[int, float, float] | None = None
    for index, segment in enumerate(positive):
        rate = rates[segment.cell]
        following_depth = depth + rate * segment.length_mm
        following_dot = depth_dot + rate * segment.length_dot
        if not math.isfinite(following_depth) or not math.isfinite(following_dot):
            raise TransportError("flight optical partition is not representable in binary64")
        if optical_depth == depth or optical_depth == following_depth:
            raise TransportError("flight draw lies exactly on an optical partition boundary")
        if depth < optical_depth < following_depth:
            selected = (
                segment.cell,
                segment.entry_mm + (optical_depth - depth) / rate,
                segment.entry_dot - depth_dot / rate,
            )
        left_ray = _endpoint_ray(ray, segment.exit_mm, segment.exit_dot)
        before = OpticalDepthSide("collision", segment.cell, left_ray)
        if index + 1 < len(positive):
            next_segment = positive[index + 1]
            right_ray = _endpoint_ray(ray, next_segment.entry_mm, next_segment.entry_dot)
            after = OpticalDepthSide("collision", next_segment.cell, right_ray)
        else:
            after = OpticalDepthSide("escape", None, left_ray)
        boundaries.append(
            OpticalDepthBoundary(
                f"flight:{index}",
                following_depth,
                following_dot,
                _boundary_weight(following_depth, following_dot),
                before,
                after,
            )
        )
        depth, depth_dot = following_depth, following_dot
    if selected is None:
        return OpticalFlightTangent(
            "escape", None, ray, None, 0.0, depth, depth_dot, tuple(boundaries)
        )
    cell, distance, derivative = selected
    collision = _endpoint_ray(ray, distance, derivative)
    return OpticalFlightTangent(
        "collision", cell, collision, distance, derivative, depth, depth_dot, tuple(boundaries)
    )


def optical_boundary_observation(
    boundary: OpticalDepthBoundary,
    *,
    before_value: float,
    after_value: float,
) -> float:
    """Apply an actual signed continuation jump; missing sides cannot become zero."""
    if not all(math.isfinite(value) for value in (before_value, after_value)):
        raise TransportError("optical boundary continuations must return finite complete scores")
    result = boundary.weight * (before_value - after_value)
    if not math.isfinite(result):
        raise TransportError("optical boundary contribution is not representable in binary64")
    return result


__all__ = [
    "BoxRayInterval",
    "OpticalDepthBoundary",
    "OpticalDepthSide",
    "OpticalFlightTangent",
    "material_box_interval",
    "optical_boundary_observation",
    "optical_flight_tangent",
]
