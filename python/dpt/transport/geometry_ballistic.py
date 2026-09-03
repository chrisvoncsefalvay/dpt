"""Ballistic generator insertions pulled back to original emitted rays.

For a transverse face crossing, integrating its spatial delta along the
original ray gives 1/abs(direction dot normal). The original source-coordinate
law and weight stay fixed; no source-fluence normalisation or support derivative
is added to this gauge estimator. Grazing rays use whole source-ray side jumps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from dpt.geometry import Vector3, dot

from .geometry_detector_proposals import validate_redundant_planes
from .geometry_interfaces import InterfacePoint, MaterialInterface, interface_point_tangent
from .geometry_tangents import GridGeometryTangent, RayTangent, RigidMotion
from .model import TransportError, TransportSpec


@dataclass(frozen=True, slots=True)
class BallisticInterfaceCrossing:
    point: InterfacePoint
    incoming: RayTangent
    distance_mm: float
    inverse_normal_projection: float


def ballistic_interface_crossings(
    ray: RayTangent,
    spec: TransportSpec,
    tangent: GridGeometryTangent | None = None,
    *,
    source_gauge: RigidMotion | None = None,
    redundant_planes: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[BallisticInterfaceCrossing, ...]:
    """Enumerate every transverse material-face crossing without a shifted origin."""
    redundant_planes = validate_redundant_planes(redundant_planes, spec)
    inverse = spec.object_pose.inverse()
    origin, direction = inverse.point(ray.origin_mm), inverse.direction(ray.direction)
    if any(not math.isfinite(value) for value in (*origin, *direction)) or any(
        not math.isfinite(count * step) or count * step <= 0.0
        for count, step in zip(spec.grid.shape_xyz, spec.grid.spacing_mm, strict=True)
    ):
        raise TransportError("ballistic grid extent/frame is not representable in binary64")
    nx, ny, _ = spec.grid.shape_xyz
    result: list[BallisticInterfaceCrossing] = []
    for axis, count in enumerate(spec.grid.shape_xyz):
        if direction[axis] == 0.0:
            continue
        other_axes = tuple(i for i in range(3) if i != axis)
        for face in range(count + 1):
            if (axis, face) in redundant_planes:
                continue
            location = spec.grid.origin_mm[axis] + face * spec.grid.spacing_mm[axis]
            distance = (location - origin[axis]) / direction[axis]
            if not math.isfinite(distance):
                raise TransportError("ballistic crossing distance is not representable in binary64")
            if distance <= 0.0:
                continue
            local = cast(Vector3, tuple(origin[i] + distance * direction[i] for i in range(3)))
            if any(
                local[i] < spec.grid.origin_mm[i] or local[i] > spec.grid.upper_mm[i]
                for i in other_axes
            ):
                continue
            if source_gauge is not None:
                active_tangent = GridGeometryTangent() if tangent is None else tangent
                point = ray.point(distance)
                local_velocity = [0.0, 0.0, 0.0]
                local_velocity[axis] = (
                    active_tangent.origin_dot[axis] + face * active_tangent.spacing_dot[axis]
                )
                shape_velocity = spec.object_pose.direction(cast(Vector3, tuple(local_velocity)))
                motion_velocity = active_tangent.motion.point_velocity(point)
                velocity = cast(
                    Vector3,
                    tuple(a + b for a, b in zip(shape_velocity, motion_velocity, strict=True)),
                )
                normal = spec.object_pose.direction(
                    cast(Vector3, tuple(float(i == axis) for i in range(3)))
                )
                if (
                    dot(
                        cast(
                            Vector3,
                            tuple(
                                a - b
                                for a, b in zip(
                                    velocity, source_gauge.point_velocity(point), strict=True
                                )
                            ),
                        ),
                        normal,
                    )
                    == 0.0
                ):
                    continue
            cell_indices = [0, 0, 0]
            coordinate: list[float] = []
            tangential: list[int] = []
            for i in other_axes:
                position = (local[i] - spec.grid.origin_mm[i]) / spec.grid.spacing_mm[i]
                if position == math.floor(position):
                    raise TransportError("ballistic crossing lies on an interface-patch edge")
                index = math.floor(position)
                if not 0 <= index < spec.grid.shape_xyz[i]:
                    raise TransportError(
                        "ballistic crossing patch is not representable in binary64"
                    )
                cell_indices[i] = index
                tangential.append(index)
                coordinate.append(position - index)
            cell_indices[axis] = face - 1
            negative = (
                (cell_indices[2] * ny + cell_indices[1]) * nx + cell_indices[0]
                if face > 0
                else None
            )
            cell_indices[axis] = face
            positive = (
                (cell_indices[2] * ny + cell_indices[1]) * nx + cell_indices[0]
                if face < count
                else None
            )
            interface = MaterialInterface(
                axis, face, cast(tuple[int, int], tuple(tangential)), negative, positive
            )
            point = interface_point_tangent(
                interface, cast(tuple[float, float], tuple(coordinate)), spec, tangent
            )
            actual = ray.point(distance)
            scale = max(*(math.ulp(v) for v in (*actual, *point.point_mm)), math.ulp(distance))
            if math.hypot(*(point.point_mm[i] - actual[i] for i in range(3))) > 512.0 * scale:
                raise TransportError(
                    "ballistic crossing world point is not representable in binary64"
                )
            projection = abs(dot(ray.direction, point.normal))
            if projection == 0.0 or not math.isfinite(1.0 / projection):
                raise TransportError("ballistic crossing needs a strictly transverse finite chart")
            result.append(
                BallisticInterfaceCrossing(
                    point, RayTangent(point.point_mm, ray.direction), distance, 1.0 / projection
                )
            )
    result.sort(key=lambda x: x.distance_mm)
    if len(result) > spec.max_crossings:
        raise TransportError("ballistic crossing guard invalidates the entire geometry observation")
    return tuple(result)


__all__ = ["BallisticInterfaceCrossing", "ballistic_interface_crossings"]
