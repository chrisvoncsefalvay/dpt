"""Attached collision-volume and source measures for geometry differentiation.

Collision vertices use a fixed unit-cube parameterisation of the material grid.
Labels consequently remain fixed while points, flight directions, attenuation
segments and all volume/area Jacobians move. Source-support discontinuities
are integrated on separately sampled planes in this fixed cube.
"""

from __future__ import annotations

# The geometry modules share their checked vector arithmetic internally.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass
from itertools import product
from typing import cast

from dpt.geometry import Vector3, cross, dot

from .geometry_tangents import (
    GridGeometryTangent,
    RayTangent,
    RigidMotion,
    _add,
    _scale,
    _sub,
    detector_corner_velocity,
    normalised_direction_jet,
)
from .model import OrientedDetector, PlanarDetector, TransportError, TransportSpec
from .source import ParallelBeam, PointConeSource
from .uncollided_geometry import SourceGeometryTangent


@dataclass(frozen=True, slots=True)
class VertexTangent:
    """An attached world vertex with its fixed material-cell index."""

    point_mm: Vector3
    point_dot: Vector3
    cell: int
    volume_jacobian: float
    volume_jacobian_dot: float


def collision_vertex_tangent(
    coordinate: Vector3,
    spec: TransportSpec,
    tangent: GridGeometryTangent | None = None,
) -> VertexTangent:
    """Map a fixed unit-cube vertex to world coordinates and material volume.

    The Jacobian is the complete box volume in mm cubed. Rotations preserve it;
    local spacing changes contribute its trace derivative. Source/interface
    boundary vertices may lie on cube faces; ordinary draws are interior.
    """
    tangent = GridGeometryTangent() if tangent is None else tangent
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinate):
        raise TransportError("collision coordinates must lie in the closed unit cube")
    grid, pose = spec.grid, spec.object_pose
    extents = tuple(
        count * spacing for count, spacing in zip(grid.shape_xyz, grid.spacing_mm, strict=True)
    )
    local = cast(Vector3, tuple(grid.origin_mm[i] + coordinate[i] * extents[i] for i in range(3)))
    local_dot = cast(
        Vector3,
        tuple(
            tangent.origin_dot[i] + coordinate[i] * grid.shape_xyz[i] * tangent.spacing_dot[i]
            for i in range(3)
        ),
    )
    point = pose.point(local)
    point_dot = _add(tangent.motion.point_velocity(point), pose.direction(local_dot))
    indices = tuple(
        min(count - 1, math.floor(value * count))
        for value, count in zip(coordinate, grid.shape_xyz, strict=True)
    )
    nx, ny, _ = grid.shape_xyz
    cell = (indices[2] * ny + indices[1]) * nx + indices[0]
    jacobian = math.prod(extents)
    derivative = jacobian * math.fsum(
        step_dot / step for step_dot, step in zip(tangent.spacing_dot, grid.spacing_mm, strict=True)
    )
    if not all(math.isfinite(value) for value in (*point, *point_dot, jacobian, derivative)):
        raise TransportError("collision-volume geometry is not representable in binary64")
    return VertexTangent(point, point_dot, cell, jacobian, derivative)


@dataclass(frozen=True, slots=True)
class DetectorVertexTangent:
    """Attached detector-area endpoint, pixel identity and area Jacobian."""

    point_mm: Vector3
    point_dot: Vector3
    normal: Vector3
    normal_dot: Vector3
    pixel: int
    area_jacobian: float
    area_jacobian_dot: float


def detector_vertex_tangent(
    coordinate: tuple[float, float],
    detector: PlanarDetector | OrientedDetector,
    motion: RigidMotion | None = None,
    spacing_dot: tuple[float, float] = (0.0, 0.0),
) -> DetectorVertexTangent:
    """Parameterise detector support so moving edges remain in the measure.

    The metadata's origin is the spacing anchor. Pixel identity is fixed in this
    reference square; its changing area and world location both participate in
    the derivative. A path integral using this chart needs no detached hit-bin
    derivative or epsilon around moving detector edges.
    """
    motion = RigidMotion() if motion is None else motion
    if not all(math.isfinite(x) and 0.0 < x < 1.0 for x in coordinate):
        raise TransportError("detector-area coordinates must lie inside the unit square")
    if len(spacing_dot) != 2 or not all(math.isfinite(x) for x in spacing_dot):
        raise TransportError("detector spacing tangent must have two finite components")
    width, height = detector.shape
    step_x, step_y = detector.spacing_xy_mm
    amount_x, amount_y = coordinate[0] * width, coordinate[1] * height
    point = _add(
        detector.lower_corner_mm,
        _add(_scale(detector.u, amount_x * step_x), _scale(detector.v, amount_y * step_y)),
    )
    correction = _sub(
        detector_corner_velocity(detector, motion, spacing_dot),
        motion.point_velocity(detector.lower_corner_mm),
    )
    point_dot = _add(
        correction,
        _add(
            motion.point_velocity(point),
            _add(
                _scale(detector.u, amount_x * spacing_dot[0]),
                _scale(detector.v, amount_y * spacing_dot[1]),
            ),
        ),
    )
    area = width * height * step_x * step_y
    area_dot = area * (spacing_dot[0] / step_x + spacing_dot[1] / step_y)
    pixel = math.floor(amount_y) * width + math.floor(amount_x)
    return DetectorVertexTangent(
        point,
        point_dot,
        detector.normal,
        motion.direction_velocity(detector.normal),
        pixel,
        area,
        area_dot,
    )


@dataclass(frozen=True, slots=True)
class SourceFluxTangent:
    """Incident fluence density at a collision vertex and its attached jet.

    Density is source weight per mm squared before attenuation. Four signed
    support functions are ordered lower-u, upper-u, lower-v, upper-v; their
    positive intersection defines the original source footprint. The smooth
    density is retained outside support solely for its one-sided boundary
    continuation. ``visible`` reports whether all support functions are positive.
    """

    incoming: RayTangent
    distance_mm: float
    distance_dot: float
    density: float
    density_dot: float
    support: tuple[float, float, float, float]
    support_dot: tuple[float, float, float, float]

    @property
    def visible(self) -> bool:
        return all(value > 0.0 for value in self.support)


def source_flux_tangent(
    source: ParallelBeam | PointConeSource,
    point_mm: Vector3,
    point_dot: Vector3,
    tangent: SourceGeometryTangent | None = None,
) -> SourceFluxTangent:
    """Evaluate the original area-normalised source law at a moving vertex.

    Parallel fluence is ``weight/(source_area*abs(direction.normal))``.
    A cone aims uniformly at its declared target area, giving angular density
    ``target_distance**2/(target_area*cosine)`` and fluence that density divided
    by collision distance squared. All source, target, collision and direction
    derivatives are attached. Hard source support is handled independently.
    """
    tangent = SourceGeometryTangent() if tangent is None else tangent
    if isinstance(source, ParallelBeam):
        ex, ey = source.extent_xy_mm
        if ex <= 0.0 or ey <= 0.0 or source.direction[2] == 0.0:
            raise TransportError("volume source measure requires positive transverse source area")
        normal: Vector3 = (0.0, 0.0, 1.0 if source.direction[2] > 0.0 else -1.0)
        normal_dot = tangent.motion.direction_velocity(normal)
        direction, direction_dot = (
            source.direction,
            _add(tangent.motion.direction_velocity(source.direction), tangent.direction_dot),
        )
        lower, lower_dot = source.lower_mm, tangent.motion.point_velocity(source.lower_mm)
        offset, offset_dot = _sub(point_mm, lower), _sub(point_dot, lower_dot)
        numerator = dot(offset, normal)
        numerator_dot = dot(offset_dot, normal) + dot(offset, normal_dot)
        denominator = dot(direction, normal)
        denominator_dot = dot(direction_dot, normal) + dot(direction, normal_dot)
        distance = numerator / denominator
        distance_dot = (numerator_dot - distance * denominator_dot) / denominator
        if distance <= 0.0 or not math.isfinite(distance):
            raise TransportError("collision vertex must lie downstream of the source plane")
        launch = _sub(point_mm, _scale(direction, distance))
        launch_dot = _sub(
            point_dot, _add(_scale(direction_dot, distance), _scale(direction, distance_dot))
        )
        source_offset = _sub(launch, lower)
        source_offset_dot = _sub(launch_dot, lower_dot)
        coordinates: list[float] = []
        derivatives: list[float] = []
        for basis, extent, extent_dot in zip(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (ex, ey), tangent.extent_dot, strict=True
        ):
            coordinate = dot(source_offset, basis) / extent
            derivative = (
                dot(source_offset_dot, basis)
                + dot(source_offset, tangent.motion.direction_velocity(basis))
                - coordinate * extent_dot
            ) / extent
            coordinates.append(coordinate)
            derivatives.append(derivative)
        area = ex * ey
        area_dot = area * (tangent.extent_dot[0] / ex + tangent.extent_dot[1] / ey)
        density = source.weight / (area * denominator)
        density_dot = -density * (area_dot / area + denominator_dot / denominator)
        ray = RayTangent(launch, direction, launch_dot, direction_dot)
    else:
        if tangent.extent_dot != (0.0, 0.0):
            raise TransportError("cone extents belong to the explicit launch detector")
        if tangent.direction_dot != (0.0, 0.0, 0.0):
            raise TransportError("cone directions belong to its emitting point and launch target")
        geometry = source.geometry
        position, position_dot = (
            geometry.source_mm,
            tangent.motion.point_velocity(geometry.source_mm),
        )
        displacement, displacement_dot = _sub(point_mm, position), _sub(point_dot, position_dot)
        direction, direction_dot = normalised_direction_jet(displacement, displacement_dot)
        distance = math.hypot(*displacement)
        distance_dot = dot(direction, displacement_dot)
        ray = RayTangent(position, direction, position_dot, direction_dot)
        from .geometry_tangents import detector_hit_tangent

        detector = OrientedDetector(geometry)
        hit = detector_hit_tangent(
            ray, detector, tangent.target_motion, spacing_dot=tangent.target_spacing_dot
        )
        normal_dot = tangent.target_motion.direction_velocity(detector.normal)
        projection = dot(direction, detector.normal)
        projection_dot = dot(direction_dot, detector.normal) + dot(direction, normal_dot)
        area = math.prod(detector.shape) * math.prod(detector.spacing_xy_mm)
        area_score = math.fsum(
            velocity / spacing
            for velocity, spacing in zip(
                tangent.target_spacing_dot, detector.spacing_xy_mm, strict=True
            )
        )
        density = source.weight * (hit.distance_mm / distance) ** 2 / (area * projection)
        density_dot = density * (
            2.0 * hit.distance_dot / hit.distance_mm
            - 2.0 * distance_dot / distance
            - projection_dot / projection
            - area_score
        )
        coordinates = [hit.column / detector.shape[0], hit.row / detector.shape[1]]
        derivatives = [hit.column_dot / detector.shape[0], hit.row_dot / detector.shape[1]]
    u, v = coordinates
    u_dot, v_dot = derivatives
    return SourceFluxTangent(
        ray,
        distance,
        distance_dot,
        density,
        density_dot,
        (u, 1.0 - u, v, 1.0 - v),
        (u_dot, -u_dot, v_dot, -v_dot),
    )


@dataclass(frozen=True, slots=True)
class VolumeBoundary:
    """A moving plane in a unit-cube collision chart, with an exact area sampler."""

    key: str
    normal: Vector3
    offset: float
    normal_dot: Vector3
    offset_dot: float

    def __post_init__(self) -> None:
        if (
            not all(
                math.isfinite(value)
                for value in (*self.normal, self.offset, *self.normal_dot, self.offset_dot)
            )
            or math.hypot(*self.normal) == 0.0
        ):
            raise TransportError("volume boundary requires a finite transverse plane")

    def polygon(self) -> tuple[Vector3, ...]:
        """Intersect the plane with all twelve cube edges and order its polygon."""
        points: list[Vector3] = []
        for axis in range(3):
            others = tuple(i for i in range(3) if i != axis)
            for fixed in product((0.0, 1.0), repeat=2):
                origin = [0.0, 0.0, 0.0]
                for other, value in zip(others, fixed, strict=True):
                    origin[other] = value
                value = self.offset + dot(self.normal, cast(Vector3, tuple(origin)))
                slope = self.normal[axis]
                if slope == 0.0:
                    if value == 0.0:
                        for endpoint in (0.0, 1.0):
                            origin[axis] = endpoint
                            points.append(cast(Vector3, tuple(origin)))
                    continue
                coordinate = -value / slope
                if 0.0 <= coordinate <= 1.0:
                    origin[axis] = coordinate
                    points.append(cast(Vector3, tuple(origin)))
        points = list(dict.fromkeys(points))
        if len(points) < 3:
            return ()
        nonzero = tuple(i for i, value in enumerate(self.normal) if value != 0.0)
        if len(nonzero) == 1:
            axis = nonzero[0]
            coordinate = -self.offset / self.normal[axis]
            if coordinate in (0.0, 1.0):
                if self.normal_dot == (0.0, 0.0, 0.0) and self.offset_dot == 0.0:
                    return ()
                raise TransportError("moving source boundary coincides with object support")
        centre = cast(
            Vector3, tuple(math.fsum(point[i] for point in points) / len(points) for i in range(3))
        )
        unit = _scale(self.normal, 1.0 / math.hypot(*self.normal))
        radius = _sub(points[0], centre)
        first = _scale(radius, 1.0 / math.hypot(*radius))
        second = cross(unit, first)
        points.sort(
            key=lambda point: math.atan2(
                dot(_sub(point, centre), second), dot(_sub(point, centre), first)
            )
        )
        return tuple(points)

    def sample(self, coordinate: tuple[float, float]) -> tuple[Vector3, float] | None:
        """Return a uniform polygon point and its signed coarea/Jacobian weight.

        Triangle selection and radial sampling share the first uniform via its
        exact conditional remainder; the second uniform supplies the other
        barycentric coordinate. The weight is ``polygon_area*g_dot/|normal|``.
        """
        if not all(math.isfinite(value) and 0.0 < value < 1.0 for value in coordinate):
            raise TransportError("volume boundary coordinates must lie inside the unit square")
        polygon = self.polygon()
        if not polygon:
            return None
        first = polygon[0]
        triangles = tuple((polygon[i], polygon[i + 1]) for i in range(1, len(polygon) - 1))
        areas = tuple(
            0.5 * math.hypot(*cross(_sub(left, first), _sub(right, first)))
            for left, right in triangles
        )
        total = math.fsum(areas)
        target, cumulative = coordinate[0] * total, 0.0
        for (left, right), area in zip(triangles, areas, strict=True):
            if area > 0.0 and target < cumulative + area:
                radial = math.sqrt((target - cumulative) / area)
                point = _add(
                    _scale(first, 1.0 - radial),
                    _add(
                        _scale(left, radial * (1.0 - coordinate[1])),
                        _scale(right, radial * coordinate[1]),
                    ),
                )
                g_dot = dot(self.normal_dot, point) + self.offset_dot
                return point, total * g_dot / math.hypot(*self.normal)
            cumulative += area
        raise TransportError("boundary triangle draw was lost to floating-point rounding")


def source_volume_boundaries(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    object_tangent: GridGeometryTangent | None = None,
    source_tangent: SourceGeometryTangent | None = None,
) -> tuple[VolumeBoundary, ...]:
    """Build the four source-footprint support planes in collision coordinates.

    With a cone, multiplying each rational footprint coordinate by its positive
    ray-plane denominator gives an affine constraint with the same support and
    coarea measure. The attached denominator derivative is included. This chart
    requires the material box downstream of a transverse parallel source plane
    or wholly in the cone launch plane's forward hemisphere.
    """
    object_tangent = GridGeometryTangent() if object_tangent is None else object_tangent
    source_tangent = SourceGeometryTangent() if source_tangent is None else source_tangent

    def constraints(coordinate: Vector3) -> tuple[tuple[float, ...], tuple[float, ...]]:
        vertex = collision_vertex_tangent(coordinate, spec, object_tangent)
        flux = source_flux_tangent(source, vertex.point_mm, vertex.point_dot, source_tangent)
        if isinstance(source, ParallelBeam):
            return flux.support, flux.support_dot
        detector = OrientedDetector(source.geometry)
        normal = detector.normal
        normal_dot = source_tangent.target_motion.direction_velocity(normal)
        # d*r is the unnormalised source-to-vertex displacement. The denominator
        # has units mm, but its units cancel between g_dot and |grad g|.
        denominator = dot(flux.incoming.direction, normal) * flux.distance_mm
        denominator_dot = math.fsum(
            (
                dot(flux.incoming.direction_dot, normal) * flux.distance_mm,
                dot(flux.incoming.direction, normal_dot) * flux.distance_mm,
                dot(flux.incoming.direction, normal) * flux.distance_dot,
            )
        )
        return tuple(value * denominator for value in flux.support), tuple(
            derivative * denominator + value * denominator_dot
            for value, derivative in zip(flux.support, flux.support_dot, strict=True)
        )

    # Check the physical chart on every box corner before boundary preparation.
    for corner in product((0.0, 1.0), repeat=3):
        constraints(cast(Vector3, corner))
    origin, origin_dot = constraints((0.0, 0.0, 0.0))
    basis = tuple(
        constraints(cast(Vector3, tuple(float(i == axis) for i in range(3)))) for axis in range(3)
    )
    boundaries: list[VolumeBoundary] = []
    for edge in range(4):
        normal = cast(Vector3, tuple(basis[axis][0][edge] - origin[edge] for axis in range(3)))
        normal_dot = cast(
            Vector3, tuple(basis[axis][1][edge] - origin_dot[edge] for axis in range(3))
        )
        if math.hypot(*normal) == 0.0:
            if origin[edge] == 0.0:
                raise TransportError("the entire collision volume lies on a source boundary")
            continue
        boundary = VolumeBoundary(
            f"source:{edge}", normal, origin[edge], normal_dot, origin_dot[edge]
        )
        if boundary.polygon():
            boundaries.append(boundary)
    return tuple(boundaries)


__all__ = [
    "DetectorVertexTangent",
    "SourceFluxTangent",
    "VertexTangent",
    "VolumeBoundary",
    "collision_vertex_tangent",
    "detector_vertex_tangent",
    "source_flux_tangent",
    "source_volume_boundaries",
]
