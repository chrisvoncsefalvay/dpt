"""Signed collision-generator insertions on moving material interfaces.

The physical continuation law is evaluated at the numerical base geometry.
Neither outgoing angles nor free/bound Compton energies are differentiated in
these surface terms. Direct grazing collimated illumination needs its separate
whole-source-ray jump chart; its angular delta is not a smooth volume radiance.
"""

from __future__ import annotations

# Geometry modules share their checked vector arithmetic internally.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass
from itertools import product
from typing import cast

from dpt.geometry import Vector3, cross, dot

from .geometry_path_measure import SourceFluxTangent, source_flux_tangent
from .geometry_tangents import GridGeometryTangent, RigidMotion, _add, _sub
from .model import TransportError, TransportSpec
from .source import ParallelBeam, PointConeSource
from .uncollided_geometry import SourceGeometryTangent, _source_ray


@dataclass(frozen=True, slots=True)
class MaterialInterface:
    """One grid-face rectangle, normal from negative to positive axis side."""

    axis: int
    face: int
    tangential_cells: tuple[int, int]
    negative_cell: int | None
    positive_cell: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tangential_cells", tuple(self.tangential_cells))
        if (
            type(self.axis) is not int
            or not 0 <= self.axis < 3
            or type(self.face) is not int
            or self.face < 0
        ):
            raise TransportError("interface axis and face must be nonnegative valid integers")
        if len(self.tangential_cells) != 2 or any(
            type(value) is not int or value < 0 for value in self.tangential_cells
        ):
            raise TransportError("interface tangential cells must be two nonnegative integers")
        if any(
            value is not None and (type(value) is not int or value < 0)
            for value in (self.negative_cell, self.positive_cell)
        ):
            raise TransportError("interface cell sides must be valid indices or exterior vacuum")

    @property
    def key(self) -> str:
        first, second = self.tangential_cells
        return f"interface:{self.axis}:{self.face}:{first}:{second}"


def material_interfaces(spec: TransportSpec) -> tuple[MaterialInterface, ...]:
    """Enumerate every internal and exterior face patch with explicit cell sides."""
    nx, ny, _ = spec.grid.shape_xyz
    result: list[MaterialInterface] = []
    for axis, count in enumerate(spec.grid.shape_xyz):
        tangential = tuple(i for i in range(3) if i != axis)
        for face, first, second in product(
            range(count + 1),
            range(spec.grid.shape_xyz[tangential[0]]),
            range(spec.grid.shape_xyz[tangential[1]]),
        ):
            indices = [0, 0, 0]
            indices[tangential[0]], indices[tangential[1]] = first, second
            indices[axis] = face - 1
            negative = ((indices[2] * ny + indices[1]) * nx + indices[0]) if face > 0 else None
            indices[axis] = face
            positive = ((indices[2] * ny + indices[1]) * nx + indices[0]) if face < count else None
            result.append(MaterialInterface(axis, face, (first, second), negative, positive))
    return tuple(result)


def _check_interface(interface: MaterialInterface, spec: TransportSpec) -> None:
    tangential = tuple(i for i in range(3) if i != interface.axis)
    count = spec.grid.shape_xyz[interface.axis]
    if interface.face > count or any(
        cell >= spec.grid.shape_xyz[axis]
        for cell, axis in zip(interface.tangential_cells, tangential, strict=True)
    ):
        raise TransportError("material interface does not belong to the current grid")
    nx, ny, _ = spec.grid.shape_xyz
    indices = [0, 0, 0]
    for axis, cell in zip(tangential, interface.tangential_cells, strict=True):
        indices[axis] = cell
    indices[interface.axis] = interface.face - 1
    negative = ((indices[2] * ny + indices[1]) * nx + indices[0]) if interface.face > 0 else None
    indices[interface.axis] = interface.face
    positive = (
        ((indices[2] * ny + indices[1]) * nx + indices[0]) if interface.face < count else None
    )
    if interface.negative_cell != negative or interface.positive_cell != positive:
        raise TransportError("material interface has incorrect explicit cell sides")


@dataclass(frozen=True, slots=True)
class InterfacePoint:
    interface: MaterialInterface
    coordinate: tuple[float, float]
    point_mm: Vector3
    point_dot: Vector3
    normal: Vector3
    normal_dot: Vector3
    area_mm2: float

    def relative_velocity(self, source_gauge: RigidMotion | None = None) -> float:
        """Normal object-interface velocity relative to this source path's gauge."""
        gauge = RigidMotion() if source_gauge is None else source_gauge
        return dot(_sub(self.point_dot, gauge.point_velocity(self.point_mm)), self.normal)


def interface_point_tangent(
    interface: MaterialInterface,
    coordinate: tuple[float, float],
    spec: TransportSpec,
    tangent: GridGeometryTangent | None = None,
) -> InterfacePoint:
    """Map a fixed face rectangle coordinate to an attached world surface point."""
    tangent = GridGeometryTangent() if tangent is None else tangent
    _check_interface(interface, spec)
    if len(coordinate) != 2 or any(
        not math.isfinite(value) or not 0.0 < value < 1.0 for value in coordinate
    ):
        raise TransportError("interface coordinates must lie inside the unit square")
    tangential = tuple(i for i in range(3) if i != interface.axis)
    indices = [0.0, 0.0, 0.0]
    indices[interface.axis] = float(interface.face)
    for axis, cell, amount in zip(tangential, interface.tangential_cells, coordinate, strict=True):
        indices[axis] = cell + amount
    local = cast(
        Vector3,
        tuple(spec.grid.origin_mm[i] + indices[i] * spec.grid.spacing_mm[i] for i in range(3)),
    )
    local_dot = cast(
        Vector3,
        tuple(tangent.origin_dot[i] + indices[i] * tangent.spacing_dot[i] for i in range(3)),
    )
    point = spec.object_pose.point(local)
    point_dot = _add(tangent.motion.point_velocity(point), spec.object_pose.direction(local_dot))
    axis_vector = cast(Vector3, tuple(float(i == interface.axis) for i in range(3)))
    normal = spec.object_pose.direction(axis_vector)
    normal_dot = tangent.motion.direction_velocity(normal)
    area = math.prod(spec.grid.spacing_mm[i] for i in tangential)
    if (
        not all(math.isfinite(value) for value in (*point, *point_dot, *normal_dot, area))
        or area <= 0
    ):
        raise TransportError("material interface measure is not representable in binary64")
    return InterfacePoint(interface, coordinate, point, point_dot, normal, normal_dot, area)


def source_path_gauge(
    source: ParallelBeam | PointConeSource,
    coordinate: tuple[float, float],
    tangent: SourceGeometryTangent | None = None,
) -> RigidMotion:
    """Rigidly transport a whole angular-addressed path under source geometry.

    Each original source coordinate has its own smooth rigid gauge. Translation
    equals the launch-point jet; angular velocity is ``d cross d_dot``. It maps
    the original direction to its jet and preserves every scattering cosine,
    shell/momentum mark and photon energy along that conditional path. Source
    extents/target spacing enter through their actual launch geometry, with the
    original source weight and unit-square denominator unchanged.
    """
    tangent = SourceGeometryTangent() if tangent is None else tangent
    ray, _, _ = _source_ray(source, coordinate, tangent)
    return RigidMotion(
        translation=ray.origin_dot,
        angular=cross(ray.direction, ray.direction_dot),
        pivot_mm=ray.origin_mm,
    )


@dataclass(frozen=True, slots=True)
class DirectInterfaceIncident:
    flux: SourceFluxTangent
    source_coordinate: tuple[float, float]
    source_gauge: RigidMotion


def direct_interface_incident(
    point: InterfacePoint,
    source: ParallelBeam | PointConeSource,
    source_tangent: SourceGeometryTangent | None = None,
) -> DirectInterfaceIncident | None:
    """Resolve smooth ballistic incident fluence and its original source address.

    A grazing collimated angular delta has a whole-ray transmission/interaction
    jump. The regular collision-generator product is invalid for that direct
    component and is rejected here. Source-ray silhouette continuations must
    supply it separately; collided angular radiance remains a surface term.
    """
    flux = source_flux_tangent(source, point.point_mm, (0.0, 0.0, 0.0))
    if not flux.visible:
        return None
    if isinstance(source, ParallelBeam) and dot(flux.incoming.direction, point.normal) == 0.0:
        raise TransportError("grazing ballistic interface requires its whole-source-ray jump chart")
    coordinate = flux.support[0], flux.support[2]
    return DirectInterfaceIncident(
        flux, coordinate, source_path_gauge(source, coordinate, source_tangent)
    )


def _asinh_ratio(value: float, height: float) -> float:
    if abs(value) <= height:
        return math.asinh(value / height)
    ratio = height / abs(value)
    return math.copysign(
        math.log(abs(value)) - math.log(height) + math.log1p(math.sqrt(1.0 + ratio * ratio)), value
    )


def _scaled_sinh(height: float, value: float) -> float:
    if value == 0.0:
        return 0.0
    logarithm = math.log(height) + abs(value) - math.log(2.0)
    return math.copysign(math.exp(logarithm) * -math.expm1(-2.0 * abs(value)), value)


@dataclass(frozen=True, slots=True)
class InterfaceBridgeSample:
    """A detached surface proposal conditioned on the previous collision point."""

    point: InterfacePoint
    log_area_density: float
    log_geometric_importance: float
    axis_normalisers: tuple[float, float]


def sample_interface_bridge(
    interface: MaterialInterface,
    previous_point_mm: Vector3,
    coordinate: tuple[float, float],
    spec: TransportSpec,
    tangent: GridGeometryTangent | None = None,
) -> InterfaceBridgeSample:
    """Cancel the near-interface inverse-square bridge singularity analytically.

    With normal separation ``h>0`` and tangential offsets ``x,y``, the proposal
    is the product of normalised ``1/sqrt(h*h+t*t)`` densities. Both one-variable
    CDFs have exact asinh primitives. Its importance factor for the physical
    inverse-square bridge is bounded by the product of its normalisers, which
    grows at most as ``log(1/h)**2`` near the plane. No accepted-arc conditioning,
    epsilon offset, arbitrary profile tail or unnormalised density is used.
    """
    if len(coordinate) != 2 or any(
        not math.isfinite(value) or not 0.0 < value < 1.0 for value in coordinate
    ):
        raise TransportError("interface bridge draws must lie inside the unit square")
    _check_interface(interface, spec)
    previous = spec.object_pose.inverse().point(previous_point_mm)
    face = (
        spec.grid.origin_mm[interface.axis] + interface.face * spec.grid.spacing_mm[interface.axis]
    )
    height = abs(previous[interface.axis] - face)
    if not math.isfinite(height) or height == 0.0:
        raise TransportError("surface bridge requires a previous vertex off its exact plane")
    tangential = tuple(i for i in range(3) if i != interface.axis)
    result: list[float] = []
    offsets: list[float] = []
    normalisers: list[float] = []
    for axis, cell, draw in zip(tangential, interface.tangential_cells, coordinate, strict=True):
        lower = spec.grid.origin_mm[axis] + cell * spec.grid.spacing_mm[axis] - previous[axis]
        upper = lower + spec.grid.spacing_mm[axis]
        first, last = _asinh_ratio(lower, height), _asinh_ratio(upper, height)
        normal = last - first
        if not math.isfinite(normal) or normal <= 0.0:
            raise TransportError("interface proposal normaliser is not representable")
        offset = _scaled_sinh(height, (1.0 - draw) * first + draw * last)
        amount = (offset - lower) / spec.grid.spacing_mm[axis]
        if not 0.0 < amount < 1.0:
            raise TransportError("surface proposal point is not distinguishable in binary64")
        result.append(amount)
        offsets.append(offset)
        normalisers.append(normal)
    mapped = interface_point_tangent(
        interface, cast(tuple[float, float], tuple(result)), spec, tangent
    )
    represented = spec.object_pose.inverse().point(mapped.point_mm)
    offsets = []
    for axis, cell, draw, normal in zip(
        tangential, interface.tangential_cells, coordinate, normalisers, strict=True
    ):
        lower = spec.grid.origin_mm[axis] + cell * spec.grid.spacing_mm[axis] - previous[axis]
        first = _asinh_ratio(lower, height)
        offset = represented[axis] - previous[axis]
        reconstructed = (_asinh_ratio(offset, height) - first) / normal
        uncertainty = (
            8.0
            * (math.ulp(represented[axis]) + math.ulp(previous[axis]))
            / (math.hypot(height, offset) * normal)
        )
        if not math.isfinite(uncertainty) or uncertainty > math.sqrt(math.ulp(1.0)):
            raise TransportError("surface proposal coordinate is not distinguishable in binary64")
        tolerance = uncertainty + 512.0 * math.ulp(1.0) * (1.0 + abs(first) / normal)
        if abs(reconstructed - draw) > tolerance:
            raise TransportError("surface proposal coordinate is not distinguishable in binary64")
        offsets.append(offset)
    distance = math.hypot(height, *offsets)
    log_density = -math.fsum(
        math.log(math.hypot(height, offset)) + math.log(normal)
        for offset, normal in zip(offsets, normalisers, strict=True)
    )
    log_importance = -2.0 * math.log(distance) - log_density
    return InterfaceBridgeSample(
        mapped, log_density, log_importance, cast(tuple[float, float], tuple(normalisers))
    )


def interface_generator_observation(
    point: InterfacePoint,
    *,
    negative_extinction_mm_inverse: float,
    positive_extinction_mm_inverse: float,
    negative_collision_response: float | None,
    positive_collision_response: float | None,
    unchanged_direction_response: float,
    incident_importance: float,
    source_gauge: RigidMotion | None = None,
) -> float:
    """Contract the two physical collision generators with incident radiance.

    Each response is a complete independently continued detector measurement at
    the base geometry. Collision responses include actual conditional channel
    selection and any resulting fluorescence/Compton energy law. ``None`` is
    accepted solely on a zero-rate vacuum side. The unchanged-direction response
    is the physical continuation without inserting a collision, with unchanged
    incident energy. Missing non-vacuum continuations never become zero.
    """
    rates = (negative_extinction_mm_inverse, positive_extinction_mm_inverse)
    if any(not math.isfinite(rate) or rate < 0.0 for rate in rates):
        raise TransportError("interface extinction rates must be finite and nonnegative")
    responses = (negative_collision_response, positive_collision_response)
    for rate, response in zip(rates, responses, strict=True):
        if response is None and rate != 0.0:
            raise TransportError(
                "a non-vacuum interface requires its actual collision continuation"
            )
        if response is not None and not math.isfinite(response):
            raise TransportError("collision continuation must be finite and complete")
    if not all(
        math.isfinite(value) for value in (unchanged_direction_response, incident_importance)
    ):
        raise TransportError("incident and unchanged-direction continuations must be finite")
    negative, positive = responses
    jump = math.fsum(
        (
            rates[0] * (0.0 if negative is None else negative),
            -rates[1] * (0.0 if positive is None else positive),
            -(rates[0] - rates[1]) * unchanged_direction_response,
        )
    )
    value = point.relative_velocity(source_gauge) * incident_importance * jump
    if not math.isfinite(value):
        raise TransportError("interface generator contribution is not representable in binary64")
    return value


__all__ = [
    "DirectInterfaceIncident",
    "InterfaceBridgeSample",
    "InterfacePoint",
    "MaterialInterface",
    "direct_interface_incident",
    "interface_generator_observation",
    "interface_point_tangent",
    "material_interfaces",
    "sample_interface_bridge",
    "source_path_gauge",
]
