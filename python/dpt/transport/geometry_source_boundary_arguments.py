"""Typed launch jets and prevalidated affine source-boundary arguments."""

# The adapter shares the frozen, validated source chart with the host driver.
# pyright: reportPrivateUsage=false
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dpt.geometry import Vector3

from .geometry_tangents import AffineBoundary
from .model import TransportError
from .source import ParallelBeam, PointConeSource
from .uncollided_geometry import SourceGeometryTangent, _source_ray

ZERO: Vector3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class SourceGeometryArguments:
    kind: int
    origin_mm: Vector3
    direction: Vector3
    extent: tuple[float, float]
    motion_translation: Vector3
    motion_angular: Vector3
    motion_pivot: Vector3
    extent_dot: tuple[float, float]
    direction_dot: Vector3
    target_origin: Vector3
    target_u: Vector3
    target_v: Vector3
    target_shape: tuple[int, int]
    target_spacing: tuple[float, float]
    target_motion_translation: Vector3
    target_motion_angular: Vector3
    target_motion_pivot: Vector3
    target_spacing_dot: tuple[float, float]
    weight: float


@dataclass(frozen=True, slots=True)
class AffineSourceBoundaryArguments:
    a: float
    b: float
    c: float
    a_dot: float
    b_dot: float
    c_dot: float
    axis: int
    face: int
    family: int
    present: int


def source_geometry_arguments(
    source: ParallelBeam | PointConeSource,
    tangent: SourceGeometryTangent | None = None,
) -> SourceGeometryArguments:
    """Preserve the existing source law, its fixed unit square and its weight.

    Parallel extents expand about the lower corner. A point cone retains its
    DetectorGeometry launch target and its first-pixel-centre spacing anchor.
    The host's source-ray charts validate all four corners before packing.
    No source area, fluence or accepted-ray normalisation is introduced.
    """
    tangent = SourceGeometryTangent() if tangent is None else tangent
    for coordinate in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        _source_ray(source, coordinate, tangent)
    if isinstance(source, ParallelBeam):
        return SourceGeometryArguments(
            0,
            source.lower_mm,
            source.direction,
            source.extent_xy_mm,
            tangent.motion.translation,
            tangent.motion.angular,
            tangent.motion.pivot_mm,
            tangent.extent_dot,
            tangent.direction_dot,
            ZERO,
            ZERO,
            ZERO,
            (0, 0),
            (0.0, 0.0),
            ZERO,
            ZERO,
            ZERO,
            (0.0, 0.0),
            source.weight,
        )
    geometry = source.geometry
    if max(geometry.shape) > 2**31 - 1:
        raise TransportError("native launch-target shape exceeds signed 32-bit indexing")
    return SourceGeometryArguments(
        1,
        geometry.source_mm,
        ZERO,
        (0.0, 0.0),
        tangent.motion.translation,
        tangent.motion.angular,
        tangent.motion.pivot_mm,
        (0.0, 0.0),
        ZERO,
        geometry.origin_mm,
        geometry.u,
        geometry.v,
        (geometry.shape[1], geometry.shape[0]),
        geometry.spacing_mm,
        tangent.target_motion.translation,
        tangent.target_motion.angular,
        tangent.target_motion.pivot_mm,
        tangent.target_spacing_dot,
        source.weight,
    )


def affine_source_boundary_arguments(
    boundary: AffineBoundary | None,
    *,
    axis: int,
    face: int,
    family: Literal["material", "detector"],
) -> AffineSourceBoundaryArguments:
    """Pack an owner's prepared chart and retain its side-jump identity.

    The preparation owner constructs material and detector coefficients with
    the validated acquisition charts. This adapter checks the clipped line;
    unsupported active coincidence raises before a native launch. An absent
    chart is retained explicitly as a null proposal, with no accepted redraw.
    """
    if family not in ("material", "detector"):
        raise TransportError("source boundary family must be material or detector")
    maximum_axis = 2 if family == "material" else 1
    if type(axis) is not int or not 0 <= axis <= maximum_axis:
        raise TransportError("source boundary axis is invalid for its family")
    if type(face) is not int or not 0 <= face <= 2**31 - 1:
        raise TransportError("source boundary face exceeds signed 32-bit indexing")
    present = boundary is not None and boundary.endpoints() is not None
    coefficients = (0.0, 0.0, 0.0) if boundary is None else boundary.coefficients
    derivatives = (0.0, 0.0, 0.0) if boundary is None else boundary.coefficients_dot
    return AffineSourceBoundaryArguments(
        *coefficients, *derivatives, axis, face, int(family == "detector"), int(present)
    )


__all__ = [
    "AffineSourceBoundaryArguments",
    "SourceGeometryArguments",
    "affine_source_boundary_arguments",
    "source_geometry_arguments",
]
