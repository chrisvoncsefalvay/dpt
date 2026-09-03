"""Typed geometry arguments for native free-Compton terminal boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dpt.geometry import Vector3

from .compton_geometry_boundary import FreeComptonEnergyBoundary
from .geometry_tangents import RigidMotion, detector_corner_velocity
from .model import OrientedDetector, PlanarDetector, TransportError


@dataclass(frozen=True, slots=True)
class BoundaryRayArguments:
    origin_mm: Vector3
    direction: Vector3
    origin_dot: Vector3
    direction_dot: Vector3


@dataclass(frozen=True, slots=True)
class FreeComptonBoundaryArguments:
    incoming: BoundaryRayArguments
    incident_energy: float
    threshold_energy: float
    incident_energy_dot: float
    threshold_energy_dot: float


@dataclass(frozen=True, slots=True)
class BoundaryDetectorArguments:
    corner: Vector3
    u: Vector3
    v: Vector3
    normal: Vector3
    shape: tuple[int, int]
    spacing: tuple[float, float]
    motion_translation: Vector3
    motion_angular: Vector3
    motion_pivot: Vector3
    corner_correction: Vector3
    spacing_dot: tuple[float, float]


def free_compton_boundary_arguments(
    boundary: FreeComptonEnergyBoundary,
    detector: PlanarDetector | OrientedDetector,
    motion: RigidMotion | None = None,
    *,
    spacing_dot: tuple[float, float] = (0.0, 0.0),
) -> tuple[FreeComptonBoundaryArguments, BoundaryDetectorArguments]:
    """Copy validated rigid geometry and preserve the detector spacing anchor.

    Planar detector spacing expands about its lower corner, while an oriented
    detector uses its first pixel centre. The resulting correction preserves
    those existing physical conventions in the attached unit-square chart.
    Active endpoint and coincident-edge chart admission remains a native result
    status, so a driver can distinguish null draws from failed histories.
    """
    if len(spacing_dot) != 2 or any(not math.isfinite(value) for value in spacing_dot):
        raise TransportError("detector spacing tangent must have two finite components")
    motion = RigidMotion() if motion is None else motion
    corner_velocity = detector_corner_velocity(detector, motion, spacing_dot)
    rigid_velocity = motion.point_velocity(detector.lower_corner_mm)
    correction: Vector3 = (
        corner_velocity[0] - rigid_velocity[0],
        corner_velocity[1] - rigid_velocity[1],
        corner_velocity[2] - rigid_velocity[2],
    )
    incoming = boundary.incoming
    return (
        FreeComptonBoundaryArguments(
            BoundaryRayArguments(
                incoming.origin_mm, incoming.direction, incoming.origin_dot, incoming.direction_dot
            ),
            boundary.incident_energy_kev,
            boundary.threshold_energy_kev,
            boundary.incident_energy_dot,
            boundary.threshold_energy_dot,
        ),
        BoundaryDetectorArguments(
            detector.lower_corner_mm,
            detector.u,
            detector.v,
            detector.normal,
            detector.shape,
            detector.spacing_xy_mm,
            motion.translation,
            motion.angular,
            motion.pivot_mm,
            correction,
            spacing_dot,
        ),
    )


__all__ = [
    "BoundaryDetectorArguments",
    "BoundaryRayArguments",
    "FreeComptonBoundaryArguments",
    "free_compton_boundary_arguments",
]
