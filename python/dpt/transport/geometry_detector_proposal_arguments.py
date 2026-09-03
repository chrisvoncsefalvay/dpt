"""Immutable host plans for detached native detector proposals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import cast

from dpt.geometry import Matrix3, Vector3

from .geometry_detector_proposals import DetectorAreaProposal, validate_redundant_planes
from .model import TransportError, TransportSpec


@dataclass(frozen=True, slots=True)
class DetectorProposalPlaneArguments:
    coefficients: Vector3
    normal_separation_mm: float
    axis: int
    position_mm: float


@dataclass(frozen=True, slots=True)
class DetectorProposalGeometryArguments:
    inverse_rotation: Matrix3
    inverse_translation: Vector3
    detector_lower: Vector3
    detector_u: Vector3
    detector_v: Vector3
    detector_shape: tuple[int, int]
    detector_spacing: tuple[float, float]


def detector_proposal_arguments(
    proposal: DetectorAreaProposal,
) -> tuple[DetectorProposalPlaneArguments, ...]:
    """Copy one vertex's checked proposal into native plane fields.

    The returned density is per unit square. Its identity remains with the
    history owner, who retains the detached proposal when replaying a score.
    """
    if not isinstance(cast(object, proposal), DetectorAreaProposal):
        raise TransportError("detector proposal must be a DetectorAreaProposal")
    records: list[DetectorProposalPlaneArguments] = []
    for plane in proposal.planes:
        if (
            plane.axis not in (0, 1, 2)
            or plane.face < 0
            or any(not math.isfinite(value) for value in plane.coefficients)
            or max(abs(plane.coefficients[0]), abs(plane.coefficients[1])) == 0.0
            or not math.isfinite(plane.normal_separation_mm)
            or plane.normal_separation_mm <= 0.0
        ):
            raise TransportError(
                "detector proposal planes require finite coefficients and positive separation"
            )
        records.append(
            DetectorProposalPlaneArguments(
                plane.coefficients, plane.normal_separation_mm, plane.axis, 0.0
            )
        )
    return tuple(records)


def detector_proposal_plan_arguments(
    spec: TransportSpec,
    *,
    redundant_planes: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[tuple[DetectorProposalPlaneArguments, ...], DetectorProposalGeometryArguments]:
    """Prepare grid-plane coefficients once for native conditional-vertex sampling.

    The native function derives each separation from its current physical
    vertex. This plan owns no device buffers and imports no numerical runtime.
    """
    if not isinstance(cast(object, spec), TransportSpec):
        raise TransportError("detector proposal plan requires a TransportSpec")
    redundant_planes = validate_redundant_planes(redundant_planes, spec)
    for corner in product((0.0, 1.0), repeat=3):
        local = cast(
            Vector3,
            tuple(
                spec.grid.origin_mm[axis]
                + corner[axis] * (spec.grid.upper_mm[axis] - spec.grid.origin_mm[axis])
                for axis in range(3)
            ),
        )
        if spec.detector.source_distance(spec.object_pose.point(local)) >= 0.0:
            raise TransportError(
                "geometry detector requires strict separation from the material box"
            )
    inverse = spec.object_pose.inverse()
    lower = inverse.point(spec.detector.lower_corner_mm)
    u, v = inverse.direction(spec.detector.u), inverse.direction(spec.detector.v)
    width = spec.detector.shape[0] * spec.detector.spacing_xy_mm[0]
    height = spec.detector.shape[1] * spec.detector.spacing_xy_mm[1]
    records: list[DetectorProposalPlaneArguments] = []
    for axis, count in enumerate(spec.grid.shape_xyz):
        a, b = width * u[axis], height * v[axis]
        if a == 0.0 and b == 0.0:
            continue
        for face in range(count + 1):
            if (axis, face) in redundant_planes:
                continue
            position = spec.grid.origin_mm[axis] + face * spec.grid.spacing_mm[axis]
            coefficients = (a, b, lower[axis] - position)
            if any(not math.isfinite(value) for value in (*coefficients, position)):
                raise TransportError(
                    "detector proposal coefficients are not representable in binary64"
                )
            records.append(DetectorProposalPlaneArguments(coefficients, 0.0, axis, position))
    return (
        tuple(records),
        DetectorProposalGeometryArguments(
            inverse.rotation,
            inverse.translation_mm,
            spec.detector.lower_corner_mm,
            spec.detector.u,
            spec.detector.v,
            spec.detector.shape,
            spec.detector.spacing_xy_mm,
        ),
    )


__all__ = [
    "DetectorProposalGeometryArguments",
    "DetectorProposalPlaneArguments",
    "detector_proposal_arguments",
    "detector_proposal_plan_arguments",
]
