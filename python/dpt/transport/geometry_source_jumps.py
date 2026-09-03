"""Complete physical source-ray side jumps for collimated material silhouettes."""

from __future__ import annotations

# Geometry components share their checked source/chart primitives internally.
# pyright: reportPrivateUsage=false
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, cast

from dpt.geometry import Vector3

from .geometry_tangents import AffineBoundary, RayTangent
from .model import TransportError, TransportSpec
from .source import ParallelBeam, PointConeSource
from .uncollided_geometry import (
    AcquisitionGeometryTangent,
    SourceGeometryTangent,
    _material_boundaries,
    _source_ray,
)


@dataclass(frozen=True, slots=True)
class SourceRayMaterialBoundary:
    boundary: AffineBoundary
    axis: int
    face: int


@dataclass(frozen=True, slots=True)
class SourceRayJumpSample:
    """An exact source address and its whole physical two-sided continuations.

    Both continuations start with the same emitted ray/energy. The ``side``
    override holds its initial straight flight on the declared material side
    without an epsilon displacement. After the first physical collision, the
    override is cleared and every supported interaction/future is continued.
    A transmitted-only response would omit the scattered part of this jump.
    """

    key: str
    source_coordinate: tuple[float, float]
    ray: RayTangent
    coarea_weight: float
    source_weight: float
    negative_side: tuple[int, int, Literal[-1, 1]]
    positive_side: tuple[int, int, Literal[-1, 1]]


def admit_joint_source_boundary_chart(
    boundaries: Sequence[AffineBoundary],
    *,
    family: Literal["source", "material/detector", "opaque/detector"] = "source",
) -> None:
    """Reject active coincident lines in the complete represented source chart.

    Rational coefficient ratios preserve the exact represented affine line;
    normalising each line by a rounded Euclidean norm can merge distinct lines
    or fail to recognise proportional ones. Stationary duplicate rows have no
    support derivative and retain their existing null contribution.
    """
    lines: dict[tuple[Fraction, Fraction, Fraction], AffineBoundary] = {}
    for boundary in boundaries:
        a, b, c = (Fraction(value) for value in boundary.coefficients)
        scale = a if a else b
        if not scale:
            raise TransportError("a prepared boundary line requires a nonzero affine gradient")
        key = a / scale, b / scale, c / scale
        previous = lines.get(key)
        if previous is not None and (
            previous.coefficients_dot != (0.0, 0.0, 0.0)
            or boundary.coefficients_dot != (0.0, 0.0, 0.0)
        ):
            raise TransportError(f"coincident active {family} boundaries need a joint chart")
        lines[key] = boundary


def material_source_ray_boundaries(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    tangent: AcquisitionGeometryTangent,
    *,
    equivalent_whole_faces: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[SourceRayMaterialBoundary, ...]:
    """Prepare all required source-area material jumps at the numerical base."""
    return tuple(
        SourceRayMaterialBoundary(boundary, axis, face)
        for boundary, axis, face in _material_boundaries(
            source, spec, tangent, equivalent_whole_faces=equivalent_whole_faces
        )
    )


def sample_material_source_ray_jump(
    source: ParallelBeam | PointConeSource,
    record: SourceRayMaterialBoundary,
    coordinate: float,
    *,
    spec: TransportSpec | None = None,
) -> SourceRayJumpSample:
    sample = record.boundary.sample(coordinate)
    if sample is None:
        raise TransportError(
            "material source-ray boundary is absent from its declared source chart"
        )
    address, weight = sample
    ray = _source_ray(source, address, SourceGeometryTangent())[0]
    if isinstance(source, PointConeSource):
        if spec is None:
            raise TransportError("a cone angular side chart requires its material geometry")
        inverse = spec.object_pose.inverse()
        local = inverse.direction(ray.direction)
        resolution = 512 * math.ulp(max(1.0, *(abs(value) for value in local)))
        if abs(local[record.axis]) > resolution:
            raise TransportError("cone angular sample does not resolve its attached grazing plane")
        attached = tuple(0.0 if axis == record.axis else value for axis, value in enumerate(local))
        length = math.hypot(*attached)
        if length == 0.0 or not math.isfinite(length):
            raise TransportError("cone angular side direction is not representable")
        direction = spec.object_pose.direction(
            cast(Vector3, tuple(value / length for value in attached))
        )
        ray = RayTangent(ray.origin_mm, direction)
    return SourceRayJumpSample(
        record.boundary.key,
        address,
        ray,
        weight,
        source.weight,
        (record.axis, record.face, -1),
        (record.axis, record.face, 1),
    )


def material_source_ray_jump_observation(
    sample: SourceRayJumpSample,
    *,
    negative_complete_response: float,
    positive_complete_response: float,
) -> float:
    """Contract complete unit-photon future responses with the signed coarea.

    Responses include the signed pixel seed and physical photon/energy scoring,
    including all positive-order scattering and fluorescence futures. They
    exclude only the separately multiplied original ``source.weight``. Their
    full physical histories must terminate successfully; guard failures may not
    be supplied as zero or treated as absorption. Conditional sampling of one
    side or rejection of difficult continuations changes this estimator.
    """
    if not all(
        math.isfinite(value) for value in (negative_complete_response, positive_complete_response)
    ):
        raise TransportError("source-ray side responses must be finite complete physical scores")
    value = (
        sample.source_weight
        * sample.coarea_weight
        * (positive_complete_response - negative_complete_response)
    )
    if not math.isfinite(value):
        raise TransportError("source-ray side jump is not representable in binary64")
    return value


__all__ = [
    "SourceRayJumpSample",
    "SourceRayMaterialBoundary",
    "admit_joint_source_boundary_chart",
    "material_source_ray_boundaries",
    "material_source_ray_jump_observation",
    "sample_material_source_ray_jump",
]
