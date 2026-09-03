"""Detached forward-disc importance measure for zero-binding Compton jets.

A fixed outgoing-energy mark produces a joint geometry derivative of order
1/theta**2 near the forward direction of a zero-binding shell. The normalised
rho**(-3/2) detector-area component makes its squared importance integrable.
It is mixed with the complete grid-plane proposal; outside-disc/rectangle
samples remain null observations under the original draw denominator.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from typing import cast

from dpt.geometry import Vector3, dot

from .geometry_detector_proposals import (
    DetectorAreaProposal,
    prepare_detector_area_proposal,
    sample_detector_area_proposal,
)
from .geometry_path_measure import detector_vertex_tangent
from .geometry_tangents import RayTangent
from .model import TransportError, TransportSpec


@dataclass(frozen=True, slots=True)
class ForwardDisc:
    center_mm: Vector3
    center_coordinate: tuple[float, float]
    radius_mm: float
    extent_mm: tuple[float, float]
    area_mm2: float

    def __post_init__(self) -> None:
        if (
            len(self.center_mm) != 3
            or len(self.center_coordinate) != 2
            or len(self.extent_mm) != 2
            or any(
                not math.isfinite(v)
                for v in (
                    *self.center_mm,
                    *self.center_coordinate,
                    *self.extent_mm,
                    self.radius_mm,
                    self.area_mm2,
                )
            )
            or any(v <= 0.0 for v in (*self.extent_mm, self.radius_mm, self.area_mm2))
            or self.area_mm2 != self.extent_mm[0] * self.extent_mm[1]
        ):
            raise TransportError(
                "forward detector disc requires finite positive area/extents/radius"
            )


@dataclass(frozen=True, slots=True)
class ForwardAreaMixture:
    regular: DetectorAreaProposal
    forward: ForwardDisc | None
    identity: str

    def log_density(self, coordinate: tuple[float, float]) -> float:
        regular = self.regular.log_density(coordinate)
        if self.forward is None:
            return regular
        disc = self.forward
        radius = math.hypot(
            *(disc.extent_mm[i] * (coordinate[i] - disc.center_coordinate[i]) for i in range(2))
        )
        if not math.isfinite(radius) or radius == 0.0:
            raise TransportError(
                "forward detector density is not representable at its exact center"
            )
        if radius > disc.radius_mm:
            return math.log(0.5) + regular
        radial = (
            math.log(disc.area_mm2)
            - math.log(4.0 * math.pi)
            - 0.5 * math.log(disc.radius_mm)
            - 1.5 * math.log(radius)
        )
        scale = max(regular, radial)
        value = (
            math.log(0.5) + scale + math.log(math.exp(regular - scale) + math.exp(radial - scale))
        )
        if not math.isfinite(value):
            raise TransportError(
                "forward detector mixture density is not representable in binary64"
            )
        return value


@dataclass(frozen=True, slots=True)
class ForwardAreaSample:
    coordinate: tuple[float, float] | None
    log_reference_density: float | None
    component: str
    reference_identity: str


def prepare_forward_area_mixture(
    incoming: RayTangent,
    spec: TransportSpec,
    *,
    redundant_planes: frozenset[tuple[int, int]] = frozenset(),
) -> ForwardAreaMixture:
    """Freeze the complete proposal at the numerical prefix/detector base."""
    extent = (
        spec.detector.shape[0] * spec.detector.spacing_xy_mm[0],
        spec.detector.shape[1] * spec.detector.spacing_xy_mm[1],
    )
    area = extent[0] * extent[1]
    if not math.isfinite(area) or area <= 0.0:
        raise TransportError("forward detector proposal requires finite positive area")
    regular = prepare_detector_area_proposal(
        incoming.origin_mm, spec, redundant_planes=redundant_planes
    )
    denominator = dot(incoming.direction, spec.detector.normal)
    forward = None
    if denominator > 0.0:
        height = dot(
            cast(
                Vector3,
                tuple(spec.detector.lower_corner_mm[i] - incoming.origin_mm[i] for i in range(3)),
            ),
            spec.detector.normal,
        )
        distance = height / denominator
        center = incoming.point(distance)
        extent = (
            spec.detector.shape[0] * spec.detector.spacing_xy_mm[0],
            spec.detector.shape[1] * spec.detector.spacing_xy_mm[1],
        )
        offset = cast(
            Vector3, tuple(center[i] - spec.detector.lower_corner_mm[i] for i in range(3))
        )
        center_coordinate = (
            dot(offset, spec.detector.u) / extent[0],
            dot(offset, spec.detector.v) / extent[1],
        )
        # A conservative covering disc has a strict margin over all rectangle
        # corners; this is a proposal radius, not a physical cutoff/epsilon.
        radius = 2.0 * max(
            math.hypot(*(extent[i] * (corner[i] - center_coordinate[i]) for i in range(2)))
            for corner in product((0.0, 1.0), repeat=2)
        )
        if (
            distance <= 0.0
            or radius <= 0.0
            or not all(math.isfinite(v) for v in (*center, *center_coordinate, *extent, radius))
        ):
            raise TransportError("forward detector proposal is not representable in binary64")
        forward = ForwardDisc(center, center_coordinate, radius, extent, extent[0] * extent[1])
    payload = (
        regular.identity,
        None
        if forward is None
        else (forward.center_mm, forward.center_coordinate, forward.radius_mm, forward.extent_mm),
    )
    identity = hashlib.sha256(
        json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()
    return ForwardAreaMixture(regular, forward, identity)


def _guard_forward_endpoint(
    mixture: ForwardAreaMixture, coordinate: tuple[float, float], spec: TransportSpec
) -> None:
    disc = mixture.forward
    if disc is None:
        return
    ideal = tuple(disc.extent_mm[i] * (coordinate[i] - disc.center_coordinate[i]) for i in range(2))
    endpoint = detector_vertex_tangent(coordinate, spec.detector).point_mm
    displacement = cast(Vector3, tuple(endpoint[i] - disc.center_mm[i] for i in range(3)))
    actual = (dot(displacement, spec.detector.u), dot(displacement, spec.detector.v))
    radius = math.hypot(*ideal)
    represented_radius = math.hypot(*actual)
    if radius == 0.0 or represented_radius == 0.0 or not math.isfinite(represented_radius):
        raise TransportError("forward detector endpoint is not distinguishable in binary64")
    if math.hypot(*(actual[i] - ideal[i] for i in range(2))) / max(
        radius, represented_radius
    ) > math.sqrt(math.ulp(1.0)):
        raise TransportError("forward detector endpoint is not distinguishable in binary64")


def sample_forward_area_mixture(
    mixture: ForwardAreaMixture,
    spec: TransportSpec,
    *,
    component_draw: float,
    regular_component_draw: float,
    coordinate: tuple[float, float],
) -> ForwardAreaSample:
    """Sample the full disc/regular mixture without accepting or renormalising arcs."""
    if (
        not math.isfinite(component_draw)
        or not 0.0 < component_draw < 1.0
        or any(not math.isfinite(v) or not 0.0 < v < 1.0 for v in coordinate)
    ):
        raise TransportError("forward detector draws must be open-interval uniforms")
    if mixture.forward is None or component_draw < 0.5:
        sampled = sample_detector_area_proposal(
            mixture.regular, component_draw=regular_component_draw, coordinate=coordinate
        )
        mapped = sampled.coordinate
        component = "regular"
    else:
        disc = mixture.forward
        radius = disc.radius_mm * coordinate[0] * coordinate[0]
        if radius == 0.0 or not math.isfinite(radius):
            raise TransportError("forward detector radius is not representable in binary64")
        angle = 2.0 * math.pi * coordinate[1]
        mapped = (
            disc.center_coordinate[0] + radius * math.cos(angle) / disc.extent_mm[0],
            disc.center_coordinate[1] + radius * math.sin(angle) / disc.extent_mm[1],
        )
        if not all(math.isfinite(v) for v in mapped):
            raise TransportError("forward detector coordinate overflows binary64")
        if not all(0.0 < v < 1.0 for v in mapped):
            return ForwardAreaSample(None, None, "forward-null", mixture.identity)
        # Retain the exact draw's radial CDF and azimuth in the actual world
        # chart, just as the regular proposal checks its actual plane levels.
        actual_radius = math.hypot(
            *(disc.extent_mm[i] * (mapped[i] - disc.center_coordinate[i]) for i in range(2))
        )
        if abs(math.sqrt(actual_radius / disc.radius_mm) - coordinate[0]) > 512.0 * math.ulp(1.0):
            raise TransportError(
                "forward detector radial coordinate is not distinguishable in binary64"
            )
        actual_angle = math.atan2(
            disc.extent_mm[1] * (mapped[1] - disc.center_coordinate[1]),
            disc.extent_mm[0] * (mapped[0] - disc.center_coordinate[0]),
        ) / (2.0 * math.pi)
        actual_angle %= 1.0
        angular_error = abs(actual_angle - coordinate[1])
        angular_error = min(angular_error, 1.0 - angular_error)
        if angular_error > math.sqrt(math.ulp(1.0)):
            raise TransportError("forward detector azimuth is not distinguishable in binary64")
        component = "forward"
        # The regular density remains in every mixture sample. Its actual
        # world-plane representation is therefore validated as well.
        sample_detector_area_proposal(mixture.regular, component_draw=0.25, coordinate=mapped)
    _guard_forward_endpoint(mixture, mapped, spec)
    return ForwardAreaSample(mapped, mixture.log_density(mapped), component, mixture.identity)


__all__ = [
    "ForwardAreaMixture",
    "ForwardAreaSample",
    "ForwardDisc",
    "prepare_forward_area_mixture",
    "sample_forward_area_mixture",
]
