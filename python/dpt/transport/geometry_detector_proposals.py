"""Detached detector-area proposals for grazing attenuation-gradient moments.

Uniform detector samples give logarithmically divergent squared face-root jets
when both endpoints approach a discontinuous material plane. A uniform/logarithmic
mixture retains full detector support and samples that local singularity directly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from typing import cast

from dpt.geometry import RigidTransform, Vector3

from .geometry_path_measure import detector_vertex_tangent
from .model import OrientedDetector, PlanarDetector, TransportError, TransportSpec


@dataclass(frozen=True, slots=True)
class DetectorPlaneProposal:
    axis: int
    face: int
    coefficients: tuple[float, float, float]
    normal_separation_mm: float

    @property
    def sampled_axis(self) -> int:
        return int(abs(self.coefficients[1]) > abs(self.coefficients[0]))


def _primitive(value: float, separation: float) -> float:
    if value == 0.0:
        return 0.0
    if abs(value) <= separation:
        return math.copysign(math.log1p(abs(value) / separation), value)
    return math.copysign(
        math.log(abs(value)) - math.log(separation) + math.log1p(separation / abs(value)), value
    )


def _inverse_primitive(value: float, separation: float) -> float:
    if value == 0.0:
        return 0.0
    return math.copysign(
        math.exp(math.log(separation) + abs(value)) * -math.expm1(-abs(value)), value
    )


@dataclass(frozen=True, slots=True)
class DetectorReferenceChart:
    """Actual immutable detector endpoint and inverse-object floating point chart."""

    detector: PlanarDetector | OrientedDetector
    inverse_object_pose: RigidTransform
    grid_origin_mm: Vector3
    grid_spacing_mm: Vector3


@dataclass(frozen=True, slots=True)
class DetectorAreaProposal:
    """Numerical base mixture with a uniform half and one log term per grid plane."""

    planes: tuple[DetectorPlaneProposal, ...]
    identity: str
    chart: DetectorReferenceChart | None = None

    def log_density(self, coordinate: tuple[float, float]) -> float:
        if len(coordinate) != 2 or any(
            not math.isfinite(value) or not 0.0 < value < 1.0 for value in coordinate
        ):
            raise TransportError("detector proposal coordinates must lie inside the unit square")
        if not self.planes:
            return 0.0
        logarithms = [math.log(0.5)]
        mixture = math.log(0.5 / len(self.planes))
        for plane in self.planes:
            a, b, c = plane.coefficients
            axis = plane.sampled_axis
            coefficient = (a, b)[axis]
            other = (b, a)[axis] * coordinate[1 - axis] + c
            lower, upper = sorted((other, other + coefficient))
            width = _primitive(upper, plane.normal_separation_mm) - _primitive(
                lower, plane.normal_separation_mm
            )
            level = coefficient * coordinate[axis] + other
            if not math.isfinite(width) or width <= 0.0:
                raise TransportError("detector-plane proposal normaliser is not representable")
            logarithms.append(
                mixture
                + math.log(abs(coefficient))
                - math.log(width)
                - math.log(plane.normal_separation_mm + abs(level))
            )
        scale = max(logarithms)
        return scale + math.log(math.fsum(math.exp(value - scale) for value in logarithms))


@dataclass(frozen=True, slots=True)
class DetectorAreaProposalSample:
    coordinate: tuple[float, float]
    log_reference_density: float
    component: int
    reference_identity: str


def validate_redundant_planes(
    redundant_planes: frozenset[tuple[int, int]], spec: TransportSpec
) -> frozenset[tuple[int, int]]:
    if not isinstance(cast(object, redundant_planes), frozenset) or any(
        not isinstance(cast(object, plane), tuple)
        or len(plane) != 2
        or type(plane[0]) is not int
        or type(plane[1]) is not int
        or plane[0] not in (0, 1, 2)
        or not 0 <= plane[1] <= spec.grid.shape_xyz[plane[0]]
        for plane in redundant_planes
    ):
        raise TransportError("redundant proposal planes require immutable valid axis/face labels")
    return redundant_planes


def prepare_detector_area_proposal(
    previous_point_mm: Vector3,
    spec: TransportSpec,
    *,
    redundant_planes: frozenset[tuple[int, int]] = frozenset(),
) -> DetectorAreaProposal:
    """Prepare fixed grid-plane densities conditional on one physical collision.

    At every plane, ``delta`` is the previous vertex's normal separation. The
    conditional one-coordinate density is proportional to
    ``1/(delta+abs(a*u+b*v+c))`` with the other coordinate uniform. Its exact
    primitive is ``sign(t)*log(1+abs(t)/delta)``. A strictly positive uniform
    component prevents unsupported detector regions or proposal-dependent
    target laws. Geometry jets never differentiate this detached proposal.
    """
    redundant_planes = validate_redundant_planes(redundant_planes, spec)
    inverse = spec.object_pose.inverse()
    for corner in product((0.0, 1.0), repeat=3):
        local_corner = cast(
            Vector3,
            tuple(
                spec.grid.origin_mm[axis]
                + corner[axis] * (spec.grid.upper_mm[axis] - spec.grid.origin_mm[axis])
                for axis in range(3)
            ),
        )
        if spec.detector.source_distance(spec.object_pose.point(local_corner)) >= 0.0:
            raise TransportError(
                "geometry detector requires strict separation from the material box"
            )
    previous = inverse.point(previous_point_mm)
    lower = inverse.point(spec.detector.lower_corner_mm)
    u, v = inverse.direction(spec.detector.u), inverse.direction(spec.detector.v)
    width = spec.detector.shape[0] * spec.detector.spacing_xy_mm[0]
    height = spec.detector.shape[1] * spec.detector.spacing_xy_mm[1]
    planes: list[DetectorPlaneProposal] = []
    for axis, count in enumerate(spec.grid.shape_xyz):
        a, b = width * u[axis], height * v[axis]
        if a == 0.0 and b == 0.0:
            continue
        for face in range(count + 1):
            if (axis, face) in redundant_planes:
                continue
            position = spec.grid.origin_mm[axis] + face * spec.grid.spacing_mm[axis]
            separation = abs(previous[axis] - position)
            if not math.isfinite(separation) or separation == 0.0:
                raise TransportError("detector proposal collision lies on a grid-plane boundary")
            planes.append(
                DetectorPlaneProposal(axis, face, (a, b, lower[axis] - position), separation)
            )
    payload = (
        previous_point_mm,
        [(p.axis, p.face, p.coefficients, p.normal_separation_mm) for p in planes],
    )
    identity = hashlib.sha256(
        json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()
    return DetectorAreaProposal(
        tuple(planes),
        identity,
        DetectorReferenceChart(spec.detector, inverse, spec.grid.origin_mm, spec.grid.spacing_mm),
    )


def sample_detector_area_proposal(
    proposal: DetectorAreaProposal,
    *,
    component_draw: float,
    coordinate: tuple[float, float],
) -> DetectorAreaProposalSample:
    """Draw the complete mixture and retain its full density, not its component density."""
    if not math.isfinite(component_draw) or not 0.0 < component_draw < 1.0:
        raise TransportError("detector proposal component draw must lie inside (0,1)")
    if len(coordinate) != 2 or any(
        not math.isfinite(value) or not 0.0 < value < 1.0 for value in coordinate
    ):
        raise TransportError("detector proposal coordinates must lie inside the unit square")
    component, mapped = -1, coordinate
    if proposal.planes and component_draw >= 0.5:
        component = min(
            len(proposal.planes) - 1, int((component_draw - 0.5) * 2.0 * len(proposal.planes))
        )
        plane = proposal.planes[component]
        axis = plane.sampled_axis
        a, b, c = plane.coefficients
        coefficient = (a, b)[axis]
        other = (b, a)[axis] * coordinate[1 - axis] + c
        first, last = (
            _primitive(other, plane.normal_separation_mm),
            _primitive(other + coefficient, plane.normal_separation_mm),
        )
        transformed = (1.0 - coordinate[axis]) * first + coordinate[axis] * last
        level = _inverse_primitive(transformed, plane.normal_separation_mm)
        amount = (level - other) / coefficient
        values = list(coordinate)
        values[axis] = amount
        mapped = cast(tuple[float, float], tuple(values))
        if not 0.0 < amount < 1.0:
            raise TransportError("detector proposal coordinate is not distinguishable in binary64")
        represented = coefficient * amount + other
        reconstructed = (_primitive(represented, plane.normal_separation_mm) - first) / (
            last - first
        )
        uncertainty = (
            16.0
            * (math.ulp(represented) + math.ulp(other))
            / ((plane.normal_separation_mm + abs(represented)) * abs(last - first))
        )
        if (
            not math.isfinite(uncertainty)
            or uncertainty > math.sqrt(math.ulp(1.0))
            or (abs(reconstructed - coordinate[axis]) > uncertainty + 512.0 * math.ulp(1.0))
        ):
            raise TransportError("detector proposal coordinate is not distinguishable in binary64")
    if proposal.chart is not None:
        chart = proposal.chart
        endpoint = detector_vertex_tangent(mapped, chart.detector).point_mm
        local_endpoint = chart.inverse_object_pose.point(endpoint)
        for plane in proposal.planes:
            a, b, c = plane.coefficients
            axis = plane.sampled_axis
            coefficient = (a, b)[axis]
            other = (b, a)[axis] * mapped[1 - axis] + c
            first = _primitive(other, plane.normal_separation_mm)
            last = _primitive(other + coefficient, plane.normal_separation_mm)
            width = last - first
            represented = coefficient * mapped[axis] + other
            face = chart.grid_origin_mm[plane.axis] + plane.face * chart.grid_spacing_mm[plane.axis]
            actual = local_endpoint[plane.axis] - face
            # Every component participates in the full mixture PDF, including
            # when the uniform half was drawn. Verify its CDF at the endpoint
            # which the physical kernels will actually receive.
            uncertainty = (
                16.0
                * (math.ulp(local_endpoint[plane.axis]) + math.ulp(face))
                / ((plane.normal_separation_mm + abs(actual)) * abs(width))
            )
            discrepancy = abs(
                (
                    _primitive(actual, plane.normal_separation_mm)
                    - _primitive(represented, plane.normal_separation_mm)
                )
                / width
            )
            if (
                not math.isfinite(uncertainty)
                or uncertainty > math.sqrt(math.ulp(1.0))
                or discrepancy > uncertainty + 512.0 * math.ulp(1.0)
                or abs(actual - represented)
                / (plane.normal_separation_mm + max(abs(actual), abs(represented)))
                > math.sqrt(math.ulp(1.0))
            ):
                raise TransportError(
                    "detector proposal world endpoint is not distinguishable in binary64"
                )
    return DetectorAreaProposalSample(
        mapped, proposal.log_density(mapped), component, proposal.identity
    )


__all__ = [
    "DetectorAreaProposal",
    "DetectorAreaProposalSample",
    "DetectorPlaneProposal",
    "DetectorReferenceChart",
    "prepare_detector_area_proposal",
    "sample_detector_area_proposal",
]
