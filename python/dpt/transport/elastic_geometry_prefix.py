"""Extinction-adapted detached-reference prefix proposals for elastic geometry.

One terminated path is one statistical observation. Every next-event prefix of
that path shares its original-history denominator; prefixes are never counted
as independent histories. Physical absorption and escape select the path length,
without an additional geometric collision-order importance weight.
"""

from __future__ import annotations

# Reference modules share their checked path-integral primitives internally.
# pyright: reportPrivateUsage=false
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, cast

from dpt.geometry import Vector3, cross, dot

from .elastic_geometry import (
    ElasticGeometryMedium,
    _path_integrand,
    _rayleigh_phase,
)
from .geometry_admission import admit_elastic_geometry_chart
from .geometry_path_measure import (
    collision_vertex_tangent,
    source_flux_tangent,
    source_volume_boundaries,
)
from .geometry_tangents import (
    RayTangent,
    _add,
    _scale,
    _sub,
    cell_segments_tangent,
    normalised_direction_jet,
)
from .model import TransportError, TransportSpec
from .source import ParallelBeam, PointConeSource
from .uncollided_geometry import AcquisitionGeometryTangent


def _uniform(value: float) -> None:
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise TransportError("geometry proposal draws must lie strictly inside (0,1)")


@dataclass(frozen=True, slots=True)
class GeometryEventDraw:
    """One fixed-layout elastic proposal event, with bounded angular trials.

    Survival chooses scattering versus absorption, component chooses coherent
    versus isotropic scattering, flight supplies an exponential optical depth.
    Each angular trial contains polar, rejection and azimuth uniforms. IID and
    complete-path randomised designs may supply these explicitly; no draw is
    silently generated, discarded or reassigned from another event.
    """

    survival: float
    component: float
    flight: float
    angles: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "angles", tuple(tuple(trial) for trial in self.angles))
        for value in (self.survival, self.component, self.flight):
            _uniform(value)
        if not self.angles or any(len(trial) != 3 for trial in self.angles):
            raise TransportError("geometry event requires three-uniform angular trials")
        for trial in self.angles:
            for value in trial:
                _uniform(value)


@dataclass(frozen=True, slots=True)
class GeometryPrefixDraws:
    first_vertex: Vector3
    detector: tuple[float, float]
    events: tuple[GeometryEventDraw, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "first_vertex", tuple(self.first_vertex))
        object.__setattr__(self, "detector", tuple(self.detector))
        object.__setattr__(self, "events", tuple(self.events))
        if len(self.first_vertex) != 3 or len(self.detector) != 2:
            raise TransportError("geometry prefix draws have incompatible coordinate dimensions")
        for value in (*self.first_vertex, *self.detector):
            _uniform(value)
        if not self.events or any(
            not isinstance(cast(object, event), GeometryEventDraw) for event in self.events
        ):
            raise TransportError("geometry prefix requires explicit event records")


@dataclass(frozen=True, slots=True)
class GeometryPrefixHistory:
    """One complete absorbed/escaped reference path and its proposal densities."""

    vertices: tuple[Vector3, ...]
    transition_log_densities: tuple[float, ...]
    detector: tuple[float, float]
    terminal: Literal["absorbed", "escaped"]
    reference_identity: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices", tuple(tuple(vertex) for vertex in self.vertices))
        object.__setattr__(self, "transition_log_densities", tuple(self.transition_log_densities))
        object.__setattr__(self, "detector", tuple(self.detector))
        if not self.vertices or len(self.transition_log_densities) != len(self.vertices) - 1:
            raise TransportError("a prefix record needs one transition per following collision")
        if any(
            len(vertex) != 3
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in vertex)
            for vertex in self.vertices
        ):
            raise TransportError("prefix collision coordinates must lie in the unit cube")
        if len(self.detector) != 2 or any(
            not math.isfinite(value) or not 0.0 < value < 1.0 for value in self.detector
        ):
            raise TransportError("prefix detector coordinates must lie in the unit square")
        if self.terminal not in ("absorbed", "escaped") or any(
            not math.isfinite(value) for value in self.transition_log_densities
        ):
            raise TransportError(
                "a prefix record requires finite densities and physical termination"
            )
        if len(self.reference_identity) != 64 or any(
            character not in "0123456789abcdef" for character in self.reference_identity
        ):
            raise TransportError("prefix reference identity must be a SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class GeometryBoundaryPrefixHistory:
    boundary_coordinate: tuple[float, float]
    history: GeometryPrefixHistory

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary_coordinate", tuple(self.boundary_coordinate))
        if len(self.boundary_coordinate) != 2:
            raise TransportError("boundary continuation requires a two-dimensional coordinate")
        for value in self.boundary_coordinate:
            _uniform(value)
        if not isinstance(cast(object, self.history), GeometryPrefixHistory):
            raise TransportError("boundary continuation requires a completed prefix record")


@dataclass(frozen=True, slots=True)
class ElasticPrefixGeometryEstimate:
    mean: float
    interior: float
    source_boundary: float
    original_histories: int
    prefixes: int
    boundary_original_histories: tuple[tuple[str, int], ...]
    original_observations: tuple[tuple[float, float], ...]
    boundary_observations: tuple[tuple[str, tuple[float, ...]], ...]

    @property
    def derivative(self) -> float:
        return math.fsum((self.interior, self.source_boundary))


def _identity(
    source: ParallelBeam | PointConeSource, spec: TransportSpec, medium: ElasticGeometryMedium
) -> str:
    numerical_tables = tuple(
        None
        if table is None
        else (table.atomic_number, table.momentum_transfer_inverse_angstrom, table.form_factor)
        for table in medium.atomic_tables
    )
    payload = {
        "layout": "geometry-elastic-prefix-v1",
        "source": asdict(source),
        "grid": asdict(spec.grid),
        "pose": asdict(spec.object_pose),
        "detector": asdict(spec.detector),
        "energy": spec.energy_kev,
        "extinction": medium.extinction_mm_inverse,
        "isotropic": medium.isotropic_mm_inverse,
        "rayleigh": medium.rayleigh_mm_inverse,
        "atomic": numerical_tables,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def _angular_density(
    medium: ElasticGeometryMedium, cell: int, energy: float, cosine: float
) -> float:
    total = medium.isotropic_mm_inverse[cell] + medium.rayleigh_mm_inverse[cell]
    if total <= 0.0:
        raise TransportError("a scattering proposal requires a positive scattering partial")
    numerator = medium.isotropic_mm_inverse[cell] / (4.0 * math.pi)
    coherent = medium.rayleigh_mm_inverse[cell]
    if coherent > 0.0:
        table = medium.atomic_tables[cell]
        assert table is not None
        numerator += coherent * _rayleigh_phase(energy, cosine, 0.0, table)[0]
    return numerator / total


def _scatter_direction(incoming: Vector3, cosine: float, azimuth: float) -> Vector3:
    anchor: Vector3 = (0.0, 0.0, 1.0) if abs(incoming[2]) < 0.9 else (1.0, 0.0, 0.0)
    first = cross(incoming, anchor)
    first = _scale(first, 1.0 / math.hypot(*first))
    second = cross(incoming, first)
    radius = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return _add(
        _scale(incoming, cosine),
        _add(_scale(first, radius * math.cos(azimuth)), _scale(second, radius * math.sin(azimuth))),
    )


def _flight_exit(ray: RayTangent, spec: TransportSpec) -> float:
    local = ray.in_object_frame(spec.object_pose, AcquisitionGeometryTangent().object.motion)
    distances: list[float] = []
    for axis, direction in enumerate(local.direction):
        if direction > 0.0:
            distances.append((spec.grid.upper_mm[axis] - local.origin_mm[axis]) / direction)
        elif direction < 0.0:
            distances.append((spec.grid.origin_mm[axis] - local.origin_mm[axis]) / direction)
    result = min(distances)
    if not math.isfinite(result) or result <= 0.0:
        raise TransportError("reference collision must be strictly inside the material box")
    return result


def sample_elastic_geometry_prefix(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    medium: ElasticGeometryMedium,
    draws: GeometryPrefixDraws,
    *,
    first_vertex: Vector3 | None = None,
) -> GeometryPrefixHistory:
    """Trace a detached-base analogue proposal using the supplied coordinate record.

    A supplied ``first_vertex`` binds a separately sampled source-boundary
    continuation. Every finite guard failure invalidates the complete history.
    Base proposal densities are retained for subsequent attached geometry jets.
    """
    if spec.scattering_law != "isotropic-elastic":
        raise TransportError("elastic prefix proposals require fixed-energy scattering")
    if len(medium.extinction_mm_inverse) != math.prod(spec.grid.shape):
        raise TransportError("reference medium must match the grid")
    initial = draws.first_vertex if first_vertex is None else first_vertex
    vertex = collision_vertex_tangent(initial, spec)
    incoming = source_flux_tangent(source, vertex.point_mm, (0.0, 0.0, 0.0)).incoming.direction
    vertices = [initial]
    densities: list[float] = []
    identity = _identity(source, spec, medium)
    for event in draws.events:
        isotropic = medium.isotropic_mm_inverse[vertex.cell]
        coherent = medium.rayleigh_mm_inverse[vertex.cell]
        scattering = isotropic + coherent
        extinction = medium.extinction_mm_inverse[vertex.cell]
        if scattering == 0.0 or event.survival >= scattering / extinction:
            return GeometryPrefixHistory(
                tuple(vertices), tuple(densities), draws.detector, "absorbed", identity
            )
        cosine, azimuth = 2.0 * event.angles[0][0] - 1.0, 2.0 * math.pi * event.angles[0][2]
        if event.component < coherent / scattering:
            table = medium.atomic_tables[vertex.cell]
            assert table is not None
            accepted = False
            from .elastic_geometry import _form_factor

            for polar, rejection, angle in event.angles[: spec.max_angle_trials]:
                cosine = 2.0 * polar - 1.0
                q = spec.energy_kev / 12.398419843320026 * math.sqrt((1.0 - cosine) / 2.0)
                form = _form_factor(table, q)[0] / table.atomic_number
                if rejection < 0.5 * (1.0 + cosine * cosine) * form * form:
                    azimuth, accepted = 2.0 * math.pi * angle, True
                    break
            if not accepted:
                raise TransportError("angular guard invalidates the complete geometry history")
        outgoing = _scatter_direction(incoming, cosine, azimuth)
        ray = RayTangent(vertex.point_mm, outgoing)
        exit_distance = _flight_exit(ray, spec)
        segments = cell_segments_tangent(
            ray, spec.grid, spec.object_pose, maximum_distance_mm=exit_distance
        )
        if len(segments) > spec.max_crossings:
            raise TransportError("crossing guard invalidates the complete geometry history")
        optical_depth = -math.log(event.flight)
        cumulative = 0.0
        distance: float | None = None
        for segment in segments:
            rate = medium.extinction_mm_inverse[segment.cell]
            piece = rate * segment.length_mm
            if rate > 0.0 and optical_depth < cumulative + piece:
                distance = segment.entry_mm + (optical_depth - cumulative) / rate
                break
            cumulative += piece
        if distance is None:
            return GeometryPrefixHistory(
                tuple(vertices), tuple(densities), draws.detector, "escaped", identity
            )
        if len(vertices) >= spec.max_events:
            raise TransportError("event guard invalidates the complete geometry history")
        point = ray.point(distance)
        local = spec.object_pose.inverse().point(point)
        coordinate = cast(
            Vector3,
            tuple(
                (local[i] - spec.grid.origin_mm[i])
                / (spec.grid.shape_xyz[i] * spec.grid.spacing_mm[i])
                for i in range(3)
            ),
        )
        if any(not 0.0 < value < 1.0 for value in coordinate) or coordinate == vertices[-1]:
            raise TransportError("reference collision is not distinguishable in binary64")
        next_vertex = collision_vertex_tangent(coordinate, spec)
        phase = _angular_density(medium, vertex.cell, spec.energy_kev, cosine)
        total_next = medium.extinction_mm_inverse[next_vertex.cell]
        densities.append(
            math.fsum(
                (
                    math.log(scattering / extinction),
                    math.log(phase),
                    math.log(total_next),
                    math.log(next_vertex.volume_jacobian),
                    -2.0 * math.log(distance),
                    -optical_depth,
                )
            )
        )
        vertices.append(coordinate)
        vertex, incoming = next_vertex, outgoing
    raise TransportError("explicit proposal draw budget exhausted before physical termination")


def _prefix_observation(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    medium: ElasticGeometryMedium,
    tangent: AcquisitionGeometryTangent,
    cotangent: tuple[float, ...],
    history: GeometryPrefixHistory,
    *,
    active_source_edge: int | None = None,
) -> tuple[float, float]:
    if history.reference_identity != _identity(source, spec, medium):
        raise TransportError("reference prefix identity does not match the current physical model")
    if history.terminal not in ("absorbed", "escaped") or len(history.transition_log_densities) != (
        len(history.vertices) - 1
    ):
        raise TransportError("prefix history must have a complete physical terminal state")
    if len(history.vertices) > spec.max_events:
        raise TransportError("prefix event guard invalidates the complete geometry estimate")
    values: list[tuple[float, float]] = []
    log_density = 0.0
    points = tuple(collision_vertex_tangent(coordinate, spec) for coordinate in history.vertices)
    incoming = source_flux_tangent(source, points[0].point_mm, (0.0, 0.0, 0.0)).incoming.direction
    # Retained densities are verified against the complete detached physical law.
    # A record cannot change a proposal weight whilst retaining its model digest.
    for index, retained in enumerate(history.transition_log_densities):
        first, second = points[index], points[index + 1]
        displacement = _sub(second.point_mm, first.point_mm)
        distance = math.hypot(*displacement)
        outgoing, _ = normalised_direction_jet(displacement, (0.0, 0.0, 0.0))
        phase = _angular_density(medium, first.cell, spec.energy_kev, dot(incoming, outgoing))
        scattering = (
            medium.isotropic_mm_inverse[first.cell] + medium.rayleigh_mm_inverse[first.cell]
        )
        total, following_total = (
            medium.extinction_mm_inverse[first.cell],
            medium.extinction_mm_inverse[second.cell],
        )
        if scattering <= 0.0 or total <= 0.0 or following_total <= 0.0 or phase <= 0.0:
            raise TransportError("retained prefix transition has zero physical probability")
        segments = cell_segments_tangent(
            RayTangent(first.point_mm, outgoing),
            spec.grid,
            spec.object_pose,
            maximum_distance_mm=distance,
        )
        depth = math.fsum(
            medium.extinction_mm_inverse[segment.cell] * segment.length_mm for segment in segments
        )
        expected = math.fsum(
            (
                math.log(scattering / total),
                math.log(phase),
                math.log(following_total),
                math.log(second.volume_jacobian),
                -2.0 * math.log(distance),
                -depth,
            )
        )
        tolerance = 128.0 * math.ulp(1.0) * (1.0 + abs(expected) + depth)
        if abs(expected - retained) > tolerance:
            raise TransportError("retained prefix density does not match the physical transition")
        incoming = outgoing
    for order in range(1, len(history.vertices) + 1):
        if order > 1:
            log_density += history.transition_log_densities[order - 2]
        values.append(
            _path_integrand(
                source,
                spec,
                medium,
                tangent,
                cotangent,
                history.vertices[:order],
                history.detector,
                active_source_edge=active_source_edge,
                log_importance=-log_density,
            )
        )
    return math.fsum(value for value, _ in values), math.fsum(
        derivative for _, derivative in values
    )


def elastic_prefix_geometry_jvp(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    medium: ElasticGeometryMedium,
    *,
    tangent: AcquisitionGeometryTangent,
    cotangent: Sequence[float],
    histories: Sequence[GeometryPrefixHistory],
    source_boundaries: Mapping[str, Sequence[GeometryBoundaryPrefixHistory]],
) -> ElasticPrefixGeometryEstimate:
    """Contract complete elastic geometry derivatives by original terminated path.

    First collision vertices use uniform material-box coordinates. Every later
    reference transition uses scattering survival, its full angular density and
    analogue extinction flight. Proposal densities remain detached whilst the
    physical collision vertices, paths, coefficients, volume and detector-area
    measures attach to the caller's geometry tangent. Scoring every prefix sums
    the complete collision-order series without a separate order importance law.

    Each original history contributes the sum of all its correlated next-event
    prefixes. The returned original observations retain this grouping. Each
    separately sampled source-boundary continuation has its own original-history
    mean and grouping. An IID uncertainty reduction must use these observations,
    never the number of prefixes; QMC requires independent randomisations.
    """
    if spec.scattering_law != "isotropic-elastic":
        raise TransportError("elastic prefix geometry requires a fixed-energy physical model")
    if len(medium.extinction_mm_inverse) != math.prod(spec.grid.shape):
        raise TransportError("elastic medium must match the material grid")
    seeds = tuple(cotangent)
    if len(seeds) != spec.detector.pixels or any(not math.isfinite(value) for value in seeds):
        raise TransportError("geometry cotangent must have one finite signed value per pixel")
    if not histories:
        raise TransportError("geometry prefix histories must be nonempty")
    admit_elastic_geometry_chart(source, spec, medium, tangent)
    boundaries = source_volume_boundaries(source, spec, tangent.object, tangent.source)
    if set(source_boundaries) != {boundary.key for boundary in boundaries}:
        raise TransportError("all source-boundary continuation histories are required")
    observations = tuple(
        _prefix_observation(source, spec, medium, tangent, seeds, history) for history in histories
    )
    boundary_observations: list[tuple[str, tuple[float, ...]]] = []
    for boundary in boundaries:
        records = source_boundaries[boundary.key]
        if not records:
            raise TransportError("each source boundary requires original continuation histories")
        values: list[float] = []
        edge = int(boundary.key.split(":")[1])
        for record in records:
            sampled = boundary.sample(record.boundary_coordinate)
            assert sampled is not None
            point, weight = sampled
            if record.history.vertices[0] != point:
                raise TransportError(
                    "source-boundary continuation is not bound to its sampled point"
                )
            value = _prefix_observation(
                source, spec, medium, tangent, seeds, record.history, active_source_edge=edge
            )[0]
            values.append(weight * value)
        boundary_observations.append((boundary.key, tuple(values)))
    return ElasticPrefixGeometryEstimate(
        math.fsum(value for value, _ in observations) / len(observations),
        math.fsum(derivative for _, derivative in observations) / len(observations),
        math.fsum(math.fsum(values) / len(values) for _, values in boundary_observations),
        len(histories),
        sum(len(history.vertices) for history in histories),
        tuple((key, len(values)) for key, values in boundary_observations),
        observations,
        tuple(boundary_observations),
    )


__all__ = [
    "ElasticPrefixGeometryEstimate",
    "GeometryBoundaryPrefixHistory",
    "GeometryEventDraw",
    "GeometryPrefixDraws",
    "GeometryPrefixHistory",
    "elastic_prefix_geometry_jvp",
    "sample_elastic_geometry_prefix",
]
