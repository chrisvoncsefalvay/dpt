"""Collision-volume Monte Carlo geometry derivatives for elastic transport.

The collision-order Neumann series includes all positive orders. This geometric
order proposal has no general finite-variance guarantee; the detached physical
prefix estimator supplies the production proposal. A separate
uncollided estimator supplies order zero. Fixed reference vertices remove the
moving material-volume measure; source-footprint planes supply the remaining
support terms, and detector area is differentiated as a moving measure.
"""

from __future__ import annotations

# The geometry modules share their checked vector arithmetic internally.
# pyright: reportPrivateUsage=false
import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, cast

from dpt.geometry import Vector3, dot

from .geometry_admission import admit_elastic_geometry_chart
from .geometry_path_measure import (
    collision_vertex_tangent,
    detector_vertex_tangent,
    source_flux_tangent,
    source_volume_boundaries,
)
from .geometry_tangents import RayTangent, _sub, cell_segments_tangent, normalised_direction_jet
from .model import TransportError, TransportSpec
from .physical import HC_KEV_ANGSTROM, AtomicScatteringTable
from .source import ParallelBeam, PointConeSource
from .uncollided_geometry import AcquisitionGeometryTangent


@dataclass(frozen=True, slots=True)
class CollisionOrderLaw:
    """Positive collision orders with ``P(N=n)=q*(1-q)**(n-1)``.

    Every order has positive probability; ``q=1`` would silently discard
    multiple scattering and is rejected. A sample above the explicit event
    guard invalidates the estimate and is never redrawn or treated as escape.
    """

    stopping_probability: float = 0.5

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.stopping_probability)
            or not 0.0 < self.stopping_probability < 1.0
        ):
            raise TransportError("collision-order stopping probability must lie in (0,1)")

    def inverse_probability(self, order: int) -> float:
        if type(order) is not int or order <= 0:
            raise TransportError("a collision path must have a positive integer order")
        try:
            result = math.exp(
                -math.log(self.stopping_probability)
                - (order - 1) * math.log1p(-self.stopping_probability)
            )
        except OverflowError as error:
            raise TransportError("collision-order importance weight overflows binary64") from error
        return result

    def sample_order(self, coordinate: float) -> int:
        """Invert one uniform coordinate without conditioning on a resource guard."""
        if not math.isfinite(coordinate) or not 0.0 < coordinate < 1.0:
            raise TransportError("collision-order coordinate must lie inside (0,1)")
        return 1 + math.floor(math.log1p(-coordinate) / math.log1p(-self.stopping_probability))


@dataclass(frozen=True, slots=True)
class ElasticGeometryMedium:
    """Fixed extinction and elastic collision partials in x-fastest cell order.

    Extinction and partials include material and voxel density, in mm inverse.
    The collision kernel is ``isotropic/(4*pi) + rayleigh*phase(E,cosine)``;
    each nonzero coherent partial has an immutable represented atomic table.
    Absorption is the nonnegative difference from extinction. These physical
    coefficients are held fixed whilst geometry moves.
    """

    extinction_mm_inverse: tuple[float, ...]
    isotropic_mm_inverse: tuple[float, ...]
    rayleigh_mm_inverse: tuple[float, ...] = ()
    atomic_tables: tuple[AtomicScatteringTable | None, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "extinction_mm_inverse",
            "isotropic_mm_inverse",
            "rayleigh_mm_inverse",
            "atomic_tables",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        count = len(self.extinction_mm_inverse)
        if not count or len(self.isotropic_mm_inverse) != count:
            raise TransportError("elastic partials must match a nonempty extinction field")
        if not self.rayleigh_mm_inverse:
            object.__setattr__(self, "rayleigh_mm_inverse", (0.0,) * count)
        if not self.atomic_tables:
            object.__setattr__(self, "atomic_tables", (None,) * count)
        if len(self.rayleigh_mm_inverse) != count or len(self.atomic_tables) != count:
            raise TransportError("coherent partials and atomic tables must match the cell field")
        for total, isotropic, coherent, table in zip(
            self.extinction_mm_inverse,
            self.isotropic_mm_inverse,
            self.rayleigh_mm_inverse,
            self.atomic_tables,
            strict=True,
        ):
            if (
                not all(
                    math.isfinite(value) and value >= 0.0 for value in (total, isotropic, coherent)
                )
                or isotropic + coherent > total
            ):
                raise TransportError(
                    "elastic partials must be finite and sum to at most extinction"
                )
            if coherent > 0.0 and not isinstance(table, AtomicScatteringTable):
                raise TransportError("a coherent partial requires its represented atomic table")


@dataclass(frozen=True, slots=True)
class CollisionPathSample:
    """Reference collision vertices and detector point for one sampled order.

    Vertices are independent uniform points of the unit cube, conditional on
    the original order draw; detector coordinates are uniform on the square.
    IID or independently randomised complete-path designs are valid. The
    caller retains coordinate provenance, including the collision-order draw.
    """

    vertices: tuple[Vector3, ...]
    detector: tuple[float, float]
    vertex_law: Literal["uniform-volume", "reference-ray-volume"] = "uniform-volume"

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices", tuple(tuple(vertex) for vertex in self.vertices))
        object.__setattr__(self, "detector", tuple(self.detector))
        if not self.vertices or any(
            len(vertex) != 3
            or not all(math.isfinite(value) and 0.0 < value < 1.0 for value in vertex)
            for vertex in self.vertices
        ):
            raise TransportError("collision samples require interior three-dimensional vertices")
        if len(self.detector) != 2 or not all(
            math.isfinite(value) and 0.0 < value < 1.0 for value in self.detector
        ):
            raise TransportError("detector samples require interior two-dimensional coordinates")
        if self.vertex_law not in ("uniform-volume", "reference-ray-volume"):
            raise TransportError("unsupported reference collision-vertex law")


@dataclass(frozen=True, slots=True)
class SourceBoundaryPathSample:
    """A first collision on a support plane, followed by ordinary path vertices."""

    boundary: tuple[float, float]
    following_vertices: tuple[Vector3, ...]
    detector: tuple[float, float]
    vertex_law: Literal["uniform-volume", "reference-ray-volume"] = "uniform-volume"

    def __post_init__(self) -> None:
        validated = CollisionPathSample(
            ((0.5, 0.5, 0.5), *self.following_vertices), self.detector, self.vertex_law
        )
        object.__setattr__(self, "following_vertices", validated.vertices[1:])
        object.__setattr__(self, "detector", validated.detector)
        object.__setattr__(self, "boundary", tuple(self.boundary))
        if len(self.boundary) != 2 or not all(
            math.isfinite(value) and 0.0 < value < 1.0 for value in self.boundary
        ):
            raise TransportError("source-boundary coordinates must lie inside the unit square")


def _reference_exit(point: Vector3, direction: Vector3) -> float:
    return min(
        (1.0 - point[i]) / value if value > 0.0 else -point[i] / value
        for i, value in enumerate(direction)
        if value != 0.0
    )


def sample_reference_collision_vertex(point: Vector3, coordinate: Vector3) -> Vector3:
    """Sample inverse-square volume density in a fixed unit-cube chart.

    Two uniforms choose a uniform spherical direction. The third chooses a
    uniform distance to the cube exit. The density per reference volume is
    ``1/(4*pi*exit_distance*distance**2)`` and has full interior support. It
    cancels coincident-vertex inverse-square flight singularities in path
    importance weights. Geometry remains attached to the sampled local vertex.
    """
    if (
        len(point) != 3
        or len(coordinate) != 3
        or not all(math.isfinite(value) and 0.0 < value < 1.0 for value in (*point, *coordinate))
    ):
        raise TransportError("reference ray-volume draws require interior three-dimensional points")
    cosine = 2.0 * coordinate[0] - 1.0
    radius = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    azimuth = 2.0 * math.pi * coordinate[1]
    direction = (radius * math.cos(azimuth), radius * math.sin(azimuth), cosine)
    distance = coordinate[2] * _reference_exit(point, direction)
    result = cast(Vector3, tuple(point[i] + distance * direction[i] for i in range(3)))
    if result == point or any(not 0.0 < value < 1.0 for value in result):
        raise TransportError("reference collision draw is not distinguishable in binary64")
    return result


@dataclass(frozen=True, slots=True)
class ElasticGeometryEstimate:
    mean: float
    interior: float
    source_boundary: float
    histories: int
    boundary_histories: tuple[tuple[str, int], ...]

    @property
    def derivative(self) -> float:
        return math.fsum((self.interior, self.source_boundary))


def _form_factor(table: AtomicScatteringTable, q: float) -> tuple[float, float]:
    nodes = table.momentum_transfer_inverse_angstrom
    if not nodes[0] <= q <= nodes[-1]:
        raise TransportError("Rayleigh momentum transfer leaves the represented table")
    index = min(len(nodes) - 2, max(0, bisect_right(nodes, q) - 1))
    left, right = nodes[index : index + 2]
    slope = (table.form_factor[index + 1] - table.form_factor[index]) / (right - left)
    return table.form_factor[index] + (q - left) * slope, slope


def _rayleigh_phase(
    energy: float, cosine: float, cosine_dot: float, table: AtomicScatteringTable
) -> tuple[float, float]:
    qmax = energy / HC_KEV_ANGSTROM
    if table.momentum_transfer_inverse_angstrom[-1] < qmax:
        raise TransportError("Rayleigh table must cover the entire source-energy angular support")
    nodes = (-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526)
    weights = (0.34785484513745385, 0.6521451548625461, 0.6521451548625461, 0.34785484513745385)
    integrals: list[float] = []
    for left, raw_right in pairwise(table.momentum_transfer_inverse_angstrom):
        right = min(raw_right, qmax)
        if right <= left:
            continue
        radius, midpoint = (right - left) / 2.0, (right + left) / 2.0
        for node, weight in zip(nodes, weights, strict=True):
            q = midpoint + radius * node
            c = 1.0 - 2.0 * (q / qmax) ** 2
            form = _form_factor(table, q)[0]
            integrals.append(radius * weight * q * form * form * (1.0 + c * c))
    normal = 2.0 * math.pi * math.fsum(integrals) * 4.0 / (qmax * qmax)
    q = qmax * math.sqrt(max(0.0, (1.0 - cosine) / 2.0))
    form, slope = _form_factor(table, q)
    numerator = (1.0 + cosine * cosine) * form * form
    numerator_dot = 2.0 * cosine * cosine_dot * form * form
    if q != 0.0:
        q_dot = -qmax * qmax * cosine_dot / (4.0 * q)
        numerator_dot += (1.0 + cosine * cosine) * 2.0 * form * slope * q_dot
    elif cosine_dot != 0.0:
        raise TransportError("the Rayleigh forward-angle jet is outside its smooth chart")
    return numerator / normal, numerator_dot / normal


def _segment_depth(
    ray: RayTangent,
    length: float,
    length_dot: float,
    spec: TransportSpec,
    medium: ElasticGeometryMedium,
    tangent: AcquisitionGeometryTangent,
) -> tuple[float, float]:
    segments = cell_segments_tangent(
        ray,
        spec.grid,
        spec.object_pose,
        tangent.object,
        maximum_distance_mm=length,
        maximum_distance_dot=length_dot,
    )
    return (
        math.fsum(medium.extinction_mm_inverse[s.cell] * s.length_mm for s in segments),
        math.fsum(medium.extinction_mm_inverse[s.cell] * s.length_dot for s in segments),
    )


def _path_integrand(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    medium: ElasticGeometryMedium,
    tangent: AcquisitionGeometryTangent,
    cotangent: tuple[float, ...],
    vertices: tuple[Vector3, ...],
    detector_coordinate: tuple[float, float],
    *,
    active_source_edge: int | None = None,
    vertex_law: str = "uniform-volume",
    log_importance: float = 0.0,
) -> tuple[float, float]:
    points = tuple(
        collision_vertex_tangent(coordinate, spec, tangent.object) for coordinate in vertices
    )
    detector = detector_vertex_tangent(
        detector_coordinate, spec.detector, tangent.detector, tangent.detector_spacing_dot
    )
    seed = cotangent[detector.pixel]
    if seed == 0.0 or source.weight == 0.0:
        return 0.0, 0.0
    flux = source_flux_tangent(source, points[0].point_mm, points[0].point_dot, tangent.source)
    if any(value <= 0.0 for index, value in enumerate(flux.support) if index != active_source_edge):
        return 0.0, 0.0
    factors: list[tuple[float, float]] = [(flux.density, flux.density_dot)]
    depth, depth_dot = _segment_depth(
        flux.incoming, flux.distance_mm, flux.distance_dot, spec, medium, tangent
    )
    incoming, incoming_dot = flux.incoming.direction, flux.incoming.direction_dot
    for index, vertex in enumerate(points):
        next_point = points[index + 1].point_mm if index + 1 < len(points) else detector.point_mm
        next_dot = points[index + 1].point_dot if index + 1 < len(points) else detector.point_dot
        displacement = _sub(next_point, vertex.point_mm)
        displacement_dot = _sub(next_dot, vertex.point_dot)
        outgoing, outgoing_dot = normalised_direction_jet(displacement, displacement_dot)
        distance = math.hypot(*displacement)
        distance_dot = dot(outgoing, displacement_dot)
        cosine = max(-1.0, min(1.0, dot(incoming, outgoing)))
        cosine_dot = dot(incoming_dot, outgoing) + dot(incoming, outgoing_dot)
        phase = medium.isotropic_mm_inverse[vertex.cell] / (4.0 * math.pi)
        phase_dot = 0.0
        coherent = medium.rayleigh_mm_inverse[vertex.cell]
        if coherent > 0.0:
            table = medium.atomic_tables[vertex.cell]
            assert table is not None
            probability, probability_dot = _rayleigh_phase(
                spec.energy_kev, cosine, cosine_dot, table
            )
            phase += coherent * probability
            phase_dot += coherent * probability_dot
        if phase == 0.0:
            return 0.0, 0.0
        if vertex_law == "reference-ray-volume" and index + 1 < len(points):
            reference_displacement = _sub(vertices[index + 1], vertices[index])
            reference_distance = math.hypot(*reference_displacement)
            if reference_distance == 0.0:
                raise TransportError("coincident reference collision vertices are invalid")
            reference_direction = cast(
                Vector3, tuple(value / reference_distance for value in reference_displacement)
            )
            geometric_factor = (
                4.0
                * math.pi
                * _reference_exit(vertices[index], reference_direction)
                * (reference_distance / distance) ** 2
            )
        else:
            geometric_factor = 1.0 / (distance * distance)
        factors.extend(
            (
                (vertex.volume_jacobian, vertex.volume_jacobian_dot),
                (phase, phase_dot),
                (geometric_factor, -2.0 * distance_dot / distance * geometric_factor),
            )
        )
        ray = RayTangent(vertex.point_mm, outgoing, vertex.point_dot, outgoing_dot)
        segment, segment_dot = _segment_depth(ray, distance, distance_dot, spec, medium, tangent)
        depth += segment
        depth_dot += segment_dot
        incoming, incoming_dot = outgoing, outgoing_dot
    projection = dot(incoming, detector.normal)
    projection_dot = dot(incoming_dot, detector.normal) + dot(incoming, detector.normal_dot)
    if projection <= 0.0:
        raise TransportError("a collision-to-detector segment must be outgoing")
    factors.extend(
        ((projection, projection_dot), (detector.area_jacobian, detector.area_jacobian_dot))
    )
    if spec.scoring == "energy-kev":
        factors.append((spec.energy_kev, 0.0))
    if any(
        value <= 0.0 or not math.isfinite(value) or not math.isfinite(derivative)
        for value, derivative in factors
    ):
        raise TransportError("path measure is outside its finite positive-factor chart")
    logarithm = math.fsum(
        (math.log(abs(seed)), *(math.log(value) for value, _ in factors), -depth, log_importance)
    )
    score_dot = math.fsum((*(derivative / value for value, derivative in factors), -depth_dot))
    try:
        value = math.copysign(math.exp(logarithm), seed)
        derivative = (
            0.0
            if score_dot == 0.0
            else math.copysign(math.exp(logarithm + math.log(abs(score_dot))), seed * score_dot)
        )
    except OverflowError as error:
        raise TransportError("geometry path contribution overflows binary64") from error
    return value, derivative


def elastic_geometry_jvp(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    medium: ElasticGeometryMedium,
    *,
    tangent: AcquisitionGeometryTangent,
    cotangent: Sequence[float],
    order_law: CollisionOrderLaw,
    histories: Sequence[CollisionPathSample],
    source_boundaries: Mapping[str, Sequence[SourceBoundaryPathSample]],
) -> ElasticGeometryEstimate:
    """Estimate all positive collision orders and their full geometry derivative.

    Every sampled order is divided by its geometric probability. Uniform local
    collision vertices carry one complete-box volume Jacobian per collision;
    normalised outgoing phase densities carry inverse-square flight Jacobians,
    and the detector endpoint carries its area and projected-solid-angle factor.
    The source footprint's four support planes are sampled separately. Interior
    and boundary histories have separate means and require complete, independent
    randomisation to estimate uncertainty. No uncertainty is fabricated here.

    This reference integrates absorption, isotropic elastic and represented
    Rayleigh scattering at fixed energy. Compton/fluorescence require joint
    energy-transfer kernels and energy-support boundary charts. Forward analogue,
    continuous-absorption, Woodcock, NEE, splitting and roulette policies estimate
    the same elastic expected image; this independent path measure differentiates
    that expectation without replaying or changing their random coordinates.
    """
    if spec.scattering_law != "isotropic-elastic":
        raise TransportError("elastic geometry requires a constant-energy scattering model")
    if len(medium.extinction_mm_inverse) != math.prod(spec.grid.shape):
        raise TransportError("elastic geometry medium must match the material grid")
    seeds = tuple(cotangent)
    if len(seeds) != spec.detector.pixels or not all(math.isfinite(value) for value in seeds):
        raise TransportError("cotangent must match the detector and contain finite signed values")
    if not histories:
        raise TransportError("geometry histories must be nonempty")
    admit_elastic_geometry_chart(source, spec, medium, tangent)
    boundaries = source_volume_boundaries(source, spec, tangent.object, tangent.source)
    if set(source_boundaries) != {boundary.key for boundary in boundaries}:
        raise TransportError("all source-support boundary histories are required")
    values: list[tuple[float, float]] = []
    for history in histories:
        if len(history.vertices) > spec.max_events:
            raise TransportError("collision-order guard invalidates the complete geometry estimate")
        value, derivative = _path_integrand(
            source,
            spec,
            medium,
            tangent,
            seeds,
            history.vertices,
            history.detector,
            vertex_law=history.vertex_law,
        )
        inverse_probability = order_law.inverse_probability(len(history.vertices))
        values.append((value * inverse_probability, derivative * inverse_probability))
    terms: list[float] = []
    counts: list[tuple[str, int]] = []
    for boundary in boundaries:
        records = source_boundaries[boundary.key]
        if not records:
            raise TransportError("each source boundary requires nonempty histories")
        edge = int(boundary.key.split(":")[1])
        local_terms: list[float] = []
        for record in records:
            sample = boundary.sample(record.boundary)
            assert sample is not None
            vertex, weight = sample
            vertices = (vertex, *record.following_vertices)
            if len(vertices) > spec.max_events:
                raise TransportError("boundary collision-order guard invalidates the estimate")
            value = _path_integrand(
                source,
                spec,
                medium,
                tangent,
                seeds,
                vertices,
                record.detector,
                active_source_edge=edge,
                vertex_law=record.vertex_law,
            )[0]
            local_terms.append(value * weight * order_law.inverse_probability(len(vertices)))
        terms.append(math.fsum(local_terms) / len(local_terms))
        counts.append((boundary.key, len(records)))
    return ElasticGeometryEstimate(
        math.fsum(value for value, _ in values) / len(values),
        math.fsum(derivative for _, derivative in values) / len(values),
        math.fsum(terms),
        len(values),
        tuple(counts),
    )


__all__ = [
    "CollisionOrderLaw",
    "CollisionPathSample",
    "ElasticGeometryEstimate",
    "ElasticGeometryMedium",
    "SourceBoundaryPathSample",
    "elastic_geometry_jvp",
    "sample_reference_collision_vertex",
]
