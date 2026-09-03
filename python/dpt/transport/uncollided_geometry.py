"""Edge-sampled geometry derivatives of the uncollided transport component.

This component integrates the original source law, including photons that miss
the object or detector. Extinction includes every absorption and scattering
channel. Interior face jets and detector/material boundary samples have separate
denominators and are never substituted by finite differences of forward images.
Scattered transport is a separate term in the transport expansion.
"""

from __future__ import annotations

# The geometry modules share their checked vector arithmetic internally.
# pyright: reportPrivateUsage=false
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import Literal, cast

from dpt.geometry import Vector3, dot

from .geometry_tangents import (
    AffineBoundary,
    GridGeometryTangent,
    RayTangent,
    RigidMotion,
    _add,
    _sub,
    cell_segments_tangent,
    detector_corner_velocity,
    detector_hit_tangent,
    normalised_direction_jet,
)
from .model import TransportError, TransportSpec
from .source import ParallelBeam, PointConeSource


@dataclass(frozen=True, slots=True)
class SourceGeometryTangent:
    """Attached source geometry, preserving its declared normalisation.

    For a parallel beam, ``motion`` moves the source rectangle and direction;
    ``extent_dot`` changes its two full extents about the lower corner. For a
    cone, ``motion`` moves the emitting point while ``target_motion`` moves its
    explicitly declared launch rectangle. Moving the measurement detector does
    not silently change the source law: use the same target and detector motion
    to differentiate a physically shared acquisition rectangle.
    """

    motion: RigidMotion = field(default_factory=RigidMotion)
    target_motion: RigidMotion = field(default_factory=RigidMotion)
    extent_dot: tuple[float, float] = (0.0, 0.0)
    target_spacing_dot: tuple[float, float] = (0.0, 0.0)
    direction_dot: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        values = tuple(self.extent_dot)
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise TransportError("source extent tangent must have two finite components")
        object.__setattr__(self, "extent_dot", values)
        target_values = tuple(self.target_spacing_dot)
        if len(target_values) != 2 or not all(math.isfinite(value) for value in target_values):
            raise TransportError("launch-target spacing tangent must have two finite components")
        object.__setattr__(self, "target_spacing_dot", target_values)
        direction_values = tuple(self.direction_dot)
        if len(direction_values) != 3 or not all(
            math.isfinite(value) for value in direction_values
        ):
            raise TransportError("beam direction tangent must have three finite components")
        object.__setattr__(self, "direction_dot", direction_values)


@dataclass(frozen=True, slots=True)
class AcquisitionGeometryTangent:
    """Independent object, source and measurement-detector geometry direction."""

    object: GridGeometryTangent = field(default_factory=GridGeometryTangent)
    source: SourceGeometryTangent = field(default_factory=SourceGeometryTangent)
    detector: RigidMotion = field(default_factory=RigidMotion)
    detector_spacing_dot: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        values = tuple(self.detector_spacing_dot)
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise TransportError("detector spacing tangent must have two finite components")
        object.__setattr__(self, "detector_spacing_dot", values)


@dataclass(frozen=True, slots=True)
class UncollidedGeometryEstimate:
    """Contracted forward mean and the three terms of its directional derivative.

    ``interior`` includes all attached path geometry and attenuation lengths.
    ``material_boundary`` samples hard material-support jumps of parallel rays.
    ``detector_boundary`` samples every outer and interior pixel edge. An
    arbitrary signed image cotangent is contracted before all three reductions.
    """

    mean: float
    interior: float
    material_boundary: float
    detector_boundary: float
    interior_samples: int
    boundary_samples: tuple[tuple[str, int], ...]

    @property
    def derivative(self) -> float:
        return math.fsum((self.interior, self.material_boundary, self.detector_boundary))


def _source_ray(
    source: ParallelBeam | PointConeSource,
    coordinate: tuple[float, float],
    tangent: SourceGeometryTangent,
) -> tuple[RayTangent, Vector3, Vector3]:
    u, v = coordinate
    if not all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in coordinate):
        raise TransportError("source sample coordinates must lie in the closed unit square")
    if isinstance(source, ParallelBeam):
        position = _add(
            source.lower_mm, (u * source.extent_xy_mm[0], v * source.extent_xy_mm[1], 0.0)
        )
        position_dot = _add(
            tangent.motion.point_velocity(position),
            (u * tangent.extent_dot[0], v * tangent.extent_dot[1], 0.0),
        )
        direction_dot = _add(
            tangent.motion.direction_velocity(source.direction), tangent.direction_dot
        )
        ray = RayTangent(position, source.direction, position_dot, direction_dot)
        return ray, ray.direction, ray.direction_dot
    if tangent.extent_dot != (0.0, 0.0):
        raise TransportError("cone launch extents belong to its explicit target geometry")
    if tangent.direction_dot != (0.0, 0.0, 0.0):
        raise TransportError("cone directions belong to its emitting point and launch target")
    geometry = source.geometry
    target = cast(
        Vector3,
        tuple(
            geometry.origin_mm[i]
            + (u * geometry.shape[1] - 0.5) * geometry.spacing_mm[0] * geometry.u[i]
            + (v * geometry.shape[0] - 0.5) * geometry.spacing_mm[1] * geometry.v[i]
            for i in range(3)
        ),
    )
    position = geometry.source_mm
    position_dot = tangent.motion.point_velocity(position)
    displacement = _sub(target, position)
    target_dot = cast(
        Vector3,
        tuple(
            tangent.target_motion.point_velocity(target)[i]
            + (u * geometry.shape[1] - 0.5) * tangent.target_spacing_dot[0] * geometry.u[i]
            + (v * geometry.shape[0] - 0.5) * tangent.target_spacing_dot[1] * geometry.v[i]
            for i in range(3)
        ),
    )
    displacement_dot = _sub(target_dot, position_dot)
    direction, direction_dot = normalised_direction_jet(displacement, displacement_dot)
    return (
        RayTangent(position, direction, position_dot, direction_dot),
        (displacement),
        displacement_dot,
    )


def _weighted_transmission(depth: float, *factors: float) -> float:
    """Combine attenuation and signed factor exponents before final rounding."""
    if any(value == 0.0 for value in factors):
        return 0.0
    sign = math.prod(-1 if value < 0.0 else 1 for value in factors)
    logarithm = math.fsum((*(math.log(abs(value)) for value in factors), -depth))
    try:
        value = sign * math.exp(logarithm)
    except OverflowError as error:
        raise TransportError("the requested geometry contribution overflows binary64") from error
    if not math.isfinite(value):
        raise TransportError("the requested geometry contribution is not finite")
    return value


def _ray_measure(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    rates: tuple[float, ...],
    cotangent: tuple[float, ...],
    tangent: AcquisitionGeometryTangent,
    coordinate: tuple[float, float],
    *,
    material_side: tuple[int, int, int] | None = None,
    detector_side: tuple[int, int, int] | None = None,
) -> tuple[float, float]:
    ray, _, _ = _source_ray(source, coordinate, tangent.source)
    numerator = dot(_sub(spec.detector.lower_corner_mm, ray.origin_mm), spec.detector.normal)
    denominator = dot(ray.direction, spec.detector.normal)
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise TransportError("primary detector intersection is not representable in binary64")
    if denominator <= 0.0 or numerator <= 0.0:
        return 0.0, 0.0
    hit = detector_hit_tangent(
        ray, spec.detector, tangent.detector, spacing_dot=tangent.detector_spacing_dot
    )
    pixel = hit.pixel
    if detector_side is not None:
        axis, boundary_index, sign = detector_side
        coordinates = [hit.column, hit.row]
        indices = [math.floor(x) for x in coordinates]
        indices[axis] = boundary_index - int(sign < 0)
        x, y = indices
        width, height = spec.detector.shape
        pixel = y * width + x if 0 <= x < width and 0 <= y < height else -1
    if pixel < 0 or cotangent[pixel] == 0.0:
        return 0.0, 0.0
    side = cast(tuple[int, int, Literal[-1, 1]] | None, material_side)
    segments = cell_segments_tangent(
        ray,
        spec.grid,
        spec.object_pose,
        tangent.object,
        maximum_distance_mm=hit.distance_mm,
        maximum_distance_dot=hit.distance_dot,
        side=side,
    )
    depth = math.fsum(rates[segment.cell] * segment.length_mm for segment in segments)
    depth_dot = math.fsum(rates[segment.cell] * segment.length_dot for segment in segments)
    score = spec.energy_kev if spec.scoring == "energy-kev" else 1.0
    value = _weighted_transmission(depth, source.weight, score, cotangent[pixel])
    derivative = _weighted_transmission(depth, source.weight, score, cotangent[pixel], -depth_dot)
    return value, derivative


def _affine_values(function: Callable[[tuple[float, float]], float]) -> tuple[float, float, float]:
    c = function((0.0, 0.0))
    return function((1.0, 0.0)) - c, function((0.0, 1.0)) - c, c


def _detector_boundary(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    tangent: AcquisitionGeometryTangent,
    axis: int,
    boundary_index: int,
) -> AffineBoundary | None:
    detector = spec.detector
    if isinstance(source, ParallelBeam):
        launches = tuple(
            _source_ray(source, uv, tangent.source)[0]
            for uv in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
        )
        if all(
            dot(ray.direction, detector.normal) <= 0.0
            and dot(_sub(detector.lower_corner_mm, ray.origin_mm), detector.normal) > 0.0
            for ray in launches
        ):
            # A strictly separated finite detector has no primary support in
            # a neighbourhood of an away-facing or parallel launch direction.
            return None
    basis = (detector.u, detector.v)[axis]
    spacing = detector.spacing_xy_mm[axis]
    corner, normal = detector.lower_corner_mm, detector.normal
    corner_dot = detector_corner_velocity(detector, tangent.detector, tangent.detector_spacing_dot)
    basis_dot = tangent.detector.direction_velocity(basis)
    normal_dot = tangent.detector.direction_velocity(normal)
    spacing_dot = tangent.detector_spacing_dot[axis]

    def constraint(coordinate: tuple[float, float]) -> tuple[float, float]:
        ray, displacement, displacement_dot = _source_ray(source, coordinate, tangent.source)
        offset = _sub(ray.origin_mm, corner)
        offset_dot = _sub(ray.origin_dot, corner_dot)
        first = dot(offset, basis) - boundary_index * spacing
        first_dot = dot(offset_dot, basis) + dot(offset, basis_dot) - (boundary_index * spacing_dot)
        denominator = dot(displacement, normal)
        denominator_dot = dot(displacement_dot, normal) + dot(displacement, normal_dot)
        plane = -dot(offset, normal)
        plane_dot = -dot(offset_dot, normal) - dot(offset, normal_dot)
        along = dot(displacement, basis)
        along_dot = dot(displacement_dot, basis) + dot(displacement, basis_dot)
        return first * denominator + plane * along, math.fsum(
            (
                first_dot * denominator,
                first * denominator_dot,
                plane_dot * along,
                plane * along_dot,
            )
        )

    coefficients = _affine_values(lambda coordinate: constraint(coordinate)[0])
    derivatives = _affine_values(lambda coordinate: constraint(coordinate)[1])
    if math.hypot(*coefficients[:2]) == 0.0:
        if coefficients[2] == 0.0:
            # A fixed launch with a translating detector has an exactly
            # affine edge constraint. A zero coefficient derivative then
            # preserves this half-open ownership throughout that coordinate.
            # Merely checking a zero first derivative of a rotating chart
            # would not establish the same statement.
            if (
                derivatives == (0.0, 0.0, 0.0)
                and tangent.source == SourceGeometryTangent()
                and tangent.detector.angular == (0.0, 0.0, 0.0)
                and tangent.detector_spacing_dot == (0.0, 0.0)
            ):
                return None
            raise TransportError("the whole source is coincident with a detector pixel edge")
        return None
    boundary = AffineBoundary(f"detector:{axis}:{boundary_index}", coefficients, derivatives)
    return boundary if boundary.endpoints() is not None else None


def _material_boundaries(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    tangent: AcquisitionGeometryTangent,
    *,
    equivalent_whole_faces: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[tuple[AffineBoundary, int, int], ...]:
    if isinstance(source, PointConeSource):
        inverse = spec.object_pose.inverse()
        positions = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        launched = tuple(_source_ray(source, uv, tangent.source) for uv in positions)
        samples = tuple(
            ray.in_object_frame(spec.object_pose, tangent.object.motion) for ray, _, _ in launched
        )
        local = samples[0].origin_mm
        displacement = tuple(inverse.direction(value) for _, value, _ in launched)
        common_motion = (
            tangent.source.motion == tangent.source.target_motion == tangent.object.motion
            and tangent.source.target_spacing_dot == (0.0, 0.0)
        )
        displacement_dot = tuple(
            (0.0, 0.0, 0.0)
            if common_motion
            else inverse.direction(_sub(velocity, tangent.object.motion.direction_velocity(value)))
            for _, value, velocity in launched
        )
        out: list[tuple[AffineBoundary, int, int]] = []
        for axis, count in enumerate(spec.grid.shape_xyz):
            for index in range(count + 1):
                if local[axis] == spec.grid.origin_mm[axis] + index * spec.grid.spacing_mm[axis]:
                    face_dot = (
                        tangent.object.origin_dot[axis] + index * tangent.object.spacing_dot[axis]
                    )
                    # With the emitter attached to the plane, its angular
                    # support is n·(target(U)-emitter)=0. The unnormalised
                    # displacement keeps this constraint affine in original U.
                    coefficients = (
                        displacement[1][axis] - displacement[0][axis],
                        displacement[2][axis] - displacement[0][axis],
                        displacement[0][axis],
                    )
                    derivatives = (
                        displacement_dot[1][axis] - displacement_dot[0][axis],
                        displacement_dot[2][axis] - displacement_dot[0][axis],
                        displacement_dot[0][axis],
                    )
                    normal_departure = samples[0].origin_dot[axis] != face_dot
                    emitter_on_face = all(
                        spec.grid.origin_mm[k] <= local[k] < spec.grid.upper_mm[k]
                        for k in range(3)
                        if k != axis
                    )
                    if math.hypot(*coefficients[:2]) == 0.0:
                        if normal_departure and emitter_on_face:
                            raise TransportError(
                                "cone emitter relative normal motion starts on a material interface"
                            )
                        if coefficients[2] == 0.0 and derivatives != (0.0, 0.0, 0.0):
                            raise TransportError(
                                "the whole cone source leaves a coplanar material angular chart"
                            )
                        continue
                    boundary = AffineBoundary(f"material:{axis}:{index}", coefficients, derivatives)
                    visible = _cone_grazing_visible(source, boundary, spec, axis)
                    if normal_departure and (emitter_on_face or visible):
                        raise TransportError(
                            "cone emitter relative normal motion starts on a material interface"
                        )
                    if visible and derivatives != (0.0, 0.0, 0.0):
                        # Validate coincidence with original source support.
                        boundary.endpoints()
                        out.append((boundary, axis, index))
        return tuple(out)
    samples = tuple(
        _source_ray(source, xy, tangent.source)[0].in_object_frame(
            spec.object_pose, tangent.object.motion
        )
        for xy in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    )
    out: list[tuple[AffineBoundary, int, int]] = []
    for axis, count in enumerate(spec.grid.shape_xyz):
        if samples[0].direction[axis] != 0.0:
            continue
        for index in range(count + 1):
            face = spec.grid.origin_mm[axis] + index * spec.grid.spacing_mm[axis]
            face_dot = tangent.object.origin_dot[axis] + index * tangent.object.spacing_dot[axis]
            coefficients = (
                samples[1].origin_mm[axis] - samples[0].origin_mm[axis],
                samples[2].origin_mm[axis] - samples[0].origin_mm[axis],
                samples[0].origin_mm[axis] - face,
            )
            derivatives = (
                samples[1].origin_dot[axis] - samples[0].origin_dot[axis],
                samples[2].origin_dot[axis] - samples[0].origin_dot[axis],
                samples[0].origin_dot[axis] - face_dot,
            )
            if math.hypot(*coefficients[:2]) == 0.0:
                if coefficients[2] == 0.0:
                    if (axis, index) in equivalent_whole_faces:
                        continue
                    if samples[0].direction_dot[axis] != 0.0:
                        from .geometry_opaque_source import _whole_grazing_visible

                        if _whole_grazing_visible(source, spec, axis):
                            raise TransportError(
                                "a whole-source grazing material-face chart leaves its direction"
                            )
                    if derivatives != (0.0, 0.0, 0.0):
                        raise TransportError(
                            "the whole source lies on an active material interface"
                        )
                continue
            boundary = AffineBoundary(f"material:{axis}:{index}", coefficients, derivatives)
            if boundary.endpoints() is not None:
                if samples[0].direction_dot[axis] != 0.0:
                    from .geometry_opaque_source import OpaqueSourceBoundary, _grazing_visible

                    if _grazing_visible(
                        source,
                        OpaqueSourceBoundary(index, boundary, ((axis, int(index == count)),)),
                        spec,
                    ):
                        raise TransportError(
                            "a grazing material-face chart requires a directional boundary limit"
                        )
                out.append((boundary, axis, index))
    return tuple(out)


def _cone_grazing_visible(
    source: PointConeSource, boundary: AffineBoundary, spec: TransportSpec, selected_axis: int
) -> bool:
    """Positive source-line measure with a positive forward face interval.

    Writing a ray on the clipped angular line as o+t*(d0+s*d1), substitute
    w=t*s. Source-line support and the two tangential box intervals then form
    an exact two-dimensional affine feasibility problem in (t,w).
    """
    a, b, c = boundary.coefficients
    if a == 0.0 and -c / b in (0.0, 1.0):
        ends = ((0.0, -c / b), (1.0, -c / b))
    elif b == 0.0 and -c / a in (0.0, 1.0):
        ends = ((-c / a, 0.0), (-c / a, 1.0))
    else:
        ends = AffineBoundary(boundary.key, boundary.coefficients, (0, 0, 0)).endpoints()
    if ends is None:
        return False
    inverse = spec.object_pose.inverse()
    first = _source_ray(source, ends[0], SourceGeometryTangent())
    last = _source_ray(source, ends[1], SourceGeometryTangent())
    origin = inverse.point(first[0].origin_mm)
    direction = inverse.direction(first[1])
    following = inverse.direction(last[1])
    constraints = [
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(-1), Fraction(0)),
    ]
    for axis in range(3):
        if axis == selected_axis:
            continue
        a = Fraction(direction[axis])
        b = Fraction(following[axis]) - a
        lo, hi, o = map(
            Fraction, (spec.grid.origin_mm[axis], spec.grid.upper_mm[axis], origin[axis])
        )
        if a == b == 0:
            if not lo <= o < hi:
                return False
        else:
            constraints.extend(((a, b, o - lo), (-a, -b, hi - o)))
    witnessed = [False] * len(constraints)
    for first_row, last_row in combinations(constraints, 2):
        a, b, c = first_row
        d, e, f = last_row
        determinant = a * e - b * d
        if determinant == 0:
            continue
        t, w = (b * f - c * e) / determinant, (c * d - a * f) / determinant
        slacks = [a * t + b * w + c for a, b, c in constraints]
        if any(value < 0 for value in slacks):
            continue
        witnessed = [seen or value > 0 for seen, value in zip(witnessed, slacks, strict=True)]
        if all(witnessed):
            return True
    return False


def uncollided_geometry_boundaries(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    tangent: AcquisitionGeometryTangent | None = None,
) -> tuple[AffineBoundary, ...]:
    """Return each required boundary chart for the uncollided geometry integral.

    Detector pixel edges are present even if their current cotangents happen to
    coincide. Parallel-beam material side faces are sampled explicitly, whilst
    ordinary transverse material crossings contribute through interval jets.
    Boundary coordinates use their own caller-declared sampling design.
    """
    tangent = AcquisitionGeometryTangent() if tangent is None else tangent
    if isinstance(source, ParallelBeam) and any(value <= 0 for value in source.extent_xy_mm):
        raise TransportError("this source-area boundary chart requires two positive extents")
    # All source corners must have forward transverse detector intersections.
    for coordinate in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        detector_hit_tangent(_source_ray(source, coordinate, tangent.source)[0], spec.detector)
    boundaries = [record[0] for record in _material_boundaries(source, spec, tangent)]
    for axis, count in enumerate(spec.detector.shape):
        for index in range(count + 1):
            boundary = _detector_boundary(source, spec, tangent, axis, index)
            if boundary is not None:
                boundaries.append(boundary)
    lines: set[tuple[float, float, float]] = set()
    for boundary in boundaries:
        a, b, c = boundary.coefficients
        scale = math.hypot(a, b) * (-1.0 if (a < 0.0 or (a == 0.0 and b < 0.0)) else 1.0)
        identity = (a / scale, b / scale, c / scale)
        if identity in lines:
            raise TransportError("coincident material/detector boundaries need a joint chart")
        lines.add(identity)
    return tuple(boundaries)


def uncollided_geometry_jvp(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    *,
    extinction_mm_inverse: Sequence[float],
    cotangent: Sequence[float],
    tangent: AcquisitionGeometryTangent,
    source_coordinates: Sequence[tuple[float, float]],
    boundary_coordinates: Mapping[str, Sequence[float]],
) -> UncollidedGeometryEstimate:
    """Estimate the contracted uncollided mean and its boundary-complete JVP.

    Extinction is a finite nonnegative coefficient per x-fastest material cell,
    including its density and the sum of absorption and all scattering partials
    at the supplied monoenergetic source energy. Thus the uncollided term applies
    to elastic, Rayleigh, free/bound Compton and fluorescence transport. Source
    weights and the selected photon-count/keV score retain their existing units.

    Source coordinates are samples from the unit square. Each returned boundary
    key requires its own nonempty sequence of uniform (0,1) coordinates. IID and
    independently randomised low-discrepancy coordinates are both valid; this
    function does not invent a variance estimate for correlated designs. The
    caller's sampling provenance and samples remain unchanged. Fixed-grid face
    charts use exact side classification, with no offset epsilon.

    This host reference computes the uncollided term. A full transport geometry
    derivative additionally integrates interface collision sources and scattered
    detector/source support terms; it cannot be obtained by adding this result
    to the existing fixed-geometry density score.
    """
    rates, seeds = tuple(extinction_mm_inverse), tuple(cotangent)
    if len(rates) != math.prod(spec.grid.shape) or any(
        not math.isfinite(value) or value < 0.0 for value in rates
    ):
        raise TransportError("extinction requires one finite nonnegative coefficient per cell")
    if len(seeds) != spec.detector.pixels or not all(math.isfinite(value) for value in seeds):
        raise TransportError("cotangent requires one finite signed value per detector pixel")
    if not source_coordinates:
        raise TransportError("the source-coordinate sequence must be nonempty")
    boundaries = uncollided_geometry_boundaries(source, spec, tangent)
    expected = {boundary.key for boundary in boundaries}
    if set(boundary_coordinates) != expected:
        raise TransportError("boundary-coordinate keys must match all prepared boundary charts")
    values = [
        _ray_measure(source, spec, rates, seeds, tangent, coordinate)
        for coordinate in source_coordinates
    ]
    material_terms: list[float] = []
    detector_terms: list[float] = []
    counts: list[tuple[str, int]] = []
    for boundary in boundaries:
        coordinates = tuple(boundary_coordinates[boundary.key])
        if not coordinates:
            raise TransportError("each boundary requires a nonempty coordinate sequence")
        family, axis, index = boundary.key.split(":")
        terms: list[float] = []
        for coordinate in coordinates:
            sample = boundary.sample(coordinate)
            assert sample is not None
            point, weight = sample
            sides: list[float] = []
            for sign in (-1, 1):
                selector = (int(axis), int(index), sign)
                sides.append(
                    _ray_measure(
                        source,
                        spec,
                        rates,
                        seeds,
                        tangent,
                        point,
                        material_side=selector if family == "material" else None,
                        detector_side=selector if family == "detector" else None,
                    )[0]
                )
            terms.append(weight * (sides[1] - sides[0]))
        mean = math.fsum(terms) / len(terms)
        (material_terms if family == "material" else detector_terms).append(mean)
        counts.append((boundary.key, len(coordinates)))
    return UncollidedGeometryEstimate(
        math.fsum(value for value, _ in values) / len(values),
        math.fsum(derivative for _, derivative in values) / len(values),
        math.fsum(material_terms),
        math.fsum(detector_terms),
        len(values),
        tuple(counts),
    )


__all__ = [
    "AcquisitionGeometryTangent",
    "SourceGeometryTangent",
    "UncollidedGeometryEstimate",
    "uncollided_geometry_boundaries",
    "uncollided_geometry_jvp",
]
