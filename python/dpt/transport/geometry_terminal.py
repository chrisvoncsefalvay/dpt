"""Complete mixed terminal detector measures in the source-conditioned gauge.

Prefixes and all intermediate physical marks remain fixed. The detector moves
relative to that prefix's source gauge; only its terminal connection, area,
angular law and final response are differentiated here. Material motion enters
through the separate collision-generator insertions, never a duplicate object
optical-flight jet. All terms from this prefix are one original-history sample.
"""

from __future__ import annotations

# Geometry modules share validated vector and terminal-law internals.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass, replace

from dpt.geometry import Vector3, cross, dot

from .compton_geometry_boundary import FreeComptonEnergyBoundary
from .geometry_detector_proposals import (
    prepare_detector_area_proposal,
    sample_detector_area_proposal,
)
from .geometry_forward_proposals import prepare_forward_area_mixture, sample_forward_area_mixture
from .geometry_histories import GeometryPhysicalMedium, PhysicalGeometryPrefix
from .geometry_path_measure import detector_vertex_tangent
from .geometry_tangents import RayTangent, RigidMotion, _sub, cell_segments_tangent
from .model import TransportError, TransportSpec
from .physical_geometry_events import (
    PhysicalTerminalKernel,
    continuing_radiative_collision,
    terminal_physical_kernels,
)
from .uncollided_geometry import AcquisitionGeometryTangent


@dataclass(frozen=True, slots=True)
class TerminalGeometryObservation:
    value: float
    derivative: float
    cutoff_boundary: float
    outgoing_channels: int
    reference_density_identity: str


def relative_rigid_motion(motion: RigidMotion, gauge: RigidMotion) -> RigidMotion:
    """Subtract two spatial velocity fields without assuming their pivots agree."""
    first = _sub(motion.translation, cross(motion.angular, motion.pivot_mm))
    second = _sub(gauge.translation, cross(gauge.angular, gauge.pivot_mm))
    return RigidMotion(_sub(first, second), _sub(motion.angular, gauge.angular))


def _connection(
    kernel: PhysicalTerminalKernel,
    point: Vector3,
    point_dot: Vector3,
    normal: Vector3,
    normal_dot: Vector3,
    area: float,
    area_dot: float,
    pixel: int,
    medium: GeometryPhysicalMedium,
    spec: TransportSpec,
    cotangent: tuple[float, ...],
    log_reference_density: float,
    *,
    energy_override: float | None = None,
) -> tuple[float, float]:
    energy = kernel.energy_kev if energy_override is None else energy_override
    energy_dot = kernel.energy_dot if energy_override is None else 0.0
    displacement = _sub(point, kernel.outgoing.origin_mm)
    distance = math.hypot(*displacement)
    distance_dot = dot(kernel.outgoing.direction, _sub(point_dot, kernel.outgoing.origin_dot))
    projection = dot(normal, kernel.outgoing.direction)
    projection_dot = dot(normal_dot, kernel.outgoing.direction) + dot(
        normal, kernel.outgoing.direction_dot
    )
    if distance <= 0.0 or projection <= 0.0:
        raise TransportError("terminal detector connection must be strictly separated and outgoing")
    rates, rate_dots = medium.extinction_jet(energy, energy_dot, spec)
    segments = cell_segments_tangent(
        kernel.outgoing,
        spec.grid,
        spec.object_pose,
        maximum_distance_mm=distance,
        maximum_distance_dot=distance_dot,
    )
    if len(segments) > spec.max_crossings:
        raise TransportError("terminal crossing guard invalidates the entire geometry observation")
    depth = math.fsum(rates[s.cell] * s.length_mm for s in segments)
    depth_dot = math.fsum(
        rates[s.cell] * s.length_dot + rate_dots[s.cell] * s.length_mm for s in segments
    )
    try:
        measure = math.exp(
            math.log(area)
            + math.log(projection)
            - 2 * math.log(distance)
            - log_reference_density
            - depth
        )
    except OverflowError as error:
        raise TransportError("terminal geometry weight overflows binary64") from error
    explicit = energy if spec.scoring == "energy-kev" else 1.0
    explicit_dot = energy_dot if spec.scoring == "energy-kev" else 0.0
    value = cotangent[pixel] * measure * kernel.density * explicit
    derivative = (
        cotangent[pixel]
        * measure
        * (
            explicit
            * (
                kernel.density_dot
                + kernel.density
                * (
                    area_dot / area
                    + projection_dot / projection
                    - 2 * distance_dot / distance
                    - depth_dot
                )
            )
            + kernel.density * explicit_dot
        )
    )
    if not math.isfinite(value) or not math.isfinite(derivative):
        raise TransportError("terminal geometry contribution is not representable in binary64")
    return value, derivative


def terminal_geometry_observation(
    prefix: PhysicalGeometryPrefix,
    spec: TransportSpec,
    medium: GeometryPhysicalMedium,
    tangent: AcquisitionGeometryTangent,
    source_gauge: RigidMotion,
    cotangent: tuple[float, ...],
    *,
    detector_component: float,
    detector_coordinate: tuple[float, float],
    bound_energy_uniform: float | None = None,
    cutoff_azimuth: float | None = None,
    forward_component: float | None = None,
    redundant_planes: frozenset[tuple[int, int]] = frozenset(),
) -> TerminalGeometryObservation:
    """Integrate every radiative terminal channel and its moving cutoff jump.

    The detector area uses the full uniform/logarithmic mixture. Bound Compton
    uses a detached uniform keV mark across the entire surviving energy range,
    evaluates the complete joint law and divides by its keV density once.
    A free-electron cutoff uses the separate full-azimuth signed coarea measure.
    No intermediate energy/support jet is inferred on this rigid path chart.
    """
    medium.validate(spec)
    if len(cotangent) != spec.detector.pixels or any(not math.isfinite(v) for v in cotangent):
        raise TransportError("terminal geometry cotangent must have one finite value per pixel")
    reference = replace(spec, estimator="analogue")
    if not continuing_radiative_collision(
        prefix.energy_kev,
        medium.material_ids[prefix.cell],
        reference,
        medium.absorption_mm_inverse,
        medium.scattering_mm_inverse,
        medium.physical,
    ):
        return TerminalGeometryObservation(0.0, 0.0, 0.0, 0, "absent-radiative-measure")
    proposal = prepare_detector_area_proposal(
        prefix.incoming.origin_mm, spec, redundant_planes=redundant_planes
    )
    if spec.scattering_law == "bound-electron-compton":
        if forward_component is None:
            raise TransportError(
                "bound terminal geometry requires its forward-disc mixture coordinate"
            )
        if (
            bound_energy_uniform is None
            or not math.isfinite(bound_energy_uniform)
            or not 0.0 < bound_energy_uniform < 1.0
        ):
            raise TransportError(
                "bound terminal measure requires its own explicit energy coordinate"
            )
        mixture = prepare_forward_area_mixture(
            RayTangent(prefix.incoming.origin_mm, prefix.incoming.direction),
            spec,
            redundant_planes=redundant_planes,
        )
        sampled = sample_forward_area_mixture(
            mixture,
            spec,
            component_draw=forward_component,
            regular_component_draw=detector_component,
            coordinate=detector_coordinate,
        )
        if sampled.coordinate is None:
            return TerminalGeometryObservation(0.0, 0.0, 0.0, 0, mixture.identity)
        proposal_identity = mixture.identity
    else:
        sampled = sample_detector_area_proposal(
            proposal, component_draw=detector_component, coordinate=detector_coordinate
        )
        proposal_identity = proposal.identity
    assert sampled.coordinate is not None and sampled.log_reference_density is not None
    motion = relative_rigid_motion(tangent.detector, source_gauge)
    endpoint = detector_vertex_tangent(
        sampled.coordinate, spec.detector, motion, tangent.detector_spacing_dot
    )
    incoming = RayTangent(prefix.incoming.origin_mm, prefix.incoming.direction)
    mark, energy_weight = None, 1.0
    if spec.scattering_law == "bound-electron-compton":
        if (
            bound_energy_uniform is None
            or not math.isfinite(bound_energy_uniform)
            or not 0.0 < bound_energy_uniform < 1.0
        ):
            raise TransportError(
                "bound terminal measure requires its own explicit open-interval energy coordinate"
            )
        assert medium.physical is not None
        lower = medium.physical.cutoff_kev
        energy_weight = max(0.0, prefix.energy_kev - lower)
        mark = (
            lower + bound_energy_uniform * energy_weight
            if energy_weight > 0.0
            else 0.5 * prefix.energy_kev
        )
    kernels = terminal_physical_kernels(
        incoming,
        endpoint.point_mm,
        prefix.energy_kev,
        medium.material_ids[prefix.cell],
        reference,
        medium.absorption_mm_inverse,
        medium.scattering_mm_inverse,
        medium.physical,
        endpoint_dot=endpoint.point_dot,
        bound_outgoing_kev=mark,
    )
    values: list[float] = []
    derivatives: list[float] = []
    for kernel in kernels:
        if medium.physical is not None and kernel.energy_kev <= medium.physical.cutoff_kev:
            continue
        value, derivative = _connection(
            kernel,
            endpoint.point_mm,
            endpoint.point_dot,
            endpoint.normal,
            endpoint.normal_dot,
            endpoint.area_jacobian,
            endpoint.area_jacobian_dot,
            endpoint.pixel,
            medium,
            spec,
            cotangent,
            sampled.log_reference_density,
        )
        factor = energy_weight if kernel.measure == "solid-angle-energy" else 1.0
        values.append(factor * value)
        derivatives.append(factor * derivative)
    cutoff = 0.0
    if (
        spec.scattering_law == "free-electron-compton"
        and medium.physical is not None
        and medium.physical.cutoff_kev > 0.0
    ):
        if cutoff_azimuth is None:
            raise TransportError(
                "free Compton terminal cutoff requires its full-azimuth boundary coordinate"
            )
        boundary = FreeComptonEnergyBoundary(
            incoming, prefix.energy_kev, medium.physical.cutoff_kev
        )
        # On this gauge chart both incident and cutoff energies are fixed.
        # At the angular endpoints eligibility is identically empty/full;
        # its geometry derivative is zero even for a moving detector.
        cut_sample = (
            boundary.sample(
                cutoff_azimuth, spec.detector, motion, spacing_dot=tangent.detector_spacing_dot
            )
            if -1.0 < boundary.cosine < 1.0
            else None
        )
        if cut_sample is not None and cut_sample.reference_area_weight != 0.0:
            unfiltered = replace(medium.physical, cutoff_kev=0.0)
            cut_kernels = terminal_physical_kernels(
                incoming,
                cut_sample.endpoint.point_mm,
                prefix.energy_kev,
                medium.material_ids[prefix.cell],
                reference,
                medium.absorption_mm_inverse,
                medium.scattering_mm_inverse,
                unfiltered,
            )
            for kernel in cut_kernels:
                if kernel.channel == "free-compton":
                    zero: Vector3 = (0.0, 0.0, 0.0)
                    value, _ = _connection(
                        kernel,
                        cut_sample.endpoint.point_mm,
                        zero,
                        cut_sample.endpoint.normal,
                        zero,
                        cut_sample.endpoint.area_jacobian,
                        0.0,
                        cut_sample.endpoint.pixel,
                        medium,
                        spec,
                        cotangent,
                        0.0,
                        energy_override=medium.physical.cutoff_kev,
                    )
                    cutoff += cut_sample.reference_area_weight * value
    derivative = math.fsum((*derivatives, cutoff))
    value = math.fsum(values)
    if not math.isfinite(value) or not math.isfinite(derivative):
        raise TransportError("combined terminal geometry observation overflows binary64")
    return TerminalGeometryObservation(value, derivative, cutoff, len(kernels), proposal_identity)


__all__ = ["TerminalGeometryObservation", "relative_rigid_motion", "terminal_geometry_observation"]
