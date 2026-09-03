"""Full-history transport geometry reference in the source-conditioned gauge.

The expectation uses the actual physical transport law independently of the
forward variance-reduction policy. Generator insertions replace moving-material
flight jets and source-fluence/support jets. All radiative orders, complete
two-sided futures, terminal detector connections and primary pixel jumps are
summed in one original photon observation before confidence reduction.
"""

from __future__ import annotations

# Geometry components share their validated source/connection primitives.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass, replace
from typing import cast

from .geometry_ballistic import ballistic_interface_crossings
from .geometry_histories import (
    GeometryPhysicalMedium,
    GeometryUniforms,
    complete_geometry_history,
    geometry_uniform,
)
from .geometry_interfaces import (
    InterfacePoint,
    MaterialInterface,
    interface_generator_observation,
    sample_interface_bridge,
    source_path_gauge,
)
from .geometry_opaque_source import (
    opaque_boundary_sides,
    opaque_material_hit,
    opaque_source_boundaries,
)
from .geometry_source_jumps import (
    admit_joint_source_boundary_chart,
    material_source_ray_boundaries,
    material_source_ray_jump_observation,
    sample_material_source_ray_jump,
)
from .geometry_startup_admission import admit_source_startup_chart
from .geometry_tangents import (
    AffineBoundary,
    RayTangent,
    RigidMotion,
    cell_segments_tangent,
    detector_hit_tangent,
)
from .geometry_terminal import terminal_geometry_observation
from .geometry_whole_source_admission import admit_whole_source_geometry
from .model import TransportError, TransportSpec
from .physical_geometry_events import continuing_radiative_collision, terminal_physical_kernels
from .source import ParallelBeam, PointConeSource
from .uncollided_geometry import (
    AcquisitionGeometryTangent,
    SourceGeometryTangent,
    _detector_boundary,
    _ray_measure,
    _source_ray,
)


@dataclass(frozen=True, slots=True)
class ComposedGeometryObservation:
    """One original photon, including correlated auxiliary futures and nulls."""

    original_history: int
    mean: float
    derivative: float
    terminal_geometry: float
    ballistic_generator: float
    collided_generator: float
    grazing_source: float
    primary_detector: float
    physical_prefixes: int
    continuation_histories: int
    random_identity: str


def geometry_channel_count(medium: GeometryPhysicalMedium) -> int:
    """Stable reserved slots: noncoherent, coherent, then material-local lines."""
    return 2 + (
        0
        if medium.physical is None
        else max((len(lines) for lines in medium.physical.fluorescence), default=0)
    )


def material_plane_count(spec: TransportSpec) -> int:
    return sum(size + 1 for size in spec.grid.shape_xyz)


def material_plane_at(index: int, spec: TransportSpec) -> tuple[int, int]:
    if type(index) is not int or not 0 <= index < material_plane_count(spec):
        raise TransportError("selected geometry plane must belong to its complete measure")
    for axis, size in enumerate(spec.grid.shape_xyz):
        if index <= size:
            return axis, index
        index -= size + 1
    raise AssertionError("validated material plane index")


def material_patch_count(spec: TransportSpec) -> int:
    sizes = spec.grid.shape_xyz
    return sum((sizes[i] + 1) * math.prod(sizes[j] for j in range(3) if j != i) for i in range(3))


def material_patch_at(index: int, spec: TransportSpec) -> MaterialInterface:
    """Decode a face rectangle without allocating a grid-sized interface list."""
    if type(index) is not int or not 0 <= index < material_patch_count(spec):
        raise TransportError("selected geometry patch must belong to its complete measure")
    sizes = spec.grid.shape_xyz
    nx, ny, _ = sizes
    for axis in range(3):
        other = tuple(i for i in range(3) if i != axis)
        tangential_count = sizes[other[0]] * sizes[other[1]]
        count = (sizes[axis] + 1) * tangential_count
        if index >= count:
            index -= count
            continue
        face, remainder = divmod(index, tangential_count)
        first, second = divmod(remainder, sizes[other[1]])
        indices = [0, 0, 0]
        indices[other[0]], indices[other[1]] = first, second
        indices[axis] = face - 1
        negative = ((indices[2] * ny + indices[1]) * nx + indices[0]) if face > 0 else None
        indices[axis] = face
        positive = (
            ((indices[2] * ny + indices[1]) * nx + indices[0]) if face < sizes[axis] else None
        )
        return MaterialInterface(axis, face, (first, second), negative, positive)
    raise AssertionError("validated material patch index")


def _auxiliary(design: GeometryUniforms, history: int, event: int, role: int, lane: int) -> float:
    return geometry_uniform(design, (history, 0, 0), event, 0x30000000 | role, lane)


def _transmission(
    ray: RayTangent,
    energy: float,
    spec: TransportSpec,
    medium: GeometryPhysicalMedium,
    distance: float,
) -> float:
    segments = cell_segments_tangent(ray, spec.grid, spec.object_pose, maximum_distance_mm=distance)
    if len(segments) > spec.max_crossings:
        raise TransportError("connection crossing guard invalidates the complete geometry history")
    rates = medium.extinction(energy, spec)
    return math.exp(-math.fsum(rates[segment.cell] * segment.length_mm for segment in segments))


def _generator(
    point: InterfacePoint,
    incoming: RayTangent,
    energy: float,
    spec: TransportSpec,
    medium: GeometryPhysicalMedium,
    design: GeometryUniforms,
    lineage: tuple[int, int, int],
    cotangent: tuple[float, ...],
    gauge: RigidMotion,
    importance: float,
) -> tuple[float, int]:
    # A stationary face has identically zero contribution, even for collimated
    # grazing illumination. Evaluate this before requesting any side future.
    if point.relative_velocity(gauge) == 0.0 or importance == 0.0:
        return 0.0, 0
    rates = medium.extinction(energy, spec)
    sides = (point.interface.negative_cell, point.interface.positive_cell)
    responses: list[float | None] = []
    count = 0
    for sign, cell in zip((-1, 1), sides, strict=True):
        if cell is None or rates[cell] == 0.0:
            responses.append(None)
            continue
        future = complete_geometry_history(
            incoming,
            energy,
            spec,
            medium,
            design,
            lineage=lineage,
            cotangent=cotangent,
            forced_collision_cell=cell,
            initial_material_side=(point.interface.axis, point.interface.face, sign),
        )
        responses.append(future.response)
        count += 1
    # The unchanged-direction value is single-valued at a transverse interface.
    # At a tangent interface it has the actual incident-side chart. Incoming
    # bridge directions are transverse except a measure-zero rejected chart.
    local_direction = spec.object_pose.inverse().direction(incoming.direction)
    if local_direction[point.interface.axis] == 0.0:
        raise TransportError("nonzero grazing generator needs a complete source-ray side chart")
    unchanged = complete_geometry_history(
        incoming,
        energy,
        spec,
        medium,
        design,
        lineage=lineage,
        cotangent=cotangent,
    )
    count += 1
    value = interface_generator_observation(
        point,
        negative_extinction_mm_inverse=0.0 if sides[0] is None else rates[sides[0]],
        positive_extinction_mm_inverse=0.0 if sides[1] is None else rates[sides[1]],
        negative_collision_response=responses[0],
        positive_collision_response=responses[1],
        unchanged_direction_response=unchanged.response,
        incident_importance=importance,
        source_gauge=gauge,
    )
    return value, count


def _primary_boundaries(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    tangent: AcquisitionGeometryTangent,
) -> tuple[AffineBoundary, ...]:
    boundaries: list[AffineBoundary] = []
    for axis, size in enumerate(spec.detector.shape):
        for face in range(size + 1):
            boundary = _detector_boundary(source, spec, tangent, axis, face)
            if boundary is not None:
                boundaries.append(boundary)
    return tuple(boundaries)


def _unattenuated_primary_score(
    source: ParallelBeam | PointConeSource,
    coordinate: tuple[float, float],
    energy: float,
    spec: TransportSpec,
    cotangent: tuple[float, ...],
    detector_side: tuple[int, int, int] | None = None,
) -> float:
    ray = _source_ray(source, coordinate, SourceGeometryTangent())[0]
    normal = spec.detector.normal
    numerator = sum(
        (a - b) * c
        for a, b, c in zip(spec.detector.lower_corner_mm, ray.origin_mm, normal, strict=True)
    )
    denominator = sum(a * b for a, b in zip(ray.direction, normal, strict=True))
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise TransportError("opaque primary intersection is not representable in binary64")
    if denominator <= 0.0 or numerator <= 0.0:
        return 0.0
    hit = detector_hit_tangent(ray, spec.detector)
    pixel = hit.pixel
    if detector_side is not None:
        axis, face, sign = detector_side
        indices = [math.floor(hit.column), math.floor(hit.row)]
        indices[axis] = face - int(sign < 0)
        column, row = indices
        width, height = spec.detector.shape
        pixel = row * width + column if 0 <= column < width and 0 <= row < height else -1
    result = (
        0.0
        if pixel < 0
        else source.weight * cotangent[pixel] * (energy if spec.scoring == "energy-kev" else 1.0)
    )
    if not math.isfinite(result):
        raise TransportError("complete opaque escape score is not representable in binary64")
    return result


def _opaque_observation(
    source: ParallelBeam | PointConeSource,
    coordinate: tuple[float, float],
    energy: float,
    spec: TransportSpec,
    design: GeometryUniforms,
    original_history: int,
    cotangent: tuple[float, ...],
    tangent: AcquisitionGeometryTangent,
) -> ComposedGeometryObservation:
    boundaries = opaque_source_boundaries(source, spec, tangent)
    detector_boundaries = _primary_boundaries(source, spec, tangent)
    admit_joint_source_boundary_chart(
        (*[row.boundary for row in boundaries], *detector_boundaries),
        family="opaque/detector",
    )
    ray = _source_ray(source, coordinate, SourceGeometryTangent())[0]
    mean = (
        0.0
        if opaque_material_hit(ray, spec)
        else _unattenuated_primary_score(source, coordinate, energy, spec, cotangent)
    )
    selected = min(17, int(_auxiliary(design, original_history, 0, 3, 1) * 18))
    shadow = 0.0
    for row in boundaries:
        if row.index != selected:
            continue
        sampled = row.boundary.sample(_auxiliary(design, original_history, 0, 3, 2))
        assert sampled is not None
        uv, coarea = sampled
        negative_hit, positive_hit = opaque_boundary_sides(source, uv, row, spec)
        response = _unattenuated_primary_score(source, uv, energy, spec, cotangent)
        shadow = 18 * coarea * response * (int(negative_hit) - int(positive_hit))
        break
    edge_count = sum(size + 1 for size in spec.detector.shape)
    selected_edge = min(
        edge_count - 1, int(_auxiliary(design, original_history, 0, 4, 0) * edge_count)
    )
    axis = 0 if selected_edge <= spec.detector.shape[0] else 1
    face = selected_edge if axis == 0 else selected_edge - spec.detector.shape[0] - 1
    detector_term = 0.0
    for boundary in detector_boundaries:
        if boundary.key != f"detector:{axis}:{face}":
            continue
        sampled = boundary.sample(_auxiliary(design, original_history, 0, 4, 1))
        assert sampled is not None
        uv, coarea = sampled
        if not opaque_material_hit(_source_ray(source, uv, SourceGeometryTangent())[0], spec):
            positive = _unattenuated_primary_score(
                source, uv, energy, spec, cotangent, (axis, face, 1)
            )
            negative = _unattenuated_primary_score(
                source, uv, energy, spec, cotangent, (axis, face, -1)
            )
            detector_term = edge_count * coarea * (positive - negative)
        break
    derivative = shadow + detector_term
    if not math.isfinite(derivative):
        raise TransportError("complete opaque geometry derivative is not representable in binary64")
    return ComposedGeometryObservation(
        original_history,
        mean,
        derivative,
        0.0,
        0.0,
        0.0,
        shadow,
        detector_term,
        0,
        0,
        design.identity,
    )


def composed_geometry_observation(
    source: ParallelBeam | PointConeSource,
    source_coordinate: tuple[float, float],
    energy_kev: float,
    spec: TransportSpec,
    medium: GeometryPhysicalMedium,
    design: GeometryUniforms,
    *,
    original_history: int,
    cotangent: tuple[float, ...],
    tangent: AcquisitionGeometryTangent,
) -> ComposedGeometryObservation:
    """Contract one full physical geometry JVP with its original-history seed.

    ``source_coordinate`` and ``energy_kev`` are the original emitted marks.
    The provider must implement the complete geometry-gauge-v1 reservation:
    lineage ``(history, branch, offset)`` maps local event f to offset+f, with
    branches 0 (prefix), 1 (ballistic), 2 (grazing source) and 3+j (prefix channel).
    Prefix-channel offset is e*max_events. Auxiliary roles use 0x30000000.
    All conditional continuations share paired coordinates, retain null draws
    and terminate physically; any resource/numerical failure aborts this result.

    The primary smooth pathwise derivative, old source-fluence derivatives and
    old material optical-flight derivatives are absent because the signed
    generators and full source-ray jumps supply their complete replacement.
    """
    medium.validate(spec)
    if type(original_history) is not int or original_history < 0:
        raise TransportError("geometry observation requires its original unsigned history ID")
    if len(cotangent) != spec.detector.pixels or any(not math.isfinite(v) for v in cotangent):
        raise TransportError("geometry cotangent requires one finite signed value per pixel")
    if medium.physical is not None and energy_kev <= medium.physical.cutoff_kev:
        return _opaque_observation(
            source,
            source_coordinate,
            energy_kev,
            spec,
            design,
            original_history,
            cotangent,
            tangent,
        )
    channels = geometry_channel_count(medium)
    if 3 + channels > 4096 or spec.max_events * spec.max_events > 0x7FFFFFFF:
        raise TransportError("complete geometry continuation coordinate reservation is too large")
    launch = _source_ray(source, source_coordinate, SourceGeometryTangent())[0]
    gauge = source_path_gauge(source, source_coordinate, tangent.source)
    # Check source support charts before any conditionally absent contribution.
    empty_medium = not any(
        (*medium.absorption_mm_inverse, *medium.scattering_mm_inverse)
    ) or not any(medium.voxel_density)
    admission = admit_whole_source_geometry(
        source, spec, medium, tangent, emitted_energy_kev=energy_kev
    )
    redundant_planes = admission.redundant_detector_planes
    source_boundaries = (
        ()
        if empty_medium
        else material_source_ray_boundaries(
            source,
            spec,
            tangent,
            equivalent_whole_faces=admission.source.equivalent_faces,
        )
    )
    if not empty_medium:
        admit_source_startup_chart(
            source, spec, medium, tangent, emitted_energies_kev=(energy_kev,)
        )
    detector_boundaries = _primary_boundaries(source, spec, tangent)
    admit_joint_source_boundary_chart(
        (*[record.boundary for record in source_boundaries], *detector_boundaries),
        family="material/detector",
    )
    reference = complete_geometry_history(
        launch,
        energy_kev,
        spec,
        medium,
        design,
        lineage=(original_history, 0, 0),
        cotangent=cotangent,
    )
    rates = medium.extinction(energy_kev, spec)
    primary_spec = replace(spec, energy_kev=energy_kev)
    primary = _ray_measure(
        source, primary_spec, rates, cotangent, AcquisitionGeometryTangent(), source_coordinate
    )[0]
    values: list[float] = [primary]
    terminal_terms: list[float] = []
    collided_terms: list[float] = []
    continuation_count = 0
    patch_count = material_patch_count(spec)
    analogue = replace(spec, estimator="analogue")
    for prefix in reference.prefixes:
        if not continuing_radiative_collision(
            prefix.energy_kev,
            medium.material_ids[prefix.cell],
            spec,
            medium.absorption_mm_inverse,
            medium.scattering_mm_inverse,
            medium.physical,
        ):
            continue
        event = prefix.event
        terminal = terminal_geometry_observation(
            prefix,
            spec,
            medium,
            tangent,
            gauge,
            cotangent,
            detector_coordinate=(
                _auxiliary(design, original_history, event, 0, 0),
                _auxiliary(design, original_history, event, 0, 1),
            ),
            detector_component=_auxiliary(design, original_history, event, 0, 2),
            forward_component=_auxiliary(design, original_history, event, 0, 3),
            bound_energy_uniform=_auxiliary(design, original_history, event, 1, 0),
            cutoff_azimuth=_auxiliary(design, original_history, event, 1, 1),
            redundant_planes=redundant_planes,
        )
        values.append(source.weight * terminal.value)
        terminal_terms.append(source.weight * terminal.derivative)
        index = min(
            patch_count - 1, int(_auxiliary(design, original_history, event, 1, 2) * patch_count)
        )
        patch = material_patch_at(index, spec)
        if (patch.axis, patch.face) in redundant_planes:
            continue
        bridge = sample_interface_bridge(
            patch,
            prefix.incoming.origin_mm,
            (
                _auxiliary(design, original_history, event, 2, 0),
                _auxiliary(design, original_history, event, 2, 1),
            ),
            spec,
            tangent.object,
        )
        if bridge.point.relative_velocity(gauge) == 0.0:
            continue
        mark, energy_weight = None, 1.0
        if spec.scattering_law == "bound-electron-compton":
            assert medium.physical is not None
            lower = medium.physical.cutoff_kev
            energy_weight = max(0.0, prefix.energy_kev - lower)
            mark = lower + _auxiliary(design, original_history, event, 2, 2) * energy_weight
            if energy_weight == 0.0:
                mark = 0.5 * prefix.energy_kev
        kernels = terminal_physical_kernels(
            RayTangent(prefix.incoming.origin_mm, prefix.incoming.direction),
            bridge.point.point_mm,
            prefix.energy_kev,
            medium.material_ids[prefix.cell],
            analogue,
            medium.absorption_mm_inverse,
            medium.scattering_mm_inverse,
            medium.physical,
            bound_outgoing_kev=mark,
        )
        for kernel in kernels:
            if medium.physical is not None and kernel.energy_kev <= medium.physical.cutoff_kev:
                continue
            channel = (
                1
                if kernel.channel == "rayleigh"
                else 2 + cast(int, kernel.line_index)
                if kernel.channel == "fluorescence"
                else 0
            )
            distance = math.dist(prefix.incoming.origin_mm, bridge.point.point_mm)
            transmission = _transmission(kernel.outgoing, kernel.energy_kev, spec, medium, distance)
            importance = (
                source.weight
                * patch_count
                * kernel.density
                * transmission
                * math.exp(bridge.log_geometric_importance)
            )
            if kernel.measure == "solid-angle-energy":
                importance *= energy_weight
            contribution, futures = _generator(
                bridge.point,
                RayTangent(bridge.point.point_mm, kernel.outgoing.direction),
                kernel.energy_kev,
                spec,
                medium,
                design,
                (original_history, 3 + channel, event * spec.max_events),
                cotangent,
                gauge,
                importance,
            )
            collided_terms.append(contribution)
            continuation_count += futures
    plane_count = material_plane_count(spec)
    selected_plane = material_plane_at(
        min(plane_count - 1, int(_auxiliary(design, original_history, 0, 3, 0) * plane_count)), spec
    )
    ballistic_terms: list[float] = []
    for crossing in (
        ()
        if empty_medium
        else ballistic_interface_crossings(
            launch,
            spec,
            tangent.object,
            source_gauge=gauge,
            redundant_planes=redundant_planes,
        )
    ):
        if (crossing.point.interface.axis, crossing.point.interface.face) != selected_plane:
            continue
        transmission = _transmission(launch, energy_kev, spec, medium, crossing.distance_mm)
        importance = source.weight * plane_count * transmission * crossing.inverse_normal_projection
        contribution, futures = _generator(
            crossing.point,
            crossing.incoming,
            energy_kev,
            spec,
            medium,
            design,
            (original_history, 1, 0),
            cotangent,
            gauge,
            importance,
        )
        ballistic_terms.append(contribution)
        continuation_count += futures
    grazing = 0.0
    selected_source_plane = material_plane_at(
        min(plane_count - 1, int(_auxiliary(design, original_history, 0, 3, 1) * plane_count)), spec
    )
    for record in source_boundaries:
        if (record.axis, record.face) != selected_source_plane:
            continue
        sample = sample_material_source_ray_jump(
            source, record, _auxiliary(design, original_history, 0, 3, 2), spec=spec
        )
        if sample.coarea_weight == 0.0:
            break
        negative = complete_geometry_history(
            sample.ray,
            energy_kev,
            spec,
            medium,
            design,
            lineage=(original_history, 2, 0),
            cotangent=cotangent,
            initial_material_side=sample.negative_side,
        )
        positive = complete_geometry_history(
            sample.ray,
            energy_kev,
            spec,
            medium,
            design,
            lineage=(original_history, 2, 0),
            cotangent=cotangent,
            initial_material_side=sample.positive_side,
        )
        grazing = plane_count * material_source_ray_jump_observation(
            sample,
            negative_complete_response=negative.response,
            positive_complete_response=positive.response,
        )
        continuation_count += 2
        break
    primary_detector = 0.0
    # The selector includes absent lines in the complete discrete edge measure.
    edge_count = sum(size + 1 for size in spec.detector.shape)
    edge_index = min(
        edge_count - 1, int(_auxiliary(design, original_history, 0, 4, 0) * edge_count)
    )
    axis = 0 if edge_index <= spec.detector.shape[0] else 1
    face = edge_index if axis == 0 else edge_index - spec.detector.shape[0] - 1
    for boundary in detector_boundaries:
        if boundary.key != f"detector:{axis}:{face}":
            continue
        sampled = boundary.sample(_auxiliary(design, original_history, 0, 4, 1))
        assert sampled is not None
        coordinate, coarea = sampled
        negative = _ray_measure(
            source,
            primary_spec,
            rates,
            cotangent,
            AcquisitionGeometryTangent(),
            coordinate,
            detector_side=(axis, face, -1),
        )[0]
        positive = _ray_measure(
            source,
            primary_spec,
            rates,
            cotangent,
            AcquisitionGeometryTangent(),
            coordinate,
            detector_side=(axis, face, 1),
        )[0]
        primary_detector = edge_count * coarea * (positive - negative)
        break
    terminal_sum, ballistic_sum, collided_sum = (
        math.fsum(terminal_terms),
        math.fsum(ballistic_terms),
        math.fsum(collided_terms),
    )
    mean = math.fsum(values)
    derivative = math.fsum((terminal_sum, ballistic_sum, collided_sum, grazing, primary_detector))
    if not math.isfinite(mean) or not math.isfinite(derivative):
        raise TransportError("complete geometry history is not representable in binary64")
    return ComposedGeometryObservation(
        original_history,
        mean,
        derivative,
        terminal_sum,
        ballistic_sum,
        collided_sum,
        grazing,
        primary_detector,
        len(reference.prefixes),
        continuation_count,
        design.identity,
    )


__all__ = [
    "ComposedGeometryObservation",
    "composed_geometry_observation",
    "geometry_channel_count",
    "material_patch_at",
    "material_patch_count",
    "material_plane_at",
    "material_plane_count",
]
