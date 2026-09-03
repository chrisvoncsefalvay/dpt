"""Original-history native assembly of the full physical geometry estimator.

Every auxiliary side continuation belongs to the original emitted history.
Material generator insertions replace moving-interface optical jets, while
terminal detector jets are evaluated relative to that history's source gauge.
"""

# Warp annotations are runtime DSL records.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false
# pyright: reportPrivateUsage=false
# Mutable Warp loop variables require explicit constructors.
# ruff: noqa: RUF046, UP018
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import warp as wp

from ._constants import NUMERICAL_FAILURE
from .compton_geometry_boundary_kernels import (
    NativeBoundaryDetector,
    NativeBoundaryEndpoint,
    NativeBoundaryRay,
    NativeFreeComptonBoundary,
    _boundary_endpoint,
    sample_free_compton_energy_boundary,
)
from .geometry_ballistic_kernels import (
    material_patch_count,
    material_plane_count,
    select_material_patch,
    select_material_plane,
    selected_plane_crossing,
)
from .geometry_connection_kernels import connection_optical_depth, terminal_connection_observation
from .geometry_detector_proposal_kernels import (
    DetectorProposalPlane,
    sample_detector_area_proposal_from_vertex,
)
from .geometry_forward_proposal_kernels import ForwardProposalGeometry, sample_forward_area_mixture
from .geometry_interface_kernels import (
    InterfaceGauge,
    InterfaceGrid,
    InterfacePatch,
    NativeInterfacePoint,
    _hypot,
    interface_generator_observation,
    interface_relative_velocity,
    sample_interface_bridge,
    source_path_gauge,
)
from .geometry_opaque_kernels import (
    NativeOpaqueBoundary,
    opaque_boundary_sides,
    opaque_material_hit,
)
from .geometry_reference_kernels import (
    NativeGeometryRay,
    _cell_factor,
    _makereference_response,
    prefix_step,
    reference_response,
)
from .geometry_sampling_kernels import GeometryRandomDesign, geometry_auxiliary_random4
from .geometry_source_boundary_kernels import (
    NativeAffineSourceBoundary,
    NativeSourceGeometry,
    sample_source_boundary,
    source_ray_tangent,
)
from .kernels import Parameters, coefficients_at, energy_coordinate
from .physical_geometry_kernels import (
    NativePhysicalTerminal,
    _kn_normalisation,
    _kn_shape_slope,
    _makeconditional_physical_event_jet,
    _maketerminal_physical_kernel_jet,
    _physical_rates,
    conditional_physical_event_jet,
    terminal_physical_kernel_jet,
)
from .physical_kernels import energy_table_value, spectrum_index
from .sampling_kernels import transport_random4

_GEOMETRY_OPTIONS = {
    "fast_math": False,
    "fuse_fp": False,
    "enable_backward": False,
    "max_unroll": 1,
}
wp.set_module_options(_GEOMETRY_OPTIONS)
FAILURE = wp.constant(NUMERICAL_FAILURE)
MASS = wp.constant(wp.float64(510.99895069))


@wp.struct
class NativeGeometryColumn:
    source: NativeSourceGeometry
    grid: InterfaceGrid
    detector: NativeBoundaryDetector
    source_boundary_offset: int
    detector_boundary_offset: int
    opaque_boundary_offset: int
    empty_medium: int
    redundant_planes: wp.array(dtype=wp.int32)


@wp.func
def _redundant_plane(column: NativeGeometryColumn, axis: int, face: int) -> int:
    count = material_plane_count(column.grid)
    if column.redundant_planes.shape[0] == 0:
        return 0
    if column.redundant_planes.shape[0] != count or axis < 0 or axis > 2:
        return -1
    if face < 0 or face > column.grid.shape[axis]:
        return -1
    index = int(face)
    for preceding in range(axis):
        index += column.grid.shape[preceding] + 1
    flag = column.redundant_planes[index]
    if flag != 0 and flag != 1:
        return -1
    return int(flag)


@wp.struct
class NativeGeometryObservation:
    value: wp.float64
    derivative: wp.float64
    prefixes: int
    futures: int
    status: int


@wp.struct
class NativeGeometrySideRay:
    ray: NativeGeometryRay
    status: int


@wp.func
def _reference_ray(ray: NativeBoundaryRay) -> NativeGeometryRay:
    result = NativeGeometryRay()
    result.position = ray.origin_mm
    result.direction = ray.direction
    return result


@wp.func
def _source_side_ray(
    source: NativeSourceGeometry, coordinate: wp.vec2d, axis: int, face: int, p: Parameters
) -> NativeGeometrySideRay:
    result = NativeGeometrySideRay()
    result.status = FAILURE
    launched = source_ray_tangent(source, coordinate)
    if launched.status != 0:
        return result
    result.ray = _reference_ray(launched.ray)
    if source.kind == 1:
        if axis < 0 or axis >= 3 or face < 0 or face > p.shape[axis]:
            return result
        local = p.inverse_rotation * result.ray.direction
        if wp.abs(local[axis]) > wp.float64(512.0 * 2.220446049250313e-16):
            return result
        origin = p.inverse_rotation * result.ray.position + p.inverse_translation
        location = p.origin[axis] + wp.float64(face) * p.spacing[axis]
        resolution = wp.float64(512.0 * 2.220446049250313e-16) * wp.max(
            wp.float64(1.0), wp.max(wp.abs(origin[axis]), wp.abs(location))
        )
        if wp.abs(origin[axis] - location) > resolution:
            return result
        local[axis] = wp.float64(0.0)
        length = _hypot(local[0], _hypot(local[1], local[2]))
        if length <= wp.float64(0.0) or not wp.isfinite(length):
            return result
        direction = wp.transpose(p.inverse_rotation) * (local / length)
        result.ray.direction = direction
    result.status = 0
    return result


@wp.func
def _relative_detector(
    detector: NativeBoundaryDetector, gauge: InterfaceGauge
) -> NativeBoundaryDetector:
    result = NativeBoundaryDetector()
    result.corner = detector.corner
    result.u = detector.u
    result.v = detector.v
    result.normal = detector.normal
    result.shape = detector.shape
    result.spacing = detector.spacing
    result.motion_translation = (
        detector.motion_translation
        - wp.cross(detector.motion_angular, detector.motion_pivot)
        - gauge.translation
        + wp.cross(gauge.angular, gauge.pivot)
    )
    result.motion_angular = detector.motion_angular - gauge.angular
    result.motion_pivot = wp.vec3d(wp.float64(0.0))
    result.corner_correction = detector.corner_correction
    result.spacing_dot = detector.spacing_dot
    return result


@wp.func
def _ballistic_active(
    ray: NativeBoundaryRay, axis: int, face: int, grid: InterfaceGrid, gauge: InterfaceGauge
) -> int:
    inverse = wp.transpose(grid.rotation)
    origin = inverse * (ray.origin_mm - grid.translation)
    direction = inverse * ray.direction
    if direction[axis] == wp.float64(0.0):
        return 0
    distance = (
        grid.origin[axis] + wp.float64(face) * grid.spacing[axis] - origin[axis]
    ) / direction[axis]
    if not wp.isfinite(distance):
        return -1
    if distance <= wp.float64(0.0):
        return 0
    point = ray.origin_mm + distance * ray.direction
    local_velocity = wp.vec3d(wp.float64(0.0))
    local_velocity[axis] = grid.origin_dot[axis] + wp.float64(face) * grid.spacing_dot[axis]
    velocity = (
        grid.motion_translation
        + wp.cross(grid.motion_angular, point - grid.motion_pivot)
        + grid.rotation * local_velocity
    )
    gauge_velocity = gauge.translation + wp.cross(gauge.angular, point - gauge.pivot)
    local_normal = wp.vec3d(wp.float64(0.0))
    local_normal[axis] = wp.float64(1.0)
    normal = grid.rotation * local_normal
    scalar = wp.dot(velocity - gauge_velocity, normal)
    if not wp.isfinite(scalar):
        return -1
    if scalar == wp.float64(0.0):
        return 0
    return 1


@wp.func
def _continuing_radiative(
    energy: wp.float64,
    material: int,
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    p: Parameters,
) -> int:
    rates = _physical_rates(energy, material, energies, absorption, scattering, p)
    if rates.status != 0:
        return -1
    cutoff = wp.float64(0.0)
    if p.physics.enabled != 0:
        cutoff = p.physics.cutoff
    if energy <= cutoff:
        return 0
    if rates.rayleigh > wp.float64(0.0):
        return 1
    if rates.scattering > rates.rayleigh:
        if p.compton != 2:
            return 1
        for shell in range(
            p.physics.compton_shell_offsets[material], p.physics.compton_shell_offsets[material + 1]
        ):
            if (
                p.physics.compton_occupancy[shell] > wp.float64(0.0)
                and energy - p.physics.compton_binding[shell] > cutoff
            ):
                return 1
    if rates.absorption > wp.float64(0.0) and p.physics.enabled != 0:
        for line in range(
            p.physics.fluorescence_offsets[material], p.physics.fluorescence_offsets[material + 1]
        ):
            if (
                energy >= p.physics.fluorescence_threshold[line]
                and p.physics.fluorescence_energy[line] > cutoff
            ):
                probability = energy_table_value(
                    p.physics.fluorescence_probability, line, rates.coordinate, p.energy_bins
                )
                if not wp.isfinite(probability) or probability < wp.float64(0.0):
                    return -1
                if probability > wp.float64(0.0):
                    return 1
    return 0


@wp.func
def _primary(
    ray: NativeGeometryRay,
    energy: wp.float64,
    side_axis: int,
    side_face: int,
    side_sign: int,
    attenuate: int,
    seed_image: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    p: Parameters,
) -> NativeGeometryObservation:
    result = NativeGeometryObservation()
    result.status = FAILURE
    origin = p.inverse_rotation * ray.position + p.inverse_translation
    direction = p.inverse_rotation * ray.direction
    denominator = wp.dot(direction, p.detector_normal)
    if not wp.isfinite(denominator):
        return result
    if denominator <= wp.float64(0.0):
        result.status = 0
        return result
    distance = wp.dot(p.detector_corner - origin, p.detector_normal) / denominator
    if not wp.isfinite(distance):
        return result
    if distance <= wp.float64(0.0):
        result.status = 0
        return result
    displacement = origin + distance * direction - p.detector_corner
    column = wp.dot(displacement, p.detector_u) / p.detector_spacing[0]
    row = wp.dot(displacement, p.detector_v) / p.detector_spacing[1]
    if not wp.isfinite(column) or not wp.isfinite(row):
        return result
    pixel = int(-1)
    ix = int(-1)
    iy = int(-1)
    if column >= wp.float64(0.0) and column < wp.float64(p.detector_shape[0]):
        ix = int(wp.floor(column))
    if row >= wp.float64(0.0) and row < wp.float64(p.detector_shape[1]):
        iy = int(wp.floor(row))
    if side_axis == 0:
        ix = side_face - int(side_sign < 0)
    elif side_axis == 1:
        iy = side_face - int(side_sign < 0)
    if ix >= 0 and ix < p.detector_shape[0] and iy >= 0 and iy < p.detector_shape[1]:
        pixel = iy * p.detector_shape[0] + ix
    if pixel < 0:
        result.status = 0
        return result
    depth = wp.float64(0.0)
    if attenuate != 0:
        zero = wp.vec3d(wp.float64(0.0))
        optical = connection_optical_depth(
            ray.position,
            ray.direction,
            zero,
            zero,
            distance,
            wp.float64(0.0),
            energy,
            wp.float64(0.0),
            material_ids,
            voxel_density,
            energies,
            absorption,
            scattering,
            density,
            p,
        )
        if optical.status != 0:
            return result
        depth = optical.depth
    score = wp.float64(1.0)
    if p.energy_score != 0:
        score = energy
    result.value = seed_image[pixel] * score * wp.exp(-depth)
    if wp.isfinite(result.value):
        result.status = 0
    return result


def _make_generator(label: str, response: Any):
    """Bind a prepared model without duplicating the mathematical body."""
    reference_response = response

    @wp.func(name="_generator" if label == "generic" else f"_generator_{label}", module="unique")
    def _generator(
        point: NativeInterfacePoint,
        patch: InterfacePatch,
        gauge: InterfaceGauge,
        ray: NativeGeometryRay,
        energy: wp.float64,
        importance: wp.float64,
        branch: int,
        event_offset: int,
        coordinate_limit: int,
        seed: wp.uint64,
        history: wp.uint64,
        seed_image: wp.array(dtype=wp.float64),
        material_ids: wp.array(dtype=wp.int32),
        voxel_density: wp.array(dtype=wp.float64),
        energies: wp.array(dtype=wp.float64),
        absorption: wp.array(dtype=wp.float64),
        scattering: wp.array(dtype=wp.float64),
        density: wp.array(dtype=wp.float64),
        p: Parameters,
    ) -> NativeGeometryObservation:
        result = NativeGeometryObservation()
        result.status = FAILURE
        if point.status != 0 or gauge.status != 0 or not wp.isfinite(importance):
            return result
        if interface_relative_velocity(point, gauge) == wp.float64(0.0) or importance == wp.float64(
            0.0
        ):
            result.status = 0
            return result
        local_direction = p.inverse_rotation * ray.direction
        if local_direction[patch.axis] == wp.float64(0.0):
            return result
        rates = wp.vec2d(wp.float64(0.0))
        responses = wp.vec2d(wp.float64(0.0))
        present = wp.vec2i(0)
        coordinate = energy_coordinate(energy, energies, p)
        for side in range(2):
            cell = patch.negative_cell
            if side == 1:
                cell = patch.positive_cell
            if cell >= 0:
                material = material_ids[cell]
                factor = _cell_factor(cell, material, voxel_density, density, p)
                partials = coefficients_at(material, coordinate, absorption, scattering, p)
                rates[side] = factor * (partials[0] + partials[1])
                if (
                    factor < wp.float64(0.0)
                    or not wp.isfinite(rates[side])
                    or rates[side] < wp.float64(0.0)
                ):
                    return result
                if rates[side] > wp.float64(0.0):
                    future = reference_response(
                        ray,
                        energy,
                        cell,
                        patch.axis,
                        patch.face,
                        2 * side - 1,
                        material_ids,
                        voxel_density,
                        energies,
                        absorption,
                        scattering,
                        density,
                        p,
                        seed,
                        history,
                        branch,
                        event_offset,
                        coordinate_limit,
                        seed_image,
                    )
                    if future.status != 0:
                        return result
                    responses[side] = future.value
                    present[side] = 1
                    result.futures += 1
        unchanged = reference_response(
            ray,
            energy,
            -1,
            -1,
            0,
            0,
            material_ids,
            voxel_density,
            energies,
            absorption,
            scattering,
            density,
            p,
            seed,
            history,
            branch,
            event_offset,
            coordinate_limit,
            seed_image,
        )
        if unchanged.status != 0:
            return result
        result.futures += 1
        contracted = interface_generator_observation(
            point,
            gauge,
            rates[0],
            rates[1],
            responses[0],
            responses[1],
            present[0],
            present[1],
            unchanged.value,
            importance,
        )
        if contracted.status != 0:
            return result
        result.derivative = contracted.value
        result.status = 0
        return result

    return _generator


_generator = _make_generator("generic", reference_response)


def _make_terminal(model: int, label: str, terminal: Any):
    """Bind a prepared model without duplicating the mathematical body."""
    terminal_physical_kernel_jet = terminal

    @wp.func(name="_terminal" if model < 0 else f"_terminal_{label}", module="unique")
    def _terminal(
        ray: NativeGeometryRay,
        energy: wp.float64,
        cell: int,
        detector: NativeBoundaryDetector,
        planes: wp.array(dtype=DetectorProposalPlane),
        proposal: ForwardProposalGeometry,
        auxiliary: wp.vec4d,
        energy_and_cutoff: wp.vec4d,
        seed_image: wp.array(dtype=wp.float64),
        material_ids: wp.array(dtype=wp.int32),
        voxel_density: wp.array(dtype=wp.float64),
        energies: wp.array(dtype=wp.float64),
        absorption: wp.array(dtype=wp.float64),
        scattering: wp.array(dtype=wp.float64),
        density: wp.array(dtype=wp.float64),
        p: Parameters,
    ) -> NativeGeometryObservation:
        result = NativeGeometryObservation()
        result.status = FAILURE
        coordinate = wp.vec2d(auxiliary[0], auxiliary[1])
        log_density = wp.float64(0.0)
        if (wp.static(model == 2)) if wp.static(model >= 0) else (p.compton == 2):
            forward_draw = sample_forward_area_mixture(
                planes,
                0,
                planes.shape[0],
                proposal,
                ray.position,
                ray.direction,
                auxiliary[3],
                auxiliary[2],
                coordinate,
            )
            if forward_draw.status == -1:
                result.status = 0
                return result
            if forward_draw.status != 0:
                return result
            coordinate = forward_draw.coordinate
            log_density = forward_draw.log_reference_density
        else:
            regular_draw = sample_detector_area_proposal_from_vertex(
                planes, 0, planes.shape[0], proposal.regular, ray.position, auxiliary[2], coordinate
            )
            if regular_draw.status != 0:
                return result
            coordinate = regular_draw.coordinate
            log_density = regular_draw.log_reference_density
        endpoint = _boundary_endpoint(coordinate, detector)
        if endpoint.pixel < 0:
            return result
        material = material_ids[cell]
        channels = int(2)
        if p.physics.enabled != 0:
            channels += (
                p.physics.fluorescence_offsets[material + 1]
                - p.physics.fluorescence_offsets[material]
            )
        mark = wp.float64(0.0)
        energy_weight = wp.float64(1.0)
        if (wp.static(model == 2)) if wp.static(model >= 0) else (p.compton == 2):
            energy_weight = wp.max(wp.float64(0.0), energy - p.physics.cutoff)
            mark = p.physics.cutoff + energy_and_cutoff[0] * energy_weight
            if energy_weight == wp.float64(0.0):
                mark = energy * wp.float64(0.5)
        zero = wp.vec3d(wp.float64(0.0))
        for channel in range(channels):
            kind = wp.min(channel, 2)
            kernel = terminal_physical_kernel_jet(
                ray.position,
                ray.direction,
                zero,
                zero,
                endpoint.point_mm,
                endpoint.point_dot,
                energy,
                material,
                kind,
                channel - 2,
                mark,
                energies,
                absorption,
                scattering,
                p,
            )
            if kernel.status != 0:
                return result
            if kernel.present != 0 and (p.physics.enabled == 0 or kernel.energy > p.physics.cutoff):
                observation = terminal_connection_observation(
                    kernel,
                    endpoint,
                    log_density,
                    seed_image,
                    material_ids,
                    voxel_density,
                    energies,
                    absorption,
                    scattering,
                    density,
                    p,
                )
                if observation.status != 0:
                    return result
                factor = wp.float64(1.0)
                if kernel.measure == 1:
                    factor = energy_weight
                result.value += factor * observation.value
                result.derivative += factor * observation.derivative
        if (
            (
                (p.physics.enabled != 0 and p.physics.cutoff > wp.float64(0.0))
                if wp.static(model == 1)
                else wp.static(False)
            )
            if wp.static(model >= 0)
            else (p.compton == 1 and p.physics.enabled != 0 and p.physics.cutoff > wp.float64(0.0))
        ):
            cosine = (
                wp.float64(1.0) - MASS * ((energy - p.physics.cutoff) / energy) / p.physics.cutoff
            )
            if cosine > wp.float64(-1.0) and cosine < wp.float64(1.0):
                boundary = NativeFreeComptonBoundary()
                boundary.incoming.origin_mm = ray.position
                boundary.incoming.direction = ray.direction
                boundary.incident_energy = energy
                boundary.threshold_energy = p.physics.cutoff
                cut = sample_free_compton_energy_boundary(boundary, energy_and_cutoff[1], detector)
                if cut.status == FAILURE:
                    return result
                if cut.status == 0 and cut.reference_area_weight != wp.float64(0.0):
                    # Evaluate the limiting physical density independently of the
                    # reached-photon cutoff guard, at the exact boundary energy.
                    rates = _physical_rates(energy, material, energies, absorption, scattering, p)
                    law = _kn_shape_slope(energy, cosine)
                    normaliser = _kn_normalisation(energy)
                    if rates.status != 0 or normaliser <= wp.float64(0.0):
                        return result
                    kernel = NativePhysicalTerminal()
                    kernel.origin = ray.position
                    kernel.direction = cut.outgoing.direction
                    kernel.energy = p.physics.cutoff
                    kernel.density = (
                        ((rates.scattering - rates.rayleigh) / rates.total) * law[0] / normaliser
                    )
                    kernel.present = int(kernel.density > wp.float64(0.0))
                    endpoint = NativeBoundaryEndpoint()
                    endpoint.point_mm = cut.endpoint.point_mm
                    endpoint.normal = cut.endpoint.normal
                    endpoint.pixel = cut.endpoint.pixel
                    endpoint.area_jacobian = cut.endpoint.area_jacobian
                    value = terminal_connection_observation(
                        kernel,
                        endpoint,
                        wp.float64(0.0),
                        seed_image,
                        material_ids,
                        voxel_density,
                        energies,
                        absorption,
                        scattering,
                        density,
                        p,
                    )
                    if value.status != 0:
                        return result
                    result.derivative += cut.reference_area_weight * value.value
        if wp.isfinite(result.value) and wp.isfinite(result.derivative):
            result.status = 0
        return result

    return _terminal


_terminal = _make_terminal(-1, "generic", terminal_physical_kernel_jet)


@wp.func
def _opaque_history(
    column: NativeGeometryColumn,
    launch: NativeBoundaryRay,
    energy: wp.float64,
    detector_boundaries: wp.array(dtype=NativeAffineSourceBoundary),
    opaque_boundaries: wp.array(dtype=NativeOpaqueBoundary),
    seed_image: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    p: Parameters,
    random: GeometryRandomDesign,
    seed: wp.uint64,
    history: wp.uint64,
) -> NativeGeometryObservation:
    """Differentiate the canonical initial-cutoff box shadow in original U."""
    result = NativeGeometryObservation()
    result.status = FAILURE
    hit = opaque_material_hit(launch, column.grid, 0, 0, 0, 0, 0, 0)
    if hit.status != 0:
        return result
    if hit.hit == 0:
        score = _primary(
            _reference_ray(launch),
            energy,
            -1,
            0,
            0,
            0,
            seed_image,
            material_ids,
            voxel_density,
            energies,
            absorption,
            scattering,
            density,
            p,
        )
        if score.status != 0:
            return result
        result.value = column.source.weight * score.value
    draw = geometry_auxiliary_random4(seed, history, wp.uint32(0), wp.uint32(0x30000003), random)
    selected = wp.min(17, int(draw[1] * wp.float64(18.0)))
    row = opaque_boundaries[column.opaque_boundary_offset + selected]
    sampled = sample_source_boundary(row.boundary, draw[2])
    if sampled.status == FAILURE:
        return result
    if sampled.status == 0 and sampled.coarea_weight != wp.float64(0.0):
        sides = opaque_boundary_sides(column.source, sampled.coordinate, row, column.grid)
        boundary_launch = source_ray_tangent(column.source, sampled.coordinate)
        if sides.status != 0 or boundary_launch.status != 0:
            return result
        response = _primary(
            _reference_ray(boundary_launch.ray),
            energy,
            -1,
            0,
            0,
            0,
            seed_image,
            material_ids,
            voxel_density,
            energies,
            absorption,
            scattering,
            density,
            p,
        )
        if response.status != 0:
            return result
        result.derivative += (
            wp.float64(18.0)
            * sampled.coarea_weight
            * column.source.weight
            * response.value
            * wp.float64(sides.negative_hit - sides.positive_hit)
        )
    edge_count = p.detector_shape[0] + p.detector_shape[1] + 2
    detector_draw = geometry_auxiliary_random4(
        seed, history, wp.uint32(0), wp.uint32(0x30000004), random
    )
    edge = wp.min(edge_count - 1, int(detector_draw[0] * wp.float64(edge_count)))
    pixel_sample = sample_source_boundary(
        detector_boundaries[column.detector_boundary_offset + edge], detector_draw[1]
    )
    if pixel_sample.status == FAILURE:
        return result
    if pixel_sample.status == 0 and pixel_sample.coarea_weight != wp.float64(0.0):
        pixel_launch = source_ray_tangent(column.source, pixel_sample.coordinate)
        if pixel_launch.status != 0:
            return result
        pixel_hit = opaque_material_hit(pixel_launch.ray, column.grid, 0, 0, 0, 0, 0, 0)
        if pixel_hit.status != 0:
            return result
        if pixel_hit.hit == 0:
            pixel_ray = _reference_ray(pixel_launch.ray)
            negative = _primary(
                pixel_ray,
                energy,
                pixel_sample.axis,
                pixel_sample.face,
                -1,
                0,
                seed_image,
                material_ids,
                voxel_density,
                energies,
                absorption,
                scattering,
                density,
                p,
            )
            positive = _primary(
                pixel_ray,
                energy,
                pixel_sample.axis,
                pixel_sample.face,
                1,
                0,
                seed_image,
                material_ids,
                voxel_density,
                energies,
                absorption,
                scattering,
                density,
                p,
            )
            if negative.status != 0 or positive.status != 0:
                return result
            result.derivative += (
                column.source.weight
                * wp.float64(edge_count)
                * pixel_sample.coarea_weight
                * (positive.value - negative.value)
            )
    if not wp.isfinite(result.value) or not wp.isfinite(result.derivative):
        return result
    result.status = 0
    return result


def _makegeometry_history(
    model: int,
    packed: bool,
    label: str,
    collision: Any,
    terminal: Any,
    response: Any,
    generator: Any,
    terminal_observation: Any,
):
    """Bind a prepared model without duplicating the mathematical body."""
    conditional_physical_event_jet = collision
    terminal_physical_kernel_jet = terminal
    reference_response = response
    _generator = generator
    _terminal = terminal_observation

    @wp.func(name="geometry_history" if model < 0 else f"geometry_history_{label}", module="unique")
    def geometry_history(
        column: NativeGeometryColumn,
        planes: wp.array(dtype=DetectorProposalPlane),
        proposal: ForwardProposalGeometry,
        source_boundaries: wp.array(dtype=NativeAffineSourceBoundary),
        detector_boundaries: wp.array(dtype=NativeAffineSourceBoundary),
        opaque_boundaries: wp.array(dtype=NativeOpaqueBoundary),
        seed_image: wp.array(dtype=wp.float64),
        material_ids: wp.array(dtype=wp.int32),
        voxel_density: wp.array(dtype=wp.float64),
        energies: wp.array(dtype=wp.float64),
        absorption: wp.array(dtype=wp.float64),
        scattering: wp.array(dtype=wp.float64),
        density: wp.array(dtype=wp.float64),
        p: Parameters,
        random: GeometryRandomDesign,
        seed: wp.uint64,
        history: wp.uint64,
    ) -> NativeGeometryObservation:
        result = NativeGeometryObservation()
        result.status = FAILURE
        if wp.static(model >= 0):
            if p.compton != wp.static(model) or p.compton_packed != wp.static(int(packed)):
                return result
        source_draw = transport_random4(
            seed, history, wp.uint32(0x80000000), wp.uint32(0), p.sampling
        )
        coordinate = wp.vec2d(source_draw[0], source_draw[1])
        launch = source_ray_tangent(column.source, coordinate)
        if launch.status != 0:
            return result
        gauge = source_path_gauge(
            launch.ray.origin_mm,
            launch.ray.direction,
            launch.ray.origin_dot,
            launch.ray.direction_dot,
        )
        if gauge.status != 0:
            return result
        ray = _reference_ray(launch.ray)
        energy = p.source_energy
        if p.physics.enabled != 0 and p.physics.spectrum_bins > 0:
            spectral = transport_random4(
                seed, history, wp.uint32(0x80000002), wp.uint32(0), p.sampling
            )
            energy = p.physics.spectrum_energies[spectrum_index(spectral[0], p.physics)]
        if p.physics.enabled != 0 and energy <= p.physics.cutoff:
            return _opaque_history(
                column,
                launch.ray,
                energy,
                detector_boundaries,
                opaque_boundaries,
                seed_image,
                material_ids,
                voxel_density,
                energies,
                absorption,
                scattering,
                density,
                p,
                random,
                seed,
                history,
            )
        primary = _primary(
            ray,
            energy,
            -1,
            0,
            0,
            1,
            seed_image,
            material_ids,
            voxel_density,
            energies,
            absorption,
            scattering,
            density,
            p,
        )
        if primary.status != 0:
            return result
        result.value = column.source.weight * primary.value
        plane_count = material_plane_count(column.grid)
        patch_count = material_patch_count(column.grid)
        if plane_count <= 0 or patch_count <= 0:
            return result
        global_draw = geometry_auxiliary_random4(
            seed, history, wp.uint32(0), wp.uint32(0x30000003), random
        )
        selected = wp.min(plane_count - 1, int(global_draw[0] * wp.float64(plane_count)))
        plane = select_material_plane(selected, column.grid)
        if plane.status != 0:
            return result
        active_ballistic = _ballistic_active(launch.ray, plane.axis, plane.face, column.grid, gauge)
        if active_ballistic < 0:
            return result
        redundant_ballistic = _redundant_plane(column, plane.axis, plane.face)
        if redundant_ballistic < 0:
            return result
        if column.empty_medium == 0 and active_ballistic != 0 and redundant_ballistic == 0:
            crossing = selected_plane_crossing(launch.ray, selected, column.grid)
            if crossing.status == FAILURE:
                return result
            if crossing.status == 0 and interface_relative_velocity(
                crossing.point, gauge
            ) != wp.float64(0.0):
                zero = wp.vec3d(wp.float64(0.0))
                attenuation = connection_optical_depth(
                    ray.position,
                    ray.direction,
                    zero,
                    zero,
                    crossing.distance_mm,
                    wp.float64(0.0),
                    energy,
                    wp.float64(0.0),
                    material_ids,
                    voxel_density,
                    energies,
                    absorption,
                    scattering,
                    density,
                    p,
                )
                if attenuation.status != 0:
                    return result
                insertion = _generator(
                    crossing.point,
                    crossing.patch,
                    gauge,
                    _reference_ray(crossing.incoming),
                    energy,
                    column.source.weight
                    * wp.float64(plane_count)
                    * wp.exp(-attenuation.depth)
                    * crossing.inverse_normal_projection,
                    1,
                    0,
                    p.sampling.max_events,
                    seed,
                    history,
                    seed_image,
                    material_ids,
                    voxel_density,
                    energies,
                    absorption,
                    scattering,
                    density,
                    p,
                )
                if insertion.status != 0:
                    return result
                result.derivative += insertion.derivative
                result.futures += insertion.futures
        source_index = wp.min(plane_count - 1, int(global_draw[1] * wp.float64(plane_count)))
        sampled = sample_source_boundary(
            source_boundaries[column.source_boundary_offset + source_index], global_draw[2]
        )
        if sampled.status == FAILURE:
            return result
        if sampled.status == 0 and sampled.coarea_weight != wp.float64(0.0):
            side_launch = _source_side_ray(
                column.source, sampled.coordinate, sampled.axis, sampled.face, p
            )
            if side_launch.status != 0:
                return result
            side_ray = side_launch.ray
            minus = reference_response(
                side_ray,
                energy,
                -1,
                sampled.axis,
                sampled.face,
                -1,
                material_ids,
                voxel_density,
                energies,
                absorption,
                scattering,
                density,
                p,
                seed,
                history,
                2,
                0,
                p.sampling.max_events,
                seed_image,
            )
            plus = reference_response(
                side_ray,
                energy,
                -1,
                sampled.axis,
                sampled.face,
                1,
                material_ids,
                voxel_density,
                energies,
                absorption,
                scattering,
                density,
                p,
                seed,
                history,
                2,
                0,
                p.sampling.max_events,
                seed_image,
            )
            if minus.status != 0 or plus.status != 0:
                return result
            result.derivative += (
                column.source.weight
                * wp.float64(plane_count)
                * sampled.coarea_weight
                * (plus.value - minus.value)
            )
            result.futures += 2
        detector_count = p.detector_shape[0] + p.detector_shape[1] + 2
        detector_draw = geometry_auxiliary_random4(
            seed, history, wp.uint32(0), wp.uint32(0x30000004), random
        )
        detector_index = wp.min(
            detector_count - 1, int(detector_draw[0] * wp.float64(detector_count))
        )
        sampled = sample_source_boundary(
            detector_boundaries[column.detector_boundary_offset + detector_index], detector_draw[1]
        )
        if sampled.status == FAILURE:
            return result
        if sampled.status == 0 and sampled.coarea_weight != wp.float64(0.0):
            boundary_launch = source_ray_tangent(column.source, sampled.coordinate)
            if boundary_launch.status != 0:
                return result
            boundary_ray = _reference_ray(boundary_launch.ray)
            pixel_minus = _primary(
                boundary_ray,
                energy,
                sampled.axis,
                sampled.face,
                -1,
                1,
                seed_image,
                material_ids,
                voxel_density,
                energies,
                absorption,
                scattering,
                density,
                p,
            )
            pixel_plus = _primary(
                boundary_ray,
                energy,
                sampled.axis,
                sampled.face,
                1,
                1,
                seed_image,
                material_ids,
                voxel_density,
                energies,
                absorption,
                scattering,
                density,
                p,
            )
            if pixel_minus.status != 0 or pixel_plus.status != 0:
                return result
            result.derivative += (
                column.source.weight
                * wp.float64(detector_count)
                * sampled.coarea_weight
                * (pixel_plus.value - pixel_minus.value)
            )
        detector = _relative_detector(column.detector, gauge)
        original = ray
        event = int(0)
        alive = int(1)
        while alive != 0:
            if event >= p.max_events:
                return result
            prefix = prefix_step(
                original,
                energy,
                -1,
                0,
                0,
                material_ids,
                voxel_density,
                energies,
                absorption,
                scattering,
                density,
                p,
                seed,
                history,
                0,
                event,
                p.sampling.max_events,
            )
            if prefix.status != 0:
                return result
            if prefix.terminal != 0:
                alive = 0
            else:
                result.prefixes += 1
                radiative = _continuing_radiative(
                    energy, material_ids[prefix.cell], energies, absorption, scattering, p
                )
                if radiative < 0:
                    return result
                if radiative != 0:
                    auxiliary = geometry_auxiliary_random4(
                        seed, history, wp.uint32(event), wp.uint32(0x30000000), random
                    )
                    energy_draw = geometry_auxiliary_random4(
                        seed, history, wp.uint32(event), wp.uint32(0x30000001), random
                    )
                    terminal = _terminal(
                        prefix.ray,
                        energy,
                        prefix.cell,
                        detector,
                        planes,
                        proposal,
                        auxiliary,
                        energy_draw,
                        seed_image,
                        material_ids,
                        voxel_density,
                        energies,
                        absorption,
                        scattering,
                        density,
                        p,
                    )
                    if terminal.status != 0:
                        return result
                    result.value += column.source.weight * terminal.value
                    result.derivative += column.source.weight * terminal.derivative
                    patch_index = wp.min(
                        patch_count - 1, int(energy_draw[2] * wp.float64(patch_count))
                    )
                    patch = select_material_patch(patch_index, column.grid)
                    if patch.status != 0:
                        return result
                    redundant_patch = _redundant_plane(column, patch.patch.axis, patch.patch.face)
                    if redundant_patch < 0:
                        return result
                    if redundant_patch == 0:
                        bridge_draw = geometry_auxiliary_random4(
                            seed, history, wp.uint32(event), wp.uint32(0x30000002), random
                        )
                        bridge = sample_interface_bridge(
                            patch.patch,
                            prefix.ray.position,
                            wp.vec2d(bridge_draw[0], bridge_draw[1]),
                            column.grid,
                        )
                        if bridge.status != 0:
                            return result
                        if interface_relative_velocity(bridge.point, gauge) != wp.float64(0.0):
                            mark = wp.float64(0.0)
                            energy_weight = wp.float64(1.0)
                            if (
                                (wp.static(model == 2))
                                if wp.static(model >= 0)
                                else (p.compton == 2)
                            ):
                                energy_weight = wp.max(wp.float64(0.0), energy - p.physics.cutoff)
                                mark = p.physics.cutoff + bridge_draw[2] * energy_weight
                                if energy_weight == wp.float64(0.0):
                                    mark = energy * wp.float64(0.5)
                            material = material_ids[prefix.cell]
                            channels = int(2)
                            if p.physics.enabled != 0:
                                channels += (
                                    p.physics.fluorescence_offsets[material + 1]
                                    - p.physics.fluorescence_offsets[material]
                                )
                            for channel in range(channels):
                                zero = wp.vec3d(wp.float64(0.0))
                                kernel = terminal_physical_kernel_jet(
                                    prefix.ray.position,
                                    prefix.ray.direction,
                                    zero,
                                    zero,
                                    bridge.point.point_mm,
                                    zero,
                                    energy,
                                    material,
                                    wp.min(channel, 2),
                                    channel - 2,
                                    mark,
                                    energies,
                                    absorption,
                                    scattering,
                                    p,
                                )
                                if kernel.status != 0:
                                    return result
                                if kernel.present != 0 and (
                                    p.physics.enabled == 0 or kernel.energy > p.physics.cutoff
                                ):
                                    displacement = bridge.point.point_mm - prefix.ray.position
                                    distance = _hypot(
                                        _hypot(displacement[0], displacement[1]), displacement[2]
                                    )
                                    optical = connection_optical_depth(
                                        prefix.ray.position,
                                        kernel.direction,
                                        zero,
                                        zero,
                                        distance,
                                        wp.float64(0.0),
                                        kernel.energy,
                                        wp.float64(0.0),
                                        material_ids,
                                        voxel_density,
                                        energies,
                                        absorption,
                                        scattering,
                                        density,
                                        p,
                                    )
                                    if optical.status != 0:
                                        return result
                                    importance = (
                                        column.source.weight
                                        * wp.float64(patch_count)
                                        * kernel.density
                                        * wp.exp(bridge.log_geometric_importance - optical.depth)
                                    )
                                    if kernel.measure == 1:
                                        importance *= energy_weight
                                    incoming = NativeGeometryRay()
                                    incoming.position = bridge.point.point_mm
                                    incoming.direction = kernel.direction
                                    insertion = _generator(
                                        bridge.point,
                                        patch.patch,
                                        gauge,
                                        incoming,
                                        kernel.energy,
                                        importance,
                                        3 + channel,
                                        event * p.max_events,
                                        p.sampling.max_events,
                                        seed,
                                        history,
                                        seed_image,
                                        material_ids,
                                        voxel_density,
                                        energies,
                                        absorption,
                                        scattering,
                                        density,
                                        p,
                                    )
                                    if insertion.status != 0:
                                        return result
                                    result.derivative += insertion.derivative
                                    result.futures += insertion.futures
                zero = wp.vec3d(wp.float64(0.0))
                factor = _cell_factor(
                    prefix.cell, material_ids[prefix.cell], voxel_density, density, p
                )
                collision = conditional_physical_event_jet(
                    prefix.ray.position,
                    prefix.ray.direction,
                    zero,
                    zero,
                    energy,
                    material_ids[prefix.cell],
                    factor,
                    energies,
                    absorption,
                    scattering,
                    p,
                    seed,
                    history,
                    wp.uint32(event),
                    0,
                )
                if collision.status != 0 and collision.status != 1:
                    return result
                if (
                    collision.status == 1
                    or collision.channel == 6
                    or (p.physics.enabled != 0 and collision.energy <= p.physics.cutoff)
                ):
                    alive = 0
                else:
                    original.position = prefix.ray.position
                    original.direction = collision.direction
                    energy = collision.energy
                    event += 1
        if wp.isfinite(result.value) and wp.isfinite(result.derivative):
            result.status = 0
        return result

    return geometry_history


geometry_history = _makegeometry_history(
    -1,
    False,
    "generic",
    conditional_physical_event_jet,
    terminal_physical_kernel_jet,
    reference_response,
    _generator,
    _terminal,
)


def _make_geometry_observations(history: Any, model: int = -1):
    """Register one observation kernel with only its selected function closure."""
    geometry_history = history
    module_options = _GEOMETRY_OPTIONS
    if type(model) is int and model == 2:
        module_options = {**_GEOMETRY_OPTIONS, "optimization_level": 2}

    @wp.kernel(module="unique", module_options=module_options)
    def geometry_observations(
        columns: wp.array(dtype=NativeGeometryColumn),
        planes: wp.array(dtype=DetectorProposalPlane),
        proposal: ForwardProposalGeometry,
        source_boundaries: wp.array(dtype=NativeAffineSourceBoundary),
        detector_boundaries: wp.array(dtype=NativeAffineSourceBoundary),
        opaque_boundaries: wp.array(dtype=NativeOpaqueBoundary),
        seed_image: wp.array(dtype=wp.float64),
        material_ids: wp.array(dtype=wp.int32),
        voxel_density: wp.array(dtype=wp.float64),
        energies: wp.array(dtype=wp.float64),
        absorption: wp.array(dtype=wp.float64),
        scattering: wp.array(dtype=wp.float64),
        density: wp.array(dtype=wp.float64),
        p: Parameters,
        random: GeometryRandomDesign,
        seed: wp.uint64,
        first_history: wp.uint64,
        values: wp.array2d(dtype=wp.float64),
        derivatives: wp.array2d(dtype=wp.float64),
        counts: wp.array2d(dtype=wp.vec2i),
        status: wp.array(dtype=wp.int32),
    ):
        history_index, parameter_index = wp.tid()
        result = geometry_history(
            columns[parameter_index],
            planes,
            proposal,
            source_boundaries,
            detector_boundaries,
            opaque_boundaries,
            seed_image,
            material_ids,
            voxel_density,
            energies,
            absorption,
            scattering,
            density,
            p,
            random,
            seed,
            first_history + wp.uint64(history_index),
        )
        if result.status != 0:
            wp.atomic_or(status, 0, FAILURE)
            values[history_index, parameter_index] = wp.float64(wp.nan)
            derivatives[history_index, parameter_index] = wp.float64(wp.nan)
            counts[history_index, parameter_index] = wp.vec2i(0, 0)
        else:
            values[history_index, parameter_index] = result.value
            derivatives[history_index, parameter_index] = result.derivative
            counts[history_index, parameter_index] = wp.vec2i(result.prefixes, result.futures)

    return geometry_observations


geometry_observations = _make_geometry_observations(geometry_history)


@wp.kernel(module="unique", module_options=_GEOMETRY_OPTIONS)
def reduce_geometry_observations(
    values: wp.array2d(dtype=wp.float64),
    derivatives: wp.array2d(dtype=wp.float64),
    mean: wp.array(dtype=wp.float64),
    gradient: wp.array(dtype=wp.float64),
    second_moment: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    parameter = wp.tid()
    if status[0] != 0:
        mean[parameter] = wp.float64(wp.nan)
        gradient[parameter] = wp.float64(wp.nan)
        second_moment[parameter] = wp.float64(wp.nan)
        return
    denominator = wp.float64(derivatives.shape[0])
    value = wp.float64(0.0)
    derivative = wp.float64(0.0)
    moment = wp.float64(0.0)
    for history in range(derivatives.shape[0]):
        value += values[history, parameter] / denominator
        observation = derivatives[history, parameter]
        derivative += observation / denominator
        moment += (observation / denominator) * observation
    if not wp.isfinite(value) or not wp.isfinite(derivative) or not wp.isfinite(moment):
        wp.atomic_or(status, 0, FAILURE)
        value = wp.float64(wp.nan)
        derivative = wp.float64(wp.nan)
        moment = wp.float64(wp.nan)
    mean[parameter] = value
    gradient[parameter] = derivative
    second_moment[parameter] = moment


@dataclass(frozen=True)
class _GeometryKernelBundle:
    geometry_observations: Any
    reduce_geometry_observations: Any


@lru_cache(maxsize=6)
def geometry_kernel_bundle(model: int, packed: bool) -> _GeometryKernelBundle:
    """Prepare one of six fixed model/sampler closures; device arrays stay arguments.

    Runtime validation and every random mark remain in their canonical bodies.
    The unique observation module contains one kernel, and reduction is separate.
    No energy, material, pointer or device value enters this bounded cache key.
    """
    if type(model) is not int or model not in (0, 1, 2) or type(packed) is not bool:
        raise ValueError("geometry kernel specialization requires a canonical model and sampler")
    label = f"m{model}_p{int(packed)}"
    collision = _makeconditional_physical_event_jet(model, packed, label)
    terminal = _maketerminal_physical_kernel_jet(model, label)
    response = _makereference_response(label, collision)
    generator = _make_generator(label, response)
    terminal_observation = _make_terminal(model, label, terminal)
    history = _makegeometry_history(
        model, packed, label, collision, terminal, response, generator, terminal_observation
    )
    return _GeometryKernelBundle(
        _make_geometry_observations(history, model), reduce_geometry_observations
    )
