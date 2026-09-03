"""Terminal optical-depth jets with fixed material geometry and moving rays.

Object/source coefficient motion belongs to the signed generator estimator.
This module differentiates only the terminal connection in that fixed chart.
Statuses invalidate the original photon; no failed interval is discarded.
"""

import warp as wp

from ._constants import NUMERICAL_FAILURE
from .compton_geometry_boundary_kernels import NativeBoundaryEndpoint
from .geometry_interface_kernels import _finite3, _hypot
from .kernels import Parameters, coefficients_at, energy_coordinate
from .physical_geometry_kernels import NativePhysicalTerminal, _ray_jet_valid

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})
FAILURE = wp.constant(NUMERICAL_FAILURE)
INFINITE_DISTANCE = wp.constant(wp.float64(1.7976931348623157e308))


@wp.struct
class NativeOpticalConnection:
    depth: wp.float64
    depth_dot: wp.float64
    segments: int
    status: int


@wp.struct
class NativeTerminalObservation:
    value: wp.float64
    derivative: wp.float64
    status: int


@wp.func
def connection_optical_depth(
    position: wp.vec3d,
    direction: wp.vec3d,
    position_dot: wp.vec3d,
    direction_dot: wp.vec3d,
    maximum_distance: wp.float64,
    maximum_distance_dot: wp.float64,
    energy: wp.float64,
    energy_dot: wp.float64,
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    p: Parameters,
) -> NativeOpticalConnection:
    """Integrate the actual cell partition and each analytic face-root jet."""
    result = NativeOpticalConnection()
    result.status = FAILURE
    if (
        _ray_jet_valid(position, direction, position_dot, direction_dot) == 0
        or not wp.isfinite(maximum_distance)
        or maximum_distance <= wp.float64(0.0)
        or not wp.isfinite(maximum_distance_dot)
        or not wp.isfinite(energy)
        or not wp.isfinite(energy_dot)
        or energy < energies[0]
        or energy > energies[p.energy_bins - 1]
    ):
        return result
    coordinate = energy_coordinate(energy, energies, p)
    lower = int(coordinate[0])
    if energy_dot != wp.float64(0.0):
        for node in range(1, p.energy_bins - 1):
            if energy == energies[node]:
                return result
    origin = p.inverse_rotation * position + p.inverse_translation
    ray = p.inverse_rotation * direction
    origin_dot = p.inverse_rotation * position_dot
    ray_dot = p.inverse_rotation * direction_dot
    if (
        not _finite3(origin)
        or not _finite3(ray)
        or not _finite3(origin_dot)
        or not _finite3(ray_dot)
    ):
        return result
    next_index = wp.vec3i(0, 0, 0)
    step = wp.vec3i(0, 0, 0)
    for axis in range(3):
        if ray[axis] == wp.float64(0.0) and (
            origin_dot[axis] != wp.float64(0.0) or ray_dot[axis] != wp.float64(0.0)
        ):
            for face_index in range(p.shape[axis] + 1):
                if origin[axis] == p.origin[axis] + wp.float64(face_index) * p.spacing[axis]:
                    return result
        upper = p.origin[axis] + wp.float64(p.shape[axis]) * p.spacing[axis]
        if ray[axis] > wp.float64(0.0):
            step[axis] = 1
            next_index[axis] = p.shape[axis] + 1
            if origin[axis] < p.origin[axis]:
                next_index[axis] = 0
            elif origin[axis] <= upper:
                next_index[axis] = (
                    int(wp.floor((origin[axis] - p.origin[axis]) / p.spacing[axis])) + 1
                )
        elif ray[axis] < wp.float64(0.0):
            step[axis] = -1
            next_index[axis] = -1
            if origin[axis] > upper:
                next_index[axis] = p.shape[axis]
            elif origin[axis] >= p.origin[axis]:
                next_index[axis] = (
                    int(wp.ceil((origin[axis] - p.origin[axis]) / p.spacing[axis])) - 1
                )
    entry = wp.float64(0.0)
    entry_dot = wp.float64(0.0)
    depth = wp.float64(0.0)
    depth_dot = wp.float64(0.0)
    intervals = int(0)  # noqa: UP018, RUF046 - explicit Warp integer initialization
    for iteration in range(p.shape[0] + p.shape[1] + p.shape[2] + 4):  # noqa: B007
        roots = wp.vec3d(INFINITE_DISTANCE, INFINITE_DISTANCE, INFINITE_DISTANCE)
        root_dots = wp.vec3d(wp.float64(0.0))
        for axis in range(3):
            index = next_index[axis]
            if step[axis] != 0 and index >= 0 and index <= p.shape[axis]:
                face = p.origin[axis] + wp.float64(index) * p.spacing[axis]
                root = (face - origin[axis]) / ray[axis]
                if root > entry:
                    roots[axis] = root
                    root_dots[axis] = (-origin_dot[axis] - root * ray_dot[axis]) / ray[axis]
                else:
                    return result
        exit_distance = wp.min(maximum_distance, wp.min(roots[0], wp.min(roots[1], roots[2])))
        exit_dot = maximum_distance_dot
        selected = int(-1)  # noqa: UP018, RUF046 - explicit Warp integer initialization
        for axis in range(3):
            if roots[axis] == exit_distance and exit_distance < maximum_distance:
                if selected >= 0 and root_dots[axis] != exit_dot:
                    return result
                selected = axis
                exit_dot = root_dots[axis]
        if not wp.isfinite(exit_distance) or not wp.isfinite(exit_dot) or exit_distance <= entry:
            return result
        midpoint = origin + (entry + (exit_distance - entry) * wp.float64(0.5)) * ray
        cell = wp.vec3i(0, 0, 0)
        inside = int(1)  # noqa: UP018, RUF046 - explicit Warp integer initialization
        for axis in range(3):
            if (
                midpoint[axis] < p.origin[axis]
                or midpoint[axis] >= p.origin[axis] + wp.float64(p.shape[axis]) * p.spacing[axis]
            ):
                inside = 0
            else:
                cell[axis] = int(wp.floor((midpoint[axis] - p.origin[axis]) / p.spacing[axis]))
        if inside != 0:
            intervals += 1
            if intervals > p.max_crossings:
                return result
            index = (cell[2] * p.shape[1] + cell[1]) * p.shape[0] + cell[0]
            material = material_ids[index]
            if material < 0 or material >= density.shape[0]:
                return result
            factor = density[material]
            if p.voxel_density != 0:
                factor *= voxel_density[index]
            rates = coefficients_at(material, coordinate, absorption, scattering, p)
            rate = factor * (rates[0] + rates[1])
            rate_dot = wp.float64(0.0)
            if p.energy_bins > 1:
                offset = material * p.energy_bins + lower
                slope = (
                    absorption[offset + 1]
                    + scattering[offset + 1]
                    - absorption[offset]
                    - scattering[offset]
                ) / (energies[lower + 1] - energies[lower])
                rate_dot = factor * slope * energy_dot
            if not wp.isfinite(rate) or rate < wp.float64(0.0) or not wp.isfinite(rate_dot):
                return result
            depth += rate * (exit_distance - entry)
            depth_dot += rate * (exit_dot - entry_dot) + rate_dot * (exit_distance - entry)
        if exit_distance == maximum_distance:
            if not wp.isfinite(depth) or not wp.isfinite(depth_dot):
                return result
            result.depth = depth
            result.depth_dot = depth_dot
            result.segments = intervals
            result.status = 0
            return result
        for axis in range(3):
            if roots[axis] == exit_distance:
                next_index[axis] += step[axis]
        entry = exit_distance
        entry_dot = exit_dot
    return result


@wp.func
def terminal_connection_observation(
    kernel: NativePhysicalTerminal,
    endpoint: NativeBoundaryEndpoint,
    log_reference_density: wp.float64,
    seed: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    p: Parameters,
) -> NativeTerminalObservation:
    """Compose phase, area, inverse-square, attenuation and signed response jets."""
    result = NativeTerminalObservation()
    result.status = FAILURE
    if kernel.status != 0 or endpoint.pixel < 0 or endpoint.pixel >= seed.shape[0]:
        return result
    if kernel.present == 0:
        result.status = 0
        return result
    displacement = endpoint.point_mm - kernel.origin
    distance = _hypot(_hypot(displacement[0], displacement[1]), displacement[2])
    distance_dot = wp.dot(kernel.direction, endpoint.point_dot - kernel.origin_dot)
    projection = wp.dot(endpoint.normal, kernel.direction)
    projection_dot = wp.dot(endpoint.normal_dot, kernel.direction) + wp.dot(
        endpoint.normal, kernel.direction_dot
    )
    if distance <= wp.float64(0.0) or projection <= wp.float64(0.0) or not wp.isfinite(distance):
        return result
    optical = connection_optical_depth(
        kernel.origin,
        kernel.direction,
        kernel.origin_dot,
        kernel.direction_dot,
        distance,
        distance_dot,
        kernel.energy,
        kernel.energy_dot,
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
    area = endpoint.area_jacobian
    if not wp.isfinite(area) or area <= wp.float64(0.0):
        return result
    measure = wp.exp(
        wp.log(area)
        + wp.log(projection)
        - wp.float64(2.0) * wp.log(distance)
        - log_reference_density
        - optical.depth
    )
    explicit = wp.float64(1.0)
    explicit_dot = wp.float64(0.0)
    if p.energy_score != 0:
        explicit = kernel.energy
        explicit_dot = kernel.energy_dot
    value = seed[endpoint.pixel] * measure * kernel.density * explicit
    derivative = (
        seed[endpoint.pixel]
        * measure
        * (
            explicit
            * (
                kernel.density_dot
                + kernel.density
                * (
                    endpoint.area_jacobian_dot / area
                    + projection_dot / projection
                    - wp.float64(2.0) * distance_dot / distance
                    - optical.depth_dot
                )
            )
            + kernel.density * explicit_dot
        )
    )
    if not wp.isfinite(value) or not wp.isfinite(derivative):
        return result
    result.value = value
    result.derivative = derivative
    result.status = 0
    return result
