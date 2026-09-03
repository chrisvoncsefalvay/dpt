"""Complete analogue detector continuations for physical geometry measures.

The supplied incoming keV and random identity are fixed. A continuation samples
the complete physical collision law and evaluates its escaped photon/energy
response against a signed pixel seed. Physical resource guards invalidate the
whole observation. These helpers write no model, seed or sampling arrays.
"""

# Warp annotations and decorators are runtime DSL records.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false
# pyright: reportPrivateUsage=false, reportUntypedClassDecorator=false
# Mutable Warp loop counters require explicit integer constructors.
# ruff: noqa: UP018, RUF046
import importlib
from typing import Any

from .geometry_interface_kernels import _ulp
from .kernels import (
    Parameters,
    cell_index,
    coefficients_at,
    cross_faces,
    detector_pixel,
    distances_to_faces,
    energy_coordinate,
    grid_face,
)
from .physical_geometry_kernels import (
    _ray_jet_valid,
    _reference_model_valid,
    conditional_physical_event_jet,
)
from .sampling_kernels import transport_random4

wp: Any = importlib.import_module("warp")
wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})


@wp.struct
class NativeGeometryRay:
    """Immutable world mm position and unit direction for one continuation."""

    position: wp.vec3d
    direction: wp.vec3d


@wp.struct
class NativeGeometryPrefix:
    """Reached collision or completed flight, before the physical channel mark.

    ``terminal``0 is a real collision,1 an escape and2 local cutoff deposition.
    ``status``0 is valid and4 invalidates the whole history. ``event`` is the
    explicitly mapped counter and ``orders`` is initialised to zero for the
    caller's prefix enumeration. Crossings count positive cell intervals in the
    complete box partition, including those beyond the realised collision.
    Energy is keV, distance mm and total_optical_depth is dimensionless.
    """

    ray: NativeGeometryRay
    energy: wp.float64
    cell: int
    event: int
    orders: int
    terminal: int
    status: int
    crossings: int
    distance: wp.float64
    total_optical_depth: wp.float64


@wp.struct
class NativeGeometryResponse:
    """Complete signed detector response and physical terminal diagnostic.

    ``terminal``0/1/2 identifies escape/absorption/cutoff. ``status``0 permits
    reduction and4 invalidates it. collision_count includes a forced first
    collision. final_energy is the last incident keV for absorption/cutoff and
    the escaped surviving keV for escape, matching the host reference. It is
    neither an energy-balance ledger nor a deposited-energy measurement.
    """

    value: wp.float64
    status: int
    collision_count: int
    final_energy: wp.float64
    terminal: int


@wp.func
def _valid_reference(
    ray: NativeGeometryRay,
    energy: wp.float64,
    branch: int,
    address_event: int,
    coordinate_event_limit: int,
    p: Parameters,
    seed: wp.uint64,
    history: wp.uint64,
) -> int:
    zero = wp.vec3d(wp.float64(0.0))
    valid = int(1)
    if (
        _ray_jet_valid(ray.position, ray.direction, zero, zero) == 0
        or _reference_model_valid(p) == 0
        or p.max_events < 1
        or p.max_crossings < 1
        or coordinate_event_limit < p.max_events
        or address_event < 0
        or address_event >= coordinate_event_limit
        or branch < 0
        or branch >= 4096
        or not wp.isfinite(energy)
        or energy <= wp.float64(0.0)
    ):
        valid = 0
    if p.sampling.kind != 0:
        if (
            p.sampling.max_events < coordinate_event_limit
            or p.sampling.max_trials < p.max_angle_trials
            or branch >= p.sampling.max_branches
            or seed != p.sampling.seed
            or history < p.sampling.first_history
            or history - p.sampling.first_history >= wp.uint64(p.sampling.count)
        ):
            valid = 0
    return valid


@wp.func
def _valid_buffers(
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    p: Parameters,
) -> int:
    cells = wp.int64(p.shape[0]) * wp.int64(p.shape[1]) * wp.int64(p.shape[2])
    valid = int(1)
    if (
        p.shape[0] < 1
        or p.shape[1] < 1
        or p.shape[2] < 1
        or cells != wp.int64(material_ids.shape[0])
        or energies.shape[0] != p.energy_bins
        or absorption.shape[0] != scattering.shape[0]
        or absorption.shape[0] % p.energy_bins != 0
        or density.shape[0] != absorption.shape[0] // p.energy_bins
        or (p.voxel_density != 0 and voxel_density.shape[0] != material_ids.shape[0])
    ):
        valid = 0
    return valid


@wp.func
def _cell_factor(
    cell: int,
    material: int,
    voxel_density: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    p: Parameters,
) -> wp.float64:
    if material < 0 or material >= density.shape[0]:
        return wp.float64(-1.0)
    factor = density[material]
    if not wp.isfinite(factor) or factor <= wp.float64(0.0):
        return wp.float64(-1.0)
    if p.voxel_density != 0:
        voxel = voxel_density[cell]
        if not wp.isfinite(voxel) or voxel < wp.float64(0.0):
            return wp.float64(-1.0)
        factor *= voxel
        if voxel > wp.float64(0.0) and factor <= wp.float64(0.0):
            return wp.float64(-1.0)
    if not wp.isfinite(factor):
        return wp.float64(-1.0)
    return factor


@wp.func
def prefix_step(
    ray: NativeGeometryRay,
    energy: wp.float64,
    side_axis: int,
    side_face: int,
    side_sign: int,
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    p: Parameters,
    seed: wp.uint64,
    history: wp.uint64,
    branch: int,
    address_event: int,
    coordinate_event_limit: int,
) -> NativeGeometryPrefix:
    """Select one fixed exponential-depth flight in the complete cell partition.

    Model arrays are prepared, immutable material-major/energy-fastest mm^-1
    coefficients, dimensionless positive material densities and optional
    nonnegative voxel density factors. A side_axis of minus one disables the
    exact parallel side label. Other side labels select face or face-1 without
    displacing a ray by an epsilon. Both physical and coordinate budgets are
    explicit, and the complete partition crossing guard applies after an early
    realised collision as it does in the host reference.

    """
    result = NativeGeometryPrefix()
    result.ray.position = ray.position
    result.ray.direction = ray.direction
    result.energy = energy
    result.event = address_event
    result.cell = -1
    result.terminal = 1
    result.status = 4
    if (
        _valid_reference(
            ray, energy, branch, address_event, coordinate_event_limit, p, seed, history
        )
        == 0
    ):
        return result
    if (
        _valid_buffers(material_ids, voxel_density, energies, absorption, scattering, density, p)
        == 0
    ):
        return result
    local_position = p.inverse_rotation * ray.position + p.inverse_translation
    local_direction = p.inverse_rotation * ray.direction
    zero = wp.vec3d(wp.float64(0.0))
    if _ray_jet_valid(local_position, local_direction, zero, zero) == 0:
        return result
    if side_axis < -1 or side_axis > 2:
        return result
    selected_side = int(-1)
    if side_axis >= 0:
        if (
            side_face < 0
            or side_face > p.shape[side_axis]
            or (side_sign != -1 and side_sign != 1)
            or wp.abs(local_direction[side_axis]) > wp.float64(512.0 * 2.220446049250313e-16)
        ):
            return result
        location = p.origin[side_axis] + wp.float64(side_face) * p.spacing[side_axis]
        scale = wp.max(wp.float64(1.0), wp.max(wp.abs(location), wp.abs(local_position[side_axis])))
        for axis in range(3):
            scale = wp.max(scale, wp.abs(ray.position[axis]))
            scale = wp.max(scale, wp.abs(local_position[axis]))
        if wp.abs(local_position[side_axis] - location) > wp.float64(512.0) * _ulp(scale):
            return result
        # The declared side chart fixes this local component. Ordinary
        # unlabelled rays and transverse forced collisions do not enter here.
        local_direction[side_axis] = wp.float64(0.0)
        selected_side = side_face - int(side_sign < 0)
        if selected_side < 0 or selected_side >= p.shape[side_axis]:
            result.status = 0
            return result
    entry = wp.float64(0.0)
    exit_distance = wp.float64(wp.inf)
    absent = int(0)
    for axis in range(3):
        lower = p.origin[axis]
        upper = grid_face(lower, p.spacing[axis], p.shape[axis])
        if local_direction[axis] == wp.float64(0.0):
            if axis != side_axis and (
                local_position[axis] < lower or local_position[axis] >= upper
            ):
                absent = 1
        else:
            first = (lower - local_position[axis]) / local_direction[axis]
            second = (upper - local_position[axis]) / local_direction[axis]
            if wp.isnan(first) or wp.isnan(second):
                return result
            entry = wp.max(entry, wp.min(first, second))
            exit_distance = wp.min(exit_distance, wp.max(first, second))
    if absent != 0 or exit_distance <= entry:
        result.status = 0
        return result
    if not wp.isfinite(entry) or not wp.isfinite(exit_distance):
        return result
    if p.physics.enabled != 0 and energy <= p.physics.cutoff:
        result.terminal = 2
        result.status = 0
        return result
    if energy < energies[0] or energy > energies[p.energy_bins - 1]:
        return result
    if p.compton == 2 and (energy < wp.float64(0.1) or energy > wp.float64(5000.0)):
        return result
    draw = transport_random4(
        seed, history, wp.uint32(address_event), wp.uint32(branch) * wp.uint32(65536), p.sampling
    )[0]
    if not wp.isfinite(draw) or draw <= wp.float64(0.0) or draw >= wp.float64(1.0):
        return result
    optical_depth = -wp.log(draw)
    current = local_position + entry * local_direction
    for axis in range(3):
        current[axis] = wp.clamp(
            current[axis], p.origin[axis], grid_face(p.origin[axis], p.spacing[axis], p.shape[axis])
        )
    cell = cell_index(current, local_direction, p)
    if side_axis >= 0:
        cell[side_axis] = selected_side
    coordinate = energy_coordinate(energy, energies, p)
    travelled = entry
    depth = wp.float64(0.0)
    alive = int(1)
    found = int(0)
    steps = wp.int64(0)
    maximum_steps = wp.int64(p.shape[0]) + wp.int64(p.shape[1]) + wp.int64(p.shape[2]) + wp.int64(3)
    while alive != 0:
        flat = (cell[2] * p.shape[1] + cell[1]) * p.shape[0] + cell[0]
        material = material_ids[flat]
        if material < 0 or material >= density.shape[0]:
            return result
        factor = _cell_factor(flat, material, voxel_density, density, p)
        if factor < wp.float64(0.0):
            return result
        partials = coefficients_at(material, coordinate, absorption, scattering, p)
        unscaled = partials[0] + partials[1]
        if (
            not wp.isfinite(partials[0])
            or not wp.isfinite(partials[1])
            or partials[0] < wp.float64(0.0)
            or partials[1] < wp.float64(0.0)
            or not wp.isfinite(unscaled)
        ):
            return result
        rate = factor * unscaled
        if not wp.isfinite(rate) or (
            factor > wp.float64(0.0) and unscaled > wp.float64(0.0) and rate <= wp.float64(0.0)
        ):
            return result
        faces = distances_to_faces(current, local_direction, cell, p)
        length = wp.min(faces[0], wp.min(faces[1], faces[2]))
        if not wp.isfinite(length) or length < wp.float64(0.0):
            return result
        if length > wp.float64(0.0):
            result.crossings += 1
            if result.crossings > p.max_crossings:
                return result
        following = depth + rate * length
        if not wp.isfinite(following):
            return result
        if rate > wp.float64(0.0) and length > wp.float64(0.0):
            if optical_depth == depth or optical_depth == following:
                return result
            if found == 0 and depth < optical_depth and optical_depth < following:
                result.distance = travelled + (optical_depth - depth) / rate
                result.ray.position = ray.position + result.distance * ray.direction
                result.cell = flat
                result.terminal = 0
                found = 1
        depth = following
        travelled += length
        if not wp.isfinite(travelled):
            return result
        current += length * local_direction
        crossing = cross_faces(current, local_direction, cell, faces, length, p)
        current = crossing.position
        cell = crossing.cell
        alive = crossing.inside
        steps += wp.int64(1)
        if steps > maximum_steps:
            return result
    result.total_optical_depth = depth
    zero = wp.vec3d(wp.float64(0.0))
    if _ray_jet_valid(result.ray.position, result.ray.direction, zero, zero) == 0:
        return result
    result.status = 0
    return result


def _makereference_response(label: str, collision: Any):
    """Bind a prepared model without duplicating the mathematical body."""
    conditional_physical_event_jet = collision

    @wp.func(
        name="reference_response" if label == "generic" else f"reference_response_{label}",
        module="unique",
    )
    def reference_response(
        initial: NativeGeometryRay,
        energy: wp.float64,
        forced_cell: int,
        side_axis: int,
        side_face: int,
        side_sign: int,
        material_ids: wp.array(dtype=wp.int32),
        voxel_density: wp.array(dtype=wp.float64),
        energies: wp.array(dtype=wp.float64),
        absorption: wp.array(dtype=wp.float64),
        scattering: wp.array(dtype=wp.float64),
        density: wp.array(dtype=wp.float64),
        p: Parameters,
        seed: wp.uint64,
        history: wp.uint64,
        branch: int,
        event_offset: int,
        coordinate_event_limit: int,
        seed_image: wp.array(dtype=wp.float64),
    ) -> NativeGeometryResponse:
        """Complete one signed physical detector continuation at fixed source keV.

        A forced_cell of minus one starts an ordinary flight, while a valid cell
        starts its explicit positive-rate conditional collision. Physical local
        events satisfy f<p.max_events and canonical addressed events are
        event_offset+f<coordinate_event_limit. Only the local conditional helper's
        Parameters copy widens its counter admission to that declared coordinate
        limit. The original physical guard and sampling record remain unchanged.
        Exact parallel side ownership persists until an outgoing local direction
        leaves its face. Source weights/energy-spectrum marks are caller owned.
        seed_image is one immutable finite float64 signed seed per row-major pixel;
        the caller validates the complete seed/model before launching continuations.

        """
        result = NativeGeometryResponse()
        result.status = 4
        result.final_energy = energy
        if (
            _valid_reference(
                initial, energy, branch, event_offset, coordinate_event_limit, p, seed, history
            )
            == 0
        ):
            return result
        if (
            _valid_buffers(
                material_ids, voxel_density, energies, absorption, scattering, density, p
            )
            == 0
        ):
            return result
        pixels = wp.int64(p.detector_shape[0]) * wp.int64(p.detector_shape[1])
        if (
            p.detector_shape[0] < 1
            or p.detector_shape[1] < 1
            or pixels != wp.int64(seed_image.shape[0])
        ):
            return result
        if forced_cell < -1 or forced_cell >= material_ids.shape[0]:
            return result
        if side_axis < -1 or side_axis > 2:
            return result
        if side_axis >= 0 and (
            side_face < 0 or side_face > p.shape[side_axis] or (side_sign != -1 and side_sign != 1)
        ):
            return result
        ray = NativeGeometryRay()
        ray.position = initial.position
        ray.direction = initial.direction
        incident_energy = energy
        cell = forced_cell
        active_side = side_axis
        local_event = int(0)
        physical_budget = p.max_events
        zero = wp.vec3d(wp.float64(0.0))
        while True:
            result.final_energy = incident_energy
            if (
                p.physics.enabled != 0
                and incident_energy <= p.physics.cutoff
                and (cell >= 0 or local_event > 0)
            ):
                result.terminal = 2
                result.status = 0
                return result
            if local_event >= physical_budget:
                return result
            mapped = wp.int64(event_offset) + wp.int64(local_event)
            if mapped >= wp.int64(coordinate_event_limit):
                return result
            address_event = int(mapped)
            if cell < 0:
                prefix = prefix_step(
                    ray,
                    incident_energy,
                    active_side,
                    side_face,
                    side_sign,
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
                    address_event,
                    coordinate_event_limit,
                )
                if prefix.status != 0:
                    return result
                if prefix.terminal == 2:
                    result.terminal = 2
                    result.status = 0
                    return result
                if prefix.terminal == 1:
                    local_position = p.inverse_rotation * ray.position + p.inverse_translation
                    local_direction = p.inverse_rotation * ray.direction
                    pixel = detector_pixel(local_position, local_direction, p)
                    if pixel == -2:
                        return result
                    if pixel >= 0:
                        value = seed_image[pixel]
                        if p.energy_score != 0:
                            value *= incident_energy
                        if not wp.isfinite(value):
                            return result
                        result.value = value
                    result.terminal = 0
                    result.status = 0
                    return result
                ray = prefix.ray
                cell = prefix.cell
            material = material_ids[cell]
            factor = _cell_factor(cell, material, voxel_density, density, p)
            if factor <= wp.float64(0.0):
                return result
            # A new struct prevents a widened counter budget aliasing the caller's
            # physical guard. Array/physical/sampling records stay read-only.
            counter_parameters = Parameters()
            counter_parameters.origin = p.origin
            counter_parameters.spacing = p.spacing
            counter_parameters.shape = p.shape
            counter_parameters.detector_lower = p.detector_lower
            counter_parameters.detector_spacing = p.detector_spacing
            counter_parameters.detector_shape = p.detector_shape
            counter_parameters.detector_z = p.detector_z
            counter_parameters.detector_general = p.detector_general
            counter_parameters.detector_corner = p.detector_corner
            counter_parameters.detector_u = p.detector_u
            counter_parameters.detector_v = p.detector_v
            counter_parameters.detector_normal = p.detector_normal
            counter_parameters.inverse_rotation = p.inverse_rotation
            counter_parameters.inverse_translation = p.inverse_translation
            counter_parameters.source_energy = p.source_energy
            counter_parameters.energy_bins = p.energy_bins
            counter_parameters.max_events = coordinate_event_limit
            counter_parameters.max_crossings = p.max_crossings
            counter_parameters.max_angle_trials = p.max_angle_trials
            counter_parameters.compton = p.compton
            counter_parameters.compton_packed = p.compton_packed
            counter_parameters.energy_score = p.energy_score
            counter_parameters.continuous_absorption = p.continuous_absorption
            counter_parameters.voxel_density = p.voxel_density
            counter_parameters.physics = p.physics
            counter_parameters.sampling = p.sampling
            collision = conditional_physical_event_jet(
                ray.position,
                ray.direction,
                zero,
                zero,
                incident_energy,
                material,
                factor,
                energies,
                absorption,
                scattering,
                counter_parameters,
                seed,
                history,
                wp.uint32(address_event),
                branch,
            )
            result.collision_count += 1
            if collision.status == 1:
                result.terminal = 1
                if collision.channel == 6:
                    result.terminal = 2
                result.status = 0
                return result
            if collision.status != 0:
                return result
            ray.position = collision.origin
            ray.direction = collision.direction
            incident_energy = collision.energy
            if active_side >= 0:
                outgoing_local = p.inverse_rotation * ray.direction
                if outgoing_local[active_side] != wp.float64(0.0):
                    active_side = -1
            local_event += 1
            cell = -1

    return reference_response


reference_response = _makereference_response("generic", conditional_physical_event_jet)
