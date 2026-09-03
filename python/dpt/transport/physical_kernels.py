"""Atomic angular samplers and explicit energy/majorant records for Warp transport."""

# Warp annotations are runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false
import warp as wp

from dpt.kernels.numerics import product3

from .sampling_kernels import RandomDesign, transport_random4
from .spectrum_kernels import _spectrum_index

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})
HC = wp.constant(wp.float64(12.398419843320026))
PI = wp.constant(wp.float64(3.14159265358979323846))


@wp.struct
class PhysicalParameters:
    enabled: int
    sampling: RandomDesign
    cutoff: wp.float64
    woodcock: int
    random_branch: int
    rayleigh: wp.array(dtype=wp.float64)
    majorant: wp.array(dtype=wp.float64)
    q_offsets: wp.array(dtype=wp.int32)
    q: wp.array(dtype=wp.float64)
    form_factor: wp.array(dtype=wp.float64)
    scattering_function: wp.array(dtype=wp.float64)
    atomic_number: wp.array(dtype=wp.float64)
    compton_shell_offsets: wp.array(dtype=wp.int32)
    compton_occupancy: wp.array(dtype=wp.float64)
    compton_binding: wp.array(dtype=wp.float64)
    compton_height: wp.array(dtype=wp.float64)
    fluorescence_offsets: wp.array(dtype=wp.int32)
    fluorescence_energy: wp.array(dtype=wp.float64)
    fluorescence_threshold: wp.array(dtype=wp.float64)
    fluorescence_probability: wp.array(dtype=wp.float64)
    spectrum_bins: int
    spectrum_energies: wp.array(dtype=wp.float64)
    spectrum_cdf: wp.array(dtype=wp.float64)
    record_deposition: int
    deposition: wp.array(dtype=wp.float64)


@wp.func
def record_deposition(
    physical: PhysicalParameters,
    voxel: int,
    energy: wp.float64,
    source_weight: wp.float64,
    source_amplitude: wp.float64,
):
    if physical.record_deposition != 0:
        wp.atomic_add(physical.deposition, voxel, product3(energy, source_weight, source_amplitude))


@wp.func
def spectrum_index(draw: wp.float64, physical: PhysicalParameters) -> int:
    return _spectrum_index(physical.spectrum_cdf, draw)


@wp.func
def energy_table_value(
    values: wp.array(dtype=wp.float64), material: int, coordinate: wp.vec2d, bins: int
) -> wp.float64:
    offset = material * bins
    if bins == 1:
        return values[offset]
    lower = offset + int(coordinate[0])
    return (wp.float64(1.0) - coordinate[1]) * values[lower] + coordinate[1] * values[lower + 1]


@wp.func
def atomic_value(
    q: wp.float64, material: int, physical: PhysicalParameters, coherent: int
) -> wp.float64:
    lower = physical.q_offsets[material]
    upper = physical.q_offsets[material + 1] - 1
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if physical.q[middle] <= q:
            lower = middle
        else:
            upper = middle
    fraction = (q - physical.q[lower]) / (physical.q[upper] - physical.q[lower])
    left = physical.form_factor[lower]
    right = physical.form_factor[upper]
    if coherent == 0:
        left = physical.scattering_function[lower]
        right = physical.scattering_function[upper]
    return (wp.float64(1.0) - fraction) * left + fraction * right


@wp.func
def rayleigh_scatter_event(
    energy: wp.float64,
    material: int,
    seed: wp.uint64,
    history: wp.uint64,
    event: wp.uint32,
    max_trials: int,
    physical: PhysicalParameters,
) -> wp.vec4d:
    result = wp.vec4d(wp.float64(0.0), wp.float64(0.0), energy, wp.float64(6.0))
    for trial in range(max_trials):
        angular = transport_random4(
            seed,
            history,
            event,
            wp.uint32(0xC0000000)
            + wp.uint32(physical.random_branch) * wp.uint32(65536)
            + wp.uint32(trial),
            physical.sampling,
        )
        cosine = wp.float64(2.0) * angular[0] - wp.float64(1.0)
        q = energy / HC * wp.sqrt((wp.float64(1.0) - cosine) * wp.float64(0.5))
        factor = atomic_value(q, material, physical, 1) / physical.atomic_number[material]
        acceptance = wp.float64(0.5) * (wp.float64(1.0) + cosine * cosine) * factor * factor
        if angular[1] < acceptance:
            result = wp.vec4d(cosine, wp.float64(2.0) * PI * angular[2], energy, wp.float64(0.0))
            break
    return result


@wp.func
def rayleigh_phase_pdf(
    energy: wp.float64, material: int, cosine: wp.float64, physical: PhysicalParameters
) -> wp.float64:
    """Solid-angle density normalised for the represented piecewise-linear F.

    Four-point Gauss integration is exact for each degree-seven q integrand.
    The final interval is clipped at E/hc, rather than extending table support.
    """
    q_max = energy / HC
    integral = wp.float64(0.0)
    nodes = wp.vec4d(
        -0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526
    )
    weights = wp.vec4d(
        0.34785484513745385, 0.6521451548625461, 0.6521451548625461, 0.34785484513745385
    )
    for index in range(physical.q_offsets[material], physical.q_offsets[material + 1] - 1):
        left = physical.q[index]
        right = wp.min(physical.q[index + 1], q_max)
        if right > left:
            radius = (right - left) * wp.float64(0.5)
            midpoint = (right + left) * wp.float64(0.5)
            for node in range(4):
                q = midpoint + radius * nodes[node]
                c = wp.float64(1.0) - wp.float64(2.0) * (q / q_max) * (q / q_max)
                factor = atomic_value(q, material, physical, 1)
                integral += radius * weights[node] * q * factor * factor * (wp.float64(1.0) + c * c)
    integral *= wp.float64(4.0) / (q_max * q_max)
    q = q_max * wp.sqrt((wp.float64(1.0) - cosine) * wp.float64(0.5))
    factor = atomic_value(q, material, physical, 1)
    return (wp.float64(1.0) + cosine * cosine) * factor * factor / (wp.float64(2.0) * PI * integral)


@wp.func
def fluorescence_event(
    energy: wp.float64,
    material: int,
    coordinate: wp.vec2d,
    bins: int,
    seed: wp.uint64,
    history: wp.uint64,
    event: wp.uint32,
    physical: PhysicalParameters,
) -> wp.vec4d:
    draws = transport_random4(
        seed,
        history,
        event,
        wp.uint32(2) + wp.uint32(physical.random_branch) * wp.uint32(65536),
        physical.sampling,
    )
    cumulative = wp.float64(0.0)
    result = wp.vec4d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    for line in range(
        physical.fluorescence_offsets[material], physical.fluorescence_offsets[material + 1]
    ):
        if energy >= physical.fluorescence_threshold[line]:
            cumulative += energy_table_value(
                physical.fluorescence_probability, line, coordinate, bins
            )
            if draws[0] < cumulative:
                result = wp.vec4d(
                    wp.float64(2.0) * draws[1] - wp.float64(1.0),
                    wp.float64(2.0) * PI * draws[2],
                    physical.fluorescence_energy[line],
                    wp.float64(0.0),
                )
                break
    return result
