"""Replicate-axis transport checkpoints using the canonical complete-history walk.

History buffers have layout (replicate, history); derivative columns prepend a
column axis. Detector buffers have layout (replicate, pixel). Each replicate
retains its own original-history denominator and random identity range.
"""

# Warp annotations are executable DSL forms, as in the canonical kernel module.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false
import warp as wp

from dpt.kernels.numerics import product3, product4
from dpt.kernels.random import random4

from ._constants import (
    EVENT_BUDGET,
    INPUT_MASK,
    NUMERICAL_MASK,
    RANDOM_NAMESPACE_BIT,
    SOURCE_AMPLITUDE,
)
from .kernels import (
    Parameters,
    _density_derivative_contribution,
    _derivative_factors,
    _normalised_score_contribution,
    valid_history,
    walk,
)

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})
TILE = 256


@wp.struct
class SourceParameters:
    kind: int
    lower: wp.vec3d
    position: wp.vec3d
    direction: wp.vec3d
    u: wp.vec3d
    v: wp.vec3d
    extent: wp.vec2d
    weight: wp.float64


@wp.kernel
def sample_sources(
    source: SourceParameters,
    seeds: wp.array(dtype=wp.uint64),
    first: wp.array(dtype=wp.uint64),
    count: int,
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
):
    flat = wp.tid()
    replicate = flat // count
    history = flat % count
    draw = random4(
        seeds[replicate],
        first[replicate] + wp.uint64(history),
        wp.uint32(RANDOM_NAMESPACE_BIT),
        wp.uint32(0),
    )
    if source.kind == 0:
        positions[flat] = source.lower + wp.vec3d(
            source.extent[0] * draw[0], source.extent[1] * draw[1], wp.float64(0.0)
        )
        directions[flat] = source.direction
    else:
        endpoint = (
            source.lower
            + source.extent[0] * draw[0] * source.u
            + source.extent[1] * draw[1] * source.v
        )
        delta = endpoint - source.position
        scale = wp.max(wp.abs(delta[0]), wp.max(wp.abs(delta[1]), wp.abs(delta[2])))
        positions[flat] = source.position
        directions[flat] = wp.normalize(delta / scale)
    weights[flat] = source.weight


@wp.kernel
def trace_batched(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: Parameters,
    seeds: wp.array(dtype=wp.uint64),
    first: wp.array(dtype=wp.uint64),
    count: int,
    source_amplitude: wp.float64,
    active_materials: wp.vec4i,
    active_count: int,
    emit_primal: int,
    out_pixel: wp.array(dtype=wp.int32),
    out_score: wp.array(dtype=wp.float64),
    out_derivatives: wp.array(dtype=wp.float64),
    out_history_status: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    flat = wp.tid()
    replicate, history = flat // count, flat % count
    base_weight = wp.float64(0.0)
    amplitude = wp.float64(1.0)
    if emit_primal != 0:
        base_weight = weights[flat]
        amplitude = source_amplitude
    result = walk(
        positions[flat],
        directions[flat],
        base_weight,
        amplitude,
        seeds[replicate],
        first[replicate] + wp.uint64(history),
        SOURCE_AMPLITUDE,
        active_materials,
        active_count,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        parameters,
    )
    out_pixel[flat] = result.pixel
    out_history_status[flat] = result.status
    out_score[flat] = result.score
    if result.status >= EVENT_BUDGET:
        wp.atomic_or(status, 0, 1 << result.status)
    if not wp.isfinite(result.score):
        wp.atomic_or(status, 0, NUMERICAL_MASK)
    stride = seeds.shape[0] * count
    for column in range(active_count):
        derivative = wp.float64(0.0)
        if result.pixel >= 0:
            energy = wp.float64(1.0)
            if parameters.energy_score != 0:
                energy = result.energy
            factor, derivative_amplitude = _derivative_factors(
                result.density_scores[column], active_materials[column], source_amplitude
            )
            derivative = _density_derivative_contribution(
                weights[flat],
                derivative_amplitude,
                energy,
                factor,
                result.absorption_depth,
                parameters.continuous_absorption,
            )
        out_derivatives[column * stride + flat] = derivative
        if not wp.isfinite(derivative):
            wp.atomic_or(status, 0, NUMERICAL_MASK)


@wp.kernel
def find_tally_scale(
    pixel: wp.array(dtype=wp.int32),
    score: wp.array(dtype=wp.float64),
    history_status: wp.array(dtype=wp.int32),
    pixels: int,
    count: int,
    out_scale: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    flat = wp.tid()
    if valid_history(pixel[flat], score[flat], history_status[flat], pixels, status):
        target = (flat // count) * pixels + pixel[flat]
        wp.atomic_max(out_scale, target, wp.abs(score[flat]))


@wp.kernel
def tally_sum(
    pixel: wp.array(dtype=wp.int32),
    score: wp.array(dtype=wp.float64),
    history_status: wp.array(dtype=wp.int32),
    pixels: int,
    count: int,
    scale: wp.array(dtype=wp.float64),
    out_sum: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    flat = wp.tid()
    if valid_history(pixel[flat], score[flat], history_status[flat], pixels, status):
        target = (flat // count) * pixels + pixel[flat]
        if scale[target] > wp.float64(0.0):
            contribution, participating = _normalised_score_contribution(
                flat, target, score, scale, status
            )
            if participating != 0:
                wp.atomic_add(out_sum, target, contribution)


@wp.kernel
def finish_mean(
    sums: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    count: int,
    out_mean: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    flat = wp.tid()
    value = scale[flat] * (sums[flat] / wp.float64(count))
    out_mean[flat] = value
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, NUMERICAL_MASK)


@wp.func
def objective_value(
    a: wp.float64,
    b: wp.float64,
    y: wp.float64,
    weight: wp.float64,
    loss_mode: int,
    status: wp.array(dtype=wp.int32),
) -> wp.float64:
    if not wp.isfinite(a) or not wp.isfinite(b) or not wp.isfinite(y) or not wp.isfinite(weight):
        wp.atomic_or(status, 0, NUMERICAL_MASK)
    if weight < wp.float64(0.0):
        wp.atomic_or(status, 0, INPUT_MASK)
    value = product3(weight, a - y, b)
    if loss_mode != 0:
        value = product4(wp.float64(0.5), weight, a - y, b - y)
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, NUMERICAL_MASK)
    return value


@wp.kernel
def objective_tiles(
    mean_a: wp.array(dtype=wp.float64),
    other_b: wp.array(dtype=wp.float64),
    observed: wp.array(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    pixels: int,
    tiles: int,
    loss_mode: int,
    partial: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    block, lane = wp.tid()
    replicate, tile = block // tiles, block % tiles
    value = wp.float64(0.0)
    if wp.block_dim() == 1:
        for offset in range(TILE):
            pixel = tile * TILE + offset
            if pixel < pixels:
                flat = replicate * pixels + pixel
                value += objective_value(
                    mean_a[flat], other_b[flat], observed[pixel], weights[pixel], loss_mode, status
                )
    else:
        pixel = tile * TILE + lane
        if pixel < pixels:
            flat = replicate * pixels + pixel
            value = objective_value(
                mean_a[flat], other_b[flat], observed[pixel], weights[pixel], loss_mode, status
            )
    total = wp.tile_sum(wp.tile(value))
    wp.tile_store(partial, total, offset=block)


@wp.kernel
def finish_values(
    source: wp.array(dtype=wp.float64),
    stride: int,
    column: int,
    destination: wp.array(dtype=wp.float64),
):
    replicate = wp.tid()
    destination[replicate * stride + column] = source[replicate]


@wp.kernel
def combine_values(
    previous: wp.array(dtype=wp.float64),
    current: wp.array(dtype=wp.float64),
    average: int,
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    value = current[index] - previous[index]
    if average != 0:
        value = current[index] / wp.float64(2.0) + previous[index] / wp.float64(2.0)
    current[index] = value
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, NUMERICAL_MASK)
