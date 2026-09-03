"""Sparse coalesced original-history log-voxel-density scores and moments."""

# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportPrivateUsage=false
# Warp's typed loop variables are explicit constructors.
import warp as wp

from dpt.kernels.numerics import _product_body, attenuation_state

from . import kernels as transport
from ._constants import EVENT_BUDGET, INPUT_MASK, NUMERICAL_MASK, SOURCE_AMPLITUDE

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})


@wp.func_native(_product_body(("a", "b", "c", "d", "e")))
def product5(
    a: wp.float64, b: wp.float64, c: wp.float64, d: wp.float64, e: wp.float64
) -> wp.float64:
    """Combine original source, amplitude, energy, cotangent and coalesced score."""
    ...


@wp.func_native(_product_body(("a", "b", "c", "d", "e"), attenuated=True))
def attenuated_product5(
    state: wp.vec2d,
    a: wp.float64,
    b: wp.float64,
    c: wp.float64,
    d: wp.float64,
    e: wp.float64,
) -> wp.float64:
    """Apply the log attenuation before one final binary64 rounding."""
    ...


@wp.kernel
def validate_active(
    active: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    voxel = wp.tid()
    flag = active[voxel]
    if flag != 0 and flag != 1:
        wp.atomic_or(status, 0, INPUT_MASK)
    if flag == 1 and voxel_density.shape[0] != 0:
        value = voxel_density[voxel]
        if not wp.isfinite(value) or value <= wp.float64(0.0):
            wp.atomic_or(status, 0, INPUT_MASK)


@wp.kernel
def validate_cotangent(cotangent: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
    if not wp.isfinite(cotangent[wp.tid()]):
        wp.atomic_or(status, 0, INPUT_MASK)


@wp.kernel
def replay_scores(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: transport.Parameters,
    seed: wp.uint64,
    first_history: wp.uint64,
    source_amplitude: wp.float64,
    cotangent: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    capacity: int,
    keys: wp.array(dtype=wp.int32),
    scores: wp.array(dtype=wp.float64),
    values: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    record_status: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    history = wp.tid()
    records = transport.VoxelRecords()
    records.enabled = 1
    records.capacity = capacity
    records.history = history
    records.active = active
    records.voxel = keys
    records.score = scores
    records.status = record_status
    # The primal is unused. A zero source weight preserves terminal path state
    # without prematurely overflowing or underflowing its final score.
    result = transport.walk_recorded(
        positions[history],
        directions[history],
        wp.float64(0.0),
        wp.float64(1.0),
        seed,
        first_history + wp.uint64(history),
        SOURCE_AMPLITUDE,
        wp.vec4i(SOURCE_AMPLITUDE),
        0,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        parameters,
        records,
    )
    if result.status >= EVENT_BUDGET:
        wp.atomic_or(status, 0, 1 << result.status)
    energy = wp.float64(1.0)
    if parameters.energy_score != 0:
        energy = result.energy
    state = attenuation_state(result.absorption_depth)
    for slot in range(capacity):
        index = history * capacity + slot
        voxel = keys[index]
        if voxel >= 0:
            value = wp.float64(0.0)
            if result.pixel >= 0 and result.status < EVENT_BUDGET:
                factor = cotangent[result.pixel]
                if parameters.continuous_absorption != 0:
                    value = attenuated_product5(
                        state, weights[history], source_amplitude, energy, factor, scores[index]
                    )
                else:
                    value = product5(
                        weights[history], source_amplitude, energy, factor, scores[index]
                    )
            values[index] = value
            if not wp.isfinite(value):
                wp.atomic_or(status, 0, NUMERICAL_MASK)
            else:
                wp.atomic_max(scale, voxel, wp.abs(value))


@wp.kernel
def sum_coalesced(
    keys: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    sums: wp.array(dtype=wp.float64),
    hits: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    voxel = keys[index]
    if voxel >= 0:
        wp.atomic_add(hits, voxel, 1)
        if scale[voxel] > wp.float64(0.0):
            value = values[index] / scale[voxel]
            if not wp.isfinite(value) or (
                value == wp.float64(0.0) and values[index] != wp.float64(0.0)
            ):
                # Reject lossy normalisation rather than silently erase a signed
                # term whose contribution may matter after cancellation.
                wp.atomic_or(status, 0, NUMERICAL_MASK)
            else:
                wp.atomic_add(sums, voxel, value)


@wp.kernel
def centred_coalesced(
    keys: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    mean: wp.array(dtype=wp.float64),
    centred: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    voxel = keys[index]
    if voxel >= 0 and scale[voxel] > wp.float64(0.0):
        value = values[index]
        average = mean[voxel]
        residual = wp.float64(0.0)
        residual = transport._centred_residual(value, average, scale, voxel, residual)
        term = residual * residual
        if not wp.isfinite(term):
            wp.atomic_or(status, 0, NUMERICAL_MASK)
        else:
            wp.atomic_add(centred, voxel, term)


@wp.kernel
def finish_variance(
    mean: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    hits: wp.array(dtype=wp.int32),
    centred: wp.array(dtype=wp.float64),
    count: int,
    out_variance: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    voxel = wp.tid()
    missing = count - hits[voxel]
    total = centred[voxel]
    if missing > 0 and scale[voxel] > wp.float64(0.0):
        residual = mean[voxel] / scale[voxel]
        total += wp.float64(missing) * residual * residual
    n = wp.float64(count)
    sigma = scale[voxel] * wp.sqrt(total / n / (n - wp.float64(1.0)))
    variance = sigma * sigma
    out_variance[voxel] = variance
    if missing < 0 or not wp.isfinite(variance):
        wp.atomic_or(status, 0, NUMERICAL_MASK)
