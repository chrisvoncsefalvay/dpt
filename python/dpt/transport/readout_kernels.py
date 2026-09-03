"""Complete original-history readout and fixed-order reductions."""

import warp as wp

from ._constants import ABSORBED, ESCAPED, NUMERICAL_FAILURE, ROULETTE_TERMINATED
from .kernels import Parameters, ReadoutRecords, walk_readout

wp.set_module_options({"enable_backward": False})


@wp.kernel
def trace_readout(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    p: Parameters,
    seed: wp.uint64,
    first_history: wp.uint64,
    amplitude: wp.float64,
    branches: int,
    readout_parameters: ReadoutRecords,
    history_status: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    h = wp.tid()
    records = readout_parameters
    records.history = h
    history_status[h] = ESCAPED
    split_weight = weights[h] / wp.float64(branches)
    if weights[h] != wp.float64(0.0) and split_weight == wp.float64(0.0):
        history_status[h] = NUMERICAL_FAILURE
        wp.atomic_or(status, 0, 1 << NUMERICAL_FAILURE)
        return
    for branch in range(branches):
        records.branch = branch
        result = walk_readout(
            positions[h],
            directions[h],
            split_weight,
            amplitude,
            seed,
            first_history + wp.uint64(h),
            material_ids,
            voxel_density,
            energies,
            absorption,
            scattering,
            density,
            p,
            records,
        )
        if result.status == ROULETTE_TERMINATED:
            history_status[h] = ROULETTE_TERMINATED
        if (
            result.status != ESCAPED
            and result.status != ABSORBED
            and result.status != ROULETTE_TERMINATED
        ):
            history_status[h] = result.status
            wp.atomic_or(status, 0, 1 << result.status)


@wp.kernel
def readout_plain(
    scores: wp.array(dtype=wp.float64),
    count: int,
    columns: int,
    mean: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    value = scores[index] / wp.float64(count)
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, 1 << NUMERICAL_FAILURE)
    else:
        wp.atomic_add(mean, index % columns, value)


@wp.kernel
def readout_deterministic(
    scores: wp.array(dtype=wp.float64),
    count: int,
    columns: int,
    mean: wp.array(dtype=wp.float64),
    variance: wp.array(dtype=wp.float64),
    with_variance: int,
    write_mean: int,
    status: wp.array(dtype=wp.int32),
):
    column = wp.tid()
    # Kahan summation in increasing original-history order is independent of
    # CUDA scheduling, splitting order and detector atomic occupancy.
    total = wp.float64(0.0)
    error = wp.float64(0.0)
    for h in range(count):
        value = scores[h * columns + column] / wp.float64(count) - error
        updated = total + value
        error = (updated - total) - value
        total = updated
    if write_mean != 0:
        mean[column] = total
    if with_variance != 0:
        moment = wp.float64(0.0)
        correction = wp.float64(0.0)
        for h in range(count):
            centred = (scores[h * columns + column] - total) / wp.sqrt(
                wp.float64(count) * wp.float64(count - 1)
            )
            value = centred * centred - correction
            updated = moment + value
            correction = (updated - moment) - value
            moment = updated
        variance[column] = moment
        if not wp.isfinite(moment):
            wp.atomic_or(status, 0, 1 << NUMERICAL_FAILURE)
    if not wp.isfinite(total):
        wp.atomic_or(status, 0, 1 << NUMERICAL_FAILURE)
