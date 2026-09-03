"""Expected-score coefficient estimators on the canonical photon path."""

import warp as wp

from dpt.kernels.numerics import product3

from ._constants import ABSORBED, ESCAPED, NUMERICAL_FAILURE
from .kernels import CoefficientRecords, Parameters, walk, walk_coefficient

wp.set_module_options({"enable_backward": False})


@wp.kernel
def coefficient_derivative_histories(
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
    source_amplitude: wp.float64,
    active: CoefficientRecords,
    out_pixel: wp.array(dtype=wp.int32),
    out_score: wp.array(dtype=wp.float64),
    out_likelihood: wp.array(dtype=wp.float64),
    out_pathwise: wp.array(dtype=wp.float64),
    out_status: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    h = wp.tid()
    trace = walk_coefficient(
        positions[h],
        directions[h],
        weights[h],
        source_amplitude,
        seed,
        first_history + wp.uint64(h),
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        p,
        active,
    )
    likelihood = product3(trace.score, trace.coefficient_likelihood, wp.float64(1.0))
    pathwise = product3(trace.score, trace.coefficient_pathwise, wp.float64(1.0))
    if not wp.isfinite(likelihood) or not wp.isfinite(pathwise):
        trace.status = NUMERICAL_FAILURE
    out_pixel[h] = trace.pixel
    out_score[h] = trace.score
    out_likelihood[h] = likelihood
    out_pathwise[h] = pathwise
    out_status[h] = trace.status
    if trace.status != ESCAPED and trace.status != ABSORBED:
        wp.atomic_or(status, 0, 1 << trace.status)


@wp.kernel
def spectrum_derivative_histories(
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
    active: int,
    probability: wp.float64,
    out_pixel: wp.array(dtype=wp.int32),
    out_score: wp.array(dtype=wp.float64),
    out_derivative: wp.array(dtype=wp.float64),
    out_status: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    h = wp.tid()
    result = walk(
        positions[h],
        directions[h],
        weights[h],
        amplitude,
        seed,
        first_history + wp.uint64(h),
        -1,
        wp.vec4i(-1),
        0,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        p,
    )
    derivative = result.score * (wp.float64(int(result.spectrum_index == active)) - probability)
    if not wp.isfinite(derivative):
        result.status = NUMERICAL_FAILURE
    out_pixel[h] = result.pixel
    out_score[h] = result.score
    out_derivative[h] = derivative
    out_status[h] = result.status
    if result.status != ESCAPED and result.status != ABSORBED:
        wp.atomic_or(status, 0, 1 << result.status)
