"""Small algebraic reductions around the canonical material and spectral operators."""

from functools import cache

import warp as wp

OPTIONS = {"fast_math": False, "fuse_fp": True, "enable_backward": False}
wp.set_module_options(OPTIONS)


@cache
def get_add_fisher(double: bool = False):
    value_type = wp.float64 if double else wp.float32

    @wp.kernel(module="unique", module_options=OPTIONS)
    def add_fisher(
        mean: wp.array(dtype=value_type),
        path_derivative: wp.array(dtype=value_type),
        pixels: int,
        fisher: wp.array(dtype=wp.vec3d),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        value = wp.float64(mean[pixel])
        first = wp.float64(path_derivative[pixel])
        second = wp.float64(path_derivative[pixels + pixel])
        if value <= wp.float64(0.0) or not wp.isfinite(value):
            wp.atomic_or(status, 0, 2)
        else:
            contribution = wp.vec3d(first * first, first * second, second * second) / value
            total = fisher[pixel] + contribution
            fisher[pixel] = total
            if not wp.isfinite(total[0]) or not wp.isfinite(total[1]) or not wp.isfinite(total[2]):
                wp.atomic_or(status, 0, 2)

    return add_fisher


@cache
def get_fisher_row_seed(double: bool = False):
    value_type = wp.float64 if double else wp.float32

    @wp.kernel(module="unique", module_options=OPTIONS)
    def fisher_row_seed(
        fisher: wp.array(dtype=wp.vec3d),
        path_of_ones: wp.array(dtype=value_type),
        inverse_square_root: wp.vec3d,
        seed: wp.array(dtype=value_type),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        block = fisher[pixel]
        a, b, c = inverse_square_root[0], inverse_square_root[1], inverse_square_root[2]
        first = a * a * block[0] + wp.float64(2.0) * a * b * block[1] + b * b * block[2]
        last = b * b * block[0] + wp.float64(2.0) * b * c * block[1] + c * c * block[2]
        off = a * b * block[0] + (a * c + b * b) * block[1] + b * c * block[2]
        delta = first - last
        scale = wp.max(wp.abs(delta), wp.float64(2.0) * wp.abs(off))
        discriminant = wp.float64(0.0)
        if scale > wp.float64(0.0):
            x = delta / scale
            y = wp.float64(2.0) * off / scale
            discriminant = scale * wp.sqrt(x * x + y * y)
        kappa = wp.float64(0.5) * (first + last + discriminant)
        value = value_type(kappa * wp.float64(path_of_ones[pixel]))
        seed[pixel] = value
        if not wp.isfinite(value) or value < value_type(0.0):
            wp.atomic_or(status, 0, 2)

    return fisher_row_seed


@wp.kernel
def add_regulariser(
    projected_curvature: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    coefficients: wp.vec3d,
    minimum_metric_eigenvalue: wp.float64,
    custom_penalty: bool,
    penalty_curvature: wp.array(dtype=wp.float64),
    raw_curvature: wp.array(dtype=wp.float64),
    extrema: wp.array(dtype=wp.float64),
    counts: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    voxel = wp.tid()
    x, y, z = voxel % nx, (voxel // nx) % ny, voxel // (nx * ny)
    degree = wp.float64(0.0)
    if x > 0:
        degree += coefficients[0]
    if x + 1 < nx:
        degree += coefficients[0]
    if y > 0:
        degree += coefficients[1]
    if y + 1 < ny:
        degree += coefficients[1]
    if z > 0:
        degree += coefficients[2]
    if z + 1 < nz:
        degree += coefficients[2]
    value = wp.float64(projected_curvature[voxel])
    if value == wp.float64(0.0):
        wp.atomic_add(counts, 0, 1)
    value += wp.float64(2.0) * degree / minimum_metric_eigenvalue
    if custom_penalty:
        contribution = penalty_curvature[voxel]
        if not wp.isfinite(contribution) or contribution < wp.float64(0.0):
            wp.atomic_or(status, 0, 2)
            return
        value += contribution
    raw_curvature[voxel] = value
    if not wp.isfinite(value) or value < wp.float64(0.0):
        wp.atomic_or(status, 0, 2)
        return
    wp.atomic_min(extrema, 0, value)
    wp.atomic_max(extrema, 1, value)


@wp.kernel
def floor_curvature(
    raw_curvature: wp.array(dtype=wp.float64),
    floor: wp.float64,
    counts: wp.array(dtype=wp.int32),
):
    voxel = wp.tid()
    value = raw_curvature[voxel]
    if value < floor:
        raw_curvature[voxel] = floor
        wp.atomic_add(counts, 1, 1)


@wp.kernel
def normalise_curvature(
    curvature: wp.array(dtype=wp.float64),
    normaliser: wp.float64,
    scale: wp.array(dtype=wp.float32),
    status: wp.array(dtype=wp.int32),
):
    voxel = wp.tid()
    value = wp.float32(curvature[voxel] / normaliser)
    scale[voxel] = value
    if not wp.isfinite(value) or value <= wp.float32(0.0):
        wp.atomic_or(status, 0, 2)
