"""Finite differences, TV dual projection and L1/support proximal kernels."""

# Warp DSL annotations are runtime expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false

from functools import cache

import warp as wp

from dpt.kernels.numerics import product3

OPTIONS = {"fast_math": False, "fuse_fp": True, "enable_backward": False}


@wp.func_native("""
// Represent each physical forward difference by a bounded mantissa/exponent.
// The exponent range is bounded by the finite binary64 inputs. This is local
// arithmetic, not an extended dual state or a rounded weighted unit vector.
auto split = [](double value, int& exponent) {
    int correction = 0;
    if (wp::abs(value) < 2.2250738585072014e-308) {
        value *= 18014398509481984.0;
        correction = 54;
    }
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    exponent = static_cast<int>((bits >> 52) & 0x7ff) - 1022 - correction;
    bits = (bits & 0x800fffffffffffffull) | 0x3fe0000000000000ull;
    memcpy(&value, &bits, sizeof(value));
    return value;
};
double delta_m[3], inverse_m[3], difference_m[3];
int delta_e[3], inverse_e[3], difference_e[3];
int largest_e = -10000;
for (int axis = 0; axis < 3; ++axis) {
    double delta = right[axis] - left;
    int overflow_shift = 0;
    if (!wp::isfinite(delta)) {
        // Only opposite-sign large finite endpoints can overflow subtraction.
        delta = 0.5 * right[axis] - 0.5 * left;
        overflow_shift = 1;
    }
    delta_m[axis] = 0.0;
    delta_e[axis] = 0;
    difference_m[axis] = 0.0;
    difference_e[axis] = -10000;
    inverse_m[axis] = split(inverse[axis], inverse_e[axis]);
    if (delta != 0.0) {
        delta_m[axis] = split(delta, delta_e[axis]);
        delta_e[axis] += overflow_shift;
        difference_m[axis] = delta_m[axis] * inverse_m[axis];
        difference_e[axis] = delta_e[axis] + inverse_e[axis];
        if (difference_e[axis] > largest_e) largest_e = difference_e[axis];
    }
}
if (largest_e == -10000 || weight == 0.0)
    return wp::vec_t<4, double>(0.0, 0.0, 0.0, 0.0);
double scaled_norm = 0.0;
for (int axis = 0; axis < 3; ++axis) {
    const double component = ldexp(difference_m[axis], difference_e[axis] - largest_e);
    scaled_norm += isotropic ? component * component : wp::abs(component);
}
if (isotropic) scaled_norm = sqrt(scaled_norm);
int weight_e;
const double weight_m = split(weight, weight_e);
const double value = ldexp(weight_m * scaled_norm, weight_e + largest_e);
double edge[3] = {0.0, 0.0, 0.0};
for (int axis = 0; axis < 3; ++axis) {
    if (difference_m[axis] != 0.0) {
        if (isotropic) {
            const double mantissa = difference_m[axis] * weight_m * inverse_m[axis] / scaled_norm;
            edge[axis] = ldexp(mantissa,
                difference_e[axis] + weight_e + inverse_e[axis] - largest_e);
        } else {
            const double mantissa = (difference_m[axis] > 0.0 ? 1.0 : -1.0)
                * weight_m * inverse_m[axis];
            edge[axis] = ldexp(mantissa, weight_e + inverse_e[axis]);
        }
    }
}
return wp::vec_t<4, double>(value, edge[0], edge[1], edge[2]);
""")
def tv_value_and_edges(
    left: wp.float64, right: wp.vec3d, inverse: wp.vec3d, weight: wp.float64, isotropic: bool
) -> wp.vec4d:
    """Return lambda*phi(Dx) and lambda*dphi(Dx)/h with one final rounding."""
    ...


@wp.func
def forward_neighbours(
    values: wp.array(dtype=wp.float64), index: int, nx: int, ny: int, nz: int
) -> wp.vec3d:
    x = index % nx
    y = (index // nx) % ny
    z = index // (nx * ny)
    result = wp.vec3d(values[index], values[index], values[index])
    if x + 1 < nx:
        result[0] = values[index + 1]
    if y + 1 < ny:
        result[1] = values[index + nx]
    if z + 1 < nz:
        result[2] = values[index + nx * ny]
    return result


@wp.func
def transpose_edges(
    edges: wp.array(dtype=wp.vec3d), index: int, nx: int, ny: int, nz: int
) -> wp.float64:
    x = index % nx
    y = (index // nx) % ny
    z = index // (nx * ny)
    result = wp.float64(0.0)
    if x + 1 < nx:
        result -= edges[index][0]
    if x > 0:
        result += edges[index - 1][0]
    if y + 1 < ny:
        result -= edges[index][1]
    if y > 0:
        result += edges[index - nx][1]
    if z + 1 < nz:
        result -= edges[index][2]
    if z > 0:
        result += edges[index - nx * ny][2]
    return result


@wp.func
def scaled_pair(
    left: wp.float64, right: wp.float64, factor: wp.float64, inverse: wp.float64
) -> wp.float64:
    difference = right - left
    if wp.isfinite(difference):
        return product3(difference, factor, inverse)
    # Overflowing subtraction implies opposite large signs. If the weighted
    # final difference is representable, both weighted endpoints are too.
    return product3(right, factor, inverse) - product3(left, factor, inverse)


@wp.func
def forward_difference(
    values: wp.array(dtype=wp.float64),
    index: int,
    nx: int,
    ny: int,
    nz: int,
    inverse: wp.vec3d,
    factor: wp.float64,
) -> wp.vec3d:
    x = index % nx
    y = (index // nx) % ny
    z = index // (nx * ny)
    result = wp.vec3d(0.0, 0.0, 0.0)
    if x + 1 < nx:
        result[0] = scaled_pair(values[index], values[index + 1], factor, inverse[0])
    if y + 1 < ny:
        result[1] = scaled_pair(values[index], values[index + nx], factor, inverse[1])
    if z + 1 < nz:
        result[2] = scaled_pair(values[index], values[index + nx * ny], factor, inverse[2])
    return result


@wp.func
def transpose_difference(
    dual: wp.array(dtype=wp.vec3d), index: int, nx: int, ny: int, nz: int, inverse: wp.vec3d
) -> wp.float64:
    x = index % nx
    y = (index // nx) % ny
    z = index // (nx * ny)
    result = wp.float64(0.0)
    if x + 1 < nx:
        result -= dual[index][0] * inverse[0]
    if x > 0:
        result += dual[index - 1][0] * inverse[0]
    if y + 1 < ny:
        result -= dual[index][1] * inverse[1]
    if y > 0:
        result += dual[index - nx][1] * inverse[1]
    if z + 1 < nz:
        result -= dual[index][2] * inverse[2]
    if z > 0:
        result += dual[index - nx * ny][2] * inverse[2]
    return result


@wp.func
def shrink(value: wp.float64, threshold: wp.float64) -> wp.float64:
    return wp.sign(value) * wp.max(wp.abs(value) - threshold, wp.float64(0.0))


@wp.kernel(module="unique", module_options=OPTIONS)
def validate_support(mask: wp.array(dtype=wp.uint8), status: wp.array(dtype=wp.int32)):
    if mask[wp.tid()] > wp.uint8(1):
        wp.atomic_or(status, 0, 1)


@cache
def initialise_kernel(binary64: bool):
    dtype = wp.float64 if binary64 else wp.float32

    @wp.kernel(module="unique", module_options=OPTIONS)
    def initialise(
        source: wp.array(dtype=dtype),
        mask: wp.array(dtype=wp.uint8),
        l1: wp.float64,
        enforce: bool,
        y: wp.array(dtype=wp.float64),
        x: wp.array(dtype=wp.float64),
        extrapolated: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        value = wp.float64(source[i])
        if not wp.isfinite(value):
            wp.atomic_or(status, 0, 1)
        y[i] = value
        result = shrink(value, l1)
        if mask[i] == wp.uint8(0):
            if enforce and value != wp.float64(0.0):
                wp.atomic_or(status, 0, 8)
            result = wp.float64(0.0)
        x[i] = result
        extrapolated[i] = result

    return initialise


@cache
def penalty_kernel(isotropic: bool):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def penalty(
        values: wp.array(dtype=wp.float64),
        nx: int,
        ny: int,
        nz: int,
        inverse: wp.vec3d,
        tv: wp.float64,
        l1: wp.float64,
        dual: wp.array(dtype=wp.vec3d),
        contributions: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        magnitude = wp.float64(0.0)
        tangent = wp.vec3d(0.0, 0.0, 0.0)
        if tv > wp.float64(0.0):
            evaluated = tv_value_and_edges(
                values[i],
                forward_neighbours(values, i, nx, ny, nz),
                inverse,
                tv,
                wp.static(isotropic),
            )
            magnitude = evaluated[0]
            tangent = wp.vec3d(evaluated[1], evaluated[2], evaluated[3])
        # During penalty evaluation this private buffer contains spacing-aware
        # edge derivatives. Proximal execution resets it to its actual dual.
        dual[i] = tangent
        value = magnitude + l1 * wp.abs(values[i])
        contributions[2 * i] = value
        contributions[2 * i + 1] = wp.float64(0.0)
        if (
            not wp.isfinite(value)
            or not wp.isfinite(tangent[0])
            or not wp.isfinite(tangent[1])
            or not wp.isfinite(tangent[2])
        ):
            wp.atomic_or(status, 0, 2)

    return penalty


@cache
def gradient_kernel(binary64: bool):
    dtype = wp.float64 if binary64 else wp.float32

    @wp.kernel(module="unique", module_options=OPTIONS)
    def gradient(
        values: wp.array(dtype=wp.float64),
        mask: wp.array(dtype=wp.uint8),
        dual: wp.array(dtype=wp.vec3d),
        nx: int,
        ny: int,
        nz: int,
        l1: wp.float64,
        destination: wp.array(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        value = wp.float64(0.0)
        if mask[i] != wp.uint8(0):
            value = transpose_edges(dual, i, nx, ny, nz) + l1 * wp.sign(values[i])
        result = dtype(value)
        destination[i] = result
        if not wp.isfinite(result):
            wp.atomic_or(status, 0, 2)

    return gradient


@cache
def dual_kernel(isotropic: bool):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def update(
        extrapolated: wp.array(dtype=wp.float64),
        nx: int,
        ny: int,
        nz: int,
        inverse: wp.vec3d,
        sigma: wp.float64,
        radius: wp.float64,
        dual: wp.array(dtype=wp.vec3d),
        status: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        value = dual[i] + forward_difference(extrapolated, i, nx, ny, nz, inverse, sigma)
        if not wp.isfinite(value[0]) or not wp.isfinite(value[1]) or not wp.isfinite(value[2]):
            wp.atomic_or(status, 0, 2)
        if wp.static(isotropic):
            largest = wp.max(wp.abs(value[0]), wp.max(wp.abs(value[1]), wp.abs(value[2])))
            # Normalise before multiplying by radius: radius/norm may underflow
            # even when every projected component is representable.
            safe_radius = radius * wp.float64(0.9999999999999982)
            if largest > wp.float64(0.0):
                scaled = value / largest
                small_norm = wp.sqrt(wp.dot(scaled, scaled))
                if largest > safe_radius / small_norm:
                    value = scaled * (safe_radius / small_norm)
        else:
            for axis in range(3):
                value[axis] = wp.clamp(value[axis], -radius, radius)
        dual[i] = value

    return update


@wp.kernel(module="unique", module_options=OPTIONS)
def primal_kernel(
    source: wp.array(dtype=wp.float64),
    previous: wp.array(dtype=wp.float64),
    mask: wp.array(dtype=wp.uint8),
    dual: wp.array(dtype=wp.vec3d),
    nx: int,
    ny: int,
    nz: int,
    inverse: wp.vec3d,
    primal_step: wp.float64,
    l1: wp.float64,
    following: wp.array(dtype=wp.float64),
    extrapolated: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    ratio = primal_step / (wp.float64(1.0) + primal_step)
    # Incremental form preserves an unchanged large primal value when its
    # regularization correction lies below one representable increment.
    correction = source[i] - previous[i] - transpose_difference(dual, i, nx, ny, nz, inverse)
    value = previous[i] + ratio * correction
    result = shrink(value, ratio * l1)
    if mask[i] == wp.uint8(0):
        result = wp.float64(0.0)
    following[i] = result
    extrapolated[i] = result + (result - previous[i])
    if not wp.isfinite(result) or not wp.isfinite(extrapolated[i]):
        wp.atomic_or(status, 0, 2)


@cache
def certificate_kernel(isotropic: bool):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def certificate(
        values: wp.array(dtype=wp.float64),
        source: wp.array(dtype=wp.float64),
        mask: wp.array(dtype=wp.uint8),
        dual: wp.array(dtype=wp.vec3d),
        nx: int,
        ny: int,
        nz: int,
        inverse: wp.vec3d,
        tv: wp.float64,
        l1: wp.float64,
        contributions: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        value, y = values[i], source[i]
        magnitude = wp.float64(0.0)
        if tv > wp.float64(0.0):
            evaluated = tv_value_and_edges(
                value, forward_neighbours(values, i, nx, ny, nz), inverse, tv, wp.static(isotropic)
            )
            magnitude = evaluated[0]
        residual = value - y
        primal = (wp.float64(0.5) * residual) * residual + l1 * wp.abs(value) + magnitude
        z = -transpose_difference(dual, i, nx, ny, nz, inverse)
        candidate = y + z
        bound = (wp.float64(0.5) * y) * y
        if mask[i] != wp.uint8(0) and wp.abs(candidate) > l1:
            # Expanded conjugate avoids subtracting two nearly equal y-squares.
            displacement = z - wp.sign(candidate) * l1
            bound = -displacement * (y + wp.float64(0.5) * displacement)
        contributions[2 * i] = primal
        contributions[2 * i + 1] = bound
        if not wp.isfinite(primal) or not wp.isfinite(bound) or not wp.isfinite(candidate):
            wp.atomic_or(status, 0, 2)

    return certificate


@cache
def write_kernel(binary64: bool):
    dtype = wp.float64 if binary64 else wp.float32

    @wp.kernel(module="unique", module_options=OPTIONS)
    def write(
        source: wp.array(dtype=wp.float64),
        destination: wp.array(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        value = dtype(source[i])
        destination[i] = value
        if not wp.isfinite(value):
            wp.atomic_or(status, 0, 2)

    return write
