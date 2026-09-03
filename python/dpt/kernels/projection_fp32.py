"""FP32 midpoint projection with matched volume and geometric derivatives.

Physical calibration and pose are transformed to volume-local coordinates in
FP64 once per ray. Clipping, interpolation, slopes, physical steps and serial
sums then use FP32. The pose chain rule and final reduction use FP64 outside
the sample loop. An optional FP64 volume destination only widens each completed
FP32 scatter increment at its atomic reduction boundary.
"""

# Warp annotations are executable DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false

from functools import cache

import warp as wp

from dpt.kernels.geometry import _local_right_pose_adjoint, unpack_rotation
from dpt.kernels.projection import Configuration, corner_index, ray_for_pixel, valid_ray

OPTIONS = {"fast_math": False, "fuse_fp": True, "enable_backward": False}
wp.set_module_options(OPTIONS)
Vec12d = wp.types.vector(length=12, dtype=wp.float64)
Vec8f = wp.types.vector(length=8, dtype=wp.float32)


@wp.struct
class Ray32:
    origin: wp.vec3
    direction: wp.vec3
    length: wp.float32


@wp.struct
class Interval32:
    lower: wp.float32
    upper: wp.float32
    lower_origin_gradient: wp.vec3
    lower_direction_gradient: wp.vec3
    upper_origin_gradient: wp.vec3
    upper_direction_gradient: wp.vec3


@wp.struct
class Sample32:
    value: wp.float32
    gradient: wp.vec3


@wp.struct
class Coordinates32:
    lower: wp.vec3i
    fraction: wp.vec3
    slope: wp.vec3


@wp.func
def finite_point(point: wp.vec3) -> bool:
    return wp.isfinite(point[0]) and wp.isfinite(point[1]) and wp.isfinite(point[2])


@wp.func
def inside_support(point: wp.vec3, shape: wp.vec3i) -> bool:
    return (
        finite_point(point)
        and point[0] >= wp.float32(-0.5)
        and point[1] >= wp.float32(-0.5)
        and point[2] >= wp.float32(-0.5)
        and point[0] <= wp.float32(shape[0]) - wp.float32(0.5)
        and point[1] <= wp.float32(shape[1]) - wp.float32(0.5)
        and point[2] <= wp.float32(shape[2]) - wp.float32(0.5)
    )


@wp.func
def finite_interval(origin: wp.vec3, direction: wp.vec3, shape: wp.vec3i) -> Interval32:
    """Clip the same finite segment and retain the selected face derivatives."""
    interval = Interval32()
    interval.lower = wp.float32(0.0)
    interval.upper = wp.float32(1.0)
    for axis in range(3):
        low = wp.float32(-0.5)
        high = wp.float32(shape[axis]) - wp.float32(0.5)
        velocity = direction[axis]
        if velocity == wp.float32(0.0):
            if origin[axis] < low or origin[axis] > high:
                interval.upper = wp.float32(-1.0)
        else:
            first = (low - origin[axis]) / velocity
            last = (high - origin[axis]) / velocity
            if first > last:
                temporary = first
                first = last
                last = temporary
            if first > interval.lower:
                interval.lower = first
                interval.lower_origin_gradient = wp.vec3()
                interval.lower_direction_gradient = wp.vec3()
                interval.lower_origin_gradient[axis] = -wp.float32(1.0) / velocity
                interval.lower_direction_gradient[axis] = -first / velocity
            if last < interval.upper:
                interval.upper = last
                interval.upper_origin_gradient = wp.vec3()
                interval.upper_direction_gradient = wp.vec3()
                interval.upper_origin_gradient[axis] = -wp.float32(1.0) / velocity
                interval.upper_direction_gradient[axis] = -last / velocity
    return interval


@wp.func_native("return fmaf(a, b, c);")
def fma(a: wp.float32, b: wp.float32, c: wp.float32) -> wp.float32: ...


@wp.func
def sample_position(lower: wp.float32, step: wp.float32, index: int) -> wp.float32:
    return fma(wp.float32(index) + wp.float32(0.5), step, lower)


@wp.func
def sample_point(origin: wp.vec3, direction: wp.vec3, position: wp.float32) -> wp.vec3:
    return wp.vec3(
        fma(position, direction[0], origin[0]),
        fma(position, direction[1], origin[1]),
        fma(position, direction[2], origin[2]),
    )


@wp.func
def coordinates(point: wp.vec3, shape: wp.vec3i) -> Coordinates32:
    result = Coordinates32()
    for axis in range(3):
        high = wp.float32(shape[axis] - 1)
        value = wp.clamp(point[axis], wp.float32(0.0), high)
        result.lower[axis] = wp.min(int(wp.floor(value)), wp.max(shape[axis] - 2, 0))
        result.fraction[axis] = value - wp.float32(result.lower[axis])
        if point[axis] >= wp.float32(0.0) and point[axis] < high:
            result.slope[axis] = wp.float32(1.0)
    return result


@wp.func
def corner_factors(fraction: wp.vec3, corner: int) -> wp.vec3:
    result = fraction
    for axis in range(3):
        if ((corner >> axis) & 1) == 0:
            result[axis] = wp.float32(1.0) - fraction[axis]
    return result


@wp.func
def corner_weight(factors: wp.vec3) -> wp.float32:
    return (factors[0] * factors[1]) * factors[2]


@wp.func
def sample_field(
    field: wp.array(dtype=wp.float32), point: wp.vec3, shape: wp.vec3i, field_offset: int = 0
) -> Sample32:
    result = Sample32()
    if not inside_support(point, shape):
        return result
    cell = coordinates(point, shape)
    for corner in range(8):
        value = field[field_offset + corner_index(cell.lower, shape, corner)]
        factors = corner_factors(cell.fraction, corner)
        result.value = fma(corner_weight(factors), value, result.value)
        for axis in range(3):
            derivative = cell.slope[axis]
            if ((corner >> axis) & 1) == 0:
                derivative = -derivative
            for other in range(3):
                if other != axis:
                    derivative *= factors[other]
            result.gradient[axis] = fma(derivative, value, result.gradient[axis])
    return result


@wp.func
def sample_pose_field(
    field: wp.array(dtype=wp.float32), point: wp.vec3, shape: wp.vec3i
) -> Sample32:
    """Keep scalar values unchanged while interpolating adjacent-corner slopes."""
    result = Sample32()
    if not inside_support(point, shape):
        return result
    cell = coordinates(point, shape)
    values = Vec8f()
    for corner in range(8):
        value = field[corner_index(cell.lower, shape, corner)]
        values[corner] = value
        factors = corner_factors(cell.fraction, corner)
        result.value = fma(corner_weight(factors), value, result.value)

    fx, fy, fz = cell.fraction[0], cell.fraction[1], cell.fraction[2]
    ax, ay, az = wp.float32(1.0) - fx, wp.float32(1.0) - fy, wp.float32(1.0) - fz
    # Form differences before interpolation to avoid an orthogonal background.
    # Each axis is completed before forming the next pair of intermediates.
    x0 = fma(fy, values[3] - values[2], ay * (values[1] - values[0]))
    x1 = fma(fy, values[7] - values[6], ay * (values[5] - values[4]))
    result.gradient[0] = cell.slope[0] * fma(fz, x1, az * x0)
    y0 = fma(fx, values[3] - values[1], ax * (values[2] - values[0]))
    y1 = fma(fx, values[7] - values[5], ax * (values[6] - values[4]))
    result.gradient[1] = cell.slope[1] * fma(fz, y1, az * y0)
    z0 = fma(fx, values[5] - values[1], ax * (values[4] - values[0]))
    z1 = fma(fx, values[7] - values[3], ax * (values[6] - values[2]))
    result.gradient[2] = cell.slope[2] * fma(fy, z1, ay * z0)
    return result


@cache
def get_scatter_field(double_output: bool = False):
    dtype = wp.float64 if double_output else wp.float32

    @wp.func
    def scatter(
        gradient: wp.array(dtype=dtype),
        point: wp.vec3,
        shape: wp.vec3i,
        factor: wp.float32,
        status: wp.array(dtype=wp.int32),
        field_offset: int = 0,
    ):
        if not inside_support(point, shape):
            return
        cell = coordinates(point, shape)
        for corner in range(8):
            weight = corner_weight(corner_factors(cell.fraction, corner))
            increment = factor * weight
            if not wp.isfinite(increment):
                wp.atomic_or(status, 0, 2)
            wp.atomic_add(
                gradient, field_offset + corner_index(cell.lower, shape, corner), dtype(increment)
            )

    return scatter


@wp.func_native("""
#if defined(__CUDA_ARCH__)
const float increment = __fmul_rn(factor, weight);
if (fabsf(increment) < 0x1p-126f) return copysignf(0.0f, increment);
return increment;
#else
volatile float increment = factor * weight;
return increment;
#endif
""")
def scatter_increment(factor: wp.float32, weight: wp.float32) -> wp.float32:
    """Retain product rounding and CUDA global-atomic subnormal input handling."""
    ...


@wp.func
def bounded_scatter_increment(factor: wp.float32, weight: wp.float32) -> wp.float32:
    """Round before flushing subnormal outputs, including the normal boundary."""
    return scatter_increment(factor, weight)


@wp.func
def flush_cell(
    gradient: wp.array(dtype=wp.float32),
    lower: wp.vec3i,
    shape: wp.vec3i,
    sums: Vec8f,
):
    for corner in range(8):
        wp.atomic_add(gradient, corner_index(lower, shape, corner), sums[corner])


@cache
def get_face_shift(axis: int, positive: bool):
    """Keep static corner indices when carrying a shared face into the next cell."""
    axis_bit = 1 << axis
    shared_bit = 1 if positive else 0

    @wp.func
    def shift(
        gradient: wp.array(dtype=wp.float32),
        lower: wp.vec3i,
        shape: wp.vec3i,
        sums: Vec8f,
    ) -> Vec8f:
        carried = Vec8f()
        for corner in range(8):
            if ((corner >> axis) & 1) == shared_bit:
                carried[corner ^ axis_bit] = sums[corner]
            else:
                wp.atomic_add(gradient, corner_index(lower, shape, corner), sums[corner])
        return carried

    return shift


_shift_x_positive = get_face_shift(0, True)
_shift_x_negative = get_face_shift(0, False)
_shift_y_positive = get_face_shift(1, True)
_shift_y_negative = get_face_shift(1, False)
_shift_z_positive = get_face_shift(2, True)
_shift_z_negative = get_face_shift(2, False)


@wp.func
def transition_cell(
    gradient: wp.array(dtype=wp.float32),
    lower: wp.vec3i,
    following: wp.vec3i,
    shape: wp.vec3i,
    sums: Vec8f,
) -> Vec8f:
    """Flush departing corners or retain the original full-flush fallback."""
    if shape[0] > 1 and shape[1] > 1 and shape[2] > 1:
        delta = following - lower
        if delta[1] == 0 and delta[2] == 0:
            if delta[0] == 1:
                return _shift_x_positive(gradient, lower, shape, sums)
            if delta[0] == -1:
                return _shift_x_negative(gradient, lower, shape, sums)
        if delta[0] == 0 and delta[2] == 0:
            if delta[1] == 1:
                return _shift_y_positive(gradient, lower, shape, sums)
            if delta[1] == -1:
                return _shift_y_negative(gradient, lower, shape, sums)
        if delta[0] == 0 and delta[1] == 0:
            if delta[2] == 1:
                return _shift_z_positive(gradient, lower, shape, sums)
            if delta[2] == -1:
                return _shift_z_negative(gradient, lower, shape, sums)
    flush_cell(gradient, lower, shape, sums)
    return Vec8f()


@cache
def get_forward(double_output: bool = False):
    dtype = wp.float64 if double_output else wp.float32

    @wp.kernel(module="unique", module_options=OPTIONS)
    def forward(
        field: wp.array(dtype=wp.float32),
        pose: wp.array(dtype=wp.float64),
        config: Configuration,
        output: wp.array(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        precise = ray_for_pixel(config, pose, pixel)
        ray = Ray32(wp.vec3(precise.origin), wp.vec3(precise.direction), wp.float32(precise.length))
        if (
            not valid_ray(precise)
            or not finite_point(ray.origin)
            or not finite_point(ray.direction)
            or not wp.isfinite(ray.length)
            or ray.length <= wp.float32(0.0)
        ):
            wp.atomic_or(status, 0, 2)
            output[pixel] = dtype(0.0)
            return
        interval = finite_interval(ray.origin, ray.direction, config.shape)
        total = wp.float32(0.0)
        if interval.upper > interval.lower:
            step = (interval.upper - interval.lower) / wp.float32(config.samples)
            for index in range(config.samples):
                position = sample_position(interval.lower, step, index)
                point = sample_point(ray.origin, ray.direction, position)
                if not finite_point(point):
                    wp.atomic_or(status, 0, 2)
                total += sample_field(field, point, config.shape).value
            total *= ray.length * step
        output[pixel] = dtype(total)
        if not wp.isfinite(total):
            wp.atomic_or(status, 0, 2)

    return forward


@cache
def get_vjp(
    active_volume: bool,
    matrix_gradient: bool,
    active_pose: bool,
    *,
    per_ray: bool = False,
    ray_coordinates: bool = False,
    double_seed: bool = False,
    double_volume_gradient: bool = False,
):
    if ray_coordinates and (not per_ray or matrix_gradient):
        raise ValueError("ray-coordinate derivatives require per-ray six-component output")
    if per_ray and (active_volume or matrix_gradient or not active_pose):
        raise ValueError("per-ray diagnostics require only the six local pose coordinates")
    components = 12 if matrix_gradient else 6
    seed_dtype = wp.float64 if double_seed else wp.float32
    volume_dtype = wp.float64 if double_volume_gradient else wp.float32
    scatter = get_scatter_field(double_volume_gradient)
    carry_shared_face = active_volume and not active_pose and not double_volume_gradient

    @wp.kernel(module="unique", module_options=OPTIONS)
    def vjp(
        field: wp.array(dtype=wp.float32),
        pose: wp.array(dtype=wp.float64),
        config: Configuration,
        seeds: wp.array(dtype=seed_dtype),
        volume_gradient: wp.array(dtype=volume_dtype),
        partials: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        block, lane = wp.tid()
        pixel = block * wp.block_dim() + lane
        gradient = Vec12d()
        if pixel < config.pixels:
            precise = ray_for_pixel(config, pose, pixel)
            ray = Ray32(
                wp.vec3(precise.origin), wp.vec3(precise.direction), wp.float32(precise.length)
            )
            if (
                not valid_ray(precise)
                or not finite_point(ray.origin)
                or not finite_point(ray.direction)
                or not wp.isfinite(ray.length)
                or ray.length <= wp.float32(0.0)
            ):
                wp.atomic_or(status, 0, 2)
            else:
                interval = finite_interval(ray.origin, ray.direction, config.shape)
                span = interval.upper - interval.lower
                seed = wp.float32(seeds[pixel])
                if not wp.isfinite(seed):
                    wp.atomic_or(status, 0, 2)
                if span > wp.float32(0.0) and seed != wp.float32(0.0):
                    step = span / wp.float32(config.samples)
                    physical_step = ray.length * step
                    volume_factor = seed * physical_step
                    mean = wp.float32(0.0)
                    direct_origin, direct_direction = wp.vec3(), wp.vec3()
                    shift_lower, shift_upper = wp.float32(0.0), wp.float32(0.0)
                    if wp.static(active_volume and not double_volume_gradient):
                        pending_cell = wp.vec3i(-1, -1, -1)
                        cell_sums = Vec8f()
                        bounded_factor = wp.isfinite(volume_factor) and wp.abs(
                            volume_factor
                        ) <= wp.float32(1.0)
                    for index in range(config.samples):
                        position = sample_position(interval.lower, step, index)
                        point = sample_point(ray.origin, ray.direction, position)
                        if not finite_point(point):
                            wp.atomic_or(status, 0, 2)
                        if wp.static(active_pose):
                            value = sample_pose_field(field, point, config.shape)
                            mean += value.value
                            direct_origin += value.gradient
                            direct_direction += position * value.gradient
                            alpha = (wp.float32(index) + wp.float32(0.5)) / wp.float32(
                                config.samples
                            )
                            along = wp.dot(value.gradient, ray.direction)
                            shift_lower += (wp.float32(1.0) - alpha) * along
                            shift_upper += alpha * along
                        if wp.static(active_volume):
                            if wp.static(double_volume_gradient):
                                scatter(volume_gradient, point, config.shape, volume_factor, status)
                            else:
                                if inside_support(point, config.shape):
                                    cell = coordinates(point, config.shape)
                                    if (
                                        cell.lower[0] != pending_cell[0]
                                        or cell.lower[1] != pending_cell[1]
                                        or cell.lower[2] != pending_cell[2]
                                    ):
                                        if wp.static(carry_shared_face):
                                            if pending_cell[0] >= 0:
                                                if bounded_factor:
                                                    cell_sums = transition_cell(
                                                        volume_gradient,
                                                        pending_cell,
                                                        cell.lower,
                                                        config.shape,
                                                        cell_sums,
                                                    )
                                                else:
                                                    # General factors retain the original flushes.
                                                    flush_cell(
                                                        volume_gradient,
                                                        pending_cell,
                                                        config.shape,
                                                        cell_sums,
                                                    )
                                                    cell_sums = Vec8f()
                                            else:
                                                cell_sums = Vec8f()
                                            pending_cell = cell.lower
                                        else:
                                            if pending_cell[0] >= 0:
                                                flush_cell(
                                                    volume_gradient,
                                                    pending_cell,
                                                    config.shape,
                                                    cell_sums,
                                                )
                                            pending_cell = cell.lower
                                            cell_sums = Vec8f()
                                    if bounded_factor:
                                        # Unit-bounded terms cannot overflow an int32 sample sum.
                                        # Product rounding and output FTZ remain unchanged.
                                        for corner in range(8):
                                            weight = corner_weight(
                                                corner_factors(cell.fraction, corner)
                                            )
                                            increment = bounded_scatter_increment(
                                                volume_factor, weight
                                            )
                                            cell_sums[corner] += increment
                                    else:
                                        for corner in range(8):
                                            weight = corner_weight(
                                                corner_factors(cell.fraction, corner)
                                            )
                                            increment = scatter_increment(volume_factor, weight)
                                            if not wp.isfinite(increment):
                                                wp.atomic_or(status, 0, 2)
                                                wp.atomic_add(
                                                    volume_gradient,
                                                    corner_index(cell.lower, config.shape, corner),
                                                    increment,
                                                )
                                            else:
                                                partial_total = cell_sums[corner] + increment
                                                if not wp.isfinite(partial_total):
                                                    # Flush before a finite partial overflows.
                                                    wp.atomic_add(
                                                        volume_gradient,
                                                        corner_index(
                                                            cell.lower, config.shape, corner
                                                        ),
                                                        cell_sums[corner],
                                                    )
                                                    cell_sums[corner] = increment
                                                else:
                                                    cell_sums[corner] = partial_total
                                elif pending_cell[0] >= 0:
                                    flush_cell(
                                        volume_gradient, pending_cell, config.shape, cell_sums
                                    )
                                    pending_cell = wp.vec3i(-1, -1, -1)
                                    cell_sums = Vec8f()
                    if wp.static(active_volume and not double_volume_gradient):
                        if pending_cell[0] >= 0:
                            flush_cell(volume_gradient, pending_cell, config.shape, cell_sums)
                    if wp.static(active_pose):
                        lower_seed = -mean + span * shift_lower
                        upper_seed = mean + span * shift_upper
                        origin_gradient = (
                            span * direct_origin
                            + lower_seed * interval.lower_origin_gradient
                            + upper_seed * interval.upper_origin_gradient
                        )
                        direction_gradient = (
                            span * direct_direction
                            + lower_seed * interval.lower_direction_gradient
                            + upper_seed * interval.upper_direction_gradient
                        )
                        factor = seed * ray.length / wp.float32(config.samples)
                        # One widening after the complete FP32 sample loop.
                        source_adjoint = wp.float64(factor) * (
                            wp.transpose(config.object_to_grid) * wp.vec3d(origin_gradient)
                        )
                        direction_adjoint = wp.float64(factor) * (
                            wp.transpose(config.object_to_grid) * wp.vec3d(direction_gradient)
                        )
                        if wp.static(ray_coordinates):
                            rotation = unpack_rotation(pose)
                            source_world = rotation * source_adjoint
                            direction_world = rotation * direction_adjoint
                            integral = wp.float64(mean * step)
                            for axis in range(3):
                                direction_world[axis] += (
                                    wp.float64(seed)
                                    * integral
                                    * precise.world_direction[axis]
                                    / precise.length
                                )
                                gradient[axis] = source_world[axis] - direction_world[axis]
                                gradient[axis + 3] = direction_world[axis]
                        elif wp.static(matrix_gradient):
                            for row in range(3):
                                for column in range(3):
                                    gradient[3 * row + column] = (
                                        precise.world_source_relative[row] * source_adjoint[column]
                                        + precise.world_direction[row] * direction_adjoint[column]
                                    )
                            translation = -(unpack_rotation(pose) * source_adjoint)
                            for axis in range(3):
                                gradient[9 + axis] = translation[axis]
                        else:
                            tangent = _local_right_pose_adjoint(
                                source_adjoint,
                                precise.object_source,
                                direction_adjoint,
                                precise.object_direction,
                            )
                            for axis in range(6):
                                gradient[axis] = tangent[axis]
        if wp.static(per_ray):
            if pixel < config.pixels:
                for component in range(components):
                    component_value = gradient[component]
                    partials[wp.int64(pixel) * wp.int64(components) + wp.int64(component)] = (
                        component_value
                    )
                    if not wp.isfinite(component_value):
                        wp.atomic_or(status, 0, 2)
        elif wp.static(active_pose):
            for component in range(components):
                total = wp.tile_sum(wp.tile(gradient[component]))
                wp.tile_store(partials, total, offset=block * components + component)

    return vjp
