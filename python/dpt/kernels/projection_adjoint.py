"""Deterministic voxel gathering of the canonical strict projection weights."""

# Warp DSL annotations are executable type expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false

from functools import cache

import warp as wp

from dpt.kernels.geometry import unpack_rotation
from dpt.kernels.numerics import product3, product4
from dpt.kernels.projection import (
    OPTIONS,
    Configuration,
    advance_cells,
    begin_cells,
    cell_fraction,
    cell_lower,
    cell_segment,
    corner_index,
    finite_interval,
    finite_point,
    inside_support,
    ray_for_pixel,
    scaled_corner_weight,
    valid_ray,
)


@wp.func
def point_contribution(
    voxel: int, point: wp.vec3d, shape: wp.vec3i, seed: wp.float64
) -> wp.float64:
    result = wp.float64(0.0)
    if inside_support(point, shape):
        lower = cell_lower(point, shape)
        fraction = cell_fraction(point, lower, shape)
        for corner in range(8):
            if corner_index(lower, shape, corner) == voxel:
                result += scaled_corner_weight(fraction, corner, seed)
    return result


@wp.func
def cell_contribution(
    voxel: int,
    lower: wp.vec3i,
    first_point: wp.vec3d,
    second_point: wp.vec3d,
    shape: wp.vec3i,
    seed: wp.float64,
) -> wp.float64:
    result = wp.float64(0.0)
    first_fraction = cell_fraction(first_point, lower, shape)
    second_fraction = cell_fraction(second_point, lower, shape)
    for corner in range(8):
        if corner_index(lower, shape, corner) == voxel:
            first = seed
            second = seed
            if not inside_support(first_point, shape):
                first = wp.float64(0.0)
            if not inside_support(second_point, shape):
                second = wp.float64(0.0)
            first = scaled_corner_weight(first_fraction, corner, first)
            second = scaled_corner_weight(second_fraction, corner, second)
            result += first + second
    return result


@cache
def get_gather(double_output: bool = True, double_seed: bool = False, cell_gauss: bool = False):
    dtype = wp.float64 if double_output else wp.float32
    seed_dtype = wp.float64 if double_seed else wp.float32

    @wp.kernel(module="unique", module_options=OPTIONS)
    def gather(
        pose: wp.array(dtype=wp.float64),
        config: Configuration,
        cotangent: wp.array(dtype=seed_dtype),
        accumulate: bool,
        output: wp.array(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        voxel = wp.tid()
        total = wp.float64(0.0)
        for pixel in range(config.pixels):
            ray = ray_for_pixel(config, pose, pixel)
            if not valid_ray(ray):
                wp.atomic_or(status, 0, 2)
                continue
            if cotangent[pixel] == seed_dtype(0.0):
                continue
            interval = finite_interval(ray.origin, ray.direction, config.shape)
            span = interval.upper - interval.lower
            if span <= wp.float64(0.0):
                continue
            factor = wp.float64(cotangent[pixel]) * ray.length
            if wp.static(cell_gauss):
                traversal = begin_cells(ray, interval, config.shape)
                start = interval.lower
                segments = wp.int64(0)
                limit = (
                    wp.int64(config.shape[0])
                    + wp.int64(config.shape[1])
                    + wp.int64(config.shape[2])
                    + wp.int64(1)
                )
                while start < interval.upper and segments < limit:
                    segment = cell_segment(ray, traversal, start, config.shape)
                    if segment.end <= start:
                        wp.atomic_or(status, 0, 2)
                        break
                    if not finite_point(segment.first_point) or not finite_point(
                        segment.second_point
                    ):
                        wp.atomic_or(status, 0, 2)
                    volume_factor = factor * segment.radius
                    if not wp.isfinite(factor):
                        volume_factor = product3(
                            wp.float64(cotangent[pixel]), ray.length, segment.radius
                        )
                    if segment.shared_cell:
                        total += cell_contribution(
                            voxel,
                            segment.lower,
                            segment.first_point,
                            segment.second_point,
                            config.shape,
                            volume_factor,
                        )
                    else:
                        total += point_contribution(
                            voxel, segment.first_point, config.shape, volume_factor
                        )
                        total += point_contribution(
                            voxel, segment.second_point, config.shape, volume_factor
                        )
                    start = segment.end
                    traversal = advance_cells(
                        traversal, ray, config.shape, segment.end, interval.upper
                    )
                    segments += wp.int64(1)
                if start < interval.upper:
                    wp.atomic_or(status, 0, 2)
            else:
                step = span / wp.float64(config.samples)
                volume_factor = factor * span / wp.float64(config.samples)
                if not wp.isfinite(factor):
                    volume_factor = product4(
                        wp.float64(cotangent[pixel]),
                        ray.length,
                        span,
                        wp.float64(1.0) / wp.float64(config.samples),
                    )
                for sample in range(config.samples):
                    position = interval.lower + (wp.float64(sample) + wp.float64(0.5)) * step
                    point = ray.origin + position * ray.direction
                    if not finite_point(point):
                        wp.atomic_or(status, 0, 2)
                    total += point_contribution(voxel, point, config.shape, volume_factor)
        if accumulate:
            total += wp.float64(output[voxel])
        output[voxel] = dtype(total)
        if not wp.isfinite(output[voxel]):
            wp.atomic_or(status, 0, 2)

    return gather


@wp.struct
class GatherRay:
    origin: wp.vec3d
    direction: wp.vec3d
    length: wp.float64
    lower: wp.float64
    span: wp.float64
    step: wp.float64
    point_error: wp.vec3d


def _outward_native(body: str) -> str:
    """Supply an IEEE-754 adjacent-value step without platform math headers."""
    return (
        """
auto nextafter = [](double value, double direction) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint64_t magnitude = bits & 0x7fffffffffffffffull;
    if (magnitude > 0x7ff0000000000000ull) return value;
    const bool increasing = direction > 0.0;
    if (magnitude == 0ull) {
        bits = increasing ? 1ull : 0x8000000000000001ull;
    } else if ((value > 0.0) == increasing) {
        if (magnitude == 0x7ff0000000000000ull) return value;
        ++bits;
    } else {
        --bits;
    }
    memcpy(&value, &bits, sizeof(value));
    return value;
};
"""
        + body
    )


@wp.func_native(
    _outward_native("""
double lower = 0.0, upper = 0.0;
for (int axis = 0; axis < 3; ++axis) {
    const double product = a[axis] * b[axis];
    lower = nextafter(lower + nextafter(product, -INFINITY), -INFINITY);
    upper = nextafter(upper + nextafter(product, INFINITY), INFINITY);
}
return wp::vec_t<2, double>(lower, upper);
""")
)
def enclosed_dot(a: wp.vec3d, b: wp.vec3d) -> wp.vec2d:
    """Enclose the real dot product of the stored binary64 coefficients."""
    ...


@wp.func_native(
    _outward_native("""
double result_low = 0.0, result_high = 0.0;
for (int axis = 0; axis < 3; ++axis) {
    const double first = nextafter(lower[axis] - origin[axis], -INFINITY);
    const double last = nextafter(upper[axis] - origin[axis], INFINITY);
    const double lo = coefficient[axis] >= 0.0 ? first : last;
    const double hi = coefficient[axis] >= 0.0 ? last : first;
    result_low = nextafter(result_low + nextafter(coefficient[axis] * lo, -INFINITY), -INFINITY);
    result_high = nextafter(result_high + nextafter(coefficient[axis] * hi, INFINITY), INFINITY);
}
return wp::vec_t<2, double>(result_low, result_high);
""")
)
def enclosed_box_dot(
    coefficient: wp.vec3d, lower: wp.vec3d, upper: wp.vec3d, origin: wp.vec3d
) -> wp.vec2d:
    """Enclose coefficient dot (box-origin), widening every primitive."""
    ...


@wp.func_native(
    _outward_native("""
if (!wp::isfinite(numerator[0]) || !wp::isfinite(numerator[1])
    || !wp::isfinite(denominator[0]) || !wp::isfinite(denominator[1])
    || !(denominator[0] > 0.0 || denominator[1] < 0.0))
    return wp::vec_t<2, double>(NAN, NAN);
double lower = INFINITY, upper = -INFINITY;
for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
        const double quotient = numerator[i] / denominator[j];
        if (!wp::isfinite(quotient)) return wp::vec_t<2, double>(NAN, NAN);
        lower = wp::min(lower, nextafter(quotient, -INFINITY));
        upper = wp::max(upper, nextafter(quotient, INFINITY));
    }
}
return wp::vec_t<2, double>(lower, upper);
""")
)
def enclosed_quotient(numerator: wp.vec2d, denominator: wp.vec2d) -> wp.vec2d:
    """Return an outward quotient or NaNs for a denominator crossing zero."""
    ...


@wp.func_native(_outward_native("return nextafter(value - radius, -INFINITY);"))
def subtract_outward(value: wp.float64, radius: wp.float64) -> wp.float64: ...


@wp.func_native(_outward_native("return nextafter(value + radius, INFINITY);"))
def add_outward(value: wp.float64, radius: wp.float64) -> wp.float64: ...


@wp.func_native(
    _outward_native("""
return nextafter(1.7763568394002505e-15 * wp::max(wp::abs(origin), wp::abs(direction)), INFINITY);
""")
)
def point_roundoff_radius(origin: wp.float64, direction: wp.float64) -> wp.float64:
    """Cover the canonical fused or unfused point evaluation, including subnormals."""
    ...


@wp.kernel(module="unique", module_options=OPTIONS)
def prepare_cone_coefficients(
    pose: wp.array(dtype=wp.float64),
    config: Configuration,
    coefficients: wp.array(dtype=wp.vec3d),
):
    ray = ray_for_pixel(config, pose, 0)
    inverse = wp.transpose(unpack_rotation(pose))
    column = config.object_to_grid * (inverse * config.column_step)
    row = config.object_to_grid * (inverse * config.row_step)
    coefficients[0] = ray.origin
    coefficients[1] = wp.cross(row, ray.direction)
    coefficients[2] = wp.cross(ray.direction, column)
    coefficients[3] = wp.cross(column, row)


@cache
def get_prepare_rays(cone: bool):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def prepare(
        pose: wp.array(dtype=wp.float64),
        config: Configuration,
        coefficients: wp.array(dtype=wp.vec3d),
        rays: wp.array(dtype=GatherRay),
        metrics: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        original = ray_for_pixel(config, pose, pixel)
        if not valid_ray(original):
            wp.atomic_or(status, 0, 2)
        interval = finite_interval(original.origin, original.direction, config.shape)
        ray = GatherRay()
        ray.origin = original.origin
        ray.direction = original.direction
        ray.length = original.length
        ray.lower = interval.lower
        ray.span = wp.max(interval.upper - interval.lower, wp.float64(0.0))
        ray.step = ray.span / wp.float64(config.samples)
        for axis in range(3):
            ray.point_error[axis] = point_roundoff_radius(ray.origin[axis], ray.direction[axis])
            wp.atomic_max(metrics, axis + 2, ray.point_error[axis])
        # Validate all canonical nodes even if later culling visits no voxel.
        # This preserves ray-wide failure semantics and validates the snapshot.
        if ray.span > wp.float64(0.0):
            for sample in range(config.samples):
                position = ray.lower + (wp.float64(sample) + wp.float64(0.5)) * ray.step
                point = ray.origin + position * ray.direction
                if not finite_point(point):
                    wp.atomic_or(status, 0, 2)
        rays[pixel] = ray
        if wp.static(cone):
            denominator = enclosed_dot(coefficients[3], original.direction)
            column_range = enclosed_quotient(
                enclosed_dot(coefficients[1], original.direction), denominator
            )
            row_range = enclosed_quotient(
                enclosed_dot(coefficients[2], original.direction), denominator
            )
            if (
                not wp.isfinite(column_range[0])
                or not wp.isfinite(column_range[1])
                or not wp.isfinite(row_range[0])
                or not wp.isfinite(row_range[1])
            ):
                wp.atomic_max(metrics, 5, wp.float64(1.0))
            else:
                column = wp.float64(pixel % config.width)
                row = wp.float64(pixel // config.width)
                column_error = wp.max(
                    wp.abs(subtract_outward(column_range[0], column)),
                    wp.abs(add_outward(column_range[1], -column)),
                )
                row_error = wp.max(
                    wp.abs(subtract_outward(row_range[0], row)),
                    wp.abs(add_outward(row_range[1], -row)),
                )
                wp.atomic_max(metrics, 0, column_error)
                wp.atomic_max(metrics, 1, row_error)

    return prepare


@wp.struct
class VoxelSupport:
    lower: wp.vec3d
    upper: wp.vec3d


@wp.func
def voxel_support(voxel: int, shape: wp.vec3i) -> VoxelSupport:
    result = VoxelSupport()
    indices = wp.vec3i(
        voxel % shape[0], (voxel // shape[0]) % shape[1], voxel // (shape[0] * shape[1])
    )
    for axis in range(3):
        result.lower[axis] = wp.float64(indices[axis] - 1)
        result.upper[axis] = wp.float64(indices[axis] + 1)
        # Clamping keeps the neighbouring zero-weight corner in the cell.
        # Retain it as well: an overflowing cotangent times zero still has
        # the canonical numerical-failure semantics.
        if indices[axis] <= 1:
            result.lower[axis] = wp.float64(-0.5)
        if indices[axis] >= shape[axis] - 2:
            result.upper[axis] = wp.float64(shape[axis]) - wp.float64(0.5)
    return result


@wp.struct
class PixelBounds:
    first_column: int
    last_column: int
    first_row: int
    last_row: int
    fallback: int


@wp.func
def pixel_bounds(
    support: VoxelSupport,
    config: Configuration,
    coefficients: wp.array(dtype=wp.vec3d),
    metrics: wp.array(dtype=wp.float64),
) -> PixelBounds:
    result = PixelBounds()
    result.last_column = config.width
    result.last_row = config.pixels // config.width
    lower = wp.vec3d()
    upper = wp.vec3d()
    for axis in range(3):
        lower[axis] = subtract_outward(support.lower[axis], metrics[axis + 2])
        upper[axis] = add_outward(support.upper[axis], metrics[axis + 2])
    denominator = enclosed_box_dot(coefficients[3], lower, upper, coefficients[0])
    column = enclosed_quotient(
        enclosed_box_dot(coefficients[1], lower, upper, coefficients[0]), denominator
    )
    row = enclosed_quotient(
        enclosed_box_dot(coefficients[2], lower, upper, coefficients[0]), denominator
    )
    if (
        metrics[5] != wp.float64(0.0)
        or not wp.isfinite(metrics[0])
        or not wp.isfinite(metrics[1])
        or not wp.isfinite(metrics[2])
        or not wp.isfinite(metrics[3])
        or not wp.isfinite(metrics[4])
        or not wp.isfinite(column[0])
        or not wp.isfinite(column[1])
        or not wp.isfinite(row[0])
        or not wp.isfinite(row[1])
    ):
        result.fallback = 1
    else:
        column[0] = subtract_outward(column[0], metrics[0])
        column[1] = add_outward(column[1], metrics[0])
        row[0] = subtract_outward(row[0], metrics[1])
        row[1] = add_outward(row[1], metrics[1])
        if (
            column[1] < wp.float64(0.0)
            or row[1] < wp.float64(0.0)
            or column[0] > wp.float64(config.width - 1)
            or row[0] > wp.float64(result.last_row - 1)
        ):
            result.last_column = 0
            result.last_row = 0
        else:
            result.first_column = int(wp.ceil(wp.max(column[0], wp.float64(0.0))))
            result.last_column = int(wp.floor(wp.min(column[1], wp.float64(config.width - 1)))) + 1
            result.first_row = int(wp.ceil(wp.max(row[0], wp.float64(0.0))))
            result.last_row = int(wp.floor(wp.min(row[1], wp.float64(result.last_row - 1)))) + 1
    return result


@wp.func_native(
    _outward_native("""
double lower = -1.776356839400251e-15;
double upper = 1.0 + 1.776356839400251e-15;
for (int axis = 0; axis < 3; ++axis) {
    const double first = nextafter(support_lower[axis] - point_error[axis], -INFINITY);
    const double last = nextafter(support_upper[axis] + point_error[axis], INFINITY);
    if (direction[axis] == 0.0) {
        if (origin[axis] < first || origin[axis] > last)
            return wp::vec_t<2, double>(1.0, 0.0);
    } else {
        double a = nextafter(first - origin[axis], -INFINITY);
        double b = nextafter(last - origin[axis], INFINITY);
        if (direction[axis] < 0.0) { const double swap = a; a = b; b = swap; }
        a = nextafter(a / direction[axis], -INFINITY);
        b = nextafter(b / direction[axis], INFINITY);
        if (!wp::isfinite(a) || !wp::isfinite(b))
            return wp::vec_t<2, double>(NAN, NAN);
        lower = wp::max(lower, a);
        upper = wp::min(upper, b);
    }
}
return wp::vec_t<2, double>(lower, upper);
""")
)
def enclosed_support_interval(
    origin: wp.vec3d,
    direction: wp.vec3d,
    support_lower: wp.vec3d,
    support_upper: wp.vec3d,
    point_error: wp.vec3d,
) -> wp.vec2d: ...


@wp.func_native(
    _outward_native("""
const double radius = 1.776356839400251e-15;
const double low = nextafter(nextafter(interval[0] - radius, -INFINITY) - origin, -INFINITY);
const double high = nextafter(nextafter(interval[1] + radius, INFINITY) - origin, INFINITY);
return wp::vec_t<2, double>(
    nextafter(nextafter(low / step, -INFINITY) - 0.5, -INFINITY),
    nextafter(nextafter(high / step, INFINITY) - 0.5, INFINITY));
""")
)
def enclosed_sample_indices(
    interval: wp.vec2d, origin: wp.float64, step: wp.float64
) -> wp.vec2d: ...


@wp.func
def sample_bounds(ray: GatherRay, support: VoxelSupport, samples: int) -> wp.vec2i:
    result = wp.vec2i(0, samples)
    if ray.span <= wp.float64(0.0):
        return wp.vec2i(0, 0)
    if ray.step <= wp.float64(0.0):
        return result
    interval = enclosed_support_interval(
        ray.origin, ray.direction, support.lower, support.upper, ray.point_error
    )
    if interval[0] > interval[1]:
        return wp.vec2i(0, 0)
    indices = enclosed_sample_indices(interval, ray.lower, ray.step)
    if not wp.isfinite(indices[0]) or not wp.isfinite(indices[1]):
        return result
    if indices[1] < wp.float64(0.0) or indices[0] > wp.float64(samples - 1):
        return wp.vec2i(0, 0)
    result[0] = int(wp.ceil(wp.max(indices[0], wp.float64(0.0))))
    result[1] = int(wp.floor(wp.min(indices[1], wp.float64(samples - 1)))) + 1
    return result


@cache
def get_count_gather_work(cone: bool):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def count(
        config: Configuration,
        rays: wp.array(dtype=GatherRay),
        coefficients: wp.array(dtype=wp.vec3d),
        metrics: wp.array(dtype=wp.float64),
        max_rays_per_voxel: int,
        max_queries: wp.int64,
        totals: wp.array(dtype=wp.int64),
    ):
        block, lane = wp.tid()
        voxel = block * wp.block_dim() + lane
        ray_count = wp.int64(0)
        node_count = wp.int64(0)
        limited = wp.int64(0)
        fallbacks = wp.int64(0)
        if (
            voxel < config.shape[0] * config.shape[1] * config.shape[2]
            and totals[0] + totals[1] <= max_queries
        ):
            support = voxel_support(voxel, config.shape)
            bounds = PixelBounds()
            bounds.last_column = config.pixels
            bounds.last_row = 1
            if wp.static(cone):
                bounds = pixel_bounds(support, config, coefficients, metrics)
            count = (bounds.last_column - bounds.first_column) * (
                bounds.last_row - bounds.first_row
            )
            if count > max_rays_per_voxel:
                limited = wp.int64(1)
            else:
                ray_count = wp.int64(count)
                fallbacks = wp.int64(bounds.fallback)
                for row in range(bounds.first_row, bounds.last_row):
                    for column in range(bounds.first_column, bounds.last_column):
                        pixel = column
                        if wp.static(cone):
                            pixel += row * config.width
                        nodes = sample_bounds(rays[pixel], support, config.samples)
                        node_count += wp.int64(wp.max(nodes[1] - nodes[0], 0))
        wp.tile_atomic_add(totals, wp.tile_sum(wp.tile(ray_count)), offset=0)
        wp.tile_atomic_add(totals, wp.tile_sum(wp.tile(node_count)), offset=1)
        wp.tile_atomic_add(totals, wp.tile_sum(wp.tile(limited)), offset=2)
        wp.tile_atomic_add(totals, wp.tile_sum(wp.tile(fallbacks)), offset=3)

    return count


@cache
def get_culled_gather(double_output: bool, double_seed: bool, cone: bool):
    dtype = wp.float64 if double_output else wp.float32
    seed_dtype = wp.float64 if double_seed else wp.float32

    @wp.kernel(module="unique", module_options=OPTIONS)
    def gather(
        config: Configuration,
        rays: wp.array(dtype=GatherRay),
        coefficients: wp.array(dtype=wp.vec3d),
        metrics: wp.array(dtype=wp.float64),
        cotangent: wp.array(dtype=seed_dtype),
        accumulate: bool,
        output: wp.array(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        voxel = wp.tid()
        support = voxel_support(voxel, config.shape)
        bounds = PixelBounds()
        bounds.last_column = config.pixels
        bounds.last_row = 1
        if wp.static(cone):
            bounds = pixel_bounds(support, config, coefficients, metrics)
        total = wp.float64(0.0)
        for row in range(bounds.first_row, bounds.last_row):
            for column in range(bounds.first_column, bounds.last_column):
                pixel = column
                if wp.static(cone):
                    pixel += row * config.width
                if cotangent[pixel] == seed_dtype(0.0):
                    continue
                ray = rays[pixel]
                nodes = sample_bounds(ray, support, config.samples)
                factor = wp.float64(cotangent[pixel]) * ray.length
                volume_factor = factor * ray.span / wp.float64(config.samples)
                if not wp.isfinite(factor):
                    volume_factor = product4(
                        wp.float64(cotangent[pixel]),
                        ray.length,
                        ray.span,
                        wp.float64(1.0) / wp.float64(config.samples),
                    )
                for sample in range(nodes[0], nodes[1]):
                    position = ray.lower + (wp.float64(sample) + wp.float64(0.5)) * ray.step
                    point = ray.origin + position * ray.direction
                    total += point_contribution(voxel, point, config.shape, volume_factor)
        if accumulate:
            total += wp.float64(output[voxel])
        output[voxel] = dtype(total)
        if not wp.isfinite(output[voxel]):
            wp.atomic_or(status, 0, 2)

    return gather
