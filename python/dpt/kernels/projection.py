"""Streaming finite-ray quadrature and an explicit first-order discrete VJP.

One lane traces one ray. Strict values and pose derivatives use FP64 registers
with native FP32, FP16 or BF16 field storage. Fast sampling uses FP32 cell
arithmetic and FP64 ray sums. Reverse recomputes each sample; no ray-by-sample tensor or
image Jacobian is retained. Pose partials use a fixed tree; active-volume
cotangents use explicitly selected FP32 or FP64 scatter atomics and are not
bitwise deterministic.
"""

# Warp annotations are executable DSL expressions; host interfaces remain strict.
# The optional GPU import is resolved only when an operator is prepared.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false

from functools import cache
from typing import Any

import warp as wp

from dpt.kernels.geometry import (
    _local_right_pose_adjoint,
    unpack_rotation,
    unpack_translation,
)
from dpt.kernels.numerics import product3, product4
from dpt.kernels.projection_checks import (
    get_dependency_marker as get_dependency_marker,
)
from dpt.kernels.projection_checks import (
    get_validate_field as get_validate_field,
)
from dpt.kernels.projection_checks import (
    get_validate_finite as get_validate_finite,
)
from dpt.kernels.projection_checks import (
    validate_finite as validate_finite,
)
from dpt.kernels.projection_checks import (
    validate_nonnegative as validate_nonnegative,
)

OPTIONS = {"fast_math": False, "fuse_fp": True, "enable_backward": False}
wp.set_module_options(OPTIONS)
Vec12d = wp.types.vector(length=12, dtype=wp.float64)
Vec8d = wp.types.vector(length=8, dtype=wp.float64)


@wp.struct
class Configuration:
    shape: wp.vec3i
    object_to_grid: wp.mat33d
    grid_origin: wp.vec3d
    source: wp.vec3d
    detector_origin: wp.vec3d
    column_step: wp.vec3d
    row_step: wp.vec3d
    width: int
    pixels: int
    samples: int
    is_ray_bundle: bool
    ray_sources: wp.array(dtype=wp.vec3d)
    ray_endpoints: wp.array(dtype=wp.vec3d)


@wp.struct
class Ray:
    origin: wp.vec3d
    direction: wp.vec3d
    object_source: wp.vec3d
    object_direction: wp.vec3d
    world_source_relative: wp.vec3d
    world_direction: wp.vec3d
    length: wp.float64


@wp.func
def ray_for_pixel(config: Configuration, pose: wp.array(dtype=wp.float64), pixel: int) -> Ray:
    ray = Ray()
    source = config.source
    end = config.detector_origin
    if config.is_ray_bundle:
        source = config.ray_sources[pixel]
        end = config.ray_endpoints[pixel]
    else:
        row, column = pixel // config.width, pixel % config.width
        end = (
            config.detector_origin
            + wp.float64(column) * config.column_step
            + wp.float64(row) * config.row_step
        )
    inverse = wp.transpose(unpack_rotation(pose))
    ray.world_source_relative = source - unpack_translation(pose)
    ray.world_direction = end - source
    ray.object_source = inverse * ray.world_source_relative
    ray.object_direction = inverse * ray.world_direction
    ray.origin = config.object_to_grid * (ray.object_source - config.grid_origin)
    ray.direction = config.object_to_grid * ray.object_direction
    ray.length = wp.length(ray.world_direction)
    return ray


@wp.func
def finite_point(point: wp.vec3d) -> bool:
    return wp.isfinite(point[0]) and wp.isfinite(point[1]) and wp.isfinite(point[2])


@wp.func
def valid_ray(ray: Ray) -> bool:
    return (
        finite_point(ray.origin)
        and finite_point(ray.direction)
        and finite_point(ray.object_source)
        and finite_point(ray.object_direction)
        and finite_point(ray.world_source_relative)
        and finite_point(ray.world_direction)
        and wp.isfinite(ray.length)
        and ray.length > wp.float64(0.0)
    )


@wp.func
def inside_support(point: wp.vec3d, shape: wp.vec3i) -> bool:
    return (
        finite_point(point)
        and point[0] >= wp.float64(-0.5)
        and point[1] >= wp.float64(-0.5)
        and point[2] >= wp.float64(-0.5)
        and point[0] <= wp.float64(shape[0]) - wp.float64(0.5)
        and point[1] <= wp.float64(shape[1]) - wp.float64(0.5)
        and point[2] <= wp.float64(shape[2]) - wp.float64(0.5)
    )


@wp.struct
class Interval:
    lower: wp.float64
    upper: wp.float64
    lower_origin_gradient: wp.vec3d
    lower_direction_gradient: wp.vec3d
    upper_origin_gradient: wp.vec3d
    upper_direction_gradient: wp.vec3d


@wp.func
def finite_interval(origin: wp.vec3d, direction: wp.vec3d, shape: wp.vec3i) -> Interval:
    """Clip a finite segment and retain the active face derivatives.

    Exact parallelism is handled separately; arbitrary epsilon thresholds would
    change the physical interval. At tied faces a derivative is not unique:
    strict comparisons choose the first active axis, a documented branch value.
    """
    interval = Interval()
    interval.lower = wp.float64(0.0)
    interval.upper = wp.float64(1.0)
    for axis in range(3):
        low = wp.float64(-0.5)
        high = wp.float64(shape[axis]) - wp.float64(0.5)
        velocity = direction[axis]
        if velocity == wp.float64(0.0):
            if origin[axis] < low or origin[axis] > high:
                interval.upper = wp.float64(-1.0)
        else:
            first = (low - origin[axis]) / velocity
            last = (high - origin[axis]) / velocity
            if first > last:
                temporary = first
                first = last
                last = temporary
            if first > interval.lower:
                interval.lower = first
                interval.lower_origin_gradient = wp.vec3d(0.0)
                interval.lower_direction_gradient = wp.vec3d(0.0)
                interval.lower_origin_gradient[axis] = -wp.float64(1.0) / velocity
                interval.lower_direction_gradient[axis] = -first / velocity
            if last < interval.upper:
                interval.upper = last
                interval.upper_origin_gradient = wp.vec3d(0.0)
                interval.upper_direction_gradient = wp.vec3d(0.0)
                interval.upper_origin_gradient[axis] = -wp.float64(1.0) / velocity
                interval.upper_direction_gradient[axis] = -last / velocity
    return interval


@wp.struct
class Sample:
    value: wp.float64
    gradient: wp.vec3d


@wp.struct
class Cell:
    lower: wp.vec3i
    values: Vec8d


@wp.func
def cell_lower(point: wp.vec3d, shape: wp.vec3i) -> wp.vec3i:
    lower = wp.vec3i()
    for axis in range(3):
        coordinate = wp.clamp(point[axis], wp.float64(0.0), wp.float64(shape[axis] - 1))
        lower[axis] = wp.min(int(wp.floor(coordinate)), wp.max(shape[axis] - 2, 0))
    return lower


@wp.func
def cell_fraction(point: wp.vec3d, lower: wp.vec3i, shape: wp.vec3i) -> wp.vec3d:
    fraction = wp.vec3d()
    for axis in range(3):
        coordinate = wp.clamp(point[axis], wp.float64(0.0), wp.float64(shape[axis] - 1))
        fraction[axis] = coordinate - wp.float64(lower[axis])
    return fraction


@wp.func
def corner_index(lower: wp.vec3i, shape: wp.vec3i, corner: int) -> int:
    x = wp.min(lower[0] + (corner & 1), shape[0] - 1)
    y = wp.min(lower[1] + ((corner >> 1) & 1), shape[1] - 1)
    z = wp.min(lower[2] + ((corner >> 2) & 1), shape[2] - 1)
    return (z * shape[1] + y) * shape[0] + x


@wp.func
def load_cell(
    field: wp.array(dtype=Any), lower: wp.vec3i, shape: wp.vec3i, field_offset: int = 0
) -> Cell:
    cell = Cell()
    cell.lower = lower
    for corner in range(8):
        cell.values[corner] = wp.float64(field[field_offset + corner_index(lower, shape, corner)])
    return cell


@wp.func
def sample_cell_fraction(
    cell: Cell, point: wp.vec3d, shape: wp.vec3i, fraction: wp.vec3d
) -> Sample:
    """Evaluate the original strict interpolation for an admitted realised cell."""
    result = Sample()
    slope = wp.vec3d()
    for axis in range(3):
        if point[axis] >= wp.float64(0.0) and point[axis] < wp.float64(shape[axis] - 1):
            slope[axis] = wp.float64(1.0)
    # Separable interpolation reuses the same eight loads for value and slope.
    # Form differences in FP64: FP32 subtraction would lose small field slopes.
    # The compiler removes slope work from the forward-only caller.
    v000, v100, v010, v110 = cell.values[0], cell.values[1], cell.values[2], cell.values[3]
    v001, v101, v011, v111 = cell.values[4], cell.values[5], cell.values[6], cell.values[7]
    dx00, dx10 = v100 - v000, v110 - v010
    dx01, dx11 = v101 - v001, v111 - v011
    fx, fy, fz = fraction[0], fraction[1], fraction[2]
    # Weighted lerp preserves a small endpoint next to a huge neighbour.
    # a+t*(b-a) would erase b at t=1 when the difference rounds to -a.
    ax, ay, az = wp.float64(1.0) - fx, wp.float64(1.0) - fy, wp.float64(1.0) - fz
    x00, x10 = ax * v000 + fx * v100, ax * v010 + fx * v110
    x01, x11 = ax * v001 + fx * v101, ax * v011 + fx * v111
    xy0, xy1 = ay * x00 + fy * x10, ay * x01 + fy * x11
    result.value = az * xy0 + fz * xy1
    # Differentiate corner values before interpolation. Subtracting two
    # already interpolated values can erase a small slope beside a huge
    # orthogonal background, even when its cotangent is representable.
    dx0, dx1 = ay * dx00 + fy * dx10, ay * dx01 + fy * dx11
    dy0 = ax * (v010 - v000) + fx * (v110 - v100)
    dy1 = ax * (v011 - v001) + fx * (v111 - v101)
    dz0 = ax * (v001 - v000) + fx * (v101 - v100)
    dz1 = ax * (v011 - v010) + fx * (v111 - v110)
    result.gradient = wp.vec3d(
        slope[0] * (az * dx0 + fz * dx1),
        slope[1] * (az * dy0 + fz * dy1),
        slope[2] * (ay * dz0 + fy * dz1),
    )
    return result


@wp.func
def sample_cell(cell: Cell, point: wp.vec3d, shape: wp.vec3i) -> Sample:
    """Value and grid-coordinate slope of the half-cell-extended sampled field."""
    result = Sample()
    if not inside_support(point, shape):
        return result
    fraction = cell_fraction(point, cell.lower, shape)
    return sample_cell_fraction(cell, point, shape, fraction)


@wp.func
def sample_field(
    field: wp.array(dtype=Any), point: wp.vec3d, shape: wp.vec3i, field_offset: int = 0
) -> Sample:
    if not inside_support(point, shape):
        return Sample()
    return sample_cell(
        load_cell(field, cell_lower(point, shape), shape, field_offset), point, shape
    )


@wp.struct
class MixedCoordinates:
    lower: wp.vec3i
    fraction: wp.vec3


@wp.func
def mixed_coordinates(point: wp.vec3d, shape: wp.vec3i) -> MixedCoordinates:
    coordinates = MixedCoordinates()
    for axis in range(3):
        value = wp.clamp(wp.float32(point[axis]), wp.float32(0.0), wp.float32(shape[axis] - 1))
        coordinates.lower[axis] = wp.min(int(wp.floor(value)), wp.max(shape[axis] - 2, 0))
        coordinates.fraction[axis] = value - wp.float32(coordinates.lower[axis])
    return coordinates


@wp.func
def mixed_corner_weight(fraction: wp.vec3, corner: int) -> wp.float32:
    weight = wp.float32(1.0)
    for axis in range(3):
        if ((corner >> axis) & 1) == 0:
            weight *= wp.float32(1.0) - fraction[axis]
        else:
            weight *= fraction[axis]
    return weight


@wp.func
def sample_mixed_field(
    field: wp.array(dtype=Any), point: wp.vec3d, shape: wp.vec3i, field_offset: int = 0
) -> wp.float64:
    value = wp.float64(0.0)
    if inside_support(point, shape):
        coordinates = mixed_coordinates(point, shape)
        for corner in range(8):
            weight = wp.float64(mixed_corner_weight(coordinates.fraction, corner))
            value += weight * wp.float64(
                field[field_offset + corner_index(coordinates.lower, shape, corner)]
            )
    return value


@wp.func
def sample_float32_field(
    field: wp.array(dtype=Any), point: wp.vec3d, shape: wp.vec3i
) -> wp.float32:
    """Interpolate a native stored field with FP32 products and a cell sum."""
    value = wp.float32(0.0)
    if inside_support(point, shape):
        coordinates = mixed_coordinates(point, shape)
        for corner in range(8):
            weight = mixed_corner_weight(coordinates.fraction, corner)
            value = mixed_value_fma(
                weight, wp.float32(field[corner_index(coordinates.lower, shape, corner)]), value
            )
    return value


@wp.func_native("return fmaf(weight, value, partial);")
def mixed_value_fma(weight: wp.float32, value: wp.float32, partial: wp.float32) -> wp.float32:
    """Round each weighted cell accumulation once on CPU and CUDA."""
    ...


@wp.struct
class Midpoints:
    entry: wp.vec3
    step: wp.vec3


@wp.func
def begin_midpoints(ray: Ray, interval: Interval, samples: int) -> Midpoints:
    traversal = Midpoints()
    entry = ray.origin + interval.lower * ray.direction
    step = ((interval.upper - interval.lower) / wp.float64(samples)) * ray.direction
    traversal.entry = wp.vec3(wp.float32(entry[0]), wp.float32(entry[1]), wp.float32(entry[2]))
    traversal.step = wp.vec3(wp.float32(step[0]), wp.float32(step[1]), wp.float32(step[2]))
    return traversal


@wp.func
def mixed_midpoint_point(traversal: Midpoints, sample: int) -> wp.vec3d:
    return mixed_point_fma(traversal.entry, traversal.step, sample)


@wp.func_native("""
const float offset = static_cast<float>(sample) + 0.5f;
return wp::vec_t<3, double>(
    static_cast<double>(fmaf(offset, step[0], entry[0])),
    static_cast<double>(fmaf(offset, step[1], entry[1])),
    static_cast<double>(fmaf(offset, step[2], entry[2])));
""")
def mixed_point_fma(entry: wp.vec3, step: wp.vec3, sample: int) -> wp.vec3d:
    """One explicit FP32 FMA per grid axis, shared by forward and scatter."""
    ...


@wp.func
def scaled_corner_weight(fraction: wp.vec3d, corner: int, seed: wp.float64) -> wp.float64:
    """Multiply a cotangent by the strict trilinear factors in axis order."""
    contribution = seed
    for axis in range(3):
        if ((corner >> axis) & 1) == 0:
            contribution *= wp.float64(1.0) - fraction[axis]
        else:
            contribution *= fraction[axis]
    return contribution


@cache
def get_scatter_realised(double: bool = False, float32_increments: bool = False):
    """Strict joint scatter after the shared support and coordinate evaluation."""
    dtype = wp.float64 if double else wp.float32

    @wp.func
    def scatter(
        gradient: wp.array(dtype=dtype),
        lower: wp.vec3i,
        fraction: wp.vec3d,
        shape: wp.vec3i,
        seed: wp.float64,
        status: wp.array(dtype=wp.int32),
    ):
        for corner in range(8):
            contribution = scaled_corner_weight(fraction, corner, seed)
            rounded = dtype(contribution)
            if wp.static(float32_increments):
                rounded = dtype(wp.float32(contribution))
            if not wp.isfinite(rounded):
                wp.atomic_or(status, 0, 2)
            wp.atomic_add(gradient, corner_index(lower, shape, corner), rounded)

    return scatter


@cache
def get_scatter_field(
    double: bool = False,
    mixed: bool = False,
    float32_increments: bool = False,
    certified_support: bool = False,
    certified_range: bool = False,
):
    dtype = wp.float64 if double else wp.float32

    @wp.func
    def scatter(
        gradient: wp.array(dtype=dtype),
        point: wp.vec3d,
        shape: wp.vec3i,
        seed: wp.float64,
        status: wp.array(dtype=wp.int32),
        field_offset: int = 0,
    ):
        if wp.static(not certified_support):
            if not finite_point(point):
                wp.atomic_or(status, 0, 2)
            if not inside_support(point, shape):
                return
        lower = cell_lower(point, shape)
        fraction = cell_fraction(point, lower, shape)
        coordinates = MixedCoordinates()
        if wp.static(mixed):
            coordinates = mixed_coordinates(point, shape)
            lower = coordinates.lower
        for corner in range(8):
            contribution = seed
            if wp.static(mixed):
                contribution *= wp.float64(mixed_corner_weight(coordinates.fraction, corner))
            else:
                contribution = scaled_corner_weight(fraction, corner, seed)
            rounded = dtype(contribution)
            if wp.static(float32_increments):
                # Preserve the original FP32 term and its range diagnostic
                # while allowing a wider destination to sum those terms.
                rounded = dtype(wp.float32(contribution))
            if wp.static(not certified_range):
                if not wp.isfinite(rounded):
                    wp.atomic_or(status, 0, 2)
            wp.atomic_add(gradient, field_offset + corner_index(lower, shape, corner), rounded)

    return scatter


@cache
def get_scatter_cell(double: bool = False, float32_increments: bool = False):
    dtype = wp.float64 if double else wp.float32

    @wp.func
    def scatter(
        gradient: wp.array(dtype=dtype),
        lower: wp.vec3i,
        first_point: wp.vec3d,
        second_point: wp.vec3d,
        shape: wp.vec3i,
        seed: wp.float64,
        status: wp.array(dtype=wp.int32),
        field_offset: int = 0,
    ):
        if not finite_point(first_point) or not finite_point(second_point):
            wp.atomic_or(status, 0, 2)
        first_fraction = cell_fraction(first_point, lower, shape)
        second_fraction = cell_fraction(second_point, lower, shape)
        # Both nodes belong to this interpolation cell. Combine their corner
        # weights in FP64 before the single rounding and atomic per corner.
        for corner in range(8):
            first = seed
            second = seed
            if not inside_support(first_point, shape):
                first = wp.float64(0.0)
            if not inside_support(second_point, shape):
                second = wp.float64(0.0)
            first = scaled_corner_weight(first_fraction, corner, first)
            second = scaled_corner_weight(second_fraction, corner, second)
            rounded = dtype(first + second)
            if wp.static(float32_increments):
                rounded = dtype(wp.float32(first + second))
            if not wp.isfinite(rounded):
                wp.atomic_or(status, 0, 2)
            wp.atomic_add(gradient, field_offset + corner_index(lower, shape, corner), rounded)

    return scatter


@wp.struct
class CellTraversal:
    plane: wp.vec3i
    step: wp.vec3i
    next: wp.vec3d
    inverse_direction: wp.vec3d


@wp.func_native("""
if (inverse == 0.0) return delta / direction;
const double quotient = delta * inverse;
if (!wp::isfinite(quotient) || wp::abs(quotient) < 2.2250738585072014e-308)
    return delta / direction;
// Correct the cached reciprocal product using its exact FMA residual.
return fma(fma(-quotient, direction, delta), inverse, quotient);
""")
def crossing_quotient(delta: wp.float64, direction: wp.float64, inverse: wp.float64) -> wp.float64:
    """Cached reciprocal with one correction; range extremes retain division."""
    ...


@wp.func
def cell_crossing(
    ray: Ray,
    axis: int,
    plane: int,
    shape: wp.vec3i,
    upper: wp.float64,
    inverse: wp.float64 = 0.0,
) -> wp.float64:
    if ray.direction[axis] == wp.float64(0.0) or plane < 0 or plane >= shape[axis]:
        return upper
    return wp.min(
        crossing_quotient(wp.float64(plane) - ray.origin[axis], ray.direction[axis], inverse), upper
    )


@wp.func
def begin_cells(ray: Ray, interval: Interval, shape: wp.vec3i) -> CellTraversal:
    """Next strictly forward interpolation plane, including half-cell edge slabs."""
    traversal = CellTraversal()
    entry = ray.origin + interval.lower * ray.direction
    for axis in range(3):
        velocity = ray.direction[axis]
        if velocity != wp.float64(0.0):
            inverse = wp.float64(1.0) / velocity
            if (
                wp.isfinite(inverse)
                and wp.abs(inverse) >= wp.float64(2.2250738585072014e-308)
                and wp.abs(velocity) >= wp.float64(2.2250738585072014e-308)
            ):
                traversal.inverse_direction[axis] = inverse
        if velocity > wp.float64(0.0):
            traversal.step[axis] = 1
            traversal.plane[axis] = wp.max(int(wp.ceil(entry[axis])), 0)
        elif velocity < wp.float64(0.0):
            traversal.step[axis] = -1
            traversal.plane[axis] = wp.min(int(wp.floor(entry[axis])), shape[axis] - 1)
        crossing = cell_crossing(
            ray,
            axis,
            traversal.plane[axis],
            shape,
            interval.upper,
            traversal.inverse_direction[axis],
        )
        if crossing <= interval.lower:
            traversal.plane[axis] += traversal.step[axis]
            crossing = cell_crossing(
                ray,
                axis,
                traversal.plane[axis],
                shape,
                interval.upper,
                traversal.inverse_direction[axis],
            )
        traversal.next[axis] = crossing
    return traversal


@wp.struct
class CellSegment:
    end: wp.float64
    radius: wp.float64
    first_position: wp.float64
    second_position: wp.float64
    first_point: wp.vec3d
    second_point: wp.vec3d
    lower: wp.vec3i
    shared_cell: bool


@wp.func
def cell_segment(
    ray: Ray, traversal: CellTraversal, start: wp.float64, shape: wp.vec3i
) -> CellSegment:
    segment = CellSegment()
    segment.end = wp.min(traversal.next[0], wp.min(traversal.next[1], traversal.next[2]))
    segment.radius = (segment.end - start) * wp.float64(0.5)
    centre = start + segment.radius
    offset = segment.radius / wp.sqrt(wp.float64(3.0))
    segment.first_position = centre - offset
    segment.second_position = centre + offset
    segment.first_point = ray.origin + segment.first_position * ray.direction
    segment.second_point = ray.origin + segment.second_position * ray.direction
    segment.lower = cell_lower(ray.origin + centre * ray.direction, shape)
    segment.shared_cell = True
    for axis in range(3):
        lower = wp.float64(segment.lower[axis])
        upper = lower + wp.float64(1.0)
        first_coordinate = wp.clamp(
            segment.first_point[axis], wp.float64(0.0), wp.float64(shape[axis] - 1)
        )
        second_coordinate = wp.clamp(
            segment.second_point[axis], wp.float64(0.0), wp.float64(shape[axis] - 1)
        )
        if first_coordinate < lower or second_coordinate < lower:
            segment.shared_cell = False
        if segment.lower[axis] < shape[axis] - 2 and (
            first_coordinate >= upper or second_coordinate >= upper
        ):
            # Extremely short intervals can round a node onto the adjacent
            # cell. Preserve the original branch interpolation in that case.
            segment.shared_cell = False
    return segment


@wp.func
def advance_cells(
    traversal: CellTraversal, ray: Ray, shape: wp.vec3i, end: wp.float64, upper: wp.float64
) -> CellTraversal:
    for axis in range(3):
        # Advance every tied plane; no epsilon changes which interval is integrated.
        if traversal.next[axis] <= end:
            traversal.plane[axis] += traversal.step[axis]
            traversal.next[axis] = cell_crossing(
                ray, axis, traversal.plane[axis], shape, upper, traversal.inverse_direction[axis]
            )
    return traversal


@wp.func
def support_value(
    field: wp.array(dtype=Any), point: wp.vec3d, shape: wp.vec3i, field_offset: int = 0
) -> wp.float64:
    # A clipped endpoint can round a few ulps outside its active support face.
    bounded = point
    for axis in range(3):
        bounded[axis] = wp.clamp(
            point[axis], wp.float64(-0.5), wp.float64(shape[axis]) - wp.float64(0.5)
        )
    return sample_field(field, bounded, shape, field_offset).value


@wp.func
def sample_segment(
    field: wp.array(dtype=Any),
    segment: CellSegment,
    shape: wp.vec3i,
    field_offset: int = 0,
) -> tuple[Sample, Sample]:
    first = Sample()
    second = Sample()
    if segment.shared_cell:
        cell = load_cell(field, segment.lower, shape, field_offset)
        first = sample_cell(cell, segment.first_point, shape)
        second = sample_cell(cell, segment.second_point, shape)
    else:
        first = sample_field(field, segment.first_point, shape, field_offset)
        second = sample_field(field, segment.second_point, shape, field_offset)
    return first, second


@wp.func
def _scalar_segment_action(
    field: wp.array(dtype=Any),
    segment: CellSegment,
    shape: wp.vec3i,
    voxels: int,
    integral: wp.float64,
) -> wp.float64:
    if segment.shared_cell:
        cell = load_cell(field, segment.lower, shape)
        integral += segment.radius * sample_cell(cell, segment.first_point, shape).value
        integral += segment.radius * sample_cell(cell, segment.second_point, shape).value
    else:
        integral += segment.radius * sample_field(field, segment.first_point, shape).value
        integral += segment.radius * sample_field(field, segment.second_point, shape).value
    return integral


@cache
def get_cell_integrator(action: Any):
    """Specialise the canonical cell traversal with a fixed segment action."""

    @wp.func
    def integrate(
        field: wp.array(dtype=Any),
        ray: Ray,
        interval: Interval,
        shape: wp.vec3i,
        voxels: int,
        total: Any,
        status: wp.array(dtype=wp.int32),
    ) -> Any:
        traversal = begin_cells(ray, interval, shape)
        start = interval.lower
        segments = wp.int64(0)
        limit = wp.int64(shape[0]) + wp.int64(shape[1]) + wp.int64(shape[2]) + wp.int64(1)
        while start < interval.upper and segments < limit:
            segment = cell_segment(ray, traversal, start, shape)
            if segment.end <= start:
                wp.atomic_or(status, 0, 2)
                break
            if not finite_point(segment.first_point) or not finite_point(segment.second_point):
                wp.atomic_or(status, 0, 2)
            total = action(field, segment, shape, voxels, total)
            start = segment.end
            traversal = advance_cells(traversal, ray, shape, segment.end, interval.upper)
            segments += wp.int64(1)
        if start < interval.upper:
            wp.atomic_or(status, 0, 2)
        return total

    return integrate


@cache
def _get_scalar_midpoint_action(mixed: bool, float32_partials: bool):
    @wp.func
    def action(
        field: wp.array(dtype=Any),
        point: wp.vec3d,
        shape: wp.vec3i,
        voxels: int,
        integral: wp.float64,
    ) -> wp.float64:
        if wp.static(float32_partials):
            integral += wp.float64(sample_float32_field(field, point, shape))
        elif wp.static(mixed):
            integral += sample_mixed_field(field, point, shape)
        else:
            integral += sample_field(field, point, shape).value
        return integral

    return action


@cache
def get_midpoint_integrator(action: Any, mixed: bool = False):
    """Specialise the fixed midpoint traversal without changing its ray recurrence."""

    @wp.func
    def integrate_midpoints(
        field: wp.array(dtype=Any),
        ray: Ray,
        interval: Interval,
        shape: wp.vec3i,
        samples: int,
        voxels: int,
        total: Any,
        status: wp.array(dtype=wp.int32),
    ):
        step = (interval.upper - interval.lower) / wp.float64(samples)
        midpoints = Midpoints()
        if wp.static(mixed):
            midpoints = begin_midpoints(ray, interval, samples)
        for sample in range(samples):
            position = interval.lower + (wp.float64(sample) + wp.float64(0.5)) * step
            point = ray.origin + position * ray.direction
            if wp.static(mixed):
                point = mixed_midpoint_point(midpoints, sample)
            if not finite_point(point):
                wp.atomic_or(status, 0, 2)
            total = action(field, point, shape, voxels, total)
        return total, step

    return integrate_midpoints


@cache
def get_forward(
    double_output: bool = False,
    cell_gauss: bool = False,
    mixed: bool = False,
    storage_precision: str = "float32",
    float32_partials: bool = False,
):
    dtype = wp.float64 if double_output else wp.float32
    field_dtype = getattr(wp, storage_precision)
    integrate_cells = get_cell_integrator(_scalar_segment_action)
    integrate_midpoints = get_midpoint_integrator(
        _get_scalar_midpoint_action(mixed, float32_partials), mixed
    )

    @wp.kernel(module="unique", module_options=OPTIONS)
    def forward(
        field: wp.array(dtype=field_dtype),
        pose: wp.array(dtype=wp.float64),
        config: Configuration,
        output: wp.array(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        ray = ray_for_pixel(config, pose, pixel)
        if not valid_ray(ray):
            wp.atomic_or(status, 0, 2)
            output[pixel] = dtype(0.0)
            return
        interval = finite_interval(ray.origin, ray.direction, config.shape)
        integral = wp.float64(0.0)
        if interval.upper > interval.lower:
            if wp.static(cell_gauss):
                integral = integrate_cells(field, ray, interval, config.shape, 0, integral, status)
                integral *= ray.length
            else:
                integral, step = integrate_midpoints(
                    field, ray, interval, config.shape, config.samples, 0, integral, status
                )
                integral *= ray.length * step
        output[pixel] = dtype(integral)
        if not wp.isfinite(output[pixel]):
            wp.atomic_or(status, 0, 2)

    return forward


@wp.func
def midpoints_inside(
    ray: Ray, interval: Interval, step: wp.float64, samples: int, shape: wp.vec3i
) -> bool:
    """Certify the sample sequence with conservative endpoint rounding margins."""
    first_position = interval.lower + wp.float64(0.5) * step
    last_position = interval.lower + (wp.float64(samples - 1) + wp.float64(0.5)) * step
    first = ray.origin + first_position * ray.direction
    last = ray.origin + last_position * ray.direction
    certified = inside_support(first, shape) and inside_support(last, shape)
    for axis in range(3):
        # Cover fused and unfused endpoint rounding relative to the monotone line.
        radius = wp.float64(1.4210854715202004e-14) * (
            wp.abs(ray.origin[axis]) + wp.abs(ray.direction[axis]) + wp.float64(1.0)
        )
        if (
            not wp.isfinite(radius)
            or wp.min(first[axis], last[axis]) < -wp.float64(0.5) + radius
            or wp.max(first[axis], last[axis]) > wp.float64(shape[axis]) - wp.float64(0.5) - radius
        ):
            certified = False
    return certified


@cache
def get_vjp(
    active_volume: bool,
    matrix_gradient: bool,
    active_pose: bool,
    *,
    per_ray: bool = False,
    ray_coordinates: bool = False,
    double_seed: bool = False,
    cell_gauss: bool = False,
    double_volume_gradient: bool = False,
    mixed: bool = False,
    storage_precision: str = "float32",
    float32_volume_increments: bool = False,
):
    if mixed and (active_pose or cell_gauss):
        raise ValueError("mixed midpoint sampling requires fixed pose")
    if ray_coordinates and (not per_ray or matrix_gradient):
        raise ValueError("ray-coordinate derivatives require per-ray six-component output")
    if per_ray and (active_volume or matrix_gradient or not active_pose):
        raise ValueError("per-ray diagnostics require only the six local pose coordinates")
    components = 12 if matrix_gradient else 6
    seed_dtype = wp.float64 if double_seed else wp.float32
    volume_dtype = wp.float64 if double_volume_gradient else wp.float32
    field_dtype = getattr(wp, storage_precision)
    scatter = get_scatter_field(double_volume_gradient, mixed, float32_volume_increments)
    scatter_cell = get_scatter_cell(double_volume_gradient, float32_volume_increments)
    reuse_coordinates = active_volume and active_pose and not (mixed or cell_gauss)
    scatter_realised = get_scatter_realised(double_volume_gradient, float32_volume_increments)
    certify_volume_support = active_volume and not (
        active_pose or mixed or cell_gauss or double_volume_gradient
    )
    scatter_certified = get_scatter_field(
        double_volume_gradient, mixed, float32_volume_increments, True, True
    )

    @wp.kernel(module="unique", module_options=OPTIONS)
    def vjp(
        field: wp.array(dtype=field_dtype),
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
            ray = ray_for_pixel(config, pose, pixel)
            if not valid_ray(ray):
                wp.atomic_or(status, 0, 2)
            else:
                interval = finite_interval(ray.origin, ray.direction, config.shape)
                span = interval.upper - interval.lower
                if seeds[pixel] == seed_dtype(0.0):
                    span = wp.float64(0.0)
                if span > wp.float64(0.0):
                    origin_gradient = wp.vec3d()
                    direction_gradient = wp.vec3d()
                    integral = wp.float64(0.0)
                    factor = wp.float64(seeds[pixel]) * ray.length
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
                            if wp.static(active_pose):
                                first, second = sample_segment(field, segment, config.shape)
                                origin_gradient += segment.radius * first.gradient
                                origin_gradient += segment.radius * second.gradient
                                direction_gradient += (
                                    segment.radius * segment.first_position * first.gradient
                                )
                                direction_gradient += (
                                    segment.radius * segment.second_position * second.gradient
                                )
                                if wp.static(ray_coordinates):
                                    integral += segment.radius * first.value
                                    integral += segment.radius * second.value
                            if wp.static(active_volume):
                                volume_factor = factor * segment.radius
                                if not wp.isfinite(factor):
                                    # The unclipped ray can be much longer than
                                    # this segment. Retain all finite factors
                                    # until its representable weight is formed.
                                    volume_factor = product3(
                                        wp.float64(seeds[pixel]), ray.length, segment.radius
                                    )
                                if segment.shared_cell:
                                    scatter_cell(
                                        volume_gradient,
                                        segment.lower,
                                        segment.first_point,
                                        segment.second_point,
                                        config.shape,
                                        volume_factor,
                                        status,
                                    )
                                else:
                                    scatter(
                                        volume_gradient,
                                        segment.first_point,
                                        config.shape,
                                        volume_factor,
                                        status,
                                    )
                                    scatter(
                                        volume_gradient,
                                        segment.second_point,
                                        config.shape,
                                        volume_factor,
                                        status,
                                    )
                            start = segment.end
                            traversal = advance_cells(
                                traversal, ray, config.shape, segment.end, interval.upper
                            )
                            segments += wp.int64(1)
                        if start < interval.upper:
                            wp.atomic_or(status, 0, 2)
                        if wp.static(active_pose):
                            # Field continuity cancels all internal moving-cell terms.
                            # The zero-extended support still has external endpoint terms.
                            lower_value = support_value(
                                field, ray.origin + interval.lower * ray.direction, config.shape
                            )
                            upper_value = support_value(
                                field, ray.origin + interval.upper * ray.direction, config.shape
                            )
                            origin_gradient += (
                                upper_value * interval.upper_origin_gradient
                                - lower_value * interval.lower_origin_gradient
                            )
                            direction_gradient += (
                                upper_value * interval.upper_direction_gradient
                                - lower_value * interval.lower_direction_gradient
                            )
                    else:
                        step = span / wp.float64(config.samples)
                        volume_factor = factor * span / wp.float64(config.samples)
                        if not wp.isfinite(factor):
                            volume_factor = product4(
                                wp.float64(seeds[pixel]),
                                ray.length,
                                span,
                                wp.float64(1.0) / wp.float64(config.samples),
                            )
                        midpoints = Midpoints()
                        if wp.static(mixed):
                            midpoints = begin_midpoints(ray, interval, config.samples)
                        mean = wp.float64(0.0)
                        direct_origin = wp.vec3d()
                        direct_direction = wp.vec3d()
                        shift_lower = wp.float64(0.0)
                        shift_upper = wp.float64(0.0)
                        power_of_two_samples = (
                            config.samples > 0 and (config.samples & (config.samples - 1)) == 0
                        )
                        inverse_samples = wp.float64(0.0)
                        if power_of_two_samples:
                            inverse_samples = wp.float64(1.0) / wp.float64(config.samples)
                        if wp.static(certify_volume_support):
                            # Clamped trilinear factors cannot enlarge a finite seed.
                            certified = wp.abs(volume_factor) <= wp.float64(
                                3.4028234663852886e38
                            ) and midpoints_inside(
                                ray, interval, step, config.samples, config.shape
                            )
                        for sample in range(config.samples):
                            # Binary scaling is exact for positive int32 powers of two.
                            # Other counts retain the original rounded division.
                            alpha = wp.float64(0.0)
                            if power_of_two_samples:
                                alpha = (wp.float64(sample) + wp.float64(0.5)) * inverse_samples
                            else:
                                alpha = (wp.float64(sample) + wp.float64(0.5)) / wp.float64(
                                    config.samples
                                )
                            position = (
                                interval.lower + (wp.float64(sample) + wp.float64(0.5)) * step
                            )
                            point = ray.origin + position * ray.direction
                            if wp.static(mixed):
                                point = mixed_midpoint_point(midpoints, sample)
                            if wp.static(certify_volume_support):
                                if not certified and not finite_point(point):
                                    wp.atomic_or(status, 0, 2)
                            else:
                                if not finite_point(point):
                                    wp.atomic_or(status, 0, 2)
                            if wp.static(reuse_coordinates):
                                supported = inside_support(point, config.shape)
                                lower = wp.vec3i()
                                fraction = wp.vec3d()
                                if supported:
                                    lower = cell_lower(point, config.shape)
                                    fraction = cell_fraction(point, lower, config.shape)
                            if wp.static(active_pose):
                                if wp.static(reuse_coordinates):
                                    value = Sample()
                                    if supported:
                                        value = sample_cell_fraction(
                                            load_cell(field, lower, config.shape),
                                            point,
                                            config.shape,
                                            fraction,
                                        )
                                else:
                                    value = sample_field(field, point, config.shape)
                                mean += value.value
                                direct_origin += value.gradient
                                direct_direction += position * value.gradient
                                along = wp.dot(value.gradient, ray.direction)
                                shift_lower += (wp.float64(1.0) - alpha) * along
                                shift_upper += alpha * along
                            if wp.static(active_volume):
                                if wp.static(reuse_coordinates):
                                    if supported:
                                        scatter_realised(
                                            volume_gradient,
                                            lower,
                                            fraction,
                                            config.shape,
                                            volume_factor,
                                            status,
                                        )
                                elif wp.static(certify_volume_support):
                                    if certified:
                                        scatter_certified(
                                            volume_gradient,
                                            point,
                                            config.shape,
                                            volume_factor,
                                            status,
                                        )
                                    else:
                                        scatter(
                                            volume_gradient,
                                            point,
                                            config.shape,
                                            volume_factor,
                                            status,
                                        )
                                else:
                                    scatter(
                                        volume_gradient,
                                        point,
                                        config.shape,
                                        volume_factor,
                                        status,
                                    )
                        if wp.static(active_pose):
                            if wp.static(ray_coordinates):
                                integral = mean * step
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
                        factor /= wp.float64(config.samples)
                    if wp.static(active_pose):
                        source_adjoint = factor * (
                            wp.transpose(config.object_to_grid) * origin_gradient
                        )
                        direction_adjoint = factor * (
                            wp.transpose(config.object_to_grid) * direction_gradient
                        )
                        if not wp.isfinite(factor):
                            # Zero or small geometric derivatives can remain
                            # representable even when seed*ray.length does not.
                            source_gradient = wp.transpose(config.object_to_grid) * origin_gradient
                            ray_gradient = wp.transpose(config.object_to_grid) * direction_gradient
                            normalisation = wp.float64(1.0)
                            if wp.static(not cell_gauss):
                                normalisation /= wp.float64(config.samples)
                            for axis in range(3):
                                source_adjoint[axis] = product4(
                                    wp.float64(seeds[pixel]),
                                    ray.length,
                                    normalisation,
                                    source_gradient[axis],
                                )
                                direction_adjoint[axis] = product4(
                                    wp.float64(seeds[pixel]),
                                    ray.length,
                                    normalisation,
                                    ray_gradient[axis],
                                )
                        if wp.static(ray_coordinates):
                            # L=|e-s| I(s,e-s). Unlike rigid pose derivatives,
                            # independent endpoint motion changes the norm.
                            rotation = unpack_rotation(pose)
                            source_world = rotation * source_adjoint
                            direction_world = rotation * direction_adjoint
                            for axis in range(3):
                                direction_world[axis] += product3(
                                    wp.float64(seeds[pixel]),
                                    integral,
                                    ray.world_direction[axis] / ray.length,
                                )
                                gradient[axis] = source_world[axis] - direction_world[axis]
                                gradient[axis + 3] = direction_world[axis]
                        elif wp.static(matrix_gradient):
                            for row in range(3):
                                for column in range(3):
                                    gradient[3 * row + column] = (
                                        ray.world_source_relative[row] * source_adjoint[column]
                                        + ray.world_direction[row] * direction_adjoint[column]
                                    )
                            translation_adjoint = -(unpack_rotation(pose) * source_adjoint)
                            for axis in range(3):
                                gradient[9 + axis] = translation_adjoint[axis]
                        else:
                            pose_adjoint = _local_right_pose_adjoint(
                                source_adjoint,
                                ray.object_source,
                                direction_adjoint,
                                ray.object_direction,
                            )
                            for axis in range(3):
                                gradient[axis] = pose_adjoint[axis]
                                gradient[axis + 3] = pose_adjoint[axis + 3]
        # The diagnostic writes the same ray derivative before any reduction.
        # Its caller owns O(6 P) storage; the production specialisation retains
        # only reduction partials and performs exactly its existing reduction.
        if wp.static(per_ray):
            if pixel < config.pixels:
                for component in range(components):
                    component_value = gradient[component]
                    partials[wp.int64(pixel) * wp.int64(components) + wp.int64(component)] = (
                        component_value
                    )
                    if not wp.isfinite(component_value):
                        wp.atomic_or(status, 0, 2)
        # All lanes, including padded rays, participate in the fixed reduction.
        elif wp.static(active_pose):
            for component in range(components):
                values = wp.tile(gradient[component])
                total = wp.tile_sum(values)
                wp.tile_store(partials, total, offset=block * components + component)

    return vjp
