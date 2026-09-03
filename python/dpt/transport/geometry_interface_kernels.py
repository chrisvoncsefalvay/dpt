"""Binary64 material-interface measures and signed collision-generator primitives.

Normals point from the negative grid side to the positive side. Each result
has status zero on success and ``HistoryStatus.NUMERICAL_FAILURE`` on failure.
The history owner must propagate a failed status before consuming any result.
Physical collision continuations are supplied at the base geometry.
"""

# Warp annotations are runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false
import warp as wp

from ._constants import NUMERICAL_FAILURE

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})
FAILURE = wp.constant(NUMERICAL_FAILURE)
EPSILON = wp.constant(wp.float64(2.220446049250313e-16))
CDF_RESOLUTION = wp.constant(wp.float64(1.4901161193847656e-8))


@wp.struct
class InterfaceGrid:
    origin: wp.vec3d
    spacing: wp.vec3d
    shape: wp.vec3i
    rotation: wp.mat33d
    translation: wp.vec3d
    origin_dot: wp.vec3d
    spacing_dot: wp.vec3d
    motion_translation: wp.vec3d
    motion_angular: wp.vec3d
    motion_pivot: wp.vec3d


@wp.struct
class InterfacePatch:
    axis: int
    face: int
    tangential_cells: wp.vec2i
    negative_cell: int
    positive_cell: int


@wp.struct
class InterfaceGauge:
    translation: wp.vec3d
    angular: wp.vec3d
    pivot: wp.vec3d
    status: int


@wp.struct
class NativeInterfacePoint:
    coordinate: wp.vec2d
    point_mm: wp.vec3d
    point_dot: wp.vec3d
    normal: wp.vec3d
    normal_dot: wp.vec3d
    area_mm2: wp.float64
    status: int


@wp.struct
class NativeInterfaceBridge:
    point: NativeInterfacePoint
    log_area_density: wp.float64
    log_geometric_importance: wp.float64
    axis_normalisers: wp.vec2d
    status: int


@wp.struct
class NativeInterfaceObservation:
    value: wp.float64
    status: int


@wp.func_native("""
#if defined(__CUDACC__)
return hypot(a, b);
#else
return __builtin_hypot(a, b);
#endif
""")
def _hypot(a: wp.float64, b: wp.float64) -> wp.float64: ...


@wp.func_native("""
const double absolute = wp::abs(value);
#if defined(__CUDACC__)
if (absolute <= height) return asinh(value / height);
const double ratio = height / absolute;
return copysign(log(absolute) - log(height) + log1p(sqrt(1.0 + ratio*ratio)), value);
#else
if (absolute <= height) return __builtin_asinh(value / height);
const double ratio = height / absolute;
return __builtin_copysign(__builtin_log(absolute) - __builtin_log(height)
    + __builtin_log1p(__builtin_sqrt(1.0 + ratio*ratio)), value);
#endif
""")
def _asinh_ratio(value: wp.float64, height: wp.float64) -> wp.float64: ...


@wp.func_native("""
if (value == 0.0) return 0.0;
const double absolute = wp::abs(value);
#if defined(__CUDACC__)
return copysign(exp(log(height) + absolute - log(2.0)) * -expm1(-2.0*absolute), value);
#else
return __builtin_copysign(__builtin_exp(__builtin_log(height) + absolute
    - __builtin_log(2.0)) * -__builtin_expm1(-2.0*absolute), value);
#endif
""")
def _scaled_sinh(height: wp.float64, value: wp.float64) -> wp.float64: ...


@wp.func_native("""
const double absolute = wp::abs(value);
#if defined(__CUDACC__)
const double next = nextafter(absolute, static_cast<double>(INFINITY));
return wp::isfinite(next) ? next - absolute : absolute - nextafter(absolute, 0.0);
#else
const double next = __builtin_nextafter(absolute, INFINITY);
return wp::isfinite(next) ? next - absolute : absolute - __builtin_nextafter(absolute, 0.0);
#endif
""")
def _ulp(value: wp.float64) -> wp.float64: ...


@wp.func_native("""
// Error-free additions retain the low parts of this three-term contraction.
const double ab = a + b;
const double bv = ab - a;
const double error_ab = (a - (ab - bv)) + (b - bv);
const double abc = ab + c;
const double cv = abc - ab;
const double error_abc = (ab - (abc - cv)) + (c - cv);
return abc + (error_ab + error_abc);
""")
def _sum3(a: wp.float64, b: wp.float64, c: wp.float64) -> wp.float64: ...


@wp.func
def _finite3(value: wp.vec3d) -> bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.func
def _tangent_axes(axis: int) -> wp.vec2i:
    result = wp.vec2i(0, 1)
    if axis == 0:
        result = wp.vec2i(1, 2)
    elif axis == 1:
        result = wp.vec2i(0, 2)
    return result


@wp.func
def _valid_patch(patch: InterfacePatch, grid: InterfaceGrid) -> bool:
    valid = patch.axis >= 0 and patch.axis < 3
    for axis in range(3):
        valid = valid and grid.shape[axis] > 0
        valid = valid and wp.isfinite(grid.origin[axis])
        valid = valid and wp.isfinite(grid.spacing[axis]) and grid.spacing[axis] > wp.float64(0.0)
    if valid:
        axes = _tangent_axes(patch.axis)
        valid = patch.face >= 0 and patch.face <= grid.shape[patch.axis]
        for i in range(2):
            valid = valid and patch.tangential_cells[i] >= 0
            valid = valid and patch.tangential_cells[i] < grid.shape[axes[i]]
        if valid:
            cell = wp.vec3i(0)
            cell[axes[0]] = patch.tangential_cells[0]
            cell[axes[1]] = patch.tangential_cells[1]
            # Explicit constructors declare mutable Warp branch variables.
            negative = int(-1)  # noqa: RUF046, UP018
            positive = int(-1)  # noqa: RUF046, UP018
            if patch.face > 0:
                cell[patch.axis] = patch.face - 1
                negative = (cell[2] * grid.shape[1] + cell[1]) * grid.shape[0] + cell[0]
            if patch.face < grid.shape[patch.axis]:
                cell[patch.axis] = patch.face
                positive = (cell[2] * grid.shape[1] + cell[1]) * grid.shape[0] + cell[0]
            valid = patch.negative_cell == negative and patch.positive_cell == positive
    return valid


@wp.func
def interface_point_tangent(
    patch: InterfacePatch, coordinate: wp.vec2d, grid: InterfaceGrid
) -> NativeInterfacePoint:
    result = NativeInterfacePoint()
    result.status = FAILURE
    valid = _valid_patch(patch, grid)
    for i in range(2):
        valid = valid and wp.isfinite(coordinate[i])
        valid = valid and coordinate[i] > wp.float64(0.0) and coordinate[i] < wp.float64(1.0)
    if valid:
        axes = _tangent_axes(patch.axis)
        indices = wp.vec3d(wp.float64(0.0))
        indices[patch.axis] = wp.float64(patch.face)
        for i in range(2):
            indices[axes[i]] = wp.float64(patch.tangential_cells[i]) + coordinate[i]
        local = grid.origin + wp.cw_mul(indices, grid.spacing)
        local_dot = grid.origin_dot + wp.cw_mul(indices, grid.spacing_dot)
        point = grid.rotation * local + grid.translation
        point_dot = grid.motion_translation + wp.cross(
            grid.motion_angular, point - grid.motion_pivot
        )
        point_dot += grid.rotation * local_dot
        axis_normal = wp.vec3d(wp.float64(0.0))
        axis_normal[patch.axis] = wp.float64(1.0)
        normal = grid.rotation * axis_normal
        normal_dot = wp.cross(grid.motion_angular, normal)
        area = grid.spacing[axes[0]] * grid.spacing[axes[1]]
        if _finite3(point) and _finite3(point_dot) and _finite3(normal) and _finite3(normal_dot):
            if wp.isfinite(area) and area > wp.float64(0.0):
                result.coordinate = coordinate
                result.point_mm = point
                result.point_dot = point_dot
                result.normal = normal
                result.normal_dot = normal_dot
                result.area_mm2 = area
                result.status = 0
    return result


@wp.func
def _direction_jet_orthogonal(
    direction: wp.vec3d, direction_dot: wp.vec3d, tangent_length: wp.float64
) -> bool:
    """Check finite unit-vector jets without an overflowing norm tolerance.

    Keep the original arithmetic whenever the robust norm is representable.
    If its finite components have an unrepresentable norm, divide the relative
    criterion by their largest magnitude. No ray jet is clipped or repaired.
    """
    if wp.isfinite(tangent_length):
        return wp.abs(wp.dot(direction, direction_dot)) <= wp.float64(1.0e-12) * wp.max(
            wp.float64(1.0), tangent_length
        )
    scale = wp.max(
        wp.abs(direction_dot[0]), wp.max(wp.abs(direction_dot[1]), wp.abs(direction_dot[2]))
    )
    scaled = wp.vec3d(direction_dot[0] / scale, direction_dot[1] / scale, direction_dot[2] / scale)
    scaled_length = _hypot(_hypot(scaled[0], scaled[1]), scaled[2])
    return wp.abs(wp.dot(direction, scaled)) <= wp.float64(1.0e-12) * scaled_length


@wp.func
def source_path_gauge(
    origin: wp.vec3d, direction: wp.vec3d, origin_dot: wp.vec3d, direction_dot: wp.vec3d
) -> InterfaceGauge:
    result = InterfaceGauge()
    result.status = FAILURE
    if (
        _finite3(origin)
        and _finite3(direction)
        and _finite3(origin_dot)
        and _finite3(direction_dot)
    ):
        length = _hypot(_hypot(direction[0], direction[1]), direction[2])
        tangent_length = _hypot(_hypot(direction_dot[0], direction_dot[1]), direction_dot[2])
        if wp.abs(length * length - wp.float64(1.0)) <= wp.float64(1.0e-12):
            if _direction_jet_orthogonal(direction, direction_dot, tangent_length):
                result.translation = origin_dot
                result.angular = wp.cross(direction, direction_dot)
                result.pivot = origin
                if _finite3(result.angular):
                    result.status = 0
    return result


@wp.func
def interface_relative_velocity(point: NativeInterfacePoint, gauge: InterfaceGauge) -> wp.float64:
    gauge_velocity = gauge.translation + wp.cross(gauge.angular, point.point_mm - gauge.pivot)
    return wp.dot(point.point_dot - gauge_velocity, point.normal)


@wp.func
def direct_interface_status(
    point: NativeInterfacePoint, incoming_direction: wp.vec3d, collimated: int, visible: int
) -> int:
    """Reject the smooth direct-fluence chart for grazing collimated illumination."""
    status = point.status
    if not _finite3(incoming_direction):
        status = FAILURE
    elif visible != 0 and collimated != 0:
        if wp.dot(incoming_direction, point.normal) == wp.float64(0.0):
            status = FAILURE
    return status


@wp.func
def sample_interface_bridge(
    patch: InterfacePatch, previous_point_mm: wp.vec3d, coordinate: wp.vec2d, grid: InterfaceGrid
) -> NativeInterfaceBridge:
    """Draw an exact product-asinh proposal per physical face area.

    ``exp(-log_area_density)`` already supplies the surface measure, so its
    caller must not multiply by ``point.area_mm2`` again. The inverse-square
    bridge multiplier is ``exp(log_geometric_importance)``. All draws and
    proposal parameters are detached from the differentiated geometry.
    """
    result = NativeInterfaceBridge()
    result.status = FAILURE
    result.point.status = FAILURE
    valid = _valid_patch(patch, grid) and _finite3(previous_point_mm)
    for i in range(2):
        valid = valid and wp.isfinite(coordinate[i])
        valid = valid and coordinate[i] > wp.float64(0.0) and coordinate[i] < wp.float64(1.0)
    if valid:
        inverse = wp.transpose(grid.rotation)
        previous = inverse * previous_point_mm - inverse * grid.translation
        face = grid.origin[patch.axis] + wp.float64(patch.face) * grid.spacing[patch.axis]
        height = wp.abs(previous[patch.axis] - face)
        valid = _finite3(previous) and wp.isfinite(height) and height > wp.float64(0.0)
        axes = _tangent_axes(patch.axis)
        amounts = wp.vec2d(wp.float64(0.0))
        normalisers = wp.vec2d(wp.float64(0.0))
        firsts = wp.vec2d(wp.float64(0.0))
        if valid:
            for i in range(2):
                axis = axes[i]
                lower = (
                    grid.origin[axis]
                    + wp.float64(patch.tangential_cells[i]) * grid.spacing[axis]
                    - previous[axis]
                )
                upper = lower + grid.spacing[axis]
                first = _asinh_ratio(lower, height)
                last = _asinh_ratio(upper, height)
                normal = last - first
                valid = valid and wp.isfinite(normal) and normal > wp.float64(0.0)
                if valid:
                    offset = _scaled_sinh(
                        height, (wp.float64(1.0) - coordinate[i]) * first + coordinate[i] * last
                    )
                    amount = (offset - lower) / grid.spacing[axis]
                    valid = valid and amount > wp.float64(0.0) and amount < wp.float64(1.0)
                    amounts[i] = amount
                    normalisers[i] = normal
                    firsts[i] = first
        if valid:
            point = interface_point_tangent(patch, amounts, grid)
            valid = point.status == 0
            represented = inverse * point.point_mm - inverse * grid.translation
            offsets = wp.vec2d(wp.float64(0.0))
            log_parts = wp.vec2d(wp.float64(0.0))
            for i in range(2):
                axis = axes[i]
                offset = represented[axis] - previous[axis]
                normal = normalisers[i]
                radius = _hypot(height, offset)
                reconstructed = (_asinh_ratio(offset, height) - firsts[i]) / normal
                uncertainty = (
                    wp.float64(8.0)
                    * (_ulp(represented[axis]) + _ulp(previous[axis]))
                    / (radius * normal)
                )
                tolerance = uncertainty + wp.float64(512.0) * EPSILON * (
                    wp.float64(1.0) + wp.abs(firsts[i]) / normal
                )
                valid = valid and wp.isfinite(uncertainty) and uncertainty <= CDF_RESOLUTION
                valid = valid and wp.abs(reconstructed - coordinate[i]) <= tolerance
                offsets[i] = offset
                log_parts[i] = wp.log(radius) + wp.log(normal)
            if valid:
                distance = _hypot(_hypot(height, offsets[0]), offsets[1])
                log_density = -(log_parts[0] + log_parts[1])
                log_importance = -wp.float64(2.0) * wp.log(distance) - log_density
                if wp.isfinite(log_density) and wp.isfinite(log_importance):
                    result.point = point
                    result.log_area_density = log_density
                    result.log_geometric_importance = log_importance
                    result.axis_normalisers = normalisers
                    result.status = 0
    return result


@wp.func
def interface_generator_observation(
    point: NativeInterfacePoint,
    gauge: InterfaceGauge,
    negative_extinction: wp.float64,
    positive_extinction: wp.float64,
    negative_collision: wp.float64,
    positive_collision: wp.float64,
    negative_present: int,
    positive_present: int,
    unchanged_direction: wp.float64,
    incident_importance: wp.float64,
) -> NativeInterfaceObservation:
    """Contract completed left/right collision and unchanged-direction scores."""
    result = NativeInterfaceObservation()
    result.status = FAILURE
    valid = point.status == 0 and gauge.status == 0
    valid = valid and wp.isfinite(negative_extinction) and negative_extinction >= wp.float64(0.0)
    valid = valid and wp.isfinite(positive_extinction) and positive_extinction >= wp.float64(0.0)
    valid = valid and (negative_present != 0 or negative_extinction == wp.float64(0.0))
    valid = valid and (positive_present != 0 or positive_extinction == wp.float64(0.0))
    if negative_present != 0:
        valid = valid and wp.isfinite(negative_collision)
    if positive_present != 0:
        valid = valid and wp.isfinite(positive_collision)
    valid = valid and wp.isfinite(unchanged_direction) and wp.isfinite(incident_importance)
    if valid:
        left = wp.float64(0.0)
        right = wp.float64(0.0)
        if negative_present != 0:
            left = negative_extinction * negative_collision
        if positive_present != 0:
            right = -positive_extinction * positive_collision
        unchanged = -(negative_extinction - positive_extinction) * unchanged_direction
        jump = _sum3(left, right, unchanged)
        value = interface_relative_velocity(point, gauge) * incident_importance * jump
        if wp.isfinite(value):
            result.value = value
            result.status = 0
    return result
