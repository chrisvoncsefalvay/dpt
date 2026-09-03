"""Attached source rays and affine unit-square boundary coarea primitives.

Status 0 is a regular primitive, -1 a null boundary chart and 4 a fatal original
history. The driver contracts the positive-minus-negative complete source-ray
continuation. The source weight and original source measure remain explicit.
"""

# Warp annotations are runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false
# pyright: reportPrivateUsage=false
import warp as wp

from ._constants import NUMERICAL_FAILURE
from .compton_geometry_boundary_kernels import NativeBoundaryRay
from .geometry_interface_kernels import _direction_jet_orthogonal, _finite3, _hypot, _sum3

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})
FAILURE = wp.constant(NUMERICAL_FAILURE)
NULL = wp.constant(-1)


@wp.struct
class NativeSourceGeometry:
    kind: int
    origin_mm: wp.vec3d
    direction: wp.vec3d
    extent: wp.vec2d
    motion_translation: wp.vec3d
    motion_angular: wp.vec3d
    motion_pivot: wp.vec3d
    extent_dot: wp.vec2d
    direction_dot: wp.vec3d
    target_origin: wp.vec3d
    target_u: wp.vec3d
    target_v: wp.vec3d
    target_shape: wp.vec2i
    target_spacing: wp.vec2d
    target_motion_translation: wp.vec3d
    target_motion_angular: wp.vec3d
    target_motion_pivot: wp.vec3d
    target_spacing_dot: wp.vec2d
    weight: wp.float64


@wp.struct
class NativeSourceRaySample:
    ray: NativeBoundaryRay
    source_coordinate: wp.vec2d
    source_weight: wp.float64
    status: int


@wp.struct
class NativeAffineSourceBoundary:
    a: wp.float64
    b: wp.float64
    c: wp.float64
    a_dot: wp.float64
    b_dot: wp.float64
    c_dot: wp.float64
    axis: int
    face: int
    family: int
    present: int


@wp.struct
class NativeSourceBoundarySample:
    coordinate: wp.vec2d
    coarea_weight: wp.float64
    levelset_dot: wp.float64
    line_length: wp.float64
    axis: int
    face: int
    family: int
    status: int


@wp.func
def source_ray_tangent(source: NativeSourceGeometry, coordinate: wp.vec2d) -> NativeSourceRaySample:
    """Launch a full attached ray at the fixed original source coordinate."""
    result = NativeSourceRaySample()
    result.status = FAILURE
    result.source_coordinate = coordinate
    result.source_weight = source.weight
    valid = (
        wp.isfinite(coordinate[0])
        and wp.isfinite(coordinate[1])
        and coordinate[0] >= wp.float64(0.0)
        and coordinate[0] <= wp.float64(1.0)
        and coordinate[1] >= wp.float64(0.0)
        and coordinate[1] <= wp.float64(1.0)
        and _finite3(source.origin_mm)
        and _finite3(source.motion_translation)
        and _finite3(source.motion_angular)
        and _finite3(source.motion_pivot)
        and wp.isfinite(source.weight)
        and source.weight >= wp.float64(0.0)
    )
    ray = NativeBoundaryRay()
    if valid:
        if source.kind == 0:
            valid = (
                _finite3(source.direction)
                and _finite3(source.direction_dot)
                and wp.isfinite(source.extent[0])
                and wp.isfinite(source.extent[1])
                and source.extent[0] >= wp.float64(0.0)
                and source.extent[1] >= wp.float64(0.0)
                and wp.isfinite(source.extent_dot[0])
                and wp.isfinite(source.extent_dot[1])
            )
            if valid:
                ray.origin_mm = source.origin_mm + wp.vec3d(
                    coordinate[0] * source.extent[0],
                    coordinate[1] * source.extent[1],
                    wp.float64(0.0),
                )
                ray.direction = source.direction
                ray.origin_dot = source.motion_translation + (
                    wp.cross(source.motion_angular, ray.origin_mm - source.motion_pivot)
                    + wp.vec3d(
                        coordinate[0] * source.extent_dot[0],
                        coordinate[1] * source.extent_dot[1],
                        wp.float64(0.0),
                    )
                )
                ray.direction_dot = (
                    wp.cross(source.motion_angular, source.direction) + source.direction_dot
                )
        elif source.kind == 1:
            valid = (
                _finite3(source.target_origin)
                and _finite3(source.target_u)
                and _finite3(source.target_v)
                and _finite3(source.target_motion_translation)
                and _finite3(source.target_motion_angular)
                and _finite3(source.target_motion_pivot)
                and source.target_shape[0] > 0
                and source.target_shape[1] > 0
                and wp.isfinite(source.target_spacing[0])
                and wp.isfinite(source.target_spacing[1])
                and source.target_spacing[0] > wp.float64(0.0)
                and source.target_spacing[1] > wp.float64(0.0)
                and wp.isfinite(source.target_spacing_dot[0])
                and wp.isfinite(source.target_spacing_dot[1])
            )
            if valid:
                amount_x = coordinate[0] * wp.float64(source.target_shape[0]) - wp.float64(0.5)
                amount_y = coordinate[1] * wp.float64(source.target_shape[1]) - wp.float64(0.5)
                target = source.target_origin + (
                    source.target_u * (amount_x * source.target_spacing[0])
                    + source.target_v * (amount_y * source.target_spacing[1])
                )
                ray.origin_mm = source.origin_mm
                ray.origin_dot = source.motion_translation + wp.cross(
                    source.motion_angular, ray.origin_mm - source.motion_pivot
                )
                target_dot = source.target_motion_translation + (
                    wp.cross(source.target_motion_angular, target - source.target_motion_pivot)
                    + (
                        source.target_u * (amount_x * source.target_spacing_dot[0])
                        + source.target_v * (amount_y * source.target_spacing_dot[1])
                    )
                )
                displacement = target - ray.origin_mm
                displacement_dot = target_dot - ray.origin_dot
                length = _hypot(_hypot(displacement[0], displacement[1]), displacement[2])
                valid = (
                    _finite3(target)
                    and _finite3(target_dot)
                    and _finite3(displacement_dot)
                    and wp.isfinite(length)
                    and length > wp.float64(0.0)
                )
                if valid:
                    ray.direction = displacement * (wp.float64(1.0) / length)
                    ray.direction_dot = (
                        displacement_dot - ray.direction * wp.dot(ray.direction, displacement_dot)
                    ) * (wp.float64(1.0) / length)
        else:
            valid = False
        direction_dot_length = _hypot(
            _hypot(ray.direction_dot[0], ray.direction_dot[1]), ray.direction_dot[2]
        )
        if (
            valid
            and _finite3(ray.origin_mm)
            and _finite3(ray.direction)
            and _finite3(ray.origin_dot)
            and _finite3(ray.direction_dot)
            and wp.abs(wp.dot(ray.direction, ray.direction) - wp.float64(1.0))
            <= wp.float64(1.0e-12)
            and _direction_jet_orthogonal(ray.direction, ray.direction_dot, direction_dot_length)
        ):
            result.ray = ray
            result.status = 0
    return result


@wp.func
def sample_source_boundary(
    boundary: NativeAffineSourceBoundary, coordinate: wp.float64
) -> NativeSourceBoundarySample:
    """Sample the clipped affine line with its signed unit-square coarea weight.

    Endpoints follow the frozen host chart's edge order and exact deduplication.
    An active whole-edge coincidence, underresolved open mark, nonfinite
    intermediate or nonzero weight that underflows is fatal. No epsilon,
    clipped draw or accepted redraw is used.
    The driver multiplies the supplied positive-minus-negative continuation jump
    and the unchanged source weight once; there is no extra face-area factor.
    """
    result = NativeSourceBoundarySample()
    result.status = FAILURE
    result.axis = boundary.axis
    result.face = boundary.face
    result.family = boundary.family
    valid = (
        wp.isfinite(coordinate)
        and coordinate > wp.float64(0.0)
        and coordinate < wp.float64(1.0)
        and boundary.face >= 0
        and boundary.axis >= 0
        and (
            (boundary.family == 0 and boundary.axis <= 2)
            or (boundary.family == 1 and boundary.axis <= 1)
        )
        and (boundary.present == 0 or boundary.present == 1)
    )
    if valid:
        if boundary.present == 0:
            result.status = NULL
        else:
            valid = (
                wp.isfinite(boundary.a)
                and wp.isfinite(boundary.b)
                and wp.isfinite(boundary.c)
                and wp.isfinite(boundary.a_dot)
                and wp.isfinite(boundary.b_dot)
                and wp.isfinite(boundary.c_dot)
                and (boundary.a != wp.float64(0.0) or boundary.b != wp.float64(0.0))
            )
            if valid:
                points_u = wp.vec4d()
                points_v = wp.vec4d()
                count = int(0)  # noqa: UP018, RUF046 -- mutable Warp loop variable
                for edge in range(4):
                    point_x = wp.float64(0.0)
                    point_y = wp.float64(0.0)
                    intersects = bool(False)  # noqa: UP018
                    if edge < 2 and boundary.b != wp.float64(0.0):
                        if edge == 1:
                            point_x = wp.float64(1.0)
                        point_y = -(boundary.a * point_x + boundary.c) / boundary.b
                        valid = valid and wp.isfinite(point_y)
                        intersects = point_y >= wp.float64(0.0) and point_y <= wp.float64(1.0)
                    elif edge >= 2 and boundary.a != wp.float64(0.0):
                        if edge == 3:
                            point_y = wp.float64(1.0)
                        point_x = -(boundary.b * point_y + boundary.c) / boundary.a
                        valid = valid and wp.isfinite(point_x)
                        intersects = point_x >= wp.float64(0.0) and point_x <= wp.float64(1.0)
                    if intersects:
                        duplicate = bool(False)  # noqa: UP018
                        for index in range(count):
                            if point_x == points_u[index] and point_y == points_v[index]:
                                duplicate = True
                        if not duplicate:
                            points_u[count] = point_x
                            points_v[count] = point_y
                            count = count + 1
                if valid:
                    if count < 2:
                        result.status = NULL
                    elif count == 2:
                        whole_edge = (
                            boundary.a == wp.float64(0.0)
                            and (
                                -boundary.c / boundary.b == wp.float64(0.0)
                                or -boundary.c / boundary.b == wp.float64(1.0)
                            )
                        ) or (
                            boundary.b == wp.float64(0.0)
                            and (
                                -boundary.c / boundary.a == wp.float64(0.0)
                                or -boundary.c / boundary.a == wp.float64(1.0)
                            )
                        )
                        if whole_edge:
                            if (
                                boundary.a_dot == wp.float64(0.0)
                                and boundary.b_dot == wp.float64(0.0)
                                and boundary.c_dot == wp.float64(0.0)
                            ):
                                result.status = NULL
                        else:
                            first = wp.vec2d(points_u[0], points_v[0])
                            last = wp.vec2d(points_u[count - 1], points_v[count - 1])
                            result.coordinate = first + coordinate * (last - first)
                            length = _hypot(last[0] - first[0], last[1] - first[1])
                            product_x = boundary.a_dot * result.coordinate[0]
                            product_y = boundary.b_dot * result.coordinate[1]
                            levelset_dot = _sum3(product_x, product_y, boundary.c_dot)
                            gradient_scale = wp.max(wp.abs(boundary.a), wp.abs(boundary.b))
                            norm = _hypot(boundary.a / gradient_scale, boundary.b / gradient_scale)
                            quotient = levelset_dot / gradient_scale
                            weight = (length / norm) * quotient
                            valid = (
                                wp.isfinite(result.coordinate[0])
                                and wp.isfinite(result.coordinate[1])
                                and result.coordinate[0] >= wp.float64(0.0)
                                and result.coordinate[0] <= wp.float64(1.0)
                                and result.coordinate[1] >= wp.float64(0.0)
                                and result.coordinate[1] <= wp.float64(1.0)
                                and not (
                                    result.coordinate[0] == first[0]
                                    and result.coordinate[1] == first[1]
                                )
                                and not (
                                    result.coordinate[0] == last[0]
                                    and result.coordinate[1] == last[1]
                                )
                                and wp.isfinite(length)
                                and length > wp.float64(0.0)
                                and wp.isfinite(product_x)
                                and wp.isfinite(product_y)
                                and wp.isfinite(levelset_dot)
                                and wp.isfinite(quotient)
                                and wp.isfinite(weight)
                                and (weight != wp.float64(0.0) or levelset_dot == wp.float64(0.0))
                                and (
                                    product_x != wp.float64(0.0)
                                    or boundary.a_dot == wp.float64(0.0)
                                    or result.coordinate[0] == wp.float64(0.0)
                                )
                                and (
                                    product_y != wp.float64(0.0)
                                    or boundary.b_dot == wp.float64(0.0)
                                    or result.coordinate[1] == wp.float64(0.0)
                                )
                            )
                            if valid:
                                result.line_length = length
                                result.levelset_dot = levelset_dot
                                result.coarea_weight = weight
                                result.status = 0
    return result
