"""Attached source-coordinate side limits for cutoff material-box shadows."""

from __future__ import annotations

import warp as wp

from ._constants import NUMERICAL_FAILURE
from .compton_geometry_boundary_kernels import NativeBoundaryRay
from .geometry_interface_kernels import (
    InterfaceGrid,
    _direction_jet_orthogonal,
    _finite3,
    _hypot,
    _ulp,
)
from .geometry_source_boundary_kernels import (
    NativeAffineSourceBoundary,
    NativeSourceGeometry,
    NativeSourceRaySample,
    source_ray_tangent,
)

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})
FAILURE = wp.constant(NUMERICAL_FAILURE)
NULL = wp.constant(-1)


@wp.struct
class NativeOpaqueBoundary:
    boundary: NativeAffineSourceBoundary
    face_count: int
    axis0: int
    high0: int
    axis1: int
    high1: int


@wp.struct
class NativeOpaqueHit:
    status: int
    hit: int


@wp.struct
class NativeOpaqueSides:
    status: int
    negative_hit: int
    positive_hit: int


@wp.func
def source_coordinate_ray(
    source: NativeSourceGeometry, coordinate: wp.vec2d, coordinate_dot: wp.vec2d
) -> NativeSourceRaySample:
    """Vary original emission coordinates while holding physical geometry fixed."""
    nominal = NativeSourceGeometry()
    nominal.kind = source.kind
    nominal.origin_mm = source.origin_mm
    nominal.direction = source.direction
    nominal.extent = source.extent
    nominal.target_origin = source.target_origin
    nominal.target_u = source.target_u
    nominal.target_v = source.target_v
    nominal.target_shape = source.target_shape
    nominal.target_spacing = source.target_spacing
    nominal.weight = source.weight
    sample = source_ray_tangent(nominal, coordinate)
    if sample.status == 0:
        valid = wp.isfinite(coordinate_dot[0]) and wp.isfinite(coordinate_dot[1])
        if source.kind == 0:
            sample.ray.origin_dot = wp.vec3d(
                coordinate_dot[0] * source.extent[0],
                coordinate_dot[1] * source.extent[1],
                wp.float64(0.0),
            )
        else:
            displacement = (
                source.target_origin
                + source.target_u
                * (
                    (coordinate[0] * wp.float64(source.target_shape[0]) - wp.float64(0.5))
                    * source.target_spacing[0]
                )
                + source.target_v
                * (
                    (coordinate[1] * wp.float64(source.target_shape[1]) - wp.float64(0.5))
                    * source.target_spacing[1]
                )
                - source.origin_mm
            )
            displacement_dot = source.target_u * (
                coordinate_dot[0] * wp.float64(source.target_shape[0]) * source.target_spacing[0]
            ) + source.target_v * (
                coordinate_dot[1] * wp.float64(source.target_shape[1]) * source.target_spacing[1]
            )
            length = _hypot(_hypot(displacement[0], displacement[1]), displacement[2])
            valid = valid and wp.isfinite(length) and length > wp.float64(0.0)
            if valid:
                sample.ray.direction_dot = (
                    displacement_dot
                    - sample.ray.direction * wp.dot(sample.ray.direction, displacement_dot)
                ) * (wp.float64(1.0) / length)
        if (
            not valid
            or not _finite3(sample.ray.origin_dot)
            or not _finite3(sample.ray.direction_dot)
        ):
            sample.status = FAILURE
    return sample


@wp.func
def opaque_material_hit(
    ray: NativeBoundaryRay,
    grid: InterfaceGrid,
    side: int,
    face_count: int,
    axis0: int,
    high0: int,
    axis1: int,
    high1: int,
) -> NativeOpaqueHit:
    """Half-open forward box hit with exact lexicographic attached-face limits."""
    result = NativeOpaqueHit()
    result.status = FAILURE
    valid = (
        (side == -1 or side == 0 or side == 1)
        and face_count >= 0
        and face_count <= 2
        and _finite3(ray.origin_mm)
        and _finite3(ray.direction)
        and _finite3(ray.origin_dot)
        and _finite3(ray.direction_dot)
        and _finite3(grid.origin)
        and _finite3(grid.spacing)
        and _finite3(grid.translation)
    )
    tangent_length = _hypot(
        _hypot(ray.direction_dot[0], ray.direction_dot[1]), ray.direction_dot[2]
    )
    valid = valid and wp.abs(wp.dot(ray.direction, ray.direction) - wp.float64(1.0)) <= wp.float64(
        1.0e-12
    )
    valid = valid and _direction_jet_orthogonal(ray.direction, ray.direction_dot, tangent_length)
    if face_count >= 1:
        valid = valid and axis0 >= 0 and axis0 < 3 and (high0 == 0 or high0 == 1)
    if face_count == 2:
        valid = valid and axis1 >= 0 and axis1 < 3 and axis0 != axis1 and (high1 == 0 or high1 == 1)
    upper = wp.vec3d()
    for axis in range(3):
        upper[axis] = grid.origin[axis] + wp.float64(grid.shape[axis]) * grid.spacing[axis]
        valid = valid and grid.shape[axis] > 0 and grid.spacing[axis] > wp.float64(0.0)
        valid = valid and wp.isfinite(upper[axis]) and upper[axis] > grid.origin[axis]
        for other in range(3):
            valid = valid and wp.isfinite(grid.rotation[axis, other])
    if valid:
        # A proper rigid pose is required even when the ray would miss the box.
        gram = wp.transpose(grid.rotation) * grid.rotation
        for axis in range(3):
            for other in range(3):
                target = wp.float64(0.0)
                if axis == other:
                    target = wp.float64(1.0)
                valid = valid and wp.abs(gram[axis, other] - target) <= wp.float64(1.0e-10)
        valid = valid and wp.determinant(grid.rotation) > wp.float64(0.0)
    if valid:
        inverse = wp.transpose(grid.rotation)
        origin = inverse * (ray.origin_mm - grid.translation)
        direction = inverse * ray.direction
        origin_dot = inverse * ray.origin_dot
        direction_dot = inverse * ray.direction_dot
        valid = (
            _finite3(origin)
            and _finite3(direction)
            and _finite3(origin_dot)
            and _finite3(direction_dot)
        )
        scale = wp.float64(0.0)
        for axis in range(3):
            scale = wp.max(scale, wp.abs(ray.origin_mm[axis]))
            scale = wp.max(scale, wp.abs(origin[axis]))
            scale = wp.max(scale, wp.abs(grid.origin[axis]))
            scale = wp.max(scale, wp.abs(grid.spacing[axis]))
            scale = wp.max(scale, wp.abs(upper[axis]))
            scale = wp.max(scale, wp.abs(grid.translation[axis]))
        resolution = wp.float64(512.0) * _ulp(scale)
        anchor = wp.float64(0.0)
        if valid and face_count == 2:
            if direction[axis0] == wp.float64(0.0) or direction[axis1] == wp.float64(0.0):
                valid = False
            else:
                location0 = grid.origin[axis0]
                location1 = grid.origin[axis1]
                if high0 == 1:
                    location0 = upper[axis0]
                if high1 == 1:
                    location1 = upper[axis1]
                anchor = (location0 - origin[axis0]) / direction[axis0]
                second = (location1 - origin[axis1]) / direction[axis1]
                root_resolution = wp.max(
                    resolution / wp.abs(direction[axis0]), resolution / wp.abs(direction[axis1])
                )
                root_resolution = wp.max(root_resolution, wp.float64(512.0) * _ulp(anchor))
                root_resolution = wp.max(root_resolution, wp.float64(512.0) * _ulp(second))
                valid = (
                    wp.isfinite(anchor)
                    and wp.isfinite(second)
                    and wp.abs(anchor - second) <= root_resolution
                )
        near = wp.float64(0.0)
        near_dot = wp.float64(0.0)
        far = wp.float64(wp.inf)
        far_dot = wp.float64(0.0)
        miss = bool(False)  # noqa: UP018 -- Warp requires a dynamic variable before the loop.
        for axis in range(3):
            if valid and not miss:
                lo = grid.origin[axis]
                hi = upper[axis]
                o = origin[axis]
                d = direction[axis]
                od = origin_dot[axis]
                dd = direction_dot[axis]
                if face_count == 1 and axis == axis0:
                    location = lo
                    if high0 == 1:
                        location = hi
                    valid = wp.abs(o - location) <= resolution
                    o = location
                if valid:
                    if d == wp.float64(0.0):
                        if o < lo or o > hi:
                            miss = True
                        elif o == lo and wp.float64(side) * od < wp.float64(0.0):
                            miss = True
                        elif o == hi and (side == 0 or wp.float64(side) * od >= wp.float64(0.0)):
                            miss = True
                        elif side != 0 and dd != wp.float64(0.0) and (o == lo or o == hi):
                            valid = False
                    else:
                        first = (lo - o) / d
                        second = (hi - o) / d
                        if face_count == 2:
                            if (axis == axis0 and high0 == 0) or (axis == axis1 and high1 == 0):
                                first = anchor
                            if (axis == axis0 and high0 == 1) or (axis == axis1 and high1 == 1):
                                second = anchor
                        first_dot = wp.float64(side) * ((-od - first * dd) / d)
                        second_dot = wp.float64(side) * ((-od - second * dd) / d)
                        valid = (
                            wp.isfinite(first)
                            and wp.isfinite(second)
                            and wp.isfinite(first_dot)
                            and wp.isfinite(second_dot)
                        )
                        if first > second or (first == second and first_dot > second_dot):
                            temporary = first
                            temporary_dot = first_dot
                            first = second
                            first_dot = second_dot
                            second = temporary
                            second_dot = temporary_dot
                        if first > near or (first == near and first_dot > near_dot):
                            near = first
                            near_dot = first_dot
                        if second < far or (second == far and second_dot < far_dot):
                            far = second
                            far_dot = second_dot
        if valid:
            result.status = 0
            if not miss and (far > near or (far == near and side != 0 and far_dot > near_dot)):
                result.hit = 1
    return result


@wp.func
def opaque_boundary_sides(
    source: NativeSourceGeometry,
    coordinate: wp.vec2d,
    record: NativeOpaqueBoundary,
    grid: InterfaceGrid,
) -> NativeOpaqueSides:
    result = NativeOpaqueSides()
    result.status = FAILURE
    norm = _hypot(record.boundary.a, record.boundary.b)
    if record.boundary.present == 0:
        result.status = NULL
    elif (
        record.boundary.present == 1
        and (record.face_count == 1 or record.face_count == 2)
        and wp.isfinite(norm)
        and norm > wp.float64(0.0)
    ):
        sample = source_coordinate_ray(
            source, coordinate, wp.vec2d(record.boundary.a / norm, record.boundary.b / norm)
        )
        if sample.status == 0:
            negative = opaque_material_hit(
                sample.ray,
                grid,
                -1,
                record.face_count,
                record.axis0,
                record.high0,
                record.axis1,
                record.high1,
            )
            positive = opaque_material_hit(
                sample.ray,
                grid,
                1,
                record.face_count,
                record.axis0,
                record.high0,
                record.axis1,
                record.high1,
            )
            if negative.status == 0 and positive.status == 0:
                result.status = 0
                result.negative_hit = negative.hit
                result.positive_hit = positive.hit
    return result
