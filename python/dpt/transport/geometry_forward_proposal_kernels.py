"""Full-disc detector proposals for zero-binding bound-Compton geometry jets.

The radial density is proportional to rho**(-3/2) and is mixed with the full
regular detector proposal. Status zero is a completed sample, minus one is an
original null draw and four invalidates the history. No draw is retried.
"""

# Warp annotations name runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false
import warp as wp

from ._constants import NUMERICAL_FAILURE
from .geometry_detector_proposal_kernels import (
    DetectorProposalGeometry,
    DetectorProposalPlane,
    sample_detector_area_proposal_from_vertex,
)

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})
FAILURE = wp.constant(NUMERICAL_FAILURE)
EPSILON = wp.constant(wp.float64(2.220446049250313e-16))
RESOLUTION = wp.constant(wp.float64(1.4901161193847656e-8))
TWO_PI = wp.constant(wp.float64(6.283185307179586476925286766559))


@wp.struct
class ForwardProposalGeometry:
    regular: DetectorProposalGeometry
    detector_normal: wp.vec3d


@wp.struct
class NativeForwardDisc:
    center_mm: wp.vec3d
    center_coordinate: wp.vec2d
    radius_mm: wp.float64
    extent_mm: wp.vec2d
    area_mm2: wp.float64
    present: int
    status: int


@wp.struct
class NativeForwardAreaSample:
    coordinate: wp.vec2d
    log_reference_density: wp.float64
    component: int
    status: int


@wp.func_native("""
#if defined(__CUDACC__)
return hypot(a, b);
#else
return __builtin_hypot(a, b);
#endif
""")
def _hypot(a: wp.float64, b: wp.float64) -> wp.float64: ...


@wp.func
def _finite3(value: wp.vec3d) -> bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.func
def prepare_forward_disc(
    previous_point_mm: wp.vec3d,
    incoming_direction: wp.vec3d,
    geometry: ForwardProposalGeometry,
) -> NativeForwardDisc:
    """Freeze a detached covering disc at the current physical collision ray."""
    result = NativeForwardDisc()
    result.status = FAILURE
    regular = geometry.regular
    if (
        not _finite3(previous_point_mm)
        or not _finite3(incoming_direction)
        or not _finite3(geometry.detector_normal)
        or wp.abs(wp.dot(incoming_direction, incoming_direction) - wp.float64(1.0))
        > wp.float64(1e-12)
    ):
        return result
    extent = wp.vec2d(
        wp.float64(regular.detector_shape[0]) * regular.detector_spacing[0],
        wp.float64(regular.detector_shape[1]) * regular.detector_spacing[1],
    )
    area = extent[0] * extent[1]
    if (
        not wp.isfinite(area)
        or area <= wp.float64(0.0)
        or not wp.isfinite(extent[0])
        or not wp.isfinite(extent[1])
        or extent[0] <= wp.float64(0.0)
        or extent[1] <= wp.float64(0.0)
    ):
        return result
    denominator = wp.dot(incoming_direction, geometry.detector_normal)
    result.status = 0
    if denominator <= wp.float64(0.0):
        return result
    height = wp.dot(regular.detector_lower - previous_point_mm, geometry.detector_normal)
    distance = height / denominator
    center = previous_point_mm + incoming_direction * distance
    offset = center - regular.detector_lower
    coordinate = wp.vec2d(
        wp.dot(offset, regular.detector_u) / extent[0],
        wp.dot(offset, regular.detector_v) / extent[1],
    )
    radius = wp.float64(0.0)
    for x in range(2):
        for y in range(2):
            value = _hypot(
                extent[0] * (wp.float64(x) - coordinate[0]),
                extent[1] * (wp.float64(y) - coordinate[1]),
            )
            radius = wp.max(radius, value)
    radius = wp.float64(2.0) * radius
    if (
        not wp.isfinite(distance)
        or distance <= wp.float64(0.0)
        or not _finite3(center)
        or not wp.isfinite(coordinate[0])
        or not wp.isfinite(coordinate[1])
        or not wp.isfinite(radius)
        or radius <= wp.float64(0.0)
        or not wp.isfinite(area)
        or area <= wp.float64(0.0)
        or not wp.isfinite(extent[0])
        or not wp.isfinite(extent[1])
        or extent[0] <= wp.float64(0.0)
        or extent[1] <= wp.float64(0.0)
    ):
        result.status = FAILURE
        return result
    result.center_mm = center
    result.center_coordinate = coordinate
    result.radius_mm = radius
    result.extent_mm = extent
    result.area_mm2 = area
    result.present = 1
    return result


@wp.func
def sample_forward_area_mixture(
    planes: wp.array(dtype=DetectorProposalPlane),
    offset: int,
    count: int,
    geometry: ForwardProposalGeometry,
    previous_point_mm: wp.vec3d,
    incoming_direction: wp.vec3d,
    component_draw: wp.float64,
    regular_component_draw: wp.float64,
    coordinate: wp.vec2d,
) -> NativeForwardAreaSample:
    """Retain full azimuth, outside-rectangle nulls and both mixture densities."""
    result = NativeForwardAreaSample()
    result.status = FAILURE
    valid = offset >= 0 and count >= 0 and offset <= planes.shape[0]
    valid = valid and count <= planes.shape[0] - offset
    valid = (
        valid
        and wp.isfinite(component_draw)
        and component_draw > wp.float64(0.0)
        and component_draw < wp.float64(1.0)
    )
    for axis in range(2):
        valid = valid and wp.isfinite(coordinate[axis])
        valid = valid and coordinate[axis] > wp.float64(0.0) and coordinate[axis] < wp.float64(1.0)
    if not valid:
        return result
    # A full-disc miss is a null only after the entire detached plan is admitted.
    # In particular, null draws cannot hide a collision exactly on a grid plane.
    previous_local = (
        geometry.regular.inverse_rotation * previous_point_mm + geometry.regular.inverse_translation
    )
    if not _finite3(previous_local):
        return result
    for index in range(count):
        plane = planes[offset + index]
        if (
            plane.axis < 0
            or plane.axis >= 3
            or not wp.isfinite(plane.position_mm)
            or not _finite3(plane.coefficients)
            or wp.max(wp.abs(plane.coefficients[0]), wp.abs(plane.coefficients[1]))
            == wp.float64(0.0)
        ):
            return result
        separation = wp.abs(previous_local[plane.axis] - plane.position_mm)
        if not wp.isfinite(separation) or separation <= wp.float64(0.0):
            return result
    disc = prepare_forward_disc(previous_point_mm, incoming_direction, geometry)
    if disc.status != 0:
        return result
    mapped = coordinate
    regular_draw = regular_component_draw
    if disc.present != 0 and component_draw >= wp.float64(0.5):
        result.component = 1
        radius = (disc.radius_mm * coordinate[0]) * coordinate[0]
        if not wp.isfinite(radius) or radius <= wp.float64(0.0):
            return result
        angle = TWO_PI * coordinate[1]
        mapped = disc.center_coordinate + wp.vec2d(
            radius * wp.cos(angle) / disc.extent_mm[0], radius * wp.sin(angle) / disc.extent_mm[1]
        )
        if not wp.isfinite(mapped[0]) or not wp.isfinite(mapped[1]):
            return result
        if (
            mapped[0] <= wp.float64(0.0)
            or mapped[0] >= wp.float64(1.0)
            or mapped[1] <= wp.float64(0.0)
            or mapped[1] >= wp.float64(1.0)
        ):
            result.status = -1
            return result
        delta = wp.cw_mul(disc.extent_mm, mapped - disc.center_coordinate)
        actual_radius = _hypot(delta[0], delta[1])
        radial_error = wp.abs(wp.sqrt(actual_radius / disc.radius_mm) - coordinate[0])
        actual_angle = wp.atan2(delta[1], delta[0]) / TWO_PI
        actual_angle -= wp.floor(actual_angle)
        angular_error = wp.abs(actual_angle - coordinate[1])
        angular_error = wp.min(angular_error, wp.float64(1.0) - angular_error)
        if (
            not wp.isfinite(radial_error)
            or radial_error > wp.float64(512.0) * EPSILON
            or not wp.isfinite(angular_error)
            or angular_error > RESOLUTION
        ):
            return result
        regular_draw = wp.float64(0.25)
    regular = sample_detector_area_proposal_from_vertex(
        planes, offset, count, geometry.regular, previous_point_mm, regular_draw, mapped
    )
    if regular.status != 0:
        return result
    mapped = regular.coordinate
    log_density = regular.log_reference_density
    if disc.present != 0:
        delta = wp.cw_mul(disc.extent_mm, mapped - disc.center_coordinate)
        radius = _hypot(delta[0], delta[1])
        regular_geometry = geometry.regular
        amount_x = (
            mapped[0] * wp.float64(regular_geometry.detector_shape[0])
        ) * regular_geometry.detector_spacing[0]
        amount_y = (
            mapped[1] * wp.float64(regular_geometry.detector_shape[1])
        ) * regular_geometry.detector_spacing[1]
        endpoint = regular_geometry.detector_lower + (
            regular_geometry.detector_u * amount_x + regular_geometry.detector_v * amount_y
        )
        displacement = endpoint - disc.center_mm
        actual = wp.vec2d(
            wp.dot(displacement, regular_geometry.detector_u),
            wp.dot(displacement, regular_geometry.detector_v),
        )
        represented_radius = _hypot(actual[0], actual[1])
        discrepancy = _hypot(actual[0] - delta[0], actual[1] - delta[1]) / wp.max(
            radius, represented_radius
        )
        if (
            radius <= wp.float64(0.0)
            or represented_radius <= wp.float64(0.0)
            or not wp.isfinite(represented_radius)
            or not wp.isfinite(discrepancy)
            or discrepancy > RESOLUTION
        ):
            return result
        if radius <= disc.radius_mm:
            radial = (
                wp.log(disc.area_mm2)
                - wp.log(wp.float64(2.0) * TWO_PI)
                - wp.float64(0.5) * wp.log(disc.radius_mm)
                - wp.float64(1.5) * wp.log(radius)
            )
            scale = wp.max(log_density, radial)
            log_density = (
                wp.log(wp.float64(0.5))
                + scale
                + wp.log(wp.exp(log_density - scale) + wp.exp(radial - scale))
            )
        else:
            log_density += wp.log(wp.float64(0.5))
    if wp.isfinite(log_density):
        result.coordinate = mapped
        result.log_reference_density = log_density
        result.status = 0
    return result
