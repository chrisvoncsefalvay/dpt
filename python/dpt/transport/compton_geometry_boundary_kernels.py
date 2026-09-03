"""Full-azimuth free-Compton energy-boundary samples on an attached detector.

Status zero supplies a regular coarea sample, status -1 is a null proposal draw
and status 4 invalidates the complete original history. Null draws remain in
the original denominator. A driver contracts the completed higher-energy minus
lower-energy continuation jump using either world-area or reference-area weight.
"""

# Warp annotations are runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false
# The shared primitives are internal to the native transport modules.
# pyright: reportPrivateUsage=false
import warp as wp

from ._constants import ELECTRON_REST_ENERGY_KEV, NUMERICAL_FAILURE
from .geometry_interface_kernels import _finite3, _hypot

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})
MASS = wp.constant(wp.float64(ELECTRON_REST_ENERGY_KEV))
TWO_PI = wp.constant(wp.float64(6.283185307179586476925286766559))
FAILURE = wp.constant(NUMERICAL_FAILURE)
NULL = wp.constant(-1)


@wp.struct
class NativeBoundaryRay:
    origin_mm: wp.vec3d
    direction: wp.vec3d
    origin_dot: wp.vec3d
    direction_dot: wp.vec3d


@wp.struct
class NativeFreeComptonBoundary:
    incoming: NativeBoundaryRay
    incident_energy: wp.float64
    threshold_energy: wp.float64
    incident_energy_dot: wp.float64
    threshold_energy_dot: wp.float64


@wp.struct
class NativeBoundaryDetector:
    corner: wp.vec3d
    u: wp.vec3d
    v: wp.vec3d
    normal: wp.vec3d
    shape: wp.vec2i
    spacing: wp.vec2d
    motion_translation: wp.vec3d
    motion_angular: wp.vec3d
    motion_pivot: wp.vec3d
    corner_correction: wp.vec3d
    spacing_dot: wp.vec2d


@wp.struct
class NativeBoundaryEndpoint:
    point_mm: wp.vec3d
    point_dot: wp.vec3d
    normal: wp.vec3d
    normal_dot: wp.vec3d
    pixel: int
    area_jacobian: wp.float64
    area_jacobian_dot: wp.float64


@wp.struct
class NativeComptonBoundarySample:
    coordinate: wp.vec2d
    endpoint: NativeBoundaryEndpoint
    outgoing: NativeBoundaryRay
    outgoing_energy: wp.float64
    outgoing_energy_dot: wp.float64
    levelset_dot: wp.float64
    world_area_weight: wp.float64
    reference_area_weight: wp.float64
    status: int


@wp.func
def free_compton_boundary_cosine(boundary: NativeFreeComptonBoundary) -> wp.vec2d:
    """Stable threshold cone cosine and its incident/threshold energy tangent."""
    cosine = (
        wp.float64(1.0)
        - MASS
        * ((boundary.incident_energy - boundary.threshold_energy) / boundary.incident_energy)
        / boundary.threshold_energy
    )
    cosine_dot = MASS * (
        (boundary.threshold_energy_dot / boundary.threshold_energy) / boundary.threshold_energy
        - (boundary.incident_energy_dot / boundary.incident_energy) / boundary.incident_energy
    )
    return wp.vec2d(cosine, cosine_dot)


@wp.func
def _detector_motion(point: wp.vec3d, detector: NativeBoundaryDetector) -> wp.vec3d:
    return detector.motion_translation + wp.cross(
        detector.motion_angular, point - detector.motion_pivot
    )


@wp.func
def _boundary_endpoint(
    coordinate: wp.vec2d, detector: NativeBoundaryDetector
) -> NativeBoundaryEndpoint:
    result = NativeBoundaryEndpoint()
    amount_x = coordinate[0] * wp.float64(detector.shape[0])
    amount_y = coordinate[1] * wp.float64(detector.shape[1])
    result.point_mm = detector.corner + (
        detector.u * (amount_x * detector.spacing[0])
        + detector.v * (amount_y * detector.spacing[1])
    )
    result.point_dot = detector.corner_correction + (
        _detector_motion(result.point_mm, detector)
        + (
            detector.u * (amount_x * detector.spacing_dot[0])
            + detector.v * (amount_y * detector.spacing_dot[1])
        )
    )
    result.normal = detector.normal
    result.normal_dot = wp.cross(detector.motion_angular, detector.normal)
    result.area_jacobian = (
        wp.float64(detector.shape[0] * detector.shape[1])
        * detector.spacing[0]
        * detector.spacing[1]
    )
    result.area_jacobian_dot = result.area_jacobian * (
        detector.spacing_dot[0] / detector.spacing[0]
        + detector.spacing_dot[1] / detector.spacing[1]
    )
    result.pixel = int(wp.floor(amount_y)) * detector.shape[0] + int(wp.floor(amount_x))
    return result


@wp.func
def _regular_pixel_edge_chart(
    boundary: NativeFreeComptonBoundary, cosine: wp.vec2d, detector: NativeBoundaryDetector
) -> bool:
    # The explicit constructor declares a mutable Warp loop variable.
    valid = bool(True)  # noqa: UP018
    if cosine[0] == wp.float64(0.0):
        k = boundary.incoming.direction
        k_dot = boundary.incoming.direction_dot
        corner_dot = _detector_motion(detector.corner, detector) + detector.corner_correction
        for axis in range(2):
            other = 1 - axis
            basis_axis = detector.u
            basis_other = detector.v
            if axis == 1:
                basis_axis = detector.v
                basis_other = detector.u
            edge_direction = basis_other * (
                wp.float64(detector.shape[other]) * detector.spacing[other]
            )
            if wp.dot(k, edge_direction) == wp.float64(0.0):
                edge_direction_dot = wp.cross(
                    detector.motion_angular, edge_direction
                ) + basis_other * (wp.float64(detector.shape[other]) * detector.spacing_dot[other])
                for index in range(detector.shape[axis] + 1):
                    shift = basis_axis * (wp.float64(index) * detector.spacing[axis])
                    displacement = detector.corner + shift - boundary.incoming.origin_mm
                    if wp.dot(k, displacement) == wp.float64(0.0):
                        first_dot = corner_dot + (
                            wp.cross(detector.motion_angular, shift)
                            + basis_axis * (wp.float64(index) * detector.spacing_dot[axis])
                        )
                        displacement_dot = first_dot - boundary.incoming.origin_dot
                        first_numerator_dot = wp.dot(k_dot, displacement) + wp.dot(
                            k, displacement_dot
                        )
                        edge_numerator_dot = wp.dot(k_dot, edge_direction) + wp.dot(
                            k, edge_direction_dot
                        )
                        if (
                            cosine[1] != wp.float64(0.0)
                            or first_numerator_dot != wp.float64(0.0)
                            or edge_numerator_dot != wp.float64(0.0)
                        ):
                            valid = False
    return valid


@wp.func
def sample_free_compton_energy_boundary(
    boundary: NativeFreeComptonBoundary, coordinate: wp.float64, detector: NativeBoundaryDetector
) -> NativeComptonBoundarySample:
    """Sample the unconditioned azimuth and retain exact physical coarea weights.

    The host adapter validates rigid frames and pixel metadata. Every computed
    attached endpoint, energy jet and final weight must remain representable.
    The result's reference weight already divides its physical-area weight by
    the detector chart Jacobian, so the driver includes that Jacobian once.
    """
    result = NativeComptonBoundarySample()
    result.status = FAILURE
    valid = (
        wp.isfinite(coordinate) and coordinate > wp.float64(0.0) and coordinate < wp.float64(1.0)
    )
    valid = (
        valid
        and wp.isfinite(boundary.incident_energy)
        and boundary.incident_energy > wp.float64(0.0)
    )
    valid = (
        valid
        and wp.isfinite(boundary.threshold_energy)
        and boundary.threshold_energy > wp.float64(0.0)
    )
    valid = (
        valid
        and wp.isfinite(boundary.incident_energy_dot)
        and wp.isfinite(boundary.threshold_energy_dot)
    )
    valid = (
        valid and _finite3(boundary.incoming.origin_mm) and _finite3(boundary.incoming.direction)
    )
    valid = (
        valid
        and _finite3(boundary.incoming.origin_dot)
        and _finite3(boundary.incoming.direction_dot)
    )
    if valid:
        cosine = free_compton_boundary_cosine(boundary)
        if wp.isfinite(cosine[0]) and wp.isfinite(cosine[1]):
            if wp.abs(cosine[0]) > wp.float64(1.0):
                result.status = NULL
            elif wp.abs(cosine[0]) == wp.float64(1.0):
                if cosine[1] == wp.float64(0.0):
                    result.status = NULL
            else:
                height = wp.dot(detector.corner - boundary.incoming.origin_mm, detector.normal)
                if height < wp.float64(0.0):
                    result.status = NULL
                elif wp.isfinite(height) and height > wp.float64(0.0):
                    if _regular_pixel_edge_chart(boundary, cosine, detector):
                        k = boundary.incoming.direction
                        anchor = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
                        if wp.abs(k[2]) >= wp.float64(0.9):
                            anchor = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
                        first = wp.cross(k, anchor)
                        first_length = _hypot(_hypot(first[0], first[1]), first[2])
                        first = first * (wp.float64(1.0) / first_length)
                        second = wp.cross(k, first)
                        phi = TWO_PI * coordinate
                        radius = wp.sqrt(
                            (wp.float64(1.0) - cosine[0]) * (wp.float64(1.0) + cosine[0])
                        )
                        direction = k * cosine[0] + (
                            first * (radius * wp.cos(phi)) + second * (radius * wp.sin(phi))
                        )
                        projection = wp.dot(direction, detector.normal)
                        if projection <= wp.float64(0.0):
                            result.status = NULL
                        elif wp.isfinite(projection):
                            distance = height / projection
                            if wp.isfinite(distance) and distance > wp.float64(0.0):
                                point = boundary.incoming.origin_mm + direction * distance
                                offset = point - detector.corner
                                uv = wp.vec2d(
                                    wp.dot(offset, detector.u)
                                    / (wp.float64(detector.shape[0]) * detector.spacing[0]),
                                    wp.dot(offset, detector.v)
                                    / (wp.float64(detector.shape[1]) * detector.spacing[1]),
                                )
                                if wp.isfinite(uv[0]) and wp.isfinite(uv[1]):
                                    if (
                                        uv[0] <= wp.float64(0.0)
                                        or uv[0] >= wp.float64(1.0)
                                        or uv[1] <= wp.float64(0.0)
                                        or uv[1] >= wp.float64(1.0)
                                    ):
                                        result.status = NULL
                                    else:
                                        endpoint = _boundary_endpoint(uv, detector)
                                        displacement = (
                                            endpoint.point_mm - boundary.incoming.origin_mm
                                        )
                                        displacement_dot = (
                                            endpoint.point_dot - boundary.incoming.origin_dot
                                        )
                                        distance = _hypot(
                                            _hypot(displacement[0], displacement[1]),
                                            displacement[2],
                                        )
                                        if wp.isfinite(distance) and distance > wp.float64(0.0):
                                            outgoing = displacement * (wp.float64(1.0) / distance)
                                            outgoing_dot = (
                                                displacement_dot
                                                - outgoing * wp.dot(outgoing, displacement_dot)
                                            ) * (wp.float64(1.0) / distance)
                                            actual_cosine = wp.dot(k, outgoing)
                                            actual_cosine_dot = wp.dot(
                                                boundary.incoming.direction_dot, outgoing
                                            ) + wp.dot(k, outgoing_dot)
                                            energy = wp.float64(1.0) / (
                                                wp.float64(1.0) / boundary.incident_energy
                                                + (wp.float64(1.0) - actual_cosine) / MASS
                                            )
                                            energy_dot = (
                                                energy
                                                * energy
                                                * (
                                                    (
                                                        boundary.incident_energy_dot
                                                        / boundary.incident_energy
                                                    )
                                                    / boundary.incident_energy
                                                    + actual_cosine_dot / MASS
                                                )
                                            )
                                            g_dot = actual_cosine_dot - cosine[1]
                                            world_weight = (TWO_PI * distance) * (
                                                distance * g_dot / projection
                                            )
                                            reference_weight = world_weight / endpoint.area_jacobian
                                            levelset_dot = (
                                                energy_dot - boundary.threshold_energy_dot
                                            )
                                            valid = (
                                                _finite3(endpoint.point_mm)
                                                and _finite3(endpoint.point_dot)
                                                and _finite3(endpoint.normal_dot)
                                            )
                                            valid = (
                                                valid
                                                and _finite3(outgoing)
                                                and _finite3(outgoing_dot)
                                            )
                                            valid = (
                                                valid
                                                and wp.isfinite(endpoint.area_jacobian)
                                                and endpoint.area_jacobian > wp.float64(0.0)
                                            )
                                            valid = valid and wp.isfinite(
                                                endpoint.area_jacobian_dot
                                            )
                                            valid = (
                                                valid
                                                and wp.isfinite(energy)
                                                and wp.isfinite(energy_dot)
                                                and wp.isfinite(levelset_dot)
                                            )
                                            valid = (
                                                valid
                                                and wp.isfinite(world_weight)
                                                and wp.isfinite(reference_weight)
                                            )
                                            if valid:
                                                result.coordinate = uv
                                                result.endpoint = endpoint
                                                result.outgoing.origin_mm = (
                                                    boundary.incoming.origin_mm
                                                )
                                                result.outgoing.origin_dot = (
                                                    boundary.incoming.origin_dot
                                                )
                                                result.outgoing.direction = outgoing
                                                result.outgoing.direction_dot = outgoing_dot
                                                result.outgoing_energy = energy
                                                result.outgoing_energy_dot = energy_dot
                                                result.levelset_dot = levelset_dot
                                                result.world_area_weight = world_weight
                                                result.reference_area_weight = reference_weight
                                                result.status = 0
    return result
