"""Binary64 detector-area mixtures for Monte Carlo geometry gradients.

The density is with respect to the unit detector square. Multiply the physical
detector area once when forming a next-event score. These detached proposals
carry no geometry derivatives. A nonzero status invalidates the whole history.
"""

# Warp annotations name runtime DSL types.
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
class DetectorProposalPlane:
    coefficients: wp.vec3d
    normal_separation_mm: wp.float64
    axis: int
    position_mm: wp.float64


@wp.struct
class DetectorProposalGeometry:
    inverse_rotation: wp.mat33d
    inverse_translation: wp.vec3d
    detector_lower: wp.vec3d
    detector_u: wp.vec3d
    detector_v: wp.vec3d
    detector_shape: wp.vec2i
    detector_spacing: wp.vec2d


@wp.struct
class NativeDetectorDensity:
    log_reference_density: wp.float64
    status: int


@wp.struct
class NativeDetectorProposalSample:
    coordinate: wp.vec2d
    log_reference_density: wp.float64
    component: int
    status: int


@wp.func_native("""
if (value == 0.0) return 0.0;
const double absolute = wp::abs(value);
#if defined(__CUDACC__)
if (absolute <= separation) return copysign(log1p(absolute/separation), value);
return copysign(log(absolute)-log(separation)+log1p(separation/absolute), value);
#else
if (absolute <= separation) return __builtin_copysign(__builtin_log1p(absolute/separation), value);
return __builtin_copysign(__builtin_log(absolute)-__builtin_log(separation)
    +__builtin_log1p(separation/absolute), value);
#endif
""")
def _primitive(value: wp.float64, separation: wp.float64) -> wp.float64: ...


@wp.func_native("""
if (value == 0.0) return 0.0;
const double absolute = wp::abs(value);
#if defined(__CUDACC__)
return copysign(exp(log(separation)+absolute)*-expm1(-absolute), value);
#else
return __builtin_copysign(__builtin_exp(__builtin_log(separation)+absolute)
    *-__builtin_expm1(-absolute), value);
#endif
""")
def _inverse_primitive(value: wp.float64, separation: wp.float64) -> wp.float64: ...


@wp.func_native("""
const double absolute = wp::abs(value);
#if defined(__CUDACC__)
const double next = nextafter(absolute, static_cast<double>(INFINITY));
return wp::isfinite(next) ? next-absolute : absolute-nextafter(absolute, 0.0);
#else
const double next = __builtin_nextafter(absolute, INFINITY);
return wp::isfinite(next) ? next-absolute : absolute-__builtin_nextafter(absolute, 0.0);
#endif
""")
def _ulp(value: wp.float64) -> wp.float64: ...


@wp.func
def _open_coordinate(coordinate: wp.vec2d) -> bool:
    return (
        wp.isfinite(coordinate[0])
        and wp.isfinite(coordinate[1])
        and coordinate[0] > wp.float64(0.0)
        and coordinate[0] < wp.float64(1.0)
        and coordinate[1] > wp.float64(0.0)
        and coordinate[1] < wp.float64(1.0)
    )


@wp.func
def _separation(
    plane: DetectorProposalPlane, previous_local: wp.vec3d, use_vertex: int
) -> wp.float64:
    separation = plane.normal_separation_mm
    if use_vertex != 0:
        separation = wp.float64(-1.0)
        if plane.axis >= 0 and plane.axis < 3:
            separation = wp.abs(previous_local[plane.axis] - plane.position_mm)
    return separation


@wp.func
def _axis(coefficients: wp.vec3d) -> int:
    axis = int(0)  # noqa: RUF046, UP018
    if wp.abs(coefficients[1]) > wp.abs(coefficients[0]):
        axis = 1
    return axis


@wp.func
def _density(
    planes: wp.array(dtype=DetectorProposalPlane),
    offset: int,
    count: int,
    previous_local: wp.vec3d,
    use_vertex: int,
    coordinate: wp.vec2d,
) -> NativeDetectorDensity:
    result = NativeDetectorDensity()
    result.status = FAILURE
    valid = offset >= 0 and count >= 0 and offset <= planes.shape[0]
    valid = valid and count <= planes.shape[0] - offset and _open_coordinate(coordinate)
    scale = wp.log(wp.float64(0.5))
    total = wp.float64(1.0)
    if valid and count == 0:
        result.log_reference_density = wp.float64(0.0)
        result.status = 0
    elif valid:
        mixture = wp.log(wp.float64(0.5)) - wp.log(wp.float64(count))
        for index in range(count):
            plane = planes[offset + index]
            separation = _separation(plane, previous_local, use_vertex)
            coefficients = plane.coefficients
            axis = _axis(coefficients)
            coefficient = coefficients[axis]
            other = coefficients[1 - axis] * coordinate[1 - axis] + coefficients[2]
            # Keep exactly the same chart contraction as inverse-CDF reconstruction.
            represented = coefficient * coordinate[axis] + other
            first = _primitive(other, separation)
            last = _primitive(other + coefficient, separation)
            width = wp.abs(last - first)
            valid_plane = (
                wp.isfinite(separation)
                and separation > wp.float64(0.0)
                and wp.isfinite(coefficient)
                and coefficient != wp.float64(0.0)
                and wp.isfinite(other)
                and wp.isfinite(represented)
                and wp.isfinite(width)
                and width > wp.float64(0.0)
                and wp.isfinite(separation + wp.abs(represented))
            )
            if valid_plane:
                term = (
                    mixture
                    + wp.log(wp.abs(coefficient))
                    - wp.log(width)
                    - wp.log(separation + wp.abs(represented))
                )
                next_scale = wp.max(scale, term)
                total = total * wp.exp(scale - next_scale) + wp.exp(term - next_scale)
                scale = next_scale
                valid = valid and wp.isfinite(term) and wp.isfinite(total)
            else:
                valid = False
        if valid:
            value = scale + wp.log(total)
            if wp.isfinite(value):
                result.log_reference_density = value
                result.status = 0
    return result


@wp.func
def _sample(
    planes: wp.array(dtype=DetectorProposalPlane),
    offset: int,
    count: int,
    previous_local: wp.vec3d,
    use_vertex: int,
    component_draw: wp.float64,
    coordinate: wp.vec2d,
) -> NativeDetectorProposalSample:
    result = NativeDetectorProposalSample()
    result.status = FAILURE
    result.component = -1
    valid = offset >= 0 and count >= 0 and offset <= planes.shape[0]
    valid = valid and count <= planes.shape[0] - offset and _open_coordinate(coordinate)
    valid = valid and wp.isfinite(component_draw)
    valid = valid and component_draw > wp.float64(0.0) and component_draw < wp.float64(1.0)
    mapped = coordinate
    if valid and count > 0 and component_draw >= wp.float64(0.5):
        component = wp.min(
            count - 1, int((component_draw - wp.float64(0.5)) * wp.float64(2.0) * wp.float64(count))
        )
        plane = planes[offset + component]
        separation = _separation(plane, previous_local, use_vertex)
        axis = _axis(plane.coefficients)
        coefficient = plane.coefficients[axis]
        other = plane.coefficients[1 - axis] * coordinate[1 - axis] + plane.coefficients[2]
        first, last = _primitive(other, separation), _primitive(other + coefficient, separation)
        width = last - first
        transformed = (wp.float64(1.0) - coordinate[axis]) * first + coordinate[axis] * last
        level = _inverse_primitive(transformed, separation)
        amount = (level - other) / coefficient
        represented = coefficient * amount + other
        reconstructed = (_primitive(represented, separation) - first) / width
        uncertainty = (
            wp.float64(16.0)
            * (_ulp(represented) + _ulp(other))
            / ((separation + wp.abs(represented)) * wp.abs(width))
        )
        valid = (
            wp.isfinite(separation)
            and separation > wp.float64(0.0)
            and wp.isfinite(coefficient)
            and coefficient != wp.float64(0.0)
            and wp.isfinite(width)
            and width != wp.float64(0.0)
            and wp.isfinite(amount)
            and amount > wp.float64(0.0)
            and amount < wp.float64(1.0)
            and wp.isfinite(uncertainty)
            and uncertainty <= CDF_RESOLUTION
            and wp.abs(reconstructed - coordinate[axis])
            <= uncertainty + wp.float64(512.0) * EPSILON
        )
        if valid:
            mapped[axis] = amount
            result.component = component
    if valid:
        density = _density(planes, offset, count, previous_local, use_vertex, mapped)
        if density.status == 0:
            result.coordinate = mapped
            result.log_reference_density = density.log_reference_density
            result.status = 0
    return result


@wp.func
def detector_area_log_density(
    planes: wp.array(dtype=DetectorProposalPlane),
    offset: int,
    count: int,
    coordinate: wp.vec2d,
) -> NativeDetectorDensity:
    """Evaluate all components of a fixed vertex's unit-square mixture."""
    return _density(planes, offset, count, wp.vec3d(wp.float64(0.0)), 0, coordinate)


@wp.func
def sample_detector_area_proposal(
    planes: wp.array(dtype=DetectorProposalPlane),
    offset: int,
    count: int,
    component_draw: wp.float64,
    coordinate: wp.vec2d,
) -> NativeDetectorProposalSample:
    """Sample a fixed plane table; return the complete mixture density."""
    return _sample(planes, offset, count, wp.vec3d(wp.float64(0.0)), 0, component_draw, coordinate)


@wp.func
def sample_detector_area_proposal_from_vertex(
    planes: wp.array(dtype=DetectorProposalPlane),
    offset: int,
    count: int,
    geometry: DetectorProposalGeometry,
    previous_point_mm: wp.vec3d,
    component_draw: wp.float64,
    coordinate: wp.vec2d,
) -> NativeDetectorProposalSample:
    """Use an immutable geometry plan and the current physical collision vertex.

    Each plane separation is computed on the device. Every mixture component
    checks the actual detector endpoint before returning a successful sample.
    """
    previous_local = geometry.inverse_rotation * previous_point_mm + geometry.inverse_translation
    result = _sample(planes, offset, count, previous_local, 1, component_draw, coordinate)
    for axis in range(3):
        if not wp.isfinite(previous_local[axis]):
            result.status = FAILURE
    if result.status == 0:
        mapped = result.coordinate
        amount_x = (mapped[0] * wp.float64(geometry.detector_shape[0])) * geometry.detector_spacing[
            0
        ]
        amount_y = (mapped[1] * wp.float64(geometry.detector_shape[1])) * geometry.detector_spacing[
            1
        ]
        # Match the actual detector endpoint chart before computing its local plane level.
        world = geometry.detector_lower + (
            geometry.detector_u * amount_x + geometry.detector_v * amount_y
        )
        local = geometry.inverse_rotation * world + geometry.inverse_translation
        for axis in range(3):
            if not wp.isfinite(local[axis]):
                result.status = FAILURE
        for index in range(count):
            plane = planes[offset + index]
            separation = _separation(plane, previous_local, 1)
            axis = _axis(plane.coefficients)
            coefficient = plane.coefficients[axis]
            other = plane.coefficients[1 - axis] * mapped[1 - axis] + plane.coefficients[2]
            represented = coefficient * mapped[axis] + other
            actual = local[plane.axis] - plane.position_mm
            width = _primitive(other + coefficient, separation) - _primitive(other, separation)
            uncertainty = (
                wp.float64(16.0)
                * (_ulp(local[plane.axis]) + _ulp(plane.position_mm))
                / ((separation + wp.abs(actual)) * wp.abs(width))
            )
            cdf_error = wp.abs(
                (_primitive(actual, separation) - _primitive(represented, separation)) / width
            )
            relative_error = wp.abs(actual - represented) / (
                separation + wp.max(wp.abs(actual), wp.abs(represented))
            )
            if (
                not wp.isfinite(uncertainty)
                or uncertainty > CDF_RESOLUTION
                or not wp.isfinite(cdf_error)
                or cdf_error > uncertainty + wp.float64(512.0) * EPSILON
                or not wp.isfinite(relative_error)
                or relative_error > CDF_RESOLUTION
            ):
                result.status = FAILURE
    return result
