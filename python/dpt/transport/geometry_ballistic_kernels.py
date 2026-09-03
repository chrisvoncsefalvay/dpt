"""Selected original-ray material crossings and constant-space face decoding.

Each selected plane belongs to the complete grid-plane measure. The caller
multiplies its contribution by that measure's size once. Status minus one is a
genuine miss or grazing ray; status four invalidates the complete history.

"""

# Warp annotations name runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false
# Shared helpers belong to the internal native geometry interface.
# pyright: reportPrivateUsage=false
import warp as wp

from ._constants import NUMERICAL_FAILURE
from .compton_geometry_boundary_kernels import NativeBoundaryRay
from .geometry_interface_kernels import (
    InterfaceGrid,
    InterfacePatch,
    NativeInterfacePoint,
    _finite3,
    _hypot,
    _tangent_axes,
    _ulp,
    interface_point_tangent,
)

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})
FAILURE = wp.constant(NUMERICAL_FAILURE)
NULL = wp.constant(-1)
INT_MAX = wp.constant(wp.float64(2147483647.0))


@wp.struct
class NativePlaneResult:
    axis: int
    face: int
    status: int


@wp.struct
class NativePatchResult:
    patch: InterfacePatch
    status: int


@wp.struct
class NativeBallisticCrossing:
    patch: InterfacePatch
    point: NativeInterfacePoint
    incoming: NativeBoundaryRay
    distance_mm: wp.float64
    inverse_normal_projection: wp.float64
    status: int


@wp.func
def _grid_admitted(grid: InterfaceGrid) -> bool:
    valid = _finite3(grid.origin) and _finite3(grid.spacing)
    cells = wp.float64(1.0)
    for axis in range(3):
        valid = valid and grid.shape[axis] > 0 and grid.spacing[axis] > wp.float64(0.0)
        upper = grid.origin[axis] + wp.float64(grid.shape[axis]) * grid.spacing[axis]
        valid = valid and wp.isfinite(upper) and upper > grid.origin[axis]
        cells *= wp.float64(grid.shape[axis])
    # Products are evaluated in binary64 before any signed-index multiplication.
    # Every admitted integer product is <=2**31-1 and therefore exactly represented.
    return valid and cells <= INT_MAX


@wp.func
def _rigid_rotation_admitted(rotation: wp.mat33d) -> bool:
    valid = bool(True)  # noqa: UP018
    for first in range(3):
        for second in range(3):
            valid = valid and wp.isfinite(rotation[first, second])
            product = wp.float64(0.0)
            for component in range(3):
                product += rotation[component, first] * rotation[component, second]
            expected = wp.float64(0.0)
            if first == second:
                expected = wp.float64(1.0)
            valid = (
                valid and wp.isfinite(product) and wp.abs(product - expected) <= wp.float64(1e-10)
            )
    first_column = wp.vec3d(rotation[0, 0], rotation[1, 0], rotation[2, 0])
    second_column = wp.vec3d(rotation[0, 1], rotation[1, 1], rotation[2, 1])
    third_column = wp.vec3d(rotation[0, 2], rotation[1, 2], rotation[2, 2])
    determinant = wp.dot(wp.cross(first_column, second_column), third_column)
    return valid and wp.isfinite(determinant) and determinant >= wp.float64(0.0)


@wp.func
def material_plane_count(grid: InterfaceGrid) -> int:
    result = int(-1)  # noqa: RUF046, UP018
    total = wp.float64(3.0)
    for axis in range(3):
        total += wp.float64(grid.shape[axis])
    if _grid_admitted(grid) and total <= INT_MAX:
        result = int(total)
    return result


@wp.func
def select_material_plane(index: int, grid: InterfaceGrid) -> NativePlaneResult:
    result = NativePlaneResult()
    result.status = FAILURE
    count = material_plane_count(grid)
    if index < 0 or index >= count:
        return result
    remainder = index
    for axis in range(3):
        if remainder <= grid.shape[axis]:
            result.axis = axis
            result.face = remainder
            result.status = 0
            return result
        remainder -= grid.shape[axis] + 1
    return result


@wp.func
def material_patch_count(grid: InterfaceGrid) -> int:
    result = int(-1)  # noqa: RUF046, UP018
    total = wp.float64(0.0)
    for axis in range(3):
        axes = _tangent_axes(axis)
        total += (
            (wp.float64(grid.shape[axis]) + wp.float64(1.0))
            * wp.float64(grid.shape[axes[0]])
            * wp.float64(grid.shape[axes[1]])
        )
    if _grid_admitted(grid) and total <= INT_MAX:
        result = int(total)
    return result


@wp.func
def _patch(axis: int, face: int, tangential: wp.vec2i, grid: InterfaceGrid) -> InterfacePatch:
    result = InterfacePatch()
    result.axis = axis
    result.face = face
    result.tangential_cells = tangential
    result.negative_cell = -1
    result.positive_cell = -1
    axes = _tangent_axes(axis)
    cell = wp.vec3i(0)
    cell[axes[0]] = tangential[0]
    cell[axes[1]] = tangential[1]
    if face > 0:
        cell[axis] = face - 1
        result.negative_cell = (cell[2] * grid.shape[1] + cell[1]) * grid.shape[0] + cell[0]
    if face < grid.shape[axis]:
        cell[axis] = face
        result.positive_cell = (cell[2] * grid.shape[1] + cell[1]) * grid.shape[0] + cell[0]
    return result


@wp.func
def select_material_patch(index: int, grid: InterfaceGrid) -> NativePatchResult:
    result = NativePatchResult()
    result.status = FAILURE
    count = material_patch_count(grid)
    if index < 0 or index >= count:
        return result
    remainder = index
    for axis in range(3):
        axes = _tangent_axes(axis)
        tangential_count = grid.shape[axes[0]] * grid.shape[axes[1]]
        axis_count = (grid.shape[axis] + 1) * tangential_count
        if remainder < axis_count:
            face = remainder // tangential_count
            within = remainder % tangential_count
            first = within // grid.shape[axes[1]]
            second = within % grid.shape[axes[1]]
            result.patch = _patch(axis, face, wp.vec2i(first, second), grid)
            result.status = 0
            return result
        remainder -= axis_count
    return result


@wp.func
def selected_plane_crossing(
    ray: NativeBoundaryRay, plane_index: int, grid: InterfaceGrid
) -> NativeBallisticCrossing:
    """Pull one transverse material-plane delta back to its original source ray."""
    result = NativeBallisticCrossing()
    result.status = FAILURE
    plane = select_material_plane(plane_index, grid)
    if (
        plane.status != 0
        or not _finite3(ray.origin_mm)
        or not _finite3(ray.direction)
        or not _finite3(grid.translation)
        or not _rigid_rotation_admitted(grid.rotation)
        or wp.abs(wp.dot(ray.direction, ray.direction) - wp.float64(1.0)) > wp.float64(1e-12)
    ):
        return result
    inverse = wp.transpose(grid.rotation)
    origin = inverse * (ray.origin_mm - grid.translation)
    direction = inverse * ray.direction
    if not _finite3(origin) or not _finite3(direction):
        return result
    if direction[plane.axis] == wp.float64(0.0):
        result.status = NULL
        return result
    location = grid.origin[plane.axis] + wp.float64(plane.face) * grid.spacing[plane.axis]
    if not wp.isfinite(location):
        return result
    distance = (location - origin[plane.axis]) / direction[plane.axis]
    if not wp.isfinite(distance):
        return result
    if distance <= wp.float64(0.0):
        result.status = NULL
        return result
    local = origin + distance * direction
    if not _finite3(local):
        return result
    axes = _tangent_axes(plane.axis)
    coordinate = wp.vec2d(wp.float64(0.0))
    tangential = wp.vec2i(0)
    for i in range(2):
        axis = axes[i]
        upper = grid.origin[axis] + wp.float64(grid.shape[axis]) * grid.spacing[axis]
        if local[axis] < grid.origin[axis] or local[axis] > upper:
            result.status = NULL
            return result
        position = (local[axis] - grid.origin[axis]) / grid.spacing[axis]
        integer = wp.floor(position)
        if (
            not wp.isfinite(position)
            or position == integer
            or integer < wp.float64(0.0)
            or integer >= wp.float64(grid.shape[axis])
        ):
            return result
        tangential[i] = int(integer)
        coordinate[i] = position - integer
    patch = _patch(plane.axis, plane.face, tangential, grid)
    point = interface_point_tangent(patch, coordinate, grid)
    if point.status != 0:
        return result
    actual = ray.origin_mm + distance * ray.direction
    scale = _ulp(distance)
    for axis in range(3):
        scale = wp.max(scale, wp.max(_ulp(actual[axis]), _ulp(point.point_mm[axis])))
    discrepancy = point.point_mm - actual
    error = _hypot(_hypot(discrepancy[0], discrepancy[1]), discrepancy[2])
    projection = wp.abs(wp.dot(ray.direction, point.normal))
    reciprocal = wp.float64(1.0) / projection
    if (
        not _finite3(actual)
        or not wp.isfinite(error)
        or error > wp.float64(512.0) * scale
        or projection == wp.float64(0.0)
        or not wp.isfinite(reciprocal)
    ):
        return result
    incoming = NativeBoundaryRay()
    incoming.origin_mm = point.point_mm
    incoming.direction = ray.direction
    result.patch = patch
    result.point = point
    result.incoming = incoming
    result.distance_mm = distance
    result.inverse_normal_projection = reciprocal
    result.status = 0
    return result


__all__ = [
    "NativeBallisticCrossing",
    "NativePatchResult",
    "NativePlaneResult",
    "material_patch_count",
    "material_plane_count",
    "select_material_patch",
    "select_material_plane",
    "selected_plane_crossing",
]
