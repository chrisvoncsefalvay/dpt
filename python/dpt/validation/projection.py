"""Independent CPU references for the sampled-field projection contract.

These scalar routines are test oracles, never production execution paths. The
piecewise Gauss rule integrates a trilinear interpolant along a straight line
exactly up to FP64 arithmetic: each segment is a polynomial of degree at most
three. It therefore separates quadrature error from voxel discretisation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Literal, cast

from dpt.contracts import ContractError
from dpt.geometry import DetectorGeometry, RayBundleGeometry, RigidTransform, Vector3
from dpt.volumes import GridSpec


def _coordinates(grid: GridSpec, pose: RigidTransform, point: Vector3) -> Vector3:
    # Expand both inverse maps independently of the device or host geometry helpers.
    displacement = [point[i] - pose.translation_mm[i] for i in range(3)]
    object_point = [
        sum(pose.rotation[3 * j + i] * displacement[j] for j in range(3)) for i in range(3)
    ]
    return cast(
        Vector3,
        tuple(
            sum(
                grid.orientation[3 * j + i] * (object_point[j] - grid.origin_mm[j])
                for j in range(3)
            )
            / grid.spacing_mm[i]
            for i in range(3)
        ),
    )


def _segment(
    grid: GridSpec, geometry: DetectorGeometry, pose: RigidTransform, row: int, column: int
) -> tuple[Vector3, Vector3, float, float, float]:
    if not 0 <= row < geometry.shape[0] or not 0 <= column < geometry.shape[1]:
        raise ContractError("reference pixel is outside detector")
    endpoint: Vector3 = cast(
        Vector3,
        tuple(
            geometry.origin_mm[i]
            + column * geometry.spacing_mm[0] * geometry.u[i]
            + row * geometry.spacing_mm[1] * geometry.v[i]
            for i in range(3)
        ),
    )
    return _clip_segment(grid, pose, geometry.source_mm, endpoint)


def _clip_segment(
    grid: GridSpec, pose: RigidTransform, source: Vector3, endpoint: Vector3
) -> tuple[Vector3, Vector3, float, float, float]:
    start = _coordinates(grid, pose, source)
    end = _coordinates(grid, pose, endpoint)
    direction: Vector3 = cast(Vector3, tuple(end[i] - start[i] for i in range(3)))
    lower, upper = 0.0, 1.0
    for axis, extent in enumerate(reversed(grid.shape)):
        if direction[axis] == 0.0:
            if start[axis] < -0.5 or start[axis] > extent - 0.5:
                return start, direction, 0.0, 0.0, 0.0
        else:
            faces = [
                (-0.5 - start[axis]) / direction[axis],
                (extent - 0.5 - start[axis]) / direction[axis],
            ]
            lower = max(lower, min(faces))
            upper = min(upper, max(faces))
    length = math.sqrt(sum((endpoint[i] - source[i]) ** 2 for i in range(3)))
    return start, direction, lower, upper, length


def sample_reference(values: Sequence[float], grid: GridSpec, point: Vector3) -> float:
    """Return ``sum_corner weight(point, corner) * values[corner]``.

    ``values`` is a flat sequence of ``grid.voxels`` samples in x-fastest
    z/y/x order. ``point`` contains three x/y/z grid-index coordinates.
    Coordinates are clamped to the outer samples inside the half-spacing
    support and return zero outside. The result has the supplied field's units.
    Nested linear interpolation evaluates the same field independently of
    the device corner-weight sum.

    Example:
        >>> grid = GridSpec((1, 1, 2), (1.0, 1.0, 1.0))
        >>> sample_reference((0.25, 0.75), grid, (0.5, 0.0, 0.0))
        0.5

    """
    if len(values) != grid.voxels:
        raise ContractError("reference field length does not match grid")
    if any(not math.isfinite(x) for x in point):
        raise ContractError("reference coordinate is not finite")
    xyz = tuple(reversed(grid.shape))
    if any(point[i] < -0.5 or point[i] > xyz[i] - 0.5 for i in range(3)):
        return 0.0
    q = [min(max(point[i], 0.0), xyz[i] - 1.0) for i in range(3)]
    lo = [math.floor(x) for x in q]
    hi = [min(lo[i] + 1, xyz[i] - 1) for i in range(3)]
    fraction = [q[i] - lo[i] for i in range(3)]
    planes: list[float] = []
    for z in (lo[2], hi[2]):
        rows: list[float] = []
        for y in (lo[1], hi[1]):
            a = values[(z * xyz[1] + y) * xyz[0] + lo[0]]
            b = values[(z * xyz[1] + y) * xyz[0] + hi[0]]
            rows.append(a + (b - a) * fraction[0])
        planes.append(rows[0] + (rows[1] - rows[0]) * fraction[1])
    return planes[0] + (planes[1] - planes[0]) * fraction[2]


def integrate_sampled_field(
    values: Sequence[float],
    grid: GridSpec,
    geometry: DetectorGeometry,
    pose: RigidTransform,
    row: int,
    column: int,
    *,
    midpoint_samples: int | None = None,
    validate: bool = True,
    field_domain: Literal["physical", "signed_linear"] = "physical",
) -> float:
    """Return ``L[row,column] = integral_segment mu_interpolated(x) ds``.

    ``values`` contains ``grid.voxels`` attenuation samples in mm^-1,
    flattened x fastest from ``grid.shape = (nz, ny, nx)``. ``geometry``
    defines finite world-mm source/pixel segments with panel shape
    ``(rows, columns)``, and ``pose`` maps object coordinates into world.
    The result is one dimensionless optical depth. With no midpoint count,
    two-point Gauss integration on each interpolation piece integrates its
    cubic restriction up to FP64 arithmetic. A positive ``midpoint_samples``
    selects an independent fixed midpoint rule.

    ``validate=False`` skips the full-field finite/nonnegative scan only. The
    caller must validate the field once and keep it unchanged across reference
    evaluations. Shape, pixel and quadrature contracts remain checked.
    signed_linear permits negative mathematical fields and still checks finiteness.

    Example:
        >>> grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
        >>> panel = DetectorGeometry(
        ...     (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0),
        ...     (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (1, 1),
        ... )
        >>> integrate_sampled_field((0.25,), grid, panel, RigidTransform(), 0, 0)
        0.25

    """
    return _integrate_segment(
        values,
        grid,
        _segment(grid, geometry, pose, row, column),
        midpoint_samples=midpoint_samples,
        validate=validate,
        field_domain=field_domain,
    )


def integrate_ray_bundle(
    values: Sequence[float],
    grid: GridSpec,
    geometry: RayBundleGeometry,
    pose: RigidTransform,
    ray: int,
    *,
    midpoint_samples: int | None = None,
    validate: bool = True,
    field_domain: Literal["physical", "signed_linear"] = "physical",
) -> float:
    """Return ``L[ray] = integral_source_endpoint mu_interpolated(x) ds``.

    ``values`` is an x-fastest attenuation sequence of ``grid.voxels``
    entries in mm^-1. The ray bundle holds equal-length tuples of world-mm
    source/end three-vectors, and ``ray`` selects one segment. ``pose`` maps
    object into world coordinates. The scalar optical depth is dimensionless.
    The default sorted-plane Gauss rule integrates each cubic piece, whilst
    a positive ``midpoint_samples`` selects fixed midpoint quadrature.
    ``field_domain`` and ``validate`` follow ``integrate_sampled_field``.

    Example:
        >>> grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
        >>> rays = RayBundleGeometry(((-2.0, 0.0, 0.0),), ((2.0, 0.0, 0.0),))
        >>> integrate_ray_bundle((0.25,), grid, rays, RigidTransform(), 0)
        0.25

    """
    source, endpoint = geometry.segment(ray)
    return _integrate_segment(
        values,
        grid,
        _clip_segment(grid, pose, source, endpoint),
        midpoint_samples=midpoint_samples,
        validate=validate,
        field_domain=field_domain,
    )


def _integrate_segment(
    values: Sequence[float],
    grid: GridSpec,
    segment: tuple[Vector3, Vector3, float, float, float],
    *,
    midpoint_samples: int | None,
    validate: bool,
    field_domain: Literal["physical", "signed_linear"],
) -> float:
    if field_domain not in ("physical", "signed_linear"):
        raise ContractError("reference field_domain must be physical or signed_linear")
    if len(values) != grid.voxels or (
        validate
        and any(not math.isfinite(x) or (field_domain == "physical" and x < 0) for x in values)
    ):
        qualifier = "finite/non-negative" if field_domain == "physical" else "finite"
        raise ContractError(f"reference attenuation must match the grid and be {qualifier}")
    start, direction, lower, upper, length = segment
    if upper <= lower:
        return 0.0

    def sample(t: float) -> float:
        point: Vector3 = cast(Vector3, tuple(start[i] + t * direction[i] for i in range(3)))
        return sample_reference(values, grid, point)

    if midpoint_samples is not None:
        if type(midpoint_samples) is not int or midpoint_samples < 1:
            raise ContractError("midpoint_samples must be a positive integer")
        step = (upper - lower) / midpoint_samples
        return (
            length
            * step
            * math.fsum(sample(lower + (i + 0.5) * step) for i in range(midpoint_samples))
        )
    breaks = {lower, upper}
    for axis, extent in enumerate(reversed(grid.shape)):
        if direction[axis] != 0:
            for centre in range(extent):
                t = (centre - start[axis]) / direction[axis]
                if lower < t < upper:
                    breaks.add(t)
    knots = sorted(breaks)
    integrals: list[float] = []
    for lo, hi in pairwise(knots):
        centre, radius = (lo + hi) / 2, (hi - lo) / 2
        offset = radius / math.sqrt(3)
        integrals.append(radius * (sample(centre - offset) + sample(centre + offset)))
    return length * math.fsum(integrals)


def quadratic_sample_values(grid: GridSpec, *, intercept: float, curvature: Vector3) -> list[float]:
    """Sample ``mu(x) = intercept + sum_i curvature[i] * x[i]**2``.

    ``x`` is the three-vector of object coordinates in mm at each voxel
    centre. The scalar intercept has units mm^-1 and the three curvature
    entries have units mm^-3. Both are finite and non-negative. The returned
    list contains ``grid.voxels`` attenuation values in x-fastest z/y/x order.
    This polynomial defines a numerical fixture.

    Example:
        >>> grid = GridSpec((1, 1, 2), (1.0, 1.0, 1.0))
        >>> quadratic_sample_values(grid, intercept=0.5, curvature=(0.25, 0.0, 0.0))
        [0.5, 0.75]

    """
    if (
        not math.isfinite(intercept)
        or intercept < 0
        or any(not math.isfinite(x) or x < 0 for x in curvature)
    ):
        raise ContractError("quadratic reference coefficients must be finite/non-negative")
    result: list[float] = []
    for z in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for x in range(grid.shape[2]):
                indices = (x, y, z)
                point = [
                    grid.origin_mm[i]
                    + sum(
                        grid.orientation[3 * i + j] * indices[j] * grid.spacing_mm[j]
                        for j in range(3)
                    )
                    for i in range(3)
                ]
                result.append(intercept + sum(curvature[i] * point[i] ** 2 for i in range(3)))
    return result


def integrate_quadratic_field(
    grid: GridSpec,
    geometry: DetectorGeometry,
    pose: RigidTransform,
    row: int,
    column: int,
    *,
    intercept: float,
    curvature: Vector3,
) -> float:
    """Integrate ``mu(x) = intercept + sum_i curvature[i] * x[i]**2`` exactly.

    ``geometry`` defines a finite world-mm source/pixel segment, ``pose``
    maps object coordinates into world, and ``row, column`` select a pixel
    in its ``(rows, columns)`` panel. Intercept is in mm^-1 and the three
    non-negative curvature entries are in mm^-3. The returned dimensionless
    scalar clips the continuous polynomial to the sampled grid's support.
    Comparing it with ``integrate_sampled_field`` separates voxel
    discretisation from quadrature error.

    Example:
        >>> grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
        >>> panel = DetectorGeometry(
        ...     (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0),
        ...     (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (1, 1),
        ... )
        >>> round(integrate_quadratic_field(
        ...     grid, panel, RigidTransform(), 0, 0,
        ...     intercept=0.5, curvature=(1.0, 0.0, 0.0),
        ... ), 8)
        0.58333333

    """
    if (
        not math.isfinite(intercept)
        or intercept < 0
        or any(not math.isfinite(x) or x < 0 for x in curvature)
    ):
        raise ContractError("quadratic reference coefficients must be finite/non-negative")
    start, direction, lower, upper, length = _segment(grid, geometry, pose, row, column)
    if upper <= lower:
        return 0.0
    origin = [
        grid.origin_mm[i]
        + sum(grid.orientation[3 * i + j] * start[j] * grid.spacing_mm[j] for j in range(3))
        for i in range(3)
    ]
    delta = [
        sum(grid.orientation[3 * i + j] * direction[j] * grid.spacing_mm[j] for j in range(3))
        for i in range(3)
    ]
    span = upper - lower
    # Shift the integration coordinate to its midpoint to avoid subtraction of
    # nearly equal squared/cubed endpoint primitives for a short clipped segment.
    centre = [(origin[i] + ((lower + upper) / 2) * delta[i]) for i in range(3)]
    mean = intercept + sum(
        curvature[i] * (centre[i] ** 2 + delta[i] ** 2 * span**2 / 12) for i in range(3)
    )
    return length * span * mean
