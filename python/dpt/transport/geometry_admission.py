"""Admission of differentiable elastic geometry charts, shared by estimators."""

from __future__ import annotations

from itertools import product

# Geometry estimators share checked vector primitives within this package.
# pyright: reportPrivateUsage=false
from typing import TYPE_CHECKING, cast

from dpt.geometry import Vector3

from .geometry_path_measure import source_volume_boundaries
from .geometry_tangents import RayTangent, _add
from .model import TransportError, TransportSpec
from .source import ParallelBeam, PointConeSource
from .uncollided_geometry import AcquisitionGeometryTangent

if TYPE_CHECKING:
    from .elastic_geometry import ElasticGeometryMedium


def admit_elastic_geometry_chart(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    medium: ElasticGeometryMedium,
    tangent: AcquisitionGeometryTangent,
) -> None:
    for corner in product((0.0, 1.0), repeat=3):
        local_corner = cast(
            Vector3,
            tuple(
                spec.grid.origin_mm[axis]
                + corner[axis] * (spec.grid.upper_mm[axis] - spec.grid.origin_mm[axis])
                for axis in range(3)
            ),
        )
        if spec.detector.source_distance(spec.object_pose.point(local_corner)) >= 0.0:
            raise TransportError(
                "geometry detector requires strict separation from the material box"
            )
    if isinstance(source, ParallelBeam):
        ray = RayTangent(
            source.lower_mm,
            source.direction,
            tangent.source.motion.point_velocity(source.lower_mm),
            _add(
                tangent.source.motion.direction_velocity(source.direction),
                tangent.source.direction_dot,
            ),
        )
        local = ray.in_object_frame(spec.object_pose, tangent.object.motion)
        for direction, derivative in zip(local.direction, local.direction_dot, strict=True):
            if direction == 0.0 and derivative != 0.0:
                raise TransportError("active angular motion leaves a grazing material-face chart")
    else:
        local_source = spec.object_pose.inverse().point(source.geometry.source_mm)
        if all(
            lower <= coordinate <= upper
            for coordinate, lower, upper in zip(
                local_source, spec.grid.origin_mm, spec.grid.upper_mm, strict=True
            )
        ):
            raise TransportError("geometry source point requires strict separation from the object")
    boundaries = source_volume_boundaries(source, spec, tangent.object, tangent.source)
    for boundary in boundaries:
        nonzero = tuple(axis for axis, value in enumerate(boundary.normal) if value != 0.0)
        if len(nonzero) != 1 or (
            boundary.normal_dot == (0.0, 0.0, 0.0) and boundary.offset_dot == 0.0
        ):
            continue
        axis = nonzero[0]
        location = -boundary.offset / boundary.normal[axis]
        count = spec.grid.shape_xyz[axis]
        for face in range(1, count):
            if location != face / count:
                continue
            # The interface is physically discontinuous only if one represented
            # collision/extinction law differs across an adjacent cell pair.
            nx, ny, nz = spec.grid.shape_xyz
            for z in range(nz):
                for y in range(ny):
                    for x in range(nx):
                        indices = [x, y, z]
                        if indices[axis] != face - 1:
                            continue
                        left = (z * ny + y) * nx + x
                        indices[axis] += 1
                        right = (indices[2] * ny + indices[1]) * nx + indices[0]
                        fields = (
                            medium.extinction_mm_inverse,
                            medium.isotropic_mm_inverse,
                            medium.rayleigh_mm_inverse,
                        )
                        unequal = any(values[left] != values[right] for values in fields)
                        if medium.rayleigh_mm_inverse[left] > 0.0 and (
                            medium.rayleigh_mm_inverse[right] > 0.0
                        ):
                            first, second = medium.atomic_tables[left], medium.atomic_tables[right]
                            assert first is not None and second is not None
                            unequal = (
                                unequal
                                or first.momentum_transfer_inverse_angstrom
                                != (second.momentum_transfer_inverse_angstrom)
                                or first.form_factor != second.form_factor
                            )
                        if unequal:
                            raise TransportError(
                                "active source support coincides with a "
                                "discontinuous material interface"
                            )


__all__ = ["admit_elastic_geometry_chart"]
