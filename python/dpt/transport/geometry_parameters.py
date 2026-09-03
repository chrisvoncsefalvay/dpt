"""Named active geometry coordinates and elastic transport cotangent contraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from dpt.geometry import Vector3, cross, vector3

from .elastic_geometry import (
    CollisionOrderLaw,
    CollisionPathSample,
    ElasticGeometryEstimate,
    ElasticGeometryMedium,
    SourceBoundaryPathSample,
    elastic_geometry_jvp,
)
from .elastic_geometry_prefix import (
    ElasticPrefixGeometryEstimate,
    GeometryBoundaryPrefixHistory,
    GeometryPrefixHistory,
    elastic_prefix_geometry_jvp,
)
from .geometry_tangents import GridGeometryTangent, RigidMotion
from .model import TransportError, TransportSpec
from .source import ParallelBeam, PointConeSource
from .uncollided_geometry import AcquisitionGeometryTangent, SourceGeometryTangent


@dataclass(frozen=True, slots=True)
class GeometryParameter:
    """One physical coordinate of object, source, launch target or detector.

    Cartesian axes are x,y,z for translations, rotations and grid fields;
    extents/spacings use their existing column/row or x/y order. Translation,
    origin, spacing and extent derivatives are per mm. Rotations are spatial
    radians about the supplied world pivot. The source weight stays fixed.
    """

    component: Literal["object", "source", "launch-target", "detector"]
    coordinate: Literal[
        "translation",
        "rotation",
        "grid-origin",
        "grid-spacing",
        "extent",
        "pixel-spacing",
        "beam-direction",
    ]
    axis: int
    pivot_mm: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pivot_mm", vector3(self.pivot_mm, "geometry pivot"))
        compatible = {
            "object": {"translation", "rotation", "grid-origin", "grid-spacing"},
            "source": {"translation", "rotation", "extent", "beam-direction"},
            "launch-target": {"translation", "rotation", "pixel-spacing"},
            "detector": {"translation", "rotation", "pixel-spacing"},
        }
        if self.component not in compatible or self.coordinate not in compatible[self.component]:
            raise TransportError("active geometry coordinate is incompatible with its component")
        dimensions = 2 if self.coordinate in ("extent", "pixel-spacing") else 3
        if type(self.axis) is not int or not 0 <= self.axis < dimensions:
            raise TransportError("active geometry axis is outside the coordinate's dimension")

    def tangent(self, source: ParallelBeam | PointConeSource) -> AcquisitionGeometryTangent:
        """Return the exact basis direction and reject absent source coordinates."""
        unit = cast(Vector3, tuple(float(i == self.axis) for i in range(3)))
        two = cast(tuple[float, float], tuple(float(i == self.axis) for i in range(2)))
        if self.coordinate == "translation":
            motion = RigidMotion(translation=unit)
        elif self.coordinate == "rotation":
            motion = RigidMotion(angular=unit, pivot_mm=self.pivot_mm)
        else:
            motion = RigidMotion()
        if self.component == "object":
            return AcquisitionGeometryTangent(
                object=GridGeometryTangent(
                    motion=motion,
                    origin_dot=unit if self.coordinate == "grid-origin" else (0.0, 0.0, 0.0),
                    spacing_dot=unit if self.coordinate == "grid-spacing" else (0.0, 0.0, 0.0),
                )
            )
        if self.component == "source":
            if isinstance(source, PointConeSource) and self.coordinate in (
                "extent",
                "beam-direction",
            ):
                raise TransportError(
                    "point emission has position; cone orientation belongs to target"
                )
            return AcquisitionGeometryTangent(
                source=SourceGeometryTangent(
                    motion=motion,
                    extent_dot=two if self.coordinate == "extent" else (0.0, 0.0),
                    direction_dot=cross(unit, source.direction)
                    if self.coordinate == "beam-direction" and isinstance(source, ParallelBeam)
                    else (0.0, 0.0, 0.0),
                )
            )
        if self.component == "launch-target":
            if not isinstance(source, PointConeSource):
                raise TransportError("a parallel source has no launch-target rectangle")
            return AcquisitionGeometryTangent(
                source=SourceGeometryTangent(
                    target_motion=motion,
                    target_spacing_dot=two if self.coordinate == "pixel-spacing" else (0.0, 0.0),
                )
            )
        return AcquisitionGeometryTangent(
            detector=motion,
            detector_spacing_dot=two if self.coordinate == "pixel-spacing" else (0.0, 0.0),
        )


@dataclass(frozen=True, slots=True)
class GeometryVJP:
    """Gradient in the caller's named coordinate order, contracted with an image seed."""

    parameters: tuple[GeometryParameter, ...]
    gradient: tuple[float, ...]
    columns: tuple[ElasticGeometryEstimate, ...]


@dataclass(frozen=True, slots=True)
class PrefixGeometryVJP:
    """Named geometry transpose with complete physical-history observations."""

    parameters: tuple[GeometryParameter, ...]
    gradient: tuple[float, ...]
    columns: tuple[ElasticPrefixGeometryEstimate, ...]


def elastic_prefix_geometry_vjp(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    medium: ElasticGeometryMedium,
    *,
    parameters: Sequence[GeometryParameter],
    cotangent: Sequence[float],
    histories: Sequence[GeometryPrefixHistory],
    source_boundaries: Mapping[str, Sequence[GeometryBoundaryPrefixHistory]],
) -> PrefixGeometryVJP:
    """Contract named columns with one shared detached physical path proposal.

    Each returned column retains the same original-history identities and sums
    its correlated prefixes before reduction. Boundary records retain separate
    denominators. Pixel seeds may be signed. The caller supplies every boundary
    required by each selected active coordinate; stationary boundary terms have
    zero velocity but retain the same sample identity.
    """
    selected = tuple(parameters)
    if not selected or len(selected) > 64 or len(set(selected)) != len(selected):
        raise TransportError("geometry VJP requires one to sixty-four distinct coordinates")
    if any(not isinstance(cast(object, parameter), GeometryParameter) for parameter in selected):
        raise TransportError("geometry VJP parameters must be GeometryParameter records")
    columns = tuple(
        elastic_prefix_geometry_jvp(
            source,
            spec,
            medium,
            tangent=parameter.tangent(source),
            cotangent=cotangent,
            histories=histories,
            source_boundaries=source_boundaries,
        )
        for parameter in selected
    )
    return PrefixGeometryVJP(selected, tuple(column.derivative for column in columns), columns)


def elastic_geometry_vjp(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    medium: ElasticGeometryMedium,
    *,
    parameters: Sequence[GeometryParameter],
    cotangent: Sequence[float],
    order_law: CollisionOrderLaw,
    histories: Sequence[CollisionPathSample],
    source_boundaries: Mapping[str, Sequence[SourceBoundaryPathSample]],
) -> GeometryVJP:
    """Contract the full elastic path/support JVP against named geometry columns.

    Every column reuses the caller's original complete path coordinates; neither
    random source nor collision coordinates are redrawn. The signed pixel seed
    is contracted inside each path and source-support continuation. Column
    identity and separate boundary/interior terms remain available in the result.
    This host reference returns immutable FP64 Python values without device
    allocation, tape use or implicit data transfer.
    """
    selected = tuple(parameters)
    if not selected or len(selected) > 64 or len(set(selected)) != len(selected):
        raise TransportError("geometry VJP requires one to sixty-four distinct coordinates")
    if any(not isinstance(cast(object, parameter), GeometryParameter) for parameter in selected):
        raise TransportError("geometry VJP parameters must be GeometryParameter records")
    columns = tuple(
        elastic_geometry_jvp(
            source,
            spec,
            medium,
            tangent=parameter.tangent(source),
            cotangent=cotangent,
            order_law=order_law,
            histories=histories,
            source_boundaries=source_boundaries,
        )
        for parameter in selected
    )
    return GeometryVJP(selected, tuple(column.derivative for column in columns), columns)


__all__ = [
    "GeometryParameter",
    "GeometryVJP",
    "PrefixGeometryVJP",
    "elastic_geometry_vjp",
    "elastic_prefix_geometry_vjp",
]
