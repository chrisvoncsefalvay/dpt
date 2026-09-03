"""Validated host arguments for native material-interface primitives."""

from __future__ import annotations

from dataclasses import dataclass

from dpt.geometry import Matrix3, Vector3

from .geometry_interfaces import MaterialInterface, interface_point_tangent
from .geometry_tangents import GridGeometryTangent
from .model import TransportSpec


@dataclass(frozen=True, slots=True)
class InterfaceGridArguments:
    origin: Vector3
    spacing: Vector3
    shape: tuple[int, int, int]
    rotation: Matrix3
    translation: Vector3
    origin_dot: Vector3
    spacing_dot: Vector3
    motion_translation: Vector3
    motion_angular: Vector3
    motion_pivot: Vector3


@dataclass(frozen=True, slots=True)
class InterfacePatchArguments:
    axis: int
    face: int
    tangential_cells: tuple[int, int]
    negative_cell: int
    positive_cell: int


def interface_arguments(
    interface: MaterialInterface,
    spec: TransportSpec,
    tangent: GridGeometryTangent | None = None,
) -> tuple[InterfacePatchArguments, InterfaceGridArguments]:
    """Validate explicit cell sides and copy immutable binary64 geometry fields.

    Exterior vacuum is encoded by cell index ``-1``. The face normal points
    towards its positive axis, and shape is returned in device ``(x,y,z)`` order.
    The caller packs these values into the corresponding native Warp structs.
    """
    tangent = GridGeometryTangent() if tangent is None else tangent
    interface_point_tangent(interface, (0.5, 0.5), spec, tangent)
    return (
        InterfacePatchArguments(
            interface.axis,
            interface.face,
            interface.tangential_cells,
            -1 if interface.negative_cell is None else interface.negative_cell,
            -1 if interface.positive_cell is None else interface.positive_cell,
        ),
        InterfaceGridArguments(
            spec.grid.origin_mm,
            spec.grid.spacing_mm,
            spec.grid.shape_xyz,
            spec.object_pose.rotation,
            spec.object_pose.translation_mm,
            tangent.origin_dot,
            tangent.spacing_dot,
            tangent.motion.translation,
            tangent.motion.angular,
            tangent.motion.pivot_mm,
        ),
    )


__all__ = ["InterfaceGridArguments", "InterfacePatchArguments", "interface_arguments"]
