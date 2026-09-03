"""Explicit TIGRE cone geometry maps with tilted panels and ZYZ acquisitions.

The native physical frame is TIGRE (X,Y,Z), with a zero-angle source on +X,
columns increasing +Y and Python output rows increasing +Z. Geometry fields
offOrigin uses TIGRE's Python (Z,Y,X) order; offDetector is (V,U).
rotDetector contains (rx,ry,rz), applying Rz(rz) Ry(ry) Rx(rx).
Conventions follow CERN/TIGRE commit 6b0951a8a88aa88a7db5b2a95181a33a95aa8b28:
Common/CUDA/ray_interpolated_projection.cu and Python's _types.pxd interface.
All maps use millimetres and radians. No third-party implementation is vendored.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, cast

from dpt.contracts import ContractError, finite_tuple
from dpt.geometry import (
    BatchedGeometry,
    DetectorGeometry,
    Matrix3,
    RigidTransform,
    Vector3,
    cross,
    matmul,
    matvec,
    transpose,
)

_IDENTITY_FRAME = RigidTransform()


def _rotation(axis: int, angle: float) -> Matrix3:
    cosine, sine = math.cos(angle), math.sin(angle)
    if axis == 0:
        return (1.0, 0.0, 0.0, 0.0, cosine, -sine, 0.0, sine, cosine)
    if axis == 1:
        return (cosine, 0.0, sine, 0.0, 1.0, 0.0, -sine, 0.0, cosine)
    return (cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0)


def _acquisition(angles: tuple[float, float, float]) -> Matrix3:
    return matmul(matmul(_rotation(2, angles[0]), _rotation(1, angles[1])), _rotation(2, angles[2]))


def _detector_rotation(angles: tuple[float, float, float]) -> Matrix3:
    return matmul(matmul(_rotation(2, angles[2]), _rotation(1, angles[1])), _rotation(0, angles[0]))


def _rows(values: Any, width: int, count: int, name: str) -> tuple[tuple[float, ...], ...]:
    """Read native scalar/broadcast arrays into owned immutable records."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    if width == 1:
        if not isinstance(values, (tuple, list)):
            values = [values] * count
        scalars = list(cast(tuple[Any, ...] | list[Any], values))
        if len(scalars) == 1:
            scalars *= count
        if len(scalars) != count:
            raise ContractError(f"{name} must contain one scalar per view")
        return tuple(finite_tuple((entry,), name, minimum=None, length=1) for entry in scalars)
    if not isinstance(values, (tuple, list)):
        raise ContractError(f"{name} must contain {width} coordinates per view")
    rows = list(cast(tuple[Any, ...] | list[Any], values))
    if len(rows) == width and not isinstance(rows[0], (tuple, list)):
        rows = [rows] * count
    if len(rows) != count:
        raise ContractError(f"{name} must contain one row per view")
    return tuple(finite_tuple(tuple(row), name, minimum=None, length=width) for row in rows)


@dataclass(frozen=True, slots=True)
class TigreConeGeometry:
    """Own TIGRE cone acquisition values independently of the TIGRE runtime.

    Detector shape and spacing are (rows,columns). Each acquisition angle is
    (alpha,theta,psi), applying Rz(alpha) Ry(theta) Rz(psi). Detector rotation
    is TIGRE rotDetector=(rx,ry,rz), applying Rz(rz) Ry(ry) Rx(rx).
    Origin offsets are (Z,Y,X), detector offsets (V,U), and COR follows TIGRE's
    transverse correction (-COR sin(alpha),COR cos(alpha),0).
    """

    shape: tuple[int, int]
    spacing_rows_columns_mm: tuple[float, float]
    source_object_distance_mm: tuple[float, ...]
    source_detector_distance_mm: tuple[float, ...]
    angles_rad: tuple[tuple[float, float, float], ...]
    detector_offset_rows_columns_mm: tuple[tuple[float, float], ...] = ()
    detector_rotation_xyz_rad: tuple[tuple[float, float, float], ...] = ()
    origin_offset_zyx_mm: tuple[tuple[float, float, float], ...] = ()
    centre_rotation_mm: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(object, self.shape), tuple)
            or len(self.shape) != 2
            or any(type(item) is not int or item <= 0 for item in self.shape)
        ):
            raise ContractError("TIGRE detector shape must be positive (rows,columns)")
        spacing = finite_tuple(
            self.spacing_rows_columns_mm, "TIGRE detector spacing", positive=True, length=2
        )
        object.__setattr__(self, "spacing_rows_columns_mm", spacing)
        if not isinstance(cast(object, self.angles_rad), tuple) or not self.angles_rad:
            raise ContractError("TIGRE angles must be non-empty immutable rows")
        angles = tuple(
            finite_tuple(row, "TIGRE ZYZ angles", minimum=None, length=3) for row in self.angles_rad
        )
        object.__setattr__(self, "angles_rad", angles)
        count = len(angles)
        for name in ("source_object_distance_mm", "source_detector_distance_mm"):
            object.__setattr__(
                self, name, finite_tuple(getattr(self, name), name, positive=True, length=count)
            )
        if any(
            sdd <= sod
            for sod, sdd in zip(
                self.source_object_distance_mm, self.source_detector_distance_mm, strict=True
            )
        ):
            raise ContractError("TIGRE source-detector distance must exceed source-object distance")
        for name, width in (
            ("detector_offset_rows_columns_mm", 2),
            ("detector_rotation_xyz_rad", 3),
            ("origin_offset_zyx_mm", 3),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise ContractError(f"{name} must have one immutable row per view")
            supplied = cast(tuple[Any, ...], values)
            if not supplied:
                supplied = ((0.0,) * width,) * count
            if len(supplied) != count:
                raise ContractError(f"{name} must have one immutable row per view")
            object.__setattr__(
                self,
                name,
                tuple(finite_tuple(row, name, minimum=None, length=width) for row in supplied),
            )
        corrections = self.centre_rotation_mm or (0.0,) * count
        object.__setattr__(
            self, "centre_rotation_mm", finite_tuple(corrections, "COR", minimum=None, length=count)
        )
        # Verify finite panel extents and right-handed metric geometry once.
        self.to_dpt()

    def to_dpt(self, *, world_from_tigre: RigidTransform = _IDENTITY_FRAME) -> BatchedGeometry:
        """Map native physical rays into an explicit DPT world frame."""
        rows, columns = self.shape
        dv, du = self.spacing_rows_columns_mm
        panels: list[DetectorGeometry] = []
        for view, angles in enumerate(self.angles_rad):
            acquisition = _acquisition(angles)
            tilt = _detector_rotation(self.detector_rotation_xyz_rad[view])
            rotation = matmul(acquisition, tilt)
            u = matvec(rotation, (0.0, 1.0, 0.0))
            v = matvec(rotation, (0.0, 0.0, 1.0))
            sod, sdd = self.source_object_distance_mm[view], self.source_detector_distance_mm[view]
            offset_v, offset_u = self.detector_offset_rows_columns_mm[view]
            centre = matvec(acquisition, (sod - sdd, offset_u, offset_v))
            source = matvec(acquisition, (sod, 0.0, 0.0))
            off_z, off_y, off_x = self.origin_offset_zyx_mm[view]
            correction = self.centre_rotation_mm[view]
            translation = (
                -off_x - correction * math.sin(angles[0]),
                -off_y + correction * math.cos(angles[0]),
                -off_z,
            )
            source = cast(Vector3, tuple(source[axis] + translation[axis] for axis in range(3)))
            origin = cast(
                Vector3,
                tuple(
                    centre[axis]
                    + translation[axis]
                    - 0.5 * (columns - 1) * du * u[axis]
                    - 0.5 * (rows - 1) * dv * v[axis]
                    for axis in range(3)
                ),
            )
            panel = DetectorGeometry(
                world_from_tigre.point(source),
                world_from_tigre.point(origin),
                world_from_tigre.direction(u),
                world_from_tigre.direction(v),
                (du, dv),
                self.shape,
            )
            for row in (0, rows - 1):
                for column in (0, columns - 1):
                    endpoint = panel.pixel_centre(row, column)
                    if not all(math.isfinite(value) for value in endpoint) or not math.isfinite(
                        math.dist(panel.source_mm, endpoint)
                    ):
                        raise ContractError(
                            "TIGRE detector extent exceeds the finite segment range"
                        )
            panels.append(panel)
        return BatchedGeometry(tuple(panels))

    @classmethod
    def from_dpt(
        cls,
        geometry: BatchedGeometry,
        *,
        angles_rad: tuple[tuple[float, float, float], ...],
        world_from_tigre: RigidTransform = _IDENTITY_FRAME,
        origin_offset_zyx_mm: tuple[tuple[float, float, float], ...] = (),
        centre_rotation_mm: tuple[float, ...] = (),
    ) -> TigreConeGeometry:
        """Export flat views for explicit acquisition angles and object offsets.

        These calibration values disambiguate the source trajectory. Views
        must share spacing and fit the stated source orbit; export rejects
        arbitrary source translations instead of inventing calibration.
        """
        count = len(geometry.views)
        angles = cast(
            tuple[tuple[float, float, float], ...], _rows(angles_rad, 3, count, "angles_rad")
        )
        offsets = cast(
            tuple[tuple[float, float, float], ...],
            _rows(origin_offset_zyx_mm or ((0.0, 0.0, 0.0),) * count, 3, count, "offOrigin"),
        )
        corrections = tuple(
            row[0] for row in _rows(centre_rotation_mm or (0.0,) * count, 1, count, "COR")
        )
        inverse = world_from_tigre.inverse()
        sdds: list[float] = []
        sods: list[float] = []
        detector_offsets: list[tuple[float, float]] = []
        tilts: list[tuple[float, float, float]] = []
        spacing: tuple[float, float] | None = None
        for view, panel in enumerate(geometry.views):
            if not isinstance(panel, DetectorGeometry):
                raise ContractError("TIGRE cone export requires flat cone detector views")
            if spacing is not None and spacing != panel.spacing_mm:
                raise ContractError("TIGRE views must share detector spacing")
            spacing = panel.spacing_mm
            alpha = angles[view][0]
            off_z, off_y, off_x = offsets[view]
            shift = (
                -off_x - corrections[view] * math.sin(alpha),
                -off_y + corrections[view] * math.cos(alpha),
                -off_z,
            )
            unrotate = transpose(_acquisition(angles[view]))
            source = inverse.point(panel.source_mm)
            source_base = matvec(
                unrotate, cast(Vector3, tuple(source[axis] - shift[axis] for axis in range(3)))
            )
            if source_base[0] <= 0.0 or max(abs(source_base[1]), abs(source_base[2])) > 1e-9 * max(
                1.0, abs(source_base[0])
            ):
                raise ContractError("source does not match the supplied TIGRE acquisition orbit")
            u = matvec(unrotate, inverse.direction(panel.u))
            v = matvec(unrotate, inverse.direction(panel.v))
            normal = cross(u, v)
            rotation: Matrix3 = (
                normal[0],
                u[0],
                v[0],
                normal[1],
                u[1],
                v[1],
                normal[2],
                u[2],
                v[2],
            )
            ry = math.asin(max(-1.0, min(1.0, -rotation[6])))
            if abs(math.cos(ry)) > 1e-10:
                rz = math.atan2(rotation[3], rotation[0])
                rx = math.atan2(rotation[7], rotation[8])
            else:
                rz, rx = math.atan2(-rotation[1], rotation[4]), 0.0
            tilts.append((rx, ry, rz))
            rows, columns = panel.shape
            centre = cast(
                Vector3,
                tuple(
                    panel.origin_mm[axis]
                    + 0.5 * (columns - 1) * panel.spacing_mm[0] * panel.u[axis]
                    + 0.5 * (rows - 1) * panel.spacing_mm[1] * panel.v[axis]
                    for axis in range(3)
                ),
            )
            centre_native = inverse.point(centre)
            centre_base = matvec(
                unrotate,
                cast(Vector3, tuple(centre_native[axis] - shift[axis] for axis in range(3))),
            )
            sods.append(source_base[0])
            sdds.append(source_base[0] - centre_base[0])
            detector_offsets.append((centre_base[2], centre_base[1]))
        assert spacing is not None
        return cls(
            geometry.shape_rows_columns,
            (spacing[1], spacing[0]),
            tuple(sods),
            tuple(sdds),
            angles,
            tuple(detector_offsets),
            tuple(tilts),
            offsets,
            corrections,
        )

    @classmethod
    def from_tigre(cls, geometry: Any, angles_rad: Any) -> TigreConeGeometry:
        """Copy an existing native TIGRE geometry without changing its arrays."""
        if getattr(geometry, "mode", "cone") != "cone":
            raise ContractError("TIGRE import requires cone mode")
        angles = angles_rad.tolist() if hasattr(angles_rad, "tolist") else angles_rad
        if not isinstance(angles, (tuple, list)) or not angles:
            raise ContractError("TIGRE angles must be non-empty")
        angle_rows = list(cast(tuple[Any, ...] | list[Any], angles))
        if not isinstance(angle_rows[0], (tuple, list)):
            angle_rows = [(angle, 0.0, 0.0) for angle in angle_rows]
        count = len(angle_rows)
        shape_value = (
            geometry.nDetector.tolist()
            if hasattr(geometry.nDetector, "tolist")
            else geometry.nDetector
        )
        shape = tuple(shape_value)
        spacing_value = (
            geometry.dDetector.tolist()
            if hasattr(geometry.dDetector, "tolist")
            else geometry.dDetector
        )
        return cls(
            cast(tuple[int, int], shape),
            cast(tuple[float, float], tuple(spacing_value)),
            tuple(row[0] for row in _rows(geometry.DSO, 1, count, "DSO")),
            tuple(row[0] for row in _rows(geometry.DSD, 1, count, "DSD")),
            cast(tuple[tuple[float, float, float], ...], _rows(angle_rows, 3, count, "angles")),
            cast(
                tuple[tuple[float, float], ...],
                _rows(getattr(geometry, "offDetector", (0.0, 0.0)), 2, count, "offDetector"),
            ),
            cast(
                tuple[tuple[float, float, float], ...],
                _rows(getattr(geometry, "rotDetector", (0.0, 0.0, 0.0)), 3, count, "rotDetector"),
            ),
            cast(
                tuple[tuple[float, float, float], ...],
                _rows(getattr(geometry, "offOrigin", (0.0, 0.0, 0.0)), 3, count, "offOrigin"),
            ),
            tuple(row[0] for row in _rows(getattr(geometry, "COR", 0.0), 1, count, "COR")),
        )

    def to_tigre(self) -> tuple[Any, Any]:
        """Create a native TIGRE cone geometry and owned NumPy acquisition angles.

        Volume discretisation remains the caller's choice: set nVoxel,dVoxel,
        sVoxel before passing the geometry to TIGRE's projector.
        """
        np: Any = importlib.import_module("numpy")
        tigre: Any = importlib.import_module("tigre")
        geometry = tigre.geometry()
        geometry.mode = "cone"
        geometry.nDetector = np.array(self.shape, dtype=np.int32)
        geometry.dDetector = np.array(self.spacing_rows_columns_mm, dtype=np.float64)
        geometry.sDetector = geometry.nDetector * geometry.dDetector
        for name, value in (
            ("DSO", self.source_object_distance_mm),
            ("DSD", self.source_detector_distance_mm),
            ("offDetector", self.detector_offset_rows_columns_mm),
            ("rotDetector", self.detector_rotation_xyz_rad),
            ("offOrigin", self.origin_offset_zyx_mm),
            ("COR", self.centre_rotation_mm),
        ):
            setattr(geometry, name, np.array(value, dtype=np.float64))
        return geometry, np.array(self.angles_rad, dtype=np.float64)
