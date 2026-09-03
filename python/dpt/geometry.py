"""Right-handed rigid geometry in millimetres, with explicit object-to-world maps.

These immutable host contracts import without Warp. Rotations are row-major
matrices acting on column vectors; an acquisition origin is its first pixel
centre. Small pose transfers belong at the optimisation control boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, cast

from dpt.contracts import ContractError, finite_tuple

type Vector3 = tuple[float, float, float]
type Matrix3 = tuple[float, float, float, float, float, float, float, float, float]
IDENTITY: Matrix3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def vector3(value: Vector3, name: str) -> Vector3:
    return cast(Vector3, finite_tuple(value, name, minimum=None, length=3))


def dot(a: Vector3, b: Vector3) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def cross(a: Vector3, b: Vector3) -> Vector3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def matvec(matrix: Matrix3, vector: Vector3) -> Vector3:
    return cast(
        Vector3, tuple(sum(matrix[3 * i + j] * vector[j] for j in range(3)) for i in range(3))
    )


def transpose(matrix: Matrix3) -> Matrix3:
    return cast(Matrix3, tuple(matrix[3 * j + i] for i in range(3) for j in range(3)))


def matmul(left: Matrix3, right: Matrix3) -> Matrix3:
    return cast(
        Matrix3,
        tuple(
            sum(left[3 * i + k] * right[3 * k + j] for k in range(3))
            for i in range(3)
            for j in range(3)
        ),
    )


def rotation_matrix(value: Matrix3, name: str = "rotation") -> Matrix3:
    value = cast(Matrix3, finite_tuple(value, name, minimum=None, length=9))
    product = matmul(transpose(value), value)
    if max(abs(a - b) for a, b in zip(product, IDENTITY, strict=True)) > 1e-10:
        raise ContractError(f"{name} must be orthonormal to 1e-10")
    columns = [(value[i], value[i + 3], value[i + 6]) for i in range(3)]
    if dot(cross(columns[0], columns[1]), columns[2]) < 0.0:
        raise ContractError(f"{name} must be right-handed")
    return cast(Matrix3, tuple(float(x) for x in value))


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """T_WO maps object points into world coordinates; translation is in mm."""

    rotation: Matrix3 = IDENTITY
    translation_mm: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", rotation_matrix(self.rotation))
        object.__setattr__(self, "translation_mm", vector3(self.translation_mm, "translation"))

    def point(self, point_mm: Vector3) -> Vector3:
        rotated = matvec(self.rotation, point_mm)
        return cast(Vector3, tuple(rotated[i] + self.translation_mm[i] for i in range(3)))

    def direction(self, direction: Vector3) -> Vector3:
        return matvec(self.rotation, direction)

    def inverse(self) -> RigidTransform:
        inverse_rotation = transpose(self.rotation)
        translation = matvec(inverse_rotation, self.translation_mm)
        return RigidTransform(inverse_rotation, cast(Vector3, tuple(-x for x in translation)))

    def compose(self, right: RigidTransform) -> RigidTransform:
        """Return self @ right; the right-hand map is applied first."""
        return RigidTransform(
            matmul(self.rotation, right.rotation), self.point(right.translation_mm)
        )

    def packed(self) -> tuple[float, ...]:
        """Twelve float64 device values: row-major rotation, then translation."""
        return (*self.rotation, *self.translation_mm)


@dataclass(frozen=True, slots=True)
class DetectorGeometry:
    """Finite source-to-pixel segments; u increases columns and v increases rows.

    Shape is (rows, columns), while spacing_mm is (column spacing, row spacing).
    """

    source_mm: Vector3
    origin_mm: Vector3
    u: Vector3
    v: Vector3
    spacing_mm: tuple[float, float]
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        for name in ("source_mm", "origin_mm", "u", "v"):
            object.__setattr__(self, name, vector3(getattr(self, name), name))
        if abs(dot(self.u, self.u) - 1.0) > 1e-10 or abs(dot(self.v, self.v) - 1.0) > 1e-10:
            raise ContractError("detector basis vectors must have unit length")
        if abs(dot(self.u, self.v)) > 1e-10:
            raise ContractError("detector basis vectors must be perpendicular")
        spacing = finite_tuple(self.spacing_mm, "pixel spacing", positive=True, length=2)
        if len(self.shape) != 2 or any(type(x) is not int or x <= 0 for x in self.shape):
            raise ContractError("detector shape is (height, width), both positive integers")
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "spacing_mm", spacing)
        offset: Vector3 = cast(
            Vector3, tuple(self.source_mm[i] - self.origin_mm[i] for i in range(3))
        )
        if abs(dot(offset, cross(self.u, self.v))) <= 1e-12:
            raise ContractError("source must lie outside the detector plane")
        if self.pixels > 2**31 - 1:
            raise ContractError("detector exceeds signed 32-bit indexing")

    @classmethod
    def from_axes(
        cls,
        *,
        source_mm: Vector3,
        origin_mm: Vector3,
        u: Vector3,
        v: Vector3,
        rows: int,
        columns: int,
        spacing_column_mm: float,
        spacing_row_mm: float,
    ) -> DetectorGeometry:
        """Construct with named pixel counts and spacings for each detector axis."""
        return cls(
            source_mm,
            origin_mm,
            u,
            v,
            (spacing_column_mm, spacing_row_mm),
            (rows, columns),
        )

    @property
    def shape_rows_columns(self) -> tuple[int, int]:
        """Pixel counts in (rows, columns) order, matching the output image."""
        return self.shape

    @property
    def spacing_columns_rows_mm(self) -> tuple[float, float]:
        """Pixel spacings along (u, v), in (columns, rows) order."""
        return self.spacing_mm

    @property
    def pixels(self) -> int:
        return self.shape[0] * self.shape[1]

    def pixel_centre(self, row: int, column: int) -> Vector3:
        if type(row) is not int or type(column) is not int:
            raise ContractError("pixel indices must be integers")
        if not 0 <= row < self.shape[0] or not 0 <= column < self.shape[1]:
            raise ContractError("pixel index outside detector")
        return cast(
            Vector3,
            tuple(
                self.origin_mm[i]
                + column * self.spacing_mm[0] * self.u[i]
                + row * self.spacing_mm[1] * self.v[i]
                for i in range(3)
            ),
        )

    def projection_matrix(self) -> tuple[tuple[float, float, float, float], ...]:
        """Map ``[x,y,z,1]`` to homogeneous ``[column,row,depth]``.

        ``column = (P X)[0]/(P X)[2]`` and ``row = (P X)[1]/(P X)[2]``.
        Coordinates are world millimetres; pixel (0,0) is ``origin_mm``.
        For example, applying this matrix to ``pixel_centre(2,3)`` gives
        column 3 and row 2 after homogeneous division.
        """
        normal = cross(self.u, self.v)
        offset = cast(Vector3, tuple(self.origin_mm[i] - self.source_mm[i] for i in range(3)))
        distance = dot(normal, offset)
        rows: list[tuple[float, float, float, float]] = []
        for axis, spacing in zip((self.u, self.v), self.spacing_mm, strict=True):
            coefficient = cast(
                Vector3,
                tuple(
                    (distance * axis[i] - dot(axis, offset) * normal[i]) / spacing for i in range(3)
                ),
            )
            rows.append((*coefficient, -dot(coefficient, self.source_mm)))
        rows.append((*normal, -dot(normal, self.source_mm)))
        return tuple(rows)

    def project_point(self, point_mm: Vector3) -> tuple[float, float]:
        """Return ``(row,column)`` where the source-to-point line meets the panel.

        The result is in continuous pixel coordinates and may lie outside the
        panel. For example, ``project_point(pixel_centre(2,3)) == (2,3)``.
        """
        point = (*vector3(point_mm, "point_mm"), 1.0)
        projected = tuple(
            sum(a * b for a, b in zip(row, point, strict=True)) for row in self.projection_matrix()
        )
        if abs(projected[2]) <= 1e-12:
            raise ContractError("source-to-point line is parallel to the detector plane")
        return projected[1] / projected[2], projected[0] / projected[2]


@dataclass(frozen=True, slots=True)
class RayBundleGeometry:
    """Explicit finite world segments, with one flattened output per segment.

    Each source/end pair defines the full finite segment; directions are not
    implicitly normalised and there are no infinite or extended rays. All
    segments use the projection workspace's shared object pose. view_offsets
    groups rays for acquisitions with a common pose, without averaging paths
    or assigning detector-aperture physics.
    """

    sources_mm: tuple[Vector3, ...]
    endpoints_mm: tuple[Vector3, ...]
    view_offsets: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(object, self.sources_mm), tuple)
            or not isinstance(cast(object, self.endpoints_mm), tuple)
            or not self.sources_mm
            or len(self.sources_mm) != len(self.endpoints_mm)
        ):
            raise ContractError("ray sources/endpoints must be equal non-empty immutable tuples")
        sources = tuple(vector3(source, "ray source") for source in self.sources_mm)
        endpoints = tuple(vector3(endpoint, "ray endpoint") for endpoint in self.endpoints_mm)
        if any(
            not math.isfinite(math.dist(source, endpoint)) or source == endpoint
            for source, endpoint in zip(sources, endpoints, strict=True)
        ):
            raise ContractError("ray segments must have finite positive length")
        if self.pixels > 2**31 - 1:
            raise ContractError("ray bundle exceeds signed 32-bit indexing")
        offsets = self.view_offsets if self.view_offsets is not None else (0, self.pixels)
        if (
            not isinstance(cast(object, offsets), tuple)
            or len(offsets) < 2
            or any(type(offset) is not int for offset in offsets)
            or offsets[0] != 0
            or offsets[-1] != self.pixels
            or any(left >= right for left, right in pairwise(offsets))
        ):
            raise ContractError("view_offsets must partition the bundle into non-empty views")
        object.__setattr__(self, "sources_mm", sources)
        object.__setattr__(self, "endpoints_mm", endpoints)
        object.__setattr__(self, "view_offsets", offsets)

    @classmethod
    def from_views(cls, views: tuple[DetectorGeometry, ...]) -> RayBundleGeometry:
        """Flatten flat panels in view, row, column order, retaining view offsets."""
        if (
            not isinstance(cast(object, views), tuple)
            or not views
            or any(not isinstance(cast(object, view), DetectorGeometry) for view in views)
        ):
            raise ContractError("ray bundle views must be a non-empty tuple of DetectorGeometry")
        sources: list[Vector3] = []
        endpoints: list[Vector3] = []
        offsets = [0]
        for view in views:
            for row in range(view.shape[0]):
                for column in range(view.shape[1]):
                    sources.append(view.source_mm)
                    endpoints.append(view.pixel_centre(row, column))
            offsets.append(len(sources))
        return cls(tuple(sources), tuple(endpoints), tuple(offsets))

    @property
    def pixels(self) -> int:
        """Number of independently projected finite segments."""
        return len(self.sources_mm)

    @property
    def rays(self) -> int:
        return self.pixels

    def segment(self, ray: int) -> tuple[Vector3, Vector3]:
        if type(ray) is not int or not 0 <= ray < self.pixels:
            raise ContractError("ray index must be an integer inside the bundle")
        return self.sources_mm[ray], self.endpoints_mm[ray]


@dataclass(frozen=True, slots=True)
class ParallelGeometry:
    """Finite parallel segments ``s[r,c] = p[r,c] - length * direction``.

    ``origin_mm`` is the endpoint of pixel (0,0), ``u`` increases columns and
    ``v`` increases rows. ``shape`` is (rows,columns), and ``spacing_mm`` is
    (column,row). All lengths are millimetres. For example, direction (0,0,1)
    and length 200 place each source 200 mm behind its endpoint.
    """

    origin_mm: Vector3
    direction: Vector3
    length_mm: float
    u: Vector3
    v: Vector3
    spacing_mm: tuple[float, float]
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        direction = vector3(self.direction, "parallel direction")
        if abs(dot(direction, direction) - 1.0) > 1e-10:
            raise ContractError("parallel direction must have unit length")
        length = finite_tuple((self.length_mm,), "parallel length", positive=True)[0]
        origin = vector3(self.origin_mm, "parallel origin")
        source = cast(Vector3, tuple(origin[i] - length * direction[i] for i in range(3)))
        panel = DetectorGeometry(source, origin, self.u, self.v, self.spacing_mm, self.shape)
        object.__setattr__(self, "origin_mm", origin)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "length_mm", length)
        for name in ("u", "v", "spacing_mm", "shape"):
            object.__setattr__(self, name, getattr(panel, name))

    @property
    def pixels(self) -> int:
        return self.shape[0] * self.shape[1]

    def pixel_centre(self, row: int, column: int) -> Vector3:
        _pixel_index(row, column, self.shape)
        return cast(
            Vector3,
            tuple(
                self.origin_mm[i]
                + column * self.spacing_mm[0] * self.u[i]
                + row * self.spacing_mm[1] * self.v[i]
                for i in range(3)
            ),
        )

    def segment(self, row: int, column: int) -> tuple[Vector3, Vector3]:
        """Return source and endpoint for the requested (row,column)."""
        endpoint = self.pixel_centre(row, column)
        return cast(
            Vector3, tuple(endpoint[i] - self.length_mm * self.direction[i] for i in range(3))
        ), endpoint


def _pixel_index(row: int, column: int, shape: tuple[int, int]) -> None:
    if type(row) is not int or type(column) is not int:
        raise ContractError("pixel indices must be integers")
    if not 0 <= row < shape[0] or not 0 <= column < shape[1]:
        raise ContractError("pixel index outside detector")


@dataclass(frozen=True, slots=True)
class CurvedDetectorGeometry:
    """Cylindrical pixels ``p = axis + radius*cos(theta) + u*R*sin(theta)``.

    The cylinder axis passes through ``source_mm`` along ``v``. Pixel (0,0)
    is ``origin_mm``; increasing columns follows tangent ``u`` with arc length
    ``spacing_mm[0]``. Rows move along ``v`` by ``spacing_mm[1]``. Shape is
    (rows,columns). A 1 mm column spacing on a 100 mm radius gives 0.01 rad
    between adjacent columns.
    """

    source_mm: Vector3
    origin_mm: Vector3
    u: Vector3
    v: Vector3
    spacing_mm: tuple[float, float]
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        panel = DetectorGeometry(
            self.source_mm, self.origin_mm, self.u, self.v, self.spacing_mm, self.shape
        )
        for name in ("source_mm", "origin_mm", "u", "v", "spacing_mm", "shape"):
            object.__setattr__(self, name, getattr(panel, name))
        radial = self.radial_mm
        if not math.isfinite(self.radius_mm) or self.radius_mm <= 0.0:
            raise ContractError("curved detector radius must be finite and positive")
        if abs(dot(radial, self.u)) > 1e-10 * self.radius_mm:
            raise ContractError("curved detector u must be tangent to its source-centred cylinder")
        if self.spacing_mm[0] * (self.shape[1] - 1) >= 2 * math.pi * self.radius_mm:
            raise ContractError("curved detector columns must cover less than one full revolution")

    @property
    def radial_mm(self) -> Vector3:
        offset = cast(Vector3, tuple(self.origin_mm[i] - self.source_mm[i] for i in range(3)))
        return cast(Vector3, tuple(offset[i] - dot(offset, self.v) * self.v[i] for i in range(3)))

    @property
    def radius_mm(self) -> float:
        return math.sqrt(dot(self.radial_mm, self.radial_mm))

    @property
    def pixels(self) -> int:
        return self.shape[0] * self.shape[1]

    def pixel_centre(self, row: int, column: int) -> Vector3:
        _pixel_index(row, column, self.shape)
        radial = self.radial_mm
        angle = column * self.spacing_mm[0] / self.radius_mm
        return cast(
            Vector3,
            tuple(
                self.origin_mm[i]
                + (math.cos(angle) - 1.0) * radial[i]
                + math.sin(angle) * self.radius_mm * self.u[i]
                + row * self.spacing_mm[1] * self.v[i]
                for i in range(3)
            ),
        )


type PanelGeometry = DetectorGeometry | ParallelGeometry | CurvedDetectorGeometry
type AstraVector = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AstraVectorGeometry:
    """ASTRA vector rows ``(source/direction,centre,u_pixel,v_pixel)``.

    ``vectors`` has shape (views,12); ``shape_rows_columns`` is the detector
    shape. Positions and pixel vectors are millimetres. ``kind`` is cone_vec
    or parallel3d_vec. Pass these values to ASTRA's ``create_proj_geom``.
    """

    kind: Literal["cone_vec", "parallel3d_vec"]
    shape_rows_columns: tuple[int, int]
    vectors: tuple[AstraVector, ...]

    def __post_init__(self) -> None:
        if self.kind not in ("cone_vec", "parallel3d_vec"):
            raise ContractError("ASTRA kind must be cone_vec or parallel3d_vec")
        _panel_shape(self.shape_rows_columns)
        if not isinstance(cast(object, self.vectors), tuple) or not self.vectors:
            raise ContractError("ASTRA vectors must be a non-empty immutable tuple")
        object.__setattr__(
            self,
            "vectors",
            tuple(
                finite_tuple(row, "ASTRA vector", minimum=None, length=12) for row in self.vectors
            ),
        )


@dataclass(frozen=True, slots=True)
class BatchedGeometry:
    """Store O(views) panel metadata for an image of shape (views,rows,columns).

    Rays flatten in view,row,column order. ``prepare_geometry`` generates
    source/end arrays on the device; no Python pixel tuples are constructed.
    For example, ``BatchedGeometry((front_panel,side_panel))`` has two views
    with the same detector shape, even when their pixel spacings differ.
    """

    views: tuple[PanelGeometry, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(object, self.views), tuple)
            or not self.views
            or any(
                not isinstance(
                    cast(object, view), (DetectorGeometry, ParallelGeometry, CurvedDetectorGeometry)
                )
                for view in self.views
            )
        ):
            raise ContractError("views must be a non-empty immutable tuple of panel geometries")
        if any(view.shape != self.views[0].shape for view in self.views):
            raise ContractError("batched views must share (rows,columns)")
        if self.pixels > 2**31 - 1:
            raise ContractError("batched geometry exceeds signed 32-bit indexing")

    @property
    def shape_rows_columns(self) -> tuple[int, int]:
        return self.views[0].shape

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.views), *self.shape_rows_columns

    @property
    def pixels(self) -> int:
        return len(self.views) * self.views[0].pixels

    @property
    def view_offsets(self) -> tuple[int, ...]:
        count = self.views[0].pixels
        return tuple(view * count for view in range(len(self.views) + 1))

    def segment(self, view: int, row: int, column: int) -> tuple[Vector3, Vector3]:
        """Return one finite segment without expanding any other pixels."""
        if type(view) is not int or not 0 <= view < len(self.views):
            raise ContractError("view index must be inside the batch")
        panel = self.views[view]
        if isinstance(panel, ParallelGeometry):
            return panel.segment(row, column)
        return panel.source_mm, panel.pixel_centre(row, column)

    def to_astra_vectors(self) -> AstraVectorGeometry:
        """Export flat panels to ASTRA cone_vec or parallel3d_vec in millimetres.

        ASTRA stores the panel centre, whereas DPT stores its first pixel
        centre. The conversion applies half the (rows-1,columns-1) offset.
        DPT's finite parallel length stays in DPT; ASTRA rays are unbounded.
        """
        parallel = isinstance(self.views[0], ParallelGeometry)
        if any(
            isinstance(view, CurvedDetectorGeometry)
            or isinstance(view, ParallelGeometry) != parallel
            for view in self.views
        ):
            raise ContractError(
                "ASTRA vector export requires uniformly flat cone or parallel views"
            )
        rows, columns = self.shape_rows_columns
        vectors: list[AstraVector] = []
        for view in self.views:
            u_pixel = tuple(view.spacing_mm[0] * value for value in view.u)
            v_pixel = tuple(view.spacing_mm[1] * value for value in view.v)
            centre = tuple(
                view.origin_mm[i] + 0.5 * (columns - 1) * u_pixel[i] + 0.5 * (rows - 1) * v_pixel[i]
                for i in range(3)
            )
            source = view.direction if isinstance(view, ParallelGeometry) else view.source_mm
            vectors.append((*source, *centre, *u_pixel, *v_pixel))
        return AstraVectorGeometry(
            "parallel3d_vec" if parallel else "cone_vec", self.shape_rows_columns, tuple(vectors)
        )

    @classmethod
    def from_astra_vectors(
        cls,
        geometry: AstraVectorGeometry,
        *,
        length_scale_mm: float = 1.0,
        parallel_length_mm: float | None = None,
    ) -> BatchedGeometry:
        """Import ASTRA rows using ``DPT_mm = ASTRA_coordinate*length_scale_mm``.

        Parallel imports require ``parallel_length_mm`` to choose DPT's finite
        source-to-panel extent. For example, a scale of 10 converts centimetre
        positions and pixel vectors into millimetres.
        """
        scale = finite_tuple((length_scale_mm,), "length scale", positive=True)[0]
        if geometry.kind not in ("cone_vec", "parallel3d_vec"):
            raise ContractError("ASTRA kind must be cone_vec or parallel3d_vec")
        if geometry.kind == "parallel3d_vec" and parallel_length_mm is None:
            raise ContractError("parallel ASTRA import requires parallel_length_mm")
        rows, columns = geometry.shape_rows_columns
        views: list[PanelGeometry] = []
        for values in geometry.vectors:
            values = finite_tuple(values, "ASTRA vector", minimum=None, length=12)
            centre = cast(Vector3, tuple(scale * x for x in values[3:6]))
            u_raw = cast(Vector3, tuple(scale * x for x in values[6:9]))
            v_raw = cast(Vector3, tuple(scale * x for x in values[9:12]))
            du, dv = math.sqrt(dot(u_raw, u_raw)), math.sqrt(dot(v_raw, v_raw))
            if min(du, dv) <= 0.0:
                raise ContractError("ASTRA pixel vectors must have positive length")
            u, v = (
                cast(Vector3, tuple(x / du for x in u_raw)),
                cast(Vector3, tuple(x / dv for x in v_raw)),
            )
            origin = cast(
                Vector3,
                tuple(
                    centre[i] - 0.5 * (columns - 1) * u_raw[i] - 0.5 * (rows - 1) * v_raw[i]
                    for i in range(3)
                ),
            )
            source = cast(Vector3, tuple(values[:3]))
            if geometry.kind == "cone_vec":
                views.append(
                    DetectorGeometry(
                        cast(Vector3, tuple(scale * x for x in source)),
                        origin,
                        u,
                        v,
                        (du, dv),
                        (rows, columns),
                    )
                )
            else:
                norm = math.sqrt(dot(source, source))
                if norm <= 0.0:
                    raise ContractError("ASTRA parallel direction must have positive length")
                views.append(
                    ParallelGeometry(
                        origin,
                        cast(Vector3, tuple(x / norm for x in source)),
                        cast(float, parallel_length_mm),
                        u,
                        v,
                        (du, dv),
                        (rows, columns),
                    )
                )
        return cls(tuple(views))


@dataclass(frozen=True, slots=True)
class ConeBeamIntrinsics:
    """Circular cone geometry with source ``(0,-SOD,0)`` before rotation.

    ``SDD > SOD > 0`` are millimetres. The detector centre is
    ``(offset_column,SDD-SOD,offset_row)``; rotation is around world z.
    Detector shape is (rows,columns), spacing and offsets are (column,row).
    ``view(0)`` therefore places the source on negative y and the panel on
    positive y. One row yields a fan-beam acquisition.
    """

    source_object_distance_mm: float
    source_detector_distance_mm: float
    shape: tuple[int, int]
    spacing_mm: tuple[float, float]
    detector_offset_mm: tuple[float, float] = (0.0, 0.0)
    centre_mm: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        _panel_shape(self.shape)
        sod, sdd = finite_tuple(
            (self.source_object_distance_mm, self.source_detector_distance_mm),
            "SOD/SDD",
            positive=True,
            length=2,
        )
        if sdd <= sod:
            raise ContractError("source-detector distance must exceed source-object distance")
        object.__setattr__(self, "source_object_distance_mm", sod)
        object.__setattr__(self, "source_detector_distance_mm", sdd)
        object.__setattr__(
            self,
            "spacing_mm",
            finite_tuple(self.spacing_mm, "pixel spacing", positive=True, length=2),
        )
        object.__setattr__(
            self,
            "detector_offset_mm",
            finite_tuple(self.detector_offset_mm, "detector offset", minimum=None, length=2),
        )
        object.__setattr__(self, "centre_mm", vector3(self.centre_mm, "centre_mm"))
        # The existing panel contract supplies count/basis/finite-length checks.
        self.view(0.0)

    def view(self, angle_rad: float) -> DetectorGeometry:
        """Build one O(1) view at the specified rotation angle in radians."""
        angle = finite_tuple((angle_rad,), "view angle", minimum=None)[0]
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation: Matrix3 = (cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0)
        transform = RigidTransform(rotation, self.centre_mm)
        rows, columns = self.shape
        origin: Vector3 = (
            self.detector_offset_mm[0] - 0.5 * (columns - 1) * self.spacing_mm[0],
            self.source_detector_distance_mm - self.source_object_distance_mm,
            self.detector_offset_mm[1] - 0.5 * (rows - 1) * self.spacing_mm[1],
        )
        return DetectorGeometry(
            transform.point((0.0, -self.source_object_distance_mm, 0.0)),
            transform.point(origin),
            transform.direction((1.0, 0.0, 0.0)),
            transform.direction((0.0, 0.0, 1.0)),
            self.spacing_mm,
            self.shape,
        )

    def batch(self, angles_rad: tuple[float, ...]) -> BatchedGeometry:
        """Build compact view descriptions for an image of shape (V,rows,columns)."""
        return BatchedGeometry(
            tuple(self.view(angle) for angle in finite_tuple(angles_rad, "angles", minimum=None))
        )

    def ray_derivatives(self, angle_rad: float) -> tuple[tuple[Vector3, Vector3], ...]:
        """Return ``(d_source,d_endpoint)`` for SOD,SDD,column offset,row offset.

        The four pairs apply to every pixel in this view. Derivatives are
        dimensionless and ordered as the intrinsic parameters above. These
        are derivatives of ray coordinates; an optical-depth VJP combines
        them with the projector's source and endpoint cotangents.
        """
        angle = finite_tuple((angle_rad,), "view angle", minimum=None)[0]
        x: Vector3 = (math.cos(angle), math.sin(angle), 0.0)
        y: Vector3 = (-math.sin(angle), math.cos(angle), 0.0)
        minus_y = cast(Vector3, tuple(-value for value in y))
        zero: Vector3 = (0.0, 0.0, 0.0)
        return ((minus_y, minus_y), (zero, y), (zero, x), (zero, (0.0, 0.0, 1.0)))


def _panel_shape(shape: tuple[int, int]) -> None:
    if (
        not isinstance(cast(object, shape), tuple)
        or len(shape) != 2
        or any(type(count) is not int or count <= 0 for count in shape)
    ):
        raise ContractError("detector shape must be (rows,columns), both positive integers")
    if shape[0] * shape[1] > 2**31 - 1:
        raise ContractError("detector exceeds signed 32-bit indexing")


def _skew(vector: Vector3) -> Matrix3:
    x, y, z = vector
    return (0.0, -z, y, z, 0.0, -x, -y, x, 0.0)


def _coefficients(theta2: float) -> tuple[float, float, float, float, float]:
    # Series avoid cancellation in both the exponential and its derivative.
    if not math.isfinite(theta2):
        raise ContractError("rotation norm exceeds the finite geometry range")
    if theta2 < 1e-2:
        a = 1 - theta2 / 6 + theta2**2 / 120 - theta2**3 / 5040
        b = 0.5 - theta2 / 24 + theta2**2 / 720 - theta2**3 / 40320
        c = 1 / 6 - theta2 / 120 + theta2**2 / 5040 - theta2**3 / 362880
        db = -1 / 12 + theta2 / 180 - theta2**2 / 6720 + theta2**3 / 453600
        dc = -1 / 60 + theta2 / 1260 - theta2**2 / 60480 + theta2**3 / 4989600
        return a, b, c, db, dc
    theta = math.sqrt(theta2)
    sin, cos = math.sin(theta), math.cos(theta)
    return (
        sin / theta,
        (1 - cos) / theta2,
        (theta - sin) / (theta2 * theta),
        (theta * sin - 2 * (1 - cos)) / theta2**2,
        (3 * sin - theta * (2 + cos)) / (theta2**2 * theta),
    )


def compose_pose(anchor: RigidTransform, increment: tuple[float, ...]) -> RigidTransform:
    """Compute ``G(xi) = anchor @ exp(xi^)`` for a six-coordinate increment.

    ``increment`` has shape ``(6,)`` and order ``(tx,ty,tz,rx,ry,rz)``:
    translation coordinates in mm, then rotation coordinates in radians.
    The SE(3) exponential couples translation and rotation through its left
    Jacobian. ``anchor`` and the returned ``RigidTransform`` map object points
    into world coordinates; the increment acts on the right, in object axes.
    A solver retains its fixed anchor while using the same curvature history.

    A quarter-turn anchor maps a local x displacement onto world y.

    """
    increment = finite_tuple(increment, "pose increment", minimum=None, length=6)
    translation: Vector3 = (increment[0], increment[1], increment[2])
    rotation: Vector3 = (increment[3], increment[4], increment[5])
    omega = _skew(rotation)
    omega2 = matmul(omega, omega)
    a, b, c, _, _ = _coefficients(dot(rotation, rotation))
    matrix: Matrix3 = cast(
        Matrix3, tuple(IDENTITY[i] + a * omega[i] + b * omega2[i] for i in range(9))
    )
    velocity: Matrix3 = cast(
        Matrix3, tuple(IDENTITY[i] + b * omega[i] + c * omega2[i] for i in range(9))
    )
    return anchor.compose(RigidTransform(matrix, matvec(velocity, translation)))


def right_jacobian_se3(increment: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    """Compute ``J_r(xi)``: ``exp((xi+d)^) ≈ exp(xi^) @ exp((J_r(xi)d)^)``.

    ``increment`` is ``xi``, shape ``(6,)``, in ``(tx,ty,tz,rx,ry,rz)`` order,
    with translation in mm and rotation in radians. The returned tuple has
    shape ``(6,6)``: rows index local-right coordinates at ``exp(xi^)``;
    columns index additive fixed-anchor chart coordinates. The upper-right
    block maps rotational changes in radians to translational changes in mm.
    The relation holds to first order in ``d`` and includes translation/rotation
    coupling from the analytic derivative of the SE(3) exponential.

    A translated chart couples a z rotation into local y translation.

    """
    transform = compose_pose(RigidTransform(), increment)
    rho: Vector3 = (increment[0], increment[1], increment[2])
    rotation: Vector3 = (increment[3], increment[4], increment[5])
    omega = _skew(rotation)
    omega2 = matmul(omega, omega)
    _, b, c, db, dc = _coefficients(dot(rotation, rotation))
    rt = transpose(transform.rotation)
    velocity: Matrix3 = cast(
        Matrix3, tuple(IDENTITY[i] + b * omega[i] + c * omega2[i] for i in range(9))
    )
    translation_block = matmul(rt, velocity)
    rotation_block = tuple(IDENTITY[i] - b * omega[i] + c * omega2[i] for i in range(9))
    coupling: list[Vector3] = []
    for axis in range(3):
        basis: Vector3 = cast(Vector3, tuple(float(j == axis) for j in range(3)))
        derivative = _skew(basis)
        left, right = matmul(derivative, omega), matmul(omega, derivative)
        dv: Matrix3 = cast(
            Matrix3,
            tuple(
                db * rotation[axis] * omega[i]
                + b * derivative[i]
                + dc * rotation[axis] * omega2[i]
                + c * (left[i] + right[i])
                for i in range(9)
            ),
        )
        coupling.append(matvec(rt, matvec(dv, rho)))
    return tuple(
        tuple(
            translation_block[3 * i + j]
            if i < 3 and j < 3
            else coupling[j - 3][i]
            if i < 3
            else rotation_block[3 * (i - 3) + (j - 3)]
            if j >= 3
            else 0.0
            for j in range(6)
        )
        for i in range(6)
    )


def chart_gradient(
    increment: tuple[float, ...], right_gradient: tuple[float, ...]
) -> tuple[float, ...]:
    """Compute ``g_chart = J_r(xi).T @ g_right`` at fixed-anchor coordinates ``xi``.

    Both arguments and the result have shape ``(6,)`` and order
    ``(tx,ty,tz,rx,ry,rz)``. ``increment`` contains translation in mm and
    rotation in radians. ``right_gradient`` differentiates the objective under
    a local-right perturbation of the current pose; the result differentiates
    additive coordinates in ``anchor @ exp(xi^)`` with the anchor held fixed.
    Their translation entries are derivatives per mm and rotation entries
    derivatives per radian. The Jacobian transpose carries the cotangent
    between these coordinate systems, including translation/rotation coupling.

    A local y derivative acquires a chart rotation contribution after translation.

    """
    right_gradient = finite_tuple(right_gradient, "right tangent gradient", minimum=None, length=6)
    jacobian = right_jacobian_se3(increment)
    return tuple(sum(jacobian[i][j] * right_gradient[i] for i in range(6)) for j in range(6))
