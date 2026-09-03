"""Metric flat-panel recovery from homogeneous world-mm projection matrices."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from dpt.contracts import ContractError, finite_scalar, finite_tuple
from dpt.geometry import DetectorGeometry, Vector3, dot, vector3

type ProjectiveRows = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


@dataclass(frozen=True, slots=True)
class ProjectionMatrix:
    """Map world millimetres to homogeneous column,row,depth coordinates.

    Pixel indices refer to pixel centres. A matrix determines ray lines;
    to_detector also receives the physical perpendicular panel distance and
    a point in the positive-depth hemisphere to determine finite segments.
    """

    rows: ProjectiveRows

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.rows), tuple) or len(self.rows) != 3:
            raise ContractError("projection matrix must have three immutable rows")
        rows = tuple(finite_tuple(row, "matrix row", minimum=None, length=4) for row in self.rows)
        object.__setattr__(self, "rows", cast(ProjectiveRows, rows))

    @classmethod
    def from_detector(cls, geometry: DetectorGeometry) -> ProjectionMatrix:
        """Export a metric detector with positive depth towards its pixels."""
        rows = geometry.projection_matrix()
        reference = (*geometry.origin_mm, 1.0)
        depth = sum(a * b for a, b in zip(rows[2], reference, strict=True))
        sign = -1.0 if depth < 0.0 else 1.0
        return cls(
            cast(ProjectiveRows, tuple(tuple(sign * entry for entry in row) for row in rows))
        )

    def project(self, point_mm: Vector3) -> tuple[float, float]:
        """Return continuous (row,column), requiring finite positive depth."""
        point = (*vector3(point_mm, "point_mm"), 1.0)
        coordinates = tuple(
            sum(a * b for a, b in zip(row, point, strict=True)) for row in self.rows
        )
        if not all(math.isfinite(value) for value in coordinates) or coordinates[2] <= 0.0:
            raise ContractError("point must have finite positive projective depth")
        row, column = coordinates[1] / coordinates[2], coordinates[0] / coordinates[2]
        if not math.isfinite(row) or not math.isfinite(column):
            raise ContractError("projected pixel coordinates exceed the finite range")
        return row, column

    def to_detector(
        self,
        *,
        shape: tuple[int, int],
        detector_distance_mm: float,
        positive_depth_point_mm: Vector3,
    ) -> DetectorGeometry:
        """Recover orthogonal detector axes, spacing, source and pixel (0,0).

        detector_distance_mm is the positive perpendicular source-to-panel
        distance. positive_depth_point_mm lies in the intended source-to-panel
        hemisphere, so homogeneous matrix sign cannot reverse finite rays.
        Skewed, affine, degenerate and nonrepresentable geometries are rejected.
        For an oblique source, perpendicular distance differs from the
        source-to-detector-centre segment length.
        """
        distance = finite_scalar(detector_distance_mm, "detector_distance_mm", minimum=0.0)
        if distance == 0.0:
            raise ContractError("detector_distance_mm must be positive")
        point = (*vector3(positive_depth_point_mm, "positive_depth_point_mm"), 1.0)
        scale = math.hypot(*self.rows[2][:3])
        if scale == 0.0 or not math.isfinite(scale):
            raise ContractError("projection matrix requires a finite perspective depth axis")
        depth = sum(a * b for a, b in zip(self.rows[2], point, strict=True)) / scale
        if not math.isfinite(depth) or depth == 0.0:
            raise ContractError("positive depth point must lie outside the source plane")
        sign = -1.0 if depth < 0.0 else 1.0
        rows = tuple(tuple(sign * entry / scale for entry in row) for row in self.rows)
        if any(not math.isfinite(entry) for row in rows for entry in row):
            raise ContractError("normalised projection matrix exceeds the finite range")
        normal = cast(Vector3, rows[2][:3])
        axes: list[Vector3] = []
        offsets: list[float] = []
        focal: list[float] = []
        for row in rows[:2]:
            parallel = dot(cast(Vector3, row[:3]), normal)
            perpendicular = tuple(row[axis] - parallel * normal[axis] for axis in range(3))
            length = math.hypot(*perpendicular)
            if length == 0.0 or not math.isfinite(length):
                raise ContractError("projection matrix has a degenerate pixel axis")
            axes.append(cast(Vector3, tuple(value / length for value in perpendicular)))
            offsets.append(parallel)
            focal.append(length)
        u, v = axes
        if abs(dot(u, v)) > 1e-10:
            raise ContractError("projection matrix must describe orthogonal detector pixel axes")
        source_coordinates = (
            (-rows[0][3] + offsets[0] * rows[2][3]) / focal[0],
            (-rows[1][3] + offsets[1] * rows[2][3]) / focal[1],
            -rows[2][3],
        )
        source = cast(
            Vector3,
            tuple(
                u[axis] * source_coordinates[0]
                + v[axis] * source_coordinates[1]
                + normal[axis] * source_coordinates[2]
                for axis in range(3)
            ),
        )
        spacing = (distance / focal[0], distance / focal[1])
        origin = cast(
            Vector3,
            tuple(
                source[axis]
                + distance * normal[axis]
                - spacing[0] * offsets[0] * u[axis]
                - spacing[1] * offsets[1] * v[axis]
                for axis in range(3)
            ),
        )
        geometry = DetectorGeometry(source, origin, u, v, spacing, shape)
        for row in (0, shape[0] - 1):
            for column in (0, shape[1] - 1):
                endpoint = geometry.pixel_centre(row, column)
                if not all(math.isfinite(value) for value in endpoint) or not math.isfinite(
                    math.dist(source, endpoint)
                ):
                    raise ContractError("metric detector extent exceeds the finite segment range")
        return geometry
