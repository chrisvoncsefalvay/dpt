"""Analytic ellipsoids and sampled volumes for reproducible transport checks."""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from dpt.contracts import ContractError, finite_scalar
from dpt.geometry import (
    IDENTITY,
    DetectorGeometry,
    Matrix3,
    Vector3,
    matvec,
    rotation_matrix,
    transpose,
    vector3,
)
from dpt.volumes import GridSpec


@dataclass(frozen=True, slots=True)
class Ellipsoid:
    """``mu(x) = coefficient`` where ``sum((R.T @ (x-centre)/axes)**2) <= 1``.

    ``axes_mm`` contains three positive semi-axes, ``centre_mm`` contains three
    object-coordinate lengths, and the nine row-major ``orientation`` entries
    map local ellipsoid axes into object coordinates. Attenuation is mm^-1.

    Example:
        >>> sphere = Ellipsoid((1.0, 1.0, 1.0), 0.25)
        >>> sphere.optical_depth((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        0.5

    """

    axes_mm: Vector3
    attenuation_mm_inverse: float
    centre_mm: Vector3 = (0.0, 0.0, 0.0)
    orientation: Matrix3 = IDENTITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes_mm", vector3(self.axes_mm, "axes_mm"))
        if min(self.axes_mm) <= 0:
            raise ContractError("ellipsoid axes must be positive")
        object.__setattr__(self, "centre_mm", vector3(self.centre_mm, "centre_mm"))
        object.__setattr__(self, "orientation", rotation_matrix(self.orientation))
        object.__setattr__(
            self,
            "attenuation_mm_inverse",
            finite_scalar(self.attenuation_mm_inverse, "attenuation_mm_inverse", minimum=0),
        )

    def optical_depth(self, source_mm: Vector3, target_mm: Vector3) -> float:
        """Return ``L = coefficient * length(segment intersect ellipsoid)``.

        ``source_mm`` and ``target_mm`` are three object-coordinate lengths
        in mm. The returned optical depth is dimensionless and clips the
        analytic chord to this finite segment. Misses and tangencies return
        zero, as does a zero-length segment.

        Example:
            >>> model = Ellipsoid((2.0, 1.0, 1.0), 0.25)
            >>> model.optical_depth((-3.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            0.5

        """
        source, target = vector3(source_mm, "source_mm"), vector3(target_mm, "target_mm")
        displacement = cast(Vector3, tuple(b - a for a, b in zip(source, target, strict=True)))
        local = matvec(
            transpose(self.orientation),
            cast(Vector3, tuple(a - b for a, b in zip(source, self.centre_mm, strict=True))),
        )
        direction = matvec(transpose(self.orientation), displacement)
        origin = tuple(x / axis for x, axis in zip(local, self.axes_mm, strict=True))
        scaled = tuple(x / axis for x, axis in zip(direction, self.axes_mm, strict=True))
        scaled_length = math.hypot(*scaled)
        if scaled_length == 0:
            return 0.0
        unit = tuple(x / scaled_length for x in scaled)
        along = math.fsum(x * y for x, y in zip(origin, unit, strict=True))
        nearest = tuple(x - along * y for x, y in zip(origin, unit, strict=True))
        radius_squared = 1.0 - math.fsum(x * x for x in nearest)
        if not all(math.isfinite(x) for x in (scaled_length, along, radius_squared)):
            raise ContractError("ray exceeds the finite analytic ellipsoid range")
        if radius_squared <= 0:
            return 0.0
        half_chord = math.sqrt(radius_squared)
        # Clip relative to the closest point: remote rays retain a small chord
        # without subtracting two large entry/exit distances.
        lower = max(-half_chord, along)
        upper = min(half_chord, scaled_length + along)
        length = max(0.0, upper - lower) * (math.hypot(*displacement) / scaled_length)
        return finite_scalar(self.attenuation_mm_inverse * length, "analytic optical depth")


def sample_ellipsoids(
    grid: GridSpec, ellipsoids: Iterable[Ellipsoid], *, dtype: str = "float32"
) -> Any:
    """Return ``values[z,y,x] = sum_j mu_j(grid.grid_to_object((x,y,z)))``.

    The C-contiguous NumPy array has ``grid.shape = (nz, ny, nx)`` with x
    fastest and dtype ``float32`` or ``float64``. Coordinates and semi-axes
    are in mm, and attenuation is in mm^-1. Oriented grids and ellipsoids use
    their declared object-frame maps. Overlapping ellipsoids add.

    Sampling uses one z slab at a time, so coordinate scratch grows with the
    in-plane shape. The array contains centre samples, whose interpolated
    field differs from the continuous ellipsoid at its boundary.

    Example:
        >>> grid = GridSpec((3, 3, 3), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0))
        >>> values = sample_ellipsoids(grid, (Ellipsoid((1.0, 1.0, 1.0), 0.25),))
        >>> values.shape, values.dtype.name
        ((3, 3, 3), 'float32')
        >>> int((values > 0).sum())
        7

    """
    np: Any = importlib.import_module("numpy")
    if dtype not in ("float32", "float64"):
        raise ContractError("phantom dtype must be float32 or float64")
    models = tuple(ellipsoids)
    if any(not isinstance(cast(object, model), Ellipsoid) for model in models):
        raise ContractError("supply Ellipsoid objects")
    nz, ny, nx = grid.shape
    result = np.zeros(grid.shape, dtype=dtype)
    x = np.arange(nx, dtype=np.float64)[None, :] * grid.spacing_mm[0]
    y = np.arange(ny, dtype=np.float64)[:, None] * grid.spacing_mm[1]
    for z in range(nz):
        slab = [
            grid.origin_mm[axis]
            + grid.orientation[3 * axis] * x
            + grid.orientation[3 * axis + 1] * y
            + grid.orientation[3 * axis + 2] * z * grid.spacing_mm[2]
            for axis in range(3)
        ]
        if any(not bool(np.isfinite(coordinate).all()) for coordinate in slab):
            raise ContractError("sample coordinates exceed the finite phantom range")
        for model in models:
            inverse = transpose(model.orientation)
            distance = np.zeros((ny, nx), dtype=np.float64)
            for axis in range(3):
                coordinate = (
                    sum(inverse[3 * axis + k] * (slab[k] - model.centre_mm[k]) for k in range(3))
                    / model.axes_mm[axis]
                )
                distance += coordinate * coordinate
            result[z] += (distance <= 1.0) * model.attenuation_mm_inverse
        if not bool(np.isfinite(result[z]).all()):
            raise ContractError("sampled attenuation exceeds the requested storage range")
    return result


def circular_cone_beam(
    angles_radians: Iterable[float],
    *,
    source_radius_mm: float,
    detector_radius_mm: float,
    shape_rows_columns: tuple[int, int],
    spacing_column_row_mm: tuple[float, float],
    centre_mm: Vector3 = (0.0, 0.0, 0.0),
) -> tuple[DetectorGeometry, ...]:
    """Set ``source = centre + source_radius * (cos(angle), sin(angle), 0)``.

    The detector centre is ``centre - detector_radius * normal``. Each angle
    in radians gives one ``DetectorGeometry`` with flat pixel centres, columns
    along ``(-sin(angle), cos(angle), 0)`` and rows along positive z. Both
    radii, the three-entry centre and pixel spacings are in mm. Panel shape
    is ``(rows, columns)`` and spacing is ``(column, row)``.

    At zero angle the source lies on positive x. ``ConeBeamIntrinsics`` uses
    a different zero-angle convention, so retain the selected panel records.

    Example:
        >>> panel, = circular_cone_beam(
        ...     (0.0,), source_radius_mm=10.0, detector_radius_mm=20.0,
        ...     shape_rows_columns=(3, 5), spacing_column_row_mm=(0.5, 0.7),
        ... )
        >>> panel.source_mm
        (10.0, 0.0, 0.0)
        >>> panel.pixel_centre(1, 2)
        (-20.0, 0.0, 0.0)

    """
    source_radius = finite_scalar(source_radius_mm, "source_radius_mm", minimum=0)
    detector_radius = finite_scalar(detector_radius_mm, "detector_radius_mm", minimum=0)
    if source_radius == 0 or detector_radius == 0:
        raise ContractError("source and detector radii must be positive")
    centre = vector3(centre_mm, "centre_mm")
    if len(shape_rows_columns) != 2 or any(
        type(x) is not int or x <= 0 for x in shape_rows_columns
    ):
        raise ContractError("shape_rows_columns needs positive integer rows and columns")
    if len(spacing_column_row_mm) != 2:
        raise ContractError("spacing_column_row_mm needs column and row spacing")
    column_pitch, row_pitch = (
        finite_scalar(x, "pixel spacing", minimum=0) for x in spacing_column_row_mm
    )
    if min(column_pitch, row_pitch) <= 0:
        raise ContractError("pixel spacing must be positive")
    rows, columns = shape_rows_columns
    panels: list[DetectorGeometry] = []
    for raw_angle in angles_radians:
        angle = finite_scalar(raw_angle, "angle_radians")
        normal = (math.cos(angle), math.sin(angle), 0.0)
        u = (-normal[1], normal[0], 0.0)
        v = (0.0, 0.0, 1.0)
        source = cast(Vector3, tuple(centre[k] + source_radius * normal[k] for k in range(3)))
        origin = cast(
            Vector3,
            tuple(
                centre[k]
                - detector_radius * normal[k]
                - 0.5 * (columns - 1) * column_pitch * u[k]
                - 0.5 * (rows - 1) * row_pitch * v[k]
                for k in range(3)
            ),
        )
        panels.append(
            DetectorGeometry(source, origin, u, v, (column_pitch, row_pitch), (rows, columns))
        )
    return tuple(panels)
