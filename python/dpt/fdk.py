"""Circular full-orbit FDK initialisation in inverse millimetres.

The reconstruction applies cone weighting, a one-dimensional ramp filter and
voxel-driven interpolation. It provides an approximate starting volume for
iterative reconstruction. The matched discrete adjoint is
``projection_adjoint_gather``.

Reference: Feldkamp, Davis and Kress, Practical cone-beam algorithm,
JOSA A 1, 612-619 (1984), https://doi.org/10.1364/JOSAA.1.000612.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from dpt.contracts import ContractError, finite_tuple, integer
from dpt.geometry import ConeBeamIntrinsics, RigidTransform
from dpt.volumes import GridSpec

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type ProjectionArray = NDArray[np.float32] | FloatArray


@dataclass(frozen=True, slots=True)
class FDKSpec:
    """A circular cone scan with equally spaced angles covering one full orbit.

    ``geometry`` stores SOD, SDD and detector coordinates in millimetres.
    ``angles_rad`` has one angle per view, ordered counterclockwise without
    repeating the first view. ``window`` selects the Ram-Lak ramp or a Hann
    window on that ramp. For example, 64 views use
    ``tuple(2*pi*view/64 for view in range(64))``.
    """

    geometry: ConeBeamIntrinsics
    angles_rad: tuple[float, ...]
    window: Literal["ram-lak", "hann"] = "ram-lak"

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.geometry), ConeBeamIntrinsics):
            raise ContractError("FDK geometry must be ConeBeamIntrinsics")
        angles = finite_tuple(self.angles_rad, "angles_rad", minimum=None)
        if len(angles) < 3:
            raise ContractError("FDK needs at least three equally spaced full-orbit views")
        step = 2.0 * math.pi / len(angles)
        if any(
            abs((angle - angles[0]) % (2.0 * math.pi) - view * step) > 1e-10
            for view, angle in enumerate(angles)
        ):
            raise ContractError("FDK angles must cover one equally spaced counterclockwise orbit")
        if self.window not in ("ram-lak", "hann"):
            raise ContractError("FDK window must be ram-lak or hann")
        spacing = self.virtual_column_spacing_mm
        if not math.isfinite(spacing) or spacing <= 0.0 or not math.isfinite(0.25 / spacing):
            raise ContractError("FDK virtual detector spacing exceeds the FP64 filter range")
        object.__setattr__(self, "angles_rad", angles)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Sinogram dimensions in view,row,column order."""
        return len(self.angles_rad), *self.geometry.shape

    @property
    def virtual_column_spacing_mm(self) -> float:
        """Detector spacing projected to the plane through the rotation centre."""
        return self.geometry.spacing_mm[0] * (
            self.geometry.source_object_distance_mm / self.geometry.source_detector_distance_mm
        )


@dataclass(frozen=True, slots=True)
class FDKExecution:
    """Bound host reconstruction work and coordinate scratch allocations.

    ``max_work`` counts voxel/view evaluations before launch; each coordinate
    block contains at most ``block_voxels`` entries. The projection sample
    limit bounds the copied FP64 weighted and filtered sinograms.
    """

    max_work: int = 100_000_000
    max_projection_samples: int = 16_777_216
    block_voxels: int = 16_384

    def __post_init__(self) -> None:
        for name in ("max_work", "max_projection_samples", "block_voxels"):
            integer(getattr(self, name), name, minimum=1)


@dataclass(frozen=True, slots=True)
class FDKResult:
    """Attenuation volume[nz,ny,nx] in inverse mm, with its grid and scan."""

    volume: FloatArray
    grid: GridSpec
    spec: FDKSpec


def _filter(sinogram: ProjectionArray, spec: FDKSpec) -> FloatArray:
    import numpy as np

    geometry = spec.geometry
    rows, columns = geometry.shape
    u = (np.arange(columns, dtype=np.float64) - 0.5 * (columns - 1)) * geometry.spacing_mm[
        0
    ] + geometry.detector_offset_mm[0]
    v = (np.arange(rows, dtype=np.float64) - 0.5 * (rows - 1)) * geometry.spacing_mm[
        1
    ] + geometry.detector_offset_mm[1]
    distance = geometry.source_detector_distance_mm
    weight = distance / np.hypot(distance, np.hypot(v[:, None], u[None, :]))
    padded = 1 << (max(2 * columns, 64) - 1).bit_length()
    frequencies = np.fft.rfftfreq(padded)
    # Integrating |frequency| over the Nyquist band gives h[0]=1/(4*d²),
    # h[n]=-1/(pi²*n²*d²) for odd n and zero for other even n. The discrete
    # convolution includes one factor d, so the stored weights have units 1/mm.
    indices = np.arange(padded, dtype=np.int64)
    lags = np.where(indices <= padded // 2, indices, indices - padded)
    impulse = np.zeros(padded, dtype=np.float64)
    impulse[0] = 0.25 / spec.virtual_column_spacing_mm
    odd = (lags % 2) != 0
    impulse[odd] = (
        -1.0
        / (math.pi * math.pi * lags[odd].astype(np.float64) ** 2)
        / spec.virtual_column_spacing_mm
    )
    ramp = np.fft.rfft(impulse).real
    if spec.window == "hann":
        ramp *= 0.5 * (1.0 + np.cos(math.pi * frequencies / frequencies[-1]))
    weighted = np.asarray(sinogram, dtype=np.float64) * weight[None, :, :]
    filtered = np.fft.irfft(
        np.fft.rfft(weighted, n=padded, axis=-1) * ramp[None, None, :], n=padded, axis=-1
    )[..., :columns]
    return np.ascontiguousarray(filtered, dtype=np.float64)


def _sample_panel(panel: FloatArray, row: FloatArray, column: FloatArray) -> FloatArray:
    import numpy as np

    rows, columns = panel.shape
    valid = (row >= 0.0) & (row <= rows - 1) & (column >= 0.0) & (column <= columns - 1)
    bounded_row = np.clip(row, 0.0, rows - 1)
    bounded_column = np.clip(column, 0.0, columns - 1)
    r0 = np.floor(bounded_row).astype(np.int64)
    c0 = np.floor(bounded_column).astype(np.int64)
    r1 = np.minimum(r0 + 1, rows - 1)
    c1 = np.minimum(c0 + 1, columns - 1)
    fr, fc = bounded_row - r0, bounded_column - c0
    first = (1.0 - fc) * panel[r0, c0] + fc * panel[r0, c1]
    second = (1.0 - fc) * panel[r1, c0] + fc * panel[r1, c1]
    return np.where(valid, (1.0 - fr) * first + fr * second, 0.0)


def fdk_initialise(
    sinogram: ProjectionArray,
    grid: GridSpec,
    spec: FDKSpec,
    *,
    pose: RigidTransform | None = None,
    execution: FDKExecution | None = None,
) -> FDKResult:
    """Mu[z,y,x] = integral_beta w_beta(x)*q_beta(x)*d_beta/2 is the circular FDK
    initialiser.

    Sinogram is a finite FP32/FP64 NumPy array (views,rows,columns) containing
    dimensionless optical depths. FDKSpec describes at least three equally spaced
    full-orbit cone views, with SOD/SDD, offsets and spacings in mm. Preweight is
    SDD/sqrt(SDD^2+u^2+v^2), ramp filtering uses virtual spacing du*SOD/SDD and
    cycles/mm, and backprojection uses SOD^2/(SOD+y_view)^2. The returned
    FDKResult.volume is contiguous FP64 (nz,ny,nx) attenuation in mm^-1, with
    x fastest and grid positions at sample centres. Pose maps those sample
    centres into the scan's world frame. NumPy FP64 work uses bounded coordinate
    blocks under FDKExecution. Ringing may produce signed values and the
    iterative solver applies physical constraints. No gradient product is supplied.

    >>> import math, numpy as np
    >>> from dpt.fdk import FDKSpec, fdk_initialise
    >>> from dpt.geometry import ConeBeamIntrinsics
    >>> from dpt.volumes import GridSpec
    >>> scan = FDKSpec(
    ...     ConeBeamIntrinsics(10.0, 20.0, (1, 3), (1.0, 1.0)),
    ...     tuple(2 * math.pi * v / 4 for v in range(4)),
    ... )
    >>> result = fdk_initialise(
    ...     np.zeros(scan.shape, dtype=np.float64),
    ...     GridSpec((1, 1, 1), (1.0, 1.0, 1.0)),
    ...     scan,
    ... )
    >>> result.volume.shape
    (1, 1, 1)
    >>> float(result.volume[0, 0, 0])
    0.0

    """
    import numpy as np

    if not isinstance(cast(object, spec), FDKSpec) or not isinstance(cast(object, grid), GridSpec):
        raise ContractError("FDK needs an FDKSpec and GridSpec")
    selected_pose = RigidTransform() if pose is None else pose
    if not isinstance(cast(object, selected_pose), RigidTransform):
        raise ContractError("FDK pose must be RigidTransform")
    budget = FDKExecution() if execution is None else execution
    if not isinstance(cast(object, budget), FDKExecution):
        raise ContractError("FDK execution must be FDKExecution")
    work = grid.voxels * len(spec.angles_rad)
    samples = math.prod(spec.shape)
    if work > budget.max_work or samples > budget.max_projection_samples:
        raise ContractError(
            f"FDK work {work} and projection samples {samples} exceed execution limits"
        )
    if not isinstance(cast(object, sinogram), np.ndarray):
        raise ContractError("FDK sinogram must be a host NumPy float32 or float64 array")
    if sinogram.shape != spec.shape or sinogram.dtype not in (
        np.dtype(np.float32),
        np.dtype(np.float64),
    ):
        raise ContractError(f"FDK sinogram must be float32 or float64 with shape {spec.shape}")
    if not bool(np.isfinite(sinogram).all()):
        raise ContractError("FDK line integrals must be finite")
    filtered = _filter(sinogram, spec)
    geometry = spec.geometry
    result = np.zeros(grid.voxels, dtype=np.float64)
    rows, columns = geometry.shape
    sod, sdd = geometry.source_object_distance_mm, geometry.source_detector_distance_mm
    angular_half_step = math.pi / len(spec.angles_rad)
    nx, ny = grid.shape[2], grid.shape[1]
    transform = selected_pose.compose(RigidTransform(grid.orientation, grid.origin_mm))
    for begin in range(0, grid.voxels, budget.block_voxels):
        end = min(begin + budget.block_voxels, grid.voxels)
        indices = np.arange(begin, end, dtype=np.int64)
        local = (
            (indices % nx) * grid.spacing_mm[0],
            ((indices // nx) % ny) * grid.spacing_mm[1],
            (indices // (nx * ny)) * grid.spacing_mm[2],
        )
        world = tuple(
            np.asarray(
                sum(transform.rotation[3 * axis + column] * local[column] for column in range(3))
                + transform.translation_mm[axis]
                - geometry.centre_mm[axis],
                dtype=np.float64,
            )
            for axis in range(3)
        )
        if any(not bool(np.isfinite(coordinate).all()) for coordinate in world):
            raise ContractError("FDK grid coordinates exceed finite FP64 range")
        for view, angle in enumerate(spec.angles_rad):
            cosine, sine = math.cos(angle), math.sin(angle)
            x_view = cosine * world[0] + sine * world[1]
            y_view = -sine * world[0] + cosine * world[1]
            depth = np.asarray(sod + y_view, dtype=np.float64)
            if bool((depth <= 0.0).any()):
                raise ContractError("FDK grid must lie in front of every source plane")
            projected_u = sdd * x_view / depth
            projected_v = sdd * world[2] / depth
            column = (projected_u - geometry.detector_offset_mm[0]) / geometry.spacing_mm[
                0
            ] + 0.5 * (columns - 1)
            row = (projected_v - geometry.detector_offset_mm[1]) / geometry.spacing_mm[1] + 0.5 * (
                rows - 1
            )
            if not bool(np.isfinite(row).all()) or not bool(np.isfinite(column).all()):
                raise ContractError("FDK projected coordinates exceed finite FP64 range")
            values = _sample_panel(
                filtered[view],
                np.asarray(row, dtype=np.float64),
                np.asarray(column, dtype=np.float64),
            )
            result[begin:end] += angular_half_step * (sod / depth) ** 2 * values
    if not bool(np.isfinite(result).all()):
        raise ContractError("FDK reconstruction exceeds finite FP64 range")
    return FDKResult(result.reshape(grid.shape), grid, spec)
