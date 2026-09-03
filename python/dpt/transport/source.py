"""Fixed-distribution parallel sources keyed by original history identity."""

# Internal workspace launch plumbing is shared within the transport package.
# pyright: reportPrivateUsage=false
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from dpt._runtime import load_kernels, require_no_tape
from dpt.geometry import DetectorGeometry

from .forward import TransportWorkspace
from .model import OrientedDetector, TransportError, _freeze_fields, _physical, _require_instance
from .rng import HistoryBatch, SourceIdentity
from .sampling import _bind_sampling


@dataclass(frozen=True, slots=True)
class ParallelBeam:
    """Uniform rectangular source in an xy plane, with one fixed unit direction.

    Extent zero in both axes specifies a deterministic pencil. `weight` is the
    expected source score per sampled ray before the separate source amplitude;
    it is not silently multiplied by area, exposure or inverse-square factors.
    The caller sets that physical normalisation. The distribution stays fixed
    throughout recovery, and its randomisation follows the structural batch's
    declared IID or complete-path sampling law.
    """

    lower_mm: tuple[float, float, float]
    extent_xy_mm: tuple[float, float]
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    weight: float = 1.0

    def __post_init__(self) -> None:
        _freeze_fields(self, ("lower_mm", "extent_xy_mm", "direction"))
        for value in (*self.lower_mm, *self.extent_xy_mm, *self.direction):
            _physical(value, "source geometry")
        if len(self.lower_mm) != 3 or len(self.direction) != 3 or len(self.extent_xy_mm) != 2:
            raise TransportError("source vectors have incompatible dimensions")
        if any(value < 0 for value in self.extent_xy_mm):
            raise TransportError("source extents must be nonnegative")
        if abs(math.fsum(value * value for value in self.direction) - 1) > 1e-12:
            raise TransportError("source direction must be a unit vector")
        _physical(self.weight, "source weight", nonnegative=True)
        for lower, extent in zip(self.lower_mm[:2], self.extent_xy_mm, strict=True):
            if not math.isfinite(lower + extent) or (extent > 0 and lower + extent == lower):
                raise TransportError("source extent overflows its coordinate representation")


@dataclass(frozen=True, slots=True)
class PointConeSource:
    """Point emission sampled uniformly over a fixed detector rectangle.

    Each emitted direction aims at a uniform point of the declared pixel area.
    ``weight`` explicitly sets the measurement per emitted history for this
    fixed sampling law. Isotropic radiance and inverse-square factors are not
    inferred. Scattered paths retain the same original-history denominator.
    """

    geometry: DetectorGeometry
    weight: float = 1.0

    def __post_init__(self) -> None:
        _physical(self.weight, "source weight", nonnegative=True)
        _require_instance(self.geometry, DetectorGeometry, "PointConeSource.geometry")
        detector = OrientedDetector(self.geometry)
        for column in (0, detector.shape[0]):
            for row in (0, detector.shape[1]):
                corner = tuple(
                    detector.lower_corner_mm[i]
                    + column * detector.spacing_xy_mm[0] * detector.u[i]
                    + row * detector.spacing_xy_mm[1] * detector.v[i]
                    - self.geometry.source_mm[i]
                    for i in range(3)
                )
                distance = math.hypot(*corner)
                if not math.isfinite(distance) or distance == 0:
                    raise TransportError(
                        "source-detector segments must have finite positive lengths"
                    )


def sample_point_cone(
    source: PointConeSource,
    *,
    batch: HistoryBatch,
    workspace: TransportWorkspace,
    out_position: Any,
    out_direction: Any,
    out_weight: Any,
    stream: Any = None,
) -> SourceIdentity:
    r"""Sample ``direction = (target-source)/norm(target-source)`` from the detector area.

    With independent ``U,V`` uniform on ``(0,1)``, a target point is
    ``y=lower_corner+U*width_mm*u+V*height_mm*v`` and the emitted unit direction
    is ``(y-source_mm)/norm(y-source_mm)``. This is a uniform detector-area law,
    whose probability density per solid angle is ``distance**2/(area*cosine)``.
    The supplied ``source.weight`` explicitly sets the score under this source
    law. No isotropic-emission or inverse-square correction is applied implicitly.

    Each output is a contiguous caller-owned Warp array of shape
    ``(batch.count,)`` on ``workspace.context.device``. Positions have dtype
    ``wp.vec3d`` in world mm, directions ``wp.vec3d`` and unit norm, and weights
    ``wp.float64`` in the caller's measurement normalisation. Outputs must be
    disjoint from one another and prepared input buffers. The point must lie
    on the source side of the workspace detector and the batch fit its capacity.
    ``stream`` must match the prepared stream and buffers remain alive until it
    completes. The call enqueues work without a download or synchronisation.

    A reserved source counter range maps each original history identity to
    its target point. Replaying the same ``HistoryBatch`` reproduces the samples.
    Returned ``SourceIdentity`` identifies this law and batch, with no device
    allocation. The source distribution stays fixed in material-density replay,
    and the sampler has no Warp tape derivative. Source-amplitude derivatives
    act on the later measurement multiplier rather than on the sampled targets.

    """
    require_no_tape()
    sampling = _bind_sampling(workspace, batch)
    source_kernels = (
        workspace._kernels
        if batch.sampling is None
        else load_kernels("dpt.transport.sampling_kernels")
    )
    context = workspace.context
    context.assert_stream(stream)
    wp = context.wp
    geometry = source.geometry
    if (
        batch.count > workspace.max_histories
        or workspace.spec.detector.source_distance(geometry.source_mm) >= 0
    ):
        raise TransportError("source batch capacity or source-detector geometry is invalid")
    for name, value, dtype in (
        ("out_position", out_position, wp.vec3d),
        ("out_direction", out_direction, wp.vec3d),
        ("out_weight", out_weight, wp.float64),
    ):
        context.array(value, name, dtype=dtype, shape=(batch.count,))
    context.disjoint(
        workspace._reads(),
        [
            ("out_position", out_position),
            ("out_direction", out_direction),
            ("out_weight", out_weight),
        ],
    )
    corner = tuple(
        geometry.origin_mm[i]
        - 0.5 * (geometry.spacing_mm[0] * geometry.u[i] + geometry.spacing_mm[1] * geometry.v[i])
        for i in range(3)
    )
    workspace._launch(
        source_kernels.sample_point_source,
        batch.count,
        [
            wp.vec3d(*geometry.source_mm),
            wp.vec3d(*corner),
            wp.vec3d(*geometry.u),
            wp.vec3d(*geometry.v),
            wp.vec2d(
                geometry.spacing_mm[0] * geometry.shape[1],
                geometry.spacing_mm[1] * geometry.shape[0],
            ),
            source.weight,
            wp.uint64(batch.seed),
            wp.uint64(batch.first_history),
            out_position,
            out_direction,
            out_weight,
            *([] if batch.sampling is None else [sampling]),
        ],
    )
    return SourceIdentity(
        batch.seed, batch.first_history, batch.count, "point-cone-area-v1", batch.sampling
    )


def sample_parallel_beam(
    source: ParallelBeam,
    *,
    batch: HistoryBatch,
    workspace: TransportWorkspace,
    out_position: Any,
    out_direction: Any,
    out_weight: Any,
    stream: Any = None,
) -> SourceIdentity:
    r"""Return ``position = lower + (U*extent_x, V*extent_y, 0)`` for uniform ``U,V``.

    Each position is ``(x0+U*extent_x,y0+V*extent_y,z0)`` for independent uniform
    ``U,V``. Its direction is the declared constant unit vector and its weight
    is exactly ``source.weight``. A zero extent gives a deterministic coordinate,
    so two zero extents specify a pencil. Positive extents define uniform area
    sampling, with density ``1/(extent_x*extent_y)`` on that rectangle. The caller
    sets exposure and physical normalisation explicitly through the weight and
    the subsequent source-amplitude multiplier.

    ``out_position`` and ``out_direction`` are contiguous ``wp.vec3d`` arrays,
    while ``out_weight`` is ``wp.float64``, each of shape ``(batch.count,)`` on
    the prepared workspace device. Positions are world mm, directions are
    dimensionless, and weight units follow the chosen detector measurement.
    All outputs must be disjoint from one another and prepared input buffers.
    The rectangle's corners must lie on the source side of the detector and
    the batch must fit workspace capacity. The optional stream must match its
    owning stream and all arrays remain alive until queued work completes.

    No device array is allocated and no download or synchronisation occurs.
    Event bit 31 reserves source draws apart from interaction events, so the
    same ``HistoryBatch`` reproduces each original history's sample independently
    of launch decomposition. Returned ``SourceIdentity`` records the batch and
    source-law identifier. Geometry, extents and sampling law remain fixed in
    material-density replay. The sampler is not differentiable through a Warp
    tape and the later source-amplitude derivative leaves these samples fixed.

    """
    require_no_tape()
    sampling = _bind_sampling(workspace, batch)
    source_kernels = (
        workspace._kernels
        if batch.sampling is None
        else load_kernels("dpt.transport.sampling_kernels")
    )
    context = workspace.context
    context.assert_stream(stream)
    wp = context.wp
    corners = (
        (source.lower_mm[0] + x, source.lower_mm[1] + y, source.lower_mm[2])
        for x in (0.0, source.extent_xy_mm[0])
        for y in (0.0, source.extent_xy_mm[1])
    )
    if batch.count > workspace.max_histories or any(
        workspace.spec.detector.source_distance(corner) >= 0 for corner in corners
    ):
        raise TransportError("source batch capacity or source-detector geometry is invalid")
    for name, value, dtype in (
        ("out_position", out_position, wp.vec3d),
        ("out_direction", out_direction, wp.vec3d),
        ("out_weight", out_weight, wp.float64),
    ):
        context.array(value, name, dtype=dtype, shape=(batch.count,))
    context.disjoint(
        workspace._reads(),
        [
            ("out_position", out_position),
            ("out_direction", out_direction),
            ("out_weight", out_weight),
        ],
    )
    workspace._launch(
        source_kernels.sample_parallel_source,
        batch.count,
        [
            wp.vec3d(*source.lower_mm),
            wp.vec2d(*source.extent_xy_mm),
            wp.vec3d(*source.direction),
            source.weight,
            wp.uint64(batch.seed),
            wp.uint64(batch.first_history),
            out_position,
            out_direction,
            out_weight,
            *([] if batch.sampling is None else [sampling]),
        ],
    )
    return SourceIdentity(
        batch.seed, batch.first_history, batch.count, "parallel-rectangle-v1", batch.sampling
    )
