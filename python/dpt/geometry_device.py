"""Generate ``(views,rows,columns)`` finite rays from compact panel metadata.

Only O(views) coordinates cross the host boundary. Pixel expansion and optional
per-view object-frame transforms run on the owning Warp stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._runtime import (
    DeviceContext,
    StatusReadback,
    load_kernels,
    prepare_context,
    require_no_tape,
)
from dpt.contracts import ContractError, raise_device_status
from dpt.geometry import BatchedGeometry, CurvedDetectorGeometry, ParallelGeometry

if TYPE_CHECKING:
    import warp as wp


@dataclass(frozen=True, slots=True)
class DeviceRayBundleGeometry:
    """Device segments ``s,e`` of shape (V*rows*columns,3), stored as FP64 vec3d.

    Arrays flatten in view,row,column order and belong to ``context``. Retain
    this record while a projection uses its arrays. ``coordinate_frame`` is
    world for prepared panels and object after ``in_object_frame(poses)``.
    For example, two 64 by 64 panels have 8192 finite segments.
    """

    geometry: BatchedGeometry
    context: DeviceContext
    sources_mm: Any = field(repr=False)
    endpoints_mm: Any = field(repr=False)
    coordinate_frame: Literal["world", "object"] = "world"
    _metadata: Any = field(default=None, repr=False)
    _modes: Any = field(default=None, repr=False)
    _status: Any = field(init=False, repr=False)
    _status_readback: StatusReadback = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.context.array(
            self.sources_mm, "sources_mm", dtype=self.context.wp.vec3d, shape=(self.pixels,)
        )
        self.context.array(
            self.endpoints_mm, "endpoints_mm", dtype=self.context.wp.vec3d, shape=(self.pixels,)
        )
        self.context.disjoint(
            [], [("sources_mm", self.sources_mm), ("endpoints_mm", self.endpoints_mm)]
        )
        if self.coordinate_frame not in ("world", "object"):
            raise ContractError("coordinate_frame must be world or object")
        with self.context.scope():
            status = self.context.wp.zeros(1, dtype=self.context.wp.int32, device=self.device)
        object.__setattr__(self, "_status", status)
        object.__setattr__(self, "_status_readback", StatusReadback(self.context, status))

    @property
    def pixels(self) -> int:
        return self.geometry.pixels

    @property
    def shape_rows_columns(self) -> tuple[int, int]:
        return self.geometry.shape_rows_columns

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.geometry.shape

    @property
    def view_offsets(self) -> tuple[int, ...]:
        return self.geometry.view_offsets

    @property
    def device(self) -> wp.Device:
        return self.context.device

    @property
    def stream(self) -> wp.Stream | None:
        return self.context.stream

    @property
    def scratch_bytes(self) -> int:
        metadata = 0 if self._metadata is None else int(self._metadata.size) * 8
        modes = 0 if self._modes is None else int(self._modes.size) * 4
        return self.pixels * 48 + metadata + modes + 4

    def validate_poses(
        self, poses: wp.array[wp.float64], *, stream: wp.Stream | None = None
    ) -> None:
        """Check finite rigid FP64 poses of shape (V,12) before a transform epoch.

        Each rotation satisfies ``R.T @ R = I`` to 1e-10 and has positive
        determinant. This explicit checkpoint reads one status word. Validated
        poses can be reused with ``in_object_frame(..., validate=False)`` until
        the caller modifies them.
        """
        require_no_tape()
        ctx = self.context
        ctx.assert_stream(stream)
        ctx.array(poses, "poses", dtype=ctx.wp.float64, ndim=2, shape=(self.shape[0], 12))
        kernels = load_kernels("dpt.kernels.batched_geometry")
        with ctx.scope():
            self._status.zero_()
            ctx.wp.launch(
                kernels.validate_poses,
                dim=self.shape[0],
                inputs=[poses],
                outputs=[self._status],
                device=self.device,
                stream=self.stream,
                record_tape=False,
            )
        raise_device_status(self._status_readback.read(), "batched rigid pose")

    def in_object_frame(
        self,
        poses: wp.array[wp.float64],
        *,
        out: DeviceRayBundleGeometry | None = None,
        stream: wp.Stream | None = None,
        validate: bool = True,
    ) -> DeviceRayBundleGeometry:
        """s_object=R_WO^T*(s_world-t_WO) and e_object=R_WO^T*(e_world-t_WO).

        Poses is contiguous FP64 (V,12), ordered row-major object-to-world rotation
        then translation in mm. Input world rays and returned object rays are
        vec3d (V*rows*columns,) source/end arrays in mm, flattened in view,row,column
        order. A supplied compatible out record reuses its buffers for repeated
        transforms, while omission allocates object-frame rays. The returned record
        owns its stream/context and is marked coordinate_frame="object". Project
        those rays with an identity shared pose. Checked calls validate finite rigid
        poses first and validate=False reuses an explicit unchanged validate_poses
        epoch. This transform exposes no VJP and differentiating acquisitions uses
        the projector's explicit ray-coordinate product.

        """
        require_no_tape()
        if self.coordinate_frame != "world":
            raise ContractError("per-view object transforms require world-frame rays")
        ctx = self.context
        ctx.assert_stream(stream)
        ctx.array(poses, "poses", dtype=ctx.wp.float64, ndim=2, shape=(self.shape[0], 12))
        if type(validate) is not bool:
            raise ContractError("validate must be a boolean")
        if validate:
            self.validate_poses(poses, stream=stream)
        if out is None:
            with ctx.scope():
                out = DeviceRayBundleGeometry(
                    self.geometry,
                    ctx,
                    ctx.wp.empty(self.pixels, dtype=ctx.wp.vec3d, device=ctx.device),
                    ctx.wp.empty(self.pixels, dtype=ctx.wp.vec3d, device=ctx.device),
                    "object",
                )
        if (
            out.geometry != self.geometry
            or out.coordinate_frame != "object"
            or out.device != self.device
        ):
            raise ContractError(
                "out must retain this batch on the same device in object coordinates"
            )
        out.context.assert_stream(self.stream)
        ctx.disjoint(
            [("sources", self.sources_mm), ("endpoints", self.endpoints_mm), ("poses", poses)],
            [("out_sources", out.sources_mm), ("out_endpoints", out.endpoints_mm)],
        )
        kernels = load_kernels("dpt.kernels.batched_geometry")
        ctx.wp.launch(
            kernels.object_frame,
            dim=self.pixels,
            inputs=[self.sources_mm, self.endpoints_mm, poses, self.geometry.views[0].pixels],
            outputs=[out.sources_mm, out.endpoints_mm],
            device=self.device,
            stream=self.stream,
            record_tape=False,
        )
        return out


def prepare_geometry(
    geometry: BatchedGeometry,
    *,
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
) -> DeviceRayBundleGeometry:
    """Generate finite ``(V,rows,columns)`` device rays from O(V) host metadata.

    Flat cone/fan, parallel and cylindrical panels can share one batch. Each
    view transfers eighteen FP64 values and one mode flag; a single kernel
    expands all pixels. For example, ``prepare_geometry(intrinsics.batch(angles))``
    prepares a multiview cone acquisition without host ray lists.
    """
    require_no_tape()
    if not isinstance(cast(object, geometry), BatchedGeometry):
        raise ContractError("geometry must be BatchedGeometry")
    ctx = prepare_context(device=device, stream=stream)
    kernels = load_kernels("dpt.kernels.batched_geometry")
    metadata: list[tuple[float, ...]] = []
    modes: list[int] = []
    for panel in geometry.views:
        if isinstance(panel, ParallelGeometry):
            source, length, radial, mode = panel.direction, panel.length_mm, (0.0, 0.0, 0.0), 1
        elif isinstance(panel, CurvedDetectorGeometry):
            source, length, radial, mode = panel.source_mm, panel.radius_mm, panel.radial_mm, 2
        else:
            source, length, radial, mode = panel.source_mm, 0.0, (0.0, 0.0, 0.0), 0
        metadata.append(
            (*source, *panel.origin_mm, *panel.u, *panel.v, *panel.spacing_mm, length, *radial)
        )
        modes.append(mode)
    with ctx.scope():
        packed = ctx.wp.array(metadata, dtype=ctx.wp.float64, ndim=2, device=ctx.device)
        flags = ctx.wp.array(modes, dtype=ctx.wp.int32, device=ctx.device)
        sources = ctx.wp.empty(geometry.pixels, dtype=ctx.wp.vec3d, device=ctx.device)
        endpoints = ctx.wp.empty(geometry.pixels, dtype=ctx.wp.vec3d, device=ctx.device)
        ctx.wp.launch(
            kernels.generate,
            dim=geometry.shape,
            inputs=[packed, flags, geometry.shape_rows_columns[0], geometry.shape_rows_columns[1]],
            outputs=[sources, endpoints],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
    return DeviceRayBundleGeometry(geometry, ctx, sources, endpoints, "world", packed, flags)
