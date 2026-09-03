"""Device multiview projection and exact first-order acquisition derivatives.

Images have shape (views,rows,columns). Object poses are FP64 (views,12),
with row-major R_WO followed by translation in millimetres. Pose cotangents
use one local right SE(3) increment per view, ordered tx,ty,tz,rx,ry,rz.
Cone intrinsic cotangents are SOD,SDD,column offset,row offset in millimetres.
"""

# ruff: noqa: N803 -- optical depth retains the established L symbol.
# Module-owned scratch is shared with the canonical projection implementation.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._runtime import load_kernels, prepare_context, require_no_tape
from dpt.contracts import ContractError, finite_tuple, integer
from dpt.geometry import BatchedGeometry, ConeBeamIntrinsics, RigidTransform
from dpt.geometry_device import DeviceRayBundleGeometry, prepare_geometry
from dpt.projection import (
    ProjectionSpec,
    ProjectionWorkspace,
    _vjp_kernel,
    prepare_projection,
    project_optical_depth,
    projection_pose_sensitivities,
    projection_vjp,
)
from dpt.volumes import GridSpec

if TYPE_CHECKING:
    import warp as wp


def projection_ray_vjp(
    mu: wp.array[wp.float32] | wp.array[wp.float16] | wp.array[wp.bfloat16],
    pose: wp.array[wp.float64],
    *,
    cotangent: wp.array[wp.float32] | wp.array[wp.float64] | None,
    workspace: ProjectionWorkspace,
    out_ray_coordinates: wp.array[wp.float64],
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Out_ray_coordinates[p,:] = cotangent[p]*dL[p]/d(source_xyz,endpoint_xyz).

    Mu is native field storage (voxels,) in mm^-1 and pose is FP64 (12,)
    object-to-world rotation followed by translation in mm. Cotangent is
    (pixels,) in depth precision. The FP64
    destination is (pixels,6), ordered source x/y/z then endpoint x/y/z, with
    units of depth-covector per mm. Object pose, field and grid are fixed and
    the acquisition coordinates are differentiated. Strict midpoint/cell-Gauss
    sampling includes clipping, sample motion and finite segment norm, including
    native half field storage. Rows already include their weights and have no
    reduction. Checked calls validate inputs and completion diagnostics.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_mu: Any = cast(Any, mu)
    _api_pose: Any = cast(Any, pose)
    _api_cotangent: Any = cast(Any, cotangent)
    _api_out_ray_coordinates: Any = cast(Any, out_ray_coordinates)
    require_no_tape()
    if type(validate) is not bool:
        raise ContractError("validate must be a boolean")
    if workspace.spec.sampling_precision not in ("float32", "float64"):
        raise ContractError("ray-coordinate derivatives require strict float64 sampling")
    if workspace._recorded_tape is not None:
        raise ContractError("the outstanding tape owns the projection inputs")
    ctx, wp = workspace.context, workspace.context.wp
    ctx.assert_stream(stream)
    pixels = workspace.geometry.pixels
    ctx.array(_api_mu, "mu", dtype=workspace.field_dtype, shape=(workspace.grid.voxels,))
    ctx.array(_api_pose, "pose", dtype=wp.float64, shape=(12,))
    ctx.array(_api_cotangent, "cotangent", dtype=workspace.dtype, shape=(pixels,))
    ctx.array(
        _api_out_ray_coordinates, "out_ray_coordinates", dtype=wp.float64, ndim=2, shape=(pixels, 6)
    )
    reads = [("mu", _api_mu), ("pose", _api_pose), ("cotangent", _api_cotangent)]
    if isinstance(workspace.geometry, DeviceRayBundleGeometry):
        reads.extend(
            [("sources", workspace.geometry.sources_mm), ("ends", workspace.geometry.endpoints_mm)]
        )
    ctx.disjoint(reads, [("out_ray_coordinates", _api_out_ray_coordinates)])
    if validate:
        workspace.validate_inputs(_api_mu, _api_pose, cotangent=_api_cotangent)
    wp.launch_tiled(
        _vjp_kernel(
            workspace,
            False,
            False,
            True,
            per_ray=True,
            ray_coordinates=True,
            double_seed=workspace.spec.precision == "float64",
            cell_gauss=workspace.spec.integration == "cell_gauss",
            storage_precision=workspace.spec.storage_precision,
        ),
        dim=(pixels + 127) // 128 if ctx.device.is_cuda else pixels,
        inputs=[_api_mu, _api_pose, workspace._configuration, _api_cotangent],
        outputs=[workspace._empty, _api_out_ray_coordinates.flatten(), workspace._status],
        device=workspace.device,
        stream=workspace.stream,
        block_dim=128,
        record_tape=False,
    )
    if validate:
        workspace.check_status()


@dataclass(frozen=True, slots=True)
class BatchedProjectionWorkspace:
    """Own compact acquisition metadata, mutable device rays and VJP scratch.

    Reuse for sequential forward/VJP calls. Original field and pose arrays
    remain unchanged through their call; each call refreshes rays from the
    current poses. The workspace's buffers and stream belong to one caller.
    """

    grid: GridSpec
    geometry: BatchedGeometry
    spec: ProjectionSpec
    active_intrinsics: bool
    _world: DeviceRayBundleGeometry = field(repr=False)
    _object: DeviceRayBundleGeometry = field(repr=False)
    _projection: ProjectionWorkspace = field(repr=False)
    _identity: Any = field(repr=False)
    _sensitivities: Any = field(repr=False)
    _jacobians: Any = field(repr=False)
    _partials: tuple[Any, ...] = field(repr=False)
    _kernels: Any = field(repr=False)

    @property
    def device(self) -> wp.Device:
        return self._projection.device

    @property
    def stream(self) -> wp.Stream | None:
        return self._projection.stream

    @property
    def dtype(self) -> Any:
        return self._projection.dtype

    @property
    def scratch_bytes(self) -> int:
        return (
            self._world.scratch_bytes
            + self._object.scratch_bytes
            + self._projection.scratch_bytes
            + 96
            + int(self._sensitivities.size) * 8
            + int(self._jacobians.size) * 8
            + sum(int(array.size) * 8 for array in self._partials)
        )

    def check_status(self) -> None:
        self._projection.check_status()

    def clear_status(self) -> None:
        self._projection.clear_status()

    def _refresh(
        self, mu: Any, poses: Any, seed: Any, stream: wp.Stream | None, validate: bool
    ) -> None:
        require_no_tape()
        if type(validate) is not bool:
            raise ContractError("validate must be a boolean")
        self._projection.context.assert_stream(stream)
        if validate:
            self.check_status()
        self._world.in_object_frame(poses, out=self._object, validate=validate, stream=stream)
        if validate:
            self._projection.validate_inputs(mu, self._identity, cotangent=seed)

    def _reduce(self, destination: Any, poses: Any, *, intrinsic: bool, accumulate: bool) -> None:
        wp = self._projection.context.wp
        components = 4 if intrinsic else 6
        source = self._sensitivities.flatten()
        count = self.geometry.views[0].pixels
        views = self.geometry.shape[0]
        for level, following in enumerate(self._partials):
            blocks = (count + 127) // 128
            wp.launch_tiled(
                self._kernels.reduce_views(components, intrinsic=intrinsic and level == 0),
                dim=views * blocks,
                inputs=[source, count, blocks, poses, self._jacobians],
                outputs=[following],
                device=self.device,
                stream=self.stream,
                block_dim=128,
                record_tape=False,
            )
            source, count = following, blocks
            if count == 1:
                break
        wp.launch(
            self._kernels.finish_views(components, accumulate),
            dim=(views, components),
            inputs=[source],
            outputs=[destination, self._projection._status],
            device=self.device,
            stream=self.stream,
            record_tape=False,
        )


def _prepare_batched_projection(
    grid: GridSpec,
    geometry: BatchedGeometry,
    spec: ProjectionSpec,
    *,
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
    max_bytes: int = 256 * 1024 * 1024,
    _intrinsic_jacobians: tuple[tuple[tuple[float, ...], ...], ...] | None = None,
) -> BatchedProjectionWorkspace:
    """Prepare flat, parallel or curved device rays without host pixel lists.

    Forward output is (V,rows,columns); one rigid FP64 (V,12) pose belongs to
    each view. Active pose VJPs use strict sampling and a fixed per-view tree.
    max_bytes bounds resident geometry and derivative scratch before allocation.
    Use prepare_cone_projection to activate intrinsic derivatives.
    """
    if not isinstance(spec, ProjectionSpec):
        raise ContractError("spec must be a ProjectionSpec")
    require_no_tape()
    if not isinstance(cast(object, geometry), BatchedGeometry):
        raise ContractError("geometry must be BatchedGeometry")
    integer(max_bytes, "max_bytes", minimum=1, maximum=2**63 - 1)
    active_intrinsics = _intrinsic_jacobians is not None
    if active_intrinsics and spec.sampling_precision not in ("float32", "float64"):
        raise ContractError("active intrinsics require strict float64 sampling")
    # Rays require 48 bytes in each frame. Sensitivities retain six FP64
    # coordinates per pixel; partial levels retain at most six per tile.
    pixels, views = geometry.pixels, geometry.shape[0]
    count = geometry.views[0].pixels
    partial_counts: list[int] = []
    if spec.active_pose or active_intrinsics:
        while True:
            count = (count + 127) // 128
            partial_counts.append(views * count * 6)
            if count == 1:
                break
        if partial_counts[0] > 2**31 - 1:
            raise ContractError("batched reduction exceeds signed 32-bit tile indexing")
    context = prepare_context(device=device, stream=stream)
    reduction_bytes = 0
    if spec.active_pose:
        reduction_count = max(1, (pixels + 127) // 128 if context.device.is_cuda else pixels)
        while True:
            reduction_bytes += 96 * reduction_count
            if reduction_count == 1:
                break
            reduction_count = (reduction_count + 127) // 128
    upper = (
        96 * pixels
        + views * 148
        + 108
        + (48 * pixels if partial_counts else 0)
        + (192 * views if active_intrinsics else 0)
        + 8 * sum(partial_counts)
        + reduction_bytes
    )
    if upper > max_bytes:
        raise ContractError("batched geometry and derivative scratch exceed max_bytes")
    world = prepare_geometry(geometry, device=device, stream=stream)
    ctx, wp = world.context, world.context.wp
    with ctx.scope():
        identity = wp.array(RigidTransform().packed(), dtype=wp.float64, device=ctx.device)
        poses = wp.array(
            (RigidTransform().packed(),) * views, dtype=wp.float64, ndim=2, device=ctx.device
        )
        object_rays = world.in_object_frame(poses, validate=False)
        projection = prepare_projection(
            grid,
            object_rays,
            spec,
            device=device,
            stream=stream,
        )
        sensitivities = wp.empty(
            (pixels if partial_counts else 0, 6), dtype=wp.float64, device=ctx.device
        )
        jacobians = (
            wp.array(_intrinsic_jacobians, dtype=wp.float64, ndim=3, device=ctx.device)
            if active_intrinsics
            else wp.empty((0, 4, 6), dtype=wp.float64, device=ctx.device)
        )
        partials = tuple(
            wp.empty(size, dtype=wp.float64, device=ctx.device) for size in partial_counts
        )
    result = BatchedProjectionWorkspace(
        grid,
        geometry,
        spec,
        active_intrinsics,
        world,
        object_rays,
        projection,
        identity,
        sensitivities,
        jacobians,
        partials,
        load_kernels("dpt.kernels.batched_projection"),
    )
    if result.scratch_bytes > max_bytes:
        raise ContractError("batched geometry and derivative scratch exceed max_bytes")
    return result


def prepare_batched_projection(
    grid: GridSpec,
    geometry: BatchedGeometry,
    spec: ProjectionSpec,
    *,
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
    max_bytes: int = 256 * 1024 * 1024,
) -> BatchedProjectionWorkspace:
    """Prepare cone, fan, parallel or curved device views from O(V) metadata.

    Images are (V,rows,columns); rigid object poses are FP64 (V,12).
    Active pose VJPs use strict sampling and a fixed FP64 per-view tree.
    max_bytes bounds resident geometry and derivative scratch before allocation.
    Use prepare_cone_projection for active cone calibration derivatives.
    """
    return _prepare_batched_projection(
        grid,
        geometry,
        spec,
        device=device,
        stream=stream,
        max_bytes=max_bytes,
    )


def prepare_cone_projection(
    grid: GridSpec,
    intrinsics: ConeBeamIntrinsics | tuple[ConeBeamIntrinsics, ...],
    angles_rad: tuple[float, ...],
    spec: ProjectionSpec,
    *,
    active_intrinsics: bool = False,
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
    max_bytes: int = 256 * 1024 * 1024,
) -> BatchedProjectionWorkspace:
    """Prepare circular cone views with shared or independent per-view calibration.

    Intrinsics may be one shared record or one record per angle. Each active
    intrinsic VJP row is (SOD,SDD,column offset,row offset). Sum these rows to
    differentiate a shared calibration. Shapes/spacings and view angles are fixed.
    """
    if type(active_intrinsics) is not bool:
        raise ContractError("active_intrinsics must be a boolean")
    angles = finite_tuple(angles_rad, "angles_rad", minimum=None)
    records = (
        (intrinsics,) * len(angles) if isinstance(intrinsics, ConeBeamIntrinsics) else intrinsics
    )
    if (
        not isinstance(cast(object, records), tuple)
        or len(records) != len(angles)
        or any(not isinstance(cast(object, item), ConeBeamIntrinsics) for item in records)
    ):
        raise ContractError("intrinsics must be one cone record or one per angle")
    panels = BatchedGeometry(
        tuple(item.view(angle) for item, angle in zip(records, angles, strict=True))
    )
    jacobians = (
        tuple(
            tuple((*source, *endpoint) for source, endpoint in item.ray_derivatives(angle))
            for item, angle in zip(records, angles, strict=True)
        )
        if active_intrinsics
        else None
    )
    return _prepare_batched_projection(
        grid,
        panels,
        spec,
        device=device,
        stream=stream,
        max_bytes=max_bytes,
        _intrinsic_jacobians=jacobians,
    )


def project_batched_optical_depth(
    mu: wp.array[wp.float32] | wp.array[wp.float16] | wp.array[wp.bfloat16],
    poses: wp.array[wp.float64],
    *,
    workspace: BatchedProjectionWorkspace,
    out_L: wp.array[wp.float32] | wp.array[wp.float64],
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """L[v,r,c] = integral mu(R[v]^T*(ray_world(s)-t[v])) ds.

    Mu is a shared native (voxels,) attenuation field in mm^-1, flattened from
    (nz,ny,nx) with x fastest. Poses is FP64 (V,12), each row containing an
    object-to-world rotation then translation in mm. Out_L is (V,rows,columns)
    in the declared depth precision and has dimensionless optical depth.
    Current poses regenerate object-frame rays on every call. The prepared
    quadrature and sampling policy determine local arithmetic, while ray setup
    and accumulation retain FP64. A forward can later be differentiated using
    these same unchanged field/poses and the declared active flags. Outputs
    are disjoint caller-owned storage on the owning device/stream.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_mu: Any = cast(Any, mu)
    _api_poses: Any = cast(Any, poses)
    _api_out_l: Any = cast(Any, out_L)
    ctx = workspace._projection.context
    ctx.array(_api_out_l, "out_L", dtype=workspace.dtype, ndim=3, shape=workspace.geometry.shape)
    ctx.disjoint([("mu", _api_mu), ("poses", _api_poses)], [("out_L", _api_out_l)])
    workspace._refresh(_api_mu, _api_poses, None, stream, validate)
    project_optical_depth(
        _api_mu,
        workspace._identity,
        workspace=workspace._projection,
        out_L=_api_out_l.flatten(),
        stream=stream,
        validate=False,
    )
    if validate:
        workspace.check_status()


def batched_projection_vjp(
    mu: wp.array[wp.float32] | wp.array[wp.float16] | wp.array[wp.bfloat16],
    poses: wp.array[wp.float64],
    *,
    cotangent: wp.array[wp.float32] | wp.array[wp.float64] | None,
    workspace: BatchedProjectionWorkspace,
    out_pose_tangent: wp.array[wp.float64] | None = None,
    out_intrinsics: wp.array[wp.float64] | None = None,
    out_mu: wp.array[wp.float32] | wp.array[wp.float64] | None = None,
    stream: wp.Stream | None = None,
    validate: bool = True,
    accumulate: bool = False,
    volume_gradient_precision: Literal["float32", "float64"] = "float32",
) -> None:
    """Grad = J(mu,poses,intrinsics)^T*cotangent combines per-view projection products.

    Mu is native (voxels,) attenuation storage in mm^-1 and poses is FP64
    (V,12), ordered row-major object-to-world rotation then translation in mm.
    Cotangent is (V,rows,columns) in depth precision. Out_pose_tangent is FP64
    (V,6), ordered tx,ty,tz,rx,ry,rz in mm/rad for a local-right increment per
    view. Out_intrinsics is FP64 (V,4), ordered SOD,SDD,column offset,row offset
    in mm. Sum those rows when calibration is shared across views. Out_mu is
    (voxels,) in volume_gradient_precision, FP32/FP64, summing all views using
    scatter atomics. Request only parameters active at preparation and choose
    accumulate to add existing destinations.
    The VJP regenerates current rays and includes the strict clipping/sampling
    branches, independently of a preceding forward's mutable scratch. Acquisition
    angles, shape, spacing and grid are fixed.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_mu: Any = cast(Any, mu)
    _api_poses: Any = cast(Any, poses)
    _api_cotangent: Any = cast(Any, cotangent)
    _api_out_pose_tangent: Any = cast(Any, out_pose_tangent)
    _api_out_intrinsics: Any = cast(Any, out_intrinsics)
    _api_out_mu: Any = cast(Any, out_mu)
    require_no_tape()
    if type(accumulate) is not bool:
        raise ContractError("accumulate must be a boolean")
    ctx, wp = workspace._projection.context, workspace._projection.context.wp
    ctx.array(
        _api_cotangent, "cotangent", dtype=workspace.dtype, ndim=3, shape=workspace.geometry.shape
    )
    writes: list[tuple[str, Any]] = []
    views = workspace.geometry.shape[0]
    if _api_out_pose_tangent is not None:
        if not workspace.spec.active_pose:
            raise ContractError("per-view object pose is fixed in this workspace")
        ctx.array(
            _api_out_pose_tangent, "out_pose_tangent", dtype=wp.float64, ndim=2, shape=(views, 6)
        )
        writes.append(("out_pose_tangent", _api_out_pose_tangent))
    if _api_out_intrinsics is not None:
        if not workspace.active_intrinsics:
            raise ContractError("cone intrinsics are fixed in this workspace")
        ctx.array(_api_out_intrinsics, "out_intrinsics", dtype=wp.float64, ndim=2, shape=(views, 4))
        writes.append(("out_intrinsics", _api_out_intrinsics))
    if _api_out_mu is not None:
        if not workspace.spec.active_volume:
            raise ContractError("volume is fixed in this workspace")
        if volume_gradient_precision not in ("float32", "float64"):
            raise ContractError("volume_gradient_precision must be float32 or float64")
        ctx.array(
            _api_out_mu,
            "out_mu",
            dtype=getattr(wp, volume_gradient_precision),
            shape=(workspace.grid.voxels,),
        )
        writes.append(("out_mu", _api_out_mu))
    if not writes:
        raise ContractError("request at least one batched projection gradient")
    ctx.disjoint([("mu", _api_mu), ("poses", _api_poses), ("cotangent", _api_cotangent)], writes)
    workspace._refresh(_api_mu, _api_poses, _api_cotangent.flatten(), stream, validate)
    if _api_out_pose_tangent is not None:
        projection_pose_sensitivities(
            _api_mu,
            workspace._identity,
            cotangent=_api_cotangent.flatten(),
            workspace=workspace._projection,
            out_jacobian=workspace._sensitivities.flatten(),
            validate=False,
        )
        workspace._reduce(_api_out_pose_tangent, _api_poses, intrinsic=False, accumulate=accumulate)
    if _api_out_intrinsics is not None:
        projection_ray_vjp(
            _api_mu,
            workspace._identity,
            cotangent=_api_cotangent.flatten(),
            workspace=workspace._projection,
            out_ray_coordinates=workspace._sensitivities,
            validate=False,
        )
        workspace._reduce(_api_out_intrinsics, _api_poses, intrinsic=True, accumulate=accumulate)
    if _api_out_mu is not None:
        projection_vjp(
            _api_mu,
            workspace._identity,
            cotangent=_api_cotangent.flatten(),
            workspace=workspace._projection,
            out_mu=_api_out_mu,
            validate=False,
            accumulate=accumulate,
            volume_gradient_precision=volume_gradient_precision,
        )
    if validate:
        workspace.check_status()
