"""Fixed-density material-fraction paths using the canonical sampled-field projector.

Fields are material-major, flat FP32 arrays with M*V values; paths have M*P
values in millimetres, using the supplied ProjectionSpec precision for both
paths and their cotangents. Field-gradient destinations use explicitly selected
FP32 or FP64 storage, independently of the FP32 fields.
Each basis field is a volume fraction. Fractions are
non-negative and sum to at most one at every sample; the remainder is vacuum.
The FP64 sum admits 2^-24 excess for independently rounded binary32 fractions.
The explicit nonnegative domain instead accepts dimensionless equivalent-basis
fields without a sum cap. It uses the same integration and derivatives; basis
scales belong in the supplied attenuation coefficients. There is no hidden
normalisation or inferred CT-to-material conversion.
"""

from __future__ import annotations

# Composition intentionally owns the nested projector's diagnostics/configuration.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._runtime import DeviceContext, StatusReadback, load_kernels, require_no_tape
from dpt.contracts import ContractError, integer, raise_device_status
from dpt.geometry import DetectorGeometry, RayBundleGeometry
from dpt.projection import (
    ProjectionSpec,
    ProjectionWorkspace,
    _forward_kernel,
    prepare_projection,
)
from dpt.projection import projection_vjp as _vjp
from dpt.volumes import GridSpec

if TYPE_CHECKING:
    import warp as wp


@dataclass(slots=True)
class MaterialProjectionWorkspace:
    """Prepared views retain their caller-owned parent buffers for the whole solve.

    Serial launches reuse a single projection workspace. The optional fused
    forward shares the ray/traversal across materials and reuses canonical
    interpolation helpers. No per-evaluation views or device arrays are made.
    Fused reverse shares traversal and combines seeded channel slopes for one
    pose reduction; field scatter retains independent material destinations.
    """

    projection: ProjectionWorkspace
    materials: int
    fields: Any
    paths: Any
    cotangent_paths: Any
    grad_fields: Any
    field_views: tuple[Any, ...]
    path_views: tuple[Any, ...]
    seed_views: tuple[Any, ...]
    gradient_views: tuple[Any, ...]
    kernels: Any = field(repr=False)
    status: Any = field(repr=False)
    field_domain: Literal["fractions", "nonnegative"] = "fractions"
    channel_execution: Literal["serial", "fused"] = "serial"
    volume_gradient_precision: Literal["float32", "float64"] = "float32"
    _float32_volume_increments: bool = field(default=False, repr=False)
    _status_readback: StatusReadback = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._status_readback = StatusReadback(self.context, self.status)

    @property
    def context(self) -> DeviceContext:
        return self.projection.context

    @property
    def scratch_bytes(self) -> int:
        return self.projection.scratch_bytes + 4

    @property
    def host_bytes(self) -> int:
        return self.projection.host_bytes + self._status_readback.host_bytes

    def clear_status(self) -> None:
        self.projection.clear_status()
        with self.context.scope():
            self.status.zero_()

    def check_status(self) -> None:
        self.projection.check_status()
        raise_device_status(
            self._status_readback.read(),
            "material projection",
            messages={1: "invalid material path cotangents"},
        )

    def consume_status(self) -> None:
        """Deliver all owned diagnostics before allowing a checked retry."""
        projection_code = self.projection._status_readback.read(reset=True)
        material_code = self._status_readback.read(reset=True)
        self.projection._raise_status(projection_code)
        raise_device_status(
            material_code, "material projection", messages={1: "invalid material path cotangents"}
        )

    def validate_inputs(self, *, seeds: bool = False, stream: wp.Stream | None = None) -> None:
        """Check current bound fractions and optional cotangents without projecting.

        Fraction-domain roundoff permits a total up to 1 + 2^-24. Nonnegative
        fields have no sum cap. Values are never changed. This checks no pose
        and does not certify later field mutations.
        """
        require_no_tape()
        self.context.assert_stream(stream)
        if seeds and self.cotangent_paths is None:
            raise ContractError("material validation requires a bound path cotangent")
        _validate(self, seeds=seeds)


def prepare_material_projection(
    grid: GridSpec,
    geometry: DetectorGeometry | RayBundleGeometry,
    spec: ProjectionSpec,
    *,
    materials: int,
    fields: wp.array[wp.float32],
    out_paths: wp.array[wp.float32] | wp.array[wp.float64],
    cotangent_paths: wp.array[wp.float32] | wp.array[wp.float64] | None = None,
    out_fields: wp.array[wp.float32] | wp.array[wp.float64] | None = None,
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
    field_domain: Literal["fractions", "nonnegative"] = "fractions",
    channel_execution: Literal["serial", "fused"] = "serial",
    volume_gradient_precision: Literal["float32", "float64"] = "float32",
    _float32_volume_increments: bool = False,
) -> MaterialProjectionWorkspace:
    """Bind fields; nonnegative equivalent-basis fields have no fraction sum cap.

    Both domains use dimensionless fields, paths in mm and supplied basis
    attenuation in mm^-1. The caller records any mass-density normalisation.
    ``spec.precision`` selects FP32 or FP64 paths and path cotangents. Fields
    remain FP32 and ``volume_gradient_precision`` separately selects the
    caller-owned FP32 or FP64 gradient destination, with FP32 retained by default.
    ``channel_execution="fused"`` shares traversal across materials in both products;
    the default serial path preserves the existing execution policy.

    ``cotangent_paths`` binds the caller-owned path covector, with M*P entries
    in the declared path precision. Omission or None prepares forward-only paths.
    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_fields: Any = cast(Any, fields)
    _api_out_paths: Any = cast(Any, out_paths)
    _api_cotangent_paths: Any = cast(Any, cotangent_paths)
    _api_out_fields: Any = cast(Any, out_fields)
    if not isinstance(spec, ProjectionSpec):
        raise ContractError("spec must be a ProjectionSpec")
    if volume_gradient_precision not in ("float32", "float64"):
        raise ContractError("volume_gradient_precision must be float32 or float64")
    if type(_float32_volume_increments) is not bool or (
        _float32_volume_increments and volume_gradient_precision != "float64"
    ):
        raise ContractError("FP32 volume increments require an FP64 gradient destination")
    if field_domain not in ("fractions", "nonnegative"):
        raise ContractError("field_domain must be fractions or nonnegative")
    if spec.field_domain != "physical":
        raise ContractError("material projection requires the physical projection field domain")
    if spec.storage_precision != "float32":
        raise ContractError("material projection requires float32 field storage")
    if spec.sampling_precision == "fast_float32":
        raise ContractError(
            "material projection supports float32, float64 or mixed_float32 sampling"
        )
    if channel_execution not in ("serial", "fused"):
        raise ContractError("channel_execution must be serial or fused")
    if channel_execution == "fused" and spec.sampling_precision not in ("float32", "float64"):
        raise ContractError("fused material projection requires fp32 or fp64 sampling")
    integer(materials, "materials", minimum=1, maximum=32)
    integer(materials * grid.voxels, "material field length", minimum=1)
    integer(materials * geometry.pixels, "material path length", minimum=1)
    projection = prepare_projection(grid, geometry, spec, device=device, stream=stream)
    ctx, wp = projection.context, projection.context.wp
    _api_fields = _material_array(
        _api_fields, "fields", ctx, wp.float32, ((materials, grid.voxels), (materials, *grid.shape))
    )
    path_dtype = wp.float64 if spec.precision == "float64" else wp.float32
    _api_out_paths = _material_array(
        _api_out_paths, "out_paths", ctx, path_dtype, ((materials, geometry.pixels),)
    )
    reads = [("fields", _api_fields)]
    writes = [("out_paths", _api_out_paths)]
    if _api_cotangent_paths is not None:
        _api_cotangent_paths = _material_array(
            _api_cotangent_paths,
            "cotangent_paths",
            ctx,
            path_dtype,
            ((materials, geometry.pixels),),
        )
        reads.append(("cotangent_paths", _api_cotangent_paths))
    if _api_out_fields is not None:
        if not spec.active_volume:
            raise ContractError("material fields are fixed in this projection specification")
        if _api_cotangent_paths is None:
            raise ContractError("material field gradients require a bound path cotangent")
        _api_out_fields = _material_array(
            _api_out_fields,
            "out_fields",
            ctx,
            getattr(wp, volume_gradient_precision),
            ((materials, grid.voxels), (materials, *grid.shape)),
        )
        writes.append(("out_fields", _api_out_fields))
    ctx.disjoint(reads, writes)
    with ctx.scope():
        status = wp.zeros(1, dtype=wp.int32, device=ctx.device)
    field_views = tuple(
        _api_fields[m * grid.voxels : (m + 1) * grid.voxels] for m in range(materials)
    )
    path_views = tuple(
        _api_out_paths[m * geometry.pixels : (m + 1) * geometry.pixels] for m in range(materials)
    )
    seeds = (
        tuple(
            _api_cotangent_paths[m * geometry.pixels : (m + 1) * geometry.pixels]
            for m in range(materials)
        )
        if _api_cotangent_paths is not None
        else ()
    )
    gradients = (
        tuple(_api_out_fields[m * grid.voxels : (m + 1) * grid.voxels] for m in range(materials))
        if _api_out_fields is not None
        else ()
    )
    return MaterialProjectionWorkspace(
        projection,
        materials,
        _api_fields,
        _api_out_paths,
        _api_cotangent_paths,
        _api_out_fields,
        field_views,
        path_views,
        seeds,
        gradients,
        load_kernels("dpt.kernels.material_projection"),
        status,
        field_domain,
        channel_execution,
        volume_gradient_precision,
        _float32_volume_increments,
    )


def _material_array(
    values: Any, name: str, context: DeviceContext, dtype: Any, shapes: tuple[tuple[int, ...], ...]
) -> Any:
    for shape in shapes:
        if getattr(values, "ndim", 1) == len(shape):
            context.array(values, name, dtype=dtype, shape=shape, ndim=len(shape))
            return values.flatten()
    context.array(values, name, dtype=dtype, shape=(math.prod(shapes[0]),))
    return values


def _validate(workspace: MaterialProjectionWorkspace, *, seeds: bool) -> None:
    ctx, projection = workspace.context, workspace.projection
    workspace.check_status()
    ctx.wp.launch(
        workspace.kernels.validate_fractions
        if workspace.field_domain == "fractions"
        else workspace.kernels.validate_nonnegative,
        dim=projection.grid.voxels,
        inputs=[workspace.fields, workspace.materials, projection.grid.voxels],
        outputs=[workspace.status],
        device=ctx.device,
        stream=ctx.stream,
        record_tape=False,
    )
    if seeds:
        ctx.wp.launch(
            workspace.kernels.validate_seeds(projection.spec.precision),
            dim=workspace.materials * projection.geometry.pixels,
            inputs=[workspace.cotangent_paths],
            outputs=[workspace.status],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
    workspace.consume_status()


def project_material_paths(
    pose: wp.array[wp.float64],
    *,
    workspace: MaterialProjectionWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """A[m,p] = integral f_m(r_p(s)) ds projects dimensionless material fields into mm.

    The workspace binds FP32 fields in material-major (M,voxels) or
    (M,nz,ny,nx) layout and paths in (M,P), with equivalent flat contiguous
    storage. Paths use declared FP32/FP64 precision. Pose is FP64 (12,),
    ordered row-major object-to-world rotation then translation in mm. Fraction
    fields are nonnegative and sum to at most one within the stated rounding
    tolerance, while nonnegative equivalent-basis fields have no sum cap.
    Serial execution projects each material and fused execution shares canonical
    ray traversal. Sampling follows the projection policy: float32 uses FP32
    local coordinates, interpolation and material-path accumulation; float64
    and the serial mixed_float32 comparison accumulate in FP64. Initial ray
    setup remains FP64. The bound destinations are overwritten without allocating
    views or device arrays. Declared field/pose activity supports an explicit material
    VJP and ambient tapes are rejected.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_pose: Any = cast(Any, pose)
    require_no_tape()
    ctx, projection = workspace.context, workspace.projection
    ctx.assert_stream(stream)
    if projection._recorded_tape is not None:
        raise ContractError("the outstanding tape owns the material projector")
    ctx.array(_api_pose, "pose", dtype=ctx.wp.float64, shape=(12,))
    ctx.disjoint([("pose", _api_pose)], [("paths", workspace.paths)])
    if validate:
        workspace.validate_inputs(stream=ctx.stream)
        ctx.wp.launch(
            projection._geometry_kernels.validate_pose,
            dim=1,
            inputs=[_api_pose],
            outputs=[projection._status],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
        workspace.consume_status()
    with ctx.scope():
        if workspace.channel_execution == "fused":
            ctx.wp.launch(
                _arithmetic_kernels(workspace).get_material_forward(
                    workspace.materials,
                    projection.spec.precision == "float64",
                    projection.spec.integration == "cell_gauss",
                ),
                dim=projection.geometry.pixels,
                inputs=[
                    workspace.fields,
                    _api_pose,
                    projection._configuration,
                    projection.grid.voxels,
                ],
                outputs=[workspace.paths, projection._status],
                device=ctx.device,
                stream=ctx.stream,
                block_dim=128,
                record_tape=False,
            )
        else:
            # Fractions and pose are already certified once. Launch the
            # canonical per-field kernel without repeating volume scans/waits.
            kernel = _forward_kernel(projection)
            for material in range(workspace.materials):
                ctx.wp.launch(
                    kernel,
                    dim=projection.geometry.pixels,
                    inputs=[workspace.field_views[material], _api_pose, projection._configuration],
                    outputs=[workspace.path_views[material], projection._status],
                    device=ctx.device,
                    stream=ctx.stream,
                    block_dim=128,
                    record_tape=False,
                )
    if validate:
        workspace.consume_status()


def material_projection_vjp(
    pose: wp.array[wp.float64],
    *,
    workspace: MaterialProjectionWorkspace,
    out_pose_tangent: wp.array[wp.float64] | None = None,
    out_pose_matrix: wp.array[wp.float64] | None = None,
    stream: wp.Stream | None = None,
    validate: bool = True,
    accumulate: bool = False,
    check_volume_gradient: bool = True,
) -> None:
    """Grad_f[m,:] = A(pose)^T*cotangent_paths[m,:] and grad_pose=sum_{m,p}
    c[m,p]*dA[m,p]/dpose.

    FP32 fields and their separately selected FP32/FP64 gradient destination have material-major
    (M,voxels) or (M,nz,ny,nx) layout. The bound path covector is (M,P)
    in the declared FP32/FP64 path precision, with paths in mm. Pose is
    FP64 (12,) object-to-world storage. Out_pose_tangent is FP64 (6,) for
    local-right tx,ty,tz,rx,ry,rz in mm/rad. Out_pose_matrix is FP64 (12,)
    for row-major rotation then translation in mm, with exactly one pose
    destination selected.
    Only declared active field/pose parameters are differentiated and caller
    covectors are preserved. Field gradients use the canonical scatter-atomic
    contract. Serial/fused policies share traversal but have distinct sum orders.
    These are ambient field derivatives, so constrained fraction updates must
    still respect the simplex. Deferred field checks require unchecked
    accumulation and a final finite check of every field-gradient entry.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_pose: Any = cast(Any, pose)
    _api_out_pose_tangent: Any = cast(Any, out_pose_tangent)
    _api_out_pose_matrix: Any = cast(Any, out_pose_matrix)
    require_no_tape()
    if type(check_volume_gradient) is not bool:
        raise ContractError("check_volume_gradient must be a boolean")
    if workspace.volume_gradient_precision not in ("float32", "float64"):
        raise ContractError("volume_gradient_precision must be float32 or float64")
    if type(workspace._float32_volume_increments) is not bool or (
        workspace._float32_volume_increments and workspace.volume_gradient_precision != "float64"
    ):
        raise ContractError("FP32 volume increments require an FP64 gradient destination")
    if not check_volume_gradient and (validate or not accumulate):
        raise ContractError("deferred volume checks require unchecked accumulating material VJPs")
    ctx, projection = workspace.context, workspace.projection
    ctx.assert_stream(stream)
    if projection._recorded_tape is not None:
        raise ContractError("the outstanding tape owns the material projector")
    if _api_out_pose_tangent is not None and _api_out_pose_matrix is not None:
        raise ContractError("request either a right-tangent or a storage pose gradient")
    ctx.array(_api_pose, "pose", dtype=ctx.wp.float64, shape=(12,))
    if not workspace.seed_views:
        raise ContractError("prepare a path cotangent before requesting material VJPs")
    writes: list[tuple[str, Any]] = []
    pose_target = (
        _api_out_pose_tangent if _api_out_pose_tangent is not None else _api_out_pose_matrix
    )
    if pose_target is not None:
        if not projection.spec.active_pose:
            raise ContractError("pose is fixed in this projection specification")
        pose_name = "out_pose_tangent" if _api_out_pose_tangent is not None else "out_pose_matrix"
        ctx.array(
            pose_target,
            pose_name,
            dtype=ctx.wp.float64,
            shape=(6 if _api_out_pose_tangent is not None else 12,),
        )
        writes.append((pose_name, pose_target))
    if workspace.grad_fields is not None:
        ctx.array(
            workspace.grad_fields,
            "out_fields",
            dtype=getattr(ctx.wp, workspace.volume_gradient_precision),
            shape=(workspace.materials * projection.grid.voxels,),
        )
        if len(workspace.gradient_views) != workspace.materials:
            raise ContractError("material gradient views must retain their prepared destination")
        for material, values in enumerate(workspace.gradient_views):
            ctx.array(
                values,
                "material gradient view",
                dtype=getattr(ctx.wp, workspace.volume_gradient_precision),
                shape=(projection.grid.voxels,),
            )
            expected_pointer = int(
                workspace.grad_fields.ptr
            ) + material * projection.grid.voxels * (
                8 if workspace.volume_gradient_precision == "float64" else 4
            )
            if int(values.ptr) != expected_pointer:
                raise ContractError(
                    "material gradient views must retain their prepared destination"
                )
        writes.append(("out_fields", workspace.grad_fields))
    if not writes:
        raise ContractError("request a pose or material field gradient")
    ctx.disjoint(
        [
            ("fields", workspace.fields),
            ("paths", workspace.paths),
            ("cotangent_paths", workspace.cotangent_paths),
            ("pose", _api_pose),
        ],
        writes,
    )
    if validate:
        workspace.validate_inputs(seeds=True, stream=ctx.stream)
        ctx.wp.launch(
            projection._geometry_kernels.validate_pose,
            dim=1,
            inputs=[_api_pose],
            outputs=[projection._status],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
        workspace.consume_status()
    if workspace.channel_execution == "fused":
        with ctx.scope():
            if workspace.grad_fields is not None and not accumulate:
                workspace.grad_fields.zero_()
            count = (
                (projection.geometry.pixels + 127) // 128
                if ctx.device.is_cuda
                else projection.geometry.pixels
            )
            ctx.wp.launch_tiled(
                _arithmetic_kernels(workspace).get_material_vjp(
                    workspace.materials,
                    projection.spec.precision == "float64",
                    projection.spec.integration == "cell_gauss",
                    workspace.grad_fields is not None,
                    _api_out_pose_tangent is not None,
                    workspace.volume_gradient_precision == "float64"
                    and workspace.grad_fields is not None,
                    workspace._float32_volume_increments,
                ),
                dim=count,
                block_dim=128,
                inputs=[
                    workspace.fields,
                    _api_pose,
                    projection._configuration,
                    projection.grid.voxels,
                    workspace.cotangent_paths,
                ],
                outputs=[
                    workspace.grad_fields
                    if workspace.grad_fields is not None
                    else projection._empty,
                    projection._reduction.partials[0]
                    if projection._reduction is not None
                    else projection._empty_partials,
                    projection._status,
                ],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
            if _api_out_pose_tangent is not None:
                assert projection._reduction is not None
                projection._reduction.finish(
                    count,
                    _api_out_pose_tangent,
                    projection._status,
                    components=6,
                    accumulate=accumulate,
                )
            if workspace.grad_fields is not None and check_volume_gradient:
                ctx.wp.launch(
                    projection._kernels.get_validate_finite(
                        workspace.volume_gradient_precision == "float64"
                    ),
                    dim=workspace.grad_fields.size,
                    inputs=[workspace.grad_fields],
                    outputs=[projection._status],
                    device=ctx.device,
                    stream=ctx.stream,
                    record_tape=False,
                )
        if validate:
            workspace.consume_status()
        if _api_out_pose_matrix is not None:
            for material in range(workspace.materials):
                _vjp(
                    workspace.field_views[material],
                    _api_pose,
                    cotangent=workspace.seed_views[material],
                    workspace=projection,
                    out_pose_matrix=_api_out_pose_matrix,
                    stream=ctx.stream,
                    validate=False,
                    accumulate=accumulate or material > 0,
                )
            if validate:
                workspace.consume_status()
        return
    for material in range(workspace.materials):
        # Pose contributions share one destination; distinct field slices must
        # each honour the caller's overwrite/accumulate request independently.
        field_gradient = workspace.gradient_views[material] if workspace.gradient_views else None
        if (
            pose_target is not None
            and field_gradient is not None
            and material > 0
            and not accumulate
        ):
            with ctx.scope():
                field_gradient.zero_()
        _vjp(
            workspace.field_views[material],
            _api_pose,
            cotangent=workspace.seed_views[material],
            workspace=projection,
            out_pose_tangent=_api_out_pose_tangent,
            out_pose_matrix=_api_out_pose_matrix,
            out_mu=field_gradient,
            stream=ctx.stream,
            validate=False,
            accumulate=accumulate or (pose_target is not None and material > 0),
            check_volume_gradient=check_volume_gradient,
            volume_gradient_precision=workspace.volume_gradient_precision,
            _float32_volume_increments=workspace._float32_volume_increments,
        )
    if validate:
        workspace.consume_status()


def _arithmetic_kernels(workspace: MaterialProjectionWorkspace) -> Any:
    if workspace.projection.spec.sampling_precision == "float32":
        return load_kernels("dpt.kernels.material_projection_fp32")
    return workspace.kernels
