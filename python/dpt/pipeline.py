"""Prepared material projection followed by polychromatic transmission."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._runtime import DeviceContext, StatusCheckpoint, require_no_tape
from dpt.contracts import ContractError
from dpt.geometry import DetectorGeometry, RayBundleGeometry
from dpt.material_projection import (
    MaterialProjectionWorkspace,
    material_projection_vjp,
    prepare_material_projection,
    project_material_paths,
)
from dpt.projection import ProjectionSpec
from dpt.spectral import (
    SpectralInputs,
    prepare_spectral,
    spectral_signal,
    spectral_vjp,
)
from dpt.volumes import GridSpec

if TYPE_CHECKING:
    from collections.abc import Generator

    import warp as wp

    FloatArray = wp.array[wp.float32] | wp.array[wp.float64]


@dataclass(frozen=True, slots=True)
class SpectralGradients:
    """Cotangents of a seeded image, ``dL = <grad_image, dI>``.

    ``fields`` has shape ``(M,nz,ny,nx)`` and binary32 storage. ``pose`` has six
    binary64 partial derivatives with respect to local-right increment
    coordinates ``(tx,ty,tz,rx,ry,rz)``: translation measured in mm followed
    by rotation in radians. Coefficients have shape ``(M,K)``;
    weights and responses follow their
    shared ``(K,)`` or pixel-specific ``(K,P)`` layout. Shared spectral cotangents
    use binary64 storage. Fixed parameters have a value of ``None``.

    Arrays belong to the pipeline and are overwritten by its next VJP. Clone a
    buffer when retaining gradients across evaluations.
    """

    fields: wp.array[wp.float32] | None
    pose: wp.array[wp.float64] | None
    coefficients: FloatArray | None
    weights: FloatArray | None
    response: FloatArray | None


class SpectralPipeline:
    """Compute ``I[p] = sum_k w[k,p] r[k,p] exp(-sum_m mu[m,k] A[m,p])``.

    ``A`` contains ray integrals of dimensionless material fields, in mm. Fields
    have shape ``(M,nz,ny,nx)`` and the pose contains twelve binary64 values: a
    row-major rotation followed by translation in mm. A planar detector returns
    an image with shape ``(rows,columns)``; explicit rays return ``(P,)``.

    The pipeline owns reusable scratch and takes independent field/pose copies
    on every forward. VJP uses the most recent successful forward, including
    when the caller subsequently edits its original arrays. Fixed spectral
    factors are checked when preparing ``SpectralInputs``; field and pose values
    are checked at each forward. Calls execute serially on the inputs' context.
    ``execution="graph"`` captures forward and reverse launches on CUDA. Input
    staging, value checks and completion checkpoints remain outside capture.

    Example::

        pipeline = SpectralPipeline(grid, geometry, projection_spec, inputs)
        image = pipeline.forward(fields, pose)
        gradients = pipeline.vjp(grad_image)

    The returned image and gradients are reusable pipeline buffers. Clone them
    to retain a result across subsequent calls.
    """

    def __init__(
        self,
        grid: GridSpec,
        geometry: DetectorGeometry | RayBundleGeometry,
        projection: ProjectionSpec,
        inputs: SpectralInputs,
        *,
        field_domain: Literal["fractions", "nonnegative"] = "fractions",
        channel_execution: Literal["serial", "fused"] = "serial",
        execution: Literal["eager", "graph"] = "eager",
    ) -> None:
        require_no_tape()
        if execution not in ("eager", "graph"):
            raise ContractError("execution must be eager or graph")
        if inputs.pixels != geometry.pixels:
            raise ContractError("spectral inputs and geometry must have the same pixel count")
        if projection.precision != inputs.spec.precision:
            raise ContractError("projection and spectral precision must match")
        if (projection.active_pose or projection.active_volume) and not inputs.spec.active_paths:
            raise ContractError("field or pose derivatives require active spectral paths")
        self.grid, self.geometry, self.inputs = grid, geometry, inputs
        self.context: DeviceContext = inputs.context
        ctx = self.context
        if execution == "graph" and not ctx.device.is_cuda:
            raise ContractError("graph execution requires CUDA")
        self.execution = execution
        self._forward_graph: Any = None
        self._vjp_graph: Any = None
        ctx.assert_stream(ctx.stream)
        materials, pixels = inputs.spec.materials, geometry.pixels
        self._lock = Lock()
        self._has_forward = False
        with ctx.scope():
            self._fields = cast(
                "wp.array[wp.float32]",
                ctx.wp.empty((materials, *grid.shape), dtype=ctx.wp.float32, device=ctx.device),
            )
            self._pose = cast(
                "wp.array[wp.float64]", ctx.wp.empty(12, dtype=ctx.wp.float64, device=ctx.device)
            )
            paths = ctx.wp.empty(
                (materials, pixels), dtype=getattr(ctx.wp, projection.precision), device=ctx.device
            )
            adj_paths = ctx.wp.empty_like(paths)
            grad_fields = ctx.wp.empty_like(self._fields) if projection.active_volume else None
            self._pose_gradient = cast(
                "wp.array[wp.float64] | None",
                ctx.wp.empty(6, dtype=ctx.wp.float64, device=ctx.device)
                if projection.active_pose
                else None,
            )
            self._mean = cast(
                "FloatArray",
                ctx.wp.empty(
                    pixels, dtype=getattr(ctx.wp, projection.precision), device=ctx.device
                ),
            )
            self._cotangent = ctx.wp.empty_like(self._mean)
        self._material: MaterialProjectionWorkspace = prepare_material_projection(
            grid,
            geometry,
            projection,
            materials=materials,
            fields=self._fields,
            out_paths=paths,
            cotangent_paths=adj_paths,
            out_fields=grad_fields,
            device=ctx.device,
            stream=ctx.stream,
            field_domain=field_domain,
            channel_execution=channel_execution,
        )
        self._spectral = prepare_spectral(
            inputs.spec, max_pixels=pixels, device=ctx.device, stream=ctx.stream
        )
        self._checkpoint = StatusCheckpoint(ctx, (self._material, self._spectral))
        with ctx.scope():
            self._coefficients_gradient = self._parameter_gradient(
                inputs.spec.active_coefficients, (materials, inputs.spec.energies), shared=True
            )
            self._weights_gradient = self._parameter_gradient(
                inputs.spec.active_weights,
                (inputs.spec.energies,)
                if inputs.spec.shared_weights
                else (inputs.spec.energies, pixels),
                shared=inputs.spec.shared_weights,
            )
            self._response_gradient = self._parameter_gradient(
                inputs.spec.active_response,
                (inputs.spec.energies,)
                if inputs.spec.shared_response
                else (inputs.spec.energies, pixels),
                shared=inputs.spec.shared_response,
            )
        self._gradients = SpectralGradients(
            cast("wp.array[wp.float32] | None", grad_fields),
            self._pose_gradient,
            self._coefficients_gradient,
            self._weights_gradient,
            self._response_gradient,
        )
        self._image_shape = geometry.shape if isinstance(geometry, DetectorGeometry) else (pixels,)
        self._image = cast("FloatArray", cast(Any, self._mean).reshape(self._image_shape))

    def _parameter_gradient(
        self, active: bool, shape: tuple[int, ...], *, shared: bool
    ) -> FloatArray | None:
        if not active:
            return None
        ctx = self.context
        return cast(
            "FloatArray",
            ctx.wp.empty(
                shape, dtype=ctx.wp.float64 if shared else ctx.wp.float32, device=ctx.device
            ),
        )

    @contextmanager
    def _evaluation(self) -> Generator[None, None, None]:
        require_no_tape()
        self.context.assert_stream(self.context.stream)
        if not self._lock.acquire(blocking=False):
            raise ContractError("a pipeline evaluation is already running")
        try:
            yield
        except BaseException:
            if self.context.device.is_cuda:
                self.context.wp.synchronize_stream(self.context.stream)
            raise
        finally:
            self._lock.release()

    def _field_view(self, fields: wp.array[wp.float32]) -> wp.array[wp.float32]:
        ctx = self.context
        dimensions = getattr(fields, "ndim", 1)
        if dimensions == 4:
            shape = (self.inputs.spec.materials, *self.grid.shape)
        elif dimensions == 2:
            shape = (self.inputs.spec.materials, self.grid.voxels)
        else:
            shape = (self.inputs.spec.materials * self.grid.voxels,)
        ctx.array(fields, "fields", dtype=ctx.wp.float32, shape=shape, ndim=len(shape))
        return cast("wp.array[wp.float32]", fields.flatten())

    def forward(self, fields: wp.array[wp.float32], pose: wp.array[wp.float64]) -> FloatArray:
        """I[p] = sum_k w[k,p]*r[k,p]*exp(-sum_m mu[m,k]*integral f_m(r_p(s)) ds).

        Fields is FP32 (M,nz,ny,nx), (M,voxels) or equivalent flat material-major
        storage of dimensionless fractions/equivalent-basis fields. Pose is FP64
        (12,) row-major object-to-world rotation then translation in mm. The returned
        image is (rows,columns), or flat (P,) for an explicit ray bundle, in matched
        projection/spectral precision. Its units follow the supplied response per
        photon and output_domain may select its natural logarithm. Coefficients,
        weights and response are the fixed validated copies in SpectralInputs.
        Each successful call copies the current field/pose into an owned snapshot
        for the declared active VJP parameters and replaces the preceding state.
        A failed forward invalidates that state. The image belongs to the pipeline
        and is overwritten by its next forward, so clone it to retain a result.

        """
        with self._evaluation():
            ctx = self.context
            self._has_forward = False
            fields = self._field_view(fields)
            ctx.array(pose, "pose", dtype=ctx.wp.float64, shape=(12,))
            with ctx.scope():
                ctx.wp.copy(self._fields.flatten(), fields, stream=ctx.stream)
                ctx.wp.copy(self._pose, pose, stream=ctx.stream)
            if self.execution == "eager":
                project_material_paths(self._pose, workspace=self._material, validate=True)
                self._launch_signal()
            else:
                self._material.validate_inputs(stream=ctx.stream)
                self._material.projection.validate_pose(self._pose, stream=ctx.stream)
                if self._forward_graph is None:
                    self._launch_forward()
                    self._checkpoint.consume()
                    with (
                        ctx.scope(),
                        ctx.wp.ScopedCapture(device=ctx.device, stream=ctx.stream) as capture,
                    ):
                        self._launch_forward()
                    self._forward_graph = capture.graph
                ctx.wp.capture_launch(self._forward_graph, stream=ctx.stream)
            self._checkpoint.consume()
            self._has_forward = True
            return self._image

    def vjp(self, grad_image: FloatArray) -> SpectralGradients:
        """Gradients = J(saved_fields,saved_pose,factors)^T*grad_image.

        Grad_image is (rows,columns) or flat (P,) in retained image precision.
        The saved fields have FP32 (M,nz,ny,nx) layout and saved pose is FP64 (12,)
        row-major object-to-world rotation/translation in mm. SpectralGradients
        returns FP32 fields (M,nz,ny,nx), FP64 local-right pose (6,) in mm/rad,
        FP64 coefficients (M,K) and shared factor gradients (K,), or FP32
        pixel-specific factor gradients (K,P). Fixed parameters return None.
        Active field/pose requires active_paths. The first-order chain recomputes
        the selected spectral derivatives and canonical material reverse from the
        latest successful owned snapshot, preserving the caller's image covector.
        Returned buffers belong to the pipeline and the next VJP overwrites them.

        """
        with self._evaluation():
            if not self._has_forward:
                raise ContractError("run a successful forward before requesting its VJP")
            ctx = self.context
            shape = (
                self._image_shape if getattr(grad_image, "ndim", 1) > 1 else (self.geometry.pixels,)
            )
            ctx.array(
                grad_image,
                "grad_image",
                dtype=self._spectral.dtype,
                shape=shape,
                ndim=len(shape),
            )
            with ctx.scope():
                ctx.wp.copy(self._cotangent, grad_image.flatten(), stream=ctx.stream)
            self._spectral.validate_cotangent(self._cotangent, stream=ctx.stream)
            if self.execution == "eager":
                self._launch_vjp()
            else:
                if self._vjp_graph is None:
                    self._launch_vjp()
                    self._checkpoint.consume()
                    with (
                        ctx.scope(),
                        ctx.wp.ScopedCapture(device=ctx.device, stream=ctx.stream) as capture,
                    ):
                        self._launch_vjp()
                    self._vjp_graph = capture.graph
                ctx.wp.capture_launch(self._vjp_graph, stream=ctx.stream)
            self._checkpoint.consume()
            return self._gradients

    def _launch_signal(self) -> None:
        spectral_signal(
            self._material.paths,
            self.inputs.coefficients,
            self.inputs.weights,
            self.inputs.response,
            out_mean=self._mean,
            workspace=self._spectral,
            stream=self.context.stream,
            validate=False,
        )

    def _launch_forward(self) -> None:
        project_material_paths(
            self._pose,
            workspace=self._material,
            stream=self.context.stream,
            validate=False,
        )
        self._launch_signal()

    def _launch_vjp(self) -> None:
        spectral_vjp(
            self._material.paths,
            self.inputs.coefficients,
            self.inputs.weights,
            self.inputs.response,
            cotangent=self._cotangent,
            out_grad_paths=(
                self._material.cotangent_paths if self.inputs.spec.active_paths else None
            ),
            out_grad_coefficients=self._coefficients_gradient,
            out_grad_weights=self._weights_gradient,
            out_grad_response=self._response_gradient,
            workspace=self._spectral,
            stream=self.context.stream,
            validate=False,
        )
        if self._pose_gradient is not None or self._material.grad_fields is not None:
            material_projection_vjp(
                self._pose,
                workspace=self._material,
                out_pose_tangent=self._pose_gradient,
                stream=self.context.stream,
                validate=False,
            )
