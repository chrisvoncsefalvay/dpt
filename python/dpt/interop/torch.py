"""Differentiate ``L_p = integral mu(G^-1 r_p(s)) ds`` through Torch.

Scalar projection returns dimensionless optical depth; material paths
feed the spectral sum ``lambda_p = sum_k w_kp R_kp exp(-sum_m C_mk A_mp)``.
Torch owns the returned tensors and gradients. Warp views their storage
on the same device; CUDA events connect caller and workspace streams.
Native reverse kernels provide first-order gradients. Use
:func:`dpt.torch.geometry.compose_pose` to connect twelve packed pose
entries to six rigid-motion coordinates in mm and radians.

"""

from __future__ import annotations

from copy import copy
from typing import Any

import torch

from dpt._runtime import StatusCheckpoint, require_no_tape
from dpt.contracts import ContractError
from dpt.interop._torch_streams import stream_bridge
from dpt.material_projection import (
    MaterialProjectionWorkspace,
    material_projection_vjp,
    project_material_paths,
)
from dpt.projection import ProjectionWorkspace
from dpt.projection import project_optical_depth as _project
from dpt.projection import projection_vjp as _vjp
from dpt.spectral import SpectralWorkspace, spectral_signal, spectral_vjp

__all__ = [
    "ProjectionFunction",
    "SpectralProjection",
    "SpectralProjectionFunction",
    "project_optical_depth",
    "project_spectral_signal",
]


def _pack_cotangent(torch: Any, value: Any) -> Any:
    """Return flat storage with the element stride required by native kernels."""
    packed = value.contiguous().reshape(-1)
    # Torch treats singleton dimensions as contiguous regardless of their
    # stride. Reduction and sliced singleton seeds can therefore still carry
    # zero or non-unit strides after contiguous(); Warp requires stride one.
    if packed.stride() != (1,):
        canonical = torch.empty(packed.shape, dtype=packed.dtype, device=packed.device)
        canonical.copy_(packed)
        packed = canonical
    return packed


def _bind_material_fields(
    workspace: MaterialProjectionWorkspace, fields: Any, gradient: Any = None
) -> MaterialProjectionWorkspace:
    """Clone binding metadata while retaining the prepared owning-stream scratch."""
    bound = copy(workspace)
    wp, voxels = workspace.context.wp, workspace.projection.grid.voxels
    bound.fields = wp.from_torch(fields.reshape(-1), requires_grad=False)
    bound.field_views = tuple(
        bound.fields[material * voxels : (material + 1) * voxels]
        for material in range(workspace.materials)
    )
    # Reverse requests only its selected destinations. A bound field-gradient
    # buffer from preparation must not become a hidden destination for a
    # pose-only call.
    bound.grad_fields = None
    bound.gradient_views = ()
    if gradient is not None:
        bound.grad_fields = wp.from_torch(gradient.reshape(-1), requires_grad=False)
        bound.gradient_views = tuple(
            bound.grad_fields[material * voxels : (material + 1) * voxels]
            for material in range(workspace.materials)
        )
    return bound


def _check_spectral_aliases(
    material: MaterialProjectionWorkspace,
    spectral: SpectralWorkspace,
    fields: Any,
    factors: tuple[Any, ...],
) -> None:
    """Certify new tensor bindings against every prepared writable buffer."""
    ctx = material.context
    # These scratch bindings are internal workspace metadata; they are not
    # part of the public operator interface exposed to Torch callers.
    projection: Any = material.projection
    spectral_scratch: Any = spectral
    if any(not value.is_contiguous() for value in factors):
        raise ContractError("spectral inputs must use contiguous tensor storage")
    reads = [
        (name, ctx.wp.from_torch(value, requires_grad=False))
        for name, value in zip(
            ("fields", "pose", "coefficients", "weights", "response"),
            (fields, *factors),
            strict=True,
        )
    ]
    writes = [
        ("material_paths", material.paths),
        ("material_path_cotangents", material.cotangent_paths),
        ("material_status", material.status),
        ("projection_status", projection._status),
        ("spectral_status", spectral_scratch._status),
        ("spectral_partials", spectral_scratch._partials),
        ("spectral_log_mean", spectral_scratch._log_mean),
    ]
    reduction = getattr(projection, "_reduction", None)
    if reduction is not None:
        writes.extend(
            (f"projection_partials_{index}", partial)
            for index, partial in enumerate(reduction.partials)
        )
    ctx.disjoint(reads, [(name, value) for name, value in writes if value is not None])


class ProjectionFunction(torch.autograd.Function):
    """First-order field and packed-pose derivatives with fixed acquisition."""

    @staticmethod
    def forward(
        owner: Any,
        volume: Any,
        pose: Any,
        workspace: ProjectionWorkspace,
        validate: bool,
    ) -> Any:
        require_no_tape()
        if not isinstance(volume, torch.Tensor) or not isinstance(pose, torch.Tensor):
            raise ContractError("volume and packed pose must be Torch tensors")
        if volume.dtype != torch.float32 or not volume.is_contiguous():
            raise ContractError("volume must be contiguous Torch float32 storage")
        if tuple(volume.shape) not in (workspace.grid.shape, (workspace.grid.voxels,)):
            raise ContractError("volume must have the prepared grid's zyx or flat shape")
        if (
            pose.dtype != torch.float64
            or tuple(pose.shape) != (12,)
            or not pose.is_contiguous()
            or pose.device != volume.device
        ):
            raise ContractError("pose must be a contiguous float64 (12,) tensor on volume's device")
        if pose.requires_grad and not workspace.spec.active_pose:
            raise ContractError("pose is fixed; prepare active_pose=True for its derivative")
        if volume.requires_grad and not workspace.spec.active_volume:
            raise ContractError("prepare active_volume=True for the volume derivative")
        if type(validate) is not bool:
            raise ContractError("validate must be a boolean")
        ctx, wp = workspace.context, workspace.context.wp
        output_shape = getattr(workspace.geometry, "shape", (workspace.geometry.pixels,))
        dtype = torch.float64 if workspace.spec.precision == "float64" else torch.float32
        with torch.no_grad(), stream_bridge(torch, (volume, pose), workspace) as bridge:
            output = torch.empty(output_shape, dtype=dtype, device=volume.device)
            bridge.record(output)
            _project(
                wp.from_torch(volume.reshape(-1), requires_grad=False),
                wp.from_torch(pose, requires_grad=False),
                workspace=workspace,
                out_L=wp.from_torch(output.reshape(-1), requires_grad=False),
                validate=validate,
                stream=ctx.stream,
            )
            workspace.consume_status()
        # PyTorch validates version counters when saved_tensors is read
        # in backward, including a fixed pose mutated after forward.
        owner.save_for_backward(volume, pose)
        owner.workspace = workspace
        return output

    @staticmethod
    def backward(owner: Any, seed: Any) -> tuple[Any, Any, None, None]:
        if torch.is_grad_enabled():
            raise ContractError(
                "the Torch projection adapter supports first-order derivatives only"
            )
        require_no_tape()
        volume, pose = owner.saved_tensors
        workspace = owner.workspace
        ctx, wp = workspace.context, workspace.context.wp
        with (
            torch.no_grad(),
            stream_bridge(torch, (volume, pose, seed), workspace) as bridge,
        ):
            # Autograd can pass expanded or strided cotangents. Pack
            # that seed explicitly; preserve the caller's seed storage.
            packed_seed = _pack_cotangent(torch, seed)
            gradient = (
                torch.empty_like(volume, memory_format=torch.contiguous_format)
                if owner.needs_input_grad[0]
                else None
            )
            pose_gradient = torch.empty_like(pose) if owner.needs_input_grad[1] else None
            bridge.record(packed_seed)
            bridge.record(*(value for value in (gradient, pose_gradient) if value is not None))
            _vjp(
                wp.from_torch(volume.reshape(-1), requires_grad=False),
                wp.from_torch(pose, requires_grad=False),
                workspace=workspace,
                cotangent=wp.from_torch(packed_seed, requires_grad=False),
                out_mu=wp.from_torch(gradient.reshape(-1), requires_grad=False)
                if gradient is not None
                else None,
                out_pose_matrix=wp.from_torch(pose_gradient, requires_grad=False)
                if pose_gradient is not None
                else None,
                validate=True,
                stream=ctx.stream,
            )
            workspace.consume_status()
        return gradient, pose_gradient, None, None


class SpectralProjectionFunction(torch.autograd.Function):
    """Selected field, packed-pose and spectral-factor derivatives."""

    @staticmethod
    def forward(
        owner: Any,
        fields: Any,
        pose: Any,
        coefficients: Any,
        weights: Any,
        response: Any,
        material: MaterialProjectionWorkspace,
        spectral: SpectralWorkspace,
        validate: bool,
    ) -> Any:
        require_no_tape()
        factors = (pose, coefficients, weights, response)
        if not isinstance(fields, torch.Tensor) or any(
            not isinstance(value, torch.Tensor) for value in factors
        ):
            raise ContractError("material fields and spectral inputs must be Torch tensors")
        grid = material.projection.grid
        if (
            fields.dtype != torch.float32
            or not fields.is_contiguous()
            or tuple(fields.shape)
            not in ((material.materials, *grid.shape), (material.materials, grid.voxels))
        ):
            raise ContractError("material fields must be contiguous float32 material-major storage")
        if pose.dtype != torch.float64 or tuple(pose.shape) != (12,) or not pose.is_contiguous():
            raise ContractError("pose must be contiguous float64 (12,) storage")
        if any(value.dtype != torch.float32 for value in factors[1:]):
            raise ContractError("spectral coefficients, weights and response use float32")
        if any(value.device != fields.device for value in factors):
            raise ContractError("all spectral projection tensors must share one device")
        for name, value, active in (
            ("fields", fields, material.projection.spec.active_volume),
            ("pose", pose, material.projection.spec.active_pose),
            ("coefficients", coefficients, spectral.spec.active_coefficients),
            ("weights", weights, spectral.spec.active_weights),
            ("response", response, spectral.spec.active_response),
        ):
            if value.requires_grad and not active:
                raise ContractError(f"{name} is fixed; prepare its active flag for autograd")
        if fields.requires_grad or pose.requires_grad:
            if not spectral.spec.active_paths or material.cotangent_paths is None:
                raise ContractError(
                    "trainable fields or pose require active spectral paths "
                    "and prepared material path cotangents"
                )
        if (
            spectral.spec.materials != material.materials
            or spectral.spec.precision != material.projection.spec.precision
            or spectral.context.device != material.context.device
        ):
            raise ContractError("material and spectral workspaces have incompatible specifications")
        material.context.assert_stream(spectral.context.stream)
        if type(validate) is not bool:
            raise ContractError("validate must be a boolean")
        ctx, wp = material.context, material.context.wp
        checkpoint = StatusCheckpoint(ctx, (material, spectral), aggregate=False)
        output_shape = getattr(
            material.projection.geometry, "shape", (material.projection.geometry.pixels,)
        )
        dtype = torch.float64 if spectral.spec.precision == "float64" else torch.float32
        with torch.no_grad(), stream_bridge(torch, (fields, *factors), material) as bridge:
            _check_spectral_aliases(material, spectral, fields, factors)
            checkpoint.consume()
            output = torch.empty(output_shape, dtype=dtype, device=fields.device)
            bridge.record(output)
            bound = _bind_material_fields(material, fields)
            project_material_paths(
                wp.from_torch(pose, requires_grad=False),
                workspace=bound,
                stream=ctx.stream,
                validate=validate,
            )
            spectral_signal(
                bound.paths,
                wp.from_torch(coefficients, requires_grad=False),
                wp.from_torch(weights, requires_grad=False),
                wp.from_torch(response, requires_grad=False),
                out_mean=wp.from_torch(output.reshape(-1), requires_grad=False),
                workspace=spectral,
                stream=ctx.stream,
                validate=validate,
            )
            checkpoint.consume()
        owner.save_for_backward(fields, *factors)
        owner.material, owner.spectral = material, spectral
        owner.checkpoint = checkpoint
        return output

    @staticmethod
    def backward(owner: Any, seed: Any) -> tuple[Any, ...]:
        if torch.is_grad_enabled():
            raise ContractError("the Torch spectral adapter supports first-order derivatives only")
        require_no_tape()
        fields, pose, coefficients, weights, response = owner.saved_tensors
        material, spectral = owner.material, owner.spectral
        ctx, wp = material.context, material.context.wp
        with (
            torch.no_grad(),
            stream_bridge(
                torch, (fields, pose, coefficients, weights, response, seed), material
            ) as bridge,
        ):
            owner.checkpoint.consume()
            gradients = tuple(
                torch.empty_like(value, memory_format=torch.contiguous_format)
                if owner.needs_input_grad[index]
                else None
                for index, value in enumerate((fields, pose, coefficients, weights, response))
            )
            (
                gradient,
                pose_gradient,
                coefficient_gradient,
                weight_gradient,
                response_gradient,
            ) = gradients
            packed_seed = _pack_cotangent(torch, seed)
            bridge.record(packed_seed, *(value for value in gradients if value is not None))
            bound = _bind_material_fields(material, fields, gradient)
            # Prepared paths are shared scratch, so recompute from the
            # saved immutable field before each reverse evaluation.
            project_material_paths(
                wp.from_torch(pose, requires_grad=False),
                workspace=bound,
                stream=ctx.stream,
                validate=True,
            )
            spectral_vjp(
                bound.paths,
                wp.from_torch(coefficients, requires_grad=False),
                wp.from_torch(weights, requires_grad=False),
                wp.from_torch(response, requires_grad=False),
                cotangent=wp.from_torch(packed_seed, requires_grad=False),
                out_grad_paths=bound.cotangent_paths
                if gradient is not None or pose_gradient is not None
                else None,
                out_grad_weights=wp.from_torch(weight_gradient, requires_grad=False)
                if weight_gradient is not None
                else None,
                out_grad_response=wp.from_torch(response_gradient, requires_grad=False)
                if response_gradient is not None
                else None,
                out_grad_coefficients=wp.from_torch(coefficient_gradient, requires_grad=False)
                if coefficient_gradient is not None
                else None,
                workspace=spectral,
                stream=ctx.stream,
                validate=True,
            )
            if gradient is not None or pose_gradient is not None:
                material_projection_vjp(
                    wp.from_torch(pose, requires_grad=False),
                    workspace=bound,
                    out_pose_matrix=wp.from_torch(pose_gradient, requires_grad=False)
                    if pose_gradient is not None
                    else None,
                    stream=ctx.stream,
                    validate=True,
                )
            owner.checkpoint.consume()
        return (*gradients, None, None, None)


class SpectralProjection(torch.nn.Module):
    """Supplied spectral model with registered parameters and a field batch loop."""

    def __init__(
        self,
        material_workspace: MaterialProjectionWorkspace,
        spectral_workspace: SpectralWorkspace,
        pose: Any,
        coefficients: Any,
        weights: Any,
        response: Any,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.material_workspace, self.spectral_workspace = (
            material_workspace,
            spectral_workspace,
        )
        for name, value in (
            ("pose", pose),
            ("coefficients", coefficients),
            ("weights", weights),
            ("response", response),
        ):
            if isinstance(value, torch.nn.Parameter):
                self.register_parameter(name, value)
            else:
                self.register_buffer(name, value)

    def forward(self, fields: Any) -> Any:
        material, grid = self.material_workspace, self.material_workspace.projection.grid
        single_shapes = (
            (material.materials, *grid.shape),
            (material.materials, grid.voxels),
        )

        def evaluate(item: Any) -> Any:
            return project_spectral_signal(
                item,
                self.pose,
                self.coefficients,
                self.weights,
                self.response,
                material_workspace=material,
                spectral_workspace=self.spectral_workspace,
            )

        if tuple(fields.shape) in single_shapes:
            return evaluate(fields)
        if tuple(fields.shape[1:]) not in single_shapes or fields.shape[0] < 1:
            raise ContractError(
                "batch material fields must have shape (B,M,nz,ny,nx) or (B,M,voxels)"
            )
        return torch.stack([evaluate(item) for item in fields])


def project_optical_depth(
    volume: Any,
    pose: Any,
    *,
    workspace: ProjectionWorkspace,
    validate: bool = True,
) -> Any:
    """Compute ``L_p = integral mu(G^-1 r_p(s)) ds`` and its first-order VJP.

    ``volume`` is contiguous FP32 ``(nz,ny,nx)`` or flat x-fastest storage
    in mm^-1. ``pose`` is FP64 ``(12,)``: row-major object-to-world rotation,
    then translation in mm. Both tensors share the workspace device.
    The result has detector shape ``(rows,columns)`` or flat ray-bundle
    shape ``(P,)`` and ``workspace.spec.precision``; optical depth is
    dimensionless. Quadrature and clipping follow the prepared projector.

    Set ``active_volume`` or ``active_pose`` in ``ProjectionSpec`` for each
    trainable tensor. Pose backward returns twelve storage partials; a
    rigid chart supplies its own chain rule. Each output owns Torch
    storage. Saved input tensors stay unchanged until backward, and
    Torch checks their version counters. CUDA events order the caller
    and workspace streams for both passes.

    """
    return ProjectionFunction.apply(volume, pose, workspace, validate)


def project_spectral_signal(
    fields: Any,
    pose: Any,
    coefficients: Any,
    weights: Any,
    response: Any,
    *,
    material_workspace: MaterialProjectionWorkspace,
    spectral_workspace: SpectralWorkspace,
    validate: bool = True,
) -> Any:
    """Compute ``lambda_p = sum_k w_kp R_kp exp(-sum_m C_mk A_mp)``.

    ``fields`` has contiguous FP32 shape ``(M,nz,ny,nx)`` or material-major
    flat storage. The prepared field domain selects volume fractions or
    nonnegative basis fields. Paths ``A_mp`` are in mm; FP32 coefficients
    have shape ``(M,K)`` and units mm^-1. FP32 weights and response each
    have shape ``(K,)`` when shared, or ``(K,P)`` for pixel factors.
    ``pose`` has FP64 shape ``(12,)``, packed rotation then translation
    in mm. All five tensors share the prepared workspaces' device.

    The output owns Torch storage and has detector shape or flat shape
    ``(P,)``. Precision follows ``spectral_workspace.spec.precision``;
    ``output_domain`` selects ``lambda`` or its natural logarithm.
    Linear units are the product of weight and response units. Set the
    corresponding active flags for every trainable input. Field and
    pose derivatives also need active spectral paths and prepared path
    cotangents. Backward recomputes from the five saved tensors, whose
    version counters Torch checks. CUDA events join the owning streams.

    """
    return SpectralProjectionFunction.apply(
        fields,
        pose,
        coefficients,
        weights,
        response,
        material_workspace,
        spectral_workspace,
        validate,
    )
