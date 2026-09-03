"""Estimate ``mu = E[X(theta)]`` and apply ``J_hat_B.T @ c`` in Torch.

A forward averages every original history in batch A, including
misses and absorbed photons. Backward traces an independent batch B
to estimate the Jacobian of the expected measurement. Its score
combines likelihood terms with direct weight derivatives. For a
loss ``L(mu_hat_A)``, the two batches form the independent product
``J_hat_B.T @ grad L(mu_hat_A)``; its finite-sample expectation is
``J.T @ E[grad L(mu_hat_A)]``. A squared-loss gradient is affine in
the measurement, so its expected product equals the ideal gradient.

Trainable coordinates are dimensionless log voxel density, log
material density and direct source amplitude. The source distribution,
geometry and coefficient tables are fixed. Each outstanding forward
owns its physical-density snapshot and native workspaces; Torch owns
the output and uses version counters for the original input tensors.

"""

from __future__ import annotations

# Public Python boundaries retain runtime guards even with typed declarations;
# private snapshot buffers are shared by helpers within this module only.
# pyright: reportUnnecessaryIsInstance=false, reportPrivateUsage=false
import importlib
import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import torch

from dpt._runtime import require_no_tape
from dpt.contracts import ContractError
from dpt.interop._torch_streams import stream_bridge
from dpt.interop.torch import _pack_cotangent
from dpt.transport.derivatives import (
    TRANSPORT_VJP_BYTES_PER_PARAMETER,
    TransportParameter,
    TransportVJPWorkspace,
    prepare_transport_vjp,
    transport_vjp,
)
from dpt.transport.estimators import history_mean, prepare_estimators
from dpt.transport.forward import TransportWorkspace, prepare_transport, trace_histories
from dpt.transport.model import TransportSpec
from dpt.transport.rng import HistoryBatch, require_independent
from dpt.transport.source import (
    ParallelBeam,
    PointConeSource,
    sample_parallel_beam,
    sample_point_cone,
)
from dpt.transport.voxel_derivatives import (
    VoxelDensityVJPWorkspace,
    prepare_voxel_density_vjp,
    voxel_density_vjp,
)

__all__ = [
    "ExpectedTransport",
    "ExpectedTransportEstimate",
    "ExpectedTransportEstimator",
    "ExpectedTransportFunction",
    "ExpectedTransportWorkspace",
    "TransportAutogradSpec",
    "estimate_expected_transport",
    "expected_transport_signal",
    "prepare_expected_transport",
]


@dataclass(frozen=True, slots=True)
class TransportAutogradSpec:
    """Named score coordinates and selected entries in otherwise fixed charts.

    ``None`` selects every voxel/material; an empty tuple fixes the whole chart.
    Unselected tensor entries receive zero gradients. Voxel IDs use x-fastest
    flattening. Source amplitude multiplies the measurement, never its sampling
    law, and its native derivative remains valid at zero amplitude.
    """

    voxel_parameterization: Literal["log-voxel-density"] = "log-voxel-density"
    material_parameterization: Literal["log-material-density"] = "log-material-density"
    source_parameterization: Literal["source-amplitude"] = "source-amplitude"
    active_voxels: tuple[int, ...] | None = None
    active_materials: tuple[int, ...] | None = None
    active_source_amplitude: bool = True

    def __post_init__(self) -> None:
        if (
            self.voxel_parameterization != "log-voxel-density"
            or self.material_parameterization != "log-material-density"
            or self.source_parameterization != "source-amplitude"
        ):
            raise ContractError("transport autograd requires its named supported score coordinates")
        for name in ("active_voxels", "active_materials"):
            values = getattr(self, name)
            if values is not None:
                values = tuple(values)
                if any(type(value) is not int or value < 0 for value in values):
                    raise ContractError(f"{name} must contain nonnegative integer IDs")
                if len(set(values)) != len(values):
                    raise ContractError(f"{name} must contain distinct IDs")
                object.__setattr__(self, name, values)
        if type(self.active_source_amplitude) is not bool:
            raise ContractError("active_source_amplitude must be a boolean")


@dataclass(frozen=True, slots=True)
class ExpectedTransportEstimator:
    """Explicit forward/replay identities and the upstream cotangent's origin.

    ``forward-batch`` declares c(A), the normal composed-loss case. ``deterministic``
    declares a fixed cotangent independent of all histories. ``external-batch``
    requires the actual external batch, including its source-draw identity.
    Preparation binds the actual library sampler tokens before testing independence.
    Batch/source declarations are mathematical contracts, not empirical guarantees.
    """

    forward_batch: HistoryBatch
    derivative_batch: HistoryBatch
    cotangent_origin: Literal["forward-batch", "deterministic", "external-batch"] = "forward-batch"
    external_cotangent_batch: HistoryBatch | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.forward_batch, HistoryBatch) or not isinstance(
            self.derivative_batch, HistoryBatch
        ):
            raise ContractError("expected transport requires explicit HistoryBatch identities")
        if self.cotangent_origin not in ("forward-batch", "deterministic", "external-batch"):
            raise ContractError("declare the expected-score cotangent origin")
        if self.cotangent_origin == "external-batch":
            if not isinstance(self.external_cotangent_batch, HistoryBatch):
                raise ContractError("external-batch cotangents require their HistoryBatch identity")
        elif self.external_cotangent_batch is not None:
            raise ContractError(
                "external_cotangent_batch requires cotangent_origin='external-batch'"
            )

    @property
    def cotangent_batch(self) -> HistoryBatch | None:
        if self.cotangent_origin == "forward-batch":
            return self.forward_batch
        return self.external_cotangent_batch

    def bind_source(self, source: ParallelBeam | PointConeSource) -> ExpectedTransportEstimator:
        """Attach actual library source counters without weakening structural laws."""
        if isinstance(source, ParallelBeam):
            sampler = "parallel-rectangle-v1"
        elif isinstance(source, PointConeSource):
            sampler = "point-cone-area-v1"
        else:
            raise ContractError("expected transport requires a fixed library source model")

        def bind(batch: HistoryBatch) -> HistoryBatch:
            if batch.source_identity is not None and batch.source_identity.sampler != sampler:
                raise ContractError("HistoryBatch source token disagrees with the actual sampler")
            return batch.with_library_source(sampler)

        result = replace(
            self,
            forward_batch=bind(self.forward_batch),
            derivative_batch=bind(self.derivative_batch),
        )
        require_independent(result.derivative_batch, result.forward_batch)
        if result.cotangent_batch is not None:
            require_independent(result.derivative_batch, result.cotangent_batch)
        return result


@dataclass(frozen=True, slots=True)
class ExpectedTransportWorkspace:
    """Owned fixed physics, source law, coordinates and per-forward storage budget."""

    source: ParallelBeam | PointConeSource
    parameters: TransportAutogradSpec
    slots_per_history: int
    max_snapshot_bytes: int
    _template: TransportWorkspace = field(repr=False)
    _active: Any = field(repr=False)
    _fixed_bytes: int = field(repr=False)

    @property
    def context(self) -> Any:
        return self._template.context

    @property
    def transport_spec(self) -> TransportSpec:
        return self._template.spec

    @property
    def max_histories(self) -> int:
        return self._template.max_histories


@dataclass(frozen=True, slots=True)
class ExpectedTransportEstimate:
    """Owned signal Tensor with immutable forward, score and source identities.

    The FP64 image is normalised by forward_batch.count, including misses. Its
    backward score is normalised independently by derivative_batch.count. Units
    are the TransportSpec score per launched original history; no dose/exposure
    or additional deterministic primary contribution is inferred.
    """

    signal: Any
    estimator: ExpectedTransportEstimator
    parameters: TransportAutogradSpec
    transport_spec: TransportSpec


@dataclass(slots=True)
class _Snapshot:
    transport: TransportWorkspace
    estimator: ExpectedTransportEstimator
    voxel_density: Any
    material_density: Any
    amplitude: Any
    amplitude_value: float
    derivative_source: tuple[Any, Any, Any] | None
    voxel_vjp: VoxelDensityVJPWorkspace | None
    groups: tuple[TransportVJPWorkspace, ...]


def prepare_expected_transport(
    workspace: TransportWorkspace,
    *,
    source: ParallelBeam | PointConeSource,
    parameters: TransportAutogradSpec | None = None,
    slots_per_history: int = 64,
    max_snapshot_bytes: int = 256 * 1024 * 1024,
) -> ExpectedTransportWorkspace:
    """Prepare score coordinates ``rho_v = exp(u_v)`` and ``rho_m = exp(u_m)``.

    The native workspace supplies fixed material IDs, absorption and
    scattering tables in mm^-1, detector geometry and history capacity.
    ``source`` fixes a ``ParallelBeam`` or ``PointConeSource`` distribution.
    ``parameters`` selects active voxel/material IDs and direct amplitude;
    voxel IDs flatten ``(nz,ny,nx)`` with x fastest. Density multipliers
    are dimensionless and must exponentiate to finite positive values.

    Preparation copies the fixed tables on the owning device and stream.
    Each later forward owns a density snapshot, history buffers and score
    scratch. ``slots_per_history`` is a positive power-of-two voxel score
    capacity. ``max_snapshot_bytes`` bounds planned buffer storage,
    including the fixed buffers. The returned workspace is reused across
    independent forward nodes with different density tensors and batches.

    """
    require_no_tape()
    if not isinstance(workspace, TransportWorkspace):
        raise ContractError("workspace must be a prepared TransportWorkspace")
    if not isinstance(source, ParallelBeam | PointConeSource):
        raise ContractError("source must be a fixed ParallelBeam or PointConeSource")
    if parameters is None:
        parameters = TransportAutogradSpec()
    if not isinstance(parameters, TransportAutogradSpec):
        raise ContractError("parameters must be a TransportAutogradSpec")
    if (
        type(slots_per_history) is not int
        or slots_per_history <= 0
        or slots_per_history & (slots_per_history - 1)
    ):
        raise ContractError("slots_per_history must be a positive power of two")
    if type(max_snapshot_bytes) is not int or max_snapshot_bytes <= 0:
        raise ContractError("max_snapshot_bytes must be a positive integer")
    voxels, materials = math.prod(workspace.spec.grid.shape), workspace.spec.grid.materials
    for name, extent in (("active_voxels", voxels), ("active_materials", materials)):
        selection = getattr(parameters, name)
        if selection is not None and any(index >= extent for index in selection):
            raise ContractError(f"{name} contains an ID outside the prepared model")
    fixed_bytes = 8 * voxels + 16 * materials * len(workspace.spec.energy_nodes)
    fixed_bytes += workspace.scratch_bytes
    if fixed_bytes > max_snapshot_bytes:
        raise ContractError("fixed transport buffers exceed max_snapshot_bytes")
    workspace.check_status()
    context, wp = workspace.context, workspace.context.wp
    selected = set(parameters.active_voxels or ())
    with context.scope():
        material_ids = wp.clone(workspace.material_ids)
        absorption = wp.clone(workspace.absorption)
        scattering = wp.clone(workspace.scattering)
        if parameters.active_voxels is None:
            active = wp.ones(voxels, dtype=wp.int32, device=context.device)
        elif not selected:
            active = wp.zeros(voxels, dtype=wp.int32, device=context.device)
        else:
            active = wp.array(
                [int(index in selected) for index in range(voxels)],
                dtype=wp.int32,
                device=context.device,
            )
    template = prepare_transport(
        workspace.spec,
        material_ids=material_ids,
        absorption=absorption,
        scattering=scattering,
        max_histories=workspace.max_histories,
        device=str(context.device),
        stream=context.stream,
    )
    return ExpectedTransportWorkspace(
        source, parameters, slots_per_history, max_snapshot_bytes, template, active, fixed_bytes
    )


def _group_parameters(
    workspace: ExpectedTransportWorkspace, needs: tuple[bool, ...]
) -> tuple[tuple[TransportParameter, ...], ...]:
    selected = workspace.parameters.active_materials
    material_ids = range(workspace.transport_spec.grid.materials) if selected is None else selected
    parameters = (
        [TransportParameter("log-material-density", index) for index in material_ids]
        if needs[1]
        else []
    )
    if needs[2]:
        parameters.append(TransportParameter("source-amplitude"))
    return tuple(tuple(parameters[start : start + 4]) for start in range(0, len(parameters), 4))


def _planned_bytes(
    workspace: ExpectedTransportWorkspace,
    estimator: ExpectedTransportEstimator,
    needs: tuple[bool, ...],
    groups: tuple[tuple[TransportParameter, ...], ...],
) -> int:
    spec = workspace.transport_spec
    voxels, materials, pixels = (
        math.prod(spec.grid.shape),
        spec.grid.materials,
        spec.detector.pixels,
    )
    forward, derivative = estimator.forward_batch.count, estimator.derivative_batch.count
    capacity = max(forward, derivative)
    # Sources, forward sparse histories/moments, physical snapshots and owned
    # Tensor outputs/gradients. Include conservative temporary parameter outputs.
    required = workspace._fixed_bytes + 84 * forward + (56 * derivative if any(needs) else 0)
    required += 28 * pixels + 16 * voxels + 32 * materials + 32 + 4 + 8 * len(spec.energy_nodes)
    if needs[0]:
        records = capacity * workspace.slots_per_history
        if records >= 2**31:
            raise ContractError("sparse score storage exceeds signed 32-bit indexing")
        required += 20 * records + 28 * voxels + 4
    for group in groups:
        required += TRANSPORT_VJP_BYTES_PER_PARAMETER * len(group)
    if required > workspace.max_snapshot_bytes:
        raise ContractError(
            f"expected-score snapshot needs {required} buffer bytes, exceeding max_snapshot_bytes"
        )
    return required


def _sample_source(
    workspace: ExpectedTransportWorkspace, transport: TransportWorkspace, batch: HistoryBatch
) -> tuple[Any, Any, Any]:
    context, wp = transport.context, transport.context.wp
    with context.scope():
        positions = wp.empty(batch.count, dtype=wp.vec3d, device=context.device)
        directions = wp.empty(batch.count, dtype=wp.vec3d, device=context.device)
        weights = wp.empty(batch.count, dtype=wp.float64, device=context.device)
    source = workspace.source
    if isinstance(source, ParallelBeam):
        token = sample_parallel_beam(
            source,
            batch=batch,
            workspace=transport,
            out_position=positions,
            out_direction=directions,
            out_weight=weights,
            stream=context.stream,
        )
    else:
        token = sample_point_cone(
            source,
            batch=batch,
            workspace=transport,
            out_position=positions,
            out_direction=directions,
            out_weight=weights,
            stream=context.stream,
        )
    if token != batch.source_identity:
        raise ContractError("actual source sampler identity differs from the estimator declaration")
    return positions, directions, weights


def _check_tensors(
    torch: Any, values: tuple[Any, ...], workspace: ExpectedTransportWorkspace
) -> None:
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise ContractError("transport score coordinates and amplitude must be Torch tensors")
    spec, device = workspace.transport_spec, values[0].device
    if device.type not in ("cpu", "cuda") or any(value.device != device for value in values):
        raise ContractError("transport tensors must share one explicit CPU or CUDA device")
    if workspace.context.wp.device_from_torch(device) != workspace.context.device:
        raise ContractError("Torch tensors and expected transport workspace must share a device")
    if any(value.dtype != torch.float64 or not value.is_contiguous() for value in values):
        raise ContractError("transport score tensors must be contiguous FP64")
    if tuple(values[0].shape) not in ((math.prod(spec.grid.shape),), spec.grid.shape):
        raise ContractError(
            "log_voxel_density must have the grid shape or its flattened x-fastest shape"
        )
    if tuple(values[1].shape) != (spec.grid.materials,):
        raise ContractError("log_material_density must have one entry per material")
    if tuple(values[2].shape) not in ((), (1,)):
        raise ContractError("source_amplitude must be a scalar or one-entry tensor")
    for value, selected, name in (
        (values[0], workspace.parameters.active_voxels, "log_voxel_density"),
        (values[1], workspace.parameters.active_materials, "log_material_density"),
    ):
        if value.requires_grad and selected == ():
            raise ContractError(f"{name} is fixed; select active entries before autograd")
    if values[2].requires_grad and not workspace.parameters.active_source_amplitude:
        raise ContractError("source_amplitude is fixed; activate its score before autograd")


class ExpectedTransportFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        owner: Any,
        log_voxel_density: Any,
        log_material_density: Any,
        source_amplitude: Any,
        workspace: ExpectedTransportWorkspace,
        estimator: ExpectedTransportEstimator,
    ) -> Any:
        require_no_tape()
        if not isinstance(workspace, ExpectedTransportWorkspace) or not isinstance(
            estimator, ExpectedTransportEstimator
        ):
            raise ContractError("expected transport requires its typed workspace and estimator")
        estimator = estimator.bind_source(workspace.source)
        if (
            max(estimator.forward_batch.count, estimator.derivative_batch.count)
            > workspace.max_histories
        ):
            raise ContractError("estimator history batch exceeds expected transport capacity")
        values = (log_voxel_density, log_material_density, source_amplitude)
        _check_tensors(torch, values, workspace)
        needs = tuple(bool(value.requires_grad) for value in values)
        groups = _group_parameters(workspace, needs)
        _planned_bytes(workspace, estimator, needs, groups)
        context, wp = workspace.context, workspace.context.wp
        with torch.no_grad(), stream_bridge(torch, values, workspace) as bridge:
            workspace._template.check_status()
            densities = tuple(torch.exp(value.detach()) for value in values[:2])
            if any(
                not bool(torch.isfinite(value).all().item()) or not bool((value > 0).all().item())
                for value in densities
            ):
                raise ContractError(
                    "exp(log density) must remain finite and positive; "
                    "overflow/underflow is invalid"
                )
            amplitude = source_amplitude.detach().clone()
            amplitude_value = float(amplitude.item())
            if not math.isfinite(amplitude_value) or amplitude_value < 0:
                raise ContractError("source amplitude must be finite and nonnegative")
            voxel, material = densities[0].reshape(-1), densities[1]
            bridge.record(voxel, material, amplitude)
            template = workspace._template
            transport = prepare_transport(
                template.spec,
                material_ids=template.material_ids,
                absorption=template.absorption,
                scattering=template.scattering,
                voxel_density=wp.from_torch(voxel, requires_grad=False),
                max_histories=max(estimator.forward_batch.count, estimator.derivative_batch.count),
                device=str(context.device),
                stream=context.stream,
            )
            forward_source = _sample_source(workspace, transport, estimator.forward_batch)
            derivative_source = (
                _sample_source(workspace, transport, estimator.derivative_batch)
                if any(needs)
                else None
            )
            voxel_workspace = (
                prepare_voxel_density_vjp(
                    transport,
                    active=workspace._active,
                    slots_per_history=workspace.slots_per_history,
                    max_bytes=workspace.max_snapshot_bytes,
                )
                if needs[0]
                else None
            )
            material_workspaces = tuple(
                prepare_transport_vjp(
                    transport, parameters=group, max_bytes=workspace.max_snapshot_bytes
                )
                for group in groups
            )
            count = estimator.forward_batch.count
            with context.scope():
                outputs = {
                    name: wp.empty(count, dtype=dtype, device=context.device)
                    for name, dtype in (
                        ("out_pixel", wp.int32),
                        ("out_score", wp.float64),
                        ("out_energy", wp.float64),
                        ("out_events", wp.int32),
                        ("out_status", wp.int32),
                    )
                }
            result = torch.empty(
                template.spec.detector.image_shape,
                dtype=torch.float64,
                device=values[0].device,
            )
            bridge.record(result)
            trace_histories(
                *forward_source,
                wp.from_torch(material, requires_grad=False),
                batch=estimator.forward_batch,
                workspace=transport,
                source_amplitude=amplitude_value,
                stream=context.stream,
                **outputs,
            )
            history_mean(
                outputs["out_pixel"],
                outputs["out_score"],
                outputs["out_status"],
                batch=estimator.forward_batch,
                workspace=prepare_estimators(transport),
                out_mean=wp.from_torch(result.reshape(-1), requires_grad=False),
                stream=context.stream,
            )
        owner.save_for_backward(*values)
        owner.workspace = workspace
        owner.snapshot = _Snapshot(
            transport,
            estimator,
            voxel,
            material,
            amplitude,
            amplitude_value,
            derivative_source,
            voxel_workspace,
            material_workspaces,
        )
        return result

    @staticmethod
    def backward(owner: Any, cotangent: Any) -> tuple[Any, ...]:
        if torch.is_grad_enabled():
            raise ContractError("expected transport supports first-order score derivatives only")
        require_no_tape()
        # Access originals first to preserve standard version-counter errors.
        values = owner.saved_tensors
        snapshot: _Snapshot = owner.snapshot
        workspace: ExpectedTransportWorkspace = owner.workspace
        transport, context = snapshot.transport, snapshot.transport.context
        wp = context.wp
        with (
            torch.no_grad(),
            stream_bridge(torch, (*values, cotangent), workspace) as bridge,
        ):
            transport.check_status()
            if snapshot.voxel_vjp is not None:
                snapshot.voxel_vjp.check_status()
            if (
                cotangent.dtype != torch.float64
                or tuple(cotangent.shape) != transport.spec.detector.image_shape
            ):
                raise ContractError("expected-signal cotangent must match the FP64 detector image")
            packed = _pack_cotangent(torch, cotangent)
            if not bool(torch.isfinite(packed).all().item()):
                raise ContractError("expected-signal cotangent must be finite")
            bridge.record(
                packed,
                snapshot.voxel_density,
                snapshot.material_density,
                snapshot.amplitude,
            )
            gradients = tuple(
                torch.zeros_like(value, memory_format=torch.contiguous_format)
                if owner.needs_input_grad[index]
                else None
                for index, value in enumerate(values)
            )
            bridge.record(*(gradient for gradient in gradients if gradient is not None))
            seed = wp.from_torch(packed, requires_grad=False)
            material = wp.from_torch(snapshot.material_density, requires_grad=False)
            derivative_source = snapshot.derivative_source
            if derivative_source is None:
                raise ContractError("missing owned derivative source snapshot")
            if gradients[0] is not None:
                if snapshot.voxel_vjp is None:
                    raise ContractError("missing active voxel score workspace")
                voxel_density_vjp(
                    *derivative_source,
                    material,
                    seed,
                    batch=snapshot.estimator.derivative_batch,
                    workspace=snapshot.voxel_vjp,
                    out_gradient=wp.from_torch(gradients[0].reshape(-1), requires_grad=False),
                    cotangent_batch=snapshot.estimator.cotangent_batch,
                    source_amplitude=snapshot.amplitude_value,
                    stream=context.stream,
                )
            for group in snapshot.groups:
                output = torch.empty(
                    len(group.parameters), dtype=torch.float64, device=values[0].device
                )
                bridge.record(output)
                transport_vjp(
                    *derivative_source,
                    material,
                    seed,
                    batch=snapshot.estimator.derivative_batch,
                    workspace=group,
                    out_gradient=wp.from_torch(output, requires_grad=False),
                    cotangent_batch=snapshot.estimator.cotangent_batch,
                    source_amplitude=snapshot.amplitude_value,
                    stream=context.stream,
                )
                for column, parameter in enumerate(group.parameters):
                    if parameter.material is not None:
                        if gradients[1] is None:
                            raise ContractError("missing active material gradient destination")
                        gradients[1][parameter.material] = output[column]
                    else:
                        if gradients[2] is None:
                            raise ContractError("missing active amplitude gradient destination")
                        gradients[2].reshape(-1)[0] = output[column]
        return (*gradients, None, None)


class ExpectedTransport(torch.nn.Module):
    """Registered trainable score coordinates with a fixed native/source model."""

    def __init__(
        self,
        log_voxel_density: Any,
        log_material_density: Any,
        source_amplitude: Any,
        *,
        workspace: ExpectedTransportWorkspace,
        estimator: ExpectedTransportEstimator,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.workspace, self.estimator = workspace, estimator
        for name, value in (
            ("log_voxel_density", log_voxel_density),
            ("log_material_density", log_material_density),
            ("source_amplitude", source_amplitude),
        ):
            if isinstance(value, torch.nn.Parameter):
                self.register_parameter(name, value)
            elif isinstance(value, torch.Tensor):
                self.register_buffer(name, value)
            else:
                raise ContractError(f"{name} must be a Torch tensor")

    def forward(
        self, *, estimator: ExpectedTransportEstimator | None = None
    ) -> ExpectedTransportEstimate:
        return estimate_expected_transport(
            self.log_voxel_density,
            self.log_material_density,
            self.source_amplitude,
            workspace=self.workspace,
            estimator=self.estimator if estimator is None else estimator,
        )


def estimate_expected_transport(
    log_voxel_density: Any,
    log_material_density: Any,
    source_amplitude: Any,
    *,
    workspace: ExpectedTransportWorkspace,
    estimator: ExpectedTransportEstimator,
) -> ExpectedTransportEstimate:
    """Return ``mu_hat_A = sum_i X_i(theta)/N_A`` with a score-based backward.

    All inputs are contiguous FP64 Torch tensors on the prepared device.
    Log voxel density has shape ``(nz,ny,nx)`` or flat x-fastest ``(V,)``;
    log material density has shape ``(M,)``. Direct source amplitude is a
    nonnegative scalar or ``(1,)``. Densities are dimensionless multipliers
    of the fixed coefficient model. Amplitude multiplies the score while
    leaving its sampling law fixed, including at zero amplitude.

    ``estimator`` supplies independent forward and derivative history
    identities. Each original history remains in its batch denominator.
    The returned ``ExpectedTransportEstimate`` contains an owned FP64
    detector-shaped ``signal`` and the resolved estimator/model metadata.
    Its units are the source weight times amplitude, with an additional
    keV factor for energy scoring. Backward estimates ``J_B.T @ c`` for
    the upstream cotangent and writes gradients for selected active
    coordinates. Geometry, source law and coefficient tables stay fixed.

    """
    if not isinstance(workspace, ExpectedTransportWorkspace) or not isinstance(
        estimator, ExpectedTransportEstimator
    ):
        raise ContractError("expected transport requires its typed workspace and estimator")
    resolved = estimator.bind_source(workspace.source)
    torch = importlib.import_module("torch")
    values = (log_voxel_density, log_material_density, source_amplitude)
    if not torch.is_grad_enabled():
        values = tuple(
            value.detach() if isinstance(value, torch.Tensor) else value for value in values
        )
    signal = ExpectedTransportFunction.apply(*values, workspace, resolved)
    return ExpectedTransportEstimate(
        signal, resolved, workspace.parameters, workspace.transport_spec
    )


def expected_transport_signal(
    log_voxel_density: Any,
    log_material_density: Any,
    source_amplitude: Any,
    *,
    workspace: ExpectedTransportWorkspace,
    estimator: ExpectedTransportEstimator,
) -> Any:
    """Return the FP64 tensor ``mu_hat_A`` from :func:`estimate_expected_transport`.

    Inputs use its same contiguous FP64 layout: dimensionless log voxel
    density ``(nz,ny,nx)`` or ``(V,)``, log material density ``(M,)``, and
    scalar or ``(1,)`` direct amplitude. The result owns detector-shaped
    Torch storage on the workspace device, in source-score units or
    source-score times keV. The explicit independent derivative batch
    supplies the first-order expected-score VJP. Use this tensor form
    directly in a Torch loss; use ``estimate_expected_transport`` to keep
    the resolved history and parameter records alongside the signal.

    """
    return estimate_expected_transport(
        log_voxel_density,
        log_material_density,
        source_amplitude,
        workspace=workspace,
        estimator=estimator,
    ).signal
