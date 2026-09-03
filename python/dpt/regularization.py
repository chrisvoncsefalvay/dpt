"""Resident TV/L1 penalties and composite proximal steps with fixed support.

R(x)=lambda_tv sum_v phi(Dx_v)+lambda_l1 sum_v |x_v|+I_support(x).
D uses forward differences divided by physical grid spacing. At the outer grid
boundary its outward difference is zero. Excluded support samples are fixed at
zero; differences across internal support boundaries remain part of the TV.
"""

# Public Python boundaries validate runtime inputs.
# pyright: reportUnnecessaryIsInstance=false
# Workspace internals are used by the composition functions in this module.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._reductions import prepare_reduction
from dpt._runtime import (
    CompletionReadback,
    StatusCheckpoint,
    StatusReadback,
    load_kernels,
    prepare_context,
    require_no_tape,
)
from dpt.contracts import ContractError, NumericalError, finite_scalar, integer, raise_device_status
from dpt.volumes import GridSpec

if TYPE_CHECKING:
    import warp as wp


@dataclass(frozen=True, slots=True)
class RegularizationSpec:
    """R=sum_v lambda_tv*phi(Dx_v)+lambda_l1*|x_v| with no implicit voxel measure.

    Isotropic TV uses phi(d)=sqrt(dx²+dy²+dz²); anisotropic TV uses
    phi(d)=|dx|+|dy|+|dz| along the declared local grid axes. Both use the
    physical x/y/z spacing in mm. At zero, the reported subgradient is zero.
    Input, gradient and final proximal-output precision is declared explicitly;
    internal primal/dual arithmetic and reductions use FP64.
    """

    tv_weight: float = 0.0
    l1_weight: float = 0.0
    tv_norm: Literal["isotropic", "anisotropic"] = "isotropic"
    precision: Literal["float32", "float64"] = "float64"
    boundary: Literal["neumann"] = "neumann"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tv_weight", finite_scalar(self.tv_weight, "tv_weight", minimum=0.0)
        )
        object.__setattr__(
            self, "l1_weight", finite_scalar(self.l1_weight, "l1_weight", minimum=0.0)
        )
        if self.tv_norm not in ("isotropic", "anisotropic") or self.precision not in (
            "float32",
            "float64",
        ):
            raise ContractError("declare isotropic/anisotropic TV and float32/float64 storage")
        if self.boundary != "neumann":
            raise ContractError("TV outer-grid boundary must use zero outward differences")


@dataclass(frozen=True, slots=True)
class RegularizationProxPolicy:
    """Stop when P-D <= absolute_gap+relative_gap*max(|P|,|D|)."""

    max_iterations: int = 2000
    check_interval: int = 25
    absolute_gap: float = 1e-10
    relative_gap: float = 1e-8

    def __post_init__(self) -> None:
        integer(self.max_iterations, "max_iterations", minimum=1)
        integer(self.check_interval, "check_interval", minimum=1)
        object.__setattr__(
            self, "absolute_gap", finite_scalar(self.absolute_gap, "absolute_gap", minimum=0.0)
        )
        object.__setattr__(
            self, "relative_gap", finite_scalar(self.relative_gap, "relative_gap", minimum=0.0)
        )


@dataclass(frozen=True, slots=True)
class RegularizationProxResult:
    """Certificate for min_x 1/2||x-y||²+step*R(x) on the declared support.

    primal/dual/gap certify the FP64 iterate before its one final cast to output
    storage. Float32 output adds its explicit final rounding. iteration_budget
    returns a feasible iterate and certificate without claiming convergence.
    """

    iterations: int
    reason: Literal["closed_form", "primal_dual_gap", "iteration_budget"]
    primal: float
    dual: float
    gap: float

    @property
    def converged(self) -> bool:
        return self.reason in ("closed_form", "primal_dual_gap")


@dataclass(slots=True)
class RegularizationWorkspace:
    """Prepared O(V) buffers on one device/stream, with an owned support copy.

    Public fields grid/spec are immutable descriptors. Calls reject concurrent,
    recursive and ambient-tape execution. Caller input stays unchanged and
    caller outputs must be disjoint from it and from private scratch.
    """

    grid: GridSpec
    spec: RegularizationSpec
    context: Any
    _kernels: Any = field(repr=False)
    _mask: Any = field(repr=False)
    _source: Any = field(repr=False)
    _x: Any = field(repr=False)
    _following: Any = field(repr=False)
    _extrapolated: Any = field(repr=False)
    _dual: Any = field(repr=False)
    _contributions: Any = field(repr=False)
    _control: Any = field(repr=False)
    _status: Any = field(repr=False)
    _reduction: Any = field(repr=False)
    _inverse: Any = field(repr=False)
    _iteration_step: float = field(repr=False)
    _status_readback: Any = field(init=False, repr=False)
    _checkpoint: Any = field(init=False, repr=False)
    _completion: Any = field(init=False, repr=False)
    _lock: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._status_readback = StatusReadback(self.context, self._status)
        self._checkpoint = StatusCheckpoint(self.context, (self,))
        self._completion = CompletionReadback(self._checkpoint, (self._control,))
        self._lock = Lock()

    @property
    def dtype(self) -> Any:
        return getattr(self.context.wp, self.spec.precision)

    @property
    def scratch_bytes(self) -> int:
        """Owned device buffers, including support, reductions and completion."""
        return (
            self.grid.voxels * (1 + 4 * 8 + 3 * 8 + 2 * 8)
            + 16
            + 4
            + self._reduction.scratch_bytes
            + self._checkpoint.device_bytes
            + self._completion.device_bytes
        )

    @property
    def host_bytes(self) -> int:
        """Prepared scalar certificate/status staging; no field staging."""
        return (
            self._status_readback.host_bytes
            + self._checkpoint.host_bytes
            + self._completion.host_bytes
        )

    def _raise_status(self, code: int) -> None:
        raise_device_status(
            code,
            "regularization",
            messages={
                1: "fields must be finite and support masks binary",
                8: "penalty evaluation requires zero values outside fixed support",
            },
        )

    def check_status(self) -> None:
        self._raise_status(self._status_readback.read())

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()

    @contextmanager
    def _exclusive(self) -> Generator[None, None, None]:
        require_no_tape()
        if not self._lock.acquire(blocking=False):
            raise ContractError("a regularization workspace cannot be shared concurrently")
        completed = False
        try:
            with self.context.scope():
                self._checkpoint.consume()
                yield
            completed = True
        finally:
            try:
                if not completed and self.context.device.is_cuda:
                    self.context.wp.synchronize_stream(self.context.stream)
            finally:
                self._lock.release()

    def _arrays(self, source: Any, output: Any, *, stream: wp.Stream | None = None) -> None:
        ctx = self.context
        ctx.assert_stream(stream)
        ctx.array(source, "field", dtype=self.dtype, shape=(self.grid.voxels,))
        ctx.array(output, "output", dtype=self.dtype, shape=(self.grid.voxels,))
        private = (
            *self._reduction.partials,
            self._checkpoint._device,
            self._completion._payload,
            self._status_readback._host,
            self._checkpoint._readback._host,
            self._completion._readback._host,
            self._mask,
            self._source,
            self._x,
            self._following,
            self._extrapolated,
            self._dual,
            self._contributions,
            self._control,
            self._status,
        )
        ctx.disjoint(
            [("field", source), *(("private scratch", a) for a in private)], [("output", output)]
        )
        ctx.disjoint([("field", source)], [("private scratch", a) for a in private])

    def _initialise(self, source: Any, l1: float, *, enforce: bool) -> None:
        ctx, wp = self.context, self.context.wp
        wp.launch(
            self._kernels.initialise_kernel(self.spec.precision == "float64"),
            dim=self.grid.voxels,
            inputs=[source, self._mask, l1, enforce],
            outputs=[self._source, self._x, self._extrapolated, self._status],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
        # Input failures are delivered before a potentially long proximal loop.
        self._checkpoint.consume()

    def _read_certificate(self, current: Any, tv: float, l1: float) -> tuple[float, float, float]:
        ctx, wp = self.context, self.context.wp
        nz, ny, nx = self.grid.shape
        wp.launch(
            self._kernels.certificate_kernel(self.spec.tv_norm == "isotropic"),
            dim=self.grid.voxels,
            inputs=[
                current,
                self._source,
                self._mask,
                self._dual,
                nx,
                ny,
                nz,
                self._inverse,
                tv,
                l1,
            ],
            outputs=[self._contributions, self._status],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
        self._reduction.finish(
            self.grid.voxels, self._control, self._status, source=self._contributions, components=2
        )
        (controls,) = self._completion.consume()
        primal, dual = float(controls[0]), float(controls[1])
        gap = primal - dual
        tolerance = 128 * math.ulp(1.0) * max(abs(primal), abs(dual))
        if not math.isfinite(gap) or gap < -tolerance:
            raise NumericalError("invalid primal-dual regularization certificate")
        return primal, dual, max(0.0, gap)


def prepare_regularization(
    grid: GridSpec,
    spec: RegularizationSpec,
    *,
    support: wp.array[wp.uint8] | None = None,
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
    max_bytes: int = 256 << 20,
) -> RegularizationWorkspace:
    """Prepare R(x) on grid.shape=(nz,ny,nx), with physical spacing=(x,y,z).

    support is a device uint8[grid.voxels] binary mask, copied and validated once.
    None enables every voxel. Owned buffer memory is bounded before allocation.
    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_support: Any = cast(Any, support)
    require_no_tape()
    if not isinstance(grid, GridSpec) or not isinstance(spec, RegularizationSpec):
        raise ContractError("regularization requires validated GridSpec and RegularizationSpec")
    integer(max_bytes, "max_bytes", minimum=1, maximum=2**63 - 1)
    if grid.voxels > (2**31 - 1) // 2:
        raise ContractError("regularization packed reductions exceed signed 32-bit indexing")
    if grid.voxels * 128 + 8192 > max_bytes:
        raise ContractError("regularization workspace exceeds max_bytes")
    inverse = tuple(1 / h for h in grid.spacing_mm)
    if not all(map(math.isfinite, inverse)):
        raise ContractError("regularization grid spacing has unrepresentable inverse")
    operator_bound = 2 * math.hypot(*inverse)
    iteration_step = 1.0
    if spec.tv_weight > 0:
        if not math.isfinite(operator_bound) or operator_bound <= 0:
            raise ContractError("regularization difference norm has unrepresentable bound")
        iteration_step = 0.99 / operator_bound
        if not math.isfinite(iteration_step) or iteration_step == 0:
            raise ContractError("regularization difference step is unrepresentable")
    ctx = prepare_context(device=device, stream=stream)
    wp = ctx.wp
    kernels = load_kernels("dpt.kernels.regularization")
    if _api_support is not None:
        ctx.array(_api_support, "support", dtype=wp.uint8, shape=(grid.voxels,))
    reduction = prepare_reduction(ctx, grid.voxels, components=2)
    with ctx.scope():
        mask = wp.ones(grid.voxels, dtype=wp.uint8, device=ctx.device)
        status = wp.zeros(1, dtype=wp.int32, device=ctx.device)
        if _api_support is not None:
            wp.copy(mask, _api_support, stream=ctx.stream)
            wp.launch(
                kernels.validate_support,
                dim=grid.voxels,
                inputs=[mask],
                outputs=[status],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        arrays = [wp.empty(grid.voxels, dtype=wp.float64, device=ctx.device) for _ in range(4)]
        dual = wp.empty(grid.voxels, dtype=wp.vec3d, device=ctx.device)
        contributions = wp.empty(2 * grid.voxels, dtype=wp.float64, device=ctx.device)
        control = wp.empty(2, dtype=wp.float64, device=ctx.device)
    workspace = RegularizationWorkspace(
        grid,
        spec,
        ctx,
        kernels,
        mask,
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        dual,
        contributions,
        control,
        status,
        reduction,
        wp.vec3d(*inverse),
        iteration_step,
    )
    workspace.check_status()
    return workspace


def regularization_value_and_gradient(
    field: wp.array[wp.float32] | wp.array[wp.float64],
    *,
    out_gradient: wp.array[wp.float32] | wp.array[wp.float64],
    workspace: RegularizationWorkspace,
    stream: wp.Stream | None = None,
) -> float:
    """R(x) = lambda_tv*sum_v phi(Dx_v)+lambda_l1*sum_v |x_v| and
    grad_R=D^T*p+lambda_l1*sign(x).

    Field and out_gradient are disjoint (voxels,) arrays in the declared FP32/FP64
    precision, flattened from (nz,ny,nx) with x fastest. D uses forward differences
    divided by physical x/y/z spacing in mm, with zero outward differences at the
    outer boundary. Isotropic p is lambda_tv*Dx/||Dx|| and anisotropic p uses
    component signs, choosing zero subgradients at zeros. The field must be zero
    outside the owned support mask and its inactive-coordinate derivative is zero.
    Weights declare penalty units with no implicit voxel measure. Only field
    values are differentiated. FP64 internal arithmetic and reduction precede
    gradient storage and the returned binary64 scalar penalty. Only scalar
    value/status cross the host boundary.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_field: Any = cast(Any, field)
    _api_out_gradient: Any = cast(Any, out_gradient)
    with workspace._exclusive():
        workspace._arrays(_api_field, _api_out_gradient, stream=stream)
        ctx, wp = workspace.context, workspace.context.wp
        workspace._initialise(_api_field, 0.0, enforce=True)
        nz, ny, nx = workspace.grid.shape
        spec = workspace.spec
        wp.launch(
            workspace._kernels.penalty_kernel(spec.tv_norm == "isotropic"),
            dim=workspace.grid.voxels,
            inputs=[workspace._x, nx, ny, nz, workspace._inverse, spec.tv_weight, spec.l1_weight],
            outputs=[workspace._dual, workspace._contributions, workspace._status],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
        wp.launch(
            workspace._kernels.gradient_kernel(spec.precision == "float64"),
            dim=workspace.grid.voxels,
            inputs=[
                workspace._x,
                workspace._mask,
                workspace._dual,
                nx,
                ny,
                nz,
                spec.l1_weight,
            ],
            outputs=[_api_out_gradient, workspace._status],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
        workspace._reduction.finish(
            workspace.grid.voxels,
            workspace._control,
            workspace._status,
            source=workspace._contributions,
            components=2,
        )
        (controls,) = workspace._completion.consume()
        return float(controls[0])


def regularization_prox(
    field: wp.array[wp.float32] | wp.array[wp.float64],
    *,
    step: float,
    out: wp.array[wp.float32] | wp.array[wp.float64],
    workspace: RegularizationWorkspace,
    policy: RegularizationProxPolicy | None = None,
    stream: wp.Stream | None = None,
) -> RegularizationProxResult:
    """Out = argmin_x ||x-field||^2/2 + step*R(x), with x fixed to zero outside support.

    Field and out are disjoint (voxels,) arrays in the declared FP32/FP64 storage,
    flattened from (nz,ny,nx) with x fastest. They share field units and positive
    step scales the declared TV/L1 weights. L1/support alone uses exact soft
    thresholding and support projection. TV uses resident Chambolle-Pock updates
    with the physical finite-difference operator and its exact transpose.
    RegularizationProxResult returns scalar primal, dual and gap values for the
    FP64 iterate before final output rounding, along with iterations and reason.
    Policy stops at its absolute/relative primal-dual gap or iteration budget.
    Inputs remain unchanged and no differentiation product is formed. Only the
    scalar certificate/status leave the device.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_field: Any = cast(Any, field)
    _api_out: Any = cast(Any, out)
    step = finite_scalar(step, "step", minimum=0.0)
    if step == 0:
        raise ContractError("regularization proximal step must be positive")
    tv, l1 = step * workspace.spec.tv_weight, step * workspace.spec.l1_weight
    if (
        not math.isfinite(tv)
        or not math.isfinite(l1)
        or (workspace.spec.tv_weight > 0 and tv == 0)
        or (workspace.spec.l1_weight > 0 and l1 == 0)
    ):
        raise NumericalError("unrepresentable proximal regularization coefficient")
    rules = RegularizationProxPolicy() if policy is None else policy
    with workspace._exclusive():
        workspace._arrays(_api_field, _api_out, stream=stream)
        ctx, wp = workspace.context, workspace.context.wp
        workspace._initialise(_api_field, l1, enforce=False)
        workspace._dual.zero_()
        current, following = workspace._x, workspace._following
        nz, ny, nx = workspace.grid.shape
        reason: Literal["closed_form", "primal_dual_gap", "iteration_budget"] = (
            "closed_form" if tv == 0 else "iteration_budget"
        )
        iterations = 0
        primal, dual, gap = workspace._read_certificate(current, tv, l1)
        if tv > 0 and gap <= rules.absolute_gap + rules.relative_gap * max(abs(primal), abs(dual)):
            reason = "primal_dual_gap"
        elif tv > 0:
            for iterations in range(1, rules.max_iterations + 1):
                wp.launch(
                    workspace._kernels.dual_kernel(workspace.spec.tv_norm == "isotropic"),
                    dim=workspace.grid.voxels,
                    inputs=[
                        workspace._extrapolated,
                        nx,
                        ny,
                        nz,
                        workspace._inverse,
                        workspace._iteration_step,
                        tv,
                    ],
                    outputs=[workspace._dual, workspace._status],
                    device=ctx.device,
                    stream=ctx.stream,
                    record_tape=False,
                )
                wp.launch(
                    workspace._kernels.primal_kernel,
                    dim=workspace.grid.voxels,
                    inputs=[
                        workspace._source,
                        current,
                        workspace._mask,
                        workspace._dual,
                        nx,
                        ny,
                        nz,
                        workspace._inverse,
                        workspace._iteration_step,
                        l1,
                    ],
                    outputs=[following, workspace._extrapolated, workspace._status],
                    device=ctx.device,
                    stream=ctx.stream,
                    record_tape=False,
                )
                current, following = following, current
                if iterations % rules.check_interval == 0 or iterations == rules.max_iterations:
                    primal, dual, gap = workspace._read_certificate(current, tv, l1)
                    if gap <= rules.absolute_gap + rules.relative_gap * max(abs(primal), abs(dual)):
                        reason = "primal_dual_gap"
                        break
        wp.launch(
            workspace._kernels.write_kernel(workspace.spec.precision == "float64"),
            dim=workspace.grid.voxels,
            inputs=[current],
            outputs=[_api_out, workspace._status],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
        workspace._checkpoint.consume()
        return RegularizationProxResult(iterations, reason, primal, dual, gap)
