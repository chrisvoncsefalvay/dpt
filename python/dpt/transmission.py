"""CUDA transmission with explicit storage, numerical and differentiation contracts.

Importing this module does not initialise CUDA or import Warp. Checked calls
validate device contents; ``validate=False`` is the asynchronous integration
boundary for callers that guarantee the current inputs satisfy these contracts.
No output arrays are allocated by an evaluation. See ``TransmissionWorkspace``
for stream ownership and checked completion of custom tape gradients.
"""

from __future__ import annotations

# Scientific API names mirror the chapter symbols; private scratch is module-owned.
# ruff: noqa: N803, N815
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._reductions import ReductionTree, prepare_reduction
from dpt._runtime import (
    DeviceContext,
    StatusReadback,
    ensure_tape,
    load_kernels,
    prepare_context,
    require_no_tape,
)
from dpt.contracts import ContractError, NumericalError, binary32_scalar, raise_device_status

BeamMode = Literal["none", "scalar", "device-scalar", "per-pixel"]
_BEAM_MODES: dict[str, int] = {"none": 0, "scalar": 1, "device-scalar": 2, "per-pixel": 3}
_LIMIT = 2**31 - 1
_TILE = 256

if TYPE_CHECKING:
    import warp as wp


class TransmissionError(ContractError):
    """Input, layout, aliasing or ownership violates the operator contract."""


class GradientRangeError(NumericalError):
    """A requested gradient cannot be represented in its binary32 destination."""


@dataclass(frozen=True, slots=True)
class TransmissionOutputs:
    """Selected caller-owned binary32 forward destinations on one device.

    Each selected buffer has the optical-depth input's length. Their contents
    and lifetime follow ``transmit`` and no arrays are allocated by this record.
    """

    transmittance: wp.array[wp.float32] | None = None
    counts: wp.array[wp.float32] | None = None
    log_transmittance: wp.array[wp.float32] | None = None
    removed: wp.array[wp.float32] | None = None

    def __post_init__(self) -> None:
        if all(
            value is None
            for value in (self.transmittance, self.counts, self.log_transmittance, self.removed)
        ):
            raise TransmissionError("request at least one output")


@dataclass(frozen=True, slots=True)
class TransmissionSpec:
    """Compile-independent input policy; the beam scalar's value remains dynamic."""

    beam: BeamMode = "none"
    active_L: bool = True
    active_beam: bool = False

    def __post_init__(self) -> None:
        if self.beam not in _BEAM_MODES:
            raise TransmissionError(f"unsupported beam representation: {self.beam!r}")
        if self.active_beam and self.beam not in ("device-scalar", "per-pixel"):
            raise TransmissionError("an active beam must be a CUDA array, not a Python scalar")


@dataclass(slots=True)
class TransmissionWorkspace:
    """Prepared scratch on one CUDA stream; never concurrently share a workspace.

    Inputs must stay unchanged until backward completes. Custom tape callbacks
    enqueue range checks asynchronously; call ``check_status()`` after backward
    before accepting the gradients. Use ``tape.zero()`` between independent
    reverse passes. Second derivatives through the custom callbacks are not
    supported. Workspace memory is O(P / 256) only for an active scalar beam.
    """

    spec: TransmissionSpec
    max_pixels: int
    context: DeviceContext
    _kernels: Any = field(repr=False)
    _empty: Any = field(repr=False)
    _status: Any = field(repr=False)
    _reduction: ReductionTree | None = field(repr=False)
    _status_readback: StatusReadback = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._status_readback = StatusReadback(self.context, self._status)

    @property
    def device(self) -> wp.Device:
        return self.context.device

    @property
    def stream(self) -> wp.Stream | None:
        return self.context.stream

    @property
    def _wp(self) -> Any:
        return self.context.wp

    @property
    def scratch_bytes(self) -> int:
        """Device scratch excluding caller-owned inputs, outputs and their adjoints."""
        return 4 + (self._reduction.scratch_bytes if self._reduction is not None else 0)

    def clear_status(self) -> None:
        """Clear accumulated diagnostics explicitly, on the owning stream."""
        with self.context.scope():
            self._status.zero_()

    def check_status(self) -> None:
        """Synchronise the owning stream and reject any accumulated numerical error."""
        self._raise_status(self._status_readback.read())

    def consume_status(self) -> None:
        """Deliver and reset one completed checkpoint, including reported errors."""
        self._raise_status(self._status_readback.read(reset=True))

    @property
    def host_staging_bytes(self) -> int:
        return self._status_readback.host_bytes

    def _raise_status(self, code: int) -> None:
        raise_device_status(
            code, "transmission", input_error=TransmissionError, range_error=GradientRangeError
        )

    def warm(
        self,
        *,
        forward_masks: tuple[int, ...] = (),
        vjp_masks: tuple[int, ...] = (),
        inverse: bool = False,
    ) -> tuple[str, ...]:
        """Compile selected variants without evaluating or allocating image outputs.

        Mask bits are T=1, counts=2, log_T=4 and removed=8. Each requested
        VJP mask prepares every active output subset, including overwrite and
        tape accumulation. Mask zero prepares a VJP with all seeds omitted.
        ``inverse=True`` also prepares decrement inversion and its VJP.
        """
        require_no_tape(contract_error=TransmissionError)
        if type(inverse) is not bool:
            raise TransmissionError("inverse must be a boolean")
        for masks, minimum in ((forward_masks, 1), (vjp_masks, 0)):
            if not isinstance(cast(object, masks), tuple) or any(
                type(mask) is not int or not minimum <= mask <= 15 for mask in masks
            ):
                raise TransmissionError(
                    "warm masks must be immutable tuples of legal integer masks"
                )
            if self.spec.beam == "none" and any(mask & 2 for mask in masks):
                raise TransmissionError("count variants require an open beam")
        if vjp_masks and not (self.spec.active_L or self.spec.active_beam):
            raise TransmissionError("VJP warming requires an active input")
        kernels: list[tuple[Any, int]] = [
            (
                self._kernels.get_forward_kernel(mask, _BEAM_MODES[self.spec.beam]),
                _TILE,
            )
            for mask in forward_masks
        ]
        for mask in vjp_masks:
            for write_depth, write_beam in ((True, False), (False, True), (True, True)):
                if write_depth and not self.spec.active_L:
                    continue
                if write_beam and not (self.spec.active_beam and self.spec.beam == "per-pixel"):
                    continue
                for accumulate in (False, True):
                    kernels.append(
                        (
                            self._kernels.get_vjp_kernel(
                                mask,
                                _BEAM_MODES[self.spec.beam],
                                write_depth,
                                write_beam,
                                accumulate,
                            ),
                            _TILE,
                        )
                    )
        if vjp_masks and self.spec.active_beam and self.spec.beam == "device-scalar":
            kernels.append((self._kernels.get_beam_partial_kernel(), _TILE))
            assert self._reduction is not None
            kernels.append((self._reduction._kernels.reduce_kernel(_TILE, 1), _TILE))
            kernels.extend(
                (self._reduction._kernels.finish_kernel(True, accumulate), _TILE)
                for accumulate in (False, True)
            )
        if inverse:
            kernels.append((self._kernels.get_inverse_kernel(), _TILE))
            kernels.extend(
                (self._kernels.get_inverse_vjp_kernel(accumulate), _TILE)
                for accumulate in (False, True)
            )
        modules: dict[tuple[str, int], Any] = {
            (kernel.module.name, block): kernel.module for kernel, block in kernels
        }
        with self.context.scope():
            self.check_status()
            for (_, block), module in modules.items():
                self._wp.load_module(module, device=self.device, block_dim=block)
        return tuple(f"{name}@block{block}" for name, block in modules)


def prepare_transmission(
    spec: TransmissionSpec | None = None,
    *,
    device: str | wp.Device = "cuda:0",
    max_pixels: int,
    stream: wp.Stream | None = None,
) -> TransmissionWorkspace:
    """Allocate persistent diagnostics and reduction scratch, never image outputs."""
    if spec is not None and not isinstance(spec, TransmissionSpec):
        raise TransmissionError("spec must be a TransmissionSpec")
    if isinstance(max_pixels, bool) or type(max_pixels) is not int:
        raise TransmissionError("max_pixels must be an integer")
    if not 0 <= max_pixels <= _LIMIT:
        raise TransmissionError(f"max_pixels must lie in [0, {_LIMIT}]")
    context = prepare_context(device=device, stream=stream, contract_error=TransmissionError)
    wp, kernels = context.wp, load_kernels("dpt.kernels.transmission")
    spec = spec or TransmissionSpec()
    reduction = None
    with context.scope():
        empty = wp.empty(0, dtype=wp.float32, device=context.device)
        status = wp.zeros(1, dtype=wp.int32, device=context.device)
        if spec.active_beam and spec.beam == "device-scalar":
            reduction = prepare_reduction(context, max_pixels, tile=_TILE)
    return TransmissionWorkspace(spec, max_pixels, context, kernels, empty, status, reduction)


def _array(
    value: Any, name: str, workspace: TransmissionWorkspace, length: int | None = None
) -> int:
    size = workspace.context.array(
        value,
        name,
        dtype=workspace._wp.float32,
        shape=(length,) if length is not None else None,
    )
    if size > workspace.max_pixels and length != 1:
        raise TransmissionError(f"{name} exceeds workspace capacity")
    return size


def _beam(n0: Any, count: int, workspace: TransmissionWorkspace) -> tuple[Any, float]:
    mode = workspace.spec.beam
    if mode == "none":
        if n0 is not None:
            raise TransmissionError("this workspace declares no open-beam input")
        return workspace._empty, 0.0
    if mode == "scalar":
        try:
            return workspace._empty, binary32_scalar(n0, "n0", minimum=0.0)
        except ContractError as error:
            raise TransmissionError(str(error)) from error
    _array(n0, "n0", workspace, 1 if mode == "device-scalar" else count)
    return n0, 0.0


def _values(workspace: TransmissionWorkspace, arrays: list[tuple[str, Any, str]]) -> None:
    wp, kernels = workspace._wp, workspace._kernels
    workspace.check_status()
    for _, array, kind in arrays:
        if array.size:
            wp.launch(
                getattr(kernels, kind),
                dim=array.size,
                inputs=[array],
                outputs=[workspace._status],
                stream=workspace.stream,
                record_tape=False,
            )
    try:
        workspace.check_status()
    except TransmissionError as error:
        for name, array, kind in arrays:
            for index, item in enumerate(array.numpy()):
                value = float(item)
                valid = math.isfinite(value)
                if kind != "finite_seed":
                    valid = valid and value >= 0
                if kind == "decrement_domain":
                    valid = valid and value < 1
                if not valid:
                    raise TransmissionError(f"{name}[{index}]={value!r} violates {kind}") from error
        raise


def transmit(
    L: wp.array[wp.float32],
    n0: float | wp.array[wp.float32] | None = None,
    *,
    out_T: wp.array[wp.float32] | None = None,
    out_counts: wp.array[wp.float32] | None = None,
    out_log_T: wp.array[wp.float32] | None = None,
    out_removed: wp.array[wp.float32] | None = None,
    outputs: TransmissionOutputs | None = None,
    workspace: TransmissionWorkspace,
    stream: wp.Stream | None = None,
    tape: wp.Tape | None = None,
    validate: bool = True,
) -> None:
    """T=exp(-L), counts=n0*T, log_T=-L and removed=-expm1(-L).

    L is finite nonnegative dimensionless optical depth in FP32 (pixels,).
    Each requested output is caller-owned FP32 (pixels,) and at least one is
    required. T and removed are dimensionless, log_T is a natural logarithm and
    counts is an expected photon population. N0 is a nonnegative host scalar,
    an FP32 pixel array or absent according to the declared beam policy.
    TransmissionOutputs groups the four destinations. The retained out_* names
    remain available. This deterministic map performs no sampling or transfer.
    FP64 local evaluation rounds once to the FP32 destinations. Active L/beam
    flags govern the first-order reverse products and tape recording. Original
    inputs must remain unchanged through backward. Validate=False keeps launches
    asynchronous and requires an explicit completed diagnostic checkpoint.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_l: Any = cast(Any, L)
    _api_n0: Any = cast(Any, n0)
    _api_out_t: Any = cast(Any, out_T)
    _api_out_counts: Any = cast(Any, out_counts)
    _api_out_log_t: Any = cast(Any, out_log_T)
    _api_out_removed: Any = cast(Any, out_removed)
    if outputs is not None:
        if not isinstance(cast(object, outputs), TransmissionOutputs):
            raise TransmissionError("outputs must be TransmissionOutputs")
        if any(
            value is not None
            for value in (_api_out_t, _api_out_counts, _api_out_log_t, _api_out_removed)
        ):
            raise TransmissionError("supply outputs or individual transmission destinations")
        _api_out_t, _api_out_counts, _api_out_log_t, _api_out_removed = (
            outputs.transmittance,
            outputs.counts,
            outputs.log_transmittance,
            outputs.removed,
        )
    ensure_tape(tape, contract_error=TransmissionError)
    wp, kernels = workspace._wp, workspace._kernels
    selected = workspace.context.assert_stream(stream)
    count = _array(_api_l, "L", workspace)
    beam, scalar = _beam(_api_n0, count, workspace)
    output_arrays = (_api_out_t, _api_out_counts, _api_out_log_t, _api_out_removed)
    mask = sum(1 << i for i, value in enumerate(output_arrays) if value is not None)
    if mask == 0:
        raise TransmissionError("request at least one output")
    if _api_out_counts is not None and workspace.spec.beam == "none":
        raise TransmissionError("counts require an explicit open-beam input")
    writes: list[tuple[str, Any]] = []
    for name, value in zip(
        ("out_T", "out_counts", "out_log_T", "out_removed"), output_arrays, strict=True
    ):
        if value is not None:
            _array(value, name, workspace, count)
            writes.append((name, value))
    workspace.context.disjoint([("L", _api_l), ("n0", beam)], writes)
    arrays = _tape_arrays(_api_l, beam, output_arrays, workspace) if tape is not None else []
    # Stream ownership was checked above; callers supply cross-stream ordering.
    with workspace.context.scope():
        if validate:
            _values(
                workspace, [("L", _api_l, "finite_nonnegative"), ("n0", beam, "finite_nonnegative")]
            )
        if count:
            wp.launch(
                kernels.get_forward_kernel(mask, _BEAM_MODES[workspace.spec.beam]),
                dim=count,
                inputs=[_api_l, beam, scalar],
                outputs=[
                    value if value is not None else workspace._empty for value in output_arrays
                ],
                stream=selected,
                block_dim=_TILE,
                record_tape=False,
            )
        if tape is not None:
            _record_transmission(tape, _api_l, _api_n0, beam, output_arrays, workspace, arrays)


def _tape_arrays(
    L: Any, beam: Any, outputs: tuple[Any, ...], workspace: TransmissionWorkspace
) -> list[Any]:
    arrays = [value for value in outputs if value is not None]
    if workspace.spec.active_L:
        arrays.append(L)
    if workspace.spec.active_beam:
        arrays.append(beam)
    if not workspace.spec.active_L and not workspace.spec.active_beam:
        raise TransmissionError("recording requires at least one active input")
    for array in arrays:
        if array.grad is None:
            raise TransmissionError("allocate participating tape arrays with requires_grad=True")
    workspace.context.disjoint(
        [("forward input", L), ("beam", beam)]
        + [("forward output", a) for a in outputs if a is not None],
        [("gradient", a.grad) for a in arrays],
    )
    return arrays


def _dependencies(
    tape: Any, reads: tuple[Any, Any], outputs: tuple[Any, ...], workspace: TransmissionWorkspace
) -> None:
    if not workspace._wp.config.verify_autograd_array_access:
        return
    # Public access markers diagnose recorded writes before backward. A zero-dim
    # record gives Tape.reset()/backward() ownership of fixed inputs and view
    # parents too, without allocating gradients or launching a marker thread.
    for value in outputs:
        if value is not None:
            value.mark_write()
    for value in reads:
        value.mark_read()
    padded = [value if value is not None else workspace._empty for value in outputs]
    padded += [workspace._empty] * (4 - len(padded))
    tape.record_launch(
        workspace._kernels.dependency_marker,
        dim=0,
        max_blocks=0,
        inputs=list(reads),
        outputs=padded,
        device=workspace.device,
        block_dim=_TILE,
    )


def _record_transmission(
    tape: Any,
    L: Any,
    n0: Any,
    beam: Any,
    outputs: tuple[Any, ...],
    workspace: TransmissionWorkspace,
    arrays: list[Any],
) -> None:
    _dependencies(tape, (L, beam), outputs, workspace)

    def backward() -> None:
        workspace.context.assert_stream()
        transmission_vjp(
            L,
            n0,
            cotangent_T=outputs[0].grad if outputs[0] is not None else None,
            cotangent_counts=outputs[1].grad if outputs[1] is not None else None,
            cotangent_log_T=outputs[2].grad if outputs[2] is not None else None,
            cotangent_removed=outputs[3].grad if outputs[3] is not None else None,
            out_grad_L=L.grad if workspace.spec.active_L else None,
            out_grad_n0=beam.grad if workspace.spec.active_beam else None,
            workspace=workspace,
            validate=False,
            _accumulate=True,
        )
        with workspace.context.scope():
            for value in outputs:
                if value is not None and not value.retain_grad:
                    value.grad.zero_()

    tape.record_func(backward, arrays)


def transmission_vjp(
    L: wp.array[wp.float32],
    n0: float | wp.array[wp.float32] | None = None,
    *,
    cotangent_T: wp.array[wp.float32] | None = None,
    cotangent_counts: wp.array[wp.float32] | None = None,
    cotangent_log_T: wp.array[wp.float32] | None = None,
    cotangent_removed: wp.array[wp.float32] | None = None,
    out_grad_L: wp.array[wp.float32] | None = None,
    out_grad_n0: wp.array[wp.float32] | None = None,
    workspace: TransmissionWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
    _accumulate: bool = False,
) -> None:
    """grad_L = -T*cT - n0*T*cCounts - cLog + T*cRemoved.

    The beam product is T*cCounts per pixel, summed for a shared scalar beam.
    L and all selected cotangents have FP32 shape (pixels,), with the units of
    their corresponding forward outputs. N0 follows the prepared scalar/pixel
    beam policy. Out_grad_L is FP32 (pixels,), while out_grad_n0 is FP32 (1,)
    for a scalar beam or (pixels,) for a pixel beam. Missing covectors are zero
    and the caller's covectors are preserved. Active parameters are declared by
    the workspace. Scalar illumination uses a fixed FP64 reduction before the
    final FP32 cast. Standalone calls overwrite destinations.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_l: Any = cast(Any, L)
    _api_n0: Any = cast(Any, n0)
    _api_cotangent_t: Any = cast(Any, cotangent_T)
    _api_cotangent_counts: Any = cast(Any, cotangent_counts)
    _api_cotangent_log_t: Any = cast(Any, cotangent_log_T)
    _api_cotangent_removed: Any = cast(Any, cotangent_removed)
    _api_out_grad_l: Any = cast(Any, out_grad_L)
    _api_out_grad_n0: Any = cast(Any, out_grad_n0)
    require_no_tape(contract_error=TransmissionError)
    wp, kernels = workspace._wp, workspace._kernels
    selected = workspace.context.assert_stream(stream)
    count = _array(_api_l, "L", workspace)
    beam, scalar = _beam(_api_n0, count, workspace)
    seeds = (_api_cotangent_t, _api_cotangent_counts, _api_cotangent_log_t, _api_cotangent_removed)
    mask = sum(1 << i for i, value in enumerate(seeds) if value is not None)
    if _api_cotangent_counts is not None and workspace.spec.beam == "none":
        raise TransmissionError("a count cotangent requires an open-beam input")
    reads = [("L", _api_l), ("n0", beam)]
    values = [("L", _api_l, "finite_nonnegative"), ("n0", beam, "finite_nonnegative")]
    for name, seed in zip(
        ("cotangent_T", "cotangent_counts", "cotangent_log_T", "cotangent_removed"),
        seeds,
        strict=True,
    ):
        if seed is not None:
            _array(seed, name, workspace, count)
            reads.append((name, seed))
            values.append((name, seed, "finite_seed"))
    writes: list[tuple[str, Any]] = []
    if _api_out_grad_l is not None:
        if not workspace.spec.active_L:
            raise TransmissionError("L is declared fixed")
        _array(_api_out_grad_l, "out_grad_L", workspace, count)
        writes.append(("out_grad_L", _api_out_grad_l))
    if _api_out_grad_n0 is not None:
        if not workspace.spec.active_beam:
            raise TransmissionError("n0 is declared fixed")
        _array(
            _api_out_grad_n0,
            "out_grad_n0",
            workspace,
            1 if workspace.spec.beam == "device-scalar" else count,
        )
        writes.append(("out_grad_n0", _api_out_grad_n0))
    if not writes:
        raise TransmissionError("request at least one active input gradient")
    workspace.context.disjoint(reads, writes)
    # Stream ownership was checked above; callers supply cross-stream ordering.
    with workspace.context.scope():
        if validate:
            _values(workspace, values)
        per_pixel_beam = _api_out_grad_n0 is not None and workspace.spec.beam == "per-pixel"
        if count and (_api_out_grad_l is not None or per_pixel_beam):
            wp.launch(
                kernels.get_vjp_kernel(
                    mask,
                    _BEAM_MODES[workspace.spec.beam],
                    _api_out_grad_l is not None,
                    per_pixel_beam,
                    _accumulate,
                ),
                dim=count,
                inputs=[_api_l, beam, scalar]
                + [v if v is not None else workspace._empty for v in seeds],
                outputs=[
                    _api_out_grad_l if _api_out_grad_l is not None else workspace._empty,
                    _api_out_grad_n0 if per_pixel_beam else workspace._empty,
                    workspace._status,
                ],
                stream=selected,
                block_dim=_TILE,
                record_tape=False,
            )
        if _api_out_grad_n0 is not None and workspace.spec.beam == "device-scalar":
            if count and _api_cotangent_counts is not None:
                reduction = workspace._reduction
                assert reduction is not None
                blocks = (count + _TILE - 1) // _TILE
                wp.launch_tiled(
                    kernels.get_beam_partial_kernel(),
                    dim=blocks,
                    inputs=[_api_l, _api_cotangent_counts],
                    outputs=[reduction.partials[0]],
                    block_dim=_TILE,
                    stream=selected,
                    record_tape=False,
                )
                reduction.finish(
                    blocks,
                    _api_out_grad_n0,
                    workspace._status,
                    accumulate=_accumulate,
                )
            elif not _accumulate:
                _api_out_grad_n0.zero_()
        if validate:
            workspace.check_status()


def optical_depth_from_removed(
    delta: wp.array[wp.float32],
    *,
    out_L: wp.array[wp.float32],
    workspace: TransmissionWorkspace,
    stream: wp.Stream | None = None,
    tape: wp.Tape | None = None,
    validate: bool = True,
) -> None:
    """L = -log1p(-delta) inverts the removed fraction delta=1-T.

    Delta and out_L are disjoint contiguous FP32 (pixels,) arrays of
    dimensionless values. Every delta must be finite and satisfy 0 <= delta < 1.
    The cancellation-safe local calculation rounds to FP32 output. The active
    input is delta when recording the optional first-order tape rule. Inputs
    remain alive and unchanged until backward completes. Checked calls deliver
    domain/range diagnostics and unchecked calls require workspace.check_status().

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_delta: Any = cast(Any, delta)
    _api_out_l: Any = cast(Any, out_L)
    ensure_tape(tape, contract_error=TransmissionError)
    wp, kernels = workspace._wp, workspace._kernels
    selected = workspace.context.assert_stream(stream)
    count = _array(_api_delta, "delta", workspace)
    _array(_api_out_l, "out_L", workspace, count)
    workspace.context.disjoint([("delta", _api_delta)], [("out_L", _api_out_l)])
    if tape is not None:
        if _api_delta.grad is None or _api_out_l.grad is None:
            raise TransmissionError("inverse tape arrays require preallocated gradients")
        workspace.context.disjoint(
            [("delta", _api_delta), ("out_L", _api_out_l)],
            [("delta.grad", _api_delta.grad), ("out_L.grad", _api_out_l.grad)],
        )
    # Stream ownership was checked above; callers supply cross-stream ordering.
    with workspace.context.scope():
        if validate:
            _values(workspace, [("delta", _api_delta, "decrement_domain")])
        if count:
            wp.launch(
                kernels.get_inverse_kernel(),
                dim=count,
                inputs=[_api_delta],
                outputs=[_api_out_l],
                stream=selected,
                block_dim=_TILE,
                record_tape=False,
            )
        if tape is not None:
            _dependencies(tape, (_api_delta, workspace._empty), (_api_out_l,), workspace)

            def backward() -> None:
                workspace.context.assert_stream()
                optical_depth_from_removed_vjp(
                    _api_delta,
                    _api_out_l.grad,
                    out_grad_delta=_api_delta.grad,
                    workspace=workspace,
                    validate=False,
                    _accumulate=True,
                )
                if not _api_out_l.retain_grad:
                    _api_out_l.grad.zero_()

            cast(Any, tape).record_func(backward, [_api_delta, _api_out_l])


def optical_depth_from_removed_vjp(
    delta: wp.array[wp.float32],
    cotangent: wp.array[wp.float32] | None,
    *,
    out_grad_delta: wp.array[wp.float32],
    workspace: TransmissionWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
    _accumulate: bool = False,
) -> None:
    """Grad_delta[p] = cotangent[p]/(1-delta[p]) is the inverse-transmission VJP.

    Delta, cotangent and out_grad_delta are disjoint FP32 (pixels,) arrays.
    Delta is a finite dimensionless removed fraction in [0,1), the covector is
    signed and finite, and the destination has the covector's units because both
    depth and delta are dimensionless. The local calculation rounds to FP32 and
    reports an unrepresentable result. Standalone calls preserve both inputs and
    overwrite the destination. Only the removed fraction is differentiated.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_delta: Any = cast(Any, delta)
    _api_cotangent: Any = cast(Any, cotangent)
    _api_out_grad_delta: Any = cast(Any, out_grad_delta)
    require_no_tape(contract_error=TransmissionError)
    wp, kernels = workspace._wp, workspace._kernels
    selected = workspace.context.assert_stream(stream)
    count = _array(_api_delta, "delta", workspace)
    _array(_api_cotangent, "cotangent", workspace, count)
    _array(_api_out_grad_delta, "out_grad_delta", workspace, count)
    workspace.context.disjoint(
        [("delta", _api_delta), ("cotangent", _api_cotangent)],
        [("out_grad_delta", _api_out_grad_delta)],
    )
    # Stream ownership was checked above; callers supply cross-stream ordering.
    with workspace.context.scope():
        if validate:
            _values(
                workspace,
                [
                    ("delta", _api_delta, "decrement_domain"),
                    ("cotangent", _api_cotangent, "finite_seed"),
                ],
            )
        if count:
            wp.launch(
                kernels.get_inverse_vjp_kernel(_accumulate),
                dim=count,
                inputs=[_api_delta, _api_cotangent],
                outputs=[_api_out_grad_delta, workspace._status],
                stream=selected,
                block_dim=_TILE,
                record_tape=False,
            )
        if validate:
            workspace.check_status()
