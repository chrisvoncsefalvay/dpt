"""Deterministic detector operators and explicitly separate observation models.

Expected response, spatial spreading, electronic calibration and random draws
are distinct operations with caller-owned buffers. An energy-integrating mean
is not passed to a Poisson count sampler by the library. Imports are CPU-safe;
all numerical image work uses the single CUDA implementation.
"""

from __future__ import annotations

# Public Python boundaries validate runtime inputs; workspace scratch stays module-owned.
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._runtime import (
    DeviceContext,
    StatusReadback,
    load_kernels,
    prepare_context,
    require_no_tape,
)
from dpt.contracts import (
    ContractError,
    finite_scalar,
    finite_tuple,
    integer,
    raise_device_status,
)
from dpt.materials import Provenance

if TYPE_CHECKING:
    import warp as wp


@dataclass(frozen=True, slots=True)
class GaussianReadNoiseExecution:
    """Electronic-noise sampling policy retained by a detector workspace.

    ``scalar-box-muller`` preserves the historical per-pixel counter and cosine
    member. ``box-muller-pairs`` uses both members of one transform at each
    absolute even-pixel anchor in domain 0xffffffff. Both policies generate
    independent standard normal variates and require the same algorithm in the
    observation identity. Changing this record changes the sample coordinates.

    Example::

        policy = GaussianReadNoiseExecution("box-muller-pairs")
        identity = ObservationIdentity(17, 3, read_noise_algorithm=policy.algorithm)
        assert identity.read_noise_algorithm == "box-muller-pairs"

    The independent normal pair is the Box and Muller transformation,
    `Annals of Mathematical Statistics 29 (1958), 610-611
    <https://doi.org/10.1214/aoms/1177706645>`_. Its implementation is
    ``dpt.kernels.detector.gaussian_read_noise_pairs``.
    """

    algorithm: Literal["scalar-box-muller", "box-muller-pairs"] = "scalar-box-muller"

    def __post_init__(self) -> None:
        if self.algorithm not in ("scalar-box-muller", "box-muller-pairs"):
            raise ContractError("unsupported Gaussian read-noise execution policy")


@dataclass(slots=True)
class DetectorWorkspace:
    """Reusable diagnostics and scalar-reduction scratch on one CUDA stream."""

    max_pixels: int
    context: DeviceContext
    _kernels: Any = field(repr=False)
    _checks: Any = field(repr=False)
    _status: Any = field(repr=False)
    _partials: Any = field(repr=False)
    _read_noise_execution: GaussianReadNoiseExecution = field(
        default_factory=GaussianReadNoiseExecution
    )
    _status_readback: StatusReadback = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._status_readback = StatusReadback(self.context, self._status)

    @property
    def read_noise_execution(self) -> GaussianReadNoiseExecution:
        """Immutable read-noise policy bound when this workspace was prepared."""
        return self._read_noise_execution

    @property
    def reduction_groups(self) -> int:
        """Read-only launch bound derived from the allocated scratch capacity."""
        return int(self._partials.size)

    @property
    def scratch_bytes(self) -> int:
        return 4 + 8 * int(self._partials.size)

    @property
    def host_bytes(self) -> int:
        return self._status_readback.host_bytes

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()

    def check_status(self) -> None:
        """Synchronise and reject a numerical failure; no failed draw is an observation."""
        code = self._status_readback.read()
        raise_device_status(
            code,
            "detector",
            messages={4: "Poisson draw budget exhausted; discard the entire realisation"},
        )

    def consume_status(self) -> None:
        """Deliver and clear one diagnostic checkpoint on the owning stream."""
        raise_device_status(
            self._status_readback.read(reset=True),
            "detector",
            messages={4: "Poisson draw budget exhausted; discard the entire realisation"},
        )

    def _values(self, arrays: list[tuple[Any, str]]) -> None:
        self.check_status()
        for array, domain in arrays:
            if not array.size:
                continue
            kernel = {
                "finite": self._checks.check_finite,
                "nonnegative": self._checks.check_nonnegative,
                "positive": self._kernels.check_positive,
                "rate": self._kernels.get_rate_check(array.dtype == self.context.wp.float64),
            }[domain]
            self.context.wp.launch(
                kernel,
                dim=array.size,
                inputs=[array],
                outputs=[self._status],
                device=self.context.device,
                stream=self.context.stream,
                record_tape=False,
            )
        self.consume_status()

    def _pixels(self, values: Any, name: str) -> int:
        count = self.context.array(values, name, dtype=self.context.wp.float32)
        if count > self.max_pixels:
            raise ContractError(f"{name} exceeds workspace capacity")
        return count

    def _compound_values(self, rates: Any, scores: Any) -> None:
        self.check_status()
        count = max(int(rates.size), int(scores.size))
        if count:
            self.context.wp.launch(
                self._kernels._get_compound_input_check(
                    rates.dtype == self.context.wp.float64,
                    rates.size == scores.size,
                    bool(rates.size),
                ),
                dim=count,
                inputs=[rates, scores],
                outputs=[self._status],
                device=self.context.device,
                stream=self.context.stream,
                record_tape=False,
            )
        self.consume_status()


def prepare_detector(
    *,
    max_pixels: int,
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
    reduction_groups: int = 64,
    read_noise_execution: GaussianReadNoiseExecution | None = None,
) -> DetectorWorkspace:
    """Prepare reusable detector scratch and the immutable read-noise policy.

    ``max_pixels`` is the maximum flat image extent, and ``reduction_groups``
    bounds scalar-reduction scratch. Preparation owns O(groups) scratch on the
    selected CPU or CUDA device and allocates no detector image. The default
    ``GaussianReadNoiseExecution()`` retains the historical per-pixel algorithm.
    Select ``GaussianReadNoiseExecution("box-muller-pairs")`` to reuse both
    independent normals from each absolute even-pixel anchor. Observation
    identities must name the same policy before ``add_gaussian_read_noise``.
    """
    integer(max_pixels, "max_pixels")
    integer(reduction_groups, "reduction_groups", minimum=1, maximum=256)
    if read_noise_execution is not None and not isinstance(
        read_noise_execution, GaussianReadNoiseExecution
    ):
        raise ContractError("read_noise_execution must be a GaussianReadNoiseExecution")
    execution = (
        GaussianReadNoiseExecution() if read_noise_execution is None else read_noise_execution
    )
    context = prepare_context(device=device, stream=stream)
    kernels = load_kernels("dpt.kernels.detector")
    checks = load_kernels("dpt.kernels.spectral")
    with context.scope():
        status = context.wp.zeros(1, dtype=context.wp.int32, device=context.device)
        partials = context.wp.empty(
            min(reduction_groups, max(1, (max_pixels + 255) // 256)),
            dtype=context.wp.float64,
            device=context.device,
        )
    return DetectorWorkspace(max_pixels, context, kernels, checks, status, partials, execution)


def cast_signal(
    source: wp.array[wp.float32] | wp.array[wp.float64],
    *,
    out_signal: wp.array[wp.float32] | wp.array[wp.float64],
    workspace: DetectorWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Overwrite ``out_signal[p] = round_destination(source[p])``.

    Both contiguous arrays use binary32 or binary64 and have identical shapes,
    with total size at most ``workspace.max_pixels``. Source values must be
    finite. A flat image has shape
    ``(P,)``, and image/batch axes are also preserved. Units are unchanged.
    Binary64-to-binary32 conversion uses ordinary IEEE rounding, including
    rounding a small tail to zero, and diagnoses overflow.

    The caller owns both disjoint buffers on the workspace device. This call
    returns ``None``, leaves the source unchanged and runs on the owning stream.
    ``validate=False`` defers the diagnostic checkpoint to
    ``workspace.check_status()`` before output use. Conversion preserves the
    meaning of the signal, including its declared count or deposited-energy units.

    Example::

        import warp as wp
        from dpt.detector import cast_signal, prepare_detector

        workspace = prepare_detector(max_pixels=2, device="cpu")
        source = wp.array([1.5, 2.0], dtype=wp.float64, device="cpu")
        output = wp.empty(2, dtype=wp.float32, device="cpu")
        cast_signal(source, out_signal=output, workspace=workspace)
        assert output.numpy().tolist() == [1.5, 2.0]

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_source: Any = cast(Any, source)
    _api_out_signal: Any = cast(Any, out_signal)
    require_no_tape()
    ctx = workspace.context
    ctx.assert_stream(stream)
    for name, values in (("source", _api_source), ("out_signal", _api_out_signal)):
        if getattr(values, "dtype", None) not in (ctx.wp.float32, ctx.wp.float64):
            raise ContractError(f"{name} must use binary32 or binary64 precision")
    count = ctx.array(_api_source, "source", dtype=_api_source.dtype, ndim=_api_source.ndim)
    if count > workspace.max_pixels:
        raise ContractError("source exceeds detector workspace capacity")
    ctx.array(
        _api_out_signal,
        "out_signal",
        dtype=_api_out_signal.dtype,
        shape=tuple(_api_source.shape),
        ndim=_api_source.ndim,
    )
    ctx.disjoint([("source", _api_source)], [("out_signal", _api_out_signal)])
    _api_source, _api_out_signal = _api_source.flatten(), _api_out_signal.flatten()
    with ctx.scope():
        if validate:
            workspace.check_status()
            if count:
                ctx.wp.launch(
                    workspace._checks.get_value_check(_api_source.dtype == ctx.wp.float64, False),
                    dim=count,
                    inputs=[_api_source],
                    outputs=[workspace._status],
                    device=ctx.device,
                    stream=ctx.stream,
                    record_tape=False,
                )
            workspace.consume_status()
        if count:
            ctx.wp.launch(
                workspace._kernels.get_signal_cast(
                    _api_source.dtype == ctx.wp.float64, _api_out_signal.dtype == ctx.wp.float64
                ),
                dim=count,
                inputs=[_api_source],
                outputs=[_api_out_signal, workspace._status],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            workspace.consume_status()


@dataclass(frozen=True, slots=True)
class CalibrationSpec:
    """y = gain * exposure * mean + offset, in the declared output unit.

    Gain is strictly positive; exposure is non-negative; offset is any finite
    electronic baseline. A fixed flat-field may vary by pixel. Optimised gain
    and offset are shared scalars: unconstrained correction images are excluded.
    Exposure and gain cannot both be active because their scale is unidentifiable.
    """

    output_unit: str | None = None
    shared_gain: bool = True
    shared_offset: bool = True
    active_mean: bool = True
    active_gain: bool = False
    active_exposure: bool = False
    active_offset: bool = False

    def __post_init__(self) -> None:
        if self.output_unit is not None and (
            not isinstance(self.output_unit, str) or not self.output_unit.strip()
        ):
            raise ContractError("calibration output unit is required")
        for name in (
            "shared_gain",
            "shared_offset",
            "active_mean",
            "active_gain",
            "active_exposure",
            "active_offset",
        ):
            if type(getattr(self, name)) is not bool:
                raise ContractError(f"{name} must be a boolean")
        if self.active_gain and self.active_exposure:
            raise ContractError("fix gain or exposure before fitting the other scale")
        if self.active_gain and not self.shared_gain:
            raise ContractError("a fitted gain must be a shared acquisition parameter")
        if self.active_offset and not self.shared_offset:
            raise ContractError("a fitted offset must be a shared acquisition parameter")


def _calibration_inputs(
    mean: Any,
    gain: Any,
    exposure: Any,
    offset: Any,
    spec: CalibrationSpec,
    workspace: DetectorWorkspace,
) -> tuple[int, list[tuple[str, Any]]]:
    ctx = workspace.context
    pixels = workspace._pixels(mean, "mean")
    for name, array, size in (
        ("gain", gain, 1 if spec.shared_gain else pixels),
        ("exposure", exposure, 1),
        ("offset", offset, 1 if spec.shared_offset else pixels),
    ):
        ctx.array(array, name, dtype=ctx.wp.float32, shape=(size,))
    return pixels, [("mean", mean), ("gain", gain), ("exposure", exposure), ("offset", offset)]


def calibrate(
    mean: wp.array[wp.float32],
    gain: wp.array[wp.float32],
    exposure: wp.array[wp.float32],
    offset: wp.array[wp.float32],
    *,
    out_signal: wp.array[wp.float32],
    spec: CalibrationSpec,
    workspace: DetectorWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Overwrite ``out_signal[p] = gain[p] exposure[0] mean[p] + offset[p]``.

    Finite nonnegative ``mean`` and the caller-owned destination are flat binary32 ``(P,)``
    images, with ``P <= workspace.max_pixels``. ``exposure`` is binary32
    ``(1,)``, finite and nonnegative. ``gain`` is finite and strictly positive, and
    ``offset`` is finite. Their binary32 arrays have shape ``(1,)`` when the
    corresponding ``spec.shared_*`` flag is true, otherwise ``(P,)``.
    Exposure is a declared relative scale, gain converts the mean's units to
    ``spec.output_unit`` and offset has that output unit.

    Binary64 intermediates are rounded once into binary32 output. All buffers
    are contiguous, disjoint and on the workspace device. Inputs remain
    unchanged, the result overwrites the destination on the owning stream and
    the return value is ``None``. Checked calls validate before writes, while
    ``validate=False`` requires a later ``workspace.check_status()`` checkpoint.

    Example::

        import warp as wp
        from dpt.detector import CalibrationSpec, prepare_detector

        workspace = prepare_detector(max_pixels=2, device="cpu")
        mean = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
        gain = wp.array([2.0], dtype=wp.float32, device="cpu")
        exposure = wp.array([3.0], dtype=wp.float32, device="cpu")
        offset = wp.array([1.0], dtype=wp.float32, device="cpu")

        from dpt.detector import calibrate
        output = wp.empty(2, dtype=wp.float32, device="cpu")
        calibrate(mean, gain, exposure, offset, out_signal=output,
                  spec=CalibrationSpec(output_unit="ADU"), workspace=workspace)
        assert output.numpy().tolist() == [7.0, 13.0]

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_mean: Any = cast(Any, mean)
    _api_gain: Any = cast(Any, gain)
    _api_exposure: Any = cast(Any, exposure)
    _api_offset: Any = cast(Any, offset)
    _api_out_signal: Any = cast(Any, out_signal)
    require_no_tape()
    ctx = workspace.context
    ctx.assert_stream(stream)
    pixels, reads = _calibration_inputs(
        _api_mean, _api_gain, _api_exposure, _api_offset, spec, workspace
    )
    ctx.array(_api_out_signal, "out_signal", dtype=ctx.wp.float32, shape=(pixels,))
    ctx.disjoint(reads, [("out_signal", _api_out_signal)])
    with ctx.scope():
        if validate:
            workspace._values(
                [
                    (_api_mean, "nonnegative"),
                    (_api_gain, "positive"),
                    (_api_exposure, "nonnegative"),
                    (_api_offset, "finite"),
                ]
            )
        if pixels:
            ctx.wp.launch(
                workspace._kernels.get_calibration(spec.shared_gain, spec.shared_offset),
                dim=pixels,
                inputs=[_api_mean, _api_gain, _api_exposure, _api_offset],
                outputs=[_api_out_signal, workspace._status],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            workspace.consume_status()


def calibration_vjp(
    mean: wp.array[wp.float32],
    gain: wp.array[wp.float32],
    exposure: wp.array[wp.float32],
    offset: wp.array[wp.float32],
    *,
    cotangent: wp.array[wp.float32] | None,
    out_grad_mean: wp.array[wp.float32] | wp.array[wp.float64] | None = None,
    out_grad_gain: wp.array[wp.float32] | wp.array[wp.float64] | None = None,
    out_grad_exposure: wp.array[wp.float32] | wp.array[wp.float64] | None = None,
    out_grad_offset: wp.array[wp.float32] | wp.array[wp.float64] | None = None,
    spec: CalibrationSpec,
    workspace: DetectorWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Apply ``bar_mean[p]=g[p] gain[p] exposure[0]`` and shared scalar VJPs.

    For ``g=cotangent``, ``bar_gain=sum_p g[p] exposure[0] mean[p]``,
    ``bar_exposure=sum_p g[p] gain[p] mean[p]`` and ``bar_offset=sum_p g[p]``.
    ``mean`` and the signed cotangent are flat binary32 ``(P,)`` arrays.
    ``exposure`` is binary32 ``(1,)``, while gain/offset have shape ``(1,)``
    or ``(P,)`` according to ``spec.shared_gain``/``spec.shared_offset``.
    Mean and exposure are finite and nonnegative, gain is finite and strictly
    positive, and offset/cotangent are finite and may be signed. Exposure is a
    declared relative scale, gain converts mean units to ``spec.output_unit``
    and offset has that output unit.

    Request at least one destination declared active in ``spec``.
    ``out_grad_mean`` is binary32 ``(P,)``, and each shared scalar destination
    is binary32 or binary64 ``(1,)``. Their units are the seeded scalar per
    corresponding input unit. Fitted gain and offset must be shared, and gain
    and exposure cannot both be active. Shared reductions use binary64 trees,
    so a binary64 scalar destination can preserve a derivative needed by a
    subsequent parameter chart.

    The caller owns disjoint contiguous input/output buffers on the workspace
    device. The call overwrites requested gradients, leaves inputs unchanged
    and returns ``None`` on the owning stream. ``validate=False`` defers the
    diagnostic checkpoint to ``workspace.check_status()`` before gradient use.

    Example::

        import warp as wp
        from dpt.detector import CalibrationSpec, prepare_detector

        workspace = prepare_detector(max_pixels=2, device="cpu")
        mean = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
        gain = wp.array([2.0], dtype=wp.float32, device="cpu")
        exposure = wp.array([3.0], dtype=wp.float32, device="cpu")
        offset = wp.array([1.0], dtype=wp.float32, device="cpu")

        from dpt.detector import calibration_vjp
        cotangent = wp.array([1.0, 1.0], dtype=wp.float32, device="cpu")
        mean_gradient = wp.empty(2, dtype=wp.float32, device="cpu")
        gain_gradient = wp.empty(1, dtype=wp.float64, device="cpu")
        calibration_vjp(
            mean, gain, exposure, offset, cotangent=cotangent,
            out_grad_mean=mean_gradient, out_grad_gain=gain_gradient,
            spec=CalibrationSpec(output_unit="ADU", active_gain=True), workspace=workspace,
        )
        assert mean_gradient.numpy().tolist() == [6.0, 6.0]
        assert gain_gradient.numpy().tolist() == [9.0]

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_mean: Any = cast(Any, mean)
    _api_gain: Any = cast(Any, gain)
    _api_exposure: Any = cast(Any, exposure)
    _api_offset: Any = cast(Any, offset)
    _api_cotangent: Any = cast(Any, cotangent)
    _api_out_grad_mean: Any = cast(Any, out_grad_mean)
    _api_out_grad_gain: Any = cast(Any, out_grad_gain)
    _api_out_grad_exposure: Any = cast(Any, out_grad_exposure)
    _api_out_grad_offset: Any = cast(Any, out_grad_offset)
    require_no_tape()
    ctx = workspace.context
    ctx.assert_stream(stream)
    pixels, reads = _calibration_inputs(
        _api_mean, _api_gain, _api_exposure, _api_offset, spec, workspace
    )
    ctx.array(_api_cotangent, "cotangent", dtype=ctx.wp.float32, shape=(pixels,))
    writes: list[tuple[str, Any]] = []
    for name, output, active, size in (
        ("mean", _api_out_grad_mean, spec.active_mean, pixels),
        ("gain", _api_out_grad_gain, spec.active_gain, 1),
        ("exposure", _api_out_grad_exposure, spec.active_exposure, 1),
        ("offset", _api_out_grad_offset, spec.active_offset, 1),
    ):
        if output is not None:
            if not active:
                raise ContractError(f"calibration {name} was declared fixed")
            dtype = ctx.wp.float32
            if name != "mean":
                dtype = getattr(output, "dtype", None)
                if dtype not in (ctx.wp.float32, ctx.wp.float64):
                    raise ContractError("shared calibration gradients must be binary32 or binary64")
            ctx.array(output, f"out_grad_{name}", dtype=dtype, shape=(size,))
            writes.append((f"out_grad_{name}", output))
    if not writes:
        raise ContractError("request at least one active calibration gradient")
    ctx.disjoint([*reads, ("cotangent", _api_cotangent)], writes)
    with ctx.scope():
        if validate:
            workspace._values(
                [
                    (_api_mean, "nonnegative"),
                    (_api_gain, "positive"),
                    (_api_exposure, "nonnegative"),
                    (_api_offset, "finite"),
                    (_api_cotangent, "finite"),
                ]
            )
        if pixels and _api_out_grad_mean is not None:
            ctx.wp.launch(
                workspace._kernels.get_calibration_pixel_vjp(spec.shared_gain, True),
                dim=pixels,
                inputs=[_api_gain, _api_exposure, _api_cotangent],
                outputs=[_api_out_grad_mean, workspace._status],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        groups = min(workspace.reduction_groups, max(1, (pixels + 255) // 256))
        for kind, output in enumerate(
            (_api_out_grad_gain, _api_out_grad_exposure, _api_out_grad_offset)
        ):
            if output is None:
                continue
            if not pixels:
                output.zero_()
                continue
            ctx.wp.launch_tiled(
                workspace._kernels.get_calibration_partials(spec.shared_gain, kind),
                dim=groups,
                block_dim=256,
                inputs=[_api_mean, _api_gain, _api_exposure, _api_cotangent, pixels, groups],
                outputs=[workspace._partials],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
            ctx.wp.launch(
                workspace._checks.get_finish_shared(output.dtype == ctx.wp.float64),
                dim=1,
                inputs=[workspace._partials, groups],
                outputs=[output, workspace._status],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            workspace.consume_status()


@dataclass(frozen=True, slots=True)
class BlurSpec:
    """Finite detector response: Bx[r,c] = sum_ij h[i,j] x[r+i-a,c+j-b].

    Samples outside the detector are zero. The non-negative odd-sized kernel
    sums to at most one, representing a passive spread with optional loss.
    The edge is never renormalised; transpose therefore reverses the offsets.
    """

    height: int
    width: int
    kernel_height: int
    kernel_width: int
    weights: tuple[float, ...]
    provenance: Provenance | None = None
    boundary: Literal["zero"] = "zero"

    def __post_init__(self) -> None:
        for name in ("height", "width", "kernel_height", "kernel_width"):
            integer(getattr(self, name), name, minimum=1)
        if self.height * self.width > 2**31 - 1:
            raise ContractError("detector dimensions exceed the flat index range")
        if self.kernel_height % 2 != 1 or self.kernel_width % 2 != 1:
            raise ContractError("spatial kernel dimensions must be odd, with a unique centre")
        finite_tuple(self.weights, "blur weights")
        if len(self.weights) != self.kernel_height * self.kernel_width:
            raise ContractError("blur weights must match the declared kernel footprint")
        # Weights are uploaded as binary32. A PSF normalised in that precision
        # can straddle one after the host's accurate sum; allow four binary32
        # ulps of normalisation error without rescaling a caller's stencil.
        if math.fsum(self.weights) > 1.0 + 4 * 2**-23:
            raise ContractError("passive blur weights must sum to at most one")
        if self.boundary != "zero":
            raise ContractError("only explicit zero-extension boundaries are supported")
        if self.provenance is not None and not isinstance(self.provenance, Provenance):
            raise ContractError(
                "spatial-response provenance must be a Provenance record when supplied"
            )


@dataclass(frozen=True, slots=True)
class SeparableBlurSpec:
    """Zero-extended passive response with h[i,j]=vertical[i]*horizontal[j].

    Both factors remain in their supplied scale; only their product's total
    mass is constrained. The implementation retains a binary64 intermediate.
    """

    height: int
    width: int
    vertical: tuple[float, ...]
    horizontal: tuple[float, ...]
    provenance: Provenance | None = None
    boundary: Literal["zero"] = "zero"

    def __post_init__(self) -> None:
        integer(self.height, "height", minimum=1)
        integer(self.width, "width", minimum=1)
        if self.height * self.width > 2**31 - 1:
            raise ContractError("detector dimensions exceed the flat index range")
        for name in ("vertical", "horizontal"):
            values = getattr(self, name)
            finite_tuple(values, name)
            if len(values) % 2 != 1:
                raise ContractError("separable blur factors must have odd lengths")
        mass = math.fsum(self.vertical) * math.fsum(self.horizontal)
        if not math.isfinite(mass) or mass > 1.0 + 4 * 2**-23:
            raise ContractError("passive blur weights must sum to at most one")
        if self.boundary != "zero":
            raise ContractError("only explicit zero-extension boundaries are supported")
        if self.provenance is not None and not isinstance(self.provenance, Provenance):
            raise ContractError(
                "spatial-response provenance must be a Provenance record when supplied"
            )


@dataclass(slots=True)
class BlurWorkspace:
    spec: BlurSpec | SeparableBlurSpec
    detector: DetectorWorkspace
    _weights: Any = field(repr=False)
    _intermediate: Any = field(default=None, repr=False)

    @property
    def scratch_bytes(self) -> int:
        return self.detector.scratch_bytes + 8 * (
            int(self._weights.size)
            + (int(self._intermediate.size) if self._intermediate is not None else 0)
        )

    def clear_status(self) -> None:
        self.detector.clear_status()

    @property
    def host_bytes(self) -> int:
        return self.detector.host_bytes

    def check_status(self) -> None:
        self.detector.check_status()

    def consume_status(self) -> None:
        self.detector.consume_status()


def prepare_blur(
    spec: BlurSpec | SeparableBlurSpec,
    *,
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
) -> BlurWorkspace:
    """Upload the fixed stencil in binary64, preserving its declared host coefficients.

    Stencils are small immutable calibration data. Narrowing their coefficients
    to binary32 can erase a positive tail whose product with a bright input is
    representable, and can change passive total mass. Images still use binary32.
    """
    detector = prepare_detector(max_pixels=spec.height * spec.width, device=device, stream=stream)
    ctx = detector.context
    with ctx.scope():
        values = (
            spec.vertical + spec.horizontal if isinstance(spec, SeparableBlurSpec) else spec.weights
        )
        weights = ctx.wp.array(values, dtype=ctx.wp.float64, device=ctx.device)
        intermediate = (
            ctx.wp.empty(spec.height * spec.width, dtype=ctx.wp.float64, device=ctx.device)
            if isinstance(spec, SeparableBlurSpec)
            else None
        )
    return BlurWorkspace(spec, detector, weights, intermediate)


def _apply_blur(
    source: Any,
    output: Any,
    workspace: BlurWorkspace,
    transpose: bool,
    stream: wp.Stream | None,
    validate: bool,
) -> None:
    require_no_tape()
    spec, detector = workspace.spec, workspace.detector
    ctx = detector.context
    ctx.assert_stream(stream)
    size = spec.height * spec.width
    ctx.array(source, "source", dtype=ctx.wp.float32, shape=(size,))
    ctx.array(output, "output", dtype=ctx.wp.float32, shape=(size,))
    ctx.disjoint([("source", source), ("weights", workspace._weights)], [("output", output)])
    with ctx.scope():
        if validate:
            detector._values([(source, "finite")])
        if isinstance(spec, SeparableBlurSpec):
            # (V H)^T = H^T V^T; zero extension is identical on both passes.
            axes = (True, False) if transpose else (False, True)
            for index, vertical in enumerate(axes):
                length = len(spec.vertical) if vertical else len(spec.horizontal)
                offset = 0 if vertical else len(spec.vertical)
                ctx.wp.launch(
                    detector._kernels.get_separable_blur(length, vertical, transpose, index == 1),
                    dim=size,
                    inputs=[
                        source if index == 0 else workspace._intermediate,
                        workspace._weights,
                        offset,
                        spec.height,
                        spec.width,
                    ],
                    outputs=[workspace._intermediate if index == 0 else output, detector._status],
                    device=ctx.device,
                    stream=ctx.stream,
                    record_tape=False,
                )
        else:
            ctx.wp.launch(
                detector._kernels.get_blur_dynamic(
                    spec.kernel_height, spec.kernel_width, transpose
                ),
                dim=size,
                inputs=[source, workspace._weights, spec.height, spec.width],
                outputs=[output, detector._status],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            detector.consume_status()


def blur(
    source: wp.array[wp.float32] | wp.array[wp.float64],
    *,
    out_signal: wp.array[wp.float32],
    workspace: BlurWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Overwrite ``y[r,c] = sum_ij h[i,j] x[r+i-a,c+j-b]`` with zero extension.

    ``source`` and caller-owned ``out_signal`` are flat binary32 ``(H*W,)``
    arrays in row-major order for the workspace's declared image height/width.
    Signed finite signals are accepted. The nonnegative odd-sized stencil has
    mass at most one and is retained in binary64. A separable stencil evaluates
    the same declared response through two passes with a binary64 intermediate.
    Image sums use binary64 before final binary32 storage. Output units match
    the input because stencil coefficients are dimensionless.

    Samples outside the image are zero and cropped edges are not renormalised.
    Inputs and output must occupy disjoint contiguous buffers on the workspace
    device. This call overwrites the output, leaves the source unchanged and
    returns ``None`` on the owning stream. With ``validate=False``, inspect
    ``workspace.check_status()`` before consuming the result.

    Example::

        import warp as wp
        from dpt.detector import BlurSpec, prepare_blur

        workspace = prepare_blur(BlurSpec(2, 2, 1, 1, (0.5,)), device="cpu")

        from dpt.detector import blur
        source = wp.array([2.0, 4.0, 6.0, 8.0], dtype=wp.float32, device="cpu")
        output = wp.empty(4, dtype=wp.float32, device="cpu")
        blur(source, out_signal=output, workspace=workspace)
        assert output.numpy().tolist() == [1.0, 2.0, 3.0, 4.0]

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_source: Any = cast(Any, source)
    _api_out_signal: Any = cast(Any, out_signal)
    _apply_blur(_api_source, _api_out_signal, workspace, False, stream, validate)


def blur_transpose(
    cotangent: wp.array[wp.float32] | None,
    *,
    out_grad_signal: wp.array[wp.float32] | wp.array[wp.float64],
    workspace: BlurWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Overwrite ``out_grad_signal = B^T cotangent`` for the prepared blur matrix.

    The signed finite ``cotangent`` and caller-owned destination are flat
    binary32 ``(H*W,)`` arrays in row-major order. The transpose reverses
    stencil offsets with the same zero extension as ``blur``. For a separable
    response it also reverses the pass order, so this is the algebraic transpose
    of the discrete forward operator including its cropped boundaries.

    Coefficients and intermediate sums use binary64 before binary32 output,
    and the cotangent's units are preserved. All buffers must be contiguous,
    disjoint and on the workspace device. The input remains unchanged, the
    destination is overwritten and the call returns ``None`` on the owning
    stream. Unchecked calls require ``workspace.check_status()`` before use.

    Example::

        import warp as wp
        from dpt.detector import BlurSpec, prepare_blur

        workspace = prepare_blur(BlurSpec(2, 2, 1, 1, (0.5,)), device="cpu")

        from dpt.detector import blur_transpose
        cotangent = wp.array([1.0, 1.0, 1.0, 1.0], dtype=wp.float32, device="cpu")
        gradient = wp.empty(4, dtype=wp.float32, device="cpu")
        blur_transpose(cotangent=cotangent, out_grad_signal=gradient, workspace=workspace)
        assert gradient.numpy().tolist() == [0.5, 0.5, 0.5, 0.5]

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_cotangent: Any = cast(Any, cotangent)
    _api_out_grad_signal: Any = cast(Any, out_grad_signal)
    _apply_blur(_api_cotangent, _api_out_grad_signal, workspace, True, stream, validate)


@dataclass(frozen=True, slots=True)
class ObservationIdentity:
    """A reproducible random experiment independent of launch tiling.

    ``seed`` is the unsigned 64-bit random-number key, independent of the
    signed image cotangents passed to deterministic reverse operators.

    Counter identity = (observation_id << 32) | global_pixel. Poisson bin k has
    domain 1 + energy_offset + k; electronic read noise has domain 0x7fffffff.
    ``read_noise_algorithm="box-muller-pairs"`` instead uses domain 0xffffffff,
    keyed by the absolute even-pixel anchor. This algorithm is part of the
    immutable experiment identity and must match the prepared workspace policy.
    Transport owns domain 0 and its bounded 0x8/0xc rejection namespaces. Reusing an identity is
    common random numbers, not an independent replication. Keep bin identities
    stable when batching, and assign a fresh observation_id for a fresh image.
    ``draw_budget`` bounds rejection proposals, never the sampled count. Below
    rate 10 an inverse-CDF draw uses one proposal with a fixed internal work
    bound; higher rates use at most ``draw_budget`` transformed proposals.
    """

    seed: int
    observation_id: int
    pixel_offset: int = 0
    energy_offset: int = 0
    draw_budget: int = 256
    read_noise_algorithm: Literal["scalar-box-muller", "box-muller-pairs"] = "scalar-box-muller"

    def __post_init__(self) -> None:
        integer(self.seed, "seed", maximum=2**64 - 1)
        integer(self.observation_id, "observation_id", maximum=2**32 - 1)
        integer(self.pixel_offset, "pixel_offset", maximum=2**32 - 1)
        integer(self.energy_offset, "energy_offset", maximum=2**31 - 3)
        integer(self.draw_budget, "draw_budget", minimum=1)
        GaussianReadNoiseExecution(self.read_noise_algorithm)

    def validate_extent(self, pixels: int, energies: int = 1) -> None:
        integer(pixels, "pixels")
        integer(energies, "energies", minimum=1)
        if self.pixel_offset + pixels > 2**32:
            raise ContractError("pixel identities would wrap")
        if self.energy_offset + energies > 2**31 - 2:
            raise ContractError("Poisson domains would overlap the read-noise/transport namespace")


def sample_poisson_counts(
    expected_counts: wp.array[wp.float32] | wp.array[wp.float64],
    *,
    out_counts: wp.array[wp.uint64],
    identity: ObservationIdentity,
    workspace: DetectorWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Draw ``out_counts[p] ~ Poisson(expected_counts[p])`` independently by pixel.

    The finite nonnegative photon-count means are a flat binary32/binary64
    ``(P,)`` array with values in ``[0, 10^9]`` and
    ``P <= workspace.max_pixels``. The caller-owned destination is uint64
    ``(P,)`` in observed photons, and a zero mean returns zero. This observation
    generator acts on count-domain means and has no derivative through the draw.

    ``ObservationIdentity`` fixes the random key, observation and global pixel
    addresses, with the Poisson bin selected by ``energy_offset``. Reusing an
    identity gives common random numbers, while a fresh observation ID gives a
    fresh image. Small means use inverse-CDF sampling and larger means use
    bounded rejection with binary64 acceptance arithmetic. ``draw_budget``
    bounds rejection proposals, never the sampled count. Budget exhaustion
    invalidates the whole realisation and its diagnostic persists until
    ``workspace.consume_status()`` or ``workspace.clear_status()``.

    Input and output must be contiguous, disjoint and on the workspace device.
    The input remains unchanged, the destination is overwritten and the call
    returns ``None`` on the owning stream. ``validate=False`` requires
    ``workspace.check_status()`` before accepting the output as an observation.

    Example::

        import warp as wp
        from dpt.detector import ObservationIdentity, prepare_detector, sample_poisson_counts

        workspace = prepare_detector(max_pixels=2, device="cpu")
        means = wp.array([0.0, 4.0], dtype=wp.float64, device="cpu")
        counts = wp.empty(2, dtype=wp.uint64, device="cpu")
        sample_poisson_counts(means, out_counts=counts,
                              identity=ObservationIdentity(123, 7), workspace=workspace)
        observed = counts.numpy()
        assert int(observed[0]) == 0

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_expected_counts: Any = cast(Any, expected_counts)
    _api_out_counts: Any = cast(Any, out_counts)
    require_no_tape()
    ctx = workspace.context
    ctx.assert_stream(stream)
    rate_dtype = getattr(_api_expected_counts, "dtype", None)
    if rate_dtype not in (ctx.wp.float32, ctx.wp.float64):
        raise ContractError("expected counts must be binary32 or binary64")
    pixels = ctx.array(_api_expected_counts, "expected_counts", dtype=rate_dtype)
    if pixels > workspace.max_pixels:
        raise ContractError("expected_counts exceeds workspace capacity")
    identity.validate_extent(pixels)
    ctx.array(_api_out_counts, "out_counts", dtype=ctx.wp.uint64, shape=(pixels,))
    ctx.disjoint([("expected_counts", _api_expected_counts)], [("out_counts", _api_out_counts)])
    with ctx.scope():
        if validate:
            workspace._values([(_api_expected_counts, "rate")])
        if pixels:
            ctx.wp.launch(
                workspace._kernels.get_poisson_counts(rate_dtype == ctx.wp.float64),
                dim=pixels,
                inputs=[
                    _api_expected_counts,
                    ctx.wp.uint64(identity.seed),
                    ctx.wp.uint32(identity.observation_id),
                    ctx.wp.uint32(identity.pixel_offset),
                    ctx.wp.uint32(identity.energy_offset),
                    identity.draw_budget,
                ],
                outputs=[_api_out_counts, workspace._status],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            workspace.check_status()


def sample_compound_poisson(
    detected_bin_means: wp.array[wp.float32] | wp.array[wp.float64],
    photon_scores: wp.array[wp.float32],
    *,
    energies: int,
    shared_scores: bool,
    out_signal: wp.array[wp.float32],
    identity: ObservationIdentity,
    workspace: DetectorWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Draw ``Y[p] = sum_k score[k,p] N[k,p]`` with independent Poisson bin counts.

    ``N[k,p] ~ Poisson(detected_bin_means[k,p])``. The rate buffer is flat
    energy-major ``(K*P,)`` binary32/binary64, with each finite mean in
    ``[0, 10^9]`` and ``K=energies``. ``photon_scores`` is finite nonnegative
    binary32 ``(K,)`` when ``shared_scores=True``, otherwise flat energy-major
    ``(K*P,)``. It supplies deterministic output units per detected photon.
    The caller-owned ``out_signal`` is binary32 ``(P,)`` in those output units,
    with ``P <= workspace.max_pixels``.

    At fixed scores, ``E[Y[p]]=sum_k rate[k,p] score[k,p]`` and
    ``Var[Y[p]]=sum_k rate[k,p] score[k,p]^2`` before spatial spreading.
    Bin sums use compensated binary64 arithmetic before output rounding.
    Within-bin random deposition requires its own observation law. Apply
    spatial spreading and independent electronic noise in the order specified
    by the detector model.

    ``ObservationIdentity`` retains the same pixel/bin counter addresses under
    batching, and a fresh observation ID defines a fresh realisation. Rejection
    budget exhaustion invalidates the entire draw and leaves a diagnostic
    pending until explicit consumption or clearing. All buffers are contiguous,
    disjoint and on the workspace device. Inputs remain unchanged, the output
    is overwritten and the call returns ``None`` on the owning stream.
    ``validate=False`` requires ``workspace.check_status()`` before observation use.

    Example::

        import warp as wp
        from dpt.detector import ObservationIdentity, prepare_detector, sample_compound_poisson

        workspace = prepare_detector(max_pixels=1, device="cpu")
        rates = wp.array([0.0, 4.0], dtype=wp.float64, device="cpu")
        scores = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
        signal = wp.empty(1, dtype=wp.float32, device="cpu")
        sample_compound_poisson(rates, scores, energies=2, shared_scores=True,
                                out_signal=signal, identity=ObservationIdentity(123, 7),
                                workspace=workspace)
        observed = float(signal.numpy()[0])
        assert observed >= 0.0 and observed % 2.0 == 0.0

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_detected_bin_means: Any = cast(Any, detected_bin_means)
    _api_photon_scores: Any = cast(Any, photon_scores)
    _api_out_signal: Any = cast(Any, out_signal)
    require_no_tape()
    integer(energies, "energies", minimum=1, maximum=65536)
    if type(shared_scores) is not bool:
        raise ContractError("shared_scores must be a boolean")
    ctx = workspace.context
    ctx.assert_stream(stream)
    rate_dtype = getattr(_api_detected_bin_means, "dtype", None)
    if rate_dtype not in (ctx.wp.float32, ctx.wp.float64):
        raise ContractError("detected bin means must be binary32 or binary64")
    size = ctx.array(_api_detected_bin_means, "detected_bin_means", dtype=rate_dtype)
    if size % energies:
        raise ContractError("detected bin means must have energy-major shape (K,P)")
    pixels = size // energies
    if pixels > workspace.max_pixels or size > 2**31 - 1:
        raise ContractError("detected bin means exceed the declared pixel/index capacity")
    identity.validate_extent(pixels, energies)
    ctx.array(
        _api_photon_scores,
        "photon_scores",
        dtype=ctx.wp.float32,
        shape=(energies * (1 if shared_scores else pixels),),
    )
    ctx.array(_api_out_signal, "out_signal", dtype=ctx.wp.float32, shape=(pixels,))
    ctx.disjoint(
        [("detected_bin_means", _api_detected_bin_means), ("photon_scores", _api_photon_scores)],
        [("out_signal", _api_out_signal)],
    )
    with ctx.scope():
        if validate:
            workspace._compound_values(_api_detected_bin_means, _api_photon_scores)
        if pixels:
            ctx.wp.launch(
                workspace._kernels.get_compound_poisson(
                    energies, shared_scores, rate_dtype == ctx.wp.float64
                ),
                dim=pixels,
                inputs=[
                    _api_detected_bin_means,
                    _api_photon_scores,
                    pixels,
                    ctx.wp.uint64(identity.seed),
                    ctx.wp.uint32(identity.observation_id),
                    ctx.wp.uint32(identity.pixel_offset),
                    ctx.wp.uint32(identity.energy_offset),
                    identity.draw_budget,
                ],
                outputs=[_api_out_signal, workspace._status],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            workspace.check_status()


def add_gaussian_read_noise(
    signal: wp.array[wp.float32],
    *,
    standard_deviation: float,
    out_signal: wp.array[wp.float32],
    identity: ObservationIdentity,
    workspace: DetectorWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Draw ``out_signal[p] = signal[p] + sigma Z[p]`` with independent standard normals.

    The input and caller-owned output are flat binary32 ``(P,)`` arrays with
    ``P <= workspace.max_pixels``. Signals may be signed and must be finite.
    ``standard_deviation=sigma`` is a finite nonnegative host scalar in the
    signal's units, so the added noise has variance ``sigma^2`` in squared
    signal units. Binary64 Box-Muller arithmetic is rounded into binary32 output.
    No clipping is applied to either the noise or the resulting signal.

    ``ObservationIdentity`` fixes pixel counter addresses in the electronic
    read-noise domain. Reusing an identity reproduces the same noise, while a
    fresh observation ID gives a fresh realisation. Input and output must be
    contiguous, disjoint and on the workspace device. The source remains
    unchanged, the destination is overwritten and the call returns ``None``
    on the owning stream. Diagnostics persist until consumption or clearing,
    and ``validate=False`` requires ``workspace.check_status()`` before use.

    The workspace and observation identity select the same immutable noise
    algorithm. ``scalar-box-muller`` retains the historical pixel counters and
    cosine member. ``box-muller-pairs`` emits both independent members at each
    absolute even-pixel anchor, reusing the Philox block, logarithm and square
    root. Odd offsets, tails and chunks reproduce the same absolute image.
    Set ``GaussianReadNoiseExecution("box-muller-pairs")`` at preparation and
    ``ObservationIdentity(..., read_noise_algorithm="box-muller-pairs")`` at
    sampling. Repeated identities provide common random numbers.

    Example::

        import warp as wp
        from dpt.detector import ObservationIdentity, add_gaussian_read_noise, prepare_detector

        workspace = prepare_detector(max_pixels=2, device="cpu")
        signal = wp.array([3.0, 5.0], dtype=wp.float32, device="cpu")
        first = wp.empty(2, dtype=wp.float32, device="cpu")
        second = wp.empty(2, dtype=wp.float32, device="cpu")
        identity = ObservationIdentity(123, 7)
        for output in (first, second):
            add_gaussian_read_noise(signal, standard_deviation=0.5, out_signal=output,
                                    identity=identity, workspace=workspace)
        assert first.numpy().tolist() == second.numpy().tolist()

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_signal: Any = cast(Any, signal)
    _api_out_signal: Any = cast(Any, out_signal)
    require_no_tape()
    sigma = finite_scalar(standard_deviation, "standard_deviation", minimum=0)
    ctx = workspace.context
    ctx.assert_stream(stream)
    pixels = workspace._pixels(_api_signal, "signal")
    identity.validate_extent(pixels)
    if identity.read_noise_algorithm != workspace.read_noise_execution.algorithm:
        raise ContractError("observation identity and workspace read-noise policies differ")
    ctx.array(_api_out_signal, "out_signal", dtype=ctx.wp.float32, shape=(pixels,))
    ctx.disjoint([("signal", _api_signal)], [("out_signal", _api_out_signal)])
    with ctx.scope():
        if validate:
            workspace._values([(_api_signal, "finite")])
        if pixels:
            paired = workspace.read_noise_execution.algorithm == "box-muller-pairs"
            ctx.wp.launch(
                workspace._kernels.gaussian_read_noise_pairs
                if paired
                else workspace._kernels.gaussian_read_noise,
                dim=(pixels + (identity.pixel_offset & 1) + 1) // 2 if paired else pixels,
                inputs=[
                    _api_signal,
                    sigma,
                    ctx.wp.uint64(identity.seed),
                    ctx.wp.uint32(identity.observation_id),
                    ctx.wp.uint32(identity.pixel_offset),
                ],
                outputs=[_api_out_signal, workspace._status],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            workspace.check_status()
