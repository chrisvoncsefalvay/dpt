"""Prepared flat/dark correction and logarithmic attenuation with an explicit VJP."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._runtime import (
    DeviceContext,
    StatusReadback,
    load_kernels,
    prepare_context,
    require_no_tape,
)
from dpt.contracts import ContractError, integer, raise_device_status

if TYPE_CHECKING:
    import warp as wp

    FloatArray = wp.array[wp.float32] | wp.array[wp.float64]

CalibrationPrecision = Literal["float32", "float64"]


@dataclass(frozen=True, slots=True)
class LogCalibrationResult:
    """Optical depth ``L = -log((I-D)/(I0-D))``, with the input image's shape.

    ``optical_depth`` uses the declared binary32 or binary64 precision and is
    dimensionless. Its storage belongs to the calibration operator and is reused
    by the next forward call. Clone the array to retain an earlier image.
    """

    optical_depth: FloatArray
    precision: CalibrationPrecision


@dataclass(frozen=True, slots=True)
class LogCalibrationGradients:
    """Image-shaped cotangents of ``sum_p grad_depth[p] L[p]``.

    For ``A=I-D`` and ``B=I0-D``, the three arrays contain ``-g/A``, ``g/B``
    and ``g(1/A-1/B)`` for signal, flat and dark respectively. Their precision
    matches the forward image. Storage is reused by the next VJP.
    """

    signal: FloatArray
    flat: FloatArray
    dark: FloatArray


class LogCalibration:
    """Evaluate ``L[p] = -log((I[p]-D[p])/(I0[p]-D[p]))`` and its VJP.

    Signal, flat and dark images have the same contiguous shape and use the
    declared binary32 or binary64 dtype on one explicit device. All three
    frames use the same intensity units. The corrected signal and flat must
    be finite and strictly positive; a signal above the flat gives negative
    optical depth, as can occur with noisy observations.

    Preparation copies and validates fixed flat/dark frames once. Each forward
    retains its own signal copy for subsequent VJPs, so editing the caller's
    frames preserves the retained calculation. Arithmetic uses binary64 before
    one final storage rounding. Small contrasts use cancellation-safe log1p;
    larger contrasts use a difference of logarithms.

    Calls run sequentially on the owning stream. Checked calls deliver one
    completion status. ``check=False`` queues asynchronous work; call
    ``check_status()`` before consuming its result. Diagnostics persist until
    explicitly consumed or cleared.

    Example::

        operator = LogCalibration(flat, dark, precision="float64", device="cpu")
        result = operator.forward(signal)
        gradients = operator.vjp(grad_depth)
    """

    def __init__(
        self,
        flat: FloatArray,
        dark: FloatArray,
        *,
        precision: CalibrationPrecision = "float32",
        device: str = "cuda:0",
        stream: Any = None,
    ) -> None:
        require_no_tape()
        if precision not in ("float32", "float64"):
            raise ContractError("calibration precision must be float32 or float64")
        self.context: DeviceContext = prepare_context(device=device, stream=stream)
        ctx = self.context
        if not isinstance(flat, ctx.wp.array):
            raise ContractError("flat must be a Warp array on the calibration device")
        self._shape = cast(tuple[int, ...], cast(Any, flat).shape)
        self._pixels = integer(int(flat.size), "calibration pixels", minimum=1)
        self._precision: CalibrationPrecision = precision
        self._dtype: Any = getattr(ctx.wp, precision)
        ctx.array(flat, "flat", dtype=self._dtype, shape=self._shape, ndim=len(self._shape))
        ctx.array(dark, "dark", dtype=self._dtype, shape=self._shape, ndim=len(self._shape))
        self._kernels: Any = load_kernels("dpt.kernels.log_calibration")
        self._lock = RLock()
        self._has_forward = False
        self._pending = False
        self._failed = False
        with ctx.scope():
            self._flat = cast(
                "FloatArray", ctx.wp.empty(self._shape, dtype=self._dtype, device=ctx.device)
            )
            self._dark = cast("FloatArray", ctx.wp.empty_like(self._flat))
            self._signal = cast("FloatArray", ctx.wp.empty_like(self._flat))
            self._seed = cast("FloatArray", ctx.wp.empty_like(self._flat))
            self._reference = ctx.wp.empty(self._shape, dtype=ctx.wp.float64, device=ctx.device)
            depth = cast("FloatArray", ctx.wp.empty_like(self._flat))
            gradients = tuple(cast("FloatArray", ctx.wp.empty_like(self._flat)) for _ in range(3))
            self._status = ctx.wp.zeros(1, dtype=ctx.wp.int32, device=ctx.device)
            ctx.wp.copy(self._flat, flat, stream=ctx.stream)
            ctx.wp.copy(self._dark, dark, stream=ctx.stream)
        self._flat_view = cast("FloatArray", self._flat.flatten())
        self._dark_view = cast("FloatArray", self._dark.flatten())
        self._signal_view = cast("FloatArray", self._signal.flatten())
        self._seed_view = cast("FloatArray", self._seed.flatten())
        self._reference_view = self._reference.flatten()
        self._depth_view = cast("FloatArray", depth.flatten())
        self._gradient_views = tuple(
            cast("FloatArray", gradient.flatten()) for gradient in gradients
        )
        self._result = LogCalibrationResult(depth, precision)
        self._gradients = LogCalibrationGradients(*gradients)
        self._status_readback = StatusReadback(ctx, self._status)
        with ctx.scope():
            ctx.wp.launch(
                self._kernels.get_prepare(precision == "float64"),
                dim=self._pixels,
                inputs=[self._flat_view, self._dark_view],
                outputs=[self._reference_view, self._status],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        self._pending = True
        self.check_status()

    @property
    def precision(self) -> CalibrationPrecision:
        return self._precision

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def device(self) -> Any:
        return self.context.device

    @property
    def stream(self) -> Any:
        return self.context.stream

    @property
    def scratch_bytes(self) -> int:
        """Owned device storage for snapshots, outputs, gradients and diagnostics."""
        element_bytes = 8 if self.precision == "float64" else 4
        return self._pixels * (8 * element_bytes + 8) + 4

    @property
    def host_staging_bytes(self) -> int:
        return self._status_readback.host_bytes

    def _raise_status(self, code: int) -> None:
        raise_device_status(
            code,
            "flat/dark log calibration",
            messages={
                1: (
                    "frames and cotangents must be finite; "
                    "corrected signal and flat must be positive"
                ),
                2: "optical depth or gradient cannot be represented in the declared precision",
            },
        )

    def check_status(self) -> None:
        """Complete owning-stream work and deliver persistent domain/range errors."""
        with self._lock:
            code = self._status_readback.read()
            if code:
                self._failed = True
                self._has_forward = False
                self._raise_status(code)
            self._pending = self._failed = False

    def consume_status(self) -> None:
        """Deliver and reset a completed diagnostic; failed outputs are discarded."""
        with self._lock:
            code = self._status_readback.read(reset=True)
            self._pending = self._failed = False
            if code:
                self._has_forward = False
            self._raise_status(code)

    def clear_status(self) -> None:
        """Discard accumulated diagnostics and the retained forward explicitly."""
        with self._lock, self.context.scope():
            self._status.zero_()
            self._pending = self._failed = self._has_forward = False

    def _before_call(self, stream: Any, check: bool) -> None:
        require_no_tape()
        if type(check) is not bool:
            raise ContractError("check must be a boolean")
        self.context.assert_stream(self.stream if stream is None else stream)
        if self._pending or self._failed:
            self.check_status()

    def forward(
        self, signal: FloatArray, *, stream: Any = None, check: bool = True
    ) -> LogCalibrationResult:
        """Return ``L = -log((signal-dark)/(flat-dark))`` in ``optical_depth``.

        ``signal`` is a contiguous image with exactly the operator's prepared
        shape, including any batch axes, and its declared binary32/binary64 dtype.
        Signal, flat and dark use the same intensity units. Both corrected frames
        must be finite and strictly positive, while signal above flat gives a
        negative optical depth. Arithmetic uses binary64 and cancellation-safe
        logarithms before rounding the dimensionless result to the declared dtype.

        The returned ``LogCalibrationResult.optical_depth`` has the same shape and
        precision as the input. Its storage belongs to the operator and is reused
        by the next forward, so clone it to retain an earlier result. The operator
        copies the signal for subsequent VJPs and leaves the caller's image unchanged.
        Calls run sequentially on the owning stream. ``check=False`` defers the
        completion checkpoint to ``operator.check_status()`` before result use.

        Example::

            import math
            import warp as wp
            from dpt.calibration import LogCalibration

            flat = wp.array([[10.0, 10.0]], dtype=wp.float64, device="cpu")
            dark = wp.array([[2.0, 2.0]], dtype=wp.float64, device="cpu")
            signal = wp.array([[6.0, 6.0]], dtype=wp.float64, device="cpu")
            operator = LogCalibration(flat, dark, precision="float64", device="cpu")

            result = operator.forward(signal)
            assert result.optical_depth.shape == (1, 2)
            assert math.isclose(float(result.optical_depth.numpy()[0, 0]), math.log(2.0))

        """
        with self._lock:
            self._before_call(stream, check)
            ctx = self.context
            ctx.array(signal, "signal", dtype=self._dtype, shape=self._shape, ndim=len(self._shape))
            self._has_forward = False
            with ctx.scope():
                ctx.wp.copy(self._signal, signal, stream=ctx.stream)
                ctx.wp.launch(
                    self._kernels.get_forward(self.precision == "float64"),
                    dim=self._pixels,
                    inputs=[
                        self._signal_view,
                        self._flat_view,
                        self._dark_view,
                        self._reference_view,
                    ],
                    outputs=[self._depth_view, self._status],
                    device=ctx.device,
                    stream=ctx.stream,
                    record_tape=False,
                )
            self._pending = self._has_forward = True
            if check:
                self.check_status()
            return self._result

    def vjp(
        self, grad_depth: FloatArray, *, stream: Any = None, check: bool = True
    ) -> LogCalibrationGradients:
        """Return ``(-g/A, g/B, g(1/A-1/B))`` for signal, flat and dark.

        Here ``A=signal-dark``, ``B=flat-dark`` and ``g=grad_depth``. The signed
        cotangent is a finite contiguous image with the successful forward's exact shape
        and binary32/binary64 dtype, including its batch axes. The retained signal
        and fixed flat/dark snapshots supply the current derivatives, and a
        successful forward is required before this call.

        The returned ``LogCalibrationGradients.signal``, ``flat`` and ``dark`` each
        have the input image's shape and precision, with units of the seeded scalar
        per intensity unit. Arithmetic uses binary64 before final storage rounding.
        The operator owns these three arrays and reuses them on the next VJP, so
        clone any derivative that must survive that call. The cotangent remains
        unchanged. Calls run sequentially on the owning stream, and ``check=False``
        requires ``operator.check_status()`` before consuming the gradients.

        Example::

            import math
            import warp as wp
            from dpt.calibration import LogCalibration

            flat = wp.array([[10.0, 10.0]], dtype=wp.float64, device="cpu")
            dark = wp.array([[2.0, 2.0]], dtype=wp.float64, device="cpu")
            signal = wp.array([[6.0, 6.0]], dtype=wp.float64, device="cpu")
            operator = LogCalibration(flat, dark, precision="float64", device="cpu")

            operator.forward(signal)
            cotangent = wp.array([[1.0, 1.0]], dtype=wp.float64, device="cpu")
            gradients = operator.vjp(cotangent)
            assert gradients.signal.numpy().tolist() == [[-0.25, -0.25]]
            assert gradients.flat.numpy().tolist() == [[0.125, 0.125]]
            assert gradients.dark.numpy().tolist() == [[0.125, 0.125]]

        """
        with self._lock:
            self._before_call(stream, check)
            if not self._has_forward:
                raise ContractError("run a successful calibration forward before its VJP")
            ctx = self.context
            ctx.array(
                grad_depth,
                "grad_depth",
                dtype=self._dtype,
                shape=self._shape,
                ndim=len(self._shape),
            )
            with ctx.scope():
                ctx.wp.copy(self._seed, grad_depth, stream=ctx.stream)
                ctx.wp.launch(
                    self._kernels.get_vjp(self.precision == "float64"),
                    dim=self._pixels,
                    inputs=[
                        self._signal_view,
                        self._flat_view,
                        self._dark_view,
                        self._reference_view,
                        self._seed_view,
                    ],
                    outputs=[*self._gradient_views, self._status],
                    device=ctx.device,
                    stream=ctx.stream,
                    record_tape=False,
                )
            self._pending = True
            if check:
                self.check_status()
            return self._gradients

    def warm(self) -> tuple[str, ...]:
        """Compile forward/VJP variants on the owning device using existing scratch."""
        with self._lock:
            self._before_call(self.stream, True)
            double = self.precision == "float64"
            kernels = (
                self._kernels.get_prepare(double),
                self._kernels.get_forward(double),
                self._kernels.get_vjp(double),
            )
            with self.context.scope():
                for kernel in kernels:
                    self.context.wp.load_module(kernel.module, device=self.device)
            return tuple(kernel.module.name for kernel in kernels)


__all__ = ["LogCalibration", "LogCalibrationGradients", "LogCalibrationResult"]
