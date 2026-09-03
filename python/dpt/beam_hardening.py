"""Polychromatic attenuation and water-equivalent path calibration.

The supplied material table, source spectrum and detector response define the
calibration. Their provenance travels with the prepared operator.
"""

from __future__ import annotations

# Host record boundaries validate runtime values as well as typed callers.
# pyright: reportUnnecessaryIsInstance=false
import math
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Literal, cast

from dpt._runtime import StatusReadback, load_kernels, prepare_context, require_no_tape
from dpt.contracts import ContractError, finite_scalar, finite_tuple, integer, raise_device_status
from dpt.materials import MaterialTable, Provenance, validate_energy_grid

if TYPE_CHECKING:
    import warp as wp
    from warp._src.context import Device

    Array = wp.array[wp.float64]
    Stream = wp.Stream


@dataclass(frozen=True, slots=True)
class BeamHardeningSpec:
    """Fixed spectral calibration inputs in keV, mm and inverse mm.

    ``source_weights[e] * detector_response[e]`` supplies the incident detected
    weight at each energy. Its common scale cancels from the log ratio. The
    material table must identify the composition, density and coefficient
    source; use a water table for a water-equivalent calibration. Zero-weight
    nodes contribute no photons, while every active attenuation coefficient
    must be strictly positive to give an invertible calibration.
    """

    material: MaterialTable
    energies_kev: tuple[float, ...]
    source_weights: tuple[float, ...]
    detector_response: tuple[float, ...]
    maximum_length_mm: float
    edge_side: Literal["below", "above"]
    spectrum_provenance: Provenance
    response_provenance: Provenance

    def __post_init__(self) -> None:
        if not isinstance(self.material, MaterialTable) or self.material.provenance is None:
            raise ContractError("beam hardening requires a material table with provenance")
        if not isinstance(self.spectrum_provenance, Provenance) or not isinstance(
            self.response_provenance, Provenance
        ):
            raise ContractError("beam hardening requires spectrum and response provenance")
        validate_energy_grid(self.energies_kev)
        finite_tuple(self.source_weights, "source_weights", length=len(self.energies_kev))
        finite_tuple(self.detector_response, "detector_response", length=len(self.energies_kev))
        limit = finite_scalar(self.maximum_length_mm, "maximum_length_mm", minimum=0)
        if limit <= 0:
            raise ContractError("maximum_length_mm must be strictly positive")
        self.material.at_energies(self.energies_kev, edge_side=self.edge_side)


@dataclass(frozen=True, slots=True)
class BeamHardeningExecution:
    """Bounded binary64 inversion; iterations stop when its bracket is resolved.

    The default also resolves very small positive optical depths. Most physical
    inputs resolve in about 54 bisections. Exhausting the chosen bound raises a
    numerical error and preserves the diagnostic.
    """

    maximum_iterations: int = 2048
    max_owned_bytes: int = 1 << 30

    def __post_init__(self) -> None:
        integer(self.maximum_iterations, "maximum_iterations", minimum=1, maximum=4096)
        integer(self.max_owned_bytes, "max_owned_bytes", minimum=1, maximum=2**63 - 1)


_DEFAULT_EXECUTION = BeamHardeningExecution()


class WaterEquivalentCalibration:
    r"""Map material path length to optical depth and invert the spectral curve.

    For fixed incident detected weights ``w[e]`` and coefficients ``mu[e]``,
    ``p(l) = -log(sum_e w[e] exp(-mu[e] l) / sum_e w[e])``. The forward
    derivative is the transmitted-weight mean of ``mu`` and the inverse
    derivative is its reciprocal. Length is in mm, optical depth is
    dimensionless, and all arrays are contiguous binary64 with ``shape`` on the
    owning device. Fixed inputs are immutable preparation metadata.

    Weak optical depth uses expm1/log1p, while strong attenuation uses a scaled
    log-sum-exp. The inverse resolves a bracket in ``[0, maximum_length_mm]``
    without extrapolating the supplied calibration. ``forward_vjp`` differentiates
    the latest forward call and ``inverse_vjp`` the latest inverse call. Retained
    derivatives have independent storage, so editing a returned image or caller
    input preserves its VJP. Returned arrays are reused by subsequent calls.

    """

    def __init__(
        self,
        spec: BeamHardeningSpec,
        *,
        shape: tuple[int, ...],
        execution: BeamHardeningExecution = _DEFAULT_EXECUTION,
        device: str = "cuda:0",
        stream: Stream | None = None,
    ) -> None:
        require_no_tape()
        if not isinstance(spec, BeamHardeningSpec) or not isinstance(
            execution, BeamHardeningExecution
        ):
            raise ContractError("calibration requires BeamHardeningSpec and BeamHardeningExecution")
        if not isinstance(shape, tuple) or not 1 <= len(shape) <= 4:
            raise ContractError("calibration shape must be a tuple with one to four dimensions")
        for dimension in shape:
            integer(dimension, "calibration dimension", minimum=1)
        self._pixels = integer(math.prod(shape), "calibration pixels", minimum=1)
        coefficients = spec.material.at_energies(spec.energies_kev, edge_side=spec.edge_side)
        active = [
            index
            for index, (source, response) in enumerate(
                zip(spec.source_weights, spec.detector_response, strict=True)
            )
            if source > 0 and response > 0
        ]
        if not active or any(coefficients[index] <= 0 for index in active):
            raise ContractError(
                "calibration needs detected photons and strictly positive active attenuation"
            )
        active_coefficients = tuple(coefficients[index] for index in active)
        if self._pixels * 5 * 8 + len(active) * 3 * 8 + 8 > execution.max_owned_bytes:
            raise ContractError("calibration storage exceeds max_owned_bytes")
        if any(not math.isfinite(value * spec.maximum_length_mm) for value in active_coefficients):
            raise ContractError("calibration coefficient-length products must fit binary64")
        logs = tuple(
            math.log(spec.source_weights[index]) + math.log(spec.detector_response[index])
            for index in active
        )
        pivot = max(logs)
        scaled = tuple(math.exp(value - pivot) for value in logs)
        if any(value == 0 for value in scaled):
            raise ContractError("relative detected spectrum cannot be represented in binary64")
        total = math.fsum(scaled)
        weights = tuple(value / total for value in scaled)
        if any(value == 0 for value in weights):
            raise ContractError("normalised detected spectrum cannot be represented in binary64")
        log_weights = tuple(value - pivot - math.log(total) for value in logs)
        self._spec = spec
        self._execution = execution
        self._shape = shape
        self.context = prepare_context(device=device, stream=stream)
        self._kernels = load_kernels("dpt.kernels.beam_hardening")
        self._maximum_coefficient = max(active_coefficients)
        self._lock = RLock()
        self._pending = self._failed = self._has_forward = self._has_inverse = False
        ctx = self.context
        with ctx.scope():
            self._coefficients = ctx.wp.array(
                active_coefficients, dtype=ctx.wp.float64, device=ctx.device
            )
            self._weights = ctx.wp.array(weights, dtype=ctx.wp.float64, device=ctx.device)
            self._log_weights = ctx.wp.array(log_weights, dtype=ctx.wp.float64, device=ctx.device)
            self._depth = cast(
                "Array", ctx.wp.empty(shape, dtype=ctx.wp.float64, device=ctx.device)
            )
            self._length = cast("Array", ctx.wp.empty_like(self._depth))
            self._gradient = cast("Array", ctx.wp.empty_like(self._depth))
            self._forward_slope = cast("Array", ctx.wp.empty_like(self._depth))
            self._inverse_slope = cast("Array", ctx.wp.empty_like(self._depth))
            self._status = ctx.wp.zeros(1, dtype=ctx.wp.int32, device=ctx.device)
        self._status_readback = StatusReadback(ctx, self._status)

    @property
    def spec(self) -> BeamHardeningSpec:
        return self._spec

    @property
    def execution(self) -> BeamHardeningExecution:
        return self._execution

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def device(self) -> Device:
        return self.context.device

    @property
    def stream(self) -> Stream | None:
        return self.context.stream

    @property
    def scratch_bytes(self) -> int:
        """Owned device storage, excluding the four-byte host status staging."""
        return self._pixels * 5 * 8 + int(self._coefficients.size) * 3 * 8 + 4

    def _raise_status(self, code: int) -> None:
        raise_device_status(
            code,
            "water-equivalent calibration",
            messages={
                1: "length, depth and cotangents must be finite and within the calibration range",
                4: "inversion exhausted its iteration bound before resolving the binary64 bracket",
            },
        )

    def check_status(self) -> None:
        """Complete owning-stream work and deliver persistent diagnostics."""
        with self._lock:
            code = self._status_readback.read()
            if code:
                self._failed = True
                self._has_forward = self._has_inverse = False
                self._raise_status(code)
            self._pending = self._failed = False

    def consume_status(self) -> None:
        """Deliver and reset a diagnostic, discarding failed retained states."""
        with self._lock:
            code = self._status_readback.read(reset=True)
            self._pending = self._failed = False
            if code:
                self._has_forward = self._has_inverse = False
            self._raise_status(code)

    def clear_status(self) -> None:
        """Clear diagnostics and both retained derivative states explicitly."""
        with self._lock, self.context.scope():
            self._status.zero_()
            self._pending = self._failed = self._has_forward = self._has_inverse = False

    def _before_call(self, check: bool, stream: Stream | None) -> None:
        require_no_tape()
        if type(check) is not bool:
            raise ContractError("check must be a boolean")
        self.context.assert_stream(self.stream if stream is None else stream)
        if self._pending or self._failed:
            self.check_status()

    def forward(
        self, length_mm: Array, *, check: bool = True, stream: Stream | None = None
    ) -> Array:
        """Return ``p(l) = -log(sum_e w_e exp(-mu_e l) / sum_e w_e)``.

        The detected incident weight is ``w_e = source_weights[e] *
        detector_response[e]`` and ``mu_e`` is the supplied attenuation in mm⁻¹.
        ``length_mm`` contains finite lengths in ``[0, spec.maximum_length_mm]``, and the
        returned optical depth is dimensionless. Both arrays have ``self.shape``
        and contiguous FP64 Warp storage on the calibration's owning device.

        The caller owns ``length_mm`` and the calibration owns the returned array,
        which the next ``forward`` call overwrites. The derivative of the latest
        successful forward is retained independently of the input and output, so
        editing either array preserves the corresponding ``forward_vjp``. An
        explicit ``stream`` must match the owning stream, and ``check=False``
        defers diagnostic readback.

        """
        with self._lock:
            self._before_call(check, stream)
            ctx = self.context
            ctx.array(
                length_mm, "length_mm", dtype=ctx.wp.float64, shape=self.shape, ndim=len(self.shape)
            )
            with ctx.scope():
                ctx.wp.launch(
                    self._kernels.forward,
                    dim=self._pixels,
                    inputs=[
                        length_mm.flatten(),
                        self._coefficients,
                        self._weights,
                        self._log_weights,
                        self._maximum_coefficient,
                        self.spec.maximum_length_mm,
                    ],
                    outputs=[self._depth.flatten(), self._forward_slope.flatten(), self._status],
                    device=ctx.device,
                    stream=ctx.stream,
                    record_tape=False,
                )
            self._pending = self._has_forward = True
            if check:
                self.check_status()
            return self._depth

    def inverse(
        self, optical_depth: Array, *, check: bool = True, stream: Stream | None = None
    ) -> Array:
        """Solve ``l = p⁻¹(optical_depth)`` within the prepared calibration.

        The curve is ``p(l) = -log(sum_e w_e exp(-mu_e l) / sum_e w_e)``, with
        ``w_e = source_weights[e] * detector_response[e]`` and supplied ``mu_e`` in
        mm⁻¹. The finite dimensionless input must lie in ``[0, p(maximum_length_mm)]``,
        and the returned length is in mm. Both arrays have ``self.shape`` and
        contiguous FP64 Warp storage on the calibration's owning device.

        Bisection resolves a binary64 bracket in ``[0, spec.maximum_length_mm]``
        and reports a numerical error if its iteration bound is exhausted. The
        caller owns ``optical_depth`` and the calibration owns the length array,
        which the next ``inverse`` call overwrites. The latest successful inverse
        retains its derivative independently of both arrays for ``inverse_vjp``.
        An explicit ``stream`` must match the owning stream, and ``check=False``
        defers diagnostic readback.

        """
        with self._lock:
            self._before_call(check, stream)
            ctx = self.context
            ctx.array(
                optical_depth,
                "optical_depth",
                dtype=ctx.wp.float64,
                shape=self.shape,
                ndim=len(self.shape),
            )
            with ctx.scope():
                ctx.wp.launch(
                    self._kernels.inverse,
                    dim=self._pixels,
                    inputs=[
                        optical_depth.flatten(),
                        self._coefficients,
                        self._weights,
                        self._log_weights,
                        self._maximum_coefficient,
                        self.spec.maximum_length_mm,
                        self.execution.maximum_iterations,
                    ],
                    outputs=[self._length.flatten(), self._inverse_slope.flatten(), self._status],
                    device=ctx.device,
                    stream=ctx.stream,
                    record_tape=False,
                )
            self._pending = self._has_inverse = True
            if check:
                self.check_status()
            return self._length

    def _vjp(self, cotangent: Array, *, invert: bool, check: bool, stream: Stream | None) -> Array:
        with self._lock:
            self._before_call(check, stream)
            if not (self._has_inverse if invert else self._has_forward):
                raise ContractError("VJP requires its corresponding successful calibration call")
            ctx = self.context
            ctx.array(
                cotangent, "cotangent", dtype=ctx.wp.float64, shape=self.shape, ndim=len(self.shape)
            )
            with ctx.scope():
                ctx.wp.launch(
                    self._kernels.vjp,
                    dim=self._pixels,
                    inputs=[
                        cotangent.flatten(),
                        (self._inverse_slope if invert else self._forward_slope).flatten(),
                        int(invert),
                    ],
                    outputs=[self._gradient.flatten(), self._status],
                    device=ctx.device,
                    stream=ctx.stream,
                    record_tape=False,
                )
            self._pending = True
            if check:
                self.check_status()
            return self._gradient

    def forward_vjp(
        self, cotangent: Array, *, check: bool = True, stream: Stream | None = None
    ) -> Array:
        """Return ``g_l = g_p * p'(l)`` for the latest successful ``forward``.

        Here ``p'(l) = sum_e w_e mu_e exp(-mu_e l) / sum_e w_e exp(-mu_e l)``,
        with ``w_e = source_weights[e] * detector_response[e]`` and supplied
        ``mu_e`` in mm⁻¹. ``cotangent`` is a finite, signed optical-depth cotangent and the
        result is its length cotangent, with the inverse-mm factor supplied by
        ``p'(l)``. Both arrays have ``self.shape`` and contiguous FP64 Warp storage
        on the calibration's owning device.

        A successful ``forward`` must precede this call. Its retained slope is
        independent of caller input and returned-depth edits, while each new
        successful forward replaces that slope. The caller owns ``cotangent`` and
        the calibration owns the returned gradient array, which either VJP
        method overwrites on its next call. An explicit ``stream`` must match the
        owning stream, and ``check=False`` defers diagnostic readback.

        """
        return self._vjp(cotangent, invert=False, check=check, stream=stream)

    def inverse_vjp(
        self, cotangent: Array, *, check: bool = True, stream: Stream | None = None
    ) -> Array:
        """Return ``g_p = g_l / p'(l)`` for the latest successful ``inverse``.

        Here ``p'(l) = sum_e w_e mu_e exp(-mu_e l) / sum_e w_e exp(-mu_e l)``,
        with ``w_e = source_weights[e] * detector_response[e]`` and supplied
        ``mu_e`` in mm⁻¹. ``cotangent`` is a finite, signed length cotangent and the result
        is the cotangent of dimensionless optical depth, with the mm factor
        supplied by ``dl/dp = 1/p'(l)``. Both arrays have ``self.shape`` and contiguous FP64
        Warp storage on the calibration's owning device.

        A successful ``inverse`` must precede this call. Its retained slope is
        independent of caller input and returned-length edits, while each new
        successful inverse replaces that slope. The caller owns ``cotangent`` and
        the calibration owns the returned gradient array, which either VJP
        method overwrites on its next call. An explicit ``stream`` must match the
        owning stream, and ``check=False`` defers diagnostic readback.

        """
        return self._vjp(cotangent, invert=True, check=check, stream=stream)
