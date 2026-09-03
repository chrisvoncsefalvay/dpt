"""Connect ``T=exp(-L)``, detector response and scalar losses in Torch.

Transmission, affine calibration and spatial blur return owned FP32
images. Objective losses return FP64 scalars and multiply their
upstream cotangents before casting to the prediction's precision.
Observations, objective weights, masks and PSFs are fixed. Prepared
policies select active inputs, scalar/pixel layouts and reductions;
native VJPs connect them to ordinary first-order Torch backward.

"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
from dataclasses import replace
from typing import Any, Literal

import torch

from dpt._runtime import require_no_tape
from dpt.contracts import ContractError, NumericalError
from dpt.detector import BlurWorkspace, CalibrationSpec, DetectorWorkspace
from dpt.detector import blur as _blur
from dpt.detector import blur_transpose as _blur_vjp
from dpt.detector import calibrate as _calibrate
from dpt.detector import calibration_vjp as _calibration_vjp
from dpt.interop._torch_streams import stream_bridge
from dpt.interop.torch import _pack_cotangent
from dpt.objectives import ObjectiveWorkspace, evaluate_objective, prepare_objective
from dpt.transmission import TransmissionWorkspace, transmission_vjp
from dpt.transmission import transmit as _transmit

TransmissionOutput = Literal["transmission", "counts", "log_transmission", "removed"]
__all__ = ["TransmissionOutput", "affine_calibrate", "blur", "objective", "transmit"]
_OUTPUTS = {
    "transmission": ("out_T", "cotangent_T"),
    "counts": ("out_counts", "cotangent_counts"),
    "log_transmission": ("out_log_T", "cotangent_log_T"),
    "removed": ("out_removed", "cotangent_removed"),
}


def _tensor(torch: Any, value: Any, name: str, dtype: Any) -> None:
    if not isinstance(value, torch.Tensor):
        raise ContractError(f"{name} must be a Torch tensor")
    if value.dtype != dtype or not value.is_contiguous():
        raise ContractError(f"{name} must use contiguous Torch {dtype} storage")


def _fixed(value: Any, name: str) -> None:
    if value is not None and value.requires_grad:
        raise ContractError(f"{name} is fixed in the native objective")


def _active(value: Any, active: bool, name: str) -> None:
    if value.requires_grad and not active:
        raise ContractError(f"{name} was declared fixed in the native operator")


def _array(wp: Any, value: Any) -> Any:
    return wp.from_torch(value.reshape(-1), requires_grad=False)


def _first_order(torch: Any) -> None:
    if torch.is_grad_enabled():
        raise ContractError("DPT Torch operators provide first-order derivatives only")
    require_no_tape()


def _validate_flag(validate: bool) -> None:
    if type(validate) is not bool:
        raise ContractError("validate must be a boolean")


def _checked_round(torch: Any, values: Any, like: Any) -> Any:
    """Keep scalar-loss cotangents in FP64 until the input's gradient boundary."""
    rounded = values.to(dtype=like.dtype)
    if not torch.isfinite(rounded).all().item():
        raise NumericalError("Torch objective gradient exceeds its input storage range")
    return rounded


class TransmissionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        owner: Any,
        depth: Any,
        beam: Any,
        output: TransmissionOutput,
        workspace: TransmissionWorkspace,
        validate: bool,
    ) -> Any:
        require_no_tape()
        _validate_flag(validate)
        _tensor(torch, depth, "optical_depth", torch.float32)
        _active(depth, workspace.spec.active_L, "optical_depth")
        if type(output) is not str or output not in _OUTPUTS:
            raise ContractError("select transmission, counts, log_transmission or removed")
        tensor_beam = isinstance(beam, torch.Tensor)
        if workspace.spec.beam in ("device-scalar", "per-pixel"):
            _tensor(torch, beam, "open_beam", torch.float32)
            _active(beam, workspace.spec.active_beam, "open_beam")
            if beam.device != depth.device:
                raise ContractError("open_beam and optical_depth must share a device")
        elif tensor_beam:
            raise ContractError("Torch beam tensors require device-scalar or per-pixel policy")
        tensors = (depth, beam) if tensor_beam else (depth,)
        with stream_bridge(torch, tensors, workspace) as bridge:
            result = torch.empty_like(depth)
            bridge.record(result)
            native_beam = _array(workspace.context.wp, beam) if tensor_beam else beam
            _transmit(
                _array(workspace.context.wp, depth),
                native_beam,
                **{_OUTPUTS[output][0]: _array(workspace.context.wp, result)},
                workspace=workspace,
                validate=validate,
            )
            workspace.consume_status()
        owner.save_for_backward(*tensors)
        owner.workspace, owner.output, owner.validate = workspace, output, validate
        owner.tensor_beam, owner.scalar_beam = tensor_beam, None if tensor_beam else beam
        return result

    @staticmethod
    def backward(owner: Any, seed: Any) -> tuple[Any, Any, None, None, None]:
        _first_order(torch)
        tensors = owner.saved_tensors
        depth = tensors[0]
        beam = tensors[1] if owner.tensor_beam else owner.scalar_beam
        workspace, wp = owner.workspace, owner.workspace.context.wp
        with stream_bridge(torch, (*tensors, seed), workspace) as bridge:
            seed = _pack_cotangent(torch, seed)
            bridge.record(seed)
            grad_depth = torch.empty_like(depth) if owner.needs_input_grad[0] else None
            grad_beam = torch.empty_like(beam) if owner.needs_input_grad[1] else None
            bridge.record(*(value for value in (grad_depth, grad_beam) if value is not None))
            transmission_vjp(
                _array(wp, depth),
                _array(wp, beam) if owner.tensor_beam else beam,
                **{_OUTPUTS[owner.output][1]: _array(wp, seed)},
                out_grad_L=_array(wp, grad_depth) if grad_depth is not None else None,
                out_grad_n0=_array(wp, grad_beam) if grad_beam is not None else None,
                workspace=workspace,
                validate=owner.validate,
            )
            workspace.consume_status()
        return grad_depth, grad_beam, None, None, None


class CalibrationFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        owner: Any,
        mean: Any,
        gain: Any,
        exposure: Any,
        offset: Any,
        spec: CalibrationSpec,
        workspace: DetectorWorkspace,
        validate: bool,
    ) -> Any:
        require_no_tape()
        _validate_flag(validate)
        tensors = (mean, gain, exposure, offset)
        for name, value in zip(("mean", "gain", "exposure", "offset"), tensors, strict=True):
            _tensor(torch, value, name, torch.float32)
            _active(value, getattr(spec, f"active_{name}"), name)
        with stream_bridge(torch, tensors, workspace) as bridge:
            result = torch.empty_like(mean)
            bridge.record(result)
            _calibrate(
                *(_array(workspace.context.wp, value) for value in tensors),
                out_signal=_array(workspace.context.wp, result),
                spec=spec,
                workspace=workspace,
                validate=validate,
            )
            workspace.consume_status()
        owner.save_for_backward(*tensors)
        owner.workspace, owner.spec, owner.validate = workspace, spec, validate
        return result

    @staticmethod
    def backward(owner: Any, seed: Any) -> tuple[Any, Any, Any, Any, None, None, None]:
        _first_order(torch)
        tensors = owner.saved_tensors
        workspace, wp = owner.workspace, owner.workspace.context.wp
        with stream_bridge(torch, (*tensors, seed), workspace) as bridge:
            seed = _pack_cotangent(torch, seed)
            bridge.record(seed)
            gradients = tuple(
                torch.empty_like(value) if active else None
                for value, active in zip(tensors, owner.needs_input_grad[:4], strict=True)
            )
            bridge.record(*(value for value in gradients if value is not None))
            outputs = {
                f"out_grad_{name}": _array(wp, value) if value is not None else None
                for name, value in zip(
                    ("mean", "gain", "exposure", "offset"), gradients, strict=True
                )
            }
            _calibration_vjp(
                *(_array(wp, value) for value in tensors),
                cotangent=_array(wp, seed),
                **outputs,
                spec=owner.spec,
                workspace=workspace,
                validate=owner.validate,
            )
            workspace.consume_status()
        return gradients[0], gradients[1], gradients[2], gradients[3], None, None, None


class BlurFunction(torch.autograd.Function):
    @staticmethod
    def forward(owner: Any, source: Any, workspace: BlurWorkspace, validate: bool) -> Any:
        require_no_tape()
        _validate_flag(validate)
        _tensor(torch, source, "signal", torch.float32)
        shape = (workspace.spec.height, workspace.spec.width)
        if tuple(source.shape) not in (shape, (shape[0] * shape[1],)):
            raise ContractError("blur signal must have the prepared detector shape")
        with stream_bridge(torch, source, workspace.detector) as bridge:
            result = torch.empty_like(source)
            bridge.record(result)
            wp = workspace.detector.context.wp
            _blur(
                _array(wp, source),
                out_signal=_array(wp, result),
                workspace=workspace,
                validate=validate,
            )
            workspace.consume_status()
        owner.save_for_backward(source)
        owner.workspace, owner.validate = workspace, validate
        return result

    @staticmethod
    def backward(owner: Any, seed: Any) -> tuple[Any, None, None]:
        _first_order(torch)
        (source,) = owner.saved_tensors
        workspace = owner.workspace
        with stream_bridge(torch, (source, seed), workspace.detector) as bridge:
            seed = _pack_cotangent(torch, seed)
            bridge.record(seed)
            gradient = torch.empty_like(source)
            bridge.record(gradient)
            wp = workspace.detector.context.wp
            _blur_vjp(
                _array(wp, seed),
                out_grad_signal=_array(wp, gradient),
                workspace=workspace,
                validate=owner.validate,
            )
            workspace.consume_status()
        return gradient, None, None


class ObjectiveFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        owner: Any,
        prediction: Any,
        observation: Any,
        weights: Any,
        valid: Any,
        workspace: ObjectiveWorkspace,
        validate: bool,
    ) -> Any:
        require_no_tape()
        _validate_flag(validate)
        dtype = torch.float64 if workspace.spec.precision == "float64" else torch.float32
        _tensor(torch, prediction, "prediction", dtype)
        _tensor(torch, observation, "observation", torch.float32)
        fixed = [("observation", observation)]
        if weights is not None:
            _tensor(torch, weights, "weights", torch.float32)
            fixed.append(("weights", weights))
        if valid is not None:
            _tensor(torch, valid, "valid", torch.uint8)
            fixed.append(("valid", valid))
        for name, value in fixed:
            _fixed(value, name)
            if value.shape != prediction.shape:
                raise ContractError(f"{name} must have the prediction's shape")
        tensors = (prediction, *(value for _, value in fixed))
        wp = workspace.context.wp
        with stream_bridge(torch, tensors, workspace) as bridge:
            result = torch.empty((), dtype=torch.float64, device=prediction.device)
            bridge.record(result)
            evaluate_objective(
                _array(wp, prediction),
                _array(wp, observation),
                out_loss=_array(wp, result),
                weights=_array(wp, weights) if weights is not None else None,
                valid=_array(wp, valid) if valid is not None else None,
                workspace=workspace,
                validate=validate,
            )
            workspace.consume_status()
        owner.save_for_backward(*tensors)
        owner.has_weights, owner.has_valid = weights is not None, valid is not None
        owner.workspace, owner.validate = workspace, validate
        if prediction.requires_grad:
            # Each node retains canonical FP64 derivative scratch. Preparation
            # happens once in forward, outside the backward kernel sequence.
            owner.seed_workspace = prepare_objective(
                replace(workspace.spec, precision="float64"),
                max_pixels=workspace.max_pixels,
                device=str(workspace.context.device),
                stream=workspace.context.stream,
            )
        return result

    @staticmethod
    def backward(owner: Any, seed: Any) -> tuple[Any, None, None, None, None, None]:
        _first_order(torch)
        tensors = owner.saved_tensors
        prediction, observation = tensors[:2]
        weights = tensors[2] if owner.has_weights else None
        valid = tensors[-1] if owner.has_valid else None
        workspace, wp = owner.workspace, owner.workspace.context.wp
        with stream_bridge(torch, (*tensors, seed), workspace) as bridge:
            # A pending error on the caller's original workspace remains
            # observable even though each node owns derivative scratch.
            workspace.check_status()
            if not torch.isfinite(seed).all().item():
                raise ContractError("objective cotangent must be finite")
            if seed.item() == 0:
                gradient = torch.zeros_like(prediction)
                bridge.record(gradient)
            else:
                # Recompute this node's seed after any intervening forward.
                # The existing FP64 native path delays image-gradient rounding
                # until after multiplication by the scalar loss cotangent.
                bound = owner.seed_workspace
                promoted = prediction.to(dtype=torch.float64)
                native_seed = torch.empty_like(promoted)
                loss = torch.empty((), dtype=torch.float64, device=prediction.device)
                bridge.record(promoted, native_seed, loss)
                evaluate_objective(
                    _array(wp, promoted),
                    _array(wp, observation),
                    out_loss=_array(wp, loss),
                    out_cotangent=_array(wp, native_seed),
                    weights=_array(wp, weights) if weights is not None else None,
                    valid=_array(wp, valid) if valid is not None else None,
                    workspace=bound,
                    validate=owner.validate,
                )
                bound.consume_status()
                gradient = _checked_round(torch, native_seed * seed, prediction)
                bridge.record(gradient)
        return gradient, None, None, None, None, None


def transmit(
    optical_depth: Any,
    open_beam: Any = None,
    *,
    output: TransmissionOutput = "transmission",
    workspace: TransmissionWorkspace,
    validate: bool = True,
) -> Any:
    """Compute ``T=exp(-L)``, counts ``n0*T``, log transmission ``-L`` or ``1-T``.

    ``optical_depth`` is nonnegative dimensionless contiguous FP32, flat
    ``(P,)`` or image-shaped. Output preserves its shape, device and FP32
    storage. ``output`` selects ``transmission``, ``counts``,
    ``log_transmission`` or ``removed``; removed transmission uses
    ``-expm1(-L)``. Counts are an expected photon count per pixel.

    ``workspace.spec.beam`` chooses no beam, a Python scalar, one FP32
    tensor value, or an FP32 per-pixel tensor. Active beam derivatives
    use device-scalar or per-pixel tensors on the depth's device. The
    prepared ``active_L`` and ``active_beam`` flags select gradients.
    Outputs own Torch storage; saved depth and beam tensors remain
    unchanged until their first-order backward completes.

    """
    return TransmissionFunction.apply(optical_depth, open_beam, output, workspace, validate)


def affine_calibrate(
    mean: Any,
    gain: Any,
    exposure: Any,
    offset: Any,
    *,
    spec: CalibrationSpec,
    workspace: DetectorWorkspace,
    validate: bool = True,
) -> Any:
    """Compute ``y_p = gain_p * exposure * mean_p + offset_p``.

    All four arguments are contiguous FP32 tensors on the workspace
    device. ``mean`` is flat ``(P,)`` or image-shaped; gain and offset
    are scalar or per-pixel according to ``CalibrationSpec``. Exposure
    has one value. Output preserves the mean's shape and owns FP32 Torch
    storage in ``spec.output_unit``. Gain and exposure convert the input
    mean's units into that unit; offset has the output unit.

    ``active_mean``, ``active_gain``, ``active_exposure`` and
    ``active_offset`` select first-order derivatives. Active nuisance
    parameters are shared scalars. Choose gain or exposure as the active
    acquisition scale, since their product defines the same scale.

    """
    return CalibrationFunction.apply(mean, gain, exposure, offset, spec, workspace, validate)


def blur(signal: Any, *, workspace: BlurWorkspace, validate: bool = True) -> Any:
    """Compute ``(Bx)[r,c] = sum_ij h[i,j] x[r+i-a,c+j-b]``.

    ``signal`` is contiguous FP32 ``(rows,columns)`` or flat ``(P,)`` on
    the prepared device. The fixed odd-sized passive PSF, its centre
    ``(a,b)`` and panel dimensions come from ``workspace.spec``. Samples
    beyond the panel are zero, so edge pixels retain the specified lost
    mass. The FP32 result owns Torch storage and preserves image shape
    and signal units. Backward applies the exact transpose ``B.T`` with
    the same zero-extension rule; only the signal is active.

    """
    return BlurFunction.apply(signal, workspace, validate)


def objective(
    prediction: Any,
    observation: Any,
    *,
    weights: Any = None,
    valid: Any = None,
    workspace: ObjectiveWorkspace,
    validate: bool = True,
) -> Any:
    """Compute half squared error ``0.5*sum_p w_p*(prediction_p-observation_p)^2``.

    ``workspace.spec.kind`` also selects Poisson half-deviance
    ``sum_p w_p*(prediction_p-observation_p+observation_p*log(observation_p/prediction_p))``;
    a zero observation contributes the prediction. Predictions use the
    prepared FP32 or FP64 precision and flat or image-shaped storage.
    Observations and optional weights are fixed contiguous FP32 tensors;
    optional validity masks are fixed uint8 tensors of the same extent.
    All tensors share the workspace device. Invalid pixels contribute
    zero. ``reduction="mean"`` divides by the full pixel count.

    The output is an owned FP64 scalar. Squared-error units are squared
    signal units times weight; Poisson input represents expected counts
    and observed counts. Only prediction has an autograd gradient.
    The native seed and scalar loss cotangent multiply in FP64 before
    rounding to prediction storage, preserving small upstream scales.

    """
    return ObjectiveFunction.apply(prediction, observation, weights, valid, workspace, validate)
