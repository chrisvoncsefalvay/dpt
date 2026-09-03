"""Explicit zero-copy array views with device, dtype and layout checks."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast

from dpt._runtime import DeviceContext
from dpt.contracts import ContractError

if TYPE_CHECKING:
    import warp as wp


class DLPackProvider(Protocol):
    """A producer that exports storage and its device through the DLPack protocol."""

    def __dlpack__(self, *, stream: int | None = None) -> object: ...

    def __dlpack_device__(self) -> tuple[int, int]: ...


class CUDAArrayProvider(Protocol):
    """A producer with the legacy CUDA array interface and explicit device declaration."""

    @property
    def __cuda_array_interface__(self) -> Mapping[str, object]: ...


def as_warp_array(
    value: wp.array[Any] | DLPackProvider | CUDAArrayProvider,
    *,
    context: DeviceContext,
    dtype: object,
    flatten: bool = True,
    source_device: str | wp.Device | None = None,
) -> wp.array[Any]:
    """view[i]=value[i] with shared storage and unchanged element precision.

    ``value`` is a contiguous Warp/Torch array, DLPack exporter or legacy CUDA
    array-interface object on the device owned by ``context``. ``dtype`` must
    exactly match its native storage dtype. For input shape (n0,...,nk), return
    a flat Warp view of shape (product(shape),) when ``flatten=True`` or retain
    the original shape when false. Element order and physical units remain
    those of the caller's buffer, with no device transfer or dtype conversion.

    This creates an array view, not a derivative operator. Imported Torch views
    have ``requires_grad=False`` and DPT's explicit operators determine their
    active parameters. The producer storage must remain alive while the view
    is used. CUDA producer streams are ordered before the workspace stream,
    including Torch's external-stream lifetime tracking and DLPack handoff.
    Legacy CUDA interfaces also require an explicit ``source_device`` because
    their pointer descriptor does not identify the device.

    """
    # Dispatch inspects framework metadata after the public protocol boundary.
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_value: Any = cast(Any, value)
    if type(flatten) is not bool:
        raise ContractError("flatten must be a boolean")
    wp = context.wp
    if isinstance(_api_value, wp.array):
        array = _api_value
    else:
        try:
            with context.scope():
                if type(_api_value).__module__.split(".")[0] == "torch":
                    torch = importlib.import_module("torch")
                    if _api_value.device.type not in ("cpu", "cuda"):
                        raise ContractError("Torch array views support only CPU and CUDA devices")
                    if wp.device_from_torch(_api_value.device) != context.device:
                        raise ContractError(
                            "array and workspace devices differ; implicit uploads are forbidden"
                        )
                    if context.device.is_cuda:
                        assert context.stream is not None
                        context.wait_for(
                            wp.stream_from_torch(torch.cuda.current_stream(_api_value.device))
                        )
                        _api_value.record_stream(
                            torch.cuda.ExternalStream(
                                cast(Any, context.stream).cuda_stream, device=_api_value.device
                            )
                        )
                    array = wp.from_torch(_api_value, requires_grad=False)
                elif hasattr(_api_value, "__dlpack__") and hasattr(_api_value, "__dlpack_device__"):
                    device_type, ordinal = _api_value.__dlpack_device__()
                    compatible = (
                        device_type in (2, 13)
                        and context.device.is_cuda
                        and ordinal == context.device.ordinal
                    ) or (device_type == 1 and not context.device.is_cuda)
                    if not compatible:
                        raise ContractError(
                            "array and workspace devices differ; implicit uploads are forbidden"
                        )
                    # Infer the actual dtype. Never reinterpret an incompatible
                    # exporter by passing the requested dtype into DLPack.
                    array = wp.from_dlpack(_api_value)
                elif hasattr(_api_value, "__cuda_array_interface__"):
                    if not context.device.is_cuda or source_device is None:
                        raise ContractError(
                            "a legacy CUDA array view requires explicit source_device"
                        )
                    if wp.get_device(source_device) != context.device:
                        raise ContractError("array and workspace devices differ")
                    descriptor = _api_value.__cuda_array_interface__
                    np = importlib.import_module("numpy")
                    source_dtype = np.dtype(descriptor["typestr"])
                    if not source_dtype.isnative:
                        raise ContractError("array storage must use native byte order")
                    actual_dtype = wp.dtype_from_numpy(source_dtype)
                    if actual_dtype != dtype or descriptor["data"][1]:
                        raise ContractError(
                            "CUDA array storage must have the exact dtype and be writable"
                        )
                    producer = descriptor.get("stream")
                    if producer is not None:
                        if type(producer) is not int or producer <= 0:
                            raise ContractError(
                                "CUDA array interface has an invalid producer stream"
                            )
                        context.wait_for(wp.Stream(context.device, cuda_stream=producer))
                    array = wp.array(
                        data=_api_value, dtype=dtype, device=context.device, copy=False
                    )
                else:
                    raise ContractError(
                        "a zero-copy array view requires Warp, Torch or an array protocol"
                    )
        except ContractError:
            raise
        except (BufferError, TypeError, ValueError, RuntimeError) as error:
            raise ContractError(f"cannot create a zero-copy Warp array view: {error}") from error
    context.array(array, "array view", dtype=dtype, ndim=array.ndim)
    return array.flatten() if flatten else array


__all__ = ["CUDAArrayProvider", "DLPackProvider", "as_warp_array"]
