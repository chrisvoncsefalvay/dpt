"""Order borrowed Torch storage on an operator's owning CUDA stream."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from dpt.contracts import ContractError


@dataclass(slots=True)
class StreamBridge:
    """Keep stream wrappers and events alive and register tensor consumers."""

    torch: Any
    context: Any
    caller: Any = None
    owning: Any = None
    ready: Any = None
    complete: Any = None

    def record(self, *tensors: Any) -> None:
        """Protect allocator reuse on the streams that consume this storage."""
        if self.owning is not None:
            for tensor in tensors:
                tensor.record_stream(self.owning)
                if self.caller.cuda_stream != self.owning.cuda_stream:
                    tensor.record_stream(self.caller)


@contextmanager
def stream_bridge(
    torch: Any, tensors: Any | Sequence[Any], workspace: Any
) -> Generator[StreamBridge, None, None]:
    """Queue caller → operator → caller dependencies without a device-wide wait.

    Callers follow Torch's usual producer-stream rules before invoking an
    adapter. Inside this scope, Torch allocations and native launches use the
    operator's stream. Register any new output or packed cotangent with
    ``bridge.record``. Checked native operations retain their owning-stream
    completion checks. A workspace and its scratch remain live throughout the
    scope; exceptional exits complete that stream before releasing aliases.
    """
    values: tuple[Any, ...] = (
        (cast(Any, tensors),)
        if isinstance(tensors, torch.Tensor)
        else tuple(cast(Sequence[Any], tensors))
    )
    if not values or any(not isinstance(value, torch.Tensor) for value in values):
        raise ContractError("the Torch bridge requires one or more tensors")
    context = getattr(workspace, "context", workspace)
    device: Any = values[0].device
    if device.type not in ("cpu", "cuda"):
        raise ContractError("the Torch adapter supports explicit CPU and CUDA tensors")
    if any(value.device != device for value in values):
        raise ContractError("all adapter tensors must share one device")
    if context.wp.device_from_torch(device) != context.device:
        raise ContractError("Torch tensors and operator must share a device")
    bridge = StreamBridge(torch, context)
    if not context.device.is_cuda:
        with context.scope():
            yield bridge
        return
    bridge.caller = torch.cuda.current_stream(device)
    bridge.owning = context.wp.stream_to_torch(context.stream)
    bridge.ready = torch.cuda.Event()
    bridge.complete = torch.cuda.Event()
    bridge.ready.record(bridge.caller)
    bridge.owning.wait_event(bridge.ready)
    with torch.cuda.stream(bridge.owning), context.scope():
        bridge.record(*values)
        try:
            yield bridge
        except BaseException:
            context.wp.synchronize_stream(context.stream)
            raise
        finally:
            bridge.complete.record(bridge.owning)
            bridge.caller.wait_event(bridge.complete)
