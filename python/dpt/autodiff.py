"""First-order tape ownership for composed deterministic operators.

A tape is an execution lifetime, not an image Jacobian. This small owner makes
output seeding, error checkpoints and discard/reset explicit. Scientific
operators still receive the underlying tape through their tape= argument.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from dpt._runtime import (
    DeviceContext,
    assert_tape_owner,
    load_warp,
    prepare_context,
    tape_execution_scope,
)
from dpt.contracts import ContractError


class CheckedWorkspace(Protocol):
    def clear_status(self) -> None: ...
    def check_status(self) -> None: ...


def _workspace_context(workspaces: Sequence[CheckedWorkspace]) -> DeviceContext | None:
    """Resolve declared owners using metadata before any status or tape mutation."""
    selected: DeviceContext | None = None
    for workspace in workspaces:
        context = getattr(workspace, "context", None)
        if not isinstance(context, DeviceContext):
            # Metadata-only external/fake workspaces need not declare an owner.
            continue
        if selected is None:
            selected = context
        elif context.wp is not selected.wp or context.device != selected.device:
            raise ContractError("FirstOrderPass workspaces must share one Warp runtime and device")
        elif (
            context.device.is_cuda
            and cast(Any, context.stream).cuda_stream != cast(Any, selected.stream).cuda_stream
        ):
            raise ContractError("FirstOrderPass workspaces must share one owning CUDA stream")
    return selected


@dataclass(slots=True)
class FirstOrderPass:
    """Own one forward/reverse evaluation and its explicit diagnostic checkpoint.

    Use as a context manager; pass .tape into canonical forward operators.
    Call backward after exiting the context. Keep original inputs immutable
    until close(). A new iteration requires a fresh context entry after close.
    """

    workspaces: Sequence[CheckedWorkspace]
    tape: Any = field(default=None, init=False, repr=False)
    _state: str = field(default="new", init=False)
    _recording_lease: Any = field(default=None, init=False, repr=False)
    _context: DeviceContext | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> FirstOrderPass:
        if self._state != "new":
            raise ContractError("a FirstOrderPass owns exactly one evaluation")
        self._context = _workspace_context(self.workspaces)
        self._recording_lease = tape_execution_scope()
        self._recording_lease.__enter__()
        try:
            for workspace in self.workspaces:
                workspace.check_status()
            for workspace in self.workspaces:
                workspace.clear_status()
            self.tape = load_warp().Tape()
            self.tape.__enter__()
        except BaseException:
            self._recording_lease.__exit__(None, None, None)
            self._recording_lease = None
            raise
        self._state = "recording"
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._state != "recording":
            raise ContractError("exit requires an active recording context")
        assert_tape_owner()
        try:
            self.tape.__exit__(exc_type, exc_value, traceback)
        finally:
            self._state = "recorded"
            self._recording_lease.__exit__(exc_type, exc_value, traceback)
            self._recording_lease = None
        if exc_type is not None:
            self.close()

    def backward(self, output: Any, cotangent: Any) -> None:
        """Input gradients = J_recorded^T*cotangent complete the recorded first-order pass.

        Output is a caller-owned Warp array with a preallocated gradient. Cotangent
        has exactly its shape, dtype, device and units of loss per output unit.
        Active inputs are those declared by the recorded canonical
        operators and retained unchanged through completion. Backward requires an
        exited forward-recording context, seeds that output and checks every owned
        workspace's numerical status. A failed reverse closes the pass. Close after
        successful use to clear accumulated cotangents and release the graph.

        """
        if self._state != "recorded":
            raise ContractError("backward requires one completed, unconsumed forward pass")
        try:
            with tape_execution_scope():
                wp = load_warp()
                if not isinstance(output, wp.array) or output.grad is None:
                    raise ContractError("output must be a Warp array with a preallocated gradient")
                ctx = self._context
                if ctx is None:
                    ctx = prepare_context(device=output.device)
                    self._context = ctx
                ctx.array(output, "output", dtype=output.dtype, ndim=output.ndim)
                ctx.array(
                    cotangent,
                    "cotangent",
                    dtype=output.dtype,
                    shape=tuple(output.shape),
                    ndim=output.ndim,
                )
                with ctx.scope():
                    self.tape.backward(grads={output: cotangent})
                for workspace in self.workspaces:
                    workspace.check_status()
                self._state = "complete"
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Clear accumulated cotangents and release every abandoned recording."""
        if self._state == "recording":
            raise ContractError("exit the recording context before closing it")
        if self.tape is not None and self._state != "closed":
            with (
                tape_execution_scope(),
                self._context.scope() if self._context is not None else nullcontext(),
            ):
                self.tape.zero()
                self.tape.reset()
                for workspace in self.workspaces:
                    discard = getattr(workspace, "discard_recording", None)
                    if discard is not None:
                        discard(self.tape)
        self._state = "closed"
