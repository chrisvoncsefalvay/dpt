"""Warp execution, stream ownership and prepared host readback.

Every workspace binds to one stream. Context methods validate metadata only;
device-content checks and their synchronisation belong to the calling operator.
Readback buffers allocate host staging once during preparation; their explicit
copies and checkpoints always use the owning stream.
"""

from __future__ import annotations

import gc
import importlib
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock, RLock, get_ident
from typing import TYPE_CHECKING, Any, cast

from dpt.contracts import ContractError, NumericalError

if TYPE_CHECKING:
    import warp as wp

_scope_locks: dict[tuple[int, str], Any] = {}
_scope_locks_guard = Lock()
_tape_lock = Lock()
_tape_owner: int | None = None
_capture_lifetime_lock = RLock()
_capture_lifetime_depth = 0


@contextmanager
def _capture_lifetime(*, inherited_capture: bool = False) -> Generator[None, None, None]:
    """Keep cyclic finalisers outside one complete native capture lifetime.

    Pinned host buffers use CUDA frees during finalisation. An unreachable
    workspace cycle may therefore invalidate an unrelated CUDA capture if
    Python collects it while a captured launch allocates Python metadata.
    Collect once before capture, then defer automatic cyclic collection until
    the outermost scope ends. Ordinary reference-counted finalisation remains
    active; prepared capture bodies retain their fixed allocations.

    An inherited capture without this lifetime scope is refused before changing
    collection state, because its eventual end is outside this builder's control.
    The caller's disabled-GC setting is retained without an explicit collection.
    """
    global _capture_lifetime_depth
    with _capture_lifetime_lock:
        outermost = _capture_lifetime_depth == 0
        if outermost and inherited_capture:
            raise ContractError("prepared graph capture must start outside an active capture")
        restore = outermost and gc.isenabled()
        _capture_lifetime_depth += 1
        try:
            if restore:
                gc.disable()
                gc.collect()
            yield
        finally:
            _capture_lifetime_depth -= 1
            if restore:
                gc.enable()


@contextmanager
def scoped_graph_capture(context: DeviceContext) -> Generator[Any, None, None]:
    """Capture a private prepared graph with its cyclic-finalisation guard.

    The existing runtime scope serialises DPT capture builders. A pre-existing
    Warp capture is rejected before collection or native capture begins; these
    builders perform host checkpoints and own their complete top-level capture.
    """
    with context.scope():
        inherited = any(device.is_capturing for device in context.wp.get_devices())
        with _capture_lifetime(inherited_capture=inherited):
            with context.wp.ScopedCapture(device=context.device, stream=context.stream) as capture:
                yield capture


def _scope_lock(wp: Any, device: Any) -> Any:
    # ScopedStream also changes Warp's process-global default device. One lock
    # per Warp runtime protects scopes across CPU and different CUDA devices;
    # host enqueues serialize, while GPU work remains asynchronous.
    key = (id(wp), "default-device-and-streams")
    with _scope_locks_guard:
        lock = _scope_locks.get(key)
        if lock is None:
            lock = RLock()
            _scope_locks[key] = lock
        return lock


@contextmanager
def cpu_test_context():
    """Select CPU for legacy kernel regressions without redirecting CUDA requests.

    Production operators also accept an explicit ``device="cpu"``. This scoped
    helper remains available for existing test callers.
    """
    wp = load_warp()
    with _scope_lock(wp, "cpu"), wp.ScopedDevice("cpu"):
        yield


def load_warp() -> Any:
    try:
        return importlib.import_module("warp")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "This operator requires Warp; install "
            "`pip install 'differentiable-photon-transport[warp]'`"
        ) from error


def load_kernels(module: str) -> Any:
    return importlib.import_module(module)


@dataclass(frozen=True, slots=True)
class DeviceContext:
    """Device and stream identity retained for a workspace's entire lifetime."""

    wp: Any = field(repr=False)
    device: wp.Device
    stream: wp.Stream | None
    contract_error: type[ContractError] = field(default=ContractError, repr=False)

    def assert_stream(self, stream: wp.Stream | None = None) -> wp.Stream | None:
        if not self.device.is_cuda:
            if stream is not None:
                raise self.contract_error("CPU execution does not own a CUDA stream")
            return None
        # Preserve the default-stream contract while waiting out temporary
        # stream changes made by another DPT scope on this device.
        with _scope_lock(self.wp, self.device):
            selected = self.wp.get_stream(self.device) if stream is None else stream
        assert self.stream is not None
        selected_handle = cast(int, cast(Any, selected).cuda_stream)
        owning_handle = cast(int, cast(Any, self.stream).cuda_stream)
        if selected.device != self.device or selected_handle != owning_handle:
            raise self.contract_error("use the workspace's owning stream on its CUDA device")
        return selected

    @contextmanager
    def scope(self) -> Generator[Any, None, None]:
        """Protect DPT host enqueues on the owning stream without GPU waits.

        The lock covers DPT scopes across devices. External code that changes
        Warp's current stream concurrently must provide its own coordination.
        """
        with _scope_lock(self.wp, self.device):
            manager = (
                self.wp.ScopedStream(self.stream, sync_enter=False)
                if self.device.is_cuda
                else self.wp.ScopedDevice(self.device)
            )
            with manager as owning:
                yield owning

    def wait_for(self, producer: wp.Event | wp.Stream | None = None) -> None:
        """Order subsequent workspace work after a CUDA event or producer stream.

        The dependency must belong to this device. Enqueue an explicit wait on
        the owning stream; this method performs no host synchronization.
        """
        if producer is None:
            return
        if not self.device.is_cuda:
            raise self.contract_error("CPU execution cannot wait for a CUDA event or stream")
        if not isinstance(producer, (self.wp.Event, self.wp.Stream)):
            raise self.contract_error("wait_for requires a Warp CUDA event or stream")
        if producer.device != self.device:
            raise self.contract_error("dependency and workspace must share a CUDA device")
        assert self.stream is not None
        with self.scope():
            if isinstance(producer, self.wp.Event):
                self.stream.wait_event(cast("wp.Event", producer))
            else:
                self.stream.wait_stream(cast("wp.Stream", producer))

    def array(
        self,
        value: Any,
        name: str,
        *,
        dtype: Any,
        shape: tuple[int, ...] | None = None,
        ndim: int = 1,
    ) -> int:
        if not isinstance(value, self.wp.array):
            raise self.contract_error(f"{name} must be a Warp array on {self.device}")
        if value.device != self.device or value.dtype != dtype:
            precision = "binary32" if dtype == self.wp.float32 else str(dtype)
            raise self.contract_error(
                f"{name} has the wrong device or dtype; expected {precision} on {self.device}"
            )
        if not value.is_contiguous or value.ndim != ndim:
            dimensions = "one-dimensional" if ndim == 1 else f"{ndim}-dimensional"
            raise self.contract_error(f"{name} must be contiguous and {dimensions}")
        if shape is not None and tuple(value.shape) != shape:
            raise self.contract_error(f"{name} has shape {value.shape}; expected {shape}")
        return int(value.size)

    def disjoint(self, reads: list[tuple[str, Any]], writes: list[tuple[str, Any]]) -> None:
        """Reject byte-range overlap, including views and unequal element widths."""

        for name, array in reads + writes:
            if array.size and not array.is_contiguous:
                raise self.contract_error(f"{name} must be contiguous for overlap checks")

        def bounds(array: Any) -> tuple[int, int]:
            start = int(array.ptr)
            return start, start + int(array.size) * self.wp.types.type_size_in_bytes(array.dtype)

        for index, (name, destination) in enumerate(writes):
            if not destination.size:
                continue
            lo, hi = bounds(destination)
            for other, source in reads + writes[:index]:
                if not source.size:
                    continue
                source_lo, source_hi = bounds(source)
                if lo < source_hi and source_lo < hi:
                    raise self.contract_error(f"{name} overlaps {other}")


def prepare_context(
    *,
    device: str | wp.Device | None = "cuda:0",
    stream: wp.Stream | None = None,
    contract_error: type[ContractError] = ContractError,
) -> DeviceContext:
    """Resolve explicit CPU or CUDA ownership before repeated evaluation.

    CUDA remains the default and an unavailable CUDA request always fails; CPU
    execution requires selecting the CPU device explicitly.
    """
    wp = load_warp()
    try:
        dev = wp.get_device(device)
    except (ValueError, RuntimeError) as error:
        raise contract_error(
            f"cannot select Warp device {device!r}; check CUDA availability with "
            "dpt-check-gpu or select device='cpu' explicitly"
        ) from error
    active_tape()
    if not dev.is_cuda:
        if device is None:
            raise contract_error("select device='cpu' explicitly for CPU execution")
        if stream is not None:
            raise contract_error("CPU execution does not own a CUDA stream")
        return DeviceContext(wp, dev, None, contract_error)
    with _scope_lock(wp, dev):
        selected = wp.get_stream(dev) if stream is None else stream
    if selected.device != dev:
        raise contract_error("stream and device differ")
    return DeviceContext(wp, dev, selected, contract_error)


@dataclass(slots=True)
class HostReadback:
    """Prepared host staging for one contiguous workspace-owned array.

    ``read()`` returns the same host view on every call. Its contents are valid
    until the next read; callers that retain results must copy the view. CUDA
    staging is pinned and both the copy and wait name the workspace's stream.
    """

    context: DeviceContext
    source: Any
    _host: Any = field(init=False, repr=False)
    _values: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ctx = self.context
        ctx.array(self.source, "readback source", dtype=self.source.dtype, ndim=self.source.ndim)
        self._host = ctx.wp.empty(
            shape=self.source.shape,
            dtype=self.source.dtype,
            device="cpu",
            pinned=ctx.device.is_cuda,
        )
        self._values = self._host.numpy()

    @property
    def host_bytes(self) -> int:
        return int(self.source.size) * self.context.wp.types.type_size_in_bytes(self.source.dtype)

    def read(self) -> Any:
        ctx = self.context
        with ctx.scope():
            ctx.wp.copy(self._host, self.source, stream=ctx.stream)
            if ctx.device.is_cuda:
                ctx.wp.synchronize_stream(ctx.stream)
        return self._values


@dataclass(slots=True)
class StatusReadback(HostReadback):
    """Reusable four-byte status checkpoint with an opt-in consume operation."""

    def __post_init__(self) -> None:
        self.context.array(self.source, "status", dtype=self.context.wp.int32, shape=(1,))
        super(StatusReadback, self).__post_init__()

    def read(self, *, reset: bool = False) -> int:
        ctx = self.context
        with ctx.scope():
            code = int(super(StatusReadback, self).read()[0])
            if reset:
                # Only consume a successfully delivered checkpoint. A failed
                # copy/wait must leave pending errors observable for retry.
                self.source.zero_()
        return code


@dataclass(slots=True)
class StatusCheckpoint:
    """One prepared readback for the diagnostics of a composed evaluation.

    A successful checkpoint transfers only status words and waits once. Error
    decoding uses each workspace's canonical checker on the exceptional path.
    ``check()`` preserves statuses; ``consume()`` delivers every captured error
    as notes and structured ``status_diagnostics`` on the first typed exception
    before clearing the captured statuses. ``begin()`` refuses pending unchecked
    failures before any clearing. ``aggregate=False`` borrows each workspace's
    prepared reader instead of allocating another device/host status buffer;
    those diagnostics use separate copies but retain the same delivery contract.
    """

    context: DeviceContext
    workspaces: Sequence[Any]
    aggregate: bool = True
    _entries: list[tuple[Any, Any]] = field(init=False, repr=False)
    _device: Any = field(init=False, repr=False)
    _readback: HostReadback | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.aggregate) is not bool:
            raise self.context.contract_error("aggregate must be a boolean")
        self.workspaces = tuple(self.workspaces)
        self._entries = []
        seen: set[int] = set()

        def add(workspace: Any) -> None:
            owner = getattr(workspace, "context", self.context)
            if owner.wp is not self.context.wp or owner.device != self.context.device:
                raise self.context.contract_error("checkpoint workspaces must share a device")
            self.context.assert_stream(owner.stream)
            for attribute in ("projection", "detector"):
                nested = getattr(workspace, attribute, None)
                if nested is not None:
                    add(nested)
            status = getattr(workspace, "_status", None)
            if status is None:
                status = getattr(workspace, "status", None)
            if status is None:
                return
            self.context.array(status, "status", dtype=self.context.wp.int32, shape=(1,))
            pointer = int(status.ptr)
            if pointer not in seen:
                seen.add(pointer)
                self._entries.append((workspace, status))

        for workspace in self.workspaces:
            add(workspace)
        if not self._entries:
            raise self.context.contract_error("a status checkpoint requires workspace diagnostics")
        if self.aggregate:
            self._device = self.context.wp.empty(
                len(self._entries), dtype=self.context.wp.int32, device=self.context.device
            )
            self._readback = HostReadback(self.context, self._device)
        else:
            for workspace, status in self._entries:
                reader = getattr(workspace, "_status_readback", None)
                if not isinstance(reader, StatusReadback) or reader.source.ptr != status.ptr:
                    raise self.context.contract_error(
                        "borrowed checkpoints require prepared workspace status readers"
                    )
            self._device = self._readback = None

    @property
    def device_bytes(self) -> int:
        return 4 * len(self._entries) if self.aggregate else 0

    @property
    def host_bytes(self) -> int:
        return self._readback.host_bytes if self._readback is not None else 0

    def _check(self, *, consume: bool) -> None:
        ctx = self.context
        with ctx.scope():
            if self._readback is not None:
                for index, (_, status) in enumerate(self._entries):
                    ctx.wp.copy(self._device, status, dest_offset=index, count=1, stream=ctx.stream)
                codes = self._readback.read()
            else:
                # Read every status before delivery/reset. A failed later copy
                # leaves earlier successfully copied failures observable too.
                codes = [workspace._status_readback.read() for workspace, _ in self._entries]
            self._deliver(codes, consume=consume)

    def _deliver(self, codes: Sequence[Any], *, consume: bool) -> None:
        """Decode a completed snapshot while the caller holds the owning scope."""
        failures: list[ContractError | NumericalError] = []
        diagnostics: list[tuple[str, int, str]] = []
        for code, (workspace, _) in zip(codes, self._entries, strict=True):
            if int(code):
                try:
                    workspace.check_status()
                except (ContractError, NumericalError) as error:
                    failures.append(error)
                    diagnostics.append((type(workspace).__name__, int(code), str(error)))
        # A failed copy/wait or decoder never reaches clearing. Preserve
        # the first exception's established public type and attach every
        # delivered workspace/code, instead of discarding later errors.
        if failures:
            for owner, code, message in diagnostics:
                failures[0].add_note(f"{owner} status {code}: {message}")
            failures[0].__dict__["status_diagnostics"] = tuple(diagnostics)
        if consume:
            try:
                for workspace in self.workspaces:
                    workspace.clear_status()
            except BaseException as error:
                # A reset can itself fail. Preserve the captured diagnostics on
                # that failure even if earlier, already-delivered words cleared.
                for owner, code, message in diagnostics:
                    error.add_note(f"{owner} status {code}: {message}")
                error.__dict__["status_diagnostics"] = tuple(diagnostics)
                raise
        if failures:
            raise failures[0]

    def check(self) -> None:
        self._check(consume=False)

    def consume(self) -> None:
        self._check(consume=True)

    def begin(self) -> None:
        self.check()
        for workspace in self.workspaces:
            workspace.clear_status()


@dataclass(slots=True)
class CompletionReadback:
    """Prepared loss/gradient/status epilogue with one pinned payload transfer.

    Sources are contiguous FP64 arrays on the checkpoint's owning device. The
    status words are exactly representable in the FP64 payload. ``consume()``
    returns stable source views whose contents last until the next completion;
    the caller retains this helper for their lifetime. Pending-error checks at
    evaluation entry remain the separate checkpoint's responsibility.
    """

    checkpoint: StatusCheckpoint
    sources: Sequence[Any]
    _payload: Any = field(init=False, repr=False)
    _readback: HostReadback = field(init=False, repr=False)
    _offsets: tuple[int, ...] = field(init=False, repr=False)
    _status_offset: int = field(init=False, repr=False)
    _views: tuple[Any, ...] = field(init=False, repr=False)
    _kernels: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.checkpoint.aggregate:
            raise self.checkpoint.context.contract_error(
                "completion readback requires an aggregate status checkpoint"
            )
        ctx = self.checkpoint.context
        checkpoint: Any = self.checkpoint  # Shared prepared internal status bindings.
        self.sources = tuple(self.sources)
        if not self.sources:
            raise ctx.contract_error("completion readback requires numeric control sources")
        offsets: list[int] = []
        total = 0
        for index, source in enumerate(self.sources):
            ctx.array(source, f"completion source {index}", dtype=ctx.wp.float64)
            offsets.append(total)
            total += int(source.size)
        self._offsets = tuple(offsets)
        self._status_offset = total
        total += len(checkpoint._entries)
        if total > 2**31 - 1:
            raise ctx.contract_error("completion payload exceeds the bounded index range")
        with ctx.scope():
            self._payload = ctx.wp.empty(total, dtype=ctx.wp.float64, device=ctx.device)
            self._readback = HostReadback(ctx, self._payload)
        reader: Any = self._readback  # Borrow the prepared host view without a transfer.
        self._views = tuple(
            reader._values[offset : offset + int(source.size)]
            for offset, source in zip(self._offsets, self.sources, strict=True)
        )
        self._kernels = load_kernels("dpt.kernels.control_readback")

    @property
    def device_bytes(self) -> int:
        return int(self._payload.size) * 8

    @property
    def host_bytes(self) -> int:
        return self._readback.host_bytes

    def consume(self) -> tuple[Any, ...]:
        """Deliver all captured statuses, then return the completed control views."""
        checkpoint: Any = self.checkpoint
        ctx = self.checkpoint.context
        with ctx.scope():
            for offset, source in zip(self._offsets, self.sources, strict=True):
                if source.size:
                    ctx.wp.copy(
                        self._payload,
                        source,
                        dest_offset=offset,
                        count=source.size,
                        stream=ctx.stream,
                    )
            for index, (_, status) in enumerate(checkpoint._entries):
                ctx.wp.copy(
                    checkpoint._device, status, dest_offset=index, count=1, stream=ctx.stream
                )
            ctx.wp.launch(
                self._kernels.pack_status,
                dim=len(checkpoint._entries),
                inputs=[checkpoint._device, self._status_offset],
                outputs=[self._payload],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
            payload = self._readback.read()
            codes = payload[self._status_offset :]
            # Conversion must not round or clear a corrupted completion word.
            if any(
                not (-(2**31) <= float(code) < 2**31) or float(code) != int(code) for code in codes
            ):
                raise ctx.contract_error("completion payload contains invalid status words")
            checkpoint._deliver(codes, consume=True)
        return self._views


@contextmanager
def tape_execution_scope() -> Generator[None, None, None]:
    """Reject overlapping DPT-owned recording/reverse passes across threads.

    Warp's ambient tape is process-global. External Warp tape users still need
    their own coordination; this lease protects DPT's FirstOrderPass owner.
    """
    global _tape_owner
    if not _tape_lock.acquire(blocking=False):
        raise ContractError("DPT tape execution is already owned; tape passes must not overlap")
    try:
        require_no_tape()
        _tape_owner = get_ident()
        yield
    finally:
        _tape_owner = None
        _tape_lock.release()


def assert_tape_owner() -> None:
    if _tape_owner is not None and _tape_owner != get_ident():
        raise ContractError("DPT tape execution belongs to another thread")


def active_tape() -> Any:
    # Warp 1.17 has no public ambient-tape accessor. Keep this compatibility
    # dependency isolated; explicit tape identity prevents omitted derivatives.
    try:
        runtime = importlib.import_module("warp._src.context").runtime
        return None if runtime is None else runtime.tape
    except (ImportError, AttributeError) as error:
        raise ContractError(
            "DPT cannot inspect Warp's ambient tape state. This compatibility boundary "
            "is tested with warp-lang 1.17.0; install the supported version or use a "
            "DPT release that supports your Warp version."
        ) from error


def require_no_tape(*, contract_error: type[ContractError] = ContractError) -> None:
    if _tape_owner is not None and _tape_owner != get_ident():
        raise contract_error("DPT tape execution belongs to another thread")
    if active_tape() is not None:
        raise contract_error(
            "this operation does not support recording on an ambient tape "
            "(including higher-order recording); Warp's ambient tape is process-global, "
            "so tape-recorded passes must not overlap across threads"
        )


def ensure_tape(tape: Any, *, contract_error: type[ContractError] = ContractError) -> None:
    if _tape_owner is not None and _tape_owner != get_ident():
        raise contract_error("DPT tape execution belongs to another thread")
    active = active_tape()
    if active is not None and active is not tape:
        raise contract_error(
            "pass the active Warp tape explicitly; Warp's ambient tape is process-global, "
            "so tape-recorded passes must not overlap across threads"
        )
