"""Explicit backend selection, prepared readbacks and process-global tape ownership."""

import importlib
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dpt._runtime import (
    DeviceContext,
    HostReadback,
    StatusCheckpoint,
    StatusReadback,
    ensure_tape,
    prepare_context,
)
from dpt.autodiff import FirstOrderPass
from dpt.contracts import ContractError
from dpt.gpu_check import _probe_projection
from dpt.transmission import prepare_transmission, transmit


def test_explicit_cpu_preparation_runs_the_scientific_probe_without_test_scope() -> None:
    wp: Any = importlib.import_module("warp")
    context = prepare_context(device="cpu")
    assert not context.device.is_cuda
    assert context.stream is None
    _probe_projection(wp, context.device)
    with pytest.raises(ContractError, match="CPU execution"):
        prepare_context(device="cpu", stream=cast(Any, object()))
    with (
        wp.ScopedDevice("cpu"),
        pytest.raises(ContractError, match="select device='cpu' explicitly"),
    ):
        prepare_context(device=None)


def test_requested_cuda_is_never_redirected_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from dpt import _runtime

    seen: list[Any] = []

    class Warp:
        @staticmethod
        def get_device(device: Any) -> None:
            seen.append(device)
            raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(_runtime, "load_warp", lambda: Warp())
    with pytest.raises(ContractError, match="select device='cpu' explicitly"):
        prepare_context()
    assert seen == ["cuda:0"]


def test_prepared_readback_reuses_host_storage_and_consume_is_opt_in(
    warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    wp: Any = importlib.import_module("warp")
    context = prepare_context(device=warp_device)
    status = wp.zeros(1, dtype=wp.int32, device=warp_device)
    values = wp.array([1.0, 2.0], dtype=wp.float64, device=warp_device)
    reader = StatusReadback(context, status)
    vector_reader = HostReadback(context, values)
    assert reader.host_bytes == 4
    assert vector_reader.host_bytes == 16
    first = vector_reader.read()

    def unexpected_allocation(*_: Any, **__: Any) -> Any:
        raise AssertionError("readback allocated after preparation")

    monkeypatch.setattr(wp, "empty", unexpected_allocation)
    with context.scope():
        status.fill_(7)
        values.fill_(3.0)
    assert vector_reader.read() is first
    assert first.tolist() == [3.0, 3.0]
    assert reader.read() == 7
    assert reader.read() == 7
    assert reader.read(reset=True) == 7
    assert reader.read() == 0


def test_failed_status_copy_preserves_the_pending_failure(
    warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    wp: Any = importlib.import_module("warp")
    context = prepare_context(device=warp_device)
    status = wp.array([9], dtype=wp.int32, device=warp_device)
    reader = StatusReadback(context, status)
    original_copy = wp.copy

    def failed_copy(*_: Any, **__: Any) -> None:
        raise RuntimeError("readback failed")

    monkeypatch.setattr(wp, "copy", failed_copy)
    with pytest.raises(RuntimeError, match="readback failed"):
        reader.read(reset=True)
    monkeypatch.setattr(wp, "copy", original_copy)
    assert reader.read() == 9


@pytest.mark.parametrize("dependency", ["event", "stream"])
def test_context_dependency_orders_an_explicit_producer_before_readback(
    warp_device: str, dependency: str
) -> None:
    wp: Any = importlib.import_module("warp")
    context = prepare_context(device=warp_device)
    context.wait_for(None)
    if not context.device.is_cuda:
        with pytest.raises(ContractError, match="CPU execution cannot wait"):
            context.wait_for(cast(Any, object()))
        return
    producer = wp.Stream(context.device)
    consumer = wp.Stream(context.device)
    with wp.ScopedStream(producer, sync_enter=False):
        source = wp.empty(1, dtype=wp.int32, device=context.device)
        source.fill_(11)
        event = wp.Event(context.device)
        producer.record_event(event)
    context = prepare_context(device=warp_device, stream=consumer)
    reader = StatusReadback(context, source)
    context.wait_for(event if dependency == "event" else producer)
    assert reader.read() == 11
    with pytest.raises(ContractError, match="Warp CUDA event or stream"):
        context.wait_for(cast(Any, object()))


def test_composed_checkpoint_preserves_pending_errors_and_consumes_all_delivered_errors(
    warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    wp: Any = importlib.import_module("warp")
    context = prepare_context(device=warp_device)

    class Workspace:
        def __init__(self, code: int) -> None:
            self._status = wp.array([code], dtype=wp.int32, device=warp_device)
            self.reader = StatusReadback(context, self._status)

        def check_status(self) -> None:
            if code := self.reader.read():
                raise ContractError(f"workspace failure {code}")

        def clear_status(self) -> None:
            with context.scope():
                self._status.zero_()

    first, second = Workspace(3), Workspace(5)
    checkpoint = StatusCheckpoint(context, [first, second])
    assert checkpoint.device_bytes == checkpoint.host_bytes == 8
    with pytest.raises(ContractError, match="failure 3"):
        checkpoint.begin()
    assert (first.reader.read(), second.reader.read()) == (3, 5)
    with pytest.raises(ContractError, match="failure 3") as delivered:
        checkpoint.consume()
    assert any("status 3" in note for note in delivered.value.__notes__)
    assert any("status 5" in note for note in delivered.value.__notes__)
    assert (first.reader.read(), second.reader.read()) == (0, 0)

    waits: list[Any] = []
    synchronize = wp.synchronize_stream

    def count_wait(stream: Any) -> None:
        waits.append(stream)
        synchronize(stream)

    monkeypatch.setattr(wp, "synchronize_stream", count_wait)
    checkpoint.check()
    assert waits == ([context.stream] if context.device.is_cuda else [])


def test_borrowed_checkpoint_allocates_no_buffers_and_failed_later_read_preserves_all_statuses(
    warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    wp: Any = importlib.import_module("warp")
    context = prepare_context(device=warp_device)

    class Workspace:
        def __init__(self, code: int) -> None:
            self._status = wp.array([code], dtype=wp.int32, device=warp_device)
            self._status_readback = StatusReadback(context, self._status)

        def check_status(self) -> None:
            if code := self._status_readback.read():
                raise ContractError(f"workspace failure {code}")

        def clear_status(self) -> None:
            with context.scope():
                self._status.zero_()

    first, second = Workspace(3), Workspace(5)

    def unexpected_allocation(*_: Any, **__: Any) -> Any:
        raise AssertionError("borrowed checkpoint allocated device/host staging")

    monkeypatch.setattr(wp, "empty", unexpected_allocation)
    checkpoint = StatusCheckpoint(context, (first, second), aggregate=False)
    assert checkpoint.device_bytes == checkpoint.host_bytes == 0
    copy = wp.copy

    def failed_second_copy(destination: Any, source: Any, **kwargs: Any) -> None:
        if source.ptr == second._status.ptr:
            raise RuntimeError("second read failed")
        copy(destination, source, **kwargs)

    monkeypatch.setattr(wp, "copy", failed_second_copy)
    with pytest.raises(RuntimeError, match="second read failed"):
        checkpoint.consume()
    monkeypatch.setattr(wp, "copy", copy)
    assert (first._status_readback.read(), second._status_readback.read()) == (3, 5)
    with pytest.raises(ContractError, match="failure 3") as delivered:
        checkpoint.consume()
    assert {item[1] for item in delivered.value.__dict__["status_diagnostics"]} == {3, 5}
    assert (first._status_readback.read(), second._status_readback.read()) == (0, 0)


def test_first_order_owner_rejects_other_threads_and_releases_lease() -> None:
    entered, release = Event(), Event()
    passes: list[FirstOrderPass] = []
    failures: list[BaseException] = []

    def record() -> None:
        try:
            with FirstOrderPass([]) as owner:
                passes.append(owner)
                entered.set()
                assert release.wait(5)
            owner.close()
        except BaseException as error:
            failures.append(error)

    thread = Thread(target=record)
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(ContractError, match="must not overlap"):
            with FirstOrderPass([]):
                pass
        with pytest.raises(ContractError, match="another thread"):
            ensure_tape(passes[0].tape)
        with pytest.raises(ContractError, match="another thread"):
            passes[0].__exit__(None, None, None)
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert not failures
    with FirstOrderPass([]) as subsequent:
        pass
    subsequent.close()


def test_scopes_on_distinct_devices_preserve_warps_global_default_device() -> None:
    runtime = SimpleNamespace(default_device="initial")
    first_device = SimpleNamespace(is_cuda=False, name="first-device")
    second_device = SimpleNamespace(is_cuda=False, name="second-device")
    entered, attempted, entered_second, release = (Event() for _ in range(4))
    failures: list[BaseException] = []

    class Scope:
        def __init__(self, device: Any) -> None:
            self.device = device

        def __enter__(self) -> Any:
            self.saved = runtime.default_device
            runtime.default_device = self.device
            return self.device

        def __exit__(self, *_: Any) -> None:
            runtime.default_device = self.saved

    wp = SimpleNamespace(ScopedDevice=Scope)
    first = DeviceContext(wp, cast(Any, first_device), None)
    second = DeviceContext(wp, cast(Any, second_device), None)

    def first_scope() -> None:
        try:
            with first.scope():
                entered.set()
                assert release.wait(5)
                assert runtime.default_device is first_device
        except BaseException as error:
            failures.append(error)

    def second_scope() -> None:
        try:
            attempted.set()
            with second.scope():
                entered_second.set()
                assert runtime.default_device is second_device
        except BaseException as error:
            failures.append(error)

    a, b = Thread(target=first_scope), Thread(target=second_scope)
    a.start()
    assert entered.wait(5)
    b.start()
    try:
        assert attempted.wait(5)
        assert not entered_second.wait(0.05)
    finally:
        release.set()
        a.join(5)
        b.join(5)
    assert not a.is_alive() and not b.is_alive()
    assert not failures
    assert runtime.default_device == "initial"


def test_pending_failure_prevents_tape_entry_without_leaking_the_lease() -> None:
    class FailedWorkspace:
        def check_status(self) -> None:
            raise ContractError("pending unchecked failure")

        def clear_status(self) -> None:
            raise AssertionError("pending error must remain observable")

    with pytest.raises(ContractError, match="pending unchecked failure"):
        with FirstOrderPass([FailedWorkspace()]):
            pass
    with FirstOrderPass([]) as subsequent:
        pass
    subsequent.close()


def test_explicit_cpu_tape_backward_preserves_seed_and_recovers_after_invalid_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dpt import autodiff

    wp: Any = importlib.import_module("warp")
    workspace = prepare_transmission(max_pixels=2, device="cpu")
    depth = wp.ones(2, dtype=wp.float32, device="cpu", requires_grad=True)
    output = wp.empty(2, dtype=wp.float32, device="cpu", requires_grad=True)
    seed = wp.array([2.0, -3.0], dtype=wp.float32, device="cpu")
    with FirstOrderPass([workspace]) as owner:
        transmit(depth, out_log_T=output, workspace=workspace, tape=owner.tape)

    def unexpected_context_preparation(**_: Any) -> None:
        raise AssertionError("backward discarded the workspace's owning context")

    monkeypatch.setattr(autodiff, "prepare_context", unexpected_context_preparation)
    owner.backward(output, seed)
    assert depth.grad.numpy().tolist() == [-2.0, 3.0]
    assert seed.numpy().tolist() == [2.0, -3.0]
    owner.close()
    assert depth.grad.numpy().tolist() == [0.0, 0.0]
    assert output.grad.numpy().tolist() == [0.0, 0.0]
    assert seed.numpy().tolist() == [2.0, -3.0]
    with FirstOrderPass([workspace]) as bad:
        transmit(depth, out_log_T=output, workspace=workspace, tape=bad.tape)
    with pytest.raises(ContractError, match="shape"):
        bad.backward(output, wp.ones(1, dtype=wp.float32, device="cpu"))
    assert depth.grad.numpy().tolist() == [0.0, 0.0]
    with FirstOrderPass([workspace]) as recovered:
        pass
    recovered.close()


@pytest.mark.parametrize("mismatch", ["runtime", "device", "stream"])
def test_first_order_pass_rejects_ambiguous_owners_before_status_mutation(mismatch: str) -> None:
    wp = object()
    device = SimpleNamespace(is_cuda=True, alias="cuda:0")
    stream = SimpleNamespace(device=device, cuda_stream=17)
    first = DeviceContext(wp, cast(Any, device), cast(Any, stream))
    second = DeviceContext(
        object() if mismatch == "runtime" else wp,
        cast(
            Any, SimpleNamespace(is_cuda=True, alias="cuda:1") if mismatch == "device" else device
        ),
        cast(
            Any, SimpleNamespace(device=device, cuda_stream=23) if mismatch == "stream" else stream
        ),
    )

    class Workspace:
        def __init__(self, context: DeviceContext) -> None:
            self.context = context

        def check_status(self) -> None:
            raise AssertionError("ambiguous ownership reached a status read")

        def clear_status(self) -> None:
            raise AssertionError("ambiguous ownership reached a status reset")

    with pytest.raises(ContractError, match="workspaces must share"):
        with FirstOrderPass([Workspace(first), Workspace(second)]):
            raise AssertionError("ambiguous ownership entered a recording")


def test_gradient_free_output_closes_recorded_cpu_pass_on_workspace_context() -> None:
    wp: Any = importlib.import_module("warp")
    workspace = prepare_transmission(max_pixels=2, device="cpu")
    depth = wp.ones(2, dtype=wp.float32, device="cpu", requires_grad=True)
    output = wp.empty(2, dtype=wp.float32, device="cpu", requires_grad=True, retain_grad=True)
    seed = wp.array([2.0, -3.0], dtype=wp.float32, device="cpu")
    with FirstOrderPass([workspace]) as owner:
        transmit(depth, out_log_T=output, workspace=workspace, tape=owner.tape)
    depth.grad.fill_(7.0)
    output.grad.fill_(11.0)
    with pytest.raises(ContractError, match="preallocated gradient"):
        owner.backward(wp.empty(2, dtype=wp.float32, device="cpu"), seed)
    assert depth.grad.numpy().tolist() == [0.0, 0.0]
    assert output.grad.numpy().tolist() == [0.0, 0.0]
    assert seed.numpy().tolist() == [2.0, -3.0]
    owner.close()
    with FirstOrderPass([]) as recovered:
        pass
    recovered.close()
