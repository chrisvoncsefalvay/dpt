"""Host-only checks of Warp ownership, layout, aliases and tape compatibility."""

from threading import Event, Thread
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dpt import _runtime
from dpt._runtime import DeviceContext, active_tape, ensure_tape, require_no_tape
from dpt.contracts import ContractError


def test_private_tape_api_failure_has_actionable_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_runtime.importlib, "import_module", lambda _: SimpleNamespace())
    with pytest.raises(ContractError, match=r"warp-lang 1\.17\.0"):
        active_tape()
    monkeypatch.setattr(
        _runtime.importlib, "import_module", lambda _: SimpleNamespace(runtime=SimpleNamespace())
    )
    with pytest.raises(ContractError, match="ambient tape state"):
        active_tape()


def test_explicit_tape_identity_is_enforced_and_process_global_limit_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tape = object()
    monkeypatch.setattr(_runtime, "active_tape", lambda: tape)
    ensure_tape(tape)
    with pytest.raises(ContractError, match="process-global"):
        ensure_tape(object())
    with pytest.raises(ContractError, match="process-global"):
        require_no_tape()
    monkeypatch.setattr(_runtime, "active_tape", lambda: None)
    ensure_tape(None)
    require_no_tape()


def test_stream_identity_rejects_foreign_device_and_foreign_stream() -> None:
    device = SimpleNamespace(is_cuda=True)
    owning = SimpleNamespace(device=device, cuda_stream=17)
    wp = SimpleNamespace(get_stream=lambda _: owning)
    context = DeviceContext(wp, cast(Any, device), cast(Any, owning))
    assert context.assert_stream() is owning
    assert context.assert_stream(cast(Any, owning)) is owning
    wp.get_stream = lambda _: SimpleNamespace(device=device, cuda_stream=18)
    with pytest.raises(ContractError, match="owning stream"):
        context.assert_stream()
    assert context.assert_stream(cast(Any, owning)) is owning
    for wrong in (
        SimpleNamespace(device=device, cuda_stream=18),
        SimpleNamespace(device=object(), cuda_stream=17),
    ):
        with pytest.raises(ContractError, match="owning stream"):
            context.assert_stream(cast(Any, wrong))


def test_metadata_and_byte_range_aliasing_reject_invalid_views() -> None:
    class Array:
        def __init__(self, pointer: int, size: int, *, contiguous: bool = True) -> None:
            self.ptr = pointer
            self.size = size
            self.device = "cuda-fixture"
            self.dtype = "float32"
            self.shape = (size,)
            self.ndim = 1
            self.is_contiguous = contiguous

    wp: Any = SimpleNamespace(
        array=Array,
        float32="float32",
        types=SimpleNamespace(type_size_in_bytes=lambda _: 4),
    )
    context = DeviceContext(wp, cast(Any, "cuda-fixture"), None)
    left, adjacent, overlapping = Array(100, 4), Array(116, 4), Array(112, 4)
    assert context.array(left, "left", dtype="float32", shape=(4,)) == 4
    context.disjoint([("left", left)], [("adjacent", adjacent)])
    context.disjoint([("left", left)], [("empty", Array(100, 0))])
    with pytest.raises(ContractError, match="overlaps"):
        context.disjoint([("left", left)], [("overlapping", overlapping)])
    with pytest.raises(ContractError, match="overlaps"):
        context.disjoint([], [("left", left), ("same", left)])
    with pytest.raises(ContractError, match="contiguous"):
        context.array(Array(100, 4, contiguous=False), "strided", dtype="float32")
    with pytest.raises(ContractError, match="shape"):
        context.array(left, "left", dtype="float32", shape=(5,))


def test_scopes_serialize_host_stream_state_and_allow_nested_reentry() -> None:
    class Device:
        is_cuda = True

        def __init__(self) -> None:
            self.stream: Any = SimpleNamespace(cuda_stream=0)

        def __str__(self) -> str:
            return "cuda-thread-fixture"

    device = Device()
    entered_first, attempted_second, entered_second, release_first = (Event() for _ in range(4))
    stream_a = SimpleNamespace(device=device, cuda_stream=17)
    stream_b = SimpleNamespace(device=device, cuda_stream=19)
    observed: list[int] = []
    failures: list[str] = []

    class Scope:
        def __init__(self, stream: Any, sync_enter: bool) -> None:
            assert sync_enter is False
            self.stream = stream

        def __enter__(self) -> Any:
            self.saved = device.stream
            device.stream = self.stream
            return self.stream

        def __exit__(self, *_: object) -> None:
            device.stream = self.saved

    wp = SimpleNamespace(ScopedStream=Scope)
    first, second = (
        DeviceContext(wp, cast(Any, device), cast(Any, stream_a)),
        DeviceContext(wp, cast(Any, device), cast(Any, stream_b)),
    )

    def enqueue_first() -> None:
        try:
            with first.scope() as owned:
                assert owned is stream_a
                with first.scope():
                    assert device.stream is stream_a
                entered_first.set()
                assert release_first.wait(2)
                observed.append(device.stream.cuda_stream)
        except Exception as error:
            failures.append(repr(error))

    def enqueue_second() -> None:
        try:
            attempted_second.set()
            with second.scope():
                entered_second.set()
                observed.append(device.stream.cuda_stream)
        except Exception as error:
            failures.append(repr(error))

    a, b = Thread(target=enqueue_first), Thread(target=enqueue_second)
    a.start()
    assert entered_first.wait(2)
    b.start()
    try:
        assert attempted_second.wait(2)
        assert not entered_second.wait(0.05)
    finally:
        release_first.set()
        a.join(2)
        b.join(2)
    assert not a.is_alive() and not b.is_alive()
    assert not failures
    assert observed == [17, 19]
    assert device.stream.cuda_stream == 0
