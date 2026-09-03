"""Prepared completion transfers preserve controls and every diagnostic epoch."""

import importlib
from typing import Any

import pytest

from dpt._runtime import CompletionReadback, StatusCheckpoint, StatusReadback, prepare_context
from dpt.contracts import ContractError


class _Workspace:
    def __init__(self, context: Any) -> None:
        self.context = context
        self._status = context.wp.zeros(1, dtype=context.wp.int32, device=context.device)
        self.reader = StatusReadback(context, self._status)

    def check_status(self) -> None:
        if code := self.reader.read():
            raise ContractError(f"workspace failure {code}")

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()


def _prepared(device: str) -> tuple[Any, ...]:
    wp: Any = importlib.import_module("warp")
    context = prepare_context(device=device)
    first, second = _Workspace(context), _Workspace(context)
    controls = (
        wp.array([1.25, -2.0, 1e-200], dtype=wp.float64, device=context.device),
        wp.array([float(2**53 - 1)], dtype=wp.float64, device=context.device),
    )
    checkpoint = StatusCheckpoint(context, (first, second))
    completion = CompletionReadback(checkpoint, controls)
    completion.consume()  # Warm the actual pack kernel before allocation checks.
    return wp, context, first, second, controls, completion


def test_completion_reuses_control_views_with_one_copy_and_no_new_buffers(
    warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    wp, context, first, second, controls, completion = _prepared(warp_device)
    views = completion.consume()
    assert views[0].tolist() == [1.25, -2.0, 1e-200]
    assert views[1].tolist() == [float(2**53 - 1)]
    assert completion.host_bytes == completion.device_bytes == 48
    copies: list[Any] = []
    waits: list[Any] = []
    copy, synchronize = wp.copy, wp.synchronize_stream

    def counted_copy(destination: Any, source: Any, **kwargs: Any) -> None:
        if destination.ptr == completion._readback._host.ptr:
            copies.append(kwargs.get("stream"))
        copy(destination, source, **kwargs)

    def counted_wait(stream: Any) -> None:
        waits.append(stream)
        synchronize(stream)

    def unexpected_allocation(*_: Any, **__: Any) -> Any:
        raise AssertionError("completion allocated after preparation")

    monkeypatch.setattr(wp, "copy", counted_copy)
    monkeypatch.setattr(wp, "synchronize_stream", counted_wait)
    for name in ("empty", "zeros", "empty_like"):
        monkeypatch.setattr(wp, name, unexpected_allocation)
    with context.scope():
        controls[0].fill_(6.0)
        controls[1].fill_(-0.5)
    following = completion.consume()
    assert following[0] is views[0] and following[1] is views[1]
    assert views[0].tolist() == [6.0] * 3
    assert views[1].tolist() == [-0.5]
    assert copies == [context.stream]
    assert waits == ([context.stream] if context.device.is_cuda else [])
    assert (first.reader.read(), second.reader.read()) == (0, 0)


@pytest.mark.parametrize("codes", [(3, 5), (2**31 - 1, -(2**31))])
def test_completion_delivers_every_exact_int32_code_before_retry(
    warp_device: str, codes: tuple[int, int]
) -> None:
    _, context, first, second, _, completion = _prepared(warp_device)
    with context.scope():
        first._status.fill_(codes[0])
        second._status.fill_(codes[1])
    with pytest.raises(ContractError, match=f"failure {codes[0]}") as error:
        completion.consume()
    assert tuple(item[1] for item in error.value.__dict__["status_diagnostics"]) == codes
    assert (first.reader.read(), second.reader.read()) == (0, 0)
    assert completion.consume()[0].tolist() == [1.25, -2.0, 1e-200]


@pytest.mark.parametrize("failure", ["copy", "wait", "decoder"])
def test_failed_completion_keeps_all_pending_statuses(
    warp_device: str, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    wp, context, first, second, _, completion = _prepared(warp_device)
    with context.scope():
        first._status.fill_(3)
        second._status.fill_(5)

    def fail(*_: Any, **__: Any) -> Any:
        raise RuntimeError(f"completion {failure} failed")

    with monkeypatch.context() as patch:
        if failure == "copy":
            copy = wp.copy

            def failed_payload_copy(destination: Any, source: Any, **kwargs: Any) -> None:
                if destination.ptr == completion._readback._host.ptr:
                    fail()
                copy(destination, source, **kwargs)

            patch.setattr(wp, "copy", failed_payload_copy)
        elif failure == "wait":
            if context.device.is_cuda:
                patch.setattr(wp, "synchronize_stream", fail)
            else:
                # CPU copies complete synchronously; fail immediately after
                # the real snapshot to exercise the same delivery boundary.
                read = completion._readback.read

                def failed_completed_read() -> Any:
                    read()
                    return fail()

                patch.setattr(
                    type(completion._readback), "read", lambda self: failed_completed_read()
                )
        else:
            patch.setattr(second, "check_status", fail)
        with pytest.raises(RuntimeError, match=f"completion {failure} failed"):
            completion.consume()
    assert (first.reader.read(), second.reader.read()) == (3, 5)
    with pytest.raises(ContractError) as error:
        completion.consume()
    assert {item[1] for item in error.value.__dict__["status_diagnostics"]} == {3, 5}
    assert (first.reader.read(), second.reader.read()) == (0, 0)


def test_completion_reset_failure_retains_delivered_diagnostics_and_uncleared_word(
    warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, context, first, second, _, completion = _prepared(warp_device)
    with context.scope():
        first._status.fill_(3)
        second._status.fill_(5)

    def failed_reset() -> None:
        raise RuntimeError("reset failed")

    with monkeypatch.context() as patch:
        patch.setattr(second, "clear_status", failed_reset)
        with pytest.raises(RuntimeError, match="reset failed") as error:
            completion.consume()
    assert {item[1] for item in error.value.__dict__["status_diagnostics"]} == {3, 5}
    assert (first.reader.read(), second.reader.read()) == (0, 5)
    with pytest.raises(ContractError, match="failure 5"):
        completion.consume()
    assert second.reader.read() == 0
