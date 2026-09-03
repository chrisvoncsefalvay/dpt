"""Real Python collector contracts for prepared native graph lifetimes."""

import gc
import weakref
from collections.abc import Generator

import pytest

from dpt._runtime import _capture_lifetime
from dpt.contracts import ContractError


@pytest.fixture
def collector_state() -> Generator[list[str], None, None]:
    enabled, thresholds = gc.isenabled(), gc.get_threshold()
    collections: list[str] = []

    def witness(phase: str, info: dict[str, int]) -> None:
        if phase == "start":
            collections.append(phase)

    # Isolate explicit collections from allocation-threshold timing in this test.
    gc.set_threshold(10**8, *thresholds[1:])
    gc.callbacks.append(witness)
    try:
        yield collections
    finally:
        gc.callbacks.remove(witness)
        gc.set_threshold(*thresholds)
        if enabled:
            gc.enable()
        else:
            gc.disable()


def test_outer_capture_collects_once_before_body_and_restores_enabled_state(
    collector_state: list[str],
) -> None:
    gc.enable()
    with _capture_lifetime():
        assert not gc.isenabled()
        assert collector_state == ["start"]
    assert gc.isenabled()


def test_nested_scope_retains_cycles_until_after_outer_capture(
    collector_state: list[str],
) -> None:
    class ResidentCycle:
        def __init__(self) -> None:
            self.resident = self

    gc.enable()
    with _capture_lifetime():
        resident = ResidentCycle()
        observed = weakref.ref(resident)
        del resident
        with _capture_lifetime(inherited_capture=True):
            assert not gc.isenabled()
            assert observed() is not None
            assert collector_state == ["start"]
        assert not gc.isenabled()
        assert observed() is not None
    assert gc.isenabled()
    gc.collect()
    assert observed() is None


def test_exception_restores_collection_and_next_capture_collects_once(
    collector_state: list[str],
) -> None:
    gc.enable()
    with pytest.raises(ValueError, match="body failed"):
        with _capture_lifetime():
            raise ValueError("body failed")
    assert gc.isenabled()
    with _capture_lifetime():
        assert collector_state == ["start", "start"]
    assert gc.isenabled()


def test_callers_disabled_collection_setting_is_preserved_without_collection(
    collector_state: list[str],
) -> None:
    gc.disable()
    with _capture_lifetime():
        with _capture_lifetime(inherited_capture=True):
            assert not gc.isenabled()
    assert not gc.isenabled()
    assert collector_state == []


def test_unowned_inherited_capture_is_rejected_without_mutating_collection(
    collector_state: list[str],
) -> None:
    for enabled in (True, False):
        gc.enable() if enabled else gc.disable()
        with pytest.raises(ContractError, match="outside an active capture"):
            with _capture_lifetime(inherited_capture=True):
                raise AssertionError("an inherited capture must not enter this body")
        assert gc.isenabled() is enabled
        assert collector_state == []
