"""Sampling bindings preserve complete records across repeated calls and resets."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from dpt.transport import HistoryBatch, SamplingDesign, SamplingIdentity, TransportError
from dpt.transport.kernels import Parameters, RandomDesign
from dpt.transport.sampling import _bind_sampling


def _workspace() -> Any:
    return SimpleNamespace(
        _parameters=Parameters(),
        _kernels=SimpleNamespace(RandomDesign=RandomDesign),
        _sampling=None,
    )


def _bytes(record: Any) -> bytes:
    return bytes(record.__ctype__())


def test_repeated_iid_binding_preserves_canonical_record() -> None:
    workspace = _workspace()
    batch = HistoryBatch(73, 0, 64)
    expected = _bytes(RandomDesign())
    for _ in range(3):
        record = _bind_sampling(workspace, batch)
        assert _bytes(record) == expected


def test_iid_binding_resets_noncanonical_zero_kind_and_design_records() -> None:
    workspace = _workspace()
    batch = HistoryBatch(73, 0, 64)
    expected = _bytes(RandomDesign())
    for kind in (0, 2):
        record = RandomDesign()
        record.kind = kind
        record.seed = 73
        record.count = 64
        workspace._parameters.sampling = record
        assert _bytes(workspace._parameters.sampling) != expected
        bound = _bind_sampling(workspace, batch)
        assert _bytes(bound) == expected
        assert _bytes(workspace._parameters.sampling) == expected


def test_design_binding_preserves_identity_validation_and_record() -> None:
    workspace = _workspace()
    identity = SamplingIdentity(SamplingDesign("stratified", 2, 4), 73, 0, 64)
    batch = HistoryBatch(73, 0, 64).with_sampling(identity)
    record = RandomDesign()
    record.kind = 1
    record.seed = 73
    record.count = 64
    workspace._sampling = SimpleNamespace(identity=identity, _record=record)
    assert _bind_sampling(workspace, batch) is record
    assert _bytes(workspace._parameters.sampling) == _bytes(record)
    _bind_sampling(workspace, HistoryBatch(74, 0, 64))
    assert _bytes(workspace._parameters.sampling) == _bytes(RandomDesign())
    workspace._sampling = None
    with pytest.raises(TransportError, match="identity"):
        _bind_sampling(workspace, cast(Any, batch))
