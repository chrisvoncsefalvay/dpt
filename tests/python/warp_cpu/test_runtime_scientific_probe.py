"""Execute the CUDA preflight's mathematical probe on Warp CPU for regression."""

import importlib
from typing import Any

import pytest

from dpt._runtime import cpu_test_context, prepare_context
from dpt.contracts import ContractError
from dpt.gpu_check import _probe_projection

wp: Any = importlib.import_module("warp")


def test_preflight_projector_and_adjoint_probe_on_cpu() -> None:
    with cpu_test_context():
        _probe_projection(wp, wp.get_device("cpu"))


@pytest.mark.parametrize("placement", ["read", "write", "read-only"])
def test_disjoint_rejects_real_strided_views(placement: str, warp_device: str) -> None:
    context = prepare_context(device=warp_device)
    base = wp.zeros(16, dtype=wp.float32, device=warp_device)
    strided = base[::2]
    overlapping = base[10:11]
    assert not strided.is_contiguous
    if placement == "read":
        reads, writes = [("strided", strided)], [("element10", overlapping)]
    elif placement == "write":
        reads, writes = [("element10", overlapping)], [("strided", strided)]
    else:
        reads, writes = [("strided", strided)], []
    with pytest.raises(ContractError, match="strided must be contiguous"):
        context.disjoint(reads, writes)


def test_disjoint_preserves_contiguous_byte_ranges(warp_device: str) -> None:
    context = prepare_context(device=warp_device)
    base = wp.zeros(16, dtype=wp.float32, device=warp_device)
    context.disjoint([("head", base[:8])], [("tail", base[8:])])
    context.disjoint([("head", base[:8])], [("empty", wp.empty(0, device=warp_device))])
    vectors = base.reshape((4, 4)).view(wp.vec4)
    context.disjoint([("vector_head", vectors[:2])], [("tail", base[8:])])
    with pytest.raises(ContractError, match="element10 overlaps head"):
        context.disjoint([("head", base[:12])], [("element10", base[10:11])])
    with pytest.raises(ContractError, match="element10 overlaps vector_head"):
        context.disjoint([("vector_head", vectors[:3])], [("element10", base[10:11])])
