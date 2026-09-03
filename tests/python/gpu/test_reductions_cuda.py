"""Packed FP64 tree regressions independent of any scientific contribution kernel."""

import importlib
import math
from typing import Any

import pytest

from dpt._reductions import prepare_reduction
from dpt._runtime import prepare_context

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("tile", [128, 256])
@pytest.mark.parametrize("components", [1, 6, 12])
def test_reusable_packed_tree_respects_logical_rows(tile: int, components: int) -> None:
    ctx = prepare_context()
    capacity = tile * tile + 1
    tree = prepare_reduction(ctx, capacity, tile=tile, components=12)
    values = [float((index % 17) - 8) for index in range(capacity * components)]
    source = wp.array(values, dtype=wp.float64, device=ctx.device)
    output = wp.empty(components, dtype=wp.float64, device=ctx.device)
    status = wp.zeros(1, dtype=wp.int32, device=ctx.device)
    # Reuse after a long sum exposes stale-tail reads and empty-sum reads.
    for size in (capacity, tile + 1, tile - 1, 1, 0):
        tree.finish(size, output, status, components=components, source=source)
        ctx.wp.synchronize_stream(ctx.stream)
        expected = [
            math.fsum(values[c : size * components : components]) for c in range(components)
        ]
        assert output.numpy().tolist() == expected
        assert status.numpy().tolist() == [0]


@pytest.mark.parametrize("binary32", [False, True])
def test_final_store_accumulates_before_rounding_and_flags_range(binary32: bool) -> None:
    ctx = prepare_context()
    tree = prepare_reduction(ctx, 1)
    dtype = wp.float32 if binary32 else wp.float64
    source = wp.array([2.0**-24], dtype=wp.float64, device=ctx.device)
    output = wp.ones(1, dtype=dtype, device=ctx.device)
    status = wp.zeros(1, dtype=wp.int32, device=ctx.device)
    tree.finish(1, output, status, source=source, accumulate=True)
    expected = 1.0 if binary32 else 1.0 + 2.0**-24
    assert output.numpy().tolist() == [expected]
    source = wp.array([1e300 if binary32 else math.inf], dtype=wp.float64, device=ctx.device)
    tree.finish(1, output, status, source=source)
    assert status.numpy().tolist() == [2]
