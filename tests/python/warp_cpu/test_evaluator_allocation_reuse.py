"""Observe every owned Warp array in warmed prepared evaluator calls."""

import importlib
from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from helpers.evaluators import _evaluate, _evaluator

wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize("kind", ["primary", "spectral", "material"])
def test_prepared_evaluators_reuse_all_owned_warp_arrays(kind, warp_device, monkeypatch, request):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        evaluator, parameters = _evaluator(kind, warp_device)
        expected = _evaluate(evaluator, parameters)
        allocations: list[tuple[str, int]] = []
        original = wp.array.__init__

        def observe(array, *args, **kwargs):
            original(array, *args, **kwargs)
            # Views/reshape descriptors own no storage. Intercepting the array
            # constructor also observes unnamed temporary allocations.
            if getattr(array, "deleter", None) is not None:
                allocations.append((str(array.device), array.capacity))

        monkeypatch.setattr(wp.array, "__init__", observe)
        for _ in range(3):
            actual = _evaluate(evaluator, parameters)
            if parameters is None:
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            else:
                np.testing.assert_allclose(actual.loss, expected.loss, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(actual.gradient, expected.gradient, rtol=1e-7, atol=1e-9)
        request.node.user_properties.append(("owned_warp_array_allocations", len(allocations)))
        assert allocations == []
