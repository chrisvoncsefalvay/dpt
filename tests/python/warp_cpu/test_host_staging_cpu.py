"""Production CPU evaluators remain usable without CUDA host allocation."""

import importlib
import math
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.transport.inverse import prepare_transport_inverse
from dpt.transport.rng import HistoryBatch
from helpers import evaluators
from helpers import transport_inputs as transport

wp: Any = importlib.import_module("warp")


@pytest.fixture
def no_pinned_allocator(monkeypatch: pytest.MonkeyPatch) -> None:
    # CUDA-enabled Warp wheels implement this allocator through cudaMallocHost.
    # Simulate its unavailability even on platforms whose CPU wheel has a stub.
    def unavailable(size_in_bytes: int) -> None:
        raise RuntimeError("CPU pinned allocator is unavailable without a CUDA driver")

    monkeypatch.setattr(wp.get_device("cpu").get_allocator(pinned=True), "allocate", unavailable)
    with pytest.raises(RuntimeError, match="without a CUDA driver"):
        wp.empty(1, device="cpu", pinned=True)


@pytest.mark.parametrize("kind", ["primary", "spectral", "material"])
def test_cpu_evaluators_and_secant_need_no_cuda_host_allocator(kind: str, no_pinned_allocator):
    with cpu_test_context():
        evaluator, parameters = evaluators._evaluator(kind, "cpu")
        if kind == "material":
            evaluator._prepare_secant()
            assert not evaluator._secant_host.pinned
            assert not evaluator._secant_host_status.pinned
            assert math.isfinite(evaluator.evaluate())
            assert np.all(np.isfinite(evaluator.gradient.numpy()))
        else:
            assert not evaluator._pose_host.pinned
            if kind == "spectral":
                assert not evaluator._parameters_host.pinned
            result = evaluator(parameters)
            assert math.isfinite(result.loss)
            assert all(math.isfinite(value) for value in result.gradient)


def test_cpu_transport_gradient_and_local_model_need_no_cuda_host_allocator(no_pinned_allocator):
    with cpu_test_context():
        original = transport.absorbing_oracle("continuous-absorption", 32, "cpu")
        oracle = prepare_transport_inverse(
            original.workspace,
            source=original.source,
            parameters=original.parameters,
            observation=original.observation,
            pixel_weights=original.pixel_weights,
            base_density=(1.0,),
            local_model=True,
        )
        assert not oracle._parameter_host.pinned
        assert not oracle._scalar_host.pinned
        assert not oracle._model_host.pinned
        first, second = HistoryBatch(341, 0, 32), HistoryBatch(341, 32, 32)
        gradient, curvature = oracle.model_replicate((0.0,), first, second)
        expected_gradient = (math.exp(-1.0) - math.exp(-1.5)) * -math.exp(-1.0)
        np.testing.assert_allclose(gradient, (expected_gradient,), rtol=1e-12)
        np.testing.assert_allclose(curvature, ((math.exp(-2.0),),), rtol=1e-12)
