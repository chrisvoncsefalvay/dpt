"""Real expected-score CUDA handoffs and owned outstanding-forward lifetimes."""

import gc
import importlib

import numpy as np
import pytest

from helpers.torch_expected_transport_inputs import analytic, coordinates, evaluate, layer_problem

torch = pytest.importorskip("torch", reason="optional Torch runtime is not installed")
wp = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("reverse_order", [False, True])
def test_expected_score_foreign_streams_and_allocator_lifetimes(reverse_order, monkeypatch):
    wp.init()
    owner = torch.cuda.Stream(device="cuda:0")
    producer = torch.cuda.Stream(device="cuda:0")
    caller = torch.cuda.Stream(device="cuda:0")
    backward = torch.cuda.Stream(device="cuda:0")
    with torch.cuda.stream(producer):
        p = layer_problem(torch, wp, "cuda:0", stream=wp.stream_from_torch(owner))
        values = coordinates(p)

    def forbid_global_wait(*args, **kwargs):
        raise AssertionError("expected-score adapter must not synchronise the whole CUDA device")

    monkeypatch.setattr(torch.cuda, "synchronize", forbid_global_wait)
    caller.wait_stream(producer)
    with torch.cuda.stream(caller):
        expected, gradients = analytic(p, 0.5)
        first = evaluate(p).signal
        second = evaluate(p).signal
        seed = torch.full_like(first, 0.5)
    backward.wait_stream(caller)
    with torch.cuda.stream(backward):
        results = (second, first) if reverse_order else (first, second)
        for result in results:
            result.backward(seed)
        retained = tuple(value.grad for value in values)
        output = first.detach()
    default = torch.cuda.default_stream("cuda:0")
    default.wait_stream(backward)
    with torch.cuda.stream(default):
        np.testing.assert_allclose(output.cpu().numpy(), [[expected]], rtol=3e-13, atol=3e-13)
        for value, target in zip(retained, gradients, strict=True):
            np.testing.assert_allclose(value.cpu().numpy(), 2 * target, rtol=3e-13, atol=3e-13)
    del p, values, first, second, result, results, seed
    gc.collect()
    with torch.cuda.stream(caller):
        for _ in range(32):
            churn = torch.empty(4096, dtype=torch.float64, device="cuda:0")
            churn.fill_(13)
    default.wait_stream(caller)
    with torch.cuda.stream(default):
        np.testing.assert_allclose(output.cpu().numpy(), [[expected]], rtol=3e-13, atol=3e-13)
        for value, target in zip(retained, gradients, strict=True):
            np.testing.assert_allclose(value.cpu().numpy(), 2 * target, rtol=3e-13, atol=3e-13)
