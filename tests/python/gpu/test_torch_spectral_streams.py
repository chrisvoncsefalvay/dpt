"""Active spectral factors use explicit caller and operator CUDA stream ordering."""

import gc
import importlib

import numpy as np
import pytest

from helpers.torch_spectral_inputs import (
    evaluate,
    spectral_problem,
    spectral_reference,
    tensor_inputs,
)

torch = importlib.import_module("torch")
wp = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("shared", [False, True])
def test_active_spectral_foreign_streams_and_allocator_lifetimes_without_global_wait(
    shared, monkeypatch
):
    owner = torch.cuda.Stream(device="cuda:0")
    producer = torch.cuda.Stream(device="cuda:0")
    caller = torch.cuda.Stream(device="cuda:0")
    backward = torch.cuda.Stream(device="cuda:0")
    with torch.cuda.stream(producer):
        problem = spectral_problem(
            torch,
            wp,
            "cuda:0",
            shared_weights=shared,
            shared_response=shared,
            stream=wp.stream_from_torch(owner),
        )
        with torch.no_grad():
            problem.weights.mul_(0.8)
            problem.response.mul_(1.1)
    caller.wait_stream(producer)
    seed_array = np.linspace(-0.3, 0.8, 6).reshape(2, 3)
    expected, gradients = spectral_reference(problem, seed_array)

    def forbid_global_wait(*args, **kwargs):
        raise AssertionError("the spectral adapter must not synchronise the whole CUDA device")

    monkeypatch.setattr(torch.cuda, "synchronize", forbid_global_wait)
    with torch.cuda.stream(caller):
        result = evaluate(problem)
        seed = torch.tensor(seed_array, dtype=torch.float64, device="cuda:0")
    backward.wait_stream(caller)
    with torch.cuda.stream(backward):
        result.backward(seed)
        retained = [value.grad for value in tensor_inputs(problem) if value.requires_grad]
    torch.cuda.current_stream("cuda:0").wait_stream(backward)
    del seed
    gc.collect()
    with torch.cuda.stream(caller):
        for _ in range(32):
            churn = torch.empty(4096, dtype=torch.float32, device="cuda:0")
            churn.fill_(13)
    np.testing.assert_allclose(result.detach().cpu().numpy(), expected, rtol=3e-12, atol=3e-12)
    assert all(torch.isfinite(value).all().item() for value in retained)
    for name, gradient in gradients.items():
        np.testing.assert_allclose(getattr(problem, name).grad.cpu().numpy(), gradient, rtol=6e-6)
