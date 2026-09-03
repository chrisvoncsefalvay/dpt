"""Actual CUDA seed-boundary regressions; invalid seeds never reach Warp's copy."""

import importlib
from typing import Any

import pytest

from dpt.autodiff import FirstOrderPass
from dpt.contracts import ContractError
from dpt.transmission import prepare_transmission, transmit

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("kind", ["short", "int32", "float64", "cpu", "strided", "rank"])
def test_invalid_seed_closes_the_pass_without_a_partial_cotangent(kind: str) -> None:
    workspace = prepare_transmission(max_pixels=2)
    depth = wp.ones(2, dtype=wp.float32, device="cuda:0", requires_grad=True)
    output = wp.empty(2, dtype=wp.float32, device="cuda:0", requires_grad=True)
    with FirstOrderPass([workspace]) as reverse:
        transmit(depth, out_log_T=output, workspace=workspace, tape=reverse.tape)
    if kind == "strided":
        seed = wp.ones(4, dtype=wp.float32, device="cuda:0")[::2]
    else:
        shape = (1, 2) if kind == "rank" else (1 if kind == "short" else 2,)
        dtype = wp.int32 if kind == "int32" else wp.float64 if kind == "float64" else wp.float32
        seed = wp.ones(shape, dtype=dtype, device="cpu" if kind == "cpu" else "cuda:0")
    with pytest.raises(ContractError):
        reverse.backward(output, seed)
    assert depth.grad.numpy().tolist() == [0.0, 0.0]
    with pytest.raises(ContractError, match="unconsumed"):
        reverse.backward(output, wp.ones(2, dtype=wp.float32, device="cuda:0"))
    reverse.close()


def test_valid_seed_preserves_caller_values_and_completes_backward() -> None:
    workspace = prepare_transmission(max_pixels=2)
    depth = wp.ones(2, dtype=wp.float32, device="cuda:0", requires_grad=True)
    output = wp.empty(2, dtype=wp.float32, device="cuda:0", requires_grad=True)
    seed = wp.array([2.0, -3.0], dtype=wp.float32, device="cuda:0")
    with FirstOrderPass([workspace]) as reverse:
        transmit(depth, out_log_T=output, workspace=workspace, tape=reverse.tape)
    reverse.backward(output, seed)
    assert depth.grad.numpy().tolist() == [-2.0, 3.0]
    assert seed.numpy().tolist() == [2.0, -3.0]
    reverse.close()


@pytest.mark.parametrize("mode", ["valid", "invalid-seed", "gradient-free-output"])
def test_nondefault_workspace_stream_owns_seed_backward_and_cleanup(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = wp.get_device("cuda:0")
    caller_stream = wp.get_stream(device)
    owning_stream = wp.Stream(device)
    assert owning_stream.cuda_stream != caller_stream.cuda_stream
    workspace = prepare_transmission(max_pixels=2, stream=owning_stream)
    with workspace.context.scope():
        depth = wp.ones(2, dtype=wp.float32, device=device, requires_grad=True)
        output = wp.empty(2, dtype=wp.float32, device=device, requires_grad=True, retain_grad=True)
        seed = wp.array([2.0, -3.0], dtype=wp.float32, device=device)
        invalid_seed = wp.ones(1, dtype=wp.float32, device=device)
        gradient_free = wp.empty(2, dtype=wp.float32, device=device)
    assert wp.get_stream(device).cuda_stream == caller_stream.cuda_stream
    with FirstOrderPass([workspace]) as reverse:
        transmit(
            depth, out_log_T=output, workspace=workspace, tape=reverse.tape, stream=owning_stream
        )
    assert wp.get_stream(device).cuda_stream == caller_stream.cuda_stream
    backward = reverse.tape.backward
    zero = reverse.tape.zero
    backward_streams: list[int] = []
    cleanup_streams: list[int] = []

    def observed_backward(**kwargs: Any) -> None:
        stream = int(wp.get_stream(device).cuda_stream)
        backward_streams.append(stream)
        assert stream == owning_stream.cuda_stream
        backward(**kwargs)

    def observed_zero() -> None:
        stream = int(wp.get_stream(device).cuda_stream)
        cleanup_streams.append(stream)
        assert stream == owning_stream.cuda_stream
        zero()

    monkeypatch.setattr(reverse.tape, "backward", observed_backward)
    monkeypatch.setattr(reverse.tape, "zero", observed_zero)
    if mode == "valid":
        reverse.backward(output, seed)
        assert wp.get_stream(device).cuda_stream == caller_stream.cuda_stream
        wp.synchronize_stream(owning_stream)
        assert depth.grad.numpy().tolist() == [-2.0, 3.0]
        assert output.grad.numpy().tolist() == [2.0, -3.0]
    else:
        with workspace.context.scope():
            depth.grad.fill_(7.0)
            output.grad.fill_(11.0)
        with pytest.raises(ContractError, match=r"shape|preallocated gradient"):
            reverse.backward(
                gradient_free if mode == "gradient-free-output" else output,
                invalid_seed if mode == "invalid-seed" else seed,
            )
    reverse.close()
    assert wp.get_stream(device).cuda_stream == caller_stream.cuda_stream
    wp.synchronize_stream(owning_stream)
    assert depth.grad.numpy().tolist() == [0.0, 0.0]
    assert output.grad.numpy().tolist() == [0.0, 0.0]
    assert seed.numpy().tolist() == [2.0, -3.0]
    assert reverse.tape.launches == []
    assert backward_streams == ([owning_stream.cuda_stream] if mode == "valid" else [])
    assert cleanup_streams
    assert all(stream == owning_stream.cuda_stream for stream in cleanup_streams)
