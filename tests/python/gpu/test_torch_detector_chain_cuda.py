"""Actual foreign-stream execution and storage lifetime for the Torch detector chain."""

import importlib
from typing import Any

import pytest

from dpt.contracts import ContractError
from dpt.detector import BlurSpec, CalibrationSpec, prepare_blur, prepare_detector
from dpt.objectives import ObjectiveSpec, prepare_objective
from dpt.torch.operators import affine_calibrate, blur, objective, transmit
from dpt.transmission import TransmissionSpec, prepare_transmission
from helpers.torch_detector_oracles import blur_matrix

torch: Any = importlib.import_module("torch")
wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


def test_detector_chain_orders_foreign_forward_backward_streams_without_global_wait(monkeypatch):
    producer = torch.cuda.Stream(device="cuda:0")
    caller = torch.cuda.Stream(device="cuda:0")
    owner = torch.cuda.Stream(device="cuda:0")
    backward_caller = torch.cuda.Stream(device="cuda:0")
    native_stream = wp.stream_from_torch(owner)
    transmission = prepare_transmission(
        TransmissionSpec(beam="per-pixel", active_beam=True),
        max_pixels=6,
        device="cuda:0",
        stream=native_stream,
    )
    detector = prepare_detector(max_pixels=6, device="cuda:0", stream=native_stream)
    spec = BlurSpec(2, 3, 1, 3, (0.1, 0.5, 0.2))
    spreading = prepare_blur(spec, device="cuda:0", stream=native_stream)
    loss_workspace = prepare_objective(
        ObjectiveSpec(), max_pixels=6, device="cuda:0", stream=native_stream
    )
    with torch.cuda.stream(producer):
        depth = torch.tensor(
            [[0.1, 0.3, 0.5], [0.2, 0.7, 1.1]], device="cuda:0", requires_grad=True
        )
        beam = torch.tensor(
            [[10, 20, 30], [40, 50, 60]],
            dtype=torch.float32,
            device="cuda:0",
            requires_grad=True,
        )
        gain = torch.tensor(1.2, device="cuda:0", requires_grad=True)
        exposure = torch.tensor(0.7, device="cuda:0")
        offset = torch.tensor(0.1, device="cuda:0", requires_grad=True)
        observed = torch.full((2, 3), 5.0, device="cuda:0")
    caller.wait_stream(producer)

    def forbidden(*args, **kwargs):
        raise AssertionError("detector Torch bridge must not synchronise the whole device")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(wp, "synchronize_device", forbidden)
    with torch.cuda.stream(caller):
        counts = transmit(depth, beam, output="counts", workspace=transmission)
        calibrated = affine_calibrate(
            counts,
            gain,
            exposure,
            offset,
            spec=CalibrationSpec(
                output_unit="detector counts", active_gain=True, active_offset=True
            ),
            workspace=detector,
        )
        spread = blur(calibrated, workspace=spreading)
        loss = objective(spread, observed, workspace=loss_workspace)
        counts_snapshot, spread_snapshot = counts.detach().clone(), spread.detach().clone()
        assert counts.data_ptr() != depth.data_ptr()
        del counts, calibrated, spread, transmission, detector, spreading, loss_workspace
        # Bound allocator churn while autograd retains only the required owners.
        temporary = [torch.empty((64, 64), device="cuda:0") for _ in range(8)]
        for value in temporary:
            value.fill_(-99)
        del temporary
    backward_caller.wait_stream(caller)
    with torch.cuda.stream(backward_caller):
        loss.backward()
    torch.cuda.current_stream("cuda:0").wait_stream(backward_caller)
    matrix = torch.tensor(blur_matrix(spec), dtype=torch.float64, device="cuda:0")
    image_seed = matrix.T @ (spread_snapshot.double() - observed.double()).reshape(-1)
    mean_seed = image_seed.reshape(2, 3) * gain.detach().double() * exposure.double()
    torch.testing.assert_close(
        depth.grad.double(), -mean_seed * counts_snapshot.double(), rtol=4e-6, atol=2e-6
    )
    torch.testing.assert_close(
        beam.grad.double(), mean_seed * torch.exp(-depth.detach().double()), rtol=4e-6, atol=2e-6
    )
    torch.testing.assert_close(
        gain.grad.double(),
        (image_seed * counts_snapshot.double().reshape(-1) * exposure.double()).sum(),
        rtol=4e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(offset.grad.double(), image_seed.sum(), rtol=4e-6, atol=2e-6)


def test_detector_wrappers_reject_host_tensor_and_mixed_devices_on_cuda():
    workspace = prepare_transmission(max_pixels=2, device="cuda:0")
    host = torch.ones(2, dtype=torch.float32, device="cpu")
    with pytest.raises(ContractError, match="device"):
        transmit(host, workspace=workspace)
    workspace = prepare_transmission(
        TransmissionSpec(beam="per-pixel"), max_pixels=2, device="cuda:0"
    )
    depth = torch.ones(2, dtype=torch.float32, device="cuda:0")
    with pytest.raises(ContractError, match="device"):
        transmit(depth, host, output="counts", workspace=workspace)


def test_fp64_loss_backward_uses_native_owning_stream_for_promoted_seed():
    owning = torch.cuda.Stream(device="cuda:0")
    caller = torch.cuda.Stream(device="cuda:0")
    workspace = prepare_objective(
        ObjectiveSpec(weighted=True),
        max_pixels=1,
        device="cuda:0",
        stream=wp.stream_from_torch(owning),
    )
    with torch.cuda.stream(caller):
        prediction = torch.tensor([1e-30], device="cuda:0", requires_grad=True)
        observation = torch.zeros(1, device="cuda:0")
        weight = torch.tensor([1e-30], device="cuda:0")
        seed = torch.tensor(1e30, dtype=torch.float64, device="cuda:0")
        objective(prediction, observation, weights=weight, workspace=workspace).backward(seed)
        expected = prediction.detach().double() * weight.double() * seed
        torch.testing.assert_close(prediction.grad.double(), expected, rtol=2e-6, atol=0)
    torch.cuda.current_stream("cuda:0").wait_stream(caller)
