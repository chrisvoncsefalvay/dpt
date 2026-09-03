"""Actual flat/dark calibration ownership on a nondefault CUDA stream."""

import importlib
from typing import Any

import numpy as np
import pytest

from dpt.calibration import LogCalibration
from dpt.contracts import ContractError

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


def test_log_calibration_uses_owning_nondefault_stream_and_pinned_status(monkeypatch):
    device = wp.get_device("cuda:0")
    stream = wp.Stream(device=device)
    with wp.ScopedStream(stream):
        flat = wp.array([[20, 30]], dtype=wp.float64, device=device)
        dark = wp.array([[1, 2]], dtype=wp.float64, device=device)
        signal = wp.array([[10, 15]], dtype=wp.float64, device=device)
        seed = wp.array([[0.3, -0.4]], dtype=wp.float64, device=device)
    operator = LogCalibration(flat, dark, precision="float64", device="cuda:0", stream=stream)
    assert operator._status_readback._host.pinned
    waits = []
    original = wp.synchronize_stream

    def waited(owning):
        waits.append(owning)
        original(owning)

    monkeypatch.setattr(wp, "synchronize_stream", waited)
    result = operator.forward(signal)
    gradients = operator.vjp(seed)
    assert waits == [stream, stream]
    np.testing.assert_allclose(result.optical_depth.numpy(), -np.log([[9 / 19, 13 / 28]]))
    np.testing.assert_allclose(gradients.signal.numpy(), -np.asarray([[0.3 / 9, -0.4 / 13]]))
    foreign = wp.Stream(device=device)
    with pytest.raises(ContractError, match="owning stream"):
        operator.forward(signal, stream=foreign)
    operator.forward(signal, check=False)
    operator.check_status()


def test_log_calibration_rejects_cpu_frames_signal_and_cotangent_on_cuda():
    host = wp.array([2], dtype=wp.float64, device="cpu")
    with pytest.raises(ContractError, match="device"):
        LogCalibration(host, host, precision="float64", device="cuda:0")
    flat = wp.array([2], dtype=wp.float64, device="cuda:0")
    dark = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    operator = LogCalibration(flat, dark, precision="float64", device="cuda:0")
    with pytest.raises(ContractError, match="device"):
        operator.forward(host)
    operator.forward(flat)
    with pytest.raises(ContractError, match="device"):
        operator.vjp(host)
