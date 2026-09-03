"""Bounded sampled transfer graph replay and stream ownership on real CUDA."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.detector_transfer import DetectorTransferSpec, detector_transfer, prepare_detector_transfer

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


def spec() -> DetectorTransferSpec:
    return DetectorTransferSpec(
        (20.0,), (0.5,), (0.5,), "counts", (1, 3), (0.25, 0.5, 0.25), (0.5, 0.8), 1.0
    )


def test_transfer_graph_replays_current_frequencies_and_diagnostics() -> None:
    model = spec()
    workspace = prepare_detector_transfer(model, max_frequencies=3)
    ctx = workspace.context
    frequencies = wp.zeros((3, 2), dtype=wp.float64, device=ctx.device)
    mtf, nps, dqe = (wp.empty(3, dtype=wp.float64, device=ctx.device) for _ in range(3))
    arrays = (frequencies, mtf, nps, dqe)
    pointers = tuple(a.ptr for a in arrays)

    def launch() -> None:
        detector_transfer(
            frequencies, out_mtf=mtf, out_nps=nps, out_dqe=dqe, workspace=workspace, validate=False
        )

    with ctx.scope():
        launch()
        workspace.check_status()
        with wp.ScopedCapture(device=ctx.device, stream=ctx.stream) as capture:
            launch()
        for x in (0.0, 0.25, 0.75):
            frequencies.assign(np.array([[x, 0.0], [0.0, 0.2], [-x, 0.4]]))
            wp.capture_launch(capture.graph, stream=ctx.stream)
            workspace.check_status()
            power = (0.5 + 0.5 * np.cos(2 * np.pi * frequencies.numpy()[:, 0] * 0.5)) ** 2
            np.testing.assert_allclose(mtf.numpy(), np.sqrt(power), rtol=1e-14)
            np.testing.assert_allclose(nps.numpy(), 0.4 * (10 * power + 1), rtol=1e-14)
            np.testing.assert_allclose(dqe.numpy(), 5 * power / (10 * power + 1), rtol=1e-14)
        frequencies.assign(np.array([[2.0, 0.0], [0.0, 0.2], [0.0, 0.4]]))
        wp.capture_launch(capture.graph, stream=ctx.stream)
        with pytest.raises(ContractError, match="invalid fixed"):
            workspace.check_status()
    assert pointers == tuple(a.ptr for a in arrays)


def test_transfer_requires_owner_stream_and_orders_producer_data() -> None:
    producer, owner, other = (wp.Stream("cuda:0") for _ in range(3))
    workspace = prepare_detector_transfer(spec(), max_frequencies=1, stream=owner)
    with wp.ScopedStream(producer):
        frequencies = wp.array([[0.5, 0.0]], dtype=wp.float64, device="cuda:0")
        ready = wp.record_event()
    outputs = tuple(wp.empty(1, dtype=wp.float64, device="cuda:0") for _ in range(3))
    with pytest.raises(ContractError, match="owning stream"):
        detector_transfer(
            frequencies,
            out_mtf=outputs[0],
            out_nps=outputs[1],
            out_dqe=outputs[2],
            workspace=workspace,
            stream=other,
        )
    workspace.context.wait_for(ready)
    detector_transfer(
        frequencies,
        out_mtf=outputs[0],
        out_nps=outputs[1],
        out_dqe=outputs[2],
        workspace=workspace,
        stream=owner,
    )
    assert outputs[0].numpy()[0] == pytest.approx(0.5, rel=1e-14)
