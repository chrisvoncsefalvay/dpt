"""Explicit expected-signal cotangents with analytic signed contractions."""

# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportMissingParameterType=false
import importlib
import math
from typing import Any

import numpy as np
import pytest

from dpt.transport.derivatives import TransportParameter, prepare_transport_vjp, transport_vjp
from dpt.transport.forward import prepare_transport
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.rng import HistoryBatch

wp: Any = importlib.import_module("warp")


def slab(device, count, estimator):
    def arr(v, dtype=None):
        return wp.array(v, dtype=wp.float64 if dtype is None else dtype, device=device)

    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (2, 1, 1), 2),
        PlanarDetector((-1.0, -1.0), (1.0, 1.0), (2, 2), 3.0),
        80.0,
        "analytic cotangent contraction",
        estimator=estimator,
    )
    workspace = prepare_transport(
        spec,
        material_ids=arr([0, 1], wp.int32),
        absorption=arr([0.2, 0.5]),
        scattering=arr([0.0, 0.0]),
        max_histories=count,
        device=device,
    )
    inputs = [
        arr([[0.0, 0.0, -1.0]] * count, wp.vec3d),
        arr([[0.0, 0.0, 1.0]] * count, wp.vec3d),
        arr([1.0] * count),
        arr([1.0, 1.0]),
    ]
    return workspace, inputs


@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
@pytest.mark.parametrize("seed", [-3.0, 4.0])
def test_expected_signal_vjp_matches_independent_analytic_signed_contraction(
    estimator, seed, warp_device
):
    count = 65536
    transport, inputs = slab(warp_device, count, estimator)
    workspace = prepare_transport_vjp(
        transport,
        parameters=(
            TransportParameter("log-material-density", 0),
            TransportParameter("log-material-density", 1),
            TransportParameter("source-amplitude"),
        ),
    )
    cotangent = wp.array([2.0, -5.0, 7.0, seed], dtype=wp.float64, device=warp_device)
    gradient = wp.empty(3, dtype=wp.float64, device=warp_device)
    pointers = {name: value.ptr for name, value in workspace._arrays.items()}
    transport_vjp(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        cotangent,
        batch=HistoryBatch(73429, 0, count),
        workspace=workspace,
        out_gradient=gradient,
        source_amplitude=0.7,
    )
    transmission = math.exp(-0.7)
    expected = np.array(
        [-0.2 * 0.7 * transmission * seed, -0.5 * 0.7 * transmission * seed, transmission * seed]
    )
    standard_error = math.sqrt(transmission * (1 - transmission) / count)
    allowance = 7 * standard_error * np.abs([0.2 * 0.7 * seed, 0.5 * 0.7 * seed, seed])
    assert (np.abs(gradient.numpy() - expected) <= allowance + 1e-13).all()
    assert pointers == {name: value.ptr for name, value in workspace._arrays.items()}


def test_zero_amplitude_vjp_and_independent_cotangent_requirement(warp_device):
    transport, inputs = slab(warp_device, 17, "continuous-absorption")
    workspace = prepare_transport_vjp(
        transport, parameters=(TransportParameter("source-amplitude"),)
    )
    cotangent = wp.array([0.0, 0.0, 0.0, -2.0], dtype=wp.float64, device=warp_device)
    gradient = wp.empty(1, dtype=wp.float64, device=warp_device)
    batch = HistoryBatch(552, 0, 17)
    transport_vjp(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        cotangent,
        batch=batch,
        workspace=workspace,
        out_gradient=gradient,
        source_amplitude=0.0,
    )
    assert gradient.numpy()[0] == pytest.approx(-2 * math.exp(-0.7), rel=1e-14)
    with pytest.raises(TransportError, match="disjoint"):
        transport_vjp(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            cotangent,
            batch=batch,
            cotangent_batch=batch,
            workspace=workspace,
            out_gradient=gradient,
        )
    with pytest.raises(TransportError, match="preparation budget"):
        prepare_transport_vjp(
            transport, parameters=(TransportParameter("source-amplitude"),), max_bytes=1
        )
