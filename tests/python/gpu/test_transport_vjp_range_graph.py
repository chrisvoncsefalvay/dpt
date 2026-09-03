"""Real CUDA graph replay of contracted material/source scores and sticky status."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
import importlib
import math
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pytest

from dpt.transport.derivatives import TransportParameter, prepare_transport_vjp, transport_vjp
from dpt.transport.forward import prepare_transport
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.rng import HistoryBatch

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


def test_native_material_source_vjp_capture_changes_cotangent_and_preserves_status() -> None:
    device = "cuda:0"
    count = 19
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
        80.0,
        "pure-absorption Decimal graph oracle",
        estimator="continuous-absorption",
    )
    transport = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=device),
        absorption=wp.array([1000.0], dtype=wp.float64, device=device),
        scattering=wp.array([0.0], dtype=wp.float64, device=device),
        max_histories=count,
        device=device,
    )
    context = transport.context
    workspace = prepare_transport_vjp(
        transport,
        parameters=(
            TransportParameter("log-material-density", 0),
            TransportParameter("source-amplitude"),
        ),
    )
    positions = wp.array([[0.0, 0.0, -1.0]] * count, dtype=wp.vec3d, device=device)
    directions = wp.array([[0.0, 0.0, 1.0]] * count, dtype=wp.vec3d, device=device)
    weights = wp.array([1.0] * count, dtype=wp.float64, device=device)
    density = wp.array([1.0], dtype=wp.float64, device=device)
    cotangent = wp.array([1e300], dtype=wp.float64, device=device)
    gradient = wp.empty(2, dtype=wp.float64, device=device)
    pointers = tuple(array.ptr for array in workspace._arrays.values())
    batch = HistoryBatch(82, 100, count)

    def evaluate(*, validate: bool) -> None:
        transport_vjp(
            positions,
            directions,
            weights,
            density,
            cotangent,
            batch=batch,
            workspace=workspace,
            out_gradient=gradient,
            validate=validate,
        )

    evaluate(validate=True)
    with context.scope(), wp.ScopedCapture(device=context.device, stream=context.stream) as capture:
        evaluate(validate=False)
    for factor in (1.0, -3.0, 0.0, 2.0):
        with context.scope():
            cotangent.fill_(factor * 1e300)
            wp.capture_launch(capture.graph, stream=context.stream)
        workspace.check_status()
        with localcontext() as decimal_context:
            decimal_context.prec = 100
            source = (-Decimal(1000)).exp() * Decimal.from_float(factor * 1e300)
            expected = [float(-Decimal(1000) * source), float(source)]
        np.testing.assert_allclose(gradient.numpy(), expected, rtol=5e-14, atol=0.0)
    assert pointers == tuple(array.ptr for array in workspace._arrays.values())
    for value in (math.nan, 1e300):
        with context.scope():
            cotangent.fill_(value)
            wp.capture_launch(capture.graph, stream=context.stream)
        with pytest.raises(TransportError, match="invalid device transport inputs"):
            workspace.check_status()
    transport.clear_status()
    with context.scope():
        wp.capture_launch(capture.graph, stream=context.stream)
    workspace.check_status()
