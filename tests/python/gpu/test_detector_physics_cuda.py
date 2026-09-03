"""Explicit stream ownership and bounded graph replay for detector physics."""

from __future__ import annotations

import importlib
from typing import Any, Literal

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.detector_physics import (
    DetectorResponse,
    FirstInteractionModel,
    pixel_solid_angle,
    prepare_pixel_solid_angle,
    prepare_scintillator_response,
    scintillator_response,
    scintillator_response_vjp,
)
from dpt.geometry import DetectorGeometry
from dpt.materials import MaterialTable
from helpers.detector_physics_oracles import decimal_derivatives

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_detector_physics_graph_replays_current_slab_parameters(
    precision: Literal["float32", "float64"],
) -> None:
    stream = wp.Stream("cuda:0")
    geometry = DetectorGeometry(
        (0.0, 0.0, -10.0),
        (-1.0, -0.5, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.5),
        (2, 5),
    )
    table = MaterialTable("analytic total-interaction fixture", (20.0, 80.0), (0.5, 0.125), "mm^-1")
    model = FirstInteractionModel("ideal-full-energy")
    acceptance = prepare_pixel_solid_angle(geometry, stream=stream)
    workspace = prepare_scintillator_response(
        table,
        table.energies_kev,
        model=model,
        edge_side="above",
        max_pixels=10,
        precision=precision,
        stream=stream,
    )
    ctx = workspace.context
    with ctx.scope():
        solid, cosine = (wp.empty((1, 2, 5), dtype=wp.float64, device=ctx.device) for _ in range(2))
        thickness = wp.full(10, 0.1, dtype=wp.float64, device=ctx.device)
        response = wp.empty((2, 10), dtype=workspace.dtype, device=ctx.device)
        seed = wp.ones((2, 10), dtype=workspace.dtype, device=ctx.device)
        gt, gc = (wp.empty(10, dtype=wp.float64, device=ctx.device) for _ in range(2))
        flat_cosine = cosine.flatten()
        pointers = tuple(a.ptr for a in (solid, cosine, thickness, response, seed, gt, gc))

        def launch() -> None:
            pixel_solid_angle(
                out_solid_angle=solid,
                out_incidence_cosine=cosine,
                workspace=acceptance,
                stream=stream,
                validate=False,
            )
            scintillator_response(
                thickness,
                flat_cosine,
                out_response=response,
                workspace=workspace,
                stream=stream,
                validate=False,
            )
            scintillator_response_vjp(
                thickness,
                flat_cosine,
                cotangent=seed,
                out_thickness=gt,
                out_incidence_cosine=gc,
                workspace=workspace,
                stream=stream,
                validate=False,
            )

        launch()
        acceptance.check_status()
        workspace.check_status()
        with wp.ScopedCapture(device=ctx.device, stream=stream) as capture:
            launch()
        for t in (0.0, 0.75, 2.0):
            thickness.fill_(t)
            wp.capture_launch(capture.graph, stream=stream)
            acceptance.check_status()
            workspace.check_status()
            cosines = flat_cosine.numpy()
            for p, c in enumerate(cosines):
                recipe = DetectorResponse(table, table.energies_kev, t, model, "above", float(c))
                np.testing.assert_allclose(
                    response.numpy()[:, p],
                    recipe.values,
                    rtol=8e-8 if precision == "float32" else 1e-14,
                    atol=0,
                )
                expected = [
                    decimal_derivatives(mu, t, float(c), e, 1.0)
                    for mu, e in zip(table.coefficients, table.energies_kev, strict=True)
                ]
                assert gt.numpy()[p] == pytest.approx(sum(v[0] for v in expected), rel=2e-13)
                assert gc.numpy()[p] == pytest.approx(
                    sum(v[1] for v in expected), rel=2e-13, abs=1e-15
                )
        assert pointers == tuple(a.ptr for a in (solid, cosine, thickness, response, seed, gt, gc))


def test_detector_physics_stream_ownership_and_producer_handoff() -> None:
    producer, owner, unrelated = (wp.Stream("cuda:0") for _ in range(3))
    table = MaterialTable("analytic", (20.0,), (0.5,), "mm^-1")
    model = FirstInteractionModel("interaction-counting")
    workspace = prepare_scintillator_response(
        table, table.energies_kev, model=model, edge_side="above", max_pixels=1, stream=owner
    )
    with wp.ScopedStream(producer):
        thickness = wp.full(1, 0.7, dtype=wp.float64, device="cuda:0")
        cosine = wp.full(1, 0.5, dtype=wp.float64, device="cuda:0")
        produced = wp.record_event()
    response = wp.empty((1, 1), dtype=wp.float32, device="cuda:0")
    with pytest.raises(ContractError, match="owning stream"):
        scintillator_response(
            thickness, cosine, out_response=response, workspace=workspace, stream=unrelated
        )
    workspace.context.wait_for(produced)
    scintillator_response(
        thickness, cosine, out_response=response, workspace=workspace, stream=owner
    )
    assert response.numpy()[0, 0] == pytest.approx(-np.expm1(-0.7), rel=1e-7)
