"""Explicit variant preparation and reusable diagnostic acceptance boundaries."""

import importlib
import math
from typing import Any

import pytest

from dpt.contracts import NumericalError
from dpt.objectives import ObjectiveSpec, evaluate_objective, prepare_objective
from dpt.transmission import TransmissionError, TransmissionSpec, prepare_transmission, transmit

wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize("beam", ["none", "scalar", "device-scalar", "per-pixel"])
def test_warming_compiles_real_modules_without_evaluating_outputs(
    beam: Any, warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = prepare_transmission(
        TransmissionSpec(beam=beam, active_beam=beam in ("device-scalar", "per-pixel")),
        max_pixels=257,
        device=warp_device,
    )
    load_module = wp.load_module
    modules: list[tuple[Any, int]] = []

    def loaded(module: Any, *, device: Any, block_dim: int) -> None:
        modules.append((module, block_dim))
        load_module(module, device=device, block_dim=block_dim)

    def must_not_evaluate(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("warming evaluated a numerical kernel")

    with monkeypatch.context() as patch:
        patch.setattr(wp, "load_module", loaded)
        patch.setattr(wp, "launch", must_not_evaluate)
        names = workspace.warm(
            forward_masks=(5,) if beam == "none" else (7,),
            vjp_masks=(5,) if beam == "none" else (7,),
            inverse=True,
        )
    assert len(names) == len(modules) and len(names) >= 6
    assert workspace.host_staging_bytes == 4
    field = wp.array([0.0, 1.0, 2.0], dtype=wp.float32, device=warp_device)
    output = wp.empty(3, dtype=wp.float32, device=warp_device)
    beam_value = (
        None
        if beam == "none"
        else 2.0
        if beam == "scalar"
        else wp.full(1 if beam == "device-scalar" else 3, 2.0, dtype=wp.float32, device=warp_device)
    )
    transmit(field, beam_value, out_T=output, workspace=workspace)
    assert output.numpy().tolist() == pytest.approx([1.0, math.exp(-1), math.exp(-2)], rel=2e-7)


def test_warm_rejects_unreachable_variants_before_loading(warp_device: str) -> None:
    workspace = prepare_transmission(max_pixels=2, device=warp_device)
    for masks in ((0,), (16,), (True,), (2,)):
        with pytest.raises(TransmissionError):
            workspace.warm(forward_masks=masks)


def test_objective_explicit_consume_delivers_then_resets_and_reuses_staging(
    warp_device: str,
) -> None:
    workspace = prepare_objective(
        ObjectiveSpec(precision="float64"), max_pixels=3, device=warp_device
    )
    values = wp.array([math.inf, 2.0, 3.0], dtype=wp.float64, device=warp_device)
    observations = wp.zeros(3, dtype=wp.float32, device=warp_device)
    loss = wp.empty(1, dtype=wp.float64, device=warp_device)
    evaluate_objective(values, observations, out_loss=loss, workspace=workspace, validate=False)
    with pytest.raises(NumericalError):
        workspace.consume_status()
    workspace.check_status()
    values.assign([1.0, 2.0, 3.0])
    evaluate_objective(values, observations, out_loss=loss, workspace=workspace)
    assert float(loss.numpy()[0]) == 7.0
    assert workspace.host_staging_bytes == 4
