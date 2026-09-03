"""Opt-in atomic and fused readouts retain original-history laws and fatal guards."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportMissingParameterType=false
import importlib
from typing import Any, Literal

import numpy as np
import pytest

from dpt.contracts import NumericalError
from dpt.transport.estimators import (
    history_mean,
    prepare_estimators,
    prepare_fused_mean,
    trace_mean,
)
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.transport.model import (
    IncompleteHistoryError,
    MaterialGrid,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from dpt.transport.rng import HistoryBatch

wp: Any = importlib.import_module("warp")


def slab(
    device,
    count,
    *,
    estimator: Literal["analogue", "continuous-absorption"] = "analogue",
    compton=False,
    max_crossings=65536,
):
    def arr(values, dtype=None):
        return wp.array(values, dtype=wp.float64 if dtype is None else dtype, device=device)

    spec = TransportSpec(
        MaterialGrid((-2.0, -2.0, 0.0), (4.0, 4.0, 1.0), (2, 1, 1), 1),
        PlanarDetector((-20.0, -20.0), (5.0, 10.0), (8, 4), 3.0),
        80.0,
        "canonical fused tally fixture",
        estimator=estimator,
        scattering_law="free-electron-compton" if compton else "isotropic-elastic",
        coefficient_energies_kev=(0.0, 80.0) if compton else (),
        max_crossings=max_crossings,
    )
    bins = len(spec.energy_nodes)
    workspace = prepare_transport(
        spec,
        material_ids=arr([0, 0], wp.int32),
        absorption=arr([0.1] * bins),
        scattering=arr([0.3] * bins),
        max_histories=count,
        device=device,
    )
    inputs = [
        arr([[0.0, 0.0, -1.0]] * count, wp.vec3d),
        arr([[0.0, 0.0, 1.0]] * count, wp.vec3d),
        arr([1.0] * count),
        arr([1.0]),
    ]
    return workspace, inputs


@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
@pytest.mark.parametrize("compton", [False, True])
@pytest.mark.parametrize("strategy", ["grouped", "plain"])
def test_staged_plain_grouped_and_fused_means_agree_on_same_complete_paths(
    estimator, compton, strategy, warp_device
):
    count = 1025
    workspace, inputs = slab(warp_device, count, estimator=estimator, compton=compton)
    outputs = {
        f"out_{name}": wp.empty(count, dtype=dtype, device=warp_device)
        for name, dtype in (
            ("pixel", wp.int32),
            ("score", wp.float64),
            ("energy", wp.float64),
            ("events", wp.int32),
            ("status", wp.int32),
        )
    }
    batch = HistoryBatch(43824, 2**49, count)
    trace_histories(*inputs, batch=batch, workspace=workspace, **outputs)
    mean = wp.empty(32, dtype=wp.float64, device=warp_device)
    history_mean(
        outputs["out_pixel"],
        outputs["out_score"],
        outputs["out_status"],
        batch=batch,
        workspace=prepare_estimators(workspace, tally_strategy=strategy),
        out_mean=mean,
    )
    expected = (
        np.bincount(
            np.maximum(outputs["out_pixel"].numpy(), 0),
            weights=outputs["out_score"].numpy(),
            minlength=32,
        )
        / count
    )
    np.testing.assert_allclose(mean.numpy(), expected, rtol=2e-13, atol=1e-15)
    fused = prepare_fused_mean(
        workspace,
        score_bound=wp.full(32, 1.0, dtype=wp.float64, device=warp_device),
        tally_strategy=strategy,
    )
    assert fused.scratch_bytes == 32 * 8
    trace_mean(*inputs, batch=batch, workspace=fused, out_mean=mean)
    np.testing.assert_allclose(mean.numpy(), expected, rtol=2e-13, atol=1e-15)


def test_fused_tally_resource_guards_and_unsafe_bounds_remain_fatal(warp_device):
    workspace, inputs = slab(warp_device, 257, max_crossings=1)
    bound = wp.full(32, 1.0, dtype=wp.float64, device=warp_device)
    fused = prepare_fused_mean(workspace, score_bound=bound)
    out = wp.empty_like(bound)
    with pytest.raises(IncompleteHistoryError):
        trace_mean(*inputs, batch=HistoryBatch(9424, 0, 257), workspace=fused, out_mean=out)
    with pytest.raises(IncompleteHistoryError):
        fused.check_status()
    workspace, inputs = slab(warp_device, 257)
    fused = prepare_fused_mean(
        workspace, score_bound=wp.full(32, 0.01, dtype=wp.float64, device=warp_device)
    )
    with pytest.raises(TransportError, match="invalid device"):
        trace_mean(*inputs, batch=HistoryBatch(9424, 0, 257), workspace=fused, out_mean=out)


def test_fused_bound_cannot_silently_underflow_nonzero_scores(warp_device):
    workspace, inputs = slab(warp_device, 257)
    inputs[2] = wp.full(257, 1e-300, dtype=wp.float64, device=warp_device)
    fused = prepare_fused_mean(
        workspace, score_bound=wp.full(32, 1e300, dtype=wp.float64, device=warp_device)
    )
    with pytest.raises(NumericalError):
        trace_mean(
            *inputs,
            batch=HistoryBatch(9424, 0, 257),
            workspace=fused,
            out_mean=wp.empty(32, dtype=wp.float64, device=warp_device),
        )
