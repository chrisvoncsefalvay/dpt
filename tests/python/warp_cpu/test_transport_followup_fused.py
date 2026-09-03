"""Bounded vector replay checked against scalar replays and analytic expectations."""

# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportMissingParameterType=false, reportUntypedFunctionDecorator=false
import importlib
import math
from typing import Any, Literal

import numpy as np
import pytest

from dpt.transport.derivatives import TransportParameter, derivative_columns, derivative_histories
from dpt.transport.forward import prepare_transport
from dpt.transport.inverse import prepare_transport_inverse
from dpt.transport.model import (
    IncompleteHistoryError,
    MaterialGrid,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam

wp: Any = importlib.import_module("warp")


def model(
    device,
    *,
    estimator: Literal["analogue", "continuous-absorption"] = "analogue",
    compton=False,
    count=1024,
    scatter=0.25,
    **kwargs,
):
    spec = TransportSpec(
        MaterialGrid((-2.0, -2.0, 0.0), (4.0, 4.0, 1.0), (2, 1, 1), 2),
        PlanarDetector((-20.0, -20.0), (20.0, 20.0), (2, 2), 3.0),
        80.0,
        "analytic layered vector-replay fixture",
        estimator=estimator,
        scattering_law="free-electron-compton" if compton else "isotropic-elastic",
        coefficient_energies_kev=(0.0, 80.0) if compton else (),
        **kwargs,
    )

    def arr(values, dtype=None):
        return wp.array(values, dtype=wp.float64 if dtype is None else dtype, device=device)

    bins = len(spec.energy_nodes)
    workspace = prepare_transport(
        spec,
        material_ids=arr([0, 1], wp.int32),
        absorption=arr([0.2] * bins + [0.5] * bins),
        scattering=arr([scatter] * (2 * bins)),
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
@pytest.mark.parametrize("compton", [False, True])
@pytest.mark.parametrize("columns", [1, 2, 3, 4])
def test_fused_columns_match_scalar_counter_replay(estimator, compton, columns, warp_device):
    count = 1024
    workspace, inputs = model(warp_device, estimator=estimator, compton=compton, count=count)
    selected = (
        TransportParameter("log-material-density", 1),
        TransportParameter("source-amplitude"),
        TransportParameter("log-material-density", 0),
        TransportParameter("log-source-amplitude"),
    )[:columns]
    batch = HistoryBatch(7148, 2**47, count)
    pixel = wp.empty(count, dtype=wp.int32, device=warp_device)
    status = wp.empty_like(pixel)
    fused = wp.empty(count * columns, dtype=wp.float64, device=warp_device)
    derivative_columns(
        *inputs,
        parameters=selected,
        batch=batch,
        workspace=workspace,
        out_pixel=pixel,
        out_derivatives=fused,
        out_status=status,
        source_amplitude=0.7,
    )
    expected_pixels, expected_status = pixel.numpy().copy(), status.numpy().copy()
    values = fused.numpy().reshape(columns, count)
    for column, parameter in enumerate(selected):
        scalar = wp.empty(count, dtype=wp.float64, device=warp_device)
        derivative_histories(
            *inputs,
            parameter=parameter,
            batch=batch,
            workspace=workspace,
            out_pixel=pixel,
            out_derivative=scalar,
            out_status=status,
            source_amplitude=0.7,
        )
        np.testing.assert_array_equal(pixel.numpy(), expected_pixels)
        np.testing.assert_array_equal(status.numpy(), expected_status)
        np.testing.assert_array_equal(scalar.numpy(), values[column])


@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
def test_fused_layer_gradients_have_independent_analytic_expectation(estimator, warp_device):
    count = 65536
    workspace, inputs = model(warp_device, estimator=estimator, count=count, scatter=0.0)
    selected = tuple(TransportParameter("log-material-density", i) for i in range(2))
    derivatives = wp.empty(2 * count, dtype=wp.float64, device=warp_device)
    derivative_columns(
        *inputs,
        parameters=selected,
        batch=HistoryBatch(4251, 0, count),
        workspace=workspace,
        out_pixel=wp.empty(count, dtype=wp.int32, device=warp_device),
        out_derivatives=derivatives,
        out_status=wp.empty(count, dtype=wp.int32, device=warp_device),
    )
    transmission = math.exp(-0.7)
    # Known Bernoulli variance gives the uncertainty independently of measured output.
    standard_error = math.sqrt(transmission * (1 - transmission) / count)
    for depth, actual in zip(
        (0.2, 0.5), derivatives.numpy().reshape(2, count).mean(axis=1), strict=True
    ):
        assert abs(actual + depth * transmission) <= 7 * depth * standard_error + 1e-13


@pytest.mark.parametrize("dimension", [1, 2, 3, 5, 8])
def test_opt_in_oracle_fuses_walks_and_reuses_storage(dimension, warp_device):
    count = 17
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (dimension, 1, 1), dimension),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), dimension + 1.0),
        80.0,
        "analytic deterministic fused oracle",
        estimator="continuous-absorption",
    )

    def arr(v, dtype=None):
        return wp.array(v, dtype=wp.float64 if dtype is None else dtype, device=warp_device)

    workspace = prepare_transport(
        spec,
        material_ids=arr(list(range(dimension)), wp.int32),
        absorption=arr([0.1] * dimension),
        scattering=arr([0.0] * dimension),
        max_histories=count,
        device=warp_device,
    )
    oracle = prepare_transport_inverse(
        workspace,
        source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
        parameters=tuple(TransportParameter("log-material-density", i) for i in range(dimension)),
        observation=arr([0.2]),
        pixel_weights=arr([2.0]),
        base_density=(1.0,) * dimension,
        fused_derivatives=True,
        local_model=True,
    )
    pointers = {k: v.ptr for k, v in oracle._arrays.items()}
    first, second = HistoryBatch(132, 0, count), HistoryBatch(132, count, count)
    gradient, metric = oracle.model_replicate((0.0,) * dimension, first, second)
    mean = math.exp(-0.1 * dimension)
    assert gradient == pytest.approx((2 * (mean - 0.2) * -0.1 * mean,) * dimension, abs=2e-14)
    for row in metric:
        assert row == pytest.approx((2 * (0.1 * mean) ** 2,) * dimension, abs=2e-14)
    assert oracle.histories_traced == count * (1 + math.ceil(dimension / 4))
    assert pointers == {k: v.ptr for k, v in oracle._arrays.items()}


def test_fused_replay_keeps_budget_failure_and_status_readback_storage(warp_device):
    workspace, inputs = model(warp_device, count=32, scatter=1e6, max_events=1)
    pointer = workspace._status_readback._host.ptr
    assert workspace.host_bytes == 4
    with pytest.raises(IncompleteHistoryError):
        derivative_columns(
            *inputs,
            parameters=(TransportParameter("log-material-density", 0),),
            batch=HistoryBatch(91, 0, 32),
            workspace=workspace,
            out_pixel=wp.empty(32, dtype=wp.int32, device=warp_device),
            out_derivatives=wp.empty(32, dtype=wp.float64, device=warp_device),
            out_status=wp.empty(32, dtype=wp.int32, device=warp_device),
        )
    with pytest.raises(IncompleteHistoryError):
        workspace.check_status()
    with pytest.raises(IncompleteHistoryError):
        workspace.consume_status()
    workspace.check_status()
    assert pointer == workspace._status_readback._host.ptr


def test_fused_budget_is_explicit(warp_device):
    workspace, _ = model(warp_device, count=32)

    def arr(values):
        return wp.array(values, dtype=wp.float64, device=warp_device)

    with pytest.raises(TransportError, match="preparation budget"):
        prepare_transport_inverse(
            workspace,
            source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
            parameters=(TransportParameter("log-material-density", 0),),
            observation=arr([0.0] * 4),
            pixel_weights=arr([1.0] * 4),
            base_density=(1.0, 1.0),
            fused_derivatives=True,
            fused_derivative_max_bytes=1,
        )


def test_symmetric_cross_fit_matches_both_independent_factors_and_reduces_variance(warp_device):
    count = 256
    workspace, _ = model(warp_device, count=count, scatter=0.0)

    def arr(values):
        return wp.array(values, dtype=wp.float64, device=warp_device)

    transmission = math.exp(-0.7)
    common: dict[str, Any] = dict(
        source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
        parameters=(TransportParameter("log-material-density", 0),),
        observation=arr([0.0, 0.0, 0.0, transmission]),
        pixel_weights=arr([0.0, 0.0, 0.0, 1.0]),
        base_density=(1.0, 1.0),
        fused_derivatives=True,
    )
    one_sided = prepare_transport_inverse(workspace, **common)
    symmetric = prepare_transport_inverse(workspace, **common, symmetric_gradient=True)
    values, originals = [], []
    for replicate in range(256):
        first = HistoryBatch(4421, 2 * replicate * count, count)
        second = HistoryBatch(4421, (2 * replicate + 1) * count, count)
        forward = one_sided.gradient_replicate((0.0,), first, second)[0]
        reverse = one_sided.gradient_replicate((0.0,), second, first)[0]
        value = symmetric.gradient_replicate((0.0,), first, second)[0]
        assert value == forward / 2 + reverse / 2
        values.append(value)
        originals.append(forward)
    variance_of_mean = transmission * (1 - transmission) / count
    expected_variance = 0.2**2 * (0.5 * transmission**2 * variance_of_mean + variance_of_mean**2)
    # Independent Bernoulli moment algebra supplies both the zero mean and variance.
    assert abs(np.mean(values)) < 7 * math.sqrt(expected_variance / len(values))
    assert np.var(values, ddof=1) == pytest.approx(expected_variance, rel=0.25)
    assert np.var(values, ddof=1) < 0.7 * np.var(originals, ddof=1)
    assert not symmetric.supports_local_model
