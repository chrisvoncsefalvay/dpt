"""Independent Decimal oracles for native material/source cotangent range."""

# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportMissingParameterType=false
import importlib
import math
from decimal import Decimal, localcontext
from typing import Any, Literal

import numpy as np
import pytest

from dpt.contracts import NumericalError
from dpt.transport.derivatives import (
    TransportParameter,
    derivative_columns,
    prepare_transport_vjp,
    transport_vjp,
)
from dpt.transport.forward import prepare_transport
from dpt.transport.model import (
    IncompleteHistoryError,
    MaterialGrid,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from dpt.transport.rng import HistoryBatch

wp: Any = importlib.import_module("warp")
MIN_SUBNORMAL = math.ulp(0.0)


def slab(
    device,
    weights,
    cotangents,
    *,
    tau=0.0,
    x=None,
    voxel_density=None,
    cells=1,
    energy=80.0,
    scoring: Literal["photon-count", "energy-kev"] = "photon-count",
    capacity=None,
):
    def arr(values, dtype=None):
        return wp.array(values, dtype=wp.float64 if dtype is None else dtype, device=device)

    count = len(weights)
    if x is None:
        x = [-1.5] * count
    spec = TransportSpec(
        MaterialGrid((-2.0, -1.0, 0.0), (4.0, 2.0, 1.0 / cells), (cells, 1, 1), 2),
        PlanarDetector((-2.0, -1.0), (1.0, 2.0), (4, 1), 2.0),
        energy,
        "independent pure-absorption cotangent range oracle",
        estimator="continuous-absorption",
        scoring=scoring,
        max_crossings=1 if cells > 1 else 1024,
    )
    transport = prepare_transport(
        spec,
        material_ids=arr([0] * cells, wp.int32),
        absorption=arr([tau, 0.0]),
        scattering=arr([0.0, 0.0]),
        max_histories=count if capacity is None else capacity,
        voxel_density=None if voxel_density is None else arr(voxel_density),
        device=device,
    )
    inputs = (
        arr([[value, 0.0, -1.0] for value in x], wp.vec3d),
        arr([[0.0, 0.0, 1.0]] * count, wp.vec3d),
        arr(weights),
        arr([1.0, 1.0]),
        arr(cotangents),
    )
    return transport, inputs


def decimal_gradient(weights, cotangents, x, tau, amplitude, *, vacuum=False, score_energy=1.0):
    # Sum before multiplication/averaging, so opposite extended-range terms
    # cancel exactly in this independent pure-absorption analytic model.
    with localcontext() as ctx:
        ctx.prec = 1200
        total = Decimal(0)
        for weight, coordinate in zip(weights, x, strict=True):
            if -2.0 <= coordinate < 2.0:
                index = math.floor(coordinate + 2.0)
                total += Decimal.from_float(weight) * Decimal.from_float(cotangents[index])
        depth = Decimal.from_float(0.0 if vacuum else tau)
        signal = total * Decimal.from_float(score_energy) * (-depth).exp() / len(weights)
        amplitude_d = Decimal.from_float(amplitude)
        return np.array(
            [
                float(-depth * amplitude_d * signal),
                0.0,
                float(signal),
                float(amplitude_d * signal),
            ]
        )


@pytest.mark.parametrize(
    ("weights", "cotangents", "x", "tau", "amplitude"),
    [
        ([1.0], [1e300, 0.0, 0.0, 0.0], [-1.5], 1000.0, 1.0),
        ([1.0], [-1e300, 0.0, 0.0, 0.0], [-1.5], 1000.0, 1.0),
        ([1.0], [1e300, 0.0, 0.0, 0.0], [-1.5], 1000.0, 0.0),
        ([1e300], [1e300, 0.0, 0.0, 0.0], [-1.5], 2000.0, 1e300),
        ([1e308, 0.0], [2.0, 0.0, 0.0, 0.0], [-1.5, -1.5], 0.0, 1.0),
        ([1e308, 1e308], [2.0, -2.0, 0.0, 0.0], [-1.5, -0.5], 0.0, 1.0),
        (
            [1e308, 1e308, 1e-310],
            [1e308, -1e308, 1.0, 0.0],
            [-1.5, -0.5, 0.5],
            0.0,
            1.0,
        ),
        (
            [1e308, 1e308, 1.0],
            [1e308, -1e308, 1e300, 0.0],
            [-1.5, -0.5, 0.5],
            1000.0,
            1.0,
        ),
        ([MIN_SUBNORMAL] * 3, [1.0, 0.0, 0.0, 0.0], [-1.5] * 3, 0.0, 1.0),
        ([3 * MIN_SUBNORMAL, 0.0], [1.0, 0.0, 0.0, 0.0], [-1.5] * 2, 0.0, 1.0),
        ([MIN_SUBNORMAL, 0.0], [1.0, 0.0, 0.0, 0.0], [-1.5] * 2, 0.0, 1.0),
        ([3 * MIN_SUBNORMAL, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [-1.5] * 3, 0.0, 1.0),
        ([1.0, 2.0, 3.0], [-2.0, 5.0, -0.25, 0.0], [-1.5, -0.5, 0.5], 0.7, 0.7),
        ([1.0, 2.0, 1e308], [2.0, 0.0, 0.0, 0.0], [-1.5, -1.5, 10.0], 0.7, 1.0),
        ([0.0], [1e308, 0.0, 0.0, 0.0], [-1.5], 1000.0, 1e308),
        ([1e308], [1e308, 0.0, 0.0, 0.0], [-1.5], 8192.0, 1e308),
        ([1e308], [1e308, 0.0, 0.0, 0.0], [-1.5], 8193.0, 1e308),
    ],
)
def test_original_factors_cancellation_and_mean_match_decimal(
    weights, cotangents, x, tau, amplitude, warp_device
):
    transport, inputs = slab(warp_device, weights, cotangents, tau=tau, x=x)
    parameters = (
        TransportParameter("log-material-density", 0),
        TransportParameter("log-material-density", 1),
        TransportParameter("source-amplitude"),
    )
    if amplitude > 0:
        parameters += (TransportParameter("log-source-amplitude"),)
    workspace = prepare_transport_vjp(transport, parameters=parameters)
    gradient = wp.empty(len(parameters), dtype=wp.float64, device=warp_device)
    pointers = {key: array.ptr for key, array in workspace._arrays.items()}
    for _repeat in range(2):
        transport_vjp(
            *inputs,
            batch=HistoryBatch(49265, 0, len(weights)),
            workspace=workspace,
            out_gradient=gradient,
            source_amplitude=amplitude,
        )
        expected = decimal_gradient(weights, cotangents, x, tau, amplitude)[: len(parameters)]
        np.testing.assert_allclose(gradient.numpy(), expected, rtol=5e-14, atol=0.0)
    assert pointers == {key: array.ptr for key, array in workspace._arrays.items()}
    assert workspace.scratch_bytes == sum(array.size * 8 for array in workspace._arrays.values())


def test_contracted_extreme_does_not_change_uncontracted_derivative_columns(warp_device):
    transport, inputs = slab(warp_device, [1.0], [1e300, 0.0, 0.0, 0.0], tau=1000.0)
    parameters = (
        TransportParameter("log-material-density", 0),
        TransportParameter("source-amplitude"),
    )
    derivatives = wp.empty(2, dtype=wp.float64, device=warp_device)
    derivative_columns(
        *inputs[:4],
        parameters=parameters,
        batch=HistoryBatch(49265, 0, 1),
        workspace=transport,
        out_pixel=wp.empty(1, dtype=wp.int32, device=warp_device),
        out_derivatives=derivatives,
        out_status=wp.empty(1, dtype=wp.int32, device=warp_device),
    )
    np.testing.assert_array_equal(derivatives.numpy(), [0.0, 0.0])
    workspace = prepare_transport_vjp(transport, parameters=parameters)
    gradient = wp.empty(2, dtype=wp.float64, device=warp_device)
    transport_vjp(
        *inputs,
        batch=HistoryBatch(49265, 0, 1),
        workspace=workspace,
        out_gradient=gradient,
    )
    expected = decimal_gradient([1.0], [1e300, 0.0, 0.0, 0.0], [-1.5], 1000.0, 1.0)
    np.testing.assert_allclose(gradient.numpy(), expected[[0, 2]], rtol=5e-14, atol=0.0)


@pytest.mark.parametrize(
    ("energy", "weight", "cotangent", "amplitude", "tau"),
    [
        (1e308, 1e308, 1e308, 1e308, 3000.0),
        (MIN_SUBNORMAL, 1e308, 1e308, 1e-308, 0.0),
        (1e-300, 1e-300, 1e300, 1e300, 0.0),
    ],
)
def test_energy_scoring_retains_all_original_factor_exponents(
    energy, weight, cotangent, amplitude, tau, warp_device
):
    transport, inputs = slab(
        warp_device,
        [weight],
        [cotangent, 0.0, 0.0, 0.0],
        tau=tau,
        energy=energy,
        scoring="energy-kev",
    )
    workspace = prepare_transport_vjp(
        transport,
        parameters=(
            TransportParameter("log-material-density", 0),
            TransportParameter("source-amplitude"),
            TransportParameter("log-source-amplitude"),
        ),
    )
    gradient = wp.empty(3, dtype=wp.float64, device=warp_device)
    transport_vjp(
        *inputs,
        batch=HistoryBatch(49265, 0, 1),
        workspace=workspace,
        out_gradient=gradient,
        source_amplitude=amplitude,
    )
    expected = decimal_gradient(
        [weight], [cotangent, 0.0, 0.0, 0.0], [-1.5], tau, amplitude, score_energy=energy
    )[[0, 2, 3]]
    np.testing.assert_allclose(gradient.numpy(), expected, rtol=5e-14, atol=0.0)


@pytest.mark.parametrize("case", range(16))
def test_extended_signed_cancellation_preserves_small_residual_across_exponents(case, warp_device):
    generator = np.random.default_rng(807325 + case)
    large_weight = math.ldexp(
        float(generator.uniform(0.5, 1.0)), int(generator.integers(600, 1024))
    )
    large_seed = math.ldexp(float(generator.uniform(0.5, 1.0)), int(generator.integers(600, 1024)))
    small_weight = math.ldexp(float(generator.uniform(0.5, 1.0)), int(generator.integers(-1000, 1)))
    tau = float(generator.choice([0.0, 0.7, 200.0, 1000.0]))
    amplitude = float(generator.choice([0.0, 0.7, 1.0]))
    weights = [large_weight, large_weight, small_weight, 0.0, 0.0]
    cotangents = [large_seed, -large_seed, 1.0, 0.0]
    x = [-1.5, -0.5, 0.5, 1.5, 1.5]
    transport, inputs = slab(warp_device, weights, cotangents, x=x, tau=tau)
    workspace = prepare_transport_vjp(
        transport,
        parameters=(
            TransportParameter("log-material-density", 0),
            TransportParameter("source-amplitude"),
        ),
    )
    gradient = wp.empty(2, dtype=wp.float64, device=warp_device)
    transport_vjp(
        *inputs,
        batch=HistoryBatch(17, 0, len(weights)),
        workspace=workspace,
        out_gradient=gradient,
        source_amplitude=amplitude,
    )
    expected = decimal_gradient(weights, cotangents, x, tau, amplitude)[[0, 2]]
    np.testing.assert_allclose(gradient.numpy(), expected, rtol=5e-14, atol=0.0)


def test_fixed_voxel_vacuum_and_zero_amplitude_keep_direct_source_derivative(warp_device):
    transport, inputs = slab(
        warp_device, [1e-200], [-1e200, 0.0, 0.0, 0.0], tau=1000.0, voxel_density=[0.0]
    )
    workspace = prepare_transport_vjp(
        transport,
        parameters=(
            TransportParameter("log-material-density", 0),
            TransportParameter("source-amplitude"),
        ),
    )
    gradient = wp.empty(2, dtype=wp.float64, device=warp_device)
    transport_vjp(
        *inputs,
        batch=HistoryBatch(22, 0, 1),
        workspace=workspace,
        out_gradient=gradient,
        source_amplitude=0.0,
    )
    np.testing.assert_allclose(gradient.numpy(), [0.0, -1.0], rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("cotangent", [math.nan, math.inf, -math.inf])
def test_unreachable_nonfinite_cotangent_is_sticky_until_explicit_clear(cotangent, warp_device):
    transport, inputs = slab(warp_device, [1.0], [1.0, cotangent, 0.0, 0.0])
    workspace = prepare_transport_vjp(
        transport, parameters=(TransportParameter("source-amplitude"),)
    )
    gradient = wp.empty(1, dtype=wp.float64, device=warp_device)
    transport_vjp(
        *inputs,
        batch=HistoryBatch(22, 0, 1),
        workspace=workspace,
        out_gradient=gradient,
        validate=False,
    )
    with pytest.raises(TransportError, match="invalid device transport inputs"):
        workspace.check_status()
    with transport.context.scope():
        inputs[4].fill_(1.0)
    transport_vjp(
        *inputs,
        batch=HistoryBatch(22, 0, 1),
        workspace=workspace,
        out_gradient=gradient,
        validate=False,
    )
    with pytest.raises(TransportError, match="invalid device transport inputs"):
        workspace.check_status()
    transport.clear_status()
    transport_vjp(
        *inputs,
        batch=HistoryBatch(22, 0, 1),
        workspace=workspace,
        out_gradient=gradient,
    )
    assert gradient.numpy()[0] == 1.0


def test_only_final_overflow_invalidates_contraction(warp_device):
    transport, inputs = slab(warp_device, [1e308], [1e308, 0.0, 0.0, 0.0])
    workspace = prepare_transport_vjp(
        transport, parameters=(TransportParameter("source-amplitude"),)
    )
    with pytest.raises(NumericalError, match="nonfinite transport arithmetic"):
        transport_vjp(
            *inputs,
            batch=HistoryBatch(22, 0, 1),
            workspace=workspace,
            out_gradient=wp.empty(1, dtype=wp.float64, device=warp_device),
        )


def test_incomplete_path_still_invalidates_entire_contraction(warp_device):
    transport, inputs = slab(warp_device, [1.0], [1.0, 0.0, 0.0, 0.0], tau=0.7, cells=2)
    workspace = prepare_transport_vjp(
        transport, parameters=(TransportParameter("source-amplitude"),)
    )
    with pytest.raises(IncompleteHistoryError, match="budget exhausted"):
        transport_vjp(
            *inputs,
            batch=HistoryBatch(22, 0, 1),
            workspace=workspace,
            out_gradient=wp.empty(1, dtype=wp.float64, device=warp_device),
        )


def test_log_amplitude_domain_and_preparation_budget_remain_explicit(warp_device):
    transport, inputs = slab(warp_device, [1.0], [1.0, 0.0, 0.0, 0.0])
    workspace = prepare_transport_vjp(
        transport, parameters=(TransportParameter("log-source-amplitude"),)
    )
    with pytest.raises(TransportError, match="strictly positive amplitude"):
        transport_vjp(
            *inputs,
            batch=HistoryBatch(22, 0, 1),
            workspace=workspace,
            out_gradient=wp.empty(1, dtype=wp.float64, device=warp_device),
            source_amplitude=0.0,
        )
    with pytest.raises(TransportError, match="preparation budget"):
        prepare_transport_vjp(
            transport, parameters=(TransportParameter("source-amplitude"),), max_bytes=1
        )


def test_preparation_budget_is_independent_of_maximum_history_capacity(warp_device):
    from dpt.transport.derivatives import TRANSPORT_VJP_BYTES_PER_PARAMETER

    transport, _inputs = slab(warp_device, [1.0], [1.0, 0.0, 0.0, 0.0], capacity=2**31 - 1)
    parameters = (
        TransportParameter("log-material-density", 0),
        TransportParameter("log-material-density", 1),
        TransportParameter("source-amplitude"),
        TransportParameter("log-source-amplitude"),
    )
    required = 4 * TRANSPORT_VJP_BYTES_PER_PARAMETER
    workspace = prepare_transport_vjp(transport, parameters=parameters, max_bytes=required)
    assert workspace.scratch_bytes == required == 22400
    with pytest.raises(TransportError, match="preparation budget"):
        prepare_transport_vjp(transport, parameters=parameters, max_bytes=required - 1)
