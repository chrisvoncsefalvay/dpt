"""Production transport checks against independent deterministic and sampling oracles."""

import importlib
import math
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pytest
from scipy import stats

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError, NumericalError
from dpt.kernels.numerics import (
    attenuated_product4,
    attenuated_product4_from_state,
    attenuation_state,
    product3,
    product4,
)
from dpt.stochastic_recovery import StochasticPolicy, recover_expected_signal
from dpt.transport.derivatives import TransportParameter, derivative_histories
from dpt.transport.experiments import AbsorptionExperiment
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.transport.kernels import coefficients, compton_scatter
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.recovery_experiments import absorption_cases, scattering_policy
from dpt.transport.rng import HistoryBatch
from helpers.transport_inputs import absorbing_oracle
from helpers.transport_scattering_reference import (
    independent_isotropic_cube,
)

# Warp's executable annotations and NumPy/SciPy stubs are outside host type checking.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUntypedFunctionDecorator=false


wp: Any = importlib.import_module("warp")
Parameters: Any = importlib.import_module("dpt.transport.kernels").Parameters


@wp.kernel
def inspect_products(
    factors: wp.array2d(dtype=wp.float64),
    tau: wp.float64,
    out: wp.array2d(dtype=wp.float64),
):
    i = wp.tid()
    a, b, c, d = factors[i, 0], factors[i, 1], factors[i, 2], factors[i, 3]
    out[i, 0] = product3(a, b, c)
    out[i, 1] = product4(a, b, c, d)
    out[i, 2] = attenuated_product4(tau, a, b, c, d)
    out[i, 3] = attenuated_product4_from_state(attenuation_state(tau), a, b, c, d)


@wp.kernel
def inspect_compton(energy: wp.float64, out: wp.array(dtype=wp.vec4d)):
    i = wp.tid()
    out[i] = compton_scatter(energy, wp.uint64(153571), wp.uint64(i), wp.uint32(0), 4096)


@wp.kernel
def inspect_coefficients(
    query: wp.array(dtype=wp.vec2d),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    params: Parameters,
    out: wp.array(dtype=wp.vec2d),
):
    i = wp.tid()
    out[i] = coefficients(int(query[i][0]), query[i][1], energies, absorption, scattering, params)


def products(factors, tau, device):
    source = wp.array(np.array(factors, dtype=np.float64), dtype=wp.float64, device=device)
    out = wp.empty((len(factors), 4), dtype=wp.float64, device=device)
    wp.launch(inspect_products, dim=len(factors), inputs=[source, tau, out], device=device)
    return out.numpy()


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_product_helpers_preserve_nonfinite_factors_in_every_position(bad, warp_device):
    factors = []
    for index in range(4):
        row = [2.0, 3.0, -5.0, 7.0]
        row[index] = bad
        factors.append(row)
    result = products(factors, 2.0, warp_device)
    assert np.all(~np.isfinite(result[:, 1:]))
    assert np.all(~np.isfinite(result[:3, 0]))
    # A zero cannot hide an invalid factor from the operator's numerical status.
    for index in range(4):
        factors[index][(index + 1) % 4] = 0.0
    result = products(factors, 2.0, warp_device)
    assert np.all(np.isnan(result[:, 1:]))


@pytest.mark.parametrize("tau", [math.nan, math.inf, -math.inf, -1.0])
def test_attenuated_product_rejects_invalid_optical_depth(tau, warp_device):
    result = products([[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]], tau, warp_device)
    assert np.all(np.isnan(result[:, 2:]))


@pytest.mark.parametrize("tau", [0.0, 745.0, 800.0, 3000.0, 8193.0])
def test_range_preserving_products_match_high_precision_original_factors(tau, warp_device):
    factors = [
        [1e300, 1e300, 1e-300, 1e-300],
        [1e-300, 1e-300, 1e300, 1e300],
        [1e300, 1e300, 1e300, 1e300],
        [1.0, 1.0, 1.0, 1.0],
        [-1e300, 1e300, 1e-300, 1.0],
        [math.ulp(0.0), 2.0**1023, 0.5, 1.0],
    ]
    result = products(factors, tau, warp_device)
    with localcontext() as context:
        context.prec = 1600
        attenuation = (-Decimal.from_float(tau)).exp()
        for row, actual in zip(factors, result, strict=True):
            for count, column in ((3, 0), (4, 1), (4, 2), (4, 3)):
                expected = Decimal(1)
                for factor in row[:count]:
                    expected *= Decimal.from_float(factor)
                if column >= 2:
                    expected *= attenuation
                expected_float = float(expected)
                if math.isinf(expected_float):
                    assert actual[column] == expected_float
                else:
                    assert abs(actual[column] - expected_float) <= 3 * math.ulp(expected_float)
    np.testing.assert_array_equal(result[:, 2], result[:, 3])


def test_all_declared_recovery_modes_have_valid_host_policies():
    for mode, count in (("original", 1), ("ablation", 4), ("family", 36)):
        cases = absorption_cases(mode)
        assert len(cases) == count
        for _, config in cases:
            assert isinstance(config.recovery_policy(), StochasticPolicy)
            assert config.recovery_policy().replicates >= 4
    assert isinstance(scattering_policy(), StochasticPolicy)
    with pytest.raises(ContractError, match="replicates"):
        AbsorptionExperiment(replicates=2)
    with pytest.raises(ContractError, match="final validation"):
        replace(AbsorptionExperiment(), final_validation_batch=262145)


def test_recovery_experiment_module_has_an_executable_standalone_cli():
    result = subprocess.run(
        [sys.executable, "-m", "dpt.transport.recovery_experiments", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--mode" in result.stdout and "family" in result.stdout


def test_family_stationarity_tolerances_resolve_every_nonstationary_initial_density():
    for _, config in absorption_cases("family"):
        depth = config.layer_optical_depths[0]
        target = math.exp(-depth * config.target_density)
        current = math.exp(-depth * config.initial_density)
        gradient = (current - target) * -depth * config.initial_density * current
        policy = config.recovery_policy()
        assert abs(gradient) > policy.gradient_tolerance + policy.numerical_gradient_allowance
        assert policy.gradient_tolerance == pytest.approx(1e-5 * target**2, rel=1e-14)


def trace(
    spec,
    material_ids,
    absorption,
    scattering,
    positions,
    count,
    device,
    seed=717,
    log_density=0.0,
    derivative=False,
):
    workspace = prepare_transport(
        spec,
        material_ids=wp.array(material_ids, dtype=wp.int32, device=device),
        absorption=wp.array(absorption, dtype=wp.float64, device=device),
        scattering=wp.array(scattering, dtype=wp.float64, device=device),
        max_histories=count,
        device=device,
    )
    directions = np.tile([0.0, 0.0, 1.0], (count, 1))
    outputs = {
        f"out_{name}": wp.empty(count, dtype=dtype, device=device)
        for name, dtype in (
            ("pixel", wp.int32),
            ("score", wp.float64),
            ("energy", wp.float64),
            ("events", wp.int32),
            ("status", wp.int32),
        )
    }
    inputs = (
        wp.array(positions, dtype=wp.vec3d, device=device),
        wp.array(directions, dtype=wp.vec3d, device=device),
        wp.ones(count, dtype=wp.float64, device=device),
        wp.full(spec.grid.materials, math.exp(log_density), dtype=wp.float64, device=device),
    )
    if derivative:
        derivative_histories(
            *inputs,
            parameter=TransportParameter("log-material-density", 0),
            batch=HistoryBatch(seed, 0, count),
            workspace=workspace,
            out_pixel=outputs["out_pixel"],
            out_derivative=outputs["out_score"],
            out_status=outputs["out_status"],
        )
        return {name: outputs[f"out_{name}"].numpy() for name in ("pixel", "score", "status")}
    trace_histories(
        *inputs,
        batch=HistoryBatch(seed, 0, count),
        workspace=workspace,
        **outputs,
    )
    return {name.removeprefix("out_"): value.numpy() for name, value in outputs.items()}


def test_asymmetric_detector_hits_have_independent_row_major_indices(warp_device):
    detector = PlanarDetector((-1.1, -0.7), (0.7, 1.3), (5, 3), 3.0)
    assert (detector.width, detector.height, detector.image_shape) == (5, 3, (3, 5))
    positions = [
        (-1.1 + (column + 0.5) * 0.7, -0.7 + (row + 0.5) * 1.3, -1.0)
        for row in range(3)
        for column in range(5)
    ]
    positions.extend([(-1.2, 0.0, -1.0), (2.5, 0.0, -1.0), (0.0, 3.3, -1.0)])
    spec = TransportSpec(
        MaterialGrid((-3.0, -3.0, 0.0), (6.0, 7.0, 1.0), (1, 1, 1), 1),
        detector,
        80.0,
        "vacuum pixel-address regression fixture",
    )
    with cpu_test_context():
        actual = trace(spec, [0], [0.0], [0.0], positions, len(positions), warp_device)
    np.testing.assert_array_equal(actual["pixel"], [*range(15), -1, -1, -1])
    np.testing.assert_array_equal(actual["score"], [1.0] * 15 + [0.0] * 3)
    np.testing.assert_array_equal(actual["status"], np.zeros(len(positions)))
    # Mirroring and column-major tally mutations must change the expected image.
    expected = np.arange(15).reshape(detector.image_shape)
    assert not np.array_equal(expected, expected[:, ::-1])
    assert not np.array_equal(expected.ravel(), expected.T.ravel())


def test_coefficient_interpolation_material_offsets_nodes_and_interiors(warp_device):
    energies = np.array([1.0, 20.0, 80.0, 120.0])
    absorption = np.array([[0.7, 0.2, 0.13, 0.1], [4.0, 2.0, 0.9, 0.3]])
    scattering = np.array([[0.04, 0.03, 0.02, 0.01], [0.6, 0.4, 0.2, 0.1]])
    query = [(material, energy) for material in range(2) for energy in (1, 7, 20, 47, 80, 91, 120)]
    p = Parameters()
    p.energy_bins = len(energies)
    out = wp.empty(len(query), dtype=wp.vec2d, device=warp_device)
    wp.launch(
        inspect_coefficients,
        dim=len(query),
        inputs=[
            wp.array(query, dtype=wp.vec2d, device=warp_device),
            wp.array(energies, dtype=wp.float64, device=warp_device),
            wp.array(absorption.ravel(), dtype=wp.float64, device=warp_device),
            wp.array(scattering.ravel(), dtype=wp.float64, device=warp_device),
            p,
            out,
        ],
        device=warp_device,
    )
    expected = [
        [np.interp(energy, energies, absorption[m]), np.interp(energy, energies, scattering[m])]
        for m, energy in query
    ]
    np.testing.assert_allclose(out.numpy(), expected, rtol=2e-15, atol=0)


@pytest.mark.parametrize("energy", [20.0, 80.0, 662.0, 6000.0])
def test_compton_cosine_and_azimuth_goodness_of_fit(energy, warp_device):
    count = 2**18
    out = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(inspect_compton, dim=count, inputs=[energy, out], device=warp_device)
    values = out.numpy()
    np.testing.assert_array_equal(values[:, 3], np.zeros(count))
    edges = np.linspace(-1.0, 1.0, 33)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    cosine = (edges[1:] + edges[:-1])[:, None] / 2 + nodes * np.diff(edges)[:, None] / 2
    ratio = 1 / (1 + energy / 510.99895069 * (1 - cosine))
    density = ratio**2 * (ratio + 1 / ratio - (1 - cosine**2))
    masses = np.diff(edges) / 2 * (density @ weights)
    expected = masses / masses.sum() * count
    observed = np.histogram(values[:, 0], edges)[0]
    # Two independent 32-bin checks per energy, each with false failure <1e-6.
    assert stats.chisquare(observed, expected).pvalue > 1e-6
    azimuth = np.histogram(values[:, 1], np.linspace(0, 2 * math.pi, 33))[0]
    assert stats.chisquare(azimuth).pvalue > 1e-6
    predicted_energy = energy / (1 + energy / 510.99895069 * (1 - values[:, 0]))
    np.testing.assert_allclose(values[:, 2], predicted_energy, rtol=3e-15)


@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
def test_real_transport_oracle_recovers_known_density(estimator, warp_device):
    policy = StochasticPolicy(
        iterations=60,
        replicates=8,
        initial_batch=1024,
        maximum_batch=32768,
        unique_history_budget=8_000_000,
        gradient_tolerance=1e-4,
        numerical_gradient_allowance=1e-13,
    )
    with cpu_test_context():
        oracle = absorbing_oracle(estimator, policy.maximum_batch, warp_device)
        result = recover_expected_signal(oracle, (0.0,), seed=8157, policy=policy)
    density = math.exp(result.parameters[0])
    assert abs(density - 1.5) < (0.035 if estimator == "analogue" else 0.004)
    assert result.reason in ("gradient_band", "sampling_unresolved")
    assert any(step.accepted for step in result.steps)


@pytest.mark.parametrize("albedo", [0.2, 0.6, 0.9])
@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
def test_mixed_albedo_multiple_scattering_agrees_with_independent_all_orders_mc(
    albedo, estimator, warp_device
):
    count = 2**18
    reference = independent_isotropic_cube(albedo, count)
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 2.0), (1, 1, 1), 1),
        PlanarDetector((-3.0, -4.0), (2.0, 4.0), (3, 2), 3.0),
        80.0,
        "declared mixed-albedo all-orders cube",
        estimator=estimator,
    )
    with cpu_test_context():
        actual = trace(
            spec,
            [0],
            [1.2 * (1 - albedo)],
            [1.2 * albedo],
            np.tile([0.0, 0.0, -1.0], (count, 1)),
            count,
            warp_device,
        )
    np.testing.assert_array_equal(actual["status"] <= 1, np.ones(count, dtype=bool))
    scores = np.asarray(actual["score"], dtype=np.float64)
    pixels = np.asarray(actual["pixel"], dtype=np.int32)
    hit = pixels >= 0
    mean = np.bincount(pixels[hit], weights=scores[hit], minlength=6) / count
    second = np.bincount(pixels[hit], weights=scores[hit] ** 2, minlength=6) / count
    variance = (second - mean**2) / (count - 1)
    reference_variance = reference * (1 - reference) / (count - 1)
    # Per-pixel two-sided four-sigma test, false failure about 6.3e-5 under the
    # CLT; coarse pixels have hundreds of hits and independent reference streams.
    assert np.all(np.abs(mean - reference) < 4 * np.sqrt(variance + reference_variance))
    assert np.sum(actual["events"] >= 2) > 1000


def test_homogeneous_voxel_subdivision_preserves_scattered_histories(warp_device):
    count = 4096
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 2.0), (1, 1, 1), 1),
        PlanarDetector((-3.0, -4.0), (2.0, 4.0), (3, 2), 3.0),
        80.0,
        "homogeneous lattice subdivision regression",
        estimator="continuous-absorption",
    )
    finer = replace(
        spec,
        grid=MaterialGrid((-1.0, -1.0, 0.0), (0.5, 2 / 3, 0.4), (5, 3, 4), 1),
    )
    positions = np.tile([0.0, 0.0, -1.0], (count, 1))
    with cpu_test_context():
        coarse = trace(spec, [0], [0.3], [0.9], positions, count, warp_device)
        fine = trace(finer, [0] * 60, [0.3], [0.9], positions, count, warp_device)
    for name in ("pixel", "events", "status", "energy"):
        np.testing.assert_array_equal(coarse[name], fine[name])
    np.testing.assert_allclose(coarse["score"], fine["score"], rtol=3e-14, atol=0)


def test_unchecked_zero_amplitude_cannot_hide_infinite_source_weight(warp_device):
    with cpu_test_context():
        oracle = absorbing_oracle("analogue", 1, warp_device)
        w = oracle.workspace
        outputs = {
            f"out_{name}": wp.empty(1, dtype=dtype, device=warp_device)
            for name, dtype in (
                ("pixel", wp.int32),
                ("score", wp.float64),
                ("energy", wp.float64),
                ("events", wp.int32),
                ("status", wp.int32),
            )
        }
        # An empty material path makes the score exactly 0*Inf, independently
        # of survival sampling. The unchecked kernel must retain the NaN status.
        trace_histories(
            wp.array([[0.0, 0.0, 1.5]], dtype=wp.vec3d, device=warp_device),
            wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3d, device=warp_device),
            wp.array([math.inf], dtype=wp.float64, device=warp_device),
            wp.ones(1, dtype=wp.float64, device=warp_device),
            batch=HistoryBatch(71, 0, 1),
            workspace=w,
            source_amplitude=0.0,
            validate=False,
            **outputs,
        )
        with pytest.raises(NumericalError, match="nonfinite"):
            w.check_status()
    assert outputs["out_status"].numpy()[0] == 4
    assert math.isnan(outputs["out_score"].numpy()[0])


def test_repeated_transport_inverse_calls_create_no_owned_device_array_allocations(
    warp_device, monkeypatch, request: pytest.FixtureRequest
):
    with cpu_test_context():
        oracle = absorbing_oracle("continuous-absorption", 32, warp_device)
        first, second = HistoryBatch(341, 0, 32), HistoryBatch(341, 32, 32)
        oracle.gradient_replicate((0.0,), first, second)
        oracle.change_replicate((0.0,), (0.01,), first, second)
        allocations = []
        original = wp.array.__init__

        def monitored(array, *args, **kwargs):
            original(array, *args, **kwargs)
            # Slice/reshape descriptors have no deleter and own no memory.
            # Observe the array constructor itself, including temporary buffers.
            if getattr(array, "deleter", None) is not None:
                allocations.append((str(array.device), array.capacity))

        monkeypatch.setattr(wp.array, "__init__", monitored)
        for i in range(5):
            coordinate = i * 0.01
            oracle.gradient_replicate((coordinate,), first, second)
            oracle.change_replicate((coordinate,), (coordinate + 0.01,), first, second)
        operator_device = str(wp.get_device(warp_device))
        device_allocations = [item for item in allocations if item[0] == operator_device]
        host_allocations = [item for item in allocations if item[0] != operator_device]
        request.node.user_properties.extend(
            (
                ("owned_operator_device_allocations", len(device_allocations)),
                ("owned_host_status_allocations", len(host_allocations)),
            )
        )
        # The repeated-call contract covers device buffers. Checked CUDA status
        # readbacks use array.numpy(), which stages a four-byte word on the CPU.
        # Keep observing every allocation: CUDA temporaries and larger/unrelated
        # host buffers fail instead of being hidden by the device filter.
        assert device_allocations == []
        assert all(item == ("cpu", 4) for item in host_allocations)


@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
@pytest.mark.parametrize("law", ["isotropic-elastic", "free-electron-compton"])
def test_multiple_scatter_likelihood_derivative_integrates_forward_difference(
    estimator, law, warp_device
):
    count = 2**18
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 2.0), (1, 1, 1), 1),
        PlanarDetector((-2.0, -2.0), (4.0, 4.0), (1, 1), 3.0),
        80.0,
        "multiple-scatter likelihood-ratio integration identity",
        coefficient_energies_kev=(1.0, 80.0),
        scattering_law=law,
        estimator=estimator,
    )
    h = 0.05
    positions = np.tile([0.0, 0.0, -1.0], (count, 1))
    # Both partial coefficients change with energy, so interpolation after
    # every Compton collision is exercised rather than hidden by a flat table.
    absorption = [0.5, 0.3]
    scattering = [0.8, 1.2]
    with cpu_test_context():
        before = trace(
            spec,
            [0],
            absorption,
            scattering,
            positions,
            count,
            warp_device,
            seed=76371,
            log_density=-h,
        )["score"]
        after = trace(
            spec,
            [0],
            absorption,
            scattering,
            positions,
            count,
            warp_device,
            seed=76371,
            log_density=h,
        )["score"]
        derivative = []
        for index, coordinate in enumerate((-h, 0.0, h)):
            derivative.append(
                trace(
                    spec,
                    [0],
                    absorption,
                    scattering,
                    positions,
                    count,
                    warp_device,
                    seed=98717 + 1543 * index,
                    log_density=coordinate,
                    derivative=True,
                )["score"]
            )
    change = after - before  # Common random numbers reduce the difference variance.
    integral = h / 3 * (derivative[0].mean() + 4 * derivative[1].mean() + derivative[2].mean())
    variance = change.var(ddof=1) / count + (h / 3) ** 2 * sum(
        weight**2 * values.var(ddof=1) / count
        for weight, values in zip((1, 4, 1), derivative, strict=True)
    )
    assert abs(change.mean() - integral) < 4 * math.sqrt(variance) + 2e-6
    # The signal is resolved strongly enough to reject a zero derivative.
    assert abs(change.mean()) > 10 * math.sqrt(variance)
