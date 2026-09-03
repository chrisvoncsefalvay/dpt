"""Independent expected-value and complete-history readout references."""

# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportMissingParameterType=false, reportPrivateUsage=false
# Warp kernel annotations use runtime DSL expressions.
# pyright: reportInvalidTypeForm=false
import importlib
import math
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest

from dpt.contracts import NumericalError
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.parameters import CoefficientParameter, coefficient_derivative_histories
from dpt.transport.physical import (
    PhysicalTransport,
    xraylib_atomic_scattering,
    xraylib_fluorescence_lines,
)
from dpt.transport.readout import TransportReadout, prepare_transport_readout, trace_readout
from dpt.transport.rng import HistoryBatch
from dpt.transport.source_spectrum import (
    SourceSpectrum,
    SpectrumParameter,
    spectrum_derivative_histories,
    spectrum_logit_derivative,
)

wp: Any = importlib.import_module("warp")
transport_kernels: Any = importlib.import_module("dpt.transport.kernels")


@wp.kernel
def _kn_phase_reference_values(
    energies: wp.array(dtype=wp.float64),
    cosines: wp.array(dtype=wp.float64),
    out: wp.array(dtype=wp.float64, ndim=2),
):
    energy, angle = wp.tid()
    out[energy, angle] = transport_kernels.free_electron_phase_pdf(energies[energy], cosines[angle])


def slab(
    device: str,
    count: int,
    absorption: float | list[float] = 0.23,
    scattering: float | list[float] = 0.0,
    estimator: Literal["analogue", "continuous-absorption"] = "continuous-absorption",
    nodes: tuple[float, ...] = (),
):
    def array(value, dtype=None):
        return wp.array(value, dtype=wp.float64 if dtype is None else dtype, device=device)

    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
        80.0,
        "analytic slab",
        coefficient_energies_kev=nodes,
        estimator=estimator,
        max_events=128,
    )
    transport = prepare_transport(
        spec,
        material_ids=array([0], wp.int32),
        absorption=array(absorption if isinstance(absorption, list) else [absorption]),
        scattering=array(scattering if isinstance(scattering, list) else [scattering]),
        max_histories=count,
        device=device,
    )
    inputs = (
        array([[0.0, 0.0, -1.0]] * count, wp.vec3d),
        array([[0.0, 0.0, 1.0]] * count, wp.vec3d),
        array([1.0] * count),
        array([1.0]),
    )
    return transport, inputs, array


@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
def test_coefficient_expected_score_separate_terms_and_replay(estimator, warp_device):
    count = 8192
    transport, inputs, arr = slab(warp_device, count, estimator=estimator)
    pixel = arr([0] * count, wp.int32)
    score = arr([0.0] * count)
    likelihood = arr([0.0] * count)
    pathwise = arr([0.0] * count)
    status = arr([0] * count, wp.int32)
    batch = HistoryBatch(281731, 0, count)
    coefficient_derivative_histories(
        *inputs,
        parameter=CoefficientParameter("absorption", 0, 0),
        batch=batch,
        workspace=transport,
        out_pixel=pixel,
        out_score=score,
        out_likelihood=likelihood,
        out_pathwise=pathwise,
        out_status=status,
    )
    expected = -0.23 * math.exp(-0.23)
    actual = float(np.mean(likelihood.numpy() + pathwise.numpy()))
    allowance = 6 * 0.23 * math.sqrt(math.exp(-0.23) * (1 - math.exp(-0.23)) / count) + 1e-14
    assert actual == pytest.approx(expected, abs=allowance)
    if estimator == "continuous-absorption":
        assert not likelihood.numpy().any()
        np.testing.assert_allclose(pathwise.numpy(), expected, rtol=1e-14)
    else:
        assert not pathwise.numpy().any()
    rawpixel = arr([0] * count, wp.int32)
    rawscore = arr([0.0] * count)
    trace_histories(
        *inputs,
        batch=batch,
        workspace=transport,
        out_pixel=rawpixel,
        out_score=rawscore,
        out_energy=arr([0.0] * count),
        out_events=arr([0] * count, wp.int32),
        out_status=arr([0] * count, wp.int32),
    )
    np.testing.assert_array_equal(pixel.numpy(), rawpixel.numpy())
    np.testing.assert_array_equal(score.numpy(), rawscore.numpy())


def test_interpolated_log_node_derivative_against_analytic_and_finite_difference(warp_device):
    transport, inputs, arr = slab(
        warp_device, 19, absorption=[0.1, 0.5], scattering=[0.0, 0.0], nodes=(40.0, 120.0)
    )
    readout = prepare_transport_readout(
        transport,
        spec=TransportReadout(next_event=True, tally_strategy="deterministic"),
        parameter=CoefficientParameter("absorption", 0, 1),
    )
    mean = arr([0.0])
    derivative = arr([0.0])
    trace_readout(
        *inputs,
        batch=HistoryBatch(123, 0, 19),
        workspace=readout,
        out_mean=mean,
        out_derivative_mean=derivative,
    )
    analytic = -0.25 * math.exp(-0.3)
    epsilon = 1e-5
    finite = (
        math.exp(-(0.1 + 0.5 * math.exp(epsilon)) / 2)
        - math.exp(-(0.1 + 0.5 * math.exp(-epsilon)) / 2)
    ) / (2 * epsilon)
    assert derivative.numpy()[0] == pytest.approx(analytic, rel=1e-14)
    assert derivative.numpy()[0] == pytest.approx(finite, rel=1e-10)
    assert mean.numpy()[0] == pytest.approx(math.exp(-0.3), rel=1e-14)


def test_readout_modes_energy_conservation_and_fixed_order(warp_device):
    transport, inputs, arr = slab(warp_device, 2048, absorption=0.12, scattering=0.35)
    batch = HistoryBatch(2781, 9, 2048)
    values = []
    for mode in ("total", "primary", "scatter"):
        readout = prepare_transport_readout(
            transport, spec=TransportReadout(mode=mode, tally_strategy="deterministic")
        )
        mean = arr([0.0])
        variance = arr([0.0])
        trace_readout(
            *inputs, batch=batch, workspace=readout, out_mean=mean, out_variance_of_mean=variance
        )
        first = mean.numpy().copy()
        trace_readout(
            *inputs, batch=batch, workspace=readout, out_mean=mean, out_variance_of_mean=variance
        )
        np.testing.assert_array_equal(first.view(np.uint64), mean.numpy().view(np.uint64))
        history = readout._scores.numpy()[: batch.count]
        assert variance.numpy()[0] == pytest.approx(
            np.var(history, ddof=1) / batch.count, rel=1e-13
        )
        values.append(first[0])
    assert values[0] == pytest.approx(values[1] + values[2], rel=1e-14)
    binned = prepare_transport_readout(
        transport,
        spec=TransportReadout(energy_edges_kev=(0.0, 40.0, 81.0), tally_strategy="deterministic"),
    )
    output = arr([0.0, 0.0])
    trace_readout(*inputs, batch=batch, workspace=binned, out_mean=output)
    assert output.numpy()[0] == 0
    assert sum(output.numpy()) == pytest.approx(values[0], rel=1e-14)
    excluded = prepare_transport_readout(
        transport,
        spec=TransportReadout(energy_edges_kev=(0.0, 40.0), tally_strategy="deterministic"),
    )
    one_bin = arr([0.0])
    trace_readout(*inputs, batch=batch, workspace=excluded, out_mean=one_bin)
    assert one_bin.numpy()[0] == 0.0


def test_forced_primary_roulette_and_splitting_analytic_means(warp_device):
    count = 8192
    transport, inputs, arr = slab(warp_device, count, absorption=0.23, estimator="analogue")
    target = math.exp(-0.23)
    for branches in (1, 4):
        readout = prepare_transport_readout(
            transport,
            spec=TransportReadout(
                next_event=True, split_branches=branches, tally_strategy="deterministic"
            ),
        )
        mean = arr([0.0])
        trace_readout(*inputs, batch=HistoryBatch(491, 0, count), workspace=readout, out_mean=mean)
        assert mean.numpy()[0] == pytest.approx(target, rel=1e-14)
    with pytest.raises(TransportError, match="preparation budget"):
        prepare_transport_readout(transport, spec=TransportReadout(), max_bytes=1)


def test_next_event_unbiased_against_direct_transport_and_derivative_difference(warp_device):
    count = 8192
    transport, inputs, arr = slab(warp_device, count, absorption=0.15, scattering=0.4)
    direct = prepare_transport_readout(
        transport, spec=TransportReadout(mode="scatter", tally_strategy="deterministic")
    )
    forced = prepare_transport_readout(
        transport,
        spec=TransportReadout(mode="scatter", next_event=True, tally_strategy="deterministic"),
        parameter=CoefficientParameter("scattering", 0, 0),
    )
    plain = arr([0.0])
    connected = arr([0.0])
    derivative = arr([0.0])
    differences = []
    derivatives = []
    for replicate in range(8):
        batch = HistoryBatch(73512 + replicate, 0, count)
        trace_readout(*inputs, batch=batch, workspace=direct, out_mean=plain)
        trace_readout(
            *inputs,
            batch=batch,
            workspace=forced,
            out_mean=connected,
            out_derivative_mean=derivative,
        )
        differences.append(float(connected.numpy()[0] - plain.numpy()[0]))
        derivatives.append(float(derivative.numpy()[0]))
    assert (
        abs(np.mean(differences))
        <= 6 * np.std(differences, ddof=1) / math.sqrt(len(differences)) + 1e-4
    )
    # Independent endpoint batches give a statistical finite-difference reference.
    epsilon = 0.04
    endpoints = []
    for sign in (-1, 1):
        endpoint, endpointinputs, _ = slab(
            warp_device, count, absorption=0.15, scattering=0.4 * math.exp(sign * epsilon)
        )
        workspace = prepare_transport_readout(
            endpoint,
            spec=TransportReadout(mode="scatter", next_event=True, tally_strategy="deterministic"),
        )
        estimates = []
        for replicate in range(8):
            trace_readout(
                *endpointinputs,
                batch=HistoryBatch(175312 + replicate + 100 * sign, 0, count),
                workspace=workspace,
                out_mean=connected,
            )
            estimates.append(float(connected.numpy()[0]))
        endpoints.append(estimates)
    finite = (np.array(endpoints[1]) - np.array(endpoints[0])) / (2 * epsilon)
    difference = np.mean(derivatives) - np.mean(finite)
    error = math.sqrt(np.var(derivatives, ddof=1) / 8 + np.var(finite, ddof=1) / 8)
    assert abs(difference) <= 6 * error + 2e-4


def test_spectrum_logit_expected_score_against_independent_discrete_sum(warp_device):
    spectrum = SourceSpectrum((40.0, 80.0), (0.25, 0.75), "discrete sum reference")
    transport, _, arr = physical_slab(
        warp_device,
        8,
        absorption=[0.1, 0.5],
        scattering=[0.0, 0.0],
        rayleigh=[0.0, 0.0],
        spectrum=spectrum,
    )
    scores = arr([2.0] * 2 + [5.0] * 6)
    selected = arr([0] * 2 + [1] * 6, wp.int32)
    out = arr([0.0] * 8)
    spectrum_logit_derivative(
        arr([0] * 8, wp.int32),
        scores,
        selected,
        spectrum=spectrum,
        energy_bin=0,
        batch=HistoryBatch(121, 0, 8),
        workspace=transport,
        out_derivative=out,
    )
    expected = 0.25 * 2.0 - 0.25 * (0.25 * 2.0 + 0.75 * 5.0)
    assert np.mean(out.numpy()) == pytest.approx(expected, rel=1e-14)


def test_mixed_rate_precision_against_numeric_and_statistical_reference(warp_device):
    count = 2048
    transport, inputs, arr = slab(warp_device, count, absorption=0.230000000021, scattering=0.31)
    means = []
    for precision in ("reference", "mixed-rates"):
        workspace = prepare_transport_readout(
            transport, spec=TransportReadout(precision=precision, tally_strategy="deterministic")
        )
        mean = arr([0.0])
        trace_readout(
            *inputs, batch=HistoryBatch(2971, 0, count), workspace=workspace, out_mean=mean
        )
        means.append(float(mean.numpy()[0]))
    assert abs(means[1] - means[0]) < 5e-7


def test_mixed_interpolation_arithmetic_against_independent_absorbing_slab(warp_device):
    count = 8
    absorption = [0.1000000031, 0.4700000149]
    transport, inputs, arr = slab(
        warp_device, count, absorption=absorption, scattering=[0.0, 0.0], nodes=(40.0, 120.0)
    )
    material_density = 1.2300000123
    inputs = (*inputs[:3], arr([material_density]))
    rates = (
        material_density * (absorption[0] + 0.5 * (absorption[1] - absorption[0])),
        float(
            np.float32(material_density)
            * np.float32(
                np.float32(0.5) * np.float32(absorption[0])
                + np.float32(0.5) * np.float32(absorption[1])
            )
        ),
    )
    values = []
    for precision, rate in zip(("reference", "mixed-rates"), rates, strict=True):
        readout = prepare_transport_readout(
            transport,
            spec=TransportReadout(
                next_event=True, precision=precision, tally_strategy="deterministic"
            ),
        )
        mean = arr([0.0])
        trace_readout(
            *inputs, batch=HistoryBatch(53279, 0, count), workspace=readout, out_mean=mean
        )
        values.append(float(mean.numpy()[0]))
        assert values[-1] == pytest.approx(math.exp(-rate), rel=2e-15)
    assert values[0] != values[1]


def test_free_electron_next_event_pdf_against_independent_quadrature(warp_device):
    energies = np.array([1e-8, 1.0, 25.0, 26.0, 80.0, 511.0, 5000.0])
    cosine, weights = np.polynomial.legendre.leggauss(256)
    output = wp.empty((len(energies), len(cosine)), dtype=wp.float64, device=warp_device)
    wp.launch(
        _kn_phase_reference_values,
        dim=output.shape,
        inputs=[
            wp.array(energies, dtype=wp.float64, device=warp_device),
            wp.array(cosine, dtype=wp.float64, device=warp_device),
            output,
        ],
        device=warp_device,
    )
    ratio = 1 / (1 + energies[:, None] / 510.99895069 * (1 - cosine))
    shape = ratio**3 + ratio - ratio**2 * (1 - cosine**2)
    reference = shape / (2 * math.pi * np.sum(shape * weights, axis=1)[:, None])
    assert np.allclose(output.numpy(), reference, rtol=2e-12, atol=1e-14)


def test_source_splitting_original_history_variance_matches_binomial_reference(warp_device):
    count = 8192
    transport, inputs, arr = slab(warp_device, count, absorption=0.7, estimator="analogue")
    p = math.exp(-0.7)
    for branches in (1, 4):
        workspace = prepare_transport_readout(
            transport,
            spec=TransportReadout(split_branches=branches, tally_strategy="deterministic"),
        )
        mean = arr([0.0])
        variance = arr([0.0])
        trace_readout(
            *inputs,
            batch=HistoryBatch(42317, 0, count),
            workspace=workspace,
            out_mean=mean,
            out_variance_of_mean=variance,
        )
        expected = p * (1 - p) / (count * branches)
        assert variance.numpy()[0] == pytest.approx(expected, rel=0.09)
        assert mean.numpy()[0] == pytest.approx(p, abs=6 * math.sqrt(expected))


def test_weight_roulette_preserves_expected_score_in_independent_replicates(warp_device):
    count = 4096
    transport, inputs, arr = slab(warp_device, count, absorption=1.1, scattering=1.2)
    reference = prepare_transport_readout(
        transport, spec=TransportReadout(tally_strategy="deterministic")
    )
    roulette = prepare_transport_readout(
        transport,
        spec=TransportReadout(
            roulette_threshold=0.8, roulette_survival=0.4, tally_strategy="deterministic"
        ),
    )
    direct = arr([0.0])
    reduced = arr([0.0])
    differences = []
    terminated = 0
    for replicate in range(12):
        batch = HistoryBatch(9913 + replicate, 0, count)
        trace_readout(*inputs, batch=batch, workspace=reference, out_mean=direct)
        trace_readout(*inputs, batch=batch, workspace=roulette, out_mean=reduced)
        differences.append(float(reduced.numpy()[0] - direct.numpy()[0]))
        terminated += int(np.sum(roulette._history_status.numpy()[:count] == 8))
    assert terminated > 0
    assert abs(np.mean(differences)) <= 6 * np.std(differences, ddof=1) / math.sqrt(12) + 1e-4


def test_killed_roulette_history_does_not_run_an_angular_rejection(warp_device):
    count = 128
    base, inputs, array = slab(
        warp_device, count, absorption=[0.1, 0.1], scattering=[10.0, 10.0], nodes=(1.0, 80.0)
    )
    transport = prepare_transport(
        replace(
            base.spec, scattering_law="free-electron-compton", max_events=1, max_angle_trials=1
        ),
        material_ids=base.material_ids,
        absorption=base.absorption,
        scattering=base.scattering,
        max_histories=count,
        device=warp_device,
    )
    readout = prepare_transport_readout(
        transport,
        spec=TransportReadout(roulette_threshold=2.0, roulette_survival=2**-32),
    )
    trace_readout(
        *inputs, batch=HistoryBatch(92761, 0, count), workspace=readout, out_mean=array([0.0])
    )
    assert np.all(np.isin(readout._history_status.numpy(), (0, 8)))
    assert np.sum(readout._history_status.numpy() == 8) > 0


def physical_slab(device, count, absorption, scattering, rayleigh, spectrum=None, majorant=()):
    transport, inputs, array = slab(
        device,
        count,
        absorption=absorption,
        scattering=scattering,
        estimator="analogue",
        nodes=(40.0, 80.0) if spectrum is not None else (),
    )
    atomic = xraylib_atomic_scattering(13, (0.0, 0.1, 1.0, 10.0))
    physical = PhysicalTransport(
        (atomic,), tuple(rayleigh), majorant_mm_inverse=majorant, spectrum=spectrum
    )
    transport = prepare_transport(
        transport.spec,
        material_ids=transport.material_ids,
        absorption=transport.absorption,
        scattering=transport.scattering,
        max_histories=count,
        device=device,
        physical=physical,
    )
    return transport, inputs, array


def test_prepared_spectral_logit_replay_and_forced_primary_against_discrete_sum(warp_device):
    count = 8192
    source = SourceSpectrum((40.0, 80.0), (0.25, 0.75), "analytic discrete absorbing spectrum")
    transport, inputs, array = physical_slab(
        warp_device, count, [0.1, 0.5], [0.0, 0.0], [0.0, 0.0], source
    )
    expected = 0.25 * math.exp(-0.1) - 0.25 * (0.25 * math.exp(-0.1) + 0.75 * math.exp(-0.5))
    batch = HistoryBatch(519103, 0, count)
    derivative = array([0.0] * count)
    spectrum_derivative_histories(
        *inputs,
        parameter=SpectrumParameter(0),
        batch=batch,
        workspace=transport,
        out_pixel=array([0] * count, wp.int32),
        out_score=array([0.0] * count),
        out_derivative=derivative,
        out_status=array([0] * count, wp.int32),
    )
    values = derivative.numpy()
    assert abs(np.mean(values) - expected) <= 6 * np.std(values, ddof=1) / math.sqrt(count)
    workspace = prepare_transport_readout(
        transport,
        spec=TransportReadout(next_event=True, tally_strategy="deterministic"),
        parameter=SpectrumParameter(0),
    )
    mean = array([0.0])
    gradient = array([0.0])
    trace_readout(
        *inputs, batch=batch, workspace=workspace, out_mean=mean, out_derivative_mean=gradient
    )
    forced = workspace._derivatives.numpy()[:count]
    assert abs(gradient.numpy()[0] - expected) <= 6 * np.std(forced, ddof=1) / math.sqrt(count)
    assert gradient.numpy()[0] == pytest.approx(np.mean(forced), rel=1e-13)


def test_spectral_logit_gradient_against_independent_statistical_finite_difference(warp_device):
    count, replicates, epsilon = 4096, 12, 0.15
    source = SourceSpectrum(
        (40.0, 80.0), (0.25, 0.75), "spectral logit finite-difference reference"
    )
    transport, inputs, array = physical_slab(
        warp_device, count, [0.1, 0.5], [0.0, 0.0], [0.0, 0.0], source
    )
    readout = prepare_transport_readout(
        transport,
        spec=TransportReadout(next_event=True, tally_strategy="deterministic"),
        parameter=SpectrumParameter(0),
    )
    mean, gradient = array([0.0]), array([0.0])
    derivatives = []
    for replicate in range(replicates):
        trace_readout(
            *inputs,
            batch=HistoryBatch(89303 + replicate, 0, count),
            workspace=readout,
            out_mean=mean,
            out_derivative_mean=gradient,
        )
        derivatives.append(float(gradient.numpy()[0]))
    endpoints = []
    for sign in (-1, 1):
        p = 0.25 * math.exp(sign * epsilon) / (0.25 * math.exp(sign * epsilon) + 0.75)
        shifted = SourceSpectrum((40.0, 80.0), (p, 1 - p), "independent perturbed spectral source")
        endpoint, endpointinputs, _ = physical_slab(
            warp_device, count, [0.1, 0.5], [0.0, 0.0], [0.0, 0.0], shifted
        )
        forced = prepare_transport_readout(
            endpoint, spec=TransportReadout(next_event=True, tally_strategy="deterministic")
        )
        values = []
        for replicate in range(replicates):
            trace_readout(
                *endpointinputs,
                batch=HistoryBatch(198307 + sign * 31 + replicate, 0, count),
                workspace=forced,
                out_mean=mean,
            )
            values.append(float(mean.numpy()[0]))
        endpoints.append(np.array(values))
    finite = (endpoints[1] - endpoints[0]) / (2 * epsilon)
    error = math.sqrt(
        np.var(derivatives, ddof=1) / replicates + np.var(finite, ddof=1) / replicates
    )
    assert abs(np.mean(derivatives) - np.mean(finite)) <= 6 * error + 2e-4


def test_mixed_rate_rounding_cannot_exceed_a_fixed_coherent_partial(warp_device):
    count = 128
    coefficient = 0.900000002
    transport, inputs, array = physical_slab(
        warp_device, count, [0.0], [coefficient], [coefficient]
    )
    readout = prepare_transport_readout(transport, spec=TransportReadout(precision="mixed-rates"))
    with pytest.raises(NumericalError):
        trace_readout(
            *inputs, batch=HistoryBatch(41231, 0, count), workspace=readout, out_mean=array([0.0])
        )


def test_mixed_rate_rounding_must_obey_the_declared_woodcock_majorant(warp_device):
    count = 128
    coefficient = 0.90000003
    transport, inputs, array = physical_slab(
        warp_device,
        count,
        [0.0],
        [coefficient],
        [0.0],
        majorant=(math.nextafter(coefficient, math.inf),),
    )
    readout = prepare_transport_readout(transport, spec=TransportReadout(precision="mixed-rates"))
    with pytest.raises(NumericalError):
        trace_readout(
            *inputs, batch=HistoryBatch(51231, 0, count), workspace=readout, out_mean=array([0.0])
        )


@pytest.mark.parametrize("woodcock", [False, True])
def test_physical_primary_excludes_null_events_and_has_correct_node_score(woodcock, warp_device):
    count = 8192
    transport, inputs, array = physical_slab(
        warp_device, count, [0.3], [0.0], [0.0], majorant=(1.1,) if woodcock else ()
    )
    workspace = prepare_transport_readout(
        transport,
        spec=TransportReadout(mode="primary", tally_strategy="deterministic"),
        parameter=CoefficientParameter("absorption", 0, 0),
    )
    mean = array([0.0])
    gradient = array([0.0])
    trace_readout(
        *inputs,
        batch=HistoryBatch(37113, 0, count),
        workspace=workspace,
        out_mean=mean,
        out_derivative_mean=gradient,
    )
    expected = math.exp(-0.3)
    assert mean.numpy()[0] == pytest.approx(
        expected, abs=6 * math.sqrt(expected * (1 - expected) / count)
    )
    derivatives = workspace._derivatives.numpy()[:count]
    assert abs(gradient.numpy()[0] + 0.3 * expected) <= 6 * np.std(derivatives, ddof=1) / math.sqrt(
        count
    )


def test_rayleigh_next_event_and_fixed_coherent_partial_node_gradient(warp_device):
    count = 4096
    transport, inputs, array = physical_slab(warp_device, count, [0.15], [0.4], [0.2])
    plain = prepare_transport_readout(
        transport, spec=TransportReadout(mode="scatter", tally_strategy="deterministic")
    )
    forced = prepare_transport_readout(
        transport,
        spec=TransportReadout(mode="scatter", next_event=True, tally_strategy="deterministic"),
        parameter=CoefficientParameter("scattering", 0, 0),
    )
    mean = array([0.0])
    direct = array([0.0])
    gradient = array([0.0])
    differences = []
    gradients = []
    for replicate in range(12):
        batch = HistoryBatch(53917 + replicate, 0, count)
        trace_readout(*inputs, batch=batch, workspace=plain, out_mean=direct)
        trace_readout(
            *inputs, batch=batch, workspace=forced, out_mean=mean, out_derivative_mean=gradient
        )
        differences.append(float(mean.numpy()[0] - direct.numpy()[0]))
        gradients.append(float(gradient.numpy()[0]))
    assert abs(np.mean(differences)) <= 6 * np.std(differences, ddof=1) / math.sqrt(12) + 1e-4
    epsilon = 0.08
    endpoints = []
    for sign in (-1, 1):
        endpoint, endpointinputs, _ = physical_slab(
            warp_device, count, [0.15], [0.4 * math.exp(sign * epsilon)], [0.2]
        )
        readout = prepare_transport_readout(
            endpoint,
            spec=TransportReadout(mode="scatter", next_event=True, tally_strategy="deterministic"),
        )
        values = []
        for replicate in range(12):
            trace_readout(
                *endpointinputs,
                batch=HistoryBatch(82111 + replicate + 100 * sign, 0, count),
                workspace=readout,
                out_mean=mean,
            )
            values.append(float(mean.numpy()[0]))
        endpoints.append(values)
    finite = (np.array(endpoints[1]) - np.array(endpoints[0])) / (2 * epsilon)
    error = math.sqrt(np.var(finite, ddof=1) / 12 + np.var(gradients, ddof=1) / 12)
    assert abs(np.mean(finite) - np.mean(gradients)) <= 6 * error + 5e-4


def test_fluorescence_next_event_matches_independently_detected_line_photons(warp_device):
    count, replicates = 4096, 12
    base, inputs, array = slab(
        warp_device,
        count,
        absorption=[0.5, 0.5],
        scattering=[0.0, 0.0],
        nodes=(1.0, 80.0),
        estimator="analogue",
    )
    lines = xraylib_fluorescence_lines(29, (1.0, 80.0), lines=("KL2_LINE", "KL3_LINE"))
    transport = prepare_transport(
        base.spec,
        material_ids=base.material_ids,
        absorption=base.absorption,
        scattering=base.scattering,
        physical=PhysicalTransport(
            (xraylib_atomic_scattering(29, (0.0, 0.1, 1.0, 10.0)),),
            (0.0, 0.0),
            fluorescence=(lines,),
        ),
        max_histories=count,
        device=warp_device,
    )
    direct = prepare_transport_readout(
        transport, spec=TransportReadout(mode="scatter", tally_strategy="deterministic")
    )
    forced = prepare_transport_readout(
        transport,
        spec=TransportReadout(mode="scatter", next_event=True, tally_strategy="deterministic"),
    )
    mean = array([0.0])
    reference = array([0.0])
    differences, forced_values = [], []
    for replicate in range(replicates):
        trace_readout(
            *inputs,
            batch=HistoryBatch(837211 + replicate, 0, count),
            workspace=direct,
            out_mean=reference,
        )
        trace_readout(
            *inputs,
            batch=HistoryBatch(981337 + replicate, 0, count),
            workspace=forced,
            out_mean=mean,
        )
        differences.append(float(mean.numpy()[0] - reference.numpy()[0]))
        forced_values.append(float(mean.numpy()[0]))
    assert np.mean(forced_values) > 0
    assert abs(np.mean(differences)) <= 6 * np.std(differences, ddof=1) / math.sqrt(replicates)
