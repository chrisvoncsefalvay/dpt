"""Native impulse scattering, full-path replay and joint-density checks."""

# Warp decorators and annotations are runtime DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
import importlib
import math
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import binom

from dpt.transport import (
    HistoryBatch,
    MaterialGrid,
    PhysicalTransport,
    PlanarDetector,
    SamplingDesign,
    TransportSpec,
    prepare_transport_sampling,
    xraylib_atomic_scattering,
    xraylib_bound_compton,
)
from dpt.transport.bound_compton import impulse_angular_weight
from dpt.transport.energy_balance import trace_energy_balance
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.transport.parameters import CoefficientParameter
from dpt.transport.readout import TransportReadout, prepare_transport_readout, trace_readout

wp: Any = importlib.import_module("warp")
bound: Any = importlib.import_module("dpt.transport.bound_compton_kernels")
physical_kernels: Any = importlib.import_module("dpt.transport.physical_kernels")
transport_kernels: Any = importlib.import_module("dpt.transport.kernels")


@wp.kernel(enable_backward=False)
def _sample_bound(
    energy: wp.float64, physical: physical_kernels.PhysicalParameters, out: wp.array(dtype=wp.vec4d)
):
    index = wp.tid()
    out[index] = bound.bound_scatter_event(
        energy, 0, wp.uint64(84301), wp.uint64(index), wp.uint32(0), 4096, physical
    )


@wp.kernel(enable_backward=False)
def _fixed_angle(
    energy: wp.float64,
    cosine: wp.float64,
    physical: physical_kernels.PhysicalParameters,
    out: wp.array(dtype=wp.vec4d),
):
    index = wp.tid()
    out[index] = bound.bound_energy_event(
        energy, cosine, 0, wp.uint64(48217), wp.uint64(index), wp.uint32(0), 0, physical
    )


@wp.kernel(enable_backward=False)
def _pdf(
    energy: wp.float64,
    physical: physical_kernels.PhysicalParameters,
    cosines: wp.array(dtype=wp.float64),
    out: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    out[index] = bound.bound_phase_pdf(energy, cosines[index], 0, physical)


@wp.kernel(enable_backward=False)
def _joint_scores(
    energy: wp.float64,
    physical: physical_kernels.PhysicalParameters,
    states: wp.array(dtype=wp.vec2d),
    out: wp.array(dtype=wp.vec3d),
):
    index = wp.tid()
    out[index] = bound.bound_joint_scores(energy, states[index][0], states[index][1], 0, physical)


@wp.kernel(enable_backward=False)
def _normalised_density_score(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: transport_kernels.Parameters,
    out: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    result = transport_kernels.walk(
        positions[index],
        directions[index],
        weights[index],
        wp.float64(1.0),
        wp.uint64(99121),
        wp.uint64(index),
        0,
        wp.vec4i(-1),
        0,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        parameters,
    )
    out[index] = result.density_score
    status[index] = result.status


def _workspace(
    device: str,
    energy: float = 80.0,
    count: int = 8192,
    max_trials: int = 4096,
    max_events: int = 128,
    atomic_number: int = 13,
) -> Any:
    table = xraylib_bound_compton(atomic_number)
    atomic = xraylib_atomic_scattering(atomic_number, (0.0, 0.005, 0.1, 1.0, 10.0, 100.0, 1000.0))
    physical = PhysicalTransport((atomic,), (0.0, 0.0), cutoff_kev=0.1, bound_compton=(table,))
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-100.0, -100.0), (200.0, 200.0), (1, 1), 2.0),
        energy,
        "actual aluminium shells and analytical impulse profile",
        coefficient_energies_kev=(0.1, energy),
        scattering_law="bound-electron-compton",
        max_events=max_events,
        max_angle_trials=max_trials,
    )
    return prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=device),
        absorption=wp.zeros(2, dtype=wp.float64, device=device),
        scattering=wp.full(2, 0.3, dtype=wp.float64, device=device),
        max_histories=count,
        physical=physical,
        device=device,
    )


@pytest.mark.parametrize("energy", [0.1, 0.2])
def test_low_energy_hydrogen_conditional_sampler_removes_tiny_cross_section(
    warp_device: str, energy: float
) -> None:
    # A broad coefficient support is independent of this fixed-angle native
    # check, where actual H binding/profile suppress total scattering by70orders.
    workspace = _workspace(warp_device, energy + 0.001, 4096, atomic_number=1)
    out = wp.empty(4096, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _sample_bound,
        dim=4096,
        device=warp_device,
        inputs=[energy, workspace._parameters.physics, out],
    )
    values = out.numpy()
    assert np.all(values[:, 3] == 0)
    assert np.all(values[:, 2] <= energy - 0.0136)
    table = workspace.physical.bound_compton[0]
    expected = (
        quad(
            lambda c: c * impulse_angular_weight(energy, c, table),
            -1.0,
            1.0,
            epsabs=1e-100,
            epsrel=1e-10,
        )[0]
        / quad(
            lambda c: impulse_angular_weight(energy, c, table),
            -1.0,
            1.0,
            epsabs=1e-100,
            epsrel=1e-10,
        )[0]
    )
    assert abs(values[:, 0].mean() - expected) < 7 * values[:, 0].std(ddof=1) / math.sqrt(4096)


@pytest.mark.parametrize("energy", [5.0, 80.0, 5000.0])
def test_native_angle_distribution_and_phase_normalisation(warp_device: str, energy: float) -> None:
    count = 16384
    workspace = _workspace(warp_device, energy, count)
    out = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _sample_bound,
        dim=count,
        device=warp_device,
        inputs=[energy, workspace._parameters.physics, out],
    )
    samples = out.numpy()
    assert np.all(samples[:, 3] == 0)
    assert np.all((samples[:, 2] > 0) & (samples[:, 2] < energy))
    table = workspace.physical.bound_compton[0]
    edges = np.linspace(-1, 1, 13)
    masses = np.array(
        [
            quad(lambda c: impulse_angular_weight(energy, c, table), a, b, epsabs=1e-11)[0]
            for a, b in pairwise(edges)
        ]
    )
    probabilities = masses / masses.sum()
    counts = np.histogram(samples[:, 0], edges)[0]
    lower = binom.ppf(1e-6 / 24, count, probabilities)
    upper = binom.ppf(1 - 1e-6 / 24, count, probabilities)
    assert np.all((counts >= lower) & (counts <= upper))
    cosines = np.array([-0.9, 0.0, 0.8, 0.99])
    pdf = wp.empty(4, dtype=wp.float64, device=warp_device)
    wp.launch(
        _pdf,
        dim=4,
        device=warp_device,
        inputs=[
            energy,
            workspace._parameters.physics,
            wp.array(cosines, dtype=wp.float64, device=warp_device),
            pdf,
        ],
    )
    expected = np.array(
        [impulse_angular_weight(energy, c, table) / (2 * math.pi * masses.sum()) for c in cosines]
    )
    np.testing.assert_allclose(pdf.numpy(), expected, rtol=1e-7, atol=1e-13)


def test_native_conditional_shell_energy_distribution_and_momentum_conservation(
    warp_device: str,
) -> None:
    from dpt.transport.bound_compton import shell_accessibility

    energy, cosine, count = 80.0, 0.3, 16384
    workspace = _workspace(warp_device, energy, count)
    out = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _fixed_angle,
        dim=count,
        device=warp_device,
        inputs=[energy, cosine, workspace._parameters.physics, out],
    )
    values = out.numpy()
    assert np.all(values[:, 3] == 0)
    assert np.all(values[:, 0] + values[:, 1] <= energy)
    transfer = np.sqrt(energy**2 + values[:, 0] ** 2 - 2 * energy * values[:, 0] * cosine)
    residual = values[:, 0] - energy + energy * values[:, 0] * (1 - cosine) / 510.99895069
    np.testing.assert_allclose(residual, values[:, 2] * transfer, rtol=3e-10, atol=3e-13)
    shells = workspace.physical.bound_compton[0].shells
    weights = np.array([s.occupancy * shell_accessibility(energy, cosine, s) for s in shells])
    probabilities = weights / weights.sum()
    counts = np.array([np.count_nonzero(values[:, 1] == s.binding_kev) for s in shells])
    assert np.all(counts >= binom.ppf(1e-6 / 12, count, probabilities))
    assert np.all(counts <= binom.ppf(1 - 1e-6 / 12, count, probabilities))
    stationary = energy / (1 + energy / 510.99895069 * (1 - cosine))
    assert values[:, 0].std() > 0.1
    assert np.any(values[:, 0] < stationary) and np.any(values[:, 0] > stationary)


def _source(device: str, count: int) -> tuple[Any, Any, Any, Any]:
    return (
        wp.array(np.tile((0.0, 0.0, -1.0), (count, 1)), dtype=wp.vec3d, device=device),
        wp.array(np.tile((0.0, 0.0, 1.0), (count, 1)), dtype=wp.vec3d, device=device),
        wp.ones(count, dtype=wp.float64, device=device),
        wp.ones(1, dtype=wp.float64, device=device),
    )


def _outputs(device: str, count: int) -> dict[str, Any]:
    return {
        "out_" + name: wp.empty(count, dtype=dtype, device=device)
        for name, dtype in (
            ("pixel", wp.int32),
            ("score", wp.float64),
            ("energy", wp.float64),
            ("events", wp.int32),
            ("status", wp.int32),
        )
    }


def test_production_energy_balance_and_density_score_normalisation(warp_device: str) -> None:
    count = 16384
    workspace = _workspace(warp_device, count=count)
    source, outputs = _source(warp_device, count), _outputs(warp_device, count)
    extras = {
        "out_" + name: wp.empty(count, dtype=dtype, device=warp_device)
        for name, dtype in (
            ("initial_energy", wp.float64),
            ("deposited_energy", wp.float64),
            ("spectrum_index", wp.int32),
        )
    }
    local = wp.empty(1, dtype=wp.float64, device=warp_device)
    trace_energy_balance(
        *source,
        batch=HistoryBatch(99121, 0, count),
        workspace=workspace,
        **outputs,
        **extras,
        out_local_deposition=local,
    )
    np.testing.assert_allclose(
        extras["out_initial_energy"].numpy(),
        outputs["out_energy"].numpy() + extras["out_deposited_energy"].numpy(),
        atol=2e-13,
        rtol=0,
    )
    assert extras["out_deposited_energy"].numpy().max() > 1
    derivative = wp.empty(count, dtype=wp.float64, device=warp_device)
    status = wp.empty(count, dtype=wp.int32, device=warp_device)
    wp.launch(
        _normalised_density_score,
        dim=count,
        device=warp_device,
        inputs=[*source, *workspace._model_inputs(), derivative, status],
    )
    values = derivative.numpy()
    assert np.all(status.numpy() == 0)
    # The complete conservative path law has total probability one. Its
    # independent score normalisation identity remains zero with Doppler energy.
    assert abs(values.mean()) < 7 * values.std(ddof=1) / math.sqrt(count)


def test_randomised_bound_history_replay_is_identical_across_chunks(warp_device: str) -> None:
    count = 1024
    workspace = _workspace(warp_device, count=count, max_trials=32, max_events=8)
    design = SamplingDesign("scrambled-sobol", 8, 32, layout="bound-compton-v3")
    batch = prepare_transport_sampling(
        workspace, batch=HistoryBatch(141231, 0, count), design=design
    )
    source = _source(warp_device, count)
    outputs = _outputs(warp_device, count)
    trace_histories(*source, batch=batch, workspace=workspace, **outputs)
    reference = {name: value.numpy().copy() for name, value in outputs.items()}
    # Same actual source rows and original addresses, separated launch ownership.
    for start in (0, 512):
        shard = batch.shard(2, start // 512).batch
        local_source = [a[start : start + 512] for a in source[:3]] + [source[3]]
        local_outputs = {name: value[start : start + 512] for name, value in outputs.items()}
        trace_histories(*local_source, batch=shard, workspace=workspace, **local_outputs)
    for name, value in outputs.items():
        np.testing.assert_array_equal(value.numpy(), reference[name])


def test_bound_next_event_matches_direct_detection_and_coefficient_gradient(
    warp_device: str,
) -> None:
    from dataclasses import replace

    count = 16384
    workspace = _workspace(warp_device, count=count)
    # A finite 20 mm detector supplies a well-conditioned area proposal while
    # retaining real escape misses and intervening attenuation.
    spec = replace(
        workspace.spec, detector=PlanarDetector((-10.0, -10.0), (20.0, 20.0), (1, 1), 2.0)
    )
    workspace = prepare_transport(
        spec,
        material_ids=workspace.material_ids,
        absorption=workspace.absorption,
        scattering=workspace.scattering,
        max_histories=count,
        physical=workspace.physical,
        device=warp_device,
    )
    source = _source(warp_device, count)
    batch = HistoryBatch(9981701, 0, count)
    out_mean = wp.empty(1, dtype=wp.float64, device=warp_device)
    out_variance = wp.empty(1, dtype=wp.float64, device=warp_device)
    direct = prepare_transport_readout(workspace, spec=TransportReadout(mode="scatter"))
    trace_readout(
        *source, batch=batch, workspace=direct, out_mean=out_mean, out_variance_of_mean=out_variance
    )
    direct_mean, direct_variance = out_mean.numpy()[0], out_variance.numpy()[0]
    next_event = prepare_transport_readout(
        workspace,
        spec=TransportReadout(mode="scatter", next_event=True),
        parameter=CoefficientParameter("scattering", 0, 1),
    )
    out_derivative = wp.empty(1, dtype=wp.float64, device=warp_device)
    trace_readout(
        *source,
        batch=batch,
        workspace=next_event,
        out_mean=out_mean,
        out_variance_of_mean=out_variance,
        out_derivative_mean=out_derivative,
    )
    next_mean, next_variance = out_mean.numpy()[0], out_variance.numpy()[0]
    assert direct_mean > 0 and next_mean > 0
    assert abs(direct_mean - next_mean) < 7 * math.sqrt(direct_variance + next_variance)
    reference_gradient = next_event._derivatives.numpy().copy()
    step = 0.02
    means = []
    per_history = []
    for sign in (-1, 1):
        rates = wp.array([0.3, 0.3 * math.exp(sign * step)], dtype=wp.float64, device=warp_device)
        candidate = prepare_transport(
            spec,
            material_ids=workspace.material_ids,
            absorption=workspace.absorption,
            scattering=rates,
            max_histories=count,
            physical=workspace.physical,
            device=warp_device,
        )
        readout = prepare_transport_readout(
            candidate, spec=TransportReadout(mode="scatter", next_event=True)
        )
        trace_readout(*source, batch=batch, workspace=readout, out_mean=out_mean)
        means.append(out_mean.numpy()[0])
        per_history.append(readout._scores.numpy().copy())
    difference = (per_history[1] - per_history[0]) / (2 * step) - reference_gradient
    assert abs(difference.mean()) < 7 * difference.std(ddof=1) / math.sqrt(count) + step**2


def test_native_incident_score_marks_exact_shell_threshold(warp_device: str) -> None:
    table = xraylib_bound_compton(13)
    energy = table.shells[0].binding_kev
    workspace = _workspace(warp_device, energy, count=1)
    states = wp.array([(0.0, 0.97 * energy)], dtype=wp.vec2d, device=warp_device)
    out = wp.empty(1, dtype=wp.vec3d, device=warp_device)
    wp.launch(
        _joint_scores,
        dim=1,
        device=warp_device,
        inputs=[energy, workspace._parameters.physics, states, out],
    )
    scores = out.numpy()[0]
    assert np.isfinite(scores[:2]).all()
    assert np.isnan(scores[2])


@pytest.mark.parametrize("energy", [1.0, 80.0, 5000.0])
def test_joint_geometry_scores_include_incident_energy_and_normaliser(
    warp_device: str, energy: float
) -> None:
    from dpt.transport.bound_compton import bound_compton_joint_pdf, bound_compton_joint_scores

    workspace = _workspace(warp_device, energy, count=4)
    table = workspace.physical.bound_compton[0]

    def normalisation(e: float) -> float:
        # Direct-c adaptivity misses the narrow forward-angle shell transition
        # at MeV energies. Resolve it in sin(theta/2), split at each shell's
        # zero-momentum threshold, and include the exact transformed Jacobian.
        thresholds = (
            math.sqrt(shell.binding_kev * 510.99895069 / (2 * e * (e - shell.binding_kev)))
            for shell in table.shells
            if e > shell.binding_kev
        )
        points = sorted({0.0, 1.0, *(point for point in thresholds if 0 < point < 1)})
        return (
            2
            * math.pi
            * sum(
                quad(
                    lambda s: 4 * s * impulse_angular_weight(e, 1 - 2 * s * s, table),
                    a,
                    b,
                    epsabs=1e-14,
                    epsrel=1e-12,
                    limit=300,
                )[0]
                for a, b in pairwise(points)
            )
        )

    step = energy * 1e-5
    n = normalisation(energy)
    n_score = (
        math.log(normalisation(energy - 2 * step))
        - 8 * math.log(normalisation(energy - step))
        + 8 * math.log(normalisation(energy + step))
        - math.log(normalisation(energy + 2 * step))
    ) / (12 * step)
    cosines = np.array([-0.7, 0.0, 0.5, 0.8])
    outgoing = energy / (1 + energy / 510.99895069 * (1 - cosines)) * 0.97
    states = wp.array(np.column_stack((cosines, outgoing)), dtype=wp.vec2d, device=warp_device)
    out = wp.empty(4, dtype=wp.vec3d, device=warp_device)
    wp.launch(
        _joint_scores,
        dim=4,
        device=warp_device,
        inputs=[energy, workspace._parameters.physics, states, out],
    )
    expected = np.array(
        [
            bound_compton_joint_scores(energy, c, e, table, normalisation_energy_score=n_score)
            for c, e in zip(cosines, outgoing, strict=True)
        ]
    )
    np.testing.assert_allclose(out.numpy(), expected, rtol=2e-7, atol=2e-12)
    # Independently perturb the complete normalised joint density at fixed
    # outgoing energy, including the changing shell limit and angular integral.
    for index, (cosine, emitted) in enumerate(zip(cosines, outgoing, strict=True)):
        logs = []
        for offset in (-2, -1, 1, 2):
            incident = energy + offset * step
            logs.append(
                math.log(
                    bound_compton_joint_pdf(
                        incident,
                        cosine,
                        emitted,
                        table,
                        solid_angle_normalisation=normalisation(incident),
                    )
                )
            )
        finite = (logs[0] - 8 * logs[1] + 8 * logs[2] - logs[3]) / (12 * step)
        assert out.numpy()[index, 2] == pytest.approx(finite, rel=2e-7, abs=2e-11)
    assert math.isfinite(n) and n > 0
