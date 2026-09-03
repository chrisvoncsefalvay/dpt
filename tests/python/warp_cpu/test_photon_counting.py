"""Independent event, convolution and renewal references for photon counting."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from decimal import Decimal, localcontext
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from scipy.stats import poisson

from dpt._runtime import load_warp
from dpt.contracts import ContractError, NumericalError
from dpt.materials import Provenance
from dpt.photon_counting import (
    ChargeCloudPattern,
    ChargeSharingIdentity,
    ChargeSharingSpec,
    RectangularPileupSpec,
    charge_sharing_mean,
    charge_sharing_vjp,
    prepare_charge_sharing,
    prepare_rectangular_pileup,
    rectangular_pileup_mean,
    rectangular_pileup_readout,
    rectangular_pileup_vjp,
    sample_charge_sharing,
)

wp = load_warp()


def provenance() -> Provenance:
    law = b"analytic test law: discrete supplied charge clouds and independent lattice marks"
    return Provenance(
        "analytic fixture defined by the test equations",
        hashlib.sha256(law).hexdigest(),
        "Apache-2.0",
        law.decode(),
    )


def cloud_spec() -> ChargeSharingSpec:
    return ChargeSharingSpec(
        (2, 3),
        (10.0, 20.0),
        (
            ChargeCloudPattern(((0, 0),), (1.0,)),
            ChargeCloudPattern(((0, 0), (0, 1)), (0.5, 0.5)),
        ),
        (0.3, 0.7, 0.8, 0.2),
        (3.0, 7.0, 12.0, 19.0),
        provenance(),
    )


def pileup_spec() -> RectangularPileupSpec:
    return RectangularPileupSpec(2.0, (1, 3), (0.6, 0.4), (2, 4, 7, 11), provenance())


def array(values: Any, device: str, dtype: Any = wp.float64) -> Any:
    return wp.array(values, dtype=dtype, device=device)


def cloud_reference(
    spec: ChargeSharingSpec, incident: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    counts = np.zeros((spec.bins, spec.pixels))
    ledger = np.zeros((4, spec.pixels))
    for e, energy in enumerate(spec.energies_kev):
        for p in range(spec.pixels):
            row, col = divmod(p, spec.detector_shape[1])
            for m, pattern in enumerate(spec.patterns):
                weight = incident[e, p] * spec.pattern_probabilities[e * len(spec.patterns) + m]
                for offset, fraction in zip(pattern.offsets, pattern.fractions, strict=True):
                    rr, cc = row + offset[0], col + offset[1]
                    pulse = energy * fraction
                    if not (0 <= rr < spec.detector_shape[0] and 0 <= cc < spec.detector_shape[1]):
                        ledger[3, p] += weight * pulse
                    elif pulse < spec.bin_edges_kev[0]:
                        ledger[1, p] += weight * pulse
                    elif pulse >= spec.bin_edges_kev[-1]:
                        ledger[2, p] += weight * pulse
                    elif pulse > 0:
                        b = int(np.searchsorted(spec.bin_edges_kev, pulse, side="right")) - 1
                        counts[b, rr * spec.detector_shape[1] + cc] += weight
                        ledger[0, p] += weight * pulse
    return counts, ledger


def mean_cloud(
    spec: ChargeSharingSpec, incident: np.ndarray[Any, Any], device: str
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], Any]:
    work = prepare_charge_sharing(spec, max_events=1, device=device)
    counts = wp.empty((spec.bins, spec.pixels), dtype=wp.float64, device=device)
    ledger = wp.empty((4, spec.pixels), dtype=wp.float64, device=device)
    charge_sharing_mean(
        array(incident, device),
        out_counts=counts,
        out_accounting_kev=ledger,
        workspace=work,
    )
    return counts.numpy(), ledger.numpy(), work


def mean_pileup(
    spec: RectangularPileupSpec,
    rate: np.ndarray[Any, Any],
    gate: np.ndarray[Any, Any],
    blind: np.ndarray[Any, Any],
    device: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], Any]:
    work = prepare_rectangular_pileup(spec, max_pixels=len(rate), device=device)
    counts = wp.empty((spec.bins, len(rate)), dtype=wp.float64, device=device)
    ledger = wp.empty((6, len(rate)), dtype=wp.float64, device=device)
    rectangular_pileup_mean(
        array(rate, device),
        array(gate, device),
        array(blind, device),
        out_counts_s=counts,
        out_accounting=ledger,
        workspace=work,
    )
    return counts.numpy(), ledger.numpy(), work


def pileup_reference(
    spec: RectangularPileupSpec, rate: float, gate: float, blind: float
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Enumerate mark convolutions by event count, independently of Panjer recursion.

    Positive marks bound the number of terms needed below the finite readout.
    All larger event counts are in overflow, whose mass and first moment are
    retained by the analytic infinite Poisson tail and total energy moment.
    """
    top = spec.bin_edges_steps[-1]
    mark = np.zeros(top)
    for step, probability in zip(spec.mark_steps, spec.mark_probabilities, strict=True):
        if step < top:
            mark[step] = probability
    pulse = mark.copy()
    distribution = np.zeros(top)
    for additional in range((top - 1) // spec.mark_steps[0]):
        distribution += poisson.pmf(additional, rate * gate) * pulse
        pulse = np.convolve(pulse, mark)[:top]
    h = rate / (1 + rate * (gate + blind))
    counts = np.array([h * distribution[lo:hi].sum() for lo, hi in pairwise(spec.bin_edges_steps)])
    energies = np.arange(top) * spec.energy_quantum_kev
    below = spec.bin_edges_steps[0]
    accepted_energy = h * np.dot(distribution[below:], energies[below:])
    below_energy = h * np.dot(distribution[:below], energies[:below])
    total_pulse_energy = h * (1 + rate * gate) * spec.mean_mark_kev
    ledger = np.array(
        [
            h * distribution[:below].sum(),
            h * (1 - distribution.sum()),
            below_energy,
            total_pulse_energy - accepted_energy - below_energy,
            accepted_energy,
            h * rate * blind * spec.mean_mark_kev,
        ]
    )
    return counts, ledger


def test_cloud_mean_has_bin_migration_edge_escape_and_complete_energy(
    warp_device: str,
) -> None:
    spec = cloud_spec()
    incident = np.arange(1, 13, dtype=float).reshape(2, 6)
    counts, ledger, work = mean_cloud(spec, incident, warp_device)
    ref_counts, ref_ledger = cloud_reference(spec, incident)
    np.testing.assert_allclose(counts, ref_counts, rtol=3e-15, atol=2e-14)
    np.testing.assert_allclose(ledger, ref_ledger, rtol=3e-15, atol=2e-14)
    np.testing.assert_allclose(
        ledger.sum(axis=0), np.array(spec.energies_kev) @ incident, rtol=3e-15
    )
    assert np.all(ledger[3, [2, 5]] > 0)
    assert np.all(ledger[2] > 0)
    assert work.scratch_bytes + work.host_bytes < 4096


def test_cloud_mean_threshold_loss_and_zero_fraction_do_not_count(
    warp_device: str,
) -> None:
    spec = replace(
        cloud_spec(),
        energies_kev=(10.0,),
        patterns=(ChargeCloudPattern(((0, 0), (0, 1), (1, 0)), (0.2, 0.8, 0.0)),),
        pattern_probabilities=(1.0,),
        bin_edges_kev=(3.0, 9.0),
    )
    counts, ledger, _ = mean_cloud(spec, np.ones((1, 6)), warp_device)
    assert counts.sum() == 4
    np.testing.assert_array_equal(ledger[1], np.full(6, 2.0))
    np.testing.assert_allclose(ledger.sum(axis=0), 10.0)


def test_cloud_energy_products_rescue_finite_result_from_intermediate_range_loss(
    warp_device: str,
) -> None:
    spec = ChargeSharingSpec(
        (1, 1),
        (1e300,),
        (ChargeCloudPattern(((0, 1),), (1.0,)), ChargeCloudPattern(((0, 0),), (1.0,))),
        (1e-300, 1.0),
        (0.0, 1e301),
        provenance(),
    )
    _counts, ledger, work = mean_cloud(spec, np.array([[1e-300]]), warp_device)
    assert ledger[3, 0] == pytest.approx(1e-300, rel=3e-15, abs=0)
    assert ledger[0, 0] == pytest.approx(1.0, rel=3e-15)
    output = wp.empty((1, 1), dtype=wp.float64, device=warp_device)
    ledger_seed = np.zeros((4, 1))
    ledger_seed[3, 0] = 1e100
    charge_sharing_vjp(
        array(np.zeros((1, 1)), warp_device),
        array(ledger_seed, warp_device),
        out_incident_photons=output,
        workspace=work,
    )
    assert output.numpy()[0, 0] == pytest.approx(1e100, rel=3e-15)


def test_cloud_vjp_is_independent_dense_transpose_with_signed_ledger_seeds(
    warp_device: str,
) -> None:
    spec = cloud_spec()
    rng = np.random.default_rng(19)
    seed = rng.normal(size=(spec.bins, spec.pixels))
    energy_seed = rng.normal(size=(4, spec.pixels))
    work = prepare_charge_sharing(spec, max_events=1, device=warp_device)
    gradient = wp.empty((spec.energies, spec.pixels), dtype=wp.float64, device=warp_device)
    charge_sharing_vjp(
        array(seed, warp_device),
        array(energy_seed, warp_device),
        out_incident_photons=gradient,
        workspace=work,
    )
    expected = np.zeros((spec.energies, spec.pixels))
    for e in range(spec.energies):
        for p in range(spec.pixels):
            basis = np.zeros_like(expected)
            basis[e, p] = 1
            counts, ledger = cloud_reference(spec, basis)
            expected[e, p] = np.sum(counts * seed) + np.sum(ledger * energy_seed)
    np.testing.assert_allclose(gradient.numpy(), expected, rtol=4e-15, atol=3e-14)


def sampled_clouds(
    spec: ChargeSharingSpec,
    n: int,
    identity: ChargeSharingIdentity,
    device: str,
    pixel: int = 1,
) -> dict[str, Any]:
    work = prepare_charge_sharing(spec, max_events=n, device=device)
    outputs: dict[str, Any] = {
        "out_pixels": wp.empty((n, spec.fragments), dtype=wp.int32, device=device),
        "out_times_s": wp.empty((n, spec.fragments), dtype=wp.float64, device=device),
        "out_energies_kev": wp.empty((n, spec.fragments), dtype=wp.float64, device=device),
        "out_bins": wp.empty((n, spec.fragments), dtype=wp.int32, device=device),
        "out_patterns": wp.empty(n, dtype=wp.int32, device=device),
        "out_accounting_kev": wp.empty((n, 4), dtype=wp.float64, device=device),
    }
    times = np.linspace(0, 1, n)
    sample_charge_sharing(
        array(np.full(n, pixel), device, wp.int32),
        array(np.zeros(n, dtype=np.int32), device, wp.int32),
        array(times, device),
        identity=identity,
        workspace=work,
        **outputs,
    )
    return {key: value.numpy() for key, value in outputs.items()}


def test_cloud_sampler_preserves_true_energy_coincidence_and_count_correlation(
    warp_device: str,
) -> None:
    n = 40000
    spec = cloud_spec()
    sampled = sampled_clouds(spec, n, ChargeSharingIdentity(712, 31), warp_device)
    split = sampled["out_patterns"] == 1
    assert abs(split.mean() - 0.7) < 6 * np.sqrt(0.7 * 0.3 / n)
    np.testing.assert_array_equal(sampled["out_accounting_kev"].sum(axis=1), np.full(n, 10.0))
    np.testing.assert_array_equal(sampled["out_energies_kev"].sum(axis=1), np.full(n, 10.0))
    np.testing.assert_array_equal(sampled["out_times_s"][:, 0], sampled["out_times_s"][:, 1])
    central_low = sampled["out_bins"][:, 0] == 0
    neighbour_low = sampled["out_bins"][:, 1] == 0
    np.testing.assert_array_equal(central_low, neighbour_low)
    assert np.cov(central_low, neighbour_low, ddof=0)[0, 1] == pytest.approx(0.21, abs=0.006)
    reference = np.random.default_rng(91).choice([0, 1], size=n, p=[0.3, 0.7])
    assert abs(split.sum() - reference.sum()) < 7 * np.sqrt(2 * n * 0.21)


def test_cloud_counter_replay_split_extent_and_edge_escape(warp_device: str) -> None:
    spec = cloud_spec()
    full = sampled_clouds(spec, 19, ChargeSharingIdentity(73, 5), warp_device, pixel=2)
    replay = sampled_clouds(spec, 19, ChargeSharingIdentity(73, 5), warp_device, pixel=2)
    split = sampled_clouds(spec, 7, ChargeSharingIdentity(73, 5, 12), warp_device, pixel=2)
    np.testing.assert_array_equal(full["out_patterns"], replay["out_patterns"])
    np.testing.assert_array_equal(full["out_patterns"][12:], split["out_patterns"])
    chosen = full["out_patterns"] == 1
    np.testing.assert_array_equal(full["out_accounting_kev"][chosen, 3], np.full(chosen.sum(), 5.0))
    np.testing.assert_array_equal(full["out_pixels"][chosen, 1], np.full(chosen.sum(), -1))
    with pytest.raises(ContractError, match="wrap"):
        ChargeSharingIdentity(1, 2, 2**32 - 2).validate_extent(3)


@pytest.mark.parametrize(
    "rate,gate,blind",
    [
        (0.0, 0.1, 0.2),
        (1e-12, 0.1, 0.2),
        (0.3, 0.2, 0.7),
        (3.0, 0.4, 0.2),
        (100.0, 0.5, 0.1),
        (1e12, 0.5, 0.1),
    ],
)
def test_pileup_mean_matches_independent_poisson_mark_convolutions(
    rate: float, gate: float, blind: float, warp_device: str
) -> None:
    spec = pileup_spec()
    counts, ledger, _ = mean_pileup(
        spec, np.array([rate]), np.array([gate]), np.array([blind]), warp_device
    )
    ref_counts, ref_ledger = pileup_reference(spec, rate, gate, blind)
    np.testing.assert_allclose(counts[:, 0], ref_counts, rtol=2e-13, atol=1e-25)
    np.testing.assert_allclose(ledger[:, 0], ref_ledger, rtol=2e-13, atol=2e-15 * max(1, rate))
    h = rate / (1 + rate * (gate + blind))
    assert counts.sum() + ledger[:2].sum() == pytest.approx(h, rel=3e-15, abs=1e-25)
    assert ledger[2:].sum() == pytest.approx(rate * spec.mean_mark_kev, rel=3e-15, abs=1e-25)


def test_pileup_log_recursion_rescues_poisson_mode_after_zero_term_underflows(
    warp_device: str,
) -> None:
    spec = RectangularPileupSpec(1.0, (1,), (1.0,), (1, 790, 810, 901), provenance())
    counts, ledger, _ = mean_pileup(
        spec, np.array([800.0]), np.array([1.0]), np.array([0.0]), warp_device
    )
    h = 800 / 801
    expected = [
        h * (poisson.cdf(hi - 2, 800) - poisson.cdf(lo - 2, 800))
        for lo, hi in pairwise(spec.bin_edges_steps)
    ]
    np.testing.assert_allclose(counts[:, 0], expected, rtol=2e-11, atol=1e-13)
    assert counts[1, 0] > 0.25
    assert ledger[1, 0] == pytest.approx(h * poisson.sf(899, 800), rel=2e-8)
    assert ledger[2:].sum() == pytest.approx(800.0, rel=3e-15)


def test_nonparalysable_deadtime_and_rate_limits(warp_device: str) -> None:
    spec = RectangularPileupSpec(1.0, (1,), (1.0,), (0, 2), provenance())
    rates = np.array([0.0, 1e-12, 1.0, 1e12])
    counts, ledger, _ = mean_pileup(spec, rates, np.zeros(4), np.full(4, 0.2), warp_device)
    np.testing.assert_allclose(counts[0], rates / (1 + rates * 0.2), rtol=3e-15)
    assert counts[0, 1] / rates[1] == pytest.approx(1.0, abs=3e-13)
    assert counts[0, -1] == pytest.approx(5.0, rel=6e-12)
    assert counts[0, -1] > rates[-1] * np.exp(-rates[-1] * 0.2)
    np.testing.assert_array_equal(ledger[1], np.zeros(4))
    np.testing.assert_allclose(ledger[2:].sum(axis=0), rates, rtol=3e-15)


def test_pileup_parameter_vjp_matches_independent_reference_finite_differences(
    warp_device: str,
) -> None:
    spec = pileup_spec()
    rate, gate, blind = (
        np.array([0.7, 4.0]),
        np.array([0.11, 0.4]),
        np.array([0.2, 0.07]),
    )
    seed = np.array([[0.7, -0.3], [-0.9, 0.8], [0.5, 1.1]])
    energy_seed = np.array(
        [
            [0.3, -0.4],
            [0.7, 0.1],
            [-0.02, 0.05],
            [0.12, -0.13],
            [-0.05, 0.03],
            [0.07, -0.02],
        ]
    )
    work = prepare_rectangular_pileup(spec, max_pixels=2, device=warp_device)
    grads = [wp.empty(2, dtype=wp.float64, device=warp_device) for _ in range(3)]
    rectangular_pileup_vjp(
        array(rate, warp_device),
        array(gate, warp_device),
        array(blind, warp_device),
        array(seed, warp_device),
        array(energy_seed, warp_device),
        out_rate_s=grads[0],
        out_gate_s=grads[1],
        out_blind_s=grads[2],
        workspace=work,
    )
    for k, values in enumerate((rate, gate, blind)):
        expected = np.zeros(2)
        for p in range(2):
            step = 2e-5 * values[p]
            parameters = [rate[p], gate[p], blind[p]]
            parameters[k] += step
            plus_counts, plus_ledger = pileup_reference(spec, *parameters)
            parameters[k] -= 2 * step
            minus_counts, minus_ledger = pileup_reference(spec, *parameters)
            expected[p] = (
                np.dot(plus_counts - minus_counts, seed[:, p])
                + np.dot(plus_ledger - minus_ledger, energy_seed[:, p])
            ) / (2 * step)
        np.testing.assert_allclose(grads[k].numpy(), expected, rtol=3e-8, atol=4e-10)


def test_pileup_zero_rate_derivative_and_energy_balance_derivative(
    warp_device: str,
) -> None:
    spec = pileup_spec()
    rates = np.array([0.0, 1e-12, 3.0, 1e6])
    work = prepare_rectangular_pileup(spec, max_pixels=4, device=warp_device)
    seed = np.zeros((spec.bins, 4))
    ledger_seed = np.zeros((6, 4))
    ledger_seed[2:] = 1
    grads = [wp.empty(4, dtype=wp.float64, device=warp_device) for _ in range(3)]
    rectangular_pileup_vjp(
        array(rates, warp_device),
        array(np.full(4, 0.2), warp_device),
        array(np.full(4, 0.3), warp_device),
        array(seed, warp_device),
        array(ledger_seed, warp_device),
        out_rate_s=grads[0],
        out_gate_s=grads[1],
        out_blind_s=grads[2],
        workspace=work,
    )
    np.testing.assert_allclose(grads[0].numpy(), spec.mean_mark_kev, rtol=3e-15)
    np.testing.assert_allclose(grads[1].numpy(), 0.0, atol=3e-9)
    np.testing.assert_allclose(grads[2].numpy(), 0.0, atol=3e-9)


def readout(
    spec: RectangularPileupSpec,
    times: np.ndarray[Any, Any],
    energies: np.ndarray[Any, Any],
    gate: float,
    blind: float,
    device: str,
) -> dict[str, Any]:
    pixels, capacity = times.shape
    work = prepare_rectangular_pileup(
        spec, max_pixels=pixels, max_events_per_pixel=capacity, device=device
    )
    outputs: dict[str, Any] = {
        "out_times_s": wp.empty((pixels, capacity), dtype=wp.float64, device=device),
        "out_energies_kev": wp.empty((pixels, capacity), dtype=wp.float64, device=device),
        "out_pulse_counts": wp.empty(pixels, dtype=wp.int32, device=device),
        "out_bin_counts": wp.empty((spec.bins, pixels), dtype=wp.uint64, device=device),
        "out_accounting_kev": wp.empty((4, pixels), dtype=wp.float64, device=device),
    }
    rectangular_pileup_readout(
        array(times, device),
        array(energies, device),
        array(np.full(pixels, capacity), device, wp.int32),
        array(np.full(pixels, gate), device),
        array(np.full(pixels, blind), device),
        workspace=work,
        **outputs,
    )
    return {key: value.numpy() for key, value in outputs.items()}


def test_actual_event_readout_has_half_open_gate_blind_boundaries_and_overflow(
    warp_device: str,
) -> None:
    spec = RectangularPileupSpec(1.0, (1,), (1.0,), (3, 8, 12), provenance())
    times = np.array([[0.0, 0.0, 0.099, 0.1, 0.199, 0.2, 0.299, 0.4]])
    energies = np.array([[2.0, 3.0, 9.0, 100.0, 200.0, 1.0, 1.0, 4.0]])
    result = readout(spec, times, energies, 0.1, 0.1, warp_device)
    assert result["out_pulse_counts"][0] == 3
    np.testing.assert_array_equal(result["out_times_s"][0, :3], [0.0, 0.2, 0.4])
    np.testing.assert_array_equal(result["out_energies_kev"][0, :3], [14.0, 2.0, 4.0])
    np.testing.assert_array_equal(result["out_bin_counts"][:, 0], [1, 0])
    np.testing.assert_array_equal(result["out_accounting_kev"][:, 0], [4.0, 2.0, 14.0, 300.0])
    assert result["out_accounting_kev"].sum() == energies.sum()
    assert np.all(result["out_energies_kev"][0, 3:] == 0)


def test_zero_gate_and_blind_reads_every_coincident_event(warp_device: str) -> None:
    spec = RectangularPileupSpec(1.0, (1,), (1.0,), (0, 100), provenance())
    result = readout(
        spec,
        np.array([[0.0, 0.0, 0.1]]),
        np.array([[2.0, 3.0, 4.0]]),
        0.0,
        0.0,
        warp_device,
    )
    assert result["out_pulse_counts"][0] == 3
    np.testing.assert_array_equal(result["out_energies_kev"][0], [2.0, 3.0, 4.0])


def test_supplied_event_example_integrates_weak_shared_pulses_before_thresholding(
    warp_device: str,
) -> None:
    from dpt.examples.photon_counting import read_shared_events

    charge = ChargeSharingSpec(
        (1, 3),
        (10.0,),
        (ChargeCloudPattern(((0, 0), (0, 1)), (0.5, 0.5)),),
        (1.0,),
        (6.0, 11.0, 20.0),
        provenance(),
    )
    readout_spec = RectangularPileupSpec(1.0, (5,), (1.0,), (6, 11, 20), provenance())
    result = read_shared_events(
        charge,
        readout_spec,
        np.array([1, 1, 2], dtype=np.int32),
        np.zeros(3, dtype=np.int32),
        np.array([0.0, 0.01, 0.2]),
        np.full(3, 0.1),
        np.full(3, 0.05),
        identity=ChargeSharingIdentity(91, 12),
        device=warp_device,
    )
    assert result.bin_counts.sum() == 2
    assert result.readout_accounting_kev[0].sum() == 20
    assert result.readout_accounting_kev[1].sum() == 5
    assert result.charge_accounting_kev[:, 3].sum() == 5
    assert result.readout_accounting_kev.sum() + result.charge_accounting_kev[:, 3].sum() == 30


def test_poisson_event_stream_readout_matches_stationary_cluster_law(
    warp_device: str,
) -> None:
    spec = pileup_spec()
    rng = np.random.default_rng(441)
    rate, gate, blind, duration = 3.0, 0.25, 0.1, 20000.0
    times = np.cumsum(rng.exponential(1 / rate, size=int(1.1 * rate * duration)))
    times = times[times < duration]
    marks = (
        rng.choice(spec.mark_steps, size=len(times), p=spec.mark_probabilities)
        * spec.energy_quantum_kev
    )
    result = readout(spec, times[None, :], marks[None, :], gate, blind, warp_device)
    expected_counts, expected_ledger = pileup_reference(spec, rate, gate, blind)
    observed = result["out_bin_counts"][:, 0] / duration
    np.testing.assert_allclose(observed, expected_counts, atol=0.015, rtol=0)
    h = rate / (1 + rate * (gate + blind))
    assert result["out_pulse_counts"][0] / duration == pytest.approx(h, abs=0.025)
    assert result["out_accounting_kev"].sum() == pytest.approx(marks.sum(), rel=3e-15)
    expected_energy = np.array(
        [expected_ledger[4], expected_ledger[2], expected_ledger[3], expected_ledger[5]]
    )
    np.testing.assert_allclose(
        result["out_accounting_kev"][:, 0] / duration,
        expected_energy,
        atol=0.12,
        rtol=0,
    )
    # A separate cluster experiment checks the Poisson number of marks in a gate.
    additional = rng.poisson(rate * gate, size=40000)
    cluster_steps = rng.choice(spec.mark_steps, size=40000, p=spec.mark_probabilities)
    for n, extra in enumerate(additional):
        cluster_steps[n] += rng.choice(spec.mark_steps, size=extra, p=spec.mark_probabilities).sum()
    histogram = np.histogram(cluster_steps, bins=spec.bin_edges_steps)[0] / 40000 * h
    np.testing.assert_allclose(histogram, expected_counts, atol=0.02, rtol=0)


def test_invalid_inputs_leave_sticky_diagnostics_and_reject_aliases(
    warp_device: str,
) -> None:
    spec = pileup_spec()
    work = prepare_rectangular_pileup(spec, max_pixels=1, device=warp_device)
    counts = wp.empty((spec.bins, 1), dtype=wp.float64, device=warp_device)
    ledger = wp.empty((6, 1), dtype=wp.float64, device=warp_device)
    rate = array([-1.0], warp_device)
    zero = array([0.0], warp_device)
    rectangular_pileup_mean(
        rate,
        zero,
        zero,
        out_counts_s=counts,
        out_accounting=ledger,
        workspace=work,
        validate=False,
    )
    with pytest.raises(ContractError, match="fixed device input"):
        work.check_status()
    with pytest.raises(ContractError, match="fixed device input"):
        rectangular_pileup_mean(
            array([1.0], warp_device),
            zero,
            zero,
            out_counts_s=counts,
            out_accounting=ledger,
            workspace=work,
        )
    work.clear_status()
    with pytest.raises(NumericalError, match="represented"):
        rectangular_pileup_mean(
            array([1e308], warp_device),
            array([2.0], warp_device),
            zero,
            out_counts_s=counts,
            out_accounting=ledger,
            workspace=work,
        )
    work.clear_status()
    with pytest.raises(ContractError, match="overlaps"):
        rectangular_pileup_mean(
            rate,
            zero,
            zero,
            out_counts_s=ledger[: spec.bins],
            out_accounting=ledger,
            workspace=work,
        )


@pytest.mark.parametrize("defect", ["negative", "unsorted", "nan", "energy"])
def test_readout_rejects_corrupt_event_prefix(defect: str, warp_device: str) -> None:
    times, energies = np.array([[0.0, 0.1]]), np.array([[1.0, 2.0]])
    if defect == "negative":
        times[0, 0] = -0.1
    elif defect == "unsorted":
        times[0] = [0.2, 0.1]
    elif defect == "nan":
        times[0, 1] = np.nan
    else:
        energies[0, 1] = 0
    with pytest.raises(ContractError):
        readout(pileup_spec(), times, energies, 0.1, 0.2, warp_device)


@pytest.mark.parametrize("defect", ["mass", "duplicate", "offset", "edges", "provenance", "budget"])
def test_preparation_rejects_unphysical_calibration_and_budget(defect: str) -> None:
    with pytest.raises(ContractError):
        if defect == "mass":
            ChargeCloudPattern(((0, 0),), (0.8,))
        elif defect == "duplicate":
            ChargeCloudPattern(((0, 0), (0, 0)), (0.5, 0.5))
        elif defect == "offset":
            ChargeCloudPattern(((True, 0),), (1.0,))
        elif defect == "edges":
            replace(pileup_spec(), bin_edges_steps=(1, 1))
        elif defect == "provenance":
            replace(cloud_spec(), provenance=None)
        else:
            prepare_rectangular_pileup(
                pileup_spec(), max_pixels=10, device="cpu", max_owned_bytes=1
            )


def _one_mark_decimal(rate: float, gate: float, blind: float, weight: float) -> tuple[float, ...]:
    """Closed one-mark law, not the implementation's finite recurrence."""
    with localcontext() as ctx:
        ctx.prec = 90
        r, g, t, w = (Decimal.from_float(v) for v in (rate, gate, blind, weight))
        denominator = 1 + r * g + r * t
        h, probability = r / denominator, (-r * g).exp()
        count = h * probability
        dr = w * probability * (1 / denominator**2 - h * g)
        dg = -w * probability * (h**2 + h * r)
        dt = -w * probability * h**2
        return tuple(float(v) for v in (count, dr, dg, dt))


@pytest.mark.parametrize(
    ("rate", "gate", "blind", "weight"),
    [
        (1.0e200, 0.0, 0.0, 0.0),
        (1.0e200, 0.0, 0.0, 1.0e-200),
        (1.0e-308, 1.0e308, 1.0e308, 1.0e308),
        (1.0e200, 1.0e-197, 0.0, 1.0),
    ],
)
def test_pileup_range_products_and_rate_rescued_probability(
    warp_device: str, rate: float, gate: float, blind: float, weight: float
) -> None:
    spec = RectangularPileupSpec(1.0, (1,), (1.0,), (0, 2), provenance())
    counts, ledger, workspace = mean_pileup(
        spec, np.array([rate]), np.array([gate]), np.array([blind]), warp_device
    )
    reference = _one_mark_decimal(rate, gate, blind, weight)
    assert float(counts[0, 0]) == pytest.approx(reference[0], rel=3.0e-12, abs=0.0)
    assert reference[0] > 0.0
    assert np.isfinite(ledger).all()
    assert float(ledger[2:, 0].sum()) == pytest.approx(rate, rel=3.0e-15, abs=0.0)
    gradients = [wp.empty(1, dtype=wp.float64, device=warp_device) for _ in range(3)]
    rectangular_pileup_vjp(
        array([rate], warp_device),
        array([gate], warp_device),
        array([blind], warp_device),
        array([[weight]], warp_device),
        array(np.zeros((6, 1)), warp_device),
        out_rate_s=gradients[0],
        out_gate_s=gradients[1],
        out_blind_s=gradients[2],
        workspace=workspace,
    )
    for output, expected in zip(gradients, reference[1:], strict=True):
        assert float(output.numpy()[0]) == pytest.approx(expected, rel=3.0e-12, abs=0.0)
    if weight == 0.0:
        assert all(float(output.numpy()[0]) == 0.0 for output in gradients)


def test_pileup_energy_balance_vjp_cancels_before_large_products(
    warp_device: str,
) -> None:
    spec = RectangularPileupSpec(1.0, (1,), (1.0,), (0, 2), provenance())
    _, _, workspace = mean_pileup(
        spec, np.array([1.0e200]), np.array([0.0]), np.array([0.0]), warp_device
    )
    ledger_seed = np.zeros((6, 1))
    ledger_seed[2:] = 1.0
    gradients = [wp.empty(1, dtype=wp.float64, device=warp_device) for _ in range(3)]
    rectangular_pileup_vjp(
        array([1.0e200], warp_device),
        array([0.0], warp_device),
        array([0.0], warp_device),
        array([[0.0]], warp_device),
        array(ledger_seed, warp_device),
        out_rate_s=gradients[0],
        out_gate_s=gradients[1],
        out_blind_s=gradients[2],
        workspace=workspace,
    )
    assert [float(output.numpy()[0]) for output in gradients] == [1.0, 0.0, 0.0]
