"""Independent FFT, covariance and sampled shot-noise references for MTF/DQE."""

from __future__ import annotations

import importlib
import math
from dataclasses import replace
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.detector_physics import DetectorResponse, FirstInteractionModel
from dpt.detector_transfer import DetectorTransferSpec, detector_transfer, prepare_detector_transfer
from dpt.materials import MaterialTable, Provenance
from dpt.physical_data import PhotonCoefficientTable

wp: Any = importlib.import_module("warp")


def model(*, electronic: float = 1.25) -> DetectorTransferSpec:
    return DetectorTransferSpec(
        (7.0, 11.0),
        (0.4, 2.4),
        (0.8, 7.2),
        "keV",
        (3, 3),
        (0.0, 0.0625, 0.0, 0.125, 0.625, 0.0625, 0.0, 0.125, 0.0),
        (0.5, 0.8),
        electronic,
    )


def embedded_psf(spec: DetectorTransferSpec, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape)
    for row in range(spec.psf_shape[0]):
        for col in range(spec.psf_shape[1]):
            out[
                (row - spec.psf_shape[0] // 2) % shape[0], (col - spec.psf_shape[1] // 2) % shape[1]
            ] += spec.psf_weights[row * spec.psf_shape[1] + col]
    return out


def frequency_grid(spec: DetectorTransferSpec, shape: tuple[int, int]) -> np.ndarray:
    rows = np.fft.fftfreq(shape[0], d=spec.spacing_mm[1])
    columns = np.fft.fftfreq(shape[1], d=spec.spacing_mm[0])
    x, y = np.meshgrid(columns, rows)
    return np.stack([x, y], axis=-1).reshape(-1, 2)


def evaluate(
    spec: DetectorTransferSpec, frequencies: np.ndarray, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workspace = prepare_detector_transfer(spec, max_frequencies=len(frequencies), device=device)
    f = wp.array(frequencies, dtype=wp.float64, device=device)
    outputs = tuple(wp.empty(len(frequencies), dtype=wp.float64, device=device) for _ in range(3))
    original = f.numpy().copy()
    detector_transfer(
        f, out_mtf=outputs[0], out_nps=outputs[1], out_dqe=outputs[2], workspace=workspace
    )
    np.testing.assert_array_equal(f.numpy(), original)
    assert workspace.scratch_bytes == len(spec.psf_weights) * 8 + 4
    assert workspace.host_bytes == 4
    return outputs[0].numpy(), outputs[1].numpy(), outputs[2].numpy()


@pytest.mark.parametrize("read_variance", [0.0, 1.25, 100.0])
def test_transfer_matches_independent_discrete_fft(read_variance: float, warp_device: str) -> None:
    spec = model(electronic=read_variance)
    shape = (8, 16)
    frequencies = frequency_grid(spec, shape)
    mtf, nps, dqe = evaluate(spec, frequencies, warp_device)
    transform = np.fft.fft2(embedded_psf(spec, shape)).reshape(-1)
    power = abs(transform) ** 2
    population = np.sum(spec.incident_bin_photons)
    mean = np.dot(spec.incident_bin_photons, spec.first_moments)
    variance = np.dot(spec.incident_bin_photons, spec.second_moments)
    expected_nps = np.prod(spec.spacing_mm) * (variance * power + read_variance)
    expected_dqe = mean**2 * power / (population * (variance * power + read_variance))
    np.testing.assert_allclose(mtf, abs(transform) / sum(spec.psf_weights), rtol=3e-14, atol=1e-16)
    np.testing.assert_allclose(nps, expected_nps, rtol=3e-14, atol=1e-14)
    np.testing.assert_allclose(dqe, expected_dqe, rtol=3e-14, atol=1e-16)
    assert mtf[0] == pytest.approx(1.0, abs=1e-15)
    assert (dqe >= 0).all() and (dqe <= 1).all()
    assert spec.nps_unit == "(keV)^2 mm^2"


def test_dqe_has_correct_thinning_and_energy_weighting(warp_device: str) -> None:
    table = MaterialTable("analytic", (20.0, 80.0), (0.5, 0.125), "mm^-1")
    counting = DetectorResponse(
        table, table.energies_kev, 0.7, FirstInteractionModel("interaction-counting"), "above"
    )
    ideal = DetectorResponse(
        table, table.energies_kev, 0.7, FirstInteractionModel.ideal_full_energy(), "above"
    )
    populations = (10.0, 40.0)
    kwargs: Any = dict(psf_shape=(1, 1), psf_weights=(1.0,), spacing_mm=(0.5, 0.5))
    counts_spec = DetectorTransferSpec.from_response(counting, populations, **kwargs)
    energy_spec = DetectorTransferSpec.from_response(ideal, populations, **kwargs)
    frequency = np.array([[0.0, 0.0], [0.5, 0.7]])
    count_dqe = evaluate(counts_spec, frequency, warp_device)[2]
    probabilities = np.array(counting.values)
    expected_count = float(np.dot(populations, probabilities) / sum(populations))
    np.testing.assert_allclose(count_dqe, expected_count, rtol=1e-14)
    energy_dqe = evaluate(energy_spec, frequency, warp_device)[2]
    expected_energy = float(
        np.dot(populations, probabilities * [20, 80]) ** 2
        / (sum(populations) * np.dot(populations, probabilities * np.array([20, 80]) ** 2))
    )
    np.testing.assert_allclose(energy_dqe, expected_energy, rtol=2e-14)
    assert energy_dqe[0] < count_dqe[0]
    assert counts_spec.response_recipe is counting


def test_photoelectric_branch_keeps_bernoulli_second_moment(warp_device: str) -> None:
    provenance = Provenance(
        "analytic:total=5,photoelectric=2cm2/g", "f" * 64, "CC0-1.0", "Analytic branch fixture"
    )
    total = MaterialTable("analytic compound", (60.0,), (5.0,), "cm^2/g", provenance, 4.0)
    branch = PhotonCoefficientTable(
        "analytic compound", "photoelectric", (60.0,), (2.0,), 4.0, provenance
    )
    response = DetectorResponse(
        total,
        (60.0,),
        0.7,
        FirstInteractionModel.photoelectric_only(total, branch, (60.0,), edge_side="above"),
        "above",
    )
    spec = DetectorTransferSpec.from_response(
        response, (20.0,), psf_shape=(1, 1), psf_weights=(1.0,), spacing_mm=(1.0, 1.0)
    )
    efficiency = -math.expm1(-2.0 * 0.7) * 0.4
    assert spec.first_moments[0] == pytest.approx(efficiency * 60)
    assert spec.second_moments[0] == pytest.approx(efficiency * 60**2)
    assert evaluate(spec, np.array([[0.0, 0.0]]), warp_device)[2][0] == pytest.approx(
        efficiency, rel=1e-14
    )


def test_noise_spectrum_matches_dense_circulant_covariance(warp_device: str) -> None:
    spec = model()
    rows, cols = 5, 7
    size = rows * cols
    impulse = embedded_psf(spec, (rows, cols))
    matrix = np.zeros((size, size))
    for source in range(size):
        sr, sc = divmod(source, cols)
        for destination in range(size):
            dr, dc = divmod(destination, cols)
            matrix[destination, source] = impulse[(dr - sr) % rows, (dc - sc) % cols]
    variance = np.dot(spec.incident_bin_photons, spec.second_moments)
    covariance = variance * matrix @ matrix.T + spec.white_read_noise_variance * np.eye(size)
    eigen_nps = np.fft.fft2(covariance[0].reshape(rows, cols)).real * np.prod(spec.spacing_mm)
    measured = evaluate(spec, frequency_grid(spec, (rows, cols)), warp_device)[1]
    np.testing.assert_allclose(measured.reshape(rows, cols), eigen_nps, rtol=3e-14, atol=1e-14)
    assert covariance[0, 1] > 0  # deterministic event spreading correlates pixels


def test_bounded_compound_poisson_ensemble_matches_predicted_nps(warp_device: str) -> None:
    spec = model()
    shape = (8, 16)
    samples = 8192
    rng = np.random.default_rng(23071)
    signal = 2.0 * rng.poisson(7.0 * 0.2, size=(samples, *shape)) + 3.0 * rng.poisson(
        11.0 * 0.8, size=(samples, *shape)
    )
    signal = signal.astype(np.float64) - (7 * 0.2 * 2 + 11 * 0.8 * 3)
    transfer = np.fft.fft2(embedded_psf(spec, shape))
    spread = np.fft.ifft2(np.fft.fft2(signal, axes=(-2, -1)) * transfer, axes=(-2, -1)).real
    observed = spread + rng.normal(0, math.sqrt(spec.white_read_noise_variance), size=spread.shape)
    periodograms = (
        abs(np.fft.fft2(observed, axes=(-2, -1))) ** 2 / np.prod(shape) * np.prod(spec.spacing_mm)
    )
    estimated = np.mean(periodograms, axis=0).reshape(-1)
    predicted = evaluate(spec, frequency_grid(spec, shape), warp_device)[1]
    # Real DC/Nyquist modes have at most twice the Gaussian periodogram
    # variance. Six standard errors conservatively covers all 128 contrasts;
    # the small compound-Poisson fourth-cumulant contribution is <0.2% here.
    tolerance = 6 * math.sqrt(2 / samples) + 0.002
    np.testing.assert_allclose(estimated, predicted, rtol=tolerance, atol=0)
    wrong_variance = np.dot(spec.incident_bin_photons, np.array(spec.first_moments) ** 2)
    assert abs(wrong_variance / spec.quantum_variance_before_spread - 1) > 2 * tolerance


def test_transfer_invalid_frequency_sticky_status_shape_and_alias(warp_device: str) -> None:
    spec = model()
    workspace = prepare_detector_transfer(spec, max_frequencies=2, device=warp_device)
    frequencies = wp.array([[0.0, 0.0], [2.0, 0.0]], dtype=wp.float64, device=warp_device)
    outputs = tuple(wp.empty(2, dtype=wp.float64, device=warp_device) for _ in range(3))
    detector_transfer(
        frequencies,
        out_mtf=outputs[0],
        out_nps=outputs[1],
        out_dqe=outputs[2],
        workspace=workspace,
        validate=False,
    )
    with pytest.raises(ContractError, match="invalid fixed"):
        workspace.check_status()
    frequencies.zero_()
    with pytest.raises(ContractError):
        detector_transfer(
            frequencies,
            out_mtf=outputs[0],
            out_nps=outputs[1],
            out_dqe=outputs[2],
            workspace=workspace,
        )
    workspace.clear_status()
    detector_transfer(
        frequencies, out_mtf=outputs[0], out_nps=outputs[1], out_dqe=outputs[2], workspace=workspace
    )
    with pytest.raises(ContractError, match="overlaps"):
        detector_transfer(
            frequencies,
            out_mtf=outputs[0],
            out_nps=outputs[0],
            out_dqe=outputs[2],
            workspace=workspace,
        )
    with pytest.raises(ContractError, match="shape"):
        detector_transfer(
            frequencies,
            out_mtf=wp.empty(1, dtype=wp.float64, device=warp_device),
            out_nps=outputs[1],
            out_dqe=outputs[2],
            workspace=workspace,
        )
    with pytest.raises(ContractError, match="max_owned_bytes"):
        prepare_detector_transfer(spec, max_frequencies=2, device=warp_device, max_owned_bytes=79)


def test_supplied_deposition_requires_independent_second_moment(warp_device: str) -> None:
    provenance = Provenance(
        "analytic:deposition mean=10keV",
        "a" * 64,
        "CC0-1.0",
        "Analytic two-level deposition fixture",
    )
    response = DetectorResponse(
        MaterialTable("analytic", (20.0,), (0.5,), "mm^-1"),
        (20.0,),
        0.7,
        FirstInteractionModel("supplied-deposition", (10.0,), provenance),
        "above",
    )
    kwargs: Any = dict(psf_shape=(1, 1), psf_weights=(1.0,), spacing_mm=(1.0, 1.0))
    with pytest.raises(ContractError, match="conditional second"):
        DetectorTransferSpec.from_response(response, (20.0,), **kwargs)
    with pytest.raises(ContractError, match="mean²"):
        DetectorTransferSpec.from_response(
            response,
            (20.0,),
            conditional_second_moments=(99.0,),
            second_moments_provenance=provenance,
            **kwargs,
        )
    spec = DetectorTransferSpec.from_response(
        response,
        (20.0,),
        conditional_second_moments=(200.0,),
        second_moments_provenance=provenance,
        **kwargs,
    )
    efficiency = -math.expm1(-0.5 * 0.7)
    assert evaluate(spec, np.array([[0.0, 0.0]]), warp_device)[2][0] == pytest.approx(
        efficiency / 2, rel=1e-14
    )


def test_transfer_retains_scaled_noise_and_has_defined_null_modes(warp_device: str) -> None:
    tiny = DetectorTransferSpec(
        (1e300,), (1.0,), (1.0,), "signal", (1, 1), (1e-300,), (1e150, 1e150)
    )
    mtf, nps, dqe = evaluate(tiny, np.array([[0.0, 0.0]]), warp_device)
    assert mtf[0] == 1.0
    assert nps[0] == pytest.approx(1.0, rel=2e-15)
    assert dqe[0] == pytest.approx(1.0, rel=2e-15)
    null = DetectorTransferSpec((20.0,), (0.0,), (0.0,), "signal", (1, 1), (1.0,), (1.0, 1.0))
    mtf, nps, dqe = evaluate(null, np.array([[0.0, 0.0]]), warp_device)
    assert mtf[0] == 1.0 and nps[0] == 0.0 and dqe[0] == 0.0


@pytest.mark.parametrize("spacing", [(1e-154, 1e-154), (1e-161, 1e-162)])
def test_transfer_scales_noise_before_unscaled_overflow(
    spacing: tuple[float, float], warp_device: str
) -> None:
    spec = DetectorTransferSpec(
        (1.0,), (1e154,), (1e308,), "signal", (1, 1), (1.0,), spacing, 1e308
    )
    mtf, nps, dqe = evaluate(spec, np.array([[0.0, 0.0]]), warp_device)
    with localcontext() as context:
        context.prec = 100
        noise = Decimal.from_float(1e308)
        expected = float(
            Decimal.from_float(spacing[0]) * Decimal.from_float(spacing[1]) * 2 * noise
        )
        expected_dqe = float(Decimal.from_float(1e154) ** 2 / (2 * noise))
    assert mtf[0] == 1.0
    assert nps[0] == pytest.approx(expected, rel=3e-15, abs=0)
    assert dqe[0] == pytest.approx(expected_dqe, rel=3e-15)


def test_transfer_does_not_round_quantum_tail_before_area_scaling(warp_device: str) -> None:
    spec = DetectorTransferSpec(
        (1.0,), (1e-100,), (1e-200,), "signal", (1, 1), (3e-62,), (1e100, 1e100)
    )
    mtf, nps, dqe = evaluate(spec, np.array([[0.0, 0.0]]), warp_device)
    with localcontext() as context:
        context.prec = 100
        expected = float(
            Decimal.from_float(1e-200)
            * Decimal.from_float(3e-62) ** 2
            * Decimal.from_float(1e100) ** 2
        )
    assert mtf[0] == 1.0
    assert nps[0] == pytest.approx(expected, rel=3e-15, abs=0)
    assert dqe[0] == pytest.approx(1.0, rel=3e-15)


def test_transfer_exact_lattice_nyquist_null_has_zero_dqe(warp_device: str) -> None:
    spec = DetectorTransferSpec(
        (1.0,), (1.0,), (1.0,), "signal", (1, 3), (0.0, 0.5, 0.5), (1.0, 1.0)
    )
    mtf, nps, dqe = evaluate(spec, np.array([[0.5, 0.0], [-0.5, 0.0]]), warp_device)
    np.testing.assert_array_equal(mtf, [0.0, 0.0])
    np.testing.assert_array_equal(nps, [0.0, 0.0])
    np.testing.assert_array_equal(dqe, [0.0, 0.0])


def test_transfer_sums_combined_subnormal_noise_before_final_rounding(warp_device: str) -> None:
    spec = DetectorTransferSpec(
        (1.0,), (1.0,), (1.0,), "signal", (1, 1), (1.0,), (1.5e-162, 1.5e-162), 1.0
    )
    mtf, nps, dqe = evaluate(spec, np.array([[0.0, 0.0]]), warp_device)
    with localcontext() as context:
        context.prec = 100
        expected = float(Decimal.from_float(1.5e-162) ** 2 * 2)
    assert expected == 5e-324
    assert mtf[0] == 1.0 and nps[0] == expected
    assert dqe[0] == 0.5


@pytest.mark.parametrize("weights", [(0.25, 0.25, 1e-200), (1e-200, 0.25, 0.25)])
def test_weak_nonzero_transfer_retains_noise_and_efficiency(
    weights: tuple[float, ...], warp_device: str
) -> None:
    spec = DetectorTransferSpec((1e300,), (1.0,), (1.0,), "signal", (1, 3), weights, (1e150, 1e150))
    mtf, nps, dqe = evaluate(spec, np.array([[5e-151, 0.0]]), warp_device)
    with localcontext() as context:
        context.prec = 100
        expected_nps = float(
            Decimal.from_float(1e300)
            * Decimal.from_float(1e-200) ** 2
            * Decimal.from_float(1e150) ** 2
        )
    assert mtf[0] == pytest.approx(2e-200, rel=3e-15, abs=0)
    assert nps[0] == pytest.approx(expected_nps, rel=3e-15, abs=0)
    assert dqe[0] == pytest.approx(1.0, rel=3e-15)


@pytest.mark.parametrize(
    "defect", ["second", "zero-psf", "active-psf", "zero-fluence", "even", "spacing"]
)
def test_transfer_model_rejects_undefined_or_nonphysical_inputs(defect: str) -> None:
    variants: Any = {
        "second": {"second_moments": (0.01, 1.0)},
        "zero-psf": {"psf_weights": (0.0,) * 9},
        "active-psf": {"psf_weights": (1.0,) * 9},
        "zero-fluence": {"incident_bin_photons": (0.0, 0.0)},
        "even": {"psf_shape": (2, 3)},
        "spacing": {"spacing_mm": (0.5, 0.0)},
    }
    with pytest.raises(ContractError):
        replace(model(), **variants[defect])
