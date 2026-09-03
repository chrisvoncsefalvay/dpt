"""Powered conditional-law checks for the canonical free-electron sampler.

Independent differential-law quadrature supplies 32-bin probabilities. Known-
multinomial Monte Carlo ranks calibrate the ideal IID conditional-law model,
separately from the fixed production RNG and its 32-bit rejection discretisation.
Alternative trials estimate power conditional on the recorded calibration.
"""

# Warp's decorators/annotations are runtime DSL, while host tests remain typed.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false, reportUnknownParameterType=false
import importlib
import json
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy import stats

from dpt.validation.transport import compton_energy, klein_nishina_total_ratio

wp: Any = importlib.import_module("warp")
kernels: Any = importlib.import_module("dpt.transport.kernels")
DEVICE = "cpu"
BINS = 32
ALPHA = 2**-14
NULL_DRAWS = 2**16 - 1
POWER_DRAWS = 2**13
SAMPLE_CHUNK = 2**18
HISTORIES = {1.0: 2**24, 80.0: SAMPLE_CHUNK, 1000.0: SAMPLE_CHUNK}


@pytest.fixture(autouse=True)
def _device_scope(warp_device: str):
    global DEVICE
    DEVICE = warp_device
    with wp.ScopedDevice(DEVICE):
        yield


@wp.kernel
def _sample(
    energy: wp.float64,
    first_history: wp.uint64,
    out_values: wp.array(dtype=wp.vec4d),
):
    history = first_history + wp.uint64(wp.tid())
    out_values[wp.tid()] = kernels.compton_scatter(
        energy, wp.uint64(95371), history, wp.uint32(0), 4096
    )


def _kn_masses(energy: float, nodes_per_bin: int) -> np.ndarray:
    # Integrate the physical differential cross section directly. Neither the
    # rejection envelope nor any production kernel output defines this law.
    nodes, weights = np.polynomial.legendre.leggauss(nodes_per_bin)
    edges = np.linspace(-1.0, 1.0, BINS + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2
    radii = np.diff(edges) / 2
    cosine = midpoints[:, None] + radii[:, None] * nodes
    ratio = 1 / (1 + energy / 510.99895069 * (1 - cosine))
    differential = ratio**2 * (ratio + 1 / ratio - (1 - cosine**2))
    return radii * np.sum(weights * differential, axis=1)


def _pearson(counts: np.ndarray, probabilities: np.ndarray, histories: int) -> np.ndarray:
    expected = histories * probabilities
    return np.sum((counts - expected) ** 2 / expected, axis=-1)


def _calibration(probabilities: np.ndarray, histories: int, seed: int) -> np.ndarray:
    # Calibration uses a separate host PCG64 stream, never the sampler's Philox
    # histories. With a known null (no fitted parameters), exchangeable ranks
    # are finite-sample valid, including the +1 numerator and denominator.
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(histories, probabilities, size=NULL_DRAWS)
    return np.sort(_pearson(counts, probabilities, histories))


def _rank_p(statistic: float, null_statistics: np.ndarray) -> float:
    at_least_as_large = len(null_statistics) - int(
        np.searchsorted(null_statistics, statistic, side="left")
    )
    return (1 + at_least_as_large) / (len(null_statistics) + 1)


def _power_plan(
    probabilities: np.ndarray,
    alternative: np.ndarray,
    histories: int,
    null_statistics: np.ndarray,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(histories, alternative, size=POWER_DRAWS)
    statistic = _pearson(counts, probabilities, histories)
    # p <= alpha requires strictly exceeding the fourth largest null statistic;
    # equal statistics remain conservative under the rank's tie handling.
    rank_limit = int(ALPHA * (len(null_statistics) + 1))
    critical = float(null_statistics[-rank_limit])
    rejected = int(np.count_nonzero(statistic > critical))
    lower = float(stats.beta.ppf(0.01, rejected, POWER_DRAWS - rejected + 1)) if rejected else 0.0
    return {
        "draws": POWER_DRAWS,
        "rejected": rejected,
        "estimated_power": rejected / POWER_DRAWS,
        "one_sided_99_percent_CP_lower": lower,
        "seed": seed,
    }


def _fine_bin_alternative(probabilities: np.ndarray, sign: int) -> np.ndarray:
    # Move 2% of the smaller adjacent-bin mass, preserving each old quarter's
    # total exactly. The original four-bin check cannot detect this distortion.
    transfer = sign * 0.02 * np.minimum(probabilities[::2], probabilities[1::2])
    alternative = probabilities.copy()
    alternative[::2] -= transfer
    alternative[1::2] += transfer
    np.testing.assert_allclose(
        alternative.reshape(4, 8).sum(axis=1),
        probabilities.reshape(4, 8).sum(axis=1),
        rtol=0,
        atol=2e-16,
    )
    return alternative


@pytest.mark.parametrize("energy", [1.0, 80.0, 1000.0])
def test_klein_nishina_angular_distribution_and_energy_conservation(
    energy: float, record_testsuite_property: Callable[[str, object], None]
) -> None:
    histories = HISTORIES[energy]
    coarse, masses = _kn_masses(energy, 16), _kn_masses(energy, 32)
    refinement_error = float(np.max(np.abs(masses - coarse)))
    np.testing.assert_allclose(coarse, masses, rtol=1e-13, atol=2e-15)
    integral = float(np.sum(masses))
    exact_normalisation = float(klein_nishina_total_ratio(energy))
    normalisation_error = abs(integral * 3 / 8 - exact_normalisation) / exact_normalisation
    assert normalisation_error < 1e-12
    probabilities = masses / integral
    assert histories * float(np.min(probabilities)) > 1000
    cosine_counts = np.zeros(BINS, dtype=np.int64)
    azimuth_counts = np.zeros(BINS, dtype=np.int64)
    values = wp.empty(SAMPLE_CHUNK, dtype=wp.vec4d, device=DEVICE)
    for first in range(0, histories, SAMPLE_CHUNK):
        wp.launch(
            _sample, dim=SAMPLE_CHUNK, inputs=[energy, wp.uint64(first), values], device=DEVICE
        )
        observed = values.numpy()
        assert np.all(observed[:, 3] == 0)
        assert np.all((observed[:, 0] >= -1) & (observed[:, 0] < 1))
        assert np.all((observed[:, 1] >= 0) & (observed[:, 1] < 2 * math.pi))
        cosine_counts += np.histogram(observed[:, 0], bins=BINS, range=(-1, 1))[0]
        azimuth_counts += np.histogram(observed[:, 1], bins=BINS, range=(0, 2 * math.pi))[0]
        expected_energy = energy / (1 + energy / 510.99895069 * (1 - observed[:, 0]))
        np.testing.assert_allclose(observed[:, 2], expected_energy, rtol=1e-14, atol=0)
        for cosine, _, outgoing, _ in observed[::2048]:
            assert float(outgoing) == pytest.approx(
                float(compton_energy(energy, float(cosine))), rel=1e-14
            )
    assert int(cosine_counts.sum()) == int(azimuth_counts.sum()) == histories
    calibration_seed = 610001 + int(energy)
    cosine_null = _calibration(probabilities, histories, calibration_seed)
    cosine_statistic = float(_pearson(cosine_counts, probabilities, histories))
    cosine_p = _rank_p(cosine_statistic, cosine_null)
    assert cosine_p > ALPHA
    uniform = np.full(BINS, 1 / BINS)
    azimuth_seed = 710001 + int(energy)
    azimuth_null = _calibration(uniform, histories, azimuth_seed)
    azimuth_statistic = float(_pearson(azimuth_counts, uniform, histories))
    azimuth_p = _rank_p(azimuth_statistic, azimuth_null)
    assert azimuth_p > ALPHA

    fine_mutants = []
    for sign in (-1, 1):
        plan = _power_plan(
            probabilities,
            _fine_bin_alternative(probabilities, sign),
            histories,
            cosine_null,
            810001 + int(energy) + (sign + 1),
        )
        assert plan["one_sided_99_percent_CP_lower"] > 0.99
        mutant = cosine_counts.copy()
        transfer = sign * np.floor(0.02 * np.minimum(mutant[::2], mutant[1::2])).astype(np.int64)
        mutant[::2] -= transfer
        mutant[1::2] += transfer
        np.testing.assert_array_equal(
            mutant.reshape(4, 8).sum(axis=1), cosine_counts.reshape(4, 8).sum(axis=1)
        )
        mutant_p = _rank_p(float(_pearson(mutant, probabilities, histories)), cosine_null)
        assert mutant_p <= ALPHA
        fine_mutants.append(
            {
                "sign": sign,
                "actual_rank_p": mutant_p,
                "origin": "2% adjacent-bin transfer in the actual production histogram",
                "power": plan,
            }
        )

    edges = np.linspace(-1.0, 1.0, BINS + 1)
    thomson = np.diff(edges + edges**3 / 3) / (8 / 3)
    thomson /= thomson.sum()
    thomson_plan = _power_plan(probabilities, thomson, histories, cosine_null, 910001 + int(energy))
    assert thomson_plan["one_sided_99_percent_CP_lower"] > 0.99
    thomson_seed = 1010001 + int(energy)
    thomson_mutant = np.random.default_rng(thomson_seed).multinomial(histories, thomson)
    thomson_p = _rank_p(float(_pearson(thomson_mutant, probabilities, histories)), cosine_null)
    assert thomson_p <= ALPHA
    half_azimuth = np.zeros_like(azimuth_counts)
    half_azimuth[: BINS // 2] = azimuth_counts[: BINS // 2] + azimuth_counts[BINS // 2 :]
    half_azimuth_p = _rank_p(float(_pearson(half_azimuth, uniform, histories)), azimuth_null)
    assert half_azimuth_p <= ALPHA
    record_testsuite_property(
        f"KN-law-{energy:g}keV-{DEVICE}",
        json.dumps(
            {
                "device": DEVICE,
                "histories": histories,
                "chunk_histories": SAMPLE_CHUNK,
                "sampler_seed": 95371,
                "first_history": 0,
                "bins": BINS,
                "null_draws": NULL_DRAWS,
                "rank_alpha": ALPHA,
                "null_cosine_seed": calibration_seed,
                "null_azimuth_seed": azimuth_seed,
                "quadrature_refinement_max_abs": refinement_error,
                "normalisation_relative_error": normalisation_error,
                "cosine_counts": cosine_counts.tolist(),
                "azimuth_counts": azimuth_counts.tolist(),
                "cosine_critical_statistic": float(cosine_null[-4]),
                "azimuth_critical_statistic": float(azimuth_null[-4]),
                "minimum_rank_p": 1 / (NULL_DRAWS + 1),
                "cosine_statistic": cosine_statistic,
                "cosine_rank_p": cosine_p,
                "azimuth_statistic": azimuth_statistic,
                "azimuth_rank_p": azimuth_p,
                "fine_bin_mutants": fine_mutants,
                "Thomson_mutant": {
                    "actual_rank_p": thomson_p,
                    "seed": thomson_seed,
                    "power": thomson_plan,
                    "origin": "independent host PCG64 multinomial with analytic Thomson bin masses",
                },
                "half_azimuth_mutant_rank_p": half_azimuth_p,
                "half_azimuth_mutant_origin": "production azimuth histogram folded modulo pi",
                "law": "ideal IID free-electron Klein-Nishina cosine; marginal uniform azimuth",
                "family_null_alpha_upper_per_device": 7 * ALPHA,
                "family_scope": "seven declared production null-law gates only",
                "power_confidence": "individual one-sided 99% CP, conditional on null calibration",
                "calibration": (
                    "ideal IID known multinomial null; independent host PCG64; "
                    "conservative P+1 ranks"
                ),
            },
            sort_keys=True,
        ),
    )
