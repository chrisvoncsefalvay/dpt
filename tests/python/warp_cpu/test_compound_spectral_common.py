"""Compound sampling preserves bin identities, layouts, moments and failures."""

# Warp annotations are executable DSL expressions.
# pyright: reportInvalidTypeForm=false

import importlib
import math
from dataclasses import replace
from decimal import Decimal, localcontext
from itertools import product
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import NumericalError
from dpt.detector import (
    ObservationIdentity,
    prepare_detector,
    sample_compound_poisson,
    sample_poisson_counts,
)
from dpt.kernels.spectral_common import STRICT, compensated_product_sum_update
from dpt.validation.detector import compound_poisson_moments

wp: Any = importlib.import_module("warp")
MODES = list(product((False, True), repeat=2))


@wp.kernel(module="unique", module_options=STRICT)
def _original_compound_prefixes(
    counts: wp.array(dtype=wp.uint64),
    scores: wp.array(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float64),
):
    total = wp.float64(0.0)
    compensation = wp.float64(0.0)
    for energy in range(counts.shape[0]):
        count = counts[energy]
        # These four statements retain the b0119ce compound expression and
        # compiler fusion boundary, including multiplication before correction.
        corrected = wp.float64(count) * wp.float64(scores[energy]) - compensation
        updated = total + corrected
        compensation = (updated - total) - corrected
        total = updated
        output[energy, 0] = total
        output[energy, 1] = compensation


@wp.kernel(module="unique", module_options=STRICT)
def _factored_compound_prefixes(
    counts: wp.array(dtype=wp.uint64),
    scores: wp.array(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float64),
):
    total = wp.float64(0.0)
    compensation = wp.float64(0.0)
    for energy in range(counts.shape[0]):
        total, compensation = compensated_product_sum_update(
            total, compensation, wp.float64(counts[energy]), wp.float64(scores[energy])
        )
        output[energy, 0] = total
        output[energy, 1] = compensation


def _decimal_prefixes(
    counts: tuple[int, ...], scores: tuple[float, ...], fused: bool
) -> np.ndarray:
    total, compensation = 0.0, 0.0
    result = []
    with localcontext() as context:
        context.prec = 200
        for count, score in zip(counts, scores, strict=True):
            exact_product = Decimal(count) * Decimal.from_float(score)
            if fused:
                corrected = float(exact_product - Decimal.from_float(compensation))
            else:
                corrected = float(exact_product) - compensation
            updated = total + corrected
            compensation = (updated - total) - corrected
            total = updated
            result.append((total, compensation))
    return np.array(result, dtype=np.float64)


@pytest.mark.parametrize("case", ["near_limit", "small_bins", "exact_products"])
def test_compound_prefixes_preserve_original_device_fusion_and_decimal_states(
    case: str, warp_device: str
) -> None:
    if case == "near_limit":
        counts_host = (1000000000, 999999998, 999999999, 999999995)
        scores_host = (2.0, 1.0 + 2.0**-23, 2.0 - 2.0**-23, 2.0**-22)
    elif case == "small_bins":
        counts_host = (1,) * 6
        scores_host = (2.0**54, 1.0, 1.0, 1.0, 1.0, 1.0)
    else:
        counts_host = (1000000000,) * 4
        scores_host = (2.0, 2.0 - 2.0**-23, 1.0 + 2.0**-23, 2.0**-22)
    counts = wp.array(counts_host, dtype=wp.uint64, device=warp_device)
    scores = wp.array(scores_host, dtype=wp.float32, device=warp_device)
    original = wp.empty((len(counts_host), 2), dtype=wp.float64, device=warp_device)
    factored = wp.empty_like(original)
    wp.launch(
        _original_compound_prefixes,
        dim=1,
        inputs=[counts, scores],
        outputs=[original],
        device=warp_device,
    )
    wp.launch(
        _factored_compound_prefixes,
        dim=1,
        inputs=[counts, scores],
        outputs=[factored],
        device=warp_device,
    )
    before, after = original.numpy(), factored.numpy()
    assert before.tobytes() == after.tobytes()
    separate = _decimal_prefixes(counts_host, scores_host, fused=False)
    fused = _decimal_prefixes(counts_host, scores_host, fused=True)
    # fuse_fp permits contraction; its actual choice is platform/compiler
    # dependent. The preserved expression chooses the baseline for this device.
    assert before.tobytes() in (separate.tobytes(), fused.tobytes())
    if case == "near_limit":
        assert separate.tobytes() != fused.tobytes()
        assert separate[-1, 0] != fused[-1, 0]
        assert separate[1, 1] == 2.0**-22


@pytest.mark.parametrize("wide,shared_scores", MODES)
@pytest.mark.parametrize("boundary_identity", [False, True])
def test_compound_bins_match_unit_counts_and_pixel_batching(
    wide: bool, shared_scores: bool, boundary_identity: bool, warp_device: str
) -> None:
    with cpu_test_context():
        energies, pixels = 4, 257
        dtype = wp.float64 if wide else wp.float32
        rates_host = np.array([0.1, 9.5, 10.0, 100.0])[:, None] + np.arange(pixels) % 3
        score_grid = np.array([0.5, 2.0, 8.0, 40.0], dtype=np.float32)[:, None]
        score_grid = np.broadcast_to(score_grid, (energies, pixels)).copy()
        if not shared_scores:
            score_grid *= 1 + np.arange(pixels) % 5
        scores = wp.array(
            (score_grid[:, 0] if shared_scores else score_grid).ravel(),
            dtype=wp.float32,
            device=warp_device,
        )
        rates = wp.array(rates_host.ravel(), dtype=dtype, device=warp_device)
        identity = ObservationIdentity(
            0xFEDCBA9876543210,
            2**32 - 1 if boundary_identity else 23,
            pixel_offset=2**32 - pixels if boundary_identity else 19,
            energy_offset=2**31 - 2 - energies if boundary_identity else 13,
        )
        workspace = prepare_detector(max_pixels=pixels, device=warp_device)
        output = wp.empty(pixels, dtype=wp.float32, device=warp_device)
        sample_compound_poisson(
            rates,
            scores,
            energies=energies,
            shared_scores=shared_scores,
            out_signal=output,
            identity=identity,
            workspace=workspace,
        )
        # Unit count sampling is an unchanged independent kernel. Select each
        # bin's domain, then sum the integer counts with exact binary scores.
        counts = []
        for energy in range(energies):
            unit = wp.empty(pixels, dtype=wp.uint64, device=warp_device)
            sample_poisson_counts(
                rates[energy * pixels : (energy + 1) * pixels],
                out_counts=unit,
                identity=replace(identity, energy_offset=identity.energy_offset + energy),
                workspace=workspace,
            )
            counts.append(unit.numpy())
        expected = np.array(
            [
                math.fsum(int(counts[k][p]) * float(score_grid[k, p]) for k in range(energies))
                for p in range(pixels)
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(output.numpy(), expected)
        batched = []
        for lower, upper in ((0, 129), (129, pixels)):
            batch_rates = wp.array(
                rates_host[:, lower:upper].ravel(), dtype=dtype, device=warp_device
            )
            batch_scores = (
                scores
                if shared_scores
                else wp.array(
                    score_grid[:, lower:upper].ravel(),
                    dtype=wp.float32,
                    device=warp_device,
                )
            )
            batch = wp.empty(upper - lower, dtype=wp.float32, device=warp_device)
            sample_compound_poisson(
                batch_rates,
                batch_scores,
                energies=energies,
                shared_scores=shared_scores,
                out_signal=batch,
                identity=replace(identity, pixel_offset=identity.pixel_offset + lower),
                workspace=workspace,
            )
            batched.append(batch.numpy())
        np.testing.assert_array_equal(np.concatenate(batched), output.numpy())


@pytest.mark.parametrize("wide,shared_scores", MODES)
def test_compound_ensemble_matches_independent_cumulant_moments(
    wide: bool, shared_scores: bool, warp_device: str
) -> None:
    with cpu_test_context():
        pixels, energies = 65536, 4
        dtype = wp.float64 if wide else wp.float32
        rate_values = (0.1, 9.999, 10.0, 500.0)
        rate_grid = np.broadcast_to(np.array(rate_values)[:, None], (energies, pixels)).copy()
        score_grid = np.broadcast_to(
            np.array([0.5, 1.5, 80.0, 3.0], dtype=np.float32)[:, None],
            (energies, pixels),
        ).copy()
        if not shared_scores:
            score_grid[:, 1::2] *= 2.0
        rates = wp.array(rate_grid.ravel(), dtype=dtype, device=warp_device)
        scores = wp.array(
            (score_grid[:, 0] if shared_scores else score_grid).ravel(),
            dtype=wp.float32,
            device=warp_device,
        )
        output = wp.empty(pixels, dtype=wp.float32, device=warp_device)
        workspace = prepare_detector(max_pixels=pixels, device=warp_device)
        sample_compound_poisson(
            rates,
            scores,
            energies=energies,
            shared_scores=shared_scores,
            out_signal=output,
            identity=ObservationIdentity(17011, 41),
            workspace=workspace,
        )
        exact_rates = tuple(float(value) for value in rates.numpy().reshape(energies, pixels)[:, 0])
        samples = output.numpy().astype(np.float64)
        for parity in (0, 1):
            selected = samples[parity::2]
            mean, variance, fourth = compound_poisson_moments(
                exact_rates, tuple(float(value) for value in score_grid[:, parity])
            )
            size = len(selected)
            mean_error = 6.0 * math.sqrt(variance / size)
            variance_error = 6.0 * math.sqrt(
                (fourth - ((size - 3) / (size - 1)) * variance**2) / size,
            )
            assert abs(float(np.mean(selected)) - mean) <= mean_error
            assert abs(float(np.var(selected, ddof=1)) - variance) <= variance_error
            assert mean > mean_error and variance > variance_error


@pytest.mark.parametrize("wide,shared_scores", MODES)
@pytest.mark.parametrize("failure", ["overflow", "rejection"])
def test_compound_failure_status_stays_pending_until_explicit_clear(
    wide: bool, shared_scores: bool, failure: str, warp_device: str
) -> None:
    with cpu_test_context():
        energies, pixels = 3, 4096
        dtype = wp.float64 if wide else wp.float32
        rate = 1e9 if failure == "overflow" else 10.0
        score = float(np.finfo(np.float32).max) if failure == "overflow" else 1.0
        rates = wp.full(energies * pixels, rate, dtype=dtype, device=warp_device)
        scores = wp.full(
            energies if shared_scores else energies * pixels,
            score,
            dtype=wp.float32,
            device=warp_device,
        )
        output = wp.empty(pixels, dtype=wp.float32, device=warp_device)
        workspace = prepare_detector(max_pixels=pixels, device=warp_device)
        match = "cannot be represented" if failure == "overflow" else "budget"
        identity = ObservationIdentity(0, 0, draw_budget=256 if failure == "overflow" else 1)
        with pytest.raises(NumericalError, match=match):
            sample_compound_poisson(
                rates,
                scores,
                energies=energies,
                shared_scores=shared_scores,
                out_signal=output,
                identity=identity,
                workspace=workspace,
            )
        with pytest.raises(NumericalError, match=match):
            workspace.check_status()
        workspace.clear_status()
        workspace.check_status()
        rates.zero_()
        sample_compound_poisson(
            rates,
            scores,
            energies=energies,
            shared_scores=shared_scores,
            out_signal=output,
            identity=identity,
            workspace=workspace,
        )
        np.testing.assert_array_equal(output.numpy(), np.zeros(pixels, dtype=np.float32))


@pytest.mark.parametrize("wide,shared_scores", MODES)
def test_compound_zero_and_deep_rates_keep_exact_zero_with_large_scores(
    wide: bool, shared_scores: bool, warp_device: str
) -> None:
    with cpu_test_context():
        energies, pixels = 3, 257
        dtype = wp.float64 if wide else wp.float32
        rates = wp.array(
            np.broadcast_to(np.array([0.0, 1e-300, 1e-30])[:, None], (energies, pixels)).ravel(),
            dtype=dtype,
            device=warp_device,
        )
        scores = wp.full(
            energies if shared_scores else energies * pixels,
            float(np.finfo(np.float32).max),
            dtype=wp.float32,
            device=warp_device,
        )
        workspace = prepare_detector(max_pixels=pixels, device=warp_device)
        output = wp.empty(pixels, dtype=wp.float32, device=warp_device)
        sample_compound_poisson(
            rates,
            scores,
            energies=energies,
            shared_scores=shared_scores,
            out_signal=output,
            identity=ObservationIdentity(17, 9, draw_budget=1),
            workspace=workspace,
        )
        workspace.check_status()
        np.testing.assert_array_equal(output.numpy(), np.zeros(pixels, dtype=np.float32))
