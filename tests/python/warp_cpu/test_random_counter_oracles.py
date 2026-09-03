"""Portable detector, Poisson endpoint and counter-address regression oracles."""

import importlib
from dataclasses import replace
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.detector import (
    BlurSpec,
    CalibrationSpec,
    ObservationIdentity,
    blur,
    blur_transpose,
    calibrate,
    calibration_vjp,
    prepare_blur,
    prepare_detector,
    sample_compound_poisson,
    sample_poisson_counts,
)
from dpt.materials import Provenance
from dpt.spectral import SpectralSpec, prepare_spectral, spectral_bin_counts
from dpt.validation.detector import matrix_product, poisson_log_mass, spatial_response_matrix

# Warp kernel annotations are DSL expressions rather than Python type expressions.
# pyright: reportInvalidTypeForm=false


FIXTURE = Provenance("analytic:detector-counters", "0" * 64, "Apache-2.0", "Exact arrays")
STRICT = {"enable_backward": False, "fast_math": False}
wp: Any = importlib.import_module("warp")
detector_kernels: Any = importlib.import_module("dpt.kernels.detector")
random_kernels: Any = importlib.import_module("dpt.kernels.random")


@wp.kernel(module="unique", module_options=STRICT)
def inverse_probe(
    rates: wp.array(dtype=wp.float64), u: wp.float64, output: wp.array(dtype=wp.uint64)
):
    p = wp.tid()
    output[p] = detector_kernels.poisson_inverse_small(rates[p], u)


@wp.kernel(module="unique", module_options=STRICT)
def log_mass_probe(count: wp.float64, rate: wp.float64, output: wp.array(dtype=wp.float64)):
    output[0] = detector_kernels.poisson_log_probability(count, rate)


@wp.kernel(module="unique", module_options=STRICT)
def words_probe(
    seed: wp.uint64,
    identity: wp.uint64,
    event: wp.uint32,
    domain: wp.uint32,
    output: wp.array(dtype=wp.float64),
):
    words = random_kernels.random4(seed, identity, event, domain)
    for i in range(4):
        output[i] = words[i]


@wp.kernel(module="unique", module_options=STRICT)
def counter_probe(
    seed: wp.uint64, observation: wp.uint32, domain: wp.uint32, output: wp.array(dtype=wp.float64)
):
    p = wp.tid()
    identity = (wp.uint64(observation) << wp.uint64(32)) | wp.uint64(p)
    output[p] = detector_kernels.uniform_pair(seed, identity, wp.uint32(0), domain)[0]


def test_small_rate_upper_uniform_endpoint_matches_decimal_quantile(warp_device):
    rates = np.linspace(0.001, 9.999, 4000, dtype=np.float32).astype(np.float64)
    maximum_uniform = float(2**53 / (2**53 + 2))
    # Reproduce the previous lower-CDF saturation fault on these actual rates.
    failed_rates = []
    for rate in rates:
        probability = np.exp(-rate)
        cumulative = probability
        for count in range(1, 129):
            probability *= rate / count
            cumulative += probability
        if cumulative < maximum_uniform:
            failed_rates.append(float(rate))
    assert failed_rates
    values = np.array(failed_rates)
    output = wp.empty(len(values), dtype=wp.uint64, device=warp_device)
    wp.launch(
        inverse_probe,
        dim=len(values),
        inputs=[
            wp.array(values, dtype=wp.float64, device=warp_device),
            wp.float64(maximum_uniform),
        ],
        outputs=[output],
        device=warp_device,
    )
    expected = []
    with localcontext() as context:
        context.prec = 90
        u = Decimal.from_float(maximum_uniform)
        for rate in values:
            mean = Decimal.from_float(float(rate))
            probability = (-mean).exp()
            cumulative = probability
            count = 0
            while cumulative < u:
                count += 1
                probability *= mean / count
                cumulative += probability
            expected.append(count)
    assert list(output.numpy()) == expected


@pytest.mark.parametrize(
    "count,rate",
    [
        (0, 10),
        (1, 10),
        (15, 10),
        (16, 16),
        (18, 22),
        (22, 18),
        (1000, 999.9),
        (10**9, 1e9),
        (10**9 - 400000, 1e9),
        (10**9 + 400000, 1e9),
    ],
)
def test_poisson_acceptance_log_mass_matches_independent_definition(count, rate, warp_device):
    output = wp.empty(1, dtype=wp.float64, device=warp_device)
    wp.launch(
        log_mass_probe,
        dim=1,
        inputs=[wp.float64(count), wp.float64(rate)],
        outputs=[output],
        device=warp_device,
    )
    assert float(output.numpy()[0]) == pytest.approx(
        float(poisson_log_mass(count, rate)), rel=0, abs=5e-13
    )


# Random123's published Philox4x32-10 KAT words, not generated expectations:
# https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
@pytest.mark.parametrize(
    "seed,identity,event,domain,expected",
    [
        (0, 0, 0, 0, (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        (
            2**64 - 1,
            2**64 - 1,
            2**32 - 1,
            2**32 - 1,
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            0x299F31D0A4093822,
            0x85A308D3243F6A88,
            0x13198A2E,
            0x03707344,
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ],
)
def test_philox_known_answer_words(seed, identity, event, domain, expected, warp_device):
    output = wp.empty(4, dtype=wp.float64, device=warp_device)
    wp.launch(
        words_probe,
        dim=1,
        inputs=[wp.uint64(seed), wp.uint64(identity), wp.uint32(event), wp.uint32(domain)],
        outputs=[output],
        device=warp_device,
    )
    assert tuple(int(x * 2**32) for x in output.numpy()) == expected


def test_counter_namespaces_and_adjacent_pixels_decorrelate(warp_device):
    n = 65536
    draws = []
    for seed, observation, domain in ((713, 23, 1), (719, 23, 1), (713, 24, 1), (713, 23, 2)):
        output = wp.empty(n, dtype=wp.float64, device=warp_device)
        wp.launch(
            counter_probe,
            dim=n,
            inputs=[wp.uint64(seed), wp.uint32(observation), wp.uint32(domain)],
            outputs=[output],
            device=warp_device,
        )
        draws.append(output.numpy())
    for comparison in draws[1:]:
        assert not np.array_equal(draws[0], comparison)
        assert abs(np.corrcoef(draws[0], comparison)[0, 1]) < 6 / np.sqrt(n)
    assert abs(np.corrcoef(draws[0][:-1], draws[0][1:])[0, 1]) < 6 / np.sqrt(n - 1)


@pytest.mark.parametrize("height,width,kh,kw", [(7, 5, 3, 5), (3, 3, 7, 7), (4, 9, 9, 3)])
def test_blur_and_transpose_match_dense_matrix(height, width, kh, kw, warp_device):
    with cpu_test_context():
        rng = np.random.default_rng(571)
        weights = rng.random(kh * kw)
        weights /= weights.sum()
        spec = BlurSpec(height, width, kh, kw, tuple(weights), FIXTURE)
        workspace = prepare_blur(spec, device=warp_device)
        source = wp.array(rng.normal(size=height * width), dtype=wp.float32, device=warp_device)
        seed = wp.array(rng.normal(size=height * width), dtype=wp.float32, device=warp_device)
        output, gradient = wp.empty_like(source), wp.empty_like(seed)
        blur(source, out_signal=output, workspace=workspace)
        blur_transpose(seed, out_grad_signal=gradient, workspace=workspace)
        matrix = spatial_response_matrix(
            height, width, kh, kw, tuple(float(np.float32(x)) for x in weights)
        )
        assert output.numpy() == pytest.approx(
            matrix_product(matrix, tuple(float(x) for x in source.numpy())), rel=2e-6, abs=2e-8
        )
        assert gradient.numpy() == pytest.approx(
            matrix_product(matrix, tuple(float(x) for x in seed.numpy()), transpose=True),
            rel=2e-6,
            abs=2e-8,
        )
        assert np.dot(seed.numpy().astype(float), output.numpy()) == pytest.approx(
            np.dot(source.numpy().astype(float), gradient.numpy()), rel=2e-6, abs=2e-8
        )


def test_calibration_values_and_nuisance_cotangents(warp_device):
    with cpu_test_context():
        workspace = prepare_detector(max_pixels=3, device=warp_device)
        spec = CalibrationSpec("ADU", active_exposure=True, active_offset=True)
        arrays = [
            wp.array(x, dtype=wp.float32, device=warp_device)
            for x in ([0, 2, 10], [3], [4], [-2], [1, -2, 0.5])
        ]
        mean, gain, exposure, offset, seed = arrays
        signal, gm = wp.empty_like(mean), wp.empty_like(mean)
        ge, go = wp.empty_like(exposure), wp.empty_like(offset)
        calibrate(mean, gain, exposure, offset, out_signal=signal, spec=spec, workspace=workspace)
        calibration_vjp(
            mean,
            gain,
            exposure,
            offset,
            cotangent=seed,
            out_grad_mean=gm,
            out_grad_exposure=ge,
            out_grad_offset=go,
            spec=spec,
            workspace=workspace,
        )
        assert list(signal.numpy()) == [-2, 22, 118]
        assert list(gm.numpy()) == [12, -24, 6]
        assert list(ge.numpy()) == [3]
        assert list(go.numpy()) == [-0.5]


@pytest.mark.parametrize("rate_dtype", [wp.float32, wp.float64])
def test_spectral_bin_to_compound_sampler_and_rate_precision_replay(rate_dtype, warp_device):
    with cpu_test_context():
        precision = "float64" if rate_dtype == wp.float64 else "float32"
        policy = SpectralSpec(
            1, 2, (FIXTURE,), FIXTURE, FIXTURE, "counts", "analytic", precision=precision
        )
        spectral = prepare_spectral(policy, max_pixels=513, device=warp_device)
        detector = prepare_detector(max_pixels=513, device=warp_device)
        paths = wp.zeros(513, dtype=rate_dtype, device=warp_device)
        mu = wp.array([1, 2], dtype=wp.float32, device=warp_device)
        weights = wp.array([3, 10], dtype=wp.float32, device=warp_device)
        response = wp.array([1, 1], dtype=wp.float32, device=warp_device)
        rates = wp.empty(1026, dtype=rate_dtype, device=warp_device)
        spectral_bin_counts(paths, mu, weights, response, out_bin_counts=rates, workspace=spectral)
        signal = wp.empty(513, dtype=wp.float32, device=warp_device)
        scores = wp.array([2, 4], dtype=wp.float32, device=warp_device)
        identity = ObservationIdentity(319, 17)
        sample_compound_poisson(
            rates,
            scores,
            energies=2,
            shared_scores=True,
            out_signal=signal,
            identity=identity,
            workspace=detector,
        )
        counts = [wp.empty(513, dtype=wp.uint64, device=warp_device) for _ in range(2)]
        for energy in range(2):
            sample_poisson_counts(
                rates[energy * 513 : (energy + 1) * 513],
                out_counts=counts[energy],
                identity=replace(identity, energy_offset=energy),
                workspace=detector,
            )
        assert np.array_equal(signal.numpy(), 2 * counts[0].numpy() + 4 * counts[1].numpy())
