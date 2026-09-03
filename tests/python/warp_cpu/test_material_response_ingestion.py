"""Independent spectral and separable detector acceptance cases."""

import importlib
import math
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError, NumericalError
from dpt.detector import (
    BlurSpec,
    ObservationIdentity,
    SeparableBlurSpec,
    blur,
    blur_transpose,
    cast_signal,
    prepare_blur,
    prepare_detector,
    sample_compound_poisson,
    sample_poisson_counts,
)
from dpt.materials import MassFractionMixture, MaterialBasis, MaterialTable, Provenance
from dpt.spectral import SpectralSpec, Spectrum, prepare_spectral, spectral_signal, spectral_vjp
from dpt.validation.spectral import spectral_reference

wp: Any = importlib.import_module("warp")
FIXTURE = Provenance(
    "analytic:material-response-tables", "0" * 64, "Apache-2.0", "Supplied analytic arrays"
)


def test_optional_metadata_preserves_supplied_records():
    spec = SpectralSpec(1, 2)
    assert spec.coefficients_provenance is None
    assert replace(spec, coefficients_provenance=(FIXTURE,)).coefficients_provenance == (FIXTURE,)
    with pytest.raises(ContractError, match="per material"):
        replace(spec, coefficients_provenance=())
    with pytest.raises(ContractError, match="non-empty"):
        replace(spec, output_unit="")
    assert Spectrum((10.0,), (1.0,), "bin-fluence").provenance is None
    assert BlurSpec(1, 1, 1, 1, (1.0,)).provenance is None


def test_explicit_uniform_bin_averaging_splits_absorption_edges():
    table = MaterialTable(
        "analytic edge", (10.0, 20.0, 20.0, 40.0), (4.0, 2.0, 8.0, 4.0), "cm^2/g", density_g_cm3=2.0
    )
    actual = table.bin_averages((10.0, 40.0), weighting="uniform-energy", quadrature_order=32)
    assert actual == pytest.approx((40 * math.log(2) / 30,), rel=3e-14)
    linear = replace(table, interpolation="linear")
    assert linear.bin_averages((10.0, 40.0), weighting="uniform-energy") == pytest.approx(
        (1.0,), rel=3e-14
    )
    with pytest.raises(ContractError, match="support"):
        table.bin_averages((9.0, 40.0), weighting="uniform-energy")


def test_host_conversions_preserve_layout_and_detect_narrowing_range_loss():
    table = MaterialTable("a", (10.0, 20.0), (1.0, 2.0), "mm^-1")
    basis = MaterialBasis(
        (table, replace(table, name="b", coefficients=(2.0, 4.0))), "declared fixed-density basis"
    )
    host = basis.to_host((10.0, 20.0), edge_side="above", dtype="float64")
    np.testing.assert_array_equal(host, [[1.0, 2.0], [2.0, 4.0]])
    assert host.flags.c_contiguous and host.dtype == np.float64
    spectrum = Spectrum((10.0, 20.0), (2.0, 3.0), "density", quadrature_weights_kev=(5.0, 7.0))
    np.testing.assert_array_equal(spectrum.to_host(dtype="float32"), [10.0, 21.0])
    with pytest.raises(ContractError, match="range"):
        replace(table, coefficients=(1e-60, 1e-60)).to_host((10.0,), edge_side="above")


def test_declared_mass_mixture_keeps_components_and_applies_density_once():
    edge = MaterialTable(
        "edge",
        (10.0, 20.0, 20.0, 40.0),
        (4.0, 2.0, 8.0, 4.0),
        "cm^2/g",
        FIXTURE,
        density_g_cm3=99.0,
    )
    flat = MaterialTable("flat", (10.0, 40.0), (2.0, 2.0), "cm^2/g", density_g_cm3=0.01)
    mixture = MassFractionMixture("declared mixture", (edge, flat), (0.25, 0.75), 2.0)
    assert mixture.component_provenance == (FIXTURE, None)
    assert mixture.tables[0] is edge
    assert mixture.at_energies((20.0,), edge_side="below") == pytest.approx((0.4,))
    assert mixture.at_energies((20.0,), edge_side="above") == pytest.approx((0.7,))
    assert mixture.bin_averages(
        (10.0, 40.0), weighting="uniform-energy", quadrature_order=32
    ) == pytest.approx(((0.25 * 200 * math.log(2) / 30 + 0.75 * 2) * 0.2,), rel=3e-14)
    basis = MaterialBasis((mixture,), "explicit homogeneous mass mixture at supplied density")
    np.testing.assert_allclose(basis.to_host((20.0,), edge_side="above"), [[0.7]], rtol=1e-7)
    with pytest.raises(ContractError, match="sum to one"):
        replace(mixture, mass_fractions=(0.5, 0.75))
    with pytest.raises(ContractError, match=r"cm\^2/g"):
        replace(mixture, tables=(replace(edge, units="mm^-1", density_g_cm3=None), flat))


@pytest.mark.parametrize("transpose", [False, True])
def test_separable_zero_edge_operator_matches_dense_matrix(transpose, warp_device):
    with cpu_test_context():
        h, w = 4, 5
        vertical, horizontal = (0.1, 0.3, 0.4), (0.2, 0.3, 0.1)
        spec = SeparableBlurSpec(h, w, vertical, horizontal)
        matrix = np.zeros((h * w, h * w))
        for row in range(h):
            for col in range(w):
                for i, vi in enumerate(vertical):
                    for j, hj in enumerate(horizontal):
                        r, c = row + i - 1, col + j - 1
                        if 0 <= r < h and 0 <= c < w:
                            matrix[row * w + col, r * w + c] += vi * hj
        values = np.linspace(-3, 5, h * w, dtype=np.float32)
        source = wp.array(values, dtype=wp.float32, device=warp_device)
        out = wp.empty_like(source)
        workspace = prepare_blur(spec, device=warp_device)
        if transpose:
            blur_transpose(source, out_grad_signal=out, workspace=workspace)
        else:
            blur(source, out_signal=out, workspace=workspace)
        expected = (matrix.T if transpose else matrix) @ values
        np.testing.assert_allclose(out.numpy(), expected, rtol=2e-7, atol=3e-8)
        assert workspace.scratch_bytes == workspace.detector.scratch_bytes + 8 * (h * w + 6)


def test_separable_intermediate_retains_representable_tail_in_both_orders(warp_device):
    with cpu_test_context():
        source = wp.array([1e-20], dtype=wp.float32, device=warp_device)
        output = wp.empty_like(source)
        workspace = prepare_blur(SeparableBlurSpec(1, 1, (1e60,), (1e-60,)), device=warp_device)
        blur(source, out_signal=output, workspace=workspace)
        np.testing.assert_array_equal(output.numpy(), source.numpy())
        blur_transpose(source, out_grad_signal=output, workspace=workspace)
        np.testing.assert_array_equal(output.numpy(), source.numpy())


@pytest.mark.parametrize("sharing", list(product((False, True), repeat=2)))
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("zero_parameter", ["weights", "response"])
def test_shaped_no_copy_log_signal_and_all_products(
    sharing, precision, zero_parameter, warp_device
):
    with cpu_test_context():
        m, k, p = 2, 3, 4
        sw, sr = sharing
        dtype = wp.float64 if precision == "float64" else wp.float32
        spec = SpectralSpec(
            m,
            k,
            shared_weights=sw,
            shared_response=sr,
            precision=precision,
            output_domain="log",
            active_weights=True,
            active_response=True,
            active_coefficients=True,
        )
        workspace = prepare_spectral(spec, max_pixels=p, device=warp_device)
        paths = wp.array(
            [[1.0, 50.0, 1200.0, 1700.0], [2.0, 5.0, 0.0, 1.0]], dtype=dtype, device=warp_device
        )
        mu = wp.array([[0.5, 1.0, 2.0], [0.25, 0.5, 0.75]], dtype=wp.float32, device=warp_device)
        weight_values = [3.0, 2.0, 0.0 if zero_parameter == "weights" else 3.0]
        response_values = [4.0, 0.5, 0.0 if zero_parameter == "response" else 2.0]
        weights = wp.array(
            weight_values if sw else [[value] * p for value in weight_values],
            dtype=wp.float32,
            device=warp_device,
        )
        response = wp.array(
            response_values if sr else [[value] * p for value in response_values],
            dtype=wp.float32,
            device=warp_device,
        )
        seed = wp.array([1.0, -2.0, 0.0, 3.0], dtype=dtype, device=warp_device)
        output = wp.empty(p, dtype=dtype, device=warp_device)
        gp = wp.empty_like(paths)
        gw = wp.empty(weights.shape, dtype=wp.float64 if sw else wp.float32, device=warp_device)
        gr = wp.empty(response.shape, dtype=wp.float64 if sr else wp.float32, device=warp_device)
        gc = wp.empty(mu.shape, dtype=wp.float64, device=warp_device)
        pointers = [x.ptr for x in (paths, mu, weights, response)]
        spectral_signal(paths, mu, weights, response, out_mean=output, workspace=workspace)
        spectral_vjp(
            paths,
            mu,
            weights,
            response,
            cotangent=seed,
            out_grad_paths=gp,
            out_grad_weights=gw,
            out_grad_response=gr,
            out_grad_coefficients=gc,
            workspace=workspace,
        )
        # Decimal computes its own unrounded linear mean and all derivatives,
        # so the oracle survives depths beyond either floating-point exp range.
        reference = spectral_reference(
            *(tuple(map(float, x.numpy().reshape(-1))) for x in (paths, mu, weights, response)),
            materials=m,
            energies=k,
            seed=tuple(map(float, seed.numpy())),
            shared_weights=sw,
            shared_response=sr,
        )
        expected_mean = [float(value.ln()) for value in reference.mean]
        np.testing.assert_allclose(
            output.numpy(), expected_mean, rtol=2e-7 if precision == "float32" else 3e-14
        )
        for actual, expected, shared in (
            (gp, reference.grad_paths, False),
            (gw, reference.grad_weights, sw),
            (gr, reference.grad_response, sr),
            (gc, reference.grad_coefficients, True),
        ):
            if actual is gc or shared:
                # Shared derivatives need the individual pixel normalization;
                # compute that oracle one pixel at a time and sum afterwards.
                accum = [0.0] * actual.size
                for pixel in range(p):
                    exact = spectral_reference(
                        tuple(float(x) for x in paths.numpy()[:, pixel]),
                        tuple(float(x) for x in mu.numpy().reshape(-1)),
                        tuple(
                            float(x) for x in (weights.numpy() if sw else weights.numpy()[:, pixel])
                        ),
                        tuple(
                            float(x)
                            for x in (response.numpy() if sr else response.numpy()[:, pixel])
                        ),
                        materials=m,
                        energies=k,
                        seed=(float(seed.numpy()[pixel]),),
                    )
                    grad = (
                        exact.grad_coefficients
                        if actual is gc
                        else exact.grad_weights
                        if actual is gw
                        else exact.grad_response
                    )
                    accum = [a + float(b / exact.mean[0]) for a, b in zip(accum, grad, strict=True)]
                oracle = accum
            else:
                oracle = [
                    float(value / reference.mean[index % p]) for index, value in enumerate(expected)
                ]
            np.testing.assert_allclose(
                actual.numpy().reshape(-1),
                oracle,
                rtol=2e-6 if actual.dtype == wp.float32 else 5e-12,
                atol=1e-300,
            )
        assert pointers == [x.ptr for x in (paths, mu, weights, response)]


@pytest.mark.parametrize("sharing", list(product((False, True), repeat=2)))
@pytest.mark.parametrize("bins", ["single", "tied", "separated", "zero-weight", "zero-response"])
def test_large_depth_log_products_match_closed_form(bins, sharing, warp_device):
    sw, sr = sharing
    k, p = (1 if bins == "single" else 4), 4
    path_values = np.array([[1e3, 1e16, 1e20, 1e28]], dtype=np.float64)
    path_values = np.concatenate((path_values, 0.5 * path_values))
    coefficients = np.repeat(np.array([[1.0], [0.5]], dtype=np.float32), k, axis=1)
    relative_attenuation = np.ones((k, 1))
    if bins == "separated":
        coefficients[:, 2] *= 2
        # This bin's additional depth is at least 1250. Its value and every
        # derivative round to zero even after multiplying the original factors.
        relative_attenuation[2] = 0
    weight_values = np.array([2.0, 5.0, 7.0, 11.0][:k], dtype=np.float32)
    response_values = np.array([3.0, 13.0, 17.0, 19.0][:k], dtype=np.float32)
    if bins == "zero-weight":
        weight_values[1] = 0
    if bins == "zero-response":
        response_values[3] = 0
    if not sw:
        weight_values = weight_values[:, None] * np.arange(1, p + 1, dtype=np.float32)
    if not sr:
        response_values = response_values[:, None] * np.arange(2, p + 2, dtype=np.float32)
    seed_values = np.array([1.0, -2.0, 0.25, 3.0])

    def array(values, dtype):
        return wp.array(values, dtype=dtype, device=warp_device)

    paths = array(path_values, wp.float64)
    mu = array(coefficients, wp.float32)
    weights = array(weight_values, wp.float32)
    response = array(response_values, wp.float32)
    seed = array(seed_values, wp.float64)
    mean = wp.empty(p, dtype=wp.float64, device=warp_device)
    gp = wp.empty_like(paths)
    gw = wp.empty(weights.shape, dtype=wp.float64 if sw else wp.float32, device=warp_device)
    gr = wp.empty(response.shape, dtype=wp.float64 if sr else wp.float32, device=warp_device)
    gc = wp.empty(mu.shape, dtype=wp.float64, device=warp_device)
    workspace = prepare_spectral(
        SpectralSpec(
            2,
            k,
            precision="float64",
            output_domain="log",
            shared_weights=sw,
            shared_response=sr,
            active_weights=True,
            active_response=True,
            active_coefficients=True,
        ),
        max_pixels=p,
        device=warp_device,
    )
    spectral_signal(paths, mu, weights, response, out_mean=mean, workspace=workspace)
    spectral_vjp(
        paths,
        mu,
        weights,
        response,
        cotangent=seed,
        out_grad_paths=gp,
        out_grad_weights=gw,
        out_grad_response=gr,
        out_grad_coefficients=gc,
        workspace=workspace,
    )

    # Every contributing bin has the same depth A0 + A1/2. The exponential
    # cancels algebraically, leaving rational nuisance derivatives, including
    # nonzero one-sided derivatives at a zero weight or response.
    w = weight_values.astype(np.float64).reshape(k, 1 if sw else p)
    r = response_values.astype(np.float64).reshape(k, 1 if sr else p)
    products = np.broadcast_to(w * r * relative_attenuation, (k, p))
    normalizer = products.sum(axis=0)
    expected_weights = seed_values * r * relative_attenuation / normalizer
    expected_response = seed_values * w * relative_attenuation / normalizer
    expected_coefficients = -(path_values * seed_values) @ (products / normalizer).T
    np.testing.assert_allclose(
        mean.numpy(), np.log(normalizer) - (path_values[0] + 0.5 * path_values[1]), rtol=3e-15
    )
    np.testing.assert_allclose(gp.numpy(), -np.array([[1.0], [0.5]]) * seed_values, rtol=3e-14)
    for actual, expected, shared in (
        (gw, expected_weights, sw),
        (gr, expected_response, sr),
    ):
        np.testing.assert_allclose(
            actual.numpy(),
            expected.sum(axis=1) if shared else expected,
            rtol=3e-14 if shared else 2e-6,
            atol=0,
        )
    np.testing.assert_allclose(gc.numpy(), expected_coefficients, rtol=3e-14, atol=0)
    workspace.check_status()


def test_log_zero_primal_and_undefined_reverse_are_explicit(warp_device):
    with cpu_test_context():
        spec = SpectralSpec(1, 1, output_domain="log")
        workspace = prepare_spectral(spec, max_pixels=1, device=warp_device)

        def a(values):
            return wp.array(values, dtype=wp.float32, device=warp_device)

        paths, coefficients, weights, response = (a(v) for v in ([1.0], [1.0], [0.0], [1.0]))
        mean, gradient = wp.empty_like(paths), wp.empty_like(paths)
        spectral_signal(paths, coefficients, weights, response, out_mean=mean, workspace=workspace)
        assert mean.numpy()[0] == -np.inf
        with pytest.raises(NumericalError):
            spectral_vjp(
                paths,
                coefficients,
                weights,
                response,
                cotangent=a([1.0]),
                out_grad_paths=gradient,
                workspace=workspace,
            )
        workspace.check_status()


def test_finite_log_mean_overflow_is_distinct_from_log_zero(warp_device):
    with cpu_test_context():
        workspace = prepare_spectral(
            SpectralSpec(1, 1, output_domain="log"), max_pixels=1, device=warp_device
        )

        def a(values):
            return wp.array(values, dtype=wp.float32, device=warp_device)

        output = wp.empty(1, dtype=wp.float32, device=warp_device)
        with pytest.raises(NumericalError):
            spectral_signal(
                a([1e38]), a([1e30]), a([1.0]), a([1.0]), out_mean=output, workspace=workspace
            )
        workspace.check_status()


def test_float32_arithmetic_explicitly_gives_up_deep_tail_rescue(warp_device):
    with cpu_test_context():

        def a(values):
            return wp.array(values, dtype=wp.float32, device=warp_device)

        paths, mu, weights, response = (a(v) for v in ([120.0], [1.0], [1e30], [1e10]))
        output = wp.empty(1, dtype=wp.float32, device=warp_device)
        strict = prepare_spectral(SpectralSpec(1, 1), max_pixels=1, device=warp_device)
        spectral_signal(paths, mu, weights, response, out_mean=output, workspace=strict)
        assert output.numpy()[0] > 1e-13
        fast = prepare_spectral(
            SpectralSpec(1, 1, arithmetic="float32"), max_pixels=1, device=warp_device
        )
        spectral_signal(paths, mu, weights, response, out_mean=output, workspace=fast)
        assert output.numpy()[0] == 0


@pytest.mark.parametrize("sw,sr", list(product((False, True), repeat=2)))
def test_optional_float32_arithmetic_error_against_decimal_reference(warp_device, sw, sr):
    with cpu_test_context():
        m, k, p = 3, 7, 5
        spec = SpectralSpec(
            m,
            k,
            arithmetic="float32",
            shared_weights=sw,
            shared_response=sr,
            active_weights=True,
            active_response=True,
            active_coefficients=True,
        )
        workspace = prepare_spectral(spec, max_pixels=p, device=warp_device)

        def a(values):
            return wp.array(values, dtype=wp.float32, device=warp_device)

        paths = a([0.0, 1.0, 20.0, 50.0, 120.0] * m)
        mu = a([0.005 + i * 0.004 for i in range(m * k)])
        weights, response, seed = (
            a([1.0 + i % k for i in range(k * (1 if sw else p))]),
            a([0.1 + (i % k) * 0.1 for i in range(k * (1 if sr else p))]),
            a([1.0, -3.0, 0.0, 0.1, 2.0]),
        )
        reference = spectral_reference(
            *(tuple(map(float, x.numpy())) for x in (paths, mu, weights, response)),
            materials=m,
            energies=k,
            seed=tuple(map(float, seed.numpy())),
            shared_weights=sw,
            shared_response=sr,
        )
        output, gp, gw, gr, gc = (
            wp.empty(n, dtype=wp.float32, device=warp_device)
            for n in (p, m * p, weights.size, response.size, m * k)
        )
        spectral_signal(paths, mu, weights, response, out_mean=output, workspace=workspace)
        spectral_vjp(
            paths,
            mu,
            weights,
            response,
            cotangent=seed,
            out_grad_paths=gp,
            out_grad_weights=gw,
            out_grad_response=gr,
            out_grad_coefficients=gc,
            workspace=workspace,
        )
        for actual, expected in zip(
            (output, gp, gw, gr, gc),
            (
                reference.mean,
                reference.grad_paths,
                reference.grad_weights,
                reference.grad_response,
                reference.grad_coefficients,
            ),
            strict=True,
        ):
            np.testing.assert_allclose(
                actual.numpy(), tuple(map(float, expected)), rtol=8e-6, atol=1e-30
            )
        assert SpectralSpec(1, 1).arithmetic == "strict"
        with pytest.raises(ContractError, match="requires"):
            replace(spec, precision="float64")


def test_explicit_device_cast_preserves_shape_and_reports_overflow(warp_device):
    with cpu_test_context():
        workspace = prepare_detector(max_pixels=4, device=warp_device)
        values = wp.array([[1.0, -2.0], [1e-60, 3.0]], dtype=wp.float64, device=warp_device)
        output = wp.empty((2, 2), dtype=wp.float32, device=warp_device)
        cast_signal(values, out_signal=output, workspace=workspace)
        np.testing.assert_array_equal(output.numpy(), [[1.0, -2.0], [0.0, 3.0]])
        widened = wp.empty((2, 2), dtype=wp.float64, device=warp_device)
        cast_signal(output, out_signal=widened, workspace=workspace)
        np.testing.assert_array_equal(widened.numpy(), output.numpy())
        values.fill_(1e40)
        with pytest.raises(NumericalError):
            cast_signal(values, out_signal=output, workspace=workspace)
        workspace.check_status()
        values.fill_(4.0)
        cast_signal(values, out_signal=output, workspace=workspace)
        np.testing.assert_array_equal(output.numpy(), np.full((2, 2), 4.0))


def test_checked_spectral_input_error_allows_retry_but_pending_error_stays_sticky(warp_device):
    with cpu_test_context():
        workspace = prepare_spectral(SpectralSpec(1, 1), max_pixels=1, device=warp_device)

        def a(values):
            return wp.array(values, dtype=wp.float32, device=warp_device)

        paths, coefficients, weights, response = (a(v) for v in ([-1.0], [1.0], [1.0], [1.0]))
        output = wp.full(1, 99.0, dtype=wp.float32, device=warp_device)
        with pytest.raises(ContractError):
            spectral_signal(
                paths, coefficients, weights, response, out_mean=output, workspace=workspace
            )
        assert output.numpy()[0] == 99.0
        paths.fill_(1.0)
        spectral_signal(
            paths, coefficients, weights, response, out_mean=output, workspace=workspace
        )
        assert output.numpy()[0] == pytest.approx(math.exp(-1), rel=1e-7)
        workspace._status.fill_(2)
        with pytest.raises(NumericalError):
            spectral_signal(
                paths, coefficients, weights, response, out_mean=output, workspace=workspace
            )
        with pytest.raises(NumericalError):
            workspace.check_status()
        with pytest.raises(NumericalError):
            workspace.consume_status()
        workspace.check_status()


def test_direct_shaped_probability_validation_reports_domain_error_and_allows_retry(warp_device):
    with cpu_test_context():
        workspace = prepare_spectral(
            SpectralSpec(1, 2, shared_weights=False, shared_response=False),
            max_pixels=3,
            device=warp_device,
        )
        coefficients = wp.array([[0.1, 0.2]], dtype=wp.float32, device=warp_device)
        weights = wp.full((2, 3), 2.0, dtype=wp.float32, device=warp_device)
        response = wp.array(
            [[0.0, 0.5, 1.0], [1.0, 0.75, 0.25]], dtype=wp.float32, device=warp_device
        )
        workspace.validate_inputs(
            coefficients, weights, response, pixels=3, probability_response=True
        )
        np.testing.assert_array_equal(response.numpy(), [[0.0, 0.5, 1.0], [1.0, 0.75, 0.25]])
        response.fill_(1.5)
        workspace.validate_inputs(coefficients, weights, response, pixels=3)
        with pytest.raises(ContractError):
            workspace.validate_inputs(
                coefficients, weights, response, pixels=3, probability_response=True
            )
        workspace.check_status()
        response.fill_(0.8)
        workspace.validate_inputs(
            coefficients, weights, response, pixels=3, probability_response=True
        )


@pytest.mark.parametrize("compound", [False, True])
def test_failed_poisson_realisation_stays_pending_until_explicit_consumption(compound, warp_device):
    with cpu_test_context():
        pixels = 1024
        workspace = prepare_detector(max_pixels=pixels, device=warp_device)
        means = wp.full(pixels, 10.0, dtype=wp.float32, device=warp_device)
        output = wp.empty(pixels, dtype=wp.float32 if compound else wp.uint64, device=warp_device)
        scores = wp.array([1.0], dtype=wp.float32, device=warp_device)
        identity = ObservationIdentity(0, 0, draw_budget=1)

        def draw(*, validate=True):
            if compound:
                sample_compound_poisson(
                    means,
                    scores,
                    energies=1,
                    shared_scores=True,
                    out_signal=output,
                    identity=identity,
                    workspace=workspace,
                    validate=validate,
                )
            else:
                sample_poisson_counts(
                    means,
                    out_counts=output,
                    identity=identity,
                    workspace=workspace,
                    validate=validate,
                )

        with pytest.raises(NumericalError, match="budget"):
            draw()
        failed_output = output.numpy().copy()
        means.fill_(0.0)
        with pytest.raises(NumericalError, match="budget"):
            draw()
        np.testing.assert_array_equal(output.numpy(), failed_output)
        with pytest.raises(NumericalError, match="budget"):
            workspace.check_status()
        with pytest.raises(NumericalError, match="budget"):
            workspace.consume_status()
        draw()
        np.testing.assert_array_equal(output.numpy(), np.zeros(pixels))
        means.fill_(10.0)
        draw(validate=False)
        with pytest.raises(NumericalError, match="budget"):
            workspace.check_status()
        with pytest.raises(NumericalError, match="budget"):
            workspace.consume_status()
        workspace.check_status()
