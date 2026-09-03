# pytest.approx has intentionally dynamic third-party stubs.
# pyright: reportUnknownMemberType=false
"""CUDA acceptance against independent high-precision spectral references."""

import hashlib
import importlib
from dataclasses import replace
from typing import Any, Literal

import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.materials import Provenance
from dpt.spectral import (
    SpectralSpec,
    prepare_spectral,
    spectral_bin_counts,
    spectral_signal,
    spectral_vjp,
)
from dpt.validation.spectral import spectral_reference

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu
FIXTURE = Provenance(
    source="analytic:test_spectral explicit arrays below",
    sha256=hashlib.sha256(b"A=(0,50,120;20,5,0); mu=(.02,.04;0,.1)").hexdigest(),
    rights="Original mathematical fixture; Apache-2.0",
    description="Analytic material-path fixture",
)


def array(values: tuple[float, ...] | list[float], dtype: Any = wp.float32) -> Any:
    return wp.array(values, dtype=dtype, device="cuda:0")


def exact_values(device: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in device.numpy())


def spec(shared_weights: bool = True, shared_response: bool = True) -> SpectralSpec:
    return SpectralSpec(
        2,
        2,
        (FIXTURE, FIXTURE),
        FIXTURE,
        FIXTURE,
        "keV",
        "analytic fixed basis",
        shared_weights=shared_weights,
        shared_response=shared_response,
        active_weights=True,
        active_response=True,
        active_coefficients=True,
    )


@pytest.mark.parametrize(
    "shared_weights,shared_response", [(True, True), (True, False), (False, True), (False, False)]
)
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_mean_and_all_cotangents_against_decimal(
    shared_weights: bool, shared_response: bool, precision: Literal["float32", "float64"]
) -> None:
    policy = replace(spec(shared_weights, shared_response), precision=precision)
    dtype = wp.float64 if precision == "float64" else wp.float32
    workspace = prepare_spectral(policy, max_pixels=3)
    paths = array((0.0, 50.0 + 1e-8, 120.0, 20.0, 5.0, 0.0), dtype)
    coefficients = array((0.02, 0.04, 0.0, 0.1))
    weights = array((0.0, 2.0) if shared_weights else (0.0, 1.0, 2.0, 3.0, 0.0, 4.0))
    response = array((40.0, 80.0) if shared_response else (40.0, 0.0, 10.0, 80.0, 2.0, 0.0))
    seed = array((1.0, -3.0, 0.5), dtype)
    output = wp.empty(3, dtype=dtype, device="cuda:0")
    targets = [wp.empty_like(value) for value in (paths, coefficients, weights, response)]
    reference = spectral_reference(
        *(exact_values(value) for value in (paths, coefficients, weights, response)),
        materials=2,
        energies=2,
        seed=exact_values(seed),
        shared_weights=shared_weights,
        shared_response=shared_response,
    )
    spectral_signal(paths, coefficients, weights, response, out_mean=output, workspace=workspace)
    spectral_vjp(
        paths,
        coefficients,
        weights,
        response,
        cotangent=seed,
        out_grad_paths=targets[0],
        out_grad_coefficients=targets[1],
        out_grad_weights=targets[2],
        out_grad_response=targets[3],
        workspace=workspace,
    )
    for actual, expected in zip(
        [output, *targets],
        [
            reference.mean,
            reference.grad_paths,
            reference.grad_coefficients,
            reference.grad_weights,
            reference.grad_response,
        ],
        strict=True,
    ):
        assert exact_values(actual) == pytest.approx(
            tuple(map(float, expected)),
            rel=2e-13 if actual.dtype == wp.float64 else 2e-6,
            abs=1e-35,
        )
    assert exact_values(seed) == (1.0, -3.0, 0.5)


@pytest.mark.parametrize("pixels", [0, 1, 255, 256, 257, 513])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_repeated_shared_reductions_cover_padding_and_empty_images(
    pixels: int, precision: Literal["float32", "float64"]
) -> None:
    policy = replace(
        spec(), materials=1, energies=1, coefficients_provenance=(FIXTURE,), precision=precision
    )
    dtype = wp.float64 if precision == "float64" else wp.float32
    workspace = prepare_spectral(policy, max_pixels=max(pixels, 513))
    paths, coefficients, weights, response = (
        array([0.0] * pixels, dtype),
        array((1.0,)),
        array((2.0,)),
        array((3.0,)),
    )
    seed, gradient = array([1.0] * pixels, dtype), array((-999.0,))
    for _ in range(2):
        spectral_vjp(
            paths,
            coefficients,
            weights,
            response,
            cotangent=seed,
            out_grad_weights=gradient,
            workspace=workspace,
        )
        assert exact_values(gradient) == (float(3 * pixels),)


def test_tail_products_do_not_use_underflowed_binary32_transmission() -> None:
    policy = replace(spec(), materials=1, energies=1, coefficients_provenance=(FIXTURE,))
    workspace = prepare_spectral(policy, max_pixels=1)
    paths, coefficients, weights, response = (
        array((120.0,)),
        array((1.0,)),
        array((1e30,)),
        array((1e10,)),
    )
    output, gradient = array((0.0,)), array((0.0,))
    spectral_signal(paths, coefficients, weights, response, out_mean=output, workspace=workspace)
    spectral_vjp(
        paths,
        coefficients,
        weights,
        response,
        cotangent=array((1.0,)),
        out_grad_paths=gradient,
        workspace=workspace,
    )
    reference = spectral_reference(
        *(exact_values(value) for value in (paths, coefficients, weights, response)),
        materials=1,
        energies=1,
    )
    assert exact_values(output) == pytest.approx(tuple(map(float, reference.mean)), rel=2e-6, abs=0)
    assert exact_values(gradient) == pytest.approx(
        tuple(map(float, reference.grad_paths)), rel=2e-6, abs=0
    )


def test_checked_failure_preserves_destination_and_aliasing_is_rejected() -> None:
    policy = replace(spec(), materials=1, energies=1, coefficients_provenance=(FIXTURE,))
    workspace = prepare_spectral(policy, max_pixels=1)
    paths, coefficients, weights, response = (
        array((-1.0,)),
        array((1.0,)),
        array((2.0,)),
        array((3.0,)),
    )
    output = array((123.0,))
    with pytest.raises(ContractError):
        spectral_signal(
            paths, coefficients, weights, response, out_mean=output, workspace=workspace
        )
    assert exact_values(output) == (123.0,)
    with pytest.raises(ContractError, match="overlaps"):
        spectral_signal(paths, coefficients, weights, response, out_mean=paths, workspace=workspace)


def test_output_overflow_is_reported_without_silent_clipping() -> None:
    policy = replace(spec(), materials=1, energies=1, coefficients_provenance=(FIXTURE,))
    workspace = prepare_spectral(policy, max_pixels=1)
    with pytest.raises(NumericalError):
        spectral_signal(
            array((0.0,)),
            array((1.0,)),
            array((1e30,)),
            array((1e30,)),
            out_mean=array((0.0,)),
            workspace=workspace,
        )


def test_detected_bin_means_use_probabilities_and_precede_photon_scores() -> None:
    policy = replace(spec(), materials=1, coefficients_provenance=(FIXTURE,))
    workspace = prepare_spectral(policy, max_pixels=2)
    output = array((0.0, 0.0, 0.0, 0.0))
    paths, coefficients, weights = array((0.0, 0.0)), array((0.02, 0.04)), array((100.0, 200.0))
    spectral_bin_counts(
        paths, coefficients, weights, array((0.5, 0.25)), out_bin_counts=output, workspace=workspace
    )
    assert exact_values(output) == (50.0, 50.0, 50.0, 50.0)
    with pytest.raises(ContractError):
        spectral_bin_counts(
            paths,
            coefficients,
            weights,
            array((40.0, 80.0)),
            out_bin_counts=output,
            workspace=workspace,
        )


def test_ambient_tape_and_wrong_stream_are_explicit_errors() -> None:
    policy = replace(spec(), materials=1, energies=1, coefficients_provenance=(FIXTURE,))
    workspace = prepare_spectral(policy, max_pixels=1)
    args = [array((1.0,)) for _ in range(4)]
    with wp.Tape(), pytest.raises(ContractError, match="ambient tape"):
        spectral_signal(*args, out_mean=array((0.0,)), workspace=workspace)
    with pytest.raises(ContractError, match="owning"):
        spectral_signal(
            *args, out_mean=array((0.0,)), workspace=workspace, stream=wp.Stream("cuda:0")
        )


@pytest.mark.parametrize(
    "materials,energies,pixels", [(2, 32, 129), (8, 64, 65536), (8, 64, 262144), (32, 64, 257)]
)
def test_spectral_launch_sizes_against_exact_input_decimal(
    materials: int, energies: int, pixels: int
) -> None:
    """Uniform pixels permit a one-pixel Decimal oracle at production launch sizes."""
    np: Any = importlib.import_module("numpy")
    policy = replace(
        spec(),
        materials=materials,
        energies=energies,
        coefficients_provenance=(FIXTURE,) * materials,
    )
    workspace = prepare_spectral(policy, max_pixels=pixels)
    coefficients_host = (
        (0.01 * (np.arange(materials)[:, None] + 1) / (1 + np.arange(energies)[None, :] / energies))
        .astype(np.float32)
        .ravel()
    )
    weights_host = np.full(energies, 1000 / energies, dtype=np.float32)
    reference = spectral_reference(
        (10.0,) * materials,
        tuple(map(float, coefficients_host)),
        tuple(map(float, weights_host)),
        (1.0,) * energies,
        materials=materials,
        energies=energies,
    )
    paths = wp.full(materials * pixels, 10.0, dtype=wp.float32, device="cuda:0")
    coefficients = wp.array(coefficients_host, dtype=wp.float32, device="cuda:0")
    weights = wp.array(weights_host, dtype=wp.float32, device="cuda:0")
    response = wp.ones(energies, dtype=wp.float32, device="cuda:0")
    seed = wp.ones(pixels, dtype=wp.float32, device="cuda:0")
    output = wp.empty(pixels, dtype=wp.float32, device="cuda:0")
    gradient = wp.empty_like(paths)
    coefficient_gradient = wp.empty_like(coefficients)
    spectral_signal(paths, coefficients, weights, response, out_mean=output, workspace=workspace)
    spectral_vjp(
        paths,
        coefficients,
        weights,
        response,
        cotangent=seed,
        out_grad_paths=gradient,
        out_grad_coefficients=coefficient_gradient,
        workspace=workspace,
    )
    np.testing.assert_allclose(output.numpy(), float(reference.mean[0]), rtol=2e-6, atol=1e-35)
    actual = gradient.numpy().reshape(materials, pixels)
    for material in range(materials):
        np.testing.assert_allclose(
            actual[material], float(reference.grad_paths[material]), rtol=2e-6, atol=1e-35
        )
    np.testing.assert_allclose(
        coefficient_gradient.numpy(),
        np.array([float(value) * pixels for value in reference.grad_coefficients]),
        rtol=2e-6,
        atol=1e-35,
    )


def test_public_input_validation_distinguishes_response_moments_from_probabilities() -> None:
    workspace = prepare_spectral(spec(), max_pixels=3)
    coefficients, weights, response = (
        array((0.02, 0.04, 0.0, 0.1)),
        array((2.0, 3.0)),
        array((40.0, 80.0)),
    )
    workspace.validate_inputs(coefficients, weights, response, pixels=3)
    with pytest.raises(ContractError):
        workspace.validate_inputs(
            coefficients, weights, response, pixels=3, probability_response=True
        )
    workspace.clear_status()
    with pytest.raises(ContractError, match="shape"):
        workspace.validate_inputs(coefficients, weights[:1], response, pixels=3)
    with pytest.raises(ContractError):
        workspace.validate_inputs(coefficients, weights, array((-1.0, 1.0)), pixels=3)


@pytest.mark.parametrize("shared", [False, True])
def test_wide_signal_retains_sub_float32_steps_and_optional_parameter_gradients(
    shared: bool,
) -> None:
    policy = replace(
        spec(shared, shared),
        materials=1,
        energies=1,
        coefficients_provenance=(FIXTURE,),
        precision="float64",
    )
    workspace = prepare_spectral(policy, max_pixels=2)
    paths = array((1.0, 1.0 + 1e-9), wp.float64)
    coefficients = array((1.0,))
    weights = array((1e6,) if shared else (1e6, 1e6))
    response = array((1.0,) if shared else (1.0, 1.0))
    output = wp.empty(2, dtype=wp.float64, device="cuda:0")
    workspace.validate_inputs(coefficients, weights, response, pixels=2, paths=paths)
    spectral_signal(paths, coefficients, weights, response, out_mean=output, workspace=workspace)
    values = exact_values(output)
    assert values[0] - values[1] == pytest.approx(0.00036787946, rel=2e-6)
    # Exercise the typed empty path destination while both parameter gradients stay FP32.
    weight_grad, response_grad = wp.empty_like(weights), wp.empty_like(response)
    spectral_vjp(
        paths,
        coefficients,
        weights,
        response,
        cotangent=array((1.0, -1.0), wp.float64),
        out_grad_weights=weight_grad,
        out_grad_response=response_grad,
        workspace=workspace,
    )
    reference = spectral_reference(
        *(exact_values(v) for v in (paths, coefficients, weights, response)),
        materials=1,
        energies=1,
        seed=(1.0, -1.0),
        shared_weights=shared,
        shared_response=shared,
    )
    assert exact_values(weight_grad) == pytest.approx(
        tuple(map(float, reference.grad_weights)), rel=2e-6
    )
    assert exact_values(response_grad) == pytest.approx(
        tuple(map(float, reference.grad_response)), rel=2e-6
    )
    bins = wp.empty(2, dtype=wp.float64, device="cuda:0")
    spectral_bin_counts(
        paths, coefficients, weights, response, out_bin_counts=bins, workspace=workspace
    )
    assert exact_values(bins) == values


def test_wide_boundaries_reject_mismatched_dtypes_and_invalid_inputs_without_writes() -> None:
    policy = replace(
        spec(), materials=1, energies=1, coefficients_provenance=(FIXTURE,), precision="float64"
    )
    workspace = prepare_spectral(policy, max_pixels=1)
    paths, coefficients, weights, response = (
        array((1.0,), wp.float64),
        array((1.0,)),
        array((2.0,)),
        array((3.0,)),
    )
    output = array((123.0,), wp.float64)
    with pytest.raises(ContractError):
        spectral_signal(
            array((1.0,)), coefficients, weights, response, out_mean=output, workspace=workspace
        )
    with pytest.raises(ContractError):
        spectral_signal(
            paths,
            array((1.0,), wp.float64),
            weights,
            response,
            out_mean=output,
            workspace=workspace,
        )
    with pytest.raises(ContractError):
        workspace.validate_inputs(
            coefficients, weights, response, pixels=1, paths=array((float("nan"),), wp.float64)
        )
    workspace.clear_status()
    with pytest.raises(ContractError):
        spectral_signal(
            array((-1.0,), wp.float64),
            coefficients,
            weights,
            response,
            out_mean=output,
            workspace=workspace,
        )
    assert exact_values(output) == (123.0,)
    with pytest.raises(ContractError, match="overlaps"):
        spectral_signal(paths, coefficients, weights, response, out_mean=paths, workspace=workspace)
    with wp.Tape(), pytest.raises(ContractError, match="ambient tape"):
        spectral_signal(
            paths, coefficients, weights, response, out_mean=output, workspace=workspace
        )
    with pytest.raises(ContractError, match="owning"):
        spectral_signal(
            paths,
            coefficients,
            weights,
            response,
            out_mean=output,
            workspace=workspace,
            stream=wp.Stream("cuda:0"),
        )
    # Deferred checking must still surface an unrepresentable parameter gradient.
    workspace.clear_status()
    spectral_vjp(
        paths,
        coefficients,
        weights,
        response,
        cotangent=array((1e308,), wp.float64),
        out_grad_weights=array((0.0,)),
        workspace=workspace,
        validate=False,
    )
    with pytest.raises(NumericalError):
        workspace.check_status()
