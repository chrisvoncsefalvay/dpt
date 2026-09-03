# pytest.approx has intentionally dynamic third-party stubs.
# pyright: reportUnknownMemberType=false
"""Independent CUDA contracts for response, calibration and observations."""

import hashlib
import importlib
import math
from dataclasses import replace
from typing import Any, cast

import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.detector import (
    BlurSpec,
    CalibrationSpec,
    ObservationIdentity,
    add_gaussian_read_noise,
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
from dpt.validation.detector import (
    compound_poisson_moments,
    matrix_product,
    spatial_response_matrix,
)

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu
FIXTURE = Provenance(
    source="analytic:h=(0.125,0.5,0.375), zero extension",
    sha256=hashlib.sha256(b"h=(0.125,0.5,0.375); boundary=zero").hexdigest(),
    rights="Original mathematical fixture; Apache-2.0",
    description="Analytic stencil, not detector data",
)


def array(values: list[float] | tuple[float, ...]) -> Any:
    return wp.array(values, dtype=wp.float32, device="cuda:0")


def values(device: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in device.numpy())


def test_calibration_values_and_constrained_nuisance_cotangents() -> None:
    workspace = prepare_detector(max_pixels=3)
    spec = CalibrationSpec("ADU", active_exposure=True, active_offset=True)
    mean, gain, exposure, offset = (
        array((0.0, 2.0, 10.0)),
        array((3.0,)),
        array((4.0,)),
        array((-2.0,)),
    )
    seed = array((1.0, -2.0, 0.5))
    signal, grad_mean, grad_exposure, grad_offset = (
        array((0.0, 0.0, 0.0)),
        array((0.0, 0.0, 0.0)),
        array((0.0,)),
        array((0.0,)),
    )
    calibrate(mean, gain, exposure, offset, out_signal=signal, spec=spec, workspace=workspace)
    assert values(signal) == (-2.0, 22.0, 118.0)
    calibration_vjp(
        mean,
        gain,
        exposure,
        offset,
        cotangent=seed,
        out_grad_mean=grad_mean,
        out_grad_exposure=grad_exposure,
        out_grad_offset=grad_offset,
        spec=spec,
        workspace=workspace,
    )
    assert values(grad_mean) == (12.0, -24.0, 6.0)
    assert values(grad_exposure) == (3.0,)
    assert values(grad_offset) == (-0.5,)


def test_invalid_gain_rejected_before_destination_write() -> None:
    workspace = prepare_detector(max_pixels=1)
    output = array((999.0,))
    with pytest.raises(ContractError):
        calibrate(
            array((1.0,)),
            array((0.0,)),
            array((1.0,)),
            array((0.0,)),
            out_signal=output,
            spec=CalibrationSpec("ADU"),
            workspace=workspace,
        )
    assert values(output) == (999.0,)


def test_spatial_forward_and_transpose_match_independent_dense_matrix() -> None:
    spec = BlurSpec(2, 3, 1, 3, (0.125, 0.5, 0.375), FIXTURE)
    workspace = prepare_blur(spec)
    source, seed = array((1.0, -2.0, 3.0, 4.0, 0.0, -1.0)), array((-2.0, 1.0, 0.5, 3.0, 4.0, -0.25))
    output, gradient = wp.empty_like(source), wp.empty_like(seed)
    blur(source, out_signal=output, workspace=workspace)
    blur_transpose(seed, out_grad_signal=gradient, workspace=workspace)
    matrix = spatial_response_matrix(2, 3, 1, 3, spec.weights)
    assert values(output) == pytest.approx(matrix_product(matrix, values(source)), rel=2e-6)
    assert values(gradient) == pytest.approx(
        matrix_product(matrix, values(seed), transpose=True), rel=2e-6
    )
    left = math.fsum(a * b for a, b in zip(values(seed), values(output), strict=True))
    right = math.fsum(a * b for a, b in zip(values(source), values(gradient), strict=True))
    assert left == pytest.approx(right, rel=2e-6)


def test_observation_replay_is_independent_of_pixel_batching() -> None:
    count = 513
    workspace = prepare_detector(max_pixels=count)
    means = array([7.0 + p % 13 for p in range(count)])
    identity = ObservationIdentity(0x123456789ABCDEF0, 9)
    whole = wp.empty(count, dtype=wp.uint64, device="cuda:0")
    sample_poisson_counts(means, out_counts=whole, identity=identity, workspace=workspace)
    first, second = (
        wp.empty(257, dtype=wp.uint64, device="cuda:0"),
        wp.empty(256, dtype=wp.uint64, device="cuda:0"),
    )
    sample_poisson_counts(means[:257], out_counts=first, identity=identity, workspace=workspace)
    sample_poisson_counts(
        means[257:],
        out_counts=second,
        identity=replace(identity, pixel_offset=257),
        workspace=workspace,
    )
    assert tuple(whole.numpy()) == (*first.numpy(), *second.numpy())


@pytest.mark.parametrize("rate", [0.0, 0.1, 9.999, 10.0, 100.0, 1e6, 1e9])
def test_poisson_mean_and_variance_match_independent_moments(rate: float) -> None:
    count = 65536
    workspace = prepare_detector(max_pixels=count)
    output = wp.empty(count, dtype=wp.uint64, device="cuda:0")
    sample_poisson_counts(
        array([rate] * count),
        out_counts=output,
        identity=ObservationIdentity(7391, 13),
        workspace=workspace,
    )
    samples = tuple(float(value) for value in output.numpy())
    observed_mean = math.fsum(samples) / count
    observed_variance = math.fsum((value - observed_mean) ** 2 for value in samples) / (count - 1)
    if rate == 0:
        assert observed_mean == observed_variance == 0
    else:
        mean, variance, fourth = compound_poisson_moments((rate,), (1.0,))
        # Fixed broad stochastic acceptance bounds, not fabricated significance.
        assert abs(observed_mean - mean) < 7 * math.sqrt(variance / count)
        assert abs(observed_variance - variance) < 7 * math.sqrt((fourth - variance**2) / count)


def test_compound_mean_and_variance_use_per_photon_energy_scores() -> None:
    count = 65536
    workspace = prepare_detector(max_pixels=count)
    rates, scores = array([2.0] * count + [3.0] * count), array((40.0, 80.0))
    output = wp.empty(count, dtype=wp.float32, device="cuda:0")
    sample_compound_poisson(
        rates,
        scores,
        energies=2,
        shared_scores=True,
        out_signal=output,
        identity=ObservationIdentity(211, 17),
        workspace=workspace,
    )
    observed = values(output)
    mean, variance, fourth = compound_poisson_moments((2.0, 3.0), (40.0, 80.0))
    observed_mean = math.fsum(observed) / count
    observed_variance = math.fsum((value - observed_mean) ** 2 for value in observed) / (count - 1)
    assert abs(observed_mean - mean) < 7 * math.sqrt(variance / count)
    assert abs(observed_variance - variance) < 7 * math.sqrt((fourth - variance**2) / count)


def test_budget_exhaustion_invalidates_realisation_and_unchecked_status_persists() -> None:
    workspace = prepare_detector(max_pixels=65536)
    output = wp.empty(65536, dtype=wp.uint64, device="cuda:0")
    with pytest.raises(NumericalError, match="budget"):
        sample_poisson_counts(
            array([10.0] * 65536),
            out_counts=output,
            identity=ObservationIdentity(0, 0, draw_budget=1),
            workspace=workspace,
        )
    with pytest.raises(NumericalError, match="budget"):
        workspace.check_status()
    workspace.clear_status()
    workspace.check_status()


def test_read_noise_is_separate_and_never_clips_signed_signal() -> None:
    workspace = prepare_detector(max_pixels=2)
    output = array((0.0, 0.0))
    add_gaussian_read_noise(
        array((-4.0, 3.0)),
        standard_deviation=0,
        out_signal=output,
        identity=ObservationIdentity(1, 0),
        workspace=workspace,
    )
    assert values(output) == (-4.0, 3.0)


def test_binary64_shared_scalar_gradient_retains_log_chart_range() -> None:
    workspace = prepare_detector(max_pixels=1)
    gradient = wp.empty(1, dtype=wp.float64, device="cuda:0")
    mean, gain, exposure, offset = array((1e-30,)), array((1e30,)), array((1.0,)), array((0.0,))
    seed = array((1e-30,))
    calibration_vjp(
        mean,
        gain,
        exposure,
        offset,
        cotangent=seed,
        out_grad_gain=gradient,
        spec=CalibrationSpec("signal", active_gain=True),
        workspace=workspace,
    )
    exact = values(mean)[0] * values(seed)[0]
    assert values(gradient)[0] == pytest.approx(exact, rel=1e-12, abs=0)
    assert values(gradient)[0] * values(gain)[0] == pytest.approx(1e-30, rel=2e-6, abs=0)


def test_fixed_stencil_keeps_positive_tails_before_bright_pixel_products() -> None:
    spec = BlurSpec(1, 2, 1, 3, (0.0, 0.0, 1e-50), FIXTURE)
    workspace = prepare_blur(spec)
    output = array((0.0, 0.0))
    source = array((0.0, 1e38))
    blur(source, out_signal=output, workspace=workspace)
    assert values(output)[0] == pytest.approx(1e-50 * values(source)[1], rel=2e-6, abs=0)
    assert values(output)[1] == 0.0


def test_reduction_launch_bound_cannot_outgrow_allocated_scratch() -> None:
    workspace = prepare_detector(max_pixels=1024, reduction_groups=2)
    assert workspace.reduction_groups == 2
    with pytest.raises(AttributeError):
        cast(Any, workspace).reduction_groups = 256
    assert workspace.reduction_groups == 2


def test_single_proposal_budget_does_not_truncate_small_rate_counts() -> None:
    workspace = prepare_detector(max_pixels=4096)
    output = wp.empty(4096, dtype=wp.uint64, device="cuda:0")
    sample_poisson_counts(
        array([9.0] * 4096),
        out_counts=output,
        identity=ObservationIdentity(0, 0, draw_budget=1),
        workspace=workspace,
    )
    observed = tuple(int(value) for value in output.numpy())
    assert max(observed) > 9
    assert abs(math.fsum(observed) / len(observed) - 9.0) < 0.5


@pytest.mark.parametrize("energy_offset", [0, 9, 2**31 - 3])
def test_unit_counts_and_compound_sampling_share_energy_domains(energy_offset: int) -> None:
    workspace = prepare_detector(max_pixels=257)
    means = array([2.0 + p % 30 for p in range(257)])
    counts = wp.empty(257, dtype=wp.uint64, device="cuda:0")
    compound = wp.empty(257, dtype=wp.float32, device="cuda:0")
    identity = ObservationIdentity(17, 29, energy_offset=energy_offset)
    sample_poisson_counts(means, out_counts=counts, identity=identity, workspace=workspace)
    sample_compound_poisson(
        means,
        array((1.0,)),
        energies=1,
        shared_scores=True,
        out_signal=compound,
        identity=identity,
        workspace=workspace,
    )
    assert tuple(int(value) for value in counts.numpy()) == tuple(
        int(value) for value in compound.numpy()
    )
