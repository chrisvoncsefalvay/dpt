"""CPU workspaces retain their device when Warp's ambient default is CUDA."""

import importlib
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
import pytest

from dpt.contracts import TrialDomainError
from dpt.detector import (
    CalibrationSpec,
    ObservationIdentity,
    SeparableBlurSpec,
    add_gaussian_read_noise,
    blur,
    blur_transpose,
    calibrate,
    calibration_vjp,
    cast_signal,
    prepare_blur,
    prepare_detector,
    sample_compound_poisson,
    sample_poisson_counts,
)
from dpt.geometry import RayBundleGeometry, RigidTransform
from dpt.material_projection import (
    material_projection_vjp,
    prepare_material_projection,
    project_material_paths,
)
from dpt.projection import ProjectionSpec
from dpt.spectral import SpectralSpec, prepare_spectral, spectral_signal, spectral_vjp
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


@contextmanager
def _owned_launches(monkeypatch: pytest.MonkeyPatch, device: str):
    owner = wp.get_device(device)
    recorded = []

    def checked(function):
        def launch(*args, **kwargs):
            selected = kwargs.get("device")
            assert selected is not None, "operator relied on Warp's ambient launch device"
            assert wp.get_device(selected) == owner
            recorded.append(selected)
            return function(*args, **kwargs)

        return launch

    with monkeypatch.context() as patch:
        patch.setattr(wp, "launch", checked(wp.launch))
        patch.setattr(wp, "launch_tiled", checked(wp.launch_tiled))
        yield
    assert recorded


def _ambient():
    # On CUDA hosts this selects CUDA even for the portable CPU fixture. Do not
    # use cpu_test_context: it would hide an unqualified validation launch.
    return wp.ScopedDevice("cuda:0" if wp.is_cuda_available() else "cpu")


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("execution", ["serial", "fused"])
def test_material_projection_validates_and_projects_on_its_explicit_device(
    integration: Literal["midpoint", "cell_gauss"],
    execution: Literal["serial", "fused"],
    warp_device: str,
    monkeypatch: pytest.MonkeyPatch,
):
    with _ambient():
        grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
        rays = RayBundleGeometry(((0.0, 0.0, -2.0),), ((0.0, 0.0, 2.0),))
        fields = wp.array([[0.125], [0.375]], dtype=wp.float32, device=warp_device)
        paths = wp.full((2, 1), 19.0, dtype=wp.float64, device=warp_device)
        seeds = wp.array([[2.0], [-3.0]], dtype=wp.float64, device=warp_device)
        gradient = wp.empty_like(fields)
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
        workspace = prepare_material_projection(
            grid,
            rays,
            ProjectionSpec(17, active_volume=True, precision="float64", integration=integration),
            materials=2,
            fields=fields,
            out_paths=paths,
            cotangent_paths=seeds,
            out_fields=gradient,
            device=warp_device,
            channel_execution=execution,
        )
        with _owned_launches(monkeypatch, warp_device):
            workspace.validate_inputs(seeds=True)
            project_material_paths(pose, workspace=workspace)
            material_projection_vjp(pose, workspace=workspace)
            np.testing.assert_allclose(paths.numpy(), [[0.125], [0.375]], rtol=2e-14)
            np.testing.assert_allclose(gradient.numpy(), [[2.0], [-3.0]], rtol=2e-6)
            fields.fill_(2.0)
            paths.fill_(19.0)
            with pytest.raises(TrialDomainError):
                project_material_paths(pose, workspace=workspace)
            np.testing.assert_array_equal(paths.numpy(), [[19.0], [19.0]])
            fields.fill_(0.25)
            project_material_paths(pose, workspace=workspace)
        np.testing.assert_allclose(paths.numpy(), [[0.25], [0.25]], rtol=2e-14)


@pytest.mark.parametrize("domain", ["linear", "log"])
def test_spectral_validation_and_all_products_use_the_explicit_device(
    domain: Literal["linear", "log"], warp_device: str, monkeypatch: pytest.MonkeyPatch
):
    with _ambient():
        paths = wp.array([[0.25, 0.5]], dtype=wp.float64, device=warp_device)
        coefficients = wp.array([[1.0, 2.0]], dtype=wp.float32, device=warp_device)
        weights = wp.array([10.0, 20.0], dtype=wp.float32, device=warp_device)
        response = wp.array([[0.5, 0.75], [0.25, 0.5]], dtype=wp.float32, device=warp_device)
        workspace = prepare_spectral(
            SpectralSpec(
                1,
                2,
                shared_response=False,
                precision="float64",
                output_domain=domain,
                active_weights=True,
                active_response=True,
                active_coefficients=True,
            ),
            max_pixels=2,
            device=warp_device,
        )
        output = wp.empty(2, dtype=wp.float64, device=warp_device)
        seed = wp.array([-1.0, 2.0], dtype=wp.float64, device=warp_device)
        grad_paths = wp.empty_like(paths)
        grad_weights = wp.empty(2, dtype=wp.float64, device=warp_device)
        grad_response = wp.empty_like(response)
        grad_coefficients = wp.empty((1, 2), dtype=wp.float64, device=warp_device)
        with _owned_launches(monkeypatch, warp_device):
            workspace.validate_inputs(
                coefficients, weights, response, pixels=2, probability_response=True
            )
            spectral_signal(
                paths, coefficients, weights, response, out_mean=output, workspace=workspace
            )
            spectral_vjp(
                paths,
                coefficients,
                weights,
                response,
                cotangent=seed,
                out_grad_paths=grad_paths,
                out_grad_weights=grad_weights,
                out_grad_response=grad_response,
                out_grad_coefficients=grad_coefficients,
                workspace=workspace,
            )
        path = np.array([0.25, 0.5])
        mu = np.array([1.0, 2.0])
        transmission = np.exp(-mu[:, None] * path[None, :])
        r = np.array([[0.5, 0.75], [0.25, 0.5]])
        bins = np.array([10.0, 20.0])[:, None] * r * transmission
        mean = bins.sum(axis=0)
        scaled_seed = np.array([-1.0, 2.0]) / (mean if domain == "log" else 1.0)
        np.testing.assert_allclose(output.numpy(), np.log(mean) if domain == "log" else mean)
        np.testing.assert_allclose(
            grad_paths.numpy(), [-np.sum(mu[:, None] * bins * scaled_seed, axis=0)], rtol=2e-14
        )
        np.testing.assert_allclose(
            grad_weights.numpy(), np.sum(r * transmission * scaled_seed, axis=1), rtol=2e-14
        )
        np.testing.assert_allclose(
            grad_response.numpy(),
            np.array([10.0, 20.0])[:, None] * transmission * scaled_seed,
            rtol=2e-7,
        )
        np.testing.assert_allclose(
            grad_coefficients.numpy(), [-np.sum(bins * path * scaled_seed, axis=1)], rtol=2e-14
        )


def test_detector_conversion_calibration_blur_and_sampling_use_the_explicit_device(
    warp_device: str, monkeypatch: pytest.MonkeyPatch
):
    with _ambient():
        workspace = prepare_detector(max_pixels=2, device=warp_device)
        spatial = prepare_blur(
            SeparableBlurSpec(1, 2, (1.0,), (0.25, 0.5, 0.25)), device=warp_device
        )
        wide = wp.array([2.0, 3.0], dtype=wp.float64, device=warp_device)
        source = wp.empty(2, dtype=wp.float32, device=warp_device)
        calibrated = wp.empty_like(source)
        output = wp.empty_like(source)
        gain = wp.array([2.0], dtype=wp.float32, device=warp_device)
        exposure = wp.array([1.0], dtype=wp.float32, device=warp_device)
        offset = wp.array([0.5], dtype=wp.float32, device=warp_device)
        seed = wp.array([-1.0, 2.0], dtype=wp.float32, device=warp_device)
        grad_mean = wp.empty_like(source)
        grad_gain = wp.empty(1, dtype=wp.float64, device=warp_device)
        grad_offset = wp.empty_like(grad_gain)
        spec = CalibrationSpec(active_gain=True, active_offset=True)
        zero_rates = wp.zeros(2, dtype=wp.float64, device=warp_device)
        counts = wp.empty(2, dtype=wp.uint64, device=warp_device)
        scores = wp.array([1.0], dtype=wp.float32, device=warp_device)
        with _owned_launches(monkeypatch, warp_device):
            cast_signal(wide, out_signal=source, workspace=workspace)
            calibrate(
                source,
                gain,
                exposure,
                offset,
                out_signal=calibrated,
                spec=spec,
                workspace=workspace,
            )
            calibration_vjp(
                source,
                gain,
                exposure,
                offset,
                cotangent=seed,
                out_grad_mean=grad_mean,
                out_grad_gain=grad_gain,
                out_grad_offset=grad_offset,
                spec=spec,
                workspace=workspace,
            )
            np.testing.assert_array_equal(grad_mean.numpy(), [-2.0, 4.0])
            np.testing.assert_array_equal(grad_gain.numpy(), [4.0])
            np.testing.assert_array_equal(grad_offset.numpy(), [1.0])
            blur(calibrated, out_signal=output, workspace=spatial)
            np.testing.assert_array_equal(output.numpy(), [3.875, 4.375])
            blur_transpose(seed, out_grad_signal=output, workspace=spatial)
            np.testing.assert_array_equal(output.numpy(), [0.0, 0.75])
            sample_poisson_counts(
                zero_rates,
                out_counts=counts,
                identity=ObservationIdentity(0, 0),
                workspace=workspace,
            )
            np.testing.assert_array_equal(counts.numpy(), [0, 0])
            sample_compound_poisson(
                zero_rates,
                scores,
                energies=1,
                shared_scores=True,
                out_signal=output,
                identity=ObservationIdentity(0, 0),
                workspace=workspace,
            )
            np.testing.assert_array_equal(output.numpy(), [0.0, 0.0])
            add_gaussian_read_noise(
                seed,
                standard_deviation=0.0,
                out_signal=output,
                identity=ObservationIdentity(0, 0),
                workspace=workspace,
            )
        np.testing.assert_array_equal(output.numpy(), [-1.0, 2.0])
