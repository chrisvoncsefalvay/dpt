"""Image VJPs accept singleton and broadcast seeds without changing caller storage."""

import importlib
from contextlib import nullcontext

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.detector import BlurSpec, CalibrationSpec, prepare_blur, prepare_detector
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.interop.torch import project_optical_depth, project_spectral_signal
from dpt.material_projection import prepare_material_projection
from dpt.projection import ProjectionSpec, prepare_projection
from dpt.spectral import SpectralSpec, prepare_spectral
from dpt.torch.operators import affine_calibrate, blur, transmit
from dpt.transmission import TransmissionSpec, prepare_transmission
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec
from helpers.torch_detector_oracles import blur_matrix
from helpers.torch_expected_transport_inputs import analytic, coordinates, evaluate, layer_problem

torch = pytest.importorskip("torch", reason="optional Torch runtime is not installed")
wp = importlib.import_module("warp")


@pytest.fixture(autouse=True)
def _native_context(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def _reverse(result, kind):
    """Return the mathematical seed, checking the real seed's storage afterwards."""
    if kind == "sum":
        observed = []
        result.register_hook(lambda seed: observed.append(seed.reshape(-1).stride()))
        result.sum().backward()
        # This is the production singleton reduction path that originally failed.
        assert observed == [(0,)]
        return np.ones(tuple(result.shape))
    if kind == "singleton-zero":
        backing = torch.tensor([7, 0.7, 9], dtype=result.dtype, device=result.device)
        seed = backing.as_strided(tuple(result.shape), (0, 0), storage_offset=1)
        assert seed.is_contiguous() and seed.reshape(-1).stride() == (0,)
    elif kind == "singleton-slice":
        backing = torch.tensor([[7, 0.7, 9, 11]], dtype=result.dtype, device=result.device)
        seed = backing[:, 1:2:2]
        assert seed.is_contiguous() and seed.reshape(-1).stride() == (2,)
    else:
        assert kind == "broadcast" and result.numel() > 1
        backing = torch.tensor(0.7, dtype=result.dtype, device=result.device)
        seed = backing.expand(tuple(result.shape))
        assert not seed.is_contiguous() and 0 in seed.stride()
    before = backing.clone()
    shape, strides, pointer, version = (
        tuple(seed.shape),
        seed.stride(),
        seed.data_ptr(),
        seed._version,
    )
    expected = seed.detach().cpu().double().numpy().copy()
    result.backward(seed)
    assert tuple(seed.shape) == shape and seed.stride() == strides
    assert seed.data_ptr() == pointer and seed._version == version
    assert torch.equal(backing, before)
    return expected


def _projection_problem(device, pixels, precision):
    grid = GridSpec((2, 2, 2), (0.8, 1.2, 0.9), (-0.5, -0.5, -0.5))
    geometry = DetectorGeometry(
        (-8.0, 0.15, 0.11),
        (8.0, -0.1, -0.05),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.13, 0.17),
        (1, pixels),
    )
    spec = ProjectionSpec(47, active_volume=True, precision=precision)
    volume = torch.linspace(0.1, 0.3, grid.voxels, dtype=torch.float32, device=device)
    volume = volume.reshape(grid.shape).requires_grad_()
    pose = torch.tensor(RigidTransform().packed(), dtype=torch.float64, device=device)
    # Independent host quadrature of each basis field supplies the whole matrix.
    matrix = np.asarray(
        [
            [
                integrate_sampled_field(
                    [float(index == voxel) for index in range(grid.voxels)],
                    grid,
                    geometry,
                    RigidTransform(),
                    0,
                    column,
                    midpoint_samples=spec.samples_per_ray,
                )
                for voxel in range(grid.voxels)
            ]
            for column in range(pixels)
        ]
    )
    return grid, geometry, spec, volume, pose, matrix


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["sum", "singleton-zero", "singleton-slice", "broadcast"])
def test_projection_seed_storage_matches_independent_matrix(kind, precision, warp_device):
    pixels = 3 if kind == "broadcast" else 1
    grid, geometry, spec, volume, pose, matrix = _projection_problem(warp_device, pixels, precision)
    workspace = prepare_projection(grid, geometry, spec, device=warp_device)
    result = project_optical_depth(volume, pose, workspace=workspace)
    expected = matrix @ volume.detach().cpu().numpy().reshape(-1)
    np.testing.assert_allclose(result.detach().cpu().numpy().reshape(-1), expected, rtol=3e-6)
    seed = _reverse(result, kind)
    np.testing.assert_allclose(
        volume.grad.cpu().numpy().reshape(-1), matrix.T @ seed.reshape(-1), rtol=3e-6, atol=2e-7
    )
    assert pose.grad is None


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["sum", "singleton-zero", "singleton-slice", "broadcast"])
def test_spectral_seed_storage_matches_independent_chain(kind, precision, warp_device):
    pixels = 3 if kind == "broadcast" else 1
    grid, geometry, spec, volume, pose, matrix = _projection_problem(warp_device, pixels, precision)
    fields = torch.stack((volume.detach(), 1.3 * volume.detach())).requires_grad_()
    coefficients = torch.tensor(
        [[0.2, 0.4, 0.6], [0.6, 0.5, 0.3]],
        dtype=torch.float32,
        device=warp_device,
        requires_grad=True,
    )
    weights = torch.tensor(
        [20, 30, 10], dtype=torch.float32, device=warp_device, requires_grad=True
    )
    response = torch.tensor(
        [0.8, 0.5, 0.9], dtype=torch.float32, device=warp_device, requires_grad=True
    )
    native_dtype = wp.float64 if precision == "float64" else wp.float32
    material = prepare_material_projection(
        grid,
        geometry,
        spec,
        materials=2,
        fields=wp.from_torch(fields.reshape(-1), requires_grad=False),
        out_paths=wp.empty(2 * pixels, dtype=native_dtype, device=warp_device),
        cotangent_paths=wp.empty(2 * pixels, dtype=native_dtype, device=warp_device),
        device=warp_device,
    )
    spectral = prepare_spectral(
        SpectralSpec(
            2,
            3,
            precision=precision,
            active_paths=True,
            active_coefficients=True,
            active_weights=True,
            active_response=True,
        ),
        max_pixels=pixels,
        device=warp_device,
        stream=material.context.stream,
    )
    result = project_spectral_signal(
        fields,
        pose,
        coefficients,
        weights,
        response,
        material_workspace=material,
        spectral_workspace=spectral,
    )
    paths = fields.detach().cpu().double().numpy().reshape(2, -1) @ matrix.T
    if precision == "float32":
        paths = paths.astype(np.float32).astype(np.float64)
    coeff = coefficients.detach().cpu().double().numpy()
    weight = weights.detach().cpu().double().numpy()[:, None]
    detector_response = response.detach().cpu().double().numpy()[:, None]
    attenuation = np.exp(-(coeff.T @ paths))
    bins = weight * detector_response * attenuation
    np.testing.assert_allclose(
        result.detach().cpu().numpy().reshape(-1), bins.sum(axis=0), rtol=3e-6
    )
    seed = _reverse(result, kind).reshape(-1)
    bin_seed = bins * seed
    gradients = (
        (-coeff @ bin_seed) @ matrix,
        -(paths @ bin_seed.T),
        (detector_response * attenuation * seed).sum(axis=1),
        (weight * attenuation * seed).sum(axis=1),
    )
    for value, target in zip((fields, coefficients, weights, response), gradients, strict=True):
        np.testing.assert_allclose(
            value.grad.cpu().numpy().reshape(target.shape), target, rtol=7e-6, atol=3e-7
        )
    assert pose.grad is None


@pytest.mark.parametrize("kind", ["sum", "singleton-zero", "singleton-slice", "broadcast"])
def test_expected_transport_seed_storage_matches_analytic_absorption(kind, warp_device):
    pixels = 2 if kind == "broadcast" else 1
    problem = layer_problem(torch, wp, warp_device, detector_columns=pixels)
    result = evaluate(problem).signal
    expected, _ = analytic(problem)
    np.testing.assert_allclose(
        result.detach().cpu().numpy(), [[0, expected]] if pixels == 2 else [[expected]], rtol=3e-13
    )
    seed = _reverse(result, kind)
    _, gradients = analytic(problem, float(seed[0, -1]))
    for value, target in zip(coordinates(problem), gradients, strict=True):
        np.testing.assert_allclose(value.grad.cpu().numpy(), target, rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("kind", ["sum", "singleton-zero", "singleton-slice", "broadcast"])
def test_transmission_seed_storage_matches_analytic_counts(kind, warp_device):
    pixels = 3 if kind == "broadcast" else 1
    depth = torch.full(
        (1, pixels), 0.3, dtype=torch.float32, device=warp_device, requires_grad=True
    )
    beam = torch.tensor([37], dtype=torch.float32, device=warp_device, requires_grad=True)
    workspace = prepare_transmission(
        TransmissionSpec(beam="device-scalar", active_beam=True),
        max_pixels=pixels,
        device=warp_device,
    )
    result = transmit(depth, beam, output="counts", workspace=workspace)
    attenuation = np.exp(-depth.detach().cpu().double().numpy())
    np.testing.assert_allclose(result.detach().cpu().numpy(), 37 * attenuation, rtol=3e-6)
    seed = _reverse(result, kind)
    np.testing.assert_allclose(depth.grad.cpu().numpy(), -37 * attenuation * seed, rtol=3e-6)
    np.testing.assert_allclose(beam.grad.cpu().numpy(), [(attenuation * seed).sum()], rtol=3e-6)


@pytest.mark.parametrize("kind", ["sum", "singleton-zero", "singleton-slice", "broadcast"])
def test_calibration_seed_storage_matches_analytic_shared_scale(kind, warp_device):
    pixels = 3 if kind == "broadcast" else 1
    mean = torch.full((1, pixels), 2, dtype=torch.float32, device=warp_device, requires_grad=True)
    gain = torch.tensor(1.7, dtype=torch.float32, device=warp_device, requires_grad=True)
    exposure = torch.tensor(0.4, dtype=torch.float32, device=warp_device)
    offset = torch.tensor(-0.2, dtype=torch.float32, device=warp_device, requires_grad=True)
    workspace = prepare_detector(max_pixels=pixels, device=warp_device)
    result = affine_calibrate(
        mean,
        gain,
        exposure,
        offset,
        spec=CalibrationSpec(output_unit="counts", active_gain=True, active_offset=True),
        workspace=workspace,
    )
    g, e, o = (float(value.detach().cpu().item()) for value in (gain, exposure, offset))
    np.testing.assert_allclose(result.detach().cpu().numpy(), 2 * g * e + o, rtol=3e-6)
    seed = _reverse(result, kind)
    np.testing.assert_allclose(mean.grad.cpu().numpy(), seed * g * e, rtol=3e-6)
    assert gain.grad.item() == pytest.approx(float((seed * 2 * e).sum()), rel=3e-6)
    assert offset.grad.item() == pytest.approx(float(seed.sum()), rel=3e-6)
    assert exposure.grad is None


@pytest.mark.parametrize("kind", ["sum", "singleton-zero", "singleton-slice", "broadcast"])
def test_blur_seed_storage_matches_independent_zero_extended_transpose(kind, warp_device):
    pixels = 3 if kind == "broadcast" else 1
    spec = BlurSpec(1, pixels, 1, 3, (0.1, 0.5, 0.2))
    matrix = blur_matrix(spec)
    source = torch.full((1, pixels), 2, dtype=torch.float32, device=warp_device, requires_grad=True)
    workspace = prepare_blur(spec, device=warp_device)
    result = blur(source, workspace=workspace)
    np.testing.assert_allclose(
        result.detach().cpu().numpy().reshape(-1), matrix @ np.full(pixels, 2)
    )
    seed = _reverse(result, kind)
    np.testing.assert_allclose(
        source.grad.cpu().numpy().reshape(-1), matrix.T @ seed.reshape(-1), rtol=3e-6, atol=1e-7
    )
