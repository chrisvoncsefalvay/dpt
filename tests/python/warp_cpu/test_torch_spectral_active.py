"""Independent spectral factor, field and rigid-pose Torch chain rules."""

import importlib
from contextlib import nullcontext
from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError, NumericalError
from dpt.geometry import RigidTransform
from dpt.geometry import compose_pose as host_compose
from dpt.torch import compose_pose
from helpers.torch_spectral_inputs import (
    evaluate,
    sampled_matrix,
    spectral_problem,
    spectral_reference,
    tensor_inputs,
)

torch = pytest.importorskip("torch", reason="optional Torch runtime is not installed")
wp = importlib.import_module("warp")


@pytest.fixture(autouse=True)
def _kernel_context(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("domain", ["linear", "log"])
@pytest.mark.parametrize("sharing", [(True, True), (True, False), (False, True), (False, False)])
def test_active_factors_and_fields_match_independent_matrix_chain(
    precision, domain, sharing, warp_device
):
    problem = spectral_problem(
        torch,
        wp,
        warp_device,
        precision=precision,
        domain=domain,
        shared_weights=sharing[0],
        shared_response=sharing[1],
    )
    seed = torch.linspace(-0.8, 1.1, 12, dtype=getattr(torch, precision), device=warp_device)
    seed = seed.reshape(2, 6)[:, ::2]
    saved_seed = seed.clone()
    result = evaluate(problem)
    expected, gradients = spectral_reference(problem, seed.detach().cpu().numpy())
    assert result.shape == (2, 3) and result.dtype == getattr(torch, precision)
    assert all(result.data_ptr() != value.data_ptr() for value in tensor_inputs(problem))
    np.testing.assert_allclose(result.detach().cpu().numpy(), expected, rtol=2e-6, atol=2e-7)
    result.backward(seed)
    for name, expected_gradient in gradients.items():
        value = getattr(problem, name)
        assert value.grad.dtype == value.dtype and value.grad.shape == value.shape
        np.testing.assert_allclose(
            value.grad.cpu().numpy(), expected_gradient, rtol=7e-6, atol=3e-7
        )
    assert problem.pose.grad is None and torch.equal(saved_seed, seed)


@pytest.mark.parametrize("name", ["fields", "coefficients", "weights", "response", "pose"])
def test_explicit_active_subset_produces_only_selected_gradient(name, warp_device):
    problem = spectral_problem(torch, wp, warp_device, active=(name,))
    result = evaluate(problem)
    result.sum().backward()
    for key in ("fields", "pose", "coefficients", "weights", "response"):
        value = getattr(problem, key)
        assert (value.grad is not None) == (key == name)
        if value.grad is not None:
            assert torch.isfinite(value.grad).all().item()
    if name in ("coefficients", "weights", "response"):
        assert not problem.spectral.spec.active_paths and problem.material.cotangent_paths is None
    if name == "pose":
        assert problem.pose.grad.shape == (12,)


@pytest.mark.parametrize("domain", ["linear", "log"])
def test_contiguous_flat_factor_and_field_views_preserve_input_gradient_shapes(domain, warp_device):
    problem = spectral_problem(
        torch, wp, warp_device, domain=domain, shared_weights=False, shared_response=False
    )
    expected, gradients = spectral_reference(problem, np.ones((2, 3)))
    for name in ("fields", "coefficients", "weights", "response"):
        value = getattr(problem, name)
        setattr(problem, name, value.detach().reshape(-1).requires_grad_())
    # Fields retain the declared material-major leading dimension.
    problem.fields = problem.fields.detach().reshape(2, -1).requires_grad_()
    result = evaluate(problem)
    np.testing.assert_allclose(result.detach().cpu().numpy(), expected, rtol=3e-12)
    result.sum().backward()
    for name, gradient in gradients.items():
        value = getattr(problem, name)
        assert value.grad.shape == value.shape
        np.testing.assert_allclose(
            value.grad.cpu().numpy().reshape(-1), gradient.reshape(-1), rtol=6e-6
        )


@pytest.mark.parametrize("name", ["fields", "pose", "coefficients", "weights", "response"])
def test_trainable_input_requires_its_own_active_flag(name, warp_device):
    problem = spectral_problem(torch, wp, warp_device, active=())
    getattr(problem, name).requires_grad_()
    with pytest.raises(ContractError, match=f"{name} is fixed"):
        evaluate(problem)


@pytest.mark.parametrize("domain", ["linear", "log"])
def test_factor_zero_bins_keep_nonzero_one_sided_derivatives(domain, warp_device):
    problem = spectral_problem(torch, wp, warp_device, domain=domain)
    with torch.no_grad():
        problem.weights[0] = 0
        problem.response[1] = 0
        problem.coefficients[0, 2] = 0
    seed = torch.ones((2, 3), dtype=torch.float64, device=warp_device)
    expected, gradients = spectral_reference(problem, np.ones((2, 3)))
    result = evaluate(problem)
    np.testing.assert_allclose(result.detach().cpu().numpy(), expected, rtol=3e-12, atol=3e-12)
    result.backward(seed)
    assert problem.weights.grad[0].item() > 0 and problem.response.grad[1].item() > 0
    assert problem.coefficients.grad[0, 2].item() < 0
    for name, gradient in gradients.items():
        np.testing.assert_allclose(getattr(problem, name).grad.cpu().numpy(), gradient, rtol=6e-6)


@pytest.mark.parametrize("domain", ["linear", "log"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_weighted_underflow_tails_match_decimal_slab_chain(domain, precision, warp_device):
    problem = spectral_problem(torch, wp, warp_device, precision=precision, domain=domain)
    with torch.no_grad():
        problem.fields.fill_(0.25)
        problem.coefficients.fill_(85 if precision == "float32" else 600)
        problem.weights.fill_(1e38)
        problem.response.fill_(1e38)
    projection = problem.material.projection
    matrix = sampled_matrix(projection.grid, projection.geometry, RigidTransform())
    paths = np.full((2, projection.grid.voxels), 0.25) @ matrix.T
    if precision == "float32":
        paths = paths.astype(np.float32).astype(np.float64)
    values = []
    grad_paths = np.zeros_like(paths)
    grad_coefficients = np.zeros((2, 3))
    grad_weights = np.zeros(3)
    grad_response = np.zeros(3)
    coefficient = Decimal.from_float(problem.coefficients[0, 0].item())
    population = Decimal.from_float(problem.weights[0].item())
    response = Decimal.from_float(problem.response[0].item())
    with localcontext() as decimal_context:
        decimal_context.prec = 100
        for pixel in range(projection.geometry.pixels):
            depth = coefficient * sum(Decimal.from_float(float(path)) for path in paths[:, pixel])
            attenuation = (-depth).exp()
            bin_value = population * response * attenuation
            mean = 3 * bin_value
            values.append(float(mean.ln() if domain == "log" else mean))
            scale = 1 / mean if domain == "log" else Decimal(1)
            for material in range(2):
                grad_paths[material, pixel] = float(-3 * coefficient * bin_value * scale)
                grad_coefficients[material, :] += float(
                    -Decimal.from_float(float(paths[material, pixel])) * bin_value * scale
                )
            grad_weights += float(response * attenuation * scale)
            grad_response += float(population * attenuation * scale)
    result = evaluate(problem)
    np.testing.assert_allclose(
        result.detach().cpu().numpy().reshape(-1),
        values,
        rtol=3e-6 if precision == "float32" else 2e-10,
    )
    result.sum().backward()
    # FP32 inputs have FP32 cotangents, including tails that round to zero.
    expected = {
        "fields": (grad_paths @ matrix).reshape(problem.fields.shape),
        "coefficients": grad_coefficients,
        "weights": grad_weights,
        "response": grad_response,
    }
    for name, gradient in expected.items():
        np.testing.assert_allclose(
            getattr(problem, name).grad.cpu().numpy(),
            gradient.astype(np.float32),
            rtol=2e-5,
            atol=np.nextafter(np.float32(0), np.float32(1)),
        )


@pytest.mark.parametrize("execution", ["serial", "fused"])
def test_packed_pose_chain_matches_independent_six_coordinate_finite_differences(
    execution, warp_device
):
    problem = spectral_problem(torch, wp, warp_device, active=("pose",), execution=execution)
    anchor = problem.pose.detach()
    coordinates = [0.03, -0.02, 0.01, 0.04, -0.03, 0.02]
    tangent = torch.tensor(coordinates, dtype=torch.float64, device=warp_device, requires_grad=True)
    pose = compose_pose(anchor, tangent)
    pose.retain_grad()
    result = evaluate(problem, pose=pose)
    seed = torch.tensor(
        [[0.3, -0.2, 0.6], [-0.1, 0.4, 0.7]], dtype=torch.float64, device=warp_device
    )
    result.backward(seed)
    expected = []
    for coordinate in range(6):
        plus, minus = coordinates.copy(), coordinates.copy()
        plus[coordinate] += 1e-5
        minus[coordinate] -= 1e-5
        plus_image, _ = spectral_reference(
            problem, seed.cpu().numpy(), pose=host_compose(RigidTransform(), tuple(plus))
        )
        minus_image, _ = spectral_reference(
            problem, seed.cpu().numpy(), pose=host_compose(RigidTransform(), tuple(minus))
        )
        expected.append(np.sum((plus_image - minus_image) * seed.cpu().numpy()) / 2e-5)
    assert pose.grad.shape == (12,) and torch.isfinite(pose.grad).all().item()
    np.testing.assert_allclose(tangent.grad.cpu().numpy(), expected, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("name", ["coefficients", "weights", "response"])
def test_factor_gradients_match_centered_represented_input_differences(name, warp_device):
    problem = spectral_problem(torch, wp, warp_device, active=(name,))
    result = evaluate(problem)
    result.sum().backward()
    value = getattr(problem, name)
    expected = value.grad.detach().clone()
    index = (0, 1) if name == "coefficients" else (1,)
    old = value[index].item()
    with torch.no_grad():
        value[index] = old + 1e-3
        high_input = value[index].item()
        high = evaluate(problem).sum().item()
        value[index] = old - 1e-3
        low_input = value[index].item()
        low = evaluate(problem).sum().item()
        value[index] = old
    observed = (high - low) / (high_input - low_input)
    assert expected[index].item() == pytest.approx(observed, rel=2e-5, abs=2e-6)


def test_module_parameters_retain_registration_and_current_optimizer_values(warp_device):
    from dpt.torch import SpectralProjection

    problem = spectral_problem(
        torch, wp, warp_device, active=("coefficients", "weights", "response")
    )
    parameters = {
        name: torch.nn.Parameter(getattr(problem, name))
        for name in ("coefficients", "weights", "response")
    }
    module = SpectralProjection(problem.material, problem.spectral, problem.pose, **parameters)
    assert dict(module.named_parameters()) == parameters
    optimizer = torch.optim.SGD(module.parameters(), lr=1e-5)
    batch = torch.stack([problem.fields, 0.7 * problem.fields])
    old = None
    for _ in range(3):
        optimizer.zero_grad()
        for name, value in parameters.items():
            setattr(problem, name, value)
        result = module(batch)
        if old is not None:
            assert not torch.equal(result, old)
        expected_gradient = {
            name: np.zeros(tuple(value.shape)) for name, value in parameters.items()
        }
        for index in range(2):
            expected, gradients = spectral_reference(
                problem, np.ones((2, 3)), fields=batch[index].detach().cpu().numpy()
            )
            np.testing.assert_allclose(result[index].detach().cpu().numpy(), expected, rtol=3e-12)
            for name in parameters:
                expected_gradient[name] += gradients[name]
        result.sum().backward()
        for name, parameter in parameters.items():
            assert parameter.grad is not None
            np.testing.assert_allclose(
                parameter.grad.cpu().numpy(), expected_gradient[name], rtol=6e-6
            )
        old = result.detach().clone()
        optimizer.step()


def test_module_registers_packed_pose_parameter_and_preserves_tensor_buffer_graph(warp_device):
    from dpt.torch import SpectralProjection

    problem = spectral_problem(torch, wp, warp_device, active=("pose", "weights"))
    pose = torch.nn.Parameter(problem.pose)
    base_weights = problem.weights
    weights = 1.1 * base_weights
    module = SpectralProjection(
        problem.material, problem.spectral, pose, problem.coefficients, weights, problem.response
    )
    assert dict(module.named_parameters()) == {"pose": pose}
    assert module.weights is weights
    module(problem.fields).sum().backward()
    assert pose.grad.shape == (12,) and base_weights.grad is not None


def test_outstanding_forwards_recompute_fields_and_keep_factor_outputs_owned(warp_device):
    problem = spectral_problem(torch, wp, warp_device)
    other_fields = (0.7 * problem.fields.detach()).requires_grad_()
    first = evaluate(problem)
    preserved = first.detach().clone()
    second = evaluate(problem, fields=other_fields)
    expected_first, first_gradients = spectral_reference(problem, np.ones((2, 3)))
    expected_second, second_gradients = spectral_reference(
        problem, np.full((2, 3), 0.4), fields=other_fields.detach().cpu().numpy()
    )
    (first.sum() + 0.4 * second.sum()).backward()
    assert torch.equal(first.detach(), preserved) and first.data_ptr() != second.data_ptr()
    np.testing.assert_allclose(first.detach().cpu().numpy(), expected_first, rtol=3e-12)
    np.testing.assert_allclose(second.detach().cpu().numpy(), expected_second, rtol=3e-12)
    np.testing.assert_allclose(
        problem.fields.grad.cpu().numpy(), first_gradients["fields"], rtol=6e-6
    )
    np.testing.assert_allclose(
        other_fields.grad.cpu().numpy(), second_gradients["fields"], rtol=6e-6
    )
    for name in ("coefficients", "weights", "response"):
        np.testing.assert_allclose(
            getattr(problem, name).grad.cpu().numpy(),
            first_gradients[name] + second_gradients[name],
            rtol=6e-6,
        )


@pytest.mark.parametrize("name", ["fields", "pose", "coefficients", "weights", "response"])
def test_saved_fixed_and_active_inputs_detect_mutation_before_reverse(name, warp_device):
    problem = spectral_problem(torch, wp, warp_device)
    result = evaluate(problem)
    with torch.no_grad():
        getattr(problem, name).add_(0.01)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.sum().backward()
    assert all(value.grad is None for value in tensor_inputs(problem))


def test_factor_only_model_rejects_higher_derivatives(warp_device):
    problem = spectral_problem(torch, wp, warp_device, active=("weights",))
    result = evaluate(problem)
    with pytest.raises(ContractError, match="first-order"):
        torch.autograd.grad(result.sum(), problem.weights, create_graph=True)


def test_log_zero_mean_rejects_undefined_reverse(warp_device):
    problem = spectral_problem(torch, wp, warp_device, active=("weights",), domain="log")
    with torch.no_grad():
        problem.weights.zero_()
    result = evaluate(problem)
    assert torch.isneginf(result).all().item()
    with pytest.raises(NumericalError):
        result.sum().backward()
    assert problem.weights.grad is None


def test_field_or_pose_derivative_requires_prepared_path_cotangents(warp_device):
    problem = spectral_problem(torch, wp, warp_device, active=("fields",))
    problem.spectral.spec = replace(problem.spectral.spec, active_paths=False)
    with pytest.raises(ContractError, match="active spectral paths"):
        evaluate(problem)


def test_zero_reverse_seed_delivers_pending_status_before_native_recomputation(
    warp_device, monkeypatch
):
    from dpt.interop import torch as adapter

    problem = spectral_problem(torch, wp, warp_device, active=("weights",))
    result = evaluate(problem)
    with problem.material.context.scope():
        # Bit 2 is an unrepresentable output/gradient; bit 1 is an input contract error.
        wp.copy(problem.spectral._status, wp.array([2], dtype=wp.int32, device=warp_device))
    launches = []

    def unexpected_project(*args, **kwargs):
        launches.append(True)
        raise AssertionError("pending status must be delivered before recomputing material paths")

    monkeypatch.setattr(adapter, "project_material_paths", unexpected_project)
    with pytest.raises(NumericalError):
        result.backward(torch.zeros_like(result))
    assert launches == [] and problem.weights.grad is None
    problem.material.check_status()
    problem.spectral.check_status()


def test_zero_reverse_seed_rejects_ambient_warp_tape(warp_device):
    problem = spectral_problem(torch, wp, warp_device, active=("weights",))
    result = evaluate(problem)
    with wp.Tape(), pytest.raises(ContractError, match=r"tape|Tape"):
        result.backward(torch.zeros_like(result))
    assert problem.weights.grad is None
