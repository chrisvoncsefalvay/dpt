"""Actual Torch/CUDA storage, stream and independent projector-adjoint checks."""

import importlib
from typing import Any, Literal

import pytest

from dpt._runtime import HostReadback, prepare_context
from dpt.contracts import ContractError, NumericalError
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.interop import as_warp_array
from dpt.interop import torch as torch_bridge
from dpt.interop.torch import project_optical_depth
from dpt.material_projection import prepare_material_projection, project_material_paths
from dpt.projection import ProjectionSpec, prepare_projection
from dpt.spectral import SpectralSpec, prepare_spectral, spectral_signal
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

torch: Any = importlib.import_module("torch")
wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


def _problem(
    precision: Literal["float32", "float64"] = "float64", stream: Any = None
) -> tuple[Any, Any, Any]:
    grid = GridSpec((2, 2, 2), (0.8, 1.2, 0.9), (-0.5, -0.5, -0.5))
    geometry = DetectorGeometry(
        (-12.0, 0.15, 0.11),
        (12.0, -0.3, -0.3),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.13, 0.17),
        (3, 4),
    )
    if stream is None:
        stream = torch.cuda.current_stream("cuda:0")
    workspace = prepare_projection(
        grid,
        geometry,
        ProjectionSpec(47, active_volume=True, precision=precision),
        device="cuda:0",
        stream=wp.stream_from_torch(stream),
    )
    volume = torch.linspace(0.1, 0.3, 8, dtype=torch.float32, device="cuda:0").reshape(grid.shape)
    volume.requires_grad_()
    pose = torch.tensor(RigidTransform().packed(), dtype=torch.float64, device="cuda:0")
    return workspace, volume, pose


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_torch_projection_and_volume_gradient_match_independent_explicit_matrix(
    precision: Literal["float32", "float64"],
) -> None:
    workspace, volume, pose = _problem(precision)
    matrix = []
    for voxel in range(workspace.grid.voxels):
        basis = [float(index == voxel) for index in range(workspace.grid.voxels)]
        matrix.append(
            [
                integrate_sampled_field(
                    basis,
                    workspace.grid,
                    workspace.geometry,
                    RigidTransform(),
                    row,
                    column,
                    midpoint_samples=47,
                )
                for row in range(3)
                for column in range(4)
            ]
        )
    operator = torch.tensor(matrix, dtype=torch.float64, device="cuda:0").T
    output = project_optical_depth(volume, pose, workspace=workspace)
    assert tuple(output.shape) == (3, 4)
    assert output.dtype == (torch.float64 if precision == "float64" else torch.float32)
    expected = operator @ volume.detach().reshape(-1).to(torch.float64)
    torch.testing.assert_close(output.reshape(-1).to(torch.float64), expected, rtol=2e-6, atol=1e-7)
    storage = torch.linspace(-1.0, 1.0, 24, dtype=output.dtype, device="cuda:0").reshape(3, 8)
    seed = storage[:, ::2]
    saved_seed = seed.clone()
    output.backward(seed)
    expected_gradient = operator.T @ seed.reshape(-1).to(torch.float64)
    torch.testing.assert_close(
        volume.grad.reshape(-1).to(torch.float64), expected_gradient, rtol=3e-6, atol=2e-7
    )
    assert torch.equal(seed, saved_seed)
    assert pose.grad is None


def test_torch_tensor_mutation_is_detected_before_reverse_launches() -> None:
    workspace, volume, pose = _problem()
    output = project_optical_depth(volume, pose, workspace=workspace)
    with torch.no_grad():
        volume.add_(0.01)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()
    assert volume.grad is None


def test_torch_adapter_rejects_higher_derivatives_explicitly() -> None:
    workspace, volume, pose = _problem()
    output = project_optical_depth(volume, pose, workspace=workspace)
    with pytest.raises(ContractError, match="first-order"):
        torch.autograd.grad(output.sum(), volume, create_graph=True)


def test_torch_adapter_uses_a_prepared_nondefault_stream() -> None:
    stream = torch.cuda.Stream(device="cuda:0")
    with torch.cuda.stream(stream):
        workspace, volume, pose = _problem(stream=stream)
        output = project_optical_depth(volume, pose, workspace=workspace)
        output.sum().backward()
    torch.cuda.current_stream("cuda:0").wait_stream(stream)
    assert torch.isfinite(volume.grad).all().item()
    assert volume.grad.abs().sum().item() > 0.0


def test_torch_adapter_orders_foreign_stream_and_rejects_inactive_pose_or_dtype() -> None:
    workspace, volume, pose = _problem()
    foreign = torch.cuda.Stream(device="cuda:0")
    foreign.wait_stream(torch.cuda.current_stream("cuda:0"))
    expected = project_optical_depth(volume, pose, workspace=workspace)
    with torch.cuda.stream(foreign):
        actual = project_optical_depth(volume, pose, workspace=workspace)
        torch.testing.assert_close(actual, expected)
    with pytest.raises(ContractError, match="float32"):
        project_optical_depth(volume.to(torch.float64), pose, workspace=workspace)
    pose.requires_grad_()
    fixed = prepare_projection(
        workspace.grid,
        workspace.geometry,
        ProjectionSpec(47, active_volume=True, active_pose=False, precision="float64"),
        device="cuda:0",
        stream=workspace.context.stream,
    )
    with pytest.raises(ContractError, match="pose is fixed"):
        project_optical_depth(volume, pose, workspace=fixed)


def test_legacy_cuda_array_view_waits_for_advertised_torch_producer_stream() -> None:
    producer = torch.cuda.Stream(device="cuda:0")
    with torch.cuda.stream(producer):
        values = torch.empty(4, dtype=torch.float32, device="cuda:0")
        values.fill_(17.0)
    consumer = wp.Stream("cuda:0")
    context = prepare_context(device="cuda:0", stream=consumer)

    class LegacyCudaArray:
        def __len__(self) -> int:
            return 4

        @property
        def __cuda_array_interface__(self) -> dict[str, Any]:
            return {
                "version": 3,
                "shape": (4,),
                "typestr": "<f4",
                "data": (values.data_ptr(), False),
                "stream": producer.cuda_stream,
            }

    view = as_warp_array(
        LegacyCudaArray(), context=context, dtype=wp.float32, source_device="cuda:0"
    )
    assert view.ptr == values.data_ptr()
    assert HostReadback(context, view).read().tolist() == [17.0] * 4


def _spectral_problem() -> tuple[Any, ...]:
    projection, volume, pose = _problem()
    fields = torch.stack([0.4 * volume.detach(), 0.6 * volume.detach()]).requires_grad_()
    pixels = projection.geometry.pixels
    material = prepare_material_projection(
        projection.grid,
        projection.geometry,
        projection.spec,
        materials=2,
        fields=wp.from_torch(fields.reshape(-1), requires_grad=False),
        out_paths=wp.empty(2 * pixels, dtype=wp.float64, device="cuda:0"),
        cotangent_paths=wp.empty(2 * pixels, dtype=wp.float64, device="cuda:0"),
        device="cuda:0",
        stream=projection.context.stream,
    )
    spectral = prepare_spectral(
        SpectralSpec(2, 3, precision="float64"),
        max_pixels=pixels,
        device="cuda:0",
        stream=projection.context.stream,
    )
    coefficients = torch.tensor(
        [[0.2, 0.4, 0.6], [0.6, 0.5, 0.3]], dtype=torch.float32, device="cuda:0"
    )
    weights = torch.tensor([20.0, 30.0, 10.0], dtype=torch.float32, device="cuda:0")
    response = torch.tensor([0.8, 0.5, 0.9], dtype=torch.float32, device="cuda:0")
    return material, spectral, fields, pose, coefficients, weights, response


def _spectral_reference(
    material: Any, fields: Any, coefficients: Any, weights: Any, response: Any, seed: Any
) -> tuple[Any, Any]:
    projection = material.projection
    basis_columns = []
    for voxel in range(projection.grid.voxels):
        basis = [float(index == voxel) for index in range(projection.grid.voxels)]
        basis_columns.append(
            [
                integrate_sampled_field(
                    basis,
                    projection.grid,
                    projection.geometry,
                    RigidTransform(),
                    row,
                    column,
                    midpoint_samples=47,
                )
                for row in range(3)
                for column in range(4)
            ]
        )
    operator = torch.tensor(basis_columns, dtype=torch.float64, device="cuda:0").T
    paths = fields.detach().reshape(2, -1).to(torch.float64) @ operator.T
    attenuation = coefficients.to(torch.float64).T @ paths
    bins = (
        weights.to(torch.float64)[:, None]
        * response.to(torch.float64)[:, None]
        * torch.exp(-attenuation)
    )
    gradient_paths = -coefficients.to(torch.float64) @ (bins * seed.reshape(1, -1))
    return bins.sum(dim=0).reshape(3, 4), (gradient_paths @ operator).reshape(fields.shape)


def test_torch_spectral_projection_matches_independent_matrix_and_analytic_chain() -> None:
    material, spectral, fields, pose, coefficients, weights, response = _spectral_problem()
    original_fields_pointer = material.fields.ptr
    output = torch_bridge.project_spectral_signal(
        fields,
        pose,
        coefficients,
        weights,
        response,
        material_workspace=material,
        spectral_workspace=spectral,
    )
    seed = torch.linspace(-1.0, 1.0, 12, dtype=torch.float64, device="cuda:0").reshape(3, 4)
    expected, expected_gradient = _spectral_reference(
        material, fields, coefficients, weights, response, seed
    )
    torch.testing.assert_close(output, expected, rtol=3e-7, atol=3e-7)
    output.backward(seed)
    torch.testing.assert_close(
        fields.grad.to(torch.float64), expected_gradient, rtol=5e-6, atol=5e-7
    )
    assert material.fields.ptr == original_fields_pointer


def test_torch_spectral_module_batch_recomputes_each_saved_field_for_backward() -> None:
    material, spectral, fields, pose, coefficients, weights, response = _spectral_problem()
    module = torch_bridge.SpectralProjection(
        material, spectral, pose, coefficients, weights, response
    )
    batch = torch.stack([fields.detach(), 0.7 * fields.detach()]).requires_grad_()
    output = module(batch)
    assert tuple(output.shape) == (2, 3, 4)
    expected_gradients = []
    for index in range(2):
        expected, gradient = _spectral_reference(
            material,
            batch[index],
            coefficients,
            weights,
            response,
            torch.ones((3, 4), dtype=torch.float64, device="cuda:0"),
        )
        torch.testing.assert_close(output[index], expected, rtol=3e-7, atol=3e-7)
        expected_gradients.append(gradient)
    output.sum().backward()
    torch.testing.assert_close(
        batch.grad.to(torch.float64), torch.stack(expected_gradients), rtol=5e-6, atol=5e-7
    )


def test_torch_spectral_fixed_inputs_and_higher_derivatives_are_explicit() -> None:
    material, spectral, fields, pose, coefficients, weights, response = _spectral_problem()
    weights.requires_grad_()
    with pytest.raises(ContractError, match="weights is fixed"):
        torch_bridge.project_spectral_signal(
            fields,
            pose,
            coefficients,
            weights,
            response,
            material_workspace=material,
            spectral_workspace=spectral,
        )
    weights.requires_grad_(False)
    output = torch_bridge.project_spectral_signal(
        fields,
        pose,
        coefficients,
        weights,
        response,
        material_workspace=material,
        spectral_workspace=spectral,
    )
    with pytest.raises(ContractError, match="first-order"):
        torch.autograd.grad(output.sum(), fields, create_graph=True)


def test_torch_spectral_delivers_joint_arithmetic_failures_then_accepts_repaired_retry() -> None:
    material, spectral, fields, pose, coefficients, weights, response = _spectral_problem()
    clean_fields = fields.detach().clone()
    with torch.no_grad():
        fields[0, 0, 0, 0] = float("inf")
    with pytest.raises(NumericalError) as failure:
        torch_bridge.project_spectral_signal(
            fields,
            pose,
            coefficients,
            weights,
            response,
            material_workspace=material,
            spectral_workspace=spectral,
            validate=False,
        )
    diagnostics = failure.value.__dict__["status_diagnostics"]
    assert {owner for owner, code, _ in diagnostics if code} >= {
        "ProjectionWorkspace",
        "SpectralWorkspace",
    }
    material.check_status()
    spectral.check_status()
    with torch.no_grad():
        fields.copy_(clean_fields)
    output = torch_bridge.project_spectral_signal(
        fields,
        pose,
        coefficients,
        weights,
        response,
        material_workspace=material,
        spectral_workspace=spectral,
        validate=False,
    )
    expected, _ = _spectral_reference(
        material,
        fields,
        coefficients,
        weights,
        response,
        torch.ones_like(output),
    )
    torch.testing.assert_close(output, expected, rtol=3e-7, atol=3e-7)
    output.sum().backward()
    assert torch.isfinite(fields.grad).all().item()


def test_torch_spectral_reports_pending_unchecked_failures_before_new_launches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    material, spectral, fields, pose, coefficients, weights, response = _spectral_problem()
    clean_fields = fields.detach().clone()
    with torch.no_grad(), material.context.scope():
        fields[0, 0, 0, 0] = float("inf")
        project_material_paths(
            wp.from_torch(pose, requires_grad=False), workspace=material, validate=False
        )
        output = wp.empty(material.projection.geometry.pixels, dtype=wp.float64, device="cuda:0")
        spectral_signal(
            material.paths,
            wp.from_torch(coefficients, requires_grad=False),
            wp.from_torch(weights, requires_grad=False),
            wp.from_torch(response, requires_grad=False),
            out_mean=output,
            workspace=spectral,
            validate=False,
        )
        fields.copy_(clean_fields)
    with pytest.raises(NumericalError):
        material.check_status()
    with pytest.raises(NumericalError):
        spectral.check_status()
    launches: list[bool] = []
    native_project = torch_bridge.project_material_paths

    def counted_project(*args: Any, **kwargs: Any) -> Any:
        launches.append(True)
        return native_project(*args, **kwargs)

    monkeypatch.setattr(torch_bridge, "project_material_paths", counted_project)
    with pytest.raises(NumericalError) as failure:
        torch_bridge.project_spectral_signal(
            fields,
            pose,
            coefficients,
            weights,
            response,
            material_workspace=material,
            spectral_workspace=spectral,
            validate=False,
        )
    assert len(failure.value.__dict__["status_diagnostics"]) >= 2
    assert launches == []
    material.check_status()
    spectral.check_status()
    repaired = torch_bridge.project_spectral_signal(
        fields,
        pose,
        coefficients,
        weights,
        response,
        material_workspace=material,
        spectral_workspace=spectral,
        validate=False,
    )
    assert len(launches) == 1
    assert torch.isfinite(repaired).all().item()


@pytest.mark.parametrize("binding", ["fields", "coefficients", "path_cotangent", "partials"])
def test_torch_spectral_rechecks_tensor_aliases_before_overwriting_prepared_buffers(
    binding: str,
) -> None:
    material, spectral, fields, pose, coefficients, weights, response = _spectral_problem()
    if binding == "partials":
        spectral = prepare_spectral(
            SpectralSpec(2, 3, precision="float64", active_coefficients=True, active_weights=True),
            max_pixels=material.projection.geometry.pixels,
            device="cuda:0",
            stream=material.context.stream,
        )
    destination = (
        material.cotangent_paths
        if binding == "path_cotangent"
        else spectral._partials
        if binding == "partials"
        else material.paths
    )
    storage = wp.to_torch(destination).view(torch.float32)
    with torch.no_grad():
        storage.fill_(0.1)
    saved = storage.clone()
    if binding == "coefficients":
        coefficients = storage[: coefficients.numel()].reshape(coefficients.shape)
    else:
        fields = storage[: fields.numel()].reshape(fields.shape).requires_grad_()
    with pytest.raises(ContractError, match="overlaps"):
        torch_bridge.project_spectral_signal(
            fields,
            pose,
            coefficients,
            weights,
            response,
            material_workspace=material,
            spectral_workspace=spectral,
        )
    assert torch.equal(storage, saved)


def test_torch_projection_batches_use_declared_bundle_geometry() -> None:
    from dpt.geometry import RayBundleGeometry

    original, volume, pose = _problem()
    bundle = RayBundleGeometry.from_views((original.geometry, original.geometry))
    workspace = prepare_projection(
        original.grid, bundle, original.spec, device="cuda:0", stream=original.context.stream
    )
    batch = torch.stack([volume.detach(), 0.5 * volume.detach()]).requires_grad_()
    output = torch.stack([project_optical_depth(item, pose, workspace=workspace) for item in batch])
    assert tuple(output.shape) == (2, 24)
    torch.testing.assert_close(output[0, :12], output[0, 12:], rtol=0.0, atol=0.0)
    torch.testing.assert_close(output[1], 0.5 * output[0], rtol=0.0, atol=0.0)
    output.sum().backward()
    torch.testing.assert_close(batch.grad[0], batch.grad[1], rtol=0.0, atol=0.0)
