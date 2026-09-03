"""Numerical and autograd checks for the ordinary Projector and functional API."""

import math
from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest
import torch

from dpt._runtime import cpu_test_context
from dpt.api import project
from dpt.contracts import ContractError
from dpt.geometry import BatchedGeometry, DetectorGeometry, RigidTransform
from dpt.imaging_io import ImagingMetadata, ImagingVolume
from dpt.torch import Projector
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec


@pytest.fixture(autouse=True)
def _native_context(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def _problem(device):
    grid = GridSpec((3, 3, 4), (0.7, 0.8, 0.9), (-1.0, -1.0, -1.0))
    panel = DetectorGeometry(
        (-8.0, 0.1, -0.2),
        (8.0, -0.3, -0.4),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.2, 0.2),
        (2, 3),
    )
    geometry = BatchedGeometry((panel, replace(panel, origin_mm=(8.0, -0.24, -0.31))))
    field = torch.linspace(0.1, 0.3, grid.voxels, dtype=torch.float32, device=device).reshape(
        grid.shape
    )
    return grid, geometry, field


def _matrix(grid, geometry, device):
    columns = []
    for voxel in range(grid.voxels):
        basis = [float(index == voxel) for index in range(grid.voxels)]
        columns.append(
            [
                integrate_sampled_field(
                    basis, grid, panel, RigidTransform(), row, column, midpoint_samples=47
                )
                for panel in geometry.views
                for row in range(panel.shape[0])
                for column in range(panel.shape[1])
            ]
        )
    return torch.tensor(columns, dtype=torch.float64, device=device).T


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_projector_images_and_volume_gradient_match_independent_matrix(warp_device, precision):
    grid, geometry, field = _problem(warp_device)
    module = Projector(grid=grid, geometry=geometry, precision=precision, samples_per_ray=47)
    assert isinstance(module, torch.nn.Module)
    field.requires_grad_()
    image = module(field)
    operator = _matrix(grid, geometry, warp_device)
    expected = operator @ field.detach().reshape(-1).double()
    assert tuple(image.shape) == geometry.shape
    assert image.dtype == (torch.float32 if precision == "fp32" else torch.float64)
    # FP32 rounding is checked against an independent FP64 discretisation.
    tolerance = 5e-6 if precision == "fp32" else 2e-11
    torch.testing.assert_close(image.reshape(-1).double(), expected, rtol=tolerance, atol=tolerance)
    seed = torch.linspace(-0.5, 0.8, image.numel(), dtype=image.dtype, device=warp_device).reshape(
        image.shape
    )
    image.backward(seed)
    torch.testing.assert_close(
        field.grad.reshape(-1).double(),
        operator.T @ seed.reshape(-1).double(),
        rtol=5e-6,
        atol=5e-7,
    )


def test_projector_pose_tangent_matches_finite_differences_and_explicit_packed_pose(warp_device):
    grid, geometry, field = _problem(warp_device)
    module = Projector(grid=grid, geometry=geometry, precision="fp64", samples_per_ray=47)
    tangent = torch.tensor(
        [0.03, -0.02, 0.01, 0.04, -0.03, 0.02],
        dtype=torch.float64,
        device=warp_device,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        lambda value: module(field, value), (tangent,), eps=2e-5, atol=2e-6, rtol=2e-5
    )
    identity = torch.tensor(RigidTransform().packed(), dtype=torch.float64, device=warp_device)
    from dpt.torch import compose_pose

    expected = module(field, packed_pose=compose_pose(identity, tangent))
    torch.testing.assert_close(module(field, tangent), expected)
    with pytest.raises(ContractError, match="choose pose or packed_pose"):
        module(field, tangent, packed_pose=identity)


def test_fp32_pose_gradient_agrees_with_fp64_at_the_same_sampling_rule(warp_device):
    grid, geometry, field = _problem(warp_device)
    tangent = torch.tensor(
        [0.03, -0.02, 0.01, 0.04, -0.03, 0.02],
        dtype=torch.float64,
        device=warp_device,
        requires_grad=True,
    )
    gradients = []
    for precision in ("fp32", "fp64"):
        module = Projector(grid=grid, geometry=geometry, precision=precision, samples_per_ray=47)
        image = module(field, tangent)
        seed = torch.linspace(-0.5, 0.8, image.numel(), device=warp_device).reshape(image.shape)
        gradients.append(torch.autograd.grad((image * seed).sum(), tangent)[0])
    torch.testing.assert_close(gradients[0], gradients[1], rtol=2e-4, atol=5e-6)


@pytest.mark.parametrize("mutation", ["field", "packed_pose"])
def test_projector_detects_input_mutation_before_backward(warp_device, mutation):
    grid, geometry, field = _problem(warp_device)
    module = Projector(grid=grid, geometry=geometry, precision="fp64", samples_per_ray=47)
    field.requires_grad_()
    pose = torch.tensor(RigidTransform().packed(), dtype=torch.float64, device=warp_device)
    image = module(field, packed_pose=pose)
    with torch.no_grad():
        if mutation == "field":
            field.add_(0.01)
        else:
            pose[9].add_(0.01)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        image.sum().backward()


def test_repeated_projector_calls_keep_outputs_and_gradients_separate(warp_device):
    grid, geometry, field = _problem(warp_device)
    module = Projector(grid=grid, geometry=geometry, precision="fp64", samples_per_ray=47)
    field.requires_grad_()
    first = module(field)
    saved = first.detach().clone()
    second = module(0.5 * field)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, saved)
    torch.testing.assert_close(second, 0.5 * first)
    (first.sum() + second.sum()).backward()
    operator = _matrix(grid, geometry, warp_device)
    torch.testing.assert_close(
        field.grad.reshape(-1).double(), 1.5 * operator.sum(0), rtol=4e-6, atol=1e-7
    )


def test_inference_out_writes_requested_storage_and_rejects_active_autograd(warp_device):
    grid, geometry, field = _problem(warp_device)
    module = Projector(grid=grid, geometry=geometry, precision="fp64", samples_per_ray=47)
    expected = module(field)
    out = torch.full_like(expected, -123.0)
    assert module(field, out=out) is out
    torch.testing.assert_close(out, expected)
    field.requires_grad_()
    out.fill_(-123.0)
    with pytest.raises(ContractError, match="inference buffer"):
        module(field, out=out)
    torch.testing.assert_close(out, torch.full_like(out, -123.0))
    with torch.no_grad():
        assert module(field, out=out) is out
    torch.testing.assert_close(out, expected)


def test_functional_volume_and_prepared_tensor_use_same_projection(warp_device):
    grid, geometry, field = _problem(warp_device)
    volume = ImagingVolume(
        np.linspace(0.1, 0.3, grid.voxels, dtype=np.float32).reshape(grid.shape),
        grid,
        ImagingMetadata("synthetic", "RAS", "mm^-1"),
    )
    module = Projector(grid=grid, geometry=geometry, precision="fp64", samples_per_ray=47)
    field = torch.tensor(volume.values, dtype=torch.float32, device=warp_device, requires_grad=True)
    expected = module(field)
    actual = project(volume, geometry, device=warp_device, precision="fp64", samples_per_ray=47)
    torch.testing.assert_close(actual, expected)
    resident = project(field, prepared=module)
    torch.testing.assert_close(resident, expected)
    resident.sum().backward()
    assert field.grad is not None
    with pytest.raises(ContractError, match="precision"):
        project(field, prepared=module, precision="fp32")
    with pytest.raises(ContractError, match="convert HU"):
        project(replace(volume, metadata=replace(volume.metadata, value_units="HU")), geometry)


def test_step_resolves_one_sample_count_and_anchor_parameters_are_registered(warp_device):
    grid, geometry, field = _problem(warp_device)
    anchor = torch.nn.Parameter(
        torch.tensor(RigidTransform().packed(), dtype=torch.float64, device=warp_device)
    )
    module = Projector(grid=grid, geometry=geometry, precision="fp64", step_mm=0.2, anchor=anchor)
    assert dict(module.named_parameters())["anchor"] is anchor
    samples = math.ceil(math.hypot(4 * 0.7, 3 * 0.8, 3 * 0.9) / 0.2)
    explicit = Projector(grid=grid, geometry=geometry, precision="fp64", samples_per_ray=samples)
    assert module.samples_per_ray == samples
    torch.testing.assert_close(module(field), explicit(field))
    module(field).sum().backward()
    assert anchor.grad is not None
    with pytest.raises(ContractError, match="choose samples_per_ray or step_mm"):
        Projector(grid=grid, geometry=geometry, precision="fp64", samples_per_ray=47, step_mm=0.2)
