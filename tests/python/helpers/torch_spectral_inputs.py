"""Small spectral acquisitions and independent sampled-matrix chain rules."""

from functools import lru_cache
from types import SimpleNamespace
from typing import Literal

import numpy as np

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_projection import prepare_material_projection
from dpt.projection import ProjectionSpec
from dpt.spectral import SpectralSpec, prepare_spectral
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec


def spectral_problem(
    torch,
    wp,
    device,
    *,
    precision: Literal["float32", "float64"] = "float64",
    domain: Literal["linear", "log"] = "linear",
    shared_weights=True,
    shared_response=True,
    active=("fields", "coefficients", "weights", "response"),
    stream=None,
    execution: Literal["serial", "fused"] = "serial",
):
    grid = GridSpec((3, 3, 4), (0.7, 0.8, 0.9), (-1.0, -1.0, -1.0))
    geometry = DetectorGeometry(
        (-8.0, 0.1, -0.2),
        (8.0, -0.3, -0.4),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.2, 0.2),
        (2, 3),
    )
    values = torch.linspace(0.06, 0.2, 2 * grid.voxels, dtype=torch.float32, device=device)
    fields = values.reshape(2, *grid.shape).requires_grad_("fields" in active)
    pose = torch.tensor(RigidTransform().packed(), dtype=torch.float64, device=device)
    pose.requires_grad_("pose" in active)
    coefficients = torch.tensor(
        [[0.2, 0.4, 0.6], [0.6, 0.5, 0.3]], dtype=torch.float32, device=device
    ).requires_grad_("coefficients" in active)
    weights = torch.tensor([20.0, 30.0, 10.0], dtype=torch.float32, device=device)
    response = torch.tensor([0.8, 0.5, 0.9], dtype=torch.float32, device=device)
    pixels = geometry.pixels
    pattern = torch.linspace(0.7, 1.2, pixels, dtype=torch.float32, device=device)
    if not shared_weights:
        weights = (weights[:, None] * pattern).contiguous()
    if not shared_response:
        response = (response[:, None] * pattern.flip(0)).contiguous()
    weights.requires_grad_("weights" in active)
    response.requires_grad_("response" in active)
    path_dtype = wp.float64 if precision == "float64" else wp.float32
    paths_active = "fields" in active or "pose" in active
    material = prepare_material_projection(
        grid,
        geometry,
        ProjectionSpec(
            47,
            active_volume="fields" in active,
            active_pose="pose" in active,
            precision=precision,
            integration="cell_gauss",
        ),
        materials=2,
        fields=wp.from_torch(fields.reshape(-1), requires_grad=False),
        out_paths=wp.empty(2 * pixels, dtype=path_dtype, device=device),
        cotangent_paths=wp.empty(2 * pixels, dtype=path_dtype, device=device)
        if paths_active
        else None,
        device=device,
        stream=stream,
        channel_execution=execution,
    )
    spectral = prepare_spectral(
        SpectralSpec(
            2,
            3,
            precision=precision,
            output_domain=domain,
            shared_weights=shared_weights,
            shared_response=shared_response,
            active_paths=paths_active,
            active_coefficients="coefficients" in active,
            active_weights="weights" in active,
            active_response="response" in active,
        ),
        max_pixels=pixels,
        device=device,
        stream=material.context.stream,
    )
    return SimpleNamespace(
        material=material,
        spectral=spectral,
        fields=fields,
        pose=pose,
        coefficients=coefficients,
        weights=weights,
        response=response,
    )


def tensor_inputs(problem):
    return (problem.fields, problem.pose, problem.coefficients, problem.weights, problem.response)


def evaluate(problem, *, fields=None, pose=None):
    from dpt.interop.torch import project_spectral_signal

    return project_spectral_signal(
        problem.fields if fields is None else fields,
        problem.pose if pose is None else pose,
        problem.coefficients,
        problem.weights,
        problem.response,
        material_workspace=problem.material,
        spectral_workspace=problem.spectral,
    )


@lru_cache(maxsize=32)
def sampled_matrix(grid, geometry, pose):
    columns = []
    for voxel in range(grid.voxels):
        basis = [float(index == voxel) for index in range(grid.voxels)]
        columns.append(
            [
                integrate_sampled_field(basis, grid, geometry, pose, row, column)
                for row in range(geometry.shape[0])
                for column in range(geometry.shape[1])
            ]
        )
    return np.asarray(columns, dtype=np.float64).T


def spectral_reference(problem, seed, *, fields=None, pose=None):
    projection = problem.material.projection
    pose = RigidTransform() if pose is None else pose
    matrix = sampled_matrix(projection.grid, projection.geometry, pose)
    fields = problem.fields.detach().cpu().numpy() if fields is None else np.asarray(fields)
    coefficients = problem.coefficients.detach().cpu().numpy().astype(np.float64)
    weights = problem.weights.detach().cpu().numpy().astype(np.float64)
    response = problem.response.detach().cpu().numpy().astype(np.float64)
    paths = fields.reshape(2, -1).astype(np.float64) @ matrix.T
    if problem.spectral.spec.precision == "float32":
        paths = paths.astype(np.float32).astype(np.float64)
    if weights.ndim == 1:
        weights = weights[:, None]
    if response.ndim == 1:
        response = response[:, None]
    attenuation = np.exp(-(coefficients.T @ paths))
    bins = weights * response * attenuation
    mean = bins.sum(axis=0)
    scale = np.asarray(seed, dtype=np.float64).reshape(-1)
    logarithmic = problem.spectral.spec.output_domain == "log"
    values = np.log(mean) if logarithmic else mean
    if logarithmic:
        scale = scale / mean
    bin_seed = bins * scale
    path_gradient = -coefficients @ bin_seed
    weight_gradient = response * attenuation * scale
    response_gradient = weights * attenuation * scale
    if problem.spectral.spec.shared_weights:
        weight_gradient = weight_gradient.sum(axis=1)
    if problem.spectral.spec.shared_response:
        response_gradient = response_gradient.sum(axis=1)
    gradients = {
        "fields": (path_gradient @ matrix).reshape(fields.shape),
        "coefficients": -(paths @ bin_seed.T),
        "weights": weight_gradient,
        "response": response_gradient,
    }
    return values.reshape(projection.geometry.shape), gradients
