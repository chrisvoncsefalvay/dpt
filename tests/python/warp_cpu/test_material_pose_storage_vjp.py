"""Material packed-pose VJPs follow native storage order and scalar derivatives."""

import importlib
from types import SimpleNamespace
from typing import Any, Literal, cast

import numpy as np
import pytest

from dpt.contracts import ContractError, NumericalError, TrialDomainError
from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose
from dpt.material_projection import (
    material_projection_vjp,
    prepare_material_projection,
    project_material_paths,
)
from dpt.projection import ProjectionSpec, projection_vjp
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


def _workspace(
    device: str,
    *,
    precision: Literal["float32", "float64"] = "float64",
    integration: Literal["midpoint", "cell_gauss"] = "midpoint",
    execution: Literal["serial", "fused"] = "serial",
    active_pose: bool = True,
    active_fields: bool = True,
) -> tuple[Any, Any]:
    angle = 0.19
    orientation = (
        float(np.cos(angle)),
        -float(np.sin(angle)),
        0.0,
        float(np.sin(angle)),
        float(np.cos(angle)),
        0.0,
        0.0,
        0.0,
        1.0,
    )
    grid = GridSpec((3, 4, 5), (0.7, 1.1, 0.9), (-1.4, -1.5, -1.0), orientation)
    geometry = DetectorGeometry(
        (-12.0, -0.13, 0.23),
        (12.0, -0.57, -0.47),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.37, 0.29),
        (2, 3),
    )
    rng = np.random.default_rng(3819)
    fields = wp.array(
        rng.uniform(0.04, 0.16, (3, grid.voxels)).astype(np.float32),
        dtype=wp.float32,
        device=device,
    )
    dtype = getattr(wp, precision)
    seeds = rng.normal(size=(3, geometry.pixels))
    seeds[1, ::2] = 0.0
    workspace = prepare_material_projection(
        grid,
        geometry,
        ProjectionSpec(
            32,
            precision=precision,
            integration=integration,
            active_pose=active_pose,
            active_volume=active_fields,
        ),
        materials=3,
        fields=fields,
        out_paths=wp.empty((3, geometry.pixels), dtype=dtype, device=device),
        cotangent_paths=wp.array(seeds, dtype=dtype, device=device),
        out_fields=(
            wp.full((3, grid.voxels), 123.0, dtype=wp.float32, device=device)
            if active_fields
            else None
        ),
        device=device,
        channel_execution=execution,
    )
    transform = compose_pose(RigidTransform(), (0.07, -0.03, 0.02, 0.03, -0.02, 0.01))
    pose = wp.array(transform.packed(), dtype=wp.float64, device=device)
    return workspace, pose


def _native_sum(workspace: Any, pose: Any) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros(12, dtype=np.float64)
    gradients = []
    for material in range(workspace.materials):
        pose_gradient = wp.empty(12, dtype=wp.float64, device=workspace.context.device)
        field_gradient = (
            wp.empty(
                workspace.projection.grid.voxels,
                dtype=wp.float32,
                device=workspace.context.device,
            )
            if workspace.grad_fields is not None
            else None
        )
        projection_vjp(
            workspace.field_views[material],
            pose,
            cotangent=workspace.seed_views[material],
            workspace=workspace.projection,
            out_pose_matrix=pose_gradient,
            out_mu=field_gradient,
        )
        matrix += pose_gradient.numpy()
        if field_gradient is not None:
            gradients.append(field_gradient.numpy())
    return matrix, np.asarray(gradients).reshape(-1)


def _scalar_objective(workspace: Any, packed: np.ndarray) -> float:
    # The scalar reference expands R^T directly. These independent storage
    # perturbations deliberately leave the rigid manifold; a local-six chart
    # would exercise a different chain rule.
    pose: Any = SimpleNamespace(rotation=tuple(packed[:9]), translation_mm=tuple(packed[9:]))
    fields = workspace.fields.numpy().reshape(workspace.materials, -1)
    seeds = workspace.cotangent_paths.numpy().reshape(workspace.materials, -1)
    grid, geometry = workspace.projection.grid, workspace.projection.geometry
    return sum(
        float(seeds[material, pixel])
        * integrate_sampled_field(
            fields[material].tolist(),
            grid,
            geometry,
            pose,
            *divmod(pixel, geometry.shape[1]),
            midpoint_samples=(
                workspace.projection.spec.samples_per_ray
                if workspace.projection.spec.integration == "midpoint"
                else None
            ),
            validate=False,
        )
        for material in range(workspace.materials)
        for pixel in range(geometry.pixels)
    )


def _assert_fd_scene_is_away_from_knots(workspace: Any, packed: np.ndarray) -> None:
    grid, geometry = workspace.projection.grid, workspace.projection.geometry
    orientation = np.asarray(grid.orientation).reshape(3, 3)
    rotation, translation = packed[:9].reshape(3, 3), packed[9:]
    shape, spacing = np.asarray(tuple(reversed(grid.shape))), np.asarray(grid.spacing_mm)
    source = np.asarray(geometry.source_mm)
    origin = (
        orientation.T @ (rotation.T @ (source - translation) - np.asarray(grid.origin_mm))
    ) / spacing
    for pixel in range(geometry.pixels):
        endpoint = np.asarray(geometry.pixel_centre(*divmod(pixel, geometry.shape[1])))
        direction = (orientation.T @ (rotation.T @ (endpoint - source))) / spacing
        faces = np.sort(
            np.asarray([(-0.5 - origin) / direction, (shape - 0.5 - origin) / direction]),
            axis=0,
        )
        lower_candidates = sorted([0.0, *faces[0]], reverse=True)
        upper_candidates = sorted([1.0, *faces[1]])
        lower, upper = lower_candidates[0], upper_candidates[0]
        assert lower_candidates[0] - lower_candidates[1] > 1e-4
        assert upper_candidates[1] - upper_candidates[0] > 1e-4
        positions = (
            lower
            + (np.arange(workspace.projection.spec.samples_per_ray) + 0.5)
            * (upper - lower)
            / workspace.projection.spec.samples_per_ray
        )
        points = origin + positions[:, None] * direction
        planes = [lower, upper]
        for axis in range(3):
            # Midpoint nodes avoid interpolation/clamp knots; cell-Gauss
            # boundaries avoid simultaneous crossings on different axes.
            assert np.min(np.abs(points[:, axis, None] - np.arange(shape[axis]))) > 1e-4
            planes.extend(
                value
                for value in (np.arange(shape[axis]) - origin[axis]) / direction[axis]
                if lower < value < upper
            )
        assert np.min(np.diff(sorted(planes))) > 1e-4


@pytest.mark.parametrize("execution", ["serial", "fused"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_material_packed_pose_equals_native_sum_and_preserves_field_accumulation(
    execution: Literal["serial", "fused"],
    precision: Literal["float32", "float64"],
    integration: Literal["midpoint", "cell_gauss"],
    warp_device: str,
) -> None:
    workspace, pose = _workspace(
        warp_device, precision=precision, integration=integration, execution=execution
    )
    gradient = wp.full(12, 123.0, dtype=wp.float64, device=warp_device)
    material_projection_vjp(pose, workspace=workspace, out_pose_matrix=gradient)
    first_pose, first_fields = gradient.numpy().copy(), workspace.grad_fields.numpy().copy()
    native_pose, native_fields = _native_sum(workspace, pose)
    np.testing.assert_allclose(first_pose, native_pose, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(first_fields, native_fields, rtol=2e-6, atol=2e-7)
    assert np.linalg.norm(first_pose[:9]) > 0.01
    assert np.linalg.norm(first_pose[9:]) > 0.01
    material_projection_vjp(pose, workspace=workspace, out_pose_matrix=gradient, accumulate=True)
    np.testing.assert_allclose(gradient.numpy(), 2 * first_pose, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(
        workspace.grad_fields.numpy(), 2 * first_fields, rtol=3e-6, atol=3e-7
    )


@pytest.mark.parametrize("execution", ["serial", "fused"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_material_storage_axes_and_direction_follow_independent_scalar_derivatives(
    execution: Literal["serial", "fused"],
    precision: Literal["float32", "float64"],
    integration: Literal["midpoint", "cell_gauss"],
    warp_device: str,
) -> None:
    workspace, pose = _workspace(
        warp_device,
        precision=precision,
        integration=integration,
        execution=execution,
        active_fields=False,
    )
    gradient = wp.empty(12, dtype=wp.float64, device=warp_device)
    material_projection_vjp(pose, workspace=workspace, out_pose_matrix=gradient)
    packed, cotangent = pose.numpy(), gradient.numpy()
    _assert_fd_scene_is_away_from_knots(workspace, packed)
    directions = [*np.eye(12), np.random.default_rng(924).normal(size=12)]
    epsilon = 2e-6
    for direction in directions:
        plus = _scalar_objective(workspace, packed + epsilon * direction)
        minus = _scalar_objective(workspace, packed - epsilon * direction)
        assert (plus - minus) / (2 * epsilon) == pytest.approx(
            float(cotangent @ direction), rel=3e-7, abs=3e-8
        )


@pytest.mark.parametrize("execution", ["serial", "fused"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_material_packed_storage_chain_matches_local_right_pose_tangent(
    execution: Literal["serial", "fused"],
    precision: Literal["float32", "float64"],
    integration: Literal["midpoint", "cell_gauss"],
    warp_device: str,
) -> None:
    workspace, pose = _workspace(
        warp_device, precision=precision, integration=integration, execution=execution
    )
    packed_gradient = wp.empty(12, dtype=wp.float64, device=warp_device)
    local_gradient = wp.empty(6, dtype=wp.float64, device=warp_device)
    material_projection_vjp(pose, workspace=workspace, out_pose_matrix=packed_gradient)
    packed_fields = workspace.grad_fields.numpy().copy()
    material_projection_vjp(pose, workspace=workspace, out_pose_tangent=local_gradient)
    np.testing.assert_allclose(workspace.grad_fields.numpy(), packed_fields, rtol=2e-6, atol=2e-7)
    rotation = pose.numpy()[:9].reshape(3, 3)
    matrix, translation = (
        packed_gradient.numpy()[:9].reshape(3, 3),
        packed_gradient.numpy()[9:],
    )
    # T @ exp(delta) gives dR = R [omega]_x and dt = R u at zero.
    skew_axes = (
        np.asarray([[0, 0, 0], [0, 0, -1], [0, 1, 0]]),
        np.asarray([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]),
        np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 0]]),
    )
    expected = np.r_[
        rotation.T @ translation,
        [np.sum(matrix * (rotation @ skew)) for skew in skew_axes],
    ]
    np.testing.assert_allclose(local_gradient.numpy(), expected, rtol=3e-12, atol=3e-13)


@pytest.mark.parametrize("execution", ["serial", "fused"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_zero_material_seeds_overwrite_or_preserve_both_gradient_destinations(
    execution: Literal["serial", "fused"],
    precision: Literal["float32", "float64"],
    warp_device: str,
) -> None:
    workspace, pose = _workspace(warp_device, precision=precision, execution=execution)
    workspace.cotangent_paths.zero_()
    gradient = wp.full(12, 7.0, dtype=wp.float64, device=warp_device)
    material_projection_vjp(pose, workspace=workspace, out_pose_matrix=gradient, accumulate=True)
    np.testing.assert_array_equal(gradient.numpy(), 7.0)
    np.testing.assert_array_equal(workspace.grad_fields.numpy(), 123.0)
    material_projection_vjp(pose, workspace=workspace, out_pose_matrix=gradient)
    np.testing.assert_array_equal(gradient.numpy(), 0.0)
    np.testing.assert_array_equal(workspace.grad_fields.numpy(), 0.0)


@pytest.mark.parametrize("execution", ["serial", "fused"])
def test_material_storage_gradient_contract_errors_precede_destination_writes(
    execution: Literal["serial", "fused"], warp_device: str
) -> None:
    workspace, pose = _workspace(warp_device, execution=execution)
    gradient = wp.full(12, 7.0, dtype=wp.float64, device=warp_device)
    local = wp.full(6, 9.0, dtype=wp.float64, device=warp_device)
    for keywords, message in (
        ({"out_pose_tangent": local, "out_pose_matrix": gradient}, "either a right-tangent"),
        ({"out_pose_matrix": local}, "shape"),
        (
            {"out_pose_matrix": wp.full(12, 7.0, dtype=wp.float32, device=warp_device)},
            "dtype",
        ),
        ({"out_pose_matrix": pose}, "overlaps"),
    ):
        with pytest.raises(ContractError, match=message):
            material_projection_vjp(pose, workspace=workspace, **keywords)
    np.testing.assert_array_equal(gradient.numpy(), 7.0)
    np.testing.assert_array_equal(local.numpy(), 9.0)
    np.testing.assert_array_equal(workspace.grad_fields.numpy(), 123.0)


@pytest.mark.parametrize("execution", ["serial", "fused"])
@pytest.mark.parametrize("active_fields", [False, True])
def test_fixed_material_pose_rejects_storage_gradient_before_field_write(
    execution: Literal["serial", "fused"], active_fields: bool, warp_device: str
) -> None:
    workspace, pose = _workspace(
        warp_device, execution=execution, active_pose=False, active_fields=active_fields
    )
    gradient = wp.full(12, 7.0, dtype=wp.float64, device=warp_device)
    with pytest.raises(ContractError, match="pose is fixed"):
        material_projection_vjp(pose, workspace=workspace, out_pose_matrix=gradient)
    np.testing.assert_array_equal(gradient.numpy(), 7.0)
    if active_fields:
        np.testing.assert_array_equal(workspace.grad_fields.numpy(), 123.0)


@pytest.mark.parametrize("execution", ["serial", "fused"])
def test_invalid_material_inputs_and_pending_status_precede_packed_pose_writes(
    execution: Literal["serial", "fused"], warp_device: str
) -> None:
    workspace, pose = _workspace(warp_device, execution=execution)
    gradient = wp.full(12, 7.0, dtype=wp.float64, device=warp_device)
    workspace.cotangent_paths.fill_(float("nan"))
    with pytest.raises(ContractError, match="cotangents"):
        material_projection_vjp(pose, workspace=workspace, out_pose_matrix=gradient)
    np.testing.assert_array_equal(gradient.numpy(), 7.0)
    np.testing.assert_array_equal(workspace.grad_fields.numpy(), 123.0)
    workspace.cotangent_paths.zero_()
    workspace.projection._status.fill_(2)
    with pytest.raises(NumericalError):
        material_projection_vjp(pose, workspace=workspace, out_pose_matrix=gradient)
    np.testing.assert_array_equal(gradient.numpy(), 7.0)
    with pytest.raises(NumericalError):
        workspace.consume_status()
    workspace.fields.fill_(-0.01)
    with pytest.raises(TrialDomainError):
        material_projection_vjp(pose, workspace=workspace, out_pose_matrix=gradient)
    np.testing.assert_array_equal(gradient.numpy(), 7.0)


@pytest.mark.parametrize("execution", ["serial", "fused"])
def test_material_checked_forward_and_storage_vjp_reject_ambient_tape(
    execution: Literal["serial", "fused"], warp_device: str
) -> None:
    workspace, pose = _workspace(warp_device, execution=execution, active_fields=False)
    gradient = wp.full(12, 7.0, dtype=wp.float64, device=warp_device)
    with wp.Tape(), pytest.raises(ContractError, match="ambient"):
        material_projection_vjp(pose, workspace=workspace, out_pose_matrix=gradient)
    with wp.Tape(), pytest.raises(ContractError, match="ambient"):
        project_material_paths(pose, workspace=workspace)
    np.testing.assert_array_equal(gradient.numpy(), 7.0)


@pytest.mark.parametrize("execution", ["serial", "fused"])
def test_material_packed_pose_preserves_deferred_volume_check_contract(
    execution: Literal["serial", "fused"], warp_device: str
) -> None:
    workspace, pose = _workspace(warp_device, execution=execution)
    gradient = wp.full(12, 7.0, dtype=wp.float64, device=warp_device)
    for kwargs in ({"validate": True, "accumulate": True}, {"validate": False}):
        with pytest.raises(ContractError, match="unchecked accumulating"):
            material_projection_vjp(
                pose,
                workspace=workspace,
                out_pose_matrix=gradient,
                check_volume_gradient=False,
                **cast(Any, kwargs),
            )
    np.testing.assert_array_equal(gradient.numpy(), 7.0)
    np.testing.assert_array_equal(workspace.grad_fields.numpy(), 123.0)
    native_pose, native_fields = _native_sum(workspace, pose)
    material_projection_vjp(
        pose,
        workspace=workspace,
        out_pose_matrix=gradient,
        validate=False,
        accumulate=True,
        check_volume_gradient=False,
    )
    for view in workspace.gradient_views:
        workspace.projection.enqueue_volume_gradient_check(view)
    workspace.consume_status()
    np.testing.assert_allclose(gradient.numpy(), 7.0 + native_pose, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(
        workspace.grad_fields.numpy(), 123.0 + native_fields, rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize("execution", ["serial", "fused"])
def test_invalid_material_pose_is_rejected_before_any_packed_or_field_write(
    execution: Literal["serial", "fused"], warp_device: str
) -> None:
    workspace, pose = _workspace(warp_device, execution=execution)
    pose.fill_(float("nan"))
    gradient = wp.full(12, 7.0, dtype=wp.float64, device=warp_device)
    with pytest.raises(ContractError, match="rigid"):
        material_projection_vjp(pose, workspace=workspace, out_pose_matrix=gradient)
    np.testing.assert_array_equal(gradient.numpy(), 7.0)
    np.testing.assert_array_equal(workspace.grad_fields.numpy(), 123.0)
