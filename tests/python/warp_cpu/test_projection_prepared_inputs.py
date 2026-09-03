"""Owned projection snapshots reuse checks while retaining gradient contracts."""

# pyright: reportUnknownMemberType=false

import importlib
import math
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from dpt.contracts import ContractError, NumericalError, TrialDomainError
from dpt.geometry import RayBundleGeometry, RigidTransform
from dpt.projection import (
    ProjectionSpec,
    prepare_projection,
    project_optical_depth,
    projection_vjp,
)
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


def _fixture(device: str, **options: Any) -> tuple[Any, ...]:
    grid = GridSpec((2, 2, 2), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((0.25, 0.25, -2.0),), ((0.25, 0.25, 3.0),))
    spec = ProjectionSpec(23, active_pose=False, active_volume=True, precision="float64", **options)
    workspace = prepare_projection(grid, rays, spec, device=device)
    field = wp.array([0.125] * grid.voxels, dtype=wp.float32, device=device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=device)
    output = wp.empty(rays.pixels, dtype=wp.float64, device=device)
    return workspace, field, pose, output


def test_snapshot_is_owned_and_reuses_field_validation(
    warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, field, pose, output = _fixture(warp_device)
    checked: list[Any] = []
    original = wp.launch

    def counted(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        if kernel is workspace._kernels.validate_nonnegative:
            checked.append(kernel)
        return original(kernel, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(wp, "launch", counted)
        inputs = workspace.prepare_inputs(field, pose)
        assert len(checked) == 1
        assert inputs.device_bytes == 8 * 4 + 12 * 8
        field.fill_(-1.0)
        pose.fill_(float("nan"))
        for _ in range(3):
            project_optical_depth(inputs, workspace=workspace, out_L=output)
            assert output.numpy().tolist() == pytest.approx([0.25], abs=3e-14)
        assert len(checked) == 1
        project_optical_depth(inputs, workspace=workspace, out_L=output, validate="always")
        assert len(checked) == 2
    with pytest.raises(FrozenInstanceError):
        inputs._mu = field


def test_raw_default_detects_mutation_and_explicit_once_requires_snapshot(warp_device: str) -> None:
    workspace, field, pose, output = _fixture(warp_device)
    project_optical_depth(field, pose, workspace=workspace, out_L=output)
    field.fill_(-1.0)
    with pytest.raises(TrialDomainError, match="non-negative"):
        project_optical_depth(field, pose, workspace=workspace, out_L=output)
    workspace.clear_status()
    with pytest.raises(ContractError, match="prepare_inputs"):
        project_optical_depth(field, pose, workspace=workspace, out_L=output, validate="once")


def test_snapshot_rejects_other_workspace_and_changed_configuration(warp_device: str) -> None:
    workspace, field, pose, output = _fixture(warp_device)
    inputs = workspace.prepare_inputs(field, pose)
    other, _, _, _ = _fixture(warp_device)
    with pytest.raises(ContractError, match="original workspace"):
        project_optical_depth(inputs, workspace=other, out_L=output)
    with pytest.raises(ContractError, match="owns its pose"):
        project_optical_depth(inputs, pose, workspace=workspace, out_L=output)
    workspace.grid = GridSpec((1, 1, 8), (1.0, 1.0, 1.0))
    with pytest.raises(ContractError, match="original workspace"):
        project_optical_depth(inputs, workspace=workspace, out_L=output)


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_snapshot_vjp_matches_raw_and_independent_constant_field_derivative(
    integration: Any, precision: Any, warp_device: str
) -> None:
    workspace, field, pose, output = _fixture(warp_device, integration=integration)
    inputs = workspace.prepare_inputs(field, pose)
    cotangent = wp.array([1.75], dtype=wp.float64, device=warp_device)
    raw = wp.empty(8, dtype=getattr(wp, precision), device=warp_device)
    prepared = wp.empty(8, dtype=getattr(wp, precision), device=warp_device)
    projection_vjp(
        field,
        pose,
        cotangent=cotangent,
        workspace=workspace,
        out_mu=raw,
        volume_gradient_precision=precision,
    )
    projection_vjp(
        inputs,
        cotangent=cotangent,
        workspace=workspace,
        out_mu=prepared,
        volume_gradient_precision=precision,
    )
    assert prepared.numpy().tolist() == pytest.approx(raw.numpy().tolist(), abs=3e-7)
    # A spatially constant field has integral 2 mu along this two-mm segment.
    assert float(prepared.numpy().sum()) == pytest.approx(3.5, abs=3e-7)
    step = 2.0**-12
    values = []
    for sign in (-1.0, 1.0):
        field.fill_(0.125 + sign * step)
        changed = workspace.prepare_inputs(field, pose)
        project_optical_depth(changed, workspace=workspace, out_L=output)
        values.append(1.75 * float(output.numpy()[0]))
    assert (values[1] - values[0]) / (2 * step) == pytest.approx(3.5, abs=3e-12)


def test_once_vjp_checks_current_cotangent_before_writing(warp_device: str) -> None:
    workspace, field, pose, _ = _fixture(warp_device)
    inputs = workspace.prepare_inputs(field, pose)
    cotangent = wp.array([float("nan")], dtype=wp.float64, device=warp_device)
    gradient = wp.full(8, 19.0, dtype=wp.float32, device=warp_device)
    with pytest.raises(NumericalError):
        projection_vjp(inputs, cotangent=cotangent, workspace=workspace, out_mu=gradient)
    assert gradient.numpy().tolist() == [19.0] * 8


def test_once_vjp_combines_pending_status_and_cotangent_checkpoint(
    warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, field, pose, _ = _fixture(warp_device)
    inputs = workspace.prepare_inputs(field, pose)
    cotangent = wp.ones(1, dtype=wp.float64, device=warp_device)
    gradient = wp.empty(8, dtype=wp.float32, device=warp_device)
    original = type(workspace._status_readback).read
    reads: list[Any] = []

    def counted(readback: Any, *args: Any, **kwargs: Any) -> Any:
        if readback is workspace._status_readback:
            reads.append(readback)
        return original(readback, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(type(workspace._status_readback), "read", counted)
        projection_vjp(inputs, cotangent=cotangent, workspace=workspace, out_mu=gradient)
    assert len(reads) == 2
    workspace._status.fill_(2)
    gradient.fill_(19.0)
    with pytest.raises(NumericalError):
        projection_vjp(inputs, cotangent=cotangent, workspace=workspace, out_mu=gradient)
    assert gradient.numpy().tolist() == [19.0] * 8


@pytest.mark.parametrize("policy", ["never", False])
def test_explicit_unchecked_policy_defers_status(warp_device: str, policy: Any) -> None:
    workspace, field, pose, output = _fixture(warp_device)
    field.fill_(float("nan"))
    project_optical_depth(field, pose, workspace=workspace, out_L=output, validate=policy)
    with pytest.raises(NumericalError):
        workspace.check_status()


@pytest.mark.parametrize("policy", ["always", True])
def test_checked_policy_preserves_boolean_compatibility(warp_device: str, policy: Any) -> None:
    workspace, field, pose, output = _fixture(warp_device)
    field.fill_(-1.0)
    with pytest.raises(TrialDomainError):
        project_optical_depth(field, pose, workspace=workspace, out_L=output, validate=policy)


def test_preparation_checks_shapes_before_copy_and_cannot_hide_invalid_fields(
    warp_device: str,
) -> None:
    workspace, field, pose, _ = _fixture(warp_device)
    short = wp.ones(7, dtype=wp.float32, device=warp_device)
    with pytest.raises(ContractError, match="shape"):
        workspace.prepare_inputs(short, pose)
    field.fill_(-1.0)
    with pytest.raises(TrialDomainError):
        workspace.prepare_inputs(field, pose)


def test_invalid_validation_policy_rejected(warp_device: str) -> None:
    with pytest.raises(ContractError, match="validate"):
        ProjectionSpec(1, validate=cast(Any, "sometimes"))
    workspace, field, pose, output = _fixture(warp_device)
    with pytest.raises(ContractError, match="validate"):
        project_optical_depth(field, pose, workspace=workspace, out_L=output, validate=cast(Any, 1))


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_warm_compiles_selected_variants_without_array_allocations(
    integration: Any, warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _, _, _ = _fixture(warp_device, integration=integration)
    allocations: list[str] = []
    original = wp.empty

    def counted(*args: Any, **kwargs: Any) -> Any:
        allocations.append("empty")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(wp, "empty", counted)
        modules = workspace.warm(volume_gradient_precision="float64")
        assert modules
        assert not allocations
        assert workspace.warm(volume_gradient_precision="float64") == modules


def test_warm_rejects_inactive_vjp_and_bad_flags(warp_device: str) -> None:
    workspace, _, _, _ = _fixture(warp_device)
    with pytest.raises(ContractError, match="booleans"):
        workspace.warm(forward=cast(Any, 1))
    with pytest.raises(ContractError, match="active pose"):
        workspace.warm(matrix_gradient=True)
    workspace.spec = ProjectionSpec(3, active_pose=False)
    with pytest.raises(ContractError, match="active input"):
        workspace.warm(vjp=True)
    assert workspace.warm()


def test_pose_validation_checks_rigidity_without_scanning_field(
    warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _, pose, _ = _fixture(warp_device)
    original = wp.launch
    kernels: list[Any] = []

    def counted(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        kernels.append(kernel)
        return original(kernel, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(wp, "launch", counted)
        workspace.validate_pose(pose)
    assert kernels == [workspace._geometry_kernels.validate_pose]
    invalid = wp.array([0.0] * 12, dtype=wp.float64, device=warp_device)
    with pytest.raises(ContractError, match="rigid pose"):
        workspace.validate_pose(invalid)


@pytest.mark.parametrize("matrix_gradient", [False, True])
def test_warm_prepares_pose_reductions_without_changing_results(
    matrix_gradient: bool, warp_device: str
) -> None:
    grid = GridSpec((2, 2, 2), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((0.25, 0.25, -2.0),), ((0.25, 0.25, 3.0),))
    workspace = prepare_projection(
        grid, rays, ProjectionSpec(23, precision="float64"), device=warp_device
    )
    assert workspace.warm(matrix_gradient=matrix_gradient)
    field = wp.array([0.125] * grid.voxels, dtype=wp.float32, device=warp_device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    cotangent = wp.ones(1, dtype=wp.float64, device=warp_device)
    gradient = wp.empty(12 if matrix_gradient else 6, dtype=wp.float64, device=warp_device)
    kwargs = {"out_pose_matrix" if matrix_gradient else "out_pose_tangent": gradient}
    projection_vjp(field, pose, cotangent=cotangent, workspace=workspace, **kwargs)
    assert all(math.isfinite(float(value)) for value in gradient.numpy())


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_zero_pose_cotangents_preserve_mixed_seed_gradients(
    integration: Any, warp_device: str
) -> None:
    grid = GridSpec((2, 2, 2), (1.0, 1.0, 1.0))
    field = wp.array(
        [0.125 + 0.03125 * (voxel % 2) for voxel in range(8)], dtype=wp.float32, device=warp_device
    )
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    cases = (
        (
            RayBundleGeometry(
                ((0.25, 0.25, -2.0), (0.75, 0.25, -2.0)), ((0.25, 0.25, 3.0), (0.75, 0.25, 3.0))
            ),
            [0.0, 1.75],
        ),
        (RayBundleGeometry(((0.75, 0.25, -2.0),), ((0.75, 0.25, 3.0),)), [1.75]),
    )
    gradients = []
    for rays, seeds in cases:
        workspace = prepare_projection(
            grid,
            rays,
            ProjectionSpec(31, active_volume=True, precision="float64", integration=integration),
            device=warp_device,
        )
        cotangent = wp.array(seeds, dtype=wp.float64, device=warp_device)
        out_pose = wp.empty(6, dtype=wp.float64, device=warp_device)
        out_mu = wp.empty(8, dtype=wp.float64, device=warp_device)
        projection_vjp(
            field,
            pose,
            cotangent=cotangent,
            workspace=workspace,
            out_pose_tangent=out_pose,
            out_mu=out_mu,
            volume_gradient_precision="float64",
        )
        gradients.append((out_pose.numpy().tolist(), out_mu.numpy().tolist()))
        cotangent.zero_()
        projection_vjp(
            field,
            pose,
            cotangent=cotangent,
            workspace=workspace,
            out_pose_tangent=out_pose,
            out_mu=out_mu,
            volume_gradient_precision="float64",
        )
        assert out_pose.numpy().tolist() == [0.0] * 6
        assert out_mu.numpy().tolist() == [0.0] * 8
    for index in (0, 1):
        assert gradients[0][index] == pytest.approx(gradients[1][index], abs=3e-14)


def test_zero_pose_cotangent_keeps_invalid_ray_diagnostics(warp_device: str) -> None:
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((0.0, 0.0, 0.0),), ((0.0, 0.0, 1.0),))
    workspace = prepare_projection(grid, rays, ProjectionSpec(3), device=warp_device)
    workspace._configuration.ray_endpoints.zero_()
    field = wp.ones(1, dtype=wp.float32, device=warp_device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    cotangent = wp.zeros(1, dtype=wp.float32, device=warp_device)
    gradient = wp.empty(6, dtype=wp.float64, device=warp_device)
    with pytest.raises(NumericalError):
        projection_vjp(
            field, pose, cotangent=cotangent, workspace=workspace, out_pose_tangent=gradient
        )
