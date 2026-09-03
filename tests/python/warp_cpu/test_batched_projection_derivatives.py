"""Finite-segment, per-view pose and cone calibration derivative oracles."""

import importlib
import math
from dataclasses import replace
from typing import Any, Literal

import pytest

from dpt.batched_projection import (
    batched_projection_vjp,
    prepare_batched_projection,
    prepare_cone_projection,
    project_batched_optical_depth,
    projection_ray_vjp,
)
from dpt.contracts import ContractError, NumericalError
from dpt.geometry import (
    BatchedGeometry,
    ConeBeamIntrinsics,
    CurvedDetectorGeometry,
    ParallelGeometry,
    RayBundleGeometry,
    RigidTransform,
    compose_pose,
)
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth
from dpt.validation.projection import integrate_ray_bundle, integrate_sampled_field
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("storage", ["float32", "float16", "bfloat16"])
def test_endpoint_cotangents_include_segment_norm_with_analytic_affine_field(
    integration: Literal["midpoint", "cell_gauss"],
    storage: Literal["float32", "float16", "bfloat16"],
    warp_device: str,
) -> None:
    grid = GridSpec((9, 9, 9), (1.0, 1.0, 1.0), (-4.0, -4.0, -4.0))
    source, end = (-1.2, -1.3, -0.7), (1.7, 2.3, 1.2)
    beta = (1 / 64, -1 / 32, 1 / 128)
    values = [
        0.5 + beta[0] * (x - 4) + beta[1] * (y - 4) + beta[2] * (z - 4)
        for z in range(9)
        for y in range(9)
        for x in range(9)
    ]
    pose_host = compose_pose(RigidTransform(), (0.12, -0.21, 0.17, 0.06, -0.04, 0.09))
    workspace = prepare_projection(
        grid,
        RayBundleGeometry((source,), (end,)),
        ProjectionSpec(73, precision="float64", integration=integration, storage_precision=storage),
        device=warp_device,
    )
    mu = wp.array(values, dtype=getattr(wp, storage), device=warp_device)
    pose = wp.array(pose_host.packed(), dtype=wp.float64, device=warp_device)
    seed = wp.array([-1.7], dtype=wp.float64, device=warp_device)
    out = wp.empty((1, 6), dtype=wp.float64, device=warp_device)
    projection_ray_vjp(mu, pose, cotangent=seed, workspace=workspace, out_ray_coordinates=out)
    object_source, object_end = pose_host.inverse().point(source), pose_host.inverse().point(end)
    average = 0.5 + sum(beta[i] * (object_source[i] + object_end[i]) / 2 for i in range(3))
    length = math.dist(source, end)
    beta_world = pose_host.direction(beta)
    expected = [
        -1.7 * (sign * average * (end[i] - source[i]) / length + length * beta_world[i] / 2)
        for sign in (-1, 1)
        for i in range(3)
    ]
    assert out.numpy()[0] == pytest.approx(expected, rel=3e-12, abs=3e-12)


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_ray_coordinate_vjp_matches_central_differences_for_clipped_oblique_segments(
    integration: Literal["midpoint", "cell_gauss"], warp_device: str
) -> None:
    grid, intrinsics, angles, poses_host, values = _case()
    panel = intrinsics.view(angles[0])
    source, end = panel.source_mm, panel.pixel_centre(1, 3)
    spec = ProjectionSpec(73, precision="float64", integration=integration)
    workspace = prepare_projection(
        grid, RayBundleGeometry((source,), (end,)), spec, device=warp_device
    )
    mu = wp.array(values, dtype=wp.float32, device=warp_device)
    pose = wp.array(poses_host[0].packed(), dtype=wp.float64, device=warp_device)
    seed = wp.array([0.73], dtype=wp.float64, device=warp_device)
    out = wp.empty((1, 6), dtype=wp.float64, device=warp_device)
    projection_ray_vjp(mu, pose, cotangent=seed, workspace=workspace, out_ray_coordinates=out)
    actual = out.numpy()[0]
    for component in range(6):
        losses: list[float] = []
        for sign in (-1, 1):
            coordinates = list((*source, *end))
            coordinates[component] += sign * 1e-5
            shifted = prepare_projection(
                grid,
                RayBundleGeometry(
                    ((coordinates[0], coordinates[1], coordinates[2]),),
                    ((coordinates[3], coordinates[4], coordinates[5]),),
                ),
                spec,
                device=warp_device,
            )
            depth = wp.empty(1, dtype=wp.float64, device=warp_device)
            project_optical_depth(mu, pose, workspace=shifted, out_L=depth)
            losses.append(0.73 * float(depth.numpy()[0]))
        assert actual[component] == pytest.approx(
            (losses[1] - losses[0]) / 2e-5, rel=4e-6, abs=3e-8
        )


def _case() -> tuple[
    GridSpec, ConeBeamIntrinsics, tuple[float, ...], tuple[RigidTransform, ...], list[float]
]:
    grid = GridSpec((7, 8, 9), (0.9, 1.1, 1.2), (-3.6, -3.85, -3.6))
    intrinsics = ConeBeamIntrinsics(12.0, 30.0, (3, 5), (1.3, 1.1), (0.21, -0.37))
    angles = (0.13, 1.21, 2.4)
    poses = tuple(
        compose_pose(
            RigidTransform(), (0.13 * i, -0.11 * i, 0.07 * i, 0.021, -0.034 * i, 0.013 * i)
        )
        for i in range(1, 4)
    )
    values = [
        1 / 32 + (x + 2 * y + 3 * z + x * y) / 4096
        for z in range(7)
        for y in range(8)
        for x in range(9)
    ]
    return grid, intrinsics, angles, poses, values


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_batched_pose_intrinsic_volume_vjps_match_independent_views_and_finite_differences(
    integration: Literal["midpoint", "cell_gauss"], warp_device: str
) -> None:
    grid, intrinsics, angles, poses_host, values = _case()
    spec = ProjectionSpec(73, precision="float64", integration=integration, active_volume=True)
    workspace = prepare_cone_projection(
        grid, intrinsics, angles, spec, active_intrinsics=True, device=warp_device
    )
    mu = wp.array(values, dtype=wp.float32, device=warp_device)
    poses = wp.array(
        tuple(pose.packed() for pose in poses_host), dtype=wp.float64, ndim=2, device=warp_device
    )
    depth = wp.empty(workspace.geometry.shape, dtype=wp.float64, device=warp_device)
    weights = [
        [
            [(-1.0 if (row + col + view) % 2 else 1.0) * (1 + row + col / 7) for col in range(5)]
            for row in range(3)
        ]
        for view in range(3)
    ]
    seed = wp.array(weights, dtype=wp.float64, ndim=3, device=warp_device)
    pose_grad = wp.empty((3, 6), dtype=wp.float64, device=warp_device)
    calibration_grad = wp.empty((3, 4), dtype=wp.float64, device=warp_device)
    volume_grad = wp.empty(grid.voxels, dtype=wp.float64, device=warp_device)
    project_batched_optical_depth(mu, poses, workspace=workspace, out_L=depth)
    actual_depth = depth.numpy()
    for view in range(3):
        for row, col in ((0, 1), (1, 2), (2, 4)):
            reference = integrate_sampled_field(
                values,
                grid,
                intrinsics.view(angles[view]),
                poses_host[view],
                row,
                col,
                midpoint_samples=73 if integration == "midpoint" else None,
            )
            assert actual_depth[view, row, col] == pytest.approx(reference, rel=2e-12, abs=2e-12)
    batched_projection_vjp(
        mu,
        poses,
        cotangent=seed,
        workspace=workspace,
        out_pose_tangent=pose_grad,
        out_intrinsics=calibration_grad,
        out_mu=volume_grad,
        volume_gradient_precision="float64",
    )
    pose_values, calibration_values = pose_grad.numpy().copy(), calibration_grad.numpy().copy()
    inner_forward = sum(
        float(actual_depth[v, r, c]) * weights[v][r][c]
        for v in range(3)
        for r in range(3)
        for c in range(5)
    )
    assert sum(a * b for a, b in zip(values, volume_grad.numpy(), strict=True)) == pytest.approx(
        inner_forward, rel=3e-12, abs=3e-12
    )
    for view in range(3):
        for component in range(10):
            losses: list[float] = []
            for sign in (-1, 1):
                record = intrinsics
                host_pose = poses_host[view]
                step = 1e-5
                if component < 6:
                    increment = tuple(
                        sign * step if axis == component else 0.0 for axis in range(6)
                    )
                    host_pose = compose_pose(host_pose, increment)
                elif component == 6:
                    record = replace(
                        record,
                        source_object_distance_mm=record.source_object_distance_mm + sign * step,
                    )
                elif component == 7:
                    record = replace(
                        record,
                        source_detector_distance_mm=record.source_detector_distance_mm
                        + sign * step,
                    )
                else:
                    offsets = list(record.detector_offset_mm)
                    offsets[component - 8] += sign * step
                    record = replace(record, detector_offset_mm=tuple(offsets))
                panel = record.view(angles[view])
                loss = sum(
                    weights[view][row][col]
                    * integrate_sampled_field(
                        values,
                        grid,
                        panel,
                        host_pose,
                        row,
                        col,
                        midpoint_samples=73 if integration == "midpoint" else None,
                    )
                    for row in range(3)
                    for col in range(5)
                )
                losses.append(loss)
            expected = (losses[1] - losses[0]) / (2 * step)
            actual = (
                pose_values[view, component]
                if component < 6
                else calibration_values[view, component - 6]
            )
            assert actual == pytest.approx(expected, rel=8e-6, abs=2e-8), (
                view,
                component,
                actual,
                expected,
            )
    # Reusing the allocation and accumulating are explicit, independent of
    # which forward most recently used the shared workspace scratch.
    project_batched_optical_depth(mu, poses, workspace=workspace, out_L=depth)
    batched_projection_vjp(
        mu,
        poses,
        cotangent=seed,
        workspace=workspace,
        out_pose_tangent=pose_grad,
        out_intrinsics=calibration_grad,
        accumulate=True,
    )
    assert pose_grad.numpy() == pytest.approx(2 * pose_values, abs=2e-12)
    assert calibration_grad.numpy() == pytest.approx(2 * calibration_values, abs=2e-12)


def test_multiview_activity_shapes_aliases_and_pending_status_are_enforced(
    warp_device: str,
) -> None:
    grid, intrinsics, angles, poses_host, values = _case()
    spec = ProjectionSpec(73, active_pose=False, precision="float64")
    workspace = prepare_cone_projection(grid, intrinsics, angles, spec, device=warp_device)
    mu = wp.array(values, dtype=wp.float32, device=warp_device)
    poses = wp.array(
        tuple(pose.packed() for pose in poses_host), dtype=wp.float64, ndim=2, device=warp_device
    )
    seed = wp.ones(workspace.geometry.shape, dtype=wp.float64, device=warp_device)
    out_pose = wp.empty((3, 6), dtype=wp.float64, device=warp_device)
    with pytest.raises(ContractError, match="pose is fixed"):
        batched_projection_vjp(
            mu, poses, cotangent=seed, workspace=workspace, out_pose_tangent=out_pose
        )
    with pytest.raises(ContractError, match="intrinsics are fixed"):
        batched_projection_vjp(
            mu,
            poses,
            cotangent=seed,
            workspace=workspace,
            out_intrinsics=wp.empty((3, 4), dtype=wp.float64, device=warp_device),
        )
    with pytest.raises(ContractError, match="max_bytes"):
        prepare_batched_projection(
            grid, BatchedGeometry((intrinsics.view(0.0),)), spec, device=warp_device, max_bytes=1
        )
    active = prepare_cone_projection(
        grid, intrinsics, angles, replace(spec, active_pose=True), device=warp_device
    )
    with pytest.raises(ContractError, match="overlap"):
        batched_projection_vjp(
            mu,
            poses,
            cotangent=seed,
            workspace=active,
            out_pose_tangent=poses.flatten()[:18].reshape((3, 6)),
        )
    with active._projection.context.scope():
        active._projection._status.assign([2])
    with pytest.raises(NumericalError):
        batched_projection_vjp(
            mu, poses, cotangent=seed, workspace=active, out_pose_tangent=out_pose
        )
    active.clear_status()
    with pytest.raises(ContractError, match="strict"):
        prepare_cone_projection(
            grid,
            intrinsics,
            angles,
            replace(spec, sampling_precision="fast_float32"),
            active_intrinsics=True,
            device=warp_device,
        )


@pytest.mark.parametrize("pose_active", [False, True])
def test_segmented_tree_retains_all_non_square_view_pixels_across_multiple_levels(
    pose_active: bool, warp_device: str
) -> None:
    grid, record, _, poses_host, values = _case()
    angles = (0.31, 1.17)
    records = (
        replace(record, shape=(17, 19), spacing_mm=(0.3, 0.27)),
        replace(record, shape=(17, 19), spacing_mm=(0.28, 0.31), detector_offset_mm=(-0.11, 0.23)),
    )
    spec = ProjectionSpec(73, active_pose=pose_active, precision="float64")
    workspace = prepare_cone_projection(
        grid, records, angles, spec, active_intrinsics=True, device=warp_device
    )
    mu = wp.array(values, dtype=wp.float32, device=warp_device)
    poses = wp.array(
        tuple(pose.packed() for pose in poses_host[:2]),
        dtype=wp.float64,
        ndim=2,
        device=warp_device,
    )
    weights = [
        [[(-1.0 if (row + col) % 3 else 2.0) / 17 for col in range(19)] for row in range(17)]
        for _ in angles
    ]
    seed = wp.array(weights, dtype=wp.float64, ndim=3, device=warp_device)
    intrinsic = wp.empty((2, 4), dtype=wp.float64, device=warp_device)
    pose = wp.empty((2, 6), dtype=wp.float64, device=warp_device) if pose_active else None
    batched_projection_vjp(
        mu,
        poses,
        cotangent=seed,
        workspace=workspace,
        out_intrinsics=intrinsic,
        out_pose_tangent=pose,
    )
    actual = intrinsic.numpy()
    for view in range(2):
        step = 1e-5
        losses = []
        for sign in (-1, 1):
            shifted = replace(
                records[view],
                detector_offset_mm=(
                    records[view].detector_offset_mm[0] + sign * step,
                    records[view].detector_offset_mm[1],
                ),
            )
            losses.append(
                sum(
                    weights[view][row][col]
                    * integrate_sampled_field(
                        values,
                        grid,
                        shifted.view(angles[view]),
                        poses_host[view],
                        row,
                        col,
                        midpoint_samples=73,
                    )
                    for row in range(17)
                    for col in range(19)
                )
            )
        assert actual[view, 2] == pytest.approx(
            (losses[1] - losses[0]) / (2 * step), rel=6e-6, abs=2e-8
        )
    assert workspace.scratch_bytes < 256 * 1024 * 1024


@pytest.mark.parametrize("kind", ["cone", "fan", "parallel", "curved"])
@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_supported_panel_projection_and_pose_gradient_match_finite_segment_oracles(
    kind: str, integration: Literal["midpoint", "cell_gauss"], warp_device: str
) -> None:
    grid, intrinsics, _, poses_host, values = _case()
    shape = (1, 5) if kind == "fan" else (3, 5)
    panel = replace(intrinsics, shape=shape).view(0.37)
    if kind == "parallel":
        panel = ParallelGeometry(
            (-0.8, 18.0, -0.6),
            (0.0, 1.0, 0.0),
            30.0,
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.4, 0.5),
            shape,
        )
    elif kind == "curved":
        panel = CurvedDetectorGeometry(
            (0.0, -12.0, 0.0),
            (0.0, 18.0, -0.6),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.4, 0.5),
            shape,
        )
    batch = BatchedGeometry((panel,))
    workspace = prepare_batched_projection(
        grid,
        batch,
        ProjectionSpec(73, precision="float64", integration=integration),
        device=warp_device,
    )
    mu = wp.array(values, dtype=wp.float32, device=warp_device)
    poses = wp.array((poses_host[0].packed(),), dtype=wp.float64, ndim=2, device=warp_device)
    depth = wp.empty(batch.shape, dtype=wp.float64, device=warp_device)
    weights = [[[1.0 + (row - col) / 7 for col in range(shape[1])] for row in range(shape[0])]]
    seed = wp.array(weights, dtype=wp.float64, ndim=3, device=warp_device)
    gradient = wp.empty((1, 6), dtype=wp.float64, device=warp_device)
    project_batched_optical_depth(mu, poses, workspace=workspace, out_L=depth)
    sources, endpoints = zip(
        *(batch.segment(0, row, col) for row in range(shape[0]) for col in range(shape[1])),
        strict=True,
    )
    segments = RayBundleGeometry(sources, endpoints)
    actual = depth.numpy()
    for row in range(shape[0]):
        for col in range(shape[1]):
            expected = integrate_ray_bundle(
                values,
                grid,
                segments,
                poses_host[0],
                row * shape[1] + col,
                midpoint_samples=73 if integration == "midpoint" else None,
            )
            assert actual[0, row, col] == pytest.approx(expected, abs=2e-12)
    batched_projection_vjp(
        mu, poses, cotangent=seed, workspace=workspace, out_pose_tangent=gradient
    )
    losses = []
    for sign in (-1, 1):
        pose = compose_pose(poses_host[0], (sign * 1e-5, 0.0, 0.0, 0.0, 0.0, 0.0))
        losses.append(
            sum(
                weights[0][row][col]
                * integrate_ray_bundle(
                    values,
                    grid,
                    segments,
                    pose,
                    row * shape[1] + col,
                    midpoint_samples=73 if integration == "midpoint" else None,
                )
                for row in range(shape[0])
                for col in range(shape[1])
            )
        )
    assert gradient.numpy()[0, 0] == pytest.approx(
        (losses[1] - losses[0]) / 2e-5, rel=6e-6, abs=2e-8
    )
