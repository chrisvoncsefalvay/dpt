"""General finite rays and shared-pose view bundles use the canonical projector."""

import importlib

# pyright: reportUnknownMemberType=false
import math
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any

import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError, NumericalError
from dpt.geometry import DetectorGeometry, RayBundleGeometry, RigidTransform, compose_pose
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.validation.projection import integrate_ray_bundle
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


@pytest.fixture(autouse=True)
def _kernel_context(warp_device: str) -> Iterator[None]:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def _bundle() -> RayBundleGeometry:
    return RayBundleGeometry(
        (
            (-8.0, 0.1, 0.2),
            (8.0, 0.1, 0.2),
            (-8.0, 9.0, 9.0),
            (0.1, 0.2, 0.3),
            (-1.0, -5.0, 0.2),
            (-0.7, 0.4, -8.0),
        ),
        (
            (8.0, 0.8, 0.7),
            (-8.0, 0.8, 0.7),
            (8.0, 9.0, 9.0),
            (0.3, 0.6, 0.7),
            (-1.0, 5.0, 0.3),
            (0.2, 0.4, 8.0),
        ),
    )


@pytest.mark.parametrize("domain", ["physical", "signed_linear"])
@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_general_ray_forward_transpose_and_pose_finite_differences(
    domain: Any, integration: Any, warp_device: str
) -> None:
    grid = GridSpec((2, 3, 4), (1.2, 0.8, 1.1), (-1.8, -0.8, -0.55))
    bundle = _bundle()
    transform = compose_pose(RigidTransform(), (0.07, -0.03, 0.02, 0.01, -0.02, 0.005))
    workspace = prepare_projection(
        grid,
        bundle,
        ProjectionSpec(
            47,
            active_volume=True,
            precision="float64",
            integration=integration,
            field_domain=domain,
        ),
        device=warp_device,
    )
    values = [
        (i + 1) / 32 * ((-1.0) ** i if domain == "signed_linear" else 1.0)
        for i in range(grid.voxels)
    ]
    seed_values = [(-1.0) ** i * (i + 1) / 8 for i in range(bundle.rays)]
    field = wp.array(values, dtype=wp.float32, device=warp_device)
    pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device)
    seeds = wp.array(seed_values, dtype=wp.float64, device=warp_device)
    output = wp.empty(bundle.rays, dtype=wp.float64, device=warp_device)
    volume_gradient = wp.empty(grid.voxels, dtype=wp.float64, device=warp_device)
    pose_gradient = wp.empty(6, dtype=wp.float64, device=warp_device)
    project_optical_depth(field, pose, workspace=workspace, out_L=output)
    reference = [
        integrate_ray_bundle(
            values,
            grid,
            bundle,
            transform,
            ray,
            midpoint_samples=47 if integration == "midpoint" else None,
            field_domain=domain,
        )
        for ray in range(bundle.rays)
    ]
    assert output.numpy().tolist() == pytest.approx(reference, rel=3e-13, abs=3e-14)
    assert output.numpy()[2] == 0.0
    projection_vjp(
        field,
        pose,
        workspace=workspace,
        cotangent=seeds,
        out_mu=volume_gradient,
        out_pose_tangent=pose_gradient,
        volume_gradient_precision="float64",
    )
    matrix_columns = [
        [
            integrate_ray_bundle(
                [float(i == voxel) for i in range(grid.voxels)],
                grid,
                bundle,
                transform,
                ray,
                midpoint_samples=47 if integration == "midpoint" else None,
            )
            for ray in range(bundle.rays)
        ]
        for voxel in range(grid.voxels)
    ]
    transpose = [
        math.fsum(a * b for a, b in zip(column, seed_values, strict=True))
        for column in matrix_columns
    ]
    assert volume_gradient.numpy().tolist() == pytest.approx(transpose, rel=3e-13, abs=3e-14)

    def objective(current: RigidTransform) -> float:
        return math.fsum(
            seed_values[ray]
            * integrate_ray_bundle(
                values,
                grid,
                bundle,
                current,
                ray,
                midpoint_samples=47 if integration == "midpoint" else None,
                field_domain=domain,
            )
            for ray in range(bundle.rays)
        )

    for axis, measured in enumerate(pose_gradient.numpy().tolist()):
        errors = []
        for step in (1e-4, 3e-5, 1e-5, 3e-6):
            delta = tuple(step if i == axis else 0.0 for i in range(6))
            finite_difference = (
                objective(compose_pose(transform, delta))
                - objective(compose_pose(transform, tuple(-x for x in delta)))
            ) / (2 * step)
            errors.append(abs(finite_difference - measured))
        assert min(errors) < 2e-7 * max(1.0, abs(measured)), (axis, measured, errors)


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_flattened_views_match_separate_views_and_sum_volume_adjoint(
    integration: Any, warp_device: str
) -> None:
    grid = GridSpec((3, 3, 3), (1.1, 0.9, 1.3), (-1.1, -0.9, -1.3))
    views = (
        DetectorGeometry(
            (-8.0, 0.13, 0.27), (8.0, -0.4, -0.3), (0, 1, 0), (0, 0, 1), (0.19, 0.21), (2, 3)
        ),
        DetectorGeometry(
            (0.23, -8.0, 0.37), (-0.3, 8.0, -0.4), (1, 0, 0), (0, 0, 1), (0.17, 0.23), (3, 2)
        ),
    )
    bundle = RayBundleGeometry.from_views(views)
    assert bundle.view_offsets == (0, 6, 12)
    assert bundle.segment(0) == (views[0].source_mm, views[0].pixel_centre(0, 0))
    assert bundle.segment(11) == (views[1].source_mm, views[1].pixel_centre(2, 1))
    spec = ProjectionSpec(
        31, active_pose=False, active_volume=True, precision="float64", integration=integration
    )
    workspace = prepare_projection(grid, bundle, spec, device=warp_device)
    assert workspace.scratch_bytes == 4 + 48 * bundle.rays
    field = wp.array(
        [(i + 1) / 64 for i in range(grid.voxels)], dtype=wp.float32, device=warp_device
    )
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    output = wp.empty(bundle.rays, dtype=wp.float64, device=warp_device)
    gradient = wp.empty(grid.voxels, dtype=wp.float64, device=warp_device)
    seeds = wp.ones(bundle.rays, dtype=wp.float64, device=warp_device)
    project_optical_depth(field, pose, workspace=workspace, out_L=output)
    projection_vjp(
        field,
        pose,
        cotangent=seeds,
        workspace=workspace,
        out_mu=gradient,
        volume_gradient_precision="float64",
    )
    expected_paths: list[float] = []
    expected_gradient = wp.zeros(grid.voxels, dtype=wp.float64, device=warp_device)
    for view in views:
        view_workspace = prepare_projection(grid, view, spec, device=warp_device)
        view_output = wp.empty(view.pixels, dtype=wp.float64, device=warp_device)
        view_seeds = wp.ones(view.pixels, dtype=wp.float64, device=warp_device)
        project_optical_depth(field, pose, workspace=view_workspace, out_L=view_output)
        expected_paths.extend(view_output.numpy().tolist())
        projection_vjp(
            field,
            pose,
            cotangent=view_seeds,
            workspace=view_workspace,
            out_mu=expected_gradient,
            volume_gradient_precision="float64",
            accumulate=True,
        )
    assert output.numpy().tolist() == pytest.approx(expected_paths, rel=2e-14, abs=2e-14)
    assert gradient.numpy().tolist() == pytest.approx(
        expected_gradient.numpy().tolist(), rel=3e-14, abs=3e-14
    )


@pytest.mark.parametrize("shape", [(3, 3, 3), (1, 1, 1)])
@pytest.mark.parametrize(
    "start,end",
    [
        ((-4.0, 0.13, 0.27), (5.0, 1.7, 1.9)),
        ((-4.0, 1.0, 1.0), (5.0, 1.0, 1.0)),
        ((5.0, 1.0, 1.0), (-4.0, 1.0, 1.0)),
        ((-4.0, -4.0, -4.0), (5.0, 5.0, 5.0)),
        ((-0.4, -0.3, -0.2), (-0.1, -0.2, -0.1)),
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ((-4.0, -3.0, 0.0), (5.0, -3.0, 0.0)),
        ((-0.5, -0.5, -4.0), (-0.5, -0.5, 5.0)),
    ],
)
def test_shared_cell_nodes_match_sorted_plane_oracle_at_ties_and_edge_slabs(
    shape: tuple[int, int, int], start: Any, end: Any, warp_device: str
) -> None:
    grid = GridSpec(shape, (1.0, 1.0, 1.0))
    bundle = RayBundleGeometry((start,), (end,))
    values = [
        0.02 + 0.013 * x + 0.007 * y + 0.003 * z + 0.009 * x * y * z
        for z in range(shape[0])
        for y in range(shape[1])
        for x in range(shape[2])
    ]
    field = wp.array(values, dtype=wp.float32, device=warp_device)
    stored_values = field.numpy().tolist()
    workspace = prepare_projection(
        grid,
        bundle,
        ProjectionSpec(1, precision="float64", integration="cell_gauss"),
        device=warp_device,
    )
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    output = wp.empty(1, dtype=wp.float64, device=warp_device)
    project_optical_depth(field, pose, workspace=workspace, out_L=output)
    expected = integrate_ray_bundle(stored_values, grid, bundle, RigidTransform(), 0)
    assert float(output.numpy()[0]) == pytest.approx(expected, rel=3e-14, abs=2e-15)


def test_projection_checkpoints_reuse_host_staging_and_explicitly_consume_status(
    warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    bundle = RayBundleGeometry(((-2.0, 0.0, 0.0),), ((2.0, 0.0, 0.0),))
    workspace = prepare_projection(grid, bundle, ProjectionSpec(1), device=warp_device)
    field = wp.ones(1, dtype=wp.float32, device=warp_device)
    pose = wp.array(
        (*RigidTransform().rotation, math.nan, 0.0, 0.0), dtype=wp.float64, device=warp_device
    )
    output = wp.empty(1, dtype=wp.float32, device=warp_device)
    project_optical_depth(field, pose, workspace=workspace, out_L=output, validate=False)
    assert workspace.host_bytes == 4

    def forbid_allocation(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("status checkpoint allocated another array")

    monkeypatch.setattr(wp, "empty", forbid_allocation)
    monkeypatch.setattr(wp, "zeros", forbid_allocation)
    for _ in range(2):
        with pytest.raises(NumericalError):
            workspace.check_status()
    with pytest.raises(NumericalError):
        workspace.consume_status()
    workspace.check_status()


@pytest.mark.parametrize(
    "sources,endpoints,offsets",
    [
        ((), (), None),
        (((0.0, 0.0, 0.0),), (), None),
        (((0.0, 0.0, 0.0),), ((0.0, 0.0, 0.0),), None),
        (((math.nan, 0.0, 0.0),), ((1.0, 0.0, 0.0),), None),
        (((-1e308, 0.0, 0.0),), ((1e308, 0.0, 0.0),), None),
        (((0.0, 0.0, 0.0),), ((1.0, 0.0, 0.0),), (0, 0, 1)),
        (((0.0, 0.0, 0.0),), ((1.0, 0.0, 0.0),), (0, 2)),
        (((0.0, 0.0, 0.0),), ((1.0, 0.0, 0.0),), (False, 1)),
    ],
)
def test_invalid_bundle_metadata_rejected(sources: Any, endpoints: Any, offsets: Any) -> None:
    with pytest.raises(ContractError):
        RayBundleGeometry(sources, endpoints, offsets)


@pytest.mark.parametrize("ray", [-1, 6, True, 0.0])
def test_bundle_index_contract(ray: Any) -> None:
    with pytest.raises(ContractError, match="ray index"):
        _bundle().segment(ray)
