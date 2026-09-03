"""Conservative culling preserves canonical midpoint coefficients and failures."""

# Warp test-kernel annotations are executable DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportPrivateUsage=false

import importlib
import math
from types import SimpleNamespace
from typing import Any, Literal, cast

import numpy as np
import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.geometry import DetectorGeometry, RayBundleGeometry, RigidTransform, compose_pose
from dpt.kernels.projection import Configuration, finite_interval, ray_for_pixel
from dpt.kernels.projection_adjoint import GatherRay, pixel_bounds, sample_bounds, voxel_support
from dpt.projection import (
    PreparedProjectionAdjoint,
    ProjectionSpec,
    prepare_projection,
    prepare_projection_adjoint,
    project_optical_depth,
    projection_adjoint_gather,
    projection_vjp,
)
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


@wp.kernel
def _canonical_nodes(
    pose: wp.array(dtype=wp.float64),
    config: Configuration,  # pyright: ignore[reportGeneralTypeIssues]
    points: wp.array(dtype=wp.vec3d),
    hit: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    pixel, sample = index // config.samples, index % config.samples
    ray = ray_for_pixel(config, pose, pixel)
    interval = finite_interval(ray.origin, ray.direction, config.shape)
    span = interval.upper - interval.lower
    step = span / wp.float64(config.samples)
    position = interval.lower + (wp.float64(sample) + wp.float64(0.5)) * step
    points[index] = ray.origin + position * ray.direction
    hit[index] = int(span > wp.float64(0.0))


@wp.kernel
def _captured_nodes(
    config: Configuration,  # pyright: ignore[reportGeneralTypeIssues]
    rays: wp.array(dtype=GatherRay),
    points: wp.array(dtype=wp.vec3d),
):
    index = wp.tid()
    pixel, sample = index // config.samples, index % config.samples
    ray = rays[pixel]
    position = ray.lower + (wp.float64(sample) + wp.float64(0.5)) * ray.step
    points[index] = ray.origin + position * ray.direction


@wp.kernel
def _candidate_bounds(
    config: Configuration,  # pyright: ignore[reportGeneralTypeIssues]
    rays: wp.array(dtype=GatherRay),
    coefficients: wp.array(dtype=wp.vec3d),
    metrics: wp.array(dtype=wp.float64),
    cone: bool,
    included: wp.array(dtype=wp.int32),
    nodes: wp.array(dtype=wp.vec2i),
):
    index = wp.tid()
    voxel, pixel = index // config.pixels, index % config.pixels
    support = voxel_support(voxel, config.shape)
    included[index] = 1
    if cone:
        bounds = pixel_bounds(support, config, coefficients, metrics)
        row, column = pixel // config.width, pixel % config.width
        included[index] = int(
            row >= bounds.first_row
            and row < bounds.last_row
            and column >= bounds.first_column
            and column < bounds.last_column
        )
    nodes[index] = sample_bounds(rays[pixel], support, config.samples)


def _scene(
    kind: str, shape: tuple[int, int, int] = (2, 3, 4)
) -> tuple[GridSpec, DetectorGeometry | RayBundleGeometry, RigidTransform]:
    transform = RigidTransform()
    grid = GridSpec(shape, (1.2, 0.8, 1.1), (-1.8, -0.8, -0.55))
    source, origin = (-8.0, 0.17, 0.23), (8.0, -0.63, -0.57)
    u, v = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    spacing, panel_shape = (0.47, 0.39), (6, 13)
    if kind in ("tilted", "large_tilted"):
        angle = 0.13
        u = (math.sin(angle), math.cos(angle), 0.0)
        orientation = (
            math.cos(0.21),
            -math.sin(0.21),
            0.0,
            math.sin(0.21),
            math.cos(0.21),
            0.0,
            0.0,
            0.0,
            1.0,
        )
        grid = GridSpec(shape, grid.spacing_mm, grid.origin_mm, orientation)
        transform = compose_pose(transform, (0.07, -0.03, 0.02, 0.03, -0.02, 0.01))
        if kind == "large_tilted":
            panel_shape, spacing = (23, 71), (0.15, 0.21)
    elif kind == "inside":
        source = (0.0, 0.1, 0.2)
        panel_shape = (3, 5)
    elif kind == "negative_depth":
        source, origin = (8.0, 0.17, 0.23), (-8.0, -0.63, -0.57)
    elif kind == "underflow_plane":
        spacing = (1e-200, 1e-200)
        panel_shape = (3, 5)
    elif kind == "tolerated_rotation":
        v = (0.0, 2e-11, 1.0)
        transform = RigidTransform((1.0 + 2e-11, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    elif kind == "bundle":
        return (
            grid,
            RayBundleGeometry(
                ((-8.0, 0.0, 0.0), (8.0, 0.0, 0.0), (-8.0, 20.0, 0.0), (0.0, 0.0, 0.0)),
                ((8.0, 0.4, 0.5), (-8.0, 0.4, 0.5), (8.0, 20.0, 0.0), (0.8, 0.4, 0.5)),
            ),
            transform,
        )
    return grid, DetectorGeometry(source, origin, u, v, spacing, panel_shape), transform


def _workspace(
    kind: str, device: str, shape: tuple[int, int, int] = (2, 3, 4)
) -> tuple[Any, Any, PreparedProjectionAdjoint]:
    grid, geometry, transform = _scene(kind, shape)
    workspace = prepare_projection(
        grid,
        geometry,
        ProjectionSpec(
            37,
            active_pose=False,
            active_volume=True,
            precision="float64",
            field_domain="signed_linear",
        ),
        device=device,
    )
    pose = wp.array(transform.packed(), dtype=wp.float64, device=device)
    return (
        workspace,
        pose,
        prepare_projection_adjoint(
            pose,
            workspace=workspace,
            max_rays_per_voxel=2048 if kind == "large_tilted" else 256,
        ),
    )


@pytest.mark.parametrize(
    "kind",
    [
        "tilted",
        "large_tilted",
        "inside",
        "negative_depth",
        "underflow_plane",
        "tolerated_rotation",
        "bundle",
    ],
)
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_culled_gather_is_bitwise_equal_to_all_ray_order_and_atomic_vjp(
    kind: str, precision: Literal["float32", "float64"], warp_device: str
) -> None:
    workspace, pose, plan = _workspace(kind, warp_device)
    rng = np.random.default_rng(754103)
    field = wp.array(rng.normal(size=workspace.grid.voxels).astype(np.float32), device=warp_device)
    seed_values = rng.normal(size=workspace.geometry.pixels)
    seed_values[::4] = 0.0
    seeds = wp.array(seed_values, dtype=wp.float64, device=warp_device)
    output = wp.empty(workspace.grid.voxels, dtype=getattr(wp, precision), device=warp_device)
    baseline = wp.empty_like(output)
    # A wide panel introduces thousands of FP32 atomic additions per voxel.
    # Use FP64 scatter to test its coefficients independently of that rounding
    # order; deterministic gather remains byte-identical to the all-ray sum.
    scatter_precision: Literal["float32", "float64"] = (
        "float64" if kind == "large_tilted" else precision
    )
    scatter = wp.empty(
        workspace.grid.voxels, dtype=getattr(wp, scatter_precision), device=warp_device
    )
    projection_adjoint_gather(
        pose,
        cotangent=seeds,
        workspace=workspace,
        out_mu=baseline,
        volume_gradient_precision=precision,
    )
    projection_adjoint_gather(
        plan,
        cotangent=seeds,
        workspace=workspace,
        out_mu=output,
        volume_gradient_precision=precision,
    )
    assert baseline.numpy().tobytes() == output.numpy().tobytes()
    projection_vjp(
        field,
        pose,
        cotangent=seeds,
        workspace=workspace,
        out_mu=scatter,
        volume_gradient_precision=scatter_precision,
    )
    np.testing.assert_allclose(
        output.numpy(),
        scatter.numpy(),
        rtol=3e-6 if precision == "float32" else 2e-12,
        atol=2e-7 if precision == "float32" else 2e-12,
    )
    projection_adjoint_gather(
        pose,
        cotangent=seeds,
        workspace=workspace,
        out_mu=baseline,
        volume_gradient_precision=precision,
        accumulate=True,
    )
    projection_adjoint_gather(
        plan,
        cotangent=seeds,
        workspace=workspace,
        out_mu=output,
        volume_gradient_precision=precision,
        accumulate=True,
    )
    assert baseline.numpy().tobytes() == output.numpy().tobytes()
    assert (
        plan.quadrature_nodes
        < workspace.grid.voxels * workspace.geometry.pixels * workspace.spec.samples_per_ray
    )
    if kind in ("inside", "underflow_plane"):
        assert plan.fallback_voxels > 0
    if kind == "bundle":
        assert plan.candidate_rays == workspace.grid.voxels * workspace.geometry.pixels


@pytest.mark.parametrize(
    "shape", [(1, 2, 3), (1, 3, 2), (2, 1, 3), (3, 1, 2), (2, 3, 1), (3, 2, 1)]
)
def test_singleton_clamped_basis_is_conservatively_included(
    shape: tuple[int, int, int], warp_device: str
) -> None:
    workspace, pose, plan = _workspace("tilted", warp_device, shape)
    seed = wp.array(
        [(-1.0) ** i for i in range(workspace.geometry.pixels)],
        dtype=wp.float64,
        device=warp_device,
    )
    first = wp.empty(workspace.grid.voxels, dtype=wp.float64, device=warp_device)
    second = wp.empty_like(first)
    projection_adjoint_gather(pose, cotangent=seed, workspace=workspace, out_mu=first)
    projection_adjoint_gather(plan, cotangent=seed, workspace=workspace, out_mu=second)
    assert first.numpy().tobytes() == second.numpy().tobytes()


@pytest.mark.parametrize(
    "kind", ["tilted", "inside", "negative_depth", "underflow_plane", "bundle"]
)
def test_every_canonical_corner_contributor_survives_detector_and_sample_bounds(
    kind: str, warp_device: str
) -> None:
    workspace, pose, plan = _workspace(kind, warp_device)
    pixels, samples = workspace.geometry.pixels, workspace.spec.samples_per_ray
    points = wp.empty(pixels * samples, dtype=wp.vec3d, device=warp_device)
    captured = wp.empty_like(points)
    hits = wp.empty(pixels * samples, dtype=wp.int32, device=warp_device)
    included = wp.empty(workspace.grid.voxels * pixels, dtype=wp.int32, device=warp_device)
    node_bounds = wp.empty(workspace.grid.voxels * pixels, dtype=wp.vec2i, device=warp_device)
    wp.launch(
        _canonical_nodes,
        dim=pixels * samples,
        inputs=[pose, workspace._configuration],
        outputs=[points, hits],
        device=warp_device,
    )
    wp.launch(
        _captured_nodes,
        dim=pixels * samples,
        inputs=[workspace._configuration, plan._rays],
        outputs=[captured],
        device=warp_device,
    )
    hit_values = hits.numpy().astype(bool)
    assert points.numpy()[hit_values].tobytes() == captured.numpy()[hit_values].tobytes()
    wp.launch(
        _candidate_bounds,
        dim=workspace.grid.voxels * pixels,
        inputs=[
            workspace._configuration,
            plan._rays,
            plan._coefficients,
            plan._metrics,
            plan._cone,
        ],
        outputs=[included, node_bounds],
        device=warp_device,
    )
    mask, ranges = included.numpy(), node_bounds.numpy()
    shape_xyz = tuple(reversed(workspace.grid.shape))
    checked = 0
    for index, point in enumerate(points.numpy()):
        if not hit_values[index] or not all(
            -0.5 <= point[a] <= shape_xyz[a] - 0.5 for a in range(3)
        ):
            continue
        low = [
            min(
                math.floor(min(max(float(point[a]), 0.0), shape_xyz[a] - 1)),
                max(shape_xyz[a] - 2, 0),
            )
            for a in range(3)
        ]
        pixel, sample = divmod(index, samples)
        for corner in range(8):
            coordinate = [min(low[a] + ((corner >> a) & 1), shape_xyz[a] - 1) for a in range(3)]
            voxel = (coordinate[2] * shape_xyz[1] + coordinate[1]) * shape_xyz[0] + coordinate[0]
            pair = voxel * pixels + pixel
            assert mask[pair] == 1, (kind, voxel, pixel, sample, point)
            assert ranges[pair, 0] <= sample < ranges[pair, 1], (kind, voxel, pixel, sample, point)
            checked += 1
    assert checked > 0


def test_culled_adjoint_matches_independent_scalar_matrix_and_signed_inner_products(
    warp_device: str,
) -> None:
    workspace, pose, plan = _workspace("tilted", warp_device)
    grid, geometry, transform = _scene("tilted")
    assert isinstance(geometry, DetectorGeometry)
    matrix = np.empty((geometry.pixels, grid.voxels), dtype=np.float64)
    for voxel in range(grid.voxels):
        basis = [float(i == voxel) for i in range(grid.voxels)]
        for pixel in range(geometry.pixels):
            row, column = divmod(pixel, geometry.shape[1])
            matrix[pixel, voxel] = integrate_sampled_field(
                basis, grid, geometry, transform, row, column, midpoint_samples=37
            )
    rng = np.random.default_rng(943201)
    image = wp.empty(geometry.pixels, dtype=wp.float64, device=warp_device)
    gradient = wp.empty(grid.voxels, dtype=wp.float64, device=warp_device)
    for _ in range(20):
        values = rng.normal(size=grid.voxels).astype(np.float32)
        seeds = rng.normal(size=geometry.pixels)
        field = wp.array(values, dtype=wp.float32, device=warp_device)
        seed = wp.array(seeds, dtype=wp.float64, device=warp_device)
        project_optical_depth(field, pose, workspace=workspace, out_L=image)
        projection_adjoint_gather(plan, cotangent=seed, workspace=workspace, out_mu=gradient)
        np.testing.assert_allclose(image.numpy(), matrix @ values, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(gradient.numpy(), matrix.T @ seeds, rtol=2e-12, atol=2e-12)
        assert abs(float(image.numpy() @ seeds) - float(values @ gradient.numpy())) < 2e-12 * max(
            1.0, abs(float(image.numpy() @ seeds))
        )


def test_owned_adjoint_reuse_ignores_original_pose_mutation_and_checks_owner_seed_and_budget(
    warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, pose, plan = _workspace("tilted", warp_device)
    seed = wp.ones(workspace.geometry.pixels, dtype=wp.float64, device=warp_device)
    output = wp.empty(workspace.grid.voxels, dtype=wp.float64, device=warp_device)
    projection_adjoint_gather(plan, cotangent=seed, workspace=workspace, out_mu=output)
    original = output.numpy().copy()
    pose.fill_(float("nan"))
    # Scalar configuration is captured by value too; the plan does not reread
    # the original struct while evaluating its fixed operator.
    workspace._configuration.samples = 1
    workspace._configuration.shape[0] = 1
    ray_preparer = importlib.import_module("dpt.kernels.projection_adjoint")

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("reusing a plan must not prepare rays or count a volume again")

    monkeypatch.setattr(ray_preparer, "get_prepare_rays", forbidden)
    monkeypatch.setattr(ray_preparer, "get_count_gather_work", forbidden)
    projection_adjoint_gather(plan, cotangent=seed, workspace=workspace, out_mu=output)
    assert output.numpy().tobytes() == original.tobytes()
    with pytest.raises(ContractError, match="max_work"):
        projection_adjoint_gather(
            plan, cotangent=seed, workspace=workspace, out_mu=output, max_work=plan.work_bound - 1
        )
    assert output.numpy().tobytes() == original.tobytes()
    seed.fill_(float("nan"))
    with pytest.raises(NumericalError):
        projection_adjoint_gather(plan, cotangent=seed, workspace=workspace, out_mu=output)
    assert output.numpy().tobytes() == original.tobytes()
    with pytest.raises(NumericalError):
        workspace.consume_status()


def test_preparation_limits_and_current_diagnostics_prevent_plan_creation(warp_device: str) -> None:
    workspace, pose, plan = _workspace("inside", warp_device)
    for kwargs, message in [
        ({"max_bytes": 1}, "max_bytes"),
        ({"max_prepare_voxels": 1}, "max_prepare_voxels"),
        ({"max_work": 1}, "max_work"),
        ({"max_rays_per_voxel": 1}, "max_rays_per_voxel"),
    ]:
        with pytest.raises(ContractError, match=message):
            prepare_projection_adjoint(pose, workspace=workspace, **cast(Any, kwargs))
    workspace._status.fill_(2)
    with pytest.raises(NumericalError):
        prepare_projection_adjoint(pose, workspace=workspace)
    with pytest.raises(NumericalError):
        workspace.consume_status()
    workspace.spec = ProjectionSpec(37, integration="cell_gauss")
    with pytest.raises(ContractError, match="strict float64 midpoint"):
        prepare_projection_adjoint(pose, workspace=workspace)
    with pytest.raises(ContractError, match="original workspace configuration"):
        projection_adjoint_gather(
            plan,
            cotangent=wp.ones(workspace.geometry.pixels, dtype=wp.float64, device=warp_device),
            workspace=workspace,
            out_mu=wp.empty(workspace.grid.voxels, dtype=wp.float64, device=warp_device),
        )


def test_invalid_captured_ray_is_reported_before_zero_seed_can_hide_it(warp_device: str) -> None:
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    geometry = RayBundleGeometry(((-2e-200, 0.0, 0.0),), ((2e-200, 0.0, 0.0),))
    workspace = prepare_projection(
        grid, geometry, ProjectionSpec(3, precision="float64"), device=warp_device
    )
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    with pytest.raises(NumericalError):
        prepare_projection_adjoint(pose, workspace=workspace)


def test_zero_weight_clamped_corner_preserves_overflow_diagnostics(warp_device: str) -> None:
    grid = GridSpec((1, 1, 2), (10.0, 1.0, 1.0))
    geometry = RayBundleGeometry(((-100.0, 0.0, 0.0),), ((0.0, 0.0, 0.0),))
    workspace = prepare_projection(
        grid, geometry, ProjectionSpec(1, precision="float64"), device=warp_device
    )
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    plan = prepare_projection_adjoint(pose, workspace=workspace)
    seed = wp.array([1e308], dtype=wp.float64, device=warp_device)
    first = wp.empty(grid.voxels, dtype=wp.float64, device=warp_device)
    second = wp.empty_like(first)
    for operator, output in ((pose, first), (plan, second)):
        with pytest.raises(NumericalError):
            projection_adjoint_gather(operator, cotangent=seed, workspace=workspace, out_mu=output)
        with pytest.raises(NumericalError):
            workspace.consume_status()
    np.testing.assert_array_equal(np.isnan(first.numpy()), [True, True])
    np.testing.assert_array_equal(np.isnan(second.numpy()), [True, True])


def test_combined_ray_and_node_counter_overflow_is_rejected_before_device_allocation() -> None:
    # Node and ray counts each fit int64, but their sum does not. There is no
    # context: any device allocation or launch before rejection would fail.
    workspace: Any = SimpleNamespace(
        _recorded_tape=None,
        spec=ProjectionSpec(2, precision="float64"),
        grid=SimpleNamespace(voxels=2**31 - 1),
        geometry=SimpleNamespace(pixels=2**31 - 1),
    )
    with pytest.raises(ContractError, match="signed 64-bit range"):
        prepare_projection_adjoint(
            None,
            workspace=workspace,
            max_prepare_voxels=2**31 - 1,
            max_work=2**63 - 1,
        )


def test_prepared_adjoint_rejects_other_workspace_before_output_write(warp_device: str) -> None:
    _, _, plan = _workspace("tilted", warp_device)
    other, _, _ = _workspace("tilted", warp_device)
    seed = wp.ones(other.geometry.pixels, dtype=wp.float64, device=warp_device)
    output = wp.array(np.full(other.grid.voxels, 7.0), dtype=wp.float64, device=warp_device)
    with pytest.raises(ContractError, match="original workspace configuration"):
        projection_adjoint_gather(plan, cotangent=seed, workspace=other, out_mu=output)
    np.testing.assert_array_equal(output.numpy(), 7.0)


def test_prepared_adjoint_rejects_output_alias_of_owned_culling_metadata(warp_device: str) -> None:
    workspace, _, plan = _workspace("tilted", warp_device, (1, 2, 3))
    seed = wp.ones(workspace.geometry.pixels, dtype=wp.float64, device=warp_device)
    original = plan._metrics.numpy().copy()
    with pytest.raises(ContractError, match="overlaps prepared metrics"):
        projection_adjoint_gather(plan, cotangent=seed, workspace=workspace, out_mu=plan._metrics)
    assert plan._metrics.numpy().tobytes() == original.tobytes()
