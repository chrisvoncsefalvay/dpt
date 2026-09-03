"""Device pixel expansion and per-view object transforms against host oracles."""

import importlib
from typing import Any

import pytest

from dpt._runtime import load_kernels
from dpt.contracts import ContractError
from dpt.geometry import (
    BatchedGeometry,
    ConeBeamIntrinsics,
    CurvedDetectorGeometry,
    ParallelGeometry,
    RayBundleGeometry,
    RigidTransform,
)
from dpt.geometry_device import prepare_geometry
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.validation.projection import integrate_ray_bundle
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


def _batch() -> BatchedGeometry:
    cone = ConeBeamIntrinsics(100.0, 200.0, (3, 4), (0.7, 0.8)).view(0.4)
    parallel = ParallelGeometry(
        (-2.0, 100.0, -1.0),
        (0.0, 1.0, 0.0),
        200.0,
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.7, 0.8),
        (3, 4),
    )
    curved = CurvedDetectorGeometry(
        (0.0, 0.0, 0.0), (0.0, 100.0, -1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.7, 0.8), (3, 4)
    )
    return BatchedGeometry((cone, parallel, curved))


def test_device_expansion_matches_independent_each_panel(warp_device: str) -> None:
    batch = _batch()
    resident = prepare_geometry(batch, device=warp_device)
    sources, endpoints = resident.sources_mm.numpy(), resident.endpoints_mm.numpy()
    for view in range(3):
        for row in range(3):
            for column in range(4):
                ray = (view * 3 + row) * 4 + column
                source, endpoint = batch.segment(view, row, column)
                assert sources[ray] == pytest.approx(source, abs=3e-14)
                assert endpoints[ray] == pytest.approx(endpoint, abs=3e-14)
    assert resident.shape == (3, 3, 4)
    assert resident.view_offsets == (0, 12, 24, 36)
    assert resident.scratch_bytes == 36 * 48 + 3 * (18 * 8 + 4) + 4


def test_expansion_does_not_call_any_host_pixel_function(
    monkeypatch: pytest.MonkeyPatch, warp_device: str
) -> None:
    batch = _batch()

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("device expansion must not call host pixel functions")

    for cls in (type(batch.views[0]), ParallelGeometry, CurvedDetectorGeometry):
        monkeypatch.setattr(cls, "pixel_centre", forbidden)
    resident = prepare_geometry(batch, device=warp_device)
    assert resident.pixels == 36
    assert resident.endpoints_mm.numpy().shape == (36, 3)


def test_per_view_object_pose_transforms_and_reuses_output(warp_device: str) -> None:
    batch = _batch()
    resident = prepare_geometry(batch, device=warp_device)
    host_poses = (
        RigidTransform(),
        RigidTransform(translation_mm=(1.0, 2.0, 3.0)),
        RigidTransform((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), (-1.0, 0.5, 2.0)),
    )
    poses = wp.array(
        tuple(pose.packed() for pose in host_poses), dtype=wp.float64, ndim=2, device=warp_device
    )
    transformed = resident.in_object_frame(poses)
    assert resident.in_object_frame(poses, out=transformed, validate=False) is transformed
    sources, endpoints = transformed.sources_mm.numpy(), transformed.endpoints_mm.numpy()
    for view in range(3):
        for row, column in ((0, 0), (2, 3)):
            ray = (view * 3 + row) * 4 + column
            source, endpoint = batch.segment(view, row, column)
            inverse = host_poses[view].inverse()
            assert sources[ray] == pytest.approx(inverse.point(source), abs=3e-14)
            assert endpoints[ray] == pytest.approx(inverse.point(endpoint), abs=3e-14)
    with pytest.raises(ContractError, match="world-frame"):
        transformed.in_object_frame(poses)
    with pytest.raises(ContractError, match=r"FP64|wrong device or dtype"):
        resident.in_object_frame(
            wp.array(
                tuple(pose.packed() for pose in host_poses),
                dtype=wp.float32,
                ndim=2,
                device=warp_device,
            )
        )


@pytest.mark.parametrize(
    "entry,value,valid",
    [
        (0, 1.0 + 2e-11, True),
        (0, 1.0 + 6e-11, False),
        (0, -1.0, False),
        (0, float("nan"), False),
        (9, float("nan"), False),
        (11, float("inf"), False),
        (4, 1e100, False),
    ],
)
def test_batched_and_single_pose_validation_have_identical_contract(
    entry: int, value: float, valid: bool, warp_device: str
) -> None:
    resident = prepare_geometry(_batch(), device=warp_device)
    pose_values = list(RigidTransform().packed())
    pose_values[entry] = value
    single = wp.array(pose_values, dtype=wp.float64, device=warp_device)
    batch_values = (RigidTransform().packed(), tuple(pose_values), RigidTransform().packed())
    poses = wp.array(batch_values, dtype=wp.float64, ndim=2, device=warp_device)
    status = wp.zeros(1, dtype=wp.int32, device=warp_device)
    kernels = load_kernels("dpt.kernels.geometry")
    wp.launch(kernels.validate_pose, dim=1, inputs=[single], outputs=[status], device=warp_device)
    assert int(status.numpy()[0]) == (0 if valid else 1)
    if valid:
        resident.validate_poses(poses)
    else:
        with pytest.raises(ContractError, match="batched rigid pose"):
            resident.in_object_frame(poses)
    # A subsequent valid epoch must not retain the previous error bit.
    corrected = wp.array(
        tuple(RigidTransform().packed() for _ in range(3)),
        dtype=wp.float64,
        ndim=2,
        device=warp_device,
    )
    resident.validate_poses(corrected)


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_device_batch_projection_and_vjp_match_independent_segments(
    integration: Any, warp_device: str
) -> None:
    batch = _batch()
    resident = prepare_geometry(batch, device=warp_device)
    grid = GridSpec((5, 6, 7), (1.0, 1.0, 1.0), (-3.0, -2.5, -2.0))
    host_sources, host_endpoints = [], []
    for view in range(3):
        for row in range(3):
            for column in range(4):
                source, endpoint = batch.segment(view, row, column)
                host_sources.append(source)
                host_endpoints.append(endpoint)
    explicit = RayBundleGeometry(tuple(host_sources), tuple(host_endpoints), batch.view_offsets)
    spec = ProjectionSpec(
        40, active_pose=True, active_volume=True, precision="float64", integration=integration
    )
    prepared = prepare_projection(grid, resident, spec, device=warp_device)
    reference = prepare_projection(grid, explicit, spec, device=warp_device)
    assert prepared._configuration.ray_sources.ptr == resident.sources_mm.ptr
    assert prepared._configuration.ray_endpoints.ptr == resident.endpoints_mm.ptr
    values = [
        0.01 + 0.001 * (index % 7) + 0.002 * ((index // 7) % 6) + 0.003 * (index // 42)
        for index in range(grid.voxels)
    ]
    mu = wp.array(values, dtype=wp.float32, device=warp_device)
    transform = RigidTransform(translation_mm=(0.2, -0.3, 0.1))
    pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device)
    seed = wp.array(
        [(-1.0) ** ray * (ray + 1) / 36 for ray in range(36)], dtype=wp.float64, device=warp_device
    )
    outputs = [wp.empty(36, dtype=wp.float64, device=warp_device) for _ in range(2)]
    gradients = [wp.empty(grid.voxels, dtype=wp.float64, device=warp_device) for _ in range(2)]
    pose_gradients = [wp.empty(6, dtype=wp.float64, device=warp_device) for _ in range(2)]
    for index, workspace in enumerate((prepared, reference)):
        project_optical_depth(mu, pose, workspace=workspace, out_L=outputs[index])
        projection_vjp(
            mu,
            pose,
            cotangent=seed,
            workspace=workspace,
            out_mu=gradients[index],
            out_pose_tangent=pose_gradients[index],
            volume_gradient_precision="float64",
        )
    assert outputs[0].numpy() == pytest.approx(outputs[1].numpy(), abs=1e-12)
    assert gradients[0].numpy() == pytest.approx(gradients[1].numpy(), abs=1e-11)
    assert pose_gradients[0].numpy() == pytest.approx(pose_gradients[1].numpy(), abs=1e-10)
    expected = [
        integrate_ray_bundle(
            tuple(float(value) for value in mu.numpy()),
            grid,
            explicit,
            transform,
            ray,
            midpoint_samples=40 if integration == "midpoint" else None,
        )
        for ray in range(36)
    ]
    assert outputs[0].numpy() == pytest.approx(expected, abs=1e-11)
    lhs = sum(
        float(value) * float(gradient)
        for value, gradient in zip(mu.numpy(), gradients[0].numpy(), strict=True)
    )
    rhs = sum(
        float(value) * float(weight)
        for value, weight in zip(outputs[0].numpy(), seed.numpy(), strict=True)
    )
    assert lhs == pytest.approx(rhs, abs=1e-11)


def test_multiview_object_poses_project_against_individual_panel_oracles(warp_device: str) -> None:
    batch = _batch()
    resident = prepare_geometry(batch, device=warp_device)
    transforms = (
        RigidTransform(translation_mm=(0.2, 0.3, 0.1)),
        RigidTransform(translation_mm=(-0.1, 0.2, -0.3)),
        RigidTransform((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.3, 0.1, 0.2)),
    )
    poses = wp.array(
        tuple(pose.packed() for pose in transforms), dtype=wp.float64, ndim=2, device=warp_device
    )
    object_rays = resident.in_object_frame(poses)
    grid = GridSpec((5, 6, 7), (1.0, 1.0, 1.0), (-3.0, -2.5, -2.0))
    mu = wp.array(
        [0.02 + 0.001 * (i % 7) for i in range(grid.voxels)], dtype=wp.float32, device=warp_device
    )
    identity = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    workspace = prepare_projection(
        grid,
        object_rays,
        ProjectionSpec(40, active_pose=False, precision="float64"),
        device=warp_device,
    )
    output = wp.empty(36, dtype=wp.float64, device=warp_device)
    project_optical_depth(mu, identity, workspace=workspace, out_L=output)
    expected = []
    for view, transform in enumerate(transforms):
        sources, endpoints = [], []
        for row in range(3):
            for column in range(4):
                source, endpoint = batch.segment(view, row, column)
                sources.append(source)
                endpoints.append(endpoint)
        rays = RayBundleGeometry(tuple(sources), tuple(endpoints))
        for ray in range(12):
            expected.append(
                integrate_ray_bundle(
                    tuple(float(value) for value in mu.numpy()),
                    grid,
                    rays,
                    transform,
                    ray,
                    midpoint_samples=40,
                )
            )
    assert output.numpy() == pytest.approx(expected, abs=1e-11)
