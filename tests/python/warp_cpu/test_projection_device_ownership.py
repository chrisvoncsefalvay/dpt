"""Explicit workspace devices survive a different ambient Warp device."""

import importlib
import math
from typing import Any, Literal

import pytest

from dpt.aperture import (
    ApertureGeometry,
    aperture_projection_vjp,
    prepare_aperture_projection,
    project_aperture_counts,
)
from dpt.contracts import TrialDomainError
from dpt.geometry import RayBundleGeometry, RigidTransform
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("aperture", [False, True])
def test_projection_workspace_binds_every_launch_to_its_device(
    integration: Literal["midpoint", "cell_gauss"],
    aperture: bool,
    warp_device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CUDA builds normally select CUDA as Warp's ambient default. The test
    # deliberately avoids the legacy CPU test context, which hid this bug.
    ambient = wp.get_device("cuda:0" if wp.is_cuda_available() else "cpu")
    owner = wp.get_device(warp_device)
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((0.0, 0.0, -2.0),), ((0.0, 0.0, 2.0),))
    spec = ProjectionSpec(
        17, precision="float64", integration=integration, active_pose=False, active_volume=True
    )
    field = wp.array([0.25], dtype=wp.float32, device=owner)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=owner)
    seed = wp.array([2.0], dtype=wp.float64, device=owner)
    result = wp.empty(1, dtype=wp.float64, device=owner)
    gradient = wp.empty(1, dtype=wp.float64, device=owner)
    launched: list[Any] = []
    launch = wp.launch

    def owned_launch(*args: Any, **kwargs: Any) -> Any:
        selected = kwargs.get("device")
        assert selected is not None, "operator relied on Warp's ambient launch device"
        assert wp.get_device(selected) == owner
        launched.append(selected)
        return launch(*args, **kwargs)

    with wp.ScopedDevice(ambient):
        if aperture:
            geometry = ApertureGeometry(rays, (0, 1), (1.0,))
            workspace = prepare_aperture_projection(
                grid,
                geometry,
                spec,
                depths=wp.empty(1, dtype=wp.float64, device=owner),
                depth_cotangents=wp.empty(1, dtype=wp.float64, device=owner),
                device=warp_device,
            )
            beam = wp.array([100.0], dtype=wp.float64, device=owner)
            with monkeypatch.context() as patch:
                patch.setattr(wp, "launch", owned_launch)
                workspace.projection.validate_inputs(field, pose, cotangent=seed)
                project_aperture_counts(field, pose, beam, workspace=workspace, out_counts=result)
                aperture_projection_vjp(
                    field,
                    pose,
                    beam,
                    cotangent=seed,
                    workspace=workspace,
                    out_mu=gradient,
                    volume_gradient_precision="float64",
                )
            expected = 100.0 * math.exp(-0.25)
            assert float(result.numpy()[0]) == pytest.approx(expected, rel=2e-14)
            assert float(gradient.numpy()[0]) == pytest.approx(-2.0 * expected, rel=2e-14)
        else:
            projection = prepare_projection(grid, rays, spec, device=warp_device)
            with monkeypatch.context() as patch:
                patch.setattr(wp, "launch", owned_launch)
                projection.validate_inputs(field, pose, cotangent=seed)
                project_optical_depth(field, pose, workspace=projection, out_L=result)
                projection_vjp(
                    field,
                    pose,
                    cotangent=seed,
                    workspace=projection,
                    out_mu=gradient,
                    volume_gradient_precision="float64",
                )
            assert float(result.numpy()[0]) == pytest.approx(0.25, abs=2e-15)
            assert float(gradient.numpy()[0]) == pytest.approx(2.0, abs=2e-14)
    assert len(launched) >= 7


def test_cpu_projection_validation_detects_failure_under_ambient_device() -> None:
    ambient = wp.get_device("cuda:0" if wp.is_cuda_available() else "cpu")
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((0.0, 0.0, -2.0),), ((0.0, 0.0, 2.0),))
    with wp.ScopedDevice(ambient):
        workspace = prepare_projection(grid, rays, ProjectionSpec(17), device="cpu")
        field = wp.array([-1.0], dtype=wp.float32, device="cpu")
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cpu")
        result = wp.full(1, 19.0, dtype=wp.float32, device="cpu")
        with pytest.raises(TrialDomainError):
            project_optical_depth(field, pose, workspace=workspace, out_L=result)
        assert float(result.numpy()[0]) == 19.0
