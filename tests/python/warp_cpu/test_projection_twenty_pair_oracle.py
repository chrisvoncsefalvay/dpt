"""Twenty signed transpose pairs against an independent explicit projector matrix."""

import importlib
from typing import Any, Literal

import pytest

from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

np: Any = importlib.import_module("numpy")
wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_twenty_independent_volume_transpose_pairs(
    integration: Literal["midpoint", "cell_gauss"], warp_device: str
) -> None:
    grid = GridSpec((3, 4, 5), (1.2, 0.8, 1.1), (-2.0, -1.0, -1.0))
    detector = DetectorGeometry(
        (-12.0, 0.31, 0.43), (12.0, -0.6, -0.6), (0, 1, 0), (0, 0, 1), (0.071, 0.11), (3, 5)
    )
    transform = compose_pose(RigidTransform(), (0.13, -0.08, 0.07, 0.02, -0.03, 0.01))
    matrix = np.empty((detector.pixels, grid.voxels), dtype=np.float64)
    for voxel in range(grid.voxels):
        basis = [float(i == voxel) for i in range(grid.voxels)]
        for pixel in range(detector.pixels):
            row, column = divmod(pixel, detector.shape[1])
            matrix[pixel, voxel] = integrate_sampled_field(
                basis,
                grid,
                detector,
                transform,
                row,
                column,
                midpoint_samples=47 if integration == "midpoint" else None,
            )
    workspace = prepare_projection(
        grid,
        detector,
        ProjectionSpec(
            47,
            precision="float64",
            integration=integration,
            active_pose=False,
            active_volume=True,
            field_domain="signed_linear",
        ),
        device=warp_device,
    )
    pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device)
    output = wp.empty(detector.pixels, dtype=wp.float64, device=warp_device)
    gradient = wp.empty(grid.voxels, dtype=wp.float64, device=warp_device)
    rng = np.random.default_rng(41072)
    for pair in range(20):
        values = np.ascontiguousarray(rng.normal(size=grid.voxels), dtype=np.float32)
        seeds = np.ascontiguousarray(rng.normal(size=detector.pixels), dtype=np.float64)
        field = wp.array(values, dtype=wp.float32, device=warp_device)
        seed = wp.array(seeds, dtype=wp.float64, device=warp_device)
        project_optical_depth(field, pose, out_L=output, workspace=workspace)
        projection_vjp(
            field,
            pose,
            cotangent=seed,
            out_mu=gradient,
            workspace=workspace,
            volume_gradient_precision="float64",
        )
        measured_forward = output.numpy()
        measured_gradient = gradient.numpy()
        np.testing.assert_allclose(
            measured_forward,
            matrix @ values.astype(np.float64),
            rtol=2e-12,
            atol=2e-12,
            err_msg=f"forward pair {pair}",
        )
        np.testing.assert_allclose(
            measured_gradient,
            matrix.T @ seeds,
            rtol=2e-12,
            atol=2e-12,
            err_msg=f"adjoint pair {pair}",
        )
        left = float(measured_forward @ seeds)
        right = float(values.astype(np.float64) @ measured_gradient)
        assert abs(left - right) < 2e-12 * max(1, abs(left), abs(right)), (pair, left, right)
