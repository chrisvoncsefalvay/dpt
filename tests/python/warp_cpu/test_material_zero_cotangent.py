"""Zero material cotangents skip work without dropping other channel derivatives."""

import numpy as np
import pytest
import warp as wp

from dpt._runtime import cpu_test_context
from dpt.contracts import NumericalError
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_projection import material_projection_vjp, prepare_material_projection
from dpt.projection import ProjectionSpec
from dpt.volumes import GridSpec


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("pattern", ["mixed", "zero"])
@pytest.mark.parametrize("materials", [1, 3, 32])
def test_fused_zero_channel_guard_preserves_material_vjp(
    integration, pattern, materials, warp_device
):
    with cpu_test_context():
        grid = GridSpec((3, 4, 5), (0.7, 1.1, 0.9), (-1.4, -1.5, -1.0))
        geometry = DetectorGeometry(
            (-12.0, -0.1, 0.2),
            (12.0, -0.6, -0.5),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.14, 0.11),
            (13, 17),
        )
        host = (
            np.random.default_rng(811)
            .uniform(0.04, 0.2, (materials, grid.voxels))
            .astype(np.float32)
        )
        fields = wp.array(host, dtype=wp.float32, device=warp_device)
        seeds = np.zeros((materials, geometry.pixels), dtype=np.float64)
        if pattern == "mixed":
            # A zero channel sum is not an all-zero cotangent. Other pixels
            # have only the last channel active, or all channels inactive.
            seeds[0, ::3] = 0.25
            if materials > 1:
                seeds[1, ::3] = -0.25
            seeds[-1, 1::3] = 0.5
        cotangent = wp.array(seeds, dtype=wp.float64, device=warp_device)
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
        spec = ProjectionSpec(
            31, precision="float64", integration=integration, active_volume=True, active_pose=True
        )
        results = []
        for execution in ("serial", "fused"):
            fields_gradient = wp.full(
                (materials, grid.voxels), 7.0, dtype=wp.float64, device=warp_device
            )
            pose_gradient = wp.full(6, 3.0, dtype=wp.float64, device=warp_device)
            workspace = prepare_material_projection(
                grid,
                geometry,
                spec,
                materials=materials,
                fields=fields,
                out_paths=wp.empty(
                    (materials, geometry.pixels), dtype=wp.float64, device=warp_device
                ),
                cotangent_paths=cotangent,
                out_fields=fields_gradient,
                volume_gradient_precision="float64",
                device=warp_device,
                channel_execution=execution,
                field_domain="nonnegative",
            )
            material_projection_vjp(pose, workspace=workspace, out_pose_tangent=pose_gradient)
            first_fields, first_pose = fields_gradient.numpy().copy(), pose_gradient.numpy().copy()
            results.append((first_fields, first_pose))
            material_projection_vjp(
                pose, workspace=workspace, out_pose_tangent=pose_gradient, accumulate=True
            )
            np.testing.assert_allclose(
                fields_gradient.numpy(), 2 * first_fields, rtol=1e-10, atol=1e-14
            )
            np.testing.assert_allclose(
                pose_gradient.numpy(), 2 * first_pose, rtol=2e-12, atol=2e-13
            )
            if pattern == "zero":
                np.testing.assert_array_equal(first_fields, 0.0)
                np.testing.assert_array_equal(first_pose, 0.0)
                fields_gradient.fill_(7.0)
                pose_gradient.fill_(3.0)
                material_projection_vjp(
                    pose, workspace=workspace, out_pose_tangent=pose_gradient, accumulate=True
                )
                np.testing.assert_array_equal(fields_gradient.numpy(), 7.0)
                np.testing.assert_array_equal(pose_gradient.numpy(), 3.0)
            else:
                for material in range(materials):
                    if np.any(seeds[material]):
                        assert np.linalg.norm(first_fields[material]) > 0.01
                    else:
                        np.testing.assert_array_equal(first_fields[material], 0.0)
                assert np.linalg.norm(first_pose) > 0.01

        np.testing.assert_allclose(results[1][0], results[0][0], rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(results[1][1], results[0][1], rtol=2e-12, atol=2e-13)

        # The all-zero fast path must not bypass the kernel's ray check,
        # including when callers defer the separate host input validation.
        cotangent.zero_()
        invalid_pose = np.asarray(RigidTransform().packed(), dtype=np.float64)
        invalid_pose[9] = np.nan
        pose.assign(invalid_pose)
        material_projection_vjp(
            pose, workspace=workspace, out_pose_tangent=pose_gradient, validate=False
        )
        with pytest.raises(NumericalError):
            workspace.check_status()
