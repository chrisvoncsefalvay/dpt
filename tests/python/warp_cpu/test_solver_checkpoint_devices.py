"""Resume accepted resident solver states on explicit CPU and real CUDA devices."""

import importlib
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError
from dpt.material_reconstruction import MaterialReconstruction, MaterialReconstructionCheckpoint
from dpt.recovery import PoseChart, PrimaryPoseEvaluator, PrimaryPoseProblem, recover_pose
from dpt.registration import RecoveryPolicy
from helpers import material_inputs as materials
from helpers import pose_inputs as poses

wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize(
    "step_selection,acceleration,window",
    [("bb", "none", 1), ("geometric", "inertial", 1), ("bb", "none", 4)],
)
def test_device_npz_material_resume_preserves_accepted_trajectory_and_voxel_scale(
    tmp_path, warp_device, step_selection, acceleration, window
):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        inputs = materials._inputs("float64")
        inputs["device"] = warp_device
        selected = replace(
            inputs["settings"],
            iterations=5,
            step_selection=step_selection,
            acceleration=acceleration,
            nonmonotone_window=window,
        )
        scale = np.linspace(0.75, 1.5, inputs["grid"].voxels, dtype=np.float32).reshape(
            inputs["grid"].shape
        )
        full = MaterialReconstruction(**{**inputs, "settings": selected})
        full.set_voxel_metric_scale(scale, curvature_normaliser=3.5)
        expected = full.solve()
        first = MaterialReconstruction(**{**inputs, "settings": replace(selected, iterations=2)})
        first.set_voxel_metric_scale(scale, curvature_normaliser=3.5)
        first.solve(objective_id="fixed-count-basis-and-view-geometry-v1")
        saved = first.checkpoint()
        assert saved.state["accepted"] == 2
        path = tmp_path / "device-material.npz"
        saved.save(path)
        checkpoint = MaterialReconstructionCheckpoint.read(path)
        resumed = MaterialReconstruction(**{**inputs, "settings": selected})
        actual = resumed.solve(checkpoint=checkpoint, objective_id=checkpoint.objective_id)
        # CUDA scatter additions need not use the same atomic summation order.
        # These bounds allow a few ppm material-fraction error, not bitwise GPU identity.
        np.testing.assert_allclose(
            resumed.fractions_numpy(), full.fractions_numpy(), atol=5e-7, rtol=2e-6
        )
        np.testing.assert_allclose(
            actual["final_objective"], expected["final_objective"], rtol=2e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            [row["objective"] for row in actual["history"]],
            [row["objective"] for row in expected["history"]],
            rtol=2e-6,
            atol=1e-8,
        )
        assert actual["accepted_steps"] == expected["accepted_steps"] == 5
        assert actual["evaluations"] == expected["evaluations"] + 1
        assert actual["voxel_curvature_initial_step"] == expected["voxel_curvature_initial_step"]
        np.testing.assert_array_equal(resumed.voxel_metric_scale.numpy(), scale.reshape(-1))
        with pytest.raises(ContractError, match="identity"):
            MaterialReconstruction(**{**inputs, "settings": selected}).solve(
                checkpoint=checkpoint, objective_id="different-observation-or-basis"
            )
        changed = tuple(
            replace(view, counts=np.ascontiguousarray(view.counts + 1)) for view in inputs["views"]
        )
        with pytest.raises(ContractError, match="objective"):
            MaterialReconstruction(**{**inputs, "settings": selected, "views": changed}).solve(
                checkpoint=checkpoint, objective_id=checkpoint.objective_id
            )


def test_device_primary_recovery_checkpoint_resumes_and_rejects_changed_chart(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        grid, values = poses.phantom()
        geometry = poses.geometries()[0]
        problem = PrimaryPoseProblem(
            grid,
            geometry,
            wp.array(values, dtype=wp.float32, device=warp_device),
            wp.zeros(geometry.pixels, dtype=wp.float32, device=warp_device),
            open_beam=100.0,
            samples_per_ray=96,
        )
        chart = PoseChart(scales=(0.1, 0.1, 0.1, 0.01, 0.01, 0.01))
        synthesis = PrimaryPoseEvaluator(problem, chart, device=warp_device)
        synthesis((0.0,) * 6)
        problem.observation.assign(synthesis.prediction.numpy().astype(np.float32))
        initial = (0.25, -0.18, 0.13, 0.3, -0.4, 0.2)
        selected = RecoveryPolicy(
            max_iterations=5,
            max_evaluations=200,
            gradient_tolerance=1e-10,
            relative_loss_tolerance=0,
            step_tolerance=0,
        )
        full = recover_pose(
            PrimaryPoseEvaluator(problem, chart, device=warp_device),
            initial=initial,
            policy=selected,
        )
        first = recover_pose(
            PrimaryPoseEvaluator(problem, chart, device=warp_device),
            initial=initial,
            policy=replace(selected, max_iterations=2),
            objective_id="immutable-primary-chart-v1",
        )
        checkpoint = first.optimisation.checkpoint
        assert checkpoint is not None and checkpoint.curvature_pairs
        resumed = recover_pose(
            PrimaryPoseEvaluator(problem, chart, device=warp_device),
            policy=selected,
            checkpoint=checkpoint,
            objective_id=checkpoint.objective_id,
        )
        np.testing.assert_allclose(
            resumed.optimisation.parameters, full.optimisation.parameters, atol=1e-9, rtol=1e-7
        )
        np.testing.assert_allclose(
            resumed.optimisation.evaluation.loss,
            full.optimisation.evaluation.loss,
            atol=1e-10,
            rtol=1e-6,
        )
        assert resumed.optimisation.evaluations == full.optimisation.evaluations + 1
        with pytest.raises(ContractError, match="identity"):
            recover_pose(
                PrimaryPoseEvaluator(problem, chart, device=warp_device),
                policy=selected,
                checkpoint=checkpoint,
                objective_id="different-chart-id",
            )
        shifted_chart = replace(chart, scales=(0.2, 0.1, 0.1, 0.01, 0.01, 0.01))
        with pytest.raises(ContractError, match="oracle"):
            recover_pose(
                PrimaryPoseEvaluator(problem, shifted_chart, device=warp_device),
                policy=selected,
                checkpoint=checkpoint,
                objective_id=checkpoint.objective_id,
            )
