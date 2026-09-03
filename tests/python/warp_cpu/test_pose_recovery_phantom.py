"""Bounded production-driver recovery of an asymmetric analytic phantom.

The host runs production Warp kernels inside the explicit test context. Adding
--run-gpu also runs the same fixtures on cuda:0, with no CPU substitution there.
"""

import importlib
import json
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import pytest

from dpt._runtime import cpu_test_context
from dpt.geometry import RigidTransform
from dpt.objectives import ObjectiveSpec
from dpt.recovery import PrimaryPoseEvaluator, PrimaryPoseProblem, recover_pose
from dpt.registration import RecoveryPolicy
from dpt.spectral import SpectralSpec
from dpt.spectral_recovery import (
    CalibrationBlock,
    SpectralPoseEvaluator,
    SpectralPoseProblem,
    SpectralView,
    recover_spectral_pose,
)
from dpt.validation.recovery import pose_error
from helpers.pose_inputs import (
    LANDMARKS,
    PROVENANCE,
    _boundary_backtracking,
    _chart,
    geometries,
    phantom,
)

wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("near_boundary", [False, True], ids=["wide_chart", "near_boundary"])
def test_recover_pose_recovers_all_six_perturbed_coordinates(
    warp_device: str,
    sign: float,
    near_boundary: bool,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        grid, values = phantom()
        geometry = geometries()[0]
        attenuation = wp.array(values, dtype=wp.float32, device=warp_device)
        observed = wp.zeros(geometry.pixels, dtype=wp.float32, device=warp_device)
        initial = tuple(sign * value for value in (0.25, -0.18, 0.13, 0.3, -0.4, 0.2))
        chart = _chart(initial, near_boundary)
        problem = PrimaryPoseProblem(
            grid,
            geometry,
            attenuation,
            observed,
            open_beam=100.0,
            samples_per_ray=96,
        )
        synthesis = PrimaryPoseEvaluator(problem, chart, device=warp_device)
        synthesis((0.0,) * 6)
        observed.assign(synthesis.prediction.numpy().astype("float32"))
        evaluator = PrimaryPoseEvaluator(problem, chart, device=warp_device)
        initial_error = pose_error(chart.pose(initial), RigidTransform(), LANDMARKS)
        recovered = recover_pose(
            evaluator,
            initial=initial,
            policy=RecoveryPolicy(
                max_iterations=200,
                max_evaluations=1200,
                gradient_tolerance=1e-7,
                relative_loss_tolerance=1e-14,
            ),
        )
        error = pose_error(recovered.pose, RigidTransform(), LANDMARKS)
        assert recovered.optimisation.reason == "gradient_tolerance"
        assert error.translation_mm < 1e-4
        assert error.rotation_radians < 1e-5
        assert error.landmark_rms_mm < min(1e-4, initial_error.landmark_rms_mm / 100)
        assert recovered.optimisation.evaluation.loss < 1e-8
        if near_boundary:
            evidence = _boundary_backtracking(chart, initial, recovered.optimisation)
            evidence.update(
                translation_error_mm=error.translation_mm,
                rotation_error_radians=error.rotation_radians,
                landmark_rms_error_mm=error.landmark_rms_mm,
            )
            record_testsuite_property(
                f"primary_pose_near_boundary_{warp_device}_{sign}", json.dumps(evidence)
            )


@pytest.mark.parametrize("near_boundary", [False, True], ids=["wide_chart", "near_boundary"])
def test_spectral_pose_recovers_shared_exposure_offset_and_six_pose_coordinates(
    warp_device: str,
    near_boundary: bool,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        grid, values = phantom()
        fields = wp.array([10 * value for value in values], dtype=wp.float32, device=warp_device)
        coefficients = wp.array([0.2, 0.5], dtype=wp.float32, device=warp_device)
        weights = wp.array([100.0, 80.0], dtype=wp.float32, device=warp_device)
        response = wp.array([1.0, 2.0], dtype=wp.float32, device=warp_device)
        spectral = SpectralSpec(
            1, 2, (PROVENANCE,), PROVENANCE, PROVENANCE, "signal", "analytic test basis"
        )
        views = tuple(
            SpectralView(
                f"view-{index}",
                geometry,
                spectral,
                coefficients,
                weights,
                response,
                wp.zeros(geometry.pixels, dtype=wp.float32, device=warp_device),
                "shared",
                PROVENANCE,
                objective=ObjectiveSpec(reduction="mean"),
            )
            for index, geometry in enumerate(geometries())
        )
        initial = (0.25, -0.18, 0.13, 0.3, -0.4, 0.2, 0.2, 0.3)
        chart = _chart(initial, near_boundary)
        problem = SpectralPoseProblem(
            grid,
            1,
            fields,
            PROVENANCE,
            views,
            (CalibrationBlock("shared", fit_scale="exposure", fit_offset=True, scale_step=0.1),),
            96,
        )
        synthesis = SpectralPoseEvaluator(problem, chart, device=warp_device)
        synthesis((0.0,) * 8)
        for view, prepared in zip(views, synthesis._prepared, strict=True):
            wp.copy(view.observation, prepared.prediction)
        wp.synchronize_device(warp_device)
        evaluator = SpectralPoseEvaluator(problem, chart, device=warp_device)
        initial_error = pose_error(chart.pose(initial[:6]), RigidTransform(), LANDMARKS)
        recovered = recover_spectral_pose(
            evaluator,
            initial=initial,
            policy=RecoveryPolicy(
                max_iterations=200,
                max_evaluations=1200,
                gradient_tolerance=1e-3,
                relative_loss_tolerance=1e-12,
            ),
        )
        error = pose_error(recovered.pose, RigidTransform(), LANDMARKS)
        estimate = recovered.calibration[0]
        assert recovered.optimisation.reason == "gradient_tolerance"
        assert error.translation_mm < 1e-3
        assert error.rotation_radians < 1e-3
        assert error.landmark_rms_mm < min(1e-3, initial_error.landmark_rms_mm / 100)
        assert abs(estimate.exposure - 1.0) < 5e-5
        assert abs(estimate.offset) < 0.01
        assert estimate.gain == 1.0
        assert recovered.optimisation.evaluation.loss < 1e-5
        if near_boundary:
            evidence = _boundary_backtracking(chart, initial, recovered.optimisation)
            evidence.update(
                translation_error_mm=error.translation_mm,
                rotation_error_radians=error.rotation_radians,
                landmark_rms_error_mm=error.landmark_rms_mm,
                exposure_error=abs(estimate.exposure - 1.0),
                offset_error=abs(estimate.offset),
            )
            record_testsuite_property(
                f"spectral_pose_near_boundary_{warp_device}", json.dumps(evidence)
            )
