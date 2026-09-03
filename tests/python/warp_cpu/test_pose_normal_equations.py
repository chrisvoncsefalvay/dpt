"""Native resident normal equations checked against independent image derivatives."""

import importlib
import math
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Literal, cast

import numpy as np
import pytest
from scipy.linalg import expm

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError
from dpt.geometry import Matrix3, RigidTransform, Vector3
from dpt.materials import Provenance
from dpt.objectives import ObjectiveSpec
from dpt.pose_optimization import (
    ParameterBounds,
    PoseLeastSquaresPolicy,
    PrimaryPoseLeastSquaresEvaluator,
    recover_bounded_parameters,
    recover_pose_sensitivities,
)
from dpt.recovery import PoseChart, PrimaryPoseEvaluator, PrimaryPoseProblem
from dpt.spectral import SpectralSpec
from dpt.spectral_recovery import (
    CalibrationBlock,
    SpectralPoseEvaluator,
    SpectralPoseProblem,
    SpectralView,
)
from dpt.validation.projection import integrate_sampled_field
from helpers.pose_inputs import CHART_SCALES, geometries, phantom

wp: Any = importlib.import_module("warp")


def independent_pose(chart: PoseChart, parameters: tuple[float, ...]) -> RigidTransform:
    # Matrix exponential provides an independent fixed-chart differential,
    # including Lie translation/rotation coupling away from the anchor.
    tx, ty, tz, rx, ry, rz = np.asarray(parameters) * chart.scales
    algebra = np.asarray(
        [[0, -rz, ry, tx], [rz, 0, -rx, ty], [-ry, rx, 0, tz], [0, 0, 0, 0]], dtype=float
    )
    anchor = np.eye(4)
    anchor[:3, :3] = np.asarray(chart.anchor.rotation).reshape(3, 3)
    anchor[:3, 3] = chart.anchor.translation_mm
    matrix = anchor @ expm(algebra)
    return RigidTransform(
        cast(Matrix3, tuple(float(v) for v in matrix[:3, :3].ravel())),
        cast(Vector3, tuple(float(v) for v in matrix[:3, 3])),
    )


def case(
    device: str,
    *,
    domain: Literal["counts", "log_transmission"] = "counts",
    weighted: bool = False,
    reduction: Literal["sum", "mean"] = "sum",
    shape: tuple[int, int] = (2, 7),
    integration: Literal["midpoint", "cell_gauss"] = "midpoint",
) -> tuple[PrimaryPoseProblem, PoseChart, list[float]]:
    grid, values = phantom()
    geometry = replace(
        geometries()[0],
        shape=shape,
        origin_mm=(30.0, -0.8, -0.5),
        spacing_mm=(0.17, 0.19),
    )
    observation = wp.array(
        [19 + 0.02 * p if domain == "counts" else -0.2 + 0.001 * p for p in range(geometry.pixels)],
        dtype=wp.float32,
        device=device,
    )
    weights = (
        wp.array(
            [0.0 if p % 11 == 0 else 0.5 + p % 5 * 0.2 for p in range(geometry.pixels)],
            dtype=wp.float32,
            device=device,
        )
        if weighted
        else None
    )
    problem = PrimaryPoseProblem(
        grid,
        geometry,
        wp.array(values, dtype=wp.float32, device=device),
        observation,
        objective=ObjectiveSpec(domain=domain, weighted=weighted, reduction=reduction),
        open_beam=23.0,
        weights=weights,
        samples_per_ray=64,
        integration=integration,
    )
    chart = PoseChart(scales=(0.11, 0.09, 0.13, 0.012, 0.014, 0.009))
    # The scalar reference must see precisely the FP32 field used by native kernels.
    return problem, chart, np.asarray(values, dtype=np.float32).astype(float).tolist()


@pytest.mark.parametrize("domain", ["counts", "log_transmission"])
@pytest.mark.parametrize("weighted,reduction", [(False, "sum"), (True, "mean")])
@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_resident_normal_equations_match_independent_fixed_chart_image_jacobian(
    warp_device: str,
    domain: Literal["counts", "log_transmission"],
    weighted: bool,
    reduction: Literal["sum", "mean"],
    integration: Literal["midpoint", "cell_gauss"],
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        problem, chart, values = case(
            warp_device,
            domain=domain,
            weighted=weighted,
            reduction=reduction,
            integration=integration,
        )
        evaluator = PrimaryPoseLeastSquaresEvaluator(problem, chart, device=warp_device)
        point = (0.17, -0.11, 0.13, 0.4, -0.3, 0.2)
        actual = evaluator(point)

        def signal(parameters: tuple[float, ...]) -> np.ndarray:
            transform = independent_pose(chart, parameters)
            depths = [
                integrate_sampled_field(
                    values,
                    problem.grid,
                    problem.geometry,
                    transform,
                    *divmod(pixel, problem.geometry.shape[1]),
                    midpoint_samples=64 if integration == "midpoint" else None,
                )
                for pixel in range(problem.geometry.pixels)
            ]
            return 23 * np.exp(-np.asarray(depths)) if domain == "counts" else -np.asarray(depths)

        prediction = signal(point)
        jacobian = np.empty((problem.geometry.pixels, 6))
        for axis in range(6):
            step = 2e-5
            plus, minus = list(point), list(point)
            plus[axis] += step
            minus[axis] -= step
            jacobian[:, axis] = (signal(tuple(plus)) - signal(tuple(minus))) / (2 * step)
        residual = prediction - problem.observation.numpy().astype(float)
        weights = (
            problem.weights.numpy().astype(float) if weighted else np.ones(problem.geometry.pixels)
        )
        if reduction == "mean":
            weights /= problem.geometry.pixels
        expected_loss = 0.5 * np.dot(weights, residual * residual)
        expected_gradient = jacobian.T @ (weights * residual)
        expected_matrix = jacobian.T @ (weights[:, None] * jacobian)
        assert actual.loss == pytest.approx(expected_loss, rel=2e-12, abs=2e-12)
        np.testing.assert_allclose(actual.gradient, expected_gradient, rtol=2e-6, atol=3e-9)
        np.testing.assert_allclose(actual.gauss_newton, expected_matrix, rtol=3e-6, atol=1e-10)
        np.testing.assert_allclose(
            evaluator.predicted_images[0].numpy(), prediction, rtol=2e-13, atol=2e-13
        )
        assert evaluator.control_readback_values == 45


@pytest.mark.parametrize("pixels", [1, 127, 129, 323])
def test_multiple_view_stacking_and_reduction_tree_have_no_image_readback(
    warp_device: str, pixels: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        first, chart, _ = case(warp_device, shape=(1, pixels))
        second = replace(
            first,
            geometry=replace(first.geometry, source_mm=(-30.0, 0.3, -0.1)),
            observation=wp.array(
                [17 + 0.001 * p for p in range(pixels)], dtype=wp.float32, device=warp_device
            ),
        )
        stacked = PrimaryPoseLeastSquaresEvaluator((first, second), chart, device=warp_device)
        one = PrimaryPoseLeastSquaresEvaluator(first, chart, device=warp_device)
        two = PrimaryPoseLeastSquaresEvaluator(second, chart, device=warp_device)
        point = (0.1, -0.2, 0.15, 0.3, -0.2, 0.1)
        a, b = one(point), two(point)
        original_numpy = wp.array.numpy

        def bounded_readback(array: Any) -> Any:
            # Pinned construction happened before this guard. During production
            # evaluation only the prepared small completion payload is read.
            if array.device.is_cuda:
                pytest.fail("production normal equations downloaded a device array")
            return original_numpy(array)

        monkeypatch.setattr(wp.array, "numpy", bounded_readback)
        actual = stacked(point)
        assert actual.loss == pytest.approx(a.loss + b.loss, rel=2e-13)
        np.testing.assert_allclose(
            actual.gradient, np.asarray(a.gradient) + b.gradient, rtol=2e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            actual.gauss_newton, np.asarray(a.gauss_newton) + b.gauss_newton, rtol=2e-13, atol=1e-13
        )
        assert stacked.control_readback_values == 89


@pytest.mark.parametrize("method", ["gauss_newton", "levenberg_marquardt"])
def test_actual_primary_sensitivities_recover_all_six_coordinates(
    warp_device: str, method: Literal["gauss_newton", "levenberg_marquardt"]
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        grid, values = phantom()
        field = wp.array(values, dtype=wp.float32, device=warp_device)
        chart = PoseChart(scales=CHART_SCALES)
        problems = tuple(
            PrimaryPoseProblem(
                grid,
                geometry,
                field,
                wp.zeros(geometry.pixels, dtype=wp.float32, device=warp_device),
                open_beam=100.0,
                samples_per_ray=96,
            )
            for geometry in geometries()
        )
        for problem in problems:
            synthesis = PrimaryPoseEvaluator(problem, chart, device=warp_device)
            synthesis((0.0,) * 6)
            problem.observation.assign(synthesis.prediction.numpy().astype(np.float32))
        evaluator = PrimaryPoseLeastSquaresEvaluator(problems, chart, device=warp_device)
        recovered = recover_pose_sensitivities(
            evaluator,
            initial=(0.25, -0.18, 0.13, 0.3, -0.4, 0.2),
            policy=PoseLeastSquaresPolicy(
                method=method, max_iterations=100, max_evaluations=400, gradient_tolerance=1e-8
            ),
        )
        assert recovered.optimisation.stationary
        assert recovered.optimisation.evaluation.loss < 2e-8
        assert math.hypot(*recovered.pose.translation_mm) < 1e-5
        assert np.linalg.norm(np.asarray(recovered.pose.rotation).reshape(3, 3) - np.eye(3)) < 1e-5
        assert all(
            not row.accepted or row.loss <= recovered.optimisation.history[0].loss
            for row in recovered.optimisation.history
        )


def test_primary_translation_bound_and_native_calibration_box_reach_kkt(warp_device: str) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        problem, chart, _ = case(warp_device)
        synthesis = PrimaryPoseLeastSquaresEvaluator(problem, chart, device=warp_device)
        synthesis((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        problem.observation.assign(synthesis.predicted_images[0].numpy().astype(np.float32))
        fit = PrimaryPoseLeastSquaresEvaluator(problem, chart, device=warp_device)
        result = recover_pose_sensitivities(
            fit,
            bounds=ParameterBounds((-1.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.4, 0.0, 0.0, 0.0, 0.0, 0.0)),
        )
        assert result.optimisation.reason == "projected_gradient_tolerance"
        assert result.optimisation.parameters[0] == 0.4
        assert result.optimisation.evaluation.gradient[0] < 0

        provenance = Provenance(
            "analytic unit-spectrum calibration regression",
            "0" * 64,
            "Original regression fixture; Apache-2.0",
            "Known constant two-photon signal and bounded electronic offset",
        )
        geometry = replace(problem.geometry, shape=(1, 1))
        spectral = SpectralSpec(
            1,
            1,
            (provenance,),
            provenance,
            provenance,
            "signal",
            "constant analytic calibration fixture",
        )
        view = SpectralView(
            "constant",
            geometry,
            spectral,
            wp.zeros(1, dtype=wp.float32, device=warp_device),
            wp.array([2.0], dtype=wp.float32, device=warp_device),
            wp.ones(1, dtype=wp.float32, device=warp_device),
            wp.array([5.0], dtype=wp.float32, device=warp_device),
            "offset",
            provenance,
        )
        spectral_problem = SpectralPoseProblem(
            problem.grid,
            1,
            problem.attenuation,
            provenance,
            (view,),
            (CalibrationBlock("offset", fit_offset=True),),
            16,
        )
        native = SpectralPoseEvaluator(spectral_problem, chart, device=warp_device)
        bounded = recover_bounded_parameters(
            native, (0.0,) * 7, bounds=ParameterBounds((0.0,) * 7, (0.0,) * 6 + (1.0,))
        )
        assert bounded.reason == "projected_gradient_tolerance"
        assert bounded.parameters == (0.0,) * 6 + (1.0,)
        assert bounded.evaluation.loss == pytest.approx(2.0)
        assert bounded.evaluation.gradient[-1] == pytest.approx(-2.0)


def test_status_pending_failure_retry_dtype_stream_tape_and_budget(warp_device: str) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        problem, chart, _ = case(warp_device, weighted=True)
        with pytest.raises(ContractError, match="max_bytes"):
            PrimaryPoseLeastSquaresEvaluator(problem, chart, device=warp_device, max_bytes=10)
        with pytest.raises(ContractError, match="unmasked"):
            PrimaryPoseLeastSquaresEvaluator(
                replace(problem, objective=ObjectiveSpec(kind="poisson", domain="counts")),
                chart,
                device=warp_device,
            )
        problem.weights.assign(np.full(problem.geometry.pixels, -1.0, dtype=np.float32))
        evaluator = PrimaryPoseLeastSquaresEvaluator(problem, chart, device=warp_device)
        with pytest.raises(ContractError, match="weights"):
            evaluator((0.0,) * 6)
        problem.weights.assign(np.ones(problem.geometry.pixels, dtype=np.float32))
        assert math.isfinite(evaluator((0.0,) * 6).loss)
        with wp.Tape(), pytest.raises(ContractError, match="ambient"):
            evaluator((0.0,) * 6)
        assert evaluator._lock.acquire(blocking=False)
        try:
            with pytest.raises(ContractError, match="concurrently"):
                evaluator((0.0,) * 6)
        finally:
            evaluator._lock.release()
        with pytest.raises(ContractError, match="observation"):
            PrimaryPoseLeastSquaresEvaluator(
                replace(
                    problem,
                    observation=wp.zeros(
                        problem.geometry.pixels, dtype=wp.float64, device=warp_device
                    ),
                ),
                chart,
                device=warp_device,
            )


@pytest.mark.parametrize("domain", ["counts", "log_transmission"])
def test_primary_normal_gradient_preserves_existing_default_evaluator(
    warp_device: str, domain: Literal["counts", "log_transmission"]
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        problem, chart, _ = case(warp_device, domain=domain, weighted=True, reduction="mean")
        point = (0.13, -0.2, 0.17, 0.3, -0.1, 0.2)
        old = PrimaryPoseEvaluator(problem, chart, device=warp_device)(point)
        normal = PrimaryPoseLeastSquaresEvaluator(problem, chart, device=warp_device)(point)
        assert normal.loss == pytest.approx(old.loss, rel=2e-13, abs=1e-13)
        assert normal.gradient == pytest.approx(old.gradient, rel=2e-12, abs=2e-13)


def test_float32_depth_uses_explicit_fp64_signal_and_native_seed_precision(
    warp_device: str,
) -> None:
    from dpt.geometry import chart_gradient
    from dpt.projection import projection_vjp

    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        problem, chart, _ = case(warp_device)
        problem = replace(problem, precision="float32")
        evaluator = PrimaryPoseLeastSquaresEvaluator(problem, chart, device=warp_device)
        point = (0.1, -0.1, 0.2, 0.3, -0.2, 0.1)
        actual = evaluator(point)
        view = evaluator._views[0]
        expected_signal = problem.open_beam * np.exp(-view.depth.numpy().astype(float))
        np.testing.assert_allclose(view.prediction.numpy(), expected_signal, rtol=4e-15, atol=0)
        expected_seed = (-expected_signal).astype(np.float32)
        np.testing.assert_array_equal(view.seeds.numpy(), expected_seed)
        # Canonical native VJP and a host contraction of resident sensitivity
        # rows provide a separate check of the declared FP32 seed rounding.
        residual = expected_signal - problem.observation.numpy().astype(float)
        local = (view.jacobian.numpy().reshape(-1, 6) * residual[:, None]).sum(axis=0)
        expected = chart_gradient(chart.increment(point), tuple(float(v) for v in local))
        expected = tuple(g * scale for g, scale in zip(expected, chart.scales, strict=True))
        assert actual.gradient == pytest.approx(expected, rel=2e-12, abs=2e-12)
        seeds = wp.array(expected_seed, dtype=wp.float32, device=warp_device)
        result = wp.empty(6, dtype=wp.float64, device=warp_device)
        projection_vjp(
            problem.attenuation,
            view.pose,
            cotangent=seeds,
            out_pose_tangent=result,
            workspace=view.projection,
        )
        np.testing.assert_allclose(
            result.numpy(), view.jacobian.numpy().reshape(-1, 6).sum(axis=0), rtol=2e-12, atol=2e-12
        )
