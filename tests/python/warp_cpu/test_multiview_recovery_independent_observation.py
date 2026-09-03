"""Joint pose recovery from independently synthesised, represented observations.

The original asymmetric phantom is observed at a nonzero six-coordinate pose.
Scalar midpoint integration and a matrix exponential supply observations and
image derivatives before any production arrays or evaluator are constructed.
"""

import hashlib
import importlib
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import pairwise
from typing import Any, Literal, cast

import numpy as np
import pytest
from scipy.linalg import expm

from dpt._runtime import cpu_test_context
from dpt.geometry import DetectorGeometry, Matrix3, RigidTransform, Vector3
from dpt.objectives import ObjectiveSpec
from dpt.pose_optimization import (
    PoseLeastSquaresPolicy,
    PrimaryPoseLeastSquaresEvaluator,
    recover_pose_sensitivities,
)
from dpt.recovery import PoseChart, PrimaryPoseProblem
from dpt.validation.projection import integrate_sampled_field
from dpt.validation.recovery import pose_error
from dpt.volumes import GridSpec
from helpers.pose_inputs import CHART_SCALES, LANDMARKS, geometries, phantom

wp: Any = importlib.import_module("warp")
Domain = Literal["counts", "log_transmission"]
Method = Literal["gauss_newton", "levenberg_marquardt"]
TRUTH = (0.17, -0.11, 0.13, 0.4, -0.3, 0.2)
INITIAL = tuple(a + b for a, b in zip(TRUTH, (0.23, -0.17, 0.12, 0.28, -0.38, 0.19), strict=True))
SAMPLES = 96
FIELD_SCALE = 10.0
FD_STEP = 2e-5
GRADIENT_TOLERANCE = 1e-8
GEOMETRIC_CEILING = 1e-3
QUANTISATION_MARGIN = 0.1
CONDITION_CEILING = 1e4


@dataclass(frozen=True)
class ReferenceScene:
    grid: GridSpec
    values: tuple[float, ...]
    chart: PoseChart
    panels: tuple[DetectorGeometry, DetectorGeometry]
    weights: tuple[Any, Any]


@lru_cache(maxsize=1)
def _scene() -> ReferenceScene:
    grid, values = phantom()
    represented = tuple(
        float(value) for value in np.asarray(values, dtype=np.float32) * FIELD_SCALE
    )
    assert len(represented) == grid.voxels
    assert all(math.isfinite(value) and value >= 0 for value in represented)
    cosine, sine = math.cos(0.12), math.sin(0.12)
    chart = PoseChart(
        anchor=RigidTransform(
            rotation=(cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0),
            translation_mm=(0.07, -0.04, 0.03),
        ),
        scales=CHART_SCALES,
    )
    x_panel, y_panel = geometries()
    panels = (x_panel, replace(y_panel, shape=(10, 12), origin_mm=(-5.5, 30.0, -4.5)))
    weights = tuple(
        np.asarray(
            [
                0.0 if pixel % 7 == 0 else 0.5 + (pixel + index) % 5 * 0.2
                for pixel in range(panel.pixels)
            ],
            dtype=np.float32,
        ).astype(float)
        for index, panel in enumerate(panels)
    )
    return ReferenceScene(grid, represented, chart, panels, cast(tuple[Any, Any], weights))


def _reference_pose(parameters: tuple[float, ...]) -> RigidTransform:
    chart = _scene().chart
    tx, ty, tz, rx, ry, rz = np.asarray(parameters) * chart.scales
    algebra = np.array(
        [[0, -rz, ry, tx], [rz, 0, -rx, ty], [-ry, rx, 0, tz], [0, 0, 0, 0]], dtype=float
    )
    anchor = np.eye(4)
    anchor[:3, :3] = np.asarray(chart.anchor.rotation).reshape(3, 3)
    anchor[:3, 3] = chart.anchor.translation_mm
    matrix = anchor @ expm(algebra)
    return RigidTransform(
        cast(Matrix3, tuple(float(value) for value in matrix[:3, :3].ravel())),
        cast(Vector3, tuple(float(value) for value in matrix[:3, 3])),
    )


@lru_cache(maxsize=96)
def _depths(parameters: tuple[float, ...], view: int) -> Any:
    scene = _scene()
    panel = scene.panels[view]
    return np.asarray(
        [
            integrate_sampled_field(
                scene.values,
                scene.grid,
                panel,
                _reference_pose(parameters),
                *divmod(pixel, panel.shape[1]),
                midpoint_samples=SAMPLES,
                validate=False,
            )
            for pixel in range(panel.pixels)
        ]
    )


def _signal(parameters: tuple[float, ...], view: int, domain: Domain) -> Any:
    depths = _depths(parameters, view)
    return (
        np.asarray([(23.0 + 8.0 * view) * math.exp(-float(depth)) for depth in depths])
        if domain == "counts"
        else -depths
    )


def _jacobian(parameters: tuple[float, ...], view: int, domain: Domain, step: float) -> Any:
    result = np.empty((_scene().panels[view].pixels, 6))
    for axis in range(6):
        plus, minus = list(parameters), list(parameters)
        plus[axis] += step
        minus[axis] -= step
        result[:, axis] = (
            _signal(tuple(plus), view, domain) - _signal(tuple(minus), view, domain)
        ) / (2 * step)
    return result


def _stack(
    parameters: tuple[float, ...],
    domain: Domain,
    observations: tuple[Any, Any],
    step: float = FD_STEP,
) -> tuple[Any, Any, tuple[Any, ...]]:
    residuals, jacobians, predictions = [], [], []
    for view, panel in enumerate(_scene().panels):
        prediction = _signal(parameters, view, domain)
        root_weight = np.sqrt(_scene().weights[view] / (panel.pixels if view else 1))
        residuals.append(root_weight * (prediction - observations[view]))
        jacobians.append(root_weight[:, None] * _jacobian(parameters, view, domain, step))
        predictions.append(prediction)
    return np.concatenate(residuals), np.vstack(jacobians), tuple(predictions)


def _normal(
    residual: Any, jacobian: Any
) -> tuple[float, tuple[float, ...], tuple[tuple[float, ...], ...]]:
    loss = 0.5 * math.fsum(float(value) ** 2 for value in residual)
    gradient = tuple(
        math.fsum(float(a) * float(b) for a, b in zip(jacobian[:, axis], residual, strict=True))
        for axis in range(6)
    )
    matrix = tuple(
        tuple(
            math.fsum(
                float(a) * float(b) for a, b in zip(jacobian[:, i], jacobian[:, j], strict=True)
            )
            for j in range(6)
        )
        for i in range(6)
    )
    return loss, gradient, matrix


@lru_cache(maxsize=2)
def _reference_admission(domain: Domain) -> tuple[tuple[Any, Any], dict[str, Any]]:
    ideal = tuple(_signal(TRUTH, view, domain) for view in range(2))
    observations = cast(
        tuple[Any, Any], tuple(values.astype(np.float32).astype(float) for values in ideal)
    )
    residual, jacobian, _ = _stack(TRUTH, domain, observations)
    _, fine, _ = _stack(TRUTH, domain, observations, FD_STEP / 2)
    np.testing.assert_allclose(jacobian, fine, rtol=1e-5, atol=1e-8)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    assert singular.size == 6 and singular[-1] > 0
    condition = float(singular[0] / singular[-1])
    assert condition <= CONDITION_CEILING
    # Half an FP32 spacing bounds observation rounding at represented endpoints.
    # This is a local linear conditioning estimate, not a global nonlinear bound.
    weighted_half_ulps = []
    for view, panel in enumerate(_scene().panels):
        rounded = observations[view].astype(np.float32)
        half_ulps = np.abs(np.spacing(rounded).astype(float)) / 2
        assert np.all(np.abs(ideal[view] - observations[view]) <= half_ulps)
        weighted_half_ulps.extend(
            np.sqrt(_scene().weights[view] / (panel.pixels if view else 1)) * half_ulps
        )
    noise_bound = math.hypot(*(float(value) for value in weighted_half_ulps))
    coordinate_bound = noise_bound / float(singular[-1])
    radius = max(math.hypot(*landmark) for landmark in LANDMARKS)
    landmark_scale = math.hypot(max(CHART_SCALES[:3]), max(CHART_SCALES[3:]) * radius)
    local_landmark_estimate = coordinate_bound * landmark_scale
    assert local_landmark_estimate <= QUANTISATION_MARGIN * GEOMETRIC_CEILING
    initial_residual, initial_jacobian, _ = _stack(INITIAL, domain, observations)
    _, initial_fine, _ = _stack(INITIAL, domain, observations, FD_STEP / 2)
    np.testing.assert_allclose(initial_jacobian, initial_fine, rtol=1e-5, atol=1e-8)
    assert np.linalg.matrix_rank(initial_jacobian) == 6
    admission = {
        "source": "original Apache-2.0 phantom; independent expm/scalar96-midpoint observations",
        "field_sha256": hashlib.sha256(
            np.asarray(_scene().values, dtype=np.float32).tobytes()
        ).hexdigest(),
        "observation_sha256": [
            hashlib.sha256(values.astype(np.float32).tobytes()).hexdigest()
            for values in observations
        ],
        "truth": TRUTH,
        "initial": INITIAL,
        "panels": [panel.shape for panel in _scene().panels],
        "stacked_pixels": int(residual.size),
        "condition_number": condition,
        "sigma_min": float(singular[-1]),
        "weighted_half_ulp32_norm": noise_bound,
        "local_linear_landmark_quantisation_estimate_mm": local_landmark_estimate,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "initial_independent_loss": _normal(initial_residual, initial_jacobian)[0],
    }
    return observations, admission


def _compare_normal(
    actual: Any, reference: tuple[float, tuple[float, ...], tuple[tuple[float, ...], ...]]
) -> None:
    loss, gradient, matrix = reference
    assert actual.loss == pytest.approx(loss, rel=3e-12, abs=3e-12)
    np.testing.assert_allclose(actual.gradient, gradient, rtol=3e-6, atol=3e-9)
    np.testing.assert_allclose(actual.gauss_newton, matrix, rtol=4e-6, atol=2e-10)


@pytest.mark.parametrize("domain", ["counts", "log_transmission"])
@pytest.mark.parametrize("method", ["gauss_newton", "levenberg_marquardt"])
def test_joint_nonzero_pose_recovery_from_independent_two_view_observations(
    warp_device: str,
    domain: Domain,
    method: Method,
    request: pytest.FixtureRequest,
) -> None:
    observations, admission = _reference_admission(domain)
    reference_initial = _stack(INITIAL, domain, observations)
    initial_normal = _normal(reference_initial[0], reference_initial[1])
    scene = _scene()
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        field = wp.array(scene.values, dtype=wp.float32, device=warp_device)
        problems = tuple(
            PrimaryPoseProblem(
                scene.grid,
                panel,
                field,
                wp.array(observations[view], dtype=wp.float32, device=warp_device),
                objective=ObjectiveSpec(
                    domain=domain, weighted=True, reduction="sum" if view == 0 else "mean"
                ),
                open_beam=23.0 + 8.0 * view,
                weights=wp.array(scene.weights[view], dtype=wp.float32, device=warp_device),
                samples_per_ray=SAMPLES,
            )
            for view, panel in enumerate(scene.panels)
        )
        evaluator = PrimaryPoseLeastSquaresEvaluator(problems, scene.chart, device=warp_device)
        _compare_normal(evaluator(INITIAL), initial_normal)
        for actual_image, reference_image in zip(
            evaluator.predicted_images, reference_initial[2], strict=True
        ):
            np.testing.assert_allclose(
                actual_image.numpy(), reference_image, rtol=2e-12, atol=2e-12
            )
        permuted = PrimaryPoseLeastSquaresEvaluator(
            tuple(reversed(problems)), scene.chart, device=warp_device
        )
        _compare_normal(permuted(INITIAL), initial_normal)
        result = recover_pose_sensitivities(
            evaluator,
            initial=INITIAL,
            policy=PoseLeastSquaresPolicy(
                method=method,
                max_iterations=100,
                max_evaluations=400,
                gradient_tolerance=GRADIENT_TOLERANCE,
            ),
        )
        optimisation = result.optimisation
        assert optimisation.stationary and optimisation.reason == "gradient_tolerance"
        assert optimisation.evaluations <= 400
        assert optimisation.evaluation.loss < 1e-8
        accepted = [row.loss for row in optimisation.history if row.accepted]
        assert accepted and all(later <= earlier for earlier, later in pairwise(accepted))
        reference_final = _stack(optimisation.parameters, domain, observations)
        final_normal = _normal(reference_final[0], reference_final[1])
        _compare_normal(optimisation.evaluation, final_normal)
        assert math.hypot(*final_normal[1]) <= 2e-8
        error = pose_error(result.pose, _reference_pose(TRUTH), LANDMARKS)
        initial_error = pose_error(_reference_pose(INITIAL), _reference_pose(TRUTH), LANDMARKS)
        assert error.translation_mm < GEOMETRIC_CEILING
        assert error.rotation_radians < GEOMETRIC_CEILING
        assert error.landmark_max_mm < GEOMETRIC_CEILING
        assert error.landmark_rms_mm < initial_error.landmark_rms_mm / 100
        request.node.user_properties.extend(
            [
                ("s02_independent_admission", json.dumps(admission, sort_keys=True)),
                ("s02_device", warp_device),
                ("s02_method", method),
                ("s02_reason", optimisation.reason),
                ("s02_evaluations", optimisation.evaluations),
                ("s02_accepted_losses", json.dumps(accepted)),
                ("s02_final_reference_gradient_norm", math.hypot(*final_normal[1])),
                ("s02_translation_error_mm", error.translation_mm),
                ("s02_rotation_error_radians", error.rotation_radians),
                ("s02_landmark_max_error_mm", error.landmark_max_mm),
            ]
        )
