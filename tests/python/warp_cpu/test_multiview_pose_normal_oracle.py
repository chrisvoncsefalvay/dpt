"""Independent two-view residual/Jacobian stacking in one fixed pose chart.

References use a matrix exponential and scalar sampled-field integration, never
the production sensitivity evaluator, to construct both views' image Jacobians.
The bounded fixture exercises distinct panels, pixel counts, beams, weights and
per-view reductions on CPU and, when explicitly requested, actual CUDA.
"""

import importlib
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Literal, cast

import numpy as np
import pytest
from scipy.linalg import expm

from dpt._runtime import cpu_test_context
from dpt.geometry import Matrix3, RigidTransform, Vector3
from dpt.objectives import ObjectiveSpec
from dpt.pose_optimization import PrimaryPoseLeastSquaresEvaluator
from dpt.recovery import PoseChart, PrimaryPoseProblem
from dpt.validation.projection import integrate_sampled_field
from helpers.pose_inputs import geometries, phantom

wp: Any = importlib.import_module("warp")


def _reference_pose(chart: PoseChart, parameters: tuple[float, ...]) -> RigidTransform:
    tx, ty, tz, rx, ry, rz = np.asarray(parameters) * chart.scales
    algebra = np.array(
        [[0, -rz, ry, tx], [rz, 0, -rx, ty], [-ry, rx, 0, tz], [0, 0, 0, 0]], dtype=float
    )
    anchor = np.eye(4)
    anchor[:3, :3] = np.asarray(chart.anchor.rotation).reshape(3, 3)
    anchor[:3, 3] = chart.anchor.translation_mm
    transformed = anchor @ expm(algebra)
    return RigidTransform(
        cast(Matrix3, tuple(float(value) for value in transformed[:3, :3].ravel())),
        cast(Vector3, tuple(float(value) for value in transformed[:3, 3])),
    )


@pytest.mark.parametrize("domain", ["counts", "log_transmission"])
@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_two_view_normal_equations_match_independent_weighted_residual_jacobian_stack(
    warp_device: str,
    domain: Literal["counts", "log_transmission"],
    integration: Literal["midpoint", "cell_gauss"],
    request: pytest.FixtureRequest,
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        grid, analytic_values = phantom()
        # The independent reference uses exactly the native FP32 field inputs.
        values = np.asarray(analytic_values, dtype=np.float32).astype(float).tolist()
        field = wp.array(values, dtype=wp.float32, device=warp_device)
        chart = PoseChart(
            anchor=RigidTransform(translation_mm=(0.07, -0.04, 0.03)),
            scales=(0.11, 0.09, 0.13, 0.012, 0.014, 0.009),
        )
        problems = []
        host_observations = []
        host_weights = []
        for index, geometry in enumerate(geometries()):
            geometry = replace(
                geometry,
                shape=(2, 7) if index == 0 else (3, 5),
                origin_mm=(30.0, -0.8, -0.5) if index == 0 else (-0.6, 30.0, -0.4),
                spacing_mm=(0.17, 0.19) if index == 0 else (0.21, 0.16),
            )
            observations = np.asarray(
                [
                    19.0 + index + 0.02 * pixel
                    if domain == "counts"
                    else -0.2 + 0.03 * index + 0.001 * pixel
                    for pixel in range(geometry.pixels)
                ],
                dtype=np.float32,
            )
            weights = np.asarray(
                [
                    0.0 if pixel % 7 == 0 else 0.5 + (pixel + index) % 5 * 0.2
                    for pixel in range(geometry.pixels)
                ],
                dtype=np.float32,
            )
            reduction = "sum" if index == 0 else "mean"
            problems.append(
                PrimaryPoseProblem(
                    grid,
                    geometry,
                    field,
                    wp.array(observations, dtype=wp.float32, device=warp_device),
                    objective=ObjectiveSpec(domain=domain, weighted=True, reduction=reduction),
                    open_beam=23.0 + 8.0 * index,
                    weights=wp.array(weights, dtype=wp.float32, device=warp_device),
                    samples_per_ray=64,
                    integration=integration,
                )
            )
            host_observations.append(observations.astype(float))
            host_weights.append(weights.astype(float) / (geometry.pixels if index else 1))
        evaluator = PrimaryPoseLeastSquaresEvaluator(tuple(problems), chart, device=warp_device)
        point = (0.17, -0.11, 0.13, 0.4, -0.3, 0.2)
        actual = evaluator(point)
        residual_blocks = []
        jacobian_blocks = []
        for problem, observation, weights, prediction in zip(
            problems, host_observations, host_weights, evaluator.predicted_images, strict=True
        ):

            def signal(
                parameters: tuple[float, ...], selected: PrimaryPoseProblem = problem
            ) -> np.ndarray:
                pose = _reference_pose(chart, parameters)
                depths = np.asarray(
                    [
                        integrate_sampled_field(
                            values,
                            grid,
                            selected.geometry,
                            pose,
                            *divmod(pixel, selected.geometry.shape[1]),
                            midpoint_samples=64 if integration == "midpoint" else None,
                        )
                        for pixel in range(selected.geometry.pixels)
                    ]
                )
                return selected.open_beam * np.exp(-depths) if domain == "counts" else -depths

            expected_prediction = signal(point)
            jacobian = np.empty((problem.geometry.pixels, 6))
            for axis in range(6):
                step = 2e-5
                plus, minus = list(point), list(point)
                plus[axis] += step
                minus[axis] -= step
                jacobian[:, axis] = (signal(tuple(plus)) - signal(tuple(minus))) / (2 * step)
            root_weights = np.sqrt(weights)
            residual_blocks.append(root_weights * (expected_prediction - observation))
            jacobian_blocks.append(root_weights[:, None] * jacobian)
            np.testing.assert_allclose(
                prediction.numpy(), expected_prediction, rtol=2e-12, atol=2e-12
            )
        residual = np.concatenate(residual_blocks)
        jacobian = np.vstack(jacobian_blocks)
        assert jacobian.shape == (29, 6) and residual.shape == (29,)
        assert np.linalg.matrix_rank(jacobian) == 6
        assert actual.loss == pytest.approx(0.5 * np.dot(residual, residual), rel=3e-12, abs=3e-12)
        np.testing.assert_allclose(actual.gradient, jacobian.T @ residual, rtol=3e-6, atol=3e-9)
        np.testing.assert_allclose(
            actual.gauss_newton, jacobian.T @ jacobian, rtol=4e-6, atol=2e-10
        )
        request.node.user_properties.extend(
            [
                (
                    "s02_reference",
                    "independent expm/scalar image finite differences; sqrt(W) residual/J stack",
                ),
                ("s02_views", 2),
                ("s02_stacked_pixels", int(residual.size)),
                ("s02_per_view_reductions", "sum,mean"),
                ("s02_device", warp_device),
            ]
        )
