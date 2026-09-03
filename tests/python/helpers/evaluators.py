"""Shared fixtures and independent oracles: evaluators."""

import importlib
from typing import Any

from dpt.material_reconstruction import MaterialReconstruction
from dpt.recovery import PoseChart, PrimaryPoseEvaluator, PrimaryPoseProblem
from helpers import inverse_inputs as spectral_cases
from helpers import material_inputs as material_cases
from helpers import pose_inputs as pose_cases

wp: Any = importlib.import_module("warp")


def _evaluator(kind, device) -> tuple[Any, Any]:
    if kind == "spectral":
        evaluator, _ = spectral_cases._spectral_fixture(device)
        parameters = (0.31, -0.23, 0.19, 0.29, -0.26, 0.22, 0.15, -0.2, -0.14, 0.23)
        return evaluator, parameters
    if kind == "primary":
        grid, values = pose_cases.phantom()
        geometry = pose_cases.geometries()[0]
        problem = PrimaryPoseProblem(
            grid,
            geometry,
            wp.array(values, dtype=wp.float32, device=device),
            wp.full(geometry.pixels, 90.0, dtype=wp.float32, device=device),
            open_beam=100.0,
            samples_per_ray=96,
        )
        return PrimaryPoseEvaluator(problem, PoseChart(), device=device), (
            0.1,
            -0.1,
            0.1,
            0.01,
            -0.01,
            0.01,
        )
    inputs = material_cases._inputs("float64")
    inputs["device"] = device
    return MaterialReconstruction(**inputs), None


def _evaluate(evaluator: Any, parameters: Any) -> Any:
    return evaluator.evaluate() if parameters is None else evaluator(parameters)
