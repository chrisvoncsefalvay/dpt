"""Shared fixtures and independent oracles: material inputs."""

import importlib
from dataclasses import replace
from typing import Any, Literal

import numpy as np

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.kernels.material_reconstruction import projected_metric_step
from dpt.material_reconstruction import MaterialReconstructionSettings
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec
from helpers import material_fisher_inputs as fisher_cases
from helpers import material_solver_inputs as solver_cases

__all__ = ["fisher_cases", "integrate_sampled_field"]

wp: Any = importlib.import_module("warp")


def _metric_projection(fields, gradients, step, metric):
    n = fields.shape[1]
    a, b, c = metric
    determinant = a * c - b * b
    field = wp.array(fields.reshape(-1), dtype=wp.float32, device="cpu")
    gradient = wp.array(gradients.reshape(-1), dtype=wp.float32, device="cpu")
    out = wp.empty_like(field)
    slopes = wp.zeros(2 * n, dtype=wp.float64, device="cpu")
    mapping = wp.zeros(n, dtype=wp.float64, device="cpu")
    status = wp.zeros(1, dtype=wp.int32, device="cpu")
    wp.launch(
        projected_metric_step,
        dim=n,
        inputs=[
            field,
            gradient,
            n,
            step,
            field,
            0.0,
            False,
            field,
            wp.vec3d(*metric),
            wp.vec3d(c / determinant, -b / determinant, a / determinant),
            out,
            slopes,
            mapping,
            status,
        ],
        device="cpu",
    )
    return out.numpy().reshape(2, n), status.numpy()[0]


def _inputs(precision: Literal["float32", "float64"] = "float32", regularisation=0.3):
    grid = GridSpec((2, 3, 4), (0.5, 0.7, 0.9), (-0.75, -0.7, -0.45))
    inputs = solver_cases._inputs(grid=grid, precision=precision)
    geometry = DetectorGeometry(
        (-20, 0, 0), (20, -0.3, -0.2), (0, 1, 0), (0, 0, 1), (0.3, 0.4), (2, 3)
    )
    first = replace(
        inputs["views"][0],
        geometry=geometry,
        counts=np.array([9200, 8700], dtype=np.float32)[:, None, None]
        * np.ones((2, 2, 3), dtype=np.float32),
    )
    inputs["views"] = (first, replace(first, pose=RigidTransform(translation_mm=(0.02, 0, 0))))
    rng = np.random.default_rng(152)
    inputs["initial_fractions"] = np.ascontiguousarray(
        rng.uniform(0.05, 0.2, (2, *grid.shape)), dtype=np.float32
    )
    inputs["settings"] = MaterialReconstructionSettings(
        iterations=3,
        regularisation_mm_inverse=regularisation,
        gradient_mapping_tolerance=0,
        relative_gradient_mapping_tolerance=0,
    )
    inputs["device"] = "cpu"
    return inputs


def _independent_objective(inputs, fields):
    data = 0.0
    for view in inputs["views"]:
        for row in range(view.geometry.shape[0]):
            for column in range(view.geometry.shape[1]):
                paths = np.array(
                    [
                        integrate_sampled_field(
                            field.reshape(-1),
                            inputs["grid"],
                            view.geometry,
                            view.pose,
                            row,
                            column,
                            midpoint_samples=inputs["samples_per_ray"],
                        )
                        for field in fields
                    ]
                )
                transmission = np.exp(-inputs["coefficients"].astype(float).T @ paths)
                means = (view.weights.astype(float) * view.response.astype(float)) @ transmission
                counts = view.counts[:, row, column].astype(float)
                data += np.sum(means - counts + counts * np.log(counts / means))
    prior = 0.0
    beta = inputs["settings"].regularisation_mm_inverse
    for axis, spacing in zip((3, 2, 1), inputs["grid"].spacing_mm, strict=True):
        coefficient = beta * np.prod(inputs["grid"].spacing_mm) / spacing**2
        prior += 0.5 * coefficient * np.sum(np.diff(fields.astype(float), axis=axis) ** 2)
    return data + prior
