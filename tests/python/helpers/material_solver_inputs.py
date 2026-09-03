"""Shared fixtures and independent oracles: material solver inputs."""

import importlib
import math
from dataclasses import replace
from itertools import combinations
from typing import Any, Literal

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_reconstruction import (
    MaterialReconstructionSettings,
    MaterialReconstructionSignalView,
    MaterialReconstructionView,
)
from dpt.materials import Provenance
from dpt.spectral import SpectralSpec
from dpt.volumes import GridSpec

np: Any = importlib.import_module("numpy")


wp: Any = importlib.import_module("warp")


def _signal_inputs(
    *,
    heterogeneous: bool = False,
    shared: bool = False,
    precision: Literal["float32", "float64"] = "float32",
) -> dict[str, Any]:
    grid = GridSpec((2, 3, 4), (1.1, 0.9, 1.3), (-1.65, -0.9, -0.65)) if heterogeneous else None
    inputs = _inputs(grid=grid, precision=precision)
    inputs["spectral_spec"] = replace(
        inputs["spectral_spec"], shared_weights=shared, output_unit="stored mean intensity"
    )
    inputs["field_domain"] = "nonnegative"
    inputs["observation_model"] = "signal_wls"
    rng = np.random.default_rng(2453)
    inputs["initial_fractions"] = np.ascontiguousarray(
        rng.uniform(0.7, 1.4, size=inputs["initial_fractions"].shape), dtype=np.float32
    )
    geometry = DetectorGeometry(
        (-20.0, 0.0, 0.0), (20.0, -0.2, -0.15), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.2, 0.3), (2, 3)
    )
    weights = np.array([[70.0, 15.0], [10.0, 90.0]], dtype=np.float32)
    if not shared:
        weights = np.ascontiguousarray(
            weights[:, :, None, None] * rng.uniform(0.7, 1.3, (2, 2, 2, 3)), dtype=np.float32
        )
    response = np.ones((2, 2), dtype=np.float32)
    observations = np.ascontiguousarray(rng.uniform(35, 70, (2, 2, 3)), dtype=np.float32)
    objective_weights = np.ascontiguousarray(
        rng.uniform(0.2, 1.5, observations.shape), dtype=np.float32
    )
    valid = np.ones(observations.shape, dtype=np.uint8)
    valid[0, 0, 1] = 0
    valid[1, 1, 2] = 0
    observations[0, 0, 1] = np.nan
    observations[1, 1, 2] = np.inf
    first = MaterialReconstructionSignalView(
        geometry, RigidTransform(), observations, weights, response, objective_weights, valid
    )
    if heterogeneous:
        angle = 0.17
        pose = RigidTransform(
            (
                math.cos(angle),
                -math.sin(angle),
                0.0,
                math.sin(angle),
                math.cos(angle),
                0.0,
                0.0,
                0.0,
                1.0,
            ),
            (0.1, -0.05, 0.07),
        )
        second = replace(
            first,
            observations=observations.copy() + np.float32(0.2),
            pose=pose,
            geometry=DetectorGeometry(
                (0.1, -20.0, 0.2),
                (-0.1, 20.0, -0.2),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, -1.0),
                (0.4, 0.3),
                (2, 3),
            ),
        )
        inputs["views"] = (first, second)
    else:
        inputs["views"] = (first,)
    return inputs


def _inputs(
    *,
    grid: GridSpec | None = None,
    channels: int = 2,
    precision: Literal["float32", "float64"] = "float32",
) -> dict[str, Any]:
    """Declared numerical coefficients, not physical water/bone data."""
    grid = grid or GridSpec((1, 1, 1), (2.0, 1.0, 1.0))
    provenance = Provenance(
        source="Analytic one-voxel Beer-Lambert formula in this test",
        sha256="0" * 64,
        rights="Original test fixture, Apache-2.0",
        description="Numerical coefficient fixture; no anatomical or physical-material claim",
    )
    spec = SpectralSpec(
        materials=2,
        energies=2,
        coefficients_provenance=(provenance, provenance),
        spectrum_provenance=provenance,
        response_provenance=provenance,
        output_unit="counts",
        input_description="Two independent monochromatic count exposures for an analytic test",
        precision=precision,
    )
    geometry = DetectorGeometry(
        (-20.0, 0.0, 0.0),
        (20.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0),
        (1, 1),
    )
    coefficients = np.array([[0.11, 0.03], [0.02, 0.13]], dtype=np.float32)
    weights = np.ascontiguousarray(10000 * np.eye(2, dtype=np.float32)[:channels])
    response = np.ones((channels, 2), dtype=np.float32)
    counts = np.ascontiguousarray(np.array([[[9500]], [[9100]]], dtype=np.float32)[:channels])
    view = MaterialReconstructionView(geometry, RigidTransform(), counts, weights, response)
    return {
        "grid": grid,
        "views": (view,),
        "coefficients": coefficients,
        "spectral_spec": spec,
        "initial_fractions": np.full((2, *grid.shape), 0.1, dtype=np.float32),
        "settings": MaterialReconstructionSettings(),
        "samples_per_ray": 64,
    }


def _independent_triangle_qp(
    matrix: Any, fields: Any, gradient: Any, step: float, *, nonnegative: bool = False
) -> Any:
    """Enumerate generic inequality active sets via augmented linear systems.

    This reference does not use the production edge formula or its KKT branch
    implementation. Moderate-range comparisons use full FP64 QP objectives;
    high-common-gradient cancellation has its own independently derived test.
    """
    constraints = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
    bounds = np.array([0.0, 0.0, 1.0])
    if nonnegative:
        constraints, bounds = constraints[:2], bounds[:2]
    right = matrix @ fields - step * gradient
    solutions: list[tuple[float, Any]] = []
    for count in range(3):
        for active in combinations(range(len(bounds)), count):
            if count:
                selected = constraints[list(active)]
                block = np.block([[matrix, selected.T], [selected, np.zeros((count, count))]])
                solution = np.linalg.solve(block, np.concatenate([right, bounds[list(active)]]))
            else:
                solution = np.linalg.solve(matrix, right)
            point = solution[:2]
            if (constraints @ point > bounds + 1e-10).any():
                continue
            if count and (solution[2:] < -1e-10).any():
                continue
            displacement = point - fields
            score = float(
                0.5 * displacement @ matrix @ displacement + step * gradient @ displacement
            )
            solutions.append((score, point))
    assert solutions
    return min(solutions, key=lambda value: value[0])[1]
