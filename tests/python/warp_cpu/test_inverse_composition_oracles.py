"""Independent full-composition derivative and voxel-truth inverse regressions.

Observation synthesis and finite differences use scalar sampled-field references
and NumPy, never the evaluator or reconstruction driver's forward buffers.
The same production kernels run on CPU and, with --run-gpu, actual CUDA.
"""

import importlib
from contextlib import nullcontext
from typing import Any, Literal

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.geometry import RigidTransform
from dpt.material_reconstruction import (
    MaterialReconstruction,
    MaterialReconstructionSettings,
    MaterialReconstructionView,
)
from dpt.spectral import SpectralSpec
from dpt.volumes import GridSpec
from helpers.inverse_inputs import (
    PROVENANCE,
    _reference_paths,
    _spectral_fixture,
    _tomography_geometries,
)

wp: Any = importlib.import_module("warp")


def test_full_spectral_pose_and_nuisance_gradient_against_independent_numpy_oracle(
    warp_device: str,
) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        evaluator, reference_loss = _spectral_fixture(warp_device)
        parameters = np.asarray((0.31, -0.23, 0.19, 0.29, -0.26, 0.22, 0.15, -0.2, -0.14, 0.23))
        evaluation = evaluator(tuple(parameters))
        np.testing.assert_allclose(evaluation.loss, reference_loss(parameters), rtol=1e-5)
        gradient = np.asarray(evaluation.gradient)
        # Every coordinate is checked separately: no mixed-direction cancellation
        # can conceal a missing axis, wrong chart scale, sign, or shared-view sum.
        directions = [*np.eye(evaluator.dimension)]
        directions += [
            np.asarray((0.3, -0.2, 0.4, -0.5, 0.1, 0.6, 0, 0, 0, 0)),
            np.asarray((0.2, -0.3, 0.1, 0.4, -0.5, 0.6, -0.2, 0.3, 0.4, -0.1)),
        ]
        h = 2e-4
        numerical = []
        for direction in directions:
            high = reference_loss(parameters + h * direction)
            low = reference_loss(parameters - h * direction)
            derivative = (high - low) / (2 * h)
            numerical.append(derivative)
            np.testing.assert_allclose(gradient @ direction, derivative, rtol=4e-4, atol=3e-4)
        assert np.min(np.abs(numerical[: evaluator.dimension])) > 0.05


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_multivoxel_material_solver_recovers_independently_synthesised_truth(
    warp_device: str,
    precision: Literal["float32", "float64"],
) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        grid = GridSpec((2, 2, 2), (1.0, 1.0, 1.0), (-0.5, -0.5, -0.5))
        geometries = _tomography_geometries()
        samples = 48
        truth = np.asarray(
            [
                [[[0.13, 0.24], [0.31, 0.18]], [[0.28, 0.16], [0.21, 0.36]]],
                [[[0.34, 0.16], [0.12, 0.29]], [[0.19, 0.32], [0.27, 0.11]]],
            ],
            dtype=np.float32,
        )
        initial = np.full_like(truth, 0.07)
        coefficients = np.asarray([[0.25, 0.08], [0.06, 0.3]], dtype=np.float32)
        # Distinct finite-width spectra require both the spatial inverse and the
        # nonlinear two-material spectral inverse; neither channel is monochromatic.
        weights = np.asarray([[80000, 10000], [12000, 75000]], dtype=np.float32)
        response = np.asarray([[0.9, 0.8], [0.85, 0.95]], dtype=np.float32)
        spec = SpectralSpec(
            2,
            2,
            (PROVENANCE, PROVENANCE),
            PROVENANCE,
            PROVENANCE,
            "counts",
            "Independent analytic two-material polychromatic expected-count observations",
            precision=precision,
        )
        views = []
        projector = []
        for geometry in geometries:
            paths = _reference_paths(truth, grid, geometry, RigidTransform(), samples)
            means = (weights.astype(float) * response.astype(float)) @ np.exp(
                -(coefficients.astype(float).T @ paths)
            )
            # The Poisson driver explicitly requires integral observations. Round
            # independently synthesised means once; high fluence bounds this known
            # quantisation perturbation well below the voxel-recovery tolerance.
            counts = np.ascontiguousarray(
                np.rint(means).reshape(2, *geometry.shape), dtype=np.float32
            )
            views.append(
                MaterialReconstructionView(geometry, RigidTransform(), counts, weights, response)
            )
            basis = np.eye(grid.voxels, dtype=np.float32).reshape(grid.voxels, *grid.shape)
            projector.append(_reference_paths(basis, grid, geometry, RigidTransform(), samples).T)
        singular_values = np.linalg.svd(np.concatenate(projector), compute_uv=False)
        # Certify that voxel truth, rather than merely one equivalent projection,
        # is identifiable in this fixture and not dominated by an ill-conditioned nullspace.
        assert singular_values[-1] > 0.1
        assert singular_values[0] / singular_values[-1] < 10
        solver = MaterialReconstruction(
            grid=grid,
            views=tuple(views),
            coefficients=coefficients,
            spectral_spec=spec,
            initial_fractions=initial,
            samples_per_ray=samples,
            device=warp_device,
            settings=MaterialReconstructionSettings(
                iterations=180,
                initial_step=0.001,
                maximum_step=0.01,
                mapping_step=1e-4,
                step_selection="bb",
                gradient_mapping_tolerance=1e-3,
                relative_gradient_mapping_tolerance=1e-6,
            ),
        )
        report = solver.solve()
        recovered = solver.fractions_numpy()
        initial_rms = np.sqrt(np.mean((initial.astype(float) - truth) ** 2))
        recovered_rms = np.sqrt(np.mean((recovered.astype(float) - truth) ** 2))
        assert report["accepted_steps"] > 0
        assert recovered_rms < initial_rms / 100
        np.testing.assert_allclose(recovered, truth, rtol=0, atol=5e-4)
        assert (recovered >= 0).all()
        assert (recovered.astype(float).sum(axis=0) <= 1 + 2**-24).all()
        assert report["final_objective"] < 0.002
        independent_loss = 0.0
        for view, matrix, prediction in zip(
            views, projector, solver.predictions_numpy(), strict=True
        ):
            np.testing.assert_allclose(prediction, view.counts, rtol=0, atol=0.8)
            final_paths = recovered.reshape(2, -1).astype(float) @ matrix.T
            independent_means = (weights.astype(float) * response.astype(float)) @ np.exp(
                -(coefficients.astype(float).T @ final_paths)
            )
            expected = independent_means.reshape(view.counts.shape)
            np.testing.assert_allclose(prediction, expected, rtol=2e-7, atol=0.002)
            counts = view.counts.astype(float)
            independent_loss += np.sum(expected - counts + counts * np.log(counts / expected))
        # Predictions are stored in the selected production precision, whereas
        # the independent reference retains FP64 paths and means throughout.
        np.testing.assert_allclose(
            report["final_objective"], independent_loss, rtol=0.02, atol=2e-6
        )
