"""Sequential spectral views share consumed scratch and retain public predictions."""

from contextlib import nullcontext

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from helpers import inverse_inputs as fixtures


def test_spectral_pose_views_share_capacity_scratch_and_keep_all_predictions(warp_device: str):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        evaluator, _ = fixtures._spectral_fixture(warp_device)
        views = evaluator._prepared
        assert len(views) == 3
        for name in ("mean", "mean_seed", "image_seed"):
            assert len({getattr(view, name).ptr for view in views}) == 1
        for name in ("paths", "cotangent_paths"):
            assert len({getattr(view.material, name).ptr for view in views}) == 1
        assert views[0].spectral is views[1].spectral is views[2].spectral
        assert views[0].detector is views[1].detector is views[2].detector
        assert views[0].objective is views[2].objective
        assert views[0].blur is views[2].blur
        assert len({view.prediction.ptr for view in views}) == len(views)
        parameters = (0.31, -0.23, 0.19, 0.29, -0.26, 0.22, 0.15, -0.2, -0.14, 0.23)
        expected = evaluator(parameters)
        predictions = [array.numpy().copy() for array in evaluator.predicted_images]
        assert not np.array_equal(predictions[0], predictions[1])
        assert evaluator(parameters) == expected
        for current, previous in zip(evaluator.predicted_images, predictions, strict=True):
            np.testing.assert_array_equal(current.numpy(), previous)
        assert len(evaluator._workspaces()) == 3 + 1 + 1 + 2 + 1


@pytest.mark.parametrize("spatial_axes", [False, True])
def test_spectral_pose_accepts_public_shaped_basis_without_copy_or_input_mutation(
    spatial_axes, warp_device
):
    from dataclasses import replace

    from dpt.materials import MaterialBasis, MaterialTable
    from dpt.spectral_recovery import SpectralPoseEvaluator

    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        reference, _ = fixtures._spectral_fixture(warp_device)
        problem = reference.problem
        spectral = problem.views[0].spectral
        energies = tuple(10.0 * (index + 1) for index in range(spectral.energies))
        values = (
            problem.views[0].coefficients.numpy().reshape(spectral.materials, spectral.energies)
        )
        basis = MaterialBasis(
            tuple(
                MaterialTable(f"basis-{index}", energies, tuple(row), "mm^-1")
                for index, row in enumerate(values)
            ),
            "same fixed-density test basis, explicit point sampling",
        )
        coefficients = basis.to_device(energies, reference.context, edge_side="above")
        assert coefficients.shape == (spectral.materials, spectral.energies)
        fields_shape = (
            (spectral.materials, *problem.grid.shape)
            if spatial_axes
            else (spectral.materials, problem.grid.voxels)
        )
        fields = problem.fields.reshape(fields_shape)
        views = tuple(
            replace(
                view,
                coefficients=coefficients,
                weights=view.weights.reshape(
                    (spectral.energies, *view.geometry.shape)
                    if spatial_axes
                    else (spectral.energies, view.geometry.pixels)
                ),
                response=view.response.reshape(
                    (spectral.energies, *view.geometry.shape)
                    if spatial_axes
                    else (spectral.energies, view.geometry.pixels)
                ),
                observation=view.observation.reshape(view.geometry.shape),
                objective_weights=view.objective_weights.reshape(view.geometry.shape),
            )
            for view in problem.views
        )
        shaped = SpectralPoseEvaluator(
            replace(problem, fields=fields, views=views), reference.chart, device=warp_device
        )
        parameters = (0.31, -0.23, 0.19, 0.29, -0.26, 0.22, 0.15, -0.2, -0.14, 0.23)
        expected, actual = reference(parameters), shaped(parameters)
        np.testing.assert_allclose(actual.loss, expected.loss, rtol=1e-7)
        np.testing.assert_allclose(actual.gradient, expected.gradient, rtol=1e-6, atol=1e-8)
        for view, prepared in zip(views, shaped._prepared, strict=True):
            assert prepared.view is view
            assert prepared.material.fields.ptr == fields.ptr
            for name in ("coefficients", "weights", "response", "observation", "objective_weights"):
                assert getattr(prepared, name).ptr == getattr(view, name).ptr
                assert getattr(prepared, name).ndim == 1
        assert shaped.problem.fields is fields
        assert views[0].coefficients is coefficients
        assert views[0].weights.ndim == (3 if spatial_axes else 2)
