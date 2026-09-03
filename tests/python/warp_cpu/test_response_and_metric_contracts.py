"""Physical response moments and recoverable resident metric validation."""

import importlib
import math
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError, NumericalError
from dpt.material_reconstruction import MaterialReconstruction, MaterialReconstructionSignalView
from helpers import material_inputs as cases

wp: Any = importlib.import_module("warp")


def _energy_signal_inputs(precision, device):
    inputs = cases._inputs(precision, regularisation=0)
    inputs["device"] = device
    inputs["observation_model"] = "signal_wls"
    inputs["spectral_spec"] = replace(inputs["spectral_spec"], output_unit="keV")
    views = []
    for source in inputs["views"]:
        observations = np.ascontiguousarray(source.counts / np.float32(100))
        valid = np.ones(observations.shape, dtype=np.uint8)
        valid[0, 0, 0] = 0
        observations[0, 0, 0] = np.nan
        views.append(
            MaterialReconstructionSignalView(
                source.geometry,
                source.pose,
                observations,
                np.array([[2, 3], [1.5, 4]], dtype=np.float32),
                np.array([[50, 70], [40, 60]], dtype=np.float32),
                np.full(observations.shape, 0.7, dtype=np.float32),
                valid,
            )
        )
    inputs["views"] = tuple(views)
    return inputs


def _independent_energy_loss(inputs, fields):
    predictions = []
    loss = 0.0
    for view in inputs["views"]:
        expected = np.empty(view.observations.shape, dtype=np.float64)
        for row in range(view.geometry.shape[0]):
            for column in range(view.geometry.shape[1]):
                paths = np.asarray(
                    [
                        cases.integrate_sampled_field(
                            material.reshape(-1),
                            inputs["grid"],
                            view.geometry,
                            view.pose,
                            row,
                            column,
                            midpoint_samples=inputs["samples_per_ray"],
                        )
                        for material in fields
                    ]
                )
                transmission = np.exp(-inputs["coefficients"].astype(float).T @ paths)
                means = (view.weights.astype(float) * view.response.astype(float)) @ transmission
                expected[:, row, column] = means
                for channel in range(len(means)):
                    if view.valid[channel, row, column]:
                        residual = means[channel] - view.observations[channel, row, column]
                        loss += 0.5 * view.objective_weights[channel, row, column] * residual**2
        predictions.append(expected)
    return loss, predictions


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_energy_integrating_signal_response_matches_independent_loss_and_gradient(
    precision, warp_device
):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        inputs = _energy_signal_inputs(precision, warp_device)
        solver = MaterialReconstruction(**inputs)
        actual = solver.evaluate()
        fields = inputs["initial_fractions"]
        expected, predictions = _independent_energy_loss(inputs, fields)
        np.testing.assert_allclose(actual, expected, rtol=3e-6)
        for actual_prediction, reference in zip(
            solver.predictions_numpy(), predictions, strict=True
        ):
            np.testing.assert_allclose(actual_prediction, reference, rtol=3e-6)
        direction = np.random.default_rng(189).uniform(-0.1, 0.1, fields.shape).astype(np.float32)
        h = np.float32(0.01)
        high, _ = _independent_energy_loss(inputs, fields + h * direction)
        low, _ = _independent_energy_loss(inputs, fields - h * direction)
        product = float(solver.gradient.numpy().astype(float) @ direction.reshape(-1))
        np.testing.assert_allclose(product, (high - low) / (2 * h), rtol=3e-4)
        # The same response moments are incompatible with the independent-count model.
        poisson = cases._inputs(precision)
        poisson["views"] = tuple(
            replace(view, response=inputs["views"][0].response) for view in poisson["views"]
        )
        with pytest.raises(ContractError, match="Poisson response probabilities"):
            MaterialReconstruction(**poisson)


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_signal_response_moments_still_require_finite_nonnegative_values(bad):
    with cpu_test_context():
        inputs = _energy_signal_inputs("float64", "cpu")
        original = inputs["views"][0]
        invalid = original.response.copy()
        invalid[0, 0] = bad
        inputs["views"] = (replace(original, response=invalid),)
        with pytest.raises(ContractError, match="response"):
            MaterialReconstruction(**inputs)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, 1e-8])
def test_failed_resident_metric_update_preserves_previous_metric_and_delivers_status(
    bad, warp_device
):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        inputs = cases._inputs("float64")
        inputs["device"] = warp_device
        solver = MaterialReconstruction(**inputs)
        solver.set_voxel_metric_scale(np.ones(solver.grid.shape, dtype=np.float32))
        previous = solver.voxel_metric_scale.ptr
        expected = solver.evaluate()
        values = np.ones(solver.grid.voxels, dtype=np.float32)
        values[-1] = bad
        with solver.context.scope():
            candidate = wp.array(values, dtype=wp.float32, device=warp_device)
        with pytest.raises(ContractError, match="voxel metric scale"):
            solver.set_voxel_metric_scale(candidate)
        assert solver.voxel_metric_scale.ptr == previous
        assert solver.evaluate() == expected
        assert int(solver._status.numpy()[0]) == 0
        # A newly supplied metric cannot erase an earlier unchecked failure.
        solver._status.fill_(4)
        with pytest.raises(NumericalError, match="metric-projection face"):
            solver.set_voxel_metric_scale(np.ones(solver.grid.shape, dtype=np.float32))
        assert solver.voxel_metric_scale.ptr == previous
        assert solver.evaluate() == expected
