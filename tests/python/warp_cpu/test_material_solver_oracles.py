"""Portable solver regressions against rational and scalar forward oracles."""

import importlib
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import NumericalError, TrialDomainError
from dpt.kernels.material_reconstruction import max_partials
from dpt.material_preconditioning import prepare_fisher_scaling
from dpt.material_reconstruction import MaterialReconstruction, MaterialReconstructionSettings
from helpers import material_fisher_inputs as fisher_cases
from helpers import metric_oracles as metric_cases
from helpers.material_inputs import _independent_objective, _inputs, _metric_projection

wp: Any = importlib.import_module("warp")


@pytest.fixture(autouse=True)
def portable_context():
    with cpu_test_context():
        yield


@pytest.mark.parametrize(
    "metric,gradient,step",
    [
        ((1.0, 0.0, 1e6), (-2e-6, -3e-6), 1e-5),
        ((1.0, 0.0, 1e3), (-2e-6, -7e-6), 1e-8),
        ((1.0, 0.0, 1e6), (-3.5614e-9, -3.5614156e-9), 5.19),
    ],
)
def test_stiff_cap_vertex_roundoff_has_a_certified_projection(metric, gradient, step):
    fields = np.array([[1.0], [0.0]], dtype=np.float32)
    gradients = np.array(gradient, dtype=np.float32)[:, None]
    out, status = _metric_projection(fields, gradients, step, metric)
    expected = np.array(
        [float(x) for x in metric_cases.exact_qp(fields[:, 0], gradients[:, 0], step, metric)]
    )
    assert status == 0
    np.testing.assert_allclose(out[:, 0], expected, atol=2e-15, rtol=0)


@pytest.mark.parametrize("metric", [(1.0, 0.0, 1e6), (1e6, 0.0, 1.0), (2.0, 0.3, 1.0)])
def test_vertex_and_edge_projection_agree_with_exact_rational_qp(metric):
    rng = np.random.default_rng(2454)
    boundary = np.array([[1, 0], [0, 1], [0, 0], [0.25, 0.75], [0.3, 0]], dtype=np.float32)
    fields = np.repeat(boundary, 32, axis=0).T.copy()
    gradients = np.ascontiguousarray(
        rng.uniform(-1, 1, fields.shape) * 10.0 ** rng.uniform(-14, 2, fields.shape),
        dtype=np.float32,
    )
    # Include the complete failed backtracking window at the stiff endpoint.
    gradients[:, :13] = np.array([[-1], [-2]], dtype=np.float32) * (2.0 ** -np.arange(34, 47))
    out, status = _metric_projection(fields, gradients, 1.0, metric)
    assert status == 0
    expected = np.array(
        [
            [float(x) for x in metric_cases.exact_qp(fields[:, i], gradients[:, i], 1.0, metric)]
            for i in range(fields.shape[1])
        ]
    ).T
    np.testing.assert_allclose(out, expected, atol=2e-7, rtol=0)


@pytest.mark.parametrize("negative", [False, True])
def test_cpu_max_tiles_consume_every_row_and_respect_logical_count(negative):
    values = np.arange(1000, dtype=np.float64)
    if negative:
        values = -values - 1
    source = wp.array(values, dtype=wp.float64, device="cpu")
    partials = wp.empty(4, dtype=wp.float64, device="cpu")
    wp.launch_tiled(
        max_partials, dim=4, inputs=[source, 1000, partials], block_dim=256, device="cpu"
    )
    expected = np.array([values[i : i + 256].max() for i in range(0, 1000, 256)])
    np.testing.assert_array_equal(partials.numpy(), expected)
    final = wp.empty(1, dtype=wp.float64, device="cpu")
    wp.launch_tiled(max_partials, dim=1, inputs=[partials, 4, final], block_dim=256, device="cpu")
    assert final.numpy()[0] == values.max()
    wp.launch_tiled(max_partials, dim=1, inputs=[source, 3, final], block_dim=256, device="cpu")
    assert final.numpy()[0] == values[:3].max()


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_multivoxel_multiview_cpu_objective_and_directional_gradient(precision):
    inputs = _inputs(precision)
    solver = MaterialReconstruction(**inputs)
    fields = inputs["initial_fractions"]
    loss = solver.evaluate()
    np.testing.assert_allclose(loss, _independent_objective(inputs, fields), rtol=3e-6)
    direction = np.random.default_rng(739).uniform(-0.1, 0.1, fields.shape).astype(np.float32)
    expected = float(solver.gradient.numpy().astype(float) @ direction.reshape(-1))
    h = np.float32(0.01)
    high = _independent_objective(inputs, fields + h * direction)
    low = _independent_objective(inputs, fields - h * direction)
    np.testing.assert_allclose(expected, (high - low) / (2 * h), rtol=2e-4, atol=0.002)


@pytest.mark.parametrize("trial", [False, True])
def test_delivered_failed_evaluation_can_recover_after_repair(trial):
    inputs = _inputs()
    solver = MaterialReconstruction(**inputs)
    fields = solver.trial if trial else solver.fractions
    wp.copy(fields, solver.fractions)
    original = fields.numpy().copy()
    reference = solver.evaluate(gradient=False, trial=trial)
    fields.fill_(2.0)
    with pytest.raises(TrialDomainError):
        solver.evaluate(gradient=False, trial=trial)
    fields.assign(original)
    assert solver.evaluate(gradient=False, trial=trial) == reference


@pytest.mark.parametrize("view_index", [0, 1])
@pytest.mark.parametrize("scalar_projection", [False, True])
def test_unchecked_projection_failure_is_delivered_before_reset(view_index, scalar_projection):
    solver = MaterialReconstruction(**_inputs())
    reference = solver.evaluate()
    projection = solver.views[view_index].accepted
    status = projection.projection._status if scalar_projection else projection.status
    status.fill_(8)
    with pytest.raises(NumericalError):
        solver.evaluate()
    assert solver.evaluate() == reference


@pytest.mark.parametrize("step_selection", ["geometric", "bb"])
@pytest.mark.parametrize("acceleration", ["none", "inertial"])
@pytest.mark.parametrize("metric", [None, (2.0, -0.3, 1.0)])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_multivoxel_solve_preserves_domain_and_decreases_actual_objective(
    step_selection, acceleration, metric, precision
):
    inputs = _inputs(precision)
    inputs["settings"] = replace(
        inputs["settings"], step_selection=step_selection, acceleration=acceleration
    )
    solver = MaterialReconstruction(**inputs, material_metric=metric)
    initial = _independent_objective(inputs, inputs["initial_fractions"])
    report = solver.solve()
    assert report["accepted_steps"] > 0
    assert report["final_objective"] < initial
    fields = solver.fractions_numpy()
    assert (fields >= 0).all()
    assert (fields.astype(float).sum(axis=0) <= 1 + 2**-24).all()
    assert all(
        row["objective"] < row["objective_before"] for row in report["history"] if row["accepted"]
    )
    np.testing.assert_allclose(
        report["final_objective"], _independent_objective(inputs, fields), rtol=3e-6
    )


def test_tolerance_exit_reuses_current_value_gradient_and_predictions():
    inputs = _inputs()
    inputs["settings"] = replace(inputs["settings"], gradient_mapping_tolerance=2)
    solver = MaterialReconstruction(**inputs)
    report = solver.solve()
    assert report["termination"] == "projected_gradient_tolerance"
    assert report["accepted_steps"] == 0
    assert report["evaluations"] == 1
    assert solver._prediction_state == "accepted"


def test_failed_search_restores_predictions_without_repeating_the_gradient():
    inputs = _inputs()
    inputs["settings"] = replace(
        inputs["settings"], initial_step=1e20, maximum_step=1e20, maximum_backtracks=1
    )
    solver = MaterialReconstruction(**inputs)
    reference_loss = solver.evaluate()
    reference_gradient = solver.gradient.numpy().copy()
    reference_predictions = solver.predictions_numpy()
    calls = []
    evaluate = solver.evaluate

    def counted_evaluate(*, gradient=True, trial=False):
        calls.append((gradient, trial))
        return evaluate(gradient=gradient, trial=trial)

    solver.evaluate = counted_evaluate
    report = solver.solve()
    assert report["termination"] == "line_search_failed"
    assert report["accepted_steps"] == 0
    assert calls == [(True, False), (False, True), (False, False)]
    assert report["final_objective"] == reference_loss
    np.testing.assert_array_equal(solver.gradient.numpy(), reference_gradient)
    for expected, actual in zip(reference_predictions, solver.predictions_numpy(), strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("metric", [None, (2.0, -0.3, 1.0)])
def test_relative_stationarity_is_invariant_to_positive_objective_scale(metric):
    inputs = _inputs()
    solver = MaterialReconstruction(**inputs, material_metric=metric)
    solver.evaluate()
    gradient = solver.gradient.numpy().copy()
    diagnostics = []
    for factor in (1e-6, 1.0, 1e6):
        solver.gradient.assign(np.ascontiguousarray(factor * gradient, dtype=np.float32))
        scale = solver._initial_gradient_scale()
        mapping, _ = solver.projected_trial(1.0 / scale, use_metric=True)
        diagnostics.append(mapping / scale)
    np.testing.assert_allclose(diagnostics, diagnostics[1], rtol=2e-7)


def test_relative_policy_reports_dimensionless_diagnostic_without_changing_default():
    assert MaterialReconstructionSettings().gradient_mapping_policy == "fixed"
    inputs = _inputs()
    inputs["settings"] = replace(inputs["settings"], gradient_mapping_policy="relative")
    solver = MaterialReconstruction(**inputs)
    report = solver.solve()
    assert report["gradient_mapping_policy"] == "relative"
    assert report["gradient_mapping_metric"] == "update"
    assert (
        report["mapping_step"]
        == inputs["settings"].mapping_step / report["gradient_mapping_normaliser"]
    )
    assert (
        report["final_relative_gradient_mapping"]
        == report["final_gradient_mapping"] / report["gradient_mapping_normaliser"]
    )


@pytest.mark.parametrize("metric", [None, (2.0, -0.3, 1.0)])
def test_relative_policy_matches_under_actual_count_objective_rescaling(metric):
    reports = []
    for factor in (0.01, 1.0, 1e3):
        inputs = _inputs(regularisation=0)
        inputs["settings"] = replace(
            inputs["settings"], gradient_mapping_policy="relative", gradient_mapping_tolerance=2
        )
        inputs["views"] = tuple(
            replace(
                view,
                weights=np.ascontiguousarray(factor * view.weights, dtype=np.float32),
                counts=np.ascontiguousarray(factor * view.counts, dtype=np.float32),
            )
            for view in inputs["views"]
        )
        solver = MaterialReconstruction(**inputs, material_metric=metric)
        reports.append(solver.solve())
    assert all(report["termination"] == "projected_gradient_tolerance" for report in reports)
    np.testing.assert_allclose(
        [report["final_relative_gradient_mapping"] for report in reports],
        reports[1]["final_relative_gradient_mapping"],
        rtol=4e-7,
    )


def test_cpu_voxel_scale_bounds_include_tail_entries():
    solver = MaterialReconstruction(**_inputs())
    values = np.ones(solver.grid.shape, dtype=np.float32)
    values.reshape(-1)[-1] = 7
    values.reshape(-1)[-2] = 0.5
    solver.set_voxel_metric_scale(values)
    assert solver._voxel_metric_summary == {
        "minimum": 0.5,
        "maximum": 7.0,
        "maximum_over_minimum": 14.0,
    }


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("beta", [0.0, 0.07])
def test_cpu_fisher_scaling_matches_dense_majoriser_and_preserves_fields(precision, beta):
    inputs = fisher_cases._inputs(beta=beta, precision=precision)
    solver = MaterialReconstruction(**inputs, device="cpu")
    loss = solver.evaluate()
    fields, gradient = solver.fractions_numpy().copy(), solver.gradient.numpy().copy()
    expected, fisher, metric = fisher_cases._reference(inputs)
    result = prepare_fisher_scaling(solver)
    raw = np.maximum(expected, expected.max() * 1e-6)
    np.testing.assert_allclose(result.scale.numpy(), raw / raw.mean(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(result.diagnostics["normaliser"], raw.mean(), rtol=2e-5)
    assert np.linalg.eigvalsh(np.kron(metric, np.diag(raw)) - fisher).min() > -1e-9
    np.testing.assert_array_equal(solver.fractions_numpy(), fields)
    np.testing.assert_array_equal(solver.gradient.numpy(), gradient)
    assert solver.evaluate() == loss
