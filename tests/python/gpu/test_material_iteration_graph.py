"""Full-iteration CUDA graphs preserve accepted solver and recorder contracts."""

from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest

from dpt.contracts import ContractError, NumericalError, TrialDomainError
from dpt.material_reconstruction import MaterialReconstruction, MaterialReconstructionCheckpoint
from helpers import material_inputs as cases
from helpers import material_solver_inputs as solver_cases
from helpers.resident_penalty_inputs import MATRIX, PreparedQuadraticPenalty, quadratic_reference

pytestmark = pytest.mark.gpu

POLICIES = (
    "fixed",
    "relative_bb",
    "bb_inertial",
    "geometric_inertial",
    "bb_gll",
    "cache_off",
    "diagonal_nonnegative",
    "signal_wls",
)


def _inputs(precision: Literal["float32", "float64"], policy: str, iterations: int = 4):
    supplied = (
        solver_cases._signal_inputs(heterogeneous=True, shared=False, precision=precision)
        if policy == "signal_wls"
        else cases._inputs(precision)
    )
    options: dict[str, Any] = {
        "iterations": iterations,
        "maximum_backtracks": 12,
        "gradient_mapping_tolerance": 0.0,
        "relative_gradient_mapping_tolerance": 0.0,
    }
    if policy == "relative_bb":
        options.update(step_selection="bb", gradient_mapping_policy="relative")
        supplied["material_metric"] = (2.0, -0.3, 1.0)
        supplied["voxel_metric_scale"] = np.linspace(
            0.7, 1.3, supplied["grid"].voxels, dtype=np.float32
        ).reshape(supplied["grid"].shape)
    elif policy == "bb_inertial":
        options.update(step_selection="bb", acceleration="inertial")
        supplied["material_metric"] = (2.0, -0.3, 1.0)
    elif policy == "geometric_inertial":
        options.update(acceleration="inertial", cache_accepted_forward=False)
    elif policy == "bb_gll":
        options.update(step_selection="bb", nonmonotone_window=4)
    elif policy == "cache_off":
        options.update(cache_accepted_forward=False)
    elif policy == "diagonal_nonnegative":
        options.update(gradient_mapping_policy="relative", cache_accepted_forward=False)
        supplied["material_diagonal_metric"] = (1.4, 0.8)
        supplied["field_domain"] = "nonnegative"
    supplied["device"] = "cuda:0"
    supplied["settings"] = replace(supplied["settings"], **options)
    return supplied


def _assert_tree(actual: Any, expected: Any):
    """Compare scalar solver semantics without comparing elapsed time."""
    if isinstance(expected, dict):
        left = {k: v for k, v in actual.items() if k != "wall_seconds"}
        right = {k: v for k, v in expected.items() if k != "wall_seconds"}
        assert left.keys() == right.keys()
        for key, value in right.items():
            _assert_tree(left[key], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_tree(left, right)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=2e-8)
    else:
        assert actual == expected


def _assert_report(actual, expected, *, extra_evaluations=0):
    for key in (
        "termination",
        "field_domain",
        "observation_model",
        "accepted_steps",
        "gradient_mapping_policy",
        "gradient_mapping_metric",
        "step_selection",
        "nonmonotone_window",
        "cache_accepted_forward",
        "acceleration",
    ):
        assert actual[key] == expected[key]
    assert actual["evaluations"] == expected["evaluations"] + extra_evaluations
    for key in (
        "final_objective",
        "data_objective",
        "regularisation_objective",
        "initial_gradient_mapping",
        "final_gradient_mapping",
        "gradient_mapping_normaliser",
        "gradient_mapping_threshold",
        "mapping_step",
        "configured_mapping_step",
    ):
        np.testing.assert_allclose(actual[key], expected[key], rtol=5e-6, atol=2e-8)
    _assert_tree(actual["history"], expected["history"])
    _assert_tree(actual["rejected_numerical_trials"], expected["rejected_numerical_trials"])


def _assert_resident_state(actual, expected):
    np.testing.assert_allclose(
        actual.fractions_numpy(), expected.fractions_numpy(), rtol=2e-6, atol=5e-7
    )
    np.testing.assert_allclose(
        actual.gradient.numpy(), expected.gradient.numpy(), rtol=5e-6, atol=5e-7
    )
    for left, right in zip(actual.predictions_numpy(), expected.predictions_numpy(), strict=True):
        np.testing.assert_allclose(left, right, rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("policy", POLICIES)
def test_full_iteration_graph_matches_eager_policy_state_history_and_lifecycle(precision, policy):
    supplied = _inputs(precision, policy)
    eager = MaterialReconstruction(**supplied)
    graph = MaterialReconstruction(
        **{**supplied, "settings": replace(supplied["settings"], execution="graph")}
    )
    eager_events, graph_events = [], []
    expected = eager.solve(observe_lifecycle=eager_events.append)
    actual = graph.solve(observe_lifecycle=graph_events.append)
    assert actual["execution"] == "graph"
    assert actual["accepted_steps"] > 0
    _assert_report(actual, expected)
    _assert_resident_state(graph, eager)
    assert len(graph_events) == len(eager_events)
    for left, right in zip(graph_events, eager_events, strict=True):
        assert (left.status, left.iteration, left.accepted_steps, left.reason, left.detail) == (
            right.status,
            right.iteration,
            right.accepted_steps,
            right.reason,
            right.detail,
        )
        np.testing.assert_allclose(left.objective, right.objective, rtol=5e-6, atol=2e-8)
    assert sum(event.status == "terminal" for event in graph_events) == 1
    fields = graph.fractions_numpy()
    assert np.isfinite(fields).all() and (fields >= 0).all()
    if graph.field_domain == "fractions":
        assert (fields.astype(float).sum(axis=0) <= 1 + 2**-24).all()
    if policy != "signal_wls":
        np.testing.assert_allclose(
            actual["final_objective"], cases._independent_objective(supplied, fields), rtol=3e-6
        )


def _quadratic_inputs(
    *,
    iterations: int = 6,
    execution: Literal["eager", "graph"] = "eager",
    target: tuple[float, float] = (0.2, 0.25),
):
    supplied = solver_cases._inputs(precision="float64")
    supplied["device"] = "cuda:0"
    supplied["coefficients"] = np.zeros_like(supplied["coefficients"])
    supplied["views"] = tuple(
        replace(view, counts=np.full_like(view.counts, 10000)) for view in supplied["views"]
    )
    supplied["initial_fractions"] = np.full_like(supplied["initial_fractions"], 0.01)
    supplied["samples_per_ray"] = 8
    supplied["settings"] = replace(
        supplied["settings"],
        execution=execution,
        iterations=iterations,
        initial_step=1.0,
        maximum_step=32.0,
        maximum_backtracks=12,
        step_selection="bb",
        nonmonotone_window=4,
        gradient_mapping_tolerance=0.0,
        relative_gradient_mapping_tolerance=0.0,
    )
    supplied["penalty_operator"] = PreparedQuadraticPenalty(
        np.ones(supplied["grid"].voxels), device="cuda:0", target=target
    )
    return supplied


def test_full_iteration_graph_gll_accepts_real_ascent_against_independent_window():
    results = []
    for execution in ("eager", "graph"):
        supplied = _quadratic_inputs(execution=execution)
        solver = MaterialReconstruction(**supplied)
        initial, _ = quadratic_reference(supplied["initial_fractions"], np.ones(1), (0.2, 0.25))
        accepted_losses = [initial]

        def accepted(_iteration, row, solver=solver, accepted_losses=accepted_losses):
            independent, _ = quadratic_reference(solver.fractions_numpy(), np.ones(1), (0.2, 0.25))
            np.testing.assert_allclose(row["objective"], independent, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(
                row["line_search_reference"], max(accepted_losses[-4:]), rtol=1e-12, atol=1e-14
            )
            assert row["objective"] < row["line_search_reference"]
            accepted_losses.append(independent)

        report = solver.solve(callback=accepted)
        assert report["accepted_steps"] >= 5
        assert any(
            row["objective"] > row["objective_before"]
            for row in report["history"]
            if row["accepted"]
        )
        results.append((solver, report))
    _assert_report(results[1][1], results[0][1])
    _assert_resident_state(results[1][0], results[0][0])


@pytest.mark.parametrize("exit_kind", ["stationary", "search_failed"])
@pytest.mark.parametrize("cache", [False, True])
def test_full_iteration_graph_unaccepted_exit_preserves_incumbent_state(exit_kind, cache):
    supplied = _inputs("float64", "fixed")
    options = {"gradient_mapping_tolerance": 2.0}
    if exit_kind == "search_failed":
        options = {"initial_step": 1e20, "maximum_step": 1e20, "maximum_backtracks": 1}
    supplied["settings"] = replace(
        supplied["settings"], execution="graph", cache_accepted_forward=cache, **options
    )
    reference = MaterialReconstruction(
        **{**supplied, "settings": replace(supplied["settings"], execution="eager")}
    )
    initial_loss = reference.evaluate()
    solver = MaterialReconstruction(**supplied)
    events = []
    report = solver.solve(observe_lifecycle=events.append)
    assert report["accepted_steps"] == 0
    assert report["termination"] == (
        "projected_gradient_tolerance" if exit_kind == "stationary" else "line_search_failed"
    )
    expected_statuses = (
        ["initial", "terminal"]
        if exit_kind == "stationary"
        else ["initial", "rejected", "terminal"]
    )
    assert [event.status for event in events] == expected_statuses
    np.testing.assert_allclose(report["final_objective"], initial_loss, rtol=2e-6, atol=1e-8)
    _assert_resident_state(solver, reference)
    np.testing.assert_array_equal(solver.fractions_numpy(), supplied["initial_fractions"])


@pytest.mark.parametrize("error_type", [ValueError, NumericalError, TrialDomainError])
def test_full_iteration_graph_initial_observer_error_retains_base_evaluation(error_type):
    supplied = _inputs("float64", "bb_inertial")
    reference = MaterialReconstruction(**supplied)
    initial_loss = reference.evaluate()
    solver = MaterialReconstruction(
        **{**supplied, "settings": replace(supplied["settings"], execution="graph")}
    )
    sentinel = error_type("initial recorder failed")
    events = []

    def observer(event):
        events.append(event)
        assert event.status == "initial"
        np.testing.assert_allclose(event.objective, initial_loss, rtol=2e-6, atol=1e-8)
        _assert_resident_state(solver, reference)
        raise sentinel

    with pytest.raises(error_type) as raised:
        solver.solve(objective_id="iteration-graph-v1", observe_lifecycle=observer)
    assert raised.value is sentinel
    assert len(events) == 1
    assert solver.evaluations == 1
    _assert_resident_state(solver, reference)
    np.testing.assert_array_equal(solver.fractions_numpy(), supplied["initial_fractions"])


@pytest.mark.parametrize("site", ["accepted_observer", "legacy_callback", "save_checkpoint"])
@pytest.mark.parametrize("error_type", [ValueError, NumericalError, TrialDomainError])
def test_full_iteration_graph_recorder_error_commits_once_and_resumes(site, error_type):
    supplied = _inputs("float64", "relative_bb")
    supplied["settings"] = replace(supplied["settings"], execution="graph")
    complete = MaterialReconstruction(**supplied)
    expected = complete.solve(objective_id="iteration-graph-v1")
    solver = MaterialReconstruction(**supplied)
    sentinel = error_type("accepted recorder failed")
    order = []

    def fail_if(selected):
        if site == selected:
            checkpoint = solver.checkpoint()
            assert checkpoint.state["accepted"] == 1
            np.testing.assert_array_equal(checkpoint.fields, solver.fractions_numpy())
            raise sentinel

    def observer(event):
        order.append(event.status)
        if event.status == "accepted":
            fail_if("accepted_observer")

    def callback(iteration, row):
        assert iteration == 1 and row["accepted"]
        order.append("legacy")
        fail_if("legacy_callback")

    def save(checkpoint):
        order.append("checkpoint")
        assert checkpoint.state["accepted"] == 1
        fail_if("save_checkpoint")

    with pytest.raises(error_type) as raised:
        solver.solve(
            objective_id="iteration-graph-v1",
            observe_lifecycle=observer,
            callback=callback,
            save_checkpoint=save,
        )
    assert raised.value is sentinel
    published = [item for item in order if item != "rejected"]
    assert (
        published
        == {
            "accepted_observer": ["initial", "accepted"],
            "legacy_callback": ["initial", "accepted", "legacy", "terminal"],
            "save_checkpoint": ["initial", "accepted", "legacy", "checkpoint", "terminal"],
        }[site]
    )
    checkpoint = solver.checkpoint()
    assert checkpoint.state["accepted"] == checkpoint.state["completed_iterations"] == 1
    assert len(checkpoint.state["history"]) == 1
    assert checkpoint.state["history"][0]["accepted"]
    resumed = MaterialReconstruction(**supplied)
    actual = resumed.solve(checkpoint=checkpoint, objective_id="iteration-graph-v1")
    _assert_report(actual, expected)
    _assert_resident_state(resumed, complete)


@pytest.mark.parametrize("policy", ["relative_bb", "bb_inertial", "bb_gll", "cache_off"])
def test_full_iteration_graph_npz_resume_preserves_algorithm_trajectory(policy, tmp_path):
    supplied = _inputs("float64", policy, iterations=5)
    supplied["settings"] = replace(supplied["settings"], execution="graph")
    complete = MaterialReconstruction(**supplied)
    expected = complete.solve(objective_id="iteration-graph-v1")
    first = MaterialReconstruction(
        **{**supplied, "settings": replace(supplied["settings"], iterations=2)}
    )
    first.solve(objective_id="iteration-graph-v1")
    checkpoint = first.checkpoint()
    path = tmp_path / "accepted-material.npz"
    checkpoint.save(path)
    restored = MaterialReconstructionCheckpoint.read(path)
    resumed = MaterialReconstruction(**supplied)
    events = []
    actual = resumed.solve(
        checkpoint=restored, objective_id="iteration-graph-v1", observe_lifecycle=events.append
    )
    assert events[0].status == "resumed"
    assert events[0].iteration == events[0].accepted_steps == 2
    np.testing.assert_allclose(
        events[0].objective, checkpoint.state["accepted_loss"], rtol=2e-6, atol=1e-8
    )
    _assert_report(actual, expected, extra_evaluations=1)
    _assert_resident_state(resumed, complete)
    bad_events = []
    with pytest.raises(ContractError, match="identity"):
        MaterialReconstruction(**supplied).solve(
            checkpoint=restored,
            objective_id="changed-objective",
            observe_lifecycle=bad_events.append,
        )
    assert [event.status for event in bad_events] == ["terminal"]


def test_full_iteration_graph_relative_custom_penalty_resume_retains_normaliser():
    def supplied(iterations):
        result = _quadratic_inputs(iterations=iterations, execution="graph")
        result["settings"] = replace(
            result["settings"], gradient_mapping_policy="relative", nonmonotone_window=1
        )
        return result

    complete = MaterialReconstruction(**supplied(5))
    expected = complete.solve(objective_id="quadratic-iteration-v1")
    first = MaterialReconstruction(**supplied(2))
    first.solve(objective_id="quadratic-iteration-v1")
    checkpoint = first.checkpoint()
    resumed = MaterialReconstruction(**supplied(5))
    actual = resumed.solve(checkpoint=checkpoint, objective_id="quadratic-iteration-v1")
    assert actual["gradient_mapping_normaliser"] == checkpoint.state["mapping_normaliser"]
    _assert_report(actual, expected, extra_evaluations=1)
    _assert_resident_state(resumed, complete)


def test_full_iteration_graph_custom_penalty_certifies_independent_stationary_face():
    target = (0.8, 0.6)
    supplied = _quadratic_inputs(execution="graph", target=target)
    optimum = solver_cases._independent_triangle_qp(
        MATRIX, np.zeros(2), -MATRIX @ np.asarray(target), 1.0
    )
    assert abs(optimum.sum() - 1.0) < 2e-15
    supplied["initial_fractions"] = np.ascontiguousarray(
        optimum.reshape(2, 1, 1, 1), dtype=np.float32
    )
    supplied["settings"] = replace(
        supplied["settings"],
        gradient_mapping_policy="relative",
        nonmonotone_window=1,
        gradient_mapping_tolerance=2e-6,
    )
    solver = MaterialReconstruction(**supplied)
    initial = solver.fractions_numpy().copy()
    expected, _ = quadratic_reference(initial, np.ones(1), target)
    events = []
    report = solver.solve(observe_lifecycle=events.append)
    assert report["termination"] == "projected_gradient_tolerance"
    assert report["accepted_steps"] == 0
    assert [event.status for event in events] == ["initial", "terminal"]
    np.testing.assert_array_equal(solver.fractions_numpy(), initial)
    np.testing.assert_allclose(report["final_objective"], expected, rtol=1e-12, atol=1e-14)
