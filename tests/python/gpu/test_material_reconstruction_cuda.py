"""Independent small-field references for the real CUDA material solver."""

import importlib
import math
from dataclasses import replace
from itertools import combinations
from typing import Any, Literal

import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_reconstruction import (
    MaterialReconstruction,
    MaterialReconstructionSettings,
    MaterialReconstructionSignalView,
)
from dpt.volumes import GridSpec
from helpers.material_solver_inputs import _independent_triangle_qp, _inputs, _signal_inputs

np: Any = importlib.import_module("numpy")
wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_signal_wls_spatial_spectra_match_independent_homogeneous_paths(
    shared: bool,
    precision: Literal["float32", "float64"],
) -> None:
    inputs = _signal_inputs(shared=shared, precision=precision)
    solver = MaterialReconstruction(**inputs)
    loss = solver.evaluate()
    view = inputs["views"][0]
    fields = inputs["initial_fractions"].reshape(2).astype(np.float64)
    coefficients = inputs["coefficients"].astype(np.float64)
    expected_loss = 0.0
    expected_gradient = np.zeros(2, dtype=np.float64)
    expected_means = np.empty((2, 2, 3))
    for row in range(2):
        for col in range(3):
            y, z = -0.2 + col * 0.2, -0.15 + row * 0.3
            length = 2.0 * math.sqrt(40.0**2 + y * y + z * z) / 40.0
            transmission = np.exp(-length * (coefficients.T @ fields))
            for channel in range(2):
                weights = view.weights[channel] if shared else view.weights[channel, :, row, col]
                per_energy = weights.astype(np.float64) * transmission
                mean = float(per_energy.sum())
                expected_means[channel, row, col] = mean
                if view.valid[channel, row, col]:
                    residual = mean - float(view.observations[channel, row, col])
                    scale = float(view.objective_weights[channel, row, col])
                    expected_loss += 0.5 * scale * residual**2
                    expected_gradient -= scale * residual * length * (coefficients @ per_energy)
    np.testing.assert_allclose(solver.predictions_numpy()[0], expected_means, rtol=2e-7)
    np.testing.assert_allclose(loss, expected_loss, rtol=2e-6)
    np.testing.assert_allclose(solver.gradient.numpy(), expected_gradient, rtol=3e-6)
    # Fixed inputs are private uploads, including the original invalid targets.
    view.observations.fill(0)
    view.valid.fill(1)
    view.weights.fill(0)
    view.objective_weights.fill(999)
    assert solver.evaluate() == loss


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_signal_wls_heterogeneous_multiview_gradient_and_fixed_buffer_sharing(
    precision: Literal["float32", "float64"],
) -> None:
    inputs = _signal_inputs(heterogeneous=True, precision=precision)
    initial = inputs["initial_fractions"]
    solver = MaterialReconstruction(**inputs)
    loss = solver.evaluate()
    gradient = solver.gradient.numpy().copy()
    independent_loss = 0.0
    independent_gradient = np.zeros(initial.size, dtype=np.float64)
    for view in inputs["views"]:
        view_inputs: dict[str, Any] = {**inputs, "views": (view,)}
        single = MaterialReconstruction(**view_inputs)
        independent_loss += single.evaluate()
        independent_gradient += single.gradient.numpy()
    np.testing.assert_allclose(loss, independent_loss, rtol=2e-12)
    np.testing.assert_allclose(gradient, independent_gradient, rtol=3e-5, atol=3e-5)
    for channel in range(2):
        for name in ("weights", "response"):
            assert (
                solver.views[0].channels[channel].__getattribute__(name).ptr
                == solver.views[1].channels[channel].__getattribute__(name).ptr
            )
    direction = np.ascontiguousarray(
        np.random.default_rng(16).uniform(-0.2, 0.2, initial.shape), dtype=np.float32
    )
    exact = float(gradient.astype(np.float64) @ direction.reshape(-1))
    errors: list[float] = []
    for h in (0.02, 0.01, 0.005):
        solver.set_fractions(initial + np.float32(h) * direction)
        high = solver.evaluate(gradient=False)
        solver.set_fractions(initial - np.float32(h) * direction)
        low = solver.evaluate(gradient=False)
        errors.append(abs((high - low) / (2 * h) - exact))
    assert min(errors) < max(0.003, abs(exact) * 1e-3)


@pytest.mark.parametrize(
    "metric",
    [
        (1.0, 0.0, 1.0),
        (2.0, -0.8, 1.0),
        (0.21433530073918286, 0.6050145535007093, 1.7856646992608172),
    ],
)
def test_nonnegative_metric_projection_against_independent_orthant_qp(
    metric: tuple[float, float, float],
) -> None:
    inputs = _inputs()
    inputs["initial_fractions"] = np.array([1.2, 0.3], dtype=np.float32).reshape(2, 1, 1, 1)
    solver = MaterialReconstruction(**inputs, field_domain="nonnegative", material_metric=metric)
    matrix = np.array([[metric[0], metric[1]], [metric[1], metric[2]]], dtype=np.float64)
    fields = inputs["initial_fractions"].reshape(2).astype(np.float64)
    constraints = -np.eye(2)
    targets = [(-2.0, -1.0), (3.0, -1.0), (-1.0, 4.0), (2.0, 3.0), (0.0, 0.0)]
    targets += list(np.random.default_rng(773).uniform(-2, 4, (15, 2)))
    for target in targets:
        gradient = np.ascontiguousarray(matrix @ (fields - np.array(target)), dtype=np.float32)
        with solver.context.scope():
            solver.gradient.assign(gradient)
        solver.projected_trial(1.0)
        solutions: list[tuple[float, Any]] = []
        for count in range(3):
            for active in combinations(range(2), count):
                selected = constraints[list(active)]
                block = np.block([[matrix, selected.T], [selected, np.zeros((count, count))]])
                solution = np.linalg.solve(
                    block,
                    np.concatenate(
                        [matrix @ fields - gradient.astype(np.float64), np.zeros(count)]
                    ),
                )
                point = solution[:2]
                if (point < -1e-10).any() or (count and (solution[2:] < -1e-10).any()):
                    continue
                delta = point - fields
                solutions.append((float(0.5 * delta @ matrix @ delta + gradient @ delta), point))
        assert solutions
        expected = min(solutions, key=lambda item: item[0])[1]
        np.testing.assert_allclose(solver.trial.numpy(), expected, rtol=2e-6, atol=2e-7)
    # The default stationarity mapping still uses the actual nonnegative domain.
    with solver.context.scope():
        solver.gradient.assign(np.array([-1.0, -2.0], dtype=np.float32))
    solver.projected_trial(1.0, use_metric=False)
    np.testing.assert_allclose(solver.trial.numpy(), fields + np.array([1.0, 2.0]), rtol=1e-7)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_nonnegative_wls_solve_recovers_a_solution_outside_the_fraction_simplex(
    precision: Literal["float32", "float64"],
) -> None:
    inputs = _inputs(precision=precision)
    truth = np.array([1.2, 0.7], dtype=np.float64)
    means = np.ascontiguousarray(
        100 * np.exp(-2 * (inputs["coefficients"].astype(float).T @ truth)), dtype=np.float32
    ).reshape(2, 1, 1)
    old = inputs["views"][0]
    inputs["views"] = (
        MaterialReconstructionSignalView(
            old.geometry,
            old.pose,
            means,
            np.ascontiguousarray(100 * np.eye(2), dtype=np.float32),
            old.response,
            np.ones_like(means),
            np.ones(means.shape, dtype=np.uint8),
        ),
    )
    inputs["spectral_spec"] = replace(inputs["spectral_spec"], output_unit="stored signal")
    inputs["settings"] = MaterialReconstructionSettings(
        iterations=100, initial_step=0.01, mapping_step=0.001
    )
    solver = MaterialReconstruction(
        **inputs, field_domain="nonnegative", observation_model="signal_wls"
    )
    report = solver.solve()
    assert report["observation_model"] == "signal_wls"
    assert report["field_domain"] == "nonnegative"
    np.testing.assert_allclose(solver.fractions_numpy().reshape(2), truth, atol=2e-4)
    assert report["final_objective"] < 1e-5


@pytest.mark.parametrize("case", ["empty", "invalid_target", "invalid_mask", "default_domain"])
def test_signal_input_contract_rejects_invalid_fit_before_cuda(case: str) -> None:
    inputs = _signal_inputs()
    view = inputs["views"][0]
    if case == "empty":
        view.valid.fill(0)
    elif case == "invalid_target":
        view.valid.fill(1)
    elif case == "invalid_mask":
        view.valid.fill(2)
    else:
        inputs["field_domain"] = "fractions"
    with pytest.raises(ContractError):
        MaterialReconstruction(**inputs)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_multichannel_loss_and_gradient_against_independent_beer_lambert(
    precision: Literal["float32", "float64"],
) -> None:
    inputs = _inputs(precision=precision)
    solver = MaterialReconstruction(**inputs)
    loss = solver.evaluate()
    coefficients = inputs["coefficients"].astype(np.float64)
    fractions = inputs["initial_fractions"].reshape(2).astype(np.float64)
    observed = inputs["views"][0].counts.reshape(2).astype(np.float64)
    # Each channel is a distinct monoenergetic exposure through a 2 mm voxel.
    means = 10000 * np.exp(-2.0 * (coefficients.T @ fractions))
    expected_loss = np.sum(means - observed + observed * np.log(observed / means))
    expected_gradient = 2.0 * coefficients @ (observed - means)
    np.testing.assert_allclose(solver.predictions_numpy()[0].reshape(2), means, rtol=2e-7)
    np.testing.assert_allclose(loss, expected_loss, rtol=5e-6)
    np.testing.assert_allclose(solver.gradient.numpy(), expected_gradient, rtol=3e-6)
    # A separate one-channel solve verifies that the second contribution is
    # not simply overwriting the first path cotangent.
    first = MaterialReconstruction(**_inputs(channels=1, precision=precision))
    first.evaluate()
    np.testing.assert_allclose(
        solver.gradient.numpy() - first.gradient.numpy(),
        2.0 * coefficients[:, 1] * (observed[1] - means[1]),
        rtol=4e-6,
    )


def test_composed_directional_derivative() -> None:
    inputs = _inputs()
    solver = MaterialReconstruction(**inputs)
    solver.evaluate()
    direction = np.array([0.3, -0.2], dtype=np.float32).reshape(2, 1, 1, 1)
    derivative = float(solver.gradient.numpy().astype(np.float64) @ direction.reshape(2))
    errors: list[float] = []
    for step in (0.02, 0.01, 0.005):
        solver.set_fractions(inputs["initial_fractions"] + np.float32(step) * direction)
        high = solver.evaluate(gradient=False)
        solver.set_fractions(inputs["initial_fractions"] - np.float32(step) * direction)
        low = solver.evaluate(gradient=False)
        errors.append(abs((high - low) / (2 * step) - derivative))
    assert min(errors) < 2.0e-3


def test_fp64_composition_preserves_sub_fp32_count_residuals_near_analytic_mle() -> None:
    """Numerical fixture, not evidence of anatomical reconstruction quality."""
    inputs = _inputs(precision="float64")
    coefficients = inputs["coefficients"].astype(np.float64)
    observed = inputs["views"][0].counts.reshape(2).astype(np.float64)
    optimum = np.linalg.solve(2 * coefficients.T, -np.log(observed / 10000))
    fields = (optimum + np.array([2e-7, -1e-7])).astype(np.float32)
    inputs["initial_fractions"] = fields.reshape(2, 1, 1, 1)
    means = 10000 * np.exp(-2 * coefficients.T @ fields.astype(np.float64))
    np.testing.assert_array_equal(means.astype(np.float32), observed.astype(np.float32))
    residual = (means - observed) / observed
    # The omitted term has relative size below 1e-20 for these residuals.
    expected_loss = float(np.sum(observed * residual**2 * (0.5 - residual / 3 + residual**2 / 4)))
    expected_gradient = 2 * coefficients @ (observed - means)
    solver = MaterialReconstruction(**inputs)
    loss = solver.evaluate(gradient=True)
    assert loss > 0
    np.testing.assert_allclose(loss, expected_loss, rtol=2e-5, atol=1e-17)
    np.testing.assert_allclose(solver.predictions_numpy()[0].reshape(2), means, rtol=2e-14)
    np.testing.assert_allclose(solver.gradient.numpy(), expected_gradient, rtol=2e-5, atol=1e-10)
    assert solver.views[0].paths.dtype == wp.float64
    assert solver.views[0].image_seed.dtype == wp.float64
    assert solver.views[0].channel_seed.dtype == wp.float64
    assert solver.views[0].path_seed.ptr == solver.views[0].path_total.ptr
    assert solver.fractions.dtype == solver.gradient.dtype == wp.float32


@pytest.mark.parametrize("target", [(-0.4, 0.8), (0.8, 0.8), (2.0, -0.2), (0.2, 0.3), (-1.0, -2.0)])
def test_projected_update_is_euclidean_simplex_projection(target: tuple[float, float]) -> None:
    solver = MaterialReconstruction(**_inputs())
    original = solver.fractions_numpy().reshape(2)
    gradient = (original - np.array(target, dtype=np.float32)).astype(np.float32)
    with solver.context.scope():
        solver.gradient.assign(gradient)
    mapping, slope = solver.projected_trial(1.0)
    represented = original.astype(np.float64) - gradient.astype(np.float64)
    projected = np.maximum(represented, 0)
    if np.sum(projected) > 1:
        a = np.clip((represented[0] - represented[1] + 1) / 2, 0, 1)
        projected = np.array([a, 1 - a])
    np.testing.assert_allclose(solver.trial.numpy(), projected, atol=5e-8)
    np.testing.assert_allclose(mapping, np.max(np.abs(projected - original)), atol=1e-14)
    np.testing.assert_allclose(
        slope, gradient.astype(np.float64) @ (solver.trial.numpy() - original), atol=1e-7
    )
    assert np.sum(solver.trial.numpy(), dtype=np.float64) <= 1 + 2**-24


def test_anisotropic_regularisation_against_independent_edge_reference() -> None:
    grid = GridSpec((2, 3, 4), (0.7, 1.3, 2.1), (-1.0, -1.0, -0.5))
    inputs = _inputs(grid=grid)
    inputs["coefficients"].fill(0)
    initial = inputs["initial_fractions"]
    initial[:] = np.linspace(0.02, 0.4, initial.size, dtype=np.float32).reshape(initial.shape)
    beta = 0.73
    inputs["settings"] = replace(inputs["settings"], regularisation_mm_inverse=beta)
    solver = MaterialReconstruction(**inputs)
    loss = solver.evaluate()
    expected = np.zeros_like(initial, dtype=np.float64)
    penalty = 0.0
    for axis, spacing in enumerate(reversed(grid.spacing_mm), 1):
        low = [slice(None)] * 4
        high = [slice(None)] * 4
        low[axis], high[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(low), tuple(high)
        difference = initial[b].astype(np.float64) - initial[a].astype(np.float64)
        coefficient = beta * math.prod(grid.spacing_mm) / spacing**2
        penalty += 0.5 * coefficient * float(np.sum(difference**2))
        expected[a] -= coefficient * difference
        expected[b] += coefficient * difference
    np.testing.assert_allclose(
        solver.gradient.numpy().reshape(initial.shape), expected, rtol=2e-6, atol=2e-8
    )
    observed = inputs["views"][0].counts.reshape(2).astype(np.float64)
    data_loss = np.sum(10000 - observed + observed * np.log(observed / 10000))
    np.testing.assert_allclose(loss - data_loss, penalty, rtol=1e-9, atol=1e-10)


def test_observations_are_frozen_and_reference_data_are_not_inputs() -> None:
    inputs = _inputs()
    solver = MaterialReconstruction(**inputs)
    original = solver.evaluate()
    gradient = solver.gradient.numpy()
    # Host-side evaluation/reference preparation cannot change copied fitting
    # observations or spectra. Only an explicit new solver can accept new data.
    inputs["views"][0].counts.fill(0)
    inputs["views"][0].weights.fill(123)
    inputs["coefficients"].fill(99)
    inputs["initial_fractions"].fill(0.3)
    np.testing.assert_allclose(solver.evaluate(), original, rtol=0, atol=0)
    np.testing.assert_allclose(solver.gradient.numpy(), gradient, rtol=0, atol=0)
    with pytest.raises(ContractError, match="trial evaluations"):
        solver.evaluate(gradient=True, trial=True)


@pytest.mark.parametrize("step_selection", ["geometric", "bb"])
@pytest.mark.parametrize("acceleration", ["none", "inertial"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_solve_recovers_analytic_two_material_mle_and_refreshes_outputs(
    step_selection: str,
    acceleration: str,
    precision: Literal["float32", "float64"],
) -> None:
    inputs = _inputs(precision=precision)
    inputs["settings"] = replace(
        inputs["settings"],
        iterations=150,
        gradient_mapping_tolerance=0.003,
        relative_gradient_mapping_tolerance=0,
        step_selection=step_selection,
        acceleration=acceleration,
    )
    solver = MaterialReconstruction(**inputs)
    callbacks: list[dict[str, Any]] = []
    report = solver.solve(lambda iteration, row: callbacks.append({"iteration": iteration, **row}))
    observations = inputs["views"][0].counts.reshape(2).astype(np.float64)
    reference = np.linalg.solve(
        2 * inputs["coefficients"].astype(np.float64).T, -np.log(observations / 10000)
    )
    np.testing.assert_allclose(solver.fractions_numpy().reshape(2), reference, rtol=0, atol=1e-4)
    np.testing.assert_allclose(solver.predictions_numpy()[0].reshape(2), observations, atol=0.03)
    assert report["termination"] == "projected_gradient_tolerance"
    assert report["accepted_steps"] == len(callbacks)
    assert all(row["objective"] < row["objective_before"] for row in callbacks)
    np.testing.assert_allclose(report["final_objective"], solver.evaluate(), atol=1e-12)


def test_failed_line_search_preserves_accepted_state() -> None:
    inputs = _inputs()
    inputs["settings"] = replace(
        inputs["settings"], initial_step=1.0e5, maximum_step=1.0e5, maximum_backtracks=1
    )
    solver = MaterialReconstruction(**inputs)
    before = solver.fractions_numpy()
    report = solver.solve()
    assert report["termination"] == "line_search_failed"
    np.testing.assert_array_equal(solver.fractions_numpy(), before)
    assert report["history"][-1]["accepted"] is False
    assert solver.predictions_numpy()[0].shape == (2, 1, 1)


def test_callback_failure_propagates_after_accepted_update() -> None:
    inputs = _inputs()
    solver = MaterialReconstruction(**inputs)

    def fail(_iteration: int, _record: dict[str, Any]) -> None:
        raise NumericalError("recording callback failed")

    with pytest.raises(NumericalError, match="recording callback failed"):
        solver.solve(fail)
    assert not np.array_equal(solver.fractions_numpy(), inputs["initial_fractions"])


def test_zero_counts_have_finite_analytic_loss_and_gradient() -> None:
    inputs = _inputs()
    inputs["views"][0].counts.fill(0)
    solver = MaterialReconstruction(**inputs)
    loss = solver.evaluate()
    matrix = 2 * inputs["coefficients"].astype(np.float64)
    means = 10000 * np.exp(-matrix.T @ inputs["initial_fractions"].reshape(2))
    np.testing.assert_allclose(loss, np.sum(means), rtol=1e-7)
    # The canonical field VJP scatters 64 samples with FP32 atomics. Its
    # accumulated rounding exceeds one store's unit roundoff at this scale.
    np.testing.assert_allclose(solver.gradient.numpy(), -matrix @ means, rtol=3e-6)


@pytest.mark.parametrize("materials", [1, 3, 8])
def test_simplex_general_material_count(materials: int) -> None:
    inputs = _inputs()
    provenance = inputs["spectral_spec"].coefficients_provenance[0]
    inputs["spectral_spec"] = replace(
        inputs["spectral_spec"],
        materials=materials,
        coefficients_provenance=(provenance,) * materials,
    )
    inputs["coefficients"] = np.full((materials, 2), 0.1, dtype=np.float32)
    initial = np.full((materials, 1, 1, 1), 0.1 / materials, dtype=np.float32)
    inputs["initial_fractions"] = initial
    solver = MaterialReconstruction(**inputs)
    target = np.linspace(2.5, -0.5, materials, dtype=np.float32)
    gradient = initial.reshape(materials) - target
    with solver.context.scope():
        solver.gradient.assign(gradient)
    solver.projected_trial(1)
    values = initial.reshape(materials).astype(np.float64) - gradient.astype(np.float64)
    ordered = np.sort(values)[::-1]
    thresholds = (np.cumsum(ordered) - 1) / np.arange(1, materials + 1)
    active = np.flatnonzero(ordered > thresholds)[-1]
    expected = np.maximum(values - thresholds[active], 0)
    np.testing.assert_allclose(solver.trial.numpy(), expected, atol=5e-8)


def test_heterogeneous_multiview_gradient_additivity_and_directional_reference() -> None:
    grid = GridSpec((2, 3, 4), (1.1, 0.9, 1.3), (-1.65, -0.9, -0.65))
    inputs = _inputs(grid=grid)
    rng = np.random.default_rng(57294)
    initial = np.ascontiguousarray(rng.uniform(0.07, 0.3, size=(2, *grid.shape)), dtype=np.float32)
    inputs["initial_fractions"] = initial
    counts = np.full((2, 2, 3), 8500, dtype=np.float32)
    first = replace(
        inputs["views"][0],
        counts=counts,
        geometry=DetectorGeometry(
            (-20.0, 0.2, 0.3),
            (20.0, -0.2, -0.3),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.35, 0.45),
            (2, 3),
        ),
    )
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
        counts=counts.copy() + np.float32(100),
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
    solver = MaterialReconstruction(**inputs)
    total = solver.evaluate()
    gradient = solver.gradient.numpy().copy()
    independent_loss = 0.0
    independent_gradient = np.zeros(initial.size, dtype=np.float64)
    for view in (first, second):
        view_inputs: dict[str, Any] = {**inputs, "views": (view,)}
        separate = MaterialReconstruction(**view_inputs)
        independent_loss += separate.evaluate()
        independent_gradient += separate.gradient.numpy()
    np.testing.assert_allclose(total, independent_loss, rtol=2e-12)
    np.testing.assert_allclose(gradient, independent_gradient, rtol=2e-5, atol=5e-5)
    direction = np.ascontiguousarray(rng.uniform(-0.2, 0.2, size=initial.shape), dtype=np.float32)
    exact = float(gradient.astype(np.float64) @ direction.reshape(-1))
    errors: list[float] = []
    for step in (0.02, 0.01, 0.005):
        solver.set_fractions(initial + np.float32(step) * direction)
        high = solver.evaluate(gradient=False)
        solver.set_fractions(initial - np.float32(step) * direction)
        low = solver.evaluate(gradient=False)
        errors.append(abs((high - low) / (2 * step) - exact))
    assert min(errors) < max(0.02, abs(exact) * 1e-3)


@pytest.mark.parametrize(
    "metric",
    [
        (1.0, 0.0, 1.0),
        (2.0, -0.8, 1.0),
        (0.21433530073918286, 0.6050145535007093, 1.7856646992608172),
    ],
)
def test_material_metric_projection_against_independent_active_set_qp(
    metric: tuple[float, float, float],
) -> None:
    inputs = _inputs()
    initial = np.array([0.2, 0.3], dtype=np.float32).reshape(2, 1, 1, 1)
    inputs["initial_fractions"] = initial
    solver = MaterialReconstruction(**inputs, material_metric=metric)
    matrix = np.array([[metric[0], metric[1]], [metric[1], metric[2]]], dtype=np.float64)
    fields = initial.reshape(2).astype(np.float64)
    targets = [
        (-0.4, -0.1),
        (1.2, -0.2),
        (-0.1, 1.3),
        (0.4, -0.4),
        (-0.2, 0.3),
        (0.7, 0.8),
        (0.25, 0.4),
    ]
    targets += list(np.random.default_rng(8912).uniform(-1.0, 2.0, size=(20, 2)))
    for target in targets:
        gradient = np.ascontiguousarray(matrix @ (fields - np.array(target)), dtype=np.float32)
        with solver.context.scope():
            solver.gradient.assign(gradient)
        _, slope = solver.projected_trial(1.0)
        actual = solver.trial.numpy().astype(np.float64)
        expected = _independent_triangle_qp(matrix, fields, gradient.astype(np.float64), 1.0)
        np.testing.assert_allclose(actual, expected, atol=7e-8, rtol=0)
        assert min(actual) >= 0 and sum(actual) <= 1 + 2**-24
        displacement = actual - fields
        bound = -float(displacement @ matrix @ displacement)
        assert slope <= bound + max(1e-7, abs(bound) * 1e-6)


@pytest.mark.parametrize(
    "metric",
    [(1.0, 0.9, 1.0), (1.0000002, 0.999996, 0.9999999)],
)
def test_metric_projection_preserves_cap_tangent_under_huge_common_gradient(
    metric: tuple[float, float, float],
) -> None:
    inputs = _inputs()
    initial = np.array([0.2, 0.3], dtype=np.float32).reshape(2, 1, 1, 1)
    inputs["initial_fractions"] = initial
    solver = MaterialReconstruction(**inputs, material_metric=metric)
    with solver.context.scope():
        solver.gradient.assign(np.array([-1.0e20, -1.0e20], dtype=np.float32))
    _, slope = solver.projected_trial(1.0)
    # The common linear component is constant on the sum-one edge. Compute its
    # independent scalar minimiser with Decimal to retain the tangent curvature
    # even when the valid metric is close to the supported condition limit.
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 60
        a, b, d = (Decimal.from_float(value) for value in metric)
        f0, f1 = (Decimal.from_float(float(value)) for value in initial.reshape(2))
        t = float(((a - b) * f0 + (b - d) * f1 + d - b) / (a + d - 2 * b))
    np.testing.assert_allclose(solver.trial.numpy(), [t, 1 - t], atol=4e-8)
    assert math.isfinite(slope) and slope < 0


def test_metric_changes_update_but_preserves_euclidean_mapping() -> None:
    inputs = _inputs()
    solver = MaterialReconstruction(**inputs, material_metric=(2.0, 0.8, 1.0))
    ordinary = MaterialReconstruction(**inputs)
    gradient = np.array([0.2, -0.1], dtype=np.float32)
    for item in (solver, ordinary):
        with item.context.scope():
            item.gradient.assign(gradient)
    expected, _ = ordinary.projected_trial(1.0)
    actual, _ = solver.projected_trial(1.0, use_metric=False)
    assert actual == expected
    np.testing.assert_array_equal(solver.trial.numpy(), ordinary.trial.numpy())
    solver.projected_trial(1.0)
    assert not np.array_equal(solver.trial.numpy(), ordinary.trial.numpy())


@pytest.mark.parametrize(
    ("field", "gradient", "active"),
    [
        ((0.0, 0.8879817724227905), (1893.68115234375, 4692.27685546875), 1),
        ((0.1142057254910469, 0.0), (56873.25, 219942.203125), 0),
        ((0.4, 0.0), (0.01, 1.0), 0),
    ],
)
def test_metric_active_edges_remain_valid_when_step_shrinks(
    field: tuple[float, float], gradient: tuple[float, float], active: int
) -> None:
    """Two real-case failure voxels and an independent small-gradient case.

    The solved edge stationarity residual inherits field-scale FP64 rounding,
    even when the proposed displacement is many orders of magnitude smaller.
    """
    from decimal import Decimal, localcontext

    metric = (0.21433530073918286, 0.6050145535007093, 1.7856646992608172)
    inputs = _inputs()
    initial = np.array(field, dtype=np.float32).reshape(2, 1, 1, 1)
    represented_gradient = np.array(gradient, dtype=np.float32)
    inputs["initial_fractions"] = initial
    solver = MaterialReconstruction(**inputs, material_metric=metric)
    with solver.context.scope():
        solver.gradient.assign(represented_gradient)
    for step in (5e-6, 1e-10, 9e-18):
        _, slope = solver.projected_trial(step)
        with localcontext() as context:
            context.prec = 60
            f = [Decimal.from_float(float(value)) for value in initial.reshape(2)]
            g = [Decimal.from_float(float(value)) for value in represented_gradient]
            diagonal = Decimal.from_float(metric[2 * active])
            cross = Decimal.from_float(metric[1])
            alpha = Decimal.from_float(step)
            position = max(Decimal(0), min(Decimal(1), f[active] - alpha * g[active] / diagonal))
            # The inactive coordinate's nonnegative derivative certifies that
            # this independent one-dimensional edge solution is the 2D QP minimum.
            assert cross * (position - f[active]) + alpha * g[1 - active] >= 0
            expected = np.zeros(2, dtype=np.float32)
            expected[active] = float(position)
        np.testing.assert_array_equal(solver.trial.numpy(), expected)
        assert math.isfinite(slope) and slope <= 0


@pytest.mark.parametrize("step_selection", ["geometric", "bb"])
@pytest.mark.parametrize("acceleration", ["none", "inertial"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_metric_solver_recovers_the_same_analytic_mle(
    step_selection: str, acceleration: str, precision: Literal["float32", "float64"]
) -> None:
    inputs = _inputs(precision=precision)
    coefficients = inputs["coefficients"].astype(np.float64)
    fisher = coefficients @ coefficients.T
    matrix = 2 * fisher / np.trace(fisher)
    inputs["settings"] = replace(
        inputs["settings"],
        iterations=100,
        gradient_mapping_tolerance=0.003,
        relative_gradient_mapping_tolerance=0,
        step_selection=step_selection,
        acceleration=acceleration,
    )
    solver = MaterialReconstruction(
        **inputs, material_metric=(float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[1, 1]))
    )
    report = solver.solve()
    observed = inputs["views"][0].counts.reshape(2).astype(np.float64)
    expected = np.linalg.solve(2 * coefficients.T, -np.log(observed / 10000))
    np.testing.assert_allclose(solver.fractions_numpy().reshape(2), expected, atol=1e-4, rtol=0)
    assert report["termination"] == "projected_gradient_tolerance"
    assert report["gradient_mapping_metric"] == "euclidean"
    final_mapping, _ = solver.projected_trial(solver.settings.mapping_step, use_metric=False)
    assert final_mapping == report["final_gradient_mapping"]
    assert all(row["objective"] < row["objective_before"] for row in report["history"])


@pytest.mark.parametrize("metric", [None, (2.0, -0.8, 1.0)])
def test_secant_step_matches_independent_quadratic_and_constrained_update(
    metric: tuple[float, float, float] | None,
) -> None:
    inputs = _inputs()
    inputs["settings"] = replace(inputs["settings"], step_selection="bb")
    solver = MaterialReconstruction(**inputs, material_metric=metric)
    quadratic = np.array([[8.0, 1.5], [1.5, 3.0]])
    previous = np.array([0.2, 0.3], dtype=np.float32)
    current = np.array([0.31, 0.22], dtype=np.float32)
    previous_gradient = np.asarray(quadratic @ previous - np.array([3.0, -0.2]), dtype=np.float32)
    current_gradient = np.asarray(quadratic @ current - np.array([3.0, -0.2]), dtype=np.float32)
    private: Any = solver
    private._prepare_secant()
    solver.set_fractions(previous.reshape(2, 1, 1, 1))
    with solver.context.scope():
        solver.gradient.assign(previous_gradient)
    private._remember_secant()
    solver.set_fractions(current.reshape(2, 1, 1, 1))
    with solver.context.scope():
        solver.gradient.assign(current_gradient)
    step = private._secant_step(solver.settings)
    matrix = (
        np.eye(2) if metric is None else np.array([[metric[0], metric[1]], [metric[1], metric[2]]])
    )
    displacement = current.astype(np.float64) - previous.astype(np.float64)
    change = current_gradient.astype(np.float64) - previous_gradient.astype(np.float64)
    expected = float(displacement @ matrix @ displacement / (displacement @ change))
    assert math.isclose(step, expected, rel_tol=2e-14)
    _, slope = solver.projected_trial(step)
    trial = solver.trial.numpy().astype(np.float64)
    oracle = _independent_triangle_qp(
        matrix, current.astype(np.float64), current_gradient.astype(np.float64), step
    )
    np.testing.assert_allclose(trial, oracle, atol=7e-8, rtol=0)
    assert slope <= 0
    assert np.min(trial) >= 0 and np.sum(trial) <= 1 + 2**-24
    pointers = (
        private._secant_fields.ptr,
        private._secant_gradient.ptr,
        private._secant_products.ptr,
        private._secant_totals.ptr,
    )
    private._prepare_secant()
    assert pointers == (
        private._secant_fields.ptr,
        private._secant_gradient.ptr,
        private._secant_products.ptr,
        private._secant_totals.ptr,
    )


@pytest.mark.parametrize(
    "curvature,expected", [(0.0, None), (-2.0, None), (1e20, 1e-16), (1e-12, 1.0)]
)
def test_secant_step_rejects_bad_curvature_and_clips_positive_proposals(
    curvature: float,
    expected: float | None,
) -> None:
    inputs = _inputs()
    inputs["settings"] = replace(inputs["settings"], step_selection="bb")
    solver = MaterialReconstruction(**inputs)
    private: Any = solver
    private._prepare_secant()
    for remembered, fields in ((True, [0.2, 0.3]), (False, [0.3, 0.2])):
        values = np.array(fields, dtype=np.float32)
        solver.set_fractions(values.reshape(2, 1, 1, 1))
        with solver.context.scope():
            solver.gradient.assign(np.ascontiguousarray(curvature * values, dtype=np.float32))
        if remembered:
            private._remember_secant()
    result = private._secant_step(solver.settings)
    if expected is None:
        assert result is None
    else:
        assert math.isclose(result, expected, rel_tol=1e-14)


def test_overflowing_secant_reduction_falls_back_without_poisoning_a_valid_trial() -> None:
    inputs = _inputs()
    inputs["settings"] = replace(inputs["settings"], step_selection="bb")
    solver = MaterialReconstruction(**inputs, material_metric=(1e308, 0.0, 1e308))
    private: Any = solver
    private._prepare_secant()
    for remembered, values in ((True, [1.0, 0.0]), (False, [0.0, 1.0])):
        field = np.asarray(values, dtype=np.float32)
        solver.set_fractions(field.reshape(2, 1, 1, 1))
        with solver.context.scope():
            solver.gradient.assign(field)
        if remembered:
            private._remember_secant()
    # Every input/product is finite, but s.T H s = 2e308 overflows FP64.
    assert private._secant_step(solver.settings) is None
    mapping, slope = solver.projected_trial(0.1, use_metric=False)
    assert math.isfinite(mapping) and math.isfinite(slope) and slope < 0


@pytest.mark.parametrize(
    "metric",
    [
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 2.0, 1.0),
        (1.0, 0.0, 1e-8),
        (math.nan, 0.0, 1.0),
    ],
)
def test_invalid_material_metric_is_rejected(metric: tuple[float, float, float]) -> None:
    with pytest.raises(ContractError, match="material_metric"):
        MaterialReconstruction(**_inputs(), material_metric=metric)


def test_coupled_metric_requires_exactly_two_materials() -> None:
    inputs = _inputs()
    inputs["spectral_spec"] = replace(
        inputs["spectral_spec"],
        materials=1,
        coefficients_provenance=(inputs["spectral_spec"].coefficients_provenance[0],),
    )
    with pytest.raises(ContractError, match="exactly two materials"):
        MaterialReconstruction(**inputs, material_metric=(1.0, 0.0, 1.0))


@pytest.mark.parametrize("metric", [None, (2.0, -0.8, 1.0)])
@pytest.mark.parametrize("nonnegative", [False, True])
@pytest.mark.parametrize("step", [0.07, 1e-20])
def test_inertial_projection_matches_independent_qp_and_preserves_true_slope(
    metric: tuple[float, float, float] | None, nonnegative: bool, step: float
) -> None:
    inputs = _inputs()
    inputs["initial_fractions"] = np.array([0.7, 0.2], dtype=np.float32).reshape(2, 1, 1, 1)
    solver = MaterialReconstruction(
        **inputs, material_metric=metric, field_domain="nonnegative" if nonnegative else "fractions"
    )
    private: Any = solver
    private._prepare_inertia()
    previous = solver.fractions_numpy().reshape(2).astype(np.float64)
    current = np.array([0.1, 0.85], dtype=np.float32)
    solver.set_fractions(current.reshape(2, 1, 1, 1))
    matrix = (
        np.eye(2) if metric is None else np.array([[metric[0], metric[1]], [metric[1], metric[2]]])
    )
    centre = current.astype(np.float64) + 0.95 * (current.astype(np.float64) - previous)
    for gradient in ([1.0, -0.5], [-1.0, 2.0], [0.3, 0.1]):
        grad = np.asarray(gradient, dtype=np.float32)
        with solver.context.scope():
            solver.gradient.assign(grad)
        diagnostic_before = solver.projected_trial(1e-3, use_metric=False)[0]
        _, slope = private._projected_trial(step, momentum=0.95)
        trial = solver.trial.numpy().astype(np.float64)
        expected = _independent_triangle_qp(
            matrix, centre, grad.astype(np.float64), step, nonnegative=nonnegative
        )
        np.testing.assert_allclose(trial, expected, atol=1.5e-7, rtol=0)
        assert math.isclose(
            slope, float(grad.astype(np.float64) @ (trial - current)), abs_tol=1e-14
        )
        np.testing.assert_array_equal(solver.fractions_numpy().reshape(2), current)
        np.testing.assert_array_equal(solver.gradient.numpy(), grad)
        assert solver.projected_trial(1e-3, use_metric=False)[0] == diagnostic_before


@pytest.mark.parametrize("rejection", ["non_descent", "armijo", "nonfinite"])
def test_inertial_rejection_restarts_at_same_step_without_changing_accepted_state(
    monkeypatch: pytest.MonkeyPatch,
    rejection: str,
) -> None:
    """Fault-inject only the proposal verdict; all fields/objectives use real CUDA."""
    inputs = _inputs()
    inputs["settings"] = replace(
        inputs["settings"], iterations=2, acceleration="inertial", step_selection="bb"
    )
    solver = MaterialReconstruction(**inputs)
    private: Any = solver
    original = private._projected_trial
    calls: list[dict[str, Any]] = []

    def reject_momentum(step: float, *, use_metric: bool = True, momentum: float = 0.0) -> Any:
        values = original(step, use_metric=use_metric, momentum=momentum)
        if use_metric:
            calls.append(
                {
                    "step": step,
                    "momentum": momentum,
                    "fields": solver.fractions_numpy().copy(),
                    "gradient": solver.gradient.numpy().copy(),
                }
            )
        if momentum:
            if rejection == "nonfinite":
                with solver.context.scope():
                    private._status.fill_(2)
                raise NumericalError("injected inertial proposal failure")
            return values[0], 1.0 if rejection == "non_descent" else -1e100
        return values

    monkeypatch.setattr(solver, "_projected_trial", reject_momentum)
    result = solver.solve()
    assert result["accepted_steps"] == 2
    index = next(i for i, row in enumerate(calls) if row["momentum"])
    before, retry = calls[index : index + 2]
    assert retry["momentum"] == 0 and retry["step"] == before["step"]
    np.testing.assert_array_equal(before["fields"], retry["fields"])
    np.testing.assert_array_equal(before["gradient"], retry["gradient"])
    assert result["history"][-1]["inertial_restart"] == rejection
    assert all(row["objective"] < row["objective_before"] for row in result["history"])


@pytest.mark.parametrize("metric", [None, (2.0, -0.8, 1.0)])
@pytest.mark.parametrize("nonnegative", [False, True])
def test_voxel_scaled_metric_matches_independent_block_qps_and_secant(
    metric: tuple[float, float, float] | None, nonnegative: bool
) -> None:
    grid = GridSpec((1, 1, 3), (1.0, 1.0, 1.0))
    inputs = _inputs(grid=grid)
    solver = MaterialReconstruction(
        **inputs,
        material_metric=metric,
        field_domain="nonnegative" if nonnegative else "fractions",
    )
    private: Any = solver
    matrix = (
        np.eye(2) if metric is None else np.array([[metric[0], metric[1]], [metric[1], metric[2]]])
    )
    scale = np.array([0.1, 1.0, 40.0], dtype=np.float32)
    previous = np.array([[0.2, 0.3, 0.7], [0.4, 0.1, 0.2]], dtype=np.float32)
    current = np.array([[0.5, 0.2, 0.1], [0.1, 0.6, 0.85]], dtype=np.float32)
    quadratic = np.array([[3.0, 0.7], [0.7, 8.0]])
    old_gradient = (quadratic @ previous).astype(np.float32)
    gradient = (quadratic @ current).astype(np.float32)
    solver.set_fractions(previous.reshape(2, *grid.shape))
    private._prepare_inertia()
    private._prepare_secant()
    with solver.context.scope():
        solver.gradient.assign(old_gradient.ravel())
    private._remember_secant()
    solver.set_fractions(current.reshape(2, *grid.shape))
    with solver.context.scope():
        solver.gradient.assign(gradient.ravel())
    unscaled_diagnostic = solver.projected_trial(1e-3, use_metric=False)[0]
    host_scale = scale.reshape(grid.shape).copy()
    solver.set_voxel_metric_scale(host_scale)
    host_scale.fill(7.0)
    np.testing.assert_array_equal(solver.voxel_metric_scale.numpy(), scale)
    assert solver.projected_trial(1e-3, use_metric=False)[0] == unscaled_diagnostic
    _, slope = private._projected_trial(0.2, momentum=0.6)
    trial = solver.trial.numpy().reshape(2, 3).astype(np.float64)
    centre = current.astype(np.float64) + 0.6 * (current.astype(np.float64) - previous)
    for voxel in range(3):
        expected = _independent_triangle_qp(
            float(scale[voxel]) * matrix,
            centre[:, voxel],
            gradient[:, voxel].astype(np.float64),
            0.2,
            nonnegative=nonnegative,
        )
        np.testing.assert_allclose(trial[:, voxel], expected, atol=1e-7, rtol=0)
    assert math.isclose(
        slope, float(np.sum(gradient.astype(np.float64) * (trial - current))), abs_tol=1e-12
    )
    displacement = current.astype(np.float64) - previous
    change = gradient.astype(np.float64) - old_gradient
    numerator = sum(
        float(scale[i]) * displacement[:, i] @ matrix @ displacement[:, i] for i in range(3)
    )
    expected_step = numerator / np.sum(displacement * change)
    policy = replace(solver.settings, maximum_step=100.0, step_selection="bb")
    assert math.isclose(private._secant_step(policy), expected_step, rel_tol=2e-14)
    assert solver.projected_trial(1e-3, use_metric=False)[0] == unscaled_diagnostic


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, 1e-8])
def test_invalid_resident_voxel_scale_preserves_the_previous_metric(bad: float) -> None:
    inputs = _inputs(grid=GridSpec((1, 1, 2), (1.0, 1.0, 1.0)))
    solver = MaterialReconstruction(**inputs)
    solver.set_voxel_metric_scale(np.ones(inputs["grid"].shape, dtype=np.float32))
    previous_pointer = solver.voxel_metric_scale.ptr
    with solver.context.scope():
        candidate = wp.array(np.array([1.0, bad], dtype=np.float32), device=solver.device)
    with pytest.raises(ContractError, match="voxel metric scale"):
        solver.set_voxel_metric_scale(candidate)
    assert solver.voxel_metric_scale.ptr == previous_pointer
    solver.evaluate(gradient=True)
    mapping, slope = solver.projected_trial(0.01)
    assert math.isfinite(mapping) and math.isfinite(slope)


@pytest.mark.parametrize("shape", [(1, 1, 3), (16, 16, 16)])
def test_voxel_scale_extrema_ignore_padding_at_every_reduction_level(
    shape: tuple[int, int, int],
) -> None:
    grid = GridSpec(shape, (1.0, 1.0, 1.0))
    solver = MaterialReconstruction(**_inputs(grid=grid))
    values = np.linspace(0.025, 85.0, grid.voxels, dtype=np.float32).reshape(shape)
    solver.set_voxel_metric_scale(values)
    private: Any = solver
    assert private._voxel_metric_summary["minimum"] == float(values.min())
    assert private._voxel_metric_summary["maximum"] == float(values.max())
