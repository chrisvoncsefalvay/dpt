"""Independent resident penalty/Fisher parity, including unchanged eager callers."""

import importlib
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError, NumericalError
from dpt.kernels.material_preconditioning import add_regulariser
from dpt.material_preconditioning import prepare_fisher_scaling
from dpt.material_reconstruction import MaterialReconstruction
from helpers.material_fisher_inputs import _inputs, _reference
from helpers.material_solver_inputs import _independent_triangle_qp
from helpers.resident_penalty_inputs import (
    MATRIX,
    InvalidCurvaturePenalty,
    PreparedQuadraticPenalty,
    ValueOnlyPenalty,
    quadratic_reference,
)

wp: Any = importlib.import_module("warp")


def scope(device):
    return cpu_test_context() if device == "cpu" else nullcontext()


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_custom_penalty_matches_independent_value_gradient_and_trial_preservation(
    precision, warp_device, monkeypatch
):
    with scope(warp_device):
        inputs = _inputs(precision=precision)
        factor = np.linspace(0.3, 1.1, inputs["grid"].voxels)
        base = MaterialReconstruction(**inputs, device=warp_device)
        base_loss = base.evaluate()
        base_gradient = base.gradient.numpy().astype(float)
        penalty = PreparedQuadraticPenalty(factor, device=warp_device)
        solver = MaterialReconstruction(**inputs, device=warp_device, penalty_operator=penalty)
        expected_loss, expected_gradient = quadratic_reference(inputs["initial_fractions"], factor)
        original_numpy = wp.array.numpy
        forbidden = {int(solver.fractions.ptr), int(solver.gradient.ptr), int(penalty.factor.ptr)}

        def scalar_reads_only(array, *args, **kwargs):
            assert int(array.ptr) not in forbidden, "a resident field reached the host"
            return original_numpy(array, *args, **kwargs)

        with monkeypatch.context() as guard:
            guard.setattr(wp.array, "numpy", scalar_reads_only)
            loss = solver.evaluate()
        np.testing.assert_allclose(loss - base_loss, expected_loss, rtol=3e-10, atol=2e-13)
        np.testing.assert_allclose(
            solver.gradient.numpy(), base_gradient + expected_gradient, rtol=3e-6, atol=3e-6
        )
        previous = solver.gradient.numpy().copy()
        wp.copy(solver.trial, solver.fractions)
        trial_loss = solver.evaluate(gradient=False, trial=True)
        assert trial_loss == loss
        np.testing.assert_array_equal(solver.gradient.numpy(), previous)


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("beta", [0.0, 0.07])
def test_custom_fisher_scaling_matches_full_independent_hessian_majorant(
    precision, beta, warp_device
):
    with scope(warp_device):
        inputs = _inputs(beta=beta, precision=precision)
        factor = np.linspace(0.3, 1.1, inputs["grid"].voxels)
        penalty = PreparedQuadraticPenalty(factor, device=warp_device)
        solver = MaterialReconstruction(**inputs, device=warp_device, penalty_operator=penalty)
        solver.evaluate()
        initial = solver.fractions.numpy().copy()
        gradient = solver.gradient.numpy().copy()
        raw, full_curvature, metric = _reference(inputs)
        raw += factor * np.linalg.eigvalsh(MATRIX)[-1] / np.linalg.eigvalsh(metric)[0]
        full_curvature += np.kron(MATRIX, np.diag(factor))
        floored = np.maximum(raw, raw.max() * 1e-6)
        result = prepare_fisher_scaling(solver)
        np.testing.assert_allclose(result.scale.numpy(), floored / floored.mean(), rtol=2e-5)
        np.testing.assert_allclose(result.diagnostics["normaliser"], floored.mean(), rtol=2e-5)
        np.testing.assert_allclose(
            result.diagnostics["minimum_raw_curvature"], raw.min(), rtol=2e-5
        )
        np.testing.assert_allclose(
            result.diagnostics["maximum_raw_curvature"], raw.max(), rtol=2e-5
        )
        assert result.diagnostics["custom_penalty_curvature"] == "PreparedQuadraticPenalty"
        assert np.linalg.eigvalsh(np.kron(metric, np.diag(floored)) - full_curvature).min() > -1e-9
        np.testing.assert_array_equal(solver.fractions.numpy(), initial)
        np.testing.assert_array_equal(solver.gradient.numpy(), gradient)
        # No callback accumulation may leak from one preparation to the next.
        repeated = prepare_fisher_scaling(solver)
        np.testing.assert_array_equal(repeated.scale.numpy(), result.scale.numpy())


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_invalid_custom_curvature_is_rejected_without_changing_fields_or_gradient(
    value, warp_device, monkeypatch
):
    with scope(warp_device):
        inputs = _inputs()
        penalty = InvalidCurvaturePenalty(
            np.ones(inputs["grid"].voxels), device=warp_device, value=value
        )
        solver = MaterialReconstruction(**inputs, device=warp_device, penalty_operator=penalty)
        solver.evaluate()
        fields, gradient = solver.fractions.numpy().copy(), solver.gradient.numpy().copy()
        launched = []
        launch = wp.launch

        def record_launch(kernel, *args, **kwargs):
            launched.append(kernel.key)
            return launch(kernel, *args, **kwargs)

        with monkeypatch.context() as guard:
            guard.setattr(wp, "launch", record_launch)
            with pytest.raises(NumericalError, match="Fisher voxel scaling"):
                prepare_fisher_scaling(solver)
        assert "floor_curvature" not in launched
        assert "normalise_curvature" not in launched
        np.testing.assert_array_equal(solver.fractions.numpy(), fields)
        np.testing.assert_array_equal(solver.gradient.numpy(), gradient)


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_invalid_callback_curvature_never_updates_extrema(value, warp_device):
    with scope(warp_device):
        projected = wp.array([10.0], dtype=wp.float32, device=warp_device)
        penalty = wp.array([value], dtype=wp.float64, device=warp_device)
        raw = wp.array([123.0], dtype=wp.float64, device=warp_device)
        extrema = wp.array([float("inf"), 0.0], dtype=wp.float64, device=warp_device)
        counts = wp.zeros(2, dtype=wp.int32, device=warp_device)
        status = wp.zeros(1, dtype=wp.int32, device=warp_device)
        wp.launch(
            add_regulariser,
            dim=1,
            inputs=[
                projected,
                1,
                1,
                1,
                wp.vec3d(0.0),
                1.0,
                True,
                penalty,
                raw,
                extrema,
                counts,
                status,
            ],
            device=warp_device,
            record_tape=False,
        )
        assert int(status.numpy()[0]) == 2
        np.testing.assert_array_equal(raw.numpy(), [123.0])
        np.testing.assert_array_equal(extrema.numpy(), [float("inf"), 0.0])


def test_legacy_eager_callable_retains_value_gradient_and_refuses_undeclared_curvature(warp_device):
    with scope(warp_device):
        inputs = _inputs()
        prepared = PreparedQuadraticPenalty(np.ones(inputs["grid"].voxels), device=warp_device)
        solver = MaterialReconstruction(
            **inputs, device=warp_device, penalty_operator=ValueOnlyPenalty(prepared)
        )
        assert np.isfinite(solver.evaluate())
        with pytest.raises(ContractError, match="curvature bound"):
            prepare_fisher_scaling(solver)


@pytest.mark.parametrize("target", [(0.2, 0.25), (0.35, 0.15), (-0.1, 0.3)])
def test_penalty_only_constrained_solve_matches_independent_convex_qp(target, warp_device):
    with scope(warp_device):
        inputs = _inputs(coefficients=False)
        inputs["settings"] = replace(
            inputs["settings"],
            iterations=100,
            gradient_mapping_tolerance=2e-6,
            relative_gradient_mapping_tolerance=0.0,
        )
        penalty = PreparedQuadraticPenalty(
            np.ones(inputs["grid"].voxels), device=warp_device, target=target
        )
        solver = MaterialReconstruction(**inputs, device=warp_device, penalty_operator=penalty)
        scaling = prepare_fisher_scaling(solver)
        scaling.install(solver)
        report = solver.solve()
        expected = _independent_triangle_qp(MATRIX, np.zeros(2), -MATRIX @ np.asarray(target), 1.0)
        np.testing.assert_allclose(
            solver.fractions_numpy().reshape(2, -1),
            np.repeat(expected[:, None], inputs["grid"].voxels, axis=1),
            atol=8e-6,
            rtol=0,
        )
        assert report["termination"] == "projected_gradient_tolerance"
        assert report["final_gradient_mapping"] <= report["gradient_mapping_threshold"]


def test_active_simplex_face_stationarity_matches_independent_qp(warp_device):
    with scope(warp_device):
        target = (0.8, 0.6)
        inputs = _inputs(coefficients=False)
        expected = _independent_triangle_qp(MATRIX, np.zeros(2), -MATRIX @ np.asarray(target), 1.0)
        assert abs(expected.sum() - 1.0) < 2e-15
        inputs["initial_fractions"] = np.ascontiguousarray(
            np.broadcast_to(expected[:, None], (2, inputs["grid"].voxels)), dtype=np.float32
        ).reshape(inputs["initial_fractions"].shape)
        inputs["settings"] = replace(
            inputs["settings"],
            gradient_mapping_tolerance=2e-6,
            relative_gradient_mapping_tolerance=0.0,
        )
        penalty = PreparedQuadraticPenalty(
            np.ones(inputs["grid"].voxels), device=warp_device, target=target
        )
        solver = MaterialReconstruction(**inputs, device=warp_device, penalty_operator=penalty)
        prepare_fisher_scaling(solver).install(solver)
        report = solver.solve()
        assert report["termination"] == "projected_gradient_tolerance"
        assert report["accepted_steps"] == 0
        assert report["final_gradient_mapping"] <= 2e-6
        np.testing.assert_allclose(
            solver.fractions_numpy().reshape(2, -1),
            np.repeat(expected[:, None], inputs["grid"].voxels, axis=1),
            atol=8e-8,
            rtol=0,
        )
