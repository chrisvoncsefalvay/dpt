"""Reduction capacity, path reuse and opt-in deterministic solver policies."""

import importlib
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError
from dpt.material_reconstruction import (
    MaterialReconstruction,
    MaterialReconstructionSettings,
    StepPolicy,
)
from dpt.registration import Evaluation, RecoveryPolicy, recover_parameters
from helpers import material_inputs as cases

# Warp annotation factories are consumed by its JIT, not static Python typing.
# pyright: reportInvalidTypeForm=false


wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize("voxels", [255, 256, 257, 65537])
def test_tiled_secant_and_gradient_reductions_cover_every_logical_voxel(voxels, warp_device: str):
    from dpt._reductions import prepare_reduction
    from dpt._runtime import prepare_context
    from dpt.kernels import material_reconstruction as kernels

    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        ctx = prepare_context(device=warp_device)
        values = np.random.default_rng(893).uniform(-0.2, 0.2, 2 * voxels).astype(np.float32)
        fields = wp.array(values, dtype=wp.float32, device=warp_device)
        old = wp.zeros(2 * voxels, dtype=wp.float32, device=warp_device)
        gradient = wp.array(3 * values, dtype=wp.float32, device=warp_device)
        scale_values = np.linspace(0.5, 2, voxels, dtype=np.float32)
        scale = wp.array(scale_values, dtype=wp.float32, device=warp_device)
        blocks = (2 * voxels + 255) // 256
        numerator = wp.full(blocks + 1, -1e99, dtype=wp.float64, device=warp_device)
        denominator = wp.empty_like(numerator)
        status = wp.zeros(1, dtype=wp.int32, device=warp_device)
        wp.launch_tiled(
            kernels.secant_products_tiled,
            dim=blocks,
            inputs=[
                2 * voxels,
                fields,
                gradient,
                old,
                old,
                voxels,
                True,
                wp.vec3d(2, 0.25, 1),
                True,
                scale,
                False,
                numerator,
                numerator,
                denominator,
            ],
            block_dim=256,
            device=warp_device,
        )
        total = wp.empty(1, dtype=wp.float64, device=warp_device)
        tree = prepare_reduction(ctx, blocks)
        tree.finish(blocks, total, status, source=numerator)
        first, second = values.reshape(2, voxels).astype(float)
        expected = np.sum(scale_values * (2 * first**2 + 0.5 * first * second + second**2))
        np.testing.assert_allclose(total.numpy()[0], expected, rtol=3e-15)
        assert numerator.numpy()[-1] == -1e99
        tree.finish(blocks, total, status, source=denominator)
        np.testing.assert_allclose(
            total.numpy()[0], np.dot(values.astype(float), (3 * values).astype(float)), rtol=3e-15
        )
        maximum = wp.empty((voxels + 255) // 256, dtype=wp.float64, device=warp_device)
        wp.launch_tiled(
            kernels.gradient_magnitudes_tiled,
            dim=maximum.size,
            inputs=[gradient, 2, voxels, maximum, status],
            block_dim=256,
            device=warp_device,
        )
        np.testing.assert_allclose(
            maximum.numpy().max(), np.abs((3 * values).astype(float)).max(), rtol=0
        )
        assert status.numpy()[0] == 0


def test_solver_scalar_scratch_is_tiled_and_accepted_forward_is_reused(monkeypatch):
    import dpt.material_reconstruction as module

    with cpu_test_context():
        inputs = cases._inputs("float64")
        solver = MaterialReconstruction(**inputs)
        assert solver._components.size == (solver.size + 255) // 256
        assert solver._mappings.size == (solver.grid.voxels + 255) // 256
        projected = []
        original = module.project_material_paths

        def counted(*args, **kwargs):
            projected.append(kwargs["workspace"])
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "project_material_paths", counted)
        report = solver.solve()
        assert report["accepted_steps"] == inputs["settings"].iterations
        # Every accepted update gets a gradient, but only the initial accepted
        # state needs a new forward projection. Accepted trial paths survive.
        assert sum(item in [view.accepted for view in solver.views] for item in projected) == len(
            solver.views
        )
        solver._prepare_secant()
        assert solver._secant_products.size == (solver.size + 255) // 256


def test_nonmonotone_search_is_opt_in_and_uses_a_finite_reference():
    assert MaterialReconstructionSettings().nonmonotone_window == 1
    monotone = StepPolicy(MaterialReconstructionSettings())
    nonmonotone = StepPolicy(
        MaterialReconstructionSettings(nonmonotone_window=5, step_selection="bb")
    )
    assert not monotone.accepts(1, 1.2, -0.1)
    assert nonmonotone.accepts(1, 1.2, -0.1, reference=2)
    assert not nonmonotone.accepts(1, 2, -0.1, reference=2)
    assert not nonmonotone.accepts(1, 1.2, 0.1, reference=2)
    with pytest.raises(ContractError):
        MaterialReconstructionSettings(nonmonotone_window=5, acceleration="inertial")
    with cpu_test_context():
        inputs = cases._inputs("float64")
        inputs["settings"] = replace(inputs["settings"], nonmonotone_window=5, step_selection="bb")
        report = MaterialReconstruction(**inputs).solve()
        assert report["accepted_steps"] > 0
        for item in report["history"]:
            if item["accepted"]:
                assert item["objective"] < item["line_search_reference"]


@pytest.mark.parametrize("penalty", ["quadratic", "huber"])
def test_per_material_physical_penalty_matches_numpy_directional_derivative(penalty):
    with cpu_test_context():
        inputs = cases._inputs("float64", regularisation=0)
        settings = replace(
            inputs["settings"],
            regularisation_per_material=(0.1, 0.8),
            penalty=penalty,
            huber_gradient_delta=0.03,
        )
        inputs["settings"] = settings
        solver = MaterialReconstruction(**inputs)
        loss = solver.evaluate()
        fields = inputs["initial_fractions"]
        direction = np.random.default_rng(849).uniform(-0.1, 0.1, fields.shape).astype(np.float32)
        full_gradient = solver.gradient.numpy().astype(float).copy()
        reference_inputs = {
            **inputs,
            "settings": replace(settings, regularisation_per_material=None),
        }
        reference_solver = MaterialReconstruction(**reference_inputs)
        reference_loss = reference_solver.evaluate()
        derivative = float(
            (full_gradient - reference_solver.gradient.numpy().astype(float))
            @ direction.reshape(-1)
        )

        def objective(values):
            prior = 0.0
            for material, beta in enumerate((0.1, 0.8)):
                for axis, spacing in zip((2, 1, 0), inputs["grid"].spacing_mm, strict=True):
                    differences = np.diff(values[material].astype(float), axis=axis) / spacing
                    absolute = np.abs(differences)
                    potential = (
                        0.5 * differences**2
                        if penalty == "quadratic"
                        else np.where(
                            absolute <= settings.huber_gradient_delta,
                            0.5 * differences**2,
                            settings.huber_gradient_delta
                            * (absolute - 0.5 * settings.huber_gradient_delta),
                        )
                    )
                    prior += beta * np.prod(inputs["grid"].spacing_mm) * potential.sum()
            return prior

        np.testing.assert_allclose(loss - reference_loss, objective(fields), rtol=3e-10, atol=3e-13)
        h = 0.001
        fd = (objective(fields + h * direction) - objective(fields - h * direction)) / (2 * h)
        np.testing.assert_allclose(derivative, fd, rtol=3e-3, atol=2e-5)


def test_declared_loss_resolution_stops_a_plateau_without_claiming_stationarity():
    def quantized(point):
        value = point[0]
        return Evaluation(round(value * value, 4), (2 * value,))

    base = RecoveryPolicy(
        max_iterations=5,
        line_search_evaluations=16,
        gradient_tolerance=0,
        relative_loss_tolerance=0,
        step_tolerance=0,
    )
    ordinary = recover_parameters(quantized, (0.003,), policy=base)
    resolved = recover_parameters(
        quantized, (0.003,), policy=replace(base, absolute_loss_resolution=1e-4)
    )
    assert ordinary.reason == "line_search_failed"
    assert resolved.reason == "loss_resolution"
    assert not resolved.stationary
    assert resolved.parameters == (0.003,)
    assert resolved.resolution_limited_trials > 0


@pytest.mark.parametrize("materials", [1, 2, 3, 8])
def test_weighted_simplex_tiled_projection_matches_independent_active_sets(
    materials, warp_device: str
):
    from dpt.kernels.material_reconstruction import projected_step_tiled

    voxels = 257
    rng = np.random.default_rng(801 + materials)
    fields = rng.uniform(0, 1 / materials, (materials, voxels)).astype(np.float32)
    gradients = rng.uniform(-4, 3, fields.shape).astype(np.float32)
    diagonal = np.geomspace(0.3, 30, materials)
    field_device = wp.array(fields.reshape(-1), dtype=wp.float32, device=warp_device)
    gradient_device = wp.array(gradients.reshape(-1), dtype=wp.float32, device=warp_device)
    diagonal_device = wp.array(diagonal, dtype=wp.float64, device=warp_device)
    trial = wp.empty_like(field_device)
    slopes = wp.empty(2, dtype=wp.float64, device=warp_device)
    mappings = wp.empty_like(slopes)
    status = wp.zeros(1, dtype=wp.int32, device=warp_device)
    step = 0.4
    wp.launch_tiled(
        projected_step_tiled(materials),
        dim=2,
        inputs=[
            field_device,
            gradient_device,
            voxels,
            step,
            field_device,
            0.0,
            False,
            field_device,
            True,
            diagonal_device,
            trial,
            slopes,
            mappings,
            status,
        ],
        block_dim=256,
        device=warp_device,
    )
    centre = fields.astype(float) - step * gradients.astype(float) / diagonal[:, None]
    expected = np.maximum(centre, 0)
    for voxel in range(voxels):
        if expected[:, voxel].sum() <= 1:
            continue
        # Exhaust every active support independently of breakpoint sorting.
        for mask in range(1, 1 << materials):
            active = np.array([(mask >> index) & 1 for index in range(materials)], dtype=bool)
            threshold = (centre[active, voxel].sum() - 1) / (1 / diagonal[active]).sum()
            point = centre[:, voxel] - threshold / diagonal
            if (point[active] >= -1e-13).all() and (point[~active] <= 1e-13).all():
                expected[:, voxel] = np.maximum(point, 0)
                break
        else:
            raise AssertionError("no reference active set")
    actual = trial.numpy().reshape(materials, voxels)
    np.testing.assert_allclose(actual, expected, atol=3e-8, rtol=5e-8)
    assert status.numpy()[0] == 0
    expected_slope = np.sum(gradients.astype(float) * (actual.astype(float) - fields))
    np.testing.assert_allclose(slopes.numpy().sum(), expected_slope, rtol=2e-15)
    expected_mapping = np.max(np.abs(expected - fields) / step)
    np.testing.assert_allclose(mappings.numpy().max(), expected_mapping, rtol=2e-14)


def test_fisher_install_retains_absolute_curvature_for_initial_proposal():
    from dpt.material_preconditioning import prepare_fisher_scaling

    with cpu_test_context():
        inputs = cases.fisher_cases._inputs(precision="float64")
        solver = MaterialReconstruction(**inputs, device="cpu")
        prepared = prepare_fisher_scaling(solver)
        prepared.install(solver)
        assert solver._voxel_initial_step == prepared.diagnostics["natural_initial_step"]
        report = solver.solve()
        assert report["history"][0]["proposed_step"] == min(
            inputs["settings"].maximum_step,
            inputs["settings"].initial_step / prepared.diagnostics["normaliser"],
        )
        prepared.install(solver, initialise_step=False)
        assert solver._voxel_initial_step is None


@wp.kernel
def plugin_quadratic(
    fields: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
    differentiate: bool,
    out_loss: wp.array(dtype=wp.float64),
):
    value = wp.float64(0.0)
    for index in range(fields.shape[0]):
        point = wp.float64(fields[index])
        value += wp.float64(0.3) * point * point
        if differentiate:
            gradient[index] = wp.float32(wp.float64(gradient[index]) + wp.float64(0.6) * point)
    out_loss[0] = value


class QuadraticPenalty:
    def __call__(self, fields, *, out_loss, gradient, status, context):
        wp.launch(
            plugin_quadratic,
            dim=1,
            inputs=[
                fields,
                fields if gradient is None else gradient,
                gradient is not None,
                out_loss,
            ],
            stream=context.stream,
            record_tape=False,
            device=context.device,
        )


def test_resident_penalty_adds_loss_and_canonical_gradient_without_host_field_transfer():
    from dpt.material_preconditioning import prepare_fisher_scaling

    with cpu_test_context():
        inputs = cases._inputs("float64")
        base = MaterialReconstruction(**inputs)
        expected_loss = base.evaluate()
        expected_gradient = base.gradient.numpy().astype(float)
        custom = MaterialReconstruction(**inputs, penalty_operator=QuadraticPenalty())
        actual_loss = custom.evaluate()
        fields = inputs["initial_fractions"].reshape(-1).astype(float)
        np.testing.assert_allclose(
            actual_loss - expected_loss, 0.3 * np.dot(fields, fields), atol=1e-13
        )
        np.testing.assert_allclose(
            custom.gradient.numpy(), expected_gradient + 0.6 * fields, rtol=1e-7
        )
        before = custom.gradient.numpy().copy()
        wp.copy(custom.trial, custom.fractions)
        custom.evaluate(gradient=False, trial=True)
        np.testing.assert_array_equal(custom.gradient.numpy(), before)
        with pytest.raises(ContractError, match="curvature bound"):
            prepare_fisher_scaling(custom)


def test_repaired_checked_failure_does_not_erase_a_new_unchecked_failure():
    from dpt.contracts import NumericalError, TrialDomainError

    with cpu_test_context():
        solver = MaterialReconstruction(**cases._inputs())
        original = solver.fractions.numpy().copy()
        solver.fractions.fill_(2)
        with pytest.raises(TrialDomainError):
            solver.evaluate()
        solver.fractions.assign(original)
        solver.views[-1].accepted.projection._status.fill_(8)
        with pytest.raises(NumericalError):
            solver.evaluate()
        assert np.isfinite(solver.evaluate())


def test_optional_path_cache_preserves_the_same_cpu_trajectory():
    results = []
    with cpu_test_context():
        for cache in (False, True):
            inputs = cases._inputs("float64")
            inputs["settings"] = replace(inputs["settings"], cache_accepted_forward=cache)
            solver = MaterialReconstruction(**inputs)
            report = solver.solve()
            results.append((solver.fractions_numpy().copy(), report))
            assert report["cache_accepted_forward"] is cache
            if not cache:
                assert report["cached_path_bytes"] == 0
        np.testing.assert_array_equal(results[0][0], results[1][0])
        assert results[0][1]["final_objective"] == results[1][1]["final_objective"]


@pytest.mark.parametrize("penalty", ["quadratic", "huber"])
def test_tiled_physical_roughness_matches_independent_voxel_fluxes(penalty, warp_device: str):
    from dpt.kernels.material_reconstruction import regularise_tiled

    fields = np.random.default_rng(261).uniform(0.01, 0.3, (2, 2, 3, 43)).astype(np.float32)
    coefficients = np.array(((0.3, 0.7, 0.9), (1.1, 0.1, 0.8)), dtype=float)
    deltas = np.array((0.013, 0.037, 0.023))
    gradient = np.zeros_like(fields, dtype=float)
    expected_loss = 0.0
    for axis, component in zip((3, 2, 1), range(3), strict=True):
        left = [slice(None)] * 4
        right = [slice(None)] * 4
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        difference = fields[tuple(left)].astype(float) - fields[tuple(right)].astype(float)
        coefficient = coefficients[:, component, None, None, None]
        flux = coefficient * difference
        potential = 0.5 * coefficient * difference**2
        if penalty == "huber":
            delta = deltas[component]
            flux = coefficient * np.clip(difference, -delta, delta)
            absolute = np.abs(difference)
            potential = coefficient * np.where(
                absolute <= delta, 0.5 * difference**2, delta * (absolute - 0.5 * delta)
            )
        expected_loss += potential.sum()
        gradient[tuple(left)] += flux
        gradient[tuple(right)] -= flux
    field_device = wp.array(fields.reshape(-1), dtype=wp.float32, device=warp_device)
    gradient_device = wp.zeros(fields.size, dtype=wp.float32, device=warp_device)
    coefficient_device = wp.array(coefficients.reshape(-1), dtype=wp.float64, device=warp_device)
    blocks = (fields.size + 255) // 256
    partials = wp.empty(blocks, dtype=wp.float64, device=warp_device)
    status = wp.zeros(1, dtype=wp.int32, device=warp_device)
    wp.launch_tiled(
        regularise_tiled,
        dim=blocks,
        inputs=[
            fields.size,
            field_device,
            43,
            3,
            2,
            coefficient_device,
            penalty == "huber",
            wp.vec3d(*deltas),
            True,
            gradient_device,
            partials,
            status,
        ],
        block_dim=256,
        device=warp_device,
    )
    np.testing.assert_allclose(partials.numpy().sum(), expected_loss, rtol=3e-14)
    np.testing.assert_allclose(gradient_device.numpy(), gradient.reshape(-1), atol=2e-8, rtol=7e-8)
    assert status.numpy()[0] == 0


@pytest.mark.parametrize(
    "step_selection,acceleration,window",
    [
        ("geometric", "none", 1),
        ("bb", "none", 1),
        ("geometric", "inertial", 1),
        ("bb", "inertial", 1),
        ("bb", "none", 4),
    ],
)
def test_material_npz_checkpoint_preserves_bb_inertia_and_nonmonotone_trajectory(
    tmp_path, step_selection, acceleration, window
):
    from dpt.material_reconstruction import MaterialReconstructionCheckpoint

    with cpu_test_context():
        inputs = cases._inputs("float64")
        selected = replace(
            inputs["settings"],
            iterations=5,
            step_selection=step_selection,
            acceleration=acceleration,
            nonmonotone_window=window,
        )
        full = MaterialReconstruction(**{**inputs, "settings": selected})
        full_report = full.solve()
        first = MaterialReconstruction(**{**inputs, "settings": replace(selected, iterations=2)})
        first.solve(objective_id="material-fixture-v1")
        saved = first.checkpoint()
        assert not saved.fields.flags.writeable
        path = tmp_path / "material.npz"
        saved.save(path)
        restored = MaterialReconstructionCheckpoint.read(path)
        resumed = MaterialReconstruction(**{**inputs, "settings": selected})
        report = resumed.solve(checkpoint=restored, objective_id="material-fixture-v1")
        np.testing.assert_array_equal(resumed.fractions_numpy(), full.fractions_numpy())
        assert report["final_objective"] == full_report["final_objective"]
        assert [row["objective"] for row in report["history"]] == [
            row["objective"] for row in full_report["history"]
        ]
        assert report["accepted_steps"] == full_report["accepted_steps"]
        assert report["evaluations"] == full_report["evaluations"] + 1


def test_material_checkpoint_rejects_changed_model_identity_and_numerical_policy():
    with cpu_test_context():
        inputs = cases._inputs("float64")
        first = MaterialReconstruction(**inputs)
        first.solve(objective_id="material-v1")
        saved = first.checkpoint()
        next_inputs = {**inputs, "settings": replace(inputs["settings"], iterations=5)}
        with pytest.raises(ContractError, match="identity"):
            MaterialReconstruction(**next_inputs).solve(
                checkpoint=saved, objective_id="material-v2"
            )
        with pytest.raises(ContractError, match="policy"):
            MaterialReconstruction(
                **{**next_inputs, "settings": replace(next_inputs["settings"], armijo=0.1)}
            ).solve(checkpoint=saved, objective_id="material-v1")
        changed = tuple(
            replace(view, counts=np.ascontiguousarray(view.counts + 1)) for view in inputs["views"]
        )
        with pytest.raises(ContractError, match="objective"):
            MaterialReconstruction(**{**next_inputs, "views": changed}).solve(
                checkpoint=saved, objective_id="material-v1"
            )


def test_material_graph_mode_rejects_cpu_preparation_explicitly():
    with cpu_test_context():
        inputs = cases._inputs("float64")
        with pytest.raises(ContractError, match="requires CUDA"):
            MaterialReconstruction(
                **{**inputs, "settings": replace(inputs["settings"], execution="graph")}
            )


def test_material_checkpoint_rejects_missing_secants_and_invalidates_changed_metric():
    with cpu_test_context():
        inputs = cases._inputs("float64")
        inputs["settings"] = replace(inputs["settings"], step_selection="bb")
        solver = MaterialReconstruction(**inputs)
        solver.solve(objective_id="fixed-model")
        saved = solver.checkpoint()
        malformed = replace(saved, secant_gradient=None)
        with pytest.raises(ContractError, match="missing required algorithm"):
            MaterialReconstruction(**inputs).solve(checkpoint=malformed, objective_id="fixed-model")
        solver.set_voxel_metric_scale(np.ones(solver.grid.shape, dtype=np.float32))
        with pytest.raises(ContractError, match="solve an accepted state"):
            solver.checkpoint()
