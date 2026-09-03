"""Analytic spectral-chain gradients and prepared buffer ownership."""

from __future__ import annotations

import importlib
import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.geometry import RigidTransform
from dpt.pipeline import SpectralPipeline
from dpt.projection import ProjectionSpec
from dpt.recovery import PoseChart
from dpt.spectral import SpectralInputs
from dpt.spectral_recovery import SpectralPoseEvaluator
from helpers.spectral_pipeline_inputs import make_pipeline, pose_problem

if TYPE_CHECKING:
    from dpt.pipeline import FloatArray

wp: Any = importlib.import_module("warp")


def active(array: FloatArray | None) -> FloatArray:
    assert array is not None
    return array


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("domain", ["linear", "log"])
def test_spectral_chain_matches_independent_slab_and_all_parameter_vjps(
    precision: str, shared: bool, domain: str, warp_device: str
) -> None:
    pipeline, fields, pose, factors = make_pipeline(
        warp_device, precision=precision, shared=shared, domain=domain
    )
    image = pipeline.forward(fields, pose)
    mu = factors[0].numpy().astype(np.float64)
    weights, response = (value.numpy().reshape(-1).astype(np.float64) for value in factors[1:])
    paths = np.array([0.75, 1.5])
    transmission = np.exp(-(paths @ mu))
    components = weights * response * transmission
    mean = components.sum()
    expected_image = math.log(mean) if domain == "log" else mean
    np.testing.assert_allclose(image.numpy(), [[expected_image]], rtol=2e-7)
    scale = 1.0 / mean if domain == "log" else 1.0
    seed = wp.full((1, 1), 1.0, dtype=getattr(wp, precision), device=warp_device)
    gradients = pipeline.vjp(seed)
    np.testing.assert_allclose(
        active(gradients.coefficients).numpy(),
        -paths[:, None] * components[None, :] * scale,
        rtol=2e-7,
    )
    np.testing.assert_allclose(
        active(gradients.weights).numpy().reshape(-1), response * transmission * scale, rtol=2e-7
    )
    np.testing.assert_allclose(
        active(gradients.response).numpy().reshape(-1), weights * transmission * scale, rtol=2e-7
    )
    expected_fields = np.zeros((2, 3, 3, 3), dtype=np.float64)
    expected_fields[:, 1, 1, :] = (-mu @ components * scale)[:, None]
    np.testing.assert_allclose(
        active(gradients.fields).numpy(), expected_fields, rtol=4e-7, atol=1e-8
    )
    np.testing.assert_allclose(active(gradients.pose).numpy(), 0.0, atol=1e-10)
    np.testing.assert_array_equal(seed.numpy(), [[1.0]])


def test_snapshot_preserves_reverse_after_original_buffers_are_modified(warp_device: str) -> None:
    pipeline, fields, pose, factors = make_pipeline(warp_device)
    image = pipeline.forward(fields, pose).numpy().copy()
    seed = wp.full(1, 1.0, dtype=wp.float64, device=warp_device)
    expected = active(pipeline.vjp(seed).fields).numpy().copy()
    fields.fill_(float("nan"))
    pose.fill_(float("nan"))
    for values in factors:
        values.fill_(float("nan"))
    np.testing.assert_array_equal(active(pipeline.vjp(seed).fields).numpy(), expected)
    np.testing.assert_array_equal(pipeline._image.numpy(), image)


def test_local_right_pose_gradient_agrees_with_analytic_translation_and_finite_difference(
    warp_device: str,
) -> None:
    pipeline, fields, pose, factors = make_pipeline(warp_device)
    host = fields.numpy()
    host[0] += np.array([-0.015625, 0.0, 0.015625], dtype=np.float32)[None, :, None]
    fields.assign(host)
    pipeline.forward(fields, pose)
    gradients = pipeline.vjp(wp.full(1, 1.0, dtype=wp.float64, device=warp_device))
    gradient = float(active(gradients.pose).numpy()[1])
    mu = factors[0].numpy().astype(np.float64)
    weights, response = (value.numpy().reshape(-1).astype(np.float64) for value in factors[1:])
    slope = (float(host[0, 1, 2, 1]) - float(host[0, 1, 0, 1])) / 2
    components = weights * response * np.exp(-(np.array([0.75, 1.5]) @ mu))
    analytic = 3 * slope * (mu[0] @ components)
    assert gradient == pytest.approx(analytic, rel=2e-12)
    means = []
    step = 1e-3
    for displacement in (-step, step):
        changed = wp.array(
            RigidTransform(translation_mm=(0.0, displacement, 0.0)).packed(),
            dtype=wp.float64,
            device=warp_device,
        )
        means.append(float(pipeline.forward(fields, changed).numpy()[0, 0]))
    assert gradient == pytest.approx((means[1] - means[0]) / (2 * step), rel=1e-7)


def test_prepared_factors_are_checked_once_and_reusable_results_keep_their_buffers(
    monkeypatch: pytest.MonkeyPatch, warp_device: str
) -> None:
    pipeline, fields, pose, _ = make_pipeline(warp_device)
    from dpt.spectral import SpectralWorkspace

    def repeated_fixed_scan(*_args, **_kwargs):
        raise AssertionError("fixed spectral inputs were rescanned")

    monkeypatch.setattr(SpectralWorkspace, "validate_inputs", repeated_fixed_scan)
    first = pipeline.forward(fields, pose)
    first_ptr = first.ptr
    seed = wp.full(1, 1.0, dtype=wp.float64, device=warp_device)
    field_ptr = active(pipeline.vjp(seed).fields).ptr
    fields.fill_(0.125)
    second = pipeline.forward(fields, pose)
    assert second.ptr == first_ptr
    assert active(pipeline.vjp(seed).fields).ptr == field_ptr


def test_forward_failure_invalidates_saved_state_and_checked_retry_recovers(
    warp_device: str,
) -> None:
    pipeline, fields, pose, _ = make_pipeline(warp_device)
    pipeline.forward(fields, pose)
    fields.fill_(2.0)
    with pytest.raises(NumericalError):
        pipeline.forward(fields, pose)
    seed = wp.full(1, 1.0, dtype=wp.float64, device=warp_device)
    with pytest.raises(ContractError, match="successful forward"):
        pipeline.vjp(seed)
    fields.fill_(0.125)
    pipeline.forward(fields, pose)
    assert np.all(np.isfinite(active(pipeline.vjp(seed).fields).numpy()))


def test_cotangent_and_input_layout_contracts(warp_device: str) -> None:
    pipeline, fields, pose, _ = make_pipeline(warp_device, ray_bundle=True)
    seed = wp.full(1, 1.0, dtype=wp.float64, device=warp_device)
    with pytest.raises(ContractError, match="successful forward"):
        pipeline.vjp(seed)
    assert pipeline.forward(fields, pose).shape == (1,)
    with pytest.raises(ContractError, match="dtype"):
        pipeline.vjp(wp.ones(1, dtype=wp.float32, device=warp_device))
    with pytest.raises(ContractError, match="shape"):
        pipeline.vjp(wp.ones(2, dtype=wp.float64, device=warp_device))
    with pytest.raises(ContractError, match="invalid fixed device input"):
        pipeline.vjp(wp.full(1, float("nan"), dtype=wp.float64, device=warp_device))
    assert np.all(np.isfinite(active(pipeline.vjp(seed).fields).numpy()))
    with pytest.raises(ContractError, match="shape"):
        pipeline.forward(wp.zeros(2, dtype=wp.float32, device=warp_device), pose)
    with pytest.raises(ContractError, match="successful forward"):
        pipeline.vjp(seed)


def test_pipeline_rejects_mismatched_precision_pixels_and_inactive_paths(warp_device: str) -> None:
    pipeline, _, _, _ = make_pipeline(warp_device)
    with pytest.raises(ContractError, match="precision"):
        SpectralPipeline(pipeline.grid, pipeline.geometry, ProjectionSpec(12), pipeline.inputs)
    geometry = replace(pipeline.geometry, shape=(1, 2))
    with pytest.raises(ContractError, match="pixel count"):
        SpectralPipeline(
            pipeline.grid, geometry, ProjectionSpec(12, precision="float64"), pipeline.inputs
        )
    inactive = SpectralInputs.from_arrays(
        replace(pipeline.inputs.spec, active_paths=False),
        pipeline.inputs.coefficients,
        pipeline.inputs.weights,
        pipeline.inputs.response,
        pixels=1,
        device=warp_device,
    )
    with pytest.raises(ContractError, match="active spectral paths"):
        SpectralPipeline(
            pipeline.grid, pipeline.geometry, ProjectionSpec(12, precision="float64"), inactive
        )


def test_spectral_inputs_reject_invalid_values_before_use(warp_device: str) -> None:
    pipeline, _, _, _ = make_pipeline(warp_device)
    with pytest.raises(ContractError, match="invalid fixed device input"):
        SpectralInputs.from_arrays(
            pipeline.inputs.spec,
            wp.full(4, -0.1, dtype=wp.float32, device=warp_device),
            pipeline.inputs.weights,
            pipeline.inputs.response,
            pixels=1,
            device=warp_device,
        )
    assert pipeline.inputs.storage_bytes == 32
    with pytest.raises(TypeError, match="from_arrays"):
        SpectralInputs()


def test_graph_requests_on_cpu_fail_without_changing_eager_defaults() -> None:
    pipeline, _, _, _ = make_pipeline("cpu")
    with pytest.raises(ContractError, match="requires CUDA"):
        SpectralPipeline(
            pipeline.grid,
            pipeline.geometry,
            ProjectionSpec(12, precision="float64"),
            pipeline.inputs,
            execution="graph",
        )
    with pytest.raises(ContractError, match="requires CUDA"):
        SpectralPoseEvaluator(pose_problem("cpu"), PoseChart(), device="cpu", execution="graph")
    assert pipeline.execution == "eager"


@pytest.mark.parametrize("blurred", [False, True])
def test_multiview_spectral_evaluator_refactor_preserves_repeatable_cpu_results(
    blurred: bool,
) -> None:
    evaluator = SpectralPoseEvaluator(
        pose_problem("cpu", blurred=blurred), PoseChart(), device="cpu"
    )
    point = (0.03, -0.02, 0.01, 0.1, -0.2, 0.3, 0.05, -0.4)
    first, second = evaluator(point), evaluator(point)
    assert first == second
    assert math.isfinite(first.loss)
    assert all(math.isfinite(value) for value in first.gradient)
    # Rotation chart steps of 0.1 are 0.001 rad; they resolve FP32 image changes.
    steps = (0.04, 0.04, 0.04, 0.1, 0.1, 0.1, 0.002, 0.04)
    for index in range(len(point)):
        step = steps[index]
        plus, minus = list(point), list(point)
        plus[index] += step
        minus[index] -= step
        finite_difference = (evaluator(tuple(plus)).loss - evaluator(tuple(minus)).loss) / (
            2 * step
        )
        assert first.gradient[index] == pytest.approx(finite_difference, rel=0.02, abs=0.005)
