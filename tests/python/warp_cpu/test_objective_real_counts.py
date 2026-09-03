"""Explicit real-count observations retain the existing half-deviance algebra."""

import importlib
import math
from decimal import Decimal, localcontext
from typing import Any, Literal

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError
from dpt.geometry import DetectorGeometry
from dpt.objectives import (
    ObjectiveSpec,
    evaluate_objective,
    evaluate_primary_objective,
    prepare_objective,
)
from dpt.recovery import PoseChart, PrimaryPoseEvaluator, PrimaryPoseProblem
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


@pytest.fixture(autouse=True)
def portable_context():
    with cpu_test_context():
        yield


def _decimal_reference(predicted: float, observed: float) -> tuple[Decimal, Decimal]:
    with localcontext() as context:
        context.prec = 100
        p, y = Decimal.from_float(predicted), Decimal.from_float(observed)
        if y == 0:
            return p, Decimal(1)
        return p - y + y * (y / p).ln(), Decimal(1) - y / p


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("observed", [880000.5, 880000.53])
def test_checked_real_observations_match_decimal_after_actual_fp32_storage(
    precision: Literal["float32", "float64"], observed: float
) -> None:
    workspace = prepare_objective(
        ObjectiveSpec(kind="poisson", domain="counts", precision=precision, integer_counts=False),
        max_pixels=1,
        device="cpu",
    )
    dtype = wp.float64 if precision == "float64" else wp.float32
    prediction = wp.array([900000.0], dtype=dtype, device="cpu")
    observation = wp.array([observed], dtype=wp.float32, device="cpu")
    loss = wp.empty(1, dtype=wp.float64, device="cpu")
    seed = wp.empty(1, dtype=dtype, device="cpu")
    workspace.validate_observation(observation)
    evaluate_objective(
        prediction, observation, out_loss=loss, out_cotangent=seed, workspace=workspace
    )
    stored_observed = float(observation.numpy()[0])
    assert stored_observed == float(np.float32(observed))
    expected_loss, expected_seed = _decimal_reference(float(prediction.numpy()[0]), stored_observed)
    assert float(loss.numpy()[0]) == pytest.approx(float(expected_loss), rel=2e-12)
    assert float(seed.numpy()[0]) == pytest.approx(
        float(expected_seed), rel=2e-7 if precision == "float32" else 2e-13
    )


@pytest.mark.parametrize("entrypoint", ["fixed", "objective", "primary"])
@pytest.mark.parametrize("observed", [-0.5, float("nan"), float("inf")])
def test_real_observation_policy_still_rejects_invalid_unmasked_data(
    entrypoint: str, observed: float
) -> None:
    workspace = prepare_objective(
        ObjectiveSpec(kind="poisson", domain="counts", integer_counts=False),
        max_pixels=1,
        device="cpu",
    )
    observation = wp.array([observed], dtype=wp.float32, device="cpu")
    loss = wp.full(1, 17.0, dtype=wp.float64, device="cpu")
    with pytest.raises(ContractError, match="finite and nonnegative"):
        if entrypoint == "fixed":
            workspace.validate_observation(observation)
        elif entrypoint == "objective":
            evaluate_objective(
                wp.ones(1, dtype=wp.float32, device="cpu"),
                observation,
                out_loss=loss,
                workspace=workspace,
            )
        else:
            evaluate_primary_objective(
                wp.zeros(1, dtype=wp.float64, device="cpu"),
                observation,
                out_prediction=wp.empty(1, dtype=wp.float64, device="cpu"),
                out_loss=loss,
                out_depth_cotangent=wp.empty(1, dtype=wp.float64, device="cpu"),
                workspace=workspace,
            )
    assert float(loss.numpy()[0]) == 17.0


@pytest.mark.parametrize("entrypoint", ["fixed", "objective", "primary"])
def test_integer_observations_remain_the_default_and_diagnose_fractional_counts(entrypoint: str):
    spec = ObjectiveSpec(kind="poisson", domain="counts")
    assert spec.integer_counts is True
    workspace = prepare_objective(spec, max_pixels=1, device="cpu")
    observation = wp.array([880000.5], dtype=wp.float32, device="cpu")
    loss = wp.full(1, 17.0, dtype=wp.float64, device="cpu")
    with pytest.raises(ContractError, match="nonnegative integer counts"):
        if entrypoint == "fixed":
            workspace.validate_observation(observation)
        elif entrypoint == "objective":
            evaluate_objective(
                wp.full(1, 900000.0, dtype=wp.float32, device="cpu"),
                observation,
                out_loss=loss,
                workspace=workspace,
            )
        else:
            evaluate_primary_objective(
                wp.zeros(1, dtype=wp.float64, device="cpu"),
                observation,
                open_beam=900000.0,
                out_prediction=wp.empty(1, dtype=wp.float64, device="cpu"),
                out_loss=loss,
                out_depth_cotangent=wp.empty(1, dtype=wp.float64, device="cpu"),
                workspace=workspace,
            )
    assert float(loss.numpy()[0]) == 17.0


@pytest.mark.parametrize("flag", [None, 0, 1, "false", [], np.bool_(False)])
def test_integer_count_policy_requires_an_explicit_python_bool(flag: Any):
    with pytest.raises(ContractError, match="integer_counts must be a boolean"):
        ObjectiveSpec(kind="poisson", domain="counts", integer_counts=flag)


def test_new_policy_follows_all_existing_positional_objective_arguments():
    assert ObjectiveSpec("poisson", "counts", "sum", False, False, "float64").integer_counts is True
    assert (
        ObjectiveSpec("poisson", "counts", "sum", False, False, "float64", False).integer_counts
        is False
    )


@pytest.mark.parametrize("integer_counts", [False, True])
def test_unchecked_fused_primary_checks_the_selected_fixed_observation_rule(integer_counts: bool):
    workspace = prepare_objective(
        ObjectiveSpec(kind="poisson", domain="counts", integer_counts=integer_counts),
        max_pixels=1,
        device="cpu",
    )
    observation = wp.array([880000.5], dtype=wp.float32, device="cpu")
    loss = wp.empty(1, dtype=wp.float64, device="cpu")
    seed = wp.empty(1, dtype=wp.float64, device="cpu")
    evaluate_primary_objective(
        wp.zeros(1, dtype=wp.float64, device="cpu"),
        observation,
        open_beam=900000.0,
        out_prediction=wp.empty(1, dtype=wp.float64, device="cpu"),
        out_loss=loss,
        out_depth_cotangent=seed,
        workspace=workspace,
        validate=False,
    )
    if integer_counts:
        with pytest.raises(ContractError, match="integer counts"):
            workspace.check_status()
    else:
        workspace.check_status()
        expected_loss, _ = _decimal_reference(900000.0, 880000.5)
        assert float(loss.numpy()[0]) == pytest.approx(float(expected_loss), rel=2e-12)
        assert float(seed.numpy()[0]) == -19999.5


def test_masked_weighted_real_observations_keep_zero_skipped_seeds_and_pixel_mean():
    workspace = prepare_objective(
        ObjectiveSpec(
            kind="poisson",
            domain="counts",
            integer_counts=False,
            weighted=True,
            masked=True,
            reduction="mean",
            precision="float64",
        ),
        max_pixels=4,
        device="cpu",
    )
    predictions = [900000.0, float("nan"), 2.0, 12.0]
    observations = [880000.5, float("nan"), 0.0, 12.5]
    weights = [1.0, float("nan"), 2.0, 0.5]
    seed = wp.empty(4, dtype=wp.float64, device="cpu")
    loss = wp.empty(1, dtype=wp.float64, device="cpu")
    evaluate_objective(
        wp.array(predictions, dtype=wp.float64, device="cpu"),
        wp.array(observations, dtype=wp.float32, device="cpu"),
        weights=wp.array(weights, dtype=wp.float32, device="cpu"),
        valid=wp.array([1, 0, 1, 1], dtype=wp.uint8, device="cpu"),
        out_loss=loss,
        out_cotangent=seed,
        workspace=workspace,
    )
    expected_loss = Decimal(0)
    expected_seed = np.zeros(4)
    for index in (0, 2, 3):
        value, derivative = _decimal_reference(predictions[index], observations[index])
        expected_loss += value * Decimal.from_float(weights[index] / 4)
        expected_seed[index] = float(derivative) * weights[index] / 4
    assert float(loss.numpy()[0]) == pytest.approx(float(expected_loss), rel=2e-12)
    np.testing.assert_allclose(seed.numpy(), expected_seed, rtol=2e-13, atol=0)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_primary_pose_evaluator_accepts_declared_noiseless_real_means(
    precision: Literal["float32", "float64"],
) -> None:
    attenuation = wp.array([0.05], dtype=wp.float32, device="cpu")
    beam = 1e6
    expected_mean = beam * math.exp(-2 * float(attenuation.numpy()[0]))
    observation = wp.array([expected_mean], dtype=wp.float32, device="cpu")
    assert float(observation.numpy()[0]) % 1 != 0
    problem = PrimaryPoseProblem(
        GridSpec((1, 1, 1), (2.0, 1.0, 1.0)),
        DetectorGeometry((-20, 0, 0), (20, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1), (1, 1)),
        attenuation,
        observation,
        objective=ObjectiveSpec(kind="poisson", domain="counts", integer_counts=False),
        open_beam=beam,
        precision=precision,
    )
    evaluator = PrimaryPoseEvaluator(problem, PoseChart(), device="cpu")
    result = evaluator((0.0,) * 6)
    assert evaluator._inputs_checked
    assert result.loss >= 0 and math.isfinite(result.loss)
    assert all(math.isfinite(entry) for entry in result.gradient)
    predicted = float(evaluator.prediction.numpy()[0])
    expected_loss, _ = _decimal_reference(predicted, float(observation.numpy()[0]))
    assert result.loss == pytest.approx(float(expected_loss), rel=2e-12, abs=1e-20)
    assert evaluator((0.0,) * 6).loss == result.loss
