"""Portable objective precision and numerical/domain exception regressions."""

import importlib
import math
from contextlib import nullcontext
from decimal import Decimal, localcontext
from typing import Any, Literal

import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import NumericalError, TrialDomainError
from dpt.objectives import (
    ObjectiveSpec,
    evaluate_objective,
    evaluate_primary_objective,
    prepare_objective,
)
from dpt.validation.objectives import scalar_objective

np: Any = importlib.import_module("numpy")
wp: Any = importlib.import_module("warp")
approx: Any = pytest.approx  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def test_retained_fp64_poisson_loss_matches_decimal_across_cancellation_seams(
    warp_device: str,
) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    ratios = [
        math.nextafter(0.001, 0.0),
        0.001,
        math.nextafter(0.001, math.inf),
        0.0010001,
        math.nextafter(0.25, 0.0),
        0.25,
        math.nextafter(0.25, math.inf),
        *np.geomspace(1e-9, 0.99, 96).tolist(),
    ]
    with scope:
        workspace = prepare_objective(
            ObjectiveSpec(kind="poisson", domain="counts", precision="float64"),
            max_pixels=1,
            device=warp_device,
        )
        prediction = wp.empty(1, dtype=wp.float64, device=warp_device)
        observation = wp.empty(1, dtype=wp.float32, device=warp_device)
        loss = wp.empty(1, dtype=wp.float64, device=warp_device)
        seed = wp.empty(1, dtype=wp.float64, device=warp_device)
        for observed in (1.0, float(2**20), float(np.float32(1e20))):
            observation.fill_(observed)
            for ratio in ratios:
                for sign in (-1.0, 1.0):
                    predicted = observed * (1.0 + sign * ratio)
                    prediction.fill_(predicted)
                    evaluate_objective(
                        prediction,
                        observation,
                        out_loss=loss,
                        out_cotangent=seed,
                        workspace=workspace,
                    )
                    value, derivative = scalar_objective(predicted, observed, poisson=True)
                    assert float(loss.numpy()[0]) == approx(float(value), rel=2e-15, abs=0)
                    assert float(seed.numpy()[0]) == approx(float(derivative), rel=4e-16, abs=0)


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("size", [255, 256, 257, 65537])
def test_multi_pixel_poisson_loss_and_seed_match_decimal_at_tree_boundaries(
    warp_device: str, precision: Literal["float32", "float64"], size: int
) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        workspace = prepare_objective(
            ObjectiveSpec(kind="poisson", domain="counts", precision=precision),
            max_pixels=size,
            device=warp_device,
        )
        dtype = wp.float64 if precision == "float64" else wp.float32
        prediction = wp.array(
            [float(1 + i % 13) for i in range(size)], dtype=dtype, device=warp_device
        )
        observation = wp.array(
            [float(i % 9) for i in range(size)], dtype=wp.float32, device=warp_device
        )
        loss = wp.empty(1, dtype=wp.float64, device=warp_device)
        seed = wp.empty(size, dtype=dtype, device=warp_device)
        evaluate_objective(
            prediction, observation, out_loss=loss, out_cotangent=seed, workspace=workspace
        )
        # Inputs repeat every lcm(13, 9)=117 pixels. The Decimal terms are
        # independently evaluated at exact binary input values, not by Warp.
        references = [
            scalar_objective(float(1 + i % 13), float(i % 9), poisson=True) for i in range(117)
        ]
        expected_loss = math.fsum(float(references[i % 117][0]) for i in range(size))
        expected_seed = np.array(
            [float(references[i % 117][1]) for i in range(size)],
            dtype=np.float64 if precision == "float64" else np.float32,
        )
        assert float(loss.numpy()[0]) == approx(expected_loss, rel=2e-15)
        np.testing.assert_allclose(
            seed.numpy(), expected_seed, rtol=2e-7 if precision == "float32" else 4e-16, atol=0
        )


@pytest.mark.parametrize("poisson", [False, True])
@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_predictions_are_numerical_failures(
    warp_device: str, poisson: bool, checked: bool, value: float
) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        workspace = prepare_objective(
            ObjectiveSpec(kind="poisson" if poisson else "squared_error", domain="counts"),
            max_pixels=1,
            device=warp_device,
        )
        prediction = wp.array([value], dtype=wp.float32, device=warp_device)
        observation = wp.ones(1, dtype=wp.float32, device=warp_device)
        loss = wp.full(1, 17.0, dtype=wp.float64, device=warp_device)
        seed = wp.full(1, 23.0, dtype=wp.float32, device=warp_device)
        with pytest.raises(NumericalError) as error:
            evaluate_objective(
                prediction,
                observation,
                out_loss=loss,
                out_cotangent=seed,
                workspace=workspace,
                validate=checked,
            )
            workspace.check_status()
        assert not isinstance(error.value, TrialDomainError)
        if checked:
            assert float(loss.numpy()[0]) == 17.0
            assert float(seed.numpy()[0]) == 23.0


@pytest.mark.parametrize(
    "kind,domain",
    [("poisson", "counts"), ("squared_error", "counts"), ("squared_error", "log_transmission")],
)
@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_primary_depths_are_numerical_failures(
    warp_device: str,
    kind: Literal["poisson", "squared_error"],
    domain: Literal["counts", "log_transmission"],
    checked: bool,
    value: float,
) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        workspace = prepare_objective(
            ObjectiveSpec(kind=kind, domain=domain), max_pixels=1, device=warp_device
        )
        depth = wp.array([value], dtype=wp.float64, device=warp_device)
        observation = wp.ones(1, dtype=wp.float32, device=warp_device)
        prediction = wp.empty(1, dtype=wp.float64, device=warp_device)
        loss = wp.empty(1, dtype=wp.float64, device=warp_device)
        seed = wp.empty(1, dtype=wp.float64, device=warp_device)
        with pytest.raises(NumericalError) as error:
            evaluate_primary_objective(
                depth,
                observation,
                out_prediction=prediction,
                out_loss=loss,
                out_depth_cotangent=seed,
                workspace=workspace,
                validate=checked,
            )
            workspace.check_status()
        assert not isinstance(error.value, TrialDomainError)


def test_primary_poisson_loss_uses_stable_half_deviance_away_from_subnormal_tails(
    warp_device: str,
) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        workspace = prepare_objective(
            ObjectiveSpec(kind="poisson", domain="counts"), max_pixels=1, device=warp_device
        )
        depth = wp.empty(1, dtype=wp.float64, device=warp_device)
        observation = wp.empty(1, dtype=wp.float32, device=warp_device)
        prediction = wp.empty(1, dtype=wp.float64, device=warp_device)
        loss = wp.empty(1, dtype=wp.float64, device=warp_device)
        seed = wp.empty(1, dtype=wp.float64, device=warp_device)
        for observed in (1.0, float(2**20), float(np.float32(1e20))):
            observation.fill_(observed)
            beam = 2.0 * observed
            for ratio in (0.001, 0.0010001, 0.249999, 0.25, 0.250001, 0.5, 0.99):
                for sign in (-1.0, 1.0):
                    depth.fill_(math.log(2.0) - math.log1p(sign * ratio))
                    evaluate_primary_objective(
                        depth,
                        observation,
                        open_beam=beam,
                        out_prediction=prediction,
                        out_loss=loss,
                        out_depth_cotangent=seed,
                        workspace=workspace,
                    )
                    predicted = float(prediction.numpy()[0])
                    reference, _ = scalar_objective(predicted, observed, poisson=True)
                    assert float(loss.numpy()[0]) == approx(float(reference), rel=3e-15)
                    assert float(seed.numpy()[0]) == observed - predicted


@pytest.mark.parametrize("poisson", [False, True])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_multi_pixel_weighted_masked_objective_matches_decimal(
    warp_device: str, poisson: bool, reduction: Literal["sum", "mean"]
) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        size = 257
        workspace = prepare_objective(
            ObjectiveSpec(
                kind="poisson" if poisson else "squared_error",
                domain="counts",
                precision="float64",
                weighted=True,
                masked=True,
                reduction=reduction,
            ),
            max_pixels=size,
            device=warp_device,
        )
        observations = [float(i % 9) for i in range(size)]
        predictions = [value + 0.125 for value in observations]
        weights = [0.0 if i % 3 == 0 else 0.25 if i % 3 == 1 else 2.0 for i in range(size)]
        mask = [0 if i % 4 == 0 else 1 for i in range(size)]
        denominator = size if reduction == "mean" else 1
        values = [
            scalar_objective(p, y, poisson=poisson)
            for p, y in zip(predictions, observations, strict=True)
        ]
        expected_loss = math.fsum(
            float(value[0]) * weight / denominator
            for value, weight, active in zip(values, weights, mask, strict=True)
            if active
        )
        expected_seed = [
            float(value[1]) * weight / denominator if active else 0.0
            for value, weight, active in zip(values, weights, mask, strict=True)
        ]
        for i, active in enumerate(mask):
            if not active:
                observations[i] = predictions[i] = weights[i] = math.nan
        prediction = wp.array(predictions, dtype=wp.float64, device=warp_device)
        observation = wp.array(observations, dtype=wp.float32, device=warp_device)
        weight = wp.array(weights, dtype=wp.float32, device=warp_device)
        valid = wp.array(mask, dtype=wp.uint8, device=warp_device)
        loss = wp.empty(1, dtype=wp.float64, device=warp_device)
        seed = wp.empty(size, dtype=wp.float64, device=warp_device)
        evaluate_objective(
            prediction,
            observation,
            weights=weight,
            valid=valid,
            out_loss=loss,
            out_cotangent=seed,
            workspace=workspace,
        )
        assert float(loss.numpy()[0]) == approx(expected_loss, rel=2e-15)
        np.testing.assert_allclose(seed.numpy(), expected_seed, rtol=4e-16, atol=0)


@pytest.mark.parametrize(
    "kind,domain",
    [("poisson", "counts"), ("squared_error", "counts"), ("squared_error", "log_transmission")],
)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_multi_pixel_primary_weights_mask_loss_and_depth_seed_match_decimal(
    warp_device: str,
    kind: Literal["poisson", "squared_error"],
    domain: Literal["counts", "log_transmission"],
    reduction: Literal["sum", "mean"],
) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        size = 257
        depths = [float(i % 13) / 8 for i in range(size)]
        observations = [float(i % 9) for i in range(size)]
        weights = [0.0 if i % 3 == 0 else 0.25 if i % 3 == 1 else 2.0 for i in range(size)]
        mask = [0 if i % 4 == 0 else 1 for i in range(size)]
        workspace = prepare_objective(
            ObjectiveSpec(
                kind=kind, domain=domain, weighted=True, masked=True, reduction=reduction
            ),
            max_pixels=size,
            device=warp_device,
        )
        depth = wp.array(depths, dtype=wp.float64, device=warp_device)
        observation = wp.array(
            [
                value if active else math.nan
                for value, active in zip(observations, mask, strict=True)
            ],
            dtype=wp.float32,
            device=warp_device,
        )
        weight = wp.array(
            [value if active else math.nan for value, active in zip(weights, mask, strict=True)],
            dtype=wp.float32,
            device=warp_device,
        )
        valid = wp.array(mask, dtype=wp.uint8, device=warp_device)
        prediction = wp.empty(size, dtype=wp.float64, device=warp_device)
        loss = wp.empty(1, dtype=wp.float64, device=warp_device)
        seed = wp.empty(size, dtype=wp.float64, device=warp_device)
        evaluate_primary_objective(
            depth,
            observation,
            open_beam=10.0,
            weights=weight,
            valid=valid,
            out_prediction=prediction,
            out_loss=loss,
            out_depth_cotangent=seed,
            workspace=workspace,
        )
        predictions = prediction.numpy()
        denominator = size if reduction == "mean" else 1
        reference_loss = []
        reference_seed = []
        for p, y, w, active in zip(predictions, observations, weights, mask, strict=True):
            value, derivative = scalar_objective(float(p), y, poisson=kind == "poisson")
            reference_loss.append(float(value) * w / denominator if active else 0.0)
            derivative = y - float(p) if kind == "poisson" else -float(derivative)
            if kind != "poisson" and domain == "counts":
                derivative *= float(p)
            reference_seed.append(derivative * w / denominator if active else 0.0)
        assert float(loss.numpy()[0]) == approx(math.fsum(reference_loss), rel=2e-15)
        np.testing.assert_allclose(seed.numpy(), reference_seed, rtol=4e-16, atol=0)


@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("value", [-1.0, 0.0])
def test_finite_inadmissible_poisson_mean_remains_rejectable(
    warp_device: str, checked: bool, value: float
) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        workspace = prepare_objective(
            ObjectiveSpec(kind="poisson", domain="counts"), max_pixels=1, device=warp_device
        )
        prediction = wp.array([value], dtype=wp.float32, device=warp_device)
        observation = wp.ones(1, dtype=wp.float32, device=warp_device)
        loss = wp.empty(1, dtype=wp.float64, device=warp_device)
        with pytest.raises(TrialDomainError):
            evaluate_objective(
                prediction, observation, out_loss=loss, workspace=workspace, validate=checked
            )
            workspace.check_status()


@pytest.mark.parametrize("optical,beam", [(745.0, 1.0), (750.0, 1e6), (1000.0, 1e300)])
def test_primary_subnormal_count_tail_keeps_log_mean_and_depth_seed(
    warp_device: str, optical: float, beam: float
) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        workspace = prepare_objective(
            ObjectiveSpec(kind="poisson", domain="counts"), max_pixels=1, device=warp_device
        )
        depth = wp.array([optical], dtype=wp.float64, device=warp_device)
        observation = wp.ones(1, dtype=wp.float32, device=warp_device)
        prediction = wp.empty(1, dtype=wp.float64, device=warp_device)
        loss = wp.empty(1, dtype=wp.float64, device=warp_device)
        seed = wp.empty(1, dtype=wp.float64, device=warp_device)
        evaluate_primary_objective(
            depth,
            observation,
            open_beam=beam,
            out_prediction=prediction,
            out_loss=loss,
            out_depth_cotangent=seed,
            workspace=workspace,
        )
        with localcontext() as context:
            context.prec = 100
            mean = Decimal.from_float(beam) * (-Decimal.from_float(optical)).exp()
            expected_loss = float(mean - 1 - mean.ln())
        assert float(prediction.numpy()[0]) == approx(float(mean), rel=2e-13, abs=0)
        assert float(loss.numpy()[0]) == approx(expected_loss, rel=2e-14)
        assert float(seed.numpy()[0]) == 1.0


@pytest.mark.parametrize("beam", [1.0, 1e6, 1e300])
def test_primary_count_tail_matches_decimal_across_binary64_underflow_boundaries(
    warp_device: str, beam: float
) -> None:
    minimum_normal_depth = -math.log(float.fromhex("0x1p-1022"))
    zero_rounding_depth = 1075 * math.log(2.0)
    optical_depths = [
        math.nextafter(700.0, 0.0),
        700.0,
        math.nextafter(700.0, math.inf),
        math.nextafter(minimum_normal_depth, 0.0),
        minimum_normal_depth,
        math.nextafter(minimum_normal_depth, math.inf),
        710.0,
        720.0,
        740.0,
        744.0,
        744.5,
        745.0,
        745.1,
        math.nextafter(zero_rounding_depth, 0.0),
        zero_rounding_depth,
        math.nextafter(zero_rounding_depth, math.inf),
        746.0,
        750.0,
        1000.0,
    ]
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        size = len(optical_depths)
        workspace = prepare_objective(
            ObjectiveSpec(kind="poisson", domain="counts"), max_pixels=size, device=warp_device
        )
        depth = wp.array(optical_depths, dtype=wp.float64, device=warp_device)
        observation = wp.zeros(size, dtype=wp.float32, device=warp_device)
        prediction = wp.empty(size, dtype=wp.float64, device=warp_device)
        loss = wp.empty(1, dtype=wp.float64, device=warp_device)
        seed = wp.empty(size, dtype=wp.float64, device=warp_device)
        evaluate_primary_objective(
            depth,
            observation,
            open_beam=beam,
            out_prediction=prediction,
            out_loss=loss,
            out_depth_cotangent=seed,
            workspace=workspace,
        )
        with localcontext() as context:
            context.prec = 100
            expected = [
                float(Decimal.from_float(beam) * (-Decimal.from_float(value)).exp())
                for value in optical_depths
            ]
        actual = prediction.numpy()
        for index, mean in enumerate(expected):
            if mean < float.fromhex("0x1p-1022"):
                assert float(actual[index]) == mean
            else:
                assert float(actual[index]) == approx(mean, rel=2e-15, abs=0)
        assert float(loss.numpy()[0]) == approx(math.fsum(expected), rel=3e-15, abs=0)
        for index, mean in enumerate(expected):
            assert float(seed.numpy()[index]) == approx(-mean, rel=3e-15, abs=0)


def test_unrepresentable_prediction_seed_is_a_numerical_failure(warp_device: str) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        workspace = prepare_objective(
            ObjectiveSpec(kind="poisson", domain="counts", precision="float64"),
            max_pixels=1,
            device=warp_device,
        )
        prediction = wp.array([1e-310], dtype=wp.float64, device=warp_device)
        observation = wp.ones(1, dtype=wp.float32, device=warp_device)
        loss = wp.empty(1, dtype=wp.float64, device=warp_device)
        seed = wp.empty(1, dtype=wp.float64, device=warp_device)
        evaluate_objective(prediction, observation, out_loss=loss, workspace=workspace)
        reference, _ = scalar_objective(1e-310, 1.0, poisson=True)
        assert float(loss.numpy()[0]) == approx(float(reference), rel=2e-15)
        with pytest.raises(NumericalError) as error:
            evaluate_objective(
                prediction, observation, out_loss=loss, out_cotangent=seed, workspace=workspace
            )
        assert not isinstance(error.value, TrialDomainError)
