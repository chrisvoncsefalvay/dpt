"""CUDA objective checks against independent numerical references."""

import importlib
from decimal import Decimal
from typing import Any

import pytest

from dpt.contracts import ContractError, NumericalError, TrialDomainError
from dpt.objectives import ObjectiveSpec, evaluate_objective, prepare_objective
from dpt.validation.objectives import scalar_objective

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu
approx: Any = pytest.approx  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


@pytest.mark.parametrize("size", [0, 1, 255, 256, 257, 65537])
@pytest.mark.parametrize("poisson", [False, True])
def test_objective_and_seed_against_independent_decimal_values(size: int, poisson: bool) -> None:
    spec = ObjectiveSpec(kind="poisson" if poisson else "squared_error", domain="counts")
    workspace = prepare_objective(spec, max_pixels=65537)
    predictions = [float(1 + i % 13) for i in range(size)]
    observations = [float(i % 9) for i in range(size)]
    prediction = wp.array(predictions, dtype=wp.float32, device="cuda:0")
    observation = wp.array(observations, dtype=wp.float32, device="cuda:0")
    loss = wp.empty(1, dtype=wp.float64, device="cuda:0")
    seed = wp.empty(size, dtype=wp.float32, device="cuda:0")
    evaluate_objective(
        prediction, observation, out_loss=loss, out_cotangent=seed, workspace=workspace
    )
    expected = [
        scalar_objective(p, y, poisson=poisson)
        for p, y in zip(predictions, observations, strict=True)
    ]
    expected_loss = float(sum((entry[0] for entry in expected), Decimal(0)))
    assert float(loss.numpy()[0]) == approx(expected_loss, rel=2e-12, abs=1e-13)
    values = seed.numpy()
    for actual, (_, reference) in zip(values, expected, strict=True):
        assert float(actual) == approx(float(reference), rel=2e-7, abs=1e-30)


def test_checked_invalid_poisson_leaves_destinations_untouched() -> None:
    workspace = prepare_objective(ObjectiveSpec(kind="poisson", domain="counts"), max_pixels=1)
    prediction = wp.zeros(1, dtype=wp.float32, device="cuda:0")
    observation = wp.ones(1, dtype=wp.float32, device="cuda:0")
    loss = wp.full(1, 17.0, dtype=wp.float64, device="cuda:0")
    with pytest.raises(TrialDomainError):
        evaluate_objective(prediction, observation, out_loss=loss, workspace=workspace)
    assert float(loss.numpy()[0]) == 17.0


def test_weighted_mean_uses_pixel_count_and_has_no_seed_for_zero_weight() -> None:
    workspace = prepare_objective(ObjectiveSpec(weighted=True, reduction="mean"), max_pixels=2)

    def array(values: list[float]) -> Any:
        return wp.array(values, dtype=wp.float32, device="cuda:0")

    loss = wp.empty(1, dtype=wp.float64, device="cuda:0")
    seed = wp.empty(2, dtype=wp.float32, device="cuda:0")
    evaluate_objective(
        array([3, 5]),
        array([1, 2]),
        weights=array([0, 2]),
        out_loss=loss,
        out_cotangent=seed,
        workspace=workspace,
    )
    assert float(loss.numpy()[0]) == 4.5
    assert list(seed.numpy()) == [0, 3]


@pytest.mark.parametrize("value", [1.0, 1.0 + 2**-23, 1.0 - 2**-24, 1.001, 0.999, 2**-100, 2**100])
def test_near_agreement_and_many_decades_do_not_use_a_logarithmic_floor(value: float) -> None:
    # Convert through actual FP32 storage before constructing the oracle.
    # Each term is tested alone: a huge term cannot hide small-residual damage.
    prediction = wp.array([value], dtype=wp.float32, device="cuda:0")
    observation = wp.ones(1, dtype=wp.float32, device="cuda:0")
    workspace = prepare_objective(ObjectiveSpec(kind="poisson", domain="counts"), max_pixels=1)
    loss = wp.empty(1, dtype=wp.float64, device="cuda:0")
    evaluate_objective(prediction, observation, out_loss=loss, workspace=workspace)
    expected = sum(
        (scalar_objective(float(x), 1.0, poisson=True)[0] for x in prediction.numpy()), Decimal(0)
    )
    assert float(loss.numpy()[0]) == approx(float(expected), rel=2e-12, abs=0.0)


@pytest.mark.parametrize("poisson", [False, True])
def test_explicit_mask_skips_invalid_targets_before_arithmetic(poisson: bool) -> None:
    workspace = prepare_objective(
        ObjectiveSpec(
            kind="poisson" if poisson else "squared_error",
            domain="counts",
            weighted=True,
            masked=True,
        ),
        max_pixels=4,
    )

    def array(values: list[float]) -> Any:
        return wp.array(values, dtype=wp.float32, device="cuda:0")

    predictions = [3.0, float("nan"), 0.0, 5.0]
    observations = [1.0, float("inf"), 1.0, 2.0]
    weights = [2.0, float("nan"), float("nan"), 0.5]
    valid = wp.array([1, 0, 0, 1], dtype=wp.uint8, device="cuda:0")
    loss = wp.empty(1, dtype=wp.float64, device="cuda:0")
    seed = wp.full(4, 17.0, dtype=wp.float32, device="cuda:0")
    workspace.validate_observation(array(observations), weights=array(weights), valid=valid)
    evaluate_objective(
        array(predictions),
        array(observations),
        weights=array(weights),
        valid=valid,
        out_loss=loss,
        out_cotangent=seed,
        workspace=workspace,
    )
    entries = [scalar_objective(predictions[i], observations[i], poisson=poisson) for i in (0, 3)]
    expected = sum(float(entry[0]) * weights[i] for entry, i in zip(entries, (0, 3), strict=True))
    assert float(loss.numpy()[0]) == approx(expected, rel=2e-12)
    result = seed.numpy()
    assert result[1] == 0.0 and result[2] == 0.0
    assert float(result[0]) == approx(float(entries[0][1]) * 2, rel=2e-7)
    assert float(result[3]) == approx(float(entries[1][1]) * 0.5, rel=2e-7)
    # An identical nonfinite lane becomes an error as soon as it is observed.
    valid.assign([1, 1, 0, 1])
    with pytest.raises(ContractError):
        evaluate_objective(
            array(predictions),
            array(observations),
            weights=array(weights),
            valid=valid,
            out_loss=loss,
            workspace=workspace,
        )


def test_all_masked_tiles_and_invalid_mask_encoding() -> None:
    size = 257
    workspace = prepare_objective(ObjectiveSpec(masked=True), max_pixels=size)
    values = wp.full(size, float("nan"), dtype=wp.float32, device="cuda:0")
    valid = wp.zeros(size, dtype=wp.uint8, device="cuda:0")
    seed = wp.full(size, 7.0, dtype=wp.float32, device="cuda:0")
    loss = wp.empty(1, dtype=wp.float64, device="cuda:0")
    evaluate_objective(
        values, values, valid=valid, out_loss=loss, out_cotangent=seed, workspace=workspace
    )
    assert float(loss.numpy()[0]) == 0.0
    assert all(float(x) == 0.0 for x in seed.numpy())
    valid.fill_(2)
    with pytest.raises(ContractError):
        workspace.validate_observation(values, valid=valid)


@pytest.mark.parametrize("poisson", [False, True])
@pytest.mark.parametrize("offset", [-0.125, 0.125])
def test_retained_fp64_resolves_objective_and_seed_below_fp32_count_spacing(
    poisson: bool, offset: float
) -> None:
    """Exact mathematical inputs, not a reconstruction outcome or convergence claim."""
    observed = float(2**24)
    predicted = observed + offset
    # Both perturbations disappear at this representable binary32 observation.
    legacy = wp.array([predicted], dtype=wp.float32, device="cuda:0")
    assert float(legacy.numpy()[0]) == observed
    prediction = wp.array([predicted], dtype=wp.float64, device="cuda:0")
    observation = wp.array([observed], dtype=wp.float32, device="cuda:0")
    loss = wp.empty(1, dtype=wp.float64, device="cuda:0")
    seed = wp.empty(1, dtype=wp.float64, device="cuda:0")
    workspace = prepare_objective(
        ObjectiveSpec(
            kind="poisson" if poisson else "squared_error",
            domain="counts",
            precision="float64",
        ),
        max_pixels=1,
    )
    evaluate_objective(
        prediction, observation, out_loss=loss, out_cotangent=seed, workspace=workspace
    )
    expected_loss, expected_seed = scalar_objective(predicted, observed, poisson=poisson)
    assert float(loss.numpy()[0]) > 0
    assert float(loss.numpy()[0]) == approx(float(expected_loss), rel=2e-13, abs=0)
    assert float(seed.numpy()[0]) == approx(float(expected_seed), rel=2e-13, abs=0)
    assert float(prediction.numpy()[0]) == predicted
    assert float(observation.numpy()[0]) == observed


@pytest.mark.parametrize("size", [0, 257])
def test_retained_fp64_loss_only_uses_typed_empty_seed_and_padded_reduction(size: int) -> None:
    workspace = prepare_objective(ObjectiveSpec(precision="float64"), max_pixels=257)
    prediction = wp.full(size, 1.125, dtype=wp.float64, device="cuda:0")
    observation = wp.ones(size, dtype=wp.float32, device="cuda:0")
    loss = wp.full(1, 17.0, dtype=wp.float64, device="cuda:0")
    for _ in range(2):
        evaluate_objective(prediction, observation, out_loss=loss, workspace=workspace)
        assert float(loss.numpy()[0]) == size / 128


@pytest.mark.parametrize("poisson", [False, True])
def test_retained_fp64_mask_and_weighted_mean_keep_fixed_fp32_data(poisson: bool) -> None:
    base = float(2**20)
    predictions = [base + 0.03125, float("nan"), 1.0, base - 0.03125]
    observations = [base, float("inf"), 0.0, base]
    weights = [2.0, float("nan"), 0.0, 0.5]
    workspace = prepare_objective(
        ObjectiveSpec(
            kind="poisson" if poisson else "squared_error",
            domain="counts",
            reduction="mean",
            weighted=True,
            masked=True,
            precision="float64",
        ),
        max_pixels=4,
    )
    prediction = wp.array(predictions, dtype=wp.float64, device="cuda:0")
    observation = wp.array(observations, dtype=wp.float32, device="cuda:0")
    weight = wp.array(weights, dtype=wp.float32, device="cuda:0")
    valid = wp.array([1, 0, 1, 1], dtype=wp.uint8, device="cuda:0")
    loss = wp.empty(1, dtype=wp.float64, device="cuda:0")
    seed = wp.full(4, 17.0, dtype=wp.float64, device="cuda:0")
    workspace.validate_observation(observation, weights=weight, valid=valid)
    evaluate_objective(
        prediction,
        observation,
        weights=weight,
        valid=valid,
        out_loss=loss,
        out_cotangent=seed,
        workspace=workspace,
    )
    expected = {
        i: scalar_objective(predictions[i], observations[i], poisson=poisson) for i in (0, 3)
    }
    total = sum(float(expected[i][0]) * weights[i] / 4 for i in (0, 3))
    assert float(loss.numpy()[0]) == approx(total, rel=2e-13, abs=0)
    actual = seed.numpy()
    assert float(actual[1]) == float(actual[2]) == 0
    for i in (0, 3):
        assert float(actual[i]) == approx(float(expected[i][1]) * weights[i] / 4, rel=2e-13)


@pytest.mark.parametrize("wrong", ["prediction", "observation", "weights", "out_seed"])
def test_retained_fp64_rejects_implicit_dtype_conversion_before_writing(wrong: str) -> None:
    workspace = prepare_objective(ObjectiveSpec(weighted=True, precision="float64"), max_pixels=1)
    values = {
        name: wp.ones(
            1,
            dtype=wp.float64 if name in ("prediction", "out_seed") else wp.float32,
            device="cuda:0",
        )
        for name in ("prediction", "observation", "weights", "out_seed")
    }
    values[wrong] = wp.ones(
        1,
        dtype=wp.float32 if wrong in ("prediction", "out_seed") else wp.float64,
        device="cuda:0",
    )
    loss = wp.full(1, 17.0, dtype=wp.float64, device="cuda:0")
    with pytest.raises(ContractError, match="dtype"):
        evaluate_objective(
            values["prediction"],
            values["observation"],
            weights=values["weights"],
            out_cotangent=values["out_seed"],
            out_loss=loss,
            workspace=workspace,
        )
    assert float(loss.numpy()[0]) == 17


def test_retained_fp64_poisson_validation_and_output_overlap_preserve_destinations() -> None:
    workspace = prepare_objective(
        ObjectiveSpec(kind="poisson", domain="counts", precision="float64"), max_pixels=1
    )
    prediction = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    observation = wp.ones(1, dtype=wp.float32, device="cuda:0")
    loss = wp.full(1, 17.0, dtype=wp.float64, device="cuda:0")
    seed = wp.full(1, 23.0, dtype=wp.float64, device="cuda:0")
    with pytest.raises(TrialDomainError):
        evaluate_objective(
            prediction, observation, out_loss=loss, out_cotangent=seed, workspace=workspace
        )
    assert float(loss.numpy()[0]) == 17 and float(seed.numpy()[0]) == 23
    prediction.fill_(1)
    with pytest.raises(ContractError, match="overlaps"):
        evaluate_objective(
            prediction, observation, out_loss=loss, out_cotangent=loss, workspace=workspace
        )
    with pytest.raises(ContractError, match="overlaps"):
        evaluate_objective(
            prediction, observation, out_loss=loss, out_cotangent=prediction, workspace=workspace
        )


def test_retained_fp64_reports_unrepresentable_seed_even_when_loss_is_finite() -> None:
    workspace = prepare_objective(
        ObjectiveSpec(kind="poisson", domain="counts", precision="float64"), max_pixels=1
    )
    prediction = wp.array([1e-310], dtype=wp.float64, device="cuda:0")
    observation = wp.ones(1, dtype=wp.float32, device="cuda:0")
    loss = wp.empty(1, dtype=wp.float64, device="cuda:0")
    seed = wp.empty(1, dtype=wp.float64, device="cuda:0")
    evaluate_objective(prediction, observation, out_loss=loss, workspace=workspace)
    expected, _ = scalar_objective(float(prediction.numpy()[0]), 1.0, poisson=True)
    assert float(loss.numpy()[0]) == approx(float(expected), rel=2e-13)
    with pytest.raises(NumericalError):
        evaluate_objective(
            prediction, observation, out_loss=loss, out_cotangent=seed, workspace=workspace
        )


@pytest.mark.parametrize(
    "poisson,predicted,observed,weight_value",
    [
        (True, 1e-310, 1.0, 1e-20),
        (True, 1e-310, 1.0, 0.0),
        (False, 1e160, 0.0, 1e-30),
        (False, 1e-170, 0.0, 1e38),
        (False, 1e200, 0.0, 0.0),
    ],
)
def test_retained_fp64_weighted_results_do_not_lose_representable_values(
    poisson: bool, predicted: float, observed: float, weight_value: float
) -> None:
    """Adversarial scalar arithmetic, checked against exact-input Decimal values."""
    workspace = prepare_objective(
        ObjectiveSpec(
            kind="poisson" if poisson else "squared_error",
            domain="counts",
            weighted=True,
            precision="float64",
        ),
        max_pixels=1,
    )
    prediction = wp.array([predicted], dtype=wp.float64, device="cuda:0")
    observation = wp.array([observed], dtype=wp.float32, device="cuda:0")
    weight = wp.array([weight_value], dtype=wp.float32, device="cuda:0")
    loss = wp.empty(1, dtype=wp.float64, device="cuda:0")
    seed = wp.empty(1, dtype=wp.float64, device="cuda:0")
    evaluate_objective(
        prediction,
        observation,
        weights=weight,
        out_loss=loss,
        out_cotangent=seed,
        workspace=workspace,
    )
    value, derivative = scalar_objective(
        float(prediction.numpy()[0]), float(observation.numpy()[0]), poisson=poisson
    )
    exact_weight = Decimal.from_float(float(weight.numpy()[0]))
    assert float(loss.numpy()[0]) == approx(float(value * exact_weight), rel=2e-13, abs=0)
    assert float(seed.numpy()[0]) == approx(float(derivative * exact_weight), rel=2e-13, abs=0)


def test_retained_fp64_loss_overflow_requires_checked_status() -> None:
    workspace = prepare_objective(ObjectiveSpec(precision="float64"), max_pixels=1)
    prediction = wp.array([1e200], dtype=wp.float64, device="cuda:0")
    observation = wp.zeros(1, dtype=wp.float32, device="cuda:0")
    loss = wp.empty(1, dtype=wp.float64, device="cuda:0")
    workspace.clear_status()
    evaluate_objective(prediction, observation, out_loss=loss, workspace=workspace, validate=False)
    with pytest.raises(NumericalError):
        workspace.check_status()
