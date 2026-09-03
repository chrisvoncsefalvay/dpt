"""Independent homogeneous-cell recovery and FP64 spectral/chart derivatives."""

# Third-party numerical assertions expose dynamically typed interfaces.
# pyright: reportUnknownMemberType=false

import math
from contextlib import nullcontext
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import pairwise

import pytest

from dpt._runtime import cpu_test_context
from dpt.examples.three_material import (
    ThreeMaterialEvaluator,
    fraction_complement,
    recover_three_material,
    synthetic_model,
)

# Frozen independently by Fraction depths + Decimal100 exp, before native execution.
_TRUTHS = ((0.25, 0.375), (0.5, 0.25))
_OBSERVATIONS_HEX = (
    ("0x1.2cfd40f8bdcaep+11", "0x1.f30ec8375038bp+10"),
    ("0x1.017323fd90020p+11", "0x1.23ba930c1568bp+11"),
)
_STARTS = ((0.125, 0.125), (0.625, 0.125), (0.125, 0.625))
_MU = (
    (Fraction(1, 8), Fraction(1, 32)),
    (Fraction(1, 32), Fraction(5, 32)),
    (Fraction(1, 16), Fraction(1, 16)),
)


def _decimal(value: Fraction | int) -> Decimal:
    value = Fraction(value)
    return Decimal(value.numerator) / Decimal(value.denominator)


def _oracle(
    coordinates: tuple[float, ...], observation: tuple[float, float]
) -> tuple[tuple[float, float], float, tuple[float, float]]:
    with localcontext() as context:
        context.prec = 100
        free = tuple(Fraction.from_float(value) for value in coordinates)
        fractions = (*free, Fraction(1) - sum(free))
        mean = tuple(
            Decimal(4096)
            * (-_decimal(8 * sum(f * mu[e] for f, mu in zip(fractions, _MU, strict=True)))).exp()
            for e in range(2)
        )
        residual = tuple(
            (prediction - Decimal.from_float(observed)) / Decimal(4096)
            for prediction, observed in zip(mean, observation, strict=True)
        )
        loss = sum(value * value / 2 for value in residual)
        gradient = tuple(
            sum(
                -residual[e] * mean[e] / Decimal(4096) * _decimal(8 * (_MU[j][e] - _MU[2][e]))
                for e in range(2)
            )
            for j in range(2)
        )
        return (
            (float(mean[0]), float(mean[1])),
            float(loss),
            (float(gradient[0]), float(gradient[1])),
        )


def _observation(truth_index: int) -> tuple[float, float]:
    values = _OBSERVATIONS_HEX[truth_index]
    return float.fromhex(values[0]), float.fromhex(values[1])


def _independent_inverse(observation: tuple[float, float]) -> tuple[float, float, float]:
    # The exact matrix [1/2,-1/4;-1/4,3/4] has determinant 5/16.
    with localcontext() as context:
        context.prec = 100
        right = tuple(
            -(Decimal.from_float(value) / Decimal(4096)).ln() - Decimal("0.5")
            for value in observation
        )
        first = (Decimal("0.75") * right[0] + Decimal("0.25") * right[1]) / Decimal("0.3125")
        second = (Decimal("0.25") * right[0] + Decimal("0.5") * right[1]) / Decimal("0.3125")
        return float(first), float(second), float(1 - first - second)


@pytest.mark.parametrize("truth_index", range(2))
@pytest.mark.parametrize("initial", _STARTS)
def test_three_material_recovery_from_independent_means(
    truth_index: int, initial: tuple[float, float], warp_device: str
) -> None:
    observation = _observation(truth_index)
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        result = recover_three_material(observation, initial, device=warp_device)
    reference = _independent_inverse(observation)
    assert result.optimisation.reason == "gradient_tolerance"
    assert result.optimisation.evaluations <= 240
    assert len(result.optimisation.history) <= 81
    assert result.fractions == pytest.approx(reference, abs=5e-8, rel=0)
    assert result.fractions == pytest.approx(
        fraction_complement(_TRUTHS[truth_index]), abs=5e-8, rel=0
    )
    assert math.fsum(result.fractions) == 1.0
    assert all(value >= 0 for value in result.fractions)
    assert result.response_condition < 10.0
    independent = _oracle(result.optimisation.parameters, observation)
    assert max(abs(value) for value in independent[2]) <= 2e-10
    losses = [record.loss for record in result.optimisation.history]
    assert all(right <= left for left, right in pairwise(losses))
    assert losses[-1] < losses[0] / 1000


@pytest.mark.parametrize("truth_index", range(2))
def test_native_signal_and_complement_chart_gradient_have_independent_reference(
    truth_index: int, warp_device: str
) -> None:
    observation = _observation(truth_index)
    probe = (0.3125, 0.3125)
    reference = _oracle(probe, observation)
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        evaluator = ThreeMaterialEvaluator(synthetic_model(), observation, device=warp_device)
        assert str(evaluator.context.device) == warp_device
        assert evaluator.workspace.spec.materials == 3
        assert evaluator.workspace.spec.energies == 2
        prediction = evaluator.predict(probe)
        evaluation = evaluator(probe)
    assert prediction == pytest.approx(reference[0], rel=5e-14, abs=5e-14)
    assert evaluation.loss == pytest.approx(reference[1], rel=2e-13, abs=2e-13)
    assert evaluation.gradient == pytest.approx(reference[2], rel=2e-13, abs=2e-13)


@pytest.mark.parametrize("truth_index", range(2))
def test_production_complement_chart_gradient_matches_two_finite_difference_steps(
    truth_index: int, warp_device: str
) -> None:
    observation = _observation(truth_index)
    probe = (0.3125, 0.3125)
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        evaluator = ThreeMaterialEvaluator(synthetic_model(), observation, device=warp_device)
        evaluation = evaluator(probe)
        for step in (1e-5, 5e-6):
            for axis in range(2):
                left = tuple(value - (step if j == axis else 0) for j, value in enumerate(probe))
                right = tuple(value + (step if j == axis else 0) for j, value in enumerate(probe))
                difference = (evaluator(right).loss - evaluator(left).loss) / (2 * step)
                assert difference == pytest.approx(evaluation.gradient[axis], rel=5e-8, abs=5e-10)
