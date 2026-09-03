"""CPU checks of the independent oracle, including Decimal-to-binary32 rounding."""

import math
import struct
from decimal import Decimal, localcontext

import pytest

from dpt.validation.transmission import (
    MAX_FINITE,
    MIN_SUBNORMAL,
    OVERFLOW_MIDPOINT,
    analytic_cases,
    binary32,
    directional_check,
    exact_input,
    forward_reference,
    inverse_reference,
    inverse_vjp_reference,
    round_binary32,
    scalar_beam_reference,
    ulp_distance,
    value_error,
    vjp_reference,
)


def bits(value: float) -> int:
    return struct.unpack("!I", struct.pack("!f", value))[0]


@pytest.mark.parametrize("value", [0.0, -0.0, 2.0**-149, 2.0**-126, 0.1, 1.0, float(MAX_FINITE)])
def test_exact_input_and_roundtrip(value: float) -> None:
    stored = binary32(value)
    assert bits(round_binary32(exact_input(value))) == bits(stored)
    assert exact_input(value) == Decimal.from_float(stored)


def test_rounding_midpoints_and_double_rounding_trap() -> None:
    with localcontext() as context:
        context.prec = 200
        midpoint = Decimal(1) + Decimal(2) ** -24
        perturbation = Decimal(2) ** -100
        assert bits(round_binary32(midpoint)) == bits(1.0)
        assert bits(round_binary32(midpoint + perturbation)) == bits(1.0) + 1
        assert bits(round_binary32(midpoint - perturbation)) == bits(1.0)
        odd_midpoint = Decimal(1) + 3 * Decimal(2) ** -24
        assert bits(round_binary32(odd_midpoint)) == bits(1.0) + 2
        assert bits(round_binary32(MIN_SUBNORMAL / 2)) == 0
        assert bits(round_binary32(MIN_SUBNORMAL / 2 + Decimal(2) ** -300)) == 1
        assert bits(round_binary32(3 * MIN_SUBNORMAL / 2)) == 2
        assert round_binary32(OVERFLOW_MIDPOINT) == math.inf
        assert round_binary32(OVERFLOW_MIDPOINT - 1) == float(MAX_FINITE)
        assert round_binary32(-OVERFLOW_MIDPOINT) == -math.inf


@pytest.mark.parametrize(
    "depth", [0.0, -0.0, 2**-149, 2**-30, 0.5, 1, 64, 87, 103, 110, 200, 1024, float(MAX_FINITE)]
)
@pytest.mark.parametrize("beam", [0.0, 2**-149, 1.0, 1e30, float(MAX_FINITE)])
def test_precision_stability(depth: float, beam: float) -> None:
    lower = forward_reference(depth, beam, precision=100)
    higher = forward_reference(depth, beam, precision=200)
    assert {key: bits(value) for key, value in lower.rounded().items()} == {
        key: bits(value) for key, value in higher.rounded().items()
    }
    low_grad = vjp_reference(depth, beam, seed_T=2**100, seed_counts=2**-20, precision=100)
    high_grad = vjp_reference(depth, beam, seed_T=2**100, seed_counts=2**-20, precision=200)
    assert round_binary32(low_grad.grad_L) == round_binary32(high_grad.grad_L)


def test_boundary_identities_and_signed_zero() -> None:
    for depth in [0.0, -0.0]:
        result = forward_reference(depth, 17).rounded()
        assert result["T"] == 1
        assert result["counts"] == 17
        assert bits(result["removed"]) == 0
        assert bits(result["log_T"]) == bits(depth) ^ 0x80000000
    assert vjp_reference(0, 0, seed_T=1).grad_L == -1
    assert vjp_reference(0, 0, seed_counts=1).grad_n0 == 1
    assert vjp_reference(0, 0, seed_removed=1).grad_L == 1
    assert vjp_reference(0, 0, seed_log_T=1).grad_L == -1


def test_tail_bound_and_range_restoration() -> None:
    result = forward_reference(110, 1e30).rounded()
    assert result["T"] == 0
    assert result["counts"] > 0
    gradient = vjp_reference(110, 1e30, seed_T=1e30, seed_counts=1e30)
    assert round_binary32(gradient.grad_L) < 0
    tiny_beam = vjp_reference(1, 2**-149, seed_counts=2**100)
    assert round_binary32(tiny_beam.grad_L) < 0
    huge = forward_reference(float(MAX_FINITE))
    assert huge.tail_bound > 0
    assert huge.T == 0 and huge.removed == 1
    with localcontext() as context:
        context.prec = 200
        assert (-Decimal(1024)).exp() < huge.tail_bound
        assert huge.tail_bound * MAX_FINITE**2 * 2**31 < MIN_SUBNORMAL / 2


@pytest.mark.parametrize("delta", [0.0, -0.0, 2**-149, 2**-30, 0.25, 0.5, 1 - 2**-24])
def test_inverse_precision_and_derivative(delta: float) -> None:
    assert round_binary32(inverse_reference(delta, precision=100)) == round_binary32(
        inverse_reference(delta, precision=200)
    )
    gradient, budget = inverse_vjp_reference(delta, 1)
    assert gradient >= 1 and budget > 0
    if delta == 0:
        assert gradient == 1


@pytest.mark.parametrize("value", [-1, -(2**-149), math.inf, -math.inf, math.nan])
def test_invalid_inputs(value: float) -> None:
    with pytest.raises(ValueError):
        forward_reference(value, 0)
    with pytest.raises(ValueError):
        forward_reference(1, value)
    with pytest.raises(ValueError):
        inverse_reference(value)


def test_invalid_decrement_and_seed() -> None:
    with pytest.raises(ValueError):
        inverse_reference(1)
    with pytest.raises(ValueError):
        vjp_reference(1, seed_T=math.nan)


def test_scalar_reduction_cancellation_and_empty() -> None:
    result, budget = scalar_beam_reference([0, 0, 0], [2**100, 1, -(2**100)], addition_depth=2)
    assert result == 1 and budget > 0
    assert scalar_beam_reference([], [], addition_depth=0)[0] == 0
    with pytest.raises(ValueError):
        scalar_beam_reference([0], [], addition_depth=0)
    with pytest.raises(ValueError):
        scalar_beam_reference([], [], addition_depth=-1)


def test_error_budgets_do_not_hide_subnormal_loss() -> None:
    assert ulp_distance(0.0, -0.0) == 0
    assert ulp_distance(0, 2**-149) == 1
    record = value_error(0, 8 * MIN_SUBNORMAL)
    assert not record.passed
    assert record.allowed_ulps == 1 and record.ulps == 8
    assert value_error(1, Decimal(1)).passed


def test_analytic_cases_are_independent_optical_depth_definitions() -> None:
    cases = {case.name: case for case in analytic_cases()}
    assert cases["homogeneous"].optical_depth == cases["split-homogeneous"].optical_depth
    assert cases["millimetres"].optical_depth == cases["centimetres"].optical_depth
    assert cases["linear-coefficient"].optical_depth == 1
    assert cases["added-segment"].optical_depth > cases["homogeneous"].optical_depth


def test_directional_sweep_calls_candidate_and_retains_rounding_regime() -> None:
    def evaluate(depth: float) -> float:
        return forward_reference(depth).rounded()["T"]

    interior = directional_check(evaluate, 1, -math.exp(-1))
    assert len(interior) == 22  # h=2**-24 no longer changes the right binary32 input.
    assert min(record.absolute_error for record in interior) < 1e-4
    assert interior[-1].absolute_error > min(record.absolute_error for record in interior)
    boundary = directional_check(evaluate, 0, -1)
    assert min(record.absolute_error for record in boundary) < 1e-4
    assert all(record.stencil == "one-sided-second-order" for record in boundary)
