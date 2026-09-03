"""Real CUDA first-order derivatives checked independently of rounded forward images."""

import importlib
import math
from decimal import Decimal, localcontext
from typing import Any

import pytest

from dpt.transmission import (
    GradientRangeError,
    TransmissionError,
    TransmissionSpec,
    optical_depth_from_removed_vjp,
    prepare_transmission,
    transmission_vjp,
    transmit,
)
from dpt.validation.transmission import (
    MAX_FINITE,
    directional_check,
    exact_input,
    inverse_vjp_reference,
    scalar_beam_reference,
    vjp_reference,
)

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


def _array(values: list[float]) -> Any:
    return wp.array(values, dtype=wp.float32, device="cuda:0")


def _assert_gradient(actual: float, expected: Decimal, budget: Decimal) -> None:
    with localcontext() as context:
        context.prec = 200
        error = abs(exact_input(actual) - expected)
        assert error <= budget, (actual, expected, error, budget)


def test_mixed_weighted_vjp_range_preservation_and_overwrite() -> None:
    # Each tuple is L, beam, T seed, count seed, log seed, removed seed.
    cases = [
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (1.0, 2.0, 3.0, -5.0, 7.0, -11.0),
        (0.5, 1.0, 2**100, 0.0, 0.0, 2**100),
        (110.0, 1e30, 1e30, 1e20, 0.0, 0.0),
        (200.0, 1e30, 0.0, 1e30, 0.0, 0.0),
        (1.0, 2**-149, 0.0, 2**100, 0.0, 0.0),
        (110.0, 0.0, 0.0, 0.0, 0.0, 1e30),
        (float(MAX_FINITE), float(MAX_FINITE), float(MAX_FINITE), float(MAX_FINITE), 3.0, 0.0),
    ]
    columns = [list(column) for column in zip(*cases, strict=True)]
    arrays = [_array(column) for column in columns]
    grad_depth, grad_beam = _array([123.0] * len(cases)), _array([456.0] * len(cases))
    workspace = prepare_transmission(
        TransmissionSpec(beam="per-pixel", active_beam=True), max_pixels=len(cases)
    )
    original_seeds = [array.numpy().copy() for array in arrays[2:]]
    for _ in range(2):
        transmission_vjp(
            arrays[0],
            arrays[1],
            cotangent_T=arrays[2],
            cotangent_counts=arrays[3],
            cotangent_log_T=arrays[4],
            cotangent_removed=arrays[5],
            out_grad_L=grad_depth,
            out_grad_n0=grad_beam,
            workspace=workspace,
        )
        actual_depth, actual_beam = grad_depth.numpy(), grad_beam.numpy()
        for index, (
            depth,
            beam,
            transmission_seed,
            count_seed,
            log_seed,
            removed_seed,
        ) in enumerate(cases):
            reference = vjp_reference(
                depth,
                beam,
                seed_T=transmission_seed,
                seed_counts=count_seed,
                seed_log_T=log_seed,
                seed_removed=removed_seed,
            )
            _assert_gradient(float(actual_depth[index]), reference.grad_L, reference.budget_L)
            _assert_gradient(float(actual_beam[index]), reference.grad_n0, reference.budget_n0)
        assert list(actual_depth[:4]) == [-1, 0, -1, 1]
        assert actual_beam[1] == 1
        assert actual_depth[5] == 0  # Exact equal/opposite exponential terms.
        assert actual_depth[6] != 0 and actual_depth[7] != 0 and actual_depth[8] != 0
    for array, before in zip(arrays[2:], original_seeds, strict=True):
        assert (array.numpy() == before).all()


@pytest.mark.parametrize("seed_name", ["seed_T", "seed_counts", "seed_log_T", "seed_removed"])
def test_single_seed_specialisations(seed_name: str) -> None:
    depths = [0.0, 0.5, 2.0, 64.0, 110.0]
    seeds = [1.0, -2.0, 3.0, -4.0, 1e30]
    workspace = prepare_transmission(TransmissionSpec(beam="scalar"), max_pixels=len(depths))
    output = wp.empty(len(depths), dtype=wp.float32, device="cuda:0")
    transmission_vjp(
        _array(depths),
        7.0,
        out_grad_L=output,
        workspace=workspace,
        **{seed_name.replace("seed_", "cotangent_"): _array(seeds)},
    )
    for depth, seed, actual in zip(depths, seeds, output.numpy(), strict=True):
        reference = vjp_reference(
            depth,
            7.0,
            seed_T=seed if seed_name == "seed_T" else 0.0,
            seed_counts=seed if seed_name == "seed_counts" else 0.0,
            seed_log_T=seed if seed_name == "seed_log_T" else 0.0,
            seed_removed=seed if seed_name == "seed_removed" else 0.0,
        )
        _assert_gradient(float(actual), reference.grad_L, reference.budget_L)


@pytest.mark.parametrize("count", [0, 1, 255, 256, 257, 65537])
def test_active_scalar_beam_fixed_tree_and_reset(count: int) -> None:
    depths = [float(index % 3) for index in range(count)]
    seeds = [float((index % 7) - 3) for index in range(count)]
    workspace = prepare_transmission(
        TransmissionSpec(beam="device-scalar", active_beam=True), max_pixels=count
    )
    beam = _array([0.0])
    output = _array([123.0])
    depth_array, seed_array = _array(depths), _array(seeds)
    # Warp 1.17: five warp shuffle-adds, then seven serial warp-partial adds.
    addition_depth, remaining = 0, max(1, count)
    while True:
        addition_depth += 12
        remaining = (remaining + 255) // 256
        if remaining <= 1:
            break
    expected, budget = scalar_beam_reference(depths, seeds, addition_depth=addition_depth)
    transmission_vjp(
        depth_array, beam, cotangent_counts=seed_array, out_grad_n0=output, workspace=workspace
    )
    first = float(output.numpy()[0])
    _assert_gradient(first, expected, budget)
    transmission_vjp(
        depth_array, beam, cotangent_counts=seed_array, out_grad_n0=output, workspace=workspace
    )
    assert float(output.numpy()[0]) == first
    transmission_vjp(depth_array, beam, out_grad_n0=output, workspace=workspace)
    assert output.numpy()[0] == 0


def test_scalar_reduction_cancellation_uses_magnitude_budget() -> None:
    depths = [0.0] * 257
    seeds = [2**100, 1.0, -(2**100)] + [0.0] * 254
    workspace = prepare_transmission(
        TransmissionSpec(beam="device-scalar", active_beam=True), max_pixels=len(depths)
    )
    output = _array([123])
    transmission_vjp(
        _array(depths),
        _array([0]),
        cotangent_counts=_array(seeds),
        out_grad_n0=output,
        workspace=workspace,
    )
    expected, budget = scalar_beam_reference(depths, seeds, addition_depth=24)
    assert expected == 1
    _assert_gradient(float(output.numpy()[0]), expected, budget)


def test_inverse_weighted_derivative() -> None:
    decrements = [0.0, -0.0, 2**-149, 2**-30, 0.5, 1 - 2**-24]
    seeds = [1.0, -2.0, 2**100, -3.0, 7.0, 2**50]
    workspace = prepare_transmission(max_pixels=len(decrements))
    output = wp.empty(len(decrements), dtype=wp.float32, device="cuda:0")
    seed_array = _array(seeds)
    optical_depth_from_removed_vjp(
        _array(decrements), seed_array, out_grad_delta=output, workspace=workspace
    )
    for decrement, seed, actual in zip(decrements, seeds, output.numpy(), strict=True):
        expected, budget = inverse_vjp_reference(decrement, seed)
        _assert_gradient(float(actual), expected, budget)
    assert output.numpy()[0] == 1 and output.numpy()[1] == -2
    assert list(seed_array.numpy()) == seeds


@pytest.mark.parametrize("kind", ["pointwise", "scalar", "inverse"])
def test_gradient_overflow_is_explicit(kind: str) -> None:
    if kind == "pointwise":
        workspace = prepare_transmission(TransmissionSpec(beam="scalar"), max_pixels=1)
        with pytest.raises(GradientRangeError):
            transmission_vjp(
                _array([0]),
                float(MAX_FINITE),
                cotangent_counts=_array([2]),
                out_grad_L=_array([0]),
                workspace=workspace,
            )
    elif kind == "scalar":
        workspace = prepare_transmission(
            TransmissionSpec(beam="device-scalar", active_beam=True), max_pixels=2
        )
        with pytest.raises(GradientRangeError):
            transmission_vjp(
                _array([0, 0]),
                _array([1]),
                cotangent_counts=_array([float(MAX_FINITE)] * 2),
                out_grad_n0=_array([0]),
                workspace=workspace,
            )
    else:
        workspace = prepare_transmission(max_pixels=1)
        with pytest.raises(GradientRangeError):
            optical_depth_from_removed_vjp(
                _array([0.5]),
                _array([float(MAX_FINITE)]),
                out_grad_delta=_array([0]),
                workspace=workspace,
            )


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_invalid_seed_preserves_destination(invalid: float) -> None:
    workspace = prepare_transmission(max_pixels=1)
    output = _array([123])
    with pytest.raises(TransmissionError):
        transmission_vjp(
            _array([1]), cotangent_T=_array([invalid]), out_grad_L=output, workspace=workspace
        )
    assert output.numpy()[0] == 123


def test_gpu_directional_sweeps_include_boundary_and_rounding_regions() -> None:
    workspace = prepare_transmission(max_pixels=1)
    output = _array([0])

    def evaluate(depth: float) -> float:
        transmit(_array([depth]), out_T=output, workspace=workspace)
        return float(output.numpy()[0])

    for anchor in [0.0, 0.5, 1.0, 2.0]:
        records = directional_check(evaluate, anchor, -math.exp(-anchor))
        assert len(records) >= 21
        assert min(record.absolute_error for record in records) < 2e-4
        assert records[-1].absolute_error > min(record.absolute_error for record in records)
