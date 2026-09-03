"""Actual CUDA forward validation against independent exact-input Decimal references."""

import importlib
import math
import struct
from typing import Any

import pytest

from dpt.transmission import (
    TransmissionError,
    TransmissionSpec,
    optical_depth_from_removed,
    prepare_transmission,
    transmit,
)
from dpt.validation.transmission import (
    MAX_FINITE,
    analytic_cases,
    binary32,
    forward_reference,
    inverse_reference,
    value_error,
)

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


def _array(values: list[float]) -> Any:
    return wp.array(values, dtype=wp.float32, device="cuda:0")


def _bits(value: float) -> int:
    return struct.unpack("!I", struct.pack("!f", value))[0]


def _neighbours(value: float) -> list[float]:
    middle = _bits(binary32(value))
    return [
        struct.unpack("!f", struct.pack("!I", bits))[0] for bits in range(middle - 1, middle + 2)
    ]


def _depths() -> list[float]:
    values = [
        0.0,
        -0.0,
        2**-149,
        2**-30,
        2**-20,
        2**-10,
        0.125,
        0.5,
        1,
        2,
        10,
        20,
        50,
        110,
        200,
        float(MAX_FINITE),
    ]
    for transition in [64, 126 * math.log(2), 149 * math.log(2), 150 * math.log(2)]:
        values.extend(_neighbours(transition))
    return values


def _assert_values(depths: list[float], beams: list[float], outputs: dict[str, Any]) -> None:
    observed = {name: array.numpy() for name, array in outputs.items()}
    for index, (depth, beam) in enumerate(zip(depths, beams, strict=True)):
        oracle = forward_reference(depth, beam)
        for name, values in observed.items():
            actual = float(values[index])
            if name == "log_T":
                assert _bits(actual) == _bits(-binary32(depth)), (index, depth, name)
            else:
                record = value_error(actual, getattr(oracle, name), counts=name == "counts")
                assert record.passed, (index, depth, beam, name, actual, record)
                if name in ("T", "counts") and record.regime == "subnormal":
                    # A one-ULP allowance must not hide the chosen zero/nonzero transitions.
                    expected_nonzero = oracle.rounded()[name] != 0
                    assert (actual != 0) == expected_nonzero, (depth, beam, name, actual)
                if depth == 0:
                    exact = {"T": 1.0, "counts": binary32(beam), "removed": 0.0}[name]
                    assert _bits(actual) == _bits(exact), (name, depth, beam)
                if name == "counts":
                    assert 0 <= actual <= binary32(beam)


def test_all_fifteen_output_masks_against_oracle() -> None:
    depths = _depths()
    optical_depth = _array(depths)
    workspace = prepare_transmission(TransmissionSpec(beam="scalar"), max_pixels=len(depths))
    names = ("T", "counts", "log_T", "removed")
    for mask in range(1, 16):
        outputs = {
            name: wp.empty(len(depths), dtype=wp.float32, device="cuda:0")
            for index, name in enumerate(names)
            if mask & (1 << index)
        }
        transmit(
            optical_depth,
            1e30,
            workspace=workspace,
            **{f"out_{name}": array for name, array in outputs.items()},
        )
        _assert_values(depths, [1e30] * len(depths), outputs)


@pytest.mark.parametrize("mode", ["none", "scalar", "device-scalar", "per-pixel"])
def test_beam_representations_and_dynamic_range(mode: Any) -> None:
    depths = _depths()
    beam_values = [0.0, 1.0, 2**-149, 2**-126, 2**-50, 2**50, 1e30, float(MAX_FINITE)]
    workspace = prepare_transmission(TransmissionSpec(beam=mode), max_pixels=len(depths))
    for scale in beam_values if mode != "none" else [1.0]:
        beams = [scale] * len(depths)
        if mode == "per-pixel":
            beams = [beam_values[index % len(beam_values)] for index in range(len(depths))]
        beam = (
            None
            if mode == "none"
            else (
                scale if mode == "scalar" else _array([scale] if mode == "device-scalar" else beams)
            )
        )
        names = ("T", "log_T", "removed") if mode == "none" else ("T", "counts", "log_T", "removed")
        outputs = {name: wp.empty(len(depths), dtype=wp.float32, device="cuda:0") for name in names}
        transmit(
            _array(depths),
            beam,
            workspace=workspace,
            **{f"out_{name}": array for name, array in outputs.items()},
        )
        _assert_values(depths, beams, outputs)
        if mode == "per-pixel":
            break


@pytest.mark.parametrize("count", [0, 1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1009])
def test_pixel_boundaries(count: int) -> None:
    depths = [float(index % 13) / 8 for index in range(count)]
    output = wp.empty(count, dtype=wp.float32, device="cuda:0")
    workspace = prepare_transmission(max_pixels=count)
    transmit(_array(depths), out_T=output, workspace=workspace)
    _assert_values(depths, [1.0] * count, {"T": output})


def test_independently_derived_path_cases() -> None:
    depths = [float(case.optical_depth) for case in analytic_cases()]
    output = wp.empty(len(depths), dtype=wp.float32, device="cuda:0")
    workspace = prepare_transmission(max_pixels=len(depths))
    transmit(_array(depths), out_T=output, workspace=workspace)
    _assert_values(depths, [1.0] * len(depths), {"T": output})
    observed = output.numpy()
    assert observed[2] == observed[3]  # Same homogeneous optical depth, separate definitions.
    assert observed[5] == observed[6]  # Independently specified mm/cm inputs.
    assert observed[7] <= observed[2]  # Added non-negative optical depth.


def test_weak_attenuation_inverse() -> None:
    decrements = (
        [0.0, -0.0, 2**-149, 2**-126]
        + [2.0**-power for power in range(30, 14, -1)]
        + [0.5, 1 - 2**-24]
    )
    output = wp.empty(len(decrements), dtype=wp.float32, device="cuda:0")
    workspace = prepare_transmission(max_pixels=len(decrements))
    optical_depth_from_removed(_array(decrements), out_L=output, workspace=workspace)
    for decrement, actual in zip(decrements, output.numpy(), strict=True):
        record = value_error(float(actual), inverse_reference(decrement))
        assert record.passed, (decrement, actual, record)
        if decrement == 0:
            assert _bits(float(actual)) == 0


@pytest.mark.parametrize("invalid", [-1.0, -(2**-149), math.nan, math.inf, -math.inf])
def test_invalid_depth_leaves_output_unwritten_even_with_zero_beam(invalid: float) -> None:
    workspace = prepare_transmission(TransmissionSpec(beam="scalar"), max_pixels=3)
    output = _array([19, 19, 19])
    with pytest.raises(TransmissionError, match="L\\[1\\]"):
        transmit(_array([0, invalid, 1]), 0, out_counts=output, workspace=workspace)
    assert list(output.numpy()) == [19, 19, 19]


@pytest.mark.parametrize("invalid", [-(2**-149), math.nan, math.inf, 1.0])
def test_invalid_inverse_leaves_output_unwritten(invalid: float) -> None:
    workspace = prepare_transmission(max_pixels=1)
    output = _array([19])
    with pytest.raises(TransmissionError):
        optical_depth_from_removed(_array([invalid]), out_L=output, workspace=workspace)
    assert output.numpy()[0] == 19


def test_empty_batch_still_validates_supplied_beam() -> None:
    workspace = prepare_transmission(TransmissionSpec(beam="device-scalar"), max_pixels=0)
    empty = _array([])
    with pytest.raises(TransmissionError):
        transmit(empty, _array([-1]), out_T=empty, workspace=workspace)


def test_metadata_rejects_dtype_rank_capacity_and_overlap() -> None:
    workspace = prepare_transmission(max_pixels=3)
    values = _array([0, 1, 2])
    output = wp.empty(3, dtype=wp.float32, device="cuda:0")
    for malformed in [
        wp.zeros(3, dtype=wp.float64, device="cuda:0"),
        wp.zeros((1, 3), dtype=wp.float32, device="cuda:0"),
        _array([0, 1, 2, 3]),
    ]:
        with pytest.raises(TransmissionError):
            transmit(malformed, out_T=output, workspace=workspace)
    with pytest.raises(TransmissionError, match="overlaps"):
        transmit(values, out_T=values, workspace=workspace)
    with pytest.raises(TransmissionError, match="overlaps"):
        transmit(values, out_T=output, out_log_T=output, workspace=workspace)
