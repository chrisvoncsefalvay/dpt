"""Independent exact-input reductions and zero-extended detector matrices."""

import importlib
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import NumericalError
from dpt.detector import (
    BlurSpec,
    CalibrationSpec,
    SeparableBlurSpec,
    blur,
    blur_transpose,
    calibration_vjp,
    prepare_blur,
    prepare_detector,
)

wp: Any = importlib.import_module("warp")


def _decimal(value: Any) -> Decimal:
    return Decimal.from_float(float(value))


@pytest.mark.parametrize(
    "shared_gain,destination",
    [(True, "gain"), (True, "exposure"), (False, "exposure"), (True, "offset"), (False, "offset")],
)
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("wide_destination", [False, True])
def test_calibration_reductions_match_exact_signed_derivatives(
    shared_gain: bool, destination: str, groups: int, wide_destination: bool, warp_device: str
) -> None:
    with cpu_test_context(), localcontext() as context:
        context.prec = 200
        pixels = 1537
        values = np.arange(pixels)
        mean_host = (1 + values % 7).astype(np.float32)
        seed_host = np.resize(np.array([2**30, 0.25, -(2**30), -0.5], dtype=np.float32), pixels)
        gain_host = (
            np.array([1.5], dtype=np.float32)
            if shared_gain
            else (1 + (values % 3) * 0.25).astype(np.float32)
        )
        spec = CalibrationSpec(
            "ADU",
            shared_gain=shared_gain,
            active_gain=destination == "gain",
            active_exposure=destination == "exposure",
            active_offset=destination == "offset",
        )
        inputs = [
            wp.array(value, dtype=wp.float32, device=warp_device)
            for value in (mean_host, gain_host, [0.5], [0.0], seed_host)
        ]
        workspace = prepare_detector(max_pixels=pixels, reduction_groups=groups, device=warp_device)
        output_dtype = wp.float64 if wide_destination else wp.float32
        output = wp.empty(1, dtype=output_dtype, device=warp_device)
        calibration_vjp(
            *inputs[:4],
            cotangent=inputs[4],
            spec=spec,
            workspace=workspace,
            **{f"out_grad_{destination}": output},
        )
        exact = Decimal(0)
        for pixel in range(pixels):
            derivative = _decimal(seed_host[pixel])
            if destination == "gain":
                derivative *= Decimal("0.5") * _decimal(mean_host[pixel])
            elif destination == "exposure":
                derivative *= _decimal(gain_host[0 if shared_gain else pixel]) * _decimal(
                    mean_host[pixel]
                )
            exact += derivative
        cast = np.float64 if wide_destination else np.float32
        assert float(output.numpy()[0]) == float(cast(float(exact)))
        np.testing.assert_array_equal(inputs[4].numpy(), seed_host)
        assert workspace.reduction_groups == groups
        workspace.check_status()


@pytest.mark.parametrize("destination", ["gain", "exposure"])
def test_calibration_binary64_destination_keeps_small_derivative(
    destination: str, warp_device: str
) -> None:
    with cpu_test_context(), localcontext() as context:
        context.prec = 200
        spec = CalibrationSpec(
            "ADU", active_gain=destination == "gain", active_exposure=destination == "exposure"
        )
        inputs = [
            wp.array(value, dtype=wp.float32, device=warp_device)
            for value in ([1e-30], [1e-20], [1e-20], [0.0], [1e-30])
        ]
        workspace = prepare_detector(max_pixels=1, device=warp_device)
        output = wp.empty(1, dtype=wp.float64, device=warp_device)
        calibration_vjp(
            *inputs[:4],
            cotangent=inputs[4],
            spec=spec,
            workspace=workspace,
            **{f"out_grad_{destination}": output},
        )
        factor = inputs[2 if destination == "gain" else 1]
        expected = (
            _decimal(inputs[0].numpy()[0])
            * _decimal(inputs[4].numpy()[0])
            * _decimal(factor.numpy()[0])
        )
        actual = float(output.numpy()[0])
        assert actual != 0 and np.float32(float(expected)) == 0
        assert actual == pytest.approx(float(expected), rel=2e-15, abs=0)


def _matrix(spec: BlurSpec | SeparableBlurSpec) -> list[list[Decimal]]:
    size = spec.height * spec.width
    matrix = [[Decimal(0) for _ in range(size)] for _ in range(size)]
    if isinstance(spec, BlurSpec):
        height, width = spec.kernel_height, spec.kernel_width
        weights = tuple(_decimal(value) for value in spec.weights)
    else:
        height, width = len(spec.vertical), len(spec.horizontal)
        weights = tuple(
            _decimal(vertical) * _decimal(horizontal)
            for vertical in spec.vertical
            for horizontal in spec.horizontal
        )
    for row in range(spec.height):
        for column in range(spec.width):
            target = row * spec.width + column
            for kr in range(height):
                for kc in range(width):
                    source_row, source_column = row + kr - height // 2, column + kc - width // 2
                    if 0 <= source_row < spec.height and 0 <= source_column < spec.width:
                        matrix[target][source_row * spec.width + source_column] = weights[
                            kr * width + kc
                        ]
    return matrix


@pytest.mark.parametrize("shape", [(1, 3, 1, 3), (3, 4, 5, 3), (3, 2, 3, 5)])
@pytest.mark.parametrize("separable", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_signed_blur_matches_independent_decimal_matrix(
    shape: tuple[int, int, int, int], separable: bool, transpose: bool, warp_device: str
) -> None:
    with cpu_test_context(), localcontext() as context:
        context.prec = 200
        height, width, kh, kw = shape
        spec: BlurSpec | SeparableBlurSpec
        if separable:
            spec = SeparableBlurSpec(
                height,
                width,
                tuple(2.0 ** (-i - 3) for i in range(kh)),
                tuple(2.0 ** (-i - 3) for i in range(kw)),
            )
        else:
            spec = BlurSpec(height, width, kh, kw, tuple(2.0 ** (-i - 2) for i in range(kh * kw)))
        values = np.resize(
            np.array([2**24, 1.0, -(2**24), -0.25, 3.0], dtype=np.float32), height * width
        )
        matrix = _matrix(spec)
        if transpose:
            matrix = [list(column) for column in zip(*matrix, strict=True)]
        expected = np.array(
            [
                float(
                    sum(
                        (
                            coefficient * _decimal(value)
                            for coefficient, value in zip(row, values, strict=True)
                        ),
                        Decimal(0),
                    )
                )
                for row in matrix
            ],
            dtype=np.float32,
        )
        source = wp.array(values, dtype=wp.float32, device=warp_device)
        output = wp.empty_like(source)
        workspace = prepare_blur(spec, device=warp_device)
        if transpose:
            blur_transpose(source, out_grad_signal=output, workspace=workspace)
        else:
            blur(source, out_signal=output, workspace=workspace)
        np.testing.assert_array_equal(output.numpy(), expected)
        np.testing.assert_array_equal(source.numpy(), values)
        workspace.check_status()


@pytest.mark.parametrize("transpose", [False, True])
def test_separable_overflow_is_sticky_and_clears_explicitly(
    transpose: bool, warp_device: str
) -> None:
    with cpu_test_context():
        spec = SeparableBlurSpec(
            1, 1, (1e308 if transpose else 1e-308,), (1e-308 if transpose else 1e308,)
        )
        workspace = prepare_blur(spec, device=warp_device)
        source = wp.array([np.finfo(np.float32).max], dtype=wp.float32, device=warp_device)
        output = wp.empty_like(source)
        if transpose:
            blur_transpose(source, out_grad_signal=output, workspace=workspace, validate=False)
        else:
            blur(source, out_signal=output, workspace=workspace, validate=False)
        for _ in range(2):
            with pytest.raises(NumericalError, match="represented"):
                workspace.check_status()
        workspace.clear_status()
        workspace.check_status()
        source.assign(np.array([1e-30], dtype=np.float32))
        if transpose:
            blur_transpose(source, out_grad_signal=output, workspace=workspace)
        else:
            blur(source, out_signal=output, workspace=workspace)
        np.testing.assert_array_equal(output.numpy(), source.numpy())
