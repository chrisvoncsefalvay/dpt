"""Independent flat/dark logarithm, three-input derivatives and snapshot ownership."""

import importlib
import math
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pytest

from dpt._runtime import StatusReadback
from dpt.calibration import CalibrationPrecision, LogCalibration
from dpt.contracts import ContractError, NumericalError

wp: Any = importlib.import_module("warp")


def _frame(values, precision: CalibrationPrecision, device: str):
    return wp.array(
        np.asarray(values, dtype=precision), dtype=getattr(wp, precision), device=device
    )


def _reference(signal, flat, dark, seed):
    depth, gradients = [], [[], [], []]
    with localcontext() as context:
        context.prec = 800
        for intensity, illumination, background, cotangent in zip(
            signal.reshape(-1), flat.reshape(-1), dark.reshape(-1), seed.reshape(-1), strict=True
        ):
            intensity, illumination, background, cotangent = (
                Decimal.from_float(float(value))
                for value in (intensity, illumination, background, cotangent)
            )
            corrected, reference = intensity - background, illumination - background
            depth.append(float((reference / corrected).ln()))
            for output, value in zip(
                gradients,
                (
                    -cotangent / corrected,
                    cotangent / reference,
                    cotangent * (1 / corrected - 1 / reference),
                ),
                strict=True,
            ):
                output.append(float(value))
    return np.asarray(depth).reshape(signal.shape), tuple(
        np.asarray(values).reshape(signal.shape) for values in gradients
    )


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_log_depth_and_all_cotangents_match_independent_decimal(precision, warp_device):
    signal = _frame([[80, 120, 1], [-1, 6, 25]], precision, warp_device)
    flat = _frame([[100, 80, 1], [-2, 6, 50]], precision, warp_device)
    dark = _frame([[3, -10, 0], [-3, 2, 7]], precision, warp_device)
    seed = _frame([[0.2, -0.3, 0], [0.4, 0.5, -0.6]], precision, warp_device)
    operator = LogCalibration(flat, dark, precision=precision, device=warp_device)
    result = operator.forward(signal)
    gradients = operator.vjp(seed)
    expected, expected_gradients = _reference(
        signal.numpy(), flat.numpy(), dark.numpy(), seed.numpy()
    )
    tolerance = 2e-7 if precision == "float32" else 3e-15
    np.testing.assert_allclose(result.optical_depth.numpy(), expected, rtol=tolerance, atol=0)
    for actual, target in zip(
        (gradients.signal, gradients.flat, gradients.dark), expected_gradients, strict=True
    ):
        np.testing.assert_allclose(actual.numpy(), target, rtol=tolerance, atol=0)
        assert actual.shape == signal.shape and actual.dtype == getattr(wp, precision)
    assert result.precision == precision and result.optical_depth.numpy()[0, 1] < 0


def test_fp64_weak_contrast_wide_range_and_rescued_derivatives(warp_device):
    signal = _frame([1.0, 1e-308, 1e308, 1e-310, 1e308], "float64", warp_device)
    flat = _frame(
        [2.0, 1e308, 1e-308, 2e-310, math.nextafter(1e308, math.inf)], "float64", warp_device
    )
    dark = _frame([-1e308, 0, 0, 0, 0], "float64", warp_device)
    seed = _frame([1, 0, 1e-308, 1e-320, 1e308], "float64", warp_device)
    operator = LogCalibration(flat, dark, precision="float64", device=warp_device)
    result = operator.forward(signal)
    gradients = operator.vjp(seed)
    expected, expected_gradients = _reference(
        signal.numpy(), flat.numpy(), dark.numpy(), seed.numpy()
    )
    np.testing.assert_allclose(result.optical_depth.numpy(), expected, rtol=2e-13, atol=0)
    for actual, target in zip(
        (gradients.signal, gradients.flat, gradients.dark), expected_gradients, strict=True
    ):
        np.testing.assert_allclose(actual.numpy(), target, rtol=3e-14, atol=0)
    assert result.optical_depth.numpy()[0] > 0
    assert abs(result.optical_depth.numpy()[1]) > 1400
    assert gradients.dark.numpy()[-1] > 0


@pytest.mark.parametrize("contrast_sign", [-1, 1], ids=["negative-contrast", "positive-contrast"])
@pytest.mark.parametrize("seed_sign", [-1, 1], ids=["negative-seed", "positive-seed"])
def test_fp64_dark_cotangent_preserves_rescued_exponents(contrast_sign, seed_sign, warp_device):
    smallest = math.nextafter(0.0, 1.0)
    signal_values = np.asarray([0.0, 0.0, 1e-310, 1e-200, 1e308, 0.0])
    flat_values = np.asarray(
        [
            contrast_sign * 1e-200,
            contrast_sign * smallest,
            math.nextafter(1e-310, math.inf if contrast_sign > 0 else -math.inf),
            1e-200 + contrast_sign * 1e-201,
            math.nextafter(1e308, math.inf if contrast_sign > 0 else -math.inf),
            0.0,
        ]
    )
    dark_values = np.asarray([-1e200, -1e100, 0.0, 0.0, 0.0, -1e200])
    seed_values = seed_sign * np.asarray([1e308, 1e308, 1e-310, 1e-300, 1e308, 1e308])
    signal, flat, dark, seed = (
        _frame(values, "float64", warp_device)
        for values in (signal_values, flat_values, dark_values, seed_values)
    )
    operator = LogCalibration(flat, dark, precision="float64", device=warp_device)
    operator.forward(signal)
    gradients = operator.vjp(seed)
    expected = [[], [], []]
    with localcontext() as context:
        context.prec = 1100
        for intensity, illumination, background, cotangent in zip(
            signal_values, flat_values, dark_values, seed_values, strict=True
        ):
            intensity, illumination, background, cotangent = (
                Decimal.from_float(float(value))
                for value in (intensity, illumination, background, cotangent)
            )
            corrected, reference = intensity - background, illumination - background
            # Algebraic form is independent of the kernel's exponent decomposition
            # and retains contrast that a difference of rounded reciprocals loses.
            values = (
                -cotangent / corrected,
                cotangent / reference,
                cotangent * (illumination - intensity) / (corrected * reference),
            )
            for output, value in zip(expected, values, strict=True):
                output.append(float(value))
    for actual, target in zip(
        (gradients.signal, gradients.flat, gradients.dark), expected, strict=True
    ):
        np.testing.assert_allclose(actual.numpy(), target, rtol=4e-15, atol=0)
    assert np.all(gradients.dark.numpy()[:5] != 0)
    assert gradients.dark.numpy()[-1] == 0


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("parameter", [0, 1, 2], ids=["signal", "flat", "dark"])
def test_three_input_cotangents_match_independent_central_differences(
    precision, parameter, warp_device
):
    frames = (
        np.asarray([[12, 20], [30, 40]], dtype=precision),
        np.asarray([[25, 30], [40, 45]], dtype=precision),
        np.asarray([[1, 3], [2, -1]], dtype=precision),
    )
    seed = np.asarray([[0.2, -0.3], [0.4, 0.6]], dtype=precision)
    signal, flat, dark = tuple(_frame(values, precision, warp_device) for values in frames)
    operator = LogCalibration(flat, dark, precision=precision, device=warp_device)
    operator.forward(signal)
    gradients = operator.vjp(_frame(seed, precision, warp_device))
    actual = (gradients.signal, gradients.flat, gradients.dark)[parameter].numpy()

    def loss(values):
        observed, reference, background = values
        return float(np.sum(seed * -np.log((observed - background) / (reference - background))))

    difference = np.empty_like(actual, dtype=np.float64)
    for index in np.ndindex(actual.shape):
        positive, negative = ([values.astype(float).copy() for values in frames] for _ in range(2))
        step = 1e-4
        positive[parameter][index] += step
        negative[parameter][index] -= step
        difference[index] = (loss(positive) - loss(negative)) / (2 * step)
    np.testing.assert_allclose(
        actual, difference, rtol=2e-6 if precision == "float32" else 2e-8, atol=2e-11
    )


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_fixed_and_signal_snapshots_survive_caller_and_output_mutation(precision, warp_device):
    signal = _frame([[12, 20]], precision, warp_device)
    flat = _frame([[25, 30]], precision, warp_device)
    dark = _frame([[1, 3]], precision, warp_device)
    seed = _frame([[0.2, -0.3]], precision, warp_device)
    expected, targets = _reference(signal.numpy(), flat.numpy(), dark.numpy(), seed.numpy())
    operator = LogCalibration(flat, dark, precision=precision, device=warp_device)
    with operator.context.scope():
        flat.fill_(math.nan)
        dark.fill_(math.inf)
    result = operator.forward(signal)
    np.testing.assert_allclose(result.optical_depth.numpy(), expected, rtol=2e-7)
    with operator.context.scope():
        signal.fill_(-100)
        result.optical_depth.fill_(99)
    original_seed = seed.numpy().copy()
    gradients = operator.vjp(seed)
    for actual, target in zip(
        (gradients.signal, gradients.flat, gradients.dark), targets, strict=True
    ):
        np.testing.assert_allclose(actual.numpy(), target, rtol=2e-7)
    np.testing.assert_array_equal(seed.numpy(), original_seed)


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("flat,dark", [(math.nan, 0), (1, math.inf), (1, 1), (0, 1)])
def test_fixed_frame_domains_are_checked_at_preparation(precision, flat, dark, warp_device):
    with pytest.raises(ContractError, match=r"finite|positive"):
        LogCalibration(
            _frame([flat], precision, warp_device),
            _frame([dark], precision, warp_device),
            precision=precision,
            device=warp_device,
        )


@pytest.mark.parametrize("invalid", [0.0, -1.0, math.nan, math.inf])
def test_unchecked_signal_failure_is_persistent_and_blocks_snapshot_overwrite(invalid, warp_device):
    flat = _frame([2.0], "float64", warp_device)
    dark = _frame([0.0], "float64", warp_device)
    operator = LogCalibration(flat, dark, precision="float64", device=warp_device)
    bad = _frame([invalid], "float64", warp_device)
    good = _frame([1.0], "float64", warp_device)
    operator.forward(bad, check=False)
    for _ in range(2):
        with pytest.raises(ContractError):
            operator.check_status()
    with pytest.raises(ContractError):
        operator.forward(good)
    np.testing.assert_array_equal(operator._signal.numpy(), bad.numpy())
    with pytest.raises(ContractError):
        operator.consume_status()
    operator.check_status()
    assert float(operator.forward(good).optical_depth.numpy()[0]) == pytest.approx(math.log(2))


def test_failed_status_delivery_keeps_unchecked_failure_observable(warp_device, monkeypatch):
    operator = LogCalibration(
        _frame([2], "float64", warp_device),
        _frame([0], "float64", warp_device),
        precision="float64",
        device=warp_device,
    )
    operator.forward(_frame([0], "float64", warp_device), check=False)
    with monkeypatch.context() as patch:

        def failed_read(*args, **kwargs):
            raise RuntimeError("status copy failed")

        patch.setattr(StatusReadback, "read", failed_read)
        with pytest.raises(RuntimeError, match="status copy"):
            operator.consume_status()
    with pytest.raises(ContractError):
        operator.check_status()
    with pytest.raises(ContractError):
        operator.consume_status()
    operator.check_status()


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_invalid_cotangent_and_unrepresentable_gradient_are_delivered(precision, warp_device):
    small = 1e-38 if precision == "float32" else math.ulp(0.0)
    operator = LogCalibration(
        _frame([2 * small], precision, warp_device),
        _frame([0], precision, warp_device),
        precision=precision,
        device=warp_device,
    )
    signal = _frame([small], precision, warp_device)
    operator.forward(signal)
    with pytest.raises(ContractError):
        operator.vjp(_frame([math.nan], precision, warp_device))
    with pytest.raises(ContractError):
        operator.consume_status()
    operator.forward(signal)
    with pytest.raises(NumericalError):
        operator.vjp(_frame([1e38 if precision == "float32" else 1], precision, warp_device))
    with pytest.raises(NumericalError):
        operator.check_status()
    with pytest.raises(NumericalError):
        operator.consume_status()
    operator.forward(signal)
    gradients = operator.vjp(_frame([-0.0], precision, warp_device))
    for gradient in (gradients.signal, gradients.flat, gradients.dark):
        assert gradient.numpy().tolist() == [0]
        assert not np.signbit(gradient.numpy()).any()


def test_metadata_stream_and_forward_epoch_contracts(warp_device):
    flat = _frame([[2, 3]], "float64", warp_device)
    dark = _frame([[0, 0]], "float64", warp_device)
    operator = LogCalibration(flat, dark, precision="float64", device=warp_device)
    with pytest.raises(ContractError, match="successful"):
        operator.vjp(flat)
    for value in (_frame([1, 2], "float64", warp_device), _frame([[1, 2]], "float32", warp_device)):
        with pytest.raises(ContractError, match=r"shape|dtype|dimensional"):
            operator.forward(value)
    bad_check: Any = 1
    with pytest.raises(ContractError, match="boolean"):
        operator.forward(flat, check=bad_check)
    if warp_device == "cpu":
        with pytest.raises(ContractError, match="stream"):
            operator.forward(flat, stream=object())
    result = operator.forward(flat)
    assert result.optical_depth.numpy().tolist() == [[0, 0]]
    with pytest.raises(ContractError, match=r"shape|dimensional"):
        operator.vjp(flat.flatten())
    operator.clear_status()
    with pytest.raises(ContractError, match="successful"):
        operator.vjp(flat)


def test_fp64_corrected_frame_overflow_is_a_domain_failure(warp_device):
    with pytest.raises(ContractError):
        LogCalibration(
            _frame([1e308], "float64", warp_device),
            _frame([-1e308], "float64", warp_device),
            precision="float64",
            device=warp_device,
        )
    operator = LogCalibration(
        _frame([0], "float64", warp_device),
        _frame([-1e308], "float64", warp_device),
        precision="float64",
        device=warp_device,
    )
    with pytest.raises(ContractError):
        operator.forward(_frame([1e308], "float64", warp_device))


def test_noncontiguous_image_views_are_rejected_at_the_metadata_boundary(warp_device):
    base = _frame([[1, 2], [3, 4], [5, 6], [7, 8]], "float64", warp_device)
    view = base[::2]
    flat = _frame([[10, 10], [10, 10]], "float64", warp_device)
    dark = _frame([[0, 0], [0, 0]], "float64", warp_device)
    with pytest.raises(ContractError, match="contiguous"):
        LogCalibration(view, dark, precision="float64", device=warp_device)
    operator = LogCalibration(flat, dark, precision="float64", device=warp_device)
    with pytest.raises(ContractError, match="contiguous"):
        operator.forward(view)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_warmed_calls_reuse_owned_device_and_host_storage(precision, warp_device, monkeypatch):
    flat = _frame([[20, 30]], precision, warp_device)
    dark = _frame([[1, 2]], precision, warp_device)
    signal = _frame([[10, 15]], precision, warp_device)
    seed = _frame([[0.3, -0.4]], precision, warp_device)
    operator = LogCalibration(flat, dark, precision=precision, device=warp_device)
    assert len(operator.warm()) == 3
    result = operator.forward(signal)
    gradients = operator.vjp(seed)
    expected = result.optical_depth.numpy().copy()
    expected_gradients = gradients.dark.numpy().copy()

    def allocation(*args, **kwargs):
        raise AssertionError("warmed calibration allocated a device/host array")

    monkeypatch.setattr(wp, "empty", allocation)
    monkeypatch.setattr(wp, "empty_like", allocation)
    for _ in range(3):
        assert operator.forward(signal) is result
        assert operator.vjp(seed) is gradients
        np.testing.assert_array_equal(result.optical_depth.numpy(), expected)
        np.testing.assert_array_equal(gradients.dark.numpy(), expected_gradients)
    assert operator.host_staging_bytes == 4
    assert operator.scratch_bytes == 2 * (8 * (8 if precision == "float64" else 4) + 8) + 4
