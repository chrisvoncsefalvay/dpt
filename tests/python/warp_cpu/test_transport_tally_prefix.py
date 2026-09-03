"""Both tally modes retain guarded scoring, original-history moments and replay gradients."""

# Warp array annotations and decorated functions are checked by the native compiler.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
import importlib
import math
from decimal import Decimal, localcontext
from typing import Any, Literal

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError, NumericalError
from dpt.transport._constants import INPUT_MASK, NUMERICAL_MASK
from dpt.transport.derivatives import TransportParameter, derivative_histories
from dpt.transport.estimators import history_mean, history_moments, prepare_estimators
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.transport.model import (
    HistoryStatus,
    IncompleteHistoryError,
    MaterialGrid,
    PlanarDetector,
    TransportSpec,
)
from dpt.transport.rng import HistoryBatch

wp: Any = importlib.import_module("warp")
kernels: Any = importlib.import_module("dpt.transport.kernels")
_STRATEGIES: tuple[Literal["plain", "grouped"], ...] = ("plain", "grouped")


def _array(values: Any, dtype: Any = None) -> Any:
    return wp.array(values, dtype=wp.float64 if dtype is None else dtype, device="cpu")


# Values, statuses and expected contributions are declared independently, including
# sum-pass contracts which the preceding scale-validation pass handles separately.
_CONTRIBUTIONS = [
    (0, 6.0, 0, 2.0, 3.0, 0),
    (0, -6.0, 0, 2.0, -3.0, 0),
    (0, 6.0, 1, 2.0, 3.0, 0),
    (0, 0.0, 0, 2.0, 0.0, 0),
    (0, 6.0, -1, 2.0, 0.0, 0),
    *[(0, 6.0, terminal, 2.0, 0.0, 0) for terminal in range(2, 8)],
    (-1, 6.0, 0, 2.0, 0.0, 0),
    (-2, 6.0, 0, 2.0, 0.0, 0),
    (1, 6.0, 0, 2.0, 0.0, 0),
    (2**31 - 1, 6.0, 0, 2.0, 0.0, 0),
    (0, 6.0, 0, 0.0, 0.0, 0),
    (0, 6.0, 0, -2.0, 0.0, 0),
    (0, 6.0, 0, math.nan, 0.0, 0),
    (0, 6.0, 0, math.inf, 0.0, 0),
    (0, math.nan, 0, 2.0, 0.0, NUMERICAL_MASK),
    (0, math.inf, 0, 2.0, 0.0, NUMERICAL_MASK),
    (0, -math.inf, 0, 2.0, 0.0, NUMERICAL_MASK),
    (0, 1e308, 0, 1e-308, 0.0, NUMERICAL_MASK),
    (0, float.fromhex("0x0.0000000000001p-1022"), 0, 1e308, 0.0, 0),
    (0, math.nan, 2, math.nan, 0.0, 0),
    (0, math.nan, -1, math.nan, 0.0, 0),
    (-1, math.nan, 0, math.nan, 0.0, 0),
    (2**31 - 1, math.nan, 0, math.nan, 0.0, 0),
]


@pytest.mark.parametrize("sticky", [0, INPUT_MASK | (1 << int(HistoryStatus.EVENT_BUDGET))])
@pytest.mark.parametrize(
    ("target", "score", "terminal", "scale", "expected", "new_mask"), _CONTRIBUTIONS
)
def test_plain_and_grouped_guarded_sums_preserve_contributions_and_sticky_status(
    target: int,
    score: float,
    terminal: int,
    scale: float,
    expected: float,
    new_mask: int,
    sticky: int,
) -> None:
    with cpu_test_context():
        actual = []
        for kernel in [kernels.tally_sum_plain, kernels.tally_sum]:
            sums = _array([7.0])
            status = _array([sticky], wp.int32)
            wp.launch(
                kernel,
                dim=1,
                inputs=[
                    _array([target], wp.int32),
                    _array([score]),
                    _array([terminal], wp.int32),
                    1,
                    _array([scale]),
                    sums,
                    status,
                ],
                device="cpu",
            )
            assert sums.numpy().tolist() == [7.0 + expected]
            assert status.numpy().tolist() == [sticky | new_mask]
            actual.append((sums.numpy().view(np.uint64).tolist(), status.numpy().tolist()))
        assert actual[0] == actual[1]


@pytest.mark.parametrize(("target", "terminal"), [(-1, 0), (0, -1), (0, 2), (2**31 - 1, 0)])
def test_excluded_histories_do_not_read_protected_score_or_scale(
    target: int, terminal: int
) -> None:
    with cpu_test_context():
        for kernel in [kernels.tally_sum_plain, kernels.tally_sum]:
            sums = _array([11.0])
            status = _array([INPUT_MASK], wp.int32)
            wp.launch(
                kernel,
                dim=1,
                inputs=[
                    _array([target], wp.int32),
                    _array([]),
                    _array([terminal], wp.int32),
                    1,
                    _array([]),
                    sums,
                    status,
                ],
                device="cpu",
            )
            assert sums.numpy().tolist() == [11.0]
            assert status.numpy().tolist() == [INPUT_MASK]


@pytest.mark.parametrize("scale", [0.0, -1.0, math.nan])
def test_rejected_scales_do_not_read_protected_score(scale: float) -> None:
    with cpu_test_context():
        for kernel in [kernels.tally_sum_plain, kernels.tally_sum]:
            sums = _array([11.0])
            status = _array([INPUT_MASK], wp.int32)
            wp.launch(
                kernel,
                dim=1,
                inputs=[
                    _array([0], wp.int32),
                    _array([]),
                    _array([0], wp.int32),
                    1,
                    _array([scale]),
                    sums,
                    status,
                ],
                device="cpu",
            )
            assert sums.numpy().tolist() == [11.0]
            assert status.numpy().tolist() == [INPUT_MASK]


def _workspace(
    count: int,
    pixels: int = 3,
    estimator: Literal["analogue", "continuous-absorption"] = "analogue",
) -> Any:
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (2.0 / pixels, 2.0), (pixels, 1), 2.0),
        80.0,
        "analytic one-layer absorption and original-history tally fixture",
        estimator=estimator,
    )
    return prepare_transport(
        spec,
        material_ids=_array([0], wp.int32),
        absorption=_array([0.3]),
        scattering=_array([0.0]),
        max_histories=count,
        device="cpu",
    )


def _decimal_moments(
    targets: list[int], values: list[float], pixels: int
) -> tuple[list[float], list[float]]:
    with localcontext() as context:
        context.prec = 800
        count = Decimal(len(values))
        means, variances = [], []
        for pixel in range(pixels):
            samples = [
                Decimal.from_float(value) if target == pixel else Decimal(0)
                for target, value in zip(targets, values, strict=True)
            ]
            mean = sum(samples) / count
            variance = sum((value - mean) ** 2 for value in samples) / (count * (count - 1))
            means.append(float(mean))
            variances.append(float(variance))
        return means, variances


@pytest.mark.parametrize("strategy", ["plain", "grouped"])
@pytest.mark.parametrize(
    ("terminal", "error_type"),
    [(2, IncompleteHistoryError), (4, NumericalError), (7, ContractError)],
)
def test_public_modes_keep_failures_until_explicit_clear_then_recover(
    strategy, terminal: int, error_type: type[Exception]
) -> None:
    with cpu_test_context():
        count = 33
        workspace = _workspace(count, 1)
        moments = prepare_estimators(workspace, tally_strategy=strategy)
        batch = HistoryBatch(24971, 0, count)
        pixel = _array([0] * count, wp.int32)
        score = _array([1.0] * count)
        mean = wp.empty(1, dtype=wp.float64, device="cpu")
        invalid = _array([terminal] * count, wp.int32)
        with pytest.raises(error_type):
            history_mean(pixel, score, invalid, batch=batch, workspace=moments, out_mean=mean)
        with pytest.raises(error_type):
            workspace.check_status()
        valid = _array([0] * count, wp.int32)
        with pytest.raises(error_type):
            history_mean(pixel, score, valid, batch=batch, workspace=moments, out_mean=mean)
        workspace.clear_status()
        history_mean(pixel, score, valid, batch=batch, workspace=moments, out_mean=mean)
        assert mean.numpy().tolist() == [1.0]
        workspace.check_status()


@pytest.mark.parametrize(
    ("count", "case"),
    [
        (31, "sparse"),
        (32, "sparse"),
        (33, "sparse"),
        (65, "sparse"),
        (65, "huge"),
        (65, "tiny"),
        (33, "subnormal"),
        (33, "missed"),
    ],
)
def test_public_modes_preserve_independent_signed_mean_and_variance(count: int, case: str) -> None:
    with cpu_test_context():
        pixels = 3
        targets = [-1 if i % 5 == 0 else i % 2 for i in range(count)]
        pattern = [1.0, -0.5, 0.125, 0.0, 1e-100]
        if case == "huge":
            pattern = [1e150, -5e149, 1.25e149]
        elif case == "tiny":
            pattern = [1e-150, -5e-151, 1.25e-151]
        elif case == "subnormal":
            targets = [0] * count
            pattern = [float.fromhex("0x0.0000000000001p-1022")]
        elif case == "missed":
            targets = [-1] * count
        values = [
            0.0 if target == -1 else pattern[i % len(pattern)] for i, target in enumerate(targets)
        ]
        expected_mean, expected_variance = _decimal_moments(targets, values, pixels)
        workspace = _workspace(count, pixels)
        observed = []
        for strategy in _STRATEGIES:
            moments = prepare_estimators(workspace, tally_strategy=strategy)
            pixel = _array(targets, wp.int32)
            score = _array(values)
            status = _array([1 if t == -1 else 0 for t in targets], wp.int32)
            mean = wp.empty(pixels, dtype=wp.float64, device="cpu")
            variance = wp.empty_like(mean)
            batch = HistoryBatch(82931, 2**40, count)
            history_moments(
                pixel,
                score,
                status,
                batch=batch,
                workspace=moments,
                out_mean=mean,
                out_variance_of_mean=variance,
            )
            np.testing.assert_allclose(mean.numpy(), expected_mean, rtol=2e-13, atol=5e-324)
            np.testing.assert_allclose(variance.numpy(), expected_variance, rtol=2e-13, atol=5e-324)
            if case == "subnormal":
                np.testing.assert_array_equal(
                    mean.numpy().view(np.uint64), np.asarray(expected_mean).view(np.uint64)
                )
                np.testing.assert_array_equal(variance.numpy(), np.zeros(pixels))
            from_moments = mean.numpy().copy()
            history_mean(pixel, score, status, batch=batch, workspace=moments, out_mean=mean)
            np.testing.assert_array_equal(
                mean.numpy().view(np.uint64), from_moments.view(np.uint64)
            )
            observed.append(
                (mean.numpy().view(np.uint64).tolist(), variance.numpy().view(np.uint64).tolist())
            )
        assert observed[0] == observed[1]


@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
def test_actual_forward_and_replay_gradient_tallies_preserve_analytic_layer_law(estimator) -> None:
    with cpu_test_context():
        count = 1025
        workspace = _workspace(count, 1, estimator)
        batch = HistoryBatch(90421, 2**44, count)
        amplitude = 2.5
        inputs = [
            _array([[0.0, 0.0, -1.0]] * count, wp.vec3d),
            _array([[0.0, 0.0, 1.0]] * count, wp.vec3d),
            _array([1.0] * count),
            _array([1.1]),
        ]
        outputs = {
            f"out_{name}": wp.empty(count, dtype=dtype, device="cpu")
            for name, dtype in [
                ("pixel", wp.int32),
                ("score", wp.float64),
                ("energy", wp.float64),
                ("events", wp.int32),
                ("status", wp.int32),
            ]
        }
        trace_histories(
            *inputs, batch=batch, workspace=workspace, source_amplitude=amplitude, **outputs
        )
        probability = math.exp(-0.33)
        expected_forward = amplitude * probability
        expected_density = -0.33 * expected_forward
        derivative = wp.empty(count, dtype=wp.float64, device="cpu")
        derivative_pixel = wp.empty(count, dtype=wp.int32, device="cpu")
        derivative_status = wp.empty_like(derivative_pixel)
        observed = []
        for strategy in _STRATEGIES:
            moments = prepare_estimators(workspace, tally_strategy=strategy)
            mean = wp.empty(1, dtype=wp.float64, device="cpu")
            variance = wp.empty_like(mean)
            history_moments(
                outputs["out_pixel"],
                outputs["out_score"],
                outputs["out_status"],
                batch=batch,
                workspace=moments,
                out_mean=mean,
                out_variance_of_mean=variance,
            )
            forward = float(mean.numpy()[0])
            forward_variance = float(variance.numpy()[0])
            per_mode = [forward, forward_variance]
            for parameter, analytic in [
                (TransportParameter("log-material-density", 0), expected_density),
                (TransportParameter("source-amplitude"), probability),
            ]:
                derivative_histories(
                    *inputs,
                    batch=batch,
                    workspace=workspace,
                    parameter=parameter,
                    source_amplitude=amplitude,
                    out_pixel=derivative_pixel,
                    out_derivative=derivative,
                    out_status=derivative_status,
                )
                np.testing.assert_array_equal(
                    derivative_status.numpy(), outputs["out_status"].numpy()
                )
                history_moments(
                    derivative_pixel,
                    derivative,
                    derivative_status,
                    batch=batch,
                    workspace=moments,
                    out_mean=mean,
                    out_variance_of_mean=variance,
                )
                measured = float(mean.numpy()[0])
                observed_variance = float(variance.numpy()[0])
                if estimator == "continuous-absorption":
                    assert measured == pytest.approx(analytic, rel=2e-13)
                    assert observed_variance < 1e-25
                else:
                    factor = -0.33 * amplitude if parameter.material == 0 else 1.0
                    allowance = 7 * abs(factor) * math.sqrt(probability * (1 - probability) / count)
                    assert abs(measured - analytic) <= allowance
                    realised = forward / amplitude
                    assert observed_variance == pytest.approx(
                        factor**2 * realised * (1 - realised) / (count - 1), rel=2e-12
                    )
                per_mode.extend([measured, observed_variance])
            observed.append(per_mode)
            if estimator == "continuous-absorption":
                assert forward == pytest.approx(expected_forward, rel=2e-13)
                assert forward_variance < 1e-25
            else:
                assert abs(forward - expected_forward) <= 7 * amplitude * math.sqrt(
                    probability * (1 - probability) / count
                )
                realised = forward / amplitude
                assert forward_variance == pytest.approx(
                    amplitude**2 * realised * (1 - realised) / (count - 1), rel=2e-12
                )
        np.testing.assert_array_equal(
            np.asarray(observed[0]).view(np.uint64), np.asarray(observed[1]).view(np.uint64)
        )
