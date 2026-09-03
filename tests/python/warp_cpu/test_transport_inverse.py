"""Portable integration of complete independent squared-signal gradients."""

# pytest.approx stubs belong to an optional dependency.
# pyright: reportUnknownMemberType=false
import importlib
import math
from typing import Any

import pytest

from dpt.contracts import NumericalError
from dpt.transport.derivatives import TransportParameter
from dpt.transport.forward import prepare_transport
from dpt.transport.inverse import ParallelBeam, TransportSquaredOracle, prepare_transport_inverse
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.rng import HistoryBatch

wp: Any = importlib.import_module("warp")
DEVICE = "cpu"


@pytest.fixture(autouse=True)
def _device_scope(warp_device: str):
    global DEVICE
    DEVICE = warp_device
    with wp.ScopedDevice(DEVICE):
        yield


def _oracle(histories: int, optical_depth: float = 1.0) -> TransportSquaredOracle:
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
        80.0,
        "analytic pure-absorption inverse test",
    )
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=DEVICE),
        absorption=wp.array([optical_depth], dtype=wp.float64, device=DEVICE),
        scattering=wp.array([0.0], dtype=wp.float64, device=DEVICE),
        device=DEVICE,
        max_histories=histories,
    )
    return prepare_transport_inverse(
        workspace,
        source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
        parameters=(
            TransportParameter("log-material-density", 0),
            TransportParameter("log-source-amplitude"),
        ),
        observation=wp.array([0.2], dtype=wp.float64, device=DEVICE),
        pixel_weights=wp.array([1.0], dtype=wp.float64, device=DEVICE),
        base_density=(1.0,),
    )


def test_full_expected_signal_gradient_and_parameter_chart() -> None:
    count = 2**16
    oracle = _oracle(count)
    gradient = oracle.gradient_replicate(
        (0.0, math.log(2.0)), HistoryBatch(971, 0, count), HistoryBatch(971, count, count)
    )
    mean = 2 * math.exp(-1)
    expected_density = (mean - 0.2) * -mean
    expected_log_amplitude = (mean - 0.2) * mean
    # The analytic Bernoulli standard deviations are bounded by 1/sqrt(N) for
    # this amplitude and scale; an 8-sigma conservative bound allows both factors.
    tolerance = 8 / math.sqrt(count)
    assert abs(gradient[0] - expected_density) < tolerance
    assert abs(gradient[1] - expected_log_amplitude) < tolerance
    assert gradient[0] == pytest.approx(-gradient[1], rel=1e-12)


def test_common_random_objective_difference_is_exactly_zero_at_identical_parameters() -> None:
    oracle = _oracle(512)
    value = oracle.change_replicate(
        (0.0, 0.0), (0.0, 0.0), HistoryBatch(72, 0, 512), HistoryBatch(72, 512, 512)
    )
    assert value == 0.0


def test_vacuum_amplitude_gradient_is_exact_and_all_arrays_reused() -> None:
    oracle = _oracle(64, optical_depth=0.0)
    gradient = oracle.gradient_replicate(
        (0.0, 0.0), HistoryBatch(58, 0, 64), HistoryBatch(58, 64, 64)
    )
    assert gradient[0] == 0.0
    assert gradient[1] == pytest.approx(0.8)
    with pytest.raises(TransportError, match="disjoint"):
        oracle.gradient_replicate((0.0, 0.0), HistoryBatch(58, 0, 64), HistoryBatch(58, 0, 64))


@pytest.mark.parametrize("coordinate", [1000.0, -1000.0])
def test_trial_chart_domain_error_precedes_any_history_execution(coordinate: float) -> None:
    from dpt.contracts import TrialDomainError

    oracle = _oracle(8)
    with pytest.raises(TrialDomainError):
        oracle.gradient_replicate((coordinate, 0.0), HistoryBatch(74, 0, 8), HistoryBatch(74, 8, 8))


@pytest.mark.parametrize("observation,weight", [(float("nan"), 1.0), (0.0, -1.0)])
def test_inverse_preparation_rejects_invalid_immutable_measurement(
    observation: float,
    weight: float,
) -> None:
    # Prepare a fresh tiny physical model; the failing scan occurs before any
    # source/history launch and never substitutes a mocked numerical result.
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
        80.0,
        "analytic inverse preparation validation",
    )
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=DEVICE),
        absorption=wp.array([1.0], dtype=wp.float64, device=DEVICE),
        scattering=wp.array([0.0], dtype=wp.float64, device=DEVICE),
        device=DEVICE,
        max_histories=2,
    )
    with pytest.raises(TransportError, match="invalid device"):
        prepare_transport_inverse(
            workspace,
            source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
            parameters=(TransportParameter("log-material-density", 0),),
            observation=wp.array([observation], dtype=wp.float64, device=DEVICE),
            pixel_weights=wp.array([weight], dtype=wp.float64, device=DEVICE),
            base_density=(1.0,),
        )


def test_failed_chart_upload_invalidates_cached_density_even_after_clear() -> None:
    from dpt.transport._constants import NUMERICAL_MASK

    oracle = _oracle(16)
    parameters = (0.0, 0.0)
    first, second = HistoryBatch(129, 0, 16), HistoryBatch(129, 16, 16)
    expected = oracle.gradient_replicate(parameters, first, second)
    assert any(value != 0 for value in expected)
    uploaded = oracle.parameter_upload_bytes
    assert uploaded == 16
    # A real sticky device failure must invalidate the chart cache even when
    # the chart kernel itself executes successfully before its status check.
    with oracle.workspace.context.scope():
        oracle.workspace._status.fill_(NUMERICAL_MASK)
    with pytest.raises(NumericalError):
        oracle.gradient_replicate((0.1, 0.0), first, second)
    with pytest.raises(NumericalError):
        oracle.workspace.check_status()
    oracle.workspace.clear_status()
    actual = oracle.gradient_replicate(parameters, first, second)
    # Restoring the previous coordinate requires another physical density upload;
    # returning the cached coordinate would leave exp(0.1) on the device instead.
    assert oracle.parameter_upload_bytes == uploaded + 16
    assert actual == pytest.approx(expected, rel=5e-14, abs=5e-16)
