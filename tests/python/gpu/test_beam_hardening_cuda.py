"""CUDA execution of the independent spectral calibration and VJP oracles."""

import importlib
from typing import Any

import pytest

from dpt.beam_hardening import WaterEquivalentCalibration
from dpt.contracts import ContractError
from helpers import beam_hardening_cases as cases

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    "case",
    [
        cases.test_forward_inverse_and_vjps_match_independent_decimal,
        cases.test_beam_hardening_curve_and_independent_finite_differences,
        cases.test_retained_derivatives_survive_input_and_result_mutation,
        cases.test_inverse_range_and_iteration_bound_are_explicit,
        cases.test_calibration_layout_and_preparation_contracts,
    ],
)
def test_calibration_oracle_on_cuda(case):
    case("cuda:0")


def test_calibration_nondefault_stream_status_and_foreign_stream():
    stream = wp.Stream(device="cuda:0")
    with wp.ScopedStream(stream):
        source = cases._array([1.0, 2.0], "cuda:0")
        seed = cases._array([0.2, -0.4], "cuda:0")
    calibration = WaterEquivalentCalibration(
        cases._spec(), shape=(2,), device="cuda:0", stream=stream
    )
    assert calibration._status_readback._host.pinned
    depth = calibration.forward(source, check=False)
    calibration.check_status()
    calibration.forward_vjp(seed)
    calibration.inverse(depth)
    calibration.inverse_vjp(seed)
    with pytest.raises(ContractError, match="owning stream"):
        calibration.forward(source, stream=wp.Stream(device="cuda:0"))
