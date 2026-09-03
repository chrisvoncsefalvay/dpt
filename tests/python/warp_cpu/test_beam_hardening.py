"""Independent spectral quadrature, inverse calibration and derivative checks."""

import importlib
from decimal import Decimal
from typing import Any

import numpy as np
import pytest

from dpt.beam_hardening import (
    BeamHardeningSpec,
    WaterEquivalentCalibration,
)
from dpt.contracts import ContractError
from dpt.materials import MaterialTable
from helpers.beam_hardening_cases import (
    _array,
    _decimal,
    _spec,
    test_beam_hardening_curve_and_independent_finite_differences,
    test_calibration_layout_and_preparation_contracts,
    test_forward_inverse_and_vjps_match_independent_decimal,
    test_inverse_range_and_iteration_bound_are_explicit,
    test_retained_derivatives_survive_input_and_result_mutation,
)

# Keep the original CPU pytest entry points for these shared scientific cases.
__all__ = [
    "test_beam_hardening_curve_and_independent_finite_differences",
    "test_calibration_layout_and_preparation_contracts",
    "test_forward_inverse_and_vjps_match_independent_decimal",
    "test_inverse_range_and_iteration_bound_are_explicit",
    "test_retained_derivatives_survive_input_and_result_mutation",
]

wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize("coefficients", [(1e200, 1e-200), (1e308, 1e-308)])
def test_transmitted_slope_survives_full_coefficient_range(coefficients, warp_device):
    provenance = _spec().material.provenance
    spec = BeamHardeningSpec(
        MaterialTable(
            "Analytic coefficient-range fixture",
            (1.0, 2.0),
            coefficients,
            "mm^-1",
            provenance,
        ),
        (1.0, 2.0),
        (1.0, 1.0),
        (1.0, 1.0),
        1.0,
        "above",
        _spec().spectrum_provenance,
        _spec().response_provenance,
    )
    calibration = WaterEquivalentCalibration(spec, shape=(2,), device=warp_device)
    expected_depth, expected_slope = _decimal(spec, [0.0, 1.0])
    depth = calibration.forward(_array([0.0, 1.0], warp_device))
    np.testing.assert_allclose(depth.numpy(), expected_depth, rtol=2e-14, atol=0)
    np.testing.assert_allclose(
        calibration.forward_vjp(_array([1.0, 1.0], warp_device)).numpy(),
        expected_slope,
        rtol=2e-14,
        atol=0,
    )
    np.testing.assert_array_equal(calibration.inverse(depth).numpy(), [0.0, 1.0])
    np.testing.assert_allclose(
        calibration.inverse_vjp(_array([1.0, 1.0], warp_device)).numpy(),
        1 / expected_slope,
        rtol=2e-14,
        atol=0,
    )


def test_underflowed_transmitted_probability_retains_representable_slope(warp_device):
    spec = BeamHardeningSpec(
        MaterialTable(
            "Analytic transmitted-moment fixture",
            (1.0, 2.0),
            (1e308, 1e-200),
            "mm^-1",
            _spec().material.provenance,
        ),
        (1.0, 2.0),
        (1.0, 1.0),
        (1.0, 1.0),
        1.0,
        "above",
        _spec().spectrum_provenance,
        _spec().response_provenance,
    )
    length = 800.0 / spec.material.coefficients[0]
    expected_depth, expected_slope = _decimal(spec, [length])
    calibration = WaterEquivalentCalibration(spec, shape=(1,), device=warp_device)
    depth = calibration.forward(_array([length], warp_device))
    np.testing.assert_allclose(depth.numpy(), expected_depth, rtol=2e-14, atol=0)
    np.testing.assert_allclose(
        calibration.forward_vjp(_array([1.0], warp_device)).numpy(),
        expected_slope,
        rtol=5e-13,
        atol=0,
    )
    assert expected_slope[0] > 1e-40


def test_rare_opaque_node_preserves_weak_weighted_depth_and_inverse(warp_device):
    spec = BeamHardeningSpec(
        MaterialTable(
            "Analytic weak detected-spectrum fixture",
            (1.0, 2.0),
            (1.0, 1e-20),
            "mm^-1",
            _spec().material.provenance,
        ),
        (1.0, 2.0),
        (1e-20, 1.0),
        (1.0, 1.0),
        1.0,
        "above",
        _spec().spectrum_provenance,
        _spec().response_provenance,
    )
    lengths = np.array([0.0, 1e-240, 0.2, 1.0])
    expected_depth, expected_slope = _decimal(spec, lengths)
    calibration = WaterEquivalentCalibration(spec, shape=(4,), device=warp_device)
    depth = calibration.forward(_array(lengths, warp_device))
    np.testing.assert_allclose(depth.numpy(), expected_depth, rtol=2e-14, atol=0)
    np.testing.assert_allclose(
        calibration.forward_vjp(_array(np.ones(4), warp_device)).numpy(),
        expected_slope,
        rtol=2e-14,
        atol=0,
    )
    np.testing.assert_allclose(calibration.inverse(depth).numpy(), lengths, rtol=8e-14, atol=0)
    np.testing.assert_allclose(
        calibration.inverse_vjp(_array(np.ones(4), warp_device)).numpy(),
        1 / expected_slope,
        rtol=3e-14,
        atol=0,
    )


def test_subnormal_detected_loss_is_resolved_before_reduction(warp_device):
    coefficient = float.fromhex("0x0.0000000000001p-1022")
    spec = BeamHardeningSpec(
        MaterialTable(
            "Analytic subnormal detected-loss fixture",
            (1.0, 2.0),
            (coefficient, coefficient),
            "mm^-1",
            _spec().material.provenance,
        ),
        (1.0, 2.0),
        (1.0, 1.0),
        (1.0, 1.0),
        1.0,
        "above",
        _spec().spectrum_provenance,
        _spec().response_provenance,
    )
    calibration = WaterEquivalentCalibration(spec, shape=(1,), device=warp_device)
    depth = calibration.forward(_array([1.0], warp_device))
    np.testing.assert_array_equal(depth.numpy(), [coefficient])
    np.testing.assert_array_equal(calibration.inverse(depth).numpy(), [1.0])
    np.testing.assert_array_equal(
        calibration.forward_vjp(_array([1.0], warp_device)).numpy(), [coefficient]
    )
    np.testing.assert_array_equal(
        calibration.inverse_vjp(_array([coefficient], warp_device)).numpy(), [1.0]
    )


@pytest.mark.parametrize("coefficient", [float.fromhex("0x0.0000000000001p-1022"), 1e308])
def test_subnormal_attenuation_and_inverse_cotangents_are_retained(coefficient, warp_device):
    spec = BeamHardeningSpec(
        MaterialTable(
            "Analytic scalar-range fixture",
            (1.0,),
            (coefficient,),
            "mm^-1",
            _spec().material.provenance,
        ),
        (1.0,),
        (1.0,),
        (1.0,),
        1.0,
        "above",
        _spec().spectrum_provenance,
        _spec().response_provenance,
    )
    calibration = WaterEquivalentCalibration(spec, shape=(1,), device=warp_device)
    depth = calibration.forward(_array([0.0], warp_device))
    np.testing.assert_array_equal(depth.numpy(), [0.0])
    np.testing.assert_array_equal(
        calibration.forward_vjp(_array([1.0], warp_device)).numpy(), [coefficient]
    )
    np.testing.assert_array_equal(calibration.inverse(depth).numpy(), [0.0])
    cotangent = coefficient if coefficient < 1.0 else 1e-15
    expected_gradient = float(Decimal.from_float(cotangent) / Decimal.from_float(coefficient))
    assert expected_gradient > 0
    np.testing.assert_array_equal(
        calibration.inverse_vjp(_array([cotangent], warp_device)).numpy(),
        [expected_gradient],
    )


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), 10001.0])
def test_domain_failures_persist_until_consumed(invalid, warp_device):
    calibration = WaterEquivalentCalibration(_spec(), shape=(1,), device=warp_device)
    calibration.forward(_array([invalid], warp_device), check=False)
    with pytest.raises(ContractError):
        calibration.check_status()
    with pytest.raises(ContractError):
        calibration.forward(_array([1.0], warp_device))
    with pytest.raises(ContractError):
        calibration.consume_status()
    calibration.forward(_array([1.0], warp_device))
