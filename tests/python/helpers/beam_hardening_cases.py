"""Shared exact scientific helpers for the native controls."""

import hashlib
import importlib
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pytest

from dpt.beam_hardening import (
    BeamHardeningExecution,
    BeamHardeningSpec,
    WaterEquivalentCalibration,
)
from dpt.contracts import ContractError, NumericalError
from dpt.materials import MaterialTable, Provenance

wp: Any = importlib.import_module("warp")


def _spec():
    provenance = Provenance(
        source="Analytic four-energy calibration fixture defined in test_beam_hardening.py",
        sha256=hashlib.sha256(b"mu=(0.4,0.12,0.035,0.018) per mm").hexdigest(),
        rights="Apache-2.0",
        description="Synthetic attenuation coefficients, not a measured water table",
    )
    return BeamHardeningSpec(
        MaterialTable(
            "analytic calibration fixture",
            (20.0, 40.0, 80.0, 120.0),
            (0.4, 0.12, 0.035, 0.018),
            "mm^-1",
            provenance,
        ),
        (20.0, 40.0, 80.0, 120.0),
        (0.1, 0.4, 0.35, 0.15),
        (0.7, 0.9, 1.0, 0.8),
        10000.0,
        "above",
        Provenance(
            "Analytic source weights",
            hashlib.sha256(b"w=(.1,.4,.35,.15)").hexdigest(),
            "Apache-2.0",
            "Synthetic energy-integrated photon populations",
        ),
        Provenance(
            "Analytic detector response",
            hashlib.sha256(b"R=(.7,.9,1,.8)").hexdigest(),
            "Apache-2.0",
            "Synthetic detected photon response",
        ),
    )


def _array(values, device):
    return wp.array(np.asarray(values, dtype=np.float64), dtype=wp.float64, device=device)


def _decimal(spec, lengths):
    depths, slopes = [], []
    with localcontext() as context:
        context.prec = 400
        coefficients = tuple(Decimal.from_float(value) for value in spec.material.coefficients)
        weights = tuple(
            Decimal.from_float(source) * Decimal.from_float(response)
            for source, response in zip(spec.source_weights, spec.detector_response, strict=True)
        )
        incident = sum(weights, Decimal(0))
        for length in np.asarray(lengths).reshape(-1):
            transmitted = tuple(
                weight * (-coefficient * Decimal.from_float(float(length))).exp()
                for weight, coefficient in zip(weights, coefficients, strict=True)
            )
            total = sum(transmitted, Decimal(0))
            depths.append(float(-(total / incident).ln()))
            slopes.append(
                float(
                    sum(
                        value * coefficient
                        for value, coefficient in zip(transmitted, coefficients, strict=True)
                    )
                    / total
                )
            )
    return np.asarray(depths).reshape(np.shape(lengths)), np.asarray(slopes).reshape(
        np.shape(lengths)
    )


def test_forward_inverse_and_vjps_match_independent_decimal(warp_device):
    spec = _spec()
    lengths = np.asarray([[0.0, 1e-240, 0.3], [17.0, 300.0, 10000.0]])
    seed = np.asarray([[0.2, -0.3, 0.0], [0.5, 0.7, -0.8]])
    expected, slopes = _decimal(spec, lengths)
    calibration = WaterEquivalentCalibration(spec, shape=lengths.shape, device=warp_device)
    depth = calibration.forward(_array(lengths, warp_device))
    np.testing.assert_allclose(depth.numpy(), expected, rtol=2e-14, atol=0)
    forward_gradient = calibration.forward_vjp(_array(seed, warp_device)).numpy().copy()
    np.testing.assert_allclose(forward_gradient, seed * slopes, rtol=2e-14, atol=0)
    recovered = calibration.inverse(depth)
    np.testing.assert_allclose(recovered.numpy(), lengths, rtol=8e-14, atol=0)
    np.testing.assert_allclose(
        calibration.inverse_vjp(_array(seed, warp_device)).numpy(),
        seed / slopes,
        rtol=3e-14,
        atol=0,
    )
    assert depth.dtype == wp.float64 and tuple(depth.shape) == lengths.shape


def test_beam_hardening_curve_and_independent_finite_differences(warp_device):
    spec = _spec()
    lengths = np.asarray([1.0, 30.0, 150.0, 500.0])
    seed = np.asarray([0.25, -0.5, 0.75, -1.0])
    expected, slopes = _decimal(spec, lengths)
    calibration = WaterEquivalentCalibration(spec, shape=(4,), device=warp_device)
    calibration.forward(_array(lengths, warp_device))
    forward_gradient = calibration.forward_vjp(_array(seed, warp_device)).numpy().copy()
    h = 1e-4
    plus, _ = _decimal(spec, lengths + h)
    minus, _ = _decimal(spec, lengths - h)
    np.testing.assert_allclose(
        forward_gradient, seed * (plus - minus) / (2 * h), rtol=2e-8, atol=1e-10
    )
    assert np.all(np.diff(slopes) < 0)
    assert np.all(expected < slopes[0] * lengths + 1.0)
    calibration.inverse(_array(expected, warp_device))
    inverse_gradient = calibration.inverse_vjp(_array(seed, warp_device)).numpy().copy()

    # Independent inverse roots use Decimal arithmetic and no production curve.
    def root(target):
        lo, hi = 0.0, spec.maximum_length_mm
        for _ in range(100):
            middle = (lo + hi) / 2
            value, _ = _decimal(spec, [middle])
            if value[0] < target:
                lo = middle
            else:
                hi = middle
        return (lo + hi) / 2

    numerical = np.asarray([(root(value + h) - root(value - h)) / (2 * h) for value in expected])
    np.testing.assert_allclose(inverse_gradient, seed * numerical, rtol=3e-7, atol=1e-8)


def test_retained_derivatives_survive_input_and_result_mutation(warp_device):
    calibration = WaterEquivalentCalibration(_spec(), shape=(2,), device=warp_device)
    source = _array([2.0, 40.0], warp_device)
    depth = calibration.forward(source)
    _, forward_slopes = _decimal(_spec(), [2.0, 40.0])
    inverse_input = _array(depth.numpy(), warp_device)
    result = calibration.inverse(inverse_input)
    with calibration.context.scope():
        source.fill_(999.0)
        depth.fill_(99.0)
        inverse_input.fill_(88.0)
        result.fill_(777.0)
    np.testing.assert_allclose(
        calibration.forward_vjp(_array([1.0, 1.0], warp_device)).numpy(),
        forward_slopes,
        rtol=3e-14,
    )
    np.testing.assert_allclose(
        calibration.inverse_vjp(_array([1.0, 1.0], warp_device)).numpy(),
        1 / forward_slopes,
        rtol=3e-14,
    )


def test_inverse_range_and_iteration_bound_are_explicit(warp_device):
    calibration = WaterEquivalentCalibration(
        _spec(),
        shape=(1,),
        device=warp_device,
        execution=BeamHardeningExecution(maximum_iterations=1),
    )
    with pytest.raises(NumericalError, match="iteration bound"):
        calibration.inverse(_array([1.0], warp_device))
    calibration.clear_status()
    with pytest.raises(ContractError, match="calibration range"):
        calibration.inverse(_array([1e6], warp_device))


def test_calibration_layout_and_preparation_contracts(warp_device):
    spec = _spec()
    calibration = WaterEquivalentCalibration(spec, shape=(2,), device=warp_device)
    with pytest.raises(ContractError, match="dtype"):
        calibration.forward(wp.zeros(2, dtype=wp.float32, device=warp_device))
    with pytest.raises(ContractError, match="shape"):
        calibration.forward(_array([1.0], warp_device))
    with pytest.raises(ContractError, match="successful"):
        calibration.inverse_vjp(_array([1.0, 1.0], warp_device))
