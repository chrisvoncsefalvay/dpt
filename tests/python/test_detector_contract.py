"""Detector identifiability, spatial boundary and independent noise moment contracts."""

import hashlib
from dataclasses import replace

import pytest

from dpt.detector import BlurSpec, CalibrationSpec, ObservationIdentity
from dpt.materials import Provenance
from dpt.validation.detector import (
    compound_poisson_moments,
    matrix_product,
    spatial_response_matrix,
    spread_covariance,
)

FIXTURE = Provenance(
    source="analytic: one-row three-tap stencil h=(0.125,0.5,0.375)",
    sha256=hashlib.sha256(b"h=(0.125,0.5,0.375); boundary=zero").hexdigest(),
    rights="Original mathematical fixture; Apache-2.0",
    description="Analytic spatial stencil",
)


def test_gain_exposure_gauge_and_unconstrained_pixel_corrections_rejected() -> None:
    with pytest.raises(ValueError, match="gain or exposure"):
        CalibrationSpec("ADU", active_gain=True, active_exposure=True)
    with pytest.raises(ValueError, match="shared acquisition"):
        CalibrationSpec("ADU", shared_offset=False, active_offset=True)
    spec = CalibrationSpec("ADU", shared_gain=False)
    assert not spec.active_gain


def test_blur_boundary_is_part_of_the_linear_operator() -> None:
    spec = BlurSpec(1, 3, 1, 3, (0.125, 0.5, 0.375), FIXTURE)
    matrix = spatial_response_matrix(1, 3, 1, 3, spec.weights)
    assert matrix_product(matrix, (1.0, 1.0, 1.0)) == (0.875, 1.0, 0.625)
    assert matrix_product(matrix, (1.0, 0.0, 0.0), transpose=True) == (0.5, 0.375, 0.0)
    with pytest.raises(ValueError, match="odd"):
        replace(spec, kernel_width=2, weights=(0.5, 0.5))
    with pytest.raises(ValueError, match="at most one"):
        replace(spec, weights=(0.5, 0.5, 0.5))


def test_spatial_noise_is_correlated_after_spreading() -> None:
    matrix = spatial_response_matrix(1, 3, 1, 3, (0.125, 0.5, 0.375))
    covariance = spread_covariance(matrix, (2.0, 2.0, 2.0))
    assert covariance[0][1] > 0
    assert covariance[0][1] == covariance[1][0]


def test_compound_noise_has_poisson_cumulants_not_poisson_output_variance() -> None:
    mean, variance, fourth = compound_poisson_moments((2.0, 3.0), (40.0, 80.0))
    assert mean == 320.0
    assert variance == 22400.0
    assert fourth == 2 * 40**4 + 3 * 80**4 + 3 * variance**2


def test_observation_counter_namespaces_do_not_wrap() -> None:
    identity = ObservationIdentity(2**64 - 1, 2**32 - 1, pixel_offset=2**32 - 2)
    identity.validate_extent(2)
    with pytest.raises(ValueError, match="wrap"):
        identity.validate_extent(3)
    with pytest.raises(ValueError, match="namespace"):
        ObservationIdentity(1, 0, energy_offset=2**31 - 3).validate_extent(1, 2)
    with pytest.raises(ValueError, match="observation_id"):
        ObservationIdentity(1, -1)
