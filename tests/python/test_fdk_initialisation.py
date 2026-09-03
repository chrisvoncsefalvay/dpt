"""Full-orbit FDK keeps physical normalisation and agrees with analytic spheres."""

import math
from typing import Literal, cast

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.fdk import FDKExecution, FDKSpec, _filter, fdk_initialise
from dpt.geometry import ConeBeamIntrinsics, RigidTransform
from dpt.volumes import GridSpec


def _angles(views: int = 64) -> tuple[float, ...]:
    return tuple(view * 2.0 * math.pi / views for view in range(views))


def _sphere_data(spec: FDKSpec, *, radius_mm: float = 3.0, attenuation: float = 0.02):
    """Closed-form finite central-sphere ray chords, independent of the FDK path."""
    geometry = spec.geometry
    rows, columns = geometry.shape
    u = (np.arange(columns) - 0.5 * (columns - 1)) * geometry.spacing_mm[
        0
    ] + geometry.detector_offset_mm[0]
    v = (np.arange(rows) - 0.5 * (rows - 1)) * geometry.spacing_mm[1] + geometry.detector_offset_mm[
        1
    ]
    transverse = v[:, None] ** 2 + u[None, :] ** 2
    distance_squared = (
        geometry.source_object_distance_mm**2
        * transverse
        / (geometry.source_detector_distance_mm**2 + transverse)
    )
    profile = 2.0 * attenuation * np.sqrt(np.maximum(radius_mm**2 - distance_squared, 0.0))
    return np.broadcast_to(profile, spec.shape).copy()


def test_ramp_filter_matches_independent_spatial_convolution() -> None:
    geometry = ConeBeamIntrinsics(80.0, 120.0, (3, 7), (0.4, 0.3), (0.07, -0.11))
    spec = FDKSpec(geometry, _angles(4))
    data = np.random.default_rng(94821).normal(size=spec.shape)
    actual = _filter(data, spec)
    expected = np.empty_like(actual)
    step = spec.virtual_column_spacing_mm
    for view in range(spec.shape[0]):
        for row in range(spec.shape[1]):
            v = (row - 1) * geometry.spacing_mm[1] + geometry.detector_offset_mm[1]
            for column in range(spec.shape[2]):
                terms = []
                for source_column in range(spec.shape[2]):
                    u = (source_column - 3) * geometry.spacing_mm[0] + geometry.detector_offset_mm[
                        0
                    ]
                    cosine = geometry.source_detector_distance_mm / math.hypot(
                        geometry.source_detector_distance_mm, u, v
                    )
                    lag = column - source_column
                    weight = (
                        0.25 / step
                        if lag == 0
                        else -1.0 / (math.pi**2 * lag**2 * step)
                        if lag % 2
                        else 0.0
                    )
                    terms.append(cosine * data[view, row, source_column] * weight)
                expected[view, row, column] = math.fsum(terms)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("window", ["ram-lak", "hann"])
def test_sphere_central_plane_normalisation(
    window: Literal["ram-lak", "hann"],
) -> None:
    geometry = ConeBeamIntrinsics(100.0, 150.0, (9, 129), (0.25, 0.25))
    spec = FDKSpec(geometry, _angles(), window=window)
    grid = GridSpec((1, 5, 5), (0.5, 0.5, 0.5), (-1.0, -1.0, 0.0))
    data = _sphere_data(spec)
    result = fdk_initialise(data, grid, spec)
    assert result.volume.shape == grid.shape
    assert result.volume.dtype == np.float64
    np.testing.assert_allclose(result.volume, 0.02, rtol=0.02, atol=0.0)
    blocked = fdk_initialise(data, grid, spec, execution=FDKExecution(block_voxels=7))
    np.testing.assert_array_equal(result.volume, blocked.volume)


def test_sphere_refinement_improves_central_value_without_empirical_scaling() -> None:
    grid = GridSpec((1, 1, 1), (0.5, 0.5, 0.5))
    errors = []
    for spacing, columns in ((0.5, 65), (0.25, 129)):
        spec = FDKSpec(
            ConeBeamIntrinsics(100.0, 150.0, (9, columns), (spacing, spacing)), _angles()
        )
        result = fdk_initialise(_sphere_data(spec), grid, spec)
        errors.append(abs(float(result.volume[0, 0, 0]) - 0.02))
    assert errors[1] < errors[0]
    assert errors[1] < 0.02 * 0.01


@pytest.mark.parametrize("scale", [10.0, 1000.0])
def test_millimetre_scaling_changes_reconstruction_as_inverse_length(scale: float) -> None:
    geometry = ConeBeamIntrinsics(100.0, 150.0, (7, 81), (0.3, 0.3))
    spec = FDKSpec(geometry, _angles(32))
    grid = GridSpec((1, 3, 3), (0.5, 0.5, 0.5), (-0.5, -0.5, 0.0))
    data = _sphere_data(spec)
    original = fdk_initialise(data, grid, spec).volume
    scaled_geometry = ConeBeamIntrinsics(
        100.0 * scale, 150.0 * scale, geometry.shape, (0.3 * scale, 0.3 * scale)
    )
    scaled_grid = GridSpec(
        grid.shape, (0.5 * scale, 0.5 * scale, 0.5 * scale), (-0.5 * scale, -0.5 * scale, 0.0)
    )
    scaled = fdk_initialise(data, scaled_grid, FDKSpec(scaled_geometry, spec.angles_rad)).volume
    np.testing.assert_allclose(scaled * scale, original, rtol=2e-13, atol=2e-14)


def test_virtual_detector_spacing_removes_magnification_dependence() -> None:
    grid = GridSpec((1, 5, 5), (0.5, 0.5, 0.5), (-1.0, -1.0, 0.0))
    volumes = []
    for magnification in (1.2, 2.0):
        geometry = ConeBeamIntrinsics(
            100.0, 100.0 * magnification, (7, 81), (0.2 * magnification, 0.2 * magnification)
        )
        spec = FDKSpec(geometry, _angles(32))
        assert spec.virtual_column_spacing_mm == pytest.approx(0.2, abs=3e-16)
        volumes.append(fdk_initialise(_sphere_data(spec), grid, spec).volume)
    np.testing.assert_allclose(volumes[0], volumes[1], rtol=2e-12, atol=2e-14)


def test_grid_centres_scan_centre_and_object_pose_are_consistent() -> None:
    geometry = ConeBeamIntrinsics(100.0, 150.0, (7, 81), (0.3, 0.3))
    spec = FDKSpec(geometry, _angles(32))
    data = _sphere_data(spec)
    grid = GridSpec((1, 3, 3), (0.5, 0.5, 0.5), (-0.5, -0.5, 0.0))
    original = fdk_initialise(data, grid, spec).volume
    shifted = ConeBeamIntrinsics(
        100.0, 150.0, geometry.shape, geometry.spacing_mm, centre_mm=(4.0, -3.0, 2.0)
    )
    moved = fdk_initialise(
        data,
        grid,
        FDKSpec(shifted, spec.angles_rad),
        pose=RigidTransform(translation_mm=shifted.centre_mm),
    ).volume
    np.testing.assert_allclose(moved, original, rtol=3e-13, atol=3e-14)


def test_fdk_rejects_short_duplicate_reversed_and_irregular_scan_angles() -> None:
    geometry = ConeBeamIntrinsics(100.0, 150.0, (3, 9), (0.3, 0.3))
    for angles in (
        (0.0, 1.0),
        (0.0, math.pi, 2.0 * math.pi),
        tuple(reversed(_angles(4))),
        (0.0, 0.5, 1.0, 1.5),
    ):
        with pytest.raises(ContractError, match="orbit"):
            FDKSpec(geometry, angles)
    with pytest.raises(ContractError, match="window"):
        FDKSpec(geometry, _angles(4), window=cast(Literal["ram-lak"], "cosine"))


def test_fdk_rejects_invalid_data_and_excessive_work() -> None:
    spec = FDKSpec(ConeBeamIntrinsics(100.0, 150.0, (3, 9), (0.3, 0.3)), _angles(4))
    grid = GridSpec((1, 3, 3), (0.5, 0.5, 0.5))
    data = np.ones(spec.shape, dtype=np.float64)
    with pytest.raises(ContractError, match="execution limits"):
        fdk_initialise(data, grid, spec, execution=FDKExecution(max_work=1))
    with pytest.raises(ContractError, match="shape"):
        fdk_initialise(data[:1], grid, spec)
    data.fill(float("nan"))
    with pytest.raises(ContractError, match="finite"):
        fdk_initialise(data, grid, spec)


def test_fdk_zero_data_and_source_plane_contract() -> None:
    spec = FDKSpec(ConeBeamIntrinsics(100.0, 150.0, (3, 9), (0.3, 0.3)), _angles(4))
    data = np.zeros(spec.shape, dtype=np.float64)
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    assert fdk_initialise(data, grid, spec).volume.tolist() == [[[0.0]]]
    behind = GridSpec((1, 1, 1), (1.0, 1.0, 1.0), (0.0, -101.0, 0.0))
    with pytest.raises(ContractError, match="source plane"):
        fdk_initialise(data, behind, spec)
