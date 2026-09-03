"""Derived geometry indices are checked without allocating large grids."""

# pyright: reportPrivateUsage=false
import math
from dataclasses import replace

import pytest

from dpt.transport.geometry_derivatives import _validate_geometry_index_budgets
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec


def _spec(shape, detector_shape=(1, 1)):
    return TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), tuple(reversed(shape)), 1),
        PlanarDetector((-1, -1), (1, 1), detector_shape, 2),
        80.0,
        "integer-only geometry index budget reference",
    )


def test_cell_count_does_not_imply_material_patch_count_fits():
    spec = _spec((715827882, 1, 1))
    assert math.prod(spec.grid.shape) < 2**31
    expected_patches = 5 * spec.grid.shape_xyz[0] + 1
    assert expected_patches == 3579139411 and expected_patches > 2**31 - 1
    with pytest.raises(TransportError, match="material patch indexing"):
        _validate_geometry_index_budgets(spec, 1, 1)


def test_column_offsets_and_masks_fit_individually_but_not_together():
    spec = _spec((33554428, 1, 1))
    planes = sum(size + 1 for size in spec.grid.shape_xyz)
    assert planes == 33554433 and 64 * planes == 2147483712
    _validate_geometry_index_budgets(spec, 63, 1)
    with pytest.raises(TransportError, match="source boundary and plane mask indexing"):
        _validate_geometry_index_budgets(spec, 64, 1)


def test_detector_boundary_offsets_are_independently_bounded():
    spec = _spec((1, 1, 1), (33554431, 1))
    assert 64 * sum(size + 1 for size in spec.detector.shape) > 2**31 - 1
    with pytest.raises(TransportError, match="detector boundary indexing"):
        _validate_geometry_index_budgets(spec, 64, 1)


def test_exact_signed_maximum_observation_and_coefficient_counts_are_admitted():
    spec = _spec((1, 1, 1))
    _validate_geometry_index_budgets(spec, 1, 2**31 - 1)
    with pytest.raises(TransportError, match="observation indexing"):
        _validate_geometry_index_budgets(spec, 1, 2**31)
    largest = replace(spec, grid=replace(spec.grid, materials=2**31 - 1))
    _validate_geometry_index_budgets(largest, 1, 1)
    with pytest.raises(TransportError, match="material-energy coefficient indexing"):
        _validate_geometry_index_budgets(replace(largest, coefficient_energies_kev=(40, 80)), 1, 1)
