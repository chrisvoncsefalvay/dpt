"""Explicit metadata conversions and conservative energy support contracts."""

# pyright: reportUnknownMemberType=false
import math
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest

from dpt.geometry import DetectorGeometry
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.volumes import GridSpec


def test_grid_spec_conversion_matches_all_support_corners_and_storage_axes():
    sampled = GridSpec((3, 5, 7), (2.0, 3.0, 4.0), (11.0, -8.0, 21.0))
    cells = MaterialGrid.from_grid_spec(sampled, materials=2)
    assert cells.shape == (3, 5, 7)
    assert cells.shape_xyz == (7, 5, 3)
    assert cells.origin_mm == sampled.grid_to_object(sampled.support[0])
    assert cells.upper_mm == sampled.grid_to_object(sampled.support[1])
    assert cells.to_grid_spec() == sampled
    with pytest.raises(TransportError, match="axis-aligned"):
        MaterialGrid.from_grid_spec(
            replace(sampled, orientation=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
            materials=2,
        )


def test_detector_conversion_preserves_each_pixel_centre_and_row_major_shape():
    geometry = DetectorGeometry(
        (0.0, 0.0, -10.0), (11.0, -9.0, 20.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 3.0), (5, 7)
    )
    detector = PlanarDetector.from_detector_geometry(geometry)
    assert detector.lower_xy_mm == (10.0, -10.5)
    assert detector.shape == (7, 5)
    assert detector.image_shape == geometry.shape
    assert detector.to_detector_geometry(source_mm=geometry.source_mm) == geometry
    with pytest.raises(TransportError, match="positive x/y"):
        PlanarDetector.from_detector_geometry(
            replace(geometry, u=(0.0, 1.0, 0.0), v=(-1.0, 0.0, 0.0))
        )


def spec(**kwargs):
    return TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((0.0, 0.0), (1.0, 1.0), (1, 1), 2.0),
        80.0,
        "analytic free-electron support contract",
        scattering_law="free-electron-compton",
        coefficient_energies_kev=(0.0, 80.0),
        **kwargs,
    )


@pytest.mark.parametrize("events", [1, 4, 4096, 2**31 - 1])
def test_cover_budget_support_bound_matches_independent_compton_reciprocal_identity(events):
    model = spec(max_events=events, energy_support_policy="cover-budget")
    with localcontext() as ctx:
        ctx.prec = 80
        # Constant copied from the independent physical model, not production helper.
        expected = float(1 / (1 / Decimal(80) + 2 * Decimal(events) / Decimal("510.99895069")))
    assert model.minimum_energy_kev == pytest.approx(expected, rel=3e-16)
    assert (
        replace(
            model, coefficient_energies_kev=(model.minimum_energy_kev, 80.0)
        ).energy_support_policy
        == "cover-budget"
    )
    with pytest.raises(TransportError, match="collision budget"):
        replace(
            model,
            coefficient_energies_kev=(math.nextafter(model.minimum_energy_kev, math.inf), 80.0),
        )


def test_guard_and_budget_policies_never_invent_missing_coefficients():
    narrow = replace(spec(), coefficient_energies_kev=(79.0, 80.0))
    assert narrow.energy_nodes == (79.0, 80.0)
    assert narrow.energy_support_policy == "runtime-guard"
    with pytest.raises(TransportError, match="collision budget"):
        replace(narrow, energy_support_policy="cover-budget")
    elastic = replace(
        narrow, scattering_law="isotropic-elastic", energy_support_policy="cover-budget"
    )
    assert elastic.minimum_energy_kev == 80.0
    tiny = replace(spec(), energy_kev=1e-300, coefficient_energies_kev=(0.0, 1e-300))
    assert tiny.minimum_energy_kev == 1e-300


def test_structural_library_source_identity_uses_actual_counter_addresses():
    from dpt.transport.rng import HistoryBatch, SourceIdentity

    left = HistoryBatch(2, 0, 4, "same legacy label").with_library_source("parallel-rectangle-v1")
    right = HistoryBatch(2, 4, 4, "same legacy label").with_library_source("parallel-rectangle-v1")
    assert left.independent_of(right)
    assert left.source_identity is not None
    assert not SourceIdentity(2, 0, 4, "point-cone-area-v1").independent_of(left.source_identity)
    with pytest.raises(TransportError, match="match the transport history"):
        replace(left, source_identity=SourceIdentity(3, 0, 4, "parallel-rectangle-v1"))


def test_counter_contract_accepts_index_protocol_and_rejects_boolean_identity():
    from dpt.transport.rng import HistoryBatch

    class IntegralIdentity:
        def __index__(self):
            return 17

    batch = HistoryBatch(IntegralIdentity(), IntegralIdentity(), IntegralIdentity())
    assert (batch.seed, batch.first_history, batch.count) == (17, 17, 17)
    assert all(type(value) is int for value in (batch.seed, batch.first_history, batch.count))
    for values in ((True, 0, 1), (1, False, 1), (1, 0, True)):
        with pytest.raises(TransportError, match="integer history identity"):
            HistoryBatch(*values)


def test_oriented_grid_converter_matches_support_without_resampling():
    sampled = GridSpec(
        (3, 5, 7),
        (2.0, 3.0, 4.0),
        (11.0, -8.0, 21.0),
        (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    cells, pose = MaterialGrid.from_oriented_grid_spec(sampled, materials=2)
    assert pose.point(cells.origin_mm) == sampled.grid_to_object(sampled.support[0])
    assert pose.point(cells.upper_mm) == sampled.grid_to_object(sampled.support[1])
    assert cells.shape == sampled.shape and cells.materials == 2
