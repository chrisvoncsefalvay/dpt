# pytest.approx has intentionally dynamic third-party stubs.
# pyright: reportUnknownMemberType=false
"""Physical-table ingestion contracts; these tests do not import Warp."""

import hashlib
from dataclasses import replace

import pytest

from dpt.materials import MaterialBasis, MaterialTable, Provenance

FIXTURE = Provenance(
    source="analytic:test_materials_contract; piecewise values are the fixture definition",
    sha256=hashlib.sha256(b"E=(10,20,20,40)keV; kappa=(4,2,8,4)cm2/g; rho=2g/cm3").hexdigest(),
    rights="Original mathematical fixture; Apache-2.0",
    description="Not a physical material table",
)


def table() -> MaterialTable:
    return MaterialTable(
        "analytic edge",
        (10.0, 20.0, 20.0, 40.0),
        (4.0, 2.0, 8.0, 4.0),
        "cm^2/g",
        FIXTURE,
        density_g_cm3=2.0,
    )


def test_mass_to_linear_conversion_uses_density_once() -> None:
    assert table().linear_mm_inverse == pytest.approx((0.8, 0.4, 1.6, 0.8))
    with pytest.raises(ValueError, match="already include density"):
        replace(table(), units="mm^-1")
    with pytest.raises(ValueError, match="density"):
        replace(table(), density_g_cm3=None)


def test_absorption_edge_does_not_become_a_smooth_interval() -> None:
    material = table()
    assert material.at_energies((20.0,), edge_side="below") == pytest.approx((0.4,))
    assert material.at_energies((20.0,), edge_side="above") == pytest.approx((1.6,))
    assert material.at_energies((15.0,), edge_side="above") == pytest.approx((8.0 / 15.0,))
    assert material.at_energies((30.0,), edge_side="above") == pytest.approx((32.0 / 30.0,))


@pytest.mark.parametrize("energy", [0.0, 9.99, 40.01, float("nan"), float("inf")])
def test_outside_energy_support_never_extrapolates(energy: float) -> None:
    with pytest.raises(ValueError):
        table().at_energies((energy,), edge_side="above")


def test_invalid_edges_and_log_zero_are_rejected() -> None:
    with pytest.raises(ValueError, match="two one-sided"):
        replace(table(), energies_kev=(10.0, 20.0, 20.0, 20.0))
    with pytest.raises(ValueError, match="positive coefficients"):
        replace(table(), coefficients=(1.0, 0.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="non-decreasing"):
        replace(table(), energies_kev=(10.0, 21.0, 20.0, 40.0))


def test_basis_records_mix_and_preserves_material_major_order() -> None:
    basis = MaterialBasis(
        (table(), replace(table(), name="second material")),
        "Analytic fixed-density volume fractions",
    )
    assert basis.coefficients_at((10.0, 40.0), edge_side="above") == pytest.approx((0.8,) * 4)
    with pytest.raises(ValueError, match="unique"):
        MaterialBasis((table(), table()), "defined basis")
    with pytest.raises(ValueError, match="mixing"):
        MaterialBasis((table(),), "")


def test_provenance_requires_actual_input_identity_and_rights() -> None:
    with pytest.raises(ValueError, match="sha256"):
        replace(FIXTURE, sha256="looks like a DOI")
    with pytest.raises(ValueError, match="rights"):
        replace(FIXTURE, rights="")


def test_extreme_supported_energy_ratios_do_not_overflow_interpolation() -> None:
    material = MaterialTable("analytic wide support", (1e-300, 1e300), (1.0, 4.0), "mm^-1", FIXTURE)
    assert material.at_energies((1.0,), edge_side="above") == pytest.approx((2.0,))


def test_conversion_rejects_positive_underflow_instead_of_changing_material() -> None:
    material = MaterialTable(
        "analytic range limit", (10.0,), (1e-300,), "cm^2/g", FIXTURE, density_g_cm3=1e-300
    )
    with pytest.raises(ValueError, match="underflowed"):
        _ = material.linear_mm_inverse
