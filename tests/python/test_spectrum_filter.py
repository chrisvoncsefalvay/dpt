"""Independent Beer-Lambert filtering, units, edge handling and recipe identity."""

import hashlib
import math
from decimal import Decimal, localcontext
from typing import Literal

import pytest

from dpt.contracts import ContractError
from dpt.materials import MassFractionMixture, MaterialTable, Provenance
from dpt.spectral import Spectrum

SOURCE = Provenance(
    "analytic: source populations=(1000,2000) at (40,80)keV",
    hashlib.sha256(b"source populations=(1000,2000) at (40,80)keV").hexdigest(),
    "Original analytic fixture; Apache-2.0",
    "Supplied source populations used in Beer-Lambert tests",
)
TABLE_SOURCE = Provenance(
    "analytic: attenuation=(0.05,0.2)/mm at (40,80)keV",
    hashlib.sha256(b"attenuation=(0.05,0.2)/mm at (40,80)keV").hexdigest(),
    "Original analytic fixture; Apache-2.0",
    "Supplied attenuation fixture used in Beer-Lambert tests",
)


@pytest.mark.parametrize("units", ["mm^-1", "cm^-1", "cm^2/g"])
def test_filtering_uses_attenuation_in_mm_inverse_with_density_applied_once(
    units: Literal["mm^-1", "cm^-1", "cm^2/g"],
) -> None:
    coefficients = {"mm^-1": (0.05, 0.2), "cm^-1": (0.5, 2.0), "cm^2/g": (0.25, 1.0)}
    table = MaterialTable(
        "analytic filter",
        (40.0, 80.0),
        coefficients[units],
        units,
        TABLE_SOURCE,
        density_g_cm3=2.0 if units == "cm^2/g" else None,
    )
    source = Spectrum((40.0, 80.0), (1000.0, 2000.0), "bin-fluence", SOURCE)
    filtered = source.filtered(table, thickness_mm=5.0)
    assert filtered.values == pytest.approx((1000 * math.exp(-0.25), 2000 * math.exp(-1)))
    assert source.values == (1000.0, 2000.0)
    assert filtered.provenance is SOURCE
    assert filtered.filters[0].material.provenance is TABLE_SOURCE
    assert filtered.filters[0].recipe_sha256 not in (SOURCE.sha256, TABLE_SOURCE.sha256)


def test_density_weights_remain_separate_and_filter_thickness_vjp_has_independent_oracles() -> None:
    table = MaterialTable("analytic filter", (40.0, 80.0), (0.05, 0.2), "mm^-1")
    source = Spectrum((40.0, 80.0), (100.0, 200.0), "density", SOURCE, (2.0, 3.0))
    thickness = 3.5
    filtered = source.filtered(table, thickness)
    assert filtered.representation == "density"
    assert filtered.quadrature_weights_kev == (2.0, 3.0)
    expected = (200 * math.exp(-0.05 * thickness), 600 * math.exp(-0.2 * thickness))
    assert filtered.bin_fluence == pytest.approx(expected, rel=1e-14)
    cotangent = (2.0, -0.5)
    analytic = -2 * 0.05 * expected[0] + 0.5 * 0.2 * expected[1]
    assert filtered.filter_thickness_vjp(cotangent) == pytest.approx(analytic, rel=1e-13)
    step = 1e-4
    plus, minus = (source.filtered(table, thickness + delta).bin_fluence for delta in (step, -step))
    finite_difference = sum(cotangent[k] * (plus[k] - minus[k]) for k in range(2)) / (2 * step)
    assert filtered.filter_thickness_vjp(cotangent) == pytest.approx(finite_difference, rel=1e-9)


def test_filter_recipe_hashes_are_reproducible_and_preserve_successive_filter_metadata() -> None:
    table = MaterialTable("analytic filter", (40.0, 80.0), (0.05, 0.2), "mm^-1", TABLE_SOURCE)
    source = Spectrum((40.0, 80.0), (1000.0, 2000.0), "bin-fluence", SOURCE)
    first, repeat = source.filtered(table, 2.0), source.filtered(table, 2.0)
    assert first.filters == repeat.filters
    assert len(first.filters[0].recipe_sha256) == 64
    final = first.filtered(table, 3.0)
    assert len(final.filters) == 2
    assert final.filters[0] is first.filters[0]
    assert final.filters[1].recipe_sha256 != first.filters[0].recipe_sha256
    assert final.values == pytest.approx(source.filtered(table, 5.0).values, rel=1e-14)
    assert final.provenance is SOURCE
    assert final.filter_thickness_vjp((1.0, 1.0)) == pytest.approx(
        -sum(mu * value for mu, value in zip((0.05, 0.2), final.values, strict=True)), rel=1e-13
    )


def test_absorption_edges_and_between_node_interpolants() -> None:
    table = MaterialTable("analytic edge", (40.0, 50.0, 50.0, 80.0), (0.05, 0.1, 0.2, 0.4), "mm^-1")
    line = Spectrum((50.0,), (100.0,), "bin-fluence")
    with pytest.raises(ContractError, match="edge_side"):
        line.filtered(table, 2.0)
    assert line.filtered(table, 2.0, edge_side="below").values == pytest.approx(
        (100 * math.exp(-0.2),)
    )
    assert line.filtered(table, 2.0, edge_side="above").values == pytest.approx(
        (100 * math.exp(-0.4),)
    )
    intermediate = Spectrum((45.0,), (100.0,), "bin-fluence")
    mu = 0.05 * (45 / 40) ** (math.log(2) / math.log(50 / 40))
    assert intermediate.filtered(table, 2.0).values == pytest.approx((100 * math.exp(-2 * mu),))
    for energy in (39.0, 81.0):
        with pytest.raises(ContractError, match="outside supplied support"):
            Spectrum((energy,), (100.0,), "bin-fluence").filtered(table, 2.0)


def test_mass_mixture_filter_uses_its_density_and_original_component_metadata() -> None:
    first = MaterialTable("first", (40.0, 80.0), (0.2, 0.4), "cm^2/g", density_g_cm3=1)
    second = MaterialTable("second", (40.0, 80.0), (0.6, 0.8), "cm^2/g", density_g_cm3=7)
    mixture = MassFractionMixture("mixture", (first, second), (0.25, 0.75), 2.0)
    filtered = Spectrum((40.0, 80.0), (100.0, 100.0), "bin-fluence").filtered(mixture, 5)
    assert filtered.values == pytest.approx((100 * math.exp(-0.5), 100 * math.exp(-0.7)))
    assert filtered.filters[0].material is mixture


def test_deep_filter_products_and_thickness_derivatives_retain_their_exponent_range() -> None:
    spectrum = Spectrum((40.0,), (1e300,), "bin-fluence")
    table = MaterialTable("analytic deep filter", (40.0,), (1.0,), "mm^-1")
    filtered = spectrum.filtered(table, 750.0)
    with localcontext() as context:
        context.prec = 100
        expected = float(Decimal.from_float(1e300) * (-Decimal(750)).exp())
    assert filtered.values[0] == pytest.approx(expected, rel=1e-13)
    assert filtered.values[0] > 0
    derivative_source = Spectrum((40.0,), (1.0,), "bin-fluence")
    large_mu = MaterialTable("analytic derivative range", (40.0,), (1e300,), "mm^-1")
    thickness = 750.0 / 1e300
    derivative_spectrum = derivative_source.filtered(large_mu, thickness)
    assert derivative_spectrum.values == (0.0,)
    depth = float(1e300 * thickness)
    with localcontext() as context:
        context.prec = 100
        derivative = float(-Decimal.from_float(1e300) * (-Decimal(depth)).exp())
    assert derivative_spectrum.filter_thickness_vjp((1.0,)) == pytest.approx(derivative, rel=1e-13)


@pytest.mark.parametrize("thickness", [-1.0, True, float("nan"), float("inf")])
def test_filter_thickness_must_be_finite_and_nonnegative(thickness: float) -> None:
    source = Spectrum((40.0,), (1.0,), "bin-fluence")
    table = MaterialTable("analytic", (40.0,), (0.1,), "mm^-1")
    with pytest.raises(ContractError, match="thickness_mm"):
        source.filtered(table, thickness)


def test_filter_vjp_requires_a_filter_and_matching_finite_cotangents() -> None:
    source = Spectrum((40.0,), (1.0,), "bin-fluence")
    with pytest.raises(ContractError, match="filtered spectrum"):
        source.filter_thickness_vjp((1.0,))
    filtered = source.filtered(MaterialTable("analytic", (40.0,), (0.1,), "mm^-1"), 0.0)
    assert filtered.values == source.values
    assert filtered.filter_thickness_vjp((1.0,)) == pytest.approx(-0.1)
    with pytest.raises(ContractError):
        filtered.filter_thickness_vjp((1.0, 2.0))
    with pytest.raises(ContractError):
        filtered.filter_thickness_vjp((float("nan"),))
