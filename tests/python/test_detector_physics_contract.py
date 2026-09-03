"""CPU-safe response recipes, units, provenance and explicit model selection."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal, cast

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.detector_physics import DetectorResponse, FirstInteractionModel
from dpt.materials import MaterialTable, Provenance
from dpt.physical_data import PhotonCoefficientTable
from helpers.detector_physics_oracles import decimal_derivatives, decimal_response


def material(units: Literal["mm^-1", "cm^-1", "cm^2/g"] = "mm^-1") -> MaterialTable:
    coefficients = (0.5, 0.25) if units == "mm^-1" else (5.0, 2.5)
    return MaterialTable(
        "analytic total-interaction fixture",
        (20.0, 80.0),
        coefficients,
        units,
        provenance=Provenance(
            "analytic:mu=(0.5,0.25)/mm",
            "a" * 64,
            "CC0-1.0",
            "Analytic fixture, no physical scintillator data",
        ),
        interpolation="linear",
    )


@pytest.mark.parametrize(
    "kind", ["interaction-counting", "ideal-full-energy", "supplied-deposition"]
)
@pytest.mark.parametrize("thickness", [0.0, 1e-20, 0.75, 2000.0])
def test_response_recipe_matches_independent_exponential(
    kind: Literal["interaction-counting", "supplied-deposition", "ideal-full-energy"],
    thickness: float,
) -> None:
    provenance = Provenance(
        "analytic:conditional scores=(7,13)keV",
        "b" * 64,
        "CC0-1.0",
        "Analytic first-interaction deposition fixture",
    )
    model = (
        FirstInteractionModel(kind, (7.0, 13.0), provenance)
        if kind == "supplied-deposition"
        else FirstInteractionModel(kind)
    )
    table = material()
    response = DetectorResponse(table, table.energies_kev, thickness, model, "above", 0.7)
    expected = [
        decimal_response(mu, thickness, 0.7, score)
        for mu, score in zip(table.coefficients, model.scores(table.energies_kev), strict=True)
    ]
    np.testing.assert_allclose(response.values, expected, rtol=3e-15, atol=0)
    expected_vjp = sum(
        decimal_derivatives(mu, thickness, 0.7, score, seed)[0]
        for mu, score, seed in zip(
            table.coefficients, model.scores(table.energies_kev), (2.0, -1.0), strict=True
        )
    )
    assert response.thickness_vjp((2.0, -1.0)) == pytest.approx(expected_vjp, rel=1e-13, abs=1e-200)
    assert response.source_provenance == (table.provenance, model.provenance)
    assert len(response.recipe_sha256) == 64


def test_response_units_density_edge_and_recipe_identity() -> None:
    model = FirstInteractionModel("interaction-counting")
    table = material()
    recipe = DetectorResponse(table, table.energies_kev, 0.5, model, "above")
    converted = DetectorResponse(material("cm^-1"), table.energies_kev, 0.5, model, "above")
    np.testing.assert_array_equal(recipe.values, converted.values)
    assert recipe.response_unit == "interactions/incident photon"
    assert model.signal_unit == "expected interactions"
    assert replace(recipe, thickness_mm=1.0).recipe_sha256 != recipe.recipe_sha256
    assert replace(recipe).recipe_sha256 == recipe.recipe_sha256
    edge = replace(
        table, energies_kev=(20.0, 40.0, 40.0, 80.0), coefficients=(0.5, 0.25, 1.0, 0.125)
    )
    below = DetectorResponse(edge, (40.0,), 1.0, model, "below")
    above = DetectorResponse(edge, (40.0,), 1.0, model, "above")
    assert below.values[0] < above.values[0]
    assert below.recipe_sha256 != above.recipe_sha256
    mass = replace(table, units="cm^2/g", density_g_cm3=2.0, coefficients=(2.5, 1.25))
    np.testing.assert_array_equal(
        DetectorResponse(mass, table.energies_kev, 0.5, model, "above").values, recipe.values
    )


def test_deposition_is_explicit_bounded_and_has_its_own_source() -> None:
    with pytest.raises(ContractError, match="own Provenance"):
        FirstInteractionModel("supplied-deposition", (10.0,))
    provenance = material().provenance
    with pytest.raises(ContractError, match="exceed incident"):
        DetectorResponse(
            material(),
            (20.0, 80.0),
            1.0,
            FirstInteractionModel("supplied-deposition", (21.0, 10.0), provenance),
            "above",
        )
    with pytest.raises(ContractError, match="one conditional"):
        DetectorResponse(
            material(),
            (20.0, 80.0),
            1.0,
            FirstInteractionModel("supplied-deposition", (10.0,), provenance),
            "above",
        )
    with pytest.raises(ContractError, match="declared"):
        FirstInteractionModel(cast(Any, "unlabelled"))
    with pytest.raises(ContractError, match="only supplied"):
        FirstInteractionModel("ideal-full-energy", (1.0,))


@pytest.mark.parametrize("cosine", [0.0, -1.0, 1.001, float("nan")])
def test_incidence_domain(cosine: float) -> None:
    with pytest.raises(ContractError):
        DetectorResponse(
            material(), (20.0,), 1.0, FirstInteractionModel("interaction-counting"), "above", cosine
        )


def test_host_conversion_is_explicit_and_preserves_range() -> None:
    recipe = DetectorResponse(
        material(), (20.0,), 1e-50, FirstInteractionModel("interaction-counting"), "above"
    )
    with pytest.raises(ContractError, match="lost a finite positive"):
        recipe.to_host()
    assert recipe.to_host(dtype="float64").dtype == np.dtype("float64")


def test_scored_tail_survives_probability_underflow() -> None:
    table = MaterialTable("analytic weak fixture", (1e200,), (1e-200,), "mm^-1")
    response = DetectorResponse(
        table, table.energies_kev, 1e-200, FirstInteractionModel("ideal-full-energy"), "above"
    )
    assert response.values[0] == pytest.approx(1e-200, rel=2e-13)


@pytest.mark.parametrize("cosine,score", [(1e-200, 1.0), (1.0, 1e200)])
def test_subnormal_depth_intermediate_is_not_amplified(cosine: float, score: float) -> None:
    table = MaterialTable("analytic subnormal depth", (1e200,), (1e-200,), "mm^-1")
    model = (
        FirstInteractionModel("interaction-counting")
        if score == 1.0
        else FirstInteractionModel.ideal_full_energy()
    )
    response = DetectorResponse(table, table.energies_kev, 3e-124, model, "above", cosine)
    expected = decimal_response(1e-200, 3e-124, cosine, score)
    assert response.values[0] == pytest.approx(expected, rel=3e-13, abs=0)


def test_subnormal_vjp_numerator_is_not_amplified() -> None:
    table = MaterialTable("analytic subnormal derivative", (1.0,), (1e-200,), "mm^-1")
    provenance = Provenance("analytic:score=3e-124", "d" * 64, "CC0-1.0", "Analytic fixture")
    model = FirstInteractionModel("supplied-deposition", (3e-124,), provenance)
    response = DetectorResponse(table, table.energies_kev, 0.0, model, "above", 1e-200)
    expected = decimal_derivatives(1e-200, 0.0, 1e-200, 3e-124, 1.0)[0]
    assert response.thickness_vjp((1.0,)) == pytest.approx(expected, rel=3e-13, abs=0)


def branch_tables() -> tuple[MaterialTable, PhotonCoefficientTable]:
    provenance = Provenance(
        "analytic:first interaction coefficient decomposition",
        "e" * 64,
        "CC0-1.0",
        "Analytic mass-coefficient fixture",
    )
    total = MaterialTable(
        "analytic compound", (20.0, 80.0), (5.0, 2.0), "cm^2/g", provenance, 4.0, "linear"
    )
    photoelectric = PhotonCoefficientTable(
        "analytic compound", "photoelectric", (20.0, 80.0), (3.0, 0.5), 4.0, provenance, "linear"
    )
    return total, photoelectric


def test_photoelectric_branch_response_retains_typed_sources_and_recipe() -> None:
    total, photoelectric = branch_tables()
    model = FirstInteractionModel.photoelectric_only(
        total, photoelectric, (20.0, 80.0), edge_side="above"
    )
    np.testing.assert_allclose(model.scores((20.0, 80.0)), [12.0, 20.0], rtol=2e-16)
    response = DetectorResponse(total, (20.0, 80.0), 0.5, model, "above")
    expected = [
        decimal_response(mu, 0.5, 1.0, score)
        for mu, score in zip(total.linear_mm_inverse, (12.0, 20.0), strict=True)
    ]
    np.testing.assert_allclose(response.values, expected, rtol=1e-15)
    assert model.photoelectric_table is photoelectric
    assert response.source_provenance == (total.provenance, photoelectric.provenance)
    with pytest.raises(ContractError, match="retain its total"):
        replace(response, edge_side="below")
    with pytest.raises(ContractError, match="retain its total"):
        replace(response, energies_kev=(30.0, 70.0))
    with pytest.raises(ContractError, match="declared energy"):
        model.scores((30.0, 70.0))


def test_photoelectric_energy_weight_is_applied_before_branch_rounding() -> None:
    provenance = Provenance(
        "analytic:total=1e200,photoelectric=1e-200", "e" * 64, "CC0-1.0", "Analytic range fixture"
    )
    total = MaterialTable("analytic range", (1e200,), (1e200,), "cm^2/g", provenance, 10.0)
    branch = PhotonCoefficientTable(
        total.name, "photoelectric", total.energies_kev, (1e-200,), 10.0, provenance
    )
    model = FirstInteractionModel.photoelectric_only(total, branch, (1e200,), edge_side="above")
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 100
        expected = float(
            Decimal.from_float(1e200) * Decimal.from_float(1e-200) / Decimal.from_float(1e200)
        )
    assert model.scores((1e200,))[0] == pytest.approx(expected, rel=3e-15, abs=0)
    response = DetectorResponse(total, total.energies_kev, 1.0, model, "above")
    assert response.values[0] == pytest.approx(expected, rel=3e-15, abs=0)


@pytest.mark.parametrize("defect", ["kind", "name", "support", "density", "branch"])
def test_photoelectric_branch_requires_matching_physical_inputs(defect: str) -> None:
    total, photoelectric = branch_tables()
    variants: dict[str, Any] = {
        "kind": {"kind": "energy-absorption"},
        "name": {"name": "different compound"},
        "support": {"energies_kev": (20.0, 90.0)},
        "density": {"density_g_cm3": 3.0},
        "branch": {"coefficients": (6.0, 0.5)},
    }
    with pytest.raises(ContractError):
        FirstInteractionModel.photoelectric_only(
            total, replace(photoelectric, **variants[defect]), (20.0, 80.0), edge_side="above"
        )
