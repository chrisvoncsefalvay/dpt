"""Typed interaction, edge and verified-response ingestion contracts.

The local HTML here is an invented schema fixture. Official responses and the
actual optional provider are checked separately outside the distribution.
"""

import hashlib
import importlib
import math
import subprocess
import sys
from dataclasses import replace
from typing import Literal

import pytest

from dpt.contracts import ContractError
from dpt.materials import Provenance
from dpt.physical_data import (
    CoefficientKind,
    PhotonCoefficientTable,
    ProviderIdentity,
    read_nist_coefficients,
)

FIXTURE = Provenance(
    "analytic:schema-fixture",
    hashlib.sha256(b"typed edge fixture").hexdigest(),
    "Apache-2.0 original fixture",
    "Not physical reference data",
)


def table() -> PhotonCoefficientTable:
    return PhotonCoefficientTable(
        "defined fixture",
        "photoelectric",
        (10.0, 20.0, 20.0, 40.0),
        (4.0, 2.0, 8.0, 4.0),
        2.0,
        FIXTURE,
    )


def parse(
    raw: bytes,
    kind: CoefficientKind = "total-attenuation",
    format: Literal["xcom-html", "mass-attenuation-html"] = "xcom-html",
) -> PhotonCoefficientTable:
    return read_nist_coefficients(
        raw,
        name="schema fixture",
        kind=kind,
        density_g_cm3=2.0,
        provenance=replace(FIXTURE, sha256=hashlib.sha256(raw).hexdigest()),
        format=format,
    )  # type: ignore[arg-type]


def xcom(partial_unit: str = "cm<sup>2</sup>/g") -> bytes:
    # One selected-unit header, eight numeric quantities after the edge label.
    return (
        "<table><tr><td>MeV</td>"
        + f"<td>{partial_unit}</td>" * 5
        + "<td>cm<sup>2</sup>/g</td>" * 2
        + "</tr>"
        + "<tr><td>K</td><td>.02</td><td>1</td><td>2</td><td>3</td>"
        "<td>0</td><td>0</td><td>6</td><td>5</td></tr></table>"
    ).encode()


def test_partial_coefficients_do_not_become_total_attenuation() -> None:
    with pytest.raises(ContractError, match="only total"):
        table().as_material_table()
    assert replace(table(), kind="total-attenuation").as_material_table().units == "cm^2/g"


def test_explicit_edge_side_and_density_conversion() -> None:
    assert table().linear_mm_inverse == pytest.approx((0.8, 0.4, 1.6, 0.8))
    assert table().at_energies((20.0, 30.0), edge_side="above") == pytest.approx((1.6, 32 / 30))
    assert table().at_energies((20.0,), edge_side="below") == pytest.approx((0.4,))


@pytest.mark.parametrize("side", ["below", "above"])
def test_extrapolation_is_explicit_and_preserves_outward_endpoint(
    side: Literal["below", "above"],
) -> None:
    with pytest.raises(ContractError, match="outside"):
        table().at_energies((5.0,), edge_side=side)  # type: ignore[arg-type]
    assert table().at_energies((5.0, 80.0), edge_side=side, extrapolation="constant") == (0.8, 0.8)  # type: ignore[arg-type]
    assert table().at_energies(
        (5.0, 80.0), edge_side=side, extrapolation="power-law"
    ) == pytest.approx((1.6, 0.4))  # type: ignore[arg-type]


def test_extrapolation_rejects_endpoint_edge_and_undefined_slope() -> None:
    edge = replace(table(), energies_kev=(10.0, 10.0, 20.0, 40.0))
    with pytest.raises(ContractError, match="endpoint edge"):
        edge.at_energies((5.0,), edge_side="below", extrapolation="power-law")
    with pytest.raises(ContractError, match="log-log slope"):
        replace(table(), interpolation="linear").at_energies(
            (5.0,), edge_side="below", extrapolation="power-law"
        )


def test_extrapolation_retains_adjacent_energy_slope_and_reports_range_loss() -> None:
    edge = PhotonCoefficientTable(
        "adjacent nodes",
        "photoelectric",
        (1.0, math.nextafter(1.0, 2.0)),
        (1.0, math.nextafter(1.0, 2.0)),
        10.0,
        FIXTURE,
    )
    assert edge.at_energies((2.0,), edge_side="above", extrapolation="power-law") == pytest.approx(
        (2.0,)
    )
    steep = replace(edge, energies_kev=(1.0, 2.0), coefficients=(1e300, 1e-300))
    with pytest.raises(ContractError, match="range"):
        steep.at_energies((4.0,), edge_side="above", extrapolation="power-law")


@pytest.mark.parametrize(
    "kind,column_value",
    [
        ("total-attenuation", 6.0),
        ("photoelectric", 3.0),
        ("coherent-scattering", 1.0),
        ("incoherent-scattering", 2.0),
    ],
)
def test_xcom_explicit_selected_mass_column(kind: CoefficientKind, column_value: float) -> None:
    result = parse(xcom(), kind)
    assert result.energies_kev == (20.0,)
    assert result.coefficients == (column_value,)
    assert result.linear_mm_inverse == pytest.approx((column_value * 0.2,))


def test_xcom_mixed_units_are_checked_per_selected_column() -> None:
    with pytest.raises(ContractError, match="barns/atom"):
        parse(xcom("barns/atom"), "photoelectric")
    assert parse(xcom("barns/atom")).coefficients == (6.0,)
    with pytest.raises(ContractError, match="does not supply energy"):
        parse(xcom(), "energy-absorption")


def test_mass_attenuation_pre_preserves_edges_without_reading_duplicate_html_table() -> None:
    raw = b"""<table><tr><td>fake duplicate values</td></tr></table><PRE>
    Energy mu/rho mu_en/rho
    (MeV) (cm<sup>2</sup>/g) (cm<sup>2</sup>/g)
    .01 4 3
    .02 2 1
    K .02 8 6
    .04 4 3
    </PRE>"""
    result = parse(raw, "energy-absorption", "mass-attenuation-html")
    assert result.energies_kev == (10.0, 20.0, 20.0, 40.0)
    assert result.coefficients == (3.0, 1.0, 6.0, 3.0)
    assert result.at_energies((20.0,), edge_side="above") == pytest.approx((1.2,))


def test_decimal_mev_edge_is_converted_once_before_binary64_rounding() -> None:
    raw = b"""<PRE>Energy (MeV) (cm<sup>2</sup>/g) (cm<sup>2</sup>/g)
    1.00000E-03 1 1
    1.55960E-03 2 2
    K 1.55960E-03 8 8
    2.00000E-03 4 4
    </PRE>"""
    result = parse(raw, "total-attenuation", "mass-attenuation-html")
    assert result.energies_kev[1:3] == (1.5596, 1.5596)
    assert result.at_energies((1.5596,), edge_side="below") == pytest.approx((0.4,))
    assert result.at_energies((1.5596,), edge_side="above") == pytest.approx((1.6,))


def test_response_hash_is_checked_before_parsing() -> None:
    with pytest.raises(ContractError, match="SHA256"):
        read_nist_coefficients(
            xcom(),
            name="fixture",
            kind="total-attenuation",
            density_g_cm3=1.0,
            provenance=FIXTURE,
            format="xcom-html",
        )


@pytest.mark.parametrize(
    "raw",
    [b"<table></table>", xcom().replace(b">6<", b">nan<"), xcom().replace(b">MeV<", b">keV<")],
)
def test_malformed_xcom_response_is_rejected(raw: bytes) -> None:
    with pytest.raises(ContractError):
        parse(raw)


@pytest.mark.parametrize("token", [b"nan", b"inf", b"broken", b"1e-"])
def test_corrupt_energy_row_is_not_silently_removed(token: bytes) -> None:
    corrupt = xcom().replace(b">.02<", b">" + token + b"<")
    with pytest.raises(ContractError, match="numeric row"):
        parse(corrupt)
    raw = (
        b"<PRE>(MeV) (cm<sup>2</sup>/g) (cm<sup>2</sup>/g)\n"
        b".01 4 3\n" + token + b" 2 1\n.03 8 6\n</PRE>"
    )
    with pytest.raises(ContractError, match="numeric row"):
        parse(raw, "total-attenuation", "mass-attenuation-html")


def test_provider_identity_has_portable_unique_actual_hashes() -> None:
    valid = ProviderIdentity("xraylib", "fixture", (("native.so", "a" * 64),))
    assert valid.artifacts[0][0] == "native.so"
    with pytest.raises(ContractError, match="basenames"):
        replace(valid, artifacts=(("/private/path/native.so", "a" * 64),))
    with pytest.raises(ContractError, match="unique"):
        replace(valid, artifacts=(("native.so", "a" * 64), ("native.so", "b" * 64)))


def test_optional_provider_import_is_lazy_and_missing_dependency_is_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dpt.physical_data import xraylib_coefficients

    original = importlib.import_module

    def absent(name: str):
        if name == "xraylib":
            raise ModuleNotFoundError("optional xraylib missing", name="xraylib")
        return original(name)

    monkeypatch.setattr(importlib, "import_module", absent)
    with pytest.raises(RuntimeError, match="optional xraylib"):
        xraylib_coefficients(13, (60.0,), kind="total-attenuation", density_g_cm3=2.6989)


def test_provider_queries_cannot_invent_a_two_sided_edge_by_repeating_one_energy() -> None:
    from dpt.physical_data import xraylib_coefficients

    with pytest.raises(ContractError, match="strictly increasing"):
        xraylib_coefficients(13, (1.5596, 1.5596), kind="total-attenuation", density_g_cm3=2.6989)


def test_ingestion_imports_without_numerical_or_provider_dependencies() -> None:
    script = (
        "import sys; import dpt.physical_data, dpt.reference_data; "
        "assert not ({'numpy','warp','torch','xraylib','scipy'} & set(sys.modules))"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
