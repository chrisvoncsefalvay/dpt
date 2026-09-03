"""mu(E) = rho kappa(E)/10 in mm⁻¹ for mass coefficients kappa in cm²/g.

Keep interaction-specific coefficients separate from total attenuation. Read
caller-supplied NIST response bytes or query an installed xraylib provider; both
paths retain the actual input hash, density and one-sided edge convention.

    >>> from dpt.physical_data import xraylib_coefficients
    >>> # Requires the optional xraylib package.
    >>> # aluminum = xraylib_coefficients(13, (30., 60.), kind="total-attenuation",
    >>> #                                 density_g_cm3=2.6989)

Source: https://github.com/tschoonj/xraylib
Theory: https://physics.nist.gov/PhysRefData/Xcom/Text/intro.html
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal

from dpt.contracts import ContractError, finite_scalar, finite_tuple, integer
from dpt.materials import MaterialTable, Provenance, _log_ratio, validate_energy_grid

# Shared interpolation implementation is private; public tables keep their interaction kind.
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false

CoefficientKind = Literal[
    "total-attenuation",
    "photoelectric",
    "coherent-scattering",
    "incoherent-scattering",
    "energy-absorption",
]
Extrapolation = Literal["reject", "constant", "power-law"]
_KINDS = (
    "total-attenuation",
    "photoelectric",
    "coherent-scattering",
    "incoherent-scattering",
    "energy-absorption",
)
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")


def _mev_to_kev(token: str) -> float:
    """Shift the source decimal exponent before the sole binary64 conversion."""
    try:
        parts = Decimal(token).as_tuple()
        if not isinstance(parts.exponent, int):
            raise ContractError("NIST energy must be a finite decimal value")
        return float(Decimal((parts.sign, parts.digits, parts.exponent + 3)))
    except (InvalidOperation, OverflowError, ValueError) as error:
        raise ContractError("NIST energy cannot be converted from MeV to keV") from error


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    """Installed provider version and hashes of its Python/native implementation.

    File names are basenames, so provenance remains portable between machines.
    Cross-section data compiled into the native binary are covered by its hash.
    """

    name: str
    version: str
    artifacts: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.name != "xraylib" or not isinstance(self.version, str) or not self.version:
            raise ContractError("provider identity needs the xraylib version")
        if not isinstance(self.artifacts, tuple) or not self.artifacts:
            raise ContractError("provider identity needs immutable artifact hashes")
        names: set[str] = set()
        for entry in self.artifacts:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ContractError("provider artifact must be a (basename, sha256) tuple")
            name, digest = entry
            if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
                raise ContractError("provider artifact names must be basenames")
            if (
                name in names
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ContractError("provider artifact names must be unique with SHA256 hashes")
            names.add(name)


@dataclass(frozen=True, slots=True)
class PhotonCoefficientTable:
    """κ[k] in cm²/g, with energy[k] in keV and reference density in g/cm³.

    ``kind`` names the interaction represented by κ. Duplicate energy nodes
    retain the below/above absorption-edge values. Only total attenuation can
    become a primary-projector ``MaterialTable``.
    """

    name: str
    kind: CoefficientKind
    energies_kev: tuple[float, ...]
    coefficients: tuple[float, ...]
    density_g_cm3: float
    provenance: Provenance
    interpolation: Literal["linear", "log-log"] = "log-log"
    provider: ProviderIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ContractError("physical coefficient table needs a composition name")
        if self.kind not in _KINDS:
            raise ContractError("unsupported photon coefficient kind")
        if not isinstance(self.provenance, Provenance):
            raise ContractError("physical coefficients require input provenance")
        if self.provider is not None and not isinstance(self.provider, ProviderIdentity):
            raise ContractError("provider must be a ProviderIdentity record")
        self._interpolant()

    def _interpolant(self) -> MaterialTable:
        return MaterialTable(
            self.name,
            self.energies_kev,
            self.coefficients,
            "cm^2/g",
            self.provenance,
            self.density_g_cm3,
            self.interpolation,
        )

    @property
    def linear_mm_inverse(self) -> tuple[float, ...]:
        """rho kappa/10, with finite-positive range checked at conversion."""
        return self._interpolant().linear_mm_inverse

    def as_material_table(self) -> MaterialTable:
        """Return total primary attenuation for material projection."""
        if self.kind != "total-attenuation":
            raise ContractError("only total attenuation can become a primary MaterialTable")
        return self._interpolant()

    def at_energies(
        self,
        energies_kev: tuple[float, ...],
        *,
        edge_side: Literal["below", "above"],
        extrapolation: Extrapolation = "reject",
    ) -> tuple[float, ...]:
        """Interpolate μ in mm⁻¹, with explicit edge and extrapolation policies.

        ``constant`` extends the outward one-sided endpoint. ``power-law``
        extends the nearest distinct-node log-log slope and requires positive
        endpoint coefficients. Neither policy smooths an absorption edge.
        """
        finite_tuple(energies_kev, "query energies", positive=True)
        if extrapolation not in ("reject", "constant", "power-law"):
            raise ContractError("extrapolation must be reject, constant or power-law")
        material = self._interpolant()
        if extrapolation == "reject":
            return material.at_energies(energies_kev, edge_side=edge_side)
        if edge_side not in ("below", "above"):
            raise ContractError("edge_side must be explicitly below or above")
        grid, values = self.energies_kev, self.linear_mm_inverse
        result: list[float] = []
        for energy in energies_kev:
            if grid[0] <= energy <= grid[-1]:
                result.extend(material.at_energies((energy,), edge_side=edge_side))
                continue
            endpoint = 0 if energy < grid[0] else len(grid) - 1
            if extrapolation == "constant":
                result.append(values[endpoint])
                continue
            if self.interpolation != "log-log" or len(grid) < 2:
                raise ContractError("power-law extrapolation requires a log-log slope")
            # Duplicate endpoint nodes encode a jump: its outside slope is unspecified.
            neighbour = endpoint + 1 if endpoint == 0 else endpoint - 1
            if grid[endpoint] == grid[neighbour]:
                raise ContractError("power-law extrapolation is undefined at an endpoint edge")

            def signed_log_ratio(left: float, right: float) -> float:
                return _log_ratio(left, right) if left >= right else -_log_ratio(right, left)

            slope = signed_log_ratio(values[neighbour], values[endpoint]) / signed_log_ratio(
                grid[neighbour], grid[endpoint]
            )
            log_value = math.log(values[endpoint]) + slope * signed_log_ratio(
                energy, grid[endpoint]
            )
            try:
                value = math.exp(log_value)
            except OverflowError as error:
                raise ContractError("extrapolation overflowed the coefficient range") from error
            if value == 0 or not math.isfinite(value):
                raise ContractError("extrapolation lost a finite positive coefficient's range")
            result.append(value)
        return tuple(result)


class _NistHTML(HTMLParser):
    """Read table cells and PRE text without executing or retaining HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.pres: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._pre: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "pre":
            if self._pre is not None:
                raise ContractError("nested PRE data are unsupported")
            self._pre = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            assert self._row is not None
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag == "pre" and self._pre is not None:
            self.pres.append("".join(self._pre))
            self._pre = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)
        if self._pre is not None:
            self._pre.append(data)


def read_nist_coefficients(
    raw: bytes,
    *,
    name: str,
    kind: CoefficientKind,
    density_g_cm3: float,
    provenance: Provenance,
    format: Literal["xcom-html", "mass-attenuation-html"],
    interpolation: Literal["linear", "log-log"] = "log-log",
) -> PhotonCoefficientTable:
    """mu(E)=rho*kappa(E)/10 in mm^-1 for kappa in cm²/g and rho in g/cm³.

    Parse caller-supplied UTF-8 NIST response bytes after comparing their SHA256
    with ``provenance.sha256``. ``raw`` is a non-empty byte string of at most
    16 MiB and ``name`` identifies the composition. ``kind`` selects total
    attenuation, an interaction partial or energy absorption. XCOM HTML retains
    total attenuation with coherent scattering and distinct interaction columns,
    while mass-attenuation HTML provides total and energy-absorption columns.

    ``PhotonCoefficientTable`` retains length-N energy and mass-coefficient
    tuples in source row order, in keV and cm²/g, plus positive reference density
    in g/cm³ and provenance of the original bytes. Python binary64 conversion
    preserves duplicate energy nodes for below/above edge values, with the
    chosen linear/log-log interpolation policy. ``linear_mm_inverse`` applies
    the equation above when linear coefficients are requested. This is a fixed
    physical table and no parameter derivatives are produced.

        >>> import hashlib
        >>> from dpt.materials import Provenance
        >>> from dpt.physical_data import read_nist_coefficients
        >>> raw = bytes([10]).join((b"<pre>(MeV) (cm2/g) (cm2/g)",
        ...                         b"0.03 2.0 1.0", b"0.06 1.0 0.5</pre>"))
        >>> provenance = Provenance("analytic-format-fixture", hashlib.sha256(raw).hexdigest(),
        ...                         "CC0", "Two declared coefficient rows for parsing")
        >>> table = read_nist_coefficients(raw, name="declared test material",
        ...     kind="total-attenuation", density_g_cm3=1., provenance=provenance,
        ...     format="mass-attenuation-html")
        >>> (table.energies_kev, table.coefficients, table.linear_mm_inverse)
        ((30.0, 60.0), (2.0, 1.0), (0.2, 0.1))

    """
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 16 * 1024 * 1024:
        raise ContractError("NIST response must contain at most 16 MiB of bytes")
    if not isinstance(provenance, Provenance):
        raise ContractError("NIST input requires a Provenance record")
    if hashlib.sha256(raw).hexdigest() != provenance.sha256:
        raise ContractError("NIST response SHA256 does not match provenance")
    if kind not in _KINDS:
        raise ContractError("unsupported photon coefficient kind")
    try:
        html = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("NIST response must be UTF-8 HTML") from error
    parser = _NistHTML()
    parser.feed(html)
    parser.close()
    energies: list[float] = []
    coefficients: list[float] = []
    if format == "xcom-html":
        columns = {
            "coherent-scattering": 2,
            "incoherent-scattering": 3,
            "photoelectric": 4,
            "total-attenuation": 7,
        }
        if kind not in columns:
            raise ContractError("XCOM does not supply energy-absorption coefficients")
        column = columns[kind]
        unit_rows = [row for row in parser.rows if row and row[0].strip() == "MeV"]
        if len(unit_rows) != 1 or len(unit_rows[0]) != 8:
            raise ContractError("unrecognised XCOM energy/coefficient units header")
        unit = re.sub(r"\s", "", unit_rows[0][column - 1])
        if unit != "cm2/g":
            raise ContractError("selected XCOM coefficient must be in cm^2/g, not barns/atom")
        data_start = parser.rows.index(unit_rows[0]) + 1
        for row in parser.rows[data_start:]:
            if len(row) != 9 or any(_NUMBER.fullmatch(value) is None for value in row[1:]):
                raise ContractError("malformed XCOM numeric row")
            energies.append(_mev_to_kev(row[1]))
            coefficients.append(float(row[column]))
    elif format == "mass-attenuation-html":
        if kind not in ("total-attenuation", "energy-absorption"):
            raise ContractError("NIST mass-attenuation pages do not supply partial interactions")
        if len(parser.pres) != 1:
            raise ContractError("expected one NIST mass-attenuation PRE table")
        pre = parser.pres[0]
        if "(MeV)" not in pre or pre.count("(cm2/g)") != 2:
            raise ContractError("unrecognised NIST mass-attenuation units header")
        column = 1 if kind == "total-attenuation" else 2
        lines = pre.splitlines()
        unit_lines = [index for index, line in enumerate(lines) if "(MeV)" in line]
        if len(unit_lines) != 1:
            raise ContractError("expected one NIST mass-attenuation units line")
        for line in lines[unit_lines[0] + 1 :]:
            fields = line.split()
            if not fields or set(line.strip()) == {"_"}:
                continue
            if fields and re.fullmatch(r"[KLMNOPQ](?:[1-7])?", fields[0]):
                fields = fields[1:]
            if len(fields) != 3 or any(_NUMBER.fullmatch(value) is None for value in fields):
                raise ContractError("malformed NIST mass-attenuation numeric row")
            energies.append(_mev_to_kev(fields[0]))
            coefficients.append(float(fields[column]))
    else:
        raise ContractError("unsupported NIST response format")
    return PhotonCoefficientTable(
        name,
        kind,
        tuple(energies),
        tuple(coefficients),
        density_g_cm3,
        provenance,
        interpolation,
    )


def _xraylib_identity(module: Any) -> ProviderIdentity:
    native: Any = importlib.import_module("_xraylib")
    artifacts: list[tuple[str, str]] = []
    for implementation in (module, native):
        origin = getattr(implementation, "__file__", None)
        if not isinstance(origin, str):
            raise ContractError("xraylib provider has no hashable implementation artifact")
        path = Path(origin)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        artifacts.append((path.name, digest))
    return ProviderIdentity("xraylib", str(module.__version__), tuple(sorted(artifacts)))


def xraylib_coefficients(
    material: int | str,
    energies_kev: tuple[float, ...],
    *,
    kind: CoefficientKind,
    density_g_cm3: float,
    interpolation: Literal["linear", "log-log"] = "log-log",
) -> PhotonCoefficientTable:
    """mu(E)=rho*kappa_xraylib(E)/10 for the selected interaction coefficient.

    ``material`` is an atomic number in [1,100] or an xraylib chemical/NIST
    composition string. ``energies_kev`` is a length-N tuple of distinct,
    increasing positive keV nodes and ``density_g_cm3`` is a positive reference
    density in g/cm³. The returned ``PhotonCoefficientTable`` contains binary64
    length-N energies and mass coefficients in cm²/g, with the requested
    linear/log-log interpolation policy. Its linear conversion is in mm^-1.

    ``kind`` selects total attenuation, photoelectric, coherent or incoherent
    scattering, or energy absorption. Installed xraylib is imported on this
    call and provider domain errors are reported. Provenance hashes canonical
    JSON containing the actual mass outputs and installed provider identity,
    whose Python/native file hashes identify the coefficient implementation.
    Energy, density and composition are fixed and no derivatives are returned.

    """
    validate_energy_grid(energies_kev)
    density = finite_scalar(density_g_cm3, "density_g_cm3", minimum=0)
    if density == 0:
        raise ContractError("density_g_cm3 must be strictly positive")
    if kind not in _KINDS or interpolation not in ("linear", "log-log"):
        raise ContractError("unsupported coefficient kind or interpolation")
    compound = isinstance(material, str)
    if compound:
        if not material.strip():
            raise ContractError("xraylib composition must be non-empty")
    else:
        integer(material, "atomic number", minimum=1, maximum=100)
    try:
        module: Any = importlib.import_module("xraylib")
    except ModuleNotFoundError as error:
        if error.name != "xraylib":
            raise
        raise RuntimeError(
            "install the optional xraylib dependency to query coefficients"
        ) from error
    identity = _xraylib_identity(module)
    functions = {
        "total-attenuation": "CS_Total",
        "photoelectric": "CS_Photo",
        "coherent-scattering": "CS_Rayl",
        "incoherent-scattering": "CS_Compt",
        "energy-absorption": "CS_Energy",
    }
    function = getattr(module, functions[kind] + ("_CP" if compound else ""))
    try:
        coefficients = tuple(float(function(material, energy)) for energy in energies_kev)
    except (ValueError, TypeError, OverflowError) as error:
        raise ContractError(
            f"xraylib could not evaluate the supplied composition/energy: {error}"
        ) from error
    finite_tuple(coefficients, "xraylib mass coefficients")
    response = json.dumps(
        {
            "material": material,
            "kind": kind,
            "energies_kev": energies_kev,
            "coefficients_cm2_g": coefficients,
            "provider": identity.name,
            "version": identity.version,
            "artifacts": identity.artifacts,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    provenance = Provenance(
        "https://github.com/tschoonj/xraylib",
        hashlib.sha256(response).hexdigest(),
        "xraylib BSD licences: https://github.com/tschoonj/xraylib/blob/master/license_all.txt",
        "Canonical UTF-8 JSON of the requested mass coefficients and installed provider hashes; "
        "generated before conversion to linear mm^-1 coefficients.",
    )
    return PhotonCoefficientTable(
        str(material),
        kind,
        energies_kev,
        coefficients,
        density,
        provenance,
        interpolation,
        identity,
    )
