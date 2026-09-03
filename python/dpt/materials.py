"""Explicit, CPU-safe ingestion of primary attenuation coefficients.

There are no bundled physical tables. A caller supplies the values, identifies
where they came from and records the rights under which they may be used.
Interpolation is confined to the supplied support; repeated energies encode the
two one-sided values at an absorption edge rather than an interval to smooth.
"""

from __future__ import annotations

import importlib

# Public Python boundaries validate runtime inputs; workspace scratch stays module-owned.
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false
import math
import re
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

from dpt.contracts import ContractError, finite_scalar, finite_tuple


def _host_values(values: Any, *, dtype: Literal["float32", "float64"]) -> Any:
    """An explicit conversion boundary, with positive range loss reported."""
    if dtype not in ("float32", "float64"):
        raise ContractError("host coefficient precision must be float32 or float64")
    np: Any = importlib.import_module("numpy")
    source = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore"):
        result = np.asarray(source, dtype=dtype, order="C")
    if not np.isfinite(result).all() or ((source > 0) & (result == 0)).any():
        raise ContractError("explicit host conversion lost a finite positive value's range")
    return result


def _upload_values(values: Any, context: Any, *, dtype: Literal["float32", "float64"]) -> Any:
    scalar = context.wp.float64 if dtype == "float64" else context.wp.float32
    with context.scope():
        result = context.wp.array(values, dtype=scalar, device=context.device)
    context.wp.synchronize_stream(context.stream)
    return result


def _uniform_bin_averages(
    support: tuple[float, ...],
    bin_edges_kev: tuple[float, ...],
    evaluate: Callable[[tuple[float, ...]], tuple[float, ...]],
    *,
    weighting: Literal["uniform-energy"],
    quadrature_order: int,
) -> tuple[float, ...]:
    from dpt.contracts import integer

    validate_energy_grid(bin_edges_kev)
    if len(bin_edges_kev) < 2:
        raise ContractError("bin averaging requires at least two bin edges")
    if weighting != "uniform-energy":
        raise ContractError("bin averaging requires declared uniform-energy weighting")
    integer(quadrature_order, "quadrature_order", minimum=1, maximum=128)
    if bin_edges_kev[0] < support[0] or bin_edges_kev[-1] > support[-1]:
        raise ContractError("bin edges lie outside supplied support")
    np: Any = importlib.import_module("numpy")
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    averages: list[float] = []
    for low, high in pairwise(bin_edges_kev):
        splits = (low, *sorted({e for e in support if low < e < high}), high)
        contributions: list[float] = []
        for left, right in pairwise(splits):
            centre, radius = left / 2 + right / 2, (right - left) / 2
            energies = tuple(float(centre + radius * node) for node in nodes)
            if any(not left < energy < right for energy in energies):
                raise ContractError("energy sub-bin has no representable interior quadrature nodes")
            values = evaluate(energies)
            fraction = (right - left) / (high - low)
            contributions.extend(
                float(weight) * value * fraction / 2
                for weight, value in zip(weights, values, strict=True)
            )
        averages.append(math.fsum(contributions))
    return tuple(averages)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Identity of the actual input, distinct from a bibliographic citation.

    ``sha256`` identifies the original source bytes, before unit conversion.
    Analytic fixtures must say so in ``description`` and cite their defining
    formula in ``source``; a real material name is not fixture provenance.
    """

    source: str
    sha256: str
    rights: str
    description: str

    def __post_init__(self) -> None:
        for name in ("source", "rights", "description"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"provenance {name} must be non-empty")
        if not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ContractError("provenance sha256 must contain 64 lowercase hexadecimal digits")


def validate_energy_grid(energies_kev: tuple[float, ...]) -> None:
    """Validate distinct, positive keV nodes shared by spectral inputs and response."""
    finite_tuple(energies_kev, "energies_kev", positive=True)
    if any(a >= b for a, b in pairwise(energies_kev)):
        raise ContractError("energy nodes must be strictly increasing")


def _log_ratio(high: float, low: float) -> float:
    """Retain adjacent large energies without overflowing a widely separated ratio."""
    relative = (high - low) / low
    return math.log1p(relative) if math.isfinite(relative) else math.log(high) - math.log(low)


@dataclass(frozen=True, slots=True)
class MaterialTable:
    """Total primary attenuation at one declared material composition/density.

    Arrays are immutable host metadata, never an alternative production backend.
    Energies are keV. Duplicate adjacent energies encode below/above edge values,
    in that order; at most two entries may share an energy. The density attached
    to a mass table is in g/cm³. Linear coefficients already include density.
    """

    name: str
    energies_kev: tuple[float, ...]
    coefficients: tuple[float, ...]
    units: Literal["mm^-1", "cm^-1", "cm^2/g"]
    provenance: Provenance | None = None
    density_g_cm3: float | None = None
    interpolation: Literal["linear", "log-log"] = "log-log"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ContractError("a material table needs an unambiguous material/composition name")
        if self.provenance is not None and not isinstance(self.provenance, Provenance):
            raise ContractError(
                "material-table provenance must be a Provenance record when supplied"
            )
        finite_tuple(self.energies_kev, "energies_kev", positive=True)
        finite_tuple(self.coefficients, "coefficients")
        if len(self.energies_kev) != len(self.coefficients):
            raise ContractError("energies and coefficients must have equal lengths")
        if self.units not in ("mm^-1", "cm^-1", "cm^2/g"):
            raise ContractError("unsupported coefficient units")
        if self.interpolation not in ("linear", "log-log"):
            raise ContractError("interpolation must be linear or log-log")
        if self.interpolation == "log-log" and any(c <= 0 for c in self.coefficients):
            raise ContractError("log-log interpolation requires strictly positive coefficients")
        for index, energy in enumerate(self.energies_kev[1:], 1):
            if energy < self.energies_kev[index - 1]:
                raise ContractError("energy support must be non-decreasing")
            if index > 1 and energy == self.energies_kev[index - 2]:
                raise ContractError("an absorption edge has exactly two one-sided entries")
        if self.units == "cm^2/g":
            density = self.density_g_cm3
            density = finite_scalar(density, "reference density in g/cm³", minimum=0)
            if density <= 0:
                raise ContractError("reference density must be strictly positive")
        elif self.density_g_cm3 is not None:
            raise ContractError(
                "linear coefficients already include density; do not apply it twice"
            )

    @property
    def linear_mm_inverse(self) -> tuple[float, ...]:
        """Convert source values once, without inventing a different mixture model."""
        scale = 1.0
        if self.units == "cm^-1":
            scale = 0.1
        elif self.units == "cm^2/g":
            assert self.density_g_cm3 is not None
            scale = self.density_g_cm3 / 10.0
        converted = tuple(c * scale for c in self.coefficients)
        if any(not math.isfinite(c) for c in converted):
            raise ContractError("unit conversion overflowed; coefficient table is unusable")
        if any(
            source > 0 and target == 0
            for source, target in zip(self.coefficients, converted, strict=True)
        ):
            raise ContractError("unit conversion underflowed a positive attenuation coefficient")
        return converted

    def at_energies(
        self,
        energies_kev: tuple[float, ...],
        *,
        edge_side: Literal["below", "above"],
    ) -> tuple[float, ...]:
        """Interpolate in mm⁻¹, requiring the one-sided convention at exact edges."""
        return self._at_values(energies_kev, self.linear_mm_inverse, edge_side=edge_side)

    def _at_values(
        self,
        energies_kev: tuple[float, ...],
        coefficients: tuple[float, ...],
        *,
        edge_side: Literal["below", "above"],
    ) -> tuple[float, ...]:
        if edge_side not in ("below", "above"):
            raise ContractError("edge_side must be explicitly below or above")
        finite_tuple(energies_kev, "query energies", positive=True)
        grid = self.energies_kev
        result: list[float] = []
        for energy in energies_kev:
            if energy < grid[0] or energy > grid[-1]:
                raise ContractError(f"{self.name}: {energy} keV lies outside supplied support")
            left, right = bisect_left(grid, energy), bisect_right(grid, energy)
            if left != right:
                result.append(coefficients[left if edge_side == "below" else right - 1])
                continue
            low, high = left - 1, left
            if self.interpolation == "log-log":
                fraction = _log_ratio(energy, grid[low]) / _log_ratio(grid[high], grid[low])
                value = math.exp(
                    (1.0 - fraction) * math.log(coefficients[low])
                    + fraction * math.log(coefficients[high])
                )
            else:
                fraction = (energy - grid[low]) / (grid[high] - grid[low])
                value = (1.0 - fraction) * coefficients[low] + fraction * coefficients[high]
            result.append(value)
        return tuple(result)

    def bin_averages(
        self,
        bin_edges_kev: tuple[float, ...],
        *,
        weighting: Literal["uniform-energy"],
        quadrature_order: int = 16,
    ) -> tuple[float, ...]:
        """Average the supplied interpolant under explicit uniform energy weighting.

        Gaussian quadrature splits at every table node, including absorption
        edges. This is an average of coefficients, not the exact transmission
        of a finite bin; nonuniform spectra require their own declared weights.
        """
        return _uniform_bin_averages(
            self.energies_kev,
            bin_edges_kev,
            lambda energies: self.at_energies(energies, edge_side="above"),
            weighting=weighting,
            quadrature_order=quadrature_order,
        )

    def to_host(
        self,
        energies_kev: tuple[float, ...],
        *,
        edge_side: Literal["below", "above"],
        dtype: Literal["float32", "float64"] = "float32",
    ) -> Any:
        """Convert point-sampled linear coefficients to an explicit host precision."""
        return _host_values(self.at_energies(energies_kev, edge_side=edge_side), dtype=dtype)

    def to_device(
        self,
        energies_kev: tuple[float, ...],
        context: Any,
        *,
        edge_side: Literal["below", "above"],
        dtype: Literal["float32", "float64"] = "float32",
    ) -> Any:
        """Upload supplied point samples once to the context's owning stream."""
        return _upload_values(
            self.to_host(energies_kev, edge_side=edge_side, dtype=dtype), context, dtype=dtype
        )


@dataclass(frozen=True, slots=True)
class MassFractionMixture:
    """Declared homogeneous mass mixture of supplied total-attenuation tables.

    kappa_mix(E)=sum_i w_i*kappa_i(E) in cm^2/g; the explicit mixture density
    converts that mass coefficient to mm^-1 once. This is the additivity rule
    in NIST X-Ray Mass Attenuation Coefficients, section 2, equation (6):
    https://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html
    All original component tables and their provenance remain in ``tables``.
    Component reference densities do not enter the mixture mass coefficient.
    """

    name: str
    tables: tuple[MaterialTable, ...]
    mass_fractions: tuple[float, ...]
    density_g_cm3: float
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ContractError("a mass mixture requires an explicit name")
        if (
            not isinstance(self.tables, tuple)
            or not self.tables
            or any(not isinstance(t, MaterialTable) for t in self.tables)
        ):
            raise ContractError(
                "mass mixture components must be an immutable tuple of supplied tables"
            )
        if any(table.units != "cm^2/g" for table in self.tables):
            raise ContractError("mass mixture components require coefficients in cm^2/g")
        finite_tuple(self.mass_fractions, "mass_fractions")
        if len(self.tables) != len(self.mass_fractions):
            raise ContractError("one mass fraction is required per component table")
        if abs(math.fsum(self.mass_fractions) - 1.0) > 8 * 2**-52:
            raise ContractError("mass fractions must sum to one; no normalization is inferred")
        if finite_scalar(self.density_g_cm3, "mixture density in g/cm³", minimum=0) <= 0:
            raise ContractError("mixture density must be strictly positive")
        if self.provenance is not None and not isinstance(self.provenance, Provenance):
            raise ContractError("mixture provenance must be a Provenance record when supplied")

    @property
    def component_provenance(self) -> tuple[Provenance | None, ...]:
        return tuple(table.provenance for table in self.tables)

    def at_energies(
        self, energies_kev: tuple[float, ...], *, edge_side: Literal["below", "above"]
    ) -> tuple[float, ...]:
        """Evaluate original edge-aware interpolants, then apply mass fractions/density."""
        finite_tuple(energies_kev, "query energies", positive=True)
        if edge_side not in ("below", "above"):
            raise ContractError("edge_side must be explicitly below or above")
        sampled = [
            (fraction, table._at_values(energies_kev, table.coefficients, edge_side=edge_side))
            for table, fraction in zip(self.tables, self.mass_fractions, strict=True)
            if fraction > 0
        ]
        result: list[float] = []
        scale = self.density_g_cm3 / 10.0
        for index in range(len(energies_kev)):
            try:
                mass = math.fsum(fraction * values[index] for fraction, values in sampled)
            except OverflowError as error:
                raise ContractError("mass mixture overflowed its coefficient range") from error
            value = mass * scale
            positive = any(values[index] > 0 for _, values in sampled)
            if not math.isfinite(value) or (positive and value == 0):
                raise ContractError(
                    "mixture density conversion lost a positive coefficient's range"
                )
            result.append(value)
        return tuple(result)

    def bin_averages(
        self,
        bin_edges_kev: tuple[float, ...],
        *,
        weighting: Literal["uniform-energy"],
        quadrature_order: int = 16,
    ) -> tuple[float, ...]:
        active = [
            table
            for table, fraction in zip(self.tables, self.mass_fractions, strict=True)
            if fraction > 0
        ]
        low = max(table.energies_kev[0] for table in active)
        high = min(table.energies_kev[-1] for table in active)
        if low >= high:
            raise ContractError("active mixture tables have no common bin-averaging support")
        support = tuple(
            sorted(
                {
                    low,
                    high,
                    *(
                        energy
                        for table in active
                        for energy in table.energies_kev
                        if low < energy < high
                    ),
                }
            )
        )
        return _uniform_bin_averages(
            support,
            bin_edges_kev,
            lambda energies: self.at_energies(energies, edge_side="above"),
            weighting=weighting,
            quadrature_order=quadrature_order,
        )

    def to_host(
        self,
        energies_kev: tuple[float, ...],
        *,
        edge_side: Literal["below", "above"],
        dtype: Literal["float32", "float64"] = "float32",
    ) -> Any:
        return _host_values(self.at_energies(energies_kev, edge_side=edge_side), dtype=dtype)

    def to_device(
        self,
        energies_kev: tuple[float, ...],
        context: Any,
        *,
        edge_side: Literal["below", "above"],
        dtype: Literal["float32", "float64"] = "float32",
    ) -> Any:
        return _upload_values(
            self.to_host(energies_kev, edge_side=edge_side, dtype=dtype), context, dtype=dtype
        )


@dataclass(frozen=True, slots=True)
class MaterialBasis:
    """Fixed-density, non-negative dimensionless basis fields integrated in mm.

    ``mixing_assumption`` states whether fields are volume fractions, material
    indicators or relative concentrations. There is no implicit HU conversion.
    Spatially varying mass density must be included in the declared basis field.
    """

    tables: tuple[MaterialTable | MassFractionMixture, ...]
    mixing_assumption: str

    def __post_init__(self) -> None:
        if not isinstance(self.tables, tuple) or not self.tables:
            raise ContractError("a material basis requires an immutable tuple of tables")
        if any(
            not isinstance(table, (MaterialTable, MassFractionMixture)) for table in self.tables
        ):
            raise ContractError(
                "material basis entries must be supplied tables or declared mass mixtures"
            )
        if len({table.name for table in self.tables}) != len(self.tables):
            raise ContractError("material basis names must be unique")
        if not self.mixing_assumption.strip():
            raise ContractError("the material-basis mixing assumption must be recorded")

    def coefficients_at(
        self, energies_kev: tuple[float, ...], *, edge_side: Literal["below", "above"]
    ) -> tuple[float, ...]:
        """Return material-major (material, energy) coefficients for explicit upload."""
        return tuple(
            value
            for table in self.tables
            for value in table.at_energies(energies_kev, edge_side=edge_side)
        )

    def to_host(
        self,
        energies_kev: tuple[float, ...],
        *,
        edge_side: Literal["below", "above"],
        dtype: Literal["float32", "float64"] = "float32",
    ) -> Any:
        """Return a contiguous (M,K) array in declared coefficient precision."""
        values = self.coefficients_at(energies_kev, edge_side=edge_side)
        return _host_values(values, dtype=dtype).reshape((len(self.tables), len(energies_kev)))

    def to_device(
        self,
        energies_kev: tuple[float, ...],
        context: Any,
        *,
        edge_side: Literal["below", "above"],
        dtype: Literal["float32", "float64"] = "float32",
    ) -> Any:
        """Upload material-major (M,K) coefficients without inventing a mixture."""
        return _upload_values(
            self.to_host(energies_kev, edge_side=edge_side, dtype=dtype), context, dtype=dtype
        )

    def bin_averages(
        self,
        bin_edges_kev: tuple[float, ...],
        *,
        weighting: Literal["uniform-energy"],
        quadrature_order: int = 16,
    ) -> tuple[float, ...]:
        """Material-major bin averages under the same explicitly supplied rule."""
        return tuple(
            value
            for table in self.tables
            for value in table.bin_averages(
                bin_edges_kev, weighting=weighting, quadrature_order=quadrature_order
            )
        )
