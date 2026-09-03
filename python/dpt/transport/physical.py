"""Explicit atomic scattering and local energy accounting for transport.

The unpolarised independent-atom Rayleigh model uses Thomson times F(q)^2.
Momentum transfer is q = E sqrt((1-cos(theta))/2)/hc in inverse angstroms.
Atomic tables also retain the provider's incoherent scattering functions and
their provenance. The noncoherent remainder follows the declared transport
scattering law. Photoelectron, recoil and untransported vacancy energy is local.

References: Hubbell et al., J. Phys. Chem. Ref. Data 4, 471 (1975),
https://doi.org/10.1063/1.555523 and the xraylib 4.3.0 scattering implementation,
https://github.com/tschoonj/xraylib/blob/xraylib-4.3.0/src/scattering.c.
"""

from __future__ import annotations

# These runtime contracts validate untyped callers as well as annotated clients.
# Provider hashing and transport table preparation are shared package internals.
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false
# pyright: reportUnusedFunction=false
import hashlib
import importlib
import json
import math
import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from dpt.materials import Provenance
from dpt.physical_data import ProviderIdentity, _xraylib_identity

from .bound_compton import BoundComptonTable
from .model import TransportError, TransportSpec, _physical
from .source_spectrum import SourceSpectrum

HC_KEV_ANGSTROM = 12.398419843320026


@dataclass(frozen=True, slots=True)
class AtomicScatteringTable:
    """Represent elemental atomic data and its interpolation law.

    For incident energy ``E`` and scattering cosine ``c``, momentum transfer is
    ``q = E*sqrt((1-c)/2)/hc``, where ``hc = 12.398419843320026 keV angstrom``.
    The Rayleigh angular density per steradian is proportional to
    ``(1+c*c)*F(q)**2`` and is normalised over the complete solid angle.
    ``F`` and the retained incoherent function ``S`` are dimensionless.

    All three tuples have shape ``(Q,)`` with at least two nodes. Momentum
    transfer is in inverse angstroms, increases strictly from zero, and must
    reach the largest source energy divided by ``hc``. The numerical model
    interpolates each function linearly in ``q`` with no endpoint extension.
    Values satisfy ``0<=F<=Z``, ``0<=S<=Z``, ``F(0)=Z`` and ``S(0)=0``.
    This unpolarised independent-atom model has no anomalous, molecular or
    lattice correction. A compound requires explicitly supplied effective
    data rather than an inferred elemental composition.

    The record is immutable host data. Preparation uploads it to the owning
    workspace device, where the arrays remain fixed until the workspace is
    released. Density derivatives change interaction rates while holding these
    angular tables fixed, and no derivative of the node locations is inferred.
    ``provenance`` identifies the represented data and interpolation. An optional
    ``provider`` retains hashes of the actual installed provider implementation.

    Example with the installed pinned provider::

        from dpt.transport import xraylib_atomic_scattering
        table = xraylib_atomic_scattering(13, (0.0, 0.005, 0.01, 0.1, 1.0, 10.0))
        assert table.form_factor[0] == 13.0

    Atomic reference: https://doi.org/10.1063/1.555523
    """

    atomic_number: int
    momentum_transfer_inverse_angstrom: tuple[float, ...]
    form_factor: tuple[float, ...]
    scattering_function: tuple[float, ...]
    provenance: Provenance
    provider: ProviderIdentity | None = None

    def __post_init__(self) -> None:
        if type(self.atomic_number) is not int or not 1 <= self.atomic_number <= 100:
            raise TransportError("atomic number must be in 1..100")
        for name in ("momentum_transfer_inverse_angstrom", "form_factor", "scattering_function"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        q = self.momentum_transfer_inverse_angstrom
        if len(q) < 2 or q[0] != 0 or any(b <= a for a, b in pairwise(q)):
            raise TransportError("atomic scattering q nodes must increase strictly from zero")
        if len(self.form_factor) != len(q) or len(self.scattering_function) != len(q):
            raise TransportError("atomic scattering values must match q nodes")
        for value in q:
            _physical(value, "momentum transfer", nonnegative=True)
        for name, values in (
            ("form factor", self.form_factor),
            ("scattering function", self.scattering_function),
        ):
            for value in values:
                _physical(value, name, nonnegative=True)
                if value > self.atomic_number:
                    raise TransportError(f"{name} exceeds the atomic number")
        if self.form_factor[0] != self.atomic_number or self.scattering_function[0] != 0:
            raise TransportError("atomic scattering requires F(0)=Z and S(0)=0")
        if not isinstance(self.provenance, Provenance):
            raise TransportError("atomic scattering requires data provenance")
        if self.provider is not None and not isinstance(self.provider, ProviderIdentity):
            raise TransportError("atomic scattering provider identity is invalid")


@dataclass(frozen=True, slots=True)
class FluorescenceLine:
    """Define one emitted line conditional on a photoelectric event.

    At incident energy ``E``, branch ``l`` has probability ``p_l(E)`` and emits
    an isotropic photon of energy ``E_l``. The complementary probability is
    ``1-sum_l(p_l(E))``. A radiative branch deposits ``E-E_l`` locally and the
    complementary branch deposits ``E`` locally, so each event conserves energy.
    At most one emitted photon continues from an interaction. This declared
    single-vacancy model includes local photoelectron and residual vacancy energy
    and does not transport a relaxation cascade or secondary electrons.

    ``energy_kev`` and the parent binding energy ``threshold_kev`` are positive
    keV values with ``energy_kev<threshold_kev``. ``probabilities`` is a host
    tuple of shape ``(K,)`` matching the prepared coefficient energy nodes,
    dimensionless and linearly interpolated in incident energy. Probabilities
    are zero below the vacancy threshold and all lines of a material sum to at
    most one at each node. The emitted line must lie in coefficient support
    whenever it lies above the declared local-deposition cutoff.

    Preparation uploads the immutable values to the workspace device. Material
    and voxel density gradients retain these branch probabilities and line
    energies as fixed conditional laws. This record has no Warp tape derivative.
    ``provenance`` records the explicitly selected line and probability source.

    Example with two individual aluminium transitions::

        from dpt.transport import xraylib_fluorescence_lines
        lines = xraylib_fluorescence_lines(
            13, (1.0, 2.0, 10.0), lines=("KL2_LINE", "KL3_LINE")
        )
        assert all(line.probabilities[0] == 0.0 for line in lines)

    Provider model: https://github.com/tschoonj/xraylib/wiki/The-xraylib-API-list-of-all-functions
    """

    energy_kev: float
    threshold_kev: float
    probabilities: tuple[float, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "probabilities", tuple(self.probabilities))
        _physical(self.energy_kev, "fluorescence line energy", positive=True)
        _physical(self.threshold_kev, "fluorescence threshold", positive=True)
        if self.energy_kev >= self.threshold_kev:
            raise TransportError("fluorescence line energy must be below its vacancy threshold")
        for value in self.probabilities:
            _physical(value, "fluorescence branch probability", nonnegative=True)
            if value > 1:
                raise TransportError("fluorescence branch probability exceeds one")
        if not isinstance(self.provenance, Provenance):
            raise TransportError("fluorescence lines require data provenance")


@dataclass(frozen=True, slots=True)
class PhysicalTransport:
    """Bind physical interactions, a spectrum and fixed-majorant transport.

    In material ``m`` and cell ``v``, the total rate is
    ``mu = density[m]*voxel_density[v]*(absorption[m,E]+scattering[m,E])``.
    ``rayleigh_mm_inverse[m,E]`` is the coherent partial within aggregate
    scattering, so real events select photoelectric, Rayleigh and the remaining
    declared scattering law with probabilities proportional to their rates.
    The remaining law is exactly ``TransportSpec.scattering_law``. Rayleigh
    preserves photon energy and uses the table's normalised ``Thomson*F**2``
    angular law. Optional fluorescence follows ``FluorescenceLine``.

    With a fixed majorant ``M(E)>mu(v,E)``, candidates have rate ``M`` and accept
    a real event with probability ``mu/M``. For a log-density parameter, a real
    mark contributes ``+1`` and a null mark contributes ``-mu/(M-mu)`` to the
    likelihood score. The candidate flight has zero density score because
    ``M`` stays fixed. Strict slack is checked in every cell at every energy
    node and throughout each validated dynamic-density call. Linear energy
    interpolation preserves that bound. Null events consume the explicit event
    budget and exhaustion invalidates the estimate.

    For ``M`` materials and ``K`` energy nodes, ``scattering_tables`` has length
    ``M`` and ``rayleigh_mm_inverse`` has shape ``(M*K,)`` in material-major,
    energy-fastest order, in mm inverse. ``majorant_mm_inverse`` is either empty
    or shape ``(K,)`` in mm inverse. Fluorescence is empty or contains one tuple
    of lines per material. An optional ``SourceSpectrum`` samples the incident
    energy per original history. All records are immutable host data uploaded
    once to the workspace's device and held fixed during density differentiation.
    Provider and caller-supplied tables retain their explicit provenance.

    ``bound_compton`` supplies one complete ``BoundComptonTable`` per material
    for ``scattering_law="bound-electron-compton"``. Its element matches the
    atomic angular table. Occupied-shell profiles and binding energies determine
    the conditional Doppler-broadened energy law, with a deposition cutoff of
    at least 0.1 keV. Shell data stay fixed under material differentiation.

    ``cutoff_kev`` deposits the complete residual photon energy at its current
    cell before looking up coefficients below the cutoff. Photoelectric and
    Compton recoil energy is local. Fluorescence and Woodcock require analogue
    transport. Rayleigh and spectrum sampling also support continuous absorption
    weighting, while ``trace_energy_balance`` requires the analogue estimator.
    Conditional atomic laws, cutoffs and a majorant are not active tape inputs.
    Explicit replay interfaces handle density, coefficient and spectrum scores.

    Example of an aluminium Rayleigh and single-vacancy configuration::

        import xraylib
        from dpt.transport import PhysicalTransport, xraylib_atomic_scattering
        from dpt.transport import xraylib_fluorescence_lines
        energies = (1.0, 2.0, 10.0)
        coherent = tuple(xraylib.CS_Rayl(13, e)*2.6989/10 for e in energies)
        physical = PhysicalTransport(
            (xraylib_atomic_scattering(13, (0.0, 0.005, 0.1, 1.0)),),
            coherent,
            fluorescence=(xraylib_fluorescence_lines(
                13, energies, lines=("KL2_LINE", "KL3_LINE")
            ),),
            cutoff_kev=1.0,
        )

    Supply matching macroscopic absorption and aggregate scattering arrays to
    ``prepare_transport(..., physical=physical)``. The conversion above uses
    ``(cm**2/g)*(g/cm**3)/10 = mm**-1`` and does not infer a density scale.

    """

    scattering_tables: tuple[AtomicScatteringTable, ...]
    rayleigh_mm_inverse: tuple[float, ...]
    fluorescence: tuple[tuple[FluorescenceLine, ...], ...] = ()
    cutoff_kev: float = 0.0
    majorant_mm_inverse: tuple[float, ...] = ()
    spectrum: SourceSpectrum | None = None
    bound_compton: tuple[BoundComptonTable, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "scattering_tables",
            "rayleigh_mm_inverse",
            "majorant_mm_inverse",
            "bound_compton",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "fluorescence", tuple(tuple(lines) for lines in self.fluorescence))
        _physical(self.cutoff_kev, "energy cutoff", nonnegative=True)
        if not self.scattering_tables or any(
            not isinstance(t, AtomicScatteringTable) for t in self.scattering_tables
        ):
            raise TransportError(
                "physical transport requires one atomic scattering table per material"
            )
        for value in self.rayleigh_mm_inverse:
            _physical(value, "Rayleigh coefficient", nonnegative=True)
        for value in self.majorant_mm_inverse:
            _physical(value, "Woodcock majorant", positive=True)
        if self.spectrum is not None and not isinstance(self.spectrum, SourceSpectrum):
            raise TransportError("physical spectrum must be a SourceSpectrum record")
        if any(not isinstance(t, BoundComptonTable) for t in self.bound_compton):
            raise TransportError("bound Compton tables must be physical shell records")


def xraylib_atomic_scattering(
    atomic_number: int, momentum_transfer_inverse_angstrom: tuple[float, ...]
) -> AtomicScatteringTable:
    """Query pinned atomic functions with data and implementation provenance.

    The returned dimensionless functions satisfy ``F(0)=Z`` and ``S(0)=0``.
    The represented Rayleigh law is ``p(c,phi) proportional to (1+c*c)*F(q)**2``
    with ``q=E*sqrt((1-c)/2)/hc``. ``atomic_number`` is an integer in ``1..100``
    and the host q tuple has shape ``(Q,)`` in inverse angstroms, increasing from
    zero. Positive nodes must lie in the installed provider's supported range.

    This host-only operation requires the actual installed xraylib 4.3.0 and
    queries it immediately. Zero endpoints use atomic sum rules because the
    incoherent API excludes zero. It neither downloads a provider nor extends
    unsupported positive nodes. The returned ``AtomicScatteringTable`` keeps
    hashes of the canonical queried payload, Python module and native extension,
    together with the actual BSD licence URL. No value is an active gradient
    input and a device transfer happens only when preparing transport.

    Example::

        table = xraylib_atomic_scattering(13, (0.0, 0.005, 0.1, 1.0, 10.0))
        assert table.atomic_number == 13

    Provider: https://github.com/tschoonj/xraylib/blob/xraylib-4.3.0/src/scattering.c
    """
    if type(atomic_number) is not int or not 1 <= atomic_number <= 100:
        raise TransportError("atomic number must be in 1..100")
    q = tuple(momentum_transfer_inverse_angstrom)
    if len(q) < 2 or q[0] != 0 or any(b <= a for a, b in pairwise(q)):
        raise TransportError("atomic scattering q nodes must increase strictly from zero")
    for value in q:
        _physical(value, "momentum transfer", nonnegative=True)
    module: Any = importlib.import_module("xraylib")
    if module.__version__ != "4.3.0":
        raise TransportError("atomic scattering provider must be pinned xraylib 4.3.0")
    identity = _xraylib_identity(module)
    try:
        factors = (float(atomic_number), *(float(module.FF_Rayl(atomic_number, x)) for x in q[1:]))
        functions = (0.0, *(float(module.SF_Compt(atomic_number, x)) for x in q[1:]))
    except (ValueError, TypeError, OverflowError) as error:
        raise TransportError(f"xraylib atomic scattering query failed: {error}") from error
    payload = json.dumps(
        {
            "Z": atomic_number,
            "q_inverse_angstrom": q,
            "F": factors,
            "S": functions,
            "provider": identity.version,
            "artifacts": identity.artifacts,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    provenance = Provenance(
        "https://github.com/tschoonj/xraylib/tree/xraylib-4.3.0",
        hashlib.sha256(payload).hexdigest(),
        "BSD-3-Clause: https://github.com/tschoonj/xraylib/blob/xraylib-4.3.0/license_all.txt",
        "Canonical JSON of Z, q/F/S values and installed provider hashes. "
        "Exact q=0 atomic sum rules.",
    )
    return AtomicScatteringTable(atomic_number, q, factors, functions, provenance, identity)


def xraylib_fluorescence_lines(
    atomic_number: int, energies_kev: tuple[float, ...], *, lines: tuple[str, ...]
) -> tuple[FluorescenceLine, ...]:
    """Query individual transitions for the explicit single-vacancy model.

    For each requested line, the conditional probability is
    ``p_l(E)=CS_FluorLine(Z,l,E)/CS_Photo(Z,E)`` above its parent binding energy,
    and zero below it. xraylib's jump-factor approximation, fluorescence yield
    and radiative fraction determine this ratio. A selected branch emits its
    tabulated line energy isotropically and deposits ``E-E_l`` locally. All
    complementary outcomes deposit ``E`` locally without an emitted photon.

    ``energies_kev`` is a strictly increasing positive host tuple of shape
    ``(K,)`` matching the intended coefficient grid. ``lines`` is a nonempty
    tuple of unique individual transition macro names such as ``KL2_LINE``.
    Aggregated macros and implicit shell or cascade selection are rejected.
    The returned tuple contains immutable ``FluorescenceLine`` records whose
    incident and outgoing energies are in keV and probabilities dimensionless.

    This host-only query requires installed xraylib 4.3.0. Each line retains
    hashes of its canonical queried payload and actual provider implementation
    with the BSD licence URL. Preparation uploads the fixed records, and density
    derivatives do not differentiate their thresholds, energies or branching.

    Example::

        lines = xraylib_fluorescence_lines(
            13, (1.0, 2.0, 10.0), lines=("KL2_LINE", "KL3_LINE")
        )
        assert len(lines) == 2 and lines[0].energy_kev < lines[0].threshold_kev

    Provider: https://github.com/tschoonj/xraylib/wiki/The-xraylib-API-list-of-all-functions
    """
    if type(atomic_number) is not int or not 1 <= atomic_number <= 100:
        raise TransportError("atomic number must be in 1..100")
    energies = tuple(energies_kev)
    if not energies or any(b <= a for a, b in pairwise(energies)):
        raise TransportError("fluorescence energies must increase strictly")
    for value in energies:
        _physical(value, "fluorescence incident energy", positive=True)
    if not lines or len(set(lines)) != len(lines):
        raise TransportError("fluorescence needs unique explicit individual line names")
    module: Any = importlib.import_module("xraylib")
    if module.__version__ != "4.3.0":
        raise TransportError("fluorescence provider must be pinned xraylib 4.3.0")
    identity = _xraylib_identity(module)
    result: list[FluorescenceLine] = []
    for name in lines:
        match = re.fullmatch(
            r"(K|L[1-3]|M[1-5]|N[1-7]|O[1-7]|P[1-5])"
            r"(L[1-3]|M[1-5]|N[1-7]|O[1-7]|P[1-5]|Q[1-3])_LINE",
            name,
        )
        if match is None or not hasattr(module, name):
            raise TransportError("fluorescence requires an individual xraylib transition macro")
        shell = getattr(module, match[1] + "_SHELL")
        line = getattr(module, name)
        try:
            threshold = float(module.EdgeEnergy(atomic_number, shell))
            outgoing = float(module.LineEnergy(atomic_number, line))
            probabilities = tuple(
                0.0
                if energy < threshold
                else float(module.CS_FluorLine(atomic_number, line, energy))
                / float(module.CS_Photo(atomic_number, energy))
                for energy in energies
            )
        except (ValueError, TypeError, OverflowError, ZeroDivisionError) as error:
            raise TransportError(f"xraylib fluorescence query failed: {error}") from error
        payload = json.dumps(
            {
                "Z": atomic_number,
                "line": name,
                "threshold_kev": threshold,
                "energy_kev": outgoing,
                "incident_energies_kev": energies,
                "conditional_probabilities": probabilities,
                "provider": identity.version,
                "artifacts": identity.artifacts,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        provenance = Provenance(
            "https://github.com/tschoonj/xraylib/tree/xraylib-4.3.0",
            hashlib.sha256(payload).hexdigest(),
            "BSD-3-Clause: https://github.com/tschoonj/xraylib/blob/xraylib-4.3.0/license_all.txt",
            "Canonical JSON of specified line/edge data, CS_FluorLine/CS_Photo conditional "
            "probabilities and provider identity. Jump-factor single-vacancy model.",
        )
        result.append(FluorescenceLine(outgoing, threshold, probabilities, provenance))
    return tuple(result)


def _prepare_physical_parameters(
    physical: PhysicalTransport | None, spec: TransportSpec, context: Any
) -> tuple[Any, tuple[tuple[str, Any], ...]]:
    """Validate immutable host records and upload their bounded table arrays."""
    from dpt._runtime import load_kernels

    kernels = load_kernels("dpt.transport.physical_kernels")
    parameters = kernels.PhysicalParameters()
    if physical is None:
        if spec.scattering_law == "bound-electron-compton":
            raise TransportError("bound Compton transport requires explicit physical shell tables")
        return parameters, ()
    if not isinstance(physical, PhysicalTransport):
        raise TransportError("physical must be a PhysicalTransport record")
    if spec.estimator != "analogue" and (
        any(physical.fluorescence) or physical.majorant_mm_inverse
    ):
        raise TransportError("physical fluorescence and Woodcock require analogue transport")
    if spec.max_angle_trials > 65536:
        raise TransportError("physical angular rejection reserves at most 65536 trial domains")
    materials, bins = spec.grid.materials, len(spec.energy_nodes)
    if len(physical.scattering_tables) != materials:
        raise TransportError("physical transport needs one scattering table per material")
    if len(physical.rayleigh_mm_inverse) != materials * bins:
        raise TransportError("Rayleigh table must match the material/energy coefficient shape")
    if physical.majorant_mm_inverse and len(physical.majorant_mm_inverse) != bins:
        raise TransportError("Woodcock majorant must match the coefficient energy nodes")
    if physical.cutoff_kev > spec.energy_nodes[-1]:
        raise TransportError("cutoff exceeds the supplied energy support")
    spectrum = physical.spectrum.energy_kev if physical.spectrum else ()
    if spectrum and (spectrum[0] < spec.energy_nodes[0] or spectrum[-1] > spec.energy_nodes[-1]):
        raise TransportError("source spectrum leaves coefficient energy support")
    maximum_energy = max((spec.energy_kev, *spectrum))
    if spec.scattering_law == "bound-electron-compton":
        if len(physical.bound_compton) != materials:
            raise TransportError("bound Compton needs one shell table per material")
        if spec.compton_sampler != "legacy-rejection":
            raise TransportError("bound Compton uses its own versioned impulse sampler")
        if min((spec.energy_kev, *spectrum)) < 0.1 or maximum_energy > 5000:
            raise TransportError("bound Compton incident energy domain is 0.1..5000 keV")
        if physical.cutoff_kev < 0.1:
            raise TransportError(
                "bound Compton requires a local-deposition cutoff of at least 0.1 keV"
            )
    elif physical.bound_compton:
        raise TransportError("shell tables require the bound-electron-compton scattering law")
    if spectrum and spec.energy_support_policy == "cover-budget":
        minimum_energy = spectrum[0]
        if spec.scattering_law == "free-electron-compton":
            minimum_energy = 1 / (1 / minimum_energy + 2 * spec.max_events / 510.99895069)
        if spec.energy_nodes[0] > minimum_energy and physical.cutoff_kev < spec.energy_nodes[0]:
            raise TransportError(
                "coefficient support does not cover the polyenergetic collision budget"
            )
    q_offsets: list[int] = [0]
    q: list[float] = []
    factors: list[float] = []
    functions: list[float] = []
    for table in physical.scattering_tables:
        if table.momentum_transfer_inverse_angstrom[-1] < maximum_energy / HC_KEV_ANGSTROM:
            raise TransportError("atomic scattering q support does not cover all source angles")
        q.extend(table.momentum_transfer_inverse_angstrom)
        factors.extend(table.form_factor)
        functions.extend(table.scattering_function)
        q_offsets.append(len(q))
    branches = physical.fluorescence or tuple(() for _ in range(materials))
    if len(branches) != materials:
        raise TransportError("fluorescence branches must match the material count")
    line_offsets: list[int] = [0]
    line_energy: list[float] = []
    line_threshold: list[float] = []
    line_probability: list[float] = []
    for lines in branches:
        for line in lines:
            if not isinstance(line, FluorescenceLine) or len(line.probabilities) != bins:
                raise TransportError(
                    "fluorescence probabilities must match coefficient energy nodes"
                )
            if line.energy_kev > physical.cutoff_kev and line.energy_kev < spec.energy_nodes[0]:
                raise TransportError("transported fluorescence line leaves coefficient support")
            for energy, probability in zip(spec.energy_nodes, line.probabilities, strict=True):
                if energy < line.threshold_kev and probability != 0:
                    raise TransportError(
                        "fluorescence probability is nonzero below its vacancy threshold"
                    )
            line_energy.append(line.energy_kev)
            line_threshold.append(line.threshold_kev)
            line_probability.extend(line.probabilities)
        for index in range(bins):
            if math.fsum(line.probabilities[index] for line in lines) > 1.0:
                raise TransportError("conditional fluorescence branch probabilities exceed one")
        line_offsets.append(len(line_energy))
    if max(len(q), len(line_probability), len(physical.rayleigh_mm_inverse)) >= 2**31:
        raise TransportError("physical table exceeds signed 32-bit indexing")
    wp = context.wp
    cdf = physical.spectrum.cdf if physical.spectrum else ()
    if cdf:
        if any(b <= a for a, b in pairwise((0.0, *cdf))):
            raise TransportError("source spectrum has a probability lost to binary64 CDF rounding")
    values: dict[str, Any] = {
        "rayleigh": physical.rayleigh_mm_inverse,
        "majorant": physical.majorant_mm_inverse,
        "q_offsets": q_offsets,
        "q": q,
        "form_factor": factors,
        "scattering_function": functions,
        "atomic_number": [float(t.atomic_number) for t in physical.scattering_tables],
        "fluorescence_offsets": line_offsets,
        "fluorescence_energy": line_energy,
        "fluorescence_threshold": line_threshold,
        "fluorescence_probability": line_probability,
        "spectrum_energies": spectrum,
        "spectrum_cdf": cdf,
    }
    shell_offsets: list[int] = [0]
    occupancy: list[float] = []
    binding: list[float] = []
    height: list[float] = []
    for material, table in enumerate(physical.bound_compton):
        if table.atomic_number != physical.scattering_tables[material].atomic_number:
            raise TransportError("bound Compton and atomic scattering elements must match")
        for shell in table.shells:
            occupancy.append(shell.occupancy)
            binding.append(shell.binding_kev)
            height.append(shell.momentum_height)
        shell_offsets.append(len(occupancy))
    values.update(
        {
            "compton_shell_offsets": shell_offsets,
            "compton_occupancy": occupancy,
            "compton_binding": binding,
            "compton_height": height,
        }
    )
    buffers: list[tuple[str, Any]] = []
    with context.scope():
        for name, entries in values.items():
            array = wp.array(
                entries,
                dtype=wp.int32 if name.endswith("offsets") else wp.float64,
                device=context.device,
            )
            setattr(parameters, name, array)
            buffers.append(("physical_" + name, array))
    parameters.enabled = 1
    parameters.cutoff = physical.cutoff_kev
    parameters.woodcock = int(bool(physical.majorant_mm_inverse))
    parameters.spectrum_bins = len(spectrum)
    return parameters, tuple(buffers)
