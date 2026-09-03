"""Charge-cloud pulse readout and a rectangular-gate photon-counting model.

Calibration records are immutable and importing this module loads no numerical
runtime. Preparation explicitly selects Warp CPU or CUDA and all large arrays
remain caller-owned. Energies are keV, event times are seconds and rates are s⁻¹.
"""

# Calibration buffers are owned by these prepared operators.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, cast

from dpt._runtime import (
    DeviceContext,
    StatusReadback,
    load_kernels,
    prepare_context,
    require_no_tape,
)
from dpt.contracts import ContractError, finite_scalar, finite_tuple, integer, raise_device_status
from dpt.materials import Provenance, validate_energy_grid


def _probabilities(values: tuple[float, ...], size: int, name: str) -> None:
    finite_tuple(values, name, length=size)
    if math.fsum(values) != 1.0:
        raise ContractError(f"{name} must sum to one in binary64")


def _edges(values: tuple[float, ...]) -> None:
    finite_tuple(values, "bin edges in keV")
    if len(values) < 2 or any(a >= b for a, b in pairwise(values)):
        raise ContractError("bin edges must be strictly increasing with at least two entries")


@dataclass(frozen=True, slots=True)
class ChargeCloudPattern:
    """One charge cloud with distinct (row,column) pixel offsets and energy fractions.

    The fractions sum to one and every positive component produces one coincident
    pulse. A component outside the finite detector escapes and is never folded
    into an edge pixel. Patterns describe deposited charge supplied by calibration,
    rather than fluorescence transport or a material-specific diffusion model.
    """

    offsets: tuple[tuple[int, int], ...]
    fractions: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.offsets), tuple) or not self.offsets:
            raise ContractError("offsets must be a non-empty immutable tuple")
        for offset in self.offsets:
            if not isinstance(cast(object, offset), tuple) or len(offset) != 2:
                raise ContractError("each charge-cloud offset must be (row,column)")
            for value in offset:
                integer(value, "charge-cloud offset", minimum=-(2**30), maximum=2**30)
        if len(set(self.offsets)) != len(self.offsets):
            raise ContractError("charge-cloud offsets must be distinct")
        _probabilities(self.fractions, len(self.offsets), "charge-cloud fractions")


@dataclass(frozen=True, slots=True)
class ChargeSharingSpec:
    """Finite detector, incident energies, cloud outcomes and energy-bin boundaries.

    ``pattern_probabilities`` is energy-major [E,M] flattened into a tuple. Each
    row sums to one and each event draws exactly one complete cloud, preserving
    correlations between neighbouring pulses. Bins are [edge[b],edge[b+1]), and
    pulses below the first edge or at/above the final edge have separate ledgers.
    Zero-energy components never produce a count, even when the first edge is zero.
    Supplied provenance identifies the actual calibration or analytic test law.
    """

    detector_shape: tuple[int, int]
    energies_kev: tuple[float, ...]
    patterns: tuple[ChargeCloudPattern, ...]
    pattern_probabilities: tuple[float, ...]
    bin_edges_kev: tuple[float, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(object, self.detector_shape), tuple)
            or len(self.detector_shape) != 2
        ):
            raise ContractError("detector_shape must be (rows,columns)")
        for value in self.detector_shape:
            integer(value, "detector dimension", minimum=1)
        integer(self.pixels, "detector pixels", minimum=1)
        validate_energy_grid(self.energies_kev)
        if not isinstance(cast(object, self.patterns), tuple) or not self.patterns:
            raise ContractError("patterns must be a non-empty immutable tuple")
        if any(
            not isinstance(cast(object, pattern), ChargeCloudPattern) for pattern in self.patterns
        ):
            raise ContractError("patterns must contain ChargeCloudPattern records")
        for pattern in self.patterns:
            for offset, fraction in zip(pattern.offsets, pattern.fractions, strict=True):
                if any(
                    dimension - 1 + abs(delta) > 2**31 - 1
                    for dimension, delta in zip(self.detector_shape, offset, strict=True)
                ):
                    raise ContractError("charge-cloud offsets exceed the int32 coordinate range")
                if fraction > 0 and any(energy * fraction == 0 for energy in self.energies_kev):
                    raise ContractError("positive fragment energy underflows binary64")
        count = len(self.patterns)
        finite_tuple(
            self.pattern_probabilities, "pattern probabilities", length=count * self.energies
        )
        for energy in range(self.energies):
            _probabilities(
                self.pattern_probabilities[energy * count : (energy + 1) * count],
                count,
                "pattern probabilities for each energy",
            )
        _edges(self.bin_edges_kev)
        if not isinstance(cast(object, self.provenance), Provenance):
            raise ContractError("charge sharing requires supplied calibration provenance")

    @property
    def pixels(self) -> int:
        return self.detector_shape[0] * self.detector_shape[1]

    @property
    def energies(self) -> int:
        return len(self.energies_kev)

    @property
    def bins(self) -> int:
        return len(self.bin_edges_kev) - 1

    @property
    def fragments(self) -> int:
        return max(len(pattern.fractions) for pattern in self.patterns)


@dataclass(frozen=True, slots=True)
class ChargeSharingIdentity:
    """Philox key and immutable event identities for a charge-cloud experiment.

    Counter = ((observation_id<<32)|event_index, 0xffffffff, 1). The final
    coordinates are the Philox event and domain. Existing Poisson proposals use
    event coordinates below 2³¹, electronic noise uses event zero and transport
    uses domain zero or domains at least 0x80000000, so the complete addresses
    are disjoint. Reusing an identity deliberately reuses the random experiment.
    ``event_offset`` preserves identities when an event list is split into chunks.
    """

    seed: int
    observation_id: int
    event_offset: int = 0

    def __post_init__(self) -> None:
        integer(self.seed, "seed", maximum=2**64 - 1)
        integer(self.observation_id, "observation_id", maximum=2**32 - 1)
        integer(self.event_offset, "event_offset", maximum=2**32 - 1)

    def validate_extent(self, events: int) -> None:
        if self.event_offset + events > 2**32:
            raise ContractError("charge-sharing event identities would wrap")


@dataclass(frozen=True, slots=True)
class ChargeSharingWorkspace:
    spec: ChargeSharingSpec
    max_events: int
    context: DeviceContext
    _energies: Any = field(repr=False)
    _probabilities: Any = field(repr=False)
    _fractions: Any = field(repr=False)
    _rows: Any = field(repr=False)
    _columns: Any = field(repr=False)
    _edges: Any = field(repr=False)
    _kernels: Any = field(repr=False)
    _status: Any = field(repr=False)
    _readback: StatusReadback = field(repr=False)

    @property
    def scratch_bytes(self) -> int:
        spec = self.spec
        return (
            8 * (spec.energies + len(spec.pattern_probabilities) + len(spec.bin_edges_kev))
            + 16 * len(spec.patterns) * spec.fragments
            + 4
        )

    @property
    def host_bytes(self) -> int:
        return self._readback.host_bytes

    def check_status(self) -> None:
        raise_device_status(self._readback.read(), "photon-counting charge sharing")

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()


def prepare_charge_sharing(
    spec: ChargeSharingSpec,
    *,
    max_events: int,
    device: str = "cuda:0",
    stream: Any = None,
    max_owned_bytes: int = 1 << 24,
) -> ChargeSharingWorkspace:
    """Upload small cloud calibration tables once on explicitly selected CPU/CUDA."""
    require_no_tape()
    if not isinstance(cast(object, spec), ChargeSharingSpec):
        raise ContractError("spec must be ChargeSharingSpec")
    integer(max_events, "max_events", minimum=1)
    integer(max_owned_bytes, "max_owned_bytes", minimum=1, maximum=2**63 - 1)
    owned = (
        8 * (spec.energies + len(spec.pattern_probabilities) + len(spec.bin_edges_kev))
        + 16 * len(spec.patterns) * spec.fragments
        + 8
    )
    if owned > max_owned_bytes:
        raise ContractError("charge-cloud calibration exceeds max_owned_bytes")
    ctx = prepare_context(device=device, stream=stream)
    kernels = load_kernels("dpt.kernels.photon_counting")
    fractions: list[float] = []
    rows: list[int] = []
    columns: list[int] = []
    for pattern in spec.patterns:
        for fragment in range(spec.fragments):
            offset = pattern.offsets[fragment] if fragment < len(pattern.offsets) else (0, 0)
            rows.append(offset[0])
            columns.append(offset[1])
            fractions.append(
                pattern.fractions[fragment] if fragment < len(pattern.fractions) else 0.0
            )
    with ctx.scope():
        energies = ctx.wp.array(spec.energies_kev, dtype=ctx.wp.float64, device=ctx.device)
        probabilities = ctx.wp.array(
            spec.pattern_probabilities, dtype=ctx.wp.float64, device=ctx.device
        )
        cloud = ctx.wp.array(fractions, dtype=ctx.wp.float64, device=ctx.device)
        dr = ctx.wp.array(rows, dtype=ctx.wp.int32, device=ctx.device)
        dc = ctx.wp.array(columns, dtype=ctx.wp.int32, device=ctx.device)
        edges = ctx.wp.array(spec.bin_edges_kev, dtype=ctx.wp.float64, device=ctx.device)
        status = ctx.wp.zeros(1, dtype=ctx.wp.int32, device=ctx.device)
    ctx.wp.synchronize_stream(ctx.stream)
    return ChargeSharingWorkspace(
        spec,
        max_events,
        ctx,
        energies,
        probabilities,
        cloud,
        dr,
        dc,
        edges,
        kernels,
        status,
        StatusReadback(ctx, status),
    )


def _charge_args(workspace: ChargeSharingWorkspace) -> list[Any]:
    spec = workspace.spec
    return [
        workspace._energies,
        workspace._probabilities,
        workspace._fractions,
        workspace._rows,
        workspace._columns,
        workspace._edges,
        spec.energies,
        len(spec.patterns),
        spec.fragments,
        spec.bins,
        *spec.detector_shape,
    ]


def sample_charge_sharing(
    event_pixels: Any,
    event_energy_bins: Any,
    event_times_s: Any,
    *,
    out_pixels: Any,
    out_times_s: Any,
    out_energies_kev: Any,
    out_bins: Any,
    out_patterns: Any,
    out_accounting_kev: Any,
    identity: ChargeSharingIdentity,
    workspace: ChargeSharingWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """For outcome m drawn with probability pi[e,m], emit pulses E[e]*fraction[m,f].

    All fragments of an incident event retain its arrival time. Their energies sum
    to the incident energy, including fragments escaping the finite detector.
    Each in-detector fragment produces a count when its energy lies in a supplied
    left-closed/right-open bin. One incident photon can therefore produce several
    coincident counts in neighbouring pixels.

    N is the incident event count and F is the largest supplied cloud. Pixel
    indices use row-major detector numbering, while incident-energy indices
    select entries of spec.energies_kev. Inputs are contiguous int32 pixel and
    incident-energy indices [N] and binary64 event times [N] in seconds. Fragment
    outputs have shape [N,F], with
    int32 pixels/bin codes and binary64 times/energies in seconds/keV. Chosen
    patterns are int32 [N]. The binary64 ledger [N,4] contains accepted,
    below-threshold, overflow and escaped energy in keV. Bin codes -1, -2 and -3
    mean below threshold, overflow and escaped/zero component. Escaped fragments
    retain their energy and have destination pixel -1.

    The caller owns all outputs. They must be disjoint from the inputs and one
    another, contiguous and on the workspace's device and stream. Every call
    overwrites them and uses at most max_events incident events. The immutable
    identity assigns distinct event addresses to an observation and supports
    replay. Sampling is outside a Warp tape and has no pathwise derivative.

    Example::

        import hashlib
        import warp as wp
        from dpt.materials import Provenance
        from dpt.photon_counting import (
            ChargeCloudPattern, ChargeSharingSpec, prepare_charge_sharing,
        )

        law = b"analytic equal split into the incident pixel and its right neighbour"
        provenance = Provenance("analytic charge-cloud example", hashlib.sha256(law).hexdigest(),
                                "Apache-2.0", law.decode())
        spec = ChargeSharingSpec(
            (1, 2), (20.0,), (ChargeCloudPattern(((0, 0), (0, 1)), (0.5, 0.5)),),
            (1.0,), (5.0, 15.0), provenance,
        )
        workspace = prepare_charge_sharing(spec, max_events=2, device="cpu")
        from dpt.photon_counting import ChargeSharingIdentity, sample_charge_sharing

        pixels = wp.array([0, 1], dtype=wp.int32, device="cpu")
        energy_indices = wp.array([0, 0], dtype=wp.int32, device="cpu")
        times = wp.array([0.0, 0.1], dtype=wp.float64, device="cpu")
        out_pixels = wp.empty((2, 2), dtype=wp.int32, device="cpu")
        out_times = wp.empty((2, 2), dtype=wp.float64, device="cpu")
        out_energies = wp.empty((2, 2), dtype=wp.float64, device="cpu")
        out_bins = wp.empty((2, 2), dtype=wp.int32, device="cpu")
        out_patterns = wp.empty(2, dtype=wp.int32, device="cpu")
        ledger = wp.empty((2, 4), dtype=wp.float64, device="cpu")
        sample_charge_sharing(
            pixels, energy_indices, times, out_pixels=out_pixels, out_times_s=out_times,
            out_energies_kev=out_energies, out_bins=out_bins, out_patterns=out_patterns,
            out_accounting_kev=ledger, identity=ChargeSharingIdentity(17, 3), workspace=workspace,
        )
        assert out_pixels.numpy().tolist() == [[0, 1], [1, -1]]
        assert out_times.numpy().tolist() == [[0.0, 0.0], [0.1, 0.1]]
        assert out_energies.numpy().tolist() == [[10.0, 10.0], [10.0, 10.0]]
        assert ledger.numpy().tolist() == [[20.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 10.0]]

    """
    require_no_tape()
    ctx = workspace.context
    ctx.assert_stream(stream)
    if type(validate) is not bool or not isinstance(cast(object, identity), ChargeSharingIdentity):
        raise ContractError("validate must be boolean and identity must be ChargeSharingIdentity")
    count = ctx.array(event_pixels, "event_pixels", dtype=ctx.wp.int32)
    if not 1 <= count <= workspace.max_events:
        raise ContractError("event list must fit the prepared non-empty capacity")
    identity.validate_extent(count)
    reads = [
        ("event_pixels", event_pixels),
        ("event_energy_bins", event_energy_bins),
        ("event_times_s", event_times_s),
    ]
    ctx.array(event_energy_bins, "event_energy_bins", dtype=ctx.wp.int32, shape=(count,))
    ctx.array(event_times_s, "event_times_s", dtype=ctx.wp.float64, shape=(count,))
    writes = [
        ("out_pixels", out_pixels),
        ("out_times_s", out_times_s),
        ("out_energies_kev", out_energies_kev),
        ("out_bins", out_bins),
        ("out_patterns", out_patterns),
        ("out_accounting_kev", out_accounting_kev),
    ]
    for name, array, dtype in (
        ("out_pixels", out_pixels, ctx.wp.int32),
        ("out_times_s", out_times_s, ctx.wp.float64),
        ("out_energies_kev", out_energies_kev, ctx.wp.float64),
        ("out_bins", out_bins, ctx.wp.int32),
    ):
        ctx.array(array, name, dtype=dtype, shape=(count, workspace.spec.fragments), ndim=2)
    ctx.array(out_patterns, "out_patterns", dtype=ctx.wp.int32, shape=(count,))
    ctx.array(
        out_accounting_kev, "out_accounting_kev", dtype=ctx.wp.float64, shape=(count, 4), ndim=2
    )
    ctx.disjoint(reads, writes)
    if validate:
        workspace.check_status()
    with ctx.scope():
        ctx.wp.launch(
            workspace._kernels.charge_events,
            dim=count,
            inputs=[
                event_pixels,
                event_energy_bins,
                event_times_s,
                *_charge_args(workspace),
                ctx.wp.uint64(identity.seed),
                ctx.wp.uint64((identity.observation_id << 32) | identity.event_offset),
            ],
            outputs=[
                out_pixels,
                out_times_s,
                out_energies_kev,
                out_bins,
                out_patterns,
                out_accounting_kev,
                workspace._status,
            ],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
    if validate:
        workspace.check_status()


def charge_sharing_mean(
    incident_photons: Any,
    *,
    out_counts: Any,
    out_accounting_kev: Any,
    workspace: ChargeSharingWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """E[y[b,q]] = sum[e,p,m,f] Phi[e,p]*pi[e,m]*I(q=p+offset[m,f])*I_bin(E[e]*fraction[m,f]).

    The fixed cloud probabilities and comparator bins define this linear map from
    incident photon populations to expected pulse counts. A shared event can
    contribute to several destination pixels. The energy ledger is indexed by
    incident pixel and sums to sum[e] Phi[e,p]*E[e], including threshold loss,
    overflow and escape at the detector edge.

    E is the number of incident energies, P the number of detector pixels and B
    the number of comparator bins. The contiguous binary64 input has shape
    [E,P] and units of photons. Expected counts have shape [B,P]. The binary64
    energy ledger [4,P], in keV, contains
    accepted, below-threshold, overflow and escaped energy. Inputs must be finite
    and nonnegative. Fixed calibration is held by the prepared workspace, while
    the caller owns the output arrays. Inputs and outputs must be disjoint and on
    the workspace's device and stream. A call overwrites both outputs outside a
    Warp tape. The sampled operator retains the event correlations whose mean is
    computed here.

    Example::

        import hashlib
        import warp as wp
        from dpt.materials import Provenance
        from dpt.photon_counting import (
            ChargeCloudPattern, ChargeSharingSpec, prepare_charge_sharing,
        )

        law = b"analytic equal split into the incident pixel and its right neighbour"
        provenance = Provenance("analytic charge-cloud example", hashlib.sha256(law).hexdigest(),
                                "Apache-2.0", law.decode())
        spec = ChargeSharingSpec(
            (1, 2), (20.0,), (ChargeCloudPattern(((0, 0), (0, 1)), (0.5, 0.5)),),
            (1.0,), (5.0, 15.0), provenance,
        )
        workspace = prepare_charge_sharing(spec, max_events=2, device="cpu")
        from dpt.photon_counting import charge_sharing_mean

        incident = wp.array([[3.0, 2.0]], dtype=wp.float64, device="cpu")
        counts = wp.empty((1, 2), dtype=wp.float64, device="cpu")
        ledger = wp.empty((4, 2), dtype=wp.float64, device="cpu")
        charge_sharing_mean(incident, out_counts=counts, out_accounting_kev=ledger,
                            workspace=workspace)
        assert counts.numpy().tolist() == [[3.0, 5.0]]
        assert ledger.numpy().tolist() == [[60.0, 20.0], [0.0, 0.0], [0.0, 0.0], [0.0, 20.0]]
        assert float(ledger.numpy().sum()) == 20.0 * float(incident.numpy().sum())

    """
    _charge_mean_call(incident_photons, out_counts, out_accounting_kev, workspace, stream, validate)


def _charge_mean_call(
    incident: Any,
    counts: Any,
    ledger: Any,
    workspace: ChargeSharingWorkspace,
    stream: Any,
    validate: bool,
) -> None:
    require_no_tape()
    if type(validate) is not bool:
        raise ContractError("validate must be a boolean")
    ctx, spec = workspace.context, workspace.spec
    ctx.assert_stream(stream)
    ctx.array(
        incident,
        "incident_photons",
        dtype=ctx.wp.float64,
        shape=(spec.energies, spec.pixels),
        ndim=2,
    )
    ctx.array(counts, "out_counts", dtype=ctx.wp.float64, shape=(spec.bins, spec.pixels), ndim=2)
    ctx.array(ledger, "out_accounting_kev", dtype=ctx.wp.float64, shape=(4, spec.pixels), ndim=2)
    ctx.disjoint(
        [("incident_photons", incident)], [("out_counts", counts), ("out_accounting_kev", ledger)]
    )
    if validate:
        workspace.check_status()
    with ctx.scope():
        ctx.wp.launch(
            workspace._kernels.charge_mean,
            dim=spec.pixels,
            inputs=[incident, *_charge_args(workspace)],
            outputs=[counts, ledger, workspace._status],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
    if validate:
        workspace.check_status()


def charge_sharing_vjp(
    count_cotangent: Any,
    accounting_cotangent: Any,
    *,
    out_incident_photons: Any,
    workspace: ChargeSharingWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """For the fixed mean map (y,L)=A*Phi, write A.T*(count_cotangent, accounting_cotangent).

    This is the exact transpose with respect to incident photon populations.
    Cloud probabilities, energy fractions and bin thresholds remain fixed. It
    includes the accepted, below-threshold, overflow and escaped energy ledger.
    Realised event draws and discontinuous comparator thresholds are not active
    parameters.

    E is the number of incident energies, P the number of detector pixels and B
    the number of comparator bins. Count and ledger cotangents are finite
    signed contiguous binary64 arrays [B,P] and [4,P]. The binary64 output [E,P]
    gives loss per incident photon when
    the seeds carry loss per count and loss per keV. The caller owns the output,
    which is overwritten and must be disjoint from both seeds. All arrays belong
    to the workspace's device and stream. The transpose needs no preceding forward
    call and executes outside a Warp tape.

    Example::

        import hashlib
        import warp as wp
        from dpt.materials import Provenance
        from dpt.photon_counting import (
            ChargeCloudPattern, ChargeSharingSpec, prepare_charge_sharing,
        )

        law = b"analytic equal split into the incident pixel and its right neighbour"
        provenance = Provenance("analytic charge-cloud example", hashlib.sha256(law).hexdigest(),
                                "Apache-2.0", law.decode())
        spec = ChargeSharingSpec(
            (1, 2), (20.0,), (ChargeCloudPattern(((0, 0), (0, 1)), (0.5, 0.5)),),
            (1.0,), (5.0, 15.0), provenance,
        )
        workspace = prepare_charge_sharing(spec, max_events=2, device="cpu")
        from dpt.photon_counting import charge_sharing_vjp

        count_seed = wp.array([[2.0, 5.0]], dtype=wp.float64, device="cpu")
        # Unit energy seeds include both accepted energy and edge escape.
        energy_seed = wp.array([[1.0, 1.0]] * 4, dtype=wp.float64, device="cpu")
        incident_gradient = wp.empty((1, 2), dtype=wp.float64, device="cpu")
        charge_sharing_vjp(count_seed, energy_seed, out_incident_photons=incident_gradient,
                           workspace=workspace)
        assert incident_gradient.numpy().tolist() == [[27.0, 25.0]]

    """
    require_no_tape()
    if type(validate) is not bool:
        raise ContractError("validate must be a boolean")
    ctx, spec = workspace.context, workspace.spec
    ctx.assert_stream(stream)
    ctx.array(
        count_cotangent,
        "count_cotangent",
        dtype=ctx.wp.float64,
        shape=(spec.bins, spec.pixels),
        ndim=2,
    )
    ctx.array(
        accounting_cotangent,
        "accounting_cotangent",
        dtype=ctx.wp.float64,
        shape=(4, spec.pixels),
        ndim=2,
    )
    ctx.array(
        out_incident_photons,
        "out_incident_photons",
        dtype=ctx.wp.float64,
        shape=(spec.energies, spec.pixels),
        ndim=2,
    )
    ctx.disjoint(
        [("count_cotangent", count_cotangent), ("accounting_cotangent", accounting_cotangent)],
        [("out_incident_photons", out_incident_photons)],
    )
    if validate:
        workspace.check_status()
    with ctx.scope():
        ctx.wp.launch(
            workspace._kernels.charge_vjp,
            dim=(spec.energies, spec.pixels),
            inputs=[count_cotangent, accounting_cotangent, *_charge_args(workspace)],
            outputs=[out_incident_photons, workspace._status],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
    if validate:
        workspace.check_status()


@dataclass(frozen=True, slots=True)
class RectangularPileupSpec:
    """Exact positive energy-lattice marks and a nonparalysable rectangular gate.

    Incident photons form a stationary Poisson process with rate r. A photon
    arriving while live opens a gate Δ, and all photons in [start,start+Δ) sum
    their full energies into one pulse. A subsequent blind interval τ discards
    arrivals without extending either interval. The next live arrival opens a
    new gate. This is a declared rectangular integrator with no bipolar tail.

    Independent energy marks are ``energy_quantum_kev * mark_steps`` with supplied
    probabilities. ``bin_edges_steps`` are strictly increasing lattice boundaries
    and bins are left-closed/right-open. The finite spectral readout represents
    all probability and energy above its top explicitly as overflow. This is
    the nonparalysable counting mechanism discussed by Taguchi et al. (2010),
    https://doi.org/10.1118/1.3429056, with a separately declared rectangular
    summation gate, rather than their bipolar pulse-shape approximation.
    """

    energy_quantum_kev: float
    mark_steps: tuple[int, ...]
    mark_probabilities: tuple[float, ...]
    bin_edges_steps: tuple[int, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        finite_scalar(self.energy_quantum_kev, "energy quantum in keV", minimum=0)
        if self.energy_quantum_kev <= 0:
            raise ContractError("energy quantum must be positive")
        for name, values, minimum in (
            ("mark_steps", self.mark_steps, 1),
            ("bin_edges_steps", self.bin_edges_steps, 0),
        ):
            if not isinstance(cast(object, values), tuple) or not values:
                raise ContractError(f"{name} must be a non-empty immutable tuple")
            for value in values:
                integer(value, name, minimum=minimum)
            if any(a >= b for a, b in pairwise(values)):
                raise ContractError(f"{name} must be strictly increasing")
        if len(self.bin_edges_steps) < 2:
            raise ContractError("at least two energy-bin edges are required")
        _probabilities(self.mark_probabilities, len(self.mark_steps), "mark probabilities")
        if not isinstance(cast(object, self.provenance), Provenance):
            raise ContractError("pileup requires supplied energy-mark provenance")
        if not math.isfinite(
            self.energy_quantum_kev * max(self.mark_steps[-1], self.bin_edges_steps[-1])
        ) or not math.isfinite(self.mean_mark_kev):
            raise ContractError("energy lattice cannot be represented in binary64")

    @property
    def bins(self) -> int:
        return len(self.bin_edges_steps) - 1

    @property
    def mean_mark_kev(self) -> float:
        return self.energy_quantum_kev * math.fsum(
            step * probability
            for step, probability in zip(self.mark_steps, self.mark_probabilities, strict=True)
        )


@dataclass(frozen=True, slots=True)
class RectangularPileupWorkspace:
    spec: RectangularPileupSpec
    max_pixels: int
    max_events_per_pixel: int
    context: DeviceContext
    _marks: Any = field(repr=False)
    _probabilities: Any = field(repr=False)
    _edges: Any = field(repr=False)
    _compound: Any = field(repr=False)
    _kernels: Any = field(repr=False)
    _status: Any = field(repr=False)
    _readback: StatusReadback = field(repr=False)

    @property
    def scratch_bytes(self) -> int:
        return (
            12 * len(self.spec.mark_steps)
            + 4 * len(self.spec.bin_edges_steps)
            + 8 * self.spec.bin_edges_steps[-1] * self.max_pixels
            + 4
        )

    @property
    def host_bytes(self) -> int:
        return self._readback.host_bytes

    def check_status(self) -> None:
        raise_device_status(self._readback.read(), "rectangular photon-counting pileup")

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()


def prepare_rectangular_pileup(
    spec: RectangularPileupSpec,
    *,
    max_pixels: int,
    max_events_per_pixel: int = 1,
    device: str = "cuda:0",
    stream: Any = None,
    max_owned_bytes: int = 1 << 28,
) -> RectangularPileupWorkspace:
    """Prepare O(top energy step x pixel capacity) Poisson-recursion scratch.

    The scratch resolves only the finite readout, while its complement accounts
    for the entire infinite tail. Event readout uses caller-owned pulse arrays.
    """
    require_no_tape()
    if not isinstance(cast(object, spec), RectangularPileupSpec):
        raise ContractError("spec must be RectangularPileupSpec")
    integer(max_pixels, "max_pixels", minimum=1)
    integer(max_events_per_pixel, "max_events_per_pixel", minimum=1)
    integer(max_owned_bytes, "max_owned_bytes", minimum=1, maximum=2**63 - 1)
    owned = (
        12 * len(spec.mark_steps)
        + 4 * len(spec.bin_edges_steps)
        + 8 * spec.bin_edges_steps[-1] * max_pixels
        + 8
    )
    if owned > max_owned_bytes:
        raise ContractError("pileup recursion exceeds max_owned_bytes")
    ctx = prepare_context(device=device, stream=stream)
    kernels = load_kernels("dpt.kernels.photon_counting")
    with ctx.scope():
        marks = ctx.wp.array(spec.mark_steps, dtype=ctx.wp.int32, device=ctx.device)
        probabilities = ctx.wp.array(
            spec.mark_probabilities, dtype=ctx.wp.float64, device=ctx.device
        )
        edges = ctx.wp.array(spec.bin_edges_steps, dtype=ctx.wp.int32, device=ctx.device)
        compound = ctx.wp.empty(
            (spec.bin_edges_steps[-1], max_pixels), dtype=ctx.wp.float64, device=ctx.device
        )
        status = ctx.wp.zeros(1, dtype=ctx.wp.int32, device=ctx.device)
    ctx.wp.synchronize_stream(ctx.stream)
    return RectangularPileupWorkspace(
        spec,
        max_pixels,
        max_events_per_pixel,
        ctx,
        marks,
        probabilities,
        edges,
        compound,
        kernels,
        status,
        StatusReadback(ctx, status),
    )


def _pileup_arrays(
    rate: Any,
    gate: Any,
    blind: Any,
    workspace: RectangularPileupWorkspace,
    stream: Any,
    validate: bool,
) -> tuple[DeviceContext, int]:
    require_no_tape()
    if type(validate) is not bool:
        raise ContractError("validate must be a boolean")
    ctx = workspace.context
    ctx.assert_stream(stream)
    pixels = ctx.array(rate, "rate_s", dtype=ctx.wp.float64)
    if not 1 <= pixels <= workspace.max_pixels:
        raise ContractError("rates must fit the prepared non-empty pixel capacity")
    ctx.array(gate, "gate_s", dtype=ctx.wp.float64, shape=(pixels,))
    ctx.array(blind, "blind_s", dtype=ctx.wp.float64, shape=(pixels,))
    return ctx, pixels


def rectangular_pileup_mean(
    rate_s: Any,
    gate_s: Any,
    blind_s: Any,
    *,
    out_counts_s: Any,
    out_accounting: Any,
    workspace: RectangularPileupWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """h=r/(1+r*Delta+r*tau), lambda=r*Delta; S=M0+sum[j=1..Poisson(lambda)] Mj.

    A live arrival opens a fixed rectangular gate Delta. Independent energy marks
    within that gate sum into one pulse, followed by a blind interval tau which
    discards arrivals without extending either interval. For lattice mark law p,
    c[0]=exp(-lambda), c[n]=(lambda/n)*sum[j<=n] j*p[j]*c[n-j], and the pulse law
    is g[n]=sum[j<=n] p[j]*c[n-j]. Expected bin rates are h*sum[bin] g[n].

    P is the pixel count and B is the number of finite comparator bins. Inputs
    are contiguous binary64 per-pixel arrays [P]: incident rate in s^-1,
    gate duration in seconds and blind duration in seconds. They are finite and
    nonnegative, and rate-duration loads and resulting energy rates must fit
    binary64. Counts [B,P] are pulses/s. Accounting [6,P] contains below-threshold
    and overflow count rates, then below-threshold, overflow, accepted and blind
    energy rates in keV/s. Count bins and the first two rows sum to h. Energy rows
    2..5 sum to r*E[M]. Overflow retains the entire infinite tail through the full
    moment h*(1+lambda)*E[M].

    The caller owns the disjoint output arrays, which each call overwrites on the
    workspace's device and stream. The workspace owns bounded log-probability
    scratch. Execution is outside a Warp tape. At Delta=0, the count law becomes
    the conventional nonparalysable rate r/(1+r*tau).

    Example::

        import hashlib
        import math
        import warp as wp
        from dpt.materials import Provenance
        from dpt.photon_counting import RectangularPileupSpec, prepare_rectangular_pileup

        law = b"analytic independent 20 keV marks and a rectangular integration gate"
        provenance = Provenance("analytic energy-mark example", hashlib.sha256(law).hexdigest(),
                                "Apache-2.0", law.decode())
        from dpt.photon_counting import rectangular_pileup_mean

        spec = RectangularPileupSpec(20.0, (1,), (1.0,), (0, 2, 3), provenance)
        workspace = prepare_rectangular_pileup(spec, max_pixels=1, device="cpu")
        rate = wp.array([4.0], dtype=wp.float64, device="cpu")
        gate = wp.array([0.25], dtype=wp.float64, device="cpu")
        blind = wp.array([0.25], dtype=wp.float64, device="cpu")
        counts = wp.empty((2, 1), dtype=wp.float64, device="cpu")
        ledger = wp.empty((6, 1), dtype=wp.float64, device="cpu")
        rectangular_pileup_mean(rate, gate, blind, out_counts_s=counts,
                                out_accounting=ledger, workspace=workspace)
        expected = 4.0 / (3.0 * math.e)
        assert all(math.isclose(float(v), expected, rel_tol=1e-14) for v in counts.numpy()[:, 0])
        assert math.isclose(float(counts.numpy().sum() + ledger.numpy()[:2].sum()), 4.0 / 3.0,
                            rel_tol=1e-14)
        assert math.isclose(float(ledger.numpy()[2:].sum()), 80.0, rel_tol=1e-14)

    """
    ctx, pixels = _pileup_arrays(rate_s, gate_s, blind_s, workspace, stream, validate)
    ctx.array(
        out_counts_s,
        "out_counts_s",
        dtype=ctx.wp.float64,
        shape=(workspace.spec.bins, pixels),
        ndim=2,
    )
    ctx.array(out_accounting, "out_accounting", dtype=ctx.wp.float64, shape=(6, pixels), ndim=2)
    ctx.disjoint(
        [("rate_s", rate_s), ("gate_s", gate_s), ("blind_s", blind_s)],
        [("out_counts_s", out_counts_s), ("out_accounting", out_accounting)],
    )
    if validate:
        workspace.check_status()
    with ctx.scope():
        ctx.wp.launch(
            workspace._kernels.pileup_mean,
            dim=pixels,
            inputs=[
                rate_s,
                gate_s,
                blind_s,
                workspace._marks,
                workspace._probabilities,
                workspace._edges,
                len(workspace.spec.mark_steps),
                workspace.spec.bins,
                workspace.spec.energy_quantum_kev,
                workspace.spec.mean_mark_kev,
                workspace._compound,
            ],
            outputs=[out_counts_s, out_accounting, workspace._status],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
    if validate:
        workspace.check_status()


def rectangular_pileup_vjp(
    rate_s: Any,
    gate_s: Any,
    blind_s: Any,
    count_cotangent: Any,
    accounting_cotangent: Any,
    *,
    out_rate_s: Any,
    out_gate_s: Any,
    out_blind_s: Any,
    workspace: RectangularPileupWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """For F(r,Delta,tau), write grad_(r,Delta,tau) <seeds,F> for the stationary mean law.

    The pulse derivative is dg/dlambda=p*g-g, including lambda=0. The renewal
    rate satisfies dh/dr=1/(1+r*Delta+r*tau)^2 and dh/dDelta=dh/dtau=-h^2.
    These derivatives include every count bin and all six accounting rows.
    Overflow and blind energy therefore retain their parameter dependence.
    The mean law is recomputed at the supplied parameter values and needs no
    preceding forward call.

    P is the pixel count and B is the number of finite comparator bins. Rate,
    gate and blind inputs are finite nonnegative contiguous binary64 [P]
    in s^-1, seconds and seconds. Signed count and accounting seeds have shapes
    [B,P] and [6,P]. Count seeds and accounting rows 0..1 carry loss per
    (pulse/s), while accounting rows 2..5 carry loss per (keV/s). The binary64
    outputs [P] carry loss per s^-1 for rate and loss per second for each duration.
    The caller owns these outputs, which must be mutually disjoint and disjoint
    from all inputs. Calls overwrite them on the workspace's device and stream
    outside a Warp tape. Supplied mark energies, probabilities and bin boundaries
    remain fixed.

    Example::

        import hashlib
        import math
        import warp as wp
        from dpt.materials import Provenance
        from dpt.photon_counting import RectangularPileupSpec, prepare_rectangular_pileup

        law = b"analytic independent 20 keV marks and a rectangular integration gate"
        provenance = Provenance("analytic energy-mark example", hashlib.sha256(law).hexdigest(),
                                "Apache-2.0", law.decode())
        from dpt.photon_counting import rectangular_pileup_vjp

        spec = RectangularPileupSpec(20.0, (1,), (1.0,), (0, 2), provenance)
        workspace = prepare_rectangular_pileup(spec, max_pixels=1, device="cpu")
        rate = wp.array([4.0], dtype=wp.float64, device="cpu")
        gate = wp.array([0.0], dtype=wp.float64, device="cpu")
        blind = wp.array([0.25], dtype=wp.float64, device="cpu")
        count_seed = wp.array([[1.0]], dtype=wp.float64, device="cpu")
        ledger_seed = wp.zeros((6, 1), dtype=wp.float64, device="cpu")
        grad_rate = wp.empty(1, dtype=wp.float64, device="cpu")
        grad_gate = wp.empty(1, dtype=wp.float64, device="cpu")
        grad_blind = wp.empty(1, dtype=wp.float64, device="cpu")
        rectangular_pileup_vjp(
            rate, gate, blind, count_seed, ledger_seed, out_rate_s=grad_rate,
            out_gate_s=grad_gate, out_blind_s=grad_blind, workspace=workspace,
        )
        assert math.isclose(float(grad_rate.numpy()[0]), 0.25, rel_tol=1e-14)
        assert math.isclose(float(grad_gate.numpy()[0]), -12.0, rel_tol=1e-14)
        assert math.isclose(float(grad_blind.numpy()[0]), -4.0, rel_tol=1e-14)

    """
    ctx, pixels = _pileup_arrays(rate_s, gate_s, blind_s, workspace, stream, validate)
    ctx.array(
        count_cotangent,
        "count_cotangent",
        dtype=ctx.wp.float64,
        shape=(workspace.spec.bins, pixels),
        ndim=2,
    )
    ctx.array(
        accounting_cotangent,
        "accounting_cotangent",
        dtype=ctx.wp.float64,
        shape=(6, pixels),
        ndim=2,
    )
    writes = [("out_rate_s", out_rate_s), ("out_gate_s", out_gate_s), ("out_blind_s", out_blind_s)]
    for name, array in writes:
        ctx.array(array, name, dtype=ctx.wp.float64, shape=(pixels,))
    ctx.disjoint(
        [
            ("rate_s", rate_s),
            ("gate_s", gate_s),
            ("blind_s", blind_s),
            ("count_cotangent", count_cotangent),
            ("accounting_cotangent", accounting_cotangent),
        ],
        writes,
    )
    if validate:
        workspace.check_status()
    with ctx.scope():
        ctx.wp.launch(
            workspace._kernels.pileup_vjp,
            dim=pixels,
            inputs=[
                rate_s,
                gate_s,
                blind_s,
                count_cotangent,
                accounting_cotangent,
                workspace._marks,
                workspace._probabilities,
                workspace._edges,
                len(workspace.spec.mark_steps),
                workspace.spec.bins,
                workspace.spec.energy_quantum_kev,
                workspace.spec.mean_mark_kev,
                workspace._compound,
            ],
            outputs=[out_rate_s, out_gate_s, out_blind_s, workspace._status],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
    if validate:
        workspace.check_status()


def rectangular_pileup_readout(
    event_times_s: Any,
    event_energies_kev: Any,
    event_counts: Any,
    gate_s: Any,
    blind_s: Any,
    *,
    out_times_s: Any,
    out_energies_kev: Any,
    out_pulse_counts: Any,
    out_bin_counts: Any,
    out_accounting_kev: Any,
    workspace: RectangularPileupWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """For each live event, E_pulse=E_start+sum[start<t<start+Delta] E_event.

    The starting event always contributes, including at Delta=0. Subsequent
    arrivals during [start+Delta,start+Delta+tau) are discarded as blind energy.
    An event at the blind interval's upper boundary starts the next pulse. Neither
    interval extends when a further event arrives. Output pulse energies retain
    the full sum before comparator binning, including readout overflow.

    P is the pixel count, Nmax the per-pixel event capacity and B the number of
    comparator bins. Inputs are contiguous binary64 event times and energies
    [P,Nmax], with seconds
    and keV units, and int32 valid-prefix counts [P]. Valid event times are
    nonnegative and strictly increasing per pixel. Energies are finite and
    strictly positive. Gate and blind durations are nonnegative binary64 [P]
    in seconds. Pulse times/energies are binary64 [P,Nmax], pulse counts int32 [P],
    and count bins uint64 [B,P]. The binary64 ledger [4,P] contains accepted,
    below-threshold, overflow and blind energy in keV. Its sum equals the energy
    of every input event in the valid prefixes.

    The caller owns all outputs, which must be disjoint from the inputs and one
    another and belong to the workspace's device and stream. A call overwrites
    all output storage, including unused pulse slots, and is bounded by the
    prepared pixel and event capacities. Event times and comparator decisions
    are fixed readout data. Execution is outside a Warp tape.

    Example::

        import hashlib
        import warp as wp
        from dpt.materials import Provenance
        from dpt.photon_counting import (
            RectangularPileupSpec, prepare_rectangular_pileup, rectangular_pileup_readout,
        )

        law = b"analytic event-time example; independent supplied lattice marks"
        provenance = Provenance("analytic pulse readout example", hashlib.sha256(law).hexdigest(),
                                "Apache-2.0", law.decode())
        spec = RectangularPileupSpec(10.0, (1, 2, 3, 4), (0.25,) * 4, (0, 4, 6), provenance)
        workspace = prepare_rectangular_pileup(spec, max_pixels=1, max_events_per_pixel=5,
                                              device="cpu")
        times = wp.array([[0.0, 0.1, 0.2, 0.25, 0.5]], dtype=wp.float64, device="cpu")
        energies = wp.array([[20.0, 30.0, 10.0, 20.0, 40.0]], dtype=wp.float64, device="cpu")
        event_counts = wp.array([5], dtype=wp.int32, device="cpu")
        gate = wp.array([0.2], dtype=wp.float64, device="cpu")
        blind = wp.array([0.1], dtype=wp.float64, device="cpu")
        pulse_times = wp.empty((1, 5), dtype=wp.float64, device="cpu")
        pulse_energies = wp.empty((1, 5), dtype=wp.float64, device="cpu")
        pulse_counts = wp.empty(1, dtype=wp.int32, device="cpu")
        counts = wp.empty((2, 1), dtype=wp.uint64, device="cpu")
        ledger = wp.empty((4, 1), dtype=wp.float64, device="cpu")
        rectangular_pileup_readout(
            times, energies, event_counts, gate, blind, out_times_s=pulse_times,
            out_energies_kev=pulse_energies, out_pulse_counts=pulse_counts,
            out_bin_counts=counts, out_accounting_kev=ledger, workspace=workspace,
        )
        assert pulse_counts.numpy().tolist() == [2]
        assert pulse_times.numpy().tolist() == [[0.0, 0.5, 0.0, 0.0, 0.0]]
        assert pulse_energies.numpy().tolist() == [[50.0, 40.0, 0.0, 0.0, 0.0]]
        assert counts.numpy().tolist() == [[0], [2]]
        assert ledger.numpy().tolist() == [[90.0], [0.0], [0.0], [30.0]]

    """
    require_no_tape()
    if type(validate) is not bool:
        raise ContractError("validate must be a boolean")
    ctx, spec = workspace.context, workspace.spec
    ctx.assert_stream(stream)
    ctx.array(event_times_s, "event_times_s", dtype=ctx.wp.float64, ndim=2)
    pixels, capacity = event_times_s.shape
    if (
        not 1 <= pixels <= workspace.max_pixels
        or not 1 <= capacity <= workspace.max_events_per_pixel
    ):
        raise ContractError("event array exceeds the prepared non-empty capacities")
    ctx.array(
        event_energies_kev,
        "event_energies_kev",
        dtype=ctx.wp.float64,
        shape=(pixels, capacity),
        ndim=2,
    )
    for name, array, dtype in (
        ("event_counts", event_counts, ctx.wp.int32),
        ("gate_s", gate_s, ctx.wp.float64),
        ("blind_s", blind_s, ctx.wp.float64),
        ("out_pulse_counts", out_pulse_counts, ctx.wp.int32),
    ):
        ctx.array(array, name, dtype=dtype, shape=(pixels,))
    ctx.array(out_times_s, "out_times_s", dtype=ctx.wp.float64, shape=(pixels, capacity), ndim=2)
    ctx.array(
        out_energies_kev, "out_energies_kev", dtype=ctx.wp.float64, shape=(pixels, capacity), ndim=2
    )
    ctx.array(
        out_bin_counts, "out_bin_counts", dtype=ctx.wp.uint64, shape=(spec.bins, pixels), ndim=2
    )
    ctx.array(
        out_accounting_kev, "out_accounting_kev", dtype=ctx.wp.float64, shape=(4, pixels), ndim=2
    )
    ctx.disjoint(
        [
            ("event_times_s", event_times_s),
            ("event_energies_kev", event_energies_kev),
            ("event_counts", event_counts),
            ("gate_s", gate_s),
            ("blind_s", blind_s),
        ],
        [
            ("out_times_s", out_times_s),
            ("out_energies_kev", out_energies_kev),
            ("out_pulse_counts", out_pulse_counts),
            ("out_bin_counts", out_bin_counts),
            ("out_accounting_kev", out_accounting_kev),
        ],
    )
    if validate:
        workspace.check_status()
    with ctx.scope():
        ctx.wp.launch(
            workspace._kernels.pileup_readout,
            dim=pixels,
            inputs=[
                event_times_s,
                event_energies_kev,
                event_counts,
                gate_s,
                blind_s,
                workspace._edges,
                spec.bins,
                spec.energy_quantum_kev,
                capacity,
            ],
            outputs=[
                out_times_s,
                out_energies_kev,
                out_pulse_counts,
                out_bin_counts,
                out_accounting_kev,
                workspace._status,
            ],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )
    if validate:
        workspace.check_status()
