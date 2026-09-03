"""Complete geometry-estimator identities and bounded resident random coordinates.

The physical projection retains the canonical source and collision layouts.
Auxiliary geometry draws occupy extension three of namespace zero and separate
absolute design columns. A physical projection is never the complete identity.
"""

from __future__ import annotations

# This composed design shares the existing canonical direction construction.
# pyright: reportPrivateUsage=false
import hashlib
import json
import operator
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, SupportsIndex, cast

from dpt._runtime import load_kernels, require_no_tape

from .model import TransportError, TransportSpec
from .rng import HistoryBatch, HistoryShard
from .sampling_identity import SOBOL_DIMENSIONS, SamplingDesign, SamplingIdentity

if TYPE_CHECKING:
    from dpt._runtime import DeviceContext

__all__ = [
    "GeometryHostSampler",
    "GeometrySamplingDesign",
    "GeometrySamplingIdentity",
    "GeometrySamplingShard",
    "GeometrySourceIdentity",
    "PreparedGeometrySampling",
    "prepare_geometry_sampling",
]

AUXILIARY_DOMAIN = 0x30000000
GEOMETRY_LAYOUT = "geometry-gauge-v1"
_MASK = 0xFFFFFFFF


def _integer(value: SupportsIndex, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool):
        raise TransportError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TransportError(f"{name} must be an integer") from error
    if not lower <= result <= upper:
        raise TransportError(f"{name} exceeds its geometry sampling budget")
    return result


@dataclass(frozen=True, slots=True)
class GeometrySamplingDesign:
    """One finite physical and auxiliary design for a geometry mean pullback.

    ``max_events=E`` is the actual collision guard, while physical counters reserve
    ``E**2`` events per branch. ``channel_slots=C`` comprises the noncoherent
    channel, the Rayleigh channel and every represented fluorescence line, so
    there are ``B=3+C`` branches. Branch zero is the original path, branch one
    serves the ballistic plane pair, branch two serves the source pair and branch
    ``3+j`` serves interface channel ``j``. Paired responses share these counters.

    The canonical physical reservation is ``Dp=3+B*E**2*event_stride`` and the
    auxiliary reservation has ``10*E+5`` coordinates. Both randomised laws reject
    total dimensions above 21,201. All native counters and dimension arithmetic
    must also fit signed int32. ``iid`` uses the same complete identity and the
    exact canonical Philox addresses, with no direction-table allocation.

    The three physical layout versions retain their existing coordinates.
    Auxiliary roles zero, one and two reserve respectively four, three and three
    lanes per prefix event. Global role three reserves three lanes and role four
    reserves two lanes. Unused auxiliary lanes are exactly 0.5.

    Example::

        from dpt.transport.geometry_sampling import GeometrySamplingDesign
        design = GeometrySamplingDesign("scrambled-sobol", 4, 8, 2)
        assert design.physical_dimensions == 4803
        assert design.dimensions == 4848

    The direction numbers and linear scramble are the same Joe-Kuo D(6) and
    LMS plus digital shift construction used by the canonical path sampler.

    Directions: https://web.maths.unsw.edu.au/~fkuo/sobol/
    """

    law: Literal["iid", "stratified", "scrambled-sobol"]
    max_events: int
    max_angle_trials: int
    channel_slots: int
    layout: Literal["legacy-v1", "packed-compton-v2", "bound-compton-v3"] = "legacy-v1"

    def __post_init__(self) -> None:
        if self.law not in ("iid", "stratified", "scrambled-sobol"):
            raise TransportError("unsupported geometry sampling law")
        if self.layout not in ("legacy-v1", "packed-compton-v2", "bound-compton-v3"):
            raise TransportError("unsupported physical sampling layout")
        for name, lower, upper in (
            ("max_events", 1, 46340),
            ("max_angle_trials", 1, 65536),
            ("channel_slots", 2, 4093),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, lower, upper))
        if self.dimensions > 2**31 - 1:
            raise TransportError("geometry dimension arithmetic exceeds signed int32")
        if self.law != "iid" and self.dimensions > SOBOL_DIMENSIONS:
            raise TransportError(
                f"complete geometry sampling requires {self.dimensions} dimensions, "
                f"exceeding the explicit {SOBOL_DIMENSIONS}-dimension budget"
            )

    @property
    def counter_span(self) -> int:
        """Physical event reservation, distinct from the actual collision guard."""
        return self.max_events**2

    @property
    def max_branches(self) -> int:
        """Original, ballistic, source and channel continuation branch count."""
        return 3 + self.channel_slots

    @property
    def event_stride(self) -> int:
        """Unchanged canonical coordinate count per physical branch event."""
        if self.layout == "packed-compton-v2":
            return 12 + 4 * ((self.max_angle_trials + 1) // 2) + 3 * self.max_angle_trials
        if self.layout == "bound-compton-v3":
            return 16 + 6 * self.max_angle_trials
        return 12 + 6 * self.max_angle_trials

    @property
    def physical_dimensions(self) -> int:
        """Canonical physical columns, including the three source columns."""
        return 3 + self.max_branches * self.counter_span * self.event_stride

    @property
    def auxiliary_dimensions(self) -> int:
        """Per-prefix and global geometry columns, disjoint from physical draws."""
        return 10 * self.max_events + 5

    @property
    def dimensions(self) -> int:
        """Complete physical and geometry coordinate count."""
        return self.physical_dimensions + self.auxiliary_dimensions

    def auxiliary_dimension(self, event: int, domain: int, lane: int) -> int:
        """Return an active absolute auxiliary column or reject an unused lane."""
        event = _integer(event, "auxiliary event", 0, self.max_events - 1)
        domain = _integer(domain, "auxiliary domain", 0, _MASK)
        lane = _integer(lane, "auxiliary lane", 0, 3)
        role = domain ^ AUXILIARY_DOMAIN
        if role < 3:
            starts, widths = (0, 4, 7), (4, 3, 3)
            if lane < widths[role]:
                return self.physical_dimensions + 10 * event + starts[role] + lane
        elif event == 0 and role == 3 and lane < 3:
            return self.physical_dimensions + 10 * self.max_events + lane
        elif event == 0 and role == 4 and lane < 2:
            return self.physical_dimensions + 10 * self.max_events + 3 + lane
        raise TransportError("counter address has no active geometry auxiliary lane")


@dataclass(frozen=True, slots=True)
class GeometrySourceIdentity:
    """Source namespace and original batch bound to the complete geometry design.

    The original ``SourceIdentity`` and caller namespace remain explicit within
    ``batch``. This token includes every physical continuation and auxiliary role
    rather than describing only the canonical physical projection.
    """

    design: GeometrySamplingDesign
    batch: HistoryBatch
    version: Literal["geometry-gauge-v1"] = GEOMETRY_LAYOUT

    def __post_init__(self) -> None:
        _check_complete(self.design, self.batch, self.version)


def _check_complete(design: GeometrySamplingDesign, batch: HistoryBatch, version: str) -> None:
    if not isinstance(cast(Any, design), GeometrySamplingDesign) or not isinstance(
        cast(Any, batch), HistoryBatch
    ):
        raise TransportError("geometry sampling requires a design and original HistoryBatch")
    if version != GEOMETRY_LAYOUT:
        raise TransportError("unsupported complete geometry layout version")
    if batch.sampling is not None:
        raise TransportError("geometry preparation requires a complete unbound original batch")
    if design.law != "iid" and (batch.count < 2 or batch.count & (batch.count - 1)):
        raise TransportError("a complete randomised geometry design requires a power-of-two count")
    if design.law != "iid" and batch.count > 2**30:
        raise TransportError("a complete randomised geometry design exceeds its point-index budget")


@dataclass(frozen=True, slots=True)
class GeometrySamplingIdentity:
    """Complete immutable identity retained by all original-history shards.

    The digest binds the full original range, seed, actual event guard, squared
    counter span, physical layout, branches, source namespace and structural
    source token, plus the fixed auxiliary coordinate version. A same-seed shard
    remains dependent on every other shard of that design.
    """

    design: GeometrySamplingDesign
    batch: HistoryBatch
    version: Literal["geometry-gauge-v1"] = GEOMETRY_LAYOUT

    def __post_init__(self) -> None:
        _check_complete(self.design, self.batch, self.version)

    @property
    def identity(self) -> str:
        """SHA256 of all complete structural sampling fields."""
        payload = {
            "version": self.version,
            "design": asdict(self.design),
            "batch": asdict(self.batch),
            "counter_span": self.design.counter_span,
            "physical_dimensions": self.design.physical_dimensions,
            "auxiliary_dimensions": self.design.auxiliary_dimensions,
            "auxiliary_domain": AUXILIARY_DOMAIN,
            "auxiliary_roles": ((0, 4), (1, 3), (2, 3), (3, 3), (4, 2)),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def source_identity(self) -> GeometrySourceIdentity:
        """Complete source token, including the geometry coordinate reservation."""
        return GeometrySourceIdentity(self.design, self.batch, self.version)

    @property
    def physical_projection(self) -> SamplingIdentity | None:
        """Canonical physical-only projection, never the complete design identity."""
        if self.design.law == "iid":
            return None
        physical = SamplingDesign(
            self.design.law,
            self.design.counter_span,
            self.design.max_angle_trials,
            self.design.max_branches,
            self.design.layout,
        )
        return SamplingIdentity(
            physical, self.batch.seed, self.batch.first_history, self.batch.count
        )

    def independent_of(self, other: GeometrySamplingIdentity) -> bool:
        """Use full randomisations and original source declarations for independence."""
        if not isinstance(cast(Any, other), GeometrySamplingIdentity):
            raise TransportError("geometry independence requires complete geometry identities")
        if self.design.law != "iid" or other.design.law != "iid":
            return self.batch.seed != other.batch.seed and self.batch.independent_of(other.batch)
        return self.batch.independent_of(other.batch)

    def shard(self, parts: SupportsIndex, rank: SupportsIndex) -> GeometrySamplingShard:
        """Partition the original range while retaining this complete design."""
        return GeometrySamplingShard(self, self.batch.shard(parts, rank))


@dataclass(frozen=True, slots=True)
class GeometrySamplingShard:
    """A checked original-history subset sharing its complete geometry identity."""

    identity: GeometrySamplingIdentity
    histories: HistoryShard

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(Any, self.identity), GeometrySamplingIdentity)
            or not isinstance(cast(Any, self.histories), HistoryShard)
            or self.histories.original_batch != self.identity.batch
        ):
            raise TransportError("geometry shard must belong to its complete original batch")

    @property
    def batch(self) -> HistoryBatch:
        """Original counter addresses for this subset, without projection rebinding."""
        return self.histories.batch


def _philox(seed: int, history: int, event: int, domain: int) -> tuple[int, int, int, int]:
    c0, c1, c2, c3 = history & _MASK, history >> 32, event, domain
    k0, k1 = seed & _MASK, seed >> 32
    for _ in range(10):
        p0, p1 = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (p1 >> 32) ^ c1 ^ k0, p1 & _MASK, (p0 >> 32) ^ c3 ^ k1, p0 & _MASK
        k0, k1 = (k0 + 0x9E3779B9) & _MASK, (k1 + 0xBB67AE85) & _MASK
    return c0, c1, c2, c3


def _geometry_words(identity: GeometrySamplingIdentity) -> tuple[Any, Any]:
    """Generate every absolute column using the unchanged canonical scramble recipe."""
    import numpy as np

    from .sampling import _philox_words, _sobol_directions

    dimension = np.arange(identity.design.dimensions, dtype=np.uint64)
    if identity.design.law == "stratified":
        words = _philox_words(identity.batch.seed, dimension, 0xFFFFFFFF, 0)
        multiplier, offset = words[:, 0] | np.uint32(1), words[:, 1].copy()
        multiplier.flags.writeable = offset.flags.writeable = False
        return multiplier, offset
    directions = _sobol_directions(identity.design.dimensions)
    scrambled = np.zeros_like(directions)
    for row in range(32):
        random = _philox_words(identity.batch.seed, dimension, 0xFFFFFFFD, row)[:, 0]
        diagonal = np.uint32(1 << (31 - row))
        earlier = np.uint32((_MASK << (32 - row)) & _MASK)
        mask = (random & earlier) | diagonal
        parity = np.bitwise_count(directions & mask[:, None]) & np.uint8(1)
        scrambled |= parity.astype(np.uint32) << np.uint32(31 - row)
    shift = _philox_words(identity.batch.seed, dimension, 0xFFFFFFFE, 0)[:, 0].copy()
    scrambled.flags.writeable = shift.flags.writeable = False
    return scrambled, shift


class GeometryHostSampler:
    """Exact host provider for physical and auxiliary geometry counter addresses.

    ``uniform((history, branch, event_offset), event, domain, lane)`` returns a
    binary64 open-interval value. Physical domains exclude the branch bits, which
    this provider adds explicitly. Local physical ``event`` is below the actual
    guard and ``event_offset+event`` is below ``E**2``. Auxiliary requests require
    branch and offset zero. Source sentinel events require the original lineage.
    QMC preparation uses the licensed directions and NumPy once, while subsequent
    scalar evaluations use Python integer operations without importing Warp.

    Example::

        from dpt.transport.rng import HistoryBatch
        from dpt.transport.geometry_sampling import (
            GeometryHostSampler, GeometrySamplingDesign, GeometrySamplingIdentity)
        identity = GeometrySamplingIdentity(
            GeometrySamplingDesign("iid", 4, 8, 2), HistoryBatch(17, 0, 16))
        provider = GeometryHostSampler(identity)
        assert 0.0 < provider.uniform((0, 3, 8), 1, 0, 0) < 1.0

    """

    def __init__(self, identity: GeometrySamplingIdentity) -> None:
        if not isinstance(cast(Any, identity), GeometrySamplingIdentity):
            raise TransportError("host geometry provider requires a complete identity")
        self.complete_identity = identity
        self.identity = identity.identity
        self.layout: Literal["legacy-v1", "packed-compton-v2", "bound-compton-v3"] = (
            identity.design.layout
        )
        self._projection = identity.physical_projection
        if identity.design.law == "iid":
            self._words, self._shift = (), ()
        else:
            words, shift = _geometry_words(identity)
            self._words = tuple(int(value) for value in words.reshape(-1))
            self._shift = tuple(int(value) for value in shift)

    def uniform(self, lineage: tuple[int, ...], event: int, domain: int, lane: int) -> float:
        """Evaluate one checked address without changing any original history id."""
        if not isinstance(cast(Any, lineage), tuple) or len(lineage) != 3:
            raise TransportError("geometry lineage requires history, branch and event offset")
        identity, design = self.complete_identity, self.complete_identity.design
        history = _integer(
            lineage[0],
            "original history",
            identity.batch.first_history,
            identity.batch.first_history + identity.batch.count - 1,
        )
        branch = _integer(lineage[1], "physical branch", 0, design.max_branches - 1)
        offset = _integer(lineage[2], "physical event offset", 0, design.counter_span - 1)
        if (branch < 3 and offset != 0) or offset % design.max_events:
            raise TransportError("physical lineage offset does not match its continuation role")
        event = _integer(event, "event", 0, _MASK)
        domain = _integer(domain, "domain", 0, _MASK)
        lane = _integer(lane, "lane", 0, 3)
        auxiliary = (domain & 0xF0000000) == AUXILIARY_DOMAIN
        if auxiliary:
            if branch != 0 or offset != 0:
                raise TransportError("auxiliary lineage must retain branch and offset zero")
            # Validate lane zero first so an invalid role cannot appear unused.
            design.auxiliary_dimension(event, domain, 0)
            try:
                dimension = design.auxiliary_dimension(event, domain, lane)
            except TransportError:
                return 0.5
            counter = event
        else:
            if domain & 0x0FFF0000:
                raise TransportError("physical domain must exclude lineage branch bits")
            if event in (0x80000000, 0x80000002):
                if branch != 0 or offset != 0 or domain != 0:
                    raise TransportError("source draws require the original physical lineage")
            elif event >= design.max_events or offset + event >= design.counter_span:
                raise TransportError("physical counter exceeds its local or reserved event budget")
            counter, domain = offset + event, domain | (branch << 16)
            if self._projection is not None:
                self._projection.design.dimension(counter, domain, 0)
                try:
                    dimension = self._projection.design.dimension(counter, domain, lane)
                except TransportError:
                    return 0.5
            else:
                # Use a metadata-only layout validator without imposing QMC's
                # direction limit on an IID reservation.
                _physical_address(design, counter, domain, lane)
                dimension = 0
        jitter = (_philox(identity.batch.seed, history, counter, domain)[lane] + 0.5) / 2**32
        if design.law == "iid":
            return jitter
        index = history - identity.batch.first_history
        if design.law == "stratified":
            stratum = (index * self._words[dimension] + self._shift[dimension]) & (
                identity.batch.count - 1
            )
            return min((stratum + jitter) / identity.batch.count, 0.9999999999999999)
        gray, value, bit = index ^ (index >> 1), self._shift[dimension], 0
        while gray:
            if gray & 1:
                value ^= self._words[dimension * 32 + bit]
            gray >>= 1
            bit += 1
        return (value + 0.5) / 2**32


def _physical_address(design: GeometrySamplingDesign, event: int, domain: int, lane: int) -> None:
    """Admit canonical IID addresses while retaining all historical Philox lanes."""
    if event in (0x80000000, 0x80000002) and domain == 0:
        return
    namespace, extension, role = domain >> 30, (domain >> 28) & 3, domain & 0xFFFF
    if extension:
        if (
            design.layout == "bound-compton-v3"
            and namespace == 0
            and extension in (1, 2)
            and role == 0
        ):
            return
    elif namespace == 0 and role < 4:
        return
    elif namespace == 1 and role < 2:
        return
    elif namespace == 2 and role < (
        (design.max_angle_trials + 1) // 2
        if design.layout == "packed-compton-v2"
        else design.max_angle_trials
    ):
        return
    elif namespace == 3 and role < design.max_angle_trials:
        return
    raise TransportError("counter address has no declared physical sampling role")


@dataclass(slots=True)
class PreparedGeometrySampling:
    """One owning-device allocation and records for a complete geometry design.

    ``physical_record`` is the canonical physical projection and
    ``auxiliary_record`` addresses the remaining columns of the same owned
    arrays. Geometry workspaces retain this object and its complete identity.
    Preparation never alters an ordinary transport workspace or its sampler.
    """

    identity: GeometrySamplingIdentity
    context: DeviceContext
    physical_record: Any
    auxiliary_record: Any
    _words: Any
    _shift: Any

    @property
    def allocated_bytes(self) -> int:
        """Total owned uint32 device storage, counted once across both records."""
        return 4 * (int(self._words.size) + int(self._shift.size))

    def assert_identity(
        self, identity: GeometrySamplingIdentity | GeometrySamplingShard
    ) -> HistoryBatch:
        """Check the complete token and return this execution's original range."""
        complete = identity.identity if isinstance(identity, GeometrySamplingShard) else identity
        if complete != self.identity:
            raise TransportError("geometry batch does not match the prepared complete design")
        self.context.assert_stream()
        return identity.batch


def prepare_geometry_sampling(
    context: DeviceContext,
    *,
    spec: TransportSpec,
    batch: HistoryBatch,
    design: GeometrySamplingDesign,
    max_fluorescence_lines: int,
) -> PreparedGeometrySampling:
    """Prepare physical and auxiliary records for a geometry workspace.

    The immutable transport spec must use the design's actual event guard,
    covered angular trial budget and exact physical layout. ``channel_slots``
    must equal ``2+max_fluorescence_lines`` from the represented material table.
    ``batch`` is the complete unbound original range. All upload allocations
    occur here on the owning device and stream, outside execution and Warp Tape.
    IID owns two empty uint32 arrays and retains canonical Philox coordinates.

    The records contain flat uint32 ``words`` and ``shift`` arrays, with
    ``D`` and ``D`` words for stratification or ``32*D`` and ``D`` words for
    scrambled Sobol, where ``D`` is the complete dimension count. All counters,
    uniforms and identity fields are dimensionless. The physical and auxiliary
    records share these allocations, and execution must retain the returned
    object and its owning ``DeviceContext``.

    Example::

        from dpt._runtime import prepare_context
        from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
        from dpt.transport.rng import HistoryBatch
        from dpt.transport.geometry_sampling import (
            GeometrySamplingDesign, prepare_geometry_sampling)
        spec = TransportSpec(
            MaterialGrid((0., 0., 0.), (1., 1., 1.), (1, 1, 1), 1),
            PlanarDetector((0., 0.), (1., 1.), (1, 1), 2.),
            80., "declared material table", max_events=2, max_angle_trials=4)
        prepared = prepare_geometry_sampling(
            prepare_context(device="cpu"), spec=spec, batch=HistoryBatch(17, 0, 64),
            design=GeometrySamplingDesign("stratified", 2, 4, 2),
            max_fluorescence_lines=0)
        assert prepared.physical_record.dimensions == 723
        assert prepared.allocated_bytes == 8 * 748

    """
    require_no_tape()
    from dpt._runtime import DeviceContext

    if not isinstance(cast(Any, spec), TransportSpec) or not isinstance(
        cast(Any, context), DeviceContext
    ):
        raise TransportError(
            "geometry sampling preparation requires a TransportSpec and DeviceContext"
        )
    identity = GeometrySamplingIdentity(design, batch)
    lines = _integer(max_fluorescence_lines, "represented fluorescence lines", 0, 4091)
    expected_layout = (
        "bound-compton-v3"
        if spec.scattering_law == "bound-electron-compton"
        else "packed-compton-v2"
        if spec.compton_sampler == "packed-uniform-rejection"
        else "legacy-v1"
    )
    if design.layout != expected_layout:
        raise TransportError(
            "geometry physical layout does not match the transport execution policy"
        )
    if design.max_events != spec.max_events or design.max_angle_trials < spec.max_angle_trials:
        raise TransportError(
            "geometry sampling budgets must match the actual event guard and cover angular trials"
        )
    if design.channel_slots != 2 + lines:
        raise TransportError(
            "geometry channel slots must match every represented fluorescence line"
        )
    wp = context.wp
    kernels = load_kernels("dpt.transport.geometry_sampling_kernels")
    sampling_kernels = load_kernels("dpt.transport.sampling_kernels")
    if design.law == "iid":
        with context.scope():
            resident_words = wp.empty(0, dtype=wp.uint32, device=context.device)
            resident_shift = wp.empty(0, dtype=wp.uint32, device=context.device)
    else:
        words, shift = _geometry_words(identity)
        with context.scope():
            resident_words = wp.array(words.reshape(-1), dtype=wp.uint32, device=context.device)
            resident_shift = wp.array(shift, dtype=wp.uint32, device=context.device)
    physical = sampling_kernels.RandomDesign()
    physical.kind = {"iid": 0, "stratified": 1, "scrambled-sobol": 2}[design.law]
    physical.seed, physical.first_history, physical.count = (
        wp.uint64(batch.seed),
        wp.uint64(batch.first_history),
        wp.uint32(batch.count),
    )
    physical.max_events, physical.max_trials, physical.max_branches = (
        design.counter_span,
        design.max_angle_trials,
        design.max_branches,
    )
    physical.layout = {"legacy-v1": 0, "packed-compton-v2": 1, "bound-compton-v3": 2}[design.layout]
    physical.stride, physical.dimensions = design.event_stride, design.physical_dimensions
    physical.words, physical.shift = resident_words, resident_shift
    auxiliary = kernels.GeometryRandomDesign()
    auxiliary.physical = physical
    auxiliary.event_guard, auxiliary.physical_dimensions, auxiliary.dimensions = (
        design.max_events,
        design.physical_dimensions,
        design.dimensions,
    )
    return PreparedGeometrySampling(
        identity, context, physical, auxiliary, resident_words, resident_shift
    )
