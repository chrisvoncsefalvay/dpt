"""Finite full-path coordinate budgets and complete randomised design identities."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Literal, SupportsIndex, cast

from .model import TransportError

SOBOL_DIMENSIONS = 21201
SOURCE_DIMENSIONS = 3


def _integer(value: SupportsIndex, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool):
        raise TransportError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TransportError(f"{name} must be an integer") from error
    if not lower <= result <= upper:
        raise TransportError(f"{name} is outside its declared sampling range")
    return result


@dataclass(frozen=True, slots=True)
class SamplingDesign:
    """One finite randomised design for the entire photon path.

    ``law`` is ``stratified`` or ``scrambled-sobol``. The event, angular trial
    and descendant budgets are counts and have no physical units. Every source,
    collision, null collision, angular rejection, fluorescence, next-event and
    roulette draw has a unique coordinate. Unused roles retain their coordinates
    so changing physical model features cannot renumber an existing draw.

    The dimension count is ``3 + max_branches * max_events *
    (12 + 6 * max_angle_trials)``. Both laws use the same bounded coordinate
    contract and accept at most 21,201 dimensions. ``layout="legacy-v1"``
    preserves every historical coordinate. ``packed-compton-v2`` uses
    ``12 + 4*ceil(max_angle_trials/2) + 3*max_angle_trials`` coordinates per
    event and branch. Its Compton domain identifies a block, lanes zero/one
    and two/three supply its two proposal/acceptance pairs, and collision lane
    two supplies the azimuth. Rayleigh domains still identify individual trials
    and retain three lanes after the packed Compton block coordinates.
    Preparation rejects an execution/layout mismatch. Exhaustion raises rather than
    switching to IID draws. Complete designs contain a power of two histories.

    Example::

        design = SamplingDesign("scrambled-sobol", max_events=32,
                                max_angle_trials=32, max_branches=2)
        assert design.dimensions == 13059

    Sobol directions use Joe and Kuo's D(6) construction [SIAM J. Sci. Comput.
    30 (2008), 2635-2654]. Randomisation is a binary left linear matrix scramble
    followed by an independent digital shift [Owen, J. Complexity 14 (1998),
    466-489]. Stratification uses independently keyed affine permutations of
    the dyadic strata and an independently addressed jitter within each stratum.
    """

    law: Literal["stratified", "scrambled-sobol"]
    max_events: int
    max_angle_trials: int
    max_branches: int = 1
    layout: Literal["legacy-v1", "packed-compton-v2", "bound-compton-v3"] = "legacy-v1"

    def __post_init__(self) -> None:
        if self.law not in ("stratified", "scrambled-sobol"):
            raise TransportError("sampling law must be stratified or scrambled-sobol")
        if self.layout not in ("legacy-v1", "packed-compton-v2", "bound-compton-v3"):
            raise TransportError("unsupported full-path sampling layout version")
        for name, maximum in (
            ("max_events", 2**31 - 1),
            ("max_angle_trials", 65536),
            ("max_branches", 4096),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1, maximum))
        if self.dimensions > SOBOL_DIMENSIONS:
            raise TransportError(
                f"full-path sampling requires {self.dimensions} dimensions, exceeding "
                f"the explicit {SOBOL_DIMENSIONS}-dimension budget"
            )

    @property
    def event_stride(self) -> int:
        """Number of coordinates reserved for one event of one descendant."""
        if self.layout == "packed-compton-v2":
            return 12 + 4 * ((self.max_angle_trials + 1) // 2) + 3 * self.max_angle_trials
        if self.layout == "bound-compton-v3":
            return 16 + 6 * self.max_angle_trials
        return 12 + 6 * self.max_angle_trials

    @property
    def dimensions(self) -> int:
        """Total resident coordinate count, including all bounded path roles."""
        return SOURCE_DIMENSIONS + self.max_branches * self.max_events * self.event_stride

    def dimension(self, event: int, domain: int, lane: int) -> int:
        """Map an actual counter address to one zero-based design coordinate.

        Source position lanes are coordinates 0 and 1 and spectrum is coordinate
        2. Per-event coordinates are collision 0:4, Woodcock 4, fluorescence 5:8,
        coherent channel 8, next-event 9:11, roulette 11, then three lanes for
        each Compton trial and three for each Rayleigh trial in ``legacy-v1``.
        ``packed-compton-v2`` reserves four lanes per Compton block followed by
        three per Rayleigh trial. Collision lane two becomes its independent
        Compton azimuth without acquiring a second coordinate. Invalid addresses
        and unused lanes are rejected by this host inspection interface.
        """
        event = _integer(event, "event", 0, 2**32 - 1)
        domain = _integer(domain, "domain", 0, 2**32 - 1)
        lane = _integer(lane, "lane", 0, 3)
        if event == 0x80000000 and domain == 0 and lane < 2:
            return lane
        if event == 0x80000002 and domain == 0 and lane == 0:
            return 2
        branch = (domain >> 16) & 0xFFF
        role = domain & 0xFFFF
        namespace = domain >> 30
        extension = (domain >> 28) & 3
        if event >= self.max_events or branch >= self.max_branches:
            raise TransportError("counter address exceeds its full-path sampling budget")
        base = 3 + (branch * self.max_events + event) * self.event_stride
        if extension:
            if (
                self.layout == "bound-compton-v3"
                and namespace == 0
                and extension in (1, 2)
                and role == 0
                and lane < 2
            ):
                return base + 12 + 6 * self.max_angle_trials + 2 * (extension - 1) + lane
            raise TransportError("counter address has no declared bound Compton sampling lane")
        if namespace == 0:
            starts = (0, 4, 5, 8)
            widths = (4, 1, 3, 1)
            if role < 4 and lane < widths[role]:
                return base + starts[role] + lane
        elif namespace == 1:
            if role == 0 and lane < 2:
                return base + 9 + lane
            if role == 1 and lane == 0:
                return base + 11
        elif self.layout == "packed-compton-v2":
            blocks = (self.max_angle_trials + 1) // 2
            if namespace == 2 and role < blocks and lane < 4:
                return base + 12 + 4 * role + lane
            if namespace == 3 and role < self.max_angle_trials and lane < 3:
                return base + 12 + 4 * blocks + 3 * role + lane
        elif role < self.max_angle_trials and lane < 3:
            return base + 12 + (namespace - 2) * 3 * self.max_angle_trials + 3 * role + lane
        raise TransportError("counter address has no declared active sampling lane")


@dataclass(frozen=True, slots=True)
class SamplingIdentity:
    """Identity of a complete design, retained unchanged by all history shards.

    ``seed`` selects the independent scramble or stratum permutations. Histories
    have design index ``history - first_history``. Subdividing a complete range
    changes neither its point index nor its randomisation. Original history
    addresses remain the stratum-jitter keys. Two disjoint subsets or complete
    ranges of the same seed remain dependent designs.
    """

    design: SamplingDesign
    seed: int
    first_history: int
    count: int

    def __post_init__(self) -> None:
        if not isinstance(cast(Any, self.design), SamplingDesign):
            raise TransportError("sampling identity requires a SamplingDesign")
        for name, lower, upper in (
            ("seed", 0, 2**64 - 1),
            ("first_history", 0, 2**64 - 1),
            ("count", 2, 2**30),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, lower, upper))
        if self.count & (self.count - 1):
            raise TransportError("a complete randomised design requires a power-of-two count")
        if self.first_history + self.count > 2**64:
            raise TransportError("the complete sampling design would wrap history identity")

    def independent_of(self, other: SamplingIdentity) -> bool:
        """Require distinct randomisations, including for disjoint history ranges."""
        return self.seed != other.seed
