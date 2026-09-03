"""Preparation and host inspection of randomised full-path transport designs."""

from __future__ import annotations

# Transport launch records belong to the prepared workspace.
# pyright: reportPrivateUsage=false
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files
from typing import TYPE_CHECKING, Any, cast

from dpt._runtime import load_kernels, require_no_tape

from .model import TransportError
from .rng import HistoryBatch, HistoryShard
from .sampling_identity import SamplingDesign, SamplingIdentity

if TYPE_CHECKING:
    from .forward import TransportWorkspace

__all__ = [
    "PreparedSampling",
    "_bind_sampling",
    "design_replicate_moments",
    "pool_design_shard_means",
    "prepare_transport_sampling",
    "sampling_uniforms",
]


def _philox_words(seed: int, identities: Any, event: int, domain: int) -> Any:
    """Vectorised host Philox words for preparation; not a runtime dependency."""
    import numpy as np

    mask = np.uint64(0xFFFFFFFF)
    identity = np.asarray(identities, dtype=np.uint64)
    c0 = identity & mask
    c1 = identity >> np.uint64(32)
    c2 = np.full(identity.shape, event, dtype=np.uint64)
    c3 = np.full(identity.shape, domain, dtype=np.uint64)
    k0, k1 = seed & 0xFFFFFFFF, seed >> 32
    for _ in range(10):
        p0 = np.uint64(0xD2511F53) * c0
        p1 = np.uint64(0xCD9E8D57) * c2
        c0, c1, c2, c3 = (
            (p1 >> np.uint64(32)) ^ c1 ^ np.uint64(k0),
            p1 & mask,
            (p0 >> np.uint64(32)) ^ c3 ^ np.uint64(k1),
            p0 & mask,
        )
        k0 = (k0 + 0x9E3779B9) & 0xFFFFFFFF
        k1 = (k1 + 0xBB67AE85) & 0xFFFFFFFF
    return np.stack((c0, c1, c2, c3), axis=-1).astype(np.uint32)


@lru_cache(maxsize=8)
def _sobol_directions(dimensions: int) -> Any:
    """Construct 32-bit Sobol directions from the licensed Joe-Kuo D(6) data."""
    import numpy as np

    resource = files("dpt.transport").joinpath("data/sobol_joe_kuo_6_21201.npz")
    with as_file(resource) as path, np.load(path, allow_pickle=False) as data:
        polynomials = data["poly"][:dimensions]
        initial = data["vinit"][:dimensions]
    directions = np.zeros((dimensions, 32), dtype=np.uint32)
    directions[0] = np.asarray([1 << (31 - bit) for bit in range(32)], dtype=np.uint32)
    for dimension in range(1, dimensions):
        polynomial = int(polynomials[dimension])
        degree = polynomial.bit_length() - 1
        for bit in range(degree):
            directions[dimension, bit] = int(initial[dimension, bit]) << (31 - bit)
        for bit in range(degree, 32):
            value = int(directions[dimension, bit - degree])
            value ^= value >> degree
            for coefficient in range(1, degree):
                if polynomial & (1 << (degree - coefficient)):
                    value ^= int(directions[dimension, bit - coefficient])
            directions[dimension, bit] = value
    directions.flags.writeable = False
    return directions


@lru_cache(maxsize=8)
def _design_words(identity: SamplingIdentity) -> tuple[Any, Any]:
    """Return affine stratum permutations or LMS directions and digital shifts."""
    import numpy as np

    dimension = np.arange(identity.design.dimensions, dtype=np.uint64)
    if identity.design.law == "stratified":
        words = _philox_words(identity.seed, dimension, 0xFFFFFFFF, 0)
        multiplier = words[:, 0] | np.uint32(1)
        offset = words[:, 1]
        multiplier.flags.writeable = offset.flags.writeable = False
        return multiplier, offset
    directions = _sobol_directions(identity.design.dimensions)
    scrambled = np.zeros_like(directions)
    # Output bit r depends on input bits 0..r in MSB-first notation. The unit
    # diagonal makes each binary matrix invertible and preserves net balance.
    for row in range(32):
        random = _philox_words(identity.seed, dimension, 0xFFFFFFFD, row)[:, 0]
        diagonal = np.uint32(1 << (31 - row))
        earlier = np.uint32((0xFFFFFFFF << (32 - row)) & 0xFFFFFFFF)
        mask = (random & earlier) | diagonal
        parity = np.bitwise_count(directions & mask[:, None]) & np.uint8(1)
        scrambled |= parity.astype(np.uint32) << np.uint32(31 - row)
    shift = _philox_words(identity.seed, dimension, 0xFFFFFFFE, 0)[:, 0].copy()
    scrambled.flags.writeable = shift.flags.writeable = False
    return scrambled, shift


def sampling_uniforms(
    identity: SamplingIdentity,
    *,
    event: int,
    domain: int,
    histories: Any = None,
) -> Any:
    """Inspect four full-path uniforms on the CPU without importing Warp.

    The return value is a NumPy binary64 array of shape ``(N, 4)`` with values in
    the open unit interval. ``histories`` contains original uint64 addresses and
    defaults to the complete design. Unused lanes are the exact constant 0.5.
    This function uses the same coordinate layout and randomisation as resident
    transport but evaluates points independently of launch order.

    Example::

        identity = SamplingIdentity(SamplingDesign("stratified", 4, 8), 19, 0, 256)
        optical_depth_draws = sampling_uniforms(identity, event=0, domain=0)[:, 0]
    """
    import numpy as np

    if not isinstance(cast(Any, identity), SamplingIdentity):
        raise TransportError("sampling_uniforms requires a complete SamplingIdentity")
    # Every admitted role has an active lane zero, including the two source
    # namespaces. Validate integer counters before constructing uint64 arrays.
    identity.design.dimension(event, domain, 0)
    if histories is None:
        histories = np.arange(identity.count, dtype=np.uint64) + np.uint64(identity.first_history)
    history = np.asarray(histories)
    if (
        history.ndim != 1
        or history.dtype.kind not in "iu"
        or (history.dtype.kind == "i" and bool(np.any(history < 0)))
    ):
        raise TransportError("histories must be a one-dimensional integer address array")
    history = history.astype(np.uint64)
    index = history - np.uint64(identity.first_history)
    if bool(np.any(history < identity.first_history)) or bool(np.any(index >= identity.count)):
        raise TransportError("history is outside its complete sampling design")
    words, shift = _design_words(identity)
    result = np.full((len(history), 4), 0.5, dtype=np.float64)
    jitter = _philox_words(identity.seed, history, event, domain)
    active = 0
    for lane in range(4):
        try:
            dimension = identity.design.dimension(event, domain, lane)
        except TransportError:
            continue
        active += 1
        if identity.design.law == "stratified":
            stratum = (index * np.uint64(words[dimension]) + np.uint64(shift[dimension])) & (
                identity.count - 1
            )
            uniform = (jitter[:, lane].astype(np.float64) + 0.5) / 2**32
            result[:, lane] = np.minimum(
                (stratum.astype(np.float64) + uniform) / identity.count,
                np.nextafter(1.0, 0.0),
            )
        else:
            gray = index ^ (index >> np.uint64(1))
            value = np.full(len(history), shift[dimension], dtype=np.uint32)
            for bit in range(32):
                value ^= np.where(
                    (gray >> np.uint64(bit)) & np.uint64(1), words[dimension, bit], np.uint32(0)
                )
            result[:, lane] = (value.astype(np.float64) + 0.5) / 2**32
    if active == 0:
        raise TransportError("counter address is outside the declared sampling design")
    return result


@dataclass(slots=True)
class PreparedSampling:
    """Resident finite coordinate table for one complete randomisation."""

    identity: SamplingIdentity
    _record: Any
    _words: Any
    _shift: Any

    @property
    def allocated_bytes(self) -> int:
        """Owned device bytes, excluding the containing transport workspace."""
        return 4 * (int(self._words.size) + int(self._shift.size))


def prepare_transport_sampling(
    workspace: TransportWorkspace,
    *,
    batch: HistoryBatch,
    design: SamplingDesign,
) -> HistoryBatch:
    """Bind one complete full-path design and return its structural batch token.

    Preparation allocates and uploads uint32 coordinate records on the owning
    device. Execution and all shards reuse these records without allocation.
    The event and rejection budgets must cover the prepared transport spec, and
    the descendant budget must cover any subsequently prepared splitting tree.
    Preparing a new seed is a new randomisation and replaces the resident design.

    Use ``history_mean`` or complete-batch replicate means for this design.
    ``history_moments`` rejects within-design IID variance estimates. Independent
    complete randomisations estimate the variance of their mean as
    ``sum((mean_r - mean)**2)/(R*(R-1))``.

    Example::

        batch = prepare_transport_sampling(
            transport, batch=HistoryBatch(17, 0, 4096),
            design=SamplingDesign("scrambled-sobol", 32, 32))
        shard = batch.shard(4, 1)
        # Source and path draws retain index history - batch.sampling.first_history.
    """
    require_no_tape()
    from .forward import TransportWorkspace

    if not isinstance(cast(Any, workspace), TransportWorkspace):
        raise TransportError("sampling preparation requires a TransportWorkspace")
    if not isinstance(cast(Any, batch), HistoryBatch) or not isinstance(
        cast(Any, design), SamplingDesign
    ):
        raise TransportError("sampling preparation requires a HistoryBatch and SamplingDesign")
    if batch.sampling is not None:
        raise TransportError("sampling preparation requires a complete unbound history batch")
    expected_layout = (
        "bound-compton-v3"
        if workspace.spec.scattering_law == "bound-electron-compton"
        else "packed-compton-v2"
        if workspace.spec.compton_sampler == "packed-uniform-rejection"
        else "legacy-v1"
    )
    if design.layout != expected_layout:
        raise TransportError("sampling layout version does not match the Compton execution policy")
    if design.max_events < workspace.spec.max_events or (
        design.max_angle_trials < workspace.spec.max_angle_trials
    ):
        raise TransportError(
            "sampling coordinate budget must cover the transport event/trial budgets"
        )
    identity = SamplingIdentity(design, batch.seed, batch.first_history, batch.count)
    words, shift = _design_words(identity)
    context = workspace.context
    wp = context.wp
    kernels = load_kernels("dpt.transport.sampling_kernels")
    with context.scope():
        resident_words = wp.array(words.reshape(-1), dtype=wp.uint32, device=context.device)
        resident_shift = wp.array(shift, dtype=wp.uint32, device=context.device)
    record = kernels.RandomDesign()
    record.kind = 1 if design.law == "stratified" else 2
    record.seed = wp.uint64(batch.seed)
    record.first_history = wp.uint64(batch.first_history)
    record.count = wp.uint32(batch.count)
    record.max_events = design.max_events
    record.max_trials = design.max_angle_trials
    record.max_branches = design.max_branches
    record.layout = (
        2 if design.layout == "bound-compton-v3" else int(design.layout == "packed-compton-v2")
    )
    record.stride = design.event_stride
    record.dimensions = design.dimensions
    record.words = resident_words
    record.shift = resident_shift
    workspace._sampling = PreparedSampling(identity, record, resident_words, resident_shift)
    workspace._parameters.sampling = record
    return batch.with_sampling(identity)


def _bind_sampling(workspace: TransportWorkspace, batch: HistoryBatch) -> Any:
    """Check a complete design or shard and bind its existing resident record."""
    identity = batch.sampling
    if identity is None:
        current = workspace._parameters.sampling
        payload = bytes(current.__ctype__())
        # The canonical IID record is entirely zero. Reusing it avoids Warp's
        # nested-struct copy, including its two array descriptors, on every call.
        # A nonzero record still resets fully when switching back from a design.
        if payload == bytes(len(payload)):
            return current
        record = workspace._kernels.RandomDesign()
    else:
        prepared = workspace._sampling
        if prepared is None or prepared.identity != identity:
            raise TransportError(
                "batch sampling identity does not match the prepared randomisation"
            )
        record = prepared._record
    workspace._parameters.sampling = record
    return record


def design_replicate_moments(means: Any, *, batches: tuple[HistoryBatch, ...]) -> tuple[Any, Any]:
    """Return the mean and its marginal variance across complete randomisations.

    ``means`` is a finite binary64 array of shape ``(R, ...)`` in the measurement
    units and ``R >= 2``. Every batch must be a complete randomised design with
    the same law, budgets and history count and a distinct seed. Shards cannot
    serve as replicates. The outputs have shape ``means.shape[1:]`` and units of
    measurement and squared measurement, respectively, using
    ``M2 / (R * (R - 1))`` for the variance of the replicate-average estimator.
    This host operation explicitly accepts downloaded replicate means.
    """
    import numpy as np

    from .rng import require_independent

    if len(batches) < 2 or any(
        not isinstance(cast(Any, batch), HistoryBatch) or batch.sampling is None
        for batch in batches
    ):
        raise TransportError("replicate moments require at least two complete randomised designs")
    identities = [batch.sampling for batch in batches]
    first = identities[0]
    assert first is not None
    for batch, identity in zip(batches, identities, strict=True):
        assert identity is not None
        if (batch.first_history, batch.count) != (identity.first_history, identity.count):
            raise TransportError("a history shard cannot serve as a complete design replicate")
        if (identity.design, identity.count) != (first.design, first.count):
            raise TransportError("replicates require matching full-path designs and counts")
    require_independent(*batches)
    values = np.asarray(means, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != len(batches) or not bool(np.all(np.isfinite(values))):
        raise TransportError(
            "replicate means require a finite array with one row per complete design"
        )
    scale = np.max(np.abs(values), axis=0)
    normalised = values / np.where(scale > 0, scale, 1.0)
    normalised_mean = np.mean(normalised, axis=0)
    mean = normalised_mean * scale
    root_variance = (
        np.sqrt(
            np.sum((normalised - normalised_mean) ** 2, axis=0)
            / (len(batches) * (len(batches) - 1))
        )
        * scale
    )
    variance = root_variance**2
    if not bool(np.all(np.isfinite(mean))) or not bool(np.all(np.isfinite(variance))):
        raise TransportError("replicate moments exceed their finite output range")
    return mean, variance


def pool_design_shard_means(means: Any, *, shards: tuple[HistoryShard, ...]) -> Any:
    """Combine a complete partition of one design without inventing uncertainty.

    ``means`` has finite shape ``(S, ...)`` with one local mean per history shard.
    Each local mean uses that shard's launched history count. The result has
    shape ``means.shape[1:]`` in the same measurement units and equals
    ``sum(shard.count * shard_mean) / original_count``. Shards must cover one
    complete structural design with no overlap or gap, including nested shards.
    This host operation accepts explicitly downloaded local means.
    """
    import numpy as np

    if not shards or any(not isinstance(cast(Any, shard), HistoryShard) for shard in shards):
        raise TransportError("design pooling requires structural history shards")
    original = shards[0].original_batch
    if original.sampling is None or any(shard.original_batch != original for shard in shards):
        raise TransportError("design shards must belong to one complete randomisation")
    ordered = sorted(shards, key=lambda shard: shard.offset)
    endpoint = 0
    for shard in ordered:
        if shard.offset != endpoint:
            raise TransportError("design shard partition has an overlap or gap")
        endpoint += shard.batch.count
    if endpoint != original.count:
        raise TransportError("design shard partition does not cover the complete original batch")
    values = np.asarray(means, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != len(shards) or not bool(np.all(np.isfinite(values))):
        raise TransportError("shard means require a finite array with one row per shard")
    scale = np.max(np.abs(values), axis=0)
    normalised = values / np.where(scale > 0, scale, 1.0)
    weights = np.asarray([shard.batch.count / original.count for shard in shards])
    mean = np.tensordot(weights, normalised, axes=(0, 0)) * scale
    if not bool(np.all(np.isfinite(mean))):
        raise TransportError("pooled design mean exceeds its finite output range")
    return mean
