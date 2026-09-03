"""Full-field expected-signal VJPs for positive active log-voxel densities.

For voxel v, each original history contributes c[p] X[p] (K[v] - tau[v]).
Analogue K counts all collisions; continuous absorption K counts scatterings
and tau combines scattering likelihood and explicit absorption depths. The
canonical transport walk records every realised segment, including escape.
Repeated visits are coalesced before source, cotangent and attenuation factors.

The field and cell geometry are fixed for the prepared transport lifetime.
Active density multipliers must be positive. Fixed inactive vacuum is legal;
no likelihood score is supplied at an active zero-density support boundary.
"""

from __future__ import annotations

# These internals deliberately share canonical physical traversal and ownership.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from dpt._runtime import StatusReadback, load_kernels, require_no_tape
from dpt.contracts import NumericalError

from .forward import TransportWorkspace, _validate_call
from .model import IncompleteHistoryError, TransportError, TransportSpec
from .rng import HistoryBatch, require_independent


@dataclass(frozen=True, slots=True)
class VoxelDensityVJPResult:
    """Resident output and immutable estimator identity; caller owns output storage."""

    gradient: Any
    variance_of_mean: Any
    grid_shape: tuple[int, int, int]
    batch: HistoryBatch
    cotangent_batch: HistoryBatch | None
    transport_spec: TransportSpec
    parameterization: Literal["log-voxel-density"] = "log-voxel-density"
    variance_law: str | None = None
    checked: bool = True


@dataclass(slots=True)
class VoxelDensityVJPWorkspace:
    """Bounded per-history sparse hash slots, plus dense V-entry reduction scratch.

    Storage is O(max_histories * slots_per_history + voxels), without an N-by-V
    tape. Slots limit distinct active visited cells, rather than scattering
    events or repeated visits. Exhaustion invalidates the complete estimate.
    Calls reuse all buffers and the transport's owning stream. Active masks and
    transport model arrays remain immutable until this workspace is discarded.
    """

    transport: TransportWorkspace
    slots_per_history: int
    active: Any
    _owns_active: bool
    _arrays: dict[str, Any] = field(repr=False)
    _kernels: Any = field(repr=False)
    _record_status: Any = field(repr=False)
    _record_readback: StatusReadback = field(repr=False)

    @property
    def scratch_bytes(self) -> int:
        wp = self.transport.context.wp
        return (
            sum(
                int(value.size) * wp.types.type_size_in_bytes(value.dtype)
                for value in self._arrays.values()
            )
            + 4
            + (4 * int(self.active.size) if self._owns_active else 0)
        )

    @property
    def host_bytes(self) -> int:
        return self._record_readback.host_bytes

    def check_status(self) -> None:
        self.transport.check_status()
        code = self._record_readback.read()
        if code & 1:
            raise IncompleteHistoryError(
                "distinct active visited voxels exceeded slots_per_history; "
                "discard the VJP estimate"
            )
        if code & 2:
            raise NumericalError("nonfinite complete-history voxel score; discard the VJP estimate")

    def clear_status(self) -> None:
        """Discard failed outputs before explicitly beginning an independent call."""
        self.transport.clear_status()
        with self.transport.context.scope():
            self._record_status.zero_()


def prepare_voxel_density_vjp(
    workspace: TransportWorkspace,
    *,
    slots_per_history: int = 64,
    active: Any = None,
    max_bytes: int = 256 * 1024 * 1024,
) -> VoxelDensityVJPWorkspace:
    """Bind an opt-in sparse full-field log-density VJP and explicit memory budget.

    ``active`` is a contiguous int32 V-entry mask containing only zero or one;
    omitting it activates every voxel. Selected voxel multipliers must be finite
    and positive. An omitted forward voxel field means unit multipliers.
    ``slots_per_history`` is a positive power of two, chosen for the scene's
    distinct active visited cells. It does not silently truncate long paths.
    """
    require_no_tape()
    if (
        type(slots_per_history) is not int
        or slots_per_history <= 0
        or slots_per_history & (slots_per_history - 1)
    ):
        raise TransportError("slots_per_history must be a positive power of two")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise TransportError("voxel VJP max_bytes must be a positive integer")
    voxels = math.prod(workspace.spec.grid.shape)
    records = workspace.max_histories * slots_per_history
    if records >= 2**31:
        raise TransportError("sparse voxel VJP storage exceeds signed 32-bit indexing")
    context, owns_active = workspace.context, active is None
    wp = context.wp
    required = 20 * records + 28 * voxels + 4 + (4 * voxels if owns_active else 0)
    if required > max_bytes:
        raise TransportError(
            f"voxel VJP needs {required} device bytes, exceeding preparation budget"
        )
    if active is not None:
        context.array(active, "active", dtype=wp.int32, shape=(voxels,))
    kernels = load_kernels("dpt.transport.voxel_kernels")
    workspace.check_status()
    with context.scope():
        if owns_active:
            active = wp.ones(voxels, dtype=wp.int32, device=context.device)
    workspace._launch(
        kernels.validate_active, voxels, [active, workspace.voxel_density, workspace._status]
    )
    workspace.check_status()
    with context.scope():
        arrays = {
            "keys": wp.empty(records, dtype=wp.int32, device=context.device),
            "scores": wp.empty(records, dtype=wp.float64, device=context.device),
            "values": wp.empty(records, dtype=wp.float64, device=context.device),
            "scale": wp.empty(voxels, dtype=wp.float64, device=context.device),
            "sums": wp.empty(voxels, dtype=wp.float64, device=context.device),
            "hits": wp.empty(voxels, dtype=wp.int32, device=context.device),
            "centred": wp.empty(voxels, dtype=wp.float64, device=context.device),
        }
        record_status = wp.zeros(1, dtype=wp.int32, device=context.device)
    result = VoxelDensityVJPWorkspace(
        workspace,
        slots_per_history,
        active,
        owns_active,
        arrays,
        kernels,
        record_status,
        StatusReadback(context, record_status),
    )
    assert result.scratch_bytes == required
    return result


def voxel_density_vjp(
    positions: Any,
    directions: Any,
    weights: Any,
    density: Any,
    detector_cotangent: Any,
    *,
    batch: HistoryBatch,
    workspace: VoxelDensityVJPWorkspace,
    out_gradient: Any,
    out_variance_of_mean: Any = None,
    variance_law: Literal["iid-original-histories"] | None = None,
    cotangent_batch: HistoryBatch | None = None,
    source_amplitude: float = 1.0,
    stream: Any = None,
    validate: bool = True,
) -> VoxelDensityVJPResult:
    """Grad[v] = sum_i c[p_i]*X_i*(K_i,v-tau_i,v)/N differentiates log voxel multipliers.

    Positions/directions are vec3d (N,) in mm/unit direction, weights is FP64
    (N,), density is positive FP64 (materials,) and detector_cotangent is FP64
    (pixels,). The fixed voxel_density field is dimensionless FP64 (voxels,),
    with positive selected multipliers and optional zero inactive vacuum cells.
    Out_gradient and optional out_variance_of_mean are disjoint FP64 (voxels,),
    flattened from (nz,ny,nx) with x fastest. Inactive entries are zero. Each
    history combines repeated visits before tallying its derivative, including
    all analogue interactions or continuous-absorption scattering/weight terms.
    The gradient has score-covector units per log-density and variance has their
    square. The denominator includes every original history. Optional marginal
    variance requires N >= 2 and variance_law="iid-original-histories"; complete
    stratified/Sobol designs require independent whole-design replicates instead.
    A stochastic covector must declare an independent cotangent_batch and the
    reported variance is conditional on that covector. Distinct-cell slots and
    transport resource budgets must accommodate complete paths. The returned
    VoxelDensityVJPResult references caller-owned outputs and checked status.

    """
    require_no_tape()
    transport, arrays = workspace.transport, workspace._arrays
    context, wp = transport.context, transport.context.wp
    if cotangent_batch is not None:
        require_independent(batch, cotangent_batch)
    if out_variance_of_mean is not None:
        if batch.sampling is not None:
            raise TransportError("dependent path designs require complete-batch replicate variance")
        if batch.count < 2 or variance_law != "iid-original-histories":
            raise TransportError(
                "variance requires at least two IID original histories and "
                "variance_law='iid-original-histories'"
            )
    elif variance_law is not None:
        raise TransportError("variance_law requires an out_variance_of_mean buffer")
    voxels = math.prod(transport.spec.grid.shape)
    context.array(
        detector_cotangent,
        "detector_cotangent",
        dtype=wp.float64,
        shape=(transport.spec.detector.pixels,),
    )
    context.array(out_gradient, "out_gradient", dtype=wp.float64, shape=(voxels,))
    writes = [("out_gradient", out_gradient)]
    if out_variance_of_mean is not None:
        context.array(
            out_variance_of_mean, "out_variance_of_mean", dtype=wp.float64, shape=(voxels,)
        )
        writes.append(("out_variance_of_mean", out_variance_of_mean))
    if validate:
        workspace.check_status()
    _validate_call(
        transport,
        batch,
        positions,
        directions,
        weights,
        density,
        [],
        source_amplitude,
        stream,
        validate,
    )
    context.disjoint(
        [
            *transport._reads(),
            ("positions", positions),
            ("directions", directions),
            ("weights", weights),
            ("density", density),
            ("detector_cotangent", detector_cotangent),
            ("active", workspace.active),
        ],
        [*writes, *arrays.items(), ("record_status", workspace._record_status)],
    )
    if validate:
        transport._launch(
            workspace._kernels.validate_cotangent,
            transport.spec.detector.pixels,
            [detector_cotangent, transport._status],
        )
        transport.check_status()
    records = batch.count * workspace.slots_per_history
    with context.scope():
        arrays["keys"][:records].fill_(-1)
        for name in ("scores", "values"):
            arrays[name][:records].zero_()
        for name in ("scale", "sums", "hits", "centred"):
            arrays[name].zero_()
    transport._launch(
        workspace._kernels.replay_scores,
        batch.count,
        [
            positions,
            directions,
            weights,
            density,
            *transport._model_inputs(),
            wp.uint64(batch.seed),
            wp.uint64(batch.first_history),
            source_amplitude,
            detector_cotangent,
            workspace.active,
            workspace.slots_per_history,
            arrays["keys"],
            arrays["scores"],
            arrays["values"],
            arrays["scale"],
            workspace._record_status,
            transport._status,
        ],
    )
    transport._launch(
        workspace._kernels.sum_coalesced,
        records,
        [
            arrays["keys"],
            arrays["values"],
            arrays["scale"],
            arrays["sums"],
            arrays["hits"],
            transport._status,
        ],
    )
    transport._launch(
        transport._kernels.finish_mean,
        voxels,
        [arrays["sums"], arrays["scale"], batch.count, out_gradient, transport._status],
    )
    if out_variance_of_mean is not None:
        transport._launch(
            workspace._kernels.centred_coalesced,
            records,
            [
                arrays["keys"],
                arrays["values"],
                arrays["scale"],
                out_gradient,
                arrays["centred"],
                transport._status,
            ],
        )
        transport._launch(
            workspace._kernels.finish_variance,
            voxels,
            [
                out_gradient,
                arrays["scale"],
                arrays["hits"],
                arrays["centred"],
                batch.count,
                out_variance_of_mean,
                transport._status,
            ],
        )
    if validate:
        workspace.check_status()
    return VoxelDensityVJPResult(
        out_gradient,
        out_variance_of_mean,
        transport.spec.grid.shape,
        batch,
        cotangent_batch,
        transport.spec,
        variance_law="iid-original-histories-conditional-on-cotangent"
        if out_variance_of_mean is not None
        else None,
        checked=validate,
    )
