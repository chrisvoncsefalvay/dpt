"""Original-history energy balance and cell-local deposition from production transport."""

from __future__ import annotations

import math
from typing import Any

from .forward import TransportWorkspace, _validate_call
from .model import TransportError
from .rng import HistoryBatch

# This module shares the checked launch boundary with the canonical raw tracer.
# pyright: reportPrivateUsage=false


def trace_energy_balance(
    positions: Any,
    directions: Any,
    weights: Any,
    density: Any,
    *,
    batch: HistoryBatch,
    workspace: TransportWorkspace,
    out_pixel: Any,
    out_score: Any,
    out_energy: Any,
    out_events: Any,
    out_status: Any,
    out_initial_energy: Any,
    out_deposited_energy: Any,
    out_spectrum_index: Any,
    out_local_deposition: Any,
    source_amplitude: float = 1.0,
    stream: Any = None,
    validate: bool = True,
) -> None:
    r"""Write ``E_initial[i] = E_final[i] + E_deposited[i]`` and weighted local deposition.

    Each completed physical history satisfies ``E_initial=E_final+E_deposited``
    in unweighted keV, including photons that escape without hitting the detector.
    Absorption and a declared cutoff leave zero final photon energy. The voxel
    tally is ``sum_h source_weight[h]*source_amplitude*E_deposited[h,v]``. Divide
    this sum by the original-history count to obtain mean local deposition.
    Rayleigh conserves photon energy and photoelectron, Compton recoil and
    untransported vacancy energy contribute locally under the declared model.

    Source inputs and the five ordinary sparse outputs follow ``trace_histories``.
    The extra ``out_initial_energy`` and ``out_deposited_energy`` are contiguous
    ``wp.float64`` arrays of shape ``(batch.count,)`` in unweighted keV.
    ``out_spectrum_index`` is a matching ``wp.int32`` array containing the initial
    source energy index or ``-1`` for the fixed-energy law. ``out_local_deposition``
    is ``wp.float64`` of shape ``(prod(grid.shape),)``, x contiguous, with keV
    multiplied by the source weight and amplitude. It is zeroed once per call.
    Every array resides on the workspace device and outputs must be disjoint
    from all inputs, prepared buffers and one another.

    The workspace must contain ``PhysicalTransport`` and use analogue transport.
    It replays the same production walker and random identity as ``trace_histories``.
    No electron or secondary cascade transport is inferred. The owning stream
    and buffer lifetime contract is unchanged, and checked completion invalidates
    every output if any history fails. This function has no Warp tape derivative
    and does not return derivatives of the local deposition field. Density and
    source laws retain the prepared physical assumptions.

    """
    if workspace.physical is None:
        raise TransportError("energy accounting requires explicit PhysicalTransport records")
    if workspace.spec.estimator != "analogue":
        raise TransportError("local energy accounting requires analogue transport")
    wp = workspace.context.wp
    writes = [
        ("out_pixel", out_pixel, wp.int32),
        ("out_score", out_score, wp.float64),
        ("out_energy", out_energy, wp.float64),
        ("out_events", out_events, wp.int32),
        ("out_status", out_status, wp.int32),
        ("out_initial_energy", out_initial_energy, wp.float64),
        ("out_deposited_energy", out_deposited_energy, wp.float64),
        ("out_spectrum_index", out_spectrum_index, wp.int32),
    ]
    _validate_call(
        workspace,
        batch,
        positions,
        directions,
        weights,
        density,
        writes,
        source_amplitude,
        stream,
        validate,
    )
    workspace.context.array(
        out_local_deposition,
        "out_local_deposition",
        dtype=wp.float64,
        shape=(math.prod(workspace.spec.grid.shape),),
    )
    workspace.context.disjoint(
        [
            *workspace._reads(),
            ("positions", positions),
            ("directions", directions),
            ("weights", weights),
            ("density", density),
        ],
        [(name, value) for name, value, _ in writes]
        + [("out_local_deposition", out_local_deposition)],
    )
    with workspace.context.scope():
        out_local_deposition.zero_()
    workspace._launch(
        workspace._kernels.trace_energy_balance,
        batch.count,
        [
            positions,
            directions,
            weights,
            density,
            *workspace._model_inputs(),
            wp.uint64(batch.seed),
            wp.uint64(batch.first_history),
            source_amplitude,
            *[value for _, value, _ in writes],
            out_local_deposition,
            workspace._status,
        ],
    )
    workspace._launch(
        workspace._kernels.validate_deposition,
        int(out_local_deposition.size),
        [out_local_deposition, workspace._status],
    )
    if validate:
        workspace.check_status()
