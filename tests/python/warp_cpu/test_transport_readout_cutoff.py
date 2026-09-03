"""Detector connections obey the canonical local-deposition cutoff."""

from __future__ import annotations

import importlib
import math
from typing import Any, Literal

import numpy as np
import pytest

from dpt.transport.forward import TransportWorkspace, prepare_transport, trace_histories
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.physical import (
    PhysicalTransport,
    xraylib_atomic_scattering,
    xraylib_fluorescence_lines,
)
from dpt.transport.readout import TransportReadout, prepare_transport_readout, trace_readout
from dpt.transport.rng import HistoryBatch

wp: Any = importlib.import_module("warp")
Score = Literal["photon-count", "energy-kev"]


def _slab(
    device: str,
    *,
    energy: float,
    cutoff: float,
    scoring: Score,
    count: int = 8,
    miss: bool = False,
    transparent: bool = False,
    fluorescence: bool = False,
    line_below_table: bool = False,
) -> tuple[TransportWorkspace, tuple[Any, ...]]:
    provider = pytest.importorskip("xraylib")
    line_energy = float(provider.LineEnergy(29, provider.KL3_LINE))
    nodes = (20.0 if line_below_table else line_energy, 80.0) if fluorescence else (1.0, 80.0)
    lines = xraylib_fluorescence_lines(29, nodes, lines=("KL3_LINE",)) if fluorescence else ()
    # Actual xraylib copper photoelectric rates, converted from cm^2/g to
    # mm^-1 at 8.96 g/cm^3. The transparent control sets the physical rate zero.
    absorption = tuple(
        0.0 if transparent else float(provider.CS_Photo(29, node)) * 8.96 / 10 for node in nodes
    )
    depth = 10.0 if fluorescence else 1.0
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, depth), (1, 1, 1), 1),
        PlanarDetector((-5.0, -5.0), (10.0, 10.0), (1, 1), depth + 1.0),
        energy,
        "xraylib 4.3.0 copper photoelectric slab, 8.96 g/cm^3",
        coefficient_energies_kev=nodes,
        scoring=scoring,
        max_events=128,
    )
    physical = PhysicalTransport(
        (xraylib_atomic_scattering(29, (0.0, 0.1, 1.0, 10.0)),),
        (0.0, 0.0),
        fluorescence=(lines,) if fluorescence else (),
        cutoff_kev=cutoff,
    )
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=device),
        absorption=wp.array(absorption, dtype=wp.float64, device=device),
        scattering=wp.zeros(2, dtype=wp.float64, device=device),
        physical=physical,
        max_histories=count,
        device=device,
    )
    inputs = (
        wp.array([(2.0 if miss else 0.0, 0.0, -1.0)] * count, dtype=wp.vec3d, device=device),
        wp.array([(0.0, 0.0, 1.0)] * count, dtype=wp.vec3d, device=device),
        wp.ones(count, dtype=wp.float64, device=device),
        wp.ones(1, dtype=wp.float64, device=device),
    )
    return workspace, inputs


def _direct_scores(
    workspace: TransportWorkspace, inputs: tuple[Any, ...], batch: HistoryBatch
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    device = workspace.context.device
    outputs = {
        "out_" + name: wp.empty(batch.count, dtype=dtype, device=device)
        for name, dtype in (
            ("pixel", wp.int32),
            ("score", wp.float64),
            ("energy", wp.float64),
            ("events", wp.int32),
            ("status", wp.int32),
        )
    }
    trace_histories(*inputs, batch=batch, workspace=workspace, **outputs)
    return outputs["out_score"].numpy(), outputs["out_status"].numpy()


def _readout_mean(
    workspace: TransportWorkspace,
    inputs: tuple[Any, ...],
    batch: HistoryBatch,
    *,
    next_event: bool,
    mode: Literal["total", "scatter"] = "total",
) -> float:
    readout = prepare_transport_readout(
        workspace,
        spec=TransportReadout(mode=mode, next_event=next_event, tally_strategy="deterministic"),
    )
    mean = wp.empty(1, dtype=wp.float64, device=workspace.context.device)
    trace_readout(*inputs, batch=batch, workspace=readout, out_mean=mean)
    return float(mean.numpy()[0])


@pytest.mark.parametrize("energy", (40.0, 80.0))
@pytest.mark.parametrize("transparent", (False, True))
@pytest.mark.parametrize("scoring", ("photon-count", "energy-kev"))
def test_entered_cutoff_photon_has_no_primary_connection(
    energy: float, transparent: bool, scoring: Score, warp_device: str
) -> None:
    workspace, inputs = _slab(
        warp_device, energy=energy, cutoff=80.0, scoring=scoring, transparent=transparent
    )
    batch = HistoryBatch(981729, 0, 8)
    direct, status = _direct_scores(workspace, inputs, batch)
    # Local deposition applies before any coefficient lookup or event draw,
    # including an exact-zero rate cell. The detector contribution is zero.
    np.testing.assert_array_equal(direct, 0.0)
    np.testing.assert_array_equal(status, 1)
    assert _readout_mean(workspace, inputs, batch, next_event=False) == 0.0
    assert _readout_mean(workspace, inputs, batch, next_event=True) == 0.0


@pytest.mark.parametrize("energy", (40.0, 80.0))
@pytest.mark.parametrize("scoring", ("photon-count", "energy-kev"))
def test_cutoff_photon_missing_material_retains_its_detector_connection(
    energy: float, scoring: Score, warp_device: str
) -> None:
    workspace, inputs = _slab(warp_device, energy=energy, cutoff=80.0, scoring=scoring, miss=True)
    batch = HistoryBatch(981730, 0, 8)
    direct, status = _direct_scores(workspace, inputs, batch)
    expected = energy if scoring == "energy-kev" else 1.0
    np.testing.assert_array_equal(direct, expected)
    np.testing.assert_array_equal(status, 0)
    assert _readout_mean(workspace, inputs, batch, next_event=True) == expected


@pytest.mark.parametrize("line_below_table", (False, True))
@pytest.mark.parametrize("cutoff_offset", (0.0, 0.5))
@pytest.mark.parametrize("scoring", ("photon-count", "energy-kev"))
def test_deposited_fluorescence_has_no_scatter_connection_or_table_lookup(
    line_below_table: bool, cutoff_offset: float, scoring: Score, warp_device: str
) -> None:
    provider = pytest.importorskip("xraylib")
    line_energy = float(provider.LineEnergy(29, provider.KL3_LINE))
    workspace, inputs = _slab(
        warp_device,
        energy=80.0,
        cutoff=line_energy + cutoff_offset,
        scoring=scoring,
        count=4096,
        fluorescence=True,
        line_below_table=line_below_table,
    )
    batch = HistoryBatch(981731, 0, 4096)
    # The only continuing radiative channel is this actual copper line. Every
    # emitted photon is deposited, so the scatter detector observable is zero.
    assert _readout_mean(workspace, inputs, batch, next_event=False, mode="scatter") == 0.0
    assert _readout_mean(workspace, inputs, batch, next_event=True, mode="scatter") == 0.0


@pytest.mark.parametrize("scoring", ("photon-count", "energy-kev"))
def test_fluorescence_above_cutoff_still_has_positive_detector_connections(
    scoring: Score, warp_device: str
) -> None:
    provider = pytest.importorskip("xraylib")
    line_energy = float(provider.LineEnergy(29, provider.KL3_LINE))
    workspace, inputs = _slab(
        warp_device,
        energy=80.0,
        cutoff=line_energy - 0.5,
        scoring=scoring,
        count=4096,
        fluorescence=True,
    )
    value = _readout_mean(
        workspace, inputs, HistoryBatch(981731, 0, 4096), next_event=True, mode="scatter"
    )
    assert math.isfinite(value) and value > 0.0
