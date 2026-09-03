"""Read a supplied event list through calibrated charge sharing and pileup.

``read_shared_events`` accepts an actual calibration and incident event arrays.
It samples coincident charge fragments, groups their timestamps by destination
pixel and applies the declared rectangular readout without discarding weak
fragments before integration. See the photon-counting guide for units and laws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from dpt.contracts import ContractError
from dpt.photon_counting import (
    ChargeSharingIdentity,
    ChargeSharingSpec,
    RectangularPileupSpec,
    prepare_charge_sharing,
    prepare_rectangular_pileup,
    rectangular_pileup_readout,
    sample_charge_sharing,
)


@dataclass(frozen=True, slots=True)
class SharedEventReadout:
    """Host results from one supplied finite event observation.

    ``charge_accounting_kev`` is [N,4] and ``readout_accounting_kev`` is [4,P].
    Charge rows are accepted/below/overflow/escape. Final readout rows are
    accepted/below/overflow/blind. Intermediate threshold categories describe
    the cloud pulses and do not remove them before temporal integration.
    """

    bin_counts: Any
    pulse_times_s: Any
    pulse_energies_kev: Any
    pulse_counts: Any
    charge_accounting_kev: Any
    readout_accounting_kev: Any


def read_shared_events(
    charge_spec: ChargeSharingSpec,
    readout_spec: RectangularPileupSpec,
    event_pixels: Any,
    event_energy_bins: Any,
    event_times_s: Any,
    gate_s: Any,
    blind_s: Any,
    *,
    identity: ChargeSharingIdentity,
    device: str,
) -> SharedEventReadout:
    """Execute calibrated event readout from NumPy input arrays on CPU/CUDA.

    Incident pixel and energy-bin arrays are int32[N], times are float64[N] and
    gate/blind arrays are float64[P]. Preparation, the charge-fragment readback
    and timestamp grouping occur once for this finite observation. Every
    positive fragment inside the detector enters readout, including fragments
    below an individual comparator threshold. The temporal gate opens on any
    positive fragment and its summed energy determines the final bin.
    """
    import numpy as np

    from dpt._runtime import load_warp

    wp = load_warp()
    for name, values, dtype in (
        ("event_pixels", event_pixels, np.dtype("int32")),
        ("event_energy_bins", event_energy_bins, np.dtype("int32")),
        ("event_times_s", event_times_s, np.dtype("float64")),
        ("gate_s", gate_s, np.dtype("float64")),
        ("blind_s", blind_s, np.dtype("float64")),
    ):
        if (
            not isinstance(cast(object, values), np.ndarray)
            or values.dtype != dtype
            or values.ndim != 1
        ):
            raise ContractError(f"{name} must be a one-dimensional NumPy array with dtype {dtype}")
    count = len(event_pixels)
    if count == 0 or len(event_energy_bins) != count or len(event_times_s) != count:
        raise ContractError("incident event arrays must have equal non-zero lengths")
    if len(gate_s) != charge_spec.pixels or len(blind_s) != charge_spec.pixels:
        raise ContractError("gate and blind time arrays must have one entry per detector pixel")
    cloud = prepare_charge_sharing(charge_spec, max_events=count, device=device)
    shape = (count, charge_spec.fragments)
    destinations = wp.empty(shape, dtype=wp.int32, device=device)
    times = wp.empty(shape, dtype=wp.float64, device=device)
    energies = wp.empty(shape, dtype=wp.float64, device=device)
    categories = wp.empty(shape, dtype=wp.int32, device=device)
    patterns = wp.empty(count, dtype=wp.int32, device=device)
    charge_ledger = wp.empty((count, 4), dtype=wp.float64, device=device)
    sample_charge_sharing(
        wp.array(event_pixels, dtype=wp.int32, device=device),
        wp.array(event_energy_bins, dtype=wp.int32, device=device),
        wp.array(event_times_s, dtype=wp.float64, device=device),
        out_pixels=destinations,
        out_times_s=times,
        out_energies_kev=energies,
        out_bins=categories,
        out_patterns=patterns,
        out_accounting_kev=charge_ledger,
        identity=identity,
        workspace=cloud,
    )
    pixel_values, time_values, energy_values = (
        destinations.numpy().ravel(),
        times.numpy().ravel(),
        energies.numpy().ravel(),
    )
    valid = (pixel_values >= 0) & (energy_values > 0)
    pixel_values, time_values, energy_values = (
        pixel_values[valid],
        time_values[valid],
        energy_values[valid],
    )
    event_counts = np.bincount(pixel_values, minlength=charge_spec.pixels).astype(np.int32)
    capacity = max(1, int(event_counts.max()))
    grouped_times = np.zeros((charge_spec.pixels, capacity), dtype=np.float64)
    grouped_energy = np.zeros_like(grouped_times)
    for p in range(charge_spec.pixels):
        selection = np.flatnonzero(pixel_values == p)
        order = selection[np.argsort(time_values[selection], kind="stable")]
        grouped_times[p, : len(order)] = time_values[order]
        grouped_energy[p, : len(order)] = energy_values[order]
    readout = prepare_rectangular_pileup(
        readout_spec, max_pixels=charge_spec.pixels, max_events_per_pixel=capacity, device=device
    )
    output_times = wp.empty(grouped_times.shape, dtype=wp.float64, device=device)
    output_energy = wp.empty(grouped_times.shape, dtype=wp.float64, device=device)
    output_counts = wp.empty(charge_spec.pixels, dtype=wp.int32, device=device)
    bins = wp.empty((readout_spec.bins, charge_spec.pixels), dtype=wp.uint64, device=device)
    readout_ledger = wp.empty((4, charge_spec.pixels), dtype=wp.float64, device=device)
    rectangular_pileup_readout(
        wp.array(grouped_times, dtype=wp.float64, device=device),
        wp.array(grouped_energy, dtype=wp.float64, device=device),
        wp.array(event_counts, dtype=wp.int32, device=device),
        wp.array(gate_s, dtype=wp.float64, device=device),
        wp.array(blind_s, dtype=wp.float64, device=device),
        out_times_s=output_times,
        out_energies_kev=output_energy,
        out_pulse_counts=output_counts,
        out_bin_counts=bins,
        out_accounting_kev=readout_ledger,
        workspace=readout,
    )
    return SharedEventReadout(
        bins.numpy(),
        output_times.numpy(),
        output_energy.numpy(),
        output_counts.numpy(),
        charge_ledger.numpy(),
        readout_ledger.numpy(),
    )
