"""Shared slab, diagnostic buffers and pinned aluminium fixtures."""

# Runtime Warp arrays are intentional device records.
# pyright: reportUnknownMemberType=false
import importlib
from typing import Any

import pytest

from dpt.transport.bound_compton import xraylib_bound_compton
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.physical import (
    PhysicalTransport,
    xraylib_atomic_scattering,
    xraylib_fluorescence_lines,
)
from dpt.transport.source_spectrum import SourceSpectrum

wp: Any = importlib.import_module("warp")


def _spec():
    return TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-5.0, -5.0), (10.0, 10.0), (1, 1), 3.0),
        80.0,
        "independent public geometry slab",
        max_events=4,
        max_angle_trials=2,
    )


def _outputs(workspace):
    context = workspace.context
    count, parameters = workspace.batch.count, len(workspace.parameters)
    with context.scope():
        return {
            "out_values": wp.empty((count, parameters), dtype=wp.float64, device=context.device),
            "out_derivatives": wp.empty(
                (count, parameters), dtype=wp.float64, device=context.device
            ),
            "out_counts": wp.empty((count, parameters), dtype=wp.vec2i, device=context.device),
            "out_mean": wp.empty(parameters, dtype=wp.float64, device=context.device),
            "out_gradient": wp.empty(parameters, dtype=wp.float64, device=context.device),
            "out_second_moment": wp.empty(parameters, dtype=wp.float64, device=context.device),
        }


@pytest.fixture(scope="module")
def actual_aluminium():
    provider = importlib.import_module("xraylib")
    if provider.__version__ != "4.3.0":
        raise RuntimeError("public geometry controls require pinned xraylib4.3.0")
    energies = (0.1, 10.0, 80.0, 200.0)
    conversion = 2.6989 / 10.0
    absorption = tuple(provider.CS_Photo(13, energy) * conversion for energy in energies)
    coherent = tuple(provider.CS_Rayl(13, energy) * conversion for energy in energies)
    scattering = tuple(
        (provider.CS_Rayl(13, energy) + provider.CS_Compt(13, energy)) * conversion
        for energy in energies
    )
    physical = PhysicalTransport(
        (xraylib_atomic_scattering(13, (0, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 20)),),
        coherent,
        fluorescence=(xraylib_fluorescence_lines(13, energies, lines=("KL2_LINE", "KL3_LINE")),),
        cutoff_kev=0.1,
        spectrum=SourceSpectrum(
            (10.0, 80.0), (0.5, 0.5), "Declared equal-mass two-energy source control"
        ),
    )
    return energies, absorption, scattering, physical, xraylib_bound_compton(13)
