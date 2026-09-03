"""Shared exact scientific helpers for the native controls."""

# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
import importlib
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest

from dpt.transport.forward import prepare_transport
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.physical import (
    PhysicalTransport,
    xraylib_atomic_scattering,
    xraylib_fluorescence_lines,
)
from dpt.transport.sampling import _philox_words

wp: Any = importlib.import_module("warp")

SEED = 0x123456789ABCDEF0


@pytest.fixture(scope="module")
def actual_tables():
    xraylib = importlib.import_module("xraylib")
    if xraylib.__version__ != "4.3.0":
        raise RuntimeError("native physical geometry controls require xraylib4.3.0")
    from dpt.transport.bound_compton import xraylib_bound_compton

    energies = (0.1, 10.0, 80.0, 200.0)
    conversion = 2.6989 / 10.0
    absorption = tuple(xraylib.CS_Photo(13, energy) * conversion for energy in energies)
    rayleigh = tuple(xraylib.CS_Rayl(13, energy) * conversion for energy in energies)
    scattering = tuple(
        (xraylib.CS_Rayl(13, energy) + xraylib.CS_Compt(13, energy)) * conversion
        for energy in energies
    )
    physical = PhysicalTransport(
        (xraylib_atomic_scattering(13, (0, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 20)),),
        rayleigh,
        fluorescence=(xraylib_fluorescence_lines(13, energies, lines=("KL2_LINE", "KL3_LINE")),),
        cutoff_kev=0.1,
    )
    return energies, absorption, scattering, physical, xraylib_bound_compton(13)


def _workspace(
    device,
    tables,
    law: Literal["isotropic-elastic", "free-electron-compton", "bound-electron-compton"] = (
        "free-electron-compton"
    ),
    *,
    packed=False,
    trials=256,
):
    energies, absorption, scattering, physical, shells = tables
    if law == "bound-electron-compton":
        physical = replace(physical, bound_compton=(shells,))
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 1), 1),
        PlanarDetector((-2, -2), (1, 1), (4, 4), 3),
        80.0,
        "xraylib4.3.0 Al, density2.6989g/cm3, cm^-1 converted to mm^-1",
        energies,
        scattering_law=law,
        max_angle_trials=trials,
        max_events=4,
        compton_sampler="packed-uniform-rejection" if packed else "legacy-rejection",
    )
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=device),
        absorption=wp.array(absorption, dtype=wp.float64, device=device),
        scattering=wp.array(scattering, dtype=wp.float64, device=device),
        max_histories=64,
        device=device,
        physical=physical,
    )
    return workspace


def _selected_histories(workspace):
    """Select actual Philox addresses by their channel marks before angular draws."""
    histories = np.arange(2**32 + 1, 2**32 + 50001, dtype=np.uint64)
    offset = 2 * 65536
    draws = (_philox_words(SEED, histories, 1, offset).astype(np.float64) + 0.5) / 2**32
    coherent = (_philox_words(SEED, histories, 1, offset + 3).astype(np.float64) + 0.5) / 2**32
    radiative = (_philox_words(SEED, histories, 1, offset + 2).astype(np.float64) + 0.5) / 2**32
    a, s = workspace.absorption.numpy()[1], workspace.scattering.numpy()[1]
    r = workspace.physical.rayleigh_mm_inverse[1]
    line_probability = sum(line.probabilities[1] for line in workspace.physical.fluorescence[0])
    photo = draws[:, 1] < a / (a + s)
    masks = (
        photo & (radiative[:, 0] >= line_probability),
        photo & (radiative[:, 0] < line_probability),
        ~photo & (coherent[:, 0] < r / s),
        ~photo & (coherent[:, 0] >= r / s),
    )
    selected = [histories[mask][:4] for mask in masks]
    assert all(len(group) == 4 for group in selected)
    return np.concatenate(selected)
