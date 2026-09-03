"""Explicit analytic physical-law support cases shared by geometry controls."""

from __future__ import annotations

import hashlib
from typing import Literal

from dpt.materials import Provenance
from dpt.transport.bound_compton import FINE_STRUCTURE_CONSTANT, BoundComptonTable, ComptonShell
from dpt.transport.geometry_histories import GeometryPhysicalMedium
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.physical import AtomicScatteringTable, FluorescenceLine, PhysicalTransport
from dpt.transport.source import ParallelBeam

Law = Literal["isotropic-elastic", "free-electron-compton", "bound-electron-compton"]


FORMULA = (
    b"Explicit analytic support controls: linear mm^-1 rates, constant Z=1 "
    b"Rayleigh F(q)=1, occupied impulse shell "
    b"J(p)=h*(1+2*h*abs(p))*exp(-2*h*abs(p)-2*h*h*p*p), h=1"
)


PROVENANCE = Provenance(
    "https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/compton.html",
    hashlib.sha256(FORMULA).hexdigest(),
    "Caller-defined analytical controls, without copied provider data",
    FORMULA.decode(),
)


def case(
    law: Law = "free-electron-compton",
    *,
    nodes=(0.1, 1.0, 3.0),
    energy=3.0,
    bindings=(1.0,),
    scattering=None,
    absorption=None,
    rayleigh=None,
    fluorescence=(),
    cutoff=None,
    spectrum=None,
    voxels=None,
):
    materials, width = len(bindings), len(nodes)
    cutoff = (0.1 if law == "bound-electron-compton" else 0.0) if cutoff is None else cutoff
    physical = None
    if (
        law == "bound-electron-compton"
        or cutoff > 0
        or rayleigh is not None
        or fluorescence
        or spectrum is not None
    ):
        physical = PhysicalTransport(
            tuple(
                AtomicScatteringTable(1, (0.0, 1000.0), (1.0, 1.0), (0.0, 1.0), PROVENANCE)
                for _ in bindings
            ),
            (0.0,) * (materials * width) if rayleigh is None else rayleigh,
            fluorescence=fluorescence,
            cutoff_kev=cutoff,
            spectrum=spectrum,
            bound_compton=tuple(
                BoundComptonTable(
                    1,
                    (ComptonShell("defined occupied shell", 1.0, u, FINE_STRUCTURE_CONSTANT),),
                    PROVENANCE,
                )
                for u in bindings
            )
            if law == "bound-electron-compton"
            else (),
        )
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, materials), materials),
        PlanarDetector((-2, -2), (1, 1), (8, 8), 4),
        energy,
        "Explicit analytic linear-coefficient support control, mm^-1",
        nodes,
        scattering_law=law,
        max_events=4,
    )
    medium = GeometryPhysicalMedium(
        tuple(range(materials)),
        (1.0,) * (materials * width) if absorption is None else absorption,
        (1.0,) * (materials * width) if scattering is None else scattering,
        (1.0,) * materials,
        (1.0,) * materials if voxels is None else voxels,
        physical,
    )
    return spec, medium


def pencil(x=0.25):
    return ParallelBeam((x, 0.25, -1.0), (0.0, 0.0))


def line(energy, threshold, probabilities):
    return FluorescenceLine(energy, threshold, probabilities, PROVENANCE)
