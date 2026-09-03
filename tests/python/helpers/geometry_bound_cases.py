"""Explicit analytic bound-shell model shared by geometry admission controls."""

from __future__ import annotations

import hashlib

from dpt.materials import Provenance
from dpt.transport.bound_compton import FINE_STRUCTURE_CONSTANT, BoundComptonTable, ComptonShell
from dpt.transport.geometry_histories import GeometryPhysicalMedium
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.physical import AtomicScatteringTable, PhysicalTransport

FORMULA = (
    b"Caller-defined analytic Z=1 shell: occupancy=1, binding=U, "
    b"J(p)=h*(1+2*h*abs(p))*exp(-2*h*abs(p)-2*h*h*p*p), h=1"
)


PROVENANCE = Provenance(
    "https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/penelope_compton.html",
    hashlib.sha256(FORMULA).hexdigest(),
    "Caller-defined analytic controls; no provider data copied",
    FORMULA.decode(),
)


def model(
    *,
    binding=(1.0,),
    nodes=(0.1, 1.0, 3.0),
    scattering=None,
    absorption=None,
    rayleigh=None,
    fluorescence=(),
    cutoff=0.1,
    spectrum=None,
    cells=None,
    voxels=None,
    source_energy=3.0,
):
    materials, width = len(binding), len(nodes)
    cells = tuple(range(materials)) if cells is None else cells
    physical = PhysicalTransport(
        tuple(
            AtomicScatteringTable(1, (0.0, 1000.0), (1.0, 1.0), (0.0, 1.0), PROVENANCE)
            for _ in binding
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
            for u in binding
        ),
    )
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, len(cells)), materials),
        PlanarDetector((-4, -4), (1, 1), (12, 12), 4),
        source_energy,
        "Explicit analytic shell/rate support controls; mm^-1",
        nodes,
        scattering_law="bound-electron-compton",
        max_events=4,
    )
    medium = GeometryPhysicalMedium(
        cells,
        (1.0,) * (materials * width) if absorption is None else absorption,
        (1.0,) * (materials * width) if scattering is None else scattering,
        (1.0,) * materials,
        (1.0,) * len(cells) if voxels is None else voxels,
        physical,
    )
    return spec, medium
