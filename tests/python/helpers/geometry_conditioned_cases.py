"""Addressed conditioned bound-law cases shared by host and native controls."""

import hashlib

from dpt.materials import Provenance
from dpt.transport.bound_compton import BoundComptonTable, ComptonShell
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.physical import AtomicScatteringTable, PhysicalTransport

CASES = (
    ("ordinary", 80.0, ((1.0, 1.0, 0.1),)),
    ("tiny-profile", 80.0, ((1.0, 0.0, 1e-30),)),
    ("narrow-profile", 100.0, ((1.0, 1.0, 1e6),)),
    ("negative-tail", 1.01, ((1.0, 1.0, 1e6),)),
    ("tiny-occupancy", 80.0, ((1e-320, 1.0, 0.1), (1.0, 1000.0, 0.1))),
    ("mixed-tail-occupancy", 1.01, ((1.0, 1.0, 1e6), (1e-320, 0.0, 0.1))),
)


TERMINAL_CASES = (
    *CASES[:3],
    ("surviving-negative-tail", 1.2, ((1.0, 1.0, 7.2973525693),)),
    ("representable-negative-tail", 100.0, ((1.0, 30.0, 7.2973525693),)),
    CASES[4],
    ("mixed-tail-occupancy", 1.2, ((1.0, 1.0, 1e6), (1e-320, 0.0, 0.1))),
)


def physical_case(name, energy, entries, *, trials=3):
    provenance = Provenance(
        "https://doi.org/10.1016/0168-9002(96)00652-3",
        hashlib.sha256(repr((name, entries)).encode()).hexdigest(),
        "CC0-1.0 analytic control",
        "Explicit conditioned impulse-law collision integration control",
    )
    table = BoundComptonTable(
        1,
        tuple(
            ComptonShell(str(i), occupancy, binding, height)
            for i, (occupancy, binding, height) in enumerate(entries)
        ),
        provenance,
    )
    atomic = AtomicScatteringTable(1, (0, 1000), (1, 0), (0, 1), provenance)
    physical = PhysicalTransport((atomic,), (0, 0), cutoff_kev=0.1, bound_compton=(table,))
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 1), 1),
        PlanarDetector((-5, -5), (10, 10), (1, 1), 3),
        energy,
        "addressed conditioned bound geometry law",
        (0.1, energy),
        scattering_law="bound-electron-compton",
        max_angle_trials=trials,
        max_events=4,
    )
    return spec, physical, table
