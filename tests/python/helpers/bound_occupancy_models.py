"""Explicit shell tables shared by original-law occupancy controls."""

from __future__ import annotations

import hashlib

from bound_occupancy_reference import Shells
from dpt.materials import Provenance
from dpt.transport.bound_compton import FINE_STRUCTURE_CONSTANT, BoundComptonTable, ComptonShell


def occupancy_table(shells: Shells) -> BoundComptonTable:
    definition = b"Original-law tiny-occupancy conditioning regression"
    return BoundComptonTable(
        1,
        tuple(
            ComptonShell(str(index), occupancy, binding, height * FINE_STRUCTURE_CONSTANT)
            for index, (occupancy, binding, height) in enumerate(shells)
        ),
        Provenance(
            definition.decode(),
            hashlib.sha256(definition).hexdigest(),
            "Apache-2.0",
            "Synthetic fixture; occupancies and momentum profiles are explicit",
        ),
    )


def physical_shells(table: BoundComptonTable) -> Shells:
    return tuple((s.occupancy, s.binding_kev, s.momentum_height) for s in table.shells)
