"""Reachable continuation laws for whole-source material-plane charts.

An initially quiet plane can still carry a radiative grazing singularity at a
later energy. This module certifies a sufficient chart: every represented
patch of a whole-source plane has the same measured generator on the actual
reachable continuing energy support. Such a plane is physically redundant in
the terminal detector proposal. Failure to certify that identity does not
assert that every particular readout lacks a derivative.
"""

from __future__ import annotations

# These cold admission components share exact, dependency-free source support.
# pyright: reportPrivateUsage=false
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING

from .geometry_bound_admission import _source_chart
from .geometry_grazing_admission import GrazingSourceAdmission, admit_source_grazing_chart
from .geometry_law_signatures import compare_geometry_generators
from .geometry_physical_admission import GeometryPhysicalAdmission, admit_geometry_physical_law
from .model import TransportError, TransportSpec
from .physical_geometry_events import continuing_radiative_collision
from .source import ParallelBeam, PointConeSource
from .uncollided_geometry import AcquisitionGeometryTangent

if TYPE_CHECKING:
    from .geometry_histories import GeometryPhysicalMedium


@dataclass(frozen=True, slots=True)
class WholeSourceGeometryAdmission:
    """A full-law certificate and physically redundant detector-plan planes."""

    source: GrazingSourceAdmission
    physical: GeometryPhysicalAdmission
    whole_faces: frozenset[tuple[int, int]]
    redundant_detector_planes: frozenset[tuple[int, int]]


def _whole_faces(
    source: ParallelBeam | PointConeSource, spec: TransportSpec
) -> frozenset[tuple[int, int]]:
    if not isinstance(source, ParallelBeam):
        return frozenset()
    origin, rows, _ = _source_chart(spec, source)
    faces: set[tuple[int, int]] = set()
    for axis, count in enumerate(spec.grid.shape_xyz):
        if rows[axis] != (Fraction(), Fraction(), Fraction()):
            continue
        for face in range(count + 1):
            position = spec.grid.origin_mm[axis] + face * spec.grid.spacing_mm[axis]
            if origin[axis] == Fraction(position):
                faces.add((axis, face))
                break
    return frozenset(faces)


def _plane_cells(spec: TransportSpec, axis: int, face: int):
    sizes = spec.grid.shape_xyz
    others = tuple(i for i in range(3) if i != axis)
    for a, b in product(range(sizes[others[0]]), range(sizes[others[1]])):
        indices = [0, 0, 0]
        indices[others[0]], indices[others[1]] = a, b
        cells: list[int | None] = []
        for index in (face - 1, face):
            indices[axis] = index
            cells.append(
                (indices[2] * sizes[1] + indices[1]) * sizes[0] + indices[0]
                if 0 <= index < sizes[axis]
                else None
            )
        yield cells[0], cells[1]


def admit_whole_source_geometry(
    source: ParallelBeam | PointConeSource,
    spec: TransportSpec,
    medium: GeometryPhysicalMedium,
    tangent: AcquisitionGeometryTangent,
    *,
    emitted_energy_kev: float | None = None,
) -> WholeSourceGeometryAdmission:
    """Certify source, auxiliary and radiative whole-plane support before draws.

    The ordinary source certificate covers source-flight cells and every
    physically emitted continuing atom/open interval. An admitted whole-source
    face additionally supplies a conservative closure enclosure of its opposite
    cells at the original emitted marks. These are explicit auxiliary
    declarations, never an invented
    all-cell source maximum. Their physical closure retains the original open
    energy endpoints and complete interaction laws.

    For a radiative source trace on a plane, every patch is compared over the
    retained reachable energy families. An identity permits that redundant
    plane's omission from the detached detector-area proposal. An uncertified
    identity raises a chart error; no displaced ray, one-sided derivative or
    clipping substitutes for the missing chart. Pure absorption requires no
    radiative detector-area measure and keeps the ordinary source gate.

    ``emitted_energy_kev`` identifies one explicit original host photon mark.
    Omitting it uses the actual declared spectrum/monochromatic source support,
    as required by prepared production geometry workspaces. Model arrays and
    conditional laws remain fixed; no provider or native runtime is imported.
    """
    scope_spec, scope_medium = spec, medium
    if emitted_energy_kev is not None:
        scope_spec = replace(spec, energy_kev=emitted_energy_kev)
        # This scalar reference call already owns the selected original mark;
        # the source-spectrum sampler is not replayed inside a continuation.
        if medium.physical is not None and medium.physical.spectrum is not None:
            scope_medium = replace(medium, physical=replace(medium.physical, spectrum=None))
    grazing = admit_source_grazing_chart(
        source,
        scope_spec,
        scope_medium,
        tangent,
        emitted_energies_kev=None if emitted_energy_kev is None else (emitted_energy_kev,),
    )
    auxiliary = tuple(
        (cell, energy)
        for cell in grazing.auxiliary_cells
        for energy in grazing.initial_energies_kev
    )
    physical = admit_geometry_physical_law(
        scope_spec, scope_medium, source, auxiliary_marks=auxiliary
    )
    whole = _whole_faces(source, spec)
    source_cells = set(physical.source_cells).union(grazing.auxiliary_cells)
    cutoff = 0.0 if medium.physical is None else medium.physical.cutoff_kev
    initial_extinction = {
        energy: medium.extinction(energy, scope_spec)
        for energy in grazing.initial_energies_kev
        if energy > cutoff and source_cells
    }
    radiative = any(
        energy in initial_extinction
        and initial_extinction[energy][cell] > 0
        and continuing_radiative_collision(
            energy,
            medium.material_ids[cell],
            spec,
            medium.absorption_mm_inverse,
            medium.scattering_mm_inverse,
            medium.physical,
        )
        for cell in source_cells
        for energy in grazing.initial_energies_kev
    )
    redundant: set[tuple[int, int]] = set()
    if radiative:
        for axis, face in whole:
            for left, right in _plane_cells(spec, axis, face):
                result = compare_geometry_generators(
                    spec, medium, left, right, physical.marks, physical.intervals
                )
                if not result.equivalent:
                    raise TransportError(
                        "radiative whole-source detector chart is not certified: "
                        f"material plane {(axis, face)} has different reachable "
                        "continuation-law signatures"
                    )
            redundant.add((axis, face))
    return WholeSourceGeometryAdmission(grazing, physical, whole, frozenset(redundant))


__all__ = ["WholeSourceGeometryAdmission", "admit_whole_source_geometry"]
