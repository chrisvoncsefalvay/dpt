"""Actual reachable-law controls for whole-source radiative material planes."""

from dataclasses import replace

import pytest

from dpt.materials import Provenance
from dpt.transport.geometry_detector_proposals import prepare_detector_area_proposal
from dpt.transport.geometry_forward_proposal_arguments import forward_proposal_plan_arguments
from dpt.transport.geometry_histories import GeometryPhysicalMedium
from dpt.transport.geometry_tangents import GridGeometryTangent, RigidMotion
from dpt.transport.geometry_whole_source_admission import admit_whole_source_geometry
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.physical import AtomicScatteringTable, FluorescenceLine, PhysicalTransport
from dpt.transport.source import ParallelBeam
from dpt.transport.uncollided_geometry import AcquisitionGeometryTangent

PROVENANCE = Provenance(
    "exact one-fluorescence finite-panel reference",
    "3" * 64,
    "CC0",
    "Independent source-interface generator control",
)


def scene(*, fluorescence=True, far_patch=False, homogeneous=False):
    count = 4 if far_patch else 2
    atom = AtomicScatteringTable(1, (0.0, 10.0), (1.0, 1.0), (0.0, 1.0), PROVENANCE)
    line = FluorescenceLine(40.0, 50.0, (0.0, 0.5), PROVENANCE)
    physical = PhysicalTransport(
        (atom,) * count,
        (0.0,) * (count * 2),
        ((line,),) * count if fluorescence else (),
        cutoff_kev=1.0,
    )
    spec = TransportSpec(
        MaterialGrid((-1, 0, 0), (1, 1, 1), (1, 2 if far_patch else 1, 2), count),
        PlanarDetector((0, 0.25), (0.25, 0.5), (1, 1), 2),
        80,
        "Exact finite one-fluorescence panel",
        (40, 80),
    )
    rates = (2, 0.7, 1, 0.7) if not homogeneous else (1, 0.7, 1, 0.7)
    if far_patch:
        rates = (1, 0.7, 1, 0.7, 2, 0.7, 1, 0.7)
    medium = GeometryPhysicalMedium(
        tuple(range(count)),
        rates,
        (0.0,) * (count * 2),
        (1.0,) * count,
        (1.0,) * count,
        physical,
    )
    source = ParallelBeam((0, 0.5, -1), (0, 0))
    tangent = AcquisitionGeometryTangent(object=GridGeometryTangent(motion=RigidMotion((1, 0, 0))))
    return source, spec, medium, tangent


def test_actual_fluorescent_future_jump_fails_chart_before_draws():
    source, spec, medium, tangent = scene()
    # Independent finite-panel proof gives right change -C*t*log(1/t), left0.
    with pytest.raises(TransportError, match="radiative whole-source detector chart"):
        admit_whole_source_geometry(source, spec, medium, tangent)


def test_reachable_jump_in_another_patch_is_not_hidden_by_source_column():
    source, spec, medium, tangent = scene(far_patch=True)
    with pytest.raises(TransportError, match="radiative whole-source detector chart"):
        admit_whole_source_geometry(source, spec, medium, tangent)


def test_fully_homogeneous_continuation_plane_is_redundant():
    source, spec, medium, tangent = scene(homogeneous=True)
    result = admit_whole_source_geometry(source, spec, medium, tangent)
    assert result.redundant_detector_planes == frozenset({(0, 1)})
    assert result.source.auxiliary_cells == (0, 1)
    assert result.physical.auxiliary_marks == ((0, 80), (1, 80))
    assert {mark.energy_kev for mark in result.physical.marks} == {40, 80}


def test_unused_lower_energy_difference_is_ignored_for_absorbed_source():
    source, spec, medium, tangent = scene(fluorescence=False)
    result = admit_whole_source_geometry(source, spec, medium, tangent)
    assert result.source.equivalent_faces == frozenset({(0, 1)})
    assert not result.redundant_detector_planes
    assert {mark.energy_kev for mark in result.physical.marks} == {80}


def test_unused_lower_energy_difference_is_ignored_for_elastic_source():
    source, spec, medium, tangent = scene(fluorescence=False)
    medium = replace(medium, scattering_mm_inverse=(0.0, 0.3, 0.0, 0.3))
    result = admit_whole_source_geometry(source, spec, medium, tangent)
    assert result.redundant_detector_planes == frozenset({(0, 1)})
    assert {mark.energy_kev for mark in result.physical.marks} == {80}
    assert not result.physical.intervals


def test_stationary_common_motion_keeps_homogeneous_radiative_chart():
    source, spec, medium, tangent = scene(homogeneous=True)
    tangent = replace(tangent, source=replace(tangent.source, motion=tangent.object.motion))
    result = admit_whole_source_geometry(source, spec, medium, tangent)
    assert not result.source.equivalent_faces
    assert result.redundant_detector_planes == frozenset({(0, 1)})


def test_invisible_source_trace_does_not_activate_radiative_laws():
    source, spec, medium, tangent = scene()
    source = replace(source, lower_mm=(0, 10, -1))
    result = admit_whole_source_geometry(source, spec, medium, tangent)
    assert not result.physical.marks
    assert not result.redundant_detector_planes


def test_initial_subcutoff_opacity_has_no_radiative_proposal():
    source, spec, medium, tangent = scene()
    assert medium.physical is not None
    medium = replace(medium, physical=replace(medium.physical, cutoff_kev=50))
    result = admit_whole_source_geometry(source, spec, medium, tangent, emitted_energy_kev=40)
    assert not result.physical.marks
    assert not result.redundant_detector_planes


def test_certified_redundant_plane_removes_zero_separation_component():
    source, spec, medium, tangent = scene(homogeneous=True)
    result = admit_whole_source_geometry(source, spec, medium, tangent)
    with pytest.raises(TransportError, match="grid-plane boundary"):
        prepare_detector_area_proposal((0, 0.5, 0.5), spec)
    proposal = prepare_detector_area_proposal(
        (0, 0.5, 0.5), spec, redundant_planes=result.redundant_detector_planes
    )
    assert all((plane.axis, plane.face) != (0, 1) for plane in proposal.planes)
    assert proposal.log_density((0.3, 0.7)) > -1
    full, _ = forward_proposal_plan_arguments(spec)
    filtered, _ = forward_proposal_plan_arguments(
        spec, redundant_planes=result.redundant_detector_planes
    )
    assert len(filtered) == len(full) - 1


@pytest.mark.parametrize(
    "planes", [frozenset({(0, 3)}), frozenset({(3, 0)}), frozenset({(True, 1)})]
)
def test_invalid_redundant_plane_labels_fail_before_proposal_draws(planes):
    _, spec, _, _ = scene(homogeneous=True)
    with pytest.raises(TransportError, match="valid axis/face"):
        prepare_detector_area_proposal((0, 0.5, 0.5), spec, redundant_planes=planes)
