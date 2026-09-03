"""Independent transverse startup laws, exact overlap and active channel controls."""

import math
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest

from dpt.geometry import RigidTransform
from dpt.materials import Provenance
from dpt.transport.bound_compton import BoundComptonTable, ComptonShell
from dpt.transport.geometry_histories import GeometryPhysicalMedium
from dpt.transport.geometry_startup_admission import admit_source_startup_chart
from dpt.transport.geometry_tangents import GridGeometryTangent, RigidMotion
from dpt.transport.model import (
    MaterialGrid,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from dpt.transport.physical import (
    AtomicScatteringTable,
    FluorescenceLine,
    PhysicalTransport,
)
from dpt.transport.source import ParallelBeam
from dpt.transport.source_spectrum import SourceSpectrum
from dpt.transport.uncollided_geometry import (
    AcquisitionGeometryTangent,
    SourceGeometryTangent,
)

P = Provenance(
    "analytic startup fixture",
    "0" * 64,
    "CC0-1.0",
    "explicit immutable conditional laws",
)
ACTIVE = AcquisitionGeometryTangent(source=SourceGeometryTangent(motion=RigidMotion((0, 0, 1))))


def _scene():
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (2, 1, 1), 2),
        PlanarDetector((-5, -5), (10, 10), (1, 1), 4),
        80,
        "analytic startup chart",
        (10, 80),
        max_events=4,
        max_angle_trials=2,
    )
    medium = GeometryPhysicalMedium((0, 1), (0.7, 0.7, 0.9, 0.9), (0, 0, 0, 0), (1, 1), (1, 1))
    source = ParallelBeam((0.1, 0.1, 1), (0.8, 0.8))
    return source, spec, medium


def _admit(source, spec, medium, tangent=ACTIVE, *, energies=None):
    medium.validate(spec)
    admit_source_startup_chart(source, spec, medium, tangent, emitted_energies_kev=energies)


@pytest.mark.parametrize("face", (0.0, 1.0, 2.0))
@pytest.mark.parametrize("direction", ((0, 0, 1), (0, 0, -1)))
def test_moving_full_source_plane_on_outer_or_internal_discontinuity_is_rejected(face, direction):
    source, spec, medium = _scene()
    source = replace(source, lower_mm=(0.1, 0.1, face), direction=direction)
    with pytest.raises(TransportError, match="discontinuous transverse"):
        _admit(source, spec, medium)


def test_unit_cube_boundary_has_two_distinct_analytic_one_sided_derivatives():
    source, spec, medium = _scene()
    source = replace(source, lower_mm=(0.1, 0.1, 0))
    spec = replace(spec, grid=MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 1), 1))
    medium = GeometryPhysicalMedium((0,), (0.7, 0.7), (0, 0), (1,), (1,))
    # T(t)=exp(-mu) for t<0, exp[-mu(1-t)] for 0<t<1.
    with localcontext() as context:
        context.prec = 80
        mu = Decimal.from_float(0.7)
        left = Decimal(0)
        right = mu * (-mu).exp()
        assert left != right
        assert float(right) == pytest.approx(0.7 * math.exp(-0.7), abs=1e-16)
    with pytest.raises(TransportError, match="discontinuous transverse"):
        _admit(source, spec, medium)


@pytest.mark.parametrize("velocity", ((0, 0, 0), (1, 0, 0)))
def test_stationary_relative_launch_plane_is_admitted(velocity):
    source, spec, medium = _scene()
    tangent = AcquisitionGeometryTangent(source=SourceGeometryTangent(motion=RigidMotion(velocity)))
    _admit(source, spec, medium, tangent)


def test_equal_source_and_object_motion_is_admitted():
    source, spec, medium = _scene()
    tangent = replace(ACTIVE, object=GridGeometryTangent(motion=ACTIVE.source.motion))
    _admit(source, spec, medium, tangent)


def test_internal_face_spacing_velocity_and_source_motion_cancel_exactly():
    source, spec, medium = _scene()
    tangent = replace(ACTIVE, object=GridGeometryTangent(spacing_dot=(0, 0, 1)))
    _admit(source, spec, medium, tangent)


def test_nonconstant_relative_normal_velocity_rejects_a_positive_area_kink():
    source, spec, medium = _scene()
    tangent = AcquisitionGeometryTangent(
        source=SourceGeometryTangent(motion=RigidMotion(angular=(1, 0, 0)))
    )
    with pytest.raises(TransportError, match="discontinuous transverse"):
        _admit(source, spec, medium, tangent)


@pytest.mark.parametrize("lower", ((2, 0.1, 1), (1, 0.1, 1)))
def test_zero_overlap_and_edge_only_overlap_are_admitted(lower):
    source, spec, medium = _scene()
    _admit(replace(source, lower_mm=lower), spec, medium)


def test_generic_source_inside_cell_is_admitted():
    source, spec, medium = _scene()
    _admit(replace(source, lower_mm=(0.1, 0.1, 0.25)), spec, medium)


@pytest.mark.parametrize("empty", ("zero_rates", "zero_voxel"))
def test_physically_empty_medium_is_admitted(empty):
    source, spec, medium = _scene()
    medium = (
        replace(medium, absorption_mm_inverse=(0, 0, 0, 0))
        if empty == "zero_rates"
        else replace(medium, voxel_density=(0, 0))
    )
    _admit(source, spec, medium)
    _admit(replace(source, lower_mm=(0.1, 0.1, 0)), spec, medium)


def test_distinct_material_ids_with_identical_effective_rates_are_admitted():
    source, spec, medium = _scene()
    medium = replace(medium, absorption_mm_inverse=(0.5, 0.5, 1, 1), density=(2, 1))
    _admit(source, spec, medium)


def test_discontinuous_patch_outside_source_footprint_does_not_reject():
    source, spec, medium = _scene()
    spec = replace(spec, grid=MaterialGrid((0, 0, 0), (1, 1, 1), (2, 1, 2), 2))
    medium = replace(medium, material_ids=(0, 0, 0, 1), voxel_density=(1, 1, 1, 1))
    _admit(replace(source, extent_xy_mm=(0.5, 0.8)), spec, medium)
    with pytest.raises(TransportError, match="discontinuous transverse"):
        _admit(
            replace(source, lower_mm=(1.1, 0.1, 1), extent_xy_mm=(0.5, 0.8)),
            spec,
            medium,
        )


def test_rotated_footprint_uses_polygon_not_bounding_box_overlap():
    source, spec, medium = _scene()
    c = math.sqrt(0.5)
    pose = RigidTransform((c, -c, 0, c, c, 0, 0, 0, 1))
    spec = replace(spec, object_pose=pose)
    # The source is inside the world AABB but below the actual diamond.
    _admit(replace(source, lower_mm=(0.6, 0.1, 1), extent_xy_mm=(0.05, 0.05)), spec, medium)
    with pytest.raises(TransportError, match="discontinuous transverse"):
        _admit(replace(source, lower_mm=(0, 0.5, 1), extent_xy_mm=(0.1, 0.1)), spec, medium)


def _physical(*, second_atom=None, fluorescence=(), bound=()):
    atom = AtomicScatteringTable(1, (0, 10), (1, 1), (0, 1), P)
    return PhysicalTransport(
        (atom, second_atom or atom),
        (0, 0, 0, 0),
        fluorescence=fluorescence,
        cutoff_kev=10,
        bound_compton=bound,
    )


def test_same_extinction_but_different_absorption_scattering_partition_rejects():
    source, spec, medium = _scene()
    medium = replace(
        medium,
        absorption_mm_inverse=(0.5, 0.5, 0.25, 0.25),
        scattering_mm_inverse=(0.5, 0.5, 0.75, 0.75),
    )
    with pytest.raises(TransportError, match="discontinuous transverse"):
        _admit(source, spec, medium)


def test_unused_coefficient_energies_do_not_create_an_initial_startup_kink():
    source, spec, medium = _scene()
    medium = replace(medium, absorption_mm_inverse=(0.7, 0.7, 0.9, 0.7))
    _admit(source, spec, medium)
    with pytest.raises(TransportError, match="discontinuous transverse"):
        _admit(source, spec, medium, energies=(10.0,))


def test_active_rayleigh_phase_difference_rejects_but_zero_channel_ignores_tables():
    source, spec, medium = _scene()
    atom = AtomicScatteringTable(1, (0, 10), (1, 0.5), (0, 1), P)
    physical = _physical(second_atom=atom)
    medium = replace(
        medium,
        absorption_mm_inverse=(0.5,) * 4,
        scattering_mm_inverse=(0.5,) * 4,
        physical=physical,
    )
    _admit(source, spec, medium)
    medium = replace(medium, physical=replace(physical, rayleigh_mm_inverse=(0.2,) * 4))
    with pytest.raises(TransportError, match="discontinuous transverse"):
        _admit(source, spec, medium)


def test_normalised_rayleigh_amplitude_and_redundant_knots_are_identical_laws():
    source, spec, medium = _scene()
    atom = AtomicScatteringTable(
        2, (0, 5, 10), (2, 2, 2), (0, 1, 2), replace(P, source="different provenance")
    )
    physical = replace(_physical(second_atom=atom), rayleigh_mm_inverse=(0.2,) * 4)
    medium = replace(
        medium,
        absorption_mm_inverse=(0.5,) * 4,
        scattering_mm_inverse=(0.5,) * 4,
        physical=physical,
    )
    _admit(source, spec, medium)


def test_active_fluorescence_probabilities_are_compared_without_provenance():
    source, spec, medium = _scene()
    line = FluorescenceLine(15, 20, (0, 0.2), P)
    same = replace(line, provenance=replace(P, source="other provenance"))
    physical = _physical(fluorescence=((line,), (same,)))
    medium = replace(medium, absorption_mm_inverse=(0.5,) * 4, physical=physical)
    _admit(source, spec, medium)
    medium = replace(
        medium,
        physical=replace(
            physical, fluorescence=((line,), (replace(same, probabilities=(0, 0.3)),))
        ),
    )
    with pytest.raises(TransportError, match="discontinuous transverse"):
        _admit(source, spec, medium)


def test_bound_shell_momentum_profile_is_an_active_conditional_law():
    source, spec, medium = _scene()
    shell = ComptonShell("K", 1, 1, 0.1)
    table = BoundComptonTable(1, (shell,), P)
    same = replace(
        table,
        shells=(replace(shell, name="same numeric shell"),),
        provenance=replace(P, source="other"),
    )
    spec = replace(spec, scattering_law="bound-electron-compton")
    physical = _physical(bound=(table, same))
    medium = replace(
        medium,
        absorption_mm_inverse=(0.5,) * 4,
        scattering_mm_inverse=(0.5,) * 4,
        physical=physical,
    )
    _admit(source, spec, medium)
    changed = replace(same, shells=(replace(shell, profile_zero=0.2),))
    medium = replace(medium, physical=replace(physical, bound_compton=(table, changed)))
    with pytest.raises(TransportError, match="discontinuous transverse"):
        _admit(source, spec, medium)


@pytest.mark.parametrize("axis", (0, 1))
def test_transverse_startup_face_in_permuted_object_frame(axis):
    source, spec, medium = _scene()
    if axis == 0:
        pose = RigidTransform((0, 0, 1, 0, 1, 0, -1, 0, 0))
        grid = MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 2), 2)
        source = replace(source, lower_mm=(0.1, 0.1, -1))
    else:
        pose = RigidTransform((1, 0, 0, 0, 0, -1, 0, 1, 0))
        grid = MaterialGrid((0, 0, 0), (1, 1, 1), (1, 2, 1), 2)
        source = replace(source, lower_mm=(0.1, -0.9, 1))
    spec = replace(spec, grid=grid, object_pose=pose)
    with pytest.raises(TransportError, match="discontinuous transverse"):
        _admit(source, spec, medium)


@pytest.mark.parametrize("accessible_difference", (False, True))
def test_rayleigh_signature_uses_only_positive_coherent_energy_support(
    accessible_difference,
):
    source, spec, medium = _scene()
    spec = replace(spec, coefficient_energies_kev=(0.1, 1.0, 100.0), energy_kev=100)
    atom = AtomicScatteringTable(1, (0, 0.2, 10), (1, 1, 1), (0, 0.5, 1), P)
    changed = AtomicScatteringTable(
        1, (0, 0.2, 10), (1, 0.5 if accessible_difference else 1, 0.5), (0, 0.5, 1), P
    )
    physical = PhysicalTransport((atom, changed), (1, 0, 0) * 2)
    medium = GeometryPhysicalMedium((0, 1), (0, 0, 0) * 2, (1, 0, 0) * 2, (1, 1), (1, 1), physical)
    # Coherent is identically zero for E>=1: all active q<1/hc<0.2.
    assert 1 / 12.398419843320026 < 0.2
    if accessible_difference:
        with pytest.raises(TransportError, match="discontinuous transverse"):
            _admit(source, spec, medium, energies=(0.5,))
    else:
        _admit(source, spec, medium, energies=(0.5,))


@pytest.mark.parametrize("accessible_difference", (False, True))
def test_unreachable_bound_shells_and_common_accessible_amplitude_cancel(
    accessible_difference,
):
    source, spec, medium = _scene()
    spec = replace(
        spec,
        coefficient_energies_kev=(0.1, 1.0, 100.0),
        energy_kev=100,
        scattering_law="bound-electron-compton",
    )
    active = ComptonShell("active", 1, 0, 0.1)
    first = BoundComptonTable(2, (active, ComptonShell("inactive", 1, 50, 0.2)), P)
    second = BoundComptonTable(
        3,
        (
            replace(active, occupancy=2, profile_zero=0.2 if accessible_difference else 0.1),
            ComptonShell("other inactive", 1, 80, 0.3),
        ),
        P,
    )
    atoms = tuple(AtomicScatteringTable(z, (0, 10), (z, z), (0, z), P) for z in (2, 3))
    physical = PhysicalTransport(
        atoms, (0, 0, 0) * 2, cutoff_kev=0.1, bound_compton=(first, second)
    )
    medium = GeometryPhysicalMedium((0, 1), (0, 0, 0) * 2, (1, 0, 0) * 2, (1, 1), (1, 1), physical)
    # Noncoherent is active only .1<E<1; the 50/80-keV shells cannot be sampled.
    if accessible_difference:
        with pytest.raises(TransportError, match="discontinuous transverse"):
            _admit(source, spec, medium, energies=(0.5,))
    else:
        _admit(source, spec, medium, energies=(0.5,))


def test_unreachable_fluorescence_threshold_and_probabilities_do_not_change_the_law():
    source, spec, medium = _scene()
    spec = replace(spec, coefficient_energies_kev=(10, 20, 100), energy_kev=100)
    atom = AtomicScatteringTable(1, (0, 10), (1, 1), (0, 1), P)
    first = FluorescenceLine(15, 50, (0, 0, 0.3), P)
    second = FluorescenceLine(25, 70, (0, 0, 0.7), P)
    physical = PhysicalTransport((atom, atom), (0, 0, 0) * 2, fluorescence=((first,), (second,)))
    medium = GeometryPhysicalMedium((0, 1), (1, 0, 0) * 2, (0, 0, 0) * 2, (1, 1), (1, 1), physical)
    # Absorption is positive only E<20; every line is disabled by its threshold there.
    _admit(source, spec, medium, energies=(15.0,))


def test_duplicate_fluorescence_energies_are_compared_as_threshold_gated_probabilities():
    source, spec, medium = _scene()
    line = FluorescenceLine(15, 20, (0, 0.2), P)
    half = replace(line, probabilities=(0, 0.1))
    physical = _physical(fluorescence=((line,), (half, half)))
    medium = replace(medium, absorption_mm_inverse=(0.5,) * 4, physical=physical)
    _admit(source, spec, medium)


def test_surviving_rates_ignore_differences_below_the_canonical_cutoff():
    source, spec, medium = _scene()
    spec = replace(spec, coefficient_energies_kev=(1, 10, 80), energy_kev=80)
    atom = AtomicScatteringTable(1, (0, 10), (1, 1), (0, 1), P)
    physical = PhysicalTransport((atom, atom), (0, 0, 0) * 2, cutoff_kev=10)
    medium = GeometryPhysicalMedium(
        (0, 1), (1, 0.5, 0.5, 2, 0.5, 0.5), (0, 0, 0) * 2, (1, 1), (1, 1), physical
    )
    _admit(source, spec, medium)


@pytest.mark.parametrize("use_spectrum", (False, True))
def test_source_spectrum_checks_every_original_energy_atom_before_sampling(
    use_spectrum,
):
    source, spec, medium = _scene()
    atom = AtomicScatteringTable(1, (0, 10), (1, 1), (0, 1), P)
    spectrum = (
        SourceSpectrum((10, 80), (0.25, 0.75), "explicit initial atoms") if use_spectrum else None
    )
    physical = PhysicalTransport((atom, atom), (0,) * 4, spectrum=spectrum)
    medium = replace(medium, absorption_mm_inverse=(0.7, 0.7, 0.9, 0.7), physical=physical)
    if use_spectrum:
        with pytest.raises(TransportError, match="discontinuous transverse"):
            _admit(source, spec, medium)
    else:
        _admit(source, spec, medium)


@pytest.mark.parametrize("positive_absorption_at_atom", (False, True))
def test_fluorescence_threshold_equality_at_an_emitted_atom_is_preserved(
    positive_absorption_at_atom,
):
    source, spec, medium = _scene()
    first = FluorescenceLine(15, 80, (0, 0.2), P)
    second = replace(first, probabilities=(0, 0.3))
    physical = _physical(fluorescence=((first,), (second,)))
    medium = replace(
        medium,
        absorption_mm_inverse=(0.5, 0.5 if positive_absorption_at_atom else 0) * 2,
        physical=physical,
    )
    if positive_absorption_at_atom:
        with pytest.raises(TransportError, match="discontinuous transverse"):
            _admit(source, spec, medium)
    else:
        _admit(source, spec, medium)


@pytest.mark.parametrize("energies", ((), (float("inf"),), (-1.0,), (1000.0,), (True,)))
def test_malformed_explicit_original_energies_are_rejected(energies):
    source, spec, medium = _scene()
    with pytest.raises(TransportError, match=r"emitted energ|emitted energies"):
        _admit(source, spec, medium, energies=energies)


@pytest.mark.parametrize("continuing_line", (False, True))
def test_subcutoff_fluorescence_joins_local_absorption_but_continuing_laws_differ(
    continuing_line,
):
    source, spec, medium = _scene()
    energy = 25.0 if continuing_line else 15.0
    first = FluorescenceLine(energy, 30, (0, 0.2), P)
    second = replace(first, probabilities=(0, 0.3))
    physical = replace(_physical(fluorescence=((first,), (second,))), cutoff_kev=20)
    medium = replace(medium, absorption_mm_inverse=(0.5,) * 4, physical=physical)
    if continuing_line:
        with pytest.raises(TransportError, match="discontinuous transverse"):
            _admit(source, spec, medium)
    else:
        _admit(source, spec, medium)


def test_canonical_rayleigh_endpoint_accepts_identical_complete_forward_tables():
    from dpt.transport.physical import HC_KEV_ANGSTROM

    source, spec, medium = _scene()
    energy = 7.0
    endpoint = energy / HC_KEV_ANGSTROM
    # The forward law covers this represented endpoint. Exact rational division
    # is strictly greater and must not introduce a different admission boundary.
    with localcontext() as context:
        context.prec = 90
        rational_endpoint = Decimal.from_float(energy) / Decimal.from_float(HC_KEV_ANGSTROM)
        assert rational_endpoint > Decimal.from_float(endpoint)
    spec = replace(spec, energy_kev=energy, coefficient_energies_kev=(energy,))
    atom = AtomicScatteringTable(1, (0, endpoint), (1, 1), (0, 1), P)
    physical = PhysicalTransport((atom, atom), (0.2,) * 2, cutoff_kev=0.1)
    medium = replace(
        medium,
        absorption_mm_inverse=(0.5,) * 2,
        scattering_mm_inverse=(0.5,) * 2,
        physical=physical,
    )
    _admit(source, spec, medium)
