"""Independent equation, barrier and actual-law controls without native providers."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from fractions import Fraction

from dpt.geometry import DetectorGeometry
from dpt.transport.geometry_physical_admission import (
    GeometryReachableInterval,
    admit_geometry_physical_law,
)
from dpt.transport.geometry_tangents import RayTangent
from dpt.transport.model import (
    ELECTRON_REST_ENERGY_KEV,
    TransportError,
)
from dpt.transport.physical import (
    HC_KEV_ANGSTROM,
    AtomicScatteringTable,
)
from dpt.transport.physical_geometry_events import (
    AddressedUniform,
    PhysicalEventDraws,
    conditional_physical_event,
)
from dpt.transport.source import PointConeSource
from dpt.transport.source_spectrum import SourceSpectrum
from helpers.geometry_physical_cases import PROVENANCE, case, line, pencil


def backscatter_reference(energy):
    # Original reciprocal-energy Compton equation at cos(theta)=-1.
    return 1 / (1 / Fraction(energy) + 2 / Fraction(ELECTRON_REST_ENERGY_KEV))


class GeometryPhysicalAdmissionControls(unittest.TestCase):
    def angular_case(self, qmax, *, absorption=(1.0, 1.0, 1.0), rayleigh=(1.0, 1.0, 1.0)):
        spec, medium = case(
            "isotropic-elastic",
            nodes=(1.0, 3.0, 7.0),
            energy=7.0,
            absorption=absorption,
            scattering=rayleigh,
            rayleigh=rayleigh,
        )
        assert medium.physical is not None
        table = replace(
            medium.physical.scattering_tables[0], momentum_transfer_inverse_angstrom=(0.0, qmax)
        )
        return spec, replace(medium, physical=replace(medium.physical, scattering_tables=(table,)))

    def test_angular_attained_endpoint_matches_canonical_division(self):
        spec, medium = self.angular_case(7.0 / HC_KEV_ANGSTROM)
        self.assertEqual(admit_geometry_physical_law(spec, medium, pencil()).marks[0].energy_kev, 7)

    def test_angular_open_endpoint_uses_represented_supremum(self):
        upper = 7.0
        greatest = math.nextafter(upper, -math.inf)
        spec, medium = self.angular_case(greatest / HC_KEV_ANGSTROM)
        family = GeometryReachableInterval(Fraction(1), Fraction(upper), (0,))
        certificate = admit_geometry_physical_law(
            spec,
            medium,
            pencil(2),
            auxiliary_intervals=(family,),
        )
        self.assertEqual(certificate.intervals, (family,))
        # Rounded division may have a plateau without making the upper atom attained.
        self.assertGreaterEqual(upper / HC_KEV_ANGSTROM, greatest / HC_KEV_ANGSTROM)

    def test_angular_open_fraction_endpoint_between_floats(self):
        greatest = math.nextafter(7.0, -math.inf)
        high = (Fraction(greatest) + Fraction(7)) / 2
        spec, medium = self.angular_case(greatest / HC_KEV_ANGSTROM)
        family = GeometryReachableInterval(Fraction(1), high, (0,))
        self.assertEqual(
            admit_geometry_physical_law(
                spec,
                medium,
                pencil(2),
                auxiliary_intervals=(family,),
            ).intervals,
            (family,),
        )

    def test_angular_attained_neighbour_undercoverage_rejects(self):
        spec, medium = self.angular_case(math.nextafter(7 / HC_KEV_ANGSTROM, -math.inf))
        with self.assertRaisesRegex(TransportError, "angular table support"):
            admit_geometry_physical_law(spec, medium, pencil())

    def test_angular_open_neighbour_undercoverage_rejects(self):
        greatest = math.nextafter(7.0, -math.inf)
        spec, medium = self.angular_case(math.nextafter(greatest / HC_KEV_ANGSTROM, -math.inf))
        with self.assertRaisesRegex(TransportError, "angular table support"):
            admit_geometry_physical_law(
                spec,
                medium,
                pencil(2),
                auxiliary_intervals=(GeometryReachableInterval(Fraction(1), Fraction(7), (0,)),),
            )

    def test_angular_absorption_only_mark_obeys_actual_dispatch(self):
        spec, medium = self.angular_case(0.01, rayleigh=(0.0, 0.0, 0.0))
        with self.assertRaisesRegex(TransportError, "angular table support"):
            admit_geometry_physical_law(spec, medium, pencil())

    def test_angular_absorption_only_open_family_obeys_actual_dispatch(self):
        spec, medium = self.angular_case(0.01, rayleigh=(0.0, 0.0, 0.0))
        with self.assertRaisesRegex(TransportError, "angular table support"):
            admit_geometry_physical_law(
                spec,
                medium,
                pencil(2),
                auxiliary_intervals=(GeometryReachableInterval(Fraction(1), Fraction(7), (0,)),),
            )

    def test_angular_zero_total_and_vacuum_are_exempt(self):
        spec, medium = self.angular_case(0.01, absorption=(0.0,) * 3, rayleigh=(0.0,) * 3)
        family = GeometryReachableInterval(Fraction(1), Fraction(7), (0,))
        self.assertEqual(
            admit_geometry_physical_law(
                spec,
                medium,
                pencil(),
                auxiliary_intervals=(family,),
            ).intervals,
            (family,),
        )
        self.assertEqual(
            admit_geometry_physical_law(
                spec,
                replace(medium, voxel_density=(0.0,)),
                pencil(),
            ).marks,
            (),
        )

    def test_angular_positive_total_barrier_limits_open_obligation(self):
        spec, medium = self.angular_case(
            3 / HC_KEV_ANGSTROM, absorption=(1.0, 0.0, 0.0), rayleigh=(0.0,) * 3
        )
        family = GeometryReachableInterval(Fraction(1), Fraction(7), (0,))
        self.assertEqual(
            admit_geometry_physical_law(
                spec,
                medium,
                pencil(2),
                auxiliary_intervals=(family,),
            ).intervals,
            (family,),
        )

    def test_free_positive_rate_at_first_node_reaches_unsupported_energy(self):
        spec, medium = case()
        with self.assertRaisesRegex(
            TransportError, "open photon-energy family.*coefficient support"
        ):
            admit_geometry_physical_law(spec, medium, pencil())

    def test_free_rate_barrier_admits_cutoff_below_first_node(self):
        spec, medium = case(scattering=(0.0, 0.0, 1.0))
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertEqual(
            tuple((i.lower_kev, i.upper_kev) for i in certificate.intervals),
            ((backscatter_reference(1.0), Fraction(3.0)),),
        )
        self.assertIsNone(certificate.bound_profile_conditioning_certified)

    def test_exact_barrier_last_backscatter_equals_first_node(self):
        mass = ELECTRON_REST_ENERGY_KEV
        spec, medium = case(
            nodes=(mass / 4, mass / 2, mass), energy=mass, scattering=(0.0, 0.0, 1.0)
        )
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertEqual(certificate.intervals[0].lower_kev, Fraction(mass / 4))
        self.assertEqual(certificate.intervals[0].upper_kev, Fraction(mass))

    def test_zero_first_node_rate_is_insufficient_when_barrier_is_too_low(self):
        spec, medium = case(nodes=(0.1, 0.10001, 3.0), scattering=(0.0, 0.0, 1.0))
        self.assertLess(backscatter_reference(0.10001), Fraction(0.1))
        with self.assertRaisesRegex(TransportError, "coefficient support"):
            admit_geometry_physical_law(spec, medium, pencil())

    def test_zero_rate_source_atom_does_not_seed_lower_positive_edge(self):
        spec, medium = case(energy=1.0, scattering=(1.0, 0.0, 0.0))
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertEqual(certificate.intervals, ())
        self.assertEqual(tuple(m.energy_kev for m in certificate.marks), (1.0,))

    def test_all_source_spectrum_atoms_retain_zero_rate_admission(self):
        spectrum = SourceSpectrum((1.0, 3.0), (0.5, 0.5), "Explicit zero-rate source atoms")
        spec, medium = case(scattering=(1.0, 0.0, 0.0), spectrum=spectrum)
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertEqual(tuple(m.energy_kev for m in certificate.marks), (1.0, 3.0))
        self.assertEqual(certificate.intervals, ())

    def test_another_radiative_channel_can_activate_lower_free_edge(self):
        spec, medium = case(
            scattering=(1.0, 0.0, 0.0), fluorescence=((line(0.8, 2.5, (0.0, 0.0, 1.0)),),)
        )
        with self.assertRaisesRegex(TransportError, "coefficient support"):
            admit_geometry_physical_law(spec, medium, pencil())

    def test_material_barriers_are_closed_using_actual_outgoing_cells(self):
        spec, medium = case(
            nodes=(0.1, 0.2, 1.0, 2.0, 3.0),
            bindings=(1.0, 1.0),
            scattering=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        )
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertEqual(certificate.source_cells, (0,))
        self.assertEqual(certificate.intervals[0].cells, (0, 1))
        self.assertEqual(certificate.intervals[0].lower_kev, backscatter_reference(0.2))

    def test_free_emission_does_not_expand_incident_energy_atom(self):
        spec, medium = case(
            bindings=(1.0, 1.0),
            scattering=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            absorption=(0.0,) * 3 + (1.0,) * 3,
            fluorescence=((), (line(0.5, 3.0, (0.0, 0.0, 1.0)),)),
        )
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertEqual(tuple((m.energy_kev, m.cells) for m in certificate.marks), ((3.0, (0,)),))
        seeded = admit_geometry_physical_law(spec, medium, pencil(), auxiliary_marks=((1, 3.0),))
        self.assertIn(0.5, tuple(m.energy_kev for m in seeded.marks))

    def test_absorption_only_unreachable_remote_free_rate_is_admitted(self):
        spec, medium = case(bindings=(1.0, 1.0), scattering=(0.0,) * 3 + (1.0,) * 3)
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertEqual(certificate.intervals, ())
        self.assertEqual(certificate.source_cells, (0,))

    def test_rayleigh_only_preserves_energy_and_reaches_other_cells(self):
        spec, medium = case(bindings=(1.0, 1.0), rayleigh=(1.0,) * 6)
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertEqual(certificate.intervals, ())
        self.assertEqual(certificate.marks[0].cells, (0, 1))

    def test_rayleigh_can_activate_remote_noncoherent_barrier(self):
        spec, medium = case(bindings=(1.0, 1.0), rayleigh=(1.0,) * 3 + (0.0,) * 3)
        with self.assertRaisesRegex(TransportError, "coefficient support"):
            admit_geometry_physical_law(spec, medium, pencil())

    def test_narrow_forward_rayleigh_cap_has_finite_rotation_closure(self):
        spec, medium = case(bindings=(1.0, 1.0), rayleigh=(1.0,) * 6)
        assert medium.physical is not None
        narrow = AtomicScatteringTable(
            1, (0.0, 0.0001, 1000.0), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0), PROVENANCE
        )
        physical = replace(medium.physical, scattering_tables=(narrow, narrow))
        certificate = admit_geometry_physical_law(
            spec, replace(medium, physical=physical), pencil()
        )
        self.assertEqual(certificate.marks[0].cells, (0, 1))

    def test_isotropic_elastic_single_node_has_no_lower_energy_obligation(self):
        spec, medium = case("isotropic-elastic", nodes=(3.0,), bindings=(1.0, 1.0))
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertEqual(certificate.marks[0].cells, (0, 1))
        self.assertEqual(certificate.intervals, ())

    def test_continuing_fluorescence_atom_uses_its_actual_energy(self):
        spec, medium = case(
            scattering=(0.0, 0.0, 0.0), fluorescence=((line(0.5, 2.5, (0.0, 0.0, 1.0)),),)
        )
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertEqual(tuple(m.energy_kev for m in certificate.marks), (0.5, 3.0))
        self.assertEqual(certificate.intervals, ())

    def test_cutoff_fluorescence_needs_no_continuation_table_lookup(self):
        spec, medium = case(
            nodes=(1.0, 3.0),
            scattering=(0.0, 0.0),
            cutoff=0.5,
            fluorescence=((line(0.5, 2.5, (0.0, 1.0)),),),
        )
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertEqual(tuple(m.energy_kev for m in certificate.marks), (3.0,))

    def test_continuing_fluorescence_below_table_is_rejected(self):
        spec, medium = case(
            nodes=(1.0, 3.0), scattering=(0.0, 0.0), fluorescence=((line(0.5, 2.5, (0.0, 1.0)),),)
        )
        with self.assertRaisesRegex(TransportError, "outside supplied coefficient support"):
            admit_geometry_physical_law(spec, medium, pencil())

    def test_bound_invalid_and_threshold_tuned_controls_remain(self):
        spec, medium = case("bound-electron-compton")
        with self.assertRaisesRegex(TransportError, "inaccessible shells"):
            admit_geometry_physical_law(spec, medium, pencil())
        tuned = replace(medium, scattering_mm_inverse=(0.0, 0.0, 1.0))
        certificate = admit_geometry_physical_law(spec, tuned, pencil())
        self.assertEqual(
            (certificate.intervals[0].lower_kev, certificate.intervals[0].upper_kev),
            (Fraction(0.1), Fraction(2)),
        )
        self.assertFalse(certificate.bound_profile_conditioning_certified)

    def test_cutoff_terminal_atoms_and_families_do_not_activate_laws(self):
        spec, medium = case(cutoff=1.0, energy=1.0)
        declared = GeometryReachableInterval(Fraction(0.1), Fraction(1.0), (0,))
        certificate = admit_geometry_physical_law(
            spec, medium, pencil(), auxiliary_intervals=(declared,)
        )
        self.assertEqual(certificate.marks, ())
        self.assertEqual(certificate.intervals, ())
        self.assertEqual(certificate.auxiliary_intervals, (declared,))

    def test_free_continuum_is_clipped_only_by_actual_deposition_cutoff(self):
        spec, medium = case(scattering=(0.0, 0.0, 1.0), cutoff=1.5)
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertEqual(certificate.intervals[0].lower_kev, Fraction(1.5))

    def test_auxiliary_open_energy_family_retains_endpoints_and_cells(self):
        spec, medium = case(bindings=(1.0, 1.0), scattering=(0.0,) * 6)
        declared = GeometryReachableInterval(Fraction(1), Fraction(2), (1,))
        certificate = admit_geometry_physical_law(
            spec, medium, pencil(x=4.0), auxiliary_intervals=(declared,)
        )
        self.assertEqual(certificate.auxiliary_intervals, (declared,))
        self.assertEqual(certificate.intervals, (declared,))
        self.assertEqual(certificate.scope, "source-and-conservative-auxiliary-families")

    def test_auxiliary_open_upper_is_not_a_fluorescent_atom(self):
        spec, medium = case(
            scattering=(0.0,) * 3, fluorescence=((line(0.5, 3.0, (0.0, 0.0, 1.0)),),)
        )
        declared = GeometryReachableInterval(Fraction(1), Fraction(3), (0,))
        certificate = admit_geometry_physical_law(
            spec, medium, pencil(x=4), auxiliary_intervals=(declared,)
        )
        self.assertEqual(certificate.marks, ())
        attained = admit_geometry_physical_law(
            spec, medium, pencil(x=4), auxiliary_marks=((0, 3.0),), auxiliary_intervals=(declared,)
        )
        self.assertIn(0.5, tuple(m.energy_kev for m in attained.marks))

    def test_auxiliary_free_family_descends_to_actual_rate_barrier(self):
        spec, medium = case(scattering=(0.0, 0.0, 1.0))
        declared = GeometryReachableInterval(Fraction(2), Fraction(3), (0,))
        certificate = admit_geometry_physical_law(
            spec, medium, pencil(x=4), auxiliary_intervals=(declared,)
        )
        self.assertEqual(certificate.intervals[0].lower_kev, backscatter_reference(1.0))
        self.assertEqual(certificate.intervals[0].upper_kev, Fraction(3))

    def test_auxiliary_bound_open_binding_endpoint_is_not_an_invalid_atom(self):
        spec, medium = case("bound-electron-compton")
        declared = GeometryReachableInterval(Fraction(1), Fraction(1) + Fraction(0.1), (0,))
        certificate = admit_geometry_physical_law(
            spec, medium, pencil(x=4), auxiliary_intervals=(declared,)
        )
        self.assertEqual(certificate.intervals, (declared,))
        with self.assertRaisesRegex(TransportError, "zero accessible-shell"):
            admit_geometry_physical_law(spec, medium, pencil(x=4), auxiliary_marks=((0, 1.0),))

    def test_auxiliary_bound_subbinding_open_family_is_rejected(self):
        spec, medium = case("bound-electron-compton")
        declared = GeometryReachableInterval(Fraction(0.5), Fraction(1), (0,))
        with self.assertRaisesRegex(TransportError, "inaccessible shells"):
            admit_geometry_physical_law(spec, medium, pencil(x=4), auxiliary_intervals=(declared,))

    def test_distinct_open_families_do_not_fill_an_unreachable_energy_gap(self):
        spec, medium = case(nodes=(0.1, 1.0, 2.0, 3.0), scattering=(0.0,) * 4)
        families = (
            GeometryReachableInterval(Fraction(0.1), Fraction(1), (0,)),
            GeometryReachableInterval(Fraction(2), Fraction(3), (0,)),
        )
        certificate = admit_geometry_physical_law(
            spec, medium, pencil(x=4), auxiliary_intervals=families
        )
        self.assertEqual(certificate.intervals, families)

    def test_touching_open_families_do_not_turn_common_endpoint_into_atom(self):
        spec, medium = case(scattering=(0.0,) * 3)
        families = (
            GeometryReachableInterval(Fraction(0.1), Fraction(1), (0,)),
            GeometryReachableInterval(Fraction(1), Fraction(3), (0,)),
        )
        certificate = admit_geometry_physical_law(
            spec, medium, pencil(x=4), auxiliary_intervals=families
        )
        self.assertEqual(certificate.intervals, families)
        self.assertEqual(certificate.marks, ())

    def test_source_half_open_cells_and_all_miss_are_shared_exactly(self):
        spec, medium = case(bindings=(1.0, 1.0), scattering=(0.0,) * 3 + (1.0,) * 3)
        with self.assertRaisesRegex(TransportError, "coefficient support"):
            admit_geometry_physical_law(spec, medium, pencil(x=1.0))
        miss = admit_geometry_physical_law(spec, medium, pencil(x=2.0))
        self.assertEqual(miss.source_cells, ())
        self.assertEqual(miss.intervals, ())

    def test_point_cone_uses_actual_source_cell_support_for_free_model(self):
        spec, medium = case(bindings=(1.0, 1.0), scattering=(0.0,) * 3 + (1.0,) * 3)
        geometry = DetectorGeometry(
            (0.25, 0.25, -1), (0.25, 0.25, 4), (1, 0, 0), (0, 1, 0), (0.1, 0.1), (1, 1)
        )
        certificate = admit_geometry_physical_law(spec, medium, PointConeSource(geometry))
        self.assertEqual(certificate.source_cells, (0,))
        self.assertEqual(certificate.intervals, ())

    def test_large_finite_free_energy_retains_exact_barrier_without_overflow(self):
        mass = ELECTRON_REST_ENERGY_KEV
        spec, medium = case(
            nodes=(mass / 4, mass / 2, 1e308), energy=1e308, scattering=(0.0, 0.0, 1.0)
        )
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertEqual(certificate.intervals[0].lower_kev, Fraction(mass / 4))
        self.assertEqual(certificate.intervals[0].upper_kev, Fraction(1e308))

    def test_actual_conditional_free_collision_matches_original_equation(self):
        spec, medium = case(scattering=(0.0, 0.0, 1.0))
        draws = PhysicalEventDraws(
            0,
            0,
            "legacy-v1",
            tuple(
                AddressedUniform(0, domain, lane, value)
                for domain, lane, value in (
                    (0, 1, 0.9),
                    (0x80000000, 0, 0.5),
                    (0x80000000, 1, 0.001),
                    (0x80000000, 2, 0.25),
                )
            ),
            "Explicit successful KN law coordinates, cos(theta)=0",
        )
        event = conditional_physical_event(
            RayTangent((0.5, 0.5, 0.5), (0, 0, 1)),
            3.0,
            0,
            spec,
            medium.absorption_mm_inverse,
            medium.scattering_mm_inverse,
            medium.physical,
            draws,
        )
        expected = 1 / (1 / 3.0 + 1 / ELECTRON_REST_ENERGY_KEV)
        self.assertAlmostEqual(event.energy_kev, expected, delta=1e-14)
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertLess(certificate.intervals[0].lower_kev, Fraction(event.energy_kev))
        self.assertGreater(certificate.intervals[0].upper_kev, Fraction(event.energy_kev))

    def test_finite_small_angle_rotation_cost_stays_above_rate_barrier(self):
        incoming, barrier, angle = 100.0, 99.0, math.pi / 2
        energies = []
        for count in (1, 16, 128, 4096):
            step = angle / count
            inverse_loss = count * (1 - math.cos(step)) / ELECTRON_REST_ENERGY_KEV
            energy = 1 / (1 / incoming + inverse_loss)
            energies.append(energy)
            if count == 4096:
                self.assertGreater(energy, barrier)
                self.assertLess(count * (1 / (16 * count)), 0.5)
                self.assertGreater(math.sin(step), 0.0)
        self.assertEqual(energies, sorted(energies))
        self.assertLess(energies[0], barrier)
        self.assertGreater(energies[-1], 99.99)

    def test_finite_event_policy_does_not_change_physical_support(self):
        spec, medium = case(scattering=(0.0, 0.0, 1.0))
        short = admit_geometry_physical_law(replace(spec, max_events=1), medium, pencil())
        long = admit_geometry_physical_law(replace(spec, max_events=10000), medium, pencil())
        self.assertEqual(short, long)

    def test_auxiliary_family_shapes_and_positive_factor_loss_are_rejected(self):
        with self.assertRaisesRegex(TransportError, "exact Fractions"):
            GeometryReachableInterval(0.1, 1.0, (0,))  # type: ignore[arg-type]
        with self.assertRaisesRegex(TransportError, "unique"):
            GeometryReachableInterval(Fraction(0.1), Fraction(1), (0, 0))
        spec, medium = case(scattering=(0.0, 0.0, 1.0))
        with self.assertRaisesRegex(TransportError, "valid cell families"):
            admit_geometry_physical_law(
                spec,
                medium,
                pencil(),
                auxiliary_intervals=(GeometryReachableInterval(Fraction(1), Fraction(2), (1,)),),
            )
        with self.assertRaisesRegex(TransportError, "density factors"):
            admit_geometry_physical_law(
                spec, replace(medium, density=(1e-300,), voxel_density=(1e-300,)), pencil()
            )


if __name__ == "__main__":
    unittest.main()
