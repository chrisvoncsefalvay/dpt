"""Exact generator identities, continuation counterexamples and support controls."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from fractions import Fraction

from dpt.transport.bound_compton import FINE_STRUCTURE_CONSTANT, BoundComptonTable, ComptonShell
from dpt.transport.geometry_law_signatures import (
    compare_geometry_generators,
    measured_geometry_generator,
)
from dpt.transport.geometry_physical_admission import (
    GeometryReachableInterval,
    GeometryReachableMark,
    admit_geometry_physical_law,
)
from dpt.transport.model import TransportError
from dpt.transport.physical import HC_KEV_ANGSTROM, AtomicScatteringTable
from helpers.geometry_physical_cases import PROVENANCE, case, line, pencil


def mark(energy, cells=(0, 1)):
    return (GeometryReachableMark(energy, cells),)


def interval(low, high, cells=(0, 1)):
    return (GeometryReachableInterval(Fraction(low), Fraction(high), cells),)


def profiles(medium, shells):
    assert medium.physical is not None
    tables = tuple(BoundComptonTable(z, shell, PROVENANCE) for z, shell in shells)
    atomics = tuple(
        AtomicScatteringTable(
            table.atomic_number,
            (0.0, 1000.0),
            (float(table.atomic_number), float(table.atomic_number)),
            (0.0, float(table.atomic_number)),
            PROVENANCE,
        )
        for table in tables
    )
    return replace(
        medium,
        physical=replace(
            medium.physical,
            scattering_tables=atomics,
            bound_compton=tables,
        ),
    )


class GeometryLawSignatureControls(unittest.TestCase):
    def compare(self, spec, medium, marks=(), families=()):
        return compare_geometry_generators(spec, medium, 0, 1, marks, families)

    def test_fluorescent_prefix_exposes_future_generator_difference(self):
        spec, medium = case(
            nodes=(40.0, 60.0, 80.0),
            energy=80.0,
            bindings=(1.0, 1.0),
            scattering=(0.0,) * 6,
            absorption=(2.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            fluorescence=((line(40, 60, (0, 0, 0.5)),),) * 2,
        )
        self.assertEqual(
            measured_geometry_generator(spec, medium, 0, 80),
            measured_geometry_generator(spec, medium, 1, 80),
        )
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertEqual(tuple(m.energy_kev for m in certificate.marks), (40, 80))
        result = self.compare(spec, medium, certificate.marks, certificate.intervals)
        self.assertFalse(result.equivalent)
        self.assertEqual(result.status, "uncertified")
        self.assertTrue(
            any(
                w.component == "total_mm_inverse" and w.support.lower_kev == 40
                for w in result.witnesses
            )
        )

    def test_unreachable_lower_absorption_difference_is_ignored(self):
        spec, medium = case(
            nodes=(40, 60, 80),
            energy=80,
            bindings=(1, 1),
            scattering=(0.0,) * 6,
            absorption=(2, 1, 1, 1, 1, 1),
        )
        certificate = admit_geometry_physical_law(spec, medium, pencil())
        self.assertEqual(certificate.marks[0].cells, (0,))
        self.assertTrue(self.compare(spec, medium, certificate.marks).equivalent)

    def test_effective_density_coefficient_gauge_matches(self):
        spec, medium = case(
            bindings=(1, 1), scattering=(0, 0, 2, 0, 0, 1), absorption=(2, 2, 2, 1, 1, 1)
        )
        medium = replace(medium, density=(1, 2))
        self.assertTrue(self.compare(spec, medium, mark(3), interval(1, 3)).equivalent)

    def test_vacuum_and_exact_zero_generator_match(self):
        spec, medium = case(bindings=(1, 1), scattering=(0,) * 6, absorption=(0,) * 6)
        result = compare_geometry_generators(spec, medium, 0, None, mark(3, (0,)), ())
        self.assertTrue(result.equivalent)
        self.assertEqual(
            measured_geometry_generator(spec, medium, None, 1000).total_mm_inverse, (Fraction(0),)
        )

    def test_absorbing_material_does_not_match_vacuum(self):
        spec, medium = case(bindings=(1, 1), scattering=(0,) * 6)
        result = compare_geometry_generators(spec, medium, 0, None, mark(3, (0,)), ())
        self.assertEqual(result.witnesses[0].component, "total_mm_inverse")
        self.assertEqual(result.status, "uncertified")

    def test_other_cells_do_not_activate_unreachable_family(self):
        spec, medium = case(
            bindings=(1, 1, 1), scattering=(0,) * 9, absorption=(1,) * 3 + (2,) * 3 + (3,) * 3
        )
        result = self.compare(spec, medium, mark(3, (2,)), interval(1, 3, (2,)))
        self.assertTrue(result.equivalent)
        self.assertEqual(result.supports, ())

    def test_quadratic_fluorescence_rate_is_not_replaced_by_endpoint_samples(self):
        spec, medium = case(
            nodes=(1, 3),
            energy=3,
            bindings=(1, 1),
            absorption=(1, 3, 2, 2),
            scattering=(3, 1, 2, 2),
            fluorescence=((line(0.2, 0.5, (0.75, 0.25)),), (line(0.2, 0.5, (0.375, 0.375)),)),
        )
        result = self.compare(spec, medium, families=interval(1, 3))
        self.assertTrue(any(w.component == "fluorescence" for w in result.witnesses))
        first = measured_geometry_generator(spec, medium, 0, 1).fluorescence[0][1][0]
        last = measured_geometry_generator(spec, medium, 1, 1).fluorescence[0][1][0]
        self.assertEqual(first, last)
        self.assertEqual(
            measured_geometry_generator(spec, medium, 0, 3).fluorescence,
            measured_geometry_generator(spec, medium, 1, 3).fluorescence,
        )
        # Original product of affine A(E)=E and p(E)=(4-E)/4 differs inside.
        self.assertEqual(
            measured_geometry_generator(spec, medium, 0, 2).fluorescence[0][1],
            (Fraction(2) * (4 - Fraction(2)) / 4,),
        )
        self.assertEqual(
            measured_geometry_generator(spec, medium, 1, 2).fluorescence[0][1], (Fraction(3, 4),)
        )

    def test_equal_emitted_lines_are_grouped_by_energy(self):
        spec, medium = case(
            bindings=(1, 1),
            scattering=(0,) * 6,
            fluorescence=(
                (line(0.5, 1, (0, 0.25, 0.25)), line(0.5, 1, (0, 0.25, 0.25))),
                (line(0.5, 1, (0, 0.5, 0.5)),),
            ),
        )
        self.assertTrue(self.compare(spec, medium, mark(3), interval(1, 3)).equivalent)
        self.assertEqual(len(measured_geometry_generator(spec, medium, 0, 3).fluorescence), 1)

    def test_open_upper_threshold_is_not_an_attained_fluorescence_mark(self):
        spec, medium = case(
            nodes=(10, 20, 30),
            energy=30,
            bindings=(1, 1),
            scattering=(0,) * 6,
            fluorescence=((line(5, 20, (0, 0.5, 0.5)),), ()),
        )
        self.assertTrue(self.compare(spec, medium, families=interval(10, 20)).equivalent)
        self.assertFalse(self.compare(spec, medium, mark(20)).equivalent)

    def test_subcutoff_fluorescence_is_an_extinction_only_outcome(self):
        spec, medium = case(
            bindings=(1, 1),
            scattering=(0,) * 6,
            cutoff=0.5,
            fluorescence=((line(0.5, 1, (0, 0.5, 0.5)),), ()),
        )
        self.assertTrue(self.compare(spec, medium, mark(3), interval(1, 3)).equivalent)

    def test_elastic_isotropic_and_rayleigh_rates_remain_distinct(self):
        spec, medium = case(
            "isotropic-elastic", bindings=(1, 1), scattering=(1,) * 6, rayleigh=(1,) * 3 + (0,) * 3
        )
        result = self.compare(spec, medium, mark(3))
        self.assertTrue(any(w.component == "rayleigh_mm_inverse" for w in result.witnesses))
        self.assertTrue(any(w.component == "noncoherent_mm_inverse" for w in result.witnesses))

    def test_free_channel_survives_forward_even_when_backscatter_is_below_cutoff(self):
        spec, medium = case(
            bindings=(1, 1),
            cutoff=2.99,
            scattering=(1,) * 3 + (2,) * 3,
            absorption=(2,) * 3 + (1,) * 3,
        )
        # Original KN recoil equation leaves a positive forward cap above L.
        from dpt.transport.model import ELECTRON_REST_ENERGY_KEV

        minimum = 3 / (1 + 6 / ELECTRON_REST_ENERGY_KEV)
        self.assertLess(minimum, 2.99)
        result = self.compare(spec, medium, mark(3))
        self.assertFalse(result.equivalent)
        self.assertTrue(any(w.component == "noncoherent_mm_inverse" for w in result.witnesses))

    def test_rayleigh_normalised_shape_groups_collinear_nodes_and_z_scale(self):
        spec, medium = case(
            "isotropic-elastic", bindings=(1, 1), scattering=(1,) * 6, rayleigh=(1,) * 6
        )
        assert medium.physical is not None
        tables = (
            AtomicScatteringTable(1, (0, 1, 1000), (1, 0.5, 0), (0, 0, 1), PROVENANCE),
            AtomicScatteringTable(2, (0, 0.5, 1, 1000), (2, 1.5, 1, 0), (0, 0, 0, 2), PROVENANCE),
        )
        medium = replace(medium, physical=replace(medium.physical, scattering_tables=tables))
        self.assertTrue(self.compare(spec, medium, mark(3), interval(1, 3)).equivalent)

    def test_rayleigh_differences_beyond_reached_q_are_ignored(self):
        spec, medium = case(
            "isotropic-elastic", bindings=(1, 1), scattering=(1,) * 6, rayleigh=(1,) * 6
        )
        assert medium.physical is not None
        tables = tuple(
            AtomicScatteringTable(1, (0, 1, 1000), (1, 1, last), (0, 0, 1), PROVENANCE)
            for last in (0, 1)
        )
        medium = replace(medium, physical=replace(medium.physical, scattering_tables=tables))
        self.assertTrue(self.compare(spec, medium, mark(3), interval(1, 3)).equivalent)

    def test_active_rayleigh_difference_is_uncertified(self):
        spec, medium = case(
            "isotropic-elastic", bindings=(1, 1), scattering=(1,) * 6, rayleigh=(1,) * 6
        )
        assert medium.physical is not None
        tables = tuple(
            AtomicScatteringTable(1, (0, 1000), (1, last), (0, 1), PROVENANCE) for last in (0, 1)
        )
        medium = replace(medium, physical=replace(medium.physical, scattering_tables=tables))
        self.assertTrue(
            any(
                w.component == "rayleigh_shape"
                for w in self.compare(spec, medium, mark(3)).witnesses
            )
        )

    def test_absorption_only_physical_mark_retains_dispatch_q_coverage(self):
        spec, medium = case(
            "isotropic-elastic", bindings=(1, 1), scattering=(0,) * 6, rayleigh=(0,) * 6
        )
        assert medium.physical is not None
        tables = tuple(
            replace(t, momentum_transfer_inverse_angstrom=(0, 0.01))
            for t in medium.physical.scattering_tables
        )
        medium = replace(medium, physical=replace(medium.physical, scattering_tables=tables))
        with self.assertRaisesRegex(TransportError, "angular table support"):
            self.compare(spec, medium, mark(3))

    def test_canonical_rounded_q_endpoint_is_admitted(self):
        spec, medium = case(
            "isotropic-elastic",
            nodes=(1, 3, 7),
            energy=7,
            bindings=(1, 1),
            scattering=(1,) * 6,
            rayleigh=(1,) * 6,
        )
        assert medium.physical is not None
        tables = tuple(
            replace(t, momentum_transfer_inverse_angstrom=(0, 7 / HC_KEV_ANGSTROM))
            for t in medium.physical.scattering_tables
        )
        medium = replace(medium, physical=replace(medium.physical, scattering_tables=tables))
        self.assertTrue(self.compare(spec, medium, mark(7), interval(1, 7)).equivalent)

    def test_bound_shell_grouping_reordering_and_occupancy_gauge(self):
        spec, medium = case("bound-electron-compton", bindings=(1, 1))
        h = FINE_STRUCTURE_CONSTANT
        medium = profiles(
            medium,
            (
                (2, (ComptonShell("a", 1, 1, h), ComptonShell("b", 1, 1, h))),
                (4, (ComptonShell("renamed", 4, 1, h),)),
            ),
        )
        self.assertTrue(self.compare(spec, medium, mark(3), interval(1, 3)).equivalent)

    def test_distinct_bound_profile_groups_ignore_order_and_names(self):
        spec, medium = case("bound-electron-compton", bindings=(1, 1))
        h = FINE_STRUCTURE_CONSTANT
        medium = profiles(
            medium,
            (
                (3, (ComptonShell("first", 1, 0, h), ComptonShell("last", 2, 1, 2 * h))),
                (3, (ComptonShell("different", 2, 1, 2 * h), ComptonShell("name", 1, 0, h))),
            ),
        )
        self.assertTrue(self.compare(spec, medium, mark(3), interval(1, 3)).equivalent)

    def test_inaccessible_bound_shells_do_not_change_normalised_law(self):
        spec, medium = case("bound-electron-compton", bindings=(1, 1))
        h = FINE_STRUCTURE_CONSTANT
        medium = profiles(
            medium,
            (
                (2, (ComptonShell("a", 1, 1, h), ComptonShell("dead", 1, 100, h))),
                (3, (ComptonShell("a", 1, 1, h), ComptonShell("dead", 2, 200, 2 * h))),
            ),
        )
        self.assertTrue(self.compare(spec, medium, mark(3), interval(1, 3)).equivalent)

    def test_accessible_terminal_shells_are_kept_in_surviving_normaliser(self):
        spec, medium = case("bound-electron-compton", bindings=(1, 1), cutoff=1)
        h = FINE_STRUCTURE_CONSTANT
        medium = profiles(
            medium,
            tuple(
                (2, (ComptonShell("live", 1, 0.5, h), ComptonShell("terminal", 1, 2.5, height)))
                for height in (h, 2 * h)
            ),
        )
        result = self.compare(spec, medium, mark(3))
        self.assertEqual(result.status, "uncertified")
        self.assertTrue(any(w.component == "bound_profiles" for w in result.witnesses))
        signature = measured_geometry_generator(spec, medium, 0, 3)
        self.assertEqual(len(signature.bound_profiles), 2)

    def test_terminal_only_bound_and_absorption_can_offset(self):
        spec, medium = case(
            "bound-electron-compton",
            nodes=(1, 2, 3),
            energy=2,
            bindings=(1, 1),
            cutoff=1,
            absorption=(1,) * 3 + (2,) * 3,
            scattering=(2,) * 3 + (1,) * 3,
        )
        self.assertTrue(self.compare(spec, medium, mark(2), interval(1, 2)).equivalent)
        self.assertFalse(self.compare(spec, medium, mark(3)).equivalent)

    def test_binding_and_survival_partition_keeps_open_upper_excluded(self):
        spec, medium = case(
            "bound-electron-compton", nodes=(1, 2, 3), energy=2, bindings=(1, 1), cutoff=1
        )
        result = self.compare(spec, medium, families=interval(1, 3))
        self.assertTrue(result.equivalent)
        self.assertEqual(
            tuple((s.lower_kev, s.upper_kev) for s in result.supports),
            ((Fraction(1), Fraction(2)), (Fraction(2), Fraction(3))),
        )
        self.assertTrue(all(not s.attained for s in result.supports))

    def test_cutoff_and_inactive_channels_omit_shapes(self):
        spec, medium = case(
            "bound-electron-compton",
            nodes=(0.1, 1, 3),
            energy=1,
            bindings=(1, 1),
            cutoff=1,
            scattering=(1,) * 3 + (2,) * 3,
        )
        self.assertTrue(self.compare(spec, medium, mark(1)).equivalent)
        self.assertEqual(
            measured_geometry_generator(spec, medium, 0, 1).total_mm_inverse, (Fraction(0),)
        )

    def test_undefined_bound_kernel_and_invalid_support_raise(self):
        spec, medium = case("bound-electron-compton", bindings=(1, 1))
        with self.assertRaisesRegex(TransportError, "accessible shells"):
            self.compare(spec, medium, mark(1))
        with self.assertRaisesRegex(TransportError, "coefficient range"):
            measured_geometry_generator(spec, medium, 0, 4)
        with self.assertRaisesRegex(TransportError, "valid or None"):
            measured_geometry_generator(spec, medium, 2, 3)
        with self.assertRaisesRegex(TransportError, "positive finite"):
            measured_geometry_generator(spec, medium, 0, math.nan)


if __name__ == "__main__":
    unittest.main()
