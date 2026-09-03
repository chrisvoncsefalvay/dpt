"""Dependency-free controls of actual analytic shell and source support laws."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.transport import geometry_bound_admission
from dpt.transport.bound_compton import FINE_STRUCTURE_CONSTANT, BoundComptonTable, ComptonShell
from dpt.transport.geometry_bound_admission import admit_bound_geometry_law
from dpt.transport.geometry_tangents import RayTangent
from dpt.transport.model import TransportError
from dpt.transport.physical import AtomicScatteringTable, FluorescenceLine
from dpt.transport.physical_geometry_events import (
    AddressedUniform,
    PhysicalEventDraws,
    conditional_physical_event,
)
from dpt.transport.source import ParallelBeam, PointConeSource
from dpt.transport.source_spectrum import SourceSpectrum
from helpers.geometry_bound_cases import PROVENANCE, model


def pencil(x=0.25, y=0.25, z=-1.0, direction=(0.0, 0.0, 1.0)):
    return ParallelBeam((x, y, z), (0.0, 0.0), direction)


def line(energy, threshold, probabilities):
    return FluorescenceLine(energy, threshold, probabilities, PROVENANCE)


class BoundGeometryAdmissionControls(unittest.TestCase):
    def test_independent_addressed_collision_reproduces_invalid_and_tuned_laws(self):
        spec, medium = model()
        entries = (
            (0, 1, 0.9),
            (3, 0, 0.5),
            (0x80000000, 0, 0.5),
            (0x80000000, 1, 0.001),
            (0x80000000, 2, 0.25),
            (0x10000000, 0, 0.5),
            (0x10000000, 1, 0.0667156968530888),
            (2, 0, 0.99),
        )

        def draws(event):
            return PhysicalEventDraws(
                event,
                0,
                "bound-compton-v3",
                tuple(
                    AddressedUniform(event, domain, lane, value) for domain, lane, value in entries
                ),
                "Independent custom-bound review: fixed actual-law coordinates, no generated draws",
            )

        first = conditional_physical_event(
            RayTangent((0.5, 0.5, 0.5), (0, 0, 1)),
            3.0,
            0,
            spec,
            medium.absorption_mm_inverse,
            medium.scattering_mm_inverse,
            medium.physical,
            draws(0),
        )
        self.assertAlmostEqual(first.energy_kev, 0.5, delta=1e-14)
        assert first.outgoing is not None
        with self.assertRaisesRegex(TransportError, "accessible shell envelope"):
            conditional_physical_event(
                first.outgoing,
                first.energy_kev,
                0,
                spec,
                medium.absorption_mm_inverse,
                medium.scattering_mm_inverse,
                medium.physical,
                draws(1),
            )
        tuned = replace(medium, scattering_mm_inverse=(0.0, 0.0, 1.0))
        second = conditional_physical_event(
            first.outgoing,
            first.energy_kev,
            0,
            spec,
            tuned.absorption_mm_inverse,
            tuned.scattering_mm_inverse,
            tuned.physical,
            draws(1),
        )
        self.assertEqual(second.status, "absorbed")
        self.assertIsNotNone(admit_bound_geometry_law(spec, tuned, pencil()))

    def test_concrete_reachable_subbinding_law_is_rejected(self):
        spec, medium = model()
        with self.assertRaisesRegex(TransportError, "undefined bound normaliser.*material 0"):
            admit_bound_geometry_law(spec, medium, pencil())

    def test_threshold_tuned_noncoherent_law_remains_admitted(self):
        spec, medium = model(scattering=(0.0, 0.0, 1.0))
        certificate = admit_bound_geometry_law(spec, medium, pencil())
        assert certificate is not None and certificate.interval is not None
        self.assertEqual(certificate.interval.lower_kev, Fraction(0.1))
        self.assertEqual(certificate.interval.upper_kev, 2)
        self.assertEqual(certificate.marks[0].cells, (0,))
        self.assertFalse(certificate.numerical_conditioning_certified)
        with self.assertRaises(FrozenInstanceError):
            certificate.scope = "source-physical-law"  # type: ignore[misc]

    def test_cutoff_equal_binding_is_valid_without_blanket_strict_cut(self):
        spec, medium = model(cutoff=1.0)
        certificate = admit_bound_geometry_law(spec, medium, pencil())
        assert certificate is not None and certificate.interval is not None
        self.assertEqual((certificate.interval.lower_kev, certificate.interval.upper_kev), (1, 2))

    def test_zero_binding_has_positive_ideal_support_at_all_surviving_energies(self):
        spec, medium = model(binding=(0.0,))
        certificate = admit_bound_geometry_law(spec, medium, pencil())
        assert certificate is not None and certificate.interval is not None
        self.assertEqual(certificate.interval.upper_kev, 3)

    def test_profile_conditioning_is_a_distinct_uncertified_obligation(self):
        spec, medium = model(binding=(0.0,))
        assert medium.physical is not None
        table = replace(
            medium.physical.bound_compton[0],
            shells=(
                ComptonShell("small analytic profile", 1.0, 0.0, FINE_STRUCTURE_CONSTANT * 1e-30),
            ),
        )
        physical = replace(medium.physical, bound_compton=(table,))
        certificate = admit_bound_geometry_law(spec, replace(medium, physical=physical), pencil())
        assert certificate is not None
        self.assertFalse(certificate.numerical_conditioning_certified)

    def test_only_minimum_occupied_binding_controls_ideal_outgoing_support(self):
        spec, medium = model(scattering=(0.0, 0.0, 1.0))
        assert medium.physical is not None
        bound = BoundComptonTable(
            2,
            (
                ComptonShell("lower occupied", 1.0, 1.0, FINE_STRUCTURE_CONSTANT),
                ComptonShell("upper occupied", 1.0, 2.9, FINE_STRUCTURE_CONSTANT),
            ),
            PROVENANCE,
        )
        angular = AtomicScatteringTable(2, (0.0, 1000.0), (2.0, 2.0), (0.0, 2.0), PROVENANCE)
        physical = replace(medium.physical, bound_compton=(bound,), scattering_tables=(angular,))
        certificate = admit_bound_geometry_law(spec, replace(medium, physical=physical), pencil())
        assert certificate is not None and certificate.interval is not None
        self.assertEqual(certificate.interval.upper_kev, 2)

    def test_source_discrete_mark_below_binding_needs_actual_noncoherent_rate(self):
        spec, medium = model(source_energy=0.8, scattering=(0.0, 0.0, 1.0))
        certificate = admit_bound_geometry_law(spec, medium, pencil())
        assert certificate is not None
        self.assertIsNone(certificate.interval)
        with self.assertRaisesRegex(TransportError, "at reachable 0.8"):
            admit_bound_geometry_law(
                spec, replace(medium, scattering_mm_inverse=(1.0,) * 3), pencil()
            )

    def test_spectrum_support_uses_every_positive_mass_mark(self):
        spectrum = SourceSpectrum((0.8, 3.0), (0.25, 0.75), "Explicit analytic two-mark source")
        spec, medium = model(spectrum=spectrum)
        with self.assertRaisesRegex(TransportError, "at reachable 0.8"):
            admit_bound_geometry_law(spec, medium, pencil())
        tuned = replace(medium, scattering_mm_inverse=(0.0, 0.0, 1.0))
        certificate = admit_bound_geometry_law(spec, tuned, pencil())
        assert certificate is not None
        self.assertEqual(tuple(mark.energy_kev for mark in certificate.marks), (0.8, 3.0))

    def test_terminal_source_marks_do_not_select_bound_law(self):
        for energy in (0.1, 0.2):
            spec, medium = model(source_energy=energy, cutoff=0.2)
            certificate = admit_bound_geometry_law(spec, medium, pencil())
            assert certificate is not None
            self.assertEqual(certificate.marks, ())
            self.assertIsNone(certificate.interval)

    def test_unilluminated_absorptive_only_remote_bad_material_is_admitted(self):
        spec, medium = model(binding=(1.0, 5.0), scattering=(0.0,) * 3 + (1.0,) * 3)
        certificate = admit_bound_geometry_law(spec, medium, pencil())
        assert certificate is not None
        self.assertEqual(certificate.source_cells, (0,))
        self.assertEqual(certificate.marks[0].cells, (0,))
        with self.assertRaisesRegex(TransportError, "material 1"):
            admit_bound_geometry_law(spec, medium, ParallelBeam((0.0, 0.25, -1), (2.0, 0.0)))

    def test_pencil_internal_lower_face_ownership_and_upper_face_miss(self):
        spec, medium = model(binding=(1.0, 5.0), scattering=(0.0,) * 3 + (1.0,) * 3)
        with self.assertRaisesRegex(TransportError, "material 1"):
            admit_bound_geometry_law(spec, medium, pencil(x=1.0))
        certificate = admit_bound_geometry_law(spec, medium, pencil(x=2.0))
        assert certificate is not None
        self.assertEqual(certificate.source_cells, ())
        self.assertEqual(certificate.marks, ())

    def test_positive_length_on_lower_box_face_is_owned(self):
        spec, medium = model()
        with self.assertRaisesRegex(TransportError, "undefined bound normaliser"):
            admit_bound_geometry_law(spec, medium, pencil(x=0.0))

    def test_corner_touch_and_backward_ray_have_no_material_support(self):
        spec, medium = model()
        diagonal = 2.0**-0.5
        for source in (
            pencil(z=-1, direction=(0.0, 0.0, -1.0)),
            pencil(x=-1, y=0, z=0.5, direction=(diagonal, diagonal, 0.0)),
        ):
            certificate = admit_bound_geometry_law(spec, medium, source)
            assert certificate is not None
            self.assertEqual(certificate.source_cells, ())

    def test_zero_voxel_has_no_conditional_law_obligation(self):
        spec, medium = model(voxels=(0.0,))
        certificate = admit_bound_geometry_law(spec, medium, pencil())
        assert certificate is not None
        self.assertEqual(certificate.marks, ())

    def test_bound_emission_expands_outgoing_interval_not_incident_mark(self):
        spec, medium = model(
            binding=(1.0, 4.0),
            nodes=(0.1, 1.0, 2.0, 2.5, 3.0),
            scattering=(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        certificate = admit_bound_geometry_law(spec, medium, pencil())
        assert certificate is not None and certificate.interval is not None
        self.assertEqual(certificate.marks[0].cells, (0,))
        self.assertEqual(certificate.interval.cells, (0, 1))
        self.assertEqual(certificate.interval.upper_kev, 2)

    def test_rayleigh_expands_incident_energy_support_by_finite_rotations(self):
        spec, medium = model(
            binding=(1.0, 5.0), scattering=(1.0,) * 6, rayleigh=(1.0,) * 3 + (0.0,) * 3
        )
        with self.assertRaisesRegex(TransportError, "material 1.*reachable 3"):
            admit_bound_geometry_law(spec, medium, pencil())

    def test_fluorescence_expands_only_emitted_energy(self):
        spec, medium = model(
            binding=(1.0, 4.0),
            scattering=(0.0,) * 3 + (0.0, 0.0, 1.0),
            fluorescence=((line(0.5, 2.5, (0.0, 0.0, 1.0)),), ()),
        )
        certificate = admit_bound_geometry_law(spec, medium, pencil())
        assert certificate is not None
        self.assertEqual(
            tuple((m.energy_kev, m.cells) for m in certificate.marks), ((0.5, (0, 1)), (3.0, (0,)))
        )

    def test_eligible_fluorescence_can_expose_subbinding_bad_law(self):
        spec, medium = model(
            binding=(1.0, 1.0),
            scattering=(0.0,) * 3 + (1.0,) * 3,
            fluorescence=((line(0.5, 2.5, (0.0, 0.0, 1.0)),), ()),
        )
        with self.assertRaisesRegex(TransportError, "material 1.*reachable 0.5"):
            admit_bound_geometry_law(spec, medium, pencil())
        unabsorbing = replace(medium, absorption_mm_inverse=(0.0,) * 3 + (1.0,) * 3)
        certificate = admit_bound_geometry_law(spec, unabsorbing, pencil())
        assert certificate is not None
        self.assertEqual(tuple(m.energy_kev for m in certificate.marks), (3.0,))

    def test_cutoff_fluorescence_does_not_activate_remote_bad_law(self):
        spec, medium = model(
            binding=(1.0, 5.0),
            scattering=(0.0,) * 3 + (1.0,) * 3,
            fluorescence=((line(0.1, 2.5, (0.0, 0.0, 1.0)),), ()),
        )
        certificate = admit_bound_geometry_law(spec, medium, pencil())
        assert certificate is not None
        self.assertEqual(tuple(m.energy_kev for m in certificate.marks), (3.0,))

    def test_open_frontier_is_not_an_attained_fluorescence_threshold(self):
        spec, medium = model(
            binding=(1.0, 1.0),
            nodes=(0.1, 1.0, 2.0, 3.0),
            scattering=(0.0, 0.0, 1.0, 1.0) * 2,
            absorption=(0.0,) * 4 + (1.0,) * 4,
            fluorescence=((), (line(0.5, 2.0, (0.0, 0.0, 1.0, 1.0)),)),
        )
        certificate = admit_bound_geometry_law(spec, medium, pencil())
        assert certificate is not None and certificate.interval is not None
        self.assertEqual(certificate.interval.upper_kev, 2)
        self.assertEqual(tuple(m.energy_kev for m in certificate.marks), (3.0,))
        seeded = admit_bound_geometry_law(spec, medium, pencil(), auxiliary_seeds=((1, 2.0),))
        assert seeded is not None
        self.assertIn(0.5, tuple(m.energy_kev for m in seeded.marks))
        self.assertEqual(seeded.scope, "source-and-explicit-auxiliary-law")

    def test_explicit_auxiliary_support_is_validated_separately(self):
        spec, medium = model(binding=(1.0, 5.0), scattering=(0.0,) * 3 + (1.0,) * 3)
        certificate = admit_bound_geometry_law(spec, medium, pencil())
        assert certificate is not None
        self.assertEqual(certificate.scope, "source-physical-law")
        with self.assertRaisesRegex(TransportError, "material 1"):
            admit_bound_geometry_law(spec, medium, pencil(), auxiliary_seeds=((1, 3.0),))

    def test_reachable_coefficient_gap_is_rejected_without_blanket_rule(self):
        spec, medium = model(nodes=(1.0, 3.0), scattering=(0.0, 1.0))
        with self.assertRaisesRegex(
            TransportError, "open bound-energy interval.*coefficient support"
        ):
            admit_bound_geometry_law(spec, medium, pencil())
        certificate = admit_bound_geometry_law(spec, medium, pencil(x=4.0))
        assert certificate is not None
        self.assertEqual(certificate.marks, ())
        absorption_only = replace(medium, scattering_mm_inverse=(0.0, 0.0))
        certificate = admit_bound_geometry_law(spec, absorption_only, pencil())
        assert certificate is not None
        self.assertIsNone(certificate.interval)

    def test_reachable_fluorescence_table_gap_is_rejected(self):
        spec, medium = model(
            nodes=(1.0, 3.0), scattering=(0.0, 0.0), fluorescence=((line(0.5, 2.5, (0.0, 1.0)),),)
        )
        with self.assertRaisesRegex(TransportError, "mark 0.5.*coefficient support"):
            admit_bound_geometry_law(spec, medium, pencil())

    def test_fluorescence_marks_iterate_bound_support_closure(self):
        spec, medium = model(
            binding=(2.9, 0.0),
            nodes=(0.1, 1.0, 2.0, 3.0),
            scattering=(0.0,) * 4 + (1.0,) * 4,
            fluorescence=((line(1.5, 2.5, (0.0, 0.0, 0.0, 1.0)),), ()),
        )
        certificate = admit_bound_geometry_law(spec, medium, pencil())
        assert certificate is not None and certificate.interval is not None
        self.assertEqual(certificate.interval.upper_kev, Fraction(1.5))

    def test_point_cone_actual_area_support_hit_and_miss(self):
        spec, medium = model(binding=(1.0, 5.0), scattering=(0.0,) * 3 + (1.0,) * 3)
        geometry = DetectorGeometry(
            (0.25, 0.25, -1.0), (0.25, 0.25, 4.0), (1, 0, 0), (0, 1, 0), (0.1, 0.1), (1, 1)
        )
        certificate = admit_bound_geometry_law(spec, medium, PointConeSource(geometry))
        assert certificate is not None
        self.assertEqual(certificate.source_cells, (0,))
        missed = replace(geometry, source_mm=(4.0, 0.25, -1.0), origin_mm=(4.0, 0.25, 4.0))
        certificate = admit_bound_geometry_law(spec, medium, PointConeSource(missed))
        assert certificate is not None
        self.assertEqual(certificate.source_cells, ())

    def test_point_cone_rays_continue_after_detector_target(self):
        spec, medium = model()
        # Source target plane precedes the material; the original ray continues.
        geometry = DetectorGeometry(
            (0.25, 0.25, -2.0), (0.25, 0.25, -1.0), (1, 0, 0), (0, 1, 0), (0.01, 0.01), (1, 1)
        )
        with self.assertRaisesRegex(TransportError, "undefined bound normaliser"):
            admit_bound_geometry_law(spec, medium, PointConeSource(geometry))

    def test_rotated_translated_pose_preserves_real_source_cell_support(self):
        spec, medium = model(binding=(1.0, 5.0), scattering=(0.0,) * 3 + (1.0,) * 3)
        pose = RigidTransform((0, -1, 0, 1, 0, 0, 0, 0, 1), (4.0, 0.0, 0.0))
        posed = replace(spec, object_pose=pose)
        certificate = admit_bound_geometry_law(posed, medium, pencil(x=3.75))
        assert certificate is not None
        self.assertEqual(certificate.source_cells, (0,))

    def test_admission_and_nonbound_noop_contracts(self):
        spec, medium = model(scattering=(0.0, 0.0, 1.0))
        self.assertIsNone(
            admit_bound_geometry_law(
                replace(spec, scattering_law="free-electron-compton"), medium, pencil()
            )
        )
        invalids = (
            replace(medium, material_ids=(2,)),
            replace(medium, voxel_density=(-1.0,)),
            replace(medium, absorption_mm_inverse=(float("inf"),) * 3),
            replace(medium, density=(1e308,), absorption_mm_inverse=(2.0,) * 3),
            replace(
                medium,
                density=(1e-300,),
                absorption_mm_inverse=(1e-100,) * 3,
                scattering_mm_inverse=(0.0,) * 3,
            ),
        )
        for invalid in invalids:
            with self.subTest(invalid=invalid), self.assertRaises(TransportError):
                admit_bound_geometry_law(spec, invalid, pencil())
        with self.assertRaisesRegex(TransportError, "valid cells"):
            admit_bound_geometry_law(spec, medium, pencil(), auxiliary_seeds=((1, 3.0),))

    def test_positive_cell_factor_underflow_matches_native_admission_guard(self):
        spec, medium = model(scattering=(0.0, 0.0, 1.0))
        underflow = replace(medium, density=(1e-300,), voxel_density=(1e-300,))
        with self.assertRaisesRegex(TransportError, "representable cell density factors"):
            admit_bound_geometry_law(spec, underflow, pencil())

    def test_host_support_certificate_loads_no_native_or_array_provider(self):
        # Other tests may have loaded native providers in their parent process.
        # Bind this lazy-import assertion to an actual isolated cold child.
        module = Path(geometry_bound_admission.__file__).resolve()
        script = (
            "import sys,json,pathlib; sys.path.insert(0,sys.argv[1]); "
            "import dpt.transport.geometry_bound_admission as module; "
            "print(json.dumps({'module':str(pathlib.Path(module.__file__).resolve()),"
            "'forbidden':[n for n in ('warp','torch','numpy','scipy','xraylib') "
            "if n in sys.modules]}))"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", script, str(module.parents[2])],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["module"], str(module))
        self.assertEqual(receipt["forbidden"], [])


if __name__ == "__main__":
    unittest.main()
