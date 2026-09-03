"""Rigid-frame, independent kinematic and actual-provider collision checks."""

import importlib
import math
import unittest
from dataclasses import replace
from decimal import Decimal, localcontext
from typing import Literal, cast

from dpt.geometry import Vector3, cross, dot
from dpt.transport.geometry_tangents import RayTangent
from dpt.transport.model import (
    ELECTRON_REST_ENERGY_KEV,
    MaterialGrid,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from dpt.transport.physical_geometry_events import (
    AddressedUniform,
    Layout,
    PhysicalEventDraws,
    conditional_physical_event,
    terminal_physical_kernels,
)


def _spec(
    law: Literal[
        "isotropic-elastic", "free-electron-compton", "bound-electron-compton"
    ] = "free-electron-compton",
):
    return TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 1), 1),
        PlanarDetector((-2, -2), (1, 1), (4, 4), 3),
        80.0,
        "Explicit host collision law, rates in mm^-1",
        (0.1, 80.0, 200.0),
        scattering_law=law,
        max_angle_trials=3,
        max_events=4,
    )


def _draws(entries, *, layout: Layout = "legacy-v1", branch=0):
    return PhysicalEventDraws(
        0,
        branch,
        layout,
        tuple(
            AddressedUniform(0, domain + branch * 65536, lane, value)
            for domain, lane, value in entries
        ),
        "Explicit addressed conditional-event integration coordinates",
    )


def _rotation(vector: Vector3, omega: Vector3, time: float) -> Vector3:
    magnitude = math.hypot(*omega)
    axis = cast(Vector3, tuple(v / magnitude for v in omega))
    angle = magnitude * time
    perpendicular = cross(axis, vector)
    projection = dot(axis, vector)
    return cast(
        Vector3,
        tuple(
            math.cos(angle) * v + math.sin(angle) * p + (1 - math.cos(angle)) * projection * a
            for v, p, a in zip(vector, perpendicular, axis, strict=True)
        ),
    )


class HostPhysicalGeometryEvents(unittest.TestCase):
    def test_budget_and_overflowing_rate_admission(self):
        incident = RayTangent((0, 0, 0), (0, 0, 1))
        draws = _draws(((0, 1, 0.1),))
        with self.assertRaisesRegex(TransportError, "65536"):
            conditional_physical_event(
                incident,
                80,
                0,
                replace(_spec(), max_angle_trials=65537),
                (0.2,) * 3,
                (0.8,) * 3,
                None,
                draws,
            )
        with self.assertRaisesRegex(TransportError, "finite positive total"):
            conditional_physical_event(
                incident, 80, 0, _spec(), (1e308,) * 3, (1e308,) * 3, None, draws
            )
        with self.assertRaisesRegex(TransportError, "density-scaled"):
            conditional_physical_event(
                incident, 80, 0, _spec(), (1.0,) * 3, (1.0,) * 3, None, draws, density_factor=1e308
            )
        with self.assertRaisesRegex(TransportError, "density-scaled"):
            conditional_physical_event(
                incident,
                80,
                0,
                _spec(),
                (1e-100,) * 3,
                (0.0,) * 3,
                None,
                draws,
                density_factor=1e-300,
            )

    def test_full_admitted_kn_energy_range_against_decimal_analytic_total(self):
        incident = RayTangent((0, 0, 0), (0, 0, 1))
        with localcontext() as context:
            context.prec = 80
            for energy in (0.01, 80.0, 5e6, 5e7, 5e8, 5e10, 1e155):
                spec = replace(
                    _spec(), energy_kev=energy, coefficient_energies_kev=(0.001, energy, 2 * energy)
                )
                (kernel,) = terminal_physical_kernels(
                    incident, (1, 0, 0), energy, 0, spec, (0.0,) * 3, (1.0,) * 3, None
                )
                alpha = Decimal(str(energy)) / Decimal(str(ELECTRON_REST_ENERGY_KEV))
                one, two = Decimal(1), Decimal(2)
                logarithm = (one + two * alpha).ln()
                integral = two * (
                    (one + alpha)
                    / alpha**2
                    * (two * (one + alpha) / (one + two * alpha) - logarithm / alpha)
                    + logarithm / (two * alpha)
                    - (one + 3 * alpha) / (one + two * alpha) ** 2
                )
                ratio = one / (one + alpha)
                shape = ratio**3 + ratio - ratio**2
                reference = float(shape / (two * Decimal(str(math.pi)) * integral))
                self.assertLess(abs(kernel.density / reference - 1), 2e-12)
        energy = 1e155
        spec = replace(
            _spec(), energy_kev=energy, coefficient_energies_kev=(0.1, energy, 2 * energy)
        )
        (kernel,) = terminal_physical_kernels(
            incident,
            (1, 0, 3),
            energy,
            0,
            spec,
            (0.0,) * 3,
            (1.0,) * 3,
            None,
            endpoint_dot=(1, 0, 0),
        )
        independent_cosine_dot = -3 / (math.sqrt(10) ** 3)
        independent_energy_dot = (
            (kernel.energy_kev / ELECTRON_REST_ENERGY_KEV)
            * kernel.energy_kev
            * independent_cosine_dot
        )
        self.assertTrue(math.isfinite(kernel.energy_dot))
        self.assertAlmostEqual(kernel.energy_dot / independent_energy_dot, 1, places=14)

    def test_fixed_compton_cosine_energy_and_rigid_rotation_jet(self):
        incident = RayTangent((0, 0, 0), (0, 0, 1), (0.1, -0.2, 0.3), (0.4, -0.7, 0))
        draws = _draws(
            ((0, 1, 0.8), (0x80000000, 0, 0.65), (0x80000000, 1, 0.001), (0x80000000, 2, 0.43))
        )
        event = conditional_physical_event(
            incident, 80, 0, _spec(), (0.2,) * 3, (0.8,) * 3, None, draws
        )
        self.assertEqual(event.channel, "free-compton")
        self.assertEqual(event.status, "continuing")
        self.assertIsNotNone(event.outgoing)
        ray = event.outgoing
        assert ray is not None
        self.assertAlmostEqual(dot(incident.direction, ray.direction), 0.3, places=14)
        expected_energy = 80 / (1 + 80 / ELECTRON_REST_ENERGY_KEV * (1 - 0.3))
        self.assertAlmostEqual(event.energy_kev, expected_energy, places=13)
        self.assertAlmostEqual(event.energy_kev + event.deposited_kev, 80, places=13)
        self.assertEqual(ray.origin_dot, incident.origin_dot)
        self.assertAlmostEqual(dot(ray.direction, ray.direction_dot), 0, places=14)
        self.assertAlmostEqual(
            dot(incident.direction_dot, ray.direction) + dot(incident.direction, ray.direction_dot),
            0,
            places=14,
        )
        omega = cross(incident.direction, incident.direction_dot)
        delta = 1e-6
        positive, negative = (
            _rotation(ray.direction, omega, delta),
            _rotation(ray.direction, omega, -delta),
        )
        for derivative, plus, minus in zip(ray.direction_dot, positive, negative, strict=True):
            self.assertAlmostEqual(derivative, (plus - minus) / (2 * delta), delta=1e-10)

    def test_density_factor_changes_total_only(self):
        ray = RayTangent((0, 0, 0), (0, 0, 1))
        draws = _draws(((0, 1, 0.8), (0, 2, 0.65), (0, 3, 0.43)))
        arguments = (ray, 80, 0, _spec("isotropic-elastic"), (0.2,) * 3, (0.8,) * 3, None, draws)
        first = conditional_physical_event(*arguments)
        second = conditional_physical_event(*arguments, density_factor=2.5)
        self.assertEqual(first.outgoing, second.outgoing)
        self.assertEqual(first.energy_kev, second.energy_kev)
        self.assertAlmostEqual(second.total_mm_inverse, 2.5 * first.total_mm_inverse)

    def test_absorption_and_zero_rate_guards(self):
        ray = RayTangent((0, 0, 0), (0, 0, 1))
        absorbed = conditional_physical_event(
            ray, 80, 0, _spec(), (0.8,) * 3, (0.2,) * 3, None, _draws(((0, 1, 0.1),))
        )
        self.assertEqual(
            (absorbed.status, absorbed.energy_kev, absorbed.deposited_kev), ("absorbed", 0, 80)
        )
        with self.assertRaisesRegex(TransportError, "positive total rate"):
            conditional_physical_event(
                ray, 80, 0, _spec(), (0,) * 3, (0,) * 3, None, _draws(((0, 1, 0.1),))
            )

    def test_bounded_rejection_is_invalid_and_never_absorption(self):
        entries = [(0, 1, 0.8)]
        for trial in range(3):
            entries.extend(
                (
                    (0x80000000 + trial, 0, 0.02),
                    (0x80000000 + trial, 1, 0.99),
                    (0x80000000 + trial, 2, 0.3),
                )
            )
        result = conditional_physical_event(
            RayTangent((0, 0, 0), (0, 0, 1)),
            80,
            0,
            _spec(),
            (0.2,) * 3,
            (0.8,) * 3,
            None,
            _draws(entries),
        )
        self.assertEqual(result.status, "angle-budget")
        self.assertEqual((result.energy_kev, result.deposited_kev), (80, 0))
        self.assertEqual(len(result.used_uniforms), 10)

    def test_missing_coordinates_and_duplicate_addresses_raise(self):
        with self.assertRaisesRegex(TransportError, "unique"):
            _draws(((0, 1, 0.2), (0, 1, 0.3)))
        with self.assertRaisesRegex(TransportError, "missing physical coordinate"):
            conditional_physical_event(
                RayTangent((0, 0, 0), (0, 0, 1)),
                80,
                0,
                _spec(),
                (0.2,) * 3,
                (0.8,) * 3,
                None,
                _draws(((0, 1, 0.8),)),
            )
        for value in (0, 1, math.nan):
            with self.assertRaises(TransportError):
                AddressedUniform(0, 0, 0, value)

    def test_branch_address_offset_and_packed_proposals_match_declared_layout(self):
        incident = RayTangent((0, 0, 0), (0, 0, 1))
        spec = replace(_spec(), compton_sampler="packed-uniform-rejection")
        draws = _draws(
            (
                (0, 1, 0.8),
                (0, 2, 0.43),
                (0x80000000, 0, 0.02),
                (0x80000000, 1, 0.99),
                (0x80000000, 2, 0.65),
                (0x80000000, 3, 0.001),
            ),
            layout="packed-compton-v2",
            branch=2,
        )
        result = conditional_physical_event(
            incident, 80, 0, spec, (0.2,) * 3, (0.8,) * 3, None, draws
        )
        assert result.outgoing is not None
        self.assertAlmostEqual(dot(result.outgoing.direction, incident.direction), 0.3, places=14)
        self.assertTrue(all((draw.domain >> 16) & 0xFFF == 2 for draw in result.used_uniforms))
        with self.assertRaisesRegex(TransportError, "layout"):
            conditional_physical_event(
                incident, 80, 0, _spec(), (0.2,) * 3, (0.8,) * 3, None, draws
            )

    def test_terminal_kn_endpoint_and_energy_jets_against_independent_finite_geometry(self):
        incident = RayTangent((0.1, -0.2, 0), (0, 0, 1), (0.2, -0.1, 0.3), (0.4, -0.2, 0))
        endpoint, velocity = (1.4, 0.7, 3.0), (0.3, 0.1, -0.2)
        spec = _spec()
        (result,) = terminal_physical_kernels(
            incident, endpoint, 80, 0, spec, (0.2,) * 3, (0.8,) * 3, None, endpoint_dot=velocity
        )
        delta = 1e-6
        omega = cross(incident.direction, incident.direction_dot)
        results = []
        for time in (-delta, delta):
            ray = RayTangent(
                cast(
                    Vector3,
                    tuple(
                        a + time * b
                        for a, b in zip(incident.origin_mm, incident.origin_dot, strict=True)
                    ),
                ),
                _rotation(incident.direction, omega, time),
            )
            point = cast(
                Vector3, tuple(a + time * b for a, b in zip(endpoint, velocity, strict=True))
            )
            results.append(
                terminal_physical_kernels(ray, point, 80, 0, spec, (0.2,) * 3, (0.8,) * 3, None)[0]
            )
        self.assertAlmostEqual(
            result.cosine_dot, (results[1].cosine - results[0].cosine) / (2 * delta), delta=1e-10
        )
        self.assertAlmostEqual(
            result.energy_dot,
            (results[1].energy_kev - results[0].energy_kev) / (2 * delta),
            delta=1e-8,
        )
        self.assertAlmostEqual(
            result.density_dot, (results[1].density - results[0].density) / (2 * delta), delta=1e-10
        )
        ratio = result.energy_kev / 80
        shape = ratio**3 + ratio - ratio**2 * (1 - result.cosine**2)
        alpha = 80 / ELECTRON_REST_ENERGY_KEV
        integral = 2 * (
            (1 + alpha)
            / alpha**2
            * (2 * (1 + alpha) / (1 + 2 * alpha) - math.log1p(2 * alpha) / alpha)
            + math.log1p(2 * alpha) / (2 * alpha)
            - (1 + 3 * alpha) / (1 + 2 * alpha) ** 2
        )
        self.assertAlmostEqual(result.density, 0.8 * shape / (2 * math.pi * integral), delta=3e-15)

    def test_isotropic_terminal_mass_and_analogue_contract(self):
        ray = RayTangent((0, 0, 0), (0, 0, 1), (0, 0, 0), (0.5, 0, 0))
        (kernel,) = terminal_physical_kernels(
            ray, (1, 0, 3), 80, 0, _spec("isotropic-elastic"), (0.2,) * 3, (0.8,) * 3, None
        )
        self.assertAlmostEqual(4 * math.pi * kernel.density, 0.8, places=14)
        self.assertEqual((kernel.energy_dot, kernel.density_dot), (0, 0))
        with self.assertRaisesRegex(TransportError, "analogue"):
            terminal_physical_kernels(
                ray,
                (1, 0, 3),
                80,
                0,
                replace(_spec(), estimator="continuous-absorption"),
                (0.2,) * 3,
                (0.8,) * 3,
                None,
            )


class ActualProviderPhysicalGeometryEvents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        xraylib = importlib.import_module("xraylib")

        from dpt.transport.bound_compton import xraylib_bound_compton
        from dpt.transport.physical import (
            PhysicalTransport,
            xraylib_atomic_scattering,
            xraylib_fluorescence_lines,
        )

        if xraylib.__version__ != "4.3.0":
            raise RuntimeError(
                "physical geometry qualification requires the pinned xraylib4.3.0 provider"
            )
        cls.atomic = xraylib_atomic_scattering(13, (0, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 20))
        cls.shells = xraylib_bound_compton(13)
        cls.lines = xraylib_fluorescence_lines(13, (0.1, 80, 200), lines=("KL2_LINE", "KL3_LINE"))
        cls.physical = PhysicalTransport(
            (cls.atomic,),
            (0.2,) * 3,
            fluorescence=(cls.lines,),
            cutoff_kev=0.1,
        )

    def test_real_rayleigh_frame_energy_and_terminal_derivative(self):
        incident = RayTangent((0, 0, 0), (0, 0, 1), (0, 0, 0), (0.3, -0.2, 0))
        draws = _draws(
            (
                (0, 1, 0.8),
                (3, 0, 0.01),
                (0xC0000000, 0, 0.99),
                (0xC0000000, 1, 1e-10),
                (0xC0000000, 2, 0.43),
            )
        )
        event = conditional_physical_event(
            incident, 80, 0, _spec(), (0.2,) * 3, (0.8,) * 3, self.physical, draws
        )
        self.assertEqual(event.channel, "rayleigh")
        assert event.outgoing is not None
        self.assertEqual((event.energy_kev, event.deposited_kev), (80, 0))
        self.assertAlmostEqual(
            dot(incident.direction_dot, event.outgoing.direction)
            + dot(incident.direction, event.outgoing.direction_dot),
            0,
            places=14,
        )
        endpoint, velocity = (1.4, 0.7, 3), (0.3, 0.1, -0.2)
        result = terminal_physical_kernels(
            incident,
            endpoint,
            80,
            0,
            _spec(),
            (0.2,) * 3,
            (0.8,) * 3,
            self.physical,
            endpoint_dot=velocity,
        )[0]
        epsilon = 1e-6
        omega = cross(incident.direction, incident.direction_dot)
        values = []
        for time in (-epsilon, epsilon):
            ray = RayTangent((0, 0, 0), _rotation(incident.direction, omega, time))
            target = cast(
                Vector3, tuple(a + time * b for a, b in zip(endpoint, velocity, strict=True))
            )
            values.append(
                terminal_physical_kernels(
                    ray, target, 80, 0, _spec(), (0.2,) * 3, (0.8,) * 3, self.physical
                )[0].density
            )
        self.assertAlmostEqual(
            result.density_dot, (values[1] - values[0]) / (2 * epsilon), delta=1e-10
        )

    def test_actual_fluorescence_branch_probability_and_energy_balance(self):
        draws = _draws(((0, 1, 0.01), (2, 0, 0.001), (2, 1, 0.7), (2, 2, 0.3)))
        incident = RayTangent((0, 0, 0), (0, 0, 1))
        result = conditional_physical_event(
            incident, 80, 0, _spec(), (0.2,) * 3, (0.8,) * 3, self.physical, draws
        )
        self.assertEqual(result.channel, "fluorescence")
        self.assertEqual(result.energy_kev, self.lines[0].energy_kev)
        self.assertAlmostEqual(result.energy_kev + result.deposited_kev, 80, places=13)
        kernels = terminal_physical_kernels(
            incident, (1, 0, 3), 80, 0, _spec(), (0.2,) * 3, (0.8,) * 3, self.physical
        )
        fluorescent = [item for item in kernels if item.channel == "fluorescence"]
        self.assertEqual(len(fluorescent), 2)
        self.assertAlmostEqual(
            4 * math.pi * sum(item.density for item in fluorescent),
            0.2 * sum(line.probabilities[1] for line in self.lines),
            places=14,
        )
        below = conditional_physical_event(
            incident, 0.05, 0, _spec(), (0.2,) * 3, (0.8,) * 3, self.physical, _draws(())
        )
        self.assertEqual(
            (below.channel, below.deposited_kev, below.used_uniforms), ("cutoff", 0.05, ())
        )

    def test_actual_bound_shell_momentum_kinematics_and_fixed_joint_mark(self):
        physical = replace(self.physical, bound_compton=(self.shells,))
        incident = RayTangent((0, 0, 0), (0, 0, 1), (0, 0, 0), (0.3, -0.2, 0))
        draws = _draws(
            (
                (0, 1, 0.8),
                (3, 0, 0.8),
                (0x80000000, 0, 0.65),
                (0x80000000, 1, 0.001),
                (0x80000000, 2, 0.43),
                (0x10000000, 0, 0.4),
                (0x10000000, 1, 0.7),
            ),
            layout="bound-compton-v3",
        )
        spec = _spec("bound-electron-compton")
        with self.assertRaisesRegex(TransportError, "bound-compton-v3"):
            conditional_physical_event(
                incident,
                80,
                0,
                spec,
                (0.2,) * 3,
                (0.8,) * 3,
                physical,
                replace(draws, layout="legacy-v1"),
            )
        with self.assertRaisesRegex(TransportError, "cutoff"):
            conditional_physical_event(
                incident,
                80,
                0,
                spec,
                (0.2,) * 3,
                (0.8,) * 3,
                replace(physical, cutoff_kev=0.05),
                draws,
            )
        with self.assertRaisesRegex(TransportError, "legacy rejection"):
            conditional_physical_event(
                incident,
                80,
                0,
                replace(spec, compton_sampler="packed-uniform-rejection"),
                (0.2,) * 3,
                (0.8,) * 3,
                physical,
                draws,
            )
        with self.assertRaisesRegex(TransportError, "shell tables require"):
            terminal_physical_kernels(
                incident, (1, 0, 3), 80, 0, _spec(), (0.2,) * 3, (0.8,) * 3, physical
            )
        result = conditional_physical_event(
            incident, 80, 0, spec, (0.2,) * 3, (0.8,) * 3, physical, draws
        )
        self.assertEqual(result.channel, "bound-compton")
        self.assertIsNotNone(result.shell_index)
        assert (
            result.outgoing is not None
            and result.shell_index is not None
            and result.momentum is not None
        )
        cosine = dot(incident.direction, result.outgoing.direction)
        transfer = math.sqrt(80**2 + result.energy_kev**2 - 2 * 80 * result.energy_kev * cosine)
        self.assertAlmostEqual(
            result.energy_kev
            - 80
            + 80 * result.energy_kev * (1 - cosine) / ELECTRON_REST_ENERGY_KEV,
            result.momentum * transfer,
            delta=3e-11,
        )
        self.assertLessEqual(
            result.energy_kev, 80 - self.shells.shells[result.shell_index].binding_kev
        )
        self.assertAlmostEqual(result.energy_kev + result.deposited_kev, 80, places=13)
        endpoint = (1.4, 0.7, 3)
        mark = 75.0
        kernels = terminal_physical_kernels(
            incident,
            endpoint,
            80,
            0,
            spec,
            (0.2,) * 3,
            (0.8,) * 3,
            physical,
            bound_outgoing_kev=mark,
        )
        bound = next(item for item in kernels if item.channel == "bound-compton")
        self.assertEqual(
            (bound.measure, bound.energy_kev, bound.energy_dot), ("solid-angle-energy", mark, 0)
        )
        self.assertGreater(bound.density, 0)
        epsilon = 1e-6
        omega = cross(incident.direction, incident.direction_dot)
        densities = []
        for time in (-epsilon, epsilon):
            ray = RayTangent((0, 0, 0), _rotation(incident.direction, omega, time))
            density = next(
                item.density
                for item in terminal_physical_kernels(
                    ray,
                    endpoint,
                    80,
                    0,
                    spec,
                    (0.2,) * 3,
                    (0.8,) * 3,
                    physical,
                    bound_outgoing_kev=mark,
                )
                if item.channel == "bound-compton"
            )
            densities.append(density)
        self.assertAlmostEqual(
            bound.density_dot, (densities[1] - densities[0]) / (2 * epsilon), delta=1e-9
        )
        with self.assertRaisesRegex(TransportError, "explicit joint"):
            terminal_physical_kernels(
                incident, endpoint, 80, 0, spec, (0.2,) * 3, (0.8,) * 3, physical
            )
