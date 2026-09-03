"""Moving free-electron Compton energy boundaries on a planar detector.

The detector endpoint uses its attached unit-square chart, while the sampling
proposal uses a full azimuth around the incoming direction. Boundary samples
carry both physical-area and reference-area weights, so the detector Jacobian
is included exactly once when contracting a signed continuation jump.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from dpt.geometry import Vector3, cross, dot

from .geometry_path_measure import DetectorVertexTangent, detector_vertex_tangent
from .geometry_tangents import (
    RayTangent,
    RigidMotion,
    detector_corner_velocity,
    normalised_direction_jet,
)
from .model import ELECTRON_REST_ENERGY_KEV, OrientedDetector, PlanarDetector, TransportError


def _add(left: Vector3, right: Vector3) -> Vector3:
    return cast(Vector3, tuple(a + b for a, b in zip(left, right, strict=True)))


def _sub(left: Vector3, right: Vector3) -> Vector3:
    return cast(Vector3, tuple(a - b for a, b in zip(left, right, strict=True)))


def _scale(value: Vector3, factor: float) -> Vector3:
    return cast(Vector3, tuple(factor * component for component in value))


@dataclass(frozen=True, slots=True)
class ComptonEnergyBoundarySample:
    """One full-azimuth draw on the energy threshold, including signed motion.

    ``world_area_weight`` has units mm squared per active coordinate. Multiply
    it by the physical-area integrand's higher-energy minus lower-energy jump.
    ``reference_area_weight`` multiplies a unit-square integrand that already
    includes ``endpoint.area_jacobian``. Both weights include the full ``2*pi``
    proposal denominator. Null draws remain in the original sample count.

    ``outgoing`` and ``outgoing_energy_dot`` are evaluated at the fixed attached
    detector coordinate, rather than along the moving boundary proposal. The
    level-set derivative is ``outgoing_energy_dot - threshold_energy_dot``.
    """

    coordinate: tuple[float, float]
    endpoint: DetectorVertexTangent
    outgoing: RayTangent
    outgoing_energy_kev: float
    outgoing_energy_dot: float
    levelset_dot_kev: float
    world_area_weight: float
    reference_area_weight: float


@dataclass(frozen=True, slots=True)
class FreeComptonEnergyBoundary:
    """The regular level set ``free_Compton_energy - threshold_energy = 0``.

    ``incoming`` starts at the collision vertex and carries its object/source
    geometry jet. Energies are keV and energy tangents are keV per active
    coordinate. The positive continuation is the higher outgoing-energy side.
    A response window therefore adds its lower-threshold jump and subtracts its
    upper-threshold jump. The same interface accepts unequal continuation laws
    on the two sides of a physical eligibility threshold.

    Thresholds strictly outside the free-Compton energy range have no boundary.
    An active threshold at a forward/backscatter endpoint and an actively moving
    boundary coincident with an entire pixel edge require a one-sided chart.
    """

    incoming: RayTangent
    incident_energy_kev: float
    threshold_energy_kev: float
    incident_energy_dot: float = 0.0
    threshold_energy_dot: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.incoming), RayTangent):
            raise TransportError("Compton energy boundary requires an attached incoming ray")
        for value in (self.incident_energy_kev, self.threshold_energy_kev):
            if not math.isfinite(value) or value <= 0.0:
                raise TransportError("Compton boundary energies must be finite and positive")
        if not all(
            math.isfinite(value)
            for value in (
                self.incident_energy_dot,
                self.threshold_energy_dot,
            )
        ):
            raise TransportError("Compton boundary energy tangents must be finite")
        if not all(math.isfinite(value) for value in (self.cosine, self.cosine_dot)):
            raise TransportError("Compton boundary cosine and tangent must be representable")

    @property
    def cosine(self) -> float:
        """Cone cosine ``1-m*(1/L-1/E)`` with the package electron rest energy."""
        return 1.0 - (
            ELECTRON_REST_ENERGY_KEV
            * ((self.incident_energy_kev - self.threshold_energy_kev) / self.incident_energy_kev)
            / self.threshold_energy_kev
        )

    @property
    def cosine_dot(self) -> float:
        """Attached cone motion due to incident energy and threshold motion."""
        return ELECTRON_REST_ENERGY_KEV * (
            (self.threshold_energy_dot / self.threshold_energy_kev) / self.threshold_energy_kev
            - (self.incident_energy_dot / self.incident_energy_kev) / self.incident_energy_kev
        )

    def _admit_edges(
        self,
        detector: PlanarDetector | OrientedDetector,
        motion: RigidMotion,
        spacing_dot: tuple[float, float],
    ) -> None:
        # Only the right-angle cone intersects a transverse detector in a line.
        # A moving line coincident with a pixel edge has two unequal side limits.
        if self.cosine != 0.0:
            return
        k, k_dot = self.incoming.direction, self.incoming.direction_dot
        corner = detector.lower_corner_mm
        corner_dot = detector_corner_velocity(detector, motion, spacing_dot)
        for axis, other in ((0, 1), (1, 0)):
            bases = (detector.u, detector.v)
            edge_direction = _scale(
                bases[other], detector.shape[other] * detector.spacing_xy_mm[other]
            )
            if dot(k, edge_direction) != 0.0:
                continue
            edge_direction_dot = _add(
                motion.direction_velocity(edge_direction),
                _scale(bases[other], detector.shape[other] * spacing_dot[other]),
            )
            for index in range(detector.shape[axis] + 1):
                shift = _scale(bases[axis], index * detector.spacing_xy_mm[axis])
                first = _add(corner, shift)
                displacement = _sub(first, self.incoming.origin_mm)
                if dot(k, displacement) != 0.0:
                    continue
                first_dot = _add(
                    corner_dot,
                    _add(
                        motion.direction_velocity(shift),
                        _scale(bases[axis], index * spacing_dot[axis]),
                    ),
                )
                displacement_dot = _sub(first_dot, self.incoming.origin_dot)
                first_numerator_dot = dot(k_dot, displacement) + dot(k, displacement_dot)
                edge_numerator_dot = dot(k_dot, edge_direction) + dot(k, edge_direction_dot)
                if (
                    self.cosine_dot != 0.0
                    or first_numerator_dot != 0.0
                    or edge_numerator_dot != 0.0
                ):
                    raise TransportError(
                        "a moving Compton energy boundary coincides with a pixel edge"
                    )

    def sample(
        self,
        coordinate: float,
        detector: PlanarDetector | OrientedDetector,
        motion: RigidMotion | None = None,
        *,
        spacing_dot: tuple[float, float] = (0.0, 0.0),
    ) -> ComptonEnergyBoundarySample | None:
        """Sample full azimuth without conditioning on detector acceptance.

        For ``c = incoming.direction dot outgoing.direction``, solid angle is
        ``dc dphi`` and detector area is ``r**2/(normal dot outgoing) dc dphi``.
        Thus the exact coarea weight is ``2*pi*r**2*g_dot/(normal dot outgoing)``
        for ``g=c-c_threshold``. No finite difference or coordinate displacement
        is used. A missed or backward detector intersection returns ``None``.
        """
        if not math.isfinite(coordinate) or not 0.0 < coordinate < 1.0:
            raise TransportError("Compton boundary draw must lie strictly inside (0,1)")
        if len(spacing_dot) != 2 or not all(math.isfinite(value) for value in spacing_dot):
            raise TransportError("detector spacing tangent must have two finite components")
        motion = RigidMotion() if motion is None else motion
        cosine, cosine_dot = self.cosine, self.cosine_dot
        if abs(cosine) > 1.0:
            return None
        if abs(cosine) == 1.0:
            if cosine_dot != 0.0:
                raise TransportError("an active Compton energy endpoint requires a one-sided chart")
            return None
        height = dot(_sub(detector.lower_corner_mm, self.incoming.origin_mm), detector.normal)
        if height == 0.0:
            raise TransportError("Compton collision vertex lies in the detector plane")
        if height < 0.0:
            return None
        self._admit_edges(detector, motion, spacing_dot)
        k = self.incoming.direction
        anchor: Vector3 = (0.0, 0.0, 1.0) if abs(k[2]) < 0.9 else (1.0, 0.0, 0.0)
        first = cross(k, anchor)
        first = _scale(first, 1.0 / math.hypot(*first))
        second = cross(k, first)
        phi, radius = 2.0 * math.pi * coordinate, math.sqrt((1.0 - cosine) * (1.0 + cosine))
        direction = _add(
            _scale(k, cosine),
            _add(
                _scale(first, radius * math.cos(phi)),
                _scale(second, radius * math.sin(phi)),
            ),
        )
        projection = dot(direction, detector.normal)
        if projection <= 0.0:
            return None
        distance = height / projection
        if not math.isfinite(distance) or distance <= 0.0:
            raise TransportError("Compton boundary plane intersection is not representable")
        point = _add(self.incoming.origin_mm, _scale(direction, distance))
        offset = _sub(point, detector.lower_corner_mm)
        uv = cast(
            tuple[float, float],
            tuple(
                dot(offset, basis) / (count * spacing)
                for basis, count, spacing in zip(
                    (detector.u, detector.v), detector.shape, detector.spacing_xy_mm, strict=True
                )
            ),
        )
        if not all(math.isfinite(value) for value in uv):
            raise TransportError("Compton boundary detector coordinate is not representable")
        if not all(0.0 < value < 1.0 for value in uv):
            return None
        endpoint = detector_vertex_tangent(uv, detector, motion, spacing_dot)
        displacement = _sub(endpoint.point_mm, self.incoming.origin_mm)
        displacement_dot = _sub(endpoint.point_dot, self.incoming.origin_dot)
        outgoing, outgoing_dot = normalised_direction_jet(displacement, displacement_dot)
        actual_cosine = dot(k, outgoing)
        actual_cosine_dot = dot(self.incoming.direction_dot, outgoing) + dot(k, outgoing_dot)
        inverse_energy = (
            1.0 / self.incident_energy_kev + (1.0 - actual_cosine) / ELECTRON_REST_ENERGY_KEV
        )
        energy = 1.0 / inverse_energy
        energy_dot = (
            energy
            * energy
            * (
                (self.incident_energy_dot / self.incident_energy_kev) / self.incident_energy_kev
                + actual_cosine_dot / ELECTRON_REST_ENERGY_KEV
            )
        )
        g_dot = actual_cosine_dot - cosine_dot
        distance = math.hypot(*displacement)
        world_weight = (2.0 * math.pi * distance) * (distance * g_dot / projection)
        reference_weight = world_weight / endpoint.area_jacobian
        levelset_dot = energy_dot - self.threshold_energy_dot
        values = (energy, energy_dot, levelset_dot, world_weight, reference_weight)
        if not all(math.isfinite(value) for value in values):
            raise TransportError("Compton energy boundary contribution is not representable")
        return ComptonEnergyBoundarySample(
            uv,
            endpoint,
            RayTangent(self.incoming.origin_mm, outgoing, self.incoming.origin_dot, outgoing_dot),
            energy,
            energy_dot,
            levelset_dot,
            world_weight,
            reference_weight,
        )


__all__ = ["ComptonEnergyBoundarySample", "FreeComptonEnergyBoundary"]
