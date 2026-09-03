"""Complete analogue reference histories for geometry continuation measures.

Geometry uses the physical transport expectation independently of a forward
variance-reduction policy. Every continuation is one actual photon history,
including all radiative channels and fixed source-energy marks. A finite
resource guard invalidates its parent observation instead of returning zero.
The caller supplies a fully addressed random design; this module invents no
coordinates and keeps the original-history lineage attached to every future.
"""

from __future__ import annotations

# Host geometry components share their validated collision/table internals.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from dpt.geometry import dot

from .geometry_flights import material_box_interval, optical_flight_tangent
from .geometry_tangents import RayTangent, detector_hit_tangent
from .model import TransportError, TransportSpec
from .physical import PhysicalTransport
from .physical_geometry_events import (
    AddressedUniform,
    PhysicalCollisionJet,
    PhysicalEventDraws,
    conditional_physical_event,
)


class GeometryUniforms(Protocol):
    """An explicit complete design addressed by original history and lineage.

    ``lineage`` identifies the original-history submeasurement, never a new
    independent observation for confidence reduction. Event/domain/lane retain
    the physical transport counters. Implementations bind their generator,
    seed, full geometry-coordinate layout and independent randomisation in
    ``identity``; QMC designs must retain null draws and all auxiliary futures.
    """

    identity: str
    layout: Literal["legacy-v1", "packed-compton-v2", "bound-compton-v3"]

    def uniform(self, lineage: tuple[int, ...], event: int, domain: int, lane: int) -> float: ...


def geometry_uniform(
    design: GeometryUniforms,
    lineage: tuple[int, ...],
    event: int,
    domain: int,
    lane: int,
) -> float:
    value = design.uniform(lineage, event, domain, lane)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise TransportError("geometry random coordinates must be finite open-interval uniforms")
    return value


@dataclass(frozen=True, slots=True)
class GeometryPhysicalMedium:
    """Material-major coefficients and explicit cell density/material ownership."""

    material_ids: tuple[int, ...]
    absorption_mm_inverse: tuple[float, ...]
    scattering_mm_inverse: tuple[float, ...]
    density: tuple[float, ...]
    voxel_density: tuple[float, ...]
    physical: PhysicalTransport | None = None

    def __post_init__(self) -> None:
        for field in (
            "material_ids",
            "absorption_mm_inverse",
            "scattering_mm_inverse",
            "density",
            "voxel_density",
        ):
            object.__setattr__(self, field, tuple(getattr(self, field)))

    def validate(self, spec: TransportSpec) -> None:
        cells, materials, energies = (
            math.prod(spec.grid.shape),
            spec.grid.materials,
            len(spec.energy_nodes),
        )
        if len(self.material_ids) != cells or any(
            type(v) is not int or not 0 <= v < materials for v in self.material_ids
        ):
            raise TransportError("geometry material IDs must be one valid integer per cell")
        if len(self.density) != materials or len(self.voxel_density) != cells:
            raise TransportError("geometry density arrays must match material and cell ownership")
        if (
            len(self.absorption_mm_inverse) != materials * energies
            or len(self.scattering_mm_inverse) != materials * energies
        ):
            raise TransportError(
                "geometry coefficients require material-major energy-fastest tables"
            )
        if any(
            not math.isfinite(v) or v < 0
            for v in (*self.absorption_mm_inverse, *self.scattering_mm_inverse, *self.voxel_density)
        ) or any(not math.isfinite(v) or v <= 0 for v in self.density):
            raise TransportError(
                "geometry coefficients/voxel densities must be nonnegative "
                "and material density positive"
            )
        if spec.scattering_law == "bound-electron-compton" and self.physical is None:
            raise TransportError("bound geometry requires its explicit shell/Doppler model")

    def factor(self, cell: int) -> float:
        return self.density[self.material_ids[cell]] * self.voxel_density[cell]

    def extinction(self, energy_kev: float, spec: TransportSpec) -> tuple[float, ...]:
        return self.extinction_jet(energy_kev, 0.0, spec)[0]

    def extinction_jet(
        self, energy_kev: float, energy_dot: float, spec: TransportSpec
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        nodes = spec.energy_nodes
        if not math.isfinite(energy_kev) or not nodes[0] <= energy_kev <= nodes[-1]:
            raise TransportError("geometry photon energy is outside coefficient support")
        if energy_dot != 0.0 and energy_kev in nodes[1:-1]:
            raise TransportError("active energy lies on a coefficient interpolation kink")
        lower = 0
        while lower < len(nodes) - 2 and energy_kev > nodes[lower + 1]:
            lower += 1
        fraction = (
            0.0
            if len(nodes) == 1
            else (energy_kev - nodes[lower]) / (nodes[lower + 1] - nodes[lower])
        )
        material_rates: list[float] = []
        material_dots: list[float] = []
        for material in range(spec.grid.materials):
            first = material * len(nodes) + lower
            absorption = self.absorption_mm_inverse[first]
            scattering = self.scattering_mm_inverse[first]
            rate = absorption + scattering
            slope = 0.0
            if len(nodes) > 1:
                following = (
                    self.absorption_mm_inverse[first + 1] + self.scattering_mm_inverse[first + 1]
                )
                slope = (following - rate) / (nodes[lower + 1] - nodes[lower])
                # Match the canonical two independent coefficient interpolants.
                # The subtract/add form loses the exact high node when the
                # lower coefficient is much larger than the following one.
                rate = (
                    (1.0 - fraction) * absorption + fraction * self.absorption_mm_inverse[first + 1]
                ) + (
                    (1.0 - fraction) * scattering + fraction * self.scattering_mm_inverse[first + 1]
                )
            material_rates.append(rate)
            material_dots.append(slope * energy_dot)
        return (
            tuple(self.factor(c) * material_rates[m] for c, m in enumerate(self.material_ids)),
            tuple(self.factor(c) * material_dots[m] for c, m in enumerate(self.material_ids)),
        )


@dataclass(frozen=True, slots=True)
class PhysicalGeometryPrefix:
    """One reached real collision before selecting its outgoing physical channel."""

    incoming: RayTangent
    energy_kev: float
    cell: int
    event: int
    previous_orders: int


@dataclass(frozen=True, slots=True)
class CompleteGeometryHistory:
    """All collision prefixes and the complete signed escaped detector response."""

    prefixes: tuple[PhysicalGeometryPrefix, ...]
    response: float
    terminal: Literal["escaped", "absorbed", "local-deposition"]
    energy_kev: float
    lineage: tuple[int, ...]
    random_identity: str
    used_uniforms: tuple[AddressedUniform, ...]


def physical_geometry_draws(
    design: GeometryUniforms,
    lineage: tuple[int, ...],
    event: int,
    spec: TransportSpec,
) -> PhysicalEventDraws:
    """Supply potential canonical coordinates without conditional-address reuse."""
    records: list[AddressedUniform] = []
    addresses = [(0, lane) for lane in range(4)] + [(2, lane) for lane in range(3)] + [(3, 0)]
    if spec.scattering_law == "bound-electron-compton":
        addresses.extend((0x10000000, lane) for lane in (0, 1))
    for trial in range(spec.max_angle_trials):
        addresses.extend((0xC0000000 + trial, lane) for lane in range(3))
        if spec.compton_sampler == "packed-uniform-rejection":
            if trial % 2 == 0:
                addresses.extend((0x80000000 + trial // 2, lane) for lane in range(4))
        else:
            addresses.extend((0x80000000 + trial, lane) for lane in range(3))
    for domain, lane in addresses:
        records.append(
            AddressedUniform(
                event, domain, lane, geometry_uniform(design, lineage, event, domain, lane)
            )
        )
    return PhysicalEventDraws(
        event, 0, design.layout, tuple(records), f"{design.identity}; lineage={lineage}"
    )


def conditional_geometry_collision(
    prefix: PhysicalGeometryPrefix,
    spec: TransportSpec,
    medium: GeometryPhysicalMedium,
    design: GeometryUniforms,
    lineage: tuple[int, ...],
) -> PhysicalCollisionJet:
    """Continue the actual full conditional collision law at an explicit material."""
    event = conditional_physical_event(
        prefix.incoming,
        prefix.energy_kev,
        medium.material_ids[prefix.cell],
        replace(spec, estimator="analogue"),
        medium.absorption_mm_inverse,
        medium.scattering_mm_inverse,
        medium.physical,
        physical_geometry_draws(design, lineage, prefix.event, spec),
        density_factor=medium.factor(prefix.cell),
    )
    if event.status == "angle-budget":
        raise TransportError("angular guard invalidates the entire geometry observation")
    return event


def escaped_geometry_response(
    ray: RayTangent, energy_kev: float, spec: TransportSpec, cotangent: tuple[float, ...]
) -> float:
    """Score one complete escaping photon with the signed caller pixel seed."""
    if dot(ray.direction, spec.detector.normal) <= 0.0:
        return 0.0
    hit = detector_hit_tangent(RayTangent(ray.origin_mm, ray.direction), spec.detector)
    if hit.pixel < 0:
        return 0.0
    value = cotangent[hit.pixel] * (energy_kev if spec.scoring == "energy-kev" else 1.0)
    if not math.isfinite(value):
        raise TransportError("complete escaped geometry response is not representable in binary64")
    return value


def _material_entry_exists(
    ray: RayTangent, spec: TransportSpec, side: tuple[int, int, Literal[-1, 1]] | None
) -> bool:
    interval = material_box_interval(ray, spec, side=side)
    inverse = spec.object_pose.inverse()
    origin, direction = inverse.point(ray.origin_mm), inverse.direction(ray.direction)
    half_open_miss = any(
        direction[axis] == 0.0
        and origin[axis] >= spec.grid.upper_mm[axis]
        and (side is None or side[0] != axis)
        for axis in range(3)
    )
    return interval is not None and not half_open_miss


def complete_geometry_history(
    initial: RayTangent,
    energy_kev: float,
    spec: TransportSpec,
    medium: GeometryPhysicalMedium,
    design: GeometryUniforms,
    *,
    lineage: tuple[int, ...],
    cotangent: tuple[float, ...],
    initial_material_side: tuple[int, int, Literal[-1, 1]] | None = None,
    forced_collision_cell: int | None = None,
    initial_event: int = 0,
    previous_orders: int = 0,
) -> CompleteGeometryHistory:
    """Run all physical futures using a detached analogue reference law.

    The initial material-side override is exact. It remains attached while a
    sampled outgoing ray stays parallel to that plane, and clears when the ray
    leaves it. Later off-plane directions use the ordinary physical law.
    Forced collisions start at a signed generator interface with its explicit
    cell label. Absorption and the declared cutoff are physical terminal states;
    event/crossing/angular/energy/numerical failures abort the parent history.
    """
    medium.validate(spec)
    if len(cotangent) != spec.detector.pixels or any(not math.isfinite(v) for v in cotangent):
        raise TransportError("geometry cotangent must be finite with one entry per pixel")
    if not design.identity.strip():
        raise TransportError("geometry histories require explicit random-design provenance")
    if not math.isfinite(energy_kev) or energy_kev <= 0.0:
        raise TransportError("geometry history energy must be finite and positive")
    if forced_collision_cell is not None and (
        type(forced_collision_cell) is not int
        or not 0 <= forced_collision_cell < len(medium.material_ids)
    ):
        raise TransportError("forced geometry collision must belong to a valid material cell")
    ray = RayTangent(initial.origin_mm, initial.direction)
    energy = energy_kev
    prefixes: list[PhysicalGeometryPrefix] = []
    used: list[AddressedUniform] = []
    event_number = initial_event
    cell = forced_collision_cell
    while True:
        if medium.physical is not None and energy <= medium.physical.cutoff_kev:
            # Canonical transport applies its local deposition rule after
            # material entry. A ray which misses the box still escapes; a
            # reached/forced collision deposits at its actual material point.
            missed = False
            if cell is None and not prefixes:
                missed = not _material_entry_exists(ray, spec, initial_material_side)
            if missed:
                return CompleteGeometryHistory(
                    (),
                    escaped_geometry_response(ray, energy, spec, cotangent),
                    "escaped",
                    energy,
                    lineage,
                    design.identity,
                    tuple(used),
                )
            return CompleteGeometryHistory(
                tuple(prefixes),
                0.0,
                "local-deposition",
                energy,
                lineage,
                design.identity,
                tuple(used),
            )
        if event_number >= spec.max_events:
            raise TransportError("event guard invalidates the entire geometry observation")
        if cell is None:
            if not _material_entry_exists(ray, spec, initial_material_side):
                return CompleteGeometryHistory(
                    tuple(prefixes),
                    escaped_geometry_response(ray, energy, spec, cotangent),
                    "escaped",
                    energy,
                    lineage,
                    design.identity,
                    tuple(used),
                )
            draw = geometry_uniform(design, lineage, event_number, 0, 0)
            used.append(AddressedUniform(event_number, 0, 0, draw))
            flight = optical_flight_tangent(
                ray,
                spec,
                extinction_mm_inverse=medium.extinction(energy, spec),
                optical_depth=-math.log(draw),
                material_side=initial_material_side,
            )
            if flight.terminal == "escape":
                response = escaped_geometry_response(ray, energy, spec, cotangent)
                return CompleteGeometryHistory(
                    tuple(prefixes),
                    response,
                    "escaped",
                    energy,
                    lineage,
                    design.identity,
                    tuple(used),
                )
            assert flight.cell is not None
            cell, ray = flight.cell, RayTangent(flight.ray.origin_mm, flight.ray.direction)
        prefix = PhysicalGeometryPrefix(
            ray, energy, cell, event_number, previous_orders + len(prefixes)
        )
        prefixes.append(prefix)
        collision = conditional_geometry_collision(prefix, spec, medium, design, lineage)
        used.extend(collision.used_uniforms)
        if collision.status == "angle-budget":
            raise TransportError("angular guard invalidates the entire geometry observation")
        if collision.status != "continuing":
            return CompleteGeometryHistory(
                tuple(prefixes),
                0.0,
                collision.status,
                energy,
                lineage,
                design.identity,
                tuple(used),
            )
        assert collision.outgoing is not None
        if initial_material_side is not None:
            axis, _, _ = initial_material_side
            if spec.object_pose.inverse().direction(collision.outgoing.direction)[axis] != 0.0:
                initial_material_side = None
        ray, energy = (
            RayTangent(collision.outgoing.origin_mm, collision.outgoing.direction),
            collision.energy_kev,
        )
        event_number += 1
        cell = None


__all__ = [
    "CompleteGeometryHistory",
    "GeometryPhysicalMedium",
    "GeometryUniforms",
    "PhysicalGeometryPrefix",
    "complete_geometry_history",
    "conditional_geometry_collision",
    "escaped_geometry_response",
    "geometry_uniform",
    "physical_geometry_draws",
]
