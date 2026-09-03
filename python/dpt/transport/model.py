"""Host-only contracts for fixed-grid photon transport estimators.

Supplied linear coefficients in inverse millimetres define absorption and
aggregate scattering in piecewise-constant material cells. The declared
scattering law selects isotropic elastic, free-electron Klein-Nishina/Compton
or bound-electron impulse scattering. ``PhysicalTransport`` supplies occupied
shell data for the bound-electron law and can also bind coherent Rayleigh
scattering, source energies, fluorescence and a fixed Woodcock majorant.
Energy-changing histories use explicit coefficient-energy support. Photon
energies and local deposition are in keV; polarisation is outside these laws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import IntEnum
from itertools import pairwise
from typing import Any, Literal, cast

from dpt.contracts import ContractError, finite_scalar
from dpt.geometry import IDENTITY, DetectorGeometry, RigidTransform, cross, dot
from dpt.volumes import GridSpec

ELECTRON_REST_ENERGY_KEV = 510.99895069


class TransportError(ContractError):
    """Compatibility subtype for transport physical, storage and execution contracts."""


class IncompleteHistoryError(RuntimeError):
    """A resource guard stopped a history; the entire estimate is invalid."""


class HistoryStatus(IntEnum):
    """Each launched history writes exactly one terminal status."""

    ESCAPED = 0
    ABSORBED = 1
    EVENT_BUDGET = 2
    CROSSING_BUDGET = 3
    NUMERICAL_FAILURE = 4
    ENERGY_SUPPORT = 5
    ANGLE_BUDGET = 6


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or not 0 < value < 2**31:
        raise TransportError(f"{name} must be a positive signed 32-bit integer")


def _physical(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        scalar = finite_scalar(value, name, minimum=0.0 if positive or nonnegative else None)
    except ContractError as error:
        raise TransportError(str(error)) from error
    if positive and scalar == 0:
        raise TransportError(f"{name} must be positive")
    return scalar


def _freeze_fields(instance: Any, names: tuple[str, ...]) -> None:
    for name in names:
        object.__setattr__(instance, name, tuple(getattr(instance, name)))


def _require_instance(value: Any, expected: type[Any], name: str) -> None:
    if not isinstance(value, expected):
        raise TransportError(f"{name} must be a {expected.__name__}")


@dataclass(frozen=True, slots=True)
class MaterialGrid:
    """Axis-aligned cell faces; shape is (nz, ny, nx), device indices x fastest."""

    origin_mm: tuple[float, float, float]
    spacing_mm: tuple[float, float, float]
    shape: tuple[int, int, int]
    materials: int

    @classmethod
    def from_grid_spec(cls, grid: GridSpec, *, materials: int) -> MaterialGrid:
        """Match an axis-aligned sample grid's support with constant material cells.

        This converts metadata, without converting a trilinear attenuation field
        into labels or altering its physics. The first face lies half a spacing
        below the first sample centre. Oriented grids require a ray transform.
        """
        if grid.orientation != IDENTITY:
            raise TransportError("MaterialGrid conversion requires an axis-aligned GridSpec")
        origin = cast(
            tuple[float, float, float],
            tuple(
                centre - step / 2
                for centre, step in zip(grid.origin_mm, grid.spacing_mm, strict=True)
            ),
        )
        return cls(origin, grid.spacing_mm, grid.shape, materials)

    def to_grid_spec(self) -> GridSpec:
        """Return sample-centre metadata with the same piecewise-constant support."""
        origin = cast(
            tuple[float, float, float],
            tuple(
                face + step / 2 for face, step in zip(self.origin_mm, self.spacing_mm, strict=True)
            ),
        )
        return GridSpec(self.shape, self.spacing_mm, origin)

    @classmethod
    def from_oriented_grid_spec(
        cls, grid: GridSpec, *, materials: int
    ) -> tuple[MaterialGrid, RigidTransform]:
        """Return local cells and their object pose matching an oriented sample grid.

        Pass the returned pose as ``TransportSpec.object_pose``. Cells remain
        axis-aligned in their local frame; no resampling or label interpolation
        is performed.
        """
        origin = cast(tuple[float, float, float], tuple(-step / 2 for step in grid.spacing_mm))
        return cls(origin, grid.spacing_mm, grid.shape, materials), RigidTransform(
            grid.orientation, grid.origin_mm
        )

    def __post_init__(self) -> None:
        _freeze_fields(self, ("origin_mm", "spacing_mm", "shape"))
        if len(self.origin_mm) != 3 or len(self.spacing_mm) != 3 or len(self.shape) != 3:
            raise TransportError("origin, spacing and shape must each have three components")
        _positive_integer(self.materials, "materials")
        for count, step, origin in zip(
            self.shape_xyz, self.spacing_mm, self.origin_mm, strict=True
        ):
            _positive_integer(count, "grid extent")
            _physical(step, "cell spacing", positive=True)
            _physical(origin, "cell origin")
            endpoint = origin + count * step
            if (
                not math.isfinite(endpoint)
                or origin + step == origin
                or endpoint - step == endpoint
            ):
                raise TransportError("cell faces must remain distinguishable in binary64")
        if math.prod(self.shape) >= 2**31:
            raise TransportError("the flattened material grid exceeds signed 32-bit indexing")

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        """Geometric extents for device vectors; storage shape remains (z,y,x)."""
        return self.shape[2], self.shape[1], self.shape[0]

    @property
    def upper_mm(self) -> tuple[float, float, float]:
        return cast(
            tuple[float, float, float],
            tuple(
                start + count * step
                for start, count, step in zip(
                    self.origin_mm, self.shape_xyz, self.spacing_mm, strict=True
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class PlanarDetector:
    """An outward-facing z plane, pixel lower faces (x,y), and count score.

    Pixel support is half-open. The detector must be above the entire material
    grid; an escaped straight ray can therefore hit it at most once. A recorded
    score includes both uncollided and scattered photons. Do not add a separate
    deterministic primary image to this estimate.
    """

    lower_xy_mm: tuple[float, float]
    spacing_xy_mm: tuple[float, float]
    shape: tuple[int, int]  # width, height
    z_mm: float

    @classmethod
    def from_detector_geometry(cls, detector: DetectorGeometry) -> PlanarDetector:
        """Convert an increasing-x/y z plane from pixel centres to lower faces."""
        if detector.u != (1.0, 0.0, 0.0) or detector.v != (0.0, 1.0, 0.0):
            raise TransportError("PlanarDetector conversion requires positive x/y detector axes")
        return cls(
            (
                detector.origin_mm[0] - detector.spacing_mm[0] / 2,
                detector.origin_mm[1] - detector.spacing_mm[1] / 2,
            ),
            detector.spacing_mm,
            (detector.shape[1], detector.shape[0]),
            detector.origin_mm[2],
        )

    def to_detector_geometry(self, *, source_mm: tuple[float, float, float]) -> DetectorGeometry:
        """Return matching pixel-centre metadata with an explicit point source."""
        return DetectorGeometry(
            source_mm,
            (
                self.lower_xy_mm[0] + self.spacing_xy_mm[0] / 2,
                self.lower_xy_mm[1] + self.spacing_xy_mm[1] / 2,
                self.z_mm,
            ),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            self.spacing_xy_mm,
            self.image_shape,
        )

    def __post_init__(self) -> None:
        _freeze_fields(self, ("lower_xy_mm", "spacing_xy_mm", "shape"))
        _physical(self.z_mm, "detector z")
        if any(len(value) != 2 for value in (self.lower_xy_mm, self.spacing_xy_mm, self.shape)):
            raise TransportError("detector xy fields must have two components")
        for count, step, lower in zip(
            self.shape, self.spacing_xy_mm, self.lower_xy_mm, strict=True
        ):
            _positive_integer(count, "detector extent")
            _physical(step, "pixel spacing", positive=True)
            _physical(lower, "detector origin")
            end = lower + count * step
            if not math.isfinite(end):
                raise TransportError("detector endpoint must be finite")
            if lower + step == lower or end - step == end:
                raise TransportError("detector pixels must be distinguishable in binary64")
        if self.pixels >= 2**31:
            raise TransportError("the flattened detector exceeds signed 32-bit indexing")

    @property
    def pixels(self) -> int:
        return math.prod(self.shape)

    @property
    def width(self) -> int:
        return self.shape[0]

    @property
    def height(self) -> int:
        return self.shape[1]

    @property
    def image_shape(self) -> tuple[int, int]:
        """The (rows, columns) shape for reshaping a flattened row-major tally."""
        return self.height, self.width

    @property
    def lower_corner_mm(self) -> tuple[float, float, float]:
        return (*self.lower_xy_mm, self.z_mm)

    @property
    def u(self) -> tuple[float, float, float]:
        return (1.0, 0.0, 0.0)

    @property
    def v(self) -> tuple[float, float, float]:
        return (0.0, 1.0, 0.0)

    @property
    def normal(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 1.0)

    def source_distance(self, position: tuple[float, float, float]) -> float:
        """Signed source-side distance; valid launch positions have negative values."""
        return position[2] - self.z_mm


@dataclass(frozen=True, slots=True)
class OrientedDetector:
    """An existing detector geometry with half-open pixel footprints.

    ``geometry.source_mm`` chooses the incident side of the plane; the fixed
    source sampler may use any positions on that side. This metadata retains
    pixel-centre and row/column conventions from DetectorGeometry.
    """

    geometry: DetectorGeometry

    def __post_init__(self) -> None:
        _require_instance(self.geometry, DetectorGeometry, "OrientedDetector.geometry")
        corner = self.lower_corner_mm
        for column in (0, self.shape[0]):
            for row in (0, self.shape[1]):
                point = tuple(
                    corner[i]
                    + column * self.spacing_xy_mm[0] * self.u[i]
                    + row * self.spacing_xy_mm[1] * self.v[i]
                    for i in range(3)
                )
                if not all(math.isfinite(value) for value in point):
                    raise TransportError("detector footprint corners must be finite")
        for axis, spacing in ((self.u, self.spacing_xy_mm[0]), (self.v, self.spacing_xy_mm[1])):
            if all(corner[i] + spacing * axis[i] == corner[i] for i in range(3)):
                raise TransportError("detector pixels must be distinguishable in binary64")

    @property
    def lower_corner_mm(self) -> tuple[float, float, float]:
        g = self.geometry
        return cast(
            tuple[float, float, float],
            tuple(
                g.origin_mm[i] - 0.5 * (g.spacing_mm[0] * g.u[i] + g.spacing_mm[1] * g.v[i])
                for i in range(3)
            ),
        )

    @property
    def u(self) -> tuple[float, float, float]:
        return self.geometry.u

    @property
    def v(self) -> tuple[float, float, float]:
        return self.geometry.v

    @property
    def normal(self) -> tuple[float, float, float]:
        normal = cross(self.u, self.v)
        offset = cast(
            tuple[float, float, float],
            tuple(self.geometry.source_mm[i] - self.geometry.origin_mm[i] for i in range(3)),
        )
        return (
            cast(tuple[float, float, float], tuple(-x for x in normal))
            if dot(offset, normal) > 0
            else normal
        )

    def source_distance(self, position: tuple[float, float, float]) -> float:
        offset = cast(
            tuple[float, float, float],
            tuple(position[i] - self.lower_corner_mm[i] for i in range(3)),
        )
        return dot(offset, self.normal)

    @property
    def spacing_xy_mm(self) -> tuple[float, float]:
        return self.geometry.spacing_mm

    @property
    def shape(self) -> tuple[int, int]:
        return self.geometry.shape[1], self.geometry.shape[0]

    @property
    def pixels(self) -> int:
        return self.geometry.pixels

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.geometry.shape


@dataclass(frozen=True, slots=True)
class TransportExecution:
    """Finite history guards, launch size and coefficient-support validation.

    A guard failure invalidates the estimate. Changing a guard changes allowed
    resource use without converting incomplete histories to physical outcomes.
    ``cover-budget`` checks coefficient support against the selected event guard.
    ``compton_sampler`` defaults to the retained per-proposal Philox rejection
    sampler. ``packed-uniform-rejection`` tests two Klein-Nishina proposals per
    random block and takes the independent azimuth from collision lane two. It
    preserves the scattering distribution and proposal budget while changing
    sample coordinates, so it is an explicit execution policy. QMC execution
    requires ``SamplingDesign(..., layout="packed-compton-v2")``.

    Example::

        execution = TransportExecution(max_events=32, max_angle_trials=32,
            compton_sampler="packed-uniform-rejection")
        assert execution.max_angle_trials == 32

    Kinematics and the angular density are documented in the `Geant4 Compton
    reference <https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/compton.html>`_.
    The production implementation is ``dpt.transport.kernels._make_walk``.
    """

    max_angle_trials: int = 4096
    max_events: int = 4096
    max_crossings: int = 65536
    block_dim: Literal[64, 128, 256] = 128
    energy_support_policy: Literal["runtime-guard", "cover-budget"] = "runtime-guard"
    compton_sampler: Literal["legacy-rejection", "packed-uniform-rejection"] = "legacy-rejection"

    def __post_init__(self) -> None:
        for name in ("max_angle_trials", "max_events", "max_crossings"):
            _positive_integer(getattr(self, name), name)
        if type(self.block_dim) is not int or self.block_dim not in (64, 128, 256):
            raise TransportError("block_dim must be 64, 128 or 256")
        if self.energy_support_policy not in ("runtime-guard", "cover-budget"):
            raise TransportError("unsupported coefficient energy support policy")
        if self.compton_sampler not in ("legacy-rejection", "packed-uniform-rejection"):
            raise TransportError("unsupported Compton sampler execution policy")


@dataclass(frozen=True, slots=True)
class TransportPhysics:
    """Material cells, detector, source energy and supplied scattering model.

    Resource guards and launch sizes belong to ``TransportExecution``. Source
    samples and nonnegative weights remain explicit caller-owned inputs.
    """

    grid: MaterialGrid
    detector: PlanarDetector | OrientedDetector
    energy_kev: float
    coefficient_provenance: str
    coefficient_energies_kev: tuple[float, ...] = ()
    scattering_law: Literal[
        "isotropic-elastic", "free-electron-compton", "bound-electron-compton"
    ] = "isotropic-elastic"
    scoring: Literal["photon-count", "energy-kev"] = "photon-count"
    estimator: Literal["analogue", "continuous-absorption"] = "analogue"
    object_pose: RigidTransform = field(default_factory=RigidTransform)

    def __post_init__(self) -> None:
        _freeze_fields(self, ("coefficient_energies_kev",))
        TransportSpec.from_physics(self)


@dataclass(frozen=True, slots=True)
class TransportSpec:
    """A complete physical scope and finite launch-resource policy.

    Source rays and nonnegative importance weights are supplied by the caller.
    Their sampling law must be fixed with respect to active material parameters;
    the library does not silently invent an emission distribution. Provenance
    identifies the user's coefficient source, not a fabricated material asset.
    """

    grid: MaterialGrid
    detector: PlanarDetector | OrientedDetector
    energy_kev: float
    coefficient_provenance: str
    coefficient_energies_kev: tuple[float, ...] = ()
    scattering_law: Literal[
        "isotropic-elastic", "free-electron-compton", "bound-electron-compton"
    ] = "isotropic-elastic"
    scoring: Literal["photon-count", "energy-kev"] = "photon-count"
    max_angle_trials: int = 4096
    max_events: int = 4096
    max_crossings: int = 65536
    block_dim: Literal[64, 128, 256] = 128
    estimator: Literal["analogue", "continuous-absorption"] = "analogue"
    energy_support_policy: Literal["runtime-guard", "cover-budget"] = "runtime-guard"
    object_pose: RigidTransform = field(default_factory=RigidTransform)
    compton_sampler: Literal["legacy-rejection", "packed-uniform-rejection"] = "legacy-rejection"

    @classmethod
    def from_physics(
        cls, physics: TransportPhysics, execution: TransportExecution | None = None
    ) -> TransportSpec:
        """Adapt separated records without changing retained constructor order."""
        if not isinstance(cast(object, physics), TransportPhysics):
            raise TransportError("physics must be a TransportPhysics")
        if execution is not None and not isinstance(cast(object, execution), TransportExecution):
            raise TransportError("execution must be a TransportExecution")
        selected = TransportExecution() if execution is None else execution
        return cls(
            physics.grid,
            physics.detector,
            physics.energy_kev,
            physics.coefficient_provenance,
            physics.coefficient_energies_kev,
            physics.scattering_law,
            physics.scoring,
            selected.max_angle_trials,
            selected.max_events,
            selected.max_crossings,
            selected.block_dim,
            physics.estimator,
            selected.energy_support_policy,
            physics.object_pose,
            selected.compton_sampler,
        )

    @classmethod
    def resolve(
        cls, spec: TransportSpec | TransportPhysics, execution: TransportExecution | None = None
    ) -> TransportSpec:
        """Validate the actual selected guards and coefficient-support policy."""
        if execution is not None and not isinstance(cast(object, execution), TransportExecution):
            raise TransportError("execution must be a TransportExecution")
        if isinstance(spec, TransportPhysics):
            return cls.from_physics(spec, execution)
        if not isinstance(spec, cls):
            raise TransportError("spec must be TransportSpec or TransportPhysics")
        if execution is None:
            return spec
        return replace(
            spec,
            max_angle_trials=execution.max_angle_trials,
            max_events=execution.max_events,
            max_crossings=execution.max_crossings,
            block_dim=execution.block_dim,
            energy_support_policy=execution.energy_support_policy,
            compton_sampler=execution.compton_sampler,
        )

    @property
    def physics(self) -> TransportPhysics:
        return TransportPhysics(
            self.grid,
            self.detector,
            self.energy_kev,
            self.coefficient_provenance,
            self.coefficient_energies_kev,
            self.scattering_law,
            self.scoring,
            self.estimator,
            self.object_pose,
        )

    @property
    def execution(self) -> TransportExecution:
        return TransportExecution(
            self.max_angle_trials,
            self.max_events,
            self.max_crossings,
            self.block_dim,
            self.energy_support_policy,
            self.compton_sampler,
        )

    def __post_init__(self) -> None:
        _freeze_fields(self, ("coefficient_energies_kev",))
        _require_instance(self.object_pose, RigidTransform, "object_pose")
        _physical(self.energy_kev, "source energy", positive=True)
        for node in self.coefficient_energies_kev:
            _physical(node, "coefficient energy", nonnegative=True)
        if type(self.coefficient_provenance) is not str or not self.coefficient_provenance.strip():
            raise TransportError("the supplied coefficients require a provenance identifier")
        for x in (self.grid.origin_mm[0], self.grid.upper_mm[0]):
            for y in (self.grid.origin_mm[1], self.grid.upper_mm[1]):
                for z in (self.grid.origin_mm[2], self.grid.upper_mm[2]):
                    distance = self.detector.source_distance(self.object_pose.point((x, y, z)))
                    if not math.isfinite(distance) or distance >= 0:
                        raise TransportError(
                            "the detector plane must lie above/on the outgoing side "
                            "of the entire material grid"
                        )
        nodes = self.energy_nodes
        if any(right <= left for left, right in pairwise(nodes)):
            raise TransportError("coefficient energies must be strictly increasing")
        if not nodes[0] <= self.energy_kev <= nodes[-1]:
            raise TransportError("coefficient energies must contain the source energy")
        if self.scattering_law not in (
            "isotropic-elastic",
            "free-electron-compton",
            "bound-electron-compton",
        ):
            raise TransportError("unsupported scattering law")
        if self.compton_sampler not in ("legacy-rejection", "packed-uniform-rejection"):
            raise TransportError("unsupported Compton sampler execution policy")
        if self.scattering_law != "isotropic-elastic" and len(nodes) < 2:
            raise TransportError("Compton scattering needs energy-dependent coefficient tables")
        if self.scoring not in ("photon-count", "energy-kev"):
            raise TransportError("unsupported detector score")
        if self.estimator not in ("analogue", "continuous-absorption"):
            raise TransportError("unsupported transport estimator")
        _positive_integer(self.max_angle_trials, "max_angle_trials")
        _positive_integer(self.max_events, "max_events")
        _positive_integer(self.max_crossings, "max_crossings")
        if self.energy_support_policy not in ("runtime-guard", "cover-budget"):
            raise TransportError("unsupported coefficient energy support policy")
        if self.energy_support_policy == "cover-budget" and nodes[0] > self.minimum_energy_kev:
            raise TransportError(
                "coefficient energy support does not cover the declared collision budget; "
                f"lower endpoint must be at most {self.minimum_energy_kev:.17g} keV"
            )
        if self.block_dim not in (64, 128, 256):
            raise TransportError("block_dim must be 64, 128 or 256")

    @property
    def energy_nodes(self) -> tuple[float, ...]:
        """An omitted grid declares coefficients at the one elastic source energy."""
        return self.coefficient_energies_kev or (self.energy_kev,)

    @property
    def minimum_energy_kev(self) -> float:
        """Worst-case energy after the budget's Compton backscatter sequence.

        Free-electron kinematics gives 1/E' = 1/E + (1-cos(theta))/mc².
        Every increment is at most 2/mc²; elastic energy remains unchanged.
        Resource guard failures remain fatal under either support policy.
        """
        if self.scattering_law == "isotropic-elastic":
            return self.energy_kev
        if self.scattering_law == "bound-electron-compton":
            return 0.0
        if self.energy_kev <= ELECTRON_REST_ENERGY_KEV:
            return self.energy_kev / (
                1.0 + (self.energy_kev / ELECTRON_REST_ENERGY_KEV) * (2.0 * self.max_events)
            )
        return 1.0 / (1.0 / self.energy_kev + 2.0 * self.max_events / ELECTRON_REST_ENERGY_KEV)
