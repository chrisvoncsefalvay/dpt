"""Prepared physical transport geometry means and named mean pullbacks.

Each observation combines every correlated boundary and physical continuation
belonging to one original emitted photon before reduction. Model arrays, source
charts and the complete physical and auxiliary randomisation are owned at
preparation, while execution accepts only a signed detector cotangent and
caller-owned output arrays.
"""

from __future__ import annotations

# Prepared transport records and geometry charts are shared package internals.
# pyright: reportPrivateUsage=false
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, cast

from dpt._runtime import DeviceContext, load_kernels, prepare_context, require_no_tape

from .compton_boundary_arguments import BoundaryDetectorArguments
from .forward import TransportWorkspace, prepare_transport
from .geometry_composed import material_patch_at
from .geometry_detector_proposal_arguments import DetectorProposalPlaneArguments
from .geometry_forward_proposal_arguments import (
    ForwardProposalGeometryArguments,
    forward_proposal_plan_arguments,
)
from .geometry_histories import GeometryPhysicalMedium
from .geometry_interface_arguments import InterfaceGridArguments, interface_arguments
from .geometry_opaque_source import OpaqueSourceBoundary, opaque_source_boundaries
from .geometry_parameters import GeometryParameter
from .geometry_physical_admission import GeometryPhysicalAdmission
from .geometry_sampling import (
    GeometrySamplingDesign,
    GeometrySamplingIdentity,
    PreparedGeometrySampling,
    prepare_geometry_sampling,
)
from .geometry_source_boundary_arguments import (
    AffineSourceBoundaryArguments,
    SourceGeometryArguments,
    affine_source_boundary_arguments,
    source_geometry_arguments,
)
from .geometry_source_jumps import admit_joint_source_boundary_chart, material_source_ray_boundaries
from .geometry_startup_admission import admit_source_startup_chart
from .geometry_tangents import detector_corner_velocity
from .geometry_whole_source_admission import (
    WholeSourceGeometryAdmission,
    admit_whole_source_geometry,
)
from .model import TransportError, TransportSpec
from .physical import HC_KEV_ANGSTROM, FluorescenceLine, PhysicalTransport
from .rng import HistoryBatch
from .source import ParallelBeam, PointConeSource
from .uncollided_geometry import _detector_boundary

if TYPE_CHECKING:
    import warp as wp

__all__ = [
    "TransportGeometryIdentity",
    "TransportGeometryWorkspace",
    "prepare_transport_geometry",
    "transport_geometry_vjp",
]


@dataclass(frozen=True, slots=True)
class TransportGeometryIdentity:
    """Immutable model, source, coordinate order and complete design identity.

    ``fingerprint`` binds the original transport spec, fixed material and atomic
    tables, source law and geometry, named active coordinates and complete
    original-history randomisation. The analogue reference retains the same
    physical expectation when the original forward spec uses variance reduction.
    """

    spec: TransportSpec
    source: ParallelBeam | PointConeSource
    medium: GeometryPhysicalMedium
    parameters: tuple[GeometryParameter, ...]
    sampling: GeometrySamplingIdentity
    chart_masks: tuple[tuple[int, ...], ...] = ()
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(Any, self.spec), TransportSpec)
            or not isinstance(cast(Any, self.source), (ParallelBeam, PointConeSource))
            or not isinstance(cast(Any, self.medium), GeometryPhysicalMedium)
            or not isinstance(cast(Any, self.sampling), GeometrySamplingIdentity)
        ):
            raise TransportError(
                "geometry identity requires its immutable physical and sampling records"
            )
        selected = tuple(self.parameters)
        if (
            not 1 <= len(selected) <= 64
            or any(not isinstance(cast(Any, item), GeometryParameter) for item in selected)
            or len(set(selected)) != len(selected)
        ):
            raise TransportError("geometry identity requires distinct named coordinates")
        object.__setattr__(self, "parameters", selected)
        masks = tuple(tuple(mask) for mask in self.chart_masks)
        plane_count = sum(size + 1 for size in self.spec.grid.shape_xyz)
        if masks and (
            len(masks) != len(selected)
            or any(
                len(mask) != plane_count
                or any(type(flag) is not int or flag not in (0, 1) for flag in mask)
                for mask in masks
            )
        ):
            raise TransportError(
                "geometry identity requires one exact dense plane mask per coordinate"
            )
        object.__setattr__(self, "chart_masks", masks)
        source_kind = (
            "parallel-rectangle-v1"
            if isinstance(self.source, ParallelBeam)
            else "point-cone-area-v1"
        )
        if (
            self.sampling.batch.source_identity is None
            or self.sampling.batch.source_identity.sampler != source_kind
        ):
            raise TransportError("geometry identity must bind the actual library source token")
        payload = {
            "version": "physical-geometry-mean-v2",
            "spec": asdict(self.spec),
            "source_kind": type(self.source).__name__,
            "source": asdict(self.source),
            "medium": asdict(self.medium),
            "parameters": tuple(asdict(parameter) for parameter in self.parameters),
            "complete_sampling": self.sampling.identity,
            "certified_plane_masks": masks,
        }
        try:
            encoded = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        except (TypeError, ValueError) as error:
            raise TransportError(
                "geometry model and provenance require finite serialisable identity fields"
            ) from error
        object.__setattr__(self, "fingerprint", hashlib.sha256(encoded).hexdigest())


@dataclass(frozen=True, slots=True)
class _HostColumn:
    source: SourceGeometryArguments
    grid: InterfaceGridArguments
    detector: BoundaryDetectorArguments
    source_boundaries: tuple[AffineSourceBoundaryArguments, ...]
    detector_boundaries: tuple[AffineSourceBoundaryArguments, ...]
    opaque_boundaries: tuple[OpaqueSourceBoundary | None, ...]
    empty_medium: bool
    redundant_planes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _GeometryPlan:
    identity: TransportGeometryIdentity
    columns: tuple[_HostColumn, ...]
    planes: tuple[DetectorProposalPlaneArguments, ...]
    proposal: ForwardProposalGeometryArguments
    max_fluorescence_lines: int
    admissions: tuple[WholeSourceGeometryAdmission, ...]

    @property
    def bound_admission(self) -> GeometryPhysicalAdmission | None:
        """Retain the bound-specific view of the complete physical certificate."""
        return (
            self.admissions[0].physical
            if self.identity.spec.scattering_law == "bound-electron-compton"
            else None
        )


def _validate_physical_model(spec: TransportSpec, medium: GeometryPhysicalMedium) -> None:
    """Apply canonical immutable physical contracts before any chart indexes tables."""
    physical = medium.physical
    if physical is None:
        if spec.scattering_law == "bound-electron-compton":
            raise TransportError("bound Compton transport requires explicit physical shell tables")
        return
    if not isinstance(cast(Any, physical), PhysicalTransport):
        raise TransportError("physical must be a PhysicalTransport record")
    materials, bins = spec.grid.materials, len(spec.energy_nodes)
    if len(physical.scattering_tables) != materials:
        raise TransportError("physical transport needs one scattering table per material")
    if len(physical.rayleigh_mm_inverse) != materials * bins:
        raise TransportError("Rayleigh table must match the material/energy coefficient shape")
    if len(medium.scattering_mm_inverse) != materials * bins:
        raise TransportError("geometry coefficients require material-major energy-fastest tables")
    if any(
        coherent > aggregate
        for coherent, aggregate in zip(
            physical.rayleigh_mm_inverse, medium.scattering_mm_inverse, strict=True
        )
    ):
        raise TransportError("Rayleigh partial exceeds aggregate scattering")
    if spec.estimator != "analogue" and (
        any(physical.fluorescence) or physical.majorant_mm_inverse
    ):
        raise TransportError("physical fluorescence and Woodcock require analogue transport")
    if spec.max_angle_trials > 65536:
        raise TransportError("physical angular rejection reserves at most 65536 trial domains")
    if physical.majorant_mm_inverse and len(physical.majorant_mm_inverse) != bins:
        raise TransportError("Woodcock majorant must match the coefficient energy nodes")
    if physical.cutoff_kev > spec.energy_nodes[-1]:
        raise TransportError("cutoff exceeds the supplied energy support")
    spectrum = physical.spectrum.energy_kev if physical.spectrum else ()
    if spectrum and (spectrum[0] < spec.energy_nodes[0] or spectrum[-1] > spec.energy_nodes[-1]):
        raise TransportError("source spectrum leaves coefficient energy support")
    maximum_energy = max((spec.energy_kev, *spectrum))
    if spec.scattering_law == "bound-electron-compton":
        if len(physical.bound_compton) != materials:
            raise TransportError("bound Compton needs one shell table per material")
        if spec.compton_sampler != "legacy-rejection":
            raise TransportError("bound Compton uses its own versioned impulse sampler")
        if min((spec.energy_kev, *spectrum)) < 0.1 or maximum_energy > 5000:
            raise TransportError("bound Compton incident energy domain is 0.1..5000 keV")
        if physical.cutoff_kev < 0.1:
            raise TransportError(
                "bound Compton requires a local-deposition cutoff of at least 0.1 keV"
            )
        for material, table in enumerate(physical.bound_compton):
            if table.atomic_number != physical.scattering_tables[material].atomic_number:
                raise TransportError("bound Compton and atomic scattering elements must match")
    elif physical.bound_compton:
        raise TransportError("shell tables require the bound-electron-compton scattering law")
    if spectrum and spec.energy_support_policy == "cover-budget":
        minimum_energy = spectrum[0]
        if spec.scattering_law == "free-electron-compton":
            minimum_energy = 1 / (1 / minimum_energy + 2 * spec.max_events / 510.99895069)
        if spec.energy_nodes[0] > minimum_energy and physical.cutoff_kev < spec.energy_nodes[0]:
            raise TransportError(
                "coefficient support does not cover the polyenergetic collision budget"
            )
    for table in physical.scattering_tables:
        if table.momentum_transfer_inverse_angstrom[-1] < maximum_energy / HC_KEV_ANGSTROM:
            raise TransportError("atomic scattering q support does not cover all source angles")
    branches = physical.fluorescence or tuple(() for _ in range(materials))
    if len(branches) != materials:
        raise TransportError("fluorescence branches must match the material count")
    probabilities = 0
    for lines in branches:
        for line in lines:
            if not isinstance(cast(Any, line), FluorescenceLine) or len(line.probabilities) != bins:
                raise TransportError(
                    "fluorescence probabilities must match coefficient energy nodes"
                )
            if line.energy_kev > physical.cutoff_kev and line.energy_kev < spec.energy_nodes[0]:
                raise TransportError("transported fluorescence line leaves coefficient support")
            if any(
                energy < line.threshold_kev and probability != 0
                for energy, probability in zip(spec.energy_nodes, line.probabilities, strict=True)
            ):
                raise TransportError(
                    "fluorescence probability is nonzero below its vacancy threshold"
                )
            probabilities += len(line.probabilities)
        for node in range(bins):
            if math.fsum(line.probabilities[node] for line in lines) > 1.0:
                raise TransportError("conditional fluorescence branch probabilities exceed one")
    if (
        max(
            sum(len(table.form_factor) for table in physical.scattering_tables),
            probabilities,
            len(physical.rayleigh_mm_inverse),
        )
        >= 2**31
    ):
        raise TransportError("physical table exceeds signed 32-bit indexing")
    if physical.spectrum is not None:
        cdf = physical.spectrum.cdf
        if any(b <= a for a, b in zip((0.0, *cdf[:-1]), cdf, strict=True)):
            raise TransportError("source spectrum has a probability lost to binary64 CDF rounding")


def _validate_geometry_index_budgets(spec: TransportSpec, coordinates: int, histories: int) -> None:
    """Reject derived signed native indices before constructing dense charts."""
    limit = 2**31 - 1
    shape = spec.grid.shape_xyz
    planes = sum(size + 1 for size in shape)
    patches = sum(
        (shape[axis] + 1) * shape[(axis + 1) % 3] * shape[(axis + 2) % 3] for axis in range(3)
    )
    detector_rows = sum(size + 1 for size in spec.detector.shape)
    counts = (
        (histories * coordinates, "observation"),
        (spec.grid.materials * len(spec.energy_nodes), "material-energy coefficient"),
        (planes, "material plane"),
        (patches, "material patch"),
        (coordinates * planes, "source boundary and plane mask"),
        (coordinates * detector_rows, "detector boundary"),
        (coordinates * 18, "opaque boundary"),
    )
    for count, description in counts:
        if count > limit:
            raise TransportError(f"geometry {description} indexing exceeds signed int32")


def _geometry_plan(
    spec: TransportSpec,
    source: ParallelBeam | PointConeSource,
    medium: GeometryPhysicalMedium,
    parameters: Sequence[GeometryParameter],
    batch: HistoryBatch,
    design: GeometrySamplingDesign,
) -> _GeometryPlan:
    """Validate the complete host model and prepare every fixed chart before uploads."""
    if not isinstance(cast(Any, spec), TransportSpec):
        raise TransportError("transport geometry requires a TransportSpec")
    if not isinstance(cast(Any, source), (ParallelBeam, PointConeSource)):
        raise TransportError("transport geometry requires an explicit library source law")
    source_origins = (
        (source.geometry.source_mm,)
        if isinstance(source, PointConeSource)
        else tuple(
            (source.lower_mm[0] + x, source.lower_mm[1] + y, source.lower_mm[2])
            for x in (0.0, source.extent_xy_mm[0])
            for y in (0.0, source.extent_xy_mm[1])
        )
    )
    if any(
        not math.isfinite(distance) or distance >= 0.0
        for origin in source_origins
        for distance in (spec.detector.source_distance(origin),)
    ):
        raise TransportError("geometry source must lie on the incident side of the detector plane")
    if not isinstance(cast(Any, medium), GeometryPhysicalMedium):
        raise TransportError("transport geometry requires an immutable GeometryPhysicalMedium")
    if not isinstance(cast(Any, batch), HistoryBatch) or not isinstance(
        cast(Any, design), GeometrySamplingDesign
    ):
        raise TransportError(
            "transport geometry requires an original HistoryBatch and complete geometry design"
        )
    selected = tuple(parameters)
    if not 1 <= len(selected) <= 64 or any(
        not isinstance(cast(Any, parameter), GeometryParameter) for parameter in selected
    ):
        raise TransportError(
            "transport geometry requires one to sixty-four GeometryParameter records"
        )
    if len(set(selected)) != len(selected):
        raise TransportError("transport geometry coordinates must be distinct")
    _validate_geometry_index_budgets(spec, len(selected), batch.count)
    _validate_physical_model(spec, medium)
    medium.validate(spec)
    if any(
        not math.isfinite(absorption + scattering)
        for absorption, scattering in zip(
            medium.absorption_mm_inverse, medium.scattering_mm_inverse, strict=True
        )
    ):
        raise TransportError(
            "geometry absorption plus scattering must be finite at every material and energy node"
        )
    for cell in range(len(medium.material_ids)):
        factor = medium.factor(cell)
        if not math.isfinite(factor) or (medium.voxel_density[cell] > 0.0 and factor <= 0.0):
            raise TransportError("geometry positive cell factor is outside binary64 range")
    if medium.physical is not None and medium.physical.majorant_mm_inverse:
        bins = len(spec.energy_nodes)
        for cell, material in enumerate(medium.material_ids):
            for node, majorant in enumerate(medium.physical.majorant_mm_inverse):
                offset = material * bins + node
                total = medium.absorption_mm_inverse[offset] + medium.scattering_mm_inverse[offset]
                if not math.isfinite(total):
                    raise TransportError("fixed majorant does not bound the physical total rate")
                factors = (medium.density[material], medium.voxel_density[cell], total)
                mantissa, exponent = 1.0, 0
                for value in factors:
                    part, power = math.frexp(value)
                    mantissa *= part
                    exponent += power
                try:
                    rate = math.ldexp(mantissa, exponent)
                except OverflowError:
                    rate = math.inf
                if not math.isfinite(rate) or rate >= majorant:
                    raise TransportError(
                        "fixed majorant requires strict slack at every used cell and energy node"
                    )
    expected_source = (
        "parallel-rectangle-v1" if isinstance(source, ParallelBeam) else "point-cone-area-v1"
    )
    if batch.source_identity is not None and batch.source_identity.sampler != expected_source:
        raise TransportError(
            "structural source identity does not match the actual geometry source law"
        )
    if batch.source_identity is None:
        batch = batch.with_library_source(expected_source)
    sampling = GeometrySamplingIdentity(design, batch)
    lines = (
        0
        if medium.physical is None
        else max((len(material) for material in medium.physical.fluorescence), default=0)
    )
    if design.channel_slots != 2 + lines:
        raise TransportError(
            "geometry channel slots must match every represented fluorescence line"
        )
    expected_layout = (
        "bound-compton-v3"
        if spec.scattering_law == "bound-electron-compton"
        else "packed-compton-v2"
        if spec.compton_sampler == "packed-uniform-rejection"
        else "legacy-v1"
    )
    if design.layout != expected_layout:
        raise TransportError(
            "geometry physical layout does not match the transport execution policy"
        )
    if design.max_events != spec.max_events or design.max_angle_trials < spec.max_angle_trials:
        raise TransportError(
            "geometry sampling must match the actual event guard and cover angular trials"
        )
    empty = all(
        medium.voxel_density[cell] == 0.0
        or (
            medium.absorption_mm_inverse[material * len(spec.energy_nodes) + energy]
            + medium.scattering_mm_inverse[material * len(spec.energy_nodes) + energy]
        )
        == 0.0
        for cell, material in enumerate(medium.material_ids)
        for energy in range(len(spec.energy_nodes))
    )
    initial_energies = (
        (spec.energy_kev,)
        if medium.physical is None or medium.physical.spectrum is None
        else medium.physical.spectrum.energy_kev
    )
    opaque_initial = medium.physical is not None and any(
        energy <= medium.physical.cutoff_kev for energy in initial_energies
    )
    columns: list[_HostColumn] = []
    admissions: list[WholeSourceGeometryAdmission] = []
    for parameter in selected:
        tangent = parameter.tangent(source)
        equivalent_faces: frozenset[tuple[int, int]] = frozenset()
        if not empty:
            admit_source_startup_chart(source, spec, medium, tangent)
        admission = admit_whole_source_geometry(source, spec, medium, tangent)
        admissions.append(admission)
        equivalent_faces = admission.source.equivalent_faces
        source_arguments = source_geometry_arguments(source, tangent.source)
        _, grid = interface_arguments(material_patch_at(0, spec), spec, tangent.object)
        corner_velocity = detector_corner_velocity(
            spec.detector, tangent.detector, tangent.detector_spacing_dot
        )
        rigid_velocity = tangent.detector.point_velocity(spec.detector.lower_corner_mm)
        detector = BoundaryDetectorArguments(
            spec.detector.lower_corner_mm,
            spec.detector.u,
            spec.detector.v,
            spec.detector.normal,
            spec.detector.shape,
            spec.detector.spacing_xy_mm,
            tangent.detector.translation,
            tangent.detector.angular,
            tangent.detector.pivot_mm,
            cast(
                tuple[float, float, float],
                tuple(a - b for a, b in zip(corner_velocity, rigid_velocity, strict=True)),
            ),
            tangent.detector_spacing_dot,
        )
        material_charts = {
            (record.axis, record.face): record.boundary
            for record in (
                ()
                if empty
                else material_source_ray_boundaries(
                    source, spec, tangent, equivalent_whole_faces=equivalent_faces
                )
            )
        }
        source_boundaries = tuple(
            affine_source_boundary_arguments(
                material_charts.get((axis, face)), axis=axis, face=face, family="material"
            )
            for axis, size in enumerate(spec.grid.shape_xyz)
            for face in range(size + 1)
        )
        detector_charts = tuple(
            boundary
            for axis, size in enumerate(spec.detector.shape)
            for face in range(size + 1)
            if (boundary := _detector_boundary(source, spec, tangent, axis, face)) is not None
        )
        admit_joint_source_boundary_chart(
            (*material_charts.values(), *detector_charts), family="material/detector"
        )
        detector_boundaries = tuple(
            affine_source_boundary_arguments(
                _detector_boundary(source, spec, tangent, axis, face),
                axis=axis,
                face=face,
                family="detector",
            )
            for axis, size in enumerate(spec.detector.shape)
            for face in range(size + 1)
        )
        opaque: list[OpaqueSourceBoundary | None] = [None] * 18
        if opaque_initial:
            for record in opaque_source_boundaries(source, spec, tangent):
                if not 0 <= record.index < 18 or opaque[record.index] is not None:
                    raise TransportError(
                        "opaque source charts must retain unique eighteen-row identities"
                    )
                opaque[record.index] = record
            admit_joint_source_boundary_chart(
                (*[row.boundary for row in opaque if row is not None], *detector_charts),
                family="opaque/detector",
            )
        columns.append(
            _HostColumn(
                source_arguments,
                grid,
                detector,
                source_boundaries,
                detector_boundaries,
                tuple(opaque),
                empty,
                tuple(
                    int((axis, face) in admission.redundant_detector_planes)
                    for axis, size in enumerate(spec.grid.shape_xyz)
                    for face in range(size + 1)
                ),
            )
        )
    redundant = admissions[0].redundant_detector_planes
    for admission in admissions[1:]:
        redundant = redundant.intersection(admission.redundant_detector_planes)
    planes, proposal = forward_proposal_plan_arguments(spec, redundant_planes=redundant)
    identity = TransportGeometryIdentity(
        spec,
        source,
        medium,
        selected,
        sampling,
        tuple(column.redundant_planes for column in columns),
    )
    return _GeometryPlan(identity, tuple(columns), planes, proposal, lines, tuple(admissions))


def _pack(record: Any, arguments: Any) -> Any:
    for item in fields(arguments):
        setattr(record, item.name, getattr(arguments, item.name))
    return record


@dataclass(frozen=True, slots=True)
class TransportGeometryWorkspace:
    """Owned immutable physical model, source charts and complete randomisation.

    The public ``spec`` remains the original forward policy and ``reference_spec``
    selects analogue physical histories for the same expected measurement.
    Geometry execution reuses the prepared stream and never uploads a second
    caller-supplied model mirror. Sticky status belongs to the complete estimate.
    """

    identity: TransportGeometryIdentity
    context: DeviceContext
    _reference: TransportWorkspace = field(repr=False)
    _sampling: PreparedGeometrySampling = field(repr=False)
    _density: Any = field(repr=False)
    _columns: Any = field(repr=False)
    _planes: Any = field(repr=False)
    _proposal: Any = field(repr=False)
    _source_boundaries: Any = field(repr=False)
    _detector_boundaries: Any = field(repr=False)
    _opaque_boundaries: Any = field(repr=False)
    _kernels: Any = field(repr=False)
    _validation: Any = field(repr=False)
    _admissions: tuple[WholeSourceGeometryAdmission, ...] = field(repr=False)
    _plane_masks: tuple[Any, ...] = field(repr=False)

    @property
    def spec(self) -> TransportSpec:
        return self.identity.spec

    @property
    def reference_spec(self) -> TransportSpec:
        return self._reference.spec

    @property
    def batch(self) -> HistoryBatch:
        return self.identity.sampling.batch

    @property
    def parameters(self) -> tuple[GeometryParameter, ...]:
        return self.identity.parameters

    @property
    def allocated_bytes(self) -> int:
        """Owned resident model, chart, sampler and status bytes, counted once."""
        arrays = [array for _, array in self._reads()]
        unique = {
            (
                int(array.ptr),
                int(array.size),
                self.context.wp.types.type_size_in_bytes(array.dtype),
            ): array
            for array in arrays
            if array.size
        }
        return sum(
            int(array.size) * self.context.wp.types.type_size_in_bytes(array.dtype)
            for array in unique.values()
        )

    @property
    def host_bytes(self) -> int:
        """Prepared four-byte status staging, excluding immutable host records."""
        return self._reference.host_bytes

    def check_status(self) -> None:
        """Complete the owning stream and reject any failed original observation."""
        self._reference.check_status()

    def clear_status(self) -> None:
        """Start a declared independent run after discarding failed outputs."""
        self._reference.clear_status()

    def _reads(self) -> list[tuple[str, Any]]:
        return [
            *self._reference._reads(),
            ("material_density", self._density),
            ("geometry_columns", self._columns),
            ("proposal_planes", self._planes),
            ("source_boundaries", self._source_boundaries),
            ("detector_boundaries", self._detector_boundaries),
            ("opaque_boundaries", self._opaque_boundaries),
            ("geometry_sampling_words", self._sampling._words),
            ("geometry_sampling_shift", self._sampling._shift),
            *[
                (f"certified_plane_mask_{index}", mask)
                for index, mask in enumerate(self._plane_masks)
            ],
        ]


def prepare_transport_geometry(
    spec: TransportSpec,
    *,
    source: ParallelBeam | PointConeSource,
    medium: GeometryPhysicalMedium,
    parameters: Sequence[GeometryParameter],
    batch: HistoryBatch,
    design: GeometrySamplingDesign,
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
) -> TransportGeometryWorkspace:
    """Prepare the physical mean geometry pullback on one explicit device/stream.

    ``medium`` owns host tuples: one int32 material label and dimensionless voxel
    density per cell, one positive dimensionless density per material and
    material-major energy-fastest binary64 coefficients in mm inverse. Atomic,
    fluorescence, bound-Compton and source-spectrum records remain fixed.
    Source rays and their declared normalisation come from ``source``.
    ``parameters`` specifies one to sixty-four distinct geometry coordinates in
    output order, with translations/spacings per mm and rotations per radian.

    Preparation validates and uploads this complete immutable host model once.
    It binds every dense material-plane and detector-line slot, all eighteen
    opaque source slots and the complete geometry randomisation to their owning
    stream. A missing source token is attached for the actual selected library
    sampler, while a mismatched token raises. Undefined active coincidence
    charts raise rather than changing their physical measure.

    The original spec remains available and a separate analogue reference spec
    implements its expected physical measurement. ``photon-count`` and
    ``energy-kev`` scores are supported. All later execution uses resident arrays
    with no Warp Tape, source resampling on the host or implicit model upload.

    """
    plan = _geometry_plan(spec, source, medium, parameters, batch, design)
    require_no_tape()
    context = prepare_context(device=device, stream=stream, contract_error=TransportError)
    wp = context.wp
    with context.scope():
        ids = wp.array(medium.material_ids, dtype=wp.int32, device=context.device)
        absorption = wp.array(medium.absorption_mm_inverse, dtype=wp.float64, device=context.device)
        scattering = wp.array(medium.scattering_mm_inverse, dtype=wp.float64, device=context.device)
        voxels = wp.array(medium.voxel_density, dtype=wp.float64, device=context.device)
        density = wp.array(medium.density, dtype=wp.float64, device=context.device)
    reference = prepare_transport(
        replace(spec, estimator="analogue"),
        material_ids=ids,
        absorption=absorption,
        scattering=scattering,
        voxel_density=voxels,
        max_histories=batch.count,
        device=context.device,
        stream=context.stream,
        physical=medium.physical,
    )
    sampling = prepare_geometry_sampling(
        context,
        spec=spec,
        batch=plan.identity.sampling.batch,
        design=design,
        max_fluorescence_lines=plan.max_fluorescence_lines,
    )
    reference._parameters.sampling = sampling.physical_record
    reference._parameters.physics.sampling = sampling.physical_record
    native = load_kernels("dpt.transport.geometry_driver_kernels")
    sources = load_kernels("dpt.transport.geometry_source_boundary_kernels")
    interfaces = load_kernels("dpt.transport.geometry_interface_kernels")
    detector = load_kernels("dpt.transport.compton_geometry_boundary_kernels")
    proposals = load_kernels("dpt.transport.geometry_detector_proposal_kernels")
    forward = load_kernels("dpt.transport.geometry_forward_proposal_kernels")
    opaque = load_kernels("dpt.transport.geometry_opaque_kernels")
    validation = load_kernels("dpt.transport.voxel_kernels")
    columns: list[Any] = []
    source_rows: list[Any] = []
    detector_rows: list[Any] = []
    opaque_rows: list[Any] = []
    plane_masks: list[Any] = []
    for item in plan.columns:
        column = native.NativeGeometryColumn()
        column.source = _pack(sources.NativeSourceGeometry(), item.source)
        column.grid = _pack(interfaces.InterfaceGrid(), item.grid)
        column.detector = _pack(detector.NativeBoundaryDetector(), item.detector)
        (
            column.source_boundary_offset,
            column.detector_boundary_offset,
            column.opaque_boundary_offset,
        ) = len(source_rows), len(detector_rows), len(opaque_rows)
        column.empty_medium = int(item.empty_medium)
        with context.scope():
            mask = wp.array(item.redundant_planes, dtype=wp.int32, device=context.device)
        column.redundant_planes = mask
        plane_masks.append(mask)
        columns.append(column)
        source_rows.extend(
            _pack(sources.NativeAffineSourceBoundary(), row) for row in item.source_boundaries
        )
        detector_rows.extend(
            _pack(sources.NativeAffineSourceBoundary(), row) for row in item.detector_boundaries
        )
        for row in item.opaque_boundaries:
            record = opaque.NativeOpaqueBoundary()
            if row is None:
                record.boundary = _pack(
                    sources.NativeAffineSourceBoundary(),
                    affine_source_boundary_arguments(None, axis=0, face=0, family="material"),
                )
                record.face_count = 0
            else:
                record.face_count = len(row.attached_faces)
                record.axis0, record.high0 = row.attached_faces[0]
                if record.face_count == 2:
                    record.axis1, record.high1 = row.attached_faces[1]
                record.boundary = _pack(
                    sources.NativeAffineSourceBoundary(),
                    affine_source_boundary_arguments(
                        row.boundary, axis=record.axis0, face=record.high0, family="material"
                    ),
                )
            opaque_rows.append(record)
    proposal = forward.ForwardProposalGeometry()
    proposal.regular = _pack(proposals.DetectorProposalGeometry(), plan.proposal.regular)
    proposal.detector_normal = plan.proposal.detector_normal
    with context.scope():
        resident_columns = wp.array(
            columns, dtype=native.NativeGeometryColumn, device=context.device
        )
        resident_planes = wp.array(
            [_pack(proposals.DetectorProposalPlane(), row) for row in plan.planes],
            dtype=proposals.DetectorProposalPlane,
            device=context.device,
        )
        resident_source = wp.array(
            source_rows, dtype=sources.NativeAffineSourceBoundary, device=context.device
        )
        resident_detector = wp.array(
            detector_rows, dtype=sources.NativeAffineSourceBoundary, device=context.device
        )
        resident_opaque = wp.array(
            opaque_rows, dtype=opaque.NativeOpaqueBoundary, device=context.device
        )
    return TransportGeometryWorkspace(
        plan.identity,
        context,
        reference,
        sampling,
        density,
        resident_columns,
        resident_planes,
        proposal,
        resident_source,
        resident_detector,
        resident_opaque,
        native.geometry_kernel_bundle(
            reference._parameters.compton, bool(reference._parameters.compton_packed)
        ),
        validation,
        plan.admissions,
        tuple(plane_masks),
    )


def transport_geometry_vjp(
    cotangent: wp.array[wp.float64],
    *,
    workspace: TransportGeometryWorkspace,
    out_values: wp.array2d[wp.float64],
    out_derivatives: wp.array2d[wp.float64],
    out_counts: wp.array2d[wp.vec2i],
    out_mean: wp.array[wp.float64],
    out_gradient: wp.array[wp.float64],
    out_second_moment: wp.array[wp.float64],
    validate: bool = True,
    stream: wp.Stream | None = None,
) -> None:
    """Evaluate the geometry transpose of the expected physical measurement.

    For each original photon ``h`` and named coordinate ``j``, execution sums
    the physical prefix, terminal, ballistic/interface, source and pixel-boundary
    terms into ``D[h,j]`` before computing ``gradient[j]=sum_h(D[h,j])/N`` and
    ``second_moment[j]=sum_h(D[h,j]**2)/N``. The raw second moment describes the
    original observations and is not an IID variance or a QMC confidence bound.

    ``cotangent`` is a contiguous finite signed binary64 array of shape
    ``(detector.pixels,)`` on the owning device. It contracts photon-count or keV
    scores. The contiguous binary64 ``out_values`` and ``out_derivatives`` have
    shape ``(N,P)`` and ``out_counts`` has dtype ``wp.vec2i`` with shape ``(N,P)``
    for reached prefixes and complete correlated futures. Each reduction output
    is binary64 with shape ``(P,)``. Values/means carry contracted measurement
    units, derivatives/gradients carry those units per mm or radian and the raw
    second moment carries their square. All output byte ranges must be disjoint
    from one another, the seed and every prepared model/chart/sampler buffer.

    The call allocates no device arrays and retains the prepared stream and
    immutable model. Finite cotangents are checked on device even for an
    unchecked call. ``validate=True`` checks sticky status before and after
    execution. With ``False``, call ``workspace.check_status()`` before consuming
    results. A failed future invalidates its whole original observation and all
    reductions until an explicit discarded-run status reset.

    """
    if not isinstance(cast(Any, workspace), TransportGeometryWorkspace):
        raise TransportError("transport geometry execution requires its prepared workspace")
    if type(validate) is not bool:
        raise TransportError("geometry validation policy must be boolean")
    require_no_tape()
    context, reference = workspace.context, workspace._reference
    context.assert_stream(stream)
    wp = context.wp
    count, parameters = workspace.batch.count, len(workspace.parameters)
    context.array(cotangent, "cotangent", dtype=wp.float64, shape=(workspace.spec.detector.pixels,))
    writes = (
        ("out_values", out_values, wp.float64, (count, parameters)),
        ("out_derivatives", out_derivatives, wp.float64, (count, parameters)),
        ("out_counts", out_counts, wp.vec2i, (count, parameters)),
        ("out_mean", out_mean, wp.float64, (parameters,)),
        ("out_gradient", out_gradient, wp.float64, (parameters,)),
        ("out_second_moment", out_second_moment, wp.float64, (parameters,)),
    )
    for name, array, dtype, shape in writes:
        context.array(array, name, dtype=dtype, shape=shape, ndim=len(shape))
    context.disjoint(
        [*workspace._reads(), ("cotangent", cotangent)],
        [(name, array) for name, array, _, _ in writes],
    )
    if validate:
        workspace.check_status()
    with context.scope():
        reference._launch(
            workspace._validation.validate_cotangent,
            workspace.spec.detector.pixels,
            [cotangent, reference._status],
        )
        if validate:
            workspace.check_status()
        wp.launch(
            workspace._kernels.geometry_observations,
            dim=(count, parameters),
            inputs=[
                workspace._columns,
                workspace._planes,
                workspace._proposal,
                workspace._source_boundaries,
                workspace._detector_boundaries,
                workspace._opaque_boundaries,
                cotangent,
                reference.material_ids,
                reference.voxel_density,
                reference._energies,
                reference.absorption,
                reference.scattering,
                workspace._density,
                reference._parameters,
                workspace._sampling.auxiliary_record,
                wp.uint64(workspace.batch.seed),
                wp.uint64(workspace.batch.first_history),
                out_values,
                out_derivatives,
                out_counts,
                reference._status,
            ],
            device=context.device,
            stream=context.stream,
            block_dim=workspace.spec.block_dim,
            record_tape=False,
        )
        wp.launch(
            workspace._kernels.reduce_geometry_observations,
            dim=parameters,
            inputs=[
                out_values,
                out_derivatives,
                out_mean,
                out_gradient,
                out_second_moment,
                reference._status,
            ],
            device=context.device,
            stream=context.stream,
            block_dim=workspace.spec.block_dim,
            record_tape=False,
        )
    if validate:
        workspace.check_status()
