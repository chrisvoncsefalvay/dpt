"""Host-only physical ownership and complete source/chart preparation contracts."""

# Preparation checks untyped public callers as well as annotated records.
# pyright: reportPrivateUsage=false
import dataclasses
import hashlib
import inspect
from typing import Any, cast

import pytest

from dpt.geometry import DetectorGeometry
from dpt.materials import Provenance
from dpt.transport.bound_compton import BoundComptonTable, ComptonShell
from dpt.transport.geometry_derivatives import (
    _geometry_plan,
    prepare_transport_geometry,
    transport_geometry_vjp,
)
from dpt.transport.geometry_histories import GeometryPhysicalMedium
from dpt.transport.geometry_parameters import GeometryParameter
from dpt.transport.geometry_sampling import GeometrySamplingDesign
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.physical import AtomicScatteringTable, FluorescenceLine, PhysicalTransport
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam, PointConeSource
from dpt.transport.source_spectrum import SourceSpectrum


def _model():
    spec = TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-2.0, -2.0), (1.0, 1.0), (4, 4), 3.0),
        80.0,
        "independent geometry contract slab",
        max_events=4,
        max_angle_trials=2,
    )
    source = ParallelBeam((-0.2, -0.1, -1.0), (1.4, 1.2))
    medium = GeometryPhysicalMedium((0,), (0.7,), (0.0,), (1.0,), (1.0,))
    parameters = (
        GeometryParameter("object", "grid-spacing", 2),
        GeometryParameter("source", "extent", 0),
        GeometryParameter("detector", "translation", 0),
    )
    return spec, source, medium, parameters


@pytest.mark.parametrize("law", ("iid", "stratified", "scrambled-sobol"))
def test_preparation_retains_complete_dense_charts_and_real_source_token(law):
    spec, source, medium, parameters = _model()
    batch = HistoryBatch(17, 41, 64)
    plan = _geometry_plan(
        spec, source, medium, parameters, batch, GeometrySamplingDesign(law, 4, 2, 2)
    )
    assert batch.source_identity is None
    assert plan.identity.sampling.batch.source_identity is not None
    assert plan.identity.sampling.batch.source_identity.sampler == "parallel-rectangle-v1"
    assert (
        plan.identity.spec is spec
        and plan.identity.source is source
        and plan.identity.medium is medium
    )
    assert plan.identity.parameters == parameters and len(plan.columns) == 3
    for column in plan.columns:
        assert len(column.source_boundaries) == 6 and len(column.detector_boundaries) == 10
        assert len(column.opaque_boundaries) == 18 and all(
            row is None for row in column.opaque_boundaries
        )
        assert [(row.axis, row.face) for row in column.source_boundaries] == [
            (axis, face) for axis in range(3) for face in range(2)
        ]
        assert [(row.axis, row.face) for row in column.detector_boundaries] == [
            (axis, face) for axis in range(2) for face in range(5)
        ]
        assert not column.empty_medium
    first = plan.columns[0]
    assert first.grid.spacing_dot == (0.0, 0.0, 1.0)
    assert first.source.direction_dot == (0.0, 0.0, 0.0)
    assert plan.columns[1].source.extent_dot == (1.0, 0.0)
    assert plan.columns[2].detector.motion_translation == (1.0, 0.0, 0.0)


def test_original_policy_and_full_model_parameter_order_bind_identity():
    spec, source, medium, parameters = _model()
    spec = dataclasses.replace(spec, estimator="continuous-absorption")
    batch, design = HistoryBatch(17, 0, 64), GeometrySamplingDesign("iid", 4, 2, 2)
    plan = _geometry_plan(spec, source, medium, parameters, batch, design)
    assert plan.identity.spec.estimator == "continuous-absorption"
    assert (
        plan.identity.fingerprint
        != _geometry_plan(
            spec,
            source,
            dataclasses.replace(medium, absorption_mm_inverse=(0.8,)),
            parameters,
            batch,
            design,
        ).identity.fingerprint
    )
    assert (
        plan.identity.fingerprint
        != _geometry_plan(
            spec, dataclasses.replace(source, weight=2.0), medium, parameters, batch, design
        ).identity.fingerprint
    )
    assert (
        plan.identity.fingerprint
        != _geometry_plan(
            spec, source, medium, parameters[::-1], batch, design
        ).identity.fingerprint
    )
    assert (
        plan.identity.fingerprint
        != _geometry_plan(
            spec, source, medium, parameters, HistoryBatch(18, 0, 64), design
        ).identity.fingerprint
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cast(Any, plan.identity).spec = dataclasses.replace(spec, scoring="energy-kev")


def test_preparation_admits_identical_whole_face_pencil_without_singular_source_row():
    spec, _, _, _ = _model()
    spec = dataclasses.replace(
        spec, grid=MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 2), 2)
    )
    source = ParallelBeam((1.0, 0.5, -1.0), (0.0, 0.0))
    medium = GeometryPhysicalMedium((0, 1), (0.7, 0.7), (0.0, 0.0), (1.0, 1.0), (1.0, 1.0))
    plan = _geometry_plan(
        spec,
        source,
        medium,
        (GeometryParameter("object", "translation", 0),),
        HistoryBatch(29, 0, 64),
        GeometrySamplingDesign("iid", 4, 2, 2),
    )
    assert not plan.columns[0].empty_medium
    assert len(plan.columns[0].source_boundaries) == 7
    assert not any(row.present for row in plan.columns[0].source_boundaries)
    with pytest.raises(TransportError, match="whole source lies"):
        _geometry_plan(
            spec,
            source,
            dataclasses.replace(medium, absorption_mm_inverse=(0.7, 0.9)),
            plan.identity.parameters,
            plan.identity.sampling.batch,
            plan.identity.sampling.design,
        )


def test_redundant_radiative_plane_masks_bind_columns_and_complete_physical_support():
    spec, _, _, _ = _model()
    spec = dataclasses.replace(
        spec,
        grid=MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 2), 2),
        coefficient_energies_kev=(40, 80),
        detector=PlanarDetector((-5, -5), (10, 10), (1, 1), 3),
    )
    source = ParallelBeam((1, 0.5, -1), (0, 0))
    # Elastic photons never reach 40 keV. The unused difference must not
    # invalidate the actual complete E=80 continuation-law certificate.
    medium = GeometryPhysicalMedium((0, 1), (2, 0.7, 1, 0.7), (0, 0.01, 0, 0.01), (1, 1), (1, 1))
    parameters = tuple(
        GeometryParameter(owner, "translation", 0) for owner in ("object", "source", "detector")
    )
    plan = _geometry_plan(
        spec,
        source,
        medium,
        parameters,
        HistoryBatch(19, 0, 8),
        GeometrySamplingDesign("iid", 4, 2, 2),
    )
    assert plan.identity.chart_masks == ((0, 1, 0, 0, 0, 0, 0),) * 3
    assert all(
        item.redundant_planes == plan.identity.chart_masks[i] for i, item in enumerate(plan.columns)
    )
    assert len(plan.planes) == 4 and len(plan.admissions) == 3
    assert all(item.redundant_detector_planes == frozenset({(0, 1)}) for item in plan.admissions)
    assert all(
        tuple(mark.energy_kev for mark in item.physical.marks) == (80,) for item in plan.admissions
    )


def test_complete_future_energy_jump_rejects_radiative_whole_face_before_uploads():
    spec, _, _, _ = _model()
    spec = dataclasses.replace(
        spec,
        grid=MaterialGrid((-1, 0, 0), (1, 1, 1), (1, 1, 2), 2),
        coefficient_energies_kev=(40, 80),
    )
    provenance = Provenance(
        "https://example.org/analytic", "0" * 64, "CC0", "declared finite-panel control"
    )
    atomic = AtomicScatteringTable(1, (0, 8), (1, 0), (0, 1), provenance)
    line = FluorescenceLine(40, 50, (0, 0.5), provenance)
    physical = PhysicalTransport((atomic, atomic), (0, 0, 0, 0), ((line,), (line,)), cutoff_kev=1)
    medium = GeometryPhysicalMedium(
        (0, 1), (2, 0.7, 1, 0.7), (0, 0, 0, 0), (1, 1), (1, 1), physical
    )
    with pytest.raises(
        TransportError, match="radiative whole-source detector chart is not certified"
    ):
        _geometry_plan(
            spec,
            ParallelBeam((0, 0.5, -1), (0, 0)),
            medium,
            (GeometryParameter("object", "translation", 0),),
            HistoryBatch(19, 0, 8),
            GeometrySamplingDesign("iid", 4, 2, 3),
        )


def test_public_joint_material_detector_chart_rejects_true_one_sided_kink():
    spec, _, medium, _ = _model()
    spec = dataclasses.replace(spec, detector=PlanarDetector((0, -5), (2, 10), (1, 1), 3))
    with pytest.raises(TransportError, match="coincident active material/detector"):
        _geometry_plan(
            spec,
            ParallelBeam((-0.5, 0.4, -1), (1.1, 0.2)),
            medium,
            (GeometryParameter("object", "translation", 0),),
            HistoryBatch(19, 0, 8),
            GeometrySamplingDesign("iid", 4, 2, 2),
        )


def test_preparation_uses_correlated_corner_motion_and_ignores_unreached_quadrants():
    spec, _, _, _ = _model()
    spec = dataclasses.replace(
        spec, grid=MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 2, 2), 4)
    )
    source = ParallelBeam((1.0, 1.0, -1.0), (0.0, 0.0))
    medium = GeometryPhysicalMedium(
        (0, 1, 2, 3), (0.7, 1.3, 1.7, 0.7), (0.0,) * 4, (1.0,) * 4, (1.0,) * 4
    )
    # Rotation about (2,0,0) moves the reached corner along one diagonal.
    # The positive and negative coordinate approach only cells 0 and 3.
    parameter = GeometryParameter("object", "rotation", 2, (2.0, 0.0, 0.0))
    plan = _geometry_plan(
        spec,
        source,
        medium,
        (parameter,),
        HistoryBatch(31, 0, 64),
        GeometrySamplingDesign("iid", 4, 2, 2),
    )
    assert len(plan.columns[0].source_boundaries) == 8
    assert not any(row.present for row in plan.columns[0].source_boundaries)
    with pytest.raises(TransportError, match="whole source lies"):
        _geometry_plan(
            spec,
            source,
            dataclasses.replace(medium, absorption_mm_inverse=(1.1, 1.3, 1.7, 0.7)),
            (parameter,),
            HistoryBatch(31, 0, 64),
            GeometrySamplingDesign("iid", 4, 2, 2),
        )


@pytest.mark.parametrize("law", ("iid", "stratified", "scrambled-sobol"))
def test_point_source_inside_object_is_admitted_and_target_is_independent(law):
    spec, _, medium, _ = _model()
    source = PointConeSource(
        DetectorGeometry(
            (0.4, 0.45, 0.55),
            (-1.2, -1.3, 3.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.7, 0.8),
            (4, 4),
        )
    )
    parameters = (
        GeometryParameter("launch-target", "pixel-spacing", 0),
        GeometryParameter("detector", "pixel-spacing", 1),
    )
    plan = _geometry_plan(
        spec,
        source,
        medium,
        parameters,
        HistoryBatch(1, 0, 64),
        GeometrySamplingDesign(law, 4, 2, 2),
    )
    assert plan.identity.sampling.batch.source_identity is not None
    assert plan.identity.sampling.batch.source_identity.sampler == "point-cone-area-v1"
    assert plan.columns[0].source.target_spacing_dot == (1.0, 0.0)
    assert plan.columns[0].detector.spacing_dot == (0.0, 0.0)
    assert plan.columns[1].source.target_spacing_dot == (0.0, 0.0)
    assert plan.columns[1].detector.spacing_dot == (0.0, 1.0)
    assert all(not row.present for column in plan.columns for row in column.source_boundaries)


def test_existing_source_owner_is_retained_and_mismatch_is_rejected():
    spec, source, medium, parameters = _model()
    design = GeometrySamplingDesign("iid", 4, 2, 2)
    batch = HistoryBatch(17, 0, 64).with_library_source("parallel-rectangle-v1")
    plan = _geometry_plan(spec, source, medium, parameters, batch, design)
    assert plan.identity.sampling.batch is batch
    with pytest.raises(TransportError, match="actual geometry source law"):
        _geometry_plan(
            spec,
            source,
            medium,
            parameters,
            HistoryBatch(17, 0, 64).with_library_source("point-cone-area-v1"),
            design,
        )


@pytest.mark.parametrize("cone", (False, True))
@pytest.mark.parametrize("empty", (False, True))
def test_preparation_requires_actual_emission_on_incident_detector_side(cone, empty):
    spec, _, medium, parameters = _model()
    source = (
        PointConeSource(
            DetectorGeometry(
                (0.4, 0.45, 4.0),
                (0.4, 0.45, 6.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.7, 0.8),
                (1, 1),
            )
        )
        if cone
        else ParallelBeam((0.4, 0.45, 4.0), (0.0, 0.0))
    )
    if empty:
        medium = dataclasses.replace(medium, absorption_mm_inverse=(0.0,))
    with pytest.raises(TransportError, match="incident side of the detector plane"):
        _geometry_plan(
            spec,
            source,
            medium,
            parameters,
            HistoryBatch(17, 0, 64),
            GeometrySamplingDesign("iid", 4, 2, 2),
        )


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("negative-coefficients", "nonnegative"),
        ("negative-voxel", "nonnegative"),
        ("wrong-cell-owner", "material IDs"),
        ("wrong-table", "material-major"),
        ("duplicate-parameters", "distinct"),
        ("empty-parameters", "sixty-four"),
        ("wrong-channel-slots", "channel slots"),
        ("wrong-event-guard", "actual event guard"),
        ("wrong-layout", "execution policy"),
        ("short-trials", "angular trials"),
    ),
)
def test_preparation_rejects_inconsistent_owned_model(mutation, match):
    spec, source, medium, parameters = _model()
    design = GeometrySamplingDesign("iid", 4, 2, 2)
    if mutation == "negative-coefficients":
        medium = dataclasses.replace(medium, absorption_mm_inverse=(-0.1,))
    elif mutation == "negative-voxel":
        medium = dataclasses.replace(medium, voxel_density=(-1.0,))
    elif mutation == "wrong-cell-owner":
        medium = dataclasses.replace(medium, material_ids=(1,))
    elif mutation == "wrong-table":
        medium = dataclasses.replace(medium, absorption_mm_inverse=())
    elif mutation == "duplicate-parameters":
        parameters = (parameters[0], parameters[0])
    elif mutation == "empty-parameters":
        parameters = ()
    elif mutation == "wrong-channel-slots":
        design = dataclasses.replace(design, channel_slots=3)
    elif mutation == "wrong-event-guard":
        design = dataclasses.replace(design, max_events=3)
    elif mutation == "wrong-layout":
        design = dataclasses.replace(design, layout="packed-compton-v2")
    elif mutation == "short-trials":
        design = dataclasses.replace(design, max_angle_trials=1)
    with pytest.raises(TransportError, match=match):
        _geometry_plan(spec, source, medium, parameters, HistoryBatch(17, 0, 64), design)


def test_preparation_api_has_no_caller_device_model_mirror():
    assert set(inspect.signature(prepare_transport_geometry).parameters) == {
        "spec",
        "source",
        "medium",
        "parameters",
        "batch",
        "design",
        "device",
        "stream",
    }
    assert set(inspect.signature(transport_geometry_vjp).parameters) == {
        "cotangent",
        "workspace",
        "out_values",
        "out_derivatives",
        "out_counts",
        "out_mean",
        "out_gradient",
        "out_second_moment",
        "validate",
        "stream",
    }
    _spec, source, medium, parameters = _model()
    with pytest.raises(TransportError, match="TransportSpec"):
        prepare_transport_geometry(
            cast(Any, object()),
            source=source,
            medium=medium,
            parameters=parameters,
            batch=HistoryBatch(17, 0, 64),
            design=GeometrySamplingDesign("iid", 4, 2, 2),
            device="cpu",
        )


@pytest.mark.parametrize("unused_material", (False, True))
def test_preparation_rejects_overflowing_total_rate_before_chart_preparation(unused_material):
    spec, source, medium, parameters = _model()
    if unused_material:
        spec = dataclasses.replace(spec, grid=dataclasses.replace(spec.grid, materials=2))
        medium = GeometryPhysicalMedium((0,), (0.0, 1e308), (0.0, 1e308), (1.0, 1.0), (1.0,))
    else:
        medium = dataclasses.replace(
            medium, absorption_mm_inverse=(1e308,), scattering_mm_inverse=(1e308,)
        )
    with pytest.raises(TransportError, match="absorption plus scattering must be finite"):
        _geometry_plan(
            spec,
            source,
            medium,
            parameters,
            HistoryBatch(17, 0, 64),
            GeometrySamplingDesign("iid", 4, 2, 2),
        )


def test_preparation_rejects_material_energy_index_overflow_before_model_arrays():
    spec, source, medium, parameters = _model()
    spec = dataclasses.replace(
        spec,
        grid=dataclasses.replace(spec.grid, materials=2**30),
        coefficient_energies_kev=(40.0, 80.0),
    )
    with pytest.raises(TransportError, match="material-energy coefficient indexing"):
        _geometry_plan(
            spec,
            source,
            medium,
            parameters,
            HistoryBatch(17, 0, 64),
            GeometrySamplingDesign("iid", 4, 2, 2),
        )


def test_empty_medium_uses_actual_owned_cell_factor():
    spec, source, medium, parameters = _model()
    plan = _geometry_plan(
        spec,
        source,
        dataclasses.replace(medium, voxel_density=(0.0,)),
        parameters,
        HistoryBatch(17, 0, 64),
        GeometrySamplingDesign("iid", 4, 2, 2),
    )
    assert all(column.empty_medium for column in plan.columns)
    assert all(not row.present for column in plan.columns for row in column.source_boundaries)


def test_unused_material_rates_do_not_turn_a_transparent_object_into_a_chart_gate():
    spec, _, _, _ = _model()
    spec = dataclasses.replace(spec, grid=dataclasses.replace(spec.grid, materials=2))
    source = PointConeSource(
        DetectorGeometry(
            (0.0, 0.45, 0.55),
            (-1.2, -1.3, 3.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.7, 0.8),
            (4, 4),
        )
    )
    medium = GeometryPhysicalMedium((0,), (0.0, 10.0), (0.0, 4.0), (1.0, 1.0), (1.0,))
    plan = _geometry_plan(
        spec,
        source,
        medium,
        (GeometryParameter("object", "translation", 0),),
        HistoryBatch(17, 0, 64),
        GeometrySamplingDesign("iid", 4, 2, 2),
    )
    assert plan.columns[0].empty_medium
    assert not any(row.present for row in plan.columns[0].source_boundaries)
    assert len(plan.columns[0].detector_boundaries) == 10


def test_initial_opacity_keeps_eighteen_charts_in_a_zero_rate_medium():
    spec, source, _, _ = _model()
    atomic = AtomicScatteringTable(
        1,
        (0.0, 8.0),
        (1.0, 0.0),
        (0.0, 1.0),
        Provenance(
            "https://doi.org/10.1063/1.555523",
            hashlib.sha256(b"Z=1;q=(0,8);F=(1,0);S=(0,1);linear-in-q").hexdigest(),
            "CC0-1.0 analytic control",
            "Declared linear atomic sum-rule control, used with zero scattering rate",
        ),
    )
    physical = PhysicalTransport((atomic,), (0.0,), cutoff_kev=80.0)
    plan = _geometry_plan(
        spec,
        source,
        GeometryPhysicalMedium((0,), (0.0,), (0.0,), (1.0,), (1.0,), physical),
        (GeometryParameter("object", "grid-spacing", 0),),
        HistoryBatch(17, 0, 64),
        GeometrySamplingDesign("iid", 4, 2, 2),
    )
    column = plan.columns[0]
    assert column.empty_medium and not any(row.present for row in column.source_boundaries)
    assert len(column.opaque_boundaries) == 18
    assert any(row is not None for row in column.opaque_boundaries)


@pytest.mark.parametrize("density,voxel", ((1e-300, 1e-300), (1e300, 1e300)))
def test_positive_cell_factor_range_loss_is_rejected_before_empty_optimization(density, voxel):
    spec, source, medium, parameters = _model()
    with pytest.raises(TransportError, match=r"cell factor.*binary64"):
        _geometry_plan(
            spec,
            source,
            dataclasses.replace(medium, density=(density,), voxel_density=(voxel,)),
            parameters,
            HistoryBatch(17, 0, 64),
            GeometrySamplingDesign("iid", 4, 2, 2),
        )


def test_positive_rate_underflow_does_not_prove_an_exactly_empty_medium():
    spec, source, medium, parameters = _model()
    with pytest.raises(TransportError, match="representable cell attenuation"):
        _geometry_plan(
            spec,
            source,
            dataclasses.replace(
                medium, absorption_mm_inverse=(1e-100,), density=(1e-200,), voxel_density=(1e-100,)
            ),
            parameters,
            HistoryBatch(17, 0, 64),
            GeometrySamplingDesign("iid", 4, 2, 2),
        )


@pytest.mark.parametrize("majorant,admitted", ((7.0, True), (6.0, False), (5.0, False)))
def test_fixed_majorant_original_policy_requires_strict_owned_density_slack(majorant, admitted):
    spec, source, _, parameters = _model()
    atomic = AtomicScatteringTable(
        1,
        (0.0, 8.0),
        (1.0, 0.0),
        (0.0, 1.0),
        Provenance(
            "https://doi.org/10.1063/1.555523",
            hashlib.sha256(b"Z=1;q=(0,8);F=(1,0);S=(0,1);linear-in-q").hexdigest(),
            "CC0-1.0 analytic control",
            "Declared linear atomic sum-rule control, used with zero scattering rate",
        ),
    )
    physical = PhysicalTransport((atomic,), (0.0,), majorant_mm_inverse=(majorant,))
    medium = GeometryPhysicalMedium((0,), (1.0,), (0.0,), (2.0,), (3.0,), physical)
    arguments = (spec, source, medium, parameters, HistoryBatch(17, 0, 64))
    design = GeometrySamplingDesign("iid", 4, 2, 2)
    if admitted:
        plan = _geometry_plan(*arguments, design)
        assert plan.identity.medium.physical is physical
    else:
        with pytest.raises(TransportError, match="majorant requires strict slack"):
            _geometry_plan(*arguments, design)


def test_full_source_startup_kink_is_rejected_before_sampling_or_uploads():
    spec, source, medium, _ = _model()
    source = dataclasses.replace(source, lower_mm=(-0.2, -0.1, 0.0))
    with pytest.raises(TransportError, match=r"launch plane.*discontinuous transverse"):
        _geometry_plan(
            spec,
            source,
            medium,
            (GeometryParameter("object", "translation", 2),),
            HistoryBatch(17, 0, 64),
            GeometrySamplingDesign("iid", 4, 2, 2),
        )


@pytest.mark.parametrize("active_noncoherent", (True, False))
def test_reachable_bound_normaliser_admission_is_part_of_public_preparation(active_noncoherent):
    spec, _, _, parameters = _model()
    spec = dataclasses.replace(
        spec,
        energy_kev=0.5,
        coefficient_energies_kev=(0.1, 0.5, 2.0),
        scattering_law="bound-electron-compton",
    )
    provenance = Provenance(
        "https://doi.org/10.1016/0168-9002(96)00652-3",
        hashlib.sha256(b"Z=1;K:occupancy=1;binding-keV=1;profile-zero=0.1").hexdigest(),
        "CC0-1.0 analytic control",
        "Declared one-shell analytical impulse law for reachable-normaliser admission",
    )
    atomic = AtomicScatteringTable(1, (0.0, 8.0), (1.0, 0.0), (0.0, 1.0), provenance)
    shells = BoundComptonTable(1, (ComptonShell("K", 1.0, 1.0, 0.1),), provenance)
    physical = PhysicalTransport(
        (atomic,), (0.0, 0.0, 0.0), cutoff_kev=0.1, bound_compton=(shells,)
    )
    medium = GeometryPhysicalMedium(
        (0,),
        (0.1, 0.1, 0.1),
        (1.0, 1.0, 1.0) if active_noncoherent else (0.0, 0.0, 1.0),
        (1.0,),
        (1.0,),
        physical,
    )
    source = ParallelBeam((0.4, 0.45, -1.0), (0.0, 0.0))
    arguments = (spec, source, medium, parameters[:1], HistoryBatch(17, 0, 64))
    design = GeometrySamplingDesign("iid", 4, 2, 2, layout="bound-compton-v3")
    if active_noncoherent:
        with pytest.raises(TransportError, match=r"reachable.*zero accessible-shell normaliser"):
            _geometry_plan(*arguments, design)
    else:
        plan = _geometry_plan(*arguments, design)
        assert plan.bound_admission is not None
        assert plan.bound_admission.source_cells == (0,)
        assert plan.bound_admission.source_energies_kev == (0.5,)


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("record-type", "PhysicalTransport record"),
        ("atomic-owner", "one scattering table"),
        ("rayleigh-shape", r"Rayleigh table.*shape"),
        ("rayleigh-partial", "Rayleigh partial"),
        ("fluorescence-owner", r"branches.*material count"),
        ("fluorescence-record", r"probabilities.*energy nodes"),
        ("fluorescence-nodes", r"probabilities.*energy nodes"),
        ("fluorescence-threshold", "below its vacancy threshold"),
        ("fluorescence-total", "probabilities exceed one"),
        ("line-support", "line leaves coefficient support"),
        ("cutoff-support", "cutoff exceeds"),
        ("spectrum-support", "spectrum leaves"),
        ("angular-support", "q support"),
        ("majorant-shape", r"majorant.*energy nodes"),
        ("forward-policy", "require analogue"),
    ),
)
def test_physical_structure_admission_precedes_chart_indexing_or_uploads(mutation, match):
    spec, source, medium, parameters = _model()
    provenance = Provenance(
        "https://doi.org/10.1063/1.555523",
        hashlib.sha256(b"Z=1;q=(0,8);F=(1,0);S=(0,1);linear-in-q").hexdigest(),
        "CC0-1.0 analytic control",
        "Declared linear atomic sum-rule control for immutable model admission",
    )
    atomic = AtomicScatteringTable(1, (0.0, 8.0), (1.0, 0.0), (0.0, 1.0), provenance)
    physical = PhysicalTransport((atomic,), (0.0,))
    line = FluorescenceLine(1.0, 2.0, (0.6,), provenance)
    if mutation == "record-type":
        physical = cast(Any, object())
    elif mutation == "atomic-owner":
        physical = dataclasses.replace(physical, scattering_tables=(atomic, atomic))
    elif mutation == "rayleigh-shape":
        physical = dataclasses.replace(physical, rayleigh_mm_inverse=())
    elif mutation == "rayleigh-partial":
        physical = dataclasses.replace(physical, rayleigh_mm_inverse=(0.1,))
    elif mutation == "fluorescence-owner":
        physical = dataclasses.replace(physical, fluorescence=((), ()))
    elif mutation == "fluorescence-record":
        physical = dataclasses.replace(physical, fluorescence=((cast(Any, object()),),))
    elif mutation == "fluorescence-nodes":
        physical = dataclasses.replace(
            physical, fluorescence=((dataclasses.replace(line, probabilities=(0.1, 0.1)),),)
        )
    elif mutation == "fluorescence-threshold":
        physical = dataclasses.replace(
            physical,
            cutoff_kev=75.0,
            fluorescence=((FluorescenceLine(70.0, 100.0, (0.1,), provenance),),),
        )
    elif mutation == "fluorescence-total":
        physical = dataclasses.replace(physical, cutoff_kev=2.0, fluorescence=((line, line),))
    elif mutation == "line-support":
        physical = dataclasses.replace(
            physical, fluorescence=((FluorescenceLine(70.0, 100.0, (0.0,), provenance),),)
        )
    elif mutation == "cutoff-support":
        physical = dataclasses.replace(physical, cutoff_kev=81.0)
    elif mutation == "spectrum-support":
        physical = dataclasses.replace(
            physical,
            spectrum=SourceSpectrum((70.0, 80.0), (0.5, 0.5), "Declared admission control"),
        )
    elif mutation == "angular-support":
        physical = dataclasses.replace(
            physical,
            scattering_tables=(
                AtomicScatteringTable(1, (0.0, 1.0), (1.0, 0.0), (0.0, 1.0), provenance),
            ),
        )
    elif mutation == "majorant-shape":
        physical = dataclasses.replace(physical, majorant_mm_inverse=(1.0, 2.0))
    elif mutation == "forward-policy":
        spec = dataclasses.replace(spec, estimator="continuous-absorption")
        physical = dataclasses.replace(physical, majorant_mm_inverse=(1.0,))
    medium = dataclasses.replace(medium, physical=physical)
    with pytest.raises(TransportError, match=match):
        _geometry_plan(
            spec,
            source,
            medium,
            parameters,
            HistoryBatch(17, 0, 64),
            GeometrySamplingDesign("iid", 4, 2, 2),
        )
