"""Prepared public geometry execution against absorbing and transparent laws."""

# Runtime Warp arrays are intentionally opaque to Python's static provider.
# pyright: reportUnknownMemberType=false, reportPrivateUsage=false
import dataclasses
import hashlib
import importlib
import math
from typing import Any, Literal

import numpy as np
import pytest

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.materials import Provenance
from dpt.transport._constants import BUDGET_MASK, INPUT_MASK
from dpt.transport.geometry_composed import composed_geometry_observation
from dpt.transport.geometry_derivatives import prepare_transport_geometry, transport_geometry_vjp
from dpt.transport.geometry_histories import GeometryPhysicalMedium
from dpt.transport.geometry_parameters import GeometryParameter
from dpt.transport.geometry_sampling import GeometryHostSampler, GeometrySamplingDesign
from dpt.transport.model import (
    IncompleteHistoryError,
    MaterialGrid,
    OrientedDetector,
    PlanarDetector,
    TransportError,
)
from dpt.transport.physical import (
    AtomicScatteringTable,
    PhysicalTransport,
)
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam, PointConeSource
from helpers.transport_geometry_cases import _outputs, _spec
from helpers.transport_geometry_cases import actual_aluminium as actual_aluminium

wp: Any = importlib.import_module("warp")


def _prepared(device, law: Literal["iid", "stratified", "scrambled-sobol"] = "iid", count=128):
    return prepare_transport_geometry(
        _spec(),
        source=ParallelBeam((0.4, 0.45, -1.0), (0.0, 0.0)),
        medium=GeometryPhysicalMedium((0,), (0.7,), (0.0,), (1.0,), (1.0,)),
        parameters=(
            GeometryParameter("object", "grid-spacing", 2),
            GeometryParameter("source", "extent", 0),
            GeometryParameter("detector", "translation", 0),
        ),
        batch=HistoryBatch(19, 0, count),
        design=GeometrySamplingDesign(law, 4, 2, 2),
        device=device,
    )


@pytest.mark.parametrize("law", ("iid", "stratified", "scrambled-sobol"))
def test_public_radiative_whole_plane_uses_owned_masks_and_complete_original_histories(
    law, warp_device
):
    spec = dataclasses.replace(
        _spec(),
        grid=MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 2), 2),
        max_events=8,
    )
    source = ParallelBeam((1, 0.45, -1), (0, 0))
    medium = GeometryPhysicalMedium((0, 1), (0.7, 0.7), (0.01, 0.01), (1, 1), (1, 1))
    workspace = prepare_transport_geometry(
        spec,
        source=source,
        medium=medium,
        parameters=tuple(
            GeometryParameter(owner, "translation", 0) for owner in ("object", "source", "detector")
        ),
        batch=HistoryBatch(19, 0, 8),
        design=GeometrySamplingDesign(law, 8, 2, 2),
        device=warp_device,
    )
    assert workspace.identity.chart_masks == ((0, 1, 0, 0, 0, 0, 0),) * 3
    assert len(workspace._plane_masks) == 3 and workspace._planes.size == 4
    for mask in workspace._plane_masks:
        np.testing.assert_array_equal(mask.numpy(), (0, 1, 0, 0, 0, 0, 0))
        assert mask.device == workspace.context.device and mask.dtype == wp.int32
    assert len({mask.ptr for mask in workspace._plane_masks}) == 3
    outputs = _outputs(workspace)
    with workspace.context.scope():
        seed = wp.array([-2.0], dtype=wp.float64, device=workspace.context.device)
    before_bytes = workspace.allocated_bytes
    transport_geometry_vjp(seed, workspace=workspace, **outputs)
    assert workspace.allocated_bytes == before_bytes
    provider = GeometryHostSampler(workspace.identity.sampling)
    expected = []
    for history in range(workspace.batch.count):
        uv = (
            provider.uniform((history, 0, 0), 0, 0, 0),
            provider.uniform((history, 0, 0), 0, 0, 1),
        )
        expected.append(
            [
                composed_geometry_observation(
                    source,
                    uv,
                    80,
                    spec,
                    medium,
                    provider,
                    original_history=history,
                    cotangent=(-2.0,),
                    tangent=parameter.tangent(source),
                )
                for parameter in workspace.parameters
            ]
        )
    values = np.asarray([[item.mean for item in row] for row in expected])
    derivatives = np.asarray([[item.derivative for item in row] for row in expected])
    counts = np.asarray(
        [
            [(item.physical_prefixes, item.continuation_histories) for item in row]
            for row in expected
        ]
    )
    np.testing.assert_allclose(outputs["out_values"].numpy(), values, rtol=3e-11, atol=2e-12)
    np.testing.assert_allclose(
        outputs["out_derivatives"].numpy(), derivatives, rtol=3e-11, atol=2e-12
    )
    np.testing.assert_array_equal(outputs["out_counts"].numpy(), counts)
    np.testing.assert_allclose(
        outputs["out_second_moment"].numpy(),
        np.mean(derivatives**2, axis=0),
        rtol=3e-11,
        atol=2e-12,
    )
    assert np.sum(counts[:, :, 0]) > 0
    assert np.min(values) < -2 * math.exp(-0.71)


@pytest.mark.parametrize("law", ("iid", "stratified", "scrambled-sobol"))
def test_public_mean_pullback_matches_exact_original_photon_observations(law, warp_device):
    workspace = _prepared(warp_device, law)
    outputs = _outputs(workspace)
    with workspace.context.scope():
        seed = wp.array([-2.0], dtype=wp.float64, device=workspace.context.device)
    transport_geometry_vjp(seed, workspace=workspace, **outputs)
    values, derivatives = outputs["out_values"].numpy(), outputs["out_derivatives"].numpy()
    exact_mean = -2 * math.exp(-0.7)
    np.testing.assert_allclose(values, exact_mean, rtol=4e-14, atol=0.0)
    provider = GeometryHostSampler(workspace.identity.sampling)
    # Independent absorbing-slab derivative: only the moving upper z plane
    # contributes, with inverse discrete plane probability six.
    exact = np.asarray(
        [
            12 * 0.7 * math.exp(-0.7)
            if int(provider.uniform((h, 0, 0), 0, 0x30000003, 0) * 6) == 5
            else 0.0
            for h in range(workspace.batch.count)
        ]
    )
    np.testing.assert_allclose(derivatives[:, 0], exact, rtol=4e-14, atol=2e-15)
    np.testing.assert_array_equal(derivatives[:, 1:], 0.0)
    np.testing.assert_allclose(outputs["out_mean"].numpy(), exact_mean, rtol=4e-14)
    np.testing.assert_allclose(
        outputs["out_gradient"].numpy(), np.mean(derivatives, axis=0), rtol=4e-14, atol=0.0
    )
    np.testing.assert_allclose(
        outputs["out_second_moment"].numpy(), np.mean(derivatives**2, axis=0), rtol=4e-14, atol=0.0
    )
    assert np.all(outputs["out_counts"].numpy() >= 0)
    before = {name: array.numpy().copy() for name, array in outputs.items()}
    pointers = {name: array.ptr for name, array in outputs.items()}
    bytes_before = workspace.allocated_bytes
    transport_geometry_vjp(seed, workspace=workspace, **outputs)
    assert workspace.allocated_bytes == bytes_before
    for name, array in outputs.items():
        assert array.ptr == pointers[name]
        np.testing.assert_array_equal(array.numpy(), before[name])
    assert workspace.reference_spec.estimator == "analogue"
    assert workspace._reference._parameters.max_events == 4
    assert workspace._reference._parameters.sampling.max_events == 16


def test_point_emitter_inside_transparent_box_and_signed_energy_score(warp_device):
    spec = dataclasses.replace(
        _spec(),
        detector=PlanarDetector((-2.0, -2.0), (1.0, 1.0), (4, 4), 3.0),
        scoring="energy-kev",
        estimator="continuous-absorption",
    )
    source = PointConeSource(
        DetectorGeometry(
            (0.4, 0.45, 0.55),
            (-1.2, -1.3, 3.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.7, 0.8),
            (4, 4),
        ),
        weight=2.5,
    )
    workspace = prepare_transport_geometry(
        spec,
        source=source,
        medium=GeometryPhysicalMedium((0,), (0.0,), (0.0,), (1.0,), (1.0,)),
        parameters=(
            GeometryParameter("source", "translation", 0),
            GeometryParameter("launch-target", "pixel-spacing", 0),
            GeometryParameter("detector", "pixel-spacing", 1),
        ),
        batch=HistoryBatch(23, 7, 64),
        design=GeometrySamplingDesign("scrambled-sobol", 4, 2, 2),
        device=warp_device,
    )
    outputs = _outputs(workspace)
    with workspace.context.scope():
        seed = wp.full(
            spec.detector.pixels, -1.5, dtype=wp.float64, device=workspace.context.device
        )
    transport_geometry_vjp(seed, workspace=workspace, **outputs)
    np.testing.assert_allclose(outputs["out_values"].numpy(), -1.5 * 2.5 * 80.0, rtol=4e-14)
    np.testing.assert_allclose(outputs["out_mean"].numpy(), -1.5 * 2.5 * 80.0, rtol=4e-14)
    np.testing.assert_array_equal(outputs["out_derivatives"].numpy(), 0.0)
    np.testing.assert_array_equal(outputs["out_gradient"].numpy(), 0.0)
    np.testing.assert_array_equal(outputs["out_second_moment"].numpy(), 0.0)
    assert workspace.spec.estimator == "continuous-absorption"
    assert workspace.reference_spec.estimator == "analogue"


@pytest.mark.parametrize("corner", (False, True))
def test_public_regular_whole_source_faces_match_identical_generator_slab(corner, warp_device):
    spec = dataclasses.replace(
        _spec(),
        grid=MaterialGrid(
            (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 2, 2) if corner else (1, 1, 2), 4 if corner else 2
        ),
    )
    source = ParallelBeam((1.0, 1.0 if corner else 0.5, -1.0), (0.0, 0.0))
    rates = (0.7, 1.3, 1.7, 0.7) if corner else (0.7, 0.7)
    cells = len(rates)
    medium = GeometryPhysicalMedium(
        tuple(range(cells)), rates, (0.0,) * cells, (1.0,) * cells, (1.0,) * cells
    )
    parameter = (
        GeometryParameter("object", "rotation", 2, (2.0, 0.0, 0.0))
        if corner
        else GeometryParameter("object", "translation", 0)
    )
    workspace = prepare_transport_geometry(
        spec,
        source=source,
        medium=medium,
        parameters=(parameter,),
        batch=HistoryBatch(61, 0, 64),
        design=GeometrySamplingDesign("iid", 4, 2, 2),
        device=warp_device,
    )
    outputs = _outputs(workspace)
    with workspace.context.scope():
        seed = wp.array([-2.0], dtype=wp.float64, device=workspace.context.device)
    transport_geometry_vjp(seed, workspace=workspace, **outputs)
    # The two approached media have the same absorption rate and one-mm chord.
    # Rotation about z preserves that chord and never enters the other quadrants.
    np.testing.assert_allclose(outputs["out_values"].numpy(), -2.0 * math.exp(-0.7), rtol=4e-14)
    np.testing.assert_allclose(outputs["out_derivatives"].numpy(), 0.0, atol=2e-13)
    np.testing.assert_allclose(outputs["out_gradient"].numpy(), 0.0, atol=2e-13)
    np.testing.assert_allclose(outputs["out_second_moment"].numpy(), 0.0, atol=2e-26)
    assert np.all(workspace._source_boundaries.numpy()["present"] == 0)


@pytest.mark.parametrize("law", ("iid", "stratified", "scrambled-sobol"))
def test_coplanar_point_emitter_uses_complete_angular_source_rows(law, warp_device):
    spec = _spec()
    source = PointConeSource(
        DetectorGeometry(
            (0.0, 0.5, -1.0),
            (0.0, 0.5, 3.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.8),
            (1, 1),
        )
    )
    medium = GeometryPhysicalMedium((0,), (0.7,), (0.0,), (1.0,), (1.0,))
    parameter = GeometryParameter("launch-target", "translation", 0)
    workspace = prepare_transport_geometry(
        spec,
        source=source,
        medium=medium,
        parameters=(parameter,),
        batch=HistoryBatch(53, 0, 64),
        design=GeometrySamplingDesign(law, 4, 2, 2),
        device=warp_device,
    )
    outputs = _outputs(workspace)
    with workspace.context.scope():
        seed = wp.array([-2.0], dtype=wp.float64, device=workspace.context.device)
    transport_geometry_vjp(seed, workspace=workspace, **outputs)
    host = GeometryHostSampler(workspace.identity.sampling)
    expected = [
        composed_geometry_observation(
            source,
            (
                host.uniform((history, 0, 0), 0x80000000, 0, 0),
                host.uniform((history, 0, 0), 0x80000000, 0, 1),
            ),
            spec.energy_kev,
            spec,
            medium,
            host,
            original_history=history,
            cotangent=(-2.0,),
            tangent=parameter.tangent(source),
        )
        for history in range(workspace.batch.count)
    ]
    np.testing.assert_allclose(
        outputs["out_values"].numpy()[:, 0],
        [row.mean for row in expected],
        rtol=3e-11,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        outputs["out_derivatives"].numpy()[:, 0],
        [row.derivative for row in expected],
        rtol=3e-11,
        atol=2e-12,
    )
    np.testing.assert_array_equal(
        outputs["out_counts"].numpy()[:, 0],
        [(row.physical_prefixes, row.continuation_histories) for row in expected],
    )
    assert any(row.grazing_source != 0.0 for row in expected)
    assert np.any(workspace._source_boundaries.numpy()["present"] != 0)


def test_physical_initial_cutoff_precedes_empty_extinction_optimization(warp_device):
    # A declared zero-rate medium still deposits a subcutoff photon on box entry.
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
    workspace = prepare_transport_geometry(
        _spec(),
        source=ParallelBeam((0.4, 0.45, -1.0), (0.0, 0.0)),
        medium=GeometryPhysicalMedium((0,), (0.0,), (0.0,), (1.0,), (1.0,), physical),
        parameters=(GeometryParameter("object", "grid-spacing", 2),),
        batch=HistoryBatch(31, 0, 64),
        design=GeometrySamplingDesign("stratified", 4, 2, 2),
        device=warp_device,
    )
    outputs = _outputs(workspace)
    with workspace.context.scope():
        seed = wp.array([-2.0], dtype=wp.float64, device=workspace.context.device)
    transport_geometry_vjp(seed, workspace=workspace, **outputs)
    for name in ("out_values", "out_derivatives", "out_mean", "out_gradient", "out_second_moment"):
        np.testing.assert_array_equal(outputs[name].numpy(), 0.0)


def test_runtime_alias_metadata_and_sticky_finite_seed_admission(warp_device):
    workspace = _prepared(warp_device, count=32)
    outputs = _outputs(workspace)
    with workspace.context.scope():
        seed = wp.array([-2.0], dtype=wp.float64, device=workspace.context.device)
        bad_seed = wp.array([math.inf], dtype=wp.float64, device=workspace.context.device)
        wrong_mean = wp.empty(3, dtype=wp.float32, device=workspace.context.device)
    with pytest.raises(TransportError, match="overlaps"):
        transport_geometry_vjp(
            seed, workspace=workspace, **{**outputs, "out_gradient": outputs["out_mean"]}
        )
    with pytest.raises(TransportError, match="dtype"):
        transport_geometry_vjp(seed, workspace=workspace, **{**outputs, "out_mean": wrong_mean})
    with pytest.raises(TransportError, match="shape"):
        transport_geometry_vjp(
            seed, workspace=workspace, **{**outputs, "out_values": outputs["out_values"][:31]}
        )
    scalar_outputs = (
        "out_values",
        "out_derivatives",
        "out_mean",
        "out_gradient",
        "out_second_moment",
    )
    with workspace.context.scope():
        for name in scalar_outputs:
            outputs[name].fill_(17.0)
        outputs["out_counts"].zero_()
    with pytest.raises(TransportError, match="invalid device transport inputs"):
        transport_geometry_vjp(bad_seed, workspace=workspace, **outputs)
    assert int(workspace._reference._status.numpy()[0]) & INPUT_MASK
    for name in scalar_outputs:
        np.testing.assert_array_equal(outputs[name].numpy(), 17.0)
    np.testing.assert_array_equal(outputs["out_counts"].numpy(), 0)
    workspace.clear_status()
    transport_geometry_vjp(bad_seed, workspace=workspace, validate=False, **outputs)
    resident_status = int(workspace._reference._status.numpy()[0])
    assert resident_status & INPUT_MASK == INPUT_MASK and resident_status & BUDGET_MASK
    for name in scalar_outputs:
        assert np.all(np.isnan(outputs[name].numpy()))
    with pytest.raises(IncompleteHistoryError, match="discard the estimate"):
        workspace.check_status()
    with pytest.raises(IncompleteHistoryError, match="discard the estimate"):
        transport_geometry_vjp(seed, workspace=workspace, **outputs)
    workspace.clear_status()
    transport_geometry_vjp(seed, workspace=workspace, **outputs)
    assert np.all(np.isfinite(outputs["out_mean"].numpy()))


@pytest.mark.parametrize(
    "law,packed",
    (
        ("isotropic-elastic", False),
        ("free-electron-compton", False),
        ("free-electron-compton", True),
        ("bound-electron-compton", False),
    ),
)
def test_public_owned_full_physical_model_matches_host_original_histories(
    law, packed, actual_aluminium, warp_device
):
    energies, absorption, scattering, physical, shells = actual_aluminium
    if law == "bound-electron-compton":
        physical = dataclasses.replace(physical, bound_compton=(shells,))
    spec = dataclasses.replace(
        _spec(),
        detector=PlanarDetector((-2.0, -2.0), (1.0, 1.0), (4, 4), 3.0),
        energy_kev=10.0,
        coefficient_energies_kev=energies,
        scoring="energy-kev",
        scattering_law=law,
        max_angle_trials=64,
        compton_sampler="packed-uniform-rejection" if packed else "legacy-rejection",
    )
    source = ParallelBeam((0.4, 0.45, -1.0), (0.0, 0.0), weight=1.7)
    medium = GeometryPhysicalMedium((0,), absorption, scattering, (1.0,), (1.0,), physical)
    parameters = (
        GeometryParameter("object", "grid-spacing", 2),
        GeometryParameter("source", "extent", 0),
        GeometryParameter("detector", "translation", 0),
    )
    layout = (
        "bound-compton-v3"
        if law == "bound-electron-compton"
        else "packed-compton-v2"
        if packed
        else "legacy-v1"
    )
    workspace = prepare_transport_geometry(
        spec,
        source=source,
        medium=medium,
        parameters=parameters,
        batch=HistoryBatch(19, 0, 8),
        design=GeometrySamplingDesign("iid", 4, 64, 4, layout=layout),
        device=warp_device,
    )
    outputs = _outputs(workspace)
    seed = tuple((-1.0 if index % 3 == 0 else 1.0) * (index + 1) for index in range(16))
    with workspace.context.scope():
        resident_seed = wp.array(seed, dtype=wp.float64, device=workspace.context.device)
    transport_geometry_vjp(resident_seed, workspace=workspace, **outputs)
    host = GeometryHostSampler(workspace.identity.sampling)
    expected = []
    for history in range(workspace.batch.count):
        coordinate = (
            host.uniform((history, 0, 0), 0x80000000, 0, 0),
            host.uniform((history, 0, 0), 0x80000000, 0, 1),
        )
        initial_energy = 10.0 if host.uniform((history, 0, 0), 0x80000002, 0, 0) < 0.5 else 80.0
        expected.append(
            [
                composed_geometry_observation(
                    source,
                    coordinate,
                    initial_energy,
                    spec,
                    medium,
                    host,
                    original_history=history,
                    cotangent=seed,
                    tangent=parameter.tangent(source),
                )
                for parameter in parameters
            ]
        )
    reference_values = np.asarray([[item.mean for item in row] for row in expected])
    reference_derivatives = np.asarray([[item.derivative for item in row] for row in expected])
    reference_counts = np.asarray(
        [
            [(item.physical_prefixes, item.continuation_histories) for item in row]
            for row in expected
        ]
    )
    np.testing.assert_allclose(
        outputs["out_values"].numpy(), reference_values, rtol=3e-11, atol=2e-12
    )
    np.testing.assert_allclose(
        outputs["out_derivatives"].numpy(), reference_derivatives, rtol=3e-11, atol=2e-12
    )
    np.testing.assert_array_equal(outputs["out_counts"].numpy(), reference_counts)
    np.testing.assert_allclose(
        outputs["out_gradient"].numpy(), reference_derivatives.mean(axis=0), rtol=3e-11, atol=2e-12
    )
    assert np.count_nonzero(reference_values) > 0
    assert physical.spectrum is not None and workspace.identity.medium.physical is physical


@pytest.mark.parametrize(
    "law,packed",
    (
        ("isotropic-elastic", False),
        ("free-electron-compton", False),
        ("free-electron-compton", True),
        ("bound-electron-compton", False),
    ),
)
@pytest.mark.parametrize("trial_guard", (64, 256))
def test_rotated_coplanar_cone_retains_full_physical_conditional_futures(
    law, packed, trial_guard, actual_aluminium, warp_device
):
    energies, absorption, scattering, physical, shells = actual_aluminium
    if law == "bound-electron-compton":
        physical = dataclasses.replace(physical, bound_compton=(shells,))
    angle = 0.73
    cosine, sine = math.cos(angle), math.sin(angle)
    pose = RigidTransform((cosine, 0.0, sine, 0.0, 1.0, 0.0, -sine, 0.0, cosine))
    emitter = pose.point((0.5, -1.0, 0.0))
    target = pose.point((0.5, 3.0, 0.0))
    u, v = pose.direction((1.0, 0.0, 0.0)), pose.direction((0.0, 0.0, 1.0))
    source = PointConeSource(DetectorGeometry(emitter, target, u, v, (0.8, 2.0), (1, 1)))
    spec = dataclasses.replace(
        _spec(),
        detector=OrientedDetector(DetectorGeometry(emitter, target, u, v, (6.0, 6.0), (1, 1))),
        object_pose=pose,
        energy_kev=10.0,
        coefficient_energies_kev=energies,
        scoring="energy-kev",
        scattering_law=law,
        max_angle_trials=trial_guard,
        compton_sampler="packed-uniform-rejection" if packed else "legacy-rejection",
    )
    medium = GeometryPhysicalMedium((0,), absorption, scattering, (1.0,), (1.0,), physical)
    parameter = GeometryParameter("launch-target", "translation", 0)
    layout = (
        "bound-compton-v3"
        if law == "bound-electron-compton"
        else "packed-compton-v2"
        if packed
        else "legacy-v1"
    )
    workspace = prepare_transport_geometry(
        spec,
        source=source,
        medium=medium,
        parameters=(parameter,),
        batch=HistoryBatch(71, 0, 64),
        design=GeometrySamplingDesign("iid", 4, trial_guard, 4, layout=layout),
        device=warp_device,
    )
    outputs = _outputs(workspace)
    with workspace.context.scope():
        cotangent = wp.array([-2.0], dtype=wp.float64, device=workspace.context.device)
    if trial_guard == 64:
        # The complete fixed 4x64-history ledger requires trial143 at original
        # history34, branch1/event0, Rayleigh80keV. The original guard must
        # invalidate that observation and every reduced estimate.
        with pytest.raises(IncompleteHistoryError, match="budget exhausted"):
            transport_geometry_vjp(cotangent, workspace=workspace, **outputs)
        status = int(workspace._reference._status.numpy()[0])
        assert status & BUDGET_MASK and not status & INPUT_MASK
        bad = np.flatnonzero(~np.isfinite(outputs["out_values"].numpy()[:, 0]))
        np.testing.assert_array_equal(bad, [34])
        assert math.isnan(float(outputs["out_derivatives"].numpy()[34, 0]))
        np.testing.assert_array_equal(outputs["out_counts"].numpy()[34, 0], [0, 0])
        for name in ("out_mean", "out_gradient", "out_second_moment"):
            assert np.isnan(outputs[name].numpy()).all()
        return
    transport_geometry_vjp(cotangent, workspace=workspace, **outputs)
    provider = GeometryHostSampler(workspace.identity.sampling)
    references = []
    for history in range(workspace.batch.count):
        coordinate = (
            provider.uniform((history, 0, 0), 0x80000000, 0, 0),
            provider.uniform((history, 0, 0), 0x80000000, 0, 1),
        )
        energy = 10.0 if provider.uniform((history, 0, 0), 0x80000002, 0, 0) < 0.5 else 80.0
        references.append(
            composed_geometry_observation(
                source,
                coordinate,
                energy,
                spec,
                medium,
                provider,
                original_history=history,
                cotangent=(-2.0,),
                tangent=parameter.tangent(source),
            )
        )
    derivative = np.asarray([row.derivative for row in references])
    np.testing.assert_allclose(
        outputs["out_values"].numpy()[:, 0],
        [row.mean for row in references],
        rtol=4e-11,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        outputs["out_derivatives"].numpy()[:, 0], derivative, rtol=4e-11, atol=3e-12
    )
    np.testing.assert_array_equal(
        outputs["out_counts"].numpy()[:, 0],
        [(row.physical_prefixes, row.continuation_histories) for row in references],
    )
    np.testing.assert_allclose(
        outputs["out_gradient"].numpy(), derivative.mean(), rtol=4e-11, atol=3e-12
    )
    np.testing.assert_allclose(
        outputs["out_second_moment"].numpy(), np.mean(derivative**2), rtol=4e-11, atol=3e-12
    )
    assert any(row.grazing_source != 0.0 for row in references)
    assert any(row.physical_prefixes > 0 for row in references)
    assert np.any(workspace._source_boundaries.numpy()["present"] != 0)
