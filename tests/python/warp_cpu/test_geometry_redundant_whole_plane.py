"""Original-history native assembly against independent absorbing-slab laws."""

# Warp is an explicit runtime DSL provider for the annotations below.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
import importlib
import math
from dataclasses import fields
from typing import Any

import numpy as np
import pytest

from dpt.transport.compton_boundary_arguments import free_compton_boundary_arguments
from dpt.transport.compton_geometry_boundary import FreeComptonEnergyBoundary
from dpt.transport.forward import prepare_transport
from dpt.transport.geometry_composed import (
    composed_geometry_observation,
    material_patch_at,
    material_plane_at,
)
from dpt.transport.geometry_detector_proposal_arguments import detector_proposal_plan_arguments
from dpt.transport.geometry_histories import GeometryPhysicalMedium
from dpt.transport.geometry_interface_arguments import interface_arguments
from dpt.transport.geometry_sampling import (
    GeometryHostSampler,
    GeometrySamplingDesign,
    prepare_geometry_sampling,
)
from dpt.transport.geometry_source_boundary_arguments import source_geometry_arguments
from dpt.transport.geometry_tangents import GridGeometryTangent, RayTangent, RigidMotion
from dpt.transport.geometry_whole_source_admission import admit_whole_source_geometry
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam
from dpt.transport.uncollided_geometry import AcquisitionGeometryTangent, SourceGeometryTangent

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.geometry_driver_kernels")
interfaces: Any = importlib.import_module("dpt.transport.geometry_interface_kernels")
sources: Any = importlib.import_module("dpt.transport.geometry_source_boundary_kernels")
boundaries: Any = importlib.import_module("dpt.transport.compton_geometry_boundary_kernels")
proposals: Any = importlib.import_module("dpt.transport.geometry_detector_proposal_kernels")
forward_proposals: Any = importlib.import_module("dpt.transport.geometry_forward_proposal_kernels")
opaque: Any = importlib.import_module("dpt.transport.geometry_opaque_kernels")


def _pack(record, arguments):
    for field in fields(arguments):
        setattr(record, field.name, getattr(arguments, field.name))
    return record


def _run(device, *, seed_value=-2.0):
    count = 8
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 2), 2),
        PlanarDetector((-5, -5), (10, 10), (1, 1), 3),
        80,
        "independent original-history absorbing slab",
        (80,),
        max_events=8,
        max_angle_trials=2,
    )
    source = ParallelBeam((1, 0.45, -1), (0, 0))
    motion = RigidMotion()
    source_tangent = SourceGeometryTangent(motion=motion)
    grid_tangent = GridGeometryTangent(motion=RigidMotion((1, 0, 0)))
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0, 1], dtype=wp.int32, device=device),
        absorption=wp.array([0.7, 0.7], dtype=wp.float64, device=device),
        scattering=wp.array([0.01, 0.01], dtype=wp.float64, device=device),
        max_histories=count,
        device=device,
    )
    batch = HistoryBatch(19, 0, count).with_library_source("parallel-rectangle-v1")
    sampling = prepare_geometry_sampling(
        workspace.context,
        spec=spec,
        batch=batch,
        design=GeometrySamplingDesign("iid", 8, 2, 2),
        max_fluorescence_lines=0,
    )
    workspace._parameters.sampling = sampling.physical_record
    column = native.NativeGeometryColumn()
    column.source = _pack(
        sources.NativeSourceGeometry(), source_geometry_arguments(source, source_tangent)
    )
    _, grid = interface_arguments(material_patch_at(0, spec), spec, grid_tangent)
    column.grid = _pack(interfaces.InterfaceGrid(), grid)
    _, detector = free_compton_boundary_arguments(
        FreeComptonEnergyBoundary(RayTangent((0.4, 0.45, 0.5), (0, 0, 1)), 80, 70),
        spec.detector,
        motion,
    )
    column.detector = _pack(boundaries.NativeBoundaryDetector(), detector)
    medium = GeometryPhysicalMedium((0, 1), (0.7, 0.7), (0.01, 0.01), (1, 1), (1, 1))
    tangent = AcquisitionGeometryTangent(object=grid_tangent, source=source_tangent)
    admission = admit_whole_source_geometry(source, spec, medium, tangent)
    assert admission.redundant_detector_planes == frozenset({(0, 1)})
    column.redundant_planes = wp.array(
        [int(material_plane_at(i, spec) in admission.redundant_detector_planes) for i in range(7)],
        dtype=wp.int32,
        device=device,
    )
    plan, geometry = detector_proposal_plan_arguments(
        spec, redundant_planes=admission.redundant_detector_planes
    )
    plane_array = wp.array(
        [_pack(proposals.DetectorProposalPlane(), item) for item in plan],
        dtype=proposals.DetectorProposalPlane,
        device=device,
    )
    proposal = forward_proposals.ForwardProposalGeometry()
    proposal.regular = _pack(proposals.DetectorProposalGeometry(), geometry)
    proposal.detector_normal = spec.detector.normal
    values = wp.empty((count, 1), dtype=wp.float64, device=device)
    derivatives = wp.empty((count, 1), dtype=wp.float64, device=device)
    counts = wp.empty((count, 1), dtype=wp.vec2i, device=device)
    mean = wp.empty(1, dtype=wp.float64, device=device)
    gradient = wp.empty(1, dtype=wp.float64, device=device)
    moment = wp.empty(1, dtype=wp.float64, device=device)
    density = wp.array([1, 1], dtype=wp.float64, device=device)
    image = wp.array([seed_value], dtype=wp.float64, device=device)
    wp.launch(
        native.geometry_observations,
        dim=(count, 1),
        inputs=[
            wp.array([column], dtype=native.NativeGeometryColumn, device=device),
            plane_array,
            proposal,
            wp.zeros(7, dtype=sources.NativeAffineSourceBoundary, device=device),
            wp.zeros(4, dtype=sources.NativeAffineSourceBoundary, device=device),
            wp.zeros(18, dtype=opaque.NativeOpaqueBoundary, device=device),
            image,
            workspace.material_ids,
            workspace.voxel_density,
            workspace._energies,
            workspace.absorption,
            workspace.scattering,
            density,
            workspace._parameters,
            sampling.auxiliary_record,
            wp.uint64(batch.seed),
            wp.uint64(batch.first_history),
            values,
            derivatives,
            counts,
            workspace._status,
        ],
        device=device,
    )
    wp.launch(
        native.reduce_geometry_observations,
        dim=1,
        inputs=[values, derivatives, mean, gradient, moment, workspace._status],
        device=device,
    )
    provider = GeometryHostSampler(sampling.identity)
    expected = [
        composed_geometry_observation(
            source,
            (
                provider.uniform((h, 0, 0), 0x80000000, 0, 0),
                provider.uniform((h, 0, 0), 0x80000000, 0, 1),
            ),
            80,
            spec,
            medium,
            provider,
            original_history=h,
            cotangent=(seed_value,),
            tangent=tangent,
        )
        for h in range(count)
    ]
    return (
        expected,
        values.numpy(),
        derivatives.numpy(),
        counts.numpy(),
        mean.numpy(),
        gradient.numpy(),
        moment.numpy(),
        workspace._status.numpy()[0],
    )


def test_full_native_homogeneous_radiative_whole_plane_has_complete_host_parity(device):
    expected, values, derivatives, counts, mean, gradient, moment, status = _run(device)
    assert status == 0
    np.testing.assert_allclose(values[:, 0], [e.mean for e in expected], rtol=3e-11, atol=2e-12)
    np.testing.assert_allclose(
        derivatives[:, 0], [e.derivative for e in expected], rtol=3e-11, atol=2e-12
    )
    np.testing.assert_array_equal(
        counts[:, 0], [(e.physical_prefixes, e.continuation_histories) for e in expected]
    )
    assert sum(e.physical_prefixes for e in expected) > 0
    assert any(e.mean < -2 * math.exp(-0.71) for e in expected)
    assert mean[0] == pytest.approx(values[:, 0].mean(), rel=3e-14)
    assert gradient[0] == pytest.approx(derivatives[:, 0].mean(), rel=3e-14, abs=2e-15)
    assert moment[0] == pytest.approx((derivatives[:, 0] ** 2).mean(), rel=3e-14, abs=2e-15)
