"""Original-history native assembly against independent absorbing-slab laws."""

# Warp is an explicit runtime DSL provider for the annotations below.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
import importlib
from dataclasses import fields
from typing import Any

import numpy as np
import pytest

from dpt.transport.compton_boundary_arguments import free_compton_boundary_arguments
from dpt.transport.compton_geometry_boundary import FreeComptonEnergyBoundary
from dpt.transport.forward import prepare_transport
from dpt.transport.geometry_composed import material_patch_at
from dpt.transport.geometry_detector_proposal_arguments import detector_proposal_plan_arguments
from dpt.transport.geometry_interface_arguments import interface_arguments
from dpt.transport.geometry_sampling import (
    GeometrySamplingDesign,
    prepare_geometry_sampling,
)
from dpt.transport.geometry_source_boundary_arguments import source_geometry_arguments
from dpt.transport.geometry_tangents import GridGeometryTangent, RayTangent, RigidMotion
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam
from dpt.transport.uncollided_geometry import SourceGeometryTangent

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


def _run(device, *, direction, seed_value=-2.0):
    count = 32
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 1), 1),
        PlanarDetector((-5, -5), (10, 10), (1, 1), 3),
        80,
        "independent original-history absorbing slab",
        (80,),
        max_events=4,
        max_angle_trials=2,
    )
    source = ParallelBeam((0.4, 0.45, 2), (0, 0), direction=direction)
    motion = RigidMotion()
    source_tangent = SourceGeometryTangent(motion=motion)
    grid_tangent = GridGeometryTangent(motion=RigidMotion((1, 0, 0)))
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=device),
        absorption=wp.array([0.7], dtype=wp.float64, device=device),
        scattering=wp.array([0], dtype=wp.float64, device=device),
        max_histories=count,
        device=device,
    )
    batch = HistoryBatch(19, 0, count).with_library_source("parallel-rectangle-v1")
    sampling = prepare_geometry_sampling(
        workspace.context,
        spec=spec,
        batch=batch,
        design=GeometrySamplingDesign("iid", 4, 2, 2),
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
    plan, geometry = detector_proposal_plan_arguments(spec)
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
    density = wp.array([1], dtype=wp.float64, device=device)
    image = wp.array([seed_value], dtype=wp.float64, device=device)
    wp.launch(
        native.geometry_observations,
        dim=(count, 1),
        inputs=[
            wp.array([column], dtype=native.NativeGeometryColumn, device=device),
            plane_array,
            proposal,
            wp.zeros(6, dtype=sources.NativeAffineSourceBoundary, device=device),
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
    return (
        sampling,
        values.numpy(),
        derivatives.numpy(),
        counts.numpy(),
        mean.numpy(),
        gradient.numpy(),
        moment.numpy(),
        workspace._status.numpy()[0],
    )


@pytest.mark.parametrize("direction", [(0, 0, -1), (1, 0, 0)])
def test_full_native_primary_miss_is_successful_zero_with_actual_physical_flights(
    device, direction
):
    _, values, derivatives, counts, mean, gradient, moment, status = _run(
        device, direction=direction
    )
    assert status == 0
    np.testing.assert_array_equal(values, 0)
    np.testing.assert_array_equal(derivatives, 0)
    assert mean[0] == gradient[0] == moment[0] == 0
    if direction == (0, 0, -1):
        assert np.count_nonzero(counts[:, 0, 0]) > 0
    else:
        np.testing.assert_array_equal(counts, 0)
