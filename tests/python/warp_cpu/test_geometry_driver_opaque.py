"""Initial-cutoff assembly against independent half-open box-shadow laws."""

# Warp records are explicitly imported runtime providers.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
import importlib
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dpt.transport.compton_boundary_arguments import free_compton_boundary_arguments
from dpt.transport.compton_geometry_boundary import FreeComptonEnergyBoundary
from dpt.transport.forward import prepare_transport
from dpt.transport.geometry_composed import material_patch_at
from dpt.transport.geometry_detector_proposal_arguments import detector_proposal_plan_arguments
from dpt.transport.geometry_interface_arguments import interface_arguments
from dpt.transport.geometry_opaque_source import opaque_source_boundaries
from dpt.transport.geometry_sampling import (
    GeometryHostSampler,
    GeometrySamplingDesign,
    prepare_geometry_sampling,
)
from dpt.transport.geometry_source_boundary_arguments import (
    affine_source_boundary_arguments,
    source_geometry_arguments,
)
from dpt.transport.geometry_tangents import GridGeometryTangent, RayTangent, RigidMotion
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam
from dpt.transport.source_spectrum import SourceSpectrum
from dpt.transport.uncollided_geometry import AcquisitionGeometryTangent, SourceGeometryTangent
from helpers.geometry_records import _pack
from helpers.physical_geometry_native import actual_tables as actual_tables

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.geometry_driver_kernels")
interfaces: Any = importlib.import_module("dpt.transport.geometry_interface_kernels")
sources: Any = importlib.import_module("dpt.transport.geometry_source_boundary_kernels")
boundaries: Any = importlib.import_module("dpt.transport.compton_geometry_boundary_kernels")
proposals: Any = importlib.import_module("dpt.transport.geometry_detector_proposal_kernels")
forward: Any = importlib.import_module("dpt.transport.geometry_forward_proposal_kernels")
opaque: Any = importlib.import_module("dpt.transport.geometry_opaque_kernels")


@pytest.mark.parametrize("energy,common", [(40, False), (50, False), (80, False), (40, True)])
def test_initial_cutoff_whole_history_matches_half_open_shadow(
    device, actual_tables, energy, common
):
    nodes, _, _, physical, _ = actual_tables
    physical = replace(
        physical,
        rayleigh_mm_inverse=(0,) * len(nodes),
        fluorescence=(),
        cutoff_kev=50,
        spectrum=SourceSpectrum((energy,), (1,), "explicit monoenergetic original mark"),
    )
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 1), 1),
        PlanarDetector((-2, -2), (4, 4), (1, 1), 3),
        80,
        "Independent cropped half-open opaque box shadow",
        nodes,
        max_events=4,
        max_angle_trials=2,
        scoring="energy-kev",
    )
    source = ParallelBeam((-0.5, 0.1, -1), (1.1, 0.8))
    motion = RigidMotion((1, 0, 0))
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(motion=motion),
        source=SourceGeometryTangent(motion=motion) if common else SourceGeometryTangent(),
        detector=motion if common else RigidMotion(),
    )
    count = 64
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=device),
        absorption=wp.zeros(len(nodes), dtype=wp.float64, device=device),
        scattering=wp.zeros(len(nodes), dtype=wp.float64, device=device),
        max_histories=count,
        physical=physical,
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
    column.empty_medium = 1
    column.source = _pack(
        sources.NativeSourceGeometry(), source_geometry_arguments(source, tangent.source)
    )
    _, grid = interface_arguments(material_patch_at(0, spec), spec, tangent.object)
    column.grid = _pack(interfaces.InterfaceGrid(), grid)
    _, detector = free_compton_boundary_arguments(
        FreeComptonEnergyBoundary(RayTangent((0.4, 0.45, 0.5), (0, 0, 1)), 80, 70),
        spec.detector,
        tangent.detector,
    )
    column.detector = _pack(boundaries.NativeBoundaryDetector(), detector)
    plan, geometry = detector_proposal_plan_arguments(spec)
    proposal = forward.ForwardProposalGeometry()
    proposal.regular = _pack(proposals.DetectorProposalGeometry(), geometry)
    proposal.detector_normal = spec.detector.normal
    rows = [opaque.NativeOpaqueBoundary() for _ in range(18)]
    for row in opaque_source_boundaries(source, spec, tangent):
        packed = rows[row.index]
        packed.boundary = _pack(
            sources.NativeAffineSourceBoundary(),
            affine_source_boundary_arguments(
                row.boundary,
                axis=0,
                face=0,
                family="material",
            ),
        )
        packed.face_count = len(row.attached_faces)
        packed.axis0, packed.high0 = row.attached_faces[0]
        if len(row.attached_faces) == 2:
            packed.axis1, packed.high1 = row.attached_faces[1]
    values = wp.empty((count, 1), dtype=wp.float64, device=device)
    derivatives = wp.empty((count, 1), dtype=wp.float64, device=device)
    counts = wp.empty((count, 1), dtype=wp.vec2i, device=device)
    wp.launch(
        native.geometry_observations,
        dim=(count, 1),
        inputs=[
            wp.array([column], dtype=native.NativeGeometryColumn, device=device),
            wp.array(
                [_pack(proposals.DetectorProposalPlane(), row) for row in plan],
                dtype=proposals.DetectorProposalPlane,
                device=device,
            ),
            proposal,
            wp.zeros(6, dtype=sources.NativeAffineSourceBoundary, device=device),
            wp.zeros(4, dtype=sources.NativeAffineSourceBoundary, device=device),
            wp.array(rows, dtype=opaque.NativeOpaqueBoundary, device=device),
            wp.array([1], dtype=wp.float64, device=device),
            workspace.material_ids,
            workspace.voxel_density,
            workspace._energies,
            workspace.absorption,
            workspace.scattering,
            wp.array([1], dtype=wp.float64, device=device),
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
    assert workspace._status.numpy()[0] == 0
    provider = GeometryHostSampler(sampling.identity)
    xs = np.asarray(
        [-0.5 + 1.1 * provider.uniform((h, 0, 0), 0x80000000, 0, 0) for h in range(count)]
    )
    expected_mean = np.where(xs >= 0, 0, energy) if energy <= 50 else np.full(count, energy)
    # In this cropped scene only the lower-x source-support boundary survives.
    # Moving +x decreases its opaque shadow, so its escape derivative is +E/1.1.
    # The complete selector retains all18 original proposal slots.
    expected_derivative = np.asarray(
        [
            18 * energy / 1.1
            if energy <= 50
            and not common
            and min(17, int(18 * provider.uniform((h, 0, 0), 0, 0x30000003, 1))) == 12
            else 0
            for h in range(count)
        ]
    )
    np.testing.assert_allclose(values.numpy()[:, 0], expected_mean, rtol=0, atol=2e-13)
    np.testing.assert_allclose(
        derivatives.numpy()[:, 0], expected_derivative, rtol=2e-14, atol=2e-13
    )
    np.testing.assert_array_equal(counts.numpy(), 0)
