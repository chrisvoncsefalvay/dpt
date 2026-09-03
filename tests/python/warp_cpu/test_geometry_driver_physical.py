"""Full original-history assembly controls using the actual aluminium laws."""

# Warp records are runtime objects; the test uses explicit provider imports.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
import importlib
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dpt.transport.compton_boundary_arguments import free_compton_boundary_arguments
from dpt.transport.compton_geometry_boundary import FreeComptonEnergyBoundary
from dpt.transport.forward import prepare_transport
from dpt.transport.geometry_composed import composed_geometry_observation, material_patch_at
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
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam
from dpt.transport.uncollided_geometry import AcquisitionGeometryTangent
from helpers.geometry_records import _pack
from helpers.physical_geometry_native import actual_tables as actual_tables

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.geometry_driver_kernels")
interfaces: Any = importlib.import_module("dpt.transport.geometry_interface_kernels")
sources: Any = importlib.import_module("dpt.transport.geometry_source_boundary_kernels")
boundaries: Any = importlib.import_module("dpt.transport.compton_geometry_boundary_kernels")
proposals: Any = importlib.import_module("dpt.transport.geometry_detector_proposal_kernels")
forward_proposals: Any = importlib.import_module("dpt.transport.geometry_forward_proposal_kernels")
opaque: Any = importlib.import_module("dpt.transport.geometry_opaque_kernels")


@pytest.mark.parametrize(
    "law,packed,mode",
    [
        ("isotropic-elastic", False, "spacing"),
        ("free-electron-compton", False, "spacing"),
        ("free-electron-compton", True, "detector"),
        ("bound-electron-compton", False, "spacing"),
    ],
)
def test_complete_mixed_geometry_original_histories_match_host(
    device, actual_tables, law, packed, mode
):
    energies, absorption, scattering, physical, shells = actual_tables
    if law == "bound-electron-compton":
        physical = replace(physical, bound_compton=(shells,))
    count = 8
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 1), 1),
        PlanarDetector((-2, -2), (1, 1), (4, 4), 3),
        10,
        "Actual xraylib4.3.0 aluminium geometry assembly control",
        energies,
        scattering_law=law,
        max_events=4,
        max_angle_trials=64,
        scoring="energy-kev",
        compton_sampler="packed-uniform-rejection" if packed else "legacy-rejection",
    )
    source = ParallelBeam((0.4, 0.45, -1), (0, 0), weight=1.7)
    tangent = AcquisitionGeometryTangent(
        object=GridGeometryTangent(spacing_dot=(0, 0, 1) if mode == "spacing" else (0, 0, 0)),
        detector=RigidMotion((1, 0, 0)) if mode == "detector" else RigidMotion(),
    )
    medium = GeometryPhysicalMedium((0,), absorption, scattering, (1,), (1,), physical)
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=device),
        absorption=wp.array(absorption, dtype=wp.float64, device=device),
        scattering=wp.array(scattering, dtype=wp.float64, device=device),
        max_histories=count,
        physical=physical,
        device=device,
    )
    batch = HistoryBatch(19, 0, count).with_library_source("parallel-rectangle-v1")
    layout = (
        "bound-compton-v3"
        if law == "bound-electron-compton"
        else "packed-compton-v2"
        if packed
        else "legacy-v1"
    )
    sampling = prepare_geometry_sampling(
        workspace.context,
        spec=spec,
        batch=batch,
        design=GeometrySamplingDesign("iid", 4, 64, 4, layout=layout),
        max_fluorescence_lines=2,
    )
    workspace._parameters.sampling = sampling.physical_record
    column = native.NativeGeometryColumn()
    column.source = _pack(
        sources.NativeSourceGeometry(), source_geometry_arguments(source, tangent.source)
    )
    _, grid = interface_arguments(material_patch_at(0, spec), spec, tangent.object)
    column.grid = _pack(interfaces.InterfaceGrid(), grid)
    _, detector = free_compton_boundary_arguments(
        FreeComptonEnergyBoundary(RayTangent((0.4, 0.45, 0.5), (0, 0, 1)), 10, 9),
        spec.detector,
        tangent.detector,
    )
    column.detector = _pack(boundaries.NativeBoundaryDetector(), detector)
    plan, geometry = detector_proposal_plan_arguments(spec)
    proposal = forward_proposals.ForwardProposalGeometry()
    proposal.regular = _pack(proposals.DetectorProposalGeometry(), geometry)
    proposal.detector_normal = spec.detector.normal
    planes = wp.array(
        [_pack(proposals.DetectorProposalPlane(), item) for item in plan],
        dtype=proposals.DetectorProposalPlane,
        device=device,
    )
    seed = tuple((-1 if index % 3 == 0 else 1) * (index + 1) for index in range(16))
    values = wp.empty((count, 1), dtype=wp.float64, device=device)
    derivatives = wp.empty((count, 1), dtype=wp.float64, device=device)
    counts = wp.empty((count, 1), dtype=wp.vec2i, device=device)
    wp.launch(
        native.geometry_observations,
        dim=(count, 1),
        inputs=[
            wp.array([column], dtype=native.NativeGeometryColumn, device=device),
            planes,
            proposal,
            wp.zeros(6, dtype=sources.NativeAffineSourceBoundary, device=device),
            wp.zeros(10, dtype=sources.NativeAffineSourceBoundary, device=device),
            wp.zeros(18, dtype=opaque.NativeOpaqueBoundary, device=device),
            wp.array(seed, dtype=wp.float64, device=device),
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
    expected = []
    for history in range(count):
        coordinate = (
            provider.uniform((history, 0, 0), 0x80000000, 0, 0),
            provider.uniform((history, 0, 0), 0x80000000, 0, 1),
        )
        expected.append(
            composed_geometry_observation(
                source,
                coordinate,
                10,
                spec,
                medium,
                provider,
                original_history=history,
                cotangent=seed,
                tangent=tangent,
            )
        )
    np.testing.assert_allclose(
        values.numpy()[:, 0], [e.mean for e in expected], rtol=3e-11, atol=2e-12
    )
    np.testing.assert_allclose(
        derivatives.numpy()[:, 0], [e.derivative for e in expected], rtol=3e-11, atol=2e-12
    )
    np.testing.assert_array_equal(
        counts.numpy()[:, 0], [(e.physical_prefixes, e.continuation_histories) for e in expected]
    )
    assert sum(e.physical_prefixes for e in expected) >= count
    assert any(e.mean != 0 for e in expected)
