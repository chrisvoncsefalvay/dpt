"""Original-history native assembly against independent absorbing-slab laws."""

# Warp is an explicit runtime DSL provider for the annotations below.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
import importlib
import math
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
    GeometryHostSampler,
    GeometrySamplingDesign,
    prepare_geometry_sampling,
)
from dpt.transport.geometry_source_boundary_arguments import source_geometry_arguments
from dpt.transport.geometry_tangents import GridGeometryTangent, RayTangent, RigidMotion
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam
from dpt.transport.uncollided_geometry import SourceGeometryTangent
from helpers.geometry_records import (
    _pack,
)

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.geometry_driver_kernels")
interfaces: Any = importlib.import_module("dpt.transport.geometry_interface_kernels")
sources: Any = importlib.import_module("dpt.transport.geometry_source_boundary_kernels")
boundaries: Any = importlib.import_module("dpt.transport.compton_geometry_boundary_kernels")
proposals: Any = importlib.import_module("dpt.transport.geometry_detector_proposal_kernels")
forward_proposals: Any = importlib.import_module("dpt.transport.geometry_forward_proposal_kernels")
opaque: Any = importlib.import_module("dpt.transport.geometry_opaque_kernels")


def _run(device, *, mode="fixed", seed_value=-2.0):
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
    source = ParallelBeam((0.4, 0.45, -1), (0, 0))
    motion = RigidMotion((0, 0, 1)) if mode == "common" else RigidMotion()
    source_tangent = SourceGeometryTangent(motion=motion)
    grid_tangent = GridGeometryTangent(
        motion=motion, spacing_dot=(0, 0, 1) if mode == "spacing" else (0, 0, 0)
    )
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


def test_full_native_absorbing_slab_mean_and_zero_tangent(device):
    _, values, derivatives, _, mean, gradient, moment, status = _run(device)
    assert status == 0
    np.testing.assert_allclose(values, -2 * math.exp(-0.7), rtol=3e-14, atol=0)
    np.testing.assert_array_equal(derivatives, 0)
    assert mean[0] == pytest.approx(-2 * math.exp(-0.7), rel=4e-14)
    assert gradient[0] == moment[0] == 0


def test_full_native_spacing_generator_matches_exact_selected_plane_expectation(device):
    sampling, values, derivatives, counts, mean, gradient, moment, status = _run(
        device, mode="spacing"
    )
    assert status == 0
    provider = GeometryHostSampler(sampling.identity)
    expected = np.asarray(
        [
            12 * 0.7 * math.exp(-0.7)
            if int(provider.uniform((h, 0, 0), 0, 0x30000003, 0) * 6) == 5
            else 0
            for h in range(32)
        ]
    )
    np.testing.assert_allclose(derivatives[:, 0], expected, rtol=4e-14, atol=2e-15)
    assert gradient[0] == pytest.approx(expected.mean(), rel=4e-14)
    assert moment[0] == pytest.approx((expected * expected).mean(), rel=4e-14)
    np.testing.assert_allclose(values, -2 * math.exp(-0.7), rtol=3e-14, atol=0)
    assert mean[0] == pytest.approx(-2 * math.exp(-0.7), rel=4e-14)
    assert np.count_nonzero(counts[:, 0, 1]) == np.count_nonzero(expected)


def test_full_native_common_translation_cancels_source_object_detector_geometry(device):
    _, _, derivatives, _, _, gradient, _, status = _run(device, mode="common")
    assert status == 0
    np.testing.assert_array_equal(derivatives, 0)
    assert gradient[0] == 0


def test_failed_original_history_invalidates_all_reductions(device):
    _, values, derivatives, _, mean, gradient, moment, status = _run(device, seed_value=math.inf)
    assert status == 4
    assert np.all(np.isnan(values)) and np.all(np.isnan(derivatives))
    assert np.all(np.isnan(mean)) and np.all(np.isnan(gradient)) and np.all(np.isnan(moment))
