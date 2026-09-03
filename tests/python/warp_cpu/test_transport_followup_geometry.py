"""Acquisition geometry and source sampling checked through rigid-frame invariance."""

# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportMissingParameterType=false
import importlib
import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose
from dpt.transport.derivatives import TransportParameter, derivative_histories
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.transport.inverse import prepare_transport_inverse
from dpt.transport.model import MaterialGrid, OrientedDetector, TransportError, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import PointConeSource, sample_point_cone

wp: Any = importlib.import_module("warp")


def acquisition(pose=None):
    if pose is None:
        pose = RigidTransform()
    geometry = DetectorGeometry(
        pose.point((0.0, 0.0, -2.0)),
        pose.point((-0.75, -0.6, 3.0)),
        pose.direction((1.0, 0.0, 0.0)),
        pose.direction((0.0, 1.0, 0.0)),
        (0.3, 0.4),
        (4, 6),
    )
    return TransportSpec(
        MaterialGrid((-2.0, -2.0, 0.0), (4.0, 4.0, 1.0), (1, 1, 1), 1),
        OrientedDetector(geometry),
        80.0,
        "analytic general detector acquisition",
        estimator="continuous-absorption",
        object_pose=pose,
    )


def run(device, spec, *, absorption=0.0, scattering=0.0, count=4096):
    def arr(values, dtype=None):
        return wp.array(values, dtype=wp.float64 if dtype is None else dtype, device=device)

    workspace = prepare_transport(
        spec,
        material_ids=arr([0], wp.int32),
        absorption=arr([absorption]),
        scattering=arr([scattering]),
        max_histories=count,
        device=device,
    )
    source = PointConeSource(spec.detector.geometry)
    positions = wp.empty(count, dtype=wp.vec3d, device=device)
    directions = wp.empty_like(positions)
    weights = wp.empty(count, dtype=wp.float64, device=device)
    batch = HistoryBatch(79349, 2**49, count)
    sample_point_cone(
        source,
        batch=batch,
        workspace=workspace,
        out_position=positions,
        out_direction=directions,
        out_weight=weights,
    )
    outputs = {
        f"out_{name}": wp.empty(count, dtype=dtype, device=device)
        for name, dtype in (
            ("pixel", wp.int32),
            ("score", wp.float64),
            ("energy", wp.float64),
            ("events", wp.int32),
            ("status", wp.int32),
        )
    }
    inputs = [positions, directions, weights, arr([1.0])]
    trace_histories(*inputs, batch=batch, workspace=workspace, **outputs)
    return workspace, source, inputs, outputs, batch


@pytest.mark.parametrize("turn", [(0.2, -0.3, 0.4), (0.0, math.pi / 2, 0.0), (math.pi, 0.0, 0.0)])
def test_rotated_translated_source_grid_detector_preserve_vacuum_pixels(turn, warp_device):
    original = run(warp_device, acquisition())
    pose = compose_pose(RigidTransform(translation_mm=(4.0, -7.0, 11.0)), (0.0, 0.0, 0.0, *turn))
    rotated = run(warp_device, acquisition(pose))
    for name in original[3]:
        np.testing.assert_array_equal(original[3][name].numpy(), rotated[3][name].numpy())
    pixels = original[3]["out_pixel"].numpy()
    assert (pixels >= 0).all()
    # Source samples are uniform in detector area; per-pixel probability is known.
    expected = len(pixels) / 24
    error = math.sqrt(len(pixels) * (1 / 24) * (23 / 24))
    assert np.max(np.abs(np.bincount(pixels, minlength=24) - expected)) < 7 * error


def test_oblique_point_source_absorption_and_gradient_match_independent_chord(warp_device):
    pose = compose_pose(
        RigidTransform(translation_mm=(4.0, -7.0, 11.0)), (0.0, 0.0, 0.0, 0.2, -0.3, 0.4)
    )
    spec = acquisition(pose)
    workspace, _, inputs, outputs, batch = run(warp_device, spec, absorption=0.7)
    local_directions = np.array([pose.inverse().direction(tuple(d)) for d in inputs[1].numpy()])
    chord = 1 / local_directions[:, 2]
    expected = np.exp(-0.7 * chord)
    np.testing.assert_allclose(outputs["out_score"].numpy(), expected, rtol=3e-14)
    derivative = wp.empty(batch.count, dtype=wp.float64, device=warp_device)
    derivative_histories(
        *inputs,
        parameter=TransportParameter("log-material-density", 0),
        batch=batch,
        workspace=workspace,
        out_pixel=outputs["out_pixel"],
        out_derivative=derivative,
        out_status=outputs["out_status"],
    )
    np.testing.assert_allclose(derivative.numpy(), -0.7 * chord * expected, rtol=3e-14)


def test_point_source_chunking_reuses_counter_identity(warp_device):
    workspace, source, inputs, _, batch = run(warp_device, acquisition(), count=257)
    expected = [value.numpy().copy() for value in inputs[:3]]
    for start, count in ((0, 127), (127, 130)):
        sample_point_cone(
            source,
            batch=HistoryBatch(batch.seed, batch.first_history + start, count),
            workspace=workspace,
            out_position=inputs[0][start : start + count],
            out_direction=inputs[1][start : start + count],
            out_weight=inputs[2][start : start + count],
        )
    for actual, target in zip(inputs[:3], expected, strict=True):
        np.testing.assert_array_equal(actual.numpy(), target)


def test_point_source_can_drive_expected_signal_oracle(warp_device):
    workspace, source, _, _, _ = run(warp_device, acquisition(), count=257)
    observation = wp.zeros(24, dtype=wp.float64, device=warp_device)
    weights = wp.ones(24, dtype=wp.float64, device=warp_device)
    oracle = prepare_transport_inverse(
        workspace,
        source=source,
        parameters=(TransportParameter("log-source-amplitude"),),
        observation=observation,
        pixel_weights=weights,
        base_density=(1.0,),
        fused_derivatives=True,
    )
    gradient = oracle.gradient_replicate(
        (0.0,), HistoryBatch(811, 0, 257), HistoryBatch(811, 257, 257)
    )
    assert len(gradient) == 1 and math.isfinite(gradient[0]) and gradient[0] > 0
    assert not oracle.deterministic_sampling


def test_general_detector_requires_entire_posed_grid_on_incident_side():
    spec = acquisition()
    with pytest.raises(TransportError, match="entire material grid"):
        replace(spec, object_pose=RigidTransform(translation_mm=(0.0, 0.0, 3.0)))
