"""Public zero primary responses with complete radiative physical futures."""

# Runtime Warp arrays are opaque to Python's static provider.
# pyright: reportUnknownMemberType=false, reportPrivateUsage=false
import importlib

import numpy as np
import pytest

from dpt.transport.geometry_composed import composed_geometry_observation
from dpt.transport.geometry_derivatives import prepare_transport_geometry, transport_geometry_vjp
from dpt.transport.geometry_histories import GeometryPhysicalMedium
from dpt.transport.geometry_parameters import GeometryParameter
from dpt.transport.geometry_sampling import GeometryHostSampler, GeometrySamplingDesign
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam

wp = importlib.import_module("warp")


def _prepare(direction, scattering, device):
    spec = TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-5.0, -5.0), (10.0, 10.0), (1, 1), 3.0),
        80.0,
        "Missed primary and complete isotropic radiative futures",
        max_events=16,
        max_angle_trials=2,
    )
    source = ParallelBeam((0.4, 0.45, 2.0), (0.0, 0.0), direction=direction)
    medium = GeometryPhysicalMedium(
        (0,), (0.1 if scattering else 0.0,), (scattering,), (1.0,), (1.0,)
    )
    parameters = (
        GeometryParameter("object", "grid-spacing", 2),
        GeometryParameter("source", "beam-direction", 1),
        GeometryParameter("detector", "translation", 2),
    )
    workspace = prepare_transport_geometry(
        spec,
        source=source,
        medium=medium,
        parameters=parameters,
        batch=HistoryBatch(89, 0, 128),
        design=GeometrySamplingDesign("iid", 16, 2, 2),
        device=device,
    )
    with workspace.context.scope():
        outputs = {
            "out_values": wp.empty((128, 3), dtype=wp.float64, device=device),
            "out_derivatives": wp.empty((128, 3), dtype=wp.float64, device=device),
            "out_counts": wp.empty((128, 3), dtype=wp.vec2i, device=device),
            "out_mean": wp.empty(3, dtype=wp.float64, device=device),
            "out_gradient": wp.empty(3, dtype=wp.float64, device=device),
            "out_second_moment": wp.empty(3, dtype=wp.float64, device=device),
        }
        cotangent = wp.array([2.0], dtype=wp.float64, device=device)
    return spec, source, medium, parameters, workspace, outputs, cotangent


@pytest.mark.parametrize("direction", ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)))
def test_public_backward_and_tangent_vacuum_beams_have_zero_primary(direction, warp_device):
    _, _, _, _, workspace, outputs, cotangent = _prepare(direction, 0.0, warp_device)
    transport_geometry_vjp(cotangent, workspace=workspace, **outputs)
    for name in ("out_values", "out_derivatives", "out_mean", "out_gradient", "out_second_moment"):
        np.testing.assert_array_equal(outputs[name].numpy(), 0.0)
    np.testing.assert_array_equal(workspace._reference._status.numpy(), [0])
    assert np.all(workspace._source_boundaries.numpy()["present"] == 0)
    assert np.all(workspace._detector_boundaries.numpy()["present"] == 0)


def test_public_backward_beam_keeps_positive_scattered_response_and_full_pullback(warp_device):
    spec, source, medium, parameters, workspace, outputs, cotangent = _prepare(
        (0.0, 0.0, -1.0), 0.8, warp_device
    )
    transport_geometry_vjp(cotangent, workspace=workspace, **outputs)
    provider = GeometryHostSampler(workspace.identity.sampling)
    reference = []
    for history in range(workspace.batch.count):
        coordinate = (
            provider.uniform((history, 0, 0), 0x80000000, 0, 0),
            provider.uniform((history, 0, 0), 0x80000000, 0, 1),
        )
        reference.append(
            [
                composed_geometry_observation(
                    source,
                    coordinate,
                    80.0,
                    spec,
                    medium,
                    provider,
                    original_history=history,
                    cotangent=(2.0,),
                    tangent=parameter.tangent(source),
                )
                for parameter in parameters
            ]
        )
    values = np.asarray([[row.mean for row in columns] for columns in reference])
    derivatives = np.asarray([[row.derivative for row in columns] for columns in reference])
    counts = np.asarray(
        [
            [(row.physical_prefixes, row.continuation_histories) for row in columns]
            for columns in reference
        ]
    )
    np.testing.assert_allclose(outputs["out_values"].numpy(), values, rtol=4e-11, atol=3e-12)
    np.testing.assert_allclose(
        outputs["out_derivatives"].numpy(), derivatives, rtol=4e-11, atol=3e-12
    )
    np.testing.assert_array_equal(outputs["out_counts"].numpy(), counts)
    np.testing.assert_allclose(
        outputs["out_mean"].numpy(), values.mean(axis=0), rtol=4e-11, atol=3e-12
    )
    np.testing.assert_allclose(
        outputs["out_gradient"].numpy(), derivatives.mean(axis=0), rtol=4e-11, atol=3e-12
    )
    np.testing.assert_allclose(
        outputs["out_second_moment"].numpy(),
        np.mean(derivatives**2, axis=0),
        rtol=4e-11,
        atol=3e-12,
    )
    assert np.all(values.mean(axis=0) > 0.0)
    assert any(row.physical_prefixes > 0 for columns in reference for row in columns)
