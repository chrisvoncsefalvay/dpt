"""Terminal fixed-chart depth jets against independent scalar path integrals."""

# Warp annotations use runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
import importlib
import math
from typing import Any

import numpy as np  # noqa: F401 - preserve the qualified native provider import
import pytest

from dpt.transport.forward import prepare_transport
from dpt.transport.geometry_histories import GeometryPhysicalMedium
from dpt.transport.geometry_tangents import (
    RayTangent,
    cell_segments_tangent,
    normalised_direction_jet,
)
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.geometry_connection_kernels")


@wp.kernel
def _depth(
    origin: wp.vec3d,
    direction: wp.vec3d,
    origin_dot: wp.vec3d,
    direction_dot: wp.vec3d,
    distance: wp.float64,
    distance_dot: wp.float64,
    energy: wp.float64,
    energy_dot: wp.float64,
    materials: wp.array(dtype=wp.int32),
    voxels: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    parameters: native.Parameters,
    output: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    result = native.connection_optical_depth(
        origin,
        direction,
        origin_dot,
        direction_dot,
        distance,
        distance_dot,
        energy,
        energy_dot,
        materials,
        voxels,
        energies,
        absorption,
        scattering,
        density,
        parameters,
    )
    output[0] = result.depth
    output[1] = result.depth_dot
    output[2] = wp.float64(result.segments)
    status[0] = result.status


def _scene(device):
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (0.5, 0.7, 0.9), (2, 2, 3), 2),
        PlanarDetector((-1, -1), (1, 1), (5, 5), 3),
        80,
        "independent heterogeneous linear-energy connection",
        (10, 100),
        max_events=3,
    )
    medium = GeometryPhysicalMedium(
        tuple(i % 2 for i in range(12)),
        (0.1, 0.6, 0.3, 0.8),
        (0.2, 0.4, 0.1, 0.2),
        (1.2, 0.8),
        tuple(0.5 + 0.01 * i for i in range(12)),
    )
    workspace = prepare_transport(
        spec,
        material_ids=wp.array(medium.material_ids, dtype=wp.int32, device=device),
        absorption=wp.array(medium.absorption_mm_inverse, dtype=wp.float64, device=device),
        scattering=wp.array(medium.scattering_mm_inverse, dtype=wp.float64, device=device),
        voxel_density=wp.array(medium.voxel_density, dtype=wp.float64, device=device),
        max_histories=1,
        device=device,
    )
    return spec, medium, workspace


def _run(
    workspace, medium, origin, endpoint, origin_dot, endpoint_dot, *, energy=60.0, energy_dot=0.4
):
    delta = tuple(endpoint[i] - origin[i] for i in range(3))
    delta_dot = tuple(endpoint_dot[i] - origin_dot[i] for i in range(3))
    direction, direction_dot = normalised_direction_jet(delta, delta_dot)
    distance = math.hypot(*delta)
    distance_dot = math.fsum(direction[i] * delta_dot[i] for i in range(3))
    device = workspace.context.device
    output = wp.zeros(3, dtype=wp.float64, device=device)
    status = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _depth,
        dim=1,
        inputs=[
            wp.vec3d(*origin),
            wp.vec3d(*direction),
            wp.vec3d(*origin_dot),
            wp.vec3d(*direction_dot),
            distance,
            distance_dot,
            energy,
            energy_dot,
            workspace.material_ids,
            workspace.voxel_density,
            workspace._energies,
            workspace.absorption,
            workspace.scattering,
            wp.array(medium.density, dtype=wp.float64, device=device),
            workspace._parameters,
            output,
            status,
        ],
        device=device,
    )
    return output.numpy(), status.numpy()[0]


def _independent_scalar_depth(origin, endpoint, energy, spec, medium):
    direction, _ = normalised_direction_jet(
        tuple(endpoint[i] - origin[i] for i in range(3)), (0, 0, 0)
    )
    segments = cell_segments_tangent(
        RayTangent(origin, direction),
        spec.grid,
        spec.object_pose,
        maximum_distance_mm=math.dist(origin, endpoint),
    )
    rates = medium.extinction(energy, spec)
    return math.fsum(rates[segment.cell] * segment.length_mm for segment in segments)


@pytest.mark.parametrize(
    "endpoint,origin_dot,endpoint_dot",
    (
        ((1.3, 0.8, 3.0), (0, 0, 0), (0, 0, 1)),
        ((0.4, 0.9, 3.0), (0, 0, 0), (1, 0, 0)),
        ((0.7, 0.2, 3.0), (0.1, 0.2, 0.3), (-0.2, 0.1, 0.5)),
        ((-0.3, 1.8, 3.0), (-0.1, 0.3, 0.2), (0.2, -0.1, 0.1)),
    ),
)
def test_native_heterogeneous_energy_and_terminal_ray_jets_against_independent_five_point_integral(
    device,
    endpoint,
    origin_dot,
    endpoint_dot,
):
    spec, medium, workspace = _scene(device)
    origin = (0.2, 0.4, 0.3)
    output, status = _run(workspace, medium, origin, endpoint, origin_dot, endpoint_dot)
    assert status == 0
    h = 1e-4

    def reference(t):
        start = tuple(origin[i] + t * origin_dot[i] for i in range(3))
        end = tuple(endpoint[i] + t * endpoint_dot[i] for i in range(3))
        return _independent_scalar_depth(start, end, 60 + 0.4 * t, spec, medium)

    derivative = (reference(-2 * h) - 8 * reference(-h) + 8 * reference(h) - reference(2 * h)) / (
        12 * h
    )
    assert output[0] == pytest.approx(reference(0), rel=4e-14)
    assert output[1] == pytest.approx(derivative, rel=2e-8, abs=5e-12)
    assert output[2] >= 1


def test_native_connection_rejects_active_grazing_face_and_energy_interpolation_kink(device):
    spec, medium, workspace = _scene(device)  # noqa: RUF059 - retain the complete fixture tuple
    _, status = _run(workspace, medium, (0.5, 0.4, 0.3), (0.5, 0.7, 3), (0, 0, 0), (1, 0, 0))
    assert status == 4
    # This explicit knot is added to the prepared table without changing the
    # physical values; its active derivative is undefined at the interpolation chart.
    from dataclasses import replace

    parameters = workspace._parameters
    parameters.energy_bins = 3
    workspace._energies = wp.array([10, 60, 100], dtype=wp.float64, device=device)
    workspace.absorption = wp.array([0.1, 0.3, 0.6, 0.3, 0.6, 0.8], dtype=wp.float64, device=device)
    workspace.scattering = wp.array(
        [0.2, 0.3, 0.4, 0.1, 0.15, 0.2], dtype=wp.float64, device=device
    )
    medium = replace(
        medium,
        absorption_mm_inverse=tuple(workspace.absorption.numpy()),
        scattering_mm_inverse=tuple(workspace.scattering.numpy()),
    )
    _, status = _run(workspace, medium, (0.2, 0.4, 0.3), (0.4, 0.9, 3), (0, 0, 0), (0, 0, 1))
    assert status == 4


def test_native_connection_zero_depth_outside_box_and_full_partition_guard(device):
    _, medium, workspace = _scene(device)
    result, status = _run(
        workspace, medium, (-4, -3, -1), (-4, -3, 3), (0, 0, 0), (0, 0, 1), energy_dot=0
    )
    assert status == 0 and tuple(result) == (0, 0, 0)
    workspace._parameters.max_crossings = 1
    _, status = _run(workspace, medium, (0.2, 0.4, 0.3), (1.3, 0.8, 3), (0, 0, 0), (0, 0, 1))
    assert status == 4
