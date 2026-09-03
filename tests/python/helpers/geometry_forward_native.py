"""Shared exact scientific helpers for the native controls."""

# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
import importlib
from typing import Any

from dpt.transport.geometry_forward_proposal_arguments import forward_proposal_plan_arguments
from dpt.transport.geometry_tangents import RayTangent
from dpt.transport.model import (
    MaterialGrid,
    PlanarDetector,
    TransportSpec,
)

wp: Any = importlib.import_module("warp")

native: Any = importlib.import_module("dpt.transport.geometry_forward_proposal_kernels")

regular_native: Any = importlib.import_module("dpt.transport.geometry_detector_proposal_kernels")


@wp.kernel
def _samples(
    planes: wp.array(dtype=regular_native.DetectorProposalPlane),
    offset: int,
    count: int,
    geometry: native.ForwardProposalGeometry,
    origins: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    component_draws: wp.array(dtype=wp.float64),
    regular_draws: wp.array(dtype=wp.float64),
    coordinates: wp.array(dtype=wp.vec2d),
    output: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    sample = native.sample_forward_area_mixture(
        planes,
        offset,
        count,
        geometry,
        origins[index],
        directions[index],
        component_draws[index],
        regular_draws[index],
        coordinates[index],
    )
    status[index] = sample.status
    if sample.status == 0:
        output[index, 0] = sample.coordinate[0]
        output[index, 1] = sample.coordinate[1]
        output[index, 2] = sample.log_reference_density
        output[index, 3] = wp.float64(sample.component)


def _scene(scale=1.0):
    spec = TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (scale, scale, scale), (1, 1, 1), 1),
        PlanarDetector((0.0, 0.0), (scale, scale), (1, 1), 2.0 * scale),
        80.0,
        "forward radial proposal",
    )
    incoming = RayTangent((0.4 * scale, 0.6 * scale, 0.5 * scale), (0.0, 0.0, 1.0))
    return incoming, spec


def _pack(spec, device):
    records, arguments = forward_proposal_plan_arguments(spec)
    planes = []
    for record in records:
        plane = regular_native.DetectorProposalPlane()
        plane.coefficients = wp.vec3d(*record.coefficients)
        plane.normal_separation_mm = record.normal_separation_mm
        plane.axis, plane.position_mm = record.axis, record.position_mm
        planes.append(plane)
    regular = regular_native.DetectorProposalGeometry()
    values = arguments.regular
    regular.inverse_rotation = wp.mat33d(*values.inverse_rotation)
    regular.inverse_translation = wp.vec3d(*values.inverse_translation)
    regular.detector_lower = wp.vec3d(*values.detector_lower)
    regular.detector_u = wp.vec3d(*values.detector_u)
    regular.detector_v = wp.vec3d(*values.detector_v)
    regular.detector_shape = wp.vec2i(*values.detector_shape)
    regular.detector_spacing = wp.vec2d(*values.detector_spacing)
    geometry = native.ForwardProposalGeometry()
    geometry.regular = regular
    geometry.detector_normal = wp.vec3d(*arguments.detector_normal)
    return wp.array(planes, dtype=regular_native.DetectorProposalPlane, device=device), geometry


def _run(
    incoming, spec, device, component_draws, regular_draws, coordinates, *, invalid_slice=False
):
    planes, geometry = _pack(spec, device)
    length = len(coordinates)
    output = wp.zeros((length, 4), dtype=wp.float64, device=device)
    status = wp.zeros(length, dtype=wp.int32, device=device)
    wp.launch(
        _samples,
        dim=length,
        inputs=[
            planes,
            1 if invalid_slice else 0,
            len(planes),
            geometry,
            wp.array([incoming.origin_mm] * length, dtype=wp.vec3d, device=device),
            wp.array([incoming.direction] * length, dtype=wp.vec3d, device=device),
            wp.array(component_draws, dtype=wp.float64, device=device),
            wp.array(regular_draws, dtype=wp.float64, device=device),
            wp.array(coordinates, dtype=wp.vec2d, device=device),
            output,
            status,
        ],
        device=device,
    )
    return output.numpy(), status.numpy()
