"""Independent conditional-CDF and full-mixture native proposal checks."""

# Warp kernel annotations use runtime DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
import importlib
import math
from dataclasses import replace
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pytest

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.transport.geometry_detector_proposal_arguments import (
    detector_proposal_arguments,
    detector_proposal_plan_arguments,
)
from dpt.transport.geometry_detector_proposals import (
    DetectorAreaProposal,
    DetectorPlaneProposal,
    prepare_detector_area_proposal,
    sample_detector_area_proposal,
)
from dpt.transport.model import (
    MaterialGrid,
    OrientedDetector,
    PlanarDetector,
    TransportError,
    TransportSpec,
)

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.geometry_detector_proposal_kernels")


@wp.kernel
def _samples(
    planes: wp.array(dtype=native.DetectorProposalPlane),
    offset: int,
    count: int,
    draws: wp.array(dtype=wp.float64),
    coordinates: wp.array(dtype=wp.vec2d),
    output: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    sample = native.sample_detector_area_proposal(
        planes, offset, count, draws[index], coordinates[index]
    )
    status[index] = sample.status
    if sample.status == 0:
        output[index, 0] = sample.coordinate[0]
        output[index, 1] = sample.coordinate[1]
        output[index, 2] = sample.log_reference_density
        output[index, 3] = wp.float64(sample.component)


@wp.kernel
def _vertex_samples(
    planes: wp.array(dtype=native.DetectorProposalPlane),
    count: int,
    geometry: native.DetectorProposalGeometry,
    previous: wp.array(dtype=wp.vec3d),
    draws: wp.array(dtype=wp.float64),
    coordinates: wp.array(dtype=wp.vec2d),
    output: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    sample = native.sample_detector_area_proposal_from_vertex(
        planes, 0, count, geometry, previous[index], draws[index], coordinates[index]
    )
    status[index] = sample.status
    if sample.status == 0:
        output[index, 0] = sample.coordinate[0]
        output[index, 1] = sample.coordinate[1]
        output[index, 2] = sample.log_reference_density
        output[index, 3] = wp.float64(sample.component)


def _pack(proposal: DetectorAreaProposal, device: str):
    records = detector_proposal_arguments(proposal)
    result = []
    for record in records:
        plane = native.DetectorProposalPlane()
        plane.coefficients = wp.vec3d(*record.coefficients)
        plane.normal_separation_mm = record.normal_separation_mm
        plane.axis, plane.position_mm = record.axis, record.position_mm
        result.append(plane)
    return wp.array(result, dtype=native.DetectorProposalPlane, device=device)


def _run(proposal: DetectorAreaProposal, device: str, draws, coordinates, *, offset=0, count=None):
    planes = _pack(proposal, device)
    output = wp.zeros((len(draws), 4), dtype=wp.float64, device=device)
    status = wp.zeros(len(draws), dtype=wp.int32, device=device)
    wp.launch(
        _samples,
        dim=len(draws),
        inputs=[
            planes,
            offset,
            len(proposal.planes) if count is None else count,
            wp.array(draws, dtype=wp.float64, device=device),
            wp.array(coordinates, dtype=wp.vec2d, device=device),
            output,
            status,
        ],
        device=device,
    )
    return output.numpy(), status.numpy()


def _decimal_density(planes, coordinate):
    with localcontext() as context:
        context.prec = 90
        values = [Decimal.from_float(float(value)) for value in coordinate]
        total = Decimal("0.5")
        for plane in planes:
            a, b, c = (Decimal.from_float(value) for value in plane.coefficients)
            separation = Decimal.from_float(plane.normal_separation_mm)
            axis = int(abs(b) > abs(a))
            coefficient = (a, b)[axis]
            other = (b, a)[axis] * values[1 - axis] + c

            def primitive(value, separation=separation):
                return (1 if value >= 0 else -1) * (1 + abs(value) / separation).ln()

            width = abs(primitive(other + coefficient) - primitive(other))
            level = coefficient * values[axis] + other
            total += (
                Decimal("0.5")
                / len(planes)
                * abs(coefficient)
                / (width * (separation + abs(level)))
            )
        return float(total.ln())


def test_no_planes_is_uniform(warp_device):
    proposal = DetectorAreaProposal((), "uniform")
    coordinates = [(0.17, 0.29), (0.53, 0.89)]
    output, status = _run(proposal, warp_device, [0.1, 0.9], coordinates)
    assert np.all(status == 0)
    np.testing.assert_array_equal(output[:, :2], coordinates)
    np.testing.assert_array_equal(output[:, 2], 0.0)
    np.testing.assert_array_equal(output[:, 3], -1.0)


@pytest.mark.parametrize("separation", [0.3, 1e-4, 1e-10])
@pytest.mark.parametrize("coefficients", [(1.0, 0.0, -0.5), (0.3, -1.1, 0.4)])
def test_log_cdf_matches_host_and_independent_density(warp_device, separation, coefficients):
    proposal = DetectorAreaProposal((DetectorPlaneProposal(0, 0, coefficients, separation),), "one")
    coordinates = [(0.17, 0.19), (0.79, 0.83)]
    draws = [0.75, 0.1]
    output, status = _run(proposal, warp_device, draws, coordinates)
    assert np.all(status == 0)
    for index in range(len(draws)):
        reference = sample_detector_area_proposal(
            proposal, component_draw=draws[index], coordinate=coordinates[index]
        )
        np.testing.assert_allclose(output[index, :2], reference.coordinate, atol=2e-15, rtol=2e-13)
        assert output[index, 3] == reference.component
        assert output[index, 2] == pytest.approx(reference.log_reference_density, abs=3e-13)
        # Twice the declared binary64 chart-resolution bound allows the rounded affine level.
        assert output[index, 2] == pytest.approx(
            _decimal_density(proposal.planes, output[index, :2]), abs=3e-8
        )


def test_full_mixture_density_and_association_regression(warp_device):
    plane = DetectorPlaneProposal(0, 0, (1.0, 2.0, -0.5), 1e-20)
    proposal = DetectorAreaProposal((plane,), "small")
    address = math.log1p(200.0) / math.log1p(2.0 / plane.normal_separation_mm)
    output, status = _run(proposal, warp_device, [0.75], [(0.5, address)])
    assert status[0] == 0
    assert output[0, 1] > 0.0
    exact = _decimal_density(proposal.planes, output[0, :2])
    assert output[0, 2] == pytest.approx(exact, abs=3e-13)
    # The differently grouped contraction loses the small represented level.
    naive = (
        plane.coefficients[0] * output[0, 0]
        + plane.coefficients[1] * output[0, 1]
        + plane.coefficients[2]
    )
    assert naive == 0.0
    correct = 2.0 * output[0, 1]
    assert correct > 100.0 * plane.normal_separation_mm


def test_uniform_component_uses_all_plane_densities(warp_device):
    planes = (
        DetectorPlaneProposal(0, 0, (1.0, 0.0, -0.4), 0.1),
        DetectorPlaneProposal(1, 1, (0.2, -1.1, 0.8), 0.03),
        DetectorPlaneProposal(2, 0, (-0.5, 0.7, -0.1), 0.2),
    )
    proposal = DetectorAreaProposal(planes, "three")
    output, status = _run(proposal, warp_device, [0.2], [(0.42, 0.73)])
    assert status[0] == 0
    assert output[0, 3] == -1
    assert output[0, 2] == pytest.approx(_decimal_density(planes, output[0, :2]), abs=3e-13)
    assert abs(output[0, 2] - math.log(0.5)) > 0.5


@pytest.mark.parametrize("scale", [1e-150, 1e150])
def test_physical_unit_scale_preserves_chart_density(warp_device, scale):
    proposal = DetectorAreaProposal(
        (DetectorPlaneProposal(0, 0, (scale, 0.0, -0.5 * scale), 0.03 * scale),), "scale"
    )
    output, status = _run(proposal, warp_device, [0.7], [(0.19, 0.53)])
    assert status[0] == 0
    assert output[0, 2] == pytest.approx(
        _decimal_density(proposal.planes, output[0, :2]), abs=3e-12
    )


@pytest.mark.parametrize(
    "draw,coordinate",
    [
        (0.0, (0.2, 0.3)),
        (1.0, (0.2, 0.3)),
        (float("nan"), (0.2, 0.3)),
        (0.3, (0.0, 0.3)),
        (0.3, (0.2, float("inf"))),
    ],
)
def test_invalid_draws_are_fatal_without_clipping(warp_device, draw, coordinate):
    output, status = _run(DetectorAreaProposal((), "uniform"), warp_device, [draw], [coordinate])
    assert status[0] == 4
    np.testing.assert_array_equal(output[0], 0.0)


def test_underresolved_inverse_and_invalid_slice_are_fatal(warp_device):
    proposal = DetectorAreaProposal((DetectorPlaneProposal(0, 0, (1.0, 0.0, -0.5), 1e-30),), "tiny")
    output, status = _run(proposal, warp_device, [0.75], [(0.5, 0.3)])
    assert status[0] == 4
    np.testing.assert_array_equal(output[0], 0.0)
    _, status = _run(proposal, warp_device, [0.2], [(0.2, 0.3)], offset=1, count=1)
    assert status[0] == 4


def _spec():
    return TransportSpec(
        MaterialGrid(
            origin_mm=(-1.0, -1.0, 0.0), spacing_mm=(1.0, 1.0, 1.0), shape=(2, 2, 2), materials=1
        ),
        PlanarDetector(lower_xy_mm=(-1.4, -1.6), spacing_xy_mm=(0.7, 0.8), shape=(4, 4), z_mm=4.0),
        energy_kev=80.0,
        coefficient_provenance="synthetic geometry fixture",
    )


def _run_vertex(spec, device, previous, draws, coordinates):
    records, geometry_arguments = detector_proposal_plan_arguments(spec)
    planes = []
    for record in records:
        plane = native.DetectorProposalPlane()
        plane.coefficients = wp.vec3d(*record.coefficients)
        plane.normal_separation_mm = record.normal_separation_mm
        plane.axis, plane.position_mm = record.axis, record.position_mm
        planes.append(plane)
    geometry = native.DetectorProposalGeometry()
    geometry.inverse_rotation = wp.mat33d(*geometry_arguments.inverse_rotation)
    geometry.inverse_translation = wp.vec3d(*geometry_arguments.inverse_translation)
    geometry.detector_lower = wp.vec3d(*geometry_arguments.detector_lower)
    geometry.detector_u = wp.vec3d(*geometry_arguments.detector_u)
    geometry.detector_v = wp.vec3d(*geometry_arguments.detector_v)
    geometry.detector_shape = wp.vec2i(*geometry_arguments.detector_shape)
    geometry.detector_spacing = wp.vec2d(*geometry_arguments.detector_spacing)
    output = wp.zeros((len(draws), 4), dtype=wp.float64, device=device)
    status = wp.zeros(len(draws), dtype=wp.int32, device=device)
    wp.launch(
        _vertex_samples,
        dim=len(draws),
        inputs=[
            wp.array(planes, dtype=native.DetectorProposalPlane, device=device),
            len(planes),
            geometry,
            wp.array(previous, dtype=wp.vec3d, device=device),
            wp.array(draws, dtype=wp.float64, device=device),
            wp.array(coordinates, dtype=wp.vec2d, device=device),
            output,
            status,
        ],
        device=device,
    )
    return output.numpy(), status.numpy()


@pytest.mark.parametrize("rotate", [False, True])
def test_device_vertex_plan_matches_separate_prepared_proposals(warp_device, rotate):
    spec = _spec()
    if rotate:
        cosine, sine = math.cos(0.2), math.sin(0.2)
        spec = replace(
            spec,
            object_pose=RigidTransform(
                (cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0), (0.1, -0.05, 0.0)
            ),
        )
    previous = [
        spec.object_pose.point(value) for value in ((0.21, 0.17, 0.64), (-0.33, 0.49, 1.35))
    ]
    coordinates, draws = [(0.27, 0.73), (0.63, 0.37)], [0.73, 0.23]
    actual, status = _run_vertex(spec, warp_device, previous, draws, coordinates)
    assert np.all(status == 0)
    for index in range(2):
        proposal = prepare_detector_area_proposal(previous[index], spec)
        reference = sample_detector_area_proposal(
            proposal, component_draw=draws[index], coordinate=coordinates[index]
        )
        np.testing.assert_allclose(actual[index, :2], reference.coordinate, atol=2e-15, rtol=2e-13)
        assert actual[index, 2] == pytest.approx(reference.log_reference_density, abs=3e-13)
        assert actual[index, 3] == reference.component


@pytest.mark.parametrize("component_draw", [0.25, 0.55])
def test_actual_world_endpoint_loss_is_fatal_for_each_mixture_side(warp_device, component_draw):
    root5 = math.sqrt(5.0)
    detector = OrientedDetector(
        DetectorGeometry(
            (0.0, -1.0, 0.5),
            (1.0, 2.0, 0.5),
            (1.0 / root5, 0.0, 2.0 / root5),
            (2.0 / root5, 0.0, -1.0 / root5),
            (root5, root5),
            (1, 1),
        )
    )
    spec = TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
        detector,
        80.0,
        "oblique endpoint representation",
    )
    previous = (1e-20, 0.5, 0.5)
    proposal = prepare_detector_area_proposal(previous, spec)
    coordinate = (0.5, 1e-18)
    if component_draw >= 0.5:
        plane = proposal.planes[0]
        a, b, c = plane.coefficients
        other = a * 0.5 + c
        first = (
            math.copysign(math.log1p(abs(other) / plane.normal_separation_mm), other)
            if other
            else 0.0
        )
        last = math.log1p((other + b) / plane.normal_separation_mm)
        coordinate = (
            0.5,
            (math.log1p(2e-18 / plane.normal_separation_mm) - first) / (last - first),
        )
    actual, status = _run_vertex(spec, warp_device, [previous], [component_draw], [coordinate])
    assert status[0] == 4
    np.testing.assert_array_equal(actual[0], 0.0)
    with pytest.raises(TransportError, match="world endpoint"):
        sample_detector_area_proposal(
            proposal, component_draw=component_draw, coordinate=coordinate
        )


def test_device_vertex_on_grid_plane_is_fatal(warp_device):
    actual, status = _run_vertex(_spec(), warp_device, [(0.0, 0.1, 0.5)], [0.75], [(0.2, 0.3)])
    assert status[0] == 4
    np.testing.assert_array_equal(actual[0], 0.0)


def test_host_plan_rejects_detector_intersection():
    spec = _spec()
    with pytest.raises(TransportError, match=r"detector plane.*entire material grid"):
        detector_proposal_plan_arguments(
            replace(spec, object_pose=RigidTransform(translation_mm=(0.0, 0.0, 3.0)))
        )
