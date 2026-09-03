"""Native terminal cone coarea checks against analytic areas and host jets."""

# Warp annotations are runtime DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
import importlib
import math
from dataclasses import fields, replace
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pytest

from dpt.geometry import DetectorGeometry
from dpt.transport.compton_boundary_arguments import free_compton_boundary_arguments
from dpt.transport.compton_geometry_boundary import FreeComptonEnergyBoundary
from dpt.transport.geometry_tangents import RayTangent, RigidMotion
from dpt.transport.model import (
    ELECTRON_REST_ENERGY_KEV,
    HistoryStatus,
    OrientedDetector,
    PlanarDetector,
)

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.compton_geometry_boundary_kernels")
transport_kernels: Any = importlib.import_module("dpt.transport.kernels")

ENERGY, COSINE, HEIGHT = 100.0, 0.8, 2.0
THRESHOLD = 1 / (1 / ENERGY + (1 - COSINE) / ELECTRON_REST_ENERGY_KEV)


@wp.kernel
def _sample(
    boundary: native.NativeFreeComptonBoundary,
    detector: native.NativeBoundaryDetector,
    draws: wp.array(dtype=wp.float64),
    output: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    sample = native.sample_free_compton_energy_boundary(boundary, draws[index], detector)
    status[index] = sample.status
    if sample.status == 0:
        for component in range(2):
            output[index, component] = sample.coordinate[component]
        for component in range(3):
            output[index, 2 + component] = sample.endpoint.point_mm[component]
            output[index, 5 + component] = sample.endpoint.point_dot[component]
            output[index, 8 + component] = sample.endpoint.normal[component]
            output[index, 11 + component] = sample.endpoint.normal_dot[component]
            output[index, 17 + component] = sample.outgoing.origin_mm[component]
            output[index, 20 + component] = sample.outgoing.direction[component]
            output[index, 23 + component] = sample.outgoing.origin_dot[component]
            output[index, 26 + component] = sample.outgoing.direction_dot[component]
        output[index, 14] = wp.float64(sample.endpoint.pixel)
        output[index, 15] = sample.endpoint.area_jacobian
        output[index, 16] = sample.endpoint.area_jacobian_dot
        output[index, 29] = sample.outgoing_energy
        output[index, 30] = sample.outgoing_energy_dot
        output[index, 31] = sample.levelset_dot
        output[index, 32] = sample.world_area_weight
        output[index, 33] = sample.reference_area_weight


@wp.kernel
def _cosines(
    boundaries: wp.array(dtype=native.NativeFreeComptonBoundary),
    output: wp.array(dtype=wp.vec2d),
):
    index = wp.tid()
    output[index] = native.free_compton_boundary_cosine(boundaries[index])


@wp.kernel
def _physical_kn_cap(
    boundary: native.NativeFreeComptonBoundary,
    detector: native.NativeBoundaryDetector,
    draws: wp.array(dtype=wp.float64),
    contributions: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    sample = native.sample_free_compton_energy_boundary(boundary, draws[index], detector)
    status[index] = sample.status
    if sample.status == 0:
        displacement = sample.endpoint.point_mm - boundary.incoming.origin_mm
        distance = wp.length(displacement)
        cosine = wp.dot(boundary.incoming.direction, sample.outgoing.direction)
        density = transport_kernels.free_electron_phase_pdf(boundary.incident_energy, cosine)
        projection = wp.dot(detector.normal, sample.outgoing.direction)
        contributions[index] = (
            sample.world_area_weight * density * projection / (distance * distance)
        )
    elif sample.status == -1:
        contributions[index] = wp.float64(0.0)


def _detector() -> PlanarDetector:
    return PlanarDetector((-3.0, -3.0), (3.0, 3.0), (2, 2), HEIGHT)


def _boundary(*, energy_dot: float = 0.0, threshold_dot: float = 0.0) -> FreeComptonEnergyBoundary:
    return FreeComptonEnergyBoundary(
        RayTangent((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), ENERGY, THRESHOLD, energy_dot, threshold_dot
    )


def _packed(
    boundary: FreeComptonEnergyBoundary,
    detector: PlanarDetector | OrientedDetector,
    motion: RigidMotion | None = None,
    spacing_dot: tuple[float, float] = (0.0, 0.0),
) -> tuple[Any, Any]:
    args, detector_args = free_compton_boundary_arguments(
        boundary, detector, motion, spacing_dot=spacing_dot
    )
    result, frame = native.NativeFreeComptonBoundary(), native.NativeBoundaryDetector()
    ray = native.NativeBoundaryRay()
    for field in fields(args.incoming):
        setattr(ray, field.name, getattr(args.incoming, field.name))
    result.incoming = ray
    for name in (
        "incident_energy",
        "threshold_energy",
        "incident_energy_dot",
        "threshold_energy_dot",
    ):
        setattr(result, name, getattr(args, name))
    for field in fields(detector_args):
        setattr(frame, field.name, getattr(detector_args, field.name))
    return result, frame


def _run(
    device: str,
    boundary: FreeComptonEnergyBoundary,
    detector: PlanarDetector | OrientedDetector,
    motion: RigidMotion | None = None,
    spacing_dot: tuple[float, float] = (0.0, 0.0),
    draws: list[float] | None = None,
    count: int = 128,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[float]]:
    draws = [(index + 0.5) / count for index in range(count)] if draws is None else draws
    arguments = _packed(boundary, detector, motion, spacing_dot)
    output = wp.array(np.full((len(draws), 34), np.nan), dtype=wp.float64, device=device)
    status = wp.full(len(draws), -9, dtype=wp.int32, device=device)
    wp.launch(
        _sample,
        len(draws),
        inputs=[*arguments, wp.array(draws, dtype=wp.float64, device=device)],
        outputs=[output, status],
        device=device,
        record_tape=False,
    )
    return output.numpy(), status.numpy(), draws


def _mean(
    output: np.ndarray[Any, Any],
    status: np.ndarray[Any, Any],
    moment: bool = False,
    jump: float = 1.0,
) -> float:
    assert np.all(np.isin(status, (0, -1)))
    accepted = status == 0
    scores = np.zeros(len(status))
    scores[accepted] = output[accepted, 32] * jump
    if moment:
        scores[accepted] *= output[accepted, 2]
    return math.fsum(float(value) for value in scores) / len(status)


@pytest.mark.parametrize(
    "energy_dot,threshold_dot", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.2, -0.7))
)
def test_native_energy_threshold_jet_matches_exact_disk_area(
    warp_device: str, energy_dot: float, threshold_dot: float
) -> None:
    boundary = _boundary(energy_dot=energy_dot, threshold_dot=threshold_dot)
    output, status, _ = _run(warp_device, boundary, _detector())
    assert np.all(status == 0)
    with localcontext() as context:
        context.prec = 90
        mass, e, threshold = map(Decimal.from_float, (ELECTRON_REST_ENERGY_KEV, ENERGY, THRESHOLD))
        c = 1 - mass * (e - threshold) / e / threshold
        cdot = mass * (
            Decimal.from_float(threshold_dot) / threshold / threshold
            - Decimal.from_float(energy_dot) / e / e
        )
        expected = float(
            -2 * Decimal.from_float(math.pi) * Decimal.from_float(HEIGHT) ** 2 * cdot / c**3
        )
    assert _mean(output, status) == pytest.approx(expected, rel=3e-14, abs=3e-14)


@pytest.mark.parametrize("object_moves", (False, True))
def test_native_object_detector_normal_motion_has_opposite_area_derivatives(
    warp_device: str, object_moves: bool
) -> None:
    boundary = _boundary()
    motion = None if object_moves else RigidMotion((0.0, 0.0, 1.0))
    if object_moves:
        boundary = replace(
            boundary, incoming=replace(boundary.incoming, origin_dot=(0.0, 0.0, 1.0))
        )
    output, status, _ = _run(warp_device, boundary, _detector(), motion)
    expected = 2 * math.pi * HEIGHT * (1 / COSINE**2 - 1) * (-1 if object_moves else 1)
    assert _mean(output, status) == pytest.approx(expected, rel=3e-14, abs=3e-14)


@pytest.mark.parametrize("detector_rotates", (False, True))
def test_native_incoming_angle_and_detector_rotation_have_opposite_first_moments(
    warp_device: str, detector_rotates: bool
) -> None:
    boundary = _boundary()
    motion = RigidMotion(angular=(0.0, 1.0, 0.0)) if detector_rotates else None
    if not detector_rotates:
        boundary = replace(
            boundary, incoming=replace(boundary.incoming, direction_dot=(1.0, 0.0, 0.0))
        )
    output, status, _ = _run(warp_device, boundary, _detector(), motion)
    expected = math.pi * HEIGHT**3 * (1 - COSINE**2) / COSINE**4 * (-1 if detector_rotates else 1)
    assert _mean(output, status, moment=True) == pytest.approx(expected, rel=3e-14, abs=3e-14)
    assert _mean(output, status) == pytest.approx(0.0, abs=3e-14)


def test_native_null_azimuths_keep_original_denominator_and_signed_jump(warp_device: str) -> None:
    detector = PlanarDetector((0.0, 0.0), (3.0, 3.0), (1, 1), HEIGHT)
    output, status, _ = _run(warp_device, _boundary(energy_dot=1.0), detector)
    assert np.count_nonzero(status == 0) == 32
    assert np.count_nonzero(status == -1) == 96
    assert np.isnan(output[status == -1]).all()
    expected = 2 * math.pi * HEIGHT**2 * ELECTRON_REST_ENERGY_KEV / (COSINE**3 * ENERGY**2) / 4
    assert _mean(output, status) == pytest.approx(expected, rel=3e-14)
    assert _mean(output, status, jump=-2.7) == pytest.approx(-2.7 * expected, rel=3e-14)


def test_native_common_rigid_motion_cancels_boundary_and_energy_jets(warp_device: str) -> None:
    motion = RigidMotion((0.4, -0.2, 0.3), (0.2, -0.1, 0.3), (1.0, -1.0, 0.5))
    boundary = _boundary()
    boundary = replace(
        boundary,
        incoming=replace(
            boundary.incoming,
            origin_dot=motion.point_velocity((0.0, 0.0, 0.0)),
            direction_dot=motion.direction_velocity((0.0, 0.0, 1.0)),
        ),
    )
    output, status, _ = _run(warp_device, boundary, _detector(), motion, count=16)
    assert np.all(status == 0)
    np.testing.assert_allclose(output[:, 31:33], 0.0, rtol=0, atol=3e-14)


@pytest.mark.parametrize("oriented", (False, True))
def test_native_spacing_anchor_and_reference_weight_include_jacobian_once(
    warp_device: str, oriented: bool
) -> None:
    detector = (
        OrientedDetector(
            DetectorGeometry(
                (0.0, 0.0, -1.0),
                (-1.5, -1.5, HEIGHT),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (3.0, 3.0),
                (2, 2),
            )
        )
        if oriented
        else _detector()
    )
    output, status, draws = _run(
        warp_device, _boundary(), detector, spacing_dot=(0.3, 0.6), count=16
    )
    assert np.all(status == 0)
    np.testing.assert_allclose(output[:, 33] * output[:, 15], output[:, 32], rtol=3e-15, atol=3e-15)
    for row, draw in zip(output, draws, strict=True):
        host = _boundary().sample(draw, detector, spacing_dot=(0.3, 0.6))
        assert host is not None
        assert row[5:8] == pytest.approx(host.endpoint.point_dot, rel=3e-14, abs=3e-14)
        if oriented:
            assert row[5:8] == pytest.approx(
                ((row[2] + 1.5) * 0.1, (row[3] + 1.5) * 0.2, 0.0), rel=3e-14, abs=3e-14
            )


def test_native_oblique_plane_coarea_matches_independent_curve_arclength(warp_device: str) -> None:
    angle = 0.31
    detector = OrientedDetector(
        DetectorGeometry(
            (0.0, 0.0, -1.0),
            (-2.0, -2.0, 3.0),
            (math.cos(angle), 0.0, -math.sin(angle)),
            (0.0, 1.0, 0.0),
            (1.0, 1.0),
            (6, 6),
        )
    )
    boundary = _boundary(energy_dot=1.0)
    output, status, _ = _run(warp_device, boundary, detector, draws=[0.07, 0.19, 0.39, 0.63, 0.87])
    assert np.count_nonzero(status == 0) >= 3
    for row in output[status == 0]:
        d, k, n = row[20:23], np.array(boundary.incoming.direction), np.array(detector.normal)
        r = math.hypot(*(float(value) for value in row[2:5]))
        dphi = np.cross(k, d)
        projection = float(np.dot(n, d))
        rphi = -r * float(np.dot(n, dphi)) / projection
        curve = rphi * d + r * dphi
        gradient = (k - boundary.cosine * d) / r
        plane_gradient = gradient - float(np.dot(gradient, n)) * n
        ratio = math.hypot(*(float(value) for value in curve)) / math.hypot(
            *(float(value) for value in plane_gradient)
        )
        expected = -2 * math.pi * ratio * boundary.cosine_dot
        assert row[32] == pytest.approx(expected, rel=3e-14, abs=3e-14)


def test_native_attached_endpoint_energy_jet_matches_host_all_fields(warp_device: str) -> None:
    boundary = _boundary(energy_dot=0.7, threshold_dot=0.2)
    motion = RigidMotion((0.0, 0.0, 0.9))
    output, status, draws = _run(
        warp_device, boundary, _detector(), motion, draws=[0.07, 0.17, 0.39, 0.63, 0.87]
    )
    assert np.all(status == 0)
    for row, draw in zip(output, draws, strict=True):
        host = boundary.sample(draw, _detector(), motion)
        assert host is not None
        values = (
            *host.coordinate,
            *host.endpoint.point_mm,
            *host.endpoint.point_dot,
            *host.endpoint.normal,
            *host.endpoint.normal_dot,
            float(host.endpoint.pixel),
            host.endpoint.area_jacobian,
            host.endpoint.area_jacobian_dot,
            *host.outgoing.origin_mm,
            *host.outgoing.direction,
            *host.outgoing.origin_dot,
            *host.outgoing.direction_dot,
            host.outgoing_energy_kev,
            host.outgoing_energy_dot,
            host.levelset_dot_kev,
            host.world_area_weight,
            host.reference_area_weight,
        )
        np.testing.assert_allclose(row, values, rtol=2e-13, atol=2e-13)
        c_dot = float(np.dot(boundary.incoming.direction, row[26:29]))
        expected = row[29] ** 2 * (0.7 / ENERGY**2 + c_dot / ELECTRON_REST_ENERGY_KEV)
        assert row[30] == pytest.approx(expected, rel=3e-14, abs=3e-14)
        assert row[31] == pytest.approx(expected - 0.2, rel=3e-14, abs=3e-14)


def test_native_endpoint_activity_and_inactive_thresholds_distinguish_null_and_fatal(
    warp_device: str,
) -> None:
    cases = [
        (THRESHOLD / 10, 0.0, -1),
        (ENERGY * 2, 0.0, -1),
        (ENERGY, 0.0, -1),
        (ENERGY, 1.0, int(HistoryStatus.NUMERICAL_FAILURE)),
    ]
    for threshold, threshold_dot, expected in cases:
        boundary = replace(
            _boundary(), threshold_energy_kev=threshold, threshold_energy_dot=threshold_dot
        )
        output, status, _ = _run(warp_device, boundary, _detector(), draws=[0.17])
        assert status.tolist() == [expected]
        assert np.isnan(output).all()


@pytest.mark.parametrize("which_moves", ("source", "detector", "common"))
def test_native_right_angle_pixel_edge_admission_matches_one_sided_chart(
    warp_device: str, which_moves: str
) -> None:
    motion = RigidMotion((1.0, 0.0, 0.0))
    ray = RayTangent(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        origin_dot=(1.0, 0.0, 0.0) if which_moves != "detector" else (0.0, 0.0, 0.0),
    )
    boundary = FreeComptonEnergyBoundary(
        ray, ELECTRON_REST_ENERGY_KEV, ELECTRON_REST_ENERGY_KEV / 2
    )
    output, status, _ = _run(
        warp_device,
        boundary,
        _detector(),
        motion if which_moves != "source" else None,
        draws=[0.72],
    )
    assert boundary.cosine == 0.0
    if which_moves == "common":
        assert status.tolist() == [0]
        assert output[0, 32] == 0.0
    else:
        assert status.tolist() == [int(HistoryStatus.NUMERICAL_FAILURE)]
        assert np.isnan(output).all()


def test_native_coincident_plane_draw_domain_and_backward_support_status(warp_device: str) -> None:
    output, status, _ = _run(
        warp_device, _boundary(), _detector(), draws=[0.0, 1.0, -0.1, math.nan, math.inf]
    )
    assert np.all(status == int(HistoryStatus.NUMERICAL_FAILURE)) and np.isnan(output).all()
    for detector, expected in (
        (replace(_detector(), z_mm=0.0), 4),
        (replace(_detector(), z_mm=-1.0), -1),
        (replace(_detector(), lower_xy_mm=(100.0, 100.0)), -1),
    ):
        output, status, _ = _run(warp_device, _boundary(), detector, draws=[0.17])
        assert status.tolist() == [expected] and np.isnan(output).all()


def test_native_cosine_keeps_weak_energy_difference_and_decimal_tangent(warp_device: str) -> None:
    boundaries = [
        _boundary(energy_dot=1.2, threshold_dot=-0.7),
        replace(_boundary(), threshold_energy_kev=math.nextafter(ENERGY, 0.0)),
        replace(_boundary(), incident_energy_kev=1e-150, threshold_energy_kev=0.9e-150),
    ]
    packed = [_packed(boundary, _detector())[0] for boundary in boundaries]
    output = wp.empty(len(packed), dtype=wp.vec2d, device=warp_device)
    wp.launch(
        _cosines,
        len(packed),
        inputs=[wp.array(packed, dtype=native.NativeFreeComptonBoundary, device=warp_device)],
        outputs=[output],
        device=warp_device,
        record_tape=False,
    )
    for boundary, actual in zip(boundaries, output.numpy(), strict=True):
        with localcontext() as context:
            context.prec = 90
            m, e, threshold = map(
                Decimal.from_float,
                (
                    ELECTRON_REST_ENERGY_KEV,
                    boundary.incident_energy_kev,
                    boundary.threshold_energy_kev,
                ),
            )
            expected = float(1 - m * (e - threshold) / e / threshold)
            tangent = float(
                m
                * (
                    Decimal.from_float(boundary.threshold_energy_dot) / threshold / threshold
                    - Decimal.from_float(boundary.incident_energy_dot) / e / e
                )
            )
        assert actual[0] == pytest.approx(expected, abs=2 * math.ulp(expected), rel=0)
        assert actual[1] == pytest.approx(tangent, abs=2 * math.ulp(tangent), rel=0)


def test_native_physical_kn_count_matches_differentiated_normalised_cap_probability(
    warp_device: str,
) -> None:
    boundary, detector = _boundary(threshold_dot=1.0), _detector()
    count = 128
    output = wp.array(np.full(count, np.nan), dtype=wp.float64, device=warp_device)
    status = wp.full(count, -9, dtype=wp.int32, device=warp_device)
    wp.launch(
        _physical_kn_cap,
        count,
        inputs=[
            *_packed(boundary, detector),
            wp.array(
                [(index + 0.5) / count for index in range(count)],
                dtype=wp.float64,
                device=warp_device,
            ),
        ],
        outputs=[output, status],
        device=warp_device,
        record_tape=False,
    )
    assert np.all(status.numpy() == 0)
    with localcontext() as context:
        context.prec = 90
        energy, threshold, mass = map(
            Decimal.from_float, (ENERGY, THRESHOLD, ELECTRON_REST_ENERGY_KEV)
        )
        alpha = energy / mass
        first = 1 - 2 / alpha - 2 / alpha**2
        second = (2 * alpha + 1) / alpha**2

        def antiderivative(u: Decimal) -> Decimal:
            return -1 / (2 * u * u) + first * u.ln() - second / u + u / alpha**2

        # u=E/L transforms the threshold cap CDF into an elementary integral.
        # Differentiate that closed CDF with respect to L, independently of
        # the surface sampler, its area weights and the native phase total.
        upper = energy / threshold
        denominator = antiderivative(1 + 2 * alpha) - antiderivative(Decimal(1))
        probability = (antiderivative(upper) - antiderivative(Decimal(1))) / denominator
        derivative_u = 1 / upper**3 + first / upper + second / upper**2 + 1 / alpha**2
        derivative = derivative_u * (-energy / threshold**2) / denominator
    assert 0 < probability < 1
    assert math.fsum(float(value) for value in output.numpy()) / count == pytest.approx(
        float(derivative), rel=1e-13, abs=1e-15
    )
