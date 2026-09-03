"""Independent detector density normalisation and grazing second-moment controls."""

from __future__ import annotations

import math
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest

from dpt.geometry import DetectorGeometry
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


def _scene() -> TransportSpec:
    return TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1, 1, 2), 2),
        PlanarDetector((-0.5, -0.5), (2.0, 2.0), (1, 1), 2.0),
        80.0,
        "grazing terminal interface density proposal",
    )


def test_every_relevant_face_is_a_detached_mixture_component_with_full_uniform_support() -> None:
    proposal = prepare_detector_area_proposal((0.4, 0.6, 0.5), _scene())
    assert len(proposal.planes) == 5
    assert len(proposal.identity) == 64
    for component in (0.25, 0.55, 0.65, 0.75, 0.85, 0.95):
        record = sample_detector_area_proposal(
            proposal, component_draw=component, coordinate=(0.3, 0.7)
        )
        assert record.reference_identity == proposal.identity
        assert all(0.0 < value < 1.0 for value in record.coordinate)
        assert math.exp(record.log_reference_density) >= 0.5
        assert record.log_reference_density == proposal.log_density(record.coordinate)


@pytest.mark.parametrize("coefficient", (1.0, -1.0))
def test_conditional_log_density_normalises_by_independent_split_simpson_integration(
    coefficient: float,
) -> None:
    plane = DetectorPlaneProposal(0, 1, (coefficient, 0.0, -coefficient * 0.4), 0.1)
    proposal = DetectorAreaProposal((plane,), "1" * 64)
    n, result = 512, 0.0
    # Split at the |g| cusp; no quadrature mesh straddles a nonsmooth point.
    for lower, upper in ((0.0, 0.4), (0.4, 1.0)):
        step = (upper - lower) / n
        values = []
        for i in range(n + 1):
            u = lower + i * step
            if u == 0.0:
                u = math.nextafter(0.0, 1.0)
            elif u == 1.0:
                u = math.nextafter(1.0, 0.0)
            weight = 1.0 if i in (0, n) else 4.0 if i % 2 else 2.0
            values.append(weight * math.exp(proposal.log_density((u, 0.7))))
        result += math.fsum(values) * step / 3.0
    assert result == pytest.approx(1.0, abs=2e-10)


@pytest.mark.parametrize("separation", (0.2, 1e-3, 1e-6))
def test_log_proposal_removes_the_local_two_endpoint_squared_grazing_singularity(
    separation: float,
) -> None:
    plane = DetectorPlaneProposal(0, 1, (1.0, 0.0, 0.0), separation)
    proposal = DetectorAreaProposal((plane,), "1" * 64)
    normal = math.log1p(1.0 / separation)
    # For score1/(delta+epsilon), its squared importance density is at most
    # 2*Z/(delta+epsilon). Conditional integration is <=2*Z², whose integral
    # over delta near zero is finite. Check the actual full mixture pointwise.
    for epsilon in (separation, 0.1, 0.9):
        score_squared = 1.0 / (separation + epsilon) ** 2
        density = math.exp(proposal.log_density((epsilon, 0.3)))
        assert score_squared / density <= 2.0 * normal / (separation + epsilon) * (1.0 + 1e-14)


def test_both_endpoints_cannot_be_on_a_material_face_without_a_separate_chart() -> None:
    with pytest.raises(TransportError, match="grid-plane boundary"):
        prepare_detector_area_proposal((0.5, 0.6, 0.5), _scene())


def test_log_sampler_retains_its_address_and_cannot_displace_an_unresolved_point() -> None:
    plane = DetectorPlaneProposal(0, 1, (1.0, 0.0, -0.4), 1e-100)
    proposal = DetectorAreaProposal((plane,), "1" * 64)
    with pytest.raises(TransportError, match="distinguishable"):
        sample_detector_area_proposal(proposal, component_draw=0.8, coordinate=(0.4, 0.7))


def test_detector_touching_material_requires_a_separate_endpoint_singularity_chart() -> None:
    spec = _scene()
    with pytest.raises(TransportError, match="detector"):
        spec = replace(spec, detector=replace(spec.detector, z_mm=spec.grid.upper_mm[2]))
        prepare_detector_area_proposal((0.4, 0.6, 0.5), spec)


def test_tiny_oblique_level_uses_the_same_affine_chart_as_the_sampler() -> None:
    # At u=.5, a*u+c cancels exactly. Grouping (a*u+b*v)+c instead
    # loses the small v contribution and overstates the density by 201x.
    plane = DetectorPlaneProposal(0, 0, (1.0, 2.0, -0.5), 1e-20)
    proposal = DetectorAreaProposal((plane,), "d" * 64)
    coordinate = (0.5, 1e-18)
    with localcontext() as context:
        context.prec = 80
        delta = Decimal.from_float(plane.normal_separation_mm)
        v = Decimal.from_float(coordinate[1])
        level = Decimal(2) * v
        width = (Decimal(1) + Decimal(2) / delta).ln()
        expected = Decimal("0.5") + Decimal(1) / (width * (delta + abs(level)))
    assert math.exp(proposal.log_density(coordinate)) == pytest.approx(float(expected), rel=2e-14)
    normal = math.log1p(2.0 / plane.normal_separation_mm)
    address = math.log1p(2.0 * coordinate[1] / plane.normal_separation_mm) / normal
    sampled = sample_detector_area_proposal(
        proposal, component_draw=0.75, coordinate=(0.5, address)
    )
    assert sampled.coordinate[1] == pytest.approx(coordinate[1], rel=2e-14)
    assert math.exp(sampled.log_reference_density) == pytest.approx(float(expected), rel=2e-14)


@pytest.mark.parametrize("component_draw", (0.25, 0.55))
def test_actual_world_endpoint_cannot_lose_a_tiny_oblique_plane_level(
    component_draw: float,
) -> None:
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
        "actual oblique endpoint representation",
    )
    proposal = prepare_detector_area_proposal((1e-20, 0.5, 0.5), spec)
    if component_draw < 0.5:
        coordinate = (0.5, 1e-18)
    else:
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
    with pytest.raises(TransportError, match="world endpoint"):
        sample_detector_area_proposal(
            proposal, component_draw=component_draw, coordinate=coordinate
        )
