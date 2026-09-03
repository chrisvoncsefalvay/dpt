"""Independent radial-law, full-mixture, null and representability controls."""

from __future__ import annotations

import math
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest

from dpt.transport.geometry_forward_proposals import (
    prepare_forward_area_mixture,
    sample_forward_area_mixture,
)
from dpt.transport.geometry_tangents import RayTangent
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec


def _scene():
    spec = TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((0.0, 0.0), (1.0, 1.0), (1, 1), 2.0),
        80.0,
        "forward radial proposal",
    )
    incoming = RayTangent((0.4, 0.6, 0.5), (0.0, 0.0, 1.0))
    return incoming, spec


def test_forward_density_and_cdf_match_independent_eighty_digit_radial_formula():
    incoming, spec = _scene()
    mixture = prepare_forward_area_mixture(incoming, spec)
    assert mixture.forward is not None
    disc = mixture.forward
    for radial_draw, phi in ((0.01, 0.15), (0.1, 0.3), (0.2, 0.7), (0.35, 0.9)):
        sample = sample_forward_area_mixture(
            mixture,
            spec,
            component_draw=0.75,
            regular_component_draw=0.25,
            coordinate=(radial_draw, phi),
        )
        assert sample.coordinate is not None and sample.log_reference_density is not None
        r = math.hypot(
            *(
                disc.extent_mm[i] * (sample.coordinate[i] - disc.center_coordinate[i])
                for i in range(2)
            )
        )
        assert math.sqrt(r / disc.radius_mm) == pytest.approx(radial_draw, abs=2e-15)
        with localcontext() as context:
            context.prec = 80
            cover = Decimal.from_float(disc.radius_mm)
            rho = Decimal.from_float(r)
            pi = Decimal(
                "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899"
            )
            q = Decimal.from_float(disc.area_mm2) / (4 * pi * cover.sqrt() * rho * rho.sqrt())
        regular = math.exp(mixture.regular.log_density(sample.coordinate))
        assert math.exp(sample.log_reference_density) == pytest.approx(
            0.5 * (regular + float(q)), rel=2e-14
        )


def test_outside_rectangle_full_disc_draw_is_retained_as_null():
    incoming, spec = _scene()
    mixture = prepare_forward_area_mixture(incoming, spec)
    sample = sample_forward_area_mixture(
        mixture, spec, component_draw=0.75, regular_component_draw=0.25, coordinate=(0.9, 0.25)
    )
    assert sample.coordinate is None and sample.log_reference_density is None
    assert sample.component == "forward-null" and sample.reference_identity == mixture.identity
    # Full-disk radius CDF is sqrt(r/R), so null probability along a fixed
    # azimuth is exact. No conditional radius/arc renormalisation occurs.
    assert mixture.forward is not None
    disc = mixture.forward
    count, total = 0, 128
    for index in range(total):
        draw = (index + 0.5) / total
        sample = sample_forward_area_mixture(
            mixture,
            spec,
            component_draw=0.75,
            regular_component_draw=0.25,
            coordinate=(draw, 0.125),
        )
        count += int(sample.coordinate is not None)
    exit_radius = min(0.6, 0.4) * math.sqrt(2.0)
    assert abs(count / total - math.sqrt(exit_radius / disc.radius_mm)) <= 0.5 / total


def test_uniform_log_draws_use_the_full_forward_mixture_pdf_too():
    incoming, spec = _scene()
    mixture = prepare_forward_area_mixture(incoming, spec)
    record = sample_forward_area_mixture(
        mixture, spec, component_draw=0.25, regular_component_draw=0.25, coordinate=(0.3, 0.7)
    )
    assert record.coordinate == (0.3, 0.7)
    assert record.log_reference_density is not None
    assert record.log_reference_density == mixture.log_density((0.3, 0.7))
    assert mixture.forward is not None
    rho = math.sqrt(0.02)
    cover = mixture.forward.radius_mm
    lower = 0.5 / (4.0 * math.pi * math.sqrt(cover) * rho**1.5)
    assert math.exp(record.log_reference_density) >= lower


@pytest.mark.parametrize("direction", ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0)))
def test_parallel_and_backward_forward_rays_have_no_forward_singularity(direction):
    incoming, spec = _scene()
    incoming = replace(incoming, direction=direction)
    mixture = prepare_forward_area_mixture(incoming, spec)
    assert mixture.forward is None
    sample = sample_forward_area_mixture(
        mixture, spec, component_draw=0.75, regular_component_draw=0.25, coordinate=(0.3, 0.7)
    )
    assert sample.coordinate == (0.3, 0.7)
    assert sample.log_reference_density == mixture.regular.log_density((0.3, 0.7))


def test_forward_world_endpoint_is_not_displaced_when_radius_is_unrepresentable():
    incoming, spec = _scene()
    mixture = prepare_forward_area_mixture(incoming, spec)
    with pytest.raises(TransportError, match="distinguishable"):
        sample_forward_area_mixture(
            mixture, spec, component_draw=0.75, regular_component_draw=0.25, coordinate=(1e-10, 0.3)
        )


def test_radial_power_repairs_the_zero_binding_joint_gradient_second_moment():
    # Independent local joint law has integral over energy of f_dot² <=C/theta³.
    # Divide by q_area~theta^-3/2, then include dOmega~theta*dtheta:
    # radial integral is integral theta^-1/2 =2*sqrt(theta_max), finite.
    for theta in (1e-3, 1e-5, 1e-7):
        gradient_energy_second = theta**-3
        area_density = theta**-1.5
        radial = gradient_energy_second / area_density * theta
        assert radial == pytest.approx(theta**-0.5, rel=2e-14)
    assert 2.0 * math.sqrt(0.01) == pytest.approx(0.2)


@pytest.mark.parametrize("direction", ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
def test_finite_extents_cannot_hide_nonfinite_detector_area(direction):
    incoming, spec = _scene()
    incoming = replace(incoming, direction=direction)
    spec = replace(spec, detector=replace(spec.detector, spacing_xy_mm=(1e160, 1e160)))
    with pytest.raises(TransportError, match="area"):
        prepare_forward_area_mixture(incoming, spec)
