"""Full disc, mixture density and retained-null native geometry checks."""

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

from dpt.geometry import DetectorGeometry
from dpt.transport.geometry_forward_proposal_arguments import forward_proposal_plan_arguments
from dpt.transport.geometry_forward_proposals import (
    prepare_forward_area_mixture,
    sample_forward_area_mixture,
)
from dpt.transport.model import (
    OrientedDetector,
    PlanarDetector,
    TransportError,
)
from helpers.geometry_forward_native import (
    _run,
    _scene,
)

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.geometry_forward_proposal_kernels")
regular_native: Any = importlib.import_module("dpt.transport.geometry_detector_proposal_kernels")


def test_full_radial_law_and_density_against_independent_decimal(warp_device):
    incoming, spec = _scene()
    mixture = prepare_forward_area_mixture(incoming, spec)
    assert mixture.forward is not None
    disc = mixture.forward
    coordinates = [(0.01, 0.15), (0.1, 0.3), (0.2, 0.7), (0.35, 0.9)]
    output, status = _run(incoming, spec, warp_device, [0.75] * 4, [0.25] * 4, coordinates)
    assert np.all(status == 0)
    for index, address in enumerate(coordinates):
        mapped = (float(output[index, 0]), float(output[index, 1]))
        rho = math.hypot(
            *(
                disc.extent_mm[axis] * (mapped[axis] - disc.center_coordinate[axis])
                for axis in range(2)
            )
        )
        assert math.sqrt(rho / disc.radius_mm) == pytest.approx(address[0], abs=2e-15)
        with localcontext() as context:
            context.prec = 90
            area, radius, separation = (
                Decimal.from_float(value) for value in (disc.area_mm2, disc.radius_mm, rho)
            )
            pi = Decimal(
                "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803483"
            )
            radial = area / (4 * pi * radius.sqrt() * separation * separation.sqrt())
        reference = 0.5 * (math.exp(mixture.regular.log_density(mapped)) + float(radial))
        assert output[index, 2] == pytest.approx(math.log(reference), abs=3e-13)
        assert output[index, 3] == 1


def test_full_disc_null_frequency_keeps_original_denominator(warp_device):
    incoming, spec = _scene()
    total = 128
    coordinates = [((index + 0.5) / total, 0.125) for index in range(total)]
    output, status = _run(incoming, spec, warp_device, [0.75] * total, [0.25] * total, coordinates)
    assert set(status.tolist()) == {0, -1}
    disc = prepare_forward_area_mixture(incoming, spec).forward
    assert disc is not None
    exit_radius = min(0.6, 0.4) * math.sqrt(2.0)
    accepted = int(np.count_nonzero(status == 0))
    assert abs(accepted / total - math.sqrt(exit_radius / disc.radius_mm)) <= 0.5 / total
    np.testing.assert_array_equal(output[status == -1], 0.0)


@pytest.mark.parametrize("regular_draw", [0.25, 0.75])
def test_regular_draws_include_the_forward_density(warp_device, regular_draw):
    incoming, spec = _scene()
    mixture = prepare_forward_area_mixture(incoming, spec)
    output, status = _run(incoming, spec, warp_device, [0.25], [regular_draw], [(0.3, 0.7)])
    assert status[0] == 0
    reference = sample_forward_area_mixture(
        mixture,
        spec,
        component_draw=0.25,
        regular_component_draw=regular_draw,
        coordinate=(0.3, 0.7),
    )
    assert reference.coordinate is not None and reference.log_reference_density is not None
    np.testing.assert_allclose(output[0, :2], reference.coordinate, atol=2e-15, rtol=2e-13)
    assert output[0, 2] == pytest.approx(reference.log_reference_density, abs=3e-13)
    assert output[0, 3] == 0


@pytest.mark.parametrize("direction", [(1.0, 0.0, 0.0), (0.0, 0.0, -1.0)])
def test_parallel_and_backward_direction_retains_regular_law(warp_device, direction):
    incoming, spec = _scene()
    incoming = replace(incoming, direction=direction)
    output, status = _run(incoming, spec, warp_device, [0.75], [0.25], [(0.3, 0.7)])
    assert status[0] == 0
    mixture = prepare_forward_area_mixture(incoming, spec)
    assert mixture.forward is None
    assert output[0, 2] == pytest.approx(mixture.regular.log_density((0.3, 0.7)), abs=3e-13)
    np.testing.assert_array_equal(output[0, :2], (0.3, 0.7))


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_chart_density_is_invariant_to_physical_unit_scale(warp_device, scale):
    incoming, spec = _scene(scale)
    output, status = _run(incoming, spec, warp_device, [0.75], [0.25], [(0.19, 0.73)])
    assert status[0] == 0
    mixture = prepare_forward_area_mixture(incoming, spec)
    reference = sample_forward_area_mixture(
        mixture, spec, component_draw=0.75, regular_component_draw=0.25, coordinate=(0.19, 0.73)
    )
    assert reference.coordinate is not None and reference.log_reference_density is not None
    assert output[0, 2] == pytest.approx(reference.log_reference_density, abs=3e-12)
    np.testing.assert_allclose(output[0, :2], reference.coordinate, atol=2e-15, rtol=2e-13)


@pytest.mark.parametrize("component_draw,coordinate", [(0.75, (1e-10, 0.3)), (0.25, (0.4, 0.6))])
def test_unresolved_or_exact_center_is_fatal_without_displacement(
    warp_device, component_draw, coordinate
):
    incoming, spec = _scene()
    output, status = _run(incoming, spec, warp_device, [component_draw], [0.25], [coordinate])
    assert status[0] == 4
    np.testing.assert_array_equal(output[0], 0.0)
    with pytest.raises(TransportError, match=r"distinguishable|exact center"):
        sample_forward_area_mixture(
            prepare_forward_area_mixture(incoming, spec),
            spec,
            component_draw=component_draw,
            regular_component_draw=0.25,
            coordinate=coordinate,
        )


def test_bad_slice_is_fatal_even_if_disc_draw_would_be_null(warp_device):
    incoming, spec = _scene()
    output, status = _run(
        incoming, spec, warp_device, [0.75], [0.25], [(0.9, 0.25)], invalid_slice=True
    )
    assert status[0] == 4
    np.testing.assert_array_equal(output[0], 0.0)


@pytest.mark.parametrize("direction", [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)])
def test_area_overflow_is_fatal_before_any_forward_branch(warp_device, direction):
    incoming, spec = _scene()
    spec = replace(spec, detector=PlanarDetector((0.0, 0.0), (1e160, 1e160), (1, 1), 2.0))
    incoming = replace(incoming, direction=direction)
    output, status = _run(incoming, spec, warp_device, [0.75], [0.25], [(0.3, 0.7)])
    assert status[0] == 4
    np.testing.assert_array_equal(output[0], 0.0)


def test_oriented_detector_normal_preserves_incident_side():
    _, spec = _scene()
    detector = OrientedDetector(
        DetectorGeometry(
            (0.0, 0.0, 0.0), (0.5, 0.5, 2.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (1.0, 1.0), (1, 1)
        )
    )
    _, arguments = forward_proposal_plan_arguments(replace(spec, detector=detector))
    assert arguments.detector_normal == (0.0, 0.0, 1.0)
