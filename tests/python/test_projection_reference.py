"""Independent analytic interpolation and integral limits; no GPU imports."""

# Pytest 9.1 exposes approx() with incomplete annotations; production modules stay strict.
# pyright: reportUnknownMemberType=false

import math

import pytest

from dpt.contracts import ContractError
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.validation.projection import integrate_sampled_field, sample_reference
from dpt.volumes import GridSpec


def _geometry(source: float = -10.0, detector: float = 10.0) -> DetectorGeometry:
    return DetectorGeometry(
        (source, 0.0, 0.0),
        (detector, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0),
        (1, 1),
    )


def test_single_sample_support_and_constant_outer_extension() -> None:
    grid = GridSpec((1, 1, 1), (2.0, 3.0, 4.0))
    assert sample_reference([0.25], grid, (-0.5, 0.0, 0.5)) == 0.25
    assert sample_reference([0.25], grid, (-0.50001, 0.0, 0.0)) == 0.0
    assert integrate_sampled_field(
        [0.25], grid, _geometry(), RigidTransform(), 0, 0
    ) == pytest.approx(0.5)


def test_affine_interior_reproduction_and_outer_clamp() -> None:
    grid = GridSpec((2, 3, 4), (1.0, 1.0, 1.0))
    values = [1.0 + x + 2 * y + 3 * z for z in range(2) for y in range(3) for x in range(4)]
    assert sample_reference(values, grid, (0.2, 0.7, 0.4)) == pytest.approx(3.8)
    assert sample_reference(values, grid, (-0.4, 0.7, 0.4)) == pytest.approx(3.6)


def test_exact_piecewise_integration_includes_half_cells() -> None:
    grid = GridSpec((1, 1, 3), (2.0, 1.0, 1.0))
    # Two length-2 interpolating intervals plus two length-1 constant margins.
    expected = 1.0 * 1.0 + 2.0 * (1.0 + 3.0) / 2 + 2.0 * (3.0 + 2.0) / 2 + 1.0 * 2.0
    assert integrate_sampled_field(
        [1.0, 3.0, 2.0], grid, _geometry(), RigidTransform(), 0, 0
    ) == pytest.approx(expected)


def test_source_inside_support_truncates_the_integral() -> None:
    grid = GridSpec((1, 1, 3), (2.0, 1.0, 1.0))
    assert integrate_sampled_field(
        [0.5] * 3, grid, _geometry(0.0, 10.0), RigidTransform(), 0, 0
    ) == pytest.approx(2.5)


def test_detector_before_support_has_zero_integral() -> None:
    grid = GridSpec((1, 1, 3), (2.0, 1.0, 1.0))
    assert (
        integrate_sampled_field([1.0] * 3, grid, _geometry(-10.0, -3.0), RigidTransform(), 0, 0)
        == 0.0
    )


def test_independent_midpoint_refinement_converges_to_piecewise_integral() -> None:
    grid = GridSpec((1, 1, 3), (2.0, 1.0, 1.0))
    values = [1.0, 3.0, 2.0]
    exact = integrate_sampled_field(values, grid, _geometry(), RigidTransform(), 0, 0)
    coarse = integrate_sampled_field(
        values, grid, _geometry(), RigidTransform(), 0, 0, midpoint_samples=7
    )
    fine = integrate_sampled_field(
        values, grid, _geometry(), RigidTransform(), 0, 0, midpoint_samples=701
    )
    assert abs(fine - exact) < abs(coarse - exact) / 1000


def test_continuous_quadratic_integral_is_independent_of_grid_samples() -> None:
    from dpt.validation.projection import integrate_quadratic_field

    grid = GridSpec((1, 1, 3), (2.0, 1.0, 1.0), (-2.0, 0.0, 0.0))
    # Support is [-3,3]; integral 0.2+0.1*x^2 is 1.2+1.8.
    assert math.isclose(
        integrate_quadratic_field(
            grid, _geometry(), RigidTransform(), 0, 0, intercept=0.2, curvature=(0.1, 0.0, 0.0)
        ),
        3.0,
    )


@pytest.mark.parametrize("midpoint_samples", [None, 31])
def test_prevalidated_reference_preserves_values_and_shape_contract(
    midpoint_samples: int | None,
) -> None:
    grid = GridSpec((1, 1, 3), (2.0, 1.0, 1.0))
    values = [1.0, 3.0, 2.0]
    args = (values, grid, _geometry(), RigidTransform(), 0, 0)
    assert integrate_sampled_field(
        *args, midpoint_samples=midpoint_samples, validate=False
    ) == integrate_sampled_field(*args, midpoint_samples=midpoint_samples)
    with pytest.raises(ContractError, match="reference attenuation"):
        integrate_sampled_field(
            values[:-1], grid, _geometry(), RigidTransform(), 0, 0, validate=False
        )
    with pytest.raises(ContractError, match="reference attenuation"):
        integrate_sampled_field([1.0, float("nan"), 2.0], grid, _geometry(), RigidTransform(), 0, 0)
