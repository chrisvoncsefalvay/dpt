"""Explicit aperture physics, Decimal products and matched volume/pose derivatives."""

import importlib

# pyright: reportUnknownMemberType=false
import math
from collections.abc import Iterator
from contextlib import nullcontext
from decimal import Decimal, localcontext
from typing import Any

import pytest

from dpt._runtime import cpu_test_context
from dpt.aperture import (
    ApertureGeometry,
    aperture_projection_vjp,
    prepare_aperture_projection,
    project_aperture_counts,
)
from dpt.contracts import ContractError, TrialDomainError
from dpt.geometry import DetectorGeometry, RayBundleGeometry, RigidTransform, compose_pose
from dpt.projection import ProjectionSpec
from dpt.validation.projection import integrate_ray_bundle
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


@pytest.fixture(autouse=True)
def _kernel_context(warp_device: str) -> Iterator[None]:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def _two_rays() -> ApertureGeometry:
    return ApertureGeometry(
        RayBundleGeometry(((0.0, 0.0, -2.0), (1.0, 0.0, -2.0)), ((0.0, 0.0, 2.0), (1.0, 0.0, 2.0))),
        (0, 2),
        (0.25, 0.75),
    )


def _workspace(grid: GridSpec, geometry: ApertureGeometry, warp_device: str, integration: Any):
    return prepare_aperture_projection(
        grid,
        geometry,
        ProjectionSpec(17, active_volume=True, precision="float64", integration=integration),
        depths=wp.empty(geometry.rays.rays, dtype=wp.float64, device=warp_device),
        depth_cotangents=wp.empty(geometry.rays.rays, dtype=wp.float64, device=warp_device),
        device=warp_device,
    )


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_counts_average_transmission_and_decimal_volume_beam_gradients(
    integration: Any, warp_device: str
) -> None:
    grid = GridSpec((1, 1, 2), (1.0, 1.0, 1.0))
    workspace = _workspace(grid, _two_rays(), warp_device, integration)
    field = wp.array([1.0, 3.0], dtype=wp.float32, device=warp_device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    beam = wp.array([100.0], dtype=wp.float64, device=warp_device)
    seeds = wp.array([-1.5], dtype=wp.float64, device=warp_device)
    counts = wp.empty(1, dtype=wp.float64, device=warp_device)
    gradient = wp.empty(2, dtype=wp.float64, device=warp_device)
    beam_gradient = wp.empty(1, dtype=wp.float64, device=warp_device)
    project_aperture_counts(field, pose, beam, workspace=workspace, out_counts=counts)
    aperture_projection_vjp(
        field,
        pose,
        beam,
        cotangent=seeds,
        workspace=workspace,
        out_mu=gradient,
        out_beam=beam_gradient,
        volume_gradient_precision="float64",
    )
    with localcontext() as ctx:
        ctx.prec = 80
        transmission = Decimal("0.25") * (-Decimal(1)).exp() + Decimal("0.75") * (-Decimal(3)).exp()
        expected_count = float(Decimal(100) * transmission)
        expected_gradient = [
            float(Decimal(150) * Decimal("0.25") * (-Decimal(1)).exp()),
            float(Decimal(150) * Decimal("0.75") * (-Decimal(3)).exp()),
        ]
        expected_beam = float(Decimal("-1.5") * transmission)
    assert float(counts.numpy()[0]) == pytest.approx(expected_count, rel=3e-14, abs=0.0)
    assert gradient.numpy().tolist() == pytest.approx(expected_gradient, rel=3e-14, abs=0.0)
    assert float(beam_gradient.numpy()[0]) == pytest.approx(expected_beam, rel=3e-14, abs=0.0)
    assert abs(float(counts.numpy()[0]) - 100 * math.exp(-2.5)) > 4.0
    previous = gradient.numpy().tolist()
    aperture_projection_vjp(
        field,
        pose,
        beam,
        cotangent=seeds,
        workspace=workspace,
        out_mu=gradient,
        out_beam=beam_gradient,
        volume_gradient_precision="float64",
        accumulate=True,
    )
    assert gradient.numpy().tolist() == pytest.approx([2 * x for x in previous], rel=3e-14, abs=0.0)
    assert float(beam_gradient.numpy()[0]) == pytest.approx(2 * expected_beam, rel=3e-14, abs=0.0)


@pytest.mark.parametrize(
    "depth,beam_value,seed_value",
    [
        (0.0, 0.0, 1.0),
        (740.0, 1.0, 1.0),
        (1000.0, 2.0**1000, 1.0),
        (1500.0, 2.0**1000, 2.0**1000),
    ],
)
def test_deep_counts_and_gradients_retain_original_factor_range(
    depth: float, beam_value: float, seed_value: float, warp_device: str
) -> None:
    grid = GridSpec((1, 1, 2), (1.0, 1.0, 1.0))
    workspace = _workspace(grid, _two_rays(), warp_device, "cell_gauss")
    field = wp.array([depth, depth], dtype=wp.float32, device=warp_device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    beam = wp.array([beam_value], dtype=wp.float64, device=warp_device)
    seeds = wp.array([seed_value], dtype=wp.float64, device=warp_device)
    counts = wp.empty(1, dtype=wp.float64, device=warp_device)
    gradient = wp.empty(2, dtype=wp.float64, device=warp_device)
    beam_gradient = wp.empty(1, dtype=wp.float64, device=warp_device)
    project_aperture_counts(field, pose, beam, workspace=workspace, out_counts=counts)
    aperture_projection_vjp(
        field,
        pose,
        beam,
        cotangent=seeds,
        workspace=workspace,
        out_mu=gradient,
        out_beam=beam_gradient,
        volume_gradient_precision="float64",
    )
    with localcontext() as ctx:
        ctx.prec = 100
        factor = (-Decimal.from_float(depth)).exp()
        expected_count = float(Decimal.from_float(beam_value) * factor)
        expected_gradient = [
            float(
                -Decimal.from_float(seed_value)
                * Decimal.from_float(beam_value)
                * Decimal.from_float(weight)
                * factor
            )
            for weight in (0.25, 0.75)
        ]
        expected_beam = float(Decimal.from_float(seed_value) * factor)
    assert float(counts.numpy()[0]) == pytest.approx(
        expected_count, rel=3e-14, abs=3 * math.ulp(expected_count)
    )
    for actual, expected in zip(gradient.numpy().tolist(), expected_gradient, strict=True):
        assert actual == pytest.approx(expected, rel=3e-14, abs=3 * math.ulp(expected))
    assert float(beam_gradient.numpy()[0]) == pytest.approx(
        expected_beam, rel=3e-14, abs=3 * math.ulp(expected_beam)
    )
    if depth == 1500.0:
        assert float(counts.numpy()[0]) == 0.0
        assert all(x < 0.0 for x in gradient.numpy().tolist())


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_stratified_aperture_pose_vjp_matches_independent_finite_difference(
    integration: Any, warp_device: str
) -> None:
    grid = GridSpec((3, 4, 5), (1.2, 0.8, 1.1), (-2.0, -1.0, -1.0))
    detector = DetectorGeometry(
        (-12.0, 0.31, 0.43), (12.0, -0.6, -0.6), (0, 1, 0), (0, 0, 1), (0.3, 0.4), (2, 2)
    )
    geometry = ApertureGeometry.stratified(detector, samples_rows=2, samples_columns=3)
    workspace = _workspace(grid, geometry, warp_device, integration)
    field = wp.array(
        [
            0.07 + 0.012 * x + 0.019 * y + 0.011 * z + 0.003 * x * y
            for z in range(3)
            for y in range(4)
            for x in range(5)
        ],
        dtype=wp.float32,
        device=warp_device,
    )
    values = field.numpy().tolist()
    transform = compose_pose(RigidTransform(), (0.13, -0.08, 0.07, 0.02, -0.03, 0.01))
    pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device)
    beam_values = [20.0, 30.0, 40.0, 50.0]
    seed_values = [-0.5, 1.0, 0.75, -0.25]
    beam = wp.array(beam_values, dtype=wp.float64, device=warp_device)
    seeds = wp.array(seed_values, dtype=wp.float64, device=warp_device)
    gradient = wp.empty(6, dtype=wp.float64, device=warp_device)
    aperture_projection_vjp(
        field, pose, beam, cotangent=seeds, workspace=workspace, out_pose_tangent=gradient
    )

    def objective(current: RigidTransform) -> float:
        total = []
        for pixel in range(geometry.pixels):
            transmitted = math.fsum(
                geometry.weights[ray]
                * math.exp(
                    -integrate_ray_bundle(
                        values,
                        grid,
                        geometry.rays,
                        current,
                        ray,
                        midpoint_samples=17 if integration == "midpoint" else None,
                    )
                )
                for ray in range(geometry.pixel_offsets[pixel], geometry.pixel_offsets[pixel + 1])
            )
            total.append(seed_values[pixel] * beam_values[pixel] * transmitted)
        return math.fsum(total)

    for axis, measured in enumerate(gradient.numpy().tolist()):
        errors = []
        for step in (1e-4, 3e-5, 1e-5, 3e-6):
            delta = tuple(step if i == axis else 0.0 for i in range(6))
            difference = (
                objective(compose_pose(transform, delta))
                - objective(compose_pose(transform, tuple(-x for x in delta)))
            ) / (2 * step)
            errors.append(abs(difference - measured))
        assert min(errors) < 3e-7 * max(1.0, abs(measured)), (axis, errors, measured)


def test_stratification_has_named_anisotropic_offsets_and_explicit_uniform_weights(
    warp_device: str,
) -> None:
    detector = DetectorGeometry(
        (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0, 1, 0), (0, 0, 1), (2.0, 4.0), (1, 1)
    )
    geometry = ApertureGeometry.stratified(detector, samples_rows=2, samples_columns=2)
    assert geometry.rays.endpoints_mm == (
        (2.0, -0.5, -1.0),
        (2.0, 0.5, -1.0),
        (2.0, -0.5, 1.0),
        (2.0, 0.5, 1.0),
    )
    assert geometry.weights == (0.25, 0.25, 0.25, 0.25)
    assert geometry.pixel_offsets == (0, 4)


@pytest.mark.parametrize(
    "weights,offsets",
    [
        ((0.25, 0.25), (0, 2)),
        ((-0.25, 1.25), (0, 2)),
        ((math.nan, 0.75), (0, 2)),
        ((0.25,), (0, 2)),
        ((0.25, 0.75), (0, 0, 2)),
        ((0.25, 0.75), (0, 3)),
        ((0.25, 0.75), (False, 2)),
    ],
)
def test_invalid_aperture_normalization_and_partition_are_rejected(
    weights: Any, offsets: Any
) -> None:
    with pytest.raises(ContractError):
        ApertureGeometry(_two_rays().rays, offsets, weights)


@pytest.mark.parametrize("invalid", [-1.0, math.nan, math.inf])
def test_beam_is_validated_before_count_outputs_and_aliases_are_rejected(
    invalid: float, warp_device: str
) -> None:
    workspace = _workspace(
        GridSpec((1, 1, 2), (1.0, 1.0, 1.0)), _two_rays(), warp_device, "cell_gauss"
    )
    field = wp.ones(2, dtype=wp.float32, device=warp_device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    beam = wp.array([invalid], dtype=wp.float64, device=warp_device)
    counts = wp.full(1, 123.0, dtype=wp.float64, device=warp_device)
    with pytest.raises(ContractError):
        project_aperture_counts(field, pose, beam, workspace=workspace, out_counts=counts)
    assert counts.numpy().tolist() == [123.0]
    with pytest.raises(ContractError, match="overlaps"):
        project_aperture_counts(field, pose, beam, workspace=workspace, out_counts=beam)


def test_physical_aperture_rejects_negative_fields_and_unsupported_depth_policy(
    warp_device: str,
) -> None:
    grid = GridSpec((1, 1, 2), (1.0, 1.0, 1.0))
    geometry = _two_rays()
    workspace = _workspace(grid, geometry, warp_device, "cell_gauss")
    field = wp.array([-1.0, 3.0], dtype=wp.float32, device=warp_device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    beam = wp.ones(1, dtype=wp.float64, device=warp_device)
    counts = wp.full(1, 123.0, dtype=wp.float64, device=warp_device)
    with pytest.raises(TrialDomainError):
        project_aperture_counts(field, pose, beam, workspace=workspace, out_counts=counts)
    assert counts.numpy().tolist() == [123.0]
    for spec in (
        ProjectionSpec(1),
        ProjectionSpec(1, precision="float64", field_domain="signed_linear"),
    ):
        with pytest.raises(ContractError, match="physical fields and float64"):
            prepare_aperture_projection(
                grid,
                geometry,
                spec,
                depths=workspace.depths,
                depth_cotangents=workspace.depth_cotangents,
                device=warp_device,
            )
