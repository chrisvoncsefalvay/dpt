"""Private FP64 material sums retain FP32 increments and checked public storage."""

import importlib
import math
from typing import Any, Literal

import numpy as np
import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.geometry import RayBundleGeometry, RigidTransform
from dpt.material_projection import material_projection_vjp, prepare_material_projection
from dpt.projection import ProjectionSpec, prepare_projection, projection_vjp
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")
data_kernels: Any = importlib.import_module("dpt.kernels.material_reconstruction")

Integration = Literal["midpoint", "cell_gauss"]
Execution = Literal["serial", "fused"]


def _setup(
    device: str, integration: Integration, execution: Execution, seed: float
) -> tuple[Any, ...]:
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((0.0, 0.0, -0.5),), ((0.0, 0.0, 0.5),))
    fields = wp.array([[0.125], [0.25]], dtype=wp.float32, device=device)
    paths = wp.empty((2, 1), dtype=wp.float64, device=device)
    seeds = wp.array([[seed], [-0.5 * seed]], dtype=wp.float64, device=device)
    gradient = wp.zeros((2, 1), dtype=wp.float64, device=device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=device)
    workspace = prepare_material_projection(
        grid,
        rays,
        ProjectionSpec(
            3, precision="float64", integration=integration, active_pose=False, active_volume=True
        ),
        materials=2,
        fields=fields,
        out_paths=paths,
        cotangent_paths=seeds,
        out_fields=gradient,
        volume_gradient_precision="float64",
        _float32_volume_increments=True,
        channel_execution=execution,
        device=device,
    )
    return workspace, pose, gradient, seeds


def _rounded_ray_increment(seed: float, integration: Integration) -> float:
    # The unit ray has three midpoint increments or two half-cell Gauss sums.
    if integration == "midpoint":
        return 3.0 * float(np.float32(seed / 3.0))
    return float(np.float32(seed))


def _narrow(workspace: Any, gradient: Any) -> Any:
    context = workspace.context
    with context.scope():
        # The solver starts a zeroed public-gradient epoch. Narrowing also
        # checks that destination so an earlier overflow cannot be hidden.
        destination = wp.zeros(gradient.size, dtype=wp.float32, device=context.device)
        wp.launch(
            data_kernels.narrow_data_gradient,
            dim=gradient.size,
            inputs=[workspace.grad_fields, destination, workspace.projection._status],
            stream=context.stream,
            device=context.device,
            record_tape=False,
        )
    return destination


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("execution", ["serial", "fused"])
def test_material_fp64_sum_retains_rounded_float32_increments(
    integration: Integration, execution: Execution, warp_device: str
) -> None:
    seed = 1.00000009
    workspace, pose, gradient, seeds = _setup(warp_device, integration, execution, seed)
    original_seeds = seeds.numpy().tobytes()
    expected = [
        _rounded_ray_increment(seed, integration),
        _rounded_ray_increment(-0.5 * seed, integration),
    ]
    assert expected[0] != seed
    assert gradient.dtype == wp.float64
    assert workspace.gradient_views[1].ptr - workspace.grad_fields.ptr == 8

    gradient.fill_(123.0)
    material_projection_vjp(pose, workspace=workspace)
    assert gradient.numpy().reshape(-1).tolist() == expected

    # At this offset FP32 atomics lose the small terms, while FP64 sums retain them.
    offset = float(2**24)
    gradient.fill_(offset)
    for _ in range(2):
        material_projection_vjp(
            pose,
            workspace=workspace,
            validate=False,
            accumulate=True,
            check_volume_gradient=False,
        )
    workspace.check_status()
    total = [offset + 2.0 * value for value in expected]
    assert gradient.numpy().reshape(-1).tolist() == total
    narrowed = _narrow(workspace, gradient)
    workspace.check_status()
    assert narrowed.numpy().tolist() == [float(np.float32(value)) for value in total]
    assert seeds.numpy().tobytes() == original_seeds


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("execution", ["serial", "fused"])
def test_material_fp64_final_narrow_checks_sum_range(
    integration: Integration, execution: Execution, warp_device: str
) -> None:
    workspace, pose, gradient, seeds = _setup(warp_device, integration, execution, 3e38)
    seeds.fill_(3e38)
    for _ in range(2):
        material_projection_vjp(
            pose,
            workspace=workspace,
            validate=False,
            accumulate=True,
            check_volume_gradient=False,
        )
    workspace.check_status()
    values = gradient.numpy().reshape(-1)
    assert all(math.isfinite(float(value)) and value > np.finfo(np.float32).max for value in values)

    narrowed = _narrow(workspace, gradient)
    assert not np.isfinite(narrowed.numpy()).any()
    with pytest.raises(NumericalError):
        workspace.check_status()


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("execution", ["serial", "fused"])
def test_material_float32_increment_overflow_survives_opposite_covectors(
    integration: Integration, execution: Execution, warp_device: str
) -> None:
    workspace, pose, _, seeds = _setup(warp_device, integration, execution, 2e39)
    for value in (2e39, -2e39):
        seeds.fill_(value)
        original_seeds = seeds.numpy().tobytes()
        material_projection_vjp(
            pose,
            workspace=workspace,
            validate=False,
            accumulate=True,
            check_volume_gradient=False,
        )
        assert seeds.numpy().tobytes() == original_seeds
    # No final scan or narrowing has run, so this diagnostic belongs to an increment.
    assert int(workspace.projection._status.numpy()[0]) & 2
    for _ in range(2):
        with pytest.raises(NumericalError):
            workspace.check_status()


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_public_fp64_projection_vjp_keeps_wide_increment_domain(
    integration: Integration, warp_device: str
) -> None:
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((0.0, 0.0, -0.5),), ((0.0, 0.0, 0.5),))
    workspace = prepare_projection(
        grid,
        rays,
        ProjectionSpec(
            3, precision="float64", integration=integration, active_pose=False, active_volume=True
        ),
        device=warp_device,
    )
    fields = wp.array([0.125], dtype=wp.float32, device=warp_device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    seed = wp.array([2e39], dtype=wp.float64, device=warp_device)
    gradient = wp.empty(1, dtype=wp.float64, device=warp_device)
    original_seed = seed.numpy().tobytes()
    projection_vjp(
        fields,
        pose,
        cotangent=seed,
        workspace=workspace,
        out_mu=gradient,
        volume_gradient_precision="float64",
    )
    actual = float(gradient.numpy()[0])
    assert math.isfinite(actual) and actual > np.finfo(np.float32).max
    assert actual == pytest.approx(2e39, rel=2e-15)
    assert seed.numpy().tobytes() == original_seed


@pytest.mark.parametrize("execution", ["serial", "fused"])
def test_material_fp64_vjp_rejects_rebound_gradient_views_before_writes(
    execution: Execution, warp_device: str
) -> None:
    workspace, pose, gradient, seeds = _setup(warp_device, "midpoint", execution, 1.0)
    gradient.fill_(123.0)
    original_gradient, original_seeds = gradient.numpy().tobytes(), seeds.numpy().tobytes()
    workspace.gradient_views = tuple(reversed(workspace.gradient_views))
    with pytest.raises(ContractError, match="prepared destination"):
        material_projection_vjp(pose, workspace=workspace)
    assert gradient.numpy().tobytes() == original_gradient
    assert seeds.numpy().tobytes() == original_seeds


@pytest.mark.parametrize("declared_precision", ["float32", "float64"])
def test_material_gradient_destination_requires_declared_dtype(
    declared_precision: Literal["float32", "float64"], warp_device: str
) -> None:
    workspace, _, _, seeds = _setup(warp_device, "midpoint", "serial", 1.0)
    wrong_dtype = wp.float64 if declared_precision == "float32" else wp.float32
    destination = wp.full((2, 1), 123.0, dtype=wrong_dtype, device=warp_device)
    original_destination = destination.numpy().tobytes()
    with pytest.raises(ContractError):
        prepare_material_projection(
            workspace.projection.grid,
            workspace.projection.geometry,
            workspace.projection.spec,
            materials=2,
            fields=workspace.fields,
            out_paths=workspace.paths,
            cotangent_paths=seeds,
            out_fields=destination,
            volume_gradient_precision=declared_precision,
            device=warp_device,
        )
    assert destination.numpy().tobytes() == original_destination
