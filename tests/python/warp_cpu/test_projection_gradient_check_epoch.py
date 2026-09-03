"""Explicit accumulation epochs retain overflow guards with one final scan."""

import importlib
import math
from typing import Any, Literal, cast

import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.geometry import RayBundleGeometry, RigidTransform
from dpt.material_projection import material_projection_vjp, prepare_material_projection
from dpt.projection import ProjectionSpec, prepare_projection, projection_vjp
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


def _fixture(device: str, integration: Any, precision: str) -> tuple[Any, ...]:
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((0.0, 0.0, -0.5),), ((0.0, 0.0, 0.5),))
    workspace = prepare_projection(
        grid,
        rays,
        ProjectionSpec(
            17, precision="float64", integration=integration, active_pose=False, active_volume=True
        ),
        device=device,
    )
    field = wp.array([0.125], dtype=wp.float32, device=device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=device)
    gradient = wp.zeros(1, dtype=getattr(wp, precision), device=device)
    return workspace, field, pose, gradient


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_accumulation_epoch_enqueues_one_scan_and_preserves_default(
    integration: Any,
    precision: Literal["float32", "float64"],
    warp_device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, field, pose, gradient = _fixture(warp_device, integration, precision)
    scan = workspace._kernels.get_validate_finite(precision == "float64")
    calls: list[Any] = []
    launch = wp.launch

    def counted(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        if kernel is scan:
            calls.append(kernel)
        return launch(kernel, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(wp, "launch", counted)
        for value in (1.5, -0.25, 2.0):
            seed = wp.array([value], dtype=wp.float64, device=warp_device)
            projection_vjp(
                field,
                pose,
                cotangent=seed,
                workspace=workspace,
                out_mu=gradient,
                volume_gradient_precision=precision,
                validate=False,
                accumulate=True,
                check_volume_gradient=False,
            )
        assert not calls
        workspace.enqueue_volume_gradient_check(gradient, volume_gradient_precision=precision)
        workspace.check_status()
        assert len(calls) == 1
        assert float(gradient.numpy()[0]) == pytest.approx(
            3.25, abs=2e-6 if precision == "float32" else 3e-14
        )
        projection_vjp(
            field,
            pose,
            cotangent=seed,
            workspace=workspace,
            out_mu=gradient,
            volume_gradient_precision=precision,
            validate=False,
            accumulate=True,
        )
        assert len(calls) == 2


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_final_scan_catches_atomic_overflow_after_opposite_finite_increments(
    integration: Any, precision: Literal["float32", "float64"], warp_device: str
) -> None:
    workspace, field, pose, gradient = _fixture(warp_device, integration, precision)
    large = 3e38 if precision == "float32" else 1e308
    for value in (large, large, -large):
        seed = wp.array([value], dtype=wp.float64, device=warp_device)
        projection_vjp(
            field,
            pose,
            cotangent=seed,
            workspace=workspace,
            out_mu=gradient,
            volume_gradient_precision=precision,
            validate=False,
            accumulate=True,
            check_volume_gradient=False,
        )
    # Opposite finite summands cannot recover an already infinite accumulator.
    assert not math.isfinite(float(gradient.numpy()[0]))
    workspace.enqueue_volume_gradient_check(gradient, volume_gradient_precision=precision)
    for _ in range(2):
        with pytest.raises(NumericalError):
            workspace.check_status()
    with pytest.raises(NumericalError):
        workspace.consume_status()
    workspace.check_status()


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_unchecked_default_still_catches_accumulator_overflow(
    precision: Literal["float32", "float64"], warp_device: str
) -> None:
    workspace, field, pose, gradient = _fixture(warp_device, "midpoint", precision)
    seed = wp.array(
        [3e38 if precision == "float32" else 1e308], dtype=wp.float64, device=warp_device
    )
    for _ in range(2):
        projection_vjp(
            field,
            pose,
            cotangent=seed,
            workspace=workspace,
            out_mu=gradient,
            volume_gradient_precision=precision,
            validate=False,
            accumulate=True,
        )
    with pytest.raises(NumericalError):
        workspace.check_status()


def test_deferred_checks_require_an_unchecked_accumulation_epoch(warp_device: str) -> None:
    workspace, field, pose, gradient = _fixture(warp_device, "midpoint", "float32")
    seed = wp.array([1.0], dtype=wp.float64, device=warp_device)
    gradient.fill_(19.0)
    invalid_options: tuple[dict[str, Any], ...] = (
        {"validate": True, "accumulate": True, "check_volume_gradient": False},
        {"validate": False, "accumulate": False, "check_volume_gradient": False},
        {"validate": False, "accumulate": True, "check_volume_gradient": False, "_from_tape": True},
        {"check_volume_gradient": cast(Any, 0)},
    )
    for options in invalid_options:
        with pytest.raises(ContractError):
            projection_vjp(
                field, pose, cotangent=seed, workspace=workspace, out_mu=gradient, **options
            )
        assert float(gradient.numpy()[0]) == 19.0


@pytest.mark.parametrize("execution", ["serial", "fused"])
def test_material_epoch_defers_scans_and_detects_accumulated_overflow(
    execution: Any, warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((0.0, 0.0, -0.5),), ((0.0, 0.0, 0.5),))
    fields = wp.array([[0.125], [0.25]], dtype=wp.float32, device=warp_device)
    paths = wp.empty((2, 1), dtype=wp.float64, device=warp_device)
    seeds = wp.array([[1.5], [-0.25]], dtype=wp.float64, device=warp_device)
    gradient = wp.zeros((2, 1), dtype=wp.float32, device=warp_device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    workspace = prepare_material_projection(
        grid,
        rays,
        ProjectionSpec(17, precision="float64", active_pose=False, active_volume=True),
        materials=2,
        fields=fields,
        out_paths=paths,
        cotangent_paths=seeds,
        out_fields=gradient,
        channel_execution=execution,
        device=warp_device,
    )
    scans = (
        workspace.projection._kernels.get_validate_finite(False),
        workspace.projection._kernels.validate_finite,
    )
    calls: list[Any] = []
    launch = wp.launch

    def counted(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        if any(kernel is scan for scan in scans):
            calls.append(kernel)
        return launch(kernel, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(wp, "launch", counted)
        for _ in range(3):
            material_projection_vjp(
                pose,
                workspace=workspace,
                validate=False,
                accumulate=True,
                check_volume_gradient=False,
            )
        assert not calls
        for destination in workspace.gradient_views:
            workspace.projection.enqueue_volume_gradient_check(destination)
        assert len(calls) == 2
    workspace.check_status()
    assert gradient.numpy().reshape(-1).tolist() == pytest.approx([4.5, -0.75], abs=3e-6)
    gradient.zero_()
    seeds.fill_(3e38)
    for _ in range(2):
        material_projection_vjp(
            pose, workspace=workspace, validate=False, accumulate=True, check_volume_gradient=False
        )
    for destination in workspace.gradient_views:
        workspace.projection.enqueue_volume_gradient_check(destination)
    with pytest.raises(NumericalError):
        workspace.check_status()
