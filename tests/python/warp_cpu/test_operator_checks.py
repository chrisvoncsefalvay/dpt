"""Operator diagnostics agree with independent dense and ray-integral oracles."""

from collections.abc import Iterator
from contextlib import nullcontext
from typing import Literal

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.operator_checks import (
    LinearOperator,
    check_adjoint,
    estimate_operator_norm,
    projection_linear_operator,
)
from dpt.projection import ProjectionSpec, prepare_projection
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec


@pytest.fixture(autouse=True)
def _kernel_context(warp_device: str) -> Iterator[None]:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def test_adjoint_check_detects_an_unmatched_transpose() -> None:
    matrix = np.array([[1.0, 2.0, -3.0], [-4.0, 5.0, 6.0]])
    operator = LinearOperator((3,), (2,), lambda x: matrix @ x, lambda y: matrix.T @ y)
    result = check_adjoint(operator, tolerance=1e-12)
    assert result.passed
    assert result.relative_error < 1e-15
    wrong = LinearOperator((3,), (2,), operator.forward, lambda y: 1.2 * matrix.T @ y)
    assert not check_adjoint(wrong, tolerance=1e-4).passed


def test_power_iteration_agrees_with_independent_singular_values() -> None:
    matrix = np.array([[1.0, 2.0, -3.0], [-4.0, 5.0, 6.0]])
    operator = LinearOperator((3,), (2,), lambda x: matrix @ x, lambda y: matrix.T @ y)
    norm = estimate_operator_norm(operator, iterations=100, tolerance=1e-10)
    assert norm.converged
    assert norm.value == pytest.approx(np.linalg.svd(matrix, compute_uv=False)[0], rel=1e-10)
    short = estimate_operator_norm(operator, iterations=1, tolerance=1e-14)
    assert not short.converged
    assert short.iterations == 1


@pytest.mark.parametrize("gain", [1e-300, 1e-200, 1e200, 1e300])
def test_diagnostics_preserve_scale_for_tiny_and_large_finite_operators(gain: float) -> None:
    matrix = np.array([[1.0, 2.0, -3.0], [-4.0, 5.0, 6.0]])
    operator = LinearOperator(
        (3,), (2,), lambda x: gain * (matrix @ x), lambda y: gain * (matrix.T @ y)
    )
    assert check_adjoint(operator, tolerance=1e-12).passed
    wrong = LinearOperator((3,), (2,), operator.forward, lambda y: 2.0 * gain * (matrix.T @ y))
    result = check_adjoint(wrong)
    assert not result.passed
    assert result.relative_error > 1e-3
    norm = estimate_operator_norm(operator, tolerance=1e-10)
    assert norm.converged
    assert norm.value / gain == pytest.approx(np.linalg.svd(matrix, compute_uv=False)[0], rel=1e-10)
    assert np.isfinite(norm.residual)


def test_adjoint_check_consumes_outputs_before_shared_host_scratch_is_reused() -> None:
    scratch = np.empty(3)

    def forward(x: np.ndarray) -> np.ndarray:
        np.copyto(scratch, x)
        return scratch

    def adjoint(y: np.ndarray) -> np.ndarray:
        np.copyto(scratch, 2.0 * y)
        return scratch

    result = check_adjoint(LinearOperator((3,), (3,), forward, adjoint))
    assert not result.passed
    assert result.adjoint_inner_product == pytest.approx(2.0 * result.forward_inner_product)


def test_nonrepresentable_diagnostic_scalars_raise_instead_of_false_success() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(4), rng.standard_normal(4)
    matrix = np.outer(np.sign(y), x / np.linalg.norm(x)) * np.finfo(np.float64).max
    operator = LinearOperator((4,), (4,), lambda x: matrix @ x, lambda y: np.zeros(4))
    with pytest.raises(ContractError, match="finite FP64 diagnostic range"):
        check_adjoint(operator)
    maximal = LinearOperator(
        (1,),
        (3,),
        lambda x: np.full(3, np.finfo(np.float64).max * x[0]),
        lambda y: np.array([np.finfo(np.float64).max * np.sum(y)]),
    )
    with pytest.raises(ContractError, match="finite FP64 diagnostic range"):
        estimate_operator_norm(maximal)


def test_zero_operator_has_zero_error_and_norm() -> None:
    operator = LinearOperator((3,), (2,), lambda x: np.zeros(2), lambda y: np.zeros(3))
    result = check_adjoint(operator)
    assert result.passed and result.relative_error == 0.0
    norm = estimate_operator_norm(operator)
    assert norm.converged and norm.value == 0.0 and norm.residual == 0.0


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0.0, -1.0])
def test_diagnostic_tolerance_is_finite_positive(invalid: float) -> None:
    operator = LinearOperator((1,), (1,), lambda x: x, lambda x: x)
    with pytest.raises(ContractError, match="tolerance"):
        check_adjoint(operator, tolerance=invalid)
    with pytest.raises(ContractError, match="tolerance"):
        estimate_operator_norm(operator, tolerance=invalid)


@pytest.mark.parametrize("shape", [(), (0,), (True,), (1.5,)])
def test_operator_shape_contract(shape: tuple[int, ...]) -> None:
    with pytest.raises(ContractError):
        LinearOperator(shape, (1,), lambda x: x, lambda x: x)


@pytest.mark.parametrize("output", [np.array([np.nan]), np.ones(2), np.array([1j])])
def test_invalid_operator_output_fails(output: np.ndarray) -> None:
    operator = LinearOperator((1,), (1,), lambda x: output, lambda x: x)
    with pytest.raises(ContractError, match="forward output"):
        check_adjoint(operator)


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_actual_projection_adjoint_and_norm_match_independent_matrix(
    integration: Literal["midpoint", "cell_gauss"], warp_device: str
) -> None:
    grid = GridSpec((2, 2, 3), (1.1, 0.8, 0.9), (-1.1, -0.4, -0.45))
    detector = DetectorGeometry(
        (-6.0, 0.03, 0.07),
        (6.0, -0.3, -0.3),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.2, 0.25),
        (2, 3),
    )
    spec = ProjectionSpec(
        31,
        active_pose=False,
        active_volume=True,
        precision="float64",
        integration=integration,
        field_domain="signed_linear",  # type: ignore[arg-type]
    )
    workspace = prepare_projection(grid, detector, spec, device=warp_device)
    operator = projection_linear_operator(workspace, RigidTransform())
    columns = [
        [
            integrate_sampled_field(
                [float(index == voxel) for index in range(grid.voxels)],
                grid,
                detector,
                RigidTransform(),
                row,
                column,
                midpoint_samples=31 if integration == "midpoint" else None,
            )
            for row in range(2)
            for column in range(3)
        ]
        for voxel in range(grid.voxels)
    ]
    matrix = np.asarray(columns).T
    assert check_adjoint(operator, tolerance=1e-6).passed
    cotangent = np.arange(1.0, detector.pixels + 1.0)
    assert operator.adjoint(cotangent) == pytest.approx(matrix.T @ cotangent, rel=1e-12, abs=1e-12)
    norm = estimate_operator_norm(operator, iterations=80, tolerance=1e-6)
    assert norm.converged
    assert norm.value == pytest.approx(np.linalg.svd(matrix, compute_uv=False)[0], rel=2e-6)


def test_physical_workspace_is_rejected_for_signed_linear_diagnostics(warp_device: str) -> None:
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    detector = DetectorGeometry(
        (-2.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0),
        (1, 1),
    )
    workspace = prepare_projection(grid, detector, ProjectionSpec(1), device=warp_device)
    with pytest.raises(ContractError, match="signed_linear"):
        projection_linear_operator(workspace, RigidTransform())
