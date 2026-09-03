"""Independent physical-spacing, subgradient, proximal and dual-gap witnesses."""

import importlib
import math
from contextlib import nullcontext
from typing import Any, Literal, cast

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError, NumericalError, TrialDomainError
from dpt.regularization import (
    RegularizationProxPolicy,
    RegularizationSpec,
    prepare_regularization,
    regularization_prox,
    regularization_value_and_gradient,
)
from dpt.volumes import GridSpec
from helpers.regularization_oracles import (
    anisotropic_epigraph_optimum,
    certificate,
    decimal_tv_value_and_gradient,
    difference_matrix,
    penalty,
)

wp: Any = importlib.import_module("warp")


def setup(
    device: str,
    *,
    norm: Literal["isotropic", "anisotropic"] = "isotropic",
    precision: Literal["float32", "float64"] = "float64",
    tv: float = 0.3,
    l1: float = 0.12,
    masked: bool = False,
    shape: tuple[int, int, int] = (2, 2, 3),
) -> tuple[Any, Any, Any, np.ndarray, np.ndarray]:
    grid = GridSpec(shape, (0.7, 1.3, 2.1))
    values = np.random.default_rng(97113).normal(size=grid.voxels) + 0.31
    mask = np.ones(grid.voxels, dtype=bool)
    if masked:
        mask[1::4] = False
    support = wp.array(mask.astype(np.uint8), dtype=wp.uint8, device=device)
    workspace = prepare_regularization(
        grid, RegularizationSpec(tv, l1, norm, precision), support=support, device=device
    )
    source = wp.array(values.astype(precision), dtype=getattr(wp, precision), device=device)
    output = wp.empty(grid.voxels, dtype=getattr(wp, precision), device=device)
    return workspace, source, output, values.astype(precision).astype(float), mask


@pytest.mark.parametrize("norm", ["isotropic", "anisotropic"])
@pytest.mark.parametrize("masked", [False, True])
def test_value_and_active_coordinate_gradient_against_independent_dense_operator(
    warp_device: str, norm: Literal["isotropic", "anisotropic"], masked: bool
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace, source, output, values, mask = setup(warp_device, norm=norm, masked=masked)
        values[~mask] = 0
        source.assign(values)
        matrix = difference_matrix(workspace.grid.shape, workspace.grid.spacing_mm)
        value = regularization_value_and_gradient(source, out_gradient=output, workspace=workspace)
        assert value == pytest.approx(
            penalty(values, matrix, 0.3, 0.12, norm == "isotropic"), rel=2e-13
        )
        expected = np.zeros_like(values)
        for index in np.flatnonzero(mask):
            plus, minus = values.copy(), values.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            expected[index] = (
                penalty(plus, matrix, 0.3, 0.12, norm == "isotropic")
                - penalty(minus, matrix, 0.3, 0.12, norm == "isotropic")
            ) / 2e-6
        np.testing.assert_allclose(output.numpy(), expected, rtol=2e-7, atol=4e-9)
        assert np.all(output.numpy()[~mask] == 0)


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("count", [1, 127, 129, 323])
def test_l1_support_prox_is_exact_soft_threshold_and_owned_mask(
    warp_device: str, precision: Literal["float32", "float64"], count: int
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace, source, output, values, mask = setup(
            warp_device, precision=precision, tv=0.0, l1=0.3, masked=True, shape=(1, 1, count)
        )
        result = regularization_prox(source, step=0.7, out=output, workspace=workspace)
        expected = np.sign(values) * np.maximum(np.abs(values) - 0.21, 0)
        expected[~mask] = 0
        np.testing.assert_array_equal(output.numpy(), expected.astype(precision))
        assert result.reason == "closed_form" and result.iterations == 0 and result.converged
        assert result.gap <= 1e-12 * max(1.0, result.primal)
        np.testing.assert_array_equal(source.numpy(), values.astype(precision))
        assert workspace.scratch_bytes < workspace.grid.voxels * 128 + 8192


@pytest.mark.parametrize("norm", ["isotropic", "anisotropic"])
@pytest.mark.parametrize("spacing", [0.5, 2.0])
def test_two_sample_tv_l1_solution_and_distance_certificate(
    warp_device: str, norm: Literal["isotropic", "anisotropic"], spacing: float
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace = prepare_regularization(
            GridSpec((1, 1, 2), (spacing, 1.0, 1.0)),
            RegularizationSpec(tv_weight=0.2, l1_weight=0.1, tv_norm=norm),
            device=warp_device,
        )
        source = wp.array([1.0, -1.0], dtype=wp.float64, device=warp_device)
        output = wp.empty(2, dtype=wp.float64, device=warp_device)
        result = regularization_prox(
            source,
            step=1.0,
            out=output,
            workspace=workspace,
            policy=RegularizationProxPolicy(absolute_gap=1e-13, relative_gap=1e-12),
        )
        optimum = np.asarray([1 - 0.2 / spacing - 0.1, -1 + 0.2 / spacing + 0.1])
        assert result.converged and result.gap < 2e-12
        roundoff = 128 * math.ulp(1.0) * max(abs(result.primal), abs(result.dual))
        assert (
            np.linalg.norm(output.numpy() - optimum)
            <= math.sqrt(2 * (result.gap + roundoff)) + 1e-12
        )
        matrix = difference_matrix(workspace.grid.shape, workspace.grid.spacing_mm)
        p, d = certificate(
            output.numpy(),
            source.numpy(),
            workspace._dual.numpy(),
            matrix,
            np.ones(2, dtype=bool),
            0.2,
            0.1,
            norm == "isotropic",
        )
        assert result.primal == pytest.approx(p, rel=1e-13, abs=1e-13)
        assert result.dual == pytest.approx(d, rel=1e-13, abs=1e-13)


@pytest.mark.parametrize("norm", ["isotropic", "anisotropic"])
@pytest.mark.parametrize("masked", [False, True])
def test_composite_proximal_dual_gap_is_independent_feasible_certificate(
    warp_device: str,
    norm: Literal["isotropic", "anisotropic"],
    masked: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace, source, output, values, mask = setup(warp_device, norm=norm, masked=masked)
        matrix = difference_matrix(workspace.grid.shape, workspace.grid.spacing_mm)
        original_numpy = wp.array.numpy
        with monkeypatch.context() as guard:

            def prevent_device_download(array: Any) -> Any:
                if array.device.is_cuda:
                    pytest.fail("proximal solver downloaded a full device field")
                return original_numpy(array)

            guard.setattr(wp.array, "numpy", prevent_device_download)
            result = regularization_prox(
                source,
                step=0.8,
                out=output,
                workspace=workspace,
                policy=RegularizationProxPolicy(
                    max_iterations=4000, absolute_gap=1e-11, relative_gap=1e-10
                ),
            )
        actual, dual = output.numpy(), workspace._dual.numpy()
        assert result.converged
        assert np.all(actual[~mask] == 0)
        norms = (
            np.linalg.norm(dual, axis=1) if norm == "isotropic" else np.max(np.abs(dual), axis=1)
        )
        assert np.max(norms) <= 0.24 * (1 + 2e-14)
        p, d = certificate(actual, values, dual, matrix, mask, 0.24, 0.096, norm == "isotropic")
        assert p >= d - 2e-13
        assert result.primal == pytest.approx(p, rel=2e-13, abs=2e-13)
        assert result.dual == pytest.approx(d, rel=2e-13, abs=2e-13)
        assert result.gap == pytest.approx(p - d, abs=1e-12)
        if norm == "anisotropic":
            optimum = anisotropic_epigraph_optimum(values, matrix, mask, 0.24, 0.096)
            assert np.linalg.norm(actual - optimum) <= math.sqrt(2 * result.gap) + 2e-6


def test_zero_penalty_support_and_flat_tv_zero_subgradient(warp_device: str) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace, source, output, values, mask = setup(warp_device, tv=0.0, l1=0.0, masked=True)
        result = regularization_prox(source, step=1.0, out=output, workspace=workspace)
        values[~mask] = 0
        np.testing.assert_array_equal(output.numpy(), values)
        assert result.converged and result.gap == 0
        flat = prepare_regularization(
            workspace.grid, RegularizationSpec(tv_weight=1.0), device=warp_device
        )
        source.assign(np.ones(len(values)))
        assert regularization_value_and_gradient(source, out_gradient=output, workspace=flat) == 0
        assert np.all(output.numpy() == 0)


def test_iteration_budget_is_feasible_but_not_converged(warp_device: str) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace, source, output, _, mask = setup(warp_device, tv=1.7, masked=True)
        result = regularization_prox(
            source,
            step=1.0,
            out=output,
            workspace=workspace,
            policy=RegularizationProxPolicy(
                max_iterations=1, check_interval=10, absolute_gap=0, relative_gap=0
            ),
        )
        assert result.reason == "iteration_budget" and not result.converged
        assert result.iterations == 1 and result.gap > 0
        assert np.all(output.numpy()[~mask] == 0)


def test_support_dtype_alias_status_tape_and_budget_contracts(warp_device: str) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace, source, output, values, mask = setup(warp_device, masked=True)
        with pytest.raises(TrialDomainError, match="support"):
            regularization_value_and_gradient(source, out_gradient=output, workspace=workspace)
        values[~mask] = 0
        source.assign(values)
        assert math.isfinite(
            regularization_value_and_gradient(source, out_gradient=output, workspace=workspace)
        )
        with pytest.raises(ContractError, match="overlap"):
            regularization_prox(source, step=1.0, out=source, workspace=workspace)
        with pytest.raises(ContractError, match="dtype"):
            regularization_prox(
                source,
                step=1.0,
                out=wp.empty(len(values), dtype=wp.float32, device=warp_device),
                workspace=workspace,
            )
        with wp.Tape(), pytest.raises(ContractError, match="ambient"):
            regularization_prox(source, step=1.0, out=output, workspace=workspace)
        with pytest.raises(ContractError, match="max_bytes"):
            prepare_regularization(workspace.grid, workspace.spec, device=warp_device, max_bytes=1)
        with pytest.raises(ContractError, match="binary"):
            prepare_regularization(
                workspace.grid,
                workspace.spec,
                support=wp.full(len(values), 2, dtype=wp.uint8, device=warp_device),
                device=warp_device,
            )
        source.assign(np.full(len(values), math.nan))
        with pytest.raises(ContractError, match="finite"):
            regularization_prox(source, step=1.0, out=output, workspace=workspace)
        source.assign(values)
        assert regularization_prox(source, step=1.0, out=output, workspace=workspace).converged


def test_pure_l1_huge_values_do_not_evaluate_disabled_tv_or_subtract_huge_squares(
    warp_device: str,
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace = prepare_regularization(
            GridSpec((1, 1, 2), (1.0, 1.0, 1.0)),
            RegularizationSpec(l1_weight=1e-200),
            device=warp_device,
        )
        source = wp.array([1e200, -1e200], dtype=wp.float64, device=warp_device)
        output = wp.empty(2, dtype=wp.float64, device=warp_device)
        assert regularization_value_and_gradient(
            source, out_gradient=output, workspace=workspace
        ) == pytest.approx(2.0)
        result = regularization_prox(source, step=1.0, out=output, workspace=workspace)
        assert (
            result.converged
            and result.primal == pytest.approx(2.0)
            and result.dual == pytest.approx(2.0)
        )
        with pytest.raises(NumericalError):
            regularization_prox(
                source,
                step=1e308,
                out=output,
                workspace=prepare_regularization(
                    workspace.grid, RegularizationSpec(l1_weight=1e308), device=warp_device
                ),
            )


@pytest.mark.parametrize("norm", ["isotropic", "anisotropic"])
def test_fp32_gradient_and_final_proximal_cast_match_two_sample_oracle(
    warp_device: str, norm: Literal["isotropic", "anisotropic"]
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace = prepare_regularization(
            GridSpec((1, 1, 2), (1.0, 1.0, 1.0)),
            RegularizationSpec(tv_weight=0.2, l1_weight=0.1, tv_norm=norm, precision="float32"),
            device=warp_device,
        )
        source = wp.array([1.0, -1.0], dtype=wp.float32, device=warp_device)
        output = wp.empty(2, dtype=wp.float32, device=warp_device)
        assert regularization_value_and_gradient(
            source, out_gradient=output, workspace=workspace
        ) == pytest.approx(0.6)
        np.testing.assert_array_equal(output.numpy(), np.asarray([0.3, -0.3], dtype=np.float32))
        result = regularization_prox(
            source,
            step=1.0,
            out=output,
            workspace=workspace,
            policy=RegularizationProxPolicy(absolute_gap=1e-14, relative_gap=1e-14),
        )
        assert result.converged
        np.testing.assert_allclose(output.numpy(), [0.7, -0.7], rtol=0, atol=7e-8)


@pytest.mark.parametrize("norm", ["isotropic", "anisotropic"])
def test_tv_certificate_spans_multiple_reduction_blocks(
    warp_device: str, norm: Literal["isotropic", "anisotropic"]
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace, source, output, values, mask = setup(warp_device, norm=norm, shape=(1, 1, 129))
        result = regularization_prox(
            source,
            step=0.8,
            out=output,
            workspace=workspace,
            policy=RegularizationProxPolicy(max_iterations=4000),
        )
        assert result.converged
        matrix = difference_matrix(workspace.grid.shape, workspace.grid.spacing_mm)
        p, d = certificate(
            output.numpy(),
            values,
            workspace._dual.numpy(),
            matrix,
            mask,
            0.24,
            0.096,
            norm == "isotropic",
        )
        assert result.primal == pytest.approx(p, rel=2e-13)
        assert result.dual == pytest.approx(d, rel=2e-13)
        assert result.gap == pytest.approx(p - d, abs=1e-11)


def test_owned_support_and_pending_status_stream_and_lock_lifetime(warp_device: str) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        grid = GridSpec((1, 1, 3), (1.0, 1.0, 1.0))
        support = wp.array([1, 0, 1], dtype=wp.uint8, device=warp_device)
        workspace = prepare_regularization(
            grid, RegularizationSpec(l1_weight=0.1), support=support, device=warp_device
        )
        support.assign(np.asarray([0, 1, 0], dtype=np.uint8))
        source = wp.array([1.0, 2.0, 3.0], dtype=wp.float64, device=warp_device)
        output = wp.empty(3, dtype=wp.float64, device=warp_device)
        regularization_prox(source, step=1.0, out=output, workspace=workspace)
        np.testing.assert_allclose(output.numpy(), [0.9, 0.0, 2.9], rtol=0, atol=1e-15)
        accepted = output.numpy().copy()
        workspace._status.assign(np.asarray([2], dtype=np.int32))
        with pytest.raises(NumericalError):
            regularization_prox(source, step=1.0, out=output, workspace=workspace)
        np.testing.assert_array_equal(output.numpy(), accepted)
        assert regularization_prox(source, step=1.0, out=output, workspace=workspace).converged
        assert workspace._lock.acquire(blocking=False)
        try:
            with pytest.raises(ContractError, match="concurrently"):
                regularization_prox(source, step=1.0, out=output, workspace=workspace)
        finally:
            workspace._lock.release()
        foreign = wp.Stream(warp_device) if warp_device != "cpu" else object()
        with pytest.raises(ContractError, match="stream"):
            regularization_prox(
                source, step=1.0, out=output, workspace=workspace, stream=cast(Any, foreign)
            )


def test_isotropic_projection_does_not_underflow_ratio_before_representable_components(
    warp_device: str,
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace = prepare_regularization(
            GridSpec((1, 1, 2), (1.0, 1.0, 1.0)),
            RegularizationSpec(tv_weight=1e-200),
            device=warp_device,
        )
        source = wp.array([1e200, -1e200], dtype=wp.float64, device=warp_device)
        output = wp.empty(2, dtype=wp.float64, device=warp_device)
        result = regularization_prox(source, step=1.0, out=output, workspace=workspace)
        assert (
            result.converged
            and result.primal == pytest.approx(2.0)
            and result.dual == pytest.approx(2.0)
        )
        np.testing.assert_array_equal(output.numpy(), source.numpy())
        assert workspace._dual.numpy()[0, 0] == pytest.approx(-1e-200, rel=3e-14, abs=0)


@pytest.mark.parametrize("norm", ["isotropic", "anisotropic"])
def test_constant_near_maximum_field_converges_at_initial_certificate(
    warp_device: str, norm: Literal["isotropic", "anisotropic"]
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace = prepare_regularization(
            GridSpec((1, 1, 2), (1.0, 1.0, 1.0)),
            RegularizationSpec(tv_weight=1e-308, tv_norm=norm),
            device=warp_device,
        )
        source = wp.array([1e308, 1e308], dtype=wp.float64, device=warp_device)
        output = wp.empty(2, dtype=wp.float64, device=warp_device)
        result = regularization_prox(source, step=1.0, out=output, workspace=workspace)
        assert result.reason == "primal_dual_gap" and result.iterations == 0
        assert result.primal == result.dual == result.gap == 0.0
        np.testing.assert_array_equal(output.numpy(), source.numpy())


@pytest.mark.parametrize("norm", ["isotropic", "anisotropic"])
def test_weighted_tv_difference_and_iterated_extrapolation_preserve_extreme_range(
    warp_device: str, norm: Literal["isotropic", "anisotropic"]
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace = prepare_regularization(
            GridSpec((1, 1, 2), (1.0, 1.0, 1.0)),
            RegularizationSpec(tv_weight=1e-308, tv_norm=norm),
            device=warp_device,
        )
        source = wp.array([1e308, -1e308], dtype=wp.float64, device=warp_device)
        output = wp.empty(2, dtype=wp.float64, device=warp_device)
        # Weighted |x1-x0|=2 and its gradient remain representable although
        # subtracting the endpoints before scaling overflows binary64.
        value = regularization_value_and_gradient(source, out_gradient=output, workspace=workspace)
        assert value == pytest.approx(2.0, rel=2e-15)
        np.testing.assert_allclose(output.numpy(), [1e-308, -1e-308], rtol=2e-15, atol=0)
        result = regularization_prox(source, step=1.0, out=output, workspace=workspace)
        assert result.converged and result.iterations > 0
        assert result.primal == pytest.approx(2.0, rel=2e-15)
        assert result.dual == pytest.approx(2.0, rel=4e-15)
        assert 0 <= result.gap < 1e-13
        # The exact shift is 1e-308, far below one binary64 ULP of 1e308.
        np.testing.assert_array_equal(output.numpy(), source.numpy())


def test_anisotropic_axis_subgradient_survives_other_axis_overflow_and_primal_underflow(
    warp_device: str,
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace = prepare_regularization(
            GridSpec((1, 2, 3), (1e-307, 1.0, 1.0)),
            RegularizationSpec(tv_weight=1e-308, tv_norm="anisotropic"),
            device=warp_device,
        )
        small = 1e-100
        source = wp.array(
            [100.0, small, -100.0, 100.0, math.nextafter(small, math.inf), -100.0],
            dtype=wp.float64,
            device=warp_device,
        )
        output = wp.empty(6, dtype=wp.float64, device=warp_device)
        value = regularization_value_and_gradient(source, out_gradient=output, workspace=workspace)
        assert value == pytest.approx(40.0, rel=3e-15)
        # The x-axis terms cancel at the two central voxels. The tiny y-axis
        # primal rounds to zero, but its nonzero active subgradient is retained.
        np.testing.assert_allclose(output.numpy()[[1, 4]], [-1e-308, 1e-308], rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape,spacing,weight,values",
    [
        ((1, 2, 2), (1.0, 1e-100, 1.0), 1e-250, [0.0, 1.0, 1e-200, 1.0]),
        ((1, 2, 2), (1.0, 1e-307, 1.0), 1e-308, [0.0, 1.0, math.nextafter(0.0, 1.0), 1.0]),
        (
            (1, 2, 3),
            (1.0, 1e-50, 1.0),
            1e100,
            [1e100, 0.0, -1e100, 1e100, 1e-300, -1e100],
        ),
    ],
    ids=["weighted-tangent-underflow", "subnormal-input", "unit-direction-underflow"],
)
def test_isotropic_final_gradient_retains_range_across_normalisation_weight_and_spacing(
    warp_device: str,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    weight: float,
    values: list[float],
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace = prepare_regularization(
            GridSpec(shape, spacing), RegularizationSpec(tv_weight=weight), device=warp_device
        )
        source = wp.array(values, dtype=wp.float64, device=warp_device)
        output = wp.empty(len(values), dtype=wp.float64, device=warp_device)
        expected_value, expected_gradient = decimal_tv_value_and_gradient(
            np.asarray(values), shape, spacing, weight, True
        )
        value = regularization_value_and_gradient(source, out_gradient=output, workspace=workspace)
        assert value == pytest.approx(expected_value, rel=4e-14, abs=0)
        # The first witness has a near cancellation between two lambda-sized
        # rounded edges. Allow their ULP error, while preserving strict relative
        # checks on the rescued tiny derivatives in the other two witnesses.
        allowance = 8 * math.ulp(weight) if weight == 1e-250 else 0.0
        np.testing.assert_allclose(output.numpy(), expected_gradient, rtol=4e-14, atol=allowance)


@pytest.mark.parametrize("norm", ["isotropic", "anisotropic"])
def test_tv_norm_aggregates_components_before_subnormal_weight_rounding(
    warp_device: str, norm: Literal["isotropic", "anisotropic"]
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        shape, spacing = (1, 2, 2), (1.0, 1.0, 1.0)
        weight = math.nextafter(0.0, 1.0)
        values = np.asarray([0.0, 0.49, 0.49, 0.49])
        workspace = prepare_regularization(
            GridSpec(shape, spacing),
            RegularizationSpec(tv_weight=weight, tv_norm=norm),
            device=warp_device,
        )
        source = wp.array(values, dtype=wp.float64, device=warp_device)
        output = wp.empty(4, dtype=wp.float64, device=warp_device)
        expected, _ = decimal_tv_value_and_gradient(
            values, shape, spacing, weight, norm == "isotropic"
        )
        assert expected == weight
        value = regularization_value_and_gradient(source, out_gradient=output, workspace=workspace)
        assert value == expected
        result = regularization_prox(source, step=1.0, out=output, workspace=workspace)
        assert result.primal == expected


@pytest.mark.parametrize("which", ["reduction", "completion", "completion_host"])
@pytest.mark.parametrize("role", ["source", "output"])
def test_all_owned_reduction_and_completion_regions_reject_caller_aliases(
    warp_device: str, which: str, role: str
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace = prepare_regularization(
            GridSpec((1, 1, 2), (1.0, 1.0, 1.0)),
            RegularizationSpec(tv_weight=0.2, l1_weight=0.1),
            device=warp_device,
        )
        regions = {
            "reduction": workspace._reduction.partials[0],
            "completion": workspace._completion._payload,
            "completion_host": workspace._completion._readback._host,
        }
        alias = regions[which][:2]
        source = wp.array([1.0, -1.0], dtype=wp.float64, device=warp_device)
        output = wp.empty(2, dtype=wp.float64, device=warp_device)
        if role == "source":
            source = alias
        else:
            output = alias
        # Pinned host staging has a different device on CUDA; both paths reject
        # the caller region before any field evaluation or certificate write.
        with pytest.raises(ContractError, match=r"overlap|device"):
            regularization_value_and_gradient(source, out_gradient=output, workspace=workspace)
        with pytest.raises(ContractError, match=r"overlap|device"):
            regularization_prox(source, step=1.0, out=output, workspace=workspace)


@pytest.mark.parametrize("which", ["tv", "l1"])
def test_active_proximal_coefficients_reject_underflow_without_dropping_penalty(
    warp_device: str, which: str
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        spec = RegularizationSpec(
            tv_weight=1e-308 if which == "tv" else 0.0,
            l1_weight=1e-308 if which == "l1" else 0.0,
        )
        workspace = prepare_regularization(
            GridSpec((1, 1, 2), (1.0, 1.0, 1.0)), spec, device=warp_device
        )
        source = wp.array([1.0, -1.0], dtype=wp.float64, device=warp_device)
        output = wp.full(2, 17.0, dtype=wp.float64, device=warp_device)
        with pytest.raises(NumericalError, match="coefficient"):
            regularization_prox(source, step=1e-308, out=output, workspace=workspace)
        np.testing.assert_array_equal(output.numpy(), [17.0, 17.0])


def test_zero_and_smallest_positive_proximal_coefficients_have_distinct_contracts(
    warp_device: str,
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        grid = GridSpec((1, 1, 2), (1.0, 1.0, 1.0))
        source = wp.array([1.0, -1.0], dtype=wp.float64, device=warp_device)
        output = wp.empty(2, dtype=wp.float64, device=warp_device)
        zero = prepare_regularization(grid, RegularizationSpec(), device=warp_device)
        with pytest.raises(ContractError, match="positive"):
            regularization_prox(source, step=0.0, out=output, workspace=zero)
        result = regularization_prox(source, step=1e-308, out=output, workspace=zero)
        assert result.reason == "closed_form" and result.primal == result.dual == 0
        smallest = math.nextafter(0.0, 1.0)
        active = prepare_regularization(
            grid, RegularizationSpec(l1_weight=smallest), device=warp_device
        )
        result = regularization_prox(source, step=1.0, out=output, workspace=active)
        assert result.primal == result.dual == 2 * smallest
        assert result.reason == "closed_form" and result.gap == 0


@pytest.mark.parametrize("weight,step", [(1e30, 1e30), (1e-30, 1e-30)])
def test_lower_precision_host_weights_use_binary64_proximal_arithmetic(
    warp_device: str, weight: float, step: float
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        grid = GridSpec((1, 1, 2), (1.0, 1.0, 1.0))
        spec = RegularizationSpec(l1_weight=np.float32(weight))  # pyright: ignore[reportArgumentType]
        workspace = prepare_regularization(grid, spec, device=warp_device)
        values = np.asarray([1.0, -1.0])
        source = wp.array(values, dtype=wp.float64, device=warp_device)
        output = wp.empty(2, dtype=wp.float64, device=warp_device)
        result = regularization_prox(source, step=step, out=output, workspace=workspace)
        threshold = float(np.float32(weight)) * step
        expected = np.sign(values) * np.maximum(np.abs(values) - threshold, 0)
        np.testing.assert_array_equal(output.numpy(), expected)
        assert result.reason == "closed_form" and result.gap == 0
