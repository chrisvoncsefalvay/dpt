"""Independent near-stationary references for the FP64 primary composition."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeArgument=false

import importlib
import math
from dataclasses import replace
from decimal import Decimal, localcontext
from typing import Any

import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.geometry import DetectorGeometry
from dpt.objectives import ObjectiveSpec, evaluate_primary_objective, prepare_objective
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.recovery import PoseChart, PrimaryPoseEvaluator, PrimaryPoseProblem
from dpt.registration import Evaluation, RecoveryPolicy, recover_parameters
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")
np: Any = importlib.import_module("numpy")
pytestmark = pytest.mark.gpu


def _affine_problem():
    """A z-directed ray crosses exactly 3 mm of an x-affine represented field."""
    grid = GridSpec((3, 3, 3), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0))
    geometry = DetectorGeometry(
        (0.13, 0.17, -10.0),
        (0.13, 0.17, 10.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.1, 0.1),
        (1, 1),
    )
    field = np.tile(np.array([0.03, 0.04, 0.05], dtype=np.float32), 9)
    slope = 3.0 * (float(field[2]) - float(field[1]))
    base_depth = 3.0 * (0.87 * float(field[1]) + 0.13 * float(field[2]))
    beam, observed = 1_000_000.0, 880_000.0
    optimum = (base_depth + math.log(observed / beam)) / slope
    problem = PrimaryPoseProblem(
        grid,
        geometry,
        wp.array(field, dtype=wp.float32, device="cuda:0"),
        wp.array([observed], dtype=wp.float32, device="cuda:0"),
        ObjectiveSpec(kind="poisson", domain="counts"),
        beam,
        samples_per_ray=37,
    )
    return problem, optimum, slope, base_depth


def test_fp64_near_stationary_loss_has_the_independently_predicted_slope() -> None:
    problem, optimum, slope, base_depth = _affine_problem()
    evaluator = PrimaryPoseEvaluator(problem, PoseChart())
    x = optimum + 2e-6
    actual = evaluator((x, 0.0, 0.0, 0.0, 0.0, 0.0))
    expected_mean = problem.open_beam * math.exp(-(base_depth - slope * x))
    expected_gradient = slope * (expected_mean - 880_000.0)
    assert actual.gradient[0] == pytest.approx(expected_gradient, rel=2e-7, abs=2e-10)
    for step in (1e-5, 1e-6, 1e-7):
        plus = evaluator((x + step, 0.0, 0.0, 0.0, 0.0, 0.0)).loss
        minus = evaluator((x - step, 0.0, 0.0, 0.0, 0.0, 0.0)).loss
        assert (plus - minus) / (2 * step) == pytest.approx(expected_gradient, rel=3e-6, abs=3e-10)
    assert evaluator.optical_depth.dtype == wp.float64
    assert evaluator.prediction.dtype == wp.float64
    assert evaluator.depth_seed.dtype == wp.float64
    legacy = PrimaryPoseEvaluator(replace(problem, precision="float32"), PoseChart())
    assert legacy((x, 0.0, 0.0, 0.0, 0.0, 0.0)).loss >= 0.0
    assert legacy.optical_depth.dtype == wp.float32


def test_canonical_solver_reaches_an_independently_known_poisson_minimum() -> None:
    problem, optimum, _, _ = _affine_problem()
    evaluator = PrimaryPoseEvaluator(problem, PoseChart())

    def one_coordinate(point: tuple[float, ...]) -> Evaluation:
        value = evaluator((point[0], 0.0, 0.0, 0.0, 0.0, 0.0))
        return Evaluation(value.loss, (value.gradient[0],))

    policy = RecoveryPolicy(
        gradient_tolerance=1e-8,
        relative_loss_tolerance=0.0,
        step_tolerance=0.0,
        max_iterations=100,
        max_evaluations=1000,
    )
    result = recover_parameters(one_coordinate, (0.0,), policy=policy)
    assert result.stationary, result
    assert result.parameters[0] == pytest.approx(optimum, abs=5e-11)
    assert all(b.loss <= a.loss for a, b in zip(result.history, result.history[1:], strict=False))


def test_fp64_primary_poisson_term_matches_decimal_and_retains_depth_adjoint() -> None:
    depths = [math.log(2.0) + 1e-8, 460.0, 700.0, 0.0]
    observed = [500_000.0, 1.0, 0.0, 1_000_000.0]
    beam = 1_000_000.0
    work = prepare_objective(ObjectiveSpec(kind="poisson", domain="counts"), max_pixels=4)
    depth = wp.array(depths, dtype=wp.float64, device="cuda:0")
    obs = wp.array(observed, dtype=wp.float32, device="cuda:0")
    predicted = wp.empty(4, dtype=wp.float64, device="cuda:0")
    seed = wp.empty(4, dtype=wp.float64, device="cuda:0")
    loss = wp.empty(1, dtype=wp.float64, device="cuda:0")
    evaluate_primary_objective(
        depth,
        obs,
        open_beam=beam,
        out_prediction=predicted,
        out_depth_cotangent=seed,
        out_loss=loss,
        workspace=work,
    )
    with localcontext() as ctx:
        ctx.prec = 70
        means = [Decimal(beam) * (-Decimal(x)).exp() for x in depths]
        terms = [
            m - Decimal(y) + Decimal(y) * (Decimal(y) / m).ln() if y else m
            for m, y in zip(means, observed, strict=True)
        ]
        expected_loss = float(sum(terms))
        expected_seeds = [float(Decimal(y) - m) for m, y in zip(means, observed, strict=True)]
    assert float(loss.numpy()[0]) == pytest.approx(expected_loss, rel=3e-15)
    np.testing.assert_allclose(seed.numpy(), expected_seeds, rtol=3e-8, atol=1e-10)
    assert float(seed.numpy()[1]) == 1.0
    depth.assign(np.array([1000.0, 460.0, 700.0, 0.0]))
    with pytest.raises(NumericalError):
        evaluate_primary_objective(
            depth,
            obs,
            open_beam=beam,
            out_prediction=predicted,
            out_depth_cotangent=seed,
            out_loss=loss,
            workspace=work,
        )


def test_declared_projection_precision_rejects_mismatched_output_and_seed() -> None:
    problem, _, _, _ = _affine_problem()
    work = prepare_projection(
        problem.grid, problem.geometry, ProjectionSpec(37, precision="float64")
    )
    pose = wp.array(PoseChart().anchor.packed(), dtype=wp.float64, device="cuda:0")
    wrong = wp.empty(1, dtype=wp.float32, device="cuda:0")
    with pytest.raises(ContractError):
        project_optical_depth(problem.attenuation, pose, workspace=work, out_L=wrong)
    with pytest.raises(ContractError):
        projection_vjp(
            problem.attenuation,
            pose,
            cotangent=wrong,
            out_pose_tangent=wp.empty(6, dtype=wp.float64, device="cuda:0"),
            workspace=work,
        )


@pytest.mark.parametrize("optical,beam", [(750.0, 1e6), (1000.0, 1e300)])
def test_primary_scaled_exponential_preserves_representable_count_range(
    optical: float, beam: float
) -> None:
    work = prepare_objective(ObjectiveSpec(kind="poisson", domain="counts"), max_pixels=1)
    depth = wp.array([optical], dtype=wp.float64, device="cuda:0")
    observed = wp.array([1.0], dtype=wp.float32, device="cuda:0")
    prediction = wp.empty(1, dtype=wp.float64, device="cuda:0")
    seed = wp.empty(1, dtype=wp.float64, device="cuda:0")
    loss = wp.empty(1, dtype=wp.float64, device="cuda:0")
    evaluate_primary_objective(
        depth,
        observed,
        open_beam=beam,
        out_prediction=prediction,
        out_depth_cotangent=seed,
        out_loss=loss,
        workspace=work,
    )
    with localcontext() as ctx:
        ctx.prec = 80
        expected = float(Decimal(beam) * (-Decimal(optical)).exp())
        expected_loss = float(
            Decimal(beam) * (-Decimal(optical)).exp() - 1 - (Decimal(beam).ln() - Decimal(optical))
        )
    assert float(prediction.numpy()[0]) == pytest.approx(expected, rel=2e-12, abs=5e-324)
    assert float(loss.numpy()[0]) == pytest.approx(expected_loss, rel=2e-14)
    assert float(seed.numpy()[0]) == 1.0


@pytest.mark.parametrize(
    "kind,domain",
    [("poisson", "counts"), ("squared_error", "counts"), ("squared_error", "log_transmission")],
)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_primary_fused_weights_mask_and_denominator_match_independent_formula(
    kind: Any, domain: Any, reduction: Any
) -> None:
    values = np.array([0.1, 0.7, 2.0, 0.9], dtype=np.float64)
    targets = np.array([8.0, 3.0, 0.0, np.nan], dtype=np.float32)
    weights = np.array([2.0, 0.25, 0.0, np.nan], dtype=np.float32)
    mask = np.array([1, 1, 1, 0], dtype=np.uint8)
    beam = 10.0
    work = prepare_objective(
        ObjectiveSpec(kind=kind, domain=domain, reduction=reduction, weighted=True, masked=True),
        max_pixels=4,
    )
    depth = wp.array(values, dtype=wp.float64, device="cuda:0")
    observed = wp.array(targets, dtype=wp.float32, device="cuda:0")
    weight = wp.array(weights, dtype=wp.float32, device="cuda:0")
    valid = wp.array(mask, dtype=wp.uint8, device="cuda:0")
    prediction = wp.empty(4, dtype=wp.float64, device="cuda:0")
    seed = wp.empty(4, dtype=wp.float64, device="cuda:0")
    loss = wp.empty(1, dtype=wp.float64, device="cuda:0")
    evaluate_primary_objective(
        depth,
        observed,
        open_beam=beam,
        weights=weight,
        valid=valid,
        out_prediction=prediction,
        out_depth_cotangent=seed,
        out_loss=loss,
        workspace=work,
    )
    means = beam * np.exp(-values) if domain == "counts" else -values
    difference = means[:3] - targets[:3]
    if kind == "poisson":
        terms = np.array(
            [
                means[i]
                - float(targets[i])
                + float(targets[i]) * math.log(float(targets[i]) / means[i])
                if targets[i]
                else means[i]
                for i in range(3)
            ]
        )
        derivatives = -difference
    else:
        terms = 0.5 * difference**2
        derivatives = -difference * (means[:3] if domain == "counts" else 1.0)
    normalisation = 4.0 if reduction == "mean" else 1.0
    assert float(loss.numpy()[0]) == pytest.approx(
        float(terms @ weights[:3] / normalisation), rel=2e-14
    )
    np.testing.assert_allclose(
        seed.numpy(),
        [*list(derivatives * weights[:3] / normalisation), 0.0],
        rtol=2e-14,
        atol=1e-15,
    )
    with pytest.raises(ContractError):
        evaluate_primary_objective(
            depth,
            observed,
            open_beam=beam,
            weights=weight,
            valid=valid,
            out_prediction=depth,
            out_depth_cotangent=seed,
            out_loss=loss,
            workspace=work,
        )
