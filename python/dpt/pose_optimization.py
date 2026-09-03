"""Sensitivity-based primary pose recovery and feasible box-constrained fitting.

Primary images and their six-column Jacobians remain on the device. Each view
returns one FP64 loss, six derivatives and a 6-by-6 Gauss-Newton matrix. Host
linear algebra and line search operate only on the small fixed parameter chart.
"""


# Public Python boundaries validate runtime inputs.
# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

if TYPE_CHECKING:
    import warp as wp

from dpt._reductions import prepare_reduction
from dpt._runtime import (
    CompletionReadback,
    StatusCheckpoint,
    StatusReadback,
    load_kernels,
    require_no_tape,
)
from dpt.contracts import (
    ContractError,
    NumericalError,
    TrialDomainError,
    finite_scalar,
    integer,
    raise_device_status,
)
from dpt.geometry import RigidTransform, right_jacobian_se3
from dpt.line_search import CubicWolfePolicy, CubicWolfeResult, ScalarEvaluation, cubic_strong_wolfe
from dpt.projection import (
    ProjectionSpec,
    prepare_projection,
    project_optical_depth,
    projection_pose_sensitivities,
)
from dpt.recovery import PoseChart, PrimaryPoseProblem
from dpt.registration import Evaluation, Vector
from dpt.solver_lifecycle import LifecycleObserver, LifecycleRecorder

Matrix = tuple[Vector, ...]


@dataclass(frozen=True, slots=True)
class NormalEquations:
    """f = 1/2 r.T r, g = J.T r, H = J.T J in fixed chart coordinates.

    Fixed weights and mean factors are absorbed into the residual/Jacobian.
    For n parameters, gradient has shape (n,) and gauss_newton (n,n), with
    1 <= n <= 12. Resident evaluators reduce these small controls on device.
    """

    loss: float
    gradient: Vector
    gauss_newton: Matrix


class SensitivityEvaluator(Protocol):
    def __call__(self, parameters: Vector, /) -> NormalEquations: ...


@dataclass(frozen=True, slots=True)
class ParameterBounds:
    """Inclusive box bounds in the evaluator's declared fixed coordinates.

    Infinite endpoints represent unbounded axes. A fixed axis has equal bounds.
    For PoseChart, the first three physical coordinates are scaled Lie-algebra
    translations in mm; they are not the final transform's world displacement.
    Calibration axes follow the supplied evaluator's calibration chart.
    """

    lower: Vector
    upper: Vector

    def __post_init__(self) -> None:
        if not self.lower or len(self.lower) != len(self.upper):
            raise ContractError("bounds need matching nonempty lower and upper vectors")
        for low, high in zip(self.lower, self.upper, strict=True):
            if (
                math.isnan(low)
                or math.isnan(high)
                or low > high
                or low == math.inf
                or high == -math.inf
            ):
                raise ContractError("box bounds must be ordered and admit finite coordinates")
        object.__setattr__(self, "lower", tuple(float(x) for x in self.lower))
        object.__setattr__(self, "upper", tuple(float(x) for x in self.upper))

    @classmethod
    def pose_translation(
        cls,
        chart: PoseChart,
        lower_mm: tuple[float, float, float],
        upper_mm: tuple[float, float, float],
    ) -> ParameterBounds:
        """Bound scaled chart translations, leaving chart rotations unbounded."""
        if len(lower_mm) != 3 or len(upper_mm) != 3:
            raise ContractError("translation bounds require three lower and upper values")
        return cls(
            tuple(x / s for x, s in zip(lower_mm, chart.scales[:3], strict=True))
            + (-math.inf,) * 3,
            tuple(x / s for x, s in zip(upper_mm, chart.scales[:3], strict=True)) + (math.inf,) * 3,
        )

    def project(self, parameters: Vector) -> Vector:
        if len(parameters) != len(self.lower) or not all(map(math.isfinite, parameters)):
            raise ContractError("bounded parameters need matching finite coordinates")
        return tuple(
            min(max(x, low), high)
            for x, low, high in zip(parameters, self.lower, self.upper, strict=True)
        )

    def projected_gradient(self, parameters: Vector, gradient: Vector) -> Vector:
        if (
            len(gradient) != len(self.lower)
            or not all(map(math.isfinite, gradient))
            or self.project(parameters) != parameters
        ):
            raise ContractError("projected gradient needs a feasible point and matching gradient")
        return tuple(
            0.0 if low == high or (x == low and g >= 0) or (x == high and g <= 0) else g
            for x, g, low, high in zip(parameters, gradient, self.lower, self.upper, strict=True)
        )


def _dot(left: Vector, right: Vector) -> float:
    try:
        result = math.fsum(a * b for a, b in zip(left, right, strict=True))
    except (OverflowError, ValueError) as error:
        raise NumericalError("unrepresentable small-chart inner product") from error
    if not math.isfinite(result):
        raise NumericalError("unrepresentable small-chart inner product")
    return result


def _checked(value: NormalEquations, dimension: int) -> NormalEquations:
    if (
        not isinstance(value, NormalEquations)
        or len(value.gradient) != dimension
        or len(value.gauss_newton) != dimension
        or any(len(row) != dimension for row in value.gauss_newton)
    ):
        raise ContractError("normal equations differ from the declared parameter dimension")
    if (
        not math.isfinite(value.loss)
        or not all(map(math.isfinite, value.gradient))
        or not all(math.isfinite(v) for row in value.gauss_newton for v in row)
    ):
        raise NumericalError("non-finite least-squares loss, gradient or normal matrix")
    # Congruence by sqrt(diag(H)) removes column-unit disparity. Performing
    # the larger division first avoids an overflowing product of scale factors.
    # J.T J has nonnegative diagonals; a zero diagonal has exact zero coupling.
    matrix = value.gauss_newton
    diagonal = tuple(matrix[i][i] for i in range(dimension))
    if any(x < 0 for x in diagonal):
        raise ContractError("Gauss-Newton matrix must be positive semidefinite")
    roots = tuple(math.sqrt(x) for x in diagonal)
    work = [[0.0] * dimension for _ in range(dimension)]
    tolerance = 128 * math.ulp(1.0) * dimension
    for i in range(dimension):
        work[i][i] = float(diagonal[i] > 0)
        for j in range(i):
            left, right = matrix[i][j], matrix[j][i]
            pair_scale = max(abs(left), abs(right))
            if pair_scale > 0 and abs(left / pair_scale - right / pair_scale) > tolerance:
                raise ContractError("Gauss-Newton matrix must be symmetric")
            if roots[i] == 0 or roots[j] == 0:
                if left != 0 or right != 0:
                    raise ContractError("Gauss-Newton matrix must be positive semidefinite")
                continue
            correlation = (left / max(roots[i], roots[j])) / min(roots[i], roots[j])
            if not math.isfinite(correlation) or abs(correlation) > 1 + tolerance:
                raise ContractError("Gauss-Newton matrix must be positive semidefinite")
            work[i][j] = work[j][i] = correlation
    # Symmetric largest-diagonal pivoting makes rank decisions independent of
    # coordinate order. All entries are bounded, so Schur updates cannot incur
    # the overflow of a raw huge rank-one matrix. Only round-off-sized residuals
    # of the dimensionless correlation matrix are treated as numerical zero.
    for axis in range(dimension):
        pivot_axis = max(range(axis, dimension), key=lambda i: work[i][i])
        if pivot_axis != axis:
            work[axis], work[pivot_axis] = work[pivot_axis], work[axis]
            for row in work:
                row[axis], row[pivot_axis] = row[pivot_axis], row[axis]
        pivot = work[axis][axis]
        if any(work[i][i] < -tolerance for i in range(axis, dimension)):
            raise ContractError("Gauss-Newton matrix must be positive semidefinite")
        if pivot <= tolerance:
            if any(
                abs(work[i][j]) > tolerance
                for i in range(axis, dimension)
                for j in range(axis, dimension)
            ):
                raise ContractError("Gauss-Newton matrix must be positive semidefinite")
            break
        for row in range(axis + 1, dimension):
            for column in range(row, dimension):
                updated = work[column][row] - (work[row][axis] / pivot) * work[column][axis]
                work[column][row] = work[row][column] = updated
    # Own the completed small state even when a callback reused mutable rows.
    return NormalEquations(
        float(value.loss),
        tuple(float(x) for x in value.gradient),
        tuple(tuple(float(x) for x in row) for row in value.gauss_newton),
    )


def _cholesky_step(
    matrix: Matrix, gradient: Vector, damping: float, free: tuple[int, ...]
) -> Vector | None:
    size = len(free)
    lower = [[0.0] * size for _ in range(size)]
    diagonal_scale = max(1.0, max(matrix[i][i] for i in free)) if free else 1.0
    try:
        for i, row in enumerate(free):
            for j, column in enumerate(free[: i + 1]):
                value = matrix[row][column]
                if i == j:
                    value += damping * max(matrix[row][row], diagonal_scale * 1e-12)
                value -= math.fsum(lower[i][k] * lower[j][k] for k in range(j))
                if i == j:
                    if value <= 0 or not math.isfinite(value):
                        return None
                    lower[i][j] = math.sqrt(value)
                else:
                    lower[i][j] = value / lower[j][j]
        first = [0.0] * size
        for i, axis in enumerate(free):
            first[i] = (
                -gradient[axis] - math.fsum(lower[i][j] * first[j] for j in range(i))
            ) / lower[i][i]
        solved = [0.0] * size
        for i in range(size - 1, -1, -1):
            solved[i] = (
                first[i] - math.fsum(lower[j][i] * solved[j] for j in range(i + 1, size))
            ) / lower[i][i]
        answer = [0.0] * len(gradient)
        for i, axis in enumerate(free):
            answer[axis] = solved[i]
        return tuple(answer) if all(map(math.isfinite, answer)) else None
    except (OverflowError, ValueError, ZeroDivisionError):
        return None


def _box_step(
    matrix: Matrix, gradient: Vector, point: Vector, bounds: ParameterBounds, damping: float
) -> Vector | None:
    """Solve the small convex quadratic with feasible active-set steps."""
    dimension = len(point)
    diagonal_scale = max(1.0, max(matrix[i][i] for i in range(dimension)))
    shifted = tuple(
        tuple(
            value + (damping * max(matrix[i][i], diagonal_scale * 1e-12) if i == j else 0.0)
            for j, value in enumerate(row)
        )
        for i, row in enumerate(matrix)
    )
    step = (0.0,) * dimension
    active = {
        i
        for i, (x, g, low, high) in enumerate(
            zip(point, gradient, bounds.lower, bounds.upper, strict=True)
        )
        if low == high or (x == low and g >= 0) or (x == high and g <= 0)
    }
    for _ in range(4 * dimension * dimension + 4):
        model_gradient = tuple(
            g + _dot(row, step) for g, row in zip(gradient, shifted, strict=True)
        )
        free = tuple(i for i in range(dimension) if i not in active)
        correction = _cholesky_step(shifted, model_gradient, 0.0, free)
        if correction is None:
            return None
        alpha, hit = 1.0, None
        for i in free:
            value = point[i] + step[i]
            change = correction[i]
            maximum = math.inf
            if change > 0:
                maximum = (bounds.upper[i] - value) / change
            elif change < 0:
                maximum = (bounds.lower[i] - value) / change
            if maximum < alpha:
                alpha, hit = max(0.0, maximum), i
        candidate = bounds.project(
            tuple(x + p + alpha * d for x, p, d in zip(point, step, correction, strict=True))
        )
        step = tuple(y - x for x, y in zip(point, candidate, strict=True))
        if hit is not None:
            active.add(hit)
            continue
        model_gradient = tuple(
            g + _dot(row, step) for g, row in zip(gradient, shifted, strict=True)
        )
        release: list[tuple[float, int]] = []
        for i in active:
            if bounds.lower[i] == bounds.upper[i]:
                continue
            value = point[i] + step[i]
            g = model_gradient[i]
            if value == bounds.lower[i] and g < 0:
                release.append((-g, i))
            elif value == bounds.upper[i] and g > 0:
                release.append((g, i))
        if not release:
            return step
        active.remove(max(release)[1])
    return None


@dataclass(frozen=True, slots=True)
class PoseLeastSquaresPolicy:
    method: Literal["gauss_newton", "levenberg_marquardt"] = "levenberg_marquardt"
    max_iterations: int = 100
    max_evaluations: int = 500
    gradient_tolerance: float = 1e-7
    step_tolerance: float = 1e-12
    initial_damping: float = 1e-3
    minimum_damping: float = 1e-15
    maximum_damping: float = 1e15
    acceptance_ratio: float = 1e-4
    line_search: CubicWolfePolicy = field(default_factory=CubicWolfePolicy)

    def __post_init__(self) -> None:
        if self.method not in ("gauss_newton", "levenberg_marquardt"):
            raise ContractError("method must be gauss_newton or levenberg_marquardt")
        integer(self.max_iterations, "max_iterations", minimum=1)
        integer(self.max_evaluations, "max_evaluations", minimum=1)
        for name in (
            "gradient_tolerance",
            "step_tolerance",
            "initial_damping",
            "minimum_damping",
            "maximum_damping",
            "acceptance_ratio",
        ):
            finite_scalar(getattr(self, name), name, minimum=0.0)
        if (
            not 0 < self.minimum_damping <= self.initial_damping < self.maximum_damping
            or not 0 < self.acceptance_ratio < 1
        ):
            raise ContractError("invalid LM damping interval or acceptance ratio")


@dataclass(frozen=True, slots=True)
class LeastSquaresIteration:
    iteration: int
    parameters: Vector
    loss: float
    projected_gradient_norm: float
    damping: float
    accepted: bool
    step_norm: float
    ratio: float | None


@dataclass(frozen=True, slots=True)
class LeastSquaresResult:
    parameters: Vector
    evaluation: NormalEquations
    reason: Literal[
        "gradient_tolerance",
        "projected_gradient_tolerance",
        "step_tolerance",
        "evaluation_budget",
        "iteration_budget",
        "line_search_failed",
        "damping_limit",
        "cancelled",
    ]
    evaluations: int
    history: tuple[LeastSquaresIteration, ...]
    rejected_trials: tuple[tuple[Vector, str], ...]

    @property
    def stationary(self) -> bool:
        return self.reason in ("gradient_tolerance", "projected_gradient_tolerance")


class _EvaluationBudgetError(Exception):
    pass


def recover_least_squares(
    evaluator: SensitivityEvaluator,
    initial: Vector,
    *,
    bounds: ParameterBounds | None = None,
    policy: PoseLeastSquaresPolicy | None = None,
    cancelled: Callable[[], bool] | None = None,
    observe_lifecycle: LifecycleObserver | None = None,
) -> LeastSquaresResult:
    """g=J^T*r and H=J^T*J define GN/LM steps for f(x)=r^T*r/2.

    Initial is a feasible length-D finite binary64 tuple in a fixed chart with
    1 <= D <= 12. Evaluator returns NormalEquations containing scalar loss,
    length-D gradient and D-by-D positive-semidefinite Gauss-Newton matrix in
    matching chart units. Fixed weights/mean factors belong in r and J.
    ParameterBounds holds length-D lower/upper tuples in the same coordinates.
    GN uses cubic Wolfe and LM adjusts damping from actual/predicted decrease.
    A feasible active-set solve respects the box and projected-gradient descent
    handles a singular solve. The result preserves accepted parameters and
    normal equations, with scalar termination/history diagnostics. Rejected
    trials cannot replace them. Host arithmetic uses binary64.

    >>> from dpt.pose_optimization import NormalEquations, recover_least_squares
    >>> def normal_equations(x):
    ...     return NormalEquations(0.5 * (x[0] - 2.0) ** 2, (x[0] - 2.0,), ((1.0,),))
    >>> result = recover_least_squares(normal_equations, (0.0,))
    >>> assert abs(result.parameters[0] - 2.0) < 1e-6

    """
    with LifecycleRecorder("least_squares", observe_lifecycle) as lifecycle:
        result = _recover_least_squares(
            evaluator,
            initial,
            bounds=bounds,
            policy=policy,
            cancelled=cancelled,
            lifecycle=lifecycle,
        )
        lifecycle.finish(result)
        return result


def _recover_least_squares(
    evaluator: SensitivityEvaluator,
    initial: Vector,
    *,
    bounds: ParameterBounds | None = None,
    policy: PoseLeastSquaresPolicy | None = None,
    cancelled: Callable[[], bool] | None = None,
    lifecycle: LifecycleRecorder,
) -> LeastSquaresResult:
    """Minimise f(x) using g = J.T r and H = J.T J in one fixed chart.

    The direction solves min_s g.T s + 1/2 s.T (H + lambda D) s subject
    to lower <= x+s <= upper, with a feasible active-set quadratic solve.
    GN uses lambda=0 and cubic Wolfe; LM adjusts lambda from the ratio of
    actual to predicted decrease. A projected-gradient direction supplies
    descent if the small quadratic solve is singular or loses descent.

    Initial coordinates must already be feasible. The projected-gradient KKT
    test removes only blocked outward derivatives, retaining free-axis errors.
    Rejected trials never replace the accepted parameters or normal equations.
    """
    rules: PoseLeastSquaresPolicy = PoseLeastSquaresPolicy() if policy is None else policy
    point: Vector = tuple(float(x) for x in initial)
    dimension = len(point)
    if not 1 <= dimension <= 12 or not all(map(math.isfinite, point)):
        raise ContractError("least-squares recovery needs 1-12 finite chart coordinates")
    box = (
        ParameterBounds((-math.inf,) * dimension, (math.inf,) * dimension)
        if bounds is None
        else bounds
    )
    if box.project(point) != point:
        raise ContractError("initial least-squares point must satisfy all box bounds")
    evaluations = 0
    rejected: list[tuple[Vector, str]] = []

    def evaluate(trial: Vector) -> NormalEquations:
        nonlocal evaluations
        if evaluations >= rules.max_evaluations:
            raise _EvaluationBudgetError
        evaluations += 1
        try:
            return _checked(evaluator(trial), dimension)
        except TrialDomainError as error:
            rejected.append((trial, str(error)))
            raise

    current = evaluate(point)
    minimum_damping: float = rules.minimum_damping
    maximum_damping: float = rules.maximum_damping
    acceptance_ratio: float = rules.acceptance_ratio
    damping = rules.initial_damping if rules.method == "levenberg_marquardt" else 0.0
    history: list[LeastSquaresIteration] = []

    def finish(
        reason: Literal[
            "gradient_tolerance",
            "projected_gradient_tolerance",
            "step_tolerance",
            "evaluation_budget",
            "iteration_budget",
            "line_search_failed",
            "damping_limit",
            "cancelled",
        ],
    ) -> LeastSquaresResult:
        return LeastSquaresResult(
            point, current, reason, evaluations, tuple(history), tuple(rejected)
        )

    lifecycle.start(objective=current.loss)
    for iteration in range(rules.max_iterations + 1):
        projected = box.projected_gradient(point, current.gradient)
        norm = max(map(abs, projected))
        if iteration == 0:
            history.append(
                LeastSquaresIteration(0, point, current.loss, norm, damping, True, 0.0, None)
            )
        if norm <= rules.gradient_tolerance:
            return finish(
                "gradient_tolerance"
                if projected == current.gradient
                else "projected_gradient_tolerance"
            )
        if cancelled is not None and cancelled():
            return finish("cancelled")
        if iteration == rules.max_iterations:
            return finish("iteration_budget")
        if evaluations == rules.max_evaluations:
            return finish("evaluation_budget")
        direction = _box_step(current.gauss_newton, current.gradient, point, box, damping)
        if direction is None or _dot(current.gradient, direction) >= 0:
            scale = max(1.0, norm)
            feasible = box.project(
                tuple(x - g / scale for x, g in zip(point, projected, strict=True))
            )
            direction = tuple(y - x for x, y in zip(point, feasible, strict=True))
        length = max(map(abs, direction))
        if length <= rules.step_tolerance:
            return finish("step_tolerance")
        if rules.method == "gauss_newton":
            candidates: dict[float, tuple[Vector, NormalEquations]] = {}
            bound = math.inf
            for x, d, low, high in zip(point, direction, box.lower, box.upper, strict=True):
                if d > 0:
                    bound = min(bound, (high - x) / d)
                elif d < 0:
                    bound = min(bound, (low - x) / d)

            def ray(
                step: float,
                *,
                origin: Vector = point,
                search_direction: Vector = direction,
                cache: dict[float, tuple[Vector, NormalEquations]] = candidates,
            ) -> ScalarEvaluation:
                trial = box.project(
                    tuple(x + step * d for x, d in zip(origin, search_direction, strict=True))
                )
                value = evaluate(trial)
                cache[step] = trial, value
                return ScalarEvaluation(value.loss, _dot(value.gradient, search_direction))

            try:
                search: CubicWolfeResult = cubic_strong_wolfe(
                    ray,
                    ScalarEvaluation(current.loss, _dot(current.gradient, direction)),
                    feasible_maximum=bound if math.isfinite(bound) else None,
                    policy=rules.line_search,
                )
            except _EvaluationBudgetError:
                return finish("evaluation_budget")
            if not search.accepted:
                history.append(
                    LeastSquaresIteration(
                        iteration + 1, point, current.loss, norm, damping, False, 0.0, None
                    )
                )
                lifecycle.rejected(iteration + 1, "line_search_failed")
                return finish("line_search_failed")
            point, current = candidates[search.step]
            history.append(
                LeastSquaresIteration(
                    iteration + 1,
                    point,
                    current.loss,
                    max(map(abs, box.projected_gradient(point, current.gradient))),
                    damping,
                    True,
                    length * search.step,
                    None,
                )
            )
            lifecycle.accepted(iteration + 1, current.loss)
            continue
        predicted = -_dot(current.gradient, direction) - 0.5 * _dot(
            direction, tuple(_dot(row, direction) for row in current.gauss_newton)
        )
        trial = box.project(tuple(x + d for x, d in zip(point, direction, strict=True)))
        candidate: NormalEquations | None = None
        try:
            candidate = evaluate(trial)
        except TrialDomainError:
            pass
        except _EvaluationBudgetError:
            return finish("evaluation_budget")
        ratio = (
            (current.loss - candidate.loss) / predicted
            if candidate is not None and predicted > 0
            else -math.inf
        )
        accepted: bool = (
            candidate is not None and candidate.loss < current.loss and ratio >= acceptance_ratio
        )
        if accepted:
            assert candidate is not None
            point, current = trial, candidate
            if ratio > 0.75:
                damping = max(minimum_damping, damping / 3)
            elif ratio < 0.25:
                damping *= 2
        else:
            damping *= 4
        history.append(
            LeastSquaresIteration(
                iteration + 1,
                point,
                current.loss,
                max(map(abs, box.projected_gradient(point, current.gradient))),
                damping,
                accepted,
                length,
                ratio if math.isfinite(ratio) else None,
            )
        )
        if accepted:
            lifecycle.accepted(iteration + 1, current.loss)
        else:
            lifecycle.rejected(iteration + 1, "domain" if candidate is None else "agreement")
        if damping > maximum_damping:
            return finish("damping_limit")
    return finish("iteration_budget")


@dataclass(frozen=True, slots=True)
class BoundedGradientPolicy:
    """Projected-gradient fitting with a safeguarded cubic-Wolfe search."""

    max_iterations: int = 100
    max_evaluations: int = 500
    gradient_tolerance: float = 1e-7
    step_tolerance: float = 1e-12
    line_search: CubicWolfePolicy = field(default_factory=CubicWolfePolicy)

    def __post_init__(self) -> None:
        integer(self.max_iterations, "max_iterations", minimum=1)
        integer(self.max_evaluations, "max_evaluations", minimum=1)
        finite_scalar(self.gradient_tolerance, "gradient_tolerance", minimum=0.0)
        finite_scalar(self.step_tolerance, "step_tolerance", minimum=0.0)


@dataclass(frozen=True, slots=True)
class BoundedParameterResult:
    parameters: Vector
    evaluation: Evaluation
    reason: str
    evaluations: int
    history: tuple[LeastSquaresIteration, ...]
    rejected_trials: tuple[tuple[Vector, str], ...]

    @property
    def stationary(self) -> bool:
        return self.reason in ("gradient_tolerance", "projected_gradient_tolerance")


def recover_bounded_parameters(
    evaluator: Callable[[Vector], Evaluation],
    initial: Vector,
    *,
    bounds: ParameterBounds,
    policy: BoundedGradientPolicy | None = None,
    cancelled: Callable[[], bool] | None = None,
    observe_lifecycle: LifecycleObserver | None = None,
) -> BoundedParameterResult:
    """x_next = project_box(x + a*d) minimises f using its gradient and a feasible Wolfe
    step.

    Initial is a feasible length-D finite binary64 tuple and bounds has matching
    length-D lower/upper tuples. Evaluator returns scalar loss and length-D
    gradient in the same immutable chart. The identity search metric uses first
    derivatives directly, with no constructed J^T*J. PrimaryPoseEvaluator and
    SpectralPoseEvaluator satisfy this interface, retaining their scaled pose
    and nuisance charts. BoundedParameterResult records the accepted parameters,
    Evaluation, reason and search history. TrialDomainError rejects a chart trial
    and numerical failures propagate. Lifecycle observers receive committed
    scalar states. All small host operations use binary64.

    >>> from dpt.registration import Evaluation
    >>> from dpt.pose_optimization import ParameterBounds, recover_bounded_parameters
    >>> def objective(x):
    ...     return Evaluation(0.5 * (x[0] - 2.0) ** 2, (x[0] - 2.0,))
    >>> result = recover_bounded_parameters(
    ...     objective, (0.0,), bounds=ParameterBounds((0.0,), (1.0,))
    ... )
    >>> result.parameters
    (1.0,)

    """
    options = BoundedGradientPolicy() if policy is None else policy
    rules = PoseLeastSquaresPolicy(
        method="gauss_newton",
        max_iterations=options.max_iterations,
        max_evaluations=options.max_evaluations,
        gradient_tolerance=options.gradient_tolerance,
        step_tolerance=options.step_tolerance,
        line_search=options.line_search,
    )
    dimension = len(initial)
    identity = tuple(tuple(float(i == j) for j in range(dimension)) for i in range(dimension))

    def normal(parameters: Vector) -> NormalEquations:
        value = evaluator(parameters)
        return NormalEquations(value.loss, value.gradient, identity)

    result = recover_least_squares(
        normal,
        initial,
        bounds=bounds,
        policy=rules,
        cancelled=cancelled,
        observe_lifecycle=observe_lifecycle,
    )
    return BoundedParameterResult(
        result.parameters,
        Evaluation(result.evaluation.loss, result.evaluation.gradient),
        result.reason,
        result.evaluations,
        result.history,
        result.rejected_trials,
    )


@dataclass(slots=True)
class _PrimaryView:
    problem: PrimaryPoseProblem
    projection: Any
    pose: Any
    depth: Any
    prediction: Any
    seeds: Any
    jacobian: Any
    empty: Any
    normal: Any
    reduction: Any


class PrimaryPoseLeastSquaresEvaluator:
    """r_v = prediction_v - observation_v; f = sum_v 1/2 r_v.T W_v r_v.

    Each view's six-column local-right sensitivity is mapped through the fixed
    PoseChart Jacobian before resident reduction of g = sum_v J_v.T W_v r_v
    and H = sum_v J_v.T W_v J_v. This fits one shared six-coordinate rigid pose.

    Each view's sum/mean reduction and fixed weights match ObjectiveSpec. Views
    add their objectives; a mean is per view, never divided again by view count.
    Counts use open_beam * exp(-L); log transmission uses -L. Signal arithmetic,
    Jacobians and normal equations are FP64 even with FP32 optical-depth storage.
    Attenuation, observations, weights and chart must remain immutable. Result
    images are borrowed scratch, updated by each completed trial. Concurrent,
    recursive and ambient-tape evaluation is rejected.
    """

    def __init__(
        self,
        problems: PrimaryPoseProblem | tuple[PrimaryPoseProblem, ...],
        chart: PoseChart,
        *,
        device: str = "cuda:0",
        stream: Any = None,
        max_bytes: int = 1 << 30,
    ) -> None:
        self.problems = (problems,) if isinstance(problems, PrimaryPoseProblem) else tuple(problems)
        if not self.problems or not isinstance(chart, PoseChart):
            raise ContractError("primary least squares requires views and a validated PoseChart")
        self.chart = chart
        integer(max_bytes, "max_bytes", minimum=1)
        estimated = 0
        for problem in self.problems:
            if (
                not isinstance(problem, PrimaryPoseProblem)
                or problem.objective.kind != "squared_error"
                or problem.objective.masked
                or problem.objective.domain not in ("counts", "log_transmission")
            ):
                raise ContractError(
                    "primary GN/LM requires unmasked counts/log-transmission squared error"
                )
            beam = finite_scalar(problem.open_beam, "open_beam", minimum=0.0)
            if problem.objective.domain == "counts" and beam == 0:
                raise ContractError("counts pose fitting requires positive open_beam")
            # Conservative CPU first-level tree plus all per-pixel arrays. The
            # bounded estimate is checked before allocating any owned buffer.
            estimated += problem.geometry.pixels * (8 + 8 + 8 + 48 + 43 * 8 * 2) + 4096
        if estimated > max_bytes:
            raise ContractError("primary normal-equation workspace exceeds max_bytes")
        self._views: list[_PrimaryView] = []
        self._kernels = load_kernels("dpt.kernels.pose_optimization")
        for problem in self.problems:
            projection = prepare_projection(
                problem.grid,
                problem.geometry,
                ProjectionSpec(
                    problem.samples_per_ray,
                    precision=problem.precision,
                    integration=problem.integration,
                ),
                device=device,
                stream=stream,
            )
            ctx, wp = projection.context, projection.context.wp
            if self._views:
                self.context.assert_stream(ctx.stream)
            else:
                self.context = ctx
                stream = ctx.stream
            pixels = problem.geometry.pixels
            ctx.array(
                problem.attenuation, "attenuation", dtype=wp.float32, shape=(problem.grid.voxels,)
            )
            ctx.array(problem.observation, "observation", dtype=wp.float32, shape=(pixels,))
            if problem.objective.weighted:
                ctx.array(problem.weights, "weights", dtype=wp.float32, shape=(pixels,))
            elif problem.weights is not None:
                raise ContractError("weights require a weighted objective")
            with ctx.scope():
                pose = wp.empty(12, dtype=wp.float64, device=ctx.device)
                depth = wp.empty(pixels, dtype=projection.dtype, device=ctx.device)
                prediction = wp.empty(pixels, dtype=wp.float64, device=ctx.device)
                seeds = wp.empty(pixels, dtype=projection.dtype, device=ctx.device)
                jacobian = wp.empty(6 * pixels, dtype=wp.float64, device=ctx.device)
                empty = wp.empty(0, dtype=wp.float32, device=ctx.device)
                normal = wp.empty(43, dtype=wp.float64, device=ctx.device)
            reduction = prepare_reduction(ctx, pixels, tile=128, components=43)
            self._views.append(
                _PrimaryView(
                    problem,
                    projection,
                    pose,
                    depth,
                    prediction,
                    seeds,
                    jacobian,
                    empty,
                    normal,
                    reduction,
                )
            )
        ctx, wp = self.context, self.context.wp
        with ctx.scope():
            self._status = wp.zeros(1, dtype=wp.int32, device=ctx.device)
            self._chart_device = wp.empty(36, dtype=wp.float64, device=ctx.device)
        self._pose_host = wp.empty(12, dtype=wp.float64, device="cpu", pinned=ctx.device.is_cuda)
        self._pose_view = self._pose_host.numpy()
        self._chart_host = wp.empty(36, dtype=wp.float64, device="cpu", pinned=ctx.device.is_cuda)
        self._chart_view = self._chart_host.numpy()
        self._status_readback = StatusReadback(ctx, self._status)
        self._checkpoint = StatusCheckpoint(ctx, (*(v.projection for v in self._views), self))
        self._completion = CompletionReadback(
            self._checkpoint, tuple(v.normal for v in self._views)
        )
        self._lock = Lock()
        self._validated = False
        self.estimated_owned_bytes = estimated

    def _raise_status(self, code: int) -> None:
        raise_device_status(
            code,
            "primary least squares",
            messages={1: "observations/weights must be finite and weights nonnegative"},
        )

    def check_status(self) -> None:
        self._raise_status(self._status_readback.read())

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()

    @property
    def predicted_images(self) -> tuple[wp.array[wp.float32] | wp.array[wp.float64], ...]:
        """Borrow the current device prediction arrays in their selected storage dtype."""
        return cast(
            "tuple[wp.array[wp.float32] | wp.array[wp.float64], ...]",
            tuple(v.prediction for v in self._views),
        )

    @property
    def control_readback_values(self) -> int:
        """43 FP64 numeric entries per view, plus one word per diagnostic owner."""
        return 43 * len(self._views) + len(self._views) + 1

    def __call__(self, parameters: Vector) -> NormalEquations:
        """f=r^T*r/2, g=J^T*r and H=J^T*J reduce primary images in the fixed pose chart.

        Parameters is a length-6 dimensionless binary64 tuple. PoseChart maps its
        scaled translations in mm and rotations in rad through anchor @ exp(x^).
        Each prepared view has FP32 attenuation (voxels,), observation/weights (P,)
        and FP64 depth, prediction and weighted six-column image sensitivities.
        NormalEquations returns binary64 loss, gradient (6,) and matrix (6,6),
        with fixed weights and mean factors absorbed in r/J. The chart Jacobian
        and scales convert local-right sensitivities before reduction. Only the
        pose chart is active, while grid, acquisition, fields and observation stay
        fixed. Images/Jacobians remain on device and only small controls are read
        back. Calls are sequential on the owning context.

        """
        require_no_tape()
        if not self._lock.acquire(blocking=False):
            raise ContractError("a primary least-squares evaluator cannot be shared concurrently")
        completed = False
        try:
            pose = self.chart.pose(parameters)
            chain = right_jacobian_se3(self.chart.increment(parameters))
            self._checkpoint.consume()
            ctx, wp = self.context, self.context.wp
            with ctx.scope():
                self._pose_view[:] = pose.packed()
                self._chart_view[:] = [
                    chain[i][j] * self.chart.scales[j] for i in range(6) for j in range(6)
                ]
                wp.copy(self._chart_device, self._chart_host, stream=ctx.stream)
                for view in self._views:
                    projection, problem = view.projection, view.problem
                    wp.copy(view.pose, self._pose_host, stream=ctx.stream)
                    project_optical_depth(
                        problem.attenuation,
                        view.pose,
                        out_L=view.depth,
                        workspace=projection,
                        stream=ctx.stream,
                        validate=not self._validated,
                    )
                    wp.launch(
                        self._kernels.prediction_kernel(
                            problem.precision == "float64",
                            problem.objective.domain == "counts",
                            problem.objective.weighted,
                        ),
                        dim=problem.geometry.pixels,
                        inputs=[
                            view.depth,
                            problem.observation,
                            problem.weights if problem.objective.weighted else view.empty,
                            problem.open_beam,
                            not self._validated,
                        ],
                        outputs=[view.prediction, view.seeds, self._status],
                        device=ctx.device,
                        stream=ctx.stream,
                        record_tape=False,
                    )
                    projection_pose_sensitivities(
                        problem.attenuation,
                        view.pose,
                        cotangent=view.seeds,
                        out_jacobian=view.jacobian,
                        workspace=projection,
                        stream=ctx.stream,
                        validate=False,
                    )
                    count = problem.geometry.pixels
                    blocks = (count + 127) // 128
                    wp.launch_tiled(
                        self._kernels.normal_kernel(problem.objective.weighted),
                        dim=blocks,
                        inputs=[
                            view.prediction,
                            problem.observation,
                            problem.weights if problem.objective.weighted else view.empty,
                            view.jacobian,
                            self._chart_device,
                            count,
                            1 / count if problem.objective.reduction == "mean" else 1.0,
                        ],
                        outputs=[view.reduction.partials[0]],
                        block_dim=128,
                        device=ctx.device,
                        stream=ctx.stream,
                        record_tape=False,
                    )
                    view.reduction.finish(blocks, view.normal, self._status, components=43)
                controls = self._completion.consume()
                self._validated = True
            combined = tuple(math.fsum(float(row[i]) for row in controls) for i in range(43))
            value = NormalEquations(
                combined[0],
                combined[1:7],
                tuple(combined[7 + 6 * i : 13 + 6 * i] for i in range(6)),
            )
            completed = True
            return _checked(value, 6)
        finally:
            try:
                if not completed and self.context.device.is_cuda:
                    self.context.wp.synchronize_stream(self.context.stream)
            finally:
                self._lock.release()


@dataclass(frozen=True, slots=True)
class SensitivityPoseRecoveryResult:
    pose: RigidTransform
    optimisation: LeastSquaresResult


def recover_pose_sensitivities(
    evaluator: PrimaryPoseLeastSquaresEvaluator,
    *,
    initial: Vector = (0.0,) * 6,
    bounds: ParameterBounds | None = None,
    policy: PoseLeastSquaresPolicy | None = None,
    cancelled: Callable[[], bool] | None = None,
    observe_lifecycle: LifecycleObserver | None = None,
) -> SensitivityPoseRecoveryResult:
    """T_accepted = anchor @ exp((scales*x_accepted)^) follows accepted GN/LM coordinates.

    Initial is a length-6 dimensionless binary64 tuple and optional bounds use
    that same fixed chart. The prepared PrimaryPoseLeastSquaresEvaluator returns
    scalar loss, six derivatives and a 6-by-6 Gauss-Newton matrix from resident
    images. Policy selects GN/LM, damping, feasible search and termination.
    SensitivityPoseRecoveryResult contains the accepted RigidTransform in world
    mm and LeastSquaresResult. Fixed acquisitions/fields/observations remain
    unchanged. The chart translations are scaled Lie coordinates rather than
    final world displacement. Reevaluate accepted optimisation parameters when
    its predicted image is required, since reusable scratch may hold a trial.

    """
    result = recover_least_squares(
        evaluator,
        initial,
        bounds=bounds,
        policy=policy,
        cancelled=cancelled,
        observe_lifecycle=observe_lifecycle,
    )
    return SensitivityPoseRecoveryResult(evaluator.chart.pose(result.parameters), result)
