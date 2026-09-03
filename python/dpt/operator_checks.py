"""Check <Ax, y> = <x, A*y> and estimate ||A||₂ by power iteration.

These diagnostics use explicit host arrays and FP64 scalar reductions. The
projection adapter checks the actual discretised forward and volume VJP.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from dpt.contracts import ContractError, integer

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from dpt.geometry import RigidTransform
    from dpt.projection import ProjectionWorkspace

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LinearOperator:
    """A: R^domain_shape → R^range_shape, with its Euclidean adjoint A*.

    Callables accept and return host NumPy arrays with the declared shapes.
    Inputs and outputs must be finite real values. Device adapters own their
    scratch; separate records are needed for concurrent diagnostic runs.

    Example::

        operator = LinearOperator((3,), (2,), lambda x: matrix @ x,
                                  lambda y: matrix.T @ y)
        result = check_adjoint(operator)
    """

    domain_shape: tuple[int, ...]
    range_shape: tuple[int, ...]
    forward: Callable[[FloatArray], FloatArray]
    adjoint: Callable[[FloatArray], FloatArray]

    def __post_init__(self) -> None:
        for name, shape in (("domain_shape", self.domain_shape), ("range_shape", self.range_shape)):
            if not isinstance(cast(object, shape), tuple) or not shape:
                raise ContractError(f"{name} must be a nonempty tuple")
            for dimension in shape:
                integer(dimension, name, minimum=1)
        if not callable(self.forward) or not callable(self.adjoint):
            raise ContractError("forward and adjoint must be callable")


@dataclass(frozen=True, slots=True)
class AdjointCheck:
    """Observed inner products and error scaled by the two vector norm products."""

    forward_inner_product: float
    adjoint_inner_product: float
    absolute_error: float
    relative_error: float
    tolerance: float
    passed: bool
    seed: int


@dataclass(frozen=True, slots=True)
class OperatorNorm:
    """Estimated ||A||₂, iteration count and relative A*A eigenvector residual."""

    value: float
    iterations: int
    residual: float
    converged: bool
    seed: int


def _array(value: FloatArray, shape: tuple[int, ...], name: str) -> FloatArray:
    import numpy as np

    result = np.asarray(value)
    if result.shape != shape or np.iscomplexobj(result):
        raise ContractError(f"{name} must be a real array with shape {shape}")
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ContractError(f"{name} must contain finite values")
    return result


def _tolerance(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ContractError("tolerance must be finite and positive")
    return float(value)


def _finite_scalar(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ContractError(f"{name} exceeds the finite FP64 diagnostic range")
    return value


def _scaled(values: FloatArray) -> tuple[FloatArray, float]:
    import numpy as np

    scale = float(np.max(np.abs(values)))
    return (values / scale, scale) if scale else (values, 0.0)


def _unit_and_norm(values: FloatArray) -> tuple[FloatArray, float]:
    import numpy as np

    scaled, scale = _scaled(values)
    if not scale:
        return scaled, 0.0
    scaled_norm = float(np.linalg.norm(scaled))
    norm = _finite_scalar(scale * scaled_norm, "vector norm")
    return scaled / scaled_norm, norm


def check_adjoint(
    operator: LinearOperator, *, seed: int = 0, tolerance: float = 1e-6
) -> AdjointCheck:
    """Compare <Ax, y> with <x, A*y> for reproducible normal random arrays.

    ``x`` has ``operator.domain_shape`` and ``y`` has
    ``operator.range_shape``. Their forward/adjoint callables return the
    opposite declared shapes as finite real NumPy arrays. The scalar inner
    products and norm reductions use FP64 arithmetic.

    Error is divided by max(||Ax|| ||y||, ||x|| ||A*y||), so nearly cancelling
    inner products remain well scaled. A zero map has zero relative error.
    This is a numerical test of the supplied discrete pair at one draw.

    Example:
        >>> import numpy as np
        >>> matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> operator = LinearOperator(
        ...     (2,), (2,), lambda x: matrix @ x, lambda y: matrix.T @ y,
        ... )
        >>> check_adjoint(operator, seed=17, tolerance=1e-12).passed
        True

    """
    import numpy as np

    integer(seed, "seed", minimum=0)
    tolerance = _tolerance(tolerance)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(operator.domain_shape)
    y = rng.standard_normal(operator.range_shape)
    image = _array(operator.forward(x), operator.range_shape, "forward output")
    image_scaled, image_scale = _scaled(image)
    left_scaled = float(np.vdot(image_scaled, y))
    left_norm_product = float(np.linalg.norm(image_scaled)) * float(np.linalg.norm(y))
    # Consume the forward output before adjoint: a callable may reuse its host scratch.
    backprojection = _array(operator.adjoint(y), operator.domain_shape, "adjoint output")
    back_scaled, back_scale = _scaled(backprojection)
    right_scaled = float(np.vdot(x, back_scaled))
    right_norm_product = float(np.linalg.norm(x)) * float(np.linalg.norm(back_scaled))
    scale = max(image_scale, back_scale)
    if scale:
        left_ratio, right_ratio = image_scale / scale, back_scale / scale
        scaled_error = abs(left_ratio * left_scaled - right_ratio * right_scaled)
        denominator = max(left_ratio * left_norm_product, right_ratio * right_norm_product)
        relative_error = scaled_error / denominator if denominator else 0.0
        error = _finite_scalar(scale * scaled_error, "adjoint absolute error")
    else:
        error, relative_error = 0.0, 0.0
    left = _finite_scalar(image_scale * left_scaled, "forward inner product")
    right = _finite_scalar(back_scale * right_scaled, "adjoint inner product")
    return AdjointCheck(
        left, right, error, relative_error, tolerance, relative_error <= tolerance, seed
    )


def estimate_operator_norm(
    operator: LinearOperator,
    *,
    seed: int = 0,
    iterations: int = 100,
    tolerance: float = 1e-6,
) -> OperatorNorm:
    """Estimate ||A||₂ from power iteration on A*A, using host FP64 reductions.

    Both callables use the declared domain/range shapes. Convergence means the
    relative eigenvector residual is at most tolerance; the returned value
    remains an estimate when the iteration budget is exhausted. A matched
    adjoint and linear forward map are required for the spectral-norm meaning.

    Example:
        >>> import numpy as np
        >>> matrix = np.diag([2.0, 1.0])
        >>> operator = LinearOperator(
        ...     (2,), (2,), lambda x: matrix @ x, lambda y: matrix.T @ y,
        ... )
        >>> norm = estimate_operator_norm(operator, seed=17, tolerance=1e-10)
        >>> norm.converged, round(norm.value, 8)
        (True, 2.0)

    """
    import numpy as np

    integer(seed, "seed", minimum=0)
    integer(iterations, "iterations", minimum=1)
    tolerance = _tolerance(tolerance)
    x, _ = _unit_and_norm(np.random.default_rng(seed).standard_normal(operator.domain_shape))
    residual = math.inf
    value = 0.0
    for count in range(1, iterations + 1):
        image = _array(operator.forward(x), operator.range_shape, "forward output")
        unit_image, value = _unit_and_norm(image)
        if value == 0.0:
            return OperatorNorm(0.0, count, 0.0, True, seed)
        backprojection = _array(
            operator.adjoint(unit_image), operator.domain_shape, "adjoint output"
        )
        unit_back, back_norm = _unit_and_norm(backprojection)
        if back_norm == 0.0:
            raise ContractError("the supplied adjoint annihilates a nonzero forward output")
        # A*(Ax/||Ax||) has one operator scale, avoiding squared-scale overflow.
        scale = max(value, back_norm)
        ratio = value / scale
        if ratio == 0.0:
            raise ContractError("the supplied adjoint exceeds the forward diagnostic range")
        residual = _finite_scalar(
            float(np.linalg.norm(backprojection / scale - ratio * x)) / ratio,
            "operator norm residual",
        )
        if residual <= tolerance:
            return OperatorNorm(value, count, residual, True, seed)
        x = unit_back
    return OperatorNorm(value, iterations, residual, False, seed)


def projection_linear_operator(
    workspace: ProjectionWorkspace, pose: RigidTransform
) -> LinearOperator:
    """Return ``forward(x) = A(pose) x`` and ``adjoint(y) = A(pose).T y``.

    Domain shape is (grid.voxels,), in x-fast order; range shape is
    (geometry.pixels,), in row-major order. Use a signed_linear workspace with
    active_volume=True and active_pose=False. The adapter owns field, output
    and FP64 gradient scratch and transfers diagnostic arrays explicitly.
    Each call synchronises its owning stream before returning a host result.

    The field has mm^-1 units and its line integral is dimensionless. The
    Euclidean transpose applies the selected quadrature's physical path weights.

    """
    import numpy as np

    from dpt._runtime import HostReadback
    from dpt.projection import project_optical_depth, projection_vjp

    if workspace.spec.field_domain != "signed_linear":
        raise ContractError("projection diagnostics require field_domain='signed_linear'")
    if not workspace.spec.active_volume or workspace.spec.active_pose:
        raise ContractError("projection diagnostics require active_volume=True and fixed pose")
    if workspace.spec.storage_precision != "float32":
        raise ContractError("the host projection adapter requires float32 field storage")
    ctx = workspace.context
    wp = ctx.wp
    with ctx.scope():
        field = wp.zeros(workspace.grid.voxels, dtype=wp.float32, device=ctx.device)
        pose_array = wp.array(pose.packed(), dtype=wp.float64, device=ctx.device)
        image = wp.empty(workspace.geometry.pixels, dtype=workspace.dtype, device=ctx.device)
        gradient = wp.empty(workspace.grid.voxels, dtype=wp.float64, device=ctx.device)
    image_readback = HostReadback(ctx, image)
    gradient_readback = HostReadback(ctx, gradient)

    def forward(values: FloatArray) -> FloatArray:
        values = _array(values, (workspace.grid.voxels,), "field")
        with ctx.scope():
            incoming = wp.array(values.astype(np.float32), dtype=wp.float32, device=ctx.device)
            wp.copy(field, incoming, stream=ctx.stream)
        project_optical_depth(field, pose_array, workspace=workspace, out_L=image)
        return np.asarray(image_readback.read(), dtype=np.float64).copy()

    def adjoint(values: FloatArray) -> FloatArray:
        values = _array(values, (workspace.geometry.pixels,), "cotangent")
        with ctx.scope():
            incoming = wp.array(values, dtype=workspace.dtype, device=ctx.device)
        projection_vjp(
            field,
            pose_array,
            cotangent=incoming,
            workspace=workspace,
            out_mu=gradient,
            volume_gradient_precision="float64",
        )
        return np.asarray(gradient_readback.read(), dtype=np.float64).copy()

    return LinearOperator((workspace.grid.voxels,), (workspace.geometry.pixels,), forward, adjoint)
