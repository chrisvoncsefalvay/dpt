"""Independent smooth penalty and small convex material-solver fixtures."""

# Warp annotations are interpreted by its kernel compiler.
# pyright: reportInvalidTypeForm=false

import importlib
from typing import Any

import numpy as np

from dpt.contracts import ContractError

wp: Any = importlib.import_module("warp")
MATRIX = np.array(((0.8, 0.15), (0.15, 0.4)), dtype=np.float64)


@wp.kernel
def quadratic_penalty(
    fields: wp.array(dtype=wp.float32),
    factor: wp.array(dtype=wp.float64),
    target: wp.vec2d,
    voxels: int,
    differentiate: bool,
    gradient: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    total = wp.float64(0.0)
    for voxel in range(voxels):
        first = wp.float64(fields[voxel]) - target[0]
        second = wp.float64(fields[voxels + voxel]) - target[1]
        g0 = factor[voxel] * (wp.float64(0.8) * first + wp.float64(0.15) * second)
        g1 = factor[voxel] * (wp.float64(0.15) * first + wp.float64(0.4) * second)
        total += wp.float64(0.5) * (first * g0 + second * g1)
        if differentiate:
            value0 = wp.float32(wp.float64(gradient[voxel]) + g0)
            value1 = wp.float32(wp.float64(gradient[voxels + voxel]) + g1)
            gradient[voxel] = value0
            gradient[voxels + voxel] = value1
            if not wp.isfinite(value0) or not wp.isfinite(value1):
                wp.atomic_or(status, 0, 2)
    loss[0] = total
    if not wp.isfinite(total):
        wp.atomic_or(status, 0, 2)


@wp.kernel
def quadratic_curvature(
    factor: wp.array(dtype=wp.float64),
    bound: wp.float64,
    curvature: wp.array(dtype=wp.float64),
):
    voxel = wp.tid()
    curvature[voxel] += factor[voxel] * bound


@wp.kernel
def invalid_curvature(value: wp.float64, curvature: wp.array(dtype=wp.float64)):
    curvature[wp.tid()] = value


class PreparedQuadraticPenalty:
    """Fixed resident buffers and the independently known constant Hessian."""

    graph_safe = True

    def __init__(
        self,
        factor: Any,
        *,
        device: str,
        target: tuple[float, float] = (0.12, 0.21),
    ) -> None:
        self.factor = wp.array(
            np.asarray(factor, dtype=np.float64), dtype=wp.float64, device=device
        )
        self.target = wp.vec2d(*target)
        self.voxels = self.factor.size
        self.bound = float(np.linalg.eigvalsh(MATRIX)[-1])

    def __call__(self, fields, *, out_loss, gradient, status, context):
        if fields.size != 2 * self.voxels or context.device != self.factor.device:
            raise ContractError("prepared penalty fields and device must match")
        wp.launch(
            quadratic_penalty,
            dim=1,
            inputs=[
                fields,
                self.factor,
                self.target,
                self.voxels,
                gradient is not None,
                fields if gradient is None else gradient,
                out_loss,
                status,
            ],
            device=context.device,
            stream=context.stream,
            record_tape=False,
        )

    def add_fisher_curvature(
        self,
        fields,
        *,
        out_curvature,
        minimum_metric_eigenvalue,
        status,
        context,
    ):
        wp.launch(
            quadratic_curvature,
            dim=self.voxels,
            inputs=[self.factor, self.bound / minimum_metric_eigenvalue, out_curvature],
            device=context.device,
            stream=context.stream,
            record_tape=False,
        )


class ValueOnlyPenalty:
    """Existing eager callable without graph or Fisher declarations."""

    def __init__(self, prepared):
        self.prepared = prepared

    def __call__(self, fields, *, out_loss, gradient, status, context):
        self.prepared(fields, out_loss=out_loss, gradient=gradient, status=status, context=context)


class InvalidCurvaturePenalty(PreparedQuadraticPenalty):
    def __init__(self, factor, *, device, value):
        super().__init__(factor, device=device)
        self.value = value

    def add_fisher_curvature(
        self,
        fields,
        *,
        out_curvature,
        minimum_metric_eigenvalue,
        status,
        context,
    ):
        wp.launch(
            invalid_curvature,
            dim=self.voxels,
            inputs=[self.value, out_curvature],
            device=context.device,
            stream=context.stream,
            record_tape=False,
        )


def quadratic_reference(fields, factor, target=(0.12, 0.21)):
    values = np.asarray(fields, dtype=np.float64).reshape(2, -1) - np.asarray(target)[:, None]
    gradient = (MATRIX @ values) * factor
    return float(0.5 * np.sum(values * gradient)), gradient.reshape(-1)
