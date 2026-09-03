"""Replicate-axis Jacobian Gram matrices for prepared local transport models.

Jacobians have layout (replicate, parameter, pixel). Gram reduction rows have
layout (replicate, parameter, parameter), keeping every replicate independent.
"""

# Warp annotations are executable DSL forms, as in the canonical kernel module.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false
# Warp's dimensional thread IDs and tile intrinsics are resolved by its compiler.
# pyright: reportOperatorIssue=false, reportAssignmentType=false
# pyright: reportGeneralTypeIssues=false, reportCallIssue=false, reportIndexIssue=false
import warp as wp

from ._constants import INPUT_MASK, NUMERICAL_MASK
from .recovery_kernels import _gram_product

wp.set_module_options({"fast_math": False, "fuse_fp": True, "enable_backward": False})
TILE = 256


@wp.kernel
def store_column(
    values: wp.array(dtype=wp.float64),
    column: int,
    pixels: int,
    dimension: int,
    jacobian: wp.array(dtype=wp.float64),
):
    flat = wp.tid()
    replicate, pixel = flat // pixels, flat % pixels
    jacobian[(replicate * dimension + column) * pixels + pixel] = values[flat]


@wp.func
def gram_value(
    a: wp.float64,
    b: wp.float64,
    weight: wp.float64,
    status: wp.array(dtype=wp.int32),
) -> wp.float64:
    if weight < wp.float64(0.0):
        wp.atomic_or(status, 0, INPUT_MASK)
    return _gram_product(a, b, weight, status)


@wp.kernel
def gram_tiles(
    jacobian: wp.array(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    pixels: int,
    dimension: int,
    tiles: int,
    partial: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    block, lane = wp.tid()
    pair, tile = block // tiles, block % tiles
    replicate = pair // (dimension * dimension)
    row, column = (pair // dimension) % dimension, pair % dimension
    row_start = (replicate * dimension + row) * pixels
    column_start = (replicate * dimension + column) * pixels
    value = wp.float64(0.0)
    if wp.block_dim() == 1:
        for offset in range(TILE):
            pixel = tile * TILE + offset
            if pixel < pixels:
                value += gram_value(
                    jacobian[row_start + pixel],
                    jacobian[column_start + pixel],
                    weights[pixel],
                    status,
                )
    else:
        pixel = tile * TILE + lane
        if pixel < pixels:
            value = gram_value(
                jacobian[row_start + pixel], jacobian[column_start + pixel], weights[pixel], status
            )
    total = wp.tile_sum(wp.tile(value))
    if not wp.isfinite(total[0]):
        wp.atomic_or(status, 0, NUMERICAL_MASK)
    wp.tile_store(partial, total, offset=block)


@wp.kernel
def finish_model(
    gradient: wp.array(dtype=wp.float64),
    gram: wp.array(dtype=wp.float64),
    dimension: int,
    out: wp.array(dtype=wp.float64),
):
    flat = wp.tid()
    width = dimension + dimension * dimension
    replicate, entry = flat // width, flat % width
    if entry < dimension:
        out[flat] = gradient[replicate * dimension + entry]
    else:
        out[flat] = gram[replicate * dimension * dimension + entry - dimension]
