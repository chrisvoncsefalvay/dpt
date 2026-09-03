"""Prepared dense proposal curvature; detector images remain on the owning CUDA stream."""

# Warp annotations are executable DSL forms.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false
import warp as wp

from dpt.kernels.numerics import product3

from ._constants import NUMERICAL_MASK

wp.set_module_options({"fast_math": False, "fuse_fp": True, "enable_backward": False})
TILE = 256


@wp.kernel
def flag_scattering(scattering: wp.array(dtype=wp.float64), flag: wp.array(dtype=wp.int32)):
    if scattering[wp.tid()] != wp.float64(0.0):
        wp.atomic_or(flag, 0, 1)


@wp.kernel
def store_column(
    values: wp.array(dtype=wp.float64),
    column: int,
    pixels: int,
    jacobian: wp.array(dtype=wp.float64),
):
    p = wp.tid()
    jacobian[column * pixels + p] = values[p]


@wp.func
def _gram_product(
    a: wp.float64,
    b: wp.float64,
    weight: wp.float64,
    status: wp.array(dtype=wp.int32),
) -> wp.float64:
    value = product3(a, b, weight)
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, NUMERICAL_MASK)
    return value


@wp.kernel
def gram_tiles(
    jacobian: wp.array(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    pixels: int,
    parameters: int,
    tiles: int,
    partial: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    block, lane = wp.tid()
    pair = block // tiles
    tile = block % tiles
    row = pair // parameters
    column = pair % parameters
    p = tile * TILE + lane
    value = wp.float64(0.0)
    if wp.block_dim() == 1:
        for offset in range(TILE):
            serial_pixel = tile * TILE + offset
            if serial_pixel < pixels:
                contribution = _gram_product(
                    jacobian[row * pixels + serial_pixel],
                    jacobian[column * pixels + serial_pixel],
                    weights[serial_pixel],
                    status,
                )
                value += contribution
    else:
        if p < pixels:
            value = _gram_product(
                jacobian[row * pixels + p], jacobian[column * pixels + p], weights[p], status
            )
    total = wp.tile_sum(wp.tile(value))
    wp.tile_store(partial, total, offset=block)


@wp.kernel
def sum_gram_tiles(
    values: wp.array(dtype=wp.float64), size: int, tiles: int, output: wp.array(dtype=wp.float64)
):
    block, lane = wp.tid()
    pair = block // tiles
    tile = block % tiles
    i = tile * TILE + lane
    value = wp.float64(0.0)
    if wp.block_dim() == 1:
        for offset in range(TILE):
            serial_index = tile * TILE + offset
            if serial_index < size:
                value += values[pair * size + serial_index]
    else:
        if i < size:
            value = values[pair * size + i]
    total = wp.tile_sum(wp.tile(value))
    wp.tile_store(output, total, offset=block)
