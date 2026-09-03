"""Resident primary-model sensitivities and small normal-equation reductions."""

# Warp DSL annotations are runtime expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false

from functools import cache

import warp as wp

OPTIONS = {"fast_math": False, "fuse_fp": True, "enable_backward": False}
TILE = 128
COMPONENTS = 43


@cache
def prediction_kernel(binary64: bool, counts: bool, weighted: bool):
    dtype = wp.float64 if binary64 else wp.float32

    @wp.kernel(module="unique", module_options=OPTIONS)
    def prediction(
        depth: wp.array(dtype=dtype),
        observation: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        beam: wp.float64,
        validate: bool,
        signal: wp.array(dtype=wp.float64),
        seeds: wp.array(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        value = -wp.float64(depth[pixel])
        derivative = wp.float64(-1.0)
        if wp.static(counts):
            value = beam * wp.exp(value)
            derivative = -value
        signal[pixel] = value
        seeds[pixel] = dtype(derivative)
        if not wp.isfinite(value) or not wp.isfinite(seeds[pixel]):
            wp.atomic_or(status, 0, 2)
        if validate:
            if not wp.isfinite(observation[pixel]):
                wp.atomic_or(status, 0, 1)
            if wp.static(weighted):
                weight = weights[pixel]
                if not wp.isfinite(weight) or weight < wp.float32(0.0):
                    wp.atomic_or(status, 0, 1)

    return prediction


@cache
def normal_kernel(weighted: bool):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def normal(
        signal: wp.array(dtype=wp.float64),
        observation: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        local_jacobian: wp.array(dtype=wp.float64),
        chart: wp.array(dtype=wp.float64),
        count: int,
        scale: wp.float64,
        partials: wp.array(dtype=wp.float64),
    ):
        block, lane = wp.tid()
        rows = 1
        if wp.block_dim() == 1:
            rows = TILE
        for component in range(COMPONENTS):
            total = wp.float64(0.0)
            for row in range(rows):
                pixel = block * TILE + lane + row
                if pixel < count:
                    weight = scale
                    if wp.static(weighted):
                        weight *= wp.float64(weights[pixel])
                    # Zero-weight lanes contribute exactly zero before products.
                    if weight != wp.float64(0.0):
                        residual = signal[pixel] - wp.float64(observation[pixel])
                        if component == 0:
                            total += wp.float64(0.5) * weight * residual * residual
                        else:
                            axis = component - 1
                            other = axis
                            if component >= 7:
                                axis = (component - 7) // 6
                                other = (component - 7) % 6
                            first = wp.float64(0.0)
                            second = wp.float64(0.0)
                            for local_axis in range(6):
                                local = local_jacobian[pixel * 6 + local_axis]
                                first += local * chart[local_axis * 6 + axis]
                                if component >= 7:
                                    second += local * chart[local_axis * 6 + other]
                            if component < 7:
                                total += weight * residual * first
                            else:
                                total += weight * first * second
            value = wp.tile_sum(wp.tile(total))
            wp.tile_store(partials, value, offset=block * COMPONENTS + component)

    return normal
