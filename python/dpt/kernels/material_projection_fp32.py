"""Fused material midpoint paths and VJP with FP32 sample arithmetic.

Fields are material-major FP32 nonnegative basis values or fractions. Each
material path accumulates in FP32, with FP64 geometry setup before entering
volume-local coordinates. Seeded slopes and clipping derivatives remain FP32;
only the final pose chain and reduction are FP64. Spectral response is a
separate operator with its own explicitly selected arithmetic.
"""

# Warp annotations are executable DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false

from functools import cache

import warp as wp

from dpt.kernels.geometry import _local_right_pose_adjoint
from dpt.kernels.projection import Configuration, ray_for_pixel, valid_ray
from dpt.kernels.projection_fp32 import (
    OPTIONS,
    Ray32,
    finite_interval,
    finite_point,
    get_scatter_field,
    sample_field,
    sample_point,
    sample_position,
)
from dpt.kernels.spectral_common import material_major_index

wp.set_module_options(OPTIONS)
Vec6d = wp.types.vector(length=6, dtype=wp.float64)


@cache
def get_material_forward(materials: int, double_output: bool, cell_gauss: bool):
    if cell_gauss:
        raise ValueError("FP32 material projection supports midpoint integration")
    dtype = wp.float64 if double_output else wp.float32
    vector = wp.types.vector(length=materials, dtype=wp.float32)

    @wp.kernel(module="unique", module_options=OPTIONS)
    def forward(
        fields: wp.array(dtype=wp.float32),
        pose: wp.array(dtype=wp.float64),
        config: Configuration,
        voxels: int,
        paths: wp.array(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        precise = ray_for_pixel(config, pose, pixel)
        ray = Ray32(wp.vec3(precise.origin), wp.vec3(precise.direction), wp.float32(precise.length))
        totals = vector()
        if (
            not valid_ray(precise)
            or not finite_point(ray.origin)
            or not finite_point(ray.direction)
            or not wp.isfinite(ray.length)
            or ray.length <= wp.float32(0.0)
        ):
            wp.atomic_or(status, 0, 2)
        else:
            interval = finite_interval(ray.origin, ray.direction, config.shape)
            if interval.upper > interval.lower:
                step = (interval.upper - interval.lower) / wp.float32(config.samples)
                for index in range(config.samples):
                    position = sample_position(interval.lower, step, index)
                    point = sample_point(ray.origin, ray.direction, position)
                    if not finite_point(point):
                        wp.atomic_or(status, 0, 2)
                    for material in range(materials):
                        totals[material] += sample_field(
                            fields, point, config.shape, material * voxels
                        ).value
                physical_step = ray.length * step
                for material in range(materials):
                    totals[material] *= physical_step
        for material in range(materials):
            paths[material_major_index(material, pixel, config.pixels)] = dtype(totals[material])
            if not wp.isfinite(totals[material]):
                wp.atomic_or(status, 0, 2)

    return forward


@cache
def get_material_vjp(
    materials: int,
    double_seed: bool,
    cell_gauss: bool,
    active_fields: bool,
    active_pose: bool,
    double_volume_gradient: bool = False,
    float32_volume_increments: bool = False,
):
    """Match the existing fused launch interface; increments always use FP32.

    The legacy increment flag needs no additional rounding in this path.
    A wider volume destination changes only its final atomic accumulation.
    """
    if cell_gauss:
        raise ValueError("FP32 material projection supports midpoint integration")
    seed_dtype = wp.float64 if double_seed else wp.float32
    volume_dtype = wp.float64 if double_volume_gradient else wp.float32
    scatter = get_scatter_field(double_volume_gradient)
    vector = wp.types.vector(length=materials, dtype=wp.float32)

    @wp.kernel(module="unique", module_options=OPTIONS)
    def reverse(
        fields: wp.array(dtype=wp.float32),
        pose: wp.array(dtype=wp.float64),
        config: Configuration,
        voxels: int,
        seeds: wp.array(dtype=seed_dtype),
        gradients: wp.array(dtype=volume_dtype),
        partials: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        block, lane = wp.tid()
        pixel = block * wp.block_dim() + lane
        gradient = Vec6d()
        if pixel < config.pixels:
            precise = ray_for_pixel(config, pose, pixel)
            ray = Ray32(
                wp.vec3(precise.origin), wp.vec3(precise.direction), wp.float32(precise.length)
            )
            if (
                not valid_ray(precise)
                or not finite_point(ray.origin)
                or not finite_point(ray.direction)
                or not wp.isfinite(ray.length)
                or ray.length <= wp.float32(0.0)
            ):
                wp.atomic_or(status, 0, 2)
            else:
                interval = finite_interval(ray.origin, ray.direction, config.shape)
                span = interval.upper - interval.lower
                channel_seeds = vector()
                nonzero_seed = False
                for material in range(materials):
                    channel_seeds[material] = wp.float32(
                        seeds[material_major_index(material, pixel, config.pixels)]
                    )
                    if not wp.isfinite(channel_seeds[material]):
                        wp.atomic_or(status, 0, 2)
                    if channel_seeds[material] != wp.float32(0.0):
                        nonzero_seed = True
                if span > wp.float32(0.0) and nonzero_seed:
                    step = span / wp.float32(config.samples)
                    physical_step = ray.length * step
                    mean = wp.float32(0.0)
                    direct_origin, direct_direction = wp.vec3(), wp.vec3()
                    shift_lower, shift_upper = wp.float32(0.0), wp.float32(0.0)
                    for index in range(config.samples):
                        position = sample_position(interval.lower, step, index)
                        point = sample_point(ray.origin, ray.direction, position)
                        if not finite_point(point):
                            wp.atomic_or(status, 0, 2)
                        combined_value, combined_gradient = wp.float32(0.0), wp.vec3()
                        for material in range(materials):
                            offset = material * voxels
                            seed = channel_seeds[material]
                            if wp.static(active_pose):
                                sampled = sample_field(fields, point, config.shape, offset)
                                combined_value += seed * sampled.value
                                combined_gradient += seed * sampled.gradient
                            if wp.static(active_fields):
                                scatter(
                                    gradients,
                                    point,
                                    config.shape,
                                    seed * physical_step,
                                    status,
                                    offset,
                                )
                        if wp.static(active_pose):
                            mean += combined_value
                            direct_origin += combined_gradient
                            direct_direction += position * combined_gradient
                            alpha = (wp.float32(index) + wp.float32(0.5)) / wp.float32(
                                config.samples
                            )
                            along = wp.dot(combined_gradient, ray.direction)
                            shift_lower += (wp.float32(1.0) - alpha) * along
                            shift_upper += alpha * along
                    if wp.static(active_pose):
                        lower_seed = -mean + span * shift_lower
                        upper_seed = mean + span * shift_upper
                        origin_gradient = (
                            span * direct_origin
                            + lower_seed * interval.lower_origin_gradient
                            + upper_seed * interval.upper_origin_gradient
                        )
                        direction_gradient = (
                            span * direct_direction
                            + lower_seed * interval.lower_direction_gradient
                            + upper_seed * interval.upper_direction_gradient
                        )
                        factor = ray.length / wp.float32(config.samples)
                        source_adjoint = wp.float64(factor) * (
                            wp.transpose(config.object_to_grid) * wp.vec3d(origin_gradient)
                        )
                        direction_adjoint = wp.float64(factor) * (
                            wp.transpose(config.object_to_grid) * wp.vec3d(direction_gradient)
                        )
                        gradient = _local_right_pose_adjoint(
                            source_adjoint,
                            precise.object_source,
                            direction_adjoint,
                            precise.object_direction,
                        )
        if wp.static(active_pose):
            for component in range(6):
                total = wp.tile_sum(wp.tile(gradient[component]))
                wp.tile_store(partials, total, offset=block * 6 + component)

    return reverse
