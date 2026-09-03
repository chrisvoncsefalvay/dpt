"""Joint material-fraction validation; projection physics stays in projection.py."""

# Warp annotations are executable DSL expressions; host interfaces remain strict.
# The optional GPU import is resolved only when an operator is prepared.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false

from functools import cache

import warp as wp

from dpt.kernels.geometry import _local_right_pose_adjoint
from dpt.kernels.projection import (
    OPTIONS,
    CellSegment,
    Configuration,
    advance_cells,
    begin_cells,
    cell_segment,
    finite_interval,
    finite_point,
    get_cell_integrator,
    get_midpoint_integrator,
    get_scatter_cell,
    get_scatter_field,
    ray_for_pixel,
    sample_field,
    sample_segment,
    support_value,
    valid_ray,
)
from dpt.kernels.spectral_common import material_major_index

FRACTION_SUM_LIMIT = wp.constant(wp.float64(1.0 + 2.0**-24))


@cache
def _get_material_segment_action(materials: int):
    vector = wp.types.vector(length=materials, dtype=wp.float64)

    @wp.func
    def action(
        fields: wp.array(dtype=wp.float32),
        segment: CellSegment,
        shape: wp.vec3i,
        voxels: int,
        totals: vector,
    ) -> vector:
        for material in range(materials):
            first, second = sample_segment(fields, segment, shape, material * voxels)
            totals[material] += segment.radius * first.value
            totals[material] += segment.radius * second.value
        return totals

    return action


@cache
def _get_material_midpoint_action(materials: int):
    vector = wp.types.vector(length=materials, dtype=wp.float64)

    @wp.func
    def action(
        fields: wp.array(dtype=wp.float32),
        point: wp.vec3d,
        shape: wp.vec3i,
        voxels: int,
        totals: vector,
    ) -> vector:
        for material in range(materials):
            totals[material] += sample_field(fields, point, shape, material * voxels).value
        return totals

    return action


@cache
def get_material_forward(materials: int, double_output: bool, cell_gauss: bool):
    """Trace one shared ray/traversal while integrating all material channels."""
    dtype = wp.float64 if double_output else wp.float32
    vector = wp.types.vector(length=materials, dtype=wp.float64)
    integrate_cells = get_cell_integrator(_get_material_segment_action(materials))
    integrate_midpoints = get_midpoint_integrator(_get_material_midpoint_action(materials))

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
        ray = ray_for_pixel(config, pose, pixel)
        totals = vector()
        if not valid_ray(ray):
            wp.atomic_or(status, 0, 2)
        else:
            interval = finite_interval(ray.origin, ray.direction, config.shape)
            if interval.upper > interval.lower:
                if wp.static(cell_gauss):
                    totals = integrate_cells(
                        fields, ray, interval, config.shape, voxels, totals, status
                    )
                    for material in range(materials):
                        totals[material] *= ray.length
                else:
                    totals, step = integrate_midpoints(
                        fields, ray, interval, config.shape, config.samples, voxels, totals, status
                    )
                    for material in range(materials):
                        totals[material] *= ray.length * step
        for material in range(materials):
            value = dtype(totals[material])
            paths[material_major_index(material, pixel, config.pixels)] = value
            if not wp.isfinite(value):
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
    """Linearity combines seeded channel slopes before one shared pose reduction."""
    dtype = wp.float64 if double_seed else wp.float32
    gradient_dtype = wp.float64 if double_volume_gradient else wp.float32
    scatter = get_scatter_field(double_volume_gradient, False, float32_volume_increments)
    scatter_cell = get_scatter_cell(double_volume_gradient, float32_volume_increments)
    vector = wp.types.vector(length=6, dtype=wp.float64)
    material_vector = wp.types.vector(length=materials, dtype=wp.float64)

    @wp.kernel(module="unique", module_options=OPTIONS)
    def reverse(
        fields: wp.array(dtype=wp.float32),
        pose: wp.array(dtype=wp.float64),
        config: Configuration,
        voxels: int,
        seeds: wp.array(dtype=dtype),
        gradients: wp.array(dtype=gradient_dtype),
        partials: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        block, lane = wp.tid()
        pixel = block * wp.block_dim() + lane
        gradient = vector()
        if pixel < config.pixels:
            ray = ray_for_pixel(config, pose, pixel)
            if not valid_ray(ray):
                wp.atomic_or(status, 0, 2)
            else:
                interval = finite_interval(ray.origin, ray.direction, config.shape)
                span = interval.upper - interval.lower
                # Typed initialisation allows mutation in Warp's dynamic loop.
                nonzero_seed = bool(False)  # noqa: UP018
                for material in range(materials):
                    if seeds[material_major_index(material, pixel, config.pixels)] != dtype(0.0):
                        nonzero_seed = True
                # Zero-cotangent lanes still participate in the pose reduction.
                if span > wp.float64(0.0) and nonzero_seed:
                    origin_gradient, direction_gradient = wp.vec3d(), wp.vec3d()
                    factor = ray.length
                    if wp.static(cell_gauss):
                        traversal = begin_cells(ray, interval, config.shape)
                        start = wp.float64(interval.lower)
                        segments = wp.int64(0)
                        limit = (
                            wp.int64(config.shape[0])
                            + wp.int64(config.shape[1])
                            + wp.int64(config.shape[2])
                            + wp.int64(1)
                        )
                        while start < interval.upper and segments < limit:
                            segment = cell_segment(ray, traversal, start, config.shape)
                            if segment.end <= start:
                                wp.atomic_or(status, 0, 2)
                                break
                            if not finite_point(segment.first_point) or not finite_point(
                                segment.second_point
                            ):
                                wp.atomic_or(status, 0, 2)
                            first_gradient, second_gradient = wp.vec3d(), wp.vec3d()
                            for material in range(materials):
                                seed = wp.float64(
                                    seeds[material_major_index(material, pixel, config.pixels)]
                                )
                                offset = material * voxels
                                if wp.static(active_pose):
                                    first, second = sample_segment(
                                        fields, segment, config.shape, offset
                                    )
                                    first_gradient += seed * first.gradient
                                    second_gradient += seed * second.gradient
                                if wp.static(active_fields):
                                    field_factor = seed * ray.length * segment.radius
                                    if segment.shared_cell:
                                        scatter_cell(
                                            gradients,
                                            segment.lower,
                                            segment.first_point,
                                            segment.second_point,
                                            config.shape,
                                            field_factor,
                                            status,
                                            offset,
                                        )
                                    else:
                                        scatter(
                                            gradients,
                                            segment.first_point,
                                            config.shape,
                                            field_factor,
                                            status,
                                            offset,
                                        )
                                        scatter(
                                            gradients,
                                            segment.second_point,
                                            config.shape,
                                            field_factor,
                                            status,
                                            offset,
                                        )
                            if wp.static(active_pose):
                                origin_gradient += segment.radius * first_gradient
                                origin_gradient += segment.radius * second_gradient
                                direction_gradient += (
                                    segment.radius * segment.first_position * first_gradient
                                )
                                direction_gradient += (
                                    segment.radius * segment.second_position * second_gradient
                                )
                            start = segment.end
                            traversal = advance_cells(
                                traversal, ray, config.shape, segment.end, interval.upper
                            )
                            segments += wp.int64(1)
                        if start < interval.upper:
                            wp.atomic_or(status, 0, 2)
                        if wp.static(active_pose):
                            lower_value, upper_value = wp.float64(0.0), wp.float64(0.0)
                            for material in range(materials):
                                seed = wp.float64(
                                    seeds[material_major_index(material, pixel, config.pixels)]
                                )
                                lower_value += seed * support_value(
                                    fields,
                                    ray.origin + interval.lower * ray.direction,
                                    config.shape,
                                    material * voxels,
                                )
                                upper_value += seed * support_value(
                                    fields,
                                    ray.origin + interval.upper * ray.direction,
                                    config.shape,
                                    material * voxels,
                                )
                            origin_gradient += (
                                upper_value * interval.upper_origin_gradient
                                - lower_value * interval.lower_origin_gradient
                            )
                            direction_gradient += (
                                upper_value * interval.upper_direction_gradient
                                - lower_value * interval.lower_direction_gradient
                            )
                    else:
                        step = span / wp.float64(config.samples)
                        if wp.static(active_fields):
                            field_factors = material_vector()
                            for material in range(materials):
                                # Keep cache indices static for all supported basis sizes.
                                material_index = wp.static(material)
                                seed = wp.float64(
                                    seeds[
                                        material_major_index(material_index, pixel, config.pixels)
                                    ]
                                )
                                field_factors[material_index] = (
                                    seed * ray.length * span / wp.float64(config.samples)
                                )
                        mean, shift_lower, shift_upper = (
                            wp.float64(0.0),
                            wp.float64(0.0),
                            wp.float64(0.0),
                        )
                        direct_origin, direct_direction = wp.vec3d(), wp.vec3d()
                        for sample in range(config.samples):
                            alpha = (wp.float64(sample) + wp.float64(0.5)) / wp.float64(
                                config.samples
                            )
                            position = (
                                interval.lower + (wp.float64(sample) + wp.float64(0.5)) * step
                            )
                            point = ray.origin + position * ray.direction
                            if not finite_point(point):
                                wp.atomic_or(status, 0, 2)
                            combined_value, combined_gradient = wp.float64(0.0), wp.vec3d()
                            for material in range(materials):
                                seed = wp.float64(
                                    seeds[material_major_index(material, pixel, config.pixels)]
                                )
                                offset = material * voxels
                                if wp.static(active_pose):
                                    value = sample_field(fields, point, config.shape, offset)
                                    combined_value += seed * value.value
                                    combined_gradient += seed * value.gradient
                                if wp.static(active_fields):
                                    scatter(
                                        gradients,
                                        point,
                                        config.shape,
                                        field_factors[material],
                                        status,
                                        offset,
                                    )
                            if wp.static(active_pose):
                                mean += combined_value
                                direct_origin += combined_gradient
                                direct_direction += position * combined_gradient
                                along = wp.dot(combined_gradient, ray.direction)
                                shift_lower += (wp.float64(1.0) - alpha) * along
                                shift_upper += alpha * along
                        if wp.static(active_pose):
                            lower_seed, upper_seed = (
                                -mean + span * shift_lower,
                                mean + span * shift_upper,
                            )
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
                        factor /= wp.float64(config.samples)
                    if wp.static(active_pose):
                        source_adjoint = factor * (
                            wp.transpose(config.object_to_grid) * origin_gradient
                        )
                        direction_adjoint = factor * (
                            wp.transpose(config.object_to_grid) * direction_gradient
                        )
                        pose_adjoint = _local_right_pose_adjoint(
                            source_adjoint,
                            ray.object_source,
                            direction_adjoint,
                            ray.object_direction,
                        )
                        for axis in range(3):
                            gradient[axis] = pose_adjoint[axis]
                            gradient[axis + 3] = pose_adjoint[axis + 3]
        if wp.static(active_pose):
            for component in range(6):
                total = wp.tile_sum(wp.tile(gradient[component]))
                wp.tile_store(partials, total, offset=block * 6 + component)

    return reverse


@wp.kernel(enable_backward=False)
def validate_fractions(
    fractions: wp.array(dtype=wp.float32),
    materials: int,
    voxels: int,
    status: wp.array(dtype=wp.int32),
):
    voxel = wp.tid()
    total = wp.float64(0.0)
    for material in range(materials):
        fraction = fractions[material * voxels + voxel]
        if not wp.isfinite(fraction) or fraction < wp.float32(0.0):
            wp.atomic_or(status, 0, 8)
        total += wp.float64(fraction)
    # Independently rounding a nonnegative partition to binary32 can increase
    # its sum by at most unit roundoff (2^-24). Accumulate in FP64 and allow
    # this representation error once, independent of the material count. Do
    # not renormalise fields or admit a whole binary32 ulp above unit mass.
    if total > FRACTION_SUM_LIMIT:
        wp.atomic_or(status, 0, 8)


@cache
def validate_seeds(precision: str = "float32"):
    value_dtype = wp.float64 if precision == "float64" else wp.float32

    @wp.kernel(module="unique", enable_backward=False)
    def validate(seeds: wp.array(dtype=value_dtype), status: wp.array(dtype=wp.int32)):
        if not wp.isfinite(seeds[wp.tid()]):
            wp.atomic_or(status, 0, 1)

    return validate


@wp.kernel(enable_backward=False)
def validate_nonnegative(
    fields: wp.array(dtype=wp.float32),
    materials: int,
    voxels: int,
    status: wp.array(dtype=wp.int32),
):
    voxel = wp.tid()
    for material in range(materials):
        value = fields[material * voxels + voxel]
        if not wp.isfinite(value) or value < wp.float32(0.0):
            wp.atomic_or(status, 0, 8)
