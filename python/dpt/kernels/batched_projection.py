"""Fixed per-view FP64 reductions and cone intrinsic chain rules."""

# Warp DSL annotations are executable type expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false

from functools import cache

import warp as wp

OPTIONS = {"fast_math": False, "fuse_fp": True, "enable_backward": False}


@cache
def reduce_views(components: int, intrinsic: bool = False):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def reduce(
        source: wp.array(dtype=wp.float64),
        count: int,
        blocks_per_view: int,
        poses: wp.array2d(dtype=wp.float64),
        jacobians: wp.array3d(dtype=wp.float64),
        destination: wp.array(dtype=wp.float64),
    ):
        task, lane = wp.tid()
        view = task // blocks_per_view
        block = task % blocks_per_view
        for component in range(components):
            value = wp.float64(0.0)
            rows = 1
            if wp.block_dim() == 1:
                rows = 128
            for row in range(rows):
                pixel = block * 128 + lane + row
                if pixel < count:
                    if wp.static(intrinsic):
                        # The ray cotangents live in object coordinates. Both
                        # endpoint perturbations use R_WO.T @ d_world.
                        for endpoint in range(2):
                            for axis in range(3):
                                tangent = wp.float64(0.0)
                                for world_axis in range(3):
                                    tangent += (
                                        poses[view, 3 * world_axis + axis]
                                        * jacobians[view, component, 3 * endpoint + world_axis]
                                    )
                                value += (
                                    source[
                                        (wp.int64(view) * wp.int64(count) + wp.int64(pixel))
                                        * wp.int64(6)
                                        + wp.int64(3 * endpoint + axis)
                                    ]
                                    * tangent
                                )
                    else:
                        value += source[
                            (wp.int64(view) * wp.int64(count) + wp.int64(pixel))
                            * wp.int64(components)
                            + wp.int64(component)
                        ]
            total = wp.tile_sum(wp.tile(value))
            wp.tile_store(
                destination,
                total,
                offset=wp.int32(task) * wp.int32(components) + wp.int32(component),
            )

    return reduce


@cache
def finish_views(components: int, accumulate: bool):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def finish(
        source: wp.array(dtype=wp.float64),
        destination: wp.array2d(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        view, component = wp.tid()
        result = source[wp.int64(view) * wp.int64(components) + wp.int64(component)]
        if wp.static(accumulate):
            result += destination[view, component]
        destination[view, component] = result
        if not wp.isfinite(result):
            wp.atomic_or(status, 0, 2)

    return finish
