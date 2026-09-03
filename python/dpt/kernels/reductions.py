"""One fixed FP64 reduction tree, specialised by tile and component counts."""

from functools import cache

import warp as wp

OPTIONS = {"fast_math": False, "fuse_fp": True, "enable_backward": False}


@cache
def reduce_kernel(tile: int, components: int):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def reduce(
        source: wp.array(dtype=wp.float64),
        count: int,
        destination: wp.array(dtype=wp.float64),
    ):
        block, lane = wp.tid()
        index = block * tile + lane
        for component in range(components):
            value = wp.float64(0.0)
            if wp.block_dim() == 1:
                # Warp CPU executes a tiled launch with one lane. Each block
                # must still consume a whole logical tile so the tree shrinks
                # by the same factor and retains every source row.
                for row in range(tile):
                    serial_index = block * tile + row
                    if serial_index < count:
                        value += source[serial_index * components + component]
            else:
                if index < count:
                    value = source[index * components + component]
            total = wp.tile_sum(wp.tile(value))
            wp.tile_store(destination, total, offset=block * components + component)

    return reduce


@cache
def finish_kernel(binary32: bool, accumulate: bool):
    dtype = wp.float32 if binary32 else wp.float64

    @wp.kernel(module="unique", module_options=OPTIONS)
    def finish(
        source: wp.array(dtype=wp.float64),
        nonempty: bool,
        destination: wp.array(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        component = wp.tid()
        value = wp.float64(0.0)
        if nonempty:
            value = source[component]
        if wp.static(accumulate):
            value += wp.float64(destination[component])
        result = dtype(value)
        destination[component] = result
        if not wp.isfinite(result):
            wp.atomic_or(status, 0, 2)

    return finish
