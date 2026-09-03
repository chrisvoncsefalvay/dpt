"""Exact status-word packing for prepared FP64 completion payloads."""

import warp as wp

wp.set_module_options({"fast_math": False, "fuse_fp": True, "enable_backward": False})


@wp.kernel
def pack_status(
    source: wp.array(dtype=wp.int32),
    offset: int,
    destination: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    destination[offset + index] = wp.float64(source[index])
