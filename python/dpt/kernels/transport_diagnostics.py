"""Original-history terminal counts for checked transport simulation results."""

import warp as wp

wp.set_module_options({"fast_math": False, "fuse_fp": True, "enable_backward": False})


@wp.kernel
def status_histogram(
    history_status: wp.array(dtype=wp.int32),
    histogram: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    code = history_status[wp.tid()]
    if code >= 0 and code < histogram.shape[0]:
        wp.atomic_add(histogram, code, 1)
    else:
        wp.atomic_or(status, 0, 1)
