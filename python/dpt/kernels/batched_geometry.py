"""O(V) panel metadata to finite FP64 rays, expanded only on the device."""

import warp as wp

from dpt.kernels.geometry import is_rigid_pose


@wp.kernel(enable_backward=False)
def generate(
    metadata: wp.array2d(dtype=wp.float64),
    modes: wp.array(dtype=wp.int32),
    rows: int,
    columns: int,
    sources: wp.array(dtype=wp.vec3d),
    endpoints: wp.array(dtype=wp.vec3d),
):
    view, row, column = wp.tid()
    ray = (view * rows + row) * columns + column
    source = wp.vec3d(metadata[view, 0], metadata[view, 1], metadata[view, 2])
    origin = wp.vec3d(metadata[view, 3], metadata[view, 4], metadata[view, 5])
    u = wp.vec3d(metadata[view, 6], metadata[view, 7], metadata[view, 8])
    v = wp.vec3d(metadata[view, 9], metadata[view, 10], metadata[view, 11])
    du = metadata[view, 12]
    dv = metadata[view, 13]
    length = metadata[view, 14]
    endpoint = origin + wp.float64(column) * du * u + wp.float64(row) * dv * v
    if modes[view] == 1:
        source = endpoint - length * source
    elif modes[view] == 2:
        radial = wp.vec3d(metadata[view, 15], metadata[view, 16], metadata[view, 17])
        angle = wp.float64(column) * du / length
        endpoint = (
            origin
            + (wp.cos(angle) - wp.float64(1.0)) * radial
            + wp.sin(angle) * length * u
            + wp.float64(row) * dv * v
        )
    sources[ray] = source
    endpoints[ray] = endpoint


@wp.func
def inverse_point(point: wp.vec3d, poses: wp.array2d(dtype=wp.float64), view: int):
    offset = point - wp.vec3d(poses[view, 9], poses[view, 10], poses[view, 11])
    return wp.vec3d(
        poses[view, 0] * offset[0] + poses[view, 3] * offset[1] + poses[view, 6] * offset[2],
        poses[view, 1] * offset[0] + poses[view, 4] * offset[1] + poses[view, 7] * offset[2],
        poses[view, 2] * offset[0] + poses[view, 5] * offset[1] + poses[view, 8] * offset[2],
    )


@wp.kernel(enable_backward=False)
def validate_poses(poses: wp.array2d(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
    view = wp.tid()
    rotation = wp.mat33d(
        poses[view, 0],
        poses[view, 1],
        poses[view, 2],
        poses[view, 3],
        poses[view, 4],
        poses[view, 5],
        poses[view, 6],
        poses[view, 7],
        poses[view, 8],
    )
    translation = wp.vec3d(poses[view, 9], poses[view, 10], poses[view, 11])
    if not is_rigid_pose(rotation, translation):
        wp.atomic_or(status, 0, 1)


@wp.kernel(enable_backward=False)
def validate_poses_per_view(poses: wp.array2d(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
    """Keep each view's rigid-pose diagnostics in its own status word."""
    view = wp.tid()
    rotation = wp.mat33d(
        poses[view, 0],
        poses[view, 1],
        poses[view, 2],
        poses[view, 3],
        poses[view, 4],
        poses[view, 5],
        poses[view, 6],
        poses[view, 7],
        poses[view, 8],
    )
    translation = wp.vec3d(poses[view, 9], poses[view, 10], poses[view, 11])
    if not is_rigid_pose(rotation, translation):
        wp.atomic_or(status, view, 1)


@wp.kernel(enable_backward=False)
def object_frame(
    sources: wp.array(dtype=wp.vec3d),
    endpoints: wp.array(dtype=wp.vec3d),
    poses: wp.array2d(dtype=wp.float64),
    pixels_per_view: int,
    out_sources: wp.array(dtype=wp.vec3d),
    out_endpoints: wp.array(dtype=wp.vec3d),
):
    ray = wp.tid()
    view = ray // pixels_per_view
    out_sources[ray] = inverse_point(sources[ray], poses, view)
    out_endpoints[ray] = inverse_point(endpoints[ray], poses, view)
