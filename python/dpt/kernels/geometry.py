"""Double-precision coordinate algebra shared by deterministic CUDA operators."""

# Warp annotations are executable DSL expressions; host interfaces remain strict.
# The optional GPU import is resolved only when an operator is prepared.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false

import warp as wp


@wp.func
def unpack_rotation(pose: wp.array(dtype=wp.float64)) -> wp.mat33d:
    return wp.mat33d(
        pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], pose[7], pose[8]
    )


@wp.func
def unpack_translation(pose: wp.array(dtype=wp.float64)) -> wp.vec3d:
    return wp.vec3d(pose[9], pose[10], pose[11])


@wp.func
def _local_right_rotation_adjoint(
    source_adjoint: wp.vec3d,
    object_source: wp.vec3d,
    direction_adjoint: wp.vec3d,
    object_direction: wp.vec3d,
) -> wp.vec3d:
    return wp.cross(source_adjoint, object_source) + wp.cross(direction_adjoint, object_direction)


Vec6d = wp.types.vector(length=6, dtype=wp.float64)


@wp.func
def _local_right_pose_adjoint(
    source_adjoint: wp.vec3d,
    object_source: wp.vec3d,
    direction_adjoint: wp.vec3d,
    object_direction: wp.vec3d,
) -> Vec6d:
    rotation_adjoint = _local_right_rotation_adjoint(
        source_adjoint, object_source, direction_adjoint, object_direction
    )
    gradient = Vec6d()
    for axis in range(3):
        gradient[axis] = -source_adjoint[axis]
        gradient[axis + 3] = rotation_adjoint[axis]
    return gradient


@wp.func
def is_rigid_pose(rotation: wp.mat33d, translation: wp.vec3d) -> bool:
    """Finite right-handed rigid pose with the shared 1e-10 orthogonality tolerance."""
    product = wp.transpose(rotation) * rotation
    valid = True
    for i in range(3):
        if not wp.isfinite(translation[i]):
            valid = False
    for i in range(3):
        for j in range(3):
            if not wp.isfinite(rotation[i, j]):
                valid = False
            expected = wp.float64(0.0)
            if i == j:
                expected = wp.float64(1.0)
            if wp.abs(product[i, j] - expected) > wp.float64(1e-10):
                valid = False
    if wp.determinant(rotation) < wp.float64(0.0):
        valid = False
    return valid


@wp.kernel(enable_backward=False)
def validate_pose(pose: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
    if not is_rigid_pose(unpack_rotation(pose), unpack_translation(pose)):
        wp.atomic_or(status, 0, 1)
