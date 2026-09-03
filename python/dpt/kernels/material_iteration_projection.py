"""Material projections whose step and momentum stay in resident controls.

The update arithmetic, rounding, slope and KKT certificates are the canonical
material-reconstruction implementations. These variants load only the two
algorithm scalars from fixed FP64 control storage before that arithmetic.
Control indices and array bindings are prepared by the iteration controller.
"""

from functools import cache

import warp as wp

from dpt.kernels.material_reconstruction import OPTIONS, metric_boundary_kkt

wp.set_module_options(OPTIONS)


@cache
def projected_step_tiled(materials: int, nonnegative: bool = False):
    vector = wp.types.vector(length=materials, dtype=wp.float64)

    @wp.kernel(module="unique", module_options=OPTIONS)
    def project(
        fields: wp.array(dtype=wp.float32),
        gradient: wp.array(dtype=wp.float32),
        voxels: int,
        controls: wp.array(dtype=wp.float64),
        step_index: int,
        previous: wp.array(dtype=wp.float32),
        momentum_index: int,
        use_voxel_scale: bool,
        voxel_scale: wp.array(dtype=wp.float32),
        use_diagonal: bool,
        diagonal: wp.array(dtype=wp.float64),
        trial: wp.array(dtype=wp.float32),
        slope_components: wp.array(dtype=wp.float64),
        mappings: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        step = controls[step_index]
        momentum = controls[momentum_index]
        block, lane = wp.tid()
        iterations = int(1)  # noqa: UP018, RUF046 - Mutable Warp runtime integer.
        if wp.block_dim() == 1:
            iterations = 256
        slope = wp.float64(0.0)
        maximum_mapping = wp.float64(0.0)
        for offset in range(iterations):
            voxel = block * 256 + lane + offset
            if voxel < voxels:
                local_step = step
                if use_voxel_scale:
                    local_step = step / wp.float64(voxel_scale[voxel])
                values = vector()
                ordered = vector()
                inverse_weights = vector()
                ordered_inverse = vector()
                total = wp.float64(0.0)
                maximum = wp.float64(-1.0e308)
                for material in range(materials):
                    index = material * voxels + voxel
                    weight = wp.float64(1.0)
                    if use_diagonal:
                        weight = diagonal[material]
                    inverse_weights[material] = wp.float64(1.0) / weight
                    value = wp.float64(fields[index]) - local_step * inverse_weights[
                        material
                    ] * wp.float64(gradient[index])
                    if momentum != wp.float64(0.0):
                        value += momentum * (
                            wp.float64(fields[index]) - wp.float64(previous[index])
                        )
                    values[material] = value
                    total += wp.max(value, wp.float64(0.0))
                    maximum = wp.max(maximum, value / inverse_weights[material])
                    if not wp.isfinite(value):
                        wp.atomic_or(status, 0, 2)
                threshold = wp.float64(0.0)
                if wp.static(not nonnegative) and total > wp.float64(1.0):
                    # A common shift in H*u preserves the weighted threshold
                    # problem and avoids cancellation at huge cap excursions.
                    for material in range(materials):
                        values[material] -= maximum * inverse_weights[material]
                        ordered[material] = values[material] / inverse_weights[material]
                        ordered_inverse[material] = inverse_weights[material]
                    for material in range(1, materials):
                        value = ordered[material]
                        inverse_value = ordered_inverse[material]
                        slot = material
                        while slot > 0:
                            if ordered[slot - 1] >= value:
                                break
                            ordered[slot] = ordered[slot - 1]
                            ordered_inverse[slot] = ordered_inverse[slot - 1]
                            slot -= 1
                        ordered[slot] = value
                        ordered_inverse[slot] = inverse_value
                    partial = wp.float64(0.0)
                    inverse_sum = wp.float64(0.0)
                    for material in range(materials):
                        partial += ordered[material] * ordered_inverse[material]
                        inverse_sum += ordered_inverse[material]
                        candidate = (partial - wp.float64(1.0)) / inverse_sum
                        if ordered[material] > candidate:
                            threshold = candidate
                mapping = wp.float64(0.0)
                for material in range(materials):
                    index = material * voxels + voxel
                    projected = wp.max(
                        values[material] - threshold * inverse_weights[material], wp.float64(0.0)
                    )
                    original = wp.float64(fields[index])
                    mapping = wp.max(mapping, wp.abs((projected - original) / step))
                    rounded = wp.float32(projected)
                    if not wp.isfinite(rounded):
                        wp.atomic_or(status, 0, 2)
                    trial[index] = rounded
                    slope += wp.float64(gradient[index]) * (wp.float64(rounded) - original)
                maximum_mapping = wp.max(maximum_mapping, mapping)
        wp.tile_store(slope_components, wp.tile_sum(wp.tile(slope)), offset=block)
        wp.tile_store(mappings, wp.tile_max(wp.tile(maximum_mapping)), offset=block)

    return project


@wp.kernel
def projected_metric_step_tiled(
    fields: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
    voxels: int,
    controls: wp.array(dtype=wp.float64),
    step_index: int,
    previous: wp.array(dtype=wp.float32),
    momentum_index: int,
    use_voxel_scale: bool,
    voxel_scale: wp.array(dtype=wp.float32),
    metric: wp.vec3d,
    inverse: wp.vec3d,
    trial: wp.array(dtype=wp.float32),
    slope_components: wp.array(dtype=wp.float64),
    mappings: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    step = controls[step_index]
    momentum = controls[momentum_index]
    block, lane = wp.tid()
    iterations = int(1)  # noqa: UP018, RUF046 - Mutable Warp runtime integer.
    if wp.block_dim() == 1:
        iterations = 256
    slope = wp.float64(0.0)
    maximum_mapping = wp.float64(0.0)
    for offset in range(iterations):
        voxel = block * 256 + lane + offset
        if voxel < voxels:
            local_step = step
            if use_voxel_scale:
                local_step = step / wp.float64(voxel_scale[voxel])
            field = wp.vec2d(wp.float64(fields[voxel]), wp.float64(fields[voxels + voxel]))
            centre = field
            if momentum != wp.float64(0.0):
                old = wp.vec2d(wp.float64(previous[voxel]), wp.float64(previous[voxels + voxel]))
                centre = field + momentum * (field - old)
            grad = wp.vec2d(wp.float64(gradient[voxel]), wp.float64(gradient[voxels + voxel]))
            linear = local_step * grad
            unconstrained = centre - wp.vec2d(
                inverse[0] * linear[0] + inverse[1] * linear[1],
                inverse[1] * linear[0] + inverse[2] * linear[1],
            )
            point = unconstrained
            if (
                not wp.isfinite(unconstrained[0])
                or not wp.isfinite(unconstrained[1])
                or unconstrained[0] < wp.float64(0.0)
                or unconstrained[1] < wp.float64(0.0)
                or unconstrained[0] + unconstrained[1] > wp.float64(1.0)
            ):
                # Enumerate vertices and open-edge minima, then use KKT conditions.
                # Absolute objective scores lose their ordering for a large common
                # gradient; e.g. q=(-1e20,-1e20) still has a unique cap interior point.
                denominator = (metric[0] - metric[1]) + (metric[2] - metric[1])
                cap_numerator = (metric[0] - metric[1]) * centre[0]
                cap_numerator += (metric[1] - metric[2]) * centre[1]
                cap_numerator -= local_step * (grad[0] - grad[1])
                cap_numerator += metric[2] - metric[1]
                found = bool(False)  # noqa: UP018 - Warp requires a mutable loop variable.
                for face in range(6):
                    candidate = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
                    open_edge = bool(True)  # noqa: UP018 - Explicit Warp runtime Boolean.
                    if face == 1:
                        candidate[0] = wp.float64(1.0)
                    elif face == 2:
                        candidate[1] = wp.float64(1.0)
                    elif face == 3:
                        candidate[0] = centre[0] + (metric[1] * centre[1] - linear[0]) / metric[0]
                        open_edge = candidate[0] > wp.float64(0.0) and candidate[0] < wp.float64(
                            1.0
                        )
                    elif face == 4:
                        candidate[1] = centre[1] + (metric[1] * centre[0] - linear[1]) / metric[2]
                        open_edge = candidate[1] > wp.float64(0.0) and candidate[1] < wp.float64(
                            1.0
                        )
                    elif face == 5:
                        candidate[0] = cap_numerator / denominator
                        candidate[1] = wp.float64(1.0) - candidate[0]
                        open_edge = candidate[0] > wp.float64(0.0) and candidate[0] < wp.float64(
                            1.0
                        )
                    if open_edge and metric_boundary_kkt(
                        candidate, centre, grad, local_step, metric, face
                    ):
                        point = candidate
                        found = True
                        break
                if not found:
                    wp.atomic_or(status, 0, 4)
            mapping = wp.float64(0.0)
            for material in range(2):
                index = material * voxels + voxel
                rounded = wp.float32(point[material])
                trial[index] = rounded
                slope += grad[material] * (wp.float64(rounded) - field[material])
                mapping = wp.max(mapping, wp.abs((point[material] - field[material]) / step))
                if not wp.isfinite(rounded) or not wp.isfinite(linear[material]):
                    wp.atomic_or(status, 0, 2)
            maximum_mapping = wp.max(maximum_mapping, mapping)
    wp.tile_store(slope_components, wp.tile_sum(wp.tile(slope)), offset=block)
    wp.tile_store(mappings, wp.tile_max(wp.tile(maximum_mapping)), offset=block)


@wp.kernel
def projected_metric_nonnegative_step_tiled(
    fields: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
    voxels: int,
    controls: wp.array(dtype=wp.float64),
    step_index: int,
    previous: wp.array(dtype=wp.float32),
    momentum_index: int,
    use_voxel_scale: bool,
    voxel_scale: wp.array(dtype=wp.float32),
    metric: wp.vec3d,
    inverse: wp.vec3d,
    trial: wp.array(dtype=wp.float32),
    slope_components: wp.array(dtype=wp.float64),
    mappings: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    step = controls[step_index]
    momentum = controls[momentum_index]
    block, lane = wp.tid()
    iterations = int(1)  # noqa: UP018, RUF046 - Mutable Warp runtime integer.
    if wp.block_dim() == 1:
        iterations = 256
    slope = wp.float64(0.0)
    maximum_mapping = wp.float64(0.0)
    for offset in range(iterations):
        voxel = block * 256 + lane + offset
        if voxel < voxels:
            local_step = step
            if use_voxel_scale:
                local_step = step / wp.float64(voxel_scale[voxel])
            field = wp.vec2d(wp.float64(fields[voxel]), wp.float64(fields[voxels + voxel]))
            centre = field
            if momentum != wp.float64(0.0):
                old = wp.vec2d(wp.float64(previous[voxel]), wp.float64(previous[voxels + voxel]))
                centre = field + momentum * (field - old)
            grad = wp.vec2d(wp.float64(gradient[voxel]), wp.float64(gradient[voxels + voxel]))
            linear = local_step * grad
            point = centre - wp.vec2d(
                inverse[0] * linear[0] + inverse[1] * linear[1],
                inverse[1] * linear[0] + inverse[2] * linear[1],
            )
            if (
                not wp.isfinite(point[0])
                or not wp.isfinite(point[1])
                or point[0] < wp.float64(0.0)
                or point[1] < wp.float64(0.0)
            ):
                found = bool(False)  # noqa: UP018 - Warp requires a mutable loop variable.
                for face_index in range(3):
                    candidate = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
                    face = int(0)  # noqa: UP018, RUF046 - Mutable Warp runtime integer.
                    feasible = bool(True)  # noqa: UP018 - Explicit Warp runtime Boolean.
                    if face_index == 1:
                        face = 3
                        candidate[0] = centre[0] + (metric[1] * centre[1] - linear[0]) / metric[0]
                        feasible = candidate[0] > wp.float64(0.0)
                    elif face_index == 2:
                        face = 4
                        candidate[1] = centre[1] + (metric[1] * centre[0] - linear[1]) / metric[2]
                        feasible = candidate[1] > wp.float64(0.0)
                    if feasible and metric_boundary_kkt(
                        candidate, centre, grad, local_step, metric, face
                    ):
                        point = candidate
                        found = True
                        break
                if not found:
                    wp.atomic_or(status, 0, 4)
            mapping = wp.float64(0.0)
            for material in range(2):
                index = material * voxels + voxel
                rounded = wp.float32(point[material])
                trial[index] = rounded
                slope += grad[material] * (wp.float64(rounded) - field[material])
                mapping = wp.max(mapping, wp.abs((point[material] - field[material]) / step))
                if not wp.isfinite(rounded) or not wp.isfinite(linear[material]):
                    wp.atomic_or(status, 0, 2)
            maximum_mapping = wp.max(maximum_mapping, mapping)
    wp.tile_store(slope_components, wp.tile_sum(wp.tile(slope)), offset=block)
    wp.tile_store(mappings, wp.tile_max(wp.tile(maximum_mapping)), offset=block)


@wp.kernel
def copy_f32(
    source: wp.array(dtype=wp.float32),
    destination: wp.array(dtype=wp.float32),
    count: int,
):
    index = wp.tid()
    if index < count:
        destination[index] = source[index]


@wp.kernel
def zero_f32(destination: wp.array(dtype=wp.float32), count: int):
    index = wp.tid()
    if index < count:
        destination[index] = wp.float32(0)


@wp.kernel
def copy_f64(
    source: wp.array(dtype=wp.float64),
    destination: wp.array(dtype=wp.float64),
    count: int,
):
    index = wp.tid()
    if index < count:
        destination[index] = source[index]


@wp.kernel
def zero_f64(destination: wp.array(dtype=wp.float64), count: int):
    index = wp.tid()
    if index < count:
        destination[index] = wp.float64(0)


@wp.kernel
def copy_i32(
    source: wp.array(dtype=wp.int32),
    destination: wp.array(dtype=wp.int32),
    count: int,
):
    index = wp.tid()
    if index < count:
        destination[index] = source[index]


@wp.kernel
def zero_i32(destination: wp.array(dtype=wp.int32), count: int):
    index = wp.tid()
    if index < count:
        destination[index] = wp.int32(0)
