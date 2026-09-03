"""Constrained vector updates and physical-space penalties; no imaging physics."""

from functools import cache

import warp as wp

OPTIONS = {"fast_math": False, "fuse_fp": True, "enable_backward": False}
wp.set_module_options(OPTIONS)
METRIC_ROUNDOFF = wp.constant(wp.float64(16.0 * 2.0**-52))


@cache
def add_paths(precision: str = "float32"):
    value_dtype = wp.float64 if precision == "float64" else wp.float32

    @wp.kernel(module="unique", module_options=OPTIONS)
    def add(
        source: wp.array(dtype=value_dtype),
        total: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        index = wp.tid()
        value = total[index] + wp.float64(source[index])
        total[index] = value
        if not wp.isfinite(value):
            wp.atomic_or(status, 0, 2)

    return add


@wp.kernel
def narrow_paths(
    source: wp.array(dtype=wp.float64),
    destination: wp.array(dtype=wp.float32),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    value = wp.float32(source[index])
    destination[index] = value
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, 2)


@wp.kernel
def narrow_data_gradient(
    source: wp.array(dtype=wp.float64),
    destination: wp.array(dtype=wp.float32),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    previous = destination[index]
    value = wp.float32(source[index])
    destination[index] = value
    if not wp.isfinite(previous) or not wp.isfinite(value):
        wp.atomic_or(status, 0, 2)


@wp.func
def metric_boundary_kkt(
    point: wp.vec2d,
    fields: wp.vec2d,
    gradient: wp.vec2d,
    step: wp.float64,
    metric: wp.vec3d,
    face: int,
) -> bool:
    displacement = point - fields
    q = step * gradient
    v0 = metric[0] * displacement[0] + metric[1] * displacement[1] + q[0]
    v1 = metric[1] * displacement[0] + metric[2] * displacement[1] + q[1]
    # Form the cap-tangent derivative without subtracting two huge common
    # normal components. Its error allowance must also exclude that common
    # component, or nearly tied vertices could incorrectly replace the cap.
    difference = (metric[0] - metric[1]) * displacement[0]
    difference += (metric[1] - metric[2]) * displacement[1]
    difference += step * (gradient[0] - gradient[1])
    epsilon = METRIC_ROUNDOFF
    # A small displacement still inherits rounding from constructing point
    # near a nonzero field and then subtracting the field. Bound those terms
    # separately; a displacement-only allowance vanishes as step shrinks and
    # can reject a correctly solved active edge indefinitely.
    magnitude0 = wp.abs(point[0]) + wp.abs(fields[0])
    magnitude1 = wp.abs(point[1]) + wp.abs(fields[1])
    tolerance0 = epsilon * (
        wp.abs(metric[0]) * magnitude0 + wp.abs(metric[1]) * magnitude1 + wp.abs(q[0])
    )
    tolerance1 = epsilon * (
        wp.abs(metric[1]) * magnitude0 + wp.abs(metric[2]) * magnitude1 + wp.abs(q[1])
    )
    tangent_tolerance = epsilon * (
        wp.abs(metric[0] - metric[1]) * magnitude0
        + wp.abs(metric[1] - metric[2]) * magnitude1
        + wp.abs(step * (gradient[0] - gradient[1]))
    )
    # The cap minimiser can round to an endpoint even when its exact tangent
    # displacement is nonzero. Include that endpoint rounding in the vertex
    # certificate; otherwise a stiff cap curvature leaves neither the open
    # edge nor the endpoint certifiable.
    tangent_tolerance += (
        epsilon
        * wp.abs(metric[0] + metric[2] - wp.float64(2.0) * metric[1])
        * (wp.abs(point[0]) + wp.abs(point[1]))
    )
    if face == 0:  # origin
        return v0 >= -tolerance0 and v1 >= -tolerance1
    if face == 1:  # (1,0)
        return v0 <= tolerance0 and difference <= tangent_tolerance
    if face == 2:  # (0,1)
        return v1 <= tolerance1 and difference >= -tangent_tolerance
    if face == 3:  # open horizontal edge
        return wp.abs(v0) <= tolerance0 and v1 >= -tolerance1
    if face == 4:  # open vertical edge
        return wp.abs(v1) <= tolerance1 and v0 >= -tolerance0
    # open sum-one edge
    # The cap minimiser is solved analytically before this certificate. Rechecking
    # its cancelled tangent residual can reject a correct non-dyadic solution.
    return v0 <= tolerance0


@wp.kernel
def projected_metric_step(
    fields: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
    voxels: int,
    step: wp.float64,
    previous: wp.array(dtype=wp.float32),
    momentum: wp.float64,
    use_voxel_scale: bool,
    voxel_scale: wp.array(dtype=wp.float32),
    metric: wp.vec3d,
    inverse: wp.vec3d,
    trial: wp.array(dtype=wp.float32),
    slope_components: wp.array(dtype=wp.float64),
    mappings: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    voxel = wp.tid()
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
                open_edge = candidate[0] > wp.float64(0.0) and candidate[0] < wp.float64(1.0)
            elif face == 4:
                candidate[1] = centre[1] + (metric[1] * centre[0] - linear[1]) / metric[2]
                open_edge = candidate[1] > wp.float64(0.0) and candidate[1] < wp.float64(1.0)
            elif face == 5:
                candidate[0] = cap_numerator / denominator
                candidate[1] = wp.float64(1.0) - candidate[0]
                open_edge = candidate[0] > wp.float64(0.0) and candidate[0] < wp.float64(1.0)
            if open_edge and metric_boundary_kkt(candidate, centre, grad, local_step, metric, face):
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
        slope_components[index] = grad[material] * (wp.float64(rounded) - field[material])
        mapping = wp.max(mapping, wp.abs((point[material] - field[material]) / step))
        if not wp.isfinite(rounded) or not wp.isfinite(linear[material]):
            wp.atomic_or(status, 0, 2)
    mappings[voxel] = mapping


@wp.kernel
def max_partials(
    source: wp.array(dtype=wp.float64),
    count: int,
    destination: wp.array(dtype=wp.float64),
):
    block, lane = wp.tid()
    index = block * 256 + lane
    # This reducer also checks negative metric scales to recover their minimum.
    # Zero is neutral only for the nonnegative mapping diagnostic.
    value = wp.float64(-wp.inf)
    if wp.block_dim() == 1:
        # Warp launches CPU tiles with one lane. That lane must consume the
        # whole logical tile before the host advances to the next tree level.
        for offset in range(256):
            if index + offset < count:
                value = wp.max(value, source[index + offset])
    elif index < count:
        value = source[index]
    result = wp.tile_max(wp.tile(value))
    wp.tile_store(destination, result, offset=block)


@wp.kernel
def scalar_sum(
    left: wp.array(dtype=wp.float64),
    right: wp.array(dtype=wp.float64),
    out: wp.array(dtype=wp.float64),
):
    out[0] = left[0] + right[0]


@cache
def projected_step_tiled(materials: int, nonnegative: bool = False):
    vector = wp.types.vector(length=materials, dtype=wp.float64)

    @wp.kernel(module="unique", module_options=OPTIONS)
    def project(
        fields: wp.array(dtype=wp.float32),
        gradient: wp.array(dtype=wp.float32),
        voxels: int,
        step: wp.float64,
        previous: wp.array(dtype=wp.float32),
        momentum: wp.float64,
        use_voxel_scale: bool,
        voxel_scale: wp.array(dtype=wp.float32),
        use_diagonal: bool,
        diagonal: wp.array(dtype=wp.float64),
        trial: wp.array(dtype=wp.float32),
        slope_components: wp.array(dtype=wp.float64),
        mappings: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
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
    step: wp.float64,
    previous: wp.array(dtype=wp.float32),
    momentum: wp.float64,
    use_voxel_scale: bool,
    voxel_scale: wp.array(dtype=wp.float32),
    metric: wp.vec3d,
    inverse: wp.vec3d,
    trial: wp.array(dtype=wp.float32),
    slope_components: wp.array(dtype=wp.float64),
    mappings: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
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
    step: wp.float64,
    previous: wp.array(dtype=wp.float32),
    momentum: wp.float64,
    use_voxel_scale: bool,
    voxel_scale: wp.array(dtype=wp.float32),
    metric: wp.vec3d,
    inverse: wp.vec3d,
    trial: wp.array(dtype=wp.float32),
    slope_components: wp.array(dtype=wp.float64),
    mappings: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
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
def secant_products_tiled(
    count: int,
    fields: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
    previous_fields: wp.array(dtype=wp.float32),
    previous_gradient: wp.array(dtype=wp.float32),
    voxels: int,
    coupled_metric: bool,
    metric: wp.vec3d,
    use_voxel_scale: bool,
    voxel_scale: wp.array(dtype=wp.float32),
    use_diagonal: bool,
    diagonal: wp.array(dtype=wp.float64),
    numerator: wp.array(dtype=wp.float64),
    denominator: wp.array(dtype=wp.float64),
):
    block, lane = wp.tid()
    iterations = int(1)  # noqa: UP018, RUF046 - Mutable Warp runtime integer.
    if wp.block_dim() == 1:
        iterations = 256
    numerator_sum = wp.float64(0.0)
    denominator_sum = wp.float64(0.0)
    for offset in range(iterations):
        index = block * 256 + lane + offset
        if index < count:
            displacement = wp.float64(fields[index]) - wp.float64(previous_fields[index])
            gradient_change = wp.float64(gradient[index]) - wp.float64(previous_gradient[index])
            metric_displacement = displacement
            if coupled_metric:
                voxel = index % voxels
                first = wp.float64(fields[voxel]) - wp.float64(previous_fields[voxel])
                second = wp.float64(fields[voxels + voxel]) - wp.float64(
                    previous_fields[voxels + voxel]
                )
                if index < voxels:
                    metric_displacement = metric[0] * first + metric[1] * second
                else:
                    metric_displacement = metric[1] * first + metric[2] * second
            if use_diagonal:
                metric_displacement = displacement * diagonal[index // voxels]
            if use_voxel_scale:
                metric_displacement *= wp.float64(voxel_scale[index % voxels])
            numerator_sum += displacement * metric_displacement
            denominator_sum += displacement * gradient_change
    wp.tile_store(numerator, wp.tile_sum(wp.tile(numerator_sum)), offset=block)
    wp.tile_store(denominator, wp.tile_sum(wp.tile(denominator_sum)), offset=block)


@wp.func
def roughness(
    difference: wp.float64, coefficient: wp.float64, delta: wp.float64, huber: bool
) -> wp.vec2d:
    magnitude = wp.abs(difference)
    if huber and magnitude > delta:
        derivative = coefficient * wp.sign(difference) * delta
        penalty = coefficient * delta * (magnitude - wp.float64(0.5) * delta)
        return wp.vec2d(penalty, derivative)
    return wp.vec2d(
        wp.float64(0.5) * coefficient * difference * difference, coefficient * difference
    )


@wp.kernel
def regularise_tiled(
    count: int,
    fields: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    coefficients: wp.array(dtype=wp.float64),
    huber: bool,
    delta: wp.vec3d,
    differentiate: bool,
    gradient: wp.array(dtype=wp.float32),
    components: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    block, lane = wp.tid()
    iterations = int(1)  # noqa: UP018, RUF046 - Mutable Warp runtime integer.
    if wp.block_dim() == 1:
        iterations = 256
    penalty_sum = wp.float64(0.0)
    for offset in range(iterations):
        index = block * 256 + lane + offset
        if index < count:
            voxel = index % (nx * ny * nz)
            x = voxel % nx
            y = (voxel // nx) % ny
            z = voxel // (nx * ny)
            material = index // (nx * ny * nz)
            coefficient_x = coefficients[3 * material]
            coefficient_y = coefficients[3 * material + 1]
            coefficient_z = coefficients[3 * material + 2]
            value = wp.float64(fields[index])
            derivative = wp.float64(0.0)
            penalty = wp.float64(0.0)
            if x > 0:
                derivative += roughness(
                    value - wp.float64(fields[index - 1]), coefficient_x, delta[0], huber
                )[1]
            if x + 1 < nx:
                difference = value - wp.float64(fields[index + 1])
                contribution = roughness(difference, coefficient_x, delta[0], huber)
                derivative += contribution[1]
                penalty += contribution[0]
            if y > 0:
                derivative += roughness(
                    value - wp.float64(fields[index - nx]), coefficient_y, delta[1], huber
                )[1]
            if y + 1 < ny:
                difference = value - wp.float64(fields[index + nx])
                contribution = roughness(difference, coefficient_y, delta[1], huber)
                derivative += contribution[1]
                penalty += contribution[0]
            if z > 0:
                derivative += roughness(
                    value - wp.float64(fields[index - nx * ny]), coefficient_z, delta[2], huber
                )[1]
            if z + 1 < nz:
                difference = value - wp.float64(fields[index + nx * ny])
                contribution = roughness(difference, coefficient_z, delta[2], huber)
                derivative += contribution[1]
                penalty += contribution[0]
            penalty_sum += penalty
            if differentiate:
                result = wp.float32(wp.float64(gradient[index]) + derivative)
                gradient[index] = result
                if not wp.isfinite(result):
                    wp.atomic_or(status, 0, 2)
    wp.tile_store(components, wp.tile_sum(wp.tile(penalty_sum)), offset=block)


@wp.kernel
def check_voxel_metric_scale_tiled(
    count: int,
    scale: wp.array(dtype=wp.float32),
    positive: wp.array(dtype=wp.float64),
    negative: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    block, lane = wp.tid()
    iterations = int(1)  # noqa: UP018, RUF046 - Mutable Warp runtime integer.
    if wp.block_dim() == 1:
        iterations = 256
    positive_max = wp.float64(-wp.inf)
    negative_max = wp.float64(-wp.inf)
    for offset in range(iterations):
        index = block * 256 + lane + offset
        if index < count:
            value = wp.float64(scale[index])
            if not wp.isfinite(value) or value <= wp.float64(0.0):
                wp.atomic_or(status, 0, 2)
                value = wp.float64(0.0)
            positive_max = wp.max(positive_max, value)
            negative_max = wp.max(negative_max, -value)
    wp.tile_store(positive, wp.tile_max(wp.tile(positive_max)), offset=block)
    wp.tile_store(negative, wp.tile_max(wp.tile(negative_max)), offset=block)


@wp.kernel
def gradient_magnitudes_tiled(
    gradient: wp.array(dtype=wp.float32),
    materials: int,
    voxels: int,
    magnitudes: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    block, lane = wp.tid()
    iterations = int(1)  # noqa: UP018, RUF046 - Mutable Warp runtime integer.
    if wp.block_dim() == 1:
        iterations = 256
    maximum = wp.float64(0.0)
    for offset in range(iterations):
        voxel = block * 256 + lane + offset
        if voxel < voxels:
            value = wp.float64(0.0)
            for material in range(materials):
                entry = wp.float64(gradient[material * voxels + voxel])
                if not wp.isfinite(entry):
                    wp.atomic_or(status, 0, 2)
                value = wp.max(value, wp.abs(entry))
            maximum = wp.max(maximum, value)
    wp.tile_store(magnitudes, wp.tile_max(wp.tile(maximum)), offset=block)


@wp.kernel
def trial_controls(
    maximum: wp.array(dtype=wp.float64),
    slope: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float64),
):
    output[0] = maximum[0]
    output[1] = slope[0]
    output[2] = wp.float64(status[0])
