"""Matched log-scaled mixed-shell arithmetic for native transport."""

# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
from __future__ import annotations

import warp as wp

from . import bound_compton_kernels as bound
from .physical_kernels import PhysicalParameters
from .sampling_kernels import transport_random4

MASS = wp.constant(wp.float64(510.99895069))

# Match the original bound transport module's explicit-score differentiation.
wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})


@wp.func
def impulse_mixed_log_occupancy_ratio(first: wp.float64, second: wp.float64) -> wp.float64:
    ratio = first / second
    if ratio > wp.float64(0.0) and wp.isfinite(ratio):
        return wp.log(ratio)
    return wp.log(first) - wp.log(second)


@wp.func
def impulse_mixed_profile_log(momentum: wp.float64, height: wp.float64) -> wp.float64:
    amplitude = height * abs(momentum)
    return -wp.float64(2.0) * amplitude * (wp.float64(1.0) + amplitude)


@wp.func
def impulse_mixed_profile_log_density(momentum: wp.float64, height: wp.float64) -> wp.float64:
    amplitude = height * abs(momentum)
    return (
        wp.log(height)
        + bound.impulse_tail_log1p(wp.float64(2.0) * amplitude)
        + impulse_mixed_profile_log(momentum, height)
    )


@wp.func
def impulse_mixed_mass_pair(
    energy: wp.float64, cosine: wp.float64, binding: wp.float64, height: wp.float64
) -> wp.vec2d:
    if energy <= binding or cosine >= wp.float64(1.0):
        return wp.vec2d(-wp.float64(wp.inf), wp.float64(0.0))
    _mixed_pair_0 = bound.impulse_tail_two_sum(wp.float64(1.0), -cosine)
    transfer = _mixed_pair_0[0]
    transfer_error = _mixed_pair_0[1]
    _mixed_pair_1 = bound.impulse_tail_kinematic_pair(energy, transfer, binding, transfer_error)
    numerator = _mixed_pair_1[0]
    q = _mixed_pair_1[1]
    momentum = numerator / q
    if momentum < wp.float64(0.0):
        log_profile = impulse_mixed_profile_log(momentum, height)
        if not wp.isfinite(log_profile):
            return wp.vec2d(-wp.float64(wp.inf), wp.float64(0.0))
        span = wp.float64(1.0) + momentum
        if span > wp.float64(0.0):
            log_size = (
                wp.log(wp.float64(2.0))
                + wp.log(height)
                + wp.log(span)
                + bound.impulse_tail_log1p(height * (wp.float64(2.0) - span))
            )
            if log_size > wp.float64(6.0):
                _mixed_pair_2 = bound.impulse_tail_kinematic_pair(
                    energy, wp.float64(1.0) - cosine, binding, wp.float64(0.0)
                )
                numerator = _mixed_pair_2[0]
                q = _mixed_pair_2[1]
                p_e = (
                    (wp.float64(2.0) * energy - binding)
                    * (wp.float64(1.0) - cosine)
                    * (wp.float64(1.0) / (MASS * q) - numerator / (q * q * q))
                )
                return wp.vec2d(
                    log_profile - wp.log(wp.float64(2.0)),
                    wp.float64(2.0)
                    * height
                    * ((wp.float64(1.0) - wp.float64(2.0) * height * momentum) * p_e),
                )
        state = bound.impulse_tail_state_at_cosine(energy, cosine, binding, height)
        return wp.vec2d(state.log_mass, state.score)
    mass = bound.impulse_profile_interval_mass(-wp.float64(1.0), momentum, height)
    _mixed_pair_3 = bound.impulse_tail_kinematic_pair(
        energy, wp.float64(1.0) - cosine, binding, wp.float64(0.0)
    )
    numerator = _mixed_pair_3[0]
    q = _mixed_pair_3[1]
    p_e = (
        (wp.float64(2.0) * energy - binding)
        * (wp.float64(1.0) - cosine)
        * (wp.float64(1.0) / (MASS * q) - numerator / (q * q * q))
    )
    log_mass = wp.log(mass)
    return wp.vec2d(
        log_mass, wp.exp(impulse_mixed_profile_log_density(momentum, height) - log_mass) * p_e
    )


@wp.func
def impulse_mixed_back_rate(
    energy: wp.float64, binding: wp.float64, height: wp.float64
) -> wp.float64:
    if bound.impulse_momentum_max(energy, -wp.float64(1.0), binding) >= wp.float64(0.0):
        return wp.float64(0.0)
    return bound.impulse_tail_state(energy, wp.float64(2.0), binding, height).rate


@wp.func
def impulse_mixed_envelope_integral(rate: wp.float64) -> wp.float64:
    return (
        wp.float64(2.0) if rate == wp.float64(0.0) else bound.impulse_tail_envelope_integral(rate)
    )


@wp.func
def impulse_mixed_back_relative(
    energy: wp.float64,
    binding: wp.float64,
    height: wp.float64,
    reference_binding: wp.float64,
    reference_height: wp.float64,
) -> wp.vec2d:
    _mixed_pair_0 = impulse_mixed_mass_pair(energy, -wp.float64(1.0), binding, height)
    log_mass = _mixed_pair_0[0]
    score = _mixed_pair_0[1]
    if log_mass == -wp.float64(wp.inf):
        return wp.vec2d(-wp.float64(wp.inf), wp.float64(0.0))
    if bound.impulse_momentum_max(energy, -wp.float64(1.0), binding) < wp.float64(
        0.0
    ) and bound.impulse_momentum_max(energy, -wp.float64(1.0), reference_binding) < wp.float64(0.0):
        state = bound.impulse_tail_state(energy, wp.float64(2.0), binding, height)
        reference = bound.impulse_tail_state(
            energy, wp.float64(2.0), reference_binding, reference_height
        )
        return bound.impulse_tail_relative_shell(energy, wp.float64(2.0), state, reference)
    _mixed_pair_1 = impulse_mixed_mass_pair(
        energy, -wp.float64(1.0), reference_binding, reference_height
    )
    ref_log_mass = _mixed_pair_1[0]
    ref_score = _mixed_pair_1[1]
    return wp.vec2d(log_mass - ref_log_mass, score - ref_score)


@wp.func
def impulse_mixed_angle_relative(
    energy: wp.float64,
    cosine: wp.float64,
    binding: wp.float64,
    height: wp.float64,
    reference_binding: wp.float64,
    reference_height: wp.float64,
) -> wp.vec2d:
    _mixed_pair_0 = impulse_mixed_back_relative(
        energy, binding, height, reference_binding, reference_height
    )
    back_log = _mixed_pair_0[0]
    back_centre = _mixed_pair_0[1]
    if back_log == -wp.float64(wp.inf):
        return wp.vec2d(-wp.float64(wp.inf), wp.float64(0.0))
    if bound.impulse_momentum_max(energy, -wp.float64(1.0), binding) < wp.float64(0.0):
        state = bound.impulse_tail_state(energy, wp.float64(2.0), binding, height)
        _mixed_pair_1 = bound.impulse_tail_angle_relative_at_transfer(
            energy, state, cosine + wp.float64(1.0), wp.float64(1.0) - cosine
        )
        angle_log = _mixed_pair_1[0]
        angle_centre = _mixed_pair_1[1]
        return wp.vec2d(back_log + angle_log, back_centre + angle_centre)
    _mixed_pair_2 = impulse_mixed_mass_pair(energy, cosine, binding, height)
    local_log = _mixed_pair_2[0]
    local_score = _mixed_pair_2[1]
    _mixed_pair_3 = impulse_mixed_mass_pair(
        energy, -wp.float64(1.0), reference_binding, reference_height
    )
    reference_log = _mixed_pair_3[0]
    reference_score = _mixed_pair_3[1]
    return wp.vec2d(local_log - reference_log, local_score - reference_score)


@wp.func
def impulse_mixed_local_mass_relative(
    energy: wp.float64,
    cosine: wp.float64,
    binding: wp.float64,
    height: wp.float64,
    reference_binding: wp.float64,
    reference_height: wp.float64,
) -> wp.vec2d:
    _mixed_pair_0 = impulse_mixed_mass_pair(energy, cosine, binding, height)
    local_log = _mixed_pair_0[0]
    local_score = _mixed_pair_0[1]
    if local_log == -wp.float64(wp.inf):
        return wp.vec2d(-wp.float64(wp.inf), wp.float64(0.0))
    if bound.impulse_momentum_max(energy, cosine, binding) < wp.float64(
        0.0
    ) and bound.impulse_momentum_max(energy, cosine, reference_binding) < wp.float64(0.0):
        state = bound.impulse_tail_state_at_cosine(energy, cosine, binding, height)
        reference = bound.impulse_tail_state_at_cosine(
            energy, cosine, reference_binding, reference_height
        )
        return bound.impulse_tail_relative_shell(energy, wp.float64(1.0) - cosine, state, reference)
    _mixed_pair_1 = impulse_mixed_mass_pair(energy, cosine, reference_binding, reference_height)
    reference_log = _mixed_pair_1[0]
    reference_score = _mixed_pair_1[1]
    return wp.vec2d(local_log - reference_log, local_score - reference_score)


@wp.func
def impulse_mixed_reference(energy: wp.float64, material: int, physical: PhysicalParameters) -> int:
    selected = int(-1)  # noqa: UP018, RUF046 - mutable Warp index
    best = -wp.float64(wp.inf)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        binding, height = (
            wp.float64(physical.compton_binding[shell]),
            wp.float64(physical.compton_height[shell]),
        )
        if energy > binding:
            mass = impulse_mixed_mass_pair(energy, -wp.float64(1.0), binding, height)
            if wp.isfinite(mass[0]):
                rate = impulse_mixed_back_rate(energy, binding, height)
                candidate = (
                    wp.log(physical.compton_occupancy[shell])
                    + mass[0]
                    + wp.log(impulse_mixed_envelope_integral(rate))
                )
                if candidate > best:
                    best = candidate
                    selected = shell
    return selected


@wp.func
def impulse_use_mixed_conditioning(
    energy: wp.float64, material: int, physical: PhysicalParameters
) -> bool:
    positive = int(0)  # noqa: UP018, RUF046 - mutable Warp flag
    tiny_occupancy = int(0)  # noqa: UP018, RUF046 - mutable Warp flag
    overflow = int(0)  # noqa: UP018, RUF046 - mutable Warp flag
    finite_component = int(0)  # noqa: UP018, RUF046 - mutable Warp flag
    best = -wp.float64(wp.inf)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        binding, height = (
            wp.float64(physical.compton_binding[shell]),
            wp.float64(physical.compton_height[shell]),
        )
        if energy > binding:
            if physical.compton_occupancy[shell] < wp.float64(1e-280):
                tiny_occupancy = int(1)  # noqa: UP018, RUF046
            if bound.impulse_momentum_max(energy, -wp.float64(1.0), binding) >= wp.float64(0.0):
                positive = int(1)  # noqa: UP018, RUF046
            mass = impulse_mixed_mass_pair(energy, -wp.float64(1.0), binding, height)
            if wp.isfinite(mass[0]):
                finite_component = int(1)  # noqa: UP018, RUF046
                if bound.impulse_momentum_max(energy, -wp.float64(1.0), binding) < wp.float64(0.0):
                    pair = bound.impulse_tail_kinematic_pair(
                        energy, wp.float64(2.0), binding, wp.float64(0.0)
                    )
                    q = pair[1]
                    span = (
                        energy
                        * (energy - binding)
                        * wp.float64(2.0)
                        / q
                        * (wp.float64(1.0) / MASS + wp.float64(2.0) / (q + binding))
                    )
                    log_size = (
                        wp.log(wp.float64(2.0))
                        + wp.log(height)
                        + wp.log(span)
                        + bound.impulse_tail_log1p(height * (wp.float64(2.0) - span))
                    )
                    if log_size > wp.float64(709.782712893384):
                        overflow = int(1)  # noqa: UP018, RUF046
            else:
                overflow = int(1)  # noqa: UP018, RUF046
            best = wp.max(best, wp.log(physical.compton_occupancy[shell]) + mass[0])
    if overflow != 0 and finite_component != 0:
        return True
    if best >= wp.float64(-644.7238260383328):
        return False
    if positive != 0:
        return True
    return tiny_occupancy != 0


@wp.func
def impulse_mixed_coordinate(coordinate: wp.float64) -> wp.vec3d:
    complement_coordinate = wp.float64(1.0) - coordinate
    left = coordinate * coordinate * coordinate * coordinate
    right = (
        complement_coordinate
        * complement_coordinate
        * complement_coordinate
        * complement_coordinate
    )
    total = left + right
    jacobian = (
        wp.float64(4.0)
        * coordinate
        * coordinate
        * coordinate
        * complement_coordinate
        * complement_coordinate
        * complement_coordinate
        / (total * total)
    )
    return wp.vec3d(left / total, right / total, jacobian)


@wp.func
def impulse_mixed_normalisation(
    energy: wp.float64, material: int, physical: PhysicalParameters
) -> wp.vec4d:
    reference_index = impulse_mixed_reference(energy, material, physical)
    if reference_index < 0:
        return wp.vec4d(-wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    ref_binding, ref_height = (
        wp.float64(physical.compton_binding[reference_index]),
        wp.float64(physical.compton_height[reference_index]),
    )
    ref_pair = impulse_mixed_mass_pair(energy, -wp.float64(1.0), ref_binding, ref_height)
    values, derivatives = wp.float64(0.0), wp.float64(0.0)
    radius = wp.float64(1.0 / 64.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        binding, height = (
            wp.float64(physical.compton_binding[shell]),
            wp.float64(physical.compton_height[shell]),
        )
        if energy > binding:
            log_occ = impulse_mixed_log_occupancy_ratio(
                physical.compton_occupancy[shell], physical.compton_occupancy[reference_index]
            )
            back = impulse_mixed_back_relative(energy, binding, height, ref_binding, ref_height)
            back_weight = wp.exp(log_occ + back[0])
            if back_weight > wp.float64(0.0):
                upper = bound.impulse_momentum_max(energy, -wp.float64(1.0), binding)
                if upper < wp.float64(0.0):
                    state = bound.impulse_tail_state(energy, wp.float64(2.0), binding, height)
                    weight = back_weight * impulse_mixed_envelope_integral(state.rate)
                    for panel in range(32):
                        midpoint = (wp.float64(panel) + wp.float64(0.5)) / wp.float64(32.0)
                        for node in range(8):
                            for sign in range(2):
                                x = midpoint + (
                                    wp.float64(2 * sign) - wp.float64(1.0)
                                ) * radius * bound.impulse_gauss16_node(node)
                                coordinates = impulse_mixed_coordinate(x)
                                distance = bound.impulse_tail_proposal_distance(
                                    coordinates[0], coordinates[1], state.rate
                                )
                                angle = bound.impulse_tail_angle_relative(energy, state, distance)
                                proposal_ratio = wp.exp(angle[0] + state.rate * distance)
                                kn = bound.impulse_kn_transfer_pair(
                                    energy, wp.float64(2.0) - distance
                                )
                                coefficient = (
                                    weight
                                    * radius
                                    * bound.impulse_gauss16_weight(node)
                                    * coordinates[2]
                                    * proposal_ratio
                                )
                                values += coefficient * kn[0]
                                derivatives += coefficient * (kn[1] + kn[0] * (back[1] + angle[1]))
                else:
                    for segment in range(2):
                        lo, hi = -wp.float64(1.0), wp.float64(0.0)
                        if segment == 1:
                            lo, hi = wp.float64(0.0), upper
                        interval_mass = bound.impulse_profile_interval_mass(lo, hi, height)
                        if interval_mass > wp.float64(0.0):
                            relative_mass = wp.exp(log_occ + wp.log(interval_mass) - ref_pair[0])
                            if relative_mass > wp.float64(0.0):
                                for panel in range(32):
                                    midpoint = (wp.float64(panel) + wp.float64(0.5)) / wp.float64(
                                        32.0
                                    )
                                    for node in range(8):
                                        for sign in range(2):
                                            x = midpoint + (
                                                wp.float64(2 * sign) - wp.float64(1.0)
                                            ) * radius * bound.impulse_gauss16_node(node)
                                            coordinates = impulse_mixed_coordinate(x)
                                            momentum = bound.impulse_profile_interval_inverse_pair(
                                                coordinates[0], coordinates[1], lo, hi, height
                                            )
                                            lower = bound.impulse_minimum_transfer(
                                                energy, binding, momentum
                                            )
                                            value = bound.impulse_kn_interval_pair(energy, lower)
                                            moving_limit = lower * (
                                                wp.float64(1.0) / energy
                                                + wp.float64(1.0) / (energy - binding)
                                            )
                                            derivative = (
                                                value[1]
                                                + bound.impulse_kn_transfer_pair(energy, lower)[0]
                                                * moving_limit
                                            )
                                            coefficient = (
                                                relative_mass
                                                * radius
                                                * bound.impulse_gauss16_weight(node)
                                                * coordinates[2]
                                            )
                                            values += coefficient * value[0]
                                            derivatives += coefficient * (
                                                derivative - ref_pair[1] * value[0]
                                            )
    return wp.vec4d(
        wp.float64(reference_index), bound.TWO_PI * values, bound.TWO_PI * derivatives, ref_pair[1]
    )


@wp.func
def impulse_mixed_raw_normalisation_pair(
    energy: wp.float64, material: int, physical: PhysicalParameters
) -> wp.vec2d:
    normalisation = impulse_mixed_normalisation(energy, material, physical)
    reference_index = int(normalisation[0])
    if reference_index < 0 or normalisation[1] <= wp.float64(0.0):
        return wp.vec2d(wp.float64(0.0))
    ref_pair = impulse_mixed_mass_pair(
        energy,
        -wp.float64(1.0),
        wp.float64(physical.compton_binding[reference_index]),
        wp.float64(physical.compton_height[reference_index]),
    )
    log_scale = wp.log(physical.compton_occupancy[reference_index]) + ref_pair[0]
    log_value = wp.log(normalisation[1]) + log_scale
    score = normalisation[3] + normalisation[2] / normalisation[1]
    derivative = wp.float64(0.0)
    if score != wp.float64(0.0):
        derivative = wp.exp(wp.log(wp.abs(score)) + log_value)
        if score < wp.float64(0.0):
            derivative = -derivative
    return wp.vec2d(wp.exp(log_value), derivative)


@wp.func
def impulse_mixed_phase_pdf(
    energy: wp.float64, cosine: wp.float64, material: int, physical: PhysicalParameters
) -> wp.float64:
    if cosine < -wp.float64(1.0) or cosine >= wp.float64(1.0):
        return wp.float64(0.0)
    normalisation = impulse_mixed_normalisation(energy, material, physical)
    reference_index = int(normalisation[0])
    if reference_index < 0 or normalisation[1] <= wp.float64(0.0):
        return wp.float64(0.0)
    ref_binding, ref_height = (
        wp.float64(physical.compton_binding[reference_index]),
        wp.float64(physical.compton_height[reference_index]),
    )
    total = wp.float64(0.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy > wp.float64(physical.compton_binding[shell]):
            relative = impulse_mixed_angle_relative(
                energy,
                cosine,
                wp.float64(physical.compton_binding[shell]),
                wp.float64(physical.compton_height[shell]),
                ref_binding,
                ref_height,
            )
            total += wp.exp(
                impulse_mixed_log_occupancy_ratio(
                    physical.compton_occupancy[shell], physical.compton_occupancy[reference_index]
                )
                + relative[0]
            )
    return bound.impulse_kn(energy, cosine) * total / normalisation[1]


@wp.func
def impulse_mixed_angle_proposal(
    energy: wp.float64, material: int, uniform: wp.float64, physical: PhysicalParameters
) -> wp.vec2d:
    reference_index = impulse_mixed_reference(energy, material, physical)
    if reference_index < 0:
        return wp.vec2d(wp.float64(wp.nan))
    ref_binding, ref_height = (
        wp.float64(physical.compton_binding[reference_index]),
        wp.float64(physical.compton_height[reference_index]),
    )
    total = wp.float64(0.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy > wp.float64(physical.compton_binding[shell]):
            back = impulse_mixed_back_relative(
                energy,
                wp.float64(physical.compton_binding[shell]),
                wp.float64(physical.compton_height[shell]),
                ref_binding,
                ref_height,
            )
            back_weight = wp.exp(
                impulse_mixed_log_occupancy_ratio(
                    physical.compton_occupancy[shell], physical.compton_occupancy[reference_index]
                )
                + back[0]
            )
            if back_weight > wp.float64(0.0):
                rate = impulse_mixed_back_rate(
                    energy,
                    wp.float64(physical.compton_binding[shell]),
                    wp.float64(physical.compton_height[shell]),
                )
                total += back_weight * impulse_mixed_envelope_integral(rate)
    target, cumulative = uniform * total, wp.float64(0.0)
    cosine, transfer = wp.float64(wp.nan), wp.float64(wp.nan)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        binding, height = (
            wp.float64(physical.compton_binding[shell]),
            wp.float64(physical.compton_height[shell]),
        )
        if energy > binding:
            back = impulse_mixed_back_relative(energy, binding, height, ref_binding, ref_height)
            back_weight = wp.exp(
                impulse_mixed_log_occupancy_ratio(
                    physical.compton_occupancy[shell], physical.compton_occupancy[reference_index]
                )
                + back[0]
            )
            if back_weight > wp.float64(0.0):
                rate = impulse_mixed_back_rate(energy, binding, height)
                weight = back_weight * impulse_mixed_envelope_integral(rate)
                previous = cumulative
                cumulative += weight
                if target < cumulative:
                    trailing = wp.float64(0.0)
                    for following in range(shell + 1, physical.compton_shell_offsets[material + 1]):
                        if energy > physical.compton_binding[following]:
                            following_back = impulse_mixed_back_relative(
                                energy,
                                physical.compton_binding[following],
                                physical.compton_height[following],
                                ref_binding,
                                ref_height,
                            )
                            following_weight = wp.exp(
                                impulse_mixed_log_occupancy_ratio(
                                    physical.compton_occupancy[following],
                                    physical.compton_occupancy[reference_index],
                                )
                                + following_back[0]
                            )
                            if following_weight > wp.float64(0.0):
                                following_rate = impulse_mixed_back_rate(
                                    energy,
                                    physical.compton_binding[following],
                                    physical.compton_height[following],
                                )
                                trailing += following_weight * impulse_mixed_envelope_integral(
                                    following_rate
                                )
                    fraction = (target - previous) / weight
                    complement = ((wp.float64(1.0) - uniform) * total - trailing) / weight
                    if rate == wp.float64(0.0):
                        cosine = wp.float64(2.0) * fraction - wp.float64(1.0)
                        transfer = wp.float64(2.0) * complement
                    else:
                        cosine = bound.impulse_tail_proposal_cosine(fraction, complement, rate)
                        transfer = bound.impulse_tail_proposal_transfer(
                            fraction, complement, rate, cosine
                        )
                    break
    numerator, denominator = wp.float64(0.0), wp.float64(0.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        binding, height = (
            wp.float64(physical.compton_binding[shell]),
            wp.float64(physical.compton_height[shell]),
        )
        if energy > binding:
            back = impulse_mixed_back_relative(energy, binding, height, ref_binding, ref_height)
            back_weight = wp.exp(
                impulse_mixed_log_occupancy_ratio(
                    physical.compton_occupancy[shell], physical.compton_occupancy[reference_index]
                )
                + back[0]
            )
            if back_weight > wp.float64(0.0):
                rate = impulse_mixed_back_rate(energy, binding, height)
                relative = wp.float64(0.0)
                if rate > wp.float64(0.0):
                    state = bound.impulse_tail_state(energy, wp.float64(2.0), binding, height)
                    relative = bound.impulse_tail_angle_relative_at_transfer(
                        energy, state, cosine + wp.float64(1.0), transfer
                    )[0]
                else:
                    local = impulse_mixed_mass_pair(energy, cosine, binding, height)
                    back_pair = impulse_mixed_mass_pair(energy, -wp.float64(1.0), binding, height)
                    relative = local[0] - back_pair[0]
                numerator += back_weight * wp.exp(relative)
                denominator += back_weight * wp.exp(-rate * (cosine + wp.float64(1.0)))
    acceptance = (
        wp.float64(0.5)
        * bound.impulse_kn_transfer_pair(energy, transfer)[0]
        * numerator
        / denominator
    )
    return wp.vec2d(cosine, acceptance)


@wp.func
def impulse_mixed_energy_event(
    energy: wp.float64,
    cosine: wp.float64,
    material: int,
    draws: wp.vec4d,
    physical: PhysicalParameters,
) -> wp.vec4d:
    result = wp.vec4d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(6.0))
    best = -wp.float64(wp.inf)
    reference_index = int(-1)  # noqa: UP018, RUF046 - mutable Warp index
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy > wp.float64(physical.compton_binding[shell]):
            mass = impulse_mixed_mass_pair(
                energy,
                cosine,
                wp.float64(physical.compton_binding[shell]),
                wp.float64(physical.compton_height[shell]),
            )
            candidate = wp.log(physical.compton_occupancy[shell]) + mass[0]
            if candidate > best:
                best = candidate
                reference_index = shell
    if reference_index < 0:
        return result
    ref_binding, ref_height = (
        wp.float64(physical.compton_binding[reference_index]),
        wp.float64(physical.compton_height[reference_index]),
    )
    total = wp.float64(0.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy > wp.float64(physical.compton_binding[shell]):
            relative = impulse_mixed_local_mass_relative(
                energy,
                cosine,
                wp.float64(physical.compton_binding[shell]),
                wp.float64(physical.compton_height[shell]),
                ref_binding,
                ref_height,
            )
            total += wp.exp(
                impulse_mixed_log_occupancy_ratio(
                    physical.compton_occupancy[shell], physical.compton_occupancy[reference_index]
                )
                + relative[0]
            )
    target, cumulative = draws[0] * total, wp.float64(0.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        binding, height = (
            wp.float64(physical.compton_binding[shell]),
            wp.float64(physical.compton_height[shell]),
        )
        if energy > binding:
            relative = impulse_mixed_local_mass_relative(
                energy, cosine, binding, height, ref_binding, ref_height
            )
            weight = wp.exp(
                impulse_mixed_log_occupancy_ratio(
                    physical.compton_occupancy[shell], physical.compton_occupancy[reference_index]
                )
                + relative[0]
            )
            cumulative += weight
            if weight > wp.float64(0.0) and target < cumulative:
                upper = bound.impulse_momentum_max(energy, cosine, binding)
                outgoing, momentum = wp.float64(0.0), wp.float64(0.0)
                if upper >= wp.float64(0.0):
                    momentum = bound.impulse_profile_interval_inverse(
                        draws[1], -wp.float64(1.0), upper, height
                    )
                    outgoing = bound.impulse_energy_from_momentum(energy, cosine, momentum, binding)
                    physical_momentum = bound.impulse_tail_joint_kinematic_pair(
                        energy, cosine, outgoing
                    )
                    momentum = physical_momentum[0] / physical_momentum[1]
                else:
                    state = bound.impulse_tail_state_at_cosine(energy, cosine, binding, height)
                    size = -state.cutoff
                    if size <= wp.float64(64.0) or wp.log(draws[1]) < -wp.float64(0.5) * size:
                        span = bound.impulse_tail_momentum_span(state, draws[1])
                        if (energy - state.binding) / energy < wp.float64(
                            1e-12
                        ) and size < wp.float64(0.001):
                            outgoing = bound.impulse_tail_threshold_energy(
                                energy, cosine, state, draws[1]
                            )
                        else:
                            outgoing = bound.impulse_tail_energy_from_span(
                                energy, cosine, state.binding, span
                            )
                        momentum = -wp.float64(1.0) + span
                    else:
                        deficit = bound.impulse_tail_momentum_deficit(
                            energy, cosine, state, draws[1]
                        )
                        outgoing = (
                            energy
                            - state.binding
                            - bound.impulse_tail_energy_deficit(
                                energy, cosine, state.binding, deficit
                            )
                        )
                        momentum = state.momentum - deficit
                if (
                    outgoing > wp.float64(0.0)
                    and outgoing <= energy - binding
                    and wp.isfinite(outgoing)
                ):
                    result = wp.vec4d(outgoing, binding, momentum, wp.float64(0.0))
                break
    return result


@wp.func
def impulse_mixed_scatter_event(
    energy: wp.float64,
    material: int,
    seed: wp.uint64,
    history: wp.uint64,
    event: wp.uint32,
    max_trials: int,
    physical: PhysicalParameters,
) -> wp.vec4d:
    result = wp.vec4d(wp.float64(0.0), wp.float64(0.0), energy, wp.float64(6.0))
    for trial in range(max_trials):
        domain = (
            wp.uint32(0x80000000)
            + wp.uint32(physical.random_branch) * wp.uint32(65536)
            + wp.uint32(trial)
        )
        draws = transport_random4(seed, history, event, domain, physical.sampling)
        angle = impulse_mixed_angle_proposal(energy, material, draws[0], physical)
        if draws[1] < angle[1]:
            energy_domain = wp.uint32(0x10000000) + wp.uint32(physical.random_branch) * wp.uint32(
                65536
            )
            energy_draws = transport_random4(seed, history, event, energy_domain, physical.sampling)
            broadening = impulse_mixed_energy_event(
                energy, angle[0], material, energy_draws, physical
            )
            result = wp.vec4d(angle[0], bound.TWO_PI * draws[2], broadening[0], broadening[3])
            break
    return result


@wp.func
def impulse_mixed_joint_profile(
    energy: wp.float64,
    cosine: wp.float64,
    outgoing: wp.float64,
    material: int,
    normalisation: wp.vec4d,
    physical: PhysicalParameters,
) -> wp.vec4d:
    reference_index = int(normalisation[0])
    if reference_index < 0:
        return wp.vec4d(wp.float64(0.0))
    ref_binding, ref_height = (
        wp.float64(physical.compton_binding[reference_index]),
        wp.float64(physical.compton_height[reference_index]),
    )
    ref_pair = impulse_mixed_mass_pair(energy, -wp.float64(1.0), ref_binding, ref_height)
    kinematic = bound.impulse_tail_joint_kinematic_pair(energy, cosine, outgoing)
    physical_q = kinematic[1]
    momentum = kinematic[0] / physical_q
    momentum_e = (
        -wp.float64(1.0) + outgoing * (wp.float64(1.0) - cosine) / MASS
    ) / physical_q - momentum * (energy - outgoing * cosine) / (physical_q * physical_q)
    use_anchor = momentum < wp.float64(0.0) and bound.impulse_momentum_max(
        energy, -wp.float64(1.0), ref_binding
    ) < wp.float64(0.0)
    density, slopes, centres = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        binding, h = (
            wp.float64(physical.compton_binding[shell]),
            wp.float64(physical.compton_height[shell]),
        )
        if outgoing < energy - binding:
            if impulse_mixed_profile_log(momentum, h) == -wp.float64(wp.inf):
                continue
            slope = bound.impulse_profile_log_slope(momentum, h)
            value, centre = wp.float64(0.0), wp.float64(0.0)
            if use_anchor:
                reference = bound.impulse_tail_state(
                    energy, wp.float64(2.0), ref_binding, ref_height
                )
                local = bound.impulse_tail_local_delta(energy, cosine, outgoing, reference)
                delta, delta_e = local[0], local[1]
                reference_p_e = (
                    wp.float64(2.0)
                    * reference.product_e
                    * (
                        wp.float64(1.0) / (MASS * reference.q)
                        - reference.momentum / (reference.q * reference.q)
                    )
                )
                h0, difference = reference.height, h - reference.height
                a_difference = -difference * reference.momentum - h * delta
                a_sum = -h * (reference.momentum + delta) - h0 * reference.momentum
                exponent = -wp.float64(2.0) * a_difference * (wp.float64(1.0) + a_sum)
                log_factor = (
                    wp.log(wp.float64(2.0))
                    + wp.log(h)
                    + bound.impulse_tail_log1p(-wp.float64(2.0) * h * (reference.momentum + delta))
                    - reference.log_fraction
                )
                value = wp.exp(
                    impulse_mixed_log_occupancy_ratio(
                        physical.compton_occupancy[shell],
                        physical.compton_occupancy[reference_index],
                    )
                    + exponent
                    + log_factor
                )
                if value > wp.float64(0.0):
                    first_product = difference * reference_p_e + h * delta_e
                    second_product = difference * (
                        h + h0
                    ) * reference.momentum * reference_p_e + h * h * (
                        delta * reference_p_e + reference.momentum * delta_e + delta * delta_e
                    )
                    centre = (
                        wp.float64(2.0) * first_product
                        - wp.float64(4.0) * second_product
                        - wp.float64(2.0)
                        * h
                        / (wp.float64(1.0) - wp.float64(2.0) * h * (reference.momentum + delta))
                        * (reference_p_e + delta_e)
                        - bound.impulse_tail_fraction_score(reference, wp.float64(2.0))
                    )
            else:
                value = wp.exp(
                    impulse_mixed_log_occupancy_ratio(
                        physical.compton_occupancy[shell],
                        physical.compton_occupancy[reference_index],
                    )
                    + impulse_mixed_profile_log_density(momentum, h)
                    - ref_pair[0]
                )
                if value > wp.float64(0.0):
                    centre = slope * momentum_e - ref_pair[1]
            if value > wp.float64(0.0):
                density += value
                slopes += value * slope
                centres += value * centre
    if density <= wp.float64(0.0):
        return wp.vec4d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), momentum)
    return wp.vec4d(density, slopes / density, centres / density, momentum)


@wp.func
def impulse_mixed_joint_pdf(
    energy: wp.float64,
    cosine: wp.float64,
    outgoing: wp.float64,
    material: int,
    physical: PhysicalParameters,
) -> wp.float64:
    if (
        outgoing <= wp.float64(0.0)
        or outgoing >= energy
        or cosine < -wp.float64(1.0)
        or cosine >= wp.float64(1.0)
    ):
        return wp.float64(0.0)
    normalisation = impulse_mixed_normalisation(energy, material, physical)
    if normalisation[1] <= wp.float64(0.0):
        return wp.float64(0.0)
    density = impulse_mixed_joint_profile(
        energy, cosine, outgoing, material, normalisation, physical
    )[0]
    ratio = outgoing / energy
    factor = wp.float64(1.0) + energy / MASS * (wp.float64(1.0) - cosine)
    q2 = (wp.float64(1.0) - ratio) * (wp.float64(1.0) - ratio) + wp.float64(2.0) * ratio * (
        wp.float64(1.0) - cosine
    )
    jacobian = (factor - cosine + ratio * (wp.float64(1.0) - factor * cosine)) / (
        energy * q2 * wp.sqrt(q2)
    )
    return bound.impulse_kn(energy, cosine) * density * jacobian / normalisation[1]


@wp.func
def impulse_mixed_joint_scores(
    energy: wp.float64,
    cosine: wp.float64,
    outgoing: wp.float64,
    material: int,
    physical: PhysicalParameters,
) -> wp.vec3d:
    normalisation = impulse_mixed_normalisation(energy, material, physical)
    profile = impulse_mixed_joint_profile(
        energy, cosine, outgoing, material, normalisation, physical
    )
    if profile[0] <= wp.float64(0.0) or normalisation[1] <= wp.float64(0.0):
        return wp.vec3d(wp.float64(wp.nan))
    profile_score, centre, momentum = profile[1], profile[2], profile[3]
    ratio, k = outgoing / energy, energy / MASS
    factor = wp.float64(1.0) + k * (wp.float64(1.0) - cosine)
    q2 = (wp.float64(1.0) - ratio) * (wp.float64(1.0) - ratio) + wp.float64(2.0) * ratio * (
        wp.float64(1.0) - cosine
    )
    q = wp.sqrt(q2)
    n = factor - cosine + ratio * (wp.float64(1.0) - factor * cosine)
    tau = wp.float64(1.0) / factor
    kn_term = tau + factor - wp.float64(1.0) + cosine * cosine
    kn_c = wp.float64(2.0) * k * tau + (k * tau * tau - k + wp.float64(2.0) * cosine) / kn_term
    p_c = -k * ratio / q + momentum * ratio / q2
    n_c = -k - wp.float64(1.0) + ratio * (k * cosine - factor)
    score_c = kn_c + profile_score * p_c + n_c / n + wp.float64(3.0) * ratio / q2
    p_r = n / (q * q2)
    score_outgoing = (
        profile_score * p_r
        + (wp.float64(1.0) - factor * cosine) / n
        - wp.float64(3.0) * (ratio - cosine) / q2
    ) / energy
    a_e = (wp.float64(1.0) - cosine) / MASS
    n_e = a_e * (wp.float64(1.0) - ratio * cosine) - ratio / energy * (
        wp.float64(1.0) - factor * cosine
    )
    kn_e = -wp.float64(2.0) * tau * a_e + (wp.float64(1.0) - tau * tau) * a_e / kn_term
    score_incident = (
        kn_e
        + centre
        + n_e / n
        - wp.float64(1.0) / energy
        + wp.float64(3.0) * ratio * (ratio - cosine) / (energy * q2)
        - normalisation[2] / normalisation[1]
    )
    return wp.vec3d(score_c, score_outgoing, score_incident)
