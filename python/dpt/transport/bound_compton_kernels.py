"""Matched relativistic impulse angle, shell, energy and next-event laws."""

# Warp annotations are runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false
import warp as wp

from . import bound_compton_mixed_kernels as mixed
from .physical_kernels import PhysicalParameters
from .sampling_kernels import transport_random4

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})
MASS = wp.constant(wp.float64(510.99895069))
TWO_PI = wp.constant(wp.float64(6.283185307179586476925286766559))


@wp.func
def impulse_profile_log_slope(momentum: wp.float64, height: wp.float64) -> wp.float64:
    """Matched signed log J derivative without cancellation or h*h overflow."""
    if momentum == wp.float64(0.0):
        return wp.float64(0.0)
    amplitude = height * wp.abs(momentum)
    if amplitude < wp.float64(2.2250738585072014e-308):
        magnitude = ((wp.float64(8.0) * height) * height) * wp.abs(momentum)
        result = -magnitude
        if momentum < wp.float64(0.0):
            result = magnitude
        return result
    factor = wp.float64(4.0) * (
        wp.float64(1.0) + wp.float64(1.0) / (wp.float64(1.0) + wp.float64(2.0) * amplitude)
    )
    magnitude = (factor * wp.min(height, amplitude)) * wp.max(height, amplitude)
    result = -magnitude
    if momentum < wp.float64(0.0):
        result = magnitude
    return result


@wp.func
def impulse_profile_cdf(momentum: wp.float64, height: wp.float64) -> wp.float64:
    absolute = wp.abs(momentum)
    tail = wp.float64(0.5) * wp.exp(
        -wp.float64(2.0) * height * absolute * (wp.float64(1.0) + height * absolute)
    )
    result = wp.float64(1.0) - tail
    if momentum <= wp.float64(0.0):
        result = tail
    return result


@wp.func
def impulse_profile_inverse(probability: wp.float64, height: wp.float64) -> wp.float64:
    tail = wp.min(probability, wp.float64(1.0) - probability)
    logarithm = -wp.log(wp.float64(2.0) * tail)
    magnitude = logarithm / (
        height * (wp.sqrt(wp.float64(1.0) + wp.float64(2.0) * logarithm) + wp.float64(1.0))
    )
    result = magnitude
    if probability < wp.float64(0.5):
        result = -magnitude
    return result


@wp.func
def impulse_expm1(value: wp.float64) -> wp.float64:
    """Binary64 exp(x)-1 near zero without a Warp-version dependency."""
    if wp.abs(value) >= wp.float64(0.125):
        return wp.exp(value) - wp.float64(1.0)
    term = value
    result = value
    for order in range(2, 22):
        term *= value / wp.float64(order)
        result += term
    return result


@wp.func
def impulse_log1p(value: wp.float64) -> wp.float64:
    """Binary64 log(1+x) through its convergent atanh series near zero."""
    if wp.abs(value) >= wp.float64(0.125):
        return wp.log(wp.float64(1.0) + value)
    ratio = value / (wp.float64(2.0) + value)
    square = ratio * ratio
    term = ratio
    result = ratio
    for order in range(1, 21):
        term *= square
        result += term / wp.float64(2 * order + 1)
    return wp.float64(2.0) * result


@wp.func
def impulse_profile_interval_mass(
    lower: wp.float64,
    upper: wp.float64,
    height: wp.float64,
) -> wp.float64:
    if lower >= upper:
        return wp.float64(0.0)
    if lower < wp.float64(0.0) and upper > wp.float64(0.0):
        left = (
            -wp.float64(2.0) * height * wp.abs(lower) * (wp.float64(1.0) + height * wp.abs(lower))
        )
        right = -wp.float64(2.0) * height * upper * (wp.float64(1.0) + height * upper)
        return -wp.float64(0.5) * (impulse_expm1(left) + impulse_expm1(right))
    near = lower
    far = upper
    if upper <= wp.float64(0.0):
        near = wp.abs(upper)
        far = wp.abs(lower)
    exponent = -wp.float64(2.0) * height * near * (wp.float64(1.0) + height * near)
    difference = (
        -wp.float64(2.0) * height * (upper - lower) * (wp.float64(1.0) + height * (near + far))
    )
    return -wp.float64(0.5) * wp.exp(exponent) * impulse_expm1(difference)


@wp.func
def impulse_profile_same_sign_inverse_pair(
    fraction: wp.float64,
    complement: wp.float64,
    lower: wp.float64,
    upper: wp.float64,
    height: wp.float64,
) -> wp.float64:
    near = lower
    far = upper
    weight = fraction
    other = complement
    if upper <= wp.float64(0.0):
        near = wp.abs(upper)
        far = wp.abs(lower)
        weight = complement
        other = fraction
    exponent = -wp.float64(2.0) * height * near * (wp.float64(1.0) + height * near)
    difference = (
        -wp.float64(2.0) * height * (upper - lower) * (wp.float64(1.0) + height * (near + far))
    )
    relative_mass = -impulse_expm1(difference)
    logarithm = exponent
    if weight * relative_mass <= wp.float64(0.5):
        logarithm += impulse_log1p(-weight * relative_mass)
    else:
        logarithm += wp.log(other + weight * wp.exp(difference))
    logarithm = -logarithm
    magnitude = logarithm / (
        height * (wp.sqrt(wp.float64(1.0) + wp.float64(2.0) * logarithm) + wp.float64(1.0))
    )
    if upper <= wp.float64(0.0):
        return -magnitude
    return magnitude


@wp.func
def impulse_profile_interval_inverse_pair(
    fraction: wp.float64,
    complement: wp.float64,
    lower: wp.float64,
    upper: wp.float64,
    height: wp.float64,
) -> wp.float64:
    if lower < wp.float64(0.0) and upper > wp.float64(0.0):
        left = impulse_profile_interval_mass(lower, wp.float64(0.0), height)
        right = impulse_profile_interval_mass(wp.float64(0.0), upper, height)
        displacement = fraction * right - complement * left
        if displacement < wp.float64(0.0):
            return impulse_profile_same_sign_inverse_pair(
                fraction * (left + right) / left,
                -displacement / left,
                lower,
                wp.float64(0.0),
                height,
            )
        if displacement > wp.float64(0.0):
            return impulse_profile_same_sign_inverse_pair(
                displacement / right,
                complement * (left + right) / right,
                wp.float64(0.0),
                upper,
                height,
            )
        return wp.float64(0.0)
    return impulse_profile_same_sign_inverse_pair(fraction, complement, lower, upper, height)


@wp.func
def impulse_profile_interval_inverse(
    probability: wp.float64,
    lower: wp.float64,
    upper: wp.float64,
    height: wp.float64,
) -> wp.float64:
    return impulse_profile_interval_inverse_pair(
        probability,
        wp.float64(1.0) - probability,
        lower,
        upper,
        height,
    )


@wp.func
def impulse_momentum_max(energy: wp.float64, cosine: wp.float64, binding: wp.float64) -> wp.float64:
    outgoing = energy - binding
    numerator = energy * outgoing * (wp.float64(1.0) - cosine) / MASS - binding
    denominator = wp.sqrt(
        wp.float64(2.0) * energy * outgoing * (wp.float64(1.0) - cosine) + binding * binding
    )
    result = wp.float64(0.0)
    if denominator > wp.float64(0.0):
        if wp.abs(numerator) < wp.float64(1e-3) * (wp.abs(numerator + binding) + binding):
            transfer = impulse_tail_two_sum(wp.float64(1.0), -cosine)
            pair = impulse_tail_kinematic_pair(energy, transfer[0], binding, transfer[1])
            numerator = pair[0]
            denominator = pair[1]
        result = numerator / denominator
    return result


@wp.func
def impulse_shell_mass(
    energy: wp.float64, cosine: wp.float64, shell: int, physical: PhysicalParameters
) -> wp.float64:
    result = wp.float64(0.0)
    binding = physical.compton_binding[shell]
    if energy > binding:
        height = physical.compton_height[shell]
        upper = impulse_momentum_max(energy, cosine, binding)
        result = impulse_profile_interval_mass(-wp.float64(1.0), upper, height)
    return result


@wp.func
def impulse_occupancy_scale(
    energy: wp.float64, material: int, physical: PhysicalParameters
) -> wp.float64:
    largest = wp.float64(0.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy > physical.compton_binding[shell]:
            largest = wp.max(largest, physical.compton_occupancy[shell])
    if largest > wp.float64(0.0) and largest < wp.float64(1e-280):
        return largest
    return wp.float64(1.0)


@wp.func
def impulse_scattering_weight(
    energy: wp.float64, cosine: wp.float64, material: int, physical: PhysicalParameters
) -> wp.float64:
    scale = impulse_occupancy_scale(energy, material, physical)
    return impulse_scattering_weight_scaled(energy, cosine, material, physical, scale) * scale


@wp.func
def impulse_scattering_weight_scaled(
    energy: wp.float64,
    cosine: wp.float64,
    material: int,
    physical: PhysicalParameters,
    occupancy_scale: wp.float64,
) -> wp.float64:
    total = wp.float64(0.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        total += (physical.compton_occupancy[shell] / occupancy_scale) * impulse_shell_mass(
            energy, cosine, shell, physical
        )
    return total


@wp.func
def impulse_use_scaled_tail(
    energy: wp.float64, cosine: wp.float64, material: int, physical: PhysicalParameters
) -> bool:
    occupancy_scale = impulse_occupancy_scale(energy, material, physical)
    narrow = int(0)  # noqa: UP018, RUF046 - mutable Warp flag
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy > physical.compton_binding[shell]:
            if impulse_momentum_max(energy, cosine, physical.compton_binding[shell]) >= wp.float64(
                0.0
            ):
                return False
            state = impulse_tail_state_at_cosine(
                energy, cosine, physical.compton_binding[shell], physical.compton_height[shell]
            )
            if state.rate * wp.float64(4.440892098500626e-16) > wp.float64(1e-12):
                narrow = int(1)  # noqa: UP018, RUF046
    return narrow != 0 or impulse_scattering_weight_scaled(
        energy, cosine, material, physical, occupancy_scale
    ) < wp.float64(1e-280)


@wp.func
def impulse_kn(energy: wp.float64, cosine: wp.float64) -> wp.float64:
    ratio = wp.float64(1.0) / (wp.float64(1.0) + energy / MASS * (wp.float64(1.0) - cosine))
    return ratio * ratio * (ratio + wp.float64(1.0) / ratio - wp.float64(1.0) + cosine * cosine)


@wp.func
def impulse_gauss16_node(index: int) -> wp.float64:
    first = wp.vec4d(
        0.09501250983763744, 0.2816035507792589, 0.4580167776572274, 0.6178762444026438
    )
    second = wp.vec4d(0.755404408355003, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499)
    if index < 4:
        return first[index]
    return second[index - 4]


@wp.func
def impulse_gauss16_weight(index: int) -> wp.float64:
    first = wp.vec4d(
        0.1894506104550685, 0.1826034150449236, 0.16915651939500254, 0.14959598881657673
    )
    second = wp.vec4d(
        0.12462897125553387, 0.09515851168249278, 0.06225352393864789, 0.027152459411754095
    )
    if index < 4:
        return first[index]
    return second[index - 4]


@wp.func
def impulse_kn_transfer_pair(energy: wp.float64, transfer: wp.float64) -> wp.vec2d:
    ratio = wp.float64(1.0) / (wp.float64(1.0) + energy / MASS * transfer)
    sine_squared = transfer * (wp.float64(2.0) - transfer)
    value = ratio * ratio * ratio + ratio - sine_squared * ratio * ratio
    derivative = (
        -transfer
        / MASS
        * ratio
        * ratio
        * (
            wp.float64(3.0) * ratio * ratio
            + wp.float64(1.0)
            - wp.float64(2.0) * sine_squared * ratio
        )
    )
    return wp.vec2d(value, derivative)


@wp.func
def impulse_kn_interval_pair(energy: wp.float64, lower: wp.float64) -> wp.vec2d:
    """Exact angular primitive, or smooth Gaussian rule in its Thomson limit."""
    width = wp.float64(2.0) - lower
    if width <= wp.float64(0.0):
        return wp.vec2d(wp.float64(0.0))
    alpha = energy / MASS
    if alpha < wp.float64(0.125):
        midpoint = (lower + wp.float64(2.0)) * wp.float64(0.5)
        radius = width * wp.float64(0.5)
        result = wp.vec2d(wp.float64(0.0))
        for node in range(8):
            for sign in range(2):
                transfer = midpoint + (
                    wp.float64(2 * sign) - wp.float64(1.0)
                ) * radius * impulse_gauss16_node(node)
                result += (
                    radius
                    * impulse_gauss16_weight(node)
                    * impulse_kn_transfer_pair(energy, transfer)
                )
        return result
    inverse = wp.float64(1.0) / alpha
    left = wp.float64(1.0) / (wp.float64(1.0) + alpha * lower)
    right = wp.float64(1.0) / (wp.float64(1.0) + wp.float64(2.0) * alpha)
    difference = alpha * width * left * right
    logarithm = impulse_log1p(alpha * width * left)
    c1 = wp.float64(1.0) - wp.float64(2.0) * inverse - wp.float64(2.0) * inverse * inverse
    c2 = wp.float64(2.0) * inverse + inverse * inverse
    primitive = (
        wp.float64(0.5) * difference * (left + right)
        + c1 * logarithm
        + c2 * difference
        + width * inverse
    )
    value = primitive * inverse
    left_dot = -lower * left * left
    right_dot = -wp.float64(2.0) * right * right
    difference_dot = (
        width * left * right * (wp.float64(1.0) - alpha * (lower * left + wp.float64(2.0) * right))
    )
    c1_dot = wp.float64(2.0) * inverse * inverse + wp.float64(4.0) * inverse * inverse * inverse
    c2_dot = -wp.float64(2.0) * inverse * inverse - wp.float64(2.0) * inverse * inverse * inverse
    primitive_dot = (
        wp.float64(0.5) * difference_dot * (left + right)
        + wp.float64(0.5) * difference * (left_dot + right_dot)
        + c1_dot * logarithm
        + c1 * width * left * right
        + c2_dot * difference
        + c2 * difference_dot
        - width * inverse * inverse
    )
    return wp.vec2d(value, (primitive_dot - value) * inverse / MASS)


@wp.func
def impulse_minimum_transfer(
    energy: wp.float64, binding: wp.float64, momentum: wp.float64
) -> wp.float64:
    if binding == wp.float64(0.0) and momentum <= wp.float64(0.0):
        return wp.float64(0.0)
    first = MASS * momentum
    second = wp.sqrt(binding) * wp.sqrt(binding + wp.float64(2.0) * MASS)
    scale = wp.max(wp.abs(first), second)
    root = wp.float64(0.0)
    if scale > wp.float64(0.0):
        root = scale * wp.sqrt(
            (first / scale) * (first / scale) + (second / scale) * (second / scale)
        )
    value = binding + MASS * momentum * momentum + momentum * root
    if momentum < wp.float64(0.0):
        value = (
            binding
            * binding
            * (wp.float64(1.0) - momentum)
            * (wp.float64(1.0) + momentum)
            / (binding + MASS * momentum * momentum - momentum * root)
        )
    return MASS * value / energy / (energy - binding)


@wp.func
def impulse_angular_normalisation_scaled_pair(
    energy: wp.float64,
    material: int,
    physical: PhysicalParameters,
    occupancy_scale: wp.float64,
) -> wp.vec2d:
    """Profile-quantile normaliser and analytic fixed-momentum E derivative.

    Fubini resolves narrow shell profiles without an angular transition grid.
    The moving shell endpoint has zero angular interval, hence no endpoint
    term. The two physical integrands use the same quantile integration rule.
    """
    result = wp.vec2d(wp.float64(0.0))
    radius = wp.float64(1.0 / 64.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        binding = physical.compton_binding[shell]
        height = physical.compton_height[shell]
        if energy > binding:
            upper = impulse_momentum_max(energy, -wp.float64(1.0), binding)
            for segment in range(2):
                segment_lower = -wp.float64(1.0)
                segment_upper = wp.min(upper, wp.float64(0.0))
                if segment == 1:
                    segment_lower = wp.float64(0.0)
                    segment_upper = upper
                accessibility = impulse_profile_interval_mass(segment_lower, segment_upper, height)
                if accessibility > wp.float64(0.0):
                    for interval in range(32):
                        midpoint = (wp.float64(interval) + wp.float64(0.5)) / wp.float64(32.0)
                        for node in range(8):
                            for sign in range(2):
                                coordinate = midpoint + (
                                    wp.float64(2 * sign) - wp.float64(1.0)
                                ) * radius * impulse_gauss16_node(node)
                                complement_coordinate = wp.float64(1.0) - coordinate
                                left = coordinate * coordinate * coordinate * coordinate
                                right = (
                                    complement_coordinate
                                    * complement_coordinate
                                    * complement_coordinate
                                    * complement_coordinate
                                )
                                total = left + right
                                fraction = left / total
                                complement = right / total
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
                                momentum = impulse_profile_interval_inverse_pair(
                                    fraction, complement, segment_lower, segment_upper, height
                                )
                                lower = impulse_minimum_transfer(energy, binding, momentum)
                                values = impulse_kn_interval_pair(energy, lower)
                                moving_limit = lower * (
                                    wp.float64(1.0) / energy + wp.float64(1.0) / (energy - binding)
                                )
                                values[1] += (
                                    impulse_kn_transfer_pair(energy, lower)[0] * moving_limit
                                )
                                coefficient = (
                                    (physical.compton_occupancy[shell] / occupancy_scale)
                                    * accessibility
                                    * radius
                                    * impulse_gauss16_weight(node)
                                    * jacobian
                                )
                                result += coefficient * values
    return TWO_PI * result


@wp.func
def impulse_angular_normalisation_pair(
    energy: wp.float64, material: int, physical: PhysicalParameters
) -> wp.vec2d:
    if mixed.impulse_use_mixed_conditioning(energy, material, physical):
        return mixed.impulse_mixed_raw_normalisation_pair(energy, material, physical)
    scale = impulse_occupancy_scale(energy, material, physical)
    return impulse_angular_normalisation_scaled_pair(energy, material, physical, scale) * scale


@wp.func
def impulse_angular_normalisation(
    energy: wp.float64,
    material: int,
    physical: PhysicalParameters,
) -> wp.float64:
    return impulse_angular_normalisation_pair(energy, material, physical)[0]


@wp.func
def bound_phase_pdf(
    energy: wp.float64, cosine: wp.float64, material: int, physical: PhysicalParameters
) -> wp.float64:
    if mixed.impulse_use_mixed_conditioning(energy, material, physical):
        return mixed.impulse_mixed_phase_pdf(energy, cosine, material, physical)
    occupancy_scale = impulse_occupancy_scale(energy, material, physical)
    normalisation = impulse_angular_normalisation_scaled_pair(
        energy, material, physical, occupancy_scale
    )[0]
    if normalisation == wp.float64(0.0) or impulse_use_scaled_tail(
        energy, -wp.float64(1.0), material, physical
    ):
        return impulse_tail_phase_pdf(energy, cosine, material, physical)
    result = wp.float64(0.0)
    if normalisation > wp.float64(0.0):
        result = (
            impulse_kn(energy, cosine)
            * impulse_scattering_weight_scaled(energy, cosine, material, physical, occupancy_scale)
            / normalisation
        )
    return result


@wp.func
def impulse_energy_from_momentum(
    energy: wp.float64, cosine: wp.float64, momentum: wp.float64, binding: wp.float64
) -> wp.float64:
    lower = wp.float64(0.0)
    upper = (energy - binding) / energy
    factor = wp.float64(1.0) + energy / MASS * (wp.float64(1.0) - cosine)
    for _iteration in range(56):
        ratio = (lower + upper) * wp.float64(0.5)
        q = wp.sqrt(
            (wp.float64(1.0) - ratio) * (wp.float64(1.0) - ratio)
            + wp.float64(2.0) * ratio * (wp.float64(1.0) - cosine)
        )
        value = (factor * ratio - wp.float64(1.0)) / q
        if value < momentum:
            lower = ratio
        else:
            upper = ratio
    return energy * (lower + upper) * wp.float64(0.5)


@wp.func
def bound_energy_event(
    energy: wp.float64,
    cosine: wp.float64,
    material: int,
    seed: wp.uint64,
    history: wp.uint64,
    event: wp.uint32,
    next_event: int,
    physical: PhysicalParameters,
) -> wp.vec4d:
    """Return outgoing energy, shell binding, signed momentum and status."""
    occupancy_scale = impulse_occupancy_scale(energy, material, physical)
    result = wp.vec4d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(6.0))
    total = impulse_scattering_weight_scaled(energy, cosine, material, physical, occupancy_scale)
    domain = wp.uint32(0x10000000) + wp.uint32(physical.random_branch) * wp.uint32(65536)
    if next_event != 0:
        domain += wp.uint32(0x10000000)
    draws = transport_random4(seed, history, event, domain, physical.sampling)
    if mixed.impulse_use_mixed_conditioning(energy, material, physical):
        return mixed.impulse_mixed_energy_event(energy, cosine, material, draws, physical)
    if total == wp.float64(0.0) or impulse_use_scaled_tail(energy, cosine, material, physical):
        return impulse_tail_energy_event(energy, cosine, material, draws, physical)
    cumulative = wp.float64(0.0)
    if total > wp.float64(0.0) and wp.isfinite(draws[0]) and wp.isfinite(draws[1]):
        target = draws[0] * total
        for shell in range(
            physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
        ):
            mass = impulse_shell_mass(energy, cosine, shell, physical)
            cumulative += (physical.compton_occupancy[shell] / occupancy_scale) * mass
            if target < cumulative and mass > wp.float64(0.0):
                height = physical.compton_height[shell]
                binding = physical.compton_binding[shell]
                momentum = impulse_profile_interval_inverse(
                    draws[1],
                    -wp.float64(1.0),
                    impulse_momentum_max(energy, cosine, binding),
                    height,
                )
                outgoing = impulse_energy_from_momentum(energy, cosine, momentum, binding)
                if (
                    outgoing > wp.float64(0.0)
                    and outgoing <= energy - binding
                    and wp.isfinite(outgoing)
                ):
                    result = wp.vec4d(outgoing, binding, momentum, wp.float64(0.0))
                break
    return result


@wp.func
def bound_scatter_event(
    energy: wp.float64,
    material: int,
    seed: wp.uint64,
    history: wp.uint64,
    event: wp.uint32,
    max_trials: int,
    physical: PhysicalParameters,
) -> wp.vec4d:
    """Return cosine, azimuth, broadened energy and mandatory completion status."""
    if mixed.impulse_use_mixed_conditioning(energy, material, physical):
        return mixed.impulse_mixed_scatter_event(
            energy, material, seed, history, event, max_trials, physical
        )
    occupancy_scale = impulse_occupancy_scale(energy, material, physical)
    result = wp.vec4d(wp.float64(0.0), wp.float64(0.0), energy, wp.float64(6.0))
    # For each occupied shell pmax increases monotonically with 1-cos(theta),
    # so the backscatter accessibility is a valid, energy-specific envelope.
    # Dividing by Z would retain a tiny total cross section in a conditional
    # final-state sampler, making low-energy H impossible for finite-bit draws.
    envelope = impulse_scattering_weight_scaled(
        energy, -wp.float64(1.0), material, physical, occupancy_scale
    )
    if envelope <= wp.float64(0.0) or impulse_use_scaled_tail(
        energy, -wp.float64(1.0), material, physical
    ):
        return impulse_tail_scatter_event(
            energy, material, seed, history, event, max_trials, physical
        )
    for trial in range(max_trials):
        domain = (
            wp.uint32(0x80000000)
            + wp.uint32(physical.random_branch) * wp.uint32(65536)
            + wp.uint32(trial)
        )
        draws = transport_random4(seed, history, event, domain, physical.sampling)
        cosine = wp.float64(2.0) * draws[0] - wp.float64(1.0)
        acceptance = (
            wp.float64(0.5)
            * impulse_kn(energy, cosine)
            * impulse_scattering_weight_scaled(energy, cosine, material, physical, occupancy_scale)
            / envelope
        )
        if draws[1] < acceptance:
            broadening = bound_energy_event(
                energy, cosine, material, seed, history, event, 0, physical
            )
            result = wp.vec4d(cosine, TWO_PI * draws[2], broadening[0], broadening[3])
            break
    return result


@wp.func
def bound_joint_pdf(
    energy: wp.float64,
    cosine: wp.float64,
    outgoing: wp.float64,
    material: int,
    physical: PhysicalParameters,
) -> wp.float64:
    if mixed.impulse_use_mixed_conditioning(energy, material, physical):
        return mixed.impulse_mixed_joint_pdf(energy, cosine, outgoing, material, physical)
    """Joint directional/energy density per steradian and keV for geometry scores."""
    occupancy_scale = impulse_occupancy_scale(energy, material, physical)
    if impulse_use_scaled_tail(energy, -wp.float64(1.0), material, physical):
        return impulse_tail_joint_pdf(energy, cosine, outgoing, material, physical)
    result = wp.float64(0.0)
    if outgoing > wp.float64(0.0) and outgoing < energy and cosine < wp.float64(1.0):
        ratio = outgoing / energy
        factor = wp.float64(1.0) + energy / MASS * (wp.float64(1.0) - cosine)
        q = wp.sqrt(
            (wp.float64(1.0) - ratio) * (wp.float64(1.0) - ratio)
            + wp.float64(2.0) * ratio * (wp.float64(1.0) - cosine)
        )
        momentum = (factor * ratio - wp.float64(1.0)) / q
        if wp.abs(factor * ratio - wp.float64(1.0)) < wp.float64(1e-3) * (
            wp.abs(factor * ratio) + wp.float64(1.0)
        ):
            pair = impulse_tail_joint_kinematic_pair(energy, cosine, outgoing)
            momentum = pair[0] / pair[1]
        jacobian = (factor - cosine + ratio * (wp.float64(1.0) - factor * cosine)) / (
            energy * q * q * q
        )
        density = wp.float64(0.0)
        for shell in range(
            physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
        ):
            if outgoing < energy - physical.compton_binding[shell]:
                height = physical.compton_height[shell]
                absolute = wp.abs(momentum)
                density += (
                    (physical.compton_occupancy[shell] / occupancy_scale)
                    * height
                    * (wp.float64(1.0) + wp.float64(2.0) * height * absolute)
                    * wp.exp(
                        -wp.float64(2.0) * height * absolute * (wp.float64(1.0) + height * absolute)
                    )
                )
        normalisation = impulse_angular_normalisation_scaled_pair(
            energy, material, physical, occupancy_scale
        )[0]
        if normalisation > wp.float64(0.0) and momentum > -wp.float64(1.0):
            result = impulse_kn(energy, cosine) * density * jacobian / normalisation
    return result


@wp.func
def impulse_scattering_energy_derivative(
    energy: wp.float64, cosine: wp.float64, material: int, physical: PhysicalParameters
) -> wp.float64:
    occupancy_scale = impulse_occupancy_scale(energy, material, physical)
    total = wp.float64(0.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        binding = physical.compton_binding[shell]
        if energy > binding:
            q = wp.sqrt(
                wp.float64(2.0) * energy * (energy - binding) * (wp.float64(1.0) - cosine)
                + binding * binding
            )
            if q > wp.float64(0.0):
                momentum = impulse_momentum_max(energy, cosine, binding)
                p_e = (
                    (wp.float64(2.0) * energy - binding)
                    * (wp.float64(1.0) - cosine)
                    * (wp.float64(1.0) / (MASS * q) - momentum / (q * q))
                )
                height = physical.compton_height[shell]
                absolute = wp.abs(momentum)
                density = (
                    height
                    * (wp.float64(1.0) + wp.float64(2.0) * height * absolute)
                    * wp.exp(
                        -wp.float64(2.0) * height * absolute * (wp.float64(1.0) + height * absolute)
                    )
                )
                total += (physical.compton_occupancy[shell] / occupancy_scale) * density * p_e
    return total * occupancy_scale


@wp.func
def impulse_normalisation_energy_score(
    energy: wp.float64,
    material: int,
    physical: PhysicalParameters,
) -> wp.float64:
    if mixed.impulse_use_mixed_conditioning(energy, material, physical):
        normalisation = mixed.impulse_mixed_normalisation(energy, material, physical)
        return normalisation[3] + normalisation[2] / normalisation[1]
    """Analytic incident-energy score of the matched angular normaliser."""
    occupancy_scale = impulse_occupancy_scale(energy, material, physical)
    values = impulse_angular_normalisation_scaled_pair(energy, material, physical, occupancy_scale)
    if values[0] == wp.float64(0.0) or impulse_use_scaled_tail(
        energy, -wp.float64(1.0), material, physical
    ):
        scaled = impulse_tail_normalisation(energy, material, physical)
        return scaled[3] + scaled[2] / scaled[1]
    return values[1] / values[0]


@wp.func
def bound_joint_scores(
    energy: wp.float64,
    cosine: wp.float64,
    outgoing: wp.float64,
    material: int,
    physical: PhysicalParameters,
) -> wp.vec3d:
    """Interior joint log scores in cosine, outgoing keV and incident keV.

    Shell support boundary measures are separate from these interior scores.
    """
    if mixed.impulse_use_mixed_conditioning(energy, material, physical):
        return mixed.impulse_mixed_joint_scores(energy, cosine, outgoing, material, physical)
    occupancy_scale = impulse_occupancy_scale(energy, material, physical)
    if impulse_use_scaled_tail(energy, -wp.float64(1.0), material, physical):
        return impulse_tail_joint_scores(energy, cosine, outgoing, material, physical)
    ratio = outgoing / energy
    k = energy / MASS
    factor = wp.float64(1.0) + k * (wp.float64(1.0) - cosine)
    q2 = (wp.float64(1.0) - ratio) * (wp.float64(1.0) - ratio) + wp.float64(2.0) * ratio * (
        wp.float64(1.0) - cosine
    )
    q = wp.sqrt(q2)
    momentum = (factor * ratio - wp.float64(1.0)) / q
    if wp.abs(factor * ratio - wp.float64(1.0)) < wp.float64(1e-3) * (
        wp.abs(factor * ratio) + wp.float64(1.0)
    ):
        pair = impulse_tail_joint_kinematic_pair(energy, cosine, outgoing)
        momentum = pair[0] / pair[1]
    n = factor - cosine + ratio * (wp.float64(1.0) - factor * cosine)
    density = wp.float64(0.0)
    derivative = wp.float64(0.0)
    threshold = int(0)  # noqa: UP018, RUF046 - mutable Warp loop variable
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy == physical.compton_binding[shell]:
            threshold = int(1)  # noqa: UP018, RUF046
        if outgoing < energy - physical.compton_binding[shell]:
            height = physical.compton_height[shell]
            absolute = wp.abs(momentum)
            value = (
                (physical.compton_occupancy[shell] / occupancy_scale)
                * height
                * (wp.float64(1.0) + wp.float64(2.0) * height * absolute)
                * wp.exp(
                    -wp.float64(2.0) * height * absolute * (wp.float64(1.0) + height * absolute)
                )
            )
            slope = impulse_profile_log_slope(momentum, height)
            density += value
            derivative += value * slope
    profile_score = derivative / density
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
    p_e = -ratio / (energy * q) + momentum * ratio * (ratio - cosine) / (energy * q2)
    n_e = a_e * (wp.float64(1.0) - ratio * cosine) - ratio / energy * (
        wp.float64(1.0) - factor * cosine
    )
    kn_e = -wp.float64(2.0) * tau * a_e + (wp.float64(1.0) - tau * tau) * a_e / kn_term
    score_incident = (
        kn_e
        + profile_score * p_e
        + n_e / n
        - wp.float64(1.0) / energy
        + wp.float64(3.0) * ratio * (ratio - cosine) / (energy * q2)
        - impulse_normalisation_energy_score(energy, material, physical)
    )
    if threshold != 0:
        score_incident = wp.float64(wp.nan)
    return wp.vec3d(score_c, score_outgoing, score_incident)


@wp.struct
class ImpulseTailState:
    binding: wp.float64
    height: wp.float64
    product: wp.float64
    product_e: wp.float64
    q: wp.float64
    numerator: wp.float64
    momentum: wp.float64
    span: wp.float64
    cutoff: wp.float64
    log_fraction: wp.float64
    rate: wp.float64
    score: wp.float64
    log_mass: wp.float64


@wp.func
def impulse_tail_cutoff(span: wp.float64, height: wp.float64) -> wp.vec2d:
    log_size = (
        wp.log(wp.float64(2.0))
        + wp.log(height)
        + wp.log(span)
        + impulse_tail_log1p(height * (wp.float64(2.0) - span))
    )
    if log_size > wp.float64(709.782712893384):
        return wp.vec2d(-wp.float64(wp.inf), wp.float64(0.0))
    if log_size > wp.float64(6.0):
        return wp.vec2d(-wp.exp(log_size), wp.float64(0.0))
    size = wp.float64(2.0) * height * span * (wp.float64(1.0) + height * (wp.float64(2.0) - span))
    if size == wp.float64(0.0):
        return wp.vec2d(-size, log_size)
    return wp.vec2d(-size, wp.log(-impulse_tail_expm1(-size)))


@wp.func
def impulse_tail_two_sum(first: wp.float64, second: wp.float64) -> wp.vec2d:
    total = first + second
    temporary = total - first
    error = first - (total - temporary) + (second - temporary)
    return wp.vec2d(total, error)


@wp.func
def impulse_tail_product_error(first: wp.float64, second: wp.float64) -> wp.float64:
    splitter = wp.float64(134217729.0)
    first_split = splitter * first
    second_split = splitter * second
    first_high = first_split - (first_split - first)
    second_high = second_split - (second_split - second)
    first_low = first - first_high
    second_low = second - second_high
    product = first * second
    return (
        first_high * second_high
        - product
        + first_high * second_low
        + first_low * second_high
        + first_low * second_low
    )


@wp.func
def impulse_tail_kinematic_pair(
    energy: wp.float64, transfer: wp.float64, binding: wp.float64, transfer_error: wp.float64
) -> wp.vec2d:
    _tail_pair_0 = impulse_tail_two_sum(energy, -binding)
    difference = _tail_pair_0[0]
    difference_error = _tail_pair_0[1]
    product = energy * difference
    product_error = impulse_tail_product_error(energy, difference) + energy * difference_error
    transferred = product * transfer
    transferred_error = (
        impulse_tail_product_error(product, transfer)
        + product_error * transfer
        + product * transfer_error
        + product_error * transfer_error
    )
    divided = transferred / MASS
    divided_product = divided * MASS
    division_error = (
        transferred
        - divided_product
        - impulse_tail_product_error(divided, MASS)
        + transferred_error
    ) / MASS
    _tail_pair_1 = impulse_tail_two_sum(divided, -binding)
    numerator = _tail_pair_1[0]
    numerator_error = _tail_pair_1[1]
    numerator += numerator_error + division_error
    _tail_pair_2 = impulse_tail_two_sum(wp.float64(2.0) * transferred, binding * binding)
    square = _tail_pair_2[0]
    square_error = _tail_pair_2[1]
    square_error += wp.float64(2.0) * transferred_error + impulse_tail_product_error(
        binding, binding
    )
    q = wp.sqrt(square)
    q_error = (square - q * q - impulse_tail_product_error(q, q) + square_error) / (
        wp.float64(2.0) * q
    )
    return wp.vec2d(numerator, q + q_error)


@wp.func
def impulse_tail_joint_kinematic_pair(
    energy: wp.float64, cosine: wp.float64, outgoing: wp.float64
) -> wp.vec2d:
    _tail_pair_0 = impulse_tail_two_sum(wp.float64(1.0), -cosine)
    transfer = _tail_pair_0[0]
    transfer_error = _tail_pair_0[1]
    _tail_pair_1 = impulse_tail_two_sum(energy, -outgoing)
    difference = _tail_pair_1[0]
    difference_error = _tail_pair_1[1]
    product = energy * outgoing
    product_error = impulse_tail_product_error(energy, outgoing)
    transferred = product * transfer
    transferred_error = (
        impulse_tail_product_error(product, transfer)
        + product_error * transfer
        + product * transfer_error
        + product_error * transfer_error
    )
    divided = transferred / MASS
    division_error = (
        transferred - divided * MASS - impulse_tail_product_error(divided, MASS) + transferred_error
    ) / MASS
    _tail_pair_2 = impulse_tail_two_sum(divided, -difference)
    numerator = _tail_pair_2[0]
    numerator_error = _tail_pair_2[1]
    numerator += numerator_error + division_error - difference_error
    _tail_pair_3 = impulse_tail_two_sum(wp.float64(2.0) * transferred, difference * difference)
    square = _tail_pair_3[0]
    square_error = _tail_pair_3[1]
    square_error += (
        wp.float64(2.0) * transferred_error
        + impulse_tail_product_error(difference, difference)
        + wp.float64(2.0) * difference * difference_error
        + difference_error * difference_error
    )
    q = wp.sqrt(square)
    q_error = (square - q * q - impulse_tail_product_error(q, q) + square_error) / (
        wp.float64(2.0) * q
    )
    return wp.vec2d(numerator, q + q_error)


@wp.func
def impulse_tail_state_with_transfer_error(
    energy: wp.float64,
    transfer: wp.float64,
    binding: wp.float64,
    height: wp.float64,
    transfer_error: wp.float64,
) -> ImpulseTailState:
    product = energy * (energy - binding)
    product_e = wp.float64(2.0) * energy - binding
    _tail_pair_0 = impulse_tail_kinematic_pair(energy, transfer, binding, transfer_error)
    numerator = _tail_pair_0[0]
    q = _tail_pair_0[1]
    momentum = numerator / q
    span = product * transfer / q * (wp.float64(1.0) / MASS + wp.float64(2.0) / (q + binding))
    _tail_pair_1 = impulse_tail_cutoff(span, height)
    cutoff = _tail_pair_1[0]
    log_fraction = _tail_pair_1[1]
    p_t_over_span = (wp.float64(1.0) / MASS - momentum / q) / (
        transfer * (wp.float64(1.0) / MASS + wp.float64(2.0) / (q + binding))
    )
    cutoff_ratio = (
        wp.float64(1.0) if cutoff == wp.float64(0.0) else -cutoff / -impulse_tail_expm1(cutoff)
    )
    rate = (
        p_t_over_span
        * (wp.float64(1.0) - wp.float64(2.0) * height * momentum)
        / (wp.float64(1.0) + height * (wp.float64(2.0) - span))
        * cutoff_ratio
    )
    score = rate * transfer * product_e / product
    if not wp.isfinite(cutoff):
        p_t = product * (wp.float64(1.0) / (MASS * q) - numerator / (q * q * q))
        p_e = product_e * transfer * (wp.float64(1.0) / (MASS * q) - numerator / (q * q * q))
        rate = (
            wp.float64(2.0)
            * height
            * ((wp.float64(1.0) - wp.float64(2.0) * height * momentum) * p_t)
        )
        score = (
            wp.float64(2.0)
            * height
            * ((wp.float64(1.0) - wp.float64(2.0) * height * momentum) * p_e)
        )
    a = -height * momentum
    log_mass = -wp.float64(2.0) * a * (wp.float64(1.0) + a) + log_fraction - wp.log(wp.float64(2.0))
    result = ImpulseTailState()
    result.binding = binding
    result.height = height
    result.product = product
    result.product_e = product_e
    result.q = q
    result.numerator = numerator
    result.momentum = momentum
    result.span = span
    result.cutoff = cutoff
    result.log_fraction = log_fraction
    result.rate = rate
    result.score = score
    result.log_mass = log_mass
    return result


@wp.func
def impulse_tail_state(
    energy: wp.float64, transfer: wp.float64, binding: wp.float64, height: wp.float64
) -> ImpulseTailState:
    return impulse_tail_state_with_transfer_error(
        energy, transfer, binding, height, wp.float64(0.0)
    )


@wp.func
def impulse_tail_state_at_cosine(
    energy: wp.float64, cosine: wp.float64, binding: wp.float64, height: wp.float64
) -> ImpulseTailState:
    _tail_pair_0 = impulse_tail_two_sum(wp.float64(1.0), -cosine)
    transfer = _tail_pair_0[0]
    error = _tail_pair_0[1]
    return impulse_tail_state_with_transfer_error(energy, transfer, binding, height, error)


@wp.func
def impulse_tail_fraction_score(state: ImpulseTailState, transfer: wp.float64) -> wp.float64:
    if state.cutoff < -wp.float64(745.0):
        return wp.float64(0.0)
    q_e = state.product_e * transfer / state.q
    factor = wp.float64(1.0) / MASS + wp.float64(2.0) / (state.q + state.binding)
    log_span_e = (
        state.product_e / state.product
        - q_e / state.q
        - wp.float64(2.0) * q_e / ((state.q + state.binding) * (state.q + state.binding) * factor)
    )
    phi = wp.float64(1.0)
    if state.cutoff != wp.float64(0.0):
        phi = state.cutoff * wp.exp(state.cutoff) / impulse_tail_expm1(state.cutoff)
    correction = (
        wp.float64(1.0) + state.height * (wp.float64(2.0) - wp.float64(2.0) * state.span)
    ) / (wp.float64(1.0) + state.height * (wp.float64(2.0) - state.span))
    return log_span_e * correction * phi


@wp.func
def impulse_tail_binding_delta(
    energy: wp.float64, transfer: wp.float64, first: ImpulseTailState, second: ImpulseTailState
) -> wp.vec2d:
    difference = first.binding - second.binding
    qsum = first.q + second.q
    v = first.binding + second.binding - wp.float64(2.0) * energy * transfer
    factor = -(energy * transfer / MASS + wp.float64(1.0)) * second.q - second.numerator * v / qsum
    first_q_e = first.product_e * transfer / first.q
    second_q_e = second.product_e * transfer / second.q
    second_n_e = second.product_e * transfer / MASS
    factor_e = (
        -transfer / MASS * second.q
        - (energy * transfer / MASS + wp.float64(1.0)) * second_q_e
        - second_n_e * v / qsum
        + wp.float64(2.0) * transfer * second.numerator / qsum
        + second.numerator * v * (first_q_e + second_q_e) / (qsum * qsum)
    )
    coefficient = difference / (first.q * second.q)
    delta = coefficient * factor
    delta_e = coefficient * (factor_e - factor * (first_q_e / first.q + second_q_e / second.q))
    return wp.vec2d(delta, delta_e)


@wp.func
def impulse_tail_relative_shell(
    energy: wp.float64, transfer: wp.float64, first: ImpulseTailState, reference: ImpulseTailState
) -> wp.vec2d:
    _tail_pair_0 = impulse_tail_binding_delta(energy, transfer, first, reference)
    delta = _tail_pair_0[0]
    delta_e = _tail_pair_0[1]
    height_difference = first.height - reference.height
    a_difference = -height_difference * reference.momentum - first.height * delta
    a_sum = -first.height * first.momentum - reference.height * reference.momentum
    log_weight = (
        -wp.float64(2.0) * a_difference * (wp.float64(1.0) + a_sum)
        + first.log_fraction
        - reference.log_fraction
    )
    reference_p_e = (
        reference.product_e
        * transfer
        * (
            wp.float64(1.0) / (MASS * reference.q)
            - reference.momentum / (reference.q * reference.q)
        )
    )
    first_product = height_difference * reference_p_e + first.height * delta_e
    second_product = height_difference * (
        first.height + reference.height
    ) * reference.momentum * reference_p_e + first.height * first.height * (
        delta * reference_p_e + reference.momentum * delta_e + delta * delta_e
    )
    centre = (
        wp.float64(2.0) * first_product
        - wp.float64(4.0) * second_product
        + impulse_tail_fraction_score(first, transfer)
        - impulse_tail_fraction_score(reference, transfer)
    )
    return wp.vec2d(log_weight, centre)


@wp.func
def impulse_tail_angle_relative_at_transfer(
    energy: wp.float64, state: ImpulseTailState, distance: wp.float64, transfer: wp.float64
) -> wp.vec2d:
    q = wp.sqrt(state.q * state.q - wp.float64(2.0) * state.product * distance)
    factor = state.q / MASS - wp.float64(2.0) * state.numerator / (state.q + q)
    coefficient = state.product * distance / (state.q * q)
    delta = coefficient * factor
    q0_e = wp.float64(2.0) * state.product_e / state.q
    q_e = state.product_e * (wp.float64(2.0) - distance) / q
    factor_e = (
        q0_e / MASS
        - wp.float64(4.0) * state.product_e / (MASS * (state.q + q))
        + wp.float64(2.0) * state.numerator * (q0_e + q_e) / ((state.q + q) * (state.q + q))
    )
    delta_e = (
        delta * (state.product_e / state.product - q0_e / state.q - q_e / q)
        + coefficient * factor_e
    )
    p0_e = (
        wp.float64(2.0)
        * state.product_e
        * (wp.float64(1.0) / (MASS * state.q) - state.momentum / (state.q * state.q))
    )
    log_difference = (
        -wp.float64(2.0)
        * state.height
        * delta
        * (wp.float64(1.0) - wp.float64(2.0) * state.height * state.momentum + state.height * delta)
    )
    log_difference_e = (
        -wp.float64(2.0)
        * state.height
        * (
            delta_e
            * (
                wp.float64(1.0)
                - wp.float64(2.0) * state.height * state.momentum
                + wp.float64(2.0) * state.height * delta
            )
            - wp.float64(2.0) * state.height * delta * p0_e
        )
    )
    if transfer <= wp.float64(0.0):
        return wp.vec2d(-wp.float64(wp.inf), wp.float64(0.0))
    angle = impulse_tail_state(energy, transfer, state.binding, state.height)
    log_relative = log_difference + angle.log_fraction - state.log_fraction
    centred_score = (
        log_difference_e
        + impulse_tail_fraction_score(angle, transfer)
        - impulse_tail_fraction_score(state, wp.float64(2.0))
    )
    return wp.vec2d(log_relative, centred_score)


@wp.func
def impulse_tail_angle_relative(
    energy: wp.float64, state: ImpulseTailState, distance: wp.float64
) -> wp.vec2d:
    return impulse_tail_angle_relative_at_transfer(
        energy, state, distance, wp.float64(2.0) - distance
    )


@wp.func
def impulse_tail_envelope_integral(rate: wp.float64) -> wp.float64:
    return -impulse_tail_expm1(-wp.float64(2.0) * rate) / rate


@wp.func
def impulse_tail_proposal_distance(
    fraction: wp.float64, complement: wp.float64, rate: wp.float64
) -> wp.float64:
    if rate > wp.float64(0.5):
        return -wp.log(complement + fraction * wp.exp(-wp.float64(2.0) * rate)) / rate
    return -impulse_tail_log1p(-fraction * -impulse_tail_expm1(-wp.float64(2.0) * rate)) / rate


@wp.func
def impulse_tail_proposal_cosine(
    fraction: wp.float64, complement: wp.float64, rate: wp.float64
) -> wp.float64:
    if rate > wp.float64(1.0):
        return -wp.float64(1.0) + impulse_tail_proposal_distance(fraction, complement, rate)
    if fraction < wp.float64(0.125):
        return (
            -wp.float64(1.0)
            - impulse_tail_log1p(-fraction * -impulse_tail_expm1(-wp.float64(2.0) * rate)) / rate
        )
    if complement < wp.float64(0.125):
        return (
            wp.float64(1.0)
            - impulse_tail_log1p(complement * impulse_tail_expm1(wp.float64(2.0) * rate)) / rate
        )
    square = rate * rate
    cosh_term = square / wp.float64(2.0)
    sinh_term = rate
    cosh_minus_one = cosh_term
    sinh = sinh_term
    for order in range(1, 20):
        cosh_term *= square / (
            (wp.float64(2.0) * wp.float64(order) + wp.float64(1.0))
            * (wp.float64(2.0) * wp.float64(order) + wp.float64(2.0))
        )
        sinh_term *= square / (
            wp.float64(2.0)
            * wp.float64(order)
            * (wp.float64(2.0) * wp.float64(order) + wp.float64(1.0))
        )
        cosh_minus_one += cosh_term
        sinh += sinh_term
    return -impulse_tail_log1p(cosh_minus_one + (complement - fraction) * sinh) / rate


@wp.func
def impulse_tail_proposal_transfer(
    fraction: wp.float64, complement: wp.float64, rate: wp.float64, cosine: wp.float64
) -> wp.float64:
    if rate <= wp.float64(1.0) and complement < wp.float64(0.125):
        return impulse_tail_log1p(complement * impulse_tail_expm1(wp.float64(2.0) * rate)) / rate
    return wp.float64(1.0) - cosine


@wp.func
def impulse_tail_momentum_deficit(
    energy: wp.float64, cosine: wp.float64, state: ImpulseTailState, uniform: wp.float64
) -> wp.float64:
    complement = wp.float64(1.0) - uniform
    relative_mass = -impulse_tail_expm1(state.cutoff)
    product = complement * relative_mass
    if product <= wp.float64(0.5):
        offset = -impulse_tail_log1p(-product)
        ratio = wp.float64(1.0) if product == wp.float64(0.0) else offset / product
        fraction_ratio = (
            wp.float64(1.0) if state.cutoff == wp.float64(0.0) else relative_mass / -state.cutoff
        )
        offset_over_height = (
            ratio
            * complement
            * fraction_ratio
            * wp.float64(2.0)
            * state.span
            * (wp.float64(1.0) + state.height * (wp.float64(2.0) - state.span))
        )
    else:
        offset = -wp.log(uniform + complement * wp.exp(state.cutoff))
        offset_over_height = offset / state.height
    slope = wp.float64(1.0) - wp.float64(2.0) * state.height * state.momentum
    return offset_over_height / (
        impulse_tail_hypot(slope, wp.sqrt(wp.float64(2.0) * offset)) + slope
    )


@wp.func
def impulse_tail_momentum_span(state: ImpulseTailState, uniform: wp.float64) -> wp.float64:
    size = -state.cutoff
    if size <= wp.float64(0.5):
        relative = impulse_tail_expm1(size)
        product = uniform * relative
        offset = impulse_tail_log1p(product)
        logarithm_ratio = wp.float64(1.0) if product == wp.float64(0.0) else offset / product
        fraction_ratio = wp.float64(1.0) if size == wp.float64(0.0) else relative / size
        offset_over_height = (
            logarithm_ratio
            * uniform
            * fraction_ratio
            * wp.float64(2.0)
            * state.span
            * (wp.float64(1.0) + state.height * (wp.float64(2.0) - state.span))
        )
    else:
        log_product = wp.log(uniform) + size + impulse_tail_log1p(-wp.exp(-size))
        if log_product > wp.float64(0.0):
            offset = log_product + impulse_tail_log1p(wp.exp(-log_product))
        else:
            offset = impulse_tail_log1p(wp.exp(log_product))
        offset_over_height = offset / state.height
    if offset >= wp.float64(2.2250738585072014e-308):
        offset_over_height = offset / state.height
    slope = wp.float64(1.0) + wp.float64(2.0) * state.height
    return offset_over_height / (wp.sqrt(slope * slope - wp.float64(2.0) * offset) + slope)


@wp.func
def impulse_tail_energy_from_span(
    energy: wp.float64, cosine: wp.float64, binding: wp.float64, momentum_span: wp.float64
) -> wp.float64:
    lower = wp.float64(0.0)
    upper = (energy - binding) / energy
    transfer = wp.float64(1.0) - cosine
    alpha = energy / MASS
    for _ in range(120):
        ratio = (lower + upper) * wp.float64(0.5)
        q = wp.sqrt(
            (wp.float64(1.0) - ratio) * (wp.float64(1.0) - ratio)
            + wp.float64(2.0) * ratio * transfer
        )
        span = ratio * transfer / q * (alpha + wp.float64(2.0) / (q + wp.float64(1.0) - ratio))
        if span < momentum_span:
            lower = ratio
        else:
            upper = ratio
    return energy * (lower + upper) * wp.float64(0.5)


@wp.func
def impulse_tail_threshold_energy(
    energy: wp.float64, cosine: wp.float64, state: ImpulseTailState, uniform: wp.float64
) -> wp.float64:
    size = -state.cutoff
    u = uniform
    polynomial = (
        wp.float64(1.0)
        - wp.float64(30.0) * u
        + wp.float64(150.0) * u * u
        - wp.float64(240.0) * (u * u * u)
        + wp.float64(120.0) * (u * u * u * u)
    ) / wp.float64(720.0)
    polynomial = (wp.float64(1.0) - wp.float64(2.0) * u) * (
        wp.float64(1.0) - wp.float64(12.0) * u + wp.float64(12.0) * u * u
    ) / wp.float64(120.0) + size * polynomial
    polynomial = (wp.float64(1.0) - wp.float64(6.0) * u + wp.float64(6.0) * u * u) / wp.float64(
        24.0
    ) + size * polynomial
    polynomial = (wp.float64(1.0) - wp.float64(2.0) * u) / wp.float64(6.0) + size * polynomial
    polynomial = wp.float64(0.5) + size * polynomial
    _tail_pair_0 = impulse_tail_two_sum(u, u * (wp.float64(1.0) - u) * size * polynomial)
    fraction = _tail_pair_0[0]
    fraction_error = _tail_pair_0[1]
    upper = energy - state.binding
    ratio_upper = upper / energy
    curvature = cosine + (wp.float64(1.0) + cosine) / (
        wp.float64(2.0) * (wp.float64(1.0) + energy / MASS)
    )
    correction = (
        fraction
        * (wp.float64(1.0) - fraction)
        * (
            curvature * ratio_upper
            - state.height * state.span / (wp.float64(1.0) + wp.float64(2.0) * state.height)
        )
    )
    _tail_pair_1 = impulse_tail_two_sum(fraction, fraction_error + correction)
    fraction = _tail_pair_1[0]
    fraction_error2 = _tail_pair_1[1]
    outgoing = upper * fraction
    return outgoing + (impulse_tail_product_error(upper, fraction) + upper * fraction_error2)


@wp.func
def impulse_tail_local_delta(
    energy: wp.float64, cosine: wp.float64, outgoing: wp.float64, reference: ImpulseTailState
) -> wp.vec2d:
    distance = cosine + wp.float64(1.0)
    deficit = energy - reference.binding - outgoing
    q0 = reference.q
    n0 = reference.numerator
    q = wp.sqrt(
        (energy - outgoing) * (energy - outgoing)
        + wp.float64(2.0) * energy * outgoing * (wp.float64(1.0) - cosine)
    )
    n_delta = (
        -deficit * (wp.float64(1.0) + wp.float64(2.0) * energy / MASS)
        - energy * outgoing * distance / MASS
    )
    q2_delta = (
        deficit * (wp.float64(2.0) * reference.binding + deficit - wp.float64(4.0) * energy)
        - wp.float64(2.0) * energy * outgoing * distance
    )
    factor = n_delta * q0 - n0 * q2_delta / (q + q0)
    delta = factor / (q * q0)
    q0_e = wp.float64(2.0) * reference.product_e / q0
    q_e = (energy + outgoing * (wp.float64(1.0) - cosine - wp.float64(1.0))) / q
    n_delta_e = (
        -(wp.float64(1.0) + wp.float64(2.0) * energy / MASS)
        - wp.float64(2.0) * deficit / MASS
        - outgoing * distance / MASS
    )
    q2_delta_e = (
        wp.float64(2.0) * reference.binding
        - wp.float64(2.0) * deficit
        - wp.float64(4.0) * energy
        - wp.float64(2.0) * outgoing * distance
    )
    factor_e = (
        n_delta_e * q0
        + n_delta * q0_e
        - wp.float64(2.0) * reference.product_e / MASS * q2_delta / (q + q0)
        - n0 * q2_delta_e / (q + q0)
        + n0 * q2_delta * (q_e + q0_e) / ((q + q0) * (q + q0))
    )
    delta_e = (factor_e - factor * (q_e / q + q0_e / q0)) / (q * q0)
    return wp.vec2d(delta, delta_e)


@wp.func
def impulse_tail_expm1(value: wp.float64) -> wp.float64:
    if wp.abs(value) >= wp.float64(0.5):
        return wp.exp(value) - wp.float64(1.0)
    term = value
    total = value
    correction = wp.float64(0.0)
    for order in range(2, 42):
        term *= value / wp.float64(order)
        pair = impulse_tail_two_sum(total, term)
        total = pair[0]
        correction += pair[1]
    return total + correction


@wp.func
def impulse_tail_log1p(value: wp.float64) -> wp.float64:
    if wp.abs(value) >= wp.float64(0.5):
        pair = impulse_tail_two_sum(wp.float64(1.0), value)
        return wp.log(pair[0]) + pair[1] / pair[0]
    denominator = impulse_tail_two_sum(wp.float64(2.0), value)
    ratio = value / denominator[0]
    ratio_error = (
        (value - ratio * denominator[0])
        - impulse_tail_product_error(ratio, denominator[0])
        - ratio * denominator[1]
    ) / denominator[0]
    square = ratio * ratio
    term = ratio
    total = ratio
    correction = wp.float64(0.0)
    for order in range(1, 42):
        term *= square
        pair = impulse_tail_two_sum(total, term / wp.float64(2 * order + 1))
        total = pair[0]
        correction += pair[1]
    return wp.float64(2.0) * (total + correction + ratio_error / (wp.float64(1.0) - square))


@wp.func
def impulse_tail_hypot(first: wp.float64, second: wp.float64) -> wp.float64:
    scale = wp.max(wp.abs(first), wp.abs(second))
    if scale == wp.float64(0.0):
        return wp.float64(0.0)
    return scale * wp.sqrt((first / scale) * (first / scale) + (second / scale) * (second / scale))


@wp.func
def impulse_tail_reference(
    energy: wp.float64,
    transfer: wp.float64,
    material: int,
    physical: PhysicalParameters,
) -> int:
    occupancy_scale = impulse_occupancy_scale(energy, material, physical)
    if transfer <= wp.float64(0.0):
        return -1
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy > physical.compton_binding[shell]:
            if impulse_momentum_max(
                energy, wp.float64(1.0) - transfer, physical.compton_binding[shell]
            ) >= wp.float64(0.0):
                return -1
    best = -wp.float64(wp.inf)
    reference = int(-1)  # noqa: UP018, RUF046 - mutable Warp loop variable
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy > physical.compton_binding[shell]:
            state = impulse_tail_state(
                energy, transfer, physical.compton_binding[shell], physical.compton_height[shell]
            )
            weight = wp.log(physical.compton_occupancy[shell] / occupancy_scale) + state.log_mass
            if weight > best:
                best = weight
                reference = shell
    return reference


@wp.func
def impulse_tail_normalisation(
    energy: wp.float64,
    material: int,
    physical: PhysicalParameters,
) -> wp.vec4d:
    occupancy_scale = impulse_occupancy_scale(energy, material, physical)
    reference_index = impulse_tail_reference(energy, wp.float64(2.0), material, physical)
    if reference_index < 0:
        return wp.vec4d(wp.float64(wp.nan))
    reference = impulse_tail_state(
        energy,
        wp.float64(2.0),
        physical.compton_binding[reference_index],
        physical.compton_height[reference_index],
    )
    values = wp.vec2d(wp.float64(0.0))
    radius = wp.float64(1.0 / 64.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        binding = physical.compton_binding[shell]
        if energy > binding:
            state = impulse_tail_state(
                energy, wp.float64(2.0), binding, physical.compton_height[shell]
            )
            back = impulse_tail_relative_shell(energy, wp.float64(2.0), state, reference)
            weight = (
                (physical.compton_occupancy[shell] / occupancy_scale)
                * wp.exp(back[0])
                * impulse_tail_envelope_integral(state.rate)
            )
            for panel in range(32):
                midpoint = (wp.float64(panel) + wp.float64(0.5)) / wp.float64(32.0)
                for node in range(8):
                    for sign in range(2):
                        x = midpoint + (
                            wp.float64(2 * sign) - wp.float64(1.0)
                        ) * radius * impulse_gauss16_node(node)
                        y = wp.float64(1.0) - x
                        left = x * x * x * x
                        right = y * y * y * y
                        total = left + right
                        fraction = left / total
                        complement = right / total
                        distance = impulse_tail_proposal_distance(fraction, complement, state.rate)
                        relative = impulse_tail_angle_relative(energy, state, distance)
                        proposal_ratio = wp.exp(relative[0] + state.rate * distance)
                        kn = impulse_kn_transfer_pair(energy, wp.float64(2.0) - distance)
                        jacobian = wp.float64(4.0) * x * x * x * y * y * y / (total * total)
                        coefficient = (
                            weight
                            * radius
                            * impulse_gauss16_weight(node)
                            * jacobian
                            * proposal_ratio
                        )
                        values[0] += coefficient * kn[0]
                        values[1] += coefficient * (kn[1] + kn[0] * (back[1] + relative[1]))
    return wp.vec4d(
        wp.float64(reference_index), TWO_PI * values[0], TWO_PI * values[1], reference.score
    )


@wp.func
def impulse_tail_angle_proposal(
    energy: wp.float64,
    material: int,
    uniform: wp.float64,
    physical: PhysicalParameters,
) -> wp.vec2d:
    occupancy_scale = impulse_occupancy_scale(energy, material, physical)
    reference_index = impulse_tail_reference(energy, wp.float64(2.0), material, physical)
    if reference_index < 0:
        return wp.vec2d(wp.float64(wp.nan))
    reference = impulse_tail_state(
        energy,
        wp.float64(2.0),
        physical.compton_binding[reference_index],
        physical.compton_height[reference_index],
    )
    total = wp.float64(0.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy > physical.compton_binding[shell]:
            state = impulse_tail_state(
                energy,
                wp.float64(2.0),
                physical.compton_binding[shell],
                physical.compton_height[shell],
            )
            back = impulse_tail_relative_shell(energy, wp.float64(2.0), state, reference)
            total += (
                (physical.compton_occupancy[shell] / occupancy_scale)
                * wp.exp(back[0])
                * impulse_tail_envelope_integral(state.rate)
            )
    target = uniform * total
    cumulative = wp.float64(0.0)
    distance = wp.float64(wp.nan)
    cosine = wp.float64(wp.nan)
    transfer = wp.float64(wp.nan)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy > physical.compton_binding[shell]:
            state = impulse_tail_state(
                energy,
                wp.float64(2.0),
                physical.compton_binding[shell],
                physical.compton_height[shell],
            )
            back = impulse_tail_relative_shell(energy, wp.float64(2.0), state, reference)
            weight = (
                (physical.compton_occupancy[shell] / occupancy_scale)
                * wp.exp(back[0])
                * impulse_tail_envelope_integral(state.rate)
            )
            previous = cumulative
            cumulative += weight
            if target < cumulative and weight > wp.float64(0.0):
                trailing = wp.float64(0.0)
                for following in range(shell + 1, physical.compton_shell_offsets[material + 1]):
                    if energy > physical.compton_binding[following]:
                        following_state = impulse_tail_state(
                            energy,
                            wp.float64(2.0),
                            physical.compton_binding[following],
                            physical.compton_height[following],
                        )
                        following_back = impulse_tail_relative_shell(
                            energy, wp.float64(2.0), following_state, reference
                        )
                        trailing += (
                            (physical.compton_occupancy[following] / occupancy_scale)
                            * wp.exp(following_back[0])
                            * impulse_tail_envelope_integral(following_state.rate)
                        )
                fraction = (target - previous) / weight
                complement = ((wp.float64(1.0) - uniform) * total - trailing) / weight
                cosine = impulse_tail_proposal_cosine(fraction, complement, state.rate)
                distance = cosine + wp.float64(1.0)
                transfer = impulse_tail_proposal_transfer(fraction, complement, state.rate, cosine)
                break
    numerator = wp.float64(0.0)
    denominator = wp.float64(0.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy > physical.compton_binding[shell]:
            state = impulse_tail_state(
                energy,
                wp.float64(2.0),
                physical.compton_binding[shell],
                physical.compton_height[shell],
            )
            back = impulse_tail_relative_shell(energy, wp.float64(2.0), state, reference)
            relative = impulse_tail_angle_relative_at_transfer(energy, state, distance, transfer)
            numerator += (physical.compton_occupancy[shell] / occupancy_scale) * wp.exp(
                back[0] + relative[0]
            )
            denominator += (physical.compton_occupancy[shell] / occupancy_scale) * wp.exp(
                back[0] - state.rate * distance
            )
    acceptance = (
        wp.float64(0.5) * impulse_kn_transfer_pair(energy, transfer)[0] * numerator / denominator
    )
    return wp.vec2d(cosine, acceptance)


@wp.func
def impulse_tail_energy_deficit(
    energy: wp.float64,
    cosine: wp.float64,
    binding: wp.float64,
    momentum_deficit: wp.float64,
) -> wp.float64:
    upper = (energy - binding) / energy
    lower = wp.float64(0.0)
    higher = upper
    transfer = wp.float64(1.0) - cosine
    alpha = energy / MASS
    for _iteration in range(80):
        deficit = (lower + higher) * wp.float64(0.5)
        midpoint = upper - deficit * wp.float64(0.5)
        radius = deficit * wp.float64(0.5)
        integral = wp.float64(0.0)
        for node in range(8):
            for sign in range(2):
                r = midpoint + (
                    wp.float64(2 * sign) - wp.float64(1.0)
                ) * radius * impulse_gauss16_node(node)
                q2 = (wp.float64(1.0) - r) * (wp.float64(1.0) - r) + wp.float64(2.0) * r * transfer
                derivative = (
                    transfer
                    * (wp.float64(1.0) + alpha + r * (wp.float64(1.0) - alpha * cosine))
                    / (q2 * wp.sqrt(q2))
                )
                integral += radius * impulse_gauss16_weight(node) * derivative
        if integral < momentum_deficit:
            lower = deficit
        else:
            higher = deficit
    return energy * (lower + higher) * wp.float64(0.5)


@wp.func
def impulse_tail_energy_event(
    energy: wp.float64,
    cosine: wp.float64,
    material: int,
    draws: wp.vec4d,
    physical: PhysicalParameters,
) -> wp.vec4d:
    occupancy_scale = impulse_occupancy_scale(energy, material, physical)
    result = wp.vec4d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(6.0))
    transfer = wp.float64(1.0) - cosine
    if transfer <= wp.float64(0.0) or not wp.isfinite(draws[0]) or not wp.isfinite(draws[1]):
        return result
    reference_index = impulse_tail_reference(energy, transfer, material, physical)
    if reference_index < 0:
        return result
    reference = impulse_tail_state_at_cosine(
        energy,
        cosine,
        physical.compton_binding[reference_index],
        physical.compton_height[reference_index],
    )
    total = wp.float64(0.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy > physical.compton_binding[shell]:
            state = impulse_tail_state_at_cosine(
                energy, cosine, physical.compton_binding[shell], physical.compton_height[shell]
            )
            relative = impulse_tail_relative_shell(energy, transfer, state, reference)
            total += (physical.compton_occupancy[shell] / occupancy_scale) * wp.exp(relative[0])
    target = draws[0] * total
    cumulative = wp.float64(0.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy > physical.compton_binding[shell]:
            state = impulse_tail_state_at_cosine(
                energy, cosine, physical.compton_binding[shell], physical.compton_height[shell]
            )
            relative = impulse_tail_relative_shell(energy, transfer, state, reference)
            weight = (physical.compton_occupancy[shell] / occupancy_scale) * wp.exp(relative[0])
            cumulative += weight
            if target < cumulative and weight > wp.float64(0.0):
                size = -state.cutoff
                outgoing = wp.float64(0.0)
                momentum = wp.float64(0.0)
                if size <= wp.float64(64.0) or wp.log(draws[1]) < -wp.float64(0.5) * size:
                    span = impulse_tail_momentum_span(state, draws[1])
                    if (energy - state.binding) / energy < wp.float64(1e-12) and size < wp.float64(
                        0.001
                    ):
                        outgoing = impulse_tail_threshold_energy(energy, cosine, state, draws[1])
                    else:
                        outgoing = impulse_tail_energy_from_span(
                            energy, cosine, state.binding, span
                        )
                    momentum = -wp.float64(1.0) + span
                else:
                    deficit = impulse_tail_momentum_deficit(energy, cosine, state, draws[1])
                    outgoing = (
                        energy
                        - state.binding
                        - impulse_tail_energy_deficit(energy, cosine, state.binding, deficit)
                    )
                    momentum = state.momentum - deficit
                if (
                    outgoing > wp.float64(0.0)
                    and outgoing <= energy - state.binding
                    and wp.isfinite(outgoing)
                ):
                    result = wp.vec4d(outgoing, state.binding, momentum, wp.float64(0.0))
                break
    return result


@wp.func
def impulse_tail_scatter_event(
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
        proposal = impulse_tail_angle_proposal(energy, material, draws[0], physical)
        if draws[1] < proposal[1]:
            broadening = bound_energy_event(
                energy, proposal[0], material, seed, history, event, 0, physical
            )
            result = wp.vec4d(proposal[0], TWO_PI * draws[2], broadening[0], broadening[3])
            break
    return result


@wp.func
def impulse_tail_phase_pdf(
    energy: wp.float64,
    cosine: wp.float64,
    material: int,
    physical: PhysicalParameters,
) -> wp.float64:
    occupancy_scale = impulse_occupancy_scale(energy, material, physical)
    if cosine < -wp.float64(1.0) or cosine >= wp.float64(1.0):
        return wp.float64(0.0)
    normalisation = impulse_tail_normalisation(energy, material, physical)
    if not wp.isfinite(normalisation[0]):
        return wp.float64(0.0)
    reference_index = int(normalisation[0])
    reference = impulse_tail_state(
        energy,
        wp.float64(2.0),
        physical.compton_binding[reference_index],
        physical.compton_height[reference_index],
    )
    weight = wp.float64(0.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy > physical.compton_binding[shell]:
            state = impulse_tail_state(
                energy,
                wp.float64(2.0),
                physical.compton_binding[shell],
                physical.compton_height[shell],
            )
            back = impulse_tail_relative_shell(energy, wp.float64(2.0), state, reference)
            relative = impulse_tail_angle_relative(energy, state, cosine + wp.float64(1.0))
            weight += (physical.compton_occupancy[shell] / occupancy_scale) * wp.exp(
                back[0] + relative[0]
            )
    return impulse_kn_transfer_pair(energy, wp.float64(1.0) - cosine)[0] * weight / normalisation[1]


@wp.func
def impulse_tail_joint_profile(
    energy: wp.float64,
    cosine: wp.float64,
    outgoing: wp.float64,
    material: int,
    physical: PhysicalParameters,
    normalisation: wp.vec4d,
) -> wp.vec4d:
    occupancy_scale = impulse_occupancy_scale(energy, material, physical)
    reference_index = int(normalisation[0])
    reference = impulse_tail_state(
        energy,
        wp.float64(2.0),
        physical.compton_binding[reference_index],
        physical.compton_height[reference_index],
    )
    deltas = impulse_tail_local_delta(energy, cosine, outgoing, reference)
    delta = deltas[0]
    delta_e = deltas[1]
    momentum = reference.momentum + delta
    reference_p_e = (
        wp.float64(2.0)
        * reference.product_e
        * (
            wp.float64(1.0) / (MASS * reference.q)
            - reference.momentum / (reference.q * reference.q)
        )
    )
    momentum_e = reference_p_e + delta_e
    density = wp.float64(0.0)
    derivative = wp.float64(0.0)
    centred = wp.float64(0.0)
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if outgoing < energy - physical.compton_binding[shell]:
            h = physical.compton_height[shell]
            h0 = reference.height
            difference = h - h0
            a_difference = -difference * reference.momentum - h * delta
            a_sum = -h * momentum - h0 * reference.momentum
            exponent = -wp.float64(2.0) * a_difference * (wp.float64(1.0) + a_sum)
            log_factor = (
                wp.log(wp.float64(2.0))
                + wp.log(h)
                + impulse_log1p(-wp.float64(2.0) * h * momentum)
                - reference.log_fraction
            )
            value = (physical.compton_occupancy[shell] / occupancy_scale) * wp.exp(
                exponent + log_factor
            )
            slope = impulse_profile_log_slope(momentum, h)
            first_product = difference * reference_p_e + h * delta_e
            second_product = difference * (h + h0) * reference.momentum * reference_p_e + h * h * (
                delta * reference_p_e + reference.momentum * delta_e + delta * delta_e
            )
            centre = (
                wp.float64(2.0) * first_product
                - wp.float64(4.0) * second_product
                - wp.float64(2.0)
                * h
                / (wp.float64(1.0) - wp.float64(2.0) * h * momentum)
                * momentum_e
                - impulse_tail_fraction_score(reference, wp.float64(2.0))
            )
            density += value
            derivative += value * slope
            centred += value * centre
    if density <= wp.float64(0.0):
        return wp.vec4d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), momentum)
    return wp.vec4d(density, derivative / density, centred / density, momentum)


@wp.func
def impulse_tail_joint_pdf(
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
    normalisation = impulse_tail_normalisation(energy, material, physical)
    if not wp.isfinite(normalisation[0]):
        return wp.float64(0.0)
    density = impulse_tail_joint_profile(
        energy, cosine, outgoing, material, physical, normalisation
    )[0]
    ratio = outgoing / energy
    factor = wp.float64(1.0) + energy / MASS * (wp.float64(1.0) - cosine)
    q2 = (wp.float64(1.0) - ratio) * (wp.float64(1.0) - ratio) + wp.float64(2.0) * ratio * (
        wp.float64(1.0) - cosine
    )
    jacobian = (factor - cosine + ratio * (wp.float64(1.0) - factor * cosine)) / (
        energy * q2 * wp.sqrt(q2)
    )
    return (
        impulse_kn_transfer_pair(energy, wp.float64(1.0) - cosine)[0]
        * density
        * jacobian
        / normalisation[1]
    )


@wp.func
def impulse_tail_joint_scores(
    energy: wp.float64,
    cosine: wp.float64,
    outgoing: wp.float64,
    material: int,
    physical: PhysicalParameters,
) -> wp.vec3d:
    normalisation = impulse_tail_normalisation(energy, material, physical)
    if not wp.isfinite(normalisation[0]):
        return wp.vec3d(wp.float64(wp.nan))
    profile = impulse_tail_joint_profile(
        energy, cosine, outgoing, material, physical, normalisation
    )
    momentum = profile[3]
    ratio = outgoing / energy
    k = energy / MASS
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
    score_c = kn_c + profile[1] * p_c + n_c / n + wp.float64(3.0) * ratio / q2
    p_r = n / (q * q2)
    score_outgoing = (
        profile[1] * p_r
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
        + profile[2]
        + n_e / n
        - wp.float64(1.0) / energy
        + wp.float64(3.0) * ratio * (ratio - cosine) / (energy * q2)
        - normalisation[2] / normalisation[1]
    )
    for shell in range(
        physical.compton_shell_offsets[material], physical.compton_shell_offsets[material + 1]
    ):
        if energy == physical.compton_binding[shell]:
            score_incident = wp.float64(wp.nan)
    return wp.vec3d(score_c, score_outgoing, score_incident)
