"""Binary64 polychromatic calibration, bounded inversion and analytic VJPs."""

import warp as wp

from dpt.kernels.transmission import _optical_depth_native, _removed_primary_native

STRICT = {"fast_math": False, "fuse_fp": False, "enable_backward": False}


@wp.func_native("""
double factors[3] = {a, b, c};
int exponent = 0;
for (int i = 0; i < 3; ++i) {
    double value = factors[i];
    if (value == 0.0) return -4096;
    int correction = 0;
    if (value < 2.2250738585072014e-308) {
        value *= 18014398509481984.0;
        correction = 54;
    }
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    exponent += static_cast<int>((bits >> 52) & 0x7ff) - 1022 - correction;
}
return exponent;
""")
def product_exponent(a: wp.float64, b: wp.float64, c: wp.float64) -> int: ...


@wp.func_native("""
double factors[3] = {a, b, c};
double mantissa = 1.0;
int exponent = -scale_exponent;
for (int i = 0; i < 3; ++i) {
    double value = factors[i];
    if (value == 0.0) return 0.0;
    int correction = 0;
    if (value < 2.2250738585072014e-308) {
        value *= 18014398509481984.0;
        correction = 54;
    }
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    exponent += static_cast<int>((bits >> 52) & 0x7ff) - 1022 - correction;
    bits = (bits & 0x000fffffffffffffull) | 0x3fe0000000000000ull;
    memcpy(&value, &bits, sizeof(value));
    mantissa *= value;
}
return ldexp(mantissa, exponent);
""")
def scaled_product(
    a: wp.float64, b: wp.float64, c: wp.float64, scale_exponent: int
) -> wp.float64: ...


@wp.func
def transmitted_exponent(coefficient: wp.float64, weight: wp.float64, log_scale: wp.float64):
    if log_scale >= wp.float64(-700.0):
        return product_exponent(coefficient, weight, wp.exp(log_scale))
    # A pivot term has a positive binary64 coefficient. Contributions this
    # small cannot affect its mean even at the maximum int32 energy count.
    if log_scale < wp.float64(-3000.0):
        return -4096
    shift = int(wp.floor(log_scale / wp.float64(0.6931471805599453)))
    return product_exponent(coefficient, weight, wp.float64(1.0)) + shift


@wp.func
def scaled_transmitted(
    coefficient: wp.float64,
    weight: wp.float64,
    log_scale: wp.float64,
    scale_exponent: int,
):
    if log_scale >= wp.float64(-700.0):
        return scaled_product(coefficient, weight, wp.exp(log_scale), scale_exponent)
    if log_scale < wp.float64(-3000.0):
        return wp.float64(0.0)
    shift = int(wp.floor(log_scale / wp.float64(0.6931471805599453)))
    remainder = wp.exp(log_scale - wp.float64(shift) * wp.float64(0.6931471805599453))
    return scaled_product(coefficient, weight, remainder, scale_exponent - shift)


@wp.func_native("return ldexp(value, exponent);")
def rescale(value: wp.float64, exponent: int) -> wp.float64: ...


@wp.func
def calibration_value(
    length: wp.float64,
    coefficients: wp.array(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    log_weights: wp.array(dtype=wp.float64),
    maximum_coefficient: wp.float64,
) -> wp.vec2d:
    """Return depth and its strictly positive length derivative."""
    depth = wp.float64(0.0)
    total = wp.float64(0.0)
    correction = wp.float64(0.0)
    moment = wp.float64(0.0)
    moment_correction = wp.float64(0.0)
    moment_exponent = int(-4096)  # noqa: UP018, RUF046 - Warp dynamic reduction variable.  # noqa: UP018, RUF046 - Warp requires mutable loop variables.
    loss_exponent = int(-4096)  # noqa: UP018, RUF046 - Warp dynamic reduction variable.  # noqa: UP018, RUF046 - Warp requires mutable loop variables.
    # Branch on the detected loss, rather than the most attenuated node, so
    # a rare opaque node cannot erase a weak but representable optical depth.
    for energy in range(coefficients.shape[0]):
        removed = _removed_primary_native(coefficients[energy] * length)
        loss_exponent = wp.max(
            loss_exponent, product_exponent(weights[energy], removed, wp.float64(1.0))
        )
    for energy in range(coefficients.shape[0]):
        removed = _removed_primary_native(coefficients[energy] * length)
        term = scaled_product(weights[energy], removed, wp.float64(1.0), loss_exponent) - correction
        updated = total + term
        correction = (updated - total) - term
        total = updated
    total = rescale(total, loss_exponent)
    if total <= wp.float64(0.5):
        for energy in range(coefficients.shape[0]):
            moment_exponent = wp.max(
                moment_exponent,
                transmitted_exponent(
                    coefficients[energy],
                    weights[energy],
                    -coefficients[energy] * length,
                ),
            )
        for energy in range(coefficients.shape[0]):
            term = (
                scaled_transmitted(
                    coefficients[energy],
                    weights[energy],
                    -coefficients[energy] * length,
                    moment_exponent,
                )
                - moment_correction
            )
            updated = moment + term
            moment_correction = (updated - moment) - term
            moment = updated
        depth = _optical_depth_native(total)
        moment = rescale(moment / (wp.float64(1.0) - total), moment_exponent)
    else:
        total = wp.float64(0.0)
        correction = wp.float64(0.0)
        pivot = wp.float64(-wp.inf)
        for energy in range(coefficients.shape[0]):
            pivot = wp.max(pivot, log_weights[energy] - coefficients[energy] * length)
        for energy in range(coefficients.shape[0]):
            moment_exponent = wp.max(
                moment_exponent,
                transmitted_exponent(
                    coefficients[energy],
                    wp.float64(1.0),
                    log_weights[energy] - coefficients[energy] * length - pivot,
                ),
            )
        for energy in range(coefficients.shape[0]):
            scaled = wp.exp(log_weights[energy] - coefficients[energy] * length - pivot)
            term = scaled - correction
            updated = total + term
            correction = (updated - total) - term
            total = updated
            term = (
                scaled_transmitted(
                    coefficients[energy],
                    wp.float64(1.0),
                    log_weights[energy] - coefficients[energy] * length - pivot,
                    moment_exponent,
                )
                - moment_correction
            )
            updated = moment + term
            moment_correction = (updated - moment) - term
            moment = updated
        depth = -pivot - wp.log(total)
        moment = rescale(moment / total, moment_exponent)
    return wp.vec2d(depth, moment)


@wp.kernel(module="unique", module_options=STRICT)
def forward(
    length: wp.array(dtype=wp.float64),
    coefficients: wp.array(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    log_weights: wp.array(dtype=wp.float64),
    maximum_coefficient: wp.float64,
    maximum_length: wp.float64,
    depth: wp.array(dtype=wp.float64),
    slope: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    pixel = wp.tid()
    value = length[pixel]
    if not wp.isfinite(value) or value < wp.float64(0.0) or value > maximum_length:
        wp.atomic_or(status, 0, 1)
        depth[pixel] = wp.float64(wp.nan)
        slope[pixel] = wp.float64(wp.nan)
    else:
        result = calibration_value(value, coefficients, weights, log_weights, maximum_coefficient)
        depth[pixel] = result[0]
        slope[pixel] = result[1]
        if (
            not wp.isfinite(result[0])
            or not wp.isfinite(result[1])
            or result[1] <= wp.float64(0.0)
            or (value > wp.float64(0.0) and result[0] == wp.float64(0.0))
        ):
            wp.atomic_or(status, 0, 2)


@wp.kernel(module="unique", module_options=STRICT)
def inverse(
    depth: wp.array(dtype=wp.float64),
    coefficients: wp.array(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    log_weights: wp.array(dtype=wp.float64),
    maximum_coefficient: wp.float64,
    maximum_length: wp.float64,
    maximum_iterations: int,
    length: wp.array(dtype=wp.float64),
    slope: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    pixel = wp.tid()
    target = depth[pixel]
    upper_value = calibration_value(
        maximum_length, coefficients, weights, log_weights, maximum_coefficient
    )[0]
    if not wp.isfinite(target) or target < wp.float64(0.0) or target > upper_value:
        wp.atomic_or(status, 0, 1)
        length[pixel] = wp.float64(wp.nan)
        slope[pixel] = wp.float64(wp.nan)
    else:
        low = wp.float64(0.0)
        high = maximum_length
        if target == wp.float64(0.0):
            high = wp.float64(0.0)
        elif target == upper_value:
            low = maximum_length
        for _iteration in range(maximum_iterations):
            middle = low + (high - low) * wp.float64(0.5)
            if middle == low or middle == high:
                break
            value = calibration_value(
                middle, coefficients, weights, log_weights, maximum_coefficient
            )[0]
            if value < target:
                low = middle
            elif value > target:
                high = middle
            else:
                low = middle
                high = middle
                break
        estimate = low + (high - low) * wp.float64(0.5)
        result = calibration_value(
            estimate, coefficients, weights, log_weights, maximum_coefficient
        )
        length[pixel] = estimate
        slope[pixel] = result[1]
        if not wp.isfinite(estimate) or not wp.isfinite(result[1]) or result[1] <= wp.float64(0.0):
            wp.atomic_or(status, 0, 2)
        # A caller choosing too few iterations gets a numerical failure rather
        # than a silently approximate calibration. Resolve the binary64 bracket.
        if (
            low != high
            and low + (high - low) * wp.float64(0.5) != low
            and low + (high - low) * wp.float64(0.5) != high
        ):
            wp.atomic_or(status, 0, 4)


@wp.kernel(module="unique", module_options=STRICT)
def vjp(
    seed: wp.array(dtype=wp.float64),
    slope: wp.array(dtype=wp.float64),
    invert: int,
    gradient: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    pixel = wp.tid()
    value = seed[pixel]
    result = wp.float64(0.0)
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, 1)
        result = wp.float64(wp.nan)
    elif value != wp.float64(0.0):
        if invert == 1:
            result = value / slope[pixel]
        else:
            result = value * slope[pixel]
    gradient[pixel] = result
    if not wp.isfinite(result):
        wp.atomic_or(status, 0, 2)
