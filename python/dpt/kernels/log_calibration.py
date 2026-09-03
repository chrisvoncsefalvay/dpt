"""Fused flat/dark logarithmic correction and cancellation-safe seeded derivatives."""

from functools import cache

import warp as wp

from dpt.kernels.transmission import _optical_depth_native

STRICT = {"fast_math": False, "fuse_fp": False, "enable_backward": False}


@wp.func_native("""
if (seed == 0.0 || difference == 0.0) return seed * difference;
// Retained frames have finite positive corrected intensities. Decompose each
// factor before arithmetic so no intermediate quotient or product loses range.
const auto mantissa = [&](double value, int& exponent) {
    int correction = 0;
    if (wp::abs(value) < 2.2250738585072014e-308) {
        value *= 18014398509481984.0;
        correction = 54;
    }
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    exponent = static_cast<int>((bits >> 52) & 0x7ff) - 1022 - correction;
    bits = (bits & 0x800fffffffffffffull) | 0x3fe0000000000000ull;
    memcpy(&value, &bits, sizeof(value));
    return value;
};
int seed_exponent, difference_exponent, corrected_exponent, reference_exponent;
const double seed_mantissa = mantissa(seed, seed_exponent);
const double difference_mantissa = mantissa(difference, difference_exponent);
const double corrected_mantissa = mantissa(corrected, corrected_exponent);
const double reference_mantissa = mantissa(reference, reference_exponent);
const double product_quotient = (seed_mantissa * difference_mantissa)
    / (corrected_mantissa * reference_mantissa);
return ldexp(product_quotient,
    seed_exponent + difference_exponent - corrected_exponent - reference_exponent);
""")
def _weak_dark_cotangent(
    seed: wp.float64, difference: wp.float64, corrected: wp.float64, reference: wp.float64
) -> wp.float64:
    """Compute g * (I0-I) / ((I-D)*(I0-D)) with one final exponent scaling."""
    ...


@cache
def get_prepare(double: bool):
    dtype = wp.float64 if double else wp.float32

    @wp.kernel(module="unique", module_options=STRICT)
    def prepare(
        flat: wp.array(dtype=dtype),
        dark: wp.array(dtype=dtype),
        corrected_flat: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        illumination = wp.float64(flat[pixel])
        background = wp.float64(dark[pixel])
        corrected = illumination - background
        corrected_flat[pixel] = corrected
        if (
            not wp.isfinite(illumination)
            or not wp.isfinite(background)
            or not wp.isfinite(corrected)
            or corrected <= wp.float64(0.0)
        ):
            wp.atomic_or(status, 0, 1)

    return prepare


@cache
def get_forward(double: bool):
    dtype = wp.float64 if double else wp.float32

    @wp.kernel(module="unique", module_options=STRICT)
    def forward(
        signal: wp.array(dtype=dtype),
        flat: wp.array(dtype=dtype),
        dark: wp.array(dtype=dtype),
        corrected_flat: wp.array(dtype=wp.float64),
        optical_depth: wp.array(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        intensity = wp.float64(signal[pixel])
        corrected = intensity - wp.float64(dark[pixel])
        reference = corrected_flat[pixel]
        depth = wp.float64(wp.nan)
        if not wp.isfinite(intensity) or not wp.isfinite(corrected) or corrected <= wp.float64(0.0):
            wp.atomic_or(status, 0, 1)
        else:
            difference = wp.float64(flat[pixel]) - intensity
            if wp.abs(difference) <= wp.float64(0.5) * reference:
                depth = _optical_depth_native(difference / reference)
            else:
                # A quotient can overflow or underflow although its log is finite.
                depth = wp.log(reference) - wp.log(corrected)
        rounded = dtype(depth)
        optical_depth[pixel] = rounded
        if not wp.isfinite(rounded):
            wp.atomic_or(status, 0, 2)

    return forward


@cache
def get_vjp(double: bool):
    dtype = wp.float64 if double else wp.float32

    @wp.kernel(module="unique", module_options=STRICT)
    def vjp(
        signal: wp.array(dtype=dtype),
        flat: wp.array(dtype=dtype),
        dark: wp.array(dtype=dtype),
        corrected_flat: wp.array(dtype=wp.float64),
        cotangent: wp.array(dtype=dtype),
        grad_signal: wp.array(dtype=dtype),
        grad_flat: wp.array(dtype=dtype),
        grad_dark: wp.array(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        seed = wp.float64(cotangent[pixel])
        signal_value = wp.float64(0.0)
        flat_value = wp.float64(0.0)
        dark_value = wp.float64(0.0)
        if not wp.isfinite(seed):
            wp.atomic_or(status, 0, 1)
            signal_value = wp.float64(wp.nan)
            flat_value = wp.float64(wp.nan)
            dark_value = wp.float64(wp.nan)
        elif seed != wp.float64(0.0):
            intensity = wp.float64(signal[pixel])
            corrected = intensity - wp.float64(dark[pixel])
            reference = corrected_flat[pixel]
            positive_signal = seed / corrected
            signal_value = -positive_signal
            flat_value = seed / reference
            difference = wp.float64(flat[pixel]) - intensity
            if wp.abs(difference) <= wp.float64(0.5) * reference:
                # Preserve weak sensitivity both across reciprocal cancellation
                # and when a large seed rescues an underflowed contrast ratio.
                dark_value = _weak_dark_cotangent(seed, difference, corrected, reference)
            else:
                dark_value = positive_signal - flat_value
        rounded_signal = dtype(signal_value)
        rounded_flat = dtype(flat_value)
        rounded_dark = dtype(dark_value)
        grad_signal[pixel] = rounded_signal
        grad_flat[pixel] = rounded_flat
        grad_dark[pixel] = rounded_dark
        if (
            not wp.isfinite(rounded_signal)
            or not wp.isfinite(rounded_flat)
            or not wp.isfinite(rounded_dark)
        ):
            wp.atomic_or(status, 0, 2)

    return vjp
