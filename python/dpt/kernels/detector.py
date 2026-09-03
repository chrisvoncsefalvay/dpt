"""Detector calibration, zero-extended spatial response and separate observations.

No estimator differentiates a realised count draw. Poisson draws use counter
addresses shared with transport, a bounded rejection loop and an explicit error
on budget exhaustion. There is no Gaussian replacement for a Poisson tail.
"""

# Warp annotations are executable DSL expressions; host interfaces remain strict.
# The optional GPU import is resolved only when an operator is prepared.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false

from functools import cache

import warp as wp

from dpt.kernels.random import random4
from dpt.kernels.spectral import checked_store
from dpt.kernels.spectral_common import compensated_sum_update

STRICT = {"fast_math": False, "fuse_fp": True, "enable_backward": False}
wp.set_module_options(STRICT)


@wp.kernel
def check_positive(values: wp.array(dtype=wp.float32), status: wp.array(dtype=wp.int32)):
    p = wp.tid()
    if not wp.isfinite(values[p]) or values[p] <= wp.float32(0.0):
        wp.atomic_or(status, 0, 1)


@cache
def get_rate_check(wide: bool):
    dtype = wp.float64 if wide else wp.float32

    @wp.kernel(module="unique", module_options=STRICT)
    def check(values: wp.array(dtype=dtype), status: wp.array(dtype=wp.int32)):
        p = wp.tid()
        if not wp.isfinite(values[p]) or values[p] < dtype(0.0) or values[p] > dtype(1.0e9):
            wp.atomic_or(status, 0, 1)

    return check


@cache
def _get_compound_input_check(wide: bool, equal_extents: bool, rates_present: bool):
    dtype = wp.float64 if wide else wp.float32

    @wp.kernel(module="unique", module_options=STRICT)
    def check(
        rates: wp.array(dtype=dtype),
        scores: wp.array(dtype=wp.float32),
        status: wp.array(dtype=wp.int32),
    ):
        index = wp.tid()
        if wp.static(rates_present):
            value = rates[index]
            if not wp.isfinite(value) or value < dtype(0.0) or value > dtype(1.0e9):
                wp.atomic_or(status, 0, 1)
        if wp.static(equal_extents or not rates_present):
            score = scores[index]
            if not wp.isfinite(score) or score < wp.float32(0.0):
                wp.atomic_or(status, 0, 1)
        else:
            if index < scores.shape[0]:
                score = scores[index]
                if not wp.isfinite(score) or score < wp.float32(0.0):
                    wp.atomic_or(status, 0, 1)

    return check


@cache
def get_signal_cast(input64: bool, output64: bool):
    input_dtype = wp.float64 if input64 else wp.float32
    output_dtype = wp.float64 if output64 else wp.float32

    @wp.kernel(module="unique", module_options=STRICT)
    def cast(
        source: wp.array(dtype=input_dtype),
        output: wp.array(dtype=output_dtype),
        status: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        value = output_dtype(source[p])
        output[p] = value
        if not wp.isfinite(value):
            wp.atomic_or(status, 0, 2)

    return cast


@cache
def get_calibration(shared_gain: bool, shared_offset: bool):
    @wp.kernel(module="unique", module_options=STRICT)
    def calibration(
        mean: wp.array(dtype=wp.float32),
        gain: wp.array(dtype=wp.float32),
        exposure: wp.array(dtype=wp.float32),
        offset: wp.array(dtype=wp.float32),
        output: wp.array(dtype=wp.float32),
        status: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        gi = p
        oi = p
        if wp.static(shared_gain):
            gi = 0
        if wp.static(shared_offset):
            oi = 0
        signal = wp.float64(gain[gi]) * wp.float64(exposure[0]) * wp.float64(mean[p]) + wp.float64(
            offset[oi]
        )
        output[p] = checked_store(signal, status)

    return calibration


@cache
def get_calibration_pixel_vjp(shared_gain: bool, write_mean: bool):
    @wp.kernel(module="unique", module_options=STRICT)
    def vjp(
        gain: wp.array(dtype=wp.float32),
        exposure: wp.array(dtype=wp.float32),
        seed: wp.array(dtype=wp.float32),
        output: wp.array(dtype=wp.float32),
        status: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        gi = p
        if wp.static(shared_gain):
            gi = 0
        if wp.static(write_mean):
            output[p] = checked_store(
                wp.float64(seed[p]) * wp.float64(gain[gi]) * wp.float64(exposure[0]),
                status,
            )

    return vjp


@cache
def get_calibration_partials(shared_gain: bool, kind: int):
    @wp.kernel(module="unique", module_options=STRICT)
    def partials(
        mean: wp.array(dtype=wp.float32),
        gain: wp.array(dtype=wp.float32),
        exposure: wp.array(dtype=wp.float32),
        seed: wp.array(dtype=wp.float32),
        pixels: int,
        groups: int,
        output: wp.array(dtype=wp.float64),
    ):
        group, lane = wp.tid()
        total = wp.float64(0.0)
        compensation = wp.float64(0.0)
        p = group * wp.block_dim() + lane
        while p < pixels:
            gi = p
            if wp.static(shared_gain):
                gi = 0
            term = wp.float64(seed[p])
            if wp.static(kind == 0):
                term = term * wp.float64(exposure[0]) * wp.float64(mean[p])
            elif wp.static(kind == 1):
                term = term * wp.float64(gain[gi]) * wp.float64(mean[p])
            total, compensation = compensated_sum_update(total, compensation, term)
            # Do not overflow the final int32 stride at maximum image capacity.
            if pixels - p <= groups * wp.block_dim():
                break
            p = p + groups * wp.block_dim()
        values = wp.tile(total)
        total_tile = wp.tile_sum(values)
        wp.tile_store(output, total_tile, offset=group)

    return partials


@wp.func
def blur_value(
    source: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float64),
    p: int,
    height: int,
    width: int,
    kernel_height: int,
    kernel_width: int,
    transpose: bool,
) -> wp.float64:
    row, column = p // width, p % width
    total, compensation = wp.float64(0.0), wp.float64(0.0)
    for kr in range(kernel_height):
        for kc in range(kernel_width):
            dr, dc = kr - kernel_height // 2, kc - kernel_width // 2
            if transpose:
                dr, dc = -dr, -dc
            sr, sc = row + dr, column + dc
            # Zero extension uses the same offsets in B and their reversals in B^T.
            if sr >= 0 and sr < height and sc >= 0 and sc < width:
                term = weights[kr * kernel_width + kc] * wp.float64(source[sr * width + sc])
                total, compensation = compensated_sum_update(total, compensation, term)
    return total


@cache
def get_blur_dynamic(kernel_height: int, kernel_width: int, transpose: bool):
    @wp.kernel(module="unique", module_options=STRICT)
    def blur(
        source: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float64),
        height: int,
        width: int,
        output: wp.array(dtype=wp.float32),
        status: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        output[p] = checked_store(
            blur_value(source, weights, p, height, width, kernel_height, kernel_width, transpose),
            status,
        )

    return blur


@cache
def get_separable_blur(length: int, vertical: bool, transpose: bool, final: bool):
    """One axis pass, with detector dimensions supplied at launch time."""
    input_dtype = wp.float64 if final else wp.float32
    output_dtype = wp.float32 if final else wp.float64

    @wp.kernel(module="unique", module_options=STRICT)
    def apply(
        source: wp.array(dtype=input_dtype),
        weights: wp.array(dtype=wp.float64),
        offset: int,
        height: int,
        width: int,
        output: wp.array(dtype=output_dtype),
        status: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        row, column = p // width, p % width
        total = wp.float64(0.0)
        compensation = wp.float64(0.0)
        for k in range(length):
            shift = k - length // 2
            if wp.static(transpose):
                shift = -shift
            r, c = row, column
            if wp.static(vertical):
                r = r + shift
            else:
                c = c + shift
            if r >= 0 and r < height and c >= 0 and c < width:
                term = weights[offset + k] * wp.float64(source[r * width + c])
                total, compensation = compensated_sum_update(total, compensation, term)
        if wp.static(final):
            output[p] = checked_store(total, status)
        else:
            output[p] = total
            if not wp.isfinite(total):
                wp.atomic_or(status, 0, 2)

    return apply


@wp.func
def uniform_pair(seed: wp.uint64, identity: wp.uint64, event: wp.uint32, domain: wp.uint32):
    """Two open uniforms using 53 counter bits each, without endpoint clamping."""
    draw = random4(seed, identity, event, domain)
    scale32 = wp.float64(4294967296.0)
    scale21 = wp.float64(2097152.0)
    denominator = wp.float64(9007199254740994.0)
    first = (
        wp.floor(draw[0] * scale32) * scale21 + wp.floor(draw[1] * scale21) + wp.float64(1.0)
    ) / denominator
    second = (
        wp.floor(draw[2] * scale32) * scale21 + wp.floor(draw[3] * scale21) + wp.float64(1.0)
    ) / denominator
    return wp.vec2d(first, second)


@wp.func
def poisson_log_probability(count: wp.float64, rate: wp.float64) -> wp.float64:
    """Stable log mass for a non-negative integer and a positive Poisson mean.

    Loader (2002), equations (4), (6), (7): express the mass through Stirling
    error and deviance before evaluating it. Subtracting count*log(rate) and
    lgamma(count+1) directly discards useful digits near a large mean.
    https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf
    """
    if count == wp.float64(0.0):
        return -rate
    if count < wp.float64(16.0):
        log_factorial = wp.float64(0.0)
        for integer in range(2, int(count) + 1):
            log_factorial += wp.log(wp.float64(integer))
        return -rate + count * wp.log(rate) - log_factorial
    inverse = wp.float64(1.0) / count
    square = inverse * inverse
    # Cast both operands: casting a quotient lets Warp divide in FP32 first.
    # Six Bernoulli terms; at count >= 16 the next term bounds the absolute
    # truncation error by 7/(1092*16**13) < 1.5e-18 before FP64 rounding.
    correction = inverse * (
        (wp.float64(1.0) / wp.float64(12.0))
        - square
        * (
            (wp.float64(1.0) / wp.float64(360.0))
            - square
            * (
                (wp.float64(1.0) / wp.float64(1260.0))
                - square
                * (
                    (wp.float64(1.0) / wp.float64(1680.0))
                    - square
                    * (
                        (wp.float64(1.0) / wp.float64(1188.0))
                        - square * (wp.float64(691.0) / wp.float64(360360.0))
                    )
                )
            )
        )
    )
    difference = count - rate
    deviance = wp.float64(0.0)
    if wp.abs(difference) < wp.float64(0.1) * (count + rate):
        ratio = difference / (count + rate)
        ratio_squared = ratio * ratio
        deviance = difference * ratio
        term = wp.float64(2.0) * count * ratio
        # |ratio| < .1 gives geometric convergence; 32 terms make the
        # omitted relative tail smaller than binary64 precision throughout.
        for order in range(1, 33):
            term = term * ratio_squared
            updated = deviance + term / wp.float64(2 * order + 1)
            if updated == deviance:
                break
            deviance = updated
    else:
        deviance = count * wp.log(count / rate) - difference
    return (
        -wp.float64(0.5) * wp.log(wp.float64(6.283185307179586476925286766559) * count)
        - correction
        - deviance
    )


@wp.func
def poisson_inverse_small(rate: wp.float64, u: wp.float64) -> wp.uint64:
    """Invert a small-rate Poisson without subtracting its far upper tail.

    The open uniform has 53 bits. Above its midpoint, sum backwards from an
    upper-tail term smaller than 2^-60 times the target tail probability. For
    rate < 10 this requires fewer than 128 terms. A rounded lower CDF near one
    therefore cannot manufacture rejection-budget exhaustion.
    """
    probability = wp.exp(-rate)
    count = int(0)  # noqa: UP018, RUF046 - mutable Warp loop variable
    if u <= probability:
        return wp.uint64(0)
    if u <= wp.float64(0.5):
        cumulative = probability
        while u > cumulative and count < 128:
            count = count + 1
            probability = probability * rate / wp.float64(count)
            cumulative = cumulative + probability
        return wp.uint64(count)
    target = wp.float64(1.0) - u
    threshold = target * wp.float64(8.673617379884035e-19)  # 2^-60
    while (wp.float64(count) <= rate or probability > threshold) and count < 128:
        count = count + 1
        probability = probability * rate / wp.float64(count)
    tail = probability
    while tail <= target and count > 0:
        probability = probability * wp.float64(count) / rate
        count = count - 1
        tail = tail + probability
    return wp.uint64(count)


@wp.func
def poisson_draw(
    rate: wp.float64,
    seed: wp.uint64,
    identity: wp.uint64,
    domain: wp.uint32,
    budget: int,
    status: wp.array(dtype=wp.int32),
) -> wp.uint64:
    if rate == wp.float64(0.0):
        return wp.uint64(0)
    if not wp.isfinite(rate) or rate < wp.float64(0.0) or rate > wp.float64(1.0e9):
        wp.atomic_or(status, 0, 1)
        return wp.uint64(0)
    if rate < wp.float64(10.0):
        u = uniform_pair(seed, identity, wp.uint32(0), domain)[0]
        return poisson_inverse_small(rate, u)
    else:
        # Hörmann's transformed rejection with squeeze (PTRS), 1993,
        # doi:10.1016/0167-6687(93)90997-4. All acceptance arithmetic is FP64.
        root = wp.sqrt(rate)
        b = wp.float64(0.931) + wp.float64(2.53) * root
        a = wp.float64(-0.059) + wp.float64(0.02483) * b
        inverse_alpha = wp.float64(1.1239) + wp.float64(1.1328) / (b - wp.float64(3.4))
        squeeze = wp.float64(0.9277) - wp.float64(3.6224) / (b - wp.float64(2.0))
        for trial in range(budget):
            uniforms = uniform_pair(seed, identity, wp.uint32(trial), domain)
            u = uniforms[0] - wp.float64(0.5)
            v = uniforms[1]
            distance = wp.float64(0.5) - wp.abs(u)
            candidate = wp.floor((wp.float64(2.0) * a / distance + b) * u + rate + wp.float64(0.43))
            if candidate >= wp.float64(0.0):
                if distance >= wp.float64(0.07) and v <= squeeze:
                    return wp.uint64(candidate)
                if not (distance < wp.float64(0.013) and v > distance):
                    lhs = wp.log(v * inverse_alpha / (a / (distance * distance) + b))
                    rhs = poisson_log_probability(candidate, rate)
                    if lhs <= rhs:
                        return wp.uint64(candidate)
    # A bounded failure is an invalid realisation, never an observed zero.
    wp.atomic_or(status, 0, 4)
    return wp.uint64(0)


@cache
def get_poisson_counts(wide: bool):
    dtype = wp.float64 if wide else wp.float32

    @wp.kernel(module="unique", module_options=STRICT)
    def counts(
        mean: wp.array(dtype=dtype),
        seed: wp.uint64,
        observation: wp.uint32,
        pixel_offset: wp.uint32,
        energy_offset: wp.uint32,
        budget: int,
        output: wp.array(dtype=wp.uint64),
        status: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        identity = (wp.uint64(observation) << wp.uint64(32)) | wp.uint64(
            pixel_offset + wp.uint32(p)
        )
        output[p] = poisson_draw(
            wp.float64(mean[p]), seed, identity, wp.uint32(1) + energy_offset, budget, status
        )

    return counts


@cache
def get_compound_poisson(energies: int, shared_scores: bool, wide: bool = False):
    from dpt.kernels.spectral_common import (
        compensated_product_sum_update,
        energy_pixel_index,
    )

    dtype = wp.float64 if wide else wp.float32

    @wp.kernel(module="unique", module_options=STRICT)
    def compound(
        rates: wp.array(dtype=dtype),
        scores: wp.array(dtype=wp.float32),
        pixels: int,
        seed: wp.uint64,
        observation: wp.uint32,
        pixel_offset: wp.uint32,
        energy_offset: wp.uint32,
        budget: int,
        output: wp.array(dtype=wp.float32),
        status: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        identity = (wp.uint64(observation) << wp.uint64(32)) | wp.uint64(
            pixel_offset + wp.uint32(p)
        )
        total = wp.float64(0.0)
        compensation = wp.float64(0.0)
        for energy in range(energies):
            si = energy_pixel_index(energy, p, pixels)
            if wp.static(shared_scores):
                si = energy
            count = poisson_draw(
                wp.float64(rates[energy_pixel_index(energy, p, pixels)]),
                seed,
                identity,
                wp.uint32(1) + energy_offset + wp.uint32(energy),
                budget,
                status,
            )
            total, compensation = compensated_product_sum_update(
                total, compensation, wp.float64(count), wp.float64(scores[si])
            )
        output[p] = checked_store(total, status)

    return compound


@wp.kernel
def gaussian_read_noise(
    signal: wp.array(dtype=wp.float32),
    sigma: wp.float64,
    seed: wp.uint64,
    observation: wp.uint32,
    pixel_offset: wp.uint32,
    output: wp.array(dtype=wp.float32),
    status: wp.array(dtype=wp.int32),
):
    p = wp.tid()
    identity = (wp.uint64(observation) << wp.uint64(32)) | wp.uint64(pixel_offset + wp.uint32(p))
    uniforms = uniform_pair(seed, identity, wp.uint32(0), wp.uint32(0x7FFFFFFF))
    normal = wp.sqrt(-wp.float64(2.0) * wp.log(uniforms[0])) * wp.cos(
        wp.float64(6.283185307179586476925286766559) * uniforms[1]
    )
    output[p] = checked_store(wp.float64(signal[p]) + sigma * normal, status)


@wp.kernel
def gaussian_read_noise_pairs(
    signal: wp.array(dtype=wp.float32),
    sigma: wp.float64,
    seed: wp.uint64,
    observation: wp.uint32,
    pixel_offset: wp.uint32,
    output: wp.array(dtype=wp.float32),
    status: wp.array(dtype=wp.int32),
):
    """Reuse both Box-Muller members at absolute even-pixel pair anchors."""
    pair = wp.tid()
    anchor = (wp.uint64(pixel_offset) & wp.uint64(0xFFFFFFFE)) + wp.uint64(2 * pair)
    identity = (wp.uint64(observation) << wp.uint64(32)) | anchor
    uniforms = uniform_pair(seed, identity, wp.uint32(0), wp.uint32(0xFFFFFFFF))
    radius = wp.sqrt(-wp.float64(2.0) * wp.log(uniforms[0]))
    angle = wp.float64(6.283185307179586476925286766559) * uniforms[1]
    start = wp.uint64(pixel_offset)
    end = start + wp.uint64(signal.shape[0])
    if anchor >= start and anchor < end:
        pixel = int(anchor - start)
        normal = radius * wp.cos(angle)
        output[pixel] = checked_store(wp.float64(signal[pixel]) + sigma * normal, status)
    if anchor + wp.uint64(1) >= start and anchor + wp.uint64(1) < end:
        pixel = int(anchor + wp.uint64(1) - start)
        normal = radius * wp.sin(angle)
        output[pixel] = checked_store(wp.float64(signal[pixel]) + sigma * normal, status)
