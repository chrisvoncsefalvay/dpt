"""Direct supplied-PSF Fourier transfer and compound-Poisson noise spectrum."""

import warp as wp

from dpt.kernels._native_compensated import _SCALED_KAHAN_UPDATE_CPP

wp.set_module_options({"fast_math": False, "fuse_fp": True, "enable_backward": False})


@wp.func_native(
    """
const auto decompose = [](double value, int& exponent) {
    int correction = 0;
    if (value < 2.2250738585072014e-308) {
        value *= 18014398509481984.0;
        correction = 54;
    }
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    exponent = static_cast<int>((bits >> 52) & 0x7ff) - 1022 - correction;
    bits = (bits & 0x000fffffffffffffull) | 0x3fe0000000000000ull;
    memcpy(&value, &bits, sizeof(value));
    return value;
};
const double factors[3][7] = {
    {variance, dc, dc, wp::abs(real), wp::abs(real), width, height},
    {variance, dc, dc, wp::abs(imaginary), wp::abs(imaginary), width, height},
    {read, width, height, 1.0, 1.0, 1.0, 1.0}
};
double mantissas[3] = {0.0, 0.0, 0.0};
int exponents[3] = {0, 0, 0};
int common = -16384;
for (int i = 0; i < 3; ++i) {
    double mantissa = 1.0;
    int exponent = 0;
    for (int j = 0; j < 7; ++j) {
        if (factors[i][j] == 0.0) { mantissa = 0.0; break; }
        int e;
        mantissa *= decompose(factors[i][j], e);
        exponent += e;
    }
    if (mantissa > 0.0) {
        int e;
        mantissas[i] = decompose(mantissa, e);
        exponents[i] = exponent + e;
        common = wp::max(common, exponents[i]);
    }
}
if (common == -16384) return 0.0;
double sum = 0.0, correction = 0.0;
for (int i = 0; i < 3; ++i) {
    if (mantissas[i] == 0.0) continue;
"""
    + _SCALED_KAHAN_UPDATE_CPP
    + """}
return ldexp(sum, common);
"""
)
def noise_sum(
    variance: wp.float64,
    dc: wp.float64,
    real: wp.float64,
    imaginary: wp.float64,
    read: wp.float64,
    width: wp.float64,
    height: wp.float64,
) -> wp.float64: ...


@wp.func
def hypot2(a: wp.float64, b: wp.float64):
    scale = wp.max(wp.abs(a), wp.abs(b))
    if scale == wp.float64(0.0):
        return wp.float64(0.0)
    return scale * wp.sqrt((a / scale) * (a / scale) + (b / scale) * (b / scale))


@wp.func
def compensated_add(state: wp.vec2d, term: wp.float64):
    total = state[0] + term
    correction = state[1]
    if wp.abs(state[0]) >= wp.abs(term):
        correction = correction + ((state[0] - total) + term)
    else:
        correction = correction + ((term - total) + state[0])
    return wp.vec2d(total, correction)


@wp.func
def harmonic(frequency: wp.float64, offset: int):
    if frequency == wp.float64(0.0):
        return wp.vec2d(wp.float64(1.0), wp.float64(0.0))
    if wp.abs(frequency) == wp.float64(0.5):
        sign = wp.float64(1.0)
        if offset % 2 != 0:
            sign = wp.float64(-1.0)
        return wp.vec2d(sign, wp.float64(0.0))
    phase = wp.float64(6.283185307179586476925286766559) * frequency * wp.float64(offset)
    return wp.vec2d(wp.cos(phase), wp.sin(phase))


@wp.kernel
def transfer(
    frequencies: wp.array2d(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    rows: int,
    columns: int,
    spacing_column: wp.float64,
    spacing_row: wp.float64,
    dc: wp.float64,
    population: wp.float64,
    mean: wp.float64,
    variance: wp.float64,
    read_variance: wp.float64,
    mtf: wp.array(dtype=wp.float64),
    nps: wp.array(dtype=wp.float64),
    dqe: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    p = wp.tid()
    fx = frequencies[p, 0]
    fy = frequencies[p, 1]
    if (
        not wp.isfinite(fx)
        or not wp.isfinite(fy)
        or wp.abs(fx) * spacing_column > wp.float64(0.5)
        or wp.abs(fy) * spacing_row > wp.float64(0.5)
    ):
        mtf[p] = wp.float64(0.0)
        nps[p] = wp.float64(0.0)
        dqe[p] = wp.float64(0.0)
        wp.atomic_or(status, 0, 1)
        return
    real_state = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    imaginary_state = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    for row in range(rows):
        for col in range(columns):
            x = harmonic(fx * spacing_column, col - columns // 2)
            y = harmonic(fy * spacing_row, row - rows // 2)
            weight = weights[row * columns + col] / dc
            real_state = compensated_add(real_state, weight * (x[0] * y[0] - x[1] * y[1]))
            imaginary_state = compensated_add(imaginary_state, weight * (x[1] * y[0] + x[0] * y[1]))
    real = real_state[0] + real_state[1]
    imaginary = imaginary_state[0] + imaginary_state[1]
    mtf_value = hypot2(real, imaginary)
    quantum = noise_sum(
        variance, dc, real, imaginary, wp.float64(0.0), wp.float64(1.0), wp.float64(1.0)
    )
    log_quantum = wp.float64(-1.7976931348623157e308)
    if variance > wp.float64(0.0) and mtf_value > wp.float64(0.0):
        log_quantum = (
            wp.log(variance) + wp.float64(2.0) * wp.log(dc) + wp.float64(2.0) * wp.log(mtf_value)
        )
        if quantum == wp.float64(0.0):
            quantum = wp.exp(log_quantum)
    # Sum all positive contributions at a shared binary exponent. This retains
    # combined subnormal tails and avoids overflowing their unscaled sum.
    nps_value = noise_sum(variance, dc, real, imaginary, read_variance, spacing_column, spacing_row)
    dqe_value = wp.float64(0.0)
    if variance > wp.float64(0.0) and mtf_value > wp.float64(0.0):
        scaled_mean = mean / wp.sqrt(population)
        efficiency = scaled_mean / wp.sqrt(variance)
        if mean > wp.float64(0.0) and scaled_mean < wp.float64(2.2250738585072014e-308):
            efficiency = wp.exp(
                wp.log(mean) - wp.float64(0.5) * (wp.log(population) + wp.log(variance))
            )
        fraction = wp.float64(1.0)
        if read_variance > wp.float64(0.0):
            if quantum >= wp.float64(2.2250738585072014e-308):
                if quantum >= read_variance:
                    fraction = wp.float64(1.0) / (wp.float64(1.0) + read_variance / quantum)
                else:
                    relative = quantum / read_variance
                    fraction = relative / (wp.float64(1.0) + relative)
            else:
                relative = log_quantum - wp.log(read_variance)
                fraction = wp.exp(relative) / (wp.float64(1.0) + wp.exp(relative))
        dqe_value = efficiency * efficiency * fraction
    mtf[p] = mtf_value
    nps[p] = nps_value
    dqe[p] = dqe_value
    if not wp.isfinite(mtf_value) or not wp.isfinite(nps_value) or not wp.isfinite(dqe_value):
        wp.atomic_or(status, 0, 2)
