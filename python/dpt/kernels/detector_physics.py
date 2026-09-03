"""Exact pixel acceptance and explicit first-interaction response derivatives."""

from functools import cache

import warp as wp

from dpt.kernels._native_compensated import _SCALED_KAHAN_UPDATE_CPP

STRICT = {"fast_math": False, "fuse_fp": True, "enable_backward": False}
wp.set_module_options(STRICT)


@wp.func_native("""
#if defined(__CUDACC__)
return -expm1(-value);
#else
return -__builtin_expm1(-value);
#endif
""")
def interaction_probability(value: wp.float64) -> wp.float64: ...


@wp.func
def hypot2(a: wp.float64, b: wp.float64):
    scale = wp.max(wp.abs(a), wp.abs(b))
    if scale == wp.float64(0.0):
        return wp.float64(0.0)
    return scale * wp.sqrt((a / scale) * (a / scale) + (b / scale) * (b / scale))


@wp.func_native(
    """
// Partition at the source-projection axes. All rays in each piece have
// matching component signs, so the normalized triangle denominator is >= 1.
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
double xlow[2] = {x - du * 0.5, 0.0};
double xhigh[2] = {x + du * 0.5, 0.0};
double ylow[2] = {y - dv * 0.5, 0.0};
double yhigh[2] = {y + dv * 0.5, 0.0};
double widths[2] = {du, 0.0};
double heights[2] = {dv, 0.0};
int width_corrections[2] = {0, 0};
int height_corrections[2] = {0, 0};
int nx = 1, ny = 1;
if (xlow[0] < 0.0 && xhigh[0] > 0.0) {
    nx = 2;
    xhigh[1] = xhigh[0];
    xhigh[0] = 0.0;
    widths[0] = du * 0.5 - x;
    widths[1] = du * 0.5 + x;
    if (du < 2.2250738585072014e-308) {
        // Preserve half-subnormal partition widths until determinant scaling.
        const double expanded = du * 18014398509481984.0;
        const double centre = x * 18014398509481984.0;
        widths[0] = expanded * 0.5 - centre;
        widths[1] = expanded * 0.5 + centre;
        width_corrections[0] = width_corrections[1] = -54;
    }
}
if (ylow[0] < 0.0 && yhigh[0] > 0.0) {
    ny = 2;
    yhigh[1] = yhigh[0];
    yhigh[0] = 0.0;
    heights[0] = dv * 0.5 - y;
    heights[1] = dv * 0.5 + y;
    if (dv < 2.2250738585072014e-308) {
        const double expanded = dv * 18014398509481984.0;
        const double centre = y * 18014398509481984.0;
        heights[0] = expanded * 0.5 - centre;
        heights[1] = expanded * 0.5 + centre;
        height_corrections[0] = height_corrections[1] = -54;
    }
}
double mantissas[8];
int exponents[8];
int count = 0;
for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
        double corners[4][3] = {
            {xlow[ix], ylow[iy], distance},
            {xhigh[ix], ylow[iy], distance},
            {xhigh[ix], yhigh[iy], distance},
            {xlow[ix], yhigh[iy], distance}
        };
        double scales[4], norms[4];
        double rays[4][3];
        for (int i = 0; i < 4; ++i) {
            scales[i] = wp::max(distance,
                wp::max(wp::abs(corners[i][0]), wp::abs(corners[i][1])));
            if (!wp::isfinite(scales[i])) return NAN;
            double sum = 0.0;
            for (int j = 0; j < 3; ++j) {
                rays[i][j] = corners[i][j] / scales[i];
                sum += rays[i][j] * rays[i][j];
            }
            norms[i] = sqrt(sum);
            for (int j = 0; j < 3; ++j) rays[i][j] /= norms[i];
        }
        for (int t = 0; t < 2; ++t) {
            const int a = 0, b = t == 0 ? 1 : 2, c = t == 0 ? 2 : 3;
            double denominator = 1.0;
            for (int j = 0; j < 3; ++j) {
                denominator += rays[a][j] * rays[b][j]
                    + rays[b][j] * rays[c][j] + rays[c][j] * rays[a][j];
            }
            int exponent = 0, e;
            double mantissa = decompose(widths[ix], e);
            exponent += e + width_corrections[ix];
            mantissa *= decompose(heights[iy], e);
            exponent += e + height_corrections[iy];
            mantissa *= decompose(distance, e);
            exponent += e;
            const int vertices[3] = {a, b, c};
            for (int v = 0; v < 3; ++v) {
                const int i = vertices[v];
                mantissa /= decompose(scales[i], e);
                exponent -= e;
                mantissa /= norms[i];
            }
            // At this scale the omitted atan cubic is < binary64 rounding.
            // Retain the exponent: positive triangle tails may individually
            // round to zero although their complete rectangle is representable.
            if (exponent < -512) {
                mantissas[count] = 2.0 * mantissa / denominator;
                exponents[count] = exponent;
            } else {
                const double angle = 2.0 * atan2(ldexp(mantissa, exponent), denominator);
                mantissas[count] = decompose(angle, exponents[count]);
            }
            ++count;
        }
    }
}
int common = exponents[0];
for (int i = 1; i < count; ++i) common = wp::max(common, exponents[i]);
double sum = 0.0, correction = 0.0;
for (int i = 0; i < count; ++i) {
"""
    + _SCALED_KAHAN_UPDATE_CPP
    + """}
return ldexp(sum, common);
"""
)
def rectangle_angle(
    x: wp.float64, y: wp.float64, distance: wp.float64, du: wp.float64, dv: wp.float64
) -> wp.float64: ...


@wp.kernel
def pixel_acceptance(
    metadata: wp.array2d(dtype=wp.float64),
    modes: wp.array(dtype=wp.int32),
    rows: int,
    columns: int,
    solid: wp.array(dtype=wp.float64),
    incidence: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    view, row, column = wp.tid()
    p = (view * rows + row) * columns + column
    x = metadata[view, 0] + wp.float64(column) * metadata[view, 3]
    y = metadata[view, 1] + wp.float64(row) * metadata[view, 4]
    distance = metadata[view, 2]
    du = metadata[view, 3]
    dv = metadata[view, 4]
    result = wp.float64(0.0)
    cosine = wp.float64(0.0)
    if modes[view] == 1:
        radius = metadata[view, 5]
        scale = wp.max(radius, wp.abs(y) + dv / wp.float64(2.0))
        r = radius / scale
        low = y / scale - dv / scale / wp.float64(2.0)
        high = y / scale + dv / scale / wp.float64(2.0)
        hl = hypot2(r, low)
        hh = hypot2(r, high)
        delta = high / hh - low / hl
        if low * high > wp.float64(0.0):
            # Rationalize the difference in the far axial tail. Use the
            # declared pixel width even if its endpoints round together.
            delta = r * r * (dv / scale) * (high + low) / (hh * hl * (high * hl + low * hh))
        result = (du / radius) * delta
        cosine = radius / hypot2(radius, y)
    else:
        scale = wp.max(
            distance, wp.max(wp.abs(x) + du / wp.float64(2.0), wp.abs(y) + dv / wp.float64(2.0))
        )
        xs = x / scale
        ys = y / scale
        ds = distance / scale
        result = rectangle_angle(x, y, distance, du, dv)
        cosine = ds / wp.sqrt(xs * xs + ys * ys + ds * ds)
    solid[p] = result
    incidence[p] = cosine
    if (
        not wp.isfinite(result)
        or result <= wp.float64(0.0)
        or not wp.isfinite(cosine)
        or cosine <= wp.float64(0.0)
        or cosine > wp.float64(1.0)
    ):
        wp.atomic_or(status, 0, 2)


@wp.func
def slab_depth(mu: wp.float64, thickness: wp.float64, cosine: wp.float64):
    logarithm = wp.float64(-1.7976931348623157e308)
    if mu == wp.float64(0.0) or thickness == wp.float64(0.0):
        return wp.vec2d(wp.float64(0.0), logarithm)
    product = mu * thickness
    normal = wp.float64(2.2250738585072014e-308)
    if wp.isfinite(product) and product >= normal:
        direct = product / cosine
        return wp.vec2d(direct, wp.log(direct))
    ratio = mu / cosine
    if wp.isfinite(ratio) and ratio >= normal:
        direct = ratio * thickness
        if wp.isfinite(direct) and direct >= normal:
            return wp.vec2d(direct, wp.log(direct))
    logarithm = wp.log(mu) + wp.log(thickness) - wp.log(cosine)
    return wp.vec2d(wp.exp(logarithm), logarithm)


@cache
def response_forward(wide: bool):
    dtype = wp.float64 if wide else wp.float32

    @wp.kernel(module="unique", module_options=STRICT)
    def forward(
        thickness: wp.array(dtype=wp.float64),
        cosine: wp.array(dtype=wp.float64),
        coefficients: wp.array(dtype=wp.float64),
        scores: wp.array(dtype=wp.float64),
        output: wp.array2d(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        k, p = wp.tid()
        t = thickness[p]
        c = cosine[p]
        if (
            not wp.isfinite(t)
            or t < wp.float64(0.0)
            or not wp.isfinite(c)
            or c <= wp.float64(0.0)
            or c > wp.float64(1.0)
        ):
            output[k, p] = dtype(0.0)
            wp.atomic_or(status, 0, 1)
            return
        mu = coefficients[k]
        score = scores[k]
        value = wp.float64(0.0)
        if mu > wp.float64(0.0) and t > wp.float64(0.0) and score > wp.float64(0.0):
            depth = slab_depth(mu, t, c)
            if depth[0] < wp.float64(2.2250738585072014e-308):
                value = wp.exp(wp.log(score) + depth[1])
            else:
                value = score * interaction_probability(depth[0])
        converted = dtype(value)
        output[k, p] = converted
        if not wp.isfinite(converted) or (value > wp.float64(0.0) and converted == dtype(0.0)):
            wp.atomic_or(status, 0, 2)

    return forward


@cache
def response_reverse(wide: bool):
    dtype = wp.float64 if wide else wp.float32

    @wp.kernel(module="unique", module_options=STRICT)
    def reverse(
        thickness: wp.array(dtype=wp.float64),
        cosine: wp.array(dtype=wp.float64),
        coefficients: wp.array(dtype=wp.float64),
        scores: wp.array(dtype=wp.float64),
        seed: wp.array2d(dtype=dtype),
        energies: int,
        out_thickness: wp.array(dtype=wp.float64),
        out_cosine: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        t = thickness[p]
        c = cosine[p]
        if (
            not wp.isfinite(t)
            or t < wp.float64(0.0)
            or not wp.isfinite(c)
            or c <= wp.float64(0.0)
            or c > wp.float64(1.0)
        ):
            out_thickness[p] = wp.float64(0.0)
            out_cosine[p] = wp.float64(0.0)
            wp.atomic_or(status, 0, 1)
            return
        gt = wp.float64(0.0)
        gc = wp.float64(0.0)
        for k in range(energies):
            mu = coefficients[k]
            score = scores[k]
            adj = wp.float64(seed[k, p])
            if not wp.isfinite(adj):
                wp.atomic_or(status, 0, 1)
                continue
            if mu > wp.float64(0.0) and score > wp.float64(0.0) and adj != wp.float64(0.0):
                depth = slab_depth(mu, t, c)
                exponent = wp.log(wp.abs(adj)) + wp.log(score) + wp.log(mu) - wp.log(c) - depth[0]
                sign = wp.float64(1.0)
                if adj < wp.float64(0.0):
                    sign = wp.float64(-1.0)
                dt = sign * wp.exp(exponent)
                weighted = mu * score
                numerator = weighted * adj
                direct = numerator / c
                if (
                    wp.isfinite(direct)
                    and direct != wp.float64(0.0)
                    and wp.abs(weighted) >= wp.float64(2.2250738585072014e-308)
                    and wp.abs(numerator) >= wp.float64(2.2250738585072014e-308)
                    and depth[0] < wp.float64(600.0)
                ):
                    dt = direct * wp.exp(-depth[0])
                gt = gt + dt
                if t > wp.float64(0.0):
                    gc = gc - sign * wp.exp(exponent + wp.log(t) - wp.log(c))
        out_thickness[p] = gt
        out_cosine[p] = gc
        if not wp.isfinite(gt) or not wp.isfinite(gc):
            wp.atomic_or(status, 0, 2)

    return reverse
