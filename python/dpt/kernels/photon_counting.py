"""Finite charge clouds and a nonparalysable rectangular pulse integrator."""

import warp as wp

from dpt.kernels.random import random4

wp.set_module_options({"fast_math": False, "fuse_fp": True, "enable_backward": False})


@wp.func_native(
    """
double factors[3] = {a, b, c};
double mantissa = 1.0;
int exponent = 0;
for (int i = 0; i < 3; ++i) {
    double value = factors[i];
    if (value == 0.0) return 0.0;
    if (value < 0.0) { mantissa = -mantissa; value = -value; }
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
"""
)
def product3(a: wp.float64, b: wp.float64, c: wp.float64) -> wp.float64: ...


@wp.func_native(
    """
double factors[6] = {a, b, c, d, e, f};
double mantissa = 1.0;
int exponent = 0;
for (int i = 0; i < 6; ++i) {
    double value = factors[i];
    if (value == 0.0) return 0.0;
    if (value < 0.0) { mantissa = -mantissa; value = -value; }
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
if (log_probability == -INFINITY) return 0.0;
double probability = exp(log_probability);
if (probability >= 2.2250738585072014e-308) {
    uint64_t bits;
    memcpy(&bits, &probability, sizeof(bits));
    exponent += static_cast<int>((bits >> 52) & 0x7ff) - 1022;
    bits = (bits & 0x000fffffffffffffull) | 0x3fe0000000000000ull;
    memcpy(&probability, &bits, sizeof(probability));
    mantissa *= probability;
} else {
    // Six finite factors cannot rescue a log probability below this bound.
    if (log_probability < -8192.0) return 0.0;
    int probability_exponent = static_cast<int>(floor(log_probability / 0.6931471805599453));
    double remainder = (log_probability - probability_exponent * 0.6931471805599453)
                       - probability_exponent * 2.3190468138462996e-17;
    mantissa *= exp(remainder);
    exponent += probability_exponent;
}
return ldexp(mantissa, exponent);
"""
)
def probability_product6(
    log_probability: wp.float64,
    a: wp.float64,
    b: wp.float64,
    c: wp.float64,
    d: wp.float64,
    e: wp.float64,
    f: wp.float64,
) -> wp.float64: ...


@wp.func
def probability_cotangent_product4(
    log_probability: wp.float64,
    a: wp.float64,
    b: wp.float64,
    c: wp.float64,
    d: wp.float64,
    positive: wp.float64,
    negative: wp.float64,
):
    difference = positive - negative
    if wp.isfinite(difference):
        return probability_product6(log_probability, a, b, c, d, difference, wp.float64(1.0))
    scale = wp.max(wp.abs(positive), wp.abs(negative))
    return probability_product6(
        log_probability, a, b, c, d, scale, positive / scale - negative / scale
    )


@wp.func
def add(state: wp.vec2d, term: wp.float64):
    total = state[0] + term
    correction = state[1]
    if wp.abs(state[0]) >= wp.abs(term):
        correction = correction + ((state[0] - total) + term)
    else:
        correction = correction + ((term - total) + state[0])
    return wp.vec2d(total, correction)


@wp.func
def store(value: wp.float64, status: wp.array(dtype=wp.int32)):
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, 2)
    return value


@wp.func
def category(energy: wp.float64, edges: wp.array(dtype=wp.float64), bins: int):
    if energy <= wp.float64(0.0):
        return -3
    if energy < edges[0]:
        return -1
    for b in range(bins):
        if energy < edges[b + 1]:
            return b
    return -2


@wp.func
def ledger_row(bin_index: int):
    if bin_index == -1:
        return 1
    if bin_index == -2:
        return 2
    if bin_index == -3:
        return 3
    return 0


@wp.kernel
def charge_events(
    pixels: wp.array(dtype=wp.int32),
    energy_bins: wp.array(dtype=wp.int32),
    times: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    probabilities: wp.array(dtype=wp.float64),
    fractions: wp.array(dtype=wp.float64),
    dr: wp.array(dtype=wp.int32),
    dc: wp.array(dtype=wp.int32),
    edges: wp.array(dtype=wp.float64),
    energy_count: int,
    patterns: int,
    fragments: int,
    bins: int,
    rows: int,
    columns: int,
    seed: wp.uint64,
    identity: wp.uint64,
    out_pixels: wp.array2d(dtype=wp.int32),
    out_times: wp.array2d(dtype=wp.float64),
    out_energy: wp.array2d(dtype=wp.float64),
    out_bins: wp.array2d(dtype=wp.int32),
    out_pattern: wp.array(dtype=wp.int32),
    accounting: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    n = wp.tid()
    p, e, time = int(pixels[n]), int(energy_bins[n]), wp.float64(times[n])
    for k in range(4):
        accounting[n, k] = wp.float64(0.0)
    for f in range(fragments):
        out_pixels[n, f] = -1
        out_bins[n, f] = -3
        out_energy[n, f] = wp.float64(0.0)
        out_times[n, f] = wp.float64(0.0)
    out_pattern[n] = -1
    if (
        p < 0
        or p >= rows * columns
        or e < 0
        or e >= energy_count
        or not wp.isfinite(time)
        or time < wp.float64(0.0)
    ):
        wp.atomic_or(status, 0, 1)
        return
    u = random4(seed, identity + wp.uint64(n), wp.uint32(0xFFFFFFFF), wp.uint32(1))[0]
    cumulative = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    chosen = int(patterns - 1)
    for m in range(patterns):
        cumulative = add(cumulative, probabilities[e * patterns + m])
        if u < cumulative[0] + cumulative[1]:
            chosen = m
            break
    out_pattern[n] = chosen
    for f in range(fragments):
        i = chosen * fragments + f
        energy = energies[e] * fractions[i]
        rr, cc = p // columns + dr[i], p % columns + dc[i]
        b = category(energy, edges, bins)
        destination = int(-1)  # noqa: UP018, RUF046 - Warp requires mutable loop variables.
        if rr >= 0 and rr < rows and cc >= 0 and cc < columns and energy > wp.float64(0.0):
            destination = rr * columns + cc
        else:
            b = -3
        out_pixels[n, f] = destination
        out_times[n, f] = time
        out_energy[n, f] = energy
        out_bins[n, f] = b
        k = ledger_row(b)
        accounting[n, k] = accounting[n, k] + energy


@wp.kernel
def charge_mean(
    incident: wp.array2d(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    probabilities: wp.array(dtype=wp.float64),
    fractions: wp.array(dtype=wp.float64),
    dr: wp.array(dtype=wp.int32),
    dc: wp.array(dtype=wp.int32),
    edges: wp.array(dtype=wp.float64),
    energy_count: int,
    patterns: int,
    fragments: int,
    bins: int,
    rows: int,
    columns: int,
    counts: wp.array2d(dtype=wp.float64),
    accounting: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    p = wp.tid()
    for b in range(bins):
        state = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        for e in range(energy_count):
            for m in range(patterns):
                for f in range(fragments):
                    i = m * fragments + f
                    energy = energies[e] * fractions[i]
                    rr, cc = p // columns - dr[i], p % columns - dc[i]
                    if (
                        rr >= 0
                        and rr < rows
                        and cc >= 0
                        and cc < columns
                        and category(energy, edges, bins) == b
                    ):
                        value = incident[e, rr * columns + cc]
                        if wp.isfinite(value) and value >= wp.float64(0.0):
                            state = add(state, value * probabilities[e * patterns + m])
                        else:
                            wp.atomic_or(status, 0, 1)
        counts[b, p] = store(state[0] + state[1], status)
    for k in range(4):
        state = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        for e in range(energy_count):
            value = incident[e, p]
            if not wp.isfinite(value) or value < wp.float64(0.0):
                wp.atomic_or(status, 0, 1)
            else:
                for m in range(patterns):
                    for f in range(fragments):
                        i = m * fragments + f
                        energy = energies[e] * fractions[i]
                        rr, cc = p // columns + dr[i], p % columns + dc[i]
                        b = category(energy, edges, bins)
                        if rr < 0 or rr >= rows or cc < 0 or cc >= columns:
                            b = -3
                        if ledger_row(b) == k:
                            state = add(
                                state,
                                product3(value, probabilities[e * patterns + m], energy),
                            )
        accounting[k, p] = store(state[0] + state[1], status)


@wp.kernel
def charge_vjp(
    seed: wp.array2d(dtype=wp.float64),
    energy_seed: wp.array2d(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    probabilities: wp.array(dtype=wp.float64),
    fractions: wp.array(dtype=wp.float64),
    dr: wp.array(dtype=wp.int32),
    dc: wp.array(dtype=wp.int32),
    edges: wp.array(dtype=wp.float64),
    energy_count: int,
    patterns: int,
    fragments: int,
    bins: int,
    rows: int,
    columns: int,
    output: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    e, p = wp.tid()
    for b in range(bins):
        if not wp.isfinite(seed[b, p]):
            wp.atomic_or(status, 0, 1)
    for k in range(4):
        if not wp.isfinite(energy_seed[k, p]):
            wp.atomic_or(status, 0, 1)
    state = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    for m in range(patterns):
        for f in range(fragments):
            i = m * fragments + f
            energy = energies[e] * fractions[i]
            rr, cc = p // columns + dr[i], p % columns + dc[i]
            b = category(energy, edges, bins)
            term = wp.float64(0.0)
            if rr < 0 or rr >= rows or cc < 0 or cc >= columns:
                b = -3
            elif b >= 0:
                term = seed[b, rr * columns + cc]
            probability = probabilities[e * patterns + m]
            state = add(state, term * probability)
            state = add(state, product3(energy, energy_seed[ledger_row(b), p], probability))
    output[e, p] = store(state[0] + state[1], status)


@wp.func
def log1p(x: wp.float64):
    total = wp.float64(1.0) + x
    if total == wp.float64(1.0):
        return x
    return wp.log(total) * (x / (total - wp.float64(1.0)))


@wp.func
def logadd(a: wp.float64, b: wp.float64):
    if a == wp.float64(-wp.inf):
        return b
    if b == wp.float64(-wp.inf):
        return a
    high = wp.max(a, b)
    return high + log1p(wp.exp(wp.min(a, b) - high))


@wp.func
def build_compound(
    lam: wp.float64,
    marks: wp.array(dtype=wp.int32),
    probabilities: wp.array(dtype=wp.float64),
    mark_count: int,
    top: int,
    p: int,
    compound: wp.array2d(dtype=wp.float64),
):
    # Store log probabilities so an underflowed c[0] cannot erase rescued mass
    # at a later energy step near the Poisson mode.
    compound[0, p] = -lam
    for n in range(1, top):
        value = wp.float64(-wp.inf)
        if lam > wp.float64(0.0):
            for j in range(mark_count):
                step, probability = int(marks[j]), wp.float64(probabilities[j])
                if step <= n and probability > wp.float64(0.0):
                    value = logadd(
                        value,
                        wp.log(wp.float64(step)) + wp.log(probability) + compound[n - step, p],
                    )
            value = value + wp.log(lam) - wp.log(wp.float64(n))
        compound[n, p] = value


@wp.func
def pulse_law(
    n: int,
    p: int,
    marks: wp.array(dtype=wp.int32),
    probabilities: wp.array(dtype=wp.float64),
    mark_count: int,
    compound: wp.array2d(dtype=wp.float64),
):
    first = wp.float64(-wp.inf)
    second = wp.float64(-wp.inf)
    for j in range(mark_count):
        step, probability = int(marks[j]), wp.float64(probabilities[j])
        if step <= n and probability > wp.float64(0.0):
            lp = wp.log(probability)
            first = logadd(first, lp + compound[n - step, p])
            for k in range(mark_count):
                other, other_probability = int(marks[k]), wp.float64(probabilities[k])
                if other <= n - step and other_probability > wp.float64(0.0):
                    second = logadd(
                        second,
                        lp + wp.log(other_probability) + compound[n - step - other, p],
                    )
    return wp.vec2d(first, second)


@wp.func
def step_bin(n: int, edges: wp.array(dtype=wp.int32), bins: int):
    if n < edges[0]:
        return -1
    for b in range(bins):
        if n < edges[b + 1]:
            return b
    return -2


@wp.func
def valid_parameters(
    r: wp.float64, gate: wp.float64, blind: wp.float64, status: wp.array(dtype=wp.int32)
):
    if (
        not wp.isfinite(r)
        or not wp.isfinite(gate)
        or not wp.isfinite(blind)
        or r < wp.float64(0.0)
        or gate < wp.float64(0.0)
        or blind < wp.float64(0.0)
    ):
        wp.atomic_or(status, 0, 1)
        return False
    if not wp.isfinite(r * gate) or not wp.isfinite(r * gate + r * blind):
        wp.atomic_or(status, 0, 2)
        return False
    return True


@wp.kernel
def pileup_mean(
    rates: wp.array(dtype=wp.float64),
    gates: wp.array(dtype=wp.float64),
    blind_times: wp.array(dtype=wp.float64),
    marks: wp.array(dtype=wp.int32),
    probabilities: wp.array(dtype=wp.float64),
    edges: wp.array(dtype=wp.int32),
    mark_count: int,
    bins: int,
    quantum: wp.float64,
    mean_mark: wp.float64,
    compound: wp.array2d(dtype=wp.float64),
    counts: wp.array2d(dtype=wp.float64),
    accounting: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    p = wp.tid()
    r, gate, blind = (
        wp.float64(rates[p]),
        wp.float64(gates[p]),
        wp.float64(blind_times[p]),
    )
    for b in range(bins):
        counts[b, p] = wp.float64(0.0)
    for k in range(6):
        accounting[k, p] = wp.float64(0.0)
    if not valid_parameters(r, gate, blind, status):
        return
    lam = r * gate
    h = r / (wp.float64(1.0) + r * gate + r * blind)
    build_compound(lam, marks, probabilities, mark_count, edges[bins], p, compound)
    finite_probability = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    finite_energy_rate = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    for b in range(-1, bins):
        probability = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        count_rate = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        energy_rate = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        low, high = int(1), int(edges[0])  # noqa: UP018, RUF046
        if b >= 0:
            low, high = wp.max(1, edges[b]), int(edges[b + 1])
        for n in range(low, high):
            log_g = pulse_law(n, p, marks, probabilities, mark_count, compound)[0]
            probability = add(probability, wp.exp(log_g))
            count_rate = add(
                count_rate,
                probability_product6(
                    log_g,
                    h,
                    wp.float64(1.0),
                    wp.float64(1.0),
                    wp.float64(1.0),
                    wp.float64(1.0),
                    wp.float64(1.0),
                ),
            )
            energy_rate = add(
                energy_rate,
                probability_product6(
                    log_g,
                    h,
                    wp.float64(n) * quantum,
                    wp.float64(1.0),
                    wp.float64(1.0),
                    wp.float64(1.0),
                    wp.float64(1.0),
                ),
            )
        prob = probability[0] + probability[1]
        count, en = count_rate[0] + count_rate[1], energy_rate[0] + energy_rate[1]
        finite_probability = add(finite_probability, prob)
        finite_energy_rate = add(finite_energy_rate, en)
        if b < 0:
            accounting[0, p] = store(count, status)
            accounting[2, p] = store(en, status)
        else:
            counts[b, p] = store(count, status)
            accounting[4, p] = accounting[4, p] + en
    tail = wp.float64(1.0) - (finite_probability[0] + finite_probability[1])
    if tail < wp.float64(-2.0e-12):
        wp.atomic_or(status, 0, 4)
    # The complement can be a few ulps negative when it is exactly zero.
    tail = wp.max(wp.float64(0.0), tail)
    accounting[1, p] = store(h * tail, status)
    pulse_energy = product3(h, wp.float64(1.0) + lam, mean_mark)
    overflow_energy = pulse_energy - (finite_energy_rate[0] + finite_energy_rate[1])
    if overflow_energy < -wp.float64(2.0e-12) * pulse_energy:
        wp.atomic_or(status, 0, 4)
    accounting[3, p] = store(wp.max(wp.float64(0.0), overflow_energy), status)
    accounting[4, p] = store(accounting[4, p], status)
    accounting[5, p] = store(product3(h, r * blind, mean_mark), status)


@wp.kernel
def pileup_vjp(
    rates: wp.array(dtype=wp.float64),
    gates: wp.array(dtype=wp.float64),
    blind_times: wp.array(dtype=wp.float64),
    seed: wp.array2d(dtype=wp.float64),
    energy_seed: wp.array2d(dtype=wp.float64),
    marks: wp.array(dtype=wp.int32),
    probabilities: wp.array(dtype=wp.float64),
    edges: wp.array(dtype=wp.int32),
    mark_count: int,
    bins: int,
    quantum: wp.float64,
    mean_mark: wp.float64,
    compound: wp.array2d(dtype=wp.float64),
    rate_gradient: wp.array(dtype=wp.float64),
    gate_gradient: wp.array(dtype=wp.float64),
    blind_gradient: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    p = wp.tid()
    r, gate, blind = (
        wp.float64(rates[p]),
        wp.float64(gates[p]),
        wp.float64(blind_times[p]),
    )
    rate_gradient[p] = wp.float64(0.0)
    gate_gradient[p] = wp.float64(0.0)
    blind_gradient[p] = wp.float64(0.0)
    if not valid_parameters(r, gate, blind, status):
        return
    for b in range(bins):
        if not wp.isfinite(seed[b, p]):
            wp.atomic_or(status, 0, 1)
            return
    for k in range(6):
        if not wp.isfinite(energy_seed[k, p]):
            wp.atomic_or(status, 0, 1)
            return
    denominator = wp.float64(1.0) + r * gate + r * blind
    inverse = wp.float64(1.0) / denominator
    h = r * inverse
    lam, blind_load = r * gate, r * blind
    build_compound(lam, marks, probabilities, mark_count, edges[bins], p, compound)
    # Start with the complete overflow and blind moments. Differentiate the
    # positive renewal factors before weighting them, so zero cotangents and
    # weak probabilities never encounter an overflowing h*h intermediate.
    dr, dg, dt = (
        wp.vec2d(wp.float64(0.0)),
        wp.vec2d(wp.float64(0.0)),
        wp.vec2d(wp.float64(0.0)),
    )
    dr = add(dr, product3(inverse, inverse, energy_seed[1, p]))
    dg = add(dg, -product3(h, h, energy_seed[1, p]))
    dt = add(dt, -product3(h, h, energy_seed[1, p]))
    dr = add(
        dr,
        probability_product6(
            wp.float64(0.0),
            energy_seed[3, p],
            mean_mark,
            inverse,
            inverse * (wp.float64(1.0) + lam),
            wp.float64(1.0),
            wp.float64(1.0),
        ),
    )
    dr = add(
        dr,
        probability_product6(
            wp.float64(0.0),
            energy_seed[3, p],
            mean_mark,
            inverse,
            lam,
            wp.float64(1.0),
            wp.float64(1.0),
        ),
    )
    dr = add(
        dr,
        probability_product6(
            wp.float64(0.0),
            energy_seed[5, p],
            mean_mark,
            blind_load,
            inverse * (wp.float64(1.0) + inverse),
            wp.float64(1.0),
            wp.float64(1.0),
        ),
    )
    dg = add(
        dg,
        probability_cotangent_product4(
            wp.float64(0.0),
            mean_mark,
            h,
            h,
            blind_load,
            energy_seed[3, p],
            energy_seed[5, p],
        ),
    )
    dt = add(
        dt,
        probability_cotangent_product4(
            wp.float64(0.0),
            mean_mark,
            h,
            h,
            wp.float64(1.0) + lam,
            energy_seed[5, p],
            energy_seed[3, p],
        ),
    )
    for n in range(1, edges[bins]):
        law = pulse_law(n, p, marks, probabilities, mark_count, compound)
        b = step_bin(n, edges, bins)
        for channel in range(2):
            weight, overflow_weight, energy = (
                wp.float64(energy_seed[0, p]),
                wp.float64(energy_seed[1, p]),
                wp.float64(1.0),
            )
            if channel == 0:
                if b >= 0:
                    weight = seed[b, p]
            else:
                weight, overflow_weight, energy = (
                    wp.float64(energy_seed[2, p]),
                    wp.float64(energy_seed[3, p]),
                    wp.float64(n) * quantum,
                )
                if b >= 0:
                    weight = energy_seed[4, p]
            dr = add(
                dr,
                probability_cotangent_product4(
                    law[0],
                    inverse,
                    inverse,
                    energy,
                    wp.float64(1.0),
                    weight,
                    overflow_weight,
                ),
            )
            dr = add(
                dr,
                probability_cotangent_product4(
                    law[1], h, gate, energy, wp.float64(1.0), weight, overflow_weight
                ),
            )
            dr = add(
                dr,
                -probability_cotangent_product4(
                    law[0], h, gate, energy, wp.float64(1.0), weight, overflow_weight
                ),
            )
            dg = add(
                dg,
                -probability_cotangent_product4(
                    law[0], h, h, energy, wp.float64(1.0), weight, overflow_weight
                ),
            )
            dg = add(
                dg,
                probability_cotangent_product4(
                    law[1], h, r, energy, wp.float64(1.0), weight, overflow_weight
                ),
            )
            dg = add(
                dg,
                -probability_cotangent_product4(
                    law[0], h, r, energy, wp.float64(1.0), weight, overflow_weight
                ),
            )
            dt = add(
                dt,
                -probability_cotangent_product4(
                    law[0], h, h, energy, wp.float64(1.0), weight, overflow_weight
                ),
            )
    rate_gradient[p] = store(dr[0] + dr[1], status)
    gate_gradient[p] = store(dg[0] + dg[1], status)
    blind_gradient[p] = store(dt[0] + dt[1], status)


@wp.kernel
def pileup_readout(
    times: wp.array2d(dtype=wp.float64),
    energies: wp.array2d(dtype=wp.float64),
    event_counts: wp.array(dtype=wp.int32),
    gates: wp.array(dtype=wp.float64),
    blinds: wp.array(dtype=wp.float64),
    edges: wp.array(dtype=wp.int32),
    bins: int,
    quantum: wp.float64,
    capacity: int,
    out_times: wp.array2d(dtype=wp.float64),
    out_energies: wp.array2d(dtype=wp.float64),
    pulse_counts: wp.array(dtype=wp.int32),
    bin_counts: wp.array2d(dtype=wp.uint64),
    accounting: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    p = wp.tid()
    count, gate, blind = (
        int(event_counts[p]),
        wp.float64(gates[p]),
        wp.float64(blinds[p]),
    )
    pulse_counts[p] = 0
    for i in range(capacity):
        out_times[p, i] = wp.float64(0.0)
        out_energies[p, i] = wp.float64(0.0)
    for b in range(bins):
        bin_counts[b, p] = wp.uint64(0)
    for k in range(4):
        accounting[k, p] = wp.float64(0.0)
    if (
        count < 0
        or count > capacity
        or not wp.isfinite(gate)
        or not wp.isfinite(blind)
        or gate < wp.float64(0.0)
        or blind < wp.float64(0.0)
    ):
        wp.atomic_or(status, 0, 1)
        return
    previous = wp.float64(0.0)
    for i in range(count):
        t, input_energy = wp.float64(times[p, i]), wp.float64(energies[p, i])
        if (
            not wp.isfinite(t)
            or t < previous
            or not wp.isfinite(input_energy)
            or input_energy <= wp.float64(0.0)
        ):
            wp.atomic_or(status, 0, 1)
            return
        previous = t
    i, pulses = int(0), int(0)  # noqa: UP018, RUF046
    while i < count:
        start = times[p, i]
        gate_end, blind_end = start + gate, (start + gate) + blind
        if not wp.isfinite(blind_end):
            wp.atomic_or(status, 0, 2)
            return
        state = wp.vec2d(energies[p, i], wp.float64(0.0))
        i = i + 1
        while i < count:
            if times[p, i] >= gate_end:
                break
            state = add(state, energies[p, i])
            i = i + 1
        energy = store(state[0] + state[1], status)
        out_times[p, pulses] = start
        out_energies[p, pulses] = energy
        pulses = pulses + 1
        b = int(-2)  # noqa: UP018, RUF046
        if energy < wp.float64(edges[0]) * quantum:
            b = -1
        else:
            for j in range(bins):
                if energy < wp.float64(edges[j + 1]) * quantum:
                    b = j
                    break
        k = int(2)  # noqa: UP018, RUF046
        if b == -1:
            k = 1
        elif b >= 0:
            k = 0
            bin_counts[b, p] = bin_counts[b, p] + wp.uint64(1)
        accounting[k, p] = store(accounting[k, p] + energy, status)
        while i < count:
            if times[p, i] >= blind_end:
                break
            accounting[3, p] = store(accounting[3, p] + energies[p, i], status)
            i = i + 1
    pulse_counts[p] = pulses
