"""Range-preserving material/source expected-score contractions.

Five original finite binary64 factors retain a normalised mantissa and a
separate integer exponent. Each history contributes its rounded 53-bit
mantissa to at most three signed base-2**32 bins. With N < 2**31, each bin's
absolute sum is below (2**31-1)*(2**32-1), so int64 atomics cannot overflow.
Carrying, division by the original N and final ties-to-even rounding use integer
arithmetic. Huge signed terms can cancel before the destination range is tested.

The exponent window follows the binary64 factor limits and the shared
attenuation helper's 8192-depth cutoff. Above that depth, even five maximal
finite factors produce less than half the smallest binary64 subnormal. The
window retains all significand bits of each computed extended contribution
within the shared attenuation cutoff.
Scratch depends on the four-coordinate chart, never on the number of histories.
"""

# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportPrivateUsage=false
import math

import warp as wp

from dpt.kernels.numerics import attenuation_state

from . import kernels as transport
from ._constants import EVENT_BUDGET, INPUT_MASK, NUMERICAL_MASK

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})

FACTOR_COUNT = 5
MIN_FACTOR_EXPONENT = -1073  # frexp(2**-1074) = (0.5, -1073).
MAX_FACTOR_EXPONENT = 1024
ATTENUATION_DEPTH_CUTOFF = 8192
MAX_ATTENUATION_EXPONENT = math.ceil(ATTENUATION_DEPTH_CUTOFF / math.log(2))
# Multiplication of five mantissas and one attenuation mantissa can move the
# normalised exponent down by five. Keep every bit of the resulting significand.
MIN_SIGNIFICAND_BIT = (
    FACTOR_COUNT * MIN_FACTOR_EXPONENT - FACTOR_COUNT - MAX_ATTENUATION_EXPONENT - 53
)
BIN_UNIT = (MIN_SIGNIFICAND_BIT // 32) * 32
# N <= 2**31-1 adds at most31 high bits; one further bit stores the sign.
MAX_SUM_BIT = FACTOR_COUNT * MAX_FACTOR_EXPONENT + 31
BIN_COUNT = (MAX_SUM_BIT + 1 - BIN_UNIT + 31) // 32


@wp.func_native("""
if (!wp::isfinite(a) || !wp::isfinite(b) || !wp::isfinite(c)
    || !wp::isfinite(d) || !wp::isfinite(e) || !wp::isfinite(state[0]))
    return wp::vec_t<2, double>(NAN, 0.0);
if (a == 0.0 || b == 0.0 || c == 0.0 || d == 0.0 || e == 0.0 || state[0] == 0.0)
    return wp::vec_t<2, double>(0.0, 0.0);
double factors[5] = {a, b, c, d, e};
double mantissa = 1.0;
int exponent = -static_cast<int>(state[1]);
for (int factor = 0; factor < 5; ++factor) {
    double value = factors[factor];
    int correction = 0;
    if (wp::abs(value) < 2.2250738585072014e-308) {
        value *= 18014398509481984.0;
        correction = 54;
    }
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    exponent += static_cast<int>((bits >> 52) & 0x7ff) - 1022 - correction;
    bits = (bits & 0x800fffffffffffffull) | 0x3fe0000000000000ull;
    memcpy(&value, &bits, sizeof(value));
    mantissa *= value;
}
mantissa *= state[0];
uint64_t bits;
memcpy(&bits, &mantissa, sizeof(bits));
exponent += static_cast<int>((bits >> 52) & 0x7ff) - 1022;
bits = (bits & 0x800fffffffffffffull) | 0x3fe0000000000000ull;
memcpy(&mantissa, &bits, sizeof(bits));
return wp::vec_t<2, double>(mantissa, double(exponent));
""")
def extended_product5(
    state: wp.vec2d,
    a: wp.float64,
    b: wp.float64,
    c: wp.float64,
    d: wp.float64,
    e: wp.float64,
) -> wp.vec2d:
    """Combine original factors and attenuation without restricting their exponent."""
    ...


@wp.func_native(
    """
const double value = extended[0];
if (!wp::isfinite(value)) return 2;
if (value == 0.0) return 0;
uint64_t bits;
memcpy(&bits, &value, sizeof(bits));
const bool negative = (bits >> 63) != 0;
const uint64_t significand = (bits & 0x000fffffffffffffull) | (uint64_t(1) << 52);
const int shift = static_cast<int>(extended[1]) - 53 - @UNIT@;
const int first = shift / 32;
const int offset = shift % 32;
if (shift < 0 || first + 2 >= @COUNT@) return 2;
const uint64_t low = significand << offset;
const uint64_t high = offset == 0 ? 0 : significand >> (64 - offset);
int64_t* target = bins.data + size_t(component) * @COUNT@;
const int64_t sign = negative ? -1 : 1;
wp::atomic_add(target + first, sign * int64_t(low & 0xffffffffull));
wp::atomic_add(target + first + 1, sign * int64_t(low >> 32));
wp::atomic_add(target + first + 2, sign * int64_t(high));
return 0;
""".replace("@UNIT@", str(BIN_UNIT)).replace("@COUNT@", str(BIN_COUNT))
)
def add_extended(extended: wp.vec2d, bins: wp.array(dtype=wp.int64), component: int) -> wp.int32:
    """Accumulate at most three bounded signed digits with concurrent int64 atomics."""
    ...


@wp.func_native(
    """
const int64_t* source = bins.data + size_t(component) * @COUNT@;
uint32_t magnitude[@COUNT@];
int64_t carry = 0;
for (int word = 0; word < @COUNT@; ++word) {
    const int64_t value = source[word] + carry;
    const uint32_t digit = uint32_t(uint64_t(value) & 0xffffffffull);
    magnitude[word] = digit;
    carry = (value - int64_t(digit)) / 4294967296ll;
}
const bool negative = carry < 0;
if (negative) {
    uint64_t complement_carry = 1;
    for (int word = 0; word < @COUNT@; ++word) {
        const uint64_t digit = uint64_t(~magnitude[word]) + complement_carry;
        magnitude[word] = uint32_t(digit);
        complement_carry = digit >> 32;
    }
}
// Divide the exact positive magnitude by original-history count, retaining a
// nonzero remainder as sticky information for final ties-to-even rounding.
uint64_t remainder = 0;
int last = -1;
for (int word = @COUNT@ - 1; word >= 0; --word) {
    const uint64_t dividend = (remainder << 32) | magnitude[word];
    magnitude[word] = uint32_t(dividend / uint64_t(count));
    remainder = dividend % uint64_t(count);
    if (last < 0 && magnitude[word] != 0) last = word;
}
if (last < 0) return 0.0;
uint64_t top = magnitude[last];
int top_bit = 0;
while ((top >>= 1) != 0) ++top_bit;
const int highest = last * 32 + top_bit;
int discard = highest - 52;
if (discard < @MIN_DISCARD@) discard = @MIN_DISCARD@;
const int first = discard / 32;
const int offset = discard % 32;
uint64_t retained = 0;
if (first < @COUNT@) {
    retained = uint64_t(magnitude[first]) >> offset;
    if (first + 1 < @COUNT@)
        retained |= uint64_t(magnitude[first + 1]) << (32 - offset);
    if (offset != 0 && first + 2 < @COUNT@)
        retained |= uint64_t(magnitude[first + 2]) << (64 - offset);
}
const int guard_word = (discard - 1) / 32;
const int guard_offset = (discard - 1) % 32;
const bool guard = guard_word < @COUNT@
    && ((magnitude[guard_word] >> guard_offset) & 1) != 0;
bool sticky = remainder != 0;
for (int word = 0; word < guard_word && word < @COUNT@; ++word)
    sticky = sticky || magnitude[word] != 0;
if (guard_offset != 0 && guard_word < @COUNT@)
    sticky = sticky || (magnitude[guard_word] & ((uint32_t(1) << guard_offset) - 1)) != 0;
if (guard && (sticky || (retained & 1) != 0)) ++retained;
if (retained == 0) return 0.0;
if (retained == (uint64_t(1) << 53)) {
    retained >>= 1;
    ++discard;
}
top = retained;
top_bit = 0;
while ((top >>= 1) != 0) ++top_bit;
const int exponent = @UNIT@ + discard + top_bit;
uint64_t result_bits = negative ? uint64_t(1) << 63 : 0;
if (exponent > 1023) {
    result_bits |= uint64_t(0x7ff) << 52;
} else if (exponent >= -1022) {
    const uint64_t significand = retained << (52 - top_bit);
    result_bits |= uint64_t(exponent + 1023) << 52;
    result_bits |= significand & 0x000fffffffffffffull;
} else {
    result_bits |= retained << (@UNIT@ + discard + 1074);
}
double result;
memcpy(&result, &result_bits, sizeof(result));
return result;
""".replace("@UNIT@", str(BIN_UNIT))
    .replace("@COUNT@", str(BIN_COUNT))
    .replace("@MIN_DISCARD@", str(-1074 - BIN_UNIT))
)
def finish_extended(bins: wp.array(dtype=wp.int64), component: int, count: int) -> wp.float64:
    """Divide the exact signed sum by N and round directly to the FP64 gradient."""
    ...


@wp.kernel
def validate_cotangent(cotangent: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
    if not wp.isfinite(cotangent[wp.tid()]):
        wp.atomic_or(status, 0, INPUT_MASK | NUMERICAL_MASK)


@wp.kernel
def replay_contracted(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: transport.Parameters,
    seed: wp.uint64,
    first_history: wp.uint64,
    active_materials: wp.vec4i,
    active_count: int,
    source_amplitude: wp.float64,
    cotangent: wp.array(dtype=wp.float64),
    bins: wp.array(dtype=wp.int64),
    status: wp.array(dtype=wp.int32),
):
    history = wp.tid()
    # The primal is unused. Canonical path/status/RNG behaviour remains intact.
    result = transport.walk(
        positions[history],
        directions[history],
        wp.float64(0.0),
        wp.float64(1.0),
        seed,
        first_history + wp.uint64(history),
        transport.SOURCE_AMPLITUDE,
        active_materials,
        active_count,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        parameters,
    )
    if result.status >= EVENT_BUDGET:
        wp.atomic_or(status, 0, 1 << result.status)
    if result.pixel >= 0 and result.status < EVENT_BUDGET:
        score_energy = wp.float64(1.0)
        if parameters.energy_score != 0:
            score_energy = result.energy
        state = wp.vec2d(wp.float64(1.0), wp.float64(0.0))
        if parameters.continuous_absorption != 0:
            state = attenuation_state(result.absorption_depth)
        for column in range(active_count):
            factor, amplitude = transport._derivative_factors(
                result.density_scores[column], active_materials[column], source_amplitude
            )
            value = extended_product5(
                state, weights[history], amplitude, score_energy, cotangent[result.pixel], factor
            )
            code = add_extended(value, bins, column)
            if code != 0:
                wp.atomic_or(status, 0, NUMERICAL_MASK)


@wp.kernel
def finish_contracted(
    bins: wp.array(dtype=wp.int64),
    count: int,
    out_gradient: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    column = wp.tid()
    value = finish_extended(bins, column, count)
    out_gradient[column] = value
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, NUMERICAL_MASK)
