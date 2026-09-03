"""Exact bounded sums of retained native response-band cotangents.

Each component owns little-endian two's-complement uint64 limbs, initially
zero. Binary32 values are integers times 2**-149 and use five limbs (40 bytes);
binary64 values are integers times 2**-1074 and use 34 limbs (272 bytes).
With at most 2**31-1 bands, magnitudes are below 2**308 or 2**2129,
respectively, within the signed 320/2176-bit capacities. Intermediate sums may
exceed the floating destination's range without losing subsequent cancellation.

One lane owns each component, so updates need no data atomics. An update touches
two initial limbs and at most the remaining fixed carry/borrow chain. Finishing
uses a fixed local magnitude buffer and rounds once, directly to the requested
IEEE destination, including subnormals and ties to even. Neither operation keeps
a band-sized tape. Only invalid-input/final-overflow status uses an atomic.
"""

# Warp annotations are executable DSL expressions; imports remain kernel-local.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false

from functools import cache

import warp as wp

OPTIONS = {"fast_math": False, "fuse_fp": False, "enable_backward": False}
wp.set_module_options(OPTIONS)


def _update_body(source64: bool) -> str:
    words = 34 if source64 else 5
    bits = "uint64_t" if source64 else "uint32_t"
    fraction_bits = 52 if source64 else 23
    exponent_mask = "0x7ffu" if source64 else "0xffu"
    fraction_mask = "0x000fffffffffffffull" if source64 else "0x007fffffu"
    sign_bit = 63 if source64 else 31
    return (
        """
@BITS@ bits;
memcpy(&bits, &value, sizeof(value));
const int exponent = static_cast<int>((bits >> @FRACTION_BITS@) & @EXPONENT_MASK@);
if (exponent == @EXPONENT_MASK@) return 2;
uint64_t significand = uint64_t(bits & @FRACTION_MASK@);
int shift = 0;
if (exponent != 0) {
    significand |= uint64_t(1) << @FRACTION_BITS@;
    shift = exponent - 1;
}
if (significand == 0) return 0;
const bool negative = (bits >> @SIGN_BIT@) != 0;
const int first = shift / 64;
const int offset = shift % 64;
const uint64_t low = significand << offset;
const uint64_t high = offset == 0 ? 0 : significand >> (64 - offset);
uint64_t* words = limbs.data + size_t(component) * @WORDS@;
uint64_t carry = 0;
for (int word = first; word < @WORDS@; ++word) {
    const uint64_t term = word == first ? low : (word == first + 1 ? high : 0);
    const uint64_t old = words[word];
    if (negative) {
        const uint64_t partial = old - term;
        const uint64_t updated = partial - carry;
        carry = uint64_t(old < term || partial < carry);
        words[word] = updated;
    } else {
        const uint64_t partial = old + term;
        const uint64_t updated = partial + carry;
        carry = uint64_t(partial < old || updated < partial);
        words[word] = updated;
    }
    if (word >= first + 1 && carry == 0) break;
}
return 0;
""".replace("@BITS@", bits)
        .replace("@FRACTION_BITS@", str(fraction_bits))
        .replace("@EXPONENT_MASK@", exponent_mask)
        .replace("@FRACTION_MASK@", fraction_mask)
        .replace("@SIGN_BIT@", str(sign_bit))
        .replace("@WORDS@", str(words))
    )


def _finish_body(source64: bool, output64: bool) -> str:
    words = 34 if source64 else 5
    source_unit = -1074 if source64 else -149
    fraction_bits = 52 if output64 else 23
    output_unit = -1074 if output64 else -149
    minimum_normal = -1022 if output64 else -126
    maximum_normal = 1023 if output64 else 127
    bias = 1023 if output64 else 127
    bits = "uint64_t" if output64 else "uint32_t"
    scalar = "double" if output64 else "float"
    sign_bit = 63 if output64 else 31
    exponent_mask = "0x7ffu" if output64 else "0xffu"
    return (
        """
const uint64_t* words = limbs.data + size_t(component) * @WORDS@;
const bool negative = (words[@WORDS@ - 1] >> 63) != 0;
uint64_t magnitude[@WORDS@];
uint64_t carry = negative ? 1 : 0;
int last = -1;
for (int word = 0; word < @WORDS@; ++word) {
    const uint64_t value = negative ? ~words[word] : words[word];
    magnitude[word] = value + carry;
    carry = uint64_t(carry != 0 && magnitude[word] == 0);
    if (magnitude[word] != 0) last = word;
}
if (last < 0) return static_cast<@SCALAR@>(0.0);
uint64_t top = magnitude[last];
int top_bit = 0;
while ((top >>= 1) != 0) ++top_bit;
const int highest = last * 64 + top_bit;
int discard = highest - @FRACTION_BITS@;
if (discard < @MINIMUM_DISCARD@) discard = @MINIMUM_DISCARD@;
if (discard < 0) discard = 0;
const int first = discard / 64;
const int offset = discard % 64;
uint64_t retained = 0;
if (first < @WORDS@) {
    retained = magnitude[first] >> offset;
    if (offset != 0 && first + 1 < @WORDS@)
        retained |= magnitude[first + 1] << (64 - offset);
}
if (discard != 0) {
    const int guard_word = (discard - 1) / 64;
    const int guard_offset = (discard - 1) % 64;
    const bool guard = ((magnitude[guard_word] >> guard_offset) & 1) != 0;
    bool sticky = false;
    for (int word = 0; word < guard_word; ++word)
        sticky = sticky || magnitude[word] != 0;
    if (guard_offset != 0)
        sticky = sticky || (magnitude[guard_word] & ((uint64_t(1) << guard_offset) - 1)) != 0;
    if (guard && (sticky || (retained & 1) != 0)) ++retained;
}
// Canonical positive zero includes cancellation and destination underflow.
if (retained == 0) return static_cast<@SCALAR@>(0.0);
if (retained == (uint64_t(1) << (@FRACTION_BITS@ + 1))) {
    retained >>= 1;
    ++discard;
}
top = retained;
top_bit = 0;
while ((top >>= 1) != 0) ++top_bit;
const int exponent = @SOURCE_UNIT@ + discard + top_bit;
@BITS@ result_bits = negative ? @BITS@(1) << @SIGN_BIT@ : 0;
if (exponent > @MAXIMUM_NORMAL@) {
    result_bits |= @BITS@(@EXPONENT_MASK@) << @FRACTION_BITS@;
} else if (exponent >= @MINIMUM_NORMAL@) {
    const uint64_t significand = retained << (@FRACTION_BITS@ - top_bit);
    const uint64_t fraction = significand & ((uint64_t(1) << @FRACTION_BITS@) - 1);
    result_bits |= @BITS@(exponent + @BIAS@) << @FRACTION_BITS@;
    result_bits |= @BITS@(fraction);
} else {
    // discard already accounts for the destination's subnormal quantum.
    result_bits |= @BITS@(retained << (@SOURCE_UNIT@ + discard - @OUTPUT_UNIT@));
}
@SCALAR@ result;
memcpy(&result, &result_bits, sizeof(result));
return result;
""".replace("@WORDS@", str(words))
        .replace("@SCALAR@", scalar)
        .replace("@FRACTION_BITS@", str(fraction_bits))
        .replace("@MINIMUM_DISCARD@", str(output_unit - source_unit))
        .replace("@SOURCE_UNIT@", str(source_unit))
        .replace("@OUTPUT_UNIT@", str(output_unit))
        .replace("@BITS@", bits)
        .replace("@SIGN_BIT@", str(sign_bit))
        .replace("@MAXIMUM_NORMAL@", str(maximum_normal))
        .replace("@MINIMUM_NORMAL@", str(minimum_normal))
        .replace("@EXPONENT_MASK@", exponent_mask)
        .replace("@BIAS@", str(bias))
    )


@wp.func_native(_update_body(False))
def _add32(value: wp.float32, limbs: wp.array(dtype=wp.uint64), component: int) -> wp.int32: ...


@wp.func_native(_update_body(True))
def _add64(value: wp.float64, limbs: wp.array(dtype=wp.uint64), component: int) -> wp.int32: ...


@wp.func_native(_finish_body(False, False))
def _finish32_to32(limbs: wp.array(dtype=wp.uint64), component: int) -> wp.float32: ...


@wp.func_native(_finish_body(False, True))
def _finish32_to64(limbs: wp.array(dtype=wp.uint64), component: int) -> wp.float64: ...


@wp.func_native(_finish_body(True, False))
def _finish64_to32(limbs: wp.array(dtype=wp.uint64), component: int) -> wp.float32: ...


@wp.func_native(_finish_body(True, True))
def _finish64_to64(limbs: wp.array(dtype=wp.uint64), component: int) -> wp.float64: ...


@cache
def get_accumulate(source64: bool):
    """Add source[count] exactly into zero-initialized, component-major limbs."""
    dtype = wp.float64 if source64 else wp.float32
    add = _add64 if source64 else _add32

    @wp.kernel(module="unique", module_options=OPTIONS)
    def accumulate(
        source: wp.array(dtype=dtype),
        count: int,
        limbs: wp.array(dtype=wp.uint64),
        status: wp.array(dtype=wp.int32),
    ):
        component = wp.tid()
        if component < count:
            code = add(source[component], limbs, component)
            if code != 0:
                wp.atomic_or(status, 0, code)

    return accumulate


@cache
def get_finish(source64: bool, output64: bool):
    """Round each exact sum directly; flag only genuine destination overflow."""
    dtype = wp.float64 if output64 else wp.float32

    if source64:
        finish_scalar = _finish64_to64 if output64 else _finish64_to32
    else:
        finish_scalar = _finish32_to64 if output64 else _finish32_to32

    @wp.kernel(module="unique", module_options=OPTIONS)
    def finish(
        limbs: wp.array(dtype=wp.uint64),
        count: int,
        output: wp.array(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        component = wp.tid()
        if component < count:
            value = finish_scalar(limbs, component)
            output[component] = value
            if not wp.isfinite(value):
                wp.atomic_or(status, 0, 2)

    return finish
