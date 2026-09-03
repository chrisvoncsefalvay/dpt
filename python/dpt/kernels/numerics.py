"""Shared range-preserving binary64 products for photon transport operators."""

# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false
import warp as wp

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})


def _product_body(factors: tuple[str, ...], *, attenuated: bool = False) -> str:
    """Generate straight-line native arithmetic with one mantissa decomposition."""
    ordinary_product = " * ".join(factors)
    invalid = " || ".join(f"!wp::isfinite({factor})" for factor in factors)
    zeros = " || ".join(f"{factor} == 0.0" for factor in factors)
    lines = [f"if ({invalid}) return {ordinary_product};"]
    if attenuated:
        lines.append("if (!wp::isfinite(state[0])) return state[0];")
    lines.append(f"if ({zeros}) return 0.0;")
    for factor in factors:
        lines.append(f"""
int e{factor};
const double m{factor} = [&]() {{
    double value = {factor};
    int correction = 0;
    if (wp::abs(value) < 2.2250738585072014e-308) {{
        value *= 18014398509481984.0;
        correction = 54;
    }}
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    e{factor} = static_cast<int>((bits >> 52) & 0x7ff) - 1022 - correction;
    bits = (bits & 0x800fffffffffffffull) | 0x3fe0000000000000ull;
    memcpy(&value, &bits, sizeof(value));
    return value;
}}();
""")
    mantissa = " * ".join(f"m{factor}" for factor in factors)
    exponent = " + ".join(f"e{factor}" for factor in factors)
    if attenuated:
        mantissa += " * state[0]"
        exponent += " - static_cast<int>(state[1])"
    lines.append(f"return ldexp({mantissa}, {exponent});")
    return "\n".join(lines)


@wp.func_native(_product_body(("a", "b", "c")))
def product3(a: wp.float64, b: wp.float64, c: wp.float64) -> wp.float64:
    """Multiply finite factors without premature exponent overflow or underflow."""
    ...


@wp.func_native(_product_body(("a", "b", "c", "d")))
def product4(a: wp.float64, b: wp.float64, c: wp.float64, d: wp.float64) -> wp.float64:
    """Preserve a derivative's range independently of its rounded primal value."""
    ...


@wp.func_native("""
if (!wp::isfinite(tau) || tau < 0.0)
    return wp::vec_t<2, double>(NAN, 0.0);
// Four finite binary64 factors cannot rescue attenuation above this bound.
// Check before converting the optical depth to an integer exponent.
if (tau > 8192.0) return wp::vec_t<2, double>(0.0, 0.0);
const int exponent = static_cast<int>(floor(tau * 1.4426950408889634074));
// Split ln(2) and one explicit FMA preserve the range-reduction residual.
const double residual = fma(double(exponent), 0.69314718055994530942, -tau)
    + double(exponent) * 2.3190468138462995584e-17;
return wp::vec_t<2, double>(exp(residual), double(exponent));
""")
def attenuation_state(tau: wp.float64) -> wp.vec2d:
    """Range-reduce exp(-tau) once; retain its multiplier and binary exponent."""
    ...


@wp.func_native(_product_body(("a", "b", "c", "d"), attenuated=True))
def attenuated_product4_from_state(
    state: wp.vec2d, a: wp.float64, b: wp.float64, c: wp.float64, d: wp.float64
) -> wp.float64:
    """Apply a prepared attenuation without rounding exp(-tau) on its own."""
    ...


@wp.func
def attenuated_product4(
    tau: wp.float64, a: wp.float64, b: wp.float64, c: wp.float64, d: wp.float64
) -> wp.float64:
    """Multiply original factors by exp(-tau) before final binary64 rounding."""
    return attenuated_product4_from_state(attenuation_state(tau), a, b, c, d)
