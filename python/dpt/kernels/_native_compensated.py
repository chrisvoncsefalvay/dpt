"""Shared native source for exponent-aligned binary64 compensated updates."""

_SCALED_KAHAN_UPDATE_CPP = (
    "    const double adjusted = "
    "ldexp(mantissas[i], exponents[i] - common) - correction;\n"
    "    const double next = sum + adjusted;\n"
    "    correction = (next - sum) - adjusted;\n"
    "    sum = next;\n"
)
