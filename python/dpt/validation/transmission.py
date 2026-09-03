"""Independent exact-input Decimal oracles for the CUDA transmission contract.

These scalar routines deliberately have no Warp dependency and no production
cutover. Inputs are quantised to binary32, reconstructed exactly by integer ratio,
and evaluated at the requested Decimal precision (100 digits by default).
"""

import math
import struct
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal, localcontext


def binary32(value: float) -> float:
    """Round a Python number to binary32; finite overflow is an explicit error."""
    try:
        result = struct.unpack("!f", struct.pack("!f", value))[0]
    except OverflowError as exc:
        raise ValueError("input exceeds finite binary32 range") from exc
    if not math.isfinite(result):
        raise ValueError("input must be finite binary32")
    return result


def _from_bits(bits: int) -> float:
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def _bits(value: float) -> int:
    return struct.unpack("!I", struct.pack("!f", value))[0]


def exact_input(value: float, *, nonnegative: bool = False) -> Decimal:
    """Recover the exact stored binary32 value, preserving signed zero."""
    stored = binary32(value)
    if nonnegative and stored < 0:
        raise ValueError("input must be non-negative")
    numerator, denominator = stored.as_integer_ratio()
    # A binary32 denominator is at most 2**149: 200 digits make this exact.
    with localcontext() as context:
        context.prec = 200
        result = Decimal(numerator) / Decimal(denominator)
    return result.copy_negate() if stored == 0 and math.copysign(1, stored) < 0 else result


MIN_SUBNORMAL = Decimal.from_float(_from_bits(1))
MIN_NORMAL = Decimal.from_float(_from_bits(0x00800000))
MAX_FINITE = Decimal.from_float(_from_bits(0x7F7FFFFF))
with localcontext() as _context:
    _context.prec = 200
    U32 = Decimal(2) ** -24
    U64 = Decimal(2) ** -53
    OVERFLOW_MIDPOINT = MAX_FINITE + Decimal(2) ** 103


def round_binary32(value: Decimal) -> float:
    """Round Decimal directly to nearest binary32, ties to even.

    Binary search compares exact neighbours; converting through binary64 could
    incorrectly move a Decimal value onto a binary32 midpoint. Overflow follows
    round-to-nearest semantics and returns a signed infinity.
    """
    if not value.is_finite():
        raise ValueError("reference value must be finite")
    negative = value.is_signed()
    magnitude = value.copy_abs()
    if magnitude >= OVERFLOW_MIDPOINT:
        return -math.inf if negative else math.inf
    low, high = 0, 0x7F7FFFFF
    while low < high:
        middle = (low + high + 1) // 2
        if Decimal.from_float(_from_bits(middle)) <= magnitude:
            low = middle
        else:
            high = middle - 1
    chosen = low
    if low < 0x7F7FFFFF:
        # Exact endpoints need up to 150 decimal places, independent of caller context.
        with localcontext() as context:
            context.prec = max(200, len(value.as_tuple().digits) + 160)
            midpoint = (
                Decimal.from_float(_from_bits(low)) + Decimal.from_float(_from_bits(low + 1))
            ) / 2
        if magnitude > midpoint or (magnitude == midpoint and low % 2):
            chosen += 1
    return _from_bits(chosen | (0x80000000 if negative else 0))


def _exponential(depth: Decimal) -> tuple[Decimal, Decimal]:
    """Return exp(-L), or zero with an explicit absolute bound for enormous L.

    For L >= 1024, exp(-L) <= exp(-1024) < 2**-1477. Even two
    maximum binary32 factors and 2**31 contributions give < 2**-1190,
    far below half the smallest binary32 subnormal (2**-150). This
    bound covers every exponential weighted term in the declared contract.
    """
    if depth >= 1024:
        return Decimal(0), Decimal(2) ** -1477
    return (-depth).exp(), Decimal(0)


@dataclass(frozen=True)
class ForwardReference:
    """Mathematical values and the exponential tail bound, before storage rounding."""

    T: Decimal
    counts: Decimal
    log_T: Decimal  # noqa: N815 - match the documented mathematical output name.
    removed: Decimal
    tail_bound: Decimal

    def rounded(self) -> dict[str, float]:
        return {
            name: round_binary32(getattr(self, name))
            for name in ("T", "counts", "log_T", "removed")
        }


def forward_reference(depth: float, n0: float = 1.0, *, precision: int = 100) -> ForwardReference:
    """T=exp(-L), counts=n0*T, log_T=-L and removed=1-T.

    ``depth`` is one non-negative dimensionless optical depth and ``n0`` is a
    non-negative incident photon count, defaulting to one. Both scalar inputs
    are first rounded to binary32 and then recovered exactly as ``Decimal``.
    ``precision`` sets the significant decimal digits used for evaluation,
    defaulting to 100. The returned ``ForwardReference`` holds dimensionless
    T, log_T and removed, count-valued counts and a dimensionless exponential
    tail bound, all before output storage rounding. Its ``rounded()`` method
    produces correctly rounded binary32 values as Python floats.

    This is a forward reference with no active derivative output. For L>=1024,
    T is represented by zero with an absolute bound of 2**-1477, which keeps
    the negligible tail explicit when comparing storage-rounded outputs.

        >>> from dpt.validation.transmission import forward_reference
        >>> forward_reference(0., 20.).rounded()
        {'T': 1.0, 'counts': 20.0, 'log_T': -0.0, 'removed': 0.0}

    """
    optical_depth, beam = exact_input(depth, nonnegative=True), exact_input(n0, nonnegative=True)
    with localcontext() as context:
        context.prec = precision
        exponential, bound = _exponential(optical_depth)
        return ForwardReference(
            exponential,
            (beam * exponential).copy_abs(),
            optical_depth.copy_negate(),
            (Decimal(1) - exponential).copy_abs(),
            bound,
        )


def inverse_reference(delta: float, *, precision: int = 100) -> Decimal:
    """L=-log(1-delta) for the removed fraction delta in [0,1).

    ``delta`` is a scalar dimensionless decrement. It is rounded to binary32
    and recovered as an exact ``Decimal`` before taking the logarithm. Return
    one non-negative dimensionless ``Decimal`` optical depth with ``precision``
    significant digits, defaulting to 100. This forward inverse evaluation has
    no derivative output and preserves the supplied decrement representation,
    so it does not first form a storage-rounded transmission.

        >>> from dpt.validation.transmission import inverse_reference
        >>> float(inverse_reference(0.5))
        0.6931471805599453
        >>> inverse_reference(0.) == 0
        True

    """
    decrement = exact_input(delta, nonnegative=True)
    if decrement >= 1:
        raise ValueError("decrement must be less than one")
    with localcontext() as context:
        context.prec = precision
        return (-(Decimal(1) - decrement).ln()).copy_abs()


@dataclass(frozen=True)
class VJPReference:
    """Analytic weighted derivatives and the frozen per-pixel absolute budgets."""

    grad_L: Decimal  # noqa: N815 - L denotes optical depth throughout the API.
    grad_n0: Decimal
    budget_L: Decimal  # noqa: N815 - pair the budget with grad_L.
    budget_n0: Decimal


def vjp_reference(
    depth: float,
    n0: float = 1.0,
    *,
    seed_T: float = 0.0,  # noqa: N803 - mirror the production cotangent name.
    seed_counts: float = 0.0,
    seed_log_T: float = 0.0,  # noqa: N803 - mirror the production cotangent name.
    seed_removed: float = 0.0,
    precision: int = 100,
) -> VJPReference:
    """bar_L=-T*seed_T-n0*T*seed_counts-seed_log_T+T*seed_removed.

    The beam derivative is bar_n0=T*seed_counts. ``depth`` is dimensionless
    and ``n0`` is an incident photon count. All inputs and scalar cotangents
    are interpreted as exact stored binary32 values. Cotangents have objective
    units per corresponding output unit and each defaults to zero. Both depth
    and beam are differentiated, with no activity mask.

    ``VJPReference`` returns scalar ``Decimal`` depth and beam gradients in
    objective units and objective units per photon count respectively, together
    with absolute error budgets in the same units. Evaluation uses ``precision``
    significant decimal digits, defaulting to 100. Each budget is
    32*2**-24 times the sum of absolute contributing terms plus two minimum
    binary32 subnormals, so cancellation does not shrink the allowance.

        >>> from dpt.validation.transmission import vjp_reference
        >>> result = vjp_reference(0., 10., seed_counts=2., seed_removed=1.)
        >>> (float(result.grad_L), float(result.grad_n0))
        (-19.0, 2.0)

    """
    optical_depth, beam = exact_input(depth, nonnegative=True), exact_input(n0, nonnegative=True)
    seeds = [exact_input(value) for value in (seed_T, seed_counts, seed_log_T, seed_removed)]
    with localcontext() as context:
        context.prec = precision
        exponential, _ = _exponential(optical_depth)
        derivatives = (-exponential, -beam * exponential, Decimal(-1), exponential)
        terms = [seed * derivative for seed, derivative in zip(seeds, derivatives, strict=True)]
        beam_term = seeds[1] * exponential
        return VJPReference(
            sum(terms, Decimal(0)),
            beam_term,
            32 * U32 * sum((abs(term) for term in terms), Decimal(0)) + 2 * MIN_SUBNORMAL,
            32 * U32 * abs(beam_term) + 2 * MIN_SUBNORMAL,
        )


def inverse_vjp_reference(
    delta: float, seed: float, *, precision: int = 100
) -> tuple[Decimal, Decimal]:
    """bar_delta=seed/(1-delta) for L=-log(1-delta).

    ``delta`` is a scalar dimensionless removed fraction in [0,1) and ``seed``
    is the optical-depth cotangent in objective units. Both become exact stored
    binary32 inputs. Return (gradient,absolute budget) as two scalar ``Decimal``
    values in objective units, evaluated with ``precision`` significant digits
    and a default of 100. Only the decrement is differentiated.

    The budget is 32*2**-24*abs(gradient) plus two minimum binary32 subnormals.
    The physical inverse domain is validated before forming the derivative.

        >>> from dpt.validation.transmission import inverse_vjp_reference
        >>> derivative, budget = inverse_vjp_reference(0.5, 2.)
        >>> (float(derivative), budget > 0)
        (4.0, True)

    """
    inverse_reference(delta, precision=precision)  # Validate the physical domain.
    decrement, weight = exact_input(delta), exact_input(seed)
    with localcontext() as context:
        context.prec = precision
        gradient = weight / (1 - decrement)
        return gradient, 32 * U32 * abs(gradient) + 2 * MIN_SUBNORMAL


def scalar_beam_reference(
    depths: Iterable[float],
    seeds: Iterable[float],
    *,
    addition_depth: int,
    precision: int = 100,
) -> tuple[Decimal, Decimal]:
    """bar_n0=sum_p seed_counts[p]*exp(-depths[p]) for a shared incident beam.

    ``depths`` and ``seeds`` are equal-length iterables in the same pixel order.
    Depths are non-negative dimensionless optical depths and seeds are count
    cotangents in objective units per photon count. Each element is interpreted
    as exact binary32 storage. Return a scalar ``Decimal`` shared-beam gradient
    and its absolute error budget in objective units per photon count, using
    ``precision`` significant digits, defaulting to 100.

    Only the shared beam is differentiated. ``addition_depth`` is the actual
    number of binary64 additions along the longest reduction path, including
    additions within blocks. Its gamma bound is d*2**-53/(1-d*2**-53), added
    to per-pixel budgets and final binary32 rounding. It must be non-negative
    and remain within that bound's finite domain.

        >>> from dpt.validation.transmission import scalar_beam_reference
        >>> derivative, budget = scalar_beam_reference((0., 0.), (2., -1.),
        ...                                          addition_depth=1)
        >>> (float(derivative), budget > 0)
        (1.0, True)

    """
    if addition_depth < 0:
        raise ValueError("addition_depth must be non-negative")
    with localcontext() as context:
        context.prec = precision
        references = [
            vjp_reference(depth, seed_counts=seed, precision=precision)
            for depth, seed in zip(depths, seeds, strict=True)
        ]
        product = addition_depth * U64
        if product >= 1:
            raise ValueError("addition depth exceeds the gamma-bound domain")
        total = sum((record.grad_n0 for record in references), Decimal(0))
        magnitude = sum((abs(record.grad_n0) for record in references), Decimal(0))
        element_budget = sum((record.budget_n0 for record in references), Decimal(0))
        final_rounding = U32 * abs(total) + MIN_SUBNORMAL / 2
        return total, element_budget + product / (1 - product) * magnitude + final_rounding


def ulp_distance(actual: float, expected: float) -> int:
    """Count representable binary32 steps; opposite signed zeros have distance zero."""

    def ordered(value: float) -> int:
        bits = _bits(binary32(value))
        return 0x80000000 - (bits & 0x7FFFFFFF) if bits >> 31 else 0x80000000 + bits

    return abs(ordered(actual) - ordered(expected))


@dataclass(frozen=True)
class ValueErrorRecord:
    absolute_error: Decimal
    ulps: int
    allowed_ulps: int
    regime: str
    passed: bool


def value_error(actual: float, expected: Decimal, *, counts: bool = False) -> ValueErrorRecord:
    """Apply frozen forward budgets, including one-ULP subnormal/zero checks."""
    target = round_binary32(expected)
    actual_exact = exact_input(actual)
    distance = ulp_distance(actual, target)
    subnormal = abs(expected) < MIN_NORMAL
    allowed = 1 if subnormal else (8 if counts else 4)
    with localcontext() as context:
        context.prec = 200
        absolute = abs(actual_exact - expected)
    return ValueErrorRecord(
        absolute, distance, allowed, "subnormal" if subnormal else "normal", distance <= allowed
    )


@dataclass(frozen=True)
class AnalyticCase:
    name: str
    optical_depth: Decimal
    definition: str


def analytic_cases() -> tuple[AnalyticCase, ...]:
    """Closed-form path cases, without claiming to validate a path integrator.

    All coefficients and lengths below are exact dyadic mathematical stress inputs.
    The mm and cm expressions describe the same optical depth in different units.
    """
    coefficient, distance = Decimal("0.125"), Decimal(4)
    first_length, second_length = Decimal(1), Decimal(3)
    intercept, slope = Decimal("0.125"), Decimal("0.0625")
    return (
        AnalyticCase("empty", Decimal(0), "empty path"),
        AnalyticCase("zero-attenuation", Decimal(0), "mu = 0"),
        AnalyticCase("homogeneous", coefficient * distance, "mu*d"),
        AnalyticCase(
            "split-homogeneous",
            coefficient * first_length + coefficient * second_length,
            "mu*d1 + mu*d2",
        ),
        AnalyticCase("layered", Decimal("0.25") * 2 + Decimal("0.5") * 3, "mu1*d1 + mu2*d2"),
        AnalyticCase("millimetres", Decimal("0.125") * 4, "0.125/mm * 4 mm"),
        AnalyticCase("centimetres", Decimal("1.25") * Decimal("0.4"), "1.25/cm * 0.4 cm"),
        AnalyticCase(
            "added-segment",
            coefficient * distance + Decimal("0.25"),
            "mu*d + non-negative optical depth",
        ),
        AnalyticCase(
            "linear-coefficient",
            intercept * distance + slope * distance * distance / 2,
            "integral_0^d (a+b*s) ds",
        ),
    )


@dataclass(frozen=True)
class DirectionalRecord:
    step: float
    derivative: float
    absolute_error: float
    stencil: str


def directional_check(
    evaluate: Callable[[float], float],
    anchor: float,
    analytic_derivative: float,
    *,
    exponents: Iterable[int] = range(2, 25),
) -> tuple[DirectionalRecord, ...]:
    """Sweep a real operator callback over admissible, distinct binary32 inputs.

    Interior anchors use centred differences; zero uses a second-order one-sided
    stencil. No convergence claim is inferred from the smallest step: callers retain
    all records to expose truncation, agreement and storage-rounding regimes.
    """
    centre = binary32(anchor)
    if centre < 0:
        raise ValueError("anchor must be non-negative")
    records: list[DirectionalRecord] = []
    for exponent in exponents:
        step = 2.0**-exponent
        right = binary32(centre + step)
        if right == centre:
            continue
        if centre == 0:
            second = binary32(2 * step)
            if second <= right:
                continue
            derivative = (-3 * evaluate(centre) + 4 * evaluate(right) - evaluate(second)) / (
                2 * right
            )
            stencil = "one-sided-second-order"
        else:
            if centre - step < 0:
                continue
            left = binary32(centre - step)
            if left == centre or centre - left != right - centre:
                continue
            derivative = (evaluate(right) - evaluate(left)) / (right - left)
            stencil = "centred"
        records.append(
            DirectionalRecord(
                right - centre, derivative, abs(derivative - analytic_derivative), stencil
            )
        )
    return tuple(records)
