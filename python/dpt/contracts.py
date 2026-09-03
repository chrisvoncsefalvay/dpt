"""CPU-safe scalar contracts shared by scientific operator interfaces.

Validation never converts an array, changes physical units or clips a value.
Conversions at data-ingestion boundaries must be named by their caller.
"""

from __future__ import annotations

import math
import struct
from numbers import Real
from typing import Any, cast


class ContractError(ValueError):
    """A value, layout or ownership rule required by an operator was violated."""


class NumericalError(FloatingPointError):
    """An operation cannot represent its declared result; discard its outputs."""


class TrialDomainError(NumericalError):
    """A proposed parameter point is outside its chart; the incumbent remains valid.

    Only parameter-domain failures use this type. A failed transport estimate,
    corrupt fixed input or incomplete history must never masquerade as a trial
    that can be fixed by shrinking an optimisation step.
    """


def integer(value: Any, name: str, *, minimum: int = 0, maximum: int = 2**31 - 1) -> int:
    """Require a Python integer; booleans and lossy coercions are not counts."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise ContractError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def finite_scalar(value: Any, name: str, *, minimum: float | None = None) -> float:
    """Accept a real finite scalar, optionally with an inclusive lower bound."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContractError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = "finite" if minimum is None else f"finite and at least {minimum}"
        raise ContractError(f"{name} must be {qualifier}")
    return result


def binary32_scalar(value: Any, name: str, *, minimum: float | None = None) -> float:
    """Round an explicitly supplied real scalar once to finite binary32 storage."""
    result = finite_scalar(value, name, minimum=minimum)
    try:
        result = struct.unpack("f", struct.pack("f", result))[0]
    except (OverflowError, struct.error) as error:
        raise ContractError(f"{name} cannot be represented in binary32") from error
    if not math.isfinite(result):
        raise ContractError(f"{name} cannot be represented in binary32")
    return result


def finite_tuple(
    values: Any,
    name: str,
    *,
    positive: bool = False,
    minimum: float | None = 0.0,
    length: int | None = None,
) -> tuple[float, ...]:
    """Validate immutable scalar input; physical tables default to nonnegative."""
    if not isinstance(values, tuple) or not values:
        raise ContractError(f"{name} must be a non-empty immutable tuple")
    supplied = cast(tuple[Any, ...], values)
    if length is not None and len(supplied) != length:
        raise ContractError(f"{name} must contain {length} values")
    result = tuple(finite_scalar(value, name, minimum=minimum) for value in supplied)
    if positive and any(value <= 0.0 for value in result):
        raise ContractError(f"{name} must be positive")
    return result


DEVICE_STATUS_EXCEPTIONS = {
    1: ContractError,
    8: TrialDomainError,
    2: NumericalError,
    4: NumericalError,
}


def raise_device_status(
    code: int,
    operator: str,
    *,
    input_error: type[Exception] = ContractError,
    range_error: type[Exception] = NumericalError,
    messages: dict[int, str] | None = None,
) -> None:
    """Shared checked/unchecked/fused status decoding; checking never clears.

    Fixed-input corruption takes precedence. A physical prediction outside the
    declared trial domain is rejectable even if downstream arithmetic is nonfinite.
    Other range failures abort rather than silently becoming optimiser trials.
    """
    default_messages = {
        1: "invalid fixed device input",
        8: "prediction outside trial domain",
        2: "output or gradient cannot be represented",
        4: "numerical execution failure",
    }
    descriptions = default_messages if messages is None else default_messages | messages
    for bit, error_type in DEVICE_STATUS_EXCEPTIONS.items():
        if code & bit:
            if bit == 1:
                error_type = input_error
            if bit == 2:
                error_type = range_error
            raise error_type(f"{operator}: {descriptions[bit]}")
    if code:
        raise NumericalError(f"{operator}: unknown device status {code}")
