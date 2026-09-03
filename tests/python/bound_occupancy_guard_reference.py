"""Original-law bound for an irrelevant profile beyond the binary64 exponent range."""

from __future__ import annotations

from decimal import Decimal, localcontext

import bound_occupancy_reference as original
from bound_occupancy_reference import Shells


def guard_fixtures() -> tuple[tuple[str, Shells], ...]:
    minimum = float.fromhex("0x0.0000000000001p-1022")
    huge = (1.0, 1.0, 1e200)
    broad = (minimum, 0.0, 0.01)
    return (("overflow-steep-first", (huge, broad)), ("overflow-steep-last", (broad, huge)))


def broad_reference_shells(shells: Shells) -> tuple[int, Shells]:
    index = next(index for index, shell in enumerate(shells) if shell[1] == 0.0)
    return index, (shells[index],)


def omitted_physical_log_bound(energy: float, shells: Shells) -> Decimal:
    """KN<=2 bounds the huge component by8*pi*f*M_back without exp(-1e400)."""
    with localcontext() as context:
        context.prec = original.PRECISION
        broad_index, broad = broad_reference_shells(shells)
        reference = original.original_occupancy_normalisation(energy, broad)
        index = 1 - broad_index
        occupancy, binding, height = shells[index]
        endpoint = original._endpoint(
            Decimal.from_float(energy), Decimal(-1), Decimal.from_float(binding)
        )[0]
        assert -1 < endpoint < 0
        log_mass_upper = (
            original._profile_log(endpoint, Decimal.from_float(height)) - Decimal(2).ln()
        )
        return (
            Decimal.from_float(occupancy).ln()
            + log_mass_upper
            + (8 * original.PI).ln()
            - reference.raw_value.ln()
        )
