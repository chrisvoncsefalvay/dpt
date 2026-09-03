"""Independent original-law references for saturated truncation arithmetic."""

from decimal import Decimal, localcontext

import bound_occupancy_reference as original
from bound_occupancy_reference import Shells


def overflow_fixtures():
    minimum = float.fromhex("0x0.0000000000001p-1022")
    groups = (
        ("ordinary-z2", 1.01, 2, (1.0, 1.0, 1e200), (1.0, 0.0, 0.01)),
        ("finite-cutoff-broad", 3.0, 1, (1.0, 1.0, 1e154), (minimum, 0.0, 0.01)),
        ("finite-cutoff-negative", 3.0, 1, (1.0, 1.0, 1e154), (minimum, 1.0, 1e3)),
    )
    return tuple(
        (label + suffix, energy, z, shells)
        for label, energy, z, huge, active in groups
        for suffix, shells in (("-first", (huge, active)), ("-last", (active, huge)))
    )


def finite_reference_shells(shells: Shells):
    index = min(range(len(shells)), key=lambda i: shells[i][2])
    return index, (shells[index],)


def omitted_physical_log_bound(energy: float, shells: Shells):
    with localcontext() as context:
        context.prec = original.PRECISION
        index, active = finite_reference_shells(shells)
        reference = original.original_occupancy_normalisation(energy, active)
        occupancy, binding, height = shells[1 - index]
        endpoint = original._endpoint(
            Decimal.from_float(energy), Decimal(-1), Decimal.from_float(binding)
        )[0]
        assert -1 < endpoint < 0
        log_mass = original._profile_log(endpoint, Decimal.from_float(height)) - Decimal(2).ln()
        return (
            Decimal.from_float(occupancy).ln()
            + log_mass
            + (8 * original.PI).ln()
            - reference.raw_value.ln()
        )


def original_cutoff_range(energy: float, binding: float, height: float):
    with localcontext() as context:
        context.prec = original.PRECISION
        endpoint = original._endpoint(
            Decimal.from_float(energy), Decimal(-1), Decimal.from_float(binding)
        )[0]
        h = Decimal.from_float(height)
        cutoff = 2 * h * (1 + endpoint) * (1 + h * (1 - endpoint))
        profile = original._profile_log(endpoint, h)
        return cutoff, profile
