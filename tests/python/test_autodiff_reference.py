"""Directional-check bookkeeping must not erase units or constrained boundaries."""

# Pytest 9.1 exposes approx() with incomplete annotations; production modules stay strict.
# pyright: reportUnknownMemberType=false

import pytest

from dpt.validation.autodiff import directional_sweep


def test_scaled_central_check_has_physical_units() -> None:
    records = directional_sweep(
        lambda x: x[0] ** 2 + 3 * x[1] ** 2,
        (2.0, 4.0),
        (4.0, 24.0),
        (0.4, 0.7),
        (1e-2, 1e-3, 1e-4),
        scales=(10.0, 0.01),
    )
    assert all(record.predicted == pytest.approx(16.168) for record in records)
    assert max(record.absolute_error for record in records) < 1e-7


def test_boundary_check_is_explicitly_one_sided() -> None:
    records = directional_sweep(
        lambda x: x[0] ** 2, (0.0,), (0.0,), (1.0,), (0.1, 0.01), admissible=lambda x: x[0] >= 0
    )
    assert [record.scheme for record in records] == ["forward", "forward"]
    assert records[1].absolute_error < records[0].absolute_error


def test_nonfinite_result_is_not_a_passing_check() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        directional_sweep(lambda x: float("nan"), (1.0,), (1.0,), (1.0,), (0.1,))
