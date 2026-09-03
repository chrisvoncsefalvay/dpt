"""Objective semantics and independent limits; CUDA cases are separate."""

from decimal import Decimal
from typing import Any, cast

import pytest

from dpt.contracts import ContractError
from dpt.objectives import ObjectiveSpec
from dpt.validation.objectives import scalar_objective


def test_poisson_requires_a_count_domain():
    with pytest.raises(ContractError, match="count-domain"):
        ObjectiveSpec(kind="poisson", domain="signal")


def test_empty_count_limit_retains_its_derivative():
    assert scalar_objective(0.0, 0.0, poisson=True) == (Decimal(0), Decimal(1))


def test_positive_count_at_zero_mean_is_not_repaired_by_a_floor():
    with pytest.raises(ValueError, match="zero predicted"):
        scalar_objective(0.0, 1.0, poisson=True)


def test_deviance_and_gradient_vanish_at_exact_agreement():
    assert scalar_objective(32768.0, 32768.0, poisson=True) == (Decimal(0), Decimal(0))


def test_squared_error_retains_signed_signal_domain():
    assert scalar_objective(-3.0, -1.0, poisson=False) == (Decimal(2), Decimal(-2))


def test_objective_precision_is_explicit_and_preserves_legacy_default() -> None:
    assert ObjectiveSpec().precision == "float32"
    # The new optional field follows the existing positional arguments.
    assert ObjectiveSpec("poisson", "counts", "mean", True, True).precision == "float32"
    assert ObjectiveSpec(precision="float64").precision == "float64"


@pytest.mark.parametrize("precision", ["float16", "double", "", None, True, 64])
def test_objective_rejects_unknown_precision(precision: object) -> None:
    with pytest.raises(ContractError, match="precision"):
        ObjectiveSpec(precision=cast(Any, precision))
