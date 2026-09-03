"""Hand-derived bounded SVD cases; optional scientific environment required."""

import importlib
import math

import pytest

from dpt.contracts import ContractError
from dpt.validation.identifiability import sensitivity_spectrum

importlib.import_module("numpy")


def test_units_and_independent_view_change_the_declared_sensitivity() -> None:
    weak = sensitivity_spectrum([[1, 0], [2, 0]], [1, 10])
    recovered = sensitivity_spectrum([[1, 0], [2, 0], [0, 0.1]], [1, 10])
    assert weak.rank == 1 and weak.condition_number is None
    assert recovered.rank == 2
    assert math.isclose(recovered.singular_values[0], math.sqrt(5))
    assert math.isclose(recovered.singular_values[1], 1)
    assert math.isclose(abs(weak.right_directions[-1][1]), 1)


def test_underdetermined_case_retains_the_missing_parameter_directions() -> None:
    result = sensitivity_spectrum([[1, 0, 0]], [1, 1, 1])
    assert result.singular_values == (1, 0, 0)
    assert len(result.right_directions) == 3
    assert result.rank == 1


def test_weights_and_rank_threshold_have_separate_meaning() -> None:
    result = sensitivity_spectrum(
        [[1, 0], [0, 1e-4]], [1, 1], precision_weights=[4, 1], relative_threshold=1e-3
    )
    assert result.rank == 1
    assert result.singular_values == (2, 1e-4)
    assert math.isclose(result.threshold, 0.002)
    with pytest.raises(ContractError):
        sensitivity_spectrum([[1]], [0])


@pytest.mark.parametrize(
    ("entry", "unit", "weight", "expected"),
    [(1e308, 1e10, 1e-40, 1e298), (1e-308, 1e-20, 1e40, 1e-308), (1e308, 1e10, 0, 0)],
)
def test_scaling_preserves_the_final_representable_product(
    entry: float, unit: float, weight: float, expected: float
) -> None:
    result = sensitivity_spectrum([[entry]], [unit], precision_weights=[weight])
    assert math.isclose(result.singular_values[0], expected, rel_tol=1e-14, abs_tol=0)
