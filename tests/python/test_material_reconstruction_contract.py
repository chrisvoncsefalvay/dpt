"""CPU-safe solver policy contracts; no optional runtime is imported here."""

from typing import Any

import pytest

from dpt.contracts import ContractError
from dpt.material_reconstruction import MaterialReconstructionSettings


@pytest.mark.parametrize(
    "settings",
    [
        {"iterations": 0},
        {"maximum_backtracks": 0},
        {"initial_step": 0.0},
        {"initial_step": 2.0, "maximum_step": 1.0},
        {"mapping_step": float("nan")},
        {"regularisation_mm_inverse": -1.0},
        {"relative_gradient_mapping_tolerance": -1.0},
        {"backtracking_factor": 1.0},
        {"armijo": 0.0},
        {"step_selection": "unrestricted"},
        {"acceleration": "unrestricted"},
        {"secant_minimum_step": 0.0},
        {"secant_minimum_step": float("nan")},
        {"step_selection": "bb", "secant_minimum_step": 2.0},
    ],
)
def test_invalid_material_solver_policy_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ContractError):
        MaterialReconstructionSettings(**settings)


def test_geometric_default_does_not_constrain_historical_tiny_initial_steps() -> None:
    settings = MaterialReconstructionSettings(initial_step=1e-30, maximum_step=1e-30)
    assert settings.step_selection == "geometric"
