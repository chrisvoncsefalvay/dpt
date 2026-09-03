"""Transport execution limits preserve estimator failure distinctions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from dpt.contracts import ContractError
from dpt.transport import (
    TransportExecution,
    TransportSpec,
)
from dpt.transport.model import TransportError
from helpers.execution_models import transport_physics


@pytest.mark.parametrize("name", ["max_events", "max_crossings", "max_angle_trials"])
def test_transport_execution_guards_remain_positive_signed_integer_bounds(name: str) -> None:
    for invalid in (0, -1, True, 2**31):
        with pytest.raises(TransportError, match="positive signed"):
            TransportExecution(**{name: cast(Any, invalid)})


def test_transport_selected_guard_is_used_for_coefficient_support() -> None:
    physics = replace(
        transport_physics(),
        coefficient_energies_kev=(20.0, 60.0),
        scattering_law="free-electron-compton",
    )
    one_event = TransportExecution(max_events=1, energy_support_policy="cover-budget")
    assert TransportSpec.resolve(physics, one_event).max_events == 1
    with pytest.raises(TransportError, match="collision budget"):
        TransportSpec.resolve(physics, replace(one_event, max_events=100))
    # The runtime guard has its original meaning even when cover-budget is not requested.
    assert (
        TransportSpec.resolve(
            physics, replace(one_event, max_events=100, energy_support_policy="runtime-guard")
        ).max_events
        == 100
    )


@pytest.mark.parametrize("record", [TransportExecution])
def test_execution_launch_sizes_reject_boolean_and_unsupported_sizes(record: Any) -> None:
    for invalid in (True, 32, 512):
        with pytest.raises(ContractError, match="block_dim"):
            record(block_dim=invalid)
