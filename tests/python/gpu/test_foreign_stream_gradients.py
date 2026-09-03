"""Joint rigid/factor and statistical score saved states on foreign streams."""

from __future__ import annotations

from typing import Any, Literal

import pytest

from helpers.torch_gradient_references import (
    run_joint_pose_factor_saved_state,
    run_statistical_transport_saved_state,
)

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("domain", ["linear", "log"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("execution", ["serial", "fused"])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_joint_pose_factors_and_saved_states_on_foreign_streams(
    domain: Literal["linear", "log"],
    shared: bool,
    execution: Literal["serial", "fused"],
    reverse_order: bool,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Any,
) -> None:
    run_joint_pose_factor_saved_state(
        "cuda:0", domain, shared, execution, reverse_order, monkeypatch, record_property
    )


@pytest.mark.parametrize("scoring", ["photon-count", "energy-kev"])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_expected_score_replicates_and_saved_states_on_foreign_streams(
    scoring: Literal["photon-count", "energy-kev"],
    reverse_order: bool,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Any,
) -> None:
    run_statistical_transport_saved_state(
        "cuda:0", scoring, reverse_order, monkeypatch, record_property
    )
