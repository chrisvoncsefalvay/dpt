"""Independent joint rigid-chart/factor and statistical transport acceptance."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Literal

import pytest

from dpt._runtime import cpu_test_context
from helpers.torch_gradient_references import (
    run_joint_pose_factor_saved_state,
    run_statistical_transport_saved_state,
)

pytest.importorskip("torch", reason="optional Torch runtime is not installed")


@pytest.mark.parametrize("domain", ["linear", "log"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("execution", ["serial", "fused"])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_joint_pose_factors_and_outstanding_saved_states(
    domain: Literal["linear", "log"],
    shared: bool,
    execution: Literal["serial", "fused"],
    reverse_order: bool,
    warp_device: str,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Any,
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        run_joint_pose_factor_saved_state(
            warp_device, domain, shared, execution, reverse_order, monkeypatch, record_property
        )


@pytest.mark.parametrize("scoring", ["photon-count", "energy-kev"])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_expected_score_replicates_preserve_outstanding_input_states(
    scoring: Literal["photon-count", "energy-kev"],
    reverse_order: bool,
    warp_device: str,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Any,
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        run_statistical_transport_saved_state(
            warp_device, scoring, reverse_order, monkeypatch, record_property
        )
