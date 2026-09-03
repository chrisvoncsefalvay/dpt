"""CPU contracts for scaling work without changing the stochastic inverse problem."""

import json
import math
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from dpt.contracts import ContractError
from dpt.stochastic_recovery import StochasticPolicy
from dpt.transport.recovery_experiments import (
    main,
    scattering_configuration,
    scattering_policy,
    validate_scattering_configuration,
    validate_scattering_reference,
)


def test_default_policy_preserves_the_historical_sampling_design() -> None:
    expected = StochasticPolicy(
        proposal="quadratic",
        replicates=8,
        initial_batch=4096,
        maximum_batch=65536,
        final_validation_batch=65536,
        unique_history_budget=16_000_000,
        numerical_gradient_allowance=1e-12,
    )
    assert scattering_policy() == expected


def test_multiplier_scales_all_work_and_preserves_every_acceptance_parameter() -> None:
    original, scaled = asdict(scattering_policy()), asdict(scattering_policy(16))
    work = ("initial_batch", "maximum_batch", "final_validation_batch", "unique_history_budget")
    for name in original:
        assert scaled[name] == (16 * original[name] if name in work else original[name])
    config = scattering_configuration(16)
    assert config["workspace_max_histories"] == 1_048_576
    assert config["reference_batch"] * config["reference_replicates"] == 268_435_456
    assert config["observation"] == config["initial_amplitude"] == 0.08


@pytest.mark.parametrize("multiplier", [0, -1, 1.5, True, 32768])
def test_invalid_multiplier_is_rejected_before_device_preparation(multiplier: int) -> None:
    with pytest.raises(ContractError):
        scattering_policy(multiplier)


def reference(multiplier: int = 16) -> dict[str, Any]:
    batches = [0.69, 0.71] * 128
    se = statistics.stdev(batches) / math.sqrt(len(batches))
    return {
        "mean": statistics.mean(batches),
        "standard_error": se,
        "absolute_uncertainty": 7 * se,
        "replicates": batches,
        "batch": 65536 * multiplier,
        "histories_traced": 256 * 65536 * multiplier,
        "seed": 419003,
    }


def test_reference_statistics_and_sampling_budget_are_checked_independently() -> None:
    value = reference()
    validate_scattering_reference(value, 16)
    with pytest.raises(ContractError, match="sampling design"):
        validate_scattering_reference(value, 1)
    value["mean"] += 0.01
    with pytest.raises(ContractError, match="statistics"):
        validate_scattering_reference(value, 16)


def test_new_reference_configuration_cannot_change_the_physics() -> None:
    config = json.loads(json.dumps({"seed": 419003, **scattering_configuration(16)}))
    validate_scattering_configuration(config, 16)
    config["absorption_mm_inverse"][0] = 0.4
    with pytest.raises(ContractError, match="configuration differs"):
        validate_scattering_configuration(config, 16)


@pytest.mark.parametrize("multiplier", [1, 16])
def test_reference_requires_its_physical_configuration(multiplier: int) -> None:
    with pytest.raises(ContractError, match="configuration differs"):
        validate_scattering_configuration({"seed": 419003}, multiplier)


@pytest.mark.parametrize(
    "options",
    [
        ["--mode", "original", "--sampling-multiplier", "16"],
        ["--mode", "reference", "--repetitions", "1"],
        ["--mode", "stochastic", "--repetitions", "0"],
        ["--mode", "stochastic", "--repetitions", "33"],
        ["--mode", "stochastic", "--sampling-multiplier", "32768"],
        ["--mode", "stochastic", "--seed-base", str(2**64 - 1)],
    ],
)
def test_cli_rejects_ignored_or_unrepresentable_designs_before_execution(
    options: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(sys, "argv", ["repair.py", "--output", str(output), *options])
    with pytest.raises(SystemExit) as error:
        main(__file__)
    assert error.value.code == 2
    assert not output.exists()
