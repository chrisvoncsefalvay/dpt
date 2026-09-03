"""CPU-safe regularization descriptors and explicit precision/boundary contracts."""

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.regularization import RegularizationProxPolicy, RegularizationSpec


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tv_weight": -1.0},
        {"l1_weight": float("inf")},
        {"tv_norm": "frobenius"},
        {"precision": "float16"},
        {"boundary": "periodic"},
    ],
)
def test_invalid_regularization_descriptors(kwargs: dict[str, object]) -> None:
    with pytest.raises(ContractError):
        RegularizationSpec(**kwargs)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": 0},
        {"check_interval": False},
        {"absolute_gap": -1.0},
        {"relative_gap": float("nan")},
    ],
)
def test_invalid_proximal_stopping_policy(kwargs: dict[str, object]) -> None:
    with pytest.raises(ContractError):
        RegularizationProxPolicy(**kwargs)  # pyright: ignore[reportArgumentType]


def test_real_scalar_descriptors_are_canonical_binary64_values() -> None:
    value = np.float32(1e30)
    spec = RegularizationSpec(tv_weight=value, l1_weight=value)  # pyright: ignore[reportArgumentType]
    policy = RegularizationProxPolicy(absolute_gap=value, relative_gap=value)  # pyright: ignore[reportArgumentType]
    for canonical in (spec.tv_weight, spec.l1_weight, policy.absolute_gap, policy.relative_gap):
        assert type(canonical) is float
        assert canonical == float(value)
    assert spec.l1_weight * 1e30 == float(value) * 1e30
