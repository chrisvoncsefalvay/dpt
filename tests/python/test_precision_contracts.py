"""Host policy descriptions preserve implemented storage/arithmetic distinctions."""

from __future__ import annotations

import pytest

from dpt.contracts import ContractError
from dpt.objectives import ObjectiveSpec
from dpt.precision import precision_contract
from dpt.projection import ProjectionSpec
from dpt.regularization import RegularizationSpec
from dpt.spectral import SpectralSpec
from dpt.transmission import TransmissionSpec
from dpt.transport import TransportSpec
from helpers.execution_models import transport_physics


@pytest.mark.parametrize("storage", ["float32", "float16", "bfloat16"])
def test_projection_native_field_storage_does_not_change_accumulation(storage: str) -> None:
    from typing import Any, cast

    selected = ProjectionSpec(
        8, active_pose=False, precision="float64", storage_precision=cast(Any, storage)
    )
    policy = precision_contract(selected)
    assert policy.storage("field") == storage
    assert policy.storage("depth") == "float64"
    assert policy.storage("pose") == "float64"
    assert policy.accumulation == "float64"
    assert policy.coordinates == "float64"


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_spectral_storage_and_exp_arithmetic_are_independent(precision: str) -> None:
    from typing import Any, cast

    for arithmetic in ("strict", "float32") if precision == "float32" else ("strict",):
        spec = SpectralSpec(2, 3, precision=cast(Any, precision), arithmetic=cast(Any, arithmetic))
        policy = precision_contract(spec)
        assert policy.storage("paths") == precision
        assert policy.storage("signal_cotangent") == precision
        assert policy.storage("coefficients") == "float32"
        assert policy.arithmetic == ("float64" if arithmetic == "strict" else "float32")
        assert policy.reduction == "float64"


def test_fixed_and_solver_policy_terms_name_actual_roles() -> None:
    transmission = precision_contract(TransmissionSpec())
    assert transmission.storage("depth") == "float32"
    assert transmission.arithmetic == "float64"
    objective = precision_contract(ObjectiveSpec(precision="float64", integer_counts=True))
    assert objective.storage("observation") == "float32"
    assert objective.storage("loss") == "float64"
    regularization = precision_contract(RegularizationSpec(precision="float32"))
    assert regularization.storage("field") == "float32"
    assert regularization.storage("primal_dual") == "float64"
    assert regularization.reduction == "float64"
    transport = precision_contract(TransportSpec.from_physics(transport_physics()))
    assert transport.storage("material_ids") == "int32"
    assert transport.storage("coefficients") == "float64"
    assert transport.storage("parameter_cotangent") == "float64"
    with pytest.raises(ContractError, match="no precision role"):
        transport.storage("invented")


def test_spectral_binary32_arithmetic_rejects_binary64_retained_storage() -> None:
    with pytest.raises(ContractError, match="requires float32"):
        SpectralSpec(2, 3, precision="float64", arithmetic="float32")
