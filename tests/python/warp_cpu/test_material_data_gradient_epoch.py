"""Material data sums span all views and leave value-only trials unchanged."""

import importlib
from typing import Any, Literal

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError, TrialDomainError
from dpt.material_reconstruction import MaterialReconstruction
from helpers.material_inputs import _inputs

wp: Any = importlib.import_module("warp")


@pytest.fixture(autouse=True)
def portable_context():
    with cpu_test_context():
        yield


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_multiview_data_sum_narrows_once_and_value_only_trial_preserves_it(
    precision: Literal["float32", "float64"],
) -> None:
    inputs = _inputs(precision, regularisation=0.0)
    solver = MaterialReconstruction(**inputs)
    per_view = []
    for view in inputs["views"]:
        one = MaterialReconstruction(**{**inputs, "views": (view,)})
        one.evaluate()
        per_view.append(one._data_gradient.numpy().copy())
    solver.evaluate()
    expected = per_view[0] + per_view[1]
    assert solver._data_gradient.dtype == wp.float64
    assert solver.gradient.dtype == wp.float32
    np.testing.assert_array_equal(solver._data_gradient.numpy(), expected)
    np.testing.assert_array_equal(solver.gradient.numpy(), expected.astype(np.float32))
    for view in solver.views:
        for projection in (view.accepted, view.trial):
            assert projection.volume_gradient_precision == "float64"
            assert projection._float32_volume_increments is True
            assert projection.grad_fields.ptr == solver._data_gradient.ptr

    # A fresh epoch must erase old private scratch rather than add another copy.
    solver._data_gradient.fill_(99.0)
    solver.evaluate()
    np.testing.assert_array_equal(solver._data_gradient.numpy(), expected)
    incumbent = solver.gradient.numpy().copy()
    wp.copy(solver.trial, solver.fractions, stream=solver.stream)
    solver.evaluate(gradient=False, trial=True)
    np.testing.assert_array_equal(solver.gradient.numpy(), incumbent)
    np.testing.assert_array_equal(solver._data_gradient.numpy(), expected)
    solver.trial.fill_(2.0)
    with pytest.raises(TrialDomainError):
        solver.evaluate(gradient=False, trial=True)
    np.testing.assert_array_equal(solver.gradient.numpy(), incumbent)
    np.testing.assert_array_equal(solver._data_gradient.numpy(), expected)


@pytest.mark.parametrize("mutation", ["pointer", "precision", "destination"])
def test_prepared_data_gradient_binding_rejects_replacement(mutation: str) -> None:
    solver = MaterialReconstruction(**_inputs())
    solver.evaluate()
    accepted = solver.gradient.numpy().copy()
    if mutation == "pointer":
        solver._data_gradient = wp.empty(solver.size, dtype=wp.float64, device="cpu")
    elif mutation == "precision":
        solver.views[0].accepted.volume_gradient_precision = "float32"
    else:
        solver.views[0].accepted.grad_fields = wp.empty(solver.size, dtype=wp.float64, device="cpu")
    with pytest.raises(ContractError, match="data-gradient bindings must remain fixed"):
        solver.evaluate()
    np.testing.assert_array_equal(solver.gradient.numpy(), accepted)


def test_data_gradient_entry_rejects_same_pointer_strided_view_before_writes() -> None:
    solver = MaterialReconstruction(**_inputs())
    solver.evaluate()
    accepted = solver.gradient.numpy().copy()
    data = solver._data_gradient.numpy().copy()
    projection = solver.views[0].accepted
    original = projection.gradient_views[0]
    strided = wp.array(
        ptr=original.ptr,
        dtype=wp.float64,
        shape=original.shape,
        strides=(16,),
        capacity=solver._data_gradient.size * 8,
        device=solver.device,
    )
    assert strided.ptr == original.ptr and strided.shape == original.shape
    assert strided.dtype == original.dtype and strided.device == original.device
    assert strided.is_contiguous is False
    projection.gradient_views = (strided, *projection.gradient_views[1:])
    for gradient in (True, False):
        with pytest.raises(ContractError, match="contiguous"):
            solver.evaluate(gradient=gradient)
        np.testing.assert_array_equal(solver.gradient.numpy(), accepted)
        np.testing.assert_array_equal(solver._data_gradient.numpy(), data)


def test_data_gradient_entry_requires_exact_boolean_policy_before_writes() -> None:
    solver = MaterialReconstruction(**_inputs())
    solver.evaluate()
    accepted = solver.gradient.numpy().copy()
    data = solver._data_gradient.numpy().copy()
    solver.views[0].accepted._float32_volume_increments = 1
    for gradient in (True, False):
        with pytest.raises(ContractError, match="precision policy is invalid"):
            solver.evaluate(gradient=gradient)
        np.testing.assert_array_equal(solver.gradient.numpy(), accepted)
        np.testing.assert_array_equal(solver._data_gradient.numpy(), data)
