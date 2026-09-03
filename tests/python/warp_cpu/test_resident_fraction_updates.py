"""Resident field updates preserve prepared bindings and checked retry semantics."""

import importlib
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_reconstruction import (
    MaterialReconstruction,
    MaterialReconstructionSettings,
    MaterialReconstructionView,
)
from dpt.spectral import SpectralSpec
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


def _solver(
    device,
    *,
    precision: Literal["float32", "float64"] = "float64",
    domain: Literal["fractions", "nonnegative"] = "fractions",
    graph=False,
):
    grid = GridSpec((2, 3, 4), (0.5, 0.7, 0.9), (-0.75, -0.7, -0.45))
    geometry = DetectorGeometry(
        (-20.0, 0.0, 0.0),
        (20.0, -0.3, -0.2),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.3, 0.4),
        (2, 3),
    )
    counts = np.ascontiguousarray(
        np.broadcast_to(np.array([9200, 8700])[:, None, None], (2, 2, 3)), dtype=np.float32
    )
    view = MaterialReconstructionView(
        geometry,
        RigidTransform(),
        counts,
        np.array([[10000, 2000], [2000, 10000]], dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
    )
    return MaterialReconstruction(
        grid=grid,
        views=(view, replace(view, pose=RigidTransform(translation_mm=(0.02, 0, 0)))),
        coefficients=np.array([[0.18, 0.04], [0.06, 0.27]], dtype=np.float32),
        spectral_spec=SpectralSpec(
            materials=2,
            energies=2,
            output_unit="counts",
            active_paths=True,
            shared_weights=True,
            shared_response=True,
            precision=precision,
        ),
        initial_fractions=np.full((2, *grid.shape), 0.15, dtype=np.float32),
        settings=MaterialReconstructionSettings(
            iterations=3,
            regularisation_mm_inverse=0.3,
            gradient_mapping_tolerance=0,
            relative_gradient_mapping_tolerance=0,
            execution="graph" if graph else "eager",
        ),
        samples_per_ray=48,
        field_domain=domain,
        device=device,
    )


def _candidate(solver):
    return np.linspace(0.07, 0.24, solver.size, dtype=np.float32).reshape(solver._shape)


def _upload(solver, values):
    with solver.context.scope():
        return wp.array(values.reshape(-1), dtype=wp.float32, device=solver.device)


class _PauseAfterAcceptedStepError(Exception):
    pass


def _prime_accepted_cache(solver):
    def pause(iteration, record):
        assert iteration == 1 and record["accepted"]
        raise _PauseAfterAcceptedStepError

    with pytest.raises(_PauseAfterAcceptedStepError):
        solver.solve(callback=pause, objective_id="resident-setter-fixed-fit-v1")
    assert solver._checkpoint_last_state is not None
    assert solver._prediction_state == "accepted"
    assert solver._accepted_forward_available


def _state(solver):
    return (
        solver._prediction_state,
        solver._accepted_forward_available,
        solver._checkpoint_last_state,
        solver.evaluations,
    )


def _pointers(solver):
    return (
        int(solver.fractions.ptr),
        int(solver.trial.ptr),
        int(solver.gradient.ptr),
        *((int(view.accepted.fields.ptr), int(view.trial.fields.ptr)) for view in solver.views),
        *((int(view.paths.ptr), int(view.trial_paths.ptr)) for view in solver.views),
    )


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_resident_update_matches_host_loss_gradient_and_preserves_graph_bindings(
    warp_device, precision
):
    graph = warp_device != "cpu"
    host = _solver(warp_device, precision=precision, graph=graph)
    resident = _solver(warp_device, precision=precision, graph=graph)
    _prime_accepted_cache(host)
    _prime_accepted_cache(resident)
    pointers, graphs = _pointers(resident), dict(resident._evaluation_graphs)
    candidate = _candidate(resident)
    device_candidate = _upload(resident, candidate)
    host.set_fractions(candidate)
    resident.set_fractions_resident(device_candidate)
    assert _state(host)[:3] == _state(resident)[:3] == ("uninitialised", False, None)
    assert _pointers(resident) == pointers
    assert resident._evaluation_graphs == graphs
    if graph:
        assert graphs
    np.testing.assert_array_equal(resident.fractions_numpy(), candidate)
    np.testing.assert_allclose(resident.evaluate(), host.evaluate(), rtol=2e-6, atol=1e-9)
    np.testing.assert_allclose(
        resident.gradient.numpy(), host.gradient.numpy(), rtol=3e-6, atol=2e-5
    )
    assert _pointers(resident) == pointers
    assert all(resident._evaluation_graphs[key] is value for key, value in graphs.items())


@pytest.mark.parametrize("bad", ["negative", "nan", "inf", "simplex", "above_allowance"])
def test_invalid_resident_candidate_preserves_fields_cache_and_allows_repaired_retry(
    warp_device, bad
):
    solver = _solver(warp_device)
    _prime_accepted_cache(solver)
    fields, state = solver.fractions_numpy().copy(), _state(solver)
    invalid = _candidate(solver)
    if bad in ("negative", "nan", "inf"):
        invalid.reshape(-1)[-1] = {"negative": -0.1, "nan": np.nan, "inf": np.inf}[bad]
    elif bad == "simplex":
        invalid[:, 0, 0, 0] = (0.7, 0.4)
    else:
        invalid[:, 0, 0, 0] = (1.0, 2.0**-23)
    candidate = _upload(solver, invalid)
    with pytest.raises(ContractError, match="finite nonnegative or fraction domain"):
        solver.set_fractions_resident(candidate)
    assert _state(solver) == state
    np.testing.assert_array_equal(solver.fractions_numpy(), fields)
    solver.check_status()
    valid = _candidate(solver)
    with solver.context.scope():
        candidate.assign(valid.reshape(-1))
    solver.set_fractions_resident(candidate)
    np.testing.assert_array_equal(solver.fractions_numpy(), valid)
    assert np.isfinite(solver.evaluate())
    assert np.isfinite(solver.gradient.numpy()).all()


@pytest.mark.parametrize("domain", ["fractions", "nonnegative"])
def test_resident_domain_limits_match_host_setter_exactly(warp_device, domain):
    host, resident = _solver(warp_device, domain=domain), _solver(warp_device, domain=domain)
    candidate = _candidate(host)
    candidate[:, 0, 0, 0] = (1.0, 2.0**-24) if domain == "fractions" else (2.0, 3.0)
    host.set_fractions(candidate)
    resident.set_fractions_resident(_upload(resident, candidate))
    np.testing.assert_array_equal(resident.fractions_numpy(), host.fractions_numpy())
    np.testing.assert_allclose(resident.evaluate(), host.evaluate(), rtol=2e-6)


@pytest.mark.parametrize("layout", ["dtype", "shape", "length", "strided"])
def test_resident_update_requires_flat_same_device_contiguous_fp32(warp_device, layout):
    solver = _solver(warp_device)
    solver.evaluate()
    fields, state = solver.fractions_numpy().copy(), _state(solver)
    with solver.context.scope():
        if layout == "dtype":
            candidate = wp.zeros(solver.size, dtype=wp.float64, device=solver.device)
        elif layout == "shape":
            candidate = wp.zeros(solver._shape, dtype=wp.float32, device=solver.device)
        elif layout == "length":
            candidate = wp.zeros(solver.size - 1, dtype=wp.float32, device=solver.device)
        else:
            candidate = wp.zeros(2 * solver.size, dtype=wp.float32, device=solver.device)[::2]
    with pytest.raises(ContractError):
        solver.set_fractions_resident(candidate)
    assert _state(solver) == state
    np.testing.assert_array_equal(solver.fractions_numpy(), fields)
    if warp_device != "cpu":
        with pytest.raises(ContractError, match="wrong device"):
            solver.set_fractions_resident(wp.zeros(solver.size, dtype=wp.float32, device="cpu"))
    with pytest.raises(ContractError, match="Warp array"):
        solver.set_fractions_resident(_candidate(solver).reshape(-1))


@pytest.mark.parametrize("trial", [False, True])
def test_resident_update_preserves_pending_accepted_or_trial_errors(warp_device, trial):
    solver = _solver(warp_device)
    solver.evaluate()
    before, state = solver.fractions_numpy().copy(), _state(solver)
    candidate = _upload(solver, _candidate(solver))
    projection = solver.views[-1].projection(trial).projection
    with solver.context.scope():
        projection._status.fill_(8)
    with pytest.raises(NumericalError):
        solver.set_fractions_resident(candidate)
    assert _state(solver) == state
    np.testing.assert_array_equal(solver.fractions_numpy(), before)
    with pytest.raises(NumericalError):
        projection.check_status()
    projection.clear_status()
    solver.set_fractions_resident(candidate)
    assert np.isfinite(solver.evaluate())


def test_resident_update_has_no_device_allocation_or_host_volume_readback(warp_device, monkeypatch):
    solver = _solver(warp_device)
    solver.evaluate()
    valid = _candidate(solver)
    candidate = _upload(solver, valid)
    pointers = _pointers(solver)
    copies = []
    original_copy = wp.copy

    def record_copy(destination, source, *args, **kwargs):
        copies.append((destination.device, int(destination.ptr), source.device, int(source.ptr)))
        return original_copy(destination, source, *args, **kwargs)

    def unexpected(*args, **kwargs):
        raise AssertionError("resident update allocated an array or exported a volume")

    with monkeypatch.context() as patch:
        patch.setattr(wp, "copy", record_copy)
        for name in ("empty", "empty_like", "zeros", "zeros_like", "full", "full_like"):
            patch.setattr(wp, name, unexpected)
        patch.setattr(wp.array, "numpy", unexpected)
        solver.set_fractions_resident(candidate)
        solver.set_fractions_resident(candidate)
    assert _pointers(solver) == pointers
    volume_copies = [item for item in copies if item[1] == int(solver.fractions.ptr)]
    assert len(volume_copies) == 2
    assert all(item[0] == item[2] == solver.device for item in volume_copies)
    assert all(item[3] == int(candidate.ptr) for item in volume_copies)
    np.testing.assert_array_equal(solver.fractions_numpy(), valid)
    with solver.context.scope():
        candidate.fill_(0)
    np.testing.assert_array_equal(solver.fractions_numpy(), valid)


def test_identical_owned_resident_buffer_is_validated_and_invalidates_state(warp_device):
    solver = _solver(warp_device)
    solver.evaluate()
    before, pointers = solver.fractions_numpy().copy(), _pointers(solver)
    solver.set_fractions_resident(solver.fractions)
    assert _state(solver)[:3] == ("uninitialised", False, None)
    assert _pointers(solver) == pointers
    np.testing.assert_array_equal(solver.fractions_numpy(), before)
    with solver.context.scope():
        solver.fractions.fill_(float("nan"))
    with pytest.raises(ContractError, match="finite nonnegative or fraction domain"):
        solver.set_fractions_resident(solver.fractions)
    solver.set_fractions(before)
    assert np.isfinite(solver.evaluate())


def test_resident_update_rejects_ambient_tape_before_changing_fields(warp_device):
    solver = _solver(warp_device)
    candidate = _upload(solver, _candidate(solver))
    before = solver.fractions_numpy().copy()
    with wp.Tape(), pytest.raises(ContractError, match="ambient tape"):
        solver.set_fractions_resident(candidate)
    np.testing.assert_array_equal(solver.fractions_numpy(), before)
