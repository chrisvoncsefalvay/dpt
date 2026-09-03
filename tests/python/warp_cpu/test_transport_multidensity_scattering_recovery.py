"""Bounded two-density recovery with real scatter and independent held-out data."""

# Optional NumPy and Warp stubs leave scalar conversions/array methods untyped.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
import importlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from dpt.stochastic_recovery import StochasticPolicy, recover_expected_signal
from dpt.transport.batched import prepare_transport_batched
from dpt.transport.derivatives import TransportParameter, derivative_histories
from dpt.transport.forward import TransportWorkspace, prepare_transport, trace_histories
from dpt.transport.inverse import prepare_transport_inverse
from dpt.transport.model import HistoryStatus, MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.rng import HistoryBatch, require_independent
from dpt.transport.source import ParallelBeam, sample_parallel_beam

wp: Any = importlib.import_module("warp")

_ABSORPTION = (0.4, 0.7)
_SCATTERING = (0.08, 0.08)
_TRUTH = (0.2, -0.2)
_INITIAL = (-0.35, 0.35)
_REFERENCE_COUNT = 2**18
_FIRST_HISTORY = 2**42
_SOURCE = ParallelBeam((-1.0, 0.0, -1.0), (2.0, 0.0))


def _workspace(device: str, count: int) -> TransportWorkspace:
    # Each half of the rectangular source illuminates its own material column
    # and primary detector pixel. Scattering can cross columns or leave support.
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (1.0, 2.0, 1.0), (1, 1, 2), 2),
        PlanarDetector((-1.0, -2.0), (1.0, 4.0), (2, 1), 2.0),
        80.0,
        "synthetic adjacent material columns for two-density scattering recovery",
        estimator="continuous-absorption",
        max_events=128,
        max_crossings=1024,
    )
    return prepare_transport(
        spec,
        material_ids=wp.array([0, 1], dtype=wp.int32, device=device),
        absorption=wp.array(_ABSORPTION, dtype=wp.float64, device=device),
        scattering=wp.array(_SCATTERING, dtype=wp.float64, device=device),
        max_histories=count,
        device=device,
    )


@dataclass(frozen=True)
class _ForwardReadout:
    signal: np.ndarray
    primary_signal: np.ndarray
    scattering_fraction: float
    jacobian: np.ndarray | None


def _forward(
    device: str, parameters: tuple[float, float], seed: int, *, sensitivity: bool = False
) -> _ForwardReadout:
    count = _REFERENCE_COUNT
    workspace = _workspace(device, count)
    batch = HistoryBatch(seed, _FIRST_HISTORY, count)
    inputs = [
        wp.empty(count, dtype=wp.vec3d, device=device),
        wp.empty(count, dtype=wp.vec3d, device=device),
        wp.empty(count, dtype=wp.float64, device=device),
        wp.array([math.exp(value) for value in parameters], dtype=wp.float64, device=device),
    ]
    sample_parallel_beam(
        _SOURCE,
        batch=batch,
        workspace=workspace,
        out_position=inputs[0],
        out_direction=inputs[1],
        out_weight=inputs[2],
    )
    outputs = {
        f"out_{name}": wp.empty(count, dtype=dtype, device=device)
        for name, dtype in (
            ("pixel", wp.int32),
            ("score", wp.float64),
            ("energy", wp.float64),
            ("events", wp.int32),
            ("status", wp.int32),
        )
    }
    trace_histories(*inputs, batch=batch, workspace=workspace, **outputs)
    pixel = outputs["out_pixel"].numpy()
    score = outputs["out_score"].numpy()
    events = outputs["out_events"].numpy()
    assert np.all(outputs["out_status"].numpy() == int(HistoryStatus.ESCAPED))
    assert np.all(np.isfinite(score)) and np.all((score >= 0.0) & (score <= 1.0))
    detected = pixel >= 0
    primary = detected & (events == 0)
    signal = np.bincount(pixel[detected], weights=score[detected], minlength=2) / count
    primary_signal = np.bincount(pixel[primary], weights=score[primary], minlength=2) / count
    jacobian = None
    if sensitivity:
        columns = []
        derivative = wp.empty(count, dtype=wp.float64, device=device)
        for material in range(2):
            derivative_histories(
                *inputs,
                parameter=TransportParameter("log-material-density", material),
                batch=batch,
                workspace=workspace,
                out_pixel=outputs["out_pixel"],
                out_derivative=derivative,
                out_status=outputs["out_status"],
            )
            pixel = outputs["out_pixel"].numpy()
            detected = pixel >= 0
            values = derivative.numpy()
            columns.append(
                np.bincount(pixel[detected], weights=values[detected], minlength=2) / count
            )
        jacobian = np.asarray(columns).T
    return _ForwardReadout(signal, primary_signal, float(np.mean(events > 0)), jacobian)


def _primary_reference(parameters: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    means, variances = [], []
    for eta, absorption, scattering in zip(parameters, _ABSORPTION, _SCATTERING, strict=True):
        rho = math.exp(eta)
        # A primary history selects this column with probability 1/2, has no
        # scatter with probability exp(-rho*s), and scores exp(-rho*a).
        mean = 0.5 * math.exp(-rho * (absorption + scattering))
        second_moment = 0.5 * math.exp(-rho * (2 * absorption + scattering))
        means.append(mean)
        variances.append(second_moment - mean * mean)
    return np.asarray(means), np.asarray(variances)


def test_two_identifiable_material_densities_recover_with_nonzero_scattering(
    warp_device: str, record_testsuite_property: Callable[[str, object], None]
) -> None:
    started = perf_counter()
    target_seed, held_out_seed, evaluation_seed, recovery_seed = 273001, 273019, 273037, 38107
    reference_batches = tuple(
        HistoryBatch(seed, _FIRST_HISTORY, _REFERENCE_COUNT)
        for seed in (target_seed, held_out_seed, evaluation_seed)
    )
    require_independent(*reference_batches)
    target = _forward(warp_device, _TRUTH, target_seed)
    held_out = _forward(warp_device, _TRUTH, held_out_seed, sensitivity=True)
    assert target.scattering_fraction > 0.03 and held_out.scattering_fraction > 0.03

    # Independent analytic physics checks cover the primary contribution.
    # The full scattered target below is generated by the declared production
    # model; the experiment tests optimisation/replay, not cross-code physics.
    primary_mean, primary_variance = _primary_reference(_TRUTH)
    primary_radii = []
    logarithm = math.log(2 / (1e-6 / 4))
    for material, eta in enumerate(_TRUTH):
        maximum_score = math.exp(-math.exp(eta) * _ABSORPTION[material])
        bound = max(float(primary_mean[material]), maximum_score - float(primary_mean[material]))
        linear = bound * logarithm / (3 * _REFERENCE_COUNT)
        radius = linear + math.sqrt(
            2 * float(primary_variance[material]) * logarithm / _REFERENCE_COUNT + linear**2
        )
        primary_radii.append(radius)
        assert abs(float(target.primary_signal[material] - primary_mean[material])) <= radius
        assert abs(float(held_out.primary_signal[material] - primary_mean[material])) <= radius
    assert np.all(target.signal >= target.primary_signal)
    assert np.all(held_out.signal >= held_out.primary_signal)

    # Check local identifiability under the actual scattering law. The primary
    # Jacobian is diagonal and nonsingular; measured scatter corrections retain
    # negative, diagonally dominant sensitivity and a well-conditioned rank two.
    jacobian = held_out.jacobian
    assert jacobian is not None
    assert np.all(np.diag(jacobian) < -0.08)
    assert max(abs(float(jacobian[0, 1])), abs(float(jacobian[1, 0]))) < 0.03
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    assert float(singular_values[-1]) > 0.08
    assert float(singular_values[0] / singular_values[-1]) < 2.5

    policy = StochasticPolicy(
        iterations=12,
        replicates=8,
        initial_batch=1024,
        maximum_batch=8192,
        unique_history_budget=2_000_000,
        initial_radius=0.35,
        maximum_radius=0.6,
        proposal="quadratic",
        gradient_tolerance=1e-3,
    )
    workspace = _workspace(warp_device, policy.maximum_batch)
    oracle = prepare_transport_inverse(
        workspace,
        source=_SOURCE,
        parameters=tuple(
            TransportParameter("log-material-density", material) for material in range(2)
        ),
        observation=wp.array(target.signal, dtype=wp.float64, device=warp_device),
        pixel_weights=wp.array([1.0, 1.0], dtype=wp.float64, device=warp_device),
        base_density=(1.0, 1.0),
        local_model=True,
        fused_derivatives=True,
    )
    plural = prepare_transport_batched(oracle, max_replicates=policy.replicates)
    assert plural.supports_local_model and not plural.deterministic_sampling
    recovery_started = perf_counter()
    result = recover_expected_signal(plural, _INITIAL, policy=policy, seed=recovery_seed)
    recovery_seconds = perf_counter() - recovery_started
    assert result.sampling_classification == "stochastic"
    assert result.proposal_policy == "quadratic"
    assert result.uncertainty_policy == "heuristic_euclidean_marginal_se"
    assert len(result.steps) <= policy.iterations
    assert sum(step.accepted for step in result.steps) >= 2
    assert not result.domain_rejections
    assert result.unique_histories <= policy.unique_history_budget
    assert result.histories_traced is not None
    assert result.histories_traced == plural.histories_traced
    assert 0 < result.histories_traced <= 2 * policy.unique_history_budget
    assert all(math.isfinite(value) and 0.3 < math.exp(value) < 3.0 for value in result.parameters)

    reservations = result.reservations
    assert reservations and all(item.execution == "batched" for item in reservations)
    assert all(item.status == "complete" for item in reservations)
    assert all(item.completed_replicates == policy.replicates for item in reservations)
    assert sum(item.reserved_histories for item in reservations) == result.unique_histories
    assert (
        sum(
            item.reserved_histories * (2 if item.operation == "acceptance" else 1)
            for item in reservations
        )
        == result.histories_traced
    )
    next_history = 0
    for item in reservations:
        assert item.first_history == next_history
        for first, second in item.pairs:
            assert first.seed == second.seed == recovery_seed
            assert first.first_history == next_history
            assert second.first_history == first.first_history + first.count
            next_history += first.count + second.count
            for reference_batch in reference_batches:
                assert first.independent_of(reference_batch) and second.independent_of(
                    reference_batch
                )
        assert item.reserved_histories == item.required_histories
    assert next_history == result.unique_histories

    recovered = (result.parameters[0], result.parameters[1])
    before = _forward(warp_device, _INITIAL, evaluation_seed)
    after = _forward(warp_device, recovered, evaluation_seed)
    before_parameter_error = math.dist(_INITIAL, _TRUTH)
    after_parameter_error = math.dist(recovered, _TRUTH)
    before_signal_error = float(np.linalg.norm(before.signal - held_out.signal))
    after_signal_error = float(np.linalg.norm(after.signal - held_out.signal))
    assert after_parameter_error < 0.1 * before_parameter_error
    assert all(abs(actual - truth) < 0.06 for actual, truth in zip(recovered, _TRUTH, strict=True))
    assert after_signal_error < 0.1 * before_signal_error
    record_testsuite_property(
        f"transport_multidensity_scattering_recovery_{warp_device}",
        json.dumps(
            {
                "truth": _TRUTH,
                "initial": _INITIAL,
                "recovered": recovered,
                "absorption_mm_inverse": _ABSORPTION,
                "scattering_mm_inverse": _SCATTERING,
                "target_seed": target_seed,
                "held_out_seed": held_out_seed,
                "evaluation_seed": evaluation_seed,
                "recovery_seed": recovery_seed,
                "reference_histories_per_readout": _REFERENCE_COUNT,
                "target_signal": target.signal.tolist(),
                "held_out_truth_signal": held_out.signal.tolist(),
                "held_out_initial_signal": before.signal.tolist(),
                "held_out_recovered_signal": after.signal.tolist(),
                "primary_analytic_signal": primary_mean.tolist(),
                "primary_analytic_bernstein_radii": primary_radii,
                "primary_joint_error_budget": 1e-6,
                "scatter_fraction": held_out.scattering_fraction,
                "held_out_jacobian": jacobian.tolist(),
                "sensitivity_singular_values": singular_values.tolist(),
                "parameter_error_before": before_parameter_error,
                "parameter_error_after": after_parameter_error,
                "signal_error_before": before_signal_error,
                "signal_error_after": after_signal_error,
                "unique_recovery_histories": result.unique_histories,
                "traced_recovery_histories": result.histories_traced,
                "unique_history_budget": policy.unique_history_budget,
                "iteration_budget": policy.iterations,
                "actual_iterations": len(result.steps),
                "accepted_steps": sum(step.accepted for step in result.steps),
                "stop_reason": result.reason,
                "termination_detail": result.termination_detail,
                "final_validation_outcome": result.final_validation_outcome,
                "uncertainty_policy": result.uncertainty_policy,
                "interpretation": (
                    "bounded optimisation/replay validation; "
                    "no stationarity or external scattered-physics claim"
                ),
                "recovery_seconds": recovery_seconds,
                "total_seconds": perf_counter() - started,
            },
            sort_keys=True,
        ),
    )
