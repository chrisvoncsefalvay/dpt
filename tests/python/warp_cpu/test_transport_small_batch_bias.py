"""Resolve nonlinear batch-reuse bias with real absorbing transport histories."""

# Optional NumPy and Warp stubs leave scalar conversions/array methods untyped.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
import importlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pytest

from dpt.transport.batched import prepare_transport_batched
from dpt.transport.derivatives import TransportParameter, derivative_histories
from dpt.transport.forward import TransportWorkspace, prepare_transport, trace_histories
from dpt.transport.inverse import prepare_transport_inverse
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam

wp: Any = importlib.import_module("warp")


def _absorbing_slab(device: str, count: int, depth: float) -> TransportWorkspace:
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
        80.0,
        "analytic Bernoulli survival law for small-batch gradient bias",
    )
    return prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=device),
        absorption=wp.array([depth], dtype=wp.float64, device=device),
        scattering=wp.array([0.0], dtype=wp.float64, device=device),
        device=device,
        max_histories=count,
    )


@dataclass(frozen=True)
class _BinomialLaw:
    mean: float
    variance: float
    maximum_deviation: float

    def radius(self, replicas: int, alpha: float) -> float:
        # Invert Bernstein's bound for the average of independent bounded
        # complete estimators, using exact reference moments rather than an
        # estimated standard error. Three bands share a fixed total error budget.
        logarithm = math.log(2.0 / alpha)
        linear = self.maximum_deviation * logarithm / (3.0 * replicas)
        return linear + math.sqrt(2.0 * self.variance * logarithm / replicas + linear**2)


def _binomial_law(
    count: int, depth: float, mode: Literal["independent", "symmetric", "reused"]
) -> _BinomialLaw:
    # At tau=log(2), each original history has X~Bernoulli(1/2), and
    # its log-density score derivative is -tau*X. The independent factor
    # means A,B therefore have exactly Binomial(N,1/2)/N distributions.
    levels = [(k / count, math.comb(count, k) / 2**count) for k in range(count + 1)]
    terms: list[tuple[float, float]] = []
    for a, probability_a in levels:
        other = [(a, 1.0)] if mode == "reused" else levels
        for b, probability_b in other:
            value = -depth * (a - 0.5) * b
            if mode == "symmetric":
                value = (value - depth * (b - 0.5) * a) / 2.0
            terms.append((probability_a * probability_b, value))
    mean = math.fsum(probability * value for probability, value in terms)
    variance = math.fsum(probability * (value - mean) ** 2 for probability, value in terms)
    return _BinomialLaw(mean, variance, max(abs(value - mean) for _, value in terms))


def _assert_mean_in_analytic_band(values: np.ndarray, expected: float, radius: float) -> float:
    mean = math.fsum(float(value) for value in values) / len(values)
    assert abs(mean - expected) <= radius, "outside independent analytic confidence band"
    return mean


def test_small_batch_independent_gradients_reject_resolved_reuse_bias(
    warp_device: str, record_testsuite_property: Callable[[str, object], None]
) -> None:
    histories_per_factor = 4
    replicas = 4096
    chunk = 64
    depth = math.log(2.0)
    seed = 97131
    first_history = 2**43
    total = 2 * replicas * histories_per_factor
    target_gradient = 0.0  # -tau*p*(p-y), with p=y=1/2.
    reuse_bias = -depth / (4 * histories_per_factor)
    alpha_per_band = 1e-6 / 3
    laws = {
        mode: _binomial_law(histories_per_factor, depth, mode)
        for mode in ("independent", "symmetric", "reused")
    }
    radii = {mode: law.radius(replicas, alpha_per_band) for mode, law in laws.items()}
    assert laws["independent"].mean == target_gradient
    assert laws["symmetric"].mean == target_gradient
    assert laws["reused"].mean == pytest.approx(reuse_bias, rel=2e-15)
    # The planned test resolves the mutant even when both means reach the
    # extremes of their fixed confidence bands. This guards test power.
    assert abs(reuse_bias) > radii["reused"] + max(radii["independent"], radii["symmetric"])

    workspace = _absorbing_slab(warp_device, histories_per_factor, depth)
    common: dict[str, Any] = dict(
        source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
        parameters=(TransportParameter("log-material-density", 0),),
        observation=wp.array([0.5], dtype=wp.float64, device=warp_device),
        pixel_weights=wp.array([1.0], dtype=wp.float64, device=warp_device),
        base_density=(1.0,),
        fused_derivatives=True,
    )
    one_sided = prepare_transport_batched(
        prepare_transport_inverse(workspace, **common), max_replicates=chunk
    )
    symmetric = prepare_transport_batched(
        prepare_transport_inverse(workspace, **common, symmetric_gradient=True),
        max_replicates=chunk,
    )
    addresses = tuple(
        (
            HistoryBatch(
                seed, first_history + 2 * row * histories_per_factor, histories_per_factor
            ),
            HistoryBatch(
                seed, first_history + (2 * row + 1) * histories_per_factor, histories_per_factor
            ),
        )
        for row in range(replicas)
    )
    with pytest.raises(TransportError, match="disjoint"):
        one_sided.gradient_replicate((0.0,), addresses[0][0], addresses[0][0])
    estimates: dict[str, list[float]] = {"independent": [], "symmetric": []}
    for start in range(0, replicas, chunk):
        pairs = addresses[start : start + chunk]
        estimates["independent"].extend(
            result[0] for result in one_sided.gradient_replicates((0.0,), pairs)
        )
        estimates["symmetric"].extend(
            result[0] for result in symmetric.gradient_replicates((0.0,), pairs)
        )

    # Public history APIs replay exactly those addresses in one contiguous
    # launch. This obtains the deliberately reused product from actual sampled
    # scores and derivatives without reading private oracle estimator state.
    sampled = _absorbing_slab(warp_device, total, depth)
    inputs = [
        wp.array(np.tile([0.0, 0.0, -1.0], (total, 1)), dtype=wp.vec3d, device=warp_device),
        wp.array(np.tile([0.0, 0.0, 1.0], (total, 1)), dtype=wp.vec3d, device=warp_device),
        wp.array(np.ones(total), dtype=wp.float64, device=warp_device),
        wp.array([1.0], dtype=wp.float64, device=warp_device),
    ]
    outputs = {
        f"out_{name}": wp.empty(total, dtype=dtype, device=warp_device)
        for name, dtype in (
            ("pixel", wp.int32),
            ("score", wp.float64),
            ("energy", wp.float64),
            ("events", wp.int32),
            ("status", wp.int32),
        )
    }
    batch = HistoryBatch(seed, first_history, total)
    trace_histories(*inputs, batch=batch, workspace=sampled, **outputs)
    scores = outputs["out_score"].numpy().reshape(replicas, 2, histories_per_factor)
    assert np.all((scores == 0.0) | (scores == 1.0))
    derivative = wp.empty(total, dtype=wp.float64, device=warp_device)
    derivative_histories(
        *inputs,
        parameter=TransportParameter("log-material-density", 0),
        batch=batch,
        workspace=sampled,
        out_pixel=outputs["out_pixel"],
        out_derivative=derivative,
        out_status=outputs["out_status"],
    )
    derivatives = derivative.numpy().reshape(replicas, 2, histories_per_factor)
    np.testing.assert_array_equal(derivatives, -depth * scores)
    a, b = scores.mean(axis=2).T
    da, db = derivatives.mean(axis=2).T
    independent = np.asarray(estimates["independent"])
    symmetric_values = np.asarray(estimates["symmetric"])
    np.testing.assert_allclose(independent, (a - 0.5) * db, rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(
        symmetric_values, ((a - 0.5) * db + (b - 0.5) * da) / 2.0, rtol=0.0, atol=2e-16
    )
    reused = (a - 0.5) * da
    measured = {
        "independent": _assert_mean_in_analytic_band(
            independent, target_gradient, radii["independent"]
        ),
        "symmetric": _assert_mean_in_analytic_band(
            symmetric_values, target_gradient, radii["symmetric"]
        ),
        "reused": _assert_mean_in_analytic_band(reused, reuse_bias, radii["reused"]),
    }
    # Apply the unchanged unbiased-estimator acceptance rule to the actual
    # same-batch product mutant. It must fail, with a resolved analytic margin.
    with pytest.raises(AssertionError, match="independent analytic confidence band"):
        _assert_mean_in_analytic_band(reused, target_gradient, radii["independent"])
    record_testsuite_property(
        f"transport_small_batch_bias_{warp_device}",
        json.dumps(
            {
                "histories_per_factor": histories_per_factor,
                "replicas": replicas,
                "unique_histories": total,
                "seed": seed,
                "first_history": first_history,
                "planned_joint_error_budget": 1e-6,
                "target_gradient": target_gradient,
                "analytic_reuse_bias": reuse_bias,
                "measured_means": measured,
                "bernstein_radii": radii,
            },
            sort_keys=True,
        ),
    )
