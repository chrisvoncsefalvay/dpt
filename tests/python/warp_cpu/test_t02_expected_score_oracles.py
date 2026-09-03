"""Independent native expected-score oracles supporting T02; no Torch execution.

Active coordinates are log voxel density, log material density and direct source
amplitude. Exact absorption integrals and finite binomial laws supply references;
native replay agreement alone is not treated as an independent correctness test.
"""

# Optional Warp annotations and NumPy runtime methods do not have complete stubs.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import pytest

from dpt.transport.derivatives import (
    TransportParameter,
    derivative_columns,
    prepare_transport_vjp,
    transport_vjp,
)
from dpt.transport.estimators import history_mean, prepare_estimators
from dpt.transport.forward import TransportWorkspace, prepare_transport, trace_histories
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.rng import HistoryBatch, SourceIdentity, require_independent
from dpt.transport.source import ParallelBeam, sample_parallel_beam
from dpt.transport.voxel_derivatives import prepare_voxel_density_vjp, voxel_density_vjp

wp: Any = importlib.import_module("warp")
type _LossKind = Literal["quadratic", "cubic", "reused-quadratic"]
_SOURCE = ParallelBeam((0.25, 0.25, -1.0), (0.0, 0.0), weight=1.9)
_PARAMETERS = (
    TransportParameter("log-material-density", 0),
    TransportParameter("log-material-density", 1),
    TransportParameter("source-amplitude"),
)


def _layered_transport(
    device: str,
    capacity: int,
    estimator: Literal["analogue", "continuous-absorption"],
    scoring: Literal["photon-count", "energy-kev"],
) -> tuple[TransportWorkspace, Any, np.ndarray]:
    ids = np.array([0, 1, 0, 1], dtype=np.int32)
    voxel_eta = np.log([0.5, 1.2, 0.7, 1.6])
    material_eta = np.log([1.3, 0.8])
    absorption = np.array([0.2, 0.35])
    voxel_density, material_density = np.exp(voxel_eta), np.exp(material_eta)
    spacing_z = 0.75
    spec = TransportSpec(
        MaterialGrid((-2.0, -2.0, 0.0), (4.0, 4.0, spacing_z), (4, 1, 1), 2),
        PlanarDetector((-1.5, -1.0), (1.0, 1.0), (3, 2), 4.0),
        80.0,
        "independent native T02 heterogeneous absorption integral",
        estimator=estimator,
        scoring=scoring,
    )
    workspace = prepare_transport(
        spec,
        material_ids=wp.array(ids, dtype=wp.int32, device=device),
        absorption=wp.array(absorption, dtype=wp.float64, device=device),
        scattering=wp.zeros(2, dtype=wp.float64, device=device),
        voxel_density=wp.array(voxel_density, dtype=wp.float64, device=device),
        max_histories=capacity,
        device=device,
    )
    depth = voxel_density * material_density[ids] * absorption[ids] * spacing_z
    density = wp.array(material_density, dtype=wp.float64, device=device)
    return workspace, density, depth


def _sample(
    workspace: TransportWorkspace, batch: HistoryBatch, source: ParallelBeam = _SOURCE
) -> tuple[HistoryBatch, tuple[Any, Any, Any]]:
    device = workspace.context.device
    position = wp.empty(batch.count, dtype=wp.vec3d, device=device)
    direction = wp.empty_like(position)
    weight = wp.empty(batch.count, dtype=wp.float64, device=device)
    token = sample_parallel_beam(
        source,
        batch=batch,
        workspace=workspace,
        out_position=position,
        out_direction=direction,
        out_weight=weight,
    )
    assert token == SourceIdentity(
        batch.seed, batch.first_history, batch.count, "parallel-rectangle-v1"
    )
    return replace(batch, source_identity=token), (position, direction, weight)


def _trace(
    workspace: TransportWorkspace,
    source_arrays: tuple[Any, Any, Any],
    density: Any,
    batch: HistoryBatch,
    amplitude: float,
) -> dict[str, Any]:
    outputs = {
        f"out_{name}": wp.empty(batch.count, dtype=dtype, device=workspace.context.device)
        for name, dtype in (
            ("pixel", wp.int32),
            ("score", wp.float64),
            ("energy", wp.float64),
            ("events", wp.int32),
            ("status", wp.int32),
        )
    }
    trace_histories(
        *source_arrays,
        density,
        batch=batch,
        workspace=workspace,
        source_amplitude=amplitude,
        **outputs,
    )
    assert np.all(outputs["out_status"].numpy() <= 1), "all histories must be complete"
    return outputs


@pytest.mark.parametrize("scoring", ["photon-count", "energy-kev"])
@pytest.mark.parametrize("amplitude", [0.0, 0.7])
@pytest.mark.parametrize("estimator", ["continuous-absorption", "analogue"])
def test_native_layer_log_density_and_source_vjps_have_analytic_normalization(
    scoring: Literal["photon-count", "energy-kev"],
    amplitude: float,
    estimator: Literal["analogue", "continuous-absorption"],
    warp_device: str,
) -> None:
    forward_count, derivative_count = (
        (7, 23)
        if estimator == "continuous-absorption"
        else (
            8191,
            32768,
        )
    )
    transport, density, depths = _layered_transport(
        warp_device, derivative_count, estimator, scoring
    )
    first, source_a = _sample(transport, HistoryBatch(90317, 2**40, forward_count))
    second, source_b = _sample(
        transport, HistoryBatch(90317, 2**40 + forward_count, derivative_count)
    )
    require_independent(first, second)
    outputs = _trace(transport, source_a, density, first, amplitude)
    image = wp.empty(6, dtype=wp.float64, device=warp_device)
    history_mean(
        outputs["out_pixel"],
        outputs["out_score"],
        outputs["out_status"],
        batch=first,
        workspace=prepare_estimators(transport),
        out_mean=image,
    )
    # The pencil hits column 1, row 1 of a (width=3, height=2) detector.
    hit_pixel, cotangent_value = 4, -3.0
    cotangent = wp.array(
        [2.0, -5.0, 7.0, 0.5, cotangent_value, 11.0], dtype=wp.float64, device=warp_device
    )
    voxel_gradient = wp.empty(4, dtype=wp.float64, device=warp_device)
    voxel_variance = wp.empty_like(voxel_gradient)
    result = voxel_density_vjp(
        *source_b,
        density,
        cotangent,
        batch=second,
        cotangent_batch=first,
        workspace=prepare_voxel_density_vjp(transport, slots_per_history=4),
        out_gradient=voxel_gradient,
        out_variance_of_mean=voxel_variance,
        variance_law="iid-original-histories",
        source_amplitude=amplitude,
    )
    material_gradient = wp.empty(3, dtype=wp.float64, device=warp_device)
    transport_vjp(
        *source_b,
        density,
        cotangent,
        batch=second,
        cotangent_batch=first,
        workspace=prepare_transport_vjp(transport, parameters=_PARAMETERS),
        out_gradient=material_gradient,
        source_amplitude=amplitude,
    )
    probability = math.exp(-float(depths.sum()))
    energy = 80.0 if scoring == "energy-kev" else 1.0
    base = _SOURCE.weight * energy
    expected_image = np.zeros(6)
    expected_image[hit_pixel] = amplitude * base * probability
    voxel_factors = -depths * amplitude * base * cotangent_value
    material_factors = np.array(
        [voxel_factors[::2].sum(), voxel_factors[1::2].sum(), base * cotangent_value]
    )
    # Two stochastic contrasts per analogue case, across four parameter cases,
    # share a predeclared 1e-6 family error bound using Bernoulli Hoeffding bands.
    alpha_per_contrast = 1e-6 / 8
    forward_radius = math.sqrt(math.log(2.0 / alpha_per_contrast) / (2 * forward_count))
    derivative_radius = math.sqrt(math.log(2.0 / alpha_per_contrast) / (2 * derivative_count))
    if estimator == "continuous-absorption":
        forward_radius = derivative_radius = 0.0
    assert np.all(
        np.abs(image.numpy() - expected_image) <= amplitude * base * forward_radius + 1e-12
    )
    assert np.all(
        np.abs(voxel_gradient.numpy() - voxel_factors * probability)
        <= np.abs(voxel_factors) * derivative_radius + 1e-12
    )
    assert np.all(
        np.abs(material_gradient.numpy() - material_factors * probability)
        <= np.abs(material_factors) * derivative_radius + 1e-12
    )
    assert result.batch == second and result.cotangent_batch == first and result.checked
    assert result.parameterization == "log-voxel-density"
    if estimator == "continuous-absorption":
        np.testing.assert_allclose(voxel_variance.numpy(), 0.0, atol=1e-25)
    else:
        # The source column recovers the derivative batch's observed survival
        # fraction even at zero amplitude; all log-density scores vanish there.
        observed = float(material_gradient.numpy()[2] / material_factors[2])
        expected_variance = voxel_factors**2 * observed * (1 - observed) / (derivative_count - 1)
        np.testing.assert_allclose(voxel_variance.numpy(), expected_variance, rtol=5e-12)
    if amplitude == 0.0:
        np.testing.assert_array_equal(image.numpy(), np.zeros(6))
        np.testing.assert_array_equal(voxel_gradient.numpy(), np.zeros(4))
        np.testing.assert_array_equal(material_gradient.numpy()[:2], np.zeros(2))
        assert material_gradient.numpy()[2] < 0.0


@dataclass(frozen=True)
class _BoundedLaw:
    mean: float
    variance: float
    maximum_deviation: float

    def radius(self, replicates: int, alpha: float) -> float:
        logarithm = math.log(2.0 / alpha)
        linear = self.maximum_deviation * logarithm / (3.0 * replicates)
        return linear + math.sqrt(2.0 * self.variance * logarithm / replicates + linear**2)


def _finite_binomial_law(
    forward_count: int,
    derivative_count: int,
    depth: float,
    kind: _LossKind,
) -> _BoundedLaw:
    # Independent enumeration, with no native samples or estimated variances.
    levels_a = [
        (k / forward_count, math.comb(forward_count, k) / 2**forward_count)
        for k in range(forward_count + 1)
    ]
    levels_b = [
        (k / derivative_count, math.comb(derivative_count, k) / 2**derivative_count)
        for k in range(derivative_count + 1)
    ]
    terms = []
    for a, probability_a in levels_a:
        for b, probability_b in [(a, 1.0)] if kind == "reused-quadratic" else levels_b:
            cotangent = a**2 if kind == "cubic" else a - 0.5
            terms.append((probability_a * probability_b, -depth * cotangent * b))
    mean = math.fsum(probability * value for probability, value in terms)
    variance = math.fsum(probability * (value - mean) ** 2 for probability, value in terms)
    return _BoundedLaw(mean, variance, max(abs(value - mean) for _, value in terms))


def test_native_complete_replicates_resolve_quadratic_reuse_and_cubic_sampling_bias(
    warp_device: str, request: pytest.FixtureRequest
) -> None:
    forward_count, derivative_count, replicates = 3, 5, 8192
    depth, probability = math.log(2.0), 0.5
    source = ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0))
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
        80.0,
        "independent native T02 finite-binomial expected-score law",
    )
    transport = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=warp_device),
        absorption=wp.array([depth], dtype=wp.float64, device=warp_device),
        scattering=wp.zeros(1, dtype=wp.float64, device=warp_device),
        voxel_density=wp.ones(1, dtype=wp.float64, device=warp_device),
        max_histories=replicates * derivative_count,
        device=warp_device,
    )
    density = wp.ones(1, dtype=wp.float64, device=warp_device)
    first, source_a = _sample(
        transport, HistoryBatch(119029, 2**43, replicates * forward_count), source
    )
    second, source_b = _sample(
        transport,
        HistoryBatch(119029, first.first_history + first.count, replicates * derivative_count),
        source,
    )
    require_independent(first, second)
    forward = _trace(transport, source_a, density, first, 1.0)
    backward = _trace(transport, source_b, density, second, 1.0)
    derivatives = wp.empty(2 * second.count, dtype=wp.float64, device=warp_device)
    derivative_columns(
        *source_b,
        density,
        parameters=(
            TransportParameter("log-material-density", 0),
            TransportParameter("source-amplitude"),
        ),
        batch=second,
        workspace=transport,
        out_pixel=backward["out_pixel"],
        out_derivatives=derivatives,
        out_status=backward["out_status"],
    )
    scores_a, scores_b = forward["out_score"].numpy(), backward["out_score"].numpy()
    assert np.all((scores_a == 0.0) | (scores_a == 1.0))
    assert np.all((scores_b == 0.0) | (scores_b == 1.0))
    columns = derivatives.numpy().reshape(2, second.count)
    np.testing.assert_array_equal(columns[0], -depth * scores_b)
    np.testing.assert_array_equal(columns[1], scores_b)
    a = scores_a.reshape(replicates, forward_count).mean(axis=1)
    db = columns[0].reshape(replicates, derivative_count).mean(axis=1)
    estimates: dict[_LossKind, np.ndarray] = {
        "quadratic": (a - probability) * db,
        "cubic": a**2 * db,
        "reused-quadratic": (a - probability) * (-depth * a),
    }
    quadratic_target = 0.0
    cubic_target = -depth * probability**3  # L(mu)=mu**3/3.
    cubic_bias = -depth * probability**2 * (1 - probability) / forward_count
    reuse_bias = -depth * probability * (1 - probability) / forward_count
    expected = {
        "quadratic": quadratic_target,
        "cubic": cubic_target + cubic_bias,
        "reused-quadratic": reuse_bias,
    }
    for kind, values in estimates.items():
        law = _finite_binomial_law(forward_count, derivative_count, depth, kind)
        assert law.mean == pytest.approx(expected[kind], abs=1e-16)
        radius = law.radius(replicates, 1e-6 / 3)
        measured = math.fsum(float(value) for value in values) / replicates
        assert abs(measured - law.mean) <= radius
        if kind == "cubic":
            assert abs(cubic_bias) > 2 * radius, "predeclared test must resolve nonlinear bias"
            assert abs(measured - cubic_target) > radius
        elif kind == "reused-quadratic":
            assert abs(reuse_bias) > 2 * radius, "predeclared test must resolve reuse bias"
            assert abs(measured - quadratic_target) > radius
        request.node.user_properties.extend(
            [
                (f"{kind}_expected", law.mean),
                (f"{kind}_measured", measured),
                (f"{kind}_bernstein_radius", radius),
            ]
        )
    request.node.user_properties.extend(
        [
            ("native_only_original_histories", first.count + second.count),
            ("forward_histories_per_replicate", forward_count),
            ("derivative_histories_per_replicate", derivative_count),
            ("complete_replicates", replicates),
            ("family_error_bound", 1e-6),
        ]
    )


def test_native_library_source_independence_uses_counter_addresses() -> None:
    first = HistoryBatch(43, 2**40, 7, "same-user-label").with_library_source(
        "parallel-rectangle-v1"
    )
    disjoint = HistoryBatch(43, 2**40 + 7, 23, "same-user-label").with_library_source(
        "parallel-rectangle-v1"
    )
    require_independent(first, disjoint)
    assert first.source_identity is not None and disjoint.source_identity is not None
    assert first.source_identity.independent_of(disjoint.source_identity)
    relabeled_overlap = HistoryBatch(43, 2**40 + 6, 1, "different-user-label").with_library_source(
        "point-cone-area-v1"
    )
    with pytest.raises(TransportError, match="disjoint"):
        require_independent(first, relabeled_overlap)
    # An untracked external random source cannot establish independence merely
    # by changing transport seed while retaining its random-source namespace.
    with pytest.raises(TransportError, match="disjoint"):
        require_independent(
            HistoryBatch(43, 0, 7, "external-source"), HistoryBatch(44, 0, 23, "external-source")
        )
