"""Checked simulation against analytic laws and original-history moment oracles."""

from __future__ import annotations

import importlib
import math
from typing import Any, Literal

import numpy as np
import pytest
from scipy import stats

from dpt.contracts import ContractError
from dpt.geometry import DetectorGeometry
from dpt.transport import (
    HistoryBatch,
    HistoryStatus,
    IncompleteHistoryError,
    ParallelBeam,
    PointConeSource,
    SimulationResult,
    prepare_simulation,
    simulate,
)
from dpt.transport.estimators import history_moments, prepare_estimators, prepare_fused_mean
from dpt.transport.forward import TransportWorkspace, prepare_transport
from dpt.transport.model import (
    MaterialGrid,
    OrientedDetector,
    PlanarDetector,
    TransportError,
    TransportSpec,
)

wp: Any = importlib.import_module("warp")


def _slab(
    device: str,
    count: int,
    *,
    absorption: float = 0.7,
    estimator: Literal["analogue", "continuous-absorption"] = "analogue",
    layers: int = 1,
    max_crossings: int = 65536,
) -> tuple[TransportWorkspace, Any]:
    spec = TransportSpec(
        MaterialGrid((-2.0, -2.0, 0.0), (4.0, 4.0, 1.0), (layers, 1, 1), 1),
        PlanarDetector((-2.0, -2.0), (4.0, 4.0), (1, 1), float(layers + 1)),
        80.0,
        "analytic absorption slab; original mathematical fixture, Apache-2.0",
        estimator=estimator,
        max_crossings=max_crossings,
    )
    transport = prepare_transport(
        spec,
        material_ids=wp.zeros(layers, dtype=wp.int32, device=device),
        absorption=wp.array([absorption], dtype=wp.float64, device=device),
        scattering=wp.zeros(1, dtype=wp.float64, device=device),
        max_histories=count,
        device=device,
    )
    return transport, wp.array([1.0], dtype=wp.float64, device=device)


@pytest.mark.parametrize("strategy", ["plain", "grouped"])
def test_analogue_simulation_mean_and_variance_use_original_history_denominator(
    warp_device: str,
    strategy: Literal["plain", "grouped"],
) -> None:
    count = 8192
    transport, density = _slab(warp_device, count)
    workspace = prepare_simulation(transport, tally_strategy=strategy)
    result = simulate(
        density,
        source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0), weight=2.0),
        batch=HistoryBatch(93183, 2**40, count),
        workspace=workspace,
        source_amplitude=3.0,
    )
    assert isinstance(result, SimulationResult)
    survivors = result.status_histogram.count(HistoryStatus.ESCAPED)
    assert result.status_histogram.total == count
    assert result.status_histogram.count(HistoryStatus.ABSORBED) == count - survivors
    assert result.mean.numpy().tolist() == pytest.approx([6.0 * survivors / count], rel=2e-14)
    expected_variance = 36.0 * survivors * (count - survivors) / (count**2 * (count - 1))
    assert result.estimated_variance_of_mean is not None
    assert result.estimated_variance_of_mean.numpy().tolist() == pytest.approx(
        [expected_variance], rel=2e-13
    )
    # Acceptance uses the independently known Bernoulli law, never measured SE.
    physical_probability = math.exp(-0.7)
    grid = 2**32
    probability = math.floor(grid * physical_probability + 0.5) / grid
    alpha = 1e-6
    assert stats.binom.ppf(alpha / 2, count, probability) <= survivors
    assert survivors <= stats.binom.ppf(1 - alpha / 2, count, probability)
    assert result.image_shape == (1, 1)
    assert result.mean.dtype == wp.float64
    assert str(result.mean.device) == warp_device
    assert result.batch.source_identity is not None
    assert result.batch.source_identity.sampler == "parallel-rectangle-v1"
    assert result.sampling_law == "iid_original_histories"


@pytest.mark.parametrize("strategy", ["plain", "grouped"])
def test_continuous_absorption_simulation_matches_exact_constant_score(
    warp_device: str,
    strategy: Literal["plain", "grouped"],
) -> None:
    transport, density = _slab(warp_device, 32, estimator="continuous-absorption")
    result = simulate(
        density,
        source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
        batch=HistoryBatch(3813, 0, 32),
        workspace=prepare_simulation(transport, tally_strategy=strategy),
    )
    assert float(result.mean.numpy()[0]) == pytest.approx(math.exp(-0.7), rel=2e-14)
    assert result.estimated_variance_of_mean is not None
    assert float(result.estimated_variance_of_mean.numpy()[0]) == pytest.approx(0.0, abs=1e-30)
    assert result.status_histogram.count(HistoryStatus.ESCAPED) == 32
    assert result.estimator == "continuous-absorption"


def test_point_source_pixel_moments_are_marginal_and_sum_of_means_is_one(
    warp_device: str,
) -> None:
    count = 4096
    geometry = DetectorGeometry(
        source_mm=(0.0, 0.0, -1.0),
        origin_mm=(-0.5, -0.5, 3.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        spacing_mm=(1.0, 1.0),
        shape=(2, 2),
    )
    spec = TransportSpec(
        MaterialGrid((-2.0, -2.0, 0.0), (4.0, 4.0, 1.0), (1, 1, 1), 1),
        OrientedDetector(geometry),
        80.0,
        "analytic four-pixel unattenuated cone; Apache-2.0",
    )
    transport = prepare_transport(
        spec,
        material_ids=wp.zeros(1, dtype=wp.int32, device=warp_device),
        absorption=wp.zeros(1, dtype=wp.float64, device=warp_device),
        scattering=wp.zeros(1, dtype=wp.float64, device=warp_device),
        max_histories=count,
        device=warp_device,
    )
    result = simulate(
        wp.array([1.0], dtype=wp.float64, device=warp_device),
        source=PointConeSource(geometry),
        batch=HistoryBatch(72691, 0, count),
        workspace=prepare_simulation(transport),
    )
    means = result.mean.numpy()
    successes = np.rint(count * means).astype(np.int64)
    assert math.fsum(float(value) for value in means) == pytest.approx(1.0, rel=2e-14)
    expected = successes * (count - successes) / (count**2 * (count - 1))
    assert result.estimated_variance_of_mean is not None
    assert result.estimated_variance_of_mean.numpy() == pytest.approx(expected, rel=2e-13)
    # Every original history contributes exactly one detector photon, so the
    # summed signal has zero variance; summing marginal variance would be wrong.
    assert float(np.sum(result.estimated_variance_of_mean.numpy())) > 0.0
    assert result.status_histogram.count(HistoryStatus.ESCAPED) == count
    assert result.batch.source_identity is not None
    assert result.batch.source_identity.sampler == "point-cone-area-v1"
    assert result.image_shape == (2, 2)


def test_result_images_alias_workspace_but_histogram_and_batch_are_snapshots(
    warp_device: str,
) -> None:
    transport, density = _slab(warp_device, 32, absorption=0.0)
    workspace = prepare_simulation(transport)
    first = simulate(
        density,
        source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
        batch=HistoryBatch(1, 0, 32),
        workspace=workspace,
    )
    retained = first.mean.numpy().copy()
    second = simulate(
        density,
        source=ParallelBeam((10.0, 0.0, -1.0), (0.0, 0.0)),
        batch=HistoryBatch(1, 32, 8),
        workspace=workspace,
    )
    assert first.mean.ptr == second.mean.ptr == workspace.mean.ptr
    assert first.mean.numpy().tolist() == [0.0]
    assert retained.tolist() == [1.0]
    assert first.status_histogram.total == 32
    assert first.batch.count == 32
    assert second.status_histogram.total == 8
    assert second.status_histogram.count(HistoryStatus.ESCAPED) == 8
    assert second.estimated_variance_of_mean is not None
    assert second.estimated_variance_of_mean.numpy().tolist() == [0.0]


def test_single_history_has_no_within_batch_variance_and_is_rejected(warp_device: str) -> None:
    transport, density = _slab(warp_device, 32)
    workspace = prepare_simulation(transport)
    with pytest.raises(TransportError, match=r"2\.\.workspace\.max_histories"):
        simulate(
            density,
            source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
            batch=HistoryBatch(1, 0, 1),
            workspace=workspace,
        )


def test_guard_failure_returns_no_estimate_and_remains_failed(warp_device: str) -> None:
    transport, density = _slab(warp_device, 32, absorption=0.0, layers=2, max_crossings=1)
    workspace = prepare_simulation(transport)
    with pytest.raises(IncompleteHistoryError):
        simulate(
            density,
            source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
            batch=HistoryBatch(1, 0, 32),
            workspace=workspace,
        )
    with pytest.raises(IncompleteHistoryError):
        transport.check_status()


def test_supplied_source_law_cannot_be_mislabelled_as_library_iid(warp_device: str) -> None:
    transport, density = _slab(warp_device, 32)
    source: Any = {"sampling_law": "iid", "positions": "caller-supplied draws"}
    with pytest.raises(TransportError, match="library ParallelBeam or PointConeSource"):
        simulate(
            density,
            source=source,
            batch=HistoryBatch(1, 0, 32),
            workspace=prepare_simulation(transport),
        )


@pytest.mark.parametrize("buffer", ["mean", "moment_sum"])
def test_simulation_rejects_density_aliasing_its_output_or_moment_scratch(
    warp_device: str,
    buffer: str,
) -> None:
    transport, _ = _slab(warp_device, 32)
    workspace = prepare_simulation(transport)
    density = workspace.mean if buffer == "mean" else workspace.estimators._sum
    with pytest.raises(ContractError, match="overlap"):
        simulate(
            density,
            source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
            batch=HistoryBatch(1, 0, 32),
            workspace=workspace,
        )


def test_simulation_memory_limit_precedes_new_allocations(
    warp_device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, _ = _slab(warp_device, 32)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("memory budget rejection must precede allocation")

    monkeypatch.setattr(wp, "empty", forbidden)
    with pytest.raises(TransportError, match="exceeding max_bytes"):
        prepare_simulation(transport, max_bytes=1)


def test_warmed_simulation_allocates_no_warp_arrays_per_call(
    warp_device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, density = _slab(warp_device, 32, absorption=0.0)
    workspace = prepare_simulation(transport)
    source = ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0))
    simulate(density, source=source, batch=HistoryBatch(1, 0, 32), workspace=workspace)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("simulation must reuse prepared array storage")

    for factory in ("empty", "zeros", "empty_like"):
        monkeypatch.setattr(wp, factory, forbidden)
    result = simulate(density, source=source, batch=HistoryBatch(1, 32, 16), workspace=workspace)
    assert result.status_histogram.total == 16


@pytest.mark.parametrize("strategy", ["plain", "grouped"])
def test_sparse_moments_match_independent_signed_extreme_scale_oracle(
    warp_device: str,
    strategy: Literal["plain", "grouped"],
) -> None:
    values = [8e100, -4e100, 0.0, 6e100, -2e100, 0.0, 0.0, 0.0]
    transport, _ = _slab(warp_device, len(values))
    moment = prepare_estimators(transport, tally_strategy=strategy)
    mean = wp.empty(1, dtype=wp.float64, device=warp_device)
    variance = wp.empty_like(mean)
    history_moments(
        wp.array([0, 0, -1, 0, 0, -1, -1, -1], dtype=wp.int32, device=warp_device),
        wp.array(values, dtype=wp.float64, device=warp_device),
        wp.zeros(len(values), dtype=wp.int32, device=warp_device),
        batch=HistoryBatch(1, 0, len(values)),
        workspace=moment,
        out_mean=mean,
        out_variance_of_mean=variance,
    )
    expected_mean = math.fsum(values) / len(values)
    expected_variance = math.fsum((value - expected_mean) ** 2 for value in values) / (
        len(values) * (len(values) - 1)
    )
    assert float(mean.numpy()[0]) == pytest.approx(expected_mean, rel=2e-14)
    assert float(variance.numpy()[0]) == pytest.approx(expected_variance, rel=2e-13)


def test_plain_is_default_for_mean_moments_and_fused_readout(warp_device: str) -> None:
    transport, _ = _slab(warp_device, 32)
    assert prepare_estimators(transport).tally_strategy == "plain"
    bound = wp.array([1.0], dtype=wp.float64, device=warp_device)
    assert prepare_fused_mean(transport, score_bound=bound).tally_strategy == "plain"
    assert prepare_simulation(transport).estimators.tally_strategy == "plain"
