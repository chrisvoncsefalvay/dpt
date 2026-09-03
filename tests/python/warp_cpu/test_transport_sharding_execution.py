"""Same-device shards preserve sampled paths, derivatives and original moments."""

from __future__ import annotations

import importlib
import math
from typing import Any, Literal

import numpy as np
import pytest

from dpt.transport.derivatives import TransportParameter, derivative_histories
from dpt.transport.estimators import (
    EstimatorWorkspace,
    history_mean,
    history_moments,
    prepare_estimators,
)
from dpt.transport.forward import TransportWorkspace, prepare_transport, trace_histories
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.sharding import ShardMoments, pool_shard_moments
from dpt.transport.source import ParallelBeam, sample_parallel_beam

wp: Any = importlib.import_module("warp")


def _run(
    batch: HistoryBatch,
    transport: TransportWorkspace,
    moments: EstimatorWorkspace,
    density: Any,
    source: ParallelBeam,
    device: str,
) -> dict[str, Any]:
    count = batch.count
    position = wp.empty(count, dtype=wp.vec3d, device=device)
    direction = wp.empty(count, dtype=wp.vec3d, device=device)
    weight = wp.empty(count, dtype=wp.float64, device=device)
    sampled = sample_parallel_beam(
        source,
        batch=batch,
        workspace=transport,
        out_position=position,
        out_direction=direction,
        out_weight=weight,
    )
    assert sampled == batch.source_identity
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
    trace_histories(
        position,
        direction,
        weight,
        density,
        batch=batch,
        workspace=transport,
        source_amplitude=2.5,
        **outputs,
    )
    snapshot = {name: value.numpy().copy() for name, value in outputs.items()}
    snapshot.update(
        position=position.numpy().copy(),
        direction=direction.numpy().copy(),
        weight=weight.numpy().copy(),
    )
    mean = wp.empty(transport.spec.detector.pixels, dtype=wp.float64, device=device)
    variance = wp.empty_like(mean)
    history_moments(
        outputs["out_pixel"],
        outputs["out_score"],
        outputs["out_status"],
        batch=batch,
        workspace=moments,
        out_mean=mean,
        out_variance_of_mean=variance,
    )
    snapshot["mean"] = mean.numpy().copy()
    snapshot["variance"] = variance.numpy().copy()
    derivative = wp.empty(count, dtype=wp.float64, device=device)
    for index, parameter in enumerate(
        (TransportParameter("log-material-density", 0), TransportParameter("source-amplitude"))
    ):
        derivative_histories(
            position,
            direction,
            weight,
            density,
            parameter=parameter,
            batch=batch,
            workspace=transport,
            out_pixel=outputs["out_pixel"],
            out_derivative=derivative,
            out_status=outputs["out_status"],
            source_amplitude=2.5,
        )
        assert np.array_equal(snapshot["out_pixel"], outputs["out_pixel"].numpy())
        assert np.array_equal(snapshot["out_status"], outputs["out_status"].numpy())
        snapshot[f"derivative_{index}"] = derivative.numpy().copy()
        history_mean(
            outputs["out_pixel"],
            derivative,
            outputs["out_status"],
            batch=batch,
            workspace=moments,
            out_mean=mean,
        )
        snapshot[f"gradient_mean_{index}"] = mean.numpy().copy()
    return snapshot


@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
def test_uneven_sequential_shards_preserve_paths_moments_and_material_source_gradients(
    warp_device: str,
    estimator: Literal["analogue", "continuous-absorption"],
) -> None:
    count = 1031
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (2, 1, 1), 2),
        PlanarDetector((-1.0, -1.0), (1.0, 1.0), (2, 2), 3.0),
        80.0,
        "analytic two-cell sharding fixture; original mathematical data, Apache-2.0",
        estimator=estimator,
    )
    transport = prepare_transport(
        spec,
        material_ids=wp.array([0, 1], dtype=wp.int32, device=warp_device),
        absorption=wp.array([0.2, 0.5], dtype=wp.float64, device=warp_device),
        scattering=wp.array([0.1, 0.05], dtype=wp.float64, device=warp_device),
        max_histories=count,
        device=warp_device,
    )
    density = wp.array([1.1, 0.9], dtype=wp.float64, device=warp_device)
    source = ParallelBeam((-1.25, -1.25, -1.0), (2.5, 2.5), weight=1.7)
    original = HistoryBatch(
        12739, 2**48 + 107, count, "same-source-on-all-devices"
    ).with_library_source("parallel-rectangle-v1")
    moments = prepare_estimators(transport)
    full = _run(original, transport, moments, density, source, warp_device)
    pieces: list[dict[str, Any]] = []
    records: list[ShardMoments] = []
    for rank in range(5):
        shard = original.shard(5, rank)
        piece = _run(shard.batch, transport, moments, density, source, warp_device)
        pieces.append(piece)
        window = slice(shard.offset, shard.offset + shard.batch.count)
        for name in (
            "position",
            "direction",
            "weight",
            "out_pixel",
            "out_score",
            "out_energy",
            "out_events",
            "out_status",
            "derivative_0",
            "derivative_1",
        ):
            assert np.array_equal(piece[name], full[name][window]), name
        records.append(
            ShardMoments(
                shard,
                tuple(float(value) for value in piece["mean"]),
                "canonical slab/density/source/4-pixel/2.5-amplitude snapshot",
                estimated_variance_of_mean=tuple(float(value) for value in piece["variance"]),
                status_counts=tuple(
                    int(value) for value in np.bincount(piece["out_status"], minlength=7)
                ),
            )
        )
    pooled = pool_shard_moments(records)
    assert pooled.batch == original
    assert sum(pooled.status_counts) == count
    assert np.count_nonzero(full["out_score"] == 0) > 0
    assert np.count_nonzero(full["out_events"] > 0) > 0
    # Nonnegative FP64 atomic terms admit gamma_n relative summation error;
    # the additional 32 operations cover scaling, pooling and host references.
    unit_roundoff = 2**-53
    tolerance = (count + 32) * unit_roundoff / (1 - (count + 32) * unit_roundoff)
    cotangent = (2.0, -3.0, 1.0, -5.0)
    for pixel in range(4):
        scores = tuple(
            float(score) if int(hit) == pixel else 0.0
            for hit, score in zip(full["out_pixel"], full["out_score"], strict=True)
        )
        expected_mean = math.fsum(scores) / count
        expected_variance = math.fsum((score - expected_mean) ** 2 for score in scores) / (
            count * (count - 1)
        )
        assert full["mean"][pixel] == pytest.approx(expected_mean, rel=tolerance)
        assert pooled.mean[pixel] == pytest.approx(expected_mean, rel=tolerance)
        assert full["variance"][pixel] == pytest.approx(expected_variance, rel=2 * tolerance)
        assert pooled.estimated_variance_of_mean[pixel] == pytest.approx(
            expected_variance, rel=2 * tolerance
        )
    for parameter in range(2):
        per_history = tuple(
            cotangent[int(pixel)] * float(derivative) if int(pixel) >= 0 else 0.0
            for pixel, derivative in zip(
                full["out_pixel"], full[f"derivative_{parameter}"], strict=True
            )
        )
        expected = math.fsum(per_history) / count
        full_gradient = math.fsum(
            cotangent[pixel] * float(full[f"gradient_mean_{parameter}"][pixel])
            for pixel in range(4)
        )
        weighted_gradient = math.fsum(
            record.shard.batch.count
            / count
            * math.fsum(
                cotangent[pixel] * float(piece[f"gradient_mean_{parameter}"][pixel])
                for pixel in range(4)
            )
            for record, piece in zip(records, pieces, strict=True)
        )
        absolute_tolerance = 4 * tolerance * math.fsum(map(abs, per_history)) / count
        assert full_gradient == pytest.approx(expected, rel=0, abs=absolute_tolerance)
        assert weighted_gradient == pytest.approx(expected, rel=0, abs=absolute_tolerance)
