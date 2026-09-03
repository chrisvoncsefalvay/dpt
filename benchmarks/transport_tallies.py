"""Bounded grouped/plain readout and canonical fused transport comparisons."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dpt.transport.estimators import (
    history_mean,
    prepare_estimators,
    prepare_fused_mean,
    trace_mean,
)
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam, sample_parallel_beam


def timed(action: Callable[[], None], repeats: int, count: int) -> dict[str, Any]:
    action()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        action()
        samples.append(time.perf_counter() - start)
    median = statistics.median(samples)
    return {"seconds": samples, "median_seconds": median, "histories_per_second": count / median}


def measure(device: str, pixels: int, count: int, repeats: int) -> dict[str, Any]:
    wp = importlib.import_module("warp")
    np = importlib.import_module("numpy")
    wp.config.kernel_cache_dir = str(Path.cwd() / ".warp-cache")
    wp.init()
    width = math.isqrt(pixels)
    if width * width != pixels:
        raise ValueError("square detector required")

    def array(values: Any, dtype: Any = None) -> Any:
        return wp.array(values, dtype=wp.float64 if dtype is None else dtype, device=device)

    spec = TransportSpec(
        MaterialGrid((-32.0, -32.0, 0.0), (64.0, 64.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-32.0, -32.0), (64 / width, 64 / width), (width, width), 3.0),
        80.0,
        "bounded synthetic canonical transport benchmark",
        estimator="continuous-absorption",
    )
    workspace = prepare_transport(
        spec,
        material_ids=array([0], wp.int32),
        absorption=array([0.1]),
        scattering=array([0.2]),
        max_histories=count,
        device=device,
    )
    batch = HistoryBatch(83424, 0, count)
    out_mean = wp.empty(pixels, dtype=wp.float64, device=device)
    status = wp.zeros(count, dtype=wp.int32, device=device)
    indices = np.arange(count, dtype=np.int64)
    targets = (indices * 17 % pixels).astype(np.int32)
    targets[indices % 7 == 0] = -1
    scores = ((indices % 13 - 6) / 8).astype(np.float64)
    scores[targets < 0] = 0.0
    pixel, score = array(targets, wp.int32), array(scores)
    expected = np.bincount(np.maximum(targets, 0), weights=scores, minlength=pixels) / count
    isolated = {}

    def isolated_runner(moment: Any) -> Callable[[], None]:
        def run() -> None:
            history_mean(
                pixel,
                score,
                status,
                batch=batch,
                workspace=moment,
                out_mean=out_mean,
                validate=False,
            )
            workspace.check_status()

        return run

    for strategy in ("grouped", "plain"):
        moment = prepare_estimators(workspace, tally_strategy=strategy)
        record = timed(isolated_runner(moment), repeats, count)
        record["max_abs_reference_error"] = float(np.max(np.abs(out_mean.numpy() - expected)))
        isolated[strategy] = record
    position = wp.empty(count, dtype=wp.vec3d, device=device)
    direction = wp.empty_like(position)
    weight = wp.empty(count, dtype=wp.float64, device=device)
    sample_parallel_beam(
        ParallelBeam((-32.0, -32.0, -1.0), (64.0, 64.0)),
        batch=batch,
        workspace=workspace,
        out_position=position,
        out_direction=direction,
        out_weight=weight,
    )
    inputs = [position, direction, weight, array([1.0])]
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
    bounds = wp.full(pixels, 1.0, dtype=wp.float64, device=device)
    complete = {}
    reference = None

    def staged_runner(moment: Any) -> Callable[[], None]:
        def run() -> None:
            trace_histories(*inputs, batch=batch, workspace=workspace, validate=False, **outputs)
            history_mean(
                outputs["out_pixel"],
                outputs["out_score"],
                outputs["out_status"],
                batch=batch,
                workspace=moment,
                out_mean=out_mean,
                validate=False,
            )
            workspace.check_status()

        return run

    def fused_runner(fused: Any) -> Callable[[], None]:
        def run() -> None:
            trace_mean(*inputs, batch=batch, workspace=fused, out_mean=out_mean, validate=False)
            workspace.check_status()

        return run

    for strategy in ("grouped", "plain"):
        moment = prepare_estimators(workspace, tally_strategy=strategy)
        record = timed(staged_runner(moment), repeats, count)
        mean = out_mean.numpy().copy()
        if reference is None:
            reference = mean
        record.update(
            max_abs_reference_error=float(np.max(np.abs(mean - reference))),
            readout_scratch_bytes=moment.scratch_bytes + 28 * count,
            kernel_launches=4,
            original_histories_walked=count,
        )
        complete[f"staged_{strategy}"] = record
        fused = prepare_fused_mean(workspace, score_bound=bounds, tally_strategy=strategy)
        record = timed(fused_runner(fused), repeats, count)
        record.update(
            max_abs_reference_error=float(np.max(np.abs(out_mean.numpy() - reference))),
            readout_scratch_bytes=fused.scratch_bytes,
            kernel_launches=2,
            original_histories_walked=count,
        )
        complete[f"fused_{strategy}"] = record
    return {
        "device": str(wp.get_device(device)),
        "warp_version": wp.__version__,
        "pixels": pixels,
        "histories": count,
        "repeats": repeats,
        "isolated_signed_readout": isolated,
        "transport_with_resident_source": complete,
        "scope": "one warm-up, then complete wall-clock checkpoints; "
        "source draws/staging excluded from all timed paths; complete histories",
        "normalization": "fused paths use declared per-history photon-count upper bound 1; "
        "same canonical walk, IDs, material and source buffers",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--pixels", nargs="+", type=int, choices=(1, 64, 512**2), default=[1, 64, 512**2]
    )
    parser.add_argument("--histories", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.histories <= 2**20 or not 1 <= args.repeats <= 100:
        parser.error("bounded run requires 1..1048576 histories and 1..100 repeats")
    print(
        json.dumps(
            {"runs": [measure(args.device, p, args.histories, args.repeats) for p in args.pixels]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
