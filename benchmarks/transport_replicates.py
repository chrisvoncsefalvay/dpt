"""Bounded replica-axis comparison against the same four-column scalar replay."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from dpt.transport.batched import prepare_transport_batched
from dpt.transport.derivatives import TransportParameter
from dpt.transport.forward import prepare_transport
from dpt.transport.inverse import prepare_transport_inverse
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam


def measure(
    device: str,
    parameters: int,
    replicas: int,
    count: int,
    repeats: int,
    *,
    local_model: bool = False,
) -> dict[str, Any]:
    wp = importlib.import_module("warp")
    wp.config.kernel_cache_dir = str(Path.cwd() / ".warp-cache")
    wp.init()

    def array(values: list[Any], dtype: Any = None) -> Any:
        return wp.array(values, dtype=wp.float64 if dtype is None else dtype, device=device)

    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (parameters, 1, 1), parameters),
        PlanarDetector((-2.0, -2.0), (4.0, 4.0), (1, 1), parameters + 1.0),
        80.0,
        "bounded layered replicate-axis benchmark",
        estimator="continuous-absorption",
    )
    workspace = prepare_transport(
        spec,
        material_ids=array(list(range(parameters)), wp.int32),
        absorption=array([0.1] * parameters),
        scattering=array([0.2] * parameters),
        max_histories=count,
        device=device,
    )
    oracle = prepare_transport_inverse(
        workspace,
        source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
        parameters=tuple(TransportParameter("log-material-density", i) for i in range(parameters)),
        observation=array([0.2]),
        pixel_weights=array([1.0]),
        base_density=(1.0,) * parameters,
        fused_derivatives=True,
        local_model=local_model,
    )
    batched = prepare_transport_batched(oracle, max_replicates=replicas)
    coordinates = (0.0,) * parameters
    pairs = tuple(
        (
            HistoryBatch(74291, row * 2 * count, count),
            HistoryBatch(74291, (row * 2 + 1) * count, count),
        )
        for row in range(replicas)
    )
    results: dict[str, Any] = {}
    gradients = []
    for label in ("scalar_loop", "replica_axis"):
        owner = oracle if label == "scalar_loop" else batched

        def evaluate(current: str = label) -> tuple[Any, ...]:
            if current == "scalar_loop":
                method = oracle.model_replicate if local_model else oracle.gradient_replicate
                return tuple(method(coordinates, a, b) for a, b in pairs)
            if local_model:
                return batched.model_replicates(coordinates, pairs)
            return batched.gradient_replicates(coordinates, pairs)

        evaluate()
        samples = []
        before = owner.histories_traced
        before_download = owner.scalar_download_bytes
        before_upload = owner.parameter_upload_bytes
        for _ in range(repeats):
            start = time.perf_counter()
            gradient = evaluate()
            samples.append(time.perf_counter() - start)
        gradients.append(gradient)
        walks = (owner.histories_traced - before) // repeats
        median = statistics.median(samples)
        results[label] = {
            "host_checkpoint_copies_per_evaluation": (
                replicas * (2 if local_model else 1)
                if label == "scalar_loop" or (local_model and parameters > 8)
                else 1
            ),
            "median_seconds": median,
            "seconds": samples,
            "walks_per_evaluation": walks,
            "walks_per_second": walks / median,
            "checkpoint_download_bytes_per_evaluation": (
                owner.scalar_download_bytes - before_download
            )
            // repeats,
            "metadata_upload_bytes_per_evaluation": (owner.parameter_upload_bytes - before_upload)
            // repeats,
            "owned_device_bytes": owner.allocated_bytes,
        }
    gradient_results = (
        [tuple(row[0] for row in evaluation) for evaluation in gradients]
        if local_model
        else gradients
    )
    results.update(
        device=str(wp.get_device(device)),
        warp_version=wp.__version__,
        parameters=parameters,
        replicas=replicas,
        histories_per_batch=count,
        repeats=repeats,
        max_abs_gradient_error=max(
            abs(a - b)
            for first, second in zip(*gradient_results, strict=True)
            for a, b in zip(first, second, strict=True)
        ),
        median_speedup=results["scalar_loop"]["median_seconds"]
        / results["replica_axis"]["median_seconds"],
        scope="source draws, counter replay, tally, reductions and host gradient checkpoint",
        precision="binary64 scaled atomic means; reduction order may differ",
        replay="same canonical walk, same fixed-size independent identities, four-column scores",
        batched_host_staging_bytes=batched.host_staging_bytes,
        checkpoint_kind="gradient_and_psd_proposal_metric" if local_model else "gradient",
        replica_axis_active=not local_model or parameters <= 8,
    )
    if local_model:
        results["max_abs_metric_error"] = max(
            abs(a - b)
            for first, second in zip(*gradients, strict=True)
            for row_a, row_b in zip(first[1], second[1], strict=True)
            for a, b in zip(row_a, row_b, strict=True)
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parameters", type=int, choices=range(1, 17), default=4)
    parser.add_argument("--replicas", type=int, choices=range(1, 17), default=4)
    parser.add_argument("--histories", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--local-model", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.histories <= 65536 or not 1 <= args.repeats <= 100:
        parser.error("bounded run requires 1..65536 histories and 1..100 repeats")
    if args.local_model and args.parameters > 8:
        parser.error("bounded local-model comparison supports at most eight parameters")
    print(
        json.dumps(
            measure(
                args.device,
                args.parameters,
                args.replicas,
                args.histories,
                args.repeats,
                local_model=args.local_model,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
