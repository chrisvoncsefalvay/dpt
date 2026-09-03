"""Bounded warmed replay comparison; measurements carry device and model scope."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from dpt.transport.derivatives import TransportParameter
from dpt.transport.forward import prepare_transport
from dpt.transport.inverse import prepare_transport_inverse
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam


def measure(device: str, parameters: int, count: int, repeats: int) -> dict[str, Any]:
    wp = importlib.import_module("warp")
    wp.config.kernel_cache_dir = str(Path.cwd() / ".warp-cache")
    wp.init()

    def array(values: list[Any], dtype: Any = None) -> Any:
        return wp.array(values, dtype=wp.float64 if dtype is None else dtype, device=device)

    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (parameters, 1, 1), parameters),
        PlanarDetector((-2.0, -2.0), (4.0, 4.0), (1, 1), parameters + 1.0),
        80.0,
        "bounded synthetic layered replay benchmark",
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
    results: dict[str, Any] = {}
    gradients = []
    for fused in (False, True):
        oracle = prepare_transport_inverse(
            workspace,
            source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
            parameters=tuple(
                TransportParameter("log-material-density", i) for i in range(parameters)
            ),
            observation=array([0.2]),
            pixel_weights=array([1.0]),
            base_density=(1.0,) * parameters,
            fused_derivatives=fused,
        )
        first, second = HistoryBatch(74291, 0, count), HistoryBatch(74291, count, count)
        coordinates = (0.0,) * parameters
        oracle.gradient_replicate(coordinates, first, second)
        samples = []
        before = oracle.histories_traced
        for _ in range(repeats):
            start = time.perf_counter()
            gradient = oracle.gradient_replicate(coordinates, first, second)
            samples.append(time.perf_counter() - start)
        gradients.append(gradient)
        walks = (oracle.histories_traced - before) // repeats
        results["fused" if fused else "scalar"] = {
            "median_seconds": statistics.median(samples),
            "seconds": samples,
            "walks_per_evaluation": walks,
            "walks_per_second": walks / statistics.median(samples),
            "owned_device_bytes": oracle.allocated_bytes,
        }
    results.update(
        device=str(wp.get_device(device)),
        warp_version=wp.__version__,
        parameters=parameters,
        histories_per_batch=count,
        repeats=repeats,
        fused_max_abs_gradient_error=max(abs(a - b) for a, b in zip(*gradients, strict=True)),
        median_speedup=results["scalar"]["median_seconds"] / results["fused"]["median_seconds"],
        expected_walk_ratio=(1 + parameters) / (1 + math.ceil(parameters / 4)),
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parameters", type=int, choices=range(1, 17), default=4)
    parser.add_argument("--histories", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.histories <= 2**20 or not 1 <= args.repeats <= 100:
        parser.error("bounded run requires 1..1048576 histories and 1..100 repeats")
    print(json.dumps(measure(args.device, args.parameters, args.histories, args.repeats), indent=2))


if __name__ == "__main__":
    main()
