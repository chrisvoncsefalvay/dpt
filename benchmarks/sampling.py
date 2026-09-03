"""Bounded CUDA measurements for opt-in Compton, Gaussian pairs and sparse VJPs.

The driver measures actual production launches with warmed code, preallocated
buffers and CUDA events, and reports independent wall timings. Clinical-energy
transport uses an explicitly declared homogeneous reference cube. Projection
uses a 128³ field and a 512² detector by default. Timing comparisons carry the
complete policy and dimensions rather than extrapolating RNG operation counts.
"""

from __future__ import annotations

# Dispatch before optional GPU imports so --ct ... --help works on a CPU host.
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks._entry import real_ct_route

    if real_ct_route("sampling"):
        raise SystemExit(0)

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

import dpt
from dpt.detector import (
    GaussianReadNoiseExecution,
    ObservationIdentity,
    add_gaussian_read_noise,
    prepare_detector,
)
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.projection import ProjectionSpec, prepare_projection, projection_vjp
from dpt.transport import (
    HistoryBatch,
    MaterialGrid,
    ParallelBeam,
    PlanarDetector,
    TransportSpec,
)
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.transport.source import sample_parallel_beam
from dpt.volumes import GridSpec


def _times(call, stream, repeats):
    for _ in range(3):
        call()
    wp.synchronize_stream(stream)
    event_ms, wall_ms = [], []
    start = wp.Event(device=stream.device, enable_timing=True)
    end = wp.Event(device=stream.device, enable_timing=True)
    for _ in range(repeats):
        begin = time.perf_counter()
        stream.record_event(start)
        call()
        stream.record_event(end)
        event_ms.append(float(wp.get_event_elapsed_time(start, end)))
        wall_ms.append((time.perf_counter() - begin) * 1000)
    return {
        "cuda_ms": event_ms,
        "wall_ms": wall_ms,
        "median_cuda_ms": statistics.median(event_ms),
        "median_wall_ms": statistics.median(wall_ms),
    }


def measure(device, histories, side, repeats):
    wp.init()
    cuda = wp.get_device(device)
    if not cuda.is_cuda:
        raise ValueError("Sampling throughput measurements require a CUDA device")
    source = Path(dpt.__file__).resolve().parents[1]
    hashes = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source.rglob("*.py"))
    }
    results = {
        "device": str(cuda),
        "device_name": cuda.name,
        "warp": wp.__version__,
        "source_root": str(source),
        "source_hashes": hashes,
        "repeats": repeats,
        "histories": histories,
        "detector_side": side,
        "transport": {},
        "gaussian": {},
        "projection": {},
    }
    # A 64³ material map, 32 mm homogeneous cube and up to 128 real interactions
    # exercise the canonical walker. These fixed coefficients define the stated
    # reference workload and are not presented as measured material data.
    for energy in [80.0, 150.0, 1000.0, 6000.0]:
        for sampler in ["legacy-rejection", "packed-uniform-rejection"]:
            spec = TransportSpec(
                MaterialGrid((-16.0, -16.0, 0.0), (0.5, 0.5, 0.5), (64, 64, 64), 1),
                PlanarDetector((-32.0, -32.0), (64 / side, 64 / side), (side, side), 40.0),
                energy,
                "analytic:homogeneous-reference-cube",
                (0.001, 6000.0),
                "free-electron-compton",
                max_angle_trials=512,
                max_events=128,
                compton_sampler=sampler,
            )
            workspace = prepare_transport(
                spec,
                material_ids=wp.zeros(64**3, dtype=wp.int32, device=cuda),
                absorption=wp.array([0.005, 0.005], dtype=wp.float64, device=cuda),
                scattering=wp.array([0.05, 0.05], dtype=wp.float64, device=cuda),
                max_histories=histories,
                device=cuda,
            )
            position = wp.empty(histories, dtype=wp.vec3d, device=cuda)
            direction = wp.empty_like(position)
            weight = wp.empty(histories, dtype=wp.float64, device=cuda)
            density = wp.ones(1, dtype=wp.float64, device=cuda)
            batch = HistoryBatch(7139, 0, histories)
            source_identity = sample_parallel_beam(
                ParallelBeam((-15.0, -15.0, -1.0), (30.0, 30.0)),
                batch=batch,
                workspace=workspace,
                out_position=position,
                out_direction=direction,
                out_weight=weight,
            )
            batch = replace(batch, source_identity=source_identity)
            buffers = [
                wp.empty(histories, dtype=dtype, device=cuda)
                for dtype in (wp.int32, wp.float64, wp.float64, wp.int32, wp.int32)
            ]

            def call(
                position=position,
                direction=direction,
                weight=weight,
                density=density,
                batch=batch,
                workspace=workspace,
                buffers=buffers,
            ):
                trace_histories(
                    position,
                    direction,
                    weight,
                    density,
                    batch=batch,
                    workspace=workspace,
                    out_pixel=buffers[0],
                    out_score=buffers[1],
                    out_energy=buffers[2],
                    out_events=buffers[3],
                    out_status=buffers[4],
                    validate=False,
                )

            record = _times(call, workspace.context.stream, repeats)
            workspace.check_status()
            statuses, events = buffers[4].numpy(), buffers[3].numpy()
            record.update(
                mean_events=float(events.mean()),
                status_histogram=np.bincount(statuses, minlength=7).tolist(),
                histories_per_second=histories * 1000 / record["median_cuda_ms"],
            )
            results["transport"][f"{energy}_{sampler}"] = record
    pixels = max(side**2, 2**20)
    signal = wp.full(pixels, 3.0, dtype=wp.float32, device=cuda)
    output = wp.empty_like(signal)
    for algorithm in ["scalar-box-muller", "box-muller-pairs"]:
        workspace = prepare_detector(
            max_pixels=pixels,
            device=cuda,
            read_noise_execution=GaussianReadNoiseExecution(algorithm),
        )
        identity = ObservationIdentity(97191, 123, read_noise_algorithm=algorithm)

        def noise(identity=identity, workspace=workspace):
            add_gaussian_read_noise(
                signal,
                standard_deviation=2.7,
                out_signal=output,
                identity=identity,
                workspace=workspace,
                validate=False,
            )

        record = _times(noise, workspace.context.stream, repeats)
        workspace.check_status()
        sample = output.numpy().astype(np.float64)
        record.update(
            pixels=pixels,
            mean=float(sample.mean()),
            variance=float(sample.var()),
            pixels_per_second=pixels * 1000 / record["median_cuda_ms"],
        )
        results["gaussian"][algorithm] = record
    grid = GridSpec((128, 128, 128), (1.0, 1.0, 1.0), (-64.0, -64.0, -64.0))
    detector = DetectorGeometry(
        (-300.0, 0.0, 0.0),
        (300.0, -80.0, -80.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (160 / side, 160 / side),
        (side, side),
    )
    workspace = prepare_projection(
        grid,
        detector,
        ProjectionSpec(128, precision="float64", active_volume=True, active_pose=True),
        device=cuda,
    )
    field = wp.full(grid.voxels, 0.015, dtype=wp.float32, device=cuda)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=cuda)
    out_mu = wp.empty(grid.voxels, dtype=wp.float64, device=cuda)
    out_pose = wp.empty(6, dtype=wp.float64, device=cuda)
    rng = np.random.default_rng(21911)
    for fraction in [1.0, 0.1, 0.01, 0.0]:
        covector = np.zeros(detector.pixels)
        active = round(fraction * detector.pixels)
        selected = rng.choice(detector.pixels, active, replace=False)
        covector[selected] = rng.normal(size=active)
        seeds = wp.array(covector, dtype=wp.float64, device=cuda)

        def vjp(seeds=seeds):
            projection_vjp(
                field,
                pose,
                cotangent=seeds,
                out_mu=out_mu,
                out_pose_tangent=out_pose,
                workspace=workspace,
                volume_gradient_precision="float64",
                validate=False,
            )

        record = _times(vjp, workspace.context.stream, repeats)
        workspace.check_status()
        volume = out_mu.numpy()
        record.update(
            active_rays=active,
            total_rays=detector.pixels,
            samples=128,
            volume_gradient_l2=float(np.linalg.norm(volume)),
            pose_gradient=out_pose.numpy().tolist(),
        )
        if active == 0 and (np.any(volume != 0) or np.any(out_pose.numpy() != 0)):
            raise AssertionError("zero cotangent produced a nonzero gradient")
        results["projection"][str(fraction)] = record
    for energy in [80.0, 150.0, 1000.0, 6000.0]:
        a = results["transport"][f"{energy}_legacy-rejection"]["median_cuda_ms"]
        b = results["transport"][f"{energy}_packed-uniform-rejection"]["median_cuda_ms"]
        results["transport"][f"{energy}_legacy_over_packed"] = a / b
    results["gaussian"]["scalar_over_pairs"] = (
        results["gaussian"]["scalar-box-muller"]["median_cuda_ms"]
        / results["gaussian"]["box-muller-pairs"]["median_cuda_ms"]
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--histories", type=int, default=262144)
    parser.add_argument("--detector-side", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not 128 <= args.histories <= 2**20
        or not 16 <= args.detector_side <= 1024
        or not 1 <= args.repeats <= 20
    ):
        parser.error(
            "choose 128 to 1048576 histories, 16 to 1024 pixels per side, and 1 to 20 repeats"
        )
    report = measure(args.device, args.histories, args.detector_side, args.repeats)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "device": report["device_name"]}))


if __name__ == "__main__":
    main()
