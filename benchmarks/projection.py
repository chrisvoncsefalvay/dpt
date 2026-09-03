"""Bounded warmed projection timing with independent homogeneous-box accuracy."""

from __future__ import annotations

# Dispatch before optional GPU imports so --ct ... --help works on a CPU host.
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks._entry import real_ct_route

    if real_ct_route("projection"):
        raise SystemExit(0)

import argparse
import importlib
import json
import math
import platform
import statistics
import time
from typing import Any, Literal

from dpt.contracts import integer
from dpt.experiments import describe_device
from dpt.geometry import RigidTransform
from dpt.phantoms import circular_cone_beam
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth
from dpt.volumes import GridSpec


def benchmark_projection(
    *,
    device: str,
    size: int,
    pixels_per_axis: int = 64,
    repeats: int = 5,
    sampling_precision: Literal["float64", "mixed_float32", "fast_float32"] = "float64",
    nonuniform: bool = False,
) -> dict[str, object]:
    size = integer(size, "size", minimum=8, maximum=512)
    side = integer(pixels_per_axis, "pixels_per_axis", minimum=1, maximum=256)
    repeats = integer(repeats, "repeats", minimum=1, maximum=20)
    np: Any = importlib.import_module("numpy")
    origin = -(size - 1) / 2
    grid = GridSpec((size, size, size), (1.0, 1.0, 1.0), (origin, origin, origin))
    geometry = circular_cone_beam(
        [0],
        source_radius_mm=3 * size,
        detector_radius_mm=3 * size,
        shape_rows_columns=(side, side),
        spacing_column_row_mm=(2 * size / side, 2 * size / side),
    )[0]
    spec = ProjectionSpec(
        2 * size,
        precision="float64",
        active_pose=False,
        active_volume=False,
        sampling_precision=sampling_precision,
    )
    workspace = prepare_projection(grid, geometry, spec, device=device)
    ctx, wp = workspace.context, workspace.context.wp
    attenuation = np.float32(0.02)
    host = None
    if nonuniform:
        # Keep coordinate scratch in one slab even for the 512^3 benchmark.
        axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        plane = 0.15 * np.sin(axis[None, :]) * np.cos(axis[:, None])
        host = np.empty(grid.shape, dtype=np.float32)
        for z in range(size):
            host[z] = attenuation * (1.0 + plane + 0.07 * np.cos(axis[z]))
    with ctx.scope():
        volume = (
            wp.full(grid.voxels, float(attenuation), dtype=wp.float32, device=ctx.device)
            if host is None
            else wp.array(host.reshape(-1), dtype=wp.float32, device=ctx.device)
        )
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=ctx.device)
        depth = wp.empty(geometry.pixels, dtype=wp.float64, device=ctx.device)
    project_optical_depth(volume, pose, out_L=depth, workspace=workspace)
    times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        project_optical_depth(volume, pose, out_L=depth, workspace=workspace, validate=False)
        workspace.check_status()
        times.append(time.perf_counter() - started)
    actual = depth.numpy()
    expected: list[float] = []
    for row in range(side):
        for column in range(side):
            endpoint = tuple(
                geometry.origin_mm[i]
                + column * geometry.spacing_mm[0] * geometry.u[i]
                + row * geometry.spacing_mm[1] * geometry.v[i]
                for i in range(3)
            )
            start = geometry.source_mm
            direction = tuple(endpoint[i] - start[i] for i in range(3))
            lo, hi = 0.0, 1.0
            for coordinate, step in zip(start, direction, strict=True):
                if step:
                    entry, exit = sorted(
                        ((-size / 2 - coordinate) / step, (size / 2 - coordinate) / step)
                    )
                    lo, hi = max(lo, entry), min(hi, exit)
                elif not -size / 2 <= coordinate <= size / 2:
                    lo, hi = 1.0, 0.0
            expected.append(float(attenuation) * math.hypot(*direction) * max(0.0, hi - lo))
    box_error = None if nonuniform else float(np.max(np.abs(actual - np.array(expected))))
    if box_error is not None and box_error > 2e-11 * max(1.0, float(np.max(np.abs(expected)))):
        raise RuntimeError(f"independent box projection failed: maximum absolute error {box_error}")
    strict_error = None
    if sampling_precision != "float64":
        reference_workspace = prepare_projection(
            grid,
            geometry,
            ProjectionSpec(2 * size, precision="float64", active_pose=False, active_volume=False),
            device=device,
            stream=ctx.stream,
        )
        with ctx.scope():
            reference = wp.empty(geometry.pixels, dtype=wp.float64, device=ctx.device)
        project_optical_depth(volume, pose, out_L=reference, workspace=reference_workspace)
        strict_error = float(np.max(np.abs(actual - reference.numpy())))
        if strict_error > 2e-6 * max(1.0, float(np.max(np.abs(reference.numpy())))):
            raise RuntimeError(
                f"mixed coordinate approximation exceeds its error budget: {strict_error}"
            )
    median = statistics.median(times)
    return {
        "operation": "smooth_sampled_field_midpoint_projection"
        if nonuniform
        else "homogeneous_box_midpoint_projection",
        "platform": platform.platform(),
        "device": describe_device(ctx),
        "grid_shape_zyx": list(grid.shape),
        "detector_shape_rows_columns": list(geometry.shape),
        "samples_per_ray": spec.samples_per_ray,
        "depth_precision": spec.precision,
        "sampling_precision": spec.sampling_precision,
        "repeats": repeats,
        "warmup_evaluations": 1,
        "seconds": times,
        "median_seconds": median,
        "rays_per_second": geometry.pixels / median,
        "forward_launches_per_measured_evaluation": 1,
        "completion_checkpoints_per_measured_evaluation": 1,
        "workspace_device_bytes": workspace.scratch_bytes,
        "workspace_host_bytes": workspace.host_bytes,
        "input_output_device_bytes": 4 * grid.voxels + 8 * geometry.pixels + 96,
        "maximum_absolute_error_vs_analytic_box": box_error,
        "maximum_absolute_error_vs_strict_sampling": strict_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--pixels-per-axis", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Use fixed-pose mixed coordinates and compare against strict sampling",
    )
    parser.add_argument(
        "--sampling-precision",
        choices=("float64", "mixed_float32", "fast_float32"),
        default="float64",
        help="Sampling arithmetic; both faster legacy modes retain FP64 ray sums",
    )
    parser.add_argument(
        "--nonuniform", action="store_true", help="Use a smooth nonuniform sampled field"
    )
    args = parser.parse_args()
    results = [
        benchmark_projection(
            device=args.device,
            size=n,
            pixels_per_axis=args.pixels_per_axis,
            repeats=args.repeats,
            sampling_precision="mixed_float32" if args.mixed else args.sampling_precision,
            nonuniform=args.nonuniform,
        )
        for n in args.sizes
    ]
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
