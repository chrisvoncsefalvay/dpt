"""Independent rays and directional derivatives on the prepared real chest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.common import digest_sources, load_runtime, utc
from benchmarks.fullscan_inputs import load_prepared, sha256


def reference_ray(values, grid, panel, pose, row, column, samples):
    """CPU finite-segment slab clip and explicit eight-neighbour interpolation."""
    source = np.asarray(panel.source_mm)
    target = (
        np.asarray(panel.origin_mm)
        + column * panel.spacing_mm[0] * np.asarray(panel.u)
        + row * panel.spacing_mm[1] * np.asarray(panel.v)
    )
    rotation = np.asarray(pose.rotation).reshape(3, 3)
    axes = np.asarray(grid.orientation).reshape(3, 3)
    positions = (np.stack((source, target)) - pose.translation_mm) @ rotation
    positions = ((positions - grid.origin_mm) @ axes) / grid.spacing_mm
    start, delta = positions[0], positions[1] - positions[0]
    enter, leave = 0.0, 1.0
    shape = np.asarray(values.shape[::-1])
    for axis in range(3):
        low, high = -0.5, shape[axis] - 0.5
        if delta[axis] == 0:
            if not low <= start[axis] <= high:
                return 0.0
            continue
        ends = sorted(((low - start[axis]) / delta[axis], (high - start[axis]) / delta[axis]))
        enter, leave = max(enter, ends[0]), min(leave, ends[1])
    if leave <= enter:
        return 0.0
    t = enter + (np.arange(samples) + 0.5) * (leave - enter) / samples
    points = np.clip(start + t[:, None] * delta, 0, shape - 1)
    lower = np.floor(points).astype(int)
    upper = np.minimum(lower + 1, shape - 1)
    fraction = points - lower
    sampled = np.zeros(samples, dtype=np.float64)
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                bits = np.asarray((x, y, z))
                indices = np.where(bits, upper, lower)
                weights = np.prod(np.where(bits, fraction, 1 - fraction), axis=1)
                sampled += weights * values[indices[:, 2], indices[:, 1], indices[:, 0]]
    return float(sampled.mean() * np.linalg.norm(target - source) * (leave - enter))


def discrepancy(actual, expected, rtol, atol):
    error = abs(actual - expected)
    limit = atol + rtol * abs(actual)
    return dict(
        analytic=actual,
        finite_difference=expected,
        absolute_error=error,
        allowed_absolute_error=limit,
        passed=error <= limit,
    )


def run(args):
    started_at = utc()
    runtime = load_runtime()
    torch = runtime.torch
    import dpt

    if not Path(dpt.__file__).resolve().is_relative_to(args.source.resolve() / "python"):
        raise RuntimeError("guard imported DPT from a different source checkout")
    from dpt.geometry import BatchedGeometry, RigidTransform, compose_pose
    from dpt.torch import Projector

    frozen = json.loads(args.contract.read_text())
    guard = json.loads(args.guard_contract.read_text())
    if "input_contract" in guard:
        guard = guard["input_contract"]["guard"]
    prepared = load_prepared(args.data)
    if sha256(args.data / "metadata.json") != frozen["input_metadata_sha256"]:
        raise ValueError("guard volume metadata differs from frozen contract")
    seed_metadata = json.loads(args.cotangent.with_suffix(".json").read_text())
    if seed_metadata["contract_sha256"] != sha256(args.contract):
        raise ValueError("guard cotangent belongs to another contract")
    if sha256(args.cotangent) != seed_metadata["sha256"]:
        raise ValueError("guard cotangent checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    indices = guard["views"]
    geometry = BatchedGeometry(tuple(prepared.geometry.views[i] for i in indices))
    anchor = compose_pose(RigidTransform(), tuple(guard["anchor_tangent"]))
    field_host = np.asarray(prepared.volume.values)
    field = torch.tensor(field_host, device="cuda", requires_grad=True)
    tangent = torch.zeros(6, dtype=torch.float64, device="cuda", requires_grad=True)
    seed_host = np.asarray(np.load(args.cotangent, mmap_mode="r")[indices], dtype=np.float64)
    seed_host /= np.sum(np.abs(seed_host))
    precision = torch.float64 if args.precision == "fp64" else torch.float32
    seed = torch.tensor(seed_host, device="cuda", dtype=precision)
    projector = Projector(
        grid=prepared.volume.grid,
        geometry=geometry,
        anchor=anchor,
        precision=args.precision,
        samples_per_ray=1024,
    )
    image = projector(field, tangent)
    volume_gradient, pose_gradient = torch.autograd.grad(image, (field, tangent), seed)

    def objective(output):
        return float((output.double() * seed.double()).sum().item())

    analytic = float(
        (
            pose_gradient
            * torch.tensor(guard["pose_direction"], device="cuda", dtype=torch.float64)
        ).sum()
    )
    limits = frozen["quality"]["pose_fd"][args.precision]
    pose_checks = []
    direction = torch.tensor(guard["pose_direction"], device="cuda", dtype=torch.float64)
    with torch.no_grad():
        for step in [limits["selected_step"], *limits["diagnostic_steps"]]:
            plus = objective(projector(field, step * direction))
            minus = objective(projector(field, -step * direction))
            check = discrepancy(
                analytic,
                (plus - minus) / (2 * step),
                limits["relative_tolerance"],
                limits["absolute_tolerance"],
            )
            check.update(step=step, acceptance_step=step == limits["selected_step"])
            pose_checks.append(check)
    # Report the realised FP32 chord: input rounding is distinct from the
    # derivative of this linear projector with respect to stored field values.
    z, y, x = np.ogrid[:256, :256, :256]
    perturbation = field_host.astype(np.float64) * 0.01 + (field_host > 0) * 1e-5 * (
        1 + 0.2 * np.cos(x / 13) + 0.2 * np.cos(y / 17) + 0.2 * np.cos(z / 19)
    )
    step = 0.01
    if guard.get("volume_direction_domain_fraction") != 0.5:
        raise ValueError("guard must declare its nonnegative-domain perturbation bound")
    perturbation = np.minimum(perturbation, 0.5 * field_host.astype(np.float64) / step)
    plus_host = (field_host.astype(np.float64) + step * perturbation).astype(np.float32)
    minus_host = (field_host.astype(np.float64) - step * perturbation).astype(np.float32)
    if np.min(minus_host) < 0:
        raise ArithmeticError("frozen volume perturbation left the nonnegative domain")
    realised = (plus_host.astype(np.float64) - minus_host.astype(np.float64)) / (2 * step)
    with torch.no_grad():
        plus = objective(projector(torch.tensor(plus_host, device="cuda"), tangent))
        minus = objective(projector(torch.tensor(minus_host, device="cuda"), tangent))
    volume_analytic = float(np.sum(volume_gradient.cpu().numpy().astype(np.float64) * realised))
    bound = guard["volume_fd"][args.precision]
    volume_check = discrepancy(
        volume_analytic, (plus - minus) / (2 * step), bound["rtol"], bound["atol"]
    )
    volume_check.update(step=step, direction="actual FP32 central chord; see saved guard settings")
    actual = image.detach().cpu().numpy()
    ray_checks = []
    for local_index, view in enumerate(indices):
        for row, column in guard["independent_ray_reference"]["pixels"]:
            expected = reference_ray(
                field_host,
                prepared.volume.grid,
                geometry.views[local_index],
                anchor,
                row,
                column,
                1024,
            )
            observed = float(actual[local_index, row, column])
            tolerance = 5e-4 if args.precision == "fp32" else 2e-10
            ray_checks.append(
                dict(
                    view=view,
                    pixel=[row, column],
                    reference=expected,
                    actual=observed,
                    absolute_error=abs(observed - expected),
                    tolerance=tolerance,
                    passed=abs(observed - expected) <= tolerance,
                )
            )
    result = dict(
        schema_version=1,
        harness_sha256=sha256(Path(__file__)),
        measured_at_utc=utc(),
        started_at_utc=started_at,
        precision=args.precision,
        source=digest_sources(args.source),
        input=prepared.metadata,
        frozen_contract_sha256=sha256(args.contract),
        guard_contract_sha256=sha256(args.guard_contract),
        cotangent_sha256=sha256(args.cotangent),
        guard=guard,
        pose_checks=pose_checks,
        volume_directional_check=volume_check,
        independent_rays=ray_checks,
    )
    result["passed"] = (
        pose_checks[0]["passed"]
        and volume_check["passed"]
        and all(row["passed"] for row in ray_checks)
    )
    (args.output / "guards.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "pose": pose_checks[0],
                "volume": volume_check,
                "rays_passed": sum(row["passed"] for row in ray_checks),
            }
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "cotangent", "contract", "guard-contract", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp64"), required=True)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
