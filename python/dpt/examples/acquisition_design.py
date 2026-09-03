"""Rank supplied next-view candidates by expected local target-position variance.

The known attenuation field and calibrated candidate beams are supplied inputs.
The design uses independent ideal Poisson counts, a fixed local pose chart and a
strictly positive-definite current precision. It does not acquire an image or
control imaging hardware. Only compact information matrices leave the GPU.
See the examples README for recorded policy comparisons and their limitations.
"""

from __future__ import annotations

import argparse
import importlib
import math
from typing import Any, Literal, cast

from dpt._runtime import prepare_context
from dpt.contracts import ContractError, NumericalError, finite_scalar, integer
from dpt.examples._common import (
    example_parser,
    geometry_from_json,
    grid_from_json,
    load_case,
    pose_from_json,
    required,
    run_example,
    write_array,
)
from dpt.examples._common import (
    mapping as _mapping,
)
from dpt.experiments import describe_device, prepare_output, write_json, write_metadata
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.projection import (
    ProjectionSpec,
    prepare_projection,
    project_optical_depth,
    projection_pose_sensitivities,
)
from dpt.transmission import TransmissionSpec, prepare_transmission, transmit
from dpt.volumes import GridSpec


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} needs a non-empty description")
    return value


def _positive(value: Any, name: str) -> float:
    result = finite_scalar(value, name, minimum=0.0)
    if result == 0:
        raise ContractError(f"{name} must be strictly positive")
    return result


def target_jacobians(np: Any, pose: RigidTransform, targets: Any, scales: Any) -> Any:
    """Return d(world target)/dz in mm for T_WO exp((S z)^) at z=0."""
    rotation = np.asarray(pose.rotation, dtype=np.float64).reshape(3, 3)
    result = np.empty((len(targets), 3, 6), dtype=np.float64)
    for index, (x, y, z) in enumerate(targets):
        skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        result[index, :, :3] = rotation
        result[index, :, 3:] = -rotation @ skew
    result *= scales[None, None, :]
    if not np.isfinite(result).all():
        raise NumericalError("target derivatives exceed FP64; check target coordinates and scales")
    return result


def target_variance(
    np: Any, precision: Any, target_jacobian: Any, maximum_condition: float
) -> tuple[float, Any, float]:
    """Mean target-position variance in mm², without forming an inverse to score."""
    if not np.isfinite(precision).all():
        raise NumericalError("pose precision is non-finite; discard this candidate score")
    values = np.linalg.eigvalsh(precision)
    if values[0] <= 0 or values[-1] / values[0] > maximum_condition:
        raise NumericalError(
            "pose precision must be positive definite and within maximum_precision_condition; "
            "check the stated uncertainty and parameter scales, without adding numerical jitter"
        )
    factor = np.linalg.cholesky(precision)
    right = target_jacobian.reshape(-1, 6).T
    whitened = np.linalg.solve(factor, right)
    if not np.isfinite(whitened).all():
        raise NumericalError(
            "target covariance solve exceeds FP64; check the precision and chart scales"
        )
    # hypot scales its sum of squares; individual products must not underflow
    # before they contribute to an otherwise representable total variance.
    normalisation = math.sqrt(len(target_jacobian))
    root_variance = math.hypot(*(float(value) / normalisation for value in whitened.flat))
    variance = root_variance * root_variance
    if not math.isfinite(variance) or variance <= 0:
        raise NumericalError(
            "positive target-position variance is outside FP64 range; check the stated "
            "uncertainty and physical units, or use a range-preserving score representation"
        )
    inverse_factor = np.linalg.solve(factor, np.eye(6))
    covariance = inverse_factor.T @ inverse_factor
    if not np.isfinite(covariance).all() or bool((np.diag(covariance) <= 0).any()):
        raise NumericalError(
            "local covariance is outside FP64 range; check the precision and chart scales "
            "before exporting this candidate"
        )
    return variance, covariance, float(values[-1] / values[0])


def candidate_information(
    *,
    np: Any,
    torch: Any,
    ctx: Any,
    grid: GridSpec,
    geometry: DetectorGeometry,
    attenuation: Any,
    pose: Any,
    beam_host: Any,
    scales: Any,
    samples_per_ray: int,
    precision: Literal["float32", "float64"] = "float32",
    integration: Literal["midpoint", "cell_gauss"] = "midpoint",
) -> tuple[Any, dict[str, Any]]:
    """Use canonical depth derivatives and an FP64 CUDA Fisher reduction.

    Preparation allocates one candidate's O(6P) Jacobian and weighted copy.
    Torch views alias Warp storage on the same CUDA stream. There is no
    per-pixel Python loop and no image/Jacobian download. This is an offline
    candidate evaluation, not a captured or profiled optimisation hot path.
    """
    wp, pixels = ctx.wp, geometry.pixels
    try:
        projection = prepare_projection(
            grid,
            geometry,
            ProjectionSpec(samples_per_ray, precision=precision, integration=integration),
            device=str(ctx.device),
            stream=ctx.stream,
        )
        transmission = (
            prepare_transmission(
                TransmissionSpec(beam="none"),
                max_pixels=pixels,
                device=str(ctx.device),
                stream=ctx.stream,
            )
            if precision == "float32"
            else None
        )
        with ctx.scope():
            beam = wp.array(beam_host.reshape(-1), dtype=wp.float32, device=ctx.device)
            depth = wp.empty(pixels, dtype=projection.dtype, device=ctx.device)
            log_transmission = wp.empty(
                pixels if precision == "float32" else 0, dtype=wp.float32, device=ctx.device
            )
            ones = wp.ones(pixels, dtype=projection.dtype, device=ctx.device)
            jacobian = wp.empty(6 * pixels, dtype=wp.float64, device=ctx.device)
        project_optical_depth(
            attenuation,
            pose,
            workspace=projection,
            out_L=depth,
            stream=ctx.stream,
        )
        if transmission is not None:
            transmit(
                depth,
                out_log_T=log_transmission,
                workspace=transmission,
                stream=ctx.stream,
            )
        projection_pose_sensitivities(
            attenuation,
            pose,
            cotangent=ones,
            out_jacobian=jacobian,
            workspace=projection,
            stream=ctx.stream,
        )
        beam_tensor = wp.to_torch(beam)
        log_mean = beam_tensor.to(dtype=torch.float64).log_()
        if precision == "float64":
            log_mean.sub_(wp.to_torch(depth))
        else:
            log_mean.add_(wp.to_torch(log_transmission))
        illuminated = beam_tensor > 0
        # A zero supplied beam carries no information. Positive means outside
        # the normal FP64 range are rejected; no tail, floor or pixel is hidden.
        log_tiny = math.log(float(np.finfo(np.float64).tiny))
        if bool((illuminated & (log_mean < log_tiny)).any().item()):
            raise NumericalError(
                "a positive candidate count mean is below normal FP64 range; "
                "use a range-preserving design implementation before ranking this case"
            )
        root_mean = log_mean.mul_(0.5).exp_()
        depth_jacobian = wp.to_torch(jacobian).reshape(pixels, 6)
        weighted = depth_jacobian.clone()
        weighted.mul_(scales).mul_(root_mean[:, None])
        if not bool(torch.isfinite(weighted).all().item()):
            raise NumericalError("weighted pose derivatives exceed FP64; discard this candidate")
        if bool(((depth_jacobian != 0) & illuminated[:, None] & (weighted == 0)).any().item()):
            raise NumericalError(
                "a nonzero weighted derivative underflowed; discard this candidate"
            )
        magnitude = weighted.abs()
        minimum = math.sqrt(float(np.finfo(np.float64).tiny))
        maximum = math.sqrt(float(np.finfo(np.float64).max) / (6.0 * pixels))
        if bool((((magnitude > 0) & (magnitude < minimum)) | (magnitude > maximum)).any().item()):
            raise NumericalError(
                "candidate Fisher products exceed the declared FP64 accumulation range; "
                "rescale the pose chart or use a range-preserving reduction"
            )
        # Weighted depth derivatives give J_L^T diag(lambda) J_L directly.
        # Avoid a division by rounded, potentially zero count predictions.
        information = weighted.T @ weighted
        result = information.cpu().numpy().copy()
        if not np.isfinite(result).all():
            raise NumericalError("candidate information is non-finite; discard this score")
        diagnostics = {
            "pixels": pixels,
            "precision": precision,
            "integration": integration,
            "zero_beam_pixels": int((~illuminated).sum().item()),
            "omitted_positive_beam_pixels": 0,
            "jacobian_and_weighted_copy_bytes": 2 * 6 * pixels * 8,
            "fisher_reduction": "FP64 CUDA; only the 6 by 6 information matrix is downloaded",
        }
        return 0.5 * (result + result.T), diagnostics
    finally:
        # Warp owners and the host beam must survive every pending shared-stream operation.
        wp.synchronize_stream(ctx.stream)


def main() -> None:
    run_example(example_parser(__doc__ or "Rank supplied next-view candidates"), _run)


def _run(args: argparse.Namespace) -> None:
    case = load_case(args.case)
    cfg = _mapping(required(case.config, "acquisition_design", "case"), "acquisition_design")
    for name in (
        "grid",
        "pose",
        "attenuation",
        "prior_precision",
        "targets_object_mm",
        "parameter_scales",
        "samples_per_ray",
        "max_candidate_pixels",
        "maximum_cost",
        "cost_unit",
        "current_information_description",
        "candidates",
    ):
        required(cfg, name, "acquisition_design")
    np: Any = importlib.import_module("numpy")
    grid = grid_from_json(cfg["grid"])
    pose = pose_from_json(cfg["pose"])
    attenuation_host = case.array(cfg["attenuation"], shape=grid.shape, units="mm^-1")
    if (attenuation_host < 0).any():
        raise ContractError("attenuation must be nonnegative and already converted to mm^-1")
    prior = case.array(cfg["prior_precision"], dtype="float64", shape=(6, 6), units="dimensionless")
    if not np.allclose(prior, prior.T, rtol=0.0, atol=1e-12 * float(abs(prior).max())):
        raise ContractError("prior_precision must be symmetric in the supplied scaled pose chart")
    prior = 0.5 * (prior + prior.T)
    targets = case.array(cfg["targets_object_mm"], dtype="float64", units="mm")
    if targets.ndim != 2 or targets.shape[1] != 3 or not 1 <= targets.shape[0] <= 256:
        raise ContractError(
            "targets_object_mm must contain 1 to 256 object-frame points, shape (N,3)"
        )
    raw_scales = cfg["parameter_scales"]
    if len(raw_scales) != 6:
        raise ContractError("parameter_scales needs six positive values: mm, mm, mm, rad, rad, rad")
    scales = np.asarray([_positive(value, "parameter scale") for value in raw_scales])
    target_derivatives = target_jacobians(np, pose, targets, scales)
    samples = integer(cfg["samples_per_ray"], "samples_per_ray", minimum=1)
    capacity = integer(cfg["max_candidate_pixels"], "max_candidate_pixels", minimum=1)
    maximum_cost = _positive(cfg["maximum_cost"], "maximum_cost")
    cost_unit = cfg["cost_unit"]
    if cost_unit not in ("mAs", "s", "expected_detector_photons"):
        raise ContractError(
            "cost_unit must be mAs, s or expected_detector_photons; none is absorbed dose"
        )
    current_description = _text(cfg["current_information_description"], "current information")
    maximum_condition = _positive(
        cfg.get("maximum_precision_condition", 1e12), "precision condition"
    )
    if maximum_condition < 1:
        raise ContractError("maximum_precision_condition must be at least one")
    rank_tolerance = _positive(cfg.get("rank_relative_tolerance", 1e-10), "rank tolerance")
    if rank_tolerance >= 1:
        raise ContractError("rank_relative_tolerance must be smaller than one")
    baseline, baseline_covariance, _ = target_variance(
        np, prior, target_derivatives, maximum_condition
    )
    raw_candidates = cfg["candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ContractError("candidates must list at least one supplied feasible acquisition")
    candidates: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in cast(list[Any], raw_candidates):
        candidate = _mapping(raw, "candidate")
        for field in ("name", "geometry", "open_beam", "cost", "feasibility_description"):
            required(candidate, field, "candidate")
        name = _text(candidate["name"], "candidate name")
        if name in names:
            raise ContractError(f"candidate name {name!r} is repeated; assign unique names")
        names.add(name)
        geometry = geometry_from_json(candidate["geometry"], f"{name} geometry")
        if geometry.pixels > capacity:
            raise ContractError(
                f"{name} exceeds max_candidate_pixels; declare a sufficient memory bound"
            )
        beam = case.array(candidate["open_beam"], shape=geometry.shape, units="photons/pixel")
        if (beam < 0).any() or not (beam > 0).any():
            raise ContractError(
                f"{name} needs a nonnegative beam with at least one illuminated pixel"
            )
        cost = _positive(candidate["cost"], f"{name} cost")
        if cost_unit == "expected_detector_photons":
            expected = float(np.sum(beam, dtype=np.float64))
            if not math.isclose(cost, expected, rel_tol=1e-6, abs_tol=0.0):
                raise ContractError(f"{name} cost must equal the sum of its open-beam expectations")
        candidates.append(
            {
                "name": name,
                "geometry": geometry,
                "beam": beam,
                "cost": cost,
                "feasibility_description": _text(
                    candidate["feasibility_description"], f"{name} feasibility"
                ),
            }
        )
    eligible = [candidate for candidate in candidates if candidate["cost"] <= maximum_cost]
    if not eligible:
        raise ContractError(
            "no supplied candidate fits maximum_cost; revise the candidate set or budget"
        )
    output = prepare_output(
        args.output,
        configuration=case.config,
        metadata={
            "application": "finite_candidate_acquisition_design",
            "evidence_kind": "expected local uncertainty under a fixed ideal Poisson model",
            "current_information_description": current_description,
            "pose_chart": "local right SE(3) at supplied pose; dimensionless z with supplied S",
            "execution_validation": "this run does not establish achieved registration accuracy",
            "tail_policy": (
                "reject unrepresentable positive-beam contributions; no positive tails omitted"
            ),
        },
    )
    torch: Any = importlib.import_module("torch")
    wp: Any = importlib.import_module("warp")
    torch_device = torch.device(args.device)
    if torch_device.type != "cuda":
        raise ContractError("acquisition design requires CUDA; select --device cuda:N")
    torch_stream = torch.cuda.current_stream(torch_device)
    wp.init()
    stream = wp.stream_from_torch(torch_stream)
    ctx = prepare_context(device=args.device, stream=stream)
    rows: list[dict[str, Any]] = []
    matrices: list[Any] = []
    covariances: list[Any] = []
    try:
        with (
            torch.cuda.device(torch_device),
            torch.cuda.stream(torch_stream),
            torch.no_grad(),
            ctx.scope(),
        ):
            attenuation = wp.array(
                attenuation_host.reshape(-1), dtype=wp.float32, device=ctx.device
            )
            pose_device = wp.array(pose.packed(), dtype=wp.float64, device=ctx.device)
            scales_device = torch.as_tensor(scales, dtype=torch.float64, device=torch_device)
            for candidate in eligible:
                information, diagnostics = candidate_information(
                    np=np,
                    torch=torch,
                    ctx=ctx,
                    grid=grid,
                    geometry=candidate["geometry"],
                    attenuation=attenuation,
                    pose=pose_device,
                    beam_host=candidate["beam"],
                    scales=scales_device,
                    samples_per_ray=samples,
                    precision=cfg.get("precision", "float32"),
                    integration=cfg.get("integration", "midpoint"),
                )
                variance, covariance, condition = target_variance(
                    np, prior + information, target_derivatives, maximum_condition
                )
                eigenvalues = np.linalg.eigvalsh(information)
                largest = float(eigenvalues[-1])
                if eigenvalues[0] < -rank_tolerance * max(largest, np.finfo(np.float64).tiny):
                    raise NumericalError(
                        "candidate Fisher matrix is indefinite beyond rank tolerance"
                    )
                rank = int(np.count_nonzero(eigenvalues > rank_tolerance * largest))
                rows.append(
                    {
                        "name": candidate["name"],
                        "cost": candidate["cost"],
                        "mean_target_variance_mm2": variance,
                        "local_fisher_rank": rank,
                        "posterior_precision_condition": condition,
                        "diagnostics": diagnostics,
                    }
                )
                matrices.append(information)
                covariances.append(covariance)
            # Exact score ties prefer the smaller stated cost, then the name.
            selected = min(
                rows,
                key=lambda row: (row["mean_target_variance_mm2"], row["cost"], row["name"]),
            )
            wp.synchronize_stream(stream)
    finally:
        torch_stream.synchronize()
    write_metadata(output, **describe_device(ctx), warp=wp.__version__, torch=torch.__version__)
    write_array(output, "candidate_information.npy", np.stack(matrices))
    write_array(output, "candidate_local_covariance.npy", np.stack(covariances))
    write_array(output, "current_local_covariance.npy", baseline_covariance)
    write_json(
        output / "design.json",
        {
            "selected_candidate": selected["name"],
            "cost_unit": cost_unit,
            "maximum_cost": maximum_cost,
            "current_mean_target_variance_mm2": baseline,
            "score": "mean trace of local target covariance; lower is preferred",
            "rank_relative_tolerance": rank_tolerance,
            "matrix_row_order": [row["name"] for row in rows],
            "candidates": rows,
            "excluded_by_budget": [
                candidate["name"] for candidate in candidates if candidate["cost"] > maximum_cost
            ],
            "interpretation": "predicted local uncertainty; no selected image acquired",
        },
    )


if __name__ == "__main__":
    main()
