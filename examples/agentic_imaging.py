"""Tools for an agent to inspect, fit and assess a calibrated two-view pose case.

The calling agent chooses when to run or retry a fit. DPT supplies the numerical
solver. The demo contains synthetic attenuation and rounded expected counts.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from dpt.contracts import ContractError, finite_scalar, integer
from dpt.examples._common import (
    geometry_from_json,
    grid_from_json,
    load_case,
    mapping,
    pose_from_json,
    required,
)
from dpt.experiments import write_json
from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose
from dpt.recovery import PoseChart
from dpt.registration import RecoveryPolicy
from dpt.validation.recovery import pose_error
from dpt.volumes import GridSpec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chart(configuration: dict) -> PoseChart:
    values = dict(mapping(required(configuration, "chart"), "registration.chart"))
    values["anchor"] = pose_from_json(required(values, "anchor"), "chart.anchor")
    required(values, "scales")
    return PoseChart(**values)


def inspect_case(path: Path) -> dict:
    """Check the supplied arrays and physical contracts without importing Warp."""
    case = load_case(path)
    cfg = mapping(required(case.config, "registration"), "registration")
    grid = grid_from_json(required(cfg, "grid"))
    chart = _chart(cfg)
    policy = RecoveryPolicy(**mapping(required(cfg, "policy"), "registration.policy"))
    samples = integer(required(cfg, "samples_per_ray"), "samples_per_ray", minimum=1)
    if cfg.get("precision", "float64") not in ("float32", "float64"):
        raise ContractError("precision must be float32 or float64")
    if cfg.get("integration", "midpoint") not in ("midpoint", "cell_gauss"):
        raise ContractError("integration must be midpoint or cell_gauss")
    attenuation = case.array(required(cfg, "attenuation"), shape=grid.shape, units="mm^-1")
    if np.any(attenuation < 0):
        raise ContractError("attenuation must be nonnegative")
    raw_views = required(cfg, "views")
    if not isinstance(raw_views, list) or len(raw_views) != 2:
        raise ContractError("this example requires exactly two calibrated views")
    views, identifiers = [], set()
    for raw in raw_views:
        view = mapping(raw, "view")
        identifier = required(view, "id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ContractError("view id must use letters, digits, underscores or hyphens")
        if identifier in identifiers:
            raise ContractError("view ids must be unique")
        identifiers.add(identifier)
        panel = geometry_from_json(required(view, "geometry"))
        counts = case.array(required(view, "observation"), shape=panel.shape, units="counts")
        mask = case.array(required(view, "mask"), shape=panel.shape, units="dimensionless")
        if np.any((counts < 0) | (counts > 2**24) | (counts != np.floor(counts))):
            raise ContractError("observations must contain integer counts in [0, 2**24]")
        if not np.all((mask == 0) | (mask == 1)) or not np.any(mask):
            raise ContractError("each mask must be binary and include at least one pixel")
        if np.any(counts[mask == 0] != 0):
            raise ContractError("excluded observations must already be zero")
        beam = finite_scalar(required(view, "open_beam_counts"), "open_beam_counts")
        if beam <= 0:
            raise ContractError("open_beam_counts must be positive")
        views.append({"id": identifier, "shape": panel.shape, "included_pixels": int(mask.sum())})
    return {
        "status": "inputs_valid",
        "case": str(case.path),
        "case_sha256": _sha256(case.path),
        "input_kind": case.config.get("input_kind", "supplied data; see array provenance"),
        "grid_shape": grid.shape,
        "views": views,
        "initial_pose_object_to_world": asdict(chart.anchor),
        "chart": asdict(chart),
        "policy": asdict(policy),
        "samples_per_ray": samples,
        "objective": "sum of masked Poisson half-deviances",
        "evaluation_reference_used_for_fit": False,
    }


def _cuda(device: str):
    import warp as wp

    if not wp.get_device(device).is_cuda:
        raise ContractError("image simulation and fitting require a CUDA device")
    return wp


def _simulated_views(values, grid, panels, truth, samples, device, source, rights):
    """Project fixed attenuation into rounded expected primary counts on CUDA."""
    from dpt.examples.make_case import CaseArray
    from dpt.recovery import PrimaryPoseEvaluator, PrimaryPoseProblem

    wp = _cuda(device)
    arrays = {"attenuation": CaseArray(values, "mm^-1", source, rights)}
    field = wp.array(values.reshape(-1), dtype=wp.float32, device=device)
    views = []
    for index, panel in enumerate(panels):
        evaluator = PrimaryPoseEvaluator(
            PrimaryPoseProblem(
                grid,
                panel,
                field,
                wp.zeros(panel.pixels, dtype=wp.float32, device=device),
                open_beam=1_000_000.0,
                samples_per_ray=samples,
            ),
            PoseChart(anchor=truth),
            device=device,
        )
        evaluator((0.0,) * 6)
        counts = np.rint(evaluator.prediction.numpy()).astype(np.float32).reshape(panel.shape)
        arrays[f"counts{index}"] = CaseArray(
            counts,
            "counts",
            "Simulated rounded expected primary counts; no sampled noise; "
            f"open beam 1000000; input: {source}",
            rights,
        )
        arrays[f"mask{index}"] = CaseArray(
            np.ones(panel.shape, dtype=np.float32), "dimensionless", "All pixels included", rights
        )
        views.append(
            {
                "id": f"view{index}",
                "geometry": asdict(panel),
                "observation": f"counts{index}",
                "mask": f"mask{index}",
                "open_beam_counts": 1_000_000.0,
            }
        )
    return arrays, views


def prepare_demo(output: Path, device: str) -> dict:
    """Generate a small original phantom and two noiseless, quantised views."""
    from dpt.examples.make_case import write_case

    if output.exists():
        raise ContractError("output directory already exists; choose a new directory")
    grid = GridSpec((9, 9, 9), (1.0, 1.0, 1.0), (-4.0, -4.0, -4.0))
    z, y, x = np.mgrid[-4:5, -4:5, -4:5]
    values = np.full(grid.shape, 0.002, dtype=np.float64)
    for amplitude, cx, cy, cz, sx, sy, sz in (
        (0.050, -1.3, -0.8, 0.7, 0.9, 1.2, 0.7),
        (0.035, 1.4, 0.9, -1.1, 1.1, 0.65, 1.3),
        (0.025, 0.2, -1.5, -1.4, 0.6, 0.8, 0.7),
        (0.040, -0.9, 1.2, 1.4, 0.6, 0.6, 0.9),
    ):
        values += amplitude * np.exp(
            -0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2 + ((z - cz) / sz) ** 2)
        )
    values = values.astype(np.float32)
    panels = (
        DetectorGeometry(
            (-30, 0.13, 0.17), (30, -5.5, -5.5), (0, 1, 0), (0, 0, 1), (1, 1), (12, 12)
        ),
        DetectorGeometry(
            (0.13, -30, 0.17), (-5.5, 30, -5.5), (1, 0, 0), (0, 0, 1), (1, 1), (12, 12)
        ),
    )
    truth = compose_pose(RigidTransform(), (0.12, -0.08, 0.06, 0.02, -0.015, 0.01))
    chart = PoseChart(scales=(0.1, 0.1, 0.1, 0.01, 0.01, 0.01), rotation_radius_radians=0.15)
    provenance = "Original asymmetric Gaussian attenuation phantom"
    arrays, views = _simulated_views(
        values, grid, panels, truth, 96, device, provenance, "Apache-2.0"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    case_path = write_case(
        output,
        {
            "input_kind": "synthetic noiseless phantom with rounded expected counts",
            "registration": {
                "grid": asdict(grid),
                "attenuation": "attenuation",
                "chart": asdict(chart),
                "samples_per_ray": 96,
                "precision": "float64",
                "policy": {"gradient_tolerance": 1e-4},
                "views": views,
            },
        },
        arrays,
    )
    write_json(
        output / "reference.json",
        {
            "case_sha256": _sha256(case_path),
            "source": "Known generating pose of the synthetic example",
            "uncertainty": "Known generating pose of the synthetic example",
            "pose_object_to_world": asdict(truth),
            "landmarks_object_mm": [[-2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 1]],
        },
    )
    return {
        "status": "demo_prepared",
        "case": str(case_path),
        "reference": str(output / "reference.json"),
    }


def _centred_pose(grid: GridSpec, translation_mm, rotation_degrees):
    """Place x at R(x-c)+c+d, retaining the native object-coordinate grid."""
    if len(translation_mm) != 3 or len(rotation_degrees) != 3:
        raise ContractError("initial translation and rotation each need three coordinates")
    displacement = tuple(finite_scalar(value, "initial translation") for value in translation_mm)
    rotation = tuple(
        math.radians(finite_scalar(value, "initial rotation")) for value in rotation_degrees
    )
    if math.hypot(*rotation) >= 0.15:
        raise ContractError(
            "initial rotation vector must have norm below the 0.15-radian chart radius"
        )
    nz, ny, nx = grid.shape
    centre = grid.grid_to_object(((nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2))
    pure_rotation = compose_pose(RigidTransform(), (0.0, 0.0, 0.0, *rotation))
    rotated = pure_rotation.point(centre)
    anchor = RigidTransform(
        pure_rotation.rotation,
        tuple(centre[k] + displacement[k] - rotated[k] for k in range(3)),
    )
    points = [
        grid.grid_to_object((fx * (nx - 1), fy * (ny - 1), fz * (nz - 1)))
        for fz in (0.25, 0.75)
        for fy in (0.25, 0.75)
        for fx in (0.25, 0.75)
    ]
    return anchor, centre, points


def prepare_ct(
    volume_path: Path,
    output: Path,
    device: str,
    *,
    units: str,
    source: str,
    rights: str,
    size: int = 128,
    samples_per_ray: int = 768,
    energy_kev: float = 60.0,
    initial_translation_mm=(3.0, -2.0, 1.5),
    initial_rotation_degrees=(2.0, -1.5, 1.0),
) -> dict:
    """Keep a supplied CT grid and simulate AP/lateral count images from its HU values."""
    from dpt.examples.make_case import write_case
    from dpt.imaging import standard_views
    from dpt.rendering import load_attenuation

    if output.exists():
        raise ContractError("output directory already exists; choose a new directory")
    if units != "HU":
        raise ContractError("prepare-ct requires explicitly declared HU units")
    if finite_scalar(energy_kev, "energy_kev") != 60.0:
        raise ContractError("the water-scaled HU conversion requires energy_kev=60.0")
    for name, value in (("source", source), ("rights", rights)):
        if not isinstance(value, str) or not value.strip():
            raise ContractError(f"{name} must describe the supplied CT")
    integer(size, "size", minimum=1)
    integer(samples_per_ray, "samples_per_ray", minimum=1)
    if not volume_path.name.lower().endswith((".nii", ".nii.gz")):
        raise ContractError("prepare-ct requires a NIfTI .nii or .nii.gz file")
    original_hash = _sha256(volume_path)
    volume, attenuation = load_attenuation(volume_path, units=units, energy_kev=energy_kev)
    grid = attenuation.grid
    anchor, centre, points = _centred_pose(grid, initial_translation_mm, initial_rotation_degrees)
    geometry = standard_views(grid, ("ap", "lateral"), shape=(size, size))
    chart = PoseChart(
        anchor=anchor, scales=(1, 1, 1, 0.01, 0.01, 0.01), rotation_radius_radians=0.15
    )
    provenance = f"{source.strip()}; 60 keV water-scaled HU conversion on the native CT grid"
    arrays, views = _simulated_views(
        attenuation.values,
        grid,
        geometry.views,
        RigidTransform(),
        samples_per_ray,
        device,
        provenance,
        rights.strip(),
    )
    if _sha256(volume_path) != original_hash:
        raise ContractError("source CT changed while preparing the case")
    output.parent.mkdir(parents=True, exist_ok=True)
    case_path = write_case(
        output,
        {
            "input_kind": "real CT attenuation with simulated noiseless rounded primary counts",
            "ct_input": {
                "filename": volume_path.name,
                "sha256": original_hash,
                "source": source.strip(),
                "rights": rights.strip(),
                "input_units": units,
                "input_metadata": asdict(volume.metadata),
                "resampled": False,
                "coordinate_frame": "RAS",
                "physical_centre_mm": centre,
                "attenuation_model": {
                    "name": "water-scaled monoenergetic HU",
                    "energy_kev": energy_kev,
                    "formula": "max(1 + HU/1000, 0) * 0.02059 mm^-1",
                    "mu_water_mm_inverse": 0.02059,
                },
            },
            "simulated_acquisition": {
                "view_names": ["ap", "lateral"],
                "geometry_source": "standard_views AP and lateral geometry",
                "counts_model": "rounded expected primary counts",
                "open_beam_counts": 1_000_000.0,
                "initial_displacement_mm": initial_translation_mm,
                "initial_axis_angle_degrees": initial_rotation_degrees,
                "initial_pose_definition": "R(x-c)+c+d; rotation about the CT physical centre",
            },
            "registration": {
                "grid": asdict(grid),
                "attenuation": "attenuation",
                "chart": asdict(chart),
                "samples_per_ray": samples_per_ray,
                "precision": "float64",
                "policy": {"gradient_tolerance": 1e-4},
                "views": views,
            },
        },
        arrays,
    )
    write_json(
        output / "reference.json",
        {
            "case_sha256": _sha256(case_path),
            "source": "Known simulated acquisition pose; eight grid-derived geometric points",
            "uncertainty": (
                "Known generating pose for simulated projections of the real CT"
            ),
            "landmark_kind": "eight geometric points derived from the CT grid",
            "pose_object_to_world": asdict(RigidTransform()),
            "landmarks_object_mm": points,
        },
    )
    return {
        "status": "ct_prepared",
        "case": str(case_path),
        "reference": str(output / "reference.json"),
        "source_sha256": original_hash,
        "grid_shape": grid.shape,
        "detector_shape": [size, size],
        "samples_per_ray": samples_per_ray,
    }


def fit_case(case_path: Path, output: Path, device: str, iterations: int, evaluations: int) -> dict:
    """Run one bounded solve from the immutable initial pose in the supplied case."""
    from dpt.examples.registration import SharedPoseObjective, prepare_views
    from dpt.experiments import describe_device
    from dpt.registration import recover_parameters

    inspection = inspect_case(case_path)
    case = load_case(case_path)
    cfg = case.config["registration"]
    chart = _chart(cfg)
    policy = replace(
        RecoveryPolicy(**cfg["policy"]), max_iterations=iterations, max_evaluations=evaluations
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(parents=False, exist_ok=False)
    except FileExistsError as error:
        raise ContractError("output directory already exists; choose a new directory") from error
    write_json(output / "inputs.json", inspection)
    write_json(output / "configuration.json", case.config)
    _cuda(device)
    views = prepare_views(case, cfg, chart, device)
    objective = SharedPoseObjective(views, chart)
    first_evaluation = True

    def evaluate(parameters):
        nonlocal first_evaluation
        value = objective(parameters)
        if first_evaluation:
            for view, supplied in zip(views, cfg["views"], strict=True):
                prediction = view.evaluator.prediction.numpy().reshape(
                    view.evaluator.problem.geometry.shape
                )
                observed = case.array(
                    supplied["observation"], shape=prediction.shape, units="counts"
                )
                np.save(
                    output / f"initial-prediction-{view.identifier}.npy",
                    prediction,
                    allow_pickle=False,
                )
                np.save(
                    output / f"initial-residual-{view.identifier}.npy",
                    prediction - observed,
                    allow_pickle=False,
                )
            first_evaluation = False
        return value

    result = recover_parameters(evaluate, (0.0,) * 6, policy=policy)
    diagnostics = []
    for view, supplied in zip(views, cfg["views"], strict=True):
        final = view.evaluator(result.parameters)
        predicted = view.evaluator.prediction.numpy().reshape(view.evaluator.problem.geometry.shape)
        observed = case.array(supplied["observation"], shape=predicted.shape, units="counts")
        mask = case.array(supplied["mask"], shape=predicted.shape, units="dimensionless") == 1
        residual = predicted - observed
        np.save(output / f"prediction-{view.identifier}.npy", predicted, allow_pickle=False)
        np.save(output / f"residual-{view.identifier}.npy", residual, allow_pickle=False)
        diagnostics.append(
            {
                "id": view.identifier,
                "included_pixels": view.included_pixels,
                "half_deviance": final.loss,
                "residual_rms_counts": float(np.sqrt(np.mean(residual[mask] ** 2))),
                "residual_max_abs_counts": float(np.max(np.abs(residual[mask]))),
            }
        )
    if _sha256(case.path) != inspection["case_sha256"]:
        raise ContractError("case changed while fitting")
    # Reload every fitted input to detect changed bytes before reporting completion.
    inspect_case(case.path)
    summary = {
        "status": "stationary" if result.stationary else "unresolved",
        "termination": result.reason,
        "case_sha256": inspection["case_sha256"],
        "initial_loss": result.history[0].loss,
        "final_loss": result.evaluation.loss,
        "gradient_infinity_norm": max(abs(v) for v in result.evaluation.gradient),
        "solver_evaluations": result.evaluations,
        "additional_final_view_evaluations": len(views),
        "initial_pose_object_to_world": asdict(chart.anchor),
        "pose_object_to_world": asdict(chart.pose(result.parameters)),
        "policy": asdict(policy),
        "chart": asdict(chart),
        "views": diagnostics,
        "evaluation_reference_used_for_fit": False,
        "output": str(output.resolve()),
        "interpretation": "Stationarity records convergence of the fitted objective",
    }
    write_json(output / "device.json", describe_device(views[0].evaluator.context))
    write_json(output / "optimisation.json", asdict(result))
    write_json(output / "summary.json", summary)
    return summary


def assess_run(run: Path, reference: Path | None = None) -> dict:
    """Read solver evidence; score a separate geometric reference only after fitting."""
    summary = json.loads((run / "summary.json").read_text())
    result = {**summary, "geometric_evaluation": "no independent reference supplied"}
    if reference is not None:
        truth = json.loads(reference.read_text())
        if truth["case_sha256"] != summary["case_sha256"]:
            raise ContractError("reference belongs to a different case")
        target = pose_from_json(truth["pose_object_to_world"])
        points = tuple(tuple(point) for point in truth["landmarks_object_mm"])
        result["geometric_evaluation"] = {
            "source": truth["source"],
            "uncertainty": truth["uncertainty"],
            "point_kind": truth.get("landmark_kind", "supplied reference landmarks"),
            "initial": asdict(
                pose_error(pose_from_json(summary["initial_pose_object_to_world"]), target, points)
            ),
            "final": asdict(
                pose_error(pose_from_json(summary["pose_object_to_world"]), target, points)
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("prepare-demo", help="create two synthetic calibrated images")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--device", default="cuda:0")
    ct = commands.add_parser("prepare-ct", help="simulate two projections of a supplied real CT")
    ct.add_argument("volume", type=Path)
    ct.add_argument("--output", type=Path, required=True)
    ct.add_argument("--units", choices=("HU",), required=True)
    ct.add_argument("--source", required=True)
    ct.add_argument("--rights", required=True)
    ct.add_argument("--device", default="cuda:0")
    ct.add_argument("--size", type=int, default=128)
    ct.add_argument("--samples-per-ray", type=int, default=768)
    ct.add_argument("--energy-kev", type=float, default=60.0)
    ct.add_argument("--initial-translation-mm", type=float, nargs=3, default=(3.0, -2.0, 1.5))
    ct.add_argument("--initial-rotation-degrees", type=float, nargs=3, default=(2.0, -1.5, 1.0))
    inspect = commands.add_parser("inspect", help="validate a two-view case on the host")
    inspect.add_argument("--case", type=Path, required=True)
    fit = commands.add_parser("fit", help="run one bounded fit from the case's initial pose")
    fit.add_argument("--case", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--max-iterations", type=int, default=120)
    fit.add_argument("--max-evaluations", type=int, default=800)
    assess = commands.add_parser("assess", help="read a fit and optionally score held-out truth")
    assess.add_argument("--run", type=Path, required=True)
    assess.add_argument("--reference", type=Path)
    args = parser.parse_args()
    try:
        # Warp compilation messages belong on stderr; stdout remains one JSON object.
        with contextlib.redirect_stdout(sys.stderr):
            if args.command == "prepare-demo":
                result = prepare_demo(args.output, args.device)
            elif args.command == "prepare-ct":
                result = prepare_ct(
                    args.volume,
                    args.output,
                    args.device,
                    units=args.units,
                    source=args.source,
                    rights=args.rights,
                    size=args.size,
                    samples_per_ray=args.samples_per_ray,
                    energy_kev=args.energy_kev,
                    initial_translation_mm=args.initial_translation_mm,
                    initial_rotation_degrees=args.initial_rotation_degrees,
                )
            elif args.command == "inspect":
                result = inspect_case(args.case)
            elif args.command == "fit":
                result = fit_case(
                    args.case, args.output, args.device, args.max_iterations, args.max_evaluations
                )
            else:
                result = assess_run(args.run, args.reference)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ContractError, OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
