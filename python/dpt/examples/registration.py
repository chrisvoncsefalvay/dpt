"""Fit one rigid pose to supplied calibrated primary-count radiographs.

All image formation and derivatives use the canonical CUDA operators. NumPy
handles supplied-file validation and final reports, never an image optimiser.
Supplied JSON/NPY values, physical geometry and units are checked at input
boundaries.
"""

from __future__ import annotations

import argparse
import importlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, cast

from dpt.contracts import ContractError, finite_scalar, integer
from dpt.examples._common import (
    example_parser,
    load_case,
    run_example,
    write_array,
)
from dpt.examples._common import (
    mapping as _mapping,
)
from dpt.examples._common import (
    required as _required,
)
from dpt.experiments import describe_device, prepare_output, write_json, write_metadata
from dpt.geometry import RigidTransform, Vector3
from dpt.objectives import ObjectiveSpec
from dpt.recovery import PoseChart, PrimaryPoseEvaluator, PrimaryPoseProblem
from dpt.registration import Evaluation, RecoveryPolicy, Vector, recover_parameters
from dpt.validation.recovery import pose_error
from dpt.volumes import GridSpec


def _immutable_geometry(values: dict[str, Any]) -> dict[str, Any]:
    """Decode JSON arrays at the immutable host-geometry boundary."""
    return {
        key: tuple(cast(list[Any], value)) if isinstance(value, list) else value
        for key, value in values.items()
    }


@dataclass(frozen=True, slots=True)
class PreparedView:
    identifier: str
    evaluator: PrimaryPoseEvaluator
    included_pixels: int


class SharedPoseObjective:
    """Sum views in one immutable chart; each evaluator owns its image scratch."""

    def __init__(self, views: tuple[PreparedView, ...], chart: PoseChart) -> None:
        if not views or any(view.evaluator.chart is not chart for view in views):
            raise ContractError("all views must share the same PoseChart object")
        self.views, self.chart = views, chart

    def __call__(self, parameters: Vector) -> Evaluation:
        values = tuple(view.evaluator(parameters) for view in self.views)
        return Evaluation(
            math.fsum(value.loss for value in values),
            tuple(math.fsum(value.gradient[k] for value in values) for k in range(6)),
        )


def prepare_views(
    case: Any, configuration: dict[str, Any], chart: PoseChart, device: str
) -> tuple[PreparedView, ...]:
    """Validate supplied arrays and upload fixed inputs once on a shared stream."""
    from dpt.geometry import DetectorGeometry

    wp: Any = importlib.import_module("warp")
    np: Any = importlib.import_module("numpy")
    grid = GridSpec(
        **_immutable_geometry(_mapping(_required(configuration, "grid"), "registration.grid"))
    )
    attenuation_host = case.array(
        _required(configuration, "attenuation"), shape=grid.shape, units="mm^-1"
    )
    if not bool(np.all(np.isfinite(attenuation_host) & (attenuation_host >= 0))):
        raise ContractError("attenuation must contain finite nonnegative values in mm^-1")
    samples = integer(_required(configuration, "samples_per_ray"), "samples_per_ray", minimum=1)
    raw_views = _required(configuration, "views")
    if not isinstance(raw_views, list) or not raw_views:
        raise ContractError("registration.views needs at least one calibrated count view")
    views: list[dict[str, Any]] = [
        _mapping(value, "registration.views entry") for value in cast(list[Any], raw_views)
    ]
    identifiers: set[str] = set()
    prepared: list[PreparedView] = []
    with wp.ScopedDevice(device):
        stream = wp.Stream(device=device)
        with wp.ScopedStream(stream):
            try:
                attenuation = wp.array(
                    attenuation_host.reshape(-1), dtype=wp.float32, device=device
                )
                for view in views:
                    identifier = _required(view, "id")
                    if not isinstance(identifier, str) or not re.fullmatch(
                        r"[A-Za-z0-9_-]+", identifier
                    ):
                        raise ContractError(
                            "view id must use letters, digits, underscores or hyphens"
                        )
                    if identifier in identifiers:
                        raise ContractError(
                            f"duplicate view id {identifier!r}; give each view its own id"
                        )
                    identifiers.add(identifier)
                    geometry = DetectorGeometry(
                        **_immutable_geometry(
                            _mapping(_required(view, "geometry"), "view.geometry")
                        )
                    )
                    observed = case.array(
                        _required(view, "observation"), shape=geometry.shape, units="counts"
                    )
                    mask = case.array(
                        _required(view, "mask"), shape=geometry.shape, units="dimensionless"
                    )
                    valid_counts = np.isfinite(observed) & (observed >= 0) & (observed <= 2**24)
                    if not bool(np.all(valid_counts & (observed == np.floor(observed)))):
                        raise ContractError(
                            f"view {identifier}: counts must be integers in [0, 2**24], stored as "
                            "float32; processed display images cannot use this Poisson example"
                        )
                    if not bool(np.all((mask == 0) | (mask == 1))) or not bool(np.any(mask == 1)):
                        raise ContractError(
                            f"view {identifier}: mask must be binary with included pixels"
                        )
                    if bool(np.any((mask == 0) & (observed != 0))):
                        raise ContractError(
                            f"view {identifier}: set every mask=0 observation to zero in the "
                            "supplied preprocessed array; preserve the raw source and record "
                            "this preprocessing in its provenance"
                        )
                    beam = finite_scalar(_required(view, "open_beam_counts"), "open_beam_counts")
                    if beam <= 0:
                        raise ContractError(f"view {identifier}: open_beam_counts must be positive")
                    problem = PrimaryPoseProblem(
                        grid=grid,
                        geometry=geometry,
                        attenuation=attenuation,
                        observation=wp.array(observed.reshape(-1), dtype=wp.float32, device=device),
                        objective=ObjectiveSpec(
                            kind="poisson", domain="counts", reduction="sum", weighted=True
                        ),
                        open_beam=beam,
                        weights=wp.array(mask.reshape(-1), dtype=wp.float32, device=device),
                        samples_per_ray=samples,
                        precision=configuration.get("precision", "float64"),
                        integration=configuration.get("integration", "midpoint"),
                    )
                    evaluator = PrimaryPoseEvaluator(problem, chart, device=device, stream=stream)
                    prepared.append(
                        PreparedView(identifier, evaluator, int(np.count_nonzero(mask)))
                    )
            finally:
                # Drain uploads and setup work before their owners can be released.
                wp.synchronize_stream(stream)
    return tuple(prepared)


def evaluate_reference(case: Any, configuration: dict[str, Any], recovered: RigidTransform) -> Any:
    """Read the optional reference only after the accepted pose has been fixed."""
    if "evaluation" not in configuration:
        return {"status": "no independent geometric reference supplied"}
    evaluation = _mapping(configuration["evaluation"], "registration.evaluation")
    for key in ("source", "uncertainty"):
        if not isinstance(evaluation.get(key), str) or not evaluation[key].strip():
            raise ContractError(f"evaluation.{key} must describe the reference and its limits")
    reference = RigidTransform(
        **_immutable_geometry(_mapping(_required(evaluation, "reference_pose"), "reference_pose"))
    )
    landmarks = case.array(_required(evaluation, "landmarks"), dtype="float64", units="mm")
    np: Any = importlib.import_module("numpy")
    if landmarks.ndim != 2 or landmarks.shape[1] != 3 or landmarks.shape[0] < 1:
        raise ContractError("evaluation landmarks must have shape (N, 3), with N >= 1")
    if not bool(np.all(np.isfinite(landmarks))):
        raise ContractError(
            "evaluation landmarks must contain finite object-frame coordinates in mm"
        )
    points = [cast(Vector3, tuple(float(value) for value in point)) for point in landmarks]
    return {
        "status": "evaluated against supplied reference",
        "source": evaluation["source"],
        "uncertainty": evaluation["uncertainty"],
        "reference_pose": asdict(reference),
        "errors": asdict(pose_error(recovered, reference, points)),
    }


def main() -> None:
    run_example(
        example_parser("Register a known attenuation volume to supplied primary-count views."),
        _run,
    )


def _run(args: argparse.Namespace) -> None:
    case = load_case(args.case)
    configuration = _mapping(case.config.get("registration"), "registration")
    chart_values = dict(_mapping(_required(configuration, "chart"), "registration.chart"))
    anchor = RigidTransform(
        **_immutable_geometry(_mapping(_required(chart_values, "anchor"), "chart.anchor"))
    )
    chart_values["anchor"] = anchor
    _required(chart_values, "scales")
    chart = PoseChart(**chart_values)
    policy = RecoveryPolicy(**_mapping(_required(configuration, "policy"), "registration.policy"))
    output = prepare_output(
        args.output,
        configuration=case.config,
        metadata={
            "application": "registration",
            "device_requested": args.device,
            "measurement_model": "independent primary counts with scalar open beam per view",
            "objective": "sum of masked Poisson half-deviances",
            "evaluation_reference_used_for_fit": False,
        },
    )
    views = prepare_views(case, configuration, chart, args.device)
    objective = SharedPoseObjective(views, chart)
    write_metadata(output, **describe_device(views[0].evaluator.context))
    initial: Vector = (0.0,) * 6
    result = recover_parameters(objective, initial, policy=policy)
    recovered = chart.pose(result.parameters)
    write_json(
        output / "optimisation.json",
        {
            "pose_object_to_world": asdict(recovered),
            "chart": asdict(chart),
            "policy": asdict(policy),
            "precision": views[0].evaluator.problem.precision,
            "integration": views[0].evaluator.problem.integration,
            "result": asdict(result),
            "stationary": result.stationary,
        },
    )
    final_losses: dict[str, float] = {}
    for view in views:
        evaluator = view.evaluator
        # A rejected line-search trial may be the last occupant of scratch.
        final_value = evaluator(result.parameters)
        final_losses[view.identifier] = final_value.loss
        prediction = evaluator.prediction.numpy().reshape(evaluator.problem.geometry.shape)
        write_array(output, f"prediction-{view.identifier}.npy", prediction)
    write_json(
        output / "fit-report.json",
        {
            "final_half_deviance_by_view": final_losses,
            "included_pixels_by_view": {v.identifier: v.included_pixels for v in views},
            "solver_evaluations": result.evaluations,
            # A rejected trial can fail before some view evaluators are called.
            "solver_view_evaluation_upper_bound": result.evaluations * len(views),
            "additional_final_export_evaluations": len(views),
            "evaluation": evaluate_reference(case, configuration, recovered),
        },
    )


if __name__ == "__main__":
    main()
