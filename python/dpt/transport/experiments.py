"""Reusable orchestration for declared analytic transport acceptance experiments.

The experiments choose mathematical slab inputs and compose production operators.
They do not generate material data, contain alternate photon physics, render
figures or assert acceptance without executing the checks.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from dpt.contracts import finite_scalar, integer
from dpt.experiments import (
    describe_device,
    experiment_parser,
    prepare_output,
    write_json,
    write_metadata,
)
from dpt.stochastic_recovery import StochasticPolicy, recover_expected_signal
from dpt.validation.transport import absorbing_layers

from .derivatives import TransportParameter
from .estimators import (
    ReplicateEstimate,
    history_moments,
    prepare_estimators,
    summarise_replicates,
)
from .forward import TransportWorkspace, prepare_transport, trace_histories
from .inverse import prepare_transport_inverse
from .model import MaterialGrid, PlanarDetector, TransportSpec
from .rng import HistoryBatch
from .source import ParallelBeam, sample_parallel_beam


@dataclass(frozen=True, slots=True)
class AbsorptionExperiment:
    """Pure-absorption layer fixture; no anatomical/material provenance is implied."""

    schema_version: int = 1
    layer_optical_depths: tuple[float, ...] = (0.3, 0.7)
    energy_kev: float = 80.0
    histories: int = 65536
    replicates: int = 8
    seed: int = 73129
    observed_mean: float = 0.2
    finite_difference_steps: tuple[float, ...] = (0.1, 0.03, 0.01)
    target_density: float = 1.25
    recovery_maximum_batch: int = 262144
    recovery_history_budget: int = 100000000
    estimator: Literal["analogue", "continuous-absorption"] = "analogue"
    proposal: Literal["linear", "quadratic"] = "linear"
    initial_density: float = 1.0
    numerical_gradient_allowance: float = 0.0
    gradient_tolerance: float = 1e-5
    final_validation_batch: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported experiment schema")
        if not self.layer_optical_depths or any(
            finite_scalar(value, "layer optical depth", minimum=0.0) == 0
            for value in self.layer_optical_depths
        ):
            raise ValueError("this identifiability fixture requires positive layer optical depths")
        if self.estimator not in ("analogue", "continuous-absorption"):
            raise ValueError("unknown transport estimator")
        if self.proposal not in ("linear", "quadratic"):
            raise ValueError("unknown recovery proposal")
        if finite_scalar(self.initial_density, "initial density", minimum=0.0) == 0:
            raise ValueError("initial density must be positive")
        finite_scalar(self.numerical_gradient_allowance, "numerical allowance", minimum=0.0)
        integer(self.histories, "histories", minimum=2)
        integer(self.replicates, "replicates", minimum=4)
        integer(self.seed, "seed", maximum=2**64 - 1)
        finite_scalar(self.observed_mean, "declared analytic observation")
        if finite_scalar(self.target_density, "target density", minimum=0.0) == 0:
            raise ValueError("target density must be positive")
        if not self.finite_difference_steps or any(
            finite_scalar(value, "finite-difference step", minimum=0.0) == 0
            for value in self.finite_difference_steps
        ):
            raise ValueError("finite-difference steps must be positive")
        integer(self.recovery_maximum_batch, "maximum recovery batch", minimum=self.histories)
        integer(
            self.recovery_history_budget, "recovery history budget", minimum=1, maximum=2**63 - 1
        )
        self.recovery_policy()

    def recovery_policy(self) -> StochasticPolicy:
        """Validate the complete controller configuration before preparing device buffers."""
        return StochasticPolicy(
            replicates=self.replicates,
            initial_batch=self.histories,
            maximum_batch=self.recovery_maximum_batch,
            unique_history_budget=self.recovery_history_budget,
            proposal=self.proposal,
            numerical_gradient_allowance=self.numerical_gradient_allowance,
            gradient_tolerance=self.gradient_tolerance,
            final_validation_batch=self.final_validation_batch,
        )


def _prepare(config: AbsorptionExperiment, capacity: int) -> TransportWorkspace:
    # Coefficient equals optical depth because each declared layer is exactly 1 mm.
    # This is analytic test input, never a tissue-coefficient surrogate.
    from dpt._runtime import prepare_context

    context = prepare_context()
    wp = context.wp
    layers = len(config.layer_optical_depths)
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (layers, 1, 1), layers),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), float(layers + 1)),
        config.energy_kev,
        "declared analytic 1-mm absorption layers; no physical material asset",
        estimator=config.estimator,
    )
    with context.scope():
        ids = wp.array(list(range(layers)), dtype=wp.int32, device=context.device)
        absorption = wp.array(
            list(config.layer_optical_depths), dtype=wp.float64, device=context.device
        )
        scattering = wp.zeros(layers, dtype=wp.float64, device=context.device)
    return prepare_transport(
        spec,
        material_ids=ids,
        absorption=absorption,
        scattering=scattering,
        max_histories=capacity,
        device=context.device,
        stream=context.stream,
    )


def _device_metadata(workspace: TransportWorkspace) -> dict[str, object]:
    return {
        **describe_device(workspace.context),
        "history_schedule": "complete history per CUDA thread",
        "storage": "binary64 coefficients, geometry, scores and moments; int32 grid/pixels",
        "input_provenance": workspace.spec.coefficient_provenance,
    }


def _validation(config: AbsorptionExperiment, output: Path) -> None:
    workspace = _prepare(config, config.histories)
    write_metadata(output, **_device_metadata(workspace))
    context = workspace.context
    wp = context.wp
    source = ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0))
    with context.scope():
        positions = wp.empty(config.histories, dtype=wp.vec3d, device=context.device)
        directions = wp.empty_like(positions)
        weights = wp.empty(config.histories, dtype=wp.float64, device=context.device)
        density = wp.ones(len(config.layer_optical_depths), dtype=wp.float64, device=context.device)
        outputs = {
            f"out_{name}": wp.empty(config.histories, dtype=dtype, device=context.device)
            for name, dtype in (
                ("pixel", wp.int32),
                ("score", wp.float64),
                ("energy", wp.float64),
                ("events", wp.int32),
                ("status", wp.int32),
            )
        }
        mean = wp.empty(1, dtype=wp.float64, device=context.device)
        variance = wp.empty_like(mean)
    moments = prepare_estimators(workspace)
    reference = absorbing_layers(config.layer_optical_depths)
    expected = float(reference.transmission)
    expected_variance = float(reference.variance_per_history) / config.histories
    records: list[dict[str, object]] = []
    for replicate in range(config.replicates):
        batch = HistoryBatch(config.seed, replicate * config.histories, config.histories)
        sample_parallel_beam(
            source,
            batch=batch,
            workspace=workspace,
            out_position=positions,
            out_direction=directions,
            out_weight=weights,
            stream=context.stream,
        )
        trace_histories(
            positions,
            directions,
            weights,
            density,
            batch=batch,
            workspace=workspace,
            stream=context.stream,
            **outputs,
        )
        history_moments(
            outputs["out_pixel"],
            outputs["out_score"],
            outputs["out_status"],
            batch=batch,
            workspace=moments,
            out_mean=mean,
            out_variance_of_mean=variance,
            stream=context.stream,
        )
        actual = float(mean.numpy()[0])
        records.append(
            {
                "batch": asdict(batch),
                "mean": actual,
                "variance_of_mean": float(variance.numpy()[0]),
                "absolute_error": abs(actual - expected),
                "within_declared_7_sigma_band": abs(actual - expected)
                <= 7 * math.sqrt(expected_variance),
            }
        )
    write_json(
        output / "validation.json",
        {
            "reference": expected,
            "independent_reference_variance_of_mean": expected_variance,
            "replicates": records,
            "acceptance": "all independent replicates within seven analytic standard errors",
        },
    )
    if not all(record["within_declared_7_sigma_band"] for record in records):
        raise RuntimeError("transport survival failed its declared analytic statistical check")


def _gradients(config: AbsorptionExperiment, output: Path) -> None:
    workspace = _prepare(config, config.histories)
    write_metadata(output, **_device_metadata(workspace))
    context = workspace.context
    wp = context.wp
    with context.scope():
        observed = wp.array([config.observed_mean], dtype=wp.float64, device=context.device)
        weights = wp.ones(1, dtype=wp.float64, device=context.device)
    oracle = prepare_transport_inverse(
        workspace,
        source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
        parameters=(TransportParameter("log-material-density", 0),),
        observation=observed,
        pixel_weights=weights,
        base_density=(1.0,) * len(config.layer_optical_depths),
    )
    reference = absorbing_layers(config.layer_optical_depths)
    exact_gradient = (float(reference.transmission) - config.observed_mean) * float(
        reference.log_density_derivative[0]
    )
    next_history = 0

    def pairs() -> tuple[HistoryBatch, HistoryBatch]:
        nonlocal next_history
        result = (
            HistoryBatch(config.seed, next_history, config.histories),
            HistoryBatch(config.seed, next_history + config.histories, config.histories),
        )
        next_history += 2 * config.histories
        return result

    estimates: list[ReplicateEstimate] = []
    for _ in range(config.replicates):
        a, b = pairs()
        value = oracle.gradient_replicate((0.0,), a, b)[0]
        estimates.append(ReplicateEstimate(value, (a, b)))
    gradient_summary = summarise_replicates(tuple(estimates))
    differences: list[dict[str, object]] = []
    for step in config.finite_difference_steps:
        values: list[ReplicateEstimate] = []
        for _ in range(config.replicates):
            a, b = pairs()
            value = oracle.change_replicate((-step,), (step,), a, b) / (2 * step)
            values.append(ReplicateEstimate(value, (a, b)))
        differences.append({"step": step, **asdict(summarise_replicates(tuple(values)))})
    write_json(
        output / "gradients.json",
        {
            "target": "squared error of expected photon score; log density of first layer",
            "analytic_gradient": exact_gradient,
            "independent_product_gradient": asdict(gradient_summary),
            "independent_complete_replicates": [asdict(estimate) for estimate in estimates],
            "common_random_finite_difference_sweep": differences,
            "histories_traced_including_replay": oracle.histories_traced,
            "unique_original_histories": next_history,
        },
    )
    # A zero empirical error bar is insufficient to validate a rare-event estimate.
    tolerance = 7 * gradient_summary.standard_error
    if oracle.deterministic_sampling:
        tolerance = config.numerical_gradient_allowance
    if tolerance == 0 or abs(gradient_summary.mean - exact_gradient) > tolerance:
        raise RuntimeError("independent complete loss-gradient check failed or is unresolved")


def _recovery(config: AbsorptionExperiment, output: Path) -> None:
    policy = config.recovery_policy()
    workspace = _prepare(config, config.recovery_maximum_batch)
    write_metadata(output, **_device_metadata(workspace))
    context = workspace.context
    wp = context.wp
    target_depths = (
        config.layer_optical_depths[0] * config.target_density,
        *config.layer_optical_depths[1:],
    )
    expected_observation = float(absorbing_layers(target_depths).transmission)
    with context.scope():
        observed = wp.array([expected_observation], dtype=wp.float64, device=context.device)
        weights = wp.ones(1, dtype=wp.float64, device=context.device)
    oracle = prepare_transport_inverse(
        workspace,
        source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
        parameters=(TransportParameter("log-material-density", 0),),
        observation=observed,
        pixel_weights=weights,
        base_density=(1.0,) * len(config.layer_optical_depths),
        local_model=config.proposal == "quadratic",
    )
    try:
        result = recover_expected_signal(
            oracle, (math.log(config.initial_density),), seed=config.seed, policy=policy
        )
    except Exception as error:
        snapshot = getattr(error, "recovery_diagnostics", None)
        write_json(
            output / "failure-diagnostics.json",
            {
                "error": repr(error),
                "controller": asdict(snapshot) if snapshot is not None else None,
                "histories_submitted_including_replay": oracle.histories_traced,
                "parameter_upload_bytes": oracle.parameter_upload_bytes,
                "scalar_download_bytes": oracle.scalar_download_bytes,
            },
        )
        raise
    density = math.exp(result.parameters[0])
    recovered_reference = absorbing_layers(
        (config.layer_optical_depths[0] * density, *config.layer_optical_depths[1:])
    )
    true_gradient = (float(recovered_reference.transmission) - expected_observation) * float(
        recovered_reference.log_density_derivative[0]
    )
    write_json(
        output / "recovery.json",
        {
            "result": asdict(result),
            "policy": asdict(policy),
            "target": "analytic expected measurement, not a sampled clinical observation",
            "target_density": config.target_density,
            "recovered_density": density,
            "independent_true_gradient": true_gradient,
            "independent_stationarity_pass": abs(true_gradient) <= policy.gradient_tolerance,
            "sampling_classification": "deterministic"
            if oracle.deterministic_sampling
            else "stochastic",
            "allocated_device_bytes": oracle.allocated_bytes,
            "absolute_density_error": abs(density - config.target_density),
            "expected_observation": expected_observation,
            "histories_traced_including_replay": oracle.histories_traced,
            "parameter_upload_bytes": oracle.parameter_upload_bytes,
            "scalar_download_bytes": oracle.scalar_download_bytes,
            "claim": "termination and parameter error recorded; recovery success is not inferred",
        },
    )


def main(mode: Literal["validation", "gradients", "recovery"], script: str) -> None:
    """Command boundary; executing it is an explicit scientific run, never an import effect."""
    parser = experiment_parser(script, __doc__, cuda=False)
    args = parser.parse_args()
    raw: dict[str, Any] = json.loads(args.config.read_text())
    for name in ("layer_optical_depths", "finite_difference_steps"):
        if name in raw:
            raw[name] = tuple(raw[name])
    config = AbsorptionExperiment(**raw)
    output = prepare_output(
        args.output,
        configuration={"mode": mode, **asdict(config)},
        metadata={"execution_scope": "actual CUDA mathematical validation; no visualisations"},
    )
    {"validation": _validation, "gradients": _gradients, "recovery": _recovery}[mode](
        config, output
    )
