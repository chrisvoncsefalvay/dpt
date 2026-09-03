"""Recovery comparisons using the production CUDA oracle.

Independent layer integrals supply observations and assess answers, never runtime
proposals. The sampled reference for scattering uses a separate seed pool; it is
not fed back into model construction or candidate acceptance.
"""

# The experiment driver inspects prepared workspaces for execution details.
# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from dpt.contracts import ContractError, integer
from dpt.experiments import prepare_output, write_json, write_metadata
from dpt.statistics import mean_standard_error
from dpt.stochastic_recovery import StochasticPolicy, recover_expected_signal

from .derivatives import TransportParameter
from .experiments import AbsorptionExperiment, _device_metadata, _recovery
from .forward import prepare_transport
from .inverse import prepare_transport_inverse
from .model import MaterialGrid, PlanarDetector, TransportSpec
from .rng import HistoryBatch
from .source import ParallelBeam

_ABSORPTION = 0.3
_SCATTERING = 0.1
_OBSERVATION = 0.08
_INITIAL_AMPLITUDE = 0.08


def _scattering_problem() -> tuple[TransportSpec, ParallelBeam]:
    return (
        TransportSpec(
            MaterialGrid((-0.5, -0.5, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
            PlanarDetector((-2.0, -2.0), (4.0, 4.0), (1, 1), 2.0),
            80.0,
            "declared isotropic scattering cube for recovery validation",
            estimator="continuous-absorption",
        ),
        ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
    )


def scattering_policy(sampling_multiplier: int = 1) -> StochasticPolicy:
    """Scale original-history work without changing the objective or its gates."""
    factor = integer(sampling_multiplier, "sampling_multiplier", minimum=1, maximum=32767)
    return StochasticPolicy(
        proposal="quadratic",
        replicates=8,
        initial_batch=4096 * factor,
        maximum_batch=65536 * factor,
        final_validation_batch=65536 * factor,
        unique_history_budget=16_000_000 * factor,
        numerical_gradient_allowance=1e-12,
    )


def scattering_configuration(sampling_multiplier: int = 1) -> dict[str, Any]:
    policy = scattering_policy(sampling_multiplier)
    spec, source = _scattering_problem()
    return {
        "sampling_multiplier": sampling_multiplier,
        "policy": asdict(policy),
        "workspace_max_histories": policy.maximum_batch,
        "reference_batch": policy.maximum_batch,
        "reference_replicates": 256,
        "physical_spec": asdict(spec),
        "source": asdict(source),
        "material_ids": [0],
        "absorption_mm_inverse": [_ABSORPTION],
        "scattering_mm_inverse": [_SCATTERING],
        "active_parameters": ["log-source-amplitude"],
        "base_density": [1.0],
        "pixel_weights": [1.0],
        "observation": _OBSERVATION,
        "initial_amplitude": _INITIAL_AMPLITUDE,
    }


def validate_scattering_configuration(configuration: dict[str, Any], multiplier: int) -> None:
    """Check that the reference and recovery use the same physical/sampling model."""
    expected = {"seed": 419003, **scattering_configuration(multiplier)}
    if any(
        json.dumps(configuration.get(key), sort_keys=True) != json.dumps(value, sort_keys=True)
        for key, value in expected.items()
    ):
        raise ContractError("reference physical/sampling configuration differs from recovery")


def validate_scattering_reference(reference: dict[str, Any], sampling_multiplier: int) -> None:
    """Require the separately generated reference to use the declared work scale."""
    policy = scattering_policy(sampling_multiplier)
    if (
        reference.get("batch") != policy.maximum_batch
        or len(reference.get("replicates", [])) != 256
        or reference.get("histories_traced") != 256 * policy.maximum_batch
        or reference.get("seed") != 419003
    ):
        raise ContractError("reference does not match the declared independent sampling design")
    mean, standard_error = mean_standard_error(tuple(reference["replicates"]))
    uncertainty = reference["absolute_uncertainty"]
    if (
        not math.isfinite(uncertainty)
        or uncertainty < 0
        or not math.isclose(mean, reference["mean"], rel_tol=1e-12, abs_tol=1e-15)
        or not math.isclose(standard_error, reference["standard_error"], rel_tol=1e-12)
        or not math.isclose(uncertainty, 7 * standard_error, rel_tol=1e-12)
        or mean <= uncertainty
    ):
        raise ContractError("reference statistics or positive mean interval are invalid")


def _scattering_run(
    seed: int,
    output: Path,
    *,
    reference: dict[str, Any] | None,
    sampling_multiplier: int = 1,
) -> None:
    policy = scattering_policy(sampling_multiplier)
    if reference is not None:
        validate_scattering_reference(reference, sampling_multiplier)
    from dpt._runtime import prepare_context

    context = prepare_context()
    wp = context.wp
    # Fixed mathematical cube, not a surrogate tissue or measured acquisition.
    spec, source = _scattering_problem()
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=context.device),
        absorption=wp.array([_ABSORPTION], dtype=wp.float64, device=context.device),
        scattering=wp.array([_SCATTERING], dtype=wp.float64, device=context.device),
        max_histories=policy.maximum_batch,
        device=str(context.device),
        stream=context.stream,
    )
    # Observation is declared directly. The independent reference locates its
    # optimum afterwards; no analytic/reference signal enters the oracle.
    observation = _OBSERVATION
    oracle = prepare_transport_inverse(
        workspace,
        source=source,
        parameters=(TransportParameter("log-source-amplitude"),),
        observation=wp.array([observation], dtype=wp.float64, device=context.device),
        pixel_weights=wp.ones(1, dtype=wp.float64, device=context.device),
        base_density=(1.0,),
        local_model=True,
    )
    if reference is None:
        means: list[float] = []
        for replicate in range(256):
            oracle._chart((0.0,))
            oracle._mean(
                HistoryBatch(seed, replicate * policy.maximum_batch, policy.maximum_batch),
                1.0,
                oracle._arrays["mean_a"],
            )
            workspace.check_status()
            means.append(float(oracle._arrays["mean_a"].numpy()[0]))
        mean, standard_error = mean_standard_error(tuple(means))
        write_metadata(output, **_device_metadata(workspace))
        write_json(
            output / "reference.json",
            {
                "configuration": {"seed": seed, **scattering_configuration(sampling_multiplier)},
                "mean": mean,
                "standard_error": standard_error,
                "absolute_uncertainty": 7 * standard_error,
                "uncertainty_method": (
                    "seven replicate standard errors; heuristic, not a finite-sample bound"
                ),
                "seed": seed,
                "batch": policy.maximum_batch,
                "sampling_multiplier": sampling_multiplier,
                "allocated_device_bytes": oracle.allocated_bytes,
                "replicates": means,
                "histories_traced": oracle.histories_traced,
                "purpose": "independent held-out forward mean; never optimiser input",
            },
        )
        return
    start = perf_counter()
    try:
        result = recover_expected_signal(
            oracle, (math.log(_INITIAL_AMPLITUDE),), seed=seed, policy=policy
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
    elapsed = perf_counter() - start
    amplitude = math.exp(result.parameters[0])
    mu = reference["mean"]
    error = reference["absolute_uncertainty"]
    # Monotone in this positive-signal neighbourhood; evaluate both interval ends
    # and its possible quadratic vertex to enclose the reference gradient.
    means = [mu - error, mu + error]
    vertex = observation / (2 * amplitude)
    if means[0] < vertex < means[1]:
        means.append(vertex)
    gradients = [(amplitude * value - observation) * amplitude * value for value in means]
    write_metadata(output, **_device_metadata(workspace))
    write_json(
        output / "recovery.json",
        {
            "seed": seed,
            "sampling_multiplier": sampling_multiplier,
            "policy": asdict(policy),
            "result": asdict(result),
            "amplitude": amplitude,
            "independent_optimum_interval": [
                observation / (mu + error),
                observation / (mu - error),
            ],
            "reference_gradient_interval": [min(gradients), max(gradients)],
            "reference_stationarity_pass": max(abs(value) for value in gradients)
            <= policy.gradient_tolerance,
            "histories_traced_including_replay": oracle.histories_traced,
            "allocated_device_bytes": oracle.allocated_bytes,
            "scalar_download_bytes": oracle.scalar_download_bytes,
            "parameter_upload_bytes": oracle.parameter_upload_bytes,
            "elapsed_seconds_contended": elapsed,
            "deterministic_sampling": oracle.deterministic_sampling,
        },
    )


def absorption_cases(mode: str) -> tuple[tuple[str, AbsorptionExperiment], ...]:
    """Construct declared experiment cases on the host before any CUDA preparation."""
    if mode not in ("original", "ablation", "family", "reference", "stochastic"):
        raise ContractError("unknown recovery experiment mode")
    cases: list[tuple[str, AbsorptionExperiment]] = []
    if mode == "original":
        cases = [
            (
                "original",
                AbsorptionExperiment(
                    seed=23971,
                    histories=256,
                    recovery_maximum_batch=256,
                    recovery_history_budget=1_000_000,
                    estimator="continuous-absorption",
                    proposal="quadratic",
                    numerical_gradient_allowance=1e-12,
                ),
            )
        ]
    if mode == "ablation":
        for estimator in ("analogue", "continuous-absorption"):
            for proposal in ("linear", "quadratic"):
                cases.append(
                    (
                        f"{estimator}-{proposal}",
                        AbsorptionExperiment(
                            seed=23971,
                            histories=256,
                            estimator=estimator,
                            proposal=proposal,
                            numerical_gradient_allowance=1e-12,
                            final_validation_batch=262144,
                        ),
                    )
                )
    if mode == "family":
        for depth in (0.1, 1.0, 3.0, 6.0):
            for target in (0.5, 1.25, 2.0):
                for initial in (0.6, 1.0, 1.8):
                    # The squared-signal gradient scales as transmission squared.
                    # Keep a comparable relative gate at each declared optical depth.
                    signal_squared = math.exp(-2.0 * depth * target)
                    cases.append(
                        (
                            f"depth-{depth}-target-{target}-initial-{initial}",
                            AbsorptionExperiment(
                                layer_optical_depths=(depth,),
                                target_density=target,
                                initial_density=initial,
                                histories=16,
                                replicates=4,
                                recovery_maximum_batch=16,
                                recovery_history_budget=1_000_000,
                                estimator="continuous-absorption",
                                proposal="quadratic",
                                numerical_gradient_allowance=1e-12 * signal_squared,
                                gradient_tolerance=1e-5 * signal_squared,
                            ),
                        )
                    )
    return tuple(cases)


def main(script: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("original", "ablation", "family", "reference", "stochastic"),
        required=True,
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--seed-base", type=int, default=9031001)
    parser.add_argument(
        "--sampling-multiplier",
        type=int,
        default=1,
        help="Scale all scattering batches, reference work and unique-history budget together.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=32,
        help="Number of consecutive stochastic recovery runs (1-32); final evaluation uses 32.",
    )
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 32:
        parser.error("repetitions must lie between 1 and 32")
    if args.repetitions != 32 and args.mode != "stochastic":
        parser.error("repetitions only applies to stochastic recovery")
    if args.sampling_multiplier != 1 and args.mode not in ("reference", "stochastic"):
        parser.error("sampling-multiplier only applies to scattering reference/recovery")
    try:
        sampling = scattering_configuration(args.sampling_multiplier)
    except ContractError as error:
        parser.error(str(error))
    if not 0 <= args.seed_base <= 2**64 - 1 - (args.repetitions - 1) * 104729:
        parser.error("the complete seed sequence must fit unsigned 64 bits")
    output = args.output.expanduser().resolve()
    cases = absorption_cases(args.mode)
    for name, config in cases:
        case_output = prepare_output(output / name, configuration=asdict(config))
        _recovery(config, case_output)
        record = json.loads((output / name / "recovery.json").read_text())
        print(
            name,
            record["result"]["reason"],
            record["absolute_density_error"],
            record["independent_true_gradient"],
            flush=True,
        )
    if args.mode == "reference":
        reference_output = prepare_output(
            output / "scattering-reference", configuration={"seed": 419003, **sampling}
        )
        _scattering_run(
            419003, reference_output, reference=None, sampling_multiplier=args.sampling_multiplier
        )
    if args.mode == "stochastic":
        if args.reference is None:
            raise ValueError("stochastic evaluation requires an independent reference")
        reference: dict[str, Any] = json.loads(args.reference.read_text())
        validate_scattering_reference(reference, args.sampling_multiplier)
        validate_scattering_configuration(
            reference.get("configuration", {}), args.sampling_multiplier
        )
        for i in range(args.repetitions):
            seed = args.seed_base + 104729 * i
            evaluation_config: dict[str, Any] = {
                "seed": seed,
                "evaluation_index": i,
                "repetitions": args.repetitions,
                "reference": reference,
                **sampling,
            }
            replicate_output = prepare_output(
                output / f"replicate-{i:02d}", configuration=evaluation_config
            )
            _scattering_run(
                seed,
                replicate_output,
                reference=reference,
                sampling_multiplier=args.sampling_multiplier,
            )
            print(f"completed independent evaluation {i + 1}/{args.repetitions}", flush=True)


if __name__ == "__main__":
    main(__file__)
