"""Report measured full-scan forward plus volume-only backward operations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from fullscan_report import compact_measurement, read, require, utc_interval
from update_performance_docs import table


def valid_hash(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def check_source(source, *, allow_empty=False):
    files = source.get("files", {})
    require(
        (files or allow_empty) and all(valid_hash(value) for value in files.values()),
        "missing source file hashes",
    )
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    require(source.get("sha256") == digest, "source digest does not match its file map")


def check_error(value):
    require(
        all(
            math.isfinite(value[key]) and value[key] >= 0
            for key in ("max_abs", "rmse", "relative_l2")
        ),
        "invalid recorded numerical error",
    )


def check_adjoint(value, limit):
    lhs, rhs = value["image_dot_seed_fp64"], value["field_dot_vjp_fp64"]
    require(math.isfinite(lhs) and math.isfinite(rhs), "nonfinite adjoint inner product")
    discrepancy = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
    require(
        value["passed"] is True
        and math.isclose(value["relative_discrepancy"], discrepancy, rel_tol=1e-12, abs_tol=1e-16)
        and discrepancy <= limit,
        "adjoint check failed or disagrees with its recorded inner products",
    )


def load_dpt(directory, samples):
    raw, digest = read(directory / "results.json")
    outputs, output_digest = read(directory / "outputs.json")
    require(
        raw.get("status") == "complete"
        and raw.get("experiment") == "resident_volume_only_iteration"
        and raw.get("source_unchanged") is True
        and raw.get("source") == raw.get("source_after")
        and raw.get("harness") == raw.get("harness_after")
        and raw.get("precision") == "fp32"
        and raw.get("samples_per_ray") == samples
        and raw.get("sampling", {}).get("samples_per_ray") == samples
        and not raw.get("error")
        and not raw.get("failure"),
        "DPT run must be a completed unchanged FP32 volume-only iteration",
    )
    check_source(raw["source"])
    require(bool(raw.get("environment", {}).get("gpu")), "DPT GPU model is missing")
    require(
        raw["harness"] and all(valid_hash(v) for v in raw["harness"].values()),
        "invalid helper hashes",
    )
    utc_interval(raw["started_at"], raw["completed_at"])
    inputs, contract = raw["input"], raw["contract"]
    settings = contract["settings"]
    require(
        valid_hash(contract["sha256"])
        and inputs["cotangent"]["contract_sha256"] == contract["sha256"]
        and inputs["metadata_sha256"] == settings["input_metadata_sha256"]
        and raw["harness"]["fullscan_inputs.py"] == settings["input_helper_sha256"]
        and inputs["metadata"]["arrays"]["attenuation"]["sha256"]
        == settings["field"]["file_sha256"]
        and inputs["selected_view_indices"] == list(range(360))
        and inputs["selected_geometry"] == inputs["metadata"]["geometry"]
        and len(inputs["selected_geometry"]) == 360
        and inputs["cotangent_subset_renormalised"] is False,
        "DPT input, complete acquisition or contract binding differs",
    )
    require(
        outputs["source_sha256"] == raw["source"]["sha256"]
        and outputs["input_metadata_sha256"] == inputs["metadata_sha256"]
        and outputs["cotangent_sha256"] == inputs["cotangent"]["sha256"]
        and outputs["precision"] == "fp32"
        and outputs["views"] == 360
        and outputs["samples_per_ray"] == samples
        and outputs["requested_gradients"] == ["volume"]
        and set(outputs["arrays"]) == {"images", "volume_vjp"},
        "DPT saved output manifest differs",
    )
    for name, shape in (("images", [360, 512, 512]), ("volume_vjp", [256, 256, 256])):
        record = outputs["arrays"][name]
        require(
            record["shape"] == shape
            and record["dtype"] == "float32"
            and record["file"] == name + ".npy"
            and valid_hash(record["sha256"])
            and valid_hash(record["raw_sha256"]),
            "invalid saved output identity",
        )
    require(len(raw["measurements"]) == 1, "volume-only run must contain one measured operation")
    row = raw["measurements"][0]
    config = row["configuration"]
    expected = {
        "precision": "fp32",
        "field_dtype": "float32",
        "gradient_storage_dtype": "float32",
        "requested_gradients": ["volume"],
        "samples_per_ray": samples,
        "views": 360,
        "detector_shape": [512, 512],
        "input_residency": "CUDA",
        "gradients_accumulate": False,
        "graph_reused": False,
        "pose": "fixed identity, no pose tensor or pose gradient",
        "operation": "fresh Projector forward plus volume-only torch.autograd.grad",
    }
    require(
        row["label"] == "volume_only_iteration"
        and all(config.get(key) == value for key, value in expected.items())
        and row["repeats_per_round"] == 10,
        "DPT operation, gradient request or sampling differs",
    )
    checks = raw["checks"]
    for phase in ("before_timing", "after_timing"):
        require(checks[phase]["passed"] is True, "DPT numerical gate failed")
        check_adjoint(
            checks[phase]["volume_adjoint"],
            settings["quality"]["volume_adjoint"]["relative_discrepancy_max"],
        )
        require(
            all(
                v.get("passed") is not False for v in checks[phase].values() if isinstance(v, dict)
            ),
            "nested check failed",
        )
    require(checks["no_leaf_gradient_accumulation"]["passed"] is True, "leaf gradients accumulated")
    require(
        checks["after_timing"]["images_vs_prior_operation"]["bitwise_equal"] is True
        and checks["after_timing"]["volume_vjp_vs_prior_operation"]["relative_l2"]
        <= settings["quality"]["field_vjp_vs_baseline"]["relative_l2_max"],
        "outputs changed outside the fixed tolerance during timing",
    )
    require(
        raw["objective"]["value"]
        == checks["before_timing"]["volume_adjoint"]["image_dot_seed_fp64"],
        "objective differs",
    )
    result = {
        key: raw[key]
        for key in (
            "source",
            "harness",
            "environment",
            "started_at",
            "completed_at",
            "contract",
            "input",
            "checks",
            "objective",
            "references",
            "sampling",
            "preparation_seconds",
        )
    }
    result.update(
        raw_result_sha256=digest,
        outputs_sha256=output_digest,
        outputs=outputs,
        measurement=compact_measurement(row),
    )
    result["sampling_difference_vs_1024"] = raw.get("sampling_difference_vs_1024")
    return result


def bind_reference(reference, run, samples):
    require(
        reference["results_sha256"] == run["raw_result_sha256"]
        and reference["outputs_sha256"] == run["outputs_sha256"]
        and reference["source"] == run["source"]
        and reference["completed_at"] == run["completed_at"]
        and reference["precision"] == "fp32"
        and reference["samples_per_ray"] == samples
        and reference["arrays"] == run["outputs"]["arrays"],
        "reference does not bind the supplied matching run",
    )


def check_pair(baseline, candidate, samples):
    for key in ("input", "contract", "harness"):
        require(baseline[key] == candidate[key], "baseline/candidate differs: " + key)
    require(
        baseline["measurement"]["configuration"] == candidate["measurement"]["configuration"],
        "timing boundaries differ",
    )
    bind_reference(candidate["references"]["same_sample"], baseline, samples)
    require(
        baseline["outputs"]["arrays"]["images"]["raw_sha256"]
        == candidate["outputs"]["arrays"]["images"]["raw_sha256"],
        "same-sample forward image changed",
    )
    for phase in ("before_timing", "after_timing"):
        checks = candidate["checks"][phase]
        require(checks["images_vs_reference"]["bitwise_equal"] is True, "paired image is not exact")
        error = checks["volume_vjp_vs_reference"]
        check_error(error)
        require(
            error["passed"] is True
            and error["relative_l2"]
            <= baseline["contract"]["settings"]["quality"]["field_vjp_vs_baseline"][
                "relative_l2_max"
            ],
            "paired gradient gate failed",
        )


def load_competitor(directory, inputs):
    raw, digest = read(directory / "results.json")
    require(
        raw.get("measurement_finished_at") and not raw.get("failures") and not raw.get("error"),
        "competitor run incomplete",
    )
    utc_interval(raw["measurement_started_at"], raw["measurement_finished_at"])
    check_source(raw["source"], allow_empty=True)
    require(bool(raw.get("environment", {}).get("gpu")), "competitor GPU model is missing")
    require(
        raw["harness_sources"]
        and all(valid_hash(value["sha256"]) for value in raw["harness_sources"].values()),
        "competitor helper identity is missing",
    )
    require(
        raw["input"] == inputs["metadata"]
        and raw["input_metadata"]["sha256"] == inputs["metadata_sha256"]
        and raw["cotangent"]["sha256"] == inputs["cotangent"]["sha256"],
        "competitor volume, physical rays or seed differs",
    )
    backends = {}
    for name, backend in raw["backends"].items():
        require(
            backend["sources"] and all(valid_hash(v) for v in backend["sources"].values()),
            "competitor package source hashes are missing",
        )
        if backend["package"] == "TIGRE":
            extensions = backend.get("native_extensions", {})
            require(
                {"_Ax", "_Atb"}.issubset(extensions)
                and all(valid_hash(v["sha256"]) for v in extensions.values()),
                "TIGRE installed extension hashes are missing",
            )
        require(
            raw["checks"][name + "/geometry"]["passed"] is True, "competitor geometry check failed"
        )
        check_error(backend["image_error_vs_dpt_fp64"])
        if backend.get("adjoint") is not None:
            adjoint = backend["adjoint"]
            lhs, rhs = adjoint["forward_inner_product"], adjoint["backprojection_inner_product"]
            expected = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
            require(
                math.isclose(expected, adjoint["relative_defect"], rel_tol=1e-12),
                "competitor adjoint defect differs",
            )
            check_error(backend["backprojection_error_vs_dpt_fp64_vjp"])
        backends[name] = {
            key: backend.get(key)
            for key in (
                "package",
                "version",
                "sources",
                "native_extensions",
                "algorithm",
                "sampling",
                "resident_boundary",
                "host_boundary",
                "torch_boundary",
                "backprojection",
                "resident_unsupported_reason",
                "image_error_vs_dpt_fp64",
                "adjoint",
                "backprojection_error_vs_dpt_fp64_vjp",
            )
        }
        backends[name]["source_map_sha256"] = hashlib.sha256(
            json.dumps(backend["sources"], sort_keys=True).encode()
        ).hexdigest()
    rows = []
    for row in raw["measurements"]:
        config = row["configuration"]
        backend = backends[config["backend"]]
        route = row["label"].split("/")
        require(
            len(route) == 3
            and route[0] == config["backend"]
            and route[1] in ("resident", "torch", "host")
            and route[2] == config["operation"]
            and backend.get(route[1] + "_boundary") is not None
            and config["boundary"] == backend[route[1] + "_boundary"],
            "competitor operation boundary differs from its recorded route",
        )
        require(
            config["views"] == 360
            and config["detector_shape"] == [512, 512]
            and config["volume_shape_zyx"] == [256, 256, 256]
            and config["field_dtype"] == "float32"
            and config["cotangent"]["sha256"] == inputs["cotangent"]["sha256"]
            and config["sampling"] == backend["sampling"]
            and config["algorithm"] == backend["algorithm"],
            "competitor measured configuration differs",
        )
        require(
            config["operation"] in ("forward", "backprojection", "iteration")
            and config["requested_gradients"]
            == ([] if config["operation"] == "forward" else ["volume"]),
            "competitor operation requests different gradients",
        )
        rows.append(compact_measurement(row))
    return {
        "raw_result_sha256": digest,
        "source": raw["source"],
        "dpt_source_identity_available": bool(raw["source"]["files"]),
        "harness_sources": raw["harness_sources"],
        "environment": raw["environment"],
        "measurement_started_at": raw["measurement_started_at"],
        "measurement_finished_at": raw["measurement_finished_at"],
        "reference_image": raw["reference_image"],
        "reference_gradient": raw["reference_gradient"],
        "reference_gradient_storage_dtype": raw["reference_gradient_storage_dtype"],
        "backends": backends,
        "measurements": rows,
    }


def collect(args):
    coarse = [getattr(args, f"{role}_256", None) for role in ("baseline", "candidate")]
    require(all(coarse) or not any(coarse), "Supply both N256 runs or neither")
    counts = (1024, 256) if all(coarse) else (1024,)
    runs = {
        role: {str(n): load_dpt(getattr(args, f"{role}_{n}"), n) for n in counts}
        for role in ("baseline", "candidate")
    }
    common = runs["baseline"]["1024"]["input"]
    gpu = runs["baseline"]["1024"]["environment"]["gpu"]
    for n in counts:
        check_pair(runs["baseline"][str(n)], runs["candidate"][str(n)], n)
    for role, modes in runs.items():
        require(
            all(run["source"] == modes["1024"]["source"] for run in modes.values()),
            role + " sampling runs use different sources",
        )
        for run in modes.values():
            require(run["input"] == common, "DPT sampling runs use different physical inputs")
            require(run["environment"]["gpu"] == gpu, "DPT runs use different GPU models")
            for key in ("contract", "harness"):
                require(
                    run[key] == runs["baseline"]["1024"][key],
                    "DPT sampling runs differ: " + key,
                )
        coarse = modes.get("256")
        difference = coarse["sampling_difference_vs_1024"] if coarse else None
        if difference is not None:
            reference = coarse["references"]["sampling_difference"]
            matches = [
                name
                for name, values in runs.items()
                if values["1024"]["raw_result_sha256"] == reference["results_sha256"]
            ]
            require(len(matches) == 1, "sampling reference must identify one supplied N1024 run")
            coarse["sampling_reference_role"] = matches[0]
            bind_reference(reference, runs[matches[0]]["1024"], 1024)
            require(
                difference["samples_per_ray"] == 256
                and difference["reference_samples_per_ray"] == 1024
                and "passed" not in difference,
                "sampling difference is not a same-sample gate",
            )
            for key in ("image", "volume_vjp"):
                check_error(difference[key])
        for run in modes.values():
            del run["input"]
    competitors = [load_competitor(directory, common) for directory in args.competitor]
    require(
        all(entry["environment"]["gpu"] == gpu for entry in competitors),
        "competitor and DPT GPU models differ",
    )
    if competitors:
        for item in competitors[1:]:
            for key in ("reference_image", "reference_gradient"):
                require(
                    item[key]["sha256"] == competitors[0][key]["sha256"],
                    "competitors use different numerical references",
                )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ("scope"): (
            "Full chest resampled to 256 cubed, 360 views at 512 squared, volume gradients only"
        ),
        "input": common,
        "dpt": runs,
        "competitors": competitors,
        ("array_validation"): (
            "Recorded file and raw-value identities are bound to the completed timer "
            "manifests."
        ),
    }


def text(value):
    return str(value).replace(";", ".").replace("|", "\\|").replace("\n", " ")


def number(value):
    return "unavailable" if value is None else f"{value:.6g}"


def peak(row):
    values = [r["nvml"].get("sampled_peak_process_memory_used_bytes") for r in row["rounds"]]
    return (
        max(v for v in values if v is not None) / 2**30
        if any(v is not None for v in values)
        else None
    )


def markdown(data):
    counts = [n for n in (1024, 256) if str(n) in data["dpt"]["candidate"]]
    lines = [
        "# Full-scan volume-only iterations\n",
        f"GPU: {text(data['dpt']['candidate']['1024']['environment']['gpu'])}. "
        "Complete chest resampled to 256³, with 360 views at 512².\n",
        "Each completed iteration requests volume gradients only. DPT uses a fresh "
        "forward graph and the exact realised sampling transpose. Inputs stay on CUDA.\n",
    ]
    rows, errors, dates, sampling = [], [], [], []
    for n in counts:
        run = data["dpt"]["candidate"][str(n)]
        row = run["measurement"]
        rows.append(
            [
                "DPT",
                f"FP32 midpoint, {n}",
                "CUDA resident",
                number(row["wall_median_ms"]),
                number(row["wall_p95_ms"]),
                number(row["device_median_ms"]),
                number(row["device_p95_ms"]),
                number(peak(row)),
            ]
        )
        check = run["checks"]["after_timing"]
        errors.append(
            [
                f"DPT N{n}",
                number(check["volume_adjoint"]["relative_discrepancy"]),
                number(check["volume_vjp_vs_reference"].get("relative_l2")),
            ]
        )
        dates.append([f"DPT N{n}", run["completed_at"], run["source"]["sha256"]])
        difference = run.get("sampling_difference_vs_1024")
        if difference:
            sampling.append(
                [
                    f"DPT N{n}",
                    number(difference["image"]["relative_l2"]),
                    number(difference["volume_vjp"]["relative_l2"]),
                ]
            )
    alternatives, components = [], []
    for entry in data["competitors"]:
        for row in entry["measurements"]:
            config = row["configuration"]
            if config["operation"] == "iteration":
                rows.append(
                    [
                        config["backend"],
                        text(config["algorithm"]),
                        text(config["boundary"]),
                        number(row["wall_median_ms"]),
                        number(row["wall_p95_ms"]),
                        number(row["device_median_ms"]),
                        number(row["device_p95_ms"]),
                        number(peak(row)),
                    ]
                )
            else:
                components.append(
                    [
                        row["label"],
                        text(config["boundary"]),
                        number(row["wall_median_ms"]),
                        number(row["wall_p95_ms"]),
                    ]
                )
        for name, backend in entry["backends"].items():
            alternatives.append(
                [
                    name,
                    number(backend["image_error_vs_dpt_fp64"]["relative_l2"]),
                    number(
                        (backend["backprojection_error_vs_dpt_fp64_vjp"] or {}).get("relative_l2")
                    ),
                    number((backend["adjoint"] or {}).get("relative_defect")),
                ]
            )
        dates.append(
            [
                ", ".join(entry["backends"]),
                entry["measurement_finished_at"],
                ", ".join(
                    name + ": " + backend["source_map_sha256"]
                    for name, backend in entry["backends"].items()
                ),
            ]
        )
    lines += [
        table(
            [
                "Implementation",
                "Algorithm / sampling",
                "Boundary",
                "Wall median ms",
                "Wall p95 ms",
                "Event median ms",
                "Event p95 ms",
                "Sampled process GiB",
            ],
            rows,
        ),
        "Wall timings cover completed calls. Device events use the owning CUDA stream "
        "where recorded. NVML memory is the largest sampled process allocation.\n",
        "## Numerical checks\n",
        table(
            [
                "Run",
                "Adjoint relative discrepancy",
                "Volume VJP relative L2 vs same-sample reference",
            ],
            errors,
        ),
    ]
    if sampling:
        lines += [
            "## Sampling\n",
            "Sampling differences use the saved FP32 N1024 image and gradient.\n",
            table(
                [
                    "DPT run",
                    "Image relative L2 vs FP32 N1024",
                    "Volume VJP relative L2 vs FP32 N1024",
                ],
                sampling,
            ),
        ]
    if alternatives:
        lines += [
            "## Alternative algorithms\n",
            table(
                [
                    "Tested adapter",
                    "Image relative L2 vs DPT FP64 N1024",
                    "BP relative L2 vs DPT FP64 N1024 VJP",
                    "Measured adjoint defect",
                ],
                alternatives,
            ),
            "Image and backprojection differences use the recorded DPT FP64 "
            "1024-midpoint calculation. The adjoint defect tests each backend's own "
            "forward/backprojection pair on the saved field and cotangent.\n",
            "ASTRA uses CUDA DLPack resident buffers in its Torch adapter. The TIGRE "
            "Torch adapter includes host copies and explicit matched Atb.\n",
        ]
    if components:
        lines += [
            "## Component timing boundaries\n",
            table(
                ["Measured component", "Completed boundary", "Wall median ms", "Wall p95 ms"],
                components,
            ),
        ]
    lines += [
        "## Measurement identity\n",
        table(["Run", "Completed UTC", "Package source-map SHA256"], dates),
        "Raw samples include round dates and device context. Competitor identities "
        "use the recorded backend source files and versions.\n",
    ]
    return "\n\n".join(block.strip() for block in lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("baseline", "candidate"):
        for samples in (1024, 256):
            parser.add_argument(f"--{role}-{samples}", type=Path, required=samples == 1024)
    parser.add_argument("--competitor", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args)
    document = markdown(data)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.output / "summary.md").write_text(document)
    print(args.output)


if __name__ == "__main__":
    main()
