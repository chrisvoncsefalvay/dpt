"""CPU metadata fixtures for reporting, never measured performance evidence."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "fullscan_volume_report", TOOLS / "fullscan_volume_report.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(samples=1024, role="baseline"):
    files = {"python/dpt/fixture.py": sha(role)}
    source = {"files": files, "sha256": sha(json.dumps(files, sort_keys=True))}
    contract = {
        "sha256": sha("contract"),
        "settings": {
            "input_metadata_sha256": sha("metadata"),
            "input_helper_sha256": sha("input helper"),
            "field": {"file_sha256": sha("attenuation")},
            "quality": {
                "volume_adjoint": {"relative_discrepancy_max": 2e-5},
                "field_vjp_vs_baseline": {"relative_l2_max": 2e-4},
            },
        },
    }
    inputs = {
        "metadata_sha256": sha("metadata"),
        "metadata": {
            "geometry": [{}] * 360,
            "arrays": {"attenuation": {"sha256": sha("attenuation")}},
        },
        "selected_view_indices": list(range(360)),
        "selected_geometry": [{}] * 360,
        "cotangent": {
            "sha256": sha("seed"),
            "raw_sha256": sha("seed values"),
            "contract_sha256": sha("contract"),
        },
        "cotangent_subset_renormalised": False,
    }
    outputs = {
        "source_sha256": source["sha256"],
        "input_metadata_sha256": sha("metadata"),
        "cotangent_sha256": sha("seed"),
        "precision": "fp32",
        "views": 360,
        "samples_per_ray": samples,
        "requested_gradients": ["volume"],
        "arrays": {},
    }
    for name, shape in (("images", [360, 512, 512]), ("volume_vjp", [256, 256, 256])):
        outputs["arrays"][name] = {
            "file": name + ".npy",
            "shape": shape,
            "dtype": "float32",
            "sha256": sha(name + str(samples)),
            "raw_sha256": sha(name + str(samples)),
            "bytes": 128,
        }
    zero_error = {
        "max_abs": 0.0,
        "rmse": 0.0,
        "relative_l2": 0.0,
        "bitwise_equal": True,
        "passed": True,
    }
    phase = {
        "passed": True,
        "volume_adjoint": {
            "image_dot_seed_fp64": 1.0,
            "field_dot_vjp_fp64": 1.0,
            "relative_discrepancy": 0.0,
            "passed": True,
        },
        "images_vs_reference": copy.deepcopy(zero_error),
        "volume_vjp_vs_reference": copy.deepcopy(zero_error),
        "images_vs_prior_operation": copy.deepcopy(zero_error),
        "volume_vjp_vs_prior_operation": copy.deepcopy(zero_error),
    }
    row = {
        "label": "volume_only_iteration",
        "repeats_per_round": 10,
        "configuration": {
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
            "wall_boundary": "fixture completed operation",
            "event_boundary": "fixture current stream",
        },
        "wall_median_ms": 1.0,
        "wall_p95_ms": 1.0,
        "device_median_ms": 1.0,
        "device_p95_ms": 1.0,
        "rounds": [
            {
                "round": n,
                "measured_at": f"2026-10-19T00:0{n}:00+00:00",
                "finished_at": f"2026-10-19T00:0{n}:01+00:00",
                "wall_ms": [1.0] * 10,
                "device_ms": [1.0] * 10,
                "nvml": {"samples": []},
            }
            for n in range(3)
        ],
    }
    raw = {
        "status": "complete",
        "experiment": "resident_volume_only_iteration",
        "source_unchanged": True,
        "source": source,
        "source_after": copy.deepcopy(source),
        "precision": "fp32",
        "samples_per_ray": samples,
        "harness": {
            "fullscan_inputs.py": sha("input helper"),
            "fullscan_volume_iteration.py": sha("timer"),
        },
        "started_at": "2026-10-19T00:00:00+00:00",
        "completed_at": "2026-10-19T00:03:00+00:00",
        "input": inputs,
        "contract": contract,
        "measurements": [row],
        "checks": {
            "before_timing": copy.deepcopy(phase),
            "after_timing": copy.deepcopy(phase),
            "no_leaf_gradient_accumulation": {"passed": True},
        },
        "objective": {"value": 1.0},
        "environment": {"gpu": "metadata fixture only"},
        "references": {},
        "sampling": {"samples_per_ray": samples},
        "preparation_seconds": {},
    }
    raw["harness_after"] = copy.deepcopy(raw["harness"])
    return raw, outputs


def write_run(directory, raw, outputs):
    directory.mkdir(exist_ok=True)
    for name, value in (("results.json", raw), ("outputs.json", outputs)):
        (directory / name).write_text(json.dumps(value))


def reference(run, samples):
    return {
        "results_sha256": run["raw_result_sha256"],
        "outputs_sha256": run["outputs_sha256"],
        "source": run["source"],
        "completed_at": run["completed_at"],
        "precision": "fp32",
        "samples_per_ray": samples,
        "arrays": run["outputs"]["arrays"],
    }


def test_completed_probe_metadata_and_raw_samples_are_retained(tmp_path):
    raw, outputs = fixture()
    write_run(tmp_path, raw, outputs)
    result = report.load_dpt(tmp_path, 1024)
    assert result["outputs"] == outputs
    assert [r["wall_ms"] for r in result["measurement"]["rounds"]] == [[1.0] * 10] * 3
    assert report.peak(result["measurement"]) is None


@pytest.mark.parametrize(
    "change",
    [
        "failed",
        "source",
        "helper",
        "seed",
        "views",
        "samples",
        "pose",
        "adjoint",
        "raw_count",
        "median",
        "manifest",
    ],
)
def test_invalid_evidence_is_rejected(tmp_path, change):
    raw, outputs = fixture()
    if change == "failed":
        raw["status"] = "failed"
    elif change == "source":
        raw["source"]["files"] = {}
    elif change == "helper":
        raw["harness_after"]["fullscan_inputs.py"] = sha("changed")
    elif change == "seed":
        outputs["cotangent_sha256"] = sha("different")
    elif change == "views":
        raw["input"]["selected_view_indices"].pop()
    elif change == "samples":
        outputs["samples_per_ray"] = 256
    elif change == "pose":
        raw["measurements"][0]["configuration"]["requested_gradients"] = ["volume", "pose"]
    elif change == "adjoint":
        raw["checks"]["before_timing"]["volume_adjoint"]["field_dot_vjp_fp64"] = 0.9
    elif change == "raw_count":
        raw["measurements"][0]["rounds"][0]["wall_ms"].pop()
    elif change == "median":
        raw["measurements"][0]["wall_median_ms"] = 0.5
    elif change == "manifest":
        outputs["arrays"]["images"]["raw_sha256"] = "missing"
    write_run(tmp_path, raw, outputs)
    with pytest.raises(ValueError):
        report.load_dpt(tmp_path, 1024)


def test_paired_reference_and_sampling_are_bound_independently(tmp_path):
    args = SimpleNamespace(competitor=[])
    loaded = {}
    for role in ("baseline", "candidate"):
        for samples in (1024, 256):
            directory = tmp_path / f"{role}-{samples}"
            raw, outputs = fixture(samples, role)
            if role == "candidate":
                raw["references"]["same_sample"] = reference(loaded["baseline", samples], samples)
            if samples == 256:
                raw["references"]["sampling_difference"] = reference(loaded[role, 1024], 1024)
                raw["sampling_difference_vs_1024"] = {
                    "samples_per_ray": 256,
                    "reference_samples_per_ray": 1024,
                    "image": {"max_abs": 0.1, "rmse": 0.01, "relative_l2": 0.001},
                    "volume_vjp": {"max_abs": 0.2, "rmse": 0.02, "relative_l2": 0.002},
                }
            write_run(directory, raw, outputs)
            loaded[role, samples] = report.load_dpt(directory, samples)
            setattr(args, f"{role}_{samples}", directory)
    data = report.collect(args)
    assert data["dpt"]["candidate"]["256"]["sampling_reference_role"] == "candidate"
    document = report.markdown(data)
    assert "volume gradients only" in document
    assert "FP32 N1024" in document
    assert "DPT reference" not in document
    assert "DPT optimised" not in document
    assert ";" not in document
    tables = [block for block in document.split("\n\n") if block.startswith("|")]
    assert len(tables) == 4
    assert all(line.startswith("|") for block in tables for line in block.splitlines())
    bad = copy.deepcopy(loaded["candidate", 256])
    bad["references"]["same_sample"] = reference(loaded["baseline", 1024], 1024)
    with pytest.raises(ValueError, match="matching run"):
        report.check_pair(loaded["baseline", 256], bad, 256)
    changed = json.loads((args.candidate_256 / "results.json").read_text())
    changed["environment"]["gpu"] = "different device"
    (args.candidate_256 / "results.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="GPU models"):
        report.collect(args)


def test_nullable_device_times_are_not_fabricated():
    row = fixture()[0]["measurements"][0]
    for round_ in row["rounds"]:
        round_["device_ms"] = None
    row["device_median_ms"] = row["device_p95_ms"] = None
    result = report.compact_measurement(row)
    assert result["device_median_ms"] is None
    row["device_median_ms"] = 0.0
    with pytest.raises(ValueError):
        report.compact_measurement(row)


def test_n1024_only_pair_retains_all_same_sample_gates(tmp_path):
    args = SimpleNamespace(competitor=[], baseline_256=None, candidate_256=None)
    loaded = {}
    for role in ("baseline", "candidate"):
        directory = tmp_path / role
        raw, outputs = fixture(1024, role)
        if role == "candidate":
            raw["references"]["same_sample"] = reference(loaded["baseline"], 1024)
        write_run(directory, raw, outputs)
        loaded[role] = report.load_dpt(directory, 1024)
        setattr(args, role + "_1024", directory)
    data = report.collect(args)
    assert all(set(runs) == {"1024"} for runs in data["dpt"].values())
    document = report.markdown(data)
    assert "FP32 midpoint, 1024" in document
    assert "FP32 midpoint, 256" not in document
    assert "## Sampling" not in document
    assert "DPT reference" not in document
    assert ";" not in document
    for baseline_256, candidate_256 in ((tmp_path, None), (None, tmp_path)):
        changed = copy.copy(args)
        changed.baseline_256, changed.candidate_256 = baseline_256, candidate_256
        with pytest.raises(ValueError, match="both N256"):
            report.collect(changed)
    changed = json.loads((args.candidate_1024 / "results.json").read_text())
    changed["references"]["same_sample"]["samples_per_ray"] = 256
    (args.candidate_1024 / "results.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="matching run"):
        report.collect(args)


def competitor_fixture():
    raw, _ = fixture()
    error = {"max_abs": 0.1, "rmse": 0.01, "relative_l2": 0.001}
    backend = {
        "package": "astra-toolbox",
        "version": "fixture",
        "sources": {"fixture.so": sha("backend")},
        "algorithm": "fixture algorithm",
        "torch_boundary": "fixture resident adapter",
        "sampling": {"same_discretisation_as_dpt": False},
        "image_error_vs_dpt_fp64": error,
        "backprojection_error_vs_dpt_fp64_vjp": error,
        "adjoint": {
            "forward_inner_product": 1.0,
            "backprojection_inner_product": 0.9,
            "relative_defect": 0.1,
        },
    }
    row = raw["measurements"][0]
    row["label"] = "astra-fixture/torch/iteration"
    row["configuration"].update(
        backend="astra-fixture",
        algorithm=backend["algorithm"],
        sampling=backend["sampling"],
        operation="iteration",
        boundary="fixture resident adapter",
        cotangent=raw["input"]["cotangent"],
        volume_shape_zyx=[256] * 3,
    )
    value = {
        "source": {"files": {}, "sha256": sha("{}")},
        "harness_sources": {"fixture.py": {"sha256": sha("helper")}},
        "measurement_started_at": raw["started_at"],
        "measurement_finished_at": raw["completed_at"],
        "input": raw["input"]["metadata"],
        "input_metadata": {"sha256": raw["input"]["metadata_sha256"]},
        "cotangent": raw["input"]["cotangent"],
        "checks": {"astra-fixture/geometry": {"passed": True}},
        "backends": {"astra-fixture": backend},
        "measurements": [row],
        "environment": {"gpu": "metadata fixture only"},
        "reference_image": {"sha256": sha("reference image")},
        "reference_gradient": {"sha256": sha("reference gradient")},
        "reference_gradient_storage_dtype": "float32",
    }
    return value, raw["input"]


def test_competitor_uses_backend_identity_with_empty_unrelated_dpt_map(tmp_path):
    raw, inputs = competitor_fixture()
    (tmp_path / "results.json").write_text(json.dumps(raw))
    result = report.load_competitor(tmp_path, inputs)
    assert result["dpt_source_identity_available"] is False
    assert (
        result["backends"]["astra-fixture"]["sources"]
        == raw["backends"]["astra-fixture"]["sources"]
    )


@pytest.mark.parametrize("change", ["seed", "geometry", "backend", "extension", "boundary"])
def test_competitor_identity_and_workload_changes_reject(tmp_path, change):
    raw, inputs = competitor_fixture()
    raw = copy.deepcopy(raw)
    if change == "seed":
        raw["cotangent"]["sha256"] = sha("wrong")
    elif change == "geometry":
        raw["input"]["geometry"][0] = {"source": [0, -601, 0]}
    elif change == "backend":
        raw["backends"]["astra-fixture"]["sources"] = {}
    elif change == "extension":
        raw["backends"]["astra-fixture"]["package"] = "TIGRE"
    else:
        raw["measurements"][0]["configuration"]["boundary"] = "host transfers included"
    (tmp_path / "results.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        report.load_competitor(tmp_path, inputs)
