"""CPU checks for derivative-bound saved-profile reporting."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPORT = load("_profile_reporting_test", "tools/fullscan_profile_report.py")
TIMELINE = load("_profile_timeline_test", "benchmarks/fullscan_timeline.py")


@pytest.fixture
def capture(tmp_path):
    metadata = {
        "grid": {"shape": [4, 6, 8]},
        "arrays": {"attenuation": {"shape": [8, 6, 4]}},
        "acquisition": {"views": 360},
        "sampling": {"samples_per_ray": 1024},
    }
    source = {"files": dict.fromkeys(REPORT.PROJECTION_SOURCES, "a" * 64)}
    source["sha256"] = hashlib.sha256(
        json.dumps(source["files"], sort_keys=True).encode()
    ).hexdigest()
    snapshot = tmp_path / "probe.py"
    snapshot.write_text(
        "def load_helper(path, name):\n    return path\n\ndef run(args):\n    return args\n"
    )
    snapshot_hash = REPORT.sha256(snapshot)
    config = {
        "gradients": "volume",
        "requested_gradients": ["volume"],
        "pose_requires_grad": False,
        "precision": "fp32",
        "views": 1,
        "view_indices": [0],
        "nvtx_ranges": REPORT.PROBE_RANGES,
    }
    probe = {
        **copy.deepcopy(config),
        "pose_representation": "fixed identity geometry without a pose tensor",
        "finite_outputs": True,
        "source_unchanged": True,
        "source": copy.deepcopy(source),
        "input_metadata": metadata,
        "helper_sha256": {"fullscan_inputs.py": "input", "common.py": "common"},
        "gradient_shapes": [[8, 6, 4]],
        "gradient_dtypes": ["torch.float32"],
        "target_source_files": dict(source["files"]),
        "sampling": {"rule": "clipped midpoint", "samples_per_ray": 1024},
        "current_torch_stream": 0,
        "cotangent_sha256": "seed",
        "harness_sha256": snapshot_hash,
    }
    capture = {
        "source": source,
        "source_path": "/snapshot",
        "configuration": config,
        "probe_result": probe,
        "input": {"metadata": metadata, "cotangent_sha256": "seed"},
        "probe_snapshot": {"file": "probe.py", "sha256": snapshot_hash},
        "helpers": {
            "probe": {"sha256": snapshot_hash},
            "input_helper": {"sha256": "input"},
            "measurement_common": {"sha256": "common"},
        },
    }
    command = [
        "ncu",
        "python",
        "probe.py",
        "--gradients",
        "volume",
        "--precision",
        "fp32",
        "--views",
        "1",
        "--source",
        "/snapshot",
    ]
    (tmp_path / "probe").mkdir()
    return tmp_path, capture, command


def validate(fixture):
    path, capture, command = fixture
    (path / "probe/probe.json").write_text(json.dumps(capture["probe_result"]))
    return REPORT.validate_probe_binding(path, capture, command)


def test_explicit_volume_binds(capture):
    actual = validate(capture)
    assert actual["requested_gradients"] == ["volume"]
    assert actual["gradient_metadata"] == "explicit"


@pytest.mark.parametrize(
    "key,value",
    [
        ("requested_gradients", ["volume", "pose"]),
        ("pose_requires_grad", True),
        ("gradient_shapes", [[4, 6, 8]]),
        ("gradient_dtypes", ["torch.float64"]),
        ("target_source_files", {}),
        ("current_torch_stream", True),
        ("nvtx_ranges", {}),
        ("precision", "fp64"),
        ("view_indices", [1]),
        ("sampling", {"rule": "clipped midpoint", "samples_per_ray": 256}),
        ("finite_outputs", False),
        ("harness_sha256", "other"),
    ],
)
def test_changed_probe_facts_reject(capture, key, value):
    capture[1]["probe_result"][key] = value
    with pytest.raises(ValueError):
        validate(capture)


def test_missing_explicit_gradient_facts_do_not_revert_to_joint(capture):
    for record in (capture[1]["configuration"], capture[1]["probe_result"]):
        for key in ("gradients", "requested_gradients", "pose_requires_grad"):
            del record[key]
    with pytest.raises(ValueError, match="Incomplete"):
        validate(capture)


def test_source_hash_map_mutation_rejects(capture):
    capture[1]["source"]["sha256"] = "b" * 64
    with pytest.raises(ValueError):
        validate(capture)


def test_probe_copy_mutation_rejects(capture):
    (capture[0] / "probe.py").write_text("changed")
    with pytest.raises(ValueError, match="copy"):
        validate(capture)


@pytest.fixture
def ncu_exports():
    stem = "/captured/fp32-view059-backward"
    command = ["ncu", "-o", stem, "python", "probe.py", "--view-index", "59"]
    capture = {
        "reports": [{"file": "fp32-view059-backward.ncu-repz"}],
        "commands": [
            {
                "stdout": "fp32-view059-backward.csv",
                "command": [
                    "ncu",
                    "--import",
                    stem + ".ncu-repz",
                    "--csv",
                    "--page",
                    "raw",
                ],
            }
        ],
    }
    return capture, command


@pytest.mark.parametrize("stem", ["fp32-view059-backward", "fp32-1-backward", "fp64-1-forward"])
@pytest.mark.parametrize("suffix", [".ncu-rep", ".ncu-repz"])
def test_ncu_raw_export_uses_recorded_stem(ncu_exports, stem, suffix):
    capture, command = ncu_exports
    command[2] = "/captured/" + stem
    capture["reports"][0]["file"] = stem + suffix
    capture["commands"][0]["stdout"] = stem + ".csv"
    capture["commands"][0]["command"][2] = command[2] + suffix
    assert REPORT.ncu_raw_export(capture, command) == stem + ".csv"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_output",
        "duplicate_output",
        "escaping_output",
        "wrong_binary",
        "missing_export",
        "duplicate_export",
        "wrong_import",
        "duplicate_import",
        "source_page",
        "instances_export",
        "not_csv",
    ],
)
def test_ncu_raw_export_rejects_mismatched_records(ncu_exports, mutation):
    capture, command = ncu_exports
    row = capture["commands"][0]
    exported = row["command"]
    if mutation == "missing_output":
        command.remove("-o")
    elif mutation == "duplicate_output":
        command.extend(["-o", command[2]])
    elif mutation == "escaping_output":
        command[2] = "/captured/../other/fp32-view059-backward"
    elif mutation == "wrong_binary":
        capture["reports"][0]["file"] = "fp32-1-backward.ncu-repz"
    elif mutation == "missing_export":
        row["stdout"] = "fp32-1-backward.csv"
    elif mutation == "duplicate_export":
        capture["commands"].append(copy.deepcopy(row))
    elif mutation == "wrong_import":
        exported[2] = "/another/fp32-view059-backward.ncu-repz"
    elif mutation == "duplicate_import":
        exported.extend(["--import", exported[2]])
    elif mutation == "source_page":
        exported[-1] = "source"
    elif mutation == "instances_export":
        exported.extend(["--print-metric-instances", "details"])
    elif mutation == "not_csv":
        exported.remove("--csv")
    with pytest.raises(ValueError):
        REPORT.ncu_raw_export(capture, command)


def test_angle_export_content_still_requires_recorded_hash(tmp_path, ncu_exports):
    capture, command = ncu_exports
    name = REPORT.ncu_raw_export(capture, command)
    raw = tmp_path / name
    raw.write_text("original counter export\n")
    capture.update(
        schema_version=2,
        exports=[{"file": name, "sha256": REPORT.sha256(raw), "bytes": raw.stat().st_size}],
        binary_report_checks={
            "passed": True,
            "files": capture["reports"],
            "checked_at_utc": "2026-10-20T00:00:00+00:00",
        },
    )
    capture["reports"][0].update(sha256="a" * 64, bytes=1)
    assert REPORT.check_capture_files(tmp_path, capture, [name])["status"] == "verified_exports"
    raw.write_text("modified counter export\n")
    with pytest.raises(ValueError, match="Captured export changed"):
        REPORT.check_capture_files(tmp_path, capture, [name])


def test_saved_report_mismatch_rejects(capture):
    validate(capture)
    (capture[0] / "probe/probe.json").write_text("{}")
    with pytest.raises(ValueError, match="Saved probe"):
        REPORT.validate_probe_binding(*capture)


def test_command_gradient_mismatch_rejects(capture):
    capture[2][capture[2].index("--gradients") + 1] = "both"
    with pytest.raises(ValueError, match="command"):
        validate(capture)


def test_legacy_joint_is_explicitly_labelled():
    result = REPORT.gradient_binding(
        {"backward": "joint torch.autograd.grad for field and pose"}, {}, []
    )
    assert result["requested_gradients"] == ["volume", "pose"]
    assert result["gradient_metadata"] == "legacy_declared_joint"
    assert REPORT.operation_label({"operation": "backward", **result}) == "joint VJP"


def test_unknown_legacy_derivatives_are_not_guessed():
    result = REPORT.gradient_binding({}, {}, [])
    assert result["requested_gradients"] is None
    assert (
        REPORT.operation_label({"operation": "backward", **result})
        == "backward, gradients unrecorded"
    )


def test_only_derivation_label_is_ignored_in_operation_digest(tmp_path):
    first, second = tmp_path / "first.py", tmp_path / "second.py"
    first.write_text(
        "def load_helper(path, name):\n    return path\n\ndef run(args):\n"
        "    result = {'measured': args, 'derived_from_probe_sha256': 'label'}\n"
        "    return result\n"
    )
    second.write_text(
        "def load_helper(path, name):\n    return path\n\ndef run(args):\n"
        "    result = {'measured': args}\n    return result\n"
    )
    assert REPORT.probe_operation_digest(first) == REPORT.probe_operation_digest(second)
    second.write_text(second.read_text().replace("'measured': args", "'measured': None"))
    assert REPORT.probe_operation_digest(first) != REPORT.probe_operation_digest(second)


def test_different_gradients_do_not_pair():
    operations = [
        {"id": "before", "role": "before", "comparison_signature": {"gradients": ["volume"]}},
        {"id": "after", "role": "final", "comparison_signature": {"gradients": ["volume", "pose"]}},
    ]
    with pytest.raises(ValueError, match="identical-workload"):
        REPORT.compare_captures(operations, [])
    operations[0]["role"] = "diagnostic"
    assert REPORT.compare_captures(operations, []) == []


@pytest.mark.parametrize(
    "mode,targets,pose", [("volume", ["volume"], False), ("both", ["volume", "pose"], True)]
)
def test_timeline_propagates_explicit_derivatives(mode, targets, pose):
    actual = TIMELINE.requested_gradients(
        {"gradients": mode, "requested_gradients": targets, "pose_requires_grad": pose}
    )
    assert actual["requested_gradients"] == targets
    assert actual["pose_requires_grad"] is pose


def test_timeline_keeps_older_joint_default():
    assert TIMELINE.requested_gradients({})["requested_gradients"] == ["volume", "pose"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"gradients": "volume"},
        {"current_torch_stream": 0},
        {
            "gradients": "volume",
            "requested_gradients": ["volume", "pose"],
            "pose_requires_grad": True,
        },
    ],
)
def test_timeline_rejects_ambiguous_new_metadata(metadata):
    with pytest.raises(ValueError):
        TIMELINE.requested_gradients(metadata)


def test_same_precision_volume_and_joint_timelines_do_not_collide():
    base = {
        "input_metadata": {"sampling": {"samples_per_ray": 1024}},
        "cotangent": {"sha256": "seed"},
        "view_indices": [0],
        "views": 1,
        "precision": "fp32",
        "profile_harness_sha256": "probe",
        "probe_workload": "explicit requested derivatives",
        "ranges": [{"range": "probe_backward", "occurrence": 0}],
    }
    before = {
        **base,
        "id": "baseline-volume",
        "role": "before",
        "requested_gradients": ["volume"],
        "pose_requires_grad": False,
    }
    volume = {**before, "id": "final-volume", "role": "final"}
    joint = {
        **base,
        "id": "final-joint",
        "role": "final",
        "requested_gradients": ["volume", "pose"],
        "pose_requires_grad": True,
    }
    results = REPORT.compare_timelines([before, volume, joint])
    assert len(results) == 1
    assert results[0]["after"] == "final-volume"
    assert results[0]["status"] == "matched"
    mismatched = results[0]["candidates"][1]
    assert mismatched["after"] == "final-joint"
    assert set(mismatched["different_bindings"]) == {
        "requested_gradients",
        "pose_requires_grad",
    }


def test_missing_roofline_counters_remain_unavailable(tmp_path):
    result = REPORT.roofline([], [], "fp32", tmp_path)
    assert result["created"] is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "extra,expected",
    [
        ({"view_indices": [0]}, "1 zero-angle view (saved index 0)"),
        ({"view_indices": [59]}, "1 saved view (index 59)"),
        ({"view_labels": ["ap"]}, "1 AP view"),
    ],
)
def test_roofline_workload_identifies_saved_view_without_guessing_anatomy(extra, expected):
    workload = {
        "grid": {"shape": [256, 256, 256]},
        "detector_shape": [512, 512],
        "samples_per_ray": 1024,
        "views": 1,
        "legacy": False,
        **extra,
    }
    label = REPORT.workload_label(workload)
    assert expected in label
    assert "256x256x256" in label and "512x512" in label and "1024 midpoints" in label


@pytest.mark.parametrize("available", [True, False])
def test_l2_counter_table_preserves_total_sector_intensity_and_missing_values(available):
    profile = {
        "id": "fixture:k0",
        "operation_id": "fixture",
        "role": "final",
        "ncu_kernel_id": "0",
        "kernel": "get_vjp__locals__vjp",
        "plotted_hot_kernel": True,
        "requested_gradients": ["volume"],
        "profiled_kernel_seconds": 0.01,
        "registers_per_thread": 72,
        "occupancy_achieved_percent": 40,
        "fp64_pipe_active_percent": 2,
        "register_spill_instructions": 0,
        "additional_counters": {
            "conversion_thread_instructions": 0,
            "MUFU_thread_instructions": 0,
            "RCP_thread_instructions": None,
            "local_memory_sector_bytes": 0,
        },
        "predicated_thread_opcode_counts": {"FADD": 8192, "DADD": 1024},
        "memory_counter_bytes": {"L1": 512, "L2": 4096 if available else None, "DRAM": 256},
        "roofline_traffic_bytes": {"L1": 128, "L2": 64, "DRAM": 256},
        "precision": {
            precision: {
                "add_mul_fma_flops": flops,
                "arithmetic_intensity_flops_per_byte": {"L2": flops / 64},
                "arithmetic_intensity_counter_flops_per_byte": {
                    "L2": flops / 4096 if available else None
                },
            }
            for precision, flops in (("fp32", 8192), ("fp64", 1024))
        },
        "throughput_percent_of_peak": {},
        "stall_warp_cycles_per_issued_warp_instruction": {},
    }
    before = copy.deepcopy(profile)
    row = REPORT.counters_rows([profile])[0]
    assert row["L2_counter_bytes"] == (4096 if available else None)
    assert row["FP32_FLOPs_per_L2_requested_sector_byte"] == (2 if available else None)
    assert row["FP64_FLOPs_per_L2_requested_sector_byte"] == (0.25 if available else None)
    control = copy.deepcopy(profile)
    control.update(id="control:k0", role="before")
    result = {
        "profiles": [control, profile],
        "operations": [],
        "comparisons": [],
        "timelines": [],
    }
    rendered = REPORT.markdown(result)
    values = (
        "4,096 | 2.000000 | 0.250000" if available else "unavailable | unavailable | unavailable"
    )
    assert f"| fixture:k0 | {values} |" in rendered
    assert "control:k0" not in rendered
    assert len(result["profiles"]) == 2
    assert profile == before
